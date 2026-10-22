"""Automatic filter for self-generated clips (CPU). Drops static, chaotic, flickering, blurry, too dark/bright clips using
percentile rules, then keeps the sharpest. Writes <data>/metadata.jsonl (kept clips) and <data>/curation.csv.
These are weak proxies: still look at a few kept clips. Also: --report <dir> prints the same metrics for base_*/lora_* videos."""
import argparse, csv, glob, json, os
import numpy as np, imageio.v2 as imageio


def load(path):
    return np.stack([f[..., :3] for f in imageio.get_reader(path)])


def metrics(frames):
    g = frames.astype(np.float32).mean(3) / 255.0                       # T,H,W gray
    lap = -4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1] + g[:, 1:-1, :-2] + g[:, 1:-1, 2:]
    fm = g.mean((1, 2))
    return dict(motion=float(np.abs(np.diff(g, axis=0)).mean()),        # mean frame-to-frame change
                flicker=float(np.abs(np.diff(fm, 2)).mean()),           # abrupt brightness changes
                sharp=float(lap.var(axis=(1, 2)).mean()),               # Laplacian variance
                bright=float(g.mean()))


def curate(a):
    meta = [json.loads(l) for l in open(f"{a.data}/all_metadata.jsonl", encoding="utf-8")]
    rows = []
    for m in meta:
        p = f"{a.data}/{m['file']}"
        if os.path.exists(p): rows.append({**m, **metrics(load(p))})
    pc = lambda k, q: np.percentile([r[k] for r in rows], q)
    lo_m, hi_m, lo_s, hi_f = pc("motion", 20), pc("motion", 90), pc("sharp", 25), pc("flicker", 80)
    for r in rows:
        r["ok"] = (0.05 <= r["bright"] <= 0.85 and lo_m <= r["motion"] <= hi_m and r["sharp"] >= lo_s and r["flicker"] <= hi_f)
    kept = sorted([r for r in rows if r["ok"]], key=lambda r: -r["sharp"])[:a.keep]
    if len(kept) < a.min_keep:  # fill up with the sharpest of the rest
        rest = sorted([r for r in rows if r not in kept], key=lambda r: -r["sharp"])
        kept += rest[:a.min_keep - len(kept)]
    ids = {r["file"] for r in kept}
    open(f"{a.data}/metadata.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(dict(file=r["file"], caption=r["caption"]), ensure_ascii=False) for r in kept))
    with open(f"{a.data}/curation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "kept", "motion", "flicker", "sharp", "bright"])
        for r in rows: w.writerow([r["file"], r["file"] in ids, f"{r['motion']:.4f}", f"{r['flicker']:.4f}", f"{r['sharp']:.5f}", f"{r['bright']:.3f}"])
    print(f"{len(rows)} candidates -> kept {len(kept)} (rules: motion {lo_m:.4f}-{hi_m:.4f}, sharp >= {lo_s:.5f}, flicker <= {hi_f:.4f})")


def report(d):
    print(f"{'file':14s} {'motion':>8s} {'flicker':>8s} {'sharp':>9s} {'bright':>7s}")
    for p in sorted(glob.glob(f"{d}/*.mp4")):
        m = metrics(load(p)); print(f"{os.path.basename(p):14s} {m['motion']:8.4f} {m['flicker']:8.4f} {m['sharp']:9.5f} {m['bright']:7.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/coffee_synth"); p.add_argument("--keep", type=int, default=30); p.add_argument("--min_keep", type=int, default=16)
    p.add_argument("--report", default=None, help="folder of mp4 files to print metrics for")
    a = p.parse_args(); report(a.report) if a.report else curate(a)
