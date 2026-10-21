"""Turn raw footage (phone videos) into a LoRA training set for Wan.
Each video is resampled to 16 fps, centre-cropped to portrait 9:16, resized, and cut into short windows (default 17 frames).
Needs captions.json: {"video_name.mp4": "what the video shows", ...}. Writes <out>/clips/*.mp4 + <out>/metadata.jsonl.
Tips: film VERTICAL, steady, good light, 10-30 s per video, one subject per video, no text/logos; avoid identifiable faces unless you have permission."""
import argparse, json, os
import numpy as np, imageio.v2 as imageio
from PIL import Image

EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def crop_resize(fr, H, W):
    h, w = fr.shape[:2]; ar = W / H
    ch, cw = (h, int(round(h * ar))) if w / h >= ar else (int(round(w / ar)), w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return np.array(Image.fromarray(fr[y0:y0 + ch, x0:x0 + cw]).resize((W, H), Image.LANCZOS))


def read_16fps(path, H, W, max_seconds):
    r = imageio.get_reader(path); fps = r.get_meta_data().get("fps", 30) or 30
    idx = [int(round(j * fps / 16)) for j in range(int(max_seconds * 16))]; need = set(idx); got = {}
    for i, fr in enumerate(r):
        if i in need: got[i] = crop_resize(fr[..., :3], H, W)
        if i >= idx[-1]: break
    r.close()
    return [got[i] for i in idx if i in got]


def main(a):
    caps = json.load(open(a.captions, encoding="utf-8"))
    os.makedirs(f"{a.out}/clips", exist_ok=True); meta = []
    present = sorted(f for f in os.listdir(a.raw) if f.lower().endswith(EXT))
    for f in present:
        if f not in caps: print(f"[skip] no caption for {f}")
    for name, cap in caps.items():
        path = os.path.join(a.raw, name)
        if not os.path.exists(path): print(f"[warn] missing file {name}"); continue
        frames = read_16fps(path, a.height, a.width, a.max_seconds); stride = int(a.stride * 16) if a.stride else a.frames; k = 0
        for s in range(0, len(frames) - a.frames + 1, stride):
            if k >= a.max_per_video: break
            fn = f"clips/{os.path.splitext(name)[0]}_{k:02d}.mp4"
            wr = imageio.get_writer(f"{a.out}/{fn}", fps=16, codec="libx264", quality=8, macro_block_size=1)
            for fr in frames[s:s + a.frames]: wr.append_data(fr)
            wr.close(); meta.append(dict(file=fn, caption=a.prefix + cap)); k += 1
        print(f"{name}: {k} clips")
    open(f"{a.out}/metadata.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(m, ensure_ascii=False) for m in meta))
    print(f"TOTAL {len(meta)} clips -> {a.out}  (aim for 20-40 diverse clips)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True); p.add_argument("--captions", default="captions.json"); p.add_argument("--out", default="data/coffee")
    p.add_argument("--prefix", default="Ethiopian coffee commercial style: ")
    p.add_argument("--frames", type=int, default=17); p.add_argument("--height", type=int, default=512); p.add_argument("--width", type=int, default=288)
    p.add_argument("--stride", type=float, default=0, help="seconds between windows (0 = back to back)")
    p.add_argument("--max_per_video", type=int, default=6); p.add_argument("--max_seconds", type=float, default=40)
    main(p.parse_args())
