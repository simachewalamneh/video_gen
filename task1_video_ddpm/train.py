"""Train + evaluate a small video DDPM. Saves ckpt, config, logs, loss graph, generated clips, summary.json."""
import argparse, csv, json, os, random, time, copy
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset import get_splits, preprocess
from model import VideoDenoiser
from diffusion import Diffusion


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def save_gif(clips, path, fps=6):  # clips: (N,1,T,H,W) in [-1,1] -> tiled gif
    import imageio.v2 as imageio
    v = ((clips.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()[:, 0]  # N,T,H,W
    n = len(v); cols = min(4, n); rows = (n + cols - 1) // cols
    grid = np.zeros((v.shape[1], rows * v.shape[2], cols * v.shape[3]), np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[:, r * v.shape[2]:(r + 1) * v.shape[2], c * v.shape[3]:(c + 1) * v.shape[3]] = v[i]
    imageio.mimsave(path, list(grid), duration=1000 / fps, loop=0)


@torch.no_grad()
def eval_loss(model, diff, loader, dev):
    model.eval(); g = torch.Generator(dev).manual_seed(123); tot = n = 0
    for x in loader:
        x = x.to(dev)
        t = torch.randint(0, diff.T, (x.size(0),), device=dev, generator=g)
        tot += diff.loss(model, x, t).item() * x.size(0); n += x.size(0)
    return tot / n


def motion(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean().item()


def main(a):
    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    tr, va, te = get_splits(a.n_clips, a.frames, a.size, a.seed, a.video_dir)
    ld = lambda d, sh: DataLoader(d, a.bs, shuffle=sh, drop_last=sh)
    tl, vl, tel = ld(tr, True), ld(va, False), ld(te, False)
    model = VideoDenoiser(1, a.ch, grad_ckpt=a.grad_ckpt).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diff = Diffusion(a.T, a.schedule, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    use_amp = a.amp and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ck, start, best, rows = f"{a.out}/ckpt.pt", 0, 1e9, []
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); start, best = s["epoch"] + 1, s["best"]
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(start, a.epochs):
        model.train(); run = 0
        for x in tl:
            x = x.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                loss = diff.loss(model, x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            run += loss.item()
        trl, val = run / len(tl), eval_loss(ema, diff, vl, dev)
        rows.append((ep, trl, val)); print(f"ep {ep:03d} train {trl:.4f} val(ema) {val:.4f} | {time.time()-t0:.0f}s", flush=True)
        state = dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep, best=min(best, val), config=vars(a))
        torch.save(state, ck)
        if val < best:
            best = val; torch.save(state, f"{a.out}/best.pt")
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan")
    with open(f"{a.out}/train_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss"]); w.writerows(rows)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if rows:
        plt.figure(figsize=(5, 3)); plt.plot([r[1] for r in rows], label="train"); plt.plot([r[2] for r in rows], label="val (EMA)")
        plt.xlabel("epoch"); plt.ylabel("eps-MSE"); plt.title(f"{a.schedule}, lr={a.lr}"); plt.legend(); plt.tight_layout(); plt.savefig(f"{a.out}/loss.png", dpi=120); plt.close()
    # ---- test + generation ----
    test_loss = eval_loss(ema, diff, tel, dev)
    n = a.n_samples; shape = (n, 1, a.frames, a.size, a.size)
    if dev == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    gen = diff.sample(ema, shape, a.sampler, a.ddim_steps, dev, a.seed)
    if dev == "cuda":
        torch.cuda.synchronize()
    samp_time = time.time() - t1
    save_gif(gen, f"{a.out}/generated.gif")
    real = torch.stack([te[i] for i in range(min(n, len(te)))]).to(dev)
    save_gif(real, f"{a.out}/real_test.gif")
    ref = torch.stack([preprocess(c, a.frames, a.size) for c in tr.clips[:300]]).to(dev)
    nn_d = lambda q: torch.cdist(q.flatten(1), ref.flatten(1)).min(1).values.mean().item()  # memorisation check
    summ = dict(schedule=a.schedule, lr=a.lr, T=a.T, sampler=f"{a.sampler}-{a.ddim_steps if a.sampler=='ddim' else a.T}", test_loss=test_loss,
                motion_real=motion(real), motion_gen=motion(gen), nn_dist_gen=nn_d(gen), nn_dist_real=nn_d(real),
                train_s=train_time, sample_s=samp_time, peak_mem_mb=peak, params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                weights_fp32_mb=sum(p.numel() for p in model.parameters()) * 4 / 2**20)
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
    return summ


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cosine"); p.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--T", type=int, default=200); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ema", type=float, default=0.99); p.add_argument("--ch", type=int, default=32)
    p.add_argument("--frames", type=int, default=8); p.add_argument("--size", type=int, default=32)
    p.add_argument("--n_clips", type=int, default=1500); p.add_argument("--video_dir", default=None)
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"]); p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad_ckpt", action="store_true"); p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse())
