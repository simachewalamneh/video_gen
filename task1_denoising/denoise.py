"""Task 1: image denoising pipeline (Colab-free-tier friendly).
preprocess (resize -> tensor -> normalise) -> synthetic noise (Gaussian / salt-and-pepper) -> compact U-Net or autoencoder
-> train on noisy/clean pairs -> test with PSNR + SSIM.
Extras tied to the course material: uncertainty-aware output (per-pixel variance, Gaussian NLL) and
iterative error correction (the network is applied/trained for K refinement passes)."""
import argparse, json, os, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------- data + preprocessing ----------------
def build_splits(a):
    from torchvision import datasets, transforms as T
    tf = T.Compose([T.Resize(a.size), T.CenterCrop(a.size), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])  # -> [-1,1]
    ds = datasets.ImageFolder(a.data_dir, tf) if a.data_dir else datasets.CIFAR10("data", train=True, download=True, transform=tf)
    n = min(a.n_images, len(ds))
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(a.seed))[:n].tolist()
    i, j = int(0.8 * n), int(0.9 * n)
    return [Subset(ds, idx[:i]), Subset(ds, idx[i:j]), Subset(ds, idx[j:])]  # train / val / test


# ---------------- synthetic noise ----------------
def corrupt(x, kind, level=None, gen=None):
    """x in [-1,1]. level: sigma (gaussian, on [0,1] scale) or pixel fraction (saltpepper). None -> random level per image."""
    B, dev = x.size(0), x.device
    rnd = lambda *s: torch.rand(*s, device=dev, generator=gen)
    if kind == "mixed":
        kind = "gaussian" if rnd(1).item() < 0.5 else "saltpepper"
    if kind == "gaussian":
        s = (0.05 + 0.25 * rnd(B)) if level is None else torch.full((B,), float(level), device=dev)
        return (x + 2 * s.view(B, 1, 1, 1) * torch.randn(x.shape, device=dev, generator=gen)).clamp(-1, 1)
    p = (0.02 + 0.18 * rnd(B)) if level is None else torch.full((B,), float(level), device=dev)
    p, m, out = p.view(B, 1, 1, 1), rnd(*x.shape), x.clone()
    out[m < p / 2] = -1.0                      # pepper
    out[(m >= p / 2) & (m < p)] = 1.0          # salt
    return out


# ---------------- model ----------------
class Res(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.n1, s.c1, s.n2, s.c2 = nn.GroupNorm(8, i), nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.Conv2d(o, o, 3, padding=1)
        s.sk = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(s, x):
        h = s.c1(F.silu(s.n1(x)))
        return s.c2(F.silu(s.n2(h))) + s.sk(x)


class DenoiseNet(nn.Module):
    """skip=True -> compact U-Net; skip=False -> plain autoencoder. Predicts a residual + per-pixel log-variance."""
    def __init__(s, c=32, skip=True, unc=True):
        super().__init__()
        s.skip, s.unc = skip, unc
        k = 2 if skip else 1
        s.inp = nn.Conv2d(3, c, 3, padding=1); s.e1 = Res(c, c)
        s.d1 = nn.Conv2d(c, 2 * c, 3, 2, 1); s.e2 = Res(2 * c, 2 * c)
        s.d2 = nn.Conv2d(2 * c, 4 * c, 3, 2, 1); s.mid = Res(4 * c, 4 * c)
        s.u2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2); s.r2 = Res(2 * c * k, 2 * c)
        s.u1 = nn.ConvTranspose2d(2 * c, c, 2, 2); s.r1 = Res(c * k, c)
        s.out = nn.Conv2d(c, 6 if unc else 3, 3, padding=1)

    def forward(s, x):
        h0 = s.e1(s.inp(x)); h1 = s.e2(s.d1(h0)); h = s.mid(s.d2(h1))
        h = s.u2(h); h = s.r2(torch.cat([h, h1], 1) if s.skip else h)
        h = s.u1(h); h = s.r1(torch.cat([h, h0], 1) if s.skip else h)
        o = s.out(h)
        return (x + o[:, :3], o[:, 3:].clamp(-6, 3)) if s.unc else (x + o, None)


# ---------------- metrics / baselines ----------------
def psnr(a, b):  # a,b in [0,1]; per-image PSNR averaged
    return (10 * torch.log10(1.0 / (a - b).pow(2).flatten(1).mean(1).clamp_min(1e-10))).mean().item()


def ssim(a, b):
    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None])[None, None].repeat(3, 1, 1, 1)
    f = lambda z: F.conv2d(z, w, padding=5, groups=3)
    m1, m2 = f(a), f(b); s11, s22, s12 = f(a * a) - m1 ** 2, f(b * b) - m2 ** 2, f(a * b) - m1 * m2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (s11 + s22 + C2))).mean().item()


def classical(x, kind):  # baselines: median 3x3 for salt-and-pepper, 3x3 Gaussian blur for Gaussian noise
    B, C, H, W = x.shape
    if kind == "saltpepper":
        return F.unfold(x, 3, padding=1).view(B, C, 9, H * W).median(2).values.view(B, C, H, W)
    k = torch.tensor([1., 2., 1.], device=x.device); k = (k[:, None] * k[None]); k = (k / k.sum())[None, None].repeat(C, 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=C)


def loss_fn(mu, lv, y, w):
    mse = F.mse_loss(mu.float(), y)
    if lv is None:
        return mse
    lv = lv.float()
    return mse + w * (0.5 * (torch.exp(-lv) * (y - mu.float()) ** 2 + lv)).mean()  # Gaussian NLL (uncertainty-aware)


def unrolled(model, x, steps, detach=False):  # iterative error correction: feed each output back in
    outs, cur = [], x
    for _ in range(steps):
        mu, lv = model(cur); outs.append((mu, lv)); cur = mu.detach() if detach else mu
    return outs


@torch.no_grad()
def evaluate(model, loader, a, dev):
    model.eval(); rows, to01 = [], lambda z: (z.clamp(-1, 1) + 1) / 2
    for kind, levels in (("gaussian", [0.1, 0.2, 0.3]), ("saltpepper", [0.05, 0.1, 0.2])):
        for lv in levels:
            g = torch.Generator(dev).manual_seed(1234); sums, n, ue = {}, 0, []
            for x, _ in loader:
                x = x.to(dev); xn = corrupt(x, kind, lv, g); outs = unrolled(model, xn, a.refine); b, c = x.size(0), to01(x)
                v = {"noisy_psnr": psnr(to01(xn), c), "noisy_ssim": ssim(to01(xn), c), "classical_psnr": psnr(to01(classical(xn, kind)), c)}
                for k, (mu, _) in enumerate(outs):
                    v[f"pass{k+1}_psnr"], v[f"pass{k+1}_ssim"] = psnr(to01(mu), c), ssim(to01(mu), c)
                for key, val in v.items(): sums[key] = sums.get(key, 0) + val * b
                n += b
                if outs[0][1] is not None and len(ue) < 20:  # does predicted uncertainty track real error?
                    sd = torch.exp(0.5 * outs[0][1].float()).mean(1).flatten(); er = (outs[0][0] - x).abs().mean(1).flatten()
                    ue.append(np.corrcoef(sd.cpu().numpy(), er.cpu().numpy())[0, 1])
            rows.append(dict(noise=f"{kind}:{lv}", **{k: v / n for k, v in sums.items()}, unc_err_corr=float(np.nanmean(ue)) if ue else float("nan")))
    return rows


def main(a):
    from torchvision.utils import save_image
    seed_all(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"; os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    tr, va, te = build_splits(a)
    mk = lambda d, sh: DataLoader(d, a.bs, shuffle=sh, drop_last=sh, num_workers=2, pin_memory=True)
    tl, vl, tel = mk(tr, True), mk(va, False), mk(te, False)
    model = DenoiseNet(a.ch, a.arch == "unet", a.unc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    amp = a.amp and dev == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    t0, log = time.time(), []
    for ep in range(a.epochs):
        model.train(); run = 0
        for x, _ in tl:
            x = x.to(dev); xn = corrupt(x, a.noise)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                loss = sum(loss_fn(mu, lv, x, a.nll_w) for mu, lv in unrolled(model, xn, a.refine, detach=True))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); run += loss.item()
        model.eval(); vs = 0; g = torch.Generator(dev).manual_seed(7)
        with torch.no_grad():
            for x, _ in vl:
                x = x.to(dev); mu, _ = model(corrupt(x, a.noise, None, g)); vs += F.mse_loss(mu, x).item()
        log.append((ep, run / len(tl), vs / len(vl))); print(f"ep {ep:02d} train {log[-1][1]:.4f} val_mse {log[-1][2]:.4f} | {time.time()-t0:.0f}s", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, config=vars(a)), f"{a.out}/ckpt.pt")
    train_s, peak = time.time() - t0, (torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else float("nan"))
    rows = evaluate(model, tel, a, dev)
    cols = list(rows[0].keys()); md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows: md += "| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in cols) + " |\n"
    open(f"{a.out}/results.md", "w").write(md); print(md)
    json.dump(dict(rows=rows, train_s=train_s, peak_mem_mb=peak, params_m=sum(p.numel() for p in model.parameters()) / 1e6), open(f"{a.out}/summary.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.plot([l[1] for l in log], label="train loss"); plt.plot([l[2] for l in log], label="val MSE"); plt.legend(); plt.xlabel("epoch"); plt.savefig(f"{a.out}/loss.png", dpi=120); plt.close()
    # qualitative grid: clean | noisy | pass1 | pass2 | uncertainty  (gaussian 0.2 and salt-pepper 0.1)
    x = next(iter(tel))[0][:8].to(dev); g = torch.Generator(dev).manual_seed(5); rowsimg = []
    with torch.no_grad():
        for kind, lv in (("gaussian", 0.2), ("saltpepper", 0.1)):
            xn = corrupt(x, kind, lv, g); outs = unrolled(model, xn, a.refine)
            u = (torch.exp(0.5 * outs[0][1].float()).mean(1, keepdim=True) if outs[0][1] is not None else torch.zeros_like(x[:, :1]))
            u = (u / u.amax((1, 2, 3), keepdim=True).clamp_min(1e-6) * 2 - 1).expand(-1, 3, -1, -1)
            rowsimg += [x, xn] + [o[0] for o in outs] + [u]
    save_image((torch.cat(rowsimg).clamp(-1, 1) + 1) / 2, f"{a.out}/samples.png", nrow=8, padding=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/unet"); p.add_argument("--data_dir", default=None)
    p.add_argument("--n_images", type=int, default=10000); p.add_argument("--size", type=int, default=32)
    p.add_argument("--arch", default="unet", choices=["unet", "ae"]); p.add_argument("--noise", default="mixed", choices=["gaussian", "saltpepper", "mixed"])
    p.add_argument("--unc", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--nll_w", type=float, default=0.1)
    p.add_argument("--refine", type=int, default=2); p.add_argument("--ch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--bs", type=int, default=128); p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
