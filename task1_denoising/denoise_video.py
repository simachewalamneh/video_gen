"""Apply the trained Task 1 denoiser to a real video, frame by frame.
Adds synthetic noise to each frame, denoises it, and writes a side-by-side video (clean | noisy | denoised)
plus per-video PSNR and a temporal flicker measure. Works on any mp4, e.g. a Task 2 Wan output:
  python denoise_video.py --video ../task2_lora_amharic_video/outputs/samples/lora_0.mp4 --ckpt runs/unet/ckpt.pt
Flicker = mean |frame_t - frame_(t-1)| on the [0,1] scale. If denoised flicker is above clean flicker, the
frame-independent denoiser added temporal inconsistency (the network has no temporal modelling)."""
import argparse, json, os
import numpy as np, torch, imageio.v2 as imageio
from denoise import DenoiseNet, corrupt, psnr, unrolled, seed_all


def read_frames(path, max_frames):
    r = imageio.get_reader(path); frames = []
    for f in r:                                  # stop early: do not decode the whole video
        frames.append(f[..., :3])
        if len(frames) >= max_frames: break
    r.close()
    return frames


def flicker(v):  # v: (T,3,H,W) in [0,1] -> mean absolute difference between consecutive frames
    return float((v[1:] - v[:-1]).abs().mean()) if len(v) > 1 else float("nan")


def main(a):
    seed_all(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev); c = ck["config"]
    model = DenoiseNet(c["ch"], c["arch"] == "unet", c["unc"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    frames = read_frames(a.video, a.max_frames)
    H, W = frames[0].shape[0] // 4 * 4, frames[0].shape[1] // 4 * 4   # network downsamples by 4
    x = torch.from_numpy(np.stack(frames)[:, :H, :W]).permute(0, 3, 1, 2).float() / 127.5 - 1
    g = torch.Generator(dev).manual_seed(a.seed); to01 = lambda z: (z.clamp(-1, 1) + 1) / 2
    rows, ps_noisy, ps_den, cl01, no01, de01 = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(x), a.bs):
            clean = x[i:i + a.bs].to(dev); noisy = corrupt(clean, a.noise, a.level, g)
            den = unrolled(model, noisy, c["refine"])[-1][0].clamp(-1, 1)
            for k in range(len(clean)):
                ps_noisy.append(psnr(to01(noisy[k:k+1]), to01(clean[k:k+1]))); ps_den.append(psnr(to01(den[k:k+1]), to01(clean[k:k+1])))
            cl01.append(to01(clean).cpu()); no01.append(to01(noisy).cpu()); de01.append(to01(den).cpu())
            side = torch.cat([clean, noisy, den], 3)                       # clean | noisy | denoised
            rows += list(((to01(side) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()))
    os.makedirs(a.out, exist_ok=True)
    imageio.mimsave(f"{a.out}/denoised_video.mp4", rows, fps=a.fps, macro_block_size=1)   # keep exact 3W x H size
    cl, no, de = torch.cat(cl01), torch.cat(no01), torch.cat(de01)
    res = dict(video=a.video, ckpt=a.ckpt, passes=c["refine"], noise=f"{a.noise}:{a.level}", frames=len(rows), size=f"{W}x{H}",
               psnr_noisy=float(np.mean(ps_noisy)), psnr_denoised=float(np.mean(ps_den)),
               flicker_clean=flicker(cl), flicker_noisy=flicker(no), flicker_denoised=flicker(de))
    json.dump(res, open(f"{a.out}/video_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True); p.add_argument("--ckpt", default="runs/unet/ckpt.pt"); p.add_argument("--out", default="runs/video_demo")
    p.add_argument("--noise", default="gaussian", choices=["gaussian", "saltpepper"]); p.add_argument("--level", type=float, default=0.15)
    p.add_argument("--max_frames", type=int, default=64); p.add_argument("--bs", type=int, default=8)
    p.add_argument("--fps", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())