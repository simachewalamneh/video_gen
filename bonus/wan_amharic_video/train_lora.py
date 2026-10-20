"""LoRA fine-tune Wan2.1-T2V-1.3B on a small Amharic video set, sized for a free Colab T4 (15 GB).
Memory tricks: (1) pre-cache text embeddings + VAE latents, then free the 11 GB text encoder and VAE;
(2) frozen fp16/bf16 transformer + fp32 LoRA weights only; (3) gradient checkpointing; (4) grad accumulation, batch 1;
(5) low resolution/few frames.  Flow matching: x_t=(1-s)x0+s*noise, target velocity = noise - x0 (Wan convention).
"""
import argparse, csv, gc, os, time, torch, torch.nn.functional as F
from common import MODEL_ID, seed_all, pick_dtype, encode_texts, read_video, load_metadata


def main(a):
    seed_all(a.seed); dev = "cuda"; dtype = pick_dtype(a.dtype); os.makedirs(a.out, exist_ok=True)
    meta = load_metadata(a.data_dir)
    caps = [m["caption"] for m in meta]
    cache = f"{a.out}/cache.pt"
    if os.path.exists(cache):
        embs, lats = torch.load(cache)
    else:
        embs = encode_texts(caps)  # (N,226,4096) bf16, encoder freed afterwards
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(dev)
        inv_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, -1, 1, 1, 1).to(dev)
        lats = []
        with torch.no_grad():
            for m in meta:
                x = read_video(f"{a.data_dir}/{m['file']}", a.frames, a.height, a.width)[None].to(dev)
                lats.append(((vae.encode(x).latent_dist.mode() - mean) * inv_std).half().cpu())
        lats = torch.cat(lats); del vae; gc.collect(); torch.cuda.empty_cache()
        torch.save((embs, lats), cache)
    print("cached:", tuple(embs.shape), tuple(lats.shape))

    from diffusers import WanTransformer3DModel, WanPipeline
    from diffusers.utils import get_peft_model_state_dict
    from peft import LoraConfig
    tr = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype).to(dev)
    tr.requires_grad_(False)
    tr.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.alpha, init_lora_weights=True, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    params = [p for p in tr.parameters() if p.requires_grad]
    for p in params: p.data = p.data.float()  # fp32 trainable weights
    tr.enable_gradient_checkpointing(); tr.train()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    print(f"trainable LoRA params: {sum(p.numel() for p in params)/1e6:.2f}M | dtype {dtype}")
    torch.cuda.reset_peak_memory_stats(); log = []; t0 = time.time(); N = len(meta)
    for step in range(1, a.steps + 1):
        tot = 0.0
        for _ in range(a.accum):
            i = torch.randint(0, N, (1,)).item()
            x0 = lats[i:i + 1].to(dev).float(); e = embs[i:i + 1].to(dev, dtype)
            noise = torch.randn_like(x0)
            u = torch.sigmoid(torch.randn(1, device=dev))              # logit-normal timestep sampling
            s = a.shift * u / (1 + (a.shift - 1) * u)                   # same shift as inference scheduler
            xt = (1 - s.view(1, 1, 1, 1, 1)) * x0 + s.view(1, 1, 1, 1, 1) * noise
            with torch.autocast("cuda", dtype=dtype):
                pred = tr(hidden_states=xt.to(dtype), timestep=(s * 1000), encoder_hidden_states=e, return_dict=False)[0]
            loss = F.mse_loss(pred.float(), noise - x0) / a.accum
            scaler.scale(loss).backward(); tot += loss.item()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        log.append((step, tot))
        if step % 10 == 0: print(f"step {step}/{a.steps} loss {tot:.4f} | {time.time()-t0:.0f}s | peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            WanPipeline.save_lora_weights(save_directory=a.out, transformer_lora_layers=get_peft_model_state_dict(tr))
    with open(f"{a.out}/train_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "loss"]); w.writerows(log)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.plot([l[1] for l in log]); plt.xlabel("step"); plt.ylabel("flow-matching MSE"); plt.savefig(f"{a.out}/loss.png", dpi=120)
    print(f"done in {time.time()-t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/amharic_synth"); p.add_argument("--out", default="outputs/lora_r16")
    p.add_argument("--rank", type=int, default=16); p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--frames", type=int, default=17); p.add_argument("--height", type=int, default=256); p.add_argument("--width", type=int, default=384)
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"]); p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
