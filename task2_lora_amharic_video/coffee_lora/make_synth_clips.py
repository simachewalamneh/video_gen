"""Option B data: generate varied coffee clips with pretrained Wan (GPU), to be filtered by curate_clips.py and used as a
self-curated LoRA training set (rejection-sampling style fine-tuning). Clips match the training size (512x288, 17 frames).
Reruns skip clips that already exist, so a cut Colab session can be resumed."""
import argparse, json, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import MODEL_ID, seed_all, pick_dtype, encode_texts

SUBJECTS = ["roasted coffee beans falling in slow motion", "a traditional Ethiopian clay jebena pouring dark coffee into small white cups",
            "green coffee beans roasting in a metal pan over charcoal with smoke rising", "a tray of small white coffee cups with steam rising",
            "a single coffee cup with delicate steam on a dark surface", "a clay jebena on a charcoal stove with steam rising from the spout",
            "dark roasted coffee beans in a wooden bowl"]
LIGHTS = ["warm amber light", "soft golden window light", "dark background with golden highlights", "warm glowing light"]
CAMS = ["close-up", "macro shot", "slow camera push-in on", "shallow depth of field close-up of"]
NEG = "text, letters, watermark, logo, distorted cup, extra hands, malformed hands, blurry, flicker, low quality, overexposed"


def build_prompts(n, seed):
    rng, out, seen = random.Random(seed), [], set()
    while len(out) < n:
        p = f"{rng.choice(CAMS)} {SUBJECTS[len(out) % len(SUBJECTS)]}, {rng.choice(LIGHTS)}, cinematic, photorealistic"
        if p not in seen: seen.add(p); out.append(p)
    return out


def main(a):
    seed_all(a.seed); dev = "cuda"; dtype = pick_dtype(a.dtype); os.makedirs(f"{a.out}/clips", exist_ok=True)
    prompts = build_prompts(a.n, a.seed)
    todo = [i for i in range(a.n) if not os.path.exists(f"{a.out}/clips/c{i:03d}.mp4")]
    meta = [dict(file=f"clips/c{i:03d}.mp4", prompt=p, caption=a.prefix + p, seed=a.seed + i) for i, p in enumerate(prompts)]
    if todo:
        embs = encode_texts([prompts[i] for i in todo] + [NEG])  # text encoder freed afterwards
        from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
        from diffusers.utils import export_to_video
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        tr = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
        pipe = WanPipeline.from_pretrained(MODEL_ID, transformer=tr, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=dtype).to(dev)
        pipe.vae.enable_tiling(); neg = embs[-1:].to(dev, dtype); torch.cuda.reset_peak_memory_stats()
        for j, i in enumerate(todo):
            t = time.time(); g = torch.Generator(dev).manual_seed(a.seed + i)
            fr = pipe(prompt_embeds=embs[j:j + 1].to(dev, dtype), negative_prompt_embeds=neg, height=a.height, width=a.width,
                      num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g).frames[0]
            export_to_video(fr, f"{a.out}/clips/c{i:03d}.mp4", fps=16)
            print(f"clip {i+1}/{a.n}: {time.time()-t:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
    open(f"{a.out}/all_metadata.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(m, ensure_ascii=False) for m in meta))
    print(f"{a.n} candidate clips in {a.out}/clips")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/coffee_synth"); p.add_argument("--n", type=int, default=40)
    p.add_argument("--prefix", default="Ethiopian coffee commercial style: ")
    p.add_argument("--height", type=int, default=512); p.add_argument("--width", type=int, default=288); p.add_argument("--frames", type=int, default=17)
    p.add_argument("--steps", type=int, default=25); p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--dtype", default="auto"); p.add_argument("--seed", type=int, default=100)
    main(p.parse_args())
