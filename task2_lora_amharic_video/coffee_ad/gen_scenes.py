"""Stage 1 (GPU): generate the four ad scenes with pretrained Wan2.1-1.3B (optionally + your Task 2 LoRA).
Text is NOT requested from the model (it cannot spell Ge'ez); Amharic titles are added in assemble.py."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import MODEL_ID, seed_all, pick_dtype, encode_texts

PROMPTS = [
    "Slow motion close-up of glowing roasted coffee beans falling through warm amber light on a dark background, tiny golden particles, shallow depth of field, cinematic, premium commercial",
    "Traditional Ethiopian clay jebena pouring dark coffee into small white cups on a tray, delicate steam rising, warm inviting light, close-up, cinematic, shallow depth of field",
    "Macro shot of a coffee cup on a dark reflective surface, steam forming a graceful glowing golden flame-shaped swirl above the cup, golden sparkles, slow motion, cinematic product commercial",
    "Steaming coffee cup beside a clay jebena and roasted coffee beans on a dark background with golden highlights, slow camera push-in, polished premium product commercial",
]
NEG = "text, letters, watermark, logo, distorted cup, extra hands, malformed hands, blurry, flicker, low quality, overexposed"


def main(a):
    seed_all(a.seed); dev = "cuda"; dtype = pick_dtype(a.dtype); os.makedirs(a.out, exist_ok=True)
    ids = [int(i) - 1 for i in a.scenes.split(",")]
    prompts = [a.prefix + q for q in PROMPTS]   # same style prefix as the training captions
    embs = encode_texts(prompts + [NEG])  # text encoder is freed afterwards
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers.utils import export_to_video
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    tr = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
    pipe = WanPipeline.from_pretrained(MODEL_ID, transformer=tr, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=dtype).to(dev)
    pipe.vae.enable_tiling()
    if a.lora_dir:
        pipe.load_lora_weights(a.lora_dir)
    neg = embs[-1:].to(dev, dtype); torch.cuda.reset_peak_memory_stats()
    for k in ids:
        t = time.time(); g = torch.Generator(dev).manual_seed(a.seed + k)
        fr = pipe(prompt_embeds=embs[k:k + 1].to(dev, dtype), negative_prompt_embeds=neg, height=a.height, width=a.width,
                  num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g).frames[0]
        export_to_video(fr, f"{a.out}/scene{k+1}.mp4", fps=16)
        print(f"scene {k+1}: {time.time()-t:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
    json.dump(dict(prompts=prompts, negative=NEG, args=vars(a)), open(f"{a.out}/prompts.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="scenes"); p.add_argument("--scenes", default="1,2,3,4", help="e.g. 3 to redo scene 3 only")
    p.add_argument("--height", type=int, default=512); p.add_argument("--width", type=int, default=288)  # portrait 9:16
    p.add_argument("--frames", type=int, default=33); p.add_argument("--steps", type=int, default=25); p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--lora_dir", default=None); p.add_argument("--prefix", default="", help="e.g. 'Ethiopian coffee commercial style: '"); p.add_argument("--dtype", default="auto"); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
