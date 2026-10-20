"""Generate Amharic-prompt videos with the base model and with the trained LoRA (before/after comparison)."""
import argparse, gc, os, time, torch
from common import MODEL_ID, seed_all, pick_dtype, encode_texts

DEFAULT = ['The Amharic word "ሰላም" (peace) in white Ge\'ez script slides from left to right on a blue background. የአማርኛ ቃል "ሰላም"',
           'The Amharic word "ቡና" (coffee) in white Ge\'ez script zooms in on a black background. የአማርኛ ቃል "ቡና"']


def main(a):
    seed_all(a.seed); dev = "cuda"; dtype = pick_dtype(a.dtype); os.makedirs(a.out, exist_ok=True)
    prompts = a.prompt or DEFAULT
    embs = encode_texts(prompts + [""])  # last = empty negative prompt
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
    from diffusers.utils import export_to_video
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    tr = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
    pipe = WanPipeline.from_pretrained(MODEL_ID, transformer=tr, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=dtype).to(dev)
    pipe.vae.enable_tiling()
    neg = embs[-1:].to(dev, dtype)

    def run(tag):
        for k in range(len(prompts)):
            t = time.time()
            g = torch.Generator(dev).manual_seed(a.seed + k)
            fr = pipe(prompt_embeds=embs[k:k + 1].to(dev, dtype), negative_prompt_embeds=neg, height=a.height, width=a.width,
                      num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g).frames[0]
            export_to_video(fr, f"{a.out}/{tag}_{k}.mp4", fps=8)
            print(f"{tag} prompt {k}: {time.time()-t:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)

    run("base")
    pipe.load_lora_weights(a.lora_dir)
    run("lora")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", default="outputs/lora_r16"); p.add_argument("--out", default="outputs/samples")
    p.add_argument("--prompt", action="append"); p.add_argument("--steps", type=int, default=25); p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--frames", type=int, default=17); p.add_argument("--height", type=int, default=256); p.add_argument("--width", type=int, default=384)
    p.add_argument("--dtype", default="auto"); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
