"""Shared helpers: Wan2.1-1.3B text encoding (encoder freed afterwards), video reading/preprocessing."""
import gc, json, random, numpy as np, torch, torch.nn.functional as F

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # umt5 text encoder is multilingual (covers Amharic)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def pick_dtype(name="auto"):
    if name == "bf16": return torch.bfloat16
    if name == "fp16": return torch.float16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # T4 -> fp16


@torch.no_grad()
def encode_texts(prompts, max_len=226, device="cuda"):
    """Encode prompts with umt5 (bf16), return (N,max_len,4096) bf16 on CPU, then free the 11 GB encoder."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    tok = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    out = []
    for p in prompts:
        t = tok([p], padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True, return_tensors="pt")
        h = enc(t.input_ids.to(device), t.attention_mask.to(device)).last_hidden_state.clone()
        h[:, int(t.attention_mask.sum()):] = 0
        out.append(h.cpu())
    del enc; gc.collect(); torch.cuda.empty_cache()
    return torch.cat(out)


def read_video(path, frames, H, W):
    """Preprocess: uniform temporal sampling -> centre-crop to target aspect -> resize -> normalise [-1,1]. -> (3,F,H,W)"""
    import imageio.v2 as imageio
    fr = [f[..., :3] for f in imageio.get_reader(path)]
    idx = np.linspace(0, len(fr) - 1, frames).round().astype(int)
    x = torch.from_numpy(np.stack([fr[i] for i in idx])).permute(3, 0, 1, 2).float() / 255  # 3,F,h,w
    h, w = x.shape[-2:]; ar = W / H
    ch, cw = (h, int(h * ar)) if w / h >= ar else (int(w / ar), w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    x = x[..., y0:y0 + ch, x0:x0 + cw]
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x * 2 - 1


def load_metadata(data_dir):
    return [json.loads(l) for l in open(f"{data_dir}/metadata.jsonl", encoding="utf-8") if l.strip()]
