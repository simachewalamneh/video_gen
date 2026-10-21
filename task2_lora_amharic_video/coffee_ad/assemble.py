"""Stage 2 (CPU): smooth slow motion, dissolves, upscale, exact Amharic titles -> silent ad video."""
import argparse, os, sys
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_amharic_data import find_font

TEXTS = ["ቡና አምሮዎታል?", "የኢትዮጵያ ቡና", "የባህል መዓዛ፣ የተፈጥሮ ጣዕም", "የኢትዮጵያ ቡናን ይምረጡ!"]  # have a native speaker verify


def read(path):
    return [f[..., :3] for f in imageio.get_reader(path)]


def interp(fr):  # insert a blended frame between neighbours (33 -> 65 frames): smoother slow motion
    out = []
    for i in range(len(fr) - 1):
        out += [fr[i], ((fr[i].astype(np.uint16) + fr[i + 1]) // 2).astype(np.uint8)]
    return out + [fr[-1]]


def fit_font(path, text, max_w):
    size = max_w // 6
    while size > 12:
        f = ImageFont.truetype(path, size)
        l, t, r, b = f.getbbox(text)
        if r - l <= max_w * 0.9: return f
        size -= 4
    return ImageFont.truetype(path, 12)


def overlay(frame, text, alpha, font):
    im = Image.fromarray(frame).convert("RGBA"); W, H = im.size
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    l, t, r, b = d.textbbox((0, 0), text, font=font); tw, th = r - l, b - t
    x, y, pad = (W - tw) // 2 - l, int(H * 0.78) - t, int(font.size * 0.35)
    d.rounded_rectangle([x + l - pad, y + t - pad, x + l + tw + pad, y + t + th + pad], radius=pad, fill=(0, 0, 0, int(110 * alpha)))
    d.text((x, y), text, font=font, fill=(255, 246, 225, int(255 * alpha)))  # ivory
    return np.array(Image.alpha_composite(im, layer).convert("RGB"))


def main(a):
    scenes = [interp(read(f"{a.scenes}/scene{i}.mp4")) for i in range(1, 5)]
    n, out, spans = a.dissolve, [], []
    for i, s in enumerate(scenes):
        if i == 0:
            spans.append((0, len(s))); out = list(s)
        else:
            start = len(out) - n; spans.append((start, start + len(s)))
            for k in range(n):
                w = (k + 1) / (n + 1)
                out[-n + k] = ((1 - w) * out[-n + k] + w * s[k]).astype(np.uint8)
            out += s[n:]
    W, H = a.width, a.height; fp = find_font(); fonts = [fit_font(fp, t, W) for t in TEXTS]; fade = int(0.4 * a.fps)
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    for t, fr in enumerate(out):
        fr = np.array(Image.fromarray(fr).resize((W, H), Image.LANCZOS))
        for i, (s0, s1) in enumerate(spans):
            ts, te = s0 + fade, s1 - fade
            if ts <= t < te or (s0 <= t < ts) or (te <= t < s1):
                alpha = max(0.0, min(1.0, (t - s0) / max(fade, 1), (s1 - t) / max(fade, 1)))
                fr = overlay(fr, TEXTS[i], alpha, fonts[i]); break
        w.append_data(fr)
    w.close(); print(f"wrote {a.out}: {len(out)} frames, {len(out)/a.fps:.1f}s at {W}x{H}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="scenes"); p.add_argument("--out", default="ad_silent.mp4")
    p.add_argument("--width", type=int, default=720); p.add_argument("--height", type=int, default=1280)
    p.add_argument("--fps", type=int, default=16); p.add_argument("--dissolve", type=int, default=6)
    main(p.parse_args())
