"""Create a tiny SYNTHETIC Amharic (Ge'ez script) text-animation dataset: clips of Amharic words that move/zoom/fade.
Writes <out>/clips/*.mp4 + <out>/metadata.jsonl  ({"file","caption"}).  No GPU needed.
To use your OWN videos instead: put mp4s in a folder and write the same metadata.jsonl (Amharic or English captions).
Setup (Colab): apt-get install -y fonts-noto-core   # provides Noto Sans Ethiopic
"""
import argparse, glob, json, os, random, subprocess
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

WORDS = [("ሰላም", "peace"), ("ኢትዮጵያ", "Ethiopia"), ("ቡና", "coffee"), ("ፍቅር", "love"), ("ጤና", "health"),
         ("ትምህርት", "education"), ("አመሰግናለሁ", "thank you"), ("እንኳን ደህና መጣህ", "welcome"), ("ብርሃን", "light"), ("ሀገር", "country")]
BG = {"ሰማያዊ": ((20, 40, 110), "blue"), "አረንጓዴ": ((15, 90, 45), "green"), "ጥቁር": ((15, 15, 15), "black"), "ቀይ": ((120, 25, 25), "red")}
MOTIONS = {"slide": "slides from left to right", "zoom": "zooms in", "fade": "fades in", "rise": "rises upward"}


def find_font():
    c = glob.glob("/usr/share/fonts/**/*Ethiopic*Regular*.ttf", recursive=True) + glob.glob("/usr/share/fonts/**/*Ethiopic*.ttf", recursive=True)
    if not c:
        try: c = [subprocess.check_output(["fc-match", "-f", "%{file}", ":lang=am"]).decode()]
        except Exception: pass
    if not c: raise SystemExit("No Ethiopic font. Run: apt-get install -y fonts-noto-core")
    return c[0]


def render(word, motion, bg, H, W, F_, font_path):
    frames = []
    for t in range(F_):
        p = t / (F_ - 1)
        size = int(H * (0.3 + (0.25 * p if motion == "zoom" else 0)))
        font = ImageFont.truetype(font_path, size)
        im = Image.new("RGB", (W, H), bg); d = ImageDraw.Draw(im)
        l, tp, r, b = d.textbbox((0, 0), word, font=font); tw, th = r - l, b - tp
        x, y = (W - tw) / 2, (H - th) / 2 - tp
        if motion == "slide": x = (W - tw) * (0.05 + 0.9 * p)
        if motion == "rise": y = (H - th) * (0.9 - 0.8 * p) - tp
        col = (255, 255, 255) if motion != "fade" else tuple(int(bg[k] + (255 - bg[k]) * min(1, p * 1.5)) for k in range(3))
        d.text((x, y), word, font=font, fill=col)
        frames.append(np.array(im))
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/amharic_synth"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--frames", type=int, default=17); ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed)
    os.makedirs(f"{a.out}/clips", exist_ok=True); font = find_font(); meta = []
    combos = [(w, m, b) for w in WORDS for m in MOTIONS for b in BG]; random.shuffle(combos)
    for i, ((am, en), m, b) in enumerate(combos[:a.n]):
        bg, bgen = BG[b]; fn = f"clips/{i:03d}.mp4"
        imageio.mimsave(f"{a.out}/{fn}", render(am, m, bg, a.height, a.width, a.frames, font), fps=8)
        cap = f'The Amharic word "{am}" ({en}) in white Ge\'ez script {MOTIONS[m]} on a {bgen} background. የአማርኛ ቃል "{am}"'
        meta.append(dict(file=fn, caption=cap))
    open(f"{a.out}/metadata.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(x, ensure_ascii=False) for x in meta))
    print(f"wrote {len(meta)} clips to {a.out}")
