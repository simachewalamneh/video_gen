"""Stage 3: Amharic narration (facebook/mms-tts-amh, CC-BY-NC 4.0: non-commercial; use a recorded native speaker for commercial use)
+ optional music, muxed into the final MP4. Narration lines are timed to the on-screen scenes."""
import argparse, subprocess, wave
import numpy as np, torch

LINES = [("ቡና አምሮዎታል?", 0.3), ("የባህል መዓዛ፣ የተፈጥሮ ጣዕም፣ በእያንዳንዱ ኩባያ!", 8.3), ("የኢትዮጵያ ቡናን ይምረጡ!", 12.6)]  # (text, start seconds)


def romanize(text):
    try:
        import uroman as ur
        return ur.Uroman().romanize_string(text)
    except Exception as e:
        raise SystemExit(f"uroman failed ({e}). Run: pip install uroman")


def main(a):
    import imageio_ffmpeg, imageio.v2 as imageio
    from transformers import VitsModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-amh"); model = VitsModel.from_pretrained("facebook/mms-tts-amh").to(dev).eval()
    sr = model.config.sampling_rate
    dur = imageio.get_reader(a.video).get_meta_data().get("duration") or 16.0
    track = np.zeros(int(dur * sr) + sr, np.float32)
    for text, start in LINES:
        t = romanize(text) if getattr(tok, "is_uroman", False) else text
        inp = {k: v.to(dev) for k, v in tok(t, return_tensors="pt").items()}
        with torch.no_grad(): wav = model(**inp).waveform[0].cpu().numpy()
        i = int(start * sr); track[i:i + len(wav)] += wav[:len(track) - i]
    track = track / max(1e-6, np.abs(track).max()) * 0.9
    with wave.open(a.voice, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes((track * 32767).astype(np.int16).tobytes())
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if a.music:
        cmd = [ff, "-y", "-i", a.video, "-i", a.voice, "-i", a.music, "-filter_complex",
               f"[2:a]volume=0.25,atrim=0:{dur}[m];[1:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", a.out]
    else:
        cmd = [ff, "-y", "-i", a.video, "-i", a.voice, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", a.out]
    subprocess.run(cmd, check=True); print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="ad_silent.mp4"); p.add_argument("--voice", default="voice.wav")
    p.add_argument("--music", default=None, help="optional licensed music file"); p.add_argument("--out", default="ethiopian_coffee_ad.mp4")
    main(p.parse_args())
