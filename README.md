# AI Video Generation: denoising, video diffusion and LoRA fine-tuning

Everything is sized for a free Colab T4 (15 GB) and was run there, except the video-denoising demo, which ran on a laptop CPU.

## What is in this repository

| Part | Folder | What it is | Details |
|---|---|---|---|
| Task 1 | `task1_denoising/` | Image denoiser (U-Net or autoencoder) trained on Gaussian and salt-and-pepper noise, evaluated with PSNR and SSIM, with a 2x2 ablation and a frame-by-frame video demo | `task1_denoising/README.md` |
| Task 1, coding task | `task1_denoising/video_ddpm/` | Video DDPM from scratch on moving shapes, linear vs cosine schedule | `task1_denoising/video_ddpm/README.md` |
| Task 2 | `task2_lora_amharic_video/` | LoRA fine-tuning of Wan2.1-T2V-1.3B to render Amharic (Ge'ez) words in video | `task2_lora_amharic_video/README.md` |
| Extra | `task2_lora_amharic_video/coffee_lora/`, `coffee_ad/` | Pipeline for an Ethiopian coffee advertisement (LoRA on own footage, four generated scenes, Amharic titles and narration) | Partly tested, see status |
| Notebooks | `notebooks/` | One Colab notebook per part: `task1_denoising`, `task1_video_ddpm`, `task2_lora_amharic_video`, `coffee_lora`, `coffee_ad` | |

## Results at a glance

**Task 1: image denoiser** (CIFAR-10 subset, 32x32, mean PSNR in dB over three noise levels per type, 1,000 test images)

| | Gaussian | Salt-and-pepper |
|---|---|---|
| Noisy input | 15.68 | 15.08 |
| Classical filter (3x3 blur or median) | 21.65 | 24.39 |
| Autoencoder, 1 pass | 23.80 | 29.17 |
| U-Net, 1 pass | 24.15 | 29.62 |
| U-Net, 2 refinement passes | 24.53 | 26.98 |

The U-Net with one pass is best overall. A second refinement pass helps Gaussian noise slightly and hurts salt-and-pepper noise by 1.8 to 3.4 dB; the reason was not investigated. One seed per configuration. Full tables, SSIM and the ablation are in `task1_denoising/README.md`.

**Task 1: video DDPM** (1,500 synthetic moving-shape clips, 8 frames at 32x32, 30 epochs, about 2 minutes per run)

| Run | Test eps-MSE | Motion of generated clips (real: 0.122) |
|---|---|---|
| linear schedule, lr 5e-4 | 0.0092 | 0.352 |
| cosine schedule, lr 5e-4 | 0.0145 | 0.266 |
| cosine schedule, lr 2e-3 | 0.0144 | 0.264 |

The generated clips are poor: ragged blobs, shapes that appear and vanish, and jumpy motion. Cosine gives cleaner clips than linear. The test losses of the two schedules are not comparable. Validation loss was still falling at epoch 30.

**Task 2: Wan2.1 LoRA for Amharic text in video** (rank 16, 11.80 M trainable parameters, 40 synthetic clips, 400 steps)

| | Value |
|---|---|
| Training time and peak GPU memory | 1,389 s, 3.3 GB |
| Generation per 17-frame 256x384 clip, base / LoRA | 31-32 s / 36-37 s, 10.7 GB peak |

In the inspected frames the base model draws shapes that do not look like Ge'ez, and the LoRA draws clean Ge'ez-style letterforms. Whether the words are spelled correctly, whether the motion follows the prompt, and whether unseen words work are **not verified** (see the status table).

## Status and what is verified

| Part | Status |
|---|---|
| Image denoiser | Complete. Four runs (U-Net and autoencoder, 1 and 2 passes). Re-running the first two reproduced their tables to four decimals. |
| Video denoising demo (`denoise_video.py`) | Run on CPU on an 8-frame gif (PSNR 19.3 to 27.0 dB) and a 64-frame mp4 clip (17.3 to 29.2 dB). That clip was almost static, so the second number is optimistic. Not yet tried on footage with real motion or on a Wan clip. |
| Video DDPM | Complete for 30 epochs, three experiments. **The model checkpoint is not in the repository** (`*.pt` is git-ignored). |
| Wan LoRA | One run, qualitative comparison from one frame per clip. Spelling, motion and generalisation not verified.|
| Coffee advertisement | `prepare_clips.py` and `assemble.py` tested on fake data. `gen_scenes.py` and the voice-over (`voiceover.py`, MMS-TTS with a non-commercial licence) are untested. No training on coffee footage has been run. |

## Run

Each part has a Colab notebook in `notebooks/`. Use a T4 runtime and set two Colab secrets, `GH_TOKEN` (a GitHub token starting with `ghp_`) and `HF_TOKEN` (a Hugging Face token), with notebook access on. Never put tokens in the repository or in chat. The notebooks clone the repo, run the experiments and commit the results back.

Command-line equivalents:

```bash
# Task 1: image denoiser
cd task1_denoising && pip install -r requirements.txt
python denoise.py --arch unet --out runs/unet
python denoise_video.py --video clip.mp4 --ckpt runs/unet/ckpt.pt   # frame-by-frame video demo

# Task 1: video DDPM
cd task1_denoising/video_ddpm && pip install -r requirements.txt
python run_experiments.py --epochs 30

# Task 2: Wan LoRA (needs a GPU; run on Colab)
cd task2_lora_amharic_video && pip install -r requirements.txt && pip uninstall -y torchao
python make_amharic_data.py --n 40
python train_lora.py --steps 400 --accum 2 --out outputs/lora_r16
python infer.py --lora_dir outputs/lora_r16 --out outputs/samples
```

On a machine whose system Python is managed by the operating system, create a virtual environment first (`python3 -m venv .venv`).

## Limitations

- Small scale on purpose: a free T4, a 0.76 M and a 0.67 M parameter model, an 11.8 M parameter LoRA, and synthetic data for the video experiments.
- One seed and a few samples per experiment, so small differences are not reliable.
- Task 2 training data is synthetic rendered text; the model was tested only on words it was trained on.
- Video results use simple proxy metrics (PSNR, frame-to-frame motion, distance to the nearest training clip), not standard video-generation metrics.

## Licence

MIT. See `LICENSE`.