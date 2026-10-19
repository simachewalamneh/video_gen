# iCog Labs – AI Video Generation (training assignments)

| Task | Folder | What it does | Runs on |
|---|---|---|---|
| 1 | `task1_video_ddpm/` | Video DDPM from scratch: preprocessing, forward noising, 3D-conv denoiser, training, reverse sampling, linear-vs-cosine experiments | Free Colab T4 (or CPU, slower) |
| 2 | `task2_lora_amharic_video/` | LoRA fine-tune of Wan2.1-T2V-1.3B for Amharic (Ge'ez) text-in-video, before/after comparison | Free Colab T4 |

## Run (Colab)
Open `notebooks/task1_video_ddpm.ipynb` and `notebooks/task2_lora_amharic.ipynb` (set `REPO_URL` in the first cell). Each notebook clones the repo, installs, runs, and commits results back.

## Local
```bash
cd task1_video_ddpm && pip install -r requirements.txt
python model.py                      # shape + gradient sanity checks
python train.py --epochs 30          # one run
python run_experiments.py --epochs 30   # linear vs cosine vs lr change -> results/experiments.md
```
```bash
cd task2_lora_amharic_video && pip install -r requirements.txt
apt-get install -y fonts-noto-core   # Ge'ez font (Colab)
python make_amharic_data.py --n 40
python train_lora.py --steps 300
python infer.py
```

## Slide concepts → where implemented
| Slide concept | Code |
|---|---|
| Forward diffusion, `x_t = √ᾱ x0 + √(1-ᾱ) ε` | `diffusion.py: q_sample` |
| Reverse sampling (DDPM ancestral, DDIM fast) | `diffusion.py: sample` |
| Timestep conditioning (sinusoidal + MLP), 3D conv, residual blocks, shape preservation | `model.py` |
| Training loop, `set_to_none`, cached schedule on device, seeds, checkpoints (weights + optimizer + config) | `train.py` |
| Linear vs cosine schedule, hyperparameter effect | `run_experiments.py` |
| Runtime and GPU memory measurement | `summary.json` (`train_s`, `sample_s`, `peak_mem_mb`) |
| Mixed precision, gradient checkpointing | `train.py --amp`, `--grad_ckpt`; Task 2 |
| Flow matching (velocity target) | `task2/train_lora.py` |
| LoRA `y = W0x + (α/r)BAx` (rank r) | `task2/train_lora.py` (peft) |
| Offloading / VAE tiling / cached latents | `task2/train_lora.py`, `infer.py` |
| Wan2.1 multilingual text encoder, causal video VAE | Task 2 pipeline |

## Task 1 notes
- Preprocessing: grayscale → uniform temporal sampling → centre-crop → resize → normalise to [-1,1]; train-only flip and time-reversal augmentation. Real videos: `--video_dir path/`.
- Split 80/10/10 with fixed seed. EMA weights used for validation/sampling.
- Metrics: test eps-MSE, motion energy (real vs generated), nearest-neighbour distance to training clips (memorisation check), sampling time, peak GPU memory.

## Task 2 notes
- Wan's umt5 encoder is multilingual, but base Wan cannot render Ge'ez text; the LoRA teaches it using the synthetic Amharic dataset (or your own clips + `metadata.jsonl`).
- Captions mix English with Amharic words; have an Amharic speaker verify any Amharic wording you add.
- Limitations: tiny dataset, low resolution (256×384, 17 frames), short training → expect imperfect glyphs. Document failures in `results/`.

## Results
_Fill in after running: paste `results/experiments.md`, loss graphs, generated GIFs/MP4s, and observed failure cases._
