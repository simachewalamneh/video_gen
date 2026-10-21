# AI Video Generation

| Task | Folder | Content |
|---|---|---|
| 1 | `task1_denoising/` | Image denoising: preprocessing → Gaussian / salt-and-pepper noise → compact U-Net (or autoencoder) → PSNR/SSIM |
| 2 | `task2_lora_amharic_text/` | LoRA fine-tuning of mT5-small for English→Amharic generation, optimised for Colab free tier |
| bonus | `bonus/video_ddpm/` | Video DDPM from scratch (slide 15 task): noising, 3D-conv denoiser, linear vs cosine schedule |
| bonus | `bonus/wan_amharic_video/` | LoRA on Wan2.1-1.3B for Amharic (Ge'ez) text-in-video; heavy, untested |

Run via the notebooks in `notebooks/` (set `REPO_URL` in the first cell; they commit results back to GitHub).

## Task 1 – `denoise.py`
- Data: CIFAR-10 subset (10k images, 32×32) or your own folder (`--data_dir`); 80/10/10 split, fixed seed.
- Preprocessing: resize → centre-crop → tensor → normalise to [-1,1], applied before any corruption.
- Noise: Gaussian (σ 0.05–0.3) and salt-and-pepper (2–20 % pixels), random level per image during training; fixed levels at test.
- Model: compact U-Net (`--arch unet`) or autoencoder (`--arch ae`), residual output, GroupNorm, AMP (fp16).
- Course concepts: **uncertainty-aware** head (per-pixel log-variance, Gaussian NLL; reports correlation of predicted uncertainty with real error) and **iterative error correction** (`--refine 2`: output fed back through the network, trained with detached second pass).
- Evaluation: PSNR and SSIM vs noisy input and classical baselines (median / Gaussian blur), per pass.

## Task 2 – `lora_amharic.py`
- Base: `google/mt5-small` (multilingual, covers Amharic); data: OPUS-100 am-en, or `--tsv english<TAB>amharic`.
- Only LoRA matrices (q,k,v,o; rank 8) train; gradient checkpointing; fp32 base (mT5 overflows in fp16; bf16 autocast used when the GPU supports it).
- **Error-focused replay** (iterative error correction): round 2 re-samples training pairs in proportion to their round-1 loss.
- **Uncertainty-aware decoding**: MC-dropout candidates, agreement = confidence, MBR selection, chrF on the confident half.
- Reports trainable-parameter %, peak GPU memory vs a full fine-tune estimate, chrF/BLEU for base vs LoRA vs MC-MBR.

## Results
_Paste `results.md`, `summary.json`, loss graphs, and failure cases after running._
