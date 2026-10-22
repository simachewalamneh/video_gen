# Task 1: Image denoising (U-Net vs autoencoder, with iterative refinement)

A compact denoiser trained on synthetic noise, evaluated with PSNR and SSIM against classical baselines, with a 2x2 ablation (architecture x refinement passes). Everything runs on a free Colab T4 in about one minute per model.

This is single-step image denoising: the network maps a noisy image directly to a clean one. It has no timestep input and no noise schedule, so it is not a diffusion model. There is no text prompt; the input is a noisy image and the target is the clean image.

## Files

| Path | Content |
|---|---|
| `denoise.py` | Data, noise, model, training, evaluation (single script) |
| `denoise_video.py` | Applies a trained checkpoint to a video frame by frame |
| `requirements.txt` | Python dependencies |
| `runs/<name>/` | `config.json`, `results.md`, `summary.json`, `loss.png`, `samples.png` for each run |
| `runs/unet/ckpt.pt` | Checkpoint of the default U-Net (2 refinement passes) |
| `../notebooks/task1_denoising.ipynb` | Colab notebook that runs and commits all four experiments |

## Run

```bash
pip install -r requirements.txt
python denoise.py --arch unet --out runs/unet                # U-Net, 2 passes (default)
python denoise.py --arch ae   --out runs/ae --refine 1       # autoencoder, 1 pass
python denoise.py --arch unet --refine 1 --out runs/unet_r1  # ablation
python denoise.py --arch ae   --refine 2 --out runs/ae_r2    # ablation
```

The first run downloads CIFAR-10 (170 MB). Use `--data_dir <folder>` to train on your own images (ImageFolder layout).

## Pipeline

1. **Data.** 10,000 CIFAR-10 images at 32x32, split 80/10/10 into train / validation / test with a fixed seed (`--seed 0`).
2. **Preprocessing.** Resize, centre-crop, convert to tensor, normalise to [-1, 1]. This is done before any noise is added.
3. **Noise (synthetic, added on the fly).**
   - Gaussian: sigma drawn from 0.05 to 0.30 per image (on the [0, 1] scale).
   - Salt-and-pepper: 2% to 20% of pixels set to the extremes.
   - Training uses a random level per image, and each batch is one noise kind or the other (`--noise mixed`).
4. **Training.** 15 epochs, batch 128, AdamW (lr 2e-3, weight decay 1e-4) with a one-cycle schedule, fp16 mixed precision, gradient clipping at 1.0.
5. **Evaluation.** The 1,000 held-out test images at fixed noise levels and a fixed noise seed (1234).

## Model

- **U-Net (`--arch unet`).** Base width 32 (32, 64, 128 channels). Two stride-2 downsampling steps, a bottleneck, two transposed-convolution upsampling steps, residual blocks (GroupNorm + SiLU), and skip connections by concatenation. 0.67 M parameters.
- **Autoencoder (`--arch ae`).** The same network without the skip connections. 0.62 M parameters.
- **Output.** 6 channels: a 3-channel residual added to the input, and a 3-channel per-pixel log-variance (the uncertainty head).
- **Loss.** MSE plus 0.1 x Gaussian negative log-likelihood. The NLL term can be negative, so the training loss can go below zero. Compare runs by validation MSE, not by training loss.
- **Iterative refinement (`--refine K`).** The output is fed back into the network for K passes. The loss is summed over the passes, and the input to pass 2 is detached from the graph.

## Evaluation protocol

- **PSNR and SSIM** on the [0, 1] scale, computed per image and averaged. SSIM uses an 11x11 Gaussian window.
- **Baselines.** A 3x3 median filter for salt-and-pepper and a 3x3 Gaussian blur for Gaussian noise.
- **Noise levels.** Gaussian sigma 0.1, 0.2, 0.3. Salt-and-pepper 5%, 10%, 20%.
- **Uncertainty check.** Pixel-wise correlation between predicted standard deviation and absolute error (first-pass output).

## Results

PSNR in dB (higher is better). For 2-pass models the column shows pass 2.

| Noise | Noisy | Classical | AE, 1 pass | U-Net, 1 pass | AE, 2 passes | U-Net, 2 passes |
|---|---|---|---|---|---|---|
| Gaussian 0.1 | 20.32 | 24.96 | 27.06 | 27.36 | 27.45 | 27.50 |
| Gaussian 0.2 | 14.79 | 21.29 | 23.20 | 23.63 | 24.00 | 24.10 |
| Gaussian 0.3 | 11.94 | 18.71 | 21.13 | 21.48 | 21.83 | 21.98 |
| Salt-and-pepper 5% | 18.09 | 25.57 | 31.91 | 32.51 | 28.90 | 29.12 |
| Salt-and-pepper 10% | 15.08 | 24.82 | 29.68 | 30.07 | 27.13 | 27.39 |
| Salt-and-pepper 20% | 12.05 | 22.78 | 25.92 | 26.26 | 24.11 | 24.43 |

SSIM at Gaussian 0.2 and salt-and-pepper 10%:

| Model | Gaussian 0.2 | Salt-and-pepper 10% |
|---|---|---|
| Noisy | 0.486 | 0.547 |
| AE, 1 pass | 0.812 | 0.954 |
| U-Net, 1 pass | 0.822 | 0.958 |
| AE, 2 passes (pass 2) | 0.841 | 0.921 |
| U-Net, 2 passes (pass 2) | 0.844 | 0.924 |

Mean PSNR over the three levels of each noise type:

| Model | Gaussian | Salt-and-pepper |
|---|---|---|
| Noisy | 15.68 | 15.08 |
| Classical | 21.65 | 24.39 |
| AE, 1 pass | 23.80 | 29.17 |
| U-Net, 1 pass | 24.15 | 29.62 |
| AE, 2 passes | 24.43 | 26.71 |
| U-Net, 2 passes | 24.53 | 26.98 |

Training cost (T4, 15 epochs): U-Net 2 passes 71 s, AE 1 pass 50 s, U-Net 1 pass 52 s, AE 2 passes 64 s. Peak GPU memory is recorded in each `summary.json` (811 MB for the default U-Net, 350 MB for the AE).

Figures: `runs/<name>/samples.png` (rows: clean, noisy, pass 1, pass 2 if any, uncertainty map; upper half Gaussian 0.2, lower half salt-and-pepper 10%) and `runs/<name>/loss.png` (training loss and validation MSE per epoch).

## Findings

1. **All models beat the classical baselines** by 2.2 to 5.2 dB in mean PSNR.
2. **Skip connections help slightly.** At equal passes the U-Net is 0.05 to 0.6 dB above the AE.
3. **Refinement helps Gaussian noise a little** (about 0.5 to 0.8 dB at sigma 0.2 and 0.3, and a higher SSIM).
4. **Refinement hurts salt-and-pepper noise** by 1.8 to 3.4 dB at every level. The cause was not investigated.
5. **Best overall configuration: U-Net with 1 pass** (`--arch unet --refine 1`). The script default is `--refine 2`, which is better only on Gaussian noise.
6. **Uncertainty is informative but weak.** Correlation with real error is 0.19 to 0.54, higher for salt-and-pepper and for 1-pass models.
7. **Training is stable.** Validation MSE flattens after about epoch 11 in all four runs and does not rise, so there is no sign of overfitting.

## Limitations

- One seed per configuration, so differences below about 0.5 dB should not be treated as reliable. Re-running the AE and U-Net on the same seed reproduced the tables to four decimals, which shows reproducibility, not variance across seeds.
- Small data (10k CIFAR-10 images at 32x32, 1,000 test images) and synthetic noise only. Results on real camera noise or larger images are not measured.
- The classical baselines are 3x3 filters and are weak.
 