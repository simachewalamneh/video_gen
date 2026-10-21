# Video DDPM from scratch (moving shapes): linear vs cosine noise schedule

A small video diffusion model written in PyTorch: synthetic moving-shape clips, forward noising, a 3D-convolution denoiser, the epsilon-prediction loss, DDPM and DDIM reverse sampling, and a controlled comparison of linear and cosine schedules. It is the coding task of the training deck (slide 15) and trains in about 2 minutes per run on a free Colab T4.

This folder is separate from the image denoiser in the parent folder (`../denoise.py`, see `../README.md`). That one is a single-step denoiser with no timestep; this one is a diffusion model with a timestep and a reverse sampling loop.

## Files

| Path | Content |
|---|---|
| `dataset.py` | Synthetic moving-shape clips, preprocessing, augmentation, train/val/test split |
| `model.py` | `VideoDenoiser`, a small 3D-conv network with a timestep embedding (run `python model.py` for a shape and gradient check) |
| `diffusion.py` | Linear and cosine schedules, forward noising, loss, DDPM and DDIM sampling |
| `train.py` | Training, EMA, evaluation, logs, loss graph, generated clips, `summary.json` |
| `run_experiments.py` | Runs the three experiments below and writes `results/experiments.md` and `results/results.json` |
| `runs/<name>/` | `config.json`, `train_log.csv`, `loss.png`, `generated.gif`, `real_test.gif`, `summary.json` |
| `results/` | Experiment table (`experiments.md`) and the same numbers as JSON |
| `../../notebooks/task1_video_ddpm.ipynb` | Colab notebook that runs everything and commits the results |

## Run

```bash
pip install -r requirements.txt
python model.py                                   # sanity check: output shape equals input shape, gradients are finite
python run_experiments.py --epochs 30             # linear vs cosine, then a learning-rate change (about 7 min on a T4)
python train.py --schedule cosine --lr 2e-3 --out runs/my_run   # one run
```

Use `--video_dir <folder>` to train on your own videos (mp4, avi, mov or gif). The code supports it, but it has not been run in this repo.

## Pipeline

**Data.** 1,500 synthetic clips, each a circle or a square bouncing off the walls on a dark background, with random size, speed and brightness. They are generated at 48x48 with 16 frames and split 80/10/10 into 1,200 train, 150 validation and 150 test clips (seed 0).

**Preprocessing.** Grayscale, uniform temporal sampling to 8 frames, centre crop to a square, resize to 32x32, normalise to [-1, 1]. Each clip becomes a tensor of shape (1, 8, 32, 32). Training clips get random horizontal flips and random time reversal.

**Forward noising.** For a clean clip x0, timestep t and noise eps ~ N(0, I):

x_t = sqrt(ab_t) · x0 + sqrt(1 − ab_t) · eps, where ab_t is the cumulative product of (1 − beta).

**Schedules (T = 200 steps).**
- Linear: beta from 1e-4·s to 0.02·s with s = 1000 / T, so the total noise matches the usual 1,000-step setting.
- Cosine (Nichol and Dhariwal): ab follows a squared cosine with offset 0.008, beta clamped to [1e-5, 0.999].

**Denoiser (`VideoDenoiser`, 0.76 M parameters, 2.9 MB in fp32).** An input 3D convolution, a residual block, one spatial downsampling (time is kept), two residual blocks, nearest upsampling, a skip connection, and an output convolution. Each residual block uses GroupNorm, SiLU and 3x3x3 convolutions, with the timestep embedding (sinusoidal, then a two-layer MLP) added after the first convolution. The output has the same shape as the input and predicts the noise eps.

**Loss.** L = mean( (eps_theta(x_t, t) − eps)² ), with t drawn uniformly from 0..T−1.

**Reverse sampling.**
- DDPM: the full 200-step ancestral sampler. Each step predicts x0, clamps it to [-1, 1], and takes the posterior mean plus noise.
- DDIM (eta = 0): 50 steps. This is what the reported clips use.

**Training setup.** 30 epochs, batch 32, AdamW (weight decay 1e-4), gradient clipping at 1.0, fp16 autocast, EMA of the weights with decay 0.99 (the EMA weights are used for validation, test and sampling), seed 0.

## Experiments

Three runs, changing one setting at a time (everything else as above).

| Run | Schedule | LR | Test eps-MSE | Motion of generated clips | Distance to nearest training clip | Train time (30 epochs) | Sampling time (8 clips) | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|
| `linear_lr5e-4` | linear | 5e-4 | 0.00919 | 0.352 | 72.5 | 136 s | 1.49 s | 953 MB |
| `cosine_lr5e-4` | cosine | 5e-4 | 0.01451 | 0.266 | 75.3 | 135 s | 1.46 s | 953 MB |
| `cosine_lr2e-3` | cosine | 2e-3 | 0.01436 | 0.264 | 79.5 | 134 s | 1.45 s | 953 MB |
| Real test clips (reference) | | | | 0.122 | 34.7 | | | |

**Metrics.**
- *Test eps-MSE:* the loss above on the 150 test clips, with the EMA model.
- *Motion:* mean absolute difference between consecutive frames on the [-1, 1] scale, over 8 clips. Real clips have 0.122.
- *Distance to nearest training clip:* mean L2 distance from each of 8 clips to the closest of the first 300 training clips. For real unseen test clips it is 34.7.

Training curves are in `runs/<name>/loss.png`. In all three runs the validation loss falls steeply in the first 5 epochs and then keeps improving slowly: for example, linear falls from 0.0152 at epoch 10 to 0.0084 at epoch 29, and the best validation loss is at the last epoch in every run. So training had not converged after 30 epochs, and a longer run might help. Validation loss stays below training loss (the validation model uses the EMA weights), so there is no sign of overfitting.

### How the hyperparameter changes affect the result

**Linear to cosine schedule (learning rate fixed at 5e-4).**
- Motion of the generated clips falls from 0.352 to 0.266, closer to the real value of 0.122.
- In the clips I inspected, the linear run leaves grainy residual noise in the background, and the cosine runs are cleaner.
- Test eps-MSE rises from 0.0092 to 0.0145. **This is not a quality ranking:** the two schedules put different amounts of noise at each timestep, so an average loss over timesteps measures different tasks. Compare the generated clips, not this number.
- The usual argument for the cosine schedule is that it destroys information more slowly, so more steps are spent at moderate noise levels. This was not tested here.

**Learning rate 5e-4 to 2e-3 (cosine).**
- Test loss (0.0145 vs 0.0144) and motion (0.266 vs 0.264) are almost unchanged.
- In the clips I inspected, the 2e-3 run has the darkest, cleanest background and more compact shapes. The distance to the nearest training clip rises from 75 to 80.
- With one seed and 8 clips, differences this small should not be treated as a real effect.

## Failure cases

I inspected 3 of the 8 generated clips for each run next to real clips (a visual reading, not a measurement). All three runs fail in the same ways:

1. **Irregular blobs instead of circles and squares.** Shapes have ragged outlines and sometimes merge or split.
2. **Shapes appear and disappear.** In several clips the first frames are almost empty and a shape then appears, or a shape vanishes. Real clips always contain one smooth object.
3. **Jumpy motion.** Positions change abruptly between frames. This matches the motion metric: generated clips change 2 to 3 times more between frames (0.26 to 0.35) than real ones (0.122).
4. **Residual background noise.** Strongest with the linear schedule, weaker with cosine. The background is a dark grey in the cosine runs, not pure black.
5. **Far from the data distribution.** Generated clips lie about twice as far from the training clips (72 to 80) as real unseen clips do (34.7). This does not look like copying the training set. It is consistent with the visible quality problems, and the metric cannot tell novelty from poor samples.

Likely reasons (not tested): a very small network (0.76 M parameters, one downsampling), short training (30 epochs on 1,200 clips), and only 8 frames at 32x32.

## What the submission contains (slide 15 checklist)

| Required | Status |
|---|---|
| Runnable source code and setup README | Yes (this file) |
| Configuration and training logs | Yes: `runs/<name>/config.json`, `train_log.csv` |
| Generated clips and loss graphs | Yes: `generated.gif`, `real_test.gif`, `loss.png` |
| Experiment table with results and limitations | Yes: `results/experiments.md`, and the sections above |
| Model checkpoint | **Not in the repository.** `*.pt` is git-ignored, so `ckpt.pt` and `best.pt` from the Colab runs were never committed. Re-run and add an exception to `.gitignore` to commit them. |

## Limitations

- One seed, and 8 generated clips per run for the motion and distance metrics. Differences between the cosine runs are within what noise could produce.
- The data is simple (one bouncing circle or square, grayscale, 32x32, 8 frames). Results do not transfer to natural video.
- No learned or standard video metric (such as FVD) was computed. The motion and distance metrics are simple proxies.
- Only DDIM with 50 steps is reported. The 200-step DDPM sampler is implemented but its output was not evaluated here.
- No ablation on the number of timesteps, network width, EMA decay or training length.
