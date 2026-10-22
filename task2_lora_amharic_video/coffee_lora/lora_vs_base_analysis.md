# LoRA vs Base: Video Metrics

Ethiopian coffee advertisement (Wan2.1-1.3B). 6 base clips (`base_0` to `base_5`) vs 6 LoRA clips (`lora_0` to `lora_5`).

## 1. What each metric measures

Definitions below are the standard formulations. The metric script was not available, so the exact formulas are assumed. All values are averaged over each clip.

| Metric | What it measures | Scale | Higher means |
|---|---|---|---|
| **motion** | Mean absolute pixel difference between consecutive frames. Quantifies how much the picture changes over time. | 0-1 (pixels normalized) | more movement; near 0 is a still image |
| **flicker** | Frame-to-frame fluctuation in overall brightness or texture that is not smooth motion. Quantifies temporal instability. | 0-1 | more jitter or pulsing |
| **sharp** | Variance of the Laplacian of each frame (edge/detail strength). | unbounded, depends on resolution and normalization | more edge detail (noise also raises it) |
| **bright** | Mean pixel luminance of each frame. | 0 (black) to 1 (white) | brighter image |

## 2. Results

| file | motion | flicker | sharp | bright |
|---|---|---|---|---|
| base_0 | 0.0907 | 0.0764 | 0.00061 | 0.227 |
| base_1 | 0.0138 | 0.0021 | 0.00186 | 0.390 |
| base_2 | 0.0314 | 0.0336 | 0.00270 | 0.178 |
| base_3 | 0.0005 | 0.0003 | 0.00533 | 0.167 |
| base_4 | 0.0008 | 0.0004 | 0.00860 | 0.467 |
| base_5 | 0.0017 | 0.0005 | 0.00208 | 0.170 |
| lora_0 | 0.0384 | 0.0033 | 0.00202 | 0.150 |
| lora_1 | 0.0133 | 0.0008 | 0.00132 | 0.318 |
| lora_2 | 0.0038 | 0.0009 | 0.00391 | 0.096 |
| lora_3 | 0.0007 | 0.0004 | 0.00823 | 0.166 |
| lora_4 | 0.0006 | 0.0004 | 0.00745 | 0.321 |
| lora_5 | 0.0012 | 0.0003 | 0.01307 | 0.172 |

## 3. Means

| Metric | Base mean | LoRA mean | Change |
|---|---|---|---|
| motion | 0.0232 | 0.0097 | -58% |
| flicker | 0.0189 | 0.0010 | -95% |
| sharp | 0.0035 | 0.0060 | +70% |
| bright | 0.267 | 0.204 | -24% |

## 4. Per-clip change (LoRA vs base)

| Clip | motion | flicker | sharp | bright |
|---|---|---|---|---|
| 0 | 0.0907 to 0.0384 (down) | 0.0764 to 0.0033 (down) | 0.00061 to 0.00202 (up) | 0.227 to 0.150 (down) |
| 1 | 0.0138 to 0.0133 (same) | 0.0021 to 0.0008 (down) | 0.00186 to 0.00132 (down) | 0.390 to 0.318 (down) |
| 2 | 0.0314 to 0.0038 (down) | 0.0336 to 0.0009 (down) | 0.00270 to 0.00391 (up) | 0.178 to 0.096 (down) |
| 3 | 0.0005 to 0.0007 (same) | 0.0003 to 0.0004 (same) | 0.00533 to 0.00823 (up) | 0.167 to 0.166 (same) |
| 4 | 0.0008 to 0.0006 (same) | 0.0004 to 0.0004 (same) | 0.00860 to 0.00745 (down) | 0.467 to 0.321 (down) |
| 5 | 0.0017 to 0.0012 (same) | 0.0005 to 0.0003 (same) | 0.00208 to 0.01307 (up) | 0.170 to 0.172 (same) |

## 5. Flicker relative to motion

| Clip | Base flicker/motion | LoRA flicker/motion |
|---|---|---|
| 0 | 0.84 | 0.09 |
| 1 | 0.15 | 0.06 |
| 2 | 1.07 | 0.24 |
| 3 | 0.60 | 0.57 |
| 4 | 0.50 | 0.67 |
| 5 | 0.29 | 0.25 |

## 6. Result

- **Flicker:** reduced by 95% on average, from 0.0189 to 0.0010. The reduction is concentrated in clips 0 and 2, and in those clips flicker relative to motion also fell (0.84 to 0.09 and 1.07 to 0.24).
- **Motion:** reduced by 58% on average, from 0.0232 to 0.0097. Clips 0 and 2 dropped; clip 2 fell from 0.0314 to 0.0038. Clips 3, 4 and 5 are near-static (below 0.002) in both versions.
- **Sharpness:** increased by 70% on average, from 0.0035 to 0.0060. It rose in clips 0, 2, 3 and 5 and fell in clips 1 and 4.
- **Brightness:** decreased by 24% on average, from 0.267 to 0.204. It fell in clips 0, 1, 2 and 4 and is unchanged in clips 3 and 5. lora_2 is the darkest at 0.096.
