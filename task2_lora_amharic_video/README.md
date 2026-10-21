# Task 2: LoRA fine-tuning of Wan2.1-T2V-1.3B for Amharic (Ge'ez) text in video

Resource-optimised LoRA fine-tuning of a pretrained text-to-video diffusion model on a small synthetic Amharic video set, sized for a free Colab T4 (15 GB). The notebook `../notebooks/task2_lora_amharic_video.ipynb` runs everything end to end.

**Status:** one training run (rank 16, 400 steps) completed, with a before/after comparison against the base model. The qualitative result is promising but only partly verified; see "What is not verified".

## Files

| Path | Content |
|---|---|
| `make_amharic_data.py` | Renders animated Amharic word clips and a `metadata.jsonl` of captions |
| `train_lora.py` | LoRA training with flow matching and the memory optimisations below |
| `infer.py` | Generates videos with the base model and with base + LoRA (same seeds) |
| `common.py` | Model id, text encoding, video reading, shared helpers |
| `outputs/lora_r16/` | `pytorch_lora_weights.safetensors`, `train_log.csv`, `loss.png` |
| `outputs/samples/` | `base_0.mp4`, `lora_0.mp4`, `base_1.mp4`, `lora_1.mp4` |
| `outputs/comparison.png` | Frame 8 of each of the four clips |
| `coffee_lora/`, `coffee_ad/` | Separate coffee advertisement pipeline (not part of this experiment) |

## Run

```bash
pip install -r requirements.txt && pip uninstall -y torchao
python make_amharic_data.py --n 40
python train_lora.py --steps 400 --accum 2 --out outputs/lora_r16
python infer.py --lora_dir outputs/lora_r16 --out outputs/samples
```

On Colab, `torchao` must be removed because it breaks `peft`. The notebook also changes one import in `train_lora.py` (`get_peft_model_state_dict` from `peft.utils`).

## Method

**Model.** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. Its umt5 text encoder is multilingual, so Amharic prompts are encoded without changes.

**Data.** 40 synthetic clips of 17 frames at 256x384. Each clip shows one of 10 Amharic words (for example ሰላም, ቡና) rendered in a Ge'ez font, moving in one of four ways (slides, zooms in, fades in, rises) over one of four coloured backgrounds. Each caption has an English description with the Amharic word in quotes, followed by an Amharic phrase.

**LoRA.** Rank 16, alpha 16 (scale 1), 11.80 M trainable parameters, fp32 adapter weights on a frozen fp16 transformer.

**Objective (flow matching).** For a clean latent x0, noise n and timestep s: x_s = (1 - s)·x0 + s·n. The network predicts the velocity n − x0, and the loss is the mean squared error between prediction and target. The timestep is drawn from a logit-normal distribution and warped with shift 3.0, the same shift as the inference scheduler.

**Memory optimisations (these are what make it fit on a T4).**
1. Text embeddings and VAE latents are computed once and cached, then the 11 GB text encoder and the VAE are freed.
2. Only the LoRA weights are trained; the base transformer is frozen.
3. Gradient checkpointing, fp16 autocast with a gradient scaler, batch size 1 with gradient accumulation 2.
4. Low resolution and 17 frames.

**Optimisation.** AdamW (lr 1e-4, weight decay 1e-2), cosine learning-rate decay to zero, gradient clipping at 1.0, seed 0. Each step draws 2 random clips (with replacement) and 2 random timesteps, so 400 steps are 800 forward and backward passes, about 20 passes per clip on average.

**Inference.** 25 sampling steps, guidance scale 5.0, 17 frames at 256x384, the same seed for base and LoRA. Two prompts:
1. The Amharic word "ሰላም" (peace) in white Ge'ez script slides from left to right on a blue background.
2. The Amharic word "ቡና" (coffee) in white Ge'ez script zooms in on a black background.

Both words are in the training set.

## Results (measured on a free Colab T4)

| Quantity | Value |
|---|---|
| Training time, 400 steps | 1,389 s |
| Peak GPU memory, training | 3.3 GB |
| Generation time per clip, base / LoRA | 31-32 s / 36-37 s |
| Peak GPU memory, inference | 10.7 GB |
| Trainable parameters | 11.80 M |

**Loss.** The raw loss is very noisy (about 0.005 to 0.23, with isolated spikes), because each step averages only two random samples at random noise levels. The 20-step moving average falls from about 0.045 near the start to about 0.025 near the end, with fluctuation. The loss alone does not show how good the videos are; the comparison below does.

**Base versus LoRA, frame 8 of each clip (`outputs/comparison.png`).**
- Base model: jumbled white shapes that do not look like Ge'ez script. On the second prompt some forms look Latin.
- LoRA: clean, evenly stroked letterforms in the style of Ge'ez script, on backgrounds closer to the training data (a darker blue on the first prompt, in place of the base model's bright cyan). The background colour follows the prompt (blue, then black).

**Motion, measured on all 17 frames of each clip** (white-pixel mask: horizontal centre and width of the text):

| Clip | Prompt asks for | What the text does | Measured |
|---|---|---|---|
| `base_0`, `base_1` | slide / zoom | nothing (static) | centre moves at most 2 px, width constant |
| `lora_0` (ሰላም) | slides from left to right | grows, does not slide | centre 185 to 183 px; width 122 to 146 px (+20%), height 43 to 55 px (+28%) |
| `lora_1` (ቡና) | zooms in | slides, does not grow | centre 207 to 147 px (right to left); width constant at about 162 px |

So the LoRA learned that the text moves, but the two motions are swapped between the prompts, and the slide goes right to left, whereas training slides left to right. Training zoom enlarges the text by a factor of 1.83; the growth in `lora_0` is much weaker. With two prompts and one seed this cannot say why. Possible causes (untested): too little training to tie the motion words to the motion, and only about 10 training clips per motion.

## What is not verified

- **Correct spelling.** Whether the LoRA output spells ሰላም and ቡና is not confirmed. In the inspected frames the number of glyphs looks higher than the number of syllables of each word.
- **Motion.** Only one frame per clip was inspected. Whether "slides from left to right" and "zooms in" are followed has not been checked.
- **Generalisation.** Both test prompts use training words, so the result shows that the style was learned, not that unseen words are spelled correctly. A held-out-word test has not been run.
- **Variance.** One seed, one run. No rank, step or data-size comparison exists. A rank-32 run with 120 clips was started and did not run (wrong working directory), so there is no result for it.


## Limitations

- Training data is synthetic: rendered text in a single font on flat backgrounds, with 40 clips. Real footage was not used here.
- Resolution (256x384) and length (17 frames, about 2 s at 8 fps) are far below what Wan2.1 was built for, because of the T4.
- fp16 was used because the T4 has no native bf16. Wan is usually run in bf16, so overflow is possible. None was seen in this run (no NaN loss, no black frames in the inspected frames).
- The adapter was trained with alpha equal to rank. If you change `--rank`, set `--alpha` to the same value, because the saved adapter stores no alpha and inference assumes scale 1.

## Next steps

1. Generate held-out words and word-colour-motion combinations that are not in the training set.
2. If time allows, repeat with one change at a time (rank 32, more clips, more steps) and a second seed.