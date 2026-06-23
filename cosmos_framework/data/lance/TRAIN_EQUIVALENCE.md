# Train-equivalence: Lance loader vs base loader produce the same training

The strongest correctness proof: train the *same* model with the base loader and with the
LanceDB loader and compare the loss/eval curves. Script: `benchmarks/lance/train_compare_vlm.py`.

Setup: LoRA-SFT of `Qwen/Qwen2.5-VL-3B-Instruct` (the model the cosmos VLM recipe fine-tunes),
batch size 1, 60 steps. Same model init, same LoRA seed, same sample order, same LR — the
ONLY difference is which loader produces each sample:
- `base`  = HF dataset records,
- `lance` = `LanceVLMDataset` records (token-exact + lossless-PNG-exact),
- `base2` = the base loader a SECOND time (control = the GPU/bf16 nondeterminism floor).

## Result (Qwen2.5-VL-3B, 60 steps, lr 1e-4)
| metric | value |
| ------ | ----- |
| step-0 \|base − lance\| loss | **0.00e+00** (identical inputs) |
| mean \|Δ\| base vs lance | **3.5e-03** |
| mean \|Δ\| base vs base2 (control) | 5.3e-03  (nondeterminism floor) |
| final train loss | base 0.3554 · base2 0.3535 · lance 0.3538 |
| held-out eval loss (20 samples) | base 0.3340 · base2 0.3324 · lance 0.3331 |

**Conclusion:** the base↔lance difference (3.5e-3) is *smaller* than base↔base2 (5.3e-3) —
swapping in the Lance loader perturbs training less than simply re-running the base loader on
the same GPU. The loss curves overlay (3.0 → 0.35) and eval losses match within
nondeterminism. The Lance VLM loader is a true, training-equivalent drop-in.

Note: this uses the token-exact + image-exact VLM loader (so equivalence should be near-perfect,
which it is). The action/vision-SFT loaders re-encode video lossily (~32–37 dB); their model is
the 16B Cosmos3-Nano diffusion stack — a separate, heavier run not done here.
