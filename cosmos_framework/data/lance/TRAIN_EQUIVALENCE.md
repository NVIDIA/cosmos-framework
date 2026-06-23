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

## Multi-GPU per-epoch time (does the loader speedup => faster training?)
4× L40S, DDP, LoRA-SFT Qwen2.5-VL-3B, 600 samples/epoch, `train_multigpu_time.py`
(epoch 0 = warmup, discounted):

| loader | steady per-epoch | data-wait | loss (ep2) |
| ------ | ---------------- | --------- | ---------- |
| base   | 29.7 s | 1.1% | 0.023 |
| lance  | 30.8 s | 1.1% | 0.023 |

**Compute-bound, not data-bound.** Data-wait is 1.1% — the GPUs spend ~99% of the epoch on
forward/backward and the base loader already keeps them fed via prefetch, so per-epoch time is
equal (the 3% is noise) and loss is identical. The *ceiling* on any loader speedup here is the
1.1% data-wait. The dataloader throughput wins (VLM 22× raw access, video 2.5–6.5×) reduce
wall-clock **only in the data-bound regime** (GPUs starving for data) — lighter models, many
more GPUs per CPU, or slow object-store I/O. On a single node with a heavy model, Lance's value
is freed CPU + storage/scalability + filtered reads + train-equivalence, not single-node wall-clock.
