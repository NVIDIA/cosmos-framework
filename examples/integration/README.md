# Cosmos3 Integration Demos

Minimal worked examples for **taking Cosmos3 into your own training / inference
framework**. Each demo is self-contained (one Python file) and runs end-to-end
on a single 80 GB GPU.

> **These demos use RANDOM main-transformer weights.** They do not load the
> ~30 GB Cosmos3-Nano DCP shards — only `config.json` is fetched. Losses,
> pixels, and samples are therefore *not meaningful*; the point is to show the
> API call sequence and tensor shapes so you can wire OmniMoTModel into your
> own code. For real weight loading see
> [`cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp`](../../cosmos_framework/inference/model.py)
> and the production CLIs in `cosmos_framework.scripts.{inference,train}`.
>
> This directory is **integration docs by example**, not a model zoo. It does
> not introduce any new training recipe — every file shows how to call code
> that already exists in `cosmos_framework/` from a *plain* PyTorch loop.

## Modality coverage

All three demos cover all four generation modes that Cosmos3-Nano supports:

|                              | T2I (image) | T2V (video) | ACTION_FDM | T2VS (sound+video) |
| ---------------------------- | :---------: | :---------: | :--------: | :----------------: |
| `trainer_level_inference.py` |     ✅      |     ✅      |    ✅¹     |        ✅¹         |
| `trainer_level_training.py`  |     ✅      |     ✅      |     ✅     |         ✅         |
| `net_level.py` train         |     ✅      |     ✅      |     ✅     |         ✅         |
| `net_level.py` sample        |     ✅      |     ✅      |    ✅¹     |        ✅¹         |

¹ For ACTION_FDM and T2VS, the demos feed the model **random** conditioning
  (video / actions / audio waveforms). The call sequence runs end-to-end —
  loss + backward in training, sampler + decode in inference — but the
  *output* is visual / audio noise. The wiring is what's being demonstrated.
  For meaningful samples, swap in real conditioning data via your loader.

---

## 1. Pick the right demo

Two integration levels, four cases:

|                 | **Trainer-level**                                         | **Net-level**                       |
| --------------- | --------------------------------------------------------- | ----------------------------------- |
| Module used     | `OmniMoTModel`                                            | `model.net` (= `Cosmos3VFMNetwork`) |
| Entry call(s)   | `training_step` / `generate_samples_from_batch`           | `net.forward(packed_seq, ...)`      |
| Loss + sampler  | written by cosmos_framework (rectified-flow, UniPC)       | written by **you** in the demo      |
| Effort to adopt | Lowest                                                    | Higher (you control loss & sampler) |
| File            | `trainer_level_inference.py`, `trainer_level_training.py` | `net_level.py`                      |

```
┌────────────────────────────────────────────────────────────────────────────┐
│ OmniMoTModel        ◀── Cases 1 & 2 plug in here (high-level integration)  │
│   ├── training_step(batch, iter)              → (aux, loss)                │
│   ├── generate_samples_from_batch(batch)      → {"vision": [...]}          │
│   ├── encode / decode                          (VAE)                       │
│   ├── _pack_input_sequence(...)                (PackedSequence builder)    │
│   │                                                                         │
│   └── net = Cosmos3VFMNetwork ◀── Cases 3 & 4 plug in here (low-level)      │
│             forward(packed_seq, fps_vision=...) → {"preds_vision": [...]}   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Decision matrix

| If you want to…                                                              | Use                          |
| ---------------------------------------------------------------------------- | ---------------------------- |
| Drop Cosmos3 into your training framework with minimum work                  | `trainer_level_training.py`  |
| Drop Cosmos3 into your serving / batch-inference framework with minimum work | `trainer_level_inference.py` |
| Write a custom diffusion loss / curriculum / RL objective                    | `net_level.py` (train)       |
| Write a custom sampler / guidance / consistency scheme                       | `net_level.py` (infer)       |

---

## 2. The four cases

### Case 1 — `trainer_level_inference.py` (trainer-level inference)

What you replace from cosmos_framework: the `OmniInference` pipeline, Ray serving,
the CLI entry-point in `cosmos_framework.scripts.inference`. You keep `OmniMoTModel`
and its built-in CFG + UniPC/EDM sampler.

Has a `--mode {t2i,t2v,action_fdm,t2vs}` flag. T2I/T2V batches come from
cosmos_framework's `get_sample_data` helper; action_fdm and t2vs are hand-built with
random conditioning. The model call is identical for all modes:

```python
model  = Cosmos3OmniModel.from_pretrained_dcp(ckpt_dir).model     # OmniMoTModel
batch  = build_t2iv_batch(model, ..., num_frames)                  # or build_action_fdm_batch / build_t2vs_batch
out    = model.generate_samples_from_batch(batch, seed=[0])        # ← THE call
pixels = model.decode(out["vision"][0])                            # VAE decode
# T2VS only — sound output:
# waveform = model.decode_sound(out["sound"][0])
```

### Case 2 — `trainer_level_training.py` (trainer-level training)

What you replace: `cosmos_framework.scripts.train`, the `Trainer` class, callbacks,
FSDP wiring, dataloaders. You keep `model.training_step`, which packages
flow-matching loss + sampling + packing.

```python
model = Cosmos3OmniModel.from_pretrained_dcp(ckpt_dir).model
opt   = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)

for it, batch in enumerate(my_loader):           # ← your dataloader
    aux, loss = model.training_step(batch, iteration=it)
    loss.backward()
    opt.step(); opt.zero_grad()
```

The demo round-robins through 4 batch builders so you can read the exact
`data_batch` shape `training_step` expects for every modality:

| Helper                     | Modality   | Key fields                                                                              |
| -------------------------- | ---------- | --------------------------------------------------------------------------------------- |
| `make_text_to_image_batch` | T2I        | `images`, `text_token_ids`, `image_size`, `fps`                                         |
| `make_text_to_video_batch` | T2V        | `video`, `text_token_ids`, `image_size`, `fps`, `num_frames`                            |
| `make_action_fdm_batch`    | Action FDM | + `action`, `domain_id`, `raw_action_dim`, `mode`, `sequence_plan`                      |
| `make_sound_video_batch`   | T2VS       | + `sound` (stereo @ 48 kHz, multiple of AVAE hop=1920), `sequence_plan(has_sound=True)` |

> **⚠ Gotcha — video shape differs between training and inference batches.**
> Training (`training_step`, `is_preprocessed=True`) expects a **flat** list:
> `batch[model.input_video_key] = [video]` → `[1, C, T, H, W]`.
> Inference (`cosmos_framework.inference.action.build_action_batch`) uses **nested**:
> `batch[model.input_video_key] = [[video]]` (one extra `[]`).
> Copying an inference batch into a training loop fails inside
> `_normalize_video_databatch_inplace` with an opaque error — use the flat
> convention when calling `training_step`.

### Case 3 — `net_level.py` (net-level inference)

What you replace: everything in case 1 *plus* the cosmos_framework sampler (UniPC/EDM).
You write the sampling loop by hand and call `net.forward` per step.

`sample(model, net, batch)` is generic across modalities — it splits the
final flat trajectory back into vision/action/sound chunks using the same
offset layout as `_get_velocity`, and decodes each:

```python
net = model.net                                                     # Cosmos3VFMNetwork
seq_plans, gen_clean, cond_tokens, _, xt = model._prepare_inference_data(batch, seed=[0])

for step in range(num_steps):                                       # ← Your sampling loop
    t = 1.0 - step / num_steps
    v = model._get_velocity(net=net, noise_x=xt, timestep=..., text_tokens=cond_tokens, ...)
    xt = [x + dt * v_i for x, v_i in zip(xt, v)]

# Per-modality reshape + decode (offsets mirror _get_velocity's split)
vision_latent = xt[0][:vision_dim].reshape(gen_clean.x0_tokens_vision[0].shape)
pixels        = model.decode(vision_latent)                         # always
# action:  xt[0][vision_dim:vision_dim+action_dim].reshape(...)     # if has_action
# sound :  model.decode_sound(xt[0][...sound_slice].reshape(...))   # if has_sound
```

`sample()` returns `{"pixels", "action"?, "sound_waveform"?}`. Plain Euler,
no CFG — production cosmos_framework uses UniPC + CFG; only the integrator differs.

### Case 4 — `net_level.py` (net-level training)

What you replace: everything in case 2 *plus* the flow-matching loss and
the noise schedule. You write the loss explicitly. Same per-modality batch
builders as Case 2 (T2I / T2V / ACTION_FDM / T2VS) round-robin into one
`train_one_step` that calls `net.forward` directly.

```python
net = model.net

# Build the input contract using cosmos_framework helpers
gen_clean    = model.get_data_and_condition(batch, iteration=it)
text_indexes = model._load_and_tokenize_text_data(batch, iteration=it)
seq_plans    = build_sequence_plans_from_data_batch(batch, model.input_video_key, model.input_image_key)
sigmas       = sample_my_sigmas(gen_clean.batch_size)               # ← your noise schedule
packed_seq   = model._pack_input_sequence(seq_plans, text_indexes, gen_clean, (sigmas*1000).cpu())
gen_noised   = model._add_noise_to_input(gen_clean, packed_seq, sigmas, iteration=it)
model._replace_clean_with_noised(packed_seq, gen_noised); packed_seq.to_cuda()

# The bare-net forward — this is the one line that survives a port
out = net(packed_seq, fps_vision=gen_clean.fps_vision)              # ← Your forward call

# Your loss — here flow-matching MSE, but it can be anything
v_pred, v_target = out["preds_vision"], gen_noised.vt_target_vision
loss = sum(F.mse_loss(p.float(), t.float()) for p, t in zip(v_pred, v_target))
loss.backward()                                                      # ← Your code
```

---

## 3. What you "extract" at each level

A pure level-A extraction (zero `import cosmos_framework`) is **not feasible without
re-vendoring** — `Cosmos3VFMNetwork.forward` takes a `PackedSequence`, which
~2400 lines of `cosmos_framework/data/vfm/sequence_packing.py` build. These demos show
the realistic options:

| Cosmos surface you keep                                     | Trainer-level |     Net-level     |
| ----------------------------------------------------------- | :-----------: | :---------------: |
| `Cosmos3OmniModel.from_pretrained_dcp` (loader)             |      ✅       |        ✅         |
| VAE (`model.encode` / `model.decode`)                       |      ✅       |        ✅         |
| Text tokenizer (`model.vlm_tokenizer` + `tokenize_caption`) |      ✅       |        ✅         |
| Sequence packer (`model._pack_input_sequence`)              |      ✅       |        ✅         |
| Noise scheduler (`model._add_noise_to_input`)               |      ✅       |  ❌ (your sigma)  |
| Flow-matching loss (`model._compute_losses`)                |      ✅       |  ❌ (your loss)   |
| Sampler (`UniPC` / `EDM` in `model.sampler`)                |      ✅       | ❌ (your sampler) |
| Trainer / callbacks / FSDP / dataloader                     |      ❌       |        ❌         |

The "❌" cells are exactly what you replace in net-level integration.

> **Note on underscore-prefixed methods.** Net-level integration depends on
> several `_method` names on `OmniMoTModel` — `_pack_input_sequence`,
> `_load_and_tokenize_text_data`, `_add_noise_to_input`,
> `_replace_clean_with_noised`, `_prepare_inference_data`, `_get_velocity`.
> The underscore is Python convention for "internal," but **these are the
> intended net-level integration surface today** and are exercised by the
> demos in CI. Treat them as stable for integration purposes; if cosmos_framework
> ever promotes them to public names, the demos will be updated.

---

## 4. Running the demos

### Prerequisites

1. **Install cosmos_framework** as a library (`pip install -e .` from the repo root,
   or activate the project's `.venv`).
2. **A single ≥ 80 GB GPU.** For training, the demos use SGD (zero optimizer
   state); switching to AdamW for the full 8 B model OOMs on one 80 GB GPU.
3. **HF cache access** for the auxiliary sub-models — Qwen3-VL tokenizer,
   Wan2.2 VAE, AVAE — and the Cosmos3-Nano `config.json` (single ~5 KB file).
   The main ~30 GB transformer DCP is **not** downloaded; the demos run with
   random main-transformer weights.

### Common flags

```bash
PYTHONPATH=. python examples/integration/<demo>.py                                # fetches config.json
PYTHONPATH=. python examples/integration/<demo>.py --config-dir /path/with/config.json  # local config
```

### Verified runs (single H100 80 GB)

All four modalities run end-to-end in every demo. Output shapes are
deterministic (driven by the config + input shape), but **pixel / sound /
loss values are not meaningful** because the main transformer is random:

| Demo / mode                                    | Output shape (verified)                           |
| ---------------------------------------------- | ------------------------------------------------- |
| `trainer_level_inference.py --mode t2i`        | `pixels [3, 1, 128, 128]`                         |
| `trainer_level_inference.py --mode t2v`        | `pixels [3, 33, 128, 128]`                        |
| `trainer_level_inference.py --mode action_fdm` | `pixels [3, 5, 128, 128]`                         |
| `trainer_level_inference.py --mode t2vs`       | `pixels [3, 5, 128, 128]` + `sound [2, 15360]`    |
| `trainer_level_training.py --num-iters 4`      | 4 iters round-robin T2I / T2V / ACTION_FDM / T2VS |
| `net_level.py --sample-mode t2i`               | `pixels [3, 1, 128, 128]`                         |
| `net_level.py --sample-mode t2v`               | `pixels [3, 17, 128, 128]`                        |
| `net_level.py --sample-mode action_fdm`        | `pixels [3, 5, 128, 128]` + `action [4, 64]`      |
| `net_level.py --sample-mode t2vs`              | `pixels [3, 5, 128, 128]` + `sound [2, 15360]`    |

> **Why t2v differs:** `trainer_level_inference.py` defaults to `--num-frames 33`
> (matches cosmos_framework's default sample args), while `net_level.py` defaults to
> 17 frames inside `make_text_to_video_batch` to keep the net-level demo
> fast. Same model, same code path — only the batch's `num_frames` differs.

```bash
# Point HF_HOME at a writable cache (any path); aux sub-models + the
# Cosmos3-Nano config.json auto-download into $HF_HOME/hub/... on first use.
export HF_HOME=$HOME/cosmos_assets/hf_cache

# Case 1 — trainer-level inference (default: t2i)
PYTHONPATH=. .venv/bin/python examples/integration/trainer_level_inference.py
# Other modes:
#   --mode t2v        --num-frames 33
#   --mode action_fdm
#   --mode t2vs

# Case 2 — trainer-level training, round-robins through all 4 modalities
PYTHONPATH=. .venv/bin/python examples/integration/trainer_level_training.py \
    --num-iters 4

# Cases 3 + 4 — net-level training + Euler sampling for a chosen mode
PYTHONPATH=. .venv/bin/python examples/integration/net_level.py \
    --num-train-iters 4 --num-sample-steps 8 \
    --sample-mode t2i        # or t2v / action_fdm / t2vs
```

To run against a non-default config (e.g. Cosmos3-Super) point `--config-dir`
at a directory containing that model's `config.json`.

---

## 5. Where to look next in the cosmos_framework source

| Topic                           | File                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------ |
| OmniMoTModel definition         | `cosmos_framework/model/vfm/omni_mot_model.py`                                 |
| Cosmos3VFMNetwork (`model.net`) | `cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py`                        |
| PackedSequence + packer         | `cosmos_framework/data/vfm/sequence_packing.py`                                |
| Rectified-flow loss             | `cosmos_framework/model/vfm/algorithm/loss/flow_matching.py`                   |
| UniPC / EDM samplers            | `cosmos_framework/model/vfm/diffusion/samplers/`                               |
| Checkpoint loader               | `cosmos_framework/inference/model.py` (`Cosmos3OmniModel.from_pretrained_dcp`) |
| Default sample args             | `cosmos_framework/inference/defaults/<mode>/sample_args.json`                  |
| FSDP / parallelism wrapping     | `cosmos_framework/utils/vfm/parallelism.py` (`ParallelDims`)                   |
| Production trainer (skipped)    | `cosmos_framework/scripts/train.py`, `examples/toml/*.toml`                    |
