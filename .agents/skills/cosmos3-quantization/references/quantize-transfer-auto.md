<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Quantize Cosmos3-Nano-Transfer-Auto to FP8

This recipe makes a static per-tensor FP8 (E4M3) checkpoint of `nvidia/Cosmos3-Nano-Transfer-Auto`.
It lives in `cosmos_framework/quantization/transfer_auto.py`. It does not use `quantize_fp8_checkpoint`.
That function runs its own single-view denoise loop. Transfer-Auto needs multiview inputs, WSM control
videos and LiDAR, so the recipe calibrates through the native `OmniInference.generate` on request JSONL
files.

The recipe has three steps. Each step runs alone.

| Step      | Function                                  | Result                                                    |
| --------- | ----------------------------------------- | --------------------------------------------------------- |
| calibrate | `calibrate_transfer_auto`                 | JSON with the activation amax of each projection          |
| search    | `search_weight_scales` (optional)         | JSON map: weight key to weight scale                      |
| export    | `export_transfer_auto_fp8`                | FP8 checkpoint streamed from the BF16 source              |

## What is quantized

- 504 linears: q, k, v, o and gate, up, down, in both towers (understanding and generator), in each of the
  36 layers. `projection_map` lists them: 36 layers x 2 towers x 7 projections.
- Weights: E4M3 with a scalar FP32 `weight_scale = max|W| / 448`. Activations: a scalar FP32
  `input_scale = activation_amax / 448`.
- Stays in BF16 and is copied without change: norms, embeddings, `proj_in`, `proj_out`, the time embedder,
  the LiDAR input and output projections, the visual encoder, and all auxiliary models. The VAEs are
  auxiliary files; the LiDAR VAE stays FP32. Attention math (scores, softmax, KV cache) is not quantized.

## Calibration data

- Request JSONL files for the native inference CLI (`python -m cosmos_framework.scripts.inference -i ...`).
- 7 cameras for each request. Each request runs all 35 denoising steps, with both CFG branches.
- The released checkpoint used 12 Drive-Dreams sessions and one released joint RGB/LiDAR sample.
- Run `calibrate` once for each shard of requests. Join the shard files by elementwise max
  (`calibrate --merge`). The reduction is a max, not an average, so the shard order does not matter.

## MAPS (mixed activation precision across steps)

The export writes `quantization_config.runtime.diffusion_step_policy` into `transformer/config.json` with
the upstream `add_diffusion_step_policy` helper. The runtime uses A16 activations in the first 3 and the
last 3 denoising steps and in the reasoner, and A8 (native FP8) in the other steps. The weights stay FP8 in
all steps. `hf_quant_config.json` at the root does not carry the policy (the same as the other recipes).
`--mixed-precision-steps N` changes the count (default 3).

## The three commands

Setup is the one in `quantization-readme.md` (`cu130-train` group, `nvidia-modelopt[torch]==0.44.0`).
`SRC` is the BF16 checkpoint directory, for example a snapshot of `nvidia/Cosmos3-Nano-Transfer-Auto`.

```bash
# 1. Calibrate. One run for each shard of requests (GPU needed), then merge the shard files.
python -m cosmos_framework.quantization.transfer_auto calibrate \
  --checkpoint "$SRC" --requests requests/shard0.jsonl --output amax-shard0.json
python -m cosmos_framework.quantization.transfer_auto calibrate \
  --merge amax-shard0.json amax-shard1.json --output amax.json

# 2. (Optional) Weight-scale search on captured projection inputs.
python -m cosmos_framework.quantization.transfer_auto search \
  --source "$SRC" --activation-amax amax.json --captures captures/*.pt --output weight_scales.json

# 3. Export from the pristine BF16 source. Leave out --weight-scales for the plain max recipe.
python -m cosmos_framework.quantization.transfer_auto export \
  --source "$SRC" --activation-amax amax.json --weight-scales weight_scales.json --output "$OUT"
```

The same steps are available as Python functions: `calibrate_transfer_auto`, `search_weight_scales` and
`export_transfer_auto_fp8`.

### Weight-scale search

For each projection, the search tries 12 multipliers of the max-derived weight scale
(1, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.125, 1.125, 1.25, 1.5, 2). It keeps the one with the lowest
projection-output MSE on the captured activation rows. Activations are fake-quantized with their
`input_scale`. Multiplier 1 wins ties. The output holds only the projections where another multiplier
wins. The released checkpoint (`weight-mse`) changed 451 of 504 weight scales. Its captures came from 8
sessions, the first, middle and last step, and both CFG branches.

A capture file is a `torch.save` dict: `"<tag>:<weight_key>"` (or `"<weight_key>"`) to a
`[rows, in_features]` tensor of the projection input, taken from the BF16 model. Each projection needs
rows.

### Export output

- `transformer/`: all shards of the source, with E4M3 weights and the `weight_scale` and `input_scale`
  scalars for the 504 projections. All other tensors are bit-identical to the source.
- `transformer/config.json` with `quantization_config` and `runtime.diffusion_step_policy`,
  `hf_quant_config.json`, both weight indexes.
- `quantization_metadata.json`: provenance (`collect_quantization_metadata`), `activation_amax`,
  `tensor_coverage` and `candidate_changes.weight_scales`.
- All other files of the source root are hard-linked (copied when the file systems differ).

## Reproduce a published checkpoint

Export with the `activation_amax` and the weight-scale map of the checkpoint metadata and compare every
tensor in `transformer/*.safetensors`. The two checkpoints must have no differing, missing or extra tensors.
This needs neither the calibration data nor the captures.

## Tests

`pytest tests/quantization/transfer_auto_test.py` runs on CPU with a tiny synthetic checkpoint.
