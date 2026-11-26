<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Quantize Cosmos3-Nano-Transfer-Auto to INT8 (W8A8)

This recipe makes an INT8 W8A8 checkpoint of `nvidia/Cosmos3-Nano-Transfer-Auto` for vLLM-Omni. It lives in
`cosmos_framework/quantization/int8.py`. The export is a streaming weight conversion from the BF16 source. The
FP8 recipe for the same model is in `quantize-transfer-auto.md`; the two recipes are independent.

## Format

- 504 linears: q, k, v, o and gate, up, down, in both towers (understanding and generator), in each of the
  36 layers.
- Weights: symmetric INT8 with one FP32 scale for each output channel (`scale = max|W[n,:]| / 127`).
- Activations: dynamic per-token INT8, computed at run time. The checkpoint has no `input_scale`.
- SmoothQuant, alpha 0.5: `s_c = max|X_c|^0.5 / max|W[:,c]|^0.5`. The checkpoint holds the weight `W * s`
  and the FP32 `pre_quant_scale = 1 / s`. The runtime multiplies the activation by `pre_quant_scale`
  before it quantizes the activation.
- MAPS policy in the checkpoint: `quantization_config.runtime.diffusion_step_policy` in
  `transformer/config.json`, with the same schema as the FP8 checkpoint. The runtime uses A16 activations in
  the first 3 and the last 3 denoising steps and in the reasoner, and INT8 activations in the other steps.
  The server needs no `--additional-config`. `--mixed-precision-steps N` changes the count (default 3).
  `--mixed-precision-steps 0 --reasoner native` writes no policy.
- Stays in BF16 and is copied without change: norms, embeddings, `proj_in`, `proj_out`, the time embedder, the
  LiDAR projections, the visual encoder, and all auxiliary models. Attention math is not quantized.
- Static per-tensor INT8 activations (`--static`) give bad video (FVD above 3000). Do not use them.

## Calibration

SmoothQuant needs the per-input-channel absmax of the activations of each of the 504 linears. The
`channel-amax` command records it with the native `OmniInference.generate` (GPU needed). The released
checkpoint used 4 calibration sessions of 7 cameras. All 35 denoising steps and both CFG branches count.
The reduction is a max, so the order of the sessions does not matter.

## Commands

Setup is the one in `quantization-readme.md`. `SRC` is the BF16 checkpoint directory.

```bash
# 1. Channel absmax (GPU needed). Request JSONL files are the same as for the FP8 calibration.
python -m cosmos_framework.quantization.int8 channel-amax \
  --checkpoint "$SRC" --requests requests/session*.jsonl --output channel_amax.json

# 2. Export from the pristine BF16 source (CPU is enough).
python -m cosmos_framework.quantization.int8 "$SRC" "$OUT" \
  --channel-amax channel_amax.json --smoothquant-alpha 0.5
```

Python: `collect_channel_amax` and `export_transfer_auto_int8`. Without `--channel-amax` the export is plain
dynamic INT8 without SmoothQuant. FVD is much worse.

## Serve

```bash
vllm serve "$OUT" --omni --served-model-name Cosmos3-Nano-Transfer-Auto \
  --model-class-name Cosmos3MultiviewPipeline --no-guardrails
```

No `--additional-config` is needed: the server reads the MAPS policy from the checkpoint.

## Runtime requirement

The checkpoint needs a vLLM-Omni that loads serialized INT8 diffusion checkpoints, with the SmoothQuant
`pre_quant_scale` and the INT8 path for SM100 (`torch._int_mm` and Triton kernels, because vLLM CUTLASS
refuses INT8 on SM100). This is the vLLM-Omni INT8 PR, branch `wkutak/cosmos3-int8-w8a8`. Older
vLLM-Omni versions do not load the checkpoint.

## Results

| B200, vLLM-Omni, 7 cameras x 93 frames, vs BF16 | INT8 SQ0.5 + MAPS | FP8 + MAPS |
| ----------------------------------------------- | ----------------: | ---------: |
| FVD, 8 development scenes                       |             205.5 |      274.0 |
| FVD, 2 held-out scenes                          |             154.9 |      257.8 |
| 35-step denoise, 29 frames                      |              84 s |       83 s |
| Transformer weights                             |          16.1 GiB |   16.1 GiB |

FVD is measured against BF16 from the same runtime and GPU type. MAPS is the largest quality lever and
SmoothQuant is the second.

## Tests

`pytest --noconftest -c /dev/null tests/quantization/int8_test.py` runs on CPU with a tiny synthetic
checkpoint.
