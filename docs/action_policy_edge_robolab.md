# Cosmos3 Edge native PyTorch policy server for RoboLab

This path serves a Cosmos3 Edge DCP from the native Cosmos Framework runtime
over the existing OpenPI-compatible RoboLab WebSocket protocol. It is intended
for single-user latency and integration validation on Thor.

The experimental `iter_000374000` checkpoint does not contain trained action
heads. When that checkpoint is used, rewards and task success are not policy
quality signals. The server requires an explicit opt-in before leaving only the
four action modules initialized, publishes their checksum as connection
metadata, and can persist or reload the exact state across runs.

## Thor launch

The Edge config must construct the architecture matching the DCP. For the
validated experiment it also points the vision tokenizer at the locally cached
Wan2.2 VAE through `WAN_VAE_PATH`.

```bash
export WAN_VAE_PATH=<path-to-Wan2.2_VAE.pth>
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas

python -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path <iter_000374000/model> \
  --allow-dcp-checkpoint \
  --config-file cosmos_framework/configs/base/edge_policy_native.py \
  --experiment edge_policy_native \
  --allow-missing-action-heads \
  --action-head-init-seed 0 \
  --save-action-head-state-path <experiment-dir/random_action_heads.pt> \
  --attention-backend pytorch_sdpa_cudnn \
  --deterministic-seed \
  --guidance 1.0 \
  --num-steps 4 \
  --shift 5.0 \
  --history-length 1 \
  --action-space joint_pos \
  --startup-warmup-requests 3 \
  --startup-warmup-prompt "Pick up the banana and place it in the bowl" \
  --port 8000
```

The defaults used above are full `torch.compile`, static shapes, no CUDA
graphs, action-only output, 480p transforms, 15 FPS conditioning, and a `32x8`
action chunk. Unsupported attention calls—including packed multi-sequence and
LSE-producing calls—retain the normal Cosmos backend selection. The cuDNN SDPA
override is therefore narrower than a global attention monkey patch.

On subsequent runs, replace `--save-action-head-state-path` with:

```bash
--action-head-state-path <experiment-dir/random_action_heads.pt>
```

The server refuses a state file with missing, extra, shape-mismatched, or
dtype-mismatched action tensors. It also refuses the missing-head opt-in for a
consolidated safetensors checkpoint; released policy checkpoints must load
their trained heads normally.

Use the exact evaluation prompt for startup warmups when compiling static
shapes. A materially different prompt length can trigger a later recompile.

## Validation

From the x86 RoboLab environment:

```bash
python scripts/smoke_cosmos3_policy_ws.py \
  --remote-host <thor-hostname-or-ip> \
  --remote-port 8000 \
  --prompt "Pick up the banana and place it in the bowl" \
  --requests 3

python policies/cosmos3/run.py \
  --remote-host <thor-hostname-or-ip> \
  --remote-port 8000 \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --output-folder-name edge_cudnn_plumbing_smoke \
  --headless
```

Each response contains finite `action` with shape `[32,8]` and a `timing`
dictionary with server preprocessing, policy inference, postprocessing, and
total time in milliseconds. WebSocket round-trip time minus `server_total_ms`
approximates client serialization plus network transport. RoboLab records
policy-loop and environment/action-application time under `timing` in
`episode_results.jsonl`.

For performance claims, capture OC1/OC2/OC3 counters immediately before and
after the measured requests. Any counter increase invalidates that run. Exclude
model loading, compilation, and startup warmups from steady-state latency.
