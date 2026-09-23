# Nano SFT remote Reasoner: Phase 3 test report

Date: 2026-09-22

Branch: `feat/nano-sft-remote-reasoner`

Base commit: `bd6ec1c` (`feat(training): decouple Nano Reasoner with offline K/V conditioning`)

## Scope

This report covers the Phase 3 MVP that removes the frozen Reasoner from
Generator training ranks and obtains the same per-layer K/V tensors from a
separate one-process/one-GPU gRPC service.

The implemented validation surface includes:

- shared offline/remote Reasoner construction and strict DCP loading;
- versioned identity, feature-signature, capability, and limit handshake;
- exact BF16 tensor codec, canonical `K0, V0, K1, V1, ...` ordering, per-chunk
  and whole-response SHA-256 validation;
- asynchronous provider submission with one absolute deadline and bounded
  transient retries;
- request/token admission limits, bounded gRPC concurrency, serialized GPU
  execution, and CPU staging before streaming;
- cancellation/deadline handling without queued ghost compute;
- OOM and permanent runtime-invariant unhealthy-state propagation to
  already-queued requests;
- provider cleanup on constructor, dataloader, dry-run, training-success, and
  training-failure paths;
- all-rank startup failure synchronization before any Generator rank enters
  FSDP construction; and
- fail-closed rejection of multiple causal documents (including the current
  per-view-caption representation) and specialized multiview attention metadata.

## Latest CPU regression run

Command:

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m pytest -q \
  cosmos_framework/model/generator/reasoner_remote_test.py \
  cosmos_framework/model/generator/reasoner_remote_lifecycle_test.py \
  cosmos_framework/model/generator/reasoner_remote_server_test.py \
  cosmos_framework/model/generator/reasoner_runtime_test.py \
  cosmos_framework/model/generator/reasoner_features_test.py \
  cosmos_framework/model/generator/reasoner_feature_cache_test.py \
  cosmos_framework/scripts/extract_reasoner_features_test.py \
  cosmos_framework/scripts/serve_reasoner_features_test.py \
  cosmos_framework/model/generator/omni_mot_reasoner_conditioning_test.py \
  cosmos_framework/configs/toml_config/sft_config_test.py \
  cosmos_framework/trainer/distillation_test.py \
  cosmos_framework/model/model_lifecycle_test.py
```

Result: **186 passed in 65.67 seconds**.

The run emitted 31 existing `PytestUnknownMarkWarning` messages from
`trainer/distillation_test.py` for the legacy `L0`/`CPU` markers. There were no
test failures, skips caused by this feature, OOMs, or hangs.

New failure-mode regressions specifically verify:

- executor concurrency overflow returns `RESOURCE_EXHAUSTED`;
- explicit cancellation and deadline expiry do not execute a queued request;
- an OOM marks the replica unhealthy before the next queued request can run;
- an invalid runtime output signature or fingerprint order marks the replica
  unhealthy before queued and subsequent requests can execute;
- malformed protocol/identity/dtype/order/shape/checksum inputs fail closed;
- partial streams are cancelled and token reservations are released;
- channel shutdown interrupts an in-flight RPC and retry backoff;
- constructor/handshake failures close both channel and executor;
- a rank-local startup/handshake failure is surfaced on every distributed rank,
  while providers created successfully on peer ranks are closed;
- rank-local provider/identity/sample-key/request-build/submit validation errors
  are deferred into failed futures and surfaced through the same all-rank
  pre-forward failure gate; and
- stream timing excludes client/network backpressure.

The service-config regression additionally unsets `DATASET_PATH`,
`WAN_VAE_PATH`, and `BASE_CHECKPOINT_PATH` before loading the shipped Nano SFT
recipe. The service now prunes its unused dataloader, video-VAE, and training
checkpoint subtrees before OmegaConf resolution, while preserving explicit
Reasoner model overrides. A Reasoner-only replica therefore no longer requires
those training-only environment variables.

## Static validation

- Ruff check and format check passed for every modified or added Python file.
- Pyrefly reported **0 errors** (15 existing suppressions); it also reported the
  existing `ignore-missing-source` extra-key warning from `pyrefly.toml`.
- `git diff --check` passed.
- The standard `uv-lock` pre-commit hook and `uv 0.11.14 uv lock --check`
  passed without skipping or rewriting the lock file.

## Issue encountered and resolved

The first real service launch failed during generic SFT-config resolution,
before the Reasoner was constructed, because the complete training recipe also
resolved dataset, VAE, and training-checkpoint environment variables. Supplying
those variables confirmed the GPU path, but they are unrelated to Reasoner
serving. The service-specific loader now replaces those unused subtrees before
resolution, and the no-environment regression above prevents the dependency
from returning.

## Real H20 correctness smoke

The shipped `examples/checkpoints/Cosmos3-Nano` regular DCP was loaded into two
independent H20 processes: one direct `ReasonerFeatureRuntime` reference and one
localhost gRPC service. The request was a 10-token BF16 framed prompt.

| Measurement                       |                                 Result |
| --------------------------------- | -------------------------------------: |
| Reasoner layers compared          |                                     36 |
| K/V equality                      | Bitwise equal for every K and V tensor |
| Direct extraction                 |                             0.293149 s |
| Remote localhost round trip       |                             0.326337 s |
| Server Reasoner compute           |                             0.288081 s |
| Server device-to-host copy        |                             0.002015 s |
| Server CPU stream preparation     |                             0.003000 s |
| Service allocated after load      |           16,383,586,816 B (15.26 GiB) |
| Service peak reserved during load |           16,536,043,520 B (15.40 GiB) |

The final smoke also verified the advertised four-request preflight limit. The
service process was stopped after the test and both GPUs returned to zero
reported allocation. This is a correctness smoke, not a capacity result: it
uses a short prompt, loopback networking, one request, and no concurrent
Generator ranks.

## Real seven-rank execution/resource smoke (zero-LR first iteration)

The complete remote-conditioned execution path was exercised with GPU 0
reserved for one Reasoner service replica and physical GPUs 1--7 running a
seven-rank Generator FSDP job. The job used the official eight-video BridgeData
sample, a 16,384-token packing cap, BF16, full activation checkpointing, FP32
EMA, gradient accumulation one, and eager execution. It warm-started from the
shipped Nano DCP and saved a new seven-way DCP.

The run completed one forward/backward pass and one scheduled optimizer call.
However, the recipe starts its 50-step warmup at an LR multiplier of zero, so
that optimizer call used an effective learning rate of zero. The run produced a
nonzero gradient norm and initialized Adam moment state, but an exhaustive
comparison against the warm-start DCP found that 0/405 live Generator tensors
changed (`max_abs_delta=0`). It therefore validates the remote execution,
optimizer-state, checkpoint, and resource paths, not a learned parameter update.

| Measurement                                 |                       Result |
| ------------------------------------------- | ---------------------------: |
| Optimizer calls / nonzero-LR updates        |                        1 / 0 |
| Effective optimizer LR                      |                            0 |
| Final loss                                  |                       0.2512 |
| Global gradient norm                        |                      0.49541 |
| Live Generator tensors changed              |  0 / 405 (`max_abs_delta=0`) |
| Reported iteration time                     |                      71.79 s |
| Checkpoint portion of iteration             |                      31.37 s |
| End-to-end `torchrun` wall time             |                     101.35 s |
| Generator peak allocated, min/mean/max      | 20.414 / 20.441 / 20.459 GiB |
| Generator peak reserved, min/mean/max       | 21.467 / 21.487 / 21.506 GiB |
| Generator physical peak, `nvidia-smi`       |       23.196--25.110 GiB/GPU |
| Reasoner physical peak, `nvidia-smi`        |                   18.493 GiB |
| Saved checkpoint size                       |                      104 GiB |
| CUDA OOM / RPC failure / training exception |                            0 |

The saved model metadata contains 810 leaves: 405 live Generator leaves and
405 EMA Generator leaves. It contains no `embed_tokens`, `lm_head`, or
`moe_und` parameter, confirming that the checkpoint does not recreate the
frozen Reasoner. This is a structural checkpoint result only; because the
effective learning rate was zero, it does not demonstrate learned Generator
weights or remote-training numerical parity. After graceful shutdown, all
eight GPUs returned to zero reported memory use.

After adding the final all-rank feature-signature gate, the same seven-rank job
was started again against a fresh service process and resumed the saved DCP at
iteration 1. It completed the resume-only path in 38.70 seconds without entering
another training step or writing another checkpoint. This validates provider
startup, signature synchronization, seven-way DCP reshard/load, and cleanup.
Because the loaded iteration already equalled `max_iter`, it did not validate
resume-and-continue training.

Artifacts:

- run root:
  `outputs/nano_sft_reasoner_benchmark/remote_smoke_7gpu_1step_20260922_v1`;
- Generator log: `logs/generator_train.log`;
- Reasoner startup log: `logs/reasoner_service.log`;
- 500 ms physical-GPU telemetry: `logs/nvidia_smi.csv`;
- final-code resume gate: `logs/generator_resume_gate.log` and
  `logs/reasoner_service_final_gate.log`; and
- checkpoint:
  `train/cosmos3/sft/nano_remote_smoke_7gpu/checkpoints/iter_000000001`.

This is an end-to-end execution-path and resource smoke, not a training-
correctness or throughput comparison. It uses seven Generator ranks, a shorter
packing cap, one zero-LR step, and an iteration time that includes checkpoint
writing.

## Real nonzero-LR Generator update and continuation

A fresh two-iteration run used the same one-Reasoner-plus-seven-Generator
topology and saved only `iter_000000002`. Its first optimizer call used LR 0;
after the first scheduler advance, the second call used LR `2e-6`.

| Measurement                                 |                       Result |
| ------------------------------------------- | ---------------------------: |
| Optimizer calls / nonzero-LR updates        |                        2 / 1 |
| Effective optimizer LRs                     |                  `0`, `2e-6` |
| Iteration 1 loss / gradient norm / time     |   0.2496 / 0.54616 / 41.21 s |
| Iteration 2 loss / gradient norm / time     |   0.2131 / 0.35519 / 52.28 s |
| Iteration 2 checkpoint save                 |                      32.02 s |
| End-to-end `torchrun` wall time             |                     120.62 s |
| Generator physical peak, min/mean/max       | 30.007 / 30.922 / 31.718 GiB |
| Reasoner physical peak                      |                   18.577 GiB |
| CUDA OOM / RPC failure / training exception |                            0 |

An exhaustive CPU-streaming DCP comparison against the shipped warm-start
checkpoint found that all 405 live Generator tensors changed. In total,
6,714,184,413 of 6,965,486,784 elements changed (96.3922%). The maximum
absolute delta was `2.026557922e-6`, mean absolute delta was
`2.529132027e-7`, L2 delta was `0.03197588908`, and relative L2 delta was
`1.500317368e-5`; neither checkpoint contained NaN or Inf values. All 405
optimizer step states were 2. The saved scheduler state had `last_epoch=2` and
LR `4e-6` for the next optimizer update.

This validates a real nonzero-LR Generator update through remote conditioning.
It does not establish learning quality or numerical parity against `joint`,
`inline`, or `offline`. Unlike the original one-iteration smoke, this run used
the recipe-default CFG-dropout rate of 0.1 and the default T2V/I2V/V2V
conditioning distribution of 0.7/0.2/0.1. The old and new loss/timing values
must therefore not be treated as a paired A/B comparison.

### Same-job resume and continued update

The same job was restarted against a fresh Reasoner service and restored the
iteration-2 model, optimizer, scheduler, and trainer state in 10.49 seconds.
Iteration 3 then used the restored LR of `4e-6`, reported loss 0.2138 and global
gradient norm 0.55963, and saved `iter_000000003`. Its reported 73.23-second
iteration included a 30.29-second checkpoint save; end-to-end process wall time
was 102.831 seconds.

The iter-2-to-iter-3 DCP audit found that 405/405 live Generator tensors changed:
6,763,395,246 of 6,965,486,784 elements (97.0987%), with maximum absolute delta
`4.053115845e-6`, mean absolute delta `4.378727690e-7`, relative L2 delta
`2.647678690e-5`, and no NaN or Inf. All optimizer step states advanced from 2
to 3; the scheduler advanced from `last_epoch=2` to 3 and saved the next LR as
`6e-6`. The resulting model DCP still has exactly 405 live plus 405 EMA
Generator leaves and no Reasoner leaves.

Physical peaks during the resumed process were 18.386 GiB on the Reasoner GPU
and 30.019--32.972 GiB on the seven Generator GPUs (31.104 GiB mean). No OOM,
RPC, NCCL, or training error occurred, and all eight GPUs returned to zero
memory after cleanup.

The checkpoint loader requested dataloader state, but the iteration-2 checkpoint
did not contain a `dataloader` subtree, so every rank explicitly skipped it.
This result validates operational resume-and-continue for model, optimizer,
scheduler, and trainer state. It does not validate exact datastream-position
resume or uninterrupted-versus-resumed numerical parity.

One launch attempt failed before CUDA initialization because the training script
was passed to `torchrun` by file path. That placed `cosmos_framework/scripts`
first on `sys.path`, causing the sibling `hydra.py` to shadow the installed
Hydra package. The supported module entrypoint succeeded:

```bash
torchrun --nproc_per_node=7 --module cosmos_framework.scripts.train ...
```

Artifacts:

- run root:
  `outputs/nano_sft_reasoner_benchmark/remote_smoke_7gpu_2step_20260922_v2`;
- fresh-run logs: `logs/generator_train.log`, `logs/reasoner_service.log`, and
  `logs/nvidia_smi.csv`;
- resume logs: `logs/generator_resume_iter2_to_iter3.log`,
  `logs/reasoner_service_resume_iter3.log`, and
  `logs/nvidia_smi_resume_iter3.csv`;
- failed-entrypoint log:
  `logs/generator_resume_iter2_to_iter3.failed_direct_script.log`;
- state audits: `logs/training_state_audit_iter2.json` and
  `logs/training_state_audit_iter3.json`;
- derived GPU summary: `logs/gpu_telemetry_summary.json`;
- exhaustive DCP audits:
  `logs/generator_weight_delta_iter2_vs_base.json` and
  `logs/generator_weight_delta_iter3_vs_iter2.json`; and
- checkpoints:
  `train/cosmos3/sft/nano_remote_nonzero_7gpu/checkpoints/iter_000000002`
  and `iter_000000003` (104 GiB each).

These raw artifacts are local-only: the run roots are under the gitignored
`outputs/` directory and therefore do not travel with this branch. Archive them
separately when raw logs or the 104 GiB checkpoints must be handed off.

## Existing offline A/B baseline

The earlier 8-H20 short-run comparison remains the performance baseline:

| Backend                      | Mean step time | Mean peak allocated per Generator GPU |
| ---------------------------- | -------------: | ------------------------------------: |
| Joint Reasoner + Generator   |       57.887 s |                            52.128 GiB |
| Offline K/V + Generator only |       48.015 s |                            36.498 GiB |

Offline conditioning reduced mean step time by 17.05% and peak allocated memory
by 15.63 GiB per Generator GPU in that short run. Remote conditioning has the
same Generator-side structural pruning, but its end-to-end training throughput
must still be measured under realistic service concurrency and networking.

## Remaining gates

- Run 1/2/4/8 concurrent Generator ranks against one service replica and record
  queue, compute, D2H, stream preparation, network/client wait, and training-step
  percentiles.
- Repeat remote parity with representative and maximum-length prompts.
- Compare the remote step's loss, selected gradients, and post-update weights
  against an identically configured `inline` or `offline` run.
- Persist and restore dataloader position, then compare a resumed update against
  an uninterrupted run with identical samples and RNG state.
- Derive and verify the Reasoner, tokenizer, and framing fingerprints from the
  actual checkpoint and tokenizer artifacts. The MVP strictly compares the
  operator-supplied labels, but does not yet prove that a label is the content
  digest of the artifact it names.
- Add TLS/authentication and deployment health/metrics before cross-host
  production use. The MVP binds to loopback by default and non-loopback serving
  is restricted operationally to an isolated, trusted network.
- Add multiple causal-document and specialized cached-attention support before
  using 7-view/11-view per-view captions; protocol v1 currently rejects the
  multi-document input deliberately. Shared-caption multiview also needs its
  separate attention-layout parity gate.
