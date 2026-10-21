# Run guide Q&A

Questions that come up when following the [container run guide](run-posttrain.md) and the
Joint PPO guides ([π0.5](../guides/joint-ppo-pi05-public.md),
[GR00T-N1.7](../guides/joint-ppo-gr00t-public.md)) for the first time. Timings below were measured
on one GB200 node with four GPUs, in `nvcr.io/nvidia/pytorch:26.06-py3`, with the repository on a
shared Lustre filesystem; expect different numbers on other hosts and storage.

## How long does each recipe run, and when can I stop it?

The showcase recipes keep their full training lengths. Several of them run for hours, and the
guide's launch commands run to `rl.max_steps` unless you stop them:

| Recipe               | `rl.max_steps` | Launch to first step                         | Time per step | Full run    |
| -------------------- | -------------: | -------------------------------------------- | ------------: | ----------- |
| GRPO                 |            300 | ~5 min (vLLM kernel compilation)             |         ~27 s | ~2 h 15 min |
| DPO                  |           1000 | ~1–9 min (imports on shared storage)         |          ~3 s | ~45 min     |
| PPO                  |           1000 | ~4 min                                       |         ~11 s | ~3 h        |
| OPD                  |             10 | ~3 min                                       |          ~8 s | ~4 min      |
| NFT                  |            400 | ~1 min after the reward service is up        |        ~4.6 s | ~30 min     |
| GR00T-N1.7 Joint PPO |            800 | ~5 min (pre-training validation, 100 cases)  |         ~25 s | ~6.5 h      |
| π0.5 Joint PPO       |          12000 | ~40 min (pre-training validation, 500 cases) |         ~37 s | days        |

A run is working once its log prints `step N/<max_steps>` lines with finite losses. To check that a
recipe runs end to end, a few steps are enough: the runs behind the
[qualified commands](../index.md#qualified-commands) used five updates.

Stopping a run (Ctrl-C, or cancelling the job) loses everything after its last complete
checkpoint. Recipes with `rl.save_every=0` (GRPO, DPO) write no checkpoint, so nothing can be
resumed. See [Outputs and resume](run-posttrain.md#9-outputs-and-resume).

## How do I make a short run?

Append `rl.max_steps=<N>` to the dotlist after `--`. Every Posttrain entrypoint accepts the same
override form:

```bash
# GRPO, PPO, OPD: add it after `--`
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k_public_4gpu.py -- \
  job.name=grpo_public_4gpu rl.max_steps=5

# DPO under torchrun: same, after the config's `--`
"$POSTTRAIN_PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m cosmos_framework.posttrain.entrypoints.dpo \
  --config cosmos_framework/posttrain/configs/dpo_t2i_public.py -- \
  job.name=dpo_public_4gpu rl.max_steps=5

# NFT: next to the experiment name
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.nft -- \
  experiment=cosmos3_nft_t2i_public_4gpu job.name=nft_public_4gpu rl.max_steps=5

# Joint PPO: also turn validation off (see the next question)
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/pi05_libero_spatial_public_4gpu.py -- \
  rl.max_steps=5 rl.validation.enabled=false
```

`config_resolved.yaml` in the run directory records the values the run actually used.

On the reference host, with the environment and model downloads from the guides already in place,
each of these five-step runs exits with code 0 in a few minutes:

| Recipe               | Overrides                                    | Wall time |
| -------------------- | -------------------------------------------- | --------: |
| GRPO                 | `rl.max_steps=5`                             |  ~4.5 min |
| DPO                  | `rl.max_steps=5`                             |  ~1.5 min |
| PPO                  | `rl.max_steps=5`                             |    ~4 min |
| NFT                  | `rl.max_steps=5`                             |    ~2 min |
| π0.5 Joint PPO       | `rl.max_steps=5 rl.validation.enabled=false` |  ~6.5 min |
| GR00T-N1.7 Joint PPO | `rl.max_steps=5 rl.validation.enabled=false` |  ~4.5 min |

OPD already runs ten steps by default (about four minutes).

## π0.5 prints nothing for about 40 minutes after it starts. Is it hung?

Usually not. Both Joint PPO recipes enable train-time validation with `before_training=True`.
Before the first training step, π0.5 evaluates the policy on all 500 cases of the
`libero_spatial-shared_500-eval` manifest. That takes about 40 minutes on four GPUs, and the main
log prints nothing between the last `setup progress` line and

```text
step    0/12000  val/success_rate +0.8480
```

GR00T-N1.7 does the same on its 100-case `legacy_100` manifest in about 5 minutes.

To confirm the evaluation is progressing, look inside the container:

- `nvidia-smi` shows the four GPUs in use;
- `ps` shows LIBERO simulator processes started from `$POSTTRAIN_SIM_VENV/bin/python` and four
  `vLLM-Omni::DiffusionWorker` processes;
- the newest `$RAY_TMPDIR/session_*/logs/worker-*.err` files keep growing with
  `Processed prompts` lines.

The same validation also runs periodically (π0.5 every 480 steps, GR00T every 80 steps, each
saving a checkpoint) and on the final step.

**How to change it:**

| Goal                                      | Override                                                        |
| ----------------------------------------- | --------------------------------------------------------------- |
| Skip validation entirely, for a smoke run | `rl.validation.enabled=false`                                   |
| Skip only the step-0 pass                 | `rl.validation.before_training=false`                           |
| Validate less often                       | `rl.validation.every_steps=<N>`; `0` disables the periodic pass |

`before_training=false` alone does not shorten a short run: once validation is enabled, validating
the final step is mandatory, so a five-step π0.5 run still ends with the 40-minute pass. Joint PPO
evaluates every case of its manifest and does not apply `rl.validation.max_samples`. The general
validation options are described in [Train-time model validation](train-time-validation.md).

## How often do the recipes write checkpoints?

| Recipe               |    `rl.save_every` |
| -------------------- | -----------------: |
| GRPO, DPO            | 0 (no checkpoints) |
| OPD                  |                  5 |
| NFT                  |                 10 |
| PPO                  |                100 |
| GR00T-N1.7 Joint PPO |                 80 |
| π0.5 Joint PPO       |                480 |

Change the interval with `rl.save_every=<N>`; `0` disables periodic saves. When `rl.save_every` is
above zero, a run that reaches `rl.max_steps` also saves its final step, so a five-step run of
these recipes leaves one checkpoint, `iter_000005`. Checkpoints land under
`checkpoints/iter_*` in the run directory, and `latest_checkpoint.txt` names the newest complete
one. See the [checkpoint reference](../reference/checkpoint.md) for formats, retention and resume.

## `uv sync` reports a required-version mismatch

The root project and `docker/posttrain/environment` require `uv==0.11.29`. Check the selected
executable with `which uv && uv --version`, install that exact version, then delete
`.venv-posttrain` and repeat [section 2](run-posttrain.md#2-install-the-environment).

## Which `uv` runs the first `uv sync`?

`nvcr.io/nvidia/pytorch:26.06-py3` already ships `/root/.local/bin/uv` (0.11.14), which comes
before pip's install location on `PATH`. After `python3 -m pip install uv==0.11.29`, a bare `uv`
may still run 0.11.14 and will be rejected by the root project's exact `uv==0.11.29` requirement.
Check with `which uv && uv --version`. Use `python3 -m uv` to run the pip-installed 0.11.29 before
the virtual environment is activated.

## Joint PPO fails with `LIBERO runtime dependencies differ: {'numpy': ...}`

```text
WireError: expected 'capabilities', the simulator reported:
RuntimeError: LIBERO runtime dependencies differ: {'numpy': {'expected': '1.26.4', 'actual': '2.2.6'}}
```

The simulator environment was installed with the repository's `pyproject.toml` settings applied:
its `override-dependencies` replace the lock's `numpy==1.26.4` with numpy 2.x. Rebuild the
simulator environment with the guide's `--no-config` install:

```bash
rm -rf "$POSTTRAIN_SIM_VENV"
uv venv --python 3.12 "$POSTTRAIN_SIM_VENV"
uv pip install --no-config --python "$POSTTRAIN_SIM_VENV/bin/python" --no-deps \
  -r cosmos_framework/posttrain/embodied/sims/libero/requirements.lock
"$POSTTRAIN_SIM_VENV/bin/python" -c 'import numpy; print(numpy.__version__)'  # 1.26.4
```

The same environment serves both π0.5 and GR00T-N1.7.

## Do I need a Hugging Face token?

- GRPO, DPO, PPO, OPD, NFT: no. Their models and datasets are public.
- π0.5: no. `RLinf/RLinf-Pi05-LIBERO-SFT` and the PaliGemma tokenizer are public.
- GR00T-N1.7: yes. Its processor loads `nvidia/Cosmos-Reason2-2B`, which is gated. Accept its
  license on Hugging Face, then export a read token before downloading and launching.

Export a real token or none at all: the guides' `export HF_TOKEN="<your-read-token>"` line is a
placeholder.

## uv prints `VIRTUAL_ENV=.venv-posttrain does not match the project environment path`

This appears in the first π0.5 step, which creates `.venv-posttrain-joint` while the section 2
environment is still active. uv ignores the active environment and installs into
`.venv-posttrain-joint`, so the warning is harmless. Run `deactivate` first to avoid it.

## PPO's `train/reward_mean` stays near zero

In our runs of the public PPO profile, `train/reward_mean` stayed between about `-3e-5` and `0` for
every step: the task reward (`arithmetic_exact_match`) was not earned in those steps. The profile
is qualified for wiring (all four roles, policy and critic updates, checkpointing), not for
showing learning. Use the GRPO recipe to see a reward that moves.
