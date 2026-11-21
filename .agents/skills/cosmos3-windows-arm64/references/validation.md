# What was validated, where, and what is still pending

Numbers are quoted from the runs that produced them; "one run" is not a benchmark.

## Lifecycle on PR #303 (Cosmos QA, 2026-11-03 and 11-04)

Machine: RTX Spark N1X, Windows 11 Pro ARM64, 38 GB RAM (45.45 GiB CUDA-visible), no admin
rights, no WSL. Stack: ARM64 CPython 3.13.18 (uv), torch 2.15.0.dev20260902+cu134,
torchcodec 0.17.0a0 (built against that torch), transformers 4.57.6, peft 0.20.0,
cosmos-framework at PR #303 head 4475a3a plus the guards that are now part of the PR.

| Leg                                                       | Result                                                                                                                                                                      |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| import and `train --help`                                 | pass                                                                                                                                                                        |
| reasoner config import and Cosmos3-Nano to VLM conversion | pass: 750 tensors, 4 shards, 16.3 GiB                                                                                                                                       |
| one-step LoRA smoke                                       | pass: 144 LoRA modules, 15,335,424 trainable parameters, 47 s first step, 33 GB DCP save                                                                                    |
| 20-step, 2-epoch run with validation                      | pass: train loss 2.17, validation loss 4.55 then 4.31, about 1.3 s per step after warm-up, 33 GB save in about two minutes; free host memory fell to 0.9 GB during the save |
| export (`export_vlm_dcp` with the base model)             | pass for both checkpoints: 750 tensors, 5 shards, zero residual LoRA keys                                                                                                   |
| Framework evaluate (binary, TorchCodec CUDA decode)       | pass: 4 of 4 on the four-item smoke set                                                                                                                                     |

Required overrides on that stack: `max_sequence_length 8192` (memory), sdpa on both policy
keys, async checkpointing off. All of them are now defaults of this skill's launcher or recipe.

Retest at the rebased head 7a6bca4 (main 7a4b057 plus the Windows commits; Cosmos QA, 2026-11-06), stock branch, no local
guards, no overrides, `toml` uninstalled: config import, the one-step smoke, a 20-step two-epoch run with validation
(train loss 2.17, validation 4.55 to 4.31), two 33 GB DCP saves at 132 s each, two exports with zero residual LoRA keys and
two evaluations all passed; the sdpa and synchronous-save fallbacks engaged on their own. Unit tests: 12 passed, 1 skipped,
3 failed, the failures being POSIX-only `os.statvfs` mocks. The two code findings (a second `toml` import in
`utils/workflow_status.py`, the statvfs mocks) were fixed in 69fee1f. That head was rebased onto main on
2026-11-10 and became e682c47, the commit the skill's blocks were run on (`validation-spark-a.md`; the
runs of that report made after 16:26 used 0e84637, a test-only commit that differs in `dcp_test.py` alone).

The PR then gained its review fixes, up to c8441aa. On that commit the PR's unit tests ran
on the 128 GB unit on 2026-11-16: 25 passed and 5 skipped in the four helper test files, 4 passed and 3
skipped in `dcp_test.py` and `background_store_test.py`, and the link cleanup worked in an elevated and in a
standard shell. A Linux comparison of that commit against `main` is in the PR thread. On 2026-11-17 the PR
was rebased onto `main` at 0822f1d and became 985d116, the commit Block 2 pins. The rebase changed none of
the PR's commits. The skill's blocks were rerun on 985d116 the same day (§ Rerun at the pinned commit). One difference from e682c47 matters
to this skill: the evaluator imports `toml` again (the `tomllib` change moved to PR #399), so the dependency
file lists `toml`.

## Rerun at the pinned commit (128 GB unit, 2026-11-17)

Fresh root and fresh venv, every block as written in Windows PowerShell 5.1, elevated, 1400 MHz cap.
Framework 985d116, skill commit 8ea9f02. Every block that ran passed on the first attempt.

| Block or check                                | Result                                                                                                                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Variant 1, Block 1                            | pass, 447 s; preflight `PASS`, 101.5 GiB CUDA-visible, the 72 GiB write in 175 s                                                                                              |
| Variant 1, Block 2                            | pass, 370 s; `HEAD is now at 985d116`; `toml` 0.10.2 from the dependency file; `pip check` shows the three expected lines                                                     |
| Missing pyarrow                               | `AttributeError` that names pyarrow, on first attribute access                                                                                                                |
| Variant 2                                     | not exercised: no raw release on the machine; the model converted on 2026-11-10 was used                                                                                      |
| Variant 3, `smoke_20`                         | pass, 217 s; 144 modules; epoch means 0.491 and 0.568; validation loss 0.447, then 0.343; 1.91 s per step; save 112 s; peak 45 C; no pause; 83.6 GiB peak private host memory |
| Variant 4                                     | pass, 291 s; 750 tensors, 5 shards, 0 lora keys                                                                                                                               |
| Variant 5                                     | pass, 38 s; 2 of 4, the same two misses as on 2026-11-10                                                                                                                      |
| Memorisation run, ten epochs of sixteen steps | epoch means 0.530, 0.133, 0.0009, 0.0001, then 0.0000; evaluator 12 of 16, the lower bound of Variant 5's caveat; 16 of 16 with PR #398 applied on top of the pin             |
| Unit tests of #303                            | 25 passed and 5 skipped in the four helper test files; 4 passed and 3 skipped in `dcp_test.py` and `background_store_test.py`                                                 |

No expected string of Variants 3 to 5 is printed differently at this commit. Not exercised in this rerun:
Variant 2, a standard (non-elevated) shell, the 64 GB unit.

## Fine-tuning results on the earlier stack (ITS work, 2026-10-30 to 11-04)

Machine: RTX Spark N1X, 128 GB (101.5 GiB CUDA-visible), 1600 MHz clock cap, pacer 84/78/87 C.
Stack: cosmos-framework `02b05df` (the PR #164 base) with the vendored Windows patch, `cosmos`
attention. Data: 960 binary video QA records (synthetic traffic renders), 120-record held-out set.

| Configuration                                  | Last-epoch train loss | Held-out accuracy / balanced | Train time                | Peak C |
| ---------------------------------------------- | --------------------- | ---------------------------- | ------------------------- | ------ |
| base model (no training)                       |                       | 0.758                        |                           |        |
| recipe at 128x64, 3 epochs                     | 0.150                 | 0.800 / 0.805                | 38.8 min                  | 75     |
| recipe v2: 8 frames at 480x288 + system prompt | 0.077                 | 0.833 / 0.830                | 75.5 min (9.3 min paused) | 85     |
| recipe v2 + lr 2e-5, warm-up, 4 epochs         | 0.139                 | 0.783 / 0.780                | 99 min                    | 85     |

Memorisation test (64 records, 10 epochs, lr 1e-4, recipe v2 input): mean loss 0.478 / 0.148 /
0.004 / 0.000 at epochs 1 / 3 / 5 / 10; a matched evaluator answered 64 of 64.

## Fine-tuning results on the current stack (ITS work, 2026-11-05 to 11-09)

Machine: the same 128 GB unit. Stack: cosmos-framework main `4428b96` with PR #303, sdpa on both
policy keys, torch `2.15.0.dev20260904+cu134` (nightly), torchcodec `0.17.0a0+1ebfad0` (built
against the September nightly). Same 960 records and 120-record held-out set as above.

| Configuration                                                | Last-epoch train loss                | Held-out accuracy / balanced | Train time          | Peak C |
| ------------------------------------------------------------ | ------------------------------------ | ---------------------------- | ------------------- | ------ |
| recipe v2: 8 frames at 480x288 + system prompt, 1400 MHz cap | 0.079 (epochs 0.249 / 0.163 / 0.079) | 0.808 / 0.804                | 113.2 min, no pause | 83     |

Against the earlier stack's 0.833 / 0.830 the difference is three records of 120. Step time is
1.80 s (1.64 to 2.02) at 17 to 19 W. A first attempt at 1600 MHz paused at 84 C after 15 minutes
and the GPU latched two seconds after the resume (`thermal-operating-point.md`).

Memorisation test on this stack (64 records, 10 epochs, lr 1e-4, recipe v2 input): mean loss
0.478 / 0.153 / 0.026 / 0.000 at epochs 1 / 3 / 5 / 10 in 21 minutes; the evaluator answered
64 of 64. sdpa on main memorises as `cosmos` did on the earlier stack.

Resource readings on this stack: 33 GB per DCP save (84 s after a one-step run, 132 s in QA's
run), about 83 GB peak private host memory during training.

## W14: why sdpa is right on main and was wrong before

On checkouts older than August 2026 the trainer dropped the attention mask and the HF `sdpa` and
`eager` paths derived false packed-sequence boundaries; training completed and reported a loss,
but the model converged to the label prior (loss 0.34 on yes/no sets, accuracy equal to the
base). The `cosmos` adapter did not have the defect, which is why the earlier Windows guidance
said "keep cosmos". Upstream fixed the mask handling in August (#198); on main sdpa receives the
mask and trains correctly, and on Windows the `cosmos` adapter's vision-tower path has no
backend, so #303 resolves it to sdpa. The memorisation test above is the one-command check for
either failure.

## Pending (as of 2026-11-10)

Update, 2026-11-10 evening: the first four items below were run on the 128 GB unit and are
reported in `validation-spark-a.md` (stable torch pairing, the first-day changes at both
privilege levels, the reduced release, the smoke loss on that unit). Still open from this
list: the smoke loss on the 64 GB unit, a packaged agent app, an x64 uv, and the eval cases.
New from that run: the evaluator feeds the model frames rounded to multiples of 64 pixels where
training uses multiples of 32, so its accuracy is a lower bound unless the training frame is a
multiple of 64 in both directions (`validation-spark-a.md` § Why the evaluator disagreed).

- This skill's commands on the stable torch `2.15.0+cu134` pairing with a matching TorchCodec:
  the QA run and the ITS runs used the September nightly pairing.
- The changes made on 2026-11-10 after a tester's first-day report: the clock cap in an
  elevated and in a standard shell (`clock_cap.ps1` exit codes, the launcher's cap line,
  `-RequireCap`), `gpu_probe.ps1`, the DLL helper without GStreamer, the clone without Git LFS,
  the Arm64 interpreter request with the uv directories under `$ROOT`, the preflight's progress
  output and `--output`.
- A conversion from a release downloaded without the generator-only folders
  (`windows-setup.md` § The model).
- The synthetic smoke set's loss curve on both units.
- The model conversion (Variant 2) on the pinned commit 985d116: it last ran on e682c47.
- The eval cases in `eval/eval.config` require a Windows-on-Arm host; the Linux evaluator
  runners cannot execute them. Until a Windows runner class exists, a manual run with the
  report attached to the PR is the validation.
