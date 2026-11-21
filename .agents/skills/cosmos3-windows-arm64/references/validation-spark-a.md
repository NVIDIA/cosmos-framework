# Validation on Spark A, 2026-11-10: the skill's blocks as written, first attempt

Skill commit under test: `1655bc2d2a27a627718ad509e52e86fc5f87abf8` (branch `skill/cosmos3-windows-arm64`).
Method: every documented block run literally and in order in a fresh root (`C:\cosmos3-skillval` in place of
`C:\cosmos3`, nothing else substituted), in Windows PowerShell 5.1, each block as one call. A self-correction
counts against the skill. One run is one run: the numbers below are single measurements.

Erratum, 2026-11-17: the runs of this report made after 16:26 on 2026-11-10 used framework commit
`0e84637`, not `e682c47` as written below. The two differ in one file,
`cosmos_framework/checkpoint/dcp_test.py`, a unit test that no block of the skill executes, so no
result here changes. Block 2 has since moved its pin to `985d116` (`validation.md` § Lifecycle on PR #303).

## The box

| Item                           | Value                                                                                                                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Machine                        | RTX Spark N1X, 128 GB, Windows 11 Enterprise 26H1 on Arm, build 28000.2340                                                                                                                                                                                   |
| GPU line (`nvidia-smi`)        | `NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)`, WDDM, 8128 MiB dedicated segment shown                                                                                                                                                                 |
| Driver                         | 616.08 (CUDA UMD 13.4)                                                                                                                                                                                                                                       |
| CUDA budget (preflight)        | 101.5 GiB total, 101.3 GiB free before the allocation                                                                                                                                                                                                        |
| Free disk on `C:` at the start | 90.7 GB (the stage asks for about 150 GB; the raw release and every checkpoint were deleted as soon as they had been used)                                                                                                                                   |
| Session                        | elevated (Administrator role `True`); the standard level was produced with `runas /trustlevel:0x20000`, where the same test prints `False` and `nvidia-smi -lgc` exits 4                                                                                     |
| Git / Git LFS                  | `git version 2.55.0.windows.3`; `git-lfs/3.7.1` works                                                                                                                                                                                                        |
| uv                             | `uv 0.12.9 (aarch64-pc-windows-msvc)`, `uv.exe` machine field `AA64` (Arm64)                                                                                                                                                                                 |
| Agent app                      | not packaged: `Get-AppxPackage *Claude*` returns nothing; the agent is the CLI, `%APPDATA%` is the ordinary folder                                                                                                                                           |
| FFmpeg                         | no `ffmpeg` on `PATH`; an FFmpeg 9 winarm64 shared build exists in a tools folder and was used read-only for one comparison                                                                                                                                  |
| GStreamer                      | installed: `C:\Program Files\gstreamer\1.0\msvc_arm64\bin` is on the machine `PATH`                                                                                                                                                                          |
| `powershell.exe`               | 5.1.28000.2336                                                                                                                                                                                                                                               |
| `HF_TOKEN`                     | unset                                                                                                                                                                                                                                                        |
| GPU state                      | `nvidia-smi -q` showed the `Reliability` clocks event reason active for the whole session (last boot 2026-11-08 after an unclean restart). Capped probes were healthy throughout; see "Elevation, clock cap and probe" for what it does to uncapped readings |

Not exercised on this box because of what it is: an x64 `uv.exe`, a packaged (MSIX) agent app with a redirected
`%APPDATA%`, a Git without `git-lfs` (simulated instead, see below), and a machine with no GStreamer (simulated
for the helper and for the decode test).

## Variant 1 Block 1 as written

First attempt: **pass**. The block took 377.7 s as one call and ended with exit code 0.

| Check                                                 | Expected                                       | Result | Output line                                                                                                                                                              |
| ----------------------------------------------------- | ---------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv python install cpython-3.13-windows-aarch64-none` | under `$ROOT\uv-python`, not under `%APPDATA%` | pass   | `+ cpython-3.13.15-windows-aarch64-none (python3.13.exe)`; the directory is `$ROOT\uv-python\cpython-3.13.15-windows-aarch64-none`; `%APPDATA%\uv\python` does not exist |
| `uv python find cpython-3.13-windows-aarch64-none`    | one path, `$ARM_PY` not empty                  | pass   | `C:\cosmos3-skillval\uv-python\cpython-3.13-windows-aarch64-none\python.exe`                                                                                             |
| the ARM64 assertion                                   | `ARM64 3.13.x`                                 | pass   | `ARM64 3.13.15`                                                                                                                                                          |
| torch pins                                            | `2.15.0+cu134` family, binary only             | pass   | `Successfully installed ... torch-2.15.0+cu134 torchaudio-2.11.0+cu134 torchvision-0.29.0+cu134` from `cp313-cp313-win_arm64` wheels                                     |
| preflight `--allocation-gib 72 --output`              | `"status": "PASS"`, file equals stdout, exit 0 | pass   | `"status": "PASS"`; the 494 characters of JSON on stdout are identical to `preflight.json`; exit code 0                                                                  |

Preflight report: `torch_import_seconds` 6.1, `cuda_init_seconds` 0.1, `allocation_probe.elapsed_seconds` 113.6,
`cuda_total_gib` 101.5 (`cuda_free_before_gib` 101.3, `cuda_free_after_gib` 29.3). The preflight call itself took
128 s. The claim of about 1.5 s per GiB held (1.58 s per GiB). The claim of more than five minutes for a first run
did not hold for the preflight call on this unit (2 min 8 s, with the torch files still in the file cache from the
install); the whole block took 6 min 18 s, of which the torch download and install were 4 min.

Stream checks with `--allocation-gib 0`:

- (a) Plain call: stdout parses as JSON by itself (`status` PASS); the two `preflight:` lines arrive on stderr.
- (b) The same call with `2>&1` in a script that sets `$ErrorActionPreference = 'Stop'`: the script ends with exit
  code 1 at the first progress line, which PowerShell 5.1 raises as a terminating `NativeCommandError`
  (`python.exe : preflight: importing torch (...)`); no report is printed. With the default preference the call
  returns 17 objects (2 `ErrorRecord`, 15 strings), exit code 0, and the merged text does not parse. The skill said
  only that merging "breaks the JSON"; the wording was corrected (`fcf111f`).

## TorchCodec for the stable torch

Both candidates were installed with `--no-deps` into the Block 1 venv (torch `2.15.0+cu134`), the DLL helper was
run, and eight frames of a local 1080p H.264 clip were decoded with `VideoDecoder(..., device='cuda')`. The second
run of each wheel had no GStreamer directory in the `.pth` file (torch `lib` only) and the GStreamer directories
removed from that process's `PATH`; no FFmpeg is on the machine `PATH`.

| id  | Wheel (cp313, win_arm64)   | Bundled FFmpeg       | Import       | Machine as it is                                                                 | No GStreamer registered, FFmpeg and GStreamer off `PATH` |
| --- | -------------------------- | -------------------- | ------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------- |
| A   | `0.18.0.dev20261005+cu134` | 8 (`avcodec-62.dll`) | ok, no error | `torch.Size([8, 3, 1080, 1920])` on `cuda:0`; 0.23 s to the frames, 4.0 s import | same shape and device; 0.18 s, 3.6 s import              |
| C   | `0.18.0.dev20261105+cu134` | 9 (`avcodec-63.dll`) | ok, no error | `torch.Size([8, 3, 1080, 1920])` on `cuda:0`; 0.22 s, 4.1 s import               | same shape and device; 0.19 s, 3.8 s import              |

Decision: A, `torchcodec 0.18.0.dev20261005+cu134`, is the wheel for Block 2 (the first of A, C that decodes on
CUDA without GStreamer or FFmpeg on the machine path). C works as well. The fallback venv was not needed and the
pins stay as they are. The wheels are self-contained as the skill says: `torchcodec.libs` carries the FFmpeg,
libjpeg, libpng, zlib and nvJPEG libraries. Wheel A is the one that bundles FFmpeg 8; C bundles FFmpeg 9.

How the helper was isolated: setting `$env:ProgramFiles` in the calling shell, empty or pointing at an empty
directory, has no effect on this machine. Windows rewrites `ProgramFiles` when it creates a process, the child
still sees `C:\Program Files`, and the helper registers GStreamer. The variable was therefore replaced inside the
interpreter before the helper ran; that run printed the note and exited 0.

## Variant 1 Block 2 as written

First attempt: **fail on the exit code**, with every expected line printed. Wall time 380.3 s.

- The three verification lines: `2.15.0+cu134 0.18.0.dev20261005+cu134 True`; `reasoner config imports without
  pyarrow`; `usage: train.py [-h] --sft-toml SFT_TOML [--dryrun] [--attach_vscode_debugger]` and two more lines.
- Framework: `windows-compat` at `e682c47a241650d27e912b2d90336ef44cc2ffd3`, installed editable without
  dependencies; working tree clean; the four `filter.lfs.*` settings are stored in the clone.
- The defect: the last line, `& $PY -m cosmos_framework.scripts.train --help | Select-Object -First 3`, leaves
  `$LASTEXITCODE` at -1 because `Select-Object` closes the pipe; a harness reports the block as failed. Fixed by
  putting the call in parentheses (`002d58c`); the block re-run from the start ended with exit code 0 in 29.9 s.
- The block needs a long timeout too: `pip install -r` alone took 320 s. The skill asked for one only on the
  preflight (`4269bb6`).
- `pip check` after the block reports three lines, all expected on this platform: obstore and pyarrow not
  installed, and `av<19` not met because PyAV 19.0.2 is installed (pulled in by `qwen-vl-utils`).

## Git LFS and DLL helper checks

Git LFS, with `GIT_CONFIG_GLOBAL` pointing at a scratch file whose LFS filter names a missing program and
`GIT_CONFIG_NOSYSTEM=1` (the real configuration was not touched):

| Clone                                                                                                 | Exit code | What happened                                                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| plain `git clone` (control)                                                                           | 128       | `git-lfs-missing: command not found`, `smudge filter lfs failed`, `Clone succeeded, but checkout failed`                                                                                            |
| the skill's clone line                                                                                | 0         | clean `git status`; `git config --local --list` shows `filter.lfs.smudge=`, `filter.lfs.process=`, `filter.lfs.clean=`, `filter.lfs.required=false`, so PowerShell passed the four arguments intact |
| `git fetch origin windows-compat; git checkout -B windows-compat origin/windows-compat` in that clone | 0         | HEAD `e682c47`, clean status; the 86 LFS-tracked paths stay as 132-byte pointer files                                                                                                               |

DLL helper:

| Case                                                                         | Exit code                   | Output                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dry-run` on the real venv                                                 | 0                           | `would write ...\_win_tao_dlldirs.pth`, then torch `lib`, the CUDA 13.4 `bin\arm64`, the cuDNN `bin\arm64` and GStreamer's `msvc_arm64\bin`. It lists what it found and does not label which entries are optional                                     |
| the same with nothing under `ProgramFiles` (replaced inside the interpreter) | 0 with `--force`; 2 without | with `--force`: torch `lib` only, then `note: no GStreamer directory registered (optional). ...`. Without `--force` a dry run on a venv that already has a different `.pth` stops with `refusing to replace different DLL bootstrap ...; use --force` |
| a bare `uv venv` without torch                                               | 2                           | `required DLL directory is missing: ...\bare-venv\Lib\site-packages\torch\lib (install torch into this venv first)`                                                                                                                                   |
| the bare venv with nothing under `ProgramFiles`                              | 2                           | before the fix: `no DLL directories were found`, which does not name torch `lib`; this is the state of a participant's machine. Fixed (`441be41`), now the same message as the row above                                                              |

## Elevation, clock cap and probe, as measured

The elevated column is this session; the standard column is one script per check started with
`runas /trustlevel:0x20000`, inside which the elevation test printed `False`.

| Check                                              | Elevated                                                                                                                                                                                                                                                                                                                                                             | Standard shell                                                                                                                                                                                                                                        |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clock_cap.ps1 -Apply 1400`, then `-Restore`       | exit 0 and 0. `lgc 1400 : GPU clocks set to "(gpuClkMin 1400, gpuClkMax 1400)"`, then `rgc: All done.` The read-back lines do not show the cap: they print the momentary SM clock (2353 MHz just after the apply at idle, 1397 MHz after the restore) and `clocks.max.sm` stays 3003 MHz. Under load the clock sits at the cap (1384 to 1397 MHz in the capped runs) | exit 4 and 4; `clock cap 1400 MHz REFUSED (nvidia-smi exit 4). ...` and `clock reset REFUSED (nvidia-smi exit 4). ...`; clocks unchanged                                                                                                              |
| `gpu_probe.ps1 -Python $PY`                        | `cap 1600 MHz applied`, 70.9 TFLOP/s at 1599 MHz and 38 W, `probe verdict: HEALTHY`, exit 0, `default clocks restored` (the twelve other capped probes of the session, one before each GPU job and one at the end: 70.5 to 72.8)                                                                                                                                     | `cap NOT applied (nvidia-smi exit 4; needs an elevated shell): probing uncapped`, **41.8 TFLOP/s** at 1443 to 1540 MHz and 37 W, `HEALTHY`, exit 0, `no clock cap was applied; clocks unchanged`                                                      |
| `train_lora.ps1 ... -RequireCap` (smoke arguments) | runs: job `mem_160` was started with `-RequireCap`, printed `clock cap 1400 MHz applied` and finished with exit 0                                                                                                                                                                                                                                                    | exit 1 after 1.2 s with `-RequireCap: clock cap NOT applied (nvidia-smi exit 4; changing clocks needs an elevated shell): uncapped, pacer only`; no `thermal_pacer` process and no trainer process afterwards; the `cap:` line is in the command file |

The uncapped probe reading is lower than the capped one, not higher as the script's comment said. It is not the
privilege level: an elevated probe with `-CapMHz 0` read 39.7 TFLOP/s. `nvidia-smi -q` showed
`Clocks Event Reasons: Reliability: Active` (flag `0x400`) throughout. Comment and thermal reference corrected
(`c0d7385`).

`clock_cap.log` of a refused call, as written by the launcher. The refusal text arrives clean under PowerShell
5.1, with no error-record wrapping; the driver line is doubled because the query lists the NPU as a second device:

```
[2026-11-10T15:26:27] driver: 616.08
616.08  readback(sm,max.sm,app.gfx,power.limit,default,max,min,temp): 1072 MHz, 3003 MHz, [Requested functionality has been deprecated], [N/A], [N/A], [N/A], [N/A], 39
[2026-11-10T15:26:27] lgc 1400 : The current user does not have permission to change clocks for GPU 0000000F:01:00.0.
Terminating early due to previous errors.
[2026-11-10T15:26:27] after apply: 1215 MHz, 3003 MHz, [Requested functionality has been deprecated], [N/A], [N/A], [N/A], [N/A], 39
[2026-11-10T15:26:27] clock cap 1400 MHz REFUSED (nvidia-smi exit 4). Changing clocks needs an elevated shell; the thermal pacer works without it.
```

## Per block, first attempt

| Block                                      | First attempt        | Wall time             | Expected line                                                                                                | What was seen                                                                                                                                                                                                                           | Error, fix, owner                                                                                                                                                                                                                                                                                        |
| ------------------------------------------ | -------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Variant 1 Block 1                          | pass                 | 377.7 s               | `"status": "PASS"`                                                                                           | `"status": "PASS"`, exit 0                                                                                                                                                                                                              | none                                                                                                                                                                                                                                                                                                     |
| Variant 1 Block 2                          | **fail** (exit code) | 380.3 s               | three verification lines                                                                                     | all three lines, then exit code -1                                                                                                                                                                                                      | `Select-Object -First 3` cut the pipe. Fix `002d58c`, timeout note `4269bb6`; re-run 29.9 s, exit 0. Owner: skill                                                                                                                                                                                        |
| Variant 2 (download, convert)              | pass                 | 4383.3 s              | 750 tensors, 4 shards, 16.3 GiB                                                                              | `overlaid 750/750 Cosmos3 tensors`, 4 shards, `16.3 GiB`, exit 0                                                                                                                                                                        | no error. 63 of the 73 minutes were the converter's own fetch of the complete Qwen3-VL repository at about 5 MB/s; the release itself came in 5 minutes. Text corrected (`61777d6`). Owner of the text: skill; of the behaviour: framework                                                               |
| Variant 3, `smoke_20`, elevated            | **fail** on line 1   | 0.7 s                 | `wrote 10 clips ...`                                                                                         | `FileNotFoundError: [WinError 2] The system cannot find the file specified` from `make_smoke_set.py`                                                                                                                                    | no `ffmpeg` executable on the machine. Fix `17ddd05` (PyAV fallback). Re-run from the start: pass, 287.8 s. Owner: skill. In that first attempt the launcher then stopped on the model path, because the conversion was still running; that second message is the validation's sequencing, not the skill |
| Variant 3, `smoke_20`, re-run              | pass                 | 287.8 s (job 280.9 s) | the launcher lines in order, exit 0                                                                          | `clock cap 1400 MHz applied`, `pacer armed (pause 84 C, resume 78 C, kill 87 C)`, `RUN train_smoke_20 wall=280.9s exit=0 timedOut=False`, `default clocks restored`, `run directory: ...`, `receipt: exit 0, 280.9 s wall, peak 51.0 C` | the expected log strings `train/avg_loss` and `completed successfully` are not in the logs at this framework commit. Fix `97f7b7f`. Owner: skill                                                                                                                                                         |
| Variant 3, `smoke_std`, standard shell     | pass                 | 245.6 s (job 240.3 s) | `clock cap NOT applied (...): uncapped, pacer only`; completed; `no clock cap was applied; clocks unchanged` | exactly those lines, `receipt: exit 0, 240.3 s wall, peak 60.0 C`                                                                                                                                                                       | none                                                                                                                                                                                                                                                                                                     |
| Variant 3, `smoke_nocap`, `-ClockCapMHz 0` | pass                 | 234.1 s               | `no clock cap requested`                                                                                     | `no clock cap requested: uncapped, pacer only`, exit 0, `no clock cap was applied; clocks unchanged`                                                                                                                                    | none                                                                                                                                                                                                                                                                                                     |
| Variant 4 on `smoke_20`                    | pass                 | 306.9 s               | 750 tensors, `0 lora keys`                                                                                   | `750 tensors; 0 lora keys`, 5 shards, 16.3 GiB                                                                                                                                                                                          | none; needs a long timeout (in `97f7b7f`)                                                                                                                                                                                                                                                                |
| Variant 5 on the smoke validation set      | pass                 | 35.9 s                | one row per record, the metrics summary                                                                      | 4 rows, `Overall accuracy: 0.5000`, `metrics.json` written, exit 0                                                                                                                                                                      | the row keys and paths in the skill were not the ones written (in `97f7b7f`). All four answers were `no`, see below                                                                                                                                                                                      |

## Smoke numbers

`smoke_20` checks in order: `clock cap 1400 MHz applied`; `cap: clock cap 1400 MHz applied` in
`train_smoke_20.command.txt`; `pacer armed`; `LoRA injection successful: 144 modules wrapped, 15,335,424
trainable LoRA params`; `attn_implementation: sdpa` on both policy keys in the run's `config.yaml` (the nested
`model.config.policy` and the top-level `policy`; the stdout log itself carries no attention line); frames and
pixel budget: `num_video_frames: '8'` and `video_max_pixels: '1179648'` in `config.yaml`,
`COSMOS_VIDEO_MAX_PIXELS=1179648` and `COSMOS_VIDEO_NUM_FRAMES=8` in the receipt, `video_grid_thw [4, 18, 30]`
(8 frames at 288x480) and 608 or 624 tokens per record in the log; synchronous save,
`Checkpoint save completed: Time taken: 119.52 seconds`, 35.3 GB (32.8 GiB); `default clocks restored`; the
run-directory and receipt lines; exit code 0.

The twenty loss values of `smoke_20`:

```
1.4701 0.1967 1.3869 0.3813 0.1016 0.1137 0.0506 1.1756 0.0246 0.0450
1.3479 0.0278 1.2898 0.5282 0.0118 0.4506 1.1194 1.0640 0.0193 0.0193
```

The claim that the loss falls below 1.0 by the second epoch does not describe this: single-record steps
swing between 0.01 and 1.5 in both epochs, the epoch means are 0.495 and 0.588, and the validation loss goes from
0.446 to 0.337.

| Job           | Shell and cap              | s per step after warm-up | First step | Save    | W under load (mean / peak) | SM clock under load       | Peak C | Pacer pauses | Peak private host memory | Epoch mean loss | Validation loss |
| ------------- | -------------------------- | ------------------------ | ---------- | ------- | -------------------------- | ------------------------- | ------ | ------------ | ------------------------ | --------------- | --------------- |
| `smoke_20`    | elevated, 1400 MHz applied | 1.81 (1.66 to 1.89)      | 89.3 s     | 119.5 s | 14.0 / 15.3                | 1391 MHz                  | 51     | 0            | 83.5 GiB                 | 0.495, 0.588    | 0.446, 0.337    |
| `smoke_std`   | standard, cap refused      | 2.28 (2.09 to 2.43)      | 34.3 s     | 120.8 s | 26.1 / 29.3                | 1685 MHz mean, up to 2444 | 60     | 0            | 83.5 GiB                 | 0.490, 0.600    | 0.460, 0.337    |
| `smoke_nocap` | elevated, `-ClockCapMHz 0` | 2.21 (2.03 to 2.39)      | 25.9 s     | 123.4 s | 28.4 / 32.9                | 1805 MHz mean, up to 2346 | 61     | 0            | 83.5 GiB                 | 0.477, 0.591    | 0.464, 0.356    |

Capped against uncapped on this unit on this day: the 1400 MHz cap was 10 C cooler, drew half the power and
was also about 20 % faster per step. The uncapped runs showed the clocks event reasons `0x4` (software power
cap) as well as `0x400` (reliability); the capped run only `0x400`. Twenty steps are far too short to say
anything about pacer pauses.

Variant 5 on the four validation records with the `smoke_20` export: `no` for all four (ground truth yes, no,
no, yes), accuracy 0.5, balanced accuracy 50 %. With the result of the next section this says nothing about
the 20-step model either way.

## Memorisation run (does it learn on this stack)

Stack: cosmos-framework `windows-compat` at `e682c47a241650d27e912b2d90336ef44cc2ffd3`, torch `2.15.0+cu134`,
torchcodec `0.18.0.dev20261005+cu134`, sdpa on both policy keys. Job `mem_160`: the 16 synthetic training
records, `steps_per_epoch = 16`, `num_epochs = 10`, `max_iter = 160`, `cycle_lengths = [160]`,
`save_freq_in_epoch = 10`, elevated, `clock cap 1400 MHz applied`, `-RequireCap`.

| Quantity                                                                                     | Value                                                                                                                                      |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Mean training loss per epoch, 1 to 10                                                        | 0.5333, 0.1151, 0.0005, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000                                                             |
| Largest single-step loss in epochs 5 to 10                                                   | 0.000061                                                                                                                                   |
| Validation loss per epoch (4 held-out records)                                               | 0.147, 0.0035, 0.00015, 0.00003, 0.00002, then 0.00001                                                                                     |
| Run                                                                                          | exit 0, 570 s, 1.91 s per step (1.75 to 2.14), save 129.4 s, 14.8 W mean (16.5 peak), peak 66 C, no pacer pause, `default clocks restored` |
| Export of `epoch_10`                                                                         | `750 tensors; 0 lora keys`, 5 shards, 16.3 GiB, 305 s                                                                                      |
| Framework evaluator on `annotations_train.json`, template settings                           | **12 of 16** (accuracy 0.75, balanced 71.43 %, TP 3, FP 0, TN 9, FN 4)                                                                     |
| The same with `[model] attn_implementation = "sdpa"` (attested in the log)                   | 12 of 16, the same four                                                                                                                    |
| Plain transformers `generate` from the same export, same clips, 8 frames, same system prompt | **16 of 16** (605 or 607 prompt tokens, `video_grid_thw [4, 18, 30]`)                                                                      |

The loss criterion is met with room to spare: below 0.05 by epoch 3, where iteration 4 on 64 records reached
0.026 at epoch 5 (0.478, 0.153, 0.026, 0.000 at epochs 1, 3, 5, 10). The stack learns; there is no plateau near
0.34 and the answers are not one-class.

The evaluation criterion (16 of 16) is **not met through the Framework evaluator**. Its four misses are the four
"Is there a pedestrian on the road?" records whose answer is yes (clips 0, 2, 4, 6); it answers `no` to all eight
pedestrian questions and gets all eight stopped-vehicle questions right, while reporting a loss of 0.00014. The
exported model itself answers all sixteen correctly when driven through transformers directly. Nothing was
tuned.

### Why the evaluator disagreed (diagnosed after the report, same evening)

The evaluator shows the model a different frame than training did. One process loaded the `mem_160` export
through the evaluator's own runtime class and built the input for each of the 16 records five ways:

| Input                                                                                            | Frame fed to the model                 | Video tokens | Timestamps in the prompt | Correct  |
| ------------------------------------------------------------------------------------------------ | -------------------------------------- | ------------ | ------------------------ | -------- |
| Training input (`VideoSFTProcessor` as the run's `config.yaml` builds it), cut before the answer | 480x288                                | 540          | 0.1, 0.4, 0.7, 1.0 s     | 16 of 16 |
| Evaluator path as shipped (also through `runtime.generate_tasks`: same answers)                  | **512x256**                            | 512          | **0.2, 1.2, 2.2, 3.2 s** | 12 of 16 |
| Evaluator path, timestamps corrected only                                                        | 512x256                                | 512          | 0.1, 0.4, 0.7, 1.0 s     | 12 of 16 |
| Evaluator path, frame size corrected only                                                        | 480x288, pixels identical to training  | 540          | 0.2, 1.2, 2.2, 3.2 s     | 16 of 16 |
| Evaluator path, both corrected                                                                   | identical to training, token for token | 540          | 0.1, 0.4, 0.7, 1.0 s     | 16 of 16 |

The frame size alone explains the four misses. Both differences come from the list-of-frames branch of
`qwen_vl_utils.fetch_video` (0.0.14, the newest release and the one the dependency file pins), which the
evaluator reaches through `CosmosFrameworkRuntime._prepare_transformers_batch`:

- **Frame size.** `fetch_video` passes the merged factor (patch 16 x merge 2 = 32) to `fetch_image` in the
  patch-size argument, and `fetch_image` multiplies by the merge size again. Each decoded frame is therefore
  resized to multiples of 64 before the video-level resize: 288 rounds down to 256 and 480 up to 512. Training
  hands its frames to the Hugging Face video processor, which rounds to multiples of 32 and leaves 480x288
  alone. The pedestrian is 22 pixels tall on a 288-pixel frame, and the squashed frame moves it.
- **Timestamps.** `_task_conversation` puts the sampled frame rate (6.56) on the video item as `fps`, but that
  branch reads `sample_fps` and falls back to 2.0. The prompt's frame timestamps are then 0.2 to 3.2 s where
  training saw 0.1 to 1.0 s. It did not change an answer here; a question about timing could be affected.

Excluded by measurement: decoding (both sides decode the same eight 480x288 frames with the same reader and
indices), the patch-embed compatibility step (not applied: CUDA capability 12.1, it applies to 8.0 only), the
attention keys, the exported processor.

Which sources are affected, measured with synthetic clips through both paths at the template's settings
(8 frames, 147456 pixels per frame):

| Source                                 | Training frame | Evaluator frame |                  |
| -------------------------------------- | -------------- | --------------- | ---------------- |
| 448x256, 512x256, 384x256              | unchanged      | unchanged       | identical pixels |
| 854x480                                | 512x256        | 512x256         | same size        |
| 480x288 (the smoke set)                | 480x288        | 512x256         | differs          |
| 640x360, 1024x576, 1280x720, 1920x1080 | 512x288        | 512x256         | differs          |
| 640x480                                | 416x320        | 384x320         | differs          |
| 512x320                                | 480x288        | 448x256         | differs          |

The rule: the two agree when the frame training uses is a multiple of 64 in both directions. Every common
16:9 source misses that at this pixel budget, so a participant who fine-tunes on 720p or 1080p clips is
evaluated on vertically squashed frames. Scripts and logs: `step10_eval_input_diagnosis.py`,
`step10_frame_size_check.py` and `evidence/step10/` in the hand-over folder.

## Reduced release (the tester's tip)

**Works.** A view of the raw release built with hard links without `vae\`, `sound_tokenizer\`, `assets\` and
`images\` (38 files, 29.36 GiB linked; 30 files, 3.22 GiB left out) converted with exit code 0 in 115 s:
750 tensors, 4 shards, 16.3 GiB, and all 16 output files are SHA-256-identical to the conversion from the full
release. The skill may recommend the reduced download. The saving is the 3.2 GiB of download only: the converter
does not read those folders from the release at all, it fetches its own Wan2.2 VAE (2.8 GB) and sound tokenizer
(2 GB) from the Hub in both cases, next to the complete Qwen3-VL repository (16.3 GiB).

## Fixes committed on `skill/cosmos3-windows-arm64`

The fixes were made on Spark A on the working branch `skill/cosmos3-windows-arm64-spark-a` and carried
over to `skill/cosmos3-windows-arm64` with identical content. The ids below are those of
`skill/cosmos3-windows-arm64`.

| Commit    | Title                                                                                                  | Found in                    |
| --------- | ------------------------------------------------------------------------------------------------------ | --------------------------- |
| `002d58c` | validated-fix: Block 2 usage check ended with exit code -1                                             | Block 2                     |
| `4269bb6` | validated-fix: say that Block 2 needs the ten-minute timeout as well                                   | Block 2                     |
| `441be41` | validated-fix: DLL helper names torch lib for a venv without torch on a clean machine                  | helper check                |
| `17ddd05` | validated-fix: smoke set no longer needs an ffmpeg executable (PyAV fallback, av pinned)               | Variant 3                   |
| `fcf111f` | validated-fix: say what 2>&1 does to the preflight under Windows PowerShell 5.1                        | Block 1 stream check        |
| `c0d7385` | validated-fix: an uncapped probe reads lower, not higher; record capped against uncapped               | probe, smoke runs           |
| `61777d6` | validated-fix: Variant 2 downloads, time and disk as measured; reduced release confirmed               | Variant 2                   |
| `97f7b7f` | validated-fix: expected output of Variants 3 to 5 as framework e682c47 prints it; timeouts             | Variants 3 to 5             |
| `ed44095` | validated-fix: the evaluator does not reproduce 480x288; say which frame sizes it evaluates faithfully | memorisation run, diagnosis |

After `17ddd05` Block 2 and Variant 3 were re-run from the start and passed. The text-only commits were not
followed by a re-run.

## Framework-owned findings (recorded, not patched)

1. **Evaluator input differs from training input**: `cosmos-reasoner-evaluate` resizes each frame to multiples
   of 64 where training uses multiples of 32, and stamps the frames at 2 fps instead of the sampled rate
   (§ Why the evaluator disagreed). On a memorised export it scores 12 of 16; with the frame size matched,
   16 of 16. Evaluator accuracies are lower bounds for every source whose training frame is not a multiple of
   64 in both directions, which includes the smoke set and all common 16:9 sizes. A fix belongs in
   `cosmos_framework/inference/reasoner/runtime.py`: for frames the framework has already decoded, call the
   processor the way training does (frames, `fps`, `do_sample_frames=False`, the processor's own resize)
   instead of going through `qwen_vl_utils.process_vision_info`; passing `sample_fps` alongside `fps` fixes
   the timestamps only. Not patched here.
2. **The converter's hidden downloads**: `convert_model_to_vlm_safetensors` pulls the whole
   `Qwen/Qwen3-VL-8B-Instruct` repository, the Wan2.2 VAE and the AVAE sound tokenizer through a nested
   `uv run ... hf download` into the default Hugging Face cache, ignoring `vae\` and `sound_tokenizer\` in the
   release it was given. 63 minutes here. The nested `uv` uses its default folders, not the ones Block 1 sets.
3. **Log vocabulary**: per-step loss is logged as `Iteration N: Hit counter: N/50 | Loss: ...` for the first
   fifty steps only; later steps and the completion message exist only in `status.json`.
4. **`eval/eval.config`** (not edited, as instructed) does not fit this framework commit in two places: the
   `nano-lora-smoke` prompt asks for "the training log lines containing avg_loss", of which the logs have
   none, so `outputs/loss.txt` would be empty and fail its 200-byte check unless the agent turns to
   `status.json`; and `setup.ref` is `69fee1f`, while Block 2 checks out `e682c47`.

## What remains unverified

- The size of the evaluator's error on real data (finding 1). The cause is established on the 16 synthetic
  records; how many answers a squashed frame costs on a participant's own clips was not measured. A smoke set
  at 448x256 or 512x256, which both paths would treat alike, was not trained or evaluated.
- A packaged (MSIX) agent app: Block 1's uv folders, and the converter's nested `uv run`, which Block 1's
  variables do not reach.
- An x64 `uv.exe` (this machine's is Arm64) and a Git that really lacks `git-lfs` (simulated).
- A machine without GStreamer, CUDA toolkit and cuDNN: simulated for the helper and the decode test only. The
  training, export and evaluation runs here had GStreamer's `bin` registered in the `.pth` file.
- The 64 GB unit: nothing here was run on it.
- Uncapped behaviour without the `Reliability` flag: every uncapped number above was taken with it active. A
  reboot was not requested because no probe returned `DEGRADED`.
- Pacer pauses, the latched state and long-run temperatures: the longest job here was 9.5 minutes at 66 C.
- `HF_TOKEN` set: all downloads were unauthenticated.
- The three eval cases of `eval/eval.config` (Step 9) were not run. Two reasons: disk (40 GB free; the smoke
  case needs the release, a checkpoint and an export at once), and a fresh agent session's transcript cannot be
  captured from inside this session.
- The second TorchCodec candidate (`dev20261105+cu134`, FFmpeg 9) decodes, but no training or evaluation was
  run with it.

## Follow-up after the report (2026-11-11, not run on Spark A)

- Block 2 now checks out the framework commit of this validation by id (`e682c47`) and no longer the head
  of `windows-compat`, which moves while PR #303 is in review. The two Git commands were tried against the
  fork from a fresh no-LFS clone on another machine (`HEAD is now at e682c47`, exit 0, twice in a row);
  the block as a whole was not re-run on Windows.
- Block 2 and `windows-setup.md` name the wheel chosen above (A).
- `eval/eval.config`: `setup.ref` is `e682c47`, and the `nano-lora-smoke` case asks for the per-step loss
  lines the logs contain (framework-owned finding 4). The cases themselves are still not run.

## State left on the machine

Default clocks restored after every job; the last probe of the validation read 72.8 TFLOP/s, `HEALTHY`, and the
two probes before the diagnosis jobs later that evening read 70.7, `HEALTHY`, `default clocks restored`. Kept
under the root: the
converted model (17.6 GB), the `mem_160` export (17.6 GB), the venv, the logs and receipts. Deleted: the raw
release, all four checkpoints, the `smoke_20` export. Outside the root, the skill's own commands wrote to the
default Hugging Face cache: the Qwen3-VL weights (17.6 GB, kept because they take an hour to fetch) and the
Wan2.2 VAE and Cosmos3-Nano sound tokenizer entries (removed again); pip and the converter's nested uv also
used their default per-user caches. The protected venvs, source trees, model directory, driver, system PATH and
global Git configuration were not modified. One read-only use to declare: the very first probe, before Block 1
had created a venv, ran `gpu_probe.py` with the interpreter of an existing venv on the machine.
