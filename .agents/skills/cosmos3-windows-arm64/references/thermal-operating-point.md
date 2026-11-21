# Thermal limiter and GPU health on Windows laptops (RTX Spark)

A fine-tune is a sustained load on a laptop. Run every GPU job through the scripts in
`../scripts/thermal/` so that the job has a start gate, a clock cap, an in-run pacer, a
hard guard and a receipt. All numbers below were measured on one RTX Spark N1X (Windows 11
on Arm, 128 GB, driver 6xx series) between 2026-10-28 and 2026-11-09 with one GPU job at a
time; treat them as the starting point for another unit, not as its specification.

## The toolkit

| Script                                                                                                                                  | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `clock_cap.ps1 -Apply <MHz>` / `-Restore` / (no args)                                                                                   | Apply or restore the SM clock cap with `nvidia-smi -lgc` / `-rgc`; without arguments it probes which clock and power controls the driver accepts and leaves nothing applied. Changing clocks needs an elevated shell: `-Apply` and `-Restore` exit 0 when the driver accepted the request and with nvidia-smi's code (4, no permission) when it refused. The confirmation is the driver's `GPU clocks set to "(gpuClkMin N, gpuClkMax N)"` line in the log; the read-back lines show the momentary SM clock, which at idle is not the cap. |
| `run_measured.ps1 -Tag <t> -CommandLine '<cmd>' [-Python <exe>] [-EvidenceDir <dir>] [-MaxStartTempC 62] [-TimeoutSec 3600] [-Spinner]` | Waits until the GPU is at or below the start temperature, runs the command with a 1 s sampler (`gpu_sampler.ps1`), writes `<tag>.receipt.json`, `<tag>.samples.csv` (+ `.summary.json`), stdout and stderr logs, and health snapshots before and after. `-Spinner` refuses to time a run on a degraded GPU.                                                                                                                                                                                                                                |
| `thermal_pacer.ps1 -Tag <t> [-PauseC 84] [-ResumeC 78] [-LimitC 87]`                                                                    | Runs beside a measured job: suspends the whole job tree (NtSuspendProcess, descendants included) at the pause temperature, resumes it at the resume temperature, kills the tree at the limit, and writes `<tag>.pacer.json` with the pauses and paused seconds. The cap stays applied, so unpaused steps remain comparable.                                                                                                                                                                                                                |
| `gpu_probe.ps1 [-Seconds 8] [-Python <exe>] [-CapMHz 1600] [-FloorTflops 30]`                                                           | Eight-second bf16 matmul loop (`gpu_probe.py`) with the GPU otherwise idle, sampled once a second; prints TFLOP/s and a `probe verdict: HEALTHY` or `DEGRADED` line and exits 0 or 3. It tries the 1600 MHz cap, says whether the driver accepted it, and restores the default clocks if it did; in a standard shell it probes uncapped, and that reading can be lower than the capped one (40 to 42 against 70 to 72 TFLOP/s on 2026-11-10) without the GPU being at fault.                                                               |
| `gpu_spinner.py [seconds]`                                                                                                              | Sustained matmul load for diagnostics and the `-Spinner` gate.                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

Pattern for one job (the model skill's launcher does this for you):

```powershell
powershell -ExecutionPolicy Bypass -File clock_cap.ps1 -Apply 1400     # elevated shell; exit code 4 means the cap was refused
Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','thermal_pacer.ps1','-Tag','train_job1','-EvidenceDir','.\evidence'
powershell -ExecutionPolicy Bypass -File run_measured.ps1 -Tag train_job1 -Python <venv>\Scripts\python.exe -CommandLine '"<venv>\Scripts\python.exe" -m cosmos_framework.scripts.train --sft-toml=<recipe>.toml -- <overrides>'
powershell -ExecutionPolicy Bypass -File clock_cap.ps1 -Restore
```

Restore the default clocks after every job and say so in the run log. Never run two GPU jobs at once.

## Elevation: what a standard shell cannot do

`nvidia-smi -lgc` and `-rgc` answer "The current user does not have permission to change
clocks" (exit code 4) unless the shell is elevated (Run as administrator). Reading clocks,
temperature and power, the sampler, the start gate and the pacer need no elevation. The
launcher therefore tries the cap, prints `clock cap <N> MHz applied` or `clock cap NOT applied
(...): uncapped, pacer only`, records the line in the run's command file, and continues either
way (`-RequireCap` stops instead; `-ClockCapMHz 0` skips the attempt). It restores clocks only
when it applied a cap.

Uncapped, a tester's unit ran the 480x288 input at 1995 MHz, 24 to 30 W and 66 C early in a
short run. On the 128 GB unit (2026-11-10, three 20-step smoke runs, `Reliability` clocks event
reason active throughout) the uncapped runs took 2.2 to 2.3 s per step at 26 to 28 W (peak 33 W)
and reached 60 to 61 C, in a standard shell and with `-ClockCapMHz 0` alike; the run at the
1400 MHz cap took 1.8 s per step at 14 W (peak 15 W) and reached 51 C. There the cap was faster
as well as cooler. A short smoke is fine uncapped. A long run uncapped will reach the pause temperature
and cycle the pacer, and each resume is a chance of the latched state below: use an elevated
shell for anything longer than a few minutes.

## Measured operating point (RTX Spark N1X, 128 GB unit)

Start below 62 C, pacer pause 84 C / resume 78 C / kill 87 C. Every row is for a cap the
driver accepted.

Current stack (cosmos-framework main with PR #303, sdpa attention), Cosmos3 Nano LoRA SFT at
8 frames of 480x288 (clip budget 1179648 px), 2026-11-05 to 11-09:

| Clock cap                       | s per step          | W                           | GPU C                                                | Pacer                                                              |
| ------------------------------- | ------------------- | --------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ |
| 1400 MHz (the launcher default) | 1.80 (1.64 to 2.02) | 17 to 19, peak 20.1         | 54 at 6 min, 81 at 31 min, plateau 80 to 82, peak 83 | no pause in 2880 steps (113 min)                                   |
| 1600 MHz                        | 1.74                | 17 to 18 in a 100-step test | 61 after 100 steps, 84 after 15 minutes              | paused at 15 minutes; the GPU latched two seconds after the resume |

The 1600 MHz cap buys 3 % in step time and costs the run. A 100-step test does not show it:
neither cap paused in 100 steps.

Earlier stack (cosmos-framework `02b05df`, `cosmos` attention), cap 1600 MHz, 2026-10-28 to 11-04:

| Cosmos3 Nano LoRA SFT input                  | s per step  | W        | GPU C                                               | Pacer                                         |
| -------------------------------------------- | ----------- | -------- | --------------------------------------------------- | --------------------------------------------- |
| 8 frames at 128x64 (clip budget 81920 px)    | 0.5 to 0.6  | 17 to 18 | 73 to 77                                            | no pauses                                     |
| 8 frames at 480x288 (clip budget 1179648 px) | 1.05 to 1.2 | 25 to 28 | 81 to 85                                            | about 250 short pauses (9 min) per 2880 steps |
| 8 frames at 832x480                          | 2.9         | 27 to 28 | reaches the 87 C guard in about 10 minutes uncapped | not sustainable at 1600                       |

Sim-Transfer inference (33-frame 640x480 clip): 25 to 27 s at 1600 MHz, 30 to 38 W, 73 to 80 C.

Healthy probe reading: about 70 TFLOP/s at 1599 MHz and 39 W. A reading near 7 TFLOP/s at
663 MHz and 5 to 6 W is the latched low-power state below.

## Failure modes seen, and what to do

- **Hard reset without an event-log cause** (Kernel-Power 41, bugcheck 0, no WHEA or display
  driver event). Seen under uncapped load at 90 to 94 C and, twice, at idle. Reboot, reapply
  the cap, resume; the receipts tell a chain which jobs are finished. Open items are tracked
  with the platform team.
- **Latched low-power state**: the GPU stays at 663 MHz and 5 to 7 W under full load, about
  seven to ten times slower (6.7 to 10.5 TFLOP/s on the probe against 70). Seen four times
  between 2026-11-04 and 11-09, each time within seconds of a pacer resume or of the end of a
  run that had paused. `nvidia-smi -rgc` does not clear it; a reboot does. Run `gpu_probe.ps1`
  before every job, after any run that paused and after any power event. Prefer a cap at which
  the pacer never cycles over a higher cap with pauses: for the 480x288 input that is 1400 MHz.
- **Reliability clock limiter** after display-driver events (`clocks_event_reasons.active`
  non-zero at idle): every speed number taken under it is low by about 2.4x. The flag is not a
  dependable test for the latched state above; the probe is. Check the probe before any speed
  claim. Measured on 2026-11-10 with the flag at `0x400` (`Reliability: Active` in
  `nvidia-smi -q`, two days after an unclean restart): work under an accepted clock cap was not
  slowed (probe 70 to 72 TFLOP/s at the 1600 MHz cap, training 1.8 s per step at 1400 MHz);
  uncapped work was (probe 40 to 42 TFLOP/s, training 2.2 to 2.3 s per step).

## Caveats

- The pacer identifies the job tree by a regular expression over process command lines
  (`-Pattern`); extend it when a new entry point is used.
- Under WSL the pacer cannot suspend processes inside the VM; it is a native-Windows tool.
- `nvidia-smi` memory on WDDM shows the dedicated segment, not the CUDA budget; use the
  preflight script in `../scripts/preflight_windows_arm64.py` for capacity.
