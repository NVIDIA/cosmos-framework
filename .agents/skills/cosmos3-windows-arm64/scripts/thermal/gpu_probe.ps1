# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# gpu_probe.ps1 - eight-second GPU health probe for an RTX Spark: a bf16 8192^2 matmul loop on the otherwise idle GPU under a
# clock cap, sampled by nvidia-smi once a second. Healthy N1X: about 70 TFLOP/s at the 1600 MHz cap (1599 MHz, 95 %, 39 W).
# Latched low-power state (seen four times in November 2026, each within seconds of a pacer resume or a paced run's end):
# 663 MHz, 5 to 7 W, 7 to 10 TFLOP/s, cleared only by a reboot. Run it before every job and after any run that paused.
# Usage: powershell -ExecutionPolicy Bypass -File gpu_probe.ps1 [-Seconds 8] [-Python <venv>\Scripts\python.exe] [-CapMHz 1600]
#        [-FloorTflops 30] [-LogPath <file>]
# -Python defaults to TAO_WINDOWS_PYTHON, then python on PATH. The cap needs an elevated shell; in a standard shell the probe
# says so and runs uncapped. An uncapped reading is not comparable with the capped reference and can be lower: on one healthy
# N1X it was 40 to 42 TFLOP/s at about 1500 MHz and 37 W (nvidia-smi -q showed the Reliability clocks event reason active)
# against 70 to 72 capped in the same minutes. The 30 TFLOP/s floor still separates either reading from a latched GPU at about 7.
# -CapMHz 0 skips the cap. Exit code: 0 healthy, 3 degraded, otherwise the interpreter's error.
param([double]$Seconds = 8, [string]$Python = $env:TAO_WINDOWS_PYTHON, [int]$CapMHz = 1600, [double]$FloorTflops = 30,
      [string]$LogPath = (Join-Path (Get-Location) 'evidence\clock_cap.log'))
$py = if ($Python) { $Python } else { 'python' }
$cap = Join-Path $PSScriptRoot 'clock_cap.ps1'
function Stamp { (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }
$idle = & nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,pstate,utilization.gpu,power.draw,memory.used --format=csv,noheader
Write-Output "[$(Stamp)] probe: idle snapshot: $idle"
$capApplied = $false
if ($CapMHz -gt 0) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File $cap -Apply $CapMHz -LogPath $LogPath | Out-Null
  $capExit = $LASTEXITCODE
  $capApplied = ($capExit -eq 0)
  if ($capApplied) { Write-Output "[$(Stamp)] cap $CapMHz MHz applied" }
  else { Write-Output "[$(Stamp)] cap NOT applied (nvidia-smi exit $capExit; needs an elevated shell): probing uncapped" }
} else { Write-Output "[$(Stamp)] no cap requested: probing uncapped" }
& $py -I (Join-Path $PSScriptRoot 'gpu_probe.py') $Seconds $FloorTflops
$probeExit = $LASTEXITCODE
if ($capApplied) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File $cap -Restore -LogPath $LogPath | Out-Null
  if ($LASTEXITCODE -eq 0) { Write-Output "[$(Stamp)] default clocks restored" }
  else { Write-Output "[$(Stamp)] clock reset FAILED (nvidia-smi exit $LASTEXITCODE): run clock_cap.ps1 -Restore from an elevated shell" }
} else { Write-Output "[$(Stamp)] no clock cap was applied; clocks unchanged" }
exit $probeExit
