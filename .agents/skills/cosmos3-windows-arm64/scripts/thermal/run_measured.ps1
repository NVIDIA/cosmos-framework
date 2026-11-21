# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Run any command line under the GPU health gate + 1 s sampler and write a receipt (validated on an RTX Spark N1X).
# Usage: powershell -ExecutionPolicy Bypass -File run_measured.ps1 -Tag <name> -CommandLine '"<venv>\Scripts\python.exe" <script>.py --arg'
#        [-EvidenceDir .\evidence] [-WorkingDirectory .] [-Python <venv>\Scripts\python.exe] [-TimeoutSec 3600] [-MaxStartTempC 62] [-Spinner] [-Env @{K='V'}]
# -Python (or the TAO_WINDOWS_PYTHON environment variable) is the interpreter used for the spinner and the sample summary.
param([Parameter(Mandatory)][string]$Tag, [Parameter(Mandatory)][string]$CommandLine,
      [string]$EvidenceDir = (Join-Path (Get-Location) 'evidence'), [string]$WorkingDirectory = (Get-Location).Path,
      [string]$Python = $env:TAO_WINDOWS_PYTHON,
      [int]$TimeoutSec = 3600, [int]$MaxStartTempC = 62, [switch]$Spinner, [hashtable]$Env = @{}, [string]$EnvList = '')
# -EnvList "K=V;K2=V2" sets environment variables for the child process (usable from -File invocations, where -Env cannot be passed)
if ($EnvList) { foreach ($kv in $EnvList.Split(';')) { if ($kv -match '^([^=]+)=(.*)$') { $Env[$Matches[1]] = $Matches[2] } } }
$ErrorActionPreference = 'Stop'
$ev = $EvidenceDir; $scripts = $PSScriptRoot
$pyarm = if ($Python) { $Python } else { 'python' }
$q = 'timestamp,clocks_event_reasons.active,temperature.gpu,clocks.sm,memory.used,power.draw'
New-Item -ItemType Directory -Force -Path $ev | Out-Null
$waited = 0
while ($true) {
  $h = & nvidia-smi -i 0 --query-gpu=$q --format=csv,noheader,nounits
  $temp = [int](($h -split ',\s*')[2])
  if ($temp -le $MaxStartTempC) { break }
  if ($waited % 60 -eq 0) { Write-Output "cooling: GPU $temp C > $MaxStartTempC C (waited $waited s)" }
  Start-Sleep 10; $waited += 10
}
$before = & nvidia-smi -i 0 --query-gpu=$q --format=csv
$before | Set-Content "$ev\$Tag.health_before.txt" -Encoding UTF8
"cooldown_wait_s: $waited" | Add-Content "$ev\$Tag.health_before.txt"
if ($Spinner) {
  $sp = (& cmd.exe /d /c "`"$pyarm`" $scripts\gpu_spinner.py 10 2>&1" | Select-String 'spinner done').Line
  "spinner (10 s BF16 matmul): $sp" | Add-Content "$ev\$Tag.health_before.txt"; Write-Output $sp
  if ($sp -match '\(([\d.]+) TFLOP/s') { if ([double]$Matches[1] -lt 35) { throw "GPU throughput degraded ($sp); refusing to time this run" } }
  Start-Sleep 20
}
foreach ($k in $Env.Keys) { Set-Item -Path "env:$k" -Value $Env[$k] }
$stdout = "$ev\$Tag.stdout.log"; $stderr = "$ev\$Tag.stderr.log"; $csv = "$ev\$Tag.samples.csv"; $stop = "$ev\$Tag.stop"
Remove-Item $stop -ErrorAction SilentlyContinue
$cmdline = '/d /c "' + $CommandLine + '"'
$t0 = Get-Date
$p = Start-Process -FilePath cmd.exe -ArgumentList $cmdline -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -NoNewWindow
$null = $p.Handle
$sampler = Start-Process -FilePath powershell.exe -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$scripts\gpu_sampler.ps1", '-OutCsv', $csv, '-RootPid', $p.Id, '-StopFile', $stop, '-IntervalSec', '1') -PassThru -WindowStyle Hidden
$timedOut = $false
if (-not $p.WaitForExit($TimeoutSec * 1000)) { $timedOut = $true; & taskkill.exe /PID $p.Id /T /F | Out-Null; $p.WaitForExit(15000) | Out-Null }
$p.WaitForExit(); $wall = ((Get-Date) - $t0).TotalSeconds
New-Item $stop -ItemType File -Force | Out-Null; $sampler.WaitForExit(30000) | Out-Null
$after = & nvidia-smi -i 0 --query-gpu=$q --format=csv
$after | Set-Content "$ev\$Tag.health_after.txt" -Encoding UTF8
$summary = $null
try { $summary = (& $pyarm -I "$scripts\summarize_samples.py" $csv | Out-String | ConvertFrom-Json) } catch { Write-Output "summary failed: $_" }
$receipt = [ordered]@{ Tag = $Tag; CommandLine = $CommandLine; WorkingDirectory = $WorkingDirectory; Env = $Env; Start = $t0.ToString('o')
  WallSeconds = [math]::Round($wall, 1); ExitCode = $p.ExitCode; TimedOut = $timedOut; CooldownWaitSeconds = $waited
  HealthBefore = ($before -join ' | '); HealthAfter = ($after -join ' | ')
  Stdout = $stdout; Stderr = $stderr; Samples = $csv; Summary = $summary }
$receipt | ConvertTo-Json -Depth 6 | Set-Content "$ev\$Tag.receipt.json" -Encoding UTF8
Write-Output "RUN $Tag wall=$([math]::Round($wall,1))s exit=$($p.ExitCode) timedOut=$timedOut"
Get-Content $stdout -ErrorAction SilentlyContinue | Select-String 'TIMING |RESULT |COOLDOWN|Error|Traceback|rror:' | Select-Object -Last 30 | ForEach-Object { $_.Line }
Get-Content $stderr -ErrorAction SilentlyContinue | Select-String -NotMatch 'pathfinder|_load_dll' | Select-Object -Last 25 | ForEach-Object { $_.Line }
