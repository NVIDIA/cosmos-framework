# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# thermal_pacer.ps1 - the cooldown gate applied INSIDE one measured GPU job, plus the 87 C hard guard (validated on an RTX Spark N1X, 2026-11-03).
# Polls the GPU temperature every PollSec seconds. At PauseC it SUSPENDS every process of the job tree (python.exe whose
# command line matches Pattern, plus all descendants - the multiprocessing spawn worker that holds the TorchCodec CUDA
# context included) with ntdll NtSuspendProcess, resumes them at ResumeC, and at LimitC kills the whole tree. The SM clock
# cap is untouched, so the per-step speed of the unpaused steps stays comparable; the pauses are reported separately.
# Why the tree: the 2026-11-03 arm C trip (full-resolution input, 27 W at 1600 MHz, 61 -> 87 C in ten minutes) was caught by the
# old guard, which killed only the two pattern matches; the spawn worker (parent dead, command line "multiprocessing.spawn")
# survived with the Tee pipe handle, the launcher waited for EOF, and run_measured.ps1 waited out its 90-minute timeout.
# Exits when <EvidenceDir>\<Tag>.receipt.json appears (the wrapper finished) or MaxSeconds elapse; writes <Tag>.pacer.json
# (pauses, paused seconds, max temperature, killed) and appends kills to <EvidenceDir>\thermal_events.txt.
# Usage: powershell -ExecutionPolicy Bypass -File thermal_pacer.ps1 -Tag train_<job> [-EvidenceDir ..] [-LimitC 87] [-PauseC 84]
#        [-ResumeC 78] [-MaxSeconds 10800] [-PollSec 2]
param([Parameter(Mandatory = $true)][string]$Tag,
      [string]$EvidenceDir = (Join-Path (Get-Location) 'evidence'),
      [int]$LimitC = 87, [int]$PauseC = 84, [int]$ResumeC = 78, [int]$MaxSeconds = 10800, [int]$PollSec = 2,
      [string]$Pattern = 'cosmos_framework\.scripts\.train|cosmos_rl\.evaluation\.evaluate|cosmos_embed_native\.py')
$ErrorActionPreference = 'Continue'
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class NtProc {
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
  [DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
"@
function Stamp { (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }
function Log($m) { Write-Output "[$(Stamp)] $m" }
function Get-GpuTemp {
  $t = (& nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ("$t" -match '^\s*(\d+)') { return [int]$Matches[1] } else { return -1 }
}
function Get-JobTree {
  # roots: the trainer / evaluator python processes; then every descendant by ParentProcessId (spawn workers, helpers)
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, CommandLine
  $roots = @($all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -and ($_.CommandLine -match $Pattern) -and ($_.CommandLine -notmatch 'Get-CimInstance|thermal_pacer') } | ForEach-Object { [int]$_.ProcessId })
  $set = New-Object 'System.Collections.Generic.HashSet[int]'
  $queue = New-Object 'System.Collections.Generic.Queue[int]'
  foreach ($r in $roots) { if ($set.Add($r)) { $queue.Enqueue($r) } }
  while ($queue.Count -gt 0) {
    $parent = $queue.Dequeue()
    foreach ($c in @($all | Where-Object { $_.ParentProcessId -eq $parent })) { $cid = [int]$c.ProcessId; if ($set.Add($cid)) { $queue.Enqueue($cid) } }
  }
  return @($set)
}
function Act-Tree([int[]]$pids, [string]$what) {
  $done = @()
  foreach ($pp in $pids) {
    $h = [NtProc]::OpenProcess(0x0800, $false, $pp)   # PROCESS_SUSPEND_RESUME
    if ($h -eq [IntPtr]::Zero) { Log "  $what ${pp}: OpenProcess failed (gone?)"; continue }
    if ($what -eq 'suspend') { $rc = [NtProc]::NtSuspendProcess($h) } else { $rc = [NtProc]::NtResumeProcess($h) }
    [NtProc]::CloseHandle($h) | Out-Null
    if ($rc -eq 0) { $done += $pp } else { Log "  $what ${pp}: NTSTATUS $rc" }
  }
  return @($done)
}
$t0 = Get-Date; $paused = $false; $pausedPids = @(); $pauseStart = $null; $pauses = 0; $pausedSec = 0.0; $maxT = -1; $killed = $false
$receipt = Join-Path $EvidenceDir "$Tag.receipt.json"
Log "pacer armed for ${Tag}: pause at $PauseC C, resume at $ResumeC C, kill at $LimitC C (evidence $EvidenceDir)"
while ($true) {
  $T = Get-GpuTemp
  if ($T -gt $maxT) { $maxT = $T }
  if ($T -ge $LimitC) {
    $tree = @(Get-JobTree)
    if ($paused) { $tree = @($tree + $pausedPids | Select-Object -Unique) }
    Log "THERMAL GUARD (${Tag}): GPU $T C >= $LimitC C -> killing the job tree: $($tree -join ' ')"
    Add-Content -Path (Join-Path $EvidenceDir 'thermal_events.txt') -Value "$(Stamp) $Tag killed at $T C (tree: $($tree -join ' '))"
    foreach ($pp in $tree) { Stop-Process -Id $pp -Force -ErrorAction SilentlyContinue; Log "  killed $pp" }
    $killed = $true
    break
  }
  if (-not $paused -and $T -ge $PauseC) {
    $tree = @(Get-JobTree)
    if ($tree.Count -eq 0) { Log "GPU $T C >= $PauseC C but no job process matches the pattern; nothing to pause" }
    else {
      $pausedPids = @(Act-Tree $tree 'suspend')
      if ($pausedPids.Count -gt 0) { $paused = $true; $pauseStart = Get-Date; $pauses++; Log "PAUSE $pauses at $T C: suspended $($pausedPids -join ' ')" }
    }
  } elseif ($paused -and $T -le $ResumeC) {
    $r = @(Act-Tree $pausedPids 'resume')
    $dt = ((Get-Date) - $pauseStart).TotalSeconds; $pausedSec += $dt; $paused = $false
    Log "RESUME at $T C after $([math]::Round($dt)) s: resumed $($r -join ' ')"
  }
  if (Test-Path $receipt) { Log "run $Tag finished (T=$T C); pacer exits"; break }
  if (((Get-Date) - $t0).TotalSeconds -ge $MaxSeconds) { Log "pacer timeout after $MaxSeconds s"; break }
  Start-Sleep -Seconds $PollSec
}
if ($paused) { $r = @(Act-Tree $pausedPids 'resume'); $pausedSec += ((Get-Date) - $pauseStart).TotalSeconds; Log "resumed $($r -join ' ') on exit" }
$summary = [ordered]@{ Tag = $Tag; LimitC = $LimitC; PauseC = $PauseC; ResumeC = $ResumeC; Pauses = $pauses
                       PausedSeconds = [math]::Round($pausedSec, 1); MaxTempC = $maxT; Killed = $killed
                       WallSeconds = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1) }
$summary | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $EvidenceDir "$Tag.pacer.json")
Log ("pacer summary: pauses $pauses, paused " + [math]::Round($pausedSec) + " s, max $maxT C, killed $killed")
if ($killed) { exit 2 } else { exit 0 }
