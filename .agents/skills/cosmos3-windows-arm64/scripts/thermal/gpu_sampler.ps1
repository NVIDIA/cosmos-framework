# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Samples GPU health (nvidia-smi), system memory, and the launched process tree's memory/threads/GPU-process-memory counters once per interval.
# Usage: powershell -File gpu_sampler.ps1 -OutCsv <csv> -RootPid <pid> -StopFile <path> [-IntervalSec 1]
param([Parameter(Mandatory)][string]$OutCsv, [int]$RootPid = 0, [string[]]$ProcessNames = @(), [Parameter(Mandatory)][string]$StopFile, [double]$IntervalSec = 1.0)
$ErrorActionPreference = 'SilentlyContinue'
"ts,reasons,temp_c,sm_mhz,smi_mem_used_mib,power_w,util_pct,sys_used_gib,n_procs,ws_gib,private_gib,threads,gpu_dedicated_gib,gpu_shared_gib,gpu_committed_gib,adapter_dedicated_gib,adapter_shared_gib,pids" | Out-File $OutCsv -Encoding utf8
function Descendants([int]$root) {
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name
  $set = New-Object System.Collections.Generic.HashSet[int]; [void]$set.Add($root)
  $changed = $true
  while ($changed) { $changed = $false; foreach ($p in $all) { if ($set.Contains([int]$p.ParentProcessId) -and -not $set.Contains([int]$p.ProcessId)) { [void]$set.Add([int]$p.ProcessId); $changed = $true } } }
  foreach ($p in $all) { if ($ProcessNames -contains ($p.Name -replace '\.exe$','')) { [void]$set.Add([int]$p.ProcessId) } }
  return $set
}
while (-not (Test-Path $StopFile)) {
  $t = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss.fff'
  $smi = (& nvidia-smi -i 0 --query-gpu=clocks_event_reasons.active,temperature.gpu,clocks.sm,memory.used,power.draw,utilization.gpu --format=csv,noheader,nounits) -split ',\s*'
  $os = Get-CimInstance Win32_OperatingSystem
  $sysUsed = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 3)
  $pids = if ($RootPid -gt 0) { Descendants $RootPid } else { Descendants -1 }
  $procs = Get-Process | Where-Object { $pids.Contains($_.Id) }
  $ws = 0; $priv = 0; $thr = 0
  foreach ($p in $procs) { $ws += $p.WorkingSet64; $priv += $p.PrivateMemorySize64; $thr += $p.Threads.Count }
  $ded = 0; $sha = 0; $com = 0; $aded = 0; $asha = 0
  $ctr = Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage', '\GPU Process Memory(*)\Total Committed', '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage'
  foreach ($s in $ctr.CounterSamples) {
    $path = $s.Path.ToLower()
    if ($path -like '*gpu adapter memory*') { if ($path -like '*dedicated usage') { $aded += $s.CookedValue } elseif ($path -like '*shared usage') { $asha += $s.CookedValue }; continue }
    if ($s.InstanceName -match 'pid_(\d+)_') { $pid_ = [int]$Matches[1]; if ($pids.Contains($pid_)) {
      if ($path -like '*dedicated usage') { $ded += $s.CookedValue } elseif ($path -like '*shared usage') { $sha += $s.CookedValue } elseif ($path -like '*total committed') { $com += $s.CookedValue } } }
  }
  $g = 1GB
  "$t,$($smi[0]),$($smi[1]),$($smi[2]),$($smi[3]),$($smi[4]),$($smi[5]),$sysUsed,$($procs.Count),$([math]::Round($ws/$g,3)),$([math]::Round($priv/$g,3)),$thr,$([math]::Round($ded/$g,3)),$([math]::Round($sha/$g,3)),$([math]::Round($com/$g,3)),$([math]::Round($aded/$g,3)),$([math]::Round($asha/$g,3)),$(($procs | ForEach-Object { $_.Id }) -join ' ')" | Out-File $OutCsv -Append -Encoding utf8
  Start-Sleep -Seconds $IntervalSec
}
