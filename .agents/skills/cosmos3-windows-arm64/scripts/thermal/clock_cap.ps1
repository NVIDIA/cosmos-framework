# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# clock_cap.ps1 - apply, restore or probe the GPU SM clock cap and power limit with nvidia-smi (validated on an RTX Spark N1X).
# Usage: powershell -ExecutionPolicy Bypass -File clock_cap.ps1 [-Apply <mhz>] [-Restore] [-Power <watts>] [-Plan <guid-or-name>]
#   (no args)   probe: try -lgc 1800/1600/1400 and -pl, read back each, restore with -rgc / -pl default, print a table
#   -Apply 1400 set an SM clock cap and leave it (used around a measured run); -Restore resets clocks and power limit
# Never leaves a cap applied in probe mode. Records everything to -LogPath (default .\evidence\clock_cap_probe.log).
# Changing clocks needs an elevated shell (Run as administrator). Without it nvidia-smi answers "The current user does not
# have permission to change clocks" and exits 4; -Apply and -Restore then exit with that code so that a caller can tell a
# capped run from an uncapped one. Exit 0 means the driver accepted the request. Reading clocks needs no elevation.
param([int]$Apply = 0, [switch]$Restore, [int]$Power = 0, [string]$Plan = '',
      [string]$LogPath = (Join-Path (Get-Location) 'evidence\clock_cap_probe.log'))
$log = $LogPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
function Log($s) { $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $s; Write-Output $line; Add-Content -Path $log -Value $line }
function Smi([string[]]$smiArgs) { $out = & nvidia-smi @smiArgs 2>&1 | Out-String; $script:SmiExit = $LASTEXITCODE; return $out.Trim() }  # ($args is reserved in PowerShell)
function Refused([string]$what, [int]$code) { Log "$what REFUSED (nvidia-smi exit $code). Changing clocks needs an elevated shell; the thermal pacer works without it." }
function ReadBack() { Smi @('-i','0','--query-gpu=clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit,power.default_limit,power.max_limit,power.min_limit,temperature.gpu','--format=csv,noheader') }
Log "driver: $(Smi @('--query-gpu=driver_version','--format=csv,noheader'))  readback(sm,max.sm,app.gfx,power.limit,default,max,min,temp): $(ReadBack)"
if ($Restore) {
  $msg = Smi @('-i','0','-rgc'); $code = $script:SmiExit; Log "rgc: $msg"
  $pl = (Smi @('-i','0','--query-gpu=power.default_limit','--format=csv,noheader,nounits')).Trim()
  if ($pl -match '^[\d.]+$') { Log "pl default ($pl): $(Smi @('-i','0','-pl',$pl))" }
  Log "after restore: $(ReadBack)"
  if ($code -ne 0) { Refused 'clock reset' $code; exit $code }
  exit 0
}
if ($Apply -gt 0) {
  $msg = Smi @('-i','0','-lgc',"$Apply"); $code = $script:SmiExit; Log "lgc $Apply : $msg"
  Log "after apply: $(ReadBack)"
  if ($code -ne 0) { Refused "clock cap $Apply MHz" $code; exit $code }
  exit 0
}
if ($Power -gt 0) { Log "pl $Power : $(Smi @('-i','0','-pl',"$Power"))"; Log "after pl: $(ReadBack)"; exit 0 }
if ($Plan) { Log "powercfg setactive $Plan : $(& powercfg /SETACTIVE $Plan 2>&1 | Out-String)"; Log "active scheme: $(& powercfg /GETACTIVESCHEME | Out-String)"; exit 0 }
# probe mode
Log "supported clocks (head): $((Smi @('-i','0','-q','-d','SUPPORTED_CLOCKS')) -split "`n" | Select-Object -First 12 | Out-String)"
foreach ($mhz in 1800, 1600, 1400) {
  Log "lgc $mhz -> $(Smi @('-i','0','-lgc',"$mhz"))"
  Log "   readback: $(ReadBack)"
  Log "   rgc -> $(Smi @('-i','0','-rgc'))"
}
Log "lmc 1000 (memory clock, informational) -> $(Smi @('-i','0','-lmc','1000'))"; Log "   rmc -> $(Smi @('-i','0','-rmc'))"
$pl = (Smi @('-i','0','--query-gpu=power.default_limit','--format=csv,noheader,nounits')).Trim()
Log "power.default_limit = $pl ; pl 40 -> $(Smi @('-i','0','-pl','40'))"
Log "   readback: $(ReadBack)"
if ($pl -match '^[\d.]+$') { Log "   pl default ($pl) -> $(Smi @('-i','0','-pl',$pl))" } else { Log "   (no numeric default power limit; nothing to restore)" }
Log "power plans: $(& powercfg /LIST 2>&1 | Out-String)"
Log "active scheme: $(& powercfg /GETACTIVESCHEME 2>&1 | Out-String)"
Log "final readback: $(ReadBack)"

