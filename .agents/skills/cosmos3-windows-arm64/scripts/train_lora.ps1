# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# train_lora.ps1 - LoRA SFT of the Cosmos3 Nano Reasoner on native Windows on Arm.
# Sets the environment block the trainer needs at world size 1, appends the Windows overrides and the attention keys,
# and runs the job under the thermal limiter (clock cap, start gate, pacer, receipt). One job per call.
# Usage: powershell -ExecutionPolicy Bypass -File train_lora.ps1 -Python <venv>\Scripts\python.exe -Recipe <recipe>.toml
#          -Model <VLM dir> -OutputRoot <dir> -JobName <name> -TrainAnnotation <json> -TrainMedia <dir>
#          [-ValAnnotation <json> -ValMedia <dir>] [-NumFrames 8] [-MaxPixels 1179648] [-SystemPrompt ""]
#          [-ClockCapMHz 1400] [-RequireCap] [-MaxStartTempC 62] [-PauseC 84] [-ResumeC 78] [-LimitC 87] [-TimeoutSec 14400] [-Override k=v,...]
# The clock cap needs an elevated shell. In a standard shell the driver refuses it; the launcher then says so and runs
# under the pacer alone (-RequireCap stops instead). -ClockCapMHz 0 skips the cap without trying.
param(
  [Parameter(Mandatory)][string]$Python,
  [Parameter(Mandatory)][string]$Recipe,
  [Parameter(Mandatory)][string]$Model,
  [Parameter(Mandatory)][string]$OutputRoot,
  [Parameter(Mandatory)][string]$JobName,
  [Parameter(Mandatory)][string]$TrainAnnotation,
  [Parameter(Mandatory)][string]$TrainMedia,
  [string]$ValAnnotation = "",
  [string]$ValMedia = "",
  [int]$NumFrames = 8,
  [int]$MaxPixels = 1179648,
  [string]$SystemPrompt = "",
  [int]$ClockCapMHz = 1400,
  [switch]$RequireCap,
  [int]$MaxStartTempC = 62,
  [int]$PauseC = 84,
  [int]$ResumeC = 78,
  [int]$LimitC = 87,
  [int]$TimeoutSec = 14400,
  [string[]]$Override = @()
)
$ErrorActionPreference = 'Stop'
$thermal = Join-Path $PSScriptRoot 'thermal'
$evidence = Join-Path $OutputRoot 'evidence'
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
foreach ($p in @($Python, $Recipe, $Model, $TrainAnnotation, $TrainMedia)) { if (-not (Test-Path $p)) { throw "missing: $p" } }
if (-not $ValAnnotation) { $ValAnnotation = $TrainAnnotation }
if (-not $ValMedia) { $ValMedia = $TrainMedia }
$tag = "train_$JobName"

# The environment block: gloo rendezvous at world size 1 (the trainer runs an unconditional barrier), no libuv,
# a UTF-8 console, an explicit output root, the converted model, and the data and input knobs the
# cosmos_video_conversation experiment reads (COSMOS_VIDEO_*).
$envTable = @{
  MASTER_ADDR = '127.0.0.1'; MASTER_PORT = '29500'; RANK = '0'; WORLD_SIZE = '1'; LOCAL_RANK = '0'; USE_LIBUV = '0'
  PYTHONIOENCODING = 'utf-8'; IMAGINAIRE_OUTPUT_ROOT = $OutputRoot; VLM_SAFETENSORS_PATH = $Model
  COSMOS_VIDEO_TRAIN_ANNOTATION = $TrainAnnotation; COSMOS_VIDEO_TRAIN_MEDIA = $TrainMedia
  COSMOS_VIDEO_VAL_ANNOTATION = $ValAnnotation; COSMOS_VIDEO_VAL_MEDIA = $ValMedia
  COSMOS_VIDEO_NUM_FRAMES = "$NumFrames"; COSMOS_VIDEO_MAX_PIXELS = "$MaxPixels"; COSMOS_VIDEO_SYSTEM_PROMPT = $SystemPrompt
  COSMOS_VIDEO_DECODER_DEVICE = 'cuda'; COSMOS_DATALOADER_SEED = '42'; TAO_WINDOWS_PYTHON = $Python
}

# The Windows overrides (references/training-overrides.md), then the attention keys, synchronous saves, a fresh start, the job name.
$overrides = @(
  'optimizer=adamw',
  'trainer.distributed_parallelism=ddp',
  'model.config.parallelism.data_parallel_shard_degree=1',
  'model.config.parallelism.data_parallel_replicate_degree=1',
  'model.config.policy.lora_enabled=true',
  'dataloader_train.num_workers=0', 'dataloader_train.persistent_workers=false', 'dataloader_train.prefetch_factor=null',
  'dataloader_val.num_workers=0', 'dataloader_val.persistent_workers=false', 'dataloader_val.prefetch_factor=null',
  'policy.attn_implementation=sdpa', 'model.config.policy.attn_implementation=sdpa',
  'checkpoint.dcp_async_mode_enabled=false',
  'checkpoint.load_path=',
  "job.name=$JobName"
) + $Override
$cmd = "`"$Python`" -m cosmos_framework.scripts.train --sft-toml=`"$Recipe`" -- " + ($overrides -join ' ')
Write-Output "command: $cmd"
"$((Get-Date).ToUniversalTime().ToString('o')) $cmd" | Set-Content (Join-Path $evidence "$tag.command.txt") -Encoding UTF8

# The cap is applied by the driver only for an elevated shell; say what happened instead of assuming it.
$capApplied = $false
if ($ClockCapMHz -gt 0) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $thermal 'clock_cap.ps1') -Apply $ClockCapMHz -LogPath (Join-Path $evidence 'clock_cap.log') | Out-Null
  $capExit = $LASTEXITCODE
  $capApplied = ($capExit -eq 0)
  if ($capApplied) { $capState = "clock cap $ClockCapMHz MHz applied" }
  else { $capState = "clock cap NOT applied (nvidia-smi exit $capExit; changing clocks needs an elevated shell): uncapped, pacer only" }
} else { $capState = 'no clock cap requested: uncapped, pacer only' }
Write-Output $capState
"cap: $capState" | Add-Content (Join-Path $evidence "$tag.command.txt") -Encoding UTF8
if ($RequireCap -and -not $capApplied) { throw "-RequireCap: $capState" }
$pacer = Start-Process -FilePath powershell -PassThru -WindowStyle Hidden -ArgumentList @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $thermal 'thermal_pacer.ps1'),
  '-Tag', $tag, '-EvidenceDir', $evidence, '-PauseC', $PauseC, '-ResumeC', $ResumeC, '-LimitC', $LimitC, '-MaxSeconds', ($TimeoutSec + 600))
Write-Output "pacer armed (pause $PauseC C, resume $ResumeC C, kill $LimitC C)"
try {
  & (Join-Path $thermal 'run_measured.ps1') -Tag $tag -CommandLine $cmd -EvidenceDir $evidence -WorkingDirectory $OutputRoot `
      -Python $Python -TimeoutSec $TimeoutSec -MaxStartTempC $MaxStartTempC -Env $envTable
} finally {
  if ($capApplied) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $thermal 'clock_cap.ps1') -Restore -LogPath (Join-Path $evidence 'clock_cap.log') | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Output "default clocks restored" }
    else { Write-Output "clock reset FAILED (nvidia-smi exit $LASTEXITCODE): run thermal\clock_cap.ps1 -Restore from an elevated shell" }
  } else { Write-Output "no clock cap was applied; clocks unchanged" }
  if ($pacer -and -not $pacer.HasExited) { $pacer.WaitForExit(30000) | Out-Null }
}
$receipt = Get-Content (Join-Path $evidence "$tag.receipt.json") -Raw | ConvertFrom-Json
Write-Output ("run directory: {0}\cosmos3_reasoner\cosmos_video_conversation_sft\{1}  (checkpoints\epoch_N, config.yaml)" -f $OutputRoot, $JobName)
Write-Output ("receipt: exit {0}, {1} s wall, peak {2} C" -f $receipt.ExitCode, $receipt.WallSeconds, $receipt.Summary.temp_c_max)
exit [int]$receipt.ExitCode
