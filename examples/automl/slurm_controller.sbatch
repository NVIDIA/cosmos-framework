#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Replace site placeholders and review CPU resources before submission.
#SBATCH --job-name=cosmos-automl-controller
#SBATCH --partition=REPLACE_CPU_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=cosmos-automl-controller-%j.log

set -euo pipefail
: "${AUTOML_CONTROLLER_PYTHON:?Set the shared controller interpreter}"
: "${AUTOML_MANIFEST:?Set the sealed, absolute shared manifest path}"
: "${AUTOML_WORKSPACE:?Set the absolute shared experiment workspace}"
: "${AUTOML_CONTROLLER_MAX_ALLOCATIONS:?Set the reviewed CPU allocation cap}"
: "${AUTOML_STARTUP_HELPER:?Set the sealed shared slurm_controller_startup.py path}"
: "${AUTOML_STARTUP_BASE:?Set the shared startup directory registered with the observer}"
: "${AUTOML_DEADLINE:?Set the original absolute experiment deadline}"

# The independent observer service must already be ready before sbatch and have
# this native job registered afterwards. Every requeue gets fresh probe receipts.
# Do not wrap the CLI in the startup helper's run verb: signals must reach it.
trap 'exit 75' TERM HUP INT USR1
generation=${SLURM_RESTART_COUNT:-0}
[[ "$generation" =~ ^[0-9]+$ && "$AUTOML_CONTROLLER_MAX_ALLOCATIONS" =~ ^[1-9][0-9]*$ ]]
(( generation < AUTOML_CONTROLLER_MAX_ALLOCATIONS ))
export AUTOML_STARTUP_ATTEMPT="$AUTOML_STARTUP_BASE/${SLURM_JOB_ID}-g${generation}"
"$AUTOML_CONTROLLER_PYTHON" "$AUTOML_STARTUP_HELPER" probe \
    --attempt-dir "$AUTOML_STARTUP_ATTEMPT" --deadline "$AUTOML_DEADLINE" \
    --probe-seconds 600 --minimum-run-seconds 600

# One experiment owns this CPU job. The verified successor automatically resumes
# its ledger; do not clear outputs or switch manifests on SLURM_RESTART_COUNT.
workflow=()
case "${AUTOML_WORKFLOW:-sft}" in
    sft) ;;
    prompts) workflow=(prompts) ;;
    *) echo 'AUTOML_WORKFLOW must be sft or prompts' >&2; exit 2 ;;
esac

# exec makes Python the batch process receiving B:USR1. Do not add an EXIT trap
# that cancels trials or removes credentials needed by a successor allocation.
"$AUTOML_CONTROLLER_PYTHON" "$AUTOML_STARTUP_HELPER" fanout --attempt-dir "$AUTOML_STARTUP_ATTEMPT"
trap - TERM HUP INT USR1
exec "$AUTOML_CONTROLLER_PYTHON" -m cosmos_framework.scripts.automl \
    "${workflow[@]}" "${AUTOML_ACTION:-run}" --manifest "$AUTOML_MANIFEST" \
    --workspace "$AUTOML_WORKSPACE" --approve \
    --controller-max-allocations "$AUTOML_CONTROLLER_MAX_ALLOCATIONS" \
    --controller-allocation-seconds 3600 --controller-grace-seconds 300
