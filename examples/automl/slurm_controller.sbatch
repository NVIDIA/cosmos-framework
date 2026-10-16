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
exec "$AUTOML_CONTROLLER_PYTHON" -m cosmos_framework.scripts.automl \
    "${workflow[@]}" run --manifest "$AUTOML_MANIFEST" \
    --workspace "$AUTOML_WORKSPACE" --approve \
    --controller-max-allocations "$AUTOML_CONTROLLER_MAX_ALLOCATIONS" \
    --controller-allocation-seconds 3600 --controller-grace-seconds 300
