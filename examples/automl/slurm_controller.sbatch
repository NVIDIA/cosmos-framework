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
generation=${SLURM_RESTART_COUNT:-0}
if [[ ! "$generation" =~ ^[0-9]+$ ]]; then
    echo 'SLURM_RESTART_COUNT must be a nonnegative decimal integer' >&2
    exit 2
fi
if [[ ! "$AUTOML_CONTROLLER_MAX_ALLOCATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'AUTOML_CONTROLLER_MAX_ALLOCATIONS must be a positive decimal integer without leading zeros' >&2
    exit 2
fi
# Normalize native zero padding; compare decimal strings without octal parsing
# or arithmetic overflow. No helper or signal trap runs before this cap check.
generation=${generation#"${generation%%[!0]*}"}
generation=${generation:-0}
allocation_cap=$AUTOML_CONTROLLER_MAX_ALLOCATIONS
if [[ ${#generation} -gt ${#allocation_cap} ]] || {
    [[ ${#generation} -eq ${#allocation_cap} ]] &&
        [[ "$generation" == "$allocation_cap" || "$generation" > "$allocation_cap" ]]
}; then
    echo "Controller allocation cap exhausted: generation $generation, approved count $allocation_cap" >&2
    exit 2
fi

# One experiment owns this CPU job. The verified successor automatically resumes
# its ledger; do not clear outputs or switch manifests on SLURM_RESTART_COUNT.
workflow=()
case "${AUTOML_WORKFLOW:-sft}" in
    sft) ;;
    prompts) workflow=(prompts) ;;
    *) echo 'AUTOML_WORKFLOW must be sft or prompts' >&2; exit 2 ;;
esac
action=${AUTOML_ACTION:-run}
case "$action" in
    run|resume) ;;
    *) echo 'AUTOML_ACTION must be run or resume' >&2; exit 2 ;;
esac

# Optional cross-node startup supervision. Unset helper preserves the original
# direct-CLI launch path; it does not provide pre-controller startup recovery.
if [[ -n "${AUTOML_STARTUP_HELPER:-}" ]]; then
    : "${AUTOML_STARTUP_BASE:?Set the shared startup directory registered with the observer}"
    : "${AUTOML_DEADLINE:?Set the original absolute experiment deadline}"
    : "${SLURM_JOB_ID:?The startup probe requires a native SLURM job ID}"

    stop_startup() {
        trap - TERM HUP INT USR1
        echo 'Controller startup interrupted; no automatic requeue requested' >&2
        if [[ -n "${startup_pid:-}" ]]; then
            kill -TERM "$startup_pid" 2>/dev/null || true
        fi
        exit 2
    }
    run_startup_helper() {
        # A trapped signal interrupts wait promptly, unlike a foreground Python
        # call. Forward termination to the probe; SLURM owns batch-child cleanup.
        "$AUTOML_CONTROLLER_PYTHON" "$AUTOML_STARTUP_HELPER" "$@" &
        startup_pid=$!
        if wait "$startup_pid"; then
            startup_pid=
        else
            startup_status=$?
            startup_pid=
            echo "Controller startup $1 failed (exit $startup_status); controller not started" >&2
            # Do not leak EX_TEMPFAIL (75) into site automatic-requeue policies.
            # The external supervisor owns native reconciliation and retry caps.
            exit 2
        fi
    }
    trap stop_startup TERM HUP INT USR1
    # The observer must be ready before sbatch and this job registered afterwards.
    # Every requeue gets fresh receipts; do not wrap the handoff CLI in run.
    export AUTOML_STARTUP_ATTEMPT="$AUTOML_STARTUP_BASE/${SLURM_JOB_ID}-g${generation}"
    run_startup_helper probe \
        --attempt-dir "$AUTOML_STARTUP_ATTEMPT" --deadline "$AUTOML_DEADLINE" \
        --probe-seconds 600 --minimum-run-seconds 600
    run_startup_helper fanout --attempt-dir "$AUTOML_STARTUP_ATTEMPT"
    trap - TERM HUP INT USR1
fi

# exec makes Python the batch process receiving B:USR1. Do not add an EXIT trap
# that cancels trials or removes credentials needed by a successor allocation.
exec "$AUTOML_CONTROLLER_PYTHON" -m cosmos_framework.scripts.automl \
    "${workflow[@]}" "$action" --manifest "$AUTOML_MANIFEST" \
    --workspace "$AUTOML_WORKSPACE" --approve \
    --controller-max-allocations "$AUTOML_CONTROLLER_MAX_ALLOCATIONS" \
    --controller-allocation-seconds 3600 --controller-grace-seconds 300
