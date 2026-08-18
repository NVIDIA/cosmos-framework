#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# One-time data prep for the RoboCasa recipes: convert a released RoboCasa export from LeRobot
# v2.1 to v3.0. Every task found under SRC_ROOT is converted, so this works for any split --
# target/atomic, pretrain/atomic, pretrain/composite, ...
#
#   SRC_ROOT=/path/to/robocasa/datasets/v1.0/target/atomic \
#       bash $(python -c "import cosmos_framework, pathlib; print(pathlib.Path(cosmos_framework.__file__).parent)")/scripts/convert_robocasa_to_lerobot_v30.sh
#
# Why this is needed: RoboCasa publishes LeRobot **v2.1**, and the `lerobot` pinned by
# cosmos-framework is v3.0-only -- it rejects the v2.1 layout outright. Conversion is done on
# COPIES; the released dataset is never modified.
#
# CPU-only and idempotent: a task whose destination already has meta/tasks.parquet (a v3.0-only
# marker) is skipped, so re-running after an interruption resumes rather than redoing the work.
#
# Output layout, which is what ROBOCASA_ROOT must point at:
#   <V30_ROOT>/<task>/<date>/lerobot/

set -euo pipefail

# Deliberately does NOT cd to its own directory: the script lives inside the installed package,
# while the converted dataset belongs wherever the caller is working (typically a cookbook
# recipe folder). Both roots resolve against the caller's cwd.
#
# No default for SRC_ROOT: the RoboCasa release is site-specific, so it must be given.
SRC_ROOT="${SRC_ROOT:?set SRC_ROOT=/path/to/robocasa/datasets/v1.0/target/atomic (see https://robocasa.ai)}"
V30_ROOT="${V30_ROOT:-$PWD/data/robocasa_v30}"

# Discover every task under SRC_ROOT rather than hardcoding a list: a task is any directory
# holding a <task>/<date>/lerobot export. Narrow it with TASKS_OVERRIDE="A B C" if wanted.
# `find -L` so a SRC_ROOT assembled from symlinked task directories still resolves; without it
# such a layout yields no tasks and the run stops on the check below.
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    read -ra TASKS <<< "${TASKS_OVERRIDE}"
else
    mapfile -t TASKS < <(
        find -L "${SRC_ROOT}" -mindepth 3 -maxdepth 3 -type d -name lerobot -printf '%h\n' \
            | xargs -r -n1 dirname | xargs -r -n1 basename | sort -u
    )
fi
(( ${#TASKS[@]} )) || {
    echo "ERROR: no <task>/<date>/lerobot exports found under ${SRC_ROOT}" >&2
    echo "       Is SRC_ROOT pointing at a RoboCasa split directory?" >&2
    exit 1
}
echo "discovered ${#TASKS[@]} task(s)"

echo "== v2.1 -> v3.0 conversion =="
echo "   from: ${SRC_ROOT}"
echo "   to  : ${V30_ROOT}"
for t in "${TASKS[@]}"; do
    src="$(ls -d "${SRC_ROOT}/${t}/"*/lerobot 2>/dev/null | head -1 || true)"
    if [[ -z "${src}" ]]; then echo "  ${t}: no source shard, skipping"; continue; fi
    date="$(basename "$(dirname "${src}")")"
    dst_parent="${V30_ROOT}/${t}/${date}"
    dst="${dst_parent}/lerobot"
    if [[ -f "${dst}/meta/tasks.parquet" ]]; then echo "  ${t}: already v3.0, skip"; continue; fi
    echo "  ${t}: copy + convert (${src})"
    mkdir -p "${dst_parent}"
    rm -rf "${dst}" "${dst_parent}/lerobot_old" "${dst_parent}/lerobot_v30"
    cp -r "${src}" "${dst}"
    python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
        --repo-id=lerobot --root="${dst_parent}" --push-to-hub false
    rm -rf "${dst_parent}/lerobot_old"
    # Carry over RoboCasa-specific meta the converter ignores. The loader does not need these,
    # but keeping them makes the converted copy self-describing.
    for f in modality.json embodiment.json; do
        [[ -f "${src}/meta/${f}" ]] && cp "${src}/meta/${f}" "${dst}/meta/${f}" || true
    done
done

echo "== verify =="
missing=0
for t in "${TASKS[@]}"; do
    if ls "${V30_ROOT}/${t}/"*/lerobot/meta/tasks.parquet >/dev/null 2>&1; then
        echo "  OK   ${t}"
    else
        echo "  MISS ${t}"; missing=$((missing + 1))
    fi
done
if (( missing )); then
    echo "ERROR: ${missing} task(s) failed to convert; see the log above." >&2
    exit 1
fi
echo "== done: ${#TASKS[@]} tasks under ${V30_ROOT} =="
echo "   train with: ROBOCASA_ROOT=${V30_ROOT} bash launch_sft_action_policy_robocasa_nano.sh"
