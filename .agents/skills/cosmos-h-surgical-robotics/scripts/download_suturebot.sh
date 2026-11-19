#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Download and unzip SutureBot tissues from the publisher's Hugging Face dataset at a pinned revision.
#
#   SUTUREBOT_RAW_ROOT=/data/SutureBot_raw bash scripts/download_suturebot.sh 3 10   # two tissues
#   SUTUREBOT_RAW_ROOT=/data/SutureBot_raw bash scripts/download_suturebot.sh        # all ten (221 GB of zips)
#
# The dataset is CC-BY-4.0 (https://huggingface.co/datasets/jchen396/SutureBot). Each zip is deleted
# after a successful unzip unless KEEP_ZIPS=1.

set -euo pipefail

REPO=jchen396/SutureBot
REVISION=59fd8016649d605a041f2b0e4047702d2ea45ac1
: "${SUTUREBOT_RAW_ROOT:?set SUTUREBOT_RAW_ROOT to the download directory}"
TISSUES=("$@")
[[ ${#TISSUES[@]} -gt 0 ]] || TISSUES=(1 2 3 4 5 6 7 8 9 10)

mkdir -p "$SUTUREBOT_RAW_ROOT"
for t in "${TISSUES[@]}"; do
    if [[ -d "$SUTUREBOT_RAW_ROOT/tissue_$t" ]]; then
        echo "tissue_$t already unzipped, skipping"
        continue
    fi
    hf download "$REPO" "tissue_$t.zip" --repo-type dataset --revision "$REVISION" --local-dir "$SUTUREBOT_RAW_ROOT"
    unzip -q -o "$SUTUREBOT_RAW_ROOT/tissue_$t.zip" -d "$SUTUREBOT_RAW_ROOT"
    [[ "${KEEP_ZIPS:-0}" == 1 ]] || rm -f "$SUTUREBOT_RAW_ROOT/tissue_$t.zip"
    echo "tissue_$t: $(find "$SUTUREBOT_RAW_ROOT/tissue_$t" -name ee_csv.csv | wc -l) episodes"
done
