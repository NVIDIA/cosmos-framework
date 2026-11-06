#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# timed.sh <label> <logfile> <cmd...> : run cmd, append start/end/wall seconds + exit code to $TIMES
TIMES=${TIMES:-${RECIPE:-.}/logs/times.tsv}; mkdir -p "$(dirname "$TIMES")"
label=$1; shift; log=$1; shift
start=$(date +%s); start_h=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
echo ">>> [$label] start $start_h : $*" | tee -a "$log"
"$@" >> "$log" 2>&1; rc=$?
end=$(date +%s); end_h=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
echo ">>> [$label] end $end_h rc=$rc wall=$((end-start))s" | tee -a "$log"
printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$start_h" "$end_h" "$((end-start))" "$rc" >> "$TIMES"
exit $rc
