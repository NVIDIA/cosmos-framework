#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Generate uv lock files for projects.

set -euo pipefail

for file in "$@"; do
  project_dir="$(dirname "$file")"
  if ! uv lock -q --check --project "$project_dir" &>/dev/null; then
    echo "Updating lock file for '$project_dir'" >&2
    uv lock -q --project "$project_dir"
  fi
done
