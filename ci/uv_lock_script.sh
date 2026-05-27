#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Generate uv lock files for scripts.

set -euo pipefail

for file in "$@"; do
  if head -n1 "$file" | grep -q '^#!/usr/bin/env -S uv run --script'; then
    if ! uv lock -q --check --script "$file" &>/dev/null; then
      echo "Updating lock file for '$file'" >&2
      uv lock -q --script "$file"
    fi
  fi
done
