#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Source this file from any working directory; no installation or network access.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file: source /path/to/cosmos-framework/tools/cosmos_workflows/env.sh" >&2
    exit 2
fi
export COSMOS_WORKFLOWS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
