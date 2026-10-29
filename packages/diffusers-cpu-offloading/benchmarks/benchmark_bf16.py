# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run full Cosmos3-Nano BF16 benchmarks with safety checking enabled by default."""

from runner import main

if __name__ == "__main__":
    raise SystemExit(main(precision="bf16"))
