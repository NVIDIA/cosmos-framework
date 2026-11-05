# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest.mock import patch

from cosmos_framework.utils import log, win_compat  # noqa: F401  log: imported before sys.platform is patched


def test_cosmos_attention_falls_back_to_sdpa_on_windows_only() -> None:
    with patch.object(win_compat.sys, "platform", "win32"):
        assert win_compat.attention_implementation("cosmos") == "sdpa"
        assert win_compat.attention_implementation("flash_attention_2") == "flash_attention_2"
        assert win_compat.attention_implementation("sdpa") == "sdpa"
    with patch.object(win_compat.sys, "platform", "linux"):
        assert win_compat.attention_implementation("cosmos") == "cosmos"
