# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest.mock import patch

from cosmos_framework.checkpoint import background_store


def test_tcpstore_disables_libuv_only_on_windows() -> None:
    with patch.object(background_store.sys, "platform", "win32"):
        assert background_store._tcpstore_platform_kwargs() == {"use_libuv": False}
    with patch.object(background_store.sys, "platform", "linux"):
        assert background_store._tcpstore_platform_kwargs() == {}
