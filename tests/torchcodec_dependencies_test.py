# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""NPP discovery regressions that run without installing CUDA or TorchCodec."""

import unittest
from importlib.metadata import PackageNotFoundError, PackagePath
from pathlib import Path
from unittest.mock import Mock, patch

from cosmos_framework.utils.generator import torchcodec_dependencies as dependencies


class NppDiscoveryTests(unittest.TestCase):
    def setUp(self):
        dependencies.preload_npp.cache_clear()
        self.addCleanup(dependencies.preload_npp.cache_clear)
        platform = patch.object(dependencies.sys, "platform", "linux")
        platform.start()
        self.addCleanup(platform.stop)

    def test_cuda_wheel_layouts_are_discovered_without_a_library_path(self):
        for cuda, package_name, relative in (
            ("12.8", "nvidia-npp-cu12", "nvidia/npp/lib/libnppicc.so.12"),
            ("13.0", "nvidia-npp", "nvidia/cu13/lib/libnppicc.so.13"),
        ):
            with self.subTest(cuda=cuda):
                library = PackagePath(relative)
                package = Mock(files=[PackagePath("unrelated/libnppc.so"), library])
                absolute = Path("/test-environment/site-packages") / relative
                package.locate_file.return_value = absolute
                handle = object()
                with (
                    patch.object(dependencies, "distribution", return_value=package) as lookup,
                    patch.object(dependencies.ctypes, "CDLL", side_effect=[OSError("not found"), handle]) as load,
                ):
                    self.assertIs(dependencies.preload_npp(cuda), handle)
                lookup.assert_called_once_with(package_name)
                self.assertEqual(load.call_args_list[0].args, (library.name,))
                self.assertEqual(load.call_args_list[1].args, (str(absolute),))

    def test_existing_library_is_preserved(self):
        with (
            patch.object(dependencies.ctypes, "CDLL", return_value=object()) as load,
            patch.object(dependencies, "distribution") as lookup,
        ):
            self.assertIs(dependencies.preload_npp("13.0"), load.return_value)
            lookup.assert_not_called()

    def test_cpu_or_unsupported_cuda_does_not_load_npp(self):
        with patch.object(dependencies.ctypes, "CDLL") as load:
            for cuda in (None, "11.8"):
                self.assertIsNone(dependencies.preload_npp(cuda))
            load.assert_not_called()

    def test_missing_package_keeps_torchcodec_diagnostics(self):
        with (
            patch.object(dependencies.ctypes, "CDLL", side_effect=OSError("not found")),
            patch.object(dependencies, "distribution", side_effect=PackageNotFoundError),
        ):
            self.assertIsNone(dependencies.preload_npp("13.0"))

    def test_wrong_cuda_library_is_not_loaded(self):
        package = Mock(files=[PackagePath("nvidia/npp/lib/libnppicc.so.12")])
        with (
            patch.object(dependencies.ctypes, "CDLL", side_effect=OSError("not found")) as load,
            patch.object(dependencies, "distribution", return_value=package),
        ):
            self.assertIsNone(dependencies.preload_npp("13.0"))
            load.assert_called_once_with("libnppicc.so.13")
            package.locate_file.assert_not_called()

    def test_broken_wheel_library_error_is_not_suppressed(self):
        package = Mock(files=[PackagePath("nvidia/cu13/lib/libnppicc.so.13")])
        with (
            patch.object(dependencies.ctypes, "CDLL", side_effect=[OSError("not found"), OSError("invalid ELF")]),
            patch.object(dependencies, "distribution", return_value=package),
        ):
            with self.assertRaisesRegex(OSError, "invalid ELF"):
                dependencies.preload_npp("13.0")


if __name__ == "__main__":
    unittest.main()
