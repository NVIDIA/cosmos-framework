# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unit tests for backend filtering via environment variables.
"""

import os
import unittest

import pytest

from cosmos_framework.model.attention.utils.environment import (
    filter_attention_backends,
    filter_attention_merge_backends,
    filter_multi_dim_attention_backends,
    parse_backend_filter,
)


@pytest.mark.L0
class TestParseBackendFilter(unittest.TestCase):
    """Test the parse_backend_filter function."""

    def test_basic_functionality(self):
        """Test basic ban-list and allow-list functionality."""
        default = ["flash2", "natten", "cudnn"]

        # Empty/None returns defaults
        assert parse_backend_filter(None, default) == default
        assert parse_backend_filter("", default) == default
        assert parse_backend_filter("  ", default) == default

        # Ban-list mode
        assert parse_backend_filter("-flash2", default) == ["natten", "cudnn"]
        assert parse_backend_filter("-flash2,-cudnn", default) == ["natten"]
        assert parse_backend_filter("-flash2,-natten", ["flash2", "natten"]) == []

        # Allow-list mode
        assert parse_backend_filter("natten", default) == ["natten"]
        assert parse_backend_filter("flash2,natten", default) == ["flash2", "natten"]

        # Preserves order from defaults
        default_ordered = ["flash3", "flash2", "natten", "cudnn"]
        assert parse_backend_filter("cudnn,flash2", default_ordered) == ["flash2", "cudnn"]

        # Whitespace handling
        assert parse_backend_filter(" flash2 , natten ", default) == ["flash2", "natten"]
        assert parse_backend_filter(" -flash2 , -cudnn ", default) == ["natten"]

        # Empty items ignored
        assert parse_backend_filter("flash2,,natten", ["flash2", "natten"]) == ["flash2", "natten"]

    def test_invalid_ban_backends_warns(self):
        """Test that invalid backends in ban-list only issue warnings (not errors)."""
        default = ["flash2", "natten", "cudnn"]

        # Invalid ban backends should just warn, not error (they may not be available on this GPU)
        result = parse_backend_filter("-flash3", default)
        assert result == default  # Nothing removed since flash3 wasn't in the list

        result = parse_backend_filter("-flash2,-invalid_backend", default)
        assert result == ["natten", "cudnn"]  # flash2 removed, invalid_backend ignored with warning

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_error_mixing_bans_and_allows(self):
        """Test that mixing bans and allows raises an error."""
        default = ["flash2", "natten", "cudnn"]
        # This should raise ValueError about mixing ban-list and allow-list
        parse_backend_filter("-flash2,natten", default)

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_error_invalid_allow_backends(self):
        """Test that invalid backends in allow-list raise an error."""
        default = ["flash2", "natten", "cudnn"]
        # This should raise ValueError about invalid backend in allow-list
        parse_backend_filter("flash3", default)

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_error_case_sensitivity_allow(self):
        """Test that backend names are case-sensitive in allow-list."""
        default = ["flash2", "natten"]
        # This should raise ValueError about invalid backend (case mismatch)
        parse_backend_filter("Flash2", default)


@pytest.mark.L0
class TestFilterAPIs(unittest.TestCase):
    """Test the filter_*_backends functions."""

    def setUp(self):
        """Save and clear environment variables before each test."""
        self.saved_env = {}
        for var in ["I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM", "I4_ATTN_BACKENDS_MERGE"]:
            self.saved_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment variables after each test."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_attention_backends_no_env_var(self):
        """Test filter_attention_backends with no env var returns defaults."""
        default = ["flash2", "natten"]
        result = filter_attention_backends(default)
        assert result == default

    def test_attention_backends_ban_list(self):
        """Test filter_attention_backends with ban-list."""
        os.environ["I4_ATTN_BACKENDS"] = "-flash2"
        default = ["flash2", "natten", "cudnn"]
        result = filter_attention_backends(default)
        assert result == ["natten", "cudnn"]

    def test_attention_backends_allow_list(self):
        """Test filter_attention_backends with allow-list."""
        os.environ["I4_ATTN_BACKENDS"] = "natten"
        default = ["flash2", "natten", "cudnn"]
        result = filter_attention_backends(default)
        assert result == ["natten"]

    def test_multi_dim_backends_no_env_var(self):
        """Test filter_multi_dim_attention_backends with no env var."""
        default = ["natten"]
        result = filter_multi_dim_attention_backends(default)
        assert result == default

    def test_multi_dim_backends_ban_list(self):
        """Test filter_multi_dim_attention_backends with ban-list."""
        os.environ["I4_ATTN_BACKENDS_MULTIDIM"] = "-natten"
        default = ["natten"]
        result = filter_multi_dim_attention_backends(default)
        assert result == []

    def test_merge_backends_no_env_var(self):
        """Test filter_attention_merge_backends with no env var."""
        default = ["natten"]
        result = filter_attention_merge_backends(default)
        assert result == default

    def test_merge_backends_ban_list(self):
        """Test filter_attention_merge_backends with ban-list."""
        os.environ["I4_ATTN_BACKENDS_MERGE"] = "-natten"
        default = ["natten"]
        result = filter_attention_merge_backends(default)
        assert result == []

    def test_multiple_env_vars_independent(self):
        """Test that different filter functions use independent env vars."""
        os.environ["I4_ATTN_BACKENDS"] = "-flash2"
        os.environ["I4_ATTN_BACKENDS_MULTIDIM"] = "natten"

        default_sdpa = ["flash2", "natten"]
        default_multidim = ["natten"]

        result_sdpa = filter_attention_backends(default_sdpa)
        result_multidim = filter_multi_dim_attention_backends(default_multidim)

        assert result_sdpa == ["natten"]
        assert result_multidim == ["natten"]


@pytest.mark.L0
class TestBackendListIntegration(unittest.TestCase):
    """Integration tests for backend list functions with environment variables."""

    def setUp(self):
        """Save and clear environment variables before each test."""
        self.saved_env = {}
        for var in ["I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM", "I4_ATTN_BACKENDS_MERGE"]:
            self.saved_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment variables after each test."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_sdpa_backend_filtering(self):
        """Test that get_backend_list respects backend filtering."""
        from cosmos_framework.model.attention.backends import get_backend_list

        # Test without env var (should return defaults based on arch)
        backends = get_backend_list(90)  # H100
        assert "flash3" in backends or "flash2" in backends or "natten" in backends

        # Test with ban-list
        os.environ["I4_ATTN_BACKENDS"] = "-flash3,-flash2"
        backends = get_backend_list(90)
        assert "flash3" not in backends
        assert "flash2" not in backends

    def test_multidim_backend_filtering(self):
        """Test that get_multi_dim_backend_list respects backend filtering."""
        from cosmos_framework.model.attention.backends import get_multi_dim_backend_list

        # Test without env var
        backends = get_multi_dim_backend_list(90)
        assert backends == ["natten"]

        # Test with ban-list
        os.environ["I4_ATTN_BACKENDS_MULTIDIM"] = "-natten"
        backends = get_multi_dim_backend_list(90)
        assert backends == []


@pytest.mark.L0
class TestBackendListOrdering(unittest.TestCase):
    """Pin the per-arch default backend ordering in get_backend_list.

    Ordering encodes known relative performance, so a silent reorder is a
    regression. These tests lock the exact list (with no env-var filtering) for
    each architecture branch, including the SM11x/12x block and its ordering
    difference vs SM100/103.
    """

    def setUp(self):
        """Save and clear backend env vars so defaults are exercised unmodified."""
        self.saved_env = {}
        for var in ["I4_ATTN_BACKENDS", "I4_ATTN_BACKENDS_MULTIDIM", "I4_ATTN_BACKENDS_MERGE"]:
            self.saved_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        """Restore environment variables after each test."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_default_ordering_per_arch(self):
        """Exact default ordering for every get_backend_list branch."""
        from cosmos_framework.model.attention.backends import get_backend_list

        # Below minimum supported arch -> empty.
        assert get_backend_list(74) == []

        # 75 <= arch < 80 -> NATTEN only.
        assert get_backend_list(75) == ["natten"]
        assert get_backend_list(79) == ["natten"]

        # Generic Ampere/Ada (>=80, not a special-cased arch) -> flash2 leads.
        assert get_backend_list(80) == ["flash2", "cudnn", "natten"]
        assert get_backend_list(89) == ["flash2", "cudnn", "natten"]

        # H100 (SM90) -> flash3 leads.
        assert get_backend_list(90) == ["flash3", "cudnn", "natten", "flash2"]

        # SM100/103 -> cudnn, natten, flash2 (flash2 trails natten).
        for arch in (100, 103):
            assert get_backend_list(arch) == ["cudnn", "natten", "flash2"], arch

        # SM110/120/121 -> cudnn, flash2, natten (flash2 ahead of natten).
        for arch in (110, 120, 121):
            assert get_backend_list(arch) == ["cudnn", "flash2", "natten"], arch

    def test_sm11x_12x_vs_sm100_ordering_difference(self):
        """Guard the specific flash2/natten ordering difference between blocks.

        SM100/103 place flash2 *after* natten; SM110/120/121 place flash2
        *before* natten. Assert the relative order explicitly so a future edit to
        either block that accidentally unifies them is caught.
        """
        from cosmos_framework.model.attention.backends import get_backend_list

        sm100 = get_backend_list(100)
        assert sm100.index("natten") < sm100.index("flash2")

        sm120 = get_backend_list(120)
        assert sm120.index("flash2") < sm120.index("natten")


if __name__ == "__main__":
    unittest.main()
