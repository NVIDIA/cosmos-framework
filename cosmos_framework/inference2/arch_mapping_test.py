# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the forward arch map: read_model_arch reads the reversible arch fields, and
ARCH_MAP + EXCLUDED_TRANSFORMER_ARGS exactly partition the real converter's transformer init."""

import ast
import importlib.util
from pathlib import Path

from cosmos_framework.inference2.arch_mapping import (
    ARCH_MAP,
    EXCLUDED_TRANSFORMER_ARGS,
    read_model_arch,
)


def _sample_model_config() -> dict:
    return {
        "action_gen": True,
        "max_action_dim": 64,
        "num_embodiment_domains": 32,
        "sound_gen": True,
        "sound_dim": 64,
        "sound_latent_fps": 25,
        "state_ch": 48,
        "diffusion_expert_config": {
            "base_fps": 24,
            "enable_fps_modulation": True,
            "patch_spatial": 2,
            "unified_3d_mrope_reset_spatial_ids": True,
            "unified_3d_mrope_temporal_modality_margin": 15000,
        },
    }


def test_read_model_arch_maps_every_reversible_field():
    # each ARCH_MAP entry is read from its model.config path into its transformer key.
    tc = read_model_arch(_sample_model_config())
    assert tc == {
        "action_gen": True,
        "action_dim": 64,  # <- max_action_dim
        "num_embodiment_domains": 32,
        "sound_gen": True,
        "sound_dim": 64,
        "sound_latent_fps": 25,
        "latent_channel": 48,  # <- state_ch
        "base_fps": 24,  # <- diffusion_expert_config.base_fps
        "enable_fps_modulation": True,
        "latent_patch_size": 2,  # <- diffusion_expert_config.patch_spatial
        "unified_3d_mrope_reset_spatial_ids": True,
        "unified_3d_mrope_temporal_modality_margin": 15000,
    }


def test_read_model_arch_skips_absent_fields():
    # fields missing from model.config are simply not emitted (no KeyError).
    assert read_model_arch({"action_gen": False}) == {"action_gen": False}


def test_arch_map_keys_unique_and_disjoint_from_excluded():
    tkeys = [tkey for tkey, _ in ARCH_MAP]
    assert len(tkeys) == len(set(tkeys)), "duplicate transformer key in ARCH_MAP"
    assert set(tkeys).isdisjoint(EXCLUDED_TRANSFORMER_ARGS)


def _converter_transformer_kwargs() -> set[str]:
    """AST-parse the real forward converter for the Cosmos3OmniTransformer(...) kwarg names,
    without importing the module (which pulls diffusers)."""
    spec = importlib.util.find_spec("cosmos_framework.inference2._convert_model_to_diffusers")
    assert spec and spec.origin, "converter module not found"
    tree = ast.parse(Path(spec.origin).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "Cosmos3OmniTransformer":
                return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError("Cosmos3OmniTransformer(...) call not found in the converter source")


def test_arch_map_partitions_converter_transformer_args():
    # The reversible ARCH_MAP keys + the excluded set must EXACTLY cover the real converter's
    # Cosmos3OmniTransformer(...) kwargs. A new/renamed converter arg falls in neither set and
    # fails here, forcing it to be categorized — this is what proves the map tracks reality.
    kwargs = _converter_transformer_kwargs()
    reversible = {tkey for tkey, _ in ARCH_MAP}
    excluded = set(EXCLUDED_TRANSFORMER_ARGS)
    unaccounted = kwargs - reversible - excluded
    stale = (reversible | excluded) - kwargs
    assert not unaccounted, f"converter kwargs not in ARCH_MAP or EXCLUDED: {sorted(unaccounted)}"
    assert not stale, f"ARCH_MAP/EXCLUDED entries not in converter: {sorted(stale)}"
