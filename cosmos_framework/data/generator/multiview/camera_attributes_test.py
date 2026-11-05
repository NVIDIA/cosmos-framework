# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.data.generator.multiview.camera_attributes import (
    CAMERA_ATTRIBUTES,
    CAMERA_VIEW_MAPPING,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_EXPECTED_FIELDS = {"camera_role", "camera_type", "facing", "fov_degrees"}


def test_every_camera_declares_the_fields_the_caption_header_quotes() -> None:
    """format_separate_view_captions indexes these unguarded, and prints them verbatim."""
    assert len(CAMERA_ATTRIBUTES) == 11
    for camera, attributes in CAMERA_ATTRIBUTES.items():
        assert set(attributes) == _EXPECTED_FIELDS, camera
        assert isinstance(attributes["fov_degrees"], int), camera
        assert all(isinstance(attributes[field], str) for field in _EXPECTED_FIELDS - {"fov_degrees"}), camera




def test_view_ids_match_the_rig_order() -> None:
    """The rig's view IDs index the trained model's view embeddings.

    CAMERA_VIEW_MAPPING is written out by hand so that reordering CAMERA_ATTRIBUTES
    cannot silently renumber the rig -- a derived mapping would stay self-consistent through such a
    reorder, so nothing would catch it. A divergence does not raise anywhere: it feeds one camera's
    frames under another camera's view embedding.

    This assertion is stdlib-only on purpose: it ships with camera_attributes.py so the generated
    copy carries its own guard rather than trusting the repo it was generated from.
    """
    assert CAMERA_VIEW_MAPPING == {name: index for index, name in enumerate(CAMERA_ATTRIBUTES)}
    assert sorted(CAMERA_VIEW_MAPPING.values()) == list(range(len(CAMERA_ATTRIBUTES)))


