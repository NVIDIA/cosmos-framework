# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Non-Bimanual families keep their native path through the action contract loader."""

import json
import tempfile
import unittest
from pathlib import Path

from cosmos_framework.inference.bimanual import load_bimanual_action_contract

ROOT = Path(__file__).resolve().parents[1]
TRANSFER_FIXTURE = ROOT / "tests/data/transfer_conditioning_test"
LEGACY_FIXTURE = ROOT / "tests/data/bimanual_legacy_action_test"
UNIFIED_FIXTURE = ROOT / "tests/data/bimanual_unified_action_test"


class TransferConditioningTests(unittest.TestCase):
    def test_transfer_conditioning_has_no_action_contract(self):
        """Sim-Transfer reuses the metadata block for control conditioning, not actions."""
        self.assertIsNone(load_bimanual_action_contract(TRANSFER_FIXTURE))

    def test_checkpoint_without_metadata_has_no_action_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "transformer").mkdir()
            (root / "transformer/config.json").write_text(json.dumps({"hidden_size": 16}))
            self.assertIsNone(load_bimanual_action_contract(root))

    def test_bimanual_schemas_still_resolve(self):
        """The fix must not loosen validation for real Bimanual exports."""
        self.assertIsNone(load_bimanual_action_contract(LEGACY_FIXTURE))
        self.assertIsNotNone(load_bimanual_action_contract(UNIFIED_FIXTURE))

    def test_versionless_action_conditioning_rejected(self):
        """Only the Transfer shape may omit schema_version; an action export may not."""
        for conditioning in ({"mode": "action"}, {"model_action_dim": 64}):
            with self.subTest(conditioning=conditioning), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "transformer").mkdir()
                (root / "transformer/config.json").write_text(
                    json.dumps({"cosmos3_nano_sim_bimanual": {"conditioning": conditioning}})
                )
                with self.assertRaisesRegex(ValueError, "missing schema_version"):
                    load_bimanual_action_contract(root)

    def test_unknown_action_schema_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "transformer").mkdir()
            (root / "transformer/config.json").write_text(
                json.dumps({"cosmos3_nano_sim_bimanual": {"conditioning": {"schema_version": 4}}})
            )
            with self.assertRaises(ValueError):
                load_bimanual_action_contract(root)


if __name__ == "__main__":
    unittest.main()
