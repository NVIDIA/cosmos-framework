# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synthetic evidence tests, never counted as campaign executions."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.adapter_campaign.evidence import EvidenceError
from tools.adapter_campaign.segmented import Capture, checkpoint_closure, join_phases


class PhaseTests(unittest.TestCase):
    def fixture(self):
        phases = [{"updates": [1, 2]}, {"updates": [3, 4]}, {"updates": [5, 5]}, {"updates": [6, 7]}]
        logs = [
            [[{"step": i, "rank": r} for i in range(*[p["updates"][0], p["updates"][1] + 1])] for r in range(4)]
            for p in phases
        ]
        return phases, logs

    def test_four_phases_join_exactly_once(self):
        joined = join_phases(*self.fixture(), 7)
        self.assertEqual([[v["step"] for v in rank] for rank in joined], [list(range(1, 8))] * 4)

    def test_replay_is_not_an_extra_phase(self):
        p, logs = self.fixture()
        with self.assertRaises(EvidenceError):
            join_phases(p + [p[1]], logs + [logs[1]], 7)

    def test_missing_duplicate_wrong_rank_or_unfinished_updates_rejected(self):
        mutations = [
            lambda p, l: p[1].update(updates=[2, 4]),
            lambda p, l: p[2].update(updates=[6, 6]),
            lambda p, l: l[2].pop(),
            lambda p, l: l[1][3].pop(),
            lambda p, l: l[1][2].append(l[1][2][0]),
            lambda p, l: l[0][0][0].update(step=10),
            lambda p, l: p[0].update(updates=[True, 2]),
        ]
        for mutate in mutations:
            p, logs = self.fixture()
            mutate(p, logs)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                join_phases(p, logs, 7)
        with self.assertRaises(EvidenceError):
            join_phases(*self.fixture(), 8)


class CheckpointTests(unittest.TestCase):
    def fixture(self):
        generations, large = [], {}
        names = ["data.pt", "trainer.pt", "model/.metadata", "optim/.metadata"]
        names += [f"{part}/__{rank}_0.distcp" for part in ["model", "optim"] for rank in range(4)]
        for step in [348, 467, 600, 850, 935]:
            root = "/remote/checkpoints/iter_%06d" % step
            inventory = {n: {"bytes": 100, "sha256": "1" * 64} for n in names}
            generations.append({"path": root, "files": 12, "bytes": 1200, "inventory": inventory})
            for n, v in inventory.items():
                large[root + "/" + n] = {**v, "pre_post_stat": {"st_size": 100}}
        return SimpleNamespace(
            closure={"checkpoint_generations": generations}, large={"files": large}, resolver={}
        ), root

    def test_all_generations_and_original_remote_tensors_required(self):
        checkpoint_closure(*self.fixture())

    def test_partial_checkpoint_and_stale_tensor_rejected(self):
        mutations = [
            lambda c: c.closure["checkpoint_generations"].pop(),
            lambda c: c.closure["checkpoint_generations"][-1]["inventory"].pop("optim/__3_0.distcp"),
            lambda c: c.closure["checkpoint_generations"][-1].update(bytes=1199),
            lambda c: c.large["files"]["/remote/checkpoints/iter_000935/trainer.pt"].update(sha256="2" * 64),
            lambda c: c.large["files"]["/remote/checkpoints/iter_000467/data.pt"]["pre_post_stat"].update(st_size=99),
            lambda c: c.closure["checkpoint_generations"][-1].update(path="/remote/checkpoints/iter_000934"),
        ]
        for mutate in mutations:
            c, final = self.fixture()
            mutate(c)
            with self.subTest(mutate=mutate), self.assertRaises(EvidenceError):
                checkpoint_closure(c, final)
        with self.assertRaises(EvidenceError):
            checkpoint_closure(self.fixture()[0], "/other/iter_000935")


class CaptureTests(unittest.TestCase):
    def test_raw_bind_requires_review_hash_and_exact_size(self):
        # Test the actual binding primitive independently of an actual audit.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "data.json"
            path.write_bytes(b"{}")
            from tools.adapter_campaign.evidence import Store

            c = Capture.__new__(Capture)
            c.root, c.store, c.references, c.resolver = root, Store(root), {}, {}
            sha = hashlib.sha256(b"{}").hexdigest()
            c.reviewed = {"data.json": sha}
            c.bind("/remote/raw.json", Path("data.json"), {"sha256": sha, "bytes": 2})
            self.assertEqual(c.json("/remote/raw.json"), {})
            with self.assertRaises(EvidenceError):
                c.bind("/remote/raw.json", Path("data.json"), {"sha256": sha, "bytes": 2})
            with self.assertRaises(EvidenceError):
                c.bind("/remote/second.json", Path("data.json"), {"sha256": sha, "bytes": 3})
            with self.assertRaises(EvidenceError):
                c.bind("/remote/second.json", Path("data.json"), {"sha256": "0" * 64, "bytes": 2})
            with self.assertRaises(EvidenceError):
                c.json("/uncaptured/source")
            path.write_bytes(b"[]")
            with self.assertRaises(EvidenceError):
                c.json("/remote/raw.json")


if __name__ == "__main__":
    unittest.main()
