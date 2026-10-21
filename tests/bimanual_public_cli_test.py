# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The recorded-action recipe must be usable without enabling training features."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class BimanualPublicCliTests(unittest.TestCase):
    def test_recipe_flags_parse_when_training_is_disabled(self):
        root = Path(__file__).resolve().parents[1]
        # Training changes CLI annotations at import time, so use a fresh process.
        script = textwrap.dedent(
            """
            import os

            import tyro

            from cosmos_framework.inference.args import ModelMode
            from cosmos_framework.inference.common.args import tyro_cli
            from cosmos_framework.scripts.inference import InferenceArgs
            from cosmos_framework.utils.flags import TRAINING

            assert not TRAINING
            args = tyro_cli(
                InferenceArgs,
                args=[
                    '-i', os.environ['BIMANUAL_TEST_INPUT_MANIFEST'],
                    '-o', 'outputs/test',
                    '--checkpoint-path', './Cosmos3-Nano-Sim-Bimanual',
                    '--model-mode=FORWARD_DYNAMICS', '--resolution=480', '--image-size=480',
                    '--num-steps=4', '--seed=2026',
                    '--autoregressive=True', '--parallelism-preset=latency', '--max-num-seqs=1',
                    '--no-use-torch-compile', '--no-diffusion-cache', '--no-guardrails',
                    '--video-save-quality=4',
                ],
                config=(tyro.conf.OmitArgPrefixes, tyro.conf.CascadeSubcommandArgs,
                        tyro.conf.OmitSubcommandPrefixes),
            )
            sample = args.setup.sample_overrides
            assert (sample.image_size, sample.num_steps, sample.video_save_quality) == (480, 4, 4)
            assert sample.action_chunk_size is None and sample.num_frames is None
            assert sample.domain_name is None and sample.action_path is None
            assert sample.model_mode == ModelMode.FORWARD_DYNAMICS
            rows = type(sample).from_files(args.input_files, overrides=sample)
            assert [(row.domain_name, row.action_chunk_size, row.num_frames) for row in rows] == [
                ('agibotworld', 4, 5), ('xdof_yam', 900, 901), ('hand_pose', 1800, 1801),
            ]
            assert all(row.raw_action_dim == 59 and row.fps == 30 for row in rows)
            assert all(str(row.action_path).endswith(row.domain_name + '.json') for row in rows)
            assert {mode.value for mode in ModelMode} >= {'forward_dynamics', 'inverse_dynamics', 'wam', 'text2video'}
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "examples.jsonl"
            rows = []
            for domain, count in [("agibotworld", 4), ("xdof_yam", 900), ("hand_pose", 1800)]:
                action_path = Path(directory) / f"{domain}.json"
                action_path.write_text("{}")
                rows.append(
                    {
                        "name": domain,
                        "domain_name": domain,
                        "action_path": str(action_path),
                        "raw_action_dim": 59,
                        "action_chunk_size": count,
                        "num_frames": count + 1,
                        "fps": 30,
                    }
                )
            manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            environment = dict(os.environ)
            environment.update(
                BIMANUAL_TEST_INPUT_MANIFEST=str(manifest_path),
                COSMOS_TRAINING="0",
                COSMOS_INTERNAL="0",
                COSMOS_DEVICE="cpu",
                CUDA_VISIBLE_DEVICES="",
                PYTHONPATH=str(root),
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
