# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json

from cosmos_framework.scripts.export_vlm_dcp import (
    fingerprint_model_files,
    infer_vlm_base_model,
    infer_vlm_lora_config,
)


def test_export_inputs_are_inferred_without_rewriting_user_path(tmp_path) -> None:
    supplied = tmp_path / "model-source"
    supplied.mkdir()
    (supplied / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "_target_": "cosmos_framework.model.generator.vlm_model.VLMModel",
                "model": {
                    "config": {
                        "policy": {
                            "backbone": {"model_name": str(supplied)},
                            "lora_enabled": True,
                            "lora_rank": 8,
                            "lora_alpha": 16,
                            "lora_target_modules": "q_proj,v_proj",
                        }
                    }
                },
            }
        )
    )
    assert infer_vlm_base_model(config) == str(supplied)
    assert infer_vlm_lora_config(config)["rank"] == 8
    fingerprint = fingerprint_model_files(supplied)
    assert fingerprint["source"] == str(supplied)
    assert fingerprint["resolved"] == str(supplied.resolve())
    assert fingerprint["files"]["config.json"]
