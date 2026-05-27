# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.utils.env_parsers.env_parser import EnvParser
from cosmos_framework.utils.validator import Bool, Int, String


class InferenceEnvParser(EnvParser):
    MODEL_MODULE = String(default=None)
    MODEL_CLASS = String(default=None)
    TORCH_HOME = String(default="/config/models/checkpoints")
    TRT_ENABLED = Bool(default=False)
    PORT = Int(default=8000)
    CP_SIZE = Int(default=1)
    TP_SIZE = Int(default=1)
    FSDP_ENABLED = Bool(default=False)
    CUSTOMIZATION_TYPE = String(default="")
    NIM_DEPLOYMENT = Bool(default=False)
    RUNAI_DEPLOYMENT = Bool(default=False)
    BLUR_CUDA = Bool(default=False)
    RESIZE_CUDA = Bool(default=False)


INFERENCE_ENVS = InferenceEnvParser()

if __name__ == "__main__":
    INFERENCE_ENVS.to_json("env_params.json")

    b64 = INFERENCE_ENVS.to_b64()

    env_params_restored = InferenceEnvParser(b64)

    print(env_params_restored)
