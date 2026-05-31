# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.utils.env_parsers.env_parser import EnvParser
from cosmos_framework.utils.validator import Bool, String


class CustomizationEnvParser(EnvParser):
    FLEET_FUNCTION = Bool(default=False)
    CUSTOMIZATION_TYPE = String(default="")
    DEBUG_SKIP_CUSTOMIZATION_DOWNLOAD = Bool(default=False)
    FT_AWS_ACCESS_KEY_ID = String(default="")
    FT_AWS_SECRET_ACCESS_KEY = String(default="")
    FT_AWS_REGION_NAME = String(default="")
    FT_AWS_GATEWAY_URL = String(default="")
    LAMBDA_STAGE = String(default="prod")


CUSTOMIZATION_ENVS = CustomizationEnvParser()
