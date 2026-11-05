# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""tests.helpers — shared test-only support modules (fakes, builders, utils).

Not collected by pytest (no `test_` prefix) and never production code: lightweight
fakes of production protocols live here so test-only code stays out of the source
packages. E.g. `toy_model.ToyFlowModel` (fakes `FlowMatchingPolicy`) and
`preference_env.SyntheticPreferenceEnv` (fakes the Env). Tests import them directly.
"""
