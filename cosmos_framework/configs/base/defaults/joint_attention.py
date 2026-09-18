# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Which joint understanding + generation attention pathway a run takes, and how it packs.

Its own module, and a leaf one: the vocabulary is shared between the config layer that sets it
and the model layer that reads it, and ``model_config`` cannot serve as that home because it
imports from ``models``, which would make every attention module importing it circular.
"""

from typing import Literal

# * ``"two_way"``: the ordinary dense within-sample GEN attention.
# * ``"three_way"``: the split the NATTEN sparsity path needs; use it only for that.
# * ``"multiview"``: the multiview-aware GEN attention, whose UND pass is shared and whose GEN
#   pass runs as ``MultiviewAttentionConfig.backend`` selects -- a masked FlexAttention call or
#   the maskless folds. This is what turns multiview attention on; there is no second flag.
JointAttnImplementation = Literal["two_way", "three_way", "multiview"]

# How the packer and the context-parallel sharder lay a pack out. Only two shapes exist, and
# ``build_packed_sequence`` is what knows them.
PackingLayout = Literal["two_way", "three_way"]


def packing_layout(joint_attn_implementation: JointAttnImplementation) -> PackingLayout:
    """The pack shape a pathway needs, which is not always its own name.

    "multiview" is a pathway rather than a pack shape: it changes which attention the GEN tokens
    run, and nothing about how the pack is laid down. Deriving that here keeps the knowledge in
    one place -- a site that asked ``impl == "two_way"`` about packing would silently mis-pack a
    multiview run, and the failure would be a wrong answer rather than an error.
    """
    return "three_way" if joint_attn_implementation == "three_way" else "two_way"
