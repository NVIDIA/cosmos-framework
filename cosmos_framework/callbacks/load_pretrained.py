# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback


class LoadPretrained(Callback):
    """Load HF understanding-pathway weights after DCP resume, gated by checkpoint state.

    Decision table (config flags are *intent*; the runtime probes here decide):
      * Latest checkpoint exists in load dir → DCP loaded the full model. Skip HF load.
      * No latest checkpoint, ``load_path`` set → DCP loaded full model from warm-start.
        Reload HF understanding pathway (e.g. swap Qwen3-VL → Cosmos-Reason) but skip
        the understanding→generation copy.
      * Neither → fresh init: full HF load + understanding→generation copy.

    Reads ``self.config.checkpoint`` / ``self.config.job`` (injected by
    ``CallBackGroup`` after instantiate) to build a probe checkpointer.
    """

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        from cosmos_framework.checkpoint.dcp import DistributedCheckpointer

        probe = DistributedCheckpointer(self.config.checkpoint, self.config.job, callbacks=None, disable_async=True)
        model.load_pretrained_model_if_needed(
            has_resumable_checkpoint=probe.has_resumable_checkpoint(),
            has_load_path=probe.load_path is not None,
        )
