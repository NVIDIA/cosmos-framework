# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training callback that defers AOT compilation of the VAE tokenizer.

The actual compilation logic lives in
:meth:`~cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface.compile_encode`.
This module provides a :class:`CompileTokenizer` callback that invokes it
at the right point during training (after ``compile_after_iterations``
steps, to avoid NCCL timeouts during CUDA/cuDNN warm-up).

Typical config usage
--------------------
.. code-block:: python

    CompileTokenizer(
        enabled=True,
        compile_after_iterations=3,
        warmup_resolutions=["256", "480", "720"],
    )
"""

from collections.abc import Sequence
from typing import Literal

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class CompileTokenizer(Callback):
    """Training callback that defers AOT compilation of the VAE tokenizer.

    Hooks into ``on_training_step_start``.  On the
    ``compile_after_iterations``-th step it calls
    ``Wan2pt2VAEInterface.compile_encode`` to compile and load all chunk
    variants.  Every subsequent step is a no-op.
    """

    def __init__(
        self,
        enabled: bool = False,
        compile_after_iterations: int = 3,
        warmup_resolutions: Sequence[str] | None = None,
        backend: Literal["cudagraphs", "inductor"] = "inductor",
        mode: Literal["reduce-overhead", "max-autotune"] | None = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ):
        """
        Args:
            enabled: Master switch.  When ``False`` the callback is a
                complete no-op and no compilation occurs.
            compile_after_iterations: How many training steps to skip
                before triggering compilation.  The default (3) lets CUDA
                context setup and Transformer compilation finish first.
            warmup_resolutions: Resolution keys (e.g. ``["256", "480", "720"]``)
                to AOT-compile.  Should include every resolution used in
                training.  Must be a non-empty list when *enabled* is ``True``.
        """
        super().__init__()
        self.enabled: bool = enabled
        self.compile_after_iterations: int = compile_after_iterations
        self.skip_counter: int = 0
        self.warmup_resolutions: Sequence[str] | None = warmup_resolutions
        self.backend: Literal["cudagraphs", "inductor"] = backend
        self.mode: Literal["reduce-overhead", "max-autotune"] | None = mode
        self.fullgraph: bool = fullgraph
        self.dynamic: bool = dynamic

        if self.enabled:
            if self.warmup_resolutions is None:
                raise ValueError("warmup_resolutions must be provided when enabled, got None")
            if len(self.warmup_resolutions) == 0:
                raise ValueError("warmup_resolutions must be a non-empty list when enabled, got an empty list")

    def on_training_step_start(
        self, model: OmniMoTModel, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """Called at the start of every training step.

        On the ``compile_after_iterations``-th call, triggers AOT compilation
        via ``tokenizer.compile_encode``.

        Args:
            model: The OmniMoTModel whose ``tokenizer_vision_gen`` will be compiled.
            data_batch: Current training batch (unused, required by Callback API).
            iteration: Current training iteration (unused; we track our own counter
                via ``skip_counter`` because this callback may be registered after
                iteration 0).
        """
        if not self.enabled:
            return

        tokenizer = model.tokenizer_vision_gen

        if isinstance(tokenizer, torch.jit.ScriptModule):
            log.critical(
                f"The Tokenizer model {type(tokenizer)} is a JIT model, "
                "which is not compilable. The Tokenizer will not be compiled.",
                rank0_only=False,
            )
            self.enabled = False
            return

        if self.skip_counter == self.compile_after_iterations:
            if self.warmup_resolutions is not None:
                tokenizer.compile_encode(
                    self.warmup_resolutions,
                    output_dir=self.config.job.path_local,
                    backend=self.backend,
                    mode=self.mode,
                    fullgraph=self.fullgraph,
                    dynamic=self.dynamic,
                )

        self.skip_counter += 1
