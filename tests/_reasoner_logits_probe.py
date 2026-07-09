# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""torchrun entry that captures the reasoner's first-token logits, then runs
the normal inference CLI.

Launched by ``nano_reasoner_inference_smoke_test.py`` via::

    REASONER_LOGITS_DUMP=<path> torchrun --nproc_per_node=4 \
        tests/_reasoner_logits_probe.py <inference CLI args...>

It (1) pins deterministic kernels so the first-token logits are reproducible
run-to-run on the same GPU config, (2) monkey-patches the module-global
``unified_mot._sample_next_token`` so its FIRST invocation (global rank 0)
saves the first-token logits to ``$REASONER_LOGITS_DUMP``, then (3) forwards
``sys.argv`` to ``cosmos_framework.scripts.inference.main``.

Greedy decode (the reasoner default ``do_sample=false``) consumes no sampling
RNG, so the saved logits depend only on the checkpoint + prompt + kernels.
"""

from __future__ import annotations

import os


def _install_determinism() -> None:
    # Must be set before the first cuBLAS call for deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Make flash-attention kernels deterministic (no atomic-add reductions).
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")

    import torch

    # warn_only: degrade (don't crash) on any op lacking a deterministic impl.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _install_logits_probe(dump_path: str) -> None:
    import torch

    import cosmos_framework.model.generator.mot.unified_mot as unified_mot

    original = unified_mot._sample_next_token
    state = {"saved": False}

    def _patched(logits, *args, **kwargs):
        # ``logits`` is [B, vocab] for the token being sampled. The first call
        # in a generation run is the first decoded token after the prompt.
        if not state["saved"]:
            state["saved"] = True
            if int(os.environ.get("RANK", "0")) == 0:
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                # Sample 0, full vocab, fp32 on CPU — stable to torch.load anywhere.
                torch.save(logits[0].detach().float().cpu(), dump_path)
        return original(logits, *args, **kwargs)

    unified_mot._sample_next_token = _patched


def main() -> None:
    _install_determinism()
    dump_path = os.environ["REASONER_LOGITS_DUMP"]
    _install_logits_probe(dump_path)

    from cosmos_framework.scripts.inference import main as inference_main

    inference_main()


if __name__ == "__main__":
    main()
