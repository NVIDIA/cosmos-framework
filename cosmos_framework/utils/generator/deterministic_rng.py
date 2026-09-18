# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hashlib
import random
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor


def sample_identity_seed(root: object, key: object, epoch: int = 0) -> int:
    """Return the stable seed used by deterministic sample augmentors."""
    payload = f"{root}\0{key}".encode(errors="replace")
    if epoch != 0:
        payload += f"\0{epoch}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sample_seed(data: dict[str, Any]) -> int:
    url = data.get("__url__", "")
    root = getattr(url, "root", url)
    sample_meta = getattr(url, "sample_meta", None)
    epoch = int(data.get("sample_epoch", getattr(sample_meta, "sample_epoch", 0)))
    return sample_identity_seed(root, data.get("__key__", ""), epoch)


class DeterministicAugmentor(Augmentor):
    def __init__(self, augmentor: Augmentor) -> None:
        self.augmentor = augmentor

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict) or "__key__" not in data:
            return self.augmentor(data)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()  # [N_state]
        seed = sample_seed(data)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch_generator = torch.Generator(device="cpu")
        torch_generator.manual_seed(seed)
        seeded_torch_state = torch_generator.get_state()  # [N_state]
        torch.set_rng_state(seeded_torch_state)
        try:
            return self.augmentor(data)
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
