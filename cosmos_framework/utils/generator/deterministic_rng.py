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
    local_data_prefix = getattr(getattr(url, "meta", None), "opts", {}).get("local_data_prefix")
    if isinstance(root, str) and isinstance(local_data_prefix, str):
        root = root.removeprefix(local_data_prefix.rstrip("/") + "/")
    sample_meta = getattr(url, "sample_meta", None)
    epoch = int(data.get("sample_epoch", getattr(sample_meta, "sample_epoch", 0)))
    return sample_identity_seed(root, data.get("__key__", ""), epoch)


class DeterministicAugmentor(Augmentor):
    def __init__(self, augmentor: Augmentor, *, fixed_sample_epoch: int | None = None) -> None:
        """Optionally pin the augmented view for a measured offline packing catalog."""
        if fixed_sample_epoch is not None and (type(fixed_sample_epoch) is not int or fixed_sample_epoch < 0):
            raise ValueError("fixed_sample_epoch must be a non-negative integer or None.")
        self.augmentor: Augmentor = augmentor
        self.fixed_sample_epoch: int | None = fixed_sample_epoch

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.fixed_sample_epoch is not None:
            if not isinstance(data, dict) or not isinstance(data.get("__key__"), str) or not data["__key__"]:
                raise ValueError("A fixed augmented view requires a non-empty sample key.")
            url = data.get("__url__", "")
            root = getattr(url, "root", url)
            if not isinstance(root, str) or not root:
                raise ValueError("A fixed augmented view requires a non-empty source root.")
        if not isinstance(data, dict) or "__key__" not in data:
            return self.augmentor(data)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()  # [N_state]
        # Only the RNG identity is pinned; preserve the caller's epoch metadata.
        seed_data = data if self.fixed_sample_epoch is None else {**data, "sample_epoch": self.fixed_sample_epoch}
        seed = sample_seed(seed_data)
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
