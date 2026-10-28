# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Tests for nvjpeg GPU JPEG encoder (device index, encode batch/single).

import pytest
import torch
from ludus_renderer import nvjpeg


def _nvjpeg_available():
    return torch.cuda.is_available() and nvjpeg.is_available()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nvjpeg_is_available_or_raises():
    """Import and is_available() should not raise; result depends on system."""
    nvjpeg.is_available()


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_batch_default_device():
    """Encode batch without device_index uses tensor's device (lazy encoder)."""
    device = torch.device("cuda:0")
    images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, device=device)
    jpegs = nvjpeg.encode(images, quality=85)
    assert len(jpegs) == 2
    for jpeg in jpegs:
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        # JPEG magic
        assert jpeg[:2] == b"\xff\xd8"
        assert jpeg[-2:] == b"\xff\xd9"


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_single_default_device():
    """Encode single image without device_index uses tensor's device."""
    device = torch.device("cuda:0")
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8, device=device)
    jpeg = nvjpeg.encode_single(image, quality=90)
    assert isinstance(jpeg, bytes)
    assert len(jpeg) > 0
    assert jpeg[:2] == b"\xff\xd8"
    assert jpeg[-2:] == b"\xff\xd9"


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_batch_explicit_device_index():
    """Encode with device_index=0 matches default when tensor is on cuda:0."""
    device = torch.device("cuda:0")
    images = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8, device=device)
    jpegs_default = nvjpeg.encode(images, quality=80)
    jpegs_explicit = nvjpeg.encode(images, quality=80, device_index=0)
    assert len(jpegs_default) == 1 and len(jpegs_explicit) == 1
    assert jpegs_default[0] == jpegs_explicit[0]


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_single_explicit_device_index():
    """Encode single with device_index=0 matches default when tensor is on cuda:0."""
    device = torch.device("cuda:0")
    image = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8, device=device)
    jpeg_default = nvjpeg.encode_single(image, quality=80)
    jpeg_explicit = nvjpeg.encode_single(image, quality=80, device_index=0)
    assert jpeg_default == jpeg_explicit


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_wrong_device_raises():
    """Passing device_index that does not match tensor device should raise."""
    device = torch.device("cuda:0")
    images = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.uint8, device=device)
    with pytest.raises(RuntimeError, match="Tensor must be on device"):
        nvjpeg.encode(images, quality=85, device_index=1)


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_single_from_batch_shape():
    """Single image [3, H, W] and batch of one [1, 3, H, W] produce same JPEG (same quality)."""
    device = torch.device("cuda:0")
    image = torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8, device=device)
    jpeg_single = nvjpeg.encode_single(image, quality=75)
    batch = image.unsqueeze(0)
    jpegs_batch = nvjpeg.encode(batch, quality=75)
    assert len(jpegs_batch) == 1
    assert jpeg_single == jpegs_batch[0]


@pytest.mark.skipif(not _nvjpeg_available(), reason="nvjpeg not available")
def test_encode_quality_affects_size():
    """Lower quality should generally produce smaller JPEG bytes."""
    device = torch.device("cuda:0")
    images = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8, device=device)
    jpeg_high = nvjpeg.encode(images, quality=95)[0]
    jpeg_low = nvjpeg.encode(images, quality=20)[0]
    assert len(jpeg_low) < len(jpeg_high)
