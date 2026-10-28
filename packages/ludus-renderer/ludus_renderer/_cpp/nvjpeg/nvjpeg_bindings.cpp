// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
//
// Python bindings for standalone nvjpeg encoder.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <mutex>
#include "nvjpeg_encoder.h"

namespace py = pybind11;

//------------------------------------------------------------------------
// Per-device encoder instances (lazy initialization)
//------------------------------------------------------------------------

static std::unordered_map<int, NvjpegEncoder> s_encoders;
static std::mutex s_encodersMutex;

static NvjpegEncoder& getEncoder(int device_index)
{
    std::lock_guard<std::mutex> lock(s_encodersMutex);
    auto it = s_encoders.find(device_index);
    if (it == s_encoders.end())
    {
        it = s_encoders.emplace(device_index, NvjpegEncoder(device_index)).first;
    }
    NvjpegEncoder& encoder = it->second;
    if (!encoder.isInitialized())
    {
        if (!encoder.init())
        {
            throw std::runtime_error("Failed to initialize nvjpeg encoder on device " + std::to_string(device_index));
        }
    }
    return encoder;
}

//------------------------------------------------------------------------
// Python-facing functions
//------------------------------------------------------------------------

// Check if nvjpeg is available (on device 0)
bool nvjpeg_is_available()
{
    try
    {
        NvjpegEncoder& encoder = getEncoder(0);
        return encoder.isInitialized();
    }
    catch (...)
    {
        return false;
    }
}

// Encode a batch of images to JPEG
// Input: tensor of shape [B, 3, H, W] or [3, H, W], dtype uint8, on GPU
// device_index: if < 0, use the tensor's device; otherwise use this GPU index
// Returns: list of bytes objects (one per image)
py::list nvjpeg_encode(torch::Tensor images, int quality = 85, int device_index = -1)
{
    // Input validation
    TORCH_CHECK(images.is_cuda(), "Input tensor must be on GPU (CUDA)");
    TORCH_CHECK(images.dtype() == torch::kUInt8, "Input tensor must be uint8");
    TORCH_CHECK(images.is_contiguous(), "Input tensor must be contiguous");

    int ndim = images.dim();
    TORCH_CHECK(ndim == 3 || ndim == 4, "Input must be [3, H, W] or [B, 3, H, W]");

    int batchSize, channels, height, width;
    if (ndim == 3)
    {
        batchSize = 1;
        channels = images.size(0);
        height = images.size(1);
        width = images.size(2);
    }
    else  // ndim == 4
    {
        batchSize = images.size(0);
        channels = images.size(1);
        height = images.size(2);
        width = images.size(3);
    }

    TORCH_CHECK(channels == 3, "Input must have 3 channels (RGB)");
    TORCH_CHECK(quality >= 1 && quality <= 100, "Quality must be in range [1, 100]");

    // device_index >= 0: use that GPU (tensor must be on it). device_index < 0: use tensor's device (lazy-create encoder for it).
    int dev = (device_index >= 0) ? device_index : images.device().index();
    TORCH_CHECK(dev >= 0, "Invalid device index");
    if (device_index >= 0)
        TORCH_CHECK(images.device().index() == dev, "Tensor must be on device ", dev, " when device_index is specified");

    NvjpegEncoder& encoder = getEncoder(dev);

    // Get pointer to data
    const uint8_t* dataPtr = images.data_ptr<uint8_t>();

    // Encode batch
    std::vector<std::vector<uint8_t>> jpegs = encoder.encodeBatch(
        dataPtr, batchSize, width, height, quality);

    // Convert to Python list of bytes
    py::list result;
    for (const auto& jpeg : jpegs)
    {
        result.append(py::bytes(reinterpret_cast<const char*>(jpeg.data()), jpeg.size()));
    }

    return result;
}

// Convenience function for single image
// Input: tensor of shape [3, H, W], dtype uint8, on GPU
// device_index: if < 0, use the tensor's device; otherwise use this GPU index
// Returns: bytes object
py::bytes nvjpeg_encode_single(torch::Tensor image, int quality = 85, int device_index = -1)
{
    // Input validation
    TORCH_CHECK(image.is_cuda(), "Input tensor must be on GPU (CUDA)");
    TORCH_CHECK(image.dtype() == torch::kUInt8, "Input tensor must be uint8");
    TORCH_CHECK(image.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(image.dim() == 3, "Input must be [3, H, W]");
    TORCH_CHECK(image.size(0) == 3, "Input must have 3 channels (RGB)");
    TORCH_CHECK(quality >= 1 && quality <= 100, "Quality must be in range [1, 100]");

    int height = image.size(1);
    int width = image.size(2);

    int dev = (device_index >= 0) ? device_index : image.device().index();
    TORCH_CHECK(dev >= 0, "Invalid device index");
    if (device_index >= 0)
        TORCH_CHECK(image.device().index() == dev, "Tensor must be on device ", dev, " when device_index is specified");

    NvjpegEncoder& encoder = getEncoder(dev);

    // Get pointer to data
    const uint8_t* dataPtr = image.data_ptr<uint8_t>();

    // Encode single image (CHW format)
    std::vector<uint8_t> jpeg = encoder.encodeChw(dataPtr, width, height, quality);

    return py::bytes(reinterpret_cast<const char*>(jpeg.data()), jpeg.size());
}

//------------------------------------------------------------------------
// Module definition
//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "nvjpeg GPU JPEG encoder for PyTorch tensors";

    m.def("is_available", &nvjpeg_is_available,
          "Check if nvjpeg hardware encoder is available");

    m.def("encode", &nvjpeg_encode,
          py::arg("images"),
          py::arg("quality") = 85,
          py::arg("device_index") = -1,
          R"doc(
Encode GPU tensor to JPEG bytes.

Args:
    images: GPU tensor of shape [B, 3, H, W] or [3, H, W], dtype uint8.
            Must be RGB format (not BGR).
    quality: JPEG quality (1-100, default 85)
    device_index: CUDA device index. If < 0 (default), use the tensor's device (encoder lazy-created for that device).

Returns:
    List of bytes objects, one per image in the batch.
)doc");

    m.def("encode_single", &nvjpeg_encode_single,
          py::arg("image"),
          py::arg("quality") = 85,
          py::arg("device_index") = -1,
          R"doc(
Encode a single GPU tensor to JPEG bytes.

Args:
    image: GPU tensor of shape [3, H, W], dtype uint8.
           Must be RGB format (not BGR).
    quality: JPEG quality (1-100, default 85)
    device_index: CUDA device index. If < 0 (default), use the tensor's device (encoder lazy-created for that device).

Returns:
    bytes object containing the JPEG data.
)doc");
}

//------------------------------------------------------------------------
