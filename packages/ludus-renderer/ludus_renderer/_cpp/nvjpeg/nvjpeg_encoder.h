// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
//
// Standalone nvjpeg encoder module.
// Provides GPU-accelerated JPEG encoding for PyTorch tensors.

#pragma once

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <cstdint>
#include <vector>

//------------------------------------------------------------------------
// NvjpegEncoder: Standalone GPU JPEG encoder
//
// Usage:
//   NvjpegEncoder encoder;
//   encoder.init();
//   std::vector<uint8_t> jpeg = encoder.encode(gpuRgbData, width, height, quality);
//   encoder.release();
//------------------------------------------------------------------------

class NvjpegEncoder
{
public:
    // device_index: CUDA device to use. If >= 0, encoder is bound to that device (cudaSetDevice
    // is used in init and encode). If < 0, no cudaSetDevice is called; the current device is used.
    explicit NvjpegEncoder(int device_index = -1);
    ~NvjpegEncoder();

    // Initialize the encoder (call once). If m_deviceIndex >= 0, sets that device first.
    bool init();

    // Check if encoder is initialized
    bool isInitialized() const { return m_initialized; }

    // Encode a single RGB image from GPU memory to JPEG bytes.
    // Input: gpuRgb - pointer to GPU memory with RGB data (3 bytes per pixel, HWC layout)
    // Returns: vector of JPEG bytes
    std::vector<uint8_t> encode(
        const uint8_t* gpuRgb,  // GPU pointer to RGB data [H, W, 3] or [H * W * 3]
        int width,
        int height,
        int quality = 85       // JPEG quality 1-100
    );

    // Encode a single RGB image from GPU memory (CHW layout, typical PyTorch format).
    // Input: gpuRgbChw - pointer to GPU memory with RGB data in CHW layout [3, H, W]
    // Returns: vector of JPEG bytes
    std::vector<uint8_t> encodeChw(
        const uint8_t* gpuRgbChw,  // GPU pointer to RGB data [3, H, W]
        int width,
        int height,
        int quality = 85
    );

    // Encode a batch of RGB images from GPU memory (NCHW layout).
    // Input: gpuRgbNchw - pointer to GPU memory with RGB data [B, 3, H, W]
    // Returns: vector of JPEG byte vectors
    std::vector<std::vector<uint8_t>> encodeBatch(
        const uint8_t* gpuRgbNchw,  // GPU pointer to RGB data [B, 3, H, W]
        int batchSize,
        int width,
        int height,
        int quality = 85
    );

    // Release resources
    void release();

private:
    // CUDA device index for this encoder
    int                     m_deviceIndex;

    // NVJPEG state
    nvjpegHandle_t          m_handle;
    nvjpegEncoderState_t    m_encoderState;
    nvjpegEncoderParams_t   m_encoderParams;
    bool                    m_initialized;

    // Work buffers (GPU)
    uint8_t*                m_hwcBuffer;       // For CHW->HWC conversion
    size_t                  m_hwcBufferSize;

    // Output buffer (pinned host memory)
    uint8_t*                m_outputBuffer;
    size_t                  m_outputBufferSize;

    // CUDA stream for encoding
    cudaStream_t            m_stream;
};

//------------------------------------------------------------------------
// CUDA kernel declarations for format conversion
//------------------------------------------------------------------------

// Convert CHW (PyTorch) to HWC (NVJPEG interleaved) format
void launchChwToHwc(
    const uint8_t* srcChw,  // [3, H, W]
    uint8_t* dstHwc,        // [H, W, 3]
    int width,
    int height,
    cudaStream_t stream
);

// Convert NCHW batch to HWC for single image extraction
void launchNchwToHwc(
    const uint8_t* srcNchw,  // [B, 3, H, W]
    uint8_t* dstHwc,         // [H, W, 3]
    int batchIdx,
    int width,
    int height,
    cudaStream_t stream
);

//------------------------------------------------------------------------
