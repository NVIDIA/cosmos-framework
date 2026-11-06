// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
//
// Standalone nvjpeg encoder implementation.

#include "nvjpeg_encoder.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

//------------------------------------------------------------------------
// CUDA kernels for format conversion
//------------------------------------------------------------------------

// Convert CHW (PyTorch) to HWC (NVJPEG interleaved) format
// Input:  [3, H, W] - channel-first layout
// Output: [H, W, 3] - interleaved layout
__global__ void chwToHwcKernel(
    const uint8_t* __restrict__ srcChw,
    uint8_t* __restrict__ dstHwc,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int hwSize = height * width;
    int srcIdxR = 0 * hwSize + y * width + x;  // Channel 0 (R)
    int srcIdxG = 1 * hwSize + y * width + x;  // Channel 1 (G)
    int srcIdxB = 2 * hwSize + y * width + x;  // Channel 2 (B)

    int dstIdx = (y * width + x) * 3;
    dstHwc[dstIdx + 0] = srcChw[srcIdxR];
    dstHwc[dstIdx + 1] = srcChw[srcIdxG];
    dstHwc[dstIdx + 2] = srcChw[srcIdxB];
}

// Convert NCHW batch to HWC for single image extraction
// Input:  [B, 3, H, W] - batch of channel-first images
// Output: [H, W, 3] - single interleaved image
__global__ void nchwToHwcKernel(
    const uint8_t* __restrict__ srcNchw,
    uint8_t* __restrict__ dstHwc,
    int batchIdx,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int hwSize = height * width;
    int imageSize = 3 * hwSize;
    int baseOffset = batchIdx * imageSize;

    int srcIdxR = baseOffset + 0 * hwSize + y * width + x;
    int srcIdxG = baseOffset + 1 * hwSize + y * width + x;
    int srcIdxB = baseOffset + 2 * hwSize + y * width + x;

    int dstIdx = (y * width + x) * 3;
    dstHwc[dstIdx + 0] = srcNchw[srcIdxR];
    dstHwc[dstIdx + 1] = srcNchw[srcIdxG];
    dstHwc[dstIdx + 2] = srcNchw[srcIdxB];
}

// Host launcher functions
void launchChwToHwc(
    const uint8_t* srcChw,
    uint8_t* dstHwc,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    chwToHwcKernel<<<grid, block, 0, stream>>>(srcChw, dstHwc, width, height);
}

void launchNchwToHwc(
    const uint8_t* srcNchw,
    uint8_t* dstHwc,
    int batchIdx,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    nchwToHwcKernel<<<grid, block, 0, stream>>>(srcNchw, dstHwc, batchIdx, width, height);
}

//------------------------------------------------------------------------
// NvjpegEncoder implementation
//------------------------------------------------------------------------

NvjpegEncoder::NvjpegEncoder(int device_index)
    : m_deviceIndex(device_index)
    , m_handle(nullptr)
    , m_encoderState(nullptr)
    , m_encoderParams(nullptr)
    , m_initialized(false)
    , m_hwcBuffer(nullptr)
    , m_hwcBufferSize(0)
    , m_outputBuffer(nullptr)
    , m_outputBufferSize(0)
    , m_stream(nullptr)
{
}

NvjpegEncoder::~NvjpegEncoder()
{
    release();
}

bool NvjpegEncoder::init()
{
    if (m_initialized)
        return true;

    // If device index is set, use it; otherwise use current device (no cudaSetDevice)
    if (m_deviceIndex >= 0)
    {
        cudaError_t cudaErr = cudaSetDevice(m_deviceIndex);
        if (cudaErr != cudaSuccess)
            return false;
    }

    // Create CUDA stream
    cudaError_t cudaErr = cudaStreamCreate(&m_stream);
    if (cudaErr != cudaSuccess)
        return false;

    // Initialize nvjpeg
    nvjpegStatus_t status;

    status = nvjpegCreateSimple(&m_handle);
    if (status != NVJPEG_STATUS_SUCCESS)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        return false;
    }

    status = nvjpegEncoderStateCreate(m_handle, &m_encoderState, m_stream);
    if (status != NVJPEG_STATUS_SUCCESS)
    {
        nvjpegDestroy(m_handle);
        m_handle = nullptr;
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        return false;
    }

    status = nvjpegEncoderParamsCreate(m_handle, &m_encoderParams, m_stream);
    if (status != NVJPEG_STATUS_SUCCESS)
    {
        nvjpegEncoderStateDestroy(m_encoderState);
        m_encoderState = nullptr;
        nvjpegDestroy(m_handle);
        m_handle = nullptr;
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        return false;
    }

    m_initialized = true;
    return true;
}

void NvjpegEncoder::release()
{
    if (m_deviceIndex >= 0)
        cudaSetDevice(m_deviceIndex);

    if (m_hwcBuffer)
    {
        cudaFree(m_hwcBuffer);
        m_hwcBuffer = nullptr;
        m_hwcBufferSize = 0;
    }

    if (m_outputBuffer)
    {
        cudaFreeHost(m_outputBuffer);
        m_outputBuffer = nullptr;
        m_outputBufferSize = 0;
    }

    if (m_encoderParams)
    {
        nvjpegEncoderParamsDestroy(m_encoderParams);
        m_encoderParams = nullptr;
    }

    if (m_encoderState)
    {
        nvjpegEncoderStateDestroy(m_encoderState);
        m_encoderState = nullptr;
    }

    if (m_handle)
    {
        nvjpegDestroy(m_handle);
        m_handle = nullptr;
    }

    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }

    m_initialized = false;
}

std::vector<uint8_t> NvjpegEncoder::encode(
    const uint8_t* gpuRgb,
    int width,
    int height,
    int quality)
{
    if (!m_initialized)
    {
        throw std::runtime_error("NvjpegEncoder not initialized");
    }

    if (m_deviceIndex >= 0)
        cudaSetDevice(m_deviceIndex);

    // Set quality and sampling
    nvjpegEncoderParamsSetQuality(m_encoderParams, quality, m_stream);
    nvjpegEncoderParamsSetSamplingFactors(m_encoderParams, NVJPEG_CSS_420, m_stream);

    // Setup input image (already HWC/interleaved format)
    nvjpegImage_t nvImage;
    memset(&nvImage, 0, sizeof(nvImage));
    nvImage.channel[0] = const_cast<uint8_t*>(gpuRgb);
    nvImage.pitch[0] = width * 3;  // RGB stride

    // Encode
    nvjpegStatus_t status = nvjpegEncodeImage(
        m_handle,
        m_encoderState,
        m_encoderParams,
        &nvImage,
        NVJPEG_INPUT_RGBI,
        width,
        height,
        m_stream
    );

    if (status != NVJPEG_STATUS_SUCCESS)
    {
        throw std::runtime_error("nvjpegEncodeImage failed");
    }

    // Get compressed size
    size_t compressedSize = 0;
    status = nvjpegEncodeRetrieveBitstream(
        m_handle,
        m_encoderState,
        nullptr,
        &compressedSize,
        m_stream
    );

    if (status != NVJPEG_STATUS_SUCCESS || compressedSize == 0)
    {
        throw std::runtime_error("Failed to get JPEG bitstream size");
    }

    // Sync stream before retrieving data
    cudaStreamSynchronize(m_stream);

    // Reallocate output buffer if needed
    if (compressedSize > m_outputBufferSize)
    {
        if (m_outputBuffer)
            cudaFreeHost(m_outputBuffer);
        size_t allocSize = compressedSize + (compressedSize >> 2);  // 25% headroom
        cudaHostAlloc(reinterpret_cast<void**>(&m_outputBuffer), allocSize, cudaHostAllocDefault);
        m_outputBufferSize = allocSize;
    }

    // Retrieve compressed data
    status = nvjpegEncodeRetrieveBitstream(
        m_handle,
        m_encoderState,
        m_outputBuffer,
        &compressedSize,
        0  // Default stream for retrieval
    );

    if (status != NVJPEG_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to retrieve JPEG bitstream");
    }

    cudaStreamSynchronize(0);

    // Copy to output vector
    return std::vector<uint8_t>(m_outputBuffer, m_outputBuffer + compressedSize);
}

std::vector<uint8_t> NvjpegEncoder::encodeChw(
    const uint8_t* gpuRgbChw,
    int width,
    int height,
    int quality)
{
    if (!m_initialized)
    {
        throw std::runtime_error("NvjpegEncoder not initialized");
    }

    if (m_deviceIndex >= 0)
        cudaSetDevice(m_deviceIndex);

    // Allocate HWC buffer if needed
    size_t hwcSize = (size_t)width * height * 3;
    if (hwcSize > m_hwcBufferSize)
    {
        if (m_hwcBuffer)
            cudaFree(m_hwcBuffer);
        size_t allocSize = hwcSize + (hwcSize >> 2);  // 25% headroom
        cudaMalloc(reinterpret_cast<void**>(&m_hwcBuffer), allocSize);
        m_hwcBufferSize = allocSize;
    }

    // Convert CHW to HWC
    launchChwToHwc(gpuRgbChw, m_hwcBuffer, width, height, m_stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("CHW to HWC conversion failed");
    }

    // Encode the HWC buffer
    return encode(m_hwcBuffer, width, height, quality);
}

std::vector<std::vector<uint8_t>> NvjpegEncoder::encodeBatch(
    const uint8_t* gpuRgbNchw,
    int batchSize,
    int width,
    int height,
    int quality)
{
    if (!m_initialized)
    {
        throw std::runtime_error("NvjpegEncoder not initialized");
    }

    if (m_deviceIndex >= 0)
        cudaSetDevice(m_deviceIndex);

    // Allocate HWC buffer if needed
    size_t hwcSize = (size_t)width * height * 3;
    if (hwcSize > m_hwcBufferSize)
    {
        if (m_hwcBuffer)
            cudaFree(m_hwcBuffer);
        size_t allocSize = hwcSize + (hwcSize >> 2);  // 25% headroom
        cudaMalloc(reinterpret_cast<void**>(&m_hwcBuffer), allocSize);
        m_hwcBufferSize = allocSize;
    }

    std::vector<std::vector<uint8_t>> results;
    results.reserve(batchSize);

    for (int i = 0; i < batchSize; i++)
    {
        // Convert this batch element from NCHW to HWC
        launchNchwToHwc(gpuRgbNchw, m_hwcBuffer, i, width, height, m_stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("NCHW to HWC conversion failed");
        }

        // Encode the HWC buffer
        results.push_back(encode(m_hwcBuffer, width, height, quality));
    }

    return results;
}

//------------------------------------------------------------------------
