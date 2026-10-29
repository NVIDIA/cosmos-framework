// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include <nvcomp/snappy.h>

std::vector<torch::Tensor> batch_snappy_decompress(
    torch::Tensor gpu_buffer,
    torch::Tensor data_offsets,
    torch::Tensor comp_sizes,
    torch::Tensor uncomp_sizes
) {
    TORCH_CHECK(gpu_buffer.is_cuda(), "gpu_buffer must be on CUDA");
    TORCH_CHECK(gpu_buffer.dtype() == torch::kUInt8, "gpu_buffer must be uint8");

    int n = data_offsets.size(0);
    if (n == 0) return {};

    auto device = gpu_buffer.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    uint8_t* buf_base = gpu_buffer.data_ptr<uint8_t>();

    auto offsets_cpu = data_offsets.to(torch::kCPU, torch::kInt64).contiguous();
    auto comp_cpu = comp_sizes.to(torch::kCPU, torch::kInt64).contiguous();
    auto uncomp_cpu = uncomp_sizes.to(torch::kCPU, torch::kInt64).contiguous();

    auto* off_ptr = offsets_cpu.data_ptr<int64_t>();
    auto* comp_ptr = comp_cpu.data_ptr<int64_t>();
    auto* uncomp_ptr = uncomp_cpu.data_ptr<int64_t>();

    auto pin_opts = torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true);
    auto h_comp_ptrs_t = torch::empty({(int64_t)(n * sizeof(void*))}, pin_opts);
    auto h_comp_bytes_t = torch::empty({(int64_t)(n * sizeof(size_t))}, pin_opts);
    auto h_uncomp_bytes_t = torch::empty({(int64_t)(n * sizeof(size_t))}, pin_opts);
    auto h_out_ptrs_t = torch::empty({(int64_t)(n * sizeof(void*))}, pin_opts);

    auto* h_comp_ptrs = reinterpret_cast<const void**>(h_comp_ptrs_t.data_ptr());
    auto* h_comp_bytes = reinterpret_cast<size_t*>(h_comp_bytes_t.data_ptr());
    auto* h_uncomp_bytes = reinterpret_cast<size_t*>(h_uncomp_bytes_t.data_ptr());
    auto* h_out_ptrs = reinterpret_cast<void**>(h_out_ptrs_t.data_ptr());

    auto dev_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    std::vector<torch::Tensor> outputs(n);

    size_t max_uncomp = 0;
    size_t total_uncomp = 0;

    for (int i = 0; i < n; i++) {
        h_comp_ptrs[i] = buf_base + off_ptr[i];
        h_comp_bytes[i] = (size_t)comp_ptr[i];
        h_uncomp_bytes[i] = (size_t)uncomp_ptr[i];
        if (h_uncomp_bytes[i] > max_uncomp) max_uncomp = h_uncomp_bytes[i];
        total_uncomp += h_uncomp_bytes[i];

        outputs[i] = torch::empty({(int64_t)h_uncomp_bytes[i]}, dev_opts);
        h_out_ptrs[i] = outputs[i].data_ptr<uint8_t>();
    }

    auto d_comp_ptrs = torch::empty({(int64_t)(n * sizeof(void*))}, dev_opts);
    auto d_comp_bytes = torch::empty({(int64_t)(n * sizeof(size_t))}, dev_opts);
    auto d_uncomp_buf_bytes = torch::empty({(int64_t)(n * sizeof(size_t))}, dev_opts);
    auto d_uncomp_actual = torch::empty({(int64_t)(n * sizeof(size_t))}, dev_opts);
    auto d_statuses = torch::empty({(int64_t)(n * sizeof(nvcompStatus_t))}, dev_opts);
    auto d_out_ptrs = torch::empty({(int64_t)(n * sizeof(void*))}, dev_opts);

    cudaMemcpyAsync(d_comp_ptrs.data_ptr(), h_comp_ptrs, n * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_comp_bytes.data_ptr(), h_comp_bytes, n * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_uncomp_buf_bytes.data_ptr(), h_uncomp_bytes, n * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_ptrs.data_ptr(), h_out_ptrs, n * sizeof(void*), cudaMemcpyHostToDevice, stream);

    size_t temp_bytes = 0;
    nvcompBatchedSnappyDecompressOpts_t opts = nvcompBatchedSnappyDecompressDefaultOpts;

    auto status = nvcompBatchedSnappyDecompressGetTempSizeAsync(
        (size_t)n, max_uncomp, opts, &temp_bytes, total_uncomp);
    TORCH_CHECK(status == nvcompSuccess,
        "nvcompBatchedSnappyDecompressGetTempSizeAsync failed: ", (int)status);

    auto d_temp = torch::empty({std::max((int64_t)temp_bytes, (int64_t)1)}, dev_opts);

    status = nvcompBatchedSnappyDecompressAsync(
        (const void* const*)d_comp_ptrs.data_ptr(),
        (const size_t*)d_comp_bytes.data_ptr(),
        (const size_t*)d_uncomp_buf_bytes.data_ptr(),
        (size_t*)d_uncomp_actual.data_ptr(),
        (size_t)n,
        d_temp.data_ptr(),
        temp_bytes,
        (void* const*)d_out_ptrs.data_ptr(),
        opts,
        (nvcompStatus_t*)d_statuses.data_ptr(),
        stream);
    TORCH_CHECK(status == nvcompSuccess,
        "nvcompBatchedSnappyDecompressAsync failed: ", (int)status);

    return outputs;
}
