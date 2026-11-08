// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

extern torch::Tensor decode_rle_streams(
    torch::Tensor data, torch::Tensor page_offset, torch::Tensor page_length,
    torch::Tensor page_max_rep, torch::Tensor page_max_def,
    torch::Tensor page_num_values, torch::Tensor page_out_start,
    int total_values);

extern std::vector<torch::Tensor> gather_and_analyze(
    torch::Tensor rle_rep, torch::Tensor rle_idx,
    torch::Tensor dict_xyz, torch::Tensor dict_ts,
    torch::Tensor file_info_raw,
    int n_files, int total_xyz_values, int total_ts_values, int total_rows);

std::vector<torch::Tensor> rle_gather_pipeline(
    std::vector<torch::Tensor> decompressed_pages,
    // Data page routing (CPU int32)
    torch::Tensor data_page_indices,
    // RLE metadata (CPU int32 tensors, moved to GPU here)
    torch::Tensor rle_max_rep,
    torch::Tensor rle_max_def,
    torch::Tensor rle_num_vals,
    torch::Tensor rle_out_starts,
    int total_rle_values,
    // Dict page routing (CPU int32 tensors)
    torch::Tensor xyz_dict_page_indices,
    torch::Tensor xyz_dict_byte_offsets,
    int total_xyz_dict_bytes,
    torch::Tensor ts_dict_page_indices,
    torch::Tensor ts_dict_byte_offsets,
    int total_ts_dict_bytes,
    // Gather metadata
    torch::Tensor file_info_raw,
    int n_files,
    int total_xyz_values,
    int total_ts_values,
    int total_rows
) {
    TORCH_CHECK(!decompressed_pages.empty(), "decompressed_pages must not be empty");
    auto device = decompressed_pages[0].device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();

    // ── Step 1: Route data pages → concat buffer for RLE ──
    int n_data = data_page_indices.size(0);
    auto dp_idx = data_page_indices.data_ptr<int>();

    int total_data_bytes = 0;
    std::vector<int> data_offs(n_data);
    std::vector<int> data_lens(n_data);
    for (int i = 0; i < n_data; i++) {
        data_offs[i] = total_data_bytes;
        data_lens[i] = (int)decompressed_pages[dp_idx[i]].size(0);
        total_data_bytes += data_lens[i];
    }

    auto dev_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    auto dcat = torch::empty({total_data_bytes + 4}, dev_opts);

    for (int i = 0; i < n_data; i++) {
        auto& page = decompressed_pages[dp_idx[i]];
        if (data_lens[i] > 0) {
            cudaMemcpyAsync(
                (uint8_t*)dcat.data_ptr() + data_offs[i],
                page.data_ptr(), data_lens[i],
                cudaMemcpyDeviceToDevice, stream);
        }
    }
    cudaMemsetAsync((uint8_t*)dcat.data_ptr() + total_data_bytes, 0, 4, stream);

    auto i32_dev = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto meta_offset = torch::tensor(data_offs, i32_dev);
    auto meta_length = torch::tensor(data_lens, i32_dev);

    auto rle_max_rep_dev = rle_max_rep.to(device);
    auto rle_max_def_dev = rle_max_def.to(device);
    auto rle_num_vals_dev = rle_num_vals.to(device);
    auto rle_out_starts_dev = rle_out_starts.to(device);

    // ── Step 2: RLE decode ──
    auto rle_output = decode_rle_streams(
        dcat, meta_offset, meta_length,
        rle_max_rep_dev, rle_max_def_dev, rle_num_vals_dev, rle_out_starts_dev,
        total_rle_values);

    auto out_rep = rle_output.slice(0, 0, total_rle_values);
    auto out_idx = rle_output.slice(0, 2 * total_rle_values);

    // ── Step 3: Route dict pages → flat dict tensors ──
    int n_xyz_dict = xyz_dict_page_indices.size(0);
    int n_ts_dict = ts_dict_page_indices.size(0);
    auto xyz_pi = xyz_dict_page_indices.data_ptr<int>();
    auto xyz_off = xyz_dict_byte_offsets.data_ptr<int>();
    auto ts_pi = ts_dict_page_indices.data_ptr<int>();
    auto ts_off = ts_dict_byte_offsets.data_ptr<int>();

    auto dict_xyz = torch::empty(
        {total_xyz_dict_bytes / (int64_t)sizeof(float)},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto dict_ts = torch::empty(
        {total_ts_dict_bytes / (int64_t)sizeof(int64_t)},
        torch::TensorOptions().dtype(torch::kInt64).device(device));

    for (int i = 0; i < n_xyz_dict; i++) {
        auto& page = decompressed_pages[xyz_pi[i]];
        if (page.size(0) > 0) {
            cudaMemcpyAsync(
                (uint8_t*)dict_xyz.data_ptr() + xyz_off[i],
                page.data_ptr(), page.size(0),
                cudaMemcpyDeviceToDevice, stream);
        }
    }
    for (int i = 0; i < n_ts_dict; i++) {
        auto& page = decompressed_pages[ts_pi[i]];
        if (page.size(0) > 0) {
            cudaMemcpyAsync(
                (uint8_t*)dict_ts.data_ptr() + ts_off[i],
                page.data_ptr(), page.size(0),
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    // ── Step 4: Gather + analyze ──
    return gather_and_analyze(
        out_rep, out_idx, dict_xyz, dict_ts,
        file_info_raw, n_files,
        total_xyz_values, total_ts_values, total_rows);
}
