// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rle_gather_pipeline(
    std::vector<torch::Tensor> decompressed_pages,
    torch::Tensor data_page_indices,
    torch::Tensor rle_max_rep,
    torch::Tensor rle_max_def,
    torch::Tensor rle_num_vals,
    torch::Tensor rle_out_starts,
    int total_rle_values,
    torch::Tensor xyz_dict_page_indices,
    torch::Tensor xyz_dict_byte_offsets,
    int total_xyz_dict_bytes,
    torch::Tensor ts_dict_page_indices,
    torch::Tensor ts_dict_byte_offsets,
    int total_ts_dict_bytes,
    torch::Tensor file_info_raw,
    int n_files,
    int total_xyz_values,
    int total_ts_values,
    int total_rows
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rle_gather_pipeline", &rle_gather_pipeline,
          "Fused RLE decode + gather pipeline (GIL-free)",
          py::call_guard<py::gil_scoped_release>());
}
