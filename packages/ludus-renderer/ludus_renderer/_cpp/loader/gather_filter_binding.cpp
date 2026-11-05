// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> gather_and_analyze(
    torch::Tensor rle_rep,
    torch::Tensor rle_idx,
    torch::Tensor dict_xyz,
    torch::Tensor dict_ts,
    torch::Tensor file_info_raw,
    int n_files,
    int total_xyz_values,
    int total_ts_values,
    int total_rows
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_and_analyze", &gather_and_analyze,
          "Fused dictionary gather + row analysis for polyline parquets",
          py::call_guard<py::gil_scoped_release>());
}
