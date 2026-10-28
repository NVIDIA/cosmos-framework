// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>

torch::Tensor decode_rle_streams(
    torch::Tensor data,
    torch::Tensor page_offset,
    torch::Tensor page_length,
    torch::Tensor page_max_rep,
    torch::Tensor page_max_def,
    torch::Tensor page_num_values,
    torch::Tensor page_out_start,
    int total_values
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_rle_streams", &decode_rle_streams,
          "Decode RLE/bit-packing hybrid streams with SMEM fast path",
          py::call_guard<py::gil_scoped_release>());
}
