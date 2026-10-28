// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> batch_snappy_decompress(
    torch::Tensor gpu_buffer,
    torch::Tensor data_offsets,
    torch::Tensor comp_sizes,
    torch::Tensor uncomp_sizes
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_snappy_decompress", &batch_snappy_decompress,
          "Batch Snappy decompress via nvcomp C API (GIL-free)",
          py::call_guard<py::gil_scoped_release>());
}
