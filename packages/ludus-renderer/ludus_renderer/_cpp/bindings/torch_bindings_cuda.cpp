// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include "torch_common.inl"
#include "torch_types.h"
#include <tuple>

//------------------------------------------------------------------------
// Op prototypes (CUDA-only).

torch::Tensor ludus_render_fwd_cuda(LudusCudaStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);
torch::Tensor ludus_render_fwd_cuda_ts(LudusCudaStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold, std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int, int>> cube_pool_list, std::vector<std::tuple<torch::Tensor, float, int64_t>> dot_list);
torch::Tensor ludus_render_fwd_cuda_timestamped(LudusCudaStateWrapper& stateWrapper, torch::Tensor timestamps, torch::Tensor int32_data, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor float_data, torch::Tensor polyline_pools, torch::Tensor polygon_pools, torch::Tensor cube_pools, int64_t query_timestamp_us, int max_extrapolation_us, int max_varrays_per_ts_polyline, int max_varrays_per_ts_polygon, int camera_type_id, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Ludus CUDA Context (fully CUDA-based, no OpenGL)
    pybind11::class_<LudusCudaStateWrapper>(m, "LudusCudaStateWrapper").def(pybind11::init<int>())
        .def("set_line_widths",          &LudusCudaStateWrapper::setLineWidths)
        .def("set_resolution_scale",     &LudusCudaStateWrapper::setResolutionScale)
        .def("set_depth_scaling",        &LudusCudaStateWrapper::setDepthScaling)
        .def("set_cull_radius",          &LudusCudaStateWrapper::setCullRadius)
        .def("set_max_tessellation_levels", &LudusCudaStateWrapper::setMaxTessellationLevels)
        .def("upload_color_palette",     &LudusCudaStateWrapper::uploadColorPalette)
        .def("set_msaa_samples",         &LudusCudaStateWrapper::setMsaaSamples);

    // CUDA rendering ops
    m.def("ludus_render_fwd_cuda", &ludus_render_fwd_cuda, "ludus f-theta CUDA-only rendering (no opengl)");
    m.def("ludus_render_fwd_cuda_ts", &ludus_render_fwd_cuda_ts, "ludus f-theta CUDA-only rendering with timestamped cube pools");
    m.def("ludus_render_fwd_cuda_timestamped", &ludus_render_fwd_cuda_timestamped, "ludus f-theta CUDA-only timestamped rendering from flat buffers");
}

//------------------------------------------------------------------------
