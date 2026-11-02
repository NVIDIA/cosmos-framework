// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include "torch_common.inl"
#include "torch_types.h"
#include <tuple>
#include <string>
#include <sstream>
#include "../render/ludus_types.h"

//------------------------------------------------------------------------
// Op prototypes.

torch::Tensor ludus_render_fwd_gl(LudusGLStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);
torch::Tensor ludus_render_fwd_cuda(LudusCudaStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);
torch::Tensor ludus_render_fwd_cuda_ts(LudusCudaStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold, std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int, int>> cube_pool_list, std::vector<std::tuple<torch::Tensor, float, int64_t>> dot_list);
torch::Tensor ludus_render_fwd_cuda_timestamped(LudusCudaStateWrapper& stateWrapper, torch::Tensor timestamps, torch::Tensor int32_data, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor float_data, torch::Tensor polyline_pools, torch::Tensor polygon_pools, torch::Tensor cube_pools, int64_t query_timestamp_us, int max_extrapolation_us, int max_varrays_per_ts_polyline, int max_varrays_per_ts_polygon, int camera_type_id, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);
torch::Tensor ludus_timestamped_render_batch(LudusTimestampedStateWrapper& stateWrapper, torch::Tensor queries, torch::Tensor camera_poses, std::tuple<int, int> resolution);
std::tuple<int, bool> ludusTimestampedRenderBatchToStaging(LudusTimestampedStateWrapper& stateWrapper, torch::Tensor queries, torch::Tensor camera_poses, std::tuple<int, int> resolution);
torch::Tensor ludusTimestampedGetStagingData(LudusTimestampedStateWrapper& stateWrapper, int stagingIdx);
torch::Tensor ludusTimestampedGetStagingDataAsync(LudusTimestampedStateWrapper& stateWrapper, int stagingIdx, int64_t streamPtr);
int ludusTimestampedStartAsyncHostTransfer(LudusTimestampedStateWrapper& stateWrapper, int stagingIdx);
bool ludusTimestampedIsPinnedBufferReady(LudusTimestampedStateWrapper& stateWrapper, int pinnedIdx);
bool ludusTimestampedIsHostTransferComplete(LudusTimestampedStateWrapper& stateWrapper);
torch::Tensor ludusTimestampedWaitPinnedBufferView(LudusTimestampedStateWrapper& stateWrapper, int pinnedIdx);
torch::Tensor ludusTimestampedWaitHostTransfer(LudusTimestampedStateWrapper& stateWrapper);
torch::Tensor ludusTimestampedWaitHostTransferView(LudusTimestampedStateWrapper& stateWrapper);

// NVJPEG hardware encoding
bool ludusTimestampedIsNvjpegAvailable(LudusTimestampedStateWrapper& stateWrapper);
py::bytes ludusTimestampedEncodeJpegStaging(LudusTimestampedStateWrapper& stateWrapper, int stagingIdx, int imageIdx, int quality);
py::list ludusTimestampedEncodeJpegBatchStaging(LudusTimestampedStateWrapper& stateWrapper, int stagingIdx, int quality);
py::bytes ludusTimestampedEncodeJpegPinned(LudusTimestampedStateWrapper& stateWrapper, int pinnedIdx, int imageIdx, int quality);

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Ludus GL Context (basic rendering)
    pybind11::class_<LudusGLStateWrapper>(m, "LudusGLStateWrapper").def(pybind11::init<int>())
        .def("set_context",     &LudusGLStateWrapper::setContext)
        .def("release_context", &LudusGLStateWrapper::releaseContext)
        .def("set_msaa_samples", &LudusGLStateWrapper::setMsaaSamples);

    // Ludus Timestamped Context (multi-scene temporal rendering)
    pybind11::class_<LudusTimestampedStateWrapper>(m, "LudusTimestampedStateWrapper").def(pybind11::init<int>())
        .def("set_context",              &LudusTimestampedStateWrapper::setContext)
        .def("release_context",          &LudusTimestampedStateWrapper::releaseContext)
        .def("upload_scene",             &LudusTimestampedStateWrapper::uploadScene)
        .def("upload_cameras",           &LudusTimestampedStateWrapper::uploadCameras)
        .def("upload_color_palette",     &LudusTimestampedStateWrapper::uploadColorPalette)
        .def("remove_scene",             &LudusTimestampedStateWrapper::removeScene)
        .def("clear_scenes",             &LudusTimestampedStateWrapper::clearScenes)
        .def("preallocate_buffers",      &LudusTimestampedStateWrapper::preallocateBuffers)
        .def("upload_scenes_batch",      &LudusTimestampedStateWrapper::uploadScenesBatch)
        .def("set_tessellation_threshold", &LudusTimestampedStateWrapper::setTessellationThreshold)
        .def("set_max_tessellation_levels", &LudusTimestampedStateWrapper::setMaxTessellationLevels)
        .def("set_line_widths",          &LudusTimestampedStateWrapper::setLineWidths)
        .def("set_resolution_scale",     &LudusTimestampedStateWrapper::setResolutionScale)
        .def("set_depth_scaling",        &LudusTimestampedStateWrapper::setDepthScaling)
        .def("set_cull_radius",          &LudusTimestampedStateWrapper::setCullRadius)
        .def("set_msaa_samples",         &LudusTimestampedStateWrapper::setMsaaSamples)
        .def("get_max_batch_size",       &LudusTimestampedStateWrapper::getMaxBatchSize)
        .def("swap_buffer_sets",         &LudusTimestampedStateWrapper::swapBufferSets);

    // Struct layout info for cache versioning
    m.def("get_struct_sizes", []() -> py::dict {
        py::dict d;
        d["TimestampedScene"] = (int)sizeof(TimestampedScene);
        d["TimestampedPolylinePool"] = (int)sizeof(TimestampedPolylinePool);
        d["TimestampedPolygonPool"] = (int)sizeof(TimestampedPolygonPool);
        d["ObstaclePool"] = (int)sizeof(ObstaclePool);
        d["Vertex"] = (int)sizeof(Vertex);
        d["Triangle"] = (int)sizeof(Triangle);
        d["CameraPose"] = (int)sizeof(CameraPose);
        return d;
    }, "get C++ struct sizes for cache format hashing");

    // Ludus CUDA Context (fully CUDA-based, no OpenGL)
    pybind11::class_<LudusCudaStateWrapper>(m, "LudusCudaStateWrapper").def(pybind11::init<int>())
        .def("set_line_widths",          &LudusCudaStateWrapper::setLineWidths)
        .def("set_resolution_scale",     &LudusCudaStateWrapper::setResolutionScale)
        .def("set_depth_scaling",        &LudusCudaStateWrapper::setDepthScaling)
        .def("set_cull_radius",          &LudusCudaStateWrapper::setCullRadius)
        .def("set_max_tessellation_levels", &LudusCudaStateWrapper::setMaxTessellationLevels)
        .def("upload_color_palette",     &LudusCudaStateWrapper::uploadColorPalette)
        .def("set_msaa_samples",         &LudusCudaStateWrapper::setMsaaSamples);

    // Ludus rendering ops
    m.def("ludus_render_fwd_gl", &ludus_render_fwd_gl, "ludus f-theta mesh shader rendering (opengl)");
    m.def("ludus_render_fwd_cuda", &ludus_render_fwd_cuda, "ludus f-theta CUDA-only rendering (no opengl)");
    m.def("ludus_render_fwd_cuda_ts", &ludus_render_fwd_cuda_ts, "ludus f-theta CUDA-only rendering with timestamped cube pools");
    m.def("ludus_render_fwd_cuda_timestamped", &ludus_render_fwd_cuda_timestamped, "ludus f-theta CUDA-only timestamped rendering from flat buffers");
    m.def("ludus_timestamped_render_batch", &ludus_timestamped_render_batch, "ludus timestamped batch rendering (opengl)");
    m.def("ludus_timestamped_render_to_staging", &ludusTimestampedRenderBatchToStaging, "ludus timestamped render to staging buffer (double buffer)");
    m.def("ludus_timestamped_get_staging_data", &ludusTimestampedGetStagingData, "get data from staging buffer");
    m.def("ludus_timestamped_get_staging_data_async", &ludusTimestampedGetStagingDataAsync, "get staging buffer view with GPU-side stream sync (non-blocking)");
    m.def("ludus_timestamped_start_async_host_transfer", &ludusTimestampedStartAsyncHostTransfer, "start async D2H transfer to pinned memory, returns pinned idx");
    m.def("ludus_timestamped_is_pinned_buffer_ready", &ludusTimestampedIsPinnedBufferReady, "check if specific pinned buffer is ready");
    m.def("ludus_timestamped_is_host_transfer_complete", &ludusTimestampedIsHostTransferComplete, "check if any async D2H is complete (legacy)");
    m.def("ludus_timestamped_wait_pinned_buffer_view", &ludusTimestampedWaitPinnedBufferView, "wait for specific pinned buffer and get zero-copy view");
    m.def("ludus_timestamped_wait_host_transfer", &ludusTimestampedWaitHostTransfer, "wait for async D2H and get CPU tensor (legacy)");
    m.def("ludus_timestamped_wait_host_transfer_view", &ludusTimestampedWaitHostTransferView, "wait for async D2H and get zero-copy view (legacy)");

    // NVJPEG hardware encoding
    m.def("ludus_timestamped_is_nvjpeg_available", &ludusTimestampedIsNvjpegAvailable, "check if NVJPEG hardware encoder is available");
    m.def("ludus_timestamped_encode_jpeg_staging", &ludusTimestampedEncodeJpegStaging, "encode single image from staging buffer to JPEG");
    m.def("ludus_timestamped_encode_jpeg_batch_staging", &ludusTimestampedEncodeJpegBatchStaging, "encode all images from staging buffer to JPEG");
    m.def("ludus_timestamped_encode_jpeg_pinned", &ludusTimestampedEncodeJpegPinned, "encode single image from pinned buffer to JPEG");
}

//------------------------------------------------------------------------
