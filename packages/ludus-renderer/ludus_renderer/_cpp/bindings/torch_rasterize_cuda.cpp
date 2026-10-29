// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include "torch_common.inl"
#include "torch_types.h"
#include "../common/common.h"
#include "../render/ludus_cuda.h"
#include <tuple>

//------------------------------------------------------------------------
// FLU to RDF conversion (same as GL path).

static torch::Tensor flu_to_rdf_cuda(const torch::Tensor& camera_poses)
{
    static const float kFluToRdf[16] = {
        0, -1,  0, 0,
        0,  0, -1, 0,
        1,  0,  0, 0,
        0,  0,  0, 1,
    };
    auto conv = torch::from_blob((void*)kFluToRdf, {4, 4}, torch::kFloat32)
                    .to(camera_poses.device());
    return torch::matmul(conv, camera_poses);
}

//------------------------------------------------------------------------
// CUDA renderer state wrapper.

LudusCudaStateWrapper::LudusCudaStateWrapper(int cudaDeviceIdx_)
{
    pState = new LudusCudaState();
    cudaDeviceIdx = cudaDeviceIdx_;
    ludusCudaInit(NVDR_CTX_PARAMS, *pState);
}

LudusCudaStateWrapper::~LudusCudaStateWrapper(void)
{
    ludusCudaDestroy(NVDR_CTX_PARAMS, *pState);
    delete pState;
}

void LudusCudaStateWrapper::setLineWidths(float polyline_regular, float polyline_bev,
                                           float ego_traj_regular, float ego_traj_bev,
                                           float wireframe)
{
    pState->widthPolylineRegular = polyline_regular;
    pState->widthPolylineBev = polyline_bev;
    pState->widthEgoTrajRegular = ego_traj_regular;
    pState->widthEgoTrajBev = ego_traj_bev;
    pState->widthWireframe = wireframe;
}

void LudusCudaStateWrapper::setResolutionScale(float scale)
{
    pState->resolutionScale = scale;
}

void LudusCudaStateWrapper::setDepthScaling(float enabled)
{
    pState->depthScaling = enabled;
}

void LudusCudaStateWrapper::setCullRadius(float radius)
{
    pState->cullRadiusScale = radius;
}

void LudusCudaStateWrapper::setMaxTessellationLevels(int polyline, int polygon, int cube)
{
    pState->maxTessPolyline = polyline;
    pState->maxTessPolygon = polygon;
    pState->maxTessCube = cube;
}

void LudusCudaStateWrapper::setMsaaSamples(int samples)
{
    NVDR_CHECK(samples == 0 || samples == 4,
               "CUDA renderer MSAA samples must be 0 (disabled) or 4 (4x SSAA)");
    pState->msaaSamples = samples;
}

void LudusCudaStateWrapper::uploadColorPalette(torch::Tensor colors)
{
    NVDR_CHECK(colors.dim() == 1, "color palette must be 1D tensor of packed RGBA8 uint32");
    int count = colors.size(0);
    std::vector<uint32_t> hostPalette(count);
    auto colors_cpu = colors.to(torch::kCPU).to(torch::kInt32);
    memcpy(hostPalette.data(), colors_cpu.data_ptr<int32_t>(), count * sizeof(uint32_t));
    ludusCudaUploadColorPalette(*pState, hostPalette.data(), count);
}

//------------------------------------------------------------------------
// Forward op: CUDA-only f-theta rendering.

torch::Tensor ludus_render_fwd_cuda(
    LudusCudaStateWrapper& stateWrapper,
    torch::Tensor polyline_headers,
    torch::Tensor polygon_headers,
    torch::Tensor cubes,
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor camera_intrinsics,
    torch::Tensor camera_poses,
    std::tuple<int, int> resolution,
    float tessellation_threshold)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(camera_poses));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusCudaState& s = *stateWrapper.pState;

    NVDR_CHECK_DEVICE(polyline_headers, polygon_headers, cubes, vertices, triangles, camera_intrinsics, camera_poses);
    NVDR_CHECK_CONTIGUOUS(polyline_headers, polygon_headers, cubes, vertices, triangles, camera_intrinsics, camera_poses);
    NVDR_CHECK_F32(polyline_headers, polygon_headers, cubes, vertices, camera_intrinsics, camera_poses);
    NVDR_CHECK_I32(triangles);

    NVDR_CHECK(camera_poses.get_device() == stateWrapper.cudaDeviceIdx,
               "CUDA context must reside on the same device as input tensors");

    int numPolylines = polyline_headers.size(0);
    int numPolygons = polygon_headers.size(0);
    int numCubes = cubes.size(0);
    int numVertices = vertices.size(0);
    int numTriangles = triangles.size(0);
    int numCameras = camera_intrinsics.size(0);

    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    s.tessellationThreshold = tessellation_threshold;

    const PolylineHeader* polylinePtr = numPolylines > 0 ?
        reinterpret_cast<const PolylineHeader*>(polyline_headers.data_ptr<float>()) : nullptr;
    const PolygonHeader* polygonPtr = numPolygons > 0 ?
        reinterpret_cast<const PolygonHeader*>(polygon_headers.data_ptr<float>()) : nullptr;
    const Cube* cubePtr = numCubes > 0 ?
        reinterpret_cast<const Cube*>(cubes.data_ptr<float>()) : nullptr;
    const Vertex* vertexPtr = numVertices > 0 ?
        reinterpret_cast<const Vertex*>(vertices.data_ptr<float>()) : nullptr;
    const Triangle* trianglePtr = numTriangles > 0 ?
        reinterpret_cast<const Triangle*>(triangles.data_ptr<int32_t>()) : nullptr;

    // FLU → RDF, then transpose for column-major
    torch::Tensor camera_poses_t = flu_to_rdf_cuda(camera_poses).transpose(-2, -1).contiguous();
    const CameraPose* posePtr = reinterpret_cast<const CameraPose*>(camera_poses_t.data_ptr<float>());
    const FThetaCamera* intrinsicsPtr = reinterpret_cast<const FThetaCamera*>(camera_intrinsics.data_ptr<float>());

    // Allocate output
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor out_rgba = torch::empty({numCameras, height, width, 4}, opts);

    // Render (no cube pools or dots in this entry point)
    ludusCudaRender(NVDR_CTX_PARAMS, s, stream,
                    polylinePtr, numPolylines,
                    polygonPtr, numPolygons,
                    cubePtr, numCubes,
                    vertexPtr, numVertices,
                    trianglePtr, numTriangles,
                    intrinsicsPtr, posePtr, numCameras,
                    width, height,
                    out_rgba.data_ptr<uint8_t>(),
                    nullptr, 0,
                    nullptr, 0);

    return out_rgba;
}

//------------------------------------------------------------------------
// Forward op: CUDA-only f-theta rendering with timestamped cube pools.
// cube_pool_list: list of tuples (track_ts, prefix_sum, translations, quaternions,
//                                  scales, colors, query_ts_us, max_extrap_us, render_flags)

torch::Tensor ludus_render_fwd_cuda_ts(
    LudusCudaStateWrapper& stateWrapper,
    torch::Tensor polyline_headers,
    torch::Tensor polygon_headers,
    torch::Tensor cubes,
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor camera_intrinsics,
    torch::Tensor camera_poses,
    std::tuple<int, int> resolution,
    float tessellation_threshold,
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor, int64_t, int, int>> cube_pool_list,
    std::vector<std::tuple<torch::Tensor, float, int64_t>> dot_list)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(camera_poses));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusCudaState& s = *stateWrapper.pState;

    NVDR_CHECK_DEVICE(polyline_headers, polygon_headers, cubes, vertices, triangles, camera_intrinsics, camera_poses);
    NVDR_CHECK_CONTIGUOUS(polyline_headers, polygon_headers, cubes, vertices, triangles, camera_intrinsics, camera_poses);
    NVDR_CHECK_F32(polyline_headers, polygon_headers, cubes, vertices, camera_intrinsics, camera_poses);
    NVDR_CHECK_I32(triangles);

    NVDR_CHECK(camera_poses.get_device() == stateWrapper.cudaDeviceIdx,
               "CUDA context must reside on the same device as input tensors");

    int numPolylines = polyline_headers.size(0);
    int numPolygons = polygon_headers.size(0);
    int numCubes = cubes.size(0);
    int numVertices = vertices.size(0);
    int numTriangles = triangles.size(0);
    int numCameras = camera_intrinsics.size(0);

    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    s.tessellationThreshold = tessellation_threshold;

    const PolylineHeader* polylinePtr = numPolylines > 0 ?
        reinterpret_cast<const PolylineHeader*>(polyline_headers.data_ptr<float>()) : nullptr;
    const PolygonHeader* polygonPtr = numPolygons > 0 ?
        reinterpret_cast<const PolygonHeader*>(polygon_headers.data_ptr<float>()) : nullptr;
    const Cube* cubePtr = numCubes > 0 ?
        reinterpret_cast<const Cube*>(cubes.data_ptr<float>()) : nullptr;
    const Vertex* vertexPtr = numVertices > 0 ?
        reinterpret_cast<const Vertex*>(vertices.data_ptr<float>()) : nullptr;
    const Triangle* trianglePtr = numTriangles > 0 ?
        reinterpret_cast<const Triangle*>(triangles.data_ptr<int32_t>()) : nullptr;

    torch::Tensor camera_poses_t = flu_to_rdf_cuda(camera_poses).transpose(-2, -1).contiguous();
    const CameraPose* posePtr = reinterpret_cast<const CameraPose*>(camera_poses_t.data_ptr<float>());
    const FThetaCamera* intrinsicsPtr = reinterpret_cast<const FThetaCamera*>(camera_intrinsics.data_ptr<float>());

    // Build CubePoolParams array
    int numPools = (int)cube_pool_list.size();
    std::vector<CubePoolParams> pools(numPools);
    for (int i = 0; i < numPools; i++) {
        auto& [tTs, pSum, trans, quat, sc, col, qTs, maxE, rFlags] = cube_pool_list[i];
        pools[i].trackTimestamps   = tTs.data_ptr<int64_t>();
        pools[i].prefixSum         = pSum.data_ptr<int32_t>();
        pools[i].translations      = trans.data_ptr<float>();
        pools[i].quaternions       = quat.data_ptr<float>();
        pools[i].scales            = sc.data_ptr<float>();
        pools[i].colors            = col.data_ptr<float>();
        pools[i].numCubes          = (int)pSum.size(0);
        pools[i].queryTimestampUs  = qTs;
        pools[i].maxExtrapolationUs = maxE;
        pools[i].renderFlags       = (uint32_t)rFlags;
    }

    // Build DotParams array
    int numDotGroups = (int)dot_list.size();
    std::vector<DotParams> dotGroups(numDotGroups);
    for (int i = 0; i < numDotGroups; i++) {
        auto& [positions, radius, colorVal] = dot_list[i];
        dotGroups[i].positions = reinterpret_cast<const Vertex*>(positions.data_ptr<float>());
        dotGroups[i].numDots   = (int)positions.size(0);
        dotGroups[i].radius    = radius;
        dotGroups[i].color     = (uint32_t)(colorVal & 0xFFFFFFFF);
    }

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor out_rgba = torch::empty({numCameras, height, width, 4}, opts);

    ludusCudaRender(NVDR_CTX_PARAMS, s, stream,
                    polylinePtr, numPolylines,
                    polygonPtr, numPolygons,
                    cubePtr, numCubes,
                    vertexPtr, numVertices,
                    trianglePtr, numTriangles,
                    intrinsicsPtr, posePtr, numCameras,
                    width, height,
                    out_rgba.data_ptr<uint8_t>(),
                    pools.data(), numPools,
                    dotGroups.data(), numDotGroups);

    return out_rgba;
}

//------------------------------------------------------------------------
// Timestamped rendering: flat buffer layout, all processing on GPU.

torch::Tensor ludus_render_fwd_cuda_timestamped(
    LudusCudaStateWrapper& stateWrapper,
    torch::Tensor timestamps,        // [N_ts] int64
    torch::Tensor int32_data,         // [N_i32] int32
    torch::Tensor vertices,           // [N_v, 4] float32
    torch::Tensor triangles,          // [N_t, 4] int32
    torch::Tensor float_data,         // [N_f] float32
    torch::Tensor polyline_pools,     // [num_pl_pools, 16] uint32
    torch::Tensor polygon_pools,      // [num_pg_pools, 16] uint32
    torch::Tensor cube_pools,         // [num_cb_pools, 16] uint32
    int64_t query_timestamp_us,
    int max_extrapolation_us,
    int max_varrays_per_ts_polyline,
    int max_varrays_per_ts_polygon,
    int camera_type_id,
    torch::Tensor camera_intrinsics,  // [C, 18] float32
    torch::Tensor camera_poses,       // [C, 4, 4] float32
    std::tuple<int, int> resolution,
    float tessellation_threshold)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(camera_poses));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusCudaState& s = *stateWrapper.pState;

    NVDR_CHECK(camera_poses.get_device() == stateWrapper.cudaDeviceIdx,
               "CUDA context must reside on the same device as input tensors");

    int numCameras = camera_intrinsics.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    s.tessellationThreshold = tessellation_threshold;

    // Build CudaRenderParams from state + per-call params
    CudaRenderParams params;
    params.widthPolylineRegular = s.widthPolylineRegular;
    params.widthPolylineBev = s.widthPolylineBev;
    params.widthEgoTrajRegular = s.widthEgoTrajRegular;
    params.widthEgoTrajBev = s.widthEgoTrajBev;
    params.widthWireframe = s.widthWireframe;
    params.resolutionScale = s.resolutionScale;
    params.depthScaling = s.depthScaling;
    params.cullRadiusScale = s.cullRadiusScale;
    params.tessellationThreshold = tessellation_threshold;
    params.maxTessPolyline = s.maxTessPolyline;
    params.maxTessPolygon = s.maxTessPolygon;
    params.maxTessCube = s.maxTessCube;
    params.cameraTypeId = camera_type_id;
    params.colorPaletteSize = s.colorPaletteSize;
    params.colorPalette = s.colorPalette;

    // FLU → RDF, then transpose for column-major
    torch::Tensor camera_poses_t = flu_to_rdf_cuda(camera_poses).transpose(-2, -1).contiguous();
    const CameraPose* posePtr = reinterpret_cast<const CameraPose*>(camera_poses_t.data_ptr<float>());
    const FThetaCamera* intrinsicsPtr = reinterpret_cast<const FThetaCamera*>(camera_intrinsics.data_ptr<float>());

    int numPlPools = polyline_pools.size(0);
    int numPgPools = polygon_pools.size(0);
    int numCbPools = cube_pools.size(0);

    const TsPolylinePoolHeader* plPoolPtr = numPlPools > 0 ?
        reinterpret_cast<const TsPolylinePoolHeader*>(polyline_pools.data_ptr<int32_t>()) : nullptr;
    const TsPolygonPoolHeader* pgPoolPtr = numPgPools > 0 ?
        reinterpret_cast<const TsPolygonPoolHeader*>(polygon_pools.data_ptr<int32_t>()) : nullptr;
    const TsCubePoolHeader* cbPoolPtr = numCbPools > 0 ?
        reinterpret_cast<const TsCubePoolHeader*>(cube_pools.data_ptr<int32_t>()) : nullptr;

    const int64_t* tsPtr = timestamps.numel() > 0 ? timestamps.data_ptr<int64_t>() : nullptr;
    const int32_t* i32Ptr = int32_data.numel() > 0 ? int32_data.data_ptr<int32_t>() : nullptr;
    const Vertex* vtxPtr = vertices.size(0) > 0 ?
        reinterpret_cast<const Vertex*>(vertices.data_ptr<float>()) : nullptr;
    const Triangle* triPtr = triangles.size(0) > 0 ?
        reinterpret_cast<const Triangle*>(triangles.data_ptr<int32_t>()) : nullptr;
    const float* fltPtr = float_data.numel() > 0 ? float_data.data_ptr<float>() : nullptr;

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor out_rgba = torch::empty({numCameras, height, width, 4}, opts);

    ludusCudaRenderTimestamped(NVDR_CTX_PARAMS, s, stream,
                               tsPtr, i32Ptr, vtxPtr, triPtr, fltPtr,
                               plPoolPtr, numPlPools,
                               pgPoolPtr, numPgPools,
                               cbPoolPtr, numCbPools,
                               query_timestamp_us, max_extrapolation_us,
                               max_varrays_per_ts_polyline,
                               max_varrays_per_ts_polygon,
                               params,
                               intrinsicsPtr, posePtr,
                               numCameras, width, height,
                               out_rgba.data_ptr<uint8_t>());

    return out_rgba;
}

//------------------------------------------------------------------------
