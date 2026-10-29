// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include "torch_common.inl"
#include "torch_types.h"
#include "../common/common.h"
#include "../render/ludus_gl.h"
#include <tuple>

//------------------------------------------------------------------------
// FLU (Forward-Left-Up) to RDF (Right-Down-Forward) conversion.
// All external APIs accept FLU poses; the shader's F-theta projection
// operates in RDF, so we convert at the C++ boundary.
//
// TODO: eliminate this by updating the GLSL shaders to project in FLU
// directly (swap depth axis from Z to X, adjust lateral/phi mapping).
// Requires updating every projection site: polyline, polygon, cube,
// tessellation shaders, depth testing, and backface/front_color logic.

static torch::Tensor flu_to_rdf(const torch::Tensor& camera_poses)
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
// Ludus GL state wrapper methods.

LudusGLStateWrapper::LudusGLStateWrapper(int cudaDeviceIdx_)
{
    pState = new LudusGLState();
    automatic = true;
    cudaDeviceIdx = cudaDeviceIdx_;
    memset(pState, 0, sizeof(LudusGLState));
    ludusInitGLContext(NVDR_CTX_PARAMS, *pState, cudaDeviceIdx_);
    releaseGLContext();
}

LudusGLStateWrapper::~LudusGLStateWrapper(void)
{
    setGLContext(pState->glctx);
    ludusReleaseBuffers(NVDR_CTX_PARAMS, *pState);
    releaseGLContext();
    destroyGLContext(pState->glctx);
    delete pState;
}

void LudusGLStateWrapper::setContext(void)
{
    setGLContext(pState->glctx);
}

void LudusGLStateWrapper::releaseContext(void)
{
    releaseGLContext();
}

void LudusGLStateWrapper::setMsaaSamples(int samples)
{
    // Validate sample count (must be 0, 1, 2, 4, or 8)
    if (samples != 0 && samples != 1 && samples != 2 && samples != 4 && samples != 8)
    {
        NVDR_CHECK(false, "MSAA samples must be 0, 1, 2, 4, or 8");
    }
    pState->msaaSamples = samples;
    // Force framebuffer reallocation on next render
    if (samples >= 2)
    {
        pState->width = 0;
        pState->height = 0;
    }
}

//------------------------------------------------------------------------
// Forward op: Ludus f-theta rendering.

torch::Tensor ludus_render_fwd_gl(
    LudusGLStateWrapper& stateWrapper,
    torch::Tensor polyline_headers,    // [N_pl, 8] float32: vertex_start, vertex_count, cap_style, pad, r, g, b, width
    torch::Tensor polygon_headers,     // [N_pg, 8] float32: vertex_start, vertex_count, tri_start, tri_count, r, g, b, pad
    torch::Tensor cubes,           // [N_obs, 15] float32: translation(3), scale(3), rotation(3), front_color(3), back_color(3)
    torch::Tensor vertices,            // [N_verts, 4] float32: x, y, z, pad
    torch::Tensor triangles,           // [N_tris, 3] int32: i0, i1, i2
    torch::Tensor camera_intrinsics,   // [P, 18] float32: packed FThetaCamera (72 bytes)
    torch::Tensor camera_poses,        // [P, 4, 4] float32: world-to-camera in FLU convention
    std::tuple<int, int> resolution,
    float tessellation_threshold)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(camera_poses));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusGLState& s = *stateWrapper.pState;

    // Check inputs
    NVDR_CHECK_DEVICE(polyline_headers, polygon_headers, cubes, vertices, triangles, camera_intrinsics, camera_poses);
    NVDR_CHECK_CONTIGUOUS(polyline_headers, polygon_headers, cubes, vertices, triangles, camera_intrinsics, camera_poses);
    NVDR_CHECK_F32(polyline_headers, polygon_headers, cubes, vertices, camera_intrinsics, camera_poses);
    NVDR_CHECK_I32(triangles);

    // Check that GL context was created for the correct GPU
    NVDR_CHECK(camera_poses.get_device() == stateWrapper.cudaDeviceIdx,
               "GL context must reside on the same device as input tensors");

    // Validate shapes
    int numPolylines = polyline_headers.size(0);
    int numPolygons = polygon_headers.size(0);
    int numCubes = cubes.size(0);
    int numVertices = vertices.size(0);
    int numTriangles = triangles.size(0);
    int numCameras = camera_intrinsics.size(0);

    NVDR_CHECK(polyline_headers.sizes().size() == 2 && (numPolylines == 0 || polyline_headers.size(1) == 8),
               "polyline_headers must have shape [N, 8]");
    NVDR_CHECK(polygon_headers.sizes().size() == 2 && (numPolygons == 0 || polygon_headers.size(1) == 8),
               "polygon_headers must have shape [N, 8]");
    NVDR_CHECK(cubes.sizes().size() == 2 && (numCubes == 0 || cubes.size(1) == 16),
               "cubes must have shape [N, 16]");
    NVDR_CHECK(vertices.sizes().size() == 2 && (numVertices == 0 || vertices.size(1) == 4),
               "vertices must have shape [N, 4]");
    NVDR_CHECK(triangles.sizes().size() == 2 && (numTriangles == 0 || triangles.size(1) == 4),
               "triangles must have shape [N, 4]");
    NVDR_CHECK(camera_intrinsics.sizes().size() == 2 && camera_intrinsics.size(1) == 18,
               "camera_intrinsics must have shape [P, 18]");
    NVDR_CHECK(camera_poses.sizes().size() == 3 && camera_poses.size(0) == numCameras &&
               camera_poses.size(1) == 4 && camera_poses.size(2) == 4,
               "camera_poses must have shape [P, 4, 4]");

    // Get output dimensions
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    // Set GL context
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    // Resize buffers as needed
    bool changes = false;
    ludusResizeBuffers(NVDR_CTX_PARAMS, s, changes,
                       numPolylines, numPolygons, numCubes, numVertices, numTriangles,
                       numCameras, width, height);

    // Set tessellation threshold (stored in state for shader uniforms)
    s.tessellationThreshold = tessellation_threshold;

    // Reinterpret tensors as struct arrays
    // Note: We're reinterpreting packed float tensors as our structs
    // The Python side packs data to match the C++ struct layout
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

    // FLU → RDF, then transpose for GLSL column-major layout
    torch::Tensor camera_poses_t = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();
    const CameraPose* posePtr = reinterpret_cast<const CameraPose*>(camera_poses_t.data_ptr<float>());
    const FThetaCamera* intrinsicsPtr = reinterpret_cast<const FThetaCamera*>(camera_intrinsics.data_ptr<float>());

    // Render
    ludusRender(NVDR_CTX_PARAMS, s, stream,
                polylinePtr, numPolylines,
                polygonPtr, numPolygons,
                cubePtr, numCubes,
                vertexPtr, numVertices,
                trianglePtr, numTriangles,
                intrinsicsPtr, posePtr, numCameras,
                width, height);

    // Allocate output tensor (RGBA8 = 4 bytes per pixel)
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor out_rgba = torch::empty({numCameras, height, width, 4}, opts);

    // Copy results
    ludusCopyResults(NVDR_CTX_PARAMS, s, stream, out_rgba.data_ptr<uint8_t>(), width, height, numCameras);

    // Release GL context
    if (stateWrapper.automatic)
        releaseGLContext();

    return out_rgba;
}

//------------------------------------------------------------------------
// Ludus Timestamped state wrapper methods.

LudusTimestampedStateWrapper::LudusTimestampedStateWrapper(int cudaDeviceIdx_)
{
    pState = new LudusTimestampedState();
    automatic = true;
    cudaDeviceIdx = cudaDeviceIdx_;
    memset(pState, 0, sizeof(LudusTimestampedState));
    pState->cullRadiusScale = 1.5f;
    ludusTimestampedInit(NVDR_CTX_PARAMS, *pState, cudaDeviceIdx_);
    releaseGLContext();
}

LudusTimestampedStateWrapper::~LudusTimestampedStateWrapper(void)
{
    GLContext glctx = pState->glctx;  // Save before release clears it
    if (glctx.context) {
        setGLContext(glctx);
        ludusTimestampedRelease(NVDR_CTX_PARAMS, *pState);
        releaseGLContext();
        destroyGLContext(glctx);
    }
    delete pState;
}

void LudusTimestampedStateWrapper::setContext(void)
{
    setGLContext(pState->glctx);
}

void LudusTimestampedStateWrapper::releaseContext(void)
{
    releaseGLContext();
}

void LudusTimestampedStateWrapper::setTessellationThreshold(float threshold)
{
    pState->tessellationThreshold = threshold;
}

void LudusTimestampedStateWrapper::setMaxTessellationLevels(int polyline, int polygon, int cube)
{
    NVDR_CHECK(polyline >= 0 && polyline <= 4, "max_tessellation_level_polyline must be 0..4");
    NVDR_CHECK(polygon >= 0 && polygon <= 3, "max_tessellation_level_polygon must be 0..3");
    NVDR_CHECK(cube >= 0 && cube <= 3, "max_tessellation_level_cube must be 0..3");
    pState->maxTessellationLevelPolyline = polyline;
    pState->maxTessellationLevelPolygon = polygon;
    pState->maxTessellationLevelCube = cube;
}

void LudusTimestampedStateWrapper::setLineWidths(float polyline_regular, float polyline_bev,
                                                  float ego_traj_regular, float ego_traj_bev,
                                                  float wireframe)
{
    pState->widthPolylineRegular = polyline_regular;
    pState->widthPolylineBev = polyline_bev;
    pState->widthEgoTrajRegular = ego_traj_regular;
    pState->widthEgoTrajBev = ego_traj_bev;
    pState->widthWireframe = wireframe;
}

void LudusTimestampedStateWrapper::setResolutionScale(float scale)
{
    pState->resolutionScale = scale;
}

void LudusTimestampedStateWrapper::setDepthScaling(float enabled)
{
    pState->depthScaling = enabled;
}

void LudusTimestampedStateWrapper::setCullRadius(float scale)
{
    pState->cullRadiusScale = scale;
}

void LudusTimestampedStateWrapper::setMsaaSamples(int samples)
{
    // Validate sample count (must be 0, 1, 2, 4, or 8)
    if (samples != 0 && samples != 1 && samples != 2 && samples != 4 && samples != 8)
    {
        NVDR_CHECK(false, "MSAA samples must be 0, 1, 2, 4, or 8");
    }
    pState->msaaSamples = samples;
    // Force framebuffer reallocation on next render
    if (samples >= 2)
    {
        pState->width = 0;
        pState->height = 0;
    }
}

int LudusTimestampedStateWrapper::getMaxBatchSize(void)
{
    setGLContext(pState->glctx);
    GLint maxLayers = 0;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxLayers);
    return (int)maxLayers;
}

void LudusTimestampedStateWrapper::swapBufferSets(void)
{
    if (automatic)
        setGLContext(pState->glctx);

    ludusSwapBufferSets(NVDR_CTX_PARAMS, *pState);

    if (automatic)
        releaseGLContext();
}

void LudusTimestampedStateWrapper::uploadColorPalette(torch::Tensor colors)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(colors));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedState& s = *pState;

    NVDR_CHECK(colors.get_device() == cudaDeviceIdx,
               "colors must be on same device as GL context");
    NVDR_CHECK(colors.dim() == 2 && colors.size(1) == 4,
               "colors must be [num_prim_types, 4] (RGBA)");
    NVDR_CHECK(colors.dtype() == torch::kFloat32, "colors must be float32");

    if (automatic)
        setGLContext(s.glctx);

    int numColors = colors.size(0);
    ludusUploadColorPalette(NVDR_CTX_PARAMS, s, stream, colors.data_ptr<float>(), numColors);

    if (automatic)
        releaseGLContext();
}

int LudusTimestampedStateWrapper::uploadScene(
    torch::Tensor scene_desc,
    torch::Tensor polyline_pools,
    torch::Tensor polygon_pools,
    torch::Tensor obstacle_pools,
    int max_obstacles_in_pool,
    torch::Tensor timestamps,
    torch::Tensor int32_data,
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor poses,
    torch::Tensor float_data)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(scene_desc));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedState& s = *pState;

    // Validate inputs are on correct device
    NVDR_CHECK(scene_desc.get_device() == cudaDeviceIdx,
               "scene_desc must be on same device as GL context");

    if (automatic)
        setGLContext(s.glctx);

    int sceneId = ludusUploadScene(
        NVDR_CTX_PARAMS, s, stream,
        reinterpret_cast<const TimestampedScene*>(scene_desc.data_ptr<uint8_t>()),
        reinterpret_cast<const TimestampedPolylinePool*>(polyline_pools.data_ptr<uint8_t>()),
        polyline_pools.numel() > 0 ? polyline_pools.size(0) : 0,
        reinterpret_cast<const TimestampedPolygonPool*>(polygon_pools.data_ptr<uint8_t>()),
        polygon_pools.numel() > 0 ? polygon_pools.size(0) : 0,
        reinterpret_cast<const ObstaclePool*>(obstacle_pools.data_ptr<uint8_t>()),
        obstacle_pools.numel() > 0 ? obstacle_pools.size(0) : 0,
        max_obstacles_in_pool,
        timestamps.data_ptr<int64_t>(),
        timestamps.numel(),
        int32_data.data_ptr<int32_t>(),
        int32_data.numel(),
        reinterpret_cast<const Vertex*>(vertices.data_ptr<float>()),
        vertices.size(0),
        reinterpret_cast<const Triangle*>(triangles.data_ptr<int32_t>()),
        triangles.size(0),
        reinterpret_cast<const CameraPose*>(poses.data_ptr<float>()),
        poses.size(0),
        float_data.data_ptr<float>(),
        float_data.numel()
    );

    if (automatic)
        releaseGLContext();

    return sceneId;
}

void LudusTimestampedStateWrapper::uploadCameras(torch::Tensor intrinsics)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(intrinsics));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedState& s = *pState;

    NVDR_CHECK(intrinsics.get_device() == cudaDeviceIdx,
               "intrinsics must be on same device as GL context");

    if (automatic)
        setGLContext(s.glctx);

    ludusUploadCameras(
        NVDR_CTX_PARAMS, s, stream,
        reinterpret_cast<const FThetaCamera*>(intrinsics.data_ptr<float>()),
        intrinsics.size(0)
    );

    if (automatic)
        releaseGLContext();
}

// uploadStyles removed - styles hardcoded in shader

void LudusTimestampedStateWrapper::removeScene(int sceneId)
{
    if (automatic)
        setGLContext(pState->glctx);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ludusRemoveScene(NVDR_CTX_PARAMS, *pState, sceneId, stream);

    if (automatic)
        releaseGLContext();
}

void LudusTimestampedStateWrapper::clearScenes(void)
{
    if (automatic)
        setGLContext(pState->glctx);

    ludusClearScenes(NVDR_CTX_PARAMS, *pState);

    if (automatic)
        releaseGLContext();
}

void LudusTimestampedStateWrapper::preallocateBuffers(int maxScenes, int bytesPerScene)
{
    if (automatic)
        setGLContext(pState->glctx);

    ludusPreallocateBuffers(NVDR_CTX_PARAMS, *pState, maxScenes, bytesPerScene);

    if (automatic)
        releaseGLContext();
}

int LudusTimestampedStateWrapper::uploadScenesBatch(
    torch::Tensor scene_descs,
    torch::Tensor polyline_pools,
    torch::Tensor polygon_pools,
    torch::Tensor obstacle_pools,
    torch::Tensor bounds_tensor,
    torch::Tensor timestamps,
    torch::Tensor int32_data,
    torch::Tensor vertices,
    torch::Tensor triangles,
    torch::Tensor poses,
    torch::Tensor float_data)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(timestamps));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (automatic)
        setGLContext(pState->glctx);

    // bounds_tensor: [N, 10] int32, each row is SceneUploadBounds fields
    int numScenesInBatch = bounds_tensor.size(0);
    auto bounds_a = bounds_tensor.accessor<int32_t, 2>();
    std::vector<SceneUploadBounds> bounds(numScenesInBatch);
    for (int i = 0; i < numScenesInBatch; i++) {
        bounds[i].numPolylinePools  = bounds_a[i][0];
        bounds[i].numPolygonPools   = bounds_a[i][1];
        bounds[i].numObstaclePools  = bounds_a[i][2];
        bounds[i].maxObstaclesInPool = bounds_a[i][3];
        bounds[i].numTimestamps     = bounds_a[i][4];
        bounds[i].numInt32          = bounds_a[i][5];
        bounds[i].numVertices       = bounds_a[i][6];
        bounds[i].numTriangles      = bounds_a[i][7];
        bounds[i].numPoses          = bounds_a[i][8];
        bounds[i].numFloats         = bounds_a[i][9];
    }

    int firstId = ludusUploadScenesBatch(
        NVDR_CTX_PARAMS, *pState, stream,
        numScenesInBatch,
        (const TimestampedScene*)scene_descs.data_ptr(),
        (const TimestampedPolylinePool*)polyline_pools.data_ptr(),
        (const TimestampedPolygonPool*)polygon_pools.data_ptr(),
        (const ObstaclePool*)obstacle_pools.data_ptr(),
        bounds.data(),
        (const int64_t*)timestamps.data_ptr(), (int)timestamps.numel(),
        (const int32_t*)int32_data.data_ptr(), (int)int32_data.numel(),
        (const Vertex*)vertices.data_ptr(), (int)(vertices.numel() / 4),
        (const Triangle*)triangles.data_ptr(), (int)(triangles.numel() / 4),
        (const CameraPose*)poses.data_ptr(), (int)(poses.numel() / 16),
        (const float*)float_data.data_ptr(), (int)float_data.numel()
    );

    if (automatic)
        releaseGLContext();

    return firstId;
}

//------------------------------------------------------------------------
// Forward op: Ludus timestamped batch rendering.

torch::Tensor ludus_timestamped_render_batch(
    LudusTimestampedStateWrapper& stateWrapper,
    torch::Tensor queries,          // [N, 32] uint8 packed RenderQuery structs (32 bytes each)
    torch::Tensor camera_poses,     // [N, 4, 4] float32: world-to-camera in FLU convention
    std::tuple<int, int> resolution)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedState& s = *stateWrapper.pState;

    // Check inputs
    NVDR_CHECK_DEVICE(queries, camera_poses);
    NVDR_CHECK_CONTIGUOUS(queries, camera_poses);

    NVDR_CHECK(queries.get_device() == stateWrapper.cudaDeviceIdx,
               "queries must be on same device as GL context");

    int numQueries = queries.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    // Set GL context
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    // FLU → RDF, then transpose for GLSL column-major layout
    torch::Tensor camera_poses_t = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();

    // Render batch
    ludusRenderBatch(
        NVDR_CTX_PARAMS, s, stream,
        reinterpret_cast<const RenderQuery*>(queries.data_ptr<uint8_t>()),
        reinterpret_cast<const CameraPose*>(camera_poses_t.data_ptr<float>()),
        numQueries,
        width, height
    );

    // Allocate output tensor (RGBA8 = 4 bytes per pixel)
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor out_rgba = torch::empty({numQueries, height, width, 4}, opts);

    // Copy results
    ludusCopyBatchResults(NVDR_CTX_PARAMS, s, stream,
                          out_rgba.data_ptr<uint8_t>(), width, height, numQueries);

    // Release GL context
    if (stateWrapper.automatic)
        releaseGLContext();

    return out_rgba;
}

// Async render batch with double buffering - copies to staging, returns staging pointer
// Returns tuple of (staging_idx, has_prev_data)
// Use ludusTimestampedGetStagingData to retrieve the data
std::tuple<int, bool> ludusTimestampedRenderBatchToStaging(
    LudusTimestampedStateWrapper& stateWrapper,
    torch::Tensor queries,          // [N, 32] uint8: packed RenderQuery structs
    torch::Tensor camera_poses,     // [N, 4, 4] float32: world-to-camera in FLU convention
    std::tuple<int, int> resolution)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedState& s = *stateWrapper.pState;

    // Check inputs
    NVDR_CHECK_DEVICE(queries, camera_poses);
    NVDR_CHECK_CONTIGUOUS(queries, camera_poses);
    NVDR_CHECK(queries.get_device() == stateWrapper.cudaDeviceIdx,
               "queries must be on same device as GL context");

    int numQueries = queries.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    // Set GL context
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    // FLU → RDF, then transpose for GLSL column-major layout
    torch::Tensor camera_poses_t = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();

    // Render batch
    ludusRenderBatch(
        NVDR_CTX_PARAMS, s, stream,
        reinterpret_cast<const RenderQuery*>(queries.data_ptr<uint8_t>()),
        reinterpret_cast<const CameraPose*>(camera_poses_t.data_ptr<float>()),
        numQueries,
        width, height
    );

    // Copy to staging buffer (double buffer ping-pong)
    int writtenIdx = ludusCopyBatchResultsToStaging(
        NVDR_CTX_PARAMS, s, stream,
        width, height, numQueries
    );

    // Check if previous staging buffer has valid data
    int prevIdx = 1 - writtenIdx;
    bool hasPrevData = s.stagingValid[prevIdx] != 0;

    // Release GL context
    if (stateWrapper.automatic)
        releaseGLContext();

    return std::make_tuple(writtenIdx, hasPrevData);
}

// Get data from staging buffer (waits for it to be ready) - BLOCKING version
torch::Tensor ludusTimestampedGetStagingData(
    LudusTimestampedStateWrapper& stateWrapper,
    int stagingIdx)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    if (!s.stagingValid[stagingIdx] || !s.stagingBuffer[stagingIdx])
    {
        return torch::Tensor();  // Return empty tensor
    }

    int width = s.stagingWidth;
    int height = s.stagingHeight;
    int numQueries = s.stagingNumQueries;

    // Wait for staging buffer to be ready (CPU blocking)
    cudaError_t err = cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);
    TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError());

    // Allocate output tensor on same device
    torch::TensorOptions opts = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCUDA, stateWrapper.cudaDeviceIdx);
    torch::Tensor out = torch::empty({numQueries, height, width, 4}, opts);

    // Copy from staging to output
    size_t size = (size_t)width * height * numQueries * 4;
    err = cudaMemcpy(out.data_ptr<uint8_t>(), s.stagingBuffer[stagingIdx],
                     size, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError());

    return out;
}

// Get staging buffer as zero-copy view with GPU-side stream sync (NON-BLOCKING)
// The provided stream will wait for staging to be ready, but CPU returns immediately
torch::Tensor ludusTimestampedGetStagingDataAsync(
    LudusTimestampedStateWrapper& stateWrapper,
    int stagingIdx,
    int64_t streamPtr)  // cudaStream_t as int64
{
    LudusTimestampedState& s = *stateWrapper.pState;

    if (!s.stagingValid[stagingIdx] || !s.stagingBuffer[stagingIdx])
    {
        return torch::Tensor();  // Return empty tensor
    }

    int width = s.stagingWidth;
    int height = s.stagingHeight;
    int numQueries = s.stagingNumQueries;

    // Make the provided stream wait for staging buffer to be ready (GPU-side, non-blocking)
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(streamPtr);
    cudaError_t err = cudaStreamWaitEvent(stream, s.stagingReadyEvent[stagingIdx], 0);
    TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError());

    // Return a view of the staging buffer (zero-copy)
    // The tensor wraps the existing staging buffer memory
    torch::TensorOptions opts = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCUDA, stateWrapper.cudaDeviceIdx);

    // Create tensor from existing memory (no allocation, no copy)
    auto tensor = torch::from_blob(
        s.stagingBuffer[stagingIdx],
        {numQueries, height, width, 4},
        opts
    );

    return tensor;
}

// Start async D2H transfer from staging buffer to pinned host memory
// Returns the pinned buffer index (0 or 1) that will receive data, or -1 on failure
int ludusTimestampedStartAsyncHostTransfer(
    LudusTimestampedStateWrapper& stateWrapper,
    int stagingIdx)
{
    LudusTimestampedState& s = *stateWrapper.pState;
    return ludusStartAsyncHostTransfer(NVDR_CTX_PARAMS, s, stagingIdx);
}

// Check if a specific pinned buffer is ready (non-blocking)
bool ludusTimestampedIsPinnedBufferReady(
    LudusTimestampedStateWrapper& stateWrapper,
    int pinnedIdx)
{
    LudusTimestampedState& s = *stateWrapper.pState;
    return ludusIsPinnedBufferReady(NVDR_CTX_PARAMS, s, pinnedIdx) != 0;
}

// Check if any host transfer is complete (legacy API, non-blocking)
bool ludusTimestampedIsHostTransferComplete(
    LudusTimestampedStateWrapper& stateWrapper)
{
    LudusTimestampedState& s = *stateWrapper.pState;
    return ludusIsHostTransferComplete(NVDR_CTX_PARAMS, s) != 0;
}

// Wait for a specific pinned buffer and return zero-copy view
torch::Tensor ludusTimestampedWaitPinnedBufferView(
    LudusTimestampedStateWrapper& stateWrapper,
    int pinnedIdx)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    int width, height, numQueries;
    uint8_t* hostPtr = ludusWaitPinnedBuffer(NVDR_CTX_PARAMS, s, pinnedIdx, &width, &height, &numQueries);

    if (!hostPtr || numQueries == 0)
    {
        return torch::Tensor();  // Return empty tensor
    }

    // Create a tensor that wraps the pinned buffer (zero-copy view)
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::from_blob(hostPtr, {numQueries, height, width, 4}, options);
}

// Wait for async host transfer and return data as CPU tensor
torch::Tensor ludusTimestampedWaitHostTransfer(
    LudusTimestampedStateWrapper& stateWrapper)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    int width, height, numQueries;
    uint8_t* hostPtr = ludusWaitHostTransfer(NVDR_CTX_PARAMS, s, &width, &height, &numQueries);

    if (!hostPtr || numQueries == 0)
    {
        return torch::Tensor();  // Return empty tensor
    }

    // Create CPU tensor and copy from pinned memory
    torch::TensorOptions opts = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCPU);
    torch::Tensor out = torch::empty({numQueries, height, width, 4}, opts);

    // Copy from pinned host buffer to tensor
    size_t size = (size_t)width * height * numQueries * 4;
    memcpy(out.data_ptr<uint8_t>(), hostPtr, size);

    return out;
}

// Wait for async host transfer and return a view of the pinned buffer (zero-copy)
// WARNING: This tensor is only valid until the next start_async_host_transfer call!
torch::Tensor ludusTimestampedWaitHostTransferView(
    LudusTimestampedStateWrapper& stateWrapper)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    int width, height, numQueries;
    uint8_t* hostPtr = ludusWaitHostTransfer(NVDR_CTX_PARAMS, s, &width, &height, &numQueries);

    if (!hostPtr || numQueries == 0)
    {
        return torch::Tensor();  // Return empty tensor
    }

    // Create a tensor that wraps the pinned buffer (zero-copy view)
    // The deleter is empty because the buffer is owned by the renderer state
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::from_blob(hostPtr, {numQueries, height, width, 4}, options);
}

// ========== NVJPEG Hardware Encoding ==========

// Check if NVJPEG is available
bool ludusTimestampedIsNvjpegAvailable(
    LudusTimestampedStateWrapper& stateWrapper)
{
    LudusTimestampedState& s = *stateWrapper.pState;
    return s.nvjpegInitialized != 0;
}

// Encode a single image from staging buffer to JPEG
// Returns bytes object with compressed JPEG data
py::bytes ludusTimestampedEncodeJpegStaging(
    LudusTimestampedStateWrapper& stateWrapper,
    int stagingIdx,
    int imageIdx,
    int quality)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    uint8_t* jpegData = nullptr;
    size_t jpegSize = ludusEncodeJpegStaging(NVDR_CTX_PARAMS, s, stagingIdx, imageIdx, quality, &jpegData);

    if (jpegSize == 0 || !jpegData)
    {
        return py::bytes();  // Empty bytes
    }

    return py::bytes(reinterpret_cast<char*>(jpegData), jpegSize);
}

// Encode all images from staging buffer to JPEG
// Returns list of bytes objects
py::list ludusTimestampedEncodeJpegBatchStaging(
    LudusTimestampedStateWrapper& stateWrapper,
    int stagingIdx,
    int quality)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    std::vector<std::pair<uint8_t*, size_t>> jpegs;
    int count = ludusEncodeJpegBatchStaging(NVDR_CTX_PARAMS, s, stagingIdx, quality, jpegs);

    py::list result;
    for (int i = 0; i < count; i++)
    {
        if (jpegs[i].first && jpegs[i].second > 0)
        {
            result.append(py::bytes(reinterpret_cast<char*>(jpegs[i].first), jpegs[i].second));
            delete[] jpegs[i].first;  // Free the copy we made
        }
        else
        {
            result.append(py::bytes());
        }
    }

    return result;
}

// Encode from pinned buffer
py::bytes ludusTimestampedEncodeJpegPinned(
    LudusTimestampedStateWrapper& stateWrapper,
    int pinnedIdx,
    int imageIdx,
    int quality)
{
    LudusTimestampedState& s = *stateWrapper.pState;

    uint8_t* jpegData = nullptr;
    size_t jpegSize = ludusEncodeJpegPinned(NVDR_CTX_PARAMS, s, pinnedIdx, imageIdx, quality, &jpegData);

    if (jpegSize == 0 || !jpegData)
    {
        return py::bytes();
    }

    return py::bytes(reinterpret_cast<char*>(jpegData), jpegSize);
}

//------------------------------------------------------------------------
