// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#pragma once

//------------------------------------------------------------------------
// Do not try to include OpenGL stuff when compiling CUDA kernels for torch.

#if !(defined(NVDR_TORCH) && defined(__CUDACC__))
#include "../common/framework.h"
#include "../common/glutil.h"
#include "ludus_types.h"
#include <nvjpeg.h>

// CUDA kernel for RGBA->RGB conversion with vertical flip (defined in ludus_jpeg.cu)
extern "C" void launchRgbaToRgbFlip(
    const uint8_t* srcRgba,
    uint8_t* dstRgb,
    int width,
    int height,
    cudaStream_t stream
);

//------------------------------------------------------------------------
// Ludus: F-theta mesh shader rendering state.
// Supports polylines, polygons, and cubes with fisheye camera distortion.

struct LudusGLState // Must be initializable by memset to zero.
{
    // Framebuffer dimensions
    int                     width;              // Allocated frame buffer width.
    int                     height;             // Allocated frame buffer height.
    int                     depth;              // Allocated frame buffer depth (num_cameras).

    // Buffer sizes (for resize tracking)
    int                     polylineHeaderCount;    // Allocated polyline headers.
    int                     polygonHeaderCount;     // Allocated polygon headers.
    int                     cubeCount;          // Allocated cubes.
    int                     vertexCount;            // Allocated vertices (floats).
    int                     triangleCount;          // Allocated triangles (ints).
    int                     cameraCount;            // Allocated cameras.

    // GL context
    GLContext               glctx;

    // Framebuffer objects
    GLuint                  glFBO;
    GLuint                  glColorBuffer;
    GLuint                  glDepthStencilBuffer;

    // MSAA support
    int                     msaaSamples;            // 0 or 1 = disabled, 2/4/8 = MSAA sample count
    GLuint                  glFBO_MSAA;             // MSAA framebuffer (render target)
    GLuint                  glColorBuffer_MSAA;     // MSAA color texture (GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    GLuint                  glDepthStencilBuffer_MSAA; // MSAA depth/stencil texture

    // Shader storage buffers (SSBOs)
    GLuint                  glPolylineHeaderBuffer; // SSBO 0: PolylineHeader[]
    GLuint                  glPolygonHeaderBuffer;  // SSBO 1: PolygonHeader[]
    GLuint                  glCubeBuffer;       // SSBO 2: Cube[]
    GLuint                  glVertexBuffer;         // SSBO 3: Vertex[]
    GLuint                  glTriangleBuffer;       // SSBO 4: Triangle[]
    GLuint                  glCameraIntrinsicsBuffer; // SSBO 5: FThetaCamera[]
    GLuint                  glCameraPoseBuffer;     // SSBO 6: CameraPose[]

    // Mesh shader programs (one per primitive type)
    GLuint                  glProgramPolyline;      // Task + Mesh + Fragment for polylines
    GLuint                  glProgramPolygon;       // Task + Mesh + Fragment for polygons
    GLuint                  glProgramCube;      // Task + Mesh + Fragment for cubes

    // Shader objects
    GLuint                  glTaskShaderPolyline;
    GLuint                  glMeshShaderPolyline;
    GLuint                  glTaskShaderPolygon;
    GLuint                  glMeshShaderPolygon;
    GLuint                  glTaskShaderCube;
    GLuint                  glMeshShaderCube;
    GLuint                  glFragmentShader;       // Shared fragment shader

    // CUDA-GL interop resources
    cudaGraphicsResource_t  cudaColorBuffer;
    cudaGraphicsResource_t  cudaPolylineHeaderBuffer;
    cudaGraphicsResource_t  cudaPolygonHeaderBuffer;
    cudaGraphicsResource_t  cudaCubeBuffer;
    cudaGraphicsResource_t  cudaVertexBuffer;
    cudaGraphicsResource_t  cudaTriangleBuffer;
    cudaGraphicsResource_t  cudaCameraIntrinsicsBuffer;
    cudaGraphicsResource_t  cudaCameraPoseBuffer;

    // Capability flags
    int                     hasMeshShader;          // 1 if GL_NV_mesh_shader or GL_EXT_mesh_shader available
    int                     enableZModify;          // Depth modification workaround

    // Tessellation parameters
    float                   tessellationThreshold;  // Pixel error threshold for adaptive tessellation (0 = disabled)
};

//------------------------------------------------------------------------
// Ludus F-theta rendering functions.

// Initialize Ludus GL context and compile mesh shaders.
void ludusInitGLContext(NVDR_CTX_ARGS, LudusGLState& s, int cudaDeviceIdx);

// Resize Ludus buffers as needed.
void ludusResizeBuffers(
    NVDR_CTX_ARGS,
    LudusGLState& s,
    bool& changes,
    int polylineHeaderCount,
    int polygonHeaderCount,
    int cubeCount,
    int vertexCount,
    int triangleCount,
    int cameraCount,
    int width,
    int height
);

// Render scene with f-theta cameras using mesh shaders.
void ludusRender(
    NVDR_CTX_ARGS,
    LudusGLState& s,
    cudaStream_t stream,
    // Polylines
    const PolylineHeader* polylineHeaders,
    int numPolylines,
    // Polygons
    const PolygonHeader* polygonHeaders,
    int numPolygons,
    // Cubes
    const Cube* cubes,
    int numCubes,
    // Geometry buffers
    const Vertex* vertices,
    int numVertices,
    const Triangle* triangles,
    int numTriangles,
    // Cameras
    const FThetaCamera* cameraIntrinsics,
    const CameraPose* cameraPoses,
    int numCameras,
    // Output dimensions
    int width,
    int height
);

// Copy rendered results to output tensor.
void ludusCopyResults(
    NVDR_CTX_ARGS,
    LudusGLState& s,
    cudaStream_t stream,
    uint8_t* outputPtr,
    int width,
    int height,
    int numCameras
);

// Release Ludus buffers and resources.
void ludusReleaseBuffers(NVDR_CTX_ARGS, LudusGLState& s);

//------------------------------------------------------------------------
// Ludus Timestamped Rendering State
//
// For GPU-native temporal rendering where multiple scenes are loaded once
// and hundreds of (scene_id, camera_id, timestamp_us) queries are rendered
// in a single batched draw call.

// Buffer set for double-buffered GL scene data.
// Contains all SSBOs that hold scene data + CUDA interop handles + usage counters.
struct LudusTimestampedBufferSet // memset-to-zero safe
{
    // Scene storage
    int                     maxScenes;
    int                     numScenes;

    // Capacity & usage counters
    int                     timestampsCapacity, timestampsUsed;
    int                     int32Capacity,      int32Used;
    int                     vertexCapacity,     vertexUsed;
    int                     triangleCapacity,   triangleUsed;
    int                     poseCapacity,       poseUsed;
    int                     floatCapacity,      floatUsed;
    int                     polylinePoolCapacity, polylinePoolUsed;
    int                     polygonPoolCapacity,  polygonPoolUsed;
    int                     obstaclePoolCapacity, obstaclePoolUsed;

    // Dispatch sizing
    int                     maxObstaclesPerPool;
    int                     maxCubePoolsPerScene;
    int                     maxPolylinePoolsPerScene;
    int                     maxPolygonPoolsPerScene;

    // GL SSBOs
    GLuint                  glSceneBuffer;
    GLuint                  glTimestampsBuffer;
    GLuint                  glInt32Buffer;
    GLuint                  glVertexBuffer;
    GLuint                  glTriangleBuffer;
    GLuint                  glPoseBuffer;
    GLuint                  glFloatBuffer;
    GLuint                  glPolylinePoolBuffer;
    GLuint                  glPolygonPoolBuffer;
    GLuint                  glObstaclePoolBuffer;

    // CUDA-GL interop
    cudaGraphicsResource_t  cudaSceneBuffer;
    cudaGraphicsResource_t  cudaTimestampsBuffer;
    cudaGraphicsResource_t  cudaInt32Buffer;
    cudaGraphicsResource_t  cudaVertexBuffer;
    cudaGraphicsResource_t  cudaTriangleBuffer;
    cudaGraphicsResource_t  cudaPoseBuffer;
    cudaGraphicsResource_t  cudaFloatBuffer;
    cudaGraphicsResource_t  cudaPolylinePoolBuffer;
    cudaGraphicsResource_t  cudaPolygonPoolBuffer;
    cudaGraphicsResource_t  cudaObstaclePoolBuffer;

    // GL fence for double-buffer swap synchronization
    GLsync                  glFence;
};

struct LudusTimestampedState // Must be initializable by memset to zero.
{
    // ========== Framebuffer ==========
    int                     width;              // Allocated frame buffer width
    int                     height;             // Allocated frame buffer height
    int                     maxLayers;          // Max output layers (batch size)

    // ========== Double-Buffered Scene Data ==========
    LudusTimestampedBufferSet bufferSets[2];
    int                     activeSet;          // 0 or 1: index of the set used for rendering

    // ========== Camera Storage ==========
    int                     cameraCapacity;     // Allocated cameras
    int                     numCameras;         // Loaded cameras

    // ========== Query Batch ==========
    int                     queryCapacity;      // Allocated query slots
    int                     posePerQueryCapacity; // Allocated pose-per-query slots

    // ========== GL Context ==========
    GLContext               glctx;

    // ========== Framebuffer Objects ==========
    GLuint                  glFBO;
    GLuint                  glColorBuffer;      // GL_TEXTURE_2D_ARRAY for layered output
    GLuint                  glDepthStencilBuffer;

    // ========== MSAA Support ==========
    int                     msaaSamples;            // 0 or 1 = disabled, 2/4/8 = MSAA sample count
    GLuint                  glFBO_MSAA;             // MSAA framebuffer (render target)
    GLuint                  glColorBuffer_MSAA;     // MSAA color texture (GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    GLuint                  glDepthStencilBuffer_MSAA; // MSAA depth/stencil texture

    // ========== Color Palette (configurable from Python) ==========
    GLuint                  glColorPaletteBuffer;   // SSBO: vec4[] colors per prim_type_id
    cudaGraphicsResource_t  cudaColorPaletteBuffer;
    int                     colorPaletteSize;       // Number of colors in palette

    // ========== Camera Buffers ==========
    GLuint                  glCameraIntrinsicsBuffer; // SSBO: FThetaCamera[]
    GLuint                  glCameraPoseBuffer;       // SSBO: CameraPose[]

    // ========== Query Buffers ==========
    GLuint                  glQueryBuffer;          // SSBO: RenderQuery[]
    GLuint                  glBatchDescBuffer;      // UBO: RenderBatchDescriptor

    // ========== Shader Programs ==========
    GLuint                  glProgramPolyline;      // Timestamped polyline program
    GLuint                  glProgramPolygon;       // Timestamped polygon program
    GLuint                  glProgramObstacle;      // Timestamped obstacle program

    // ========== Shader Objects ==========
    GLuint                  glTaskShaderPolyline;
    GLuint                  glMeshShaderPolyline;
    GLuint                  glTaskShaderPolygon;
    GLuint                  glMeshShaderPolygon;
    GLuint                  glTaskShaderObstacle;
    GLuint                  glMeshShaderObstacle;
    GLuint                  glFragmentShader;           // For per-vertex color interpolation
    GLuint                  glFragmentShaderPolyline;   // For polyline (uses perprimitiveNV)
    GLuint                  glFragmentShaderObstacle;   // For obstacle (gradient via perprimitiveNV)

    // ========== CUDA-GL Interop (shared / non-scene resources) ==========
    cudaGraphicsResource_t  cudaColorBuffer;
    cudaGraphicsResource_t  cudaCameraIntrinsicsBuffer;
    cudaGraphicsResource_t  cudaCameraPoseBuffer;
    cudaGraphicsResource_t  cudaQueryBuffer;

    // ========== Capability Flags ==========
    int                     hasMeshShader;
    int                     enableZModify;
    float                   tessellationThreshold;

    // ========== Max Tessellation Levels (cap adaptive subdivision) ==========
    int                     maxTessellationLevelPolyline;  // 0..4, default 4
    int                     maxTessellationLevelPolygon;   // 0..3, default 3
    int                     maxTessellationLevelCube;      // 0..3, default 3

    // ========== Configurable Widths (pixels at reference resolution 1280x720) ==========
    float                   widthPolylineRegular;   // 0 = use default (12.0)
    float                   widthPolylineBev;       // 0 = use default (5.0)
    float                   widthEgoTrajRegular;    // 0 = use default (12.0)
    float                   widthEgoTrajBev;        // 0 = use default (5.0)
    float                   widthWireframe;         // 0 = use default (3.0)
    float                   resolutionScale;        // 0 = use 1.0 (no scaling)
    float                   depthScaling;           // 1.0 = enable distance-based fog/line scaling
    int                     maxExtrapolationUs;     // Max extrapolation time in microseconds (default 500000 = 500ms)
    float                   cullRadiusScale;        // Multiplier on cam.depth_max for spatial culling (0 = disabled, 1.5 = default)

    // ========== Double Buffer for Async Transfer ==========
    // Ping-pong staging buffers to hide GL→CUDA→CPU latency
    uint8_t*                stagingBuffer[2];       // Two CUDA device staging buffers
    size_t                  stagingBufferSize;      // Allocated size per buffer (bytes)
    int                     stagingWidth;           // Dimensions of staged data
    int                     stagingHeight;
    int                     stagingNumQueries;
    int                     currentStagingIdx;      // 0 or 1 (which buffer is being written)
    int                     stagingValid[2];        // Whether staging[i] has valid data
    cudaStream_t            copyStream;             // Separate stream for async CPU transfer
    cudaEvent_t             stagingReadyEvent[2];   // Signals when staging[i] is ready for CPU copy

    // Double-buffered pinned host memory for async D2H transfer
    uint8_t*                pinnedHostBuffer[2];    // Two pinned host buffers (ping-pong)
    size_t                  pinnedHostBufferSize;   // Allocated size per buffer
    int                     currentPinnedIdx;       // Which pinned buffer is being written to
    int                     pinnedValid[2];         // Whether pinned[i] has valid data
    int                     pinnedWidth[2];         // Dimensions of data in pinned[i]
    int                     pinnedHeight[2];
    int                     pinnedNumQueries[2];
    cudaEvent_t             pinnedReadyEvent[2];    // Signals when pinned[i] is ready for CPU use

    // NVJPEG hardware encoder state
    nvjpegHandle_t          nvjpegHandle;           // NVJPEG library handle
    nvjpegEncoderState_t    nvjpegEncoderState;     // Encoder state
    nvjpegEncoderParams_t   nvjpegEncoderParams;    // Encoder parameters
    int                     nvjpegInitialized;      // Whether nvjpeg is initialized
    uint8_t*                jpegOutputBuffer;       // Output buffer for compressed JPEG
    size_t                  jpegOutputBufferSize;   // Size of output buffer
    uint8_t*                jpegFlipBuffer;         // Temp buffer for vertical flip
    size_t                  jpegFlipBufferSize;     // Size of flip buffer
};

//------------------------------------------------------------------------
// Ludus Timestamped Rendering API
//
// Usage:
// 1. ludusTimestampedInit() - Initialize context and compile shaders
// 2. ludusUploadCameras() - Upload camera intrinsics (once, or when cameras change)
// 3. ludusUploadScene() - Upload one scene's timestamped data (returns scene_id)
// 4. ludusRenderBatch() - Render batch of (scene_id, camera_id, timestamp, pose) queries
// 5. ludusCopyBatchResults() - Copy results to output tensor
// 6. ludusTimestampedRelease() - Cleanup

// Initialize timestamped rendering context.
void ludusTimestampedInit(NVDR_CTX_ARGS, LudusTimestampedState& s, int cudaDeviceIdx);

// Upload camera intrinsics. Poses are provided per-query in renderBatch.
void ludusUploadCameras(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const FThetaCamera* intrinsics,
    int numCameras
);

// ludusUploadStyles removed - styles hardcoded in shader

// Upload one scene's timestamped data. Returns scene_id for use in queries.
// The scene data is stored persistently until ludusTimestampedRelease().
int ludusUploadScene(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    // Scene metadata
    const TimestampedScene* sceneDesc,
    // Pool headers
    const TimestampedPolylinePool* polylinePools,
    int numPolylinePools,
    const TimestampedPolygonPool* polygonPools,
    int numPolygonPools,
    const ObstaclePool* obstaclePools,
    int numObstaclePools,
    int maxObstaclesInPool,  // Max obstacles in any pool (for dispatch)
    // Global data buffers
    const int64_t* timestamps,
    int numTimestamps,
    const int32_t* int32Data,
    int numInt32,
    const Vertex* vertices,
    int numVertices,
    const Triangle* triangles,
    int numTriangles,
    const CameraPose* poses,
    int numPoses,
    const float* floatData,
    int numFloats
);

// Render batch of queries. Each query specifies (scene_id, camera_id, timestamp_us).
// Camera poses are provided per-query to support dynamic viewpoints.
void ludusRenderBatch(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const RenderQuery* queries,
    const CameraPose* cameraPoses,  // One pose per query
    int numQueries,
    int width,
    int height
);

// Copy batch results to output tensor.
// Output shape: [numQueries, height, width, 4] (RGBA uint8)
void ludusCopyBatchResults(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    uint8_t* outputPtr,
    int width,
    int height,
    int numQueries
);

// ========== Double Buffer Async Transfer API ==========
// Copy rendered results to staging buffer (ping-pong double buffer).
// Returns the staging buffer index that was written to.
int ludusCopyBatchResultsToStaging(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    int width,
    int height,
    int numQueries
);

// Get pointer to staging buffer that's ready for reading (previous frame).
// Returns nullptr if no valid data available yet.
uint8_t* ludusGetReadyStagingBuffer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int* outWidth,
    int* outHeight,
    int* outNumQueries
);

// Copy from staging buffer to output tensor (synchronous).
void ludusCopyStagingToOutput(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx,
    uint8_t* outputPtr,
    int width,
    int height,
    int numQueries
);

// ========== Async Host Transfer API (Double-Buffered) ==========
// Start async D2H transfer from staging buffer to pinned host memory.
// Uses double-buffered pinned memory for true async operation.
// Returns the pinned buffer index (0 or 1) that will receive data, or -1 on failure.
int ludusStartAsyncHostTransfer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx
);

// Check if a specific pinned buffer is ready (non-blocking).
// Returns 1 if complete, 0 if still in progress or invalid.
int ludusIsPinnedBufferReady(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int pinnedIdx
);

// Check if any host transfer is complete (legacy API).
int ludusIsHostTransferComplete(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s
);

// Wait for a specific pinned buffer and get its data.
// Returns pointer to pinned host buffer, or nullptr if invalid.
uint8_t* ludusWaitPinnedBuffer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int pinnedIdx,
    int* outWidth,
    int* outHeight,
    int* outNumQueries
);

// Wait for the previous pinned buffer (legacy API).
uint8_t* ludusWaitHostTransfer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int* outWidth,
    int* outHeight,
    int* outNumQueries
);

// ========== NVJPEG Hardware Encoding ==========
// Encode image from GPU memory to JPEG.
size_t ludusEncodeJpegGpu(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    const uint8_t* gpuRgba,
    int width,
    int height,
    int quality,
    uint8_t** outJpegData
);

// Encode image from pinned buffer to JPEG.
size_t ludusEncodeJpegPinned(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int pinnedIdx,
    int imageIdx,
    int quality,
    uint8_t** outJpegData
);

// Encode image directly from staging buffer to JPEG (most efficient).
size_t ludusEncodeJpegStaging(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx,
    int imageIdx,
    int quality,
    uint8_t** outJpegData
);

// Batch encode all images from staging buffer.
int ludusEncodeJpegBatchStaging(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx,
    int quality,
    std::vector<std::pair<uint8_t*, size_t>>& outJpegs
);

// Tombstone a single scene (set valid=0, data stays in buffers).
void ludusRemoveScene(NVDR_CTX_ARGS, LudusTimestampedState& s, int sceneId, cudaStream_t stream);

// Pre-allocate all data buffers for up to maxScenes scenes.
// bytesPerScene is an estimate of average scene memory; buffers
// are sized to maxScenes * bytesPerScene (each buffer proportionally).
void ludusPreallocateBuffers(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int maxScenes,
    int bytesPerScene
);

// Batch-upload N scenes with a single map/unmap cycle.
// sceneDescs, polylinePoolsList, etc. are packed arrays of all scene data.
// sceneBounds[i] describes the per-scene offsets/counts within those arrays.
struct SceneUploadBounds {
    int numPolylinePools;
    int numPolygonPools;
    int numObstaclePools;
    int maxObstaclesInPool;
    int numTimestamps;
    int numInt32;
    int numVertices;
    int numTriangles;
    int numPoses;
    int numFloats;
};

// Upload N scenes in a single map/unmap cycle.
// All data arrays are concatenated across scenes in order.
// Returns the scene_id of the first uploaded scene.
int ludusUploadScenesBatch(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    int numScenesInBatch,
    const TimestampedScene* sceneDescs,          // [numScenesInBatch]
    const TimestampedPolylinePool* polylinePools, // [sum of numPolylinePools]
    const TimestampedPolygonPool* polygonPools,   // [sum of numPolygonPools]
    const ObstaclePool* obstaclePools,            // [sum of numObstaclePools]
    const SceneUploadBounds* bounds,              // [numScenesInBatch]
    const int64_t* timestamps,                    // [sum of numTimestamps]
    int totalTimestamps,
    const int32_t* int32Data,                     // [sum of numInt32]
    int totalInt32,
    const Vertex* vertices,                       // [sum of numVertices]
    int totalVertices,
    const Triangle* triangles,                    // [sum of numTriangles]
    int totalTriangles,
    const CameraPose* poses,                      // [sum of numPoses]
    int totalPoses,
    const float* floatData,                       // [sum of numFloats]
    int totalFloats
);

// Clear all loaded scenes in the active buffer set (but keep context alive).
void ludusClearScenes(NVDR_CTX_ARGS, LudusTimestampedState& s);

// Swap active and back buffer sets for double-buffered scene data.
// Waits for any pending GL fence on the back set before swapping.
void ludusSwapBufferSets(NVDR_CTX_ARGS, LudusTimestampedState& s);

// Upload color palette for primitive types (configurable from Python).
// colors: array of RGB values [r0,g0,b0, r1,g1,b1, ...] for each prim_type_id
// numColors: number of colors (should match PRIM_TYPE_COUNT)
void ludusUploadColorPalette(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const float* colors,
    int numColors
);

// Release all resources.
void ludusTimestampedRelease(NVDR_CTX_ARGS, LudusTimestampedState& s);

//------------------------------------------------------------------------
#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))
