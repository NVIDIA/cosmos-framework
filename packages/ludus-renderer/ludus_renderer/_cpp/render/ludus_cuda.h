// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#pragma once

#include "../common/framework.h"
#include "ludus_types.h"
#include "../cudaraster/CudaRaster.hpp"
#include <cuda_runtime.h>

//------------------------------------------------------------------------
// Fully CUDA-based Ludus renderer state.
// No OpenGL dependency. Uses CudaRaster (HPG 2011) for triangle rasterization.
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Per-frame render parameters passed to all CUDA kernels.
// Bundles all configurable settings to keep kernel signatures clean.
//------------------------------------------------------------------------

struct CudaRenderParams
{
    float widthPolylineRegular;     // 0 = use default (7.0)
    float widthPolylineBev;         // 0 = use default (3.0)
    float widthEgoTrajRegular;      // 0 = use default (12.0)
    float widthEgoTrajBev;          // 0 = use default (6.0)
    float widthWireframe;           // 0 = use default (2.0)
    float resolutionScale;          // 0 or 1.0 = no scaling
    float depthScaling;             // 1.0 = enabled, 0.0 = disabled
    float cullRadiusScale;          // 0 = disabled, default 1.5
    float tessellationThreshold;
    int   maxTessPolyline;          // 0..4, default 4
    int   maxTessPolygon;           // 0..3, default 3
    int   maxTessCube;              // 0..3, default 3
    int   cameraTypeId;             // 0 = regular, 1 = BEV
    int   colorPaletteSize;         // 0 = use hardcoded defaults
    const uint32_t* colorPalette;   // device pointer, packed RGBA8 per prim type
};

struct LudusCudaState
{
    int                     width, height, numCameras;
    CR::CudaRaster*         cr;

    // Geometry buffers (GPU, managed by renderer)
    float*                  projectedVertices;  // float4 per vertex
    int*                    triangleIndices;    // int3 per triangle
    uint32_t*               vertexColors;       // packed RGBA8 per vertex (for barycentric interpolation)
    int*                    triangleRanges;     // int2 per camera (offset, count) for range mode

    int                     maxVertices;        // per-camera max vertex count
    int                     maxTriangles;       // total max triangle count
    int                     allocatedVertices;
    int                     allocatedTriangles;

    // Atomic counter for geometry generation
    int*                    atomicVertexCount;
    int*                    atomicTriangleCount;

    // Output framebuffer (RGBA8, numCameras * width * height)
    uint8_t*                outputBuffer;
    int                     outputBufferSize;

    float                   tessellationThreshold;

    // Configurable render settings (set via Python API)
    float                   widthPolylineRegular;
    float                   widthPolylineBev;
    float                   widthEgoTrajRegular;
    float                   widthEgoTrajBev;
    float                   widthWireframe;
    float                   resolutionScale;
    float                   depthScaling;
    float                   cullRadiusScale;
    int                     maxTessPolyline;
    int                     maxTessPolygon;
    int                     maxTessCube;

    // Color palette (GPU buffer, PRIM_TYPE_COUNT entries)
    uint32_t*               colorPalette;
    int                     colorPaletteSize;

    // MSAA (implemented as 2x supersampling when msaaSamples >= 4)
    int                     msaaSamples;        // 0 = disabled, 4 = 4x SSAA
    uint8_t*                msaaBuffer;         // hi-res RGBA8 intermediate buffer
    int                     msaaBufferSize;
};

//------------------------------------------------------------------------
// Timestamped cube pool parameters (passed to render for on-GPU interpolation).
//------------------------------------------------------------------------

struct CubePoolParams
{
    const int64_t* trackTimestamps;   // [total_poses] int64, per-pose timestamps (GPU)
    const int32_t* prefixSum;         // [n_cubes] int32, cumulative track lengths (GPU)
    const float*   translations;      // [total_poses * 3] float32, per-pose xyz (GPU)
    const float*   quaternions;       // [total_poses * 4] float32, per-pose xyzw (GPU)
    const float*   scales;            // [n_cubes * 3] float32 (GPU)
    const float*   colors;            // [n_cubes * 6] float32, (front_rgb, back_rgb) (GPU)
    int            numCubes;
    int64_t        queryTimestampUs;
    int            maxExtrapolationUs;
    uint32_t       renderFlags;       // CUBE_FLAG_* bits
};

struct DotParams
{
    const Vertex*  positions;         // [numDots] world-space positions (GPU)
    int            numDots;
    float          radius;            // half-width in pixels (e.g. 3.5)
    uint32_t       color;             // packed RGBA8
};

//------------------------------------------------------------------------
// Timestamped pool headers (16 x uint32, matching GL SSBO layout).
// Offsets are global indices into the flat buffers.
//------------------------------------------------------------------------

struct TsPolylinePoolHeader {
    uint32_t num_timestamps;          // [0]
    uint32_t num_varrays;             // [1]
    uint32_t num_vertices;            // [2]
    uint32_t prim_type_id;            // [3]
    uint32_t timestamps_offset;       // [4] into timestamps buffer
    uint32_t ts_varrays_ps_offset;    // [5] into int32 buffer
    uint32_t varrays_ps_offset;       // [6] into int32 buffer
    uint32_t vertices_offset;         // [7] into vertex buffer
    uint32_t aabb_offset;             // [8] into float buffer
    uint32_t _pad[7];
};

struct TsPolygonPoolHeader {
    uint32_t num_timestamps;          // [0]
    uint32_t num_varrays;             // [1]
    uint32_t num_vertices;            // [2]
    uint32_t num_triangles;           // [3]
    uint32_t prim_type_id;            // [4]
    uint32_t timestamps_offset;       // [5]
    uint32_t ts_varrays_ps_offset;    // [6]
    uint32_t varrays_ps_offset;       // [7]
    uint32_t tri_ps_offset;           // [8]
    uint32_t vertices_offset;         // [9]
    uint32_t triangles_offset;        // [10]
    uint32_t aabb_offset;             // [11]
    uint32_t _pad[4];
};

struct TsCubePoolHeader {
    uint32_t num_cubes;               // [0]
    uint32_t num_timestamps;          // [1] global timeline
    uint32_t num_track_poses;         // [2]
    uint32_t prim_type_id;            // [3]
    uint32_t timestamps_offset;       // [4] global timestamps
    uint32_t cube_ts_ps_offset;       // [5] per-cube track prefix sum in int32
    uint32_t track_timestamps_offset; // [6] per-pose timestamps
    uint32_t translations_offset;     // [7] in float buffer
    uint32_t quaternions_offset;      // [8] in float buffer
    uint32_t scales_offset;           // [9] in float buffer
    uint32_t colors_offset;           // [10] in float buffer
    uint32_t render_flags;            // [11]
    uint32_t _pad[4];
};

//------------------------------------------------------------------------
// API functions.
//------------------------------------------------------------------------

void ludusCudaInit(NVDR_CTX_ARGS, LudusCudaState& s);
void ludusCudaDestroy(NVDR_CTX_ARGS, LudusCudaState& s);

void ludusCudaRender(
    NVDR_CTX_ARGS,
    LudusCudaState& s,
    cudaStream_t stream,
    const PolylineHeader* polylineHeaders, int numPolylines,
    const PolygonHeader* polygonHeaders, int numPolygons,
    const Cube* cubes, int numCubes,
    const Vertex* vertices, int numVertices,
    const Triangle* triangles, int numTriangles,
    const FThetaCamera* cameras, const CameraPose* poses,
    int numCameras, int width, int height,
    uint8_t* outputPtr,
    const CubePoolParams* cubePools = nullptr, int numCubePools = 0,
    const DotParams* dots = nullptr, int numDotGroups = 0);

void ludusCudaRenderTimestamped(
    NVDR_CTX_ARGS,
    LudusCudaState& s,
    cudaStream_t stream,
    const int64_t* timestamps,
    const int32_t* int32Data,
    const Vertex* vertices,
    const Triangle* triangles,
    const float* floatData,
    const TsPolylinePoolHeader* polylinePools, int numPolylinePools,
    const TsPolygonPoolHeader* polygonPools, int numPolygonPools,
    const TsCubePoolHeader* cubePools, int numCubePools,
    int64_t queryTimestampUs, int maxExtrapolationUs,
    int maxVarraysPerTsPolyline, int maxVarraysPerTsPolygon,
    const CudaRenderParams& params,
    const FThetaCamera* cameras, const CameraPose* poses,
    int numCameras, int width, int height,
    uint8_t* outputPtr);

void ludusCudaUploadColorPalette(LudusCudaState& s, const uint32_t* hostPalette, int count);

//------------------------------------------------------------------------
