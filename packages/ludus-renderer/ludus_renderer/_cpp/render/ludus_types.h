// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#pragma once
#include <cstdint>

//=============================================================================
// Ludus F-Theta Rendering Types
//
// Memory-efficient structures for mesh shader-based rendering with f-theta
// fisheye camera distortion. Primitives are procedurally tessellated on GPU.
//=============================================================================

//=============================================================================
// Enums
//=============================================================================

// Polyline end cap style
enum CapStyle : uint32_t {
    CAP_NONE  = 0,  // No cap, line ends abruptly
    CAP_FLAT  = 1,  // Flat cap (extends by half width)
    CAP_ROUND = 2   // Round cap (semicircle)
};

// Primitive type IDs for style lookup table
enum PrimTypeId : uint32_t {
    PRIM_ROAD_BOUNDARY    = 0,
    PRIM_LANE_LINE        = 1,
    PRIM_CROSSWALK        = 2,
    PRIM_STATIC_OBSTACLE  = 3,
    PRIM_EGO_TRAJECTORY   = 4,
    PRIM_OBSTACLE         = 5,  // Dynamic obstacles (uses front/back colors from ObstaclePool)
    PRIM_EGO_OBSTACLE     = 6,  // Ego vehicle obstacle
    PRIM_TYPE_COUNT       = 7
};

// Camera type IDs for per-camera-type style adjustments
enum CameraTypeId : uint32_t {
    CAMERA_TYPE_REGULAR   = 0,  // Standard perspective/fisheye camera
    CAMERA_TYPE_BEV       = 1,  // Bird's eye view (top-down orthographic)
    CAMERA_TYPE_COUNT     = 2
};

// Primitive rendering style (32 bytes)
// Lookup table indexed by prim_type_id at render time
//
// What each primitive type uses:
//   Polylines: color, width, bev_width
//   Polygons:  color only
//   Obstacles: nothing (colors from per-instance data)
//
// NOTE: Primitive styles (colors, widths) are hardcoded directly in the shader.
// No PrimitiveStyle struct needed - prim_type_id indexes shader constants.

//=============================================================================
// Primitive Headers
//
// These headers describe primitives stored in SSBOs. The actual geometry
// (vertices, triangles) is stored separately and indexed by these headers.
//=============================================================================

// Polyline: open thick line strip with solid color (32 bytes)
struct PolylineHeader {
    uint32_t vertex_start;    // Index into vertex buffer (first vertex)
    uint32_t vertex_count;    // Number of points in polyline
    float    width;           // Screen-space width in pixels
    float    color[3];        // Solid RGB color [0,1]
    uint32_t cap_style;       // CapStyle enum (CAP_NONE, CAP_FLAT, CAP_ROUND)
    uint32_t _pad;            // Padding to align to 32 bytes
};

// Polygon: filled polygon with solid color (32 bytes)
// Triangulated on CPU before upload; mesh shader handles tessellation
struct PolygonHeader {
    uint32_t vertex_start;    // Index into vertex buffer (first vertex)
    uint32_t vertex_count;    // Number of boundary vertices
    uint32_t triangle_start;  // Index into triangle buffer (first triangle)
    uint32_t triangle_count;  // Number of triangles (from triangulation)
    float    color[3];        // Solid RGB color [0,1]
    uint32_t _pad;            // Padding to align to 32 bytes
};

// Cube: oriented box defined by 9-DOF transform (64 bytes, cache-line aligned)
// Unit cube (±0.5) is procedurally generated and transformed in mesh shader
struct Cube {
    float translation[3];     // World-space center position
    float scale[3];           // Half-extents in local X, Y, Z
    float rotation[3];        // Axis-angle rotation (Rodrigues vector)
    float _pad0;              // Padding for alignment
    float front_color[3];     // RGB for front-facing triangles (+Z local)
    float back_color[3];      // RGB for back-facing triangles (-Z local)
};

//=============================================================================
// Camera Types
//=============================================================================

// F-theta fisheye camera intrinsics (72 bytes = 18 floats)
// Maps incident angle θ to pixel radius r via polynomial: r = Σ(poly[i] * θ^i)
struct FThetaCamera {
    float principal_point[2]; // Optical center (cx, cy) in pixels
    float image_size[2];      // Image dimensions (width, height) in pixels
    float fw_poly[6];         // Forward polynomial coefficients (θ → r)
    float max_ray_angle;      // Maximum valid ray angle in radians
    float max_distortion_val; // r(max_ray_angle) for extrapolation
    float max_distortion_dval;// dr/dθ at max_ray_angle for extrapolation
    float depth_max;          // Maximum depth for z-buffer normalization
    float linear_dist[4];     // 2×2 affine distortion matrix [[c,d],[e,f]]
};

// Camera extrinsics: world-to-camera transform (64 bytes)
struct CameraPose {
    float world_to_camera[16]; // 4×4 matrix in column-major order (for GLSL)
};

//=============================================================================
// Geometry Buffers
//=============================================================================

// Vertex position (16 bytes)
// Used by polylines and polygons; cubes generate vertices procedurally
struct Vertex {
    float position[3];        // World-space position (x, y, z)
    float _pad;               // Padding for 16-byte alignment
};

// Triangle indices (16 bytes - padded for std430 uvec3 alignment)
// Indices are local to the polygon's vertex range
struct Triangle {
    uint32_t indices[3];      // Local vertex indices within polygon
    uint32_t _pad;            // Padding for 16-byte alignment (matches GLSL uvec3)
};

//=============================================================================
// Task Shader Payload
//=============================================================================

// Chunk info passed from task shader to mesh shader (28 bytes)
// Used for splitting long polylines across multiple mesh shader workgroups
struct ChunkInfo {
    uint32_t primitive_id;    // Index of polyline/polygon in header array
    uint32_t camera_id;       // Camera index for this chunk
    uint32_t start_vertex;    // First vertex/triangle in chunk
    uint32_t end_vertex;      // Last vertex/triangle in chunk (inclusive)
    uint32_t subdivision;     // Subdivision level based on distortion
    uint32_t is_first;        // 1 if this chunk has start cap
    uint32_t is_last;         // 1 if this chunk has end cap
};

//=============================================================================
// Scene Descriptor (for push constants or uniform buffer)
//=============================================================================

struct SceneDescriptor {
    uint32_t num_polylines;   // Number of polyline primitives
    uint32_t num_polygons;    // Number of polygon primitives
    uint32_t num_cubes;       // Number of cube primitives
    uint32_t num_cameras;     // Number of cameras to render
    uint32_t total_vertices;  // Total vertices in vertex buffer
    uint32_t total_triangles; // Total triangles in triangle buffer
    uint32_t _pad[2];         // Padding to 32 bytes
};

//=============================================================================
// Compile-time Size Verification
//=============================================================================

static_assert(sizeof(PolylineHeader) == 32, "PolylineHeader must be 32 bytes");
static_assert(sizeof(PolygonHeader) == 32, "PolygonHeader must be 32 bytes");
static_assert(sizeof(Cube) == 64, "Cube must be 64 bytes");
static_assert(sizeof(FThetaCamera) == 72, "FThetaCamera must be 72 bytes");
static_assert(sizeof(CameraPose) == 64, "CameraPose must be 64 bytes");
static_assert(sizeof(Vertex) == 16, "Vertex must be 16 bytes");
static_assert(sizeof(Triangle) == 16, "Triangle must be 16 bytes (padded for std430 uvec3)");
static_assert(sizeof(ChunkInfo) == 28, "ChunkInfo must be 28 bytes");
static_assert(sizeof(SceneDescriptor) == 32, "SceneDescriptor must be 32 bytes");

//=============================================================================
// Timestamped Primitive Pools
//
// For GPU-native temporal rendering. Multiple scenes are uploaded once,
// then hundreds of (scene_id, camera_id, timestamp_us) queries are rendered
// in a single batched draw call. Task shaders perform binary search to
// find visible primitives at each query timestamp.
//=============================================================================

// Pool of timestamped polylines for one element type in a scene (64 bytes)
// Style (color, width, cap) is looked up at render time via prim_type_id.
// Temporal data is stored in separate buffers indexed by offsets.
struct TimestampedPolylinePool {
    uint32_t num_timestamps;        // Number of observation timestamps
    uint32_t num_varrays;           // Total polylines across all timestamps
    uint32_t num_vertices;          // Total vertices across all timestamps
    uint32_t prim_type_id;          // Primitive type for shader style lookup
    // Buffer offsets (indices into global scene buffers)
    uint32_t timestamps_offset;     // -> int64[] timestamps_us
    uint32_t ts_varrays_ps_offset;  // -> int32[] timestamped_varrays_prefix_sum
    uint32_t varrays_ps_offset;     // -> int32[] varrays_prefix_sum
    uint32_t vertices_offset;       // -> Vertex[] vertices
    uint32_t _pad[8];               // Padding to 64 bytes
};

// Pool of timestamped polygons for one element type in a scene (64 bytes)
// Style (color) is looked up at render time via prim_type_id.
// Pre-triangulated with triangle indices.
struct TimestampedPolygonPool {
    uint32_t num_timestamps;        // Number of observation timestamps
    uint32_t num_varrays;           // Total polygons across all timestamps
    uint32_t num_vertices;          // Total vertices across all timestamps
    uint32_t num_triangles;         // Total triangles across all timestamps
    uint32_t prim_type_id;          // Primitive type for shader style lookup
    // Buffer offsets
    uint32_t timestamps_offset;     // -> int64[] timestamps_us
    uint32_t ts_varrays_ps_offset;  // -> int32[] timestamped_varrays_prefix_sum
    uint32_t varrays_ps_offset;     // -> int32[] varrays_prefix_sum (vertex counts)
    uint32_t tri_ps_offset;         // -> int32[] varrays_triangle_prefix_sum
    uint32_t vertices_offset;       // -> Vertex[] vertices
    uint32_t triangles_offset;      // -> Triangle[] triangles
    uint32_t _pad[5];               // Padding to 64 bytes (11 fields * 4 = 44, need 20 more)
};

// Pool of tracked obstacles (dynamic objects) for a scene (64 bytes)
// Each obstacle has a trajectory of poses over time. Interpolation
// is done in the task shader using slerp/lerp between timestamps.
// Per-instance colors stored separately; prim_type_id controls visibility.
// Used for: dynamic obstacles (cars), traffic lights, traffic signs, ego vehicle, etc.
struct ObstaclePool {
    uint32_t num_cubes;             // Number of cubes (was: num_obstacles)
    uint32_t num_timestamps;        // Number of global timestamps
    uint32_t num_track_poses;       // Total track poses (sum over all cubes)
    uint32_t prim_type_id;          // Semantic type for color lookup
    // Buffer offsets
    uint32_t timestamps_offset;     // -> int64[] timestamps_us (global timeline)
    uint32_t cube_ts_ps_offset;     // -> int32[] per-cube track length prefix sum (was: obstacle_ts_ps_offset)
    uint32_t track_timestamps_offset; // -> int64[] per-pose timestamps (for each track point)
    uint32_t translations_offset;   // -> float[3][] translations per track pose
    uint32_t quaternions_offset;    // -> float[4][] quaternions (x,y,z,w) per track pose
    uint32_t scales_offset;         // -> float[3][] scales per cube
    uint32_t colors_offset;         // -> float[6][] (front_rgb, back_rgb) per cube
    uint32_t render_flags;          // CUBE_FLAG_* bits (e.g., CUBE_FLAG_WIREFRAME)
    uint32_t _pad1[4];              // Padding to 64 bytes
};

// Cube render flags
constexpr uint32_t CUBE_FLAG_WIREFRAME = 1;  // Draw wireframe edges in addition to solid faces

// Type alias for backward compatibility
using CubePool = ObstaclePool;

// Scene descriptor containing all pools for one scene (128 bytes)
// A scene has multiple element types (road_boundary, lane_line, etc.)
struct TimestampedScene {
    // Polyline pools (road_boundary, lane_line, static_obstacle)
    uint32_t num_polyline_pools;    // Number of polyline pools (typically 3)
    uint32_t polyline_pools_offset; // -> TimestampedPolylinePool[]
    // Polygon pools (crosswalk)
    uint32_t num_polygon_pools;     // Number of polygon pools (typically 1)
    uint32_t polygon_pools_offset;  // -> TimestampedPolygonPool[]
    // Cube pools (obstacles, traffic lights, traffic signs, ego)
    uint32_t num_cube_pools;        // Number of cube pools (was: has_obstacle_pool 0/1)
    uint32_t cube_pools_offset;     // -> CubePool[] (was: obstacle_pool_offset)
    // Global data buffer offsets
    uint32_t timestamps_buffer_offset;  // Start of this scene's timestamps in global buffer
    uint32_t int32_buffer_offset;       // Start of this scene's int32 data
    uint32_t vertex_buffer_offset;      // Start of this scene's vertices
    uint32_t triangle_buffer_offset;    // Start of this scene's triangles
    uint32_t pose_buffer_offset;        // Start of this scene's poses
    uint32_t float_buffer_offset;       // Start of this scene's float data (scales, colors)
    uint32_t _pad[20];                  // Padding to 128 bytes
};

// Render query: what to render in one output layer (16 bytes)
struct RenderQuery {
    uint32_t scene_id;              // Index into scenes array
    uint32_t camera_id;             // Index into cameras array
    int64_t  timestamp_us;          // Query timestamp in microseconds
    uint32_t camera_type_id;        // CameraTypeId for per-camera-type style selection
    uint32_t _pad[3];               // Padding to 32 bytes
};

static_assert(sizeof(RenderQuery) == 32, "RenderQuery must be 32 bytes");

// Batch render descriptor (32 bytes)
struct RenderBatchDescriptor {
    uint32_t num_queries;           // Number of render queries
    uint32_t num_scenes;            // Number of loaded scenes
    uint32_t num_cameras;           // Number of cameras
    uint32_t output_width;          // Output image width
    uint32_t output_height;         // Output image height
    float    tessellation_threshold;// Pixel error threshold (0 = disabled)
    uint32_t _pad[2];               // Padding to 32 bytes
};

//=============================================================================
// Compile-time Size Verification for Timestamped Types
//=============================================================================

static_assert(sizeof(TimestampedPolylinePool) == 64, "TimestampedPolylinePool must be 64 bytes");
static_assert(sizeof(TimestampedPolygonPool) == 64, "TimestampedPolygonPool must be 64 bytes");
static_assert(sizeof(ObstaclePool) == 64, "ObstaclePool must be 64 bytes");
static_assert(sizeof(TimestampedScene) == 128, "TimestampedScene must be 128 bytes");
// Note: RenderQuery is 32 bytes (static_assert earlier in file)
static_assert(sizeof(RenderBatchDescriptor) == 32, "RenderBatchDescriptor must be 32 bytes");

//=============================================================================
// GLSL-Compatible Constants (for embedding in shader source)
//=============================================================================

#define LUDUS_GLSL_CONSTANTS \
    "// Cap styles\n" \
    "const uint CAP_NONE  = 0u;\n" \
    "const uint CAP_FLAT  = 1u;\n" \
    "const uint CAP_ROUND = 2u;\n" \
    "\n" \
    "// Primitive types for unified dispatch\n" \
    "const uint PRIM_POLYLINE = 0u;\n" \
    "const uint PRIM_POLYGON  = 1u;\n" \
    "const uint PRIM_OBSTACLE = 2u;\n"

#define LUDUS_GLSL_STRUCTS \
    "// PolylineHeader (32 bytes)\n" \
    "struct PolylineHeader {\n" \
    "    uint  vertex_start;\n" \
    "    uint  vertex_count;\n" \
    "    float width;\n" \
    "    vec3  color;\n" \
    "    uint  cap_style;\n" \
    "};\n" \
    "\n" \
    "// PolygonHeader (32 bytes)\n" \
    "struct PolygonHeader {\n" \
    "    uint  vertex_start;\n" \
    "    uint  vertex_count;\n" \
    "    uint  triangle_start;\n" \
    "    uint  triangle_count;\n" \
    "    vec3  color;\n" \
    "};\n" \
    "\n" \
    "// Cube (64 bytes)\n" \
    "struct Cube {\n" \
    "    vec3 translation;\n" \
    "    vec3 scale;\n" \
    "    vec3 rotation;\n" \
    "    float _pad0;\n" \
    "    vec3 front_color;\n" \
    "    vec3 back_color;\n" \
    "};\n" \
    "\n" \
    "// FThetaCamera (64 bytes)\n" \
    "struct FThetaCamera {\n" \
    "    vec2  principal_point;\n" \
    "    vec2  image_size;\n" \
    "    float fw_poly[6];\n" \
    "    float max_ray_angle;\n" \
    "    float max_distortion_val;\n" \
    "    float max_distortion_dval;\n" \
    "    float depth_max;\n" \
    "    mat2  linear_dist;\n" \
    "};\n" \
    "\n" \
    "// ChunkInfo (task payload)\n" \
    "struct ChunkInfo {\n" \
    "    uint primitive_id;\n" \
    "    uint camera_id;\n" \
    "    uint start_vertex;\n" \
    "    uint end_vertex;\n" \
    "    uint subdivision;\n" \
    "    uint is_first;\n" \
    "    uint is_last;\n" \
    "};\n"

//=============================================================================
