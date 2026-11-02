// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

//=============================================================================
// Ludus Timestamped Rendering
//
// GPU-native temporal rendering for scenes with time-varying geometry.
// Multiple scenes are uploaded once, then hundreds of (scene_id, camera_id,
// timestamp_us) queries are rendered in a single batched draw call.
// Task shaders perform binary search to find visible primitives at each
// query timestamp.
//=============================================================================

#include "ludus_gl.h"
#include "../common/glutil.h"
#include "ludus_types.h"
#include <vector>
#include <cstring>

//------------------------------------------------------------------------
// Helpers

#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
static int ROUND_UP_BITS(uint32_t x, uint32_t y)
{
    if (x < (1u << y))
        return x;
    uint32_t m = 0;
    while (x & ~m)
        m = (m << 1) | 1u;
    m >>= y;
    if (!(x & m))
        return x;
    return (x | m) + 1u;
}

//------------------------------------------------------------------------
// Shader compilation helpers

static void compileTimestampedShader(NVDR_CTX_ARGS, GLuint* pShader, GLenum shaderType, const char* src_buf, bool enableZModify)
{
    std::string src(src_buf);

    // Find the #version line and insert after it
    size_t versionPos = src.find("#version");
    size_t insertPos = 0;
    if (versionPos != std::string::npos)
    {
        insertPos = src.find('\n', versionPos);
        if (insertPos != std::string::npos)
            insertPos++;
    }

    // Set preprocessor directives
    if (enableZModify)
        src.insert(insertPos, "#define IF_ZMODIFY(x) x\n");
    else
        src.insert(insertPos, "#define IF_ZMODIFY(x)\n");

    const char* cstr = src.c_str();
    *pShader = 0;
    NVDR_CHECK_GL_ERROR(*pShader = glCreateShader(shaderType));
    NVDR_CHECK_GL_ERROR(glShaderSource(*pShader, 1, &cstr, 0));
    NVDR_CHECK_GL_ERROR(glCompileShader(*pShader));

    GLint compileStatus = 0;
    NVDR_CHECK_GL_ERROR(glGetShaderiv(*pShader, GL_COMPILE_STATUS, &compileStatus));
    if (!compileStatus)
    {
        GLint infoLen = 0;
        NVDR_CHECK_GL_ERROR(glGetShaderiv(*pShader, GL_INFO_LOG_LENGTH, &infoLen));
        if (infoLen)
        {
            std::vector<char> info(infoLen + 1);
            NVDR_CHECK_GL_ERROR(glGetShaderInfoLog(*pShader, infoLen, &infoLen, &info[0]));
            LOG(ERROR) << "Timestamped shader compilation failed:\n" << &info[0];
            NVDR_CHECK(0, "Shader compilation failed");
        }
        NVDR_CHECK(0, "Shader compilation failed");
    }
}

static void constructTimestampedProgram(NVDR_CTX_ARGS, GLuint* pProgram, GLuint taskShader, GLuint meshShader, GLuint fragmentShader)
{
    *pProgram = 0;

    GLuint glProgram = 0;
    NVDR_CHECK_GL_ERROR(glProgram = glCreateProgram());
    if (taskShader)
        NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, taskShader));
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, meshShader));
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, fragmentShader));
    NVDR_CHECK_GL_ERROR(glLinkProgram(glProgram));

    GLint linkStatus = 0;
    NVDR_CHECK_GL_ERROR(glGetProgramiv(glProgram, GL_LINK_STATUS, &linkStatus));
    if (!linkStatus)
    {
        GLint infoLen = 0;
        NVDR_CHECK_GL_ERROR(glGetProgramiv(glProgram, GL_INFO_LOG_LENGTH, &infoLen));
        if (infoLen)
        {
            std::vector<char> info(infoLen + 1);
            NVDR_CHECK_GL_ERROR(glGetProgramInfoLog(glProgram, infoLen, &infoLen, &info[0]));
            LOG(ERROR) << "Timestamped program linking failed:\n" << &info[0];
            NVDR_CHECK(0, "glLinkProgram() failed");
        }
        NVDR_CHECK(0, "glLinkProgram() failed");
    }

    *pProgram = glProgram;
}

//=============================================================================
// GLSL Shader Source Code
//=============================================================================

// Common GLSL code for timestamped rendering
// Includes version, extensions, buffer layouts, and utility functions
static const char* LUDUS_TIMESTAMPED_COMMON = R"(#version 450
#extension GL_NV_mesh_shader : require
#extension GL_NV_gpu_shader5 : require
#extension GL_ARB_gpu_shader_int64 : require

// Cap styles
const uint CAP_NONE  = 0u;
const uint CAP_FLAT  = 1u;
const uint CAP_ROUND = 2u;

// Primitive type IDs (must match Python PRIM_* constants in ops.py)
const uint PRIM_ROAD_BOUNDARY    = 0u;
const uint PRIM_LANE_LINE        = 1u;   // Legacy cyan lane line
const uint PRIM_CROSSWALK        = 2u;
const uint PRIM_STATIC_OBSTACLE  = 3u;   // Legacy, avoid using
const uint PRIM_EGO_TRAJECTORY   = 4u;
const uint PRIM_OBSTACLE         = 5u;
const uint PRIM_EGO_OBSTACLE     = 6u;
const uint PRIM_WAIT_LINE        = 7u;
const uint PRIM_POLE             = 8u;
const uint PRIM_ROAD_MARKING     = 9u;
const uint PRIM_LANE_BOUNDARY    = 10u;
const uint PRIM_TRAFFIC_LIGHT    = 11u;
const uint PRIM_TRAFFIC_SIGN     = 12u;
const uint PRIM_INTERSECTION     = 13u;
const uint PRIM_ROAD_ISLAND      = 14u;
const uint PRIM_BUFFER_ZONE      = 15u;
// Lane line variants by color/style
const uint PRIM_LANE_LINE_WHITE_SOLID   = 16u;
const uint PRIM_LANE_LINE_WHITE_DASHED  = 17u;
const uint PRIM_LANE_LINE_YELLOW_SOLID  = 18u;
const uint PRIM_LANE_LINE_YELLOW_DASHED = 19u;
// Dot primitives - each vertex is rendered as a circle
const uint PRIM_DOT_YELLOW = 20u;
const uint PRIM_DOT_WHITE  = 21u;

// Camera type IDs
const uint CAMERA_TYPE_REGULAR = 0u;
const uint CAMERA_TYPE_BEV     = 1u;

//=============================================================================
// Hardcoded Primitive Styles (matching wm-render colorscheme v3)
//=============================================================================

// Colors (RGB normalized) - matching imaginaire4 v3 colorscheme
// Source: imaginaire4/imaginaire/auxiliary/world_scenario/color_scheme/config_color_hdmap.json
const vec3 COLOR_ROAD_BOUNDARY   = vec3(253.0/255.0, 1.0/255.0, 232.0/255.0);   // Magenta (253, 1, 232)
const vec3 COLOR_LANE_LINE       = vec3(98.0/255.0, 183.0/255.0, 249.0/255.0);  // Cyan (98, 183, 249) - "lanelines" (legacy)
const vec3 COLOR_CROSSWALK       = vec3(139.0/255.0, 93.0/255.0, 255.0/255.0);  // Purple (139, 93, 255)
const vec3 COLOR_STATIC_OBSTACLE = vec3(255.0/255.0, 100.0/255.0, 0.0/255.0);   // Orange (legacy)
const vec3 COLOR_EGO_TRAJECTORY  = vec3(0.0/255.0, 255.0/255.0, 0.0/255.0);     // Green (0, 255, 0)
const vec3 COLOR_WAIT_LINE       = vec3(108.0/255.0, 179.0/255.0, 59.0/255.0);  // Yellow-green (108, 179, 59)
const vec3 COLOR_POLE            = vec3(183.0/255.0, 69.0/255.0, 177.0/255.0);  // Purple-magenta (183, 69, 177)
const vec3 COLOR_ROAD_MARKING    = vec3(20.0/255.0, 254.0/255.0, 185.0/255.0);  // Cyan-green (20, 254, 185)
const vec3 COLOR_LANE_BOUNDARY   = vec3(98.0/255.0, 183.0/255.0, 249.0/255.0);  // Cyan (same as lane line)
const vec3 COLOR_TRAFFIC_LIGHT   = vec3(100.0/255.0, 100.0/255.0, 100.0/255.0); // Gray (100, 100, 100)
const vec3 COLOR_TRAFFIC_SIGN    = vec3(8.0/255.0, 2.0/255.0, 255.0/255.0);     // Blue (8, 2, 255)
const vec3 COLOR_INTERSECTION    = vec3(80.0/255.0, 80.0/255.0, 120.0/255.0);   // Dark blue-gray (approximated)
const vec3 COLOR_ROAD_ISLAND     = vec3(60.0/255.0, 120.0/255.0, 60.0/255.0);   // Dark green (approximated)
const vec3 COLOR_BUFFER_ZONE     = vec3(120.0/255.0, 80.0/255.0, 80.0/255.0);   // Dark red-brown (approximated)
// Lane line variants from config_color_geometry_laneline.json
const vec3 COLOR_LANE_LINE_WHITE  = vec3(255.0/255.0, 255.0/255.0, 255.0/255.0); // White (255, 255, 255)
const vec3 COLOR_LANE_LINE_YELLOW = vec3(255.0/255.0, 255.0/255.0, 0.0/255.0);   // Yellow (255, 255, 0)

// Default polyline widths (pixels at reference resolution 1280x720)
const float DEFAULT_WIDTH_POLYLINE_REGULAR = 7.0;
const float DEFAULT_WIDTH_POLYLINE_BEV     = 4.0;
const float DEFAULT_WIDTH_EGO_TRAJ_REGULAR = 12.0;
const float DEFAULT_WIDTH_EGO_TRAJ_BEV     = 5.0;
const float DEFAULT_WIDTH_POLE_REGULAR     = 5.0;   // Poles are thinner (reference: 5 vs 12)
const float DEFAULT_WIDTH_POLE_BEV         = 3.0;
const float DEFAULT_WIDTH_WIREFRAME        = 2.0;

// Configurable width uniforms (set from Python, or use defaults)
uniform float u_width_polyline_regular;  // 0 = use default
uniform float u_width_polyline_bev;
uniform float u_width_ego_traj_regular;
uniform float u_width_ego_traj_bev;
uniform float u_width_wireframe;
uniform float u_resolution_scale;  // Scale factor based on current resolution vs reference (1280x720)
uniform float u_depth_scaling;  // 1.0 = enable distance-based width/fog scaling, 0.0 = disable
uniform int u_max_extrapolation_us;  // Max extrapolation time in microseconds (default 500000 = 500ms)

// Compute depth-based scaling factor for line width
// z_ndc is in [-1, 1] where -1 is near, +1 is far
// Returns 1.0 at near, 0.0 at far (linear fade)
float get_depth_scale(vec4 clip) {
    if (u_depth_scaling < 0.5) return 1.0;  // Disabled
    float z_ndc = clip.z / clip.w;  // Convert to NDC
    float depth_scale = (1.0 - z_ndc) / 2.0;  // Map [-1,1] to [1,0]
    return clamp(depth_scale, 0.0, 1.0);
}

// Color palette buffer (optional, configured from Python)
// Declared early so get_prim_color can use it; actual buffer layout defined later
layout(std430, binding = 10) readonly buffer ColorPaletteBufferEarly {
    vec4 g_color_palette[];
};
uniform int u_color_palette_size;  // 0 = use hardcoded defaults

// Get default color for primitive type (hardcoded fallback)
vec3 get_default_prim_color(uint prim_type_id) {
    if (prim_type_id == PRIM_ROAD_BOUNDARY)   return COLOR_ROAD_BOUNDARY;
    if (prim_type_id == PRIM_LANE_LINE)       return COLOR_LANE_LINE;
    if (prim_type_id == PRIM_CROSSWALK)       return COLOR_CROSSWALK;
    if (prim_type_id == PRIM_STATIC_OBSTACLE) return COLOR_STATIC_OBSTACLE;
    if (prim_type_id == PRIM_EGO_TRAJECTORY)  return COLOR_EGO_TRAJECTORY;
    if (prim_type_id == PRIM_WAIT_LINE)       return COLOR_WAIT_LINE;
    if (prim_type_id == PRIM_POLE)            return COLOR_POLE;
    if (prim_type_id == PRIM_ROAD_MARKING)    return COLOR_ROAD_MARKING;
    if (prim_type_id == PRIM_LANE_BOUNDARY)   return COLOR_LANE_BOUNDARY;
    if (prim_type_id == PRIM_TRAFFIC_LIGHT)   return COLOR_TRAFFIC_LIGHT;
    if (prim_type_id == PRIM_TRAFFIC_SIGN)    return COLOR_TRAFFIC_SIGN;
    if (prim_type_id == PRIM_INTERSECTION)    return COLOR_INTERSECTION;
    if (prim_type_id == PRIM_ROAD_ISLAND)     return COLOR_ROAD_ISLAND;
    if (prim_type_id == PRIM_BUFFER_ZONE)     return COLOR_BUFFER_ZONE;
    // Lane line variants
    if (prim_type_id == PRIM_LANE_LINE_WHITE_SOLID)   return COLOR_LANE_LINE_WHITE;
    if (prim_type_id == PRIM_LANE_LINE_WHITE_DASHED)  return COLOR_LANE_LINE_WHITE;
    if (prim_type_id == PRIM_LANE_LINE_YELLOW_SOLID)  return COLOR_LANE_LINE_YELLOW;
    if (prim_type_id == PRIM_LANE_LINE_YELLOW_DASHED) return COLOR_LANE_LINE_YELLOW;
    if (prim_type_id == PRIM_DOT_YELLOW) return COLOR_LANE_LINE_YELLOW;
    if (prim_type_id == PRIM_DOT_WHITE)  return COLOR_LANE_LINE_WHITE;
    return vec3(1.0, 1.0, 1.0);  // Default white for obstacles
}

// Get color for primitive type (checks palette first, then falls back to defaults)
vec3 get_prim_color(uint prim_type_id) {
    // If color palette is configured, check if this prim_type has a custom color
    // Alpha < 0 means "use default", alpha >= 0 means "use this color"
    if (u_color_palette_size > 0 && prim_type_id < uint(u_color_palette_size)) {
        vec4 palette_color = g_color_palette[prim_type_id];
        if (palette_color.a >= 0.0) {
            return palette_color.rgb;
        }
    }
    // Fall back to hardcoded defaults
    return get_default_prim_color(prim_type_id);
}

// Check if primitive is a dot type (rendered as circles at each vertex)
bool is_dot_primitive(uint prim_type_id) {
    return prim_type_id == PRIM_DOT_YELLOW || prim_type_id == PRIM_DOT_WHITE;
}

// Get polyline width for primitive type and camera type (scaled by resolution)
float get_prim_width(uint prim_type_id, uint camera_type_id) {
    bool is_bev = (camera_type_id == CAMERA_TYPE_BEV);
    float scale = (u_resolution_scale > 0.0) ? u_resolution_scale : 1.0;

    float base_width;
    if (prim_type_id == PRIM_EGO_TRAJECTORY) {
        float default_w = is_bev ? DEFAULT_WIDTH_EGO_TRAJ_BEV : DEFAULT_WIDTH_EGO_TRAJ_REGULAR;
        float custom_w = is_bev ? u_width_ego_traj_bev : u_width_ego_traj_regular;
        base_width = (custom_w > 0.0) ? custom_w : default_w;
    } else if (prim_type_id == PRIM_POLE) {
        // Poles use thinner lines (reference: 5 pixels vs 12 for other polylines)
        base_width = is_bev ? DEFAULT_WIDTH_POLE_BEV : DEFAULT_WIDTH_POLE_REGULAR;
    } else {
        float default_w = is_bev ? DEFAULT_WIDTH_POLYLINE_BEV : DEFAULT_WIDTH_POLYLINE_REGULAR;
        float custom_w = is_bev ? u_width_polyline_bev : u_width_polyline_regular;
        base_width = (custom_w > 0.0) ? custom_w : default_w;
    }
    return base_width * scale;
}

// Get wireframe edge width (scaled by resolution)
float get_wireframe_width() {
    float scale = (u_resolution_scale > 0.0) ? u_resolution_scale : 1.0;
    float base_width = (u_width_wireframe > 0.0) ? u_width_wireframe : DEFAULT_WIDTH_WIREFRAME;
    return base_width * scale;
}

//=============================================================================
// Data Structures (must match C++ structs)
//=============================================================================

// TimestampedPolylinePool (64 bytes)
struct TimestampedPolylinePool {
    uint  num_timestamps;
    uint  num_varrays;
    uint  num_vertices;
    uint  prim_type_id;              // Index into PrimitiveStyle lookup table
    uint  timestamps_offset;
    uint  ts_varrays_ps_offset;
    uint  varrays_ps_offset;
    uint  vertices_offset;
    uint  aabb_offset;               // Per-element AABB in float buffer (6 floats each: min xyz, max xyz)
    uint  _pad1, _pad2, _pad3, _pad4, _pad5, _pad6, _pad7;  // Padding to 64 bytes
};

// TimestampedPolygonPool (64 bytes)
struct TimestampedPolygonPool {
    uint  num_timestamps;
    uint  num_varrays;
    uint  num_vertices;
    uint  num_triangles;
    uint  prim_type_id;              // Index into PrimitiveStyle lookup table
    uint  timestamps_offset;
    uint  ts_varrays_ps_offset;
    uint  varrays_ps_offset;
    uint  tri_ps_offset;
    uint  vertices_offset;
    uint  triangles_offset;
    uint  aabb_offset;               // Per-element AABB in float buffer (6 floats each: min xyz, max xyz)
    uint  _pad1, _pad2, _pad3, _pad4;  // Padding to 64 bytes (16 uints)
};

// CubePool (64 bytes) - used for obstacles, traffic lights, traffic signs, etc.
// Renamed from ObstaclePool but keeping struct name for backward compat
struct ObstaclePool {
    uint num_cubes;                  // Was: num_obstacles
    uint num_timestamps;
    uint num_track_poses;
    uint prim_type_id;               // Semantic type for color lookup
    uint timestamps_offset;          // Global timestamps for this pool
    uint cube_ts_ps_offset;          // Per-cube track length prefix sum (was: obstacle_ts_ps_offset)
    uint track_timestamps_offset;    // Per-pose timestamps in float buffer (as 2x uint for int64)
    uint translations_offset;        // Per-pose translations (3 floats each)
    uint quaternions_offset;         // Per-pose quaternions (4 floats each)
    uint scales_offset;              // Per-cube scales (3 floats each)
    uint colors_offset;              // Per-cube colors (6 floats each)
    uint render_flags;               // CUBE_FLAG_* bits (e.g., CUBE_FLAG_WIREFRAME)
    uint _pad2, _pad3, _pad4, _pad5; // 4 padding fields for 64 bytes total
};

// Cube render flags
const uint CUBE_FLAG_WIREFRAME = 1u;  // Draw wireframe edges in addition to solid faces

// TimestampedScene (128 bytes)
struct TimestampedScene {
    uint num_polyline_pools;
    uint polyline_pools_offset;
    uint num_polygon_pools;
    uint polygon_pools_offset;
    uint num_cube_pools;
    uint cube_pools_offset;
    uint timestamps_buffer_offset;
    uint int32_buffer_offset;
    uint vertex_buffer_offset;
    uint triangle_buffer_offset;
    uint pose_buffer_offset;
    uint float_buffer_offset;
    uint valid;                      // 1 = active, 0 = tombstoned (skipped during rendering)
    uint _pad[19];
};

// RenderQuery (16 bytes)
struct RenderQuery {
    uint   scene_id;
    uint   camera_id;
    int64_t timestamp_us;
    uint   camera_type_id;    // CAMERA_TYPE_REGULAR or CAMERA_TYPE_BEV
    uint   _pad1, _pad2, _pad3;  // Padding to 32 bytes
};

// FThetaCamera (72 bytes)
struct FThetaCamera {
    float cx, cy;
    float img_w, img_h;
    float poly0, poly1, poly2, poly3, poly4, poly5;
    float max_ray_angle;
    float max_distortion_val;
    float max_distortion_dval;
    float depth_max;
    float ld_c, ld_d, ld_e, ld_f;
};

// CameraPose (64 bytes)
struct CameraPose {
    mat4 world_to_camera;
};

// Vertex (16 bytes)
struct Vertex {
    float x, y, z;
    float _pad;
};

//=============================================================================
// Buffer Bindings
//=============================================================================

// Global data buffers
layout(std430, binding = 0) readonly buffer TimestampsBuffer {
    int64_t g_timestamps[];
};

layout(std430, binding = 1) readonly buffer Int32Buffer {
    int g_int32[];
};

layout(std430, binding = 2) readonly buffer VertexBuffer {
    Vertex g_vertices[];
};

layout(std430, binding = 3) readonly buffer TriangleBuffer {
    uvec3 g_triangles[];
};

layout(std430, binding = 4) readonly buffer PoseBuffer {
    CameraPose g_poses[];
};

layout(std430, binding = 5) readonly buffer FloatBuffer {
    float g_floats[];
};

// Scene metadata
layout(std430, binding = 6) readonly buffer SceneBuffer {
    TimestampedScene g_scenes[];
};

layout(std430, binding = 7) readonly buffer PolylinePoolBuffer {
    TimestampedPolylinePool g_polyline_pools[];
};

layout(std430, binding = 8) readonly buffer PolygonPoolBuffer {
    TimestampedPolygonPool g_polygon_pools[];
};

layout(std430, binding = 9) readonly buffer ObstaclePoolBuffer {
    ObstaclePool g_obstacle_pools[];
};

// Camera buffers
layout(std430, binding = 11) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera g_camera_intrinsics[];
};

layout(std430, binding = 12) readonly buffer CameraPoseBuffer {
    CameraPose g_camera_poses[];  // One per query (dynamic viewpoints)
};

// Query buffer
layout(std430, binding = 13) readonly buffer QueryBuffer {
    RenderQuery g_queries[];
};

// Uniforms
uniform uint u_num_queries;
uniform float u_tessellation_threshold;
uniform uint u_max_tessellation_polyline;  // 0..4, cap for polyline subdivision
uniform uint u_max_tessellation_polygon;    // 0..3, cap for polygon subdivision
uniform uint u_max_tessellation_cube;      // 0..3, cap for cube edge subdivision
// Spatial culling: elements beyond depth_max * scale from the camera are
// discarded in the task shader.  Scale > 1 gives headroom so nothing at
// the visible boundary pops in/out.  Set to 0 to disable culling.
uniform float u_cull_radius_scale;        // multiplier on cam.depth_max (0 = disabled, default 1.5)

//=============================================================================
// GPU Binary Search
// Returns index of last element <= target, or -1 if target < all elements
//=============================================================================

int binary_search_timestamps(uint base_offset, uint count, int64_t target) {
    if (count == 0u) return -1;

    int left = 0;
    int right = int(count) - 1;
    int result = -1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int64_t val = g_timestamps[base_offset + uint(mid)];

        if (val <= target) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

//=============================================================================
// Quaternion Math for Track Interpolation
//=============================================================================

// Quaternion dot product
float quat_dot(vec4 a, vec4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

// Quaternion normalize
vec4 quat_normalize(vec4 q) {
    float len = length(q);
    return len > 1e-10 ? q / len : vec4(0.0, 0.0, 0.0, 1.0);
}

// Quaternion spherical linear interpolation (slerp)
vec4 quat_slerp(vec4 q0, vec4 q1, float t) {
    // Ensure shortest path
    float d = quat_dot(q0, q1);
    if (d < 0.0) {
        q1 = -q1;
        d = -d;
    }

    // If quaternions are very close, use linear interpolation
    if (d > 0.9995) {
        return quat_normalize(mix(q0, q1, t));
    }

    // Spherical interpolation
    float theta_0 = acos(clamp(d, -1.0, 1.0));
    float theta = theta_0 * t;
    float sin_theta = sin(theta);
    float sin_theta_0 = sin(theta_0);

    float s0 = cos(theta) - d * sin_theta / sin_theta_0;
    float s1 = sin_theta / sin_theta_0;

    return quat_normalize(s0 * q0 + s1 * q1);
}

// Convert quaternion (x, y, z, w) to 3x3 rotation matrix
// GLSL mat3 is column-major: mat3(col0, col1, col2)
mat3 quat_to_matrix(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;

    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    // Column 0: (R00, R10, R20), Column 1: (R01, R11, R21), Column 2: (R02, R12, R22)
    return mat3(
        1.0 - (yy + zz), xy + wz, xz - wy,   // Column 0
        xy - wz, 1.0 - (xx + zz), yz + wx,   // Column 1
        xz + wy, yz - wx, 1.0 - (xx + yy)    // Column 2
    );
}

// Build 4x4 transform from translation and quaternion (with scale)
mat4 build_transform(vec3 translation, vec4 quaternion, vec3 scale) {
    mat3 rot = quat_to_matrix(quaternion);

    // Apply scale to each column of rotation matrix
    mat4 result = mat4(
        vec4(rot[0] * scale.x, 0.0),
        vec4(rot[1] * scale.y, 0.0),
        vec4(rot[2] * scale.z, 0.0),
        vec4(translation, 1.0)
    );

    return result;
}

// Binary search for track interpolation (returns index where t0 <= target < t1)
// Returns -1 if target is outside the track range
int binary_search_track(uint base_offset, uint count, int64_t target) {
    if (count < 2u) return -1;  // Need at least 2 points for interpolation

    int64_t first_ts = g_timestamps[base_offset];
    int64_t last_ts = g_timestamps[base_offset + count - 1u];

    // Check bounds
    if (target < first_ts || target > last_ts) return -1;

    int left = 0;
    int right = int(count) - 2;  // Max valid index for interpolation start
    int result = 0;

    while (left <= right) {
        int mid = (left + right) / 2;
        int64_t t0 = g_timestamps[base_offset + uint(mid)];
        int64_t t1 = g_timestamps[base_offset + uint(mid) + 1u];

        if (t0 <= target && target <= t1) {
            return mid;
        } else if (target < t0) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return -1;  // Should not reach here for valid input
}

//=============================================================================
// Spatial Culling Helpers
//=============================================================================

// AABB-AABB overlap test
bool cull_aabb_overlap(vec3 a_min, vec3 a_max, vec3 b_min, vec3 b_max) {
    return all(lessThanEqual(a_min, b_max)) && all(greaterThanEqual(a_max, b_min));
}

// Sphere-AABB overlap test
bool cull_sphere_aabb_overlap(vec3 center, float radius, vec3 b_min, vec3 b_max) {
    vec3 nearest = clamp(center, b_min, b_max);
    vec3 d = center - nearest;
    return dot(d, d) <= radius * radius;
}

// Extract camera world position from view pose
vec3 get_camera_world_pos(CameraPose pose) {
    mat3 R = mat3(pose.world_to_camera);
    return -transpose(R) * pose.world_to_camera[3].xyz;
}

//=============================================================================
// F-theta Projection
//=============================================================================

vec3 rotate_rodrigues(vec3 v, vec3 r) {
    float theta = length(r);
    if (theta < 1e-8) return v;
    vec3 k = r / theta;
    float c = cos(theta);
    float s = sin(theta);
    return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

// F-theta projection: world point -> NDC
// Supports >180° FOV fisheye cameras where points can have negative z (behind camera plane)
vec4 ftheta_project(vec3 world_pos, CameraPose pose, FThetaCamera cam) {
    vec3 cam_pt = (pose.world_to_camera * vec4(world_pos, 1.0)).xyz;
    float depth = cam_pt.z;

    float ray_norm = length(cam_pt);

    // Handle points at camera origin
    if (ray_norm < 1e-6) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    float half_pi = 1.5707963;  // π/2

    // For cameras with FOV <= 180° (max_ray_angle <= π/2), points behind camera should be clipped
    // Use pseudo-pinhole projection to push them far away in the correct direction
    if (cam.max_ray_angle <= half_pi && depth < 0.001) {
        float pseudo_focal = cam.poly1;
        float x_clip = cam_pt.x * pseudo_focal / (cam.img_w * 0.5);
        float y_clip = -cam_pt.y * pseudo_focal / (cam.img_h * 0.5);
        return vec4(x_clip * 10.0, y_clip * 10.0, 1.0, 1.0);  // Scale by 10 to push far outside
    }

    float xy_norm = length(cam_pt.xy);
    float cos_alpha = clamp(cam_pt.z / ray_norm, -1.0, 1.0);
    float alpha = acos(cos_alpha);  // alpha in [0, π]

    // Apply polynomial projection (with linear extrapolation beyond max_ray_angle)
    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a2 * a2;
    float a5 = a4 * alpha;

    float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                  cam.poly3 * a3 + cam.poly4 * a4 + cam.poly5 * a5;
    if (alpha > cam.max_ray_angle) {
        delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
    }

    float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
    vec2 pixel_rel = scale * cam_pt.xy;

    vec2 pixel_dist;
    pixel_dist.x = cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y;
    pixel_dist.y = cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y;

    vec2 pixel = pixel_dist + vec2(cam.cx, cam.cy);

    float x_ndc = 2.0 * pixel.x / cam.img_w - 1.0;
    float y_ndc = 1.0 - 2.0 * pixel.y / cam.img_h;

    // For z-buffer depth mapping:
    // - Narrow FOV (<=180°): use signed depth (cam_pt.z)
    // - Wide FOV (>180°): use ray_norm for front-camera vertices (unchanged),
    //   but push behind-camera vertices to depth_max so they never occlude
    //   forward geometry.  ray_norm alone can't distinguish front from behind.
    float z_value;
    if (cam.max_ray_angle > half_pi) {
        z_value = (depth >= 0.0) ? ray_norm : cam.depth_max;
    } else {
        z_value = depth;
    }
    float z_ndc = clamp((z_value / cam.depth_max) * 2.0 - 1.0, -1.0, 1.0);

    return vec4(x_ndc, y_ndc, z_ndc, 1.0);
}

// Inline version that takes mat4 directly
float estimate_edge_distortion_pixels_mat4(vec3 v0, vec3 v1, mat4 world_to_cam, FThetaCamera cam) {
    // Transform to camera space
    vec3 cam_pt0 = (world_to_cam * vec4(v0, 1.0)).xyz;
    vec3 cam_pt1 = (world_to_cam * vec4(v1, 1.0)).xyz;
    vec3 mid_world = (v0 + v1) * 0.5;
    vec3 cam_pt_mid = (world_to_cam * vec4(mid_world, 1.0)).xyz;

    // Just clamp depths to avoid division issues

    // For segments in front of camera, compute f-theta projection error directly
    // Use depth-clamped projection to handle near-plane cases
    float depth0 = max(cam_pt0.z, 0.001);
    float depth1 = max(cam_pt1.z, 0.001);
    float depth_mid = max(cam_pt_mid.z, 0.001);

    // Compute f-theta projection for each point
    // We inline the projection to avoid struct-passing issues
    vec2 pixel0, pixel1, pixel_mid;
    {
        float xy_norm = length(cam_pt0.xy);
        float ray_norm = length(vec3(cam_pt0.xy, depth0)) + 1e-10;
        float cos_alpha = clamp(depth0 / ray_norm, -1.0, 1.0);
        float alpha = acos(cos_alpha);
        float a2 = alpha * alpha;
        float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                      cam.poly3 * (a2 * alpha) + cam.poly4 * (a2 * a2) + cam.poly5 * (a2 * a2 * alpha);
        if (alpha > cam.max_ray_angle) {
            delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
        }
        float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
        vec2 pixel_rel = scale * cam_pt0.xy;
        vec2 pixel_dist = vec2(cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y,
                               cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y);
        pixel0 = pixel_dist + vec2(cam.cx, cam.cy);
    }
    {
        float xy_norm = length(cam_pt1.xy);
        float ray_norm = length(vec3(cam_pt1.xy, depth1)) + 1e-10;
        float cos_alpha = clamp(depth1 / ray_norm, -1.0, 1.0);
        float alpha = acos(cos_alpha);
        float a2 = alpha * alpha;
        float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                      cam.poly3 * (a2 * alpha) + cam.poly4 * (a2 * a2) + cam.poly5 * (a2 * a2 * alpha);
        if (alpha > cam.max_ray_angle) {
            delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
        }
        float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
        vec2 pixel_rel = scale * cam_pt1.xy;
        vec2 pixel_dist = vec2(cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y,
                               cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y);
        pixel1 = pixel_dist + vec2(cam.cx, cam.cy);
    }
    {
        float xy_norm = length(cam_pt_mid.xy);
        float ray_norm = length(vec3(cam_pt_mid.xy, depth_mid)) + 1e-10;
        float cos_alpha = clamp(depth_mid / ray_norm, -1.0, 1.0);
        float alpha = acos(cos_alpha);
        float a2 = alpha * alpha;
        float delta = cam.poly0 + cam.poly1 * alpha + cam.poly2 * a2 +
                      cam.poly3 * (a2 * alpha) + cam.poly4 * (a2 * a2) + cam.poly5 * (a2 * a2 * alpha);
        if (alpha > cam.max_ray_angle) {
            delta = cam.max_distortion_val + (alpha - cam.max_ray_angle) * cam.max_distortion_dval;
        }
        float scale = (xy_norm > 1e-6) ? (delta / xy_norm) : 0.0;
        vec2 pixel_rel = scale * cam_pt_mid.xy;
        vec2 pixel_dist = vec2(cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y,
                               cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y);
        pixel_mid = pixel_dist + vec2(cam.cx, cam.cy);
    }

    // Compute error: distance from projected midpoint to linear interpolation
    vec2 linear_pixel_mid = (pixel0 + pixel1) * 0.5;
    float error_pixels = length(pixel_mid - linear_pixel_mid);

    // Clamp to reasonable range to handle edge cases
    return clamp(error_pixels, 0.0, 10000.0);
}

uint compute_subdivision_level(vec3 v0, vec3 v1, CameraPose pose, FThetaCamera cam, float threshold_pixels) {
    float error = estimate_edge_distortion_pixels_mat4(v0, v1, pose.world_to_camera, cam);

    if (error < threshold_pixels) return 0u;
    if (error < threshold_pixels * 2.0) return 1u;
    if (error < threshold_pixels * 4.0) return 2u;
    return 3u;
}

//=============================================================================
// Barycentric Subdivision Utilities (shared with polygon/cube)
//=============================================================================

uint bary_vertex_count(uint level) {
    // Vertices = (n+1)(n+2)/2 where n = 2^level
    if (level == 0u) return 3u;
    if (level == 1u) return 6u;
    if (level == 2u) return 15u;
    return 45u;  // level 3
}

uint bary_triangle_count(uint level) {
    // Triangles = 4^level
    if (level == 0u) return 1u;
    if (level == 1u) return 4u;
    if (level == 2u) return 16u;
    return 64u;  // level 3
}

vec2 bary_vertex_uv(uint vertex_idx, uint level) {
    if (level == 0u) {
        if (vertex_idx == 0u) return vec2(0.0, 0.0);
        if (vertex_idx == 1u) return vec2(1.0, 0.0);
        return vec2(0.0, 1.0);
    } else if (level == 1u) {
        vec2 uvs[6];
        uvs[0] = vec2(0.0, 0.0);
        uvs[1] = vec2(1.0, 0.0);
        uvs[2] = vec2(0.0, 1.0);
        uvs[3] = vec2(0.5, 0.0);
        uvs[4] = vec2(0.5, 0.5);
        uvs[5] = vec2(0.0, 0.5);
        return uvs[vertex_idx];
    } else if (level == 2u) {
        uint row, col;
        if (vertex_idx < 5u) { row = 0u; col = vertex_idx; }
        else if (vertex_idx < 9u) { row = 1u; col = vertex_idx - 5u; }
        else if (vertex_idx < 12u) { row = 2u; col = vertex_idx - 9u; }
        else if (vertex_idx < 14u) { row = 3u; col = vertex_idx - 12u; }
        else { row = 4u; col = 0u; }
        return vec2(float(col) / 4.0, float(row) / 4.0);
    } else {
        // Level 3: 9 rows (0-8), row r has (9-r) vertices
        uint row, col;
        if (vertex_idx < 9u) { row = 0u; col = vertex_idx; }
        else if (vertex_idx < 17u) { row = 1u; col = vertex_idx - 9u; }
        else if (vertex_idx < 24u) { row = 2u; col = vertex_idx - 17u; }
        else if (vertex_idx < 30u) { row = 3u; col = vertex_idx - 24u; }
        else if (vertex_idx < 35u) { row = 4u; col = vertex_idx - 30u; }
        else if (vertex_idx < 39u) { row = 5u; col = vertex_idx - 35u; }
        else if (vertex_idx < 42u) { row = 6u; col = vertex_idx - 39u; }
        else if (vertex_idx < 44u) { row = 7u; col = vertex_idx - 42u; }
        else { row = 8u; col = 0u; }
        return vec2(float(col) / 8.0, float(row) / 8.0);
    }
}

vec3 bary_interpolate(vec3 v0, vec3 v1, vec3 v2, vec2 uv) {
    float w = 1.0 - uv.x - uv.y;
    return v0 * w + v1 * uv.x + v2 * uv.y;
}

uvec3 bary_triangle_indices(uint tri_idx, uint level) {
    if (level == 0u) {
        return uvec3(0u, 1u, 2u);
    } else if (level == 1u) {
        uvec3 tris[4];
        tris[0] = uvec3(0u, 3u, 5u);
        tris[1] = uvec3(3u, 1u, 4u);
        tris[2] = uvec3(5u, 4u, 2u);
        tris[3] = uvec3(3u, 4u, 5u);
        return tris[tri_idx];
    } else if (level == 2u) {
        uint row_start[5] = uint[5](0u, 5u, 9u, 12u, 14u);
        uint t = tri_idx;
        uint row = 0u, col = 0u;
        bool is_down = false;

        if (t < 7u) {
            row = 0u;
            if (t < 4u) { col = t; is_down = false; }
            else { col = t - 4u; is_down = true; }
        } else if (t < 12u) {
            row = 1u;
            uint lt = t - 7u;
            if (lt < 3u) { col = lt; is_down = false; }
            else { col = lt - 3u; is_down = true; }
        } else if (t < 15u) {
            row = 2u;
            uint lt = t - 12u;
            if (lt < 2u) { col = lt; is_down = false; }
            else { col = lt - 2u; is_down = true; }
        } else {
            row = 3u; col = 0u; is_down = false;
        }

        uint i0, i1, i2;
        if (!is_down) {
            i0 = row_start[row] + col;
            i1 = row_start[row] + col + 1u;
            i2 = row_start[row + 1u] + col;
        } else {
            i0 = row_start[row] + col + 1u;
            i1 = row_start[row + 1u] + col + 1u;
            i2 = row_start[row + 1u] + col;
        }
        return uvec3(i0, i1, i2);
    } else {
        // Level 3: row_start for 9 rows
        uint row_start[9] = uint[9](0u, 9u, 17u, 24u, 30u, 35u, 39u, 42u, 44u);
        uint t = tri_idx;
        uint row = 0u, col = 0u;
        bool is_down = false;

        // Row 0: 8 up + 7 down = 15, Row 1: 7 up + 6 down = 13, etc.
        if (t < 15u) {
            row = 0u;
            if (t < 8u) { col = t; is_down = false; }
            else { col = t - 8u; is_down = true; }
        } else if (t < 28u) {
            row = 1u; uint lt = t - 15u;
            if (lt < 7u) { col = lt; is_down = false; }
            else { col = lt - 7u; is_down = true; }
        } else if (t < 39u) {
            row = 2u; uint lt = t - 28u;
            if (lt < 6u) { col = lt; is_down = false; }
            else { col = lt - 6u; is_down = true; }
        } else if (t < 48u) {
            row = 3u; uint lt = t - 39u;
            if (lt < 5u) { col = lt; is_down = false; }
            else { col = lt - 5u; is_down = true; }
        } else if (t < 55u) {
            row = 4u; uint lt = t - 48u;
            if (lt < 4u) { col = lt; is_down = false; }
            else { col = lt - 4u; is_down = true; }
        } else if (t < 60u) {
            row = 5u; uint lt = t - 55u;
            if (lt < 3u) { col = lt; is_down = false; }
            else { col = lt - 3u; is_down = true; }
        } else if (t < 63u) {
            row = 6u; uint lt = t - 60u;
            if (lt < 2u) { col = lt; is_down = false; }
            else { col = lt - 2u; is_down = true; }
        } else {
            row = 7u; col = 0u; is_down = false;
        }

        uint i0, i1, i2;
        if (!is_down) {
            i0 = row_start[row] + col;
            i1 = row_start[row] + col + 1u;
            i2 = row_start[row + 1u] + col;
        } else {
            i0 = row_start[row] + col + 1u;
            i1 = row_start[row + 1u] + col + 1u;
            i2 = row_start[row + 1u] + col;
        }
        return uvec3(i0, i1, i2);
    }
}
)";

//=============================================================================
// Timestamped Polyline Task Shader
// Mirrors the non-timestamped version: each workgroup handles ONE polyline
// Dispatch: num_queries * num_pools * max_varrays_per_pool
// Each task shader decodes (query, pool, varray_offset), handles one polyline
//=============================================================================

static const char* LUDUS_TIMESTAMPED_POLYLINE_TASK_SHADER = R"(
layout(local_size_x = 1) in;

// Uniforms for dispatch decoding
uniform uint u_num_polyline_pools;
uniform uint u_max_varrays_per_pool;  // Upper bound on varrays per pool per timestamp

// Task payload to mesh shader - matches non-timestamped version
taskNV out PolylineTaskPayload {
    uint query_id;
    uint pool_id;
    uint varray_idx;            // Actual varray index (global in pool)
    uint total_points;          // Number of vertices in this polyline
    float width;
    vec3 color;
    uint cap_style;
    float is_bev;               // 1.0 for BEV cameras, 0.0 otherwise
};

void main() {
    uint work_id = gl_WorkGroupID.x;

    // Decode work_id into (query, pool, varray_offset)
    uint varray_offset = work_id % u_max_varrays_per_pool;
    uint temp = work_id / u_max_varrays_per_pool;
    uint pool_id_local = temp % u_num_polyline_pools;
    uint query_id_local = temp / u_num_polyline_pools;

    if (query_id_local >= u_num_queries) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Load query
    RenderQuery query = g_queries[query_id_local];
    TimestampedScene scene = g_scenes[query.scene_id];

    if (scene.valid == 0u || pool_id_local >= scene.num_polyline_pools) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Load pool
    TimestampedPolylinePool pool = g_polyline_pools[scene.polyline_pools_offset + pool_id_local];

    // Binary search for timestamp
    int ts_idx = binary_search_timestamps(
        scene.timestamps_buffer_offset + pool.timestamps_offset,
        pool.num_timestamps,
        query.timestamp_us
    );

    if (ts_idx < 0) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Compute varray range for this timestamp
    uint varray_start = 0u;
    if (ts_idx > 0) {
        varray_start = uint(g_int32[scene.int32_buffer_offset + pool.ts_varrays_ps_offset + uint(ts_idx) - 1u]);
    }
    uint varray_end = uint(g_int32[scene.int32_buffer_offset + pool.ts_varrays_ps_offset + uint(ts_idx)]);
    uint num_varrays = varray_end - varray_start;

    // Check if this varray_offset is valid for this timestamp
    if (varray_offset >= num_varrays) {
        gl_TaskCountNV = 0u;
        return;
    }

    uint actual_varray_idx = varray_start + varray_offset;

    // Spatial culling: test per-element AABB against camera-centred view volume
    if (u_cull_radius_scale > 0.0 && pool.aabb_offset != 0u) {
        float cull_r = g_camera_intrinsics[query.camera_id].depth_max * u_cull_radius_scale;
        CameraPose cull_pose = g_camera_poses[query_id_local];
        vec3 cam_pos = get_camera_world_pos(cull_pose);
        vec3 view_min = cam_pos - vec3(cull_r);
        vec3 view_max = cam_pos + vec3(cull_r);
        uint ab = scene.float_buffer_offset + pool.aabb_offset + actual_varray_idx * 6u;
        vec3 e_min = vec3(g_floats[ab], g_floats[ab+1u], g_floats[ab+2u]);
        vec3 e_max = vec3(g_floats[ab+3u], g_floats[ab+4u], g_floats[ab+5u]);
        if (!cull_aabb_overlap(e_min, e_max, view_min, view_max)) {
            gl_TaskCountNV = 0u;
            return;
        }
    }

    // Get vertex range for this polyline
    uint v_start = 0u;
    if (actual_varray_idx > 0u) {
        v_start = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + actual_varray_idx - 1u]);
    }
    uint v_end = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + actual_varray_idx]);
    uint total_pts = v_end - v_start;

    // Handle dot primitives differently - each vertex is a separate circle
    bool is_dot = is_dot_primitive(pool.prim_type_id);

    if (is_dot) {
        // For dots: each vertex is a dot, dispatch one mesh task per batch of dots
        // Each dot uses 9 vertices (center + 8 ring) and 8 triangles
        // With max 64 verts, we can do 7 dots per mesh (7*9=63)
        const uint DOTS_PER_CHUNK = 7u;
        uint num_chunks = (total_pts + DOTS_PER_CHUNK - 1u) / DOTS_PER_CHUNK;
        num_chunks = max(num_chunks, 1u);

        gl_TaskCountNV = num_chunks;
    } else {
        // Regular polylines
        if (total_pts < 2u) {
            gl_TaskCountNV = 0u;
            return;
        }

        // Per-segment adaptive subdivision with 128 vertex limit (per-primitive color)
        // Max capacity: 128 verts - 8 cap verts = 120 body verts = 60 effective points
        // Worst case 16x subdivision (level 4): 60 / 16 = 3.75 segments
        // Use 3 segments per chunk for safety margin
        const uint MAX_SEGMENTS_PER_CHUNK = 3u;

        uint num_segments = total_pts - 1u;
        uint num_chunks = (num_segments + MAX_SEGMENTS_PER_CHUNK - 1u) / MAX_SEGMENTS_PER_CHUNK;
        num_chunks = max(num_chunks, 1u);

        // Emit mesh tasks for each chunk
        gl_TaskCountNV = num_chunks;
    }

    // Write task payload - readable by all spawned mesh shaders
    query_id = query_id_local;
    pool_id = pool_id_local;
    varray_idx = actual_varray_idx;
    total_points = total_pts;
    width = get_prim_width(pool.prim_type_id, query.camera_type_id);
    color = get_prim_color(pool.prim_type_id);
    cap_style = CAP_ROUND;
    is_bev = (query.camera_type_id == CAMERA_TYPE_BEV) ? 1.0 : 0.0;
}
)";

//=============================================================================
// Timestamped Polyline Mesh Shader with Tessellation and Chunking
// Mirrors the non-timestamped version exactly
//=============================================================================

static const char* LUDUS_TIMESTAMPED_POLYLINE_MESH_SHADER = R"(
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 128, max_primitives = 126) out;

// Task payload from task shader
taskNV in PolylineTaskPayload {
    uint query_id;
    uint pool_id;
    uint varray_idx;
    uint total_points;
    float width;
    vec3 color;
    uint cap_style;
    float is_bev;
};

// Per-primitive color in a named block with explicit location
layout(location = 0) perprimitiveNV out PrimColorBlock {
    vec3 color;
    float is_bev;  // 1.0 for BEV cameras (disables fog), 0.0 otherwise
} prim_out[];

// Shared memory for subdivided projected points
// With 128 max vertices, we can have up to 64 effective points (2 verts each)
// Reserve indices 62, 63 for overlap vertices
shared vec4 s_clip_pos[64];
shared vec2 s_screen_pos[64];
shared vec3 s_world_pos[64];

// Shared memory indices for special values
const uint OVERLAP_BEFORE_IDX = 62u;  // Overlap vertex before chunk
const uint OVERLAP_AFTER_IDX = 63u;   // Overlap vertex after chunk
const uint COUNTER_IDX = 61u;         // Counter and flags storage

void main() {
    // Get chunk ID from mesh workgroup index
    uint chunk_id = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    // Read from task payload
    uint total_pts = total_points;

    // Load scene and pool info
    RenderQuery query = g_queries[query_id];
    TimestampedScene scene = g_scenes[query.scene_id];
    TimestampedPolylinePool pool = g_polyline_pools[scene.polyline_pools_offset + pool_id];
    FThetaCamera cam = g_camera_intrinsics[query.camera_id];
    CameraPose pose = g_camera_poses[query_id];
    uint cap_style_local = cap_style;

    // Handle dot primitives - render each vertex as a circle
    if (is_dot_primitive(pool.prim_type_id)) {
        // Dot mode: each vertex is a separate circle
        const uint DOTS_PER_CHUNK = 7u;
        uint dot_start = chunk_id * DOTS_PER_CHUNK;
        uint dot_end = min(dot_start + DOTS_PER_CHUNK, total_pts);
        uint num_dots = dot_end - dot_start;

        if (num_dots == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        // Get vertex base for this polyline
        uint v_start_base = 0u;
        if (varray_idx > 0u) {
            v_start_base = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + varray_idx - 1u]);
        }
        uint base_v_idx = scene.vertex_buffer_offset + pool.vertices_offset + v_start_base;

        // Dot radius = same as regular line half-width
        float half_width = width * 0.5;

        // Each dot: 9 vertices (center + 8 ring), 8 triangles
        uint total_verts = num_dots * 9u;
        uint total_tris = num_dots * 8u;

        gl_PrimitiveCountNV = total_tris;

        // Generate dots (only thread 0 for simplicity, could parallelize)
        if (tid == 0u) {
            for (uint d = 0u; d < num_dots; d++) {
                uint global_dot = dot_start + d;
                uint vi = base_v_idx + global_dot;
                Vertex vx = g_vertices[vi];
                vec3 world_pos = vec3(vx.x, vx.y, vx.z);

                // Project center
                vec4 center_clip = ftheta_project(world_pos, pose, cam);
                float w = center_clip.w;
                if (abs(w) < 1e-6) w = 1e-6;
                vec2 center_screen = vec2(
                    (center_clip.x / w * 0.5 + 0.5) * cam.img_w,
                    (0.5 - center_clip.y / w * 0.5) * cam.img_h
                );

                // Emit center vertex
                uint base_vert = d * 9u;
                gl_MeshVerticesNV[base_vert].gl_Position = center_clip;

                // Apply depth-based radius scaling for dots
                float dot_depth_scale = get_depth_scale(center_clip);
                float scaled_half_width = half_width * dot_depth_scale;

                // Emit 8 ring vertices (octagon for circle approximation)
                for (uint i = 0u; i < 8u; i++) {
                    float angle = float(i) * 0.785398;  // 2*PI/8
                    vec2 offset = vec2(cos(angle), sin(angle)) * scaled_half_width;
                    vec2 ring_screen = center_screen + offset;

                    // Convert back to clip space
                    vec4 ring_clip;
                    ring_clip.x = ((ring_screen.x / cam.img_w) * 2.0 - 1.0) * w;
                    ring_clip.y = ((0.5 - ring_screen.y / cam.img_h) * 2.0) * w;
                    ring_clip.z = center_clip.z;
                    ring_clip.w = w;

                    gl_MeshVerticesNV[base_vert + 1u + i].gl_Position = ring_clip;
                }

                // Emit 8 triangles (center -> ring[i] -> ring[i+1])
                uint base_tri = d * 8u;
                for (uint i = 0u; i < 8u; i++) {
                    uint next_i = (i + 1u) % 8u;
                    gl_PrimitiveIndicesNV[base_tri * 3u + i * 3u + 0u] = base_vert;
                    gl_PrimitiveIndicesNV[base_tri * 3u + i * 3u + 1u] = base_vert + 1u + i;
                    gl_PrimitiveIndicesNV[base_tri * 3u + i * 3u + 2u] = base_vert + 1u + next_i;
                    gl_MeshPrimitivesNV[base_tri + i].gl_Layer = int(query_id);
                    prim_out[base_tri + i].color = color;
                    prim_out[base_tri + i].is_bev = is_bev;
                }
            }
        }
        return;
    }

    // Regular polyline mode
    // Calculate chunk range in original segment space
    // 4 segments per chunk with overlap vertices for seamless miter joins (matching task shader)
    const uint MAX_SEGMENTS_PER_CHUNK = 3u;
    uint num_segments = total_pts - 1u;
    uint seg_start = chunk_id * MAX_SEGMENTS_PER_CHUNK;
    uint seg_end = min(seg_start + MAX_SEGMENTS_PER_CHUNK, num_segments);
    uint num_segs_in_chunk = seg_end - seg_start;

    bool is_first_chunk = (chunk_id == 0u);
    bool is_last_chunk = (seg_end >= num_segments);

    if (num_segs_in_chunk == 0u) {
        gl_PrimitiveCountNV = 0u;
        return;
    }

    float half_width = width * 0.5;

    // Get vertex start for this polyline
    uint v_start = 0u;
    if (varray_idx > 0u) {
        v_start = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + varray_idx - 1u]);
    }
    uint base_v_idx = scene.vertex_buffer_offset + pool.vertices_offset + v_start;

    // Phase 1: Compute subdivision per segment and generate subdivided points
    // Also load overlap vertices for chunk boundary miter calculation
    uint eff_point_count = 0u;
    bool has_overlap_before = false;
    bool has_overlap_after = false;

    if (tid == 0u) {
        // Load overlap vertex BEFORE this chunk (for first point miter)
        if (!is_first_chunk && seg_start > 0u) {
            uint vi_before = base_v_idx + seg_start - 1u;
            Vertex vx_before = g_vertices[vi_before];
            s_world_pos[OVERLAP_BEFORE_IDX] = vec3(vx_before.x, vx_before.y, vx_before.z);
            has_overlap_before = true;
        }

        // Load overlap vertex AFTER this chunk (for last point miter)
        if (!is_last_chunk && seg_end < num_segments) {
            uint vi_after = base_v_idx + seg_end + 1u;
            Vertex vx_after = g_vertices[vi_after];
            s_world_pos[OVERLAP_AFTER_IDX] = vec3(vx_after.x, vx_after.y, vx_after.z);
            has_overlap_after = true;
        }

        // Generate body points for this chunk's segments
        for (uint seg_idx = 0u; seg_idx < num_segs_in_chunk; seg_idx++) {
            uint global_seg = seg_start + seg_idx;
            uint vi0 = base_v_idx + global_seg;
            uint vi1 = base_v_idx + global_seg + 1u;
            Vertex vx0 = g_vertices[vi0];
            Vertex vx1 = g_vertices[vi1];
            vec3 p0 = vec3(vx0.x, vx0.y, vx0.z);
            vec3 p1 = vec3(vx1.x, vx1.y, vx1.z);

            // Compute distortion error for this segment
            float error = estimate_edge_distortion_pixels_mat4(p0, p1, pose.world_to_camera, cam);

            // Determine subdivision level based on error threshold
            // Max level 4 = 16 subsegments, 3 segs * 16 = 48 subsegments = 49 points
            uint subdiv_level = 0u;
            float tess_thresh = u_tessellation_threshold;
            if (tess_thresh > 0.0 && error > tess_thresh) subdiv_level = 1u;
            if (tess_thresh > 0.0 && error > tess_thresh * 4.0) subdiv_level = 2u;
            if (tess_thresh > 0.0 && error > tess_thresh * 16.0) subdiv_level = 3u;
            if (tess_thresh > 0.0 && error > tess_thresh * 64.0) subdiv_level = 4u;
            subdiv_level = min(subdiv_level, u_max_tessellation_polyline);

            uint num_subsegments = 1u << subdiv_level;

            // First point: only if this is the first segment in chunk
            if (seg_idx == 0u) {
                s_world_pos[eff_point_count] = p0;
                eff_point_count++;
            }

            // Interior and end points
            for (uint sub_i = 1u; sub_i <= num_subsegments; sub_i++) {
                float t = float(sub_i) / float(num_subsegments);
                vec3 world_pos = mix(p0, p1, t);
                s_world_pos[eff_point_count] = world_pos;
                eff_point_count++;

                if (eff_point_count >= 50u) break;  // 49 segments max = 98 verts + caps < 128
            }

            if (eff_point_count >= 50u) break;
        }

        // Store counts and flags via shared memory at COUNTER_IDX
        s_clip_pos[COUNTER_IDX].x = float(eff_point_count);
        s_clip_pos[COUNTER_IDX].y = has_overlap_before ? 1.0 : 0.0;
        s_clip_pos[COUNTER_IDX].z = has_overlap_after ? 1.0 : 0.0;
    }

    barrier();

    uint num_eff_points = uint(s_clip_pos[COUNTER_IDX].x);
    bool use_overlap_before = (s_clip_pos[COUNTER_IDX].y > 0.5);
    bool use_overlap_after = (s_clip_pos[COUNTER_IDX].z > 0.5);

    // 128 verts / 2 verts per point = 64 points max, minus caps = ~60 points
    // Use 50 for safety margin with overlap vertices
    num_eff_points = min(num_eff_points, 50u);

    if (num_eff_points < 2u) {
        gl_PrimitiveCountNV = 0u;
        return;
    }

    // Phase 2: Project all points - pass raw values to GPU, no clamping
    // Loop to handle more points than threads (33 points, 32 threads)
    for (uint pt = tid; pt < num_eff_points; pt += 32u) {
        vec3 world_pos = s_world_pos[pt];
        vec4 clip = ftheta_project(world_pos, pose, cam);
        s_clip_pos[pt] = clip;

        // Screen position for direction calculation
        float w = clip.w;
        if (abs(w) < 1e-6) w = 1e-6;  // Only prevent division by zero
        s_screen_pos[pt] = vec2(
            (clip.x / w * 0.5 + 0.5) * cam.img_w,
            (0.5 - clip.y / w * 0.5) * cam.img_h
        );
    }

    barrier();

    // SHARED MITER VERTICES: 2 vertices per point (left, right)
    // Segment i uses vertices from points i and i+1 (shared at joints)
    // No joint triangles needed since vertices are shared
    uint num_eff_segs = num_eff_points - 1u;

    // Caps: 4-triangle semicircles for round line ends and to hide chunk seams
    // Each cap: 4 new vertices (center + 3 arc points), 4 triangles
    // Draw caps at ALL chunk boundaries to hide gaps between chunks
    bool draw_start_cap = true;   // Always draw start cap (for polyline start or chunk seam)
    bool draw_end_cap = true;     // Always draw end cap (for polyline end or chunk seam)
    uint num_cap_verts = (draw_start_cap ? 4u : 0u) + (draw_end_cap ? 4u : 0u);
    uint num_cap_tris = (draw_start_cap ? 4u : 0u) + (draw_end_cap ? 4u : 0u);

    // Vertex layout:
    // [0, 2*num_eff_points-1]: point vertices (left, right per point)
    // [2*num_eff_points, ...]: cap vertices (if any)
    uint point_verts_end = num_eff_points * 2u;
    uint start_cap_base = point_verts_end;  // center, arc1, arc2, arc3
    uint end_cap_base = start_cap_base + (draw_start_cap ? 4u : 0u);

    uint num_tris = num_eff_segs * 2u + num_cap_tris;

    // With 3 segs × level 4 = 49 points = 48 segs × 2 + 8 caps = 104 tris max
    gl_PrimitiveCountNV = min(num_tris, 120u);

    // Phase 3: Generate point vertices (2 per point) WITH MITER JOINS
    // Point i: vertices [2*i, 2*i+1] = [left, right]
    // Vertices are shared between adjacent segments
    // Loop to handle more points than threads
    for (uint pt = tid; pt < num_eff_points; pt += 32u) {
        vec4 clip = s_clip_pos[pt];
        vec2 screen = s_screen_pos[pt];

        // For miter direction calculation, we need stable screen positions
        // If an adjacent point has bad w (off-screen), use binary search to find
        // a closer point with good w for direction calculation
        float w_min = 1.0;

        // Get directions to adjacent points for miter calculation in SCREEN SPACE
        vec2 dir_prev = vec2(0.0);
        vec2 dir_next = vec2(0.0);
        bool has_prev = (pt > 0u);
        bool has_next = (pt < num_eff_points - 1u);

        vec2 screen_for_dir = screen;
        vec4 clip_curr = clip;

        // If current point has bad w, clamp it for direction calculation
        if (clip_curr.w < w_min && pt > 0u && s_clip_pos[pt - 1u].w >= w_min) {
            // Binary search toward prev point to find good w
            vec3 good = s_world_pos[pt - 1u];
            vec3 bad = s_world_pos[pt];
            for (int iter = 0; iter < 5; iter++) {
                vec3 mid = (good + bad) * 0.5;
                vec4 mid_clip = ftheta_project(mid, pose, cam);
                if (mid_clip.w >= w_min) good = mid; else bad = mid;
            }
            vec4 clamped = ftheta_project(good, pose, cam);
            float w_c = max(clamped.w, 0.001);
            screen_for_dir = vec2((clamped.x / w_c * 0.5 + 0.5) * cam.img_w,
                                   (0.5 - clamped.y / w_c * 0.5) * cam.img_h);
        } else if (clip_curr.w < w_min && pt < num_eff_points - 1u && s_clip_pos[pt + 1u].w >= w_min) {
            // Binary search toward next point
            vec3 good = s_world_pos[pt + 1u];
            vec3 bad = s_world_pos[pt];
            for (int iter = 0; iter < 5; iter++) {
                vec3 mid = (good + bad) * 0.5;
                vec4 mid_clip = ftheta_project(mid, pose, cam);
                if (mid_clip.w >= w_min) good = mid; else bad = mid;
            }
            vec4 clamped = ftheta_project(good, pose, cam);
            float w_c = max(clamped.w, 0.001);
            screen_for_dir = vec2((clamped.x / w_c * 0.5 + 0.5) * cam.img_w,
                                   (0.5 - clamped.y / w_c * 0.5) * cam.img_h);
        }

        if (has_prev) {
            vec2 screen_prev = s_screen_pos[pt - 1u];
            // If prev point has bad w, use clamped position for direction
            if (s_clip_pos[pt - 1u].w < w_min && clip_curr.w >= w_min) {
                vec3 good = s_world_pos[pt];
                vec3 bad = s_world_pos[pt - 1u];
                for (int iter = 0; iter < 5; iter++) {
                    vec3 mid = (good + bad) * 0.5;
                    vec4 mid_clip = ftheta_project(mid, pose, cam);
                    if (mid_clip.w >= w_min) good = mid; else bad = mid;
                }
                vec4 clamped = ftheta_project(good, pose, cam);
                float w_c = max(clamped.w, 0.001);
                screen_prev = vec2((clamped.x / w_c * 0.5 + 0.5) * cam.img_w,
                                   (0.5 - clamped.y / w_c * 0.5) * cam.img_h);
            }
            vec2 d = screen_for_dir - screen_prev;
            float len = length(d);
            if (len > 0.001) dir_prev = d / len;
        }
        if (has_next) {
            vec2 screen_next = s_screen_pos[pt + 1u];
            // If next point has bad w, use clamped position for direction
            if (s_clip_pos[pt + 1u].w < w_min && clip_curr.w >= w_min) {
                vec3 good = s_world_pos[pt];
                vec3 bad = s_world_pos[pt + 1u];
                for (int iter = 0; iter < 5; iter++) {
                    vec3 mid = (good + bad) * 0.5;
                    vec4 mid_clip = ftheta_project(mid, pose, cam);
                    if (mid_clip.w >= w_min) good = mid; else bad = mid;
                }
                vec4 clamped = ftheta_project(good, pose, cam);
                float w_c = max(clamped.w, 0.001);
                screen_next = vec2((clamped.x / w_c * 0.5 + 0.5) * cam.img_w,
                                   (0.5 - clamped.y / w_c * 0.5) * cam.img_h);
            }
            vec2 d = screen_next - screen_for_dir;
            float len = length(d);
            if (len > 0.001) dir_next = d / len;
        }

        // Compute miter direction and scale in screen space
        vec2 miter;
        float miter_scale = 1.0;

        if (has_prev && has_next && length(dir_prev) > 0.001 && length(dir_next) > 0.001) {
            // Interior point: average perpendiculars
            vec2 perp_prev = vec2(-dir_prev.y, dir_prev.x);
            vec2 perp_next = vec2(-dir_next.y, dir_next.x);
            vec2 miter_sum = perp_prev + perp_next;
            float miter_len = length(miter_sum);
            if (miter_len > 0.001) {
                miter = miter_sum / miter_len;
                float cos_half = dot(miter, perp_next);
                miter_scale = (cos_half > 0.5) ? (1.0 / cos_half) : 2.0;
            } else {
                // Nearly 180° turn, use either perpendicular
                miter = perp_next;
            }
        } else if (has_next && length(dir_next) > 0.001) {
            // First point: use next segment's perpendicular
            miter = vec2(-dir_next.y, dir_next.x);
        } else if (has_prev && length(dir_prev) > 0.001) {
            // Last point: use prev segment's perpendicular
            miter = vec2(-dir_prev.y, dir_prev.x);
        } else {
            // Degenerate case
            miter = vec2(1.0, 0.0);
        }

        // Compute offset: screen-space miter direction, scaled to clip space
        // Apply distance-based width scaling (thinner lines in distance)
        float depth_scale = get_depth_scale(clip);
        float scaled_half_width = half_width * depth_scale;

        float off_x = miter.x * scaled_half_width * miter_scale * 2.0 / cam.img_w * clip.w;
        float off_y = -miter.y * scaled_half_width * miter_scale * 2.0 / cam.img_h * clip.w;

        uint base = pt * 2u;
        gl_MeshVerticesNV[base].gl_Position = vec4(clip.x - off_x, clip.y - off_y, clip.z, clip.w);
        gl_MeshVerticesNV[base + 1u].gl_Position = vec4(clip.x + off_x, clip.y + off_y, clip.z, clip.w);
    }

    // Generate cap vertices (first thread only)
    // Cap is a 4-triangle semicircle fan from center through arc points
    // Arc goes from right edge, around the back, to left edge
    if (tid == 0u) {
        // Start cap: at first point, semicircle facing backward (opposite to line direction)
        if (draw_start_cap && num_eff_points >= 2u) {
            vec4 clip0 = s_clip_pos[0u];
            vec2 screen0 = s_screen_pos[0u];
            vec2 screen1 = s_screen_pos[1u];

            // Line direction and perpendicular
            vec2 line_dir = normalize(screen1 - screen0);
            vec2 perp = vec2(-line_dir.y, line_dir.x);

            // Cap center is at the first point (no offset)
            gl_MeshVerticesNV[start_cap_base].gl_Position = clip0;

            // Arc points at 45°, 90°, 135° from right to left (going backward)
            // right = +perp, backward = -line_dir, left = -perp
            // 45°: mix of +perp and -line_dir
            // 90°: pure -line_dir
            // 135°: mix of -perp and -line_dir
            float angles[3] = float[3](0.7854, 1.5708, 2.3562);  // 45°, 90°, 135° in radians

            // Apply depth-based width scaling for cap
            float cap0_depth_scale = get_depth_scale(clip0);
            float cap0_scaled_half_width = half_width * cap0_depth_scale;

            for (int i = 0; i < 3; i++) {
                float a = angles[i];
                // Direction: rotate from +perp toward -line_dir
                vec2 arc_dir = cos(a) * perp + sin(a) * (-line_dir);

                float arc_off_x = arc_dir.x * cap0_scaled_half_width * 2.0 / cam.img_w * clip0.w;
                float arc_off_y = -arc_dir.y * cap0_scaled_half_width * 2.0 / cam.img_h * clip0.w;

                gl_MeshVerticesNV[start_cap_base + 1u + uint(i)].gl_Position =
                    vec4(clip0.x + arc_off_x, clip0.y + arc_off_y, clip0.z, clip0.w);
            }
        }

        // End cap: at last point, semicircle facing forward (in line direction)
        if (draw_end_cap && num_eff_points >= 2u) {
            uint last_pt = num_eff_points - 1u;
            vec4 clipN = s_clip_pos[last_pt];
            vec2 screenN = s_screen_pos[last_pt];
            vec2 screenN1 = s_screen_pos[last_pt - 1u];

            // Line direction (from second-to-last toward last)
            vec2 line_dir = normalize(screenN - screenN1);
            vec2 perp = vec2(-line_dir.y, line_dir.x);

            // Cap center is at the last point
            gl_MeshVerticesNV[end_cap_base].gl_Position = clipN;

            // Arc points at 45°, 90°, 135° from left to right (going forward)
            // left = -perp, forward = +line_dir, right = +perp
            float angles[3] = float[3](0.7854, 1.5708, 2.3562);

            // Apply depth-based width scaling for cap
            float capN_depth_scale = get_depth_scale(clipN);
            float capN_scaled_half_width = half_width * capN_depth_scale;

            for (int i = 0; i < 3; i++) {
                float a = angles[i];
                // Direction: rotate from -perp toward +line_dir
                vec2 arc_dir = cos(a) * (-perp) + sin(a) * line_dir;

                float arc_off_x = arc_dir.x * capN_scaled_half_width * 2.0 / cam.img_w * clipN.w;
                float arc_off_y = -arc_dir.y * capN_scaled_half_width * 2.0 / cam.img_h * clipN.w;

                gl_MeshVerticesNV[end_cap_base + 1u + uint(i)].gl_Position =
                    vec4(clipN.x + arc_off_x, clipN.y + arc_off_y, clipN.z, clipN.w);
            }
        }
    }

    // Phase 4: Set triangle indices (first thread only)
    // SHARED VERTICES: segment i uses vertices from point i and point i+1
    // Point i: vertices [2*i, 2*i+1] = [left, right]
    if (tid == 0u) {
        uint tri_idx = 0u;

        // Segment triangles: each segment uses 4 vertices from 2 adjacent points
        // Segment i (points i, i+1): left0=2*i, right0=2*i+1, left1=2*(i+1), right1=2*(i+1)+1
        for (uint seg = 0u; seg < num_eff_segs; seg++) {
            uint left0 = seg * 2u;
            uint right0 = seg * 2u + 1u;
            uint left1 = (seg + 1u) * 2u;
            uint right1 = (seg + 1u) * 2u + 1u;

            // Triangle 1: left0, right0, left1
            gl_PrimitiveIndicesNV[tri_idx * 3u] = left0;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = right0;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = left1;
            tri_idx++;

            // Triangle 2: right0, right1, left1
            gl_PrimitiveIndicesNV[tri_idx * 3u] = right0;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = right1;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = left1;
            tri_idx++;
        }

        // No joint triangles needed - vertices are shared!

        // Start cap triangles: fan from center through arc points
        // Vertices: right(1), arc1, arc2, arc3, left(0)
        // center = start_cap_base, arc1/2/3 = start_cap_base + 1/2/3
        if (draw_start_cap) {
            uint center = start_cap_base;
            uint right_edge = 1u;  // First point's right vertex
            uint left_edge = 0u;   // First point's left vertex
            uint arc1 = start_cap_base + 1u;
            uint arc2 = start_cap_base + 2u;
            uint arc3 = start_cap_base + 3u;

            // Triangle 1: center, right_edge, arc1
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = right_edge;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = arc1;
            tri_idx++;

            // Triangle 2: center, arc1, arc2
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = arc1;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = arc2;
            tri_idx++;

            // Triangle 3: center, arc2, arc3
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = arc2;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = arc3;
            tri_idx++;

            // Triangle 4: center, arc3, left_edge
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = arc3;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = left_edge;
            tri_idx++;
        }

        // End cap triangles: fan from center through arc points
        // Vertices: left(2*(n-1)), arc1, arc2, arc3, right(2*(n-1)+1)
        if (draw_end_cap) {
            uint last_pt = num_eff_points - 1u;
            uint center = end_cap_base;
            uint left_edge = last_pt * 2u;      // Last point's left vertex
            uint right_edge = last_pt * 2u + 1u; // Last point's right vertex
            uint arc1 = end_cap_base + 1u;
            uint arc2 = end_cap_base + 2u;
            uint arc3 = end_cap_base + 3u;

            // Triangle 1: center, left_edge, arc1
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = left_edge;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = arc1;
            tri_idx++;

            // Triangle 2: center, arc1, arc2
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = arc1;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = arc2;
            tri_idx++;

            // Triangle 3: center, arc2, arc3
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = arc2;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = arc3;
            tri_idx++;

            // Triangle 4: center, arc3, right_edge
            gl_PrimitiveIndicesNV[tri_idx * 3u] = center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = arc3;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = right_edge;
            tri_idx++;
        }

        // Set all primitives to layer matching query_id and uniform color
        for (uint i = 0u; i < gl_PrimitiveCountNV; i++) {
            gl_MeshPrimitivesNV[i].gl_Layer = int(query_id);
            prim_out[i].color = color;
            prim_out[i].is_bev = is_bev;
        }
    }
}
)";

//=============================================================================
// Fragment Shader (shared)
//=============================================================================

// Fragment shader for polylines - uses perprimitiveNV color
static const char* LUDUS_TIMESTAMPED_POLYLINE_FRAGMENT_SHADER = R"(
#version 460
#extension GL_NV_fragment_shader_barycentric : require

// Per-primitive color block from mesh shader
layout(location = 0) perprimitiveNV in PrimColorBlock {
    vec3 color;
    float is_bev;
} prim_in;

layout(location = 0) out vec4 out_color;

uniform float u_fog_enabled;  // 1.0 = enabled, 0.0 = disabled

IF_ZMODIFY(layout(location = 0) uniform float in_dummy;)

void main() {
    vec3 color = prim_in.color;

    // Apply distance-based fog (darken with distance) - disabled for BEV cameras
    // gl_FragCoord.z is in [0, 1] where 0=near, 1=far
    if (u_fog_enabled > 0.5 && prim_in.is_bev < 0.5) {
        float fog_factor = 1.0 - gl_FragCoord.z;  // 1.0 at near, 0.0 at far
        fog_factor = clamp(fog_factor, 0.0, 1.0);
        color *= fog_factor;
    }

    out_color = vec4(color, 1.0);
    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
}
)";

// Fragment shader for polygons - uses perprimitiveNV color (same as polylines)
static const char* LUDUS_TIMESTAMPED_FRAGMENT_SHADER = R"(
#version 460
#extension GL_NV_fragment_shader_barycentric : require

// Per-primitive color block from mesh shader
layout(location = 0) perprimitiveNV in PrimColorBlock {
    vec3 color;
    float is_bev;
} prim_in;

layout(location = 0) out vec4 out_color;

uniform float u_fog_enabled;  // 1.0 = enabled, 0.0 = disabled

IF_ZMODIFY(layout(location = 0) uniform float in_dummy;)

void main() {
    vec3 color = prim_in.color;

    // Apply distance-based fog (darken with distance) - disabled for BEV cameras
    // gl_FragCoord.z is in [0, 1] where 0=near, 1=far
    if (u_fog_enabled > 0.5 && prim_in.is_bev < 0.5) {
        float fog_factor = 1.0 - gl_FragCoord.z;  // 1.0 at near, 0.0 at far
        fog_factor = clamp(fog_factor, 0.0, 1.0);
        color *= fog_factor;
    }

    out_color = vec4(color, 1.0);
    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
}
)";

//=============================================================================
// Timestamped Obstacle Fragment Shader (gradient computed in fragment shader)
//=============================================================================

static const char* LUDUS_TIMESTAMPED_OBSTACLE_FRAGMENT_SHADER = R"(
#version 460
#extension GL_NV_fragment_shader_barycentric : require

// Per-primitive gradient data from mesh shader (no per-vertex outputs needed!)
layout(location = 0) perprimitiveNV in CubeGradientBlock {
    vec3 corner_t;      // gradient t values at triangle's 3 corners
    vec3 front_color;
    vec3 back_color;
    float is_bev;       // 1.0 for BEV cameras (disables fog), 0.0 otherwise
} prim_in;

layout(location = 0) out vec4 out_color;

uniform float u_fog_enabled;  // 1.0 = enabled, 0.0 = disabled

IF_ZMODIFY(layout(location = 0) uniform float in_dummy;)

void main() {
    // Interpolate gradient t using hardware barycentric coordinates
    float t = dot(prim_in.corner_t, gl_BaryCoordNV);

    // Compute gradient color
    vec3 color = mix(prim_in.back_color, prim_in.front_color, t);

    // Apply distance-based fog (darken with distance) - disabled for BEV cameras
    if (u_fog_enabled > 0.5 && prim_in.is_bev < 0.5) {
        float fog_factor = 1.0 - gl_FragCoord.z;
        fog_factor = clamp(fog_factor, 0.0, 1.0);
        color *= fog_factor;
    }

    out_color = vec4(color, 1.0);
    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
}
)";

//=============================================================================
// Timestamped Polygon Task/Mesh Shaders (similar structure to polyline)
//=============================================================================

static const char* LUDUS_TIMESTAMPED_POLYGON_TASK_SHADER = R"(
layout(local_size_x = 1) in;

// Uniforms for dispatch decoding (same pattern as polylines)
uniform uint u_num_polygon_pools;
uniform uint u_max_varrays_per_pool;

// Task payload - matches non-timestamped version
taskNV out PolygonTaskPayload {
    uint query_id;
    uint pool_id;
    uint varray_idx;
    uint triangle_count;
    uint vertex_count;
    uint subdivision_level;
    vec3 color;
    float is_bev;
};

void main() {
    uint work_id = gl_WorkGroupID.x;

    // Decode work_id into (query, pool, varray_offset)
    uint varray_offset = work_id % u_max_varrays_per_pool;
    uint temp = work_id / u_max_varrays_per_pool;
    uint pool_id_local = temp % u_num_polygon_pools;
    uint query_id_local = temp / u_num_polygon_pools;

    if (query_id_local >= u_num_queries) {
        gl_TaskCountNV = 0u;
        return;
    }

    RenderQuery query = g_queries[query_id_local];
    TimestampedScene scene = g_scenes[query.scene_id];

    if (scene.valid == 0u || pool_id_local >= scene.num_polygon_pools) {
        gl_TaskCountNV = 0u;
        return;
    }

    TimestampedPolygonPool pool = g_polygon_pools[scene.polygon_pools_offset + pool_id_local];

    int ts_idx = binary_search_timestamps(
        scene.timestamps_buffer_offset + pool.timestamps_offset,
        pool.num_timestamps,
        query.timestamp_us
    );

    if (ts_idx < 0) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Get varray range for this timestamp
    uint varray_start = 0u;
    if (ts_idx > 0) {
        varray_start = uint(g_int32[scene.int32_buffer_offset + pool.ts_varrays_ps_offset + uint(ts_idx) - 1u]);
    }
    uint varray_end = uint(g_int32[scene.int32_buffer_offset + pool.ts_varrays_ps_offset + uint(ts_idx)]);
    uint num_varrays = varray_end - varray_start;

    if (varray_offset >= num_varrays) {
        gl_TaskCountNV = 0u;
        return;
    }

    uint actual_varray_idx = varray_start + varray_offset;

    // Spatial culling: test per-element AABB against camera-centred view volume
    if (u_cull_radius_scale > 0.0 && pool.aabb_offset != 0u) {
        float cull_r = g_camera_intrinsics[query.camera_id].depth_max * u_cull_radius_scale;
        CameraPose cull_pose = g_camera_poses[query_id_local];
        vec3 cam_pos = get_camera_world_pos(cull_pose);
        vec3 view_min = cam_pos - vec3(cull_r);
        vec3 view_max = cam_pos + vec3(cull_r);
        uint ab = scene.float_buffer_offset + pool.aabb_offset + actual_varray_idx * 6u;
        vec3 e_min = vec3(g_floats[ab], g_floats[ab+1u], g_floats[ab+2u]);
        vec3 e_max = vec3(g_floats[ab+3u], g_floats[ab+4u], g_floats[ab+5u]);
        if (!cull_aabb_overlap(e_min, e_max, view_min, view_max)) {
            gl_TaskCountNV = 0u;
            return;
        }
    }

    // Get triangle range for this polygon
    uint tri_start = 0u;
    if (actual_varray_idx > 0u) {
        tri_start = uint(g_int32[scene.int32_buffer_offset + pool.tri_ps_offset + actual_varray_idx - 1u]);
    }
    uint tri_end = uint(g_int32[scene.int32_buffer_offset + pool.tri_ps_offset + actual_varray_idx]);
    uint total_tris = tri_end - tri_start;

    if (total_tris == 0u) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Get vertex range
    uint v_start = 0u;
    if (actual_varray_idx > 0u) {
        v_start = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + actual_varray_idx - 1u]);
    }
    uint v_end = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + actual_varray_idx]);
    uint total_verts = v_end - v_start;

    // Compute max subdivision level (sample first few triangles)
    FThetaCamera cam = g_camera_intrinsics[query.camera_id];
    CameraPose pose = g_camera_poses[query_id_local];

    uint max_subdiv = 0u;
    if (u_tessellation_threshold > 0.0) {
        uint base_v_idx = scene.vertex_buffer_offset + pool.vertices_offset + v_start;
        uint base_t_idx = scene.triangle_buffer_offset + pool.triangles_offset + tri_start;
        uint sample_count = min(total_tris, 8u);

        for (uint t = 0u; t < sample_count; t++) {
            uvec3 tri = g_triangles[base_t_idx + t];
            Vertex v0 = g_vertices[base_v_idx + tri.x];
            Vertex v1 = g_vertices[base_v_idx + tri.y];
            Vertex v2 = g_vertices[base_v_idx + tri.z];

            vec3 p0 = vec3(v0.x, v0.y, v0.z);
            vec3 p1 = vec3(v1.x, v1.y, v1.z);
            vec3 p2 = vec3(v2.x, v2.y, v2.z);

            max_subdiv = max(max_subdiv, compute_subdivision_level(p0, p1, pose, cam, u_tessellation_threshold));
            max_subdiv = max(max_subdiv, compute_subdivision_level(p1, p2, pose, cam, u_tessellation_threshold));
            max_subdiv = max(max_subdiv, compute_subdivision_level(p2, p0, pose, cam, u_tessellation_threshold));
        }
        max_subdiv = min(max_subdiv, u_max_tessellation_polygon);
    }

    // Compute chunks based on subdivision level (must match mesh shader constants)
    const uint MAX_SHARED_VERTS = 30u;
    uint num_chunks;
    uint tris_per_chunk;

    if (max_subdiv == 0u && total_verts <= MAX_SHARED_VERTS) {
        num_chunks = 1u;
    } else {
        if (max_subdiv == 0u) tris_per_chunk = 8u;
        else if (max_subdiv == 1u) tris_per_chunk = 5u;
        else if (max_subdiv == 2u) tris_per_chunk = 2u;
        else tris_per_chunk = 1u;  // Level 3: 1 tri per chunk

        num_chunks = (total_tris + tris_per_chunk - 1u) / tris_per_chunk;
        num_chunks = max(num_chunks, 1u);
    }

    gl_TaskCountNV = num_chunks;

    query_id = query_id_local;
    pool_id = pool_id_local;
    varray_idx = actual_varray_idx;
    triangle_count = total_tris;
    vertex_count = total_verts;
    subdivision_level = max_subdiv;
    color = get_prim_color(pool.prim_type_id);
    is_bev = (query.camera_type_id == CAMERA_TYPE_BEV) ? 1.0 : 0.0;
}
)";

static const char* LUDUS_TIMESTAMPED_POLYGON_MESH_SHADER = R"(
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 64) out;

taskNV in PolygonTaskPayload {
    uint query_id;
    uint pool_id;
    uint varray_idx;
    uint triangle_count;
    uint vertex_count;
    uint subdivision_level;
    vec3 color;
    float is_bev;
};

// Per-primitive color (same as polylines) - no per-vertex array needed
layout(location = 0) perprimitiveNV out PrimColorBlock {
    vec3 color;
    float is_bev;  // 1.0 for BEV cameras (disables fog), 0.0 otherwise
} prim_out[];

void main() {
    uint chunk_id = gl_WorkGroupID.x;
    uint subdiv = subdivision_level;
    uint tid = gl_LocalInvocationID.x;

    RenderQuery query = g_queries[query_id];
    TimestampedScene scene = g_scenes[query.scene_id];
    TimestampedPolygonPool pool = g_polygon_pools[scene.polygon_pools_offset + pool_id];

    FThetaCamera cam = g_camera_intrinsics[query.camera_id];
    CameraPose pose = g_camera_poses[query_id];

    // Get triangle/vertex ranges for this polygon
    uint tri_start = 0u;
    if (varray_idx > 0u) {
        tri_start = uint(g_int32[scene.int32_buffer_offset + pool.tri_ps_offset + varray_idx - 1u]);
    }
    uint tri_end = uint(g_int32[scene.int32_buffer_offset + pool.tri_ps_offset + varray_idx]);
    uint total_tris = tri_end - tri_start;

    uint v_start = 0u;
    if (varray_idx > 0u) {
        v_start = uint(g_int32[scene.int32_buffer_offset + pool.varrays_ps_offset + varray_idx - 1u]);
    }

    uint base_v_idx = scene.vertex_buffer_offset + pool.vertices_offset + v_start;
    uint base_t_idx = scene.triangle_buffer_offset + pool.triangles_offset + tri_start;

    const uint CHUNK_SIZE = 8u;       // Conservative chunking
    const uint MAX_SHARED_VERTS = 30u;  // Match original

    if (subdiv == 0u && vertex_count <= MAX_SHARED_VERTS && chunk_id == 0u) {
        // ===== NO SUBDIVISION, SHARED VERTEX MODE (small polygons) =====
        uint num_verts = min(vertex_count, 30u);
        uint num_tris = min(total_tris, 28u);

        gl_PrimitiveCountNV = num_tris;

        if (tid < num_verts) {
            Vertex vtx = g_vertices[base_v_idx + tid];
            vec3 world_pos = vec3(vtx.x, vtx.y, vtx.z);

            vec4 clip = ftheta_project(world_pos, pose, cam);
            gl_MeshVerticesNV[tid].gl_Position = clip;
        }

        barrier();

        if (tid < num_tris) {
            uvec3 tri = g_triangles[base_t_idx + tid];

            uint base = tid * 3u;
            gl_PrimitiveIndicesNV[base] = tri.x;
            gl_PrimitiveIndicesNV[base + 1u] = tri.y;
            gl_PrimitiveIndicesNV[base + 2u] = tri.z;

            gl_MeshPrimitivesNV[tid].gl_Layer = int(query_id);
            prim_out[tid].color = color;
            prim_out[tid].is_bev = is_bev;
        }
    } else if (subdiv == 1u) {
        // ===== LEVEL 1 SUBDIVISION: 4 sub-triangles per triangle =====
        uint verts_per_tri = bary_vertex_count(1u);  // 6
        uint tris_per_tri = bary_triangle_count(1u);  // 4

        uint tris_per_chunk = 5u;  // Original value
        uint chunk_start = chunk_id * tris_per_chunk;
        uint chunk_end = min(chunk_start + tris_per_chunk, total_tris);
        uint num_orig_tris = chunk_end - chunk_start;

        if (num_orig_tris == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        uint num_out_tris = num_orig_tris * tris_per_tri;
        gl_PrimitiveCountNV = num_out_tris;

        if (tid < num_orig_tris) {
            uvec3 tri = g_triangles[base_t_idx + chunk_start + tid];

            Vertex v0_data = g_vertices[base_v_idx + tri.x];
            Vertex v1_data = g_vertices[base_v_idx + tri.y];
            Vertex v2_data = g_vertices[base_v_idx + tri.z];

            vec3 v0 = vec3(v0_data.x, v0_data.y, v0_data.z);
            vec3 v1 = vec3(v1_data.x, v1_data.y, v1_data.z);
            vec3 v2 = vec3(v2_data.x, v2_data.y, v2_data.z);

            uint vert_base = tid * verts_per_tri;
            for (uint i = 0u; i < verts_per_tri; i++) {
                vec2 uv = bary_vertex_uv(i, 1u);
                vec3 pt = bary_interpolate(v0, v1, v2, uv);

                vec4 clip = ftheta_project(pt, pose, cam);
                gl_MeshVerticesNV[vert_base + i].gl_Position = clip;
            }
        }

        barrier();

        if (tid < num_orig_tris) {
            uint vert_base = tid * verts_per_tri;
            uint tri_base = tid * tris_per_tri;

            for (uint t = 0u; t < tris_per_tri; t++) {
                uvec3 idx = bary_triangle_indices(t, 1u);
                uint idx_base = (tri_base + t) * 3u;
                gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + idx.x;
                gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + idx.y;
                gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + idx.z;
                gl_MeshPrimitivesNV[tri_base + t].gl_Layer = int(query_id);
                prim_out[tri_base + t].color = color;
                prim_out[tri_base + t].is_bev = is_bev;
            }
        }
    } else if (subdiv == 2u) {
        // ===== LEVEL 2 SUBDIVISION: 16 sub-triangles per triangle =====
        const uint TRIS_PER_CHUNK = 2u;  // Original value
        uint chunk_start = chunk_id * TRIS_PER_CHUNK;
        uint chunk_end = min(chunk_start + TRIS_PER_CHUNK, total_tris);
        uint num_orig_tris = chunk_end - chunk_start;

        if (num_orig_tris == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        gl_PrimitiveCountNV = num_orig_tris * 16u;

        uint tri_local = tid / 16u;
        uint vert_local = tid % 16u;

        if (tri_local < num_orig_tris && vert_local < 15u) {
            uvec3 tri = g_triangles[base_t_idx + chunk_start + tri_local];

            Vertex v0_data = g_vertices[base_v_idx + tri.x];
            Vertex v1_data = g_vertices[base_v_idx + tri.y];
            Vertex v2_data = g_vertices[base_v_idx + tri.z];

            vec3 v0 = vec3(v0_data.x, v0_data.y, v0_data.z);
            vec3 v1 = vec3(v1_data.x, v1_data.y, v1_data.z);
            vec3 v2 = vec3(v2_data.x, v2_data.y, v2_data.z);

            vec2 uv = bary_vertex_uv(vert_local, 2u);
            vec3 pt = bary_interpolate(v0, v1, v2, uv);

            vec4 clip = ftheta_project(pt, pose, cam);
            uint vert_idx = tri_local * 15u + vert_local;
            gl_MeshVerticesNV[vert_idx].gl_Position = clip;
        }

        barrier();

        if (tri_local < num_orig_tris && vert_local < 16u) {
            uint vert_base = tri_local * 15u;
            uint tri_out_idx = tri_local * 16u + vert_local;

            uvec3 idx = bary_triangle_indices(vert_local, 2u);

            uint idx_base = tri_out_idx * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + idx.x;
            gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + idx.y;
            gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + idx.z;
            gl_MeshPrimitivesNV[tri_out_idx].gl_Layer = int(query_id);
            prim_out[tri_out_idx].color = color;
            prim_out[tri_out_idx].is_bev = is_bev;
        }
    } else if (subdiv == 3u) {
        // ===== LEVEL 3 SUBDIVISION: 64 sub-triangles per triangle =====
        // 1 original triangle per chunk (45 verts, 64 output tris)
        const uint TRIS_PER_CHUNK = 1u;
        uint chunk_start = chunk_id * TRIS_PER_CHUNK;

        if (chunk_start >= total_tris) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        gl_PrimitiveCountNV = 64u;

        // Load the single triangle for this chunk
        uvec3 tri = g_triangles[base_t_idx + chunk_start];
        Vertex v0_data = g_vertices[base_v_idx + tri.x];
        Vertex v1_data = g_vertices[base_v_idx + tri.y];
        Vertex v2_data = g_vertices[base_v_idx + tri.z];
        vec3 v0 = vec3(v0_data.x, v0_data.y, v0_data.z);
        vec3 v1 = vec3(v1_data.x, v1_data.y, v1_data.z);
        vec3 v2 = vec3(v2_data.x, v2_data.y, v2_data.z);

        // Each thread handles ~2 vertices (45 verts / 32 threads)
        for (uint v = tid; v < 45u; v += 32u) {
            vec2 uv = bary_vertex_uv(v, 3u);
            vec3 pt = bary_interpolate(v0, v1, v2, uv);
            vec4 clip = ftheta_project(pt, pose, cam);
            gl_MeshVerticesNV[v].gl_Position = clip;
        }

        barrier();

        // Each thread handles 2 triangles (64 tris / 32 threads)
        for (uint t = tid; t < 64u; t += 32u) {
            uvec3 idx = bary_triangle_indices(t, 3u);
            uint idx_base = t * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = idx.x;
            gl_PrimitiveIndicesNV[idx_base + 1u] = idx.y;
            gl_PrimitiveIndicesNV[idx_base + 2u] = idx.z;
            gl_MeshPrimitivesNV[t].gl_Layer = int(query_id);
            prim_out[t].color = color;
            prim_out[t].is_bev = is_bev;
        }
    } else {
        // ===== LARGE POLYGON WITHOUT SUBDIVISION =====
        uint chunk_start = chunk_id * CHUNK_SIZE;
        uint chunk_end = min(chunk_start + CHUNK_SIZE, total_tris);
        uint num_orig_tris = chunk_end - chunk_start;

        if (num_orig_tris == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        gl_PrimitiveCountNV = num_orig_tris;

        if (tid < num_orig_tris) {
            uvec3 tri = g_triangles[base_t_idx + chunk_start + tid];

            Vertex v0_data = g_vertices[base_v_idx + tri.x];
            Vertex v1_data = g_vertices[base_v_idx + tri.y];
            Vertex v2_data = g_vertices[base_v_idx + tri.z];

            vec3 v0 = vec3(v0_data.x, v0_data.y, v0_data.z);
            vec3 v1 = vec3(v1_data.x, v1_data.y, v1_data.z);
            vec3 v2 = vec3(v2_data.x, v2_data.y, v2_data.z);

            uint vert_base = tid * 3u;

            vec4 clip0 = ftheta_project(v0, pose, cam);
            vec4 clip1 = ftheta_project(v1, pose, cam);
            vec4 clip2 = ftheta_project(v2, pose, cam);
            gl_MeshVerticesNV[vert_base + 0u].gl_Position = clip0;
            gl_MeshVerticesNV[vert_base + 1u].gl_Position = clip1;
            gl_MeshVerticesNV[vert_base + 2u].gl_Position = clip2;

            uint idx_base = tid * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + 0u;
            gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + 1u;
            gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + 2u;
            gl_MeshPrimitivesNV[tid].gl_Layer = int(query_id);
            prim_out[tid].color = color;
            prim_out[tid].is_bev = is_bev;
        }
    }
}
)";

//=============================================================================
// Timestamped Obstacle Task/Mesh Shaders
//=============================================================================

static const char* LUDUS_TIMESTAMPED_OBSTACLE_TASK_SHADER = R"(
layout(local_size_x = 1) in;

// Uniforms for obstacle/cube dispatch
uniform uint u_max_obstacles;
uniform uint u_cube_pool_index;  // Which cube pool to render (0-based)

// Unit cube vertices for edge distortion calculation (same as non-timestamped)
const vec3 CUBE_VERTS[8] = vec3[8](
    vec3(-0.5, -0.5, -0.5), vec3(+0.5, -0.5, -0.5),
    vec3(+0.5, +0.5, -0.5), vec3(-0.5, +0.5, -0.5),
    vec3(-0.5, -0.5, +0.5), vec3(+0.5, -0.5, +0.5),
    vec3(+0.5, +0.5, +0.5), vec3(-0.5, +0.5, +0.5)
);

// 12 edges of cube (pairs of vertex indices)
const uvec2 CUBE_EDGES[12] = uvec2[12](
    uvec2(0,1), uvec2(1,2), uvec2(2,3), uvec2(3,0),  // back face
    uvec2(4,5), uvec2(5,6), uvec2(6,7), uvec2(7,4),  // front face
    uvec2(0,4), uvec2(1,5), uvec2(2,6), uvec2(3,7)   // connecting edges
);

taskNV out ObstacleTaskPayload {
    uint query_id;
    uint obstacle_id;
    mat4 object_to_world;
    vec3 scale;
    vec3 front_color;
    vec3 back_color;
    uint subdivision_level;
    uint render_flags;  // CUBE_FLAG_* bits from pool
    float is_bev;
    uint face_mask;     // Bitmask of front-facing faces (backface culling)
};

// Face normals in local space (outward-facing, matches FACE_VERTS winding)
const vec3 FACE_NORMALS[6] = vec3[6](
    vec3( 0,  0, -1),  // face 0: -Z (back)
    vec3( 0,  0, +1),  // face 1: +Z (front)
    vec3(-1,  0,  0),  // face 2: -X
    vec3(+1,  0,  0),  // face 3: +X
    vec3( 0, -1,  0),  // face 4: -Y
    vec3( 0, +1,  0)   // face 5: +Y
);

// Helper to read translation from float buffer
vec3 read_translation(uint base_offset, uint pose_idx) {
    uint off = base_offset + pose_idx * 3u;
    return vec3(g_floats[off], g_floats[off + 1u], g_floats[off + 2u]);
}

// Helper to read quaternion from float buffer (x, y, z, w)
vec4 read_quaternion(uint base_offset, uint pose_idx) {
    uint off = base_offset + pose_idx * 4u;
    return vec4(g_floats[off], g_floats[off + 1u], g_floats[off + 2u], g_floats[off + 3u]);
}

// Helper to read int64 timestamp from float buffer (stored as 2 uints)
int64_t read_track_timestamp(uint base_offset, uint pose_idx) {
    // Track timestamps stored in timestamps buffer, not float buffer
    return g_timestamps[base_offset + pose_idx];
}

void main() {
    uint work_id = gl_WorkGroupID.x;

    // Decode work_id = query_id * max_obstacles + obstacle_id
    uint query_id_local = work_id / u_max_obstacles;
    uint obstacle_id_local = work_id % u_max_obstacles;

    if (query_id_local >= u_num_queries) {
        gl_TaskCountNV = 0u;
        return;
    }

    RenderQuery query = g_queries[query_id_local];
    TimestampedScene scene = g_scenes[query.scene_id];

    if (scene.valid == 0u || scene.num_cube_pools == 0u || u_cube_pool_index >= scene.num_cube_pools) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Access the cube pool at the specified index
    ObstaclePool pool = g_obstacle_pools[scene.cube_pools_offset + u_cube_pool_index];

    if (obstacle_id_local >= pool.num_cubes) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Ego obstacle is only visible in BEV cameras
    if (pool.prim_type_id == PRIM_EGO_OBSTACLE && query.camera_type_id != CAMERA_TYPE_BEV) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Get this obstacle's track range
    uint track_start = 0u;
    if (obstacle_id_local > 0u) {
        track_start = uint(g_int32[scene.int32_buffer_offset + pool.cube_ts_ps_offset + obstacle_id_local - 1u]);
    }
    uint track_end = uint(g_int32[scene.int32_buffer_offset + pool.cube_ts_ps_offset + obstacle_id_local]);
    uint track_len = track_end - track_start;

    if (track_len < 1u) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Find bracketing timestamps for interpolation
    uint track_ts_base = scene.timestamps_buffer_offset + pool.track_timestamps_offset + track_start;
    int64_t target_ts = query.timestamp_us;

    // Check bounds with extrapolation support
    int64_t first_ts = g_timestamps[track_ts_base];
    int64_t last_ts = g_timestamps[track_ts_base + track_len - 1u];

    // Determine if we need extrapolation
    bool extrapolate_before = (target_ts < first_ts);
    bool extrapolate_after = (target_ts > last_ts);

    if (extrapolate_before) {
        int64_t dt = first_ts - target_ts;
        if (dt > int64_t(u_max_extrapolation_us) || track_len < 2u) {
            gl_TaskCountNV = 0u;  // Beyond extrapolation limit
            return;
        }
    } else if (extrapolate_after) {
        int64_t dt = target_ts - last_ts;
        if (dt > int64_t(u_max_extrapolation_us) || track_len < 2u) {
            gl_TaskCountNV = 0u;  // Beyond extrapolation limit
            return;
        }
    }

    // Binary search for interpolation interval [t0, t1] where t0 <= target <= t1
    // For extrapolation, we use the first/last interval
    int idx0 = 0;
    int idx1 = 0;
    float alpha = 0.0;

    if (track_len == 1u) {
        // Single pose - no interpolation needed
        idx0 = 0;
        idx1 = 0;
        alpha = 0.0;
    } else if (extrapolate_before) {
        // Extrapolate backwards: use first two poses
        idx0 = 0;
        idx1 = 1;
        int64_t t0 = g_timestamps[track_ts_base];
        int64_t t1 = g_timestamps[track_ts_base + 1u];
        // alpha will be negative for backwards extrapolation
        if (t1 > t0) {
            alpha = float(target_ts - t0) / float(t1 - t0);
        } else {
            alpha = 0.0;
        }
    } else if (extrapolate_after) {
        // Extrapolate forwards: use last two poses
        idx0 = int(track_len) - 2;
        idx1 = int(track_len) - 1;
        int64_t t0 = g_timestamps[track_ts_base + uint(idx0)];
        int64_t t1 = g_timestamps[track_ts_base + uint(idx1)];
        // alpha will be > 1.0 for forward extrapolation
        if (t1 > t0) {
            alpha = float(target_ts - t0) / float(t1 - t0);
        } else {
            alpha = 1.0;
        }
    } else {
        // Normal interpolation: binary search for the interval
        int left = 0;
        int right = int(track_len) - 2;
        idx0 = 0;

        while (left <= right) {
            int mid = (left + right) / 2;
            int64_t t0 = g_timestamps[track_ts_base + uint(mid)];
            int64_t t1 = g_timestamps[track_ts_base + uint(mid) + 1u];

            if (t0 <= target_ts && target_ts <= t1) {
                idx0 = mid;
                break;
            } else if (target_ts < t0) {
                right = mid - 1;
            } else {
                left = mid + 1;
                idx0 = mid + 1;  // Update in case we exit
            }
        }

        idx1 = min(idx0 + 1, int(track_len) - 1);

        // Compute interpolation factor
        int64_t t0 = g_timestamps[track_ts_base + uint(idx0)];
        int64_t t1 = g_timestamps[track_ts_base + uint(idx1)];

        if (t1 > t0) {
            alpha = float(target_ts - t0) / float(t1 - t0);
        } else {
            alpha = 0.0;
        }
    }

    // Read translations and quaternions for interpolation
    uint trans_base = scene.float_buffer_offset + pool.translations_offset + track_start * 3u;
    uint quat_base = scene.float_buffer_offset + pool.quaternions_offset + track_start * 4u;

    vec3 trans0 = vec3(
        g_floats[trans_base + uint(idx0) * 3u],
        g_floats[trans_base + uint(idx0) * 3u + 1u],
        g_floats[trans_base + uint(idx0) * 3u + 2u]
    );
    vec3 trans1 = vec3(
        g_floats[trans_base + uint(idx1) * 3u],
        g_floats[trans_base + uint(idx1) * 3u + 1u],
        g_floats[trans_base + uint(idx1) * 3u + 2u]
    );

    vec4 quat0 = vec4(
        g_floats[quat_base + uint(idx0) * 4u],
        g_floats[quat_base + uint(idx0) * 4u + 1u],
        g_floats[quat_base + uint(idx0) * 4u + 2u],
        g_floats[quat_base + uint(idx0) * 4u + 3u]
    );
    vec4 quat1 = vec4(
        g_floats[quat_base + uint(idx1) * 4u],
        g_floats[quat_base + uint(idx1) * 4u + 1u],
        g_floats[quat_base + uint(idx1) * 4u + 2u],
        g_floats[quat_base + uint(idx1) * 4u + 3u]
    );

    // Interpolate translation (lerp) and rotation (slerp)
    // For position: extrapolate linearly (alpha can be negative or > 1)
    // For orientation: clamp to first/last keyframe (match wm-render behavior)
    vec3 trans_interp = mix(trans0, trans1, alpha);
    float alpha_clamped = clamp(alpha, 0.0, 1.0);
    vec4 quat_interp = quat_slerp(quat0, quat1, alpha_clamped);

    // Get scale
    uint scale_offset = scene.float_buffer_offset + pool.scales_offset + obstacle_id_local * 3u;
    vec3 scale_local = vec3(
        g_floats[scale_offset],
        g_floats[scale_offset + 1u],
        g_floats[scale_offset + 2u]
    );

    // Spatial culling: test bounding sphere against camera-centred view volume
    if (u_cull_radius_scale > 0.0) {
        float cull_r = g_camera_intrinsics[query.camera_id].depth_max * u_cull_radius_scale;
        CameraPose cull_pose = g_camera_poses[query_id_local];
        vec3 cam_pos = get_camera_world_pos(cull_pose);
        vec3 view_min = cam_pos - vec3(cull_r);
        vec3 view_max = cam_pos + vec3(cull_r);
        float radius = length(scale_local);
        if (!cull_sphere_aabb_overlap(trans_interp, radius, view_min, view_max)) {
            gl_TaskCountNV = 0u;
            return;
        }
    }

    // Build object_to_world matrix (rotation + translation, scale applied in mesh shader)
    mat4 obj_to_world = build_transform(trans_interp, quat_interp, vec3(1.0));

    // Get colors from buffer (first 3 floats = front, next 3 = back)
    uint color_offset = scene.float_buffer_offset + pool.colors_offset + obstacle_id_local * 6u;
    vec3 front = vec3(
        g_floats[color_offset],
        g_floats[color_offset + 1u],
        g_floats[color_offset + 2u]
    );
    vec3 back = vec3(
        g_floats[color_offset + 3u],
        g_floats[color_offset + 4u],
        g_floats[color_offset + 5u]
    );

    // Compute max subdivision level from all 12 edges
    FThetaCamera cam = g_camera_intrinsics[query.camera_id];
    CameraPose view_pose = g_camera_poses[query_id_local];

    // Backface culling: compute which faces are visible from camera
    mat3 R_view = mat3(view_pose.world_to_camera);
    vec3 cam_world = -transpose(R_view) * view_pose.world_to_camera[3].xyz;
    mat3 R_obj = mat3(obj_to_world);

    uint fmask = 0u;
    for (uint f = 0u; f < 6u; f++) {
        vec3 n_world = R_obj * FACE_NORMALS[f];
        vec3 center_local = FACE_NORMALS[f] * 0.5 * scale_local;
        vec3 center_world = (obj_to_world * vec4(center_local, 1.0)).xyz;
        if (dot(n_world, cam_world - center_world) > 0.0) {
            fmask |= (1u << f);
        }
    }

    if (fmask == 0u) {
        gl_TaskCountNV = 0u;
        return;
    }

    uint max_subdiv = 0u;
    if (u_tessellation_threshold > 0.0) {
        for (uint e = 0u; e < 12u; e++) {
            vec3 v0_local = CUBE_VERTS[CUBE_EDGES[e].x] * scale_local;
            vec3 v1_local = CUBE_VERTS[CUBE_EDGES[e].y] * scale_local;
            // Transform to world space using interpolated object_to_world matrix
            vec3 v0_world = (obj_to_world * vec4(v0_local, 1.0)).xyz;
            vec3 v1_world = (obj_to_world * vec4(v1_local, 1.0)).xyz;

            uint subdiv = compute_subdivision_level(v0_world, v1_world, view_pose, cam, u_tessellation_threshold);
            max_subdiv = max(max_subdiv, subdiv);
        }
        max_subdiv = min(max_subdiv, u_max_tessellation_cube);
    }

    // Dispatch 6 faces + 6 for wireframe edges if enabled
    uint wireframe = (pool.render_flags & CUBE_FLAG_WIREFRAME) != 0u ? 6u : 0u;
    gl_TaskCountNV = 6u + wireframe;

    query_id = query_id_local;
    obstacle_id = obstacle_id_local;
    object_to_world = obj_to_world;
    scale = scale_local;
    front_color = front;
    back_color = back;
    subdivision_level = max_subdiv;
    render_flags = pool.render_flags;
    is_bev = (query.camera_type_id == CAMERA_TYPE_BEV) ? 1.0 : 0.0;
    face_mask = fmask;
}
)";

static const char* LUDUS_TIMESTAMPED_OBSTACLE_MESH_SHADER = R"(
layout(local_size_x = 32) in;
// No per-vertex output arrays - all gradient data passed per-primitive
// This allows unlimited vertices (within mesh shader spec)
layout(triangles, max_vertices = 96, max_primitives = 128) out;

taskNV in ObstacleTaskPayload {
    uint query_id;
    uint obstacle_id;
    mat4 object_to_world;
    vec3 scale;
    vec3 front_color;
    vec3 back_color;
    uint subdivision_level;  // From task shader
    uint render_flags;       // CUBE_FLAG_* bits from pool
    float is_bev;
    uint face_mask;          // Bitmask of front-facing faces
};

// Per-primitive gradient data (no per-vertex outputs needed!)
layout(location = 0) perprimitiveNV out CubeGradientBlock {
    vec3 corner_t;      // gradient t values at triangle's 3 corners
    vec3 front_color;
    vec3 back_color;
    float is_bev;       // 1.0 for BEV cameras (disables fog), 0.0 otherwise
} prim_out[];

// Unit cube vertices
const vec3 CUBE_VERTS[8] = vec3[8](
    vec3(-0.5, -0.5, -0.5), vec3(+0.5, -0.5, -0.5),
    vec3(+0.5, +0.5, -0.5), vec3(-0.5, +0.5, -0.5),
    vec3(-0.5, -0.5, +0.5), vec3(+0.5, -0.5, +0.5),
    vec3(+0.5, +0.5, +0.5), vec3(-0.5, +0.5, +0.5)
);

// Face vertex indices (CCW winding)
const uvec4 FACE_VERTS[6] = uvec4[6](
    uvec4(0, 3, 2, 1),  // -Z (back)
    uvec4(4, 5, 6, 7),  // +Z (front)
    uvec4(0, 4, 7, 3),  // -X
    uvec4(1, 2, 6, 5),  // +X
    uvec4(0, 1, 5, 4),  // -Y
    uvec4(3, 7, 6, 2)   // +Y
);

// 12 edges of cube (pairs of vertex indices)
const uvec2 CUBE_EDGES[12] = uvec2[12](
    uvec2(0,1), uvec2(1,2), uvec2(2,3), uvec2(3,0),  // back face
    uvec2(4,5), uvec2(5,6), uvec2(6,7), uvec2(7,4),  // front face
    uvec2(0,4), uvec2(1,5), uvec2(2,6), uvec2(3,7)   // connecting edges
);

// Edge wireframe color (light gray - matches reference: [200, 200, 200] / 255.0)
const vec3 EDGE_COLOR = vec3(0.784, 0.784, 0.784);

void main() {
    uint workgroup_id = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    RenderQuery query = g_queries[query_id];
    FThetaCamera cam = g_camera_intrinsics[query.camera_id];
    CameraPose view_pose = g_camera_poses[query_id];

    // Workgroups 6-11 = wireframe edges (2 edges per workgroup, subdivided to match face tessellation)
    if (workgroup_id >= 6u) {
        uint wg_edge_offset = (workgroup_id - 6u) * 2u;  // 0, 2, 4, 6, 8, 10
        float EDGE_WIDTH = get_wireframe_width();

        // Edge-to-face adjacency for backface culling of wireframe edges
        const uvec2 EDGE_FACES[12] = uvec2[12](
            uvec2(0, 4), uvec2(0, 3), uvec2(0, 5), uvec2(0, 2),
            uvec2(1, 4), uvec2(1, 3), uvec2(1, 5), uvec2(1, 2),
            uvec2(2, 4), uvec2(3, 4), uvec2(3, 5), uvec2(2, 5)
        );

        // Check if either edge in this workgroup is adjacent to a visible face
        uint e0 = wg_edge_offset;
        uint e1 = wg_edge_offset + 1u;
        bool e0_vis = (face_mask & ((1u << EDGE_FACES[e0].x) | (1u << EDGE_FACES[e0].y))) != 0u;
        bool e1_vis = (face_mask & ((1u << EDGE_FACES[e1].x) | (1u << EDGE_FACES[e1].y))) != 0u;

        if (!e0_vis && !e1_vis) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        // Number of segments per edge based on subdivision level (matches face tessellation)
        // subdiv 0 = 1 segment, subdiv 1 = 2 segments, subdiv 2 = 4 segments
        uint segs_per_edge = 1u << subdivision_level;  // 1, 2, or 4
        uint total_segments = segs_per_edge * 2u;  // 2 edges

        gl_PrimitiveCountNV = total_segments * 2u;  // 2 triangles per segment quad

        // Process both edges in this workgroup
        for (uint local_seg = tid; local_seg < total_segments; local_seg += 32u) {
            uint edge_in_wg = local_seg / segs_per_edge;  // 0 or 1
            uint seg_in_edge = local_seg % segs_per_edge;
            uint edge_id = wg_edge_offset + edge_in_wg;

            // Skip edges adjacent only to back-facing faces
            uvec2 adj = EDGE_FACES[edge_id];
            if ((face_mask & ((1u << adj.x) | (1u << adj.y))) == 0u) {
                uint base_vert = local_seg * 4u;
                gl_MeshVerticesNV[base_vert].gl_Position = vec4(0, 0, 0, 1);
                gl_MeshVerticesNV[base_vert + 1u].gl_Position = vec4(0, 0, 0, 1);
                gl_MeshVerticesNV[base_vert + 2u].gl_Position = vec4(0, 0, 0, 1);
                gl_MeshVerticesNV[base_vert + 3u].gl_Position = vec4(0, 0, 0, 1);
                continue;
            }

            // Get edge endpoints
            uvec2 edge = CUBE_EDGES[edge_id];
            vec3 v0_local = CUBE_VERTS[edge.x] * scale;
            vec3 v1_local = CUBE_VERTS[edge.y] * scale;

            vec3 v0_world = (object_to_world * vec4(v0_local, 1.0)).xyz;
            vec3 v1_world = (object_to_world * vec4(v1_local, 1.0)).xyz;

            // Interpolate two world-space points for this segment
            float t0 = float(seg_in_edge) / float(segs_per_edge);
            float t1 = float(seg_in_edge + 1u) / float(segs_per_edge);

            vec3 pt0_world = mix(v0_world, v1_world, t0);
            vec3 pt1_world = mix(v0_world, v1_world, t1);

            // Project to clip space (f-theta curvature applied here)
            vec4 clip0 = ftheta_project(pt0_world, view_pose, cam);
            vec4 clip1 = ftheta_project(pt1_world, view_pose, cam);

            // Edge direction and perpendicular for width
            vec2 ndc0 = clip0.xy / clip0.w;
            vec2 ndc1 = clip1.xy / clip1.w;
            vec2 edge_dir = normalize(ndc1 - ndc0);
            vec2 perp = vec2(-edge_dir.y, edge_dir.x);
            vec2 offset = perp * EDGE_WIDTH / vec2(cam.img_w, cam.img_h);

            // Create 4 vertices for the quad
            uint base_vert = local_seg * 4u;

            vec4 off0 = vec4(offset * clip0.w, 0.0, 0.0);
            vec4 off1 = vec4(offset * clip1.w, 0.0, 0.0);

            // Depth bias to render edges on top of faces
            float z_bias0 = -0.001 * clip0.w;
            float z_bias1 = -0.001 * clip1.w;

            vec4 p0 = clip0 - off0; p0.z += z_bias0;
            vec4 p1 = clip0 + off0; p1.z += z_bias0;
            vec4 p2 = clip1 + off1; p2.z += z_bias1;
            vec4 p3 = clip1 - off1; p3.z += z_bias1;

            gl_MeshVerticesNV[base_vert + 0u].gl_Position = p0;
            gl_MeshVerticesNV[base_vert + 1u].gl_Position = p1;
            gl_MeshVerticesNV[base_vert + 2u].gl_Position = p2;
            gl_MeshVerticesNV[base_vert + 3u].gl_Position = p3;
        }

        barrier();

        // Generate triangle indices for segment quads
        for (uint local_seg = tid; local_seg < total_segments; local_seg += 32u) {
            uint base_vert = local_seg * 4u;
            uint base_tri = local_seg * 2u;

            // Triangle 1: 0, 1, 2
            gl_PrimitiveIndicesNV[(base_tri + 0u) * 3u + 0u] = base_vert + 0u;
            gl_PrimitiveIndicesNV[(base_tri + 0u) * 3u + 1u] = base_vert + 1u;
            gl_PrimitiveIndicesNV[(base_tri + 0u) * 3u + 2u] = base_vert + 2u;
            gl_MeshPrimitivesNV[base_tri + 0u].gl_Layer = int(query_id);
            // Wireframe uses solid edge color (front = back = EDGE_COLOR, corner_t doesn't matter)
            prim_out[base_tri + 0u].corner_t = vec3(0.5);
            prim_out[base_tri + 0u].front_color = EDGE_COLOR;
            prim_out[base_tri + 0u].back_color = EDGE_COLOR;
            prim_out[base_tri + 0u].is_bev = is_bev;

            // Triangle 2: 0, 2, 3
            gl_PrimitiveIndicesNV[(base_tri + 1u) * 3u + 0u] = base_vert + 0u;
            gl_PrimitiveIndicesNV[(base_tri + 1u) * 3u + 1u] = base_vert + 2u;
            gl_PrimitiveIndicesNV[(base_tri + 1u) * 3u + 2u] = base_vert + 3u;
            gl_MeshPrimitivesNV[base_tri + 1u].gl_Layer = int(query_id);
            prim_out[base_tri + 1u].corner_t = vec3(0.5);
            prim_out[base_tri + 1u].front_color = EDGE_COLOR;
            prim_out[base_tri + 1u].back_color = EDGE_COLOR;
            prim_out[base_tri + 1u].is_bev = is_bev;
        }

        return;
    }

    // Workgroups 0-5: Face rendering (2 triangles per face)
    uint face_id = workgroup_id;
    uint subdiv = subdivision_level;

    // Backface culling: skip faces not visible from camera
    if ((face_mask & (1u << face_id)) == 0u) {
        gl_PrimitiveCountNV = 0u;
        return;
    }

    // Get face corner info
    uvec4 face = FACE_VERTS[face_id];

    // Compute per-corner gradient t values and world positions
    float corner_t[4];
    vec3 corners[4];
    for (uint i = 0u; i < 4u; i++) {
        uint vert_idx = (i == 0u) ? face.x : (i == 1u) ? face.y : (i == 2u) ? face.z : face.w;
        vec3 local_vert = CUBE_VERTS[vert_idx];
        corner_t[i] = local_vert.x + 0.5;  // 0 at back (-X), 1 at front (+X)
        corners[i] = (object_to_world * vec4(local_vert * scale, 1.0)).xyz;
    }

    // Use shared barycentric subdivision utilities
    uint verts_per_tri = bary_vertex_count(subdiv);
    uint tris_per_tri = bary_triangle_count(subdiv);

    // 2 triangles per face
    uint total_verts = verts_per_tri * 2u;
    uint total_tris = tris_per_tri * 2u;

    gl_PrimitiveCountNV = total_tris;

    // Face = 2 triangles: (c0,c1,c2) and (c0,c2,c3)
    vec3 tri_verts[6];
    float tri_t[6];
    tri_verts[0] = corners[0]; tri_verts[1] = corners[1]; tri_verts[2] = corners[2];
    tri_verts[3] = corners[0]; tri_verts[4] = corners[2]; tri_verts[5] = corners[3];
    tri_t[0] = corner_t[0]; tri_t[1] = corner_t[1]; tri_t[2] = corner_t[2];
    tri_t[3] = corner_t[0]; tri_t[4] = corner_t[2]; tri_t[5] = corner_t[3];

    // Generate vertices for both triangles (position only, no per-vertex outputs)
    for (uint t = 0u; t < 2u; t++) {
        vec3 v0 = tri_verts[t * 3u];
        vec3 v1 = tri_verts[t * 3u + 1u];
        vec3 v2 = tri_verts[t * 3u + 2u];
        uint vert_base = t * verts_per_tri;

        for (uint v = tid; v < verts_per_tri; v += 32u) {
            vec2 uv = bary_vertex_uv(v, subdiv);
            vec3 pt = bary_interpolate(v0, v1, v2, uv);
            vec4 clip = ftheta_project(pt, view_pose, cam);
            gl_MeshVerticesNV[vert_base + v].gl_Position = clip;
        }
    }

    barrier();

    // Generate triangle indices and set per-primitive data (including corner_t)
    for (uint t = 0u; t < 2u; t++) {
        uint vert_base = t * verts_per_tri;
        uint tri_base = t * tris_per_tri;
        float t0 = tri_t[t * 3u];
        float t1 = tri_t[t * 3u + 1u];
        float t2 = tri_t[t * 3u + 2u];

        for (uint i = tid; i < tris_per_tri; i += 32u) {
            uvec3 idx = bary_triangle_indices(i, subdiv);
            uint idx_base = (tri_base + i) * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + idx.x;
            gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + idx.y;
            gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + idx.z;
            gl_MeshPrimitivesNV[tri_base + i].gl_Layer = int(query_id);

            // Compute gradient t values at each corner of this sub-triangle
            vec2 uv0 = bary_vertex_uv(idx.x, subdiv);
            vec2 uv1 = bary_vertex_uv(idx.y, subdiv);
            vec2 uv2 = bary_vertex_uv(idx.z, subdiv);
            vec3 corner_t_values = vec3(
                (1.0 - uv0.x - uv0.y) * t0 + uv0.x * t1 + uv0.y * t2,
                (1.0 - uv1.x - uv1.y) * t0 + uv1.x * t1 + uv1.y * t2,
                (1.0 - uv2.x - uv2.y) * t0 + uv2.x * t1 + uv2.y * t2
            );

            // Set per-primitive gradient data
            prim_out[tri_base + i].corner_t = corner_t_values;
            prim_out[tri_base + i].front_color = front_color;
            prim_out[tri_base + i].back_color = back_color;
            prim_out[tri_base + i].is_bev = is_bev;
        }
    }
}
)";

//=============================================================================
// API Implementation
//=============================================================================

void ludusTimestampedInit(NVDR_CTX_ARGS, LudusTimestampedState& s, int cudaDeviceIdx)
{
    // Create GL context
    s.glctx = createGLContext(cudaDeviceIdx);
    setGLContext(s.glctx);

    // Version check
    GLint vMajor = 0, vMinor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &vMajor);
    glGetIntegerv(GL_MINOR_VERSION, &vMinor);
    glGetError();
    LOG(INFO) << "Ludus Timestamped: OpenGL version " << vMajor << "." << vMinor;
    NVDR_CHECK((vMajor == 4 && vMinor >= 6) || vMajor > 4, "OpenGL 4.6+ required for mesh shaders");

    // Check mesh shader support
    s.hasMeshShader = (glDrawMeshTasksNV != nullptr) ? 1 : 0;
    NVDR_CHECK(s.hasMeshShader, "GL_NV_mesh_shader extension required");
    LOG(INFO) << "Ludus Timestamped: Mesh shaders supported";

    // Check for depth modification workaround
    int capMajor = 0;
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&capMajor, cudaDevAttrComputeCapabilityMajor, cudaDeviceIdx));
    s.enableZModify = (capMajor >= 8);

    // Compile fragment shaders
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glFragmentShader, GL_FRAGMENT_SHADER,
                             LUDUS_TIMESTAMPED_FRAGMENT_SHADER, s.enableZModify);
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glFragmentShaderPolyline, GL_FRAGMENT_SHADER,
                             LUDUS_TIMESTAMPED_POLYLINE_FRAGMENT_SHADER, s.enableZModify);
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glFragmentShaderObstacle, GL_FRAGMENT_SHADER,
                             LUDUS_TIMESTAMPED_OBSTACLE_FRAGMENT_SHADER, s.enableZModify);

    // Polyline shaders (uses perprimitiveNV fragment shader)
    std::string polylineTaskSrc = std::string(LUDUS_TIMESTAMPED_COMMON) + LUDUS_TIMESTAMPED_POLYLINE_TASK_SHADER;
    std::string polylineMeshSrc = std::string(LUDUS_TIMESTAMPED_COMMON) + LUDUS_TIMESTAMPED_POLYLINE_MESH_SHADER;
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glTaskShaderPolyline, GL_TASK_SHADER_NV,
                             polylineTaskSrc.c_str(), s.enableZModify);
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glMeshShaderPolyline, GL_MESH_SHADER_NV,
                             polylineMeshSrc.c_str(), s.enableZModify);
    constructTimestampedProgram(NVDR_CTX_PARAMS, &s.glProgramPolyline,
                                s.glTaskShaderPolyline, s.glMeshShaderPolyline, s.glFragmentShaderPolyline);
    LOG(INFO) << "Ludus Timestamped: Polyline shaders compiled";

    // Polygon shaders
    std::string polygonTaskSrc = std::string(LUDUS_TIMESTAMPED_COMMON) + LUDUS_TIMESTAMPED_POLYGON_TASK_SHADER;
    std::string polygonMeshSrc = std::string(LUDUS_TIMESTAMPED_COMMON) + LUDUS_TIMESTAMPED_POLYGON_MESH_SHADER;
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glTaskShaderPolygon, GL_TASK_SHADER_NV,
                             polygonTaskSrc.c_str(), s.enableZModify);
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glMeshShaderPolygon, GL_MESH_SHADER_NV,
                             polygonMeshSrc.c_str(), s.enableZModify);
    constructTimestampedProgram(NVDR_CTX_PARAMS, &s.glProgramPolygon,
                                s.glTaskShaderPolygon, s.glMeshShaderPolygon, s.glFragmentShaderPolyline);  // Uses perprimitiveNV color
    LOG(INFO) << "Ludus Timestamped: Polygon shaders compiled";

    // Obstacle shaders
    std::string obstacleTaskSrc = std::string(LUDUS_TIMESTAMPED_COMMON) + LUDUS_TIMESTAMPED_OBSTACLE_TASK_SHADER;
    std::string obstacleMeshSrc = std::string(LUDUS_TIMESTAMPED_COMMON) + LUDUS_TIMESTAMPED_OBSTACLE_MESH_SHADER;
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glTaskShaderObstacle, GL_TASK_SHADER_NV,
                             obstacleTaskSrc.c_str(), s.enableZModify);
    compileTimestampedShader(NVDR_CTX_PARAMS, &s.glMeshShaderObstacle, GL_MESH_SHADER_NV,
                             obstacleMeshSrc.c_str(), s.enableZModify);
    constructTimestampedProgram(NVDR_CTX_PARAMS, &s.glProgramObstacle,
                                s.glTaskShaderObstacle, s.glMeshShaderObstacle, s.glFragmentShaderObstacle);
    LOG(INFO) << "Ludus Timestamped: Obstacle shaders compiled";

    // Create FBO
    NVDR_CHECK_GL_ERROR(glGenFramebuffers(1, &s.glFBO));
    NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));

    GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 };
    NVDR_CHECK_GL_ERROR(glDrawBuffers(1, draw_buffers));

    // Depth test setup
    NVDR_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
    NVDR_CHECK_GL_ERROR(glDepthFunc(GL_LESS));
    NVDR_CHECK_GL_ERROR(glClearDepth(1.0));

    // Create scene-data SSBOs for both buffer sets (double buffering)
    for (int i = 0; i < 2; i++)
    {
        auto& b = s.bufferSets[i];
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glTimestampsBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glInt32Buffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glVertexBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glTriangleBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glPoseBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glFloatBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glSceneBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glPolylinePoolBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glPolygonPoolBuffer));
        NVDR_CHECK_GL_ERROR(glGenBuffers(1, &b.glObstaclePoolBuffer));
    }
    s.activeSet = 0;
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glColorPaletteBuffer));  // Configurable colors
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glCameraIntrinsicsBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glCameraPoseBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glQueryBuffer));

    // Initialize color palette with defaults (will be overwritten by Python if configured)
    s.colorPaletteSize = 0;
    s.cudaColorPaletteBuffer = nullptr;

    // Create textures
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glColorBuffer));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glDepthStencilBuffer));

    // Create MSAA resources (storage allocated on first use if MSAA enabled)
    NVDR_CHECK_GL_ERROR(glGenFramebuffers(1, &s.glFBO_MSAA));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glColorBuffer_MSAA));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glDepthStencilBuffer_MSAA));
    s.msaaSamples = 0;  // MSAA disabled by default

    // Set default tessellation threshold and max tessellation levels
    s.tessellationThreshold = 1.0f;
    s.maxTessellationLevelPolyline = 4;
    s.maxTessellationLevelPolygon = 3;
    s.maxTessellationLevelCube = 3;

    // Initialize double buffer for async transfer
    s.stagingBuffer[0] = nullptr;
    s.stagingBuffer[1] = nullptr;
    s.stagingBufferSize = 0;
    s.stagingWidth = 0;
    s.stagingHeight = 0;
    s.stagingNumQueries = 0;
    s.currentStagingIdx = 0;
    s.stagingValid[0] = 0;
    s.stagingValid[1] = 0;
    NVDR_CHECK_CUDA_ERROR(cudaStreamCreate(&s.copyStream));
    NVDR_CHECK_CUDA_ERROR(cudaEventCreate(&s.stagingReadyEvent[0]));
    NVDR_CHECK_CUDA_ERROR(cudaEventCreate(&s.stagingReadyEvent[1]));

    // Initialize double-buffered pinned host memory for async D2H transfer
    s.pinnedHostBuffer[0] = nullptr;
    s.pinnedHostBuffer[1] = nullptr;
    s.pinnedHostBufferSize = 0;
    s.currentPinnedIdx = 0;
    s.pinnedValid[0] = 0;
    s.pinnedValid[1] = 0;
    s.pinnedWidth[0] = s.pinnedWidth[1] = 0;
    s.pinnedHeight[0] = s.pinnedHeight[1] = 0;
    s.pinnedNumQueries[0] = s.pinnedNumQueries[1] = 0;
    NVDR_CHECK_CUDA_ERROR(cudaEventCreate(&s.pinnedReadyEvent[0]));
    NVDR_CHECK_CUDA_ERROR(cudaEventCreate(&s.pinnedReadyEvent[1]));

    // Initialize NVJPEG hardware encoder
    s.nvjpegInitialized = 0;
    s.nvjpegHandle = nullptr;
    s.nvjpegEncoderState = nullptr;
    s.nvjpegEncoderParams = nullptr;
    s.jpegOutputBuffer = nullptr;
    s.jpegOutputBufferSize = 0;
    s.jpegFlipBuffer = nullptr;
    s.jpegFlipBufferSize = 0;

    nvjpegStatus_t nvjStatus;
    nvjStatus = nvjpegCreateSimple(&s.nvjpegHandle);
    if (nvjStatus == NVJPEG_STATUS_SUCCESS)
    {
        nvjStatus = nvjpegEncoderStateCreate(s.nvjpegHandle, &s.nvjpegEncoderState, 0);
        if (nvjStatus == NVJPEG_STATUS_SUCCESS)
        {
            nvjStatus = nvjpegEncoderParamsCreate(s.nvjpegHandle, &s.nvjpegEncoderParams, 0);
            if (nvjStatus == NVJPEG_STATUS_SUCCESS)
            {
                s.nvjpegInitialized = 1;
                LOG(INFO) << "Ludus Timestamped: NVJPEG hardware encoder initialized";
            }
        }
    }
    if (!s.nvjpegInitialized)
    {
        LOG(WARNING) << "Ludus Timestamped: Failed to initialize NVJPEG, falling back to CPU encoding";
    }

    LOG(INFO) << "Ludus Timestamped: Initialization complete";
}

void ludusUploadCameras(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const FThetaCamera* intrinsics,
    int numCameras)
{
    setGLContext(s.glctx);

    // Resize if needed
    if (numCameras > s.cameraCapacity)
    {
        if (s.cudaCameraIntrinsicsBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCameraIntrinsicsBuffer));

        s.cameraCapacity = ROUND_UP_BITS(numCameras, 3);
        LOG(INFO) << "Ludus Timestamped: Resizing camera buffer to " << s.cameraCapacity;

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glCameraIntrinsicsBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                         s.cameraCapacity * sizeof(FThetaCamera), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaCameraIntrinsicsBuffer,
                                                           s.glCameraIntrinsicsBuffer,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));
    }

    s.numCameras = numCameras;

    // Upload via CUDA
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaCameraIntrinsicsBuffer, stream));
    void* ptr = NULL;
    size_t size = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &size, s.cudaCameraIntrinsicsBuffer));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(ptr, intrinsics, numCameras * sizeof(FThetaCamera),
                                          cudaMemcpyDeviceToDevice, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaCameraIntrinsicsBuffer, stream));
}

void ludusUploadColorPalette(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const float* colors,
    int numColors)
{
    setGLContext(s.glctx);

    // Each color is 4 floats (RGB + padding for vec4 alignment)
    const int floatsPerColor = 4;
    const size_t bufferSize = numColors * floatsPerColor * sizeof(float);

    // Resize if needed
    if (numColors > s.colorPaletteSize)
    {
        if (s.cudaColorPaletteBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorPaletteBuffer));

        s.colorPaletteSize = numColors;
        LOG(INFO) << "Ludus Timestamped: Allocating color palette for " << numColors << " colors";

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glColorPaletteBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaColorPaletteBuffer,
                                                           s.glColorPaletteBuffer,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // Upload via CUDA (colors come as RGB, we need to expand to RGBA for vec4)
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaColorPaletteBuffer, stream));
    void* ptr = NULL;
    size_t size = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &size, s.cudaColorPaletteBuffer));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(ptr, colors, bufferSize, cudaMemcpyDeviceToDevice, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaColorPaletteBuffer, stream));

    LOG(INFO) << "Ludus Timestamped: Uploaded " << numColors << " custom colors";
}

// Helper to resize and register a buffer, preserving existing data via CUDA
static void resizeBuffer(NVDR_CTX_ARGS, GLuint glBuffer, cudaGraphicsResource_t& cudaRes,
                         int& capacity, int used, int needed, size_t elemSize, const char* name,
                         cudaStream_t stream = nullptr)
{
    if (needed > capacity)
    {
        int oldCapacity = capacity;
        int oldUsed = used;  // Amount of valid data to preserve
        void* tempCudaBuffer = nullptr;
        size_t bytesToPreserve = oldUsed * elemSize;

        // If there's existing data to preserve, copy it to CUDA temp buffer first
        if (oldUsed > 0 && cudaRes)
        {
            // Allocate CUDA temp buffer
            NVDR_CHECK_CUDA_ERROR(cudaMalloc(&tempCudaBuffer, bytesToPreserve));

            // Map old buffer and copy to temp
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaRes, stream));
            void* oldPtr = nullptr;
            size_t oldSize = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&oldPtr, &oldSize, cudaRes));
            NVDR_CHECK_CUDA_ERROR(cudaMemcpy(tempCudaBuffer, oldPtr, bytesToPreserve,
                                              cudaMemcpyDeviceToDevice));  // Synchronous copy
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaRes, stream));
        }

        // Unregister old CUDA resource
        if (cudaRes)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaRes));

        capacity = ROUND_UP_BITS(needed, 4);
        LOG(INFO) << "Ludus Timestamped: Resizing " << name << " buffer from "
                  << oldCapacity << " to " << capacity << " (preserving " << oldUsed << " elements)";

        // Resize the GL buffer
        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, capacity * elemSize, NULL, GL_DYNAMIC_DRAW));

        // Re-register with CUDA
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaRes, glBuffer,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));

        // Restore preserved data from temp buffer
        if (tempCudaBuffer)
        {
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaRes, stream));
            void* newPtr = nullptr;
            size_t newSize = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&newPtr, &newSize, cudaRes));
            NVDR_CHECK_CUDA_ERROR(cudaMemcpy(newPtr, tempCudaBuffer, bytesToPreserve,
                                              cudaMemcpyDeviceToDevice));  // Synchronous copy
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaRes, stream));

            // Free temp buffer
            NVDR_CHECK_CUDA_ERROR(cudaFree(tempCudaBuffer));
        }
    }
}

int ludusUploadScene(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const TimestampedScene* sceneDesc,
    const TimestampedPolylinePool* polylinePools,
    int numPolylinePools,
    const TimestampedPolygonPool* polygonPools,
    int numPolygonPools,
    const ObstaclePool* obstaclePools,
    int numObstaclePools,
    int maxObstaclesInPool,
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
    int numFloats)
{
    setGLContext(s.glctx);

    auto& b = s.bufferSets[s.activeSet];
    int sceneId = b.numScenes;

    // Resize scene buffer if needed (with data preservation)
    if (b.numScenes + 1 > b.maxScenes)
    {
        void* tempSceneBuffer = nullptr;
        size_t bytesToPreserve = b.numScenes * sizeof(TimestampedScene);

        // Save existing scene descriptors to temp buffer
        if (b.numScenes > 0 && b.cudaSceneBuffer)
        {
            NVDR_CHECK_CUDA_ERROR(cudaMalloc(&tempSceneBuffer, bytesToPreserve));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &b.cudaSceneBuffer, stream));
            void* oldPtr = nullptr;
            size_t oldSize = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&oldPtr, &oldSize, b.cudaSceneBuffer));
            NVDR_CHECK_CUDA_ERROR(cudaMemcpy(tempSceneBuffer, oldPtr, bytesToPreserve, cudaMemcpyDeviceToDevice));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &b.cudaSceneBuffer, stream));
        }

        if (b.cudaSceneBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(b.cudaSceneBuffer));

        b.maxScenes = ROUND_UP_BITS(b.numScenes + 1, 3);
        LOG(INFO) << "Ludus Timestamped: Resizing scene buffer to " << b.maxScenes
                  << " (preserving " << b.numScenes << " scenes)";

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, b.glSceneBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                         b.maxScenes * sizeof(TimestampedScene), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&b.cudaSceneBuffer, b.glSceneBuffer,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));

        // Restore scene descriptors from temp buffer
        if (tempSceneBuffer)
        {
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &b.cudaSceneBuffer, stream));
            void* newPtr = nullptr;
            size_t newSize = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&newPtr, &newSize, b.cudaSceneBuffer));
            NVDR_CHECK_CUDA_ERROR(cudaMemcpy(newPtr, tempSceneBuffer, bytesToPreserve, cudaMemcpyDeviceToDevice));
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &b.cudaSceneBuffer, stream));
            NVDR_CHECK_CUDA_ERROR(cudaFree(tempSceneBuffer));
        }
    }

    // Resize global data buffers (preserving existing data via CUDA)
    resizeBuffer(NVDR_CTX_PARAMS, b.glTimestampsBuffer, b.cudaTimestampsBuffer,
                 b.timestampsCapacity, b.timestampsUsed, b.timestampsUsed + numTimestamps, sizeof(int64_t), "timestamps", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glInt32Buffer, b.cudaInt32Buffer,
                 b.int32Capacity, b.int32Used, b.int32Used + numInt32, sizeof(int32_t), "int32", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glVertexBuffer, b.cudaVertexBuffer,
                 b.vertexCapacity, b.vertexUsed, b.vertexUsed + numVertices, sizeof(Vertex), "vertex", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glTriangleBuffer, b.cudaTriangleBuffer,
                 b.triangleCapacity, b.triangleUsed, b.triangleUsed + numTriangles, sizeof(Triangle), "triangle", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glPoseBuffer, b.cudaPoseBuffer,
                 b.poseCapacity, b.poseUsed, b.poseUsed + numPoses, sizeof(CameraPose), "pose", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glFloatBuffer, b.cudaFloatBuffer,
                 b.floatCapacity, b.floatUsed, b.floatUsed + numFloats, sizeof(float), "float", stream);

    // Resize pool buffers (preserving existing data via CUDA)
    resizeBuffer(NVDR_CTX_PARAMS, b.glPolylinePoolBuffer, b.cudaPolylinePoolBuffer,
                 b.polylinePoolCapacity, b.polylinePoolUsed, b.polylinePoolUsed + numPolylinePools,
                 sizeof(TimestampedPolylinePool), "polyline pool", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glPolygonPoolBuffer, b.cudaPolygonPoolBuffer,
                 b.polygonPoolCapacity, b.polygonPoolUsed, b.polygonPoolUsed + numPolygonPools,
                 sizeof(TimestampedPolygonPool), "polygon pool", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glObstaclePoolBuffer, b.cudaObstaclePoolBuffer,
                 b.obstaclePoolCapacity, b.obstaclePoolUsed, b.obstaclePoolUsed + numObstaclePools,
                 sizeof(ObstaclePool), "obstacle pool", stream);

    // Build list of resources to map (only non-null)
    std::vector<cudaGraphicsResource_t> resources;
    if (b.cudaSceneBuffer) resources.push_back(b.cudaSceneBuffer);
    if (b.cudaTimestampsBuffer) resources.push_back(b.cudaTimestampsBuffer);
    if (b.cudaInt32Buffer) resources.push_back(b.cudaInt32Buffer);
    if (b.cudaVertexBuffer) resources.push_back(b.cudaVertexBuffer);
    if (b.cudaTriangleBuffer) resources.push_back(b.cudaTriangleBuffer);
    if (b.cudaPoseBuffer) resources.push_back(b.cudaPoseBuffer);
    if (b.cudaFloatBuffer) resources.push_back(b.cudaFloatBuffer);
    if (b.cudaPolylinePoolBuffer) resources.push_back(b.cudaPolylinePoolBuffer);
    if (b.cudaPolygonPoolBuffer) resources.push_back(b.cudaPolygonPoolBuffer);
    if (b.cudaObstaclePoolBuffer) resources.push_back(b.cudaObstaclePoolBuffer);

    if (resources.empty()) {
        LOG(WARNING) << "Ludus Timestamped: No buffers to map for scene upload";
        return -1;
    }
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources((int)resources.size(), resources.data(), stream));

    // Upload data - helper macro for safe upload
    void* ptr; size_t sz;
    #define UPLOAD_IF_VALID(cudaRes, offset, src, count, elemSize) \
        if ((cudaRes) && (count) > 0) { \
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &sz, cudaRes)); \
            NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)ptr + (offset) * (elemSize), \
                                                  src, (count) * (elemSize), \
                                                  cudaMemcpyDeviceToDevice, stream)); \
        }

    // Scene descriptor (always valid since we resized above)
    UPLOAD_IF_VALID(b.cudaSceneBuffer, sceneId, sceneDesc, 1, sizeof(TimestampedScene));

    // Timestamps
    UPLOAD_IF_VALID(b.cudaTimestampsBuffer, b.timestampsUsed, timestamps, numTimestamps, sizeof(int64_t));

    // Int32 data
    UPLOAD_IF_VALID(b.cudaInt32Buffer, b.int32Used, int32Data, numInt32, sizeof(int32_t));

    // Vertices
    UPLOAD_IF_VALID(b.cudaVertexBuffer, b.vertexUsed, vertices, numVertices, sizeof(Vertex));

    // Triangles
    UPLOAD_IF_VALID(b.cudaTriangleBuffer, b.triangleUsed, triangles, numTriangles, sizeof(Triangle));

    // Poses
    UPLOAD_IF_VALID(b.cudaPoseBuffer, b.poseUsed, poses, numPoses, sizeof(CameraPose));

    // Float data
    UPLOAD_IF_VALID(b.cudaFloatBuffer, b.floatUsed, floatData, numFloats, sizeof(float));

    // Pool headers
    UPLOAD_IF_VALID(b.cudaPolylinePoolBuffer, b.polylinePoolUsed, polylinePools, numPolylinePools, sizeof(TimestampedPolylinePool));
    UPLOAD_IF_VALID(b.cudaPolygonPoolBuffer, b.polygonPoolUsed, polygonPools, numPolygonPools, sizeof(TimestampedPolygonPool));
    UPLOAD_IF_VALID(b.cudaObstaclePoolBuffer, b.obstaclePoolUsed, obstaclePools, numObstaclePools, sizeof(ObstaclePool));

    #undef UPLOAD_IF_VALID

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources((int)resources.size(), resources.data(), stream));

    // Update usage counters
    b.timestampsUsed += numTimestamps;
    b.int32Used += numInt32;
    b.vertexUsed += numVertices;
    b.triangleUsed += numTriangles;
    b.poseUsed += numPoses;
    b.floatUsed += numFloats;
    b.polylinePoolUsed += numPolylinePools;
    b.polygonPoolUsed += numPolygonPools;
    b.obstaclePoolUsed += numObstaclePools;
    b.numScenes++;

    // Track per-scene max values for dispatch sizing
    b.maxObstaclesPerPool = std::max(b.maxObstaclesPerPool, maxObstaclesInPool);
    b.maxCubePoolsPerScene = std::max(b.maxCubePoolsPerScene, numObstaclePools);
    b.maxPolylinePoolsPerScene = std::max(b.maxPolylinePoolsPerScene, numPolylinePools);
    b.maxPolygonPoolsPerScene = std::max(b.maxPolygonPoolsPerScene, numPolygonPools);

    LOG(INFO) << "Ludus Timestamped: Uploaded scene " << sceneId
               << " (polyline pools: " << numPolylinePools << "/" << b.maxPolylinePoolsPerScene
               << ", polygon pools: " << numPolygonPools << "/" << b.maxPolygonPoolsPerScene
               << ", cube pools: " << numObstaclePools << "/" << b.maxCubePoolsPerScene << ")";
    return sceneId;
}

void ludusRemoveScene(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int sceneId,
    cudaStream_t stream)
{
    setGLContext(s.glctx);

    auto& b = s.bufferSets[s.activeSet];
    if (sceneId < 0 || sceneId >= b.numScenes) {
        LOG(WARNING) << "Ludus Timestamped: removeScene invalid sceneId=" << sceneId
                     << " (numScenes=" << b.numScenes << ")";
        return;
    }

    // Map the scene buffer, set the valid field to 0 for this scene.
    // TimestampedScene is 128 bytes (32 x uint32). valid is at offset 12 (uint index).
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &b.cudaSceneBuffer, stream));
    void* ptr = nullptr;
    size_t sz = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &sz, b.cudaSceneBuffer));

    uint32_t zero = 0;
    size_t byteOffset = (size_t)sceneId * sizeof(TimestampedScene) + 12 * sizeof(uint32_t);
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(
        (char*)ptr + byteOffset, &zero, sizeof(uint32_t),
        cudaMemcpyHostToDevice, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &b.cudaSceneBuffer, stream));
    NVDR_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    LOG(INFO) << "Ludus Timestamped: Removed (tombstoned) scene " << sceneId;
}

// ---------------------------------------------------------------------------
// Pre-allocate all data buffers so that subsequent uploads rarely resize.
// ---------------------------------------------------------------------------

static void preallocBuffer(NVDR_CTX_ARGS, GLuint glBuffer,
                           cudaGraphicsResource_t& cudaRes,
                           int& capacity, size_t elemSize,
                           int targetElems, const char* name)
{
    if (targetElems <= capacity)
        return;
    if (cudaRes)
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaRes));
    capacity = ROUND_UP_BITS(targetElems, 4);
    LOG(INFO) << "Ludus Timestamped: Pre-allocating " << name
              << " buffer to " << capacity << " elements ("
              << (capacity * elemSize) / 1024 << " KB)";
    NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBuffer));
    NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                     capacity * elemSize, NULL, GL_DYNAMIC_DRAW));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(
        &cudaRes, glBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
}

void ludusPreallocateBuffers(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int maxScenes,
    int bytesPerScene)
{
    setGLContext(s.glctx);

    // Heuristic split of bytesPerScene into per-buffer counts based on
    // typical AV2 scene composition (~2 MB, heavily vertex-dominated).
    int avgVerts      = bytesPerScene / 32;   // ~60 K verts for 2 MB
    int avgTris       = avgVerts / 8;
    int avgTimestamps = avgVerts / 8;
    int avgInt32      = avgVerts / 4;
    int avgFloats     = avgVerts / 2;
    int avgPoses      = 128;
    int avgPools      = 16;

    for (int i = 0; i < 2; i++)
    {
        auto& b = s.bufferSets[i];
        preallocBuffer(NVDR_CTX_PARAMS, b.glSceneBuffer, b.cudaSceneBuffer,
                       b.maxScenes, sizeof(TimestampedScene), maxScenes, "scene");
        preallocBuffer(NVDR_CTX_PARAMS, b.glTimestampsBuffer, b.cudaTimestampsBuffer,
                       b.timestampsCapacity, sizeof(int64_t),
                       maxScenes * avgTimestamps, "timestamps");
        preallocBuffer(NVDR_CTX_PARAMS, b.glInt32Buffer, b.cudaInt32Buffer,
                       b.int32Capacity, sizeof(int32_t),
                       maxScenes * avgInt32, "int32");
        preallocBuffer(NVDR_CTX_PARAMS, b.glVertexBuffer, b.cudaVertexBuffer,
                       b.vertexCapacity, sizeof(Vertex),
                       maxScenes * avgVerts, "vertex");
        preallocBuffer(NVDR_CTX_PARAMS, b.glTriangleBuffer, b.cudaTriangleBuffer,
                       b.triangleCapacity, sizeof(Triangle),
                       maxScenes * avgTris, "triangle");
        preallocBuffer(NVDR_CTX_PARAMS, b.glPoseBuffer, b.cudaPoseBuffer,
                       b.poseCapacity, sizeof(CameraPose),
                       maxScenes * avgPoses, "pose");
        preallocBuffer(NVDR_CTX_PARAMS, b.glFloatBuffer, b.cudaFloatBuffer,
                       b.floatCapacity, sizeof(float),
                       maxScenes * avgFloats, "float");
        preallocBuffer(NVDR_CTX_PARAMS, b.glPolylinePoolBuffer, b.cudaPolylinePoolBuffer,
                       b.polylinePoolCapacity, sizeof(TimestampedPolylinePool),
                       maxScenes * avgPools, "polyline pool");
        preallocBuffer(NVDR_CTX_PARAMS, b.glPolygonPoolBuffer, b.cudaPolygonPoolBuffer,
                       b.polygonPoolCapacity, sizeof(TimestampedPolygonPool),
                       maxScenes * avgPools, "polygon pool");
        preallocBuffer(NVDR_CTX_PARAMS, b.glObstaclePoolBuffer, b.cudaObstaclePoolBuffer,
                       b.obstaclePoolCapacity, sizeof(ObstaclePool),
                       maxScenes * avgPools, "obstacle pool");
    }

    LOG(INFO) << "Ludus Timestamped: Pre-allocated buffers for "
              << maxScenes << " scenes @ " << bytesPerScene << " B/scene (both buffer sets)";
}

// ---------------------------------------------------------------------------
// Batch upload: single map/unmap for N scenes.
// ---------------------------------------------------------------------------

int ludusUploadScenesBatch(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    int numScenesInBatch,
    const TimestampedScene* sceneDescs,
    const TimestampedPolylinePool* polylinePools,
    const TimestampedPolygonPool* polygonPools,
    const ObstaclePool* obstaclePools,
    const SceneUploadBounds* bounds,
    const int64_t* timestamps,
    int totalTimestamps,
    const int32_t* int32Data,
    int totalInt32,
    const Vertex* vertices,
    int totalVertices,
    const Triangle* triangles,
    int totalTriangles,
    const CameraPose* poses,
    int totalPoses,
    const float* floatData,
    int totalFloats)
{
    setGLContext(s.glctx);

    auto& b = s.bufferSets[s.activeSet];
    int firstSceneId = b.numScenes;

    // Compute totals for pool headers
    int totalPolyPools = 0, totalGonPools = 0, totalObsPools = 0;
    for (int i = 0; i < numScenesInBatch; i++) {
        totalPolyPools += bounds[i].numPolylinePools;
        totalGonPools  += bounds[i].numPolygonPools;
        totalObsPools  += bounds[i].numObstaclePools;
    }

    // Resize scene buffer if needed
    resizeBuffer(NVDR_CTX_PARAMS, b.glSceneBuffer, b.cudaSceneBuffer,
                 b.maxScenes, b.numScenes, b.numScenes + numScenesInBatch,
                 sizeof(TimestampedScene), "scene (batch)", stream);

    // Resize data buffers (all at once, before a single map)
    resizeBuffer(NVDR_CTX_PARAMS, b.glTimestampsBuffer, b.cudaTimestampsBuffer,
                 b.timestampsCapacity, b.timestampsUsed,
                 b.timestampsUsed + totalTimestamps, sizeof(int64_t), "timestamps", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glInt32Buffer, b.cudaInt32Buffer,
                 b.int32Capacity, b.int32Used,
                 b.int32Used + totalInt32, sizeof(int32_t), "int32", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glVertexBuffer, b.cudaVertexBuffer,
                 b.vertexCapacity, b.vertexUsed,
                 b.vertexUsed + totalVertices, sizeof(Vertex), "vertex", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glTriangleBuffer, b.cudaTriangleBuffer,
                 b.triangleCapacity, b.triangleUsed,
                 b.triangleUsed + totalTriangles, sizeof(Triangle), "triangle", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glPoseBuffer, b.cudaPoseBuffer,
                 b.poseCapacity, b.poseUsed,
                 b.poseUsed + totalPoses, sizeof(CameraPose), "pose", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glFloatBuffer, b.cudaFloatBuffer,
                 b.floatCapacity, b.floatUsed,
                 b.floatUsed + totalFloats, sizeof(float), "float", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glPolylinePoolBuffer, b.cudaPolylinePoolBuffer,
                 b.polylinePoolCapacity, b.polylinePoolUsed,
                 b.polylinePoolUsed + totalPolyPools,
                 sizeof(TimestampedPolylinePool), "polyline pool", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glPolygonPoolBuffer, b.cudaPolygonPoolBuffer,
                 b.polygonPoolCapacity, b.polygonPoolUsed,
                 b.polygonPoolUsed + totalGonPools,
                 sizeof(TimestampedPolygonPool), "polygon pool", stream);
    resizeBuffer(NVDR_CTX_PARAMS, b.glObstaclePoolBuffer, b.cudaObstaclePoolBuffer,
                 b.obstaclePoolCapacity, b.obstaclePoolUsed,
                 b.obstaclePoolUsed + totalObsPools,
                 sizeof(ObstaclePool), "obstacle pool", stream);

    // --- Single map for all resources ---
    std::vector<cudaGraphicsResource_t> resources;
    if (b.cudaSceneBuffer)       resources.push_back(b.cudaSceneBuffer);
    if (b.cudaTimestampsBuffer)  resources.push_back(b.cudaTimestampsBuffer);
    if (b.cudaInt32Buffer)       resources.push_back(b.cudaInt32Buffer);
    if (b.cudaVertexBuffer)      resources.push_back(b.cudaVertexBuffer);
    if (b.cudaTriangleBuffer)    resources.push_back(b.cudaTriangleBuffer);
    if (b.cudaPoseBuffer)        resources.push_back(b.cudaPoseBuffer);
    if (b.cudaFloatBuffer)       resources.push_back(b.cudaFloatBuffer);
    if (b.cudaPolylinePoolBuffer) resources.push_back(b.cudaPolylinePoolBuffer);
    if (b.cudaPolygonPoolBuffer)  resources.push_back(b.cudaPolygonPoolBuffer);
    if (b.cudaObstaclePoolBuffer) resources.push_back(b.cudaObstaclePoolBuffer);

    if (resources.empty()) {
        LOG(WARNING) << "Ludus Timestamped: No buffers for batch upload";
        return -1;
    }
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources((int)resources.size(), resources.data(), stream));

    void* ptr; size_t sz;
    #define UPLOAD_BATCH(cudaRes, globalOff, src, count, elemSz) \
        if ((cudaRes) && (count) > 0) { \
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &sz, cudaRes)); \
            NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync((char*)ptr + (size_t)(globalOff) * (elemSz), \
                                                  src, (size_t)(count) * (elemSz), \
                                                  cudaMemcpyDeviceToDevice, stream)); \
        }

    // Scene descriptors (contiguous block)
    UPLOAD_BATCH(b.cudaSceneBuffer, b.numScenes, sceneDescs, numScenesInBatch, sizeof(TimestampedScene));

    // Bulk data (already concatenated by caller)
    UPLOAD_BATCH(b.cudaTimestampsBuffer, b.timestampsUsed, timestamps, totalTimestamps, sizeof(int64_t));
    UPLOAD_BATCH(b.cudaInt32Buffer, b.int32Used, int32Data, totalInt32, sizeof(int32_t));
    UPLOAD_BATCH(b.cudaVertexBuffer, b.vertexUsed, vertices, totalVertices, sizeof(Vertex));
    UPLOAD_BATCH(b.cudaTriangleBuffer, b.triangleUsed, triangles, totalTriangles, sizeof(Triangle));
    UPLOAD_BATCH(b.cudaPoseBuffer, b.poseUsed, poses, totalPoses, sizeof(CameraPose));
    UPLOAD_BATCH(b.cudaFloatBuffer, b.floatUsed, floatData, totalFloats, sizeof(float));

    // Pool headers (also concatenated)
    UPLOAD_BATCH(b.cudaPolylinePoolBuffer, b.polylinePoolUsed,
                 polylinePools, totalPolyPools, sizeof(TimestampedPolylinePool));
    UPLOAD_BATCH(b.cudaPolygonPoolBuffer, b.polygonPoolUsed,
                 polygonPools, totalGonPools, sizeof(TimestampedPolygonPool));
    UPLOAD_BATCH(b.cudaObstaclePoolBuffer, b.obstaclePoolUsed,
                 obstaclePools, totalObsPools, sizeof(ObstaclePool));

    #undef UPLOAD_BATCH

    // --- Single unmap ---
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources((int)resources.size(), resources.data(), stream));

    // Update counters
    b.timestampsUsed   += totalTimestamps;
    b.int32Used        += totalInt32;
    b.vertexUsed       += totalVertices;
    b.triangleUsed     += totalTriangles;
    b.poseUsed         += totalPoses;
    b.floatUsed        += totalFloats;
    b.polylinePoolUsed += totalPolyPools;
    b.polygonPoolUsed  += totalGonPools;
    b.obstaclePoolUsed += totalObsPools;

    for (int i = 0; i < numScenesInBatch; i++) {
        b.maxObstaclesPerPool    = std::max(b.maxObstaclesPerPool, bounds[i].maxObstaclesInPool);
        b.maxCubePoolsPerScene   = std::max(b.maxCubePoolsPerScene, bounds[i].numObstaclePools);
        b.maxPolylinePoolsPerScene = std::max(b.maxPolylinePoolsPerScene, bounds[i].numPolylinePools);
        b.maxPolygonPoolsPerScene  = std::max(b.maxPolygonPoolsPerScene, bounds[i].numPolygonPools);
    }
    b.numScenes += numScenesInBatch;

    LOG(INFO) << "Ludus Timestamped: Batch uploaded " << numScenesInBatch
              << " scenes (IDs " << firstSceneId << ".." << (firstSceneId + numScenesInBatch - 1) << ")";
    return firstSceneId;
}

void ludusRenderBatch(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    const RenderQuery* queries,
    const CameraPose* cameraPoses,
    int numQueries,
    int width,
    int height)
{
    setGLContext(s.glctx);

    auto& b = s.bufferSets[s.activeSet];

    // Resize framebuffer if needed
    if (width > s.width || height > s.height || numQueries > s.maxLayers)
    {
        if (s.cudaColorBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer));

        s.width = (width > s.width) ? ROUND_UP(width, 32) : s.width;
        s.height = (height > s.height) ? ROUND_UP(height, 32) : s.height;
        s.maxLayers = (numQueries > s.maxLayers) ? ROUND_UP_BITS(numQueries, 4) : s.maxLayers;

        LOG(INFO) << "Ludus Timestamped: Resizing framebuffer to "
                  << s.width << "x" << s.height << "x" << s.maxLayers;

        // Allocate color buffer (layered texture array) - used for resolve target and CUDA readback
        NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer));
        NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
                                         s.width, s.height, s.maxLayers, 0,
                                         GL_RGBA, GL_UNSIGNED_BYTE, 0));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer, 0));

        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glDepthStencilBuffer));
        NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH24_STENCIL8,
                                         s.width, s.height, s.maxLayers, 0,
                                         GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 0));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                                 s.glDepthStencilBuffer, 0));

        // Allocate MSAA buffers if MSAA is enabled
        if (s.msaaSamples >= 2)
        {
            LOG(INFO) << "Ludus Timestamped: Allocating MSAA framebuffer with " << s.msaaSamples << " samples";
            NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO_MSAA));

            // MSAA color buffer (GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
            NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.glColorBuffer_MSAA));
            NVDR_CHECK_GL_ERROR(glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.msaaSamples,
                                                        GL_RGBA8, s.width, s.height, s.maxLayers, GL_TRUE));
            NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer_MSAA, 0));

            // MSAA depth/stencil buffer
            NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.glDepthStencilBuffer_MSAA));
            NVDR_CHECK_GL_ERROR(glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.msaaSamples,
                                                        GL_DEPTH24_STENCIL8, s.width, s.height, s.maxLayers, GL_TRUE));
            NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                                     s.glDepthStencilBuffer_MSAA, 0));

            // Verify MSAA framebuffer is complete
            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE)
            {
                LOG(WARNING) << "Ludus Timestamped: MSAA framebuffer incomplete (status=" << status << "), disabling MSAA";
                s.msaaSamples = 0;
            }

            // Setup draw buffers for MSAA FBO
            GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 };
            NVDR_CHECK_GL_ERROR(glDrawBuffers(1, draw_buffers));
        }

        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&s.cudaColorBuffer, s.glColorBuffer,
                                                          GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
    }

    // Resize query buffer
    if (numQueries > s.queryCapacity)
    {
        if (s.cudaQueryBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaQueryBuffer));

        s.queryCapacity = ROUND_UP_BITS(numQueries, 4);

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glQueryBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                         s.queryCapacity * sizeof(RenderQuery), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaQueryBuffer, s.glQueryBuffer,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // Resize camera pose buffer (one per query)
    if (numQueries > s.posePerQueryCapacity)
    {
        if (s.cudaCameraPoseBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCameraPoseBuffer));

        s.posePerQueryCapacity = s.queryCapacity;  // Track capacity

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glCameraPoseBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                         s.posePerQueryCapacity * sizeof(CameraPose), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaCameraPoseBuffer, s.glCameraPoseBuffer,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // Upload queries and poses
    std::vector<cudaGraphicsResource_t> resources = {s.cudaQueryBuffer, s.cudaCameraPoseBuffer};
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources((int)resources.size(), resources.data(), stream));

    void* ptr; size_t sz;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &sz, s.cudaQueryBuffer));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(ptr, queries, numQueries * sizeof(RenderQuery),
                                          cudaMemcpyDeviceToDevice, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&ptr, &sz, s.cudaCameraPoseBuffer));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(ptr, cameraPoses, numQueries * sizeof(CameraPose),
                                          cudaMemcpyDeviceToDevice, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources((int)resources.size(), resources.data(), stream));

    // Sync before GL commands
    NVDR_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Setup framebuffer
    // If MSAA is enabled, render to MSAA FBO; otherwise render directly to regular FBO
    GLuint renderFBO = (s.msaaSamples >= 2) ? s.glFBO_MSAA : s.glFBO;
    NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, renderFBO));
    NVDR_CHECK_GL_ERROR(glViewport(0, 0, width, height));
    NVDR_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    // Bind scene-data SSBOs from active buffer set
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b.glTimestampsBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b.glInt32Buffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, b.glVertexBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, b.glTriangleBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, b.glPoseBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, b.glFloatBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, b.glSceneBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, b.glPolylinePoolBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, b.glPolygonPoolBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, b.glObstaclePoolBuffer));
    // Binding 10: Color palette (optional, shader falls back to defaults if empty)
    if (s.colorPaletteSize > 0)
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, s.glColorPaletteBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, s.glCameraIntrinsicsBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, s.glCameraPoseBuffer));
    NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, s.glQueryBuffer));

    // For each primitive type, dispatch based on (query, pool) pairs
    // We use the total number of pools uploaded (worst case: one scene has all pools)
    // The shader early-exits if pool_id >= scene.num_pools for that scene

    uint32_t maxPolylinePools = std::max(1u, (uint32_t)b.maxPolylinePoolsPerScene);
    uint32_t maxPolygonPools = std::max(1u, (uint32_t)b.maxPolygonPoolsPerScene);
    // For obstacles, use the tracked max obstacles per pool
    uint32_t maxObstacles = std::max(1u, (uint32_t)b.maxObstaclesPerPool);

    // Render polylines
    // New dispatch model: num_queries * num_pools * max_varrays_per_pool
    // Each task shader handles ONE polyline (varray), like non-timestamped version
    const uint32_t MAX_VARRAYS_PER_POOL = 1000;  // Upper bound on varrays per pool per timestamp

    if (b.polylinePoolUsed > 0)
    {
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramPolyline));

        GLint locNumQueries = glGetUniformLocation(s.glProgramPolyline, "u_num_queries");
        GLint locNumPools = glGetUniformLocation(s.glProgramPolyline, "u_num_polyline_pools");
        GLint locMaxVarrays = glGetUniformLocation(s.glProgramPolyline, "u_max_varrays_per_pool");
        GLint locTess = glGetUniformLocation(s.glProgramPolyline, "u_tessellation_threshold");
        GLint locMaxTessPolyline = glGetUniformLocation(s.glProgramPolyline, "u_max_tessellation_polyline");
        GLint locPaletteSize = glGetUniformLocation(s.glProgramPolyline, "u_color_palette_size");

        if (locNumQueries >= 0) glUniform1ui(locNumQueries, (GLuint)numQueries);
        if (locNumPools >= 0) glUniform1ui(locNumPools, maxPolylinePools);
        if (locMaxVarrays >= 0) glUniform1ui(locMaxVarrays, MAX_VARRAYS_PER_POOL);
        if (locTess >= 0) glUniform1f(locTess, s.tessellationThreshold);
        if (locMaxTessPolyline >= 0) glUniform1ui(locMaxTessPolyline, (GLuint)s.maxTessellationLevelPolyline);
        if (locPaletteSize >= 0) glUniform1i(locPaletteSize, s.colorPaletteSize);

        // Set width uniforms
        GLint locWidthPolyReg = glGetUniformLocation(s.glProgramPolyline, "u_width_polyline_regular");
        GLint locWidthPolyBev = glGetUniformLocation(s.glProgramPolyline, "u_width_polyline_bev");
        GLint locWidthEgoReg = glGetUniformLocation(s.glProgramPolyline, "u_width_ego_traj_regular");
        GLint locWidthEgoBev = glGetUniformLocation(s.glProgramPolyline, "u_width_ego_traj_bev");
        GLint locWidthWire = glGetUniformLocation(s.glProgramPolyline, "u_width_wireframe");
        GLint locResScale = glGetUniformLocation(s.glProgramPolyline, "u_resolution_scale");
        if (locWidthPolyReg >= 0) glUniform1f(locWidthPolyReg, s.widthPolylineRegular);
        if (locWidthPolyBev >= 0) glUniform1f(locWidthPolyBev, s.widthPolylineBev);
        if (locWidthEgoReg >= 0) glUniform1f(locWidthEgoReg, s.widthEgoTrajRegular);
        if (locWidthEgoBev >= 0) glUniform1f(locWidthEgoBev, s.widthEgoTrajBev);
        if (locWidthWire >= 0) glUniform1f(locWidthWire, s.widthWireframe);
        if (locResScale >= 0) glUniform1f(locResScale, s.resolutionScale);

        // Set depth scaling for distance-based line width and fog
        GLint locDepthScale = glGetUniformLocation(s.glProgramPolyline, "u_depth_scaling");
        GLint locFogEnabled = glGetUniformLocation(s.glProgramPolyline, "u_fog_enabled");
        if (locDepthScale >= 0) glUniform1f(locDepthScale, s.depthScaling);
        if (locFogEnabled >= 0) glUniform1f(locFogEnabled, s.depthScaling);  // Same as depth scaling

        // Set spatial culling radius
        GLint locCullRadius = glGetUniformLocation(s.glProgramPolyline, "u_cull_radius_scale");
        if (locCullRadius >= 0) glUniform1f(locCullRadius, s.cullRadiusScale);

        if (s.enableZModify)
        {
            GLint locDummy = glGetUniformLocation(s.glProgramPolyline, "in_dummy");
            if (locDummy >= 0) glUniform1f(locDummy, 0.0f);
        }

        uint32_t numWorkgroups = numQueries * maxPolylinePools * MAX_VARRAYS_PER_POOL;
        NVDR_CHECK_GL_ERROR(glDrawMeshTasksNV(0, numWorkgroups));
    }

    // Render polygons
    if (b.polygonPoolUsed > 0)
    {
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramPolygon));

        GLint locNumQueries = glGetUniformLocation(s.glProgramPolygon, "u_num_queries");
        GLint locNumPools = glGetUniformLocation(s.glProgramPolygon, "u_num_polygon_pools");
        GLint locMaxVarrays = glGetUniformLocation(s.glProgramPolygon, "u_max_varrays_per_pool");
        GLint locTess = glGetUniformLocation(s.glProgramPolygon, "u_tessellation_threshold");
        GLint locMaxTessPolygon = glGetUniformLocation(s.glProgramPolygon, "u_max_tessellation_polygon");
        GLint locPaletteSize = glGetUniformLocation(s.glProgramPolygon, "u_color_palette_size");

        if (locNumQueries >= 0) glUniform1ui(locNumQueries, (GLuint)numQueries);
        if (locNumPools >= 0) glUniform1ui(locNumPools, maxPolygonPools);
        if (locMaxVarrays >= 0) glUniform1ui(locMaxVarrays, MAX_VARRAYS_PER_POOL);
        if (locTess >= 0) glUniform1f(locTess, s.tessellationThreshold);
        if (locMaxTessPolygon >= 0) glUniform1ui(locMaxTessPolygon, (GLuint)s.maxTessellationLevelPolygon);
        if (locPaletteSize >= 0) glUniform1i(locPaletteSize, s.colorPaletteSize);

        // Set resolution scale for polygon shader too
        GLint locResScale = glGetUniformLocation(s.glProgramPolygon, "u_resolution_scale");
        if (locResScale >= 0) glUniform1f(locResScale, s.resolutionScale);

        // Set fog for distance-based darkening
        GLint locFogEnabled = glGetUniformLocation(s.glProgramPolygon, "u_fog_enabled");
        if (locFogEnabled >= 0) glUniform1f(locFogEnabled, s.depthScaling);

        // Set spatial culling radius
        GLint locCullRadius = glGetUniformLocation(s.glProgramPolygon, "u_cull_radius_scale");
        if (locCullRadius >= 0) glUniform1f(locCullRadius, s.cullRadiusScale);

        if (s.enableZModify)
        {
            GLint locDummy = glGetUniformLocation(s.glProgramPolygon, "in_dummy");
            if (locDummy >= 0) glUniform1f(locDummy, 0.0f);
        }

        uint32_t numWorkgroups = numQueries * maxPolygonPools * MAX_VARRAYS_PER_POOL;
        NVDR_CHECK_GL_ERROR(glDrawMeshTasksNV(0, numWorkgroups));
    }

    // Render cubes (obstacles, traffic lights, traffic signs, etc.)
    // Loop through each cube pool index and dispatch separately
    if (b.obstaclePoolUsed > 0 && b.maxCubePoolsPerScene > 0)
    {
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramObstacle));

        GLint locNumQueries = glGetUniformLocation(s.glProgramObstacle, "u_num_queries");
        GLint locMaxObs = glGetUniformLocation(s.glProgramObstacle, "u_max_obstacles");
        GLint locPoolIndex = glGetUniformLocation(s.glProgramObstacle, "u_cube_pool_index");
        GLint locTess = glGetUniformLocation(s.glProgramObstacle, "u_tessellation_threshold");
        GLint locMaxTessCube = glGetUniformLocation(s.glProgramObstacle, "u_max_tessellation_cube");
        GLint locPaletteSize = glGetUniformLocation(s.glProgramObstacle, "u_color_palette_size");

        if (locNumQueries >= 0) glUniform1ui(locNumQueries, (GLuint)numQueries);
        if (locMaxObs >= 0) glUniform1ui(locMaxObs, maxObstacles);
        if (locTess >= 0) glUniform1f(locTess, s.tessellationThreshold);
        if (locMaxTessCube >= 0) glUniform1ui(locMaxTessCube, (GLuint)s.maxTessellationLevelCube);
        if (locPaletteSize >= 0) glUniform1i(locPaletteSize, s.colorPaletteSize);

        // Set wireframe width and resolution scale for cube shader
        GLint locWidthWire = glGetUniformLocation(s.glProgramObstacle, "u_width_wireframe");
        GLint locResScale = glGetUniformLocation(s.glProgramObstacle, "u_resolution_scale");
        if (locWidthWire >= 0) glUniform1f(locWidthWire, s.widthWireframe);
        if (locResScale >= 0) glUniform1f(locResScale, s.resolutionScale);

        // Set fog for distance-based darkening
        GLint locFogEnabled = glGetUniformLocation(s.glProgramObstacle, "u_fog_enabled");
        if (locFogEnabled >= 0) glUniform1f(locFogEnabled, s.depthScaling);

        // Set spatial culling radius
        GLint locCullRadius = glGetUniformLocation(s.glProgramObstacle, "u_cull_radius_scale");
        if (locCullRadius >= 0) glUniform1f(locCullRadius, s.cullRadiusScale);

        // Set max extrapolation time for obstacle visibility
        GLint locMaxExtrap = glGetUniformLocation(s.glProgramObstacle, "u_max_extrapolation_us");
        int maxExtrap = (s.maxExtrapolationUs > 0) ? s.maxExtrapolationUs : 500000;  // Default 500ms
        if (locMaxExtrap >= 0) glUniform1i(locMaxExtrap, maxExtrap);

        if (s.enableZModify)
        {
            GLint locDummy = glGetUniformLocation(s.glProgramObstacle, "in_dummy");
            if (locDummy >= 0) glUniform1f(locDummy, 0.0f);
        }

        uint32_t numWorkgroups = numQueries * maxObstacles;

        // Dispatch for each cube pool (task shader skips if pool index out of range for scene)
        for (int poolIdx = 0; poolIdx < b.maxCubePoolsPerScene; poolIdx++)
        {
            if (locPoolIndex >= 0) glUniform1ui(locPoolIndex, (GLuint)poolIdx);
            NVDR_CHECK_GL_ERROR(glDrawMeshTasksNV(0, numWorkgroups));
        }
    }

    // MSAA resolve: blit from MSAA FBO to regular FBO for CUDA readback
    if (s.msaaSamples >= 2)
    {
        // For MSAA resolve, we need to blit each layer separately
        // Detach depth/stencil to simplify framebuffer requirements for color-only blit
        NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_READ_FRAMEBUFFER, s.glFBO_MSAA));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 0, 0));
        NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, s.glFBO));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 0, 0));

        // Blit each layer separately (glBlitFramebuffer only works on 2D regions)
        for (int layer = 0; layer < numQueries; layer++)
        {
            // Attach single layer from MSAA source
            NVDR_CHECK_GL_ERROR(glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                          s.glColorBuffer_MSAA, 0, layer));

            // Attach single layer to resolve target
            NVDR_CHECK_GL_ERROR(glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                          s.glColorBuffer, 0, layer));

            // Blit (resolve) with nearest filter (MSAA resolve uses GL_NEAREST)
            NVDR_CHECK_GL_ERROR(glBlitFramebuffer(0, 0, width, height,
                                                   0, 0, width, height,
                                                   GL_COLOR_BUFFER_BIT, GL_NEAREST));
        }

        // Restore full texture attachments for next frame
        NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer, 0));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, s.glDepthStencilBuffer, 0));
        NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO_MSAA));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer_MSAA, 0));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, s.glDepthStencilBuffer_MSAA, 0));
    }

    // Use glFinish() to ensure all GL commands complete before CUDA reads the buffer.
    // This is necessary for correct GL→CUDA synchronization when encoding JPEGs.
    glFinish();
}

void ludusCopyBatchResults(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    uint8_t* outputPtr,
    int width,
    int height,
    int numQueries)
{
    cudaArray_t array = 0;

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaColorBuffer, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, s.cudaColorBuffer, 0, 0));

    cudaMemcpy3DParms p = {0};
    p.srcArray = array;
    p.dstPtr.ptr = outputPtr;
    p.dstPtr.pitch = width * 4 * sizeof(uint8_t);  // RGBA8 = 4 bytes per pixel
    p.dstPtr.xsize = width;
    p.dstPtr.ysize = height;
    p.extent.width = width;
    p.extent.height = height;
    p.extent.depth = numQueries;
    p.kind = cudaMemcpyDeviceToDevice;
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy3DAsync(&p, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaColorBuffer, stream));
}

// Copy rendered results to staging buffer (double buffer ping-pong)
// Returns the index of the staging buffer that now has the new data
int ludusCopyBatchResultsToStaging(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    cudaStream_t stream,
    int width,
    int height,
    int numQueries)
{
    size_t requiredSize = (size_t)width * height * numQueries * 4;  // RGBA8

    // Reallocate staging buffers if needed
    if (requiredSize > s.stagingBufferSize)
    {
        // Free old buffers
        if (s.stagingBuffer[0]) { cudaFree(s.stagingBuffer[0]); s.stagingBuffer[0] = nullptr; }
        if (s.stagingBuffer[1]) { cudaFree(s.stagingBuffer[1]); s.stagingBuffer[1] = nullptr; }

        // Allocate new buffers with some headroom
        size_t allocSize = requiredSize + (requiredSize >> 2);  // 25% headroom
        NVDR_CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&s.stagingBuffer[0]), allocSize));
        NVDR_CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&s.stagingBuffer[1]), allocSize));
        s.stagingBufferSize = allocSize;
        s.stagingValid[0] = 0;
        s.stagingValid[1] = 0;
        LOG(INFO) << "Ludus: Allocated staging buffers: " << (allocSize / (1024*1024)) << " MB each";
    }

    int writeIdx = s.currentStagingIdx;
    uint8_t* dstPtr = s.stagingBuffer[writeIdx];

    // Copy from GL texture to staging buffer
    cudaArray_t array = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaColorBuffer, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, s.cudaColorBuffer, 0, 0));

    cudaMemcpy3DParms p = {0};
    p.srcArray = array;
    p.dstPtr.ptr = dstPtr;
    p.dstPtr.pitch = width * 4 * sizeof(uint8_t);
    p.dstPtr.xsize = width;
    p.dstPtr.ysize = height;
    p.extent.width = width;
    p.extent.height = height;
    p.extent.depth = numQueries;
    p.kind = cudaMemcpyDeviceToDevice;
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy3DAsync(&p, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaColorBuffer, stream));

    // Record event so we know when this staging buffer is ready
    NVDR_CHECK_CUDA_ERROR(cudaEventRecord(s.stagingReadyEvent[writeIdx], stream));

    // Mark this buffer as valid and store dimensions
    s.stagingValid[writeIdx] = 1;
    s.stagingWidth = width;
    s.stagingHeight = height;
    s.stagingNumQueries = numQueries;

    // Swap to other buffer for next frame
    s.currentStagingIdx = 1 - writeIdx;

    return writeIdx;
}

// Get the staging buffer that's ready for CPU copy (the one NOT being written to)
// Returns nullptr if no valid data, otherwise returns pointer and dimensions
uint8_t* ludusGetReadyStagingBuffer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int* outWidth,
    int* outHeight,
    int* outNumQueries)
{
    int readIdx = 1 - s.currentStagingIdx;  // The one we're not writing to

    if (!s.stagingValid[readIdx])
    {
        *outWidth = 0;
        *outHeight = 0;
        *outNumQueries = 0;
        return nullptr;
    }

    // Wait for the staging buffer to be ready
    NVDR_CHECK_CUDA_ERROR(cudaEventSynchronize(s.stagingReadyEvent[readIdx]));

    *outWidth = s.stagingWidth;
    *outHeight = s.stagingHeight;
    *outNumQueries = s.stagingNumQueries;

    return s.stagingBuffer[readIdx];
}

// Wait for staging buffer to be ready and copy to output (sync version)
void ludusCopyStagingToOutput(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx,
    uint8_t* outputPtr,
    int width,
    int height,
    int numQueries)
{
    // Wait for staging to be ready
    NVDR_CHECK_CUDA_ERROR(cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]));

    // Copy staging to output
    size_t size = (size_t)width * height * numQueries * 4;
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(outputPtr, s.stagingBuffer[stagingIdx], size, cudaMemcpyDeviceToDevice));
}

// Start async D2H transfer from staging buffer to pinned host memory
// This runs on copyStream and can overlap with rendering on the main stream
// Returns 1 if transfer was started, 0 if no valid data or transfer already pending
// Start async D2H transfer from staging buffer to pinned host buffer (double-buffered)
// Returns the pinned buffer index that will receive the data, or -1 on failure
int ludusStartAsyncHostTransfer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx)
{
    // Check if staging buffer has valid data
    if (!s.stagingValid[stagingIdx] || !s.stagingBuffer[stagingIdx])
    {
        return -1;
    }

    int width = s.stagingWidth;
    int height = s.stagingHeight;
    int numQueries = s.stagingNumQueries;
    size_t requiredSize = (size_t)width * height * numQueries * 4;

    // Reallocate pinned host buffers if needed
    if (requiredSize > s.pinnedHostBufferSize)
    {
        // Free old buffers
        if (s.pinnedHostBuffer[0]) { cudaFreeHost(s.pinnedHostBuffer[0]); s.pinnedHostBuffer[0] = nullptr; }
        if (s.pinnedHostBuffer[1]) { cudaFreeHost(s.pinnedHostBuffer[1]); s.pinnedHostBuffer[1] = nullptr; }

        size_t allocSize = requiredSize + (requiredSize >> 2);  // 25% headroom
        NVDR_CHECK_CUDA_ERROR(cudaHostAlloc(reinterpret_cast<void**>(&s.pinnedHostBuffer[0]),
                                             allocSize, cudaHostAllocDefault));
        NVDR_CHECK_CUDA_ERROR(cudaHostAlloc(reinterpret_cast<void**>(&s.pinnedHostBuffer[1]),
                                             allocSize, cudaHostAllocDefault));
        s.pinnedHostBufferSize = allocSize;
        s.pinnedValid[0] = 0;
        s.pinnedValid[1] = 0;
        LOG(INFO) << "Ludus: Allocated pinned host buffers: " << (allocSize / (1024*1024)) << " MB each";
    }

    int writeIdx = s.currentPinnedIdx;
    uint8_t* dstPtr = s.pinnedHostBuffer[writeIdx];

    // Wait for staging buffer to be ready (on copyStream)
    // This makes copyStream wait for the render stream's staging write to complete
    NVDR_CHECK_CUDA_ERROR(cudaStreamWaitEvent(s.copyStream, s.stagingReadyEvent[stagingIdx], 0));

    // Start async D2H copy on copyStream
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(dstPtr, s.stagingBuffer[stagingIdx],
                                           requiredSize, cudaMemcpyDeviceToHost, s.copyStream));

    // Record event when D2H is done
    NVDR_CHECK_CUDA_ERROR(cudaEventRecord(s.pinnedReadyEvent[writeIdx], s.copyStream));

    // Mark this pinned buffer as valid and store dimensions
    s.pinnedValid[writeIdx] = 1;
    s.pinnedWidth[writeIdx] = width;
    s.pinnedHeight[writeIdx] = height;
    s.pinnedNumQueries[writeIdx] = numQueries;

    // Swap to other buffer for next transfer
    s.currentPinnedIdx = 1 - writeIdx;

    return writeIdx;
}

// Check if a specific pinned buffer is ready (non-blocking)
// Returns 1 if complete, 0 if still in progress or invalid
int ludusIsPinnedBufferReady(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int pinnedIdx)
{
    if (!s.pinnedValid[pinnedIdx])
    {
        return 0;  // No valid data
    }

    cudaError_t status = cudaEventQuery(s.pinnedReadyEvent[pinnedIdx]);
    if (status == cudaSuccess)
    {
        return 1;  // Complete
    }
    else if (status == cudaErrorNotReady)
    {
        return 0;  // Still in progress
    }
    else
    {
        NVDR_CHECK_CUDA_ERROR(status);  // Real error
        return 0;
    }
}

// Check if any host transfer is complete (legacy API compatibility)
int ludusIsHostTransferComplete(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s)
{
    // Check the buffer that's NOT currently being written to
    int readIdx = 1 - s.currentPinnedIdx;
    return ludusIsPinnedBufferReady(NVDR_CTX_PARAMS, s, readIdx);
}

// Wait for a specific pinned buffer and get its data
// Returns pointer to pinned host buffer with dimensions, or nullptr if invalid
uint8_t* ludusWaitPinnedBuffer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int pinnedIdx,
    int* outWidth,
    int* outHeight,
    int* outNumQueries)
{
    if (!s.pinnedValid[pinnedIdx] || !s.pinnedHostBuffer[pinnedIdx])
    {
        *outWidth = 0;
        *outHeight = 0;
        *outNumQueries = 0;
        return nullptr;
    }

    // Wait for transfer to complete
    NVDR_CHECK_CUDA_ERROR(cudaEventSynchronize(s.pinnedReadyEvent[pinnedIdx]));

    // Return data
    *outWidth = s.pinnedWidth[pinnedIdx];
    *outHeight = s.pinnedHeight[pinnedIdx];
    *outNumQueries = s.pinnedNumQueries[pinnedIdx];

    return s.pinnedHostBuffer[pinnedIdx];
}

// Legacy API: Wait for the previous pinned buffer (the one not being written to)
uint8_t* ludusWaitHostTransfer(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int* outWidth,
    int* outHeight,
    int* outNumQueries)
{
    // Get the buffer that's NOT currently being written to
    int readIdx = 1 - s.currentPinnedIdx;
    return ludusWaitPinnedBuffer(NVDR_CTX_PARAMS, s, readIdx, outWidth, outHeight, outNumQueries);
}

// ========== NVJPEG Hardware Encoding ==========

// Encode a single image from GPU memory to JPEG using hardware encoder
// Returns compressed JPEG data size, or 0 on failure
// The compressed data is stored in s.jpegOutputBuffer
size_t ludusEncodeJpegGpu(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    const uint8_t* gpuRgba,  // GPU pointer to RGBA data
    int width,
    int height,
    int quality,             // JPEG quality 1-100
    uint8_t** outJpegData)   // Output: pointer to compressed data
{
    if (!s.nvjpegInitialized)
    {
        LOG(WARNING) << "NVJPEG not initialized";
        return 0;
    }

    size_t dstRowBytes = (size_t)width * 3;  // RGB output (NVJPEG expects 3 bytes/pixel)
    size_t rgbImageSize = (size_t)width * height * 3;

    // Allocate buffer for RGB data (flipped, alpha-stripped)
    if (rgbImageSize > s.jpegFlipBufferSize)
    {
        if (s.jpegFlipBuffer) cudaFree(s.jpegFlipBuffer);
        size_t allocSize = rgbImageSize + (rgbImageSize >> 2);  // 25% headroom
        NVDR_CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&s.jpegFlipBuffer), allocSize));
        s.jpegFlipBufferSize = allocSize;
    }

    // Use copyStream for encoding (allows overlap with main render stream)
    cudaStream_t encodeStream = s.copyStream;

    // Convert RGBA to RGB with vertical flip using GPU kernel
    launchRgbaToRgbFlip(gpuRgba, s.jpegFlipBuffer, width, height, encodeStream);
    NVDR_CHECK_CUDA_ERROR(cudaGetLastError());

    // Set quality and sampling
    nvjpegEncoderParamsSetQuality(s.nvjpegEncoderParams, quality, encodeStream);
    nvjpegEncoderParamsSetSamplingFactors(s.nvjpegEncoderParams, NVJPEG_CSS_420, encodeStream);

    // Setup input image using RGB buffer
    nvjpegImage_t nvImage;
    memset(&nvImage, 0, sizeof(nvImage));
    nvImage.channel[0] = s.jpegFlipBuffer;
    nvImage.pitch[0] = dstRowBytes;  // RGB stride (3 bytes per pixel)

    // Encode using RGBI - matches official NVIDIA sample pattern
    // Note: nvjpegEncodeImage is async, will be queued on encodeStream
    nvjpegStatus_t status = nvjpegEncodeImage(
        s.nvjpegHandle,
        s.nvjpegEncoderState,  // Reuse cached state
        s.nvjpegEncoderParams,
        &nvImage,
        NVJPEG_INPUT_RGBI,
        width,
        height,
        encodeStream
    );

    if (status != NVJPEG_STATUS_SUCCESS)
    {
        LOG(WARNING) << "NVJPEG encode failed with status " << status;
        return 0;
    }

    // Get compressed size (async, queued on stream)
    size_t compressedSize = 0;
    status = nvjpegEncodeRetrieveBitstream(
        s.nvjpegHandle,
        s.nvjpegEncoderState,
        nullptr,
        &compressedSize,
        encodeStream
    );

    if (status != NVJPEG_STATUS_SUCCESS || compressedSize == 0)
    {
        LOG(WARNING) << "NVJPEG failed to get bitstream size";
        return 0;
    }

    // NVIDIA pattern: Sync stream AFTER first RetrieveBitstream, BEFORE getting data
    NVDR_CHECK_CUDA_ERROR(cudaStreamSynchronize(encodeStream));

    // Reallocate output buffer if needed
    if (compressedSize > s.jpegOutputBufferSize)
    {
        if (s.jpegOutputBuffer) cudaFreeHost(s.jpegOutputBuffer);
        size_t allocSize = compressedSize + (compressedSize >> 2);  // 25% headroom
        NVDR_CHECK_CUDA_ERROR(cudaHostAlloc(reinterpret_cast<void**>(&s.jpegOutputBuffer),
                                             allocSize, cudaHostAllocDefault));
        s.jpegOutputBufferSize = allocSize;
    }

    // Retrieve compressed data - NVIDIA sample uses stream=0 for this call
    status = nvjpegEncodeRetrieveBitstream(
        s.nvjpegHandle,
        s.nvjpegEncoderState,
        s.jpegOutputBuffer,
        &compressedSize,
        0  // NVIDIA pattern: use default stream for data retrieval
    );

    if (status != NVJPEG_STATUS_SUCCESS)
    {
        LOG(WARNING) << "NVJPEG failed to retrieve bitstream";
        return 0;
    }

    // Final sync to ensure data is ready
    NVDR_CHECK_CUDA_ERROR(cudaStreamSynchronize(0));

    *outJpegData = s.jpegOutputBuffer;
    return compressedSize;
}

// Encode a single image from pinned host buffer to JPEG
// This first copies to GPU, then encodes
size_t ludusEncodeJpegPinned(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int pinnedIdx,
    int imageIdx,            // Which image in the batch (0 to numQueries-1)
    int quality,
    uint8_t** outJpegData)
{
    if (!s.pinnedValid[pinnedIdx] || !s.pinnedHostBuffer[pinnedIdx])
    {
        return 0;
    }

    // Wait for pinned buffer to be ready
    NVDR_CHECK_CUDA_ERROR(cudaEventSynchronize(s.pinnedReadyEvent[pinnedIdx]));

    int width = s.pinnedWidth[pinnedIdx];
    int height = s.pinnedHeight[pinnedIdx];
    int numQueries = s.pinnedNumQueries[pinnedIdx];

    if (imageIdx < 0 || imageIdx >= numQueries)
    {
        return 0;
    }

    // Get pointer to the specific image in the pinned buffer
    size_t imageSize = (size_t)width * height * 4;
    uint8_t* hostPtr = s.pinnedHostBuffer[pinnedIdx] + imageIdx * imageSize;

    // Need to copy to GPU for nvjpeg - use staging buffer temporarily
    // Actually, nvjpeg can work from host memory too with certain backends
    // But for hardware encoding, we need GPU memory

    // Use staging buffer[0] as temp GPU buffer for encoding
    if (!s.stagingBuffer[0] || s.stagingBufferSize < imageSize)
    {
        LOG(WARNING) << "Staging buffer too small for JPEG encoding";
        return 0;
    }

    // Copy single image from pinned to GPU
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(s.stagingBuffer[0], hostPtr, imageSize, cudaMemcpyHostToDevice));

    return ludusEncodeJpegGpu(NVDR_CTX_PARAMS, s, s.stagingBuffer[0], width, height, quality, outJpegData);
}

// Encode directly from staging buffer (no extra copy needed)
size_t ludusEncodeJpegStaging(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx,
    int imageIdx,            // Which image in the batch
    int quality,
    uint8_t** outJpegData)
{
    if (!s.stagingValid[stagingIdx] || !s.stagingBuffer[stagingIdx])
    {
        return 0;
    }

    // Full device sync to ensure staging buffer is ready
    // This is the most robust synchronization - ensures all GPU work is complete
    NVDR_CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int width = s.stagingWidth;
    int height = s.stagingHeight;
    int numQueries = s.stagingNumQueries;

    if (imageIdx < 0 || imageIdx >= numQueries)
    {
        return 0;
    }

    // Get pointer to the specific image in staging buffer
    size_t imageSize = (size_t)width * height * 4;
    uint8_t* gpuPtr = s.stagingBuffer[stagingIdx] + imageIdx * imageSize;

    return ludusEncodeJpegGpu(NVDR_CTX_PARAMS, s, gpuPtr, width, height, quality, outJpegData);
}

// Batch encode all images from staging buffer
// Returns vector of (data_ptr, size) pairs
int ludusEncodeJpegBatchStaging(
    NVDR_CTX_ARGS,
    LudusTimestampedState& s,
    int stagingIdx,
    int quality,
    std::vector<std::pair<uint8_t*, size_t>>& outJpegs)
{
    if (!s.stagingValid[stagingIdx] || !s.stagingBuffer[stagingIdx])
    {
        return 0;
    }

    // Wait for staging buffer to be ready AND ensure all prior GPU work is done
    NVDR_CHECK_CUDA_ERROR(cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]));
    NVDR_CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // Full device sync to catch any race conditions

    int width = s.stagingWidth;
    int height = s.stagingHeight;
    int numQueries = s.stagingNumQueries;
    size_t imageSize = (size_t)width * height * 4;

    outJpegs.clear();
    outJpegs.reserve(numQueries);

    for (int i = 0; i < numQueries; i++)
    {
        uint8_t* gpuPtr = s.stagingBuffer[stagingIdx] + i * imageSize;
        uint8_t* jpegData = nullptr;
        size_t jpegSize = ludusEncodeJpegGpu(NVDR_CTX_PARAMS, s, gpuPtr, width, height, quality, &jpegData);

        if (jpegSize > 0)
        {
            // Copy to new buffer since jpegOutputBuffer is reused
            uint8_t* copy = new uint8_t[jpegSize];
            memcpy(copy, jpegData, jpegSize);
            outJpegs.push_back({copy, jpegSize});
        }
        else
        {
            outJpegs.push_back({nullptr, 0});
        }
    }

    return numQueries;
}

// ========== End NVJPEG ==========

void ludusClearScenes(NVDR_CTX_ARGS, LudusTimestampedState& s)
{
    auto& b = s.bufferSets[s.activeSet];
    b.numScenes = 0;
    b.timestampsUsed = 0;
    b.int32Used = 0;
    b.vertexUsed = 0;
    b.triangleUsed = 0;
    b.poseUsed = 0;
    b.floatUsed = 0;
    b.polylinePoolUsed = 0;
    b.polygonPoolUsed = 0;
    b.obstaclePoolUsed = 0;
    b.maxObstaclesPerPool = 0;
    b.maxCubePoolsPerScene = 0;
    b.maxPolylinePoolsPerScene = 0;
    b.maxPolygonPoolsPerScene = 0;
}

void ludusSwapBufferSets(NVDR_CTX_ARGS, LudusTimestampedState& s)
{
    setGLContext(s.glctx);

    // Wait for any pending fence on the back set before swapping
    auto& back = s.bufferSets[1 - s.activeSet];
    if (back.glFence)
    {
        glClientWaitSync(back.glFence, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        glDeleteSync(back.glFence);
        back.glFence = nullptr;
    }

    s.activeSet = 1 - s.activeSet;
    LOG(INFO) << "Ludus Timestamped: Swapped to buffer set " << s.activeSet;
}

void ludusTimestampedRelease(NVDR_CTX_ARGS, LudusTimestampedState& s)
{
    // Free double buffer resources
    if (s.stagingBuffer[0]) cudaFree(s.stagingBuffer[0]);
    if (s.stagingBuffer[1]) cudaFree(s.stagingBuffer[1]);
    if (s.copyStream) cudaStreamDestroy(s.copyStream);
    if (s.stagingReadyEvent[0]) cudaEventDestroy(s.stagingReadyEvent[0]);
    if (s.stagingReadyEvent[1]) cudaEventDestroy(s.stagingReadyEvent[1]);

    // Free double-buffered pinned host memory
    if (s.pinnedHostBuffer[0]) cudaFreeHost(s.pinnedHostBuffer[0]);
    if (s.pinnedHostBuffer[1]) cudaFreeHost(s.pinnedHostBuffer[1]);
    if (s.pinnedReadyEvent[0]) cudaEventDestroy(s.pinnedReadyEvent[0]);
    if (s.pinnedReadyEvent[1]) cudaEventDestroy(s.pinnedReadyEvent[1]);

    // Free NVJPEG resources
    if (s.jpegOutputBuffer) cudaFreeHost(s.jpegOutputBuffer);
    if (s.jpegFlipBuffer) cudaFree(s.jpegFlipBuffer);
    if (s.nvjpegEncoderParams) nvjpegEncoderParamsDestroy(s.nvjpegEncoderParams);
    if (s.nvjpegEncoderState) nvjpegEncoderStateDestroy(s.nvjpegEncoderState);
    if (s.nvjpegHandle) nvjpegDestroy(s.nvjpegHandle);

    // Unregister CUDA resources for both buffer sets
    for (int i = 0; i < 2; i++)
    {
        auto& b = s.bufferSets[i];
        if (b.glFence) { glDeleteSync(b.glFence); b.glFence = nullptr; }
        if (b.cudaSceneBuffer) cudaGraphicsUnregisterResource(b.cudaSceneBuffer);
        if (b.cudaTimestampsBuffer) cudaGraphicsUnregisterResource(b.cudaTimestampsBuffer);
        if (b.cudaInt32Buffer) cudaGraphicsUnregisterResource(b.cudaInt32Buffer);
        if (b.cudaVertexBuffer) cudaGraphicsUnregisterResource(b.cudaVertexBuffer);
        if (b.cudaTriangleBuffer) cudaGraphicsUnregisterResource(b.cudaTriangleBuffer);
        if (b.cudaPoseBuffer) cudaGraphicsUnregisterResource(b.cudaPoseBuffer);
        if (b.cudaFloatBuffer) cudaGraphicsUnregisterResource(b.cudaFloatBuffer);
        if (b.cudaPolylinePoolBuffer) cudaGraphicsUnregisterResource(b.cudaPolylinePoolBuffer);
        if (b.cudaPolygonPoolBuffer) cudaGraphicsUnregisterResource(b.cudaPolygonPoolBuffer);
        if (b.cudaObstaclePoolBuffer) cudaGraphicsUnregisterResource(b.cudaObstaclePoolBuffer);
    }

    // Unregister non-scene CUDA resources
    if (s.cudaColorBuffer) cudaGraphicsUnregisterResource(s.cudaColorBuffer);
    if (s.cudaColorPaletteBuffer) cudaGraphicsUnregisterResource(s.cudaColorPaletteBuffer);
    if (s.cudaCameraIntrinsicsBuffer) cudaGraphicsUnregisterResource(s.cudaCameraIntrinsicsBuffer);
    if (s.cudaCameraPoseBuffer) cudaGraphicsUnregisterResource(s.cudaCameraPoseBuffer);
    if (s.cudaQueryBuffer) cudaGraphicsUnregisterResource(s.cudaQueryBuffer);

    // Delete GL programs and shaders (these extension functions are available)
    if (s.glProgramPolyline) glDeleteProgram(s.glProgramPolyline);
    if (s.glProgramPolygon) glDeleteProgram(s.glProgramPolygon);
    if (s.glProgramObstacle) glDeleteProgram(s.glProgramObstacle);
    if (s.glTaskShaderPolyline) glDeleteShader(s.glTaskShaderPolyline);
    if (s.glMeshShaderPolyline) glDeleteShader(s.glMeshShaderPolyline);
    if (s.glTaskShaderPolygon) glDeleteShader(s.glTaskShaderPolygon);
    if (s.glMeshShaderPolygon) glDeleteShader(s.glMeshShaderPolygon);
    if (s.glTaskShaderObstacle) glDeleteShader(s.glTaskShaderObstacle);
    if (s.glMeshShaderObstacle) glDeleteShader(s.glMeshShaderObstacle);
    if (s.glFragmentShader) glDeleteShader(s.glFragmentShader);
    if (s.glFragmentShaderPolyline) glDeleteShader(s.glFragmentShaderPolyline);
    if (s.glFragmentShaderObstacle) glDeleteShader(s.glFragmentShaderObstacle);

    // GL buffers, textures, and FBO are cleaned up when GL context is destroyed
    // (glDeleteBuffers, glDeleteTextures, glDeleteFramebuffers not available as extensions)

    // Zero out state (GL context cleanup handled by wrapper)
    memset(&s, 0, sizeof(s));
}

//=============================================================================
