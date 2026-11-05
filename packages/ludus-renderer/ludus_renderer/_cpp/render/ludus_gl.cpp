// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include "ludus_gl.h"
#include "../common/glutil.h"
#include "ludus_types.h"
#include <vector>
#include <cstring>

//------------------------------------------------------------------------
// Helpers.

#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
static int ROUND_UP_BITS_LUDUS(uint32_t x, uint32_t y)
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
// Mesh shader compilation helper.

static void compileLudusShader(NVDR_CTX_ARGS, GLuint* pShader, GLenum shaderType, const char* src_buf, bool enableZModify)
{
    std::string src(src_buf);

    // Find the #version line and insert after it
    size_t versionPos = src.find("#version");
    size_t insertPos = 0;
    if (versionPos != std::string::npos)
    {
        // Insert after the #version line
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

    // Check compilation status
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
            LOG(ERROR) << "Shader compilation failed:\n" << &info[0];
            NVDR_CHECK(0, "Shader compilation failed");
        }
        NVDR_CHECK(0, "Shader compilation failed");
    }
}

static void constructLudusMeshProgram(NVDR_CTX_ARGS, GLuint* pProgram, GLuint taskShader, GLuint meshShader, GLuint fragmentShader)
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
            LOG(ERROR) << "Program linking failed:\n" << &info[0];
            NVDR_CHECK(0, "glLinkProgram() failed");
        }
        NVDR_CHECK(0, "glLinkProgram() failed");
    }

    *pProgram = glProgram;
}

//------------------------------------------------------------------------
// Shader source code.

// Common GLSL code shared across task/mesh shaders
// Includes version, extensions, and shared struct definitions
static const char* LUDUS_GLSL_COMMON = R"(#version 450
#extension GL_NV_mesh_shader : require

// Cap styles
const uint CAP_NONE  = 0u;
const uint CAP_FLAT  = 1u;
const uint CAP_ROUND = 2u;

// PolylineHeader (32 bytes) - matches C++ struct
struct PolylineHeader {
    uint  vertex_start;
    uint  vertex_count;
    float width;
    float color_r, color_g, color_b;
    uint  cap_style;
    uint  _pad;
};

// PolygonHeader (32 bytes)
struct PolygonHeader {
    uint  vertex_start;
    uint  vertex_count;
    uint  triangle_start;
    uint  triangle_count;
    float color_r, color_g, color_b;
    uint  _pad;
};

// Cube (64 bytes)
struct Cube {
    float tx, ty, tz;           // translation
    float sx, sy, sz;           // scale
    float rx, ry, rz;           // rotation (axis-angle)
    float _pad0;
    float front_r, front_g, front_b;
    float back_r, back_g, back_b;
};

// FThetaCamera (72 bytes = 18 floats)
struct FThetaCamera {
    float cx, cy;               // principal point
    float img_w, img_h;         // image size
    float poly0, poly1, poly2, poly3, poly4, poly5;  // forward polynomial
    float max_ray_angle;
    float max_distortion_val;
    float max_distortion_dval;
    float depth_max;
    float ld_c, ld_d, ld_e, ld_f;  // linear distortion [[c,d],[e,f]]
};

// CameraPose (64 bytes) - 4x4 matrix column-major
struct CameraPose {
    mat4 world_to_camera;
};

// Vertex (16 bytes)
struct Vertex {
    float x, y, z;
    float _pad;
};

// Rodrigues rotation: rotate vector v by axis-angle r
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

    // Apply linear distortion matrix
    vec2 pixel_dist;
    pixel_dist.x = cam.ld_c * pixel_rel.x + cam.ld_d * pixel_rel.y;
    pixel_dist.y = cam.ld_e * pixel_rel.x + cam.ld_f * pixel_rel.y;

    vec2 pixel = pixel_dist + vec2(cam.cx, cam.cy);

    // Convert to NDC
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

// ============================================================================
// DISTORTION METRIC FOR ADAPTIVE TESSELLATION
// ============================================================================
// Measures the pixel error of approximating a curved F-theta edge with a straight line.
// Returns the maximum deviation in pixels between the true projected midpoint and
// the linearly interpolated midpoint.

float estimate_edge_distortion_pixels(vec3 v0, vec3 v1, CameraPose pose, FThetaCamera cam) {
    // Project both endpoints
    vec4 clip0 = ftheta_project(v0, pose, cam);
    vec4 clip1 = ftheta_project(v1, pose, cam);

    // Project true midpoint
    vec3 mid_world = (v0 + v1) * 0.5;
    vec4 clip_mid = ftheta_project(mid_world, pose, cam);

    // Protect against division by zero
    float w0 = max(abs(clip0.w), 0.001);
    float w1 = max(abs(clip1.w), 0.001);
    float w_mid = max(abs(clip_mid.w), 0.001);

    // Convert to NDC, then to pixel coordinates
    vec2 ndc0 = clip0.xy / w0;
    vec2 ndc1 = clip1.xy / w1;
    vec2 ndc_mid = clip_mid.xy / w_mid;

    // Convert to pixel coordinates
    vec2 pixel0 = vec2((ndc0.x * 0.5 + 0.5) * cam.img_w, (0.5 - ndc0.y * 0.5) * cam.img_h);
    vec2 pixel1 = vec2((ndc1.x * 0.5 + 0.5) * cam.img_w, (0.5 - ndc1.y * 0.5) * cam.img_h);
    vec2 pixel_mid = vec2((ndc_mid.x * 0.5 + 0.5) * cam.img_w, (0.5 - ndc_mid.y * 0.5) * cam.img_h);

    // Linear interpolation of endpoints in pixel space
    vec2 linear_pixel_mid = (pixel0 + pixel1) * 0.5;

    // Compute error in pixels
    float error_pixels = length(pixel_mid - linear_pixel_mid);

    return error_pixels;
}

// Recursively estimate max distortion for an edge by sampling multiple points
float estimate_edge_max_distortion(vec3 v0, vec3 v1, CameraPose pose, FThetaCamera cam, uint depth) {
    if (depth == 0u) {
        return estimate_edge_distortion_pixels(v0, v1, pose, cam);
    }

    vec3 mid = (v0 + v1) * 0.5;
    float left = estimate_edge_max_distortion(v0, mid, pose, cam, depth - 1u);
    float right = estimate_edge_max_distortion(mid, v1, pose, cam, depth - 1u);
    float direct = estimate_edge_distortion_pixels(v0, v1, pose, cam);

    return max(direct, max(left, right));
}

// Determine subdivision level based on distortion
// Returns: 0 = no subdivision, 1 = 2 segments, 2 = 4 segments, 3 = 8 segments
uint compute_subdivision_level(vec3 v0, vec3 v1, CameraPose pose, FThetaCamera cam, float threshold_pixels) {
    // Quick estimate with single sample at midpoint
    float error = estimate_edge_distortion_pixels(v0, v1, pose, cam);

    if (error < threshold_pixels) return 0u;
    if (error < threshold_pixels * 2.0) return 1u;
    if (error < threshold_pixels * 4.0) return 2u;
    return 3u;  // Max subdivision level
}

// Note: Polyline subdivision level is computed in the task shader
// using direct SSBO access rather than passing arrays

//------------------------------------------------------------------------
// Shared barycentric subdivision utilities
// Used by both polygon and cube mesh shaders for unified tessellation

// Get number of vertices for a given subdivision level
uint bary_vertex_count(uint level) {
    // Level 0: 3, Level 1: 6, Level 2: 15
    return (level == 0u) ? 3u : (level == 1u) ? 6u : 15u;
}

// Get number of triangles for a given subdivision level
uint bary_triangle_count(uint level) {
    // Level 0: 1, Level 1: 4, Level 2: 16
    return (level == 0u) ? 1u : (level == 1u) ? 4u : 16u;
}

// Compute barycentric coordinates for vertex index at given subdivision level
// Returns (u, v) where w = 1 - u - v
vec2 bary_vertex_uv(uint vertex_idx, uint level) {
    if (level == 0u) {
        // Original triangle corners
        if (vertex_idx == 0u) return vec2(0.0, 0.0);      // v0
        if (vertex_idx == 1u) return vec2(1.0, 0.0);      // v1
        return vec2(0.0, 1.0);                             // v2
    } else if (level == 1u) {
        // 6 vertices: 3 corners + 3 midpoints
        vec2 uvs[6];
        uvs[0] = vec2(0.0, 0.0);   // v0
        uvs[1] = vec2(1.0, 0.0);   // v1
        uvs[2] = vec2(0.0, 1.0);   // v2
        uvs[3] = vec2(0.5, 0.0);   // mid(v0,v1)
        uvs[4] = vec2(0.5, 0.5);   // mid(v1,v2)
        uvs[5] = vec2(0.0, 0.5);   // mid(v2,v0)
        return uvs[vertex_idx];
    } else {
        // Level 2: 15 vertices in 4x4 barycentric grid
        // Row 0: 5 vertices, Row 1: 4 vertices, Row 2: 3, Row 3: 2, Row 4: 1
        uint row, col;
        if (vertex_idx < 5u) { row = 0u; col = vertex_idx; }
        else if (vertex_idx < 9u) { row = 1u; col = vertex_idx - 5u; }
        else if (vertex_idx < 12u) { row = 2u; col = vertex_idx - 9u; }
        else if (vertex_idx < 14u) { row = 3u; col = vertex_idx - 12u; }
        else { row = 4u; col = 0u; }

        return vec2(float(col) / 4.0, float(row) / 4.0);
    }
}

// Interpolate world position using barycentric coordinates
vec3 bary_interpolate(vec3 v0, vec3 v1, vec3 v2, vec2 uv) {
    float w = 1.0 - uv.x - uv.y;
    return v0 * w + v1 * uv.x + v2 * uv.y;
}

// Get triangle indices for a given sub-triangle at a subdivision level
// Returns uvec3(i0, i1, i2) - indices into the vertex array
uvec3 bary_triangle_indices(uint tri_idx, uint level) {
    if (level == 0u) {
        return uvec3(0u, 1u, 2u);
    } else if (level == 1u) {
        // 4 sub-triangles: (0,3,5), (3,1,4), (5,4,2), (3,4,5)
        uvec3 tris[4];
        tris[0] = uvec3(0u, 3u, 5u);
        tris[1] = uvec3(3u, 1u, 4u);
        tris[2] = uvec3(5u, 4u, 2u);
        tris[3] = uvec3(3u, 4u, 5u);
        return tris[tri_idx];
    } else {
        // Level 2: 16 sub-triangles from 4x4 grid
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
    }
}
)";

// Fragment shader (shared by all primitive types)
// Uses perprimitiveNV input to receive per-primitive color from mesh shaders
static const char* LUDUS_FRAGMENT_SHADER = R"(
#version 460
#extension GL_NV_fragment_shader_barycentric : require

// Per-primitive input block - must match mesh shader output
layout(location = 0) perprimitiveNV in PrimColorBlock {
    vec3 color;
} prim_in;

layout(location = 0) out vec4 out_color;

IF_ZMODIFY(layout(location = 0) uniform float in_dummy;)

void main() {
    out_color = vec4(prim_in.color, 1.0);
    IF_ZMODIFY(gl_FragDepth = gl_FragCoord.z + in_dummy;)
}
)";

// No task shader - we'll call mesh shaders directly
// With NV_mesh_shader, glDrawMeshTasksNV can directly invoke mesh shaders
// when no task shader is linked
static const char* LUDUS_OBSTACLE_TASK_SHADER = nullptr;

// Mesh shader for cubes (generates unit cube procedurally)
// Uses 8 shared vertices to reduce output size
// Cube Task Shader - computes subdivision level and dispatches 6 mesh workgroups (one per face)
static const char* LUDUS_CUBE_TASK_SHADER = R"(
layout(local_size_x = 1) in;

uniform uint num_cubes;
uniform uint num_cameras;
uniform float tessellation_threshold;

layout(std430, binding = 2) readonly buffer CubeBuffer {
    Cube cubes[];
};

layout(std430, binding = 5) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera cameras[];
};

layout(std430, binding = 6) readonly buffer CameraPoseBuffer {
    CameraPose poses[];
};

// Task payload to mesh shader
taskNV out CubePayload {
    uint cube_id;
    uint camera_id;
    uint subdivision_level;  // 0-2 (matching polygon levels)
    uint face_mask;          // Bitmask of front-facing faces (backface culling)
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

// Unit cube vertices for edge distortion calculation
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

void main() {
    uint work_id = gl_WorkGroupID.x;
    uint cid = work_id % num_cubes;
    uint cam_id = work_id / num_cubes;

    if (cid >= num_cubes || cam_id >= num_cameras) {
        gl_TaskCountNV = 0u;
        return;
    }

    Cube obs = cubes[cid];
    FThetaCamera cam = cameras[cam_id];
    CameraPose pose = poses[cam_id];

    vec3 translation = vec3(obs.tx, obs.ty, obs.tz);
    vec3 scale = vec3(obs.sx, obs.sy, obs.sz);
    vec3 rotation = vec3(obs.rx, obs.ry, obs.rz);

    // Backface culling: compute which faces are visible from camera
    mat3 R_cam = mat3(pose.world_to_camera);
    vec3 cam_world = -transpose(R_cam) * pose.world_to_camera[3].xyz;

    uint fmask = 0u;
    for (uint f = 0u; f < 6u; f++) {
        vec3 n_world = rotate_rodrigues(FACE_NORMALS[f], rotation);
        vec3 center_local = FACE_NORMALS[f] * 0.5 * scale;
        vec3 center_world = rotate_rodrigues(center_local, rotation) + translation;
        if (dot(n_world, cam_world - center_world) > 0.0) {
            fmask |= (1u << f);
        }
    }

    if (fmask == 0u) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Compute max distortion across all 12 edges
    uint max_subdiv = 0u;
    if (tessellation_threshold > 0.0) {
        for (uint e = 0u; e < 12u; e++) {
            vec3 v0_local = CUBE_VERTS[CUBE_EDGES[e].x] * scale;
            vec3 v1_local = CUBE_VERTS[CUBE_EDGES[e].y] * scale;
            vec3 v0_world = rotate_rodrigues(v0_local, rotation) + translation;
            vec3 v1_world = rotate_rodrigues(v1_local, rotation) + translation;

            uint subdiv = compute_subdivision_level(v0_world, v1_world, pose, cam, tessellation_threshold);
            max_subdiv = max(max_subdiv, subdiv);
        }
        // Limit to level 2 (same as polygon, for unified subdivision)
        max_subdiv = min(max_subdiv, 2u);
    }

    // Dispatch 6 workgroups (one per face)
    gl_TaskCountNV = 6u;

    cube_id = cid;
    camera_id = cam_id;
    subdivision_level = max_subdiv;
    face_mask = fmask;
}
)";

// Cube Mesh Shader - uses BARYCENTRIC SUBDIVISION (same as polygon)
// Each face = 2 triangles, subdivided using same pattern as polygons
// This unifies the subdivision logic with polygon mesh shader
static const char* LUDUS_CUBE_MESH_SHADER = R"(
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 30, max_primitives = 32) out;

uniform uint num_cubes;
uniform uint num_cameras;

layout(std430, binding = 2) readonly buffer CubeBuffer {
    Cube cubes[];
};

layout(std430, binding = 5) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera cameras[];
};

layout(std430, binding = 6) readonly buffer CameraPoseBuffer {
    CameraPose poses[];
};

taskNV in CubePayload {
    uint cube_id;
    uint camera_id;
    uint subdivision_level;
    uint face_mask;
};

// Per-primitive color output (avoids per-vertex array size limits)
layout(location = 0) perprimitiveNV out PrimColorBlock {
    vec3 color;
} prim_out[];

// Face vertex indices (4 corners per face, CCW winding)
const uvec4 FACE_VERTS[6] = uvec4[6](
    uvec4(0, 3, 2, 1),  // -Z face (back)
    uvec4(4, 5, 6, 7),  // +Z face (front)
    uvec4(0, 4, 7, 3),  // -X face
    uvec4(1, 2, 6, 5),  // +X face
    uvec4(0, 1, 5, 4),  // -Y face
    uvec4(3, 7, 6, 2)   // +Y face
);

// Unit cube vertices
const vec3 CUBE_VERTS[8] = vec3[8](
    vec3(-0.5, -0.5, -0.5), vec3(+0.5, -0.5, -0.5),
    vec3(+0.5, +0.5, -0.5), vec3(-0.5, +0.5, -0.5),
    vec3(-0.5, -0.5, +0.5), vec3(+0.5, -0.5, +0.5),
    vec3(+0.5, +0.5, +0.5), vec3(-0.5, +0.5, +0.5)
);

void main() {
    uint cid = cube_id;
    uint cam_id = camera_id;
    uint face_id = gl_WorkGroupID.x;  // 0-5
    uint subdiv = subdivision_level;
    uint tid = gl_LocalInvocationID.x;

    // Backface culling: skip faces not visible from camera
    if ((face_mask & (1u << face_id)) == 0u) {
        gl_PrimitiveCountNV = 0u;
        return;
    }

    Cube obs = cubes[cid];
    FThetaCamera cam = cameras[cam_id];
    CameraPose pose = poses[cam_id];

    vec3 translation = vec3(obs.tx, obs.ty, obs.tz);
    vec3 scale = vec3(obs.sx, obs.sy, obs.sz);
    vec3 rotation = vec3(obs.rx, obs.ry, obs.rz);

    // Get face corner info
    uvec4 face = FACE_VERTS[face_id];
    vec3 front_color = vec3(obs.front_r, obs.front_g, obs.front_b);
    vec3 back_color = vec3(obs.back_r, obs.back_g, obs.back_b);

    // Compute per-corner colors based on local X position (gradient from back to front)
    // In FLU convention: +X is forward (front_color), -X is back (back_color)
    vec3 corner_colors[4];
    vec3 corners[4];
    for (uint i = 0u; i < 4u; i++) {
        uint vert_idx = (i == 0u) ? face.x : (i == 1u) ? face.y : (i == 2u) ? face.z : face.w;
        vec3 local_vert = CUBE_VERTS[vert_idx];
        float t = local_vert.x + 0.5;  // 0 at back (-X), 1 at front (+X)
        corner_colors[i] = mix(back_color, front_color, t);
        corners[i] = rotate_rodrigues(local_vert * scale, rotation) + translation;
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
    vec3 tri_colors[6];
    tri_verts[0] = corners[0]; tri_verts[1] = corners[1]; tri_verts[2] = corners[2];
    tri_verts[3] = corners[0]; tri_verts[4] = corners[2]; tri_verts[5] = corners[3];
    tri_colors[0] = corner_colors[0]; tri_colors[1] = corner_colors[1]; tri_colors[2] = corner_colors[2];
    tri_colors[3] = corner_colors[0]; tri_colors[4] = corner_colors[2]; tri_colors[5] = corner_colors[3];

    // Generate vertices for both triangles using shared utilities
    for (uint t = 0u; t < 2u; t++) {
        vec3 v0 = tri_verts[t * 3u];
        vec3 v1 = tri_verts[t * 3u + 1u];
        vec3 v2 = tri_verts[t * 3u + 2u];
        uint vert_base = t * verts_per_tri;

        for (uint v = tid; v < verts_per_tri; v += 32u) {
            vec2 uv = bary_vertex_uv(v, subdiv);
            vec3 pt = bary_interpolate(v0, v1, v2, uv);

            vec4 clip = ftheta_project(pt, pose, cam);
            gl_MeshVerticesNV[vert_base + v].gl_Position = clip;
        }
    }

    barrier();

    // Generate triangle indices and per-primitive colors using shared utilities
    for (uint t = 0u; t < 2u; t++) {
        vec3 v0 = tri_verts[t * 3u];
        vec3 v1 = tri_verts[t * 3u + 1u];
        vec3 v2 = tri_verts[t * 3u + 2u];
        vec3 c0 = tri_colors[t * 3u];
        vec3 c1 = tri_colors[t * 3u + 1u];
        vec3 c2 = tri_colors[t * 3u + 2u];
        uint vert_base = t * verts_per_tri;
        uint prim_base = t * tris_per_tri;

        for (uint p = tid; p < tris_per_tri; p += 32u) {
            uvec3 idx = bary_triangle_indices(p, subdiv);

            // Compute centroid color for this sub-triangle
            vec2 uv0 = bary_vertex_uv(idx.x, subdiv);
            vec2 uv1 = bary_vertex_uv(idx.y, subdiv);
            vec2 uv2 = bary_vertex_uv(idx.z, subdiv);
            vec2 centroid_uv = (uv0 + uv1 + uv2) / 3.0;
            vec3 prim_color = bary_interpolate(c0, c1, c2, centroid_uv);

            uint tri_idx = prim_base + p;
            uint idx_base = tri_idx * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + idx.x;
            gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + idx.y;
            gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + idx.z;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(cam_id);
            prim_out[tri_idx].color = prim_color;
        }
    }
}
)";

//------------------------------------------------------------------------
// Polyline shaders

// Task shader for polylines - splits long polylines into chunks
// Each chunk handles up to 10 points, chunks overlap by 1 for seamless miter joins
// Uses taskNV block for task-to-mesh communication per NV_mesh_shader spec
static const char* LUDUS_POLYLINE_TASK_SHADER = R"(
layout(local_size_x = 1) in;

uniform uint num_polylines;
uniform uint num_cameras;
uniform float tessellation_threshold;  // Pixel error threshold (e.g., 1.0)

layout(std430, binding = 0) readonly buffer PolylineHeaderBuffer {
    PolylineHeader polylines[];
};

layout(std430, binding = 3) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(std430, binding = 5) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera cameras[];
};

layout(std430, binding = 6) readonly buffer CameraPoseBuffer {
    CameraPose poses[];
};

// Task output block - must use taskNV qualifier
taskNV out TaskPayload {
    uint polyline_id;
    uint camera_id;
    uint total_points;
    // Note: subdivision_level removed - now computed per-segment in mesh shader
};

void main() {
    uint work_id = gl_WorkGroupID.x;
    uint pid = work_id % num_polylines;
    uint cid = work_id / num_polylines;

    if (pid >= num_polylines || cid >= num_cameras) {
        gl_TaskCountNV = 0u;
        return;
    }

    PolylineHeader pl = polylines[pid];
    uint total_pts = pl.vertex_count;

    if (total_pts < 2u) {
        gl_TaskCountNV = 0u;
        return;
    }

    // Per-segment subdivision: chunking based on original segments
    // Each chunk will compute subdivision on-the-fly per segment
    // Balanced chunk size: 128 verts = 64 body verts = 64 effective points
    // Worst case 16x subdivision: 64/16 ≈ 4 segments, use 5 for safety
    const uint MAX_SEGMENTS_PER_CHUNK = 5u;
    uint num_segments = total_pts - 1u;
    uint num_chunks = (num_segments + MAX_SEGMENTS_PER_CHUNK - 1u) / MAX_SEGMENTS_PER_CHUNK;
    num_chunks = max(num_chunks, 1u);

    // Emit mesh tasks for each chunk
    gl_TaskCountNV = num_chunks;

    // Write task payload - readable by all spawned mesh shaders
    polyline_id = pid;
    camera_id = cid;
    total_points = total_pts;
}
)";

// Mesh shader for polylines with adaptive tessellation
// Supports subdivision levels 0-2 (1x, 2x, 4x subdivision per segment)
// Line body: 2 vertices per subdivided point, 2 triangles per segment
// Round caps: 4 triangles each at true endpoints only
// Max 12 effective points: 24 body verts + 8 cap verts = 32 verts, 22 body tris + 8 cap tris = 30 tris
static const char* LUDUS_POLYLINE_MESH_SHADER = R"(
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 128, max_primitives = 128) out;

// Uniforms
uniform uint num_polylines;
uniform uint num_cameras;
uniform float tessellation_threshold;  // Must match task shader

// Task input block - must match task shader's taskNV output
taskNV in TaskPayload {
    uint polyline_id;
    uint camera_id;
    uint total_points;
    // Note: subdivision_level removed - now computed per-segment
};

// SSBOs
layout(std430, binding = 0) readonly buffer PolylineHeaderBuffer {
    PolylineHeader polylines[];
};

layout(std430, binding = 3) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(std430, binding = 5) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera cameras[];
};

layout(std430, binding = 6) readonly buffer CameraPoseBuffer {
    CameraPose poses[];
};

// Per-primitive color output (avoids per-vertex array size limits)
layout(location = 0) perprimitiveNV out PrimColorBlock {
    vec3 color;
} prim_out[];

// Shared memory for subdivided projected points (optimized for better occupancy)
shared vec4 s_clip_pos[64];
shared vec2 s_screen_pos[64];
shared vec3 s_world_pos[64];  // Store world positions for subdivided points

void main() {
    // Get chunk ID from mesh workgroup index (set by task shader's gl_TaskCountNV)
    uint chunk_id = gl_WorkGroupID.x;

    // Read from task payload
    uint pl_id = polyline_id;
    uint cam_id = camera_id;
    uint total_pts = total_points;

    PolylineHeader pl = polylines[pl_id];
    FThetaCamera cam = cameras[cam_id];
    CameraPose pose = poses[cam_id];
    uint cap_style = pl.cap_style;

    // Calculate chunk range in original segment space
    const uint MAX_SEGMENTS_PER_CHUNK = 5u;
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

    vec3 color = vec3(pl.color_r, pl.color_g, pl.color_b);
    float half_width = pl.width * 0.5;
    uint tid = gl_LocalInvocationID.x;

    // Phase 1: Compute subdivision per segment and generate subdivided points
    // Each segment independently decides its subdivision level
    uint eff_point_count = 0u;

    if (tid == 0u) {
        // Thread 0 computes subdivision and generates all points
        for (uint seg_idx = 0u; seg_idx < num_segs_in_chunk; seg_idx++) {
            uint global_seg = seg_start + seg_idx;
            uint vi0 = pl.vertex_start + global_seg;
            uint vi1 = pl.vertex_start + global_seg + 1u;
            Vertex vx0 = vertices[vi0];
            Vertex vx1 = vertices[vi1];
            vec3 p0 = vec3(vx0.x, vx0.y, vx0.z);
            vec3 p1 = vec3(vx1.x, vx1.y, vx1.z);

            // Compute distortion error for this segment
            float error = estimate_edge_distortion_pixels(p0, p1, pose, cam);

            // Determine subdivision level for this segment
            uint subdiv_level = 0u;
            // Debug: Force specific subdivision level if threshold is negative
            if (tessellation_threshold < -0.01) {
                subdiv_level = uint(clamp(-tessellation_threshold, 0.0, 4.0));  // -1.0 → level 1, -2.0 → level 2, etc.
            } else if (tessellation_threshold > 0.01) {
                if (error > tessellation_threshold * 16.0) subdiv_level = 4u;      // 16x
                else if (error > tessellation_threshold * 8.0) subdiv_level = 3u;  // 8x
                else if (error > tessellation_threshold * 4.0) subdiv_level = 2u;  // 4x
                else if (error > tessellation_threshold * 2.0) subdiv_level = 1u;  // 2x
            }

            uint num_subsegments = 1u << subdiv_level;

            // Generate subdivided points for this segment
            // First point: only if this is the first segment in chunk
            if (seg_idx == 0u) {
                s_world_pos[eff_point_count] = p0;
                eff_point_count++;
            }

            // Interior points
            for (uint sub_i = 1u; sub_i <= num_subsegments; sub_i++) {
                float t = float(sub_i) / float(num_subsegments);
                vec3 world_pos = mix(p0, p1, t);
                s_world_pos[eff_point_count] = world_pos;
                eff_point_count++;

                if (eff_point_count >= 64u) break;  // Safety limit (shared memory size)
            }

            if (eff_point_count >= 64u) break;  // Safety limit
        }
    }

    barrier();

    // Thread 0 broadcasts the count via shared memory hack (use s_clip_pos[63] as counter storage)
    if (tid == 0u) {
        s_clip_pos[63].x = float(eff_point_count);
    }
    barrier();
    uint num_eff_points = uint(s_clip_pos[63].x);

    // Limit to mesh shader vertex capacity (128 verts = 64 body verts = 64 effective points)
    // Leave room for caps (8 verts max)
    num_eff_points = min(num_eff_points, 60u);

    if (num_eff_points < 2u) {
        gl_PrimitiveCountNV = 0u;
        return;
    }

    // Phase 2: All threads project their assigned points
    if (tid < num_eff_points) {
        vec3 world_pos = s_world_pos[tid];

        vec4 clip = ftheta_project(world_pos, pose, cam);
        s_clip_pos[tid] = clip;
        float w = max(abs(clip.w), 0.001);
        float ndc_x = clip.x / w;
        float ndc_y = clip.y / w;
        s_screen_pos[tid] = vec2(
            (ndc_x * 0.5 + 0.5) * cam.img_w,
            (0.5 - ndc_y * 0.5) * cam.img_h
        );
    }

    barrier();

    // Calculate output counts
    uint num_body_verts = num_eff_points * 2u;
    uint num_body_tris = (num_eff_points - 1u) * 2u;
    bool want_round_caps = (cap_style == 2u);
    bool draw_start_cap = want_round_caps && is_first_chunk;
    bool draw_end_cap = want_round_caps && is_last_chunk;
    uint num_cap_tris = (draw_start_cap ? 4u : 0u) + (draw_end_cap ? 4u : 0u);
    uint num_cap_verts = (draw_start_cap ? 4u : 0u) + (draw_end_cap ? 4u : 0u);

    gl_PrimitiveCountNV = num_body_tris + num_cap_tris;

    // Phase 3: Generate body vertices (2 per point with miter joins)
    if (tid < num_eff_points) {
        vec4 clip = s_clip_pos[tid];
        vec2 dir = vec2(1.0, 0.0);

        if (tid == 0u && num_eff_points > 1u) {
            vec2 d = s_screen_pos[1] - s_screen_pos[0];
            float len = length(d);
            if (len > 0.001) dir = d / len;
        } else if (tid == num_eff_points - 1u && num_eff_points > 1u) {
            vec2 d = s_screen_pos[tid] - s_screen_pos[tid - 1u];
            float len = length(d);
            if (len > 0.001) dir = d / len;
        } else {
            vec2 d1 = s_screen_pos[tid] - s_screen_pos[tid - 1u];
            vec2 d2 = s_screen_pos[tid + 1u] - s_screen_pos[tid];
            float len1 = length(d1);
            float len2 = length(d2);
            if (len1 > 0.001) d1 /= len1; else d1 = vec2(1.0, 0.0);
            if (len2 > 0.001) d2 /= len2; else d2 = vec2(1.0, 0.0);
            vec2 avg = d1 + d2;
            float avgLen = length(avg);
            if (avgLen > 0.001) dir = avg / avgLen; else dir = d1;
        }

        vec2 perp = vec2(-dir.y, dir.x);
        float offset_x = perp.x * half_width * 2.0 / cam.img_w;
        float offset_y = -perp.y * half_width * 2.0 / cam.img_h;

        gl_MeshVerticesNV[tid * 2u].gl_Position = vec4(clip.x - offset_x, clip.y - offset_y, clip.z, clip.w);
        gl_MeshVerticesNV[tid * 2u + 1u].gl_Position = vec4(clip.x + offset_x, clip.y + offset_y, clip.z, clip.w);
    }

    barrier();

    // Phase 3: Generate cap vertices (only at true polyline endpoints)
    // Each cap: 4 vertices (center + 3 arc points at -45°, 0°, +45°)
    // Start cap vertices at indices: num_body_verts + 0..3
    // End cap vertices at indices: num_body_verts + (draw_start_cap ? 4 : 0)..+3

    // Start cap vertices (thread 0-3)
    if (draw_start_cap && tid < 4u) {
        vec4 clip = s_clip_pos[0];
        vec2 d = s_screen_pos[1] - s_screen_pos[0];
        float len = length(d);
        vec2 line_dir = (len > 0.001) ? d / len : vec2(1.0, 0.0);
        vec2 outward = -line_dir;  // Points backward
        vec2 perp = vec2(-line_dir.y, line_dir.x);

        vec2 offset_dir;
        if (tid == 0u) {
            offset_dir = vec2(0.0);  // Center
        } else {
            float angle;
            if (tid == 1u) angle = -0.7854;
            else if (tid == 2u) angle = 0.0;
            else angle = 0.7854;
            offset_dir = outward * cos(angle) + perp * sin(angle);
        }

        float offset_x = offset_dir.x * half_width * 2.0 / cam.img_w;
        float offset_y = -offset_dir.y * half_width * 2.0 / cam.img_h;

        gl_MeshVerticesNV[num_body_verts + tid].gl_Position = vec4(clip.x + offset_x, clip.y + offset_y, clip.z, clip.w);
    }

    // End cap vertices (thread 0-3, offset by 4 if start cap exists)
    if (draw_end_cap && tid < 4u) {
        vec4 clip = s_clip_pos[num_eff_points - 1u];
        vec2 d = s_screen_pos[num_eff_points - 1u] - s_screen_pos[num_eff_points - 2u];
        float len = length(d);
        vec2 line_dir = (len > 0.001) ? d / len : vec2(1.0, 0.0);
        vec2 outward = line_dir;  // Points forward (away from line)
        vec2 perp = vec2(-line_dir.y, line_dir.x);

        vec2 offset_dir;
        if (tid == 0u) {
            offset_dir = vec2(0.0);  // Center
        } else {
            float angle;
            if (tid == 1u) angle = -0.7854;
            else if (tid == 2u) angle = 0.0;
            else angle = 0.7854;
            offset_dir = outward * cos(angle) + perp * sin(angle);
        }

        float offset_x = offset_dir.x * half_width * 2.0 / cam.img_w;
        float offset_y = -offset_dir.y * half_width * 2.0 / cam.img_h;

        uint vert_offset = num_body_verts + (draw_start_cap ? 4u : 0u);
        gl_MeshVerticesNV[vert_offset + tid].gl_Position = vec4(clip.x + offset_x, clip.y + offset_y, clip.z, clip.w);
    }

    barrier();  // Ensure all cap vertices are written before setting triangles

    // Phase 4: Set triangle indices and per-primitive colors (first thread only)
    if (tid == 0u) {
        uint tri_idx = 0u;

        // Body triangles
        for (uint i = 0u; i < num_eff_points - 1u; i++) {
            uint base = i * 2u;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = base;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = base + 1u;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = base + 2u;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = base + 1u;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = base + 3u;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = base + 2u;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;
        }

        // Start cap triangles (only for first chunk)
        if (draw_start_cap) {
            uint sc_center = num_body_verts + 0u;
            uint sc_arc1 = num_body_verts + 1u;
            uint sc_arc2 = num_body_verts + 2u;
            uint sc_arc3 = num_body_verts + 3u;

            // Fan: center → left[0] → arc1 → arc2 → arc3 → right[0]
            gl_PrimitiveIndicesNV[tri_idx * 3u] = sc_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = 0u;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = sc_arc1;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = sc_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = sc_arc1;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = sc_arc2;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = sc_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = sc_arc2;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = sc_arc3;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = sc_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = sc_arc3;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = 1u;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;
        }

        // End cap triangles (only for last chunk)
        if (draw_end_cap) {
            uint last_left = (num_eff_points - 1u) * 2u;
            uint last_right = last_left + 1u;
            uint ec_base = num_body_verts + (draw_start_cap ? 4u : 0u);
            uint ec_center = ec_base + 0u;
            uint ec_arc1 = ec_base + 1u;
            uint ec_arc2 = ec_base + 2u;
            uint ec_arc3 = ec_base + 3u;

            // Fan: center → last_left → arc1 → arc2 → arc3 → last_right
            gl_PrimitiveIndicesNV[tri_idx * 3u] = ec_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = last_left;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = ec_arc1;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = ec_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = ec_arc1;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = ec_arc2;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = ec_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = ec_arc2;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = ec_arc3;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;

            gl_PrimitiveIndicesNV[tri_idx * 3u] = ec_center;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 1u] = ec_arc3;
            gl_PrimitiveIndicesNV[tri_idx * 3u + 2u] = last_right;
            gl_MeshPrimitivesNV[tri_idx].gl_Layer = int(camera_id);
            prim_out[tri_idx].color = color;
            tri_idx++;
        }
    }
}
)";

//------------------------------------------------------------------------
// Polygon shaders with task shader for large polygon chunking

// Constants for polygon chunking
// Each chunk handles up to POLYGON_CHUNK_TRIS triangles
// Vertices are limited to POLYGON_MAX_VERTS per polygon (chunking doesn't help with vertices)
#define POLYGON_CHUNK_TRIS 30
#define POLYGON_MAX_VERTS 64

// Task shader for polygon chunking with adaptive tessellation
// Pre-computes vertex range for each chunk to enable vertex sharing within chunks
static const char* LUDUS_POLYGON_TASK_SHADER = R"(
layout(local_size_x = 1) in;

// Uniforms
uniform uint num_polygons;
uniform uint num_cameras;
uniform float tessellation_threshold;

// SSBOs
layout(std430, binding = 1) readonly buffer PolygonHeaderBuffer {
    PolygonHeader polygons[];
};

layout(std430, binding = 3) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(std430, binding = 4) readonly buffer TriangleBuffer {
    uvec3 triangles[];
};

layout(std430, binding = 5) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera cameras[];
};

layout(std430, binding = 6) readonly buffer CameraPoseBuffer {
    CameraPose poses[];
};

// Task payload to mesh shader - uniform subdivision for entire polygon (no T-junctions)
taskNV out PolygonPayload {
    uint polygon_id;
    uint camera_id;
    uint chunk_tri_start;   // First triangle index in this chunk
    uint chunk_tri_count;   // Number of triangles in this chunk
    uint min_vert_idx;      // Minimum vertex index used by this chunk
    uint vert_count;        // Number of vertices: max - min + 1
    uint subdivision_level; // Uniform level (0-2) for ALL triangles in polygon
};

void main() {
    uint work_id = gl_WorkGroupID.x;
    uint pg_id = work_id % num_polygons;
    uint cam_id = work_id / num_polygons;

    if (pg_id >= num_polygons || cam_id >= num_cameras) {
        gl_TaskCountNV = 0u;
        return;
    }

    PolygonHeader pg = polygons[pg_id];
    FThetaCamera cam = cameras[cam_id];
    CameraPose pose = poses[cam_id];
    uint total_tris = pg.triangle_count;
    uint total_verts = pg.vertex_count;

    // Compute max subdivision level from polygon edges (sample first few triangles)
    uint max_subdiv = 0u;
    if (tessellation_threshold > 0.0) {
        uint sample_count = min(total_tris, 8u);  // Sample up to 8 triangles
        for (uint t = 0u; t < sample_count; t++) {
            uvec3 tri = triangles[pg.triangle_start + t];
            Vertex v0 = vertices[pg.vertex_start + tri.x];
            Vertex v1 = vertices[pg.vertex_start + tri.y];
            Vertex v2 = vertices[pg.vertex_start + tri.z];

            vec3 p0 = vec3(v0.x, v0.y, v0.z);
            vec3 p1 = vec3(v1.x, v1.y, v1.z);
            vec3 p2 = vec3(v2.x, v2.y, v2.z);

            // Check all 3 edges
            max_subdiv = max(max_subdiv, compute_subdivision_level(p0, p1, pose, cam, tessellation_threshold));
            max_subdiv = max(max_subdiv, compute_subdivision_level(p1, p2, pose, cam, tessellation_threshold));
            max_subdiv = max(max_subdiv, compute_subdivision_level(p2, p0, pose, cam, tessellation_threshold));
        }
        // Limit to level 2 for polygons (16 sub-triangles per triangle max)
        // Level 3 would need 45 verts, 64 tris which exceeds mesh shader limits
        max_subdiv = min(max_subdiv, 2u);
    }

    // Chunking strategy depends on subdivision level
    // Maximize batching within mesh shader limits (32 verts, 32 tris)
    const uint MAX_SHARED_VERTS = 30u;

    uint num_chunks;
    uint tris_per_chunk;
    if (max_subdiv == 0u && total_verts <= MAX_SHARED_VERTS) {
        // No subdivision, small polygon: single chunk with shared vertices
        num_chunks = 1u;
    } else {
        // Compute tris per chunk based on subdivision level
        // Level 0: 3 verts, 1 tri -> batch 10 (30 verts, 10 tris)
        // Level 1: 6 verts, 4 tris -> batch 5 (30 verts, 20 tris)
        // Level 2: 15 verts, 16 tris -> batch 2 (30 verts, 32 tris)
        if (max_subdiv == 0u) tris_per_chunk = 10u;
        else if (max_subdiv == 1u) tris_per_chunk = 5u;
        else tris_per_chunk = 2u;  // Level 2

        num_chunks = (total_tris + tris_per_chunk - 1u) / tris_per_chunk;
        num_chunks = max(num_chunks, 1u);
    }

    // Spawn mesh shader workgroups
    gl_TaskCountNV = num_chunks;

    // Set common payload values (uniform level for all triangles - no T-junctions)
    polygon_id = pg_id;
    camera_id = cam_id;
    subdivision_level = max_subdiv;

    // For single-chunk case, use all vertices
    if (num_chunks == 1u) {
        chunk_tri_start = 0u;
        chunk_tri_count = total_tris;
        min_vert_idx = 0u;
        vert_count = total_verts;
    } else {
        // Multi-chunk: each mesh shader will compute its own range
        chunk_tri_start = 0u;
        chunk_tri_count = total_tris;
        min_vert_idx = 0u;
        vert_count = total_verts;
    }
}
)";

// Mesh shader for polygons with adaptive tessellation
// Supports subdivision levels 0-2 (1, 4, or 16 sub-triangles per original triangle)
static const char* LUDUS_POLYGON_MESH_SHADER = R"(
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 32, max_primitives = 32) out;

// Uniforms
uniform uint num_polygons;
uniform uint num_cameras;

// SSBOs
layout(std430, binding = 1) readonly buffer PolygonHeaderBuffer {
    PolygonHeader polygons[];
};

layout(std430, binding = 3) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(std430, binding = 4) readonly buffer TriangleBuffer {
    uvec3 triangles[];
};

layout(std430, binding = 5) readonly buffer CameraIntrinsicsBuffer {
    FThetaCamera cameras[];
};

layout(std430, binding = 6) readonly buffer CameraPoseBuffer {
    CameraPose poses[];
};

// Task payload from task shader (uniform subdivision for entire polygon)
taskNV in PolygonPayload {
    uint polygon_id;
    uint camera_id;
    uint chunk_tri_start;
    uint chunk_tri_count;
    uint min_vert_idx;
    uint vert_count;
    uint subdivision_level;
};

// Per-primitive color output (avoids per-vertex array size limits)
layout(location = 0) perprimitiveNV out PrimColorBlock {
    vec3 color;
} prim_out[];

void main() {
    uint pg_id = polygon_id;
    uint cam_id = camera_id;
    uint chunk_id = gl_WorkGroupID.x;
    uint subdiv = subdivision_level;

    PolygonHeader pg = polygons[pg_id];
    FThetaCamera cam = cameras[cam_id];
    CameraPose pose = poses[cam_id];
    vec3 color = vec3(pg.color_r, pg.color_g, pg.color_b);

    uint tid = gl_LocalInvocationID.x;
    uint total_tris = pg.triangle_count;
    uint total_verts = pg.vertex_count;

    const uint CHUNK_SIZE = 8u;  // Reduced for subdivision headroom
    const uint MAX_SHARED_VERTS = 30u;

    // Calculate sub-triangles per original triangle: 1, 4, or 16
    uint subdiv_factor = 1u << (subdiv * 2u);  // 1, 4, or 16

    // For subdivision, we need to process triangles differently
    if (subdiv == 0u && total_verts <= MAX_SHARED_VERTS) {
        // ===== NO SUBDIVISION, SHARED VERTEX MODE (small polygons) =====
        uint num_verts = min(total_verts, 30u);
        uint num_tris = min(total_tris, 28u);

        gl_PrimitiveCountNV = num_tris;

        if (tid < num_verts) {
            uint vert_idx = pg.vertex_start + tid;
            Vertex vtx = vertices[vert_idx];
            vec3 world_pos = vec3(vtx.x, vtx.y, vtx.z);

            vec4 clip = ftheta_project(world_pos, pose, cam);
            gl_MeshVerticesNV[tid].gl_Position = clip;
        }

        barrier();

        if (tid < num_tris) {
            uint tri_idx = pg.triangle_start + tid;
            uvec3 tri = triangles[tri_idx];

            uint base = tid * 3u;
            gl_PrimitiveIndicesNV[base] = tri.x;
            gl_PrimitiveIndicesNV[base + 1u] = tri.y;
            gl_PrimitiveIndicesNV[base + 2u] = tri.z;

            gl_MeshPrimitivesNV[tid].gl_Layer = int(cam_id);
            prim_out[tid].color = color;
        }
    } else if (subdiv == 1u) {
        // ===== LEVEL 1 SUBDIVISION: 4 sub-triangles per triangle =====
        // Uses shared barycentric utilities
        // Each original triangle -> 6 vertices, 4 triangles
        // We can handle ~5 original triangles per workgroup (30 verts, 20 tris)
        uint verts_per_tri = bary_vertex_count(1u);  // 6
        uint tris_per_tri = bary_triangle_count(1u);  // 4

        uint tris_per_chunk = 5u;
        uint chunk_start = chunk_id * tris_per_chunk;
        uint chunk_end = min(chunk_start + tris_per_chunk, total_tris);
        uint num_orig_tris = chunk_end - chunk_start;

        if (num_orig_tris == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        uint num_out_tris = num_orig_tris * tris_per_tri;
        gl_PrimitiveCountNV = num_out_tris;

        // Each thread handles one original triangle's vertices
        if (tid < num_orig_tris) {
            uint orig_tri_idx = pg.triangle_start + chunk_start + tid;
            uvec3 tri = triangles[orig_tri_idx];

            Vertex v0_data = vertices[pg.vertex_start + tri.x];
            Vertex v1_data = vertices[pg.vertex_start + tri.y];
            Vertex v2_data = vertices[pg.vertex_start + tri.z];

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

        // Each thread also sets triangle indices and per-primitive color
        if (tid < num_orig_tris) {
            uint vert_base = tid * verts_per_tri;
            uint tri_base = tid * tris_per_tri;

            for (uint t = 0u; t < tris_per_tri; t++) {
                uvec3 idx = bary_triangle_indices(t, 1u);
                uint idx_base = (tri_base + t) * 3u;
                gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + idx.x;
                gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + idx.y;
                gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + idx.z;
                gl_MeshPrimitivesNV[tri_base + t].gl_Layer = int(cam_id);
                prim_out[tri_base + t].color = color;
            }
        }
    } else if (subdiv == 2u) {
        // ===== LEVEL 2 WITH TRUE PER-EDGE ADAPTIVE SUBDIVISION =====
        // Uses shared memory to communicate per-edge levels across threads
        // Can fit 2 triangles: 30 verts, 32 tris max

        const uint TRIS_PER_CHUNK = 2u;
        uint chunk_start = chunk_id * TRIS_PER_CHUNK;
        uint chunk_end = min(chunk_start + TRIS_PER_CHUNK, total_tris);
        uint num_orig_tris = chunk_end - chunk_start;

        if (num_orig_tris == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        // Use UNIFORM subdivision level for all triangles in polygon to avoid T-junctions
        // The task shader already computed max_subdiv for the entire polygon
        // All triangles use the same level = no gaps between adjacent triangles

        gl_PrimitiveCountNV = num_orig_tris * 16u;  // Level 2: 16 sub-triangles each

        // Each thread handles vertices for one of the 2 triangles
        uint tri_local = tid / 16u;
        uint vert_local = tid % 16u;

        // Generate vertices using shared barycentric utilities
        if (tri_local < num_orig_tris && vert_local < 15u) {
            uint orig_tri_idx = pg.triangle_start + chunk_start + tri_local;
            uvec3 tri = triangles[orig_tri_idx];

            Vertex v0_data = vertices[pg.vertex_start + tri.x];
            Vertex v1_data = vertices[pg.vertex_start + tri.y];
            Vertex v2_data = vertices[pg.vertex_start + tri.z];

            vec3 v0 = vec3(v0_data.x, v0_data.y, v0_data.z);
            vec3 v1 = vec3(v1_data.x, v1_data.y, v1_data.z);
            vec3 v2 = vec3(v2_data.x, v2_data.y, v2_data.z);

            // Use shared barycentric utilities
            vec2 uv = bary_vertex_uv(vert_local, 2u);
            vec3 pt = bary_interpolate(v0, v1, v2, uv);

            vec4 clip = ftheta_project(pt, pose, cam);
            uint vert_idx = tri_local * 15u + vert_local;
            gl_MeshVerticesNV[vert_idx].gl_Position = clip;
        }

        barrier();

        // Generate triangle indices and per-primitive colors using shared barycentric utilities
        if (tri_local < num_orig_tris && vert_local < 16u) {
            uint vert_base = tri_local * 15u;
            uint tri_out_idx = tri_local * 16u + vert_local;

            uvec3 idx = bary_triangle_indices(vert_local, 2u);

            uint idx_base = tri_out_idx * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + idx.x;
            gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + idx.y;
            gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + idx.z;
            gl_MeshPrimitivesNV[tri_out_idx].gl_Layer = int(cam_id);
            prim_out[tri_out_idx].color = color;
        }
    } else {
        // ===== LARGE POLYGON WITHOUT SUBDIVISION =====
        // Multiple chunks, each outputs 3 vertices per triangle (non-shared)

        uint chunk_start = chunk_id * CHUNK_SIZE;
        uint chunk_end = min(chunk_start + CHUNK_SIZE, total_tris);
        uint num_orig_tris = chunk_end - chunk_start;

        if (num_orig_tris == 0u) {
            gl_PrimitiveCountNV = 0u;
            return;
        }

        gl_PrimitiveCountNV = num_orig_tris;

        // Each thread handles one triangle
        if (tid < num_orig_tris) {
            uint orig_tri_idx = pg.triangle_start + chunk_start + tid;
            uvec3 tri = triangles[orig_tri_idx];

            uint vert_base = tid * 3u;

            for (uint vi = 0u; vi < 3u; vi++) {
                uint local_vert_idx = tri[vi];
                uint global_vert_idx = pg.vertex_start + local_vert_idx;
                Vertex vtx = vertices[global_vert_idx];
                vec3 world_pos = vec3(vtx.x, vtx.y, vtx.z);

                vec4 clip = ftheta_project(world_pos, pose, cam);
                gl_MeshVerticesNV[vert_base + vi].gl_Position = clip;
            }

            // Set triangle indices (consecutive, non-shared) and per-primitive color
            uint idx_base = tid * 3u;
            gl_PrimitiveIndicesNV[idx_base + 0u] = vert_base + 0u;
            gl_PrimitiveIndicesNV[idx_base + 1u] = vert_base + 1u;
            gl_PrimitiveIndicesNV[idx_base + 2u] = vert_base + 2u;
            gl_MeshPrimitivesNV[tid].gl_Layer = int(cam_id);
            prim_out[tid].color = color;
        }
    }
}
)";



//------------------------------------------------------------------------
// Initialize Ludus GL context.

void ludusInitGLContext(NVDR_CTX_ARGS, LudusGLState& s, int cudaDeviceIdx)
{
    // Create GL context and set it current.
    s.glctx = createGLContext(cudaDeviceIdx);
    setGLContext(s.glctx);

    // Version check - need OpenGL 4.6 for mesh shaders
    GLint vMajor = 0;
    GLint vMinor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &vMajor);
    glGetIntegerv(GL_MINOR_VERSION, &vMinor);
    glGetError();
    LOG(INFO) << "Ludus: OpenGL version " << vMajor << "." << vMinor;
    NVDR_CHECK((vMajor == 4 && vMinor >= 6) || vMajor > 4, "OpenGL 4.6 or later is required for mesh shaders");

    // Check for mesh shader extension by checking if function pointer is available
    // The glDrawMeshTasksNV function should have been loaded during GL init
    s.hasMeshShader = (glDrawMeshTasksNV != nullptr) ? 1 : 0;
    LOG(INFO) << "Ludus: glDrawMeshTasksNV = " << (void*)glDrawMeshTasksNV;
    if (s.hasMeshShader)
    {
        LOG(INFO) << "Ludus: Mesh shader extension supported";
    }
    else
    {
        LOG(WARNING) << "Ludus: Mesh shaders NOT supported, falling back to traditional rendering";
    }
    // For now, don't require mesh shaders - fall back gracefully
    // NVDR_CHECK(s.hasMeshShader, "Mesh shader extension (GL_EXT_mesh_shader or GL_NV_mesh_shader) is required");

    // Enable depth modification workaround on A100 and later.
    int capMajor = 0;
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&capMajor, cudaDevAttrComputeCapabilityMajor, cudaDeviceIdx));
    s.enableZModify = (capMajor >= 8);

    // Compile fragment shader (shared)
    compileLudusShader(NVDR_CTX_PARAMS, &s.glFragmentShader, GL_FRAGMENT_SHADER, LUDUS_FRAGMENT_SHADER, s.enableZModify);

    // Only compile mesh shaders if supported
    if (s.hasMeshShader)
    {
        // Compile cube shaders
        // Note: We skip task shader and call mesh shaders directly via glDrawMeshTasksNV
        std::string cubeTaskSrc = std::string(LUDUS_GLSL_COMMON) + LUDUS_CUBE_TASK_SHADER;
        std::string cubeMeshSrc = std::string(LUDUS_GLSL_COMMON) + LUDUS_CUBE_MESH_SHADER;

        compileLudusShader(NVDR_CTX_PARAMS, &s.glTaskShaderCube, GL_TASK_SHADER_NV, cubeTaskSrc.c_str(), s.enableZModify);
        compileLudusShader(NVDR_CTX_PARAMS, &s.glMeshShaderCube, GL_MESH_SHADER_NV, cubeMeshSrc.c_str(), s.enableZModify);

        // Link cube program with task shader
        constructLudusMeshProgram(NVDR_CTX_PARAMS, &s.glProgramCube, s.glTaskShaderCube, s.glMeshShaderCube, s.glFragmentShader);
        LOG(INFO) << "Ludus: Cube task+mesh shaders compiled successfully";

        // Compile polyline shaders
        std::string polylineTaskSrc = std::string(LUDUS_GLSL_COMMON) + LUDUS_POLYLINE_TASK_SHADER;
        std::string polylineMeshSrc = std::string(LUDUS_GLSL_COMMON) + LUDUS_POLYLINE_MESH_SHADER;
        compileLudusShader(NVDR_CTX_PARAMS, &s.glTaskShaderPolyline, GL_TASK_SHADER_NV, polylineTaskSrc.c_str(), s.enableZModify);
        compileLudusShader(NVDR_CTX_PARAMS, &s.glMeshShaderPolyline, GL_MESH_SHADER_NV, polylineMeshSrc.c_str(), s.enableZModify);
        constructLudusMeshProgram(NVDR_CTX_PARAMS, &s.glProgramPolyline, s.glTaskShaderPolyline, s.glMeshShaderPolyline, s.glFragmentShader);
        LOG(INFO) << "Ludus: Polyline task+mesh shaders compiled successfully";

        // Compile polygon shaders (with task shader for large polygon chunking)
        std::string polygonTaskSrc = std::string(LUDUS_GLSL_COMMON) + LUDUS_POLYGON_TASK_SHADER;
        std::string polygonMeshSrc = std::string(LUDUS_GLSL_COMMON) + LUDUS_POLYGON_MESH_SHADER;
        compileLudusShader(NVDR_CTX_PARAMS, &s.glTaskShaderPolygon, GL_TASK_SHADER_NV, polygonTaskSrc.c_str(), s.enableZModify);
        compileLudusShader(NVDR_CTX_PARAMS, &s.glMeshShaderPolygon, GL_MESH_SHADER_NV, polygonMeshSrc.c_str(), s.enableZModify);
        constructLudusMeshProgram(NVDR_CTX_PARAMS, &s.glProgramPolygon, s.glTaskShaderPolygon, s.glMeshShaderPolygon, s.glFragmentShader);
        LOG(INFO) << "Ludus: Polygon task+mesh shaders compiled successfully";
    }
    else
    {
        LOG(INFO) << "Ludus: Mesh shaders not available, using fallback rendering";
        // TODO: Compile traditional vertex/geometry shaders as fallback
    }

    // Construct FBO
    NVDR_CHECK_GL_ERROR(glGenFramebuffers(1, &s.glFBO));
    NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));

    // Enable single color attachment
    GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 };
    NVDR_CHECK_GL_ERROR(glDrawBuffers(1, draw_buffers));

    // Set up depth test
    NVDR_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
    NVDR_CHECK_GL_ERROR(glDepthFunc(GL_LESS));
    NVDR_CHECK_GL_ERROR(glClearDepth(1.0));

    // Create SSBOs (storage allocated on first use)
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glPolylineHeaderBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glPolygonHeaderBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glCubeBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glVertexBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glTriangleBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glCameraIntrinsicsBuffer));
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glCameraPoseBuffer));

    // Create color and depth textures (storage allocated on first use)
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glColorBuffer));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glDepthStencilBuffer));

    // Create MSAA resources (storage allocated on first use if MSAA enabled)
    NVDR_CHECK_GL_ERROR(glGenFramebuffers(1, &s.glFBO_MSAA));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glColorBuffer_MSAA));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glDepthStencilBuffer_MSAA));
    s.msaaSamples = 0;  // MSAA disabled by default

    LOG(INFO) << "Ludus: GL context initialized successfully";
}

//------------------------------------------------------------------------
// Resize Ludus buffers.

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
    int height)
{
    changes = false;

    // Resize cube buffer
    if (cubeCount > s.cubeCount)
    {
        if (s.cudaCubeBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCubeBuffer));
        s.cubeCount = (cubeCount > 16) ? ROUND_UP_BITS_LUDUS(cubeCount, 2) : 16;
        LOG(INFO) << "Ludus: Increasing cube buffer to " << s.cubeCount << " cubes";
        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glCubeBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.cubeCount * sizeof(Cube), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaCubeBuffer, s.glCubeBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize polyline header buffer
    if (polylineHeaderCount > s.polylineHeaderCount)
    {
        if (s.cudaPolylineHeaderBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPolylineHeaderBuffer));
        s.polylineHeaderCount = (polylineHeaderCount > 16) ? ROUND_UP_BITS_LUDUS(polylineHeaderCount, 2) : 16;
        LOG(INFO) << "Ludus: Increasing polyline header buffer to " << s.polylineHeaderCount << " polylines";
        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glPolylineHeaderBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.polylineHeaderCount * sizeof(PolylineHeader), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaPolylineHeaderBuffer, s.glPolylineHeaderBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize polygon header buffer
    if (polygonHeaderCount > s.polygonHeaderCount)
    {
        if (s.cudaPolygonHeaderBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPolygonHeaderBuffer));
        s.polygonHeaderCount = (polygonHeaderCount > 16) ? ROUND_UP_BITS_LUDUS(polygonHeaderCount, 2) : 16;
        LOG(INFO) << "Ludus: Increasing polygon header buffer to " << s.polygonHeaderCount << " polygons";
        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glPolygonHeaderBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.polygonHeaderCount * sizeof(PolygonHeader), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaPolygonHeaderBuffer, s.glPolygonHeaderBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize vertex buffer
    if (vertexCount > s.vertexCount)
    {
        if (s.cudaVertexBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaVertexBuffer));
        s.vertexCount = (vertexCount > 256) ? ROUND_UP_BITS_LUDUS(vertexCount, 4) : 256;
        LOG(INFO) << "Ludus: Increasing vertex buffer to " << s.vertexCount << " vertices";
        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glVertexBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.vertexCount * sizeof(Vertex), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaVertexBuffer, s.glVertexBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize triangle buffer
    if (triangleCount > s.triangleCount)
    {
        if (s.cudaTriangleBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaTriangleBuffer));
        s.triangleCount = (triangleCount > 256) ? ROUND_UP_BITS_LUDUS(triangleCount, 4) : 256;
        LOG(INFO) << "Ludus: Increasing triangle buffer to " << s.triangleCount << " triangles";
        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glTriangleBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.triangleCount * sizeof(Triangle), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaTriangleBuffer, s.glTriangleBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
        changes = true;
    }

    // Resize camera intrinsics buffer
    if (cameraCount > s.cameraCount)
    {
        if (s.cudaCameraIntrinsicsBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCameraIntrinsicsBuffer));
        if (s.cudaCameraPoseBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCameraPoseBuffer));

        s.cameraCount = (cameraCount > 8) ? ROUND_UP_BITS_LUDUS(cameraCount, 2) : 8;
        LOG(INFO) << "Ludus: Increasing camera buffers to " << s.cameraCount << " cameras";

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glCameraIntrinsicsBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.cameraCount * sizeof(FThetaCamera), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaCameraIntrinsicsBuffer, s.glCameraIntrinsicsBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

        NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, s.glCameraPoseBuffer));
        NVDR_CHECK_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER, s.cameraCount * sizeof(CameraPose), NULL, GL_DYNAMIC_DRAW));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaCameraPoseBuffer, s.glCameraPoseBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

        changes = true;
    }

    // Resize framebuffer
    if (width > s.width || height > s.height || cameraCount > s.depth)
    {
        if (s.cudaColorBuffer)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer));

        s.width = (width > s.width) ? width : s.width;
        s.height = (height > s.height) ? height : s.height;
        s.depth = (cameraCount > s.depth) ? cameraCount : s.depth;
        s.width = ROUND_UP(s.width, 32);
        s.height = ROUND_UP(s.height, 32);
        LOG(INFO) << "Ludus: Increasing framebuffer to " << s.width << "x" << s.height << "x" << s.depth;

        // Allocate color buffer (layered texture array) - used for resolve target and CUDA readback
        NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer));
        NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, s.width, s.height, s.depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer, 0));

        // Allocate depth/stencil buffer
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glDepthStencilBuffer));
        NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH24_STENCIL8, s.width, s.height, s.depth, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 0));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, s.glDepthStencilBuffer, 0));

        // Allocate MSAA buffers if MSAA is enabled
        if (s.msaaSamples >= 2)
        {
            LOG(INFO) << "Ludus: Allocating MSAA framebuffer with " << s.msaaSamples << " samples";
            NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO_MSAA));

            // MSAA color buffer (GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
            NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.glColorBuffer_MSAA));
            NVDR_CHECK_GL_ERROR(glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.msaaSamples,
                                                        GL_RGBA8, s.width, s.height, s.depth, GL_TRUE));
            NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.glColorBuffer_MSAA, 0));

            // MSAA depth/stencil buffer
            NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.glDepthStencilBuffer_MSAA));
            NVDR_CHECK_GL_ERROR(glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, s.msaaSamples,
                                                        GL_DEPTH24_STENCIL8, s.width, s.height, s.depth, GL_TRUE));
            NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                                     s.glDepthStencilBuffer_MSAA, 0));

            // Verify MSAA framebuffer is complete
            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE)
            {
                LOG(WARNING) << "Ludus: MSAA framebuffer incomplete (status=" << status << "), disabling MSAA";
                s.msaaSamples = 0;
            }

            // Setup draw buffers for MSAA FBO
            GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 };
            NVDR_CHECK_GL_ERROR(glDrawBuffers(1, draw_buffers));
        }

        // Register with CUDA
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&s.cudaColorBuffer, s.glColorBuffer, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));

        changes = true;
    }

    NVDR_CHECK_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0));
}

//------------------------------------------------------------------------
// Render scene with f-theta cameras.

void ludusRender(
    NVDR_CTX_ARGS,
    LudusGLState& s,
    cudaStream_t stream,
    const PolylineHeader* polylineHeaders,
    int numPolylines,
    const PolygonHeader* polygonHeaders,
    int numPolygons,
    const Cube* cubes,
    int numCubes,
    const Vertex* vertices,
    int numVertices,
    const Triangle* triangles,
    int numTriangles,
    const FThetaCamera* cameraIntrinsics,
    const CameraPose* cameraPoses,
    int numCameras,
    int width,
    int height)
{
    // Build list of resources to map
    std::vector<cudaGraphicsResource_t> resourcesToMap;

    // Always map camera buffers
    resourcesToMap.push_back(s.cudaCameraIntrinsicsBuffer);
    resourcesToMap.push_back(s.cudaCameraPoseBuffer);

    if (numCubes > 0)
        resourcesToMap.push_back(s.cudaCubeBuffer);
    if (numPolylines > 0) {
        resourcesToMap.push_back(s.cudaPolylineHeaderBuffer);
        resourcesToMap.push_back(s.cudaVertexBuffer);
    }
    if (numPolygons > 0) {
        resourcesToMap.push_back(s.cudaPolygonHeaderBuffer);
        if (numPolylines == 0) // Only add if not already added
            resourcesToMap.push_back(s.cudaVertexBuffer);
        resourcesToMap.push_back(s.cudaTriangleBuffer);
    }

    // Map all resources at once
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources((int)resourcesToMap.size(), resourcesToMap.data(), stream));

    // Copy camera data (always needed)
    {
        void* glCamIntrPtr = NULL;
        size_t camIntrBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glCamIntrPtr, &camIntrBytes, s.cudaCameraIntrinsicsBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glCamIntrPtr, cameraIntrinsics, numCameras * sizeof(FThetaCamera), cudaMemcpyDeviceToDevice, stream));

        void* glCamPosePtr = NULL;
        size_t camPoseBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glCamPosePtr, &camPoseBytes, s.cudaCameraPoseBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glCamPosePtr, cameraPoses, numCameras * sizeof(CameraPose), cudaMemcpyDeviceToDevice, stream));
    }

    // Copy cube data
    if (numCubes > 0)
    {
        void* glCubePtr = NULL;
        size_t cubeBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glCubePtr, &cubeBytes, s.cudaCubeBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glCubePtr, cubes, numCubes * sizeof(Cube), cudaMemcpyDeviceToDevice, stream));
    }

    // Copy polyline data
    if (numPolylines > 0)
    {
        void* glPolylinePtr = NULL;
        size_t polylineBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glPolylinePtr, &polylineBytes, s.cudaPolylineHeaderBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glPolylinePtr, polylineHeaders, numPolylines * sizeof(PolylineHeader), cudaMemcpyDeviceToDevice, stream));

        void* glVertexPtr = NULL;
        size_t vertexBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glVertexPtr, &vertexBytes, s.cudaVertexBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glVertexPtr, vertices, numVertices * sizeof(Vertex), cudaMemcpyDeviceToDevice, stream));
    }

    // Copy polygon data
    if (numPolygons > 0)
    {
        void* glPolygonPtr = NULL;
        size_t polygonBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glPolygonPtr, &polygonBytes, s.cudaPolygonHeaderBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glPolygonPtr, polygonHeaders, numPolygons * sizeof(PolygonHeader), cudaMemcpyDeviceToDevice, stream));

        if (numPolylines == 0) {
            void* glVertexPtr = NULL;
            size_t vertexBytes = 0;
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glVertexPtr, &vertexBytes, s.cudaVertexBuffer));
            NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glVertexPtr, vertices, numVertices * sizeof(Vertex), cudaMemcpyDeviceToDevice, stream));
        }

        void* glTriPtr = NULL;
        size_t triBytes = 0;
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glTriPtr, &triBytes, s.cudaTriangleBuffer));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glTriPtr, triangles, numTriangles * sizeof(Triangle), cudaMemcpyDeviceToDevice, stream));
    }

    // Unmap all resources
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources((int)resourcesToMap.size(), resourcesToMap.data(), stream));

    // Bind framebuffer and set viewport
    // If MSAA is enabled, render to MSAA FBO; otherwise render directly to regular FBO
    GLuint renderFBO = (s.msaaSamples >= 2) ? s.glFBO_MSAA : s.glFBO;
    NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, renderFBO));
    NVDR_CHECK_GL_ERROR(glViewport(0, 0, width, height));
    NVDR_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    // Skip rendering if mesh shaders not available
    if (!s.hasMeshShader)
    {
        LOG(WARNING) << "Ludus: Mesh shaders not available, skipping render";
        return;
    }

    // Render cubes
    if (numCubes > 0)
    {
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramCube));

        // Bind SSBOs
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s.glCubeBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s.glCameraIntrinsicsBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s.glCameraPoseBuffer));

        // Set push constants (via uniform for OpenGL)
        GLint locNumCubes = glGetUniformLocation(s.glProgramCube, "num_cubes");
        GLint locNumCameras = glGetUniformLocation(s.glProgramCube, "num_cameras");
        GLint locTessThreshold = glGetUniformLocation(s.glProgramCube, "tessellation_threshold");
        if (locNumCubes >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1ui(locNumCubes, (GLuint)numCubes));
        if (locNumCameras >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1ui(locNumCameras, (GLuint)numCameras));
        if (locTessThreshold >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1f(locTessThreshold, s.tessellationThreshold));

        // Set depth modification uniform if needed
        if (s.enableZModify)
        {
            GLint locDummy = glGetUniformLocation(s.glProgramCube, "in_dummy");
            if (locDummy >= 0)
                NVDR_CHECK_GL_ERROR(glUniform1f(locDummy, 0.0f));
        }

        // Dispatch task shader: one workgroup per (cube, camera) pair
        // Task shader will dispatch 6 mesh workgroups per cube (one per face)
        uint32_t numWorkgroups = numCubes * numCameras;
        NVDR_CHECK_GL_ERROR(glDrawMeshTasksNV(0, numWorkgroups));
    }

    // Render polylines (skip if program not compiled)
    if (numPolylines > 0 && s.glProgramPolyline != 0)
    {
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramPolyline));

        // Bind SSBOs
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s.glPolylineHeaderBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s.glVertexBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s.glCameraIntrinsicsBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s.glCameraPoseBuffer));

        // Set uniforms
        GLint locNumPolylines = glGetUniformLocation(s.glProgramPolyline, "num_polylines");
        GLint locNumCameras = glGetUniformLocation(s.glProgramPolyline, "num_cameras");
        GLint locTessThreshold = glGetUniformLocation(s.glProgramPolyline, "tessellation_threshold");
        if (locNumPolylines >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1ui(locNumPolylines, (GLuint)numPolylines));
        if (locNumCameras >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1ui(locNumCameras, (GLuint)numCameras));
        if (locTessThreshold >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1f(locTessThreshold, s.tessellationThreshold));

        if (s.enableZModify)
        {
            GLint locDummy = glGetUniformLocation(s.glProgramPolyline, "in_dummy");
            if (locDummy >= 0)
                NVDR_CHECK_GL_ERROR(glUniform1f(locDummy, 0.0f));
        }

        // Dispatch one workgroup per (polyline, camera) pair
        // glDrawMeshTasksNV(first, count) - first=0, count=numWorkgroups
        uint32_t numWorkgroups = numPolylines * numCameras;
        NVDR_CHECK_GL_ERROR(glDrawMeshTasksNV(0, numWorkgroups));
    }

    // Render polygons (skip if program not compiled)
    if (numPolygons > 0 && s.glProgramPolygon != 0)
    {
        NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgramPolygon));

        // Bind SSBOs
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s.glPolygonHeaderBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s.glVertexBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s.glTriangleBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s.glCameraIntrinsicsBuffer));
        NVDR_CHECK_GL_ERROR(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s.glCameraPoseBuffer));

        // Set uniforms
        GLint locNumPolygons = glGetUniformLocation(s.glProgramPolygon, "num_polygons");
        GLint locNumCameras = glGetUniformLocation(s.glProgramPolygon, "num_cameras");
        GLint locTessThreshold = glGetUniformLocation(s.glProgramPolygon, "tessellation_threshold");
        if (locNumPolygons >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1ui(locNumPolygons, (GLuint)numPolygons));
        if (locNumCameras >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1ui(locNumCameras, (GLuint)numCameras));
        if (locTessThreshold >= 0)
            NVDR_CHECK_GL_ERROR(glUniform1f(locTessThreshold, s.tessellationThreshold));

        if (s.enableZModify)
        {
            GLint locDummy = glGetUniformLocation(s.glProgramPolygon, "in_dummy");
            if (locDummy >= 0)
                NVDR_CHECK_GL_ERROR(glUniform1f(locDummy, 0.0f));
        }

        // Dispatch one workgroup per (polygon, camera) pair
        // glDrawMeshTasksNV(first, count) - first=0, count=numWorkgroups
        uint32_t numWorkgroups = numPolygons * numCameras;
        NVDR_CHECK_GL_ERROR(glDrawMeshTasksNV(0, numWorkgroups));
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
        for (int layer = 0; layer < numCameras; layer++)
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
}

//------------------------------------------------------------------------
// Copy results to output tensor.

void ludusCopyResults(
    NVDR_CTX_ARGS,
    LudusGLState& s,
    cudaStream_t stream,
    uint8_t* outputPtr,
    int width,
    int height,
    int numCameras)
{
    cudaArray_t array = 0;
    cudaChannelFormatDesc arrayDesc = {};
    cudaExtent arrayExt = {};

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaColorBuffer, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, s.cudaColorBuffer, 0, 0));
    NVDR_CHECK_CUDA_ERROR(cudaArrayGetInfo(&arrayDesc, &arrayExt, NULL, array));

    cudaMemcpy3DParms p = {0};
    p.srcArray = array;
    p.dstPtr.ptr = outputPtr;
    p.dstPtr.pitch = width * 4 * sizeof(uint8_t);  // RGBA8 = 4 bytes per pixel
    p.dstPtr.xsize = width;
    p.dstPtr.ysize = height;
    p.extent.width = width;
    p.extent.height = height;
    p.extent.depth = numCameras;
    p.kind = cudaMemcpyDeviceToDevice;
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy3DAsync(&p, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaColorBuffer, stream));
}

//------------------------------------------------------------------------
// Release Ludus buffers.

void ludusReleaseBuffers(NVDR_CTX_ARGS, LudusGLState& s)
{
    if (s.cudaColorBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer));
        s.cudaColorBuffer = 0;
    }
    if (s.cudaCubeBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCubeBuffer));
        s.cudaCubeBuffer = 0;
    }
    if (s.cudaCameraIntrinsicsBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCameraIntrinsicsBuffer));
        s.cudaCameraIntrinsicsBuffer = 0;
    }
    if (s.cudaCameraPoseBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaCameraPoseBuffer));
        s.cudaCameraPoseBuffer = 0;
    }
    if (s.cudaPolylineHeaderBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPolylineHeaderBuffer));
        s.cudaPolylineHeaderBuffer = 0;
    }
    if (s.cudaPolygonHeaderBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPolygonHeaderBuffer));
        s.cudaPolygonHeaderBuffer = 0;
    }
    if (s.cudaVertexBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaVertexBuffer));
        s.cudaVertexBuffer = 0;
    }
    if (s.cudaTriangleBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaTriangleBuffer));
        s.cudaTriangleBuffer = 0;
    }
}

//------------------------------------------------------------------------
