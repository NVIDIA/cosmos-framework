# Shader code: scale colors by depth using orthographic projection
vertex_shader_graident_color_code = """
#version 330
in vec3 in_pos;
in vec3 in_color;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
out vec3 v_color;
void main() {
    // Convert pixel coordinates to NDC coordinates
    float x_ndc = 2.0 * in_pos.x / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * in_pos.y / image_height;
    float z_ndc = (in_pos.z / depth_max) * 2.0 - 1.0;

    // Set the vertex position
    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);

    // Set the color based on depth
    float r = clamp((1.0 - z_ndc) / 2.0, 0.0, 1.0);
    v_color = in_color * r;
}
"""

vertex_shader_constant_color_code = """
#version 330
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
void main() {
    float x_ndc = 2.0 * in_pos.x / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * in_pos.y / image_height;
    float z_ndc = (in_pos.z / depth_max) * 2.0 - 1.0;
    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);
    v_color = in_color;
}
"""

# Geometry shader code: thicken line segments
geometry_shader_code = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 42) out;

uniform float u_line_width;

in vec3 v_color[];
out vec3 g_color;

void createRoundCap(vec4 center, vec2 direction, vec3 color, float line_width) {
    const int segments = 8; // Number of semicircle segments; adjust to control quality
    const float PI = 3.1415926;

    // Compute a vector perpendicular to the line segment
    vec2 perpendicular = vec2(-direction.y, direction.x);

    for (int i = 0; i <= segments; i++) {
        // Compute points on the semicircle (180-degree range)
        float angle = (float(i) / float(segments)) * PI;
        vec2 offset = cos(angle) * perpendicular + sin(angle) * direction;
        offset *= (line_width / 2.0);

        g_color = color;
        gl_Position = center + vec4(offset, 0.0, 0.0);
        EmitVertex();

        // Add the center point to form a triangle fan
        if (i > 0) {
            g_color = color;
            gl_Position = center;
            EmitVertex();
        }
    }
    EndPrimitive();
}

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // p.z is z_ndc in [-1, 1], -1 is near, 1 is far
    float avg_z_ndc = (p0.z + p1.z) / 2.0;
    // Remap z from [-1, 1] to [1, 0] for scaling factor
    float depth_scale = (1.0 - avg_z_ndc) / 2.0;

    depth_scale = clamp(depth_scale, 0.0, 1.0);
    float dynamic_line_width = u_line_width * depth_scale;

    vec2 p0_ndc = p0.xy / p0.w;
    vec2 p1_ndc = p1.xy / p1.w;

    vec2 dir = normalize(p1_ndc - p0_ndc);
    vec2 normal = vec2(-dir.y, dir.x);

    float half_width = dynamic_line_width / 2.0;

    vec4 offset = vec4(normal * half_width, 0.0, 0.0);

    g_color = v_color[0];
    gl_Position = p0 + offset;
    EmitVertex();

    g_color = v_color[0];
    gl_Position = p0 - offset;
    EmitVertex();

    g_color = v_color[1];
    gl_Position = p1 + offset;
    EmitVertex();

    g_color = v_color[1];
    gl_Position = p1 - offset;
    EmitVertex();

    EndPrimitive();


    // Add a round cap at the start
    createRoundCap(p0, -dir, v_color[0], dynamic_line_width);

    // Add a round cap at the end
    createRoundCap(p1, dir, v_color[1], dynamic_line_width);
}
"""

fragment_shader_code = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

# Edge fragment shader for InstancedVoxels: paints every cube-edge fragment a
# single uniform color (default black), ignoring the per-voxel fill color passed
# down as g_color. This is what makes voxels read as distinct cubes.
#
# The never-taken branch on g_color keeps the color varying (and thus the
# in_color vertex attribute it derives from) from being optimized out by the
# GLSL linker — otherwise binding 'in_color' on the shared instance VBO fails.
instanced_edge_fragment_shader_code = """
#version 330
uniform vec3 u_edge_color;
in vec3 g_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(u_edge_color, 1.0);
    if (g_color.r > 1e20) frag_color = vec4(g_color, 1.0);
}
"""

# Ftheta vertex shader: passes camera-space 3D point through to geometry shader
ftheta_vertex_shader_code = """
#version 330
in vec3 in_cam;
in vec3 in_color;
out vec3 v_cam;
out vec3 v_color_vs;
void main() {
    v_cam = in_cam;
    v_color_vs = in_color;
    gl_Position = vec4(0.0);
}
"""

# Ftheta geometry shader: takes 2 camera-space endpoints, subdivides, projects
# via ftheta polynomial, emits thickened triangle strip.
ftheta_geometry_shader_code = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 24) out;

uniform float c0, c1, c2, c3, c4, c5;
uniform float cam_cx, cam_cy, cam_lc, cam_ld, cam_le;
uniform float image_width, image_height, depth_max, u_line_width;
uniform int n_steps;

in vec3 v_cam[];
in vec3 v_color_vs[];
out vec3 g_color;

vec4 project_ftheta(vec3 cam) {
    float x = cam.x, y = cam.y, z = cam.z;
    float ray_norm = sqrt(x*x + y*y + z*z);
    float r = sqrt(x*x + y*y);
    float cos_a = clamp(z / max(ray_norm, 1e-6), -1.0, 1.0);
    float alpha = acos(cos_a);
    float delta = c0 + alpha*(c1 + alpha*(c2 + alpha*(c3 + alpha*(c4 + alpha*c5))));
    float scale = (r > 1e-6) ? delta / r : 0.0;
    float px_rel = scale * x;
    float py_rel = scale * y;
    float px = cam_lc * px_rel + cam_ld * py_rel + cam_cx;
    float py = cam_le * px_rel + py_rel + cam_cy;
    float x_ndc = 2.0 * px / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * py / image_height;
    float z_ndc = (z / depth_max) * 2.0 - 1.0;
    return vec4(x_ndc, y_ndc, z_ndc, 1.0);
}

void main() {
    vec3 p0 = v_cam[0];
    vec3 p1 = v_cam[1];
    if (p0.z <= 0.0 && p1.z <= 0.0) return;

    vec4 positions[11];
    for (int i = 0; i <= n_steps; i++) {
        float t = float(i) / float(n_steps);
        positions[i] = project_ftheta(mix(p0, p1, t));
    }

    for (int i = 0; i <= n_steps; i++) {
        vec4 pos = positions[i];
        vec2 dir;
        if (i < n_steps) dir = normalize(positions[i+1].xy - pos.xy);
        else dir = normalize(pos.xy - positions[i-1].xy);
        vec2 normal = vec2(-dir.y, dir.x);

        float t = float(i) / float(n_steps);
        float avg_z_ndc = pos.z;
        float depth_scale = clamp((1.0 - avg_z_ndc) / 2.0, 0.0, 1.0);
        float half_w = u_line_width * depth_scale / 2.0;
        vec4 offset = vec4(normal * half_w, 0.0, 0.0);

        g_color = mix(v_color_vs[0], v_color_vs[1], t);
        gl_Position = pos + offset; EmitVertex();
        gl_Position = pos - offset; EmitVertex();
    }
    EndPrimitive();
}
"""

# no color as output, only update the position for updating depth buffer
depth_only_vertex_shader_code = """
#version 330
in vec3 in_pos;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
void main() {
    float x_ndc = 2 * in_pos.x / image_width - 1;
    float y_ndc = 1 - 2 * in_pos.y / image_height;
    gl_Position = vec4(x_ndc, y_ndc, in_pos.z / depth_max, 1.0);
}
"""

# no color as output, only update the position for updating depth buffer
depth_only_fragment_shader_code = """
#version 330
void main() {
    // Write only depth, with no color output
}
"""

# Worldspace shaders: geometry is uploaded to the GPU once in world coordinates
# and projected per-frame in the vertex shader (camera pose passed as a uniform),
# eliminating per-frame CPU projection. The vertex shaders set `v_behind` for
# points behind the camera. Geometry shaders apply optional below-level culling,
# then expand lines into rounded quads or pass through surviving triangles.
worldspace_geometry_shader_code = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 42) out;

uniform float u_line_width;
uniform float u_below_level_filter;
uniform vec3 u_level_ground_point;
uniform vec3 u_level_ground_normal;
uniform float u_level_base_margin;
uniform float u_level_margin_slope;
uniform float u_level_max_margin;
uniform float u_level_local_range;
uniform float u_level_far_margin;
uniform float u_level_transition;

in vec3 v_color[];
in float v_behind[];
in vec3 v_level_pos[];
out vec3 g_color;

bool belowLevel(vec3 position) {
    vec3 relative = position - u_level_ground_point;
    float height = dot(relative, u_level_ground_normal);
    vec3 tangent = relative - height * u_level_ground_normal;
    float distance = length(tangent);
    float localMargin = min(
        u_level_max_margin,
        u_level_base_margin + distance * u_level_margin_slope
    );
    float blend = clamp(
        (distance - u_level_local_range) / u_level_transition,
        0.0,
        1.0
    );
    float margin = mix(localMargin, u_level_far_margin, blend);
    return height < -margin;
}

void createRoundCap(vec4 center, vec2 direction, vec3 color, float line_width) {
    const int segments = 8;
    const float PI = 3.1415926;

    vec2 perpendicular = vec2(-direction.y, direction.x);

    for (int i = 0; i <= segments; i++) {
        float angle = (float(i) / float(segments)) * PI;
        vec2 offset = cos(angle) * perpendicular + sin(angle) * direction;
        offset *= (line_width / 2.0);

        g_color = color;
        gl_Position = center + vec4(offset, 0.0, 0.0);
        EmitVertex();

        if (i > 0) {
            g_color = color;
            gl_Position = center;
            EmitVertex();
        }
    }
    EndPrimitive();
}

void main() {
    if (v_behind[0] > 0.5 || v_behind[1] > 0.5) return;
    if (u_below_level_filter > 0.5 &&
        belowLevel(v_level_pos[0]) && belowLevel(v_level_pos[1])) return;

    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    if (abs(p0.x) > 3.0 || abs(p0.y) > 3.0 ||
        abs(p1.x) > 3.0 || abs(p1.y) > 3.0) return;

    float avg_z_ndc = (p0.z + p1.z) / 2.0;
    float depth_scale = (1.0 - avg_z_ndc) / 2.0;

    depth_scale = clamp(depth_scale, 0.0, 1.0);
    float dynamic_line_width = u_line_width * depth_scale;

    vec2 p0_ndc = p0.xy / p0.w;
    vec2 p1_ndc = p1.xy / p1.w;

    vec2 dir = normalize(p1_ndc - p0_ndc);
    vec2 normal = vec2(-dir.y, dir.x);

    float half_width = dynamic_line_width / 2.0;

    vec4 offset = vec4(normal * half_width, 0.0, 0.0);

    g_color = v_color[0];
    gl_Position = p0 + offset;
    EmitVertex();

    g_color = v_color[0];
    gl_Position = p0 - offset;
    EmitVertex();

    g_color = v_color[1];
    gl_Position = p1 + offset;
    EmitVertex();

    g_color = v_color[1];
    gl_Position = p1 - offset;
    EmitVertex();

    EndPrimitive();

    createRoundCap(p0, -dir, v_color[0], dynamic_line_width);
    createRoundCap(p1, dir, v_color[1], dynamic_line_width);
}
"""

worldspace_triangle_geometry_shader_code = """
#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform float u_below_level_filter;
uniform vec3 u_level_ground_point;
uniform vec3 u_level_ground_normal;
uniform float u_level_base_margin;
uniform float u_level_margin_slope;
uniform float u_level_max_margin;
uniform float u_level_local_range;
uniform float u_level_far_margin;
uniform float u_level_transition;

in vec3 v_color[];
in float v_behind[];
in vec3 v_level_pos[];
out vec3 g_color;

bool belowLevel(vec3 position) {
    vec3 relative = position - u_level_ground_point;
    float height = dot(relative, u_level_ground_normal);
    vec3 tangent = relative - height * u_level_ground_normal;
    float distance = length(tangent);
    float localMargin = min(
        u_level_max_margin,
        u_level_base_margin + distance * u_level_margin_slope
    );
    float blend = clamp(
        (distance - u_level_local_range) / u_level_transition,
        0.0,
        1.0
    );
    float margin = mix(localMargin, u_level_far_margin, blend);
    return height < -margin;
}

void main() {
    if (v_behind[0] > 0.5 && v_behind[1] > 0.5 && v_behind[2] > 0.5) return;
    if (u_below_level_filter > 0.5 &&
        belowLevel(v_level_pos[0]) &&
        belowLevel(v_level_pos[1]) &&
        belowLevel(v_level_pos[2])) return;

    for (int i = 0; i < 3; i++) {
        g_color = v_color[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""

worldspace_ftheta_vertex_shader_code = """
#version 330

in vec3 in_pos;
in vec3 in_color;
in vec3 in_level_pos;

uniform mat4 u_world_to_camera;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
uniform float use_depth_gradient;
uniform float u_camera_depth_bias_m;

uniform vec2 u_optical_center;
uniform float u_fw_poly[6];
uniform float u_max_ray_angle;
uniform vec2 u_max_ray_distortion;
uniform mat2 u_A;

out vec3 v_color;
out float v_behind;
out vec3 v_level_pos;

void main() {
    vec4 cam_pos = u_world_to_camera * vec4(in_pos, 1.0);
    float depth = cam_pos.z;

    v_behind = (depth <= 0.0) ? 1.0 : 0.0;
    v_level_pos = in_level_pos;

    float xy_norm = length(cam_pos.xy);
    float ray_norm = length(cam_pos.xyz);
    if (ray_norm < 1e-7) ray_norm = 1e-7;
    float cos_alpha = cam_pos.z / ray_norm;

    float alpha;
    if (cos_alpha > -1.0 && cos_alpha < 1.0) {
        alpha = acos(cos_alpha);
    } else {
        alpha = xy_norm;
    }

    float delta;
    if (alpha <= u_max_ray_angle) {
        delta = u_fw_poly[0] + alpha * (u_fw_poly[1] + alpha * (u_fw_poly[2]
              + alpha * (u_fw_poly[3] + alpha * (u_fw_poly[4] + alpha * u_fw_poly[5]))));
    } else {
        delta = u_max_ray_distortion.x
              + (alpha - u_max_ray_angle) * u_max_ray_distortion.y;
    }

    vec2 pixel;
    if (xy_norm > 1e-7) {
        pixel = (delta / xy_norm) * cam_pos.xy;
    } else {
        pixel = cam_pos.xy;
    }

    vec2 px_py = u_A * pixel + u_optical_center;

    float x_ndc = 2.0 * px_py.x / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * px_py.y / image_height;
    float z_ndc = ((depth + u_camera_depth_bias_m) / depth_max) * 2.0 - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);

    float depth_z_ndc = (depth / depth_max) * 2.0 - 1.0;
    float r = use_depth_gradient > 0.5
        ? clamp((1.0 - depth_z_ndc) / 2.0, 0.0, 1.0)
        : 1.0;
    v_color = in_color * r;
}
"""

worldspace_pinhole_vertex_shader_code = """
#version 330

in vec3 in_pos;
in vec3 in_color;
in vec3 in_level_pos;

uniform mat4 u_world_to_camera;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
uniform float use_depth_gradient;
uniform float u_camera_depth_bias_m;

uniform float u_fx;
uniform float u_fy;
uniform float u_cx;
uniform float u_cy;

out vec3 v_color;
out float v_behind;
out vec3 v_level_pos;

void main() {
    vec4 cam_pos = u_world_to_camera * vec4(in_pos, 1.0);
    float depth = cam_pos.z;

    v_behind = (depth <= 0.0) ? 1.0 : 0.0;
    v_level_pos = in_level_pos;

    float proj_depth = depth;
    if (abs(proj_depth) < 1e-4) proj_depth = 1e-4;
    float px = u_fx * cam_pos.x / proj_depth + u_cx;
    float py = u_fy * cam_pos.y / proj_depth + u_cy;

    float x_ndc = 2.0 * px / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * py / image_height;
    float z_ndc = ((depth + u_camera_depth_bias_m) / depth_max) * 2.0 - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);

    float depth_z_ndc = (depth / depth_max) * 2.0 - 1.0;
    float r = use_depth_gradient > 0.5
        ? clamp((1.0 - depth_z_ndc) / 2.0, 0.0, 1.0)
        : 1.0;
    v_color = in_color * r;
}
"""


# WMv4.1 instanced-voxel shaders (static occupancy). A single unit-cube template
# (8 verts) plus per-instance data (center, height, color) is uploaded once; the
# vertex shader scales the cube by (u_box_size, u_box_size, in_height), transforms
# it to camera space via u_world_to_camera, and projects it (pinhole or ftheta)
# entirely on the GPU. Per-instance forward/lateral culling collapses out-of-range
# towers to a degenerate vertex. Depth attenuation matches the other color shaders.
instanced_pinhole_vertex_shader_code = """
#version 330

in vec3 in_vert;
in vec3 in_center;
in float in_height;
in vec3 in_color;

uniform mat4 u_world_to_camera;
uniform float u_box_size;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
uniform float use_depth_gradient;
uniform float u_max_distance;
uniform float u_max_lateral;
uniform float u_fx;
uniform float u_fy;
uniform float u_cx;
uniform float u_cy;

out vec3 v_color;

void main() {
    vec4 center_cam = u_world_to_camera * vec4(in_center, 1.0);
    if (center_cam.z <= 0.0 ||
        (u_max_distance > 0.0 && center_cam.z > u_max_distance) ||
        (u_max_lateral > 0.0 && abs(center_cam.x) > u_max_lateral)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_color = vec3(0.0);
        return;
    }

    vec3 world_pos = in_center + vec3(
        in_vert.x * u_box_size,
        in_vert.y * u_box_size,
        in_vert.z * in_height
    );

    vec4 cam_pos = u_world_to_camera * vec4(world_pos, 1.0);
    float depth = cam_pos.z;

    float proj_depth = depth;
    if (abs(proj_depth) < 1e-4) proj_depth = 1e-4;
    float px = u_fx * cam_pos.x / proj_depth + u_cx;
    float py = u_fy * cam_pos.y / proj_depth + u_cy;

    float x_ndc = 2.0 * px / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * py / image_height;
    float z_ndc = (depth / depth_max) * 2.0 - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);

    float r = use_depth_gradient > 0.5
        ? clamp((1.0 - z_ndc) / 2.0, 0.0, 1.0)
        : 1.0;
    v_color = in_color * r;
}
"""

instanced_ftheta_vertex_shader_code = """
#version 330

in vec3 in_vert;
in vec3 in_center;
in float in_height;
in vec3 in_color;

uniform mat4 u_world_to_camera;
uniform float u_box_size;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
uniform float use_depth_gradient;
uniform float u_max_distance;
uniform float u_max_lateral;

uniform vec2 u_optical_center;
uniform float u_fw_poly[6];
uniform float u_max_ray_angle;
uniform vec2 u_max_ray_distortion;
uniform mat2 u_A;

out vec3 v_color;

void main() {
    vec4 center_cam = u_world_to_camera * vec4(in_center, 1.0);
    if (center_cam.z <= 0.0 ||
        (u_max_distance > 0.0 && center_cam.z > u_max_distance) ||
        (u_max_lateral > 0.0 && abs(center_cam.x) > u_max_lateral)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_color = vec3(0.0);
        return;
    }

    vec3 world_pos = in_center + vec3(
        in_vert.x * u_box_size,
        in_vert.y * u_box_size,
        in_vert.z * in_height
    );

    vec4 cam_pos = u_world_to_camera * vec4(world_pos, 1.0);
    float depth = cam_pos.z;

    float xy_norm = length(cam_pos.xy);
    float ray_norm = length(cam_pos.xyz);
    if (ray_norm < 1e-7) ray_norm = 1e-7;
    float cos_alpha = cam_pos.z / ray_norm;

    float alpha;
    if (cos_alpha > -1.0 && cos_alpha < 1.0) {
        alpha = acos(cos_alpha);
    } else {
        alpha = xy_norm;
    }

    float delta;
    if (alpha <= u_max_ray_angle) {
        delta = u_fw_poly[0] + alpha * (u_fw_poly[1] + alpha * (u_fw_poly[2]
              + alpha * (u_fw_poly[3] + alpha * (u_fw_poly[4] + alpha * u_fw_poly[5]))));
    } else {
        delta = u_max_ray_distortion.x
              + (alpha - u_max_ray_angle) * u_max_ray_distortion.y;
    }

    vec2 pixel;
    if (xy_norm > 1e-7) {
        pixel = (delta / xy_norm) * cam_pos.xy;
    } else {
        pixel = cam_pos.xy;
    }

    vec2 px_py = u_A * pixel + u_optical_center;

    float x_ndc = 2.0 * px_py.x / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * px_py.y / image_height;
    float z_ndc = (depth / depth_max) * 2.0 - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);

    float r = use_depth_gradient > 0.5
        ? clamp((1.0 - z_ndc) / 2.0, 0.0, 1.0)
        : 1.0;
    v_color = in_color * r;
}
"""
