# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

from cosmos_auto.wm.camera.base import CameraBase
from cosmos_auto.wm.graphics_utils import (
    Geometry2D,
    LineSegment2D,
    Polygon2D,
    TriangleList2D,
    render_geometries,
)
from cosmos_auto.wm.occupancy_utils import points_below_ground_plane
from cosmos_auto.wm.pcd_utils import (
    filter_by_height_relative_to_ego,
    interpolate_polyline_to_points,
    refine_triangle_mesh_3d,
    triangulate_polygon_3d,
)
from cosmos_auto.wm.wds_utils import get_sample


def load_hdmap_colors(version="v1"):
    """
    Load hdmap colors based on the specified version.

    Args:
        version: str, version key from config_color_hdmap.json
                Available versions: 'v1', 'v3'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent / "color_scheme" / "config_color_hdmap.json"
    with open(config_path, "r") as f:
        hdmap_config = json.load(f)

    if version not in hdmap_config:
        available_versions = list(hdmap_config.keys())
        raise ValueError(f"Version '{version}' not found in hdmap config. Available versions: {available_versions}")

    return hdmap_config[version]


def load_traffic_light_colors(version="v2"):
    """
    Load traffic light colors based on the specified version.

    Args:
        version: str, version key from config_color_traffic_light.json
                Available versions: 'v2'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent / "color_scheme" / "config_color_traffic_light.json"
    with open(config_path, "r") as f:
        tl_config = json.load(f)

    if version not in tl_config:
        available_versions = list(tl_config.keys())
        raise ValueError(
            f"Version '{version}' not found in traffic light config. Available versions: {available_versions}"
        )

    return tl_config[version]


def load_sign_colors(version="v4"):
    """
    Load traffic sign colors based on the specified version.

    Args:
        version: str, version key from config_color_sign.json
                Available versions: 'v4'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent / "color_scheme" / "config_color_sign.json"
    with open(config_path, "r") as f:
        sign_config = json.load(f)

    if version not in sign_config:
        available_versions = list(sign_config.keys())
        raise ValueError(f"Version '{version}' not found in sign config. Available versions: {available_versions}")

    return sign_config[version]


def load_road_marking_colors(version="v4"):
    """
    Load WMv4 road-marking colors from config_color_road_marking.json.

    Args:
        version: str, version key from config_color_road_marking.json (e.g. 'v4').

    Returns:
        dict: category -> [r, g, b] for the seven arrow directions plus
            "stop" and "keep_clear".
    """
    config_path = Path(__file__).parent / "color_scheme" / "config_color_road_marking.json"
    with open(config_path, "r") as f:
        rm_config = json.load(f)

    if version not in rm_config:
        available_versions = list(rm_config.keys())
        raise ValueError(
            f"Version '{version}' not found in road-marking config. Available versions: {available_versions}"
        )

    return rm_config[version]


# Default configurations for backward compatibility
MINIMAP_TO_RGB = load_hdmap_colors("v1")

MINIMAP_TO_TYPE = json.load(open(Path(__file__).parent / "color_scheme" / "config_type_hdmap.json"))

LANELINE_TYPE_TO_RGB = json.load(open(Path(__file__).parent / "color_scheme" / "config_color_laneline.json"))

TL_STATUS_TO_RGB = load_traffic_light_colors("v2")

MINIMAP_TO_SEMANTIC_LABEL = {
    "lanelines": 5,
    "lanes": 5,
    "poles": 9,
    "road_boundaries": 5,
    "wait_lines": 10,
    "crosswalks": 5,
    "road_markings": 10,
}


def extract_vertices(minimap_data, vertices_list=None):
    if vertices_list is None:
        vertices_list = []

    if isinstance(minimap_data, dict):
        for key, value in minimap_data.items():
            if key == "vertices":
                vertices_list.append(value)
            else:
                extract_vertices(value, vertices_list)

    elif isinstance(minimap_data, list):
        for item in minimap_data:
            extract_vertices(item, vertices_list)

    return vertices_list


def is_actual_wait_line_label(label):
    """
    Check whether a wait-line label is explicit rather than implicit.

    We treat missing `is_implicit` as actual for backward compatibility across
    heterogeneous RDS-HQ exports.

    Args:
        label: dict or any, label payload from minimap data

    Returns:
        bool: True for explicit (actual) wait-line labels
    """
    if not isinstance(label, dict):
        return True

    shape3d = label.get("labelData", {}).get("shape3d", {})
    attributes = shape3d.get("attributes", [])
    if not isinstance(attributes, list):
        return True

    for attribute in attributes:
        if attribute.get("name") != "is_implicit":
            continue

        implicit_flag = attribute.get("flag", None)
        if isinstance(implicit_flag, bool):
            return not implicit_flag
        if isinstance(implicit_flag, str):
            return implicit_flag.lower() not in {"true", "1", "yes"}
        if isinstance(implicit_flag, (int, float)):
            return not bool(implicit_flag)
        return True

    return True


def get_type_from_name(minimap_name):
    """
    Args:
        minimap_name: str, name of the minimap

    Returns:
        minimap_type: str, type of the minimap
    """
    if minimap_name in MINIMAP_TO_TYPE:
        return MINIMAP_TO_TYPE[minimap_name]
    else:
        raise ValueError(f"Invalid minimap name: {minimap_name}")


# Order of cuboid vertices replicated to form a 16-point polyline that traces the
# cuboid edges. Shared between cuboid3d_to_polyline() and the v4 per-vertex color
# construction in create_traffic_{light_status,sign}_geometry_objects_from_data_v4
# so the color array index order stays in sync with the polyline geometry.
CUBOID_POLYLINE_INDICES = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]


def cuboid3d_to_polyline(cuboid3d_eight_vertices):
    """
    Convert cuboid3d to polyline

    Args:
        cuboid3d_eight_vertices: np.ndarray, shape (8, 3), dtype=np.float32,
            eight vertices of the cuboid3d

    Returns:
        polyline: np.ndarray, shape (16, 3), dtype=np.float32,
            polyline vertices
    """
    if isinstance(cuboid3d_eight_vertices, list):
        cuboid3d_eight_vertices = np.array(cuboid3d_eight_vertices)

    return np.array(cuboid3d_eight_vertices)[CUBOID_POLYLINE_INDICES]


def simplify_minimap(minimap_wds_file):
    """
    Args:
        minimap_wds_file: path to the webdataset file containing minimap data.
        minimap can be
        -> polygons, e.g., crosswalks, road_markings
        -> polylines, e.g., lanelines, lanes, road_boundaries, wait_lines, poles
        -> cuboid3d, e.g., traffic_signs, traffic_lights

    Returns:
        minimap_data_wo_meta_info: list of np.ndarray
            each item represents a polyline or polygon with shape [N, 3]
            cuboid3d is converted to polyline (or you can also render it as a polygon)
    """

    minimap_raw_data = get_sample(minimap_wds_file)
    minimap_key_name = [key for key in minimap_raw_data.keys() if key.endswith(".json")][0]
    minimap_name = minimap_key_name.split(".")[0]
    minimap_data = minimap_raw_data[minimap_key_name]

    # Keep only explicit (actual) wait-lines when the discriminator exists.
    # If `is_implicit` is absent, keep labels to support mixed dataset schemas.
    if minimap_name == "wait_lines" and isinstance(minimap_data, dict):
        labels = minimap_data.get("labels", None)
        if isinstance(labels, list):
            minimap_data = {**minimap_data, "labels": [label for label in labels if is_actual_wait_line_label(label)]}

    minimap_data_wo_meta_info = extract_vertices(minimap_data)

    # close the polygon
    if get_type_from_name(minimap_name) == "polygon":
        for single_polygon in minimap_data_wo_meta_info:
            single_polygon.append(single_polygon[0])

    # 8 vertices, we can also make it polyline, just repeat some edges!
    # Then we can draw it as a polyline or polygon
    if get_type_from_name(minimap_name) == "cuboid3d":
        # connected_vertices_indices = [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
        for i, eight_vertices in enumerate(minimap_data_wo_meta_info):
            connected_polyline = cuboid3d_to_polyline(eight_vertices)
            minimap_data_wo_meta_info[i] = connected_polyline

    # for each polyline, if they are list, convert them to np.ndarray
    for i, polyline_or_polygon in enumerate(minimap_data_wo_meta_info):
        if isinstance(polyline_or_polygon, list):
            minimap_data_wo_meta_info[i] = np.array(polyline_or_polygon)

    return minimap_data_wo_meta_info, minimap_name


def create_minimap_projection(
    minimap_name, minimap_data_wo_meta_info, camera_poses, camera_model, hdmap_color_version="v1"
):
    """
    Args:
        minimap_name: str, name of the minimap
        minimap_data_wo_meta_info: list of np.ndarray, results from simplify_minimap
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model
        hdmap_color_version: str, version key for hdmap colors from config_color_hdmap.json

    Returns:
        minimaps_projection: np.ndarray,
            shape (N, H, W, 3), dtype=np.uint8, projected minimap data across N frames
    """
    # Load colors based on the specified version
    minimap_to_rgb = load_hdmap_colors(hdmap_color_version)

    image_height, image_width = camera_model.height, camera_model.width
    cprint(f"Processing minimap {minimap_name} with shape {image_height}x{image_width}", "green")

    minimap_type = get_type_from_name(minimap_name)

    if minimap_type == "polygon":
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
        )
    elif minimap_type == "polyline":
        # to smooth the polyline.
        if minimap_name == "lanelines" or minimap_name == "road_boundaries":
            segment_interval = 0.8
        else:
            segment_interval = 0

        projection_images = camera_model.draw_line_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
            segment_interval=segment_interval,
        )
    elif minimap_type == "cuboid3d":
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
        )
    else:
        raise ValueError(f"Invalid minimap type: {minimap_type}")

    return projection_images


def create_minimap_projection_gl(minimap_wds_files, camera_poses, camera_model, depth_max, hdmap_color_version="v1"):
    """
    We use moderngl to render the minimap, and taking the line color and type into account.

    Args:
        minimap_wds_files: list of str,
            path to the webdataset file containing minimap data.

        camera_poses: np.ndarray, shape (T, 4, 4), dtype=np.float32, camera poses of T frames
        camera_model: CameraModel, camera model
        depth_max: float, maximum depth for rendering
        hdmap_color_version: str, version key for hdmap colors from config_color_hdmap.json

    Returns:
        minimaps_projection: np.ndarray,
            shape (T, H, W, 3), dtype=np.uint8, projected minimap data across T frames

    """
    minimap_name_to_minimap_data, minimap_to_rgb = prepare_minimap_data(minimap_wds_files, hdmap_color_version)

    minimaps_projection = []
    for camera_pose in camera_poses:
        geometry_objects = create_minimap_geometry_objects_from_data(
            minimap_name_to_minimap_data,
            camera_pose,
            camera_model,
            minimap_to_rgb,
        )
        minimaps_projection_one_frame = render_geometries(
            geometry_objects,
            camera_model.height,
            camera_model.width,
            depth_max,
            depth_gradient=True,
        )
        minimaps_projection.append(minimaps_projection_one_frame)

    return np.stack(minimaps_projection, axis=0)


def create_laneline_type_projection_gl(
    laneline_wds_file,
    camera_poses,
    camera_model,
    depth_max,
):
    """
    Args:
        laneline_wds_file: str, path to the webdataset file containing laneline data.
        camera_poses: np.ndarray, shape (T, 4, 4), dtype=np.float32, camera poses of T frames
        camera_model: CameraModel, ftheta or pinhole
        depth_max: float, maximum depth value
    """
    laneline_sample = get_sample(laneline_wds_file)
    laneline_data_list = laneline_sample["lanelines.json"]["labels"]

    laneline_line_segment_list = []

    # in RDS-HQ format, if there is only one element, it is actually a label for no laneline.
    if len(laneline_data_list) == 1:
        return np.zeros((len(camera_poses), camera_model.height, camera_model.width, 3), dtype=np.uint8)

    for laneline_data in laneline_data_list:
        attribute_list = laneline_data["labelData"]["shape3d"]["attributes"]
        color = None
        style = None

        for attribute in attribute_list:
            if attribute["name"] == "colors":
                color = attribute["enumsList"]["enumsList"][0]
            if attribute["name"] == "styles":
                style = attribute["enumsList"]["enumsList"][0]

        type_name = f"{color} {style}"

        if type_name not in LANELINE_TYPE_TO_RGB:
            type_name = "OTHER"

        rgb_float = np.array(LANELINE_TYPE_TO_RGB[type_name]) / 255.0

        polyline = laneline_data["labelData"]["shape3d"]["polyline3d"]["vertices"]
        polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.8)
        line_segment = np.stack([polyline_subdivided[0:-1], polyline_subdivided[1:]], axis=1)  # [N, 2, 3]
        laneline_line_segment_list.append((line_segment, rgb_float))

    laneline_type_projection = []  # record all frames
    for camera_pose in camera_poses:
        laneline_geometry_object_list = []
        for line_segment, rgb_float in laneline_line_segment_list:
            xy_and_depth = camera_model.get_xy_and_depth(line_segment.reshape(-1, 3), camera_pose).reshape(
                -1, 2, 3
            )  # [N, 2, 3]
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]

            if len(valid_xy_and_depth) == 0:
                continue

            # create LineSegment2D geometry object for each polyline
            laneline_geometry_object_list.append(LineSegment2D(valid_xy_and_depth, base_color=rgb_float, line_width=12))

        laneline_type_projection_one_frame = render_geometries(
            laneline_geometry_object_list, camera_model.height, camera_model.width, depth_max, depth_gradient=True
        )
        laneline_type_projection.append(laneline_type_projection_one_frame)

    laneline_type_projection = np.stack(laneline_type_projection, axis=0).astype(np.uint8)

    return laneline_type_projection


def create_traffic_light_status_projection_gl(
    traffic_light_position_wds_file,
    traffic_light_per_frame_status_wds_file,
    all_camera_poses,
    valid_frame_ids,
    camera_model,
    depth_max,
    traffic_light_color_version="v2",
):
    # Reuse new helpers to minimize duplicate code
    tl_position_list, tl_status_dict, tl_status_to_rgb, _ = prepare_traffic_light_status_data(
        traffic_light_position_wds_file,
        traffic_light_per_frame_status_wds_file,
        traffic_light_color_version,
    )
    if tl_status_dict is None:
        print(
            f"Traffic light status wds file {traffic_light_per_frame_status_wds_file} does not exist. Return all zeros."
        )
        return np.zeros((len(valid_frame_ids), camera_model.height, camera_model.width, 3), dtype=np.uint8)

    if len(tl_position_list) == 1:
        # in RDS-HQ format, if there is only one element, it is actually a label for no traffic light.
        return np.zeros((len(valid_frame_ids), camera_model.height, camera_model.width, 3), dtype=np.uint8)

    traffic_light_status_projection = []
    for frame_id in valid_frame_ids:
        camera_pose = all_camera_poses[frame_id]
        geometry_objects = create_traffic_light_status_geometry_objects_from_data(
            tl_position_list,
            tl_status_dict,
            frame_id,
            camera_pose,
            camera_model,
            tl_status_to_rgb,
        )
        if len(geometry_objects) == 0:
            traffic_light_status_projection.append(
                np.zeros((camera_model.height, camera_model.width, 3), dtype=np.uint8)
            )
        else:
            traffic_light_status_projection.append(
                render_geometries(
                    geometry_objects,
                    camera_model.height,
                    camera_model.width,
                    depth_max,
                    depth_gradient=True,
                )
            )

    return np.stack(traffic_light_status_projection, axis=0)


def cuboid3d_update_vertices_remove_others(cuboid3d):
    """
    This is specific to the cuboid3d label of traffic_signs and traffic_lights.

    Args:
        cuboid3d: dict
            {
                'center': {'x': 180.09763, 'y': 83.11682, 'z': -16.458818},
                'orientation': {'x': -0.071179695, 'y': 0.0033680226, 'z': -0.36912337}, # yaw, pitch, roll
                'dimensions': {'x': 0.01, 'y': 0.64360946, 'z': 0.7865788}, # length, width, height
                'vertices': [{}, {}, {}, {}, {}, {}, {}, {}] # usually missing,
            }

    Returns:
        updated cuboid3d: dict
            remove 'center', 'orientation', 'dimensions'
            add 'vertices' with 8 vertices
    """
    max_xyz = np.array(
        [cuboid3d["dimensions"]["x"] / 2.0, cuboid3d["dimensions"]["y"] / 2.0, cuboid3d["dimensions"]["z"] / 2.0]
    )
    min_xyz = -max_xyz

    # just a kind reminder, the order of 8 vertices here is different from build_cuboid_bounding_box() in bbox_utils.py
    # but it does not matter.
    vertices_of_cuboid = np.zeros((8, 3), dtype=np.float32)
    vertices_of_cuboid[0] = np.array([min_xyz[0], min_xyz[1], min_xyz[2]])
    vertices_of_cuboid[1] = np.array([min_xyz[0], max_xyz[1], min_xyz[2]])
    vertices_of_cuboid[2] = np.array([max_xyz[0], max_xyz[1], min_xyz[2]])
    vertices_of_cuboid[3] = np.array([max_xyz[0], min_xyz[1], min_xyz[2]])
    vertices_of_cuboid[4] = np.array([min_xyz[0], min_xyz[1], max_xyz[2]])
    vertices_of_cuboid[5] = np.array([min_xyz[0], max_xyz[1], max_xyz[2]])
    vertices_of_cuboid[6] = np.array([max_xyz[0], max_xyz[1], max_xyz[2]])
    vertices_of_cuboid[7] = np.array([max_xyz[0], min_xyz[1], max_xyz[2]])

    yaw, pitch, roll = cuboid3d["orientation"]["x"], cuboid3d["orientation"]["y"], cuboid3d["orientation"]["z"]
    rotation_matrix = R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()

    # Apply rotation.
    vertices_of_cuboid = (rotation_matrix @ vertices_of_cuboid.T).T

    # Add center translation
    center = np.array([cuboid3d["center"]["x"], cuboid3d["center"]["y"], cuboid3d["center"]["z"]])
    vertices_of_cuboid = vertices_of_cuboid + center

    # following protobuf format
    cuboid3d["vertices"] = [{"x": vertex[0], "y": vertex[1], "z": vertex[2]} for vertex in vertices_of_cuboid.tolist()]
    cuboid3d.pop("center", None)
    cuboid3d.pop("orientation", None)
    cuboid3d.pop("dimensions", None)

    return cuboid3d


def convert_cuboid3d_recursive(minimap_message):
    """
    Recursively traverse minimap_message and update any cuboid3d entries with vertices.

    Args:
        minimap_message: dict/list, potentially nested structure that may contain cuboid3d entries

    Returns:
        Updated minimap_message with cuboid3d entries converted to include vertices
    """
    if isinstance(minimap_message, dict):
        # If this is a cuboid3d dict, update it
        if "cuboid3d" in minimap_message:
            minimap_message["cuboid3d"] = cuboid3d_update_vertices_remove_others(minimap_message["cuboid3d"])
        # Recursively process all dict values
        for key in minimap_message:
            minimap_message[key] = convert_cuboid3d_recursive(minimap_message[key])

    elif isinstance(minimap_message, list):
        # Recursively process all list items
        minimap_message = [convert_cuboid3d_recursive(item) for item in minimap_message]

    return minimap_message


def transform_decoded_label(decoded_label, transformation_matrix):
    """
    Apply transformation matrix to decoded label

    Args:
        decoded_label: dict,
            decoded label from decode_static_label, can have several hierarchies,
            but the last one is always numpy array with shape [N, 3]
        transformation_matrix: 4x4 numpy array,
            transformation matrix we want to apply to the numpy array
    Returns:
        dict: transformed decoded_label with the same structure, but numpy array transformed
    """

    def transform_vertices(vertices, transformation_matrix):
        """
        Args:
            vertices: numpy array, shape [N, 3]
            transformation_matrix: 4x4 numpy array
        Returns:
            numpy array, shape [N, 3]
        """
        if vertices.shape == (0,):
            return vertices

        # add 1 to the vertices
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformed_vertices = np.dot(transformation_matrix, vertices.T).T
        return transformed_vertices[:, :3]

    # recursively apply transformation matrix to the vertices
    def transform(decoded_label, transformation_matrix):
        if isinstance(decoded_label, np.ndarray):
            return transform_vertices(decoded_label, transformation_matrix)
        elif isinstance(decoded_label, dict):
            transformed_label = {}
            for key, value in decoded_label.items():
                transformed_label[key] = transform(value, transformation_matrix)
            return transformed_label
        else:
            raise ValueError(f"Unknown type in decoded_label: {type(decoded_label)}")

    return transform(decoded_label, transformation_matrix)


def prepare_minimap_data(minimap_wds_files, hdmap_color_version="v1"):
    """
    Preload and simplify minimap data from multiple webdataset files.

    Args:
        minimap_wds_files: list[str], paths to webdataset files
        hdmap_color_version: str, version of hdmap colors

    Returns:
        dict[str, list[np.ndarray]]: mapping from minimap name to simplified geometry arrays
        dict[str, list[np.ndarray]]: mapping from minimap name to RGB values
    """
    minimap_to_rgb = load_hdmap_colors(hdmap_color_version)

    minimap_name_to_minimap_data = {}
    for minimap_wds_file in minimap_wds_files:
        minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)
        minimap_name_to_minimap_data[minimap_name] = minimap_data_wo_meta_info
    return minimap_name_to_minimap_data, minimap_to_rgb


def preprocess_minimap_for_rendering(minimap_name_to_minimap_data, minimap_to_rgb, curve_subdivide=10):
    """
    Pre-compute all static minimap work (densification, triangulation, segment
    stacking) once before the frame loop.

    Polygon layers (crosswalks, road_markings) are densified along their edges so
    that, once each vertex is ftheta-projected in the worldspace vertex shader,
    the straight screen-space chords between vertices approximate the fisheye
    curve. curve_subdivide (1-10) controls the densification interval.

    Returns a list of layer dicts consumed by create_minimap_geometry_objects_fast().
    """
    polygon_interval = max(0.8, 10.8 - curve_subdivide * 1.0)  # 9.8m at 1, 0.8m at 10
    preprocessed_layers = []

    for minimap_name, minimap_data in minimap_name_to_minimap_data.items():
        minimap_type = get_type_from_name(minimap_name)
        color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0

        if minimap_type == "polyline":
            polyline_entries = []
            for polyline in minimap_data:
                subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.2)
                segments = np.stack([subdivided[:-1], subdivided[1:]], axis=1)
                polyline_entries.append((polyline, segments))
            if polyline_entries:
                preprocessed_layers.append(
                    {
                        "name": minimap_name,
                        "layer_type": "polyline",
                        "entries": polyline_entries,
                        "color": color_float,
                        "line_width": 5 if minimap_name == "poles" else 12,
                    }
                )

        elif minimap_type in ("polygon", "cuboid3d"):
            polygon_entries = []
            for polygon in minimap_data:
                # Both crosswalks and road_markings: densify edges then triangulate
                # so long polygon edges bend under fisheye projection.
                subdivided = interpolate_polyline_to_points(polygon, segment_interval=polygon_interval)
                triangles_3d = triangulate_polygon_3d(subdivided)
                vertices = triangles_3d.reshape(-1, 3)
                if len(vertices) == 0:
                    continue
                polygon_entries.append((polygon, vertices, True))
            if polygon_entries:
                preprocessed_layers.append(
                    {
                        "name": minimap_name,
                        "layer_type": minimap_type,
                        "entries": polygon_entries,
                        "color": color_float,
                    }
                )
        else:
            raise ValueError(f"Invalid minimap type: {minimap_type}")

    return preprocessed_layers


def build_worldspace_minimap_geometry(
    preprocessed_layers,
    camera_model,
    camera_pose_init=None,
    below_level_filter=False,
):
    """
    Build WorldSpaceLines and WorldSpaceTriangles for all minimap layers.

    Geometry is uploaded to the GPU once and projected per-frame by the
    vertex shader, eliminating per-frame CPU projection entirely.

    Args:
        preprocessed_layers: list[dict], from preprocess_minimap_for_rendering.
        camera_model: CameraModel (FThetaCamera or PinholeCamera).
        camera_pose_init: np.ndarray (4,4) or None, for one-time height filtering.

    Returns:
        list[WorldSpaceLines | WorldSpaceTriangles]
    """
    from cosmos_auto.wm.graphics_utils import WorldSpaceLines, WorldSpaceTriangles

    result = []

    for layer in preprocessed_layers:
        color = layer["color"]

        if layer["layer_type"] == "polyline":
            surviving = []
            pole_level_positions = []
            for original_polyline, segments in layer["entries"]:
                if (
                    not below_level_filter
                    and camera_pose_init is not None
                    and filter_by_height_relative_to_ego(
                        original_polyline, camera_model, camera_pose_init, camera_pose_init
                    )
                ):
                    continue
                surviving.append(segments)
                if below_level_filter and layer["name"] == "poles":
                    anchor = original_polyline[np.argmin(original_polyline[:, 2])]
                    pole_level_positions.append(np.broadcast_to(anchor, segments.shape))

            if not surviving:
                continue

            all_segs = np.concatenate(surviving, axis=0)
            n_verts = len(all_segs) * 2
            colors = np.tile(color, (n_verts, 1))

            result.append(
                WorldSpaceLines(
                    segments=all_segs,
                    colors=colors,
                    line_width=layer["line_width"],
                    camera_model=camera_model,
                    below_level_filter=below_level_filter,
                    level_positions=(np.concatenate(pole_level_positions, axis=0) if pole_level_positions else None),
                )
            )

        else:
            all_triangle_verts = []

            for original_polygon, processed_vertices, is_crosswalk in layer["entries"]:
                if (
                    not below_level_filter
                    and camera_pose_init is not None
                    and filter_by_height_relative_to_ego(
                        original_polygon, camera_model, camera_pose_init, camera_pose_init
                    )
                ):
                    continue

                if is_crosswalk:
                    all_triangle_verts.append(processed_vertices)
                else:
                    verts = np.asarray(processed_vertices)
                    if not np.allclose(verts[0], verts[-1]):
                        verts = np.concatenate([verts, verts[:1]], axis=0)
                    n = len(verts) - 1
                    if n < 3:
                        continue
                    indices = np.empty((n - 2, 3), dtype=int)
                    indices[:, 0] = 0
                    indices[:, 1] = np.arange(1, n - 1)
                    indices[:, 2] = np.arange(2, n)
                    all_triangle_verts.append(verts[indices.ravel()])

            if all_triangle_verts:
                combined = np.concatenate(all_triangle_verts, axis=0)
                tri_colors = np.tile(color, (len(combined), 1))
                result.append(
                    WorldSpaceTriangles(
                        vertices=combined,
                        colors=tri_colors,
                        camera_model=camera_model,
                        below_level_filter=below_level_filter,
                    )
                )

    return result


def create_minimap_geometry_objects_fast(
    preprocessed_layers,
    camera_pose,
    camera_model,
    camera_pose_init=None,
):
    """
    Build minimap geometry objects for a single frame using pre-processed data.

    Densification, triangulation, and segment stacking are already done;
    this function only runs the per-frame ego-height filter, camera projection,
    and depth culling.
    """
    all_geometry_objects = []

    for layer in preprocessed_layers:
        if layer["layer_type"] == "polyline":
            surviving = []
            for original_polyline, segments in layer["entries"]:
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    original_polyline, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                surviving.append(segments)

            if not surviving:
                continue

            all_segments = np.concatenate(surviving, axis=0)
            xy_and_depth = camera_model.get_xy_and_depth(all_segments.reshape(-1, 3), camera_pose).reshape(-1, 2, 3)

            valid = np.all(xy_and_depth[:, :, 2] >= 0, axis=1)
            valid_xy_and_depth = xy_and_depth[valid]
            if len(valid_xy_and_depth) == 0:
                continue

            all_geometry_objects.append(
                LineSegment2D(
                    valid_xy_and_depth,
                    base_color=layer["color"],
                    line_width=layer["line_width"],
                )
            )

        else:
            surviving_vertices = []
            surviving_counts = []
            surviving_crosswalk = []

            for original_polygon, processed_vertices, is_crosswalk in layer["entries"]:
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    original_polygon, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                surviving_vertices.append(processed_vertices)
                surviving_counts.append(len(processed_vertices))
                surviving_crosswalk.append(is_crosswalk)

            if not surviving_vertices:
                continue

            all_vertices = np.concatenate(surviving_vertices, axis=0)
            all_xy_and_depth = camera_model.get_xy_and_depth(all_vertices, camera_pose)

            start_idx = 0
            for i, vertex_count in enumerate(surviving_counts):
                polygon_xy_and_depth = all_xy_and_depth[start_idx : start_idx + vertex_count]
                start_idx += vertex_count

                if surviving_crosswalk[i]:
                    triangles_proj = polygon_xy_and_depth.reshape(-1, 3, 3)
                    valid = ~np.all(triangles_proj[:, :, 2] < 0, axis=1)
                    if valid.sum() > 0:
                        all_geometry_objects.append(TriangleList2D(triangles_proj[valid], base_color=layer["color"]))
                else:
                    if np.all(polygon_xy_and_depth[:, 2] < 0):
                        continue
                    all_geometry_objects.append(Polygon2D(polygon_xy_and_depth, base_color=layer["color"]))

    return all_geometry_objects


def create_minimap_geometry_objects_from_data(
    minimap_name_to_minimap_data,
    camera_pose,
    camera_model,
    minimap_to_rgb,
    camera_pose_init=None,
    curve_subdivide=10,
):
    """
    Build geometry objects for minimap layers for a single frame.

    Args:
        minimap_name_to_minimap_data: dict[name -> list[np.ndarray]]
        camera_pose: np.ndarray (4,4)
        camera_model: CameraModel
        minimap_to_rgb: dict[str, list[np.ndarray]], mapping from minimap name to RGB values
        camera_pose_init: np.ndarray (4,4), reference camera pose for ego space transformation

    Returns:
        list: geometry objects (LineSegment2D/Polygon2D)
    """

    all_geometry_objects = []
    for minimap_name, minimap_data in minimap_name_to_minimap_data.items():
        minimap_type = get_type_from_name(minimap_name)
        # We create LineSegment2D geometry object for each polyline
        if minimap_type == "polyline":
            line_segment_list = []
            for polyline in minimap_data:
                # Filter out minimap that is under the ego vehicle
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    polyline, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                # Subdivide the polyline so that distortion is observed in camera view
                polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.2)
                line_segment = np.stack([polyline_subdivided[0:-1], polyline_subdivided[1:]], axis=1)  # [N, 2, 3]
                line_segment_list.append(line_segment)
            if len(line_segment_list) == 0:
                continue

            all_line_segments = np.concatenate(line_segment_list, axis=0)  # [N', 2, 3]
            xy_and_depth = camera_model.get_xy_and_depth(all_line_segments.reshape(-1, 3), camera_pose).reshape(
                -1, 2, 3
            )  # [N', 3]

            # filter the line segments with both vertices have depth >= 0
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]
            if len(valid_xy_and_depth) == 0:
                continue

            color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
            all_geometry_objects.append(
                LineSegment2D(
                    valid_xy_and_depth,
                    base_color=color_float,
                    line_width=5 if minimap_name == "poles" else 12,
                )
            )

        elif minimap_type == "polygon" or minimap_type == "cuboid3d":
            # merge all vertices from polygons and record indices
            polygon_vertices = []
            polygon_vertex_counts = []
            for polygon in minimap_data:
                # Filter out minimap that is under the ego vehicle
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    polygon, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                if minimap_type == "polygon":  # crosswalks and road_markings
                    # Subdivide the polygon so that distortion is observed in camera view
                    polygon_interval = max(0.8, 10.8 - curve_subdivide * 1.0)  # 9.8m at 1, 0.8m at 10
                    polygon_subdivided = interpolate_polyline_to_points(polygon, segment_interval=polygon_interval)
                    # Use triangulation to handle concave polygons in camera view
                    triangles_3d = triangulate_polygon_3d(polygon_subdivided)
                    polygon_subdivided = triangles_3d.reshape(-1, 3)
                    if len(polygon_subdivided) == 0:
                        continue
                else:  # cuboid3d: traffic_signs
                    polygon_subdivided = interpolate_polyline_to_points(polygon, segment_interval=0.2)
                polygon_vertices.append(polygon_subdivided)
                polygon_vertex_counts.append(len(polygon_subdivided))
            if len(polygon_vertices) == 0:
                continue
            # get xy and depth for all vertices at once
            all_vertices = np.concatenate(polygon_vertices, axis=0)
            all_xy_and_depth = camera_model.get_xy_and_depth(all_vertices, camera_pose)

            # recover individual polygons using recorded counts, and keep access to original 3D subdivided vertices
            start_idx = 0
            for vertex_count in polygon_vertex_counts:
                polygon_xy_and_depth = all_xy_and_depth[start_idx : start_idx + vertex_count]
                start_idx += vertex_count
                color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0

                if minimap_type == "polygon":  # crosswalks and road_markings (both triangulated)
                    triangles_proj = polygon_xy_and_depth.reshape(-1, 3, 3)
                    # Filter out triangles that are completely behind camera
                    invalid_triangles_indices = np.all(triangles_proj[:, :, 2] < 0, axis=1)
                    valid_triangles_indices = ~invalid_triangles_indices
                    if valid_triangles_indices.sum() > 0:
                        all_geometry_objects.append(
                            TriangleList2D(triangles_proj[valid_triangles_indices], base_color=color_float)
                        )
                else:  # cuboid3d: traffic_signs (subdivided polyline)
                    if np.all(polygon_xy_and_depth[:, 2] < 0):
                        continue
                    all_geometry_objects.append(Polygon2D(polygon_xy_and_depth, base_color=color_float))
        else:
            raise ValueError(f"Invalid minimap type: {minimap_type}")

    return all_geometry_objects


def prepare_traffic_light_status_data(
    traffic_light_position_wds_file,
    traffic_light_per_frame_status_wds_file,
    traffic_light_color_version="v2",
):
    """
    Preload traffic light position and per-frame status data.

    Args:
        traffic_light_position_wds_file: str, path to the webdataset file containing traffic light position data
        traffic_light_per_frame_status_wds_file: str, path to the webdataset file containing traffic light per-frame status data
        traffic_light_color_version: str, version of traffic light colors

    Returns:
        tuple: (position_list, status_dict, tl_status_to_rgb, legacy_state_map).
            status_dict is None if the status tar is unavailable. legacy_state_map
            is the source's state_to_legacy_render_state table (maps each rich
            state to red/yellow/green/unknown) when present — e.g. the LightNet
            source ships it — or None for the legacy red/yellow/green/unknown
            source whose states need no collapsing.
    """
    traffic_light_position_sample = get_sample(traffic_light_position_wds_file)
    tl_status_to_rgb = load_traffic_light_colors(traffic_light_color_version)
    if not os.path.exists(traffic_light_per_frame_status_wds_file):
        return traffic_light_position_sample["traffic_lights.json"]["labels"], None, tl_status_to_rgb, None
    traffic_light_per_frame_status_sample = get_sample(traffic_light_per_frame_status_wds_file)
    aggregated_states = traffic_light_per_frame_status_sample["aggregated_states.json"]
    return (
        traffic_light_position_sample["traffic_lights.json"]["labels"],
        aggregated_states["traffic_light_states"],
        tl_status_to_rgb,
        aggregated_states.get("state_to_legacy_render_state"),
    )


def create_traffic_light_status_geometry_objects_from_data(
    traffic_light_position_list,
    traffic_light_per_frame_status_dict,
    frame_id,
    camera_pose,
    camera_model,
    tl_status_to_rgb,
):
    """
    Build geometry objects (Polygon2D) for traffic lights for a single frame.

    Args:
        traffic_light_position_list: list[dict], traffic light position data
        traffic_light_per_frame_status_dict: dict[str, dict], traffic light per-frame status data
        frame_id: int, frame id
        camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, ftheta or pinhole camera model
        tl_status_to_rgb: dict[str, list[np.ndarray]], mapping from traffic light status to RGB values

    Returns:
        list[Polygon2D]: geometry objects for traffic lights
    """
    if traffic_light_position_list is None:
        return []
    # in RDS-HQ format, if there is only one element, it is actually a label for no traffic light.
    if len(traffic_light_position_list) == 1:
        return []

    polygon_vertices = []
    polygon_vertex_counts = []
    polygon_colors = []
    for traffic_light_index, traffic_light_data in enumerate(traffic_light_position_list):
        if traffic_light_per_frame_status_dict is None:
            # Use unknown color for traffic light without status
            signal_render_color = np.array(tl_status_to_rgb["unknown"]) / 255.0
        else:
            this_frame_status = traffic_light_per_frame_status_dict[str(traffic_light_index)]
            assert {"name": "feature_id", "text": this_frame_status["feature_id"]} in traffic_light_data["labelData"][
                "shape3d"
            ]["attributes"]
            state_list = this_frame_status["state"]
            signal_state = state_list[frame_id] if frame_id < len(state_list) else "unknown"
            signal_render_color = np.array(tl_status_to_rgb[signal_state]) / 255.0
        polyline = interpolate_polyline_to_points(
            cuboid3d_to_polyline(traffic_light_data["labelData"]["shape3d"]["cuboid3d"]["vertices"]),
            segment_interval=0.2,
        )
        polygon_vertices.append(polyline)
        polygon_vertex_counts.append(len(polyline))
        polygon_colors.append(signal_render_color)

    # projecting all points together to save time
    all_polygon_vertices = np.concatenate(polygon_vertices, axis=0)
    all_xy_and_depth = camera_model.get_xy_and_depth(all_polygon_vertices, camera_pose)

    geometry_objects = []
    start_idx = 0
    for tl_index, vertex_count in enumerate(polygon_vertex_counts):
        xy_and_depth = all_xy_and_depth[start_idx : start_idx + vertex_count]
        start_idx += vertex_count
        if np.all(xy_and_depth[:, 2] < 0):
            continue
        geometry_objects.append(Polygon2D(xy_and_depth, base_color=polygon_colors[tl_index]))

    return geometry_objects


# ---------------------------------------------------------------------------
# WMv4 helpers: front-vs-back face coloring for traffic lights and signs.
# ---------------------------------------------------------------------------

# Cuboid vertex quads for the "front" (artwork) and "back" faces, in perimeter
# order so they form a valid 2D polygon when projected. These are inverted
# relative to the BoundingBox2D doc diagram in utils/graphics_utils.py — the
# upstream RDS-HQ label pipeline puts the artwork face on vertices {2,3,7,6}
# (verified visually: the closer face of a head-on-approached Stop sign sits
# on these vertices).
_WMV4_FRONT_FACE_QUAD = (2, 3, 7, 6)
_WMV4_BACK_FACE_QUAD = (0, 1, 5, 4)

# The five non-front cuboid faces (back + the four sides), as perimeter-ordered
# vertex-index quads. Rendering the traffic light as a solid box means every face
# but the front takes the back color, so any non-front view reads as "back of
# light". (Derived from the cuboid topology — connectors i↔i+4 — same faces as
# graphics_utils.ALL_FACE_INDICES minus the front.)
_WMV4_TL_NONFRONT_FACE_QUADS = (
    _WMV4_BACK_FACE_QUAD,  # (0, 1, 5, 4) — back
    (0, 1, 2, 3),  # end cap
    (4, 5, 6, 7),  # end cap
    (1, 2, 6, 5),  # side
    (0, 3, 7, 4),  # side
)
# The 12 cuboid edges (vertex-index pairs), drawn as explicit lines so the
# uniform face colors stay crisp and the box reads with hard edges like the
# vehicle bounding boxes. Matches graphics_utils.EDGE_INDICES.
_WMV4_TL_CUBOID_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
# Vectorized index arrays (built once) used by the batched TL geometry builder:
# the six cuboid faces (front first, then the five back/side faces) and the 12
# edges, as vertex-index gathers into a projected (N, 8, 3) cuboid stack.
_WMV4_TL_FACE_QUADS = (_WMV4_FRONT_FACE_QUAD, *_WMV4_TL_NONFRONT_FACE_QUADS)  # 6 quads, front at index 0
_WMV4_TL_FACE_IDX = np.asarray(_WMV4_TL_FACE_QUADS)  # (6, 4)
_WMV4_TL_EDGE_IDX = np.asarray(_WMV4_TL_CUBOID_EDGES)  # (12, 2)


def _cuboids_below_level(cuboids, level_plane):
    """Return a mask for elevated-object cuboids wholly below the ego deck."""

    cuboids = np.asarray(cuboids, dtype=float)
    if level_plane is None or len(cuboids) == 0:
        return np.zeros(len(cuboids), dtype=bool)
    below = points_below_ground_plane(
        cuboids.reshape(-1, 3),
        level_plane,
        base_margin=0.3,
        margin_slope=np.tan(np.deg2rad(1.0)),
        max_margin=1.0,
        far_margin=2.0,
    )
    return below.reshape(-1, 8).all(axis=1)


# Pull edge lines slightly toward the camera (meters) so they win the depth test
# against their own coplanar faces. Kept smaller than a housing's front-to-back
# thickness so edges hidden behind the box stay occluded (depth is linear in
# meters: z_ndc = depth/depth_max * 2 - 1).
_WMV4_TL_EDGE_DEPTH_BIAS_M = 0.02

# One-time debug flag: when set, the v4 geometry helpers print the world-space
# centroids of the front and back face of the first cuboid they process, to
# help confirm the face convention against the actual data.
_WMV4_DEBUG_FACE_PRINTED = False


def _wmv4_maybe_debug_print_face(cuboid_vertices, label):
    """When WMV4_DEBUG_FACE=1 is set, print front/back face centroids once."""
    global _WMV4_DEBUG_FACE_PRINTED
    if _WMV4_DEBUG_FACE_PRINTED or os.environ.get("WMV4_DEBUG_FACE") != "1":
        return
    verts = np.asarray(cuboid_vertices)
    front_centroid = verts[list(_WMV4_FRONT_FACE_QUAD)].mean(axis=0)
    back_centroid = verts[list(_WMV4_BACK_FACE_QUAD)].mean(axis=0)
    cprint(
        f"[WMV4_DEBUG_FACE] {label}: front_centroid={front_centroid.tolist()} back_centroid={back_centroid.tolist()}",
        "cyan",
    )
    _WMV4_DEBUG_FACE_PRINTED = True


def _extract_sign_class(sign_label):
    """Pull the sign_class attribute string from a sign label, or None if absent.

    Accepts either the `.text` shape or the `.enum` / `.enumsList` shapes seen in
    other RDS-HQ attributes.
    """
    try:
        attributes = sign_label["labelData"]["shape3d"]["attributes"]
    except (KeyError, TypeError):
        return None
    for attribute in attributes:
        if attribute.get("name") != "sign_class":
            continue
        if isinstance(attribute.get("text"), str):
            return attribute["text"]
        if isinstance(attribute.get("enum"), str):
            return attribute["enum"]
        enums_list = attribute.get("enumsList")
        if isinstance(enums_list, list) and enums_list:
            return enums_list[0]
        if isinstance(enums_list, dict):
            inner = enums_list.get("enumsList")
            if isinstance(inner, list) and inner:
                return inner[0]
    return None


def _normalize_sign_class(sign_class_raw):
    """Normalize a raw sign_class string for exact class allowlist matching."""
    return sign_class_raw.strip().upper() if isinstance(sign_class_raw, str) else None


# Known physical traffic signs that should expose a distinct back face. This is
# renderer taxonomy, not an experiment setting; unlisted classes remain
# double-sided so incomplete taxonomy coverage does not hide their front face.
_WMV4_SINGLE_SIDED_SIGN_CLASSES = frozenset(
    {
        "TRAFFIC_SIGN_REGULATORY_R1_STOP",
        "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT",
        "TRAFFIC_SIGN_REGULATORY_R3_BIKE_LANE",
        "TRAFFIC_SIGN_WARNING_W11_PEDESTRAIN_CROSSING",
        "TRAFFIC_SIGN_REGULATORY_R3_NO_U_TURN",
        "TRAFFIC_SIGN_SCHOOL_S1_SCHOOL",
        "TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_ARROW_RIGHT",
        "TRAFFIC_SIGN_REGULATORY_R4_KEEP_RIGHT",
        "TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_ARROW_LEFT",
        "TRAFFIC_SIGN_REGULATORY_R1_YIELD_TO_PEDESTRIAN",
        "TRAFFIC_SIGN_REGULATORY_R3_NO_LEFT_TURN",
        "TRAFFIC_SIGN_REGULATORY_R3_HOV_LANE",
        "TRAFFIC_SIGN_REGULATORY_R5_DO_NOT_ENTER",
        "TRAFFIC_SIGN_REGULATORY_R3_RIGHT_LANE_MUST_TURN_RIGHT",
        "TRAFFIC_SIGN_WARNING_W13_ADVISORY_SPEED",
        "TRAFFIC_SIGN_REGULATORY_R3_NO_LEFT_U_TURN",
        "TRAFFIC_SIGN_REGULATORY_R1_YIELD",
        "TRAFFIC_SIGN_GERMAN_BASICS_MAXIMUM_SPEED",
        "TRAFFIC_SIGN_REGULATORY_R3_NO_RIGHT_TURN",
        "TRAFFIC_SIGN_DESTINATION_DISTANCE_D11_BIKE_ROUTE",
        "TRAFFIC_SIGN_WARNING_W4_MERGE",
        "TRAFFIC_SIGN_REGULATORY_R3_LEFT_U_TURN_ONLY",
        "TRAFFIC_SIGN_REGULATORY_R3_LEFT_ONLY",
        "TRAFFIC_SIGN_REGULATORY_R3_RIGHT_ONLY",
        "TRAFFIC_SIGN_WARNING_W3_SIGNAL_AHEAD",
        "TRAFFIC_SIGN_GERMAN_TRAFFIC_PRIORITY_YIELD_TO_CROSS_TRAFFIC",
        "TRAFFIC_SIGN_REGULATORY_R10_STOP_HERE_ON_RED_NON_CURVE_ARROW",
        "TRAFFIC_SIGN_REGULATORY_R3_DOUBLE_LANE_LEFT_TURN_OR_U_TURN",
        "TRAFFIC_SIGN_WARNING_W3_STOP_AHEAD",
        "TRAFFIC_SIGN_REGULATORY_R8_NO_STOPPING_ANYTIME",
        "TRAFFIC_SIGN_REGULATORY_R7_NO_PARKING_ANYTIME",
        "TRAFFIC_SIGN_REGULATORY_R7_NO_PARKING_TIME_LIMIT",
        "TRAFFIC_SIGN_REGULATORY_R10_NO_TURN_ON_RED",
        "TRAFFIC_SIGN_REGULATORY_R7_TIMED_PARKING_TIME_LIMIT",
        "TRAFFIC_SIGN_WARNING_W13_ADVISORY_SPEED_EXIT",
        "TRAFFIC_SIGN_GERMAN_TRAFFIC_PRIORITY_PRIORITY_ROAD",
        "TRAFFIC_SIGN_REGULATORY_R5_NO_TRUCKS",
        "TRAFFIC_SIGN_SCHOOL_S5_SCHOOL_SPEED_LIMIT",
        "TRAFFIC_SIGN_GERMAN_BASICS_NO_ENTER",
        "TRAFFIC_SIGN_GERMAN_DRIVING_LANE_CONTROLS_KEEP_RIGHT",
        "TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_RECTANGULAR_LEFT",
        "TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_RECTANGULAR_RIGHT",
        "TRAFFIC_SIGN_WARNING_W4_LANE_ENDS",
        "TRAFFIC_SIGN_REGULATORY_R1_TURNING_VEHICLES_YIELD_TO_PEDESTRIANS",
        "TRAFFIC_SIGN_WARNING_W11_BIKE_CROSSING",
        "TRAFFIC_SIGN_GERMAN_DRIVING_LANE_CONTROLS_GO_STRAIGHT_AHEAD",
        "TRAFFIC_SIGN_REGULATORY_R3_LEFT_LANE_MUST_TURN_LEFT",
        "TRAFFIC_SIGN_REGULATORY_R8_NO_PARKING",
        "TRAFFIC_SIGN_REGULATORY_R3_DOUBLE_LANE_LEFT_TURN_NO_U_TURN",
        "TRAFFIC_SIGN_REGULATORY_R10_LEFT_TURN_YIELD_ON_GREEN",
    }
)


def _classify_sign(sign_class_raw):
    """Map a raw sign_class string to one of the Tier-1 category keys or "other"."""
    s = _normalize_sign_class(sign_class_raw)
    if s is None:
        return "other"
    if s == "TRAFFIC_SIGN_REGULATORY_R1_STOP":
        return "stop"
    if s == "TRAFFIC_SIGN_REGULATORY_R1_YIELD":
        return "yield"
    if s in (
        "TRAFFIC_SIGN_REGULATORY_R5_DO_NOT_ENTER",
        "TRAFFIC_SIGN_GERMAN_BASICS_NO_ENTER",
    ):
        return "do_not_enter"
    if s.startswith("TRAFFIC_SIGN_REGULATORY_R6_ONEWAY"):
        return "one_way"
    # Upstream label is misspelled "PEDESTRAIN"; accept the correct spelling too.
    if s in (
        "TRAFFIC_SIGN_WARNING_W11_PEDESTRAIN_CROSSING",
        "TRAFFIC_SIGN_WARNING_W11_PEDESTRIAN_CROSSING",
    ):
        return "pedestrian_crossing"
    return "other"


def _extract_oneway_direction(sign_class_raw):
    """Return "left"/"right" for a one-way sign_class, or None if not directional.

    RDS-HQ encodes the arrow direction as the suffix of the one-way classes, e.g.
    TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_ARROW_LEFT and
    TRAFFIC_SIGN_REGULATORY_R6_ONEWAY_RECTANGULAR_RIGHT. Direction is as seen
    facing the sign (standard sign nomenclature). Returns None for any one-way
    class without a recognized suffix (rendered with the generic one-way color).
    """
    if not isinstance(sign_class_raw, str):
        return None
    s = sign_class_raw.upper()
    if not s.startswith("TRAFFIC_SIGN_REGULATORY_R6_ONEWAY"):
        return None
    if s.endswith("_LEFT"):
        return "left"
    if s.endswith("_RIGHT"):
        return "right"
    return None


# WMv4 directional road-marking (painted lane arrow) categories. The RDS-HQ
# 3d_road_markings layer tags every polygon with a `road_marking_class`; these
# seven direction-paint classes are the WMv4 arrow targets. Other combined
# classes (LEFT_RIGHT, etc.) and non-arrow markings (stop bar, bike, text,
# speed-limit, etc.) remain unclassified and use the generic road-markings
# color. Keys are shared by the color config and arrow-glyph templates.
_WMV4_ROAD_MARKING_ARROW_CLASSES = {
    "ROI_POLYGON_DIRECTION_PAINT_FORWARD": "forward",
    "ROI_POLYGON_DIRECTION_PAINT_LEFT": "left",
    "ROI_POLYGON_DIRECTION_PAINT_RIGHT": "right",
    "ROI_POLYGON_DIRECTION_PAINT_LEFT_FORWARD": "left_forward",
    "ROI_POLYGON_DIRECTION_PAINT_RIGHT_FORWARD": "right_forward",
    "ROI_POLYGON_DIRECTION_PAINT_MERGE_LEFT": "merge_left",
    "ROI_POLYGON_DIRECTION_PAINT_MERGE_RIGHT": "merge_right",
}

# ROADNET_MASK is an export/support polygon rather than visible paint and must
# never be emitted as WMv4 road-marking geometry.
_WMV4_ROAD_MARKING_EXCLUDED_CLASSES = frozenset({"ROI_POLYGON_ROADNET_MASK"})

# Interior refinement is intentionally limited to the two overlapping polygon
# layers involved in the KEEP_CLEAR/TEXT_PAINT depth-ordering issue.
_WMV4_ROAD_MARKING_INTERIOR_TESSELLATION_CLASSES = frozenset({"ROI_POLYGON_KEEP_CLEAR", "ROI_POLYGON_TEXT_PAINT"})

# Push the KEEP_CLEAR footprint 10 cm farther in camera depth so overlapping
# visible paint, including TEXT_PAINT and lane lines, remains on top. This only
# changes depth ordering; it does not move or resize the projected footprint.
_WMV4_KEEP_CLEAR_DEPTH_BIAS_M = 0.10


def _extract_road_marking_class(road_marking_label):
    """Pull the road_marking_class attribute string from a road-marking label.

    Accepts the `.text` / `.enum` / `.enumsList` attribute shapes seen across
    heterogeneous RDS-HQ exports, mirroring _extract_sign_class. Returns None if
    the attribute is absent.
    """
    try:
        attributes = road_marking_label["labelData"]["shape3d"]["attributes"]
    except (KeyError, TypeError):
        return None
    for attribute in attributes:
        if attribute.get("name") != "road_marking_class":
            continue
        if isinstance(attribute.get("text"), str):
            return attribute["text"]
        if isinstance(attribute.get("enum"), str):
            return attribute["enum"]
        enums_list = attribute.get("enumsList")
        if isinstance(enums_list, list) and enums_list:
            return enums_list[0]
        if isinstance(enums_list, dict):
            inner = enums_list.get("enumsList")
            if isinstance(inner, list) and inner:
                return inner[0]
    return None


def _classify_road_marking_arrow(road_marking_class_raw):
    """Map a raw road_marking_class string to a direction name, or None.

    Returns one of "forward"/"left"/"right"/"left_forward"/"right_forward"/
    "merge_left"/"merge_right".
    None means the marking is not one of the WMv4 turn-arrow categories (it may
    be another combined-direction arrow, a stop/yield bar, text paint, etc.) and
    should render with the generic road_markings color.
    """
    if not isinstance(road_marking_class_raw, str):
        return None
    return _WMV4_ROAD_MARKING_ARROW_CLASSES.get(road_marking_class_raw.upper())


# WMv4 text road-markings: painted regulatory words use their exported
# footprint and a per-category color. Map the raw road_marking_class to a
# category key in config_color_road_marking.json.
_WMV4_ROAD_MARKING_TEXT_CLASSES = {
    "ROI_POLYGON_ROAD_MARKING_STOP": "stop",
    "ROI_POLYGON_KEEP_CLEAR": "keep_clear",
}


def _classify_road_marking_text(road_marking_class_raw):
    """Map a raw road_marking_class string to a text category ("stop"/"keep_clear"), or None.

    None means the marking is not one of the WMv4 painted-word categories and
    should fall through to the arrow or generic road_markings handling.
    """
    if not isinstance(road_marking_class_raw, str):
        return None
    return _WMV4_ROAD_MARKING_TEXT_CLASSES.get(road_marking_class_raw.upper())


# --- WMv4 road-marking arrow glyphs (Option 2: draw an oriented polygon arrow) ---

# Painted lane arrows carry only a footprint quad + a direction class in RDS-HQ
# (no glyph geometry). Option 2 draws a recognizable arrow in place of that quad.
# The asset defines canonical straight, branch, and L-turn outlines. Each completed
# direction template is normalized, then fitted to the individual marking's
# exported width and length so same-class markings retain their per-clip size.
_WMV4_ARROW_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "road_marking_arrows" / "arrow_shape.json"

# Calibrates which world lateral direction the template's +x axis maps to. The
# arrow's forward sense is fixed geometrically (vertex order, below); only the
# left/right handedness can flip with the ground-normal sign convention. If a
# rendered LEFT arrow points right (verified against RGB), flip this to -1.0 and
# both lateral directions correct together — the road-marking analogue of
# _WMV4_ONEWAY_BASE_TEXTURE_DIRECTION.
_WMV4_ROAD_MARKING_LATERAL_SIGN = 1.0

# The arrow's travel direction is encoded in the footprint quad's full vertex
# ordering. The midpoint axis from edge (q0, q1) to edge (q2, q3) points opposite
# travel, so we negate it. Do not select the longer footprint dimension: pure
# turn-arrow footprints are often nearly square and can be wider than they are
# long, which would rotate the glyph by 90 degrees. If a future RDS-HQ export
# reverses this edge-pair convention, flip this sign to +1.0.
_WMV4_ROAD_MARKING_VERTEX_ORDER_SIGN = -1.0


def _wmv4_transform_arrow_part(base_outline, part):
    """Place a transformed copy of base_outline for one glyph part.

    Transform: scale * R(rotate_deg) * (p - pivot) + pivot + translate, with R a
    counter-clockwise rotation in the template (x_right, y_forward) plane. pivot
    lets a branch rotate about the shaft tail rather than the glyph center. scale
    is a scalar (uniform) or a [sx, sy] pair — use [-1, 1] to mirror a right-turn
    glyph into its left-turn form (applied after the rotation).
    """
    p = np.asarray(base_outline, dtype=float)
    pivot = np.asarray(part.get("pivot", [0.0, 0.0]), dtype=float)
    translate = np.asarray(part.get("translate", [0.0, 0.0]), dtype=float)
    scale = part.get("scale", 1.0)
    scale = np.asarray(scale, dtype=float) if isinstance(scale, (list, tuple)) else np.array([scale, scale])
    theta = np.radians(float(part.get("rotate_deg", 0.0)))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return scale[None, :] * ((p - pivot) @ rot.T) + pivot + translate


def _load_road_marking_arrow_templates():
    """Load the canonical arrow shapes and build per-direction triangulated templates.

    Each direction is a list of parts; each part is a transformed copy of a named
    base shape (`straight`, `turn`, or `turn_L`, in canonical asset units).
    Parts are triangulated separately (handling the concave arrow notch) and their triangles concatenated,
    so a fork (straight shaft + angled branch) renders as one filled glyph. Each
    completed direction is centered and normalized independently to span one unit
    in x and y. Returns (templates, size_scale), where x = right and y = forward.
    """
    spec = json.load(open(_WMV4_ARROW_ASSET_PATH, encoding="utf-8"))
    shapes = {
        "straight": np.asarray(spec["straight_arrow"], dtype=float),  # (N, 2), pointing +y
        "turn": np.asarray(spec["turn_arrow"], dtype=float),
        "turn_L": np.asarray(spec["turn_L"], dtype=float),
    }

    templates = {}
    for direction, parts in spec["directions"].items():
        part_tris = []
        for part in parts:
            base = shapes[part.get("shape", "straight")]
            poly = _wmv4_transform_arrow_part(base, part)
            tris_3d = triangulate_polygon_3d(np.column_stack([poly, np.zeros(len(poly))]))
            part_tris.append(tris_3d[:, :, :2])
        tris = np.concatenate(part_tris, axis=0)
        flat = tris.reshape(-1, 2)
        lower = flat.min(axis=0)
        upper = flat.max(axis=0)
        span = upper - lower
        if np.any(span < 1e-6):
            raise ValueError(f"Degenerate road-marking arrow template for {direction!r}")
        templates[direction] = ((tris - (lower + upper) / 2.0) / span).astype(float)
    return templates, float(spec.get("size_scale", 1.0))


_WMV4_ROAD_MARKING_ARROW_TEMPLATES, _WMV4_ROAD_MARKING_ARROW_SIZE_SCALE = _load_road_marking_arrow_templates()


def wmv4_road_marking_arrow_forward(quad_world_4x3):
    """Return the unit travel-forward direction of an arrow marking, in 3D world coords.

    Forward is recovered from the footprint quad's vertex ordering. The axis from
    the midpoint of edge (q0, q1) to the midpoint of edge (q2, q3) points opposite
    lane travel, so it is negated (see _WMV4_ROAD_MARKING_VERTEX_ORDER_SIGN).
    Footprint aspect ratio is intentionally ignored. This is the single source of
    truth for arrow and text orientation. Returns None for a degenerate
    (zero-length) ordered axis.

    Args:
        quad_world_4x3: np.ndarray (4, 3), the marking footprint in world coords.
    """
    quad = np.asarray(quad_world_4x3, dtype=float)
    if quad.shape != (4, 3):
        return None
    ordered_axis = ((quad[2] + quad[3]) - (quad[0] + quad[1])) / 2.0
    n = np.linalg.norm(ordered_axis)
    if n < 1e-6:
        return None
    return _WMV4_ROAD_MARKING_VERTEX_ORDER_SIGN * ordered_axis / n


def _wmv4_road_marking_arrow_world_triangles(quad_world_4x3, direction, size_scale=None):
    """Build the arrow-glyph triangles, in world space, for one arrow marking.

    The marking's footprint supplies placement, orientation, width, and length.
    The normalized template is fitted independently along the footprint's lateral
    and forward axes, preserving per-instance dimensions even for the same class.
    `size_scale` is a final uniform tuning multiplier. Returns (M, 3, 3) world
    triangles, or None for a degenerate quad or an unknown direction.

    Args:
        quad_world_4x3: np.ndarray (4, 3), the marking footprint in world coords.
        direction: str, one of the canonical arrow direction names.
        size_scale: float or None, multiplier after fitting to the footprint.
            None uses the asset's size_scale; 1.0 fills the footprint.
    """
    tris_template = _WMV4_ROAD_MARKING_ARROW_TEMPLATES.get(direction)
    if tris_template is None:
        return None
    quad = np.asarray(quad_world_4x3, dtype=float)
    if quad.shape != (4, 3):
        return None

    centroid = quad.mean(axis=0)
    # Vectors between midpoints of opposite edges give the ground normal.
    axis_a = ((quad[1] + quad[2]) - (quad[3] + quad[0])) / 2.0
    axis_b = ((quad[2] + quad[3]) - (quad[0] + quad[1])) / 2.0

    forward = wmv4_road_marking_arrow_forward(quad)
    if forward is None:
        return None
    normal = np.cross(axis_a, axis_b)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-9:
        return None
    normal = normal / n_norm
    if normal[2] < 0:  # force the ground normal to point up (world z up)
        normal = -normal
    lateral = _WMV4_ROAD_MARKING_LATERAL_SIGN * np.cross(forward, normal)
    lateral = lateral / (np.linalg.norm(lateral) + 1e-9)

    rel = quad - centroid
    proj_fwd = rel @ forward
    proj_lat = rel @ lateral
    fwd_extent = float(proj_fwd.max() - proj_fwd.min())
    lat_extent = float(proj_lat.max() - proj_lat.min())
    if fwd_extent < 1e-6 or lat_extent < 1e-6:
        return None

    effective_scale = _WMV4_ROAD_MARKING_ARROW_SIZE_SCALE if size_scale is None else size_scale
    if effective_scale <= 0:
        raise ValueError(f"road-marking arrow size_scale must be positive, got {effective_scale}")
    flat = tris_template.reshape(-1, 2)  # normalized (x_lateral, y_forward)
    world = (
        centroid[None, :]
        + effective_scale * lat_extent * flat[:, 0:1] * lateral[None, :]
        + effective_scale * fwd_extent * flat[:, 1:2] * forward[None, :]
    )
    return world.reshape(-1, 3, 3)


def prepare_road_marking_data_v4(
    road_marking_wds_file: str | Path,
    hdmap_color_version: str = "v3",
    road_marking_color_version: str = "v4",
    use_arrow_colors: bool = True,
    use_arrow_shapes: bool = False,
    arrow_size_scale: float | None = None,
    use_text_colors: bool = True,
    curve_subdivide: int = 10,
    use_interior_tessellation: bool = False,
    interior_tessellation_max_edge_m: float = 1.0,
) -> list[dict]:
    """Preload road-marking labels and precompute static world-space fill geometry.

    Each marking is reduced to a set of world-space triangles plus a color, both
    static across the clip, so the per-frame call only has to project them:

    - Arrow markings (left/right/forward/left_forward/right_forward and merges):
        * color = the configured category color (config_color_road_marking.json) when
          use_arrow_colors, else the generic road_markings color;
        * geometry = an oriented arrow glyph when use_arrow_shapes, else the
          marking's own (triangulated) footprint.
    - Text markings (stop/keep_clear):
        * color = the per-category color (config_color_road_marking.json) when
          use_text_colors, else the generic road_markings color;
        * geometry = the marking's own (triangulated) footprint.
    - ROADNET_MASK: omitted because it is an export/support polygon, not paint.
    - All other visible markings: generic road_markings color and footprint
      geometry, preserving their v3 appearance.

    Footprint geometry is edge-subdivided before triangulation so long boundary
    edges follow the ftheta projection more closely. Optional interior
    tessellation then conformingly splits long triangle edges, reducing depth
    interpolation error after nonlinear camera projection. Arrow glyph
    geometry is already triangulated and is not refined here.

    Args:
        road_marking_wds_file: str, path to the 3d_road_markings tar for the clip.
        hdmap_color_version: str, version key resolving the generic road_markings color.
        road_marking_color_version: str, version key in config_color_road_marking.json.
        use_arrow_colors: bool, color recognized arrow markings from the road-marking palette.
        use_arrow_shapes: bool, draw an arrow glyph instead of the footprint (Option 2).
        arrow_size_scale: float or None, overrides the asset's glyph size_scale
            (only affects glyphs). None uses the asset value.
        use_text_colors: bool, color text markings (stop/keep_clear) per category (Option 1).
        curve_subdivide: int, 1-10 subdivision quality for footprint polygons.
        use_interior_tessellation: bool, refine KEEP_CLEAR and TEXT_PAINT
            footprint-fill triangle interiors.
        interior_tessellation_max_edge_m: positive world-space edge limit.

    Returns:
        list[dict]: each contains world triangles, color, geometry kind, raw
            class, and camera-depth bias. Empty if the layer is absent or holds
            only the "no marking" sentinel.
    """
    if use_interior_tessellation and (
        not np.isfinite(interior_tessellation_max_edge_m) or interior_tessellation_max_edge_m <= 0
    ):
        raise ValueError("interior_tessellation_max_edge_m must be positive and finite")

    rm_color_map = load_road_marking_colors(road_marking_color_version)
    other_rgb = np.array(load_hdmap_colors(hdmap_color_version)["road_markings"]) / 255.0
    polygon_interval = max(0.8, 10.8 - curve_subdivide * 1.0)
    if not os.path.exists(road_marking_wds_file):
        return []
    labels = get_sample(road_marking_wds_file).get("road_markings.json", {}).get("labels", [])
    if len(labels) == 1 and labels[0].get("labelFamily") == "EMPTY":
        return []

    prepared = []
    for label_index, label in enumerate(labels):
        try:
            verts = np.asarray(label["labelData"]["shape3d"]["surface"]["vertices"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if verts.ndim != 2 or verts.shape[0] < 3:
            continue

        raw_class = _extract_road_marking_class(label)
        normalized_class = raw_class.strip().upper() if isinstance(raw_class, str) else None
        if normalized_class in _WMV4_ROAD_MARKING_EXCLUDED_CLASSES:
            continue
        direction = _classify_road_marking_arrow(normalized_class)
        text_cat = _classify_road_marking_text(normalized_class)

        if direction is not None and use_arrow_colors:
            color = np.array(rm_color_map[direction]) / 255.0
        elif text_cat is not None and use_text_colors:
            color = np.array(rm_color_map[text_cat]) / 255.0
        else:
            color = other_rgb

        tris_world = None
        geometry_kind = None
        should_tessellate = False
        if direction is not None and use_arrow_shapes:
            tris_world = _wmv4_road_marking_arrow_world_triangles(verts, direction, size_scale=arrow_size_scale)
            if tris_world is not None:
                geometry_kind = "arrow_glyph"
        if tris_world is None:  # footprint fill (plain marking, or glyph fell back)
            geometry_kind = "footprint"
            closed_verts = verts if np.allclose(verts[0], verts[-1]) else np.concatenate([verts, verts[:1]])
            subdivided = interpolate_polyline_to_points(closed_verts, segment_interval=polygon_interval)
            tris_world = triangulate_polygon_3d(subdivided)
            should_tessellate = (
                use_interior_tessellation and normalized_class in _WMV4_ROAD_MARKING_INTERIOR_TESSELLATION_CLASSES
            )
            if should_tessellate and len(tris_world) > 0:
                tris_world = refine_triangle_mesh_3d(tris_world, interior_tessellation_max_edge_m)
        if tris_world is None or len(tris_world) == 0:
            continue
        camera_depth_bias_m = (
            _WMV4_KEEP_CLEAR_DEPTH_BIAS_M
            if geometry_kind == "footprint" and normalized_class == "ROI_POLYGON_KEEP_CLEAR"
            else 0.0
        )
        prepared.append(
            {
                "tris_world": tris_world,
                "color": color,
                "geometry_kind": geometry_kind,
                "road_marking_class": raw_class,
                "label_index": label_index,
                "interior_tessellated": bool(geometry_kind == "footprint" and should_tessellate),
                "camera_depth_bias_m": camera_depth_bias_m,
            }
        )

    return prepared


def create_road_marking_geometry_objects_from_data_v4(prepared_road_markings, camera_pose, camera_model):
    """Project the precomputed road-marking triangles into one frame.

    Args:
        prepared_road_markings: list[dict] from prepare_road_marking_data_v4.
        camera_pose: np.ndarray (4, 4), camera pose for this frame.
        camera_model: CameraModel (FThetaCamera or PinholeCamera).

    Returns:
        list[TriangleList2D], one per marking visible this frame.
    """
    if not prepared_road_markings:
        return []
    # Vectorize the projection across ALL markings in the frame. get_xy_and_depth
    # is elementwise per point (one world_to_camera inverse + a row-wise ray2pixel),
    # so concatenating every marking's triangle vertices into a single call and
    # splitting the result back per marking is bit-identical to the old per-marking
    # loop — but replaces N ray2pixel/inverse calls with one, killing the Python and
    # per-call numpy overhead that dominated this stage.
    tris_per_marking = [marking["tris_world"].reshape(-1, 3) for marking in prepared_road_markings]
    vert_counts = [t.shape[0] for t in tris_per_marking]
    all_xy_and_depth = camera_model.get_xy_and_depth(np.concatenate(tris_per_marking, axis=0), camera_pose)
    split_points = np.cumsum(vert_counts[:-1])
    per_marking_xyd = np.split(all_xy_and_depth, split_points, axis=0)

    geometry_objects = []
    for marking, marking_xyd in zip(prepared_road_markings, per_marking_xyd):
        xy_and_depth = marking_xyd.reshape(-1, 3, 3)
        # Drop triangles entirely behind the camera; the depth buffer occludes the rest.
        keep = ~np.all(xy_and_depth[:, :, 2] < 0, axis=1)
        xy_and_depth = xy_and_depth[keep]
        if len(xy_and_depth) == 0:
            continue
        camera_depth_bias_m = float(marking.get("camera_depth_bias_m", 0.0))
        if camera_depth_bias_m:
            xy_and_depth[:, :, 2] += camera_depth_bias_m
        geometry_objects.append(TriangleList2D(xy_and_depth, base_color=marking["color"]))
    return geometry_objects


def build_worldspace_road_marking_geometry(prepared_road_markings, camera_model, below_level_filter=False):
    """Batch static road-marking triangles for per-frame GPU projection."""

    if not prepared_road_markings:
        return []
    from cosmos_auto.wm.graphics_utils import WorldSpaceTriangles

    by_depth_bias = {}
    for marking in prepared_road_markings:
        bias = float(marking.get("camera_depth_bias_m", 0.0))
        by_depth_bias.setdefault(bias, []).append(marking)

    result = []
    for bias, markings in sorted(by_depth_bias.items()):
        vertices = np.concatenate([marking["tris_world"].reshape(-1, 3) for marking in markings], axis=0)
        colors = np.concatenate(
            [np.tile(marking["color"], (marking["tris_world"].size // 3, 1)) for marking in markings],
            axis=0,
        )
        result.append(
            WorldSpaceTriangles(
                vertices=vertices,
                colors=colors,
                camera_model=camera_model,
                below_level_filter=below_level_filter,
                camera_depth_bias_m=bias,
            )
        )
    return result


def prepare_traffic_sign_data_v4(
    traffic_sign_wds_file,
    sign_color_version="v4",
    hdmap_color_version="v3",
):
    """Preload traffic sign labels and resolve the sign-class color palette.

    Args:
        traffic_sign_wds_file: str, path to the 3d_traffic_signs tar for the clip.
        sign_color_version: str, version key in config_color_sign.json.
        hdmap_color_version: str, version key in config_color_hdmap.json — used
            to resolve the "other" (non-Tier-1) sign color from the existing
            generic `traffic_signs` color, so non-Tier-1 signs match v3 visuals.

    Returns:
        tuple: (sign_label_list, sign_class_to_rgb, other_sign_rgb).
            sign_label_list is None if the sign tar is absent for this clip.
    """
    sign_class_to_rgb = load_sign_colors(sign_color_version)
    other_sign_rgb = load_hdmap_colors(hdmap_color_version)["traffic_signs"]
    if not os.path.exists(traffic_sign_wds_file):
        return None, sign_class_to_rgb, other_sign_rgb
    sign_sample = get_sample(traffic_sign_wds_file)
    return (
        sign_sample["traffic_signs.json"]["labels"],
        sign_class_to_rgb,
        other_sign_rgb,
    )


# --- WMv4 traffic-light glyphs: turn arrows + VRU (LightNet status source) ---

# Glyphs render as a single colored shape on a neutral front-housing face:
# the colored shape is the salient blob, so its geometry carries the meaning while
# the colored center-of-mass stays where the signal is — the robustness property
# we want for control. Shapes are chosen to be low-frequency and mutually
# distinct: turn arrows = a status-colored triangle filling the whole face;
# VRU = a colored "X" (stopped) or empty ring "O" (walking). All are distinct from
# a vehicle solid (full colored face) and from each other.

# Pull the glyph slightly toward the camera (scale projected depth) so it wins the
# GL_LESS depth test against the coplanar front-housing face beneath.
_WMV4_TL_GLYPH_DEPTH_BIAS = 0.985

# LightNet simple arrow states we draw a directional glyph for, mapped to a
# direction in the face's local frame. VRU states use the X/O glyphs below;
# other states (solid, compound, diagonal, u-turn, caution, …) render as a
# plain colored face. Most use the legacy-state collapse; red-yellow compounds
# retain their orange family.
_WMV4_TL_ARROW_DIRECTIONS = {
    "red_left_arrow": "left",
    "red_yellow_left_arrow": "left",
    "green_left_arrow": "left",
    "yellow_left_arrow": "left",
    "red_right_arrow": "right",
    "red_yellow_right_arrow": "right",
    "green_right_arrow": "right",
    "yellow_right_arrow": "right",
    "red_up_arrow": "up",
    "red_yellow_up_arrow": "up",
    "green_up_arrow": "up",
    "yellow_up_arrow": "up",
}

# Turn arrows are a single filled triangle (one convex piece) in face-local UV
# (u: left→right = 0→1, v: bottom→top = 0→1), matching the front-face UV
# convention. The triangle is convex so Polygon2D's TRIANGLE_FAN fills it
# correctly. Image orientation verified on clip 00fd410c…: a green_right_arrow
# renders pointing image-right and a green_left_arrow points image-left (no
# mirror). If a future label convention flips this, swap the "left"/"right" cases
# in _wmv4_tl_arrow_uv.
#
# The apex sits on the turn-side edge and the base spans the opposite edge,
# so each arrow fills the whole bounding box.
_WMV4_TL_ARROW_SHAPES = ("left", "right", "up")


def _wmv4_tl_arrow_uv(direction: str) -> list[list[tuple[float, float]]]:
    """Return a full-face turn-arrow triangle in face-local UV coordinates."""
    if direction == "right":  # apex on right edge, base on left edge
        return [[(0.0, 0.0), (1.0, 0.5), (0.0, 1.0)]]
    if direction == "left":  # apex on left edge, base on right edge
        return [[(1.0, 0.0), (0.0, 0.5), (1.0, 1.0)]]
    if direction == "up":  # apex on top edge, base on bottom edge
        return [[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]]
    raise ValueError(f"unknown arrow direction: {direction!r}")


def _wmv4_x_bar_uv(corner_a, corner_b, half_width):
    """One diagonal bar of the VRU "X": a convex quad from corner_a to corner_b.

    Endpoints are inset along the diagonal by half_width so the widened bar's
    four corners land exactly on the face edges (and stay within [0,1]), giving a
    bold X whose tips reach toward the box corners.
    """
    a = np.asarray(corner_a, dtype=float)
    b = np.asarray(corner_b, dtype=float)
    d = b - a
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])  # unit perpendicular
    a_in, b_in = a + d * half_width, b - d * half_width
    return [
        tuple(a_in + n * half_width),
        tuple(a_in - n * half_width),
        tuple(b_in - n * half_width),
        tuple(b_in + n * half_width),
    ]


def _wmv4_o_ring_uv(r_out, r_in, n_seg):
    """The VRU "O" (empty ring) as n_seg convex trapezoids between two concentric
    circles centered on the face. The front housing shows through the hole."""
    cx = cy = 0.5
    angs = [2.0 * np.pi * k / n_seg for k in range(n_seg)]
    out = [(cx + r_out * np.cos(a), cy + r_out * np.sin(a)) for a in angs]
    inn = [(cx + r_in * np.cos(a), cy + r_in * np.sin(a)) for a in angs]
    return [[out[k], out[(k + 1) % n_seg], inn[(k + 1) % n_seg], inn[k]] for k in range(n_seg)]


# VRU glyphs as lists of convex pieces in face-local UV (u: left→right, v:
# bottom→top): "x" (stopped) and "o" (walking). Turn-arrow triangles are
# generated separately by _wmv4_tl_arrow_uv for each direction.
_WMV4_TL_GLYPH_UV = {
    "x": [
        _wmv4_x_bar_uv((0.0, 1.0), (1.0, 0.0), 0.15),  # top-left ↔ bottom-right
        _wmv4_x_bar_uv((0.0, 0.0), (1.0, 1.0), 0.15),  # bottom-left ↔ top-right
    ],
    "o": _wmv4_o_ring_uv(r_out=0.46, r_in=0.30, n_seg=24),
}

# VRU (pedestrian) signals: a colored low-frequency glyph on the front housing,
# distinct from vehicle lights and from each other — "x" for stopped ("don't
# walk"), empty ring "o" for walking. Colors follow the WMv4 wishlist: red_vru→
# red X; green/yellow/white_vru → green/yellow/white O. (white_vru relies on the
# "white" entry added to the v4 traffic-light palette.)
_WMV4_TL_VRU_GLYPHS = {
    "red_vru": ("x", "red"),
    "green_vru": ("o", "green"),
    "yellow_vru": ("o", "yellow"),
    "white_vru": ("o", "white"),
}


def _wmv4_tl_render_spec(
    signal_state: str,
    legacy_state_map: dict[str, str] | None,
    tl_status_to_rgb: dict[str, list[int]],
    tl_front_housing_rgb: np.ndarray,
) -> tuple[np.ndarray, str | None, np.ndarray | None]:
    """Resolve (front_face_rgb, glyph_shape|None, glyph_rgb|None) for a TL state.

    VRU states render as a colored X/O glyph on the neutral front housing,
    simple left/right/up arrow states render as a
    status-colored turn triangle on the neutral front housing, and every other
    state paints a plain status-colored face. Red-yellow states use their own
    orange family color instead of collapsing to legacy red.

    `stateless` (observed but showing no active signal — a dark/off light) always
    resolves to its own near-black color, distinct from `unknown`
    (unobserved), which legacy_state_map would otherwise collapse it into.
    """

    def base_face_rgb() -> np.ndarray:
        # LightNet collapses every red-yellow composite to legacy red. Preserve
        # the richer family here: solids and unsupported compound directions get
        # a full orange face, while the simple arrows use this orange as glyph fill.
        if (signal_state == "red_yellow" or signal_state.startswith("red_yellow_")) and (
            "red_yellow" in tl_status_to_rgb
        ):
            color_key = "red_yellow"
        else:
            color_key = legacy_state_map.get(signal_state, "unknown") if legacy_state_map is not None else signal_state
        return np.array(tl_status_to_rgb[color_key]) / 255.0

    if signal_state == "stateless" and "stateless" in tl_status_to_rgb:
        return np.array(tl_status_to_rgb["stateless"]) / 255.0, None, None

    vru = _WMV4_TL_VRU_GLYPHS.get(signal_state)
    if vru is not None:
        shape, color_key = vru
        return tl_front_housing_rgb, shape, np.array(tl_status_to_rgb[color_key]) / 255.0
    signal_rgb = base_face_rgb()
    direction = _WMV4_TL_ARROW_DIRECTIONS.get(signal_state)
    if direction is not None:
        return tl_front_housing_rgb, direction, signal_rgb
    return signal_rgb, None, None


def _wmv4_tl_glyph_polygons(front_quad_4x3: np.ndarray, shape: str, color: np.ndarray) -> list[Polygon2D]:
    """Return colored Polygon2D pieces for a front-face glyph inscribed in a projected face.

    `shape` is one of "left"/"right"/"up" (turn triangle), "x" (VRU stopped), or
    "o" (VRU walking). front_quad_4x3 is the front face in projected (x, y, depth)
    in _WMV4_FRONT_FACE_QUAD vertex order, whose UVs are (1,0),(0,0),(0,1),(1,1) =
    bottom-right, bottom-left, top-left, top-right. `color` fills the glyph; it is
    nudged toward the camera so it draws over the front-housing face beneath it.
    """
    quad = np.asarray(front_quad_4x3, dtype=float)
    p_br, p_bl, p_tl, p_tr = quad[0], quad[1], quad[2], quad[3]

    def uv_to_point(u: float, v: float) -> np.ndarray:
        return (1 - u) * (1 - v) * p_bl + u * (1 - v) * p_br + (1 - u) * v * p_tl + u * v * p_tr

    pieces = _wmv4_tl_arrow_uv(shape) if shape in _WMV4_TL_ARROW_SHAPES else _WMV4_TL_GLYPH_UV[shape]
    polys = []
    for piece_uv in pieces:
        pts = np.array([uv_to_point(u, v) for (u, v) in piece_uv])
        pts[:, 2] *= _WMV4_TL_GLYPH_DEPTH_BIAS  # pull toward camera to win the depth test
        polys.append(Polygon2D(pts, base_color=color))
    return polys


def create_traffic_light_status_geometry_objects_from_data_v4(
    traffic_light_position_list: list[dict] | None,
    traffic_light_per_frame_status_dict: dict | None,
    frame_id: int,
    camera_pose: np.ndarray,
    camera_model: CameraBase,
    tl_status_to_rgb: dict[str, list[int]],
    legacy_state_map: dict[str, str] | None = None,
    level_plane: dict[str, np.ndarray] | None = None,
) -> list[Geometry2D]:
    """v4 variant of create_traffic_light_status_geometry_objects_from_data.

    Each TL renders as a solid cuboid (bounding box): the front (artwork) face
    uses the per-frame status color, while the back and four side faces use the
    dedicated `back` color, so any non-front view reads as the back of the light.
    Each face is a separate uniform-colored Polygon2D (no per-vertex bleed) and
    the 12 cuboid edges are drawn on top in the `edge` color, giving the
    hard-edged look of the vehicle bounding boxes.

    When the status source uses the richer LightNet taxonomy, legacy_state_map
    collapses each state to a red/yellow/green/unknown palette color. Simple
    left/right/up arrows render as full-face status-colored triangles on a
    neutral front housing; VRU states render as colored X/O glyphs. Other
    states use a plain status-colored face, with orange for red-yellow states.
    See _wmv4_tl_render_spec.
    """
    if traffic_light_position_list is None:
        return []
    if len(traffic_light_position_list) == 1 and traffic_light_position_list[0].get("labelFamily") == "EMPTY":
        return []

    tl_back_rgb = np.array(tl_status_to_rgb["back"]) / 255.0
    tl_front_housing_rgb = np.array(tl_status_to_rgb.get("front_housing", tl_status_to_rgb["back"])) / 255.0
    tl_edge_rgb = np.array(tl_status_to_rgb.get("edge", [200, 200, 200])) / 255.0

    cuboid_vertices_list = []
    front_colors = []
    glyph_shapes = []  # one per cuboid; None unless a front-face glyph should be drawn
    glyph_colors = []  # one per cuboid; fill color for the glyph, None when no glyph
    for traffic_light_index, traffic_light_data in enumerate(traffic_light_position_list):
        if not traffic_light_per_frame_status_dict:
            face_rgb = np.array(tl_status_to_rgb["unknown"]) / 255.0
            glyph_shape = glyph_rgb = None
        else:
            this_frame_status = traffic_light_per_frame_status_dict[str(traffic_light_index)]
            assert {"name": "feature_id", "text": this_frame_status["feature_id"]} in traffic_light_data["labelData"][
                "shape3d"
            ]["attributes"]
            state_list = this_frame_status["state"]
            signal_state = state_list[frame_id] if frame_id < len(state_list) else "unknown"
            face_rgb, glyph_shape, glyph_rgb = _wmv4_tl_render_spec(
                signal_state, legacy_state_map, tl_status_to_rgb, tl_front_housing_rgb
            )
        cuboid_vertices = traffic_light_data["labelData"]["shape3d"]["cuboid3d"]["vertices"]
        if traffic_light_index == 0:
            _wmv4_maybe_debug_print_face(cuboid_vertices, "traffic_light[0]")
        cuboid_vertices_list.append(np.asarray(cuboid_vertices))  # (8, 3)
        front_colors.append(face_rgb)
        glyph_shapes.append(glyph_shape)
        glyph_colors.append(glyph_rgb)

    if not cuboid_vertices_list:
        return []

    keep = ~_cuboids_below_level(cuboid_vertices_list, level_plane)
    if not keep.any():
        return []
    cuboid_vertices_list = [vertices for vertices, include in zip(cuboid_vertices_list, keep) if include]
    front_colors = [color for color, include in zip(front_colors, keep) if include]
    glyph_shapes = [shape for shape, include in zip(glyph_shapes, keep) if include]
    glyph_colors = [color for color, include in zip(glyph_colors, keep) if include]

    # Project all 8 vertices of every cuboid together to save time.
    all_vertices = np.concatenate(cuboid_vertices_list, axis=0)  # (N*8, 3)
    all_xy_and_depth = camera_model.get_xy_and_depth(all_vertices, camera_pose).reshape(-1, 8, 3)

    # Vectorize the per-cuboid geometry assembly across the whole frame: gather
    # every face quad and edge segment and compute their behind-camera masks in
    # whole-array numpy ops, instead of the per-face / per-edge np.all + slicing
    # loop that dominated the build cost. The emitted geometry objects and their
    # order are unchanged (front face, five back/side faces, one edge
    # LineSegment2D, then the glyph, per cuboid), so pixels are bit-identical.
    behind = all_xy_and_depth[..., 2] < 0  # (N, 8)
    face_pts = all_xy_and_depth[:, _WMV4_TL_FACE_IDX]  # (N, 6, 4, 3)
    face_behind = behind[:, _WMV4_TL_FACE_IDX].all(axis=2)  # (N, 6) True => fully behind camera
    edge_pts = all_xy_and_depth[:, _WMV4_TL_EDGE_IDX].copy()  # (N, 12, 2, 3)
    edge_pts[..., 2] -= _WMV4_TL_EDGE_DEPTH_BIAS_M  # nudge edges toward camera
    edge_visible = ~behind[:, _WMV4_TL_EDGE_IDX].all(axis=2)  # (N, 12) True => keep this edge

    geometry_objects = []

    # Batch every visible cuboid face into ONE TriangleList2D. Previously each face
    # was its own solid-color Polygon2D (a TRIANGLE_FAN) → hundreds of tiny draw
    # calls/frame, which the GPU-render profile showed dominates the GPU frame on
    # TL-heavy clips (~24 us/draw). Fan-triangulating each quad as (0,1,2),(0,2,3)
    # with the same per-face color and the same polygon shader is pixel-identical,
    # but collapses the per-face draws into a single batched draw. Face order
    # (cuboid-major, face 0..5) is preserved so equal-depth ties resolve identically.
    vis = ~face_behind  # (N, 6)
    if vis.any():
        face_colors = np.broadcast_to(tl_back_rgb, (len(all_xy_and_depth), _WMV4_TL_FACE_IDX.shape[0], 3)).copy()
        face_colors[:, 0, :] = np.asarray(front_colors)  # front face = status color
        quads = face_pts[vis]  # (K, 4, 3)
        quad_cols = face_colors[vis]  # (K, 3)
        tris = np.stack([quads[:, [0, 1, 2], :], quads[:, [0, 2, 3], :]], axis=1)  # (K, 2, 3, 3)
        tri_colors = np.repeat(quad_cols, 2, axis=0)  # (K*2, 3), 2 tris/quad
        geometry_objects.append(TriangleList2D(tris.reshape(-1, 3, 3), tri_colors))

    # Edges then glyph, per cuboid (unchanged relative order). Both carry depth
    # biases toward the camera, so they win over the faces regardless of draw order.
    for i in range(len(all_xy_and_depth)):
        ev_i = edge_visible[i]
        if ev_i.any():
            geometry_objects.append(LineSegment2D(edge_pts[i][ev_i], base_color=tl_edge_rgb, line_width=2.0))
        if glyph_shapes[i] is not None:
            front_quad = all_xy_and_depth[i][list(_WMV4_FRONT_FACE_QUAD)]
            if not np.all(front_quad[:, 2] < 0):
                geometry_objects.extend(_wmv4_tl_glyph_polygons(front_quad, glyph_shapes[i], glyph_colors[i]))
    return geometry_objects


def create_traffic_sign_geometry_objects_from_data_v4(
    sign_label_list: list[dict] | None,
    camera_pose: np.ndarray,
    camera_model: CameraBase,
    sign_class_to_rgb: dict[str, list[int]],
    other_sign_rgb: list[int] | np.ndarray,
    single_sided_categories: list[str] | None = None,
    level_plane: dict[str, np.ndarray] | None = None,
) -> list[Geometry2D]:
    """Build per-frame geometry for traffic signs (WMv4).

    Each sign renders as a solid rectangular face. For single-sided signs,
    choose the nearer of the front and back faces. Front color is the Tier-1
    category color (or other_sign_rgb for non-Tier-1 classes); back color is
    sign_class_to_rgb["back"].

    single_sided_categories is an iterable of Tier-1 category keys (as returned
    by _classify_sign, e.g. "stop", "yield", "pedestrian_crossing") whose signs
    render as single-sided: front artwork + a dedicated back-color face. Every
    other sign renders double-sided — only the front polygon is emitted (no
    separate back face), so it shows from any view angle. Matched
    case-insensitively.

    Raw sign classes in _WMV4_SINGLE_SIDED_SIGN_CLASSES also render with a
    distinct back face. This only affects sidedness; color still follows the
    category mapping above.
    """
    if sign_label_list is None:
        return []
    if len(sign_label_list) == 1 and sign_label_list[0].get("labelFamily") == "EMPTY":
        return []

    single_sided_category_set = {cat.lower() for cat in (single_sided_categories or [])}

    sign_back_rgb = np.array(sign_class_to_rgb["back"]) / 255.0
    other_front_rgb = np.array(other_sign_rgb) / 255.0

    cuboid_vertices_list = []
    front_colors = []
    double_sided = []  # one per cuboid; True → skip back face (sign looks the same from both sides)
    for sign_index, sign_label in enumerate(sign_label_list):
        sign_class_raw = _extract_sign_class(sign_label)
        sign_class_key = _normalize_sign_class(sign_class_raw)
        category = _classify_sign(sign_class_raw)
        # One-way signs carry a left/right arrow direction (from the sign_class
        # suffix). Their rectangular geometry is identical in either direction;
        # direction is encoded in the front-face color.
        direction = _extract_oneway_direction(sign_class_raw) if category == "one_way" else None
        if category == "other":
            front_rgb = other_front_rgb
        elif category == "one_way":
            # Distinct colors per direction (one_way_left / one_way_right); fall
            # back to the generic one_way color for direction-less one-way classes.
            color_key = f"one_way_{direction}" if direction else "one_way"
            front_rgb = np.array(sign_class_to_rgb.get(color_key, sign_class_to_rgb["one_way"])) / 255.0
        else:
            front_rgb = np.array(sign_class_to_rgb[category]) / 255.0
        try:
            cuboid_vertices = sign_label["labelData"]["shape3d"]["cuboid3d"]["vertices"]
        except (KeyError, TypeError):
            continue
        if sign_index == 0:
            _wmv4_maybe_debug_print_face(cuboid_vertices, f"traffic_sign[0]({category})")
        cuboid_vertices_list.append(np.asarray(cuboid_vertices))  # (8, 3)
        front_colors.append(front_rgb)
        is_single_sided = category in single_sided_category_set or sign_class_key in _WMV4_SINGLE_SIDED_SIGN_CLASSES
        double_sided.append(not is_single_sided)

    if not cuboid_vertices_list:
        return []

    keep = ~_cuboids_below_level(cuboid_vertices_list, level_plane)
    if not keep.any():
        return []
    cuboid_vertices_list = [vertices for vertices, include in zip(cuboid_vertices_list, keep) if include]
    front_colors = [color for color, include in zip(front_colors, keep) if include]
    double_sided = [value for value, include in zip(double_sided, keep) if include]

    all_vertices = np.concatenate(cuboid_vertices_list, axis=0)  # (N*8, 3)
    all_xy_and_depth = camera_model.get_xy_and_depth(all_vertices, camera_pose).reshape(-1, 8, 3)

    # Sign faces are solid-color Polygon2D (TRIANGLE_FAN). Collect them and emit
    # ONE batched TriangleList2D at the end (fan-triangulating each n-gon as
    # (0,k,k+1) — identical to Polygon2D's fan) so signs cost a single draw call
    # instead of one-per-sign. All sign geometry is opaque.
    _sign_tris, _sign_tri_colors = [], []

    def _fan_polygon(pts: np.ndarray, color: np.ndarray) -> None:
        pts = np.asarray(pts, dtype="f4")
        if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]  # match Polygon2D's closing behavior
        n = len(pts)
        if n < 3:
            return
        tris = np.stack(
            [
                np.broadcast_to(pts[0], (n - 2, 3)),
                pts[1 : n - 1],
                pts[2:n],
            ],
            axis=1,
        )  # (n-2, 3, 3): fan (0, k, k+1)
        _sign_tris.append(tris)
        _sign_tri_colors.append(np.broadcast_to(np.asarray(color, dtype="f4"), (n - 2, 3)))

    geometry_objects = []
    for i, cuboid_xy_and_depth in enumerate(all_xy_and_depth):
        front_quad = cuboid_xy_and_depth[list(_WMV4_FRONT_FACE_QUAD)]
        back_quad = cuboid_xy_and_depth[list(_WMV4_BACK_FACE_QUAD)]

        # A thin sign shows exactly one side at a time — the nearer face (smaller
        # camera-space depth). Render ONLY that side: the front artwork when the
        # camera faces the sign, otherwise a back-colored version with the same
        # silhouette. The front and back faces are two disconnected quads with no
        # side geometry between them, so drawing the far face too would let it
        # parallax-bleed past the near face's silhouette at grazing angles (a
        # strip that widens as the sign turns edge-on). Double-sided families
        # (e.g. INFORMATION / overhead street-name signs) always show the front
        # polygon, so they read the same from either viewing direction.
        show_back = (not double_sided[i]) and (front_quad[:, 2].mean() > back_quad[:, 2].mean())

        if show_back:
            if np.all(back_quad[:, 2] < 0):
                continue
            _fan_polygon(back_quad, sign_back_rgb)
            continue

        # Front (visible) face: a solid colored rectangle.
        if np.all(front_quad[:, 2] < 0):
            continue
        _fan_polygon(front_quad, front_colors[i])

    if _sign_tris:
        geometry_objects.append(TriangleList2D(np.concatenate(_sign_tris), np.concatenate(_sign_tri_colors)))
    return geometry_objects
