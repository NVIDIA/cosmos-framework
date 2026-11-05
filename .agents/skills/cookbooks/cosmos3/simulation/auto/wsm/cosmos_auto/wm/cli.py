# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import time
import warnings
from bisect import bisect_left
from pathlib import Path

import cv2
import hydra
import imageio.v3 as iio
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from termcolor import colored, cprint
from tqdm import tqdm

from cosmos_auto.wm.bbox_utils import (
    build_per_vertex_color_map,
    create_bbox_geometry_objects_for_frame,
    create_bbox_projection,
    create_bbox_projection_gl,
    fix_static_objects,
    interpolate_bbox,
)
from cosmos_auto.wm.camera.base import encoder_safe_resolution
from cosmos_auto.wm.camera.ftheta import FThetaCamera
from cosmos_auto.wm.camera.pinhole import PinholeCamera
from cosmos_auto.wm.chunk_utils import get_output_chunk_starts, get_render_frame_ids
from cosmos_auto.wm.config import CosmosAVConfig
from cosmos_auto.wm.graphics_utils import render_geometries
from cosmos_auto.wm.laneline_utils import (
    build_worldspace_laneline_geometry,
    create_laneline_geometry_objects_from_data,
    prepare_laneline_type_geometry_data,
)
from cosmos_auto.wm.minimap_utils import (
    build_worldspace_minimap_geometry,
    build_worldspace_road_marking_geometry,
    create_laneline_type_projection_gl,
    create_minimap_geometry_objects_from_data,
    create_minimap_projection,
    create_minimap_projection_gl,
    create_road_marking_geometry_objects_from_data_v4,
    create_traffic_light_status_geometry_objects_from_data,
    create_traffic_light_status_geometry_objects_from_data_v4,
    create_traffic_light_status_projection_gl,
    create_traffic_sign_geometry_objects_from_data_v4,
    prepare_minimap_data,
    prepare_road_marking_data_v4,
    prepare_traffic_light_status_data,
    prepare_traffic_sign_data_v4,
    preprocess_minimap_for_rendering,
    simplify_minimap,
)
from cosmos_auto.wm.occupancy_utils import (
    apply_ground_plane_elevation,
    build_ground_elevation_tree,
    create_instanced_voxels,
    cull_voxels_intersecting_world_model,
    derive_occupancy_columns_from_pc_voxelsize,
    expand_columns_to_unit_cubes,
    precompute_ground_planes,
    prepare_static_occupancy_data,
    process_static_occupancy_data,
)
from cosmos_auto.wm.pcd_utils import batch_move_points_within_bboxes_sparse, forward_warp_multiframes_sparse_depth_only
from cosmos_auto.wm.ray_utils import ray_remote, wait_for_futures
from cosmos_auto.wm.resume_utils import get_remaining_clip_list
from cosmos_auto.wm.road_boundary_utils import (
    build_worldspace_road_boundaries,
    load_colored_road_boundaries,
    route_road_boundary_layer,
)
from cosmos_auto.wm.s3_utils import get_s3_adapter
from cosmos_auto.wm.wds_utils import get_sample

USE_RAY = True
_BELOW_LEVEL_SMOOTH_WINDOW = 10


def get_resolution_mode(cfg: CosmosAVConfig) -> str:
    mode = getattr(cfg.resolution_spec, "mode", "fixed")
    if mode not in {"fixed", "source"}:
        raise ValueError(f"Invalid resolution mode: {mode}")
    return mode


def get_camera_target_resolution(
    intrinsic_resolution: tuple[int, int],
    cfg: CosmosAVConfig,
) -> tuple[int, int]:
    if get_resolution_mode(cfg) == "source":
        return encoder_safe_resolution(intrinsic_resolution)
    return tuple(cfg.resolution_spec.camera_model_resolution)


def get_nearest_available_frame_id(frame_id: int, available_frame_ids: list[int]) -> int:
    """Return the available frame ID nearest to an absolute pose-frame ID."""
    if not available_frame_ids:
        raise ValueError("No LiDAR frames are available")

    insert_at = bisect_left(available_frame_ids, frame_id)
    if insert_at == 0:
        return available_frame_ids[0]
    if insert_at == len(available_frame_ids):
        return available_frame_ids[-1]

    before = available_frame_ids[insert_at - 1]
    after = available_frame_ids[insert_at]
    return before if frame_id - before <= after - frame_id else after


def prepare_input(clip_id: str, cfg: CosmosAVConfig):
    """
    Prepare input for rendering.
    Args:
        clip_id: the id of the clip
        cfg: the config object

    Returns:
        camera_name_to_camera_poses: the pose of the camera. dict of numpy array of shape (T, 4, 4)
        render_frame_ids: the valid frame ids used for rendering. list of int
        all_object_info: the object info. list of dict
        camera_name_to_camera_model: the camera models. dict of camera model objects
    """
    render_frame_ids = get_render_frame_ids(clip_id, cfg)

    camera_name_to_camera_model = {}
    camera_name_to_camera_poses = {}

    # generate map projection for the instance buffer
    for camera_name in cfg.camera_spec.camera_list:
        print(f"Processing {clip_id} {camera_name}...")
        # load pose
        pose_folder = "pose" if cfg.novel_pose_folder is None else cfg.novel_pose_folder
        pose_file = os.path.join(cfg.input_root, pose_folder, f"{clip_id}.tar")
        camera_key = f"pose.{camera_name}.npy"
        pose_data = get_sample(pose_file)
        pose_data_this_cam = {k: v for k, v in pose_data.items() if camera_key in k}
        pose_all_frames = np.stack([pose_data_this_cam[k] for k in sorted(pose_data_this_cam.keys())])

        # interpolate bbox from 10Hz to 30Hz
        all_object_info_file = os.path.join(cfg.input_root, "all_object_info", f"{clip_id}.tar")
        all_object_info = get_sample(all_object_info_file)

        # fix static objects in jitter
        all_object_info = fix_static_objects(all_object_info)

        # interpolate bbox if box label FPS is lower than TARGET_RENDER_FPS
        all_object_info = interpolate_bbox(all_object_info, render_frame_ids)

        # read intrinsic and build camera model.
        if cfg.camera_type == "pinhole":
            intrinsic_file = os.path.join(cfg.input_root, f"{cfg.camera_type}_intrinsic", f"{clip_id}.tar")
            intrinsic_data = get_sample(intrinsic_file)
            intrinsic_this_cam = intrinsic_data[f"{cfg.camera_type}_intrinsic.{camera_name}.npy"]
            camera_model = PinholeCamera.from_numpy(intrinsic_this_cam, device="cpu")

        elif cfg.camera_type == "ftheta":
            intrinsic_file = os.path.join(cfg.input_root, f"{cfg.camera_type}_intrinsic", f"{clip_id}.tar")

            # 3-rd party dataset does not have ftheta intrinsic, we can use the default one from RDS-HQ
            if os.path.exists(intrinsic_file):
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{cfg.camera_type}_intrinsic.{camera_name}.npy"]

            else:
                cprint(f"Ftheta intrinsic file does not exist: {intrinsic_file}", "red")
                cprint(f"===> So we will use default ftheta intrinsic for rendering", "yellow", attrs=["bold"])
                intrinsic_file = str(Path(__file__).parent / "assets" / "default_ftheta_intrinsic.tar")
                assert cfg.camera_spec.camera_to_rds_hq is not None, "camera_to_rds_hq is not provided in the config"
                camera_name_in_rds_hq = cfg.camera_spec.camera_to_rds_hq[camera_name]
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{cfg.camera_type}_intrinsic.{camera_name_in_rds_hq}.npy"]

            camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device="cpu")
        else:
            raise ValueError(f"Invalid camera type: {cfg.camera_type}")

        camera_model_w, camera_model_h = get_camera_target_resolution(
            (camera_model.width, camera_model.height),
            cfg,
        )
        rescale_h = camera_model_h / camera_model.height
        rescale_w = camera_model_w / camera_model.width

        if cfg.camera_type == "pinhole":
            camera_model.rescale(rescale_h, rescale_w)
        elif get_resolution_mode(cfg) == "source":
            camera_model.rescale_to(camera_model_w, camera_model_h)
        else:
            assert (
                abs(rescale_h - rescale_w) < 0.02
            )  # only handle the case that height is downsampled by the same ratio, required by the ftheta camera model
            camera_model.rescale(rescale_h)

        camera_name_to_camera_model[camera_name] = camera_model
        camera_name_to_camera_poses[camera_name] = pose_all_frames

    return camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model


def prepare_output(
    full_video: np.ndarray,
    render_frame_ids: list[int],
    render_name: str,
    cfg: CosmosAVConfig,
    clip_id: str,
    camera_name: str,
):
    """
    Cut full video into small clips for cosmos training / inference and save them.

    Args:
        full_video: the full video to be cut. numpy array of shape (T, H, W, 3)
        render_frame_ids: the frame ids to be rendered. list of int
        render_name: the name of the rendered video. str. 'hdmap', 'lidar', or 'rgb'
        cfg: the config object
        clip_id: the id of the clip
        camera_name: the name of the camera
    """
    TARGET_RENDER_FPS = cfg.output_frame_spec.target_render_fps
    TARGET_CHUNK_FRAME = cfg.output_frame_spec.target_chunk_frame
    OVERLAP_FRAME = cfg.output_frame_spec.overlap_frame
    if get_resolution_mode(cfg) == "source":
        # Native dimensions are already represented by the rendered frames.
        camera_model_resolution = (full_video.shape[2], full_video.shape[1])
        cosmos_resolution = camera_model_resolution
        to_cosmos_resolution = "resize"
    else:
        to_cosmos_resolution = cfg.resolution_spec.to_cosmos_resolution
        camera_model_resolution = tuple(cfg.resolution_spec.camera_model_resolution)
        cosmos_resolution = tuple(cfg.resolution_spec.cosmos_resolution)

    # sometimes the full_video does not match the target resolution, e.g. 720 * 1277, we need to resize it
    if full_video.shape[1] != camera_model_resolution[1] or full_video.shape[2] != camera_model_resolution[0]:
        print(
            f"Resizing {clip_id} {render_name} video from {full_video.shape[2]}x{full_video.shape[1]} to {camera_model_resolution[0]}x{camera_model_resolution[1]}..."
        )
        full_video = np.stack(
            [cv2.resize(frame, camera_model_resolution, interpolation=cv2.INTER_LINEAR) for frame in full_video], axis=0
        )

    if cosmos_resolution != camera_model_resolution:
        if to_cosmos_resolution == "resize":
            target_w, target_h = cosmos_resolution
            full_video = np.stack(
                [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for frame in full_video],
                axis=0,
            )
        elif to_cosmos_resolution == "center-crop":
            crop_w, crop_h = cosmos_resolution
            camera_model_w, camera_model_h = camera_model_resolution
            crop_x = (camera_model_w - crop_w) // 2
            crop_y = (camera_model_h - crop_h) // 2
            full_video = full_video[:, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        else:
            raise ValueError(f"Invalid to_cosmos_resolution: {to_cosmos_resolution}")

    output_root_p = Path(cfg.output_root)
    camera_folder_name = f"{cfg.camera_type}_{camera_name}"
    (output_root_p / render_name / camera_folder_name).mkdir(parents=True, exist_ok=True)

    chunk_starts = get_output_chunk_starts(len(render_frame_ids), cfg)
    if len(render_frame_ids) < TARGET_CHUNK_FRAME and chunk_starts:
        print(
            f"Writing short final chunk for {clip_id}: "
            f"{len(render_frame_ids)} available frames < {TARGET_CHUNK_FRAME} target frames"
        )
        chunk_starts = [0]

    if cfg.s3_spec.enable_upload and cfg.s3_spec.s3_path is None:
        raise ValueError("s3_spec.s3_path is required when S3 upload is enabled")
    s3_adapter = get_s3_adapter(cfg.s3_spec.cred_path_or_profile) if cfg.s3_spec.enable_upload else None
    for cur_idx, i in enumerate(chunk_starts):
        full_video_cut = full_video[i : i + TARGET_CHUNK_FRAME]

        # use imageio v3 API with FFMPEG plugin
        # imageio v3's imwrite handles video encoding with proper parameters
        if not cfg.s3_spec.enable_upload:
            # save to local file using imageio v3
            output_path = str(output_root_p / render_name / camera_folder_name / f"{clip_id}_{cur_idx}.mp4")
            iio.imwrite(
                output_path,
                full_video_cut,
                plugin="FFMPEG",
                fps=TARGET_RENDER_FPS,
                codec="libx264",
                macro_block_size=1,
                output_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"],
            )
        else:
            # save to memory buffer for S3 upload using imageio v3
            from io import BytesIO

            video_buffer = BytesIO()
            iio.imwrite(
                video_buffer,
                full_video_cut,
                plugin="FFMPEG",
                extension=".mp4",
                fps=TARGET_RENDER_FPS,
                codec="libx264",
                macro_block_size=1,
                output_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"],
            )

        # if upload to S3
        if cfg.s3_spec.enable_upload:
            assert s3_adapter is not None
            video_buffer.seek(0)

            # build S3 path and upload
            s3_base_path = cfg.s3_spec.s3_path.rstrip("/")
            s3_video_path = f"{s3_base_path}/{render_name}/{camera_folder_name}/{clip_id}_{cur_idx}.mp4"
            bucket_name, object_name = s3_video_path.removeprefix("s3://").split("/", 1)
            s3_adapter.upload_stream(bucket_name, object_name, video_buffer, video_buffer.getbuffer().nbytes)
            print(f"Uploaded to {s3_video_path}")


@ray_remote(use_ray=USE_RAY, num_gpus=0, num_cpus=4)
def render_sample_hdmap(
    clip_id: str, cfg: CosmosAVConfig, bbox_color_version: str = "Uniform Color", hdmap_color_version: str = "v1"
):
    minimap_types = cfg.minimap_spec.minimap_types

    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = prepare_input(
        clip_id, cfg
    )
    minimap_wds_files = [
        os.path.join(cfg.input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types
    ]

    for camera_name, camera_model in camera_name_to_camera_model.items():
        pose_all_frames = camera_name_to_camera_poses[camera_name]

        minimaps_projection_merged = np.zeros(
            (len(render_frame_ids), camera_model.height, camera_model.width, 3), dtype=np.uint8
        )
        for minimap_wds_file in minimap_wds_files:
            minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)

            # all static labels
            minimap_projection = create_minimap_projection(
                minimap_name,
                minimap_data_wo_meta_info,
                pose_all_frames[render_frame_ids],
                camera_model,
                hdmap_color_version,
            )
            minimaps_projection_merged = np.maximum(minimaps_projection_merged, minimap_projection)

        # add bounding box projection to the minimap
        bounding_box_projection = create_bbox_projection(
            all_object_info, pose_all_frames, render_frame_ids, camera_model, bbox_color_version
        )
        minimaps_projection_merged = np.maximum(minimaps_projection_merged, bounding_box_projection)

        prepare_output(
            minimaps_projection_merged,
            render_frame_ids,
            "hdmap",
            cfg,
            clip_id,
            camera_name,
        )


@ray_remote(use_ray=USE_RAY, num_gpus=0.25, num_cpus=4)
def render_sample_hdmap_v2(
    clip_id: str,
    cfg: CosmosAVConfig,
    bbox_color_version: str = "Gradient Color",
    hdmap_color_version: str = "v1",
    traffic_light_color_version: str = "v2",
):
    """
    This function will separate the lane rendering and bbox rendering, and use moderngl for occlusion-awared rendering.
    """
    minimap_types = cfg.minimap_spec.minimap_types

    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = prepare_input(
        clip_id, cfg
    )

    minimap_wds_files = [
        os.path.join(cfg.input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types
    ]

    for camera_name, camera_model in camera_name_to_camera_model.items():
        pose_all_frames = camera_name_to_camera_poses[camera_name]

        # (1) all static labels
        minimap_projection = create_minimap_projection_gl(
            minimap_wds_files,
            pose_all_frames[render_frame_ids],
            camera_model,
            depth_max=200,
            hdmap_color_version=hdmap_color_version,
        )

        # (2) add bounding box projection to the minimap
        bounding_box_projection = create_bbox_projection_gl(
            all_object_info,
            pose_all_frames,
            render_frame_ids,
            camera_model,
            depth_max=200,
            color_version=bbox_color_version,
        )

        # (3) add laneline type & traffic light status projection
        laneline_type_projection = create_laneline_type_projection_gl(
            os.path.join(cfg.input_root, "3d_lanelines", f"{clip_id}.tar"),
            pose_all_frames[render_frame_ids],
            camera_model,
            depth_max=200,
        )
        traffic_light_status_projection = create_traffic_light_status_projection_gl(
            os.path.join(cfg.input_root, "3d_traffic_lights", f"{clip_id}.tar"),
            os.path.join(cfg.input_root, "3d_traffic_lights_status", f"{clip_id}.tar"),
            pose_all_frames,
            render_frame_ids,
            camera_model,
            depth_max=200,
            traffic_light_color_version=traffic_light_color_version,
        )

        laneline_and_tl_projection = np.maximum(laneline_type_projection, traffic_light_status_projection)

        render_target = {
            "minimap": minimap_projection,
            "bbox": bounding_box_projection,
            "laneline_and_traffic_light": laneline_and_tl_projection,
        }

        for name, projection in render_target.items():
            prepare_output(
                projection,
                render_frame_ids,
                name,
                cfg,
                clip_id,
                camera_name,
            )


@ray_remote(use_ray=USE_RAY, num_gpus=0.25, num_cpus=4, max_retries=3)
def render_sample_hdmap_v3(
    clip_id: str,
    cfg: CosmosAVConfig,
    bbox_color_version: str = "v3",
    hdmap_color_version: str = "v3",
    traffic_light_color_version: str = "v2",
):
    """
    This function combines all v2 components (minimap, bbox, laneline, traffic light) into
    a single view using moderngl for occlusion-aware rendering, in ONE pass for correct
    depth ordering. Different geometries are used to represent different types of lane lines.
    """
    minimap_types = cfg.minimap_spec.minimap_types
    # Remove laneline and traffic light from minimap types since they are rendered separately
    minimap_types = [
        minimap_type for minimap_type in minimap_types if minimap_type not in ["lanelines", "traffic_lights"]
    ]
    # Apply per-sub-type skip (e.g. skip_layers=[traffic_signs,poles])
    skip_layers_set = set(cfg.skip_layers)
    if "minimap" not in skip_layers_set:
        minimap_types = [t for t in minimap_types if t not in skip_layers_set]

    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = prepare_input(
        clip_id, cfg
    )

    minimap_wds_files = [
        os.path.join(cfg.input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types
    ]

    for camera_name, camera_model in camera_name_to_camera_model.items():
        pose_all_frames = camera_name_to_camera_poses[camera_name]

        # Preload static data used across frames
        minimap_name_to_minimap_data, minimap_to_rgb = prepare_minimap_data(minimap_wds_files, hdmap_color_version)
        processed_lanelines = prepare_laneline_type_geometry_data(
            os.path.join(cfg.input_root, "3d_lanelines", f"{clip_id}.tar")
        )
        tl_position_list, tl_status_dict, tl_status_to_rgb, _ = prepare_traffic_light_status_data(
            os.path.join(cfg.input_root, "3d_traffic_lights", f"{clip_id}.tar"),
            os.path.join(cfg.input_root, "3d_traffic_lights_status", f"{clip_id}.tar"),
            traffic_light_color_version,
        )
        # Precompute bbox color map once per camera/clip
        bbox_per_vertex_color_map = build_per_vertex_color_map(bbox_color_version)

        # Camera pose at the start of the clip is assumed to be the camera pose in ego space.
        camera_pose_init = pose_all_frames[0]
        combined_frames = []
        for frame_id in render_frame_ids:
            camera_pose = pose_all_frames[frame_id]

            # Build all geometry objects for this frame
            geometry_objects = []

            # minimap layers
            if "minimap" not in skip_layers_set:
                geometry_objects.extend(
                    create_minimap_geometry_objects_from_data(
                        minimap_name_to_minimap_data,
                        camera_pose,
                        camera_model,
                        minimap_to_rgb,
                        camera_pose_init=camera_pose_init,
                        curve_subdivide=cfg.curve_subdivide,
                    )
                )

            # lanelines
            if "lanelines" not in skip_layers_set:
                geometry_objects.extend(
                    create_laneline_geometry_objects_from_data(
                        processed_lanelines,
                        camera_pose,
                        camera_model,
                        camera_pose_init=camera_pose_init,
                    )
                )

            # traffic lights
            if "traffic_lights" not in skip_layers_set:
                geometry_objects.extend(
                    create_traffic_light_status_geometry_objects_from_data(
                        tl_position_list,
                        tl_status_dict,
                        frame_id,
                        camera_pose,
                        camera_model,
                        tl_status_to_rgb,
                    )
                )

            # bounding boxes
            if "bbox" not in skip_layers_set:
                geometry_objects.extend(
                    create_bbox_geometry_objects_for_frame(
                        all_object_info[f"{frame_id:06d}.all_object_info.json"],
                        camera_pose,
                        camera_model,
                        fill_face="none" if cfg.bbox_compare else cfg.bbox_fill_face,
                        fill_face_style="solid",
                        object_type_to_per_vertex_color=bbox_per_vertex_color_map,
                        color_version=bbox_color_version,
                        line_width=4,
                        edge_color=[200, 200, 200],
                        allowed_object_types=list(cfg.bbox_object_types) or None,
                        curve_subdivide=cfg.curve_subdivide,
                    )
                )
                if cfg.bbox_compare:
                    geometry_objects.extend(
                        create_bbox_geometry_objects_for_frame(
                            all_object_info[f"{frame_id:06d}.all_object_info.json"],
                            camera_pose,
                            camera_model,
                            fill_face="none",
                            fill_face_style="solid",
                            object_type_to_per_vertex_color=bbox_per_vertex_color_map,
                            color_version=bbox_color_version,
                            line_width=4,
                            edge_color=[255, 50, 50],
                            allowed_object_types=list(cfg.bbox_object_types) or None,
                            n_edge_steps=1,
                        )
                    )

            # Render once for this frame
            combined_frame = render_geometries(
                geometry_objects,
                camera_model.height,
                camera_model.width,
                depth_max=200,
                depth_gradient=cfg.depth_gradient,
            )
            combined_frames.append(combined_frame)

        combined_projection = np.stack(combined_frames, axis=0)

        prepare_output(
            combined_projection,
            render_frame_ids,
            "world_scenario",
            cfg,
            clip_id,
            camera_name,
        )


@ray_remote(use_ray=USE_RAY, num_gpus=0.2, num_cpus=4, max_retries=3)
def render_sample_hdmap_v4(clip_id: str, cfg: CosmosAVConfig):
    """WMv4.0 renderer (signs, traffic lights, road markings, lens distortion)."""
    _render_sample_hdmap_v4_impl(clip_id, cfg, render_occupancy=False)


@ray_remote(use_ray=USE_RAY, num_gpus=0.2, num_cpus=4, max_retries=3)
def render_sample_hdmap_v4_1(clip_id: str, cfg: CosmosAVConfig):
    """WMv4.1 renderer = WMv4.0 features + static occupancy voxels."""
    _render_sample_hdmap_v4_impl(clip_id, cfg, render_occupancy=True)


def _render_sample_hdmap_v4_impl(
    clip_id: str,
    cfg: CosmosAVConfig,
    render_occupancy: bool = False,
    bbox_color_version: str = "v3",
    hdmap_color_version: str = "v3",
    traffic_light_color_version: str = "v4",
    sign_color_version: str = "v4",
    road_marking_color_version: str = "v4",
) -> None:
    """
    Shared WMv4 render implementation for both v4.0 and v4.1.

    Extends v3 with:
    - Front vs. back face coloring for traffic lights (new "back" color in the
      v4 traffic-light palette).
    - Per-Tier-1 sign category coloring (5 categories) with a dedicated back
      color. Non-Tier-1 signs render with the existing v3 generic sign color.
    Traffic signs are filtered out of the generic minimap path and rendered by
    this function so v4 owns front/back coloring end-to-end.

    When ``render_occupancy`` is True (version "v4.1"), GPU-instanced static
    occupancy voxel towers are additionally loaded from
    input_root/3d_occupancy_static/<clip>.tar and drawn each frame, and the
    output is written under "world_scenario_voxels" instead of "world_scenario"
    so v4.0 and v4.1 renders do not collide.
    """
    minimap_types = cfg.minimap_spec.minimap_types
    # Lanelines, traffic lights, traffic signs, and road markings are rendered
    # separately in v4 (road markings own per-direction arrow color/shape).
    minimap_types = [
        minimap_type
        for minimap_type in minimap_types
        if minimap_type not in ["lanelines", "traffic_lights", "traffic_signs", "road_markings"]
    ]
    minimap_types, render_colored_road_boundaries = route_road_boundary_layer(
        minimap_types,
        cfg.use_road_boundary_colors,
    )

    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = prepare_input(
        clip_id, cfg
    )

    minimap_wds_files = [
        os.path.join(cfg.input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types
    ]
    prepared_road_boundaries = None
    if render_colored_road_boundaries:
        prepared_road_boundaries = load_colored_road_boundaries(
            os.path.join(cfg.input_root, "3d_road_boundaries", f"{clip_id}.tar")
        )

    for camera_name, camera_model in camera_name_to_camera_model.items():
        pose_all_frames = camera_name_to_camera_poses[camera_name]

        # Preload static data used across frames
        minimap_name_to_minimap_data, minimap_to_rgb = prepare_minimap_data(minimap_wds_files, hdmap_color_version)
        processed_lanelines = prepare_laneline_type_geometry_data(
            os.path.join(cfg.input_root, "3d_lanelines", f"{clip_id}.tar")
        )
        tl_position_list, tl_status_dict, tl_status_to_rgb, tl_legacy_state_map = prepare_traffic_light_status_data(
            os.path.join(cfg.input_root, "3d_traffic_lights", f"{clip_id}.tar"),
            os.path.join(cfg.input_root, cfg.traffic_light_status_dir, f"{clip_id}.tar"),
            traffic_light_color_version,
        )
        sign_label_list, sign_class_to_rgb, other_sign_rgb = prepare_traffic_sign_data_v4(
            os.path.join(cfg.input_root, "3d_traffic_signs", f"{clip_id}.tar"),
            sign_color_version=sign_color_version,
            hdmap_color_version=hdmap_color_version,
        )
        prepared_road_markings = prepare_road_marking_data_v4(
            os.path.join(cfg.input_root, "3d_road_markings", f"{clip_id}.tar"),
            hdmap_color_version=hdmap_color_version,
            road_marking_color_version=road_marking_color_version,
            use_arrow_colors=cfg.use_road_marking_arrow_colors,
            use_arrow_shapes=cfg.use_road_marking_arrow_shapes,
            arrow_size_scale=cfg.road_marking_arrow_size_scale,
            use_text_colors=cfg.use_road_marking_text_colors,
            curve_subdivide=cfg.curve_subdivide,
            use_interior_tessellation=cfg.use_road_marking_interior_tessellation,
            interior_tessellation_max_edge_m=(cfg.road_marking_interior_tessellation_max_edge_m),
        )
        # Precompute bbox color map once per camera/clip
        bbox_per_vertex_color_map = build_per_vertex_color_map(bbox_color_version)

        # Camera pose at the start of the clip is assumed to be the camera pose in ego space.
        camera_pose_init = pose_all_frames[0]
        level_planes = (
            precompute_ground_planes(
                pose_all_frames,
                camera_pose_init,
                smooth_window=_BELOW_LEVEL_SMOOTH_WINDOW,
            )
            if cfg.filter_below_ego_level
            else None
        )

        # Build minimap and laneline geometry once in world space; it is uploaded to
        # the GPU a single time and projected per-frame in the vertex shader (camera
        # pose passed as a uniform), eliminating per-frame CPU projection for these
        # two layers. Below-level filtering is evaluated from the current pose plane.
        preprocessed_minimap = preprocess_minimap_for_rendering(
            minimap_name_to_minimap_data, minimap_to_rgb, curve_subdivide=cfg.curve_subdivide
        )
        worldspace_minimap = build_worldspace_minimap_geometry(
            preprocessed_minimap,
            camera_model,
            camera_pose_init,
            below_level_filter=cfg.filter_below_ego_level,
        )
        worldspace_lanelines = build_worldspace_laneline_geometry(
            processed_lanelines,
            camera_model,
            camera_pose_init,
            below_level_filter=cfg.filter_below_ego_level,
        )
        worldspace_road_boundaries = (
            build_worldspace_road_boundaries(
                prepared_road_boundaries,
                camera_model,
                camera_pose_init,
                below_level_filter=cfg.filter_below_ego_level,
            )
            if prepared_road_boundaries is not None
            else []
        )
        worldspace_road_markings = (
            build_worldspace_road_marking_geometry(
                prepared_road_markings,
                camera_model,
                below_level_filter=True,
            )
            if cfg.filter_below_ego_level
            else []
        )
        worldspace_static_geometry = (
            worldspace_minimap + worldspace_lanelines + worldspace_road_boundaries + worldspace_road_markings
        )

        # WMv4.1: static occupancy voxels. Load once per clip/camera; the
        # InstancedVoxels object caches GPU buffers and is re-projected per frame
        # by updating only the camera pose (and optionally the tower bases when a
        # per-frame ground plane is used).
        instanced_voxels = None
        ground_planes = None
        original_centers = None
        if render_occupancy:
            processed_static_occupancy = None
            ground_elev_method = None
            if cfg.use_3d_bases:
                # Training mode: derive column layout, base elevations, and heights
                # entirely from pc_voxelsize_01 — 3d_occupancy_static is not read.
                occupancy_path = os.path.join(cfg.input_root, "pc_voxelsize_01", f"{clip_id}.tar")
                processed_static_occupancy = derive_occupancy_columns_from_pc_voxelsize(
                    occupancy_path,
                    max_height=cfg.voxel_max_height,
                    color_rgb=list(cfg.voxel_color) if cfg.voxel_color is not None else None,
                )
            else:
                occupancy_path = os.path.join(cfg.input_root, "3d_occupancy_static", f"{clip_id}.tar")
                static_occupancy_data = prepare_static_occupancy_data(occupancy_path)
                ground_elev_method = cfg.voxel_ground_elevation
                if static_occupancy_data is not None:
                    ground_tree, ground_z, base_elevation = None, None, None
                    if ground_elev_method == "hdmap":
                        minimap_data_for_ground = minimap_name_to_minimap_data
                        if prepared_road_boundaries is not None:
                            minimap_data_for_ground = dict(minimap_name_to_minimap_data)
                            minimap_data_for_ground["road_boundaries"] = [
                                boundary["polyline"] for boundary in prepared_road_boundaries
                            ]
                        ground_tree, ground_z = build_ground_elevation_tree(
                            minimap_data_for_ground, processed_lanelines
                        )
                    elif ground_elev_method == "base" and "base" in static_occupancy_data:
                        base_elevation = np.array(static_occupancy_data["base"], dtype=np.float32)
                    processed_static_occupancy = process_static_occupancy_data(
                        static_occupancy_data,
                        render_mode="classification",
                        ground_elevation_tree=ground_tree,
                        ground_elevation_z=ground_z,
                        max_height=cfg.voxel_max_height,
                        base_elevation=base_elevation,
                        color_rgb=list(cfg.voxel_color) if cfg.voxel_color is not None else None,
                    )
            if processed_static_occupancy is None:
                warnings.warn(
                    f"WMv4.1 occupancy is unavailable for clip {clip_id} at {occupancy_path}; "
                    "rendering the remaining world-model layers without voxels.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # WMv4.0-priority: delete voxel columns intersecting static world-
                # model elements (signs, traffic lights, poles) so they stay visible.
                if cfg.voxel_cull_world_model:
                    processed_static_occupancy = cull_voxels_intersecting_world_model(
                        processed_static_occupancy,
                        sign_label_list=sign_label_list,
                        tl_position_list=tl_position_list,
                        pole_polylines=minimap_name_to_minimap_data.get("poles"),
                        margin=cfg.voxel_cull_margin,
                    )
                # Split columns into stacked unit cubes so horizontal cube-boundary
                # lines are drawn at every voxel level (height/depth cues).
                if cfg.voxel_stacked_cubes:
                    processed_static_occupancy = expand_columns_to_unit_cubes(processed_static_occupancy)
                # Per-frame tilted-plane adjustment (not used in use_3d_bases mode).
                if not cfg.use_3d_bases and ground_elev_method == "tilted_plane":
                    ground_planes = precompute_ground_planes(
                        pose_all_frames, camera_pose_init, cfg.voxel_ground_smooth_window
                    )
                instanced_voxels = create_instanced_voxels(
                    processed_static_occupancy,
                    camera_model,
                    max_distance=cfg.voxel_max_distance,
                    max_lateral=cfg.voxel_max_lateral,
                    draw_edges=cfg.voxel_edges,
                    edge_color=list(cfg.voxel_edge_color),
                    edge_width=cfg.voxel_edge_width,
                )
                if instanced_voxels is not None and ground_planes is not None:
                    original_centers = processed_static_occupancy["base_centers"].copy()

        combined_frames = []
        for frame_id in render_frame_ids:
            camera_pose = pose_all_frames[frame_id]

            # Build all geometry objects for this frame
            geometry_objects = []

            # Static map layers: GPU-projected, only the camera pose uniform changes.
            world_to_camera = np.linalg.inv(camera_pose).astype(np.float32)
            frame_level_plane = level_planes[frame_id] if level_planes is not None else None
            for geom in worldspace_static_geometry:
                geom.set_world_to_camera(world_to_camera)
                if frame_level_plane is not None:
                    geom.set_level_plane(frame_level_plane)
            geometry_objects.extend(worldspace_static_geometry)

            # traffic lights (front status color / back dedicated color)
            geometry_objects.extend(
                create_traffic_light_status_geometry_objects_from_data_v4(
                    tl_position_list,
                    tl_status_dict,
                    frame_id,
                    camera_pose,
                    camera_model,
                    tl_status_to_rgb,
                    legacy_state_map=tl_legacy_state_map,
                    level_plane=frame_level_plane,
                )
            )

            # traffic signs (per Tier-1 category front color / dedicated back color)
            geometry_objects.extend(
                create_traffic_sign_geometry_objects_from_data_v4(
                    sign_label_list,
                    camera_pose,
                    camera_model,
                    sign_class_to_rgb,
                    other_sign_rgb,
                    single_sided_categories=cfg.single_sided_sign_categories,
                    level_plane=frame_level_plane,
                )
            )

            if not cfg.filter_below_ego_level:
                geometry_objects.extend(
                    create_road_marking_geometry_objects_from_data_v4(
                        prepared_road_markings,
                        camera_pose,
                        camera_model,
                    )
                )

            # WMv4.1: static occupancy voxels (GPU-instanced towers). Depth-tested
            # against the rest of the scene so signs/lights/boxes remain visible.
            if instanced_voxels is not None:
                if ground_planes is not None:
                    assert original_centers is not None
                    frame_ground_plane = ground_planes[frame_id]
                    adjusted = original_centers.copy()
                    adjusted[:, 2] += apply_ground_plane_elevation(original_centers[:, :2], frame_ground_plane)
                    instanced_voxels.update_centers(adjusted)
                instanced_voxels.set_camera_pose(camera_pose)
                geometry_objects.append(instanced_voxels)

            # bounding boxes
            geometry_objects.extend(
                create_bbox_geometry_objects_for_frame(
                    all_object_info[f"{frame_id:06d}.all_object_info.json"],
                    camera_pose,
                    camera_model,
                    fill_face="none" if cfg.bbox_compare else cfg.bbox_fill_face,
                    fill_face_style="solid",
                    object_type_to_per_vertex_color=bbox_per_vertex_color_map,
                    color_version=bbox_color_version,
                    line_width=4,
                    edge_color=[200, 200, 200],
                    curve_subdivide=cfg.curve_subdivide,
                )
            )
            if cfg.bbox_compare:
                # Overlay old straight edges (red) alongside the curved edges (white)
                # so the fisheye distortion is directly visible. No face fill.
                geometry_objects.extend(
                    create_bbox_geometry_objects_for_frame(
                        all_object_info[f"{frame_id:06d}.all_object_info.json"],
                        camera_pose,
                        camera_model,
                        fill_face="none",
                        fill_face_style="solid",
                        object_type_to_per_vertex_color=bbox_per_vertex_color_map,
                        color_version=bbox_color_version,
                        line_width=4,
                        edge_color=[255, 50, 50],
                        curve_subdivide=1,
                    )
                )

            # Render once for this frame
            combined_frame = render_geometries(
                geometry_objects,
                camera_model.height,
                camera_model.width,
                depth_max=200,
                depth_gradient=cfg.depth_gradient,
            )
            combined_frames.append(combined_frame)

        combined_projection = np.stack(combined_frames, axis=0)

        prepare_output(
            combined_projection,
            render_frame_ids,
            "world_scenario_voxels" if render_occupancy else "world_scenario",
            cfg,
            clip_id,
            camera_name,
        )


@ray_remote(use_ray=USE_RAY, num_gpus=1, num_cpus=2)
def render_sample_lidar(clip_id: str, cfg: CosmosAVConfig, accumulate_lidar_frames: int = 2):
    INPUT_POSE_FPS = cfg.input_fps_spec.input_pose_fps
    INPUT_LIDAR_FPS = cfg.input_fps_spec.input_lidar_fps

    camera_name_to_camera_poses, render_frame_ids, all_object_info, camera_name_to_camera_model = prepare_input(
        clip_id, cfg
    )

    for camera_name, camera_model in camera_name_to_camera_model.items():
        camera_model_w, camera_model_h = camera_model.width, camera_model.height
        pose_all_frames = camera_name_to_camera_poses[camera_name]
        # load lidar data
        lidar_tar = os.path.join(cfg.input_root, "lidar_raw", f"{clip_id}.tar")
        lidar_data = get_sample(lidar_tar)  # 'xxxxxx.xyz' & 'xxxxxx.lidar_to_world'

        # Generate lidar depth. LiDAR keys and render_frame_ids are absolute pose-
        # frame IDs, so accumulation must clamp to the pose timeline rather than
        # the number of sampled render frames.
        lidar_depth_list = []
        interp_rate = INPUT_POSE_FPS // INPUT_LIDAR_FPS
        pose_frame_max = len(pose_all_frames) - 1
        lidar_frame_ids = sorted(
            int(key.split(".")[0])
            for key in lidar_data
            if key.endswith(".lidar_raw.npz") and key.split(".")[0].isdigit()
        )
        available_lidar_frame_ids = [frame_id for frame_id in lidar_frame_ids if frame_id <= pose_frame_max]
        if not available_lidar_frame_ids:
            raise ValueError(f"No usable LiDAR frames found for clip {clip_id}")

        for frame_idx in render_frame_ids:
            accumulate_idx = [
                min(max(0, frame_idx + i * interp_rate), pose_frame_max)
                for i in range(-accumulate_lidar_frames, accumulate_lidar_frames + 1)
            ]
            low_fps_frame_indices = [
                get_nearest_available_frame_id(idx, available_lidar_frame_ids) for idx in accumulate_idx
            ]

            accumulated_points, original_bbox_dicts, moved_bbox_dicts = [], [], []

            for i in range(len(accumulate_idx)):
                lidar_points = (
                    lidar_data[f"{low_fps_frame_indices[i]:06d}.lidar_raw.npz"]["xyz"].astype(np.float32).reshape(-1, 3)
                )

                distances = np.linalg.norm(lidar_points, axis=1)
                threshold = 3.0
                lidar_points = lidar_points[distances >= threshold]
                lidar_to_world = lidar_data[f"{low_fps_frame_indices[i]:06d}.lidar_raw.npz"]["lidar_to_world"]
                lidar_points_world = (
                    lidar_to_world @ np.concatenate([lidar_points, np.ones([lidar_points.shape[0], 1])], axis=1).T
                ).T[:, :3]
                accumulated_points.append(torch.tensor(lidar_points_world, device="cuda"))
                original_bbox_dicts.append(all_object_info[f"{low_fps_frame_indices[i]:06d}.all_object_info.json"])
                moved_bbox_dicts.append(all_object_info[f"{frame_idx:06d}.all_object_info.json"])

            accumulated_points_corrected, _ = batch_move_points_within_bboxes_sparse(
                accumulated_points,
                original_bbox_dicts,
                moved_bbox_dicts,
                torch.tensor(low_fps_frame_indices),
                dynamic_only=True,
                bbox_scale=1.2,  # enlarge bbox to include all points
            )
            lengths_tensor = torch.tensor([t.shape[0] for t in accumulated_points_corrected], device="cuda")
            cat_points = torch.cat(accumulated_points_corrected, dim=0)
            w2cs = torch.inverse(torch.tensor(pose_all_frames[frame_idx])).unsqueeze(0).cuda()
            if cfg.camera_type == "ftheta":
                cx_cy_w_h = camera_model.intrinsics[:4]
                bw_poly = camera_model._bw_poly.coef
                is_bw_poly = 1
                intrinsics = torch.tensor([*cx_cy_w_h, *bw_poly, is_bw_poly]).unsqueeze(0).to(cat_points)
            else:
                intrinsics = torch.tensor(camera_model.intrinsics).unsqueeze(0).to(cat_points)
            depth, mask = forward_warp_multiframes_sparse_depth_only(
                w2cs,
                intrinsics,
                buffer_points=cat_points.cuda(),
                buffer_length=lengths_tensor.cuda(),
                target_h=camera_model_h,
                target_w=camera_model_w,
                center_only=False,
                is_ftheta=(cfg.camera_type == "ftheta"),
                expanded_kernel=4,
            )
            depth = depth.squeeze(0).squeeze(0).cpu().numpy()
            lidar_depth_list.append(depth)
        lidar_depth = np.stack(lidar_depth_list, axis=0)  # T, H, W
        # clamp to 75
        lidar_depth_vis = np.clip(lidar_depth, 0, 75) / 75.0 * 255
        lidar_depth_vis = lidar_depth_vis.astype(np.uint8)
        lidar_depth_vis = np.stack([lidar_depth_vis] * 3, axis=-1)  # T, H, W, 3

        # prepare output
        prepare_output(
            lidar_depth_vis,
            render_frame_ids,
            "lidar",
            cfg,
            clip_id,
            camera_name,
        )


@ray_remote(use_ray=USE_RAY, num_gpus=0, num_cpus=2)
def render_sample_rgb(
    clip_id: str,
    cfg: CosmosAVConfig,
):
    """
    This function is used to render / sample the RGB video for post-training use.
    """
    # Recorded RGB resampling is optional; WSM generation reads scene annotations.
    try:
        from decord import VideoReader
    except ImportError as exc:
        raise ImportError("RGB resampling requires the optional cosmos-auto-wsm[video] dependencies.") from exc

    camera_names = cfg.camera_spec.camera_list
    render_frame_ids = get_render_frame_ids(clip_id, cfg)
    frame_index_in_gt_video = [
        i // (cfg.input_fps_spec.input_pose_fps // cfg.input_fps_spec.gt_video_fps) for i in render_frame_ids
    ]

    for camera_name in camera_names:
        # load rgb
        rgb_file = os.path.join(cfg.input_root, f"{cfg.camera_type}_{camera_name}", f"{clip_id}.mp4")
        # load all frames
        if get_resolution_mode(cfg) == "source":
            vr = VideoReader(rgb_file)
        else:
            camera_model_w, camera_model_h = cfg.resolution_spec.camera_model_resolution
            vr = VideoReader(rgb_file, width=camera_model_w, height=camera_model_h)
        all_frames = vr.get_batch(frame_index_in_gt_video).asnumpy()  # T, H, W, 3

        prepare_output(
            all_frames,
            frame_index_in_gt_video,
            "videos",
            cfg,
            clip_id,
            camera_name,
        )


@hydra.main(version_base=None, config_path="conf", config_name="render")
def main(cfg: CosmosAVConfig):
    if len(cfg.skip) > 0:
        assert all(s in ["hdmap", "lidar"] for s in cfg.skip), "skip must be in ['hdmap', 'lidar']"

    if get_resolution_mode(cfg) == "source":
        print(
            "Original Resolution -> Match each camera intrinsic resolution "
            "(rounded up to encoder-safe even dimensions)."
        )
    elif cfg.resolution_spec.to_cosmos_resolution == "resize":
        cosmos_resolution = cfg.resolution_spec.cosmos_resolution
        camera_model_resolution = cfg.resolution_spec.camera_model_resolution
        if camera_model_resolution != cosmos_resolution:
            print(
                f"Original Resolution -> 1. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(camera_model_resolution, 'yellow', attrs=['bold'])} for {cfg.camera_type} camera model."
            )
            print(
                f"                    -> 2. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(cosmos_resolution, 'yellow', attrs=['bold'])} for cosmos model."
            )
        else:
            print(
                f"Original Resolution -> 1. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(cosmos_resolution, 'yellow', attrs=['bold'])} for {cfg.camera_type} camera model and cosmos model."
            )
    else:
        cosmos_resolution = cfg.resolution_spec.cosmos_resolution
        camera_model_resolution = cfg.resolution_spec.camera_model_resolution
        print(
            f"Original Resolution -> 1. {colored('Resize', 'yellow', attrs=['bold'])} to {colored(camera_model_resolution, 'yellow', attrs=['bold'])} for {cfg.camera_type} camera model."
        )
        print(
            f"                    -> 2. {colored('Center Crop', 'magenta', attrs=['bold'])} to {colored(cosmos_resolution, 'magenta', attrs=['bold'])} for cosmos model."
        )

    rendering_target = []
    if "hdmap" not in cfg.skip:
        rendering_target.append("hdmap")
    if "lidar" not in cfg.skip:
        rendering_target.append("lidar")
    if cfg.post_training:
        rendering_target.append("videos")

    print(f"Rendering target: {colored(', '.join(rendering_target), 'green', attrs=['bold'])}")

    # if novel_pose_folder is provided, we also change the output folder
    if cfg.novel_pose_folder is not None:
        cfg.output_root = f"{cfg.output_root}_{cfg.novel_pose_folder}"

    cprint(f"========== Config for rendering ==========", "green", attrs=["bold"])
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cprint(f"========== Config for rendering ==========", "green", attrs=["bold"])

    # get all clip ids
    if cfg.clip_id_json is None:
        input_root_p = Path(cfg.input_root)
        pose_folder = "pose" if cfg.novel_pose_folder is None else cfg.novel_pose_folder
        clip_list = (input_root_p / pose_folder).rglob("*.tar")
        clip_list = [c.stem for c in clip_list]
    elif cfg.clip_id_json.endswith(".json"):
        with open(cfg.clip_id_json) as clip_id_file:
            clip_list = json.load(clip_id_file)
    else:
        clip_list = [cfg.clip_id_json]

    # shuffle the clip list
    np.random.seed(0)
    np.random.shuffle(clip_list)

    if cfg.manual_split is not None:
        local_node, world_nodes = cfg.manual_split.split(",")
        local_node = int(local_node)
        world_nodes = int(world_nodes)
        clip_list = clip_list[local_node::world_nodes]

    if cfg.version == "v1":
        render_sample_hdmap_func = render_sample_hdmap
    elif cfg.version == "v2":
        render_sample_hdmap_func = render_sample_hdmap_v2
    elif cfg.version == "v3":
        render_sample_hdmap_func = render_sample_hdmap_v3
    elif cfg.version == "v4":
        render_sample_hdmap_func = render_sample_hdmap_v4
    elif cfg.version == "v4.1":
        render_sample_hdmap_func = render_sample_hdmap_v4_1
    else:
        raise ValueError(f"Invalid version: {cfg.version}")

    # resuming
    if cfg.skip_resume:
        hdmap_remaining_clip_list, lidar_remaining_clip_list, videos_remaining_clip_list = [], [], []
        if "hdmap" not in cfg.skip:
            hdmap_remaining_clip_list = clip_list
        if "lidar" not in cfg.skip:
            lidar_remaining_clip_list = clip_list
        if cfg.post_training:
            videos_remaining_clip_list = clip_list
    else:
        print(f"Now we are fetching processed clips, it may take a while...")
        start_time = time.time()
        hdmap_remaining_clip_list, lidar_remaining_clip_list, videos_remaining_clip_list = [], [], []
        if "hdmap" not in cfg.skip:
            hdmap_remaining_clip_list = get_remaining_clip_list(clip_list, cfg, "hdmap")
            print(
                f"Remaining hdmap clip list: {colored(len(hdmap_remaining_clip_list), 'green', attrs=['bold'])} / {colored(len(clip_list), 'green', attrs=['bold'])}"
            )
        if "lidar" not in cfg.skip:
            lidar_remaining_clip_list = get_remaining_clip_list(clip_list, cfg, "lidar")
            print(
                f"Remaining lidar clip list: {colored(len(lidar_remaining_clip_list), 'green', attrs=['bold'])} / {colored(len(clip_list), 'green', attrs=['bold'])}"
            )
        if cfg.post_training:
            videos_remaining_clip_list = get_remaining_clip_list(clip_list, cfg, "videos")
            print(
                f"Remaining videos clip list: {colored(len(videos_remaining_clip_list), 'green', attrs=['bold'])} / {colored(len(clip_list), 'green', attrs=['bold'])}"
            )
        end_time = time.time()
        print(
            f"Time taken to get remaining clip list: {colored(end_time - start_time, 'yellow', attrs=['bold'])} seconds"
        )

    if USE_RAY:
        ray_temp_dir = os.environ.get("RAY_TEMP_DIR")
        if ray_temp_dir:
            print(f"Using isolated Ray temp directory: {ray_temp_dir}")
            ray.init(_temp_dir=ray_temp_dir)
        else:
            ray.init()
        futures = []
        hdmap_futures = [render_sample_hdmap_func.remote(clip_id, cfg) for clip_id in hdmap_remaining_clip_list]
        futures.extend(hdmap_futures)

        lidar_futures = [render_sample_lidar.remote(clip_id, cfg) for clip_id in lidar_remaining_clip_list]
        futures.extend(lidar_futures)

        rgb_futures = [render_sample_rgb.remote(clip_id, cfg) for clip_id in videos_remaining_clip_list]
        futures.extend(rgb_futures)

        wait_for_futures(futures)
    else:
        for clip_id in tqdm(hdmap_remaining_clip_list):
            render_sample_hdmap_func(clip_id, cfg)
        for clip_id in tqdm(lidar_remaining_clip_list):
            render_sample_lidar(clip_id, cfg)
        for clip_id in tqdm(videos_remaining_clip_list):
            render_sample_rgb(clip_id, cfg)


if __name__ == "__main__":
    main()
