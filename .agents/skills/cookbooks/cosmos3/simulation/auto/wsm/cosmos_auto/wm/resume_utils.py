import json
import os
from pathlib import Path

import numpy as np
from mpire import WorkerPool
from mpire.utils import make_single_arguments

from cosmos_auto.wm.chunk_utils import get_expected_chunk_count
from cosmos_auto.wm.config import CosmosAVConfig
from cosmos_auto.wm.s3_utils import list_s3_keys

MIN_VALID_LOCAL_OUTPUT_BYTES = 1024


def attribute_name_mapping(cfg: CosmosAVConfig, attribute: str):
    if attribute in ["lidar", "videos"]:
        return attribute
    elif attribute == "hdmap":
        if cfg.version == "v1":
            return "hdmap"
        elif cfg.version == "v2":
            return "laneline_and_traffic_light"  # the last folder name is laneline_and_traffic_light
        elif cfg.version in ("v3", "v4"):
            # v3/v4 write to the world_scenario output folder
            return "world_scenario"
        elif cfg.version == "v4.1":
            # v4.1 renders occupancy voxels (render_occupancy=True) and writes to
            # world_scenario_voxels (see render_sample_hdmap_v4_1 -> prepare_output).
            # Resume must check that folder or it re-renders every clip on restart.
            return "world_scenario_voxels"
        else:
            raise ValueError(f"Invalid version: {cfg.version}")
    else:
        raise ValueError(f"Invalid attribute: {attribute}")


def find_remaining_clip_list(
    cfg: CosmosAVConfig,
    clip_id_list: list[str],
    already_generated_clip_id_chunk_id: list[str],
    multi_process: bool = False,
):
    """
    Find the remaining clip list from the filename list.

    Args:
        clip_id_list: list[str], clip id list to be processed
        already_generated_clip_id_chunk_id: list[str], already processed file list.
            Format: {clip_id}_{chunk_id}
    Return:
        remaining_clip_list: list[str], remaining clip id list
    """

    def process_chunk(clip_id_chunk_id_list):
        """
        Process the filename chunk and return the clip_id count dictionary
        """
        this_worker_clip_id_to_file = {}
        for clip_id_chunk_id in clip_id_chunk_id_list:
            clip_id = "_".join(clip_id_chunk_id.split("_")[:-1])
            if clip_id not in this_worker_clip_id_to_file:
                this_worker_clip_id_to_file[clip_id] = 0
            this_worker_clip_id_to_file[clip_id] += 1
        return this_worker_clip_id_to_file

    # split the filename list into chunks and process them in parallel
    if multi_process:
        num_workers = 8
        clip_id_chunk_id_for_workers = np.array_split(already_generated_clip_id_chunk_id, num_workers)

        # use multiprocessing to process the filename list
        clip_id_to_chunk_count = {}
        with WorkerPool(n_jobs=num_workers) as pool:
            results = pool.map_unordered(
                process_chunk,
                make_single_arguments(clip_id_chunk_id_for_workers),
                chunk_size=1,
            )

            # merge all results
            for local_clip_id_to_file in results:
                for clip_id, count in local_clip_id_to_file.items():
                    if clip_id not in clip_id_to_chunk_count:
                        clip_id_to_chunk_count[clip_id] = 0
                    clip_id_to_chunk_count[clip_id] += count

    else:
        clip_id_to_chunk_count = process_chunk(already_generated_clip_id_chunk_id)

    # Compare against the number of chunks this clip's actual frame count emits.
    remaining_clip_list = []
    for clip_id in clip_id_list:
        expected_chunk_number = get_expected_chunk_count(clip_id, cfg)
        if expected_chunk_number == 0 or clip_id_to_chunk_count.get(clip_id, 0) < expected_chunk_number:
            remaining_clip_list.append(clip_id)

    return remaining_clip_list


def get_remaining_clip_list_from_local(
    clip_id_list: list[str],
    cfg: CosmosAVConfig,
    attribute: str,
):
    folder_name = attribute_name_mapping(cfg, attribute)
    all_remaining_clip_list = set()

    for camera_name in cfg.camera_spec.camera_list:
        all_files = [
            path
            for path in (Path(cfg.output_root) / folder_name / f"{cfg.camera_type}_{camera_name}").rglob("*.mp4")
            if path.stat().st_size >= MIN_VALID_LOCAL_OUTPUT_BYTES
        ]
        already_generated_clip_id_chunk_id = [x.as_posix().split("/")[-1].split(".")[0] for x in all_files]
        this_camera_remaining_clip_list = find_remaining_clip_list(
            cfg, clip_id_list, already_generated_clip_id_chunk_id
        )

        # if this camera have remaining clip, we add it to the all_remaining_clip_list
        all_remaining_clip_list.update(this_camera_remaining_clip_list)

    return list(all_remaining_clip_list)


def get_remaining_clip_list_from_s3(
    clip_id_list: list[str],
    cfg: CosmosAVConfig,
    attribute: str,
):
    folder_name = attribute_name_mapping(cfg, attribute)
    all_remaining_clip_list = set()

    if cfg.s3_spec.s3_path is None:
        raise ValueError("s3_spec.s3_path is required when S3 upload is enabled")

    def process_camera(camera_name):
        """List one camera's output and return its remaining clips."""
        s3_prefix = os.path.join(cfg.s3_spec.s3_path, folder_name, f"{cfg.camera_type}_{camera_name}")
        all_files = list_s3_keys(s3_prefix, cfg.s3_spec.cred_path_or_profile)

        # {clip_id}_{chunk_id}
        already_generated_clip_id_chunk_id = [x.split("/")[-1].split(".")[0] for x in all_files]
        return find_remaining_clip_list(cfg, clip_id_list, already_generated_clip_id_chunk_id)

    # Process the camera list with a worker pool
    with WorkerPool(n_jobs=len(cfg.camera_spec.camera_list)) as pool:
        results = pool.map(
            process_camera,
            cfg.camera_spec.camera_list,
            chunk_size=1,
        )

        # Merge the remaining clip lists from all cameras
        for this_camera_remaining_clip_list in results:
            all_remaining_clip_list.update(this_camera_remaining_clip_list)

    return list(all_remaining_clip_list)


def get_remaining_clip_list(clip_id_list: list[str], cfg: CosmosAVConfig, attribute: str):
    """
    Get the remaining clip list from local or s3.
    If s3 is enabled, we will check the s3 folder to see if the clip is already rendered.
    If local is enabled, we will check the local folder to see if the clip is already rendered.

    Args:
        clip_id_list: list[str], clip id list to be processed
        cfg: CosmosAVConfig, config
        attribute: str, attribute name. 'hdmap', 'lidar', 'videos'

    Return:
        hdmap_remaining_clip_list: list[str]
        lidar_remaining_clip_list: list[str]
        videos_remaining_clip_list: list[str]
    """
    assert attribute in ["hdmap", "lidar", "videos"], f"Invalid attribute: {attribute}"

    if cfg.s3_spec.enable_upload:
        return get_remaining_clip_list_from_s3(clip_id_list, cfg, attribute)
    else:
        return get_remaining_clip_list_from_local(clip_id_list, cfg, attribute)
