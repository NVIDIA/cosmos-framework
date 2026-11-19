# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon


def interpolate_polyline_to_points(polyline, segment_interval=0.025):
    """
    polyline:
        numpy.ndarray, shape (N, 3) or list of points

    Returns:
        points: numpy array, shape (interpolate_num*N, 3)
    """
    polyline = np.asarray(polyline, dtype=float)

    if len(polyline) < 2:
        return polyline.copy()

    # Vectorized computation of all segments at once
    start_points = polyline[:-1]  # shape (N-1, 3)
    end_points = polyline[1:]  # shape (N-1, 3)

    # Calculate distances for all segments
    segment_vectors = end_points - start_points  # shape (N-1, 3)
    distances = np.linalg.norm(segment_vectors, axis=1)  # shape (N-1,)

    # Handle edge cases: very small distances or invalid values
    valid_mask = np.isfinite(distances) & (distances >= 1e-10)

    # Calculate interpolation numbers for all segments
    interpolate_nums = np.floor(distances / segment_interval).astype(int)
    interpolate_nums = np.clip(interpolate_nums, 2, 10000)

    # For invalid segments, set interpolate_num to 1 (just start point)
    interpolate_nums[~valid_mask] = 1

    # Calculate total number of output points
    # Valid segments: (interpolate_num - 1) points each
    # Invalid segments: 1 point each
    # Plus 1 final point
    valid_points = np.sum((interpolate_nums - 1)[valid_mask])
    invalid_points = np.sum(~valid_mask)  # Each invalid segment contributes 1 point
    total_points = valid_points + invalid_points + 1

    # Pre-allocate result array
    result = np.empty((total_points, 3), dtype=float)

    # Fill the result array efficiently
    current_idx = 0
    for i, num_points in enumerate(interpolate_nums):
        if valid_mask[i]:
            # Create interpolation parameters for this segment
            # Use endpoint=True and remove last point
            t = np.linspace(0, 1, num_points, endpoint=True)[:-1]
            # Vectorized interpolation: start + t * (end - start)
            segment_points = start_points[i] + t[:, np.newaxis] * segment_vectors[i]
        else:
            # For invalid segments, just use the start point
            segment_points = start_points[i : i + 1]

        result[current_idx : current_idx + len(segment_points)] = segment_points
        current_idx += len(segment_points)

    # Add the final point
    result[current_idx] = polyline[-1]

    return result


def recursive_transform(polythings, transform):
    """
    Args:
        polythings: list of points / list of list of points / list of list of list of points
            where points a list of serveral [x,y,z] points

        transform: np.array, shape (4, 4)

    Returns:
        transformed_polythings: list of points / list of list of points / list of list of list of points
            where points a list of serveral [x,y,z] points (in numpy array)
    """

    def recursive_transform_single(polything, transform):
        if len(polything[0]) == 3 and isinstance(polything[0][0], (int, float)):
            trans_points = np.dot(transform[:3, :3], np.array(polything).T).T + transform[:3, 3]
            return trans_points.tolist()
        else:
            return [recursive_transform_single(sub_polything, transform) for sub_polything in polything]

    return [recursive_transform_single(polything, transform) for polything in polythings]


def transform_points_to_camera(points, points_to_camera):
    """
    Args:
        points: np.ndarray, shape (N, 3), dtype=np.float32
        points_to_camera: np.ndarray, shape (4, 4), dtype=np.float32

    Returns:
        points_in_camera: np.ndarray, shape (N, 3), dtype=np.float32
    """
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_in_camera = np.einsum("ij,nj->ni", points_to_camera, points)

    return points_in_camera


def save_pcd(filename, pcd, colors=None, nrms=None):
    import open3d as o3d

    # assumes color in range [0, 1]
    # Convert torch tensors to numpy arrays if necessary
    if torch.is_tensor(pcd):
        pcd = pcd.detach().cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.detach().cpu().numpy()
    if nrms is not None and torch.is_tensor(nrms):
        nrms = nrms.detach().cpu().numpy()

    # Proceed with the existing logic
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3))
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3).astype(float))

    if nrms is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(nrms.reshape(-1, 3))

    o3d.io.write_point_cloud(filename, point_cloud)


def visualize_bbox(file_path, bboxes):
    # Create a template quad mesh of a unit box and convert it to a mesh
    unit_box = trimesh.primitives.Box().to_mesh()

    # Initialize an empty list to collect meshes
    meshes = []

    for key, bbox in bboxes.items():
        object_to_world = bbox["object_to_world"]  # 4x4 transformation matrix tensor
        object_lwh = bbox["object_lwh"]  # [3] tensor describing the length, width, and height

        # Convert tensors to numpy arrays and ensure they are on CPU
        object_to_world_np = object_to_world.cpu().numpy()
        object_lwh_np = object_lwh.cpu().numpy().flatten()

        # Create scaling matrix
        scaling_matrix = np.diag(np.append(object_lwh_np, 1.0))

        # Compute total transformation matrix
        total_transform = np.dot(object_to_world_np, scaling_matrix)

        # Create a copy of the unit box mesh
        transformed_box = unit_box.copy()

        # Apply total transformation to the mesh
        transformed_box.apply_transform(total_transform)

        # Append transformed mesh to the list
        meshes.append(transformed_box)

    # Concatenate all the meshes
    combined_mesh = trimesh.util.concatenate(meshes)

    # Save the combined mesh to file_path with trimesh
    combined_mesh.export(file_path)


def get_removal_mask(input_pcd, original_bbox_dict, h, w, x_scale=1.0, y_scale=1.4):
    """
    For each image in the batch, find the regions within each original_bbox in input_pcd,
    create masks for those regions, and aggregate the masks. The 2D bounding boxes are
    expanded in the x and y dimensions based on the provided scale factors.

    Parameters:
        input_pcd (torch.Tensor): Point cloud data of shape [B, f, 3], where
                                  B is the number of points, f is the number of frames.
        original_bbox_dict (dict): Dictionary of original bounding boxes with indices as keys.
        h (int): Height of the images.
        w (int): Width of the images.
        x_scale (float): Scale factor for expanding the 2D bounding box in the x dimension.
        y_scale (float): Scale factor for expanding the 2D bounding box in the y dimension.

    Returns:
        torch.Tensor: The aggregated mask of shape [b, h, w], where b = B_points / (h * w).
    """
    B, f, _ = input_pcd.shape
    b = B * f // (h * w)  # Calculate batch size

    # For efficiency, precompute the inverse transformations and half sizes for all original bboxes
    bbox_data = {}
    for idx, bbox in original_bbox_dict.items():
        original_object_to_world = bbox["object_to_world"]  # numpy array [4,4]
        world_to_original_object = np.linalg.inv(original_object_to_world)  # numpy array [4,4]
        world_to_original_object = torch.from_numpy(world_to_original_object).to(
            input_pcd.device, dtype=input_pcd.dtype
        )
        object_lwh = torch.from_numpy(bbox["object_lwh"]).to(input_pcd.device, dtype=input_pcd.dtype)  # shape [3]
        half_lwh = object_lwh / 2  # shape [3]
        bbox_data[idx] = {"world_to_original_object": world_to_original_object, "half_lwh": half_lwh}

    points = input_pcd.reshape(-1, 3)  # shape [B_points, 3]
    B_points = points.shape[0]

    ones = torch.ones(B_points, 1, device=input_pcd.device, dtype=input_pcd.dtype)
    points_homo = torch.cat([points, ones], dim=1)  # shape [B_points, 4]

    result_mask = torch.zeros((b, h, w), dtype=torch.bool, device=input_pcd.device)

    for idx in original_bbox_dict.keys():
        world_to_original_object = bbox_data[idx]["world_to_original_object"]
        half_lwh = bbox_data[idx]["half_lwh"]
        points_obj = torch.matmul(points_homo, world_to_original_object.T)  # shape [B_points, 4]

        within_bbox = torch.all(
            torch.abs(points_obj[:, :3]) <= (half_lwh + 1e-6), dim=1
        )  # shape [B_points], boolean tensor
        within_bbox = rearrange(within_bbox, "(b h w f)-> (b f) h w", b=b // f, h=h, w=w, f=f)

        if not within_bbox.any():
            continue

        y_coords = torch.arange(h, device=input_pcd.device).view(1, h, 1).expand(b, h, w)
        x_coords = torch.arange(w, device=input_pcd.device).view(1, 1, w).expand(b, h, w)

        x_coords_masked = x_coords.clone()
        y_coords_masked = y_coords.clone()

        x_coords_masked[~within_bbox] = w
        y_coords_masked[~within_bbox] = h

        x_coords_max_masked = x_coords.clone()
        y_coords_max_masked = y_coords.clone()

        x_coords_max_masked[~within_bbox] = -1
        y_coords_max_masked[~within_bbox] = -1

        min_x, _ = x_coords_masked.view(b, -1).min(dim=1)
        min_y, _ = y_coords_masked.view(b, -1).min(dim=1)
        max_x, _ = x_coords_max_masked.view(b, -1).max(dim=1)
        max_y, _ = y_coords_max_masked.view(b, -1).max(dim=1)
        valid_mask = (min_x <= max_x) & (min_y <= max_y)
        width_x = max_x - min_x + 1
        width_y = max_y - min_y + 1

        delta_x = ((width_x * (x_scale - 1)) / 2).to(input_pcd.dtype)
        delta_y = ((width_y * (y_scale - 1)) / 2).to(input_pcd.dtype)

        min_x_new = (min_x - delta_x).clamp(0, w - 1)
        max_x_new = (max_x + delta_x).clamp(0, w - 1)
        min_y_new = (min_y - delta_y).clamp(0, h - 1)
        max_y_new = (max_y + delta_y).clamp(0, h - 1)

        min_x_int = min_x_new.floor().long()
        max_x_int = max_x_new.ceil().long()
        min_y_int = min_y_new.floor().long()
        max_y_int = max_y_new.ceil().long()

        # Create mask per image using broadcasting
        mask = ~(
            (x_coords >= min_x_int.view(b, 1, 1))
            & (x_coords <= max_x_int.view(b, 1, 1))
            & (y_coords >= min_y_int.view(b, 1, 1))
            & (y_coords <= max_y_int.view(b, 1, 1))
            & valid_mask.view(b, 1, 1)  # Ensure we only consider valid images
        )
        result_mask = result_mask | mask

    return result_mask


def batch_move_points_within_bboxes(
    batch_input_pcd,
    batch_input_masks,
    batch_original_bbox_dicts,
    batch_moved_bbox_dicts,
    pcd_indices,
    dynamic_only,
    bbox_scale,
    dilate_mask,
    H,
    W,
):
    """
    Process a batch of point clouds and bounding boxes, optimizing computations by considering masks,
    only moving points for moving objects, and reusing computations for identical point clouds.

    Parameters:
        batch_input_pcd (torch.Tensor): Batch of point clouds, shape [B, N, 3]
        batch_input_masks (torch.Tensor): Batch of masks, shape [B, N, 1]
        batch_original_bbox_dicts (list): List of length B, each element is a dict of original bounding boxes.
        batch_moved_bbox_dicts (list): List of length B, each element is a dict of moved bounding boxes.
        pcd_indices (torch.Tensor): Tensor of size [B], indicating the point cloud index for each batch.

    Returns:
        torch.Tensor: Batch of updated point clouds, shape [B, N, 3]
        torch.Tensor: Batch of masks indicating moved points, shape [B, N]
    """

    def _compute_within_bbox(input_pcd, input_mask, original_bbox_dict):
        """
        Compute which points are within each bbox, considering only masked points.

        Returns:
            within_bbox_dict: Dictionary mapping bbox_idx to indices of points within that bbox.
        """
        N, _ = input_pcd.shape

        # Get indices of masked points
        masked_indices = torch.nonzero(input_mask.squeeze() > 0).squeeze()

        if masked_indices.numel() == 0:
            return {}  # No masked points to consider

        points = input_pcd[masked_indices, :]  # shape [M, 3]

        # Convert points to homogeneous coordinates
        ones = torch.ones(points.shape[0], 1, device=input_pcd.device, dtype=input_pcd.dtype)
        points_homo = torch.cat([points, ones], dim=1)  # shape [M, 4]

        within_bbox_dict = {}
        for idx, bbox in original_bbox_dict.items():
            # Skip if object_is_moving is False or not present
            if not bbox.get("object_is_moving", False) and dynamic_only:
                continue

            original_object_to_world = bbox["object_to_world"]  # numpy array [4,4]
            world_to_original_object = np.linalg.inv(original_object_to_world)  # numpy array [4,4]
            world_to_original_object = torch.from_numpy(world_to_original_object).to(
                input_pcd.device, dtype=input_pcd.dtype
            )
            object_lwh = torch.tensor(bbox["object_lwh"]).to(input_pcd.device, dtype=input_pcd.dtype)  # shape [3]
            half_lwh = object_lwh / 2  # shape [3]
            half_lwh *= bbox_scale

            # Transform points to object coordinate system
            points_obj = torch.matmul(points_homo, world_to_original_object.T)  # shape [M, 4]

            # Check which points are within the bbox
            within_bbox_mask = torch.all(
                torch.abs(points_obj[:, :3]) <= (half_lwh + 1e-6), dim=1
            )  # shape [M], boolean tensor

            indices_in_bbox = masked_indices[within_bbox_mask]  # Original indices in input_pcd

            if indices_in_bbox.numel() > 0:
                if dilate_mask:
                    # print ("Before: ", indices_in_bbox.numel())
                    mask_1d = torch.zeros(N, dtype=input_pcd.dtype, device=input_pcd.device)
                    mask_1d[indices_in_bbox] = 1.0
                    mask_2d = mask_1d.view(1, 1, H, W)  # shape [1, 1, H, W]
                    mask_closed = morphological_closing(mask_2d, kernel_size=11)
                    mask_closed = (mask_closed > 0.5).squeeze()  # shape [H, W]
                    new_mask_1d = mask_closed.view(N)
                    indices_in_bbox = new_mask_1d.nonzero(as_tuple=False).squeeze()
                    # print ("After: ", indices_in_bbox.numel())
                within_bbox_dict[idx] = indices_in_bbox

        return within_bbox_dict

    B, N, _ = batch_input_pcd.shape
    output_pcd = batch_input_pcd.clone()
    output_mask = torch.ones(B, N, dtype=torch.uint8, device=batch_input_pcd.device)

    pcd_indices_list = pcd_indices.tolist()
    unique_pcd_indices = set(pcd_indices_list)

    pcd_index_to_batch_indices = {}
    for b, idx in enumerate(pcd_indices_list):
        if idx not in pcd_index_to_batch_indices:
            pcd_index_to_batch_indices[idx] = []
        pcd_index_to_batch_indices[idx].append(b)

    pcd_cache = {}  # Cache to store within_bbox_dict per pcd_index

    for pcd_idx in unique_pcd_indices:
        batch_indices = pcd_index_to_batch_indices[pcd_idx]
        b0 = batch_indices[0]
        input_pcd = batch_input_pcd[b0, :, :]  # shape [N, 3]
        input_mask = batch_input_masks[b0, :, :]  # shape [N, 1]
        original_bbox_dict = batch_original_bbox_dicts[b0]

        # Compute within_bbox_dict for this pcd_index
        within_bbox_dict = _compute_within_bbox(input_pcd, input_mask, original_bbox_dict)

        # Store in cache
        pcd_cache[pcd_idx] = {
            "input_pcd": input_pcd,
            "within_bbox_dict": within_bbox_dict,
            "original_bbox_dict": original_bbox_dict,
        }

    # Now process each batch
    for b in range(B):
        pcd_idx = pcd_indices_list[b]
        cache_entry = pcd_cache[pcd_idx]
        input_pcd = cache_entry["input_pcd"]  # shape [N, 3]
        within_bbox_dict = cache_entry["within_bbox_dict"]
        original_bbox_dict = cache_entry["original_bbox_dict"]
        moved_bbox_dict = batch_moved_bbox_dicts[b]

        # Initialize mask for this batch
        mask = torch.ones(N, dtype=torch.uint8, device=input_pcd.device)

        # Output point cloud for this batch
        output_pcd_b = input_pcd.clone()

        # For each bbox_idx in within_bbox_dict
        for idx, indices in within_bbox_dict.items():
            if idx not in moved_bbox_dict:
                # Mark these points in the mask as 0
                mask[indices] = 0
                continue  # Do not move these points

            # Get transformations
            bbox_data = original_bbox_dict[idx]
            world_to_original_object = np.linalg.inv(bbox_data["object_to_world"])
            world_to_original_object = torch.from_numpy(world_to_original_object).to(
                input_pcd.device, dtype=input_pcd.dtype
            )

            moved_bbox = moved_bbox_dict[idx]
            moved_object_to_world = moved_bbox["object_to_world"]  # numpy array [4,4]
            moved_object_to_world = torch.tensor(moved_object_to_world).to(input_pcd.device, dtype=input_pcd.dtype)

            T = torch.matmul(moved_object_to_world, world_to_original_object)

            # Get points to transform
            points_to_transform = input_pcd[indices, :]  # shape [num_points, 3]
            ones = torch.ones(points_to_transform.shape[0], 1, device=input_pcd.device, dtype=input_pcd.dtype)
            points_homo = torch.cat([points_to_transform, ones], dim=1)  # shape [num_points, 4]

            # Transform points
            points_transformed = torch.matmul(points_homo, T.T)  # shape [num_points, 4]

            # Update output_pcd
            output_pcd_b[indices, :] = points_transformed[:, :3]

        output_pcd[b, :, :] = output_pcd_b
        output_mask[b, :] = mask

    return output_pcd, output_mask


def morphological_closing(mask, kernel_size):
    padding = kernel_size // 2
    # Dilation (mimicking the original closing, here only dilation is applied)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    return dilated


def batch_move_points_within_bboxes_sparse(
    batch_input_pcd,  # list of dense tensors, each tensor of shape [M, 3] (M can vary)
    batch_original_bbox_dicts,  # list of dicts (length B) with original bbox info
    batch_moved_bbox_dicts,  # list of dicts (length B) with moved bbox info
    pcd_indices,  # tensor of shape [B] mapping each batch to a shared point cloud
    dynamic_only,
    bbox_scale,
    print_runtime=False,  # New argument to enable printing runtime of sections.
):
    """
    Optimized processing of a batch of dense point clouds and bounding boxes.
    For each unique point cloud (shared across batches), the function:
      - Precomputes, for each valid bbox, its inverse (world_to_object) and half-extents.
      - Computes which points lie inside each bbox in a fully vectorized manner.
    Then for each batch (which may share the same point cloud) the moved bbox transformations
    are applied to the corresponding points, and only moved points are retained.

    Parameters:
        batch_input_pcd (list): List of dense tensors, each of shape [M, 3] with valid points.
        batch_original_bbox_dicts (list): Length-B list of dictionaries for original bboxes.
        batch_moved_bbox_dicts (list): Length-B list of dictionaries for moved bboxes.
        pcd_indices (torch.Tensor): Tensor of shape [B] mapping each batch to a shared point cloud.
        dynamic_only (bool): Only move points for objects flagged as moving.
        bbox_scale (float): Scale factor for bbox sizes.
        dilate_mask (bool): Ignored here.
        H (int): Grid height (not used in dense version but kept for compatibility).
        W (int): Grid width (not used in dense version but kept for compatibility).
        print_runtime (bool): If True, prints runtime of each section.

    Returns:
        new_batch_input_pcd (list): List of dense tensors containing only the moved points.
    """
    total_start = time.time()

    B = len(batch_input_pcd)
    eps = 1e-6

    # Use the input list directly as batch_points.
    start = time.time()
    batch_points = batch_input_pcd  # each element is a tensor of shape [M, 3]
    if print_runtime:
        print("Using input dense tensors: {:.4f} seconds".format(time.time() - start))

    # Group batches by shared point cloud (pcd_indices).
    start = time.time()
    pcd_indices_list = pcd_indices.tolist()
    unique_pcd_indices = set(pcd_indices_list)
    # Cache per unique pcd: points and computed bbox info.
    pcd_cache = {}
    if print_runtime:
        print("Grouping batches by shared point cloud: {:.4f} seconds".format(time.time() - start))

    # Precompute bbox transformations for each unique point cloud.
    start = time.time()
    for pcd_idx in unique_pcd_indices:
        b0 = pcd_indices_list.index(pcd_idx)  # representative batch for this shared point cloud
        points_b0 = batch_points[b0]  # [M, 3]
        original_bbox_dict = batch_original_bbox_dicts[b0]

        bbox_cache = {}
        within_bbox_dict = {}
        if points_b0.shape[0] > 0:
            ones = torch.ones((points_b0.shape[0], 1), device=points_b0.device, dtype=points_b0.dtype)
            points_homo = torch.cat([points_b0, ones], dim=1)  # [M, 4]
        else:
            points_homo = points_b0

        for bbox_idx, bbox in original_bbox_dict.items():
            if dynamic_only and (not bbox.get("object_is_moving", False)):
                continue
            # Precompute world_to_object (inverse transformation) as a torch tensor.
            world_to_object = torch.from_numpy(np.linalg.inv(bbox["object_to_world"])).to(
                points_b0.device, dtype=points_b0.dtype
            )
            object_lwh = torch.tensor(bbox["object_lwh"], device=points_b0.device, dtype=points_b0.dtype)
            half_lwh = (object_lwh / 2) * bbox_scale

            bbox_cache[bbox_idx] = {"world_to_object": world_to_object, "half_lwh": half_lwh}

            # Compute in a vectorized way which points lie inside this bbox.
            if points_homo.shape[0] > 0:
                points_obj = torch.matmul(points_homo, world_to_object.T)  # [M, 4]
                within_mask = torch.all(torch.abs(points_obj[:, :3]) <= (half_lwh + eps), dim=1)
                indices = torch.nonzero(within_mask, as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
            else:
                indices = torch.empty((0,), device=points_b0.device, dtype=torch.long)
            within_bbox_dict[bbox_idx] = indices

        pcd_cache[pcd_idx] = {
            "points": points_b0,
            "within_bbox_dict": within_bbox_dict,
            "bbox_cache": bbox_cache,
            "original_bbox_dict": original_bbox_dict,
        }
    if print_runtime:
        print("Precomputing bbox transformations: {:.4f} seconds".format(time.time() - start))

    # Process each batch: apply moved bbox transformation and keep only moved points.
    start = time.time()
    updated_points_list = []  # list to store updated dense tensors for each batch
    all_moved_flag = []
    for b in range(B):
        pcd_idx = pcd_indices_list[b]
        cache_entry = pcd_cache[pcd_idx]
        points_base = cache_entry["points"]  # shared for this point cloud
        within_bbox_dict = cache_entry["within_bbox_dict"]
        bbox_cache = cache_entry["bbox_cache"]
        original_bbox_dict = cache_entry["original_bbox_dict"]
        moved_bbox_dict = batch_moved_bbox_dicts[b]

        if points_base.shape[0] == 0:
            updated_points_list.append(points_base)
            continue

        points_out = points_base.clone()
        moved_flag = torch.ones(points_base.shape[0], dtype=torch.bool, device=points_base.device)

        # For each bbox that applies and has a corresponding moved version in this batch.
        for bbox_idx, indices in within_bbox_dict.items():
            if indices.numel() == 0:
                continue  # No points to move for this bbox in this batch
            if bbox_idx not in moved_bbox_dict:
                moved_flag[indices] = False
                continue  # remove those points

            # Get precomputed inverse transform.
            world_to_object = bbox_cache[bbox_idx]["world_to_object"]

            # For the moved bbox, compute T = moved_object_to_world @ world_to_object.
            moved_bbox = moved_bbox_dict[bbox_idx]
            moved_object_to_world = torch.tensor(
                moved_bbox["object_to_world"],
                device=points_out.device,
                dtype=points_out.dtype,
            )
            T = torch.matmul(moved_object_to_world, world_to_object)

            pts = points_out[indices]  # selected points: [num_points, 3]
            ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
            pts_homo = torch.cat([pts, ones], dim=1)
            pts_transformed = torch.matmul(pts_homo, T.T)[:, :3]
            points_out[indices] = pts_transformed
            # # Mark points not in this bbox as not moved.
            # mask = torch.zeros(points_base.shape[0], dtype=torch.bool, device=points_base.device)
            # mask[indices] = True
            # moved_flag = moved_flag & mask

        # Retain only the moved points.
        if moved_flag.sum() > 0:
            new_points = points_out[moved_flag]
        else:
            new_points = torch.empty((0, 3), device=points_out.device, dtype=points_out.dtype)
        all_moved_flag.append(moved_flag)
        updated_points_list.append(new_points)
    if print_runtime:
        print("Processing each batch: {:.4f} seconds".format(time.time() - start))

    if print_runtime:
        print("Total runtime: {:.4f} seconds".format(time.time() - total_start))

    return updated_points_list, all_moved_flag


def move_points_within_bboxes(input_pcd, original_bbox_dict, moved_bbox_dicts):
    """
    For each frame i, find points within each original_bbox in input_pcd[:, i],
    and shift them to the new position specified in moved_bbox_dicts[i].

    Parameters:
        input_pcd (torch.Tensor): Point cloud data of shape [B, f, 3], where
                                  B is the number of points, f is the number of frames.
        original_bbox_dict (dict): Dictionary of original bounding boxes with indices as keys.
        moved_bbox_dicts (list): A list of length f, each element is a dictionary of bounding boxes
                                 corresponding to the moved positions at frame i.

    Returns:
        torch.Tensor: The updated point cloud data with points within the original_bboxes
                      moved to their new positions, of shape [B, f, 3].
    """
    B, f, _ = input_pcd.shape
    output_pcd = input_pcd.clone()  # Create a copy to modify

    # For efficiency, precompute the inverse transformations and half sizes for all original bboxes
    bbox_data = {}
    for idx, bbox in original_bbox_dict.items():
        original_object_to_world = bbox["object_to_world"]  # numpy array [4,4]
        world_to_original_object = np.linalg.inv(original_object_to_world)  # numpy array [4,4]
        world_to_original_object = torch.from_numpy(world_to_original_object).to(
            input_pcd.device, dtype=input_pcd.dtype
        )
        object_lwh = torch.from_numpy(bbox["object_lwh"]).to(input_pcd.device, dtype=input_pcd.dtype)  # shape [3]
        half_lwh = object_lwh / 2  # shape [3]
        bbox_data[idx] = {"world_to_original_object": world_to_original_object, "half_lwh": half_lwh}

    for i in range(f):
        points = input_pcd[:, i, :]  # shape [B, 3]
        B_points = points.shape[0]
        # Convert points to homogeneous coordinates
        ones = torch.ones(B_points, 1, device=input_pcd.device, dtype=input_pcd.dtype)
        points_homo = torch.cat([points, ones], dim=1)  # shape [B, 4]

        for idx in original_bbox_dict.keys():
            # Get precomputed data
            world_to_original_object = bbox_data[idx]["world_to_original_object"]
            half_lwh = bbox_data[idx]["half_lwh"]

            # Transform points to object coordinate system
            points_obj = torch.matmul(points_homo, world_to_original_object.T)  # shape [B, 4]

            # Check which points are within the bbox
            within_bbox = torch.all(
                torch.abs(points_obj[:, :3]) <= (half_lwh + 1e-6), dim=1
            )  # shape [B], boolean tensor

            indices = torch.nonzero(within_bbox).squeeze()  # indices of points within bbox
            if indices.numel() == 0:
                continue  # No points to move for this bbox in this frame

            # Compute transformation matrix from original bbox to moved bbox[i][idx]
            moved_bbox = moved_bbox_dicts[i][idx]  # moved_bbox_dicts[i] is the bbox dict for frame i
            moved_object_to_world = moved_bbox["object_to_world"]  # numpy array [4,4]
            moved_object_to_world = torch.from_numpy(moved_object_to_world).to(input_pcd.device, dtype=input_pcd.dtype)
            T = torch.matmul(
                moved_object_to_world, world_to_original_object
            )  # T = moved_object_to_world @ world_to_original_object

            # For points within bbox, apply transformation
            points_to_transform = points_homo[indices, :]  # shape [N, 4]
            points_transformed = torch.matmul(points_to_transform, T.T)  # shape [N, 4]

            # Update output_pcd
            output_pcd[indices, i, :] = points_transformed[:, :3]

    return output_pcd


def bilinear_splatting_sparse_points_batch(
    pos: torch.Tensor,  # (N, 2): target positions (floating point) for each valid point,
    # with 1 added for a padding offset.
    src_values: torch.Tensor,  # (N, c): source pixel values (e.g. RGB or depth)
    factor: torch.Tensor,  # (N,): per–point weight factors
    batch_idx: torch.Tensor,  # (N,): integer batch index (in 0,...,b-1) for each point
    b: int,  # total number of batches
    target_h: int,
    target_w: int,  # target output image height and width
    channels: int,  # number of channels for the output (e.g. 3 for image)
    center_only: bool = False,
    expanded_kernel: Optional[int] = None,
    kernel_sigma: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    dtype = pos.dtype
    padded_h = target_h + 2
    padded_w = target_w + 2
    warped_frame = torch.zeros((b, padded_h, padded_w, channels), device=device, dtype=src_values.dtype)
    warped_weights = torch.zeros((b, padded_h, padded_w, 1), device=device, dtype=src_values.dtype)

    if center_only:
        pos_rounded = pos.round().long()  # (N,2)
        target_x = pos_rounded[:, 0].clamp(0, padded_w - 1)
        target_y = pos_rounded[:, 1].clamp(0, padded_h - 1)
        flat_idx = batch_idx * (padded_h * padded_w) + (target_y * padded_w + target_x)
        warped_frame_flat = warped_frame.view(-1, channels)
        warped_weights_flat = warped_weights.view(-1, 1)
        contrib = src_values * factor.unsqueeze(1)
        warped_frame_flat.index_add_(0, flat_idx, contrib)
        warped_weights_flat.index_add_(0, flat_idx, factor.unsqueeze(1))
    elif expanded_kernel is not None and expanded_kernel > 2:
        # Instead of expanding over all offsets at once, loop through them to save memory.
        k = expanded_kernel
        half = k // 2
        pos_floor = pos.floor()  # (N,2)

        # Loop over each kernel offset
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                # Create a tensor for the current offset (shape: (2,))
                offset = torch.tensor([dx, dy], device=device, dtype=dtype)
                # Compute the center of the current neighbor (N,2)
                neighbor_center = pos_floor + offset + 0.5
                # Compute the Gaussian weight for this neighbor for each point (N,)
                diff = pos - neighbor_center
                weight = torch.exp(-0.5 * (diff**2).sum(dim=1) / (kernel_sigma**2))
                # Compute the grid coordinate for this neighbor (N,2)
                neighbor_coords = (pos_floor + offset).long()
                target_x = neighbor_coords[:, 0].clamp(0, padded_w - 1)
                target_y = neighbor_coords[:, 1].clamp(0, padded_h - 1)
                # Compute flat index for the scatter operation
                flat_idx = batch_idx * (padded_h * padded_w) + (target_y * padded_w + target_x)
                # Multiply the weight with the per-point factor
                combined_weight = weight * factor  # (N,)
                # Scatter the contributions for both the pixel values and weights.
                warped_frame.view(-1, channels).index_add_(0, flat_idx, src_values * combined_weight.unsqueeze(1))
                warped_weights.view(-1, 1).index_add_(0, flat_idx, combined_weight.unsqueeze(1))
    else:
        # Default bilinear interpolation (4 neighbors)
        x = pos[:, 0]
        y = pos[:, 1]
        x0 = x.floor().long()
        y0 = y.floor().long()
        x1 = x0 + 1
        y1 = y0 + 1
        wa = (x1.to(dtype) - x) * (y1.to(dtype) - y)
        wb = (x1.to(dtype) - x) * (y - y0.to(dtype))
        wc = (x - x0.to(dtype)) * (y1.to(dtype) - y)
        wd = (x - x0.to(dtype)) * (y - y0.to(dtype))

        def scatter_neighbor(x_target: torch.Tensor, y_target: torch.Tensor, weight: torch.Tensor):
            x_target = x_target.clamp(0, padded_w - 1)
            y_target = y_target.clamp(0, padded_h - 1)
            flat_idx = batch_idx * (padded_h * padded_w) + (y_target * padded_w + x_target)
            w = weight * factor
            warped_frame_flat = warped_frame.view(-1, channels)
            warped_weights_flat = warped_weights.view(-1, 1)
            contrib = src_values * w.unsqueeze(1)
            warped_frame_flat.index_add_(0, flat_idx, contrib)
            warped_weights_flat.index_add_(0, flat_idx, w.unsqueeze(1))

        scatter_neighbor(x0, y0, wa)
        scatter_neighbor(x0, y1, wb)
        scatter_neighbor(x1, y0, wc)
        scatter_neighbor(x1, y1, wd)

    # Crop out the 1-pixel padding.
    warped_frame = warped_frame[:, 1:-1, 1:-1, :]
    warped_weights = warped_weights[:, 1:-1, 1:-1, :]
    return warped_frame, warped_weights


def forward_warp_multiframes_sparse_depth_only(
    transformation2: torch.Tensor,  # (b, 4, 4)
    intrinsic2: Optional[torch.Tensor],  # (b, 3, 3) or (b, 11) for fθ cameras
    buffer_points: torch.Tensor,  # (N, 3): corresponding world points for each valid pixel
    buffer_length: torch.Tensor,  # (b*v,): lengths to split the sparse buffers per view
    target_h: int,
    target_w: int,  # target output image height and width
    center_only: bool = False,
    is_ftheta: bool = False,
    expanded_kernel: Optional[int] = None,
    kernel_sigma: float = 1.0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    device = buffer_points.device
    dtype = buffer_points.dtype
    b = transformation2.shape[0]
    total_views = buffer_length.numel()  # b*v
    v = total_views // b

    # Create a per–point view index (0,...,b*v-1) by repeating each view index
    view_idx = torch.repeat_interleave(torch.arange(total_views, device=device), buffer_length)
    # Determine the batch index for each point (each view belongs to a batch; all views in a batch share the same transformation)
    batch_idx = view_idx // v  # (N,)

    # For each point, select the corresponding transformation and intrinsic matrices.
    w2c_all = transformation2[batch_idx]  # (N, 4, 4)
    intrinsic_all = intrinsic2[batch_idx]  # (N, 6) for pinhole or (N, 11) for fθ cameras

    # Project all world points at once.
    proj = project_points_sparse_batch(buffer_points, w2c_all, intrinsic_all, is_ftheta=is_ftheta)  # (N, 3)
    valid_mask = proj[:, 2] > 0
    if valid_mask.sum() == 0:
        out_mask = torch.zeros((b, target_h, target_w, 1), device=device, dtype=dtype)
        out_depth = torch.zeros((b, target_h, target_w, 1), device=device, dtype=dtype)
        return out_depth.permute(0, 3, 1, 2), out_mask.permute(0, 3, 1, 2)

    # Filter only valid points.
    proj_valid = proj[valid_mask]  # (N_valid, 3)
    batch_idx_valid = batch_idx[valid_mask]  # (N_valid,)

    # Compute target image coordinates from the projection (divide by depth)
    trans_coords = proj_valid[:, :2] / (proj_valid[:, 2:3] + 1e-7)  # (N_valid, 2)
    pos = trans_coords + 1.0  # add a 1–pixel padding offset

    # Compute a weight factor based on depth (as a proxy for uncertainty)
    sat_depth = proj_valid[:, 2].clamp(min=0, max=1000)
    log_depth = torch.log1p(sat_depth)
    max_log_depth = log_depth.max() if log_depth.numel() > 0 else 1.0
    depth_factor = torch.exp(log_depth / (max_log_depth + 1e-7) * 50)
    factor = 1.0 / (depth_factor + 1e-7)

    depth_valid = proj_valid[:, 2:3]  # (N_valid, 1)
    accum_depth, weight_depth = bilinear_splatting_sparse_points_batch(
        pos,
        depth_valid,
        factor,
        batch_idx_valid,
        b,
        target_h,
        target_w,
        1,
        center_only=center_only,
        expanded_kernel=expanded_kernel,
        kernel_sigma=kernel_sigma,
    )

    mask_img = (weight_depth > 0).float()
    final_depth = torch.where(weight_depth > 0, accum_depth / weight_depth, torch.tensor(0, device=device, dtype=dtype))

    mask_img = mask_img.permute(0, 3, 1, 2).contiguous()
    final_depth = final_depth.permute(0, 3, 1, 2).contiguous()

    return final_depth, mask_img


def project_points_sparse_batch(
    points: torch.Tensor,  # (N, 3): world points for each valid pixel
    w2c: torch.Tensor,  # (N, 4, 4): per–point world-to–camera matrices
    intrinsic: torch.Tensor,  # (N, 6) for pinhole or (N, 11) for fθ cameras
    is_ftheta: bool = False,
) -> torch.Tensor:
    """
    Projects N world points (N,3) into image space in a fully vectorized manner.
    Returns a tensor of shape (N, 3) where the first two elements are the projected
    (homogeneous) image coordinates and the third element is either the third element
    from the pinhole projection or a constant 1 for fθ cameras.
    """
    device = points.device
    dtype = points.dtype
    ones = torch.ones((points.shape[0], 1), device=device, dtype=dtype)
    points_homo = torch.cat([points, ones], dim=1)  # (N, 4)
    points_homo = points_homo.unsqueeze(-1)  # (N, 4, 1)
    camera_points_homo = torch.bmm(w2c, points_homo)  # (N, 4, 1)
    cam_coords = camera_points_homo.squeeze(-1)  # (N, 4)
    cam_coords = cam_coords[:, :3]  # (N, 3)

    if not is_ftheta:
        # For pinhole cameras, intrinsic is (N, 6) representing [fx, fy, cx, cy, width, height]
        # we convert to (N, 3, 3)
        intrinsic_matrix = torch.zeros((intrinsic.shape[0], 3, 3)).to(intrinsic)
        intrinsic_matrix[:, 0, 0] = intrinsic[:, 0]
        intrinsic_matrix[:, 1, 1] = intrinsic[:, 1]
        intrinsic_matrix[:, 0, 2] = intrinsic[:, 2]
        intrinsic_matrix[:, 1, 2] = intrinsic[:, 3]
        intrinsic_matrix[:, 2, 2] = 1.0
        proj = torch.bmm(intrinsic_matrix, cam_coords.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        return proj
    else:
        # For fθ cameras, intrinsic is (N, 11) in the format:
        # [cx, cy, width, height, poly_coeffs..., is_bw_poly]
        # First, compute common geometric quantities.
        cx = intrinsic[:, 0]  # (N,)
        cy = intrinsic[:, 1]  # (N,)
        x_cam = cam_coords[:, 0]
        y_cam = cam_coords[:, 1]
        z_cam = cam_coords[:, 2]
        xy_norm = torch.sqrt(x_cam**2 + y_cam**2 + 1e-8)
        ray_norm = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2 + 1e-8)
        cos_alpha = torch.clamp(z_cam / ray_norm, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)

        # Prepare the output (N, 3)
        proj_all = torch.empty((points.shape[0], 3), device=device, dtype=dtype)

        # Group points by unique intrinsic parameters for efficiency.
        unique_intrinsics, inv = torch.unique(intrinsic, return_inverse=True, dim=0)
        for i in range(unique_intrinsics.shape[0]):
            mask = inv == i
            if mask.sum() == 0:
                continue
            intrinsic_i = unique_intrinsics[i]
            cx_i = intrinsic_i[0]
            cy_i = intrinsic_i[1]
            width_i = intrinsic_i[2]
            height_i = intrinsic_i[3]
            # Assume poly coefficients are stored in indices 4:-1.
            bw_poly = intrinsic_i[4:-1]

            # Compute the forward polynomial function and the cutoff parameters.
            # (This function must behave identically to the one used in your first function.)
            fw_poly, max_ray_angle, max_ray_distortion = compute_fw_poly(bw_poly, width_i, height_i, (cx_i, cy_i))

            alpha_group = alpha[mask]
            delta = torch.empty_like(alpha_group)
            cond = alpha_group <= max_ray_angle
            if cond.any():
                delta[cond] = fw_poly(alpha_group[cond])
            if (~cond).any():
                delta[~cond] = max_ray_distortion[0] + (alpha_group[~cond] - max_ray_angle) * max_ray_distortion[1]
            # For points with almost zero xy_norm, set delta to 0.
            small_mask = xy_norm[mask] < 1e-6
            delta[small_mask] = 0.0
            scale = delta / (xy_norm[mask] + 1e-8)
            u = x_cam[mask] * scale + cx_i
            v = y_cam[mask] * scale + cy_i
            proj_all[mask] = torch.stack(
                [u * z_cam[mask], v * z_cam[mask], z_cam[mask]], dim=1
            )  # multiply by depth just to make it consistent with the other case

        return proj_all


def compute_fw_poly(
    bw_poly: torch.Tensor, width: torch.Tensor, height: torch.Tensor, center: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[callable, torch.Tensor, torch.Tensor]:
    width_val = width.item() if isinstance(width, torch.Tensor) else float(width)
    height_val = height.item() if isinstance(height, torch.Tensor) else float(height)
    cx = center[0].item() if isinstance(center[0], torch.Tensor) else float(center[0])
    cy = center[1].item() if isinstance(center[1], torch.Tensor) else float(center[1])

    def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
        center_component = 0.5 * image_size_component
        if principal_point_component > center_component:
            return principal_point_component
        else:
            return image_size_component - principal_point_component

    max_dist_x = compute_max_distance_to_border(width_val, cx)
    max_dist_y = compute_max_distance_to_border(height_val, cy)
    max_radius = np.sqrt(max_dist_x**2 + max_dist_y**2)

    bw_poly_np = bw_poly.cpu().numpy() if bw_poly.device.type != "cpu" else bw_poly.numpy()
    inv_coeffs = approximate_polynomial_inverse(bw_poly_np, 0.0, max_radius)
    fw_coeffs = [0.0] + [float(c) for c in inv_coeffs[1:]]

    def fw_poly(alpha: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the forward polynomial at angles alpha.
        Computes: delta = sum_{i=0}^{n} fw_coeffs[i] * alpha^i.
        """
        result = torch.zeros_like(alpha, dtype=bw_poly.dtype, device=bw_poly.device)
        for i, coeff in enumerate(fw_coeffs):
            result = result + (coeff * (alpha**i))
        return result

    MAX_ANGLE = np.pi
    DOMAIN_SAMPLE_COUNT = 2000
    STEP_SIZE = MAX_ANGLE / DOMAIN_SAMPLE_COUNT

    def eval_polynomial(x: float, coeffs: list) -> float:
        result = 0.0
        for c in reversed(coeffs):
            result = result * x + c
        return result

    def eval_polynomial_derivative(x: float, coeffs: list) -> float:
        result = 0.0
        n = len(coeffs)
        for i, c in enumerate(reversed(coeffs[:-1])):
            result = result * x + c * (n - 1 - i)
        return result

    max_ray_angle_val = 0.0
    for i in range(1, DOMAIN_SAMPLE_COUNT + 1):
        next_angle = i * STEP_SIZE
        next_r = eval_polynomial(next_angle, fw_coeffs)
        d_next_r = eval_polynomial_derivative(next_angle, fw_coeffs)
        if d_next_r <= 0.0:
            break
        max_ray_angle_val = next_angle
        if next_r > max_radius:
            break

    max_ray_angle_tensor = torch.tensor(max_ray_angle_val, dtype=bw_poly.dtype, device=bw_poly.device)

    def poly_derivative(x: float, coeffs: list) -> float:
        deriv = 0.0
        for i in range(1, len(coeffs)):
            deriv += i * coeffs[i] * (x ** (i - 1))
        return deriv

    max_val = eval_polynomial(max_ray_angle_val, fw_coeffs)
    d_val = poly_derivative(max_ray_angle_val, fw_coeffs)
    max_ray_distortion = torch.tensor([max_val, d_val], dtype=bw_poly.dtype, device=bw_poly.device)

    return fw_poly, max_ray_angle_tensor, max_ray_distortion


def approximate_polynomial_inverse(coeffs: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    """
    Computes an approximate inverse polynomial in the provided range, fixing the first coefficient to zero.

    The fitting is performed via least squares on inverted sampled points (y, x) of the provided polynomial.

    Parameters:
    -----------
    coeffs : np.ndarray
        Coefficients of the polynomial to be inverted.
    range_min : float
        Minimum value of the range for sampling.
    range_max : float
        Maximum value of the range for sampling.

    Returns:
    --------
    np.ndarray
        Coefficients of the approximate inverse polynomial.

    Raises:
    -------
    ValueError
        If the polynomial degree is not supported (i.e., not 4 or 5).
    """

    def f4(x, b, x1, x2, x3, x4):
        """4th degree polynomial."""
        return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))

    def f5(x, b, x1, x2, x3, x4, x5):
        """5th degree polynomial."""
        return b + x * (x1 + x * (x2 + x * (x3 + x * (x4 + x * x5))))

    SAMPLE_COUNT = 500
    samples_y = np.linspace(range_min, range_max, SAMPLE_COUNT)
    samples_x = eval_polynomial(samples_y, coeffs.astype(np.float64))  # use high-precision estimation

    if (poly_degree := len(coeffs) - 1) == 4:
        bounds = (
            [0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
        )
        inv_coeffs, _ = curve_fit(f4, samples_x, samples_y, bounds=bounds)
    elif poly_degree == 5:
        bounds = (
            [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf, np.inf],
        )
        inv_coeffs, _ = curve_fit(f5, samples_x, samples_y, bounds=bounds)
    else:
        raise ValueError(f"Unsupported polynomial degree {poly_degree}")

    return np.array([np.float32(val) if i > 0 else 0 for i, val in enumerate(inv_coeffs)], dtype=np.float32)


def eval_polynomial(xs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluates polynomial coeffs [D,] at given points [N,] using Horner scheme"""
    ret = np.zeros(len(xs), dtype=xs.dtype)

    for coeff in reversed(coeffs):
        ret = ret * xs + coeff

    return ret


def refine_triangle_mesh_3d(triangles, max_edge_length):
    """Conformingly subdivide a 3D triangle mesh to bound every edge length.

    Midpoints are created once per unique mesh edge and reused by all incident
    triangles. This avoids T-junctions while preserving the input mesh's exact
    piecewise-planar surface: every new vertex is an existing edge midpoint.

    Args:
        triangles: np.ndarray, shape (M, 3, 3).
        max_edge_length: positive finite edge-length limit in world units.

    Returns:
        np.ndarray, shape (N, 3, 3), with every edge no longer than the limit.
    """
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape (M, 3, 3), got {triangles.shape}")
    if not np.isfinite(max_edge_length) or max_edge_length <= 0:
        raise ValueError("max_edge_length must be positive and finite")
    if not np.all(np.isfinite(triangles)):
        raise ValueError("triangles must contain only finite coordinates")
    if len(triangles) == 0:
        return triangles.copy()

    # Shapely emits duplicate coordinate rows for shared triangle vertices.
    # Deduplicate at sub-nanometer precision so the refinement operates on an
    # indexed mesh and can split each shared edge exactly once.
    flat_vertices = triangles.reshape(-1, 3)
    rounded_vertices = np.round(flat_vertices, decimals=12)
    _, unique_indices, inverse = np.unique(rounded_vertices, axis=0, return_index=True, return_inverse=True)
    vertices = flat_vertices[unique_indices]
    faces = inverse.reshape(-1, 3)
    edge_positions = ((0, 1), (1, 2), (2, 0))
    tolerance = max_edge_length * 1e-12

    for _ in range(64):
        mesh_edges = {tuple(sorted((int(face[a]), int(face[b])))) for face in faces for a, b in edge_positions}
        split_edges = {
            edge
            for edge in mesh_edges
            if np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]) > max_edge_length + tolerance
        }
        if not split_edges:
            return vertices[faces]

        new_vertices = vertices.tolist()
        midpoint_indices = {}
        for edge in sorted(split_edges):
            midpoint_indices[edge] = len(new_vertices)
            new_vertices.append((0.5 * (vertices[edge[0]] + vertices[edge[1]])).tolist())

        new_faces = []
        for face in faces:
            face_edges = [tuple(sorted((int(face[a]), int(face[b])))) for a, b in edge_positions]
            marked_indices = [index for index, edge in enumerate(face_edges) if edge in split_edges]
            if not marked_indices:
                new_faces.append(face.tolist())
                continue
            if len(marked_indices) == 1:
                edge_index = marked_indices[0]
                a, b = edge_positions[edge_index]
                c = 3 - a - b
                midpoint = midpoint_indices[face_edges[edge_index]]
                new_faces.extend(
                    (
                        [int(face[a]), midpoint, int(face[c])],
                        [midpoint, int(face[b]), int(face[c])],
                    )
                )
                continue
            if len(marked_indices) == 2:
                unmarked_edge_index = next(index for index in range(3) if index not in marked_indices)
                b, c = edge_positions[unmarked_edge_index]
                a = 3 - b - c
                edge_ab = tuple(sorted((int(face[a]), int(face[b]))))
                edge_ca = tuple(sorted((int(face[c]), int(face[a]))))
                midpoint_ab = midpoint_indices[edge_ab]
                midpoint_ca = midpoint_indices[edge_ca]
                va, vb, vc = int(face[a]), int(face[b]), int(face[c])

                new_faces.append([va, midpoint_ab, midpoint_ca])
                diagonal_midpoint_ab_to_c = np.linalg.norm(np.asarray(new_vertices[midpoint_ab]) - vertices[vc])
                diagonal_b_to_midpoint_ca = np.linalg.norm(vertices[vb] - np.asarray(new_vertices[midpoint_ca]))
                if diagonal_midpoint_ab_to_c <= diagonal_b_to_midpoint_ca:
                    new_faces.extend(
                        (
                            [midpoint_ab, vb, vc],
                            [midpoint_ab, vc, midpoint_ca],
                        )
                    )
                else:
                    new_faces.extend(
                        (
                            [midpoint_ab, vb, midpoint_ca],
                            [vb, vc, midpoint_ca],
                        )
                    )
                continue
            if len(marked_indices) != 3:
                raise RuntimeError("invalid triangle refinement state")

            midpoint_01, midpoint_12, midpoint_20 = (midpoint_indices[edge] for edge in face_edges)
            v0, v1, v2 = (int(index) for index in face)
            new_faces.extend(
                (
                    [v0, midpoint_01, midpoint_20],
                    [midpoint_01, v1, midpoint_12],
                    [midpoint_20, midpoint_12, v2],
                    [midpoint_01, midpoint_12, midpoint_20],
                )
            )

        vertices = np.asarray(new_vertices, dtype=float)
        faces = np.asarray(new_faces, dtype=np.int64)

    raise RuntimeError("triangle refinement did not converge within 64 iterations")


def triangulate_polygon_3d(vertices3d):
    """
    Triangulate a 3D polygon using Shapely's constrained Delaunay triangulation
    over the bird's-eye plane (x, y), preserving z-coordinates from original vertices.

    Args:
        vertices3d: np.ndarray, shape (N, 3), polygon vertices in world coordinates.
                    If the polygon is closed (first vertex equals last), the last
                    duplicate vertex will be removed for triangulation.

    Returns:
        np.ndarray, shape (M, 3, 3), triangulated faces in 3D (x,y,z)
    """
    # Input validation and conversion
    verts = np.asarray(vertices3d, dtype=float)

    # Remove duplicate last vertex if polygon is closed
    if verts.shape[0] >= 2 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]

    if verts.shape[0] < 3:
        return np.empty((0, 3, 3), dtype=float)

    # Create and validate 2D polygon
    polygon = Polygon(verts[:, :2])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    # Perform constrained Delaunay triangulation
    triangles_collection = constrained_delaunay_triangles(polygon)

    # Extract triangle coordinates more efficiently
    coords_list = _extract_triangle_coordinates(triangles_collection)
    if not coords_list:
        return np.empty((0, 3, 3), dtype=float)

    # Convert to array in one operation
    all_coords = np.vstack(coords_list).reshape(-1, 3, 2)  # Shape: (num_triangles, 3, 2)
    num_triangles = all_coords.shape[0]

    # Build KDTree once
    kdtree = KDTree(verts[:, :2])

    # Process all coordinates with single query (get enough neighbors for both exact match and interpolation)
    coords_flat = all_coords.reshape(-1, 2)  # Shape: (num_triangles * 3, 2)
    max_k = min(6, len(verts))  # Get more neighbors to handle both cases efficiently

    distances, indices = kdtree.query(coords_flat, k=max_k)
    if max_k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Pre-allocate and fill result array efficiently
    triangles_3d = np.empty((num_triangles, 3, 3), dtype=float)
    result_flat = triangles_3d.reshape(-1, 3)

    # Vectorized processing with single pass
    _fill_triangle_vertices(result_flat, coords_flat, verts, distances, indices)

    return triangles_3d


def _extract_triangle_coordinates(triangles_collection):
    """Extract triangle coordinates efficiently from Shapely collection."""
    coords_list = []

    if hasattr(triangles_collection, "geoms") and triangles_collection.geoms is not None:
        for geom in triangles_collection.geoms:
            if geom.geom_type == "Polygon":
                coords = np.array(geom.exterior.coords[:-1])
                if len(coords) == 3:
                    coords_list.append(coords)
    elif triangles_collection.geom_type == "Polygon":
        coords = np.array(triangles_collection.exterior.coords[:-1])
        if len(coords) == 3:
            coords_list.append(coords)

    return coords_list


def _fill_triangle_vertices(result_flat, coords_flat, verts, distances, indices, tolerance=1e-10):
    """Fill result array with vertices using vectorized operations."""
    # Check for exact matches
    exact_matches = distances[:, 0] < tolerance

    # For exact matches, use original vertices directly
    if np.any(exact_matches):
        exact_indices = indices[exact_matches, 0]
        result_flat[exact_matches] = verts[exact_indices]

    # For points needing interpolation
    interp_mask = ~exact_matches
    if np.any(interp_mask):
        # Use pre-computed distances and indices for interpolation
        interp_distances = distances[interp_mask]
        interp_indices = indices[interp_mask]
        interp_coords = coords_flat[interp_mask]

        # Vectorized inverse distance weighting
        safe_distances = np.maximum(interp_distances, 1e-10)
        weights = 1.0 / safe_distances
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Compute interpolated z-coordinates
        z_values = verts[interp_indices, 2]
        interp_z = np.sum(weights * z_values, axis=1)

        # Assign interpolated coordinates
        result_flat[interp_mask, :2] = interp_coords
        result_flat[interp_mask, 2] = interp_z


def filter_by_height_relative_to_ego(
    geometry: np.ndarray,
    camera_model,
    camera_pose: np.ndarray,
    camera_pose_init: np.ndarray,
    underpass_threshold: float = -3.0,
    envelope_angle_deg: float = 3.0,
    min_overlap_percentage: float = 0.8,
) -> bool:
    """
    Filter out map elements that are on underpasses based on height relative to ego.
    Also filters based on overlap with a road surface envelope for horizontal elements.

    Args:
        geometry: np.ndarray
            Shape (N, 3). 3D points of the map element in world coordinates.
        camera_model: CameraModel
            Camera model (not used but kept for consistency).
        camera_pose: np.ndarray
            Shape (4, 4). Camera-to-world transformation matrix.
        camera_pose_init: np.ndarray
            Shape (4, 4). Camera-to-world transformation matrix at the start of the clip.
            This is the camera pose in ego space.
        underpass_threshold: float
            If maximum height is below this threshold, element is considered underpass.
        envelope_angle_deg: float
            Road surface envelope angle in degrees (±angle from horizontal).
        min_overlap_percentage: float
            Minimum percentage of points that must be within road surface envelope.

    Returns:
        bool: True if the element should be filtered out (is overpass/underpass/outside envelope), False otherwise
    """
    if len(geometry) == 0:
        return True  # Filter out empty geometry

    # Transform points from world coordinates to camera coordinates
    world_to_camera = np.linalg.inv(camera_pose)
    points_in_cam = camera_model.transform_points_np(geometry, world_to_camera)
    points_in_ego_space = camera_model.transform_points_np(points_in_cam, camera_pose_init)
    # In ego space: x (front), y (left), z (up)

    # Get height in ego space
    heights_in_ego_space = points_in_ego_space[:, 2]

    # Check for underpass: if maximum height is above threshold
    max_height = np.max(heights_in_ego_space)
    if max_height > underpass_threshold:
        # Not underpass
        return False

    # Additional check: road surface envelope for horizontal elements (like wait lines)
    # Check what percentage of points fall within the road surface envelope
    if envelope_angle_deg > 0 and min_overlap_percentage > 0:
        # Get distances (x, y) and heights (z) in ego space
        distances = points_in_ego_space[:, :2]
        heights = points_in_ego_space[:, 2]
        distance_norm = np.linalg.norm(distances, axis=1)

        # Calculate the road surface envelope bounds for each point based on its forward distance
        angle_rad = np.deg2rad(envelope_angle_deg)
        tan_angle = np.tan(angle_rad)

        # Calculate the z-range that falls within the road surface envelope
        # Lower bound: z = -distance * tan(angle)
        envelope_lower = -distance_norm * tan_angle

        # Check which points fall within the envelope
        within_envelope = heights >= envelope_lower
        overlap_percentage = np.sum(within_envelope) / len(heights)

        # Filter out if overlap is below threshold
        if overlap_percentage > min_overlap_percentage:
            return False  # Keep the element

    return True  # Filter out due to insufficient overlap with road surface envelope
