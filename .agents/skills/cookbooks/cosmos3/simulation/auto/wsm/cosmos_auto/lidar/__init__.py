# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LiDAR range-map rendering for RDS-HQ clips.

Ported from the Cosmos-Drive-Dreams toolkit
(``cosmos-drive-dreams-toolkits/convert_lidar_pointcloud_to_rangemap.py`` and
``utils/lidar_*.py``) with two changes:

* The NCORE and torch-scatter dependencies are replaced by a self-contained
  spinning-lidar model (:mod:`cosmos_auto.lidar.sensor_model`,
  :mod:`cosmos_auto.lidar.shutter`) and native ``torch.scatter_reduce``.
* Sweep poses for the HD-map range map (the Cosmos LiDAR control signal) are
  built by default from the RDS-HQ ``pose`` / ``vehicle_pose`` components plus
  the rig LiDAR extrinsic (:mod:`cosmos_auto.lidar.pose_source`), so the control
  signal can be produced for clips without LiDAR and for novel trajectories.
  The recorded ``lidar_raw`` sweep poses remain available as
  ``pose_source="lidar_raw"`` for comparison with the original toolkit.

Submodules import lazily so ``import cosmos_auto.lidar`` stays cheap.
"""
