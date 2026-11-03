from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraSpec:
    camera_list: List[str] = field(default_factory=list)
    camera_to_rds_hq: Optional[Dict[str, str]] = field(default_factory=dict)


@dataclass
class InputFpsSpec:
    input_pose_fps: int = field(default=30)  # fps in pose folder
    input_lidar_fps: int = field(default=10)  # fps of lidar data
    gt_video_fps: int = field(default=30)  # fps of gt video


@dataclass
class ResolutionSpec:
    mode: str = field(default="fixed")  # fixed or source (match native camera intrinsics)
    camera_model_resolution: List[int] = field(default_factory=list)  # fixed-mode camera model resolution
    cosmos_resolution: List[int] = field(default_factory=list)  # fixed-mode output resolution
    to_cosmos_resolution: str = field(default="resize")  # fixed-mode resize or crop operation


@dataclass
class OutputFrameSpec:
    target_render_fps: int = field(default=10)  # target render fps
    target_chunk_frame: int = field(default=100)  # frame number in target render fps
    overlap_frame: int = field(default=45)  # overlap frame number in target render fps
    max_chunk: int = field(default=3)  # max chunk number


@dataclass
class S3Spec:
    enable_upload: bool = field(default=False)
    s3_path: Optional[str] = field(default=None)
    cred_path_or_profile: Optional[str] = field(default=None)


@dataclass
class MinimapSpec:
    minimap_types: List[str] = field(default_factory=list)  # map name in the rig's convention


@dataclass
class CosmosAVConfig:
    input_root: str
    output_root: str

    camera_spec: CameraSpec = field(default_factory=CameraSpec)
    input_fps_spec: InputFpsSpec = field(default_factory=InputFpsSpec)
    resolution_spec: ResolutionSpec = field(default_factory=ResolutionSpec)
    output_frame_spec: OutputFrameSpec = field(default_factory=OutputFrameSpec)
    s3_spec: S3Spec = field(default_factory=S3Spec)
    minimap_spec: MinimapSpec = field(default_factory=MinimapSpec)

    clip_id_json: Optional[str] = field(default=None)
    camera_type: str = field(default="ftheta")
    post_training: bool = field(default=False)
    novel_pose_folder: Optional[str] = field(default=None)
    version: str = field(default="v4")
    skip: List[str] = field(default_factory=list)
    skip_layers: List[str] = field(
        default_factory=list
    )  # v3 only: minimap, lanelines, traffic_lights, bbox, or minimap sub-type
    bbox_object_types: List[str] = field(default_factory=list)  # v3 only: empty = all types
    bbox_fill_face: str = field(default="all")  # v3 only: none/all/front/front_and_back
    bbox_compare: bool = field(default=False)  # v3 only: overlay old straight edges in red
    # 1-10 subdivision quality for fisheye distortion. Used by v3 polygon
    # layers and WMv4 road-marking footprint polygons; 1=straight, 10=max.
    curve_subdivide: int = field(default=10)
    # WMv3/WMv4: attenuate configured RGB colors by camera-forward depth.
    # This controls color only; distance-based line-width scaling remains on.
    depth_gradient: bool = field(default=True)
    manual_split: Optional[str] = field(default=None)
    skip_resume: bool = field(default=False)
    # When a clip has fewer frames than output_frame_spec.target_chunk_frame,
    # emit one shorter chunk containing every available frame instead of
    # silently producing no MP4. Disabled by default to preserve existing
    # fixed-length dataset behavior.
    allow_short_final_chunk: bool = field(default=False)
    # WMv4: Tier-1 sign category keys (as returned by _classify_sign, e.g.
    # "stop", "yield", "do_not_enter", "one_way", "pedestrian_crossing")
    # rendered as single-sided — front artwork + a dedicated back-color face.
    # Categories not listed render double-sided (front polygon only, visible
    # from any angle). Empty by default, so every sign renders double-sided
    # unless categories are listed here — rendering a single-sided sign as
    # double-sided is less damaging than the reverse. Matched case-insensitively.
    # Has no effect when version is not "v4".
    single_sided_sign_categories: List[str] = field(default_factory=list)
    # WMv4: subfolder of input_root holding per-frame traffic-light states.
    # "3d_traffic_lights_status" is the legacy red/yellow/green/unknown source;
    # "3d_traffic_lights_lightnet_status" is the richer LightNet v25 taxonomy
    # (per-direction arrows etc.), which is collapsed to the same palette colors
    # for the face. Has no effect when version is not "v4".
    traffic_light_status_dir: str = field(default="3d_traffic_lights_status")
    # WMv4: color recognized painted lane arrows (forward/left/right,
    # left_forward/right_forward, and merge_left/merge_right) using
    # config_color_road_marking.json instead of the generic road-markings color.
    # Non-arrow road markings keep the generic color. No effect when version != "v4".
    use_road_marking_arrow_colors: bool = field(default=True)
    # WMv4: draw an oriented polygon arrow glyph in place of each turn-arrow's
    # rectangular footprint (Option 2). The glyph shape comes from
    # assets/road_marking_arrows/arrow_shape.json. Off by default; when on it
    # composes with the configured colors above. No effect when version != "v4".
    use_road_marking_arrow_shapes: bool = field(default=False)
    # WMv4: scale multiplier applied after fitting the arrow glyph to each
    # marking's exported footprint. null -> use size_scale from the arrow asset.
    # Only affects glyphs (use_road_marking_arrow_shapes). No effect when version != "v4".
    road_marking_arrow_size_scale: Optional[float] = field(default=None)
    # WMv4: render stop / keep_clear exported footprint polygons with their
    # category colors from config_color_road_marking.json instead of the generic
    # road_markings color. No effect when version != "v4".
    use_road_marking_text_colors: bool = field(default=True)
    # WMv4: conformingly subdivide KEEP_CLEAR and TEXT_PAINT footprint-fill
    # triangles whose world-space edges exceed the configured limit. Glyph
    # meshes and every other road-marking class are unchanged. Disabled by default.
    use_road_marking_interior_tessellation: bool = field(default=False)
    road_marking_interior_tessellation_max_edge_m: float = field(default=1.0)
    # WMv4/WMv4.1: render solid road boundaries colored by boundary_type.
    # Disable to restore the legacy single-color road-boundary layer.
    use_road_boundary_colors: bool = field(default=True)
    # WMv4/WMv4.1: suppress static HD-map geometry below the current driven
    # road level using smoothed pose planes. Geometry above the road is retained.
    filter_below_ego_level: bool = field(default=True)
    # ---------------------------------------------------------------------------
    # WMv4.1: static occupancy voxels. v4.1 renders everything v4.0 does (signs,
    # traffic lights, road markings, lanelines, minimap, bboxes) PLUS GPU-instanced
    # occupancy voxel towers loaded from input_root/3d_occupancy_static/<clip>.tar
    # (produced by convert_static_occupancy_to_rds_hq.py). All fields below have no
    # effect unless version == "v4.1".
    # ---------------------------------------------------------------------------
    # Maximum forward distance (meters) for rendered occupancy voxels per frame;
    # towers beyond this are culled on the GPU. null = no forward culling.
    voxel_max_distance: Optional[float] = field(default=80.0)
    # Maximum lateral distance (meters, +/-) for rendered occupancy voxels per
    # frame; towers beyond this are culled on the GPU. null = no lateral culling.
    voxel_max_lateral: Optional[float] = field(default=28.0)
    # Maximum height (meters) for occupancy voxel towers; taller towers are clamped.
    # null = no clamp.
    voxel_max_height: Optional[float] = field(default=2.0)
    # RGB color [0-255] for occupancy voxels (classification render mode).
    # null falls back to occupancy_utils.STATIC_OCCUPANCY_COLOR_RGB (orange).
    voxel_color: Optional[List[int]] = field(default_factory=lambda: [180, 180, 180])
    # Ground-elevation method for placing voxel tower bases:
    #   "none"         -> bases at z=0 (assumes flat ground)
    #   "base"         -> per-column base z read from the occupancy data 'base' field
    #                    (recommended; the converter writes this)
    #   "hdmap"        -> estimate ground from HD-map lanelines/road-boundary polylines
    #   "tilted_plane" -> per-frame ground plane derived from the camera trajectory
    voxel_ground_elevation: str = field(default="base")
    # Half-width (frames) of the sliding window used to smooth the tilted_plane
    # ground estimate. Only used when voxel_ground_elevation == "tilted_plane".
    voxel_ground_smooth_window: int = field(default=10)
    # WMv4.1 training: place each voxel tower at the true LiDAR-derived 3D ground
    # location stored in the occupancy data's 'base' field, instead of estimating
    # ground from lane geometry or the camera trajectory.  This bypasses
    # voxel_ground_elevation and disables per-frame tilted-plane adjustments, which
    # are only needed when ground elevation must be estimated.  For training we have
    # accurate RDS-HQ LiDAR data, so 3D bases are more reliable than any estimate —
    # particularly in multi-level scenes (over/underpasses) where a single estimated
    # plane cannot represent both levels simultaneously.
    # No effect when version != "v4.1".
    use_3d_bases: bool = field(default=False)
    # WMv4.1: give WMv4.0-and-earlier elements visual priority over occupancy
    # voxels by DELETING any voxel column whose 3D extent overlaps a priority
    # element.  Realistic occlusion is preserved everywhere else (voxels still
    # occlude when genuinely in front).  Static elements handled here: traffic
    # signs, traffic lights, poles; dynamic bounding boxes are culled per-frame.
    # No effect when version != "v4.1".
    voxel_cull_world_model: bool = field(default=True)
    # Extra clearance (metres) applied around every priority element when testing
    # voxel overlap for voxel_cull_world_model.
    voxel_cull_margin: float = field(default=0.2)
    # WMv4.1: draw edges around each voxel so towers read as stacked cubes rather
    # than solid blobs.  No effect when version != "v4.1".
    voxel_edges: bool = field(default=True)
    # RGB [0-255] for the voxel cube edges (default black).
    voxel_edge_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    # Cube-edge line width (pixels, distance-attenuated like other line geometry).
    voxel_edge_width: float = field(default=0.5)
    # WMv4.1: split each column into its stacked unit cubes so the edge pass draws
    # horizontal cube-boundary lines at every voxel level (explicit height/depth
    # cues) instead of one tall unbroken box.  Requires voxel_edges to be visible.
    # No effect when version != "v4.1".
    voxel_stacked_cubes: bool = field(default=True)
