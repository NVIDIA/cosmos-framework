# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Row-offset spinning LiDAR model (Hesai Pandar128 style).

The model describes the range-map lattice: one row per laser ring with its own
elevation and azimuth offset, and ``n_columns`` firing azimuths per revolution.
It replaces ``ncore.impl.sensors.lidar.RowOffsetStructuredSpinningLidarModel``
for the two operations the range-map pipeline needs:

* mapping a sensor-frame direction to its lattice cell, and
* mapping a sensor-frame direction to the relative time within the sweep at
  which the spinning head fires along it (:meth:`SpinningLidarModel.relative_frame_times`).

Conventions (identical to the Cosmos-Drive-Dreams toolkit and NCORE):

* sensor frame is FLU (x forward, y left, z up);
* elevation ``= asin(z / r)``, azimuth ``= atan2(y, x)`` in ``(-pi, pi]``;
* the head spins clockwise: column ``0`` looks backwards (azimuth ``pi``) and
  the azimuth decreases by ``2 * pi / n_columns`` per column, so the forward
  direction is the centre column of the strip;
* ring ``0`` is the top (highest elevation) ring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

DEFAULT_PARAMETERS_PATH = (
    Path(__file__).parent / "assets" / "row_offset_spinning_lidar_model_parameters.json"
)

# Tolerance (radians) applied to the vertical field-of-view check. NCORE uses a
# few float32 ulps; a slightly larger value keeps rays that sit exactly on the
# top/bottom ring after float32 round trips.
FOV_EPS_RAD = 1e-6

# Rig sensor names of the rig's top GT LiDAR across sensor-suite generations, most
# recent first. Shared by every sensor_collection / rig parser so that all of them
# pick the same lidar when no explicit sensor name is given.
RIG_LIDAR_SENSOR_PREFERENCE: tuple[str, ...] = (
    "lidar:gt:top:p128:v4p5",
    "lidar:gt:top:p128",
)

# Column mapping used by the *range-map projection* (points_to_range_map): the
# toolkit ignores per-row azimuth offsets when binning points into columns, so
# a stored column is a nominal azimuth bin shared by all rows.
_RE_SENSOR_BLOCK = re.compile(r"^lidars \{", re.MULTILINE)
# Start of the next top-level message (`lidars {`, `imus {`, `vehicle_config {`, ...): an
# unindented line that is not the closing brace of the current one.
_RE_TOP_LEVEL_LINE = re.compile(r"^(?!\})\S", re.MULTILINE)
_RE_LINE_PARAMS = re.compile(r"line_parameters \{([^}]*)\}")
_RE_SENSOR_NAME = re.compile(r'sensor_name: "([^"]+)"')
_RE_LINE_COUNT = re.compile(r"line_count: (\d+)")
_RE_FREQUENCY = re.compile(r"frequency: ([-\d.eE+]+)")
_RE_NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


@dataclass(frozen=True)
class SpinningLidarModel:
    """Lattice geometry of a row-offset spinning LiDAR.

    Attributes:
        row_elevations_rad: ``[n_rows]`` ring elevations, ring 0 first (top ring).
        row_azimuth_offsets_rad: ``[n_rows]`` per-ring azimuth offset added to the
            nominal column azimuth when the ring fires.
        n_columns: firing azimuths per revolution (range-map width).
        spinning_frequency_hz: head rotation rate; one sweep lasts ``1 / f`` seconds.
        sensor_name: optional rig sensor name the calibration came from.
    """

    row_elevations_rad: np.ndarray
    row_azimuth_offsets_rad: np.ndarray
    n_columns: int = 3600
    spinning_frequency_hz: float = 10.0
    sensor_name: str | None = None

    def __post_init__(self) -> None:
        elevations = np.asarray(self.row_elevations_rad, dtype=np.float64).reshape(-1)
        offsets = np.asarray(self.row_azimuth_offsets_rad, dtype=np.float64).reshape(-1)
        if elevations.size == 0:
            raise ValueError("row_elevations_rad must not be empty")
        if offsets.shape != elevations.shape:
            raise ValueError(
                f"row_azimuth_offsets_rad has shape {offsets.shape}, expected {elevations.shape} (one per ring)"
            )
        if self.n_columns <= 1:
            raise ValueError("n_columns must be larger than 1")
        if self.spinning_frequency_hz <= 0:
            raise ValueError("spinning_frequency_hz must be positive")
        object.__setattr__(self, "row_elevations_rad", elevations)
        object.__setattr__(self, "row_azimuth_offsets_rad", offsets)

    # ------------------------------------------------------------------ geometry
    @property
    def n_rows(self) -> int:
        return int(self.row_elevations_rad.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Range-map shape ``(n_rows, n_columns)``."""
        return self.n_rows, self.n_columns

    @property
    def column_azimuths_rad(self) -> np.ndarray:
        """``[n_columns]`` nominal azimuth of each column (clockwise, rear-start)."""
        return np.linspace(np.pi, -np.pi, self.n_columns, endpoint=False)

    @property
    def row_elevations_deg(self) -> np.ndarray:
        return np.rad2deg(self.row_elevations_rad)

    @property
    def sweep_duration_us(self) -> float:
        return 1.0e6 / self.spinning_frequency_hz

    def ray_directions(self) -> np.ndarray:
        """``[n_rows, n_columns, 3]`` unit ray through the *centre* of each lattice cell.

        Like the toolkit's ``range_map_to_ray_directions`` this uses the nominal
        column azimuths without per-row offsets, so ``range * ray`` inverts
        :func:`cosmos_auto.lidar.rangemap.points_to_range_map`. The ray of column
        ``k`` points at the cell centre ``pi - (k + 0.5) * step`` rather than at the
        column's leading edge ``pi - k * step`` (:attr:`column_azimuths_rad`):
        ``points_to_range_map`` bins azimuths by truncation, so an edge ray that is
        re-projected after float32 round-off lands in the neighbouring column
        roughly one time in eight and two unprojected cells then collide (with a
        static sensor the prefilter in :mod:`cosmos_auto.lidar.render` lost ~17% of
        its cells that way). Centre rays re-project to their own cell for every
        cell of the default 128 x 3600 lattice. Thinning at one pose remains lossy
        when the ego moves during the sweep, which is why the prefilter is opt-in.
        :attr:`column_azimuths_rad`, which drives the firing-time lookup, keeps the
        leading-edge convention.
        """
        column_step = 2.0 * np.pi / self.n_columns
        centre_azimuths = np.pi - (np.arange(self.n_columns) + 0.5) * column_step
        elevation, azimuth = np.meshgrid(
            self.row_elevations_rad, centre_azimuths, indexing="ij"
        )
        cos_elevation = np.cos(elevation)
        return np.stack(
            [
                cos_elevation * np.cos(azimuth),
                cos_elevation * np.sin(azimuth),
                np.sin(elevation),
            ],
            axis=-1,
        )

    # ------------------------------------------------------------------ firing time
    def nearest_rows(
        self, elevation_rad: torch.Tensor, chunk_size: int = 131_072
    ) -> torch.Tensor:
        """Nearest ring index for each elevation, with bounded temporary memory."""
        rows = torch.as_tensor(
            self.row_elevations_rad,
            dtype=elevation_rad.dtype,
            device=elevation_rad.device,
        )
        out = torch.empty(
            elevation_rad.shape, dtype=torch.long, device=elevation_rad.device
        )
        for start in range(0, elevation_rad.numel(), chunk_size):
            end = min(start + chunk_size, elevation_rad.numel())
            out[start:end] = torch.argmin(
                torch.abs(elevation_rad[start:end, None] - rows[None, :]), dim=1
            )
        return out

    def sensor_angles(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(elevation_rad, azimuth_rad, valid)`` for sensor-frame points.

        A direction is valid when its elevation lies inside the ring span
        ``[min(row_elevations), max(row_elevations)]`` (plus :data:`FOV_EPS_RAD`).
        The horizontal field of view is the full revolution.
        """
        radius = torch.linalg.norm(points, dim=-1)
        elevation = torch.asin(
            torch.clamp(points[:, 2] / torch.clamp(radius, min=1e-12), -1.0, 1.0)
        )
        azimuth = torch.atan2(points[:, 1], points[:, 0])
        low = float(self.row_elevations_rad.min()) - FOV_EPS_RAD
        high = float(self.row_elevations_rad.max()) + FOV_EPS_RAD
        valid = (radius > 0) & (elevation >= low) & (elevation <= high)
        return elevation, azimuth, valid

    def relative_frame_times(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relative sweep time in ``[0, 1]`` at which the head fires towards each point.

        The ring is the nearest elevation; the column is the nearest nominal
        azimuth after removing that ring's azimuth offset. Relative time is
        ``column / (n_columns - 1)`` as in NCORE, so column 0 fires at the sweep
        start and the last column at the sweep end.

        Returns:
            ``(relative_time [N], valid [N])``; ``relative_time`` is still defined
            (clamped) for invalid directions so callers can mask afterwards.
        """
        elevation, azimuth, valid = self.sensor_angles(points)
        rows = self.nearest_rows(elevation)
        offsets = torch.as_tensor(
            self.row_azimuth_offsets_rad, dtype=points.dtype, device=points.device
        )
        column_step = 2.0 * np.pi / self.n_columns
        # nominal azimuth of the firing column = azimuth - row offset; column k has azimuth pi - k*step
        column = torch.round((np.pi - (azimuth - offsets[rows])) / column_step)
        column = torch.remainder(column, self.n_columns)
        return column / (self.n_columns - 1), valid

    # ------------------------------------------------------------------ constructors
    @classmethod
    def default(cls) -> SpinningLidarModel:
        """Hesai P128 default calibration shipped with the Cosmos-Drive-Dreams toolkit."""
        return cls.from_parameters_json(DEFAULT_PARAMETERS_PATH)

    @classmethod
    def from_parameters_json(
        cls, path: str | Path, n_columns: int | None = None
    ) -> SpinningLidarModel:
        """Load from a JSON file with ``row_elevations_rad`` and ``row_azimuth_offsets_rad``.

        Accepts both the toolkit's per-clip ``lidar_sensor_config/<clip>.json``
        (two arrays) and the full default-parameter file (which also carries
        ``n_columns`` and ``spinning_frequency_hz``).
        """
        with open(path) as handle:
            params = json.load(handle)
        return cls(
            row_elevations_rad=np.asarray(
                params["row_elevations_rad"], dtype=np.float64
            ),
            row_azimuth_offsets_rad=np.asarray(
                params["row_azimuth_offsets_rad"], dtype=np.float64
            ),
            n_columns=int(n_columns or params.get("n_columns", 3600)),
            spinning_frequency_hz=float(params.get("spinning_frequency_hz", 10.0)),
            sensor_name=params.get("sensor_name"),
        )

    @classmethod
    def from_sensor_collection(
        cls,
        path_or_text: str | Path,
        sensor_name: str | None = None,
        n_columns: int = 3600,
    ) -> SpinningLidarModel:
        """Parse the RDS-HQ ``sensor_collection/<clip>.txt`` protobuf-text calibration.

        The file lists ``lidars { sensor_meta_data {...} calibration_parameters {
        rotating_lidar_parameters { line_count, line_parameters { pitch_radians,
        yaw_offset_radians } x N } } }``. proto3 text omits zero-valued fields, so
        a missing pitch or yaw defaults to ``0.0``.

        Args:
            path_or_text: path to the file, or its text content.
            sensor_name: pick this lidar when the collection has several; default
                is the first match of :data:`RIG_LIDAR_SENSOR_PREFERENCE`
                (see :func:`select_lidar_block`).
            n_columns: firing columns per revolution (not stored in the file).
        """
        text = read_sensor_collection_text(path_or_text)
        chosen_name, chosen = select_lidar_block(iter_lidar_blocks(text), sensor_name)

        line_blocks = _RE_LINE_PARAMS.findall(chosen)
        elevations: list[float] = []
        offsets: list[float] = []
        for block in line_blocks:
            pitch = re.search(r"pitch_radians: (-?[\d.eE+-]+)", block)
            yaw = re.search(r"yaw_offset_radians: (-?[\d.eE+-]+)", block)
            elevations.append(float(pitch.group(1)) if pitch else 0.0)
            offsets.append(float(yaw.group(1)) if yaw else 0.0)
        count_match = _RE_LINE_COUNT.search(chosen)
        if count_match and int(count_match.group(1)) != len(line_blocks):
            raise ValueError(
                f"parsed {len(line_blocks)} line_parameters, expected line_count {count_match.group(1)}"
            )
        if not elevations:
            raise ValueError("sensor_collection lidar block has no line_parameters")
        frequency_match = _RE_FREQUENCY.search(chosen)
        return cls(
            row_elevations_rad=np.asarray(elevations, dtype=np.float64),
            row_azimuth_offsets_rad=np.asarray(offsets, dtype=np.float64),
            n_columns=n_columns,
            spinning_frequency_hz=float(frequency_match.group(1))
            if frequency_match
            else 10.0,
            sensor_name=chosen_name or None,
        )

    def to_parameters_dict(self) -> dict[str, object]:
        """Serialise the calibration in the toolkit ``lidar_sensor_config`` layout."""
        return {
            "sensor_name": self.sensor_name,
            "n_columns": self.n_columns,
            "spinning_frequency_hz": self.spinning_frequency_hz,
            "row_elevations_rad": self.row_elevations_rad.tolist(),
            "row_azimuth_offsets_rad": self.row_azimuth_offsets_rad.tolist(),
        }


# ----------------------------------------------------------------------------- sensor_collection parsing
def read_sensor_collection_text(path_or_text: str | Path) -> str:
    """Return the proto-text content of a sensor collection given a path or the text itself."""
    if isinstance(path_or_text, Path):
        return path_or_text.read_text()
    if "\n" not in path_or_text and Path(path_or_text).exists():
        return Path(path_or_text).read_text()
    return path_or_text


def iter_lidar_blocks(text: str) -> list[tuple[str, str]]:
    """Split a sensor_collection proto-text into its top-level ``lidars {`` blocks.

    Returns ``[(sensor_name, block_text), ...]`` in file order. Each block runs
    from its ``lidars {`` line to the next top-level line (another ``lidars {``,
    or an ``imus {`` / ``vehicle_config {`` / ... message) or EOF, so sibling
    messages that follow the last lidar are never absorbed into its block;
    ``sensor_name`` is ``""`` when the block carries no ``sensor_name`` field.
    """
    starts = [match.start() for match in _RE_SENSOR_BLOCK.finditer(text)]
    if not starts:
        raise ValueError("sensor_collection has no 'lidars {' block")
    blocks: list[tuple[str, str]] = []
    for start in starts:
        # The block's own first line is unindented too, so search from the line after it.
        line_end = text.find("\n", start)
        next_top_level = _RE_TOP_LEVEL_LINE.search(
            text, len(text) if line_end < 0 else line_end + 1
        )
        end = next_top_level.start() if next_top_level else len(text)
        block = text[start:end]
        name_match = _RE_SENSOR_NAME.search(block)
        blocks.append((name_match.group(1) if name_match else "", block))
    return blocks


def select_lidar_block(
    blocks: list[tuple[str, str]], sensor_name: str | None = None
) -> tuple[str, str]:
    """Pick one ``(sensor_name, block_text)`` from :func:`iter_lidar_blocks`.

    An explicit ``sensor_name`` must match exactly. Otherwise the first name in
    :data:`RIG_LIDAR_SENSOR_PREFERENCE` present in the collection wins; a
    collection with a single lidar block is accepted as is. Raises ``KeyError``
    listing the available names when nothing matches.
    """
    names = [name for name, _ in blocks]
    if sensor_name is not None:
        for name, block in blocks:
            if name == sensor_name:
                return name, block
        raise KeyError(
            f"lidar sensor {sensor_name!r} not in sensor_collection; available: {names}"
        )
    for candidate in RIG_LIDAR_SENSOR_PREFERENCE:
        for name, block in blocks:
            if name == candidate:
                return name, block
    if len(blocks) == 1:
        return blocks[0]
    raise KeyError(
        f"could not pick a lidar from sensor_collection: none of {list(RIG_LIDAR_SENSOR_PREFERENCE)} "
        f"present and several blocks exist; available: {names}"
    )


def _braced_sub_block(text: str, field_name: str) -> str:
    """Body of the first ``<field_name> { ... }`` message in ``text`` (brace-matched)."""
    match = re.search(rf"\b{re.escape(field_name)}\s*\{{", text)
    if match is None:
        raise ValueError(
            f"sensor_collection lidar block has no '{field_name} {{' sub-message"
        )
    depth = 0
    for index in range(match.end() - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.end() : index]
    raise ValueError(f"unterminated '{field_name} {{' sub-message in sensor_collection")


def _scalar_field(
    text: str, field_name: str, context: str, default: float | None = None
) -> float:
    """Value of the first ``<field_name>: <number>`` in ``text``.

    proto3 text format omits fields that hold their default value, so a zero
    coordinate or angle is simply absent from the file (the same convention as
    the missing ``pitch_radians`` / ``yaw_offset_radians`` handling in
    :meth:`SpinningLidarModel.from_sensor_collection`). Callers that expect this
    pass ``default=0.0``; without a default a missing field raises ``ValueError``.
    """
    match = re.search(rf"\b{re.escape(field_name)}: {_RE_NUMBER}", text)
    if match is None:
        if default is None:
            raise ValueError(f"sensor_collection {context} has no '{field_name}' field")
        return default
    return float(match.group(1))


def parse_sensor_to_vehicle_transform(block: str) -> np.ndarray:
    """``[4, 4]`` sensor-to-vehicle extrinsic from one lidar block's ``sensor_to_vehicle_transform``.

    Only the ``axis_angle { x y z angle_degrees }`` and ``translation { x y z }``
    fields *inside* the ``sensor_to_vehicle_transform { ... }`` sub-message are
    read, so unrelated ``x:`` / ``y:`` / ``z:`` fields elsewhere in the block
    cannot be mistaken for the extrinsic. Both sub-messages must be present, but
    any scalar they omit defaults to ``0.0`` (proto3 text drops zero-valued
    fields): a translation with no ``y`` sits on the vehicle's centre line, and
    an ``axis_angle`` without ``angle_degrees`` (or an empty one) is the
    identity rotation. A non-zero angle about a zero-length axis is rejected.
    """
    transform_text = _braced_sub_block(block, "sensor_to_vehicle_transform")
    axis_text = _braced_sub_block(transform_text, "axis_angle")
    translation_text = _braced_sub_block(transform_text, "translation")
    axis = np.asarray(
        [_scalar_field(axis_text, name, "axis_angle", 0.0) for name in ("x", "y", "z")],
        dtype=np.float64,
    )
    angle_deg = _scalar_field(axis_text, "angle_degrees", "axis_angle", 0.0)
    transform = np.eye(4, dtype=np.float64)
    if angle_deg != 0.0:
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError(
                "sensor_collection axis_angle has a zero-length axis with a non-zero angle_degrees"
            )
        transform[:3, :3] = Rotation.from_rotvec(
            axis / norm * np.deg2rad(angle_deg)
        ).as_matrix()
    transform[:3, 3] = [
        _scalar_field(translation_text, name, "translation", 0.0)
        for name in ("x", "y", "z")
    ]
    return transform
