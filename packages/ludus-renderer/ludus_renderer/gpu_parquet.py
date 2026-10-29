# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GPU-native parquet loading pipeline.

Uploads raw tar bytes to GPU, batch-decompresses Snappy pages via nvcomp,
and decodes RLE_DICTIONARY-encoded columns with custom CUDA kernels --
eliminating CPU-side PyArrow parsing entirely.

Pipeline:
  1. CPU: read tar(s) into pinned buffer, parse metadata (tar headers +
     parquet footers + page headers) to build a page index
  2. Single cudaMemcpyAsync of entire buffer to GPU
  3. GPU: nvcomp batch Snappy decompress all pages
  4. GPU: custom CUDA kernels decode RLE dictionary indices, gather values,
     reconstruct list offsets from repetition levels
  5. Result: FlatPolylineData tensors born on GPU
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from torch import Tensor

from ludus_renderer._cuda_toolchain import configure_cuda_toolchain, find_nvidia_library

try:
    from nvidia import nvcomp as _nvcomp
except ImportError:
    _nvcomp = None

# ---------------------------------------------------------------------------
# JIT-compiled CUDA extension for RLE decode (shared memory, fast path)
# ---------------------------------------------------------------------------

_rle_cuda_ext = None


def _get_rle_cuda_ext():
    global _rle_cuda_ext
    if _rle_cuda_ext is not None:
        return _rle_cuda_ext

    import os

    from torch.utils.cpp_extension import load

    configure_cuda_toolchain()
    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")
    _rle_cuda_ext = load(
        name="rle_decode_cuda",
        sources=[
            os.path.join(csrc, "rle_decode_binding.cpp"),
            os.path.join(csrc, "rle_decode.cu"),
        ],
        verbose=False,
    )
    return _rle_cuda_ext


# ---------------------------------------------------------------------------
# JIT-compiled CUDA extension for fused gather + filter
# ---------------------------------------------------------------------------

_gather_cuda_ext = None


def _get_gather_cuda_ext():
    global _gather_cuda_ext
    if _gather_cuda_ext is not None:
        return _gather_cuda_ext

    import os

    from torch.utils.cpp_extension import load

    configure_cuda_toolchain()
    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")
    _gather_cuda_ext = load(
        name="gather_filter_cuda",
        sources=[
            os.path.join(csrc, "gather_filter_binding.cpp"),
            os.path.join(csrc, "gather_filter.cu"),
        ],
        verbose=False,
    )
    return _gather_cuda_ext


# ---------------------------------------------------------------------------
# JIT-compiled CUDA extension for batch Snappy decompress (GIL-free)
# ---------------------------------------------------------------------------

_snappy_cuda_ext = None


def _get_snappy_cuda_ext():
    global _snappy_cuda_ext
    if _snappy_cuda_ext is not None:
        return _snappy_cuda_ext

    import os

    from torch.utils.cpp_extension import load

    toolchain = configure_cuda_toolchain()
    include_dir, lib_dir, nvcomp_soname = find_nvidia_library(
        (
            "nvidia-libnvcomp-cu13",
            "nvidia-libnvcomp-cu12",
            "nvidia-libnvcomp-cu11",
            "nvidia-nvcomp-cu11",
        ),
        "nvcomp/snappy.h",
        ("libnvcomp.so.5", "libnvcomp.so"),
        toolchain,
        import_package="nvidia.libnvcomp",
    )
    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")

    _snappy_cuda_ext = load(
        name="snappy_decompress_cuda",
        sources=[
            os.path.join(csrc, "snappy_decompress_binding.cpp"),
            os.path.join(csrc, "snappy_decompress.cu"),
        ],
        extra_include_paths=[str(include_dir)],
        extra_ldflags=[f"-L{lib_dir}", f"-l:{nvcomp_soname}", f"-Wl,-rpath,{lib_dir}"],
        verbose=False,
    )
    return _snappy_cuda_ext


# ---------------------------------------------------------------------------
# JIT-compiled fused pipeline: decompress → RLE → gather in one GIL-free call
# ---------------------------------------------------------------------------

_pipeline_cuda_ext = None


def _get_pipeline_cuda_ext():
    global _pipeline_cuda_ext
    if _pipeline_cuda_ext is not None:
        return _pipeline_cuda_ext

    import os

    from torch.utils.cpp_extension import load

    configure_cuda_toolchain()
    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")

    _pipeline_cuda_ext = load(
        name="polyline_pipeline_cuda",
        sources=[
            os.path.join(csrc, "polyline_pipeline_binding.cpp"),
            os.path.join(csrc, "polyline_pipeline.cu"),
            os.path.join(csrc, "rle_decode.cu"),
            os.path.join(csrc, "gather_filter.cu"),
        ],
        verbose=False,
    )
    return _pipeline_cuda_ext


# ---------------------------------------------------------------------------
# JIT-compiled C++ extension for fast parquet page scanning
# ---------------------------------------------------------------------------

_parquet_scan_ext = None


def _get_parquet_scan_ext():
    global _parquet_scan_ext
    if _parquet_scan_ext is not None:
        return _parquet_scan_ext

    import os

    from torch.utils.cpp_extension import load

    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")
    _parquet_scan_ext = load(
        name="parquet_scan_cpp",
        sources=[os.path.join(csrc, "parquet_scan.cpp")],
        verbose=False,
    )
    return _parquet_scan_ext


# ---------------------------------------------------------------------------
# JIT-compiled unified scene loader (Phase 1 orchestrator)
# ---------------------------------------------------------------------------

_scene_loader_ext = None


def _get_scene_loader_ext():
    global _scene_loader_ext
    if _scene_loader_ext is not None:
        return _scene_loader_ext

    import os

    from torch.utils.cpp_extension import load

    toolchain = configure_cuda_toolchain()
    include_dir, lib_dir, nvcomp_soname = find_nvidia_library(
        (
            "nvidia-libnvcomp-cu13",
            "nvidia-libnvcomp-cu12",
            "nvidia-libnvcomp-cu11",
            "nvidia-nvcomp-cu11",
        ),
        "nvcomp/snappy.h",
        ("libnvcomp.so.5", "libnvcomp.so"),
        toolchain,
        import_package="nvidia.libnvcomp",
    )
    csrc = os.path.join(os.path.dirname(__file__), "_cpp", "loader")

    _scene_loader_ext = load(
        name="scene_loader_cuda",
        sources=[
            os.path.join(csrc, "scene_loader.cpp"),
            os.path.join(csrc, "snappy_decompress.cu"),
            os.path.join(csrc, "polyline_pipeline.cu"),
            os.path.join(csrc, "rle_decode.cu"),
            os.path.join(csrc, "gather_filter.cu"),
            os.path.join(csrc, "ego_transform.cu"),
            os.path.join(csrc, "obs_fused.cu"),
            os.path.join(csrc, "parquet_scan.cpp"),
            os.path.join(csrc, "camera_convert.cpp"),
        ],
        extra_include_paths=[str(include_dir)],
        extra_ldflags=[f"-L{lib_dir}", f"-l:{nvcomp_soname}", f"-Wl,-rpath,{lib_dir}"],
        extra_cflags=["-DSCENE_LOADER_BUILD"],
        extra_cuda_cflags=["-DSCENE_LOADER_BUILD"],
        verbose=False,
    )
    return _scene_loader_ext


# Parquet page types
PAGE_TYPE_DATA = 0
PAGE_TYPE_INDEX = 1
PAGE_TYPE_DICTIONARY = 2
PAGE_TYPE_DATA_V2 = 3

# ---------------------------------------------------------------------------
# Thrift compact protocol parser (minimal, for parquet page headers)
# ---------------------------------------------------------------------------


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    """Read an unsigned varint (ULEB128)."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result, pos


def _zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _skip_thrift_field(buf: bytes, pos: int, type_id: int) -> int:
    """Skip a Thrift compact protocol field value."""
    if type_id in (1, 2):  # bool true/false (value encoded in type)
        return pos
    elif type_id == 3:  # i8
        return pos + 1
    elif type_id in (4, 5, 6):  # i16, i32, i64 (zigzag varint)
        _, pos = _read_varint(buf, pos)
        return pos
    elif type_id == 7:  # double
        return pos + 8
    elif type_id == 8:  # binary (varint length + bytes)
        length, pos = _read_varint(buf, pos)
        return pos + length
    elif type_id in (9, 10):  # list, set
        header = buf[pos]
        pos += 1
        size = (header >> 4) & 0x0F
        elem_type = header & 0x0F
        if size == 15:
            size, pos = _read_varint(buf, pos)
        for _ in range(size):
            pos = _skip_thrift_field(buf, pos, elem_type)
        return pos
    elif type_id == 11:  # map
        size, pos = _read_varint(buf, pos)
        if size > 0:
            types = buf[pos]
            pos += 1
            key_type = (types >> 4) & 0x0F
            val_type = types & 0x0F
            for _ in range(size):
                pos = _skip_thrift_field(buf, pos, key_type)
                pos = _skip_thrift_field(buf, pos, val_type)
        return pos
    elif type_id == 12:  # struct
        while True:
            byte = buf[pos]
            pos += 1
            if byte == 0:
                break
            ft = byte & 0x0F
            delta = (byte >> 4) & 0x0F
            if delta == 0:
                _, pos = _read_varint(buf, pos)
            pos = _skip_thrift_field(buf, pos, ft)
        return pos
    else:
        raise ValueError(f"Unknown Thrift compact type_id: {type_id}")


def _parse_page_header(buf: bytes, offset: int) -> Tuple[int, int, int, int]:
    """Parse a parquet PageHeader (Thrift compact protocol).

    Returns:
        (page_type, uncompressed_size, compressed_size, header_byte_count)
    """
    pos = offset
    prev_fid = 0
    page_type = -1
    uncompressed_size = -1
    compressed_size = -1

    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        if byte == 0:  # STOP
            break

        type_id = byte & 0x0F
        delta = (byte >> 4) & 0x0F

        if delta == 0:
            fid_raw, pos = _read_varint(buf, pos)
            cur_fid = _zigzag_decode(fid_raw)
        else:
            cur_fid = prev_fid + delta
        prev_fid = cur_fid

        if cur_fid == 1:  # PageType (enum stored as i32)
            val, pos = _read_varint(buf, pos)
            page_type = _zigzag_decode(val)
        elif cur_fid == 2:  # uncompressed_page_size
            val, pos = _read_varint(buf, pos)
            uncompressed_size = _zigzag_decode(val)
        elif cur_fid == 3:  # compressed_page_size
            val, pos = _read_varint(buf, pos)
            compressed_size = _zigzag_decode(val)
        else:
            pos = _skip_thrift_field(buf, pos, type_id)

    return page_type, uncompressed_size, compressed_size, pos - offset


# ---------------------------------------------------------------------------
# Page index data structures
# ---------------------------------------------------------------------------


@dataclass
class PageInfo:
    """Metadata for a single parquet page within a raw byte buffer."""

    page_type: int
    data_offset: int  # byte offset of compressed page data in the buffer
    compressed_size: int
    uncompressed_size: int


@dataclass
class ColumnPageIndex:
    """Page index for one column in a parquet file."""

    path: str
    dict_page: Optional[PageInfo] = None
    data_pages: List[PageInfo] = field(default_factory=list)
    num_values: int = 0
    max_repetition_level: int = 0
    max_definition_level: int = 0


@dataclass
class ParquetPageIndex:
    """Complete page index for a parquet file."""

    columns: Dict[str, ColumnPageIndex] = field(default_factory=dict)


def scan_parquet_pages(
    raw_bytes: bytes,
    columns: Optional[List[str]] = None,
) -> ParquetPageIndex:
    """Scan a parquet file's raw bytes and build a page index.

    Uses PyArrow to read the Thrift footer (column chunk metadata), then
    manually parses page headers at the indicated offsets to locate each
    compressed data region.

    Args:
        raw_bytes: Complete raw bytes of the parquet file.
        columns: Column paths to index.  None = all columns.

    Returns:
        ParquetPageIndex mapping column_path -> page list.
    """
    pf = pq.ParquetFile(BytesIO(raw_bytes))
    meta = pf.metadata
    schema = pf.schema
    column_set = set(columns) if columns else None

    # Build path -> (max_rep, max_def) from the parquet schema
    _rep_def: Dict[str, Tuple[int, int]] = {}
    for i in range(meta.row_group(0).num_columns if meta.num_row_groups > 0 else 0):
        sc = schema.column(i)
        _rep_def[sc.path] = (sc.max_repetition_level, sc.max_definition_level)

    result = ParquetPageIndex()

    for rg_idx in range(meta.num_row_groups):
        rg = meta.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col_meta = rg.column(col_idx)
            path = col_meta.path_in_schema

            if column_set and path not in column_set:
                continue

            rep, defn = _rep_def.get(path, (0, 0))
            col_index = ColumnPageIndex(
                path=path,
                num_values=col_meta.num_values,
                max_repetition_level=rep,
                max_definition_level=defn,
            )

            dict_off = col_meta.dictionary_page_offset
            data_off = col_meta.data_page_offset
            chunk_start = dict_off if dict_off is not None and dict_off >= 0 else data_off
            chunk_end = chunk_start + col_meta.total_compressed_size

            pos = chunk_start
            while pos < chunk_end:
                ptype, uncomp, comp, hdr_size = _parse_page_header(raw_bytes, pos)
                if comp < 0:
                    break

                page = PageInfo(
                    page_type=ptype,
                    data_offset=pos + hdr_size,
                    compressed_size=comp,
                    uncompressed_size=uncomp,
                )

                if ptype == PAGE_TYPE_DICTIONARY:
                    col_index.dict_page = page
                else:
                    col_index.data_pages.append(page)

                pos += hdr_size + comp

            result.columns[path] = col_index

    return result


# ---------------------------------------------------------------------------
# nvcomp batch Snappy decompression
# ---------------------------------------------------------------------------

_nvcomp_codec = None


def _get_nvcomp_codec():
    """Get or create a cached nvcomp Snappy codec with RAW bitstream kind."""
    global _nvcomp_codec
    if _nvcomp_codec is not None:
        return _nvcomp_codec
    if _nvcomp is None:
        raise RuntimeError("nvidia-nvcomp-cu12 is not installed")
    _nvcomp_codec = _nvcomp.Codec(
        algorithm="snappy",
        bitstream_kind=_nvcomp.BitstreamKind.RAW,
    )
    return _nvcomp_codec


def batch_decompress_pages(
    gpu_buffer: Tensor,
    pages: List[PageInfo],
) -> List[Tensor]:
    """Batch-decompress Snappy-compressed parquet pages on GPU.

    Args:
        gpu_buffer: uint8 tensor on CUDA containing the raw file data.
        pages: List of PageInfo with offsets into gpu_buffer.

    Returns:
        List of decompressed uint8 tensors on the same GPU device.
    """
    if not pages:
        return []

    # nvcomp rejects 0-length buffers; filter them out, decompress the rest,
    # then stitch empty tensors back into the correct positions.
    non_empty = [(i, p) for i, p in enumerate(pages) if p.compressed_size > 0]

    if not non_empty:
        return [torch.empty(0, dtype=torch.uint8, device=gpu_buffer.device) for _ in pages]

    codec = _get_nvcomp_codec()
    slices = [gpu_buffer[p.data_offset : p.data_offset + p.compressed_size] for _, p in non_empty]
    arrays = _nvcomp.as_arrays(slices)
    decoded = codec.decode(arrays)
    decoded_tensors = [torch.as_tensor(d, device=gpu_buffer.device) for d in decoded]

    result = [None] * len(pages)
    empty = torch.empty(0, dtype=torch.uint8, device=gpu_buffer.device)
    for (orig_idx, _), dt in zip(non_empty, decoded_tensors):
        result[orig_idx] = dt
    for i in range(len(result)):
        if result[i] is None:
            result[i] = empty
    return result


# ---------------------------------------------------------------------------
# CUDA kernel for RLE/bit-packing decode (shared memory, speculative fast path)
#
# Single kernel: 1 thread block (256 threads) per RLE stream.
# Fast path: speculative all-BP validation → each thread unpacks 1 group of 8.
# Slow path: thread 0 scans headers in SMEM → all threads cooperative unpack.
# ---------------------------------------------------------------------------


def decode_data_pages_gpu(
    decompressed_pages: List[Tensor],
    max_rep_levels: List[int],
    max_def_levels: List[int],
    num_values_list: List[int],
    device: torch.device,
    scratch: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode decompressed data pages using CUDA SMEM kernel.

    Single kernel launch: 1 block (256 threads) per RLE stream.
    Speculative all-BP fast path for fully parallel decode.
    Sequential fallback for streams with RLE groups.

    Returns:
        (rep_levels, def_levels, indices) as flat int32 GPU tensors.
    """
    ext = _get_rle_cuda_ext()

    n_pages = len(decompressed_pages)
    total_values = sum(num_values_list)

    # Concatenate decompressed pages with 4B padding
    total_decomp = sum(t.shape[0] for t in decompressed_pages) + 4
    if scratch and "dcat" in scratch and scratch["dcat"].shape[0] >= total_decomp:
        dcat = scratch["dcat"]
    else:
        dcat = torch.empty(total_decomp, dtype=torch.uint8, device=device)

    d_offsets: List[int] = []
    pos = 0
    for t in decompressed_pages:
        d_offsets.append(pos)
        dcat[pos : pos + t.shape[0]] = t
        pos += t.shape[0]
    dcat[pos : pos + 4] = 0
    concat_data = dcat[: pos + 4]

    page_out_starts: List[int] = []
    s = 0
    for nv in num_values_list:
        page_out_starts.append(s)
        s += nv

    meta_offset = torch.tensor(d_offsets, dtype=torch.int32, device=device)
    meta_length = torch.tensor([t.shape[0] for t in decompressed_pages], dtype=torch.int32, device=device)
    meta_max_rep = torch.tensor(max_rep_levels, dtype=torch.int32, device=device)
    meta_max_def = torch.tensor(max_def_levels, dtype=torch.int32, device=device)
    meta_num_val = torch.tensor(num_values_list, dtype=torch.int32, device=device)
    meta_page_out = torch.tensor(page_out_starts, dtype=torch.int32, device=device)

    output = ext.decode_rle_streams(
        concat_data,
        meta_offset,
        meta_length,
        meta_max_rep,
        meta_max_def,
        meta_num_val,
        meta_page_out,
        total_values,
    )

    out_rep = output[:total_values]
    out_def = output[total_values : 2 * total_values]
    out_idx = output[2 * total_values :]
    return out_rep, out_def, out_idx


def rep_levels_to_offsets_gpu(rep_levels: Tensor) -> Tensor:
    """Convert repetition levels to CSR-style row offsets on GPU.

    rep_level == 0 marks the start of a new top-level row (list).
    """
    boundaries = torch.where(rep_levels == 0)[0].to(torch.int32)
    n_rows = boundaries.shape[0]
    offsets = torch.empty(n_rows + 1, dtype=torch.int32, device=rep_levels.device)
    offsets[:n_rows] = boundaries
    offsets[n_rows] = rep_levels.shape[0]
    return offsets


# ---------------------------------------------------------------------------
# Column assembly: pages -> FlatPolylineData on GPU
# ---------------------------------------------------------------------------


@dataclass
class _PolylineColumnSpec:
    """Specifies the parquet columns needed for one polyline parquet file."""

    data_col: str
    pts_field: str
    x_path: str
    y_path: str
    z_path: str
    ts_path: str = "key.timestamp_micros"


def _make_polyline_spec(data_col: str, pts_field: str) -> _PolylineColumnSpec:
    base = f"{data_col}.{pts_field}.list.element"
    return _PolylineColumnSpec(
        data_col=data_col,
        pts_field=pts_field,
        x_path=f"{base}.x",
        y_path=f"{base}.y",
        z_path=f"{base}.z",
    )


# Specs for each AV2 polyline parquet file
POLYLINE_SPECS = {
    "cf_road_boundary.parquet": _make_polyline_spec(
        "cf_road_boundary",
        "road_boundary_polyline",
    ),
    "dw_lane_line.parquet": _make_polyline_spec(
        "dw_lane_line",
        "points",
    ),
    "cf_crosswalks.parquet": _make_polyline_spec(
        "cf_crosswalks",
        "crosswalk_area",
    ),
    "cf_static_obstacle.parquet": _make_polyline_spec(
        "cf_static_obstacle",
        "boundary_points",
    ),
}


# ---------------------------------------------------------------------------
# High-level single-scene and batch GPU-native loaders
# ---------------------------------------------------------------------------


def _unpack_cpp_scan_result(
    flat: Tensor,
    column_paths: List[str],
) -> Optional[ParquetPageIndex]:
    """Unpack the flat int64 tensor from C++ scan into ParquetPageIndex."""
    data = flat.numpy()
    if len(data) == 0:
        return None

    result = ParquetPageIndex()
    pos = 0
    for col_path in column_paths:
        if pos >= len(data):
            return None
        if data[pos] == -1:
            return None
        num_values = int(data[pos])
        pos += 1
        max_rep = int(data[pos])
        pos += 1
        max_def = int(data[pos])
        pos += 1
        has_dict = int(data[pos])
        pos += 1
        dict_data_offset = int(data[pos])
        pos += 1
        dict_comp = int(data[pos])
        pos += 1
        dict_uncomp = int(data[pos])
        pos += 1
        n_data_pages = int(data[pos])
        pos += 1

        col_index = ColumnPageIndex(
            path=col_path,
            num_values=num_values,
            max_repetition_level=max_rep,
            max_definition_level=max_def,
        )
        if has_dict:
            col_index.dict_page = PageInfo(
                page_type=PAGE_TYPE_DICTIONARY,
                data_offset=dict_data_offset,
                compressed_size=dict_comp,
                uncompressed_size=dict_uncomp,
            )
        for _ in range(n_data_pages):
            d_off = int(data[pos])
            pos += 1
            d_comp = int(data[pos])
            pos += 1
            d_uncomp = int(data[pos])
            pos += 1
            col_index.data_pages.append(
                PageInfo(
                    page_type=PAGE_TYPE_DATA,
                    data_offset=d_off,
                    compressed_size=d_comp,
                    uncompressed_size=d_uncomp,
                )
            )
        result.columns[col_path] = col_index

    return result


def scan_polyline_metadata(
    pinned: Tensor,
    entries: List["TarFileEntry"],
    min_points_map: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, Tuple], Dict[str, object]]:
    """CPU-only metadata scan for polyline parquet files.

    Uses a C++ parquet footer/page-header parser for speed, falling back
    to the Python+PyArrow path if the C++ extension isn't available.

    Returns:
        (file_metas, initial_results) where file_metas maps
        basename -> (page_index, offset, spec, min_pts) and
        initial_results maps basename -> None for skipped files.
    """
    if min_points_map is None:
        min_points_map = {
            "cf_road_boundary.parquet": 2,
            "dw_lane_line.parquet": 2,
            "cf_crosswalks.parquet": 3,
            "cf_static_obstacle.parquet": 2,
        }

    entry_map: Dict[str, "TarFileEntry"] = {}
    for e in entries:
        basename = e.name.rsplit("/", 1)[-1] if "/" in e.name else e.name
        entry_map[basename] = e

    # Try C++ fast path
    try:
        scan_ext = _get_parquet_scan_ext()
    except Exception:
        scan_ext = None

    file_metas: Dict[str, Tuple] = {}
    results: Dict[str, object] = {}

    for pq_basename, spec in POLYLINE_SPECS.items():
        entry = entry_map.get(pq_basename)
        if entry is None:
            results[pq_basename] = None
            continue

        columns_needed = [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]

        if scan_ext is not None:
            flat = scan_ext.scan_parquet_pages_cpp(
                pinned,
                entry.offset,
                entry.size,
                columns_needed,
            )
            page_index = _unpack_cpp_scan_result(flat, columns_needed)
        else:
            pq_bytes = pinned[entry.offset : entry.offset + entry.size].numpy().tobytes()
            page_index = scan_parquet_pages(pq_bytes, columns=columns_needed)

        if page_index is None:
            results[pq_basename] = None
            continue

        skip = False
        for col in columns_needed:
            if col not in page_index.columns or not page_index.columns[col].data_pages:
                skip = True
                break
        if skip:
            results[pq_basename] = None
            continue

        min_pts = min_points_map.get(pq_basename, 2)
        file_metas[pq_basename] = (page_index, entry.offset, spec, min_pts)

    return file_metas, results


def prepare_pipeline_plan(
    file_metas: Dict[str, Tuple],
) -> Optional[Dict]:
    """Pre-compute all metadata tensors for the fused C++ pipeline.

    This is pure Python that runs before any ThreadPoolExecutor to avoid
    GIL contention during the overlap window. All tensors are created on
    CPU; the C++ pipeline moves the small ones to GPU internally.

    Returns None if file_metas is empty.
    """
    if not file_metas:
        return None

    all_page_offsets: List[int] = []
    all_page_comp: List[int] = []
    all_page_uncomp: List[int] = []
    all_pages: List[PageInfo] = []

    data_page_indices: List[int] = []
    xyz_dict_page_indices: List[int] = []
    xyz_dict_byte_offsets: List[int] = []
    ts_dict_page_indices: List[int] = []
    ts_dict_byte_offsets: List[int] = []

    all_max_reps: List[int] = []
    all_max_defs: List[int] = []
    all_num_vals: List[int] = []

    file_order: List[str] = []
    dict_xyz_byte_cursor = 0
    dict_ts_byte_cursor = 0

    for pq_basename, (page_index, parquet_offset, spec, _) in file_metas.items():
        file_order.append(pq_basename)
        for col_idx, col_path in enumerate([spec.x_path, spec.y_path, spec.z_path, spec.ts_path]):
            col = page_index.columns[col_path]
            if col.dict_page is not None:
                pi = len(all_page_offsets)
                abs_off = col.dict_page.data_offset + parquet_offset
                all_page_offsets.append(abs_off)
                all_page_comp.append(col.dict_page.compressed_size)
                all_page_uncomp.append(col.dict_page.uncompressed_size)
                all_pages.append(
                    PageInfo(
                        page_type=col.dict_page.page_type,
                        data_offset=abs_off,
                        compressed_size=col.dict_page.compressed_size,
                        uncompressed_size=col.dict_page.uncompressed_size,
                    )
                )
                if col_idx < 3:
                    xyz_dict_page_indices.append(pi)
                    xyz_dict_byte_offsets.append(dict_xyz_byte_cursor)
                    dict_xyz_byte_cursor += col.dict_page.uncompressed_size
                else:
                    ts_dict_page_indices.append(pi)
                    ts_dict_byte_offsets.append(dict_ts_byte_cursor)
                    dict_ts_byte_cursor += col.dict_page.uncompressed_size
            for i, dp in enumerate(col.data_pages):
                pi = len(all_page_offsets)
                abs_off = dp.data_offset + parquet_offset
                all_page_offsets.append(abs_off)
                all_page_comp.append(dp.compressed_size)
                all_page_uncomp.append(dp.uncompressed_size)
                all_pages.append(
                    PageInfo(
                        page_type=dp.page_type,
                        data_offset=abs_off,
                        compressed_size=dp.compressed_size,
                        uncompressed_size=dp.uncompressed_size,
                    )
                )
                if i == 0:
                    data_page_indices.append(pi)
            all_max_reps.append(col.max_repetition_level)
            all_max_defs.append(col.max_definition_level)
            all_num_vals.append(col.num_values)

    out_starts: List[int] = []
    s = 0
    for nv in all_num_vals:
        out_starts.append(s)
        s += nv
    total_rle_values = s

    file_info_list: List[int] = []
    total_xyz = 0
    total_ts = 0
    total_rows = 0
    row_off_cursor = 0
    dict_xyz_off = 0
    dict_ts_off = 0
    stream_idx = 0

    for fi_idx, pq_basename in enumerate(file_order):
        page_index, _, spec, min_pts = file_metas[pq_basename]
        si_x, si_y, si_z, si_ts = stream_idx, stream_idx + 1, stream_idx + 2, stream_idx + 3
        n_xyz = all_num_vals[si_x]
        n_rows = all_num_vals[si_ts]

        dx_col = page_index.columns[spec.x_path]
        dy_col = page_index.columns[spec.y_path]
        dz_col = page_index.columns[spec.z_path]
        dx_len = dx_col.dict_page.uncompressed_size // 4
        dy_len = dy_col.dict_page.uncompressed_size // 4
        dz_len = dz_col.dict_page.uncompressed_size // 4

        file_info_list.extend(
            [
                out_starts[si_x],
                out_starts[si_y],
                out_starts[si_z],
                out_starts[si_ts],
                out_starts[si_x],
                n_xyz,
                n_rows,
                min_pts,
                dict_xyz_off,
                dict_xyz_off + dx_len,
                dict_xyz_off + dx_len + dy_len,
                dict_ts_off,
                total_xyz,
                total_ts,
                row_off_cursor,
                total_rows,
                fi_idx * 2,
            ]
        )

        dict_xyz_off += dx_len + dy_len + dz_len
        dts_col = page_index.columns[spec.ts_path]
        dict_ts_off += dts_col.dict_page.uncompressed_size // 8
        total_xyz += n_xyz
        total_ts += n_rows
        total_rows += n_rows
        row_off_cursor += n_rows + 1
        stream_idx += 4

    return {
        "page_offsets": torch.tensor(all_page_offsets, dtype=torch.int64),
        "page_comp_sizes": torch.tensor(all_page_comp, dtype=torch.int64),
        "page_uncomp_sizes": torch.tensor(all_page_uncomp, dtype=torch.int64),
        "data_page_indices": torch.tensor(data_page_indices, dtype=torch.int32),
        "rle_max_rep": torch.tensor(all_max_reps, dtype=torch.int32),
        "rle_max_def": torch.tensor(all_max_defs, dtype=torch.int32),
        "rle_num_vals": torch.tensor(all_num_vals, dtype=torch.int32),
        "rle_out_starts": torch.tensor(out_starts, dtype=torch.int32),
        "total_rle_values": total_rle_values,
        "xyz_dict_page_indices": torch.tensor(xyz_dict_page_indices, dtype=torch.int32),
        "xyz_dict_byte_offsets": torch.tensor(xyz_dict_byte_offsets, dtype=torch.int32),
        "total_xyz_dict_bytes": dict_xyz_byte_cursor,
        "ts_dict_page_indices": torch.tensor(ts_dict_page_indices, dtype=torch.int32),
        "ts_dict_byte_offsets": torch.tensor(ts_dict_byte_offsets, dtype=torch.int32),
        "total_ts_dict_bytes": dict_ts_byte_cursor,
        "file_info_raw": torch.tensor(file_info_list, dtype=torch.int32),
        "n_files": len(file_order),
        "total_xyz_values": total_xyz,
        "total_ts_values": total_ts,
        "total_rows": total_rows,
        "file_order": file_order,
        "file_info_list": file_info_list,
        "all_num_vals": all_num_vals,
        "all_pages": all_pages,
    }


def load_polylines_gpu_native(
    tar_path: str,
    device: torch.device,
    min_points_map: Optional[Dict[str, int]] = None,
    preloaded: Optional[Tuple[Tensor, List["TarFileEntry"]]] = None,
    pre_scanned: Optional[Tuple[Dict[str, Tuple], Dict[str, object]]] = None,
    pre_plan: Optional[Dict] = None,
) -> Dict[str, object]:
    """Load all polyline parquet files from a tar using GPU-native decoding.

    Batches all GPU work across all 4 parquet files into single kernel
    launches: one nvcomp decompress, one CUDA RLE decode, then async
    dictionary gather and filtering with a single final sync.

    Args:
        tar_path: Path to the AV2 scene tar file.
        device: CUDA device.
        min_points_map: Optional per-file minimum point count override.
            Keys are parquet basenames. Defaults to 2 for polylines, 3 for polygons.
        preloaded: Optional (pinned_tensor, entries) from a prior
            ``read_tar_to_pinned_buffer`` call to avoid re-reading the tar.
        pre_scanned: Optional (file_metas, initial_results) from a prior
            ``scan_polyline_metadata`` call to skip the CPU metadata scan.
        pre_plan: Optional pre-computed pipeline plan from
            ``prepare_pipeline_plan`` for the fused GIL-free C++ path.

    Returns:
        Dict mapping parquet basename -> FlatPolylineData (or None).
    """
    import time as _time

    from .clipgt import FlatPolylineData

    if min_points_map is None:
        min_points_map = {
            "cf_road_boundary.parquet": 2,
            "dw_lane_line.parquet": 2,
            "cf_crosswalks.parquet": 3,
            "cf_static_obstacle.parquet": 2,
        }

    if preloaded is not None:
        pinned, entries = preloaded
    else:
        pinned, entries = read_tar_to_pinned_buffer(tar_path)

    _b0 = _time.perf_counter()
    gpu_buffer = pinned.to(device, non_blocking=True)
    _b1 = _time.perf_counter()

    if pre_scanned is not None:
        file_metas, results = pre_scanned
        results = dict(results)
    else:
        entry_map: Dict[str, TarFileEntry] = {}
        for e in entries:
            basename = e.name.rsplit("/", 1)[-1] if "/" in e.name else e.name
            entry_map[basename] = e

        file_metas: Dict[str, Tuple] = {}
        results: Dict[str, object] = {}

        for pq_basename, spec in POLYLINE_SPECS.items():
            entry = entry_map.get(pq_basename)
            if entry is None:
                results[pq_basename] = None
                continue

            pq_bytes = pinned[entry.offset : entry.offset + entry.size].numpy().tobytes()
            columns_needed = [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]
            page_index = scan_parquet_pages(pq_bytes, columns=columns_needed)

            skip = False
            for col in columns_needed:
                if col not in page_index.columns or not page_index.columns[col].data_pages:
                    skip = True
                    break
            if skip:
                results[pq_basename] = None
                continue

            min_pts = min_points_map.get(pq_basename, 2)
            file_metas[pq_basename] = (page_index, entry.offset, spec, min_pts)

    _b2 = _time.perf_counter()

    if not file_metas:
        print(
            f"    [B breakdown] H2D: {(_b1 - _b0) * 1000:.2f}ms, "
            f"cpu_scan: {(_b2 - _b1) * 1000:.2f}ms, "
            f"(no files to decode), total: {(_b2 - _b0) * 1000:.2f}ms"
        )
        return results

    # ── Try fused pipeline: decompress (GIL-free) → RLE+gather (GIL-free) ──
    plan = pre_plan
    if plan is None:
        plan = prepare_pipeline_plan(file_metas)

    _used_pipeline = False
    if plan is not None:
        try:
            # Step 1: decompress via C++ snappy (GIL-free)
            snappy_ext = _get_snappy_cuda_ext()
            decomp_pages = snappy_ext.batch_snappy_decompress(
                gpu_buffer,
                plan["page_offsets"],
                plan["page_comp_sizes"],
                plan["page_uncomp_sizes"],
            )
            _b3 = _time.perf_counter()

            # Step 2: fused RLE + gather (GIL-free)
            pipe_ext = _get_pipeline_cuda_ext()
            kern_out = pipe_ext.rle_gather_pipeline(
                decomp_pages,
                plan["data_page_indices"],
                plan["rle_max_rep"],
                plan["rle_max_def"],
                plan["rle_num_vals"],
                plan["rle_out_starts"],
                plan["total_rle_values"],
                plan["xyz_dict_page_indices"],
                plan["xyz_dict_byte_offsets"],
                plan["total_xyz_dict_bytes"],
                plan["ts_dict_page_indices"],
                plan["ts_dict_byte_offsets"],
                plan["total_ts_dict_bytes"],
                plan["file_info_raw"],
                plan["n_files"],
                plan["total_xyz_values"],
                plan["total_ts_values"],
                plan["total_rows"],
            )
            k_verts, k_ts, k_row_off, k_lengths, k_valid_mask, k_valid_cum, k_counts = kern_out
            file_order = plan["file_order"]
            file_info_list = plan["file_info_list"]
            all_num_vals = plan["all_num_vals"]
            n_files = plan["n_files"]
            total_xyz = plan["total_xyz_values"]
            total_ts = plan["total_ts_values"]
            total_rows = plan["total_rows"]
            _b3b = _time.perf_counter()
            _used_pipeline = True
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    [B] fused pipeline failed ({e}), falling back to separate kernels")

    if not _used_pipeline:
        # ── PHASE 2: Collect ALL pages for ONE nvcomp batch ───────────────
        all_pages: List[PageInfo] = []
        page_labels: List[Tuple[str, str, str]] = []

        for pq_basename, (page_index, parquet_offset, spec, _) in file_metas.items():
            for col_path in [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]:
                col = page_index.columns[col_path]
                if col.dict_page is not None:
                    all_pages.append(
                        PageInfo(
                            page_type=col.dict_page.page_type,
                            data_offset=col.dict_page.data_offset + parquet_offset,
                            compressed_size=col.dict_page.compressed_size,
                            uncompressed_size=col.dict_page.uncompressed_size,
                        )
                    )
                    page_labels.append((pq_basename, col_path, "dict"))
                for i, dp in enumerate(col.data_pages):
                    all_pages.append(
                        PageInfo(
                            page_type=dp.page_type,
                            data_offset=dp.data_offset + parquet_offset,
                            compressed_size=dp.compressed_size,
                            uncompressed_size=dp.uncompressed_size,
                        )
                    )
                    page_labels.append((pq_basename, col_path, f"data_{i}"))

        # ── PHASE 3: ONE batch decompress (C++ GIL-free, fallback to Python) ──
        non_empty = [(i, p) for i, p in enumerate(all_pages) if p.compressed_size > 0]
        decomp_map: Dict[Tuple[str, str, str], Tensor] = {}
        _used_cpp = False

        if non_empty:
            try:
                snappy_ext = _get_snappy_cuda_ext()
                ne_offsets = torch.tensor([p.data_offset for _, p in non_empty], dtype=torch.int64)
                ne_comp = torch.tensor([p.compressed_size for _, p in non_empty], dtype=torch.int64)
                ne_uncomp = torch.tensor([p.uncompressed_size for _, p in non_empty], dtype=torch.int64)
                ne_tensors = snappy_ext.batch_snappy_decompress(gpu_buffer, ne_offsets, ne_comp, ne_uncomp)
                ne_iter = iter(ne_tensors)
                empty = torch.empty(0, dtype=torch.uint8, device=device)
                for i, label in enumerate(page_labels):
                    if all_pages[i].compressed_size > 0:
                        decomp_map[label] = next(ne_iter)
                    else:
                        decomp_map[label] = empty
                _used_cpp = True
            except Exception:
                decomp_map.clear()

        if not _used_cpp:
            decompressed = batch_decompress_pages(gpu_buffer, all_pages)
            for label, dtensor in zip(page_labels, decompressed):
                decomp_map[label] = dtensor

        _b3 = _time.perf_counter()

        # ── PHASE 4: ONE RLE decode for ALL data pages (16 streams) ───────
        all_data_tensors: List[Tensor] = []
        all_max_reps: List[int] = []
        all_max_defs: List[int] = []
        all_num_vals: List[int] = []
        stream_file_order: List[str] = []

        for pq_basename, (page_index, _, spec, _) in file_metas.items():
            for col_path in [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]:
                col_meta = page_index.columns[col_path]
                all_data_tensors.append(decomp_map[(pq_basename, col_path, "data_0")])
                all_max_reps.append(col_meta.max_repetition_level)
                all_max_defs.append(col_meta.max_definition_level)
                all_num_vals.append(col_meta.num_values)
                stream_file_order.append(pq_basename)

        out_rep, out_def, out_idx = decode_data_pages_gpu(
            all_data_tensors,
            all_max_reps,
            all_max_defs,
            all_num_vals,
            device,
        )

        _b4 = _time.perf_counter()

        out_starts: List[int] = []
        s = 0
        for nv in all_num_vals:
            out_starts.append(s)
            s += nv

        # ── PHASE 5: Fused CUDA gather + row analysis (ONE kernel) ────
        gf_ext = _get_gather_cuda_ext()

        dict_xyz_parts: List[Tensor] = []
        dict_ts_parts: List[Tensor] = []
        file_info_list = []
        file_order: List[str] = []

        total_xyz = 0
        total_ts = 0
        total_rows = 0
        row_off_cursor = 0
        dict_xyz_off = 0
        dict_ts_off = 0

        stream_idx = 0
        for pq_basename, (page_index, _, spec, min_pts) in file_metas.items():
            si_x = stream_idx
            si_y = stream_idx + 1
            si_z = stream_idx + 2
            si_ts = stream_idx + 3
            n_xyz = all_num_vals[si_x]
            n_rows = all_num_vals[si_ts]

            dx = decomp_map[(pq_basename, spec.x_path, "dict")].view(torch.float32)
            dy = decomp_map[(pq_basename, spec.y_path, "dict")].view(torch.float32)
            dz = decomp_map[(pq_basename, spec.z_path, "dict")].view(torch.float32)
            dts = decomp_map[(pq_basename, spec.ts_path, "dict")].view(torch.int64)

            dict_xyz_parts.extend([dx, dy, dz])
            dict_ts_parts.append(dts)

            fi_vals = [
                out_starts[si_x],
                out_starts[si_y],
                out_starts[si_z],
                out_starts[si_ts],
                out_starts[si_x],
                n_xyz,
                n_rows,
                min_pts,
                dict_xyz_off,
                dict_xyz_off + len(dx),
                dict_xyz_off + len(dx) + len(dy),
                dict_ts_off,
                total_xyz,
                total_ts,
                row_off_cursor,
                total_rows,
                len(file_order) * 2,
            ]
            file_info_list.extend(fi_vals)
            file_order.append(pq_basename)

            dict_xyz_off += len(dx) + len(dy) + len(dz)
            dict_ts_off += len(dts)
            total_xyz += n_xyz
            total_ts += n_rows
            total_rows += n_rows
            row_off_cursor += n_rows + 1
            stream_idx += 4

        n_files = len(file_order)

        flat_dict_xyz = (
            torch.cat(dict_xyz_parts) if dict_xyz_parts else torch.empty(0, dtype=torch.float32, device=device)
        )
        flat_dict_ts = torch.cat(dict_ts_parts) if dict_ts_parts else torch.empty(0, dtype=torch.int64, device=device)

        fi_tensor = torch.tensor(file_info_list, dtype=torch.int32)

        kern_out = gf_ext.gather_and_analyze(
            out_rep,
            out_idx,
            flat_dict_xyz,
            flat_dict_ts,
            fi_tensor,
            n_files,
            total_xyz,
            total_ts,
            total_rows,
        )
        k_verts, k_ts, k_row_off, k_lengths, k_valid_mask, k_valid_cum, k_counts = kern_out

    # ── PHASE 6: Async postprocess (NO sync — defer to Phase F) ──
    # Slice kernel outputs using CPU-side file_info. Skip k_counts check and
    # unique_consecutive here; Phase F computes them after a natural sync point,
    # when the GPU pipeline is guaranteed complete.
    FIELDS_PER_FILE = 17
    xyz_cursor = 0
    ts_cursor = 0
    roff_cursor = 0

    for fi_idx, pq_basename in enumerate(file_order):
        fi_base = fi_idx * FIELDS_PER_FILE
        n_xyz_f = file_info_list[fi_base + 5]
        n_rows = file_info_list[fi_base + 6]

        if n_rows == 0:
            results[pq_basename] = None
            xyz_cursor += n_xyz_f
            ts_cursor += n_rows
            roff_cursor += n_rows + 1
            continue

        results[pq_basename] = FlatPolylineData(
            timestamps_us=k_ts[ts_cursor : ts_cursor + n_rows],
            vertices=k_verts[xyz_cursor : xyz_cursor + n_xyz_f],
            row_offsets=k_row_off[roff_cursor : roff_cursor + n_rows + 1],
            unique_timestamps=None,
            ts_counts_prefix_sum=None,
        )

        xyz_cursor += n_xyz_f
        ts_cursor += n_rows
        roff_cursor += n_rows + 1

    _b5 = _time.perf_counter()
    if _used_pipeline:
        print(
            f"    [B breakdown] H2D: {(_b1 - _b0) * 1000:.2f}ms, "
            f"cpu_scan: {(_b2 - _b1) * 1000:.2f}ms, "
            f"decompress: {(_b3 - _b2) * 1000:.2f}ms, "
            f"rle+gather(C++): {(_b3b - _b3) * 1000:.2f}ms, "
            f"slice: {(_b5 - _b3b) * 1000:.2f}ms, "
            f"total: {(_b5 - _b0) * 1000:.2f}ms [ASYNC]"
        )
    else:
        print(
            f"    [B breakdown] H2D: {(_b1 - _b0) * 1000:.2f}ms, "
            f"cpu_scan: {(_b2 - _b1) * 1000:.2f}ms, "
            f"decompress: {(_b3 - _b2) * 1000:.2f}ms, "
            f"rle_decode: {(_b4 - _b3) * 1000:.2f}ms, "
            f"gather+compact: {(_b5 - _b4) * 1000:.2f}ms, "
            f"total: {(_b5 - _b0) * 1000:.2f}ms"
        )

    return results


def is_gpu_parquet_available() -> bool:
    """Check if the GPU-native parquet pipeline is available (nvcomp + CUDA ext)."""
    return _nvcomp is not None


# ---------------------------------------------------------------------------
# Tar batch reader
# ---------------------------------------------------------------------------


@dataclass
class TarFileEntry:
    """Location of a file within a tar byte buffer."""

    name: str
    offset: int  # byte offset in the tar buffer
    size: int


def scan_tar_members(buf, buf_len: Optional[int] = None) -> List[TarFileEntry]:
    """Parse tar headers to locate member files.

    Accepts bytes, memoryview, or numpy uint8 arrays (e.g. pinned buffer
    slices).  Only 512-byte headers are converted to bytes for parsing;
    file content is never copied.

    Tar format: 512-byte headers followed by file data (rounded up to
    512-byte blocks).  No compression (we only handle .tar, not .tar.gz).
    """
    if buf_len is None:
        buf_len = len(buf)
    entries = []
    pos = 0

    while pos + 512 <= buf_len:
        header = bytes(buf[pos : pos + 512])

        if header == b"\x00" * 512:
            break

        name = header[0:100].split(b"\x00", 1)[0].decode("utf-8", errors="replace")

        size_str = header[124:136].split(b"\x00", 1)[0].strip()
        if not size_str:
            break
        file_size = int(size_str, 8)

        type_flag = header[156:157]

        prefix = header[345:500].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        if prefix:
            name = prefix + "/" + name

        data_offset = pos + 512

        if type_flag in (b"0", b"\x00", b""):
            entries.append(TarFileEntry(name=name, offset=data_offset, size=file_size))

        pos = data_offset + ((file_size + 511) // 512) * 512

    return entries


_pinned_bufs: List[Optional[Tensor]] = [None, None]
_active_buf: int = 0
_prefetch_future = None
_prefetch_path: Optional[str] = None
_prefetch_executor = None


def _ensure_pinned(buf_idx: int, nbytes: int) -> None:
    """Ensure _pinned_bufs[buf_idx] is at least nbytes large."""
    global _pinned_bufs
    if _pinned_bufs[buf_idx] is None or _pinned_bufs[buf_idx].size(0) < nbytes:
        new_size = max(nbytes, 16 * 1024 * 1024)
        _pinned_bufs[buf_idx] = torch.empty(new_size, dtype=torch.uint8, pin_memory=True)


def _read_tar_into(tar_path: str, buf_idx: int) -> Tuple[Tensor, List[TarFileEntry]]:
    """Read tar into a specific pinned buffer slot."""
    import os

    nbytes = os.path.getsize(tar_path)
    _ensure_pinned(buf_idx, nbytes)
    pinned = _pinned_bufs[buf_idx][:nbytes]
    pin_mv = memoryview(pinned.numpy())
    with open(tar_path, "rb") as f:
        f.readinto(pin_mv)
    entries = scan_tar_members(pin_mv, nbytes)
    return pinned, entries


def prefetch_tar(tar_path: str) -> None:
    """Start reading a tar file into pinned memory on a background thread.

    Uses the inactive pinned buffer (double buffering) so prefetch
    doesn't conflict with any in-flight scene processing on the active buffer.
    Call after load_av2_scene (or via its prefetch_next param) to hide I/O
    latency behind downstream processing.
    """
    global _prefetch_future, _prefetch_path, _prefetch_executor
    from concurrent.futures import ThreadPoolExecutor

    if _prefetch_executor is None:
        _prefetch_executor = ThreadPoolExecutor(max_workers=1)
    if _prefetch_future is not None:
        _prefetch_future.result()
    buf_idx = 1 - _active_buf
    _prefetch_path = tar_path
    _prefetch_future = _prefetch_executor.submit(_read_tar_into, tar_path, buf_idx)


def read_tar_to_pinned_buffer(tar_path: str) -> Tuple[Tensor, List[TarFileEntry]]:
    """Read a tar file into a reusable pinned-memory CPU tensor.

    If the path was previously prefetched via prefetch_tar(), returns the
    pre-read result immediately (zero I/O wait). Otherwise reads synchronously.
    Uses double-buffered pinned memory to avoid cudaHostAlloc overhead.
    """
    global _active_buf, _prefetch_future, _prefetch_path

    if _prefetch_future is not None and _prefetch_path == tar_path:
        pinned, entries = _prefetch_future.result()
        _active_buf = 1 - _active_buf
        _prefetch_future = None
        _prefetch_path = None
        return pinned, entries

    _prefetch_future = None
    _prefetch_path = None
    pinned, entries = _read_tar_into(tar_path, _active_buf)
    return pinned, entries


def read_tars_to_pinned_buffer(
    tar_paths: List[str],
) -> Tuple[Tensor, List[Tuple[str, int, List[TarFileEntry]]]]:
    """Read multiple tar files into a single pinned-memory buffer.

    Uses threaded I/O to overlap Lustre reads and file.readinto() to
    write directly into pinned memory (no intermediate bytes copies).

    Returns:
        (pinned_tensor, tar_info_list) where tar_info_list contains
        (tar_path, base_offset, entries) for each tar.  All offsets
        in entries are relative to the start of the combined buffer.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    sizes = [os.path.getsize(p) for p in tar_paths]
    offsets: List[int] = []
    total = 0
    for s in sizes:
        offsets.append(total)
        total += s

    _ensure_pinned(_active_buf, total)
    pinned = _pinned_bufs[_active_buf][:total]
    pin_np = pinned.numpy()

    def _read_one(idx: int) -> Tuple[str, int, List[TarFileEntry]]:
        off, sz = offsets[idx], sizes[idx]
        mv = memoryview(pin_np[off : off + sz])
        with open(tar_paths[idx], "rb") as f:
            f.readinto(mv)
        entries = scan_tar_members(mv, sz)
        for e in entries:
            e.offset += off
        return (tar_paths[idx], off, entries)

    with ThreadPoolExecutor(max_workers=min(len(tar_paths), 8)) as pool:
        tar_info = list(pool.map(_read_one, range(len(tar_paths))))

    return pinned, tar_info


# ---------------------------------------------------------------------------
# GpuParquetDecoder -- stateful decoder with pre-allocated scratch buffers
# ---------------------------------------------------------------------------


class GpuParquetDecoder:
    """Pre-allocated GPU-native parquet decoder with batched kernel launches.

    Architecture (all GPU work batched across scenes in single launches):
      Phase 1: Read all tars into combined pinned buffer (CPU, sequential)
      Phase 2: Single async GPU upload
      Phase 3: ONE nvcomp decompress call for ALL pages across ALL scenes
      Phase 4: ONE CUDA SMEM kernel for ALL data pages
      Phase 5: Per-file gather + assembly (light GPU ops)

    This ensures GPU kernels see ALL work at once, giving sub-linear
    batch scaling through massive thread-level parallelism.

    Example::

        decoder = GpuParquetDecoder(device)
        cache: Dict[str, Dict] = {}

        for step in training_loop:
            uncached = [p for p in batch_paths if p not in cache]
            if uncached:
                scenes = decoder.load_scenes(uncached)
                for path, scene in zip(uncached, scenes):
                    cache[path] = scene
            batch = [cache[p] for p in batch_paths]
    """

    def __init__(
        self,
        device: torch.device,
        initial_tar_mb: int = 64,
        initial_decode_values: int = 512_000,
    ):
        self.device = device

        init_bytes = initial_tar_mb * 1024 * 1024

        self._pin = torch.empty(init_bytes, dtype=torch.uint8, pin_memory=True)
        self._gpu = torch.empty(init_bytes, dtype=torch.uint8, device=device)

        dc = 4 * 1024 * 1024
        self._dcat = torch.empty(dc, dtype=torch.uint8, device=device)
        self._dcat_cap = dc

        mp = 128
        self._m_off = torch.empty(mp, dtype=torch.int32, device=device)
        self._m_len = torch.empty(mp, dtype=torch.int32, device=device)
        self._m_rep = torch.empty(mp, dtype=torch.int32, device=device)
        self._m_def = torch.empty(mp, dtype=torch.int32, device=device)
        self._m_nv = torch.empty(mp, dtype=torch.int32, device=device)
        self._m_po = torch.empty(mp, dtype=torch.int32, device=device)
        self._meta_cap = mp

        _get_rle_cuda_ext()

    # -- capacity helpers (amortized doubling) --

    def _grow_buf(self, attr: str, needed: int, **kw):
        cur = getattr(self, attr)
        if cur.shape[0] >= needed:
            return
        setattr(self, attr, torch.empty(max(needed, cur.shape[0] * 2), **kw))

    def _ensure_staging(self, nbytes: int):
        self._grow_buf("_pin", nbytes, dtype=torch.uint8, pin_memory=True)
        self._grow_buf("_gpu", nbytes, dtype=torch.uint8, device=self.device)

    def _ensure_dcat(self, nbytes: int):
        if nbytes <= self._dcat_cap:
            return
        new = max(nbytes, self._dcat_cap * 2)
        self._dcat = torch.empty(new, dtype=torch.uint8, device=self.device)
        self._dcat_cap = new

    def _ensure_meta(self, n_pages: int):
        if n_pages <= self._meta_cap:
            return
        new = max(n_pages, self._meta_cap * 2)
        for attr in ("_m_off", "_m_len", "_m_rep", "_m_def", "_m_nv", "_m_po"):
            setattr(self, attr, torch.empty(new, dtype=torch.int32, device=self.device))
        self._meta_cap = new

    # -- public API --

    def load_scene(
        self,
        tar_path: str,
        min_points_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, object]:
        """Load a single scene.  Convenience wrapper around ``load_scenes``."""
        return self.load_scenes([tar_path], min_points_map)[0]

    def load_scenes(
        self,
        tar_paths: List[str],
        min_points_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, object]]:
        """Load a variable-size batch of scenes from tar files.

        All GPU work (nvcomp decompress, CUDA SMEM decode) is batched
        into single kernel launches across ALL scenes for maximal parallelism.

        Args:
            tar_paths: 1 … N paths to AV2 scene tar files.
            min_points_map: Optional per-parquet min-point override.

        Returns:
            List of dicts, one per scene, mapping parquet basename to
            ``FlatPolylineData`` (or ``None`` if absent / filtered out).
        """
        if not tar_paths:
            return []

        if min_points_map is None:
            min_points_map = {
                "cf_road_boundary.parquet": 2,
                "dw_lane_line.parquet": 2,
                "cf_crosswalks.parquet": 3,
                "cf_static_obstacle.parquet": 2,
            }

        # ---- Phase 1: threaded direct-to-pinned tar reads ----
        import os
        from concurrent.futures import ThreadPoolExecutor

        # 1a: get file sizes (cheap syscalls, primes Lustre metadata cache)
        sizes = [os.path.getsize(tp) for tp in tar_paths]
        tar_offsets: List[int] = []
        total_bytes = 0
        for s in sizes:
            tar_offsets.append(total_bytes)
            total_bytes += s

        # 1b: ensure pinned + GPU buffers are large enough
        self._ensure_staging(total_bytes)
        pin_np = self._pin[:total_bytes].numpy()

        # 1c: threaded readinto + tar header scan (parallel, no locks needed)
        entries_list: List[Optional[List[TarFileEntry]]] = [None] * len(tar_paths)

        def _read_tar(idx: int) -> None:
            off, sz = tar_offsets[idx], sizes[idx]
            mv = memoryview(pin_np[off : off + sz])
            with open(tar_paths[idx], "rb") as f:
                f.readinto(mv)
            ents = scan_tar_members(mv, sz)
            for e in ents:
                e.offset += off
            entries_list[idx] = ents

        n_workers = min(len(tar_paths), 8)
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_read_tar, range(len(tar_paths))))
        else:
            _read_tar(0)

        # 1d: scan parquet metadata (sequential -- needs ordered index building)
        file_metas: List[dict] = []
        all_pages: List[PageInfo] = []
        dp_max_reps: List[int] = []
        dp_max_defs: List[int] = []
        dp_num_vals: List[int] = []
        dp_decomp_idx: List[int] = []

        for scene_idx, (tar_off, entries) in enumerate(zip(tar_offsets, entries_list)):
            entry_map: Dict[str, TarFileEntry] = {}
            for e in entries:
                base = e.name.rsplit("/", 1)[-1] if "/" in e.name else e.name
                entry_map[base] = e

            for pq_name, spec in POLYLINE_SPECS.items():
                entry = entry_map.get(pq_name)
                if entry is None:
                    continue

                pq_bytes = bytes(pin_np[entry.offset : entry.offset + entry.size])
                pq_gpu_off = entry.offset
                cols = [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]
                page_index = scan_parquet_pages(pq_bytes, columns=cols)

                if any(c not in page_index.columns for c in cols):
                    continue
                if any(not page_index.columns[c].data_pages for c in cols):
                    continue

                fm: dict = {
                    "scene_idx": scene_idx,
                    "pq_name": pq_name,
                    "spec": spec,
                    "min_pts": min_points_map.get(pq_name, 2),
                    "page_index": page_index,
                    "decomp_map": {},
                    "data_page_global": {},
                }

                for cp in cols:
                    ci = page_index.columns[cp]
                    if ci.dict_page is not None:
                        fm["decomp_map"][(cp, "dict")] = len(all_pages)
                        all_pages.append(
                            PageInfo(
                                page_type=ci.dict_page.page_type,
                                data_offset=ci.dict_page.data_offset + pq_gpu_off,
                                compressed_size=ci.dict_page.compressed_size,
                                uncompressed_size=ci.dict_page.uncompressed_size,
                            )
                        )
                    for j, dp in enumerate(ci.data_pages):
                        decomp_i = len(all_pages)
                        fm["decomp_map"][(cp, f"data_{j}")] = decomp_i
                        all_pages.append(
                            PageInfo(
                                page_type=dp.page_type,
                                data_offset=dp.data_offset + pq_gpu_off,
                                compressed_size=dp.compressed_size,
                                uncompressed_size=dp.uncompressed_size,
                            )
                        )
                        if j == 0:
                            fm["data_page_global"][cp] = len(dp_max_reps)
                            dp_max_reps.append(ci.max_repetition_level)
                            dp_max_defs.append(ci.max_definition_level)
                            dp_num_vals.append(ci.num_values)
                            dp_decomp_idx.append(decomp_i)

                file_metas.append(fm)

        # ---- Phase 2: single async GPU upload ----
        self._gpu[:total_bytes].copy_(self._pin[:total_bytes], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        gpu_buf = self._gpu[:total_bytes]

        # ---- Phase 3: ONE nvcomp decompress for ALL pages ----
        empty_results: List[Dict[str, object]] = [{pq: None for pq in POLYLINE_SPECS} for _ in tar_paths]
        if not all_pages:
            return empty_results

        decompressed = batch_decompress_pages(gpu_buf, all_pages)

        n_data_pages = len(dp_max_reps)
        if n_data_pages == 0:
            return empty_results

        # ---- Phase 4: ONE CUDA SMEM kernel for ALL data pages ----
        total_values = sum(dp_num_vals)

        dp_tensors = [decompressed[dp_decomp_idx[i]] for i in range(n_data_pages)]
        total_decomp = sum(t.shape[0] for t in dp_tensors) + 4
        self._ensure_dcat(total_decomp)
        d_offsets: List[int] = []
        pos = 0
        for t in dp_tensors:
            d_offsets.append(pos)
            self._dcat[pos : pos + t.shape[0]] = t
            pos += t.shape[0]
        self._dcat[pos : pos + 4] = 0
        concat_data = self._dcat[: pos + 4]

        page_out_starts: List[int] = []
        s = 0
        for nv in dp_num_vals:
            page_out_starts.append(s)
            s += nv

        self._ensure_meta(n_data_pages)
        self._m_off[:n_data_pages].copy_(torch.tensor(d_offsets, dtype=torch.int32))
        self._m_len[:n_data_pages].copy_(torch.tensor([t.shape[0] for t in dp_tensors], dtype=torch.int32))
        self._m_rep[:n_data_pages].copy_(torch.tensor(dp_max_reps, dtype=torch.int32))
        self._m_def[:n_data_pages].copy_(torch.tensor(dp_max_defs, dtype=torch.int32))
        self._m_nv[:n_data_pages].copy_(torch.tensor(dp_num_vals, dtype=torch.int32))
        self._m_po[:n_data_pages].copy_(torch.tensor(page_out_starts, dtype=torch.int32))

        ext = _get_rle_cuda_ext()
        output = ext.decode_rle_streams(
            concat_data,
            self._m_off[:n_data_pages],
            self._m_len[:n_data_pages],
            self._m_rep[:n_data_pages],
            self._m_def[:n_data_pages],
            self._m_nv[:n_data_pages],
            self._m_po[:n_data_pages],
            total_values,
        )

        out_rep = output[:total_values]
        out_def = output[total_values : 2 * total_values]
        out_idx = output[2 * total_values :]

        # ---- Phase 5: per-file gather + assembly ----
        from .clipgt import FlatPolylineData

        results: List[Dict[str, object]] = [{pq: None for pq in POLYLINE_SPECS} for _ in tar_paths]

        for fm in file_metas:
            scene_idx = fm["scene_idx"]
            pq_name = fm["pq_name"]
            spec = fm["spec"]
            min_pts = fm["min_pts"]
            page_index = fm["page_index"]
            decomp_map = fm["decomp_map"]
            data_page_global = fm["data_page_global"]

            cols = [spec.x_path, spec.y_path, spec.z_path, spec.ts_path]

            xyz: Dict[str, Tensor] = {}
            rep_gpu: Optional[Tensor] = None
            for cp in cols[:3]:
                cm = page_index.columns[cp]
                dv = decompressed[decomp_map[(cp, "dict")]].view(torch.float32)
                dp_gidx = data_page_global[cp]
                dp_start = page_out_starts[dp_gidx]
                nv = dp_num_vals[dp_gidx]
                pi = out_idx[dp_start : dp_start + nv].to(torch.int64)
                xyz[cp] = torch.index_select(dv, 0, pi)
                if rep_gpu is None and cm.max_repetition_level > 0:
                    rep_gpu = out_rep[dp_start : dp_start + nv]

            ts_dv = decompressed[decomp_map[(spec.ts_path, "dict")]].view(torch.int64)
            dp_gidx = data_page_global[spec.ts_path]
            dp_start = page_out_starts[dp_gidx]
            nv = dp_num_vals[dp_gidx]
            ts_pi = out_idx[dp_start : dp_start + nv].to(torch.int64)
            ts = torch.index_select(ts_dv, 0, ts_pi)

            if rep_gpu is None:
                continue

            row_off = rep_levels_to_offsets_gpu(rep_gpu)
            n_rows = row_off.shape[0] - 1
            vertices = torch.stack([xyz[cols[0]], xyz[cols[1]], xyz[cols[2]]], dim=1)

            lengths = row_off[1:] - row_off[:-1]
            valid = lengths >= min_pts
            nan_mask = torch.isnan(vertices).any(dim=1)
            if nan_mask.any():
                vi = torch.arange(vertices.shape[0], device=self.device)
                rid = torch.searchsorted(row_off, vi, right=True) - 1
                rid.clamp_(0, n_rows - 1)
                nc = torch.zeros(n_rows, dtype=torch.int32, device=self.device)
                nc.scatter_add_(
                    0,
                    rid[nan_mask],
                    torch.ones(nan_mask.sum(), dtype=torch.int32, device=self.device),
                )
                valid = valid & (nc == 0)

            if not valid.any():
                continue

            vidx = torch.where(valid)[0]
            vs = row_off[:-1][vidx]
            vl = lengths[vidx]
            vts = ts[vidx].to(torch.int64)

            tv = vl.sum().item()
            if vidx.shape[0] == n_rows and tv == vertices.shape[0]:
                cv = vertices
            else:
                cl = torch.cumsum(vl, dim=0)
                ns = torch.zeros_like(cl)
                if cl.shape[0] > 1:
                    ns[1:] = cl[:-1]
                w = torch.arange(tv, device=self.device) - ns.repeat_interleave(vl)
                g = vs.repeat_interleave(vl) + w
                cv = vertices[g.long()]

            no = torch.zeros(vidx.shape[0] + 1, dtype=torch.int32, device=self.device)
            torch.cumsum(vl, dim=0, out=no[1:])

            results[scene_idx][pq_name] = FlatPolylineData(
                timestamps_us=vts,
                vertices=cv,
                row_offsets=no,
            )

        return results
