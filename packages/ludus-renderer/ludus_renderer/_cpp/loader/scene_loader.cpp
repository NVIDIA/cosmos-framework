// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#define SCENE_LOADER_BUILD
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <chrono>
#include <cmath>
#include <sstream>

#include "json_minimal.h"

// Forward declarations of existing C++ functions
extern std::vector<torch::Tensor> batch_snappy_decompress(
    torch::Tensor gpu_buffer,
    torch::Tensor data_offsets,
    torch::Tensor comp_sizes,
    torch::Tensor uncomp_sizes);

extern std::vector<torch::Tensor> rle_gather_pipeline(
    std::vector<torch::Tensor> decompressed_pages,
    torch::Tensor data_page_indices,
    torch::Tensor rle_max_rep,
    torch::Tensor rle_max_def,
    torch::Tensor rle_num_vals,
    torch::Tensor rle_out_starts,
    int total_rle_values,
    torch::Tensor xyz_dict_page_indices,
    torch::Tensor xyz_dict_byte_offsets,
    int total_xyz_dict_bytes,
    torch::Tensor ts_dict_page_indices,
    torch::Tensor ts_dict_byte_offsets,
    int total_ts_dict_bytes,
    torch::Tensor file_info_raw,
    int n_files,
    int total_xyz_values,
    int total_ts_values,
    int total_rows);

extern torch::Tensor scan_parquet_pages_cpp(
    torch::Tensor pinned_buf,
    int64_t file_offset,
    int64_t file_size,
    std::vector<std::string> column_paths);

extern void obs_yaw_to_quat(torch::Tensor packed);
extern std::vector<torch::Tensor> obs_group_and_gather(
    torch::Tensor sorted_id, torch::Tensor sorted_ts,
    torch::Tensor sorted_pack, torch::Tensor sorted_class,
    bool use_class);

extern std::vector<torch::Tensor> unique_consecutive_segments(
    torch::Tensor ts, torch::Tensor seg_offsets,
    torch::Tensor seg_sizes, int n_segs);

extern std::vector<torch::Tensor> compute_camera_params(
    torch::Tensor poly_coeffs,
    torch::Tensor poly_lengths,
    torch::Tensor is_bw_poly,
    torch::Tensor cx_raw,
    torch::Tensor cy_raw,
    torch::Tensor img_w_raw,
    torch::Tensor img_h_raw,
    torch::Tensor cx_scaled,
    torch::Tensor cy_scaled,
    torch::Tensor img_w_scaled,
    torch::Tensor img_h_scaled,
    torch::Tensor poly_scale_t);

// ---------------------------------------------------------------------------
// Persistent thread pool (created once, lives for process lifetime)
// ---------------------------------------------------------------------------
namespace {

class ThreadPool {
public:
    ThreadPool(size_t n) {
        for (size_t i = 0; i < n; i++) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mu_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F>
    std::future<typename std::result_of<F()>::type> submit(F&& f) {
        using R = typename std::result_of<F()>::type;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

ThreadPool& get_pool() {
    static ThreadPool pool(8);
    return pool;
}

// AV2 polyline file specs (hardcoded column paths)
struct PolylineFileSpec {
    std::string basename;
    std::string x_path, y_path, z_path;
    std::string ts_path;
    int min_points;
};

static const PolylineFileSpec FILE_SPECS[] = {
    {"cf_road_boundary.parquet",
     "cf_road_boundary.road_boundary_polyline.list.element.x",
     "cf_road_boundary.road_boundary_polyline.list.element.y",
     "cf_road_boundary.road_boundary_polyline.list.element.z",
     "key.timestamp_micros", 2},
    {"dw_lane_line.parquet",
     "dw_lane_line.points.list.element.x",
     "dw_lane_line.points.list.element.y",
     "dw_lane_line.points.list.element.z",
     "key.timestamp_micros", 2},
    {"cf_crosswalks.parquet",
     "cf_crosswalks.crosswalk_area.list.element.x",
     "cf_crosswalks.crosswalk_area.list.element.y",
     "cf_crosswalks.crosswalk_area.list.element.z",
     "key.timestamp_micros", 3},
    {"cf_static_obstacle.parquet",
     "cf_static_obstacle.boundary_points.list.element.x",
     "cf_static_obstacle.boundary_points.list.element.y",
     "cf_static_obstacle.boundary_points.list.element.z",
     "key.timestamp_micros", 2},
};
constexpr int N_FILE_SPECS = 4;

// Scan result for one column (parsed from flat tensor output)
struct ColScan {
    int64_t num_values;
    int max_rep, max_def;
    int physical_type; // parquet: 1=INT32,2=INT64,4=FLOAT,5=DOUBLE,6=BYTE_ARRAY
    bool has_dict;
    int64_t dict_data_offset, dict_comp, dict_uncomp;
    struct DataPage { int64_t data_offset, comp, uncomp; };
    std::vector<DataPage> data_pages;

    int value_byte_width() const {
        switch (physical_type) {
            case 1: return 4;  // INT32
            case 2: return 8;  // INT64
            case 4: return 4;  // FLOAT
            case 5: return 8;  // DOUBLE
            default: return 0; // BYTE_ARRAY or unknown
        }
    }
};

// Parse scan_parquet_pages_cpp output for one file (n_cols columns)
bool unpack_scan(const int64_t* data, int64_t len, int n_cols,
                 std::vector<ColScan>& out) {
    out.clear();
    int64_t pos = 0;
    for (int c = 0; c < n_cols; c++) {
        if (pos >= len || data[pos] == -1) return false;
        ColScan cs;
        cs.num_values = data[pos++];
        cs.max_rep = (int)data[pos++];
        cs.max_def = (int)data[pos++];
        cs.physical_type = (int)data[pos++];
        cs.has_dict = (data[pos++] != 0);
        cs.dict_data_offset = data[pos++];
        cs.dict_comp = data[pos++];
        cs.dict_uncomp = data[pos++];
        int n_dp = (int)data[pos++];
        for (int d = 0; d < n_dp; d++) {
            ColScan::DataPage dp;
            dp.data_offset = data[pos++];
            dp.comp = data[pos++];
            dp.uncomp = data[pos++];
            cs.data_pages.push_back(dp);
        }
        out.push_back(std::move(cs));
    }
    return true;
}

// ── CPU Snappy decompressor (minimal, for small parquet pages) ──
bool cpu_snappy_decompress(const uint8_t* src, size_t src_len,
                           uint8_t* dst, size_t dst_cap, size_t& out_len) {
    size_t sp = 0;
    // Read uncompressed length (varint)
    uint64_t ulen = 0;
    int shift = 0;
    while (sp < src_len) {
        uint8_t b = src[sp++];
        ulen |= (uint64_t)(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    out_len = (size_t)ulen;
    if (out_len > dst_cap) return false;

    size_t dp = 0;
    while (sp < src_len && dp < out_len) {
        uint8_t tag = src[sp++];
        int type = tag & 3;
        if (type == 0) { // literal
            int len_m1 = (tag >> 2) & 0x3F;
            size_t length;
            if (len_m1 < 60) {
                length = (size_t)len_m1 + 1;
            } else {
                int extra = len_m1 - 59;
                uint32_t v = 0;
                std::memcpy(&v, src + sp, extra);
                sp += extra;
                length = (size_t)v + 1;
            }
            if (sp + length > src_len || dp + length > out_len) return false;
            std::memcpy(dst + dp, src + sp, length);
            sp += length;
            dp += length;
        } else {
            size_t length, offset;
            if (type == 1) { // copy with 1-byte offset
                length = (size_t)((tag >> 2) & 7) + 4;
                offset = ((size_t)(tag >> 5) << 8) | (size_t)src[sp++];
            } else if (type == 2) { // copy with 2-byte offset
                length = (size_t)((tag >> 2) & 0x3F) + 1;
                uint16_t off16;
                std::memcpy(&off16, src + sp, 2);
                sp += 2;
                offset = (size_t)off16;
            } else { // type == 3, copy with 4-byte offset
                length = (size_t)((tag >> 2) & 0x3F) + 1;
                uint32_t off32;
                std::memcpy(&off32, src + sp, 4);
                sp += 4;
                offset = (size_t)off32;
            }
            if (offset == 0 || offset > dp || dp + length > out_len) return false;
            const uint8_t* copy_src = dst + dp - offset;
            if (offset >= length) {
                std::memcpy(dst + dp, copy_src, length);
            } else {
                // Overlapping: repeat pattern (offset < length)
                for (size_t i = 0; i < length; i++) {
                    dst[dp + i] = copy_src[i % offset];
                }
            }
            dp += length;
        }
    }
    return dp == out_len;
}

// Read raw values from a PLAIN-encoded parquet data page.
// Handles Snappy decompression and rep/def level skipping.
// Returns pointer to values and count, using provided scratch buffer for decompression.
const uint8_t* read_plain_page_values(
    const uint8_t* pinned_ptr,
    int64_t data_offset,    // absolute offset in pinned_buf
    int64_t comp_size,
    int64_t uncomp_size,
    int max_rep, int max_def,
    std::vector<uint8_t>& scratch,
    int64_t& values_len
) {
    const uint8_t* page_data;
    int64_t page_len;

    if (comp_size != uncomp_size) {
        // Snappy-compressed
        scratch.resize(uncomp_size);
        size_t out_len;
        if (!cpu_snappy_decompress(pinned_ptr + data_offset, comp_size,
                                   scratch.data(), uncomp_size, out_len)) {
            values_len = 0;
            return nullptr;
        }
        page_data = scratch.data();
        page_len = (int64_t)out_len;
    } else {
        page_data = pinned_ptr + data_offset;
        page_len = uncomp_size;
    }

    // Skip rep/def levels (V1 data page format)
    int64_t pos = 0;
    if (max_rep > 0 && pos + 4 <= page_len) {
        uint32_t rep_len;
        std::memcpy(&rep_len, page_data + pos, 4);
        pos += 4 + rep_len;
    }
    if (max_def > 0 && pos + 4 <= page_len) {
        uint32_t def_len;
        std::memcpy(&def_len, page_data + pos, 4);
        pos += 4 + def_len;
    }

    values_len = page_len - pos;
    return page_data + pos;
}

// Decode RLE/bit-packed hybrid encoded dictionary indices.
// Format: [bit_width: 1 byte] then groups of RLE or bit-packed runs.
// For definition levels: [4-byte LE length] [bit_width: 1 byte] [data...]
// Returns decoded indices in 'out', returns number decoded.
int64_t decode_rle_bitpack_hybrid(const uint8_t* data, int64_t data_len,
                                   int bit_width, int64_t max_values,
                                   int32_t* out, int64_t out_cap) {
    if (bit_width == 0 || data_len == 0) {
        int64_t n = std::min(max_values, out_cap);
        std::memset(out, 0, n * sizeof(int32_t));
        return n;
    }
    int64_t pos = 0;
    int64_t written = 0;
    uint32_t mask = (1u << bit_width) - 1;

    while (pos < data_len && written < max_values) {
        uint32_t header = 0;
        int shift = 0;
        while (pos < data_len) {
            uint8_t b = data[pos++];
            header |= (uint32_t)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }

        if (header & 1) {
            int count = (header >> 1) * 8;
            int bytes_needed = (count * bit_width + 7) / 8;
            if (pos + bytes_needed > data_len) break;
            const uint8_t* packed = data + pos;
            int64_t to_decode = std::min((int64_t)count, max_values - written);

            if (bit_width == 8) {
                for (int64_t i = 0; i < to_decode; i++)
                    out[written++] = (int32_t)packed[i];
            } else if (bit_width == 12) {
                for (int64_t i = 0; i < to_decode; i++) {
                    int bit_pos = (int)(i * 12);
                    int byte_idx = bit_pos >> 3;
                    int bit_off = bit_pos & 7;
                    uint32_t v;
                    std::memcpy(&v, packed + byte_idx, 4);
                    out[written++] = (int32_t)((v >> bit_off) & 0xFFF);
                }
            } else if (bit_width == 16) {
                for (int64_t i = 0; i < to_decode; i++) {
                    uint16_t v;
                    std::memcpy(&v, packed + i * 2, 2);
                    out[written++] = (int32_t)v;
                }
            } else {
                int bit_pos = 0;
                for (int64_t i = 0; i < to_decode; i++) {
                    int byte_idx = bit_pos >> 3;
                    int bit_off = bit_pos & 7;
                    uint32_t v = 0;
                    std::memcpy(&v, packed + byte_idx,
                                std::min(4, bytes_needed - byte_idx));
                    out[written++] = (int32_t)((v >> bit_off) & mask);
                    bit_pos += bit_width;
                }
            }
            pos += bytes_needed;
        } else {
            int count = header >> 1;
            int value_bytes = (bit_width + 7) / 8;
            if (pos + value_bytes > data_len) break;
            uint32_t val = 0;
            std::memcpy(&val, data + pos, value_bytes);
            val &= mask;
            pos += value_bytes;
            int64_t n = std::min((int64_t)count, max_values - written);
            std::fill_n(out + written, n, (int32_t)val);
            written += n;
        }
    }
    return written;
}

// Decompress a page into pre-allocated buffer, or return pointer to raw data if uncompressed.
// Returns pointer to decompressed data, or nullptr on failure.
// If decompression was needed, data is in 'scratch' (which is resized as needed).
const uint8_t* decompress_page_into(const uint8_t* pinned_ptr,
                                     int64_t abs_offset, int64_t comp, int64_t uncomp,
                                     std::vector<uint8_t>& scratch) {
    const uint8_t* raw = pinned_ptr + abs_offset;
    if (comp == uncomp) return raw;
    if ((int64_t)scratch.size() < uncomp) scratch.resize(uncomp);
    size_t out_len;
    if (cpu_snappy_decompress(raw, comp, scratch.data(), uncomp, out_len) && (int64_t)out_len == uncomp) {
        return scratch.data();
    }
    return nullptr;
}

// Decode one RLE_DICTIONARY column directly into a typed output buffer.
// dict_data: pointer to decompressed dictionary values
// n_dict: number of dictionary entries
// value_size: bytes per value (4 for float/int32, 8 for int64/double)
// out_ptr: destination buffer (must hold col.num_values * value_size bytes)
// Returns number of values written.
int64_t decode_rle_dict_column_into(
    const uint8_t* pinned_ptr, int64_t base_offset,
    const ColScan& col, int value_size,
    const uint8_t* dict_data, int64_t n_dict,
    uint8_t* out_ptr, std::vector<uint8_t>& scratch,
    std::vector<int32_t>& idx_buf)
{
    int64_t total_written = 0;

    for (auto& dp : col.data_pages) {
        const uint8_t* page_data = decompress_page_into(
            pinned_ptr, dp.data_offset + base_offset, dp.comp, dp.uncomp, scratch);
        if (!page_data) continue;
        int64_t page_len = dp.uncomp;

        int64_t pos = 0;
        if (col.max_rep > 0 && pos + 4 <= page_len) {
            uint32_t rep_len;
            std::memcpy(&rep_len, page_data + pos, 4);
            pos += 4 + rep_len;
        }
        if (col.max_def > 0 && pos + 4 <= page_len) {
            uint32_t def_len;
            std::memcpy(&def_len, page_data + pos, 4);
            pos += 4 + def_len;
        }
        if (pos >= page_len) continue;

        int bit_width = (int)page_data[pos++];
        int64_t indices_len = page_len - pos;
        int64_t remaining = col.num_values - total_written;

        int64_t n_decoded = decode_rle_bitpack_hybrid(
            page_data + pos, indices_len, bit_width, remaining,
            idx_buf.data(), (int64_t)idx_buf.size());

        if (value_size == 4) {
            const float* dict_f = reinterpret_cast<const float*>(dict_data);
            float* out_f = reinterpret_cast<float*>(out_ptr + total_written * 4);
            for (int64_t i = 0; i < n_decoded; i++) {
                int32_t idx = idx_buf[i];
                out_f[i] = (idx >= 0 && idx < n_dict) ? dict_f[idx] : 0.0f;
            }
        } else if (value_size == 8) {
            const int64_t* dict_i = reinterpret_cast<const int64_t*>(dict_data);
            int64_t* out_i = reinterpret_cast<int64_t*>(out_ptr + total_written * 8);
            for (int64_t i = 0; i < n_decoded; i++) {
                int32_t idx = idx_buf[i];
                out_i[i] = (idx >= 0 && idx < n_dict) ? dict_i[idx] : 0;
            }
        } else {
            for (int64_t i = 0; i < n_decoded; i++) {
                int32_t idx = idx_buf[i];
                if (idx >= 0 && idx < n_dict) {
                    std::memcpy(out_ptr + (total_written + i) * value_size,
                                dict_data + idx * value_size, value_size);
                }
            }
        }
        total_written += n_decoded;
    }
    return total_written;
}

// Extract rig_json string from calibration_estimate.parquet in pinned buffer.
// Handles both PLAIN and DICTIONARY encoded BYTE_ARRAY columns.
std::string extract_rig_json_from_pinned(
    torch::Tensor pinned_buf, int64_t cal_off, int64_t cal_sz
) {
    std::vector<std::string> cal_cols = {"calibration_estimate.rig_json"};
    auto scan_result = scan_parquet_pages_cpp(pinned_buf, cal_off, cal_sz, cal_cols);
    auto* sdata = scan_result.data_ptr<int64_t>();
    int64_t slen = scan_result.size(0);

    std::vector<ColScan> cols;
    if (!unpack_scan(sdata, slen, 1, cols) || cols[0].data_pages.empty())
        return "";

    const uint8_t* pinned_ptr = pinned_buf.data_ptr<uint8_t>();
    std::vector<uint8_t> scratch;
    auto& col = cols[0];

    const uint8_t* vals;
    int64_t vals_len;

    if (col.has_dict) {
        // DICTIONARY encoding: the actual string is in the dictionary page.
        // For a single-row calibration table, the first dict entry is the value.
        int64_t dict_abs = col.dict_data_offset + cal_off;
        vals = read_plain_page_values(pinned_ptr, dict_abs,
            col.dict_comp, col.dict_uncomp, 0, 0, scratch, vals_len);
    } else {
        int64_t abs_off = col.data_pages[0].data_offset + cal_off;
        vals = read_plain_page_values(pinned_ptr, abs_off,
            col.data_pages[0].comp, col.data_pages[0].uncomp,
            col.max_rep, col.max_def, scratch, vals_len);
    }

    if (!vals || vals_len < 4) return "";

    // BYTE_ARRAY in PLAIN encoding: [4-byte LE length] [bytes]
    uint32_t str_len;
    std::memcpy(&str_len, vals, 4);
    if ((int64_t)(4 + str_len) > vals_len) return "";
    return std::string(reinterpret_cast<const char*>(vals + 4), str_len);
}

// Read a single numeric column generically (PLAIN or RLE_DICTIONARY).
// Returns a 1-D tensor of the appropriate dtype.
torch::Tensor read_numeric_column(
    const uint8_t* pinned_ptr, int64_t base_offset,
    const ColScan& col,
    std::vector<uint8_t>& scratch,
    std::vector<uint8_t>& dict_scratch,
    std::vector<int32_t>& idx_buf)
{
    int vw = col.value_byte_width();
    if (vw == 0 || col.num_values == 0)
        return torch::empty({0}, torch::kFloat32);

    auto dtype = (vw == 8)
        ? ((col.physical_type == 2) ? torch::kInt64 : torch::kFloat64)
        : ((col.physical_type == 1) ? torch::kInt32 : torch::kFloat32);

    auto tensor = torch::empty({col.num_values}, dtype);
    uint8_t* out = reinterpret_cast<uint8_t*>(tensor.data_ptr());

    if (col.has_dict) {
        const uint8_t* dict = decompress_page_into(
            pinned_ptr, col.dict_data_offset + base_offset,
            col.dict_comp, col.dict_uncomp, dict_scratch);
        if (!dict) return torch::empty({0}, dtype);
        int64_t n_dict = col.dict_uncomp / vw;
        if ((int64_t)idx_buf.size() < col.num_values)
            idx_buf.resize(col.num_values);
        decode_rle_dict_column_into(
            pinned_ptr, base_offset, col, vw,
            dict, n_dict, out, scratch, idx_buf);
    } else {
        int64_t written = 0;
        for (auto& dp : col.data_pages) {
            int64_t vals_len;
            auto vals = read_plain_page_values(
                pinned_ptr, dp.data_offset + base_offset,
                dp.comp, dp.uncomp,
                col.max_rep, col.max_def, scratch, vals_len);
            if (!vals) continue;
            int64_t n = vals_len / vw;
            int64_t copy_n = std::min(n, col.num_values - written);
            std::memcpy(out + written * vw, vals, copy_n * vw);
            written += copy_n;
        }
    }
    return tensor;
}

// AV2 obstacle class ID -> color index
// Returns index into a 5-entry palette: 0=Car,1=Truck,2=Pedestrian,3=Cyclist,4=Other
int obstacle_class_to_color_idx(int64_t class_id) {
    switch (class_id) {
        case 1281: return 0; // Car
        case 1286: return 0; // other_vehicle -> Car
        case 1282: case 1283: case 1284: case 1285: return 1; // Truck/bus/trailer
        case 3329: case 3330: return 1; // train/trolley -> Truck
        case 2308: case 2309: return 2; // Pedestrian
        case 2305: case 2306: case 2307: return 3; // Cyclist
        default: return 0; // default to Car
    }
}

// Color palette: [front_r,g,b, back_r,g,b] per type (matches OBSTACLE_COLORS_V3)
static const float OBS_COLORS[5][6] = {
    {0.f/255, 46.f/255, 136.f/255,  126.f/255, 206.f/255, 255.f/255},  // Car
    {204.f/255, 55.f/255, 0.f/255,  255.f/255, 192.f/255, 64.f/255},   // Truck
    {148.f/255, 0.f/255, 62.f/255,  255.f/255, 124.f/255, 171.f/255},  // Pedestrian
    {0.f/255, 80.f/255, 66.f/255,   102.f/255, 208.f/255, 198.f/255},  // Cyclist
    {53.f/255, 26.f/255, 20.f/255,  166.f/255, 136.f/255, 125.f/255},  // Other
};

// DEAD_START: parse_obstacles_from_pinned — superseded by decode_obstacles_cpu + decode_obstacles_gpu
#if 0
std::vector<torch::Tensor> parse_obstacles_from_pinned(
    torch::Tensor pinned_buf, int64_t obs_off, int64_t obs_sz)
{
    auto empty_result = [&]() -> std::vector<torch::Tensor> {
        return {
            torch::empty({0}, torch::kInt64),       // timestamps_us
            torch::empty({0}, torch::kInt32),        // track_prefix_sum
            torch::empty({0}, torch::kInt64),        // track_timestamps
            torch::empty({0, 3}, torch::kFloat32),   // translations
            torch::empty({0, 4}, torch::kFloat32),   // quaternions
            torch::empty({0, 3}, torch::kFloat32),   // scales
            torch::empty({0, 6}, torch::kFloat32),   // colors
        };
    };

    std::vector<std::string> obs_cols = {
        "key.timestamp_micros",                // 0: int64
        "object_fused.obstacle_id",            // 1: numeric (grouping key)
        "object_fused.cuboid_3D_center.x",     // 2: float
        "object_fused.cuboid_3D_center.y",     // 3: float
        "object_fused.cuboid_3D_center.z",     // 4: float
        "object_fused.obstacle_direction.x",   // 5: float
        "object_fused.obstacle_direction.y",   // 6: float
        "object_fused.cuboid_3D_halfAxisXYZ.x",// 7: float
        "object_fused.cuboid_3D_halfAxisXYZ.y",// 8: float
        "object_fused.cuboid_3D_halfAxisXYZ.z",// 9: float
        "object_fused.obstacle_class",         // 10: int or BYTE_ARRAY
    };

    auto scan_result = scan_parquet_pages_cpp(pinned_buf, obs_off, obs_sz, obs_cols);
    auto* sdata = scan_result.data_ptr<int64_t>();
    int64_t slen = scan_result.size(0);

    // Try parsing all 11 columns; if obstacle_class fails, parse 10
    std::vector<ColScan> cols;
    bool has_class_col = unpack_scan(sdata, slen, 11, cols);
    if (!has_class_col) {
        cols.clear();
        if (!unpack_scan(sdata, slen, 10, cols)) return empty_result();
    }

    int64_t n_rows = cols[0].num_values;
    if (n_rows == 0) return empty_result();

    const uint8_t* pinned_ptr = pinned_buf.data_ptr<uint8_t>();
    std::vector<uint8_t> scratch, dict_scratch;
    std::vector<int32_t> idx_buf(n_rows);

    // Read all numeric columns
    auto ts_all = read_numeric_column(pinned_ptr, obs_off, cols[0], scratch, dict_scratch, idx_buf);
    auto obs_id = read_numeric_column(pinned_ptr, obs_off, cols[1], scratch, dict_scratch, idx_buf);
    auto cx = read_numeric_column(pinned_ptr, obs_off, cols[2], scratch, dict_scratch, idx_buf);
    auto cy = read_numeric_column(pinned_ptr, obs_off, cols[3], scratch, dict_scratch, idx_buf);
    auto cz = read_numeric_column(pinned_ptr, obs_off, cols[4], scratch, dict_scratch, idx_buf);
    auto dx = read_numeric_column(pinned_ptr, obs_off, cols[5], scratch, dict_scratch, idx_buf);
    auto dy = read_numeric_column(pinned_ptr, obs_off, cols[6], scratch, dict_scratch, idx_buf);
    auto hx = read_numeric_column(pinned_ptr, obs_off, cols[7], scratch, dict_scratch, idx_buf);
    auto hy = read_numeric_column(pinned_ptr, obs_off, cols[8], scratch, dict_scratch, idx_buf);
    auto hz = read_numeric_column(pinned_ptr, obs_off, cols[9], scratch, dict_scratch, idx_buf);

    // Read obstacle_class if available (for color mapping)
    torch::Tensor obs_class;
    if (has_class_col && cols[10].physical_type != 6) {
        obs_class = read_numeric_column(pinned_ptr, obs_off, cols[10], scratch, dict_scratch, idx_buf);
    }

    // Ensure all tensors are the right type for math
    if (ts_all.scalar_type() != torch::kInt64) ts_all = ts_all.to(torch::kInt64);
    cx = cx.to(torch::kFloat32); cy = cy.to(torch::kFloat32); cz = cz.to(torch::kFloat32);
    dx = dx.to(torch::kFloat32); dy = dy.to(torch::kFloat32);
    hx = hx.to(torch::kFloat32); hy = hy.to(torch::kFloat32); hz = hz.to(torch::kFloat32);

    // Convert obstacle_id to float64 for grouping (handles both int and float id types)
    auto obs_id_f = obs_id.to(torch::kFloat64);

    // Filter NaN obstacle_ids
    auto valid_mask = ~torch::isnan(obs_id_f);
    if (!valid_mask.all().item<bool>()) {
        auto valid_idx = valid_mask.nonzero().squeeze(1);
        ts_all = ts_all.index({valid_idx});
        obs_id_f = obs_id_f.index({valid_idx});
        cx = cx.index({valid_idx}); cy = cy.index({valid_idx}); cz = cz.index({valid_idx});
        dx = dx.index({valid_idx}); dy = dy.index({valid_idx});
        hx = hx.index({valid_idx}); hy = hy.index({valid_idx}); hz = hz.index({valid_idx});
        if (obs_class.defined()) obs_class = obs_class.index({valid_idx});
        n_rows = ts_all.size(0);
        if (n_rows == 0) return empty_result();
    }

    // Vectorized yaw -> quaternion (z-axis rotation)
    auto yaw = torch::atan2(dy, dx);
    auto half_yaw = yaw * 0.5f;
    auto qz = torch::sin(half_yaw);
    auto qw = torch::cos(half_yaw);
    auto zeros = torch::zeros({n_rows}, torch::kFloat32);

    auto tquats = torch::stack({cx, cy, cz, zeros, zeros, qz, qw}, 1); // [N, 7]

    // Group by obstacle_id: sort, find group boundaries
    auto order = torch::argsort(obs_id_f, /*dim=*/0, /*descending=*/false);
    auto sorted_id = obs_id_f.index({order});
    auto sorted_ts = ts_all.index({order});
    auto sorted_tquats = tquats.index({order});
    auto sorted_hx = hx.index({order});
    auto sorted_hy = hy.index({order});
    auto sorted_hz = hz.index({order});

    // Find group boundaries (where sorted_id changes)
    auto diff = sorted_id.slice(0, 1) - sorted_id.slice(0, 0, -1);
    auto boundary_mask = (diff.abs() > 0.5);
    auto boundary_idx = boundary_mask.nonzero().squeeze(1) + 1;

    int64_t n_boundaries = boundary_idx.size(0);
    int64_t n_groups = n_boundaries + 1;

    // Build group start/end arrays
    auto starts = torch::zeros({n_groups}, torch::kInt64);
    auto ends = torch::full({n_groups}, n_rows, torch::kInt64);
    if (n_boundaries > 0) {
        starts.slice(0, 1).copy_(boundary_idx);
        ends.slice(0, 0, -1).copy_(boundary_idx);
    }
    auto starts_a = starts.accessor<int64_t, 1>();
    auto ends_a = ends.accessor<int64_t, 1>();

    // Filter groups with >= 2 timestamps, collect track data
    std::vector<int64_t> valid_starts, valid_ends;
    for (int64_t g = 0; g < n_groups; g++) {
        if (ends_a[g] - starts_a[g] >= 2) {
            valid_starts.push_back(starts_a[g]);
            valid_ends.push_back(ends_a[g]);
        }
    }

    int64_t n_tracks = (int64_t)valid_starts.size();
    if (n_tracks == 0) return empty_result();

    // Build output tensors
    std::vector<int> track_lengths;
    int64_t total_poses = 0;
    for (int64_t t = 0; t < n_tracks; t++) {
        int len = (int)(valid_ends[t] - valid_starts[t]);
        track_lengths.push_back(len);
        total_poses += len;
    }

    auto track_ps = torch::cumsum(
        torch::tensor(track_lengths, torch::kInt32), 0);
    auto track_ts = torch::empty({total_poses}, torch::kInt64);
    auto translations = torch::empty({total_poses, 3}, torch::kFloat32);
    auto quaternions = torch::empty({total_poses, 4}, torch::kFloat32);
    auto scales = torch::empty({n_tracks, 3}, torch::kFloat32);
    auto colors = torch::empty({n_tracks, 6}, torch::kFloat32);

    auto track_ts_a = track_ts.accessor<int64_t, 1>();
    auto trans_a = translations.accessor<float, 2>();
    auto quat_a = quaternions.accessor<float, 2>();
    auto scales_a = scales.accessor<float, 2>();
    auto colors_a = colors.accessor<float, 2>();

    auto s_ts = sorted_ts.accessor<int64_t, 1>();
    auto s_tq = sorted_tquats.accessor<float, 2>();
    auto s_hx = sorted_hx.accessor<float, 1>();
    auto s_hy = sorted_hy.accessor<float, 1>();
    auto s_hz = sorted_hz.accessor<float, 1>();

    // Optional: accessor for obstacle_class color mapping
    bool use_class = obs_class.defined() && obs_class.size(0) == n_rows;
    torch::Tensor sorted_class;
    int64_t* sc_ptr = nullptr;
    if (use_class) {
        sorted_class = obs_class.to(torch::kInt64).index({order}).contiguous();
        sc_ptr = sorted_class.data_ptr<int64_t>();
    }

    int64_t pos = 0;
    for (int64_t t = 0; t < n_tracks; t++) {
        int64_t s = valid_starts[t], e = valid_ends[t];

        for (int64_t i = s; i < e; i++, pos++) {
            track_ts_a[pos] = s_ts[i];
            trans_a[pos][0] = s_tq[i][0];
            trans_a[pos][1] = s_tq[i][1];
            trans_a[pos][2] = s_tq[i][2];
            quat_a[pos][0] = s_tq[i][3];
            quat_a[pos][1] = s_tq[i][4];
            quat_a[pos][2] = s_tq[i][5];
            quat_a[pos][3] = s_tq[i][6];
        }

        int64_t last = e - 1;
        scales_a[t][0] = s_hx[last] * 2.0f;
        scales_a[t][1] = s_hy[last] * 2.0f;
        scales_a[t][2] = s_hz[last] * 2.0f;

        int cidx = use_class ? obstacle_class_to_color_idx(sc_ptr[last]) : 0;
        for (int c = 0; c < 6; c++) colors_a[t][c] = OBS_COLORS[cidx][c];
    }

    // Global unique sorted timestamps (sort then unique_consecutive is cleaner than unique)
    auto sorted_track_ts = std::get<0>(torch::sort(track_ts));
    auto global_ts = std::get<0>(torch::unique_consecutive(sorted_track_ts));

    return {global_ts, track_ps, track_ts, translations, quaternions, scales, colors};
}
#endif // DEAD_END: parse_obstacles_from_pinned

// Decode-only variant: uses pre-scanned ColScan metadata (no re-scan).
std::pair<torch::Tensor, torch::Tensor> decode_ego_columns(
    const uint8_t* pinned_ptr, int64_t ego_off,
    std::vector<ColScan>& cols
) {
    int64_t n_rows = cols[0].num_values;
    if (n_rows == 0) {
        return {torch::empty({0}, torch::kInt64),
                torch::empty({0, 7}, torch::kFloat32)};
    }

    std::vector<uint8_t> scratch, dict_scratch;
    std::vector<int32_t> idx_buf(n_rows);

    auto timestamps = torch::empty({n_rows}, torch::kInt64);
    {
        const uint8_t* dict_data = decompress_page_into(
            pinned_ptr, cols[0].dict_data_offset + ego_off,
            cols[0].dict_comp, cols[0].dict_uncomp, dict_scratch);
        if (!dict_data) {
            return {torch::empty({0}, torch::kInt64),
                    torch::empty({0, 7}, torch::kFloat32)};
        }
        int64_t n_dict = cols[0].dict_uncomp / 8;
        decode_rle_dict_column_into(
            pinned_ptr, ego_off, cols[0], 8,
            dict_data, n_dict,
            reinterpret_cast<uint8_t*>(timestamps.data_ptr<int64_t>()),
            scratch, idx_buf);
    }

    auto poses = torch::empty({n_rows, 7}, torch::kFloat32);
    float* poses_ptr = poses.data_ptr<float>();
    std::vector<float> col_buf(n_rows);

    for (int ci = 0; ci < 7; ci++) {
        auto& col = cols[ci + 1];
        const uint8_t* dict_data = decompress_page_into(
            pinned_ptr, col.dict_data_offset + ego_off,
            col.dict_comp, col.dict_uncomp, dict_scratch);
        if (!dict_data) continue;
        int64_t n_dict = col.dict_uncomp / 4;
        decode_rle_dict_column_into(
            pinned_ptr, ego_off, col, 4,
            dict_data, n_dict,
            reinterpret_cast<uint8_t*>(col_buf.data()),
            scratch, idx_buf);
        for (int64_t r = 0; r < n_rows; r++) {
            poses_ptr[r * 7 + ci] = col_buf[r];
        }
    }

    return {timestamps, poses};
}

// Result of CPU-only obstacle column decoding (passed to GPU phase).
struct ObsCpuResult {
    torch::Tensor ts_all;       // [n] int64
    torch::Tensor obs_id_f;     // [n] float64
    torch::Tensor packed_f32;   // [n, 8] float32: cx,cy,cz,dx,dy,hx,hy,hz
    torch::Tensor obs_class;    // [n] or undefined
    int64_t n_rows = 0;
    bool valid = false;
};

ObsCpuResult decode_obstacles_cpu(
    const uint8_t* pinned_ptr, int64_t obs_off,
    std::vector<ColScan>& cols, bool has_class_col
) {
    ObsCpuResult result;
    int64_t n_rows = cols[0].num_values;
    if (n_rows == 0) return result;

    std::vector<uint8_t> scratch, dict_scratch;
    std::vector<int32_t> idx_buf(n_rows);

    auto ts_all = read_numeric_column(pinned_ptr, obs_off, cols[0], scratch, dict_scratch, idx_buf);
    auto obs_id = read_numeric_column(pinned_ptr, obs_off, cols[1], scratch, dict_scratch, idx_buf);
    auto cx = read_numeric_column(pinned_ptr, obs_off, cols[2], scratch, dict_scratch, idx_buf);
    auto cy = read_numeric_column(pinned_ptr, obs_off, cols[3], scratch, dict_scratch, idx_buf);
    auto cz = read_numeric_column(pinned_ptr, obs_off, cols[4], scratch, dict_scratch, idx_buf);
    auto dx = read_numeric_column(pinned_ptr, obs_off, cols[5], scratch, dict_scratch, idx_buf);
    auto dy = read_numeric_column(pinned_ptr, obs_off, cols[6], scratch, dict_scratch, idx_buf);
    auto hx = read_numeric_column(pinned_ptr, obs_off, cols[7], scratch, dict_scratch, idx_buf);
    auto hy = read_numeric_column(pinned_ptr, obs_off, cols[8], scratch, dict_scratch, idx_buf);
    auto hz = read_numeric_column(pinned_ptr, obs_off, cols[9], scratch, dict_scratch, idx_buf);

    torch::Tensor obs_class;
    if (has_class_col && cols[10].physical_type != 6) {
        obs_class = read_numeric_column(pinned_ptr, obs_off, cols[10], scratch, dict_scratch, idx_buf);
    }

    if (ts_all.scalar_type() != torch::kInt64) ts_all = ts_all.to(torch::kInt64);
    cx = cx.to(torch::kFloat32); cy = cy.to(torch::kFloat32); cz = cz.to(torch::kFloat32);
    dx = dx.to(torch::kFloat32); dy = dy.to(torch::kFloat32);
    hx = hx.to(torch::kFloat32); hy = hy.to(torch::kFloat32); hz = hz.to(torch::kFloat32);

    auto packed_f32 = torch::stack({cx, cy, cz, dx, dy, hx, hy, hz}, 1);

    auto obs_id_f = obs_id.to(torch::kFloat64);
    auto valid_mask = ~torch::isnan(obs_id_f);
    if (!valid_mask.all().item<bool>()) {
        auto valid_idx = valid_mask.nonzero().squeeze(1);
        ts_all = ts_all.index({valid_idx});
        obs_id_f = obs_id_f.index({valid_idx});
        packed_f32 = packed_f32.index({valid_idx});
        if (obs_class.defined()) obs_class = obs_class.index({valid_idx});
        n_rows = ts_all.size(0);
        if (n_rows == 0) return result;
    }

    result.ts_all = ts_all;
    result.obs_id_f = obs_id_f;
    result.packed_f32 = packed_f32;
    result.obs_class = obs_class;
    result.n_rows = n_rows;
    result.valid = true;
    return result;
}

std::vector<torch::Tensor> obs_empty_result() {
    return {
        torch::empty({0}, torch::kInt64),
        torch::empty({0}, torch::kInt32),
        torch::empty({0}, torch::kInt64),
        torch::empty({0, 3}, torch::kFloat32),
        torch::empty({0, 4}, torch::kFloat32),
        torch::empty({0, 3}, torch::kFloat32),
        torch::empty({0, 6}, torch::kFloat32),
    };
}

std::vector<torch::Tensor> decode_obstacles_gpu(
    const ObsCpuResult& cpu, torch::Device device
) {
    if (!cpu.valid) return obs_empty_result();

    auto d_ts = cpu.ts_all.to(device, /*non_blocking=*/true);
    auto d_id = cpu.obs_id_f.to(device, true);
    auto d_packed = cpu.packed_f32.to(device, true);

    obs_yaw_to_quat(d_packed);

    auto order = torch::argsort(d_id, 0, false);
    auto sorted_id = d_id.index_select(0, order);
    auto sorted_ts = d_ts.index_select(0, order);
    auto sorted_pack = d_packed.index_select(0, order);

    bool use_class = cpu.obs_class.defined() && cpu.obs_class.size(0) == cpu.n_rows;
    torch::Tensor sorted_class;
    if (use_class) {
        auto d_class = cpu.obs_class.to(torch::kInt64).to(device, true);
        sorted_class = d_class.index_select(0, order);
    }

    return obs_group_and_gather(sorted_id, sorted_ts, sorted_pack,
                                sorted_class, use_class);
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// Camera parsing from rig JSON (C++ — avoids Python/PyArrow/scipy overhead)
// ---------------------------------------------------------------------------

struct CameraResult {
    torch::Tensor pp;        // [N, 2] float32
    torch::Tensor sz;        // [N, 2] float32
    torch::Tensor fw_poly;   // [N, 6] float32
    torch::Tensor ld;        // [N, 2, 2] float32
    torch::Tensor s2r;       // [N, 4, 4] float32
    torch::Tensor mra;       // [N] float32
    torch::Tensor names;     // [L] uint8 (null-separated camera name bytes)
    int n_cameras = 0;
};

static void euler_xyz_to_rotation(double roll, double pitch, double yaw, float out[9]) {
    double ca = std::cos(roll),  sa = std::sin(roll);
    double cb = std::cos(pitch), sb = std::sin(pitch);
    double cc = std::cos(yaw),   sc = std::sin(yaw);
    out[0] = (float)(cb*cc);             out[1] = (float)(-cb*sc);            out[2] = (float)(sb);
    out[3] = (float)(sa*sb*cc + ca*sc);  out[4] = (float)(-sa*sb*sc + ca*cc); out[5] = (float)(-sa*cb);
    out[6] = (float)(-ca*sb*cc + sa*sc); out[7] = (float)(ca*sb*sc + sa*cc);  out[8] = (float)(ca*cb);
}

static void mat3_mul(const float a[9], const float b[9], float out[9]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) sum += a[i*3+k] * b[k*3+j];
            out[i*3+j] = sum;
        }
}

static std::vector<double> parse_poly_string(const std::string& s) {
    std::vector<double> coeffs;
    std::istringstream iss(s);
    double v;
    while (iss >> v) coeffs.push_back(v);
    while (coeffs.size() < 6) coeffs.push_back(0.0);
    return coeffs;
}

static CameraResult parse_cameras_from_rig_json(
    const uint8_t* json_bytes, int64_t json_len,
    int target_w, int target_h
) {
    CameraResult result;
    if (json_len <= 0) return result;

    auto root = json_parse(reinterpret_cast<const char*>(json_bytes), (size_t)json_len);
    const auto& rig = root["rig"];
    if (!rig.is_object()) {
        fprintf(stderr, "    [cam_debug] rig is not object (type=%d), root type=%d, json_len=%ld\n",
                (int)rig.type, (int)root.type, (long)json_len);
        return result;
    }
    const auto& sensors = rig["sensors"];
    if (!sensors.is_array()) {
        fprintf(stderr, "    [cam_debug] sensors is not array (type=%d)\n", (int)sensors.type);
        return result;
    }

    struct CamInfo {
        std::string name;
        std::vector<double> poly;
        bool is_bw;
        double cx, cy, width, height;
        float linear_cde[3];
        float rot[9];        // final 3x3 rotation
        float trans[3];
    };
    std::vector<CamInfo> cams;
    constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;

    for (size_t si = 0; si < sensors.size(); si++) {
        const auto& sensor = sensors[si];
        std::string name = sensor.get_string("name");
        if (name.substr(0, 7) != "camera:") continue;
        const auto& props = sensor["properties"];
        if (props.is_null()) continue;

        std::string poly_key;
        if (props.has("polynomial"))   poly_key = "polynomial";
        else if (props.has("bw-poly")) poly_key = "bw-poly";
        else continue;

        auto poly = parse_poly_string(props.get_string(poly_key));

        std::string poly_type = props.get_string("polynomial-type");
        bool is_bw;
        if (poly_type == "angle-to-pixeldistance") is_bw = false;
        else if (poly_type == "pixeldistance-to-angle") is_bw = true;
        else if (poly_key == "bw-poly") is_bw = true;
        else is_bw = poly.size() > 1 && std::abs(poly[1]) < 1.0;

        float lc = (float)props.get_number("linear-c", 1.0);
        float ld = (float)props.get_number("linear-d", 0.0);
        float le = (float)props.get_number("linear-e", 0.0);

        const auto& s2r_flu = sensor["nominalSensor2Rig_FLU"];
        if (s2r_flu.is_null()) continue;
        const auto& rpy_arr = s2r_flu["roll-pitch-yaw"];
        const auto& t_arr = s2r_flu["t"];
        if (!rpy_arr.is_array() || rpy_arr.size() < 3) continue;
        if (!t_arr.is_array() || t_arr.size() < 3) continue;

        double rpy[3] = {
            rpy_arr[0].number() * DEG2RAD,
            rpy_arr[1].number() * DEG2RAD,
            rpy_arr[2].number() * DEG2RAD
        };
        float rot[9];
        euler_xyz_to_rotation(rpy[0], rpy[1], rpy[2], rot);

        if (sensor.has("correction_sensor_R_FLU")) {
            const auto& corr = sensor["correction_sensor_R_FLU"];
            const auto& crpy = corr["roll-pitch-yaw"];
            if (crpy.is_array() && crpy.size() >= 3) {
                double cr[3] = {
                    crpy[0].number() * DEG2RAD,
                    crpy[1].number() * DEG2RAD,
                    crpy[2].number() * DEG2RAD
                };
                float corr_rot[9], combined[9];
                euler_xyz_to_rotation(cr[0], cr[1], cr[2], corr_rot);
                mat3_mul(rot, corr_rot, combined);
                std::memcpy(rot, combined, sizeof(rot));
            }
        }

        CamInfo ci;
        ci.name = name;
        ci.poly = poly;
        ci.is_bw = is_bw;
        ci.cx = props.get_number("cx");
        ci.cy = props.get_number("cy");
        ci.width = props.get_number("width");
        ci.height = props.get_number("height");
        ci.linear_cde[0] = lc; ci.linear_cde[1] = ld; ci.linear_cde[2] = le;
        std::memcpy(ci.rot, rot, sizeof(rot));
        ci.trans[0] = (float)t_arr[0].number();
        ci.trans[1] = (float)t_arr[1].number();
        ci.trans[2] = (float)t_arr[2].number();
        cams.push_back(std::move(ci));
    }

    int n = (int)cams.size();
    if (n == 0) return result;
    result.n_cameras = n;

    int max_poly_len = 0;
    for (auto& c : cams) max_poly_len = std::max(max_poly_len, (int)c.poly.size());

    auto poly_coeffs = torch::zeros({n, max_poly_len}, torch::kFloat64);
    auto poly_lengths = torch::empty({n}, torch::kInt32);
    auto is_bw_t = torch::empty({n}, torch::kBool);
    auto cx_raw = torch::empty({n}, torch::kFloat64);
    auto cy_raw = torch::empty({n}, torch::kFloat64);
    auto w_raw = torch::empty({n}, torch::kFloat64);
    auto h_raw = torch::empty({n}, torch::kFloat64);
    auto cx_sc = torch::empty({n}, torch::kFloat64);
    auto cy_sc = torch::empty({n}, torch::kFloat64);
    auto w_sc = torch::empty({n}, torch::kFloat64);
    auto h_sc = torch::empty({n}, torch::kFloat64);
    auto pscale_t = torch::empty({n}, torch::kFloat64);

    auto pp_cpu = torch::empty({n, 2}, torch::kFloat32);
    auto sz_cpu = torch::empty({n, 2}, torch::kFloat32);
    auto ld_cpu = torch::zeros({n, 2, 2}, torch::kFloat32);
    auto s2r_cpu = torch::zeros({n, 4, 4}, torch::kFloat32);

    auto pc_a = poly_coeffs.accessor<double, 2>();
    auto pl_a = poly_lengths.accessor<int32_t, 1>();
    auto bw_a = is_bw_t.accessor<bool, 1>();
    auto cx_r_a = cx_raw.accessor<double, 1>();
    auto cy_r_a = cy_raw.accessor<double, 1>();
    auto w_r_a = w_raw.accessor<double, 1>();
    auto h_r_a = h_raw.accessor<double, 1>();
    auto cx_s_a = cx_sc.accessor<double, 1>();
    auto cy_s_a = cy_sc.accessor<double, 1>();
    auto w_s_a = w_sc.accessor<double, 1>();
    auto h_s_a = h_sc.accessor<double, 1>();
    auto ps_a = pscale_t.accessor<double, 1>();
    auto pp_a = pp_cpu.accessor<float, 2>();
    auto sz_a = sz_cpu.accessor<float, 2>();
    auto ld_a = ld_cpu.accessor<float, 3>();
    auto s2r_a = s2r_cpu.accessor<float, 3>();

    std::string name_bytes;
    for (int i = 0; i < n; i++) {
        auto& c = cams[i];
        int plen = (int)c.poly.size();
        for (int j = 0; j < plen; j++) pc_a[i][j] = (double)(float)c.poly[j];
        pl_a[i] = plen;
        bw_a[i] = c.is_bw;
        cx_r_a[i] = c.cx;
        cy_r_a[i] = c.cy;
        w_r_a[i] = c.width;
        h_r_a[i] = c.height;

        double sx = 1.0, sy = 1.0, ps = 1.0;
        double cxs = c.cx, cys = c.cy, ws = c.width, hs = c.height;
        if (target_w > 0 && target_h > 0) {
            sx = (double)target_w / c.width;
            sy = (double)target_h / c.height;
            ps = (sx + sy) / 2.0;
            cxs = c.cx * sx;
            cys = c.cy * sy;
            ws = (double)target_w;
            hs = (double)target_h;
        }
        cx_s_a[i] = cxs;
        cy_s_a[i] = cys;
        w_s_a[i] = ws;
        h_s_a[i] = hs;
        ps_a[i] = ps;
        pp_a[i][0] = (float)cxs;
        pp_a[i][1] = (float)cys;
        sz_a[i][0] = (float)ws;
        sz_a[i][1] = (float)hs;

        ld_a[i][0][0] = c.linear_cde[0]; // c
        ld_a[i][0][1] = c.linear_cde[1]; // d
        ld_a[i][1][0] = c.linear_cde[2]; // e
        ld_a[i][1][1] = 1.0f;

        s2r_a[i][0][0] = c.rot[0]; s2r_a[i][0][1] = c.rot[1]; s2r_a[i][0][2] = c.rot[2];
        s2r_a[i][1][0] = c.rot[3]; s2r_a[i][1][1] = c.rot[4]; s2r_a[i][1][2] = c.rot[5];
        s2r_a[i][2][0] = c.rot[6]; s2r_a[i][2][1] = c.rot[7]; s2r_a[i][2][2] = c.rot[8];
        s2r_a[i][0][3] = c.trans[0]; s2r_a[i][1][3] = c.trans[1]; s2r_a[i][2][3] = c.trans[2];
        s2r_a[i][3][3] = 1.0f;

        if (i > 0) name_bytes.push_back('\0');
        name_bytes += c.name;
    }

    auto cam_result = compute_camera_params(
        poly_coeffs, poly_lengths, is_bw_t,
        cx_raw, cy_raw, w_raw, h_raw,
        cx_sc, cy_sc, w_sc, h_sc, pscale_t
    );

    result.pp = pp_cpu;
    result.sz = sz_cpu;
    result.fw_poly = cam_result[0];  // [N, 6] float32
    result.ld = ld_cpu;
    result.s2r = s2r_cpu;
    result.mra = cam_result[1];      // [N] float32
    result.names = torch::empty({(int64_t)name_bytes.size()}, torch::kUInt8);
    std::memcpy(result.names.data_ptr<uint8_t>(), name_bytes.data(), name_bytes.size());

    return result;
}


// ---------------------------------------------------------------------------
// load_scene_gpu: single C++ entry point for the GPU scene loading pipeline
// ---------------------------------------------------------------------------
//
// Returns a flat vector of tensors:
//   [0..19]: polyline data (5 tensors × 4 files)
//   [20]: ego_timestamps (int64 [N_ego])
//   [21]: ego_poses_tquat (float32 [N_ego, 7])
//   [22]: rig_json_bytes (uint8 [L])
//   [23..29]: obstacle pool tensors (timestamps_us, track_ps, track_ts,
//             translations, quaternions, scales, colors)
//   [30..31]: crosswalk triangulation (triangle_prefix_sum, triangles)
//   [32..38]: camera data (pp, sz, fw_poly, ld, s2r, mra, names) — if target_w > 0
//
// Total: 32 or 39 tensors.
std::vector<torch::Tensor> load_scene_gpu(
    torch::Tensor pinned_buf,
    std::vector<std::string> entry_names,
    std::vector<int64_t> entry_offsets,
    std::vector<int64_t> entry_sizes,
    int device_index,
    int target_w,
    int target_h
) {

    c10::cuda::CUDAGuard device_guard(device_index);
    auto device = torch::Device(torch::kCUDA, device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index).stream();

    // Build entry lookup: basename -> (offset, size)
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> entry_map;
    for (size_t i = 0; i < entry_names.size(); i++) {
        auto& name = entry_names[i];
        auto slash = name.rfind('/');
        std::string base = (slash != std::string::npos) ? name.substr(slash + 1) : name;
        entry_map[base] = {entry_offsets[i], entry_sizes[i]};
    }

    // ── Phase 1: PARALLEL metadata scanning for ALL parquet files ──
    // Scan polyline + ego + obs + cal files concurrently to reduce serial scan time.
    struct FileData {
        int spec_idx;
        std::vector<ColScan> cols;
        int64_t parquet_offset;
        bool valid = false;
    };

    // Polyline scan results (one per file spec)
    struct PolyScanResult {
        int spec_idx;
        int64_t offset;
        torch::Tensor flat;
    };

    // Launch polyline scans in parallel
    std::vector<std::future<PolyScanResult>> poly_futures;
    for (int si = 0; si < N_FILE_SPECS; si++) {
        auto it = entry_map.find(FILE_SPECS[si].basename);
        if (it == entry_map.end()) continue;
        auto off = it->second.first;
        auto sz = it->second.second;
        poly_futures.push_back(get_pool().submit(
            [&pinned_buf, si, off, sz]() -> PolyScanResult {
                std::vector<std::string> col_paths = {
                    FILE_SPECS[si].x_path, FILE_SPECS[si].y_path,
                    FILE_SPECS[si].z_path, FILE_SPECS[si].ts_path
                };
                auto flat = scan_parquet_pages_cpp(pinned_buf, off, sz, col_paths);
                return {si, off, flat};
            }));
    }

    // Launch ego scan in parallel
    int64_t ego_off = 0, ego_sz = 0;
    bool has_ego = false;
    std::future<torch::Tensor> ego_scan_future;
    {
        auto it = entry_map.find("egomotion_estimate.parquet");
        if (it != entry_map.end()) {
            has_ego = true;
            ego_off = it->second.first;
            ego_sz = it->second.second;
            ego_scan_future = get_pool().submit(
                [&pinned_buf, ego_off, ego_sz]() {
                    std::vector<std::string> ego_cols = {
                        "key.timestamp_micros",
                        "egomotion_estimate.location.x",
                        "egomotion_estimate.location.y",
                        "egomotion_estimate.location.z",
                        "egomotion_estimate.orientation.x",
                        "egomotion_estimate.orientation.y",
                        "egomotion_estimate.orientation.z",
                        "egomotion_estimate.orientation.w",
                    };
                    return scan_parquet_pages_cpp(pinned_buf, ego_off, ego_sz, ego_cols);
                });
        }
    }

    // Launch obstacle scan in parallel
    int64_t obs_off = 0, obs_sz = 0;
    bool has_obs = false;
    std::future<torch::Tensor> obs_scan_future;
    {
        auto it = entry_map.find("object_fused.parquet");
        if (it != entry_map.end()) {
            has_obs = true;
            obs_off = it->second.first;
            obs_sz = it->second.second;
            obs_scan_future = get_pool().submit(
                [&pinned_buf, obs_off, obs_sz]() {
                    std::vector<std::string> obs_cols = {
                        "key.timestamp_micros",
                        "object_fused.obstacle_id",
                        "object_fused.cuboid_3D_center.x",
                        "object_fused.cuboid_3D_center.y",
                        "object_fused.cuboid_3D_center.z",
                        "object_fused.obstacle_direction.x",
                        "object_fused.obstacle_direction.y",
                        "object_fused.cuboid_3D_halfAxisXYZ.x",
                        "object_fused.cuboid_3D_halfAxisXYZ.y",
                        "object_fused.cuboid_3D_halfAxisXYZ.z",
                        "object_fused.obstacle_class",
                    };
                    return scan_parquet_pages_cpp(pinned_buf, obs_off, obs_sz, obs_cols);
                });
        }
    }

    // Find calibration_estimate.parquet offset/size (parsed inside ego_cal_future)
    int64_t cal_off = 0, cal_sz = 0;
    bool has_cal = false;
    {
        auto it = entry_map.find("calibration_estimate.parquet");
        if (it != entry_map.end()) {
            has_cal = true;
            cal_off = it->second.first;
            cal_sz = it->second.second;
        }
    }

    // Collect polyline scan results
    std::vector<FileData> file_datas;
    for (auto& f : poly_futures) {
        auto result = f.get();
        auto* data = result.flat.data_ptr<int64_t>();
        int64_t len = result.flat.size(0);

        FileData fd;
        fd.spec_idx = result.spec_idx;
        fd.parquet_offset = result.offset;
        if (unpack_scan(data, len, 4, fd.cols)) {
            bool skip = false;
            for (auto& c : fd.cols) {
                if (c.data_pages.empty()) { skip = true; break; }
            }
            if (!skip) {
                fd.valid = true;
                file_datas.push_back(std::move(fd));
            }
        }
    }

    // Collect ego/obs scan results (ready for decode-only phase later)
    torch::Tensor ego_scan_flat, obs_scan_flat;
    if (has_ego) ego_scan_flat = ego_scan_future.get();
    if (has_obs) obs_scan_flat = obs_scan_future.get();


    // Prepare empty output tensors
    auto empty_f32_3 = torch::empty({0, 3}, torch::kFloat32);
    auto empty_i64 = torch::empty({0}, torch::kInt64);
    auto empty_i32 = torch::empty({0}, torch::kInt32);

    std::vector<torch::Tensor> output(40);
    for (int i = 0; i < N_FILE_SPECS; i++) {
        output[i * 5 + 0] = empty_f32_3;
        output[i * 5 + 1] = empty_i64;
        output[i * 5 + 2] = empty_i32;
        output[i * 5 + 3] = empty_i64;
        output[i * 5 + 4] = empty_i32;
    }
    output[23] = empty_i64;  output[24] = empty_i32;  output[25] = empty_i64;
    output[26] = empty_f32_3; output[27] = torch::empty({0, 4}, torch::kFloat32);
    output[28] = empty_f32_3; output[29] = torch::empty({0, 6}, torch::kFloat32);
    output[30] = empty_i32;  output[31] = torch::empty({0, 3}, torch::kInt32);

    // ── Submit ALL CPU-only decode to thread pool immediately (overlaps plan+gpu_launch) ──
    torch::Tensor ego_timestamps, ego_poses_tquat, rig_json_bytes;
    torch::Tensor ego_ts_gpu, ego_tq_gpu;
    std::vector<torch::Tensor> obstacle_tensors;
    std::future<CameraResult> camera_future;
    const uint8_t* pinned_ptr = pinned_buf.data_ptr<uint8_t>();

    auto ego_cal_future = get_pool().submit([&, target_w, target_h]() {
        if (has_ego && ego_scan_flat.defined()) {
            auto* sdata = ego_scan_flat.data_ptr<int64_t>();
            int64_t slen = ego_scan_flat.size(0);
            std::vector<ColScan> ecols;
            if (unpack_scan(sdata, slen, 8, ecols)) {
                std::tie(ego_timestamps, ego_poses_tquat) =
                    decode_ego_columns(pinned_ptr, ego_off, ecols);
            }
        }
        if (!ego_timestamps.defined()) {
            ego_timestamps = torch::empty({0}, torch::kInt64);
            ego_poses_tquat = torch::empty({0, 7}, torch::kFloat32);
        }

        if (has_cal) {
            auto json_str = extract_rig_json_from_pinned(pinned_buf, cal_off, cal_sz);
            if (!json_str.empty()) {
                rig_json_bytes = torch::empty({(int64_t)json_str.size()}, torch::kUInt8);
                std::memcpy(rig_json_bytes.data_ptr<uint8_t>(), json_str.data(), json_str.size());
                const uint8_t* rj_ptr = rig_json_bytes.data_ptr<uint8_t>();
                int64_t rj_len = rig_json_bytes.size(0);
                camera_future = get_pool().submit(
                    [rj_ptr, rj_len, target_w, target_h]() {
                        return parse_cameras_from_rig_json(rj_ptr, rj_len, target_w, target_h);
                    });
            } else {
                static bool dbg = (std::getenv("SCENE_LOADER_VERBOSE") != nullptr);
                if (dbg) {
                    fprintf(stderr, "    [cal_debug] extract_rig_json returned empty, has_cal=%d, cal_off=%ld, cal_sz=%ld\n",
                            has_cal ? 1 : 0, (long)cal_off, (long)cal_sz);
                    fflush(stderr);
                }
            }
        }
        if (!rig_json_bytes.defined()) {
            rig_json_bytes = torch::empty({0}, torch::kUInt8);
        }
    });

    std::future<ObsCpuResult> obs_cpu_future;
    if (has_obs && obs_scan_flat.defined()) {
        obs_cpu_future = get_pool().submit([&]() -> ObsCpuResult {
            auto* sdata = obs_scan_flat.data_ptr<int64_t>();
            int64_t slen = obs_scan_flat.size(0);
            std::vector<ColScan> ocols;
            bool has_class = unpack_scan(sdata, slen, 11, ocols);
            if (!has_class) {
                ocols.clear();
                if (unpack_scan(sdata, slen, 10, ocols))
                    return decode_obstacles_cpu(pinned_ptr, obs_off, ocols, false);
                return {};
            }
            return decode_obstacles_cpu(pinned_ptr, obs_off, ocols, true);
        });
    }

    if (file_datas.empty()) {
        ego_cal_future.get();
        ObsCpuResult obs_cpu;
        if (obs_cpu_future.valid()) obs_cpu = obs_cpu_future.get();
        obstacle_tensors = decode_obstacles_gpu(obs_cpu, device);
        output[20] = ego_timestamps;
        output[21] = ego_poses_tquat;
        output[22] = rig_json_bytes;
        for (int i = 0; i < 7; i++) output[23 + i] = obstacle_tensors[i];
        if (rig_json_bytes.defined() && rig_json_bytes.size(0) > 0) {
            auto cam = parse_cameras_from_rig_json(
                rig_json_bytes.data_ptr<uint8_t>(), rig_json_bytes.size(0),
                target_w, target_h);
            if (cam.n_cameras > 0) {
                output.resize(39);
                output[32] = cam.pp.to(device); output[33] = cam.sz.to(device);
                output[34] = cam.fw_poly.to(device); output[35] = cam.ld.to(device);
                output[36] = cam.s2r.to(device); output[37] = cam.mra; output[38] = cam.names;
            }
        }
        return output;
    }

    // ── Phase 2a: Launch H2D early so it overlaps with plan building ──
    auto gpu_buffer = pinned_buf.to(device, /*non_blocking=*/true);

    // ── Phase 2b: Build pipeline plan (CPU, overlaps with H2D transfer) ──
    std::vector<int64_t> all_page_offsets, all_page_comp, all_page_uncomp;
    std::vector<int> data_page_indices;
    std::vector<int> xyz_dict_page_indices, ts_dict_page_indices;
    std::vector<int> xyz_dict_byte_offsets, ts_dict_byte_offsets;
    std::vector<int> all_max_reps, all_max_defs, all_num_vals;
    int dict_xyz_byte_cursor = 0, dict_ts_byte_cursor = 0;

    constexpr int FIELDS_PER_FILE = 17;
    std::vector<int> file_info_list;
    std::vector<int> file_order_spec_idx;

    int total_xyz = 0, total_ts = 0, total_rows = 0;
    int row_off_cursor = 0, dict_xyz_off = 0, dict_ts_off = 0;

    for (size_t fi = 0; fi < file_datas.size(); fi++) {
        auto& fd = file_datas[fi];
        file_order_spec_idx.push_back(fd.spec_idx);

        for (int ci = 0; ci < 4; ci++) {
            auto& cs = fd.cols[ci];
            if (cs.has_dict) {
                int pi = (int)all_page_offsets.size();
                int64_t abs_off = cs.dict_data_offset + fd.parquet_offset;
                all_page_offsets.push_back(abs_off);
                all_page_comp.push_back(cs.dict_comp);
                all_page_uncomp.push_back(cs.dict_uncomp);

                if (ci < 3) {
                    xyz_dict_page_indices.push_back(pi);
                    xyz_dict_byte_offsets.push_back(dict_xyz_byte_cursor);
                    dict_xyz_byte_cursor += (int)cs.dict_uncomp;
                } else {
                    ts_dict_page_indices.push_back(pi);
                    ts_dict_byte_offsets.push_back(dict_ts_byte_cursor);
                    dict_ts_byte_cursor += (int)cs.dict_uncomp;
                }
            }
            for (size_t di = 0; di < cs.data_pages.size(); di++) {
                int pi = (int)all_page_offsets.size();
                int64_t abs_off = cs.data_pages[di].data_offset + fd.parquet_offset;
                all_page_offsets.push_back(abs_off);
                all_page_comp.push_back(cs.data_pages[di].comp);
                all_page_uncomp.push_back(cs.data_pages[di].uncomp);
                if (di == 0) data_page_indices.push_back(pi);
            }
            all_max_reps.push_back(cs.max_rep);
            all_max_defs.push_back(cs.max_def);
            all_num_vals.push_back((int)cs.num_values);
        }
    }

    std::vector<int> out_starts;
    int total_rle_values = 0;
    for (auto nv : all_num_vals) {
        out_starts.push_back(total_rle_values);
        total_rle_values += nv;
    }

    int stream_idx = 0;
    for (size_t fi = 0; fi < file_datas.size(); fi++) {
        auto& fd = file_datas[fi];
        int si_x = stream_idx, si_y = stream_idx + 1;
        int si_z = stream_idx + 2, si_ts = stream_idx + 3;
        int n_xyz = all_num_vals[si_x];
        int n_rows = all_num_vals[si_ts];

        auto& dx_col = fd.cols[0];
        auto& dy_col = fd.cols[1];
        auto& dz_col = fd.cols[2];
        int dx_len = (int)dx_col.dict_uncomp / 4;
        int dy_len = (int)dy_col.dict_uncomp / 4;
        int dz_len = (int)dz_col.dict_uncomp / 4;

        file_info_list.insert(file_info_list.end(), {
            out_starts[si_x], out_starts[si_y], out_starts[si_z], out_starts[si_ts],
            out_starts[si_x],
            n_xyz, n_rows, FILE_SPECS[fd.spec_idx].min_points,
            dict_xyz_off,
            dict_xyz_off + dx_len,
            dict_xyz_off + dx_len + dy_len,
            dict_ts_off,
            total_xyz, total_ts, row_off_cursor, total_rows,
            (int)fi * 2
        });

        dict_xyz_off += dx_len + dy_len + dz_len;
        dict_ts_off += (int)fd.cols[3].dict_uncomp / 8;
        total_xyz += n_xyz;
        total_ts += n_rows;
        total_rows += n_rows;
        row_off_cursor += n_rows + 1;
        stream_idx += 4;
    }

    int n_files = (int)file_datas.size();

    // ── Phase 4: Snappy decompress (async GPU) ──
    auto t_offsets = torch::tensor(all_page_offsets, torch::kInt64);
    auto t_comp = torch::tensor(all_page_comp, torch::kInt64);
    auto t_uncomp = torch::tensor(all_page_uncomp, torch::kInt64);

    auto decomp_pages = batch_snappy_decompress(gpu_buffer, t_offsets, t_comp, t_uncomp);

    // ── Phase 5: Fused RLE + gather (async GPU) ──
    auto t_dpi = torch::tensor(data_page_indices, torch::kInt32);
    auto t_max_rep = torch::tensor(all_max_reps, torch::kInt32);
    auto t_max_def = torch::tensor(all_max_defs, torch::kInt32);
    auto t_num_vals = torch::tensor(all_num_vals, torch::kInt32);
    auto t_out_starts = torch::tensor(out_starts, torch::kInt32);
    auto t_xyz_dpi = torch::tensor(xyz_dict_page_indices, torch::kInt32);
    auto t_xyz_dbo = torch::tensor(xyz_dict_byte_offsets, torch::kInt32);
    auto t_ts_dpi = torch::tensor(ts_dict_page_indices, torch::kInt32);
    auto t_ts_dbo = torch::tensor(ts_dict_byte_offsets, torch::kInt32);
    auto t_fi = torch::tensor(file_info_list, torch::kInt32);

    auto kern_out = rle_gather_pipeline(
        decomp_pages, t_dpi,
        t_max_rep, t_max_def, t_num_vals, t_out_starts, total_rle_values,
        t_xyz_dpi, t_xyz_dbo, dict_xyz_byte_cursor,
        t_ts_dpi, t_ts_dbo, dict_ts_byte_cursor,
        t_fi, n_files, total_xyz, total_ts, total_rows);

    auto k_verts = kern_out[0];    // [total_xyz, 3]
    auto k_ts = kern_out[1];       // [total_ts]
    auto k_row_off = kern_out[2];  // [total_rows + n_files]

    // ── Collect CPU decode results ──
    ego_cal_future.get();
    ObsCpuResult obs_cpu;
    if (obs_cpu_future.valid()) obs_cpu = obs_cpu_future.get();

    // Run obs GPU on a separate stream so it doesn't queue behind polyline kernels.
    // unique_consecutive inside decode_obstacles_gpu syncs the current stream;
    // on a dedicated stream this only waits for obs ops, not the polyline pipeline.
    auto obs_stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false, device_index);
    {
        at::cuda::CUDAStreamGuard guard(obs_stream);
        obstacle_tensors = decode_obstacles_gpu(obs_cpu, device);
    }

    // Ego H2D on the default stream (needed by slice+xform road-boundary transform)
    if (ego_timestamps.size(0) > 0) {
        ego_ts_gpu = ego_timestamps.to(device, /*non_blocking=*/true);
        ego_tq_gpu = ego_poses_tquat.to(device, /*non_blocking=*/true);
    }

    // ── Phase 6: Slice per-file + road boundary transform (ALL ASYNC — no sync) ──
    struct SegInfo { int spec_idx; int ts_offset; int n_rows; };
    std::vector<SegInfo> seg_info;
    std::vector<int32_t> seg_off_vec, seg_size_vec;
    {
        int xyz_cursor = 0, ts_cursor = 0, roff_cursor = 0;

        for (size_t fi = 0; fi < file_datas.size(); fi++) {
            int fi_base = (int)fi * FIELDS_PER_FILE;
            int n_xyz_f = file_info_list[fi_base + 5];
            int n_rows = file_info_list[fi_base + 6];
            int spec_idx = file_order_spec_idx[fi];

            if (n_rows == 0) {
                xyz_cursor += n_xyz_f;
                ts_cursor += n_rows;
                roff_cursor += n_rows + 1;
                continue;
            }

            auto verts = k_verts.slice(0, xyz_cursor, xyz_cursor + n_xyz_f);
            auto ts = k_ts.slice(0, ts_cursor, ts_cursor + n_rows);
            auto roff = k_row_off.slice(0, roff_cursor, roff_cursor + n_rows + 1);

            // Road boundary: road_boundary_polyline is already in world frame,
            // no ego-to-world transform needed.

            output[spec_idx * 5 + 0] = verts;
            output[spec_idx * 5 + 1] = ts;
            output[spec_idx * 5 + 2] = roff;

            seg_info.push_back({spec_idx, ts_cursor, n_rows});
            seg_off_vec.push_back(ts_cursor);
            seg_size_vec.push_back(n_rows);

            xyz_cursor += n_xyz_f;
            ts_cursor += n_rows;
            roff_cursor += n_rows + 1;
        }
    }

    // Launch batched unique_consecutive kernel — async, no GPU sync!
    torch::Tensor uniq_buf, prefix_buf, n_uniq_buf;
    int n_segs = (int)seg_info.size();
    if (n_segs > 0) {
        auto seg_off_t = torch::tensor(seg_off_vec, torch::kInt32).to(device, true);
        auto seg_size_t = torch::tensor(seg_size_vec, torch::kInt32).to(device, true);
        auto ucs_result = unique_consecutive_segments(k_ts, seg_off_t, seg_size_t, n_segs);
        uniq_buf = ucs_result[0];
        prefix_buf = ucs_result[1];
        n_uniq_buf = ucs_result[2];
    }


    // Ensure obs GPU work (on separate stream) is complete before accessing obs tensors
    obs_stream.synchronize();

    // ── Ego clip bounds — async GPU ops (searchsorted), no CPU sync ──
    torch::Tensor ego_clip_lo, ego_clip_hi;
    bool do_ego_clip = false;
    bool has_obs_ts = !obstacle_tensors.empty() && obstacle_tensors[0].size(0) > 0;
    if (ego_ts_gpu.defined() && ego_ts_gpu.size(0) > 0 && (total_ts > 0 || has_obs_ts)) {
        torch::Tensor scene_min, scene_max;
        if (total_ts > 0) {
            scene_min = k_ts.min();
            scene_max = k_ts.max();
        }
        if (has_obs_ts) {
            auto obs_min = obstacle_tensors[0].min();
            auto obs_max = obstacle_tensors[0].max();
            if (scene_min.defined()) {
                scene_min = torch::min(scene_min, obs_min);
                scene_max = torch::max(scene_max, obs_max);
            } else {
                scene_min = obs_min;
                scene_max = obs_max;
            }
        }
        ego_clip_lo = torch::searchsorted(ego_ts_gpu, scene_min.reshape({1}));
        ego_clip_hi = torch::searchsorted(ego_ts_gpu, scene_max.reshape({1}),
                                           /*out_int32=*/false, /*right=*/true);
        do_ego_clip = true;
    }

    // ── THE sync — all GPU work (main stream) completes here ──
    cudaStreamSynchronize(stream);

    // ── Post-sync: finalize outputs (all GPU data ready, reads are instant) ──

    // 1. Slice batched unique results into per-file outputs
    if (n_segs > 0) {
        auto n_uniq_cpu = n_uniq_buf.cpu();
        auto* nu_ptr = n_uniq_cpu.data_ptr<int32_t>();
        for (int si = 0; si < n_segs; si++) {
            int off = seg_info[si].ts_offset;
            int nu = nu_ptr[si];
            output[seg_info[si].spec_idx * 5 + 3] = uniq_buf.slice(0, off, off + nu);
            output[seg_info[si].spec_idx * 5 + 4] = prefix_buf.slice(0, off, off + nu);
        }
    }

    // 2. Ego clipping (searchsorted results ready after sync)
    if (do_ego_clip) {
        int64_t lo = ego_clip_lo.item<int64_t>();
        int64_t hi = ego_clip_hi.item<int64_t>();
        ego_ts_gpu = ego_ts_gpu.slice(0, lo, hi);
        ego_tq_gpu = ego_tq_gpu.slice(0, lo, hi);
    }

    output[20] = ego_ts_gpu.defined() ? ego_ts_gpu : torch::empty({0}, torch::kInt64);
    output[21] = ego_tq_gpu.defined() ? ego_tq_gpu : torch::empty({0, 7}, torch::kFloat32);
    output[22] = rig_json_bytes;

    for (int i = 0; i < 7; i++) output[23 + i] = obstacle_tensors[i];

    // 3. Crosswalk triangulation (xw_roff.cpu() is instant after sync)
    {
        auto& xw_verts = output[2 * 5 + 0];
        auto& xw_roff = output[2 * 5 + 2];
        if (xw_verts.size(0) > 0 && xw_roff.size(0) > 1) {
            int64_t n_polys = xw_roff.size(0) - 1;
            auto roff_cpu = xw_roff.cpu();
            auto* rp = roff_cpu.data_ptr<int32_t>();

            auto tri_ps_cpu = torch::empty({n_polys}, torch::kInt32);
            auto varrays_ps_cpu = torch::empty({n_polys}, torch::kInt32);
            auto* tp = tri_ps_cpu.data_ptr<int32_t>();
            auto* vp = varrays_ps_cpu.data_ptr<int32_t>();
            int total_tris = 0;
            for (int64_t i = 0; i < n_polys; i++) {
                int vc = rp[i + 1] - rp[i];
                total_tris += (vc > 2) ? vc - 2 : 0;
                tp[i] = total_tris;
                vp[i] = rp[i + 1];
            }

            output[39] = varrays_ps_cpu.to(device, /*non_blocking=*/true);

            if (total_tris > 0) {
                auto triangles_cpu = torch::empty({total_tris, 3}, torch::kInt32);
                auto* tr = triangles_cpu.data_ptr<int32_t>();
                int t = 0;
                for (int64_t i = 0; i < n_polys; i++) {
                    int vc = rp[i + 1] - rp[i];
                    for (int k = 0; k < vc - 2; k++) {
                        tr[t * 3]     = 0;
                        tr[t * 3 + 1] = k + 1;
                        tr[t * 3 + 2] = k + 2;
                        t++;
                    }
                }
                output[30] = tri_ps_cpu.to(device, /*non_blocking=*/true);
                output[31] = triangles_cpu.to(device, /*non_blocking=*/true);
            }
        }
    }

    // 4. Camera H2D (submitted early from ego_cal lambda, overlapped with entire pipeline)
    if (camera_future.valid()) {
        auto cam = camera_future.get();
        if (cam.n_cameras > 0) {
            output.resize(39);
            output[32] = cam.pp.to(device, /*non_blocking=*/true);
            output[33] = cam.sz.to(device, /*non_blocking=*/true);
            output[34] = cam.fw_poly.to(device, /*non_blocking=*/true);
            output[35] = cam.ld.to(device, /*non_blocking=*/true);
            output[36] = cam.s2r.to(device, /*non_blocking=*/true);
            output[37] = cam.mra;
            output[38] = cam.names;
        }
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_scene_gpu", &load_scene_gpu,
          "Unified GPU scene loader with native ego/cal/obstacle parsing + pool-ready output",
          py::call_guard<py::gil_scoped_release>(),
          py::arg("pinned_buf"), py::arg("entry_names"), py::arg("entry_offsets"),
          py::arg("entry_sizes"), py::arg("device_index"),
          py::arg("target_w") = -1, py::arg("target_h") = -1);
    m.def("extract_rig_json", &extract_rig_json_from_pinned,
          "Extract rig JSON string from calibration_estimate.parquet in pinned buffer",
          py::arg("pinned_buf"), py::arg("cal_offset"), py::arg("cal_size"));
}
