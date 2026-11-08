// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int GF_BLOCK = 256;

struct FileInfo {
    int idx_x_off;      // offset in rle_idx for x column
    int idx_y_off;      // offset in rle_idx for y column
    int idx_z_off;      // offset in rle_idx for z column
    int idx_ts_off;     // offset in rle_idx for ts column
    int rep_off;        // offset in rle_rep for x column (used for row boundaries)
    int n_xyz;          // number of xyz values (same for x, y, z)
    int n_rows;         // number of rows (= ts num_values)
    int min_pts;        // minimum vertices per polyline

    int dict_x_off;     // offset in dict_xyz flat array
    int dict_y_off;
    int dict_z_off;
    int dict_ts_off;    // offset in dict_ts flat array

    int vert_out_off;   // vertex output start (index into [total_xyz, 3])
    int ts_out_off;     // timestamp output start
    int row_out_off;    // row_offsets output start
    int info_out_off;   // lengths/nan/valid/cumsum output start
    int counts_out_off; // index in counts_out (2 int64s per file)
};

// One block per file. Per-row scratch (row_starts, nan_counts) lives in
// shared memory when it fits, otherwise falls back to global memory.
__global__ void gather_and_analyze_kernel(
    const int32_t* __restrict__ rle_rep,
    const int32_t* __restrict__ rle_idx,
    const float*   __restrict__ dict_xyz,
    const int64_t* __restrict__ dict_ts,
    float*         __restrict__ vert_out,       // [total_xyz, 3]
    int64_t*       __restrict__ ts_out,         // [total_ts]
    int32_t*       __restrict__ row_offsets_out, // [total_rows + n_files]
    int32_t*       __restrict__ lengths_out,
    int32_t*       __restrict__ valid_mask_out,
    int32_t*       __restrict__ valid_cumsum_out,
    int64_t*       __restrict__ counts_out,     // [n_files, 2]
    const FileInfo* __restrict__ finfo,
    int n_files,
    int32_t*       __restrict__ g_row_starts,   // [total_rows] global fallback
    int32_t*       __restrict__ g_nan_counts,   // [total_rows] global fallback
    int smem_row_cap                            // max rows that fit in SMEM
) {
    const int fid = blockIdx.x;
    if (fid >= n_files) return;

    const FileInfo fi = finfo[fid];
    const int n_xyz = fi.n_xyz;
    const int n_rows = fi.n_rows;
    const int tid = threadIdx.x;

    extern __shared__ int smem[];
    const bool use_smem = (n_rows <= smem_row_cap);

    int* s_row_start;
    int* s_nan;
    if (use_smem) {
        s_row_start = smem;
        s_nan       = smem + n_rows;
    } else {
        s_row_start = g_row_starts + fi.info_out_off;
        s_nan       = g_nan_counts + fi.info_out_off;
    }

    // Initialize nan counts
    for (int r = tid; r < n_rows; r += GF_BLOCK)
        s_nan[r] = 0;
    __syncthreads();

    // ── Phase 1: Scan rep levels → row start positions ───────────
    if (tid == 0) {
        const int32_t* rep = rle_rep + fi.rep_off;
        int row = -1;
        for (int i = 0; i < n_xyz; i++) {
            if (rep[i] == 0) {
                row++;
                s_row_start[row] = i;
            }
        }
    }
    __syncthreads();

    // ── Phase 2: Gather xyz + NaN check ──────────────────────────
    const int32_t* ix = rle_idx + fi.idx_x_off;
    const int32_t* iy = rle_idx + fi.idx_y_off;
    const int32_t* iz = rle_idx + fi.idx_z_off;
    const float* dx = dict_xyz + fi.dict_x_off;
    const float* dy = dict_xyz + fi.dict_y_off;
    const float* dz = dict_xyz + fi.dict_z_off;
    float* vo = vert_out + fi.vert_out_off * 3;

    for (int i = tid; i < n_xyz; i += GF_BLOCK) {
        float vx = dx[ix[i]];
        float vy = dy[iy[i]];
        float vz = dz[iz[i]];
        vo[i * 3 + 0] = vx;
        vo[i * 3 + 1] = vy;
        vo[i * 3 + 2] = vz;

        if (isnan(vx) | isnan(vy) | isnan(vz)) {
            int lo = 0, hi = n_rows - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (s_row_start[mid] <= i) lo = mid; else hi = mid - 1;
            }
            atomicAdd(&s_nan[lo], 1);
        }
    }
    __syncthreads();

    // ── Phase 3: Gather timestamps ───────────────────────────────
    const int32_t* its = rle_idx + fi.idx_ts_off;
    const int64_t* dts = dict_ts + fi.dict_ts_off;
    int64_t* to = ts_out + fi.ts_out_off;

    for (int i = tid; i < n_rows; i += GF_BLOCK)
        to[i] = dts[its[i]];

    // ── Phase 4: Compute row lengths, valid mask, prefix sum ─────
    // Write row_offsets, lengths, valid_mask, valid_cumsum to global mem.
    // Thread 0 does a serial pass — n_rows is small (< 2K typically).
    if (tid == 0) {
        int32_t* ro = row_offsets_out + fi.row_out_off;
        int32_t* lo = lengths_out + fi.info_out_off;
        int32_t* vm = valid_mask_out + fi.info_out_off;
        int32_t* vc = valid_cumsum_out + fi.info_out_off;

        int offset = 0;
        int valid_sum = 0;
        int64_t total_verts = 0;
        ro[0] = 0;

        for (int r = 0; r < n_rows; r++) {
            int start = s_row_start[r];
            int end = (r + 1 < n_rows) ? s_row_start[r + 1] : n_xyz;
            int len = end - start;

            offset += len;
            ro[r + 1] = offset;
            lo[r] = len;

            int valid = (len >= fi.min_pts && s_nan[r] == 0) ? 1 : 0;
            vm[r] = valid;
            valid_sum += valid;
            vc[r] = valid_sum;
            if (valid)
                total_verts += len;
        }

        counts_out[fi.counts_out_off]     = (int64_t)valid_sum;
        counts_out[fi.counts_out_off + 1] = total_verts;
    }
}

std::vector<torch::Tensor> gather_and_analyze(
    torch::Tensor rle_rep,
    torch::Tensor rle_idx,
    torch::Tensor dict_xyz,
    torch::Tensor dict_ts,
    torch::Tensor file_info_raw,   // FileInfo structs as raw bytes on CPU
    int n_files,
    int total_xyz_values,
    int total_ts_values,
    int total_rows
) {
    auto device = rle_rep.device();
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

    auto vertices   = torch::empty({total_xyz_values, 3}, f32);
    auto timestamps = torch::empty({total_ts_values}, i64);
    auto row_off    = torch::empty({total_rows + n_files}, i32);
    auto lengths    = torch::empty({total_rows}, i32);
    auto valid_mask = torch::empty({total_rows}, i32);
    auto valid_cum  = torch::empty({total_rows}, i32);
    auto counts     = torch::empty({n_files * 2}, i64);

    constexpr int FI_INTS = sizeof(FileInfo) / sizeof(int32_t);
    const FileInfo* fi_cpu = reinterpret_cast<const FileInfo*>(
        file_info_raw.data_ptr<int32_t>());

    // Query hardware SMEM limit → max rows that fit in 2*n_rows*sizeof(int)
    int dev;
    cudaGetDevice(&dev);
    int max_smem_bytes;
    cudaDeviceGetAttribute(&max_smem_bytes,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int smem_row_cap = max_smem_bytes / (2 * (int)sizeof(int));

    // Partition files: smem-fitting vs global-fallback
    std::vector<int> smem_ids, glob_ids;
    smem_ids.reserve(n_files);
    glob_ids.reserve(n_files);
    for (int f = 0; f < n_files; f++) {
        if (fi_cpu[f].n_rows <= smem_row_cap)
            smem_ids.push_back(f);
        else
            glob_ids.push_back(f);
    }

    // Global-memory scratch shared by both launches (only used by global path)
    auto g_row_starts = torch::empty({total_rows}, i32);
    auto g_nan_counts = torch::zeros({total_rows}, i32);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Helper: build a group's FileInfo tensor, upload, and launch kernel
    auto launch_group = [&](const std::vector<int>& ids, bool use_smem) {
        if (ids.empty()) return;
        int ng = (int)ids.size();

        auto group_raw = torch::empty({ng * FI_INTS},
            torch::TensorOptions().dtype(torch::kInt32));
        FileInfo* gp = reinterpret_cast<FileInfo*>(group_raw.data_ptr<int32_t>());
        int group_max_rows = 0;
        for (int i = 0; i < ng; i++) {
            gp[i] = fi_cpu[ids[i]];
            if (gp[i].n_rows > group_max_rows)
                group_max_rows = gp[i].n_rows;
        }

        int smem_bytes = 0;
        int cap = 0;
        if (use_smem) {
            smem_bytes = 2 * group_max_rows * (int)sizeof(int);
            cap = smem_row_cap;
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(
                    gather_and_analyze_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes);
            }
        }

        auto group_gpu = group_raw.to(device);

        gather_and_analyze_kernel<<<ng, GF_BLOCK, smem_bytes, stream>>>(
            rle_rep.data_ptr<int32_t>(),
            rle_idx.data_ptr<int32_t>(),
            dict_xyz.data_ptr<float>(),
            dict_ts.data_ptr<int64_t>(),
            vertices.data_ptr<float>(),
            timestamps.data_ptr<int64_t>(),
            row_off.data_ptr<int32_t>(),
            lengths.data_ptr<int32_t>(),
            valid_mask.data_ptr<int32_t>(),
            valid_cum.data_ptr<int32_t>(),
            counts.data_ptr<int64_t>(),
            reinterpret_cast<const FileInfo*>(group_gpu.data_ptr<int32_t>()),
            ng,
            g_row_starts.data_ptr<int32_t>(),
            g_nan_counts.data_ptr<int32_t>(),
            cap
        );
    };

    // Launch 1: SMEM path — small SMEM allocation, high occupancy
    launch_group(smem_ids, true);
    // Launch 2: Global path — zero SMEM, no occupancy penalty
    launch_group(glob_ids, false);

    return {vertices, timestamps, row_off, lengths, valid_mask, valid_cum, counts};
}
