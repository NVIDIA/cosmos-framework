// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Fused CUDA kernels for obstacle post-processing
//
// Replaces ~40 ATen ops + 4 GPU syncs in decode_obstacles_gpu with
// 6 custom kernels + ~10 ATen ops + 2 GPU syncs.
//
// Key savings:
//   - yaw→quat: 1 kernel replaces 6 ATen ops (atan2, mul, sin, cos, 2× copy_)
//   - group assignment: boundary mark + cumsum replaces unique_consecutive (avoids 1 sync)
//   - pose filtering: cumsum-based compaction replaces 2× nonzero (avoids 2 syncs)
//   - output gathering: 2 kernels replace ~12 ATen gather/stack/build ops
// ---------------------------------------------------------------------------

__global__ void obs_yaw_to_quat_kernel(float* __restrict__ packed, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float dx = packed[i * 8 + 3];
    float dy = packed[i * 8 + 4];
    float half_yaw = atan2f(dy, dx) * 0.5f;
    packed[i * 8 + 3] = sinf(half_yaw);
    packed[i * 8 + 4] = cosf(half_yaw);
}

__global__ void obs_mark_boundaries_kernel(
    const double* __restrict__ sorted_id, int* __restrict__ boundary, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    boundary[i] = (i == 0) ? 0 : (fabs(sorted_id[i] - sorted_id[i - 1]) > 0.5 ? 1 : 0);
}

__global__ void obs_group_sizes_kernel(
    const int* __restrict__ group_id, int* __restrict__ group_sizes, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    atomicAdd(&group_sizes[group_id[i]], 1);
}

__global__ void obs_validity_kernel(
    const int* __restrict__ group_sizes,
    const int* __restrict__ group_id,
    int* __restrict__ pose_valid,
    int* __restrict__ group_last,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int gid = group_id[i];
    int valid = (group_sizes[gid] >= 2) ? 1 : 0;
    pose_valid[i] = valid;
    int is_last = (i == N - 1) || (group_id[i + 1] != gid);
    group_last[i] = is_last ? valid : 0;
}

__global__ void obs_gather_poses_kernel(
    const int* __restrict__ pose_valid,
    const int* __restrict__ compact_idx,
    const int64_t* __restrict__ sorted_ts,
    const float* __restrict__ sorted_pack,
    int64_t* __restrict__ out_ts,
    float* __restrict__ out_trans,
    float* __restrict__ out_quat,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !pose_valid[i]) return;
    int j = compact_idx[i] - 1;
    out_ts[j] = sorted_ts[i];
    const float* sp = sorted_pack + i * 8;
    out_trans[j * 3]     = sp[0];
    out_trans[j * 3 + 1] = sp[1];
    out_trans[j * 3 + 2] = sp[2];
    out_quat[j * 4]     = 0.0f;
    out_quat[j * 4 + 1] = 0.0f;
    out_quat[j * 4 + 2] = sp[3];
    out_quat[j * 4 + 3] = sp[4];
}

__global__ void obs_gather_tracks_kernel(
    const int* __restrict__ group_last,
    const int* __restrict__ track_compact_idx,
    const int* __restrict__ group_sizes,
    const int* __restrict__ group_id,
    const float* __restrict__ sorted_pack,
    const int64_t* __restrict__ sorted_class,
    bool use_class,
    float* __restrict__ out_scales,
    float* __restrict__ out_colors,
    int* __restrict__ out_lengths,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !group_last[i]) return;
    int t = track_compact_idx[i] - 1;
    const float* sp = sorted_pack + i * 8;
    out_scales[t * 3]     = sp[5] * 2.0f;
    out_scales[t * 3 + 1] = sp[6] * 2.0f;
    out_scales[t * 3 + 2] = sp[7] * 2.0f;
    out_lengths[t] = group_sizes[group_id[i]];

    int cidx = 0;
    if (use_class) {
        int64_t cls = sorted_class[i];
        if (cls == 1282 || cls == 1283 || cls == 1284 || cls == 1285 ||
            cls == 3329 || cls == 3330) cidx = 1;
        else if (cls == 2308 || cls == 2309) cidx = 2;
        else if (cls == 2305 || cls == 2306 || cls == 2307) cidx = 3;
    }
    const float palette[] = {
        0.f/255, 46.f/255, 136.f/255,  126.f/255, 206.f/255, 255.f/255,
        204.f/255, 55.f/255, 0.f/255,  255.f/255, 192.f/255, 64.f/255,
        148.f/255, 0.f/255, 62.f/255,  255.f/255, 124.f/255, 171.f/255,
        0.f/255, 80.f/255, 66.f/255,   102.f/255, 208.f/255, 198.f/255,
        53.f/255, 26.f/255, 20.f/255,  166.f/255, 136.f/255, 125.f/255,
    };
    for (int c = 0; c < 6; c++)
        out_colors[t * 6 + c] = palette[cidx * 6 + c];
}


// ---------------------------------------------------------------------------
// Batched unique_consecutive per segment — no CPU sync needed
// Each thread handles one segment (sequential scan, fine for ~100-1000 elements)
// ---------------------------------------------------------------------------
__global__ void unique_consecutive_segments_kernel(
    const int64_t* __restrict__ ts,
    const int* __restrict__ seg_offsets,
    const int* __restrict__ seg_sizes,
    int n_segs,
    int64_t* __restrict__ out_unique,
    int32_t* __restrict__ out_prefix,
    int32_t* __restrict__ out_n_unique
) {
    int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segs) return;

    int off = seg_offsets[seg];
    int n = seg_sizes[seg];
    if (n == 0) { out_n_unique[seg] = 0; return; }

    int nu = 0;
    int64_t prev = ts[off];
    int cnt = 1;
    int32_t running = 0;

    for (int i = 1; i < n; i++) {
        int64_t v = ts[off + i];
        if (v != prev) {
            out_unique[off + nu] = prev;
            running += cnt;
            out_prefix[off + nu] = running;
            nu++;
            cnt = 1;
            prev = v;
        } else {
            cnt++;
        }
    }
    out_unique[off + nu] = prev;
    running += cnt;
    out_prefix[off + nu] = running;
    nu++;
    out_n_unique[seg] = nu;
}

std::vector<torch::Tensor> unique_consecutive_segments(
    torch::Tensor ts,
    torch::Tensor seg_offsets,
    torch::Tensor seg_sizes,
    int n_segs
) {
    auto device = ts.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    int64_t total = ts.size(0);

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    auto out_unique = torch::empty({total}, opts_i64);
    auto out_prefix = torch::empty({total}, opts_i32);
    auto out_n_unique = torch::empty({n_segs}, opts_i32);

    if (n_segs > 0 && total > 0) {
        unique_consecutive_segments_kernel<<<n_segs, 1, 0, stream>>>(
            ts.data_ptr<int64_t>(),
            seg_offsets.data_ptr<int>(),
            seg_sizes.data_ptr<int>(),
            n_segs,
            out_unique.data_ptr<int64_t>(),
            out_prefix.data_ptr<int32_t>(),
            out_n_unique.data_ptr<int32_t>());
    }
    return {out_unique, out_prefix, out_n_unique};
}

// ---------------------------------------------------------------------------
// C++ wrapper: in-place yaw → quaternion on packed[N, 8] GPU tensor
// ---------------------------------------------------------------------------
void obs_yaw_to_quat(torch::Tensor packed) {
    int N = packed.size(0);
    if (N == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream(packed.device().index()).stream();
    constexpr int T = 256;
    obs_yaw_to_quat_kernel<<<(N + T - 1) / T, T, 0, stream>>>(
        packed.data_ptr<float>(), N);
}

// ---------------------------------------------------------------------------
// C++ wrapper: group-by-id + filter + gather with only 2 GPU syncs
//
// Takes argsorted obstacle data and produces the 7 CubePool output tensors.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> obs_group_and_gather(
    torch::Tensor sorted_id,     // [N] float64
    torch::Tensor sorted_ts,     // [N] int64
    torch::Tensor sorted_pack,   // [N, 8] float32
    torch::Tensor sorted_class,  // [N] int64, or empty
    bool use_class
) {
    int64_t N = sorted_id.size(0);
    auto device = sorted_id.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    constexpr int T = 256;
    int blocks = (int)((N + T - 1) / T);

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    auto empty7 = [&]() -> std::vector<torch::Tensor> {
        return {
            torch::empty({0}, opts_i64), torch::empty({0}, opts_i32),
            torch::empty({0}, opts_i64), torch::empty({0, 3}, opts_f32),
            torch::empty({0, 4}, opts_f32), torch::empty({0, 3}, opts_f32),
            torch::empty({0, 6}, opts_f32),
        };
    };

    // Step 1: boundary marks → cumsum → group_id  (0 syncs)
    auto boundary = torch::empty({N}, opts_i32);
    obs_mark_boundaries_kernel<<<blocks, T, 0, stream>>>(
        sorted_id.data_ptr<double>(), boundary.data_ptr<int>(), (int)N);

    auto group_id = boundary.cumsum(0, torch::kInt32);

    // Step 2: group sizes via atomicAdd (over-allocate to N, avoids sync for n_groups)
    auto group_sizes = torch::zeros({N}, opts_i32);
    obs_group_sizes_kernel<<<blocks, T, 0, stream>>>(
        group_id.data_ptr<int>(), group_sizes.data_ptr<int>(), (int)N);

    // Step 3: mark valid poses + last-in-group  (0 syncs)
    auto pose_valid = torch::empty({N}, opts_i32);
    auto group_last = torch::empty({N}, opts_i32);
    obs_validity_kernel<<<blocks, T, 0, stream>>>(
        group_sizes.data_ptr<int>(), group_id.data_ptr<int>(),
        pose_valid.data_ptr<int>(), group_last.data_ptr<int>(), (int)N);

    // Step 4: compaction indices via cumsum  (0 syncs)
    auto pose_compact = pose_valid.cumsum(0, torch::kInt32);
    auto track_compact = group_last.cumsum(0, torch::kInt32);

    // SYNC 1: read total_poses and n_tracks (batch into one GPU→CPU copy)
    auto info = torch::stack({pose_compact.select(0, N - 1),
                              track_compact.select(0, N - 1)});
    auto info_cpu = info.cpu();
    int64_t total_poses = info_cpu[0].item<int64_t>();
    int64_t n_tracks   = info_cpu[1].item<int64_t>();

    if (n_tracks == 0) return empty7();

    // Step 5: gather poses
    auto out_ts    = torch::empty({total_poses}, opts_i64);
    auto out_trans = torch::empty({total_poses, 3}, opts_f32);
    auto out_quat  = torch::empty({total_poses, 4}, opts_f32);

    obs_gather_poses_kernel<<<blocks, T, 0, stream>>>(
        pose_valid.data_ptr<int>(), pose_compact.data_ptr<int>(),
        sorted_ts.data_ptr<int64_t>(), sorted_pack.data_ptr<float>(),
        out_ts.data_ptr<int64_t>(), out_trans.data_ptr<float>(),
        out_quat.data_ptr<float>(), (int)N);

    // Step 6: gather tracks (scales, colors, lengths)
    auto out_scales  = torch::empty({n_tracks, 3}, opts_f32);
    auto out_colors  = torch::empty({n_tracks, 6}, opts_f32);
    auto out_lengths = torch::empty({n_tracks}, opts_i32);

    int64_t* sc_ptr = (use_class && sorted_class.defined() && sorted_class.size(0) == N)
        ? sorted_class.data_ptr<int64_t>() : nullptr;

    obs_gather_tracks_kernel<<<blocks, T, 0, stream>>>(
        group_last.data_ptr<int>(), track_compact.data_ptr<int>(),
        group_sizes.data_ptr<int>(), group_id.data_ptr<int>(),
        sorted_pack.data_ptr<float>(),
        sc_ptr, sc_ptr != nullptr,
        out_scales.data_ptr<float>(), out_colors.data_ptr<float>(),
        out_lengths.data_ptr<int>(), (int)N);

    // Step 7: track prefix sum
    auto track_ps = out_lengths.cumsum(0, torch::kInt32);

    // Step 8: global unique sorted timestamps (SYNC 2 — inside unique_consecutive)
    auto sorted_out_ts = std::get<0>(torch::sort(out_ts));
    auto global_ts = std::get<0>(torch::unique_consecutive(sorted_out_ts));

    return {global_ts, track_ps, out_ts, out_trans, out_quat, out_scales, out_colors};
}
