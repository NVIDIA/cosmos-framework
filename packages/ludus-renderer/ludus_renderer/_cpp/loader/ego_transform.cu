// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void ego_transform_kernel(
    float* __restrict__ verts,
    const int64_t* __restrict__ ts,
    const int32_t* __restrict__ roff,
    const int64_t* __restrict__ ego_ts,
    const float* __restrict__ ego_tq,
    int n_verts,
    int n_rows,
    int n_ego
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_verts) return;

    // Binary search roff to find which row this vertex belongs to.
    // roff[i] gives the cumulative vertex count; find largest i where roff[i] <= v + roff[0].
    int base = roff[0];
    int target = v + base;
    int lo = 0, hi = n_rows;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (roff[mid] <= target) lo = mid;
        else hi = mid - 1;
    }

    // Interpolation bracket in ego_ts (equivalent to searchsorted + clamp)
    int64_t timestamp = ts[lo];
    int elo = 0, ehi = n_ego;
    while (elo < ehi) {
        int mid = (elo + ehi) >> 1;
        if (ego_ts[mid] < timestamp) elo = mid + 1;
        else ehi = mid;
    }
    int idx = elo;
    if (idx < 1) idx = 1;
    if (idx >= n_ego) idx = n_ego - 1;

    // Interpolation weight
    int64_t t_lo = ego_ts[idx - 1];
    int64_t dt = ego_ts[idx] - t_lo;
    if (dt < 1) dt = 1;
    float alpha = (float)(timestamp - t_lo) / (float)dt;
    float one_minus_alpha = 1.0f - alpha;

    // Interpolate translation + quaternion from ego_tq [n_ego, 7]: tx ty tz qx qy qz qw
    const float* tq0 = ego_tq + (idx - 1) * 7;
    const float* tq1 = ego_tq + idx * 7;

    float tx = tq0[0] * one_minus_alpha + tq1[0] * alpha;
    float ty = tq0[1] * one_minus_alpha + tq1[1] * alpha;
    float tz = tq0[2] * one_minus_alpha + tq1[2] * alpha;

    float qx = tq0[3] * one_minus_alpha + tq1[3] * alpha;
    float qy = tq0[4] * one_minus_alpha + tq1[4] * alpha;
    float qz = tq0[5] * one_minus_alpha + tq1[5] * alpha;
    float qw = tq0[6] * one_minus_alpha + tq1[6] * alpha;

    // Normalize quaternion
    float inv_norm = rsqrtf(fmaxf(qx*qx + qy*qy + qz*qz + qw*qw, 1e-16f));
    qx *= inv_norm; qy *= inv_norm; qz *= inv_norm; qw *= inv_norm;

    // Quaternion (xyzw) to rotation matrix, then rotate + translate the vertex
    float r00 = 1 - 2*(qy*qy + qz*qz), r01 = 2*(qx*qy - qz*qw), r02 = 2*(qx*qz + qy*qw);
    float r10 = 2*(qx*qy + qz*qw), r11 = 1 - 2*(qx*qx + qz*qz), r12 = 2*(qy*qz - qx*qw);
    float r20 = 2*(qx*qz - qy*qw), r21 = 2*(qy*qz + qx*qw), r22 = 1 - 2*(qx*qx + qy*qy);

    float vx = verts[v * 3], vy = verts[v * 3 + 1], vz = verts[v * 3 + 2];

    verts[v * 3]     = r00*vx + r01*vy + r02*vz + tx;
    verts[v * 3 + 1] = r10*vx + r11*vy + r12*vz + ty;
    verts[v * 3 + 2] = r20*vx + r21*vy + r22*vz + tz;
}

void ego_transform_fused(
    torch::Tensor verts,
    torch::Tensor ts,
    torch::Tensor roff,
    torch::Tensor ego_ts,
    torch::Tensor ego_tq
) {
    int n_verts = verts.size(0);
    int n_rows = ts.size(0);
    int n_ego = ego_ts.size(0);
    if (n_verts == 0 || n_rows == 0 || n_ego == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(verts.device().index()).stream();

    constexpr int THREADS = 256;
    int blocks = (n_verts + THREADS - 1) / THREADS;

    ego_transform_kernel<<<blocks, THREADS, 0, stream>>>(
        verts.data_ptr<float>(),
        ts.data_ptr<int64_t>(),
        roff.data_ptr<int32_t>(),
        ego_ts.data_ptr<int64_t>(),
        ego_tq.data_ptr<float>(),
        n_verts, n_rows, n_ego
    );
}
