// Host-side launch arguments shared by the torch bindings (g++) and the nvcc-compiled launchers.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace dgint8 {

struct Args {
    const int8_t* a;      // [m, k] row-major (K-major)
    const int8_t* b;      // [n, k] row-major (K-major), i.e. nn.Linear weight
    const float* sfa;     // [ceil(k/128), sfa_ld] fp32, MN-major TMA-aligned: sfa[kb * sfa_ld + m], sfa_ld = round_up(m, 4)
    const float* sfb;     // [n/128, ceil(k/128)] fp32 contiguous (K-major)
    void* d;              // [m, n] bf16 row-major
    int m, n, k;
    int sfa_ld;
    cudaStream_t stream;
};

struct CfgInfo {
    int block_m, block_n, stages, mcast, on_a, shape_n, shape_k, math_threads, smem_max;
    const char* name;
};

}  // namespace dgint8
