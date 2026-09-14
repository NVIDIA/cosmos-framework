// Host launcher for the INT8 port of DeepGEMM's SM90 1D2D kernel: TMA descriptor construction mirrors
// csrc/jit_kernels/impls/runtime_utils.hpp (make_tma_a/b/cd/sf_desc), the launch mirrors deep_jit's cuLaunchKernelEx
// (cluster dims attribute, dynamic smem) and the smem budget mirrors csrc/jit_kernels/heuristics/sm90.hpp.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <deep_gemm/impls/sm90_int8_gemm_1d2d.cuh>
#include "dgint8_args.h"

namespace dgint8 {

#define DGINT8_CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(_e) + " at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                   const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

inline EncodeTiledFn get_encode_fn() {
    static EncodeTiledFn fn = nullptr;
    if (fn == nullptr) {
        void* ptr = nullptr;
        cudaDriverEntryPointQueryResult status;
        DGINT8_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status));
        if (ptr == nullptr or status != cudaDriverEntryPointSuccess)
            throw std::runtime_error("cuTensorMapEncodeTiled not available from the driver");
        fn = reinterpret_cast<EncodeTiledFn>(ptr);
    }
    return fn;
}

inline CUtensorMapSwizzle to_swizzle(int mode) {
    switch (mode) {
        case 0: case 16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    }
    throw std::runtime_error("bad swizzle mode");
}

// 2D tiled TMA descriptor: gmem dims {inner, outer}, box {box_inner, box_outer}; outer stride in elements.
inline CUtensorMap make_tma_2d(const void* ptr, CUtensorMapDataType dtype, int elem_size,
                               uint64_t gmem_inner, uint64_t gmem_outer, uint64_t gmem_outer_stride_elems,
                               uint32_t box_inner, uint32_t box_outer, int swizzle_mode) {
    if (swizzle_mode != 0)
        box_inner = swizzle_mode / elem_size;   // TMA box inner dim is at most one swizzle atom (runtime_utils.hpp does the same)
    if (reinterpret_cast<uintptr_t>(ptr) % 16 != 0) throw std::runtime_error("TMA base pointer must be 16B aligned");
    if ((gmem_outer_stride_elems * elem_size) % 16 != 0) throw std::runtime_error("TMA outer stride must be a multiple of 16B");
    CUtensorMap map;
    const cuuint64_t gmem_dims[2] = {gmem_inner, gmem_outer};
    const cuuint64_t gmem_strides[1] = {gmem_outer_stride_elems * static_cast<uint64_t>(elem_size)};
    const cuuint32_t smem_dims[2] = {box_inner, box_outer};
    const cuuint32_t elem_strides[2] = {1, 1};
    const CUresult r = get_encode_fn()(&map, dtype, 2, const_cast<void*>(ptr), gmem_dims, gmem_strides, smem_dims, elem_strides,
                                       CU_TENSOR_MAP_INTERLEAVE_NONE, to_swizzle(swizzle_mode),
                                       CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
        throw std::runtime_error("cuTensorMapEncodeTiled failed with code " + std::to_string(static_cast<int>(r)) +
                                 " (inner=" + std::to_string(gmem_inner) + ", outer=" + std::to_string(gmem_outer) +
                                 ", box=" + std::to_string(box_inner) + "x" + std::to_string(box_outer) + ")");
    return map;
}

constexpr int ceil_div_i(int a, int b) { return (a + b - 1) / b; }
constexpr int align_i(int a, int b) { return ceil_div_i(a, b) * b; }

// heuristics/utils.hpp::get_swizzle_mode
constexpr int swizzle_mode_for(int block_size_bytes) {
    return block_size_bytes % 128 == 0 ? 128 : block_size_bytes % 64 == 0 ? 64 : block_size_bytes % 32 == 0 ? 32 : 16;
}

// heuristics/sm90.hpp::get_pipeline_config (Kernel1D2D, bf16 out): smem for a given K
constexpr int smem_size_for(int block_m, int block_n, int stages, int k) {
    const int smem_cd = align_i(block_m * block_n * 2, 1024);
    const int smem_barriers = 16 * 8 * 2;
    const int per_stage = block_m * 128 + block_n * 128 + align_i(block_m * 4, 128);
    const int use_uniform_sfb = (128 % block_n == 0) ? 1 : 2;
    const int smem_extra_sfb = align_i(ceil_div_i(k, 128) * 4 * use_uniform_sfb, 8);
    return smem_cd + smem_barriers + smem_extra_sfb + stages * per_stage;
}
constexpr int kSmemCapacity = 232448;

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA>
void launch(const Args& args) {
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreads = BLOCK_M <= 64 ? 128 : 256;
    constexpr uint32_t kNumSMs = 132;   // H100 SXM; checked at runtime
    constexpr uint32_t kSwizzleAB = 128;                                   // 128 int8 = 128B per K-row
    constexpr uint32_t kSwizzleD = swizzle_mode_for(BLOCK_N * 2);          // bf16 output
    constexpr uint32_t TMA_D_BLOCK_N = kSwizzleD / 2;
    static_assert(smem_size_for(BLOCK_M, BLOCK_N, kNumStages, 128) <= kSmemCapacity, "too many stages");

    auto kernel = &deep_gemm::sm90_int8_gemm_1d2d_impl<
        cute::UMMA::Major::K, 0u, SHAPE_N, SHAPE_K, 1u,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kSwizzleAB, kSwizzleAB, kSwizzleD,
        kNumStages, kNumTMAThreads, kNumMathThreads,
        kNumTMAMulticast, kIsTMAMulticastOnA,
        kNumSMs, deep_gemm::GemmType::Normal,
        cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity>;

    const int m = args.m, n = args.n, k = args.k;
    if (SHAPE_N != 0 and static_cast<int>(SHAPE_N) != n) throw std::runtime_error("kernel compiled for a different N");
    if (SHAPE_K != 0 and static_cast<int>(SHAPE_K) != k) throw std::runtime_error("kernel compiled for a different K");
    if (k % 128 != 0 or n % 128 != 0) throw std::runtime_error("need K % 128 == 0 and N % 128 == 0");
    if (args.sfa_ld % 4 != 0 or args.sfa_ld < m) throw std::runtime_error("sfa leading dim must be round_up(M, 4)");
    const int smem_size = smem_size_for(BLOCK_M, BLOCK_N, kNumStages, k);
    if (smem_size > kSmemCapacity) throw std::runtime_error("smem budget exceeded for this K");

    static bool attr_set = false;
    if (not attr_set) {
        int dev = 0, num_sms = 0;
        DGINT8_CUDA_CHECK(cudaGetDevice(&dev));
        DGINT8_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
        if (num_sms != static_cast<int>(kNumSMs)) throw std::runtime_error("kernel compiled for 132 SMs, device has " + std::to_string(num_sms));
        DGINT8_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemCapacity));
        attr_set = true;
    }

    // TMA descriptors (runtime_utils.hpp: make_tma_a_desc / make_tma_b_desc / make_tma_cd_desc / make_tma_sf_desc)
    const CUtensorMap tma_a = make_tma_2d(args.a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, k, m, k, BLOCK_K, BLOCK_M, kSwizzleAB);
    const CUtensorMap tma_b = make_tma_2d(args.b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, k, n, k, BLOCK_K, BLOCK_N, kSwizzleAB);
    const CUtensorMap tma_d = make_tma_2d(args.d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, n, m, n, TMA_D_BLOCK_N, BLOCK_M, kSwizzleD);
    const CUtensorMap tma_sfa = make_tma_2d(args.sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, args.sfa_ld, ceil_div_i(k, 128), args.sfa_ld,
                                            BLOCK_M, 1, 0);

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(kNumSMs, 1, 1);
    config.blockDim = dim3(kNumTMAThreads + kNumMathThreads, 1, 1);
    config.dynamicSmemBytes = smem_size;
    config.stream = args.stream;
    cudaLaunchAttribute attrs[1];
    config.numAttrs = 0;
    if constexpr (kNumTMAMulticast > 1) {
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = kNumTMAMulticast; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs; config.numAttrs = 1;
    }
    DGINT8_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel,
        const_cast<float*>(args.sfb), static_cast<int*>(nullptr),
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        tma_a, tma_b, tma_d, tma_sfa));
}

}  // namespace dgint8
