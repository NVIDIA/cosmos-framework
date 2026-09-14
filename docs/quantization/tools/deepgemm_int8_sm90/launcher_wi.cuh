// Host launcher for the wave-interleave experiment kernel (sm90_int8_gemm_1d2d_wi_impl, see gen_kernel_wi.py).
// Same TMA descriptors / smem budget / launch as launcher.cuh (reused), only the kernel symbol differs. Only included by the
// cfg_XX.cu files that gen.py emits with a non-zero wave mode, so the default cfg ids are unaffected.
#pragma once
#include "launcher.cuh"
#include <deep_gemm/impls/sm90_int8_gemm_1d2d_wi.cuh>

namespace dgint8 {

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA, uint32_t kWaveMode>
void launch_wi(const Args& args) {
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreads = BLOCK_M <= 64 ? 128 : 256;
    constexpr uint32_t kNumSMs = 132;   // H100 SXM; checked at runtime
    constexpr uint32_t kSwizzleAB = 128;
    constexpr uint32_t kSwizzleD = swizzle_mode_for(BLOCK_N * 2);
    constexpr uint32_t TMA_D_BLOCK_N = kSwizzleD / 2;
    static_assert(smem_size_for(BLOCK_M, BLOCK_N, kNumStages, 128) <= kSmemCapacity, "too many stages");
    static_assert(kWaveMode != 0, "use launch<> for the default kernel");

    auto kernel = &deep_gemm::sm90_int8_gemm_1d2d_wi_impl<
        cute::UMMA::Major::K, 0u, SHAPE_N, SHAPE_K, 1u,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kSwizzleAB, kSwizzleAB, kSwizzleD,
        kNumStages, kNumTMAThreads, kNumMathThreads,
        kNumTMAMulticast, kIsTMAMulticastOnA,
        kNumSMs, deep_gemm::GemmType::Normal,
        cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity, kWaveMode>;

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
