// Host launcher for the OCC (occupancy-2, approach D) variants: launcher.cuh helpers + kernels sm90_int8_gemm_1d2d_occ_impl /
// sm90_int8_gemm_1d2d_occdb_impl with kMinBlocksPerSM co-resident CTAs per SM, a persistent grid of 132 * kMinBlocksPerSM CTAs,
// and a one-time report of the REAL occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor / cudaOccupancyMaxActiveClusters),
// register count and local (spill) bytes of the compiled kernel.
#pragma once
#include "../launcher.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "sm90_int8_gemm_1d2d_occ.cuh"
#include "sm90_int8_gemm_1d2d_occdb.cuh"
#include "../dgint8_args.h"
#include "occ_report.h"

namespace dgint8 {

// exact smem layout of the kernel: D tile | A stages | B stages | SFA stages (128B aligned) | SFB (8B aligned) | 2 barriers per stage
constexpr int occ_smem_size_for(int block_m, int block_n, int stages, int k, bool half_d = false) {
    const int smem_cd = align_i(block_m * (half_d ? swizzle_mode_for(block_n * 2) / 2 : block_n) * 2, 1024);   // kHalfD: one swizzle atom
    const int per_stage = block_m * 128 + block_n * 128 + align_i(block_m * 4, 128);
    const int use_uniform_sfb = (128 % block_n == 0) ? 1 : 2;
    const int smem_extra_sfb = align_i(ceil_div_i(k, 128) * 4 * use_uniform_sfb, 8);
    const int smem_barriers = stages * 2 * 8;
    return smem_cd + stages * per_stage + smem_extra_sfb + smem_barriers;
}
constexpr int kOccSmemPerSM = 233472;        // 228 KB
constexpr int kOccSmemReservedPerCTA = 1024;  // cudaDevAttrReservedSharedMemoryPerBlock on sm_90
constexpr int occ_smem_limit(int min_blocks) { return kOccSmemPerSM / min_blocks - kOccSmemReservedPerCTA; }

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumStages,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA, uint32_t kNumMathThreads, uint32_t kMinBlocksPerSM, uint32_t kMathRegs, bool kDB, bool kHalfD>
void launch_occ(const Args& args, const char* name, OccReport* report = nullptr) {
    constexpr uint32_t BLOCK_K = 128;
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumSMs = 132 * kMinBlocksPerSM;   // persistent grid = all co-resident CTAs; H100 SXM, checked at runtime
    constexpr uint32_t kSwizzleAB = 128;
    constexpr uint32_t kSwizzleD = swizzle_mode_for(BLOCK_N * 2);
    constexpr uint32_t TMA_D_BLOCK_N = kSwizzleD / 2;
    static_assert(kNumMathThreads == 128 or kNumMathThreads == 256, "math threads");
    static_assert(BLOCK_M >= 64 or kNumMathThreads == 128, "one math warpgroup for BLOCK_M < 64");
    static_assert(not kDB or (BLOCK_M == 64 and kNumMathThreads == 128) or (BLOCK_M == 128 and kNumMathThreads == 256), "DB needs one 64-row wave per warpgroup");
    static_assert(occ_smem_size_for(BLOCK_M, BLOCK_N, kNumStages, 128, kHalfD) <= kSmemCapacity, "too many stages");
    static_assert(128 * 40 + kNumMathThreads * kMathRegs <= (65536 / kMinBlocksPerSM) / 256 * 256 || kMinBlocksPerSM == 1, "setmaxnreg budget exceeds the per-CTA register pool");

    auto kernel = [] {
        if constexpr (kDB)
            return &deep_gemm::sm90_int8_gemm_1d2d_occdb_impl<
                cute::UMMA::Major::K, 0u, SHAPE_N, SHAPE_K, 1u, BLOCK_M, BLOCK_N, BLOCK_K, kSwizzleAB, kSwizzleAB, kSwizzleD,
                kNumStages, kNumTMAThreads, kNumMathThreads, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, deep_gemm::GemmType::Normal,
                cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity, kMinBlocksPerSM, kMathRegs, kHalfD>;
        else
            return &deep_gemm::sm90_int8_gemm_1d2d_occ_impl<
                cute::UMMA::Major::K, 0u, SHAPE_N, SHAPE_K, 1u, BLOCK_M, BLOCK_N, BLOCK_K, kSwizzleAB, kSwizzleAB, kSwizzleD,
                kNumStages, kNumTMAThreads, kNumMathThreads, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, deep_gemm::GemmType::Normal,
                cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity, kMinBlocksPerSM, kMathRegs, kHalfD>;
    }();

    const int m = args.m, n = args.n, k = args.k;
    if (SHAPE_N != 0 and static_cast<int>(SHAPE_N) != n) throw std::runtime_error("kernel compiled for a different N");
    if (SHAPE_K != 0 and static_cast<int>(SHAPE_K) != k) throw std::runtime_error("kernel compiled for a different K");
    if (k % 128 != 0 or n % 128 != 0) throw std::runtime_error("need K % 128 == 0 and N % 128 == 0");
    if (args.sfa_ld % 4 != 0 or args.sfa_ld < m) throw std::runtime_error("sfa leading dim must be round_up(M, 4)");
    const int smem_size = occ_smem_size_for(BLOCK_M, BLOCK_N, kNumStages, k, kHalfD);
    if (smem_size > kSmemCapacity) throw std::runtime_error("smem budget exceeded for this K");

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

    static OccReport rep;
    static bool attr_set = false;
    if (not attr_set) {
        int dev = 0, num_sms = 0;
        DGINT8_CUDA_CHECK(cudaGetDevice(&dev));
        DGINT8_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
        if (num_sms * static_cast<int>(kMinBlocksPerSM) != static_cast<int>(kNumSMs))
            throw std::runtime_error("kernel compiled for 132 SMs, device has " + std::to_string(num_sms));
        DGINT8_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemCapacity));
        DGINT8_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, static_cast<int>(cudaSharedmemCarveoutMaxShared)));
        int nb = 0;
        DGINT8_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, kNumTMAThreads + kNumMathThreads, smem_size));
        cudaFuncAttributes fa;
        DGINT8_CUDA_CHECK(cudaFuncGetAttributes(&fa, kernel));
        rep.occupancy = nb; rep.regs = fa.numRegs; rep.local_bytes = static_cast<int>(fa.localSizeBytes); rep.smem = smem_size;
        if constexpr (kNumTMAMulticast > 1) {
            int nc = 0;
            if (cudaOccupancyMaxActiveClusters(&nc, kernel, &config) == cudaSuccess) rep.clusters = nc; else (void)cudaGetLastError();
        }
        std::printf("[dgint8_occ] %s: occupancy %d CTAs/SM (target %u)%s; ptxas regs %d (launch budget), local/spill %d B, static smem %zu B, "
                    "dyn smem %d B (limit for %u CTAs/SM: %d B), grid %u CTAs x %u threads, cluster %u, math threads %u, setmaxnreg math %u / tma 40\n",
                    name, nb, kMinBlocksPerSM, (kNumTMAMulticast > 1 ? (", max active clusters " + std::to_string(rep.clusters)).c_str() : ""),
                    fa.numRegs, rep.local_bytes, fa.sharedSizeBytes, smem_size, kMinBlocksPerSM, occ_smem_limit(kMinBlocksPerSM),
                    kNumSMs, kNumTMAThreads + kNumMathThreads, kNumTMAMulticast, kNumMathThreads, kMathRegs);
        std::fflush(stdout);
        attr_set = true;
    }
    if (report != nullptr) *report = rep;

    DGINT8_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel,
        const_cast<float*>(args.sfb), static_cast<int*>(nullptr),
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        tma_a, tma_b, tma_d, tma_sfa));
}

}  // namespace dgint8
