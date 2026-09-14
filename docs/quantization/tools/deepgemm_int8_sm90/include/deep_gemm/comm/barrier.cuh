// Copied (trimmed) from DeepGEMM deep_gemm/include/deep_gemm/comm/barrier.cuh (MIT License, Copyright (c) 2025 DeepSeek).
// Only the cluster barrier helper used by the SM90 GEMM kernel is kept (the NVLink/grid-sync helpers pull in MoE layouts).
#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/cluster_sm90.hpp>

namespace deep_gemm::comm {

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

} // namespace deep_gemm::comm
