// Copied from DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek); INT8 port adds an int32 fence.
#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::ptx {

CUTLASS_DEVICE void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

CUTLASS_DEVICE void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// (INT8 port) fence for int32 WGMMA accumulators
CUTLASS_DEVICE void warpgroup_fence_operand(int32_t& reg) {
    asm volatile("" : "+r"(reg) :: "memory");
}

template <int N>
CUTLASS_DEVICE void warpgroup_wait() {
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

} // namespace deep_gemm::ptx
