"""Generate include/deep_gemm/impls/sm90_int8_gemm_1d2d.cuh: DeepGEMM's SM90 FP8 1D2D kernel ported to INT8 (s8 x s8 -> s32 WGMMA,
fp32 software promotion with the (1,128,128) block scales). Every edit is a checked string replacement of the pristine upstream
source so the diff to DeepGEMM stays auditable. Re-run after editing this file."""
import os
B = '/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/int8_bench'
SRC = f'{B}/third_party/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh'
import sys
DST = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include/deep_gemm/impls/sm90_int8_gemm_1d2d.cuh')
s = open(SRC).read()

def rep(old, new, n):
    global s
    c = s.count(old); assert c == n, (old, c, n)
    s = s.replace(old, new)

rep('sm90_fp8_gemm_1d2d_impl', 'sm90_int8_gemm_1d2d_impl', 1)
rep('mma::sm90::FP8MMASelector<BLOCK_N>', 'mma::sm90::S8MMASelector<BLOCK_N>', 1)
rep('"Only support per-128-channel FP8 scaling"', '"Only support per-128-channel INT8 scaling (BLOCK_K == scale granularity)"', 1)
rep('__nv_fp8_e4m3', 'int8_t', 7)   # sizeof (A, B, WGMMA A padding check), 2 smem reinterpret_casts, 2 tma::copy dtype args
rep('float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};',
    '// (INT8 port) per-k-block accumulator is int32 (zeroed by the WGMMA scale-d predicate at k == 0), promoted into fp32 `final_accum`\n'
    '            int32_t accum[WGMMA::kNumAccum];\n'
    '            float final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};', 1)
for j in range(4):
    rep(f'* accum[i * 4 + {j}];', f'* promote_i32<kPromoteMode>(accum[i * 4 + {j}]);', 1)
rep('            auto empty_barrier_arrive = [&]() {\n                if constexpr (kNumTMAMulticast == 1) {\n                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();\n                } else {\n                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();\n                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();\n                }\n            };', '            auto empty_barrier_arrive_at = [&](const uint32_t& s) {\n                if constexpr (kNumTMAMulticast == 1) {\n                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();\n                } else {\n                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();\n                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();\n                }\n            };\n            auto empty_barrier_arrive = [&]() { empty_barrier_arrive_at(stage_idx); };', 1)

# 2) overlapped-promotion mainloop (DG_INT8_OVERLAP=1, BLOCK_M in {64, 128}: one 64-row wave per math warpgroup):
#    double-buffered int32 accumulators, the promotion of k-block i runs while the WGMMAs of k-block i+1 execute.
old_loop_start = """                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {"""
assert s.count(old_loop_start) == 1
overlap = r"""                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                  if constexpr (kOverlap and BLOCK_M / WAVE_BLOCK_M == 1 and BLOCK_M >= WGMMA::M) {
                    // (INT8 port) overlapped promotion: WGMMA(k+1) is in flight while promote(k) runs on the CUDA cores
                    int32_t accum2[2][WGMMA::kNumAccum];
                    float ps_0_0 = 0, ps_1_0 = 0, ps_0_1 = 0, ps_1_1 = 0;   // scales of the deferred (previous) k-block
                    uint32_t prev_stage_idx = 0;
                    auto issue = [&](int32_t* acc) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
                        #pragma unroll
                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                            ptx::warpgroup_fence_operand(acc[i]);
                        ptx::warpgroup_arrive();
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            a_desc.reg32_[0] = a_desc_base_lo + (k * WGMMA::K) / 16;
                            b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                            WGMMA::wgmma(a_desc, b_desc, acc, k);
                        }
                        ptx::warpgroup_commit_batch();
                        #pragma unroll
                        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                            ptx::warpgroup_fence_operand(acc[i]);
                    };
                    auto promote = [&](const int32_t* acc) {
                        #pragma unroll
                        for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            const bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                            final_accum[i * 4 + 0] += (predicate ? ps_0_0 : ps_0_1) * promote_i32<kPromoteMode>(acc[i * 4 + 0]);
                            final_accum[i * 4 + 1] += (predicate ? ps_0_0 : ps_0_1) * promote_i32<kPromoteMode>(acc[i * 4 + 1]);
                            final_accum[i * 4 + 2] += (predicate ? ps_1_0 : ps_1_1) * promote_i32<kPromoteMode>(acc[i * 4 + 2]);
                            final_accum[i * 4 + 3] += (predicate ? ps_1_0 : ps_1_1) * promote_i32<kPromoteMode>(acc[i * 4 + 3]);
                        }
                    };
                    // Load this k-block's scales (all smem reads before `warpgroup_arrive`) and remember them for the deferred promotion
                    auto load_scales = [&](const uint32_t& k_block_idx) {
                        const float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx);
                        float scale_b_1 = 0;
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);
                        full_barriers[stage_idx]->wait(phase);
                        const float scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + r_0);
                        const float scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + r_1);
                        ps_0_0 = scale_a_0 * scale_b_0, ps_1_0 = scale_a_1 * scale_b_0;
                        if constexpr (not kMustUseUniformedScaleB)
                            ps_0_1 = scale_a_0 * scale_b_1, ps_1_1 = scale_a_1 * scale_b_1;
                    };
                    float cs_0_0, cs_1_0, cs_0_1, cs_1_1;   // current k-block scales (become `ps_*` once the previous block is promoted)
                    uint32_t k_block_idx = 0;
                    // Prologue: k-block 0 into buffer 0
                    load_scales(k_block_idx);
                    cs_0_0 = ps_0_0, cs_1_0 = ps_1_0, cs_0_1 = ps_0_1, cs_1_1 = ps_1_1;
                    issue(accum2[0]);
                    prev_stage_idx = stage_idx;
                    advance_pipeline(k_block_idx);
                    uint32_t last_buf = 0;
                    while (k_block_idx < num_total_k_blocks) {
                        // k-block k into buffer 1, promote buffer 0 (k-1) while it runs
                        load_scales(k_block_idx);
                        {   const float n00 = ps_0_0, n10 = ps_1_0, n01 = ps_0_1, n11 = ps_1_1;
                            ps_0_0 = cs_0_0, ps_1_0 = cs_1_0, ps_0_1 = cs_0_1, ps_1_1 = cs_1_1;
                            cs_0_0 = n00, cs_1_0 = n10, cs_0_1 = n01, cs_1_1 = n11; }
                        issue(accum2[1]);
                        ptx::warpgroup_wait<1>();
                        empty_barrier_arrive_at(prev_stage_idx);
                        promote(accum2[0]);
                        prev_stage_idx = stage_idx;
                        advance_pipeline(k_block_idx);
                        last_buf = 1;
                        if (k_block_idx >= num_total_k_blocks)
                            break;
                        // k-block k+1 into buffer 0, promote buffer 1 (k) while it runs
                        load_scales(k_block_idx);
                        {   const float n00 = ps_0_0, n10 = ps_1_0, n01 = ps_0_1, n11 = ps_1_1;
                            ps_0_0 = cs_0_0, ps_1_0 = cs_1_0, ps_0_1 = cs_0_1, ps_1_1 = cs_1_1;
                            cs_0_0 = n00, cs_1_0 = n10, cs_0_1 = n01, cs_1_1 = n11; }
                        issue(accum2[0]);
                        ptx::warpgroup_wait<1>();
                        empty_barrier_arrive_at(prev_stage_idx);
                        promote(accum2[1]);
                        prev_stage_idx = stage_idx;
                        advance_pipeline(k_block_idx);
                        last_buf = 0;
                    }
                    // Epilogue: drain the last k-block
                    ptx::warpgroup_wait<0>();
                    empty_barrier_arrive_at(prev_stage_idx);
                    ps_0_0 = cs_0_0, ps_1_0 = cs_1_0, ps_0_1 = cs_0_1, ps_1_1 = cs_1_1;
                    if (last_buf == 1) promote(accum2[1]); else promote(accum2[0]);
                  } else {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {"""
s = s.replace(old_loop_start, overlap)
old_loop_end = """                        }
                    }
                });
            } else {"""
assert s.count(old_loop_end) == 1
s = s.replace(old_loop_end, """                        }
                    }
                  }
                });
            } else {""")
# promotion helper + mode macro, inserted before the kernel template
helper = r'''
// (INT8 port) exact int32 -> fp32 promotion of a 128-deep int8 dot product: |x| <= 128 * 127 * 127 = 2064512 < 2^22.
//   mode 0: cvt.rn.f32.s32 (I2F/I2FP SASS)
//   mode 1: magic-number add (IADD on the integer pipe + FADD): float_bits(x + 1.5 * 2^23) - 1.5 * 2^23 is exact for |x| < 2^22
//   mode 2: EXPERIMENT ONLY: bit-reinterpret (no conversion, wrong results) to measure the cost of the conversion itself
#ifndef DG_INT8_PROMOTE_MODE
#define DG_INT8_PROMOTE_MODE 0
#endif
static constexpr uint32_t kPromoteMode = DG_INT8_PROMOTE_MODE;
#ifndef DG_INT8_OVERLAP
#define DG_INT8_OVERLAP 0
#endif
static constexpr bool kOverlap = DG_INT8_OVERLAP != 0;   // overlapped promotion mainloop (BLOCK_M in {64, 128} only)

template <uint32_t kMode>
CUTLASS_DEVICE float promote_i32(const int32_t& x) {
    if constexpr (kMode == 0) {
        return static_cast<float>(x);
    } else if constexpr (kMode == 1) {
        return __int_as_float(x + 0x4B400000) - 12582912.0f;
    } else {
        return __int_as_float(x);   // mode 2: EXPERIMENT ONLY (wrong numerics) -- FP8-equivalent instruction count, upper bound for the promotion cost
    }
}

'''
marker = 'template <cute::UMMA::Major kMajorSFB,'
assert s.count(marker) == 1
s = s.replace(marker, helper + marker)
hdr = ('// INT8 port of DeepGEMM deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh (MIT License, Copyright (c) 2025 DeepSeek).\n'
       '// Generated by kernels/deepgemm_int8/gen_kernel.py -- do not edit by hand. Changes vs upstream: A/B element type int8_t,\n'
       '// S8MMASelector (wgmma m64nNk32.s32.s8.s8), int32 per-block accumulator, fp32 promotion `final += scale * float(accum)`.\n'
       '// TMA pipeline, scheduler, multicast, epilogue (STSM + TMA store) are unchanged.\n')
open(DST, 'w').write(hdr + s)
print('wrote', DST)
