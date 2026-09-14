"""Generate the occupancy-2 ("OCC", approach D) variants of the INT8 1D2D kernel into this directory (nothing outside occ/ is touched):
  occ/sm90_int8_gemm_1d2d_occ.cuh    kernel `sm90_int8_gemm_1d2d_occ_impl`   (from ../include/deep_gemm/impls/sm90_int8_gemm_1d2d.cuh)
  occ/sm90_int8_gemm_1d2d_occdb.cuh  kernel `sm90_int8_gemm_1d2d_occdb_impl` (from ../include/deep_gemm/impls/sm90_int8_gemm_1d2d_db.cuh)

Idea: ONE math warpgroup per CTA (128 math + 128 TMA threads) and TWO (or three) co-resident CTAs per SM, so the two warpgroups that
drive the SM's tensor cores concurrently belong to different CTAs on different tiles: their k-loops drift out of phase and one CTA's
int32->fp32 promotion (64 I2F + 64 FFMA per thread per k-block) overlaps the other CTA's WGMMAs without any barrier engineering.
Kernel edits (checked string replacements, everything else byte-identical to the source kernels):
  * two extra template parameters `kMinBlocksPerSM, kMathRegs`;
  * `__launch_bounds__(kNumTMAThreads + kNumMathThreads, kMinBlocksPerSM)` (ptxas launch budget 65536 / (256 * kMinBlocksPerSM) regs);
  * `kNumMathRegisters = kMathRegs` (setmaxnreg.inc of the math warpgroup; the producer keeps setmaxnreg.dec 40), e.g. 2 CTAs/SM:
    128 x 40 + 128 x 216 = 32768 regs = half the register file; 3 CTAs/SM: launch 80 -> 128 x 40 + 128 x 120 = 20480;
  * `WAVE_BLOCK_M = WGMMA::M * (kNumMathThreads / 128)` so BLOCK_M = 128 with a single math warpgroup runs two sequential 64-row waves
    (the source kernels always pair BLOCK_M = 128 with two warpgroups);
  * the DG_INT8_OVERLAP experiment path is dropped (as in gen_kernel_pp.py); helpers come from the default header.
The persistent grid is kNumSMs = 132 * kMinBlocksPerSM CTAs (scheduler stride = grid), see launcher_occ.cuh.
Per-element math and k order are unchanged -> bit-identical to the default kernel."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)

EPILOGUE = r"""            // (OCC) D tile written and TMA-stored in kNumDPasses passes through a buffer of kAtomsPerPass swizzle atoms (1 pass = the default epilogue)
            constexpr uint32_t kNumDAtoms = BLOCK_N / TMA_D_BLOCK_N;
            constexpr uint32_t kAtomsPerPass = kHalfD ? 1 : kNumDAtoms;
            constexpr uint32_t kNumDPasses = kNumDAtoms / kAtomsPerPass;
            DG_STATIC_ASSERT(kNumDAtoms % kAtomsPerPass == 0, "Invalid D passes");
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= kAtomsPerPass, "Too many TMA blocks");
            #pragma unroll
            for (uint32_t pass = 0; pass < kNumDPasses; ++ pass) {
                // Wait for the previous TMA store (previous pass, or the previous tile) to finish reading the buffer
                if (threadIdx.x < kAtomsPerPass)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

                // Write back to shared memory using STSM
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    auto m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t i = pass * kAtomsPerPass * (TMA_D_BLOCK_N / 8); i < (pass + 1) * kAtomsPerPass * (TMA_D_BLOCK_N / 8); ++ i) {
                        // Swizzle or padding into the correct address
                        uint8_t* smem_ptr = nullptr;
                        if constexpr (kSwizzleDMode > 0) {
                            constexpr uint32_t kNumBankGroupBytes = 16;
                            auto atom_offset = i / (TMA_D_BLOCK_N / 8) - pass * kAtomsPerPass, in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                            auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                            constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                            col ^= row % (kSwizzleDMode / 16);
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                                warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                                m_offset * kSwizzleDMode +                                 // Wave offset
                                atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset within the buffer
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                        } else {
                            smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        }
                        ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

                // Use TMA store to write back to global memory
                if (threadIdx.x < kAtomsPerPass) {
                    auto in_block_n_offset = (pass * kAtomsPerPass + threadIdx.x) * TMA_D_BLOCK_N;
                    auto smem_ptr = smem_d + threadIdx.x * TMA_D_BLOCK_N * BLOCK_M;
                    auto n_idx = epilogue_op_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                    auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                    if constexpr (kGemmType == GemmType::Batched) {
                        cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                      n_idx, m_idx, scheduler.current_group_idx);
                    } else {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
            }
            __syncwarp();
        }
    }
"""

def gen(src, dst, old_name, new_name, title):
    s = open(src).read()
    def rep(old, new, n):
        nonlocal s
        c = s.count(old); assert c == n, (src, old[:100], c, n)
        s = s.replace(old, new)
    hdr_end = s.index('#pragma once')
    s = (f'// {title}\n'
         f'// Generated by kernels/deepgemm_int8/occ/gen_kernel_occ.py from {os.path.relpath(src, PKG)} -- do not edit by hand.\n'
         '// Approach D: one math warpgroup per CTA, kMinBlocksPerSM co-resident CTAs per SM (launch bounds + lowered setmaxnreg), so the\n'
         '// promotion of one CTA overlaps the WGMMAs of the other. Same math, same k order -> bit-identical to the default kernel.\n') + s[hdr_end:]
    rep(f'{old_name}(float* sfb,', f'{new_name}(float* sfb,', 1)
    rep('          typename epilogue_op_t>\nCUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void',
        '          typename epilogue_op_t,\n          uint32_t kMinBlocksPerSM, uint32_t kMathRegs>\n'
        'CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, kMinBlocksPerSM) void', 1)
    rep('    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 232;',
        '    constexpr uint32_t kNumMathRegisters = kMathRegs;   // (OCC) lowered so kMinBlocksPerSM CTAs fit the register file', 1)
    rep('            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;',
        '            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * (kNumMathThreads / 128);   // (OCC) waves per warpgroup', 1)
    # (OCC kHalfD) optional half-D epilogue: the bf16 D tile goes through a smem buffer of ONE swizzle atom (BLOCK_M x 64 columns = 8 KB for
    # BLOCK_M = 64 instead of the full 16 KB tile), written and TMA-stored in BLOCK_N / TMA_D_BLOCK_N passes (thread 0 waits for the previous
    # pass's store to finish reading the buffer). Frees 8 KB -> 64x128 fits 4 stages under the 2-CTA smem limit. Values are unchanged.
    rep('          uint32_t kMinBlocksPerSM, uint32_t kMathRegs>\n', '          uint32_t kMinBlocksPerSM, uint32_t kMathRegs, bool kHalfD>\n', 1)
    rep('    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);',
        '    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * (kHalfD ? kSwizzleDMode / 2 : BLOCK_N) * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);   // (OCC kHalfD) one swizzle atom\n'
        '    DG_STATIC_ASSERT(not kHalfD or kSwizzleDMode > 0, "kHalfD needs a swizzled D store");', 1)
    e0 = s.index('            // Wait last TMA store to be finished\n'); e1 = s.index('#else\n    if (blockIdx.x == 0 and threadIdx.x == 0)')
    assert e0 < e1 and s.count('            // Wait last TMA store to be finished\n') == 1
    s = s[:e0] + EPILOGUE + s[e1:]
    if 'template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>' in s:
        # default header: drop the shared helpers (they come from the default header) and the DG_INT8_OVERLAP experiment path
        i0 = s.index('template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>\nCUTLASS_DEVICE void dispatch_num_former_iters(')
        i1 = s.index('template <cute::UMMA::Major kMajorSFB,')
        assert i0 < i1
        s = s[:i0] + '// helpers (dispatch_num_former_iters, kPromoteMode / promote_i32, kOverlap) are shared with the default kernel\n\n' + s[i1:]
        rep('#include <deep_gemm/scheduler/gemm.cuh>\n', '#include <deep_gemm/scheduler/gemm.cuh>\n#include <deep_gemm/impls/sm90_int8_gemm_1d2d.cuh>\n', 1)
        start = '                  if constexpr (kOverlap and BLOCK_M / WAVE_BLOCK_M == 1 and BLOCK_M >= WGMMA::M) {'
        end = '                  } else {\n                    #pragma unroll 8\n'
        j0 = s.index(start); j1 = s.index(end); assert j0 < j1 and s.count(start) == 1 and s.count(end) == 1
        s = s[:j0] + '                  {\n                    #pragma unroll 8\n' + s[j1 + len(end):]
    else:
        assert '#include <deep_gemm/impls/sm90_int8_gemm_1d2d.cuh>' in s, "DB header must include the default header for the helpers"
    assert s.count(new_name) == 1 and old_name + '(' not in s
    open(dst, 'w').write(s)
    print('wrote', dst)

gen(os.path.join(PKG, 'include/deep_gemm/impls/sm90_int8_gemm_1d2d.cuh'), os.path.join(HERE, 'sm90_int8_gemm_1d2d_occ.cuh'),
    'sm90_int8_gemm_1d2d_impl', 'sm90_int8_gemm_1d2d_occ_impl',
    'Occupancy-2 (OCC) variant of the INT8 port of DeepGEMM sm90_fp8_gemm_1d2d.cuh (MIT License, Copyright (c) 2025 DeepSeek).')
gen(os.path.join(PKG, 'include/deep_gemm/impls/sm90_int8_gemm_1d2d_db.cuh'), os.path.join(HERE, 'sm90_int8_gemm_1d2d_occdb.cuh'),
    'sm90_int8_gemm_1d2d_db_impl', 'sm90_int8_gemm_1d2d_occdb_impl',
    'Occupancy-2 (OCC) + double-buffered-accumulator (DB) variant of the INT8 port of DeepGEMM sm90_fp8_gemm_1d2d.cuh (MIT License, Copyright (c) 2025 DeepSeek).')
