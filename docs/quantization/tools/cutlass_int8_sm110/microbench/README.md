# Thor (sm_110a) instruction-issue microbenchmarks behind the g128 promotion analysis

Build/run (from this directory): `nvcc -O3 -std=c++17 -arch=sm_110a -I.. -o issue_rate issue_rate.cu && ./issue_rate`, same for `promo_patterns.cu`.
Measured 2026-09-14 at 1386 MHz (120 W mode), 80 blocks x 256 threads, 32 independent chains per thread.

## issue_rate: per-SM throughput of the promotion instructions and of the register-accumulator tensor path

| instruction | per clk per SM | note |
|---|---:|---|
| FFMA / FMUL / FADD | 125 (= 128, 1 warp-instr per SMSP per clk) | full rate |
| I2F (`cvt.rn.f32.s32`, SASS `I2FP.F32.S32`) | 64 | **half rate** -- the int32 -> fp32 conversion of the MMA partial sums |
| F2I (`cvt.rzi.s32.f32`) | 16 | quarter rate (not used in the GEMM; only in the quantizer) |
| FFMA2 (`fma.rn.f32x2`) | 60 pairs | split into two FFMA by ptxas: no packed FP32 on Thor |
| `mma.sync.m16n8k32 s8` (register accumulator) | 2034 MAC/clk/SM = 112.7 TOPS | **1/4 of tcgen05** (8192 MAC/clk/SM, ~454 TOPS) |
| `mma.sync.m16n8k16 bf16` | 1017 MAC/clk/SM = 56.4 TOPS | |

(The IADD row of the raw output is not meaningful: ptxas fuses the chained adds into IADD3.)

## promo_patterns: issue clocks per promoted element (per lane), register-only

| pattern | instructions per element per K group | clk / element |
|---|---|---:|
| per-col g128 today | I2F + FMUL(s_a*s_w) + FFMA | 3.27 |
| W-block g128 today | I2F + FFMA | 2.04 |
| per-col with TMEM-pre-biased accumulator | FFMA(fb, s_a, -M*s_a) + FFMA(t, s_w, acc) | **2.06** |
| per-col with the bias added by IADD in registers | IADD + 2 FFMA | 4.06 (worse) |
| W-block with TMEM-pre-biased accumulator | FFMA + FADD | 2.05 (no gain) |

Bias trick: pre-store 0x4B400000 (bit pattern of 1.5 * 2^23) in the int32 TMEM accumulator before each K group and accumulate onto it; the
tcgen05.ld result is then already the fp32 value 1.5*2^23 + x for every |x| <= 128*127*127 (g128; exhaustively checked, 0 mismatches;
g256 also fits: 4,129,024 < 2^22), so the I2F disappears and t = fma(fb, s_a, -M*s_a) is bit-identical to float(x)*s_a when M*s_a is
exact (clear the 2 low mantissa bits of s_a). Cost: one tcgen05.st of the bias per stage per K group, first MMA of the group in
accumulate mode, initial fill. Not implemented yet.
