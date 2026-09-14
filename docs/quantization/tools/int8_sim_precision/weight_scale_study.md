# Cosmos3-Nano gen-tower weight scale layout study (INT8 symmetric, qmax 127)

Checkpoint: `/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/hf_cache/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa/transformer`  
Weights: 252 gen-tower linears (36 layers x q,k,v,o,gate,up,down); diffusers names `self_attn.add_{q,k,v}_proj`, `self_attn.to_add_out` (= `{q,k,v,o}_proj_moe_gen`, q/k/v are NOT fused), `mlp_moe_gen.{gate,up,down}_proj`. Shapes (N x K): q/o 4096x4096, k/v 1024x4096, gate/up 12288x4096, down 4096x12288. Total 6.946 B params.  
Math: scale = absmax/127 (fp32), q = torch.round(W/scale) (half-even), clamp +-127, deq = q*scale. Groups along K. GEMM proxy: X ~ N(0,1) [2048,K] bf16-representable, fp32 matmul (TF32 off), rel-L2 of X@Wdeq^T vs X@W^T; same X for every scheme of a given K.  
Aggregation: SNR/rel-L2/GEMM are *pooled* (energy-weighted: sum of squared errors over sum of squared signal across the weights in the group); `mean` columns are plain means of per-weight numbers. clip frac = clipped elements / all elements.

## Overall (all 252 weights)

| scheme | SNR dB (pooled) | SNR dB (mean) | SNR dB (min weight) | rel-L2 | GEMM rel-L2 | max err / max|W| (worst) | clip frac |
|---|---|---|---|---|---|---|---|
| (a) per-col g128 [S0] | 42.54 | 42.64 | 35.38 | 0.0075 | 0.0075 | 0.0039 | 0.00e+00 |
| (b) W 128x128 blockwise | 35.40 | 35.62 | 23.25 | 0.0170 | 0.0170 | 0.0039 | 0.00e+00 |
| (c) separable s_w[n]*c[kb], max-ratio (no clip) | 37.89 | 38.60 | 27.77 | 0.0127 | 0.0128 | 0.0039 | 0.00e+00 |
| (d0) separable log rank-1 fit, unscaled (gamma mean 1.000, range 1.000-1.000) | 23.73 | 24.89 | 11.48 | 0.0651 | 0.0649 | 0.9340 | 5.31e-03 |
| (d1) separable log fit, rescaled clip<=0.01% (gamma mean 1.785, range 1.408-4.314) | 33.47 | 34.02 | 26.37 | 0.0212 | 0.0212 | 0.8884 | 9.72e-05 |
| (d2) separable log fit, rescaled clip<=0.1% (gamma mean 1.273, range 1.181-2.133) | 29.52 | 29.71 | 17.76 | 0.0334 | 0.0334 | 0.9186 | 9.68e-04 |
| (d3) separable log fit, rescaled exact no-clip (gamma mean 6.028, range 1.875-27.310) | 25.44 | 28.43 | 16.21 | 0.0535 | 0.0535 | 0.1613 | 0.00e+00 |
| (h) separable: c[kb] from log fit, s_w[n] per-row no-clip | 38.59 | 38.76 | 30.12 | 0.0118 | 0.0118 | 0.0260 | 0.00e+00 |
| (e) per-col g64 [reference] | 43.60 | 43.62 | 37.64 | 0.0066 | 0.0066 | 0.0039 | 0.00e+00 |
| (f) per-channel | 37.89 | 38.60 | 27.77 | 0.0128 | 0.0128 | 0.0039 | 0.00e+00 |
| (g) per-tensor | 23.18 | 25.32 | 13.10 | 0.0694 | 0.0694 | 0.0039 | 0.00e+00 |
| (extra) separable no-clip, g64 | 37.89 | 38.61 | 27.77 | 0.0127 | 0.0127 | 0.0039 | 0.00e+00 |
| (extra) W 64x64 blockwise | 37.53 | 37.52 | 26.38 | 0.0133 | 0.0133 | 0.0039 | 0.00e+00 |

## Per layer type: weight SNR dB (pooled)

| scheme | q | k | v | o | gate | up | down | all |
|---|---|---|---|---|---|---|---|---|
| (a) per-col g128 [S0] | 42.96 | 42.47 | 42.75 | 43.71 | 41.57 | 42.97 | 42.80 | 42.54 |
| (b) W 128x128 blockwise | 35.70 | 34.08 | 37.96 | 37.65 | 34.94 | 35.33 | 35.17 | 35.40 |
| (c) separable s_w[n]*c[kb], max-ratio (no clip) | 39.41 | 38.20 | 39.28 | 39.51 | 36.04 | 39.63 | 37.93 | 37.89 |
| (d0) separable log rank-1 fit, unscaled | 26.12 | 23.56 | 25.80 | 22.81 | 21.18 | 26.51 | 25.32 | 23.73 |
| (d1) separable log fit, rescaled clip<=0.01% | 35.38 | 32.39 | 36.50 | 32.60 | 33.38 | 34.52 | 32.51 | 33.47 |
| (d2) separable log fit, rescaled clip<=0.1% | 30.73 | 27.27 | 31.40 | 28.43 | 28.81 | 30.85 | 29.37 | 29.52 |
| (d3) separable log fit, rescaled exact no-clip | 30.85 | 30.24 | 32.57 | 26.44 | 25.92 | 27.62 | 22.67 | 25.44 |
| (h) separable: c[kb] from log fit, s_w[n] per-row no-clip | 39.45 | 38.29 | 39.30 | 39.97 | 37.79 | 39.63 | 37.96 | 38.59 |
| (e) per-col g64 [reference] | 43.89 | 43.54 | 43.70 | 44.47 | 42.90 | 43.88 | 43.76 | 43.60 |
| (f) per-channel | 39.41 | 38.20 | 39.27 | 39.48 | 36.03 | 39.63 | 37.93 | 37.89 |
| (g) per-tensor | 25.30 | 25.92 | 30.37 | 21.83 | 22.95 | 23.71 | 22.58 | 23.18 |
| (extra) separable no-clip, g64 | 39.41 | 38.20 | 39.30 | 39.53 | 36.04 | 39.64 | 37.93 | 37.89 |
| (extra) W 64x64 blockwise | 37.52 | 36.35 | 39.23 | 39.20 | 37.01 | 37.60 | 37.49 | 37.53 |

## Per layer type: GEMM-proxy rel-L2 (pooled)

| scheme | q | k | v | o | gate | up | down | all |
|---|---|---|---|---|---|---|---|---|
| (a) per-col g128 [S0] | 0.0071 | 0.0075 | 0.0073 | 0.0065 | 0.0083 | 0.0071 | 0.0072 | 0.0075 |
| (b) W 128x128 blockwise | 0.0164 | 0.0198 | 0.0126 | 0.0131 | 0.0179 | 0.0171 | 0.0174 | 0.0170 |
| (c) separable s_w[n]*c[kb], max-ratio (no clip) | 0.0107 | 0.0123 | 0.0109 | 0.0106 | 0.0158 | 0.0104 | 0.0127 | 0.0128 |
| (d0) separable log rank-1 fit, unscaled | 0.0495 | 0.0664 | 0.0513 | 0.0723 | 0.0869 | 0.0473 | 0.0542 | 0.0649 |
| (d1) separable log fit, rescaled clip<=0.01% | 0.0170 | 0.0240 | 0.0150 | 0.0234 | 0.0214 | 0.0188 | 0.0237 | 0.0212 |
| (d2) separable log fit, rescaled clip<=0.1% | 0.0291 | 0.0433 | 0.0269 | 0.0379 | 0.0363 | 0.0287 | 0.0340 | 0.0334 |
| (d3) separable log fit, rescaled exact no-clip | 0.0287 | 0.0308 | 0.0235 | 0.0477 | 0.0508 | 0.0416 | 0.0736 | 0.0535 |
| (h) separable: c[kb] from log fit, s_w[n] per-row no-clip | 0.0106 | 0.0122 | 0.0108 | 0.0100 | 0.0129 | 0.0104 | 0.0126 | 0.0118 |
| (e) per-col g64 [reference] | 0.0064 | 0.0067 | 0.0065 | 0.0060 | 0.0072 | 0.0064 | 0.0065 | 0.0066 |
| (f) per-channel | 0.0107 | 0.0123 | 0.0109 | 0.0106 | 0.0158 | 0.0104 | 0.0127 | 0.0128 |
| (g) per-tensor | 0.0543 | 0.0506 | 0.0303 | 0.0810 | 0.0712 | 0.0652 | 0.0744 | 0.0694 |
| (extra) separable no-clip, g64 | 0.0107 | 0.0123 | 0.0108 | 0.0106 | 0.0158 | 0.0104 | 0.0127 | 0.0127 |
| (extra) W 64x64 blockwise | 0.0133 | 0.0152 | 0.0109 | 0.0110 | 0.0141 | 0.0132 | 0.0133 | 0.0133 |

## Per layer type: relative L2 error (pooled)

| scheme | q | k | v | o | gate | up | down | all |
|---|---|---|---|---|---|---|---|---|
| (a) per-col g128 [S0] | 0.0071 | 0.0075 | 0.0073 | 0.0065 | 0.0083 | 0.0071 | 0.0072 | 0.0075 |
| (b) W 128x128 blockwise | 0.0164 | 0.0198 | 0.0126 | 0.0131 | 0.0179 | 0.0171 | 0.0174 | 0.0170 |
| (c) separable s_w[n]*c[kb], max-ratio (no clip) | 0.0107 | 0.0123 | 0.0109 | 0.0106 | 0.0158 | 0.0104 | 0.0127 | 0.0127 |
| (d0) separable log rank-1 fit, unscaled | 0.0495 | 0.0664 | 0.0513 | 0.0723 | 0.0873 | 0.0473 | 0.0542 | 0.0651 |
| (d1) separable log fit, rescaled clip<=0.01% | 0.0170 | 0.0240 | 0.0150 | 0.0235 | 0.0214 | 0.0188 | 0.0237 | 0.0212 |
| (d2) separable log fit, rescaled clip<=0.1% | 0.0291 | 0.0433 | 0.0269 | 0.0379 | 0.0363 | 0.0287 | 0.0340 | 0.0334 |
| (d3) separable log fit, rescaled exact no-clip | 0.0287 | 0.0308 | 0.0235 | 0.0477 | 0.0506 | 0.0416 | 0.0736 | 0.0535 |
| (h) separable: c[kb] from log fit, s_w[n] per-row no-clip | 0.0106 | 0.0122 | 0.0108 | 0.0100 | 0.0129 | 0.0104 | 0.0126 | 0.0118 |
| (e) per-col g64 [reference] | 0.0064 | 0.0067 | 0.0065 | 0.0060 | 0.0072 | 0.0064 | 0.0065 | 0.0066 |
| (f) per-channel | 0.0107 | 0.0123 | 0.0109 | 0.0106 | 0.0158 | 0.0104 | 0.0127 | 0.0128 |
| (g) per-tensor | 0.0543 | 0.0506 | 0.0303 | 0.0810 | 0.0712 | 0.0652 | 0.0743 | 0.0694 |
| (extra) separable no-clip, g64 | 0.0107 | 0.0123 | 0.0108 | 0.0106 | 0.0158 | 0.0104 | 0.0127 | 0.0127 |
| (extra) W 64x64 blockwise | 0.0133 | 0.0152 | 0.0109 | 0.0110 | 0.0141 | 0.0132 | 0.0133 | 0.0133 |

## Per layer type: max element error / max|W| (worst weight)

| scheme | q | k | v | o | gate | up | down | all |
|---|---|---|---|---|---|---|---|---|
| (a) per-col g128 [S0] | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (b) W 128x128 blockwise | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (c) separable s_w[n]*c[kb], max-ratio (no clip) | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (d0) separable log rank-1 fit, unscaled | 0.8304 | 0.8300 | 0.8403 | 0.8966 | 0.8894 | 0.8644 | 0.9340 | 0.9340 |
| (d1) separable log fit, rescaled clip<=0.01% | 0.7324 | 0.6972 | 0.7385 | 0.7770 | 0.8024 | 0.7909 | 0.8884 | 0.8884 |
| (d2) separable log fit, rescaled clip<=0.1% | 0.7951 | 0.7871 | 0.8020 | 0.8557 | 0.8538 | 0.8358 | 0.9186 | 0.9186 |
| (d3) separable log fit, rescaled exact no-clip | 0.0244 | 0.0394 | 0.0163 | 0.0312 | 0.1613 | 0.0255 | 0.0402 | 0.1613 |
| (h) separable: c[kb] from log fit, s_w[n] per-row no-clip | 0.0058 | 0.0080 | 0.0061 | 0.0107 | 0.0260 | 0.0072 | 0.0049 | 0.0260 |
| (e) per-col g64 [reference] | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (f) per-channel | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (g) per-tensor | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (extra) separable no-clip, g64 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |
| (extra) W 64x64 blockwise | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0039 |

## Per layer type: clip fraction (only d-variants clip; a/b/c/e/f/g are 0 by construction)

| scheme | q | k | v | o | gate | up | down | all |
|---|---|---|---|---|---|---|---|---|
| (d0) separable log rank-1 fit, unscaled | 4.90e-03 | 4.85e-03 | 5.05e-03 | 8.53e-03 | 5.33e-03 | 4.92e-03 | 4.80e-03 | 5.31e-03 |
| (d1) separable log fit, rescaled clip<=0.01% | 9.68e-05 | 9.74e-05 | 9.58e-05 | 9.78e-05 | 9.75e-05 | 9.68e-05 | 9.75e-05 | 9.72e-05 |
| (d2) separable log fit, rescaled clip<=0.1% | 9.66e-04 | 9.71e-04 | 9.64e-04 | 9.76e-04 | 9.69e-04 | 9.64e-04 | 9.69e-04 | 9.68e-04 |
| (d3) separable log fit, rescaled exact no-clip | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

## Per layer type: rescale factor gamma for the (d) log-fit variants (mean over weights; scale = gamma * exp(u+v)/127)

| scheme | q | k | v | o | gate | up | down | all |
|---|---|---|---|---|---|---|---|---|
| (d0) separable log rank-1 fit, unscaled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| (d1) separable log fit, rescaled clip<=0.01% | 1.636 | 1.818 | 1.628 | 2.294 | 1.809 | 1.618 | 1.690 | 1.785 |
| (d2) separable log fit, rescaled clip<=0.1% | 1.230 | 1.260 | 1.243 | 1.449 | 1.262 | 1.222 | 1.245 | 1.273 |
| (d3) separable log fit, rescaled exact no-clip | 4.172 | 4.508 | 3.353 | 7.228 | 6.691 | 6.082 | 10.161 | 6.028 |

## Separability of the absmax matrix A[n,kb] (g=128), additive rank-1 fit in log2 domain

log2 A[n,kb] ~ u[n] + v[kb] (least squares; closed form = row means + col means - grand mean, the fixed point of alternating means). Residual in bits; '> 1 bit' = a block whose absmax is more than 2x off the separable prediction. `row effect std` / `col effect std` = spread of the per-channel / per-K-block log2 factors; `total log2A std` = spread before the fit.

| type | total log2A std (bits) | row effect std | col effect std | resid std (bits, mean) | resid std (max weight) | max |resid| (bits) | frac entries >1 bit | entries >1 bit | channels with any block >1 bit | K-blocks with any channel >1 bit | log2(gamma_noclip) mean (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q | 0.458 | 0.364 | 0.050 | 0.266 | 0.482 | 2.83 | 5.45e-03 | 25712/4718592 | 12853/147456 | 1041/1152 | 2.01 |
| k | 0.514 | 0.402 | 0.067 | 0.302 | 0.606 | 3.84 | 1.05e-02 | 12339/1179648 | 5264/36864 | 1020/1152 | 2.10 |
| v | 0.393 | 0.256 | 0.083 | 0.266 | 0.614 | 3.28 | 7.73e-03 | 9115/1179648 | 3237/36864 | 598/1152 | 1.65 |
| o | 0.593 | 0.449 | 0.214 | 0.289 | 0.419 | 3.94 | 7.46e-03 | 35205/4718592 | 16615/147456 | 990/1152 | 2.74 |
| gate | 0.474 | 0.310 | 0.132 | 0.301 | 0.625 | 4.78 | 1.28e-02 | 181035/14155776 | 67883/442368 | 1147/1152 | 2.64 |
| up | 0.367 | 0.244 | 0.038 | 0.259 | 0.556 | 3.84 | 5.85e-03 | 82833/14155776 | 39745/442368 | 1151/1152 | 2.56 |
| down | 0.441 | 0.328 | 0.036 | 0.286 | 0.488 | 4.05 | 6.43e-03 | 90982/14155776 | 28233/147456 | 3448/3456 | 3.30 |
| all | 0.463 | 0.336 | 0.089 | 0.281 | 0.625 | 4.78 | 8.06e-03 | 437221/54263808 | 173830/1400832 | 9395/10368 | 2.43 |

## Reading

- **(c) as specified degenerates to per-channel.** With N >> K/128 (4096..12288 rows vs 32..96 K-blocks) every K-block contains at least one row whose block absmax is that row's global absmax, so c[kb] = max_n A[n,kb]/s_w[n] = 1 for (almost) every kb: the spread of c[kb] is 0.013 bits on average (max 0.76 bits, exactly 0 for q/k/down). Hence (c) == (f) per-channel to within 0.22 dB on any weight, and identical pooled.
- **(h) separable with the fitted c[kb]** (c[kb]=exp(v[kb]) from the log rank-1 fit, s_w[n] set per row for exact no-clip): 38.59 dB, GEMM proxy 0.0118, scale/absmax 1.65x. This is the best no-clip separable layout, 3.95 dB below (a) and +3.19 dB vs (b); it beats per-channel by only 0.70 dB because the per-K-block (column) effect is tiny (0.089 bits std vs 0.336 bits for the per-channel effect).
- max err / max|W| for the clipped (d0/d1/d2) variants is dominated by the clipped outliers (up to ~0.9 = an outlier quantized to +-127 at a scale that is far too small), not by rounding; for every non-clipping scheme it is 1/(2*127) = 0.0039.
- Bottom line: what per-col g128 buys over per-channel (4.66 dB) is the *non-separable* intra-row variation of absmax along K (residual 0.281 bits std after the rank-1 fit); a separable s_w[n]*c[kb] layout cannot recover it. Separable no-clip layouts land at the per-channel level (~37.9 dB), which is still 2.49 dB better than 128x128 blockwise (b) - i.e. per-channel scale beats 128-row sharing, consistent with handoff 3.6 (per-output-channel scale is the pillar). Clipping-based rank-1 fits (d1/d2) are worse than (b).
- Baselines (pooled weight SNR over all 252 gen-tower weights): per-col g128 (a) 42.54 dB, 128x128 blockwise (b) 35.40 dB, per-col g64 (e) 43.60 dB, per-channel (f) 37.89 dB, per-tensor (g) 23.18 dB. The a-vs-b gap is 7.14 dB in weight SNR (GEMM proxy rel-L2 0.0075 vs 0.0170).
- (c) separable no-clip: 37.89 dB, i.e. 4.65 dB below (a) and +2.49 dB vs (b). GEMM proxy 0.0128. Zero clipping by construction; the price is that scale is on average 1.73x the per-block absmax.
- (d) separable log rank-1 fit: unscaled it clips 5.31e-03 of elements (23.73 dB, clipping dominates); rescaled to clip<=0.01%: 33.47 dB (gamma mean 1.785); clip<=0.1%: 29.52 dB (gamma 1.273); exact no-clip: 25.44 dB (gamma 6.028). Best (d) variant is d1 clip<=0.01% at 33.47 dB, 9.08 dB below (a) and -1.93 dB vs (b).
- Separability: after removing the per-channel and per-K-block log2 factors the residual std of log2(A) is 0.281 bits (mean over weights; max 0.625), from a total spread of 0.463 bits (row effect 0.336, col effect 0.089). 8.06e-03 of (n,kb) blocks deviate by >1 bit (437221 of 54263808); 173830/1400832 channels and 9395/10368 K-blocks have at least one such block. The no-clip rescale of the fit costs log2(gamma) = 2.43 bits on average (max 4.77), which is the resolution lost to the single worst block of each weight.

## Worst weights per scheme (lowest SNR)

| scheme | 3 lowest-SNR weights |
|---|---|
| (a) per-col g128 [S0] | layers.4.mlp_moe_gen.gate_proj.weight 35.38; layers.3.mlp_moe_gen.gate_proj.weight 35.67; layers.0.self_attn.add_v_proj.weight 36.76 |
| (b) W 128x128 blockwise | layers.1.mlp_moe_gen.up_proj.weight 23.25; layers.0.self_attn.add_k_proj.weight 25.71; layers.1.mlp_moe_gen.down_proj.weight 26.01 |
| (c) separable s_w[n]*c[kb], max-ratio (no clip) | layers.4.mlp_moe_gen.gate_proj.weight 27.77; layers.1.mlp_moe_gen.gate_proj.weight 29.61; layers.1.self_attn.add_v_proj.weight 30.20 |
| (d0) separable log rank-1 fit, unscaled | layers.1.self_attn.add_v_proj.weight 11.48; layers.4.mlp_moe_gen.gate_proj.weight 11.71; layers.0.self_attn.add_v_proj.weight 12.43 |
| (d1) separable log fit, rescaled clip<=0.01% | layers.0.self_attn.add_v_proj.weight 26.37; layers.34.self_attn.add_k_proj.weight 26.96; layers.1.self_attn.add_v_proj.weight 27.09 |
| (d2) separable log fit, rescaled clip<=0.1% | layers.1.self_attn.add_v_proj.weight 17.76; layers.0.self_attn.add_v_proj.weight 18.61; layers.0.self_attn.add_k_proj.weight 20.01 |
| (d3) separable log fit, rescaled exact no-clip | layers.1.mlp_moe_gen.gate_proj.weight 16.21; layers.1.mlp_moe_gen.up_proj.weight 16.25; layers.0.self_attn.add_k_proj.weight 16.56 |
| (h) separable: c[kb] from log fit, s_w[n] per-row no-clip | layers.0.self_attn.add_v_proj.weight 30.12; layers.1.self_attn.add_v_proj.weight 30.75; layers.3.mlp_moe_gen.gate_proj.weight 30.77 |
| (e) per-col g64 [reference] | layers.3.mlp_moe_gen.gate_proj.weight 37.64; layers.4.mlp_moe_gen.gate_proj.weight 37.65; layers.0.self_attn.add_v_proj.weight 38.53 |
| (f) per-channel | layers.4.mlp_moe_gen.gate_proj.weight 27.77; layers.1.mlp_moe_gen.gate_proj.weight 29.61; layers.1.self_attn.add_v_proj.weight 30.14 |
| (g) per-tensor | layers.1.mlp_moe_gen.up_proj.weight 13.10; layers.1.mlp_moe_gen.down_proj.weight 13.17; layers.2.mlp_moe_gen.down_proj.weight 14.13 |
| (extra) separable no-clip, g64 | layers.4.mlp_moe_gen.gate_proj.weight 27.77; layers.1.mlp_moe_gen.gate_proj.weight 29.61; layers.1.self_attn.add_v_proj.weight 30.30 |
| (extra) W 64x64 blockwise | layers.1.mlp_moe_gen.up_proj.weight 26.38; layers.0.self_attn.add_k_proj.weight 27.96; layers.2.mlp_moe_gen.up_proj.weight 28.11 |
