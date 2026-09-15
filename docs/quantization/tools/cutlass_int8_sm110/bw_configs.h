// Block-scaled configurations. X(idx, TileM, TileN, TileK, ClusterM, ClusterN, KernelSchedule, EpilogueSchedule, GranM, GranN, GranK, desc)
// GranN = 128: DeepSeek-style 128x128 weight blocks; GranN = 1: per output channel per 128 K ("per-col g128", handoff 3.6). GranM = 1.
// Epilogue tag variants: Epi2Sm16/Epi2Sm64 force a 128x16/128x64 epilogue sub-tile (smaller TMEM fragments in the promotion loop);
// Epi2SmSep = separable weight scale (per-column s_w[n] applied in the epilogue, mainloop sees c[nb,g] or 1).
// All configs run on the 8-warp-promotion shadow kernel (include/cutlass/gemm/kernel/...); 256x256 tiles need it (128 fp32 regs/thread).
#pragma once
#if defined(THOR_BW_DEBUG_ONLY16)
// debug build: a single config (cfg16) so the driver links with only bwcfg_{int8,fp8}_16.o
#define THOR_BW_CFG_LIST(X) X(16, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x256x128 c2x1 sf<1,1,128> per-col, epi tile 128x32")
#define THOR_NUM_BW_CFGS 1
#else
#define THOR_BW_CFG_LIST(X)                                                                                             \
  X(0, 128, 128, 128, 1, 1, BwSched1Sm, Epi1Sm, 1, 128, 128, "1SM 128x128x128 c1x1 sf<1,128,128>")                   \
  X(1, 256, 128, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x128x128 c2x1 sf<1,128,128> (example 81)")      \
  X(2, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x256x128 c2x1 sf<1,128,128>")                   \
  X(3, 128, 128, 128, 1, 1, BwSched1Sm, Epi1Sm, 1, 1, 128, "1SM 128x128x128 c1x1 sf<1,1,128> per-col W")             \
  X(4, 256, 128, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x128x128 c2x1 sf<1,1,128> per-col W")             \
  X(5, 128, 256, 128, 1, 1, BwSched1Sm, Epi1Sm, 1, 128, 128, "1SM 128x256x128 c1x1 sf<1,128,128>")                   \
  X(6, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x256x128 c2x1 sf<1,1,128> per-col W")             \
  X(7, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x128x256 c2x1 sf<1,1,128> per-col W (TileK 256)")  \
  X(8, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x128x256 c2x1 sf<1,128,128> W-block (TileK 256)") \
  X(9, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm16, 1, 1, 128, "2SM 256x128x256 c2x1 sf<1,1,128> per-col, epi tile 128x16") \
  X(10, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm16, 1, 128, 128, "2SM 256x128x256 c2x1 sf<1,128,128>, epi tile 128x16") \
  X(11, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm64, 1, 1, 128, "2SM 256x128x256 c2x1 sf<1,1,128> per-col, epi tile 128x64") \
  X(12, 256, 128, 256, 2, 1, BwSched2Sm, Epi2SmSep, 1, 128, 128, "2SM 256x128x256 c2x1 SEPARABLE: sf<1,128,128> mainloop (c[g] folded into A scales) + per-col s_w[n] in the epilogue") \
  X(13, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x256x128 c2x1 sf<1,128,128> W-block (8-warp promotion)")   \
  X(14, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm16, 1, 128, 128, "2SM 256x256x128 c2x1 sf<1,128,128> W-block, epi tile 128x16")  \
  X(15, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm16, 1, 1, 128, "2SM 256x256x128 c2x1 sf<1,1,128> per-col, epi tile 128x16")      \
  X(16, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x256x128 c2x1 sf<1,1,128> per-col, epi tile 128x32")            \
  X(17, 256, 256, 128, 2, 1, BwSched2Sm, Epi2SmSep, 1, 128, 128, "2SM 256x256x128 c2x1 SEPARABLE s_w[n]*c[nb,g] (W-block mainloop + per-col epilogue)") \
  X(18, 256, 256, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x256x256 c2x1 sf<1,1,128> per-col, epi 128x32 (TileK 256)")   \
  X(19, 256, 256, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x256x256 c2x1 sf<1,128,128> W-block (TileK 256)")   \
  X(20, 256, 256, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 256, "2SM 256x256x256 c2x1 sf<1,1,256> per-col g256 (TileK 256; promotion every 256 K)")   \
  X(21, 256, 256, 256, 2, 1, BwSched2Sm, Epi2Sm16, 1, 1, 256, "2SM 256x256x256 c2x1 sf<1,1,256> per-col g256, epi tile 128x16")

#define THOR_NUM_BW_CFGS 22
#endif
