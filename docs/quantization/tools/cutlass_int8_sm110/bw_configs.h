// Block-scaled configurations. X(idx, TileM, TileN, TileK, ClusterM, ClusterN, KernelSchedule, EpilogueSchedule, GranM, GranN, GranK, desc)
// GranN = 128: DeepSeek-style 128x128 weight blocks (one fp32 per (128 channels, 128 K)); GranN = 1: per output channel per 128 K
// (the "per-col g128" layout, handoff 3.6). GranM = 1: per token per 128 K activations.
#pragma once
#define THOR_BW_CFG_LIST(X)                                                                                             \
  X(0, 128, 128, 128, 1, 1, BwSched1Sm, Epi1Sm, 1, 128, 128, "1SM 128x128x128 c1x1 sf<1,128,128>")                   \
  X(1, 256, 128, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x128x128 c2x1 sf<1,128,128> (example 81)")      \
  X(2, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x256x128 c2x1 sf<1,128,128>")                   \
  X(3, 128, 128, 128, 1, 1, BwSched1Sm, Epi1Sm, 1, 1, 128, "1SM 128x128x128 c1x1 sf<1,1,128> per-col W")             \
  X(4, 256, 128, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x128x128 c2x1 sf<1,1,128> per-col W")             \
  X(5, 128, 256, 128, 1, 1, BwSched1Sm, Epi1Sm, 1, 128, 128, "1SM 128x256x128 c1x1 sf<1,128,128>")                   \
  X(6, 256, 256, 128, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x256x128 c2x1 sf<1,1,128> per-col W")                \
  X(7, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 1, 128, "2SM 256x128x256 c2x1 sf<1,1,128> per-col W (TileK 256, GB200 opt3)") \
  X(8, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm, 1, 128, 128, "2SM 256x128x256 c2x1 sf<1,128,128> (TileK 256)")               \
  X(9, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm16, 1, 1, 128, "2SM 256x128x256 c2x1 sf<1,1,128> per-col, epi tile 128x16") \
  X(10, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm16, 1, 128, 128, "2SM 256x128x256 c2x1 sf<1,128,128>, epi tile 128x16") \
  X(11, 256, 128, 256, 2, 1, BwSched2Sm, Epi2Sm64, 1, 1, 128, "2SM 256x128x256 c2x1 sf<1,1,128> per-col, epi tile 128x64")

#define THOR_NUM_BW_CFGS 12
