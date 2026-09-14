// Tile / cluster / schedule configurations instantiated for BOTH fp8 (e4m3, fp32 acc) and int8 (s32 acc).
// X(idx, TileM, TileN, TileK, ClusterM, ClusterN, KernelSchedule, EpilogueSchedule, description)
// MmaTile is the tcgen05 MMA tile: for 2SM schedules TileM is split across the CTA pair (256 => 128 rows per CTA).
// TileK = 128 elements for 8-bit types = one 128B swizzle atom (4 k32 MMAs per stage), matching the CUTLASS profiler's SM100 int8/fp8 kernels.
// Thor has 20 SMs and (measured) ~1.1 TB/s aggregate L2->SMEM bandwidth, so large tiles + TMA multicast (cluster 2x2 / 4x1 = 4 CTAs,
// 5 clusters resident) matter more than on B200.
#pragma once
#define THOR_CFG_LIST(X)                                                                          \
  X(0, 128, 128, 128, 1, 1, Sched1Sm, Epi1Sm, "1SM 128x128x128 cluster1x1x1")                     \
  X(1, 128, 256, 128, 1, 1, Sched1Sm, Epi1Sm, "1SM 128x256x128 cluster1x1x1")                     \
  X(2, 256, 128, 128, 2, 1, Sched2Sm, Epi2Sm, "2SM 256x128x128 cluster2x1x1")                     \
  X(3, 256, 256, 128, 2, 1, Sched2Sm, Epi2Sm, "2SM 256x256x128 cluster2x1x1")                     \
  X(4, 128, 128, 128, 1, 2, Sched1Sm, Epi1Sm, "1SM 128x128x128 cluster1x2x1 (A multicast)")       \
  X(5, 256, 128, 128, 2, 2, Sched2Sm, Epi2Sm, "2SM 256x128x128 cluster2x2x1 (example-70 style)")   \
  X(6, 128, 128, 64, 1, 1, Sched1Sm, Epi1Sm, "1SM 128x128x64 cluster1x1x1 (K=64 atom)")           \
  X(7, 64, 128, 128, 1, 1, Sched1Sm, Epi1Sm, "1SM 64x128x128 cluster1x1x1 (small-M)")             \
  X(8, 128, 128, 128, 2, 1, Sched1Sm, Epi1Sm, "1SM 128x128x128 cluster2x1x1 (B multicast)")       \
  X(9, 256, 256, 128, 2, 2, Sched2Sm, Epi2Sm, "2SM 256x256x128 cluster2x2x1 (A multicast, 4 CTAs)") \
  X(10, 256, 256, 128, 4, 1, Sched2Sm, Epi2Sm, "2SM 256x256x128 cluster4x1x1 (B multicast, 4 CTAs)") \
  X(11, 256, 256, 128, 4, 2, Sched2Sm, Epi2Sm, "2SM 256x256x128 cluster4x2x1 (A+B multicast, 8 CTAs)") \
  X(12, 256, 256, 256, 2, 1, Sched2Sm, Epi2Sm, "2SM 256x256x256 cluster2x1x1 (TileK=256)")        \
  X(13, 128, 256, 128, 2, 2, Sched1Sm, Epi1Sm, "1SM 128x256x128 cluster2x2x1 (A+B multicast, 4 CTAs)")

#define THOR_NUM_CFGS 14

// Configs also instantiated with the per-row x per-col epilogue scale (rccfg_*.cu; driver flag --scale=rowcol).
#define THOR_RC_CFG_LIST(X)                                                                        \
  X(2, 256, 128, 128, 2, 1, Sched2Sm, Epi2Sm, "2SM 256x128x128 cluster2x1x1")                     \
  X(3, 256, 256, 128, 2, 1, Sched2Sm, Epi2Sm, "2SM 256x256x128 cluster2x1x1")
