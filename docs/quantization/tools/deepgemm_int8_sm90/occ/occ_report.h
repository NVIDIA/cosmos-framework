// Occupancy / register / spill report of a compiled OCC kernel instantiation, filled on its first launch (shared by the g++ bindings and the nvcc launchers).
#pragma once
namespace dgint8 {
struct OccReport { int occupancy = -1, clusters = -1, regs = -1, local_bytes = -1, smem = -1; };
}
