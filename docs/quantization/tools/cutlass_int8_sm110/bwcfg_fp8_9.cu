#include "blockwise_gemm.cuh"
#include "bw_configs.h"
#define X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) THOR_BW_DEF(fp8, cutlass::float_e4m3_t, float, I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#define THOR_ONLY 9
#include "bw_cfg_select.h"
