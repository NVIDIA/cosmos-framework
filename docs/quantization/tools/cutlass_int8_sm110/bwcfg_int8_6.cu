#include "blockwise_gemm.cuh"
#include "bw_configs.h"
#define X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) THOR_BW_DEF(int8, int8_t, int32_t, I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#define THOR_ONLY 6
#include "bw_cfg_select.h"
