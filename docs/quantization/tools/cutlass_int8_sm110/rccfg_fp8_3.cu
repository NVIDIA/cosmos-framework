#include "pertensor_gemm.cuh"
#include "configs.h"
#undef THOR_CFG_LIST
#define THOR_CFG_LIST THOR_RC_CFG_LIST
#define X(I, TM, TN, TK, CM, CN, S, E, D) THOR_DEF_RC(fp8, cutlass::float_e4m3_t, float, I, TM, TN, TK, CM, CN, S, E, D)
#define THOR_ONLY 3
#include "cfg_select.h"
