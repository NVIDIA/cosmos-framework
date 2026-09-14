#include "pertensor_gemm.cuh"
#include "configs.h"
#undef THOR_CFG_LIST
#define THOR_CFG_LIST THOR_RC_CFG_LIST
#define X(I, TM, TN, TK, CM, CN, S, E, D) THOR_DEF_RC(int8, int8_t, int32_t, I, TM, TN, TK, CM, CN, S, E, D)
#define THOR_ONLY 3
#include "cfg_select.h"
