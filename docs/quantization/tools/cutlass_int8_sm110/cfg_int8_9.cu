#include "pertensor_gemm.cuh"
#include "configs.h"
#define X(I, TM, TN, TK, CM, CN, S, E, D) THOR_DEF(int8, int8_t, int32_t, I, TM, TN, TK, CM, CN, S, E, D)
#define THOR_ONLY 9
#include "cfg_select.h"
