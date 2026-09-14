#include "pertensor_gemm.cuh"
#include "configs.h"
#define X(I, TM, TN, TK, CM, CN, S, E, D) THOR_DEF(fp8, cutlass::float_e4m3_t, float, I, TM, TN, TK, CM, CN, S, E, D)
#define THOR_ONLY 2
#include "cfg_select.h"
