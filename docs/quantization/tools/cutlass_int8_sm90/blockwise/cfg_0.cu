#include "int8_blockwise_gemm_sm90.cuh"
using G0 = int8bw::Int8BlockwiseGemm<128, 128, 128, 1, 2>;
size_t int8bw_ws_0(int M, int N, int K) { return G0::workspace_size(M, N, K); }
void int8bw_run_0(const int8_t* A, const int8_t* B, const float* sfa, const float* sfb, void* D, int M, int N, int K, void* ws, cudaStream_t s) {
  G0::run(A, B, sfa, sfb, reinterpret_cast<cutlass::bfloat16_t*>(D), M, N, K, ws, s);
}
int int8bw_stages_0() { return G0::Stages; }
