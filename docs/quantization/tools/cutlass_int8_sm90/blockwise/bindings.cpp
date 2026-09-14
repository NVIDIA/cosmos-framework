#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>
size_t int8bw_ws_0(int M, int N, int K);
void int8bw_run_0(const int8_t*, const int8_t*, const float*, const float*, void*, int, int, int, void*, cudaStream_t);
int int8bw_stages_0();
torch::Tensor int8_blockwise_mm(torch::Tensor A, torch::Tensor W, torch::Tensor sfa, torch::Tensor sfb) {
  TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kInt8 && W.dtype() == torch::kInt8 && A.dim() == 2 && W.dim() == 2 && A.is_contiguous() && W.is_contiguous());
  const int M = A.size(0), K = A.size(1), N = W.size(0);
  TORCH_CHECK(W.size(1) == K, "K mismatch"); TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
  TORCH_CHECK(M % 4 == 0, "M must be a multiple of 4 (TMA-loaded SFA); pad on the python side");
  TORCH_CHECK(sfa.dtype() == torch::kFloat32 && sfa.is_contiguous() && sfa.dim() == 2 && sfa.size(0) == K / 128 && sfa.size(1) == M, "sfa must be [K/128, M] fp32 contiguous");
  TORCH_CHECK(sfb.dtype() == torch::kFloat32 && sfb.is_contiguous() && sfb.dim() == 2 && sfb.size(0) == N / 128 && sfb.size(1) == K / 128, "sfb must be [N/128, K/128] fp32 contiguous");
  const c10::cuda::OptionalCUDAGuard guard(A.device());
  auto D = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));
  size_t ws = int8bw_ws_0(M, N, K);
  auto workspace = torch::empty({(int64_t)std::max<size_t>(ws, 1)}, A.options().dtype(torch::kUInt8));
  int8bw_run_0(A.data_ptr<int8_t>(), W.data_ptr<int8_t>(), sfa.data_ptr<float>(), sfb.data_ptr<float>(), D.data_ptr(), M, N, K, workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  return D;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_blockwise_mm", &int8_blockwise_mm, "SM90 CUTLASS int8 GEMM with (1,128,128) block scaling in the mainloop");
  m.def("stages", &int8bw_stages_0);
}
