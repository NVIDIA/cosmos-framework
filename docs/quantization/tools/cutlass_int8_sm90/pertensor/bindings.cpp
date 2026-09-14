#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include <string>
size_t int8sm90_ws_0(int M, int N, int K);
void int8sm90_run_0(const int8_t*, const int8_t*, const float*, const float*, void*, int, int, int, void*, cudaStream_t);

using WsFn = size_t(*)(int,int,int);
using RunFn = void(*)(const int8_t*, const int8_t*, const float*, const float*, void*, int, int, int, void*, cudaStream_t);
static std::pair<WsFn, RunFn> pick(int cfg) {
  switch (cfg) {
    case 0: return std::make_pair(&int8sm90_ws_0, &int8sm90_run_0);
    default: TORCH_CHECK(false, "unknown config ", cfg);
  }
}
static const std::vector<std::string> kDescs = {
    "128x256x128 cluster2x1 cooperative (TMA warp-specialized, s8*s8->s32, EVT sa[m]*sb[n] -> bf16)"
};

// A: [M,K] int8 row-major; W: [N,K] int8 row-major; sa: [M] fp32; sb: [N] fp32 -> D [M,N] bf16 = (A @ W^T) * sa[:,None] * sb[None,:]
torch::Tensor int8_scaled_mm(torch::Tensor A, torch::Tensor W, torch::Tensor sa, torch::Tensor sb, int64_t cfg) {
  TORCH_CHECK(A.is_cuda() && W.is_cuda() && A.dtype() == torch::kInt8 && W.dtype() == torch::kInt8, "A/W must be CUDA int8");
  TORCH_CHECK(A.dim() == 2 && W.dim() == 2 && A.is_contiguous() && W.is_contiguous(), "A/W must be 2D contiguous");
  TORCH_CHECK(A.size(1) == W.size(1), "K mismatch");
  TORCH_CHECK(sa.dtype() == torch::kFloat32 && sb.dtype() == torch::kFloat32 && sa.is_contiguous() && sb.is_contiguous(), "scales fp32 contiguous");
  const int M = A.size(0), K = A.size(1), N = W.size(0);
  TORCH_CHECK(sa.numel() == M && sb.numel() == N, "sa must have M elements, sb N elements");
  TORCH_CHECK(K % 16 == 0 && N % 8 == 0, "K%16==0 and N%8==0 required");
  const c10::cuda::OptionalCUDAGuard guard(A.device());
  auto D = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));
  auto fns = pick((int)cfg);
  size_t ws = fns.first(M, N, K);
  auto workspace = torch::empty({(int64_t)std::max<size_t>(ws, 1)}, A.options().dtype(torch::kUInt8));
  fns.second(A.data_ptr<int8_t>(), W.data_ptr<int8_t>(), sa.data_ptr<float>(), sb.data_ptr<float>(), D.data_ptr(), M, N, K,
             workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  return D;
}
std::vector<std::string> configs() { return kDescs; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("int8_scaled_mm", &int8_scaled_mm, "SM90 CUTLASS int8 scaled GEMM"); m.def("configs", &configs); }
