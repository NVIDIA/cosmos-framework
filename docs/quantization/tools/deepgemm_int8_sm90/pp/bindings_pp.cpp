// torch bindings of the math-warpgroup ping-pong extension (module deepgemm_int8_pp_ext); mirrors ../bindings.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "pp_configs.h"

torch::Tensor int8_gemm_g128_pp(torch::Tensor A, torch::Tensor W, torch::Tensor sfa, torch::Tensor sfb,
                                c10::optional<torch::Tensor> out, int64_t cfg) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kInt8 && A.dim() == 2 && A.is_contiguous(), "A must be int8 [M,K] contiguous");
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kInt8 && W.dim() == 2 && W.is_contiguous(), "W must be int8 [N,K] contiguous");
    const int M = A.size(0), K = A.size(1), N = W.size(0);
    TORCH_CHECK(W.size(1) == K, "K mismatch");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "K and N must be multiples of 128");
    const int G = K / 128, Mp = (M + 3) / 4 * 4;
    TORCH_CHECK(sfa.dtype() == torch::kFloat32 && sfa.is_contiguous() && sfa.dim() == 2 && sfa.size(0) == G && sfa.size(1) == Mp,
                "sfa must be fp32 [K/128, round_up(M,4)] contiguous (kernel layout; see prepare_sfa)");
    TORCH_CHECK(sfb.dtype() == torch::kFloat32 && sfb.is_contiguous() && sfb.dim() == 2 && sfb.size(0) == N / 128 && sfb.size(1) == G,
                "sfb must be fp32 [N/128, K/128] contiguous");
    TORCH_CHECK(cfg >= 0 && cfg < kDgInt8PpNumCfgs, "bad config index");
    const auto& info = kDgInt8PpCfgs[cfg];
    TORCH_CHECK(info.shape_n == 0 || info.shape_n == N, "config ", cfg, " was compiled for N=", info.shape_n);
    TORCH_CHECK(info.shape_k == 0 || info.shape_k == K, "config ", cfg, " was compiled for K=", info.shape_k);
    const c10::cuda::OptionalCUDAGuard guard(A.device());
    torch::Tensor D;
    if (out.has_value()) {
        D = out.value();
        TORCH_CHECK(D.dtype() == torch::kBFloat16 && D.is_contiguous() && D.size(0) == M && D.size(1) == N, "out must be bf16 [M,N] contiguous");
    } else {
        D = torch::empty({M, N}, A.options().dtype(torch::kBFloat16));
    }
    dgint8::Args args{A.data_ptr<int8_t>(), W.data_ptr<int8_t>(), sfa.data_ptr<float>(), sfb.data_ptr<float>(), D.data_ptr(),
                      M, N, K, Mp, at::cuda::getCurrentCUDAStream()};
    kDgInt8PpRuns[cfg](args);
    return D;
}

std::vector<std::vector<int64_t>> configs() {
    std::vector<std::vector<int64_t>> r;
    for (int i = 0; i < kDgInt8PpNumCfgs; ++i) {
        const auto& c = kDgInt8PpCfgs[i];
        r.push_back({c.block_m, c.block_n, c.stages, c.mcast, c.on_a, c.shape_n, c.shape_k, c.math_threads, c.smem_max /* = pp mode */});
    }
    return r;
}
std::vector<std::string> config_names() {
    std::vector<std::string> r;
    for (int i = 0; i < kDgInt8PpNumCfgs; ++i) r.push_back(kDgInt8PpCfgs[i].name);
    return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm_g128_pp", &int8_gemm_g128_pp, "math-warpgroup ping-pong variant of the DeepGEMM INT8 port",
          py::arg("A"), py::arg("W"), py::arg("sfa"), py::arg("sfb"), py::arg("out") = py::none(), py::arg("cfg") = 0);
    m.def("configs", &configs);
    m.def("config_names", &config_names);
}
