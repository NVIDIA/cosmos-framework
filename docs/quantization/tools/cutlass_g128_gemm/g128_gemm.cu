// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
//
// Group-wise (g128) scaled 8-bit GEMM on Blackwell SM100, derived from CUTLASS
// examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu.
//
//   D[M,N] (bf16) = sum_g  SFA[m,g] * SFB[n,g] * ( sum_{k in group g} A[m,k] * W[n,k] )
//
//   A[M,K] 8-bit row-major,   SFA fp32: one scale per (row m, 128 consecutive K)      -> "1x128" (ScaleGranularityM=1)
//   W[N,K] 8-bit row-major,   SFB fp32: one scale per (col n, 128 consecutive K)      -> "1x128" (ScaleGranularityN=1, SFN=1)
//                                       or per (128 cols, 128 K)                     -> "128x128" (SFN=128, DeepSeek layout)
//   Scales vary along K, so they are applied inside the mainloop once per K tile (TileK == ScaleGranularityK == 128).
//
//   -DKIND_FP8  : A/W = e4m3, fp32 MMA accumulator       (stock CUTLASS blockwise collective)
//   -DKIND_INT8 : A/W = int8, int32 MMA accumulator, fp32 scales/promotion  (needs the patched collective, see README)
//   -DTM/-DTN/-DTK/-DCM/-DCN : MMA tile and cluster; -DSFN : weight-scale N granularity (1 or 128)
//
// Scale memory layout (CUTLASS Sm100BlockwiseScaleConfig, default MN-major): SFA is stored [K/128][M] with m fastest,
// SFB is stored [K/128][N/SFN] with the n-block fastest. The host code below fills through the cute layout object, so it
// does not depend on that detail; a production quantizer must emit exactly this layout.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "needs CUDA >= 12.8 and -arch=sm_100a"
#endif

#if defined(KIND_INT8)
using ElementAB          = int8_t;
using ElementMmaAccum    = int32_t;    // tcgen05 kind::i8 accumulates in int32
static constexpr char const* kKind = "int8";
#elif defined(KIND_FP8)
using ElementAB          = cutlass::float_e4m3_t;
using ElementMmaAccum    = float;
static constexpr char const* kKind = "fp8";
#else
#error "define KIND_INT8 or KIND_FP8"
#endif
using ElementScale       = float;      // block scale factors and the promoted (running) accumulator
using ElementCompute     = float;

#ifndef TM
#define TM 256
#endif
#ifndef TN
#define TN 128
#endif
#ifndef TK
#define TK 128
#endif
#ifndef CM
#define CM 2
#endif
#ifndef CN
#define CN 1
#endif
#ifndef SFN
#define SFN 1
#endif
static constexpr int kSFM = 1, kSFN = SFN, kSFK = 128;
static_assert(TK % kSFK == 0, "g128: TileK must be a multiple of the K scale granularity (128)");

using LayoutA = cutlass::layout::RowMajor;      // A[M,K], K contiguous
using LayoutB = cutlass::layout::ColumnMajor;   // B[K,N] = W[N,K]^T, K contiguous
using LayoutC = cutlass::layout::RowMajor;      // D[M,N]
constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using MmaTileShape_MNK = Shape<Int<TM>, Int<TN>, Int<TK>>;
using ClusterShape_MNK = Shape<Int<CM>, Int<CN>, _1>;

using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<kSFM, kSFN, kSFK>;
using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

// Epilogue consumes the promoted fp32 accumulator; the mainloop MMA accumulates in ElementMmaAccum (int32 for INT8).
// For INT8 the shadowed collective in ./include decouples scale / promoted-accumulator types from the MMA accumulator.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementScale, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementAB, cute::tuple<LayoutA, LayoutSFA>, AlignmentAB,
    ElementAB, cute::tuple<LayoutB, LayoutSFB>, AlignmentAB,
    ElementMmaAccum,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm100Blockwise
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1);} } while (0)
#define CUTLASS_CHECK(x) do { cutlass::Status s = (x); if (s != cutlass::Status::kSuccess) { \
  std::fprintf(stderr, "CUTLASS error %s at %s:%d\n", cutlassGetStatusString(s), __FILE__, __LINE__); std::exit(1);} } while (0)

struct Options {
  bool help = false;
  int m = 4096, n = 4096, k = 4096;
  int iterations = 50, warmup = 10;
  bool flush_l2 = true;
  int verify_samples = 2048;
  uint64_t seed = 2026;
  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("flush_l2", flush_l2, true);
    cmd.get_cmd_line_argument("verify_samples", verify_samples);
  }
  void usage() const {
    std::printf("g128_gemm_%s_%dx%dx%d_c%dx%d_sfn%d --m= --n= --k= [--iterations=50 --warmup=10 --flush_l2=1 --verify_samples=2048]\n",
                kKind, TM, TN, TK, CM, CN, SFN);
  }
};

static inline uint64_t lcg(uint64_t& s) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 33; }
template <class T> static void fill_operand(std::vector<T>& h, uint64_t seed);
template <> void fill_operand<int8_t>(std::vector<int8_t>& h, uint64_t seed) {
  for (auto& v : h) v = static_cast<int8_t>(static_cast<int>(lcg(seed) % 255) - 127);
}
template <> void fill_operand<cutlass::float_e4m3_t>(std::vector<cutlass::float_e4m3_t>& h, uint64_t seed) {
  for (auto& v : h) v = cutlass::float_e4m3_t(static_cast<float>(static_cast<int>(lcg(seed) % 17) - 8));   // exact in e4m3
}
static inline double to_double(int8_t v) { return static_cast<double>(v); }
static inline double to_double(cutlass::float_e4m3_t v) { return static_cast<double>(static_cast<float>(v)); }
// scales: realistic magnitude (absmax/127 with absmax in [0.5, 2.0]) but exactly representable dyadic values are not
// required; fp32 is what the kernel multiplies with, and the reference uses the same fp32 values in fp64 arithmetic.
static inline float rand_scale(uint64_t& s) { return (0.5f + static_cast<float>(lcg(s) % 1536) / 1024.0f) / 127.0f; }

int main(int argc, char const** argv) {
  Options opt; opt.parse(argc, argv);
  if (opt.help) { opt.usage(); return 0; }
  cudaDeviceProp props; int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
  if (props.major != 10) { std::fprintf(stderr, "needs an SM100 GPU, got sm_%d%d\n", props.major, props.minor); return 2; }
  int const M_true = opt.m, N = opt.n, K = opt.k;
  if (K % kSFK) { std::fprintf(stderr, "K must be a multiple of %d\n", kSFK); return 2; }
  // Per-row scale factors are copied with 16-byte cp.async, so the SFA layout requires M % 4 == 0 (cuBLASLt blockwise
  // has the same restriction). Pad rows with zeros and time the padded GEMM; TFLOPS are reported for the true M.
  int const M = (M_true + 3) / 4 * 4;
  int const KB = K / kSFK, NB = (N + kSFN - 1) / kSFN;
  size_t const nA = size_t(M) * K, nB = size_t(N) * K, nD = size_t(M) * N;

  // operands
  std::vector<ElementAB> hA(nA), hB(nB);
  fill_operand(hA, opt.seed + 1);
  fill_operand(hB, opt.seed + 2);
  for (size_t i = size_t(M_true) * K; i < nA; ++i) hA[i] = ElementAB(0);   // padded rows
  // logical scales: sa[m][kb], sw[nb][kb]
  std::vector<float> sa(size_t(M) * KB), sw(size_t(NB) * KB);
  { uint64_t s = opt.seed + 3; for (auto& v : sa) v = rand_scale(s); s = opt.seed + 4; for (auto& v : sw) v = rand_scale(s); }

  // CUTLASS scale layouts for this problem size, filled through the layout object
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  size_t const nSFA = size(filter_zeros(layout_SFA)), nSFB = size(filter_zeros(layout_SFB));
  std::vector<float> hSFA(nSFA, 0.f), hSFB(nSFB, 0.f);
  for (int m = 0; m < M; ++m) for (int kb = 0; kb < KB; ++kb)
    hSFA[layout_SFA(make_coord(make_coord(m % kSFM, m / kSFM), make_coord(0, kb), 0))] = sa[size_t(m) * KB + kb];
  for (int n = 0; n < N; ++n) for (int kb = 0; kb < KB; ++kb)
    hSFB[layout_SFB(make_coord(make_coord(n % kSFN, n / kSFN), make_coord(0, kb), 0))] = sw[size_t(n / kSFN) * KB + kb];

  cutlass::device_memory::allocation<ElementAB> dA(nA), dB(nB);
  cutlass::device_memory::allocation<float> dSFA(nSFA), dSFB(nSFB);
  cutlass::device_memory::allocation<ElementD> dD(nD);
  CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), nA * sizeof(ElementAB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), nB * sizeof(ElementAB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dSFA.get(), hSFA.data(), nSFA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dSFB.get(), hSFB.data(), nSFB * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD.get(), 0, nD * sizeof(ElementD)));

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {dA.get(), stride_A, dB.get(), stride_B, dSFA.get(), layout_SFA, dSFB.get(), layout_SFB},
    {{}, nullptr, stride_C, dD.get(), stride_D}
  };
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.hw_info.device_id = dev;
  args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> ws(ws_size);
  CUTLASS_CHECK(gemm.can_implement(args));
  CUTLASS_CHECK(gemm.initialize(args, ws.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- exact sampled verification: Y = sum_g sa[m,g] sw[n,g] <A[m,g,:], W[n,g,:]> ----
  std::string verdict = "skipped"; double max_rel = 0.0;
  if (opt.verify_samples > 0) {
    std::vector<ElementD> hD(nD);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD.get(), nD * sizeof(ElementD), cudaMemcpyDeviceToHost));
    uint64_t s = opt.seed + 5; int bad = 0;
    for (int i = 0; i < opt.verify_samples; ++i) {
      int mi = int(lcg(s) % M_true), ni = int(lcg(s) % N);
      if (i < 8) mi = M_true - 1 - i;
      if (i >= 8 && i < 16) ni = N - 1 - (i - 8);
      double y = 0.0;
      ElementAB const* a = hA.data() + size_t(mi) * K;
      ElementAB const* b = hB.data() + size_t(ni) * K;
      for (int kb = 0; kb < KB; ++kb) {
        double dot = 0.0;
        for (int kk = kb * kSFK; kk < (kb + 1) * kSFK; ++kk) dot += to_double(a[kk]) * to_double(b[kk]);
        y += dot * double(sa[size_t(mi) * KB + kb]) * double(sw[size_t(ni / kSFN) * KB + kb]);
      }
      double got = double(static_cast<float>(hD[size_t(mi) * N + ni]));
      double rel = std::fabs(got - y) / std::max(std::fabs(y), 1e-3);
      max_rel = std::max(max_rel, rel);
      if (rel > 1.5e-2) { if (bad < 5) std::fprintf(stderr, "MISMATCH m=%d n=%d ref=%g got=%g\n", mi, ni, y, got); ++bad; }
    }
    verdict = bad ? "FAIL" : "PASS";
  }

  // ---- timing ----
  for (int i = 0; i < opt.warmup; ++i) CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  double us_med = 0.0, us_mean = 0.0;
  if (opt.flush_l2) {
    size_t const flush_bytes = size_t(512) << 20;
    cutlass::device_memory::allocation<uint8_t> flush(flush_bytes);
    std::vector<float> t(opt.iterations);
    for (int i = 0; i < opt.iterations; ++i) {
      CUDA_CHECK(cudaMemsetAsync(flush.get(), i & 0xff, flush_bytes));
      CUDA_CHECK(cudaEventRecord(e0));
      CUTLASS_CHECK(gemm.run());
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); t[i] = ms * 1000.f;
    }
    std::vector<float> sorted(t); std::sort(sorted.begin(), sorted.end());
    us_med = sorted[sorted.size() / 2];
    double sum = 0; for (float v : t) sum += v; us_mean = sum / t.size();
  } else {
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < opt.iterations; ++i) CUTLASS_CHECK(gemm.run());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    us_med = us_mean = ms * 1000.0 / opt.iterations;
  }
  double tflops = 2.0 * M_true * N * K / (us_med * 1e-6) / 1e12;
  std::printf("RESULT kind=%s_g128 tile=%dx%dx%d cluster=%dx%d sfn=%d m=%d m_pad=%d n=%d k=%d flush=%d us_med=%.2f us_mean=%.2f tflops=%.1f "
              "verify=%s max_rel=%.2e sm=%d%d\n",
              kKind, TM, TN, TK, CM, CN, SFN, M_true, M, N, K, int(opt.flush_l2), us_med, us_mean, tflops,
              verdict.c_str(), max_rel, props.major, props.minor);
  return verdict == "FAIL" ? 1 : 0;
}
