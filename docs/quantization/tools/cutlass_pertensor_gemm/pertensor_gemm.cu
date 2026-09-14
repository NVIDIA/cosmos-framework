// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
//
// Per-tensor scaled 8-bit GEMM on Blackwell SM100 (tcgen05), derived from CUTLASS
// examples/70_blackwell_gemm/70_blackwell_fp8_gemm.cu.
//
//   D[M,N] (bf16) = scale_a * scale_b * (A[M,K] @ W[N,K]^T)
//
// The same source builds two kernels:
//   -DKIND_FP8  : A/W = e4m3, fp32 accumulator  (== official Cosmos3 FP8 per-tensor path)
//   -DKIND_INT8 : A/W = int8, int32 accumulator (the INT8 per-tensor twin)
// The per-tensor scales are applied in the epilogue by ScaledLinCombPerRowBiasEltAct
// (Z = scale_a * scale_b * alpha * acc), exactly like the FP8 example, so the two kernels
// differ only in the MMA kind. Tile / cluster shape come from -DTM/-DTN/-DTK/-DCM/-DCN.
//
// Layout matches a Linear layer: A row-major [M,K] (K contiguous), W stored [N,K] row-major,
// i.e. B = W^T is a K-major "ColumnMajor" [K,N] operand; D row-major [M,N].

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
#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

#ifndef ARCH
#define ARCH 100            // 100 = Blackwell SM100 (tcgen05, 1SM/2SM UMMA); 90 = Hopper SM90 (wgmma)
#endif
#if ARCH == 100 && !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "needs CUDA >= 12.8 and -arch=sm_100a"
#endif

#if defined(KIND_INT8)
using ElementAB          = int8_t;
using ElementAccumulator = int32_t;
static constexpr char const* kKind = "int8";
#elif defined(KIND_FP8)
using ElementAB          = cutlass::float_e4m3_t;
using ElementAccumulator = float;
static constexpr char const* kKind = "fp8";
#else
#error "define KIND_INT8 or KIND_FP8"
#endif

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

using LayoutA = cutlass::layout::RowMajor;      // A[M,K], K contiguous
using LayoutB = cutlass::layout::ColumnMajor;   // B[K,N] = W[N,K]^T, K contiguous
using LayoutC = cutlass::layout::RowMajor;      // D[M,N]
constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;   // 16 elements

using ElementC       = cutlass::bfloat16_t;     // never loaded (beta = 0), type needed by the fusion
using ElementD       = cutlass::bfloat16_t;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementCompute = float;
using ElementBias    = cutlass::bfloat16_t;

using MmaTileShape_MNK = Shape<Int<TM>, Int<TN>, Int<TK>>;   // SM100: MMA tile (2SM if CM % 2 == 0); SM90: CTA tile
using ClusterShape_MNK = Shape<Int<CM>, Int<CN>, _1>;

#if ARCH == 100
using ArchTag          = cutlass::arch::Sm100;
#if defined(STREAMK)
// Stream-K needs the explicit (static-cluster) 1SM/2SM schedules, as in example 74 and the SM100 stream-K unit
// tests; KernelScheduleAuto picks the dynamic-cluster/CLC policy and the stream-K kernel then never terminates.
using MainloopSchedule = cute::conditional_t<(CM % 2 == 0), cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                                            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
using EpilogueSchedule = cute::conditional_t<(CM % 2 == 0), cutlass::epilogue::TmaWarpSpecialized2Sm,
                                                            cutlass::epilogue::TmaWarpSpecialized1Sm>;
#else
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;           // picks 1SM/2SM from ClusterShape
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#endif
#else
using ArchTag          = cutlass::arch::Sm90;
// Cooperative (default, example 54): 2 consumer warpgroups share one 128xN tile.
// Pingpong (-DPINGPONG): 2 consumer warpgroups alternate on separate 64xN tiles, overlapping epilogue with MMA.
#if defined(PINGPONG)
#if defined(KIND_FP8) && defined(FP8_FAST_ACCUM)
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
#else
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
#endif
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
#else
#if defined(KIND_FP8) && defined(FP8_FAST_ACCUM)
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;  // == cuBLASLt use_fast_accum
#else
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
#endif
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#endif
#endif

#if defined(STREAMK)
using TileScheduler = cutlass::gemm::StreamKScheduler;
static constexpr char const* kSched = "streamk";
#elif defined(PINGPONG)
using TileScheduler = void;
static constexpr char const* kSched = "pingpong";
#else
using TileScheduler = void;   // SM100: cluster-launch-control scheduler; SM90: persistent
static constexpr char const* kSched = "default";
#endif

// Z = scale_a * scale_b * alpha * acc (+ beta * scale_c * C + bias, both disabled at runtime); D = Z
using FusionOp = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
    cutlass::epilogue::thread::Identity, ElementD, ElementCompute, ElementBias, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutC, AlignmentC,
    EpilogueSchedule,
    FusionOp
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    ElementAB, LayoutA, AlignmentAB,
    ElementAB, LayoutB, AlignmentAB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
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
  int verify_samples = 4096;
  float scale_a = 1.0f / 127.0f, scale_b = 1.0f / 127.0f;
  int swizzle = 0;
  uint64_t seed = 2026;
  int splits = 1;                       // stream-K binaries only
  std::string decomposition = "Heuristic";   // Heuristic | StreamK | SplitK | DataParallel
  std::string reduction = "Deterministic";   // Deterministic | Nondeterministic

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
    cmd.get_cmd_line_argument("scale_a", scale_a);
    cmd.get_cmd_line_argument("scale_b", scale_b);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    cmd.get_cmd_line_argument("splits", splits);
    cmd.get_cmd_line_argument("decomposition", decomposition);
    cmd.get_cmd_line_argument("reduction", reduction);
  }
  void usage() const {
    std::printf("pertensor_gemm_%s_%dx%dx%d_c%dx%d [sm_%d %s] --m= --n= --k= [--iterations=50 --warmup=10 --flush_l2=1 "
                "--verify_samples=4096 --scale_a= --scale_b= --swizzle=0]\n", kKind, TM, TN, TK, CM, CN, ARCH, kSched);
  }
};

// Deterministic host-side operand fill: uniform int8 in [-127,127] (both kinds; e4m3 holds every
// integer in [-8,8] exactly, so for fp8 we use ints in [-8,8] to keep the reference exact as well).
static inline uint64_t lcg(uint64_t& s) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s >> 33; }

template <class T> static void fill_operand(std::vector<T>& h, uint64_t seed);
template <> void fill_operand<int8_t>(std::vector<int8_t>& h, uint64_t seed) {
  for (auto& v : h) v = static_cast<int8_t>(static_cast<int>(lcg(seed) % 255) - 127);
}
template <> void fill_operand<cutlass::float_e4m3_t>(std::vector<cutlass::float_e4m3_t>& h, uint64_t seed) {
  for (auto& v : h) v = cutlass::float_e4m3_t(static_cast<float>(static_cast<int>(lcg(seed) % 17) - 8));
}
static inline double to_double(int8_t v) { return static_cast<double>(v); }
static inline double to_double(cutlass::float_e4m3_t v) { return static_cast<double>(static_cast<float>(v)); }

int main(int argc, char const** argv) {
  Options opt; opt.parse(argc, argv);
  if (opt.help) { opt.usage(); return 0; }

  cudaDeviceProp props; int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
  if (props.major != ARCH / 10) { std::fprintf(stderr, "binary built for sm_%d, GPU is sm_%d%d\n", ARCH, props.major, props.minor); return 2; }

  int const M = opt.m, N = opt.n, K = opt.k;
  size_t const nA = size_t(M) * K, nB = size_t(N) * K, nD = size_t(M) * N;

  std::vector<ElementAB> hA(nA), hB(nB);
  fill_operand(hA, opt.seed + 1);
  fill_operand(hB, opt.seed + 2);

  cutlass::device_memory::allocation<ElementAB> dA(nA), dB(nB);
  cutlass::device_memory::allocation<ElementD> dD(nD);
  CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), nA * sizeof(ElementAB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), nB * sizeof(ElementAB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD.get(), 0, nD * sizeof(ElementD)));

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {dA.get(), stride_A, dB.get(), stride_B},
    {{}, nullptr, stride_C, dD.get(), stride_D}
  };
  // hw_info must be filled explicitly: the stream-K scheduler sizes/initializes its reduction workspace from
  // args.hw_info.sm_count at get_workspace_size() time (sm_count == 0 -> undersized workspace -> kernel spins forever).
  args.hw_info.device_id = dev;
  args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
  auto& f = args.epilogue.thread;
  f.alpha = 1.0f; f.beta = 0.0f;
  f.scale_a = opt.scale_a; f.scale_b = opt.scale_b; f.scale_c = 1.0f;
  f.bias_ptr = nullptr;
  args.scheduler.max_swizzle_size = opt.swizzle;
#if defined(STREAMK)
  {
    using SKParams = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
    args.scheduler.splits = opt.splits;
    args.scheduler.decomposition_mode =
        opt.decomposition == "StreamK" ? SKParams::DecompositionMode::StreamK :
        opt.decomposition == "SplitK" ? SKParams::DecompositionMode::SplitK :
        opt.decomposition == "DataParallel" ? SKParams::DecompositionMode::DataParallel :
        SKParams::DecompositionMode::Heuristic;
    args.scheduler.reduction_mode = opt.reduction == "Nondeterministic" ?
        SKParams::ReductionMode::Nondeterministic : SKParams::ReductionMode::Deterministic;
  }
  std::string sched_desc = std::string(kSched) + ":" + opt.decomposition + (opt.decomposition == "SplitK" ? "x" + std::to_string(opt.splits) : "");
#else
  std::string sched_desc = kSched;
#endif

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> ws(ws_size);
  CUTLASS_CHECK(gemm.can_implement(args));
  CUTLASS_CHECK(gemm.initialize(args, ws.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- verification on random (m,n) samples against an exact host dot product ----
  std::string verdict = "skipped";
  double max_rel = 0.0;
  if (opt.verify_samples > 0) {
    std::vector<ElementD> hD(nD);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD.get(), nD * sizeof(ElementD), cudaMemcpyDeviceToHost));
    uint64_t s = opt.seed + 3;
    int bad = 0;
    for (int i = 0; i < opt.verify_samples; ++i) {
      int mi = int(lcg(s) % M), ni = int(lcg(s) % N);
      // pin a few samples to the last row/col to exercise the residue tiles
      if (i < 8) mi = M - 1 - i;
      if (i >= 8 && i < 16) ni = N - 1 - (i - 8);
      double acc = 0.0;
      ElementAB const* a = hA.data() + size_t(mi) * K;
      ElementAB const* b = hB.data() + size_t(ni) * K;
      for (int kk = 0; kk < K; ++kk) acc += to_double(a[kk]) * to_double(b[kk]);
      double ref = acc * double(opt.scale_a) * double(opt.scale_b);
      double got = double(static_cast<float>(hD[size_t(mi) * N + ni]));
      double denom = std::max(std::fabs(ref), 1e-3);
      double rel = std::fabs(got - ref) / denom;
      max_rel = std::max(max_rel, rel);
      if (rel > 1.5e-2) { if (bad < 5) std::fprintf(stderr, "MISMATCH m=%d n=%d ref=%g got=%g\n", mi, ni, ref, got); ++bad; }
    }
    verdict = bad ? "FAIL" : "PASS";
  }

  // ---- timing ----
  for (int i = 0; i < opt.warmup; ++i) CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  double us_med = 0.0, us_mean = 0.0;
  if (opt.flush_l2) {
    size_t const flush_bytes = size_t(512) << 20;   // > B200 L2 (126 MB), matches the H100 report's flush-then-time scheme
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
  double tflops = 2.0 * M * N * K / (us_med * 1e-6) / 1e12;
  std::printf("RESULT kind=%s tile=%dx%dx%d cluster=%dx%d sched=%s m=%d n=%d k=%d flush=%d us_med=%.2f us_mean=%.2f tflops=%.1f "
              "verify=%s max_rel=%.2e sm=%d%d\n",
              kKind, TM, TN, TK, CM, CN, sched_desc.c_str(), M, N, K, int(opt.flush_l2), us_med, us_mean, tflops,
              verdict.c_str(), max_rel, props.major, props.minor);
  return verdict == "FAIL" ? 1 : 0;
}
