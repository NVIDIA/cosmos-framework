// CUTLASS 3.x / SM90 INT8 x INT8 -> INT32 GEMM (TMA + wgmma, warp-specialized), epilogue: D[m,n] = bf16(acc * sa[m] * sb[n]).
// Layout: A[M,K] row-major (K-major), B given as W[N,K] row-major == ColumnMajor KxN (K-major)  => "TN" GEMM, D[M,N] row-major bf16.
// per-tensor scaling = pass sa/sb vectors filled with the scalar (cheap); per-row x per-col also supported directly.
#pragma once
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
#include <string>

namespace int8sm90 {
using namespace cute;

template <class TileShape, class ClusterShape, class KernelSchedule, class EpilogueSchedule>
struct Int8Gemm {
  using ElementA = int8_t;  using LayoutA = cutlass::layout::RowMajor;     static constexpr int AlignA = 16;
  using ElementB = int8_t;  using LayoutB = cutlass::layout::ColumnMajor;  static constexpr int AlignB = 16;
  using ElementD = cutlass::bfloat16_t; using LayoutD = cutlass::layout::RowMajor; static constexpr int AlignD = 8;
  using ElementAcc = int32_t; using ElementCompute = float;

  // EVT: D = bf16( (acc * sb[n]) * sa[m] )
  using Accum   = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA  = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<_1, _0, _0>>;
  using ScaleB  = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<_0, _1, _0>>;
  using Mul0    = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0    = cutlass::epilogue::fusion::Sm90EVT<Mul0, ScaleB, Accum>;
  using Mul1    = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT     = cutlass::epilogue::fusion::Sm90EVT<Mul1, ScaleA, EVT0>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute,
      void, LayoutD, AlignD, ElementD, LayoutD, AlignD, EpilogueSchedule, EVT>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
                                                          cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static size_t workspace_size(int M, int N, int K) { return Gemm::get_workspace_size(make_args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K)); }

  static typename Gemm::Arguments make_args(int8_t const* A, int8_t const* B, float const* sa, float const* sb, cutlass::bfloat16_t* D,
                                            int M, int N, int K) {
    using StrideA = typename GemmKernel::StrideA; using StrideB = typename GemmKernel::StrideB; using StrideD = typename GemmKernel::StrideD;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
                                  {A, stride_A, B, stride_B},
                                  {{}, nullptr, stride_D, D, stride_D}};
    // EVT arguments: {child0 (ScaleA), child1 (EVT0 = {ScaleB, Accum, Mul0}), Mul1}
    args.epilogue.thread = {{sa, 0.f, {}}, {{sb, 0.f, {}}, {}, {}}, {}};
    return args;
  }

  static void run(int8_t const* A, int8_t const* B, float const* sa, float const* sb, cutlass::bfloat16_t* D, int M, int N, int K,
                  void* workspace, cudaStream_t stream) {
    Gemm gemm;
    auto args = make_args(A, B, sa, sb, D, M, N, K);
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("cutlass can_implement failed: ") + cutlassGetStatusString(st));
    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("cutlass initialize failed: ") + cutlassGetStatusString(st));
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("cutlass run failed: ") + cutlassGetStatusString(st));
  }
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  using EpiCoop = cutlass::epilogue::TmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;     using EpiPing = cutlass::epilogue::TmaWarpSpecialized;
}  // namespace int8sm90

// Per-config entry points (each instantiated in its own .cu so ninja compiles them in parallel).
#define INT8SM90_DECL(i) size_t int8sm90_ws_##i(int M, int N, int K); \
  void int8sm90_run_##i(const int8_t* A, const int8_t* B, const float* sa, const float* sb, void* D, int M, int N, int K, void* ws, cudaStream_t s);
#define INT8SM90_DEF(i, TM, TN, TK, CM, CN, SCHED, EPI) \
  using G##i = int8sm90::Int8Gemm<cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>, cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>, int8sm90::SCHED, int8sm90::EPI>; \
  size_t int8sm90_ws_##i(int M, int N, int K) { return G##i::workspace_size(M, N, K); } \
  void int8sm90_run_##i(const int8_t* A, const int8_t* B, const float* sa, const float* sb, void* D, int M, int N, int K, void* ws, cudaStream_t s) { \
    G##i::run(A, B, sa, sb, reinterpret_cast<cutlass::bfloat16_t*>(D), M, N, K, ws, s); }
