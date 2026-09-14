// CUTLASS 3.x / Blackwell-family (Thor sm_110a, also sm_100a) per-tensor scaled GEMM, one template for FP8 and INT8.
//   D[M,N] = bf16( (A[M,K] . W[N,K]^T) * sa * sb )
// A row-major (K-major), W = nn.Linear weight [N,K] row-major == ColumnMajor KxN (K-major) => "TN" GEMM, D row-major bf16.
// FP8: e4m3 x e4m3 -> fp32 accumulate (tcgen05.mma kind::f8f6f4). INT8: s8 x s8 -> int32 accumulate (kind::i8).
// The scale product is applied in the epilogue as an EVT: Sm90ScalarBroadcast<float, ., 2, multiplies> (sa*sb, host scalars or
// device pointers) times Sm90AccFetch, converted to bf16. Derived from CUTLASS examples/70_blackwell_gemm/70_blackwell_fp8_gemm.cu
// with the heavy fusion (bias/ReLU/aux/amax) replaced by the plain per-tensor scale, so FP8 and INT8 share an identical structure.
#pragma once
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
#include <string>

namespace thor {
using namespace cute;

template <class ElementAB_, class ElementAcc_, class MmaTile_, class Cluster_, class KernelSchedule_, class EpiSchedule_,
          class TileScheduler_ = void>
struct PerTensorGemm {
  using ElementA = ElementAB_;  using LayoutA = cutlass::layout::RowMajor;     static constexpr int AlignA = 16;
  using ElementB = ElementAB_;  using LayoutB = cutlass::layout::ColumnMajor;  static constexpr int AlignB = 16;
  using ElementD = cutlass::bfloat16_t; using LayoutD = cutlass::layout::RowMajor; static constexpr int AlignD = 8;
  using ElementAcc = ElementAcc_;
  using ElementCompute = float;
  using MmaTile = MmaTile_;
  using Cluster = Cluster_;

  // EVT: D = bf16( acc * (sa * sb) )
  using Scale = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float, Stride<_0, _0, _0>, 2, cutlass::multiplies>;
  using Mul = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementCompute,
                                                     cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Mul, Scale, cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute,
      void, LayoutD, AlignD, ElementD, LayoutD, AlignD, EpiSchedule_, EVT>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule_>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
                                                          TileScheduler_>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // raster: 0 = scheduler heuristic, 1 = AlongM, 2 = AlongN. swizzle = max_swizzle_size (tiles grouped along the fast raster dim;
  // on Thor's 32 MB L2 this decides whether W is re-streamed from DRAM for every M row-block: see README).
  static typename Gemm::Arguments make_args(ElementA const* A, ElementB const* B, ElementD* D, int M, int N, int K,
                                            float sa, float sb, float const* sa_ptr, float const* sb_ptr, int swizzle, int raster) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
                                  {A, stride_A, B, stride_B},
                                  {{}, nullptr, stride_D, D, stride_D}};
    // EVT arguments: {Scale{scalars, scalar_ptrs, dScalar}, AccFetch{}, Mul{}}
    args.epilogue.thread = {{{sa, sb}, {sa_ptr, sb_ptr}, {}}, {}, {}};
    args.scheduler.max_swizzle_size = swizzle;
    using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
    return args;
  }
};

// Type-erased handle so the benchmark driver can hold any instantiation.
struct GemmHandle {
  virtual ~GemmHandle() = default;
  // A/B/D device pointers of the instantiation's element types; sa/sb host scalars (used when sa_ptr/sb_ptr are null).
  virtual void init(void const* A, void const* B, void* D, int M, int N, int K, float sa, float sb,
                    float const* sa_ptr, float const* sb_ptr, int swizzle, int raster, cudaStream_t stream) = 0;
  virtual void run(cudaStream_t stream) = 0;
  virtual std::string desc() const = 0;
  virtual int tile_m() const = 0;
  virtual int tile_n() const = 0;
  virtual int tile_k() const = 0;
  virtual int stages() const = 0;
  virtual size_t smem_bytes() const = 0;
};

template <class G>
struct HandleImpl : GemmHandle {
  using Gemm = typename G::Gemm;
  Gemm gemm;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  std::string d;
  explicit HandleImpl(std::string desc_) : d(std::move(desc_)) {}
  ~HandleImpl() override { if (workspace) cudaFree(workspace); }

  void init(void const* A, void const* B, void* D, int M, int N, int K, float sa, float sb, float const* sa_ptr,
            float const* sb_ptr, int swizzle, int raster, cudaStream_t stream) override {
    auto args = G::make_args(reinterpret_cast<typename G::ElementA const*>(A), reinterpret_cast<typename G::ElementB const*>(B),
                             reinterpret_cast<typename G::ElementD*>(D), M, N, K, sa, sb, sa_ptr, sb_ptr, swizzle, raster);
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess)
      throw std::runtime_error(std::string("can_implement failed: ") + cutlassGetStatusString(st));
    size_t ws = Gemm::get_workspace_size(args);
    if (ws > workspace_bytes) {
      if (workspace) cudaFree(workspace);
      if (cudaMalloc(&workspace, ws) != cudaSuccess) throw std::runtime_error("workspace cudaMalloc failed");
      workspace_bytes = ws;
    }
    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess)
      throw std::runtime_error(std::string("initialize failed: ") + cutlassGetStatusString(st));
  }
  void run(cudaStream_t stream) override {
    cutlass::Status st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("run failed: ") + cutlassGetStatusString(st));
  }
  std::string desc() const override { return d; }
  int tile_m() const override { return size<0>(typename G::MmaTile{}); }
  int tile_n() const override { return size<1>(typename G::MmaTile{}); }
  int tile_k() const override { return size<2>(typename G::MmaTile{}); }
  int stages() const override { return G::CollectiveMainloop::DispatchPolicy::Stages; }
  size_t smem_bytes() const override { return sizeof(typename G::GemmKernel::SharedStorage); }
};

using Sched1Sm = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using Sched2Sm = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using Epi1Sm = cutlass::epilogue::TmaWarpSpecialized1Sm;
using Epi2Sm = cutlass::epilogue::TmaWarpSpecialized2Sm;
using SchedAuto = cutlass::gemm::collective::KernelScheduleAuto;
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;

}  // namespace thor

// Per-config factory, one instantiation per translation unit (see configs.h / cfg_*.cu).
#define THOR_DECL(DT, I) thor::GemmHandle* thor_make_##DT##_##I();
#define THOR_DEF(DT, ELEM, ACC, I, TM, TN, TK, CM, CN, SCHED, EPI, DESC)                                                       \
  thor::GemmHandle* thor_make_##DT##_##I() {                                                                                   \
    using G = thor::PerTensorGemm<ELEM, ACC, cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>,                        \
                                  cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>, thor::SCHED, thor::EPI>;               \
    return new thor::HandleImpl<G>(DESC);                                                                                      \
  }
