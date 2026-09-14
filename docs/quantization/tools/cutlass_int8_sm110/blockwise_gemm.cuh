// CUTLASS 3.x / Blackwell-family (Thor sm_110a) block-scaled ("g128") GEMM: scales applied INSIDE the mainloop per K block.
//   D[M,N] = bf16( sum_kb  sfa[m, kb] * sfb[n, kb] * sum_{k in kb} A[m,k] * W[n,k] )
// sfa: one fp32 per (GranM rows, GranK columns of K) of A -> per token per 128-K with GranM=1; sfb: one per (GranN rows of W, GranK)
// -> per output channel per 128-K with GranN=1 (the "per-col g128" layout of handoff 3.6) or 128x128 blocks with GranN=128.
// FP8 uses CUTLASS's stock SM100 blockwise collective (KernelScheduleSm100Blockwise) through the CollectiveBuilder. INT8 needs the
// ported collective (int32 partial accumulators in TMEM, fp32 scales, fp32 full accumulator) selected by thor::BlockwiseMainloop below.
// Epilogue: the same Sm90EVT as the per-tensor kernel with the scalar broadcast set to 1.0, so the epilogue cost is identical.
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
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
#include <string>

namespace thor {
using namespace cute;

// Mainloop selection: stock CUTLASS builder for fp8 (and for int8 once the ported collective exists this alias is overridden).
template <class ElementAB, class LayoutA, class LayoutSFA, class LayoutB, class LayoutSFB, class ElementAcc, class MmaTile, class Cluster,
          class StageCount, class KernelSchedule>
struct BlockwiseMainloop {
  using type = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementAB, cute::tuple<LayoutA, LayoutSFA>, 16, ElementAB, cute::tuple<LayoutB, LayoutSFB>, 16, ElementAcc, MmaTile, Cluster,
      StageCount, KernelSchedule>::CollectiveOp;
};

template <class ElementAB_, class ElementAcc_, class MmaTile_, class Cluster_, class KernelSchedule_, class EpiSchedule_, int GranM, int GranN,
          int GranK, class EpiTile_ = cutlass::epilogue::collective::EpilogueTileAuto, bool ColScaleEpi_ = false>
struct BlockwiseGemm {
  // ColScaleEpi_: "separable weight scale" variant s_w[n,g] = s_w[n] * c[g]. The mainloop only sees the per-K-block factor
  // (c[g] is folded into the activation scales sfa'[m,g] = sfa[m,g]*c[g] at quantization time, sfb is passed as all-ones or
  // c[g]), and the per-output-channel factor s_w[n] is applied once per output element in the epilogue (Sm90RowBroadcast).
  static constexpr bool ColScaleEpi = ColScaleEpi_;
  using ElementA = ElementAB_;  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementAB_;  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementD = cutlass::bfloat16_t; using LayoutD = cutlass::layout::RowMajor; static constexpr int AlignD = 8;
  using ElementAcc = ElementAcc_;  // MMA accumulator type (fp32 for fp8, int32 for int8)
  using ElementCompute = float;
  using MmaTile = MmaTile_;
  using Cluster = Cluster_;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<GranM, GranN, GranK>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  // The epilogue always sees fp32 (the full accumulator after in-mainloop scaling).
  using ElementEpiAcc = float;

  using ScaleScalar = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float, Stride<_0, _0, _0>, 2, cutlass::multiplies>;
  using ScaleCol = cutlass::epilogue::fusion::Sm90RowBroadcast<0, MmaTile, float, float, Stride<_0, _1, _0>>;   // one fp32 per output column n
  using Scale = cute::conditional_t<ColScaleEpi, ScaleCol, ScaleScalar>;
  using Mul = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementD, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<Mul, Scale, cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
      EpiTile_, ElementEpiAcc, ElementCompute,
      void, LayoutD, AlignD, ElementD, LayoutD, AlignD, EpiSchedule_, EVT>::CollectiveOp;

  using CollectiveMainloop = typename BlockwiseMainloop<
      ElementA, LayoutA, LayoutSFA, LayoutB, LayoutSFB, ElementAcc, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule_>::type;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ElementSF = cute::remove_cv_t<cute::remove_pointer_t<decltype(typename CollectiveMainloop::Arguments{}.ptr_SFA)>>;

  static typename Gemm::Arguments make_args(ElementA const* A, ElementB const* B, ElementSF const* sfa, ElementSF const* sfb, ElementD* D,
                                            int M, int N, int K, int swizzle, int raster, float const* colscale = nullptr) {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
                                  {A, stride_A, B, stride_B, sfa, layout_SFA, sfb, layout_SFB},
                                  {{}, nullptr, stride_D, D, stride_D}};
    if constexpr (ColScaleEpi) {
      args.epilogue.thread = {{colscale, 1.f, {}}, {}, {}};   // RowBroadcast: {ptr, null_default, dRow}
    } else {
      args.epilogue.thread = {{{1.f, 1.f}, {nullptr, nullptr}, {}}, {}, {}};
    }
    args.scheduler.max_swizzle_size = swizzle;
    using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
    return args;
  }
};

struct BwHandle {
  virtual ~BwHandle() = default;
  virtual void init(void const* A, void const* B, void const* sfa, void const* sfb, void* D, int M, int N, int K, int swizzle, int raster,
                    cudaStream_t stream, float const* colscale = nullptr) = 0;
  virtual bool colscale_epi() const = 0;
  virtual void run(cudaStream_t stream) = 0;
  virtual std::string desc() const = 0;
  virtual int tile_m() const = 0;
  virtual int tile_n() const = 0;
  virtual int tile_k() const = 0;
  virtual int gran_m() const = 0;
  virtual int gran_n() const = 0;
  virtual int gran_k() const = 0;
  virtual int stages() const = 0;
  virtual size_t smem_bytes() const = 0;
  virtual int sf_bytes() const = 0;  // sizeof(ElementSF): 4 for fp32 scales (int32 = stock collective with int8 -> unusable)
};

template <class G>
struct BwHandleImpl : BwHandle {
  using Gemm = typename G::Gemm;
  Gemm gemm;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  std::string d;
  explicit BwHandleImpl(std::string desc_) : d(std::move(desc_)) {}
  ~BwHandleImpl() override { if (workspace) cudaFree(workspace); }
  void init(void const* A, void const* B, void const* sfa, void const* sfb, void* D, int M, int N, int K, int swizzle, int raster,
            cudaStream_t stream, float const* colscale = nullptr) override {
    auto args = G::make_args(reinterpret_cast<typename G::ElementA const*>(A), reinterpret_cast<typename G::ElementB const*>(B),
                             reinterpret_cast<typename G::ElementSF const*>(sfa), reinterpret_cast<typename G::ElementSF const*>(sfb),
                             reinterpret_cast<typename G::ElementD*>(D), M, N, K, swizzle, raster, colscale);
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("can_implement failed: ") + cutlassGetStatusString(st));
    size_t ws = Gemm::get_workspace_size(args);
    if (ws > workspace_bytes) {
      if (workspace) cudaFree(workspace);
      if (cudaMalloc(&workspace, ws) != cudaSuccess) throw std::runtime_error("workspace cudaMalloc failed");
      workspace_bytes = ws;
    }
    st = gemm.initialize(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("initialize failed: ") + cutlassGetStatusString(st));
  }
  void run(cudaStream_t stream) override {
    cutlass::Status st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("run failed: ") + cutlassGetStatusString(st));
  }
  std::string desc() const override { return d; }
  int tile_m() const override { return size<0>(typename G::MmaTile{}); }
  int tile_n() const override { return size<1>(typename G::MmaTile{}); }
  int tile_k() const override { return size<2>(typename G::MmaTile{}); }
  int gran_m() const override { return size<0, 0>(typename G::LayoutSFA{}.shape()); }
  int gran_n() const override { return size<0, 0>(typename G::LayoutSFB{}.shape()); }
  int gran_k() const override { return size<1, 0>(typename G::LayoutSFA{}.shape()); }
  int stages() const override { return G::CollectiveMainloop::DispatchPolicy::Stages; }
  size_t smem_bytes() const override { return sizeof(typename G::GemmKernel::SharedStorage); }
  int sf_bytes() const override { return (int)sizeof(typename G::ElementSF); }
  bool colscale_epi() const override { return G::ColScaleEpi; }
};

using BwSchedAuto = cutlass::gemm::KernelScheduleSm100Blockwise;
using BwSched1Sm = cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100;
using BwSched2Sm = cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100;
using Epi1Sm = cutlass::epilogue::TmaWarpSpecialized1Sm;
using Epi2Sm = cutlass::epilogue::TmaWarpSpecialized2Sm;
using EpiAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
}  // namespace thor

#define THOR_BW_DECL(DT, I) thor::BwHandle* thor_bw_make_##DT##_##I();
// EPI: epilogue schedule tag; it may be wrapped as thor::EpiTileN<Sched, N> to force a 128xN epilogue sub-tile (smaller TMEM
// load fragments in the promotion loop => lower register pressure).
namespace thor {
template <class Sched, int N> struct EpiTileN { using sched = Sched; using tile = cute::Shape<cute::_128, cute::Int<N>>; };
template <class E> struct EpiTraits { using sched = E; using tile = cutlass::epilogue::collective::EpilogueTileAuto; static constexpr bool colscale = false; };
template <class S, int N> struct EpiTraits<EpiTileN<S, N>> { using sched = S; using tile = cute::Shape<cute::_128, cute::Int<N>>; static constexpr bool colscale = false; };
template <class E> struct EpiSep { using inner = E; };   // wrap an epilogue tag: separable-scale variant (per-column scale in the epilogue)
template <class E> struct EpiTraits<EpiSep<E>> { using sched = typename EpiTraits<E>::sched; using tile = typename EpiTraits<E>::tile; static constexpr bool colscale = true; };
using Epi2SmSep = EpiSep<Epi2Sm>;
using Epi2Sm16 = EpiTileN<Epi2Sm, 16>;   // comma-free aliases for the X-macro config list
using Epi2Sm64 = EpiTileN<Epi2Sm, 64>;
using Epi1Sm16 = EpiTileN<Epi1Sm, 16>;
}  // namespace thor
#define THOR_BW_DEF(DT, ELEM, ACC, I, TM, TN, TK, CM, CN, SCHED, EPI, GM, GN, GK, DESC)                                            \
  thor::BwHandle* thor_bw_make_##DT##_##I() {                                                                                       \
    using G = thor::BlockwiseGemm<ELEM, ACC, cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>,                             \
                                  cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>, thor::SCHED,                                 \
                                  typename thor::EpiTraits<thor::EPI>::sched, GM, GN, GK, typename thor::EpiTraits<thor::EPI>::tile,       \
                                  thor::EpiTraits<thor::EPI>::colscale>;                                                                  \
    return new thor::BwHandleImpl<G>(DESC);                                                                                         \
  }
