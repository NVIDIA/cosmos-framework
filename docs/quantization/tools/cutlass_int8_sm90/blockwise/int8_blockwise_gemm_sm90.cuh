// CUTLASS 3.x / SM90 INT8 GEMM with DeepSeek-style (1,128,128) block scaling inside the mainloop.
//   A[M,K] int8, fp32 scale per (row, 128-K block):   SFA element (m,kb) at kb*M + m        (torch [K/128, M] contiguous)
//   W[N,K] int8, fp32 scale per (128-N, 128-K) block: SFB element (nb,kb) at nb*(K/128) + kb (torch [N/128, K/128] contiguous)
//   D[M,N] = bf16( sum_kb SFA[m,kb]*SFB[n/128,kb] * (int32 dot over block kb) )
#pragma once
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>
#include <string>

namespace cutlass::gemm {
template <int Stages_, class ClusterShape_ = cute::Shape<cute::_1, cute::_1, cute::_1>, class KernelSchedule = KernelTmaWarpSpecializedCooperative>
struct MainloopSm90TmaGmmaWarpSpecializedBlockwiseInt8 : MainloopSm90TmaGmmaWarpSpecialized<Stages_, ClusterShape_, KernelSchedule> {};
namespace kernel::detail {
// REQUIRED: the SM90 warp-specialized kernels only run the extra "MainloopAux" producer warp (which cp.async-loads the block scales and
// arrives on the mainloop barrier) when this trait is true for the mainloop's DispatchPolicy. Without it the consumers wait forever.
template <int Stages, class ClusterShape, class KernelSchedule>
struct HasAuxiliaryLoad<MainloopSm90TmaGmmaWarpSpecializedBlockwiseInt8<Stages, ClusterShape, KernelSchedule>> : cute::true_type {};
}  // namespace kernel::detail
}  // namespace cutlass::gemm

#include "sm90_mma_int8_blockwise.cuh"   // CollectiveMma<MainloopSm90TmaGmmaWarpSpecializedBlockwiseInt8, ...>

namespace cutlass::epilogue::fusion {
// The kernel hands the epilogue an int32-typed fragment whose bits are the fp32 main accumulator: reinterpret, do not convert.
struct Sm90AccFetchBitcastF32 : Sm90VisitorImpl<> {
  using Sm90VisitorImpl<>::Sm90VisitorImpl;
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<float, FragmentSize> visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      static_assert(sizeof(ElementAccumulator) == sizeof(float));
      return reinterpret_cast<Array<float, FragmentSize> const&>(frg_acc);
    }
  };
  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) { return ConsumerStoreCallbacks{}; }
};
}  // namespace cutlass::epilogue::fusion

namespace int8bw {
using namespace cute;
namespace cdet = cutlass::gemm::collective::detail;

template <int TM, int TN, int TK, int CM, int CN>
struct Int8BlockwiseGemm {
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ClusterShape = Shape<Int<CM>, Int<CN>, _1>;
  static constexpr int GranM = 1, GranN = 128, GranK = 128;
  static_assert(TK == GranK, "one 128-K scale block per k-tile");
  static_assert(TN % GranN == 0);
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<GranM, GranN, GranK, GMMA::Major::MN, GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using ElementA = int8_t; using ElementB = int8_t; using ElementAcc = int32_t; using ElementD = cutlass::bfloat16_t;
  using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
  using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  using StrideD = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;

  using EVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetchBitcastF32>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, float, void, cutlass::layout::RowMajor, 8, ElementD, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative, EVT>::CollectiveOp;

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ElementA, ElementB, ElementAcc, TileShape, GMMA::Major::K, GMMA::Major::K>(), Layout<Shape<_2, _1, _1>>{}));
  using GmemTiledCopyA = decltype(cdet::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
  using GmemTiledCopyB = decltype(cdet::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
  using SmemLayoutAtomA = decltype(cdet::ss_smem_selector<GMMA::Major::K, ElementA, Int<TM>, Int<TK>>());
  using SmemLayoutAtomB = decltype(cdet::ss_smem_selector<GMMA::Major::K, ElementB, Int<TN>, Int<TK>>());
  static constexpr int ScaleMsPerTile = TM / GranM, ScaleNsPerTile = TN / GranN;
  static constexpr int Stages = cdet::compute_stage_count_with_blockwise_scale<cdet::sm90_smem_capacity_bytes, ElementA, ElementB, float, TileShape,
      ScaleMsPerTile, ScaleNsPerTile>(cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>{});
  using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedBlockwiseInt8<Stages, ClusterShape, cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy, TileShape, ElementA, cute::tuple<StrideA, LayoutSFA>, ElementB, cute::tuple<StrideB, LayoutSFB>, TiledMma,
      GmemTiledCopyA, SmemLayoutAtomA, void, cute::identity, GmemTiledCopyB, SmemLayoutAtomB, void, cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename Gemm::Arguments make_args(int8_t const* A, int8_t const* B, float const* sfa, float const* sfb, ElementD* D, int M, int N, int K) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    return typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
                                    {A, stride_A, B, stride_B, sfa, layout_SFA, sfb, layout_SFB}, {{}, nullptr, stride_D, D, stride_D}};
  }
  static size_t workspace_size(int M, int N, int K) { return Gemm::get_workspace_size(make_args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K)); }
  static void run(int8_t const* A, int8_t const* B, float const* sfa, float const* sfb, ElementD* D, int M, int N, int K, void* ws, cudaStream_t stream) {
    Gemm gemm; auto args = make_args(A, B, sfa, sfb, D, M, N, K);
    cutlass::Status st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("cutlass can_implement failed: ") + cutlassGetStatusString(st));
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("cutlass initialize failed: ") + cutlassGetStatusString(st));
    st = gemm.run(stream);
    if (st != cutlass::Status::kSuccess) throw std::runtime_error(std::string("cutlass run failed: ") + cutlassGetStatusString(st));
  }
};
}  // namespace int8bw
