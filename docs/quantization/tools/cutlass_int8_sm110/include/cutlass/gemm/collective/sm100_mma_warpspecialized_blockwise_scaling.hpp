/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
// ---------------------------------------------------------------------------------------------------------------
// LOCAL PATCH (cosmos-framework docs/quantization/tools/cutlass_g128_gemm): INT8 support for the SM100 blockwise-scaling
// mainloop. Upstream ties the scale-factor type and the promoted (running) accumulator type to TiledMma::ValTypeC, which
// is int32 for kind::i8 MMAs. This copy decouples them: the MMA still accumulates in ValTypeC (TMEM), while scale factors
// and the promoted accumulator are fp32 whenever ValTypeC is an integer type. For floating-point MMAs the file is
// behaviourally identical to upstream. Shadowed via -I ordering (this directory precedes the CUTLASS include dir).
// ---------------------------------------------------------------------------------------------------------------

#include "cutlass/cutlass.h"
#include "cute/arch/simd_sm100.hpp"   // THOR PATCH: f32x2 packed math for the promotion loop
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// Both DMA Load and MMA methods of this class must be run by a single thread that's picked by elect_one
template <
  int Stages,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,   // Static cluster shape or dynamic (int, int, _1)
  class TileShape_,     // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape>,
    TileShape_,
    ElementA_,
    StridePairA_,
    ElementB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  // LOCAL PATCH: fp32 scales / promoted accumulator for integer MMA accumulators (see banner).
  using ElementPromoted = cute::conditional_t<cute::is_integral_v<typename TiledMma::ValTypeC>, float, typename TiledMma::ValTypeC>;
  using ElementSF = ElementPromoted;

  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedBlockwiseScaling<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;

  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = cute::remove_cvref_t<decltype(get<0>(StridePairA_{}))>;
  using LayoutSFA = cute::remove_cvref_t<decltype(get<1>(StridePairA_{}))>;
  using ElementSFA = ElementSF;
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = cute::remove_cvref_t<decltype(get<0>(StridePairB_{}))>;
  using LayoutSFB = cute::remove_cvref_t<decltype(get<1>(StridePairB_{}))>;
  using ElementSFB = ElementSF;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static_assert(size<0>(TileShape{}) % ScaleGranularityM == 0 and ScaleGranularityM <= size<0>(TileShape{}), "Scale Granularity M must divide Tile Shape");

  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;
  static_assert(size<1>(TileShape{}) % ScaleGranularityN == 0 and ScaleGranularityN <= size<1>(TileShape{}), "Scale Granularity N must divide Tile Shape");

  static_assert(size<1, 0>(LayoutSFA{}) == size<1, 0>(LayoutSFB{}), "Vector size K must be equal for SFA and SFB");

  static constexpr int ScaleGranularityK = size<1, 0>(LayoutSFA{});
  static constexpr int ScaleKsPerTile = size<2>(TileShape{}) / ScaleGranularityK;
  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0 and ScaleGranularityK <= size<2>(TileShape{}), "Scale Granularity K must divide Tile Shape");
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0, "Scale Granularity K must be divisible by MMA_K");

  static constexpr int K_BLOCK_MMAS_PER_SCALE_K = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      size<0,1>(LayoutSFA{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K,
      size<0,1>(LayoutSFB{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K>;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                       "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  static_assert(size<0>(CtaShape_MNK{}) >= ScaleGranularityM, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<1>(CtaShape_MNK{}) >= ScaleGranularityN, "Scale Granularity must be smaller than or equal to the tile shape");
  static_assert(size<2>(CtaShape_MNK{}) >= ScaleGranularityK, "Scale Granularity must be smaller than or equal to the tile shape");

  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(CtaShape_MNK{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(CtaShape_MNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();

  static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

  static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
                (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
                "ElementA and ElementB should be both runtime or both static.");

  static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

  using ElementAccumulator = typename TiledMma::ValTypeC;
  // THOR PATCH (bias): pre-biased int32 accumulator. Every accumulator stage is kept filled with BiasBits = 0x4B400000 (the bit
  // pattern of 1.5*2^23) by the promotion warps before it is released, and the MMA accumulates onto it instead of overwriting.
  // The int32 partial sum x then reads back from TMEM as the fp32 bit pattern of (1.5*2^23 + x) -- exact for |x| < 2^22, which
  // covers g128 (128*127*127) and g256 (256*127*127) -- so the half-rate I2F disappears: t = fma(fb, sa, -1.5*2^23*sa) == x*sa
  // (bit-identical to FMUL(float(x), sa) when 1.5*2^23*sa is exact, i.e. sa's two low mantissa bits are 0; the kernel clears them),
  // then acc = fma(t, sb, acc). Two FFMA per element instead of I2F + FMUL + FFMA. Only for integer accumulators with per-column
  // weight scales (the W-block path gains nothing: its I2F already overlaps the FFMA).
#if defined(G128_OPT_BIAS)
#if !defined(G128_OPT_PIPE2)
#error "G128_OPT_BIAS requires G128_OPT_PIPE2: the bias-aware promotion (register bias add, fma(fb, sa, -1.5*2^23*sa)) exists only in the PIPE2 branch of accum()"
#endif
  static constexpr bool UseBias = cute::is_same_v<ElementAccumulator, int32_t> && ScaleGranularityN == 1;
#else
  static constexpr bool UseBias = false;
#endif
  static constexpr int32_t BiasBits = 0x4B400000;
  static constexpr float   BiasF    = 12582912.f;
  // 1.5*2^23 + x must stay in [2^23, 2^24) for the bit pattern to equal the float value: |x| <= GK * 128 * 128 <= 2^22 -> GK <= 256.
  static_assert(!UseBias || ScaleGranularityK <= 256, "TMEM bias trick needs a K group of at most 256 int8 products");
  using GmemTiledCopyA = cute::remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopySFA = cute::remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopyB = cute::remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB_{}))>;
  using GmemTiledCopySFB = cute::remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB_{}))>;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopABPipeline = cutlass::PipelineTmaUmmaAsync<
                                DispatchPolicy::Stages,
                                ClusterShape,
                                AtomThrShapeMNK>;
  using MainloopABPipelineState = typename MainloopABPipeline::PipelineState;

  using MainloopSFPipeline = cutlass::PipelineAsync<DispatchPolicy::Stages>;
  using MainloopSFPipelineState = typename MainloopSFPipeline::PipelineState;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
                                  AccumulatorPipelineStageCount,
                                  AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  static constexpr int CopyAlignmentSFA = GmemTiledCopySFA::AtomNumVal::value * sizeof(typename GmemTiledCopySFA::ValType) / sizeof(ElementSF);
  static constexpr int CopyAlignmentSFB = GmemTiledCopySFB::AtomNumVal::value * sizeof(typename GmemTiledCopySFB::ValType) / sizeof(ElementSF);

  static constexpr int AlignmentSFA = CopyAlignmentSFA * (GmemTiledCopySFA::AtomNumVal::value > 1 ?
      (size<0,1>(LayoutSFA{}.stride()) == 1 ? ScaleGranularityM : ScaleGranularityK) : 1);
  static constexpr int AlignmentSFB = CopyAlignmentSFB * (GmemTiledCopySFB::AtomNumVal::value > 1 ?
      (size<0,1>(LayoutSFB{}.stride()) == 1 ? ScaleGranularityN : ScaleGranularityK) : 1);


  // Two arrivals per thread in the warp (1 arrival and 1 arrival through cp.async.mbarrier)
  static constexpr int NumMainloopSFProducerThreadEvents = 64;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtomA must be rank 2 (M,K)");
  static_assert(((size<0,0>(MmaShapeA_MK{}) * size<1>(MmaShapeA_MK{})) % size<0>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeA_MK{}) * size<2>(MmaShapeA_MK{})) % size<1>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtomB must be rank 2 (N,K)");
  static_assert(((size<0,0>(MmaShapeB_NK{}) * size<1>(MmaShapeB_NK{})) % size<0>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(((size<0,1>(MmaShapeB_NK{}) * size<2>(MmaShapeB_NK{})) % size<1>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>)) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        (cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD_MULTICAST>)),
      "GmemTiledCopy - invalid TMA copy atom specified.");
  static_assert(
      (size(AtomThrShapeMNK{}) == 1 &&
        (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)) ||
      (size(AtomThrShapeMNK{}) == 2 &&
        (cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD> || cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD_MULTICAST>)),
      "GmemTiledCopy -  invalid TMA copy atom specified.");

  using TmaInternalElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, cutlass::tfloat32_t, ElementAMma>;
  using TmaInternalElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, cutlass::tfloat32_t, ElementBMma>;

  using SmemAllocTypeA = cute::conditional_t<cute::sizeof_bits_v<ElementAMma> < 8, uint8_t, ElementAMma>;
  using SmemAllocTypeB = cute::conditional_t<cute::sizeof_bits_v<ElementBMma> < 8, uint8_t, ElementBMma>;

  using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
  using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

  using RuntimeDataTypeA = cute::conditional_t<IsRuntimeDataTypeA, cute::UMMA::MXF8F6F4Format, void*>;
  using RuntimeDataTypeB = cute::conditional_t<IsRuntimeDataTypeB, cute::UMMA::MXF8F6F4Format, void*>;

  using SmemLayoutScaleA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutScaleB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutScaleA>> smem_SFA;
      cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutScaleB>> smem_SFB;
    } tensors;

    using PipelineABStorage = typename MainloopABPipeline::SharedStorage;
    using PipelineSFStorage = typename MainloopSFPipeline::SharedStorage;
    using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

    struct PipelineStorage {
      alignas(16) PipelineABStorage pipeline_ab;
      alignas(16) PipelineSFStorage pipeline_sf;
      alignas(16) AccumulatorPipelineStorage pipeline_accum;
    };
  };

  // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
  static constexpr uint32_t TmaTransactionBytes =
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>) +
    cutlass::bits_to_bytes(size(AtomThrShapeMNK{}) * cosize(take<0,3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);

  template<class AccTensor>
  struct TmemStorage {
    AccTensor accumulators;
  };

  template<
    class KTileCount,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB
  >
  struct LoadABParams {
    // for scheduler
    KTileCount k_tiles;
    // for input tensor values
    GTensorPartitionedA tAgA_mkl;
    GTensorPartitionedB tBgB_nkl;
    STensorA tAsA;
    STensorB tBsB;

    // the TMA multicast masks
    uint16_t mcast_mask_a;
    uint16_t mcast_mask_b;

    CUTLASS_DEVICE
    LoadABParams (
        KTileCount k_tiles_,
        GTensorPartitionedA tAgA_mkl_, GTensorPartitionedB tBgB_nkl_,
        STensorA tAsA_, STensorB tBsB_,
        uint16_t mcast_mask_a_, uint16_t mcast_mask_b_)
    : k_tiles(k_tiles_)
    , tAgA_mkl(tAgA_mkl_), tBgB_nkl(tBgB_nkl_)
    , tAsA(tAsA_), tBsB(tBsB_)
    , mcast_mask_a(mcast_mask_a_), mcast_mask_b(mcast_mask_b_) {}
  };

  template<
    class KTileCount,
    class GTensorScaleA, class GTensorScaleB,
    class IdentTensorScaleA, class IdentTensorScaleB,
    class STensorScaleA, class STensorScaleB
  >
  struct LoadSFParams {
    // for scheduler
    KTileCount k_tiles;

    GTensorScaleA gSFA_mkl;
    GTensorScaleB gSFB_nkl;
    IdentTensorScaleA identSFA_mkl;
    IdentTensorScaleB identSFB_nkl;
    STensorScaleA sSFA;
    STensorScaleB sSFB;

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;

    CUTLASS_DEVICE
    LoadSFParams (
        KTileCount k_tiles_,
        GTensorScaleA gSFA_mkl_, GTensorScaleB gSFB_nkl_,
        IdentTensorScaleA identSFA_mkl_, IdentTensorScaleB identSFB_nkl_,
        STensorScaleA sSFA_, STensorScaleB sSFB_,
        LayoutSFA layout_SFA_, LayoutSFB layout_SFB_)
    : k_tiles(k_tiles_)
    , gSFA_mkl(gSFA_mkl_), gSFB_nkl(gSFB_nkl_)
    , identSFA_mkl(identSFA_mkl_), identSFB_nkl(identSFB_nkl_)
    , sSFA(sSFA_), sSFB(sSFB_)
    , layout_SFA(layout_SFA_), layout_SFB(layout_SFB_) {}
  };

  template<class FragmentA, class FragmentB>
  struct MmaParams {
    TiledMma tiled_mma;
    FragmentA tCrA;
    FragmentB tCrB;

    CUTLASS_DEVICE
    MmaParams (
        TiledMma tiled_mma_,
        FragmentA tCrA_, FragmentB tCrB_)
    : tiled_mma(tiled_mma_)
    , tCrA(tCrA_), tCrB(tCrB_) {}
  };

  template<
    class STensorScaleA, class STensorScaleB
  >
  struct AccumTransformParams {
    // for scheduler

    STensorScaleA sSFA;
    STensorScaleB sSFB;

    CUTLASS_DEVICE
    AccumTransformParams (
        STensorScaleA sSFA_, STensorScaleB sSFB_)
    :  sSFA(sSFA_), sSFB(sSFB_) {}
  };


  // Host side kernel arguments
  struct Arguments {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
    RuntimeDataTypeA runtime_data_type_a{};
    RuntimeDataTypeB runtime_data_type_b{};
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using TMA_B = decltype(make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
    RuntimeDataTypeA runtime_data_type_a;
    RuntimeDataTypeB runtime_data_type_b;

    ElementSF const* ptr_SFA;
    LayoutSFA layout_SFA;
    ElementSF const* ptr_SFB;
    LayoutSFB layout_SFB;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster)
    , runtime_data_type_a_(params.runtime_data_type_a)
    , runtime_data_type_b_(params.runtime_data_type_b) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
    ProblemShape const& problem_shape,
    Arguments const& args,
    [[maybe_unused]] void* workspace,
    cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

    // Cluster layout for TMA construction
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));
    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<TmaInternalElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<TmaInternalElementB>(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk_fallback);

    return {
      tma_load_a,
      tma_load_b,
      tma_load_a_fallback,
      tma_load_b_fallback,
      hw_info.cluster_shape_fallback,
      args.runtime_data_type_a,
      args.runtime_data_type_b,
      args.ptr_SFA,
      args.layout_SFA,
      args.ptr_SFB,
      args.layout_SFB
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();
    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;

    bool implementable = true;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    bool implementable_sf = cutlass::detail::check_alignment<CopyAlignmentSFA>(args.layout_SFA);
    implementable_sf = implementable_sf && cutlass::detail::check_alignment<CopyAlignmentSFB>(args.layout_SFB);

    if (!implementable_sf) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for Scale Factors.\n");
    }

    return implementable && implementable_sf;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE void
  prefetch_tma_descriptors() {
    cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
    cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
  }

  /// Construct A Single Stage's Accumulator Shape
  CUTLASS_DEVICE static
  auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));     // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

    return acc_shape;
  }

  template <class TmemStorage>
  CUTLASS_DEVICE static
  auto
  slice_accumulator(TmemStorage tmem_storage, int stage) {
    return cute::make_tuple(tmem_storage.accumulators(_,_,_,stage));
  }

  template<class EpilogueTile, bool IsOverlappingAccum = false>
  CUTLASS_DEVICE static
  auto
  init_tmem_tensors(EpilogueTile epi_tile) {
    TiledMma tiled_mma;
    auto acc_shape = partition_accumulator_shape();
    // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE) where ACC_PIPE=2 so we can double buffer our accumulators for mainloop and epilogue.
    Tensor accumulators = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
        tiled_mma, acc_shape, EpilogueTile{});
    TmemStorage<decltype(accumulators)> tmem_storage;
    tmem_storage.accumulators = accumulators;
    return tmem_storage;
  }

  template<class AccTensor>
  CUTLASS_DEVICE static
  void
  set_tmem_offsets(TmemStorage<AccTensor>& tmem_storage, uint32_t tmem_base_addr) {
    tmem_storage.accumulators.data() = tmem_base_addr;
  }

  /// Set up the data needed by this collective for load.
  /// Return load params containing
  /// gA_mkl - The tiled tma tensor for input A
  /// gB_nkl - The tiled tma tensor for input B
  /// tAsA - partitioned smem tensor for A
  /// tBsB - partitioned smem tensor for B
  /// mcast_mask_a - tma multicast mask for A
  /// mcast_mask_b - tma multicast mask for B
  template <class ProblemShape_MNKL,
            class MainloopParams>
  CUTLASS_DEVICE auto
  load_ab_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));

    // Tile the tensors and defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M, BLK_K, m, k, l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N, BLK_K, n, k, l)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);                                       // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);                                       // (MMA, MMA_N, MMA_K, n, k, l)

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});      // (MMA,MMA_M,MMA_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});      // (MMA,MMA_N,MMA_K,PIPE)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    // TMA Multicast Masks
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    LoadABParams load_params {
      shape<3>(gA_mkl),                               // for scheduler
      tAgA_mkl, tBgB_nkl, tAsA, tBsB,                 // for input tensor values
      mcast_mask_a, mcast_mask_b,                     // multicast masks
    };
    return load_params;
  }

  /// Set up the data needed by this collective for load.
  /// Return load params containing
  /// tSFAgSFA_mkl - partitioned gmem tensor for SFA
  /// tSFBgSFB_nkl - partitioned gmem tensor for SFB
  /// tSFAIdentSFA_mkl - partitioned identity tensor for SFA in gmem
  /// tSFBIdentSFB_nkl - partitioned identity tensor for SFB in gmem
  /// tSFAsSFA - partitioned smem tensor for SFA
  /// tSFBsSFB - partitioned smem tensor for SFB
  /// layout_SFA - layout of SFA in gmem
  /// layout_SFB - layout of SFB in gmem
  template <class ProblemShape_MNKL,
            class MainloopParams>
  CUTLASS_DEVICE auto
  load_sf_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      MainloopParams const& mainloop_params,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor mSFA_mkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_SFA), mainloop_params.layout_SFA);    // (m,k,l)
    Tensor mSFB_nkl = make_tensor(make_gmem_ptr(mainloop_params.ptr_SFB), mainloop_params.layout_SFB);    // (n,k,l)

    Tensor SFA_mkl_ident = make_identity_tensor(shape(mainloop_params.layout_SFA));

    Tensor SFB_nkl_ident = make_identity_tensor(shape(mainloop_params.layout_SFB));

    // Tile the tensors and defer the slice
    Tensor gSFA_mkl = local_tile(mSFA_mkl, CtaShape_MNK{},
        make_coord(_,_,_), Step<_1, X,_1>{});                                                 // (BLK_M, BLK_K, m, k, l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, CtaShape_MNK{},
        make_coord(_,_,_), Step< X,_1,_1>{});                                                 // (BLK_N, BLK_K, n, k, l)

    Tensor identSFA_mkl = local_tile(SFA_mkl_ident, CtaShape_MNK{},
        make_coord(_,_,_), Step<_1, X,_1>{});                                                 // (BLK_M, BLK_K, m, k, l)
    Tensor identSFB_nkl = local_tile(SFB_nkl_ident, CtaShape_MNK{},
        make_coord(_,_,_), Step< X,_1,_1>{});                                                 // (BLK_N, BLK_K, n, k, l)

    static_assert(rank(decltype(gSFA_mkl){}) == 5);
    static_assert(rank(decltype(gSFB_nkl){}) == 5);

    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
        SmemLayoutScaleA{});                                                                          // (CTA_M,CTA_K,P)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
        SmemLayoutScaleB{});                                                                          // (CTA_M,CTA_K,P)

    LoadSFParams load_params {
      size<3>(gSFA_mkl),
      gSFA_mkl, gSFB_nkl,                             // for input scale tensor values
      identSFA_mkl, identSFB_nkl,                     // for predicating scale tensor copies
      sSFA, sSFB,                                     // for scale tensor values
      mainloop_params.layout_SFA,                     // for predicating scale tensor copies
      mainloop_params.layout_SFB                      // for predicating scale tensor copies
    };
    return load_params;
  }


  /// Set up the data needed by this collective for mma compute.
  template <class AccTensor>
  CUTLASS_DEVICE auto
  mma_init(
      [[maybe_unused]] TmemStorage<AccTensor> tmem_tensors,
      TensorStorage& shared_tensors) const {
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Allocate "fragments/descriptors" for A and B matrices
    Tensor tCrA_ = TiledMma::make_fragment_A(sA);                                              // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB_ = TiledMma::make_fragment_B(sB);                                              // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(rank(tCrA_) == _4{});

    auto mma_tile_shape_A = make_shape(get<0>(shape(tCrA_.layout())),
                                       get<1>(shape(tCrA_.layout())),
                                       Int<K_BLOCK_MMAS_PER_SCALE_K>{},
                                       _1{});

    auto mma_tile_shape_B = make_shape(get<0>(shape(tCrB_.layout())),
                                       get<1>(shape(tCrB_.layout())),
                                       Int<K_BLOCK_MMAS_PER_SCALE_K>{},
                                       _1{});

    Tensor tCrA = flat_divide(tCrA_,
        mma_tile_shape_A)(_,_,_,_0{},_0{},_0{},_,_);                      // (MMA,MMA_M,MMA_K_PER_SCALE,MMA_K_REST,PIPE)

    Tensor tCrB = flat_divide(tCrB_,
        mma_tile_shape_B)(_,_,_,_0{},_0{},_0{},_,_);                      // (MMA,MMA_N,MMA_K_PER_SCALE,MMA_K_REST,PIPE)


    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));                                          // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));

    TiledMma tiled_mma;

    if constexpr (IsRuntimeDataType) {
      // Update instruction descriptor according to runtime argument.
      // Applying bitmask (0b111) to help compiler deduce that the conversion and assignment are safe.
      tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
      tiled_mma.idesc_.b_format_ = uint8_t(runtime_data_type_b_) & 0b111;
    }
    MmaParams<decltype(tCrA), decltype(tCrB)> mma_params {
      tiled_mma,
      tCrA, tCrB
    };
    return mma_params;
  }

  /// Set up the data needed by this collective for transform.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  accum_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      TensorStorage& shared_tensors) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.begin()),
        SmemLayoutScaleA{});                                                        // (ScaleMsPerTile,ScakeKsPerTile,P)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.begin()),
        SmemLayoutScaleB{});                                                        // (ScaleNsPerTile,ScaleKsPerTile,P)


    AccumTransformParams transform_params {
      sSFA, sSFB                        // for input tensor values
    };
    return transform_params;
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class LoadABParams,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_ab(
      MainloopABPipeline mainloop_pipeline,
      MainloopABPipelineState mainloop_pipe_producer_state,
      LoadABParams const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_k_tiles,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

      using BarrierType = typename MainloopABPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      auto curr_mainloop_pipe_producer_state = mainloop_pipe_producer_state;
      ++mainloop_pipe_producer_state;
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
        copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_ab_tail(
      MainloopABPipeline mainloop_pipeline,
      MainloopABPipelineState mainloop_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
  }

  /// Perform a collective-scoped transform
  /// Load producer Perspective
  template <
    class LoadSFParams,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load_sf(
      MainloopSFPipeline mainloop_sf_pipeline,
      MainloopSFPipelineState mainloop_sf_pipe_producer_state,
      LoadSFParams const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_k_tiles,
          gSFA_mkl, gSFB_nkl,
          identSFA_mkl, identSFB_nkl,
          sSFA, sSFB,
          layout_SFA, layout_SFB] = load_inputs;

    // slice out the work coord from partitioned tensors
    GmemTiledCopySFA scale_copy_a{};
    GmemTiledCopySFB scale_copy_b{};

    Tensor gSFA_k_compact = filter_zeros(
      gSFA_mkl(_, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)));               // (BLK_M_CPT, BLK_K_CPT, k_cpt)
    Tensor gSFB_k_compact = filter_zeros(
      gSFB_nkl(_, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)));               // (BLK_N_CPT, BLK_K_CPT, k_cpt)

    Tensor identSFA_k_compact = filter_zeros(
        identSFA_mkl(_, _, get<0>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)), 
        gSFA_k_compact.stride());                                                       // (BLK_M_CPT, BLK_K_CPT, k_cpt)
    Tensor identSFB_k_compact = filter_zeros(
        identSFB_nkl(_, _, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl)), 
        gSFB_k_compact.stride());                                                       // (BLK_N_CPT, BLK_K_CPT, k_cpt)

    Tensor sSFA_compact = filter_zeros(sSFA);                                               // (BLK_M_CPT, BLK_K_CPT, P)
    Tensor sSFB_compact = filter_zeros(sSFB);                                               // (BLK_N_CPT, BLK_K_CPT, P)

    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(threadIdx.x % size(scale_copy_a));
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(threadIdx.x % size(scale_copy_b));

    Tensor tSFAgSFA_k_compact = thr_scale_copy_a.partition_S(gSFA_k_compact);                  // (CPY, BLK_M, BLK_K, k)
    Tensor tSFAIdentSFA_k_compact = thr_scale_copy_a.partition_S(identSFA_k_compact);          // (CPY, BLK_M, BLK_K, k)

    Tensor tSFAsSFA_compact = thr_scale_copy_a.partition_D(sSFA_compact);

    Tensor tSFBgSFB_k_compact = thr_scale_copy_b.partition_S(gSFB_k_compact);                  // (CPY, BLK_N, BLK_K, k)
    Tensor tSFBIdentSFB_k_compact = thr_scale_copy_b.partition_S(identSFB_k_compact);          // (CPY, BLK_N, BLK_K, k)
    Tensor tSFBsSFB_compact = thr_scale_copy_b.partition_D(sSFB_compact);

    Tensor thr_tile_pSFA = make_fragment_like<bool>(tSFAgSFA_k_compact(_0{},_,_,_0{}));
    Tensor thr_tile_pSFB = make_fragment_like<bool>(tSFBgSFB_k_compact(_0{},_,_,_0{}));

    // Issue the loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK pipe_producer_state for _writing_
      mainloop_sf_pipeline.producer_acquire(mainloop_sf_pipe_producer_state);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFA); ++i) {
        Tensor tSFAIdentSFA_compact = tSFAIdentSFA_k_compact(_0{},_,_,*k_tile_iter);
        thr_tile_pSFA(i) = elem_less(tSFAIdentSFA_compact(i), 
            shape(filter_zeros(layout_SFA))) && threadIdx.x % 32 < size(scale_copy_a);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(thr_tile_pSFB); ++i) {
        Tensor tSFBIdentSFB_compact = tSFBIdentSFB_k_compact(_0{},_,_,*k_tile_iter);
        thr_tile_pSFB(i) = elem_less(tSFBIdentSFB_compact(i), 
            shape(filter_zeros(layout_SFB))) && threadIdx.x % 32 < size(scale_copy_b);
      }

      copy_if(scale_copy_a, thr_tile_pSFA, tSFAgSFA_k_compact(_,_,_,*k_tile_iter), 
          tSFAsSFA_compact(_,_,_,mainloop_sf_pipe_producer_state.index()));
      copy_if(scale_copy_b, thr_tile_pSFB, tSFBgSFB_k_compact(_,_,_,*k_tile_iter), 
          tSFBsSFB_compact(_,_,_,mainloop_sf_pipe_producer_state.index()));
      mainloop_sf_pipeline.producer_commit(mainloop_sf_pipe_producer_state, cutlass::arch::cpasync_barrier_arrive_noinc);

      __syncwarp();

      ++mainloop_sf_pipe_producer_state;
      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_sf_pipe_producer_state, k_tile_iter);
  }

  /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
  CUTLASS_DEVICE void
  load_sf_tail(
      MainloopSFPipeline mainloop_sf_pipeline,
      MainloopSFPipelineState mainloop_sf_pipe_producer_state) {
    // Issue the epilogue waits
    // This helps avoid early exit of ctas in Cluster
    // Waits for all stages to either be released (all
    // Consumer UNLOCKs), or if the stage was never used
    // then would just be acquired since the phase was
    // still inverted from make_producer_start_state
    mainloop_sf_pipeline.producer_tail(mainloop_sf_pipe_producer_state);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class TmemStorage,
    class MmaParams,
    class CtaTileCoord
  >
  CUTLASS_DEVICE auto
  mma(
      cute::tuple<MainloopABPipeline,
                  AccumulatorPipeline> pipelines,
      cute::tuple<MainloopABPipelineState,
                  AccumulatorPipelineState> pipeline_states,
      TmemStorage tmem_storage,
      MmaParams const& mma_inputs,
      CtaTileCoord cta_tile_coord,
      int k_tile_count) {
    auto [tiled_mma, tCrA, tCrB] = mma_inputs;

    auto [mainloop_pipeline,
          accumulator_pipeline] = pipelines;

    auto [mainloop_pipe_consumer_state,
          accumulator_pipe_producer_state] = pipeline_states;

    uint32_t skip_wait = k_tile_count <= 0;
    auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

    //
    // PIPELINED MAIN LOOP
    //
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // WAIT on mainloop_pipe_consumer_state until its data are available
      // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);

      // Compute on k_tile
      int read_stage = mainloop_pipe_consumer_state.index();
      // Save current mainlop pipeline read state
      auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

      // Advance mainloop_pipe
      ++mainloop_pipe_consumer_state;
      --k_tile_count;
      skip_wait = k_tile_count <= 0;
      // Peek at next iteration
      barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

      CUTLASS_PRAGMA_UNROLL
      for (int scale_k_iter = 0; scale_k_iter < size<3>(tCrA); ++scale_k_iter) {
        accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

        auto acc = get<0>(slice_accumulator(tmem_storage, accumulator_pipe_producer_state.index()));
        static_assert(is_tmem<remove_cvref_t<decltype(acc)>>::value, "Accumulator must be tmem resident.");
        static_assert(rank(remove_cvref_t<decltype(acc)>{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

        // for each set of scale_k_blocks we zero the accumulator
        if constexpr (UseBias) {
          // THOR PATCH (bias): a stage that has been consumed once holds the bias written by the promotion warps -> accumulate onto
          // it. The first `Stages` acquisitions of the kernel see uninitialised TMEM -> overwrite (the consumer adds the bias itself).
          #if defined(CUTLASS_ARCH_TCGEN_ENABLED)
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
          tiled_mma.accumulate_ = accumulator_pipe_producer_state.count() >= cute::remove_cvref_t<decltype(accumulator_pipeline)>::Stages
                                      ? UMMA::ScaleOut::One : UMMA::ScaleOut::Zero;
        } else {
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        }
        // Unroll the K mode manually so we can set scale C to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma,
                     tCrA(_,_,k_block,scale_k_iter,read_stage),
                     tCrB(_,_,k_block,scale_k_iter,read_stage),
                     acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
        ++accumulator_pipe_producer_state;
      }
      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);

    }

    return make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state);
  }

  /// Transform
  template <
    class AccumTransformParams,
    class TmemStorage,
    class CtaTileCoord,
    class CopyOpT2R,
    class EpilogueTile
  >
  CUTLASS_DEVICE auto
  accum(
      cute::tuple<AccumulatorPipeline, MainloopSFPipeline> pipelines,
      cute::tuple<AccumulatorPipelineState, MainloopSFPipelineState> consumer_states,
      TmemStorage tmem_storage,
      AccumTransformParams const& transform_inputs,
      CtaTileCoord cta_tile_coord,
      CopyOpT2R,
      EpilogueTile,
      int k_tile_count,
      int part_idx = 0,      // THOR PATCH (8-warp promotion): this warpgroup promotes epilogue sub-tiles
      int num_parts = 1) {   //   [NSUB*part_idx/num_parts, NSUB*(part_idx+1)/num_parts); with num_parts > 1 the last
                             //   accumulator stage of the tile is NOT released (the kernel does the TMEM hand-off first).

    static_assert(size<0>(EpilogueTile{}) <= size<0>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");
    static_assert(size<1>(EpilogueTile{}) <= size<1>(CtaShape_MNK{}), "Restrict epilogue tile to be smaller than or equal to CTA Tile");


    //
    // PIPELINED Transform
    //

    Tensor acc = get<0>(slice_accumulator(tmem_storage, _0{}));

    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});

    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                          // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    // Append N with a stride of 0 to SFA
    Tensor sSFA_ = transform_inputs.sSFA;
    Tensor sSFA = make_tensor(sSFA_.data(), make_layout(
      make_shape(get<0>(sSFA_.shape()), get<1>(CtaShape_MNK{}), get<1>(sSFA_.shape()), get<2>(sSFA_.shape())),
      make_stride(get<0>(sSFA_.stride()), _0{}, get<1>(sSFA_.stride()), get<2>(sSFA_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFA) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFA) == size<1>(tAcc));

    Tensor sSFA_epi = flat_divide(sSFA, EpilogueTile{});

    // Append M with a stride of 0 to SFB
    Tensor sSFB_ = transform_inputs.sSFB;
    Tensor sSFB = make_tensor(sSFB_.data(), make_layout(
      make_shape(get<0>(CtaShape_MNK{}), get<0>(sSFB_.shape()), get<1>(sSFB_.shape()), get<2>(sSFB_.shape())),
      make_stride(_0{}, get<0>(sSFB_.stride()), get<1>(sSFB_.stride()), get<2>(sSFB_.stride()))
    ));

    CUTE_STATIC_ASSERT_V(size<0>(sSFB) == size<0>(tAcc));
    CUTE_STATIC_ASSERT_V(size<1>(sSFB) == size<1>(tAcc));

    Tensor sSFB_epi = flat_divide(sSFB, EpilogueTile{});

    TiledCopy tiled_t2r_epi = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_,_,_0{},_0{}));

    int thread_idx = threadIdx.x % size(tiled_t2r_epi);

    ThrCopy thread_t2r_epi = tiled_t2r_epi.get_slice(thread_idx);

    Tensor acc_ident_epi = make_identity_tensor(shape(tAcc_epi));

    Tensor tTR_rAcc_epi = thread_t2r_epi.partition_D(acc_ident_epi);                // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

    Tensor tTR_sSFA_epi = thread_t2r_epi.partition_D(sSFA_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    Tensor tTR_sSFB_epi = thread_t2r_epi.partition_D(sSFB_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

    static_assert(rank(decltype(tTR_sSFA_epi){}) == 7);

    Tensor tTR_FullAcc = make_tensor<ElementPromoted>(shape(tTR_rAcc_epi));
    Tensor tTR_PartAcc = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));

    Tensor tTR_rSFA_compact = make_fragment_like<ElementSF>(filter_zeros(tTR_sSFA_epi(_,_,_,_,_,_,_0{})));
    Tensor tTR_rSFB_compact = make_fragment_like<ElementSF>(filter_zeros(tTR_sSFB_epi(_,_,_,_,_,_,_0{})));

    Layout tTR_rSFA_layout = make_layout(tTR_sSFA_epi(_,_,_,_,_,_,_0{}).shape(), tTR_rSFA_compact.stride());
    Layout tTR_rSFB_layout = make_layout(tTR_sSFB_epi(_,_,_,_,_,_,_0{}).shape(), tTR_rSFB_compact.stride());

    // Zero our accumulator (only the sub-tiles this warpgroup owns, so the other half stays dead in the register allocator)
    {
      constexpr int EPI_M0 = decltype(size<2>(tAcc_epi))::value;
      constexpr int EPI_N0 = decltype(size<3>(tAcc_epi))::value;
      constexpr int NSUB0  = EPI_M0 * EPI_N0;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < NSUB0; ++s) {
        if (s * num_parts >= NSUB0 * part_idx && s * num_parts < NSUB0 * (part_idx + 1)) clear(tTR_FullAcc(_,_,_,s / EPI_N0,s % EPI_N0));
      }
    }

    auto [accumulator_pipeline, mainloop_sf_pipeline] = pipelines;
    auto [accumulator_pipe_state, mainloop_sf_pipe_state] = consumer_states;

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {

      mainloop_sf_pipeline.consumer_wait(mainloop_sf_pipe_state);
      int read_idx = mainloop_sf_pipe_state.index();

      copy(filter_zeros(tTR_sSFA_epi(_,_,_,_,_,_,read_idx)), tTR_rSFA_compact);
      CUTE_STATIC_ASSERT_V(cosize(tTR_rSFA_layout) == size(tTR_rSFA_compact));
      Tensor tTR_rSFA = make_tensor(tTR_rSFA_compact.data(), tTR_rSFA_layout);
#if defined(G128_OPT_PROMOTION)
      // LOCAL PATCH (opt): with per-column weight scales (ScaleGranularityN == 1) the per-thread SFB fragment holds one
      // fp32 per column of the tile (128-256 registers) and spills. In the T2R layout all lanes of a warp own the same
      // columns, so read SFB straight from smem (warp-uniform broadcast loads) and keep the stage until the loop is done.
      Tensor tTR_rSFB = tTR_sSFB_epi(_,_,_,_,_,_,read_idx);
#else
      copy(filter_zeros(tTR_sSFB_epi(_,_,_,_,_,_,read_idx)), tTR_rSFB_compact);
      CUTE_STATIC_ASSERT_V(cosize(tTR_rSFB_layout) == size(tTR_rSFB_compact));
      Tensor tTR_rSFB = make_tensor(tTR_rSFB_compact.data(), tTR_rSFB_layout);

      mainloop_sf_pipeline.consumer_release(mainloop_sf_pipe_state);
      ++mainloop_sf_pipe_state;
#endif

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < ScaleKsPerTile; ++k_block) {

        accumulator_pipeline.consumer_wait(accumulator_pipe_state);
        // THOR PATCH (bias): the first `Stages` stages a CTA consumes were overwritten by the MMA (no bias yet); later ones carry it.
        bool const pre_biased = UseBias && accumulator_pipe_state.count() >= cute::remove_cvref_t<decltype(accumulator_pipeline)>::Stages;
        bool const keep_stage = num_parts > 1 && k_tile_count == 1 && k_block == ScaleKsPerTile - 1;   // kept for the hand-off

        Tensor acc = get<0>(slice_accumulator(tmem_storage, accumulator_pipe_state.index()));
        Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
        Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                   // (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        Tensor tTR_tAcc = thread_t2r_epi.partition_S(tAcc_epi);                     // (T2R, T2R_M, T2R_N, EPI_M, EPI_N)

#if defined(G128_OPT_PIPE2)
        // THOR PATCH (pipe2): triple-buffered TMEM sub-tile loads with a tcgen05.wait::ld only every second sub-tile.
        // wait::ld waits for ALL outstanding loads, so a distance-1 double buffer can hide at most one sub-tile of math
        // (~60 clk for 16 columns) against ~120 clk of TMEM load latency. Here loads run two sub-tiles ahead and each
        // wait covers two loads issued during the previous two sub-tiles of math.
        {
          constexpr int EPI_M = decltype(size<2>(tAcc_epi))::value;
          constexpr int EPI_N = decltype(size<3>(tAcc_epi))::value;
          constexpr int NSUB  = EPI_M * EPI_N;
          Tensor P0 = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));
          Tensor P1 = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));
          Tensor P2 = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));
          auto load_sub = [&](int s, auto& dst) { copy(tiled_t2r_epi, tTR_tAcc(_,_,_,s / EPI_N, s % EPI_N), dst); };
          int const s_begin = NSUB * part_idx / num_parts;
          int const s_end   = NSUB * (part_idx + 1) / num_parts;
          load_sub(s_begin, P0);
          if (s_begin + 1 < s_end) load_sub(s_begin + 1, P1);
          cutlass::arch::fence_view_async_tmem_load();
          CUTLASS_PRAGMA_UNROLL
          for (int s0 = 0; s0 < NSUB; ++s0) {
            int const s = s0 + s_begin;            // s0 is the compile-time-unrolled index relative to the range start
            if (s0 >= s_end - s_begin) break;
            auto& cur = (s0 % 3 == 0) ? P0 : (s0 % 3 == 1) ? P1 : P2;
            if (s + 2 < s_end) {
              auto& nxt2 = ((s0 + 2) % 3 == 0) ? P0 : ((s0 + 2) % 3 == 1) ? P1 : P2;
              if constexpr (UseBias) cutlass::arch::fence_view_async_tmem_store();   // nxt2's registers fed the previous sub-tile's bias stores
              load_sub(s + 2, nxt2);
            }
            int const epi_m = s / EPI_N;
            int const epi_n = s % EPI_N;
            auto scale_a = tTR_rSFA(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
            auto sSFB_sub = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
            Tensor rSFB_sub_compact = make_fragment_like<ElementSF>(filter_zeros(sSFB_sub));
            {
              // column scales of this sub-tile: contiguous in smem (MN-major SFB), 128-bit loads into the compact fragment
              auto sSFB_c = filter_zeros(sSFB_sub);
              constexpr int NSB = decltype(size(sSFB_c))::value;
              if constexpr (NSB % 4 == 0) {
                uint32_t saddr = cute::cast_smem_ptr_to_uint(&sSFB_c(0));
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < NSB / 4; ++q) {
                  float x, y, z, w;
                  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "r"(saddr + 16 * q));
                  rSFB_sub_compact(4 * q + 0) = x; rSFB_sub_compact(4 * q + 1) = y;
                  rSFB_sub_compact(4 * q + 2) = z; rSFB_sub_compact(4 * q + 3) = w;
                }
              } else {
                copy(sSFB_c, rSFB_sub_compact);
              }
            }
            Tensor scale_b = make_tensor(rSFB_sub_compact.data(), make_layout(shape(sSFB_sub), rSFB_sub_compact.stride()));
            Tensor full_acc = tTR_FullAcc(_,_,_,epi_m,epi_n);
            if constexpr (UseBias) {
              // THOR PATCH (bias): see UseBias. Row scales of this thread (one per distinct row; zero strides along N).
              if (!pre_biased) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(cur); ++i) cur(i) += BiasBits;
              }
              auto sa_c = filter_zeros(scale_a);
              Tensor sa_m = make_fragment_like<ElementPromoted>(sa_c);   // sa with the 2 low mantissa bits cleared (exact -M*sa)
              Tensor c_m  = make_fragment_like<ElementPromoted>(sa_c);   // -1.5*2^23 * sa
              CUTLASS_PRAGMA_UNROLL
              for (int r = 0; r < size(sa_c); ++r) {
                sa_m(r) = __uint_as_float(__float_as_uint(static_cast<float>(sa_c(r))) & ~3u);
                c_m(r)  = -BiasF * sa_m(r);
              }
              Tensor sa_t = make_tensor(sa_m.data(), make_layout(shape(scale_a), sa_m.stride()));
              Tensor c_t  = make_tensor(c_m.data(),  make_layout(shape(scale_a), c_m.stride()));
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(full_acc); ++i) {
                float const fb = __int_as_float(cur(i));                       // = 1.5*2^23 + x exactly
                float const t  = fmaf(fb, sa_t(i), c_t(i));                    // = x * sa, rounded once
                full_acc(i) = fmaf(t, static_cast<float>(scale_b(i)), full_acc(i));
              }
              // Re-arm this sub-tile with the bias, sourcing the stores from the just-consumed fragment's own registers (no extra
              // register pressure: the promotion warps have none to spare). Part 1 leaves the hand-off stage to the kernel.
              if (!(keep_stage && part_idx == 1)) bias_rearm_subtile(tmem_storage, accumulator_pipe_state, cur, CopyOpT2R{}, EpilogueTile{}, s);
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(full_acc); ++i) {
                ElementPromoted scale = scale_a(i) * scale_b(i);
                full_acc(i) += scale * static_cast<ElementPromoted>(cur(i));
              }
            }
            if ((s0 % 2 == 1) || (s + 1 == s_end)) cutlass::arch::fence_view_async_tmem_load();   // loads s+1 (if odd) and s+2 landed
          }
        }
#elif defined(G128_OPT_PIPELINE)
        // LOCAL PATCH (pipeline): tcgen05.wait::ld waits for *all* outstanding TMEM loads of the thread, so the stock
        // load -> wait -> FMA sequence serialises TMEM latency with the promotion math. Double-buffer the sub-tile
        // fragment: issue the load of sub-tile j+1 before promoting sub-tile j, then wait once.
        constexpr int EPI_M = decltype(size<2>(tAcc_epi))::value;
        constexpr int EPI_N = decltype(size<3>(tAcc_epi))::value;
        constexpr int NSUB  = EPI_M * EPI_N;
        Tensor tTR_PartAcc1 = make_tensor<ElementAccumulator>(shape(tTR_rAcc_epi(_,_,_,_0{},_0{})));
#if !defined(G128_EXP_NO_LOAD)
        copy(tiled_t2r_epi, tTR_tAcc(_,_,_,0,0), tTR_PartAcc);
        cutlass::arch::fence_view_async_tmem_load();
#endif
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < NSUB; ++j) {
          int const epi_m = j / EPI_N;
          int const epi_n = j % EPI_N;
          auto& cur = (j % 2 == 0) ? tTR_PartAcc : tTR_PartAcc1;
          auto& nxt = (j % 2 == 0) ? tTR_PartAcc1 : tTR_PartAcc;
#if !defined(G128_EXP_NO_LOAD)
          if (j + 1 < NSUB) {
            copy(tiled_t2r_epi, tTR_tAcc(_,_,_,(j + 1) / EPI_N,(j + 1) % EPI_N), nxt);
          }
#endif
          auto scale_a = tTR_rSFA(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
          auto sSFB_sub = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
          Tensor rSFB_sub_compact = make_fragment_like<ElementSF>(filter_zeros(sSFB_sub));
#if defined(G128_OPT_SBVEC)
          // THOR PATCH: the sub-tile's distinct column scales are contiguous in smem (MN-major SFB, one row of the K block);
          // load them with 128-bit loads straight into the compact fragment instead of an element-wise cute::copy, which
          // ptxas software-pipelined into LDS.128 + 4 MOVs per 4 scales (128 MOVs per K block for per-column scales).
          {
            auto sSFB_c = filter_zeros(sSFB_sub);
            constexpr int NSB = decltype(size(sSFB_c))::value;
            if constexpr (NSB % 4 == 0) {
              ElementSF const* sp = &sSFB_c(0);
              uint32_t saddr = cute::cast_smem_ptr_to_uint(sp);
              // volatile 128-bit shared loads issued back to back: ptxas may not sink them below the FMULs one quad at a
              // time (which exposed ~25 clk of LDS latency 32x per K block); all NSB scales land before the math starts.
              CUTLASS_PRAGMA_UNROLL
              for (int q = 0; q < NSB / 4; ++q) {
                float x, y, z, w;
                asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "r"(saddr + 16 * q));
                rSFB_sub_compact(4 * q + 0) = x; rSFB_sub_compact(4 * q + 1) = y;
                rSFB_sub_compact(4 * q + 2) = z; rSFB_sub_compact(4 * q + 3) = w;
              }
            } else {
              copy(sSFB_c, rSFB_sub_compact);
            }
          }
#else
          copy(filter_zeros(sSFB_sub), rSFB_sub_compact);
#endif
          Tensor scale_b = make_tensor(rSFB_sub_compact.data(), make_layout(shape(sSFB_sub), rSFB_sub_compact.stride()));
          Tensor full_acc = tTR_FullAcc(_,_,_,epi_m,epi_n);
#if defined(G128_OPT_PACKED) && !defined(G128_EXP_NO_FMA)
          // THOR PATCH (packed promotion): convert the int32/fp32 partials once, then apply the scales with f32x2 packed
          // FMUL/FFMA (fma.rn.f32x2, 2 elements per issue slot). Fragments made by make_tensor<T>(shape) are compact
          // left-major, so consecutive i are adjacent registers and can be paired.
          {
            constexpr int NE = decltype(size(full_acc))::value;
            static_assert(NE % 2 == 0, "sub-tile fragment must have an even number of elements");
            // compile-time indexed local arrays: pure register renaming for ptxas, no memory traffic
            float pf[NE];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < NE; ++i) pf[i] = static_cast<float>(cur(i));            // I2FP (no-op for fp8)
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < NE; i += 2) {
              float2 sc, acc2, pv;
              float2 sa2 = make_float2(scale_a(i), scale_a(i + 1));
              float2 sb2 = make_float2(scale_b(i), scale_b(i + 1));
              cute::mul(sc, sa2, sb2);                                                   // FMUL2 (one per pair; hoisted by ptxas when uniform)
              pv = make_float2(pf[i], pf[i + 1]);
              acc2 = make_float2(full_acc(i), full_acc(i + 1));
              cute::fma(acc2, sc, pv, acc2);                                             // FFMA2
              full_acc(i) = acc2.x; full_acc(i + 1) = acc2.y;
            }
          }
#elif !defined(G128_EXP_NO_FMA)
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(full_acc); ++i) {
            ElementPromoted scale = scale_a(i) * scale_b(i);
            full_acc(i) += scale * static_cast<ElementPromoted>(cur(i));
          }
#else
          if (j == 0) full_acc(0) += static_cast<ElementPromoted>(cur(0)) * scale_a(0) * scale_b(0);  // keep the loads alive
#endif
#if !defined(G128_EXP_NO_LOAD)
          cutlass::arch::fence_view_async_tmem_load();   // sub-tile j+1 landed
#endif
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int epi_m = 0; epi_m < size<2>(tAcc_epi); ++epi_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int epi_n = 0; epi_n < size<3>(tAcc_epi); ++epi_n) {

            auto scale_a = tTR_rSFA(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
#if defined(G128_OPT_PROMOTION)
            // LOCAL PATCH (opt): stage only this sub-tile's column scales (<= 32 fp32) from smem into registers.
            auto sSFB_sub = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
            Tensor rSFB_sub_compact = make_fragment_like<ElementSF>(filter_zeros(sSFB_sub));
            copy(filter_zeros(sSFB_sub), rSFB_sub_compact);
            Tensor scale_b = make_tensor(rSFB_sub_compact.data(), make_layout(shape(sSFB_sub), rSFB_sub_compact.stride()));
#else
            auto scale_b = tTR_rSFB(_,_,_,epi_m,epi_n,k_block * ScaleGranularityK);
#endif

            Tensor full_acc = tTR_FullAcc(_,_,_,epi_m,epi_n);
            // Compute tmem load predication if necessary
            copy(tiled_t2r_epi, tTR_tAcc(_,_,_,epi_m,epi_n), tTR_PartAcc);
            cutlass::arch::fence_view_async_tmem_load();

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(full_acc); ++i) {
              ElementPromoted scale = scale_a(i) * scale_b(i);
              full_acc(i) += scale * static_cast<ElementPromoted>(tTR_PartAcc(i));   // LOCAL PATCH: int32 -> fp32 for INT8
            }
          }
        }
#endif
        cutlass::arch::fence_view_async_tmem_load();
        if constexpr (UseBias) {
          // THOR PATCH (bias): the sub-tiles were re-armed one by one in the loop above; make the stores visible before the release.
          // For the stage kept for the hand-off, part 1's sub-tiles are re-armed by the kernel after the epilogue group has read them.
          if (!keep_stage || part_idx == 0) {
            cutlass::arch::fence_view_async_tmem_store();                       // bias stores of this stage have landed
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
          }
        }
        if (!keep_stage) {   // THOR PATCH: keep the last stage for the hand-off
          accumulator_pipeline.consumer_release(accumulator_pipe_state);
          // release acc
          ++accumulator_pipe_state;
        }
      }
#if defined(G128_OPT_PROMOTION)
      mainloop_sf_pipeline.consumer_release(mainloop_sf_pipe_state);
      ++mainloop_sf_pipe_state;
#endif

      --k_tile_count;
    }

    return cute::make_tuple(tTR_FullAcc, tiled_t2r_epi, cute::make_tuple(accumulator_pipe_state, mainloop_sf_pipe_state));
 }

  // THOR PATCH (8-warp promotion): hand the second warpgroup's half of the fp32 full accumulator to the epilogue warpgroup
  // through the TMEM stage that held the tile's last partial (both groups have finished reading it; it is released afterwards).
  template <class TmemStorage, class AccumulatorPipelineState, class FrgEngine, class FrgLayout, class CopyOpT2R, class EpilogueTile>
  CUTLASS_DEVICE void
  handoff_store(TmemStorage tmem_storage, AccumulatorPipelineState const& state, cute::Tensor<FrgEngine, FrgLayout>& tTR_FullAcc,
                CopyOpT2R, EpilogueTile, int part_idx, int num_parts) {
    Tensor acc = get<0>(slice_accumulator(tmem_storage, state.index()));
    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});
    auto tiled_r2t = make_tmem_copy(cute::TMEM::tmem_load_to_store(CopyOpT2R{}), tAcc_epi(_,_,_0{},_0{}));
    auto thr_r2t = tiled_r2t.get_slice(threadIdx.x % size(tiled_r2t));
    Tensor tTR_tAcc = thr_r2t.partition_D(tAcc_epi);                                       // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    constexpr int EPI_N = decltype(size<3>(tAcc_epi))::value;
    constexpr int NSUB  = decltype(size<2>(tAcc_epi))::value * EPI_N;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < NSUB; ++s) {
      if (s * num_parts >= NSUB * part_idx && s * num_parts < NSUB * (part_idx + 1)) {
        Tensor src = recast<ElementAccumulator>(tTR_FullAcc(_,_,_,s / EPI_N,s % EPI_N));   // same 32-bit width, raw bits
        copy(tiled_r2t, src, tTR_tAcc(_,_,_,s / EPI_N,s % EPI_N));
      }
    }
    cutlass::arch::fence_view_async_tmem_store();
  }

  template <class TmemStorage, class AccumulatorPipelineState, class FrgEngine, class FrgLayout, class TiledCopyT2R, class EpilogueTile>
  CUTLASS_DEVICE void
  handoff_load(TmemStorage tmem_storage, AccumulatorPipelineState const& state, cute::Tensor<FrgEngine, FrgLayout>& tTR_FullAcc,
               TiledCopyT2R tiled_t2r_epi, EpilogueTile, int part_idx, int num_parts) {
    Tensor acc = get<0>(slice_accumulator(tmem_storage, state.index()));
    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});
    auto thr_t2r = tiled_t2r_epi.get_slice(threadIdx.x % size(tiled_t2r_epi));
    Tensor tTR_tAcc = thr_t2r.partition_S(tAcc_epi);
    constexpr int EPI_N = decltype(size<3>(tAcc_epi))::value;
    constexpr int NSUB  = decltype(size<2>(tAcc_epi))::value * EPI_N;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < NSUB; ++s) {
      if (s * num_parts >= NSUB * part_idx && s * num_parts < NSUB * (part_idx + 1)) {
        Tensor dst = recast<ElementAccumulator>(tTR_FullAcc(_,_,_,s / EPI_N,s % EPI_N));
        copy(tiled_t2r_epi, tTR_tAcc(_,_,_,s / EPI_N,s % EPI_N), dst);
      }
    }
    cutlass::arch::fence_view_async_tmem_load();
  }

  // THOR PATCH (bias): re-arm epilogue sub-tile s of accumulator stage `state` with BiasBits, storing from the first registers of the
  // consumed TMEM fragment `cur` (they are dead after the promotion; the caller waits with fence_view_async_tmem_store() before that
  // fragment is reloaded). 8-column stores in a non-unrolled loop: ptxas gives every static tcgen05.st its own source register block.
  template <class TmemStorage, class AccumulatorPipelineState, class CurEngine, class CurLayout, class CopyOpT2R, class EpilogueTile>
  CUTLASS_DEVICE void
  bias_rearm_subtile(TmemStorage tmem_storage, AccumulatorPipelineState const& state, cute::Tensor<CurEngine, CurLayout>& cur,
                     CopyOpT2R, EpilogueTile, int s) {
    Tensor acc = get<0>(slice_accumulator(tmem_storage, state.index()));
    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});
    auto tiled_st = make_tmem_copy(SM100_TMEM_STORE_32dp32b8x{}, tAcc_epi(_,_,_0{},_0{}));
    auto thr_st = tiled_st.get_slice(threadIdx.x % size(tiled_st));
    Tensor tST = thr_st.partition_D(tAcc_epi);                                               // TMEM side  (V_tmem,ST_M,ST_N,EPI_M,EPI_N)
    Tensor tRS = thr_st.partition_S(make_identity_tensor(shape(tAcc_epi)));                 // register side (V_reg,ST_M,ST_N,EPI_M,EPI_N)
    constexpr int EPI_N = decltype(size<3>(tAcc_epi))::value;
    constexpr int NV = decltype(size(tRS(_,_0{},_0{},_0{},_0{})))::value;
    static_assert(NV <= decltype(size(cur))::value, "fragment too small for the bias store");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NV; ++i) cur(i) = ElementAccumulator(BiasBits);
    Tensor src = make_tensor(cur.data(), shape(tRS(_,_0{},_0{},_0{},_0{})));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(tST); ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < size<2>(tST); ++n) copy(tiled_st, src, tST(_,m,n,s / EPI_N,s % EPI_N));
    }
  }

  // THOR PATCH (bias): write BiasBits into this warpgroup's epilogue sub-tiles (part_idx of num_parts, as in accum) of accumulator
  // stage `state`: 8-column TMEM stores from an 8-register constant fragment. The caller issues fence_view_async_tmem_store()
  // before handing the stage back to the MMA.
  template <class TmemStorage, class AccumulatorPipelineState, class FrgEngine, class FrgLayout, class CopyOpT2R, class EpilogueTile>
  CUTLASS_DEVICE void
  bias_refill(TmemStorage tmem_storage, AccumulatorPipelineState const& state, cute::Tensor<FrgEngine, FrgLayout> const&,
              CopyOpT2R, EpilogueTile, int part_idx, int num_parts) {
    Tensor acc = get<0>(slice_accumulator(tmem_storage, state.index()));
    Tensor tAcc = acc(make_coord(_,_),_0{},_0{});
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});
#if !defined(G128_BIAS_ST_WIDTH)
#define G128_BIAS_ST_WIDTH 16
#endif
    using BiasStOp = cute::conditional_t<G128_BIAS_ST_WIDTH == 8, SM100_TMEM_STORE_32dp32b8x,
                     cute::conditional_t<G128_BIAS_ST_WIDTH == 16, SM100_TMEM_STORE_32dp32b16x, SM100_TMEM_STORE_32dp32b32x>>;
    auto tiled_st = make_tmem_copy(BiasStOp{}, tAcc_epi(_,_,_0{},_0{}));
    auto thr_st = tiled_st.get_slice(threadIdx.x % size(tiled_st));
    Tensor tST = thr_st.partition_D(tAcc_epi);                                               // TMEM side  (V_tmem,ST_M,ST_N,EPI_M,EPI_N)
    Tensor tRS = thr_st.partition_S(make_identity_tensor(shape(tAcc_epi)));                 // register side (V_reg,ST_M,ST_N,EPI_M,EPI_N)
    constexpr int EPI_N = decltype(size<3>(tAcc_epi))::value;
    constexpr int NSUB  = decltype(size<2>(tAcc_epi))::value * EPI_N;
    Tensor cst = make_tensor<ElementAccumulator>(shape(tRS(_,_0{},_0{},_0{},_0{})));
    // Opaque constants: with a plain immediate nvcc materialises a fresh register block for every tcgen05.st operand list and
    // hoists all of them out of the loop (16 stores x 8 = 128 live registers -> 1 KB of spills). One block, reused by all stores.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(cst); ++i) {
      int32_t v;
      asm volatile("mov.b32 %0, 0x4B400000;" : "=r"(v));
      cst(i) = ElementAccumulator(v);
    }
    // Not unrolled on purpose: ptxas gives every static tcgen05.st its own source register block (16 unrolled stores x 8 regs =
    // 128 registers -> 1 KB of spills, even with wait::st between them); one store instruction in a runtime loop uses one block.
    int const s_lo = NSUB * part_idx / num_parts, s_hi = NSUB * (part_idx + 1) / num_parts;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int s = s_lo; s < s_hi; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tST); ++m) {
        CUTLASS_PRAGMA_NO_UNROLL                       // also one static store instruction across the column groups
        for (int n = 0; n < size<2>(tST); ++n) copy(tiled_st, cst, tST(_,m,n,s / EPI_N,s % EPI_N));
      }
    }
  }

protected:

  typename Params::TMA_A const* observed_tma_load_a_{nullptr};
  typename Params::TMA_B const* observed_tma_load_b_{nullptr};

  RuntimeDataTypeA runtime_data_type_a_{};
  RuntimeDataTypeB runtime_data_type_b_{};

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
