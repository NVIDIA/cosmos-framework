"""Generate the INT8 (1,128,128)-blockwise SM90 GEMM sources by porting CUTLASS's FP8 blockwise collective. Re-run after editing."""
import os, re
D=os.path.dirname(os.path.abspath(__file__))
C=os.path.join(os.environ['CUTLASS_DIR'], 'include', 'cutlass')  # CUTLASS >= 4.2 checkout
lines=open(f'{C}/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp').read().split('\n')
out=[]; removed=0; skip_next=False; inc_done=False
for l in lines:
    if skip_next: skip_next=False; removed+=1; continue
    if 'is_same_v<ElementAccumulator, ElementBlockScale>' in l:
        out.append('  // (INT8 port) ElementAccumulator(int32) != ElementBlockScale(float) by design'); skip_next=True; removed+=1; continue
    out.append(l)
    if (not inc_done) and l.strip().startswith('#include "cutlass/gemm/dispatch_policy.hpp"'):
        out.append('#include "int8_accumulation.cuh"  // (INT8 port)'); inc_done=True
assert inc_done and removed==2, (inc_done, removed)
src='\n'.join(out)
n0=src.count('MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8'); assert n0>=2
src=src.replace('MainloopSm90TmaGmmaWarpSpecializedBlockwiseFP8','MainloopSm90TmaGmmaWarpSpecializedBlockwiseInt8')
src,n1=re.subn(r'using ElementBlockScale\s*=\s*ElementAccumulator;','using ElementBlockScale = float;  // INT8 port: fp32 block scales, int32 MMA accumulator',src); assert n1==1,n1
n2=src.count('GmmaFP8Accumulation'); assert n2>=3
src=src.replace('GmmaFP8Accumulation','GmmaInt8Accumulation')
open(f'{D}/sm90_mma_int8_blockwise.cuh','w').write(src)
print(f"collective: policy renames={n0}, assert lines removed={removed}, ElementBlockScale fixed={n1}, accumulation renames={n2}")

open(f'{D}/int8_accumulation.cuh','w').write(r'''// INT8 variant of cutlass::gemm::collective::GmmaFP8Accumulation.
// wgmma s8*s8 accumulates in int32; each 128-K block the int32 partials are converted to fp32 (exact: |sum| <= 128*127*127 < 2^24)
// and multiply-added with the fp32 block scales into the fp32 main accumulator. The main accumulator shares registers with the kernel's
// int32 fragment (reinterpreted), so the epilogue must bitcast (Sm90AccFetchBitcastF32).
#pragma once
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
namespace cutlass::gemm::collective {
template <class EngineAccum, class LayoutAccum>
struct GmmaInt8Accumulation {
  using TensorAccum = cute::Tensor<EngineAccum, LayoutAccum>;
  using ElementAccumulator = typename EngineAccum::value_type;
  static_assert(cute::is_same_v<ElementAccumulator, int32_t>, "GmmaInt8Accumulation expects an int32 MMA accumulator");
  static_assert(is_static<LayoutAccum>::value, "Accumulator Layout should be static");
  static_assert(is_rmem<TensorAccum>::value, "Accumulator tensor must be rmem resident.");
 private:
  TensorAccum& accum_;        // kernel fragment (int32 storage) holding the fp32 main accumulator bits
  TensorAccum accum_temp_;    // int32 partial accumulator fed to wgmma
  uint32_t accum_promotion_interval_, mma_count_per_mainloop_iteration_, mma_count_, reset_accum_flag_;
  CUTLASS_DEVICE float& main(int i) { return reinterpret_cast<float&>(accum_(i)); }
  template <class EA, class LA, class EB, class LB>
  CUTLASS_DEVICE void scale_core(cute::Tensor<EA, LA> const& sA, cute::Tensor<EB, LB> const& sB) {
    static_assert(LayoutAccum{}.shape() == LA{}.shape(), "Accumulator and scaleA must have same shape.");
    static_assert(LayoutAccum{}.shape() == LB{}.shape(), "Accumulator and scaleB must have same shape.");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) main(i) += static_cast<float>(accum_temp_(i)) * (sA(i) * sB(i));
  }
  template <class E, class L>
  CUTLASS_DEVICE void scale_core(cute::Tensor<E, L> const& s) {
    static_assert(LayoutAccum{}.shape() == L{}.shape(), "Accumulator and scale must have same shape.");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) main(i) += static_cast<float>(accum_temp_(i)) * s(i);
  }
  CUTLASS_DEVICE void scale_core(float const& s) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) main(i) += static_cast<float>(accum_temp_(i)) * s;
  }
 public:
  CUTLASS_DEVICE GmmaInt8Accumulation(TensorAccum& accum, uint32_t accum_promotion_interval, uint32_t mma_count_per_mainloop_iteration)
      : accum_(accum), accum_promotion_interval_(accum_promotion_interval), mma_count_per_mainloop_iteration_(mma_count_per_mainloop_iteration),
        mma_count_(0), reset_accum_flag_(0) {
    accum_temp_ = cute::make_fragment_like(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) main(i) = 0.f;
  }
  CUTLASS_DEVICE TensorAccum& operator()() { return accum_temp_; }
  CUTLASS_DEVICE bool prepare_if_needed() { return reset_accum_flag_; }
  template <class... S> CUTLASS_DEVICE void scale_if_needed(S const&... s) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) { scale_core(s...); mma_count_ = 0; }
  }
  template <class... S> CUTLASS_DEVICE void scale(S const&... s) { scale_core(s...); }
  template <class... S> CUTLASS_DEVICE void scale_residue_if_needed(S const&... s) { if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) scale_core(s...); }
};
}  // namespace cutlass::gemm::collective
''')

open(f'{D}/int8_blockwise_gemm_sm90.cuh','w').write(r'''// CUTLASS 3.x / SM90 INT8 GEMM with DeepSeek-style (1,128,128) block scaling inside the mainloop.
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
''')

open(f'{D}/cfg_0.cu','w').write('''#include "int8_blockwise_gemm_sm90.cuh"
using G0 = int8bw::Int8BlockwiseGemm<128, 128, 128, 1, 2>;
size_t int8bw_ws_0(int M, int N, int K) { return G0::workspace_size(M, N, K); }
void int8bw_run_0(const int8_t* A, const int8_t* B, const float* sfa, const float* sfb, void* D, int M, int N, int K, void* ws, cudaStream_t s) {
  G0::run(A, B, sfa, sfb, reinterpret_cast<cutlass::bfloat16_t*>(D), M, N, K, ws, s);
}
int int8bw_stages_0() { return G0::Stages; }
''')
open(f'{D}/bindings.cpp','w').write('''#include <torch/extension.h>
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
''')
open(f'{D}/__init__.py','w').write('''"""CUTLASS SM90 INT8 GEMM with (1,128,128) block scaling in the mainloop (torch cpp_extension build).
int8_blockwise_mm(A[M,K] int8, W[N,K] int8, sfa[K//128, M] fp32, sfb[N//128, K//128] fp32) -> bf16 [M,N]; pads M to a multiple of 4."""
import os, glob, torch
_HERE = os.path.dirname(os.path.abspath(__file__)); _BENCH = os.path.dirname(os.path.dirname(_HERE))
CUTLASS = os.environ.get("CUTLASS_DIR", os.path.join(_BENCH, "third_party", "DeepGEMM", "third-party", "cutlass"))
BUILD_DIR = os.environ.get("INT8BW_BUILD_DIR", os.path.join(_BENCH, "third_party", "cutlass_int8_bw_build"))
_ext = None
def load(verbose=False):
    global _ext
    if _ext is not None: return _ext
    from torch.utils.cpp_extension import load as _load
    os.makedirs(BUILD_DIR, exist_ok=True); os.environ.setdefault("MAX_JOBS", "8"); os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    _ext = _load(name="int8sm90_bw_ext", sources=[os.path.join(_HERE, "bindings.cpp")] + sorted(glob.glob(os.path.join(_HERE, "cfg_*.cu"))),
                 build_directory=BUILD_DIR, verbose=verbose,
                 extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(CUTLASS, "tools", "util", "include"), _HERE],
                 extra_cflags=["-O3", "-std=c++17"],
                 extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG",
                                    "-gencode", "arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xcompiler", "-Wno-psabi", "-Xptxas", "-v"])
    return _ext
def int8_blockwise_mm(A, W, sfa, sfb):
    """sfa: [K//128, M] (scale of row m, k-block kb at sfa[kb, m]); sfb: [N//128, K//128]."""
    ext = load(); M = A.shape[0]; Mp = (M + 3) // 4 * 4
    if Mp != M:
        A = torch.cat([A, A.new_zeros(Mp - M, A.shape[1])], 0); sfa = torch.cat([sfa, sfa.new_zeros(sfa.shape[0], Mp - M)], 1)
    out = ext.int8_blockwise_mm(A, W, sfa.contiguous(), sfb.contiguous())
    return out[:M] if Mp != M else out
''')
print("files:", sorted(os.listdir(D)))
