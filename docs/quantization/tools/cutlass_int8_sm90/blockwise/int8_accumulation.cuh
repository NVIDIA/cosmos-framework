// INT8 variant of cutlass::gemm::collective::GmmaFP8Accumulation.
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
