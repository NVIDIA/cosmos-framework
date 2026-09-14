# CUTLASS per-tensor FP8 / INT8 GEMM benchmarks for Thor (sm_110a). Needs a CUTLASS >= 4.8 checkout (headers only) and nvcc 13.x.
#   make -j14 CUTLASS=/path/to/cutlass
CUTLASS ?= /home/pzeren/thor/cutlass
ARCH    ?= sm_110a
NVCC    ?= nvcc
BUILD   ?= build
# G128_INC: shadow CUTLASS header with the INT8 blockwise collective patch (fp32 scales / fp32 promoted accumulator for integer MMA
# accumulators, from ../cutlass_g128_gemm, GB200 work). It must precede the CUTLASS include path. G128_OPT: its optimisation flags.
G128_INC ?= ./include
G128_OPT ?= -DG128_OPT_PROMOTION -DG128_OPT_PIPE2 -DG128_OPT_SBVEC
# G128_OPT_BIAS: TMEM pre-biased promotion (per-col INT8, no I2F). Correct (verify PASS) but the tcgen05.st re-arm pins registers
# in the promotion warps and spills (~0.8 KB), so it is 2-3x SLOWER as of 2026-09-14; see README "TMEM pre-bias: status". Off by default.
NVFLAGS  = -O3 -std=c++17 -arch=$(ARCH) -I$(G128_INC) -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include \
           --expt-relaxed-constexpr -DNDEBUG -Xcompiler -Wno-psabi -diag-suppress 177,20012
BWFLAGS  = $(NVFLAGS) $(G128_OPT)

CFG_SRCS := $(wildcard cfg_*.cu) $(wildcard rccfg_*.cu)
CFG_OBJS := $(patsubst %.cu,$(BUILD)/%.o,$(CFG_SRCS))
BWCFG_SRCS := $(wildcard bwcfg_*.cu)
BWCFG_OBJS := $(patsubst %.cu,$(BUILD)/%.o,$(BWCFG_SRCS))
BW_DEFS := $(if $(wildcard bwcfg_int8_0.cu),-DTHOR_BW_HAVE_INT8,)

all: pertensor_gemm blockwise_gemm cublaslt_bench $(BUILD)/membw

blockwise_gemm: $(BUILD)/blockwise_gemm.o $(BWCFG_OBJS)
	$(NVCC) -arch=$(ARCH) -o $@ $^

$(BUILD)/bwcfg_%.o: bwcfg_%.cu blockwise_gemm.cuh bw_configs.h bw_cfg_select.h $(wildcard $(G128_INC)/cutlass/gemm/collective/*.hpp) $(wildcard $(G128_INC)/cutlass/gemm/kernel/*.hpp) $(wildcard $(G128_INC)/cutlass/arch/*.h) | $(BUILD)
	$(NVCC) $(BWFLAGS) -c $< -o $@

$(BUILD)/blockwise_gemm.o: blockwise_gemm.cu blockwise_gemm.cuh bw_configs.h bench_util.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) $(BW_DEFS) -c $< -o $@

pertensor_gemm: $(BUILD)/pertensor_gemm.o $(CFG_OBJS)
	$(NVCC) -arch=$(ARCH) -o $@ $^

$(BUILD)/cfg_%.o: cfg_%.cu pertensor_gemm.cuh configs.h cfg_select.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/rccfg_%.o: rccfg_%.cu pertensor_gemm.cuh configs.h cfg_select.h | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/pertensor_gemm.o: pertensor_gemm.cu pertensor_gemm.cuh configs.h bench_util.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/membw: membw.cu bench_util.cuh | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -o $@ $<

cublaslt_bench: cublaslt_bench.cu bench_util.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -o $@ $< -lcublasLt -lcublas

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD) pertensor_gemm blockwise_gemm cublaslt_bench

.PHONY: all clean
