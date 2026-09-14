# CUTLASS per-tensor FP8 / INT8 GEMM benchmarks for Thor (sm_110a). Needs a CUTLASS >= 4.8 checkout (headers only) and nvcc 13.x.
#   make -j14 CUTLASS=/path/to/cutlass
CUTLASS ?= /home/pzeren/thor/cutlass
ARCH    ?= sm_110a
NVCC    ?= nvcc
BUILD   ?= build
NVFLAGS  = -O3 -std=c++17 -arch=$(ARCH) -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include \
           --expt-relaxed-constexpr -DNDEBUG -Xcompiler -Wno-psabi

CFG_SRCS := $(wildcard cfg_*.cu)
CFG_OBJS := $(patsubst %.cu,$(BUILD)/%.o,$(CFG_SRCS))

all: pertensor_gemm cublaslt_bench $(BUILD)/membw

pertensor_gemm: $(BUILD)/pertensor_gemm.o $(CFG_OBJS)
	$(NVCC) -arch=$(ARCH) -o $@ $^

$(BUILD)/cfg_%.o: cfg_%.cu pertensor_gemm.cuh configs.h cfg_select.h | $(BUILD)
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
	rm -rf $(BUILD) pertensor_gemm cublaslt_bench

.PHONY: all clean
