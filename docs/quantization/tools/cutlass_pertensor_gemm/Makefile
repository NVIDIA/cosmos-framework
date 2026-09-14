# Builds one binary per (kind, config). Usage: make -j10 CUTLASS=~/cutlass [ARCH=100|101|110|90] [CONFIGS=...] [KINDS=...]
# ARCH=110 targets Thor (sm_110a); INT8 tcgen05 is enabled by CUTLASS on sm_100a/101a/110a (cute/arch/config.hpp), not on sm_103a (GB300).
# config name: TMxTNxTK_cCMxCN[_sk|_pp]   (_sk = stream-K tile scheduler, _pp = SM90 pingpong schedule)
.DEFAULT_GOAL := all
CUTLASS ?= $(HOME)/cutlass
NVCC    ?= nvcc
ARCH    ?= 100
OUT     ?= build_sm$(ARCH)
NVCCFLAGS = -std=c++17 -O3 -arch=sm_$(ARCH)a -DARCH=$(ARCH) --expt-relaxed-constexpr \
            -DCUTLASS_ENABLE_GDC_FOR_SM100=1 -DCUTLASS_ENABLE_GDC_FOR_SM90=1 \
            -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include -lcuda -Xcompiler -Wno-psabi -diag-suppress 177
ifeq ($(ARCH),90)
NVCCFLAGS += -DFP8_FAST_ACCUM
endif

CONFIGS ?= 256x128x128_c2x1 256x256x128_c2x1 128x128x128_c1x1 256x128x64_c2x2 128x256x128_c1x1
KINDS   ?= int8 fp8

# $(1)=config-without-_sk -> "TM TN TK CM CN"
cfg_parts = $(subst x, ,$(subst _c, ,$(1)))
# $(1)=kind $(2)=config
define build_rule
$(OUT)/pertensor_gemm_$(1)_$(2): pertensor_gemm.cu | $(OUT)
	$(NVCC) $(NVCCFLAGS) -DKIND_$(shell echo $(1) | tr a-z A-Z) $(if $(filter %_sk,$(2)),-DSTREAMK,) $(if $(filter %_pp,$(2)),-DPINGPONG,) \
	  -DTM=$(word 1,$(call cfg_parts,$(patsubst %_pp,%,$(patsubst %_sk,%,$(2))))) -DTN=$(word 2,$(call cfg_parts,$(patsubst %_pp,%,$(patsubst %_sk,%,$(2))))) \
	  -DTK=$(word 3,$(call cfg_parts,$(patsubst %_pp,%,$(patsubst %_sk,%,$(2))))) \
	  -DCM=$(word 4,$(call cfg_parts,$(patsubst %_pp,%,$(patsubst %_sk,%,$(2))))) -DCN=$(word 5,$(call cfg_parts,$(patsubst %_pp,%,$(patsubst %_sk,%,$(2))))) -o $$@ $$<
TARGETS += $(OUT)/pertensor_gemm_$(1)_$(2)
endef
$(foreach k,$(KINDS),$(foreach c,$(CONFIGS),$(eval $(call build_rule,$(k),$(c)))))

all: $(TARGETS)
$(OUT):
	mkdir -p $(OUT)
clean:
	rm -rf $(OUT)
.PHONY: all clean
