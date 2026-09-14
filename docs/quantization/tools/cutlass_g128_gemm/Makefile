# g128 (group-wise 1x128 / SFNx128 scaled) 8-bit GEMM binaries, one per (kind, config).
# config name: TMxTNxTK_cCMxCN_sfnS   e.g. 256x128x128_c2x1_sfn1
.DEFAULT_GOAL := all
CUTLASS ?= $(HOME)/cutlass
NVCC    ?= nvcc
ARCH    ?= 100      # Blackwell family: 100 (GB200/B200), 101, 110 (Thor). INT8 tcgen05 is not available on 103 (GB300).
OUT     ?= build
NVCCFLAGS = $(EXTRA) -std=c++17 -O3 -arch=sm_$(ARCH)a --expt-relaxed-constexpr -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
            -I./include -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include -lcuda -Xcompiler -Wno-psabi -diag-suppress 177
CONFIGS ?= 256x128x128_c2x1_sfn1 256x128x128_c2x1_sfn128
KINDS   ?= fp8
cfg_parts = $(subst x, ,$(subst _sfn, ,$(subst _c, ,$(1))))
define build_rule
$(OUT)/g128_gemm_$(1)_$(2): g128_gemm.cu $(wildcard include/*.hpp include/cutlass/gemm/collective/*.hpp include/cutlass/gemm/collective/builders/*.inl) | $(OUT)
	$(NVCC) $(NVCCFLAGS) -DKIND_$(shell echo $(1) | tr a-z A-Z) \
	  -DTM=$(word 1,$(call cfg_parts,$(2))) -DTN=$(word 2,$(call cfg_parts,$(2))) -DTK=$(word 3,$(call cfg_parts,$(2))) \
	  -DCM=$(word 4,$(call cfg_parts,$(2))) -DCN=$(word 5,$(call cfg_parts,$(2))) -DSFN=$(word 6,$(call cfg_parts,$(2))) -o $$@ $$<
TARGETS += $(OUT)/g128_gemm_$(1)_$(2)
endef
$(foreach k,$(KINDS),$(foreach c,$(CONFIGS),$(eval $(call build_rule,$(k),$(c)))))
all: $(TARGETS)
$(OUT):
	mkdir -p $(OUT)
clean:
	rm -rf $(OUT)
.PHONY: all clean
