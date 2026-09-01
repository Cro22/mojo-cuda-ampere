# Builds the C++ side of the shootout, on either NVIDIA (nvcc/CUDA) or AMD
# (hipcc/ROCm). The platform is auto-detected from which toolchain sees a GPU,
# exactly the way the NVIDIA arch was already auto-detected — you never name the
# card by hand.
#
#   make              # build all three binaries into build/ for the GPU present
#   make ARCH=sm_75   # NVIDIA: force a target arch (overrides auto-detection)
#   make GFX=gfx90a   # AMD: force an offload arch (overrides auto-detection)
#   make hipify       # AMD: (re)generate the HIP sources from the .cu (see below)
#   make clean
#
# --- platform detection ----------------------------------------------------
# If rocminfo reports a gfx target we build the ROCm path; otherwise CUDA. GFX is
# overridable on the command line, and empty on an NVIDIA box, which selects the
# else branch below. This mirrors the CUDA compute_cap probe: the toolchain asks
# the GPU what it is and compiles for that.
GFX ?= $(shell rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')

ROOT    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD   := $(ROOT)/bench/build
BINS    := $(BUILD)/reduction $(BUILD)/softmax $(BUILD)/matmul

all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

ifneq ($(GFX),)
# ======================= AMD / ROCm / HIP path =============================
# UNTESTED until run on an AMD box: hipcc, --offload-arch from rocminfo, rocBLAS
# for the matmul vendor ref, hipCUB for the reduction vendor ref. The kernel
# sources are HIPified copies of the .cu (see the `hipify` target). We tag the
# CSV `impl` column "hip" via -DBENCH_IMPL so these rows don't collide with CUDA.
HIPCC    ?= hipcc
ROCM     ?= /opt/rocm
HIPFLAGS := -O3 -std=c++20 --offload-arch=$(GFX) -I$(ROCM)/include -DBENCH_IMPL=\"hip\"

# HIPified sources live under kernels/<k>/hip/, generated from the pristine .cu
# by hipify-perl. They are build artifacts (gitignored), regenerated on the box.
HIP_HDR  := $(ROOT)/bench/bench.hip.h
HIP_SRC   = $(ROOT)/kernels/$1/hip/$1.hip.cpp

hipify: $(HIP_HDR) \
        $(ROOT)/kernels/reduction/hip/reduction.hip.cpp \
        $(ROOT)/kernels/softmax/hip/softmax.hip.cpp \
        $(ROOT)/kernels/matmul/hip/matmul.hip.cpp
	@echo ">> HIP sources generated (review warp intrinsics + vendor calls; see docs/amd-port.md)"

$(HIP_HDR): $(ROOT)/bench/bench.cuh
	hipify-perl $< > $@

# Generate a HIP source from each .cu and repoint its local include at the
# HIPified header (hipify-perl rewrites <cuda_runtime.h> but not "bench.cuh").
$(ROOT)/kernels/%/hip/%.hip.cpp: $(ROOT)/kernels/%/cuda/%.cu $(HIP_HDR)
	mkdir -p $(dir $@)
	hipify-perl $< | sed 's#bench\.cuh#bench.hip.h#' > $@

$(BUILD)/reduction: $(ROOT)/kernels/reduction/hip/reduction.hip.cpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -o $@ $<          # hipCUB is header-only under $(ROCM)/include
$(BUILD)/softmax: $(ROOT)/kernels/softmax/hip/softmax.hip.cpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -o $@ $<
$(BUILD)/matmul: $(ROOT)/kernels/matmul/hip/matmul.hip.cpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -lrocblas -o $@ $<

else
# ======================= NVIDIA / CUDA path (unchanged) ====================
NVCC    ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# Target arch is auto-detected from the compute capability of the GPU present,
# so the same Makefile builds and RUNS on any card: RTX 3090 -> sm_86, T4 ->
# sm_75. Hardcoding sm_86 does not just run slow on a T4, it fails to launch.
# `compute_cap` is a query (works under WSL); override with `make ARCH=sm_80`.
# Falls back to sm_86 if nvidia-smi is unavailable (e.g. building GPU-less).
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH    ?= sm_$(if $(DETECTED_CC),$(DETECTED_CC),86)
NVFLAGS := -O3 -std=c++20 -arch=$(ARCH)

# CUDA 13 ships CUB/Thrust under .../include/cccl, which is not on nvcc's default
# search path. Point the reduction build at it so <cub/...> resolves.
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(realpath $(NVCC)))
CCCL_INC  := $(CUDA_HOME)/targets/x86_64-linux/include/cccl

$(BUILD)/reduction: $(ROOT)/kernels/reduction/cuda/reduction.cu | $(BUILD)
	$(NVCC) $(NVFLAGS) -I$(CCCL_INC) -o $@ $<

$(BUILD)/softmax: $(ROOT)/kernels/softmax/cuda/softmax.cu | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $<

$(BUILD)/matmul: $(ROOT)/kernels/matmul/cuda/matmul.cu | $(BUILD)
	$(NVCC) $(NVFLAGS) -lcublas -o $@ $<
endif

clean:
	rm -rf $(BUILD)

.PHONY: all clean hipify
