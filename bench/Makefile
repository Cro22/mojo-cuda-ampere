# Builds the CUDA side of the shootout.
#
#   make            # build all three CUDA binaries into build/
#   make ARCH=sm_75 # force a target arch (overrides auto-detection)
#   make clean
NVCC    ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)

# Target arch is auto-detected from the compute capability of the GPU present,
# so the same Makefile builds and RUNS on any card: RTX 3090 -> sm_86, T4 ->
# sm_75. Hardcoding sm_86 does not just run slow on a T4, it fails to launch.
# `compute_cap` is a query (works under WSL); override with `make ARCH=sm_80`.
# Falls back to sm_86 if nvidia-smi is unavailable (e.g. building GPU-less).
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH    ?= sm_$(if $(DETECTED_CC),$(DETECTED_CC),86)
NVFLAGS := -O3 -std=c++20 -arch=$(ARCH)
ROOT    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..
BUILD   := $(ROOT)/bench/build

# CUDA 13 ships CUB/Thrust under .../include/cccl, which is not on nvcc's default
# search path. Point the reduction build at it so <cub/...> resolves.
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(realpath $(NVCC)))
CCCL_INC  := $(CUDA_HOME)/targets/x86_64-linux/include/cccl

BINS := $(BUILD)/reduction $(BUILD)/softmax $(BUILD)/matmul

all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/reduction: $(ROOT)/kernels/reduction/cuda/reduction.cu | $(BUILD)
	$(NVCC) $(NVFLAGS) -I$(CCCL_INC) -o $@ $<

$(BUILD)/softmax: $(ROOT)/kernels/softmax/cuda/softmax.cu | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $<

$(BUILD)/matmul: $(ROOT)/kernels/matmul/cuda/matmul.cu | $(BUILD)
	$(NVCC) $(NVFLAGS) -lcublas -o $@ $<

clean:
	rm -rf $(BUILD)

.PHONY: all clean
