# Builds the CUDA side of the shootout. Ampere / RTX 3090 = sm_86.
#
#   make            # build all three CUDA binaries into build/
#   make clean
NVCC    ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH    ?= sm_86
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
