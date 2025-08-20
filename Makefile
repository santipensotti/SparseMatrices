# Makefile for SparseMatrices project
# Usage:
#   make            # build bin/ejemplo
#   make clean      # remove build artifacts

NVCC ?= nvcc
# Adjust Eigen include if needed (e.g., /opt/homebrew/include/eigen3 on Apple Silicon)
EIGEN_INC ?= /usr/include/eigen3
# Add project root so #include "helpers/mtxToCuda.h" resolves
INC := -I$(EIGEN_INC) -I. -I./helpers

# Try common CUDA lib locations; harmless if not present
CUDA_LIB_DIRS ?= /usr/local/cuda/lib64 /usr/local/cuda/lib
LDFLAGS := $(addprefix -L,$(CUDA_LIB_DIRS)) -lcusparse -lcudart

CXXSTD ?= c++17
CXXFLAGS := -O3 -std=$(CXXSTD) -w -Xcompiler "-fopenmp"

SRC := src/cusparseSpMV.cu
BIN_DIR := bin
OUT := $(BIN_DIR)/ejemplo

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) helpers/mtxToCuda.h | $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(INC) $< -o $@ $(LDFLAGS)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
