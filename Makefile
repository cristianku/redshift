CUDA_DIR ?= /usr/local/cuda
NVCC ?= $(CUDA_DIR)/bin/nvcc
JOBS ?= 20
NVFLAGS = -O3 -std=c++17 -arch=sm_70 --split-compile $(JOBS) -Xcompiler=-fPIC,-Wall,-Wextra -lineinfo

.PHONY: all test test-cuda memcheck racecheck bench-mm reference
all: build/libqvelox.so
build/%.o: src/%.cu src/kernels.h src/runtime.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -c $< -o $@
build/libqvelox.so: build/kernels.o build/runtime.o
	$(NVCC) -shared $^ -o $@ -Xlinker -rpath -Xlinker $(CUDA_DIR)/lib64
build/runtime-reference.o: src/runtime.cu src/kernels.h src/runtime.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -DQVELOX_REFERENCE_PROJECTIONS -c $< -o $@
build/libqvelox-reference.so: build/kernels.o build/runtime-reference.o
	$(NVCC) -shared $^ -o $@ -Xlinker -rpath -Xlinker $(CUDA_DIR)/lib64
reference: build/libqvelox-reference.so
test:
	python3 -m unittest discover -s tests -v
test-cuda: all
	QVELOX_CUDA=1 python3 -m unittest discover -s tests -v
memcheck: all
	QVELOX_CUDA=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 python3 -m unittest discover -s tests -p test_cuda.py -v
racecheck: all
	QVELOX_CUDA=1 /usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 1 python3 -m unittest discover -s tests -p test_cuda.py -v

build/bench-mm: tools/bench_mm.cu build/kernels.o src/kernels.h
	$(NVCC) $(NVFLAGS) -I src $< build/kernels.o -o $@
bench-mm: build/bench-mm
	build/bench-mm
