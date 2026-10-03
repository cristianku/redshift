NVCC ?= /usr/local/cuda/bin/nvcc
JOBS ?= 20
NVFLAGS = -O3 -std=c++17 -arch=sm_70 --split-compile $(JOBS) -Xcompiler=-fPIC,-Wall,-Wextra -lineinfo

.PHONY: all test test-cuda memcheck
all: build/libqvelox.so
build/%.o: src/%.cu src/kernels.h
	@mkdir -p build
	$(NVCC) $(NVFLAGS) -c $< -o $@
build/libqvelox.so: build/kernels.o build/runtime.o
	$(NVCC) -shared $^ -o $@ -Xlinker -rpath -Xlinker /usr/local/cuda/lib64
test:
	python3 -m unittest discover -s tests -v
test-cuda: all
	QVELOX_CUDA=1 python3 -m unittest discover -s tests -v
memcheck: all
	QVELOX_CUDA=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 python3 -m unittest discover -s tests -p test_cuda.py -v
