#!/usr/bin/env bash
set -euo pipefail
# Uses existing build products only; does not modify or rebuild llama.cpp.
source_dir=${1:?pass llama.cpp source directory}
build_dir=${2:?pass llama.cpp build directory}
cuda_dir=${3:-/usr/local/cuda}
mkdir -p build
c++ -O3 -std=c++17 -I"$source_dir/include" -I"$source_dir/ggml/include" \
  tools/llama_reference.cpp -o build/llama-reference \
  "$build_dir/src/libllama.a" "$build_dir/ggml/src/libggml.a" -ldl \
  "$build_dir/ggml/src/libggml-cpu.a" "$build_dir/ggml/src/ggml-cuda/libggml-cuda.a" \
  "$build_dir/ggml/src/libggml-base.a" -fopenmp -lpthread -lm \
  -L"$cuda_dir/lib64" -Wl,-rpath,"$cuda_dir/lib64" -lcudart -lcublas -lcublasLt -lculibos \
  -L"$cuda_dir/lib64/stubs" -lcuda
