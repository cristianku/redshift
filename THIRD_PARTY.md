# Source provenance

QVelox is MIT licensed. The complete license and retained copyright notices are in LICENSE.

`src/kernels.cu` adapts the Q4_K projection from the experimental `misc/qwen27b-probe/native.cu`, and the Q6_K projection, depthwise convolution, Gated DeltaNet normalization and recurrence from `ds4_qwen4_cuda.cuh` in Cristian's DS4 checkout (base commit `63383b7c3f8347a0e3be2eca4d580b3f3d98244c`). The experimental file was untracked in that checkout. Their implementation descends from DS4/GGML; the original notices are preserved.

Upstream: https://github.com/antirez/ds4 (MIT). GGUF quantization layouts originate in GGML: https://github.com/ggml-org/ggml (MIT).

QVelox has no link-time or runtime dependency on DS4 or llama.cpp. The independent loader, model orchestration, CUDA memory ownership, attention kernel and benchmark harness live in this repository.
