# QVelox initial core implementation plan

> Execution: Codex implements directly using executing-plans and test-driven-development. Work stays on main, without commits or push.

## Goal and approved design

Implement the README first benchmark milestone: an independent Volta/Qwen dense numeric runtime that evaluates every vocabulary row for groups of 1, 2, 4 and 8 input tokens. Reuse the existing Q4_K_M model on llama-v100-exp; no model downloads, no power changes. This milestone measures verification capacity, not accepted speculative tokens or a promised 100 generated tokens/s.

## Architecture

Python standard-library GGUF header reader validates model layout without mapping the weights. A narrow C ABI owns an independent CUDA runtime: bounded uploads, quantized projections, recurrent DeltaNet, causal attention and all-row output. Python coordinates tests and measurements. No runtime dependency on DS4 or llama.cpp. The first interface accepts token IDs; tokenizer/chat integration and MTP follow after the measured feasibility gate.

## Constraints

- Test host: llama-v100-exp, 10.10.10.55; Tesla V100 32 GB, CUDA 12.9, 20 build jobs.
- Existing model: /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf.
- qwen35 GGUF layout, 64 blocks, embedding 5120, FFN 17408, attention 24/4 heads with dim 256, DeltaNet 16/48 heads with dim 128, vocabulary 248320.
- Keep GPU weights quantized; upload with a reusable 16 MiB buffer. Never register/map the complete model into pinned host memory.
- Correctness precedes performance. Every proposed row includes its output projection. Report batch throughput separately from generation.
- The repository declares MIT; see LICENSE and THIRD_PARTY.md for provenance.

## Review focus

Truncated or overlapping GGUF tensors; unsupported model metadata; non-finite or incorrectly decoded quantized values; nonzero-prefix causality and recurrent state; optimistic timing that omits output rows or state restoration.

## Tasks

- [x] 1. GGUF inspection and validated model manifest. Files: qvelox/gguf.py, qvelox/model.py, tests/test_gguf.py. Test synthetic headers, bounds and layout rejection. Interface: read_gguf(path), validate_qwen27b(model).
- [x] 2. Independent CUDA kernels and narrow ABI. Files: src/kernels.cu, src/runtime.cu, src/runtime.h, qvelox/runtime.py, tests/test_cuda.py, Makefile. Compare Q4_K/Q6_K/Q8_0, RMS, attention and recurrent state with independent CPU arithmetic; run compute-sanitizer on small fixtures.
- [x] 3. Complete model graph and group verification. Files: same runtime plus qvelox/__main__.py, tests/test_model.py. Compare every batch logit row with sequential evaluation at the same prefix, including state after a rejected suffix and a continuation.
- [x] 4. Reproducible 1/2/4/8 measurements and report. Use repeated alternating measurements on the same context and weights. Save raw JSON with source hashes, settings, GPU properties and clear limitations. Update README without claiming MTP or chatbot completion.

## Progress

- Initial repository inspected: clean main, user-written README preserved.
- Test host verified: GPU idle, existing main and MTP weights available, CUDA 12.9, 20 CPU cores. No downloads needed.
- 2026-10-03: numeric proof of concept implemented and tested in a new temporary
  directory on the explicitly authorized experimental host. Work continued on main.
- Ruling: the existing kernel/runtime split is retained instead of introducing
  src/qvelox.cu; Python owns manifest validation, native code owns allocations,
  bounded uploads, forward execution and complete checkpoints.
- Ruling: checkpoints copy recurrent/convolution state and the valid KV prefix;
  accepted-suffix handling uses restore and replay. This establishes correctness
  before optimizing checkpoint or acceptance costs.
- Ruling: timing includes all logits and state restoration; supplied-token
  throughput is not reported as speculative generation throughput.
- Results, reproduction commands and limitations: [poc-results.md](poc-results.md).
