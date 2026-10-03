# Proposed Benchmark Plan

This is a documented plan, **not an executed experiment**. The first objective
is a baseline for the actual checkout: previous Redshift reports do not match
all current sources. Any future server activity remains subject to Cristian's
authorizations and operational constraints.

For the next priority—turning verification capacity into actual generation
speed—see [04 — From verification capacity to generation speed](04-verification-to-generation.md).
That follow-up starts with controlled acceptance from 0 to 8 inputs and records
the locally prepared replay/transition comparison before introducing a proposer.

## 1. Identify the environment

Record GPU model/variant/VRAM, driver, toolkit, loaded libraries, clocks,
temperatures, power, and throttling reasons during measurement. Record source,
binary, and GGUF hashes, numerical settings, KV cache configuration, and the
exact model.

Example read-only checks in the available measurement environment:

```sh
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvcc --version
nsys --version
ncu --version
ldd ./ds4-bench
```

Do not change drivers, toolkits, or services to collect this information.

## 2. Updated baseline and profile

Measure single-token decode, groups of 2/4/8, larger prefill, and short/long
contexts that fit available VRAM separately. Use the same GGUF, inputs, KV budget,
and requested output. Redshift full-vocabulary comparisons must retain copies
of all logits in the optimized candidate as well.

Warm up, run at least five alternating A/B repetitions, and save raw samples,
medians, and dispersion. Increase repetitions when the margin is comparable to
noise. Measure loading/repacking separately and include it for frequent-start
use cases. Measure peak VRAM, TTFT, latency/token, and workload-specific throughput.

Use Nsight Systems to observe kernels, copies, synchronization, and timeline gaps.
Use Nsight Compute for kernel counters: bandwidth, registers/spills, occupancy,
stalls, and execution-unit utilization. Profiled measurements are diagnostic;
measure final performance without the profiler as well.

**Profiler version matters:** current Nsight Compute documentation lists Volta
as unsupported, while NVIDIA documents GV100 support in archived version 2025.1.
Check profiler/driver/toolkit compatibility before use rather than assuming the
latest version can profile V100.
[Current support](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/gpu-support.html),
[2025.1 requirements](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_1).
If counters are unavailable, start with timelines, CUDA events, and compiler
statistics, and state which information is missing.

## 3. Ordered experiments

| Experiment | Question | Required evidence |
|---|---|---|
| E0 — current baseline | Which kernels dominate today? | Updated timeline, hashes, timings by family |
| E1 — DP4A layout/geometry | Can small groups close the gap? | Quantize + matmul + reduce included; bandwidth, registers, VRAM |
| E2 — graphs on/off / fusion | How much does submission cost? | Total time and CPU/GPU gaps, not just kernel duration |
| E3 — DP4A vs FP16 Tensor Cores | At which shape/N is switching beneficial? | Conversion/dequantization included, precision and memory |
| E4 — Volta attention | Does attention dominate at long contexts? | Latency by length, logits and state, no shared-memory overflow |
| E5 — 12.x toolchains | Does identical source produce different machine code? | Equivalent builds, actual kernels, complete benchmark |

For E1, use actual model shapes and Q4/Q6/Q8 quantization; vary rows/tokens per
thread, split-K, and warp counts in a controlled way. For E3, begin with an
isolated cuBLAS FP16 comparison before writing a custom kernel.
Do not reopen the failed Redshift WMMA experiment without explaining what
changes: a wider shape, a different pipeline, or a lower correction cost.

Two existing DS4 controls allow comparisons without new APIs:

- `DS4_CUDA_DECODE_GRAPHS=0`: disables decode graphs.
- `DS4_CUDA_MMQ=0`: disables MMQ; do not apply it indiscriminately to MXFP4 models,
  for which the code reports no dequantization + cuBLAS fallback.

Do not assume the same controls exist in Redshift.

## 4. Numerical validation

Compare complete logits: RMSE, maximum error, cosine similarity, and argmax.
Matching argmax on a few synthetic tokens is insufficient. Retain the available
FP32 oracle and check every group size from 1 to 8, partial tiles, zeros, outliers,
and long contexts. Do not relax tolerances to make a candidate pass.

Check grouped/sequential execution, checkpoint restore, and rejected suffixes
where applicable. Different precision also requires real-prompt and quality/sampling
checks. A speculative loop must verify acceptance distribution and state after
commit. Run memcheck/racecheck/synccheck on modified kernels with a compatible
sanitizer version.

## 5. What qualifies as an improvement

Accept a candidate only when the benefit is repeatable in the target workload,
exceeds variability, and has explicit VRAM and precision costs. A microbenchmark
advantage must survive the complete forward pass. Do not silently degrade decode
to improve prefill; retain separate dispatch paths when measurements justify them.

For speculation:

```text
generated throughput = tokens actually emitted /
                       (draft + verify + accept/commit time + other overhead)
```

140 verified tokens/s does not automatically mean 140 generated tokens/s.
Measure accepted token counts and draft/commit costs.

An approximate bound for a resident dense model is:

```text
minimum weight time ≈ actual weight bytes read / effective HBM bandwidth
```

This is a model to compare with counters, not a throughput prediction.
For MoE, use bytes for the experts actually activated; include state, KV,
activations, and extra passes. Keep GB/GiB units consistent and use measured bandwidth.

The deliverable of this exploration is the plan and documentation.
Optimizations and GPU measurements are subsequent activities; no new speedup
was demonstrated in this session.
