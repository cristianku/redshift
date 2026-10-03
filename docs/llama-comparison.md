# Redshift vs llama.cpp on the V100 — 2026-10-03

This report records the **initial FP32-activation comparison**. See the subsequent
[projection optimization report](kernel-optimization.md) for the current
140.7 tokens/s eight-candidate result, FP32 fidelity checks, and fresh llama.cpp timings.

The proof of concept is slower than the existing llama.cpp engine on the same
V100 and checkpoint. Its improvement over its own single-token path is not an
advantage over llama.cpp. No generated-token throughput from speculative
acceptance has been demonstrated by this comparison.

## Comparable full-vocabulary measurements

Both engines receive identical synthetic token IDs at the same saved prefix.
Each timed call restores its checkpoint on the GPU, executes the full graph,
and returns all 248,320 logits for every candidate. We measure five repetitions
in alternating group order, after warmup. Engines run separately on the same
physical card, with FP16 K/V caches and no MTP/draft model.

| Prefix | Group | Redshift verified tokens/s | llama.cpp verified tokens/s | llama.cpp / Redshift |
|---|---:|---:|---:|---:|
| 16 | 1 | 19.69 | 27.72 | 1.41x |
| 16 | 2 | 30.30 | 51.34 | 1.69x |
| 16 | 4 | 43.14 | 84.58 | 1.96x |
| 16 | 8 | 41.03 | 128.74 | 3.14x |
| 128 | 1 | 19.56 | 27.56 | 1.41x |
| 128 | 2 | 30.16 | 51.73 | 1.72x |
| 128 | 4 | 43.04 | 84.18 | 1.96x |
| 128 | 8 | 40.91 | 129.18 | 3.16x |

The Redshift path includes Python/ctypes allocation and row copies; the reference
uses C++ vector allocation/copies. Raw reports also separate forward time from
checkpoint restoration. They are complete verification timings, not chatbot or
accepted speculative tokens/s. In particular, llama.cpp's 128.7 input tokens/s
for eight candidates does not establish 128.7 generated tokens/s.

A separate run of the installed `llama bench`, performing 64 sequential decode
steps per repetition, measures 31.61 tokens/s at depth 16 and 31.68 at depth 128.
That standard decode benchmark has a different protocol: it does not restore a
checkpoint before every token. It must not be substituted for the matched group
measurements above.

## Numerical comparison

At prefix length 16, the argmax token agrees on all eight candidate rows.
Every logit is compared: cosine similarity is at least 0.99982, row RMSE ranges
from 0.0172 to 0.0497, and maximum absolute difference is 0.3004.
The logits are close, not identical. This is limited numerical evidence on
synthetic IDs; it does not establish text quality or equivalence for arbitrary
prompts. llama.cpp's own grouped/sequential paths also show numerical differences;
its argmax remains equal for these fixtures.

Across both prefixes, argmax agrees in 16/16 rows. With the longer prefix,
the differences increase: the combined minimum cosine similarity is 0.99851,
maximum absolute error 0.7218 and maximum row RMSE 0.1462. We have not isolated
the cause of the discrepancy, and this comparison does not establish exact
equivalence or validate arbitrary text prompts.

## Provenance and isolation

- Host: `llama-v100` (`10.10.10.134`).
- GPU: Tesla V100-PCIE-32GB, UUID `GPU-1015f4d2-b64e-83ee-147f-780a38d7e199`.
  This is the same physical GPU identified in the earlier `llama-v100-exp` run.
- Reference: existing llama.cpp build, commit `a97cce86a8addeb9f40cba7a261c94b1f0c576cb`.
- Model: `/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf`.
  The full SHA-256 is `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`,
  identical to the earlier PoC measurement.
- The standalone harness links existing static libraries; the upstream source,
  installed executable, service and MTP configuration are untouched.
- Temporary work directory: `/tmp/redshift-compare-nYT07Df6`.

Reports:

- [Matched prefix-16 comparison](../reports/llama-v100-comparison-prefix16.json)
- [Matched prefix-128 comparison](../reports/llama-v100-comparison-prefix128.json)
- [Installed llama-bench decode baseline](../reports/llama-v100-native-bench.json)
- [Model/binary hashes and final server status](../reports/llama-v100-provenance.txt)

The harness and analysis script are in `tools/llama_reference.cpp`,
`tools/build_llama_reference.sh` and `tools/compare_engines.py`.

```sh
make -j20
bash tools/build_llama_reference.sh /opt/src/llama.cpp /opt/build/llama-v100-sm70
build/llama-reference /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf 16 5 llama-prefix16
python3 tools/compare_engines.py \
  /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  llama-prefix16 --output comparison-prefix16.json
```

The current next performance question is kernel efficiency relative to this
reference, especially the eight-row path. These measurements do not support a
claim that a separate runtime is already faster than llama.cpp.

## Profiling the performance gap

A follow-up Nsight Systems capture profiles three warmed Redshift calls of
eight tokens at the same prefix. Quantized projections account for 97.5% of
GPU kernel time: Q4_K 53.9%, Q8_0 35.4%, Q6_K 8.2%. The GPU kernels take about
192 ms per call. Copying every output logit back to the host takes about 0.80 ms;
checkpoint device-to-device copies take about 0.61 ms. This localizes the main
bottleneck to projection arithmetic, rather than Python output copying or state
restoration. These instrumented timings diagnose the bottleneck; the original
unprofiled measurements remain the performance results.

Redshift's Q4/Q6/Q8 kernels unpack weights into FP32 values and accumulate
ordinary floating-point products for each candidate. The observed llama.cpp
trace uses `mul_mat_vec_q`, Q8_1 activation quantization, and eight-column
`mul_mat_q` kernels for Q4_K/Q6_K. The installed backend uses packed integer
DP4A dot products on the relevant Volta path. It performs different arithmetic
and uses optimized matrix/vector kernels; reusing the same weights across
candidates alone does not reproduce those efficiencies.

Compiled Redshift Q4 kernel register use increases from 52 registers/thread
at four candidates to 70 at eight. The compiler reports no local-memory spill
for these projection kernels. Register pressure is a further optimization
lead, but its individual contribution to the slowdown has not been isolated.

- [Redshift eight-token profile](../reports/llama-v100-profile-redshift-b8.txt)
- [llama.cpp kernel trace](../reports/llama-v100-profile-llama.txt)

The llama.cpp trace includes warmup and multiple group sizes. Its aggregate
percentages should not be compared directly with the eight-token-only Redshift
profile; it establishes which kernel families actually execute.
