# V100 projection optimization — 2026-10-03

The optimized numeric PoC verifies **125.84 tokens/s in groups of eight**, up
from 41.03, on the same V100 and 27B checkpoint. A fresh llama.cpp measurement
is 129.40 tokens/s: Redshift remains 2.8% behind at prefix 16 and 3.9% behind
at prefix 128. This is a 3.05–3.07x improvement over the original PoC at eight
candidates, not a demonstrated advantage over llama.cpp.

These are supplied-token verification measurements. There is still no tokenizer,
draft/acceptance loop, generated-token benchmark, or validated text quality.

## Matched full-model measurements

Each measurement includes GPU checkpoint restoration, the entire model, every
vocabulary logit for every candidate, and copying those logits to the host.
Five repetitions alternate group order after warmup. Both engines run separately
on the same GPU. The Python/C++ output-allocation difference remains as documented
in the [initial comparison](llama-comparison.md). Loading and prefix construction
are excluded. The llama.cpp harness, checkpoint, input IDs, and configuration
are unchanged; llama.cpp was measured again for this comparison.

| Prefix | Group | Original Redshift tokens/s | Optimized Redshift tokens/s | Fresh llama.cpp tokens/s | Improvement over original |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 19.69 | 23.10 | 27.80 | 1.17x |
| 16 | 2 | 30.30 | 40.70 | 51.98 | 1.34x |
| 16 | 4 | 43.14 | 69.72 | 84.67 | 1.62x |
| 16 | 8 | 41.03 | 125.84 | 129.40 | 3.07x |
| 128 | 1 | 19.56 | 22.94 | 27.81 | 1.17x |
| 128 | 2 | 30.16 | 40.49 | 52.09 | 1.34x |
| 128 | 4 | 43.04 | 69.36 | 84.94 | 1.61x |
| 128 | 8 | 40.91 | 124.79 | 129.88 | 3.05x |

## Changes and cost

- Quantize each 32-value activation group once to signed Q8, then compute packed
  integer DP4A dot products with FP32 scaling and accumulation. Adjacent
  projections of the same unchanged input reuse this quantization.
- Repack weights once on upload for coalesced access across 32 output rows.
  Q4 remains four-bit; Q6 values expand to signed bytes. Scales expand to FP32.
  Embeddings retain their original GGUF representation.
- Use scalar or four-row vector loads depending on shape/batch; the eight-token
  Q8/Q6 path reuses a vector of weights for two tokens per thread.
- Partition the inner dimension into 1, 4, or 16 pieces according to output
  size. Reduce partials in a fixed order without atomics. The partition is
  identical across batch sizes, preserving chunk consistency.
- Keep the original FP32-activation projection as `qv_test_mm` for diagnostics;
  `qv_test_mm_fast` exercises the actual packed DP4A path.

The loaded runtime uses **23,314 MiB (22.77 GiB)** at prefix 16, versus the
original 18,950 MiB: about 4.26 GiB extra. Prefix 128 uses 23,328 MiB. Observed
load times were 10.88 and 14.07 seconds, including repacking, versus 7.10 seconds
in the initial PoC run; these are individual observations, not controlled load
benchmarks. No new library dependency was added.

Direct DP4A substitution alone reached only about 46 tokens/s at eight candidates.
Weight layout, vector reuse, and sufficient parallelism were necessary for the
larger gain. A shared-memory activation-cache experiment regressed performance
and was removed. Final microbenchmarks include input quantization and partial
reduction, while excluding allocation, upload, and one-time weight preparation.

## Numerical results and validation

The arithmetic is approximate because activations are now Q8. Original input
sums are retained in FP32 for the affine Q4 offset correction. This is not
bitwise equivalence with either the old FP32 path or llama.cpp.

All 16 tested argmax results agree with llama.cpp. At prefix 16, minimum cosine
similarity is 0.999548, maximum row RMSE 0.06657, and maximum absolute logit error
0.3850. At prefix 128 these are 0.996709, 0.17747, and 0.8884. **The longer-prefix
logit agreement is worse than the original PoC** (minimum cosine 0.998509,
maximum RMSE 0.14616). Matching argmax on these synthetic fixtures does not
establish general model quality, sampling equivalence, or speculative acceptance
correctness. Numerical fidelity needs further evaluation before a text generator
can use this result as a quality claim.

- All **21 tests pass** on the V100, including the full-model grouped/sequential
  and rejected-suffix replay checks, with their previous tolerances unchanged.
- The new independent CPU reference checks all 1–8 candidate counts, zero
  activation blocks, an outlier, short/partial Q8 widths, and output rows across
  the 32-row packing boundary. It checks both quantized arithmetic and the
  explicit per-element quantization-error bound against the original dot product.
- CUDA fixture memcheck: **0 errors**. Racecheck: **0 hazards**.
- Local Mac: 12 CPU/header tests pass; nine CUDA/model tests explicitly skip.
- Source hashes in both final reports match the local sources. The model and
  reference binary hashes match the earlier comparison. GPU returns to 4 MiB
  and 0% utilization, with `llama.service` active.

## Evidence and reproduction

- [Prefix 16](../reports/llama-v100-dp4a-prefix16.json) and
  [prefix 128](../reports/llama-v100-dp4a-prefix128.json), including source/library
  hashes, full numerical metrics, fresh native timings, and every timed sample.
- [Full tests](../reports/llama-v100-dp4a-tests.txt),
  [memcheck](../reports/llama-v100-dp4a-memcheck.txt),
  [racecheck](../reports/llama-v100-dp4a-racecheck.txt).
- [K=5120 microbenchmark](../reports/llama-v100-dp4a-micro-5120.csv) and
  [K=17408 microbenchmark](../reports/llama-v100-dp4a-micro-17408.csv).
- [Model/library/reference hashes and final host state](../reports/llama-v100-dp4a-provenance.txt).

```sh
make -j20
make bench-mm
build/bench-mm 17408 5120
QVELOX_CUDA=1 QVELOX_MODEL=/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  python3 -m unittest discover -s tests -v
make memcheck
make racecheck
# With the previously built reference harness:
build/llama-reference /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf 16 5 reference16
python3 tools/compare_engines.py /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  reference16 --output comparison16.json
```

Work remains on local `main`; no commit, push, or deployment was performed.
Compilation and GPU checks ran in the previously authorized temporary directory
`/tmp/redshift-compare-nYT07Df6` on `llama-v100`. The model briefly loaded by the
local development worker was unloaded after its slots became idle; the router
service and configuration were not changed.
