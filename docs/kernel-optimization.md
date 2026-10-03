# V100 projection optimization — 2026-10-03

## Second iteration: Q16 activations and measured launch geometry

The current numeric PoC verifies **140.71 tokens/s in groups of eight**, versus
129.74 for llama.cpp, while sharply reducing error against the original FP32
graph. Single-token verification remains slower.

The second iteration replaces the lossy Q8 activation representation with two
signed-byte components per value. For each group of 32 inputs, `s=max(abs(x))/32639`
and `q=round(x/s)`. The decomposition `q=256*high+low` lets two signed DP4A
operations retain nearly 16-bit activation precision, while scaling and
accumulation remain FP32. Q4 affine offsets still use the original FP32 input sum.
This does not change the checkpoint's weight quantization.

Aligned `int4` activation loads and a unified projection template replace three
separate kernels. The number of rows and tokens handled by each thread is chosen
from V100 measurements for each quantization/shape/batch family. At eight inputs,
Q4 benefits from reusing four weights across two tokens; most Q8/Q6 projections
benefit from four rows and one token per thread. Merely adding a second DP4A to
the old launch geometry was slower. A trial WMMA path with residual correction
was also slower and was removed.

`make reference` builds `build/libqvelox-reference.so`: the same graph with the
original GGUF layout and original FP32-activation projections. This separates
new projection error from the existing differences between Redshift and llama.cpp.
The diagnostic build is separate from the normal runtime and changes no public ABI.

### Final matched timing

Five alternating-order repetitions per group after warmup, with fresh native
llama.cpp runs before each Redshift run. Same V100, GGUF weights, input IDs and
reference harness as round one. Timing includes restoration, the complete graph,
all vocabulary projections, and copying every logit to the host. Model loading
and prefix preparation are excluded. The existing Python/C++ host allocation
difference remains. Both engines run separately.

| Prefix | Group | Previous Redshift tokens/s | Q16 Redshift tokens/s | Fresh llama.cpp tokens/s | Redshift vs llama.cpp |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 23.10 | 25.31 | 27.82 | -9.0% |
| 16 | 2 | 40.70 | 48.13 | 52.11 | -7.7% |
| 16 | 4 | 69.72 | 86.50 | 84.76 | +2.1% |
| 16 | 8 | 125.84 | 140.71 | 129.74 | +8.5% |
| 128 | 1 | 22.94 | 25.10 | 27.81 | -9.8% |
| 128 | 2 | 40.49 | 47.62 | 52.15 | -8.7% |
| 128 | 4 | 69.36 | 85.80 | 84.60 | +1.4% |
| 128 | 8 | 124.79 | 140.12 | 129.67 | +8.1% |

Eight-token verification improves 11.8–12.3% over round one and measures 8.1–8.5%
ahead of llama.cpp. Groups of four are 1.4–2.1% ahead; groups of one and two remain
7.7–9.8% behind. The small four-token margin should not be generalized beyond
these measurements. The eight-token result is about 3.43x the original FP32 PoC.
These are **supplied-token verification rates, not generated/accepted tokens/s**.

VRAM is still 23,314 MiB at prefix 16 and 23,328 MiB at prefix 128. Observed model
load times, including weight repacking, were 11.75 and 12.49 seconds. Weight
storage is unchanged from round one; the wider activation scratch adds less than
0.2 MiB. The optimized library SHA-256 is
`4103bf510d4da412f4db11f8570812d009635a085d32f27cab29fbea44daf320`.

Against llama.cpp, all 16 argmax results still match. Maximum row RMSE is 0.04956
at prefix 16 and 0.14669 at prefix 128, close to the original FP32 PoC rather than
the increased Q8 error. Minimum cosines are 0.999822 and 0.998498 respectively;
maximum absolute logit differences are 0.3003 and 0.7243. No sampling or text-
quality equivalence is established by these synthetic rows.

Evidence: [prefix 16](../reports/llama-v100-q16-prefix16.json),
[prefix 128](../reports/llama-v100-q16-prefix128.json),
[K=5120 microbenchmark](../reports/llama-v100-q16-micro-5120.csv),
[K=17408 microbenchmark](../reports/llama-v100-q16-micro-17408.csv),
and [provenance](../reports/llama-v100-q16-provenance.txt).

### Precision against the original FP32 graph

Each cell is the worst RMSE among eight complete vocabulary logit rows after the
specified prefix. Both candidate libraries use the same IDs and model. The
libraries are loaded sequentially to avoid requiring two resident models.

| Prefix tokens | Previous Q8 activation path | New Q16 activation path | RMSE reduction |
|---|---:|---:|---:|
| 16 | 0.067162 | 0.00031565 | 213x |
| 128 | 0.117699 | 0.00100459 | 117x |
| 1024 | 0.106382 | 0.00033811 | 315x |

All 24 Q16 argmax results match the original graph; the worst absolute logit
difference is 0.00559. This is a numerical fixture, not a text-quality benchmark.
The remaining Redshift/llama.cpp difference must not be attributed entirely to
activation quantization. See the complete [precision report](../reports/llama-v100-q16-precision.json).

All 21 CUDA/full-model tests pass, including unchanged grouped/sequential and
rejected-suffix tolerances. Complete-model sequential equivalence now covers
every group size from one through eight. The independent projection oracle now also exercises
131 output rows, covering partial vector tiles under the new small-batch routes.
Memcheck reports zero errors and racecheck zero hazards. Evidence:
[tests](../reports/llama-v100-q16-tests.txt),
[memcheck](../reports/llama-v100-q16-memcheck.txt),
[racecheck](../reports/llama-v100-q16-racecheck.txt).

### What the NInfer comparisons actually establish

The user supplied [NInfer-4090](https://github.com/sergiuszm/ninfer-4090) and
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090). Their published generated-token
rates and Redshift's supplied-token verification rates measure different work.
The [4090 comparison](https://github.com/sergiuszm/ninfer-4090/blob/rtx4090-port/docs/llamacpp-comparison.md)
reports about a 10% shallow decode advantage without speculation, 25% on shallow
code with MTP on both engines, and 82% on prose at 128K with MTP on both. The
weight artifacts differ. The 3090 README reports 71 generated tokens/s for one
request and 165.33 aggregate tokens/s for eight simultaneous requests, with MTP3;
those eight requests are not Redshift's eight candidates from one sequence.
These are their published measurements, not benchmarks reproduced here.

Source inspection of NInfer-4090 at `aeeba414459d5d6989d57d8487c9d7a2f54bddd3`
identifies applicable ideas: shape-specific SIMT projection routes, reused CUDA
Graphs, a GPU draft/verify/accept loop, and replaying compact GDN transition inputs
when committing an accepted prefix. Its Q4 small-batch dispatch also uses SIMT
rather than Tensor Cores for many projections. Relevant source paths:
`src/ops/linear/q4/q4_dispatch.cpp`, `src/core/decode_graph.cpp`,
`src/targets/qwen3_6/impl/runtime/mtp_impl.h`, and
`docs/maintainer/replayssm-gdn.md`. No NInfer source was incorporated in this change.

Redshift still needs a complete generation/acceptance loop and efficient accepted-
prefix state commit before it can make the same comparison. Its current rollback
replays the entire accepted token prefix through the model. A future replay of
only recorded recurrent transitions must preserve the exact state-update order;
mathematical equivalence alone is insufficient. No MTP speedup is claimed here.

### Reproduce the FP32 precision comparison

```sh
make -j20 all reference
python3 tools/compare_precision.py /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --candidate build/libqvelox.so --output precision.json
```

The previous-Q8 comparison additionally passes a saved round-one library with
`--candidate`. Both library hashes are recorded in the precision report.

## First iteration: Q8 activation DP4A (historical results)

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
