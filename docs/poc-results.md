# Redshift numeric proof of concept — 2026-10-03

This report records the **initial numeric PoC**. See the subsequent
[projection optimization report](kernel-optimization.md) for the current
140.7 tokens/s eight-candidate result, FP32 fidelity checks, and fresh llama.cpp timings.

The independent runtime now loads the existing Qwen3.8-27B Q4_K_M GGUF,
executes all 64 layers on a Tesla V100 32 GB, and returns all 248,320 logits
for each of 1, 2, 4 or 8 supplied token IDs. GPU weights stay quantized;
uploads use a reusable 16 MiB host buffer. There is no DS4/llama.cpp runtime
dependency.

## Measurements

Five alternating-order repetitions per group size, after a correctness/warmup
pass. The same nonzero prefix checkpoint is restored before each measurement.
The following times include restoration, full model evaluation, every output
projection, copying all logits to the host, and Python output allocation/copies.
They exclude model loading and initial prefix construction.

| Prefix | Group | Median time | Verified input tokens/s |
|---|---:|---:|---:|
| 16 tokens | 1 | 50.96 ms | 19.62 |
| 16 tokens | 2 | 66.06 ms | 30.27 |
| 16 tokens | 4 | 92.67 ms | 43.16 |
| 16 tokens | 8 | 194.92 ms | 41.04 |
| 128 tokens | 1 | 51.00 ms | 19.61 |
| 128 tokens | 2 | 66.38 ms | 30.13 |
| 128 tokens | 4 | 93.18 ms | 42.93 |
| 128 tokens | 8 | 195.25 ms | 40.97 |

At this prefix length, a group of four processes supplied tokens about 2.20x
faster than the single-token path. Eight is slower per token than four in this
implementation. Loading took 7.10 seconds; allocated GPU memory reported by
`nvidia-smi` was 18,950 MiB with the model and checkpoint buffers loaded.

Raw report: [poc-v100-prefix16.json](../reports/poc-v100-prefix16.json).
It records each timing, exact input IDs, model/source/library SHA-256 hashes,
GPU/driver/compiler information, output checksums and numerical comparison errors.
The [128-token-prefix report](../reports/poc-v100-prefix128.json) confirms similar
throughput. All grouped logits matched the sequential path exactly in both runs.
That result applies to these fixtures and precision settings; it is not a
guarantee for every possible input.

## Correctness and scope

- Quantized Q4_K, Q6_K and Q8_0 projections, weighted RMS, gated DeltaNet and
  causal attention are compared against independent scalar CPU arithmetic.
- Full-model groups 1/2/4/8 match sequential evaluation at a nonzero prefix;
  every vocabulary logit row is checked, including finite values and argmax.
- Rejected-suffix testing restores a checkpoint, replays the accepted prefix,
  and checks a continuation against a fresh equivalent run. The checkpoint
  includes recurrent state, convolution history and the valid FP16 KV prefix.
- Invalid calls and context overflow must leave the position unchanged.
- Small CUDA fixtures pass compute-sanitizer memcheck and racecheck.
- The final server run passed all 20 tests; memcheck reported zero errors and
  racecheck zero hazards. [Validation log](../reports/poc-v100-validation.txt).
- Fixed two pre-existing reduction defects: the block sum was only available
  in the first warp, and attention could reuse shared reduction storage before
  every warp finished reading it.
  A separately compiled copy with the old block reduction fails the RMS test
  as expected: [regression evidence](../reports/poc-v100-rms-regression.txt).

**These are verification-capacity measurements, not generated or accepted
speculative tokens/s.** Inputs are synthetic token IDs. There is no tokenizer,
proposal model, acceptance loop, chat interface or text-quality validation yet.
A subsequent [comparison with llama.cpp](llama-comparison.md) measures both
performance and all logits: llama.cpp is faster in every group; argmax agrees
on 16/16 tested rows, with nonzero logit differences. General numerical
equivalence remains unverified. The 100 generated tokens/s research target
has not been demonstrated.

The graph follows the existing explicit `qwen35` GGUF schema. Its gated
DeltaNet head broadcasting agrees with the GGUF graph in
[llama.cpp's Qwen3.5 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp);
that source check is not a numerical cross-engine comparison.

## Reproduce on the CUDA test host

```sh
make -j20
make test-cuda
QVELOX_CUDA=1 QVELOX_MODEL=/opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  python3 -m unittest discover -s tests -v
make memcheck
make racecheck
python3 -m qvelox inspect /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf
python3 -m qvelox bench /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  --prefix 16 --repeats 5 --output benchmark.json
```

Local `make test` runs the CPU/header checks and explicitly skips CUDA and
full-model tests unless their environment variables are supplied. The Python
package and ABI retain the original `qvelox`/`qv_` names.

Validation ran in `/tmp/redshift-xNTyQ95T` on `llama-v100-exp` (`10.10.10.55`).
No inference service or existing server checkout was changed.
