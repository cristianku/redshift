# Tesla V100: Useful Resources and Concrete Opportunities

## Hardware capabilities

| Resource | Availability on Volta | Potential use |
|---|---|---|
| Tensor Cores | FP16 inputs, FP16 or FP32 accumulation | Matrix operations and attention with enough work per tile |
| DP4A | Dot products of four byte pairs, integer accumulation | Quantized weights and small batches |
| Separate FP32 / INT32 pipelines | Overlapping independent operations | Interleave addressing and computation |
| HBM2 | Standard V100: up to 900 GB/s theoretical | Coalesced weight streaming |
| Shared memory / L1 | Combined 128 KiB pool; up to 96 KiB shared memory per SM | Tile reuse balanced against cache and occupancy |
| Registers | 64K 32-bit registers per SM | Reuse without overloading each thread |
| CUDA Graphs | Available with a compatible stack | Reduce repeated CPU submission |

Sources: [Volta Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html),
[V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf),
[PTX ISA](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html),
[cuBLAS and CUDA Graphs](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cuda-graphs-support).

Bandwidth is a theoretical peak, not a measurement of our card. Do not attribute
SXM/NVLink performance to a PCIe V100. Previous Redshift reports identify the
PCIe 32 GB variant; no live verification was performed in this session.

Volta lacks the native TF32, BF16, INT8/INT4, FP8, and FP4 Tensor Core capabilities
of later architectures. GGUF Q4/Q6/Q8 weight storage is still possible:
storage formats and compute instructions are different things. DP4A is not
an INT8 Tensor Core operation.

## DS4 code findings

| Finding | Local evidence | Consequence |
|---|---|---|
| Quantized MMQ uses DP4A on sm_70 | [`mmq.cuh`](../../ds4/cuda/mmq/mmq.cuh), `vec_dot_mma` / `vec_dot_dp4a` selection around line 3678 | Volta primitives in `mma.cuh` do not mean this path uses them |
| sm_70 FP16 WMMA kernels already exist | [`ds4_cuda.cu`](../../ds4/ds4_cuda.cu), `indexer_scores_wmma*`, `>=700` guard | The backend already has some Tensor Core paths |
| GEMM fallback dequantizes Q8 to FP16 | `ds4_gpu_matmul_q8_0` in `ds4_cuda.cu`, `cublasGemmEx` path | Compare dequantization + GEMM + copies against MMQ |
| Token-tiled attention requires Ampere | `ds4_cuda_attn_tokentile_arch_ok`: `prop.major >= 8`; Qwen variants guarded by `>=800` | A new Volta FP16 variant is possible; lowering the guard is insufficient |
| Decode graphs already exist | `ds4_gpu_decode_graphs_supported`, around line 964 | Measure hits/replays and coverage before proposing more graphs |
| Optional MMQ has exclusion conditions | `cuda_use_mmq`: quality mode, multi-GPU, `DS4_CUDA_MMQ=0` | Check the dispatch actually reached by the target workload |

Starting with version 11, cuBLAS can automatically choose Tensor Cores when
beneficial. `CUBLAS_GEMM_DEFAULT` does not prove Tensor Cores are absent;
similarly, setting `CUBLAS_TF32_TENSOR_OP_MATH` cannot create TF32 on Volta.
Observe the actual selection in a profile.
[cuBLAS 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#tensor-core-usage).

## Redshift findings: what has already been tried

[`kernel-optimization.md`](../docs/kernel-optimization.md) documents:

- Weight repacking for coalesced access and vector loads.
- Two byte components for Q16 activations, two DP4A operations, and FP32 scaling/accumulation.
- Activation reuse between adjacent projections of the same input.
- Different geometries for Q4/Q6/Q8, shapes, and token counts.
- A slower WMMA experiment with residual correction, subsequently removed.
- A slower shared-memory activation-cache experiment, subsequently removed.

The code inspected in [`kernels.cu`](../src/kernels.cu) still contains
`ActivationQ16`, `__dp4a`, and shape-specific dispatch in `k_mm_packed`.
No CUDA Graph call was found in the Redshift `src` files inspected.

Previous JSON reports measure complete verification with full-vocabulary logits:

| Prefix | Tokens per group | Redshift verified tokens/s | llama.cpp verified tokens/s |
|---|---:|---:|---:|
| 16 | 1 | 25.31 | 27.82 |
| 16 | 4 | 86.50 | 84.76 |
| 16 | 8 | 140.71 | 129.74 |
| 128 | 1 | 25.10 | 27.81 |
| 128 | 8 | 140.12 | 129.67 |

Sources: [prefix 16](../reports/llama-v100-q16-prefix16.json),
[prefix 128](../reports/llama-v100-q16-prefix128.json).
These numbers measure **supplied tokens being verified**, not generated tokens
or tokens accepted by a speculative loop. Measurements include GPU restore,
forward execution, and logit copies; the engines retain a Python/C++ host
allocation difference.

At the evidence collection time, hashes of `src/kernels.cu`, `src/runtime.cu`,
`src/runtime.h`, and `qvelox/runtime.py` differed from the reports.
These are results for the measured revision, **not benchmarks of the current checkout**.

The profile attributing 97.5% of kernel time to projections belongs to the
initial FP32 PoC ([report](../docs/llama-comparison.md)). Profiling must be repeated
after DP4A/Q16. Do not treat that percentage as the current bottleneck.

## Opportunity priorities

### 1. Layout, reuse, and small-group dispatch — high priority

Build on the existing DP4A path: actual bytes read, coalescing, weight reuse
across tokens, registers, split-K, and reduction. Previous reports show the most
obvious gap at groups of one and two tokens. More rows/tokens per thread can
improve reuse while reducing concurrency or causing spills. Sweep quantization
and shape families rather than relying on one global parameter.

### 2. FP16 Tensor Cores for prefill and larger batches — conditional priority

Compare DP4A against dequantization + cuBLAS FP16 and, where beneficial,
tile-fused dequantization + Volta WMMA/MMA. Keeping weights compressed in HBM
can avoid permanent FP16 expansion. Padding, conversion, and small N can erase
the gain. Redshift already documents a failed WMMA attempt; do not repeat it
without a precise new hypothesis.

CUTLASS `mma_sm70.h` is a reference for Volta geometry. It can support an isolated
experiment without automatically introducing a new dependency.

### 3. CUDA Graphs and fusion — priority after profiling

In DS4, compare existing graphs on/off and identify operations outside them.
In Redshift, measure gaps between kernels and CPU submission overhead.
Norm/quantize fusion or fused epilogues can reduce reads and launches; graphs can
reduce host overhead. Neither necessarily reduces weight bytes read from HBM.
Graphs require stable buffers and dependencies; verify restore and recurrent state.

### 4. Volta-specific FP16 attention — priority for long contexts

Study an online/fused kernel that avoids materializing the full score matrix.
DS4 and Redshift already have online variants: compare and specialize the paths
reached on sm_70 rather than simply adding “FlashAttention.” FP16 inputs require
numerical validation.

[SparkAttention, version 3](https://arxiv.org/abs/2502.12784v3) investigates
Tensor Cores and attention fusion on V100 and is a relevant research lead.
Its speedups against PyTorch concern MHA training and do not establish gains
for our decode, GQA, sparse attention, or recurrent state. The library's code
was neither evaluated nor imported.

### 5. Memory, copies, and concurrency — deployment-dependent

Keep repeatedly used weights and scratch in VRAM, avoid unnecessary
synchronization, and assess pinned/asynchronous copies where transfers are needed.
Expanding weights or caches costs VRAM: the Redshift Q16 report records
23,314 MiB, approximately 22.77 GiB, at prefix 16.
Measure the complete working set before increasing context or concurrency.

Consider MPS only for concurrent processes and NVLink only for multiple GPUs
with a verified connection. Neither automatically accelerates one request on
a PCIe V100. Changes to MPS, clocks, power limits, or ECC are not part of the
initial proposal.

## Decision

The strongest path is optimizing work executed with CUDA 12.9 rather than
forcing CUDA 13 compatibility. Start with DP4A/layout/dispatch for short decode;
compare FP16 Tensor Cores for larger prefill; profile attention and state for
long contexts. Do not assign speedup percentages before measuring.
