# Redshift

**Redshift** is an experimental high-performance inference engine for **dense LLMs**, initially focused on **Qwen3.8** and **NVIDIA Volta GPUs**, especially the **Tesla V100 32 GB**.

The project is built around one question:

> How far can we push dense-model inference on Volta when the runtime is designed specifically for the hardware instead of treating it as a generic CUDA target?

## Goals

Redshift is not intended to be another general-purpose inference framework.

The initial objective is a small, specialized runtime optimized for:

- Qwen3.8 dense models
- NVIDIA Tesla V100 / Volta
- GGUF quantized weights
- low-batch interactive inference
- aggressive reduction of memory traffic during decode
- multi-token verification / speculative decoding
- Volta-specific quantized CUDA kernels
- efficient reuse of loaded weights across several candidate tokens

A long-term performance target is to investigate whether **~100 generated tokens/s** can be approached on suitable quantized Qwen3.8 models and hardware configurations.

That number is a **research target, not a claimed benchmark**.

## Why Redshift?

For a large dense model, ordinary autoregressive decoding repeatedly streams a substantial fraction of the model weights for every generated token.

On a Tesla V100, memory bandwidth is therefore one of the main physical limits.

Simply rewriting an existing inference engine while keeping the same one-token-at-a-time execution model is unlikely to produce a dramatic breakthrough.

Redshift instead explores a different execution strategy:

1. propose several tokens,
2. process or verify them together,
3. reuse model weights while they are already being consumed,
4. accept as many verified tokens as possible,
5. restore model state correctly when speculation fails.

The aim is to increase **useful generated tokens per pass over the weights**.

## Initial architecture

```text
GGUF model
    |
    v
Model loader
    |
    +--> tokenizer
    |
    +--> quantized weights
             |
             v
       Volta CUDA kernels
             |
             v
       Qwen3.8 runtime
             |
      +------+------+
      |             |
   normal        speculative
   decode          decode
      |             |
      +------+------+
             |
             v
          sampler
```

## Core areas

### GGUF

The first implementation will use GGUF as the model container.

Where appropriate, Redshift may reuse or adapt compatible open-source components, while keeping licensing and attribution requirements explicit.

### Volta kernels

Kernel work will initially target the V100 rather than attempting to support every CUDA architecture.

Areas of investigation include:

- quantized matrix-vector and small matrix-matrix operations
- tensor-core-friendly execution where useful
- fused dequantization and compute
- reduced intermediate memory traffic
- persistent/reused data where the architecture permits it
- grouped execution for 2, 4, or 8-token verification

### Multi-token verification

A key experiment is measuring the real cost of verifying multiple proposed tokens in one model pass.

The first benchmark milestone is to compare complete verification of:

- 1 token
- 2 tokens
- 4 tokens
- 8 tokens

including all required final projections and state updates.

### Speculative decoding / MTP

Redshift will investigate speculative decoding and model-native multi-token prediction where supported.

Correctness comes first: rejected proposals must restore all relevant model state and caches exactly.

## Target hardware

Initial development target:

| Component | Target |
|---|---|
| GPU | NVIDIA Tesla V100 |
| Architecture | Volta / SM70 |
| VRAM | 32 GB preferred |
| CUDA | Volta-compatible CUDA toolchain |
| Model family | Qwen3.8 dense |
| Model format | GGUF |
| Primary mode | Single-user / low-batch inference |

Support for newer NVIDIA architectures may come later, but the first implementation will intentionally optimize for Volta instead of hiding hardware differences behind a generic abstraction.

## Development strategy

The project will be developed benchmark-first.

Before building a large runtime, each proposed optimization should answer a measurable question.

Early milestones:

- [x] minimal GGUF reader
- [ ] tokenizer integration
- [x] Qwen3.8 model metadata inspection
- [x] CPU references for kernel correctness
- [x] minimal CUDA execution path
- [ ] baseline single-token decode benchmark
- [x] Volta-specific quantized kernels
- [x] complete 2/4/8-token verification benchmark
- [ ] speculative decoding prototype
- [ ] profiling of memory bandwidth and kernel occupancy
- [ ] end-to-end generation benchmark

## Principles

- **Measure before optimizing.**
- **Optimize for the actual GPU.**
- **Correctness before cleverness.**
- **Avoid abstractions that hide expensive operations.**
- **Keep the core small enough to understand and profile.**
- **Do not claim performance that has not been measured.**

## Status

Redshift is currently in the **research and initial implementation phase**.

A numeric proof of concept now runs the full 27B graph on a V100 32 GB.
With a 16-token prefix, measured throughput including state restoration is
19.6 / 30.3 / 43.2 / 41.0 supplied tokens/s for groups of 1 / 2 / 4 / 8.
These are **verification-capacity measurements**, not generated or accepted
speculative tokens/s. Tokenizer, proposal/acceptance loop and external-engine
numerical validation remain open.

See [the proof-of-concept report](docs/poc-results.md) for raw measurements,
validation and reproduction commands. The internal Python package remains
`qvelox`; `python3 -m qvelox --help` lists inspection and benchmark commands.

APIs, file layout and kernel interfaces should be considered unstable.

## License

Redshift is licensed under the **MIT License**. See [LICENSE](LICENSE).

Third-party components incorporated into the project retain their respective copyright notices and license requirements.
