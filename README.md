# Redshift

**Redshift** is an experimental high-performance inference engine for **dense LLMs**, initially focused on **Qwen3.8** and **NVIDIA Volta GPUs**, especially the **Tesla V100 32 GB**.

The project is built around one question:

> How far can we push dense-model inference on Volta when the runtime is designed specifically for the hardware instead of treating it as a generic CUDA target?

## Install on a V100 server

Supported target: **Ubuntu 24.04 x86_64 with one Tesla V100 32 GB**, systemd,
16 GB RAM and at least 50 GB free disk space for CUDA, the model and the build.
In an LXC, the host must already expose the NVIDIA GPU and working driver to the
container. On bare metal the installer can install the proprietary R580 driver;
if it is missing, reboot the new server yourself and rerun the same command.

```sh
sudo apt-get update
sudo apt-get install -y git
git clone https://github.com/cristianku/redshift.git
cd redshift
sudo ./scripts/install-redshift.sh --host 0.0.0.0 --public-host YOUR_SERVER_IP
```

Replace `YOUR_SERVER_IP` with the address clients can reach. With no arguments,
the endpoint binds to `127.0.0.1:8081`. The installer:

1. Installs missing build/Python dependencies and **CUDA Toolkit 12.9**.
2. **Automatically downloads Qwen3.8-27B Q4_K_M from Hugging Face** if the model
   is absent, then verifies its SHA-256 and supported GGUF schema.
3. Builds Redshift in `/opt/redshift/releases/`, creates an isolated Python
   environment, and runs checks before switching the active release.
4. Creates a dedicated `redshift` user and **`redshift.service`**, enables startup
   at boot, and checks health plus a real inference request.
5. Writes `/opt/redshift/copilot-model.json` for the VS Code client.

The default model is [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF),
file `Qwen3.8-27B-Q4_K_M.gguf`, pinned to revision
`97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827`. Its SHA-256 is
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
and download size is **18,973,870,432 bytes (about 19 GB)**.
[Pinned model metadata](https://huggingface.co/api/models/ggml-org/Qwen3.8-27B-GGUF/revision/97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827?blobs=true).
No Hugging Face account or token is needed for this public artifact. The installer
downloads only the text backbone; the current endpoint does not use MTP or vision
projectors. Inference itself never downloads models.

Existing verified weights in `/opt/models/qwen3.8-27b/` or the installation cache
are reused. To use a different local location and **prevent model downloads**:

```sh
sudo ./scripts/install-redshift.sh --host 10.10.10.55 \
  --model /opt/models/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf
```

A missing explicit `--model` fails instead of downloading. A custom HTTP(S) URL
is supported with `--model-url URL --model-sha256 SHA256`; it must still be a GGUF
compatible with this runtime. Preview without installing or contacting the network:

```sh
./scripts/install-redshift.sh --dry-run --host 10.10.10.55
```

For an already prepared server with limited Internet access, `--offline` forbids
all package/model downloads and `--reuse-python /path/to/existing/.venv/bin/python`
copies an existing compatible dependency environment. It requires existing CUDA,
compiler and local weights. This migration uses that mode, rather than repeating
the 19 GB download.

```sh
systemctl status redshift
journalctl -u redshift -n 30
curl http://YOUR_SERVER_IP:8081/health
```

Rerunning the installer creates a new release and preserves the previous one.
Failed activation restores the previous service/release. Existing unrelated
directories, service units and drop-ins are refused. NVIDIA driver/toolkit choices
are explained in [NVIDIA's architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).
The endpoint currently has no TLS or authentication: bind to loopback or a trusted LAN.

## Use with GitHub Copilot in VS Code

Open **Chat: Manage Language Models → Add Models → Custom Endpoint**, choose
**Chat Completions**, and merge the generated `/opt/redshift/copilot-model.json`
into VS Code's `chatLanguageModels.json`. Then select
**Redshift Qwen3.8 27B (V100)** in the chat model picker. No API key is required
by this endpoint. [Complete endpoint details](docs/endpoint.md).

The current configured server is `redshift-v100` (`10.10.10.55:8081`). The former
PoC on `llama-v100` was removed. If Copilot reports that static instructions/tools
exceed the context budget, use fewer tools or Ask mode; the runtime's actual
139,264-token capacity must not be exceeded by a larger advertised window.

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
- [x] tokenizer integration
- [x] Qwen3.8 model metadata inspection
- [x] CPU references for kernel correctness
- [x] minimal CUDA execution path
- [x] baseline single-token decode benchmark
- [x] Volta-specific quantized kernels
- [x] complete 2/4/8-token verification benchmark
- [ ] speculative decoding prototype
- [ ] profiling of memory bandwidth and kernel occupancy
- [x] end-to-end generation benchmark

## Principles

- **Measure before optimizing.**
- **Optimize for the actual GPU.**
- **Correctness before cleverness.**
- **Avoid abstractions that hide expensive operations.**
- **Keep the core small enough to understand and profile.**
- **Do not claim performance that has not been measured.**

## Status

Redshift is currently in the **research and initial implementation phase**.

A numeric proof of concept runs the full 27B graph on a V100 32 GB.
After [projection-kernel optimization](docs/kernel-optimization.md), measured
throughput at prefix 16 is **25.3 / 48.1 / 86.5 / 140.7 supplied tokens/s** for
groups of 1 / 2 / 4 / 8, including state restoration and all vocabulary logits.
A fresh matched llama.cpp run measures **27.8 / 52.1 / 84.8 / 129.7 tokens/s**.
Redshift is 8.5% faster at eight candidates and still slower at one and two.

These are **verification-capacity measurements**, not generated or accepted
speculative tokens/s. The proposal/acceptance loop remains open.
The optimized runtime uses about 22.8 GiB of VRAM and two-byte quantized
activations. Against the original FP32 projection graph, maximum row RMSE is
0.0010 across tested prefixes of 16, 128 and 1024 tokens, with all 24 argmax
results matching. General text quality and sampling equivalence remain unvalidated.

A [working HTTP endpoint](docs/endpoint.md) now adds the GGUF tokenizer, chat
template, ordinary autoregressive generation, SSE streaming and function-tool
calls. It runs Redshift directly and can be configured as a GitHub Copilot custom
endpoint in VS Code. A V100 smoke test generated an Italian code explanation at
about **25 tokens/s including prompt processing**. This is an end-to-end generation
measurement, separate from the eight-token verification benchmark above.
The endpoint has a 139,264-token runtime capacity; its live long-prompt smoke uses
2,339 tokens and does not establish quality over the full window.

See [the proof-of-concept report](docs/poc-results.md) for raw measurements,
validation and reproduction commands. The internal Python package remains
`qvelox`; `python3 -m qvelox --help` lists inspection and benchmark commands.

APIs, file layout and kernel interfaces should be considered unstable.

## License

Redshift is licensed under the **MIT License**. See [LICENSE](LICENSE).

Third-party components incorporated into the project retain their respective copyright notices and license requirements.
