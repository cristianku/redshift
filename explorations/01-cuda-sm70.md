# CUDA on sm_70: What We Can Modify

## The actual limit

Tesla V100 is Volta, compute capability 7.0. CUDA 13.0 removes offline compilation
and library support for Maxwell, Pascal, and Volta. NVIDIA recommends retaining
CUDA 12.9 for these architectures and identifies R580 as the last compatible
driver family. This does not prevent developing new kernels with the older toolkit.

Sources: [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures),
[NVIDIA architecture compatibility guide](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/).

The “CUDA Version” field in `nvidia-smi` describes driver compatibility;
it does not identify the toolkit used to compile an application. Future profiles
must record the driver, `nvcc --version`, and loaded libraries separately.

## Modifiable components

| Component | Options | Assessment |
|---|---|---|
| Our engines' `.cu` / `.cuh` kernels | Layout, tiling, fusion, precision, dispatch, compatible inline PTX | Primary target |
| DS4 / Redshift runtime | Buffer reuse, graphs, copies, batching, state handling | Can eliminate work and overhead |
| Vendored llama.cpp code | Explicit fork, tracked changes, MIT license compliance | Feasible; requires testing and maintenance |
| CUTLASS C++ | Study Volta primitives and kernels under BSD 3-Clause | Feasible; not every modern example supports sm_70 |
| cuBLAS 12.x | Select APIs, types, workspace, and available algorithms | We can configure its use, rather than modify its internal kernels as an open project |
| nvcc / ptxas / NVIDIA binary runtime | No normal public fork workflow to restore Volta in CUDA 13 | Not recommended |
| V100 hardware | Software cannot add execution units from later GPUs | Physical limit |

DS4 records kernel provenance and its upstream pin in
[`cuda/mmq/VENDOR.md`](../../ds4/cuda/mmq/VENDOR.md). Some descriptions in that
file are historical; dispatch findings were checked against the actual code.
CUTLASS retains dedicated primitives in
[`mma_sm70.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm70.h).
Its [official repository](https://github.com/NVIDIA/cutlass) distinguishes the
APIs and documents the license. A primitive's presence does not prove that every
CUTLASS release or kernel builds with our toolchain.

The [CUDA 12.9 license](https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html)
distinguishes components and terms: the toolkit should not be treated as one
open-source project. Before reusing external sources, check the specific
component's license. No new external source was incorporated in this research.

## Why changing a version check is insufficient

Forcing `sm_75` on V100 does not make Turing instructions executable.
Removing a `__CUDA_ARCH__ >= 800` guard does not convert a TF32/Ampere kernel
into a Volta kernel. A genuine Volta variant must change instructions, fragment
layouts, loads, and numerical precision where necessary.

PTX documents FP16 `mma` with the `m8n8k4` shape for sm_70; the cited INT8/INT4
integer MMA instructions require at least sm_75. `cp.async` requires sm_80.
These are hardware requirements, separate from toolkit versions.
[PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html).

## Local build configuration

- DS4: [`Makefile`](../../ds4/Makefile) has configurable `CUDA_ARCH`;
  on Linux, `make cuda CUDA_ARCH=sm_70` selects Volta.
- `cuda-spark` selects sm_121, which is not the V100 target.
- Redshift: [`Makefile`](../Makefile) already uses `-arch=sm_70`.
- DS4 uses `--use_fast_math`; the inspected Redshift Makefile does not set it.
  Preserve this difference in precision comparisons rather than copying the flag
  automatically to obtain speed.

Selecting the correct target **does not prove** that the entire DS4 backend
builds or works on V100. Builds and tests with the selected toolchain are required.
They were not performed here.

## Proposed strategy

Keep CUDA 12.9 and compatible libraries as a reproducible baseline.
Record explicit paths and versions rather than relying only on the
`/usr/local/cuda` symlink. Where useful, compare already available 12.x toolkits
with identical sources and flags; a newer 12.x toolkit does not guarantee faster
kernels.

Do not arbitrarily mix the 12.9 compiler and CUDA 13 libraries expecting Volta
support. A container can make the user-space environment reproducible, but it
cannot add hardware support or replace the host driver. This research neither
authorizes nor proposes an automatic server update.
