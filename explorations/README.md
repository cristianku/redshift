# CUDA and Tesla V100 Explorations

Research dated **October 3, 2026**. Goal: identify which parts of the stack we can
modify and which `sm_70` features could accelerate inference.

## Answers to the two questions

1. **We can modify our CUDA kernels and runtime** while compiling them with
   CUDA 12.9. Modifying CUDA 13 to restore Volta is not a supported route:
   both compiler target and library support have been removed.
   [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures).
2. **The V100 still offers optimization opportunities**, especially coalesced
   weight access, DP4A, weight reuse across tokens, specialized launch geometry,
   and FP16 Tensor Cores for suitable shapes. Effectiveness depends on the workload:
   a kernel that wins on eight tokens may lose on single-token decode.

The recommendation is to preserve a compatible toolchain and invest in kernels
and execution paths, starting with an updated profile. No concrete reason emerged
to attempt a fork of the proprietary toolkit.

## Documents

- [01 — What we can modify in the CUDA stack](01-cuda-sm70.md)
- [02 — V100 features, code findings, and opportunities](02-v100-performance.md)
- [03 — Experiment plan and acceptance criteria](03-benchmark-plan.md)
- [04 — From verification capacity to generation speed](04-verification-to-generation.md)
- [Local evidence: commits, hashes, and report summaries](local-evidence.json)

## Scope and limitations

This folder is stored in **Redshift**, the repository dedicated to the V100.
The research also covers code in the neighboring **DS4** repository.
The initial research in documents 01–03 modified no source code, dependencies,
configurations, or services. Document 04 records the subsequent, approved local
controlled-acceptance implementation and the remaining experiments.
No SSH access, installation, CUDA compilation, or GPU benchmark was performed
for this exploration. The local machine runs macOS and `nvcc` was not found in PATH.
The controlled-acceptance follow-up also has no CUDA build or GPU measurements;
its local tests do not certify native state correctness or a speedup.

The conclusions distinguish:

- **Verified in local code:** dispatch behavior and existing structures.
- **Documented by NVIDIA:** compatibility and architecture features.
- **Previous results:** existing Redshift reports, inspected but not rerun.
- **Hypotheses to measure:** possible improvements without promised percentages.

Some current Redshift source hashes **do not match** the Q16 reports.
Those benchmarks do not certify the current checkout. Details are recorded
in the evidence file. The evidence snapshot retains its original collection date.
