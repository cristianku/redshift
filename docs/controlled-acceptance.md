# Controlled acceptance: cost of keeping a verified prefix

This experiment is prepared locally; no CUDA build, GPU correctness result or
performance measurement is available from this change. It uses synthetic token
IDs and imposed acceptance counts. It does not implement a proposer or sampler,
and accepted inputs per second are not generated tokens per second.

## Compared paths

Both paths evaluate all candidate inputs and download every vocabulary logit.
Acceptance is supplied only after verification; it cannot influence recording.

- **Replay:** save the current checkpoint, evaluate the candidate group, restore
  on partial acceptance and replay the accepted inputs with `advance`. This is
  the existing full-model forward, but downloads only greedy IDs on replay.
  Zero acceptance restores without evaluating; full acceptance keeps the final
  state without restoring.
- **Transitions:** `verify` saves only initial DeltaNet state/history and records
  raw QKV plus prepared QKV/decay/beta for each recurrent layer. `commit(k)`
  restores that initial recurrent state and replays only its first `k`
  transitions and convolution-history shifts. It does not repeat projections,
  attention, normalization, FFN or the vocabulary projection. Attention KV rows
  are append-only, so lowering the valid position keeps the accepted rows and
  makes the rejected suffix unreachable. Future appends overwrite that suffix.
  Full acceptance skips restore and transition replay.

The transition path deliberately reuses the existing recurrence kernel and
recomputes its output into scratch. A kernel that updates only state could be a
later experiment; there is no unmeasured claim that this path is optimal.

The recurrent start copy is **149.625 MiB**; the eight-row trace is
**30.140625 MiB**, for **179.765625 MiB** of additional device buffer payload.
These buffers are allocated lazily and remain allocated until session destruction.
Trace capacity stays eight even when testing smaller groups. Full boundary
snapshots would require 149.625 MiB per boundary, about 1.17 GiB for eight copies.
The existing checkpoint also copies **64 KiB per valid KV token** per save or
restore. These are sizes derived from code, not measured HBM traffic or peak VRAM.

The public checkpoint is independent of a verification transaction. While a
transaction is pending, normal evaluation, advance, checkpoint and another
verification are rejected. Commit consumes the transaction once. Reset or restore
abort it. Invalid acceptance counts leave it available for a valid commit.

## Run in an authorized CUDA environment

Build the library from the recorded checkout before running. The Python ABI now
requires the new native symbols; an older library must be rebuilt.

```sh
make
QVELOX_CUDA=1 QVELOX_MODEL=/path/to/existing-model.gguf make test
python3 tools/bench_commit.py /path/to/existing-model.gguf \
  --prefixes 0 128 2048 8192 --groups 8 --repeats 6 \
  --output reports/controlled-acceptance.json
```

Use `--groups 2 4 8` to measure fixed candidate counts. This does not implement
an adaptive policy. No model download, installation, service or power change is
part of these commands. Use the existing compatible sm_70 CUDA toolchain.

The harness hashes the model, library and relevant sources before timing. It
loads one model instance, warms every cell, alternates strategy order and cell
order, and saves raw samples, phase medians, min/max and standard deviation.
Model hashing, loading, initial prefill, fixture restore and validation are
excluded from cycle times. Each timed cycle includes its own start-state save.
Lazy allocation is excluded by warmup; both paths then run with the trace buffers
resident. GPU metadata/VRAM readings from `nvidia-smi` are snapshots and include
other GPU processes; they are not peak-memory or traffic counters.

For replay, checkpoint, verification, restore and forward replay have separate
timings. For transitions, the initial recurrent save and trace recording are
included in **verify_seconds**; restore and transition replay are together in
**commit_seconds**. Compare **total_seconds**, not commit alone. All times are
synchronous host wall times, including Python/ctypes overhead, allocation and
host logit copying. No CUDA-event/kernel-only timings are claimed.

Every prefix/group/acceptance/strategy is checked against a sequential accepted
prefix followed by two continuation tokens. Complete continuation logits must
remain finite, share argmax and satisfy the existing model-test tolerance
`max(abs(actual-reference)/(1+abs(reference))) <= 3e-4`. A failure aborts the run
without producing a successful report. Model tests additionally compare every
verification row, rejected-suffix overwrites, consecutive transactions and aborts.

Correct continuation, including after overwriting rejected rows, is the gate
before interpreting speed. This tolerance is a grouped/sequential state check;
it is not FP32 sampling-equivalence or Q16 acceptance-quality evidence.

Next, add an actual proposer and include proposal time, acceptance selection,
correction/bonus tokens and sampler costs in emitted-token throughput. Draft,
prompt lookup and checkpoint-supported MTP remain separate follow-up experiments.

## Local verification on October 3, 2026

- Four benchmark-accounting/validation tests and the C ABI header test passed.
- `git diff --check` passed.
- `make` could not build: `/usr/local/cuda/bin/nvcc` is absent on this Mac.
- The full local suite ran 64 tests: 27 passed, 13 GPU/model tests skipped,
  24 errors (23 `test_server.ServerTests` socket binds denied by the sandbox,
  and `test_text` import failed because `tokenizers` is absent).
- Retrying `test_server.py` with loopback socket access ran 23 tests: 5 passed,
  12 failed and 6 errored; completion paths cannot import `tokenizers`.
  No unrelated tokenizer/server changes or dependency installations were made.

The local tests certify benchmark accounting and header compatibility only.
The native transition implementation remains uncompiled and unverified on GPU;
no measured speedup is available yet.
