# From verification capacity to generation speed

Cristian's exploration priorities, recorded on **October 3, 2026**. The question
is how to convert fast verification into faster generation after accounting for
proposal, acceptance, state commit and sampling. Verification throughput alone
does not answer it.

## Priorities, in order

1. **Commit after partial acceptance.** When 8 candidates are verified and only
   the first 3 accepted, compare restoring and replaying the accepted prefix
   against saving/replaying only the necessary recurrent transitions. Measure
   the cost of preserving the initial state and recording transitions as well
   as restore/commit, across acceptance counts and context lengths.
2. **A proposer for the real model.** Compare a small draft and prompt-derived
   proposals. Test MTP only if the actual checkpoint supports it. On code,
   Italian and tool calls, measure proposal time, mean accepted-prefix length
   and its distribution; include all costs in emitted-token throughput.
3. **Adaptive candidate count.** Compare fixed groups of 2/4/8 with a policy
   driven by recent acceptances. Include policy/proposal/verification/commit
   costs and wasted work when few candidates are accepted. The fixed-group
   comparison is available in the prepared harness; no adaptive policy exists.
4. **Actual HBM traffic of repacked weights.** Compare the existing expanded
   layout with a more compact Q6 representation. Fewer bytes can require more
   decode instructions; measure traffic, instruction costs, VRAM and the
   complete forward pass, not only a matrix microbenchmark. Buffer sizes are
   not a substitute for measured HBM traffic.
5. **Precision and acceptance.** Compare Q16 with the existing FP32 projection
   reference on full probability distributions and speculative acceptance,
   especially at long contexts. Keep the same underlying model/quantized
   weights and sampler configuration so this isolates activation arithmetic.
   Matching argmax alone does not establish sampling equivalence.

## First experiment: acceptance controlled from 0 to 8

Force acceptance **after** verifying the candidate group, without a draft.
This separates state-management cost from proposal quality. Compare identical
supplied IDs, all-logit verification and the same initial state in both paths:

- **Full-forward replay:** checkpoint, verify, restore on partial acceptance,
  replay the accepted prefix with the existing lightweight `advance` ABI.
- **Recurrent transitions:** save initial DeltaNet state/history and record raw
  QKV plus prepared QKV/decay/beta during verification. Restore/replay only the
  accepted recurrence/history transitions; reuse the already appended KV rows
  by shortening their valid prefix. Acceptance is unknown during recording.

Zero acceptance restores without replay; full acceptance retains the final state.
Check complete continuation logits against a sequential accepted prefix before
interpreting timings. Include recording overhead in the total cycle; comparing
commit alone could conceal a slower verification path.

The implemented buffer payload is **149.625 MiB** for the recurrent start copy
plus **30.140625 MiB** for eight transition rows: **179.765625 MiB** extra.
Saving eight full recurrent boundary snapshots would instead cost about
**1.17 GiB**. The existing checkpoint also copies **64 KiB per valid KV token**
on save/restore. These sizes are derived from current code, not measured traffic
or peak VRAM. This transition path is an experimental comparison, not a proven
optimal commit implementation.

## Current status and next evidence

Prepared locally on `main`, preserving existing edits:

- [Controlled-acceptance benchmark](../tools/bench_commit.py), supporting
  groups 2/4/8, short/long prefixes, alternating repetitions, raw phase timings,
  dispersion, hashes and continuation validation.
- Native `verify`/`commit` path and model tests for all acceptance counts,
  rejected-suffix overwrites, consecutive transactions and aborts.
- [Detailed protocol, commands and local verification](../docs/controlled-acceptance.md).

Four benchmark tests and the C ABI header test pass locally. CUDA is unavailable
on this Mac; the native path has **not been compiled or validated on GPU**, and
no performance measurements exist for this experiment. The broader local suite
also encounters missing `tokenizers` and sandbox socket restrictions, as recorded
in the protocol. No remote execution, installation or power change was performed.

After GPU state validation and controlled-acceptance timings, introduce a real
proposer. Report tokens actually emitted divided by total proposal, verification,
acceptance/commit, correction/bonus-token and sampler time. Only then evaluate
adaptive candidate counts and generation speed on the three requested workloads.
