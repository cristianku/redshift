# Redshift endpoint proof of concept

The server runs the Redshift CUDA graph directly. Python provides GGUF byte-BPE
tokenization, the embedded Jinja chat template, HTTP and streaming. It does not
proxy inference to llama.cpp. Generation currently uses ordinary autoregressive
decoding; speculative proposal/acceptance and MTP are not implemented.

## Run

On a CUDA host with the supported Qwen3.8-27B-Q4_K_M GGUF:

```sh
make -j20
python3 -m venv .venv
.venv/bin/pip install -r requirements-server.txt
.venv/bin/python -m qvelox.server /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8081 --context 139264
```

`GET /health`, `GET /v1/models` and `POST /v1/chat/completions` are available.
The model ID is `redshift-qwen3.8-27b`. Context includes the rendered prompt,
tool definitions, and completion. Oversized requests return HTTP 400.

```sh
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"redshift-qwen3.8-27b","messages":[{"role":"user","content":"Quanto fa 6 per 7?"}],"temperature":0,"max_tokens":64,"stream":true}'
```

The current instance is on **`redshift-v100` (`10.10.10.55:8081`)**, installed
under `/opt/redshift` with isolated dependencies and the dedicated `redshift`
service user. **`redshift.service` is enabled at boot.** Install from a new Ubuntu
server with [the README procedure](../README.md#install-on-a-v100-server).
Status and logs:

```sh
ssh redshift-v100 systemctl status redshift
ssh redshift-v100 journalctl -u redshift -n 30
```

To release its GPU memory after testing, stop **only** this instance with
`ssh redshift-v100 systemctl stop redshift`. The original PoC was removed from
`llama-v100` at the user's request. The two LXC containers expose the same physical
GPU: keep other inference models unloaded while Redshift is resident.
This PoC has no authentication or TLS;
use loopback or a trusted LAN, not a public listener.

## GitHub Copilot in VS Code

Use **Chat: Manage Language Models → Add Models → Custom Endpoint**, API type
**Chat Completions**, and the model entry in [copilot-model.json](copilot-model.json).
Merge that provider into `chatLanguageModels.json`, preserving existing entries.
Select **Redshift Qwen3.8 27B (V100)** in the chat model picker. No API key is
required by this LAN endpoint.

The configuration declares the actual 139,264-token window, 122,880 input tokens
and 16,384 output tokens. The server counts the final rendered prompt including
chat-template and tool overhead and rejects requests that exceed that capacity.
Vision and thinking are disabled in this profile.
`temperature: 0` uses the fast GPU argmax path; sampling is available but slower.
This config selects the chat/agent model, not the inline-completion engine.

Reference: [VS Code custom endpoint configuration](https://code.visualstudio.com/docs/agent-customization/language-models#_add-a-custom-endpoint-model).

## Validation

- Complete native, HTTP and text suite: **63 tests passed, zero skipped**, on
  the V100 in 40.970 seconds after the final protocol refinements.
- CUDA memcheck: zero errors. Racecheck: zero errors or warnings.
- Tokenizer matched the installed native vocabulary-only oracle exactly:
  10 cases, 7,899 tokens, including Unicode, code, special tokens and tool history.
  `tools/tokenizer_reference.cpp` links existing native libraries for this check;
  it never loads model weights or creates an inference context.
- Real HTTP smoke covers JSON/SSE agreement, a streamed function call, a tool
  result followed by a natural-language answer, sampling, context rejection and
  generation across the former 2,048-token attention limit.

Final measurements and served-source hashes are in
[endpoint-results.json](endpoint-results.json). The code-generation case produced
164 completion tokens in 6.656 seconds: **24.64 tokens/s including prefill**, with
first content after 0.322 seconds. A cold 2,339-token prompt reached first content
after 16.888 seconds. GPU usage with the 32,768-position runtime was 27,408 MiB
(about 26.8 GiB). These are single-user smoke measurements, not a broad benchmark.

The local VS Code model configuration was merged and read back successfully,
with a timestamped backup of its previous contents. Copilot's observed
`No response was returned` was a client-side `compaction_static_context_blocked`:
static instructions and tools exceeded the usable context budget before any
inference request. The configuration now advertises the full actual window.
If the static payload still exceeds that budget, use Ask mode or fewer tools;
increasing the declared limit beyond runtime capacity is invalid.
An IDE conversation after this adjustment remains unverified.

Migration on October 3 used the installer in **offline mode**, with the existing
model and Python environment. The staged build ran 86 tests (14 GPU/optional
tests skipped), then the service passed health and a real inference probe.
The automatic 19 GB download was deliberately not repeated because the user has
limited Internet bandwidth. The pinned Hugging Face revision/checksum were
verified from its metadata, and download publication/checksum handling has local
regression tests.

Reproduce the actual endpoint check with:

```sh
python3 tools/smoke_endpoint.py --url http://10.10.10.55:8081 \
  --output /tmp/redshift-live-smoke.json
```

## Context increase on October 3, 2026

The Copilot profile now requests 122,880 input and 16,384 output tokens, a
139,264-token runtime capacity. The model metadata declares 262,144 positions;
the former 32,768 ceiling was a runtime validation limit. Checkpoint snapshots
are allocated on demand and attention snapshots contain only the saved prefix.
The endpoint never calls the public checkpoint API, so it does not reserve a
second full KV cache. This is necessary for this capacity on the 32 GiB V100.
The staged suite ran 87 tests: 86 passed and one optional tokenizer-oracle test
was skipped. The attention closed-form test covers position 139,263. The active
service reports context 139,264 and returned `REDSHIFT_OK` on a real inference
probe. GPU memory after that probe was 31,818 MiB used and 677 MiB free.
The full IDE Agent exchange is a separate verification, still pending.
[Context increase evidence](context139264-results.json).

## Current limits

One generation runs at a time; concurrent inference requests get HTTP 429.
Exact resident prefixes are reused, but unrelated conversations and changed
prefixes require prefill again. Long cold prompts can take tens of seconds;
SSE headers and keepalives are sent during prefill.

Text and function tools are supported. Tools are returned to the client and
never executed by the server. `tool_choice` accepts `auto` and `none`.
Structured JSON output, vision, penalties and logprobs are unsupported.
Tool parameters get structural type validation, not full JSON Schema validation.
Malformed completed tool output fails explicitly. Token-budget truncation is
reported as `length`, without exposing an incomplete executable tool call.

The runtime allocates 139,264 positions and the attention kernel is checked at
that boundary. Public checkpoint buffers are allocated only when requested,
with attention snapshots sized to the valid prefix; HTTP generation uses no
public checkpoint. The live text smoke exercises 2,339 prompt tokens; it does not
establish retrieval quality over the full window or general model-quality parity.
The VS Code picker and a full IDE agent session require separate UI verification.
