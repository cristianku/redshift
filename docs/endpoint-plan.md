# Redshift Copilot endpoint implementation plan

Goal: run Redshift's own CUDA inference on llama-v100 and expose an OpenAI
Chat Completions endpoint usable from GitHub Copilot in VS Code.

The user explicitly requested this endpoint after clarification that the existing
port 8080 is llama.cpp. Use a separate port and process; preserve that service.
Work on main, no commit/push. Existing model weights remain read-only.

Architecture: Python HTTP server with one serialized GPU session, GGUF-derived
byte BPE tokenizer and embedded Jinja chat template, Redshift C ABI for prefill
and decoding. Pin tokenizers/Jinja2 in an isolated environment. No inference
proxy or fallback to llama.cpp. Context target 32768 tokens, bounded by actual
VRAM checks; extend attention without a context-sized shared-memory array.

API: GET /health, GET /v1/models, POST /v1/chat/completions (ordinary JSON or SSE).
Model ID redshift-qwen3.8-27b. Text chat and function calls; reject images and
unsupported modalities. Report truthful usage, finish reasons, context errors,
busy state and cancellation. Tool calls are returned to Copilot, never executed
by this server. Default non-thinking greedy generation; document measured speed.

## Work and ownership

- [x] Text codec and protocol parsing: qvelox/text.py, tests/test_text.py.
  TextCodec(metadata): encode(text), decode(ids), render(messages, tools=None,
  enable_thinking=False); eos_ids. parse_assistant(text, tools=None) produces an
  OpenAI assistant message with content/tool_calls/reasoning_content.
  DeltaParser(tools=None): feed(text) and finish() return lists of OpenAI deltas.
  Tests cover Unicode/byte boundaries, official template, tools and malformed calls.
- [x] Server and orchestration: qvelox/server.py, tests/test_server.py.
  Uses Runtime(model,context), evaluate(token_ids), advance(token_ids)->argmax IDs,
  reset(), position and close(). Prefill in chunks <=8, reuse exact resident token
  prefix across continuations. Stream real generated tokens and serialize sessions.
  Tests use a small deterministic runtime seam for HTTP/SSE, overflow, busy,
  disconnects, tool-result turns and errors. No fake production model path.
- [x] CUDA/runtime: src/kernels.*, src/runtime.*, qvelox/runtime.py,
  tests/test_cuda.py, tests/test_model.py. Extend capacity to32768 using tiled
  attention and preserve <=2048 arithmetic as a regression path. Add advance()
  which runs the same graph but downloads only GPU argmax IDs. Tests compare it
  with all-logit evaluation and long-context attention with an independent oracle.
- [x] Integrate and verify: requirements-server.txt, documentation, smoke tools.
  Compare tokenizer IDs to a vocabulary-only native reference; run CPU/full-model
  CUDA tests and sanitizer. Start a separate endpoint and exercise real text,
  streaming, tool call/result and continuation. Record startup/memory/speed.
  Provide the exact Copilot configuration with verified context/capabilities.

Review focus: wrong tokenizer IDs; byte-fragment corruption; malformed tool
arguments; state leaking between clients; prompts overflowing actual context;
disconnects holding the GPU lock; accidental llama.cpp inference fallback.

Execution: permitted local worker failed twice earlier in this session and other
local workers are prohibited. Use two scoped implementation workers under the
parallel-agent skill, while the primary owns CUDA and final live verification.
No simultaneous GPU tests or inference-worker loads.
