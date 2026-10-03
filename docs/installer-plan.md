# Redshift server installer

Goal: reproduce the Redshift endpoint on an empty Ubuntu 24.04 x86_64 server,
and move the current installation from llama-v100 to redshift-v100 (10.10.10.55).
The user explicitly requests both installation and removal from the former host.
Work on main, preserve dirty files, no commits or pushes.

Architecture: a shell bootstrap and standard-library Python installer. Reuse
working NVIDIA drivers; install the proprietary R580 driver only on bare metal
when missing. Pin CUDA Toolkit 12.9 for Volta. In containers, require GPU access
from the host. Never configure passthrough or reboot hosts automatically.

The installer consumes its own repository checkout. Build into a new release
under /opt/redshift, use an isolated Python environment, validate the GGUF and
run CPU tests before activating a dedicated redshift systemd service/user.
Keep the previous release and service definition for rollback on failed health
or inference checks. Refuse unmanaged install directories or service files.

Model input: automatic download from ggml-org/Qwen3.8-27B-GGUF at pinned revision
97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827, or --model FILE / --model-url URL.
Verify SHA-256; the default digest is the measured existing Q4_K_M artifact and
matches that public Hugging Face revision. Reuse local weights when present.
Do not overwrite existing weights. Emit a complete Copilot model configuration.

- [x] Installer scripts: packages, CUDA/driver preflight, model validation,
      source staging, compilation, venv, tests, service, health, rollback.
- [x] Focused tests: unmanaged targets, invalid arguments, download checksum,
      safe service quoting, failed activation rollback, dry-run without writes.
- [x] Deploy with this installer OFFLINE on redshift-v100 using existing read-only GGUF.
- [x] Verify real HTTP and systemd with existing weights; exercise safeguards locally.
- [x] Update local Copilot URL to 10.10.10.55 and actual context32768.
- [x] Document one-command installation and actual checks/limitations.

User explicitly forbids a fresh/full installation test that downloads the model
again. No such test performed. API metadata verifies the pinned source; local
tests cover automatic selection, checksums and publication. UI agent testing,
full model download and bare-metal driver provisioning are not claimed verified.

Removal already completed: redshift-poc stopped and its transient unit and
/tmp/redshift-serve-YpjhmsyU removed from llama-v100; port8081 closed and llama
still active. A backup is kept only in local build/migration-backup.
Earlier authorized benchmark artifacts are not part of that endpoint installation.
