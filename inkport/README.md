# Inkport

Train a small, private, specialist AI on your own computer, and **prove** it got better before it is ever used.

```
React UI (src/) ──Tauri IPC──► Rust supervisor (src-tauri/) ──NDJSON over stdio──► Python engine (engine/)
   5 screens                    hardware · runtime bootstrap                       ingest · verifiers · LoRA/QLoRA/DoRA
   events: training://*         job supervision · watchdog                         SFT → RFT/GRPO → before-vs-after gate
           benchmark://result   axum gateway 127.0.0.1:11435/v1  ◄──HTTP+secret──  inference worker (transformers/MLX/llama.cpp)
```

The shared contract lives in `contracts/`. Rust, Python and TypeScript all read it, and all three test suites validate against the same fixtures.

## Repository layout

```
inkport/
├── contracts/                     single source of truth
│   ├── schemas/event.schema.json  NDJSON engine→supervisor protocol (12 event types)
│   ├── schemas/item.schema.json   dataset rows (train/dev/test.jsonl)
│   ├── schemas/verifier.schema.json  verifiers.json
│   ├── schemas/hardware.schema.json  HardwareReport + Recommendation
│   ├── schemas/plan.schema.json   TrainingPlan
│   ├── tiers.json                 hardware → model / quant / LoRA rank / batch / context
│   ├── templates.json             the 4 starter templates
│   └── fixtures/                  events.ndjson, tier_cases.json (tested in Rust and Python)
├── engine/                        Python sidecar (`python -m engine <cmd>`)
│   ├── cli.py · events.py         commands, NDJSON protocol, heartbeats, parent-death watch
│   ├── hardware_profile.py        tier router, memory model, auto-downshift, plan builder
│   ├── pipeline/ingest_and_verify.py  files → items → dedup → hidden test split → verifiers
│   ├── pipeline/train_and_gate.py     baseline → replay → SFT → RFT/GRPO → gate → promote/rollback
│   ├── pipeline/arena.py          original vs trained, side by side
│   ├── verify/                    json/schema/field/rule(CEL-like safe DSL)/SQL/sandboxed tests/citation/judge
│   ├── backends/                  torch (+peft, bitsandbytes, Unsloth, TRL GRPO) · mlx · llama.cpp
│   ├── inference_worker.py        OpenAI-style worker (streaming), proxied by the Rust gateway
│   ├── adapters.py                signed .inkpatch export/import, MLX→PEFT, GGUF export
│   └── smoke.py                   offline tiny model + synthetic data for the e2e test
├── src-tauri/                     Rust: Tauri v2 shell + headless CLI
│   └── src/{main,lib,desktop,app,contract,events,hardware,sidecar,api_server,runtime,project}.rs, bin/inkport-cli.rs
├── src/                           React 19 + TypeScript + Tailwind 4
│   ├── App.tsx  lib/{contract,ipc,mock}.ts  components/ui.tsx
│   └── screens/{SystemReady,TemplateLauncher,DataRulesStudio,TrainingArena,DeployPanel}.tsx
├── tests/python/                  pytest: stats, rules, sandbox, SQL, ingest/leakage/contract, templates, slow e2e
├── scripts/                       verify_e2e.sh / .ps1, prepare_bundle.sh, make_icon.py
├── package.json · pyproject.toml · Cargo.toml (workspace) · src-tauri/tauri.conf.json
```

## The IPC contract in one page

**Tauri commands** (`src/lib/ipc.ts` ↔ `src-tauri/src/desktop.rs`):

- `detect_hardware`, `runtime_status`, `bootstrap_runtime{downloadModel}`
- `list_templates`, `create_project{request}`, `list_projects`, `get_project`
- `ingest_dataset{projectId, paths, rules}`, `preview_items`, `compile_rules`
- `start_training_job{request}` → job id, `cancel_training_job{jobId}`
- `run_benchmark_arena{projectId, prompt?, n}`
- `start_local_api{projectId}` → `{base_url, token, model}`, `stop_local_api`
- `export_adapter`, `import_adapter{trust}`

Errors come back as `{code, message, retryable}`.

**Event channels:**

| Channel | Carries |
|---|---|
| `training://progress` | stage, progress (overall %, ETA, loss, dev pass, drift), eval, gate, artifact, error, done |
| `training://log` | log |
| `benchmark://result` | benchmark (original vs trained, each with verifier pass/score) |
| `runtime://progress` | setup steps |
| `api://status` | API state |

Every engine line is `{"v":1,"type":…,"job_id":…,"ts":…}` plus the type's fields. The supervisor adds `job_kind` and `project_id`.

**Local API** (`http://127.0.0.1:11435/v1`): `GET /models`, `POST /chat/completions` (with `"stream": true` support).

- It needs `Authorization: Bearer <token>`. The token lives in `<home>/api_token`, and the app shows it.
- The Host header must be loopback; other Hosts are refused (DNS-rebinding protection).
- Browser Origins are refused unless they are the app itself or a browser extension.
- Model ids are `inkport/<project>` (active version) and `inkport/<project>@base`.

## How the guarantees are enforced

| Claim | Mechanism (file) |
|---|---|
| No test leakage | Exact dedup by SHA-256 of the normalized prompt. MinHash-LSH near-duplicate clusters, plus template group hints (e.g. paraphrases of one SQL question), keep related items in the same split. The split is decided by HMAC-SHA256(project salt, group). `train` refuses to run if any train/dev item or group overlaps the test set, or if the test set's Merkle root changed. (`ingest_and_verify.py`, `train_and_gate.py::verify_data`) |
| Only better models ship | The candidate is compared with the active version (or base) on the hidden test. Promotion requires +2 pts, plus a paired-bootstrap 95% lower bound > 0 (≥100 items) or exact McNemar p < 0.05 (<100, marked *preliminary*). It also requires no significant general-skill drop, no format regression, and audit verifiers passing. Anything else keeps the current version (automatic rollback). (`train_and_gate.py::gate`) |
| Catastrophic forgetting guard | Self-replay: the base model's own answers to general prompts are mixed into SFT. A held-out replay slice measures drift (ΔNLL); exceeding the limit rolls back to the best checkpoint and halves the learning rate, and a second trip stops training. A general-skills suite runs in the gate. |
| Crash resume | `runs/<id>/state.json` records the stages already completed, and checkpoints (adapter + optimizer) are written atomically every N steps. `train --resume <run_id>` continues; the supervisor's watchdog kills stalled jobs. |
| Untrusted code can't escape | Rules are parsed with `ast` and run by a whitelisting interpreter (no `eval`). Model-written Python goes through an AST screen, then a separate `python -I -S` process with PEP 578 audit hooks (no file writes, reads outside the stdlib, sockets, subprocesses or ctypes), POSIX rlimits and a timeout. SQL runs on a read-only, immutable SQLite with an authorizer that allows only SELECT/READ/FUNCTION, and a step limit. |
| Shared adapters can't poison you | A `.inkpatch` is Ed25519-signed with a SHA-256 for every file. Import checks zip safety, signature, publisher trust, adapter config (no `modules_to_save` or custom code), and every tensor's name, shape, dtype and finiteness against your base model. The adapter then has to win **your** hidden-test gate before activation. |

## Setup

Prerequisites (all platforms): Rust ≥ 1.80, Node 22 + pnpm, [uv](https://docs.astral.sh/uv/), plus the Tauri v2 system dependencies:

- **Linux:** `libwebkit2gtk-4.1-dev build-essential libssl-dev libayatana-appindicator3-dev librsvg2-dev`
- **macOS:** Xcode Command Line Tools
- **Windows:** MSVC Build Tools + WebView2 (preinstalled on Windows 11)

### Run the app in development

First create the engine environment (CPU PyTorch shown; on NVIDIA use `--extra cuda`, on Apple Silicon `--extra mlx`):

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

Then point the app at it and start Tauri:

```bash
pnpm install
```

```bash
INKPORT_PYTHON=$PWD/.venv/bin/python pnpm tauri dev
```

Without `INKPORT_PYTHON`, the app's **Set up everything** button builds its own private runtime with uv. It picks CUDA/ROCm/XPU/CPU PyTorch or MLX for the detected hardware and downloads the recommended Qwen3 model.

`pnpm dev` alone opens the UI in a browser with a simulated engine. That is useful for UI work only.

### Package installers

```bash
pnpm tauri build
```

This produces `.deb`/`.rpm`/AppImage on Linux, `.dmg`/`.app` on macOS and `.msi`/NSIS on Windows. `scripts/prepare_bundle.sh` stages the engine wheel and the `uv` binary as resources. Installers must be built on each target OS; code signing and notarization need your own certificates.

### Verify everything end to end

On Linux or macOS:

```bash
scripts/verify_e2e.sh
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_e2e.ps1
```

The script:

1. Runs the Python and Rust test suites.
2. Builds the headless `inkport-cli`, the same supervisor code the desktop app uses.
3. Builds the tiny offline model and synthetic small-business data.
4. Through the Rust CLI: ingests, trains, gates (expects PROMOTE), and runs the arena.
5. Serves the model and checks the API: 401 without a key, 403 for a hostile Host/Origin, a JSON chat answer, and SSE streaming.
6. Exports a signed adapter, checks that a tampered copy is rejected, and re-imports the good one through the gate.
7. Builds the frontend.

It runs on CPU in about 2 minutes, plus about 1 minute the first time to build the tiny model.

## Status

| Area | State |
|---|---|
| Linux, CPU: engine, Rust supervisor/CLI, gateway, desktop app, `.deb` | Built and verified here (e2e script green; desktop app driven through real IPC under Xvfb) |
| Python tests | 39 fast + 1 slow end-to-end, passing |
| Rust tests | 8, passing (contract fixtures, tiers, nvidia-smi parsing, gateway security + streaming proxy, projects) |
| NVIDIA CUDA (bitsandbytes QLoRA, Unsloth, TRL GRPO) | Implemented, **not run in this environment** (no GPU). Validate with `verify_e2e` on a CUDA box. |
| Apple Silicon MLX (training, adapters, MLX→PEFT) | Implemented against the mlx-lm API, **not run here** (no Mac). |
| Windows | Code paths and `verify_e2e.ps1` written, **not run here**. |
| llama.cpp GGUF export / inference | Optional; set `INKPORT_LLAMA_CPP` to a llama.cpp checkout for export, install `.[llamacpp]` for inference. |
| Not built yet | Cloud burst / swarm, OCR for scanned PDFs, teacher-model question drafting for documents, Windows sandbox job objects. |

The smoke model is ~1.4M parameters. It proves the plumbing, not quality: it reaches roughly 5–15% on the hidden expense test versus 0% for its base. Real gains come from real backbones (Qwen3 4B–30B) on real data.
