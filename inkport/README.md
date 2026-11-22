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
│   ├── src/{main,lib,desktop,app,contract,events,hardware,sidecar,api_server,runtime,project}.rs, bin/inkport-cli.rs
│   ├── src/bootstrap.rs           bundled payload: locate, verify (manifest pinned at build), isolated offline env, warm-up, accelerators
│   ├── src/packs.rs               optional add-on packs: resume-safe, checksum-verified, atomic installs
│   ├── bundle/<platform>.json     per-OS installer configs that map the payload into the app (release builds only)
│   └── entitlements.plist         macOS hardened-runtime exceptions for the embedded Python
├── src/                           React 19 + TypeScript + Tailwind 4
│   ├── App.tsx  lib/{contract,ipc,mock}.ts  components/ui.tsx
│   └── screens/{SystemReady,TemplateLauncher,DataRulesStudio,TrainingArena,DeployPanel}.tsx
├── tests/python/                  pytest: stats, rules, sandbox, SQL, ingest/leakage/contract, templates, slow e2e
├── packaging/                     lock.json (pinned runtime/llama.cpp/model inputs), requirements/, prune.json, packs/catalog.json
├── scripts/                       build_all_in_one.sh/.ps1, clean_machine_check.sh/.ps1, verify_e2e.sh/.ps1, make_icon.py
│   └── lib/                       assemble_payload.py, build_llamacpp.py, starter_model.py, finalize_payload.py, common.py
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

## Install (no build tools needed)

Ready-made installers are built by the **Inkport installers** GitHub Actions workflow; open its
latest successful run and download one artifact. Everything is inside: nothing else is downloaded,
and the app works offline.

| Your computer | Download | Then |
|---|---|---|
| Windows 10/11 (64-bit) | `Inkport-Windows-x64-Installer` | Unzip, run `Inkport_<version>_x64-setup.exe`. It installs for your user only (no administrator password) and adds Inkport to the Start menu. |
| Windows, no installer | `Inkport-Windows-x64-Portable` | Unzip, extract the inner zip anywhere, double-click `Inkport\Inkport.exe`. |
| Mac with Apple Silicon | `Inkport-macOS-AppleSilicon` | Unzip, open the `.dmg`, drag Inkport to Applications. |
| Linux (64-bit) | `Inkport-Linux-x64` | `chmod +x` the `.AppImage` and run it, or install the `.deb`. |

The builds are not signed with paid certificates yet, so the first launch needs one extra click:
on Windows choose **More info -> Run anyway** in the SmartScreen prompt; on macOS right-click the
app and choose **Open** (if macOS says the app is damaged, run `xattr -cr /Applications/Inkport.app`
once).

Good to know on Windows: about 2.1 GB on disk; training without an NVIDIA graphics card runs on the
processor (roughly 20-40 minutes for a short run on 200 examples, about 9 GB of memory); answers
use any Vulkan-capable graphics card and fall back to the processor on their own. Set
`INKPORT_ANSWERS_ON_CPU=1` to keep answers off the graphics card.

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

### One-and-done installers (everything included, works offline)

The installer is the only download. Inside it is everything needed to train, verify and serve
models with no network: a relocatable CPython (python-build-standalone) with the engine and its
libraries pre-installed, llama.cpp with this OS's GPU backends, a 4-bit starter model, sample data,
and an integrity manifest. Users never install Python, CUDA, compilers, Git or build tools, and
there is no setup wizard: the first launch verifies the files and lands on the ready screen.

| | Linux x86_64 | Windows x64 | macOS Apple Silicon |
|---|---|---|---|
| Installers | AppImage, `.deb`, portable `.tar.zst` (no root, no FUSE) | NSIS `.exe` (per-user, no admin), portable `.zip` | `.dmg`, `.app` |
| Training | PyTorch CPU (`cpu` flavor) or CUDA (`cuda` flavor) | PyTorch CPU or CUDA | MLX on the Apple GPU (no PyTorch shipped) |
| Inference | llama.cpp: Vulkan (NVIDIA/AMD/Intel) [+ CUDA], CPU fallback | llama.cpp: Vulkan [+ CUDA], CPU fallback | MLX; llama.cpp with Metal for GGUF |
| Starter model | Qwen3 1.7B, GGUF Q4_K_M (~1.1 GB): served by llama.cpp, dequantized in memory for training | same | Qwen3 1.7B, MLX 4-bit (~1 GB) |
| Never shipped | Windows/macOS binaries, CUDA in the `cpu` flavor | Linux/macOS binaries | CUDA, Vulkan, PyTorch |

Build on each target OS (the Python runtime can also be assembled cross-platform with uv):

```bash
scripts/build_all_in_one.sh
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_all_in_one.ps1
```

Options: `--flavor cuda` / `-Flavor cuda` for NVIDIA training, `--smoke-model` / `-SmokeModel` to
bundle the tiny offline test model (CI), `--debug`, `--skip-payload`.

Windows specifics:

- NSIS cannot hold more than 2 GiB. `src-tauri/nsis/installer.nsi` is Tauri's template with
  per-file instead of solid compression, so only the compressed total (about 1.4 GiB) has to fit.
  The MSI format has the same limit and no such switch, so the build defaults to
  `-Bundles nsis` in CI; a portable zip is always produced as well.
- `scripts/lib/pe_audit.py` reads the import table of every bundled `.exe`/`.dll`/`.pyd` and
  fails the build if one needs a Visual C++/OpenMP runtime DLL that is not shipped next to it
  (build machines have those DLLs system-wide, so a runtime test alone cannot catch this).
- llama.cpp is built in a short folder at the drive root (`<drive>:\ikb`), because the Vulkan
  shader generator otherwise exceeds the 260-character path limit. `-VulkanOptional` falls back
  to processor-only answers when the Vulkan SDK is missing or the backend does not build.

Pipeline:

1. `scripts/lib/assemble_payload.py`: pinned CPython (SHA-256 checked), dependencies installed
   into it, dead weight pruned (`packaging/prune.json`), every module precompiled.
2. `scripts/lib/build_llamacpp.py`: llama.cpp from pinned sources, with the per-OS backends above
   and `$ORIGIN`/`@loader_path` library paths. Built without any HTTPS client.
3. `scripts/lib/starter_model.py`: official safetensors from Hugging Face, each file verified,
   quantized by us (GGUF Q4_K_M or MLX 4-bit).
4. `scripts/lib/finalize_payload.py`: samples, pack catalog, licenses, and `manifest.json` with
   every file's size and SHA-256. The manifest's own SHA-256 is compiled into the app binary
   (`src-tauri/build.rs`).
5. Tauri packaging with `src-tauri/bundle/<platform>.json`, which maps the payload into the app as
   `payload/`. On Linux the AppImage gets the payload added untouched after linuxdeploy, because
   linuxdeploy would rewrite the libraries and break their checksums. On macOS every nested
   Mach-O is signed with the hardened runtime (`src-tauri/entitlements.plist`) before Tauri signs
   and notarizes the app. On Windows the MSVC runtime ships app-locally and the Rust app links its
   C runtime statically.

Every input is pinned in `packaging/lock.json`.

At launch, `src-tauri/src/bootstrap.rs`:

- Finds the payload and checks the manifest against the compiled-in hash.
- Hashes every file on first launch (multi-threaded; 5.2 GB took 1.8 s here) and checks sizes plus
  critical files afterwards.
- Runs the bundled interpreter isolated and in UTF-8 mode (`-I -B -X utf8`, user `PYTHON*`
  variables removed) with Hugging Face/transformers forced offline.
- Warms up the engine and detects accelerators: MLX/CUDA/CPU for training, and
  Metal/CUDA/Vulkan/CPU via `llama-server --list-devices` for answers.

A missing GPU is never an error.

**Optional add-ons** (`packaging/packs/catalog.json`, `src-tauri/src/packs.rs`) are larger base
models such as Qwen3 8B.
- **One click, resume-safe:** downloads continue with HTTP Range after a pause, crash or dropped
  connection.
- **Verified:** each file is checked against the SHA-256 or git blob SHA-1 the Hub publishes for
  the exact commit.
- **Atomic:** a pack appears only after every file has been verified.
- **Never required:** nothing in core training, verification or the local API uses them.

### Clean-machine check

On Linux or macOS:

```bash
scripts/clean_machine_check.sh build/installers/linux-x86_64-cpu/Inkport_0.1.0_amd64.AppImage
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\clean_machine_check.ps1 -Installer .\Inkport_0.1.0_x64-setup.exe
```

Add `-Quick` to skip training (about three minutes: install, integrity, practice exam, answers
from the bundled model). The `.exe`, the `.msi` and the portable `.zip` are all accepted.

The check installs or unpacks the installer without root and cuts off the network:
- Linux: a private network namespace with only loopback.
- macOS: `sandbox-exec`.
- Windows: Firewall rules when run as admin.

On Linux it also masks the host's python/pip/node/cargo/uv/cmake/gcc/git. It then checks:

1. Pre-flight (every file verified).
2. A project built from the bundled samples.
3. Training plus the hidden-test gate.
4. The local API: 401 without a key, model list, chat, `@base`, streaming.
5. A full re-verification.
6. That nothing was written inside the installation.

### Window check (the real window, used like a person would)

The clean-machine check drives the engine through the command line. The window check opens the
real app window and uses it, with the real engine behind it: create a project, build the exam from
the bundled samples, start training, leave the step and come back, read the verdict, start the
local API, chat with the trained model, stop a second run; then it closes the app, opens it again
and checks that the project, its exam and its trained version are still there. A last pass
(`goals`) takes every goal that ships sample data through "Try the sample data" with every
suggested rule, which must end in an exam that can be trained on.

```bash
# Linux: needs Xvfb, WebKitWebDriver (package webkit2gtk-driver), dbus-run-session and node
scripts/window_check.sh build/installers/linux-x86_64-cpu/Inkport_0.1.0_amd64.AppImage
WINDOW_PHASES="first reopen goals" scripts/window_check.sh ...     # a full training run, a restart, every goal
```

```powershell
# Windows: needs node; drives WebView2 through its remote debugging port
pwsh scripts\window_check.ps1 -Installer .\Inkport_0.1.0_x64-setup.exe
```

On Windows it also installs into, and keeps its data in, folders with spaces and non-Latin
characters, starts Inkport a second time (which must not open a second copy), ends a serving
process abruptly (no helper may be left running), and re-verifies the installation afterwards.
Both scripts run `scripts/gui_e2e.cjs`, which also works against the browser preview
(`--mode=preview`). Training in the window offers only the two real run types, which take hours
with a real model; for tests, `INKPORT_TEST_PRESET` and `INKPORT_TEST_OVERRIDES` (read by the
supervisor when a run starts) shorten it. macOS has no way to drive its webview from outside, so
CI only checks there that the window opens, runs its start-up check and stays open.

### Interface flow tests

```bash
pnpm dev --host 127.0.0.1 &
node scripts/ui_flows_test.cjs      # needs the playwright package with a Chromium
```

35 checks against the browser preview, whose simulated engine follows the real backend's rules:
leaving a step while training runs, Stop, a crash, a rejected run, too little data, an adapter
from an unknown publisher, reopening a trained project, a second project. Other outcomes can be
reviewed by hand with `?outcome=reject`, `?outcome=fail`, `?outcome=few` and `?untrusted=1`.

`node scripts/ui_monkey_test.cjs <seed> <steps>` clicks and types at random in the same preview
and fails on any page error, console error or empty screen (replayable by seed).

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
| Linux all-in-one installers (AppImage, `.deb`, portable tarball) | Built here (`cuda` flavor, starter = offline test model). All three pass `clean_machine_check.sh` offline with host toolchains masked (~80-95 s each). The installed desktop app was launched offline and reaches "ready" in about 4 s after verifying 25,689 files (5.2 GB). |
| Integrity / pre-flight / tamper detection | Verified: a 1-byte change in the model is reported as `checksum mismatch: models/starter/...`. |
| GGUF starter flow | Verified with a bundled GGUF: train from the dequantized GGUF, gate, export the GGUF LoRA with the bundled converter, then serve base + LoRA with the bundled llama-server, with `@base` switching the adapter off per request. |
| Rust tests | 15 passing (bootstrap integrity and pinning, packs resume and checksums against a fake Hub, gateway incl. llama upstream, contracts, tiers, projects) |
| Python tests | 41 fast + 1 slow, passing |
| **Not run here** | The real Qwen3 1.7B download and quantization (Hugging Face is blocked in this environment). The `cpu` flavor (the PyTorch CPU index is blocked; installers were ~3 GB here mainly because the `cuda` flavor carries ~3 GB of NVIDIA libraries, so expect roughly 1.5-2.5 GB for `cpu` with the real starter). The Windows and macOS builds, signing and notarization (no such machines here). GPU backends (no GPU here). |
| Not built yet | Cloud burst / swarm, OCR for scanned PDFs, teacher-model question drafting for documents. |

The bundled test model is ~1.4M parameters. It proves the plumbing, not quality. Real gains come from real backbones (the Qwen3 starter, or the optional 8B pack) on real data.
