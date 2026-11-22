// Browser preview mode: a simulation of the Rust supervisor + engine event stream, used only when
// the UI runs outside Tauri (pnpm dev in a normal browser). It follows the real backend's rules
// closely (job kinds on events, BUSY guards, stop, a second gate event once the version is saved,
// project state that survives "reopening") so the screens behave here as they do in the app.
//
// URL switches for reviewing other outcomes: ?outcome=reject | fail | few   ?untrusted=1

import type { Channel } from "./contract";

type Cb = (p: unknown) => void;
const listeners = new Map<Channel, Set<Cb>>();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const params = new URLSearchParams(typeof location !== "undefined" ? location.search : "");
const OUTCOME = params.get("outcome") ?? "promote";
const SPEED = Number(params.get("speed") ?? "1") || 1;
// ?ram=8 simulates a Windows laptop without a graphics card for training and with that much memory.
const SMALL_PC_RAM = Number(params.get("ram") ?? "0") || 0;
let ts = 1_763_650_000;

function emit(channel: Channel, payload: unknown) {
  listeners.get(channel)?.forEach((cb) => cb(payload));
}

export function mockListen(channel: Channel, cb: Cb): () => void {
  if (!listeners.has(channel)) listeners.set(channel, new Set());
  listeners.get(channel)!.add(cb);
  return () => listeners.get(channel)?.delete(cb);
}

const hardware = {
  os: "macos",
  arch: "aarch64",
  cpu_brand: "Apple M3 Pro",
  cpu_cores: 12,
  ram_gb: 36,
  disk_free_gb: 412,
  apple_silicon: true,
  gpus: [{ vendor: "apple", name: "Apple M3 Pro GPU", vram_gb: 36, unified: true }],
  recommendation: { tier: "M32", vendor: "apple", memory_gb: 36, model_id: "qwen3-8b", model_label: "Qwen3 8B", backend: "mlx", quant: "mlx4", lora_r: 32, micro_batch: 2, grad_accum: 8, max_seq: 4096, stage2: "rft", rft_k: 8, plain: "Great: Apple Silicon Mac with plenty of memory" },
  summary: ["Apple Silicon Mac with 36 GB of shared memory", "36 GB of system memory", "412 GB free on disk", "Recommended AI brain: Qwen3 8B (Great: Apple Silicon Mac with plenty of memory)"],
};

const templates = [
  { id: "business_rules", title: "Small Business & Finance Rules", tagline: "Sort expenses, read invoices, and fill in fields the way your business does.", proof: "exact", proof_label: "Checked exactly against your own past records", accepts: [".xlsx", ".csv", ".tsv", ".jsonl", ".json", ".md", ".txt", ".pdf", ".docx"], needs: "A spreadsheet of past examples, Excel or CSV (for example with the columns: description, category, amount). Optional: a policy document.", example_rules: ["category must be one of fuel, meals, software, office_supplies, travel", "amount must be positive", "if the description mentions Uber then category must be travel"] },
  { id: "code_sql", title: "Code & SQL Specialist", tagline: "Answer questions about your spreadsheets with SQL, or write functions that pass your tests.", proof: "exact", proof_label: "Every answer is run and its result compared", accepts: [".xlsx", ".csv", ".tsv", ".py", ".jsonl", ".json"], needs: "One or more spreadsheets (Excel or CSV), or Python files with examples in their docstrings.", example_rules: ["must not mention DROP", "no more than 120 words"] },
  { id: "docs_legal", title: "Personal Docs & Legal Prep", tagline: "Answer questions from your leases, contracts, and paperwork, quoting the exact passage.", proof: "citation", proof_label: "Every answer must quote your document word for word", accepts: [".pdf", ".docx", ".md", ".txt"], needs: "The documents you want it to know (PDF, Word, Markdown or text).", example_rules: ["must include a quote", "no more than 80 words"] },
  { id: "custom", title: "Custom Skill", tagline: "Teach any skill from examples of questions and the answers you expect.", proof: "rules", proof_label: "Checked against your expected answers and your rules", accepts: [".xlsx", ".csv", ".tsv", ".jsonl", ".json"], needs: "A spreadsheet (Excel or CSV) of examples with a question column and an answer column.", example_rules: ["no more than 50 words", 'must mention "thank you"'] },
];

interface MockProject {
  id: string;
  slug: string;
  name: string;
  template: string;
  dir: string;
  base_model: { id: string; path: string };
  active_version: string | null;
  counts: { train: number; dev: number; test: number } | null;
  latest_report: Record<string, unknown> | null;
  inputs: string[];
  rules: Record<string, unknown>[];
}
const projects: MockProject[] = [];
let packInstalled = false;
let training: { jobId: string; projectId: string; cancelled: boolean; history: unknown[] } | null = null;
let apiFor: string | null = null;

const samples = [
  ["Paid $42 at Shell for the delivery van", '{"category": "fuel", "amount": 42}', '{"category": "travel"}', '{"category": "fuel", "amount": 42}'],
  ["Zoom receipt $15 monthly plan", '{"category": "software", "amount": 15}', "Sure! This looks like a software expense of about $15.", '{"category": "software", "amount": 15}'],
  ["Card charge: Staples $87", '{"category": "office_supplies", "amount": 87}', '{"category": "supplies", "amount": "87"}', '{"category": "office_supplies", "amount": 87}'],
  ["Reimburse $230 spent at Hilton", '{"category": "travel", "amount": 230}', '{"category": "meals", "amount": 230}', '{"category": "travel", "amount": 230}'],
];

const verifiers = [
  { id: "json_valid", kind: "json_valid", role: "gate", description: "Reply is a JSON object" },
  { id: "field:category", kind: "field_match", role: "reward", description: "category matches your record" },
  { id: "field:amount", kind: "field_match", role: "reward", description: "amount matches your record" },
];

const apiInfo = (p: MockProject) => ({ url: "http://127.0.0.1:11435/v1/chat/completions", base_url: "http://127.0.0.1:11435/v1", token: "ik_preview0000000000000000000000000000000000000000000000000000000000", model: `inkport/${p.slug}`, base_model: `inkport/${p.slug}@base`, version: p.active_version ?? "base", engine: "llama.cpp (Vulkan0: NVIDIA GeForce RTX 4060 (8188 MiB, 7700 MiB free))" });

function busy(message: string) {
  return { code: "BUSY", message, retryable: true };
}

async function simulateTraining(job: NonNullable<typeof training>, project: MockProject) {
  const ev = (o: Record<string, unknown>) => ({ v: 1, job_id: job.jobId, job_kind: "train", ts: ts++, project_id: job.projectId, ...o });
  const send = (channel: Channel, payload: Record<string, unknown>) => {
    if (["stage", "progress", "eval", "gate", "benchmark", "error"].includes(String(payload.type))) {
      if (payload.type === "progress") job.history = job.history.filter((e) => (e as { type: string }).type !== "progress");
      job.history.push(payload);
    }
    emit(channel, payload);
  };
  const finish = (ok: boolean, result: Record<string, unknown>) => {
    if (training === job) training = null;
    emit("training://progress", ev({ type: "done", ok, result }));
  };
  const stages: [string, string, number, number][] = [
    ["verify_data", "Checking that no hidden test question leaked into training", 0, 2],
    ["baseline", "Measuring the original model on the hidden test", 2, 18],
    ["sft", "Learning from 196 examples", 22, 70],
    ["rft1", "Round 1: practicing and keeping only answers that pass your checks", 70, 84],
    ["gate", "Final exam: original vs trained on hidden questions", 84, 98],
  ];
  const incumbent = project.active_version;
  for (const [stage, detail, o0, o1] of stages) {
    send("training://progress", ev({ type: "stage", stage, status: "started", detail }));
    const steps = stage === "sft" ? 24 : 6;
    for (let i = 1; i <= steps; i++) {
      await sleep((stage === "sft" ? 260 : 150) / SPEED);
      if (job.cancelled) {
        emit("training://progress", ev({ type: "error", code: "CANCELLED", message: "Stopped. Nothing was saved from this run.", retryable: true }));
        return finish(false, { code: "CANCELLED" });
      }
      if (OUTCOME === "fail" && stage === "sft" && i === 9) {
        emit("training://progress", ev({ type: "error", code: "OUT_OF_MEMORY", message: "Your computer ran out of memory while training. Close other programs (web browsers use a lot of memory) and try again. If it keeps happening, restart the computer first, or use Quick run.", retryable: true }));
        return finish(false, { code: "OUT_OF_MEMORY" });
      }
      const dev = stage === "sft" ? Math.min(0.84, 0.05 + i * 0.035) : stage === "rft1" ? 0.84 + i * 0.01 : null;
      send("training://progress", ev({ type: "progress", stage, step: i, total: steps, pct: (100 * i) / steps, overall_pct: o0 + ((o1 - o0) * i) / steps, eta_s: Math.max(0, (24 - i) * 9 + (stage === "sft" ? 120 : 40)), metrics: { loss: stage === "sft" ? 2.1 * Math.exp(-i / 8) + 0.18 : 0.2, lr: 2e-4, tok_s: 2140 + i * 3, dev_pass: dev, replay_dnll: 0.02 + i * 0.001 } }));
      if (stage === "sft" && i % 6 === 0) {
        const s = samples[(i / 6 - 1) % samples.length];
        send("benchmark://result", ev({ type: "benchmark", source: "training_dev", case_id: `c${i}`, prompt: s[0], reference: s[1], baseline: { output: s[2], pass: false, score: 0 }, trained: { output: i >= 12 ? s[3] : '{"category": "fuel"}', pass: i >= 12, score: i >= 12 ? 1 : 0.5 }, step: i * 20, judge: null }));
        emit("training://log", ev({ type: "log", level: "info", message: `sft step ${i * 20}/480: dev pass ${Math.round((dev ?? 0) * 100)}%, drift +0.031 nats/token` }));
      }
    }
    if (stage === "baseline") send("training://progress", ev({ type: "eval", split: "test", model: "base", n: 120, passed: 38, pass_rate: 0.317, ci95: [0.24, 0.4] }));
    send("training://progress", ev({ type: "stage", stage, status: "completed", detail: stage === "baseline" ? "Original model: 38/120 correct" : stage === "gate" ? (OUTCOME === "reject" || incumbent ? "REJECT" : "PROMOTE") : "" }));
    if (stage === "rft1") send("training://progress", ev({ type: "stage", stage: "stage2", status: "completed", detail: "" }));
  }
  const reject = OUTCOME === "reject" || !!incumbent;
  const cand = reject ? (incumbent ? 0.867 : 0.333) : 0.867;
  const ref = incumbent ? 0.867 : 0.317;
  send("training://progress", ev({ type: "eval", split: "test", model: "candidate", n: 120, passed: Math.round(cand * 120), pass_rate: cand, ci95: [0.79, 0.92] }));
  const explanation = reject
    ? `Your trained model passed ${Math.round(cand * 100)}% of the 120 hidden test questions, versus ${Math.round(ref * 100)}% for the ${incumbent ? `current version ${incumbent}` : "base"} (${((cand - ref) * 100 >= 0 ? "+" : "") + ((cand - ref) * 100).toFixed(1)} points). Not promoted because the improvement is too small or not statistically reliable yet. ${incumbent ? "Your current version stays active." : "Nothing was saved, and the original model is unchanged."}`
    : "Your trained model passed 87% of the 120 hidden test questions, versus 32% for the base (+55.0 points). That difference is statistically solid (95% range +45.8 to +63.3 points). General skills: no meaningful drop.";
  const gate = { type: "gate", decision: reject ? "REJECT" : "PROMOTE", label: "VERIFIED", delta_pts: (cand - ref) * 100, baseline_pass_rate: ref, candidate_pass_rate: cand, checks: { task: !reject, general: true, format: true, audit: true, leakage: true }, report_path: `${project.dir}/runs/run-preview/report.json`, explanation };
  send("training://progress", ev({ ...gate, version: null }));
  if (!reject) {
    send("training://progress", ev({ type: "stage", stage: "promote", status: "started", detail: "Saving your new version" }));
    await sleep(900 / SPEED);
    project.active_version = "v1";
    project.latest_report = { decision: "PROMOTE", label: "VERIFIED", explanation, promoted_version: "v1", compared_against: "base", test: { n: 120, baseline_pass_rate: ref, candidate_pass_rate: cand, delta_pts: (cand - ref) * 100 } };
    send("training://progress", ev({ ...gate, version: "v1", report_path: `${project.dir}/versions/v1/report.json` }));
    send("training://progress", ev({ type: "stage", stage: "promote", status: "completed", detail: "v1" }));
  } else if (!incumbent) {
    project.latest_report = { decision: "REJECT", label: "VERIFIED", explanation, compared_against: "base", test: { n: 120, baseline_pass_rate: ref, candidate_pass_rate: cand, delta_pts: (cand - ref) * 100 } };
  }
  finish(true, { decision: gate.decision, explanation });
}

const view = (p: MockProject) => ({ id: p.id, slug: p.slug, name: p.name, template: p.template, dir: p.dir, base_model: p.base_model, active_version: p.active_version, counts: p.counts, latest_report: p.latest_report });
function find(id: unknown): MockProject {
  const p = projects.find((x) => x.id === id || x.slug === id);
  if (!p) throw { code: "ERROR", message: `project '${String(id)}' not found`, retryable: false };
  return p;
}

export async function mockInvoke<T>(cmd: string, args: Record<string, unknown> = {}): Promise<T> {
  await sleep(250 / SPEED);
  switch (cmd) {
    case "detect_hardware":
      if (SMALL_PC_RAM) return { ...hardware, os: "windows", arch: "x86_64", cpu_brand: "Intel Core i5", cpu_cores: 4, ram_gb: SMALL_PC_RAM, apple_silicon: false, gpus: [], summary: [`Intel Core i5, ${SMALL_PC_RAM} GB memory`] } as T;
      return hardware as T;
    case "preflight": {
      for (let i = 1; i <= 10; i++) {
        emit("runtime://progress", { type: "preflight", phase: "verify", done: i * 190_000_000, total: 1_900_000_000, pct: i * 10 });
        await sleep(90 / SPEED);
      }
      emit("runtime://progress", { type: "preflight", phase: "warmup", done: 1, total: 1, pct: 100 });
      return { ok: true, bundled: true, offline_core: true, platform: "macos-aarch64", flavor: "mlx", app_version: "0.1.0", engine_version: "0.1.0", python: "/Applications/Inkport.app/Contents/Resources/payload/python/bin/python3", training_accelerator: SMALL_PC_RAM ? "Processor (CPU)" : "Apple GPU (MLX)", inference_devices: ["MTL0: Apple M3 Pro (27648 MiB, 27647 MiB free)"], starter_model: { id: "qwen3-1.7b", label: "Qwen3 1.7B (included)", format: "mlx", license: "Apache-2.0", size_mb: 968, params_b: 1.7 }, verify: { ok: true, mode: "full", files_checked: 11842, bytes_checked: 1_900_000_000, seconds: 1.4, manifest_pinned: true, problems: [] }, seconds: 2.6, problems: [] } as T;
    }
    case "list_models":
      return [{ id: "qwen3-1.7b", label: "Qwen3 1.7B (included)", path: "", format: "mlx", bundled: true }, ...(packInstalled ? [{ id: "qwen3-8b", label: "Qwen3 8B (add-on)", path: "", format: "mlx", pack: "qwen3-8b-mlx" }] : [])] as T;
    case "list_packs":
      return [
        { id: "qwen3-8b-mlx", title: "Larger brain: Qwen3 8B", description: "Smarter answers and better results on harder tasks. Best on Macs with 24 GB of memory or more.", kind: "model", approx_size_mb: 4620, license: "Apache-2.0", installed: packInstalled, installing: false, partial_bytes: 0, path: null },
        { id: "qwen3-4b-instruct-mlx", title: "Balanced brain: Qwen3 4B Instruct", description: "A middle ground between the included model and the 8B add-on. Good on Macs with 16 GB of memory.", kind: "model", approx_size_mb: 2260, license: "Apache-2.0", installed: false, installing: false, partial_bytes: 0, path: null },
      ] as T;
    case "install_pack": {
      const id = String(args.packId);
      void (async () => {
        for (let i = 1; i <= 20; i++) {
          await sleep(150 / SPEED);
          emit("packs://progress", { type: "pack", pack_id: id, status: "downloading", file: "model.safetensors", done: i * 231_000_000, total: 4_620_000_000, pct: i * 5, message: "" });
        }
        packInstalled = true;
        emit("packs://progress", { type: "pack", pack_id: id, status: "installed", file: "", done: 1, total: 1, pct: 100, message: "" });
      })();
      return undefined as T;
    }
    case "cancel_pack":
    case "remove_pack":
      return true as T;
    case "sample_data":
      return "/Applications/Inkport.app/Contents/Resources/payload/samples" as T;
    case "runtime_status":
      return { ready: false, backends: {}, home: "~/Library/Application Support/Inkport", message: "Not set up yet" } as T;
    case "bootstrap_runtime": {
      const steps = ["Creating a private Python 3.12", "Installing PyTorch", "Installing the Inkport engine", "Checking everything works", "Downloading Qwen3 8B"];
      for (let i = 0; i < steps.length; i++) {
        emit("runtime://progress", { type: "runtime", step: i + 1, total: steps.length, label: steps[i], status: "running", detail: "", pct: (100 * i) / steps.length });
        await sleep((i === 4 ? 1400 : 600) / SPEED);
        emit("runtime://progress", { type: "runtime", step: i + 1, total: steps.length, label: steps[i], status: "completed", detail: i === 1 ? "not needed on Apple Silicon (uses MLX)" : "", pct: (100 * (i + 1)) / steps.length });
      }
      return { ready: true, engine_version: "0.1.0", backends: { mlx: true, torch: false }, home: "~/Library/Application Support/Inkport", message: "Ready" } as T;
    }
    case "list_templates":
      return templates as T;
    case "list_projects":
      return projects.map(view) as T;
    case "create_project": {
      const req = args.request as { name: string; template: string; base_model_id?: string | null };
      const name = req.name.trim();
      if (projects.some((p) => p.name.toLowerCase() === name.toLowerCase())) throw { code: "PROJECT_EXISTS", message: `A project named '${name}' already exists. Open it from the list of projects, or choose another name.`, retryable: false };
      let slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "project";
      for (let n = 2; projects.some((p) => p.slug === slug); n++) slug = `${slug.replace(/-\d+$/, "")}-${n}`;
      const p: MockProject = { id: `p${projects.length + 1}`, slug, name, template: req.template, dir: "~/Inkport/projects/" + slug, base_model: { id: req.base_model_id ?? "qwen3-1.7b", path: "" }, active_version: null, counts: null, latest_report: null, inputs: [], rules: [] };
      projects.push(p);
      return view(p) as T;
    }
    case "get_project":
      return view(find(args.projectId)) as T;
    case "compile_rules": {
      const texts = (args.texts as string[]) ?? [];
      return { rules: texts.map((t) => (/one of|positive|words|mention|sum|required|between/.test(t) ? { ok: true, text: t, expr: "…", explanation: t.replace("must be", "→"), error: "", self_test: { references: 50, accepted: 50, mutants: 40, caught: 38, discrimination: 0.95, ok: true } } : { ok: false, text: t, expr: "", explanation: "", error: "I couldn't turn that into a check. Try phrasing like: “category must be one of A, B, C”, “amount must be positive”, or “no more than 50 words”." })) } as T;
    }
    case "ingest_dataset": {
      const p = find(args.projectId);
      if (training?.projectId === p.id) throw busy("Training is running for this project. Stop it, or wait for it to finish, before rebuilding the practice exam.");
      const ev = (o: Record<string, unknown>) => ({ v: 1, job_id: "job-ingest", job_kind: "ingest", ts: ts++, project_id: p.id, ...o });
      for (const s of ["parse", "build_items", "rules", "split"]) {
        emit("training://progress", ev({ type: "stage", stage: s, status: "started", detail: "" }));
        await sleep(300 / SPEED);
        emit("training://progress", ev({ type: "stage", stage: s, status: "completed", detail: "" }));
      }
      emit("training://progress", ev({ type: "done", ok: true, result: {} }));
      const few = OUTCOME === "few";
      p.counts = few ? { train: 7, dev: 0, test: 5 } : { train: 286, dev: 40, test: 120 };
      p.inputs = (args.paths as string[]) ?? [];
      // The sample has no "price" column: such a rule reads fine but cannot be used with these files.
      p.rules = ((args.rules as string[]) ?? []).map((t) =>
        /\bprice\b/i.test(t)
          ? { ok: false, text: t, expr: "", explanation: "", error: "There is no column called “price” in your examples. The columns your AI fills in are: category, amount.", depends_on_data: true }
          : { ok: true, text: t, expr: "…", explanation: t, error: "" },
      );
      (window as unknown as { __lastRulesSent?: string[] }).__lastRulesSent = (args.rules as string[]) ?? [];
      const blocker = few ? "Only 12 different examples were found. Inkport needs at least 30 to set aside a fair final exam (100 or more gives a result you can trust). Add more examples and build the exam again." : null;
      return { counts: p.counts, leakage_free: true, duplicates_removed: few ? 0 : 7, warnings: [], rules: p.rules, verifiers, seconds: 1.3, ready: !few, blocker } as T;
    }
    case "preview_items": {
      const p = find(args.projectId);
      const few = OUTCOME === "few";
      return { items: p.counts ? samples.map((s, i) => ({ item_id: `i${i}`, split: "dev", prompt: s[0], reference: s[1], source: `expenses.csv:${12 + i}` })) : [], counts: p.counts ?? { train: 0, dev: 0, test: 0 }, test_locked: true, warnings: [], verifiers, inputs: p.inputs, rules: p.rules, duplicates_removed: few ? 0 : 7, leakage_free: true, ready: !!p.counts && !few, blocker: null } as T;
    }
    case "start_training_job": {
      const req = args.request as { project_id: string };
      const p = find(req.project_id);
      if (training) throw busy(training.projectId === p.id ? "Training is already running for this project." : "Training is already running for another project. Stop it there, or wait for it to finish, then start this one.");
      training = { jobId: `job-${Math.random().toString(16).slice(2, 10)}`, projectId: p.id, cancelled: false, history: [] };
      void simulateTraining(training, p);
      return training.jobId as T;
    }
    case "cancel_training_job":
      if (training && training.jobId === args.jobId) training.cancelled = true;
      return undefined as T;
    case "training_status":
      return (training ? { job_id: training.jobId, kind: "train", project_id: training.projectId, events: [...training.history] } : null) as T;
    case "run_benchmark_arena": {
      if (training) throw busy("Training is running. You can compare the two models when it has finished.");
      const prompt = (args.prompt as string) || samples[0][0];
      await sleep(500 / SPEED);
      emit("benchmark://result", { v: 1, job_id: "job-arena", job_kind: "arena", ts: ts++, project_id: args.projectId, type: "benchmark", source: "arena", case_id: `a${ts}`, prompt, reference: null, baseline: { output: "This appears to be a fuel purchase.", pass: false, score: 0 }, trained: { output: '{"category": "fuel", "amount": 42}', pass: true, score: 1 }, step: null, judge: null });
      return { cases: 1, wins: { trained: 1, baseline: 0, tie: 0 } } as T;
    }
    case "start_local_api": {
      const p = find(args.projectId);
      await sleep(700 / SPEED);
      apiFor = p.id;
      return apiInfo(p) as T;
    }
    case "stop_local_api":
      apiFor = null;
      return undefined as T;
    case "api_status": {
      const p = projects.find((x) => x.id === apiFor);
      return (p ? { running: true, project_id: p.id, info: apiInfo(p) } : { running: false, project_id: null, info: null }) as T;
    }
    case "chat_local_api": {
      if (!apiFor) throw { code: "API_STOPPED", message: "The local API is not running. Press “Start local API” first.", retryable: true };
      await sleep(600 / SPEED);
      const last = ((args.messages as { content: string }[]) ?? []).slice(-1)[0]?.content ?? "";
      const amount = /\$?(\d+(?:\.\d+)?)/.exec(last)?.[1] ?? "0";
      return `{"category": "fuel", "amount": ${amount}}` as T;
    }
    case "export_adapter":
      await sleep(400 / SPEED);
      return { path: String(args.dest), sha256: "33b93bf6277b4edb34b501ca3ded0d7dd088c60bbccf84cb59c95c40a00d1ea0", version: "v1", publisher: "1b8502ca92f1…" } as T;
    case "import_adapter": {
      await sleep(900 / SPEED);
      if (params.get("untrusted") && !args.trust) throw { code: "UNTRUSTED_PUBLISHER", message: "UNTRUSTED_PUBLISHER:9f2c41d07a5e1b8802c7", retryable: false };
      return { import: { signature_verified: true, sha256_verified: true, publisher: "1b8502ca92f1…", checks: { tensors: 112, rank: 32 } }, gate: { decision: "REJECT", explanation: "Your trained model passed 87% of the 120 hidden test questions, versus 87% for the current version v1 (+0.0 points). Not promoted because the improvement is too small or not statistically reliable yet. Your current version stays active." } } as T;
    }
    default:
      throw { code: "UNKNOWN", message: `preview mode has no mock for ${cmd}`, retryable: false };
  }
}
