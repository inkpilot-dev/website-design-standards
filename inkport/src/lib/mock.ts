// Browser preview mode: a faithful simulation of the Rust supervisor + engine event stream.
// Used only when the UI runs outside Tauri (pnpm dev in a normal browser).

import type { Channel } from "./contract";

type Cb = (p: unknown) => void;
const listeners = new Map<Channel, Set<Cb>>();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let ts = 1_763_650_000;
const ev = (o: Record<string, unknown>) => ({ v: 1, job_id: "job-preview", ts: ts++, project_id: "p1", ...o });

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
  { id: "business_rules", title: "Small Business & Finance Rules", tagline: "Sort expenses, read invoices, and fill in fields the way your business does.", proof: "exact", proof_label: "Checked exactly against your own past records", accepts: [".csv", ".jsonl", ".json", ".md", ".txt", ".pdf"], needs: "A spreadsheet of past examples (for example: description, category, amount). Optional: a policy document.", example_rules: ["category must be one of fuel, meals, software, office_supplies", "amount must be positive", "total must equal the sum of the line items"] },
  { id: "code_sql", title: "Code & SQL Specialist", tagline: "Answer questions about your spreadsheets with SQL, or write functions that pass your tests.", proof: "exact", proof_label: "Every answer is run and its result compared", accepts: [".csv", ".py", ".jsonl", ".json"], needs: "One or more spreadsheets (CSV), or Python files with examples in their docstrings.", example_rules: ["must not mention DROP", "no more than 120 words"] },
  { id: "docs_legal", title: "Personal Docs & Legal Prep", tagline: "Answer questions from your leases, contracts, and paperwork, quoting the exact passage.", proof: "citation", proof_label: "Every answer must quote your document word for word", accepts: [".pdf", ".md", ".txt", ".docx"], needs: "The documents you want it to know (PDF, Word, Markdown or text).", example_rules: ["must include a quote", "no more than 80 words"] },
  { id: "custom", title: "Custom Skill", tagline: "Teach any skill from examples of questions and the answers you expect.", proof: "rules", proof_label: "Checked against your expected answers and your rules", accepts: [".csv", ".jsonl", ".json"], needs: "A file of examples with a question column and an answer column.", example_rules: ["no more than 50 words", 'must mention "thank you"'] },
];

let project: Record<string, unknown> | null = null;
let trained = false;

const samples = [
  ["Paid $42 at Shell for the delivery van", '{"category": "fuel", "amount": 42}', '{"category": "travel"}', '{"category": "fuel", "amount": 42}'],
  ["Zoom receipt $15 monthly plan", '{"category": "software", "amount": 15}', "Sure! This looks like a software expense of about $15.", '{"category": "software", "amount": 15}'],
  ["Card charge: Staples $87", '{"category": "office_supplies", "amount": 87}', '{"category": "supplies", "amount": "87"}', '{"category": "office_supplies", "amount": 87}'],
  ["Reimburse $230 spent at Hilton", '{"category": "travel", "amount": 230}', '{"category": "meals", "amount": 230}', '{"category": "travel", "amount": 230}'],
];

async function simulateTraining() {
  const stages: [string, string, number, number][] = [
    ["verify_data", "Checking that no hidden test question leaked into training", 0, 2],
    ["baseline", "Measuring the original model on the hidden test", 2, 18],
    ["sft", "Learning from 196 examples", 22, 70],
    ["stage2", "Practicing with automatic grading", 70, 84],
    ["gate", "Final exam: original vs trained on hidden questions", 84, 98],
  ];
  for (const [stage, detail, o0, o1] of stages) {
    emit("training://progress", ev({ type: "stage", stage, status: "started", detail }));
    const steps = stage === "sft" ? 24 : 6;
    for (let i = 1; i <= steps; i++) {
      await sleep(stage === "sft" ? 260 : 150);
      const dev = stage === "sft" ? Math.min(0.84, 0.05 + i * 0.035) : stage === "stage2" ? 0.84 + i * 0.01 : null;
      emit("training://progress", ev({ type: "progress", stage, step: i, total: steps, pct: (100 * i) / steps, overall_pct: o0 + ((o1 - o0) * i) / steps, eta_s: Math.max(0, (24 - i) * 9 + (stage === "sft" ? 120 : 40)), metrics: { loss: stage === "sft" ? 2.1 * Math.exp(-i / 8) + 0.18 : 0.2, lr: 2e-4, tok_s: 2140 + i * 3, dev_pass: dev, replay_dnll: 0.02 + i * 0.001 } }));
      if (stage === "sft" && i % 6 === 0) {
        const s = samples[(i / 6 - 1) % samples.length];
        emit("benchmark://result", ev({ type: "benchmark", source: "training_dev", case_id: `c${i}`, prompt: s[0], reference: s[1], baseline: { output: s[2], pass: false, score: 0 }, trained: { output: i >= 12 ? s[3] : '{"category": "fuel"}', pass: i >= 12, score: i >= 12 ? 1 : 0.5 }, step: i * 20, judge: null }));
        emit("training://log", ev({ type: "log", level: "info", message: `sft step ${i * 20}/480: dev pass ${Math.round((dev ?? 0) * 100)}%, drift +0.031 nats/token` }));
      }
    }
    if (stage === "baseline") emit("training://progress", ev({ type: "eval", split: "test", model: "base", n: 120, passed: 38, pass_rate: 0.317, ci95: [0.24, 0.40] }));
    emit("training://progress", ev({ type: "stage", stage, status: "completed", detail: stage === "baseline" ? "Original model: 38/120 correct" : "" }));
  }
  emit("training://progress", ev({ type: "eval", split: "test", model: "candidate", n: 120, passed: 104, pass_rate: 0.867, ci95: [0.79, 0.92] }));
  const explanation = "Your trained model passed 87% of the 120 hidden test questions, versus 32% for the base (+55.0 points). That difference is statistically solid (95% range +45.8 to +63.3 points). General skills: no meaningful drop.";
  emit("training://progress", ev({ type: "gate", decision: "PROMOTE", label: "VERIFIED", delta_pts: 55, baseline_pass_rate: 0.317, candidate_pass_rate: 0.867, checks: { task: true, general: true, format: true, audit: true, leakage: true }, version: "v1", report_path: "~/Barnraise/projects/expense-sorter/versions/v1/report.json", explanation }));
  trained = true;
  emit("training://progress", ev({ type: "done", ok: true, result: { decision: "PROMOTE", version: "v1", explanation } }));
}

export async function mockInvoke<T>(cmd: string, args: Record<string, unknown> = {}): Promise<T> {
  await sleep(250);
  switch (cmd) {
    case "detect_hardware":
      return hardware as T;
    case "runtime_status":
      return { ready: false, backends: {}, home: "~/Library/Application Support/Barnraise", message: "Not set up yet" } as T;
    case "bootstrap_runtime": {
      const steps = ["Creating a private Python 3.12", "Installing PyTorch", "Installing the Barnraise engine", "Checking everything works", "Downloading Qwen3 8B"];
      for (let i = 0; i < steps.length; i++) {
        emit("runtime://progress", { type: "runtime", step: i + 1, total: steps.length, label: steps[i], status: "running", detail: "", pct: (100 * i) / steps.length });
        await sleep(i === 4 ? 1400 : 600);
        emit("runtime://progress", { type: "runtime", step: i + 1, total: steps.length, label: steps[i], status: "completed", detail: i === 1 ? "not needed on Apple Silicon (uses MLX)" : "", pct: (100 * (i + 1)) / steps.length });
      }
      return { ready: true, engine_version: "0.1.0", backends: { mlx: true, torch: false }, home: "~/Library/Application Support/Barnraise", message: "Ready" } as T;
    }
    case "list_templates":
      return templates as T;
    case "list_projects":
      return (project ? [project] : []) as T;
    case "create_project": {
      const req = args.request as { name: string; template: string };
      project = { id: "p1", slug: req.name.toLowerCase().replace(/[^a-z0-9]+/g, "-"), name: req.name, template: req.template, dir: "~/Barnraise/projects/" + req.name, base_model: { id: "qwen3-8b", path: "" }, active_version: null };
      return project as T;
    }
    case "get_project":
      return { ...(project ?? {}), active_version: trained ? "v1" : null } as T;
    case "compile_rules": {
      const texts = (args.texts as string[]) ?? [];
      return { rules: texts.map((t) => (/one of|positive|words|mention|sum|required|between/.test(t) ? { ok: true, text: t, expr: "…", explanation: t.replace("must be", "→"), error: "", self_test: { references: 50, accepted: 50, mutants: 40, caught: 38, discrimination: 0.95, ok: true } } : { ok: false, text: t, expr: "", explanation: "", error: "I couldn't turn that into a check. Try phrasing like: “category must be one of A, B, C”, “amount must be positive”, or “no more than 50 words”." })) } as T;
    }
    case "ingest_dataset": {
      const steps = ["parse", "build_items", "rules", "split"];
      for (const s of steps) {
        emit("training://progress", ev({ type: "stage", stage: s, status: "started", detail: "" }));
        await sleep(300);
        emit("training://progress", ev({ type: "stage", stage: s, status: "completed", detail: "" }));
      }
      return { counts: { train: 286, dev: 40, test: 120 }, leakage_free: true, duplicates_removed: 7, warnings: [], rules: [], verifiers: [{ id: "json_valid", kind: "json_valid", role: "gate", description: "Reply is a JSON object" }, { id: "field:category", kind: "field_match", role: "reward", description: "category matches your record" }, { id: "field:amount", kind: "field_match", role: "reward", description: "amount matches your record" }], seconds: 1.3 } as T;
    }
    case "preview_items":
      return { items: samples.map((s, i) => ({ item_id: `i${i}`, split: "dev", prompt: s[0], reference: s[1], source: `expenses.csv:${12 + i}` })), counts: { train: 286, dev: 40, test: 120 }, test_locked: true, warnings: [] } as T;
    case "start_training_job":
      void simulateTraining();
      return "job-preview" as T;
    case "cancel_training_job":
      return undefined as T;
    case "run_benchmark_arena": {
      const prompt = (args.prompt as string) || samples[0][0];
      emit("benchmark://result", ev({ type: "benchmark", source: "arena", case_id: "a1", prompt, reference: null, baseline: { output: "This appears to be a fuel purchase.", pass: false, score: 0 }, trained: { output: '{"category": "fuel", "amount": 42}', pass: true, score: 1 }, step: null, judge: null }));
      return { cases: 1, wins: { trained: 1, baseline: 0, tie: 0 } } as T;
    }
    case "start_local_api":
      return { url: "http://127.0.0.1:11435/v1/chat/completions", base_url: "http://127.0.0.1:11435/v1", token: "br_preview0000000000000000000000000000000000000000000000000000000000", model: "barnraise/expense-sorter", base_model: "barnraise/expense-sorter@base", version: "v1" } as T;
    case "stop_local_api":
      return undefined as T;
    case "export_adapter":
      return { path: String(args.dest), sha256: "33b93bf6277b4edb34b501ca3ded0d7dd088c60bbccf84cb59c95c40a00d1ea0", version: "v1", publisher: "1b8502ca92f1…" } as T;
    case "import_adapter":
      return { import: { signature_verified: true, sha256_verified: true, publisher: "1b8502ca92f1…", checks: { tensors: 112, rank: 32 } }, gate: { decision: "REJECT", explanation: "Your trained model passed 87% of the 120 hidden test questions, versus 87% for the current version v1 (+0.0 points). Not promoted because the improvement is too small or not statistically reliable yet. Your current version stays active." } } as T;
    default:
      throw { code: "UNKNOWN", message: `preview mode has no mock for ${cmd}` };
  }
}
