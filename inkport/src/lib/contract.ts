// TypeScript mirror of contracts/schemas/*.json and src-tauri/src/contract.rs.

export interface GpuInfo {
  vendor: "nvidia" | "amd" | "intel" | "apple" | "other";
  name: string;
  vram_gb: number;
  vram_free_gb?: number | null;
  driver?: string | null;
  compute_capability?: string | null;
  unified: boolean;
}

export interface Recommendation {
  tier: string;
  vendor: string;
  memory_gb: number;
  model_id: string;
  model_label: string;
  backend: "unsloth" | "torch" | "mlx";
  quant: "nf4" | "bf16" | "fp32" | "mlx4";
  lora_r: number;
  micro_batch: number;
  grad_accum: number;
  max_seq: number;
  stage2: "none" | "rft" | "grpo";
  rft_k: number;
  plain: string;
}

export interface HardwareReport {
  os: "macos" | "windows" | "linux" | "other";
  arch: string;
  cpu_brand: string;
  cpu_cores: number;
  ram_gb: number;
  disk_free_gb: number;
  gpus: GpuInfo[];
  apple_silicon: boolean;
  recommendation: Recommendation;
  summary: string[];
}

export interface RuntimeStatus {
  ready: boolean;
  python?: string | null;
  engine_version?: string | null;
  backends: Record<string, boolean>;
  home: string;
  message: string;
}

export interface RuntimeProgress {
  type: "runtime";
  step: number;
  total: number;
  label: string;
  status: "running" | "completed";
  detail: string;
  pct: number;
}

export interface Template {
  id: "business_rules" | "code_sql" | "docs_legal" | "custom";
  title: string;
  tagline: string;
  proof: string;
  proof_label: string;
  accepts: string[];
  needs: string;
  example_rules: string[];
}

export interface ProjectInfo {
  id: string;
  slug: string;
  name: string;
  template: Template["id"];
  dir: string;
  base_model: { id: string; path: string };
  backend?: string | null;
  quant?: string | null;
  active_version?: string | null;
  counts?: { train: number; dev: number; test: number } | null;
  latest_report?: GateReport | null;
}

export interface ApiInfo {
  url: string;
  base_url: string;
  token: string;
  model: string;
  base_model: string;
  version: string;
  engine: string;
}

export interface VerifyReport {
  ok: boolean;
  mode: "full" | "quick";
  files_checked: number;
  bytes_checked: number;
  seconds: number;
  manifest_pinned: boolean;
  problems: string[];
}

export interface PreflightReport {
  ok: boolean;
  bundled: boolean;
  offline_core: boolean;
  platform: string;
  flavor: string;
  app_version: string;
  engine_version: string | null;
  python: string | null;
  training_accelerator: string;
  inference_devices: string[];
  starter_model: { id: string; label: string; format: string; license: string; size_mb: number; params_b: number } | null;
  verify: VerifyReport;
  seconds: number;
  problems: string[];
}

export interface PreflightProgress {
  type: "preflight";
  phase: "verify" | "warmup";
  done: number;
  total: number;
  pct: number;
}

export interface BaseModel {
  id: string;
  label?: string;
  path: string;
  format?: string;
  gguf?: string;
  bundled?: boolean;
  pack?: string;
}

export interface PackState {
  id: string;
  title: string;
  description: string;
  kind: "model" | "dataset" | "verifier";
  approx_size_mb: number;
  license: string;
  installed: boolean;
  installing: boolean;
  partial_bytes: number;
  path: string | null;
}

export interface PackProgress {
  type: "pack";
  pack_id: string;
  status: "resolving" | "downloading" | "verifying" | "installed" | "failed" | "cancelled";
  file: string;
  done: number;
  total: number;
  pct: number;
  message: string;
}

export interface AppError {
  code: string;
  message: string;
  retryable: boolean;
}

export interface RuleReport {
  ok: boolean;
  text: string;
  expr: string;
  explanation: string;
  error: string;
  warning?: string;
  /** Not used because of the examples (a missing column, examples that break it), not the wording. */
  depends_on_data?: boolean;
  self_test?: { references: number; accepted: number; mutants: number; caught: number; discrimination: number | null; ok: boolean } | null;
}

export interface IngestSummary {
  counts: { train: number; dev: number; test: number };
  leakage_free: boolean;
  duplicates_removed: number;
  warnings: string[];
  rules: RuleReport[];
  verifiers: { id: string; kind: string; role: string; description: string }[];
  seconds: number;
  /** false when there are too few examples for a fair final exam; `blocker` says what to do. */
  ready?: boolean;
  blocker?: string | null;
}

export interface PreviewItem {
  item_id: string;
  split: "train" | "dev";
  prompt: string;
  reference: string | null;
  source: string;
}

export interface Preview {
  items: PreviewItem[];
  counts: { train: number; dev: number; test: number };
  test_locked: boolean;
  warnings: string[];
  verifiers?: { id: string; kind: string; role: string; description: string }[];
  /** What the exam was built from (to restore the screen when a project is reopened). */
  inputs?: string[];
  rules?: RuleReport[];
  duplicates_removed?: number;
  leakage_free?: boolean;
  ready?: boolean;
  blocker?: string | null;
}

export interface GateReport {
  decision: "PROMOTE" | "REJECT" | "INSUFFICIENT_DATA";
  label: "VERIFIED" | "PRELIMINARY" | null;
  explanation: string;
  promoted_version?: string;
  compared_against?: string;
  test?: { n: number; baseline_pass_rate: number; candidate_pass_rate: number; delta_pts: number };
}

/** The local API as the window should show it. */
export interface ApiStatus {
  running: boolean;
  project_id: string | null;
  info: ApiInfo | null;
}

/** A job in progress, with the events needed to redraw its screen. */
export interface RunningJob {
  job_id: string;
  kind: string;
  project_id: string | null;
  events: EngineEvent[];
}

// ---------------------------------------------------------------- engine events
interface Base {
  v: 1;
  job_id: string;
  ts: number;
  job_kind?: string;
  project_id?: string;
}
export interface Side {
  output: string;
  pass: boolean | null;
  score: number | null;
}
export type EngineEvent = Base &
  (
    | { type: "hello"; engine_version: string; python: string; platform: string; backends: Record<string, boolean> }
    | { type: "heartbeat" }
    | { type: "log"; level: "debug" | "info" | "warn" | "error"; message: string }
    | { type: "stage"; stage: string; status: "started" | "completed" | "skipped" | "failed"; detail?: string }
    | { type: "progress"; stage: string; step: number; total: number; pct: number; overall_pct: number; eta_s: number | null; metrics: Record<string, number | null> }
    | { type: "eval"; split: "test" | "dev" | "general"; model: "base" | "candidate" | "incumbent" | "placebo"; n: number; passed: number; pass_rate: number; ci95: [number, number] }
    | { type: "benchmark"; source: "training_dev" | "arena"; case_id: string; prompt: string; reference: string | null; baseline: Side; trained: Side; step?: number | null; judge?: unknown }
    | { type: "gate"; decision: GateReport["decision"]; label: GateReport["label"]; delta_pts: number; baseline_pass_rate: number; candidate_pass_rate: number; checks: Record<string, boolean>; version: string | null; report_path: string; explanation: string }
    | { type: "artifact"; kind: string; path: string; sha256: string }
    | { type: "ready"; host: "127.0.0.1"; port: number; model: string }
    | { type: "error"; code: string; message: string; retryable: boolean }
    | { type: "done"; ok: boolean; result: Record<string, unknown> }
  );

export type Channel = "training://progress" | "training://log" | "benchmark://result" | "runtime://progress" | "api://status" | "packs://progress";
