// Typed bridge to the Rust supervisor. Inside the desktop app this calls Tauri commands and
// listens to Tauri events; in a plain browser (`pnpm dev`) it falls back to a realistic
// simulation (./mock.ts) so the UI can be developed and reviewed without Python or a GPU.

import type { ApiInfo, AppError, BaseModel, Channel, HardwareReport, IngestSummary, PackState, PreflightReport, Preview, ProjectInfo, RuleReport, RuntimeStatus, Template } from "./contract";

export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  try {
    if (inTauri) {
      const { invoke } = await import("@tauri-apps/api/core");
      return await invoke<T>(cmd, args);
    }
    const { mockInvoke } = await import("./mock");
    return await mockInvoke<T>(cmd, args);
  } catch (e) {
    throw toAppError(e);
  }
}

export function toAppError(e: unknown): AppError {
  if (e && typeof e === "object" && "message" in e) {
    const o = e as Partial<AppError>;
    return { code: o.code ?? "ERROR", message: String(o.message), retryable: Boolean(o.retryable) };
  }
  return { code: "ERROR", message: String(e), retryable: false };
}

export async function on<T>(channel: Channel, cb: (payload: T) => void): Promise<() => void> {
  if (inTauri) {
    const { listen } = await import("@tauri-apps/api/event");
    return await listen<T>(channel, (e) => cb(e.payload));
  }
  const { mockListen } = await import("./mock");
  return mockListen(channel, cb as (p: unknown) => void);
}

/** Use inside a React effect: `return subscribe(on(...), on(...))`. Also unsubscribes listeners
 * that only finish registering after the effect was torn down (otherwise they would leak and
 * every event would be handled twice). */
export function subscribe(...pending: Promise<() => void>[]): () => void {
  let alive = true;
  const uns: (() => void)[] = [];
  for (const p of pending) void p.then((u) => (alive ? uns.push(u) : u()));
  return () => {
    alive = false;
    uns.forEach((u) => u());
  };
}

export const api = {
  detectHardware: () => call<HardwareReport>("detect_hardware"),
  preflight: (forceFull = false) => call<PreflightReport>("preflight", { forceFull }),
  listModels: () => call<BaseModel[]>("list_models"),
  listPacks: () => call<PackState[]>("list_packs"),
  installPack: (packId: string) => call<void>("install_pack", { packId }),
  cancelPack: (packId: string) => call<boolean>("cancel_pack", { packId }),
  removePack: (packId: string) => call<void>("remove_pack", { packId }),
  sampleData: () => call<string | null>("sample_data"),
  runtimeStatus: () => call<RuntimeStatus>("runtime_status"),
  bootstrapRuntime: (downloadModel = true) => call<RuntimeStatus>("bootstrap_runtime", { downloadModel }),
  listTemplates: () => call<Template[]>("list_templates"),
  createProject: (name: string, template: string, baseModelId?: string) => call<ProjectInfo>("create_project", { request: { name, template, base_model_id: baseModelId ?? null, base_model_path: null, backend: null, quant: null } }),
  listProjects: () => call<ProjectInfo[]>("list_projects"),
  getProject: (projectId: string) => call<ProjectInfo>("get_project", { projectId }),
  ingestDataset: (projectId: string, paths: string[], rules: string[]) => call<IngestSummary>("ingest_dataset", { projectId, paths, rules }),
  previewItems: (projectId: string, limit = 12) => call<Preview>("preview_items", { projectId, limit }),
  compileRules: (texts: string[], projectId?: string) => call<{ rules: RuleReport[] }>("compile_rules", { projectId, texts }),
  startTrainingJob: (projectId: string, preset: "standard" | "fast" | "smoke") => call<string>("start_training_job", { request: { project_id: projectId, preset } }),
  cancelTrainingJob: (jobId: string) => call<void>("cancel_training_job", { jobId }),
  runBenchmarkArena: (projectId: string, prompt?: string, n = 4) => call<{ cases: number; wins: Record<string, number> }>("run_benchmark_arena", { projectId, prompt, n }),
  startLocalApi: (projectId: string) => call<ApiInfo>("start_local_api", { projectId }),
  stopLocalApi: () => call<void>("stop_local_api"),
  exportAdapter: (projectId: string, dest: string) => call<{ path: string; sha256: string; version: string; publisher: string }>("export_adapter", { projectId, dest }),
  importAdapter: (projectId: string, file: string, trust = false) => call<{ import: { signature_verified: boolean; sha256_verified: boolean; publisher: string; checks: { tensors: number; rank: number } }; gate?: { decision: string; explanation: string; version?: string } }>("import_adapter", { projectId, file, trust }),
};

/** Native file pickers (desktop) with a graceful browser fallback. */
export async function pickFiles(extensions: string[]): Promise<string[]> {
  if (inTauri) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const res = await open({ multiple: true, directory: false, filters: [{ name: "Your files", extensions: extensions.map((e) => e.replace(/^\./, "")) }] });
    return Array.isArray(res) ? res : res ? [res] : [];
  }
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = extensions.join(",");
    input.onchange = () => resolve(Array.from(input.files ?? []).map((f) => `~/Documents/${f.name}`));
    input.click();
  });
}

export async function pickSavePath(defaultName: string): Promise<string | null> {
  if (inTauri) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    return await save({ defaultPath: defaultName, filters: [{ name: "Inkport adapter", extensions: ["inkpatch"] }] });
  }
  return `~/Downloads/${defaultName}`;
}

export async function onFileDrop(cb: (paths: string[]) => void): Promise<() => void> {
  if (!inTauri) return () => {};
  const { getCurrentWebview } = await import("@tauri-apps/api/webview");
  return await getCurrentWebview().onDragDropEvent((e) => {
    if (e.payload.type === "drop") cb(e.payload.paths);
  });
}
