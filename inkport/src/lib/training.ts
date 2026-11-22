// Everything the "Train & prove" screen shows, as a pure function of the engine's events.
// The state lives in App (not in the screen), so leaving the step, or switching project, never
// loses a run that is still going; `attachRunning` rebuilds it from the supervisor's replay.

import type { AppError, EngineEvent, RunningJob } from "./contract";

export type Bench = Extract<EngineEvent, { type: "benchmark" }>;
export type Gate = Extract<EngineEvent, { type: "gate" }>;
export type Preset = "standard" | "fast";
export type StageMark = "active" | "done" | "failed";

export interface TrainingState {
  jobId: string | null;
  projectId: string | null;
  preset: Preset;
  running: boolean;
  /** A run ended in this session: "ok" (a verdict), "stopped" (by the user) or "failed". */
  ended: null | "ok" | "stopped" | "failed";
  overall: number;
  eta: number | null;
  stages: Record<string, StageMark>;
  detail: string;
  baseRate: number | null;
  devPass: number | null;
  candRate: number | null;
  loss: number[];
  metrics: Record<string, number | null>;
  duels: Bench[];
  gate: Gate | null;
  error: AppError | null;
  logs: string[];
}

export const idleTraining: TrainingState = {
  jobId: null,
  projectId: null,
  preset: "standard",
  running: false,
  ended: null,
  overall: 0,
  eta: null,
  stages: {},
  detail: "",
  baseRate: null,
  devPass: null,
  candRate: null,
  loss: [],
  metrics: {},
  duels: [],
  gate: null,
  error: null,
  logs: [],
};

export type TrainingAction =
  | { type: "starting"; projectId: string; preset: Preset }
  | { type: "started"; jobId: string }
  | { type: "start_failed"; error: AppError }
  | { type: "attach"; job: RunningJob }
  | { type: "event"; event: EngineEvent }
  | { type: "log"; line: string }
  | { type: "dismiss_error" }
  | { type: "reset" };

/** rft1, rft2, grpo... are all the "Practice" step. */
const stageKey = (stage: string) => (stage.startsWith("rft") || stage === "grpo" ? "stage2" : stage);

function applyEvent(s: TrainingState, e: EngineEvent): TrainingState {
  // Only a training job's events belong here. (The same channel also carries building the exam,
  // previews, rule checks, imports and exports; their `done` used to end the screen's run.)
  if (e.job_kind && e.job_kind !== "train") return s;
  if (s.jobId && e.job_id && e.job_id !== s.jobId) return s;
  if (!s.jobId && s.projectId && e.project_id && e.project_id !== s.projectId) return s;
  if (!s.running) return s; // a late event after the run was closed
  switch (e.type) {
    case "progress": {
      const loss = e.metrics.loss;
      return {
        ...s,
        overall: Math.max(s.overall, Math.min(100, e.overall_pct)),
        eta: e.eta_s,
        metrics: e.metrics,
        loss: typeof loss === "number" ? [...s.loss.slice(-119), loss] : s.loss,
        devPass: typeof e.metrics.dev_pass === "number" ? e.metrics.dev_pass : s.devPass,
      };
    }
    case "stage": {
      const key = stageKey(e.stage);
      // "Practice" has several rounds that each start and complete; it is done when stage2 is.
      const subRound = key !== e.stage;
      let mark: StageMark | undefined = s.stages[key];
      if (e.status === "failed") mark = "failed";
      else if ((e.status === "completed" || e.status === "skipped") && !subRound) mark = "done";
      else if (mark !== "done") mark = "active";
      const stages = { ...s.stages, [key]: mark };
      // A finished stage's detail is a machine word ("PROMOTE", "v1"); keep the sentence instead.
      const detail = e.status === "started" && e.detail ? e.detail : s.detail;
      return { ...s, stages, detail };
    }
    case "eval":
      if (e.split !== "test") return s;
      if (e.model === "base" || e.model === "incumbent") return { ...s, baseRate: e.model === "incumbent" || s.baseRate === null ? e.pass_rate : s.baseRate };
      if (e.model === "candidate") return { ...s, candRate: e.pass_rate };
      return s;
    case "benchmark":
      if (e.source !== "training_dev") return s;
      return { ...s, duels: [e, ...s.duels.filter((x) => x.case_id !== e.case_id)].slice(0, 6) };
    case "gate":
      // PROMOTE arrives twice: first the verdict (version null, nothing saved yet), then again
      // with the version once it has been written.
      return { ...s, gate: e, baseRate: e.baseline_pass_rate, candRate: e.candidate_pass_rate };
    case "error":
      if (e.code === "CANCELLED") return { ...s, ended: "stopped" };
      return { ...s, error: { code: e.code, message: e.message, retryable: e.retryable } };
    case "done": {
      if (e.ok) return { ...s, running: false, ended: "ok", overall: 100, eta: 0 };
      const code = typeof e.result?.code === "string" ? e.result.code : "";
      const stopped = code === "CANCELLED" || s.ended === "stopped";
      const stages = Object.fromEntries(Object.entries(s.stages).map(([k, v]) => [k, v === "active" ? "failed" : v])) as Record<string, StageMark>;
      const error = stopped ? null : (s.error ?? { code: code || "ENGINE_FAILED", message: typeof e.result?.message === "string" ? e.result.message : "Training stopped unexpectedly. Try again.", retryable: true });
      // A "better" verdict whose version was never written means nothing was saved.
      const gate = s.gate && s.gate.decision === "PROMOTE" && !s.gate.version ? null : s.gate;
      return { ...s, running: false, ended: stopped ? "stopped" : "failed", eta: null, stages, error, gate };
    }
    default:
      return s;
  }
}

export function trainingReducer(s: TrainingState, a: TrainingAction): TrainingState {
  switch (a.type) {
    case "starting":
      return { ...idleTraining, projectId: a.projectId, preset: a.preset, running: true, detail: "Starting…" };
    case "started":
      return s.running ? { ...s, jobId: a.jobId } : s;
    case "start_failed":
      return { ...idleTraining, projectId: s.projectId, preset: s.preset, error: a.error };
    case "attach": {
      // A run that was already going (this window was showing something else, or was reopened).
      let next: TrainingState = { ...idleTraining, projectId: a.job.project_id, jobId: a.job.job_id, running: true, detail: "Training is running…", preset: s.projectId === a.job.project_id ? s.preset : "standard" };
      for (const e of a.job.events) next = applyEvent(next, { ...e, job_id: a.job.job_id, job_kind: "train" } as EngineEvent);
      return { ...next, running: true };
    }
    case "event":
      return applyEvent(s, a.event);
    case "log":
      return s.running || s.ended ? { ...s, logs: [...s.logs.slice(-200), a.line] } : s;
    case "dismiss_error":
      return { ...s, error: null };
    case "reset":
      return idleTraining;
    default:
      return s;
  }
}

/** Is a verdict fully in: PROMOTE needs its saved version, the others are final as they arrive. */
export const verdictReady = (s: TrainingState) => !!s.gate && !s.running && (s.gate.decision !== "PROMOTE" || !!s.gate.version);
