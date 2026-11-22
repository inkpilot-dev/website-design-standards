import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { Check } from "lucide-react";
import type { ApiStatus, BaseModel, EngineEvent, IngestSummary, Preview, ProjectInfo, RuleReport, Template } from "./lib/contract";
import { api, inTauri, on, subscribe, toAppError } from "./lib/ipc";
import { idleTraining, type Preset, trainingReducer } from "./lib/training";
import { Icon } from "./components/ui";
import { SystemReady } from "./screens/SystemReady";
import { TemplateLauncher } from "./screens/TemplateLauncher";
import { DataRulesStudio } from "./screens/DataRulesStudio";
import { TrainingArena } from "./screens/TrainingArena";
import { DeployPanel } from "./screens/DeployPanel";
import logo from "./assets/logo.png";

type Step = 0 | 1 | 2 | 3 | 4;
const GROUPS: { title: string; steps: { step: Step; label: string }[] }[] = [
  { title: "Get started", steps: [{ step: 0, label: "Your computer" }] },
  {
    title: "Your AI",
    steps: [
      { step: 1, label: "Choose a goal" },
      { step: 2, label: "Data & rules" },
      { step: 3, label: "Train & prove" },
      { step: 4, label: "Use it" },
    ],
  },
];

/** What the "Data & rules" screen holds for one project; kept here so leaving the step loses nothing. */
export interface DataDraft {
  files: string[];
  rules: string[];
  checks: Record<string, RuleReport>;
  summary: IngestSummary | null;
  preview: Preview | null;
  /** Files or rules changed after the exam was built, so the exam no longer matches them. */
  stale: boolean;
}

const hasExam = (p: ProjectInfo | null) => !!p?.counts && p.counts.train + p.counts.dev + p.counts.test > 0;
const API_STOPPED: ApiStatus = { running: false, project_id: null, info: null };

export function App() {
  const [step, setStep] = useState<Step>(0);
  const [ready, setReady] = useState(false);
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [template, setTemplate] = useState<Template | null>(null);
  const [models, setModels] = useState<BaseModel[]>([]);
  const [training, dispatch] = useReducer(trainingReducer, idleTraining);
  const [trainingName, setTrainingName] = useState("");
  const [apiStatus, setApiStatus] = useState<ApiStatus>(API_STOPPED);
  const [drafts, setDrafts] = useState<Record<string, DataDraft>>({});
  const projectRef = useRef<ProjectInfo | null>(null);
  projectRef.current = project;

  const refreshProject = useCallback(async () => {
    const current = projectRef.current;
    if (!current) return;
    try {
      const fresh = await api.getProject(current.id);
      setProject((p) => (p && p.id === fresh.id ? fresh : p));
    } catch {
      /* the project list will show the truth next time */
    }
  }, []);

  const refreshApi = useCallback(async () => {
    try {
      setApiStatus(await api.apiStatus());
    } catch {
      setApiStatus(API_STOPPED);
    }
  }, []);

  /** Pick up a training run that is already going (started before this window showed it). */
  const attachRunning = useCallback(async () => {
    const job = await api.trainingStatus().catch(() => null);
    if (!job) return false;
    dispatch({ type: "attach", job });
    const name = job.project_id ? await api.getProject(job.project_id).then((p) => p.name).catch(() => "") : "";
    setTrainingName(name);
    return true;
  }, []);

  // One set of listeners for the whole app: what a job reports must not depend on which step is open.
  useEffect(
    () =>
      subscribe(
        on<EngineEvent>("training://progress", (event) => dispatch({ type: "event", event })),
        on<EngineEvent>("benchmark://result", (event) => dispatch({ type: "event", event })),
        on<EngineEvent>("training://log", (e) => {
          if (e.type === "log" && e.level !== "debug" && (!e.job_kind || e.job_kind === "train")) dispatch({ type: "log", line: `${e.level.toUpperCase()}  ${e.message}` });
        }),
        on<unknown>("api://status", () => void refreshApi()),
      ),
    [refreshApi],
  );

  useEffect(() => {
    if (ready) {
      void attachRunning();
      void refreshApi();
    }
  }, [ready, attachRunning, refreshApi]);

  // A run that ended may have saved a version (or not): show the project as it is on disk now.
  useEffect(() => {
    if (training.ended) void refreshProject();
  }, [training.ended, training.jobId, refreshProject]);

  useEffect(() => {
    if (project) api.listModels().then(setModels).catch(() => {});
  }, [project?.id]);

  const examBuilt = hasExam(project);
  const canOpen = (i: Step): boolean => {
    if (i === 0) return true;
    if (!ready) return false;
    if (i === 1) return true;
    if (!project) return false;
    if (i === 2) return true;
    if (i === 3) return examBuilt || training.projectId === project.id;
    return !!project.active_version;
  };
  const isDone = (i: Step): boolean => (i === 0 ? ready : i === 1 ? !!project : i === 2 ? examBuilt : i === 3 ? !!project?.active_version : false);

  const openProject = (p: ProjectInfo, t: Template) => {
    setProject(p);
    setTemplate(t);
    setStep(p.active_version ? 4 : hasExam(p) ? 3 : 2);
    void refreshApi();
  };

  const startTraining = async (preset: Preset) => {
    const p = projectRef.current;
    if (!p || training.running) return;
    dispatch({ type: "starting", projectId: p.id, preset });
    setTrainingName(p.name);
    try {
      dispatch({ type: "started", jobId: await api.startTrainingJob(p.id, preset) });
    } catch (e) {
      const error = toAppError(e);
      // Already running (for this or another project): show that run instead of an error.
      if (error.code === "BUSY" && (await attachRunning())) return;
      dispatch({ type: "start_failed", error });
    }
  };

  const stopTraining = async () => {
    const job = training.jobId ?? (await api.trainingStatus().catch(() => null))?.job_id;
    if (job) await api.cancelTrainingJob(job).catch(() => {});
  };

  const setDraft = useCallback((id: string, draft: DataDraft) => setDrafts((d) => ({ ...d, [id]: draft })), []);

  // The model this project really uses (not the one recommended for the computer).
  const modelLabel = project ? (models.find((m) => m.id === project.base_model.id)?.label ?? project.base_model.id).replace(/\s*\([^)]*\)\s*$/, "") : null;
  const trainingElsewhere = training.running && training.projectId !== project?.id;

  return (
    <div className="flex h-full">
      <aside className="hidden w-[272px] flex-none flex-col border-r border-line md:flex">
        <div className="flex items-center gap-3 px-7 pt-8">
          <img src={logo} alt="" className="h-9 w-9" />
          <span className="text-[19px] font-bold tracking-[-0.02em]">Inkport</span>
        </div>
        <nav className="mt-10 flex-1 space-y-9 overflow-auto px-7" aria-label="Steps">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <div className="text-[15px] font-medium">{g.title}</div>
              <ul className="mt-3 border-l border-line">
                {g.steps.map(({ step: i, label }) => {
                  const enabled = canOpen(i);
                  const current = step === i;
                  return (
                    <li key={label}>
                      <button disabled={!enabled} onClick={() => setStep(i)} aria-current={current ? "step" : undefined} className={`-ml-px flex w-full items-center justify-between gap-2 border-l py-2 pl-5 pr-1 text-left text-[15px] transition-colors ${current ? "border-ink font-semibold text-ink" : enabled ? "border-transparent text-muted hover:border-faint hover:text-ink" : "border-transparent text-faint"}`}>
                        {label}
                        {i === 3 && training.running && training.projectId === project?.id ? <span className="h-1.5 w-1.5 flex-none animate-pulse rounded-full bg-ink" aria-label="training is running" /> : isDone(i) && !current ? <Icon icon={Check} size={15} className="text-faint" /> : null}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
        {(project || training.running) && (
          <dl className="border-t border-line px-7 py-5 text-[15px]">
            {(
              [
                ...(project ? [["Project", project.name], ["Model", modelLabel ?? ""]] : []),
                ...(training.running ? [["Training", `${trainingElsewhere && trainingName ? `${trainingName} · ` : ""}${Math.round(training.overall)}%`]] : []),
                ...(apiStatus.running ? [["Local API", "running"]] : []),
              ] as [string, string][]
            ).map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-4 py-1.5">
                <dt className="flex-none text-muted">{k}</dt>
                <dd className="truncate text-right font-medium" title={v}>{v}</dd>
              </div>
            ))}
          </dl>
        )}
      </aside>
      <main className="flex-1 overflow-auto">
        {!inTauri && <div className="border-b border-line px-10 py-2.5 text-[13px] text-muted">Preview mode: this browser tab simulates the desktop engine so the interface can be reviewed. Run <code className="rounded-md bg-canvas px-1.5 py-0.5 font-mono">pnpm tauri dev</code> for the real app.</div>}
        <div className="mx-auto max-w-[940px] px-10 pb-20 pt-14">
          {step === 0 && (
            <SystemReady
              onReady={() => setReady(true)}
              onContinue={() => {
                setReady(true);
                setStep(project ? (project.active_version ? 4 : examBuilt ? 3 : 2) : 1);
              }}
            />
          )}
          {step === 1 && <TemplateLauncher currentId={project?.id ?? null} onOpen={openProject} />}
          {step === 2 && project && template && (
            <DataRulesStudio
              key={project.id}
              project={project}
              template={template}
              draft={drafts[project.id] ?? null}
              onDraft={(d) => setDraft(project.id, d)}
              trainingRunning={training.running && training.projectId === project.id}
              onBuilt={refreshProject}
              onContinue={() => setStep(3)}
            />
          )}
          {step === 3 && project && (
            <TrainingArena
              key={project.id}
              project={project}
              training={training}
              trainingName={trainingName}
              onStart={startTraining}
              onStop={stopTraining}
              onDismissError={() => dispatch({ type: "dismiss_error" })}
              onDeploy={async () => {
                await refreshProject();
                setStep(4);
              }}
              onEditData={() => setStep(2)}
            />
          )}
          {step === 4 && project && <DeployPanel key={project.id} project={project} status={apiStatus} onStatus={setApiStatus} trainingRunning={training.running} onTrain={() => setStep(3)} onImported={refreshProject} />}
        </div>
      </main>
    </div>
  );
}
