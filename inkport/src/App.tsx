import { useState } from "react";
import type { HardwareReport, IngestSummary, ProjectInfo, Template } from "./lib/contract";
import { api, inTauri } from "./lib/ipc";
import { SystemReady } from "./screens/SystemReady";
import { TemplateLauncher } from "./screens/TemplateLauncher";
import { DataRulesStudio } from "./screens/DataRulesStudio";
import { TrainingArena } from "./screens/TrainingArena";
import { DeployPanel } from "./screens/DeployPanel";

type Step = 0 | 1 | 2 | 3 | 4;
const STEPS = ["Your computer", "Choose a goal", "Data & rules", "Train & prove", "Use it"];

export function App() {
  const [step, setStep] = useState<Step>(0);
  const [reached, setReached] = useState<Step>(0);
  const [hardware, setHardware] = useState<HardwareReport | null>(null);
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [template, setTemplate] = useState<Template | null>(null);
  const [, setIngest] = useState<IngestSummary | null>(null);

  const go = (s: Step) => {
    setStep(s);
    setReached((r) => (s > r ? s : r));
  };

  const refreshProject = async () => {
    if (project) setProject(await api.getProject(project.id).catch(() => project));
  };

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 flex-none flex-col border-r border-line bg-panel p-5 md:flex">
        <div className="flex items-center gap-2 text-lg font-bold">
          <span aria-hidden>🏚️</span> Barnraise
        </div>
        <p className="mt-1 text-xs text-muted">Train your own AI. Prove it got better.</p>
        <nav className="mt-8 space-y-1" aria-label="Steps">
          {STEPS.map((label, i) => {
            const enabled = i <= reached;
            return (
              <button key={label} disabled={!enabled} onClick={() => go(i as Step)} className={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left text-sm transition ${step === i ? "bg-accent-soft font-semibold text-ink" : enabled ? "text-ink hover:bg-canvas" : "text-muted/60"}`} aria-current={step === i ? "step" : undefined}>
                <span className={`inline-flex h-6 w-6 items-center justify-center rounded-full text-xs ${i < reached ? "bg-accent text-white" : step === i ? "border-2 border-accent" : "border border-line"}`}>{i < reached ? "✓" : i + 1}</span>
                {label}
              </button>
            );
          })}
        </nav>
        <div className="mt-auto space-y-1 text-xs text-muted">
          {project && <div className="truncate">Project: <span className="font-medium text-ink">{project.name}</span></div>}
          {hardware && <div className="truncate">{hardware.recommendation.model_label} · {hardware.recommendation.backend}</div>}
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        {!inTauri && <div className="border-b border-warn/40 bg-warn-soft px-6 py-2 text-xs">Preview mode: this browser tab simulates the desktop engine so the interface can be reviewed. Run <code>pnpm tauri dev</code> for the real app.</div>}
        <div className="mx-auto max-w-5xl px-6 py-8">
          {step === 0 && (
            <SystemReady
              onReady={(hw) => {
                setHardware(hw);
                go(1);
              }}
            />
          )}
          {step === 1 && (
            <TemplateLauncher
              onCreated={(p, t) => {
                setProject(p);
                setTemplate(t);
                go(p.active_version ? 3 : 2);
              }}
            />
          )}
          {step === 2 && project && template && (
            <DataRulesStudio
              project={project}
              template={template}
              onBuilt={(s) => {
                setIngest(s);
                go(3);
              }}
            />
          )}
          {step === 3 && project && (
            <TrainingArena
              project={project}
              onDeploy={async () => {
                await refreshProject();
                go(4);
              }}
            />
          )}
          {step === 4 && project && <DeployPanel project={project} />}
        </div>
      </main>
    </div>
  );
}
