import { useEffect, useState } from "react";
import { Check } from "lucide-react";
import type { BaseModel, HardwareReport, IngestSummary, ProjectInfo, Template } from "./lib/contract";
import { api, inTauri } from "./lib/ipc";
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

export function App() {
  const [step, setStep] = useState<Step>(0);
  const [reached, setReached] = useState<Step>(0);
  const [, setHardware] = useState<HardwareReport | null>(null);
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [template, setTemplate] = useState<Template | null>(null);
  const [models, setModels] = useState<BaseModel[]>([]);
  const [, setIngest] = useState<IngestSummary | null>(null);

  const go = (s: Step) => {
    setStep(s);
    setReached((r) => (s > r ? s : r));
  };

  const refreshProject = async () => {
    if (project) setProject(await api.getProject(project.id).catch(() => project));
  };

  useEffect(() => {
    if (project) api.listModels().then(setModels).catch(() => {});
  }, [project?.id]);

  // The model this project really uses (not the one recommended for the computer).
  const modelLabel = project ? (models.find((m) => m.id === project.base_model.id)?.label ?? project.base_model.id).replace(/\s*\([^)]*\)\s*$/, "") : null;

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
                  const enabled = i <= reached;
                  const current = step === i;
                  return (
                    <li key={label}>
                      <button disabled={!enabled} onClick={() => go(i)} aria-current={current ? "step" : undefined} className={`-ml-px flex w-full items-center justify-between gap-2 border-l py-2 pl-5 pr-1 text-left text-[15px] transition-colors ${current ? "border-ink font-semibold text-ink" : enabled ? "border-transparent text-muted hover:border-faint hover:text-ink" : "border-transparent text-faint"}`}>
                        {label}
                        {i < reached && !current && <Icon icon={Check} size={15} className="text-faint" />}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
        {project && (
          <dl className="border-t border-line px-7 py-5 text-[15px]">
            {[["Project", project.name], ["Model", modelLabel ?? ""]].map(([k, v]) => (
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
