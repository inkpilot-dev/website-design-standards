import { useEffect, useState } from "react";
import type { AppError, BaseModel, ProjectInfo, Template } from "../lib/contract";
import { api, toAppError } from "../lib/ipc";
import { Badge, Button, Card, ErrorNote } from "../components/ui";

const ICON: Record<Template["id"], string> = { business_rules: "🧾", code_sql: "📊", docs_legal: "📚", custom: "🧩" };
const DEFAULT_NAME: Record<Template["id"], string> = { business_rules: "Expense sorter", code_sql: "Spreadsheet SQL helper", docs_legal: "Lease and contracts helper", custom: "My custom skill" };

export function TemplateLauncher({ onCreated }: { onCreated: (p: ProjectInfo, t: Template) => void }) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [chosen, setChosen] = useState<Template | null>(null);
  const [name, setName] = useState("");
  const [models, setModels] = useState<BaseModel[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<AppError | null>(null);

  useEffect(() => {
    api.listTemplates().then(setTemplates).catch((e) => setError(toAppError(e)));
    api.listProjects().then(setProjects).catch(() => {});
    api.listModels().then((m) => {
      setModels(m);
      if (m[0]) setModelId(m[0].id);
    }).catch(() => {});
  }, []);

  const create = async () => {
    if (!chosen || !name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      onCreated(await api.createProject(name.trim(), chosen.id, modelId || undefined), chosen);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">What do you want your AI to master?</h1>
        <p className="mt-2 text-muted">Pick a starting point. Every template comes with an automatic exam, so you can see proof it got better.</p>
      </header>
      <ErrorNote error={error} />
      <div className="grid gap-4 sm:grid-cols-2">
        {templates.map((t) => (
          <button
            key={t.id}
            onClick={() => {
              setChosen(t);
              setName(DEFAULT_NAME[t.id]);
            }}
            className={`rounded-2xl border bg-panel p-5 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow ${chosen?.id === t.id ? "border-accent ring-2 ring-accent/30" : "border-line"}`}
          >
            <div className="text-3xl" aria-hidden>{ICON[t.id]}</div>
            <div className="mt-3 text-lg font-semibold">{t.title}</div>
            <p className="mt-1 text-sm text-muted">{t.tagline}</p>
            <div className="mt-3">
              <Badge tone="good">✓ {t.proof_label}</Badge>
            </div>
            <p className="mt-3 text-xs text-muted">You’ll need: {t.needs}</p>
          </button>
        ))}
      </div>
      {chosen && (
        <Card className="flex flex-wrap items-end gap-3">
          <label className="flex-1">
            <span className="text-sm font-semibold">Give it a name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} className="mt-1 w-full rounded-xl border border-line bg-canvas px-3 py-2.5 outline-none focus:border-accent" />
          </label>
          {models.length > 1 && (
            <label>
              <span className="text-sm font-semibold">AI brain</span>
              <select value={modelId} onChange={(e) => setModelId(e.target.value)} className="mt-1 block rounded-xl border border-line bg-canvas px-3 py-2.5 outline-none focus:border-accent">
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{m.label ?? m.id}</option>
                ))}
              </select>
            </label>
          )}
          <Button onClick={create} busy={busy} disabled={!name.trim()}>
            Create
          </Button>
        </Card>
      )}
      {projects.length > 0 && (
        <Card>
          <div className="mb-2 text-sm font-semibold">Or continue a project</div>
          <ul className="divide-y divide-line">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center justify-between py-2">
                <div>
                  <div className="font-medium">{p.name}</div>
                  <div className="text-xs text-muted">{templates.find((t) => t.id === p.template)?.title} · {p.active_version ? `trained version ${p.active_version}` : "not trained yet"}</div>
                </div>
                <Button variant="secondary" onClick={() => { const t = templates.find((t) => t.id === p.template); if (t) onCreated(p, t); }}>Open</Button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
