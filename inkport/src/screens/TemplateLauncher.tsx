import { useEffect, useState } from "react";
import type { AppError, BaseModel, ProjectInfo, Template } from "../lib/contract";
import { api, toAppError } from "../lib/ipc";
import { BookOpen, Check, type LucideIcon, Puzzle, Receipt, Table2 } from "lucide-react";
import { Button, Card, ChoiceCard, ErrorNote, Icon, PageHeader, SectionTitle, TextInput } from "../components/ui";

const ICON: Record<Template["id"], LucideIcon> = { business_rules: Receipt, code_sql: Table2, docs_legal: BookOpen, custom: Puzzle };
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
    <div className="space-y-10">
      <PageHeader eyebrow="New project" title="What do you want your AI to master?" lead="Pick a starting point. Every template comes with an automatic exam, so you can see proof it got better." />
      <ErrorNote error={error} />
      <div className="grid gap-5 sm:grid-cols-2">
        {templates.map((t) => (
          <ChoiceCard
            key={t.id}
            icon={ICON[t.id]}
            title={t.title}
            selected={chosen?.id === t.id}
            onClick={() => {
              setChosen(t);
              setName(DEFAULT_NAME[t.id]);
            }}
          >
            {t.tagline}
            <span className="mt-4 flex items-start gap-2 text-sm text-ink"><Icon icon={Check} className="mt-0.5 text-good" /> {t.proof_label}</span>
            <span className="mt-2 block text-sm">You’ll need: {t.needs}</span>
          </ChoiceCard>
        ))}
      </div>
      {chosen && (
        <Card className="flex flex-wrap items-end gap-4">
          <label className="min-w-[240px] flex-1">
            <span className="mb-2 block text-[15px] font-medium">Give it a name</span>
            <TextInput value={name} onChange={setName} onEnter={create} />
          </label>
          {models.length > 1 && (
            <label>
              <span className="mb-2 block text-[15px] font-medium">Model</span>
              <select value={modelId} onChange={(e) => setModelId(e.target.value)} className="block rounded-xl border border-line bg-panel px-3.5 py-2.5 text-[15px] outline-none focus:border-ink">
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{m.label ?? m.id}</option>
                ))}
              </select>
            </label>
          )}
          <Button onClick={create} busy={busy} disabled={!name.trim()}>
            Create project
          </Button>
        </Card>
      )}
      {projects.length > 0 && (
        <section className="space-y-5 pt-4">
          <SectionTitle title="Continue a project" />
          <ul className="divide-y divide-line border-y border-line">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center justify-between gap-4 py-4">
                <div className="min-w-0">
                  <div className="truncate text-[16px] font-semibold">{p.name}</div>
                  <div className="mt-0.5 text-sm text-muted">{templates.find((t) => t.id === p.template)?.title} · {p.active_version ? `trained version ${p.active_version}` : "not trained yet"}</div>
                </div>
                <Button variant="secondary" onClick={() => { const t = templates.find((t) => t.id === p.template); if (t) onCreated(p, t); }}>Open</Button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
