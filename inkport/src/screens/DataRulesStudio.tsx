import { useEffect, useState } from "react";
import type { AppError, IngestSummary, Preview, ProjectInfo, RuleReport, Template } from "../lib/contract";
import { api, onFileDrop, pickFiles, subscribe, toAppError } from "../lib/ipc";
import { Check, FolderUp, Loader, Lock, Plus, TriangleAlert, X } from "lucide-react";
import { Badge, Button, Card, Drawer, ErrorNote, Icon, PageHeader, SectionTitle, TextInput } from "../components/ui";

const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

export function DataRulesStudio({ project, template, onBuilt }: { project: ProjectInfo; template: Template; onBuilt: (s: IngestSummary) => void }) {
  const [files, setFiles] = useState<string[]>([]);
  const [rules, setRules] = useState<string[]>([]);
  const [draft, setDraft] = useState("");
  const [checks, setChecks] = useState<Record<string, RuleReport>>({});
  const [summary, setSummary] = useState<IngestSummary | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [hover, setHover] = useState(false);
  const [advanced, setAdvanced] = useState(false);

  const addFiles = (paths: string[]) => setFiles((prev) => Array.from(new Set([...prev, ...paths])));
  const [samples, setSamples] = useState<string | null>(null);
  const SAMPLE_FILES: Record<Template["id"], string[]> = { business_rules: ["expenses.csv", "policy.md"], code_sql: ["orders.csv", "customers.csv", "helpers.py"], docs_legal: ["lease.md"], custom: [] };
  useEffect(() => {
    api.sampleData().then(setSamples).catch(() => setSamples(null));
  }, []);

  useEffect(() => {
    return subscribe(onFileDrop(addFiles));
  }, []);

  const addRule = async (text: string) => {
    const t = text.trim();
    if (!t || rules.includes(t)) return;
    setRules((r) => [...r, t]);
    setDraft("");
    try {
      const res = await api.compileRules([t], project.id);
      setChecks((c) => ({ ...c, [t]: res.rules[0] }));
    } catch (e) {
      setError(toAppError(e));
    }
  };

  const build = async () => {
    setBusy(true);
    setError(null);
    try {
      const good = rules.filter((r) => checks[r]?.ok !== false);
      const s = await api.ingestDataset(project.id, files, good);
      setSummary(s);
      setPreview(await api.previewItems(project.id, 8));
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-10">
      <PageHeader eyebrow={project.name} title="Your material and your rules" lead={template.needs} />
      <ErrorNote error={error} />

      <div className="grid gap-5 lg:grid-cols-2">
        <Card className="space-y-5">
          <h2 className="text-[17px] font-semibold tracking-[-0.01em]">Add files</h2>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setHover(true);
            }}
            onDragLeave={() => setHover(false)}
            onDrop={(e) => {
              e.preventDefault();
              setHover(false);
              const names = Array.from(e.dataTransfer.files).map((f) => (f as File & { path?: string }).path ?? `~/Documents/${f.name}`);
              if (names.length) addFiles(names);
            }}
            className={`flex flex-col items-center justify-center rounded-xl border border-dashed px-6 py-9 text-center transition-colors ${hover ? "border-ink bg-canvas" : "border-faint"}`}
          >
            <Icon icon={FolderUp} size={26} />
            <div className="mt-4 text-[15px] font-medium">Drop files or folders here</div>
            <div className="mt-1 font-mono text-xs text-muted">{template.accepts.join("  ")}</div>
            <div className="mt-5 flex w-full flex-wrap justify-center gap-2">
              <Button variant="secondary" onClick={async () => addFiles(await pickFiles(template.accepts))}>
                Choose files…
              </Button>
              {samples && SAMPLE_FILES[template.id].length > 0 && (
                <Button variant="ghost" onClick={() => addFiles(SAMPLE_FILES[template.id].map((f) => `${samples}/${f}`))}>
                  Try the sample data
                </Button>
              )}
            </div>
          </div>
          {files.length > 0 && (
            <ul className="divide-y divide-line border-y border-line text-[15px]">
              {files.map((f) => (
                <li key={f} className="flex items-center justify-between gap-3 py-2.5">
                  <span className="truncate font-mono text-[13px]" title={f}>{basename(f)}</span>
                  <button className="text-faint transition-colors hover:text-bad" onClick={() => setFiles(files.filter((x) => x !== f))} aria-label={`Remove ${basename(f)}`}><Icon icon={X} /></button>
                </li>
              ))}
            </ul>
          )}
          <p className="flex items-start gap-2 text-sm leading-relaxed text-muted"><Icon icon={Lock} size={15} className="mt-0.5" /> Files stay on this computer. Personal details are detected and never shared.</p>
        </Card>

        <Card className="space-y-5">
          <h2 className="text-[17px] font-semibold tracking-[-0.01em]">Rules your AI must always follow <span className="font-normal text-muted">(optional)</span></h2>
          <div className="flex gap-2">
            <TextInput value={draft} onChange={setDraft} onEnter={() => addRule(draft)} placeholder="e.g. amount must be positive" label="New rule" />
            <Button variant="secondary" onClick={() => addRule(draft)} disabled={!draft.trim()}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {template.example_rules.filter((r) => !rules.includes(r)).map((r) => (
              <button key={r} onClick={() => addRule(r)} className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-left text-[13px] text-muted transition-colors hover:border-ink hover:text-ink"><Icon icon={Plus} size={13} /> {r}</button>
            ))}
          </div>
          {rules.length > 0 && (
            <ul className="divide-y divide-line border-y border-line">
              {rules.map((r) => {
                const c = checks[r];
                return (
                  <li key={r} className="py-3 text-[15px]">
                    <div className="flex items-start justify-between gap-3">
                      <span className="flex items-start gap-2.5">
                        {c ? c.ok ? <Icon icon={Check} className="mt-1 text-good" /> : <Icon icon={X} className="mt-1 text-bad" /> : <Icon icon={Loader} className="mt-1 animate-spin text-faint" />}
                        {r}
                      </span>
                      <button className="mt-1 text-faint transition-colors hover:text-bad" onClick={() => setRules(rules.filter((x) => x !== r))} aria-label="Remove rule"><Icon icon={X} /></button>
                    </div>
                    {c && !c.ok && <div className="mt-1.5 pl-[26px] text-sm text-bad">{c.error}</div>}
                    {c?.warning && <div className="mt-1.5 flex items-start gap-2 pl-[26px] text-sm"><Icon icon={TriangleAlert} size={14} className="mt-0.5 text-warn" /> {c.warning}</div>}
                    {c?.ok && advanced && <code className="mt-1.5 block pl-[26px] font-mono text-xs text-muted">{c.expr}</code>}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={build} busy={busy} disabled={files.length === 0} variant={summary ? "secondary" : "primary"}>
          {summary ? "Rebuild practice exam" : "Build practice exam"}
        </Button>
        {summary && <Button onClick={() => onBuilt(summary)}>Continue to training</Button>}
      </div>

      {summary && (
        <section className="space-y-6 pt-4">
          <SectionTitle title="Your practice exam" lead="These questions were built from your files. The final exam is kept hidden from training, so the result cannot be memorized." />
          <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-3">
            {(
              [
                [summary.counts.train, "practice questions"],
                [summary.counts.dev, "for checking progress"],
                [summary.counts.test, "hidden final-exam questions"],
              ] as const
            ).map(([n, label]) => (
              <div key={label} className="bg-panel p-6">
                <dd className="text-[32px] font-bold leading-none tracking-[-0.03em] tabular-nums">{n.toLocaleString()}</dd>
                <dt className="mt-2 text-sm text-muted">{label}</dt>
              </div>
            ))}
          </dl>
          <div className="flex flex-wrap items-center gap-2">
            {summary.leakage_free && <Badge tone="good"><Icon icon={Check} size={13} /> No final-exam question appears in training</Badge>}
            {summary.duplicates_removed > 0 && <Badge>{summary.duplicates_removed} duplicates removed</Badge>}
          </div>
          {summary.warnings.map((w) => (
            <div key={w} className="flex items-start gap-2.5 rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-[15px]"><Icon icon={TriangleAlert} className="mt-1 text-warn" /> {w}</div>
          ))}
          <div className="grid gap-5 md:grid-cols-2">
            {preview?.items.map((it) => (
              <div key={it.item_id} className="rounded-2xl border border-line p-5 text-[15px]">
                <div className="line-clamp-4 whitespace-pre-wrap leading-relaxed">{it.prompt}</div>
                <div className="mt-3 break-words rounded-lg bg-canvas px-3 py-2 font-mono text-[13px]">{it.reference}</div>
                <div className="mt-3 text-[13px] text-muted">from {it.source}</div>
              </div>
            ))}
          </div>
          <Drawer title="How answers are checked" open={advanced} onToggle={() => setAdvanced(!advanced)}>
            <ul className="space-y-2.5">
              {summary.verifiers.map((v) => (
                <li key={v.id} className="flex flex-wrap items-center gap-2.5">
                  <Badge tone={v.role === "gate" ? "warn" : v.role === "audit" ? "neutral" : "good"}>{v.role === "gate" ? "must pass" : v.role === "audit" ? "audit" : "scored"}</Badge>
                  <span className="text-[15px]">{v.description || v.id}</span>
                  <code className="font-mono text-xs text-muted">{v.kind}</code>
                </li>
              ))}
            </ul>
          </Drawer>
        </section>
      )}
    </div>
  );
}
