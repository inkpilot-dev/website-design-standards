import { useEffect, useState } from "react";
import type { AppError, IngestSummary, Preview, ProjectInfo, RuleReport, Template } from "../lib/contract";
import { api, onFileDrop, pickFiles, subscribe, toAppError } from "../lib/ipc";
import { Badge, Button, Card, Drawer, ErrorNote } from "../components/ui";

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
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Your material and your rules</h1>
        <p className="mt-2 text-muted">{template.needs}</p>
      </header>
      <ErrorNote error={error} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="space-y-3">
          <div className="text-sm font-semibold">1 · Add files</div>
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
            className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition ${hover ? "border-accent bg-accent-soft" : "border-line"}`}
          >
            <div className="text-3xl" aria-hidden>📥</div>
            <div className="mt-2 font-medium">Drop files or folders here</div>
            <div className="text-xs text-muted">{template.accepts.join("  ")}</div>
            <div className="mt-3 flex flex-wrap justify-center gap-2">
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
            <ul className="space-y-1 text-sm">
              {files.map((f) => (
                <li key={f} className="flex items-center justify-between rounded-lg bg-canvas px-3 py-1.5">
                  <span className="truncate" title={f}>{basename(f)}</span>
                  <button className="text-muted hover:text-bad" onClick={() => setFiles(files.filter((x) => x !== f))} aria-label={`Remove ${basename(f)}`}>✕</button>
                </li>
              ))}
            </ul>
          )}
          <p className="text-xs text-muted">🔒 Files stay on this computer. Personal details are detected and never shared.</p>
        </Card>

        <Card className="space-y-3">
          <div className="text-sm font-semibold">2 · Rules your AI must always follow <span className="font-normal text-muted">(optional)</span></div>
          <div className="flex gap-2">
            <input value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addRule(draft)} placeholder="e.g. amount must be positive" className="flex-1 rounded-xl border border-line bg-canvas px-3 py-2 text-sm outline-none focus:border-accent" />
            <Button variant="secondary" onClick={() => addRule(draft)} disabled={!draft.trim()}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {template.example_rules.filter((r) => !rules.includes(r)).map((r) => (
              <button key={r} onClick={() => addRule(r)} className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:border-accent hover:text-ink">+ {r}</button>
            ))}
          </div>
          <ul className="space-y-2">
            {rules.map((r) => {
              const c = checks[r];
              return (
                <li key={r} className="rounded-lg bg-canvas px-3 py-2 text-sm">
                  <div className="flex items-start justify-between gap-2">
                    <span>{c ? (c.ok ? "✓ " : "✗ ") : "… "}{r}</span>
                    <button className="text-muted hover:text-bad" onClick={() => setRules(rules.filter((x) => x !== r))} aria-label="Remove rule">✕</button>
                  </div>
                  {c && !c.ok && <div className="mt-1 text-xs text-bad">{c.error}</div>}
                  {c?.warning && <div className="mt-1 text-xs text-ink">⚠ {c.warning}</div>}
                  {c?.ok && advanced && <code className="mt-1 block text-xs text-muted">{c.expr}</code>}
                </li>
              );
            })}
          </ul>
        </Card>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={build} busy={busy} disabled={files.length === 0}>
          {summary ? "Rebuild practice exam" : "Build practice exam"}
        </Button>
        {summary && <Button variant="secondary" onClick={() => onBuilt(summary)}>Continue to training</Button>}
      </div>

      {summary && (
        <Card className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="good">✓ {summary.counts.train} practice questions</Badge>
            <Badge>{summary.counts.dev} for checking progress</Badge>
            <Badge tone="warn">🔒 {summary.counts.test} hidden final-exam questions</Badge>
            {summary.leakage_free && <Badge tone="good">✓ No final-exam question appears in training</Badge>}
            {summary.duplicates_removed > 0 && <Badge>{summary.duplicates_removed} duplicates removed</Badge>}
          </div>
          {summary.warnings.map((w) => (
            <div key={w} className="rounded-lg bg-warn-soft px-3 py-2 text-sm">⚠ {w}</div>
          ))}
          <div>
            <div className="mb-2 text-sm font-semibold">Sample practice questions (the final exam stays hidden)</div>
            <div className="grid gap-3 md:grid-cols-2">
              {preview?.items.map((it) => (
                <div key={it.item_id} className="rounded-xl border border-line p-3 text-sm">
                  <div className="line-clamp-4 whitespace-pre-wrap">{it.prompt}</div>
                  <div className="mt-2 rounded-lg bg-accent-soft px-2 py-1 font-mono text-xs">{it.reference}</div>
                  <div className="mt-1 text-xs text-muted">from {it.source}</div>
                </div>
              ))}
            </div>
          </div>
          <Drawer title="How answers are checked" open={advanced} onToggle={() => setAdvanced(!advanced)}>
            <ul className="space-y-1">
              {summary.verifiers.map((v) => (
                <li key={v.id} className="flex gap-2">
                  <Badge tone={v.role === "gate" ? "warn" : v.role === "audit" ? "neutral" : "good"}>{v.role === "gate" ? "must pass" : v.role === "audit" ? "audit" : "scored"}</Badge>
                  <span>{v.description || v.id}</span>
                  <code className="text-xs text-muted">{v.kind}</code>
                </li>
              ))}
            </ul>
          </Drawer>
        </Card>
      )}
    </div>
  );
}
