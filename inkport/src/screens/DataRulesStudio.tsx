import { useEffect, useRef, useState } from "react";
import { Check, FolderUp, Loader, Lock, Plus, TriangleAlert, X } from "lucide-react";
import type { AppError, IngestSummary, ProjectInfo, RuleReport, Template } from "../lib/contract";
import { api, inTauri, onFileDrop, pickFiles, subscribe, toAppError } from "../lib/ipc";
import type { DataDraft } from "../App";
import { Badge, Button, Card, Drawer, ErrorNote, Icon, Note, PageHeader, SectionTitle, Spinner, TextInput } from "../components/ui";

const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;
const SAMPLE_FILES: Record<Template["id"], string[]> = { business_rules: ["expenses.csv", "policy.md"], code_sql: ["orders.csv", "customers.csv", "helpers.py"], docs_legal: ["lease.md"], custom: [] };
const EMPTY: DataDraft = { files: [], rules: [], checks: {}, summary: null, preview: null, stale: false };

interface Props {
  project: ProjectInfo;
  template: Template;
  /** This project's files, rules and exam as last left (null the first time the step is opened). */
  draft: DataDraft | null;
  onDraft: (d: DataDraft) => void;
  trainingRunning: boolean;
  onBuilt: () => void | Promise<void>;
  onContinue: () => void;
}

export function DataRulesStudio({ project, template, draft, onDraft, trainingRunning, onBuilt, onContinue }: Props) {
  const d = draft ?? EMPTY;
  const latest = useRef(d);
  latest.current = d;
  // Always write on top of the newest draft: file drops and rule checks finish asynchronously.
  const update = (patch: Partial<DataDraft> | ((cur: DataDraft) => Partial<DataDraft>)) => {
    const cur = latest.current;
    const next = { ...cur, ...(typeof patch === "function" ? patch(cur) : patch) };
    latest.current = next;
    onDraft(next);
  };
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [hover, setHover] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [samples, setSamples] = useState<string | null>(null);

  useEffect(() => {
    api.sampleData().then(setSamples).catch(() => setSamples(null));
  }, []);

  // A project opened again after a restart already has an exam on disk: show what it was built from.
  useEffect(() => {
    if (draft || !project.counts) return;
    let alive = true;
    setRestoring(true);
    api
      .previewItems(project.id, 8)
      .then((pv) => {
        if (!alive || latest.current !== EMPTY) return;
        const reports = pv.rules ?? [];
        const summary: IngestSummary = { counts: pv.counts, leakage_free: pv.leakage_free ?? true, duplicates_removed: pv.duplicates_removed ?? 0, warnings: pv.warnings ?? [], rules: reports, verifiers: pv.verifiers ?? [], seconds: 0, ready: pv.ready ?? true, blocker: pv.blocker ?? null };
        update({ files: pv.inputs ?? [], rules: reports.map((r) => r.text), checks: Object.fromEntries(reports.map((r) => [r.text, r])), summary, preview: pv, stale: false });
      })
      .catch(() => {})
      .finally(() => alive && setRestoring(false));
    return () => {
      alive = false;
    };
  }, [project.id]);

  const addFiles = (paths: string[]) => {
    const accepted = paths.filter(Boolean);
    if (!accepted.length) return;
    update((cur) => {
      const files = Array.from(new Set([...cur.files, ...accepted]));
      return files.length === cur.files.length ? {} : { files, stale: !!cur.summary };
    });
  };

  useEffect(() => subscribe(onFileDrop(addFiles)), []);

  const addRule = async (raw: string) => {
    const t = raw.trim();
    if (!t || latest.current.rules.includes(t)) return;
    update((cur) => ({ rules: [...cur.rules, t], stale: !!cur.summary }));
    setText("");
    try {
      const res = await api.compileRules([t], project.id);
      update((cur) => ({ checks: { ...cur.checks, [t]: res.rules[0] } }));
    } catch (e) {
      // Without a verdict the rule would spin forever and still be sent to the engine.
      update((cur) => ({ checks: { ...cur.checks, [t]: { ok: false, text: t, expr: "", explanation: "", error: `This rule could not be checked (${toAppError(e).message}). Remove it and add it again.` } } }));
    }
  };

  const build = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const cur = latest.current;
      const good = cur.rules.filter((r) => cur.checks[r]?.ok !== false);
      const summary = await api.ingestDataset(project.id, cur.files, good);
      update({ summary, stale: false, preview: null });
      await onBuilt();
      const preview = await api.previewItems(project.id, 8).catch(() => null);
      update({ preview });
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setBusy(false);
    }
  };

  const summary = d.summary;
  const blocked = !!summary && summary.ready === false;
  const pendingRules = d.rules.some((r) => !d.checks[r]);
  const canContinue = !!summary && !d.stale && !blocked && !busy;

  return (
    <div className="space-y-10">
      <PageHeader eyebrow={project.name} title="Your material and your rules" lead={template.needs} />
      {trainingRunning && <Note icon={Lock}>Training is running for this project. You can look, but the practice exam cannot be rebuilt until training has finished or been stopped.</Note>}

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
              // In the desktop app the real paths arrive through Tauri's drop event (above); the
              // browser's own drop event carries no paths and is only used by the preview.
              if (inTauri) return;
              const names = Array.from(e.dataTransfer.files).map((f) => `~/Documents/${f.name}`);
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
          {restoring && <div className="flex items-center gap-2 text-sm text-muted"><Spinner /> Loading what this exam was built from…</div>}
          {d.files.length > 0 && (
            <ul className="divide-y divide-line border-y border-line text-[15px]">
              {d.files.map((f) => (
                <li key={f} className="flex items-center justify-between gap-3 py-2.5">
                  <span className="truncate font-mono text-[13px]" title={f}>{basename(f)}</span>
                  <button className="text-faint transition-colors hover:text-bad" onClick={() => update((cur) => ({ files: cur.files.filter((x) => x !== f), stale: !!cur.summary }))} aria-label={`Remove ${basename(f)}`}><Icon icon={X} /></button>
                </li>
              ))}
            </ul>
          )}
          <p className="flex items-start gap-2 text-sm leading-relaxed text-muted"><Icon icon={Lock} size={15} className="mt-0.5" /> Files stay on this computer. Nothing is uploaded.</p>
        </Card>

        <Card className="space-y-5">
          <h2 className="text-[17px] font-semibold tracking-[-0.01em]">Rules your AI must always follow <span className="font-normal text-muted">(optional)</span></h2>
          <div className="flex gap-2">
            <TextInput value={text} onChange={setText} onEnter={() => addRule(text)} placeholder="e.g. amount must be positive" label="New rule" />
            <Button variant="secondary" onClick={() => addRule(text)} disabled={!text.trim()}>Add</Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {template.example_rules.filter((r) => !d.rules.includes(r)).map((r) => (
              <button key={r} onClick={() => addRule(r)} className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-left text-[13px] text-muted transition-colors hover:border-ink hover:text-ink"><Icon icon={Plus} size={13} /> {r}</button>
            ))}
          </div>
          {d.rules.length > 0 && (
            <ul className="divide-y divide-line border-y border-line">
              {d.rules.map((r) => {
                const c: RuleReport | undefined = d.checks[r];
                return (
                  <li key={r} className="py-3 text-[15px]">
                    <div className="flex items-start justify-between gap-3">
                      <span className="flex min-w-0 items-start gap-2.5 break-words">
                        {c ? c.ok ? <Icon icon={Check} className="mt-1 text-good" /> : <Icon icon={X} className="mt-1 text-bad" /> : <Icon icon={Loader} className="mt-1 animate-spin text-faint" />}
                        {r}
                      </span>
                      <button className="mt-1 text-faint transition-colors hover:text-bad" onClick={() => update((cur) => ({ rules: cur.rules.filter((x) => x !== r), stale: !!cur.summary }))} aria-label="Remove rule"><Icon icon={X} /></button>
                    </div>
                    {c && !c.ok && <div className="mt-1.5 pl-[26px] text-sm text-bad">{c.error} This rule will not be used.</div>}
                    {c?.warning && <div className="mt-1.5 flex items-start gap-2 pl-[26px] text-sm"><Icon icon={TriangleAlert} size={14} className="mt-0.5 text-warn" /> {c.warning}</div>}
                    {c?.ok && advanced && <code className="mt-1.5 block pl-[26px] font-mono text-xs text-muted">{c.expr}</code>}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      </div>

      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={build} busy={busy} disabled={d.files.length === 0 || trainingRunning || pendingRules} variant={summary && !d.stale ? "secondary" : "primary"}>
            {busy ? "Building…" : summary ? "Rebuild practice exam" : "Build practice exam"}
          </Button>
          {summary && <Button onClick={onContinue} disabled={!canContinue}>Continue to training</Button>}
          {pendingRules && <span className="text-sm text-muted">Checking your rules…</span>}
        </div>
        {summary && d.stale && !busy && <Note tone="warn" icon={TriangleAlert}>You changed the files or rules after the exam was built. Rebuild the practice exam to use the changes.</Note>}
        <ErrorNote error={error} title="The practice exam could not be built" />
      </div>

      {summary && (
        <section className="space-y-6 pt-4">
          <SectionTitle title="Your practice exam" lead="These questions were built from your files. The final exam is kept hidden from training, so the result cannot be memorized." />
          {blocked && <Note tone="warn" icon={TriangleAlert}><span className="font-semibold">Not enough examples to train yet.</span> {summary.blocker}</Note>}
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
            <div key={w} className="flex items-start gap-2.5 rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 text-[15px] leading-relaxed"><Icon icon={TriangleAlert} className="mt-1 text-warn" /> <span className="min-w-0 break-words">{w}</span></div>
          ))}
          {d.preview && d.preview.items.length > 0 && (
            <div className="grid gap-5 md:grid-cols-2">
              {d.preview.items.map((it) => (
                <div key={it.item_id} className="rounded-2xl border border-line p-5 text-[15px]">
                  <div className="line-clamp-4 whitespace-pre-wrap break-words leading-relaxed">{it.prompt}</div>
                  <div className="mt-3 break-words rounded-lg bg-canvas px-3 py-2 font-mono text-[13px]">{it.reference}</div>
                  <div className="mt-3 text-[13px] text-muted">from {it.source}</div>
                </div>
              ))}
            </div>
          )}
          {summary.verifiers.length > 0 && (
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
          )}
        </section>
      )}
    </div>
  );
}
