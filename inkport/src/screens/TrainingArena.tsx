import { useEffect, useRef, useState } from "react";
import type { AppError, EngineEvent, ProjectInfo } from "../lib/contract";
import { api, on, subscribe, toAppError } from "../lib/ipc";
import { Badge, Button, Card, Drawer, ErrorNote, fmtEta, LiftGauge, ProgressBar, Sparkline } from "../components/ui";

type Bench = Extract<EngineEvent, { type: "benchmark" }>;
type Gate = Extract<EngineEvent, { type: "gate" }>;

const STAGES: { id: string; label: string }[] = [
  { id: "verify_data", label: "Check data" },
  { id: "baseline", label: "Test the original" },
  { id: "sft", label: "Learn" },
  { id: "stage2", label: "Practice" },
  { id: "gate", label: "Final exam" },
  { id: "promote", label: "Save" },
];

export function TrainingArena({ project, onDeploy }: { project: ProjectInfo; onDeploy: () => void }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [overall, setOverall] = useState(0);
  const [eta, setEta] = useState<number | null>(null);
  const [stageState, setStageState] = useState<Record<string, string>>({});
  const [stageDetail, setStageDetail] = useState("");
  const [baseRate, setBaseRate] = useState<number | null>(null);
  const [devPass, setDevPass] = useState<number | null>(null);
  const [candRate, setCandRate] = useState<number | null>(null);
  const [loss, setLoss] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number | null>>({});
  const [duels, setDuels] = useState<Bench[]>([]);
  const [arena, setArena] = useState<Bench[]>([]);
  const [gate, setGate] = useState<Gate | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<AppError | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [asking, setAsking] = useState(false);
  const jobRef = useRef<string | null>(null);

  useEffect(() => {
    const mine = (e: EngineEvent) => !jobRef.current || e.job_id === jobRef.current || e.job_kind === "arena";
    const progress = on<EngineEvent>("training://progress", (e) => {
      if (!mine(e)) return;
      if (e.type === "progress") {
        setOverall(e.overall_pct);
        setEta(e.eta_s);
        setMetrics(e.metrics);
        const l = e.metrics.loss;
        if (typeof l === "number") setLoss((prev) => [...prev.slice(-119), l]);
        if (typeof e.metrics.dev_pass === "number") setDevPass(e.metrics.dev_pass);
      } else if (e.type === "stage") {
        const key = e.stage.startsWith("rft") || e.stage === "grpo" ? "stage2" : e.stage;
        setStageState((s) => ({ ...s, [key]: e.status === "completed" || e.status === "skipped" ? "done" : e.status === "failed" ? "failed" : s[key] === "done" ? "done" : "active" }));
        if (e.detail) setStageDetail(e.detail);
      } else if (e.type === "eval" && e.split === "test") {
        if (e.model === "base") setBaseRate(e.pass_rate);
        if (e.model === "candidate") setCandRate(e.pass_rate);
      } else if (e.type === "gate") {
        setGate(e);
      } else if (e.type === "error") {
        setError({ code: e.code, message: e.message, retryable: e.retryable });
      } else if (e.type === "done" && e.job_kind !== "arena") {
        setRunning(false);
        setOverall(100);
        setEta(0);
      }
    });
    const results = on<EngineEvent>("benchmark://result", (e) => {
      if (e.type !== "benchmark") return;
      if (e.source === "training_dev") setDuels((d) => [e, ...d.filter((x) => x.case_id !== e.case_id)].slice(0, 6));
      else setArena((a) => [e, ...a].slice(0, 6));
    });
    const logLines = on<EngineEvent>("training://log", (e) => {
      if (e.type === "log" && e.level !== "debug") setLogs((l) => [...l.slice(-200), `${e.level.toUpperCase()}  ${e.message}`]);
    });
    return subscribe(progress, results, logLines);
  }, []);

  const start = async (preset: "standard" | "fast" | "smoke") => {
    setError(null);
    setGate(null);
    setDuels([]);
    setLoss([]);
    setStageState({});
    setOverall(0);
    setCandRate(null);
    setDevPass(null);
    setRunning(true);
    try {
      const id = await api.startTrainingJob(project.id, preset);
      jobRef.current = id;
      setJobId(id);
    } catch (e) {
      setRunning(false);
      setError(toAppError(e));
    }
  };

  const cancel = async () => {
    if (jobId) await api.cancelTrainingJob(jobId).catch((e) => setError(toAppError(e)));
  };

  const ask = async () => {
    setAsking(true);
    try {
      await api.runBenchmarkArena(project.id, prompt || undefined, prompt ? 1 : 3);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setAsking(false);
    }
  };

  const promoted = gate?.decision === "PROMOTE";
  const finished = !running && (gate !== null || overall >= 100);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Train “{project.name}”</h1>
          <p className="mt-2 text-muted">Your AI studies your examples, then sits a final exam it has never seen. It is only kept if it beats the original.</p>
        </div>
        {!running && !finished && (
          <div className="flex gap-2">
            <Button onClick={() => start("standard")}>Start training (best results)</Button>
            <Button variant="secondary" onClick={() => start("fast")}>Quick run</Button>
          </div>
        )}
        {running && <Button variant="danger" onClick={cancel}>Stop</Button>}
      </header>
      <ErrorNote error={error} onRetry={error?.retryable ? () => start("standard") : undefined} />

      {(running || finished) && (
        <Card className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm text-muted">{running ? stageDetail || "Starting…" : "Finished"}</div>
            <div className="text-sm font-semibold tabular-nums">{Math.round(overall)}% · {running ? `${fmtEta(eta)} left in this step` : "done"}</div>
          </div>
          <ProgressBar value={overall} label="Overall training progress" />
          <ol className="flex flex-wrap gap-2">
            {STAGES.map((s) => {
              const st = stageState[s.id];
              return (
                <li key={s.id} className={`rounded-full border px-3 py-1 text-xs ${st === "done" ? "border-accent/40 bg-accent-soft text-accent" : st === "active" ? "border-accent text-ink" : st === "failed" ? "border-bad text-bad" : "border-line text-muted"}`}>
                  {st === "done" ? "✓ " : st === "active" ? "● " : ""}{s.label}
                </li>
              );
            })}
          </ol>
        </Card>
      )}

      {(running || finished) && (
        <div className="grid gap-4 lg:grid-cols-5">
          <Card className="lg:col-span-2">
            <LiftGauge label={candRate !== null ? "Final exam (hidden questions)" : "Practice score so far"} before={baseRate} after={candRate ?? devPass} />
            <p className="mt-3 text-xs text-muted">“Original” is the untouched model on the hidden exam. {candRate === null ? "“Yours” is your model on practice questions while it learns." : "“Yours” is your trained model on the same hidden exam."}</p>
          </Card>
          <Card className="space-y-3 lg:col-span-3">
            <div className="text-sm font-semibold">Proof Arena: same question, both models</div>
            {duels.length === 0 && <div className="text-sm text-muted">Side-by-side answers appear here as training checks its progress.</div>}
            {duels.slice(0, 3).map((d) => (
              <Duel key={d.case_id} d={d} />
            ))}
          </Card>
        </div>
      )}

      {gate && (
        <Card className={promoted ? "border-accent/50" : "border-warn/60"}>
          <div className="flex flex-wrap items-center gap-2">
            {promoted ? <Badge tone="good">✓ Better: new version {gate.version ?? ""} saved</Badge> : gate.decision === "INSUFFICIENT_DATA" ? <Badge tone="warn">Need more examples</Badge> : <Badge tone="warn">Not better yet: your previous version stays active</Badge>}
            {gate.label && <Badge>{gate.label === "VERIFIED" ? "Verified result" : "Preliminary result"}</Badge>}
          </div>
          <p className="mt-3 text-[15px] leading-relaxed">{gate.explanation}</p>
          <div className="mt-4 flex flex-wrap gap-3">
            {promoted && <Button onClick={onDeploy}>Use it in your apps</Button>}
            <Button variant="secondary" onClick={() => start("standard")}>Train again</Button>
          </div>
        </Card>
      )}

      {finished && promoted && (
        <Card className="space-y-3">
          <div className="text-sm font-semibold">Ask both models yourself</div>
          <div className="flex gap-2">
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()} placeholder="Type a new example, or leave empty to use practice questions" className="flex-1 rounded-xl border border-line bg-canvas px-3 py-2 text-sm outline-none focus:border-accent" />
            <Button onClick={ask} busy={asking}>Compare</Button>
          </div>
          {arena.map((d, i) => (
            <Duel key={d.case_id + i} d={d} />
          ))}
        </Card>
      )}

      <Drawer title="Advanced metrics" open={advanced} onToggle={() => setAdvanced(!advanced)}>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="text-xs text-muted">Training loss</div>
            <Sparkline values={loss} />
          </div>
          <dl className="grid grid-cols-2 gap-1 font-mono text-xs">
            {Object.entries(metrics).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k}</dt>
                <dd>{v === null ? "–" : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : Number(v.toFixed(4))}</dd>
              </div>
            ))}
          </dl>
        </div>
        <pre className="mt-3 max-h-56 overflow-auto rounded-lg bg-canvas p-2 font-mono text-[11px] leading-snug">{logs.join("\n") || "Logs appear here."}</pre>
      </Drawer>
    </div>
  );
}

function Duel({ d }: { d: Bench }) {
  const mark = (p: boolean | null) => (p === null ? "" : p ? "✓" : "✗");
  return (
    <div className="rounded-xl border border-line p-3 text-sm">
      <div className="mb-2 line-clamp-2 font-medium">{d.prompt}</div>
      <div className="grid gap-2 sm:grid-cols-2">
        {([["Original", d.baseline], ["Yours", d.trained]] as const).map(([name, side]) => (
          <div key={name} className={`rounded-lg p-2 ${side.pass ? "bg-accent-soft" : side.pass === false ? "bg-bad-soft" : "bg-canvas"}`}>
            <div className="text-xs font-semibold">{name} {mark(side.pass)}</div>
            <div className="mt-1 line-clamp-4 whitespace-pre-wrap break-words font-mono text-xs">{side.output || "(no answer)"}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
