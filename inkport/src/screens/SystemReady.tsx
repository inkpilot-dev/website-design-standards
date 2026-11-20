import { useEffect, useState } from "react";
import type { AppError, HardwareReport, RuntimeProgress, RuntimeStatus } from "../lib/contract";
import { api, on, toAppError } from "../lib/ipc";
import { Badge, Button, Card, Drawer, ErrorNote, ProgressBar, Spinner } from "../components/ui";

export function SystemReady({ onReady }: { onReady: (hw: HardwareReport) => void }) {
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const [rt, setRt] = useState<RuntimeStatus | null>(null);
  const [steps, setSteps] = useState<RuntimeProgress[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [advanced, setAdvanced] = useState(false);

  const load = async () => {
    setError(null);
    try {
      const [h, r] = await Promise.all([api.detectHardware(), api.runtimeStatus()]);
      setHw(h);
      setRt(r);
    } catch (e) {
      setError(toAppError(e));
    }
  };

  useEffect(() => {
    void load();
    let un: (() => void) | undefined;
    void on<RuntimeProgress>("runtime://progress", (p) =>
      setSteps((prev) => {
        const next = prev.filter((s) => s.step !== p.step);
        return [...next, p].sort((a, b) => a.step - b.step);
      }),
    ).then((u) => (un = u));
    return () => un?.();
  }, []);

  const setup = async () => {
    setBusy(true);
    setError(null);
    setSteps([]);
    try {
      setRt(await api.bootstrapRuntime(true));
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setBusy(false);
    }
  };

  const pct = steps.length ? Math.max(...steps.map((s) => s.pct)) : 0;
  const rec = hw?.recommendation;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Let’s check your computer</h1>
        <p className="mt-2 text-muted">Barnraise trains a private AI on this machine. Nothing you add leaves your computer.</p>
      </header>
      <ErrorNote error={error} onRetry={load} />
      <Card>
        {!hw ? (
          <div className="flex items-center gap-3 text-muted">
            <Spinner /> Looking at your graphics card, memory and disk…
          </div>
        ) : (
          <ul className="space-y-3">
            {hw.summary.map((line, i) => (
              <li key={i} className="flex items-start gap-3">
                <span className={`mt-0.5 inline-flex h-5 w-5 flex-none items-center justify-center rounded-full text-xs ${line.startsWith("Only") || line.startsWith("No supported") ? "bg-warn-soft text-ink" : "bg-accent-soft text-accent"}`}>{line.startsWith("Only") || line.startsWith("No supported") ? "!" : "✓"}</span>
                <span className={i === hw.summary.length - 1 ? "font-semibold" : ""}>{line}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {rec && (
        <Card className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm text-muted">Setup</div>
              <div className="text-lg font-semibold">{rt?.ready ? "Everything is installed" : "One click installs a private copy of everything Barnraise needs"}</div>
            </div>
            {rt?.ready ? <Badge tone="good">Ready · engine {rt.engine_version}</Badge> : <Badge>Not set up yet</Badge>}
          </div>
          {(busy || steps.length > 0) && (
            <div className="space-y-3">
              <ProgressBar value={pct} label="Setup progress" />
              <ol className="space-y-1.5 text-sm">
                {steps.map((s) => (
                  <li key={s.step} className="flex items-center gap-2">
                    {s.status === "completed" ? <span className="text-accent">✓</span> : <Spinner />}
                    <span>{s.label}</span>
                    {s.detail && <span className="truncate text-xs text-muted">{s.detail}</span>}
                  </li>
                ))}
              </ol>
            </div>
          )}
          <div className="flex flex-wrap gap-3">
            {!rt?.ready && (
              <Button onClick={setup} busy={busy}>
                Set up everything
              </Button>
            )}
            <Button variant={rt?.ready ? "primary" : "secondary"} disabled={!rt?.ready} onClick={() => hw && onReady(hw)}>
              Continue
            </Button>
          </div>
          <p className="text-xs text-muted">No terminal needed. Setup downloads Python, the AI libraries for your {rec.vendor === "apple" ? "Mac" : rec.vendor === "nvidia" ? "NVIDIA card" : "computer"}, and the {rec.model_label} model.</p>
        </Card>
      )}

      {hw && (
        <Drawer title="Advanced details" open={advanced} onToggle={() => setAdvanced(!advanced)}>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs">
            {Object.entries({ tier: rec?.tier, backend: rec?.backend, quantization: rec?.quant, "LoRA rank": rec?.lora_r, "micro-batch × accumulation": `${rec?.micro_batch} × ${rec?.grad_accum}`, "max sequence": rec?.max_seq, refinement: `${rec?.stage2} (k=${rec?.rft_k})`, cpu: `${hw.cpu_brand} (${hw.cpu_cores} cores)`, ram: `${hw.ram_gb} GB`, disk: `${hw.disk_free_gb} GB free`, python: rt?.python ?? "–", home: rt?.home ?? "–" }).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k}</dt>
                <dd className="truncate">{String(v)}</dd>
              </div>
            ))}
            {hw.gpus.map((g, i) => (
              <div key={i} className="contents">
                <dt className="text-muted">gpu {i}</dt>
                <dd className="truncate">{g.name} · {g.vram_gb} GB{g.unified ? " unified" : ""}{g.driver ? ` · driver ${g.driver}` : ""}</dd>
              </div>
            ))}
          </dl>
        </Drawer>
      )}
    </div>
  );
}
