import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-2xl border border-line bg-panel p-5 shadow-sm ${className}`}>{children}</div>;
}

type Variant = "primary" | "secondary" | "ghost" | "danger";
export function Button({ children, onClick, variant = "primary", disabled, busy, className = "", title }: { children: ReactNode; onClick?: () => void; variant?: Variant; disabled?: boolean; busy?: boolean; className?: string; title?: string }) {
  const styles: Record<Variant, string> = {
    primary: "bg-accent text-on-accent hover:brightness-110",
    secondary: "border border-line bg-panel hover:bg-canvas",
    ghost: "text-muted hover:text-ink",
    danger: "border border-bad/40 text-bad hover:bg-bad-soft",
  };
  return (
    <button title={title} onClick={onClick} disabled={disabled || busy} className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${styles[variant]} ${className}`}>
      {busy && <Spinner />}
      {children}
    </button>
  );
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden />;
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "warn" | "bad" }) {
  const t = { neutral: "bg-canvas text-muted border-line", good: "bg-accent-soft text-accent border-accent/30", warn: "bg-warn-soft text-ink border-warn/40", bad: "bg-bad-soft text-bad border-bad/30" }[tone];
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${t}`}>{children}</span>;
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-line" role="progressbar" aria-valuenow={Math.round(v)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
        <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${v}%` }} />
      </div>
    </div>
  );
}

export function ErrorNote({ error, onRetry }: { error: { code: string; message: string } | null; onRetry?: () => void }) {
  if (!error) return null;
  return (
    <div className="rounded-xl border border-bad/30 bg-bad-soft p-3 text-sm" role="alert">
      <div className="font-semibold text-bad">Something went wrong</div>
      <div className="mt-1 whitespace-pre-wrap">{error.message}</div>
      {onRetry && (
        <button className="mt-2 text-sm font-semibold text-bad underline" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Drawer({ title, open, onToggle, children }: { title: string; open: boolean; onToggle: () => void; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-line">
      <button onClick={onToggle} className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold text-muted hover:text-ink" aria-expanded={open}>
        {title}
        <span className={`transition ${open ? "rotate-90" : ""}`}>›</span>
      </button>
      {open && <div className="border-t border-line px-4 py-3 text-sm">{children}</div>}
    </div>
  );
}

export function Sparkline({ values, height = 40 }: { values: number[]; height?: number }) {
  if (values.length < 2) return <div className="text-xs text-muted">collecting…</div>;
  const w = 220;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${height - ((v - min) / (max - min || 1)) * (height - 4) - 2}`).join(" ");
  return (
    <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} className="text-accent" aria-hidden>
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth={2} strokeLinejoin="round" />
    </svg>
  );
}

export function LiftGauge({ before, after, label }: { before: number | null; after: number | null; label: string }) {
  const b = before ?? 0;
  const a = after ?? 0;
  const lift = after !== null && before !== null ? Math.round((a - b) * 100) : null;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <div className="text-sm font-semibold">{label}</div>
        {lift !== null && <div className={`text-2xl font-bold ${lift > 0 ? "text-accent" : lift < 0 ? "text-bad" : "text-muted"}`}>{lift > 0 ? "+" : ""}{lift} pts</div>}
      </div>
      {[["Original", before, "bg-muted/50"], ["Yours", after, "bg-accent"]].map(([name, v, color]) => (
        <div key={name as string} className="flex items-center gap-3">
          <div className="w-16 text-xs text-muted">{name as string}</div>
          <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-line">
            <div className={`h-full rounded-full ${color as string} transition-[width] duration-700`} style={{ width: `${Math.round(((v as number | null) ?? 0) * 100)}%` }} />
          </div>
          <div className="w-10 text-right text-xs tabular-nums">{v === null ? "–" : `${Math.round((v as number) * 100)}%`}</div>
        </div>
      ))}
    </div>
  );
}

export function fmtEta(s: number | null | undefined): string {
  if (s === null || s === undefined || !isFinite(s)) return "estimating…";
  if (s < 60) return "less than a minute";
  const m = Math.round(s / 60);
  if (m < 60) return `about ${m} minute${m === 1 ? "" : "s"}`;
  const h = Math.floor(m / 60);
  return `about ${h} h ${m % 60} min`;
}
