import { type ReactNode, useEffect, useRef, useState } from "react";
import { Check, ChevronRight, Clipboard, type LucideIcon } from "lucide-react";
import { copyText } from "../lib/ipc";

/** Line icon at the size and weight used throughout the app. */
export function Icon({ icon: I, size = 16, className = "" }: { icon: LucideIcon; size?: number; className?: string }) {
  return <I size={size} strokeWidth={1.75} className={`flex-none ${className}`} aria-hidden />;
}

/** Page title block: a quiet line above, a large title, one lead paragraph, an optional action on the right. */
export function PageHeader({ eyebrow, title, lead, action }: { eyebrow: string; title: string; lead: ReactNode; action?: ReactNode }) {
  return (
    <header>
      <div className="text-[15px] text-muted">{eyebrow}</div>
      <div className="mt-2 flex items-start justify-between gap-6">
        <h1 className="min-w-0 flex-1 text-[40px] font-bold leading-[1.1] tracking-[-0.03em]">{title}</h1>
        {action && <div className="flex flex-none gap-2 pt-1">{action}</div>}
      </div>
      <p className="mt-6 max-w-[62ch] text-[17px] leading-[1.65] text-muted">{lead}</p>
    </header>
  );
}

export function SectionTitle({ title, lead }: { title: string; lead?: ReactNode }) {
  return (
    <div>
      <h2 className="text-[26px] font-bold leading-tight tracking-[-0.02em]">{title}</h2>
      {lead && <p className="mt-3 max-w-[62ch] text-[16px] leading-[1.6] text-muted">{lead}</p>}
    </div>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-2xl border border-line bg-panel p-6 ${className}`}>{children}</div>;
}

/** A card that is one big button: icon, bold title, grey description. */
export function ChoiceCard({ icon, title, children, selected, onClick, disabled }: { icon: LucideIcon; title: string; children: ReactNode; selected?: boolean; onClick: () => void; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled} aria-pressed={selected} className={`flex flex-col rounded-2xl border bg-panel p-7 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${selected ? "border-ink" : "border-line hover:border-ink"}`}>
      <Icon icon={icon} size={26} />
      <div className="mt-5 text-[17px] font-semibold tracking-[-0.01em]">{title}</div>
      <div className="mt-1.5 text-[15px] leading-[1.6] text-muted">{children}</div>
    </button>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger";
export function Button({ children, onClick, variant = "primary", disabled, busy, className = "", title }: { children: ReactNode; onClick?: () => void; variant?: Variant; disabled?: boolean; busy?: boolean; className?: string; title?: string }) {
  const styles: Record<Variant, string> = {
    primary: "border border-ink bg-ink text-panel hover:opacity-85",
    secondary: "border border-line bg-panel hover:border-ink",
    ghost: "border border-transparent text-muted hover:text-ink",
    danger: "border border-line bg-panel text-bad hover:border-bad",
  };
  return (
    <button title={title} onClick={onClick} disabled={disabled || busy} className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-[15px] font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${styles[variant]} ${className}`}>
      {busy && <Spinner />}
      {children}
    </button>
  );
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 flex-none animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden />;
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "warn" | "bad" }) {
  const t = { neutral: "border-line text-muted", good: "border-good/25 bg-good-soft text-good", warn: "border-warn/30 bg-warn-soft text-ink", bad: "border-bad/25 bg-bad-soft text-bad" }[tone];
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[13px] font-medium leading-none ${t}`}>{children}</span>;
}

/** Short inline code or a value to copy, on a light grey chip. */
export function Code({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <code className={`rounded-md bg-canvas px-1.5 py-0.5 font-mono text-[0.9em] ${className}`}>{children}</code>;
}

export function TextInput({ value, onChange, onEnter, placeholder, label, disabled }: { value: string; onChange: (v: string) => void; onEnter?: () => void; placeholder?: string; label?: string; disabled?: boolean }) {
  return (
    <input
      value={value}
      aria-label={label}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        // The Enter that confirms a Japanese/Chinese/Korean input-method candidate is not "submit".
        if (e.key === "Enter" && !e.nativeEvent.isComposing && e.keyCode !== 229) onEnter?.();
      }}
      placeholder={placeholder}
      className="w-full min-w-0 flex-1 rounded-xl border border-line bg-panel px-3.5 py-2.5 text-[15px] outline-none transition-colors placeholder:text-faint focus:border-ink disabled:opacity-60"
    />
  );
}

/** "Copy" that only says "Copied" when it worked. */
export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const copy = async () => {
    setState((await copyText(text)) ? "copied" : "failed");
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState("idle"), 1800);
  };
  return (
    <button className="inline-flex flex-none items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-[13px] font-medium transition-colors hover:border-ink" onClick={copy}>
      <Icon icon={state === "copied" ? Check : Clipboard} size={14} /> {state === "copied" ? "Copied" : state === "failed" ? "Could not copy" : label}
    </button>
  );
}

/** A short status or explanation inside the page (not an error). */
export function Note({ children, tone = "neutral", icon }: { children: ReactNode; tone?: "neutral" | "warn"; icon?: LucideIcon }) {
  return (
    <div className={`flex items-start gap-3 rounded-2xl border px-5 py-4 text-[15px] leading-relaxed ${tone === "warn" ? "border-warn/30 bg-warn-soft" : "border-line bg-canvas"}`}>
      {icon && <Icon icon={icon} size={18} className={`mt-0.5 ${tone === "warn" ? "text-warn" : "text-muted"}`} />}
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-line" role="progressbar" aria-valuenow={Math.round(v)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
      <div className="h-full rounded-full bg-ink transition-[width] duration-500" style={{ width: `${v}%` }} />
    </div>
  );
}

export function ErrorNote({ error, onRetry, title = "Something went wrong" }: { error: { code: string; message: string } | null; onRetry?: () => void; title?: string }) {
  const box = useRef<HTMLDivElement>(null);
  const message = error?.message;
  useEffect(() => {
    // Errors appear next to what caused them, which may be below or above what is on screen.
    if (message) box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [message]);
  if (!error) return null;
  return (
    <div ref={box} className="rounded-2xl border border-bad/30 bg-bad-soft p-5 text-[15px]" role="alert">
      <div className="font-semibold text-bad">{title}</div>
      <div className="mt-1 whitespace-pre-wrap break-words leading-relaxed">{error.message}</div>
      {onRetry && (
        <button className="mt-3 text-[15px] font-medium text-bad underline underline-offset-4" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Drawer({ title, open, onToggle, children }: { title: string; open: boolean; onToggle: () => void; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-line">
      <button onClick={onToggle} className="flex w-full items-center justify-between rounded-2xl px-6 py-4 text-left text-[15px] font-medium text-muted transition-colors hover:text-ink" aria-expanded={open}>
        {title}
        <Icon icon={ChevronRight} className={`transition-transform ${open ? "rotate-90" : ""}`} />
      </button>
      {open && <div className="border-t border-line px-6 py-5 text-sm">{children}</div>}
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
    <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} className="text-ink" aria-hidden>
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinejoin="round" />
    </svg>
  );
}

export function LiftGauge({ before, after, label }: { before: number | null; after: number | null; label: string }) {
  const lift = after !== null && before !== null ? Math.round((after - before) * 100) : null;
  return (
    <div>
      <div className="text-[15px] text-muted">{label}</div>
      <div className={`mt-1 text-[40px] font-bold leading-none tracking-[-0.03em] tabular-nums ${lift === null ? "text-faint" : lift < 0 ? "text-bad" : ""}`}>
        {lift === null ? "–" : `${lift > 0 ? "+" : ""}${lift}`}
        <span className="ml-1.5 text-[17px] font-medium tracking-normal text-muted">points</span>
      </div>
      <div className="mt-6 space-y-3">
        {([["Original", before, "bg-faint"], ["Yours", after, "bg-ink"]] as const).map(([name, v, color]) => (
          <div key={name} className="flex items-center gap-3">
            <div className="w-16 text-sm text-muted">{name}</div>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
              <div className={`h-full rounded-full ${color} transition-[width] duration-700`} style={{ width: `${Math.round((v ?? 0) * 100)}%` }} />
            </div>
            <div className="w-10 text-right text-sm font-medium tabular-nums">{v === null ? "–" : `${Math.round(v * 100)}%`}</div>
          </div>
        ))}
      </div>
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
