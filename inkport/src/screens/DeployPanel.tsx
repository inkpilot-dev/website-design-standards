import { useState } from "react";
import type { ApiInfo, AppError, ProjectInfo } from "../lib/contract";
import { api, inTauri, pickFiles, pickSavePath, toAppError } from "../lib/ipc";
import { Check, Clipboard, Eye, EyeOff } from "lucide-react";
import { Badge, Button, Card, ErrorNote, Icon, PageHeader, SectionTitle, TextInput } from "../components/ui";

type Tab = "curl" | "python" | "javascript";

export function DeployPanel({ project }: { project: ProjectInfo }) {
  const [info, setInfo] = useState<ApiInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [tab, setTab] = useState<Tab>("python");
  const [copied, setCopied] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [sending, setSending] = useState(false);
  const [transfer, setTransfer] = useState<string[]>([]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setInfo(await api.startLocalApi(project.id));
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    await api.stopLocalApi().catch((e) => setError(toAppError(e)));
    setInfo(null);
  };

  const copy = async (label: string, text: string) => {
    await navigator.clipboard.writeText(text).catch(() => {});
    setCopied(label);
    setTimeout(() => setCopied(""), 1500);
  };

  const send = async () => {
    if (!info || !message.trim()) return;
    const msgs = [...chat, { role: "user" as const, content: message.trim() }];
    setChat(msgs);
    setMessage("");
    setSending(true);
    try {
      let reply: string;
      if (inTauri) {
        const r = await fetch(`${info.base_url}/chat/completions`, { method: "POST", headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: info.model, messages: msgs, max_tokens: 256 }) });
        const j = await r.json();
        reply = r.ok ? j.choices[0].message.content : `Error: ${j.error?.message ?? r.status}`;
      } else {
        reply = '{"category": "fuel", "amount": 42}';
      }
      setChat([...msgs, { role: "assistant", content: reply }]);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setSending(false);
    }
  };

  const doExport = async () => {
    const dest = await pickSavePath(`${project.slug}-${project.active_version ?? "v1"}.inkpatch`);
    if (!dest) return;
    try {
      const r = await api.exportAdapter(project.id, dest);
      setTransfer([`Exported ${r.version} to ${r.path}`, `SHA-256 ${r.sha256}`, `Signed by publisher key ${r.publisher.slice(0, 16)}…`]);
    } catch (e) {
      setError(toAppError(e));
    }
  };

  const doImport = async (trust = false, file?: string) => {
    const f = file ?? (await pickFiles([".inkpatch"]))[0];
    if (!f) return;
    setTransfer(["Checking signature, SHA-256 hashes and every tensor…"]);
    try {
      const r = await api.importAdapter(project.id, f, trust);
      const lines = [`Signature verified (publisher ${r.import.publisher.slice(0, 16)}…)`, "Every file's SHA-256 matches the signed manifest", `${r.import.checks.tensors} tensors checked against your base model (rank ${r.import.checks.rank})`];
      if (r.gate) lines.push(r.gate.decision === "PROMOTE" ? `Beat your current version on your hidden exam, now active as ${r.gate.version}` : `Kept your current version: ${r.gate.explanation}`);
      setTransfer(lines);
    } catch (e) {
      const err = toAppError(e);
      if (err.code === "UNTRUSTED_PUBLISHER" && confirm("This adapter was signed by someone you haven't trusted before. Trust this publisher and continue? It will still have to pass your own hidden exam.")) {
        return doImport(true, f);
      }
      setTransfer([]);
      setError(err);
    }
  };

  const snippets: Record<Tab, string> = info
    ? {
        curl: `curl ${info.url} \\\n  -H "Authorization: Bearer ${showKey ? info.token : "YOUR_KEY"}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${info.model}", "messages": [{"role": "user", "content": "Paid $42 at Shell"}]}'`,
        python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${info.base_url}", api_key="${showKey ? info.token : "YOUR_KEY"}")\nreply = client.chat.completions.create(\n    model="${info.model}",\n    messages=[{"role": "user", "content": "Paid $42 at Shell"}],\n)\nprint(reply.choices[0].message.content)`,
        javascript: `const res = await fetch("${info.url}", {\n  method: "POST",\n  headers: { Authorization: "Bearer ${showKey ? info.token : "YOUR_KEY"}", "Content-Type": "application/json" },\n  body: JSON.stringify({ model: "${info.model}", messages: [{ role: "user", content: "Paid $42 at Shell" }] }),\n});\nconsole.log((await res.json()).choices[0].message.content);`,
      }
    : { curl: "", python: "", javascript: "" };

  const CopyButton = ({ id, text, label = "Copy" }: { id: string; text: string; label?: string }) => (
    <button className="inline-flex flex-none items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-[13px] font-medium transition-colors hover:border-ink" onClick={() => copy(id, text)}>
      <Icon icon={copied === id ? Check : Clipboard} size={14} /> {copied === id ? "Copied" : label}
    </button>
  );

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow={project.name}
        title="Use your AI anywhere on this computer"
        lead="Inkport runs a private, OpenAI-compatible endpoint. Any app that lets you set a “custom OpenAI URL” can use your model."
        action={info ? <Button variant="secondary" onClick={stop}>Stop local API</Button> : <Button onClick={start} busy={busy}>Start local API</Button>}
      />
      <ErrorNote error={error} />
      <div className="flex flex-wrap items-center gap-3 text-[15px] text-muted">
        {info ? <Badge tone="good"><span className="h-1.5 w-1.5 rounded-full bg-good" aria-hidden /> Running · version {info.version} · {info.engine}</Badge> : <Badge>Stopped</Badge>}
        Only apps on this computer can reach it.
      </div>

      {info && (
        <>
          <dl className="divide-y divide-line border-y border-line">
            {([["API address", info.base_url], ["Model name", info.model], ["API key", showKey ? info.token : "ik_" + "•".repeat(24)]] as const).map(([label, value]) => (
              <div key={label} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3.5">
                <dt className="w-32 flex-none text-[15px] text-muted">{label}</dt>
                <dd className="min-w-0 flex-1"><code className="block truncate font-mono text-sm">{value}</code></dd>
                <div className="flex flex-none gap-2">
                  {label === "API key" && (
                    <button className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-[13px] font-medium transition-colors hover:border-ink" onClick={() => setShowKey(!showKey)}>
                      <Icon icon={showKey ? EyeOff : Eye} size={14} /> {showKey ? "Hide" : "Show"}
                    </button>
                  )}
                  <CopyButton id={label} text={label === "API key" ? info.token : value} />
                </div>
              </div>
            ))}
          </dl>

          <section className="space-y-5 pt-4">
            <SectionTitle title="Call it from your code" lead="The endpoint speaks the OpenAI chat format, so existing client libraries work unchanged." />
            <div>
              <div className="flex gap-7 border-b border-line" role="tablist">
                {(["python", "javascript", "curl"] as Tab[]).map((t) => (
                  <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={`-mb-px border-b-2 pb-3 text-[15px] transition-colors ${tab === t ? "border-ink font-semibold text-ink" : "border-transparent text-muted hover:text-ink"}`}>{t === "javascript" ? "JavaScript" : t === "python" ? "Python" : "curl"}</button>
                ))}
              </div>
              <div className="relative mt-5">
                <pre className="overflow-auto rounded-2xl bg-canvas p-5 pr-28 font-mono text-[13px] leading-[1.7]">{snippets[tab]}</pre>
                <div className="absolute right-3 top-3"><CopyButton id="snippet" text={snippets[tab]} /></div>
              </div>
            </div>
          </section>

          <section className="space-y-5 pt-4">
            <SectionTitle title="Try it here" lead="Send a message to your trained model through the same endpoint your apps will use." />
            {!chatOpen ? (
              <Button variant="secondary" onClick={() => setChatOpen(true)}>Open test chat</Button>
            ) : (
              <Card className="space-y-4">
                <div className="max-h-72 space-y-2.5 overflow-auto">
                  {chat.length === 0 && <div className="text-[15px] text-muted">Send a message to your model.</div>}
                  {chat.map((m, i) => (
                    <div key={i} className={`w-fit max-w-[85%] whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-[15px] ${m.role === "user" ? "ml-auto bg-ink text-panel" : "bg-canvas font-mono text-[13px]"}`}>{m.content}</div>
                  ))}
                </div>
                <div className="flex gap-2">
                  <TextInput value={message} onChange={setMessage} onEnter={send} placeholder="Type a message" label="Message" />
                  <Button onClick={send} busy={sending}>Send</Button>
                  <Button variant="ghost" onClick={() => setChatOpen(false)}>Close</Button>
                </div>
              </Card>
            )}
          </section>
        </>
      )}

      <section className="space-y-5 pt-4">
        <SectionTitle title="Share or back up your trained skill" lead="Adapters are small files (megabytes, not gigabytes). Each one is signed and fingerprinted; imports are verified and must beat your current version on your own hidden exam before they are switched on." />
        <div className="flex flex-wrap gap-3">
          <Button variant="secondary" onClick={doExport} disabled={!project.active_version && !info}>Export adapter…</Button>
          <Button variant="secondary" onClick={() => doImport()}>Import adapter…</Button>
        </div>
        {transfer.length > 0 && (
          <ul className="space-y-1.5 rounded-2xl bg-canvas p-5 font-mono text-[13px] leading-relaxed">
            {transfer.map((l) => (
              <li key={l} className="break-all">{l}</li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
