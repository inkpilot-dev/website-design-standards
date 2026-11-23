/** Small DOM helpers for the client. */

export function esc(s: unknown): string {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

export function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el;
}

export async function post(path: string, body: unknown = {}): Promise<unknown> {
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return res.json();
}

export function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** A little blob in a faction's colours, used as the faction's icon throughout the UI. */
export function blobIcon(color: string, dark: string, light: string): string {
  return `<svg class="blob-icon" viewBox="0 0 40 34" aria-hidden="true"><path d="M3 26C3 13 10 3 20 3s17 10 17 23c0 4-4 6-17 6S3 30 3 26z" fill="${color}" stroke="${dark}" stroke-width="1.5"/><ellipse cx="13" cy="11" rx="4" ry="2.4" fill="${light}" opacity="0.55"/><ellipse cx="14.5" cy="18" rx="4" ry="4.6" fill="#fff"/><ellipse cx="25.5" cy="18" rx="4" ry="4.6" fill="#fff"/><circle cx="15.3" cy="18.8" r="2.2" fill="#1b1a17"/><circle cx="26.3" cy="18.8" r="2.2" fill="#1b1a17"/><path d="M17 25q3 2.5 6 0" stroke="${dark}" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>`;
}
