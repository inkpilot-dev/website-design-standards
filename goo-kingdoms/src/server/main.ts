/**
 * Goo Kingdoms server: serves the spectator app, streams the match to the
 * browser with server-sent events, and keeps the API key on this machine.
 *
 *   npm install
 *   npm start            -> http://localhost:3000
 *
 * Put ANTHROPIC_API_KEY in a .env file (or paste it in Settings) for Claude
 * mode. Without one, the local bot plays.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MatchController } from './match.ts';
import type { Settings, Speed } from './match.ts';
import { loadResults } from './results.ts';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..', '..');
const publicDir = path.join(root, 'public');

try {
  process.loadEnvFile(path.join(root, '.env'));
} catch {
  // No .env file: fine, the key can come from the environment or Settings.
}

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? '127.0.0.1';

if (!fs.existsSync(path.join(publicDir, 'app.js'))) {
  console.error('public/app.js is missing. Run "npm start" (which builds it) or "npm run build" first.');
  process.exit(1);
}

const match = new MatchController();

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.map': 'application/json',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
};

function sendJson(res: http.ServerResponse, code: number, body: unknown): void {
  res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
}

async function readBody(req: http.IncomingMessage): Promise<Record<string, unknown>> {
  let raw = '';
  for await (const chunk of req) {
    raw += chunk;
    if (raw.length > 100_000) throw new Error('Body too large');
  }
  if (!raw) return {};
  return JSON.parse(raw) as Record<string, unknown>;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  try {
    // Live stream of the match.
    if (url.pathname === '/api/events' && req.method === 'GET') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-store',
        Connection: 'keep-alive',
      });
      const send = (event: string, data: unknown) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      send('init', match.initPayload());
      const off = match.on(send);
      const ping = setInterval(() => res.write(': ping\n\n'), 20_000);
      req.on('close', () => {
        off();
        clearInterval(ping);
      });
      return;
    }

    if (req.method === 'POST' && url.pathname.startsWith('/api/')) {
      const body = await readBody(req);
      switch (url.pathname) {
        case '/api/new': {
          if (body.settings) match.updateSettings(body.settings as Partial<Settings>);
          const seed = Number(body.seed);
          match.newMatch(Number.isFinite(seed) && String(body.seed ?? '') !== '' ? seed : Math.floor(Math.random() * 100000), Number(body.maxDays) || undefined);
          return sendJson(res, 200, match.publicStatus());
        }
        case '/api/play': match.play(); return sendJson(res, 200, match.publicStatus());
        case '/api/pause': match.pause(); return sendJson(res, 200, match.publicStatus());
        case '/api/step': await match.step(); return sendJson(res, 200, match.publicStatus());
        case '/api/speed': match.setSpeed(String(body.speed) as Speed); return sendJson(res, 200, match.publicStatus());
        case '/api/settings': {
          if ('apiKey' in body) match.setKey(typeof body.apiKey === 'string' ? body.apiKey : null);
          match.updateSettings(body as Partial<Settings>);
          return sendJson(res, 200, match.publicStatus());
        }
        default: return sendJson(res, 404, { error: 'Unknown endpoint' });
      }
    }

    if (req.method === 'GET') {
      if (url.pathname === '/api/frame') {
        const day = Number(url.searchParams.get('day'));
        const frame = match.frames[day];
        return frame ? sendJson(res, 200, frame) : sendJson(res, 404, { error: 'No such day' });
      }
      if (url.pathname === '/api/replay') {
        res.writeHead(200, {
          'Content-Type': 'application/json',
          'Content-Disposition': `attachment; filename="goo-kingdoms-${match.matchId}.json"`,
        });
        return res.end(JSON.stringify(match.replay()));
      }
      if (url.pathname === '/api/results') return sendJson(res, 200, loadResults());
      if (url.pathname === '/api/status') return sendJson(res, 200, match.publicStatus());

      // Static files.
      const rel = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname).replace(/^\/+/, '');
      const file = path.normalize(path.join(publicDir, rel));
      if (!file.startsWith(publicDir)) return sendJson(res, 403, { error: 'Forbidden' });
      if (fs.existsSync(file) && fs.statSync(file).isFile()) {
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' });
        return fs.createReadStream(file).pipe(res);
      }
    }
    sendJson(res, 404, { error: 'Not found' });
  } catch (err) {
    sendJson(res, 400, { error: err instanceof Error ? err.message : String(err) });
  }
});

server.listen(PORT, HOST, () => {
  const s = match.publicStatus();
  console.log(`\n  Goo Kingdoms is running at http://${HOST === '127.0.0.1' ? 'localhost' : HOST}:${PORT}\n`);
  console.log(s.hasKey
    ? `  Brain: Claude (${s.settings.model}). Press Play in the app to start; each day costs real API calls.`
    : '  No ANTHROPIC_API_KEY found: the local bot plays. Add a key to .env or in Settings for Claude mode.');
  console.log('');
});
