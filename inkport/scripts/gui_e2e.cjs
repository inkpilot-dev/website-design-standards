#!/usr/bin/env node
// Drives the real Inkport window through all five steps with the real engine behind it, the way a
// person would: click, type, wait, read what is on screen. One script, three ways to reach a window:
//
//   --mode=webdriver --driver=http://127.0.0.1:4444 --app=/path/to/inkport      Linux (WebKitWebDriver)
//   --mode=cdp --cdp=http://127.0.0.1:9222                                      Windows (WebView2 started with
//                                                                               WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222)
//   --mode=preview --url=http://127.0.0.1:1420/?speed=3                         a browser tab with the simulated engine
//
//   --out=DIR        screenshots (01-ready.png, ...)
//   --phase=first    create a project, build the exam, train, use it   (default)
//   --phase=smoke    the same up to "training is running", then Stop (minutes instead of a full run)
//   --phase=reopen   after restarting the app: the project is still there and opens on "Use it"
//   --phase=goals    every goal that ships sample data: "Try the sample data" + every suggested rule
//                    must give an exam that can be trained on (no training; about a minute)
//   --train-timeout=SECONDS   how long training may take (default 900)
//   --expect=promote|any      "promote": the run must save a version (default); "any": a verdict is enough
//   --goal=business_rules|code_sql|docs_legal|custom   which goal the first/smoke phases use (default: business_rules)
//
// Set INKPORT_TEST_PRESET / INKPORT_TEST_OVERRIDES in the app's environment to keep training short.
// Exit code 0 = every check passed.
'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');

const arg = (name, dflt) => { const a = process.argv.find((x) => x.startsWith(`--${name}=`)); return a ? a.slice(name.length + 3) : dflt; };
const MODE = arg('mode', 'preview');
const OUT = arg('out', 'gui-e2e');
const PHASE = arg('phase', 'first');
const EXPECT = arg('expect', 'promote');
const TRAIN_TIMEOUT = Number(arg('train-timeout', '900')) * 1000;
// The start-and-stop pass has its own project, so that it can follow a full pass in the same data folder.
const PROJECT = arg('project', PHASE === 'smoke' ? 'Start and stop' : 'Window test');
// Which goal the 'first' and 'smoke' phases use: its title on screen, a suggested rule to add, and
// a file name that shows up under the sample questions.
const GOALS = {
  business_rules: ['Small Business & Finance Rules', 'amount must be positive', 'from expenses.csv'],
  code_sql: ['Code & SQL Specialist', 'no more than 120 words', 'from your tables'],
  docs_legal: ['Personal Docs & Legal Prep', 'no more than 80 words', 'from lease.md'],
  custom: ['Custom Skill', 'no more than 50 words', 'from support.csv'],
};
const GOAL = GOALS[arg('goal', 'business_rules')];
if (!GOAL) { console.log('unknown --goal'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const t0 = Date.now();
const stamp = () => `[${String(Math.round((Date.now() - t0) / 1000)).padStart(4)}s]`;
const check = (name, ok, detail = '') => { results.push([!!ok, name]); console.log(`${stamp()} ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); return !!ok; };
const note = (msg) => console.log(`${stamp()}       ${msg}`);

// ---- in-page helpers: everything the test does to the page goes through these ----
const HELPERS = `
  const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
  const visible = (el) => !!(el && el.getClientRects().length);
  const buttons = (scope) => [...document.querySelectorAll((scope || '') + ' button')].filter(visible);
  const findButton = (text, scope) => buttons(scope).find((b) => norm(b.textContent) === text) || buttons(scope).find((b) => norm(b.textContent).includes(text));
  const api = {
    text: () => norm(document.querySelector('main') ? document.querySelector('main').innerText : document.body.innerText),
    aside: () => norm(document.querySelector('aside') ? document.querySelector('aside').innerText : ''),
    has: (text, scope) => !!findButton(text, scope),
    enabled: (text, scope) => { const b = findButton(text, scope); return !!b && !b.disabled; },
    click: (text, scope) => { const b = findButton(text, scope); if (!b) return 'missing'; if (b.disabled) return 'disabled'; b.scrollIntoView({ block: 'center' }); b.click(); return 'ok'; },
    clickSoon: (text, scope) => { const b = findButton(text, scope); if (!b) return 'missing'; if (b.disabled) return 'disabled'; b.scrollIntoView({ block: 'center' }); setTimeout(() => b.click(), 200); return 'ok'; },
    fill: (label, value) => {
      const el = [...document.querySelectorAll('main input, main textarea')].filter(visible).find((i) => i.getAttribute('aria-label') === label || i.placeholder === label) || (label === '' ? [...document.querySelectorAll('main input')].filter(visible)[0] : null);
      if (!el) return 'missing';
      const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, value);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      return 'ok';
    },
    enter: (label) => {
      const el = [...document.querySelectorAll('main input')].filter(visible).find((i) => i.getAttribute('aria-label') === label || i.placeholder === label);
      if (!el) return 'missing';
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      return 'ok';
    },
    scrollTop: () => { const m = document.querySelector('main'); if (m) m.scrollTop = 0; return true; },
    scrollToHeading: (text) => { const h = [...document.querySelectorAll('main h1, main h2, main h3')].find((e) => norm(e.textContent).includes(text)); if (h) h.scrollIntoView({ block: 'start' }); return !!h; },
    fonts: () => document.fonts ? [...document.fonts].filter((f) => f.status === 'loaded').map((f) => f.family).filter((v, i, a) => a.indexOf(v) === i).join(',') : '',
    ua: () => navigator.userAgent,
  };
`;
const call = (expr) => `(function(){ ${HELPERS} return (${expr}); })()`;

// ---- drivers ----
function request(method, url, body, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const data = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const req = http.request({ method, hostname: u.hostname, port: u.port, path: u.pathname + u.search, headers: data ? { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': data.length } : {} }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => { const text = Buffer.concat(chunks).toString('utf8'); try { resolve(JSON.parse(text)); } catch { reject(new Error(`driver said: ${text.slice(0, 300)}`)); } });
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error('driver request timed out')));
    if (data) req.write(data);
    req.end();
  });
}

async function openWebDriver() {
  const base = arg('driver', 'http://127.0.0.1:4444');
  const app = arg('app');
  if (!app) throw new Error('--app=/path/to/inkport is required in webdriver mode');
  const caps = { capabilities: { alwaysMatch: { 'webkitgtk:browserOptions': { binary: app, args: [] } } } };
  const created = await request('POST', `${base}/session`, caps);
  const id = created.value && created.value.sessionId;
  if (!id) throw new Error('could not start a session: ' + JSON.stringify(created).slice(0, 400));
  const s = `${base}/session/${id}`;
  return {
    evaluate: async (expr, timeoutMs) => { const r = await request('POST', `${s}/execute/sync`, { script: `return ${call(expr)};`, args: [] }, timeoutMs); if (r.value && r.value.error) throw new Error(r.value.message || r.value.error); return r.value; },
    // A picture of the window. With GUI_E2E_SCREENSHOT (a command that writes a picture of the
    // screen to the file it is given; scripts/window_check.sh sets it) the screen itself is
    // captured. The driver's own screenshot is the fallback, with a time limit: the WebKitGTK of
    // Ubuntu 22.04 only answers it when the page draws a new frame, so on a page where nothing
    // moves (right after "Stop training") it never answered, and no later request did either.
    screenshot: async (file) => {
      if (process.env.GUI_E2E_SCREENSHOT) {
        try { require('child_process').execFileSync(process.env.GUI_E2E_SCREENSHOT, [file], { timeout: 30000, stdio: 'ignore' }); } catch (e) { note(`screenshot skipped: ${String(e.message).split('\n')[0]}`); }
        return;
      }
      const r = await request('GET', `${s}/screenshot`, undefined, 30000);
      if (typeof r.value === 'string') fs.writeFileSync(file, Buffer.from(r.value, 'base64'));
    },
    close: async () => { await request('DELETE', s).catch(() => {}); },
  };
}

async function openPlaywright() {
  const { chromium } = require(MODE === 'cdp' ? arg('playwright', 'playwright-core') : 'playwright');
  let browser, page;
  if (MODE === 'cdp') {
    const endpoint = arg('cdp', 'http://127.0.0.1:9222');
    for (let i = 0; ; i++) {
      try { browser = await chromium.connectOverCDP(endpoint); break; } catch (e) { if (i > 60) throw new Error(`no window to attach to at ${endpoint}: ${e.message.split('\n')[0]}`); await sleep(1000); }
    }
    for (let i = 0; i < 60 && !page; i++) { page = browser.contexts().flatMap((c) => c.pages()).find((p) => !p.url().startsWith('devtools:')); if (!page) await sleep(500); }
    if (!page) throw new Error('the app has no page to drive');
    note(`attached to ${page.url()}`);
  } else {
    browser = await chromium.launch();
    page = await (await browser.newContext({ viewport: { width: 1240, height: 860 } })).newPage();
    await page.goto(arg('url', 'http://127.0.0.1:1420/?speed=3'));
  }
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  return {
    evaluate: (expr) => page.evaluate(call(expr)),
    screenshot: (file) => page.screenshot({ path: file }).catch((e) => note(`screenshot failed: ${e.message.split('\n')[0]}`)),
    close: async () => { if (MODE === 'cdp') await browser.close().catch(() => {}); else await browser.close(); },
    errors,
  };
}

(async () => {
  const d = MODE === 'webdriver' ? await openWebDriver() : await openPlaywright();
  const q = JSON.stringify;
  const text = () => d.evaluate('api.text()');
  // Screenshots are for people; one that cannot be taken must never fail or hold up a check.
  const shot = async (name) => {
    try { await d.evaluate('api.scrollTop()', 15000); await sleep(250); await d.screenshot(path.join(OUT, name + '.png')); } catch (e) { note(`no screenshot "${name}": ${String(e.message).split('\n')[0]}`); }
  };
  const waitFor = async (what, fn, timeout = 30000, every = 500) => {
    const end = Date.now() + timeout;
    let last;
    for (;;) {
      try { last = await fn(); if (last) return last; } catch (e) { last = `error: ${e.message}`; }
      if (Date.now() > end) throw new Error(`timed out waiting for: ${what}`);
      await sleep(every);
    }
  };
  const click = async (label, scope) => {
    let r;
    try { r = await d.evaluate(`api.click(${q(label)}${scope ? ', ' + q(scope) : ''})`); } catch (e) { throw new Error(`clicking "${label}": ${e.message}`); }
    if (r !== 'ok') throw new Error(`button "${label}" is ${r}`);
    await sleep(300);
  };
  const nav = (label) => click(label, 'aside nav');
  // Press "Stop training" and wait for the window to say the run was stopped.
  //
  // On one kind of machine (WebKitGTK from Ubuntu 22.04 inside the AppImage, driven by
  // WebKitWebDriver) the driver never answers again after this click, although the app is fine:
  // its main loop is idle, the engine has stopped, and the page has moved on. So the click is
  // scheduled rather than awaited, each read of the page has a short time limit, and if the
  // driver stays silent the screen itself is read (GUI_E2E_SCREEN_TEXT: a command that prints
  // the text visible on the screen, set by scripts/window_check.sh).
  const stopFromWindow = async (limitMs = 180000) => {
    const r = await d.evaluate('api.clickSoon("Stop training", "main")', 20000);
    if (r !== 'ok') throw new Error(`button "Stop training" is ${r}`);
    const end = Date.now() + limitMs;
    let silent = 0;
    let lastNote = Date.now();
    for (;;) {
      await sleep(1500);
      try {
        if (process.env.GUI_E2E_PRETEND_DRIVER_LOST) throw new Error('pretending the driver is silent (to test reading the screen)');
        const t = await d.evaluate('api.text()', 10000);
        silent = 0;
        if (t.includes('Training was stopped')) return { via: 'window', text: t };
        if (Date.now() - lastNote > 15000) { lastNote = Date.now(); note('not stopped yet; the window shows: ' + t.slice(0, 230)); }
      } catch (e) {
        silent += 1;
        note(`no answer from the window (${silent}): ${e.message}`);
      }
      if (silent >= 3 && process.env.GUI_E2E_SCREEN_TEXT) {
        let seen = '';
        for (let i = 0; i < 20 && !/Training\s+was\s+stopped/i.test(seen); i++) {
          if (i) await sleep(3000);
          try { seen = require('child_process').execSync(process.env.GUI_E2E_SCREEN_TEXT, { encoding: 'utf8', timeout: 120000, stdio: ['ignore', 'pipe', 'ignore'] }).replace(/\s+/g, ' ').trim(); } catch { seen = ''; }
        }
        return { via: 'screen', text: seen };
      }
      if (Date.now() > end) throw new Error('timed out waiting for: the run to stop');
    }
  };
  // The verdict on a stopped run, from the page or (see above) from the screen. Returns false when
  // the driver has lost the window, so the caller knows not to use it any further.
  const checkStopped = async (stopped, screenshot) => {
    if (stopped.via === 'window') {
      check('Stop ends the run with a plain note and the start cards return', stopped.text.includes('Nothing was saved from that run') && (await d.evaluate('api.enabled("Quick run", "main")', 15000)) && !stopped.text.includes('Finished'));
      await shot(screenshot);
      return true;
    }
    note('the test driver stopped answering after the click, so the screen was read instead:');
    note('  ' + (stopped.text || '(nothing could be read from the screen)').slice(0, 420));
    check('Stop ends the run with a plain note and the start cards return (read from the screen)', /Training\s+was\s+stopped/i.test(stopped.text) && /Nothing\s+was\s+saved/i.test(stopped.text) && /Quick\s+run/i.test(stopped.text) && !/Stop\s+training/i.test(stopped.text), stopped.text ? '' : 'no text could be read from the screen');
    return false;
  };
  const waitText = (needle, timeout) => waitFor(`text "${needle}"`, async () => (await text()).includes(needle), timeout);
  // Has the test chat answered `question`? Any answer counts, well-formed or not: whether the
  // app delivers an answer is what is tested, not how good the model is (the tiny test model
  // answers an unknown shop with half a JSON object).
  const chatAnswered = (t, question) => {
    const area = (t.split('Try it here')[1] || '').split('Type a message')[0];
    if (/\{[^}]*\}/.test(area) || t.includes('The model did not answer')) return true;
    return area.includes(question) && !area.includes('thinking') && area.replace(question, '').replace(/Send a message.*?use\./, '').trim().length > 3;
  };

  try {
    // ---------------------------------------------------------------- step 1: your computer
    await waitFor('the start-up check to finish', () => d.evaluate('api.has("Continue", "main") || api.text().includes("Needs repair") || api.text().includes("could not finish") || api.text().includes("Developer build")'), 600000, 1000);
    const ready = await text();
    if (ready.includes('Developer build')) throw new Error('this is a development build without the bundled runtime; point INKPORT_BUNDLE_DIR at an assembled bundle (build/payload-<platform>)');
    await shot('01-ready');
    note(`browser engine: ${await d.evaluate('api.ua()')}`);
    note(`fonts loaded: ${await d.evaluate('api.fonts()')}`);
    if (!check('start-up check passes and the window shows the ready screen', ready.includes('Inkport is ready') && ready.includes('files verified'), ready.slice(0, 200))) throw new Error('the app did not become ready');
    check('the ready screen names the included model and where things run', ready.includes('Included model') && ready.includes('Training runs on') && ready.includes('Answers run on'));
    check('the bundled typeface is loaded', (await d.evaluate('api.fonts()')).includes('Inter'));
    await click('Continue', 'main');

    if (PHASE === 'reopen') {
      // ------------------------------------------------------------ after a restart
      await waitText('What do you want your AI to master?');
      const list = await text();
      await shot('20-reopened-list');
      check('after a restart the project is listed as trained', list.includes('Continue a project') && list.includes(PROJECT) && /trained, version v\d+/.test(list), list.slice(0, 260));
      await click('Open', 'main');
      await waitText('Use your AI anywhere on this computer');
      const use = await text();
      check('opening it lands on "Use it" with the API stopped and ready to start', use.includes('Stopped') && (await d.evaluate('api.enabled("Start local API", "main")')));
      await nav('Train & prove');
      await sleep(500);
      const tr = await text();
      await shot('21-reopened-train');
      check('"Train & prove" shows the version in use and offers a newer run', /Version v\d+ is in use/.test(tr) && tr.includes('Train a newer version') && tr.includes('Ask both models yourself'), tr.slice(0, 260));
      await nav('Data & rules');
      await waitFor('the saved exam to be restored', async () => (await text()).includes('Your practice exam'), 60000);
      const data = await text();
      await shot('22-reopened-data');
      check('"Data & rules" restores the files, the rule and the exam', data.includes('expenses.csv') && data.includes('amount must be positive') && (await d.evaluate('api.enabled("Continue to training", "main")')), data.slice(0, 200));
      await nav('Use it');
      await click('Start local API', 'main');
      await waitFor('the local API to start', async () => (await text()).includes('Running · version'), 300000, 1000);
      await click('Open test chat', 'main');
      await d.evaluate(`api.fill("Message", "Paid $19 at Costco for printer ink")`);
      await d.evaluate('api.enter("Message")');
      await waitFor('an answer in the test chat', async () => chatAnswered(await text(), 'Paid $19 at Costco for printer ink'), 300000, 1000);
      const chat = await text();
      await shot('23-reopened-chat');
      check('the trained model answers again after the restart', !chat.includes('The model did not answer'), chat.slice(chat.indexOf('Try it here'), chat.indexOf('Try it here') + 200));
      await click('Stop local API', 'main');
      await waitFor('the local API to stop', async () => (await text()).includes('Stopped'), 60000);
    } else if (PHASE === 'goals') {
      // ------------------------------------------------------------ every goal's first steps
      // What a new user does first: pick a goal, press "Try the sample data", add the suggested
      // rules, build. Every goal that ships sample data has to end with an exam that can be trained on.
      const templates = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'contracts', 'templates.json'), 'utf8')).templates;
      let n = 0;
      for (const id of ['business_rules', 'code_sql', 'docs_legal', 'custom']) {
        const t = templates.find((x) => x.id === id);
        n += 1;
        if (n > 1) await nav('Choose a goal');
        await waitText('What do you want your AI to master?');
        await click(t.title, 'main');
        await d.evaluate(`api.fill("", ${q(`Goal check ${n}`)})`);
        await click('Create project', 'main');
        await waitText('Your material and your rules');
        await click('Try the sample data', 'main');
        for (const rule of t.example_rules) await click(rule, 'main');
        if (id === 'business_rules') {
          // Reads fine, but the sample has no such column: the build must say so next to the rule.
          await d.evaluate('api.fill("New rule", "price must be positive")');
          await click('Add', 'main');
        }
        await waitFor('the rules to be checked', () => d.evaluate('api.enabled("Build practice exam", "main")'), 120000);
        await click('Build practice exam', 'main');
        await waitFor('the practice exam', async () => { const x = await text(); return x.includes('Your practice exam') || x.includes('could not be built'); }, 300000, 1000);
        await sleep(1500);
        const page = await text();
        await shot(`3${n}-goal-${id}`);
        await d.evaluate('api.scrollToHeading("Your practice exam")');
        await sleep(300);
        await d.screenshot(path.join(OUT, `3${n}-goal-${id}-exam.png`)).catch((e) => note(`no screenshot of the ${id} exam: ${String(e.message).split('\n')[0]}`));
        const from = page.indexOf('Your practice exam');
        check(`${t.title}: sample data + suggested rules give an exam that can be trained on`, page.includes('hidden final-exam questions') && !page.includes('Not enough examples') && !page.includes('could not be built') && (await d.evaluate('api.enabled("Continue to training", "main")')), page.slice(from, from + 200));
        const setAside = (page.match(/This rule (?:is not being|will not be) used/g) || []).length;
        if (id === 'business_rules') check(`${t.title}: only the rule about a missing column is set aside, and the reason is shown`, setAside === 1 && page.includes('There is no column called “price”'), `${setAside} rule(s) set aside`);
        else check(`${t.title}: every suggested rule is in use`, setAside === 0, `${setAside} rule(s) set aside`);
      }
      await nav('Choose a goal');
      await waitText('Continue a project');
      const all = await text();
      check('all four projects are listed with their exams built', ['Goal check 1', 'Goal check 2', 'Goal check 3', 'Goal check 4'].every((x) => all.includes(x)) && (all.match(/practice exam built/g) || []).length >= 4, all.slice(0, 200));
    } else {
      // ------------------------------------------------------------ step 2: choose a goal
      await waitText('What do you want your AI to master?');
      await shot('02-goal');
      await click(GOAL[0], 'main');
      await d.evaluate(`api.fill("", ${q(PROJECT)})`);
      await click('Create project', 'main');
      await waitText('Your material and your rules');
      check('a project can be created', true);
      check('a new project has training and "Use it" locked', !(await d.evaluate('api.enabled("Train & prove", "aside nav")')) && !(await d.evaluate('api.enabled("Use it", "aside nav")')));

      // ------------------------------------------------------------ step 3: data & rules
      await click('Try the sample data', 'main');
      await click(GOAL[1], 'main');
      await waitFor('the rule to be checked', () => d.evaluate('api.enabled("Build practice exam", "main")'), 120000);
      await click('Build practice exam', 'main');
      await waitFor('the practice exam', async () => { const t = await text(); return t.includes('Your practice exam') || t.includes('could not be built'); }, 300000, 1000);
      const exam = await text();
      await shot('03-exam');
      check('the practice exam is built from the bundled samples', exam.includes('hidden final-exam questions') && exam.includes('No final-exam question appears in training') && !exam.includes('could not be built'), exam.slice(exam.indexOf('Your practice exam'), exam.indexOf('Your practice exam') + 220));
      await waitFor('sample questions', async () => (await text()).includes(GOAL[2]), 120000);
      check('sample practice questions are shown', true);
      await click('Continue to training', 'main');

      // ------------------------------------------------------------ step 4: train & prove
      await waitText('What happens');
      await shot('04-train-idle');
      await click('Quick run', 'main');
      await waitFor('training to start', () => d.evaluate('api.has("Stop training", "main")'), 120000);
      check('training starts and shows a Stop button', true);
      await waitFor('first progress', async () => /\d+%/.test(await text()), 300000, 1000);
      // Leave the step and come back while it runs.
      await nav('Data & rules');
      await sleep(800);
      const locked = await text();
      check('while training, "Data & rules" explains that the exam cannot be rebuilt', locked.includes('Training is running for this project'));
      check('the sidebar shows training progress from another step', /Training \d+%/.test(await d.evaluate('api.aside()')), await d.evaluate('api.aside()'));
      await nav('Train & prove');
      await sleep(800);
      await shot('05-training');
      const still = (await d.evaluate('api.has("Stop training", "main")')) || /saved|Not better yet|Need more examples/.test(await text());
      check('coming back to "Train & prove" still shows the run', still);
      if (PHASE === 'smoke') {
        if (await checkStopped(await stopFromWindow(), '06-stopped')) {
          await nav('Choose a goal');
          await waitText('Continue a project');
          const projects = await text();
          check('the project appears in the list with its state', projects.includes(PROJECT) && projects.includes('practice exam built'));
          if (d.errors) check('no errors in the page console', d.errors.length === 0, d.errors.slice(0, 3).join(' | '));
        }
        await Promise.race([d.close(), sleep(20000)]);
        const bad = results.filter((r) => !r[0]);
        console.log(`\n${results.length - bad.length}/${results.length} window checks passed (${PHASE}, ${MODE})`);
        process.exit(bad.length ? 1 : 0);
      }
      note('waiting for the final exam…');
      await waitFor('a verdict', async () => { const t = await text(); return /new version v\d+ saved|Not better yet|Need more examples|Training did not finish|ran out of memory|Training was stopped/.test(t) && !(await d.evaluate('api.has("Stop training", "main")')); }, TRAIN_TIMEOUT, 2000);
      const verdict = await text();
      await shot('06-verdict');
      const promoted = /new version v\d+ saved/.test(verdict);
      note(verdict.slice(Math.max(0, verdict.search(/Better:|Not better yet|Need more examples|Training did not finish|ran out of memory/)), Math.max(0, verdict.search(/Better:|Not better yet|Need more examples|Training did not finish|ran out of memory/)) + 330));
      check('training ends with a verdict (not an error)', /new version v\d+ saved|Not better yet|Need more examples/.test(verdict));
      if (EXPECT === 'promote') check('the trained model beat the original and a version was saved', promoted);
      if (promoted) {
        check('all six steps of the run are ticked and it reads "Finished"', verdict.includes('Finished') && verdict.includes('100%'));
        await waitFor('"Use it" to unlock', () => d.evaluate('api.enabled("Use it", "aside nav")'), 30000);
        // Ask both models (a separate short job).
        await d.evaluate(`api.fill("Question for both models", "Paid $64 at Shell for the delivery van")`);
        await click('Compare', 'main');
        await waitFor('the comparison', async () => { const t = await text(); return (t.split('Ask both models yourself')[1] || '').includes('Original') || t.includes('could not run'); }, 600000, 1000);
        const cmp = await text();
        await shot('07-compare');
        check('"Ask both models yourself" answers with both models', !cmp.includes('could not run') && (cmp.split('Ask both models yourself')[1] || '').includes('Yours'));

        // ---------------------------------------------------------- step 5: use it
        await click('Use it in your apps', 'main');
        await waitText('Use your AI anywhere on this computer');
        await click('Start local API', 'main');
        await waitFor('the local API to start', async () => { const t = await text(); return t.includes('Running · version') || t.includes('could not be started'); }, 300000, 1000);
        const api1 = await text();
        await shot('08-api');
        check('the local API starts and shows its address, model name and a hidden key', api1.includes('Running · version v') && /http:\/\/127\.0\.0\.1:\d+\/v1/.test(api1) && api1.includes('inkport/') && api1.includes('ik_•'), api1.slice(0, 260));
        note(api1.slice(api1.indexOf('Answering with'), api1.indexOf('Answering with') + 120));
        await click('Open test chat', 'main');
        await d.evaluate(`api.fill("Message", "Paid $58 at Shell for the delivery van")`);
        await d.evaluate('api.enter("Message")');
        await waitFor('an answer in the test chat', async () => chatAnswered(await text(), 'Paid $58 at Shell for the delivery van'), 300000, 1000);
        const chat = await text();
        await shot('09-chat');
        check('the test chat gets an answer from the trained model', !chat.includes('The model did not answer'), (chat.split('Try it here')[1] || '').slice(0, 220));
        // Leave and return: the API must still be shown as running.
        await nav('Train & prove');
        await sleep(600);
        await nav('Use it');
        await sleep(600);
        check('after leaving and returning, "Use it" still shows the running API', (await text()).includes('Running · version v') && (await d.evaluate('api.has("Stop local API", "main")')));
        await click('Stop local API', 'main');
        await waitFor('the local API to stop', async () => (await text()).includes('Stopped'), 60000);
        check('the local API stops', true);
      }
      // ---------------------------------------------------------- stop a second run
      await nav('Train & prove');
      await sleep(500);
      await click('Best results', 'main');
      await waitFor('the second run to start', () => d.evaluate('api.has("Stop training", "main")'), 120000);
      await sleep(4000);
      if (!(await checkStopped(await stopFromWindow(), '10-stopped'))) {
        await Promise.race([d.close(), sleep(20000)]);
        const bad = results.filter((r) => !r[0]);
        console.log(`\n${results.length - bad.length}/${results.length} window checks passed (${PHASE}, ${MODE})`);
        process.exit(bad.length ? 1 : 0);
      }
      // ---------------------------------------------------------- the project list
      await nav('Choose a goal');
      await waitText('Continue a project');
      const list = await text();
      await shot('11-projects');
      check('the project appears in the list with its state', list.includes(PROJECT) && (promoted ? /trained, version v\d+/.test(list) : list.includes('practice exam built')));
    }
    if (d.errors) check('no errors in the page console', d.errors.length === 0, d.errors.slice(0, 3).join(' | '));
  } catch (e) {
    check(`run completed`, false, e.message);
    // What the machine and the app were doing at that moment (set by the window-check scripts).
    if (process.env.GUI_E2E_ON_FAILURE) {
      try { console.log(require('child_process').execSync(process.env.GUI_E2E_ON_FAILURE, { encoding: 'utf8', timeout: 180000, stdio: ['ignore', 'pipe', 'pipe'] }).split('\n').map((l) => '      ' + l).join('\n')); } catch (x) { note('diagnostics failed: ' + String(x.message).split('\n')[0]); }
    }
    const soon = (promise) => Promise.race([promise, sleep(20000).then(() => { throw new Error('no answer from the window'); })]);
    try { await soon(d.screenshot(path.join(OUT, '99-failure.png'))); note('on screen: ' + (await soon(d.evaluate('api.text()'))).slice(0, 700)); } catch (x) { note('the window did not answer afterwards: ' + x.message); }
  }
  await d.close();
  const failed = results.filter((r) => !r[0]);
  console.log(`\n${results.length - failed.length}/${results.length} window checks passed (${PHASE}, ${MODE})`);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.log('could not run the window test: ' + e.message); process.exit(2); });
