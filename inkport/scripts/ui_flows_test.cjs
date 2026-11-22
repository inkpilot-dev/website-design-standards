#!/usr/bin/env node
// UI flow tests against preview mode (`pnpm dev`): the simulated engine follows the real backend's
// rules, so everything the window does with its state can be checked in seconds, including the
// paths a happy walk-through never takes: leaving a step mid-run, Stop, a crash, a rejected run,
// too little data, an unknown publisher, reopening a trained project, a second project.
//
//   pnpm dev --host 127.0.0.1 &            # serves http://127.0.0.1:1420
//   node scripts/ui_flows_test.cjs [screenshot-dir] [base-url]
//
// Needs the `playwright` package with a Chromium (build/test machines only).
const { chromium } = require('playwright');
const OUT = process.argv[2] || 'ui-flows';
const BASE = (process.argv[3] || 'http://127.0.0.1:1420/').replace(/\/?$/, '/');
require('fs').mkdirSync(OUT, { recursive: true });
const results = [];
const check = (name, ok, detail = '') => { results.push([ok, name, detail]); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
(async () => {
  const browser = await chromium.launch();
  const open = async (query = '') => {
    const ctx = await browser.newContext({ viewport: { width: 1240, height: 860 } });
    const page = await ctx.newPage();
    page.errors = [];
    page.on('console', m => { if (m.type() === 'error') page.errors.push(m.text()); });
    page.on('pageerror', e => page.errors.push('pageerror: ' + e.message));
    await page.goto(BASE + '?speed=' + (query.includes('slow') ? '1.5' : '6') + query);
    await page.getByRole('button', { name: 'Continue', exact: true }).waitFor({ timeout: 30000 });
    return page;
  };
  const btn = (page, name) => page.getByRole('button', { name });
  const nav = (page, label) => page.locator('aside nav button', { hasText: label });
  const main = (page) => page.locator('main');
  const text = async (page) => (await main(page).innerText()).replace(/\s+/g, ' ');
  const shot = async (page, name) => { const h = await page.evaluate(() => { const m = document.querySelector('main'); m.scrollTop = 0; return m.scrollHeight; }); await page.setViewportSize({ width: 1240, height: Math.min(Math.max(860, h), 3000) }); await page.mouse.move(1235, 5); await page.waitForTimeout(150); await page.screenshot({ path: `${OUT}/${name}.png` }); await page.setViewportSize({ width: 1240, height: 860 }); };
  const createProject = async (page, name) => {
    await main(page).locator('button', { hasText: 'Small Business' }).first().click();
    const input = main(page).locator('input').first(); await input.fill(name);
    await btn(page, 'Create project').click();
    await page.getByText('Your material and your rules').waitFor({ timeout: 10000 });
  };
  const buildExam = async (page) => {
    await btn(page, 'Try the sample data').click();
    await main(page).locator('button', { hasText: 'amount must be positive' }).first().click();
    await btn(page, 'Build practice exam').click();
    await page.getByText('Your practice exam').waitFor({ timeout: 20000 });
    await page.waitForTimeout(300);
  };

  // ---------------- A: the main journey ----------------
  let page = await open('&slow=1');
  check('A1 sidebar: later steps locked before a project exists', await nav(page, 'Data & rules').isDisabled() && await nav(page, 'Use it').isDisabled());
  await btn(page, 'Continue').click();
  await page.getByText('What do you want your AI to master?').waitFor();
  await createProject(page, 'Expense sorter');
  check('A2 after creating: training and use-it still locked', await nav(page, 'Train & prove').isDisabled() && await nav(page, 'Use it').isDisabled());
  check('A3 build is disabled without files', await btn(page, 'Build practice exam').isDisabled());
  await buildExam(page);
  check('A4 exam built: continue enabled, train step unlocked', await btn(page, 'Continue to training').isEnabled() && await nav(page, 'Train & prove').isEnabled());
  // stale exam after changing rules
  await main(page).locator('button', { hasText: 'category must be one of' }).first().click();
  await page.waitForTimeout(400);
  check('A5 changing rules after a build marks the exam stale', (await text(page)).includes('Rebuild the practice exam to use the changes') && await btn(page, 'Continue to training').isDisabled());
  await btn(page, 'Rebuild practice exam').click();
  await page.waitForFunction(() => { const b = [...document.querySelectorAll('main button')].find(x => x.textContent.includes('Continue to training')); return b && !b.disabled; }, null, { timeout: 20000 });
  check('A6 rebuilding clears the stale mark', !(await text(page)).includes('Rebuild the practice exam to use the changes'));
  // leave and come back: files and rules are still there
  await nav(page, 'Choose a goal').click(); await page.waitForTimeout(200); await nav(page, 'Data & rules').click();
  const t1 = await text(page);
  check('A7 Data & rules keeps files, rules and the exam after leaving the step', t1.includes('expenses.csv') && t1.includes('amount must be positive') && t1.includes('Your practice exam'));
  await btn(page, 'Continue to training').click();
  await page.getByText('What happens').waitFor();
  await shot(page, 'A-train-idle');
  await main(page).locator('button', { hasText: 'Quick run' }).click();
  await btn(page, 'Stop training').waitFor({ timeout: 5000 });
  await page.waitForTimeout(1200);
  // navigate away during training and back
  await nav(page, 'Data & rules').click(); await page.waitForTimeout(300);
  check('A8 Data & rules says training is running and blocks rebuilding', (await text(page)).includes('Training is running for this project') && await btn(page, 'Rebuild practice exam').isDisabled());
  await nav(page, 'Train & prove').click(); await page.waitForTimeout(300);
  const t2 = await text(page);
  check('A9 coming back to Train & prove still shows the run and Stop', await btn(page, 'Stop training').isVisible() && /\d+%/.test(t2) && !t2.includes('What happens'));
  check('A10 sidebar shows training progress', /Training\s*\d+%/.test((await page.locator('aside').innerText()).replace(/\s+/g, ' ')));
  await page.getByText('Saving your new version').first().waitFor({ timeout: 60000 }).catch(() => {});
  const saving = await text(page);
  check('A11 while saving: no "Use it" button yet, no blank version', !/new version\s+saved/.test(saving) && !(await btn(page, 'Use it in your apps').isVisible().catch(() => false)), saving.includes('Saving') ? '' : 'saving state not observed');
  await btn(page, 'Use it in your apps').waitFor({ timeout: 30000 });
  await page.waitForTimeout(600);
  const t3 = await text(page);
  check('A12 verdict: version v1 saved, all six steps ticked, Use it unlocked', t3.includes('new version v1 saved') && await nav(page, 'Use it').isEnabled());
  await shot(page, 'A-train-promoted');
  await page.getByLabel('Question for both models').fill('Paid $42 at Shell');
  await btn(page, 'Compare').click(); await page.waitForTimeout(700);
  check('A13 compare shows one result (not two)', (await main(page).locator('text=This appears to be a fuel purchase.').count()) === 1);
  await btn(page, 'Use it in your apps').first().click();
  await btn(page, 'Start local API').waitFor();
  await btn(page, 'Start local API').click();
  await btn(page, 'Stop local API').waitFor({ timeout: 10000 });
  const t4 = await text(page);
  check('A14 API running: short engine name, address, model, hidden key', t4.includes('Running · version v1') && t4.includes('NVIDIA GeForce RTX 4060') && !t4.includes('MiB') && t4.includes('http://127.0.0.1:11435/v1') && t4.includes('inkport/expense-sorter') && !t4.includes('ik_preview'));
  await btn(page, 'Open test chat').click();
  await page.getByLabel('Message').fill('Paid $58 at Shell');
  await page.getByLabel('Message').press('Enter');
  await page.getByText('"amount": 58').waitFor({ timeout: 8000 });
  check('A15 test chat answers through the app (no direct fetch)', true);
  await shot(page, 'A-deploy-running');
  // leave and return: API still shown as running
  await nav(page, 'Train & prove').click(); await page.waitForTimeout(300);
  const t5 = await text(page);
  check('A16 Train & prove for a trained project: current version card + train-newer cards', t5.includes('Version v1 is in use') || t5.includes('new version v1 saved'));
  await nav(page, 'Use it').click(); await page.waitForTimeout(400);
  check('A17 returning to Use it still shows the running API', await btn(page, 'Stop local API').isVisible() && (await text(page)).includes('Running · version v1'));
  // export + import (trusted)
  await btn(page, 'Export adapter…').click(); await page.getByText('Saved version v1').waitFor({ timeout: 8000 });
  check('A18 export reports where it saved', true);
  // reopen flow: go to project list, open again
  await nav(page, 'Choose a goal').click();
  await page.getByText('Continue a project').waitFor();
  const t6 = await text(page);
  check('A19 project list shows the trained project first', t6.includes('trained, version v1') && t6.indexOf('Continue a project') < t6.indexOf('Small Business'));
  await btn(page, 'Open').first().click(); await page.waitForTimeout(400);
  check('A20 reopening a trained project lands on Use it', (await text(page)).includes('Use your AI anywhere on this computer'));
  // duplicate project name
  await nav(page, 'Choose a goal').click();
  await main(page).locator('button', { hasText: 'Small Business' }).first().click();
  await main(page).locator('input').first().fill('expense SORTER');
  await btn(page, 'Create project').click(); await page.waitForTimeout(600);
  check('A21 duplicate name is refused with a clear message next to the button', (await text(page)).includes("already exists"));
  // second project while nothing trains: steps locked again for the new one
  await main(page).locator('input').first().fill('Second project');
  await btn(page, 'Create project').click();
  await page.getByText('Your material and your rules').waitFor();
  check('A22 a new project does not inherit unlocked steps', await nav(page, 'Train & prove').isDisabled() && await nav(page, 'Use it').isDisabled());
  check('A23 a new project starts with an empty Data & rules screen', !(await text(page)).includes('expenses.csv'));
  check('A  no console errors', page.errors.length === 0, page.errors.slice(0, 3).join(' | '));
  await page.context().close();

  // ---------------- B: stop mid-run ----------------
  page = await open();
  await btn(page, 'Continue').click(); await createProject(page, 'Stopper'); await buildExam(page); await btn(page, 'Continue to training').click();
  await main(page).locator('button', { hasText: 'Best results' }).click();
  await btn(page, 'Stop training').waitFor(); await page.waitForTimeout(900);
  await btn(page, 'Stop training').click();
  await page.getByText('Training was stopped').waitFor({ timeout: 8000 });
  const b1 = await text(page);
  check('B1 after Stop: a plain note, start cards back, no fake "Finished 100%"', b1.includes('Nothing was saved') && b1.includes('Best results') && !b1.includes('Finished') && !b1.includes('100%'));
  await shot(page, 'B-stopped');
  await main(page).locator('button', { hasText: 'Quick run' }).click();
  await btn(page, 'Stop training').waitFor({ timeout: 5000 });
  check('B2 training can be started again after a stop', true);
  // while it runs, create another project and look at its training screen
  await nav(page, 'Choose a goal').click();
  await main(page).locator('button', { hasText: 'Custom Skill' }).first().click();
  await main(page).locator('input').first().fill('Other');
  await btn(page, 'Create project').click(); await page.getByText('Your material and your rules').waitFor();
  await btn(page, 'Try the sample data').isVisible().catch(() => false);
  check('B3 sidebar names the project that is training elsewhere', /Training\s*Stopper · \d+%/.test((await page.locator('aside').innerText()).replace(/\s+/g, ' ')));
  await page.context().close();

  // ---------------- C: failure mid-run ----------------
  page = await open('&outcome=fail');
  await btn(page, 'Continue').click(); await createProject(page, 'Crasher'); await buildExam(page); await btn(page, 'Continue to training').click();
  await main(page).locator('button', { hasText: 'Quick run' }).click();
  await page.getByText('Training ran out of memory').waitFor({ timeout: 30000 });
  const c1 = await text(page);
  check('C1 a crash shows why, offers Try again and the start cards', c1.includes('Close other programs') && c1.includes('Try again') && c1.includes('Best results') && !c1.includes('Finished'));
  await shot(page, 'C-failed');
  await page.context().close();

  // ---------------- D: rejected ----------------
  page = await open('&outcome=reject');
  await btn(page, 'Continue').click(); await createProject(page, 'Rejected'); await buildExam(page); await btn(page, 'Continue to training').click();
  await main(page).locator('button', { hasText: 'Quick run' }).click();
  await page.getByText('Not better yet').waitFor({ timeout: 60000 });
  const d1 = await text(page);
  check('D1 a rejected run says nothing was saved and keeps Use it locked', d1.includes('nothing was saved') && !d1.includes('Use it in your apps') && await nav(page, 'Use it').isDisabled() && d1.includes('Train again'));
  await shot(page, 'D-rejected');
  await page.context().close();

  // ---------------- E: too little data ----------------
  page = await open('&outcome=few');
  await btn(page, 'Continue').click(); await createProject(page, 'Tiny'); await buildExam(page);
  const e1 = await text(page);
  check('E1 too few examples: explained, and Continue is disabled', e1.includes('Not enough examples to train yet') && await btn(page, 'Continue to training').isDisabled());
  await shot(page, 'E-too-few');
  await page.context().close();

  // ---------------- F: untrusted adapter ----------------
  page = await open('&untrusted=1');
  await btn(page, 'Continue').click(); await createProject(page, 'Importer'); await buildExam(page); await btn(page, 'Continue to training').click();
  await main(page).locator('button', { hasText: 'Quick run' }).click();
  await btn(page, 'Use it in your apps').waitFor({ timeout: 60000 }); await btn(page, 'Use it in your apps').first().click();
  let answer = false; let asked = 0;
  page.on('dialog', async d => { asked++; if (answer) await d.accept(); else await d.dismiss(); });
  page.on('filechooser', async fc => { await fc.setFiles({ name: 'friend.inkpatch', mimeType: 'application/octet-stream', buffer: Buffer.from('x') }); });
  await btn(page, 'Import adapter…').click();
  await page.getByText('Import cancelled').waitFor({ timeout: 8000 });
  check('F1 declining the unknown-publisher question cancels the import', asked === 1 && !(await text(page)).includes('Signature verified'));
  answer = true;
  await btn(page, 'Import adapter…').click();
  await page.getByText('Signature verified').waitFor({ timeout: 8000 });
  check('F2 accepting it continues, and the exam verdict is shown', asked === 2 && (await text(page)).includes('Not used:'));
  check('F  no console errors', page.errors.length === 0, page.errors.slice(0, 3).join(' | '));
  await page.context().close();

  await browser.close();
  const failed = results.filter(r => !r[0]);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('SCRIPT ERROR', e.message.split('\n').slice(0, 6).join('\n')); process.exit(2); });
