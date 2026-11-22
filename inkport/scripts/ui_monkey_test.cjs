#!/usr/bin/env node
// Random clicks and typing in the browser preview (`pnpm dev`), the way an impatient person or a
// cat on the keyboard would use it: no step may produce a page error, a console error or an empty
// screen. The sequence depends only on the seed, so a failure can be replayed.
//
//   node scripts/ui_monkey_test.cjs <seed> <steps> ["&outcome=reject" | "&outcome=fail" | "&outcome=few" | "&untrusted=1"]
//
// Needs the playwright package with a Chromium (the same as scripts/ui_flows_test.cjs).
'use strict';
const { chromium } = require('playwright');
const seed0 = Number(process.argv[2] || 1), STEPS = Number(process.argv[3] || 400), QUERY = process.argv[4] || '';
let seed = seed0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const pick = (a) => a[Math.floor(rnd() * a.length)];
const TEXTS = ['amount must be positive', 'price must be positive', 'make it nice', 'if category is travel then amount must be under 500', '', '   ', 'x'.repeat(300), 'Ünïcödé 日本語 👍', '<b>hi</b>', '"quoted"', 'Paid $64 at Shell', 'My project', 'my project', 'CON', '../etc', 'a/b\\c'];
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1240, height: 860 } })).newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('dialog', (d) => (rnd() < 0.5 ? d.accept() : d.dismiss()).catch(() => {}));
  await page.goto('http://127.0.0.1:1420/?speed=' + pick(['3', '8', '20']) + QUERY);
  await page.getByRole('button', { name: 'Continue', exact: true }).waitFor({ timeout: 30000 });
  const trail = [];
  for (let i = 0; i < STEPS; i++) {
    const action = rnd();
    try {
      if (action < 0.72) {
        const buttons = await page.locator('button:visible:not([disabled])').all();
        if (buttons.length) {
          const b = pick(buttons);
          const label = ((await b.textContent().catch(() => '')) || (await b.getAttribute('aria-label').catch(() => '')) || '').trim().slice(0, 40);
          trail.push('click ' + label);
          await b.click({ timeout: 1500, noWaitAfter: true }).catch(() => {});
        }
      } else if (action < 0.9) {
        const inputs = await page.locator('main input:visible, main textarea:visible').all();
        if (inputs.length) {
          const el = pick(inputs), t = pick(TEXTS);
          trail.push('type ' + JSON.stringify(t.slice(0, 20)));
          await el.fill(t, { timeout: 1500 }).catch(() => {});
          if (rnd() < 0.6) await el.press('Enter').catch(() => {});
        }
      } else if (action < 0.95) {
        trail.push('wait');
        await page.waitForTimeout(400 + rnd() * 1500);
      } else {
        trail.push('key');
        await page.keyboard.press(pick(['Escape', 'Tab', 'Enter', 'Backspace'])).catch(() => {});
      }
      await page.waitForTimeout(20 + rnd() * 120);
      const len = await page.evaluate(() => (document.querySelector('main') ? document.querySelector('main').innerText.trim().length : -1));
      if (len < 20) errors.push(`empty screen (main text length ${len})`);
    } catch (e) { errors.push('driver: ' + e.message.split('\n')[0]); }
    if (errors.length) {
      console.log(`seed ${seed0}${QUERY}: FAILED at step ${i}: ${errors.slice(0, 3).join(' | ')}`);
      console.log('  last actions: ' + trail.slice(-12).join(' > '));
      await page.screenshot({ path: `ui-monkey-fail-${seed0}.png` }).catch(() => {});
      await browser.close();
      process.exit(1);
    }
  }
  console.log(`seed ${seed0}${QUERY}: ${STEPS} steps ok`);
  await browser.close();
})().catch((e) => { console.log('monkey could not run: ' + e.message); process.exit(2); });
