// UI-only onboarding regression. API fixtures prevent training, evaluation and downloads.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_CORE);
const catalog = require(process.env.STUDIO_CATALOG);
const demo = require(process.env.STUDIO_DEMO);
const locales = { en: require('../frontend/src/i18n/locales/en.json'), 'zh-CN': require('../frontend/src/i18n/locales/zh-CN.json') };
const areas = ['canvas', 'library', 'properties', 'toolbar', 'benchmark', 'results'];
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH, headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  page.setDefaultTimeout(8000);
  const errors = [], actions = [];
  page.on('pageerror', e => errors.push(String(e)));
  let failDemo = false, releaseDemo, heldDemo = null, demoRequests = 0;
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (['/api/train', '/api/evaluate', '/api/graph/compile', '/api/graphs/save', '/api/resources/download'].includes(path)) actions.push(path);
    if (path === '/api/templates/naml-demo') {
      demoRequests++;
      if (heldDemo) await heldDemo;
      return route.fulfill({ status: failDemo ? 500 : 200, json: failDemo ? { detail: 'Fixture: unavailable demo' } : demo });
    }
    const body = path === '/api/components' ? catalog : path === '/api/templates' ? { templates: [demo] } : path === '/api/environment' ? { warnings: [] } : path === '/api/runs' ? { runs: [] } : path === '/api/graphs' ? { graphs: [] } : path === '/api/resources/status' ? { items: [{ state: 'unnecessary' }] } : path === '/api/graph/validate' ? { valid: true, errors: [], warnings: [], runtime_checks: [], summary: { model_label: 'NAML', nodes: 5, edges: 5 } } : {};
    return route.fulfill({ json: body });
  });
  await page.addInitScript(() => localStorage.setItem('studio-language', 'en'));
  const tour = page.locator('.studio-tour');
  const dialog = page.locator('.studio-tour-card');
  const snapshot = () => page.evaluate(async () => JSON.stringify((await import('/src/store/graphStore.ts')).useStudio.getState().graph));
  const showWelcome = async locale => {
    await page.evaluate(async () => (await import('/src/store/graphStore.ts')).useStudio.setState({ welcomeVisible: true }));
    await page.getByRole('combobox', { name: 'Language / 语言' }).last().selectOption(locale);
  };
  const openDemo = async locale => {
    await page.getByRole('button', { name: locales[locale]['ui.open_naml_demo_99e6e3'], exact: true }).click();
    await dialog.waitFor({ state: 'visible' });
    await page.waitForTimeout(100);
    assert.equal(await tour.getAttribute('data-tour-step'), 'canvas');
  };
  const assertBounds = async () => {
    assert(await dialog.evaluate(el => { const r = el.getBoundingClientRect(); return r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight; }));
    assert(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth));
  };
  try {
    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5187');
    await page.getByRole('button', { name: locales.en['ui.open_naml_demo_99e6e3'], exact: true }).waitFor();
    assert.equal(await tour.count(), 0, 'no tour before the explicit welcome action');
    // Loading state prevents duplicate requests, and the tour waits for a loaded graph.
    heldDemo = new Promise(resolve => { releaseDemo = resolve; });
    const launch = page.getByRole('button', { name: locales.en['ui.open_naml_demo_99e6e3'], exact: true });
    await launch.click();
    const busyLaunch = page.getByRole('button', { name: locales.en['tour.openingDemo'], exact: true });
    assert(await busyLaunch.isDisabled());
    assert.equal(await busyLaunch.getAttribute('aria-busy'), 'true');
    assert.equal(await tour.count(), 0);
    releaseDemo(); heldDemo = null;
    await dialog.waitFor({ state: 'visible' });
    await page.waitForTimeout(700);
    assert.equal(demoRequests, 1);
    assert(await page.locator('.react-flow__node').count() > 0, 'the demo has loaded before the tour');
    const original = await snapshot();
    assert(await page.locator('.app-shell').evaluate(el => el.inert));
    assert.equal(await dialog.locator('h2').evaluate(el => document.activeElement === el), true);
    await page.keyboard.press('Tab');
    assert(await dialog.evaluate(el => el.contains(document.activeElement)), 'tab enters the guide controls');
    await page.keyboard.press('Shift+Tab');
    assert(await dialog.evaluate(el => el.contains(document.activeElement)), 'focus remains in the guide');
    await page.keyboard.press('Delete');
    assert.equal(await snapshot(), original);
    await dialog.getByRole('button', { name: locales.en['tour.close'], exact: true }).click();
    assert.equal(await tour.count(), 0);
    assert.equal(await page.locator('.app-shell').evaluate(el => el.inert), false);
    // Both languages, each target, back navigation, finish and live language switching.
    for (const locale of ['en', 'zh-CN']) {
      await showWelcome(locale);
      await openDemo(locale);
      await page.waitForTimeout(400);
      const before = await snapshot();
      for (let step = 0; step < areas.length; step++) {
        assert.equal(await tour.getAttribute('data-tour-step'), areas[step]);
        assert.equal(await dialog.locator('h2').innerText(), locales[locale][`tour.${areas[step]}.title`]);
        assert(!/\btour\.[a-z]/.test(await dialog.innerText()), 'all guide strings are translated');
        const target = await page.locator(`[data-tour-target="${areas[step]}"]`).boundingBox();
        const highlight = page.locator(`[data-tour-highlight="${areas[step]}"]`);
        assert.equal(await highlight.count(), 1);
        assert(Math.abs(Number(await highlight.getAttribute('x')) - Math.max(6, target.x - 4)) < 2);
        await assertBounds();
        if (step === 2) {
          const other = locale === 'en' ? 'zh-CN' : 'en';
          await dialog.getByRole('combobox').selectOption(other);
          assert.equal(await tour.getAttribute('data-tour-step'), 'properties');
          assert.equal(await dialog.locator('h2').innerText(), locales[other]['tour.properties.title']);
          await dialog.getByRole('combobox').selectOption(locale);
          await page.screenshot({ path: `/tmp/legoizer-tour-${locale}.png` });
        }
        if (areas[step] === 'benchmark') {
          assert.equal(await page.locator('[data-tour-target="benchmark"]').getAttribute('href'), '#/benchmark');
          assert.equal(await dialog.locator('.tour-map-benchmark').getAttribute('aria-current'), 'step');
          assert(!page.url().endsWith('#/benchmark'), 'the guide does not navigate away from the model');
          await page.screenshot({ path: `/tmp/legoizer-tour-benchmark-${locale}.png` });
        }
        if (step === 3) {
          await dialog.getByRole('button', { name: locales[locale]['tour.previous'], exact: true }).click();
          assert.equal(await tour.getAttribute('data-tour-step'), 'properties');
          await dialog.getByRole('button', { name: locales[locale]['tour.next'], exact: true }).click();
        }
        await dialog.getByRole('button', { name: locales[locale][step === areas.length - 1 ? 'tour.finish' : 'tour.next'], exact: true }).click();
      }
      assert.equal(await tour.count(), 0);
      assert.equal(await snapshot(), before, 'the complete tour leaves the demo graph unchanged');
      assert.equal(await page.locator('.app-shell').evaluate(el => el.inert), false);
      assert.equal(await page.locator('[data-tour-target="canvas"]').evaluate(el => document.activeElement === el), true);
    }
    // Replay the tour from welcome, skip with Escape and the explicit button.
    await showWelcome('en'); await openDemo('en');
    await page.keyboard.press('ArrowRight');
    assert.equal(await tour.getAttribute('data-tour-step'), 'library');
    await page.keyboard.press('ArrowLeft');
    assert.equal(await tour.getAttribute('data-tour-step'), 'canvas');
    await page.keyboard.press('Escape');
    assert.equal(await tour.count(), 0);
    await showWelcome('en'); await openDemo('en');
    await dialog.getByRole('button', { name: locales.en['tour.skip'], exact: true }).click();
    assert.equal(await tour.count(), 0);
    // Ordinary template cards and a new canvas do not start onboarding.
    await showWelcome('en');
    await page.getByRole('button').filter({ has: page.getByText(demo.name, { exact: true }) }).click();
    await page.waitForTimeout(200);
    assert.equal(await tour.count(), 0);
    await showWelcome('en');
    await page.getByRole('button').filter({ has: page.getByText(locales.en['ui.new_canvas_6cd287'], { exact: true }) }).click();
    assert.equal(await tour.count(), 0);
    // A failed load never starts a guide over the previous/empty graph.
    await showWelcome('en'); failDemo = true;
    await page.getByRole('button', { name: locales.en['ui.open_naml_demo_99e6e3'], exact: true }).click();
    await page.waitForTimeout(200);
    assert.equal(await tour.count(), 0);
    assert(await page.getByRole('button', { name: locales.en['ui.open_naml_demo_99e6e3'], exact: true }).isEnabled());
    failDemo = false;
    // Dark theme, resized and short windows, and missing/offscreen target fallback.
    await openDemo('en');
    await page.evaluate(async () => (await import('/src/store/graphStore.ts')).useStudio.setState({ theme: 'dark' }));
    await page.setViewportSize({ width: 1000, height: 700 });
    await page.waitForTimeout(150); await assertBounds();
    await page.setViewportSize({ width: 390, height: 640 });
    await page.waitForTimeout(150);
    for (let i = 0; i < areas.length; i++) {
      await assertBounds();
      if (i === 2) await page.screenshot({ path: '/tmp/legoizer-tour-mobile.png' });
      if (i < areas.length - 1) await dialog.getByRole('button', { name: locales.en['tour.next'], exact: true }).click();
    }
    await page.keyboard.press('Escape');
    await page.setViewportSize({ width: 1500, height: 1000 });
    await showWelcome('en'); await openDemo('en');
    await page.locator('[data-tour-target="canvas"]').evaluate(el => { el.style.display = 'none'; });
    await page.waitForTimeout(150);
    assert.equal(await page.locator('[data-tour-highlight]').count(), 0);
    assert(await dialog.getByText(locales.en['tour.compact'], { exact: true }).isVisible());
    await assertBounds();
    await page.keyboard.press('Escape');
    await page.locator('[data-tour-target="canvas"]').evaluate(el => { el.style.display = ''; });
    assert.deepEqual(actions, [], 'onboarding never starts training, evaluation, compilation, saving or downloads');
    assert.deepEqual(errors, []);
    console.log('PASS NAML-only success trigger, loading/failure handling, 6 regions × 2 languages, dedicated Benchmark spotlight and introduction, live language switch, spotlight alignment, keyboard/focus, skip/finish/replay, graph unchanged, no run actions, dark/resized/mobile/offscreen layouts');
  } catch (error) {
    console.error(errors);
    console.error((await page.locator('body').innerText()).slice(-4000));
    await page.screenshot({ path: '/tmp/legoizer-tour-failure.png' });
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
