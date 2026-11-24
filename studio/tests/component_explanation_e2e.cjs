// Read-only browser regression: backend operations are mocked, using a live catalog snapshot.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_CORE);
const catalog = require(process.env.STUDIO_CATALOG);
const target = catalog.components.filter(c => ['item_encoder', 'user_encoder', 'predictor', 'training'].includes(c.category));
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH, headless: true, args: ['--no-sandbox'] });
  const context = await browser.newContext({ viewport: { width: 1500, height: 1000 } });
  const page = await context.newPage();
  const errors = [], keyWarnings = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (/same key|unique.*key/i.test(m.text())) keyWarnings.push(m.text()); });
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url());
    if (!url.pathname.startsWith('/api/')) return route.continue();
    const body = url.pathname.endsWith('/components') ? catalog : url.pathname.endsWith('/graph/validate') ? { valid: true, errors: [], warnings: [], runtime_checks: [], summary: { model_label: 'Test', nodes: 0, edges: 0 } } : url.pathname.endsWith('/graphs') ? { graphs: [] } : url.pathname.endsWith('/templates') ? { templates: [] } : url.pathname.endsWith('/environment') ? { warnings: [] } : url.pathname.endsWith('/runs') ? { runs: [] } : url.pathname.endsWith('/resources/status') ? { items: [{ state: 'unnecessary' }] } : {};
    return route.fulfill({ json: body });
  });
  await page.addInitScript(() => { localStorage.setItem('legommenders-studio:welcome-seen', 'true'); localStorage.setItem('studio-language', 'en'); });
  const dialog = page.getByRole('dialog');
  const locale = value => page.evaluate(async v => { await (await import('/src/i18n/index.ts')).setLocale(v); }, value);
  const dismissTooltip = async () => {
    if (await page.locator('[data-neuron-tooltip]').count()) await dialog.locator('.explanation-header').click({ position: { x: 40, y: 20 } });
    await page.mouse.move(1480, 10);
    await page.locator('[data-neuron-tooltip]').waitFor({ state: 'hidden', timeout: 1500 });
  };
  const pause = async () => { const button = dialog.getByRole('button', { name: /^(Pause|暂停)$/, exact: true }); if (await button.count()) await button.click(); };
  const open = async id => {
    await page.locator('aside input').first().fill(id);
    const row = page.locator(`[data-component-id="${id}"]`);
    await row.hover(); await row.locator('[data-explain-component]').click(); await dialog.waitFor();
    assert.equal(await dialog.getAttribute('data-explanation-id'), id);
    assert.equal(await dialog.locator('[data-explanation-step]').getAttribute('data-explanation-step'), '0');
    assert.equal(await dialog.locator('[data-neuron-network]').count(), 1);
    assert.equal(await dialog.locator('[data-explanation-story-step]').count(), 0, 'right panel has no step navigation');
    assert.equal(await dialog.locator('.explanation-organization button').count(), 3, 'structural sections locate neuron groups without showing steps');
    assert.equal(await dialog.locator('[data-explanation-organization]').count(), 3);
    assert.equal(await dialog.locator('.explanation-progress').count(), 0);
    assert.equal(await dialog.locator('.neuron-example-note').count(), 1);
    const reduced = await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
    assert.equal(await dialog.getByRole('button', { name: /^(Pause|暂停)$/, exact: true }).count(), reduced ? 0 : 1);
    for (const text of await dialog.locator('[data-explanation-organization-body]').allTextContents()) assert(text.trim().length > 0, `${id}: every group has an organization description`);
  };
  const selectStage = async stage => {
    await pause();
    let current = Number(await dialog.locator('[data-explanation-step]').getAttribute('data-explanation-step'));
    while (current !== stage) {
      const forward = stage > current;
      await dialog.getByRole('button', { name: forward ? /^(Next|下一步)$/ : /^(Previous|上一步)$/, exact: true }).click();
      current += forward ? 1 : -1;
    }
  };
  const close = async id => {
    await page.keyboard.press('Escape'); assert.equal(await dialog.count(), 0);
    assert.equal(await page.locator('[data-neuron-tooltip]').count(), 0, 'inspector portal is removed on close');
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('data-explain-component')), id);
  };
  const assertCentered = async () => {
    await page.waitForTimeout(30);
    const geometry = await dialog.locator('[data-mechanism-centered]').evaluate(el => {
      const view = el.closest('.explanation-visual'), target = el.getBoundingClientRect(), bounds = view.getBoundingClientRect();
      const dx = target.left + target.width / 2 - bounds.left - view.clientWidth / 2;
      const dy = target.top + target.height / 2 - bounds.top - view.clientHeight / 2;
      return { x: view.scrollLeft, y: view.scrollTop, desiredX: Math.max(0, Math.min(view.scrollWidth - view.clientWidth, view.scrollLeft + dx)), desiredY: Math.max(0, Math.min(view.scrollHeight - view.clientHeight, view.scrollTop + dy)) };
    });
    assert(Math.abs(geometry.x - geometry.desiredX) <= 2 && Math.abs(geometry.y - geometry.desiredY) <= 2, `natural centered bounds: ${JSON.stringify(geometry)}`);
    assert(await dialog.locator('.explanation-visual').evaluate(el => getComputedStyle(el, '::before').content === 'none' && getComputedStyle(el, '::after').content === 'none'), 'no end spacers');
  };
  const hoverAndCheck = async (node, expected) => {
    await dismissTooltip();
    await node.scrollIntoViewIfNeeded(); await node.hover();
    await dialog.locator(`[data-neuron-trace="${expected.id}"]`).waitFor();
    assert.equal(await page.getByRole('tooltip').count(), 0, 'hover never opens a calculation panel');
    const paths = await dialog.locator('[data-neuron-edge]').evaluateAll(edges => edges.map(edge => ({ id: edge.dataset.neuronEdge, active: edge.classList.contains('is-inspected') })));
    const connected = paths.filter(edge => edge.id.split(':').includes(expected.id)).map(edge => edge.id).sort();
    assert.deepEqual(paths.filter(edge => edge.active).map(edge => edge.id).sort(), connected, 'only immediate incoming and outgoing connections are highlighted');
    assert.equal(await dialog.locator('.neuron-edge.is-organization,.neuron-edge.is-flowing').count(), 0, 'node tracing suppresses whole-group and playback edge highlights');
    await node.click();
    const tooltip = page.locator(`[data-neuron-tooltip="${expected.id}"]`); await tooltip.waitFor({ state: 'visible' });
    assert.equal(Number(await tooltip.locator('[data-neuron-tooltip-result]').getAttribute('data-neuron-tooltip-result')), expected.value || 0);
    assert((await tooltip.innerText()).includes(expected.label));
    assert.equal(await tooltip.locator('tbody tr').count(), expected.terms.length);
    assert.equal(await dialog.locator('.neuron-unit.is-source,.neuron-unit.is-organization').count(), 0, 'node inspection does not highlight a whole neuron group');
    for (const term of expected.terms) assert(await dialog.locator(`[data-neuron-edge="${term.id}:${expected.id}"].is-inspected`).count());
    assert(!(await tooltip.innerText()).match(/\bneuron\.[A-Za-z_]|undefined|NaN/));
    assert(await tooltip.evaluate(el => { const r = el.getBoundingClientRect(); return r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1; }), 'inspector fits viewport');
    assert.equal(await dialog.getByRole('button', { name: /^(Pause|暂停)$/, exact: true }).count(), 0, 'hover pauses playback');
    assert(await tooltip.evaluate(el => {
      const visual = document.querySelector('.explanation-visual').getBoundingClientRect(), info = document.querySelector('.explanation-organization').getBoundingClientRect(), box = el.getBoundingClientRect();
      return visual.width < box.width + 16 || box.right <= info.left || box.left >= info.right;
    }), 'neuron calculations leave the linked explanation readable');
    const group = await node.getAttribute('data-neuron-organization');
    await dialog.locator(`.explanation-organization-section.is-linked[data-explanation-organization="${group}"]`).waitFor();
  };
  const modelFor = id => page.evaluate(async id => {
    const { explanationFor } = await import('/src/explanations/catalog.ts');
    const { buildNeuralGraph, defaultExample } = await import('/src/explanations/neuronModels.ts');
    const family = explanationFor(id).key;
    return { ...buildNeuralGraph(family, id, defaultExample), family };
  }, id);
  try {
    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5187'); await page.waitForSelector('.lib-item');
    const graph = () => page.evaluate(async () => JSON.stringify((await import('/src/store/graphStore.ts')).useStudio.getState().graph));
    const original = await graph();
    const inspectedFamilies = new Set(); let hoverCount = 0;
    const testedLanguages = process.env.STUDIO_TEST_LANGUAGE ? [process.env.STUDIO_TEST_LANGUAGE] : ['en', 'zh-CN'];
    for (const language of testedLanguages) {
      await locale(language);
      let completed = 0;
      for (const spec of target) {
        await open(spec.id); await pause();
        const model = await modelFor(spec.id);
        const rendered = await dialog.locator('[data-neuron]').evaluateAll(nodes => nodes.map(n => [n.dataset.neuron, Number(n.dataset.neuronValue)]));
        assert.deepEqual(rendered, model.nodes.map(n => [n.id, n.value || 0]), 'every circle displays its actual executable result');
        const story = await dialog.locator('[data-explanation-organization-body]').allTextContents();
        for (let stage = 0; stage < 4; stage++) {
          await selectStage(stage);
          assert.equal(await dialog.locator('[data-explanation-step]').getAttribute('data-explanation-step'), String(stage));
          assert(await dialog.locator(`[data-neuron-stage="${stage}"]`).count()); await assertCentered();
          assert.deepEqual(await dialog.locator('[data-explanation-organization-body]').allTextContents(), story, 'organization notes stay unchanged during animation');
        }
        assert(!(await dialog.innerText()).match(/\b(?:explain|neuron|mechanism)\.[A-Za-z_]/), `${spec.id} has no unresolved translation`);
        if (language === 'zh-CN') for (const body of story) assert(/[\u4e00-\u9fff]/.test(body));
        // Inspect every node in a small complete network; cover every operation/activation pair in every family.
        let inspect = [model.nodes.at(-1)];
        if (language === 'en' && !inspectedFamilies.has(model.family)) {
          inspectedFamilies.add(model.family);
          for (let group = 0; group < 3; group++) {
            const section = dialog.locator(`[data-explanation-organization="${group}"]`);
            await section.hover(); await dialog.locator(`.explanation-organization-section.is-linked[data-explanation-organization="${group}"]`).waitFor();
            const members = await dialog.locator(`[data-neuron-organization="${group}"]`).count();
            assert(members > 0, `${spec.id} section ${group} has real nodes`);
            assert.equal(await dialog.locator(`.neuron-unit.is-organization [data-neuron-organization="${group}"]`).count(), members);
            assert(await dialog.locator('.neuron-edge.is-organization').count(), 'structural links are highlighted');
            await section.click();
            assert.equal(await dialog.locator('[data-neuron-stage-focus]').getAttribute('data-organization-centered'), String(group));
            await assertCentered();
          }
          const variants = new Set();
          inspect = model.nodes.filter(n => {
            const key = `${n.op}:${n.activation}:${n.kind}`;
            if (['dot', 'cnn', 'matching_task'].includes(model.family) || !variants.has(key)) { variants.add(key); return true; }
            return false;
          });
        }
        for (const node of inspect) { await hoverAndCheck(dialog.locator(`[data-neuron="${node.id}"]`), node); hoverCount++; }
        await close(spec.id);
        if (++completed % 10 === 0) console.log(`${language}: ${completed}/${target.length} complete`);
      }
      console.log(`PASS ${language}: ${target.length} components; static organization notes, animation controls, and centered bounds; ${hoverCount} hover traces and click panels checked`);
    }
    await locale('en');
    for (const id of ['item.cnn', 'predictor.dnn']) {
      await open(id); const nodes = await dialog.locator('[data-neuron]').count();
      for (let i = 0; i < 12; i++) {
        if (i % 2 === 0) await selectStage(2);
        await dialog.getByRole('button', { name: 'Replay', exact: true }).click();
        assert.equal(await dialog.locator('[data-neuron-network]').count(), 1);
        assert.equal(await dialog.locator('[data-neuron]').count(), nodes);
        assert.equal(await dialog.locator('.explanation-stage > .explanation-visual > *').count(), 1);
        assert.equal(await dialog.locator('[data-neuron-network]').getAttribute('data-mechanism-step'), '0');
      }
      await close(id);
    }
    const values = () => dialog.locator('[data-neuron-output]').evaluateAll(ns => ns.map(n => Number(n.dataset.value)));
    for (const [id, group, button] of [
      ['predictor.dot', 'Try another candidate', 'Cooking tutorial'],
      ['user.pooling', 'Compare aggregation rules', 'Max'],
      ['user.gru', 'Change sequence order', 'Reverse order'],
      ['item.externalhf', 'Encoding mode', 'Token embeddings only'],
      ['predictor.miner', 'Interest score reduction', 'Max'],
    ]) {
      await open(id); await pause(); const before = await values();
      await dialog.getByRole('group', { name: group, exact: true }).getByRole('button', { name: button, exact: true }).click();
      assert.notDeepEqual(await values(), before, `${id} recomputes its network`);
      if (id === 'predictor.dot') assert(Math.abs((await values())[0] - .26) < 1e-12);
      if (id === 'user.pooling') assert.deepEqual(await values(), [1, 1]);
      if (id === 'item.externalhf') assert.equal(await dialog.locator('[data-neuron-op="softmax"]').count(), 0);
      await close(id);
    }
    for (const id of ['training.matching_task', 'training.ranking_task']) {
      await open(id); await pause(); const [before, after] = await values(); assert(after < before, 'updated parameters reduce loss');
      if (id.includes('ranking')) { await dialog.getByRole('group', { name: 'Observed click label' }).getByRole('button', { name: 'y = 0' }).click(); const [b, a] = await values(); assert(a < b); }
      await close(id);
    }
    // Autoplay, static structural notes, live translation, and node animation pause.
    await open('item.cnn'); await page.mouse.move(1480, 10);
    const story = await dialog.locator('[data-explanation-organization-body]').allTextContents();
    await page.waitForTimeout(3000);
    assert.equal(await dialog.locator('[data-explanation-step]').getAttribute('data-explanation-step'), '1');
    assert.deepEqual(await dialog.locator('[data-explanation-organization-body]').allTextContents(), story);
    await pause(); const frame = await dialog.locator('[data-neuron-network]').getAttribute('data-mechanism-frame');
    await page.waitForTimeout(600); assert.equal(await dialog.locator('[data-neuron-network]').getAttribute('data-mechanism-frame'), frame);
    await dialog.getByRole('button', { name: 'Play', exact: true }).click(); await page.mouse.move(1480, 10);
    await page.waitForTimeout(500); assert.notEqual(await dialog.locator('[data-neuron-network]').getAttribute('data-mechanism-frame'), frame);
    const cnn = await modelFor('item.cnn'), conv = cnn.nodes.find(n => n.label === 'c1');
    await hoverAndCheck(dialog.locator(`[data-neuron="${conv.id}"]`), conv);
    await page.screenshot({ path: '/tmp/legoizer-neurons-cnn-en.png' });
    await locale('zh-CN'); assert((await page.getByRole('tooltip').innerText()).includes('输入与连接权重'));
    await page.screenshot({ path: '/tmp/legoizer-neurons-cnn-zh.png' });
    await dismissTooltip();
    // Keyboard focus traces connections; activation opens persistent calculations.
    const first = dialog.locator('[data-neuron]').first(); await first.focus();
    const firstId = await first.getAttribute('data-neuron'); await page.keyboard.press('ArrowRight');
    assert.notEqual(await page.evaluate(() => document.activeElement?.getAttribute('data-neuron')), firstId);
    assert.equal(await page.getByRole('tooltip').count(), 0, 'keyboard focus alone does not open calculations');
    const focused = dialog.locator('[data-neuron]:focus'); const focusedId = await focused.getAttribute('data-neuron'); await page.keyboard.press('Enter');
    await page.locator(`[data-neuron-tooltip="${focusedId}"]`).waitFor();
    const pinned = await page.getByRole('tooltip').getAttribute('data-neuron-tooltip');
    await page.mouse.move(1480, 10); await page.waitForTimeout(200); assert.equal(await page.getByRole('tooltip').getAttribute('data-neuron-tooltip'), pinned);
    const another = dialog.locator('[data-neuron]').last(), anotherId = await another.getAttribute('data-neuron');
    await another.scrollIntoViewIfNeeded(); await another.hover();
    await dialog.locator(`[data-neuron-trace="${anotherId}"]`).waitFor();
    assert.equal(await page.getByRole('tooltip').getAttribute('data-neuron-tooltip'), pinned, 'hovering another node does not replace the clicked calculation panel');
    await another.click(); await page.locator(`[data-neuron-tooltip="${anotherId}"]`).waitFor();
    await another.click(); await page.getByRole('tooltip').waitFor({ state: 'hidden' });
    await another.click(); await page.locator(`[data-neuron-tooltip="${anotherId}"]`).waitFor();
    await dialog.locator('[data-explanation-organization="0"]').focus(); await page.keyboard.press('Enter');
    await page.getByRole('tooltip').waitFor({ state: 'hidden' });
    assert.equal(await dialog.locator('[data-neuron-stage-focus]').getAttribute('data-organization-centered'), '0');
    await assertCentered();
    await selectStage(2);
    assert.equal(await page.getByRole('tooltip').count(), 0);
    await dialog.getByRole('button', { name: '关闭讲解' }).focus(); await page.keyboard.press('Shift+Tab');
    assert.equal(await page.evaluate(() => document.activeElement.textContent), '查看参数');
    await close('item.cnn');
    await locale('en'); await open('user.attention'); await pause();
    await selectStage(1);
    await dialog.getByRole('button', { name: 'Fit network', exact: true }).click();
    await assertCentered(); await page.screenshot({ path: '/tmp/legoizer-neurons-attention-overview.png' });
    await close('user.attention');
    await page.emulateMedia({ reducedMotion: 'reduce' }); await locale('zh-CN');
    await open('user.gru');
    await page.waitForTimeout(600); assert.equal(await dialog.locator('[data-neuron-network]').getAttribute('data-mechanism-frame'), '0');
    assert((await dialog.innerText()).includes('已减少动态效果'));
    await selectStage(1); await assertCentered();
    await page.evaluate(() => document.documentElement.classList.add('dark'));
    const gru = await modelFor('user.gru'), gate = gru.nodes.find(n => n.activation === 'sigmoid');
    await hoverAndCheck(dialog.locator(`[data-neuron="${gate.id}"]`), gate);
    await page.screenshot({ path: '/tmp/legoizer-neurons-dark.png' });
    await close('user.gru');
    await page.setViewportSize({ width: 390, height: 844 }); await open('predictor.dot');
    await selectStage(2); await assertCentered();
    assert(await dialog.evaluate(el => { const r = el.getBoundingClientRect(); return r.x >= 0 && r.right <= innerWidth && r.bottom <= innerHeight; }));
    assert(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), 'dialog has no horizontal overflow');
    await dialog.locator('[data-explanation-organization="1"]').click(); await assertCentered();
    assert(await dialog.locator('.explanation-visual').evaluate(el => { const pane = el.closest('.explanation-body'), r = el.getBoundingClientRect(), bounds = pane.getBoundingClientRect(); return r.top >= bounds.top - 2 && r.top < bounds.bottom; }), 'mobile organization selection brings the graph into view');
    const dot = await modelFor('predictor.dot'), product = dot.nodes.find(n => n.op === 'product');
    await hoverAndCheck(dialog.locator(`[data-neuron="${product.id}"]`), product);
    await page.screenshot({ path: '/tmp/legoizer-neurons-mobile.png' });
    await close('predictor.dot');
    assert.equal(await graph(), original, 'help never changes the user graph');
    assert.deepEqual(errors, []); assert.deepEqual(keyWarnings, []);
    console.log(`PASS: ${target.length} components × ${testedLanguages.length} languages; ${inspectedFamilies.size} model families; ${hoverCount} direct-neighbor hover traces and click-only panels; all scalar values and highlighted dependencies; component-specific organization notes without numbered steps; bidirectional group hover, click-to-locate, natural group centering; natural centering; 24 replays; candidate/order/pooling/mode/reduction controls; real parameter updates; autoplay/pause; live localization; keyboard, pin, focus return; reduced motion; dark and mobile; graph unchanged`);
  } catch (error) { console.error(errors); await page.screenshot({ path: '/tmp/legoizer-neurons-failure.png' }); throw error; }
  finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
