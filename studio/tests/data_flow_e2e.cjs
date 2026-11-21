// UI-only regression: mock API responses; never compiles or trains models.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_CORE);
const graphs = require('../frontend/src/canvas/fixtures/modelGraphs.json');
const role = id => id.startsWith('item.') ? 'item_encoder' : id.startsWith('user.') ? 'user_encoder' : id.split('.')[0];
const components = new Map();
const port = (name, type) => ({ name, type, required: false, description: '' });
for (const graph of graphs) for (const node of graph.nodes) {
  if (!components.has(node.component)) components.set(node.component, {
    id: node.component, name: node.component, role: role(node.component), category: role(node.component),
    description: '', icon: 'Box', status: 'verified', executable: true, conceptual: false,
    backend_class: null, meta_value: node.component, inputs: [], outputs: [], parameters: [],
    tasks: ['matching', 'ranking'], source: {}, detail: '', status_note: '',
  });
  const spec = components.get(node.component);
  for (const edge of graph.edges) {
    if (edge.source === node.id && !spec.outputs.some(p => p.name === edge.sourceHandle)) spec.outputs.push(port(edge.sourceHandle, edge.sourceHandle));
    if (edge.target === node.id && !spec.inputs.some(p => p.name === edge.targetHandle)) spec.inputs.push(port(edge.targetHandle, edge.targetHandle));
  }
}
for (const spec of components.values()) if (spec.role === 'dataset' && !spec.outputs.some(p => p.name === 'content')) spec.outputs.push(port('content', 'content'));
const catalog = { components: [...components.values()], categories: [...new Set([...components.values()].map(s => s.category))].map(id => ({ id, label: id, color: '#16a34a' })), port_types: [], generated_from: 'UI regression fixture' };
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH, headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
  page.setDefaultTimeout(5000);
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.route('**/api/**', route => {
    const url = route.request().url();
    if (!new URL(url).pathname.startsWith('/api/')) return route.continue();
    const body = url.endsWith('/components') ? catalog : url.endsWith('/templates') ? { templates: [] } : url.endsWith('/environment') ? { warnings: [] } : url.endsWith('/runs') ? { runs: [] } : url.endsWith('/graph/validate') ? { valid: true, errors: [], warnings: [], runtime_checks: [], summary: {} } : {};
    return route.fulfill({ json: body });
  });
  await page.addInitScript(() => localStorage.setItem('legommenders-studio:welcome-seen', 'true'));
  try {
    await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:5187');
    await page.waitForTimeout(500);
    const setGraph = async graph => {
      await page.evaluate(async graph => {
        const { useStudio } = await import('/src/store/graphStore.ts');
        useStudio.setState(s => ({ graph, graphLoadVersion: s.graphLoadVersion + 1, selectedNodeId: null, selectedEdgeId: null }));
      }, graph);
      await page.waitForTimeout(600);
    };
    for (const graph of graphs) {
      await setGraph(graph);
      assert(await page.locator('.react-flow__node[data-id="user:embedding"]').count(), graph.name);
      assert(await page.locator('.react-flow__node[data-id="positive:predictor"]').count(), graph.name);
    }
    console.log('PASS all 14 templates retain expanded nodes without rendering errors');
    const din = graphs.find(g => g.nodes.some(n => n.component === 'predictor.din'));
    await setGraph(din);
    const search = page.getByRole('textbox', { name: 'Search components' });
    await search.fill('item.cnn');
    await page.locator('.lib-item').first().dragTo(page.locator('[data-testid="model-canvas-pane"]'), { targetPosition: { x: 160, y: 200 } });
    await page.waitForTimeout(700);
    const read = () => page.evaluate(async () => (await import('/src/store/graphStore.ts')).useStudio.getState().graph);
    let graph = await read();
    const item = graph.nodes.find(n => n.component === 'item.cnn');
    assert(item);
    assert(!graph.edges.some(e => e.source === item.id || e.target === item.id));
    const orphan = page.locator(`.react-flow__node[data-id="extra:${item.id}"]`);
    assert(await orphan.count());
    await orphan.click({ button: 'right' });
    assert(await page.getByText('Replace Component…', { exact: true }).isVisible());
    await page.keyboard.press('Escape');
    // Drag actual ports while remaining in Data Flow.
    const center = async locator => {
      const b = await locator.boundingBox(); assert(b); return [b.x + b.width / 2, b.y + b.height / 2];
    };
    const drag = async (source, target) => {
      await page.mouse.move(...await center(source)); await page.mouse.down();
      await page.mouse.move(...await center(target), { steps: 20 }); await page.mouse.up();
      await page.waitForTimeout(500);
    };
    const datasetId = din.nodes.find(n => n.component.startsWith('dataset.')).id;
    await drag(page.locator(`[data-id="extra:${datasetId}"] .source[data-handleid="port:content"]`), orphan.locator('.target[data-handleid="port:content"]'));
    graph = await read();
    assert(graph.edges.some(e => e.source === datasetId && e.target === item.id));
    await page.locator(`[data-id="extra:${item.id}"]`).click();
    await page.keyboard.press('Delete');
    await page.waitForTimeout(500);
    assert(!(await read()).nodes.some(n => n.id === item.id));
    console.log('PASS Data Flow drag-add, orphan display, context menu, real-port connect and component delete');
    // Both views edit the same real module, not its display id.
    for (const label of ['Data Flow → Config', 'Config → Data Flow']) {
      await page.getByRole('button', { name: label, exact: true }).click();
      await page.waitForTimeout(600);
      const node = page.locator(label.startsWith('Data Flow') ? '[data-id="user_1"]' : '[data-id="user:encoder"]');
      await node.click();
      const properties = page.getByRole('complementary', { name: 'Properties Panel' });
      await properties.locator('input.field-input').first().fill(`Edited in ${label}`);
      assert.equal((await read()).nodes.find(n => n.id === 'user_1').label, `Edited in ${label}`);
      await search.fill('user.gru');
      await page.locator('.lib-item').first().dragTo(node);
      await page.waitForTimeout(500);
      assert.equal((await read()).nodes.find(n => n.id === 'user_1').component, 'user.gru');
      await node.click({ button: 'right' });
      assert(await page.getByText('Replace Component…', { exact: true }).isVisible());
      await page.keyboard.press('Escape');
    }
    const beforeDelete = (await read()).edges.length;
    await page.locator('.react-flow__edge').filter({ has: page.locator('path') }).evaluateAll(elements => {
      const edge = elements.find(e => e.getAttribute('data-id')?.startsWith('positive:score->objective'));
      if (!edge) throw new Error('Missing expanded score connection');
      edge.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await page.keyboard.press('Delete');
    assert.equal((await read()).edges.length, beforeDelete - 1);
    await page.keyboard.press('Control+z');
    assert.equal((await read()).edges.length, beforeDelete);
    assert.deepEqual(errors, []);
    console.log('PASS edits, drag replacement in both views, expanded-edge deletion and undo; no page errors');
    await page.screenshot({ path: '/tmp/lego-data-flow.jpg', type: 'jpeg', quality: 35, fullPage: true });
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
