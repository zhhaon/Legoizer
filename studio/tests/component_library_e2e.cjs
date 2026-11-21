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
    await setGraph(graphs[0]);
    const read = () => page.evaluate(async () => (await import('/src/store/graphStore.ts')).useStudio.getState().graph);
    const search = page.getByRole('textbox', { name: 'Search components' });
    const addFromDetails = async id => {
      await search.fill(id);
      const component = page.locator('.lib-item').filter({ has: page.getByText(id, { exact: true }) }).first();
      assert(!(await component.getAttribute('title')).includes('double-click'));
      await component.click();
      const hasSameRole = (await read()).nodes.some(n => components.get(n.component)?.role === components.get(id).role);
      const label = hasSameRole ? 'Replace Component' : 'Add to canvas';
      assert(await page.getByRole('button', { name: label, exact: true }).isVisible());
      await page.getByRole('button', { name: label, exact: true }).click();
      await page.waitForTimeout(400);
    };
    const replacement = [...components.values()].find(c => c.role === 'item_encoder' && c.id !== 'item.cnn').id;
    for (const view of ['Data Flow', 'Config']) {
      const before = await read();
      const original = before.nodes.find(n => n.component.startsWith('item.'));
      await addFromDetails(replacement);
      let graph = await read();
      assert.equal(graph.nodes.length, before.nodes.length);
      assert.equal(graph.nodes.find(n => n.id === original.id).component, replacement);
      assert.equal(graph.nodes.filter(n => n.component.startsWith('item.')).length, 1);
      // Adding the exact same component selects it instead of making duplicates.
      await addFromDetails(replacement);
      assert.equal((await read()).nodes.length, before.nodes.length);
      await addFromDetails('item.cnn');
      assert.equal((await read()).nodes.find(n => n.id === original.id).component, 'item.cnn');
      assert.deepEqual((await read()).edges, before.edges);
      if (view === 'Data Flow') {
        await page.getByRole('button', { name: 'Data Flow → Config', exact: true }).click();
        await page.waitForTimeout(600);
      }
    }
    await page.evaluate(async () => {
      const state = (await import('/src/store/graphStore.ts')).useStudio.getState();
      state.deleteNodes(state.graph.nodes.filter(n => n.component.startsWith('item.')).map(n => n.id));
    });
    const beforeAdd = (await read()).nodes.length;
    await addFromDetails('item.cnn');
    assert.equal((await read()).nodes.length, beforeAdd + 1);
    assert.deepEqual(errors, []);
    console.log('PASS dynamic Add/Replace labels match actual actions; details button replaces same-role component, preserves identity and compatible edges, avoids exact duplicates in both views, adds when role absent');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
