/**
 * Optional browser-level End-to-End Acceptance Test.
 *
 * Prerequisites:
 *   1. `bash studio/run.sh` running (backend + Vite dev server).
 *   2. A Playwright installation. Set PLAYWRIGHT_CORE to the playwright-core
 *      module path, or install `playwright` in studio/frontend.
 *
 * Usage:
 *   STUDIO_URL=http://127.0.0.1:5173 node studio/tests/browser_e2e.cjs
 *
 * It performs the exact acceptance workflow from the specification:
 *   Welcome → NAML template → replace Additive Attention with GRU →
 *   auto validation → Generate Model → real LSTUR YAML → Train → real evaluation metrics.
 */
const path = require('node:path');

const URL = process.env.STUDIO_URL || 'http://127.0.0.1:8000';
const CHROME = process.env.CHROME_PATH || undefined;

function loadChromium() {
  if (process.env.PLAYWRIGHT_CORE) return require(process.env.PLAYWRIGHT_CORE).chromium;
  try {
    return require(path.resolve(__dirname, '../frontend/node_modules/playwright')).chromium;
  } catch {
    try { return require('playwright').chromium; } catch { return null; }
  }
}

(async () => {
  const chromium = loadChromium();
  if (!chromium) {
    console.error('Playwright is not installed. Set PLAYWRIGHT_CORE or `npm i -D playwright` in studio/frontend.');
    process.exit(2);
  }
  const browser = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({
    viewport: { width: 1680, height: 1000 },
    ignoreHTTPSErrors: process.env.STUDIO_TEST_SELF_SIGNED === '1',
  });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

  const check = (name, condition) => {
    console.log(`${condition ? 'PASS' : 'FAIL'}  ${name}`);
    if (!condition) process.exitCode = 1;
  };

  try {
    await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });
    await page.evaluate(() => localStorage.clear());
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    check('Welcome screen is shown', (await page.getByText('Welcome to Legoizer').count()) === 1);

    await page.getByRole('button', { name: /Start from Scratch/ }).click();
    const scratchSearch = page.getByRole('textbox', { name: 'Search components' });
    const pane = page.locator('.react-flow__pane');
    await scratchSearch.fill('dataset.xmind');
    await page.locator('.lib-item').first().dragTo(pane, { targetPosition: { x: 140, y: 155 } });
    await scratchSearch.fill('item.cnn');
    await page.locator('.lib-item').filter({ has: page.getByText('CNN', { exact: true }) })
      .dragTo(pane, { targetPosition: { x: 520, y: 155 } });

    const viewport = page.locator('.react-flow__viewport');
    const getViewport = () => viewport.evaluate((element) => element.style.transform);
    const port = (direction) => page.locator('.react-flow__handle.' + direction + '[data-handleid="content"]');
    const center = async (locator) => {
      const box = await locator.boundingBox();
      if (!box) throw new Error('The dropped component has no visible port');
      return [box.x + box.width / 2, box.y + box.height / 2];
    };
    const before = await getViewport();
    await page.mouse.move(...await center(port('source')));
    await page.mouse.down();
    await page.mouse.move(...await center(port('target')), { steps: 16 });
    await page.mouse.up();
    await page.locator('.react-flow__edge').waitFor({ state: 'attached', timeout: 3000 });
    check('Newly dropped components connect without panning the canvas', await getViewport() === before);

    const canvas = await pane.boundingBox();
    await page.mouse.move(canvas.x + 80, canvas.y + 340);
    await page.mouse.down();
    await page.mouse.move(canvas.x + 110, canvas.y + 360, { steps: 8 });
    await page.mouse.up();
    check('Dragging blank canvas still pans', await getViewport() !== before);

    const wheelStart = await getViewport();
    await page.mouse.move(canvas.x + 90, canvas.y + 340);
    await page.mouse.wheel(54, 78);
    await page.waitForFunction((start) =>
      document.querySelector('.react-flow__viewport').style.transform !== start, wheelStart);
    check('Trackpad-style horizontal and vertical scroll pans without zooming',
      (await getViewport()).split('scale')[1] === wheelStart.split('scale')[1]);

    await page.evaluate(() => localStorage.clear());
    await page.reload({ waitUntil: 'networkidle' });
    await page.getByRole('button', { name: /Open NAML Demo/ }).click();
    await page.waitForTimeout(1800);
    const nodes = await page.locator('.lego-node').count();
    const edges = await page.locator('.react-flow__edge').count();
    check(`NAML template renders ${nodes} nodes / ${edges} edges`, nodes === 5 && edges === 7);

    check('Metrics are absent from the component palette', (await page.locator('aside').first().getByText('Evaluation', { exact: true }).count()) === 0);
    const library = page.locator('aside').first();
    await library.getByRole('textbox', { name: 'Search components' }).fill('dataset.xmind');
    const source = library.locator('.lib-item').filter({ hasText: /xmind/i }).first();
    await source.dragTo(page.locator('.react-flow__pane'), { targetPosition: { x: 170, y: 220 } });
    const datasetNode = page.locator('.react-flow__node[data-id="dataset_1"]');
    const itemInputs = page.getByRole('region', { name: 'Dataset item inputs' });
    await itemInputs.getByLabel('Item column 1', { exact: true }).waitFor();
    check('Dragging xMIND replaces the dataset in place and keeps connections',
      await page.locator('.lego-node').count() === nodes &&
      await page.locator('.react-flow__edge').count() === edges &&
      /xmind/i.test(await datasetNode.innerText()));
    check('xMIND shows its resolved title tokenizer',
      await itemInputs.getByLabel('Item column 1', { exact: true }).inputValue() === 'title@llama1');
    await page.keyboard.press('Control+z');
    await datasetNode.click();
    await itemInputs.getByLabel('Item column 2', { exact: true }).waitFor();
    check('MIND lists title@glove and category after undo',
      await itemInputs.getByLabel('Item column 1', { exact: true }).inputValue() === 'title@glove' &&
      await itemInputs.getByLabel('Item column 2', { exact: true }).inputValue() === 'category');
    await itemInputs.getByLabel('Truncation for item column 1', { exact: true }).fill('18');
    await itemInputs.getByRole('button', { name: 'Save inputs' }).click();
    check('Dataset input changes are saved',
      await itemInputs.getByLabel('Truncation for item column 1', { exact: true }).inputValue() === '18');
    await library.getByRole('textbox', { name: 'Search components' }).fill('');

    await page.locator('.react-flow__node[data-id="user_1"]').click();
    await page.waitForTimeout(300);
    check('Additive Attention node is selected',
      (await page.locator('aside').last().innerText()).includes('Additive Attention'));

    await page.getByRole('button', { name: 'Replace', exact: true }).first().click();
    await page.waitForTimeout(500);
    check('Replace Component dialog opens',
      (await page.locator('.modal-backdrop h3').first().innerText()) === 'Replace Additive Attention');

    const gruCard = page.locator('.modal-backdrop div.rounded-lg.border')
      .filter({ hasText: 'LSTUR user encoder' }).first();
    check('GRU is offered as a compatible replacement', (await gruCard.count()) === 1);
    await gruCard.getByRole('button', { name: 'Replace' }).click();
    await page.waitForTimeout(2000);
    check('Node morphed into GRU',
      (await page.locator('.react-flow__node[data-id="user_1"]').innerText()).startsWith('GRU'));
    check('Graph re-validated automatically', (await page.locator('header').innerText()).includes('Valid'));

    await page.getByRole('button', { name: 'Generate Model', exact: true }).click();
    await page.waitForTimeout(2000);
    const yaml = await page.locator('pre').first().innerText();
    check('Generated YAML selects the real GRU operator', yaml.includes('user: GRU'));
    check('Generated YAML is identified as LSTUR', yaml.includes('name: LSTUR'));

    await page.keyboard.press('Escape');
    await page.getByRole('button', { name: 'Training', exact: true }).click();
    const properties = page.getByRole('complementary', { name: 'Properties Panel' })
      .getByRole('region', { name: 'Training settings' });
    const training = page.getByRole('region', { name: 'Experiment configuration' })
      .getByRole('region', { name: 'Training settings' });
    const fields = (panel) => panel.locator('input, select').evaluateAll((elements) =>
      elements.map((element) => element.getAttribute('aria-label') || element.closest('label')?.textContent?.trim()));
    check('Properties and Training offer the same editable settings',
      JSON.stringify(await fields(properties)) === JSON.stringify(await fields(training)));
    await properties.getByRole('spinbutton', { name: 'Accumulate batches' }).fill('3');
    check('A value edited in Properties appears in Training',
      await training.getByRole('spinbutton', { name: 'Accumulate batches' }).inputValue() === '3');
    await training.getByLabel('Experiment Name', { exact: true }).fill('Studio LSTUR browser check');
    check('A value edited in Training appears in Properties',
      await properties.getByLabel('Experiment Name', { exact: true }).inputValue() === 'Studio LSTUR browser check');
    await training.getByRole('combobox', { name: 'Metric to add' }).selectOption('F1');
    await training.getByRole('button', { name: 'Add metric' }).click();
    check('Evaluation metrics are shared', await properties.getByRole('button', { name: 'Remove F1@0.5' }).count() === 1);
    const tokenField = training.getByLabel('Hugging Face Access Token (optional)');
    const canSendToken = await tokenField.count() === 1;
    if (URL.startsWith('https:')) check('Token is editable on the HTTPS page', canSendToken && await tokenField.isEnabled());
    const fakeToken = 'hf_studioBrowserCheck123456';
    if (canSendToken) {
      await properties.getByLabel('Hugging Face Access Token (optional)').fill(fakeToken);
      check('Token draft is shared between the two panels', await tokenField.inputValue() === fakeToken);
    }
    if (process.env.STUDIO_TEST_DEVICE) {
      await training.getByRole('combobox', { name: /^Device/ }).selectOption(process.env.STUDIO_TEST_DEVICE);
    }
    const started = page.waitForResponse((response) => response.url().endsWith('/api/train') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Train', exact: true }).click();
    const response = await started;
    const run = await response.json();
    if (canSendToken) {
      const headers = await response.request().allHeaders();
      check('Token is sent only as a request header and is cleared after submit',
        headers['x-hugging-face-token'] === fakeToken
        && !response.request().postData().includes(fakeToken)
        && !JSON.stringify(run).includes(fakeToken) && await tokenField.inputValue() === '');
    }
    if (process.env.STUDIO_TEST_DEVICE === 'cuda') {
      check('CUDA is selected in the browser and submitted to trainer.py',
        run.params?.device === 'cuda' && run.command[run.command.indexOf('--cuda') + 1] === '0');
    }
    check('Train submits accumulate_batch=3', run.params?.accumulate_batch === 3);
    if (!run.run_id) throw new Error(JSON.stringify(run));

    const final = await page.waitForResponse(async (response) => {
      if (!response.url().endsWith('/' + run.run_id + '/status')) return false;
      const status = (await response.json()).status;
      return ['completed', 'failed', 'cancelled'].includes(status);
    }, { timeout: 180000 });
    const result = await final.json();
    check('Trainer and evaluation completed', result.status === 'completed' && result.exit_code === 0);
    check('Run used the real trainer and emitted GAUC, MRR, NDCG', result.command.includes('trainer.py')
      && ['GAUC', 'MRR', 'NDCG@5', 'NDCG@10', 'F1@0.5'].every((metric) => Number.isFinite(result.metrics[metric])));
    if (result.error) console.log(result.error);
    console.log('  run:', run.run_id, 'metrics:', result.metrics);

    if (process.env.STUDIO_DATASET_HF_TEST === '1') {
      await datasetNode.click();
      const datasetProperties = page.getByRole('complementary', { name: 'Properties Panel' });
      await datasetProperties.getByRole('combobox', { name: 'data config', exact: true }).selectOption('config/data/mind-demo.yaml');
      await datasetProperties.getByRole('combobox', { name: 'Token resource source' }).selectOption('external_hf');
      await datasetProperties.getByRole('textbox', { name: 'model id', exact: true }).fill(
        process.env.STUDIO_EXTERNAL_HF_REPO || 'hf-internal-testing/tiny-random-bert',
      );
      await itemInputs.getByLabel('Item column 2', { exact: true }).waitFor();
      check('Dataset resolves external text tokens and keeps the category feature',
        await itemInputs.getByLabel('Item column 1', { exact: true }).inputValue() === 'title@external_hf' &&
        await itemInputs.getByLabel('Item column 2', { exact: true }).inputValue() === 'category');
      const datasetStarted = page.waitForResponse((reply) => reply.url().endsWith('/api/train') && reply.request().method() === 'POST');
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      const datasetRun = await (await datasetStarted).json();
      if (!datasetRun.run_id) throw new Error(JSON.stringify(datasetRun));
      const datasetFinal = await page.waitForResponse(async (reply) => {
        if (!reply.url().endsWith('/' + datasetRun.run_id + '/status')) return false;
        return ['completed', 'failed', 'cancelled'].includes((await reply.json()).status);
      }, { timeout: 240000 });
      const datasetResult = await datasetFinal.json();
      check('CNN trains with the Dataset external tokenizer and pretrained token embeddings',
        datasetResult.status === 'completed' && Number.isFinite(datasetResult.metrics.GAUC) &&
        datasetResult.graph.nodes.some((node) => node.component === 'item.cnn') &&
        datasetResult.config.token_resources?.owner === 'dataset');
      console.log('  external dataset run:', datasetRun.run_id, datasetResult.error || datasetResult.metrics);
    }

    if (process.env.STUDIO_EXTERNAL_HF_TEST === '1') {
      await library.getByRole('textbox', { name: 'Search components' }).fill('item.externalhf');
      await library.locator('.lib-item').dragTo(pane, { targetPosition: { x: 350, y: 175 } });
      const external = page.getByRole('complementary', { name: 'Properties Panel' });
      await external.getByRole('textbox', { name: 'model id', exact: true }).fill(
        process.env.STUDIO_EXTERNAL_HF_REPO || 'hf-internal-testing/tiny-random-bert',
      );
      const activeRepo = process.env.STUDIO_EXTERNAL_HF_REPO || 'hf-internal-testing/tiny-random-bert';
      const datasetLm = datasetNode.locator('[data-param-name="lm"] span[title]');
      await page.waitForFunction((repo) => document.querySelector(
        '.react-flow__node[data-id="dataset_1"] [data-param-name="lm"] span[title]',
      )?.textContent === repo, activeRepo);
      check('Dataset brick displays the active external model tokenizer',
        await datasetLm.innerText() === activeRepo);
      await external.getByRole('textbox', { name: 'tokenizer id', exact: true }).fill('bert-base-uncased');
      check('Dataset LM updates immediately for an explicit external tokenizer',
        await datasetLm.innerText() === 'bert-base-uncased');
      await external.getByRole('textbox', { name: 'tokenizer id', exact: true }).fill('');
      check('Dataset LM returns to the model tokenizer when the override is cleared',
        await datasetLm.innerText() === activeRepo);
      await external.getByRole('combobox', { name: 'encoding mode', exact: true }).selectOption(process.env.STUDIO_EXTERNAL_HF_MODE || 'token_embeddings');
      check('External Hugging Face replaces the Item Encoder without breaking connections',
        await page.locator('.react-flow__node[data-id="item_1"]').getByText('External Hugging Face', { exact: true }).count() === 1 &&
        await page.locator('.react-flow__edge').count() === edges);

      const externalStarted = page.waitForResponse((reply) => reply.url().endsWith('/api/train') && reply.request().method() === 'POST');
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      const externalRun = await (await externalStarted).json();
      if (!externalRun.run_id) throw new Error(JSON.stringify(externalRun));
      const externalFinal = await page.waitForResponse(async (reply) => {
        if (!reply.url().endsWith('/' + externalRun.run_id + '/status')) return false;
        return ['completed', 'failed', 'cancelled'].includes((await reply.json()).status);
      }, { timeout: 240000 });
      const externalResult = await externalFinal.json();
      check('Browser trains with the external tokenizer and selected Item Encoder mode',
        externalResult.status === 'completed' && Number.isFinite(externalResult.metrics.GAUC));
      console.log('  external run:', externalRun.run_id, externalResult.error || externalResult.metrics);
    }

    check('No JavaScript errors', errors.length === 0);
    if (errors.length) console.log('  errors:', errors.slice(0, 8));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error('FATAL', error); process.exit(1); });
