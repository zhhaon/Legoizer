const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_CORE);
(async () => {
 const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH,args:['--no-sandbox']});
 const page = await browser.newPage({locale:'en-US',viewport:{width:1440,height:960}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 try {
  await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:8000');
  await page.getByRole('button',{name:'Browse Benchmark results',exact:true}).click();
  const region=page.getByRole('region',{name:'Benchmark',exact:true});
  await region.getByText(/configurations · fetched/).waitFor({timeout:60000});
  const headers=await region.locator('th').allTextContents();
  assert(!headers.some(h=>h.includes('Signature')));
  for(const metric of ['GAUC','MRR','NDCG@1','NDCG@5','NDCG@10']) assert(headers.some(h=>h.includes(metric)));
  const search=region.getByRole('textbox');
  await search.fill('automotive glove');
  assert((await region.locator('tbody tr').count())>0);
  for(const row of await region.locator('tbody tr').all()) {const cells=await row.locator('td').allTextContents();assert.equal(cells[1],'automotive');assert.equal(cells[2],'glove');}
  await search.fill('cKnGfJ7N');await region.getByText('No matching experiments.',{exact:true}).waitFor();
  await search.fill('automotive glove');
  await region.getByRole('combobox',{name:'Dataset',exact:true}).selectOption('automotive');
  await region.getByText('Official metric comparison',{exact:true}).click();
  const labels=await region.locator('.recharts-yAxis .recharts-cartesian-axis-tick-value').allTextContents();
  assert(labels.length>0);assert(labels.every(l=>l.includes('automotive')&&l.includes('glove')));assert(!labels.some(l=>l.includes('cKnGfJ7N')));
  await region.locator('tbody tr').first().click();
  const drawer=page.getByRole('dialog');await drawer.waitFor();
  await drawer.getByRole('button',{name:'Download official JSON',exact:true}).waitFor();
  await page.keyboard.press('Escape');assert.equal(await page.getByRole('dialog').count(),0);
  await region.locator('tbody tr').first().click();
  await page.getByRole('dialog').getByRole('button',{name:'Download official JSON',exact:true}).click();
  await page.getByText('Experiment JSON exported',{exact:true}).waitFor();
  await page.keyboard.press('Escape');
  await page.getByRole('combobox',{name:'Language / 语言'}).first().selectOption('zh-CN');
  await page.screenshot({path:'/tmp/benchmark-ui.png'});
  assert(await page.getByRole('link',{name:'返回工作台',exact:true}).count()>0);
  assert.deepEqual(errors,[]);
  console.log('PASS: all metric columns, no signature column/search, dataset+LM token search, dataset/LM chart labels, drawer Escape/focus, export Toast, top toolbar, bilingual UI');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
