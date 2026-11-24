// Optional browser regression: set PLAYWRIGHT_CORE, CHROME_PATH and STUDIO_URL.
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_CORE || path.resolve(__dirname,'../frontend/node_modules/playwright'));
const URL=process.env.STUDIO_URL || 'http://127.0.0.1:8000';
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || undefined,args:['--no-sandbox']});
 const errors=[];
 try{
 for(const lang of ['en','zh-CN']){
  const page=await browser.newPage({locale:'en-US',viewport:{width:1800,height:1200}});
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(URL,{waitUntil:'networkidle'});
  await page.getByRole('button',{name:/Start from Scratch/}).click();
  await page.getByRole('combobox',{name:'Language / 语言'}).selectOption(lang);
  await page.getByRole('button',{name:lang==='en'?'Templates':'模板',exact:true}).click();
  await page.locator('.studio-toolbar button[title="config/model/naml.yaml"]').first().click();
  await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node-detail').length>20);
  await page.locator('.tab-btn').filter({hasText:'Graph JSON'}).click();
  const graph=JSON.parse(await page.locator('pre.code-view').innerText());
  await page.getByRole('button',{name:lang==='en'?'Focus canvas':'聚焦画布',exact:true}).click();
  await page.waitForTimeout(900);
  // Trackpad two-finger gestures must pan even over node configuration rows.
  const viewport = () => page.locator('.react-flow__viewport').evaluate(el => {
   const m = new DOMMatrixReadOnly(getComputedStyle(el).transform);
   return { x: m.m41, y: m.m42, zoom: m.a };
  });
  const regions = page.locator('.react-flow__node-detail [aria-label][class*="overflow-y-auto"]');
  assert.ok(await regions.count(), 'Exercise configurable Data Flow nodes');
  for (let i = 0; i < await regions.count(); i++) {
   for (const [dx, dy] of [[80, 0], [0, 80], [60, 40]]) {
    await page.getByRole('button', {name: lang === 'en' ? 'Fit' : '适应视图', exact: true}).click();
    await page.waitForTimeout(500);
    const region = regions.nth(i), box = await region.boundingBox();
    assert.ok(box, 'Configuration region is visible');
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    const before = await viewport();
    await page.mouse.wheel(dx, dy);
    await page.waitForTimeout(150);
    const after = await viewport();
    assert.equal(after.zoom, before.zoom, 'Trackpad pan preserves zoom');
    if (dx) assert.ok(after.x < before.x - 1, 'Horizontal pan over configuration');
    if (dy) assert.ok(after.y < before.y - 1, 'Vertical pan over configuration');
   }
  }
  await page.getByRole('button', {name: lang === 'en' ? 'Fit' : '适应视图', exact: true}).click();
  await page.waitForTimeout(500);
  console.log('PASS '+lang+': trackpad pans over every configuration region in both axes');
  const ids=await page.locator('.react-flow__node-detail').evaluateAll(nodes=>nodes.map(node=>node.dataset.id));
  const node=id=>page.locator('.react-flow__node-detail[data-id="'+id+'"]');
  const position=locator=>locator.evaluate(el=>{const t=el.style.transform.match(/translate\(([-\d.]+)px,\s*([-\d.]+)px\)/);return {x:Number(t[1]),y:Number(t[2])}});
  const initial={},dragged={};
  for(const id of ids)initial[id]=await position(node(id));
  for(const id of ids){
   const target=node(id),body=target.locator('.node-card-body');
   const box=await (await body.count()?body:target).boundingBox();
   assert.ok(box&&box.width>0&&box.height>0,'Visible drag surface '+id);
   const before=await position(target);
   const x=box.x+box.width/2,y=box.y+box.height/2;
   await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+22,y+19,{steps:8});await page.mouse.up();
   await page.waitForTimeout(65);
   const after=await position(target);
   assert.ok(after.x>before.x+1&&after.y>before.y+1,lang+' '+id+' failed two-axis drag: '+JSON.stringify({before,after}));
   dragged[id]=after;
  }
  await page.waitForTimeout(200);
  for(const id of ids)assert.deepEqual(await position(node(id)),dragged[id],lang+' '+id+' must not snap back');
  await page.getByRole('button',{name:lang==='en'?'Auto Layout':'自动布局',exact:true}).click();
  await page.waitForTimeout(350);
  for(const id of ids)assert.deepEqual(await position(node(id)),initial[id],'Auto layout restores '+id);
  await page.getByRole('button',{name:lang==='en'?'Exit focus':'退出聚焦',exact:true}).click();
  assert.deepEqual(JSON.parse(await page.locator('pre.code-view').innerText()),graph,'Dragging data-flow nodes does not modify configuration graph');
  console.log('PASS '+lang+': all '+ids.length+' data-flow nodes drag in both axes, keep their positions, reset through Auto Layout and preserve graph JSON');
  await page.close();
 }
 assert.deepEqual(errors,[]);console.log('PASS no JavaScript errors');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
