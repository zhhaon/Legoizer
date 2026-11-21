// Optional browser regression: set STUDIO_URL, PLAYWRIGHT_CORE and CHROME_PATH.
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_CORE || path.resolve(__dirname,'../frontend/node_modules/playwright'));
const URL=(process.env.STUDIO_URL || 'http://127.0.0.1:8000').replace(/\/$/,'');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || undefined,args:['--no-sandbox']});
 const errors=[];
 try{
 for(const lang of ['en','zh-CN']){
  const page=await browser.newPage({locale:'en-US',viewport:{width:1680,height:1050}});
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(URL,{waitUntil:'networkidle'});
  await page.getByRole('button',{name:/Start from Scratch/}).click();
  await page.getByRole('combobox',{name:'Language / 语言'}).selectOption(lang);
  const open=async id=>{
   await page.locator('aside').first().locator('input').first().fill(id);
   await page.locator('.lib-item[data-component-id="'+id+'"]').click();
   await page.locator('.modal-backdrop').waitFor();
  };
  const modal=page.locator('.modal-backdrop'),right=page.locator('aside').last();
  const keys=locator=>locator.locator('[data-property-parameter]').evaluateAll(elements=>elements.map(el=>el.dataset.propertyParameter));
  const catalog=await (await page.request.get(URL+'/api/components')).json();
  const datasets=catalog.components.filter(component=>component.role==='dataset').map(component=>component.id);
  assert.ok(datasets.length>0,'Dataset catalog is populated');
  for(const id of [...datasets,'item.cnn','user.gru','predictor.dot','training.matching_task','item.externalhf','dataset.xmind']){
   await open(id);
   await modal.locator('button.tool-btn.primary').click();
   await modal.waitFor({state:'detached'});
   await page.waitForFunction(id=>[...document.querySelectorAll('aside')].at(-1)?.textContent.includes(id),id);
   const sidebarKeys=(await keys(right)).sort();
   await open(id);
   const dialogKeys=(await keys(modal)).sort();
   assert.deepEqual(dialogKeys,sidebarKeys,lang+' '+id+' must match actual sidebar DOM');
   if(id.startsWith('dataset.')) {
    assert.ok(dialogKeys.includes('item_inputs'),'Dataset item inputs included');
    const label=lang==='en'?'Dataset item inputs':'数据集物品输入';
    assert.ok((await modal.locator('[data-param-name="item_inputs"]').innerText()).includes(label));
    assert.ok(await right.locator('[data-property-parameter="item_inputs"]').count());
   }
   if(id==='item.externalhf')assert.ok(dialogKeys.includes('item_hidden_size'),'Visible read-only property retained');
   await modal.getByRole('button',{name:lang==='en'?'Close':'关闭',exact:true}).click();
   console.log('PASS '+lang+' '+id+': modal keys equal actual sidebar keys ('+dialogKeys.length+')');
  }
  await page.close();
 }
 assert.deepEqual(errors,[]);console.log('PASS no JavaScript errors');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
