// Run studio/scripts/audit_component_defaults.py first; then set PLAYWRIGHT_CORE, CHROME_PATH and optionally STUDIO_URL.
const fs=require('node:fs'),assert=require('node:assert/strict');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_CORE || path.resolve(__dirname,'../frontend/node_modules/playwright'));
const URL=(process.env.STUDIO_URL || 'http://127.0.0.1:8000').replace(/\/$/,'');
const audited=JSON.parse(fs.readFileSync(process.env.DEFAULTS_AUDIT_REPORT || '/tmp/studio-audited-defaults.json','utf8'));
const untouched=new Set(['data_config','embed_config','lm','lang','token_source','model_id','tokenizer_id','revision','tokenizer_revision']);
const strip=column=>column.replace(/@[^ ]+/g,'');
function currentParams(record){
 const params={};
 for(const p of record.spec.parameters){
  const value=p.default;
  if(untouched.has(p.name)||p.name.startsWith('external_')){if(value!=null)params[p.name]=value;continue}
  if(p.name==='item_inputs'){
   params[p.name]=record.parameters[p.name].rows.map((row,i)=>({...row,truncate:i===0?17:row.truncate})).reverse();continue;
  }
  if(p.name==='hidden_size'){params[p.name]=1024;continue}
  if(typeof value==='boolean')params[p.name]=!value;
  else if(typeof value==='number')params[p.name]=value+1;
  else if(Array.isArray(value))params[p.name]=value.length?value.map(v=>typeof v==='number'?v+1:v):[17];
  else if(p.choices?.length)params[p.name]=p.choices.find(v=>v!==value)||value;
  else if(value===null)params[p.name]=p.type==='list'?[17,19]:p.type==='integer'||p.type==='number'?777:'ReLU';
  else params[p.name]=value;
 }
 return params;
}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || undefined,args:['--no-sandbox']});
 async function checkLanguage(lang){
  const page=await browser.newPage({locale:'en-US',viewport:{width:1680,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  const dict=JSON.parse(fs.readFileSync(path.resolve(__dirname,'../frontend/src/i18n/locales/'+lang+'.json'),'utf8'));
  try{
   await page.goto(URL,{waitUntil:'networkidle'});
   await page.getByRole('button',{name:/Start from Scratch/}).click();
   await page.getByRole('combobox',{name:'Language / 语言'}).selectOption(lang);
   const modal=page.locator('.modal-backdrop');
   const open=async id=>{
    const exit=page.getByRole('button',{name:lang==='en'?'Exit focus':'退出聚焦',exact:true});
    if(await exit.count())await exit.click();
    await page.locator('aside').first().locator('input').first().fill(id);
    await page.locator('.lib-item[data-component-id="'+id+'"]').click();await modal.waitFor();
   };
   const close=()=>modal.getByRole('button',{name:lang==='en'?'Close':'关闭',exact:true}).click();
   async function snapshot(record){
    const values={};
    const rows=modal.locator('[data-param-name]');
    for(let i=0;i<await rows.count();i++){
     const row=rows.nth(i),name=await row.getAttribute('data-param-name'),expected=record.parameters[name];
     assert.ok(expected,record.id+' unknown property '+name);
     if(expected.kind==='dataset'){
      if(expected.rows.length)await row.locator('[data-item-column]').first().waitFor();
      const actual=await row.locator('[data-item-column]').evaluateAll(items=>items.map(el=>({column:el.dataset.itemColumn,truncate:el.dataset.itemTruncate})));
      const wanted=expected.rows.map(row=>({column:strip(row.column),truncate:row.truncate==null?'':String(row.truncate)}));
      assert.deepEqual(actual,wanted,record.id+' default dataset fields');
      values[name]=actual;
     }else{
      const actual=await row.locator('[data-default-value]').textContent();
      if(expected.kind==='value')assert.deepEqual(JSON.parse(actual),expected.value,record.id+'.'+name+' real default');
      else assert.equal(actual,dict['defaults.'+expected.kind],record.id+'.'+name+' automatic/unset marker');
      values[name]=actual;
     }
    }
    return values;
   }
   let checks=0;
   for(const record of audited){
    await open(record.id);const original=await snapshot(record);await close();
    await page.locator('.tab-btn').filter({hasText:'Graph JSON'}).click();
    const graph={graph_id:'default_audit',name:'Modified current project '+record.id,task:record.spec.tasks[0]||'matching',nodes:[{id:'audit_current',component:record.id,params:currentParams(record),position:{x:0,y:0}}],edges:[],train:{hidden_size:1024,batch_size:129,epochs:2}};
    const panel=page.locator('textarea[placeholder="{ ... }"]');await panel.fill(JSON.stringify(graph));
    const response=page.waitForResponse(r=>r.url().endsWith('/api/graph/import')&&r.request().method()==='POST');
    await page.getByRole('button',{name:lang==='en'?'Import Graph JSON':'导入 Graph JSON',exact:true}).click();
    const result=await response;assert.equal(result.status(),200,record.id+' current graph import');
    await open(record.id);const changed=await snapshot(record);assert.deepEqual(changed,original,record.id+' must not depend on current project');
    checks+=Object.keys(changed).length;await close();
    console.log('PASS '+lang+' '+record.id+' ('+Object.keys(changed).length+' defaults)');
   }
   assert.deepEqual(errors,[]);console.log('PASS '+lang+' ALL '+audited.length+' components, '+checks+' visible defaults, no project-value leakage');
  }finally{await page.close()}
 }
 try{await Promise.all(['en','zh-CN'].map(checkLanguage));console.log('PASS complete bilingual default audit')}
 finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
