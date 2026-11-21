const assert = require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_CORE);
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH,args:['--no-sandbox']});
 const page=await browser.newPage({locale:'en-US',viewport:{width:1440,height:960}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 try {
  await page.goto(process.env.STUDIO_URL || 'http://127.0.0.1:8000');
  await page.getByText('New canvas',{exact:true}).click();
  await page.locator('.tab-btn').filter({hasText:'Graph JSON'}).click();
  const before=await page.locator('.tab-btn.active').innerText();
  await page.getByRole('link',{name:'Benchmark',exact:true}).click();
  await page.getByRole('main',{name:'Benchmark page'}).waitFor();
  assert(page.url().endsWith('#/benchmark'));
  assert.equal(await page.locator('.tab-btn:visible').count(),0);
  assert.equal(await page.locator('.workspace').isVisible(),false);
  await page.getByText(/configurations · fetched/).waitFor({timeout:60000});
  await page.getByRole('link',{name:'Back to Studio',exact:true}).click();
  assert.equal(await page.locator('.tab-btn.active').innerText(),before);
  assert(await page.locator('.workspace').isVisible());
  await page.goBack();await page.getByRole('main',{name:'Benchmark page'}).waitFor();
  await page.reload();await page.getByRole('main',{name:'Benchmark page'}).waitFor();
  await page.getByText(/configurations · fetched/).waitFor({timeout:60000});
  await page.getByRole('link',{name:'Back to Studio',exact:true}).click();
  assert(await page.locator('.workspace').isVisible());assert.deepEqual(errors,[]);
  console.log('PASS: standalone URL, full page navigation, hidden execution panel, preserved bottom tab, browser history, direct refresh, return to Studio, no JS errors');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
