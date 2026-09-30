#!/usr/bin/env node
/**
 * test-site.mjs —— docs/ 站点的端到端冒烟测试（jsdom）。
 *
 *   npm i -D jsdom && node scripts/test-site.mjs
 *   # 或者已有 jsdom 时直接指定模块路径：
 *   JSDOM_MODULE=/path/to/node_modules/jsdom/lib/api.js node scripts/test-site.mjs
 *
 * 用 jsdom 加载 docs/index.html，桩掉 fetch 直接读磁盘，然后驱动 SPA：
 * 首页 / 侧栏 / 专题页 / 题目页 / 搜索 / 公司视图 / 主题 / 已读进度 / 404，
 * 并断言 KaTeX、代码高亮、表格、双链与相对链接改写都生效。
 * 期望值全部从 docs/data/*.json 推导，新增专题不需要改这个文件。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = path.join(REPO, 'docs');

let JSDOM, VirtualConsole;
try {
  ({ JSDOM, VirtualConsole } = await import(process.env.JSDOM_MODULE || 'jsdom'));
} catch (err) {
  console.error('需要 jsdom：请在仓库根目录运行 `npm ci`（依赖已在 package.json 里声明），'
    + '或用 JSDOM_MODULE=<jsdom/lib/api.js 路径> 指定已有的安装。');
  console.error('原始错误：' + err.message);
  process.exit(2);
}


const vc = new VirtualConsole();
const consoleErrors = [];
vc.on('jsdomError', (e) => { if (!/Not implemented/.test(e.message)) consoleErrors.push('jsdomError: ' + e.message); });
vc.on('error', (...a) => consoleErrors.push('console.error: ' + a.join(' ')));

const html = fs.readFileSync(path.join(DOCS, 'index.html'), 'utf8');
const dom = new JSDOM(html, { runScripts: 'outside-only', url: 'http://localhost/', virtualConsole: vc, pretendToBeVisual: true });
const { window } = dom;
const { document } = window;

// --- 桩：fetch 读磁盘 ---
window.fetch = async (url) => {
  const clean = decodeURIComponent(String(url).split('?')[0].replace(/^\.\//, ''));
  const file = path.join(DOCS, clean);
  if (!fs.existsSync(file)) return { ok: false, status: 404, json: async () => ({}), text: async () => '' };
  const body = fs.readFileSync(file, 'utf8');
  return { ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body };
};
window.scrollTo = () => {};
window.Element.prototype.scrollIntoView = () => {};
window.navigator.clipboard = { writeText: async () => {} };

for (const f of ['vendor/marked.min.js', 'vendor/katex.min.js', 'vendor/highlight.min.js']) {
  window.eval(fs.readFileSync(path.join(DOCS, 'assets/site', f), 'utf8'));
}
window.eval(fs.readFileSync(path.join(DOCS, 'assets/site/app.js'), 'utf8'));
document.dispatchEvent(new window.Event('DOMContentLoaded'));

const wait = (ms = 120) => new Promise((r) => setTimeout(r, ms));
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const text = (s) => ($(s) ? $(s).textContent : '');

const catalog = JSON.parse(fs.readFileSync(path.join(DOCS, 'data/catalog.json'), 'utf8'));
const outline = JSON.parse(fs.readFileSync(path.join(DOCS, 'data/outline.json'), 'utf8'));
const TOTAL_TOPICS = catalog.topics.length;
const TOTAL_QUESTIONS = catalog.topics.reduce((n, t) => n + t.questions.length, 0);
// 已撰写题解 = 专题题 + 公司题。之前首页 badge 只数了专题题（119），
// 而公司题解早已全部写完，导致站点长期低估自己的产出；这条常量让口径只有一处定义。
const TOTAL_COMPANY_QUESTIONS = catalog.companies.reduce((n, c) => n + c.questions.length, 0);
const TOTAL_WRITTEN = TOTAL_QUESTIONS + TOTAL_COMPANY_QUESTIONS;
const FIRST = catalog.topics[0];
const FIRST_Q = FIRST.questions[0];
const results = [];
const check = (name, cond, extra = '') => { results.push({ name, ok: Boolean(cond), extra }); };

// 每个视图只应有 1 个 H1（页头那个）。这条断言是为一个长期存在的事故加的：
// stripFrontMatter 里「剥掉正文 H1」的正则匹配不到 `---` 之后的空行，全仓库 643/643 篇
// 都没被剥掉，于是题目页/专题页/公司页都在页头 H1 之后又渲染出一个逐字重复的 H1，
// 而当时所有断言都只查 `.page-head h1`，对多余的 H1 完全不设防。
function checkOneH1(label) {
  const hs = $$('h1');
  check(label + '只有一个 H1', hs.length === 1,
    hs.length === 1 ? '' : hs.length + ' 个：' + hs.map((h) => h.textContent.slice(0, 24)).join(' | '));
}

async function nav(hash) {
  window.location.hash = hash;
  await wait(260);
}

await wait(400);

// 1) 首页
check('首页渲染标题', text('.hero h1').includes('专题题解'));
checkOneH1('首页');
check('首页统计包含已撰写题解 ' + TOTAL_WRITTEN + ' 道（专题 ' + TOTAL_QUESTIONS + ' + 公司 ' + TOTAL_COMPANY_QUESTIONS + '）', text('.badges').includes('已撰写题解 ' + TOTAL_WRITTEN + ' 道'), text('.badges'));
check('首页显示题库总量 ' + outline.stats.total, text('.badges').includes(String(outline.stats.total)), text('.badges'));
// 文案不能再说「题解按专题逐步补齐」——公司题解已经写完，旧文案会误导访客
check('首页不再声称公司题解尚未撰写', !document.body.textContent.includes('题解按专题逐步补齐') && !document.body.textContent.includes('题解在计划中'), '');
check('首页有 ' + TOTAL_TOPICS + ' 个专题卡片', $$('.topic-card').length === TOTAL_TOPICS, String($$('.topic-card').length));
const TOTAL_GROUPS = new Set(catalog.companies.map((c) => c.group).filter(Boolean)).size;
check('首页含公司分组（' + TOTAL_GROUPS + ' 组）', $$('.company-card').length === TOTAL_GROUPS, String($$('.company-card').length));
check('首页含学习路线（5 阶段）', $$('.roadmap li').length === 5, String($$('.roadmap li').length));
check('首页含资料入口（3 个）', document.body.textContent.includes('资料入口') && document.body.textContent.includes('中文题库'), '');

// 2) 侧栏
check('侧栏含专题名', text('#nav').includes(FIRST.title));
const navQs = $$('#nav .nav-q').filter((a) => a.getAttribute('href').startsWith('#/q/'));
check('侧栏列出 ' + TOTAL_QUESTIONS + ' 道题解', navQs.length === TOTAL_QUESTIONS, String(navQs.length));
check('侧栏公司题库有 35 家', text('#nav').includes('35 家'), '');

// 3) 专题页
await nav('#/topic/' + FIRST.id);
const rows = $$('.q-row');
check('专题页列出 ' + FIRST.questions.length + ' 道题', rows.length === FIRST.questions.length, String(rows.length));
check('本专题 ' + FIRST.questions.length + ' 道全部标记已上线', $$('.q-state.is-done').length === FIRST.questions.length, String($$('.q-state.is-done').length));
checkOneH1('专题页');
check('专题页进度文案 ' + FIRST.questions.length + ' / ' + FIRST.questions.length, document.body.textContent.includes('已撰写 ' + FIRST.questions.length + ' / ' + FIRST.questions.length + ' 题'), text('.page-head .muted'));
await wait(200);
check('专题导读已渲染', $('#topic-overview-body h2') !== null, text('#topic-overview-body').slice(0, 60));
check('导读里的 .md 链接转成站内路由', $$('#topic-overview-body a[href="#/q/llm-internals-01"]').length >= 1, String($$('#topic-overview-body a[href^="#/q/"]').length));
check('导读里没有残留的相对 .md 链接', $$('#topic-overview-body a[href$=".md"]').filter((a) => !/^https?:/.test(a.getAttribute('href'))).length === 0, JSON.stringify($$('#topic-overview-body a[href$=".md"]').map((a) => a.getAttribute('href'))));
check('指向仓库 README 的链接转成 GitHub 地址', $$('#topic-overview-body a[href*="github.com"][href$="README.md"]').length >= 1, '');

// 4) 题目页 01
await nav('#/q/' + FIRST_Q.id);
check('题目页标题正确', text('.page-head h1') === FIRST_Q.title, text('.page-head h1').slice(0, 40));
checkOneH1('题目页');
check('题解正文渲染出小节', $$('#doc-body h2').length >= 7, String($$('#doc-body h2').length));
check('KaTeX 公式已渲染', $$('#doc-body .katex').length > 10, String($$('#doc-body .katex').length));
check('display 公式存在', $$('#doc-body .katex-display').length > 0, String($$('#doc-body .katex-display').length));
check('表格已渲染', $$('#doc-body table').length >= 4, String($$('#doc-body table').length));
check('代码高亮生效', $$('#doc-body pre code .hljs-keyword, #doc-body pre code .hljs-string, #doc-body pre code .hljs-built_in').length > 0, '');
check('右侧目录生成', $$('.toc a').length >= 7, String($$('.toc a').length));
check('双链转成站内路由', $$('#doc-body a[href^="#/q/"]').length >= 1, String($$('#doc-body a[href^="#/q/"]').length));
check('第一题 meta 行渲染', text('.meta-row').length > 0, text('.meta-row').slice(0, 60));
check('标签 chips 渲染', $$('.meta-row .chip').length >= 1, String($$('.meta-row .chip').length));
check('上一题/下一题导航存在', FIRST.questions.length < 2 || ($('.prev-next .next') !== null && $('.prev-next .next').getAttribute('href') === '#/q/' + FIRST.questions[1].id), '');

// 5) 题目页 02/03（子代理产出）
const Q2 = FIRST.questions[1], Q3 = FIRST.questions[2];
await nav('#/q/' + Q2.id);
check('02 正文渲染', $$('#doc-body h2').length >= 7, String($$('#doc-body h2').length));
check('02 KaTeX 渲染', $$('#doc-body .katex').length > 10, String($$('#doc-body .katex').length));
check('第二题 chips 渲染', $$('.meta-row .chip').length >= 1, String($$('.meta-row .chip').length));
check('双链处理正确（存在即渲染为链接或待撰写标签）', $$('#doc-body .qref, #doc-body .qref-pending').length >= 1, String($$('#doc-body .qref, #doc-body .qref-pending').length));
check('已撰写双链仍是站内链接', $$('#doc-body a.qref[href^="#/q/"]').length >= 1, '');

await nav('#/q/' + Q3.id);
check('第三题正文渲染', $$('#doc-body h2').length >= 7, String($$('#doc-body h2').length));
check('第三题含代码块', $$('#doc-body pre').length >= 2, String($$('#doc-body pre').length));
check('第三题公司变体一节存在', Array.from($$('#doc-body h2')).some((h) => h.textContent.includes('公司变体')), '');

// 5.5) 专题级双链：正文里写 [[rag]] 应渲染为专题页路由
const bpe = catalog.topics.flatMap((t) => t.questions).find((q) => q.id === 'llm-internals-06');
if (bpe) {
  await nav('#/q/' + bpe.id);
  check('专题级双链 [[rag]] 渲染为专题路由', $$('#doc-body a.qref[href="#/topic/rag"]').length >= 1, String($$('#doc-body a.qref[href^="#/topic/"]').length));
}

// 6) 搜索
await nav('#/search?q=KV cache');
await wait(300);
check('搜索面板打开', !$('#search-modal').hidden);
check('搜索有结果', $$('.search-hit').length >= 2, String($$('.search-hit').length));
check('搜索结果含高亮', $$('.search-results mark').length >= 1, String($$('.search-results mark').length));

// 7) 公司视图
await nav('#/companies');
check('公司页渲染 ' + outline.stats.companies + ' 家', $$('.company-card').length === outline.stats.companies, String($$('.company-card').length));
checkOneH1('公司列表页');
await nav('#/company/Anthropic');
check('公司详情页渲染', text('.page-head h1') === 'Anthropic', text('.page-head h1'));
checkOneH1('公司详情页');
// 旧链接兼容：outline 小节名已被规范成公司目录的 name，但历史链接与 README 小节标题
// 用的是带限定语的原名。这类路由以前直接落到「没有找到这家公司」。
await nav('#/company/' + encodeURIComponent('Amazon（AWS）'));
check('旧式公司路由仍能打开', text('.page-head h1') === 'Amazon', text('.page-head h1').slice(0, 40));
checkOneH1('旧式公司路由页');

// 7b) 遍历全部公司：每家的进度分母必须等于它的题数、且题面不能全部落到「待撰写」。
// 这条断言是为一个真实事故加的：Amazon 与智谱的公司 README name 与 README.zh-CN.md 的小节标题
// 不一致，导致匹配失败后 group/declaredTotal 静默为空、页面显示「已撰写 0 / N」，
// 而当时只测 Anthropic 的断言恰好绕过了这两个坏点。
{
  const offenders = [];
  for (const c of catalog.companies) {
    await nav('#/company/' + encodeURIComponent(c.name));
    const h1 = text('.page-head h1');
    if (h1 !== c.name) { offenders.push(`${c.name}: 标题渲染为「${h1}」`); continue; }
    const body = text('#content');
    const want = `${c.questions.length} / ${c.questions.length}`;
    // 公司页的进度写作「已撰写 N / M」；两者都必须等于该公司的题数
    const m = body.match(/已撰写\s*(\d+)\s*\/\s*(\d+)/);
    if (!m) { offenders.push(`${c.name}: 页面上找不到进度分子/分母`); continue; }
    if (`${m[1]} / ${m[2]}` !== want) offenders.push(`${c.name}: 进度为 ${m[1]} / ${m[2]}，应为 ${want}`);
    if (c.questions.length && !$$('.q-row').length) offenders.push(`${c.name}: 题面列表为空`);
    // 公司导读必须真的渲染出来。上一版断言只查标题/进度/题列表，恰好绕过了导读框，
    // 于是「fetchRaw is not defined」让 35 家导读全部失败却一直是绿的。
    if (c.overviewFile) {
      const ok = await waitFor(() => {
        const b = $('#company-overview-body');
        return b && !b.textContent.includes('正在加载') && !b.textContent.includes('加载失败');
      });
      const b = $('#company-overview-body');
      if (!ok) offenders.push(`${c.name}: 导读未渲染（${b ? b.textContent.slice(0, 40) : '无容器'}）`);
      else if ($$('#company-overview-body h2').length === 0) offenders.push(`${c.name}: 导读没有渲染出二级标题`);
    }
  }
  check('全部 ' + catalog.companies.length + ' 家公司的进度、题面与导读都正确',
    offenders.length === 0, offenders.slice(0, 6).join('；'));
}

// 8) 主题切换 / 本地进度
await nav('#/q/' + Q3.id);
$('#theme-toggle').dispatchEvent(new window.Event('click'));
check('主题切换到 light（且只翻转一次，验证监听器未重复绑定）', document.documentElement.dataset.theme === 'light', document.documentElement.dataset.theme);
const readBtn = $('[data-action="toggle-read"]');
readBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
await wait(200);
check('标记已读写入 localStorage', JSON.parse(window.localStorage.getItem('aieiq-read') || '[]').includes(Q3.id), window.localStorage.getItem('aieiq-read'));

// 9) 404
await nav('#/nope');
check('未知路由显示 404', text('#content').includes('404'), '');

// 12) 全量渲染检查：每篇题解渲染出的 ## 标题数必须与 markdown 源一致
//（能抓住「围栏写错导致整段被吞进代码块」「正文没加载出来」这类问题）
async function waitFor(fn, timeout = 1500) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    try { if (fn()) return true; } catch (e) { /* 继续等 */ }
    await wait(60);
  }
  return false;
}
const allQuestions = [...catalog.topics.flatMap((t) => t.questions), ...((catalog.companies || []).flatMap((c) => c.questions))];
const badRender = [];
for (const q of allQuestions) {
  const src = fs.readFileSync(path.join(DOCS, q.file), 'utf8');
  const expected = (src.match(/^## /gm) || []).length;
  await nav('#/q/' + q.id);
  await waitFor(() => $$('#doc-body h2').length > 0);
  const got = $$('#doc-body h2').length;
  if (got !== expected) badRender.push(q.id + ' ' + got + '/' + expected);
}
check('全部 ' + allQuestions.length + ' 篇题解渲染出的二级标题数与源文件一致', badRender.length === 0, badRender.slice(0, 6).join(' | '));

// --- 汇总 ---
const failed = results.filter((r) => !r.ok);
for (const r of results) console.log(`${r.ok ? 'PASS' : 'FAIL'}  ${r.name}${r.ok ? '' : '   -> ' + r.extra}`);
if (consoleErrors.length) { console.log('--- page errors ---'); consoleErrors.slice(0, 10).forEach((e) => console.log(e)); }
console.log(`\n${results.length - failed.length}/${results.length} passed, ${failed.length} failed`);
process.exit(failed.length || consoleErrors.length ? 1 : 0);
