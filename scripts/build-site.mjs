#!/usr/bin/env node
/**
 * build-site.mjs —— 生成 docs/data/ 下的站点数据，并校验题解文档是否符合 scripts/DOC-GUIDE.md。
 *
 *   node scripts/build-site.mjs            # 校验 + 写入 docs/data/{catalog,search-index,outline}.json
 *   node scripts/build-site.mjs --check    # 只校验（含“数据是否已过期”检查），不写文件
 *
 * 零依赖：YAML 子集解析器与 Markdown 提取都在本文件内实现。
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = path.join(ROOT, 'docs');
const DATA = path.join(DOCS, 'data');
const KNOWN_FLAGS = new Set(['--check', '--no-write']);
const unknownFlags = process.argv.slice(2).filter((a) => !KNOWN_FLAGS.has(a));
if (unknownFlags.length) {
  // 以前未知参数会被静默忽略并进入写入模式：`--chek` 这种笔误不会报错、还会改数据文件。
  console.error(`未知参数：${unknownFlags.join(' ')}（可用：--check、--no-write）`);
  process.exit(2);
}
const CHECK_ONLY = process.argv.includes('--check');
const NO_WRITE = process.argv.includes('--no-write');   // 只校验不写数据（多人并行撰写时用）

const TOPIC_SLUGS = {
  'LLM 内部原理与架构': 'llm-internals',
  '推理、服务与 GPU 性能': 'inference-serving',
  'RAG 与检索': 'rag',
  'Agent 与工具调用': 'agents',
  '微调、后训练与对齐': 'finetuning',
  '评估与可观测性': 'evaluation',
  '安全、安保与负责任 AI': 'safety',
  '多模态、语音与声音 AI': 'multimodal',
  'AI 系统设计': 'system-design',
  '编程与数据结构': 'coding'
};
const REQUIRED_SECTIONS = ['一句话答案', '面试官在考什么', '原理与推导', '数值与代码验证', '常见追问'];
const TRAILING_SECTIONS = ['相关题目', '参考资料与归属'];
const LEVELS = ['入门', '进阶', '高阶'];
const DOC_TYPES = ['question', 'topic', 'company'];
const COMPANY_GROUPS = new Set(['前沿 AI 实验室', '大型科技公司的 AI 组织', 'AI 基础设施与平台公司', 'AI 原生产品公司', '前置部署与企业级 AI']);
const SKIP_DIRS = new Set(['assets', 'data']);
const COMPANY_ROOT = '公司题库';
/** 公司专属小节（不对应跨公司专题文档，因此不算「未知专题」） */
const COMPANY_SECTION_SLUGS = new Set(['ml-fundamentals', 'applied', 'behavioral']);

const errors = [];
const warnings = [];
const rel = (p) => path.relative(ROOT, p).split(path.sep).join('/');
/** 站点内路径：相对 docs/，SPA 直接 fetch 用 */
const sitePath = (p) => path.relative(DOCS, p).split(path.sep).join('/');

/* ---------------------------------------------------------------- YAML 子集 */

/** 题面比对用的归一化：去掉空白与句末标点 */
function normQ(t) {
  return String(t || '').replace(/\s+/g, '').replace(/[。？！.?!]+$/, '');
}

function unquote(v) {
  if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) return v.slice(1, -1);
  return v;
}
function scalar(v) {
  const s = v.trim();
  if (s.startsWith('[') && s.endsWith(']')) {
    const inner = s.slice(1, -1).trim();
    return inner ? inner.split(',').map((x) => unquote(x.trim())) : [];
  }
  return unquote(s);
}
function parseYamlSubset(src, file) {
  const out = {};
  let list = null;
  let item = null;
  for (const raw of src.split('\n')) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    const indent = raw.match(/^\s*/)[0].length;
    const line = raw.trim();
    if (line.startsWith('- ')) {
      if (!list) { errors.push(`${file}: 列表项出现在非列表字段下：${line}`); continue; }
      if (out.__pendingList) { out[out.__pendingList] = list; out.__pendingList = null; }
      const rest = line.slice(2).trim();
      const m = rest.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
      if (m) { item = { [m[1]]: scalar(m[2]) }; list.push(item); }
      else { item = null; list.push(scalar(rest)); }
      continue;
    }
    const m = line.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
    if (!m) { errors.push(`${file}: 无法解析的 front matter 行：${line}`); continue; }
    const [, key, value] = m;
    if (indent > 0 && item && !(key in out)) { item[key] = scalar(value); continue; }
    // 空值可能是「空列表的开始」，也可能是「漏填了内容」。这里两者都记为 null，
    // 由后续的必备字段检查报错；以前记成 []（truthy）会让 `question:` 漏填静默通过。
    if (value === '') { list = []; item = null; out[key] = null; out.__pendingList = key; }
    else { out[key] = scalar(value); list = null; item = null; }
  }
  delete out.__pendingList;
  return out;
}

/* ---------------------------------------------------------------- 工具 */

function readDoc(file) {
  const text = fs.readFileSync(file, 'utf8');
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) { errors.push(`${rel(file)}: 缺少 front matter`); return { data: {}, body: text }; }
  return { data: parseYamlSubset(m[1], rel(file)), body: text.slice(m[0].length) };
}
function walkMd(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) { if (!SKIP_DIRS.has(entry.name)) walkMd(full, acc); }
    else if (entry.name.endsWith('.md')) acc.push(full);
  }
  return acc;
}
const countMatches = (s, re) => (s.match(re) || []).length;

function toPlain(md) {
  return md
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/\$[^$\n]*\$/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/[>*`|_]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
// 先剥围栏：代码块里出现的 `## xxx` 不是文档标题，否则会误报「规范之外的二级标题」
const firstHeading = (md) => (stripFences(md).match(/^#{1,6}\s+(.+)$/m) || [, ''])[1].trim();

function validateQuestion(doc, file, seenIds) {
  const { data, body } = doc;
  const at = rel(file);
  // asked_at 以前不在必备字段里：漏写它会静默当成「没有公司考过」，题目就从「公司变体」统计里消失。
  for (const field of ['id', 'topic', 'order', 'question', 'level', 'sources', 'related', 'asked_at']) {
    // 同时覆盖 undefined 与 null：YAML 里写成 `question:` 的空键以前会解析成 []（truthy）
    // 而骗过检查，现在解析成 null，必须在这里被拦住。
    if (data[field] === undefined || data[field] === null) errors.push(`${at}: front matter 缺少字段 ${field}`);
  }
  if (data.id && !/^[a-z0-9-]+$/.test(data.id)) errors.push(`${at}: id 只能包含小写字母、数字与连字符（当前 ${data.id}）`);
  if (data.id) {
    if (seenIds.has(data.id)) errors.push(`${at}: id 重复：${data.id}`);
    seenIds.add(data.id);
  }
  if (data.order !== undefined && !Number.isInteger(Number(data.order))) errors.push(`${at}: order 必须是整数`);
  if (data.level && !LEVELS.includes(data.level)) errors.push(`${at}: level 必须是 ${LEVELS.join(' / ')} 之一（当前 ${data.level}）`);
  // related 写成标量时 `for (const rid of data.related)` 会按字符遍历，
  // 把 "llm-internals-01" 拆成一堆单字符 id 再报一堆莫名其妙的错。
  if (data.related !== undefined && data.related !== null && !Array.isArray(data.related)) {
    errors.push(`${at}: related 必须是行内列表（如 related: [a-01, b-02]），当前是 ${typeof data.related}`);
  }
  if (!Array.isArray(data.sources) || data.sources.length === 0) errors.push(`${at}: sources 至少一条`);
  else data.sources.forEach((s, i) => {
    if (!s || typeof s !== 'object' || !s.url) errors.push(`${at}: sources[${i}] 缺少 url`);
  });

  // 必须先剥围栏：正文里的示例代码经常包含 `## xxx`（如 Markdown 渲染示例、配置片段），
  // 用原始 body 抽取会把它们当成违规的二级标题。
  const headings = [...stripFences(body).matchAll(/^##\s+(.+?)\s*$/gm)].map((m) => m[1]);
  for (const need of [...REQUIRED_SECTIONS, ...TRAILING_SECTIONS]) {
    if (!headings.includes(need)) errors.push(`${at}: 缺少二级标题「${need}」`);
  }
  const hasCompany = headings.includes('公司变体');
  if (Array.isArray(data.asked_at) && data.asked_at.length > 0 && !hasCompany) {
    errors.push(`${at}: asked_at 非空，必须包含「## 公司变体」一节`);
  }
  if (data.company && hasCompany) {
    warnings.push(`${at}: 公司题不应包含「## 公司变体」一节（公司由 company 字段表达）`);
  }
  const order = headings.filter((h) => !['一句话答案', '面试官在考什么', '原理与推导', '数值与代码验证', '常见追问', '公司变体', '相关题目', '参考资料与归属'].includes(h));
  if (order.length) errors.push(`${at}: 出现了规范之外的二级标题：${order.join('、')}`);

  // 只在围栏代码块之外统计 $（代码里的 shell $、内联代码里的 $ 不参与 KaTeX 配对）
  const prose = stripFences(body).replace(/`[^`\n]*`/g, '');
  const dollars = countMatches(prose, /(?<!\\)\$/g);   // 转义美元符号（\$47）不参与配对
  if (dollars % 2 !== 0) errors.push(`${at}: 正文 $ 数量为奇数（${dollars}），KaTeX 公式可能未闭合`);
  // 围栏配对必须与 stripFences 用同一套判定：原来这里只数行首的 ```，
  // 于是 ~~~ 围栏与缩进围栏既不参与剥离也不参与配对，两边口径不一致。
  const unclosed = unclosedFence(body);
  if (unclosed) errors.push(`${at}: 代码围栏未闭合（最后一个 ${unclosed} 围栏没有配对的收尾）`);

  const words = toPlain(body).length;
  if (words < 800) warnings.push(`${at}: 正文只有 ${words} 字，可能过于简略（参考 150–300 行）`);

  // 占位符扫描：出现这些说明还没写完
  for (const bad of ['TODO', '待补', '暂略', 'PLACEHOLDER', 'XXX', '？？？', '此处省略']) {
    if (prose.includes(bad)) errors.push(`${at}: 正文含占位符「${bad}」，说明尚未写完`);
  }
  // 中文正文里的 ASCII 双引号（规范要求用「」）
  const asciiQuotes = countMatches(prose, /"[^"\n]{1,40}"/g);
  if (asciiQuotes > 0) warnings.push(`${at}: 正文出现 ${asciiQuotes} 处 ASCII 双引号，规范要求改用「」`);

  // 代码块里被写成字面量的换行转义：print("\\n...") 会打印出反斜杠 + n 而不是换行。
  // 这是本仓库反复出现的作者笔误，直接在门禁里拦住。
  const literalNewline = countMatches(body, /\\{2,}n/g);
  if (literalNewline > 0) {
    errors.push(`${at}: 代码块里出现 ${literalNewline} 处字面量 \\\\n（应为 \\n），运行时会打印出反斜杠 n`);
  }
}

function validateTopic(doc, file) {
  const { data, body } = doc;
  const at = rel(file);
  for (const field of ['id', 'title', 'order', 'summary']) {
    if (data[field] === undefined) errors.push(`${at}: front matter 缺少字段 ${field}`);
  }
  if (data.id && !/^[a-z0-9-]+$/.test(data.id)) errors.push(`${at}: id 只能包含小写字母、数字与连字符（当前 ${data.id}）`);
  if (!firstHeading(body)) errors.push(`${at}: 专题导读缺少一级标题`);
}

/* ------------------------------------------------------------ 本地链接检查 */

/** 去掉围栏代码块，避免把代码里的 foo[name](args) 当成链接 */
function stripFences(md) {
  const out = [];
  let fence = null;
  for (const line of md.split('\n')) {
    const m = line.match(/^\s*(```+|~~~+)/);
    if (m) { if (!fence) fence = m[1][0]; else if (m[1][0] === fence) fence = null; continue; }
    if (!fence) out.push(line);
  }
  return out.join('\n');
}

/** 与 stripFences 同一套围栏判定：返回未闭合的围栏字符（``` 或 ~），全部闭合时返回 null */
function unclosedFence(md) {
  let fence = null;
  for (const line of md.split('\n')) {
    const m = line.match(/^\s*(```+|~~~+)/);
    if (!m) continue;
    if (!fence) fence = m[1][0];
    else if (m[1][0] === fence) fence = null;
  }
  return fence ? fence.repeat(3) : null;
}

/** 校验文档里的相对链接（站内 .md 与指向仓库其它文件）是否真实存在 */
function checkLocalLinks(file, body) {
  // 先剥围栏再剥行内代码：正文里常拿 `[文字](路径.md)` 当反例讲链接写法，
  // 那些是代码而不是真链接，不该参与存在性校验。
  const text = stripFences(body).replace(/`[^`\n]*`/g, '');
  const baseDir = path.dirname(file);
  // 目标可以含空格（仓库里已有「LLM 内部原理….md」这类路径），
  // 但不能含换行；括号允许一层嵌套。旧正则的 [^()\s]+ 会让含空格的链接整条漏检。
  const re = /\]\(([^()\n]+(?:\([^()]*\)[^()\n]*)*)\)/g;
  let m;
  while ((m = re.exec(text))) {
    const target = m[1];
    if (/^(https?:|mailto:|#|data:)/.test(target)) continue;
    let decoded;
    try { decoded = decodeURIComponent(target.split('#')[0]); }
    catch { errors.push(`${rel(file)}: 链接里的百分号转义非法（孤立 % 或非 UTF-8 序列）：${target}`); continue; }
    if (!decoded) continue;
    const resolved = path.resolve(baseDir, decoded);
    if (!fs.existsSync(resolved)) errors.push(`${rel(file)}: 链接指向不存在的文件 ${target}`);
  }
}

/* ---------------------------------------------------------------- 读取 docs */

const mdFiles = walkMd(DOCS);
const questions = [];
const topics = [];

const companyDocs = [];
for (const file of mdFiles) {
  const doc = readDoc(file);
  const inCompanyRoot = path.relative(DOCS, file).split(path.sep)[0] === COMPANY_ROOT;
  // type 以前没有枚举校验：写错一个字母（如 `type: questions`）会静默落进默认分支，
  // 文档被当成另一种类型处理，而构建全绿。
  if (doc.data.type !== undefined && doc.data.type !== null && !DOC_TYPES.includes(doc.data.type)) {
    errors.push(`${rel(file)}: type 必须是 ${DOC_TYPES.join(' / ')} 之一（当前 ${doc.data.type}）`);
  }
  const type = doc.data.type || (path.basename(file) === 'README.md' ? (inCompanyRoot ? 'company' : 'topic') : 'question');
  if (type === 'company') companyDocs.push({ file, ...doc, _kind: 'overview' });
  else if (type === 'topic') { validateTopic(doc, file); topics.push({ file, ...doc }); }
  else if (inCompanyRoot) companyDocs.push({ file, ...doc, _kind: 'question' });
  else questions.push({ file, ...doc });
}
for (const doc of [...topics, ...questions, ...companyDocs]) checkLocalLinks(doc.file, doc.body);

const seenIds = new Set();
questions.sort((a, b) => (Number(a.data.order) || 0) - (Number(b.data.order) || 0));
for (const q of questions) validateQuestion(q, q.file, seenIds);

const companyQuestions = companyDocs.filter((d) => d._kind === 'question');
companyQuestions.sort((a, b) => (Number(a.data.order) || 0) - (Number(b.data.order) || 0));
for (const q of companyQuestions) validateQuestion(q, q.file, seenIds);
const companyNameByDir = new Map();
for (const d of companyDocs.filter((x) => x._kind === 'overview')) {
  const dn = path.basename(path.dirname(d.file));
  companyNameByDir.set(dn, d.data.name || dn);
}
for (const q of companyQuestions) {
  const dirName = path.basename(path.dirname(q.file));
  const expect = companyNameByDir.get(dirName);
  if (!q.data.company) errors.push(`${rel(q.file)}: 公司题缺少 company 字段`);
  else if (expect && q.data.company !== expect) errors.push(`${rel(q.file)}: company 字段（${q.data.company}）与公司目录 ${dirName} 的 name（${expect}）不一致`);
  else if (!expect) warnings.push(`${rel(q.file)}: 公司目录 ${dirName} 还没有导读 README.md，无法核对 company 字段`);
  if (!q.data.topic) errors.push(`${rel(q.file)}: 公司题缺少 topic 字段（用于关联十个专题之一）`);
}
for (const d of companyDocs.filter((x) => x._kind === 'overview')) {
  const dirName = path.basename(path.dirname(d.file));
  if (!d.data.id || !d.data.title) errors.push(`${rel(d.file)}: 公司导读缺少 id 或 title`);
  if (!d.data.name) errors.push(`${rel(d.file)}: 公司导读缺少 name 字段（用于与 README 公司清单对应）`);
}

const topicById = new Map();
for (const t of topics) {
  const slug = t.data.id || TOPIC_SLUGS[t.data.title] || path.basename(path.dirname(t.file));
  topicById.set(slug, {
    id: slug,
    title: t.data.title || firstHeading(t.body),
    order: Number(t.data.order) || 99,
    summary: t.data.summary || '',
    declaredTotal: Number(t.data.total) || null,
    updated: t.data.updated || null,
    dir: path.basename(path.dirname(t.file)),
    overviewFile: sitePath(t.file),
    overviewTitle: firstHeading(t.body),
    questions: []
  });
}
for (const q of questions) {
  const slug = TOPIC_SLUGS[q.data.topic] || q.data.topic;
  let topic = topicById.get(slug);
  if (!topic) {
    topic = { id: slug, title: q.data.topic, order: 99, summary: '', declaredTotal: null, updated: null, dir: path.basename(path.dirname(q.file)), overviewFile: null, overviewTitle: q.data.topic, questions: [] };
    topicById.set(slug, topic);
    warnings.push(`${rel(q.file)}: 所属专题没有导读 README.md（专题 ${q.data.topic}）`);
  }
  topic.questions.push({
    id: q.data.id,
    order: Number(q.data.order),
    title: q.data.question || firstHeading(q.body),
    titleEn: q.data.question_en || '',
    file: sitePath(q.file),
    level: q.data.level || '',
    tags: q.data.tags || [],
    askedAt: q.data.asked_at || [],
    related: q.data.related || [],
    sources: (q.data.sources || []).map((s) => ({ title: s.title || '', url: s.url || '', author: s.author || '', published: s.published || '' })),
    updated: q.data.updated || null,
    chars: toPlain(q.body).length
  });
}
for (const topic of topicById.values()) {
  topic.questions.sort((a, b) => a.order - b.order);
  const orders = topic.questions.map((q) => q.order);
  orders.forEach((o, i) => { if (o !== i + 1) errors.push(`专题 ${topic.title}: order 必须从 1 连续编号（得到 ${orders.join(',')}）`); });
}
const topicList = [...topicById.values()].sort((a, b) => a.order - b.order);
const topicIds = new Set(topicList.map((t) => t.id));
// 提前构建 outline（公司绑定与 related 校验都要用它）。
// 注意 plannedIds 不在这里算：公司题目的 id 前缀取自公司目录的 id（amazon-04），
// 而 README 抽出来的小节此刻只有小节标题（Amazon）。早算会拼出大小写不符的「Amazon-04」，
// 让真实存在的题解被误判成「指向不存在的题目」。它被挪到公司小节绑定之后。
const outline = buildOutline();
for (const q of companyQuestions) {
  const slug = TOPIC_SLUGS[q.data.topic] || q.data.topic;
  if (q.data.topic && !topicIds.has(slug) && !COMPANY_SECTION_SLUGS.has(slug)) errors.push(`${rel(q.file)}: topic 字段（${q.data.topic}）不是已知专题或公司专属小节`);
}

/* ------------------------------------------------- 公司集合（docs/公司题库/<公司名>/） */

const companyDirs = new Map();
for (const d of companyDocs) {
  const dirName = path.basename(path.dirname(d.file));
  if (!companyDirs.has(dirName)) companyDirs.set(dirName, { dir: dirName, overview: null, questions: [] });
  const c = companyDirs.get(dirName);
  if (d._kind === 'overview') c.overview = d; else c.questions.push(d);
}
const companyList = [...companyDirs.values()].map((c) => ({
  id: (c.overview && c.overview.data.id) || 'company-' + c.dir,
  name: (c.overview && c.overview.data.name) || c.dir,
  readmeName: (c.overview && c.overview.data.readme_name) || '',
  group: '',
  summary: (c.overview && c.overview.data.summary) || '',
  updated: (c.overview && c.overview.data.updated) || null,
  dir: c.dir,
  overviewFile: c.overview ? sitePath(c.overview.file) : null,
  overviewTitle: c.overview ? firstHeading(c.overview.body) : c.dir,
  declaredTotal: null,
  questions: c.questions.map((q) => ({
    id: q.data.id,
    order: Number(q.data.order),
    title: q.data.question || firstHeading(q.body),
    titleEn: q.data.question_en || '',
    file: sitePath(q.file),
    level: q.data.level || '',
    tags: q.data.tags || [],
    topic: TOPIC_SLUGS[q.data.topic] || q.data.topic || '',
    askedAt: q.data.asked_at || [],
    related: q.data.related || [],
    sources: (q.data.sources || []).map((x) => ({ title: x.title || '', url: x.url || '', author: x.author || '', published: x.published || '' })),
    updated: q.data.updated || null,
    chars: toPlain(q.body).length
  }))
}));
for (const c of companyList) if (!c.overviewFile) warnings.push(`公司 ${c.name}: 缺少导读 README.md`);
for (const c of companyList) {
  const orders = c.questions.map((q) => q.order);
  for (let i = 0; i < orders.length; i++) {
    if (orders[i] !== i + 1) { errors.push(`公司 ${c.name}: order 必须从 1 连续编号（得到 ${orders.join(',')}）`); break; }
  }
}

/* ------------------------------------------------- 从 README.zh-CN.md 抽题单 */

function buildOutline() {
  const file = path.join(ROOT, 'README.zh-CN.md');
  if (!fs.existsSync(file)) return null;
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  const outline = { generated: new Date().toISOString(), stats: {}, topics: [], companies: [] };
  let group = null, section = null, sub = null, current = null;
  let inCommon = false;
  for (const line of lines) {
    let m = line.match(/^##\s+(.+?)\s*$/);
    if (m) {
      group = m[1]; section = null; sub = null; current = null;
      inCommon = group === '跨公司高频问题';
      continue;
    }
    m = line.match(/^###\s+(.+?)\s*$/);
    if (m) {
      section = m[1]; sub = null; current = null;
      if (inCommon) {
        const id = TOPIC_SLUGS[section];
        section = { id: id || section, title: m[1], group, order: outline.topics.length + 1, questions: [] };
        outline.topics.push(section);
      } else if (group && COMPANY_GROUPS.has(group)) {
        section = { name: m[1], group, topics: [], questions: [] };
        outline.companies.push(section);
      }
      continue;
    }
    m = line.match(/^####\s+(.+?)\s*$/);
    if (m) {
      sub = { name: m[1], questions: [] };
      if (section && section.topics) section.topics.push(sub);
      continue;
    }
    if (line.startsWith('- ')) {
      current = { order: 0, question: line.slice(2).trim(), askedAt: [], sources: [] };
      const bucket = inCommon ? section && section.questions : (sub ? sub.questions : section && section.questions);
      if (bucket) { current.order = bucket.length + 1; bucket.push(current); }
      else current = null;
      continue;
    }
    if (!current) continue;
    let mm = line.match(/^\s+- 出现于：(.+)$/);
    if (mm) { current.askedAt = [...mm[1].matchAll(/\[([^\]]+)\]\([^)]*\)/g)].map((x) => x[1]); continue; }
    mm = line.match(/^\s+- 参考答案：(.+)$/);
    if (mm) { current.sources = [...mm[1].matchAll(/\[([^\]]+)\]\((https?:[^)]+)\)/g)].map((x) => ({ title: x[1], url: x[2] })); }
  }
  // 去掉没有题目的伪章节（例如文末的「许可证」小节）
  outline.companies = outline.companies.filter((c) => c.questions.length + c.topics.reduce((x, t) => x + t.questions.length, 0) > 0);
  const topicQuestions = outline.topics.reduce((s, t) => s + t.questions.length, 0);
  const companyQuestions = outline.companies.reduce((s, c) => s + c.questions.length + c.topics.reduce((x, t) => x + t.questions.length, 0), 0);
  outline.stats = {
    topics: outline.topics.length,
    topicQuestions,
    companies: outline.companies.length,
    companyQuestions,
    total: topicQuestions + companyQuestions,
    fetched: questions.length
  };
  return outline;
}

/* ---------------------------------------------------------------- 搜索索引 */

function buildSearchIndex(catalog, outline) {
  const byTopicOrder = new Map();
  if (outline) for (const t of outline.topics) for (const q of t.questions) byTopicOrder.set(`${t.id}#${q.order}`, q);
  const items = [];
  for (const topic of catalog.topics) {
    if (topic.overviewFile) {
      const body = readDoc(path.join(DOCS, topic.overviewFile)).body;
      items.push({ kind: 'topic', id: topic.id, route: `#/topic/${topic.id}`, title: topic.title, topic: topic.title, text: toPlain(body).slice(0, 1500) });
    }
    for (const q of topic.questions) {
      const body = readDoc(path.join(DOCS, q.file)).body;
      items.push({ kind: 'question', id: q.id, route: `#/q/${q.id}`, title: q.title, topic: topic.title, askedAt: q.askedAt, text: toPlain(body).slice(0, 2500) });
    }
  }
  for (const c of catalog.companies || []) {
    if (c.overviewFile) {
      const body = readDoc(path.join(DOCS, c.overviewFile)).body;
      items.push({ kind: 'company', id: c.id, route: `#/company/${encodeURIComponent(c.name)}`, title: c.name, topic: c.name, text: toPlain(body).slice(0, 1200) });
    }
    for (const q of c.questions) {
      const body = readDoc(path.join(DOCS, q.file)).body;
      items.push({ kind: 'question', id: q.id, route: `#/q/${q.id}`, title: q.title, topic: c.name, askedAt: q.askedAt, text: toPlain(body).slice(0, 2500) });
    }
  }
  return { generated: new Date().toISOString(), items };
}

/* ---------------------------------------------------------------- 写出 */

const catalog = {
  generated: new Date().toISOString(),
  stats: { topics: topicList.length, questions: questions.length, companyQuestions: companyList.reduce((n, c) => n + c.questions.length, 0), companies: companyList.length, planned: null },
  topics: topicList,
  companies: companyList
};
if (outline) {
  catalog.stats.planned = outline.stats.total;
  // outline 的结构性缩水必须是 error。以前 README 的标题改一个字就会让整家公司从题单里
  // 静默消失、所有相关校验一并跳过，而构建仍然全绿。
  if (outline.stats.companies !== catalog.companies.length) {
    errors.push(`README.zh-CN.md 里带题目的公司小节有 ${outline.stats.companies} 个，`
      + `而公司目录有 ${catalog.companies.length} 个 —— 题单可能因为标题改名而缩水`);
  }
  if (outline.stats.topics !== catalog.topics.length) {
    errors.push(`README.zh-CN.md 的跨公司专题有 ${outline.stats.topics} 个，`
      + `而已撰写专题有 ${catalog.topics.length} 个`);
  }
  // 与上一次写出的 outline 比较：题数骤降说明抽题单出了问题（而不是真的删了题）。
  const prevOutlinePath = path.join(DATA, 'outline.json');
  if (fs.existsSync(prevOutlinePath)) {
    try {
      const prev = JSON.parse(fs.readFileSync(prevOutlinePath, 'utf8'));
      const drop = (prev.stats && prev.stats.total || 0) - outline.stats.total;
      if (drop > 0) {
        errors.push(`README.zh-CN.md 抽出的题单比上次少了 ${drop} 题`
          + `（上次 ${prev.stats.total}，本次 ${outline.stats.total}）—— 若不是有意删题，请检查标题与列表格式`);
      }
    } catch { /* 旧文件损坏则跳过比较 */ }
  }
  // 公司身份靠名字匹配，而 README.zh-CN.md 的小节标题常带限定语（如「Amazon（AWS）」「智谱 AI（GLM）」），
  // 与公司 README 的 name 字段不一定逐字相同。匹配顺序：
  //   1) name 逐字
  //   2) readme_name 显式别名（公司 README 的 front matter 里可选声明）
  //   3) 去掉空白与括号限定语后的归一化名
  // 三者都失配时报 error —— 否则该公司的 group 与 declaredTotal 会静默为空、题面校验被整体跳过，
  // 站点上表现为「已撰写 0 / N」。
  const normName = (s) => String(s || '')
    .replace(/[（(][^）)]*[）)]/g, '')
    .replace(/\s+/g, '')
    .toLowerCase();
  const byName = new Map(outline.companies.map((c) => [c.name, c]));
  const byNorm = new Map();
  for (const c of outline.companies) {
    const k = normName(c.name);
    if (!byNorm.has(k)) byNorm.set(k, c);
  }
  const boundSections = new Set();
  for (const c of catalog.companies) {
    let o = byName.get(c.name);
    if (!o && c.readmeName) o = byName.get(c.readmeName);
    if (!o) o = byNorm.get(normName(c.name));
    if (!o) {
      errors.push(`公司 ${c.name}（目录 ${c.dir}）: 无法与 README.zh-CN.md 的公司小节绑定，`
        + `请在该公司 README 的 front matter 里加一行 readme_name: <小节标题原文>`);
      continue;
    }
    if (boundSections.has(o.name)) {
      errors.push(`公司 ${c.name}: 绑定到「${o.name}」时发现该小节已被另一家公司占用`);
      continue;
    }
    boundSections.add(o.name);
    // 统一 canonical key：把 outline 小节的名字改成公司目录的 name。
    // 路由（app.js 的 viewCompany）用 outline 的名字做查找键，而公司列表用 catalog 的 name 生成链接；
    // 两者不一致时详情页会取不到数据、渲染成空白。原始小节标题保留在 readmeName 里备查。
    o.readmeName = o.name;
    o.name = c.name;
    // 同时把公司目录的 id 带过来。plannedIds（本文件上方）用 `${c.id || c.name}-NN` 拼题目 id，
    // outline 的小节没有 id 字段，于是会拼出「Amazon-04」这种大写 id；而真实题解文件的 id 是
    // 「amazon-04」。大小写不一致会让真实存在的题目被误判成「指向不存在的题目」。
    o.id = c.id;
    c.readmeSection = o.readmeName;
    c.group = o.group;
    c.declaredTotal = o.questions.length + o.topics.reduce((s2, t) => s2 + t.questions.length, 0);
    const rows = new Map();
    const add = (name, q) => rows.set(normQ(q.question), { name, order: q.order });
    for (const q of o.questions) add('', q);
    for (const t of o.topics) for (const q of t.questions) add(t.name, q);
    for (const q of c.questions) {
      const key = normQ(q.title);
      const hit = rows.get(key);
      if (!hit) errors.push(`${c.name}/${q.file}: 题面与 README.zh-CN.md 的公司题单不匹配：${q.title.slice(0, 40)}…`);
      else { q.topicName = hit.name; q.topicOrder = hit.order; }
    }
    for (const [key, v] of rows) {
      if (!c.questions.some((q) => normQ(q.title) === key)) c.pending = (c.pending || 0) + 1;
    }
  }
  // 反向检查：README.zh-CN.md 里有题目的小节，必须有公司目录绑定它。
  // 否则「README 标题改一个字」就会让整家公司从 outline 里静默消失而 CI 仍然全绿。
  // 判据用 readmeName：绑定成功的小节一定被赋过值（哪怕原名与 canonical 名相同）。
  for (const o of outline.companies) {
    const n = o.questions.length + o.topics.reduce((s2, t) => s2 + t.questions.length, 0);
    if (n > 0 && !o.readmeName) {
      errors.push(`README.zh-CN.md 的公司小节「${o.name}」（${n} 题）没有任何公司目录与之绑定`);
    }
  }
}

/* ------------------------------------ related 与正文双链校验（须在公司绑定之后） */

// 放在这里的原因：plannedIds 需要公司小节的 id 前缀，而该前缀由上面的绑定阶段从公司目录贴上来。
// 公司题目的序号必须是「全公司连续序号」，不能是子小节内序号：题解文件名里的编号来自
// company-index.mjs 的全局计数器（顶层 questions 之后按子小节顺序一路累加），例如 Anthropic
// 的 37 篇是 anthropic-01..37。若每个子小节各从 1 数起，只会产出 anthropic-01..12 这一段，
// 于是 13..37 之间任何「已规划未撰写」的题目都会被误报成「id 不存在」。
const plannedIds = outline
  ? new Set([
    ...outline.topics.flatMap((t) => t.questions.map((q) => `${t.id}-${String(q.order).padStart(2, '0')}`)),
    ...outline.companies.flatMap((c) => {
      const ids = [];
      let n = 0;
      const add = () => { n += 1; ids.push(`${c.id || c.name}-${String(n).padStart(2, '0')}`); };
      for (const q of c.questions) add();
      for (const t of c.topics) for (const q of t.questions) add();
      return ids;
    })
  ])
  : null;
const knownIds = new Set([...questions, ...companyQuestions].map((x) => x.data.id));
for (const q of [...questions, ...companyQuestions]) {
  for (const rid of q.data.related || []) {
    if (topicIds.has(rid)) continue;   // 允许指向整个专题
    if (!knownIds.has(rid)) {
      // 关键区分：id 在全站题单里（outline）但还没写 -> warning；
      // id 压根不存在 -> error。否则「写错了 id」会被「尚未撰写」这句话长期掩盖。
      if (plannedIds && plannedIds.has(rid)) warnings.push(`${rel(q.file)}: related 指向尚未撰写的题目 ${rid}`);
      else errors.push(`${rel(q.file)}: related 里的 id 不存在：${rid}`);
    }
  }
  // 正文双链此前完全不校验，死链会在站点上永久渲染成「待撰写」标签而构建全绿。
  // 必须先剥代码围栏与行内代码：Python 里的 boxes[[i]]、torch.tensor([[-100, …]]) 都长得像双链。
  const proseBody = stripFences(String(q.body || '')).replace(/`[^`\n]*`/g, '');
  for (const m of proseBody.matchAll(/\[\[([^\]]+)\]\]/g)) {
    const rid = m[1].trim();
    if (knownIds.has(rid) || topicIds.has(rid)) continue;
    if (plannedIds && plannedIds.has(rid)) warnings.push(`${rel(q.file)}: 正文双链指向尚未撰写的题目 [[${rid}]]`);
    else errors.push(`${rel(q.file)}: 正文双链的 id 不存在：[[${rid}]]`);
  }
}
const searchIndex = buildSearchIndex(catalog, outline);

const outputs = [
  [path.join(DATA, 'catalog.json'), JSON.stringify(catalog, null, 2) + '\n'],
  [path.join(DATA, 'search-index.json'), JSON.stringify(searchIndex, null, 2) + '\n']
];
if (outline) outputs.push([path.join(DATA, 'outline.json'), JSON.stringify(outline, null, 2) + '\n']);

if (!CHECK_ONLY && !NO_WRITE) {
  fs.mkdirSync(DATA, { recursive: true });
  for (const [file, content] of outputs) fs.writeFileSync(file, content);
} else if (!NO_WRITE) {
  for (const [file, content] of outputs) {
    const current = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : null;
    const strip = (s) => (s === null ? null : s.replace(/"generated": "[^"]+",\n/, ''));
    if (strip(current) !== strip(content)) errors.push(`${rel(file)} 已过期，请运行 node scripts/build-site.mjs`);
  }
}

/* ---------------------------------------------------------------- 报告 */

for (const w of warnings) console.warn('warn  ' + w);
for (const e of errors) console.error('error ' + e);
const s = catalog.stats;
console.log(
  `${CHECK_ONLY ? '[check] ' : (NO_WRITE ? '[no-write] ' : '')}专题 ${s.topics} 个 / 专题题解 ${s.questions} 道 / 公司 ${s.companies} 家 ${s.companyQuestions} 道` +
  (outline ? ` / 题库总量 ${outline.stats.total}（专题题 ${outline.stats.topicQuestions} + 公司题 ${outline.stats.companyQuestions}）` : '')
);
console.log(`errors: ${errors.length}, warnings: ${warnings.length}`);
process.exit(errors.length ? 1 : 0);
