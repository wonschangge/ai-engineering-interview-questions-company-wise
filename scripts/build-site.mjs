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
const CHECK_ONLY = process.argv.includes('--check');

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
const COMPANY_GROUPS = new Set(['前沿 AI 实验室', '大型科技公司的 AI 组织', 'AI 基础设施与平台公司', 'AI 原生产品公司', '前置部署与企业级 AI']);
const SKIP_DIRS = new Set(['assets', 'data', '公司题库']);

const errors = [];
const warnings = [];
const rel = (p) => path.relative(ROOT, p).split(path.sep).join('/');
/** 站点内路径：相对 docs/，SPA 直接 fetch 用 */
const sitePath = (p) => path.relative(DOCS, p).split(path.sep).join('/');

/* ---------------------------------------------------------------- YAML 子集 */

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
    if (value === '') { list = []; item = null; out[key] = list; }
    else { out[key] = scalar(value); list = null; item = null; }
  }
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
const firstHeading = (md) => (md.match(/^#{1,6}\s+(.+)$/m) || [, ''])[1].trim();

function validateQuestion(doc, file, seenIds) {
  const { data, body } = doc;
  const at = rel(file);
  for (const field of ['id', 'topic', 'order', 'question', 'level', 'sources', 'related']) {
    if (data[field] === undefined) errors.push(`${at}: front matter 缺少字段 ${field}`);
  }
  if (data.id && !/^[a-z0-9-]+$/.test(data.id)) errors.push(`${at}: id 只能包含小写字母、数字与连字符（当前 ${data.id}）`);
  if (data.id) {
    if (seenIds.has(data.id)) errors.push(`${at}: id 重复：${data.id}`);
    seenIds.add(data.id);
  }
  if (data.order !== undefined && !Number.isInteger(Number(data.order))) errors.push(`${at}: order 必须是整数`);
  if (data.level && !LEVELS.includes(data.level)) errors.push(`${at}: level 必须是 ${LEVELS.join(' / ')} 之一（当前 ${data.level}）`);
  if (!Array.isArray(data.sources) || data.sources.length === 0) errors.push(`${at}: sources 至少一条`);
  else data.sources.forEach((s, i) => {
    if (!s || typeof s !== 'object' || !s.url) errors.push(`${at}: sources[${i}] 缺少 url`);
  });

  const headings = [...body.matchAll(/^##\s+(.+?)\s*$/gm)].map((m) => m[1]);
  for (const need of [...REQUIRED_SECTIONS, ...TRAILING_SECTIONS]) {
    if (!headings.includes(need)) errors.push(`${at}: 缺少二级标题「${need}」`);
  }
  const hasCompany = headings.includes('公司变体');
  if (Array.isArray(data.asked_at) && data.asked_at.length > 0 && !hasCompany) {
    errors.push(`${at}: asked_at 非空，必须包含「## 公司变体」一节`);
  }
  const order = headings.filter((h) => !['一句话答案', '面试官在考什么', '原理与推导', '数值与代码验证', '常见追问', '公司变体', '相关题目', '参考资料与归属'].includes(h));
  if (order.length) errors.push(`${at}: 出现了规范之外的二级标题：${order.join('、')}`);

  const dollars = countMatches(body, /\$/g);
  if (dollars % 2 !== 0) errors.push(`${at}: $ 数量为奇数（${dollars}），KaTeX 公式可能未闭合`);
  const fences = countMatches(body, /^```/gm);
  if (fences % 2 !== 0) errors.push(`${at}: 代码围栏数量为奇数（${fences}）`);

  const words = toPlain(body).length;
  if (words < 800) warnings.push(`${at}: 正文只有 ${words} 字，可能过于简略（参考 150–300 行）`);
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

/* ---------------------------------------------------------------- 读取 docs */

const mdFiles = walkMd(DOCS);
const questions = [];
const topics = [];

for (const file of mdFiles) {
  const doc = readDoc(file);
  const type = doc.data.type || (path.basename(file) === 'README.md' ? 'topic' : 'question');
  if (type === 'topic') { validateTopic(doc, file); topics.push({ file, ...doc }); }
  else { questions.push({ file, ...doc }); }
}
const seenIds = new Set();
questions.sort((a, b) => (Number(a.data.order) || 0) - (Number(b.data.order) || 0));
for (const q of questions) validateQuestion(q, q.file, seenIds);

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
for (const q of questions) {
  for (const rid of q.data.related || []) {
    if (!questions.some((x) => x.data.id === rid)) warnings.push(`${rel(q.file)}: related 指向尚未撰写的题目 ${rid}`);
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
  return { generated: new Date().toISOString(), items };
}

/* ---------------------------------------------------------------- 写出 */

const catalog = {
  generated: new Date().toISOString(),
  stats: { topics: topicList.length, questions: questions.length, planned: null },
  topics: topicList
};
const outline = buildOutline();
if (outline) catalog.stats.planned = outline.stats.total;
const searchIndex = buildSearchIndex(catalog, outline);

const outputs = [
  [path.join(DATA, 'catalog.json'), JSON.stringify(catalog, null, 2) + '\n'],
  [path.join(DATA, 'search-index.json'), JSON.stringify(searchIndex, null, 2) + '\n']
];
if (outline) outputs.push([path.join(DATA, 'outline.json'), JSON.stringify(outline, null, 2) + '\n']);

if (!CHECK_ONLY) {
  fs.mkdirSync(DATA, { recursive: true });
  for (const [file, content] of outputs) fs.writeFileSync(file, content);
} else {
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
  `${CHECK_ONLY ? '[check] ' : ''}专题 ${s.topics} 个 / 已撰写题解 ${s.questions} 道` +
  (outline ? ` / 题库总量 ${outline.stats.total}（专题题 ${outline.stats.topicQuestions} + 公司题 ${outline.stats.companyQuestions}）` : '')
);
console.log(`errors: ${errors.length}, warnings: ${warnings.length}`);
process.exit(errors.length ? 1 : 0);
