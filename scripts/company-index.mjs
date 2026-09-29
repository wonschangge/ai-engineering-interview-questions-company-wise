#!/usr/bin/env node
/**
 * 抽取公司题库，生成每家公司的作业输入文件。
 *
 *   node scripts/company-index.mjs            # 生成 scripts/batches/company-*.json 并打印清单
 *   node scripts/company-index.mjs --list     # 只打印清单
 *   node scripts/company-index.mjs <slug>     # 只生成某一家
 *
 * 数据来源：docs/data/outline.json（中文结构与来源，来自 README.zh-CN.md）
 *          + README.md（英文题面，按 公司 → 小节 → 题序 对齐）
 *
 * 输出：scripts/batches/company-<slug>.json
 * { company, slug, dir, group, total, topics: [...], questions: [
 *     { order, topic, topicName, question, questionEn, sources: [{title,url,author,published}] } ] }
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = path.join(ROOT, 'scripts', 'batches');

/** 公司目录名 / id 前缀（短、稳定、可读） */
const SLUGS = {
  'Anthropic': 'anthropic',
  'OpenAI': 'openai',
  'Google DeepMind 与 Google AI': 'google-deepmind',
  'Meta（超级智能实验室、FAIR、Llama）': 'meta',
  'xAI': 'xai',
  'Mistral AI': 'mistral',
  'Cohere': 'cohere',
  'DeepSeek': 'deepseek',
  'Moonshot AI（Kimi）': 'moonshot',
  '智谱 AI（GLM）': 'zhipu',
  '阿里巴巴（Qwen）': 'alibaba',
  'Sarvam AI': 'sarvam',
  'Microsoft': 'microsoft',
  'Amazon（AWS）': 'amazon',
  'Apple': 'apple',
  'NVIDIA': 'nvidia',
  'Tesla': 'tesla',
  '面向消费者的规模化 ML 公司（Uber、Netflix、LinkedIn、Airbnb、Pinterest、Spotify）': 'consumer-ml',
  'Databricks': 'databricks',
  'Groq': 'groq',
  'Together AI': 'together',
  'Hugging Face': 'huggingface',
  'Scale AI': 'scale-ai',
  'Perplexity': 'perplexity',
  'Cursor（Anysphere）': 'cursor',
  'Cognition（Devin、Windsurf）': 'cognition',
  'Sierra': 'sierra',
  'Harvey': 'harvey',
  'Glean': 'glean',
  'Character.AI': 'character-ai',
  'ElevenLabs': 'elevenlabs',
  'Abridge': 'abridge',
  'Figure AI': 'figure-ai',
  'Waymo': 'waymo',
  'Palantir': 'palantir'
};

/** 公司小节名 → 专题 slug（10 个跨公司专题 + 3 个公司专属小节） */
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
  '编程与数据结构': 'coding',
  '机器学习与深度学习基础': 'ml-fundamentals',
  '应用与前置部署场景': 'applied',
  '行为与文化': 'behavioral'
};

/** 解析 README.md 的英文公司题（顺序与中文一致） */
function parseEnglishReadme(file) {
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  const companies = [];
  const GROUP_NAMES = new Set(['Frontier AI Labs', 'Big Tech AI Organizations', 'AI Infrastructure and Platform Companies', 'AI-Native Product Companies', 'Forward-Deployed and Enterprise AI']);
  let group = null, company = null, sub = null, inCommon = false;
  for (const line of lines) {
    let m = line.match(/^##\s+(.+?)\s*$/);
    if (m) {
      group = m[1]; company = null; sub = null;
      inCommon = /common questions/i.test(group);
      continue;
    }
    m = line.match(/^###\s+(.+?)\s*$/);
    if (m) {
      sub = null;
      company = (!inCommon && group && GROUP_NAMES.has(group)) ? { name: m[1], topics: [] } : null;
      if (company) companies.push(company);
      continue;
    }
    m = line.match(/^####\s+(.+?)\s*$/);
    if (m && company) { sub = { name: m[1], questions: [] }; company.topics.push(sub); continue; }
    m = line.match(/^-\s+(.+?)\s*$/);
    if (m && company && sub) {
      const text = m[1].trim();
      if (text.startsWith('[') && text.includes('](')) continue;   // 来源链接行
      if (/^https?:\/\//.test(text)) continue;
      sub.questions.push(text);
    }
  }
  return companies;
}

/** 中文公司名 → README.md 里的英文公司名 */
const EN_COMPANY = {
  'Google DeepMind 与 Google AI': 'Google DeepMind and Google AI',
  'Meta（超级智能实验室、FAIR、Llama）': 'Meta (Superintelligence Labs, FAIR, Llama)',
  'Moonshot AI（Kimi）': 'Moonshot AI (Kimi)',
  '智谱 AI（GLM）': 'Zhipu AI (GLM)',
  '阿里巴巴（Qwen）': 'Alibaba (Qwen)',
  'Amazon（AWS）': 'Amazon (AWS)',
  '面向消费者的规模化 ML 公司（Uber、Netflix、LinkedIn、Airbnb、Pinterest、Spotify）': 'Consumer-Scale ML Companies (Uber, Netflix, LinkedIn, Airbnb, Pinterest, Spotify)',
  'Cursor（Anysphere）': 'Cursor (Anysphere)',
  'Cognition（Devin、Windsurf）': 'Cognition (Devin, Windsurf)'
};

/** 英文小节名 → 中文小节名（按小节对齐，比整体扁平更稳） */
const EN_TOPIC = {
  'Coding and Data Structures': '编程与数据结构',
  'LLM Internals and Architecture': 'LLM 内部原理与架构',
  'Inference, Serving and GPU Performance': '推理、服务与 GPU 性能',
  'RAG and Retrieval': 'RAG 与检索',
  'Agents and Tool Use': 'Agent 与工具调用',
  'Fine-Tuning, Post-Training and Alignment': '微调、后训练与对齐',
  'Evaluation and Observability': '评估与可观测性',
  'Safety, Security and Responsible AI': '安全、安保与负责任 AI',
  'Multimodal, Speech and Voice AI': '多模态、语音与声音 AI',
  'AI System Design': 'AI 系统设计',
  'ML and DL Fundamentals': '机器学习与深度学习基础',
  'Applied and Forward-Deployed Scenarios': '应用与前置部署场景',
  'Behavioral and Culture': '行为与文化'
};

const outline = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs', 'data', 'outline.json'), 'utf8'));
const en = parseEnglishReadme(path.join(ROOT, 'README.md'));

const out = [];
let enMissing = 0;
for (const c of outline.companies) {
  const slug = SLUGS[c.name] || c.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const enName = EN_COMPANY[c.name] || c.name;
  const ce = en.find((x) => x.name === enName) || null;
  // 英文按「小节名（译回中文）+ 题序」建立索引
  const enByZh = new Map();
  if (ce) for (const t of ce.topics) {
    const zhName = EN_TOPIC[t.name] || t.name;
    enByZh.set(zhName, t.questions);
  }

  const questions = [];
  let order = 0;
  const sections = [
    ...(c.questions.length ? [{ name: '', questions: c.questions }] : []),
    ...c.topics.filter((t) => t.questions.length > 0)
  ];
  for (const t of sections) {
    for (let i = 0; i < t.questions.length; i++) {
      const q = t.questions[i];
      order += 1;
      const arr = (t.name ? enByZh.get(t.name) : enByZh.get('')) || [];
      const qEn = arr[i] || '';
      if (!qEn) enMissing += 1;
      questions.push({
        order,
        topic: t.name ? (TOPIC_SLUGS[t.name] || t.name) : '',
        topicName: t.name,
        question: q.question,
        questionEn: qEn || '',
        sources: (q.sources || []).map((s) => ({ title: s.title || '', url: s.url || '', author: s.author || '', published: s.published || '' }))
      });
    }
  }
  out.push({ name: c.name, group: c.group, slug, total: questions.length, questions });
}

const args = process.argv.slice(2);
const listOnly = args.includes('--list');
const only = args.find((a) => !a.startsWith('--'));
if (!listOnly) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const c of out) {
    if (only && c.slug !== only) continue;
    fs.writeFileSync(path.join(OUT_DIR, `company-${c.slug}.json`), JSON.stringify({
      company: c.name, slug: c.slug, dir: c.slug, group: c.group, idPrefix: c.slug,
      total: c.total,
      topics: [...new Set(c.questions.map((q) => q.topic).filter(Boolean))],
      questions: c.questions
    }, null, 2) + '\n');
  }
}

let total = 0;
for (const c of out) {
  total += c.total;
  const noTopic = c.questions.filter((q) => !q.topic).length;
  console.log(`${c.slug.padEnd(16)} ${String(c.total).padStart(3)} 题  ${c.group.padEnd(24)} ${c.name}` + (noTopic ? `  ⚠${noTopic} 题未归入小节` : ''));
}
console.log(`\n合计 ${out.length} 家 / ${total} 题` + (enMissing ? `（${enMissing} 题缺英文题面）` : '（中英题面全部对齐）'));
console.log(`已写入 ${path.relative(ROOT, OUT_DIR)}/company-*.json`);
if (enMissing) process.exitCode = 0;
