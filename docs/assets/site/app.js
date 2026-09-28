/* AI 工程面试题库 · 专题题解 SPA
 * 数据：data/catalog.json（题解元数据）、data/outline.json（README 题库全量清单）、data/search-index.json（搜索）
 * 内容：docs/<专题>/<文件>.md 运行时拉取渲染，支持 KaTeX 公式与 [[id]] 双链
 */
(function () {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const TOKEN_START = '\uE000';
  const TOKEN_END = '\uE001';

  const state = {
    catalog: null,
    outline: null,
    searchIndex: null,
    topics: [],
    topicById: new Map(),
    questionById: new Map(),
    companyBySection: [],
    read: new Set(),
    filter: '',
    theme: 'dark'
  };

  /* ------------------------------------------------------------------ 存储 */

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); }
      catch (e) { return fallback; }
    },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* 忽略隐私模式 */ } }
  };

  /* ------------------------------------------------------------------ 数据 */

  async function loadJSON(url) {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error('加载失败 ' + url + '（HTTP ' + res.status + '）');
    return res.json();
  }

  async function loadMarkdown(relPath) {
    const res = await fetch(encodeURI(relPath), { cache: 'no-cache' });
    if (!res.ok) throw new Error('加载失败 ' + relPath + '（HTTP ' + res.status + '）');
    return stripFrontMatter(await res.text());
  }

  function stripFrontMatter(md) {
    return md
      .replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '')
      .replace(/^#\s+.+\r?\n/, '');
  }

  /* -------------------------------------------------------- Markdown 渲染 */

  function splitByFence(md) {
    const lines = md.split('\n');
    const segs = [];
    let buf = [];
    let fence = null;
    for (const line of lines) {
      const m = line.match(/^\s*(```+|~~~+)/);
      if (m && !fence) {
        if (buf.length) segs.push({ fenced: false, text: buf.join('\n') });
        buf = [line];
        fence = m[1][0];
        continue;
      }
      if (m && fence && m[1][0] === fence) {
        buf.push(line);
        segs.push({ fenced: true, text: buf.join('\n') });
        buf = [];
        fence = null;
        continue;
      }
      buf.push(line);
    }
    if (buf.length) segs.push({ fenced: Boolean(fence), text: buf.join('\n') });
    return segs;
  }

  function renderMarkdown(md) {
    const codes = [];
    const maths = [];
    const text = splitByFence(md)
      .map((seg) => {
        if (seg.fenced) return seg.text;
        let s = seg.text.replace(/`[^`\n]*`/g, (m) => {
          codes.push(m.slice(1, -1));
          return TOKEN_START + 'C' + (codes.length - 1) + TOKEN_END;
        });
        s = s.replace(/\$\$([\s\S]+?)\$\$/g, (m, tex) => {
          maths.push({ tex: tex, display: true });
          return TOKEN_START + 'M' + (maths.length - 1) + TOKEN_END;
        });
        s = s.replace(/\$([^$\n]+?)\$/g, (m, tex) => {
          maths.push({ tex: tex, display: false });
          return TOKEN_START + 'M' + (maths.length - 1) + TOKEN_END;
        });
        return s;
      })
      .join('\n');

    const withLinks = text.replace(/\[\[([a-z0-9-]+)\]\]/g, (m, id) => {
      const q = state.questionById.get(id);
      return '[' + (q ? q.title : id) + '](wikilink:' + id + ')';
    });

    let html = marked.parse(withLinks, { gfm: true, breaks: false, headerIds: false, mangle: false });

    html = html.replace(new RegExp(TOKEN_START + 'C(\\d+)' + TOKEN_END, 'g'), (m, i) => '<code>' + esc(codes[Number(i)]) + '</code>');
    html = html.replace(new RegExp(TOKEN_START + 'M(\\d+)' + TOKEN_END, 'g'), (m, i) => {
      const item = maths[Number(i)];
      try {
        return katex.renderToString(item.tex.trim(), { displayMode: item.display, throwOnError: false, output: 'html' });
      } catch (err) {
        return '<code>' + esc(item.tex) + '</code>';
      }
    });
    return html;
  }

  function enhance(container) {
    $$('a[href^="wikilink:"]', container).forEach((a) => {
      const id = a.getAttribute('href').slice('wikilink:'.length);
      const q = state.questionById.get(id);
      if (q) {
        a.setAttribute('href', '#/q/' + id);
        a.className = 'qref';
      } else {
        const span = document.createElement('span');
        span.className = 'qref qref-pending';
        span.textContent = a.textContent + '（待撰写）';
        a.replaceWith(span);
      }
    });
    $$('pre code', container).forEach((el) => {
      try { window.hljs.highlightElement(el); } catch (e) { /* 高亮失败不影响阅读 */ }
    });
    const heads = $$('h2, h3', container);
    heads.forEach((h, i) => { h.id = 's' + (i + 1); });
    return heads;
  }

  function buildToc(heads) {
    if (heads.length < 2) return '';
    const items = heads
      .map((h) => '<li class="toc-' + h.tagName.toLowerCase() + '"><a href="#' + h.id + '" data-toc="' + h.id + '">' + esc(h.textContent) + '</a></li>')
      .join('');
    return '<nav class="toc"><div class="toc-title">本页目录</div><ul>' + items + '</ul></nav>';
  }

  /* ------------------------------------------------------------------ 路由 */

  function parseRoute() {
    const raw = location.hash.replace(/^#/, '') || '/';
    const qIndex = raw.indexOf('?');
    const path = qIndex >= 0 ? raw.slice(0, qIndex) : raw;
    const query = new URLSearchParams(qIndex >= 0 ? raw.slice(qIndex + 1) : '');
    const seg = path.split('/').filter(Boolean).map(decodeURIComponent);
    return { seg: seg, query: query };
  }

  function go(hash) {
    if (location.hash === hash) render();
    else location.hash = hash;
  }

  /* ------------------------------------------------------------- 侧边导航 */

  function renderNav() {
    const nav = $('#nav');
    const filter = state.filter.trim().toLowerCase();
    const blocks = [];

    blocks.push('<div class="nav-group"><div class="nav-group-title">专题题解</div>');
    for (const topic of state.topics) {
      const qs = topic.questions.filter((q) => !filter || q.title.toLowerCase().includes(filter) || topic.title.toLowerCase().includes(filter));
      if (!qs.length && filter) continue;
      const total = topic.declaredTotal || (state.outlineTopic(topic.id) || {}).total || qs.length;
      blocks.push(
        '<div class="nav-topic">' +
        '<a class="nav-topic-link" href="#/topic/' + topic.id + '">' + esc(topic.title) +
        '<span class="nav-count">' + topic.questions.length + '/' + total + '</span></a>' +
        '<ul class="nav-questions">' +
        qs.map((q) => {
          const read = state.read.has(q.id) ? ' is-read' : '';
          return '<li><a class="nav-q' + read + '" href="#/q/' + q.id + '">' + esc(q.title) + '</a></li>';
        }).join('') +
        '</ul></div>'
      );
    }
    blocks.push('</div>');

    const companies = state.outline ? state.outline.companies : [];
    blocks.push(
      '<div class="nav-group"><div class="nav-group-title">公司题库 <span class="nav-count">' +
      companies.length + ' 家 · 待撰写</span></div>' +
      '<ul class="nav-questions">' +
      companies.slice(0, 40).map((c) => {
        const n = c.questions.length + c.topics.reduce((s, t) => s + t.questions.length, 0);
        return '<li><a class="nav-q nav-q-pending" href="#/company/' + encodeURIComponent(c.name) + '">' + esc(c.name) + '<span class="nav-count">' + n + '</span></a></li>';
      }).join('') +
      '</ul></div>'
    );

    nav.innerHTML = blocks.join('');
  }

  /* ------------------------------------------------------------ 首页视图 */

  const ROADMAP = [
    ['第一阶段：基础扎实', 'Transformer、attention、RoPE、KV cache、prefill/decode、top-k / top-p / temperature。'],
    ['第二阶段：工程落地', 'RAG、向量检索、重排、prompt 与 context 管理、并发与限流、服务性能优化。'],
    ['第三阶段：Agent 与系统设计', 'ReAct、function calling、MCP、agent loop、tool use、系统设计与权限设计。'],
    ['第四阶段：后训练与评测', 'RLHF、DPO、GRPO、评测集构建、LLM-as-a-judge、hallucination 检测。'],
    ['第五阶段：专项准备', '按目标公司刷题，按专题回顾，把高频题整理成自己的答题模板。']
  ];

  function companyGroups() {
    if (!state.outline) return [];
    const groups = new Map();
    for (const c of state.outline.companies) {
      if (!groups.has(c.group)) groups.set(c.group, []);
      groups.get(c.group).push(c);
    }
    return Array.from(groups.entries()).map(([group, companies]) => ({
      group,
      count: companies.reduce((s, c) => s + c.questions.length + c.topics.reduce((x, t) => x + t.questions.length, 0), 0),
      names: companies.map((c) => c.name)
    }));
  }

  function viewHome() {
    const stats = state.catalog.stats;
    const ostats = state.outline ? state.outline.stats : { total: stats.planned || 0, topics: 0, topicQuestions: 0, companies: 0, companyQuestions: 0 };
    const cards = state.topics.map((t) => {
      const total = t.declaredTotal || t.questions.length;
      return '<a class="card topic-card" href="#/topic/' + t.id + '">' +
        '<div class="card-head"><h3>' + esc(t.title) + '</h3><span class="pill">' + t.questions.length + ' / ' + total + '</span></div>' +
        '<p>' + esc(t.summary || '') + '</p></a>';
    }).join('');

    const pending = state.outline
      ? state.outline.topics.filter((t) => !state.topicById.has(t.id))
        .map((t) => '<li><span class="dot"></span>' + esc(t.title) + '<span class="nav-count">' + t.questions.length + ' 题</span></li>').join('')
      : '';

    const groups = companyGroups().map((g) =>
      '<a class="card company-card" href="#/companies">' +
      '<div class="card-head"><h3>' + esc(g.group) + '</h3><span class="pill">' + g.count + ' 题</span></div>' +
      '<p class="muted">' + g.names.map(esc).join(' · ') + '</p></a>').join('');

    const roadmap = ROADMAP.map(([title, desc]) => '<li><strong>' + esc(title) + '</strong> — ' + esc(desc) + '</li>').join('');

    return '' +
      '<section class="hero">' +
      '<div class="eyebrow">AI Engineering Interview Kit</div>' +
      '<h1>AI 工程面试题 · 专题题解</h1>' +
      '<p class="subtitle">按专题整理的中文题解：先讲结论，再给推导、数值与代码验证，最后是面试官真正会追问的东西。' +
      '题面与公司分布来自本仓库的 <a href="https://github.com/wonschangge/ai-engineering-interview-questions-company-wise/blob/main/README.zh-CN.md" target="_blank" rel="noopener">README.zh-CN.md</a>。</p>' +
      '<div class="badges">' +
      '<span class="badge">已撰写题解 ' + stats.questions + ' 道</span>' +
      '<span class="badge">已上线专题 ' + stats.topics + ' 个</span>' +
      '<span class="badge">题库总量 ' + ostats.total + ' 题</span>' +
      '<span class="badge">' + ostats.companies + ' 家公司</span>' +
      '</div>' +
      '<div class="cta">' +
      '<a class="btn btn-primary" href="#/topic/' + (state.topics[0] ? state.topics[0].id : '') + '">开始刷题</a>' +
      '<button class="btn" data-action="search">搜索题目</button>' +
      '<a class="btn" href="#/companies">按公司浏览</a>' +
      '</div>' +
      '</section>' +

      '<section class="panel"><h2>专题题解</h2><div class="grid">' + cards + '</div>' +
      (pending ? '<h3 class="sub-head">规划中的专题</h3><ul class="pending-list">' + pending + '</ul>' : '') +
      '<p class="muted">每个专题的题解按「一句话答案 → 推导 → 数值与代码 → 常见追问」的固定结构撰写，文末标注参考来源与作者。</p></section>' +

      '<section class="panel"><h2>你能在这里准备什么</h2>' +
      '<div class="grid grid-3">' +
      '<div class="card"><h3>核心主题</h3><p>LLM 内部原理与架构、推理与服务性能、RAG、Agent 与工具调用、微调与后训练、评估与可观测性、安全与负责任 AI、多模态与语音、AI 系统设计。</p></div>' +
      '<div class="card"><h3>目标岗位</h3><p>AI Engineer、Gen AI / LLM Engineer、Agentic AI Engineer、ML / Applied Research Engineer、AI Platform / MLOps / LLMOps、Forward Deployed Engineer。</p></div>' +
      '<div class="card"><h3>知识属性</h3><p>不是背题清单，而是由真实面试问题构成的知识地图。真正要练的是原理、设计、trade-off 与工程落地能力。</p></div>' +
      '</div></section>' +

      (groups ? '<section class="panel"><h2>公司分组</h2><div class="grid">' + groups + '</div>' +
        '<p class="muted">共 ' + ostats.companies + ' 家公司的 ' + ostats.companyQuestions + ' 道公司专属题，当前先提供题面清单，题解按专题逐步补齐。</p></section>' : '') +

      '<section class="panel"><h2>推荐学习路线</h2><ul class="roadmap">' + roadmap + '</ul></section>' +

      '<section class="panel"><h2>怎么用这个站</h2>' +
      '<div class="grid grid-3">' +
      '<div class="card"><h3>1. 先刷跨公司高频题</h3><p>同一个知识点会被多家公司反复问。先把专题题解过一遍，建立可复述的答案骨架。</p></div>' +
      '<div class="card"><h3>2. 再按公司定位</h3><p>查目标公司的章节，看它偏工程实现还是偏原理推导，把准备重点对齐过去。</p></div>' +
      '<div class="card"><h3>3. 用追问自测</h3><p>每篇题解的「常见追问」就是二面、三面的方向；能不看稿回答，才算过关。</p></div>' +
      '</div></section>' +

      '<section class="panel"><h2>资料入口</h2>' +
      '<div class="grid grid-3">' +
      '<a class="card" href="https://github.com/wonschangge/ai-engineering-interview-questions-company-wise/blob/main/README.zh-CN.md" target="_blank" rel="noopener"><h3>中文题库</h3><p class="muted">README.zh-CN.md · 全量 ' + ostats.total + ' 道题</p></a>' +
      '<a class="card" href="https://github.com/wonschangge/ai-engineering-interview-questions-company-wise/blob/main/README.md" target="_blank" rel="noopener"><h3>英文原文</h3><p class="muted">README.md</p></a>' +
      '<a class="card" href="https://github.com/wonschangge/ai-engineering-interview-questions-company-wise" target="_blank" rel="noopener"><h3>GitHub 仓库</h3><p class="muted">wonschangge/ai-engineering-interview-questions-company-wise</p></a>' +
      '</div>' +
      '<p class="muted">题解基于公开技术文章理解后重写，每篇文末都列出参考来源；发现错误或想补充，欢迎提 issue。</p></section>';
  }

  /* ------------------------------------------------------------ 专题视图 */

  function outlineTopic(id) {
    if (!state.outline) return null;
    return state.outline.topics.find((t) => t.id === id) || null;
  }
  state.outlineTopic = outlineTopic;

  function viewTopic(id) {
    const topic = state.topicById.get(id);
    if (!topic) return viewMissing('没有找到这个专题');
    const outline = outlineTopic(id);
    const written = new Map(topic.questions.map((q) => [q.order, q]));
    const rows = (outline ? outline.questions : topic.questions).map((item) => {
      const order = item.order;
      const doc = written.get(order);
      const asked = (item.askedAt && item.askedAt.length) ? item.askedAt : (doc ? doc.askedAt : []);
      const askedHtml = asked && asked.length
        ? '<div class="chip-row">' + asked.map((c) => '<span class="chip">' + esc(c) + '</span>').join('') + '</div>'
        : '';
      if (doc) {
        return '<li class="q-row"><a href="#/q/' + doc.id + '">' +
          '<span class="q-order">' + String(order).padStart(2, '0') + '</span>' +
          '<span class="q-body"><span class="q-title">' + esc(doc.title) + '</span>' + askedHtml + '</span>' +
          '<span class="q-state is-done">已上线</span></a></li>';
      }
      return '<li class="q-row is-pending"><span class="q-order">' + String(order).padStart(2, '0') + '</span>' +
        '<span class="q-body"><span class="q-title">' + esc(item.question) + '</span>' + askedHtml + '</span>' +
        '<span class="q-state">待撰写</span></li>';
    }).join('');

    const total = topic.declaredTotal || (outline ? outline.questions.length : topic.questions.length);
    const pct = total ? Math.round((topic.questions.length / total) * 100) : 0;

    return '' +
      '<nav class="crumbs"><a href="#/">首页</a><span>/</span><span>专题</span></nav>' +
      '<header class="page-head">' +
      '<h1>' + esc(topic.title) + '</h1>' +
      '<p class="lede">' + esc(topic.summary || '') + '</p>' +
      '<div class="progress"><div class="progress-bar" style="width:' + pct + '%"></div></div>' +
      '<p class="muted">已撰写 ' + topic.questions.length + ' / ' + total + ' 题（' + pct + '%）</p>' +
      '</header>' +
      '<section class="panel"><h2>本专题题目</h2><ul class="q-list">' + rows + '</ul></section>' +
      '<section class="panel" id="topic-overview"><div class="markdown" id="topic-overview-body"><p class="muted">正在加载专题导读…</p></div></section>';
  }

  /* ------------------------------------------------------------ 题目视图 */

  function viewQuestion(id, section) {
    const q = state.questionById.get(id);
    if (!q) return viewMissing('这道题的题解还没有撰写');
    const topic = state.topicById.get(q.topicId);
    const siblings = topic ? topic.questions : [];
    const idx = siblings.findIndex((x) => x.id === id);
    const prev = idx > 0 ? siblings[idx - 1] : null;
    const next = idx >= 0 && idx < siblings.length - 1 ? siblings[idx + 1] : null;
    const asked = (q.askedAt || []).map((c) => '<span class="chip">' + esc(c) + '</span>').join('');
    const tags = (q.tags || []).map((t) => '<span class="chip chip-ghost">#' + esc(t) + '</span>').join('');
    const read = state.read.has(id);

    return '' +
      '<nav class="crumbs"><a href="#/">首页</a><span>/</span>' +
      (topic ? '<a href="#/topic/' + topic.id + '">' + esc(topic.title) + '</a><span>/</span>' : '') +
      '<span>第 ' + q.order + ' 题</span></nav>' +
      '<header class="page-head">' +
      '<h1>' + esc(q.title) + '</h1>' +
      (q.titleEn ? '<p class="lede en">' + esc(q.titleEn) + '</p>' : '') +
      '<div class="meta-row">' +
      (q.level ? '<span class="pill">' + esc(q.level) + '</span>' : '') +
      '<span class="pill pill-ghost">更新 ' + esc(q.updated || '-') + '</span>' +
      (asked ? '<span class="meta-label">出现于</span>' + asked : '') +
      (tags ? '<span class="meta-label">标签</span>' + tags : '') +
      '</div>' +
      '<div class="cta">' +
      '<button class="btn btn-small" data-action="toggle-read" data-id="' + id + '">' + (read ? '✓ 已读' : '标记已读') + '</button>' +
      '<button class="btn btn-small" data-action="copy-link">复制链接</button>' +
      '</div>' +
      '</header>' +
      '<div class="doc-layout">' +
      '<article class="markdown" id="doc-body"><p class="muted">正在加载题解…</p></article>' +
      '<aside class="toc-side" id="toc-side"></aside>' +
      '</div>' +
      '<nav class="prev-next">' +
      (prev ? '<a class="prev" href="#/q/' + prev.id + '"><span>上一题</span>' + esc(prev.title) + '</a>' : '<span></span>') +
      (next ? '<a class="next" href="#/q/' + next.id + '"><span>下一题</span>' + esc(next.title) + '</a>' : '<span></span>') +
      '</nav>' +
      (section ? '' : '');
  }

  async function fillTopicOverview(topic) {
    const box = $('#topic-overview-body');
    if (!box || !topic.overviewFile) return;
    try {
      const md = await loadMarkdown(topic.overviewFile);
      box.innerHTML = renderMarkdown(md);
      enhance(box);
    } catch (err) {
      box.innerHTML = '<p class="error">' + esc(err.message) + '</p>';
    }
  }

  async function fillQuestion(q, section) {
    const box = $('#doc-body');
    const toc = $('#toc-side');
    if (!box) return;
    try {
      const md = await loadMarkdown(q.file);
      box.innerHTML = renderMarkdown(md);
      const heads = enhance(box);
      if (toc) toc.innerHTML = buildToc(heads);
      if (section) {
        const target = $('#' + section, box);
        if (target) target.scrollIntoView({ block: 'start' });
      }
    } catch (err) {
      box.innerHTML = '<p class="error">' + esc(err.message) + '<br />如果你是本地预览，请通过 HTTP 服务访问（例如 <code>python3 -m http.server</code>），不要用 file:// 直接打开。</p>';
    }
  }

  /* -------------------------------------------------------- 公司 / 搜索结果 */

  function viewCompanies() {
    const list = state.outline ? state.outline.companies : [];
    const groups = new Map();
    for (const c of list) {
      if (!groups.has(c.group)) groups.set(c.group, []);
      groups.get(c.group).push(c);
    }
    const blocks = Array.from(groups.entries()).map(([group, companies]) =>
      '<section class="panel"><h2>' + esc(group) + '</h2><div class="grid grid-3">' +
      companies.map((c) => {
        const n = c.questions.length + c.topics.reduce((s, t) => s + t.questions.length, 0);
        return '<a class="card company-card" href="#/company/' + encodeURIComponent(c.name) + '">' +
          '<div class="card-head"><h3>' + esc(c.name) + '</h3><span class="pill">' + n + ' 题</span></div>' +
          '<p class="muted">' + c.topics.map((t) => esc(t.name)).join(' · ') + '</p></a>';
      }).join('') + '</div></section>'
    ).join('');
    return '<nav class="crumbs"><a href="#/">首页</a><span>/</span><span>公司题库</span></nav>' +
      '<header class="page-head"><h1>按公司浏览</h1><p class="lede">共 ' + list.length +
      ' 家公司的 ' + list.reduce((s, c) => s + c.questions.length + c.topics.reduce((x, t) => x + t.questions.length, 0), 0) +
      ' 道题。题解按专题撰写，公司维度的题解在计划中，当前先提供题面清单。</p></header>' + blocks;
  }

  function viewCompany(name) {
    const c = state.outline ? state.outline.companies.find((x) => x.name === name) : null;
    if (!c) return viewMissing('没有找到这家公司');
    const rows = (items, topicName) => items.map((q) =>
      '<li class="q-row is-pending"><span class="q-order">' + String(q.order).padStart(2, '0') + '</span>' +
      '<span class="q-body"><span class="q-title">' + esc(q.question) + '</span>' +
      (topicName ? '<span class="chip chip-ghost">' + esc(topicName) + '</span>' : '') +
      ((q.askedAt && q.askedAt.length) ? '' : '') + '</span>' +
      '<span class="q-state">待撰写</span></li>').join('');
    return '<nav class="crumbs"><a href="#/">首页</a><span>/</span><a href="#/companies">公司题库</a><span>/</span><span>' + esc(c.name) + '</span></nav>' +
      '<header class="page-head"><h1>' + esc(c.name) + '</h1><p class="lede">' + esc(c.group) + '</p></header>' +
      (c.questions.length ? '<section class="panel"><h2>公司专属题</h2><ul class="q-list">' + rows(c.questions, '') + '</ul></section>' : '') +
      c.topics.map((t) => '<section class="panel"><h2>' + esc(t.name) + '</h2><ul class="q-list">' + rows(t.questions, '') + '</ul></section>').join('');
  }

  async function ensureSearchIndex() {
    if (!state.searchIndex) state.searchIndex = await loadJSON('data/search-index.json');
    return state.searchIndex;
  }

  function snippet(text, terms) {
    const lower = text.toLowerCase();
    let at = -1;
    for (const t of terms) {
      const i = lower.indexOf(t);
      if (i >= 0 && (at < 0 || i < at)) at = i;
    }
    if (at < 0) return esc(text.slice(0, 160)) + '…';
    const start = Math.max(0, at - 60);
    const raw = text.slice(start, start + 200);
    let html = esc(raw);
    for (const t of terms) {
      if (!t) continue;
      html = html.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi'), '<mark>$1</mark>');
    }
    return (start > 0 ? '…' : '') + html + '…';
  }

  async function runSearch(query) {
    const box = $('#search-results');
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) { box.innerHTML = '<p class="muted">输入关键词开始搜索，例如「KV cache」「方差」「GQA」。</p>'; return; }
    box.innerHTML = '<p class="muted">搜索中…</p>';
    let index;
    try { index = await ensureSearchIndex(); }
    catch (err) { box.innerHTML = '<p class="error">' + esc(err.message) + '</p>'; return; }
    const hits = index.items.filter((it) => {
      const hay = (it.title + ' ' + it.topic + ' ' + (it.askedAt || []).join(' ') + ' ' + it.text).toLowerCase();
      return terms.every((t) => hay.includes(t));
    }).slice(0, 40);
    if (!hits.length) { box.innerHTML = '<p class="muted">没有匹配的题目。</p>'; return; }
    box.innerHTML = hits.map((it) =>
      '<a class="search-hit" href="' + it.route + '">' +
      '<div class="hit-head"><span class="pill pill-ghost">' + esc(it.topic) + '</span>' +
      '<span class="hit-title">' + esc(it.title) + '</span></div>' +
      '<p class="hit-text">' + snippet(it.text, terms) + '</p></a>'
    ).join('');
  }

  function viewMissing(message) {
    return '<section class="panel"><h1>404</h1><p class="muted">' + esc(message) + '</p>' +
      '<p><a class="btn" href="#/">回到首页</a></p></section>';
  }

  /* ------------------------------------------------------------------ 渲染 */

  function render() {
    const route = parseRoute();
    const content = $('#content');
    const [a, b, c] = route.seg;
    content.innerHTML = '';

    if (!a) content.innerHTML = viewHome();
    else if (a === 'topic') content.innerHTML = viewTopic(b);
    else if (a === 'q') content.innerHTML = viewQuestion(b, c);
    else if (a === 'companies') content.innerHTML = viewCompanies();
    else if (a === 'company') content.innerHTML = viewCompany(b);
    else if (a === 'search') content.innerHTML = '';
    else content.innerHTML = viewMissing('页面不存在：' + location.hash);

    if (a === 'search') openSearch(route.query.get('q') || '');
    else closeSearch();

    if (a === 'topic' && b) {
      const topic = state.topicById.get(b);
      if (topic) fillTopicOverview(topic);
    }
    if (a === 'q' && b) {
      const q = state.questionById.get(b);
      if (q) fillQuestion(q, c);
    }

    if (!c) window.scrollTo({ top: 0, behavior: 'auto' });
    renderNav();
    closeMenu();
    $('#content').focus({ preventScroll: true });
  }

  /* -------------------------------------------------------------- 搜索面板 */

  function openSearch(initial) {
    const modal = $('#search-modal');
    modal.hidden = false;
    const input = $('#search-input');
    if (initial != null) input.value = initial;
    input.focus();
    runSearch(input.value);
  }
  function closeSearch() {
    const modal = $('#search-modal');
    if (modal) modal.hidden = true;
  }

  /* ------------------------------------------------------------------ 交互 */

  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    const dark = $('#hljs-dark');
    const light = $('#hljs-light');
    if (dark && light) { dark.disabled = theme !== 'dark'; light.disabled = theme === 'dark'; }
    store.set('aieiq-theme', theme);
  }

  function toggleRead(id) {
    if (state.read.has(id)) state.read.delete(id); else state.read.add(id);
    store.set('aieiq-read', Array.from(state.read));
    render();
  }

  function closeMenu() { $('#sidebar').classList.remove('is-open'); }

  function bind() {
    window.addEventListener('hashchange', render);

    $('#theme-toggle').addEventListener('click', () => applyTheme(state.theme === 'dark' ? 'light' : 'dark'));
    $('#menu-toggle').addEventListener('click', () => $('#sidebar').classList.toggle('is-open'));
    $('#search-trigger').addEventListener('click', () => openSearch(null));
    $('#search-modal').addEventListener('click', (e) => { if (e.target.id === 'search-modal') closeSearch(); });

    let timer = null;
    $('#search-input').addEventListener('input', (e) => {
      clearTimeout(timer);
      const value = e.target.value;
      timer = setTimeout(() => runSearch(value), 160);
    });

    $('#sidebar-search').addEventListener('input', (e) => {
      state.filter = e.target.value;
      renderNav();
    });

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(null); }
      else if (e.key === 'Escape') closeSearch();
      else if (e.key === 'Enter' && !$('#search-modal').hidden) {
        const first = $('.search-hit');
        if (first) { location.hash = first.getAttribute('href'); closeSearch(); }
      }
    });

    document.addEventListener('click', (e) => {
      const action = e.target.closest('[data-action]');
      if (!action) return;
      const name = action.dataset.action;
      if (name === 'search') openSearch(null);
      else if (name === 'toggle-read') toggleRead(action.dataset.id);
      else if (name === 'copy-link') {
        const url = location.href;
        if (navigator.clipboard) navigator.clipboard.writeText(url);
        action.textContent = '已复制 ✓';
        setTimeout(() => { action.textContent = '复制链接'; }, 1500);
      }
    });

    // 目录点击：hash 路由内的小节锚点
    document.addEventListener('click', (e) => {
      const link = e.target.closest('a[data-toc]');
      if (!link) return;
      e.preventDefault();
      const id = link.dataset.toc;
      const target = document.getElementById(id);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', '#/q/' + state.currentDocId + '/' + id);
      }
    });
  }

  /* ------------------------------------------------------------------ 启动 */

  let booted = false;

  async function boot() {
    if (booted) return;   // 幂等：脚本被重复执行或 DOMContentLoaded 重复触发时只启动一次
    booted = true;
    applyTheme(store.get('aieiq-theme', 'dark'));
    state.read = new Set(store.get('aieiq-read', []));
    try {
      const [catalog, outline] = await Promise.all([
        loadJSON('data/catalog.json'),
        loadJSON('data/outline.json').catch(() => null)
      ]);
      state.catalog = catalog;
      state.outline = outline;
      state.topics = catalog.topics;
      for (const t of catalog.topics) {
        state.topicById.set(t.id, t);
        for (const q of t.questions) state.questionById.set(q.id, Object.assign({ topicId: t.id, topicTitle: t.title }, q));
      }
      bind();
      render();
    } catch (err) {
      $('#content').innerHTML = '<section class="panel"><h1>数据加载失败</h1><p class="error">' + esc(err.message) +
        '</p><p class="muted">请先运行 <code>node scripts/build-site.mjs</code> 生成 docs/data/，并通过 HTTP 服务打开本站。</p></section>';
    }
  }

  // 供目录锚点使用
  Object.defineProperty(state, 'currentDocId', {
    get() { const r = parseRoute(); return r.seg[0] === 'q' ? r.seg[1] : ''; }
  });

  document.addEventListener('DOMContentLoaded', boot);
})();
