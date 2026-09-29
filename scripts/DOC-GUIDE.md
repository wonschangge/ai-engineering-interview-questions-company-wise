# 题解文档写作规范（DOC-GUIDE）

本规范约束 `docs/<专题>/<序号>-<题目>.md` 这一层文档：它的目录结构、front matter 字段、正文骨架和写作要求。
`scripts/build-site.mjs` 会按同一套规则解析与校验，**不符合规范的文件会导致构建校验失败**。

## 1. 目录与文件命名

```
docs/
├── LLM 内部原理与架构/            # 专题目录（跨公司高频题专题，10 个）
│   ├── README.md                 # 专题导读（type: topic）
│   ├── 01-<题目>.md              # 题解（type: question）
│   └── 02-<题目>.md
└── 公司题库/<slug>/               # 公司专属题（每家一个目录）
    ├── README.md                 # 公司导读（front matter: type: company, id, name, title, group, summary, total, updated）
    └── NN-<中文题目>.md           # 公司题解（front matter: type: question, id: <slug>-NN, company, topic, order, ...）
    └── 01-<题目>.md
```

命名规则：

- 文件名 = `<两位序号>-<中文题目>.md`，序号与 `README.zh-CN.md` 中该专题内的题目顺序一致。
- 非法字符 `/ \ : * ? " < > |` 一律替换成 `-`；只作分隔用的 `/` 可以改写成 `、`（例如「位置插值 / YaRN」→「位置插值、YaRN」）。
- **去掉句末的 `。？！`**（句中标点保留，例如 `…替换了 ReLU、GELU.md`）；过长时可截掉句末的补充说明（如括号内的枚举）。
- 文件名长度控制在 60 个字符以内。
- **稳定 ID**：`<topic-slug>-<序号>`，例如 `llm-internals-01`。ID 一旦发布就不再改动（标题可以改，ID 不能），
  站内路由、深链、`related` 引用全部依赖它。
- 专题 slug 固定为：`llm-internals`、`inference-serving`、`rag`、`agents`、`finetuning`、`evaluation`、
  `safety`、`multimodal`、`system-design`、`coding`。

## 2. front matter

采用精简 YAML 子集，解析器只支持下面这几种写法，请勿引入其它 YAML 语法（不要多行字符串、不要锚点、不要嵌套超过两层）：

```yaml
---
type: question                       # question | topic
id: llm-internals-01                 # 全局唯一
topic: LLM 内部原理与架构              # 所属专题名
order: 1                             # 专题内序号，从 1 开始且连续
question: 解释 scaled dot-product attention，以及为什么 1/sqrt(d_k) 缩放因子很重要。
question_en: Explain scaled dot-product attention and why the 1/sqrt(d_k) scaling factor matters.
asked_at: [OpenAI, xAI]              # 行内列表；没有就写 []，不要省略该字段
level: 进阶                           # 入门 | 进阶 | 高阶
tags: [attention, softmax, 方差]
sources:
  - title: Why Do We Scale Attention by √dₖ? The Math Behind the Scaling Factor
    url: https://outcomeschool.com/blog/scaling-dot-product-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-05
related: [llm-internals-02, llm-internals-03]
updated: 2026-09-28
---
```

注意：

- 值里**不要出现半角 `": "`**（解析器按第一个 `: ` 切分）。中文全角 `：` 不受影响。
- `sources` 至少一条，`url` 必须是可访问的原文链接；改写自多篇就都列上。
- 当题干要求的内容在参考源里没有覆盖时（例如「推导公式」但源文没给公式），可以补「延伸来源」，
  并在该条 title 后加 `（延伸）`，正文第 8 节要说明哪部分来自延伸来源。

## 3. 正文骨架

章节标题**必须逐字使用**下面 8 个二级标题（第 6 节可选），站点据此生成右侧目录：

```markdown
## 一句话答案
## 面试官在考什么
## 原理与推导
## 数值与代码验证
## 常见追问
## 公司变体          <!-- 仅当 asked_at 非空时保留 -->
## 相关题目
## 参考资料与归属
```

各节要求：

1. **一句话答案**：2–4 句，面试现场 60 秒能说完的版本。先给结论，再给理由。可用 `>` 引用块突出。
2. **面试官在考什么**：拆考点（3–5 条无序列表）+ 「常见错误答案」（1–2 条）。说清这题区分什么水平的候选人。
3. **原理与推导**：主线内容。公式用 KaTeX：行内 `$...$`，独立公式 `$$...$$`。推导要分步、每一步说明依据。
   - 正文里的**美元符号要写成 `\$`**（例如 `\$47/天`）；裸 `$` 会被当成 KaTeX 分隔符，把后续正文吞进公式里（构建会报「正文 $ 数量为奇数」）。校验已忽略代码围栏与行内代码中的 `$`。
4. **数值与代码验证**：能有数字就给数字表格，能给代码就给可运行的 Python/PyTorch 片段（带语言标记的代码块）。
   数值要自己算过，与源文对照；源文数字有出入时以自己复算为准并在正文说明。
5. **常见追问**：3–6 条，形如 `- **追问**：…` 换行 `  - 要点：…`，答案要能直接背。
6. **公司变体**：列出 `asked_at` 里的公司，说明这家公司偏工程实现还是偏数学推导（依据公开面试信息，不要编造）。
7. **相关题目**：用 `[[llm-internals-02]]` 这种双链写内部引用，站点会渲染成 SPA 路由；指向整个专题时可写 `[[rag]]`（专题 id），同样会渲染成专题页路由；非本站题目用普通链接。
8. **参考资料与归属**：逐条列出来源（标题 + 作者 + 日期 + 链接），并统一加一句：
   「本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。」

## 4. 写作风格

- 简体中文技术写作，术语保留英文：attention、KV cache、prefill、decode、softmax、GQA、RoPE、batching、latency 等。
- 先结论后解释，不写「众所周知」「相信大家都知道」这类填充语。
- 每道题都要能回答「为什么」，而不是只给结论。
- 不整段逐字搬运原文（版权与质量双重原因）：理解后重写，公式、结论、数值可引用。
- 不要出现「本文」「接下来我们」之类的口水句；直接讲内容。
- 篇幅参考：150–300 行 Markdown。

## 5. 站点与构建

题解由 `docs/index.html`（hash 路由 SPA）在浏览器里直接渲染，站点数据由脚本生成：

```bash
node scripts/build-site.mjs          # 校验题解 + 生成 docs/data/{catalog,search-index,outline}.json
node scripts/build-site.mjs --check  # 只校验（含“数据是否过期”），CI 与提交前用
```

- `catalog.json`：题解元数据与专题树，SPA 的导航来源；`file` 字段是**相对 `docs/` 的站点路径**。
- `outline.json`：从 `README.zh-CN.md` 抽出的全量题库清单（专题题 + 公司题），用于展示「待撰写」占位与进度。
- `search-index.json`：站内搜索索引。
- 校验规则：front matter 必备字段、id 唯一且小写连字符、order 从 1 连续、二级标题逐字匹配、`$` 与代码围栏成对、`asked_at` 非空时必须写「公司变体」、**文档里的相对链接必须指向真实存在的文件**（代码块内的内容不参与检查）。
- 改了 `.md` 之后必须重新运行构建，否则 `--check` 会报「数据已过期」，Pages 工作流也会因此失败。
- 站点回归测试：`npm i -D jsdom && node scripts/test-site.mjs`（没有 jsdom 时可用 `JSDOM_MODULE=<jsdom/lib/api.js 路径>` 指定）。它用 jsdom 真跑一遍 SPA，断言路由、渲染、公式、搜索与链接改写；期望值从 `docs/data/*.json` 推导，新增专题不用改测试。
- 站点资产（marked / KaTeX / highlight.js）已经 vendored 到 `docs/assets/site/vendor/`，不要改成 CDN 引用。

临时脚本、抓取缓存与验算草稿一律写在仓库根的 `.work/`（已在 `.gitignore` 中忽略），不要散落到 `docs/` 下。

批次作业单放在 `scripts/batches/<专题 slug>.json`：每题一条记录（题面、front matter 期望值、参考来源、必须覆盖的知识点清单）。
它是撰写与审校的共同依据，也是后续复核「这篇有没有漏讲」的检查表；新增批次时照抄结构即可。
