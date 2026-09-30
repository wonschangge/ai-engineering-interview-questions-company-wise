# 公司题库撰写流水线（给并行 agent 用）

本文件是公司题库的**唯一操作手册**。你负责若干家公司，每家公司按下面的流程走完再进入下一家。

## 0. 铁律

- **只写自己 lane 里的公司目录**：`docs/公司题库/<slug>/`（写范围见任务卡）。不要碰别的公司、不要碰 `docs/公司题库/mistral/`。
- **不要运行 git**；提交由 lead 统一做。
- **不要运行会写数据的构建**：只能跑 `node scripts/build-site.mjs --no-write`（只校验、不写 `docs/data/`）。
- **不要修改** `scripts/*`、`docs/data/*`、`docs/<专题目录>/*`（那 10 个专题是已定稿内容，只能读）。
- 临时文件一律写到自己的 lane 目录：`.work/<lane>/...`。
- 不要用 ASCII 双引号写中文正文（用「」）；美元符号写成 `\$`（例如 `\$47/天`），裸 `$` 会被当成 KaTeX 分隔符。

## 1. 输入

每家公司一个输入文件：`scripts/batches/company-<slug>.json`

```json
{
  "company": "Mistral AI",          // front matter 的 company 字段用它（README 里的公司全名）
  "slug": "mistral",                // 目录名与 id 前缀
  "group": "前沿 AI 实验室",
  "total": 8,
  "topics": ["coding", "llm-internals", ...],
  "questions": [
    { "order": 1, "topic": "coding", "topicName": "编程与数据结构",
      "question": "中文题面（必须逐字使用）", "questionEn": "英文题面（必须逐字使用）",
      "sources": [ { "title": "...", "url": "...", "author": "...", "published": "..." } ] }
  ]
}
```

`question` / `question_en` 必须**逐字**写进文档 front matter —— 构建会把它们与 README 题单做归一化比对，不一致直接报错。

`topic` 的取值：十个跨公司专题的 slug（`llm-internals` / `inference-serving` / `rag` / `agents` / `finetuning` / `evaluation` / `safety` / `multimodal` / `system-design` / `coding`），或三个公司专属小节（`ml-fundamentals` / `applied` / `behavioral`，没有对应专题目录）。

专题 slug → 专题目录名（读它们的题解作为知识底座）：

| slug | 目录 |
| --- | --- |
| llm-internals | `docs/LLM 内部原理与架构/` |
| inference-serving | `docs/推理、服务与 GPU 性能/` |
| rag | `docs/RAG 与检索/` |
| agents | `docs/Agent 与工具调用/` |
| finetuning | `docs/微调、后训练与对齐/` |
| evaluation | `docs/评估与可观测性/` |
| safety | `docs/安全、安保与负责任 AI/` |
| multimodal | `docs/多模态、语音与声音 AI/` |
| system-design | `docs/AI 系统设计/` |
| coding | `docs/编程与数据结构/` |

**风格样例**：`docs/公司题库/mistral/`（8 篇 + 导读）是本流水线产出的参考实现，动手前先读 1–2 篇对齐语气与颗粒度。

### 1.1 `readme_name`：公司名与 README 小节标题对不上时必填

公司 README 的 `name` 是展示名（目录名、站点路由、catalog 都用它），而 `README.zh-CN.md` 的小节标题常带限定语。两家对不上的必须显式声明：

```yaml
# docs/公司题库/zhipu/README.md
name: 智谱                    # 展示名
readme_name: 智谱 AI（GLM）    # README.zh-CN.md 的 ### 小节标题原文
```

```yaml
# docs/公司题库/amazon/README.md
name: Amazon
readme_name: Amazon（AWS）
```

- 绑定顺序是 `name` 逐字 → `readme_name` 别名 → 去掉空白与括号后归一化；三者都失配会让构建**报 error**（以前是 warning，于是整家公司的 `group` / 进度分母静默为空，站点上显示「已撰写 0 / N」）。
- `company-index.mjs` 也用它查 `SLUGS` / `EN_COMPANY`，漏写会让该公司的题被误报成「缺英文题面」。
- 其余 33 家两家名字逐字相同，可以不写这个字段。

## 2. 每家公司的工作流

对一家公司，按 `order` 分块（**每块 4 题**，不要更多——并发过高会导致子代理批量失败），每块调用一次 workflow 工具，脚本如下（把 `<slug>`、`<lane>`、`<from>`、`<to>` 换成实际值）：

```js
const slug = '<slug>', lane = '<lane>', from = <from>, to = <to>;
const idx = []; for (let i = from; i <= to; i++) idx.push(i);
const pad = (n) => String(n).padStart(2, '0');
const batch = 'scripts/batches/company-' + slug + '.json';
const TOPIC_DIRS = { 'llm-internals': 'LLM 内部原理与架构', 'inference-serving': '推理、服务与 GPU 性能', 'rag': 'RAG 与检索', 'agents': 'Agent 与工具调用', 'finetuning': '微调、后训练与对齐', 'evaluation': '评估与可观测性', 'safety': '安全、安保与负责任 AI', 'multimodal': '多模态、语音与声音 AI', 'system-design': 'AI 系统设计', 'coding': '编程与数据结构' };

const briefPrompt = (i) => [
  '你在仓库 /home/chanj/WORKSPACE/ai/interview/ai-engineering-interview-questions-company-wise 工作。为 <公司全名> 的第 ' + i + ' 题生成撰写简报。',
  '1. 读题目数据：`node -e "const d=require(\'./' + batch + '\');console.log(JSON.stringify(d.questions[' + (i - 1) + '],null,2));console.log(d.company,d.group)"`',
  '2. 读该题 topic 对应的专题目录里 2–4 篇最相关的题解（front matter + 正文），复用其结论、数字与来源口径；再读该专题 README.md 拿全局视图。专题 slug → 目录名：' + JSON.stringify(TOPIC_DIRS),
  '3. 用 write 写 `.work/' + lane + '/briefs/' + slug + '-' + pad(i) + '.json`，并在回复里原样输出这份 JSON：',
  '{ "order": ' + i + ', "file": "docs/公司题库/' + slug + '/' + pad(i) + '-<中文短标题>.md", "question": "<逐字>", "question_en": "<逐字>", "company": "<公司全名>", "topic": "<topic>", "level": "<入门|进阶|高阶>", "tags": ["..."], "sources": [{"title":"","url":"","author":"","published":""}], "related": ["[[已写题解 id]]"], "brief": ["5–7 条必须覆盖的机制/数字/取舍"] }',
  '硬性：sources 只能用 (a) 批文件自带的来源，(b) 上面读到的那几篇既有题解 front matter 里已列出的来源（原样复制）。不要新编 URL。related 只能指向本仓库已写题目 id。brief 要针对这家公司的题面场景，写清生产中会踩的坑。',
  '只回复三引号包裹的 JSON。'
].join('\n');

const writePrompt = (i) => [
  '按下面简报为公司题库第 ' + i + ' 题（<公司全名>）撰写完整中文题解。',
  '简报：{{PREV}}',
  '要求：读 `scripts/DOC-GUIDE.md`；front matter 用 type: question、id: ' + slug + '-' + pad(i) + '、company: <公司全名>、topic/order/sources/related 按简报、asked_at: []（**不要**写「## 公司变体」一节）、question 与 question_en 逐字照抄简报；正文 8 节骨架（一句话答案/面试官在考什么/原理与推导/数值与代码验证/常见追问/相关题目/参考资料与归属）；数字自己复算并写明口径；与仓库统一常数一致（LLaMA-3-70B 每 token KV cache 320 KiB、LLaMA-3-8B 128 KiB、H100 bf16 989 TFLOPs dense、3.35 TB/s、roofline 295 FLOPs/byte）；中文正文不用 ASCII 双引号；美元符号写 \\$；篇幅 150–300 行。用 write 写入简报的 file 路径。',
  '最后跑 `node scripts/build-site.mjs --no-write 2>&1 | grep -A2 "' + slug + '-' + pad(i) + '"` 自查，只回复文件路径与复算的数字。'
].join('\n');

const reviewPrompt = (i) => [
  '严格审校第 ' + i + ' 题（<公司全名>）的题解。简报：{{PREV}}',
  '逐条核对：front matter（type/id/company/order/asked_at: []/question 与 question_en 逐字一致/sources 与简报一致 ≥1 条）；8 个二级标题逐字一致且顺序正确；**不应**出现「## 公司变体」；brief 每一条是否落实；公式与数字自己复核（与仓库统一常数一致）；交叉引用是否存在；`$` 成对、代码围栏成对、正文无 ASCII 双引号、无填充语、篇幅 150–300 行。',
  '发现真实问题就用 write 覆写同一文件（不要改 front matter 字段值与 8 个标题文本）。跑 `node scripts/build-site.mjs --no-write 2>&1 | grep -A2 "' + slug + '-' + pad(i) + '"` 确认无本篇错误。',
  '只回复一行：OK ' + slug + '-' + pad(i) + ' 或 FIXED ' + slug + '-' + pad(i) + ': 一句话说明改了什么。'
].join('\n');

return await pipeline(idx,
  async (_p, i) => agent(briefPrompt(i), { label: 'brief ' + i, phase: '简报' }),
  async (prev, i) => agent(writePrompt(i).replace('{{PREV}}', String(prev || '').slice(0, 20000)), { label: 'write ' + i, phase: '撰写' }),
  async (prev, i) => agent(reviewPrompt(i).replace('{{PREV}}', String(prev || '').slice(0, 20000)), { label: 'review ' + i, phase: '审校' })
);
```

把 `<公司全名>` 换成批文件里的 `company`。每块跑完检查文件是否落盘（`ls docs/公司题库/<slug>/`），**不要相信工作流返回的 FAILED 列表**——子代理有时会返回 null 但文件其实写好了；以文件系统为准。若某题缺失或审校段整段失败，单独补跑一次（可以只跑写入段或只跑审校段，审校输入从 `.work/<lane>/briefs/` 读）。

全部题目落盘后：

1. 写公司导读 `docs/公司题库/<slug>/README.md`（front matter + 题解清单表 + 这家公司在考什么 + 建议学习顺序 + 自测清单 + 参考资料）。front matter：

```yaml
---
type: company
id: <slug>
name: <公司全名>
title: <公司全名> 面试题库
group: <批文件里的 group>
summary: <一句话概括这家公司的面试侧重>
total: <题数>
updated: 2026-09-28
---
```

`group` 必须是 `前沿 AI 实验室` / `大型科技公司的 AI 组织` / `AI 基础设施与平台公司` / `AI 原生产品公司` / `前置部署与企业级 AI` 之一。

2. 校验：`node scripts/build-site.mjs --no-write`，确认自己公司的 error 为 0（其它公司未完成产生的 warning/error 可以忽略，但要在汇报里区分开）。
3. 用 `send_message` 向 `lead` 汇报：slug、题数、errors/warnings 数字、审校修掉的关键问题（每条 ≤1 行）。
4. 然后进入下一家。

## 3. 常见坑

- 文件名：≤60 字，去掉题面末尾的 `。？！`，把 `/ \ : * ? " < > |` 换成 `-`，`/` 可用 `、`。文件名必须与简报里的 `file` 一致（审校若改名，要同步改简报或保持一次命名）。
- `order` 在公司内从 1 连续编号，构建会校验。
- 正文里的相对链接必须指向真实存在的文件（构建会检查），跨专题引用请用 `../../<专题目录>/README.md` 这种路径（注意 `docs/公司题库/<slug>/` 到 `docs/` 是两级）。
- 双链写成 `[[coding-01]]`，不要包在反引号里。
- 每篇至少 1 条 source，否则构建报错。
