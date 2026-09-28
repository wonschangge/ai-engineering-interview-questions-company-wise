---
type: question
id: agents-03
topic: Agent 与工具调用
order: 3
question: 结构化输出与 function calling 有什么区别？
question_en: What is the difference between structured output and function calling?
asked_at: [Mistral AI, Apple]
level: 进阶
tags: [function-calling, 结构化输出, 约束解码, schema]
sources:
  - title: How does Function Calling work in LLMs?
    url: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms
    author: Amit Shekhar (Outcome School)
    published: ""
related: [agents-01, agents-04, agents-05, agents-08]
updated: 2026-09-28
---

## 一句话答案

> 两者约束的对象不同。**结构化输出**约束「模型的答复长什么样」：用 JSON Schema、正则或文法把生成限制在合法结构内，输出**本身就是**交付物（抽取的字段、分类标签、排序结果），调用方拿到后直接消费。**function calling** 约束「模型想做什么」：输出是一份「工具名 + 参数」的调用意图，模型自己不执行、也拿不到执行结果，必须由 runtime 执行后把结果回灌到下一轮。
> 技术上两者同源——都靠 schema 限住生成空间，function calling 就是把工具名与参数装进一个特殊 schema 的结构化输出。
> 差别全在运行时：function calling 额外需要工具注册表与能力发现、参数校验与权限、幂等与副作用管理、结果序列化与截断，以及每轮在「继续推理」与「调用工具」之间的决策（[[agents-08]]）。

## 面试官在考什么

- **概念边界。** 能不能用一句话把「数据 vs 动作」「谁执行」切开。只会说「都是让模型输出 JSON」的答案，说明没有区分「交付物」与「意图」。
- **同源关系。** 这题真正的题眼：能否指出 function calling 只是「enum 约束工具名 + 被选工具的 parameters 约束参数」的结构化输出。理解了这一点，MCP（[[agents-04]]）与工具路由（[[agents-05]]）就都能挂到同一根线上——它们标准化/裁剪的是候选工具集合，也就是那个 enum。
- **工程差异的完整清单。** 工具注册与能力发现、参数校验与权限检查、幂等与副作用管理、结果序列化与截断、调用决策，这五件事结构化输出一件都不需要。能主动列出来是分水岭。
- **成本意识。** 工具定义是常驻 prompt 的开销，每轮都要带；能报出量级并说清缓存与裁剪手段（[[inference-serving-05]]）。
- **失败模式分层。** schema 校验失败 → 重试或回灌错误；工具执行失败 → 按瞬时/参数错/语义失败分别处置（[[agents-02]]）；JSON mode 只保证「语法是 JSON」，不保证「符合你的 schema」。

常见错误答案：

- 「两者是同一个东西，只是叫法不同」。忽略了执行环节、副作用与多轮循环，恰好是本题要区分的地方。
- 「function calling 就是模型去调用函数」。模型从不执行任何东西，它只生成一段 JSON；执行权与安全边界都在调用方（[[agents-10]]）。
- 「用了 JSON mode 就等于结构化输出」。JSON mode 只保证能被 `json.loads` 解析，字段名、类型、枚举、必填项都可能错。

## 原理与推导

### 1. 定义：约束「输出形状」还是约束「意图」

设模型分布为 $p_\theta$、输入为 $x$、schema 描述的合法串集合为 $\mathcal{L}(S)$。

- **结构化输出**：要求 $y \sim p_\theta(\cdot \mid x)$ 满足 $y \in \mathcal{L}(S)$，并且 $y$ 就是要用的结果。正确性判据是「解析成功 + 字段语义正确」。
- **function calling**：给定工具集合 $T = \{t_1, \dots, t_n\}$，模型输出一条调用记录 $c = (\text{name}, \text{arguments})$，其中 $c \in \mathcal{L}(S_{\text{call}})$，而 $S_{\text{call}}$ 是由 $T$ 生成的 schema：`name` 被约束成工具名枚举，`arguments` 用被选中工具的 parameters 校验。$c$ 不是结果，是**请求**；正确性判据还要多一层「执行结果是否达成用户目标」。

一句话概括：结构化输出把语言压成**数据**，function calling 把语言压成**动作请求**。前者是单向的，后者会回到对话里，于是有了循环。

### 2. 技术同源：三种保证合法性的路线

从弱到强：

1. **生成后校验 + 重试。** 模型自由生成，调用方用 jsonschema/pydantic 校验，失败就把可操作的错误信息回灌重试。最灵活，但成本随失败率线性上升，且失败率在长输出与嵌套结构下明显变高。
2. **JSON mode。** 解码时在字符级维护一个 JSON 文法状态，屏蔽会让 JSON 非法的 token。它只保证「这段文本是合法 JSON」，**不管**字段名对不对、枚举值在不在集合里。
3. **约束解码（grammar-constrained decoding / 有限状态机）。** 把 schema、正则或 CFG 编译成自动机，每一步只在「能让整个串最终合法」的 token 上采样。JSON mode 只清零「不是 JSON」这一类错误；约束解码把判据换成你的 schema，于是违反 schema 的错误（字段名、类型、枚举、必填项、嵌套结构）也一并清零（值合法但不正确的语义错误仍然存在）。

第 3 条在数学上就是给 logits 乘一个 mask 再重归一化：

$$p_\theta(y_t = v \mid y_{<t}) \;=\; \frac{p_\theta(v \mid y_{<t}) \cdot m_t(v)}{\sum_{v'} p_\theta(v' \mid y_{<t}) \cdot m_t(v')}, \qquad m_t(v) = \mathbb{1}\big[\delta(q_t, v) \neq \bot\big]$$

$q_t$ 是自动机当前状态，$\delta$ 是状态转移，$\bot$ 表示拒绝。两个工程要点：

- **状态按 token 推进，不是按字符推进。** 一个 token 可能一次携带多个字符（`"Paris"` 可能只占一两个 token），也可能只写完半个字符串字面量。所以 $q$ 必须包含「token 内部位置」这一维度，否则要么误屏蔽合法 token，要么漏放行非法 token。
- **每步 mask 的成本要摊掉。** 朴素做法是对词表 $V$（十万量级）逐个试探，代价与 $V$ 同阶。工程上先把文法编译成 FSM，再用 token trie 预先把「每个状态允许哪些 token」索引成表，解码每步只查一次表。

**function calling 就是第 3 条的特例**：外层 schema 是 `{"name": ..., "arguments": ...}`，`name` 用工具名枚举约束，`arguments` 用被选工具的 parameters 约束。并行调用只是外层再套一层数组 schema；`tool_choice`（auto / required / 指定某个工具）是把同一个 schema 进一步收窄或放宽。部分实现为了流式可用，会先把 `arguments` 当字符串生成再解析，这也是历史上「参数 JSON 偶发不可解析」的来源；启用严格模式后这条路径同样受约束。

### 3. 工程差异：结构化输出只要 schema + 解析，function calling 要一整套运行时

| 维度 | 结构化输出 | function calling |
| --- | --- | --- |
| 目标 | 数据：抽取、分类、打分、路由 | 动作：检索、下单、发信、跑测试 |
| 谁执行 | 无需执行，调用方直接消费 | runtime 执行，模型不执行 |
| 轮次 | 单次调用即完成 | 多轮循环：调用 → 结果回灌 → 再决策 |
| 失败处理 | 校验失败 → 重试或回灌错误 | 校验失败重试；**执行失败**要分瞬时/参数错/语义失败处置（[[agents-02]]） |
| 副作用 | 纯函数，重复执行无害 | 可能写数据，必须幂等、可审计、可回滚（[[agents-10]]） |
| 上下文代价 | 一次 schema 定义 + 一次输出 | schema 每轮随请求重发，结果逐轮累积 |
| 决策负担 | 无：给什么抽什么 | 每轮要在「继续推理」与「调工具」之间选（[[agents-08]]） |
| 典型场景 | 发票字段抽取、意图分类、rerank 打分 | 查订单、下单、发邮件、执行 SQL |

function calling 多出来的五件事，缺一件都会在生产里出问题：

1. **工具注册表与能力发现。** 工具从哪来、有无版本、当前用户可见哪些；这也是 MCP（[[agents-04]]）要解决的问题。注册表还决定每轮往 prompt 里塞多少个 schema（[[agents-05]]）。
2. **参数校验与权限检查。** 结构合法不等于业务合法：日期格式、金额上限、用户是否有权操作这个资源，都必须由 runtime 判定。校验或权限拒绝时，要把可操作的错误回灌给模型，而不是抛 stack trace。
3. **幂等与副作用管理。** 分布式下没有免费的 exactly-once；工程做法是 at-least-once + 幂等键，写操作要能查询状态、能补偿（[[agents-10]]、[[agents-11]]）。
4. **结果的序列化与截断。** 工具返回可能远大于模型需要的内容，必须按 token 预算裁剪并留截断标记，同时提供「继续取数」的工具（分页、按 id 取详情）。
5. **决策：继续还是调用。** 模型要判断「我知道答案」还是「需要外部事实」，这既是能力问题也是产品策略问题（[[agents-08]]）。

反过来，把 function calling 当结构化输出用是很常见的降级设计：只需参数、不需要执行时（例如先让模型产出一个检索 query 再自己拼调用），完全可以用纯结构化输出，省掉整个执行层。判断标准只有一个——**结果需不需要回到模型**。

### 4. 成本账：工具定义是常驻开销

工具定义随每轮请求重发，所以它的成本是「个数 × 每轮 token × 轮数」。以实测的 `search_flights`（枚举 + 嵌套对象 + 详细描述）为 176 token 计：

| 工具数 | 每轮工具定义 token | 10 轮累计 | 50 轮累计 |
| --- | --- | --- | --- |
| 1 | 176 | 1.76k | 8.8k |
| 5 | 880 | 8.8k | 44k |
| 10 | 1.76k | 17.6k | 88k |
| 20 | 3.52k | 35.2k | 176k |
| 50 | 8.8k | 88k | 440k |

前缀缓存能把重复前缀的 prefill 从计算里省掉（[[inference-serving-05]]），但 token 依然占上下文窗口、依然参与注意力计算，所以「工具越多越贵」这条趋势不变。裁剪手段：只暴露当前任务需要的工具族、把长描述改成短描述 + 少量 few-shot、把枚举值挪到工具返回里而不是参数描述里。

## 数值与代码验证

### token 成本实测

口径：`tiktoken` 的 `o200k_base` 与 `cl100k_base`；JSON 用紧凑序列化 `separators=(",", ":")`；只统计对象本身的 JSON 文本，**不含** chat template、API 包装层与工具选择指令。换 tokenizer 结论会变，上线前要按目标模型实测。

| 对象 | o200k | cl100k | 说明 |
| --- | --- | --- | --- |
| `get_weather`（文章里的最小工具） | 50 | 50 | 单字段 string 参数 |
| `get_weather`（同义中文描述） | 44 | 50 | 中文短句在这个 tokenizer 下并没有更贵 |
| `search_flights`（枚举 + 嵌套对象 + 长描述） | 176 | 180 | 生产级工具的量级 |
| 结构化输出的抽取 schema（含数组嵌套） | 88 | 84 | 输出侧 schema |
| 一次 20 行明细的 JSON 参数（每行 3–4 个短字段） | 345 | 344 | 长参数的代价 |
| 一个小工具结果（`{temperature, condition, city}`） | 14 | 14 | 结果侧 |
| 10 条检索结果（标题 + 约 40 词摘要） | 584 | — | 结果侧，真实网页可达数万 token |

两点结论：第一，**参数是长 JSON 时，延迟、成本与出错面同时上升**（20 行明细 345 token，而任何一处不合法整条调用作废），所以宁可拆成多个小工具或分步生成；第二，结果侧比定义侧更容易失控，截断策略必须显式设计。

### 可靠性复利：每轮的合法调用率是乘起来的

口径：每一步独立同分布、成功率为 $p$，n 步不改写策略；「重试一次」指单步最多尝试 2 次（第 2 次成功概率仍为 $p$），因此单步有效成功率是 $1-(1-p)^2$。这是**乐观上限**：它假定每次失败都能被一次重试独立修好，而真实失败往往相关（同一个坏 schema 会在多步连续失败，重试修不了根因），所以「重试」一列的实际收益低于表中数值。

| 单步成功率 $p$ | 步数 $n$ | 无重试 $p^n$ | 每步最多试 2 次 |
| --- | --- | --- | --- |
| 0.99 | 10 | 90.4% | 99.9% |
| 0.99 | 20 | 81.8% | 99.8% |
| 0.95 | 10 | 59.9% | 97.5% |
| 0.95 | 20 | **35.8%** | **95.1%** |
| 0.90 | 20 | 12.2% | 81.8% |

$0.95^{20} = 0.3585$，也就是「每步 95% 正确」的 agent 跑 20 步只剩三分之一左右；把单步重试一次做到位（$1-0.05^2 = 0.9975$，$0.9975^{20} = 0.9512$）就能拉回 95.1%。这就是「重试要按步做、而不是整任务重跑」的定量理由。

### 无约束 JSON 的失败复利（教学模型）

口径：这是**假设模型**而非实测——设输出有 $m$ 个必须正确的结构位（字段名、类型、枚举、括号闭合各算一位），每个位置独立正确率 $q$，则一次通过的联合合法率是 $q^m$：

| 单位置正确率 $q$ | 结构位数 $m$ | 一次通过 $q^m$ | 重试一次后 $1-(1-q^m)^2$ |
| --- | --- | --- | --- |
| 0.99 | 20 | 81.8% | 96.7% |
| 0.99 | 50 | 60.5% | 84.4% |
| 0.95 | 20 | 35.8% | 58.8% |
| 0.95 | 50 | 7.7% | 14.8% |

真实失败是相关的，所以实测通常好于这个模型；但它解释了为什么「靠提示词求 JSON」在长输出与嵌套结构下会突然不可用，而约束解码是把 $q$ 这一项直接变成 1（语法维度），剩下的只有语义错误。

### 代码：工具循环里的校验、回灌与截断

```python
import json
from jsonschema import Draft202012Validator

TOOLS = {"search_flights": (search_flights_impl, SEARCH_FLIGHTS_SCHEMA)}
MAX_RESULT_TOKENS = 800

def run_tool_call(call, budget):
    """call = {"id": ..., "name": ..., "arguments": "<json string>"}"""
    tool = TOOLS.get(call["name"])
    if tool is None:                                    # 幻觉工具名：可修复错误
        return {"error": f"unknown tool {call['name']!r}; available: {list(TOOLS)}"}
    impl, schema = tool
    try:
        args = json.loads(call["arguments"])            # 严格模式下这步不会失败
    except json.JSONDecodeError as e:
        return {"error": f"arguments must be valid JSON: {e.msg}"}
    errors = sorted(Draft202012Validator(schema).iter_errors(args), key=lambda e: e.path)
    if errors:                                          # 回灌可操作信息，不回灌 stack trace
        return {"error": "; ".join(f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors)}
    if not authorized(call["name"], args, budget):       # 权限/配额永远在 runtime 判定
        return {"error": "permission denied: caller lacks scope for " + call["name"]}
    result = impl(**args, idempotency_key=call["id"])    # 幂等键来自调用 id
    text = json.dumps(result, ensure_ascii=False)
    truncated = estimate_tokens(text) > MAX_RESULT_TOKENS
    if truncated:                                       # 截断要留标记，不能静默丢
        text = truncate_with_marker(text, MAX_RESULT_TOKENS)
    return {"content": text, "truncated": truncated}
```

严格模式下的 schema 写法有个反直觉的约束：**所有 property 都要进 `required`，可选字段只能写成 nullable union**；不支持的校验关键字要在编译期就被发现，业务约束挪到运行时。

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "origin": {"type": "string"},
    "cabin": {"type": "string", "enum": ["economy", "premium", "business", "first"]},
    "max_price": {"type": ["number", "null"], "description": "null 表示不限"}
  },
  "required": ["origin", "cabin", "max_price"]
}
```

## 常见追问

- **追问**：约束解码会不会降低生成质量（把自由文本变成受限路径）？
  - 要点：会，机制有两层。① 每步只在受限支撑集上重归一化，路径偏好被改变，长推理链被压成字段填充；② 结构化字段常把数字、代码、推理内容变成字符串，token 化方式偏离预训练分布。经验上，把整段推理塞进 JSON 字段后，推理类任务的表现通常低于自由文本（幅度随任务与模型变化）。工程结论是「推理自由、结论受限」：让模型先在普通文本里推理，最后一步再用 schema 产出结果，或分成两次调用。
- **追问**：并行工具调用（一次输出多个调用）怎么协调与去重？
  - 要点：schema 是数组，模型一次吐出多条 $(name, arguments)$。runtime 要处理四件事：保持结果与调用 id 一一对应（并发执行但按 id 归位）；检测参数间的隐藏依赖（模型可能并行给出本应串行的调用，需按依赖图降级为重排执行）；用调用 id 或参数哈希做去重；只读工具才安全并行，写操作串行或走审批（[[agents-10]]、[[agents-11]]）。部分失败时每个调用的结果都要回灌，不能让成功的那几个被丢掉。
- **追问**：为什么「让模型输出 JSON 再正则解析」在生产里不可取？
  - 要点：失败率随输出长度与嵌套深度上升（上面的 $q^m$ 模型）；正则只能做浅层校验，遇到多行字符串、转义、嵌套括号就崩；失败发生在最贵的时刻——token 已经付完才发现不可用。正确做法是严格 schema/约束解码保证语法，运行时校验兜底语义（解析成功但值不合法：日期格式错、枚举外取值、幻觉出来的 id）。这类格式错误的分类处置见 [[agents-02]]。
- **追问**：严格模式下的 schema 有哪些兼容性坑？
  - 要点：多数实现要求所有 property 都在 `required`、`additionalProperties: false`，可选字段写成 `type: ["string", "null"]`；`pattern`、`format`、`minimum`/`maxLength` 这类校验关键字支持度不一，可能被编译期拒绝或被忽略，别把业务校验押在上面；并集用 `anyOf`，`oneOf`/`allOf` 在跨实现时一致性差；递归自引用（schema 内部 ref 指向自身）深度有限，深层嵌套建议拍平成扁平字段或拆成多个工具。
- **追问**：工具返回结果太大怎么办？
  - 要点：按 token 预算截断，保留头部与尾部并显式标注「已截断」，同时提供分页/按 id 取详情的能力，绝不静默丢弃（模型会把缺失当成事实）；更好的做法是在工具侧只返回模型需要的字段（id + 摘要），原始数据留在外部存储。实测 10 条带摘要的检索结果就有 584 token，一个原始网页可以到数万。
- **追问**：流式场景怎么处理？
  - 要点：增量输出意味着中途的 JSON 一定不完整，需要栈式增量 parser 才能边生成边渲染（对应的产品能力是「部分生成的结构」）；不要在前缀上用「补齐括号再解析」的取巧写法做业务判断，等结构完整或等终止符再落地副作用。

## 公司变体

- **Mistral AI**：API 面同时提供 `response_format` 的 JSON Schema 结构化输出与 `tools`/`tool_choice` 的工具调用，两条路径都以同一份 JSON Schema 为输入，因此提问角度偏**工程实现**：同一份 schema 在「直接产出数据」与「产出工具调用」两条路径上的编译差异、严格模式对 schema 子集的限制、工具定义与参数在自家 tokenizer 下的成本、以及 agentic 循环里工具结果以什么格式注入。准备时把「schema → 约束 → 解析 → 执行 → 回灌」这条链路按他们 API 的参数名过一遍，比背定义有用。
- **Apple**：端侧 Foundation Models 框架用 `@Generable` 之类的宏从原生类型生成 schema，并走约束解码（guided generation）保证输出合法，流式时会给出部分生成的结构；Apple 也开源过 constrained decoding 的实现。于是问题会落到**原理必须可实现**的层面：文法如何按 tokenizer 索引、约束状态怎么与 KV cache 和增量解析配合、端侧内存与延迟预算下 schema 能复杂到什么程度、以及非法 token 被屏蔽后模型质量如何退化。既不是纯接口题，也不是纯数学题。

## 相关题目

- [[agents-01]] — ReAct 模式：function calling 是 ReAct 里「动作」空间的现代实现
- [[agents-02]] — agent 循环中的工具错误、超时与重试：解析失败与执行失败要分层处置
- [[agents-04]] — MCP 与传统 function calling 的区别：工具描述与能力发现的标准化
- [[agents-05]] — 工具数量多少算太多、schema 怎么设计：候选 enum 越大，选择越难
- [[agents-08]] — agent 如何决定调用工具还是凭自身知识回答
- [[agents-10]] — 让 agent 的操作可逆、可审计：副作用管理
- [[agents-11]] — 高风险操作的 human-in-the-loop 审批
- [[inference-serving-05]] — prefix caching 与 prompt caching：工具定义常驻 prompt 的缓存手段

## 参考资料与归属

- [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms) — Amit Shekhar (Outcome School)，2026-06-13。上文「模型不执行函数、只输出结构化请求」「五步循环」「并行与多步调用」「与结构化输出、JSON mode 的关系」取自该文。

token 与可靠性的数值、约束解码的 FSM/词表索引机制、严格模式的兼容性坑、增量解析以及公司变体部分，属于自行推导与本地实测，不在上述参考源覆盖范围内；验算脚本口径见上文各表格。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
