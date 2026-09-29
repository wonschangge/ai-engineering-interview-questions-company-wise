---
type: question
id: mistral-06
company: Mistral AI
topic: agents
order: 6
question: function calling 与 LLM 配合时到底是如何工作的？如何让它可靠到足以支撑生产环境的 agent？
question_en: How does function calling actually work with an LLM, and how do you make it reliable enough for production agents?
asked_at: []
level: 进阶
tags: [function-calling, 约束解码, 工具 schema, 可靠性, 实现题]
sources:
  - title: LLM 中的 Function Calling 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-function-calling-work-in-llms
    author: ""
    published: ""
  - title: AI Agent Loop
    url: https://outcomeschool.com/blog/ai-agent-loop
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Writing Tools for Agents（延伸）
    url: https://www.anthropic.com/engineering/writing-tools-for-agents
    author: Anthropic
    published: ""
  - title: Building Effective Agents（延伸）
    url: https://www.anthropic.com/engineering/building-effective-agents
    author: Anthropic
    published: 2024-12-19
  - title: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains（延伸）
    url: https://arxiv.org/abs/2406.12045
    author: Yao et al.
    published: 2024-06-17
related: [agents-03, agents-05, agents-02, agents-09]
updated: 2026-09-29
---

## 一句话答案

> function calling 是两层机制。**约束解码层**把一份 JSON Schema 编译成生成约束：`name` 收窄成工具名枚举，`arguments` 用被选中工具的 `parameters` 校验；模型只产出一份调用意图，从不执行任何函数。**运行时层**解析、校验、鉴权、执行，再把结果作为 observation 回灌下一轮——「schema → 约束 → 解析 → 执行 → 回灌」就是全部链路。Mistral 的 API 面上 `response_format`（结构化输出）与 `tools`/`tool_choice`（工具调用）吃的是同一份 schema，只是编译目标不同。
> 可靠性则是一笔复利：每步正确率是指数项的底数，$0.95^{20}=0.3585$、$0.99^{20}=0.8179$，工具选择准确率从 0.95 提到 0.99，20 步任务的端到端成功率从 35.9% 升到 81.8%。所以生产化的动作不是「把提示词写好」，而是错误分类处置、幂等键、分层预算与外部验证器这四件事。

## 面试官在考什么

- **能不能把机制切成两层。** 说清「模型只产调用意图、runtime 才执行」是及格线；说出约束解码如何把 schema 编译成 mask、`name` 的枚举与 `arguments` 的校验分别约束什么，才是进阶。
- **同一份 schema 的两条路径。** `response_format` 约束的是整段输出，`tools` 约束的是外层 `{name, arguments}` 结构加被选工具的 `parameters`；能不能讲清编译差异，以及严格模式对 schema 子集（必填、`additionalProperties`、可选字段写法）的限制。
- **成本意识。** 工具定义是每轮重发的常驻开销，不是一次性成本：良好 schema 均值 223.7 tok，20 个工具单轮 4,473 tok，20 步循环累计 89,467 tok。能不能报出量级并说清分层（命名空间、两阶段选择、tool retrieval）的收益与代价。
- **会不会算账。** 把每步正确率当指数底数，而不是「差不多就行」的指标；验收要拆成工具选择准确率、参数正确率、端到端成功率三个独立指标，再加工具数消融与混淆矩阵定位下一对要合并或改名的工具。
- **生产可靠性的一整套。** 错误分类（瞬时 / 参数 / 语义 / 依赖 / 模型格式）、幂等键、超时是「结果未知」而不是失败、分层预算向下传递、终止必须由外部验证器确认、上线门槛看 pass^k 而不是平均成功率。

常见错误答案：

- 「function calling 就是模型去调用函数」。模型从不执行代码，它只是把下一步动作表达成一个受 schema 约束的 JSON；执行权、权限与安全边界全在 runtime。
- 「开了 JSON mode 或严格模式就不用校验了」。JSON mode 只保证这段文本能被解析成 JSON，字段名、类型、枚举、必填项都可能不符合你的 schema；严格模式把违反 schema 的那一类清零，但值合法而语义错误的调用（幻觉 id、越权资源、金额超限）仍然存在。
- 「多试几次、把提示词写得更强硬就能到 99%」。重试只对独立瞬时故障有效，参数错与语义失败重试是浪费预算；而且每步 0.95 在 20 步后只剩 35.9%，靠单点优化补不回来。

## 原理与推导

### 1. 机制只有两层：约束解码与运行时执行

给定工具集合 $T=\{t_1,\dots,t_n\}$ 与用户输入 $x$，一次工具调用的完整链路是：

1. **schema**：runtime 把每个工具的 name、description、parameters 序列化成一段常驻 prompt，连同 `tool_choice` 一起发给模型。
2. **约束**：解码时把外层结构编译成 mask——`name` 位只能是工具名枚举里的一个，选中 $t_i$ 后 `arguments` 位切换到 $t_i$ 的 `parameters` 自动机。数学上就是给 logits 乘 mask 再重归一化：

$$p_\theta(y_t = v \mid y_{<t}) \;=\; \frac{p_\theta(v \mid y_{<t})\cdot m_t(v)}{\sum_{v'}p_\theta(v'\mid y_{<t})\cdot m_t(v')},\qquad m_t(v)=\mathbb{1}\big[\delta(q_t,v)\neq\bot\big]$$

   $q_t$ 是自动机状态，$\delta$ 是状态转移，$\bot$ 表示拒绝。注意状态要按 token 推进（一个 token 可能携带多个字符或只写完半个字符串字面量），工程上先把文法编译成 FSM、再用 token trie 把「每个状态允许哪些 token」索引成表，避免每步对十万量级词表逐个试探。
3. **解析**：runtime 收到 `{name, arguments}`，把 `arguments` 反序列化成字典。严格模式下这一步不应失败；如果模型或服务端不支持严格约束，这里就是第一道校验点。
4. **执行**：runtime 做业务校验（日期格式、金额上限、资源归属、调用者权限），再执行函数；写操作必须带幂等键。
5. **回灌**：把结果（或结构化错误）作为一条 observation 追加进 messages，进入下一轮决策。模型看到的只有这段文本，所以它的字节数、截断标记与错误码都属于接口设计的一部分。

第 2 步与第 4 步是这道题的全部：**约束只负责让「形状」合法，合法性之外的正确性（这个工具该不该调、参数值对不对、副作用能不能重放）全部是运行时责任**。幻觉工具名（不在枚举里）在严格模式下不会出现；一旦出现，说明约束路径没生效，runtime 要把可用工具名回灌成可修复错误，而不是抛异常。

### 2. 同一份 schema，两条编译路径

| 维度 | `response_format`（结构化输出） | `tools` / `tool_choice`（工具调用） |
| --- | --- | --- |
| 约束对象 | 整段输出落在你的 schema 里 | 外层 `{name, arguments}`，`name` 为工具名枚举，`arguments` 被选中工具的 `parameters` 约束 |
| 产出语义 | 数据：抽取字段、分类、打分 | 动作请求：检索、下单、执行 SQL |
| 后续步骤 | 拿到即消费，单轮结束 | runtime 执行 → 结果回灌 → 下一轮再决策 |
| `tool_choice` | 无此概念 | `auto` / `required` / 指定工具，等于收窄或放宽外层枚举 |
| 严格模式 | 要求全部 property 进 `required`、`additionalProperties: false` | 同上，另加参数层约束；并行调用是外层再套一层数组 |

三处最容易含糊的地方：

1. **JSON mode ≠ schema 合规。** JSON mode 在字符级维护 JSON 文法状态，只保证整体是合法 JSON；严格模式（把 schema 编译成约束）才把字段名、类型、枚举、必填项一起清零。两者差了整整一类错误。
2. **严格模式吃的是 schema 子集。** 常见限制：可选字段只能写成 nullable union（把 `type` 写成字符串与 null 两个分支的并集）并进 `required`；`pattern`、`format`、`minimum` 这类校验关键字支持度不一，可能被编译期拒绝或被静默忽略；并集优先用 `anyOf`，`oneOf`/`allOf` 跨实现一致性差；递归自引用的深度有限。**业务约束（金额上限、资源归属、时间窗）不要押在 schema 关键字上**，放运行时判。
3. **同源不等于同价。** 两条路径共用 schema，但只有工具路径每轮重发全部定义并累积 observation 历史，成本与上下文占用都是它的独有开销。

### 3. 工具定义是每轮重发的常驻开销

工具定义随每轮请求重发，所以成本是「个数 × 每轮 token × 轮数」。按良好 schema 的均值 $\bar t = 223.7$ tok 计：

| 工具数 $n$ | 单轮定义 token | 20 步循环累计 | 等价 TTFT 增量 |
| --- | --- | --- | --- |
| 5 | 1,118 | 22,367 | +49 ms |
| 10 | 2,237 | 44,733 | +99 ms |
| 20 | 4,473 | 89,467 | +198 ms |
| 50 | 11,183 | 223,667 | +495 ms |

超过 10–20 个候选就要分层，50 工具口径下两阶段选择降到 2,269 tok（−79.7%）、检索式降到 1,238 tok（−88.9%）；代价是前者多一次 LLM 往返、后者漏检即失败。结论：**工具集规模是每轮都在付的税，且它同时抬高选择难度**，能删的工具先删，剩下的再分组、检索或延迟加载（只把当前任务需要的工具族注入上下文），三者的收益与代价见 [[agents-05]]。

这些 token 必须用目标模型自己的 tokenizer 复算：同一份内容换 tokenizer 的偏差实测在 0.84–1.05 倍之间，量级稳，但具体数字不能照搬（[[agents-05]]）。

### 4. 把可靠性算成复利

当每一步独立、成功率为 $p$、且 $n$ 步内不改写策略时，端到端成功率是 $p^n$；$p=0.95$ 意味着平均每 $1/(1-p)=20$ 步就出一次错：

| 每步 $p$ | $K=10$ | $K=20$ | $K=50$ |
| --- | --- | --- | --- |
| 0.99 | 0.9044 | **0.8179** | 0.6050 |
| 0.95 | 0.5987 | **0.3585** | 0.0769 |
| 0.90 | 0.3487 | 0.1216 | 0.0052 |

要让 20 步任务达到 90%，单步需要 $0.9^{1/20}=99.47\%$；50 步则需要 99.79%。**工具选择准确率不是可以「差不多就行」的指标，它是指数项的底数。** 由此推出验收口径：

- **三个独立指标**：工具选择准确率（该不该调这个工具）、参数正确率（schema 合规 + 业务合法）、端到端成功率。只报端到端会把定位信息全丢掉。
- **工具数消融**：固定任务集，暴露 5 / 10 / 20 / 50 个候选各跑一遍，看选择准确率与 $p^n$ 的衰减。
- **混淆矩阵**：最热的非对角元就是下一对要合并或改名的工具（`list_users` + `list_events` → `schedule_event`）；从没被正确选中的工具直接删。
- **描述改写 A/B**：加 when-not-to-use、加示例、加返回值说明当独立变量，一次只改一处，跑回归评测。改行为必须同时改描述，否则描述漂移会让 agent 在后半程「自信地做错」。

### 5. 错误先分类再处置，写操作必须幂等

| 类别 | 例子 | 是否重试 | 处置 |
| --- | --- | --- | --- |
| 瞬时故障 | 429、500/502/503/504、连接超时 | 是 | 指数退避 + 抖动，设最大次数；命中限流要读 `Retry-After` |
| 参数错误 | schema 不合法、枚举外取值、缺必填 | 否 | 把可操作的校验错误回灌给模型修复；修复只给一次兜底重试 |
| 语义失败 | 未找到、无权限、空结果 | 否 | 换策略（改查询、换工具、澄清），或转人工；无权限要走审批而不是硬试 |
| 依赖不可用 | 下游服务熔断、沙箱不可达 | 降级 | 熔断 + 降级路径 + 明确告知模型「该能力当前不可用」 |
| 模型侧格式错误 | 参数 JSON 不可解析、幻觉工具名 | 源头消除 | 严格 schema 在解码层清零，只留一次格式修复兜底 |

两条硬约束：

- **幂等键必须在第一次尝试之前生成，所有重试复用同一个键**；或重试前先按业务键查询状态，存在就取回结果、不存在才重放。每次重试新生成键等于没有幂等键（[[agents-02]]、[[agents-10]]）。
- **超时是「结果未知」而不是失败。** 把超时当失败直接重放，就是重复下单、重复扣款、重复发信的根因。同时，错误绝不能被包装成「成功但结果为空」——这种静默失败会让模型基于空事实继续推理，是最难查的一类 bug。

### 6. 超时与预算分层，并且要向下传递

| 层级 | 取值 | 作用 |
| --- | --- | --- |
| 连接超时 | 2 s | 区分「连不上」与「慢成功」 |
| 单次工具超时 | 10 s | 防止一个工具吃掉整步 |
| 单步超时 | 30 s | 模型加工具的总时长 |
| 整轮预算 | 墙钟 300 s + 步数 20 + token 上限 | 兜住全局 |

20 步乘 30 s 单步超时，最坏就是 600 s（加上 2 s 连接超时是 602 s），已经是 300 s 整轮预算的两倍——所以**每一步都要按剩余预算决定发不发调用**：剩余时间不够就走收尾路径（返回已完成部分 + 原因 + 可续跑），而不是发出去再等它把预算烧穿。超时后必须真正取消下游（abort signal、kill 沙箱进程），否则被放弃的调用还在烧配额，甚至留下无人读取的副作用（[[agents-09]]）。

### 7. 终止与验收：外部验证器加 pass^k

**模型自报完成不可单独采信。** 正确终止 = 显式完成信号 + 外部验证器确认目标状态达成（数据库终态比对、断言、schema 校验、测试通过）。预算耗尽不是「成功」，要返回「部分完成 + 原因 + 可续跑」，让上层能接着跑。

上线门槛看 pass^k 而不是平均成功率：τ-bench 口径下 τ-retail 的 $\text{pass}^8$ 不足 25%（$\text{pass}^1=61.2\%$、τ-airline 为 35.2%）。注意**不能把 pass^k 简化成 $({\text{pass}^1})^k$**：$0.612^8=2.0\%$，与观测到的约 25% 差 12.7 倍。原因是 $x^k$ 在 $[0,1]$ 上凸，Jensen 不等式给出 $\mathbb{E}[x^k]\ge(\mathbb{E}[x])^k$，等号只在各任务同质时成立——真实任务是一部分稳定可解、一部分稳定失败，低成功率的那部分把 $(\mathbb{E}[x])^k$ 拽下来，$\mathbb{E}[x^k]$ 却由稳定可解的那部分撑着，所以观测到的 pass^k 落在独立假设之上。口径上还要记住 $n\gg k$ 才能从 $n$ 次试验估 $\text{pass}^k$（[[agents-09]]）。

### 8. 三个最容易踩的坑，以及 Mistral 场景的取舍

- **静默失败**：工具报错或返回空值却被当作成功继续推理。轨迹必须落 `trace_id`、步号、尝试序号、工具名、错误类别与错误码、重试次数、幂等键、结果字节数与是否截断，以及**模型实际看到的 observation 全文**——只记工具名和耗时的轨迹无法回放决策。
- **返回值不截断**：10 条带摘要的检索结果已是 584 tok，原始网页可达数万 token；不截断会让上下文爆炸，并诱发模型反复调用同一个工具。要截断 + 显式标注 + 分页 + 提供「取详情」工具，绝不静默丢弃。
- **描述与实际行为漂移**：工具改了行为但描述没改，agent 会在后半程自信地做错。改行为必须同步改描述并跑回归评测。

**托管 API 与自部署的差别**：在托管 API 上，严格 schema 的约束由服务端保证，runtime 只需处理业务校验与执行错误；把 Mistral 的 open-weight 权重（Mistral 7B、Mixtral、Devstral 这类）自部署时，约束解码能力取决于推理栈是否支持 guided decoding（vLLM 的 guided decoding、Outlines/XGrammar、llama.cpp 的 GBNF 文法等），工具调用的解析器也随栈而异。所以 **runtime 校验、错误回灌与修复路径在任何部署形态下都不能省**，不能假设严格模式一定存在：把「schema 校验失败」当作一类必须能处理的可修复错误，而不是不可能事件。

## 数值与代码验证

口径：工具定义 token 用 tiktoken `cl100k_base` 的紧凑 JSON 计数（不含 chat template 与 API 包装层）；$\bar t$ 取三个「良好」schema 的均值 $(291+181+199)/3=223.7$ tok；prefill 吞吐按 70B、bf16、TP=8 on 8×H100、MFU 0.4 计，$R_{pre}=989\times10^{12}\times0.4\times8/(2\times70\times10^{9})=22{,}606$ tok/s。所有数字与 [[agents-05]]、[[agents-02]]、[[agents-09]] 同源。

表 A：工具定义的常驻开销（均值口径 $n\bar t$；20 步累计按 $n\bar t\times20$ 取整，所以 $20\times4{,}473=89{,}460$ 与表内 89,467 差 7 token，纯取整）

| 工具数 $n$ | 单轮（均值口径） | 20 步累计 | 等价 TTFT 增量 | 50 步累计 |
| --- | --- | --- | --- | --- |
| 5 | 1,118 | 22,367 | +49 ms | 55,917 |
| 10 | 2,237 | 44,733 | +99 ms | 111,833 |
| 20 | 4,473 | 89,467 | +198 ms | 223,667 |
| 50 | 11,183 | 223,667 | +495 ms | 559,167 |

表 B：50 个工具时的分层方案（token/请求，$\bar t=223.7$）

| 方案 | 计算 | token | 相对全量 |
| --- | --- | --- | --- |
| 全量平铺 | $50\bar t$ | 11,183 | — |
| 两阶段（8 个族摘要 + 族内 8 个工具） | $8\times60+8\bar t$ | 2,269 | −79.7% |
| 检索式 top-5（+ 检索器开销 120） | $5\bar t+120$ | 1,238 | −88.9% |

表 C：累积成功率与「每步重试一次」的对照（重试口径：单步最多 2 次尝试、第 2 次成功率仍为 $p$，属乐观上限）

| 每步 $p$ | $K=20$ 无重试 | 单步有效成功率 $1-(1-p)^2$ | $K=20$ 重试一次 |
| --- | --- | --- | --- |
| 0.99 | 0.8179 | 0.9999 | 0.9980 |
| 0.95 | 0.3585 | 0.9975 | 0.9512 |
| 0.90 | 0.1216 | 0.9900 | 0.8179 |

表 D：预算与 pass^k 的口径

| 项 | 计算 | 结果 |
| --- | --- | --- |
| 最坏墙钟 | $2+20\times30$ | 602 s，是 300 s 整轮预算的 2.0 倍 |
| 300 s 可跑步数 | 平均每步 12 s | 25 步 |
| τ-retail 独立假设 | $0.612^8$ | 2.0%（观测约 25%，差 12.7 倍） |

```python
# 复算本页全部数字：工具开销、分层收益、累积成功率、重试、预算、pass^k。无第三方依赖。
T_BAR = (291 + 181 + 199) / 3                     # 223.67 tok：三个「良好」schema 的均值
R_PRE = 989e12 * 0.4 * 8 / (2 * 70e9)             # 22,606 tok/s：70B bf16 TP=8 on 8xH100、MFU 0.4
CONN, TOOL_T, STEP_T, WALL, MAX_STEPS = 2, 10, 30, 300, 20

def chain(p, k):                                  # 每步独立、不改写策略时的端到端成功率
    return p ** k

def affordable(elapsed, step, est=5):             # 剩余预算守卫：不够就不要发这次调用
    return step < MAX_STEPS and elapsed + min(est, TOOL_T, STEP_T) <= WALL

for n in (5, 10, 20, 50):                         # 表 A：常驻开销
    one = round(n * T_BAR)
    print(f"n={n:2d} 单轮 {one:6d} tok  20 步 {round(n * T_BAR * 20):7d} tok  TTFT +{one / R_PRE * 1000:3.0f} ms")

full = 50 * T_BAR                                 # 表 B：分层收益
for name, v in (("两阶段", 8 * 60 + 8 * T_BAR), ("检索式 top-5", 5 * T_BAR + 120)):
    print(f"{name} {round(v):5d} tok（省 {100 * (1 - v / full):.1f}%）")

for p in (0.99, 0.95, 0.90):                      # 表 C：复利与重试
    eff = 1 - (1 - p) ** 2
    print(f"p={p:.2f} 20 步 {chain(p, 20):.4f} -> 重试一次后单步 {eff:.4f}、20 步 {chain(eff, 20):.4f}")

print(f"最坏墙钟 {CONN + MAX_STEPS * STEP_T} s，是整轮 {WALL} s 的 {(CONN + MAX_STEPS * STEP_T) / WALL:.1f} 倍")
print(f"剩余 280 s 时是否还发一次预估 5 s 的调用：{affordable(280, 19)}；剩余 298 s 时：{affordable(298, 19)}")
print(f"tau-retail pass^1=0.612 的独立假设 pass^8 = {0.612 ** 8:.4f}，观测约 25%，差 {0.25 / 0.612 ** 8:.1f} 倍")
```

严格模式下的 schema 要按可编译子集写，业务约束留给运行时：

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "query": {"type": "string", "description": "自然语言检索词，用文档中可能出现的术语，不要布尔语法。"},
    "scope": {"type": "string", "enum": ["all", "engineering", "product", "people_ops"]},
    "top_k": {"type": ["integer", "null"], "description": "返回片段数，null 表示默认 5"}
  },
  "required": ["query", "scope", "top_k"]
}
```

`top_k` 写成 nullable union 并强制进 `required`，是严格模式的典型写法；`minimum`/`maximum` 这类关键字支持度不一，把「top_k 不得超过 20」放到 runtime 校验里更稳。

与源文对照：Outcome School 的两篇文章给出的是机制描述——模型不执行函数、只输出结构化请求，runtime 执行并把结果回灌，形成 agent 循环——没有给 token 成本、累积成功率或超时预算的数字。Anthropic 的两篇给出的是工具设计方法论（把描述当 prompt 工程化、返回高信号字段、按 token 效率优化响应、用评测而非直觉决定工具集），可引用的计量只有 `detailed` 对 `concise` 响应的 206 对 72 tok，以及 Claude Code 默认把工具返回值限制在 25,000 tok 这类工程阈值。τ-bench 提供 pass^k 的定义与 τ-retail 的 61.2% / pass^8 < 25% 这组数字。本页的 $\bar t$、prefill 吞吐、分层方案、超时预算与全部表格均为自行复算，$\bar t$ 与 $R_{pre}$ 是标定参数而不是实测常数。

## 常见追问

- **追问**：约束解码会不会降低模型质量？
  - 要点：会，机制有两层。① 每步只在受限支撑集上重归一化，改变了路径偏好，长推理被压成字段填充；② 结构化字段把数字、代码与推理内容变成字符串，token 化方式偏离预训练分布。工程结论是「推理自由、结论受限」：让模型先在自由文本里推理，最后一步再产出受约束的调用或结果（[[agents-03]]）。
- **追问**：工具调用失败时，怎么区分「该重试」和「重试也没用」？
  - 要点：按错误类别分派，而不是按异常类型猜。429/5xx/连接超时是瞬时故障，退避重试；schema、枚举、缺必填是参数错误，回灌可操作校验信息让模型修一次；未找到、无权限、空结果是语义失败，默认不重试，改策略或转人工。分类判据必须落在 runtime 里而不是提示词里，否则模型会把「无权限」当成瞬时故障反复重试，把预算烧在不可能成功的调用上（[[agents-02]]）。
- **追问**：工具超时之后，怎么知道它到底生效了没有？
  - 要点：默认当作「结果未知」。写操作必须带幂等键（首次尝试前生成、重试复用），或先按业务键查状态：查到就取回结果，查不到才重放，查不到还无法判定就转人工。把超时当失败直接重放是重复下单的根因。
- **追问**：为什么不能把 pass^k 写成 $({\text{pass}^1})^k$？
  - 要点：那等于假设所有任务同质。$x^k$ 在 $[0,1]$ 上凸，Jensen 不等式给出 $\mathbb{E}[x^k]\ge(\mathbb{E}[x])^k$；τ-retail 上 $0.612^8=2.0\%$，观测却是 25% 左右，差 12.7 倍——差距本身就是「失败任务是分层的」这一诊断信号，应转向分析哪些任务必失败（[[agents-12]]）。
- **追问**：上下文被工具返回值撑爆怎么办？
  - 要点：三件事一起做——工具侧只返回模型需要的字段（id + 摘要），runtime 按 token 预算截断并显式标注，同时提供分页与「按 id 取详情」的工具。10 条带摘要的检索结果已是 584 tok，原始网页可达数万 token；静默丢弃会被模型当成事实缺失。
- **追问**：自部署 open-weight 权重时，怎么保证工具调用合法？
  - 要点：先确认推理栈是否支持 guided decoding（FSM/GBNF 一类），支持就把 schema 编译进去；不支持就必须把校验前置为运行时硬门槛，并把校验失败当作可修复错误回灌。两种形态都要保留同一套运行时校验与错误分类，因为「约束解码能保证的东西」比「业务需要保证的东西」少得多。

## 相关题目

- [[agents-03]] — 结构化输出与 function calling 的区别：同源机制与两条编译路径
- [[agents-05]] — 工具数量多少算太多、schema 怎么设计：候选枚举规模与常驻 token 开销
- [[agents-02]] — 工具调用错误、超时与重试：错误分类、幂等键与分层预算的落地细节
- [[agents-09]] — 正确终止与成本步数限制：外部验证器、部分完成与 pass^k
- [[agents-08]] — agent 如何决定调用工具还是凭自身知识回答
- [[agents-10]] — 让 agent 的操作可逆或可审计：幂等键、意图日志与审计轨迹
- [[agents-12]] — 长时间运行后的漂移诊断：误差累积与 pass^k 的口径

## 参考资料与归属

- [LLM 中的 Function Calling 是如何工作的？](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms) — Amit Shekhar (Outcome School)，2026-06-13。模型只输出结构化请求、runtime 执行并回灌、工具定义与 JSON mode 的关系取自该文。
- [AI Agent Loop](https://outcomeschool.com/blog/ai-agent-loop) — Amit Shekhar (Outcome School)，2026-05-28。agent 循环的步骤划分与循环终止的描述取自该文。
- [Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)（延伸）— Anthropic，2025-09-11。工具描述工程化、返回高信号字段、按 token 效率优化响应、用评测决定工具集等结论来自该文。
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)（延伸）— Anthropic，2024-12-19。工具与 agent 循环的工程分层、把复杂度留给必要之处的判断来自该文。
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)（延伸）— Yao et al.，2024-06-17。pass^k 的定义、τ-retail 的 $\text{pass}^1=61.2\%$ 与 $\text{pass}^8$ 不足 25% 来自该论文。

本页的 token 计量、分层收益、累积成功率与重试折算、超时预算表、运行时校验与幂等键的实现要点，属于自行推导与本地复算（脚本口径见「数值与代码验证」），不在上述参考源覆盖范围内；公司场景部分（托管 API 与自部署权重在约束解码能力上的差别）是工程判断，不引自官方文档。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
