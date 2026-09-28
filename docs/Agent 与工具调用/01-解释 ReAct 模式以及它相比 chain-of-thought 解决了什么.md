---
type: question
id: agents-01
topic: Agent 与工具调用
order: 1
question: 解释 ReAct 模式，以及相比单纯的 chain-of-thought 它解决了什么问题。
question_en: Explain the ReAct pattern and what it solves compared with plain chain-of-thought.
asked_at: []
level: 进阶
tags: [react, cot, 工具调用, 循环]
sources:
  - title: ReAct Agent
    url: https://outcomeschool.com/blog/react-agent
    author: Amit Shekhar (Outcome School)
    published: 
  - title: How does Chain-of-Thought (CoT) Prompting work?
    url: https://outcomeschool.com/blog/how-does-chain-of-thought-prompting-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: ReAct: Synergizing Reasoning and Acting in Language Models（延伸）
    url: https://arxiv.org/abs/2210.03629
    author: Yao et al. (ICLR 2023)
    published: 2022-10-06
  - title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models（延伸）
    url: https://arxiv.org/abs/2201.11903
    author: Wei et al. (NeurIPS 2022)
    published: 2022-01-28
related: [agents-02, agents-03, agents-08, rag-06]
updated: 2026-09-28
---

## 一句话答案

> CoT 让模型在给出答案前把中间推理步骤写出来：把「一次前向就要算完」换成「多写若干 token、多做几步计算」，在算术、常识与符号推理上显著提升准确率。它的结构性局限是只使用参数化知识、无法获取或改变外部状态，而且推理链是单向的静态黑箱——某一环错了，错误会顺着链条继续传播，没有外部信号来纠正。
> ReAct 把推理与行动**交错**生成，循环是 Thought → Action → Observation：模型只产出文本，由 runtime 解析并执行动作、把真实结果回灌进上下文，模型在下一步推理里读到它。它因此换来三件事：外部与时效信息、用观察纠正推理、人类可读可审计的轨迹。
> 论文口径：ALFWorld 与 WebShop 上，ReAct 只用 1–2 个 in-context 示例，绝对成功率就比用 10³–10⁵ 条任务训练出来的模仿学习 / RL 基线高 34% 与 10%；HotpotQA 上 CoT 的失败案例里 56% 是幻觉，ReAct 是 0%。两者是互补关系——纯推理任务上 CoT 更直接，需要外部事实、精确计算或副作用时才值得上循环。

## 面试官在考什么

- 能不能把 CoT 与 ReAct 放在同一根轴上比较：两者都是「把推理外化成 token」，区别在于推理链里有没有插入由 runtime 执行、会把真实结果回灌的动作。
- 是否清楚循环里的分工：模型只产出文本，解析、执行、回灌、终止判断全是 runtime 的职责。说不清这一点的人，遇到工具报错、步数上限、上下文爆炸时会归因错对象。
- 能否逐条说出 CoT 的三个结构性局限，并对应到 ReAct 的三个收益（外部信息、纠错、可审计），而不是笼统地说「ReAct 更强」。
- 是否知道两者互补：ReAct 在 HotpotQA 的 EM 上比 CoT 低 2.0 个百分点，在 FEVER 上却高 4.6 个百分点；论文里最好的成绩来自两种模式互相切换。
- 是否了解工程演化：文本解析式的 ReAct 已被原生 function calling / 结构化输出取代，但循环本身仍是今天所有 agent 的主流程。

常见错误答案：

- 「ReAct = CoT + 工具」。漏掉 interleaved（交错）这个关键词：ReAct 的每一次推理都发生在读到上一步观察之后，而不是先写完一整条推理链再去调工具。
- 「ReAct 全面优于 CoT，所以任何任务都该套循环」。纯推理、封闭可判定的任务上，循环只是把一次调用变成多次调用，token 与延迟都上去；论文自己的数据也不支持这种说法（HotpotQA 上 CoT 29.4、ReAct 27.4）。

## 原理与推导

### 1. CoT：把推理外化成 token

CoT prompting（Wei et al., NeurIPS 2022）把 few-shot 示例从 ⟨输入, 输出⟩ 换成 ⟨输入, 思维链, 输出⟩ 三元组：示例里不只有答案，还有通向答案的中间步骤，模型照着这个格式在回答前先写推理。论文的实验口径是 5 个数学应用题 benchmark（GSM8K、SVAMP、ASDiv、AQuA、MAWPS）加常识与符号推理任务，模型覆盖 GPT-3、LaMDA、PaLM、UL2、Codex；数学题用人工编写的 8 条 CoT 示例（AQuA 是选择题，改用 4 条），解码用 greedy。

论文报告的结论是：PaLM 540B 加 8 条示例在 GSM8K 上取得当时的 SOTA，超过带 verifier 的微调 GPT-3。论文图 2 上四个柱子的标注值是 PaLM 540B 标准 prompting 18、PaLM 540B + CoT 57、prior best（微调 GPT-3 + verifier）55、微调 GPT-3 175B 33，即 CoT 相对标准 prompting 抬升 39 个百分点、比 prior best 高 2 个百分点。

为什么有效，机制上有两点：

- 自回归生成让已写出的 token 成为后续生成的条件。推理步骤一旦落到上下文里，就不再是内部隐状态，而是模型下一步必须读到的输入——相当于把一个固定深度的前向计算摊到更多次前向里。论文的表述是：思维链让模型把多步问题拆成中间步骤，从而给需要更多推理步骤的问题分配更多计算。
- 每一步只做一件小事，单步难度远低于直接跳到答案；n 个小步骤串起来，能处理的问题复杂度就上去了。

论文还列了另外三条性质：思维链是可解释的窗口（能看出错在哪一步）、适用于人类可以用语言求解的任务、并且对足够大的模型只要 prompting 就能激发。最后一条有前提：CoT 是随模型规模涌现的能力，论文观察到它对小模型没有正收益，只有到约 100B 参数以上才稳定获益。

### 2. CoT 的三个结构性局限

ReAct 论文对 CoT 的判断很直接：CoT 是一个 static black box，Thought 由模型的内部表示生成、并不 grounded 在外部世界里，因此限制了它 reactive 地推理、以及更新自己知识的能力。展开成三条：

1. **只用参数化知识。** 模型知道的只有训练数据里的东西。训练截止之后的、私有的、或者长尾到没被记住的事实，要么答不出，要么编一个看起来合理的答案出来。
2. **不能获取或改变外部状态。** 精确算术、查库、读文件、下单、发邮件都需要与外部世界交互，纯文本推理链做不到，只能「猜」。
3. **错误沿链条传播，且没有外部信号纠正。** 链条是单向的：第 3 步错了，第 4 步以错误结论为前提继续推，错误被放大而不是被发现。论文在 HotpotQA / FEVER 上正是把 hallucination 与 error propagation 列为 CoT 的主要问题。

第 3 条不是靠「让模型写仔细一点」能解决的：模型没有可对照的事实来源，也没有第二个信号告诉它上一步错了。

### 3. ReAct 的循环：模型只产出文本

ReAct（Yao et al., 2022-10 / ICLR 2023）把动作空间扩大成 $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$：除了环境动作，模型还可以在语言空间 $\mathcal{L}$ 里产出 thought。两者的区别是关键：

- **thought 不影响外部环境，因此不产生 observation**，它只是把有用的信息写进上下文，供后续推理或动作使用：拆解目标、注入常识、从观察里抽关键句、跟踪进度、处理异常、改写检索词。
- **action 作用在外部环境上**，会返回 observation，也就是来自真实世界的反馈。

一条 ReAct 轨迹因此长这样：

```text
Question:      公司 A 去年的营收是公司 B 的几倍？

Thought 1:     需要两家公司去年的营收，先查 A。
Action 1:      search["公司 A 2025 年营收"]
Observation 1: 公司 A 2025 年营收 128 亿元（来源：年报摘要）
Thought 2:     拿到 A 了，还要 B 的营收才能算倍数。
Action 2:      search["公司 B 2025 年营收"]
Observation 2: 检索结果为空。
Thought 3:     换个更精确的检索词再试一次。
Action 3:      search["公司 B 年报 营业收入"]
Observation 3: 公司 B 2025 年营业收入 32 亿元。
Thought 4:     128 / 32 = 4，用计算器确认一下。
Action 4:      calculator["128 / 32"]
Observation 4: 4.0
Thought 5:     两个数都来自年报，可以给结论了。
Final Answer:  公司 A 去年营收是公司 B 的 4 倍。
```

论文里的最小可用动作空间就是 Wikipedia API 的三个动作：`search[entity]`（返回该条目开头 5 句，条目不存在就给 5 个相近实体）、`lookup[string]`（返回页面里下一句包含该串的句子，相当于浏览器 Ctrl+F）、`finish[answer]`（结束并给出答案）。作者特意说明这个动作空间比当时主流的词法/神经检索器弱得多，目的是模拟人查维基的过程，逼模型用显式推理决定查什么。

**这个循环里最重要的一条分工：模型只产出文本。** 它写出 `Action: search[...]` 这串字符；解析这行文本、判断工具存不存在、真正执行、把返回值包装成 `Observation:` 回灌进上下文，全部是 runtime（论文里是 environment）在做。模型从不执行任何东西，它只是在下一步读到执行结果。后面所有工程讨论——解析失败、工具报错、超时重试、步数上限、上下文增长——都是这条分工的直接后果。

论文的 few-shot 口径：HotpotQA 用 6 条人工编写的 ReAct 轨迹作示例、FEVER 用 3 条（作者发现示例更多并不提升效果）；决策类任务（ALFWorld、WebShop）里 thought 是稀疏出现的，由模型自己决定什么时候想。

### 4. ReAct 相对 CoT 解决了什么

**① 能获取外部信息与时效数据。** 需要的事实不再依赖模型记没记住，而是去查；论文附录专门给了 ReAct 通过检索拿到比 HotpotQA 标注更新的答案的例子。

**② 观察可以纠正推理。** 这是题干第二问的核心。CoT 的失败是单向累积的；ReAct 每一步都插入一个来自外部、可验证的观察，下一步推理以真实结果为前提，错误不会无声地往下传。论文在 HotpotQA 上各抽 50 条 ReAct 与 CoT 轨迹（含答对与答错）做人工标注，结果如下：

| 类别 | ReAct | CoT |
| --- | --- | --- |
| 成功：推理链与事实都正确 | 94% | 86% |
| 成功：推理链或事实有幻觉 | 6% | 14% |
| 失败：推理错误（含陷入重复） | 47% | 16% |
| 失败：检索结果错误（空结果或无有用信息） | 23% | — |
| 失败：幻觉 | 0% | 56% |
| 失败：标注歧义 | 29% | 28% |

口径：成功块与失败块各自在块内归一化，所以每块的两三行加起来是 100%（ReAct 失败块 47+23+29=99%，为原文取整）。这张表给出两个方向的结论：CoT 的失败主要由幻觉构成（56%），ReAct 的幻觉是 0%，代价是把 23% 的失败换成了「检索没查到有用信息」——外部反馈的质量直接决定循环的质量。

**③ 轨迹可读、可审计。** Thought、Action、Observation 都是自然语言，人可以直接看出模型为什么这么做，也能区分哪些信息来自模型内部、哪些来自外部环境。论文把 interpretability / trustworthiness / diagnosability 列为与性能并列的收益，并演示了人类在轨迹中途改写 thought 来纠正 agent 的行为。

### 5. 互补，而不是替代

论文主表用 PaLM 540B 对比了各种 prompting 方法（HotpotQA 报 EM，FEVER 报 Acc）：

| Prompt 方法 | HotpotQA (EM) | FEVER (Acc) |
| --- | --- | --- |
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC（自洽性，21 次采样投票） | 33.4 | 60.4 |
| Act（只有动作，没有 thought） | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC → ReAct | 34.2 | **64.6** |
| **ReAct → CoT-SC** | **35.1** | 62.0 |
| Supervised SoTA | 67.5 | 89.5 |

读法有三层：

- ReAct 高于 Act（HotpotQA +1.7、FEVER +2.0），说明推理确实在指导动作，而不只是多了几次工具调用。
- ReAct 与 CoT 互有胜负：FEVER 上 +4.6，HotpotQA 上 −2.0。需要精确、最新的事实（FEVER 的 SUPPORTS / REFUTES 可能只差一点措辞）时，去查比回忆可靠；而需要把多个线索组织成推理结构时，CoT 更灵活——论文的失败模式分析里也承认，交错结构这个约束本身会抬高推理错误率（47% vs 16%）。
- 最好的成绩来自组合：ReAct 在给定步数内没给出答案就退回 CoT-SC（论文设 HotpotQA 7 步、FEVER 5 步），CoT-SC 的多数答案出现次数少于采样数一半就退回 ReAct（说明内部知识不可信）。论文还报告这两种组合用 3–5 次采样就能达到 CoT-SC 用 21 次采样的水平。

判断依据可以压成一句话：**下一步需要的输入，是不是只有执行上一步之后才知道？** 是，就必须有循环；否，一次 CoT（或一次带检索的调用）通常更快更便宜。任务完全由参数化知识决定且答案可判定时——改写、分类、格式转换、纯数学推导——循环只是徒增调用次数。agentic RAG 用的是同一条判据（[[rag-06]]）。

### 6. 失败模式

- **动作解析失败。** 论文用纯文本格式，靠 few-shot 示例约束；模型可能写出不存在的工具名、参数类型不对、或多个动作粘在一行。文本解析在工程上早已被原生 function calling / 结构化输出取代（[[agents-03]]）。
- **在不需要外部信息的任务上多绕几圈。** 模型会把「再确认一下」写进 thought 然后去查，步数、token、延迟一起涨，正确率未必提升。
- **一次错误的观察把后续带偏。** 论文把 non-informative search 列为 ReAct 的主要失败模式之一（占错误案例 23%）：检索没查到有用信息会让模型在错误前提上继续推理，且不容易跳出来。
- **重复循环。** 论文观察到一种 ReAct 特有的错误模式：模型反复生成上一步已经生成过的 thought 与 action，跳不出局部循环。小模型在纯 prompting 设置下尤其明显——PaLM-8/62B 上 prompting 版 ReAct 是四种方法里最差的，而用 3000 条正确轨迹微调后它变成最好的（62B 微调后超过所有 540B prompting 方法）。工程上对应步数上限、重复动作检测与终止条件（[[agents-09]]）。
- **few-shot 轨迹设计决定稳定性。** 示例里的动作空间、格式、停止条件都会被模仿，示例与真实工具集不一致时行为会退化。

### 7. 从论文到生产：保留循环，换掉解析

变体的差别集中在「思考的时机与粒度」这一根轴上：

- **ReAct**：边想边做，thought 与 action 交错，粒度最细，适应性最好，调用次数最多。
- **Plan-and-Execute**：先把计划一次性写出来再执行，thought 粒度最粗、调用最省，但计划建立在不完整信息上，执行中发现不对要整条重规划。
- **Reflexion**：把「想」挪到失败之后，用语言化的反思写进记忆，下次尝试时带上，改的是跨尝试的学习而不是单次轨迹。
- **ReWOO**：把推理与观察解耦，先一次写出带变量占位符的计划，再并行取数、最后统一求解，省掉中间轮次的上下文重发，代价是丢掉逐步纠错。

生产里被替换掉的只有动作的表达与解析方式：原生 function calling、JSON schema、约束解码把动作变成受语法约束的字段。循环本身（读上下文 → 决定下一步 → 执行 → 回灌 → 直到终止）仍是主流程，工具错误与重试是它的失败分支（[[agents-02]]），记忆与上下文管理是它的长期运行分支（[[agents-07]]）。

## 数值与代码验证

**表 1：单步正确率与链条长度**

口径：把每一步（一次推理或一次动作）视为独立、正确率恒为 $p$ 的事件，则 $n$ 步全部正确的概率是 $p^n$。这是解释错误传播的简化模型，不是任何 benchmark 的实测结果——真实推理链的步骤相关，且一步错不等于任务失败，它只用来给出量级直觉。

| 单步正确率 $p$ / 步数 $n$ | 5 步 | 10 步 | 20 步 | 50 步 |
| --- | --- | --- | --- | --- |
| 0.95 | 77.38% | 59.87% | **35.85%** | 7.69% |
| 0.99 | 95.10% | 90.44% | 81.79% | 60.50% |
| 0.999 | 99.50% | 99.00% | 98.02% | 95.12% |

看 0.95 这一行：单步 95% 已经很乐观，20 步之后只有 35.85% 的概率整条链没出错，50 步掉到 7.69%。这解释了「让 CoT 想得更久」的收益上限——链越长，累积成功率越低；要跑到 20 步还有 80% 以上，单步正确率得在 0.99 附近（$0.99^{20} = 81.79\%$）。

ReAct 不改变 $p$，它改变的是「错误能不能被拦住」：把不可验证的自由推理切成「推理 + 由外部执行的动作」，中间插入的观察是对事实的一次重新锚定。这正是论文表 2 里 CoT 的失败 56% 来自幻觉、而 ReAct 幻觉为 0% 的直观解释。

**表 2：循环的 token 成本**

口径：基础 prompt（system 提示 + 3 个工具 schema + 用户问题）记 $P = 1000$ token；每一步循环新增的上下文（Thought 60 + Action 20 + Observation 150）记 $D = 230$ token。第 $k$ 次调用（$k = 1,\dots,n$）的输入长度是 $P + (k-1)D$——第 1 次只带基础 prompt，之后每多一步就多带一份 Thought / Action / Observation。于是 $n$ 步累计输入是 $nP + D\,n(n-1)/2$，随步数二次增长。

| 步数 $n$ | 累计输入 token | 平均每次调用 | 相当于单次 CoT 调用（1000 token） |
| --- | --- | --- | --- |
| 1 | 1000 | 1000.0 | 1.00× |
| 5 | 7300 | 1460.0 | 7.30× |
| 10 | 20350 | 2035.0 | 20.35× |
| 20 | 63700 | 3185.0 | 63.70× |

两个读法：步数从 10 涨到 20，累计输入从 20350 涨到 63700，是 3.13 倍而不是 2 倍，因为每一步都要重发全部历史；工具定义是固定开销，按 3 个 schema 约 180 token 估，在 20 步里被重复发送 20 次，光这一项就是 3600 token——工具越多，这个固定项越大（[[agents-05]]）。

**代码 1：ReAct 循环骨架**（模型只产出文本，解析 / 执行 / 回灌 / 终止判断都在 runtime 这一侧）

```python
REACT_SYSTEM_PROMPT = """你可以使用以下工具：
- search(query): 搜索外部资料
- calculator(expr): 计算数学表达式

每一步先用 Thought 写下推理，然后二选一：
  Action: <tool_name>(<input>)     # 需要外部信息或计算时
  Final Answer: <answer>           # 已经有答案时
每次只输出一个 Action，等待 Observation 之后再继续。"""


def run_react(question, tools, call_llm, max_steps=10):
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        reply = call_llm(messages)              # 模型输出 Thought + Action 或 Final Answer
        messages.append({"role": "assistant", "content": reply.text})

        if reply.final_answer is not None:      # 模型自己宣布结束
            return reply.final_answer

        try:
            observation = tools[reply.action_name](**reply.action_args)
        except KeyError:
            observation = f"工具 {reply.action_name} 不存在，可用：{sorted(tools)}"
        except Exception as exc:                # 工具报错也要变成观察，别让异常打断循环
            observation = f"工具执行失败：{exc!r}"

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "达到步数上限仍未给出答案"
```

这段代码里没有任何「智能」在循环控制之外：动作解析失败、工具抛异常、模型重复同一个动作，都要在这里变成观察或终止条件。这几条支路就是 [[agents-02]] 与 [[agents-09]] 的全部内容。

**代码 2：表 1 与表 2 的复算**

```python
def chain_success(p, n):
    return p ** n

for p in (0.95, 0.99):
    print(p, [round(chain_success(p, n), 4) for n in (5, 10, 20, 50)])
# 0.95 [0.7738, 0.5987, 0.3585, 0.0769]
# 0.99 [0.951, 0.9044, 0.8179, 0.605]

P, D = 1000, 230   # 基础 prompt；每步新增 Thought + Action + Observation
for n in (5, 10, 20):
    total = n * P + D * n * (n - 1) // 2
    print(n, total, round(total / n), round(total / P, 2))
# 5 7300 1460 7.3
# 10 20350 2035 20.35
# 20 63700 3185 63.7
```

与源文对照：论文的实测数字（ALFWorld / WebShop 的 +34% / +10%、主表 1、失败模式表 2）直接引用原文；表 1 的 $p^n$ 与表 2 的 token 成本是本节自建模型，只用于解释机制，不是论文结果。CoT 论文图 2 的 GSM8K 读数是四舍五入后的整数标注（540B + CoT 57、prior best 55、standard 18、微调 GPT-3 175B 33），精确表值以原文附录为准。

## 常见追问

- **追问**：ReAct 为什么比「先规划再执行」更鲁棒？
  - 要点：计划是在信息不完整时做的，而循环每一步都能用上一步的观察修正后续。论文把这归为 reason to act：模型一边做一边根据当前状态决定下一个子目标，而不是一次押注整条计划，计划错了下一步就能改，不会整条作废。代价是每一步都要一次模型调用，规划式方案可以把规划成本摊薄。判断依据仍是那句话——下一步的动作是否依赖上一步的结果。
- **追问**：为什么生产环境很少直接解析 Thought/Action 文本？
  - 要点：文本格式没有约束，模型可能写出不存在的工具、错误参数、多个动作粘在一起，解析失败要靠重试兜底，而重试本身又是一轮 token 与延迟。原生 function calling / 结构化输出用 JSON schema 或约束解码把动作变成受语法约束的字段，工具名与参数在生成阶段就被限制在合法集合内，服务端再做一次 schema 校验。ReAct 的循环结构不变，换掉的只是动作的表达与解析方式（[[agents-03]]）。
- **追问**：ReAct 与 agentic RAG 是什么关系？
  - 要点：agentic RAG 就是 ReAct 循环在检索场景的实例化：把 action 换成检索工具，thought 决定查什么和要不要再查，observation 是检索结果。判据也一致——下一步的检索是否依赖上一步的结果；没有依赖时，一轮并发检索加重排更便宜（[[rag-06]]）。
- **追问**：轨迹越跑越长怎么办？
  - 要点：上下文随步数线性增长，累计输入 token 是二次的。常见手段是只保留最近若干步的完整观察、把更早的轨迹压成摘要、对大块观察做截断或外置成按需取回的引用，并配合步数上限与工具预算。注意 thought 通常比 observation 便宜得多（几十 vs 几百 token），该丢的是旧观察，不是推理链（[[agents-07]]）。
- **追问**：既然 CoT 在 HotpotQA 上比 ReAct 高，是不是说明 ReAct 没用？
  - 要点：单看一个 benchmark 会得出反结论。两者在 HotpotQA 上相差 2.0 个百分点，而同一模型在 FEVER 上 ReAct 高 4.6 个百分点；到了 ALFWorld、WebShop 这类决策任务，CoT 根本没有「写下推理就能得到动作」的路径，ReAct 相对训练充分的模仿学习 / RL 基线高 34% 与 10%。论文的结论是组合最好，而不是某一方胜出。
- **追问**：为什么不把所有任务都做成循环？
  - 要点：循环的成本结构是每次调用都重发全部历史，n 步的累计输入是二次的，延迟按步数线性叠加；同时每多一步就多一次动作解析与工具报错的机会。改写、分类、结构化抽取这类一步可完成的任务进循环只增加成本和失败面。工程上的做法是按任务路由：能一次做完的不进循环，需要外部事实或副作用时才进。

## 相关题目

- [[agents-02]]：agent 循环中如何处理工具调用错误、超时与重试——本题第 3 节「runtime 负责执行与回灌」的全部分支都在那里。
- [[agents-03]]：结构化输出与 function calling 的区别——文本解析式 Action 的替代方案，是第 7 节演化路径的另一半。
- [[agents-08]]：agent 如何决定调用工具还是依据自身知识回答——ReAct 与 CoT 的分界落在单步上就是这个决策。
- [[rag-06]]：agentic RAG 与标准 RAG 的差别——ReAct 循环在检索场景的实例化，成本与延迟的算法可以互相印证。

## 参考资料与归属

1. [ReAct Agent](https://outcomeschool.com/blog/react-agent)，Amit Shekhar（Outcome School），2026-04-30。提供 ReAct 的五个组成部分（LLM、system prompt、tools、memory、loop controller）、Thought → Action → Observation 的循环图与完整 trace 示例、prompt 模板，以及从循环视角整理的常见失败模式（无限循环、选错工具、幻觉工具调用、上下文爆炸、过早停止、工具报错后卡死）。
2. [How does Chain-of-Thought (CoT) Prompting work?](https://outcomeschool.com/blog/how-does-chain-of-thought-prompting-work)，Amit Shekhar（Outcome School），2026-07-11。提供 CoT 的直觉（逐 token 自回归生成为什么让「写出步骤」有用）、zero-shot 与 few-shot CoT 的区别、适用任务，以及「不是所有问题都需要 CoT」「推理步骤不能全信」「推理步骤会增加时间与成本」这几条注意事项。
3. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)（延伸），Yao et al.（ICLR 2023），2022-10-06。第 2 节对 CoT 是 static black box、不 grounded 在外部世界的判断，第 3 节的 $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ 与 thought / action 之分、Wikipedia 动作空间与 few-shot 示例数量，第 4 节的 HotpotQA 成功与失败模式表、非信息性检索占 23% 的失败，第 5 节的主表全部数字与组合切换策略，第 6 节的重复生成失败模式，以及「只用 1–2 个 in-context 示例在 ALFWorld / WebShop 上绝对成功率 +34% / +10%」，都来自该文摘要与正文。
4. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)（延伸），Wei et al.（NeurIPS 2022），2022-01-28。8 条 CoT 示例（AQuA 用 4 条）、540B 模型在 GSM8K 上取得当时 SOTA 并超过带 verifier 的微调 GPT-3、CoT 是随规模涌现的能力、以及论文对 CoT 四条性质的总结来自该文；图 2 的四个读数在正文中已标注为图上标注值而非精确表值。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
