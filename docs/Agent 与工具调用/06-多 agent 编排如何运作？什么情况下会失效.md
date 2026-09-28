---
type: question
id: agents-06
topic: Agent 与工具调用
order: 6
question: 多 agent 编排是如何运作的？什么情况下会失效？
question_en: How does multi-agent orchestration work, and when does it fail?
asked_at: [Cognition]
level: 高阶
tags: [多-agent, 编排, 失败模式, mast]
sources:
  - title: Multi-Agent Systems
    url: https://outcomeschool.com/blog/multi-agent-systems
    author: Amit Shekhar (Outcome School)
    published: 
  - title: AI Orchestration
    url: https://outcomeschool.com/blog/ai-orchestration
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Don't Build Multi-Agents（延伸）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: Why Do Multi-Agent LLM Systems Fail?（延伸）
    url: https://arxiv.org/abs/2503.13657
    author: Cemri et al.
    published: 2025-03-17
related: [agents-05, agents-09, agents-12, agents-07]
updated: 2026-09-28
---

## 一句话答案

> 多 agent 编排 = 把一次任务切成若干各有独立上下文、独立提示与工具集的 LLM 循环，再由协调层决定「谁在什么时候、拿什么输入、产出交给谁、谁来判合格」。值得付这个复杂度的理由只有三个，而且都可度量：上下文隔离（把长上下文切成多个短上下文）、并行（独立子任务压缩墙钟时间）、专精（不同角色用不同提示、工具集甚至不同价位的模型）。
> 它失效的地方正好是它的定义：决策被分散到互不可见的上下文里，各自做出的隐含约定互相冲突；合并者的上下文成为吞吐瓶颈；成本、延迟与失败面随「agent 数 × 轮数」放大——MAST 在 7 个主流框架的 1642 条轨迹上把这类失败归纳为 3 类 14 种（分类法出自其中 150 余条轨迹的扎根分析；论文口径）。

## 面试官在考什么

- **能不能把「多 agent」翻译成工程参数**：不是「像公司一样分工」的比喻，而是三个可度量的收益轴（上下文隔离 / 并行 / 专精）与三个成本轴（token、延迟、调试复杂度）。说不出收益轴的人，也无法说出失效条件。
- **是否理解可控性来自状态传递而不是角色数量**：agent 之间怎么传状态（点对点结构化消息 / 共享黑板 / 共享文件系统 / 任务队列），每个子 agent 看到多少历史，谁负责合并与裁决冲突——这三问决定了系统能不能调试、能不能重放。
- **知不知道反方论证及其前提**：Cognition《Don't Build Multi-Agents》反对的默认架构是「并行子 agent 各自只看局部上下文」，它的前提是长程、决策相互依赖的任务；对能真正解耦、可并行验证的任务（批量处理大量互不相干的文件），多 agent 的收益依然成立。能划出这条边界才算读懂，而不是站队。
- **失败模式有没有实证口径**：MAST 的 3 类 14 种、κ=0.88 是论文的标注口径，不是「经验之谈」；能把失败模式对到自己系统里的具体位置（规格、角色、终止条件、验证），才有资格谈对策。
- **会不会算账与验证**：token 与延迟怎么随编排规模长（第 4 节给公式与复算），以及用什么指标证明「多 agent 比单 agent 好」——pass@k 与 pass^k 的区别，固定预算下的 A/B（[[agents-09]]）。

**常见错误答案**

- 「多 agent 更强，因为每个 agent 只做一件事，提示更短、工具更聚焦。」——这是把「专精」当成必然收益。专精只有在子任务可独立定义、且合并成本低于收益时才成立；否则你只是把一次决策拆成 N 次互相看不见的决策，再把冲突留给合并者。
- 「并行跑子 agent 一定更快更准。」——并行只压缩墙钟时间，不压缩总 token（常常还放大）；准确率上，N 个 agent 各自独立选择约定时，一致性概率随 N 指数下降（第 4 节算过），返工与仲裁会把并行收益吃掉。

## 原理与推导

### 三个真实理由，各自对应一个可测的量

**① 上下文隔离。** 单线程 agent 第 $t$ 轮的输入上下文近似为 $H_t = H_0 + t\Delta$（$H_0$ 是系统提示 + 工具 schema + 任务，$\Delta$ 是每轮新增的助手消息与工具结果）。跑满 $T$ 轮的输入 token 总量是

$$
\sum_{t=0}^{T-1}\left(H_0 + t\Delta\right) = T H_0 + \Delta\,\frac{T(T-1)}{2}
$$

即 **token 随单个上下文的长度二次增长**。把一个长任务切成多个短上下文，这个二次项按各子任务的轮数分别计算，总量可以显著下降（见第 4 节表 4 的示例）。附带收益是每个窗口更短，缓解 lost-in-the-middle 式的注意力稀释。

**② 并行。** 墙钟时间约等于**关键路径上的串行调用次数**乘以单次延迟（忽略排队与生成长度差异）。$N$ 个彼此独立的子任务串行要 $N\cdot T_w$ 次调用，fan-out 后关键路径只剩 $T_w$ 次。注意这只对真正独立的子任务成立；一旦子任务之间有依赖，就会退化回串行，并额外付出协调开销（[[inference-serving-09]]）。

**③ 专精。** 不同角色可以配不同 system prompt、不同工具集，甚至不同模型：便宜的小模型做路由、抽取与格式转换，贵的大模型做规划与合并。工具定义本身要占 token——每个工具（name + description + JSON Schema 参数）在 cl100k_base 下大致 200 token 量级（良好 schema 实测均值 224，极简 44、冗长 465），一个带 12 个工具的 agent 每轮要为此付出约 2700 token；专精让这 2700 token 只出现在真正需要它的那个 agent 的上下文里，而不是全局常驻（[[agents-05]]）。

### 编排形态：控制权在谁手里

| 形态 | 控制流 | 适用 | 主要失效点 |
| --- | --- | --- | --- |
| 顺序流水线 | A→B→C，固定顺序 | 后一步确实需要前一步的产物 | 单点错误沿链条放大，且无法回退 |
| 路由（router / supervisor 分派） | 分类后选一个 worker | 请求能落到少数几类 | 路由判错，整条路径都错 |
| 并行 fan-out + 汇总 | N 个 worker 同时跑，再合并 | 子任务独立、结果可合并 | 合并者上下文成瓶颈；约定冲突 |
| 评审循环（producer–critic） | 生成→评审→退回修改 | 有独立判据、可验证的任务 | critic 无独立 grounding 时变橡皮图章；循环不收敛 |
| 层级树 | 编排者→worker→子 worker | 任务天然嵌套 | 每层都放大通信与 token；归因困难 |
| 图 / 网络 | 任意 agent 可调用任意 agent | 探索型任务 | 状态散落，难以重放与调试 |

从固定工作流（顺序、并行）到完全由模型决定走向（图），控制权逐级让渡给 LLM；控制权让渡得越多，可预测性与可调试性越差。多数生产系统的正确落点不是最右端，而是「固定骨架 + 局部自主」：主线由代码决定，个别步骤内部才交给 agent 自由发挥。

### 协作机制：状态怎么传、上下文怎么切、谁合并

协作机制有四种常见做法，它们的差别不在「有几个角色」，而在**可见性与可重放性**：

| 机制 | 谁能看到什么 | 合并/裁决方式 | 风险 |
| --- | --- | --- | --- |
| 点对点结构化消息 | 只有收发双方 | 由编排者拼接 | agent 多时消息组合爆炸 |
| 共享黑板 / 共享状态 | 所有 agent 读写同一份 | 后写者覆盖或显式合并 | 需要严格的写入规则，否则互相踩 |
| 共享文件系统 + 工作目录 | 产物落盘，任何人可读原始事实 | 读文件而不是读对话 | 需要目录约定与并发写保护 |
| 结构化任务队列 | 谁领任务谁看任务书 | 队列状态即事实 | 需要幂等与超时回收 |

三个必须显式回答的工程问题：

1. **上下文怎么切分。** 子 agent 只看任务书（最省 token，最容易误解需求）、还是看主线的完整 trace（最不易误解，最贵）、还是看压缩后的摘要（需要专门的压缩模型与评估）。Cognition 的 Principle 1 是「共享完整 trace，而不只是单条消息」；代价是上下文窗口，于是他们选择训练一个专门压缩历史动作与决策的模型。
2. **谁负责合并。** fan-out $N$ 路、每路产出 $|o|$ token，合并者至少要读 $N|o|$；$N$ 增大时合并者要么上下文溢出，要么被迫摘要（丢信息），要么退化成「重新做一遍任务」。合并者是这类架构的结构性瓶颈，不是实现细节。
3. **冲突谁裁决。** 两个 worker 对同一个接口做了不同假设时，必须有一个人（代码或指定 agent）能判定哪个为准。没有裁决权，系统就只能靠概率对齐。

### 反方：上下文不共享的地方，就是隐含决策冲突的地方

Cognition 的《Don't Build Multi-Agents》给出两条原则：**共享上下文与完整 trace，而不只是单条消息**；**每个动作都携带隐含决策，冲突的决策带来坏结果**。它的典型例子：把「做一个 Flappy Bird 克隆」拆成「背景与水管」「小鸟」两个子任务并行，两侧各自按自己的理解选了美术风格与操作手感，最后合并者拿到的是两个无法拼在一起的产物——即使把原始任务原文也塞给子 agent，也无法传递前面几轮工具调用中形成的那一堆隐含约定。

按这条论证，默认应该排除不满足这两条原则的架构，退回到**单线程线性 agent**：上下文连续，谁做的决定后面都看得到；上下文溢出时用压缩模型把历史动作与关键决策压成摘要。Cognition 同时给出两个旁证：Claude Code 的子 agent（截至 2025 年 6 月）不与主 agent 并行、通常只被派去回答问题而不是写代码，其价值是调查过程不留在主历史里；2024 年流行的 edit-apply 模型（大模型输出 markdown 修改说明、小模型重写文件）之所以故障率高，正是因为决策与执行被拆到两个模型后，指令里的微小歧义会被放大——现在更常见的做法是同一个模型一步完成。

**它的前提必须说清楚**：论证针对的是长程、决策相互依赖的任务（写代码是典型）。此时子 agent 的独立性是假象，任何局部决策都可能约束全局。反之，若任务可真正解耦、且产物可用机器验证——批量处理大量互不相干的文件、对 N 份文档各做一次抽取、对 N 个候选做互不影响的打分——前提不成立，fan-out 依然划算。判断标准是：**子任务的产物能否在不阅读对方上下文的情况下被独立验收**。

### 失败模式：MAST 的 3 类 14 种（论文口径）

MAST（*Why Do Multi-Agent LLM Systems Fail?*）的做法值得面试时引用：从 7 个主流 MAS 框架收集 **1642 条**执行轨迹（摘要里写作 1600+），先用 150 余条轨迹做扎根理论分析，3 位标注者迭代标注到一致性 $\kappa = 0.88$，得到 **14 种失败模式、3 大类**；再用 LLM-as-a-Judge 扩展到全量（与人类 $\kappa = 0.77$，在两个未见过的框架与基准上 $\kappa = 0.79$）。论文同时报告，这 7 个开源 MAS 的任务失败率区间是 **41%–86.7%**。以下模式名与编号来自论文附录的失败模式清单：

- **FC1 系统设计问题（System Design Issues）**：FM-1.1 违反任务规格、FM-1.2 违反角色规格、FM-1.3 步骤重复、FM-1.4 对话历史丢失、FM-1.5 不知道终止条件。
- **FC2 agent 间不对齐（Inter-Agent Misalignment）**：FM-2.1 对话重置、FM-2.2 不请求澄清、FM-2.3 任务跑偏、FM-2.4 信息扣留、FM-2.5 忽略其他 agent 的输入、FM-2.6 推理与行动不一致。
- **FC3 任务验证缺失（Task Verification）**：FM-3.1 过早终止、FM-3.2 无验证或不完整验证、FM-3.3 错误验证。

论文的结论是：这些失败大多源于系统设计，而不是单纯的模型能力限制，因而可以在系统层处理——把任务书与输出 schema 写死（对治 FC1）、把共享状态做成显式契约而不靠各自的记忆（对治 FC2）、引入带独立 grounding 的验证角色并在关键中间产物上校验（对治 FC3）。论文自身的干预案例也说明方向：ChatDev 里 CPO 角色违反了角色规格，绕过 CEO 直接终止对话，把「CEO 有最终决定权」写进 workflow 后，任务成功率提升了 **+9.4%**（论文口径）。论文的结论更保守：这类战术性修补只能拿到局部提升，可靠的 MAS 往往需要结构性重设计。

对策清单（按可落地性排序）：接口与分工显式化（任务书 + 输出 schema + 终止条件）、共享状态而非各自记忆、独立验证角色、限制通信轮数与总预算、全量记录轨迹以便归因（[[agents-10]]、[[agents-12]]）。

### 成本、延迟与调试复杂度

成本可以用一个显式的账本估：

$$
C=\sum_{a}\sum_{t}\left(H_{a,t}\,p_{\text{in}} + o_{a,t}\,p_{\text{out}}\right),\qquad H_{a,t}=H_{0,a}+t\Delta_a
$$

其中 $a$ 遍历所有 agent，$t$ 遍历该 agent 的轮次，$p_{\text{in}}/p_{\text{out}}$ 是 input/output 单价。由于 $H_{a,t}$ 随 $t$ 线性增长，单个 agent 的轮数增加会让成本二次增长——这正是「少开几个长命 agent，多开几个短命 worker」有时反而更省 token 的原因（表 4 给了示例口径）。延迟侧看关键路径：串行 30 次调用变成「编排 12 次 + worker 10 次 + 合并 1 次 = 23 次」，理论加速只有 $30/23 \approx 1.30$ 倍（忽略排队与生成长度差异），并行度不够时多 agent 只会更慢。

调试复杂度是真的指数级上升：一次失败可能来自任一 agent 的任一步，且下游 agent 常常会「自信地」把上游错误包装成看起来合理的产物。没有逐步轨迹，你连归因都做不到。护栏至少要有四道：调用数上限、token 预算、墙钟超时、通信轮数上限；任何一道触发就降级（退回单 agent 或转人工），而不是继续让 agent 互相说服（[[agents-09]]）。

### 决策规则

1. 只有当任务能分解为**低耦合**子任务、且存在真实的并行或隔离收益时才上多 agent；否则单 agent + 好工具 + 好上下文更稳（[[agents-05]]、[[agents-07]]）。
2. 落地顺序：先建单 agent 基线并记录成功率，再加一个 critic 角色做 A/B；只有基线在**某些具体子任务上**明确失败（例如上下文溢出、工具集过大导致选错工具），才把该子任务拆出去。
3. 用两个指标衡量：**任务成功率**（能力）与 **pass^k**（一致性）。pass^k 的定义是「同一个任务独立跑 $k$ 次全部成功」的概率，$n$ 次试验中成功 $c$ 次时用 $\binom{c}{k}/\binom{n}{k}$ 估计（在各次独立同分布的简化假设下等于 $p^k$）。它与大家更熟悉的 pass@k（$k$ 次里至少成功一次）方向相反：pass@k 衡量能力上界，pass^k 衡量可靠性下界——1 次能成、8 次全成，才是可交付的 agent。
4. 每次改动都要回答「这笔复杂度买到了什么」：更多 token？更低延迟？还是更高成功率？答不上来就撤掉。

## 数值与代码验证

**表 1：累积成功率（口径：$K$ 个决策点全部正确，各自独立、单点正确率 $p$，即 $p^{K}$）**

| $p$ | $K$ | 端到端成功率 |
| --- | --- | --- |
| 0.95 | 10 | 0.5987 |
| 0.95 | 20 | **0.3585** |
| 0.99 | 20 | 0.8179 |
| 0.90 | 8 | 0.4305 |

单步 0.95 听起来很不错，20 个必须全部正确的决策点之后只剩 35.9%。多 agent 之所以危险，就是它把 $K$ 从「一条主线的步数」变成「所有 agent 步数之和」；反过来，如果每个子任务短到只有 4–5 步，$0.95^4 = 0.8145$、$0.95^5 = 0.7738$，再配上重试与验证才谈得上可靠。

**表 2：pass@k 与 pass^k 的差距（口径：单次成功率 $p=0.5$，$k=8$，各次独立）**

| 指标 | 公式 | 数值 | 含义 |
| --- | --- | --- | --- |
| pass@8 | $1-(1-p)^k$ | 0.9961 | 至少成功一次——能力上界 |
| pass^8 | $p^k$ | 0.0039 | 八次全成功——可靠性下界 |

同一个 $p=0.5$、同一个 $k=8$，两个指标一个说 99.6%、一个说 0.39%。演示里挑一次好结果很容易，pass^k 才是用户每天遇到的体验。

**表 3：隐含约定一致性（口径：$N$ 个 agent 各自从 $M$ 种互斥约定中独立均匀选择，要求全部一致，概率 $M^{-(N-1)}$）**

| $N$ | $M$ | 全部一致的概率 |
| --- | --- | --- |
| 2 | 2 | 0.5000 |
| 4 | 2 | 0.1250 |
| 4 | 4 | 0.0156 |
| 8 | 4 | 0.0001 |

这张表是 Cognition 第二条原则的定量版本：接口、单位、命名、风格这些约定只要没有被主线显式冻结，就会随并行度指数级地互相冲突。**把约定写进 Handoff 并禁止子 agent 自行决定**，是把这项概率压回 1 的前提；指望子 agent 之间「聊」出一致约定并不可靠。

**表 4：token 账本（示例口径，全部为估算，用于比较数量级，不是实测值）**

| 形态 | 调用次数 | 输入 token | 输出 token | 合计 |
| --- | --- | --- | --- | --- |
| 单 agent：$H_0=3000$、$\Delta=700$、$T=30$ | 30 | 394500 | 9000 | 403500 |
| 编排者（12 轮）+ 4 个 worker（各 10 轮）+ 合并 | 12+40+1=53 | 265800 | 16600 | 282400 |
| 4 个 agent 开 8 轮讨论（每轮重读全部记录，$H_0=4000$、$\Delta=800$） | 32 | 217600 | 未计 | 217600 |

第二行的合计反而低于第一行（约 0.70 倍），原因不是「多 agent 更省」，而是**单上下文的二次项被消掉了**；换成更长的任务、更多的重复 brief 或更长的合并输入，这个比值会翻转。第三行则是纯粹的协调开销：32 次调用烧掉 21.76 万 token 的输入，产出的信息量为零——「agents talking too much」的典型形态。任何多 agent 方案都应该先按这套账本估一遍，再决定拆不拆。

**复算脚本（可直接运行，输出即上面表 1、表 2、表 4 的数字）**

```python
# 三个口径的复算：累积成功率、pass^k、token 账本
def cum(p, k):          # k 个必须全部正确的决策点
    return p ** k
def pass_at_k(p, k):    # k 次独立尝试里至少成功一次（能力上界）
    return 1 - (1 - p) ** k
def pass_pow_k(p, k):   # k 次独立尝试全部成功（一致性下界，i.i.d. 假设下等于 p^k）
    return p ** k
def loop_input(H0, d, T):   # 单上下文的输入 token：sum_{t=0}^{T-1}(H0 + t*d)
    return T * H0 + d * T * (T - 1) // 2

print("p^K :", [(p, k, round(cum(p, k), 4)) for p, k in [(0.95, 10), (0.95, 20), (0.99, 20), (0.90, 8)]])
print("pass:", round(pass_at_k(0.5, 8), 4), round(pass_pow_k(0.5, 8), 4))
single_in, single_out = loop_input(3000, 700, 30), 300 * 30
orch_in, orch_out = loop_input(1500, 700, 12), 300 * 12
work_in, work_out = 4 * loop_input(2200, 600, 10), 4 * 300 * 10
merge_in, merge_out = 4 * 1200 + 800, 1000   # 合并者读 4 份产物 + 任务书，产出 1 份合并结果
debate = 4 * loop_input(4000, 800, 8)
single = single_in + single_out
multi = orch_in + orch_out + work_in + work_out + merge_in + merge_out
print("tokens:", single, multi, round(multi / single, 2), debate)
print("split:", (single_in, single_out), (orch_in + work_in + merge_in, orch_out + work_out + merge_out))
```

**可运行的编排骨架（worker 用假实现，重点是契约与护栏）**

```python
from dataclasses import dataclass

@dataclass
class Handoff:                 # 子 agent 之间唯一的通信契约：结构化，而不是自由文本
    name: str
    decisions: dict            # 主线程已冻结的约定：子 agent 只能读，不能各自重新决定
    artifacts: list            # 产物落在共享文件系统，而不是各自的对话记忆
    check: object = None       # 可机器验证的验收条件；缺了它 critic 只能凭感觉

@dataclass
class Budget:                  # 护栏之一：调用数；token / 墙钟 / 通信轮数同理
    calls: int = 6
    rounds: int = 1

class Orchestrator:
    def __init__(self, budget: Budget):
        self.budget, self.trace = budget, []

    def run(self, steps):
        # 关键约定在主线一次性冻结，随每个 Handoff 下发（Principle 1/2 的工程化）
        frozen = {"unit": "cm", "interface": "Player.move(dy: int) -> None"}
        for step in steps:
            for attempt in range(self.budget.rounds + 1):
                self.budget.calls -= 1
                if self.budget.calls < 0:
                    return {"status": "budget_exhausted", "failed_at": step["name"]}
                out = step["worker"](Handoff(step["name"], frozen, step["artifacts"]))
                ok = step["check"](out)          # 独立验证：不让 producer 自评
                self.trace.append((step["name"], attempt, ok, out))
                if ok:
                    break
            else:
                return {"status": "failed", "failed_at": step["name"], "trace": self.trace}
        return {"status": "ok", "trace": self.trace}

def make_steps():
    seen = {"n": 0}
    def flaky_player(handoff):
        """第一次调用无视冻结的接口约定，重试才遵守——模拟隐式决策冲突。"""
        seen["n"] += 1
        iface = "Player.jump()" if seen["n"] == 1 else handoff.decisions["interface"]
        return {"file": "player.py", "iface": iface}
    return [
        {"name": "background", "artifacts": ["bg.py"],
         "check": lambda o: o["unit"] == "cm",
         "worker": lambda h: {"unit": h.decisions["unit"], "file": "bg.py"}},
        {"name": "player", "artifacts": ["player.py"],
         "check": lambda o: o["iface"] == "Player.move(dy: int) -> None",
         "worker": flaky_player},
    ]

print(Orchestrator(Budget(calls=6, rounds=1)).run(make_steps()))
print(Orchestrator(Budget(calls=2, rounds=3)).run(make_steps()))
```

第一行输出 `status: ok`，轨迹里能看到 `player` 第 0 次验收失败、第 1 次通过——**失败被定位到具体子任务与具体尝试**；第二行输出 `budget_exhausted`，护栏在预算耗尽时立刻停止，而不是让两个 agent 继续互相说服。

## 常见追问

- **追问**：多 agent 一定比单 agent 强吗？
  - 要点：不一定。论文口径下 7 个主流 MAS 的失败率是 41%–86.7%；benchmark 上的提升常常来自更多 token 与更多次尝试（best-of-N 效应），而不是编排本身。正确顺序是先建单 agent 基线，证明它在某个具体子任务上失败，再拆。
- **追问**：什么时候 fan-out 真的划算？
  - 要点：两个条件同时成立——子任务独立（不需要读对方上下文就能完成），且产物可机器验收。批量文档抽取、对 N 个候选并行打分属于此类；「写一个功能的不同部分」不属于。
- **追问**：怎么防止子 agent 之间的隐含约定冲突？
  - 要点：约定必须在主线**显式冻结**并随任务书下发（接口签名、单位、命名、错误处理约定），子 agent 无权重新决定；产物落共享文件系统，合并者读文件而不是读对话；对无法预先冻结的约定，指定唯一裁决者。表 3 的 $M^{-(N-1)}$ 就是不做这件事的代价。
- **追问**：合并者的上下文爆了怎么办？
  - 要点：合并者要读 $N|o|$，fan-out 越大越危险。三个可选做法：限制 fan-out 宽度并分层合并（树状 merge）、让 worker 直接产出结构化结果（schema 约束，字段固定则合并可程序化）、人工定义合并规则让代码做合并。不要指望再用一个 LLM 去「总结」所有产物。
- **追问**：critic 角色怎么才不是橡皮图章？
  - 要点：critic 必须有独立 grounding——独立来源、可执行测试、外部工具，而不是与 producer 共享同一份上下文。若 critic 只看 producer 的输出，它只能评风格；MAST 的 FM-3.3「错误验证」正是这种情况。同时给 critic 明确的、有界的接受/拒绝判据与最多 1–2 轮的重试上限。
- **追问**：怎么证明这套多 agent 比单 agent 好？
  - 要点：固定总预算（token 与调用数）做 A/B，比任务成功率与 pass^k；同时看 p50/p95 延迟与每次成功任务的成本。若在同等预算下没有显著提升，撤掉编排。注意用同一套评测集与同一批种子/温度设置，并报告方差而不是单次最好成绩。

## 公司变体

- **Cognition**：公开立场明显偏工程实现而非数学推导。他们关心的不是「该有几个角色」，而是上下文在哪里被切断、动作携带的隐含决策在哪里冲突、失败能不能归因到具体一步；因此他们主张默认单线程线性 agent + 完整 trace + 专门的上下文压缩模型，并明确说 2025 年让多个 agent 协作只会得到脆弱的系统。顺着这条线，面试里更可能让你设计具体机制：状态放哪里（文件系统 vs 消息）、约定怎么冻结并下发、压缩后丢什么、失败怎么重放与归因；也常见用实证口径追问「你怎么知道多 agent 值得这笔复杂度」。回答时用可测量的量（成功率、pass^k、token 账本、关键路径延迟）而不是架构图说话，比背诵模式名更对味。

## 相关题目

- [[agents-05]]：工具数量多少算太多，以及工具 schema 怎么设计——多 agent 的「专精」收益很大一部分来自把工具集切开。
- [[agents-07]]：为长时间运行的 agent 设计记忆——共享状态与上下文压缩的存储侧设计。
- [[agents-09]]：agent 循环如何正确终止、成本与步数怎么限——多 agent 的护栏与预算口径。
- [[agents-10]]：让 agent 的操作可逆、可审计——fan-out 之后的回滚与审计需求。
- [[agents-12]]：长时间运行后的漂移诊断——轨迹记录与归因的落地方法。
- [[inference-serving-09]]：TTFT、TPOT、ITL 与 throughput 的权衡——并行 fan-out 在服务侧的真实代价。

## 参考资料与归属

- Amit Shekhar (Outcome School)，*Multi-Agent Systems*，2026-04-29：<https://outcomeschool.com/blog/multi-agent-systems>（三大支柱、常见角色、通信与协调模式、常见错误、何时该用）。
- Amit Shekhar (Outcome School)，*AI Orchestration*，2026-05-22：<https://outcomeschool.com/blog/ai-orchestration>（顺序 / 并行 / 条件 / 循环 / 编排者-worker 模式、组件、挑战与最佳实践）。
- Walden Yan (Cognition)，*Don't Build Multi-Agents*（延伸），2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>（两条上下文工程原则、Flappy Bird 例子、单线程 + 压缩的替代方案、Claude Code 子 agent 与 edit-apply 模型两个旁证）。正文第 3 节「反方」与第 6 节公司变体的立场描述来自这一篇。
- Cemri et al.，*Why Do Multi-Agent LLM Systems Fail?*（延伸），2025-03-17：<https://arxiv.org/abs/2503.13657>（MAST-Data 1642 条轨迹、7 个框架、3 类 14 种失败模式、κ=0.88 / 0.77 / 0.79、41%–86.7% 失败率区间、ChatDev 干预案例 +9.4%）。正文中的失败模式分类与全部统计数字均为该论文口径，非本文实测。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
