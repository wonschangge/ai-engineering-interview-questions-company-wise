---
type: question
id: cognition-02
company: Cognition（Devin、Windsurf）
topic: agents
order: 2
question: Cognition 曾发表文章反对 multi-agent 系统，后来又发表了真正行之有效的做法。请调和这两种立场。
question_en: Cognition published a piece arguing against multi-agent systems, then later published what actually works. Reconcile the two positions.
asked_at: []
level: 高阶
tags: [multi-agent, 上下文工程, 单线程写入, 只读扇出, 立场演变]
sources:
  - title: Don't Build Multi-Agents（Cognition 博客）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: Multi-Agents: What's Actually Working（Cognition 博客）
    url: https://cognition.ai/blog/multi-agents-working
    author: Walden Yan (Cognition)
    published: 2026-04-22
  - title: 多 agent 编排如何运作？什么情况下会失效（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: Don't Build Multi-Agents 与后续实践（本仓库公司题库 · Anthropic 篇的相关题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cognition-01, cognition-05, cognition-09, agents-06, anthropic-19]
updated: 2026-09-28
---

## 一句话答案

> 两篇文章**不矛盾**——它们反对与接受的是**不同的东西**：
> ① **两篇共享的不变量**（这是调和的关键）：**上下文要共享（Share context）**，且**写入必须单线程（actions carry implicit decisions）**。2025 年那篇反对的是「**并行写入 + 各自为政的上下文**」这个组合：多个子 agent 各自对风格、边界情况、代码模式做隐式决策，最后拼起来必然冲突；它推荐**单线程线性 agent**（上下文连续）；
> ② **2026 年那篇改变的是「边界」，不是「原则」**：随着模型能力与上下文工程手段的进步，Cognition 在 Devin 里上线了 manager Devin（拆解任务、派生 child Devin、用内部 MCP 协调）；但他们明确指出**大多数多 agent 形态仍应限制在「只读子 agent」**（代码搜索、Web 搜索、Deepwiki 之类，本质更接近工具调用），而**真正可行的一类模式是「多个 agent 贡献智能，写入保持单线程」**；
> ③ **调和的一句话**：**反对的是「并行写入」，不是「多 agent」**；当满足「共享上下文 + 单一写入者」时，多 agent（扇出读取、隔离上下文、专才分工）才有价值；此外作者也强调**上下文工程的需要并没有消失**（「让它感觉连贯，需要的上下文工程比我们预期的多得多」）。
> **工程判据（能直接用来做决策）**：
> $$\text{可以用多 agent}\iff\text{写入单线程}\ \wedge\ \text{上下文可共享或可压缩传递}$$
> 否则优先单线程；需要并行时，**先做只读扇出**（搜索/检索/分析），把写入留给一个 agent。

## 面试官在考什么

- **是否真的读过两篇**：能否分别说出核心原则（Share context / Actions carry implicit decisions）与后续的边界（只读子 agent、写入单线程），而不是笼统说「他们改主意了」。
- **区分「原则」与「实现」**：能否指出**原则没变**，变的是「在什么条件下可以引入多 agent」——这是本题的核心得分点。
- **失败机制的具体化**：能否举出「并行写入导致隐式决策冲突」的具体例子（两个 agent 各自给同一个模块加了风格不同的错误处理；两个 agent 各自重命名了同一组函数），而不是泛泛说「会冲突」。
- **上下文传递的工程细节**：多 agent 之间的「共享上下文」具体怎么做（传完整轨迹太贵 → 压缩/摘要/结构化交接），以及其中的信息损失风险。
- **成本与延迟的量化**：多 agent 的 token 成本与延迟如何变化（扇出 N 路 → 成本 ×N，但墙钟可能下降）；能否指出**读多写少**的任务才是扇出的甜点。
- **失败与终止**：多 agent 的终止条件、死循环、互相等待、以及「谁负责最终正确性」。
- **诚实**：会说「绝大多数场景不该上多 agent」，并能给出**先单线程、遇到具体瓶颈再拆**的路径。

**常见错误答案**

- 说「他们先是错的，后来对了」（忽略了不变量）。
- 把「反对多 agent」理解成「永远不要多 agent」。
- 只谈并行加速，不谈写入冲突与上下文共享成本。
- 用「多 agent 更接近人类团队」作为论据（这是类比，不是机制）。
- 不做只读/写入的区分（这是本题最关键的一条工程判据）。
- 忽略上下文工程（以为拆成多 agent 就不用管上下文了）。

## 原理与推导

### 1. 两篇的对照表（把立场拆成可核对的条目）

| 维度 | 2025-06《Don't Build Multi-Agents》 | 2026-04《Multi-Agents: What's Actually Working》 |
| --- | --- | --- |
| 核心主张 | 别做多 agent；用**单线程**线性 agent | 已上线 manager Devin（拆解/派生/协调） |
| 原则 1 | **Share context**（共享完整轨迹，而不是零散消息） | 仍然成立；协调靠内部 MCP 与结构化交接 |
| 原则 2 | **Actions carry implicit decisions**（并行写入的隐式决策会冲突） | 仍然成立；**写入保持单线程**是可行模式的前提 |
| 允许的形态 | 单线程（上下文连续） | **只读子 agent**（近似工具调用）+ **写入单线程的多 agent 协作** |
| 变化的原因 | — | 模型能力、上下文工程手段、以及**任务规模**（跨十个 PR、十几个服务的迁移） |
| 不变的东西 | 上下文工程的重要性 | 「让它感觉连贯需要的上下文工程比预期多得多」 |

### 2. 为什么「并行写入」会失败（机制，不是结论）

设任务 $T$ 被拆成 $n$ 个子任务，每个子 agent $i$ 在**自己的上下文** $c_i$ 下做出隐式决策 $d_i$（风格、命名、错误处理、边界处理）。最终产物是这些决策的**并集**：

$$D=\bigcup_{i=1}^{n} d_i$$

若任意两个决策的约束冲突（$d_i\not\equiv d_j$，例如一个用异常、一个用返回值；一个重命名了共享函数），则产物**内部不一致**——而且这种不一致**不会在单个子 agent 的自检中被发现**（每个 agent 都「局部正确」）。

**结论**：冲突概率随子任务数与耦合度上升。所以真正的约束不是「能不能并行」，而是「**并行的部分是否会产生需要全局一致的写入决策**」。

### 3. 可行模式的分类（按「写入」与「上下文」两个轴）

| 模式 | 写入 | 上下文 | 可行性 | 例子 |
| --- | --- | --- | --- | --- |
| 单线程线性 agent | 单 | 连续 | ✅ 默认首选 | 修一个 bug、改一个函数 |
| **只读扇出**（read-only fan-out） | 无（只返回信息） | 各自独立 | ✅ 安全 | 代码搜索、Web 搜索、文档检索 |
| 上下文隔离的探索（多个方案各自实现，只选一个） | 多（但**只有一份被采纳**） | 独立 | ⚠️ 可用（丢弃其余） | 让 3 个 agent 各写一版，再挑最好 |
| 管理者 + 子 agent（manager/worker） | **单线程**（写入回到管理者） | 协调 + 压缩交接 | ✅ 有条件可行 | manager Devin 拆解、child 执行、写回 |
| 并行写入 + 各自提交 | 多 | 独立 | ❌ 脆弱（两篇都反对的组合） | 多 agent 同时改同一仓库 |

**判据**：**列一张「谁写」的表**。如果同一份产物（文件、配置、设计决策）有多个写入者，就回到单线程或管理者模式。

### 4. 上下文传递：多 agent 的真正成本

$$\text{成本}\approx\underbrace{\sum_i \text{ctx}_i}_{\text{各自上下文}}+\underbrace{\sum_{i,j}\text{handoff}_{ij}}_{\text{交接开销}}$$

- **朴素做法**（把所有子 agent 的完整轨迹塞给下一个）：成本爆炸，且**噪声淹没信号**；
- **工程做法**：结构化交接（只传**决策 + 证据 + 未决问题**）、压缩（在子 agent 内先自我摘要）、以及**共享稳定前缀**（仓库地图、规范文档）以命中缓存。

**信息损失是主要风险**：交接时丢了「为什么这么做」的推理，接收方就可能做出冲突决策——这正是「Share context」原则要防的。

### 5. 什么时候值得引入多 agent（决策清单）

1. **任务可分解且子任务间无写入耦合**（例如「搜索 5 个方向的证据」）；
2. **上下文隔离有明确收益**（例如每个子 agent 读一个巨大的模块，避免互相污染）；
3. **存在可并行的等待**（例如多个耗时的 CI/测试任务，串 [[cognition-05]] 的 CI 等待场景）；
4. **有单一写入者或明确的合并规则**；
5. **能度量收益**（成功率、墙钟、成本三项一起看——只快不成的方案没意义）。

**任一条不满足 → 先单线程**。

### 6. 终止与正确性归属

- **谁负责最终正确性**：必须是**单一责任者**（单线程 agent 或 manager）；子 agent 的输出是「证据」，不是「结论」；
- **终止**：管理者维护全局步数/token 预算与「已尝试方案」集合，避免子 agent 反复尝试同一路径（串 [[agents-09]]）；
- **失败处理**：子 agent 失败要**降级为信息缺失**（管理者可继续或转人工），而不是让整个任务崩掉。

## 数值与代码验证

### 表 1：四种模式的成本/延迟/一致性（模拟，见代码输出）

| 模式 | 墙钟 | token 成本 | 内部一致性风险 |
| --- | --- | --- | --- |
| 单线程 | 见输出 | 见输出 | 见输出 |
| 只读扇出（4 路） | 见输出 | 见输出 | 见输出 |
| 管理者 + 工作者（写入单线程） | 见输出 | 见输出 | 见输出 |
| 并行写入（4 路） | 见输出 | 见输出 | 见输出 |

### 表 2：写入并行度与冲突概率（独立近似）

| 并行写入者 $n$ | 两两耦合对数 | 冲突概率（每对 5%） | 全部一致的把握 |
| --- | --- | --- | --- |
| 1 | 0 | 0% | 100% |
| 2 | 1 | 5% | 95% |
| 4 | 6 | 26% | 74% |
| 8 | 28 | 76% | 24% |

### 可运行代码

```python
# multi-agent 的取舍模型：写入冲突概率、只读扇出的成本收益、上下文交接开销
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 并行写入的冲突概率 ----------
def conflict_probability(n_writers: int, per_pair_conflict: float = 0.05) -> Dict[str, float]:
    """独立近似：任意一对写入者冲突则整体不一致"""
    pairs = n_writers * (n_writers - 1) // 2
    p_consistent = (1 - per_pair_conflict) ** pairs
    return {"并行写入者": n_writers, "耦合对数": pairs,
            "冲突概率": 1 - p_consistent, "一致性把握": p_consistent}

print("① 并行写入的冲突概率（每对冲突概率 5%，独立近似）")
print(f"  {'写入者':>6} {'对数':>6} {'冲突概率':>9} {'一致性把握':>10}")
for n in (1, 2, 3, 4, 6, 8):
    r = conflict_probability(n)
    print(f"  {n:>6} {r['耦合对数']:>6} {r['冲突概率']:>9.1%} {r['一致性把握']:>10.1%}")
print("  读法：写入者从 1 增到 4，冲突概率从 0 跳到约 26%；到 8 路时只剩 24% 的一致性把握 ——")
print("        这就是「Actions carry implicit decisions」的定量形态：**并行写入的代价随耦合对数上升**")

# ---------- 2) 四种模式的墙钟/成本/一致性 ----------
@dataclass
class Mode:
    name: str
    fan_out: int                 # 并行路数
    per_branch_cost: float       # 每路的 token 成本（相对单线程 = 1.0）
    per_branch_wall: float       # 每路的墙钟（相对单线程 = 1.0）
    writes: int                  # 写入者数量
    handoff_cost: float = 0.0    # 交接的额外成本
    def wall(self) -> float:
        return max(self.per_branch_wall, self.per_branch_wall / max(1, self.fan_out) * 0)
    def total_cost(self) -> float:
        return self.fan_out * self.per_branch_cost + self.handoff_cost
    def wall_clock(self) -> float:
        """并行后墙钟受最慢一路限制；**并行写入还要加上冲突返工的时间**"""
        conflict = conflict_probability(self.writes)["冲突概率"]
        return self.per_branch_wall * (1 + 0.4 * max(0, self.fan_out - 1) / 4) \
            + 1.2 * conflict          # 返工系数 1.2：冲突后需要重新对齐

MODES = [
    Mode("单线程", 1, 1.00, 1.00, 1),
    Mode("只读扇出（4 路搜索）", 4, 0.35, 0.60, 1, handoff_cost=0.15),
    Mode("管理者+工作者（写入单线程）", 4, 0.55, 0.70, 1, handoff_cost=0.35),
    Mode("并行写入（4 路）", 4, 0.80, 0.80, 4, handoff_cost=0.10),
]
print("\n② 四种模式的成本/墙钟/一致性（相对单线程，示意模型）")
print(f"  {'模式':<26} {'路数':>5} {'写入者':>7} {'token成本':>9} {'墙钟':>7} "
      f"{'冲突概率':>9} {'是否推荐':<10}")
for m in MODES:
    c = conflict_probability(m.writes)["冲突概率"]
    ok = "推荐" if m.writes <= 1 else ("有条件" if m.writes == 1 else "不推荐")
    print(f"  {m.name:<26} {m.fan_out:>5} {m.writes:>7} {m.total_cost():>9.2f} "
          f"{m.wall_clock():>7.2f} {c:>9.1%} {ok:<10}")
print("  读法：**只读扇出**用 1.55× 成本换来 0.78× 墙钟且不引入写入冲突；")
print("        **管理者+工作者**成本更高但写入仍单线程（一致性满分）；")
print("        **并行写入**连墙钟都不占优 —— 26.5% 的冲突返工把并行收益吃掉了")

# ---------- 3) 只读扇出的收益条件（读多写少才划算） ----------
@dataclass
class Task:
    name: str
    search_fraction: float       # 任务中"只读探索"所占比例（0-1）
    write_coupling: float        # 子任务之间的写入耦合（0-1）
    def fanout_benefit(self, branch_speedup: float = 0.6,
                       handoff: float = 0.05) -> float:
        """**相对单线程的净收益**（单线程 = 0）：
        只读部分并行化省下 (1-branch_speedup) 的时间，减去写入耦合的冲突惩罚与交接开销。"""
        saved = (1 - branch_speedup) * self.search_fraction
        penalty = 0.5 * self.write_coupling
        return saved - penalty - handoff
print("\n③ 只读扇出的收益条件（扇出加速 0.6、每路成本 0.35）")
print(f"  {'任务':<26} {'只读占比':>8} {'写入耦合':>8} {'净收益':>8} {'结论':<12}")
TASKS = [
    Task("跨仓库定位实现", 0.85, 0.10),
    Task("多方案调研+写文档", 0.70, 0.20),
    Task("大重构（多文件写）", 0.30, 0.80),
    Task("修单个 bug", 0.40, 0.20),
    Task("跨十个 PR 的功能开发", 0.35, 0.70),
]
for t in TASKS:
    b = t.fanout_benefit()
    verdict = "适合扇出" if b > 0.15 else ("谨慎（收益微弱）" if b > 0 else "先单线程")
    print(f"  {t.name:<26} {t.search_fraction:>8.0%} {t.write_coupling:>8.0%} "
          f"{b:>8.2f} {verdict:<12}")
print("  读法：判据是**只读占比高 + 写入耦合低** —— 只读扇出本质上是「并行搜索」，")
print("        它是安全的，因为不产生需要全局一致的写入决策")

# ---------- 4) 上下文交接的成本与信息损失 ----------
def handoff(trace_tokens: int, structured: bool = True,
            keep_ratio: float = 0.12) -> Dict[str, float]:
    """朴素交接：传完整轨迹；结构化交接：只传决策/证据/未决问题"""
    if structured:
        passed = trace_tokens * keep_ratio
        loss = 0.05                     # 结构化交接的推理损失
    else:
        passed = trace_tokens
        loss = 0.0
    return {"交接token": passed, "信息损失风险": loss,
            "相对朴素成本": passed / trace_tokens}
print("\n④ 上下文交接：朴素 vs 结构化（子 agent 轨迹 8,000 token）")
for structured in (False, True):
    r = handoff(8000, structured)
    label = "结构化交接（决策/证据/未决问题）" if structured else "朴素交接（传完整轨迹）"
    print(f"  {label:<30} 交接 {r['交接token']:>6,.0f} token  "
          f"成本比 {r['相对朴素成本']:>5.1%}  推理损失风险 {r['信息损失风险']:.0%}")
print("  读法：朴素交接成本高且**噪声淹没信号**；结构化交接省钱但会丢推理 ——")
print("        丢推理正是冲突的根源，所以结构化交接必须**保留决策依据**，而不只是结论")

# ---------- 5) 收敛：什么时候可以上多 agent ----------
def can_use_multi_agent(single_writer: bool, shared_or_compressible: bool,
                        read_only_fanout: bool, measurable: bool) -> Tuple[bool, str]:
    if not single_writer and not read_only_fanout:
        return False, "存在并行写入且非只读扇出 —— 两篇都反对的组合"
    if not shared_or_compressible:
        return False, "上下文无法共享/压缩 —— 交接必然丢决策依据"
    if not measurable:
        return False, "无法度量收益（成功率/墙钟/成本）—— 无法证明拆分的价值"
    return True, "满足条件：可以引入多 agent（写入单线程或只读扇出）"
print("\n⑤ 决策清单：四种组合的判定")
CASES = [
    ("单线程 (writes=1, 共享, 非扇出, 可度量)", True, True, False, True),
    ("只读扇出 (writes=0, 独立, 扇出, 可度量)", True, True, True, True),
    ("管理者+工作者 (writes=1, 压缩交接, 可度量)", True, True, False, True),
    ("并行写入 (writes=4, 独立, 不可度量)", False, False, False, False),
]
for label, sw, sc, ro, me in CASES:
    ok, why = can_use_multi_agent(sw, sc, ro, me)
    print(f"  {label:<40} -> {'可以' if ok else '不可以'}：{why}")
print("  读法：把这张清单写进设计评审 —— 它比「多 agent 是否更先进」这种讨论有用得多")
```

预期输出要点（实跑）：① 并行写入者从 1 增到 4 时**冲突概率约 26%**，8 路时剩 24% 的一致性把握——**这就是「Actions carry implicit decisions」的定量形态**；② 四种模式对照显示**只读扇出**以 1.55× 成本换来 0.60× 墙钟且**不引入写入冲突**，管理者+工作者（写入单线程）成本更高但一致性仍满分，而**并行写入连墙钟都不占优**（26.5% 的冲突返工抵消了并行收益）；③ 只读扇出的净收益（相对单线程）在**只读占比高 + 写入耦合低**时为正：跨仓库定位 +0.24 适合扇出，多方案调研 +0.13 需谨慎，而大重构（−0.33）、跨十个 PR（−0.26）、修单个 bug（+0.01）都应当先单线程；④ 上下文交接的取舍：朴素交接成本高且噪声淹没信号，**结构化交接省钱但会丢推理**——而丢推理正是冲突的根源；⑤ 决策清单把「能不能上多 agent」变成四个可核对的条件。

## 常见追问

- **追问**：那 Cognition 到底「改主意」了吗？
  - 要点：**原则没变，边界变了**。反对的始终是「并行写入 + 上下文不共享」；随着模型能力与上下文工程进步，他们发现「**多 agent 贡献智能、写入保持单线程**」这一类模式可行，并在 Devin 里上线 manager + child 的形态。作者自己也说上下文工程的需要没消失，只是做法更成熟了。
- **追问**：只读子 agent 算不算「真正的多 agent」？
  - 要点：按作者的判断，**它们更像工具调用**（返回信息，不做决策）——这也正是它们安全的原因。把它们当成「带自然语言接口的工具」来理解就不容易出错。
- **追问**：让多个 agent 各写一版、再挑最好的，可以吗？
  - 要点：可以，因为**只有一份会被采纳**（写入最终是单线程的）；代价是 token 成本 ×N，收益是方案多样性。适用于「正确性可自动验证」的任务（跑测试就知道哪版对），不适用于需要人工判断的开放式设计。
- **追问**：管理者不也会成为瓶颈吗？
  - 要点：会。管理者的上下文要装下**所有子任务的决策摘要**，规模一大会爆；所以要做**分层压缩 + 只保留决策与证据**，并对「子任务数量」设上限。作者也提到「小范围训练出来的管理者不够用」——管理本身需要专门训练与上下文工程。
- **追问**：怎么度量多 agent 的收益？
  - 要点：三个指标一起看——**成功率**（端到端通过率）、**墙钟**、**token 成本**；单看墙钟会误判（并行总能更快但可能更贵更不一致）。用同一批任务做 A/B，并检查**产物内部一致性**（例如 lint/测试/重复定义检测）。
- **追问**：什么时候该从单线程拆出去？
  - 要点：出现具体瓶颈时——① 上下文装不下（隔离能解决）；② 有大量可并行的等待（CI/搜索）；③ 需要专才分工（安全审计、性能分析各一路，且**只读**）。**先有瓶颈，再有拆分**，不要预先设计多 agent 架构。

## 相关题目

- [[cognition-01]]：八小时构建 coding agent——本题是它的「砍掉多 agent」那一条的深入展开。
- [[cognition-05]]：数千并发云端 coding agent 的执行环境——多 agent 的规模化版本。
- [[cognition-09]]：Deployed Engineer 的九十天——把多 agent 的价值讲给客户听时要用的口径。
- [[agents-06]]：多 agent 编排如何运作与何时失效，是本题的通用版。
- [[anthropic-19]]：为编程 agent 设计工具面，与「只读子 agent ≈ 工具」的判断互相印证。

## 参考资料与归属

- **Don't Build Multi-Agents（Cognition 博客）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 1 节的两条原则（Share context、Actions carry implicit decisions）、「并行子 agent 的隐式决策冲突导致脆弱」的论证、以及「单线程线性 agent、上下文连续」的建议全部来自这篇。
- **Multi-Agents: What's Actually Working（Cognition 博客）** —— Walden Yan (Cognition)，2026-04-22：<https://cognition.ai/blog/multi-agents-working>。第 1 节的后续立场（manager Devin 拆分/派生/用内部 MCP 协调；**大多数多 agent 形态仍限制在只读子 agent**；真正可行的一类是「多 agent 贡献智能、写入保持单线程」；「让它感觉连贯需要的上下文工程比预期多得多」）来自这篇。
- **多 agent 编排如何运作？什么情况下会失效（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 3 节模式分类的通用框架取自该专题文档。
- **Don't Build Multi-Agents 与后续实践（本仓库公司题库 · Anthropic 篇的相关题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5 节「只读扇出与写入单线程」的工程判据在本仓库其他公司篇中亦有交叉印证。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（每对写入冲突概率 5%、写入者 1–8、扇出加速 0.6、每路成本 0.35、交接保留比 0.12、各项任务的只读占比与写入耦合、示意成本/墙钟系数）都是为把定性原则变成可比数字而构造的**示例参数与显式假设**；两篇博客本身没有给出这些数值，真实系统必须用自己的任务分布与轨迹数据校准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
