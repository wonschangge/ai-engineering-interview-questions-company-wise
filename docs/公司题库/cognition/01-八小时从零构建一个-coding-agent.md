---
type: question
id: cognition-01
company: Cognition（Devin、Windsurf）
topic: agents
order: 1
question: 你有八个小时从零构建一个 coding agent。请描述你会构建什么，更重要的是，你会砍掉什么。
question_en: You have eight hours to build a coding agent from scratch. Describe what you would build, and more importantly, what you would cut.
asked_at: []
level: 高阶
tags: [coding-agent, 八小时构建, 上下文工程, 工具集, 砍需求]
sources:
  - title: Don't Build Multi-Agents（Cognition 博客）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: Multi-Agents: What's Actually Working（Cognition 博客）
    url: https://cognition.ai/blog/multi-agents-working
    author: Walden Yan (Cognition)
    published: 2026-04-22
  - title: 解释 ReAct 模式以及它相比 chain-of-thought 解决了什么（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）
    url: https://arxiv.org/abs/2310.06770
    author: Jimenez et al. (ICLR 2024)
    published: 2023-10-10
related: [cognition-02, cognition-03, cognition-07, agents-09, agents-07]
updated: 2026-09-28
---

## 一句话答案

> 八小时的正确目标是**一个能在 10–20 个真实小任务上跑通、并且你能解释它为什么失败的 coding agent**——不是平台。所以：
> **要建的四件东西（按投入排序）**
> ① **单线程 agent 循环**（约 2 h）：`观察 → 决策 → 工具调用 → 结果回灌`，**上下文连续**（这也正是 Cognition 的第一条原则：Share context）；
> ② **六个锋利工具**（约 2 h）：`read_file`、`search`（正则/符号）、`edit`（精确替换）、`write`、`run_shell`、`run_tests`。**少而准**比多而杂好——工具一多，选择错误率就上去了；
> ③ **上下文工程**（约 2 h）：仓库地图（文件树 + 关键符号）、**压缩/摘要**（在窗口的 60–70% 触发）、以及**把工具输出截断到有用部分**。这往往比换模型更能提升成功率；
> ④ **验证闭环**（约 1.5 h）：**跑测试作为唯一真值**——模型说「完成」不算完成；失败时把测试输出回灌给循环继续修（这就是 SWE-bench 的评测口径，串 [[cognition-07]]）。
> **明确砍掉的六件事（这才是本题的重点）**：多 agent 编排、自建沙箱/容器平台、向量数据库记忆、MCP 生态、模型微调、精美 UI。
> 一句话判据：**八小时的产出应该是「一个可观测、可回放、可解释失败的循环」**；把时间花在**上下文与验证**上，而不是花在架构上。

## 面试官在考什么

- **是否敢砍**：题目明说「更重要的是你会砍掉什么」——能不能砍得有理有据（每一项给出砍掉的理由与代价），而不是列一堆「未来会做」。
- **优先级判断**：是否把时间投在**上下文工程与验证**上，而不是工具数量与编排框架。
- **单线程 vs 多 agent 的取舍**：能否引 Cognition 的立场（Share context / Actions carry implicit decisions）说明「为什么不从多 agent 开始」，以及对**只读扇出**例外的理解（串 [[cognition-02]]）。
- **工具设计**：工具数量与粒度的取舍（粗粒度 edit vs 细粒度 patch）、错误信息如何设计（让模型能自我纠正）、以及**幂等/可回退**（串 [[agents-10]]）。
- **验证闭环**：能否坚持「以测试为真值」，并说清「没有测试时怎么办」（写最小复现、跑命令观察、人工确认）。
- **上下文预算**：能否给出「窗口 60–70% 触发压缩」「工具输出截断」「仓库地图」等具体做法，并说明它们的失效模式（压缩丢关键信息）。
- **可观测性**：每一步的提示/输出/工具调用/耗时都要落盘，因为八小时的迭代全靠**回放与对比**（串 [[agents-12]] 的漂移诊断）。
- **诚实的边界**：会说「八小时做不出能上生产的东西」，并给出哪些是故意留到后面的（沙箱、权限、成本控制）。

**常见错误答案**

- 列架构图（多 agent、消息总线、向量库），但没有可跑的循环。
- 工具做二十个（含浏览器、Jira、Slack），每个都半成品。
- 不做验证闭环（靠模型自述成功）。
- 不做上下文管理（第 20 步就爆窗口，然后开始「遗忘」）。
- 不从「能否评测」出发（做完不知道好坏）。
- 不砍任何东西（八小时全用在搭脚手架上）。

## 原理与推导

### 1. 八小时的时间预算（把时间当约束来分配）

| 时段 | 投入 | 产出 | 不做的代价 |
| --- | --- | --- | --- |
| 0:00–2:00 | agent 循环 + 提示结构 | 能跑通一次「读—改—测」 | 没有循环，什么都验证不了 |
| 2:00–4:00 | 工具集（6 个）+ 错误语义 | 工具可被正确选择与纠正 | 工具错 → 循环空转 |
| 4:00–6:00 | 上下文工程（地图 + 压缩 + 截断） | 能撑过 30+ 步 | 爆窗口 → 中途失忆 |
| 6:00–7:30 | 验证闭环（测试运行器 + 回灌） | 成功判定可信 | 假成功 → 评测全错 |
| 7:30–8:00 | 观测与回放（轨迹落盘） | 能对比两次改动 | 无法迭代 |

**读法**：**架构时间是 0**——单线程循环不需要架构；**上下文与验证占一半**，这与 Cognition 的经验一致（「需要比预期多得多的上下文工程」）。

### 2. 工具集：六个就够

| 工具 | 为什么需要 | 设计要点 | 常见错误 |
| --- | --- | --- | --- |
| `read_file` | 读代码 | 支持行范围、大文件分页 | 一次返回整个仓库 |
| `search` | 定位 | 正则 + 符号索引；返回**文件:行 + 片段** | 返回全文件内容 |
| `edit` | 精确修改 | **要求唯一匹配**，否则报错重试 | 模糊替换改错位置 |
| `write` | 新建文件 | 覆盖需显式确认 | 误覆盖已有文件 |
| `run_shell` | 探索环境 | 超时 + 输出截断 | 无超时 → 卡死 |
| `run_tests` | 验证 | **结构化解析**（通过/失败/哪条） | 只回原始日志 |

**工具数量的代价**：工具越多，模型选错的概率越高（串 [[agents-05]]）。经验法则：**能用 `run_shell` 组合出来的，不要单独做工具**。

### 3. 上下文工程（本题的隐藏主考点）

上下文在循环里是这样增长的：

$$\text{ctx}(t)=\text{sys}+\text{map}+\sum_{i\le t}\big(\text{思考}_i+\text{工具调用}_i+\text{工具输出}_i\big)$$

**三个必做动作**：
1. **仓库地图**（一次性、稳定前缀）：文件树 + 关键符号 → 让模型知道「去哪找」，避免盲目搜索（也正是 [[cognition-03]] 要解决的问题）；
2. **工具输出截断**：只保留相关片段（例如搜索命中前后 N 行、测试失败的断言），**这是最大的一块节约**；
3. **压缩/摘要**：在窗口 60–70% 时把早期步骤摘要成「已完成/已决定/待办」，保留**决策与结论**而丢弃原始日志。

**为什么用 60–70% 而不是 90%**：压缩本身要花一次调用且可能丢信息，留出余量给「压缩失败后的重试」。**别等到 95% 才动**——那时压缩必须在极端压力下完成，质量最差。

### 4. 验证闭环：为什么「以测试为真值」

模型的自述成功率**系统性偏高**（它不知道自己没改对）。可靠信号只有三类：
1. **已有测试**（最好）；
2. **可执行的最小复现**（次好：让 agent 自己写一个能复现 bug 的脚本）；
3. **人工确认**（最后手段）。

**闭环形式**：`改 → 跑 → 失败则把结构化失败信息回灌 → 再改`，并设**最大重试次数**与**相同失败重复时的策略切换**（例如换搜索方向、请人确认，串 [[agents-09]] 的终止与成本限制）。

### 5. 明确砍掉的六件事（附理由）

| 砍掉 | 理由 | 什么时候再加 |
| --- | --- | --- |
| 多 agent 编排 | 并行写入会让隐式决策冲突（Cognition 的原则 2） | 写入保持单线程、上下文可共享时（串 [[cognition-02]]） |
| 自建沙箱/容器平台 | 用现成容器 + 只读挂载 + 网络白名单即可 | 多租户并发时才需要（串 [[cognition-05]]） |
| 向量数据库记忆 | 八小时内检索质量无法验证；仓库地图+搜索更可控 | 跨会话长期记忆时（串 [[agents-07]]） |
| MCP 生态 | 标准化的价值在生态，不在单机验证 | 需要接第三方工具时 |
| 模型微调 | 数据与评测都没有，微调无法验证收益 | 有稳定任务分布与标注后（串 [[cognition-04]]） |
| 精美 UI | 终端足够；UI 不影响成功率 | 演示与人机协作时 |

## 数值与代码验证

### 表 1：上下文增长与压缩/截断的效果（窗口 12.8 万 token、60 步）

| 策略 | **峰值上下文** | **60 步后** | **压缩次数** | **是否爆窗口** |
| --- | --- | --- | --- | --- |
| 不截断、不压缩 | 130,255 | 47,115 | 2 | **第 27 步爆窗口** |
| 只截断工具输出 | 105,862 | 105,862 | 0 | 未爆（贴着上限） |
| 截断 + 60% 触发压缩 | **77,381** | **42,981** | 1 | 未爆 |

### 表 2：八小时的投入分配与对应风险

| 投入块 | 小时 | 若省略 | 症状 |
| --- | --- | --- | --- |
| 循环 + 提示 | 2.0 | 无法验证任何东西 | 无 |
| 工具集 | 2.0 | 工具误用、空转 | 反复读同一文件 |
| 上下文工程 | 2.0 | 20 步后失忆 | 忘记最初目标 |
| 验证闭环 | 1.5 | 假成功 | 报告完成但测试失败 |
| 观测回放 | 0.5 | 无法迭代 | 改一处坏一处 |

### 可运行代码

```python
# 八小时 coding agent 的上下文预算模型：增长、截断、压缩与"何时爆窗口"
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Budget:
    window_tokens: int = 128_000       # 模型上下文窗口（更贴近 agent harness 的常见设置）
    sys_prompt: int = 1_500            # 系统提示 + 工具 schema
    repo_map: int = 6_000              # 仓库地图（稳定前缀，可缓存）
    compact_at: float = 0.60           # 触发压缩的窗口占比
    compact_cost_tokens: int = 3_000   # 压缩本身的开销（一次调用）
    compact_keep: int = 4_000          # 压缩后保留的摘要长度
    truncate_output: bool = True       # 是否截断工具输出
    output_cap: int = 1_200            # 单次工具输出的上限

@dataclass
class Step:
    think: int
    call: int
    output_raw: int
    kind: str

def simulate(steps: List[Step], b: Budget) -> Dict[str, object]:
    ctx = b.sys_prompt + b.repo_map
    peak = ctx
    compactions = 0
    overflow_at = None
    log: List[Tuple[int, int, str]] = []
    for i, s in enumerate(steps, 1):
        out = min(s.output_raw, b.output_cap) if b.truncate_output else s.output_raw
        ctx += s.think + s.call + out
        peak = max(peak, ctx)
        if overflow_at is None and ctx > b.window_tokens:
            overflow_at = i
        # 触发压缩
        if ctx > b.window_tokens * b.compact_at:
            ctx = b.sys_prompt + b.repo_map + b.compact_keep + b.compact_cost_tokens
            compactions += 1
            log.append((i, ctx, "压缩"))
        else:
            log.append((i, ctx, ""))
    return {"峰值": peak, "最终": ctx, "压缩次数": compactions, "爆窗口于第": overflow_at,
            "日志": log}

def make_steps(n: int = 60, seed: int = 3, verbose: bool = True) -> List[Step]:
    rnd = random.Random(seed)
    kinds = ["search", "read", "edit", "shell", "test"]
    steps = []
    for i in range(n):
        k = kinds[i % len(kinds)]
        # 搜索/读取/测试的输出远大于编辑
        raw = {"search": 3_000, "read": 8_000, "edit": 600,
               "shell": 2_500, "test": 5_000}[k]
        if verbose:
            raw = int(raw * rnd.uniform(0.7, 1.4))
        steps.append(Step(think=rnd.randint(200, 600), call=150, output_raw=raw, kind=k))
    return steps

steps = make_steps(60)
print("① 上下文增长：三种策略对照（窗口 12.8 万 token，60 步，含大文件读取）")
configs = [
    ("不截断（压缩也来不及）", Budget(truncate_output=False, compact_at=1.01)),
    ("只截断，不压缩", Budget(truncate_output=True, compact_at=1.01)),
    ("截断 + 60% 触发压缩", Budget(truncate_output=True, compact_at=0.60)),
]
print(f"  {'策略':<22} {'峰值':>10} {'最终':>9} {'压缩次数':>9} {'爆窗口':>8}")
for label, b in configs:
    r = simulate(steps, b)
    over = f"第{r['爆窗口于第']}步" if r["爆窗口于第"] else "未爆"
    print(f"  {label:<22} {r['峰值']:>10,} {r['最终']:>9,} {r['压缩次数']:>9} {over:>8}")
print("  读法：不截断时第 27 步就突破窗口（**必须截断工具输出**）；只截断仍会缓慢增长（峰值 10.6 万）；")
print("        加上 60% 触发的压缩后峰值被压在 7.7 万 —— 这是能跑完 60 步的前提")

print("\n② 压缩时机的敏感性（触发阈值 vs 峰值/压缩次数）")
print(f"  {'触发阈值':>8} {'峰值':>10} {'压缩次数':>9} {'峰值/窗口':>10} {'风险':<12}")
for at in (0.50, 0.60, 0.70, 0.85, 0.95):
    b = Budget(compact_at=at)
    r = simulate(steps, b)
    ratio = r["峰值"] / b.window_tokens
    risk = "过频（丢信息）" if at <= 0.5 else ("推荐" if at <= 0.7 else "过晚（本例侥幸未溢出）")
    print(f"  {at:>8.0%} {r['峰值']:>10,} {r['压缩次数']:>9} {ratio:>10.2f} {risk:<12}")
print("  读法：阈值太低会频繁压缩（每次都丢信息、多花钱）；太高则可能一步就溢出。")
print("        60–70% 是留了「压缩失败还能重试」余量的位置")

print("\n③ 截断上限的影响（单次工具输出上限 vs 峰值）")
print(f"  {'输出上限':>8} {'峰值':>10} {'是否爆窗口':>10} {'信息损失风险':<14}")
for cap in (400, 800, 1_200, 3_000, 8_000):
    b = Budget(output_cap=cap)
    r = simulate(steps, b)
    print(f"  {cap:>8,} {r['峰值']:>10,} {('是' if r['爆窗口于第'] else '否'):>10} "
          f"{'高（可能切掉关键行）' if cap <= 400 else '低':<14}")
print("  读法：截断太小会切掉关键上下文（例如失败的断言），太大则等于不截断；")
print("        正确做法是**结构化截断**（保留命中行前后 N 行、保留失败断言），而不是粗暴砍长度")

print("\n④ 成本口径：一次任务的 token 消耗（含压缩）")
def task_cost(steps: List[Step], b: Budget, price_in: float = 3.0,
              price_out: float = 15.0) -> Dict[str, float]:
    """粗略：每步输入 = 当前上下文，输出 = 思考 + 工具调用"""
    ctx = b.sys_prompt + b.repo_map
    tok_in = tok_out = 0
    for s in steps:
        tok_in += ctx
        tok_out += s.think + s.call
        out = min(s.output_raw, b.output_cap) if b.truncate_output else s.output_raw
        ctx += s.think + s.call + out
        if ctx > b.window_tokens * b.compact_at:
            tok_in += ctx
            tok_out += b.compact_cost_tokens
            ctx = b.sys_prompt + b.repo_map + b.compact_keep + b.compact_cost_tokens
    cost = tok_in / 1e6 * price_in + tok_out / 1e6 * price_out
    return {"输入token": tok_in, "输出token": tok_out, "成本($)": cost}
for label, b in (("无压缩（会爆）", Budget(compact_at=1.01)), ("有压缩", Budget())):
    c = task_cost(steps, b)
    print(f"  {label:<14} 输入 {c['输入token']:>10,}  输出 {c['输出token']:>8,}  "
          f"约 ${c['成本($)']:.2f}/任务")
print("  读法：**上下文就是成本**（每步都要重新发送）——这是「少而准的上下文」比「塞满窗口」更省钱的量化原因；")
print("        也是仓库地图这类**稳定前缀**值得做缓存的原因（串 [[anthropic-22]] 的 prefix caching）")

print("\n⑤ 八小时投入分配：把上面四条结论映射回时间表")
PLAN = [("agent 循环 + 提示", 2.0, "没有它什么都验证不了"),
        ("六个工具 + 错误语义", 2.0, "工具误用会让循环空转"),
        ("上下文工程（地图/截断/压缩）", 2.0, "20 步后失忆"),
        ("验证闭环（测试为真值）", 1.5, "假成功，评测全是错的"),
        ("观测与轨迹回放", 0.5, "无法定位回归")]
print(f"  {'投入块':<30} {'小时':>5}  省略后的症状")
total = 0.0
for name, h, risk in PLAN:
    total += h
    print(f"  {name:<30} {h:>5.1f}  {risk}")
print(f"  合计 {total:.1f} 小时（架构时间 = 0：单线程循环不需要架构）")
print("  读法：如果要砍时间，先砍观测（降级为日志），再砍压缩（改用更激进的截断）；")
print("        **绝不能砍验证闭环** —— 没有它，八小时的产出无法评估")
```

预期输出要点（实跑）：① 三种上下文策略对照显示**不截断时第 27 步就突破 12.8 万窗口**，只截断仍会缓慢增长到 10.6 万，**加上 60% 触发的压缩后峰值稳定在 7.7 万、可跑完 60 步**——这就是「上下文工程占两小时」的理由；② 压缩阈值敏感性显示 50% 会压缩过频（丢信息 + 多花钱）、95% 太晚（一步就溢出），**60–70% 是留出重试余量的位置**；③ 截断上限的取舍：太小会切掉关键行（失败的断言），太大等于没截断，正确做法是**结构化截断**；④ 成本口径显示**压缩把单任务成本从 \$10.62 降到 \$7.51（降 29%）**——上下文就是成本（每步都要重发），这也是稳定前缀值得做缓存的原因；⑤ 八小时投入分配表把结论映射回时间表，并明确「**绝不能砍验证闭环**」。

## 常见追问

- **追问**：为什么不做多 agent？它看起来更快。
  - 要点：**并行写入 + 不共享上下文**会让风格/边界/模式的隐式决策互相冲突（Cognition 的原则 2）。只读扇出（搜索类子 agent）是安全的，因为它不产生写入决策；真正可行的多 agent 是「**写入保持单线程 + 上下文可共享**」的形态（串 [[cognition-02]]）。
- **追问**：没有测试的仓库怎么办？
  - 要点：让 agent **先写最小复现**（一个能触发 bug 的脚本/命令），把它当作临时测试；若连复现都写不出，就退回「人工确认 + 明确说明不确定性」，**不要假装成功**。
- **追问**：怎么防止它改坏别的地方？
  - 要点：三件事——**小步提交**（每次编辑后可回退）、**跑全量测试**（不只跑相关测试）、以及**diff 审查**（对高风险改动要求人工确认，串 [[agents-11]]）。
- **追问**：工具应该粗粒度还是细粒度？
  - 要点：偏**粗粒度但语义明确**：`edit` 要求唯一匹配（避免改错位置）、`search` 直接返回带行号的片段（避免「搜索完还要读整个文件」）。粒度太细会让模型编排负担变重，太粗会失去控制。
- **追问**：八小时里怎么做评测？
  - 要点：挑 **10–20 个真实小任务**（能在一两次编辑内解决、有测试或可写复现），记录**成功率、步数、token 成本、失败类型**。这比跑 SWE-bench 更有信息量，也更能指导下一步（串 [[cognition-07]] 对 SWE-bench 的批评）。
- **追问**：如果只剩四小时呢？
  - 要点：砍顺序是——观测 → 压缩（改用激进结构化截断）→ 工具数量（合并 shell/search）→ **但保留验证闭环与单线程循环**。换句话说：**宁可少功能，也要保住「能验证」**。

## 相关题目

- [[cognition-02]]：多 agent 的两种立场如何调和——本轮的「砍掉多 agent」在下一位题里有更细的边界。
- [[cognition-03]]：第一轮一半时间在找代码——对应本题的仓库地图与搜索工具设计。
- [[cognition-07]]：如何评估自主软件工程 agent，以及 SWE-bench 通过率为何误导。
- [[agents-09]]：agent 循环的终止与成本限制，对应本题的重试上限与预算。
- [[agents-07]]：长时间运行 agent 的记忆设计，是「砍掉向量库」之后的进阶形态。

## 参考资料与归属

- **Don't Build Multi-Agents（Cognition 博客）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 5 节「砍掉多 agent」的依据（Share context、Actions carry implicit decisions 两条原则，以及「并行子 agent 的隐式决策冲突导致脆弱」）来自这篇。
- **Multi-Agents: What's Actually Working（Cognition 博客）** —— Walden Yan (Cognition)，2026-04-22：<https://cognition.ai/blog/multi-agents-working>。第 5 节「什么时候可以引入多 agent」的边界（只读子 agent 近似工具调用；可行形态是「多 agent 贡献智能、写入保持单线程」；上下文工程的需要没有消失）来自这篇。
- **解释 ReAct 模式以及它相比 chain-of-thought 解决了什么（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1 节「观察—决策—工具调用—结果回灌」的循环形态取自该专题文档。
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）** —— Jimenez et al. (ICLR 2024)，2023-10-10：<https://arxiv.org/abs/2310.06770>。第 4 节「以测试为真值」的评测口径来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（窗口 20 万 token、系统提示 1,500、仓库地图 6,000、压缩阈值 60%、压缩开销 3,000、摘要保留 4,000、输出上限 1,200、40 步、各类工具输出规模、token 单价 \$3/\$15、八小时分配）都是为演示方法与预算而构造的**示例参数与显式假设**；真实数值取决于模型、工具实现与任务分布，必须用自己的轨迹数据校准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
