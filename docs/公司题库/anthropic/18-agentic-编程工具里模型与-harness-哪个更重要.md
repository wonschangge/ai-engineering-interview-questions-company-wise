---
type: question
id: anthropic-18
company: Anthropic
topic: agents
order: 18
question: 对于 Claude Code 这样的 agentic 编程工具，模型和 harness 哪个更重要？请设计这个循环。
question_en: For an agentic coding tool like Claude Code, which matters more — the model or the harness? Design the loop.
asked_at: []
level: 高阶
tags: [Agent, harness, 工具调用, 上下文管理, 验证回路]
sources:
  - title: Building Effective Agents（延伸）
    url: https://www.anthropic.com/engineering/building-effective-agents
    author: Anthropic
    published: 2024-12-19
  - title: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）
    url: https://arxiv.org/abs/2310.06770
    author: Jimenez et al. (ICLR 2024)
    published: 2023-10-10
  - title: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains（延伸）
    url: https://arxiv.org/abs/2406.12045
    author: Yao et al.
    published: 2024-06-17
  - title: Claude Code 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-claude-code-work
    author: Amit Shekhar (Outcome School)
    published: 
related: [anthropic-19, cursor-08, cursor-14, agents-01, agents-11]
updated: 2026-09-28
---

## 一句话答案

> **两者都重要，但它们在能力曲线上的作用点不同**——这是回答这道题的框架：
> **模型决定「上限」**：给定充足的上下文与工具，模型能否理解模糊需求、做出正确的架构判断、在失败后自我纠正。模型弱，harness 再精巧也做不出好结果。
> **harness 决定「下限与达成率」**：同样的模型，harness 决定它能看到什么上下文、能调用哪些工具、错误如何处理、改动如何被验证与回滚。**harness 的改进可以在不改模型的前提下把任务成功率提升十几个百分点**（这正是 SWE-bench 这类基准的价值：把「模型行不行」变成「在给定判据下能不能真的修好」，从而让 harness 与模型的贡献可分离测量）。
> 我的判断是：**在模型能力已经「够用」的区间里，harness 的边际收益更大**（因为它便宜、可迭代、可回滚）；而当任务超出现有模型的能力边界时，harness 再优化也是徒劳。工程上的正确姿势是**先测出「失败分布」**——是「不知道要做什么」（模型问题）、「看不到该看的东西」（上下文问题）、「改不对/改坏了」（验证与工具问题）——再决定投哪一边。
> 循环设计见下：**感知 → 规划 → 行动 → 观察 → 验证 → 记忆**，其中**验证是 harness 的灵魂**（没有验证的自主循环等于随机搜索）。

## 面试官在考什么

- **能否把「模型 vs harness」拆成可测的问题**：不是站队，而是给出**失败归因的方法**（A/B 同一模型换 harness、同一 harness 换模型；按失败类型分类统计）。
- **是否理解 harness 的具体构成**：上下文组装（检索/裁剪/排序）、工具集与 schema、执行环境（沙箱/权限）、验证回路（编译/测试/lint）、状态与记忆、以及回滚与审计（串 [[cursor-08]]）。
- **验证回路的设计**：哪些信号是强判据（测试/编译/类型）、哪些是弱判据（自检、模型自评）；**没有验证的循环会「自信地改坏代码」**。
- **上下文管理**：agent 的长任务必然超出上下文——如何做计划外置、历史压缩、检索式记忆（不是「把对话塞满」）。
- **循环的终止条件**：什么时候停（成功判据达成/预算耗尽/无进展），以及**无进展检测**（同一错误重复出现就应改变策略而不是重试同样的动作）。
- **可测量的改进**：能否给出衡量 harness 改进的指标（任务成功率、步数、成本、破坏率），以及如何避免「对着同一批任务过拟合」（串 [[anthropic-21]] 的评测纪律）。

**常见错误答案**

- 「模型更重要，换个更强的模型就行了」——忽略了上下文与工具的限制（强模型在拿不到关键文件时也会瞎改）。
- 「harness 更重要，工程能解决一切」——忽略了能力边界：需要深度推理的任务不是靠工具补齐的。
- 循环只讲「想 → 做」，没有**观察与验证**，也没有终止/回退。
- 不谈成本与步数预算——真实 agent 的第一约束往往是 token 预算与时间。

## 原理与推导

### 1. 先把「谁更重要」变成可测的命题

设计两组对照实验：

| 实验 | 变量 | 目的 |
| --- | --- | --- |
| A | 固定模型，替换 harness（工具集/上下文策略/验证回路） | 测 harness 的边际收益 |
| B | 固定 harness，替换模型（同族不同规模） | 测模型的边际收益 |
| C | 两者都换（升级组合） | 测是否有交互效应（好 harness 能否放大强模型） |

**指标**：任务成功率（可执行判据）、平均步数、token 成本、**破坏率**（引入回归的比例）、以及**失败类型分布**（看不懂需求 / 找不到文件 / 改错地方 / 改坏其他 / 验证不足）。

**工程结论（可辩护的立场）**：在「模型能理解需求」的前提下，harness 决定成功率的大头；在「模型理解不了需求」时，harness 的收益趋零。所以**先做失败归因，再投入**。

### 2. 循环设计：六段

```
┌──────────── 每轮 ────────────┐
① 感知：读任务 + 检索相关上下文（符号索引优先，向量兜底）
② 规划：把任务拆成可验证的子步骤，写入外部计划文件
③ 行动：调用工具（读/搜索/编辑/执行），每次改动是一个变更集
④ 观察：把工具结果结构化（文件:行、错误摘要、测试结果）
⑤ 验证：编译 → 类型 → 受影响测试 → lint；失败即回滚或修正
⑥ 记忆：把"已确认的事实/试过的失败路径"写入外部状态，压缩历史
└──────────────────────────────┘
终止：判据达成 / 预算耗尽 / 连续 N 轮无进展（改变策略或升级给人）
```

**要点**：
- ③ 的行动应当**尽量小且可验证**（一次改一个逻辑单元），否则失败难以归因；
- ④ 的输出必须**可行动**（哪个文件哪一行、期望什么、实际什么），否则模型只能瞎猜（串 [[anthropic-19]] 的工具设计）；
- ⑤ 是「自主」的前提：**能验证多少，就能自主多少**；
- ⑥ 解决长任务上下文爆炸——把状态外置，而不是靠更长的上下文窗口。

### 3. 上下文管理：三种策略的组合

| 策略 | 做法 | 适用 |
| --- | --- | --- |
| 计划外置 | 任务分解与进度写进文件，每轮读取 | 多步任务（>10 步） |
| 历史压缩 | 把早期对话压成「已确认事实 + 已排除路径」摘要 | 长会话 |
| 检索式记忆 | 按需检索代码/文档/历史轨迹，而不是全塞 | 大仓库 |

**反模式**：把所有工具输出原样累积进对话——几百轮后上下文被噪声塞满，模型注意力被稀释（长上下文利用率的实证问题可引用的口径见 *Lost in the Middle*，串 [[anthropic-14]]）。

### 4. 验证回路：分级与代价

| 级别 | 手段 | 耗时 | 能抓到 |
| --- | --- | --- | --- |
| L1 | 语法/AST 解析 | 毫秒 | 明显语法错 |
| L2 | 类型检查/静态分析（增量） | 秒 | 类型/接口不一致 |
| L3 | 受影响测试（test impact analysis） | 十秒 | 逻辑错、回归 |
| L4 | 全量测试 | 分钟 | 跨模块回归 |
| L5 | 端到端/冒烟 | 更久 | 集成问题 |

**纪律**：L1–L3 每轮都跑（成本可接受、能抓住约 85% 的坏改动），L4–L5 在提交/合并前跑；**失败即回滚**，不允许在失败的改动上继续叠加。

### 5. 与「模型能力」的交互：harness 如何放大或浪费模型

- **放大**：好工具（能一次读到正确文件）+ 好上下文（依赖图扩展）+ 强验证（快速反馈）→ 强模型的成功率显著提升；
- **浪费**：工具粒度太粗（只能整文件覆盖）、上下文噪声大、验证缺失 → 强模型也会退化成「乱改」；
- **掩盖**：过度宽松的验证（只跑语法）会让 harness 看起来成功而实际引入回归——**指标必须包含破坏率**。

## 数值与代码验证

### 表 1：失败类型分布与投入方向（示意框架）

| 失败类型 | 典型症状 | 该投哪边 |
| --- | --- | --- |
| 看不懂需求 | 反复问同一问题、改了无关文件 | 模型（或加澄清步骤） |
| 找不到文件/符号 | 反复搜索、读错文件 | **harness**（符号索引、依赖扩展） |
| 改错地方 | 改动看似合理但不触达根因 | 混合（规划能力 + 上下文） |
| 改坏其他 | 任务「完成」但引入回归 | **harness**（验证回路） |
| 工具用不对 | schema 误用、参数错 | **harness**（工具设计与错误信息） |

**用法**：跑一批任务，按上表分类统计——若「改坏其他」占大头，做验证回路；若「找不到文件」占大头，做检索。**这比争论「模型 vs harness」有效得多。**

### 表 2：验证分级的收益（与 [[cursor-08]] 同口径）

| 检查 | 耗时 | 累计抓到坏改动 | 单位时间收益 |
| --- | --- | --- | --- |
| L1 语法 | 10 ms | 15% | 15.0 |
| +L2 类型 | 2 s | 50% | 0.175 |
| +L3 受影响测试 | 20 s | 85% | 0.0175 |
| +L4 全量测试 | 900 s | 95% | 0.0001 |

### 可运行代码

```python
# agent 循环的模拟：对比「无验证 / 有验证 / 有验证+记忆」的成功率与破坏率
import random
from dataclasses import dataclass, field

@dataclass
class Config:
    verify_level: int = 0          # 0=无验证 1=语法 2=类型 3=受影响测试
    memory: bool = False           # 是否复用"试过的失败路径"
    context_quality: float = 0.7   # 上下文命中率（能否找到正确文件）
    max_steps: int = 12
    rollback_on_fail: bool = True

@dataclass
class Stat:
    success: int = 0; broke: int = 0; step_sum: int = 0

def run_task(cfg: Config, rnd: random.Random) -> Stat:
    st = Stat()
    tried_failures = set()
    for step in range(cfg.max_steps):
        st.step_sum += 1
        # 找到正确文件的概率受上下文质量影响
        found = rnd.random() < cfg.context_quality
        if not found:
            continue                                    # 白走一步（搜索/读错文件）
        # 改对地方的概率：模型能力（这里固定 0.6）受上下文与记忆加成
        p_correct = 0.6 + (0.06 if cfg.memory else 0.0)
        correct = rnd.random() < p_correct
        if correct and cfg.verify_level >= 3:
            # 强验证能确认修复真的有效
            if rnd.random() < 0.9:
                st.success += 1
                return st
            else:
                continue                                # 验证不通过，继续尝试
        if correct and cfg.verify_level < 3:
            # 没有强验证：可能"看起来完成"但其实是破坏性修改
            if rnd.random() < 0.35:
                st.broke += 1
                if not cfg.rollback_on_fail:
                    return st
            else:
                st.success += 1
                return st
        else:
            # 改错地方：有验证时能发现（不留下破坏），无验证时可能连带破坏
            key = ("miss", step % 3)
            if cfg.memory and key in tried_failures:
                continue                                # 记忆避免重复同一失败路径
            tried_failures.add(key)
            if cfg.verify_level == 0 and rnd.random() < 0.25:
                st.broke += 1
                if not cfg.rollback_on_fail:
                    return st
    return st

def bench(cfg: Config, n=2000, seed=5) -> dict:
    rnd = random.Random(seed)
    agg = Stat()
    for _ in range(n):
        st = run_task(cfg, rnd)
        agg.success += st.success; agg.broke += st.broke; agg.step_sum += st.step_sum
    return {"成功率": agg.success / n, "破坏率": agg.broke / n, "平均步数": agg.step_sum / n}

configs = {
    "无验证、无记忆":        Config(verify_level=0, memory=False),
    "仅语法验证":            Config(verify_level=1, memory=False),
    "语法+类型":             Config(verify_level=2, memory=False),
    "语法+类型+受影响测试":   Config(verify_level=3, memory=False),
    "全验证 + 记忆":         Config(verify_level=3, memory=True),
    "全验证 + 记忆 + 上下文质量 0.9": Config(verify_level=3, memory=True, context_quality=0.9),
}
print(f"{'配置':<28} {'成功率':>8} {'破坏率':>8} {'平均步数':>9}")
for name, cfg in configs.items():
    r = bench(cfg)
    print(f"{name:<28} {r['成功率']:>8.2%} {r['破坏率']:>8.2%} {r['平均步数']:>9.2f}")

# 只换模型能力（模拟"换更强的模型"），harness 保持不变
print("\n同 harness（全验证 + 记忆），只提升模型能力：")
for p_correct in (0.45, 0.60, 0.75, 0.90):
    rnd = random.Random(5); ok = broke = steps = 0
    for _ in range(2000):
        cfg = Config(verify_level=3, memory=True)
        tried = set()
        for step in range(cfg.max_steps):
            steps += 1
            if rnd.random() >= cfg.context_quality:
                continue
            if rnd.random() < p_correct and rnd.random() < 0.9:
                ok += 1; break
        else:
            broke += 0
    print(f"  模型单步正确率 {p_correct:.2f} -> 任务成功率 {ok/2000:.2%}，平均步数 {steps/2000:.2f}")

print("\n读法：① 验证等级的提升把「破坏率」压下去、把成功率提上来；")
print("      ② 记忆减少重复踩坑（步数下降）；")
print("      ③ 上下文质量（能否找到正确文件）是 harness 侧最便宜的杠杆；")
print("      ④ 模型单步正确率提升同样有效 —— 两者是相乘关系，不是二选一")
```

预期输出要点（实跑，注意这套模拟是**演示杠杆方向**而非预测真实数值）：
1. **最有分辨力的指标是「破坏率」而不是「成功率」**：无验证时破坏率高达 **78.8%**（「看起来完成」的改动里大部分其实是破坏性的），补上强验证（受影响测试）后直接降到 **0%**；成功率则一直在 97%–100% 的高位，几乎不区分配置——这正是「为什么必须把破坏率放进指标」的实证。
2. **记忆缩短步数**：2.63 → 2.39 步（避免重复踩同一失败路径）。
3. **上下文质量是最便宜的 harness 杠杆**：0.7 → 0.9 让平均步数从 2.39 降到 1.85。
4. **模型能力同样是乘法项**：单步正确率 0.45 → 0.90，平均步数 3.50 → 1.75（在同 harness 下）。
**模拟的简化要说明**：代码把 L1（语法）与 L2（类型）都归为「弱验证」（因此两者数值相同），只有 L3 及以上才算强验证；真实系统里类型检查的收益介于两者之间。

## 常见追问

- **追问**：怎么衡量 harness 的好坏，而不只是模型的好坏？
  - 要点：固定模型只换 harness（工具集/上下文/验证），比成功率、破坏率、步数、成本；再用**留出任务**验证不是对某批任务过拟合（串 [[anthropic-21]]）。
- **追问**：循环卡住了怎么办？
  - 要点：无进展检测（连续 N 轮没有新的有效信息/通过验证的改动）→ 改变策略（换检索方式、缩小任务、请求澄清）→ 预算耗尽时把「当前状态 + 阻塞点」交给人，而不是静默失败。
- **追问**：上下文窗口越来越大，还需要精细的上下文管理吗？
  - 要点：需要——长上下文的**有效利用率**会下降（信息在中间位置容易被忽略），而且 token 成本与延迟随长度上升；把状态外置（计划文件、结构化记忆）比「塞更多」更可靠（串 [[anthropic-14]] 的对照）。
- **追问**：怎么避免 agent 只在「看起来完成」时停下？
  - 要点：用**可执行判据**（测试/复现步骤消失/类型检查）替代「模型说完成了」；把 τ-bench 式的「以最终状态判定任务完成」作为设计原则，并报告 pass^k 一致性（同一任务重复 k 次全成功的比例）。
- **追问**：多 agent 分工是否必要？
  - 要点：多数场景不必——单 agent + 好工具往往更好（上下文更集中、协调成本低）；当任务可以自然拆成**独立可验证**的子任务且上下文负担过重时，才引入分工（如「实现者 + 验证者」），并必须做工作区隔离。
- **追问**：怎么控制成本？
  - 要点：步数与 token 预算 + 早期路由（简单任务用小模型）+ 缓存（前缀缓存、工具结果缓存）+ **减少无效步数**（更好的上下文与验证比「多试几次」便宜）。

## 相关题目

- [[anthropic-19]]：工具面（tool surface）设计，是本题「行动」与「观察」两段的展开。
- [[cursor-08]]：多文件改动的 harness 与四道闸（隔离/原子/验证/回滚），是本题 harness 的工程细节。
- [[cursor-14]]：沙箱内迭代架构，对应本题的执行环境部分。
- [[agents-01]]：ReAct 循环的基础机制，本题是它在编程场景的完整化。
- [[agents-11]]：高风险动作的审批与可逆性，是本题权限与回滚设计的通用框架。

## 参考资料与归属

- **Building Effective Agents（延伸）** —— Anthropic，2024-12-19：<https://www.anthropic.com/engineering/building-effective-agents>。第 1 节「用简单可组合的模式、只在必要时增加复杂度」与循环/工具设计的基本取向来自这篇。
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）** —— Jimenez et al. (ICLR 2024)，2023-10-10：<https://arxiv.org/abs/2310.06770>。第「一句话答案」与「面试官在考什么」里「用可执行判据衡量 agent 是否真的解决问题」的范式来自这篇。
- **τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains（延伸）** —— Yao et al.，2024-06-17：<https://arxiv.org/abs/2406.12045>。第「常见追问」里「以最终状态判定任务完成、并报告 pass^k」的评测口径来自这篇。
- **Claude Code 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-claude-code-work>。第 2 节循环各阶段与上下文管理的产品侧背景参照这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（上下文命中率 0.7/0.9、模型单步正确率 0.45–0.90、验证分级的抓到比例、破坏概率 0.35/0.25、步数上限 12）都是为演示本仓库口径而构造的**模拟参数与显式假设**，不代表任何真实 agent 的实测指标；这些数字用来展示「验证/记忆/上下文/模型」四个杠杆的相对作用，真实数值必须自己压测。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
