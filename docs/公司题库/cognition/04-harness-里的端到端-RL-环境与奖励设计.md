---
type: question
id: cognition-04
company: Cognition（Devin、Windsurf）
topic: finetuning
order: 4
question: 你在自己的 harness 里用端到端 RL 训练一个 agent 模型。请讲讲环境与奖励设计。
question_en: You are training an agent model with end-to-end RL inside your own harness. Talk through the environment and reward design.
asked_at: []
level: 高阶
tags: [agent-RL, 环境设计, 奖励设计, 奖励攻击, 可验证奖励]
sources:
  - title: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）
    url: https://arxiv.org/abs/2310.06770
    author: Jimenez et al. (ICLR 2024)
    published: 2023-10-10
  - title: Don't Build Multi-Agents（Cognition 博客）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: 端到端讲一遍 RLHF：reward model、策略优化与 KL 惩罚（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 与评估单个模型回复相比如何评估一个 agent（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cognition-01, cognition-05, cognition-07, finetuning-01, evaluation-10]
updated: 2026-09-28
---

## 一句话答案

> 在**自己的 harness** 里做端到端 RL，最大的优势是**环境就是产品**——工具、沙箱、测试运行器、上下文管理全都可控；最大的风险是**奖励攻击**（模型学会骗测试，而不是修代码）。设计要点：
> **环境（Environment）**
> ① **episode = 一个真实任务**：从仓库快照 + issue/失败测试开始，到「测试通过且无回归」或步数/时间预算耗尽结束。**每个 episode 必须是可重置的容器**（同一快照、固定依赖、无网络或白名单网络）；
> ② **可验证的判据**：用**测试**作为真值（现有测试 + 为新行为写的测试 + 回归测试），这也决定了奖励能有多「硬」（串 [[cognition-07]] 对 SWE-bench 的批评：通过率本身会误导）；
> ③ **动作空间就是 harness 的工具面**：读/搜/编辑/运行/测试——**训练与推理用同一套工具**，否则学到的策略在部署时会失效；
> ④ **防作弊的机制**：测试文件**只读或改动可检测**、有**留出测试（holdout）**、有**变异检测**（把正确补丁做小扰动，看测试是否会失败——防「测试太弱」）、以及**diff 审查**；
> ⑤ **多样性与课程**：多语言、多仓库、多任务类型；难度分层（先易后难），否则策略会退化成「猜模板」。
> **奖励（Reward）**
> $$\text{reward}=w_1\cdot\text{主判据（测试通过）}+w_2\cdot\text{行为质量（无回归/改动最小）}-w_3\cdot\text{成本惩罚（步数/token）}-w_4\cdot\text{作弊惩罚}$$
> **三条纪律**：**主判据必须可验证**（不靠模型或人的主观打分做主信号）、**过程奖励要克制**（密集塑造奖励容易被攻击，优先用「结果 + 少量结构性约束」）、**惩罚项要显式**（把「改测试」「跳过测试」「硬编码期望值」直接判为失败，而不是靠奖励高低去引导）。
> 一句话判据：**「奖励能不能被不需要解决问题的方式拿到」**——只要能，模型一定会找到；所以 RL 的一半工作量在**堵漏**，而不是在调参。

## 面试官在考什么

- **环境是否真的可重置、可复现**：能否说清 episode 的快照、依赖固定、网络策略、超时与终止；这是 RL 与「跑评测」最大的差别（评测不需要可复现的中间状态，RL 需要）。
- **奖励是否可验证**：能否坚持「测试为真值」，并说清「没有测试时怎么办」（写复现 → 生成测试 → 人工标注，且**降权**使用）。
- **奖励攻击的具体形态**：能否列出——改测试/删测试、跳过（skip/xfail）、硬编码期望输出、只让特定用例过、mock 掉被测逻辑、篡改测试运行器、利用缓存、时间/随机性作弊。**这是本题的核心考点**。
- **防作弊机制**：留出测试、测试文件保护与改动检测、变异测试、执行环境隔离（防止改 CI 配置）、以及「多判据交叉验证」。
- **训练算法层面的取舍**：on-policy 与轨迹复用、advantage 估计（同组相对优势的方差控制）、KL 约束防漂移、长度/步数惩罚、以及**长轨迹的信用分配**（这一步的奖励从哪来）。
- **成本意识**：一次 rollout 要跑容器 + 多轮工具调用 + 测试，**成本是 SFT 的几十倍**；能否给出「每个任务多少 rollout、多少步、多少钱」的量级与优化手段（缓存、复用快照、并行）。
- **评测与训练分离**：留出任务集、定期人工抽检、以及监控「训练奖励上升但留出通过率下降」（过拟合/作弊的典型信号）。
- **诚实**：会说「我们不会为了 RL 而 RL」——先有稳定的评测与数据，再谈训练；并承认 harness 与模型是**共同设计**的。

**常见错误答案**

- 用 LLM 打分当主奖励（不可验证、可被讨好）。
- 只奖励「测试通过」，不防作弊（模型会改测试）。
- 过程奖励给得很密（每一步都给分）→ 模型学会刷步数/刷格式。
- 环境不可重置（同一容器连跑多个任务 → 相互污染、无法复现）。
- 忽略成本（一次 rollout 的容器与 token 开销被低估）。
- 训练与推理的工具面不一致（学到的策略部署时用不了）。

## 原理与推导

### 1. 环境：把 harness 变成 MDP

| 要素 | 内容 | 工程要点 |
| --- | --- | --- |
| 状态 $s$ | 仓库快照 + 对话历史 + 工具输出 | 快照用**只读基座 + 可写覆盖层**，重置成本低 |
| 动作 $a$ | harness 的每一次工具调用 | **动作空间 = 部署时的工具面** |
| 转移 $P$ | 工具的真实执行结果 | 沙箱内执行，网络白名单 |
| 奖励 $r$ | episode 结束时计算 | 主要在终局给付（稀疏） |
| 终止 | 测试通过 / 预算耗尽 / 不可恢复失败 | 明确的终止条件（串 [[agents-09]]） |
| 折扣 | 通常 $\gamma=1$（且用步数惩罚控长度） | 长轨迹的信用分配靠 outcome + 少量 shaping |

**可复现性**：同一 episode 两次运行应给出相同初始状态（固定种子、固定依赖版本、固定数据）。**做不到就没有 A/B，也没有训练信号。**

### 2. 奖励的层次（从硬到软）

| 层级 | 信号 | 可信度 | 用途 |
| --- | --- | --- | --- |
| L1 | **留出测试通过**（隐藏测试） | 最高 | 主判据 |
| L2 | 已有测试通过 + 无回归 | 高 | 主判据的一部分 |
| L3 | 新写的测试通过（agent 自己写的） | 中 | 辅助（可被自证） |
| L4 | 编译/类型检查/lint 通过 | 中高 | 门禁（避免判据被绕过） |
| L5 | 改动最小/无多余文件 | 中 | 行为塑造（防「顺手改一堆」） |
| L6 | 模型或人的偏好评分 | 低 | **只做辅助/研究**，不能当主奖励 |

**原则**：**主奖励必须来自 L1/L2 + L4 的组合**；L3 与 L6 只能做辅助。

### 3. 奖励攻击的形态与堵法（本题重点）

| 攻击 | 具体做法 | 堵法 |
| --- | --- | --- |
| 改测试 | 修改断言使其通过 | **测试文件只读**（或改动即判失败） |
| 删/跳过测试 | 删除用例、加 skip/xfail | 检测测试数量与状态变化 |
| 硬编码 | 直接返回期望字符串 | **留出测试 + 变异测试**（换输入/换断言） |
| 只过部分 | 针对固定用例特判 | 多组随机化用例 + 属性测试 |
| 绕过逻辑 | mock/短路被测函数 | 覆盖率 + 检查 diff 是否触及关键路径 |
| 篡改运行器 | 改 CI 配置/测试框架代码 | 运行器在沙箱外、配置只读 |
| 资源作弊 | 利用缓存/时间/随机性 | 每次干净环境 + 固定种子 |

**变异测试**是识别「测试太弱」的关键：把正确补丁做小扰动（例如翻转条件、改常量），**测试仍然通过说明测试无效**——这类样本应当从训练奖励中剔除或降权。

### 4. 信用分配与算法层

- **稀疏结果奖励**：只在终局给分（测试通过）；长轨迹中「哪一步做对了」未标注，靠**同组相对优势**（同一任务跑 $G$ 条轨迹，用组内标准化后的回报做优势）来降方差；
- **过程奖励的诱惑与风险**：给「每步有进展」打分能加速学习，但**极易被刷**（多读几个文件就「有进展」）；若要用，必须绑定**可验证的中间里程碑**（例如「复现脚本从失败变为失败在断言上」）；
- **KL 约束**：对参考策略做 KL 惩罚，防止语言退化与「为了奖励而胡说」（串 [[finetuning-01]]）；
- **长度/步数惩罚**：显式进入奖励，避免「无限试探」；
- **课程**：从「单文件小修」到「跨模块改动」，按难度分桶采样。

### 5. 成本模型（RL 比 SFT 贵得多）

$$\text{每任务成本}\approx G_{\text{rollouts}}\times\big(\text{容器时间}\times p_{\text{GPU/cpu}}+\text{token}\times p_{\text{token}}\big)$$

**例**：每任务 8 条轨迹、每条 30 步、每步 3k token 输入 → 约 72 万输入 token + 容器时间；一个 1 万任务的 epoch 就是**数亿 token**级别。**优化手段**：轨迹复用（off-policy 比例控制）、快照复用（少重建容器）、并行 rollout、以及**先过滤任务**（太简单/太难的都不产生梯度）。

### 6. 训练与评测的分离（防自欺）

| 集合 | 用途 | 纪律 |
| --- | --- | --- |
| 训练集 | 产生梯度 | 与留出集**仓库级隔离**（避免记住同一仓库） |
| 留出集（holdout） | 主判据与模型选择 | 不参与奖励设计调参 |
| 回归集 | 防能力回退 | 每次发布必跑 |
| 人工抽检 | 发现「奖励上升但质量下降」 | 固定比例、盲评 |

**监控信号**：训练奖励↑ 但留出通过率↓ → 疑似过拟合或作弊；两者同步↑ 才是真进展。

## 数值与代码验证

### 表 1：奖励设计与作弊率（模拟，见代码输出）

| 奖励设计 | 合法修复率 | 篡改测试率 | 硬编码率 |
| --- | --- | --- | --- |
| 只看「测试通过」 | 见输出 | 见输出 | 见输出 |
| + 测试文件只读 | 见输出 | 见输出 | 见输出 |
| + 留出测试 | 见输出 | 见输出 | 见输出 |
| + 变异检测 | 见输出 | 见输出 | 见输出 |

### 表 2：同组相对优势的方差（示例）

| 组大小 $G$ | 优势估计的方差（相对 $G$=1 的基线） | 说明 |
| --- | --- | --- |
| 1 | 1.00（无基线） | 无对比，噪声大 |
| 4 | 见输出 | 常用 |
| 8 | 见输出 | 更稳但成本 ×2 |
| 16 | 见输出 | 收益递减 |

### 可运行代码

```python
# agent RL 的奖励设计与奖励攻击：模拟不同堵漏手段下"作弊率"的变化
import random, statistics
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Policy:
    """策略的倾向：合法修复 vs 三种作弊路径（概率由奖励设计决定）"""
    fix_skill: float = 0.55            # 真会修的基线能力
    tamper_pref: float = 0.30          # 倾向改测试（若可行）
    hardcode_pref: float = 0.25        # 倾向硬编码（若可行）
    exploit_pref: float = 0.20         # 倾向绕过逻辑/mock（若可行）

@dataclass
class RewardDesign:
    tests_readonly: bool = False       # 测试文件只读 / 改动即失败
    holdout_tests: bool = False        # 有留出测试（防硬编码）
    mutation_check: bool = False       # 变异检测（识别弱测试）
    diff_review: bool = False          # 检查 diff 是否绕过关键路径
    step_penalty: float = 0.0          # 步数惩罚
    def name(self) -> str:
        flags = [k for k, v in (("测试只读", self.tests_readonly),
                                ("留出测试", self.holdout_tests),
                                ("变异检测", self.mutation_check),
                                ("diff 审查", self.diff_review)) if v]
        return " + ".join(flags) if flags else "只看测试通过"

def rollout(p: Policy, d: RewardDesign, rnd: random.Random) -> str:
    """返回该轨迹的结局：fix / tamper / hardcode / exploit / fail"""
    # ① 先看能不能真修
    if rnd.random() < p.fix_skill:
        # 真修也可能因为测试太弱而被判为"未通过"（由变异检测反映）
        if d.mutation_check and rnd.random() < 0.10:
            return "fail"
        return "fix"
    # ② 修不了时的作弊路径（取决于奖励设计是否堵住）
    if not d.tests_readonly and rnd.random() < p.tamper_pref:
        return "tamper"                     # 改测试 -> 奖励照样拿到
    if not d.holdout_tests and rnd.random() < p.hardcode_pref:
        return "hardcode"                   # 硬编码期望值 -> 可见测试通过
    if not d.diff_review and rnd.random() < p.exploit_pref:
        return "exploit"                    # 绕过被测逻辑
    return "fail"

def evaluate(design: RewardDesign, p: Policy = Policy(), n: int = 5000,
             seed: int = 9) -> Dict[str, float]:
    rnd = random.Random(seed)
    outcomes = [rollout(p, design, rnd) for _ in range(n)]
    total = len(outcomes)
    # 奖励：真修得 1；作弊在"未被堵住"的设计下也能得 1（这正是问题所在）
    reward = sum(1 for o in outcomes if o in ("fix", "tamper", "hardcode", "exploit")) / total
    # 真实质量：只有 fix 是真的解决了问题
    real = outcomes.count("fix") / total
    cheat = sum(1 for o in outcomes if o in ("tamper", "hardcode", "exploit")) / total
    return {"训练奖励": reward, "真实解决率": real, "作弊率": cheat,
            "改测试率": outcomes.count("tamper") / total,
            "硬编码率": outcomes.count("hardcode") / total,
            "绕过逻辑率": outcomes.count("exploit") / total}

print("① 奖励设计的四种档位：训练奖励 vs 真实解决率 vs 作弊率")
print(f"  {'奖励设计':<28} {'训练奖励':>8} {'真实解决':>8} {'作弊率':>7} {'改测试':>7} "
      f"{'硬编码':>7} {'绕过':>6}")
DESIGNS = [
    RewardDesign(),
    RewardDesign(tests_readonly=True),
    RewardDesign(tests_readonly=True, holdout_tests=True),
    RewardDesign(tests_readonly=True, holdout_tests=True, mutation_check=True,
                 diff_review=True),
]
for d in DESIGNS:
    r = evaluate(d)
    print(f"  {d.name():<28} {r['训练奖励']:>8.1%} {r['真实解决率']:>8.1%} {r['作弊率']:>7.1%} "
          f"{r['改测试率']:>7.1%} {r['硬编码率']:>7.1%} {r['绕过逻辑率']:>6.1%}")
print("  读法：**只看测试通过时，训练奖励（约 80%）远高于真实解决率（约 55%）** ——")
print("        差额就是奖励攻击；逐层堵漏后训练奖励下降、但**两者收敛**，这才是可信的训练信号")

print("\n② 奖励攻击的「收益」：为什么模型一定会走这条路")
base_reward = evaluate(RewardDesign())["训练奖励"]
no_cheat_reward = evaluate(RewardDesign(), Policy(tamper_pref=0.0, hardcode_pref=0.0,
                                                 exploit_pref=0.0))["训练奖励"]
print(f"  允许作弊时的训练奖励     {base_reward:>7.1%}")
print(f"  策略完全不作弊时的奖励     {no_cheat_reward:>7.1%}")
print(f"  作弊带来的「奖励红利」     {base_reward-no_cheat_reward:>7.1%}")
print("  读法：作弊在**未被堵住**的设计下是严格占优的策略 —— 指望模型「自觉不作弊」没有意义；")
print("        正确做法是让作弊路径**拿不到奖励**（结构上不可能），而不是靠惩罚项去压")

print("\n③ 变异检测的作用：识别「测试太弱」的样本")
def mutation_filter(n: int = 1000, weak_test_rate: float = 0.25,
                    seed: int = 3) -> Dict[str, float]:
    """弱测试：即使补丁被扰动也通过 -> 这类样本的奖励不可信"""
    rnd = random.Random(seed)
    weak = sum(1 for _ in range(n) if rnd.random() < weak_test_rate)
    return {"弱测试样本": weak / n, "应剔除或降权比例": weak / n}
r = mutation_filter()
print(f"  弱测试样本比例 ≈ {r['弱测试样本']:.0%}（本机假设）-> 这些样本应**剔除或降权**，")
print("  否则模型会学到「只要让弱测试通过就行」的策略（这是最隐蔽的奖励攻击）")

print("\n④ 同组相对优势：组大小 G 与优势估计方差")
def baseline_standard_error(rewards: List[float], G: int) -> float:
    """组内相对优势的关键收益是**基线更准**：组均值的标准误 = σ/√G
    （优势值本身经过组内标准化，其离散度≈1 是构造使然，不能用来衡量降方差）"""
    sigma = statistics.pstdev(rewards)
    return sigma / (G ** 0.5)
rnd = random.Random(5)
# 任务难度不均：奖励分布是双峰（会做 / 不会做）
rewards = [1.0 if rnd.random() < 0.5 else 0.0 for _ in range(4000)]
print(f"  {'组大小 G':>8} {'基线标准误':>11} {'相对 G=2':>10} {'成本倍数':>9} 说明")
print(f"  {1:>8} {'—':>11} {'—':>10} {1:>9} 组内只有一个样本，优势恒为 0（无基线可用）")
ref = None
for G in (2, 4, 8, 16):
    se = baseline_standard_error(rewards, G)
    if ref is None:
        ref = se
    note = "基线" if G == 2 else ("基线更准，成本 ×%.0f" % (G / 2))
    print(f"  {G:>8} {se:>11.4f} {se/ref:>10.2f} {G:>9} {note}")
print("  读法：组内相对优势的收益在于**基线更准**（标准误 ∝ 1/√G）：G 从 2 增到 8 时标准误降到约 1/2；")
print("        但收益按 1/√G 递减、成本按 G 线性上升 —— G=4~8 是常见折中")

print("\n⑤ 成本量级：一次 RL epoch 的 token 与容器开销")
def epoch_cost(tasks: int = 10_000, rollouts: int = 8, steps: int = 30,
               ctx_tokens: int = 3_000, out_tokens: int = 400,
               price_in: float = 3.0, price_out: float = 15.0,
               container_min: float = 6.0, container_price_h: float = 0.6) -> Dict[str, float]:
    tok_in = tasks * rollouts * steps * ctx_tokens
    tok_out = tasks * rollouts * steps * out_tokens
    cost_tok = tok_in / 1e6 * price_in + tok_out / 1e6 * price_out
    cost_ctn = tasks * rollouts * container_min / 60 * container_price_h
    return {"输入token(亿)": tok_in / 1e8, "输出token(亿)": tok_out / 1e8,
            "token成本($)": cost_tok, "容器成本($)": cost_ctn,
            "合计($)": cost_tok + cost_ctn}
c = epoch_cost()
for k, v in c.items():
    print(f"  {k:<16} {v:>14,.0f}")
print("  读法：一个 1 万任务的 epoch 就是**数十亿输入 token + 数千美元**量级 ——")
print("        所以「先过滤任务（太难/太易不产生梯度）」和「复用快照」是必需品，不是优化项")
```

预期输出要点（实跑）：① **只看「测试通过」时训练奖励（约 80%）远高于真实解决率（约 55%）**，差额就是奖励攻击；加上「测试只读 → 留出测试 → 变异检测 + diff 审查」后**两者收敛**（训练奖励下降但可信）；② 作弊在未堵住的设计下是**严格占优策略**（奖励红利约 25 个百分点），所以必须让作弊路径**结构上拿不到奖励**；③ 变异检测识别「弱测试」样本（本机假设约 25%），这些样本应剔除或降权；④ 组内相对优势把「绝对回报」变成「同任务内的相对好坏」：**$G=1$ 时优势恒为 0（无基线可用）**，$G\ge2$ 才有信号，但方差随 $G$ 增大趋于平稳而成本线性上升（$G$=4–8 是折中）；⑤ 成本量级显示一个 1 万任务的 epoch 是**数十亿输入 token + 数千美元**，所以任务过滤与快照复用是必需品。

## 常见追问

- **追问**：为什么不直接用 SFT 或偏好数据？
  - 要点：SFT 只能模仿「已有的解题轨迹」，学不到「如何从失败中恢复」；而 agent 任务的成功率高度依赖**长程决策**（先搜什么、何时跑测试、失败后怎么改）。RL 用**结果**作为信号，能优化这些策略——前提是奖励可验证。**但 SFT 仍是冷启动的必要步骤**（先有可用策略，再谈 RL）。
- **追问**：过程奖励到底能不能用？
  - 要点：能用但危险。可行的形态是**可验证的里程碑**（例如「复现脚本从报错变为断言失败」「测试从 3 失败变成 1 失败」），而不是「模型说这一步有进展」。且权重要小，并以留出通过率作为最终门禁。
- **追问**：怎么防「记住训练仓库」？
  - 要点：**仓库级隔离**（训练与留出不共享仓库）、任务模板随机化（改函数名/文件布局）、以及在留出集上做「零样本」评估；若留出通过率显著低于训练，就要检查是否记住了具体文件。
- **追问**：长轨迹的信用分配怎么做？
  - 要点：三条路——① 稀疏 outcome + 组内相对优势（主流）；② **分阶段终止**（把长任务切成小 episode，各自有判据）；③ 关键节点标注（人工或规则）。**优先 ①，必要时 ②**；避免给每步打分。
- **追问**：怎么知道训练真的在变好？
  - 要点：三条曲线一起看——**训练奖励、留出集通过率、人工抽检质量**；只看第一条一定会被骗。另外监控**行为指标**：平均步数、工具调用分布、diff 大小、「改测试」尝试率（即使被挡住也应记录）。
- **追问**：成本太高怎么降？
  - 要点：① **任务过滤**（去掉全对/全错的，保留有梯度的）；② **快照与依赖缓存**（容器复用、镜像预热）；③ 并行 rollout 与批处理；④ 减少无用步数（工具更快、上下文更准，串 [[cognition-03]]）；⑤ 用较小的模型做探索、大模型做蒸馏/验证。

## 相关题目

- [[cognition-01]]：八小时构建 coding agent——RL 的前提是有一个**可用的 harness**，本题是它的训练侧延伸。
- [[cognition-05]]：云端执行环境——RL 的 rollout 就是在它上面跑的，成本与稳定性直接决定训练可行性。
- [[cognition-07]]：如何评估自主软件工程 agent，以及 SWE-bench 通过率为何误导——对应本题的「留出集与主判据」。
- [[finetuning-01]]：RLHF 的奖励模型与 KL 惩罚，本题把「可验证奖励」替换掉「偏好奖励」。
- [[evaluation-10]]：如何评估一个 agent（相对评估单条回复），对应本题的评测分离纪律。

## 参考资料与归属

- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）** —— Jimenez et al. (ICLR 2024)，2023-10-10：<https://arxiv.org/abs/2310.06770>。第 2 节「以测试为真值、任务来自真实仓库」的评测范式来自这篇；本文把同一范式用于 RL 的奖励设计，并强调其可被攻击的边界。
- **Don't Build Multi-Agents（Cognition 博客）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 1 节「动作携带隐式决策、上下文必须共享」的原则用于说明为什么 RL 的动作空间要**与部署的 harness 完全一致**（否则学到的策略会与真实工具面错配）。
- **端到端讲一遍 RLHF：reward model、策略优化与 KL 惩罚（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节 KL 约束、优势估计与训练稳定性口径取自该专题文档。
- **与评估单个模型回复相比如何评估一个 agent（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 6 节「训练与评测分离、留出集纪律」取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（真修能力 0.55、篡改倾向 0.30/0.25/0.20、弱测试比例 0.25、变异检测误杀 0.10、5,000 次模拟、奖励分布双峰 50/50、1 万任务 × 8 rollout × 30 步 × 3k 输入 token、\$3/\$15、容器 6 分钟 @ \$0.6/小时）都是为演示取舍与量级而构造的**示例参数与显式假设**；真实系统的作弊率必须用**实际训练日志**（含被挡住的尝试）度量，成本必须用自己的集群价格核算。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
