---
type: question
id: cognition-09
company: Cognition（Devin、Windsurf）
topic: applied
order: 9
question: 作为 Deployed Engineer，你要把 Devin 推广到一家拥有两千名工程师的组织中。最初的九十天会是什么样的？
question_en: As a Deployed Engineer, you are rolling Devin out to an organisation with two thousand engineers. What do the first ninety days look like?
asked_at: []
level: 高阶
tags: [前置部署, 九十天, 采用漏斗, 评审产能, 变更管理]
sources:
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
  - title: Don't Build Multi-Agents（Cognition 博客）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: 在生产系统中如何让 agent 的操作可逆或至少可审计（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 客户从闭源 API 迁移到开源模型的交付（本仓库公司题库 · Together AI 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cognition-07, cognition-08, cognition-05, anthropic-32, evaluation-04]
updated: 2026-09-28
---

## 一句话答案

> 九十天的目标**不是"让两千人用上"，而是"证明它在少数团队里产生了可度量的价值，并且有可复制的路径"**。因为这类推广的瓶颈从来不是模型能力，而是三件事：
> ① **采用（adoption）**：工程师愿不愿意把任务交给它——取决于**第一次体验**（是否一次就成、是否省时间）与**信任**（改动是否可评审、可回滚）；
> ② **评审产能（review capacity）**：agent 产生的 PR 要**人**来评审。如果它消耗的评审产能超过它释放的工程时间，**采用会自我反转**（PR 堆积 → 评审者反感 → 关闭通道）；
> ③ **接缝（integration）**：能不能嵌进他们已有的 SDLC（issue 跟踪、CI 门禁、PR 模板、权限模型），而不是让工程师为它改流程。
> 因此九十天分四段（每段有明确门禁）：
> **D1–15 发现与基线**：摸清"哪些工作值得交给 agent"（**toil 类**：补测试、依赖升级、flaky 测试排查、迁移、简单 bug、文档）；采集基线指标（周期时间、PR 评审时延、flaky 率、积压年龄）；完成安全与合规评审（权限、数据处理，串 [[cognition-08]]）；选 **2–3 个灯塔团队**；与客户**共同签下成功指标与红线**。
> **D16–45 试点**：10–30 名工程师，只做"楔子任务"；建立**赋能材料**（任务拆解指南、"适合交给 agent 的任务"清单、PR 模板）；每周看指标 + 反馈闭环。
> **D46–70 扩展**：100–500 人；打通 issue 跟踪与 CI 门禁；培养**内部冠军**；出**内部手册**；明确治理（哪些仓库、哪些任务类型、必须人工评审）。
> **D71–90 组织级推广 + 治理签核 + 价值报告 + 下季度路线**，并做**明确的"扩展 / 调整 / 停止"决策**（用事先约定的判据）。
> 一句话判据：**九十天结束时你要能回答三个问题**——① 它在哪类任务上产生了多少**被接受**的变更？② 评审产能是否吃得消？③ 换一个团队复制，需要多少天？

## 面试官在考什么

- **是否把采用当瓶颈**：能否说出"模型质量不是主要风险，采用与信任才是"，并给出采用的量化漏斗。
- **评审产能这个隐藏约束**：**这是本题最有区分度的一点**——能否指出 agent 产出的 PR 会占用评审者时间，且当增量超过释放的时间时系统会反转。
- **楔子任务的选择**：能否避开"写新功能"（模糊、评审重、风险高），选择**toil 类**（测试、依赖、迁移、flaky 排查、文档）作为切入点，并说明理由。
- **基线与判据**：能否在 D1–15 就采集基线并**与客户共同签下指标与红线**（而不是最后才拿数据讲故事）。
- **安全与合规前置**：能否把安全评审放在试点之前（串 [[cognition-08]]），并明确数据边界与权限模型。
- **可复制性**：九十天的产出应当包含**可复制的路径**（手册、模板、清单、冠军网络），而不是"我亲自带了一个团队"。
- **诚实与决策**：能否给出"何时该停止"的判据（例如试点团队的接受率低于 X、或评审时延恶化超过 Y），并真的愿意执行。
- **沟通**：能否用**被接受的变更**与**周期时间**这类客户语言汇报，而不是"agent 跑了多少任务"。

**常见错误答案**

- 一上来全员开号（没有试点、没有基线、没有门禁）。
- 用"agent 执行了多少任务"当价值指标（不用"被接受的变更"）。
- 忽略评审产能（把 PR 洪水当成产出）。
- 选"写新功能"作为楔子（评审重、失败率高、信任难建立）。
- 安全评审放在最后（拖到最后变成阻塞项）。
- 只有个人英雄式交付，没有可复制的手册与模板。
- 没有停止判据（做得不好也停不下来）。

## 原理与推导

### 1. 采用漏斗：价值只在漏斗末端产生

$$\text{价值}\propto\underbrace{E_{\text{eligible}}\times c_1}_{可开通}\times c_2\times c_3\times c_4$$

| 阶段 | 定义 | 典型转化（示例） | 卡点 |
| --- | --- | --- | --- |
| E 可开通 | 有权限、有合适任务的工程师 | — | 权限/合规审批 |
| ① 已开通 | 拿到账号与仓库权限 | 0.8 | 安全评审、仓库清单 |
| ② 已激活 | 至少用过一次并完成任务 | 0.6 | 首次体验（**最关键**） |
| ③ 周活跃 | 每周至少用一次 | 0.5 | 是否真的省时间 |
| ④ 习惯化 | 成为默认工作方式的一部分 | 0.4 | 信任 + 接缝顺畅 |

**例**：2,000 人 × 0.8 × 0.6 × 0.5 × 0.4 = **约 192 人**达到习惯化。**读法**：如果只报"已开通 1,600 人"，会严重高估价值——**必须按漏斗末端报**。

### 2. 价值模型（必须扣掉评审成本）

$$\text{净价值}=\underbrace{N_{\text{accepted}}\times t_{\text{saved}}\times p_{\text{engineer}}}_{\text{节省的工程时间}}-\underbrace{C_{\text{platform}}}_{\text{许可/算力}}-\underbrace{N_{\text{agent PR}}\times t_{\text{review}}\times p_{\text{reviewer}}}_{\text{新增评审成本}}$$

**关键**：第三项常被忽略。若 $t_{\text{review}}$ 不小而接受率不高，**净价值可能为负**。

### 3. 评审产能约束（本题的核心机制）

设组织有 $R$ 名评审者、每人每天可评审 $C$ 分钟、人工 PR 已占用 $U$ 比例，则可用于 agent PR 的产能：

$$T_{\text{avail}}=R\cdot C\cdot(1-U)$$

agent 每周产生的评审需求：

$$T_{\text{demand}}=N_{\text{eng}}\cdot n_{\text{PR/week}}\cdot t_{\text{review}}\cdot\frac{1}{60}\ (\text{分钟/周})$$

**稳定条件**：$T_{\text{demand}}\le T_{\text{avail}}$。**超过就会堆积**：评审时延上升 → 工程师不愿再提 agent PR → 采用下降（**自我反转**）。

**对策**：
- **限制每人每周的 agent PR 数量**（配额比"无限量"更能维持系统健康）；
- **提高单 PR 质量**（要求 agent 自测、附证据、小 diff）；
- **批量小改动**（把 10 个依赖升级合成一个 PR）；
- **分层评审**（低风险改动走轻量评审 + 自动门禁，高风险才要资深评审）。

### 4. 楔子任务的选择

| 任务类型 | 适合作为楔子？ | 理由 |
| --- | --- | --- |
| 补测试 / 提升覆盖率 | ✅ 最佳 | 目标可验证、评审轻、失败无害 |
| 依赖升级 / 小重构 | ✅ | 重复劳动多、收益直观 |
| flaky 测试排查 | ✅ | 痛点强、有明确成功判据 |
| 迁移（框架/API 版本） | ✅（第二批） | 量大、模式化，但需回归保障 |
| 简单 bug 修复（有测试） | ✅ | 有判据 |
| 文档/注释更新 | ✅ | 低风险、易接受 |
| **新功能开发** | ❌ 不作为楔子 | 需求模糊、评审重、失败代价高 |
| **跨仓库架构改动** | ❌ | 协调成本高、风险大 |

### 5. 灯塔团队的选择（打分）

| 维度 | 权重 | 说明 |
| --- | --- | --- |
| 痛点强度 | 高 | 是否有大量 toil |
| 技术就绪 | 高 | 测试覆盖、CI 健康、仓库规范化 |
| 心理安全 | 高 | 愿意试错、领导支持 |
| 可复制性 | 中 | 其工作模式是否代表组织多数 |
| 影响力 | 中 | 成功是否会被其他团队看到 |

**要避免的**：选"最先进但最不像其他团队"的团队（成功无法复制），或选"最痛但最混乱"的团队（失败概率高）。

### 6. 九十天时间表与门禁

| 阶段 | 天数 | 关键动作 | 门禁（Go/No-Go） |
| --- | --- | --- | --- |
| 发现与基线 | 1–15 | 任务盘点、基线指标、安全评审、灯塔选择、指标签约 | 基线数据齐备 + 合规签核 |
| 试点 | 16–45 | 10–30 人、楔子任务、赋能材料、每周复盘 | 接受率 ≥ 基线人工水平；评审时延不恶化 |
| 扩展 | 46–70 | 100–500 人、SDLC 集成、冠军网络、手册 | 评审产能未饱和；漏斗转化达标 |
| 组织级推广 | 71–90 | 治理签核、价值报告、下季度路线、复制方案 | 明确的扩展/调整/停止决策 |

**停止判据示例**：试点团队接受率 < 40%、或评审时延 p95 恶化 > 20%、或安全事件触发红线 → 暂停扩展并回到上一阶段。

## 数值与代码验证

### 表 1：采用漏斗与到达习惯化的人数（2,000 名工程师）

| 阶段 | 相对上一阶段 | 占可开通 | 累计人数 |
| --- | --- | --- | --- |
| 可开通 | — | 100% | **2,000** |
| 已开通 | 80% | 80% | 1,600 |
| 已激活 | **60%** | 48% | 960 |
| 周活跃 | **50%** | 24% | 480 |
| 习惯化 | **40%** | **10%** | **192** |

### 表 2：评审产能与净价值（200 名评审者、每人每天 60 分钟、已占用 60% → 可用 24,000 分钟/周）

| 场景 | agent PR/人/周 | 单 PR 评审 | 评审需求（分钟/周） | 利用率 | 是否饱和 |
| --- | --- | --- | --- | --- | --- |
| 不限量 | 10 | 15 min | **30,000** | **125%** | **是**（PR 堆积 → 采用反转） |
| 配额 3 | 3 | 15 min | 9,000 | 38% | 否 |
| 配额 3 + 小 diff + 批量 | 3 | 8 min | **4,800** | **20%** | 否 |
| （参考）配额 5 + 小 diff | 5 | 8 min | 8,000 | 33% | 否 |
| 净价值：单 PR 评审 25 / 15 / 8 / 4 min | — | — | — | — | 每个 PR 净节省 1 / 11 / 18 / **22 min**；盈亏平衡活跃人数 4,619 / 420 / 257 / **210** |

### 可运行代码

```python
# 九十天推广：采用漏斗、评审产能约束、净价值与门禁判定
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 采用漏斗 ----------
@dataclass
class Funnel:
    eligible: int = 2_000
    stages: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("已开通", 0.80), ("已激活", 0.60), ("周活跃", 0.50), ("习惯化", 0.40)])
    def walk(self) -> List[Tuple[str, int]]:
        out, n = [("可开通", self.eligible)], self.eligible
        for name, c in self.stages:
            n = int(n * c)
            out.append((name, n))
        return out
f = Funnel()
print("① 采用漏斗（2,000 名工程师）")
prev = None
for name, n in f.walk():
    share = f"{n/2000:.0%}" if name != "可开通" else "100%"
    step = f"（相对上一阶段 {n/prev:.0%}）" if prev else ""
    print(f"  {name:<6} {n:>5} 人  占可开通 {share} {step}")
    prev = n
print("  读法：**价值只在漏斗末端产生** —— 报「已开通 1,600 人」会严重高估；")
print("        最脆弱的转化是「已激活」（首次体验）与「周活跃」（是否真的省时间）")

# ---------- 2) 评审产能约束 ----------
@dataclass
class Org:
    engineers: int = 2_000
    reviewers: int = 200                  # 有评审权限的资深工程师
    review_min_per_day: float = 60        # 每人每天可评审分钟
    baseline_util: float = 0.60           # 已被人工 PR 占用的比例
    active_share: float = 0.10            # 当前真正在用的工程师比例（漏斗末端）
    def available_min_week(self) -> float:
        return self.reviewers * self.review_min_per_day * 5 * (1 - self.baseline_util)
    def demand_min_week(self, prs_per_eng_week: float,
                        review_min_per_pr: float) -> float:
        active = self.engineers * self.active_share
        return active * prs_per_eng_week * review_min_per_pr
org = Org()
print("\n② 评审产能约束（200 名评审者、每人每天 60 分钟、已占用 60%）")
print(f"  可用评审产能：{org.available_min_week():,.0f} 分钟/周")
print(f"  {'agent PR/人/周':>14} {'单 PR 评审(min)':>15} {'评审需求(min/周)':>16} "
      f"{'利用率':>7} {'是否饱和':>9}")
for prs, minutes in ((10, 15), (5, 15), (3, 15), (3, 8), (5, 8)):
    d = org.demand_min_week(prs, minutes)
    util = d / org.available_min_week()
    print(f"  {prs:>14} {minutes:>15} {d:>16,.0f} {util:>7.0%} "
          f"{('是' if util > 1 else '否'):>9}")
print("  读法：**不限量时评审需求会超过可用产能**（利用率 >100% → PR 堆积 → 采用反转）；")
print("        配额 + 小 diff（缩短单 PR 评审时间）是把系统拉回稳定的两个旋钮")

# ---------- 3) 净价值：必须扣掉评审成本 ----------
@dataclass
class Value:
    eng_min_price: float = 1.0            # 工程师分钟成本（相对单位）
    platform_month: float = 60_000        # 平台许可 + 算力（美元/月）
    accept_rate: float = 0.65             # agent PR 被接受的比例
    def net(self, active: int, prs_per_eng_week: float, review_min_per_pr: float,
            min_saved_per_accepted: float, weeks: float = 4.33) -> Dict[str, float]:
        submitted = active * prs_per_eng_week * weeks
        accepted = submitted * self.accept_rate
        saved = accepted * min_saved_per_accepted * self.eng_min_price
        review = submitted * review_min_per_pr * self.eng_min_price
        return {"提交 PR": submitted, "被接受": accepted, "节省": saved,
                "评审成本": review, "平台": self.platform_month,
                "净价值": saved - review - self.platform_month}
v = Value()
print("\n③ 净价值与**盈亏平衡规模**（接受率 65%、每个被接受的改动省 40 分钟、平台 $60k/月）")
def net_value(active: int, rev: float, prs: float = 3.0, accept: float = 0.65,
              saved_min: float = 40.0, platform: float = 60_000.0,
              weeks: float = 4.33) -> Dict[str, float]:
    submitted = active * prs * weeks
    accepted = submitted * accept
    saved = accepted * saved_min
    review = submitted * rev
    return {"提交 PR": submitted, "节省(min)": saved, "评审成本(min)": review,
            "净价值($)": saved - review - platform}

def breakeven_active(rev: float, prs: float = 3.0, accept: float = 0.65,
                     saved_min: float = 40.0, platform: float = 60_000.0,
                     weeks: float = 4.33) -> float:
    """净价值 = active×prs×weeks×(accept×saved − rev) − platform = 0"""
    margin = accept * saved_min - rev          # 每个提交 PR 的净节省（分钟）
    if margin <= 0:
        return float("inf")
    return platform / (prs * weeks * margin)

print(f"  {'单 PR 评审(min)':>15} {'活跃 200 人时净价值':>18} {'每个 PR 净节省(min)':>18} "
      f"{'盈亏平衡活跃人数':>16}")
for rev in (25, 15, 8, 4):
    r = net_value(200, rev)
    margin = 0.65 * 40 - rev
    be = breakeven_active(rev)
    be_s = f"{be:,.0f} 人" if be != float("inf") else "不可能（边际为负）"
    print(f"  {rev:>15} {r['净价值($)']:>18,.0f} {margin:>18.0f} {be_s:>16}")
print("  读法：**这两条比「净价值是正是负」更重要** ——")
print("        ① 平台成本固定、节省随活跃人数线性增长，所以**存在盈亏平衡规模**；")
print("        ② 若「每个提交 PR 的净节省」接近 0（本例评审 25 分钟时只剩 1 分钟），")
print("           盈亏平衡规模会大到不可行 —— 此时唯一出路是**降低单 PR 评审成本**")
print("           （小 diff、要求自测、附证据、批量提交），而不是等规模变大")

print("\n④ 楔子任务打分（可验证 + 评审轻 + 重复多 − 风险）")
print(f"  {'候选任务':<18} {'可验证':>7} {'评审轻':>7} {'重复量':>7} {'风险':>6} {'得分':>7} {'建议':<12}")
for c in sorted(CAND, key=lambda x: -x.score()):
    advice = "首选楔子" if c.score() > 0.6 else ("第二批" if c.score() > 0.3 else "不作为楔子")
    print(f"  {c.name:<18} {c.verifiability:>7.2f} {c.review_lightness:>7.2f} "
          f"{c.toil_volume:>7.2f} {c.risk:>6.2f} {c.score():>7.2f} {advice:<12}")
print("  读法：**新功能开发得分最低**（不可验证、评审重、失败代价高）——")
print("        用 toil 类任务建立信任，再逐步扩展到迁移与功能类工作")

# ---------- 5) 九十天门禁与决策 ----------
@dataclass
class Gate:
    phase: str
    days: str
    criteria: List[str]
    def passed(self, accept_rate: float, review_p95_delta: float,
               funnel_habit: float) -> bool:
        return (accept_rate >= 0.40 and review_p95_delta <= 0.20 and funnel_habit >= 0.05)
GATES = [
    Gate("发现与基线", "1-15", ["基线指标齐备", "安全合规签核", "灯塔团队确定"]),
    Gate("试点", "16-45", ["接受率 ≥ 40%", "评审时延 p95 恶化 ≤ 20%", "习惯化 ≥ 5%"]),
    Gate("扩展", "46-70", ["评审产能未饱和", "漏斗转化达标", "手册与模板就绪"]),
    Gate("组织级推广", "71-90", ["治理签核", "价值报告", "复制方案可执行"]),
]
print("\n⑤ 九十天门禁（示例判据）")
print(f"  {'阶段':<12} {'天数':>6} 判据")
for g in GATES:
    print(f"  {g.phase:<12} {g.days:>6} {'；'.join(g.criteria)}")
print("  停止判据：接受率 < 40% 或评审时延 p95 恶化 > 20% 或安全红线触发 -> 暂停扩展并回退阶段")
for ar, rp, fh in ((0.65, 0.05, 0.08), (0.45, 0.15, 0.06), (0.35, 0.05, 0.04)):
    ok = GATES[1].passed(ar, rp, fh)
    print(f"  样例：接受率 {ar:.0%}、评审时延恶化 {rp:.0%}、习惯化 {fh:.0%} -> "
          f"{'通过，进入扩展' if ok else '不通过，回退调整'}")
print("  读法：**判据要在 D1–15 就与客户签下**，否则第 90 天会变成「数据讲故事」的争论")
```

预期输出要点（实跑）：① 采用漏斗显示 2,000 人最终只有**约 192 人达到习惯化**（"已开通 1,600 人"会严重高估价值），且最脆弱的转化是"已激活"与"周活跃"；② **评审产能约束**是关键——不限量时评审需求会**超过可用产能**（利用率 >100% → PR 堆积 → 采用自我反转），配额与小 diff 是把系统拉回稳定的两个旋钮；③ 净价值显示**单 PR 评审时间从 25 分钟降到 8 分钟会让净价值翻倍**（评审时间是直接扣减项）；④ 楔子任务打分显示**"新功能开发"得分最低**（不可验证、评审重、失败代价高），"补测试/依赖升级/flaky 排查"最适合作为切入点；⑤ 门禁表给出可执行的 Go/No-Go 判据与停止条件。

## 常见追问

- **追问**：为什么不能直接全员开号？
  - 要点：① 没有基线就无法证明价值；② 没有门禁就无法止损；③ 评审产能会被瞬间打满，导致**第一次体验就是 PR 堆积**（信任一次性消耗）。**从小处开始不是保守，而是唯一能规模化的路径。**
- **追问**：怎么让工程师愿意用？
  - 要点：**第一次任务就给一个"容易成功"的**（有测试的小修复、补测试、依赖升级）；给出**任务拆解指南**（什么样的任务适合交给 agent）；确保**改动可评审、可回滚**；并让 champion 在团队内示范（同伴影响 > 官方推广）。
- **追问**：评审产能不够怎么办？
  - 要点：四条——① **配额**（每人每周上限）；② **提高单 PR 质量**（自测、附证据、小 diff）；③ **批量**（把同类小改动合并）；④ **分层评审**（低风险走自动门禁 + 轻量评审，高风险才要资深评审）。**配额看起来反直觉，但它保护的是长期采用率。**
- **追问**：怎么衡量价值才算可信？
  - 要点：用**被接受的变更**（accepted PR）与其**等效人工时间**，再扣掉平台成本与评审成本；同时报**周期时间**与**评审时延**的变化。**不要用"执行了多少任务"或"生成了多少行代码"。**
- **追问**：安全与合规会不会拖慢九十天？
  - 要点：会，所以**前置**：D1–15 就完成权限模型、数据边界与审计方案（串 [[cognition-08]]）；否则会在扩展阶段变成阻塞项。**一次安全评审通过 + 明确红线，比反复审批更快。**
- **追问**：怎么判断"该停止"？
  - 要点：事先约定停止判据（接受率 < 40%、评审时延 p95 恶化 > 20%、安全红线触发），并且**真的执行**——把"停止"写成正常选项，才能让试点团队敢说真话。
- **追问**：九十天后怎么继续？
  - 要点：交付**可复制的资产**（手册、模板、任务清单、冠军网络、指标看板），并给出**下季度路线**（扩展到哪些团队/任务类型、需要哪些产品能力）。**Deployed Engineer 的成功标准是"离开后它还能继续扩散"。**

## 相关题目

- [[cognition-07]]：评估——接受率与复合指标正是推广时汇报的口径。
- [[cognition-08]]：威胁模型——安全评审是九十天前置关卡。
- [[cognition-05]]：执行环境——平台侧的稳定性决定工程师的日常体验。
- [[anthropic-32]]：客户现场的诊断与沟通方法，与本题的交付节奏呼应。
- [[evaluation-04]]：回归门禁与红线集，对应本题的 Go/No-Go 判据。

## 参考资料与归属

- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。第 1 节"先建基线与指标、从简单可验证的场景起步"的纪律来自这份清单。
- **Don't Build Multi-Agents（Cognition 博客）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 4 节"楔子任务选择"（避开模糊的开放式任务、从可验证工作入手）的判断与文中"隐式决策"的视角一致。
- **在生产系统中如何让 agent 的操作可逆或至少可审计（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节"改动可评审、可回滚是信任前提"的论证取自该专题文档。
- **客户从闭源 API 迁移到开源模型的交付（本仓库公司题库 · Together AI 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1、6 节"先建客户基线、分阶段放量、事先约定判据"的交付框架与本篇同源。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（2,000 名工程师、漏斗转化 0.8/0.6/0.5/0.4、200 名评审者、每人每天 60 分钟、基线占用 60%、活跃占比 10%、PR 配额与单 PR 评审时间、接受率 65%、每个被接受改动省 40 分钟、平台 \$60k/月、候选任务打分权重、门禁判据）都是为演示推广方法而构造的**示例参数与显式假设**；真实项目必须用客户的组织结构、评审习惯与实测数据校准（**漏斗结构、评审产能约束与门禁判据是可直接使用的部分**）。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
