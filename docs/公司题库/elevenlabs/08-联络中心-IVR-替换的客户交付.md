---
type: question
id: elevenlabs-08
company: ElevenLabs
topic: applied
order: 8
question: 一个联络中心想用语音 agent 替换它的 IVR。请主导这次客户交付。
question_en: A contact centre wants to replace its IVR with a voice agent. Lead the customer engagement.
asked_at: []
level: 高阶
tags: [联络中心, IVR, 客户交付, 承接率, ROI, 分阶段放量]
sources:
  - title: 设计实时语音 AI Agent
    url: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Handling Overload（Google SRE Book 第 21 章）（延伸）
    url: https://sre.google/sre-book/handling-overload/
    author: Google SRE
    published: 
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
  - title: 为实时语音 agent 做 latency 预算（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [elevenlabs-07, elevenlabs-04, elevenlabs-01, evaluation-04, anthropic-32]
updated: 2026-09-28
---

## 一句话答案

> 这类项目的交付有明确的**先后与门禁**，主导交付就是把它们排对：
> ① **先量现状、再谈目标**：拿到 IVR 的真实数据——**承接率（containment）、AHT、转人工率、放弃率、CSAT、每次接触成本**，以及**来电驱动分析**（Top N 意图占多少话量）。**没有这份基线，任何目标承诺都是空谈**；
> ② **把「替换 IVR」重新定义成「用语音 agent 覆盖 Top N 意图 + 保留按键菜单兜底」**：IVR 的问题通常不是「菜单」，而是**层级太深、不识别自然语言、转人工要等**；所以目标应是**承接率↑、转人工率↓、AHT↓、CSAT↑**——而不是「把菜单换成 AI」；
> ③ **设计必须包含四件东西**：**意图覆盖表**（每个意图的自动化档位与降级路径）、**转人工设计**（带你上下文的转接，而不是重排队）、**按键兜底**（老年/听障/嘈杂环境的可达性，也是合规要求）、**身份核验**（涉及账户操作时）；
> ④ **分阶段放量 + 明确门禁**：影子（不接真实流量）→ 溢出/下班时段 → 低风险意图 → 逐步扩大；**每阶段有量化门禁与回滚开关**，并设「kill criteria」（承接率低于下限或 CSAT 掉超过阈值就回退）；
> ⑤ **ROI 要算「每次接触成本」而不是「替代多少人」**：$\text{节省}=(\text{被承接的话量}\times\text{AHT}\times\text{人力单价})-\text{平台成本}$，并用**保守/目标/乐观三档**给出盈亏平衡——**同时说明「承接率差 10 个百分点，ROI 会怎样」**（这才是负责任的承诺）。
> 一句话判据：**交付的成败不在于语音 agent 多聪明，而在于「意图覆盖是否真实、转人工是否顺畅、门禁是否敢回退」**——把这三件事做扎实，ROI 自然出现。

## 面试官在考什么

- **是否先建基线**：能否列出 IVR 的核心指标与「来电驱动分析」，并强调「没有基线就没有目标」。
- **对 IVR 问题的理解**：能否指出 IVR 的真实痛点是**层级深、不支持自然语言、转人工等待长**（而不是「没有 AI」）。
- **意图覆盖的现实主义**：能否说「Top 10 意图占 60–80% 话量，先做这些」，而不是承诺 100% 覆盖。
- **转人工设计**：能否说出「带上下文的转接」（避免用户重述）与「排队/回呼」策略；以及转人工失败时的兜底。
- **可达性与合规**：按键菜单为什么必须保留（无障碍、嘈杂环境、老年用户），以及录音告知/AI 披露/数据合规。
- **ROI 的诚实性**：能否给出三档预测、盈亏平衡点、以及「最坏情况仍可接受」的判断（串 [[evaluation-04]] 的门禁思路）。
- **交付节奏与治理**：影子 → 小流量 → 扩大 → 全会话；每周指标复盘；kill criteria 与回滚演练。
- **常见失败原因**：能否主动列出（承诺过度、无转人工、整合脆弱、不测 CSAT、忽视合规）。

**常见错误答案**

- 直接讲架构（模型、ASR/TTS 选型），不建基线与目标。
- 承诺 100% 承接或「替代 X 名坐席」。
- 没有按键兜底与转人工设计。
- ROI 只算单点，不给区间与敏感性。
- 没有 kill criteria（上了就下不来）。
- 忽略「转人工后用户体验」（重排队、重复叙述）。

## 原理与推导

### 1. 基线与目标（先量后改）

| 指标 | 含义 | 典型改善方向 |
| --- | --- | --- |
| 承接率（containment） | 无需人工即解决的比例 | ↑（IVR 常见 20–40%，语音 agent 可到 50–70%） |
| AHT | 人工平均处理时长 | ↓（转人工时带上下文，减少重述） |
| 转人工率 | 转接人工的比例 | ↓，但**不为 0**（设计参数） |
| 放弃率 | 排队中挂断 | ↓ |
| CSAT/NPS | 满意度 | ↑（**最重要的护栏指标**） |
| 每次接触成本 | 总成本 / 接触量 | ↓ |

**来电驱动分析**：把话量按意图分解，找出**Top N 意图**及其「可自动化档位」——这直接决定能承接多少。

### 2. ROI 模型（必须给区间与敏感性）

$$\text{月节省}=\underbrace{V\cdot\Delta c\cdot\text{AHT}\cdot p_{\text{agent-min}}}_{\text{被承接话量节省的人工分钟}}-\underbrace{C_{\text{platform}}}_{\text{平台成本}}$$

其中 $V$ = 月话量、$\Delta c$ = 承接率提升、$p$ = 每分钟人力成本。**关键细节**：
- 承接率提升不是全部节省——**部分被承接的话量本来也会被 IVR 解决**（要算**增量**）；
- **AHT 变化**：转人工带上下文能降低人工 AHT（这是常被忽略的第二收益源）；
- **平台成本**要含按分钟/按次计费 + 整合与运维（串 [[elevenlabs-02]] 的单位成本口径）。

**敏感性**：把 $\Delta c$ 上下浮动 10 个百分点，看 ROI 是否仍为正——**如果只差 10 个点就翻负，说明承诺过于激进**。

### 3. 分阶段放量与门禁

| 阶段 | 曝光 | 目标 | 门禁（示例） |
| --- | --- | --- | --- |
| P0 影子 | 0%（只处理录音） | 校准意图与话术 | 意图识别准确率 ≥ 95% |
| P1 下班 + 溢出 | 夜间/排队溢出 | 兜住无人时段 | 承接率 ≥ 基线、CSAT 不低于现状 |
| P2 低风险意图 | Top 2–3 意图全量 | 验证端到端 | 转人工顺畅率 ≥ 99%、错误率 ≤ 0.5% |
| P3 扩大意图 | Top 5–8 | 承接率目标达成 | CSAT 不下降、AHT 下降 |
| P4 全会话 | 100% | 常态运营 | 指标连续 4 周达标 |

**kill criteria（写进合同）**：例如「连续两周 CSAT 下降 > 5 个点，或错误率 > 1%，则回退到上一阶段」。

### 4. 必须保留的四件事（否则一定返工）

1. **按键菜单兜底**（随时可说「按 0 转人工」，且菜单仍可用）；
2. **带上下文的转人工**（意图、已核验身份、已收集字段）；
3. **身份核验**（涉及账户/账单时；三次失败转人工，串 [[elevenlabs-07]]）；
4. **披露与录音告知**（AI 身份 + 录音 + 用途）。

### 5. 失败模式与对策

| 失败模式 | 症状 | 对策 |
| --- | --- | --- |
| 意图覆盖过度承诺 | 上线后承接率远低于预期 | Top N 优先 + 三档预测 |
| 转人工体验差 | CSAT 掉、重复叙述 | 带上下文转接 + 回呼 |
| 整合脆弱 | 查单/改单失败率高 | 幂等 + 重试 + 明确的失败话术 |
| 噪声/口音导致识别差 | 反复确认、用户烦躁 | 确认策略、可切按键、必要时直接转人工 |
| 无门禁 | 问题拖着不修 | 阶段门禁 + kill criteria |
| 忽略合规 | 录音/披露/数据问题 | 事前合规评审 |

### 6. 交付时间线（示例 10 周）

| 周 | 里程碑 |
| --- | --- |
| 1–2 | 基线采集、话量分析、目标与门禁签约 |
| 3–4 | 意图设计、话术与降级路径、整合对接 |
| 5 | P0 影子运行与校准 |
| 6 | P1 下班/溢出放量 |
| 7–8 | P2/P3 扩大意图 |
| 9 | P4 全会话 + 复盘 |
| 10 | 交接、SOP、监控看板与季度复盘机制 |

## 数值与代码验证

### 表 1：三档 ROI（月话量 20 万；现有 IVR 承接率 32%）

| 场景 | 承接率 | 人工节省 | 月总成本 | 净节省 | ROI | 回本 |
|--- |--- |--- |--- |--- |--- |--- |
| 现状基线（现有 IVR） | **32.0%** | — | \$393,250（人工）/ 每次接触 \$1.966 | — | — | — |
| 保守（覆盖 Top3） | 46% | \$130,988 | \$33,868 | \$97,120 | 287% | 1.5 月 |
| 目标（Top5 + 长尾） | **54%** | \$183,612 | \$34,732 | \$148,880 | **429%** | **1.0 月** |
| 乐观（全意图优化） | 62% | \$236,236 | \$35,596 | \$200,640 | **564%** | **0.7 月** |
| agent 承接率分解（查余额 22%/85%、改地址 12%/80%、订单状态 16%/88%、退换货 11%/35%、投诉 9%/0%、长尾 30%/27%） | 理想 ≈ 54%，相对现有 IVR 的**增量**承接 ≈ **+22%** | — | — | — | — | — |

### 表 2：意图覆盖与承接贡献（示例）

| 意图 | 话量占比 | 自动化档位 | 承接贡献 |
| --- | --- | --- | --- |
| 查余额/账单 | 22% | 全自动（核验后） | 18% |
| 改地址/联系方式 | 12% | 全自动（核验后） | 10% |
| 订单状态 | 16% | 全自动 | 14% |
| 退换货 | 11% | 半自动（收集信息后转人工） | 4% |
| 投诉 | 9% | 不自动 | 0% |
| 其他长尾 | 30% | 部分 | 8% |
| **合计** | 100% | — | **约 54%** |

### 可运行代码

```python
# 联络中心 IVR 替换：基线、意图覆盖、ROI 三档与敏感性、分阶段门禁
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 现状基线 ----------
@dataclass
class Baseline:
    monthly_calls: int = 200_000        # 月话量
    ivr_containment: float = 0.32       # 现有 IVR 承接率
    transfer_rate: float = 0.55         # 转人工率
    aht_min: float = 6.5                # 人工 AHT（分钟）
    agent_cost_per_min: float = 0.55    # 人力成本（美元/分钟，含福利与分摊）
    csat: float = 0.78                  # 满意度
    def human_calls(self) -> float:
        return self.monthly_calls * self.transfer_rate
    def human_minutes(self) -> float:
        return self.human_calls() * self.aht_min
    def monthly_cost(self) -> float:
        return self.human_minutes() * self.agent_cost_per_min

base = Baseline()
print("① 现状基线（月话量 20 万）")
print(f"  现有承接率        {base.ivr_containment:>7.1%}")
print(f"  转人工话量        {base.human_calls():>7,.0f} 通/月")
print(f"  人工分钟          {base.human_minutes():>7,.0f} 分钟/月")
print(f"  人工成本          ${base.monthly_cost():>10,.0f}/月")
print(f"  每次接触成本      ${base.monthly_cost()/base.monthly_calls:>10.3f}")
print("  读法：先算清这三个数（承接率、人工分钟、每次接触成本）——它们就是后面所有结论的基线；")
print("        没有基线就承诺 ROI 是这类项目最常见的失败起点")

# ---------- 2) 意图覆盖表 ----------
@dataclass
class Intent:
    name: str
    share: float                 # 话量占比
    auto_rate: float             # 该意图可被 agent 独立解决的比例
    auth_required: bool = False
INTENTS = [
    Intent("查余额/账单", 0.22, 0.85, True),
    Intent("改地址/联系方式", 0.12, 0.80, True),
    Intent("订单状态", 0.16, 0.88, False),
    Intent("退换货（收集信息后转人工）", 0.11, 0.35, True),
    Intent("投诉", 0.09, 0.00, False),
    Intent("长尾杂项", 0.30, 0.27, False),
]
def agent_containment(intents: List[Intent], efficiency: float = 1.0) -> float:
    return sum(i.share * i.auto_rate for i in intents) * efficiency
print("\n② 意图覆盖与 agent 承接率")
print(f"  {'意图':<26} {'话量占比':>8} {'可自动':>7} {'承接贡献':>9} {'需核验':>7}")
for i in INTENTS:
    print(f"  {i.name:<26} {i.share:>8.0%} {i.auto_rate:>7.0%} "
          f"{i.share*i.auto_rate:>9.1%} {'是' if i.auth_required else '否':>7}")
ac = agent_containment(INTENTS)
print(f"  agent 承接率（理想）≈ {ac:.0%}；现有 IVR 为 {base.ivr_containment:.0%}")
print(f"  相对现有 IVR 的**增量**承接 ≈ {ac - base.ivr_containment:+.0%}")
print("  读法：**只有「增量」才产生节省** —— 已经能被 IVR 解决的话量不该重复计入 ROI；")
print("        这就是为什么三档预测里要把增量单独列出来（很多方案在这里虚高）")

# ---------- 3) ROI 三档 + 敏感性 ----------
@dataclass
class Platform:
    per_minute: float = 0.06          # 平台按分钟计费
    monthly_fixed: float = 8_000      # 运维与看板分摊
    one_time: float = 150_000         # 一次性整合与工程投入（按 12 个月摊销）
    contained_min: float = 1.6        # 被承接通话的 agent 分钟（含确认）
    transferred_min: float = 0.7      # 未承接但已进入 agent 的分钟（收集信息后转人工）
    def monthly_cost(self, contained_calls: float, transferred_calls: float) -> float:
        minutes = contained_calls * self.contained_min + transferred_calls * self.transferred_min
        return (minutes * self.per_minute + self.monthly_fixed
                + self.one_time / 12)     # ★ 一次性投入必须计入分母

def roi(containment: float, aht_reduction: float = 0.0,
        platform: Platform = Platform(), b: Baseline = base) -> Dict[str, float]:
    """containment 为 agent 的绝对承接率；只把「相对 IVR 的增量」算作节省。
    ★ 平台成本覆盖**所有进入 agent 的话量**（含最终转人工的），而不是只算被承接的。"""
    delta = max(0.0, containment - b.ivr_containment)
    deflected = b.monthly_calls * delta
    # 收益一：被承接话量不再占用人工
    saved_minutes = deflected * b.aht_min
    # 收益二：转人工的通话因带上下文而 AHT 下降
    remaining_human = b.monthly_calls * (1 - containment)
    saved_minutes += remaining_human * b.aht_min * aht_reduction
    saving = saved_minutes * b.agent_cost_per_min
    # 成本：进入 agent 的全部话量（承接 + 转人工）+ 固定 + 一次性摊销
    contained_calls = b.monthly_calls * containment
    transferred_calls = b.monthly_calls * (1 - containment)
    cost = platform.monthly_cost(contained_calls, transferred_calls)
    net = saving - cost
    return {"增量承接": delta, "被承接话量": deflected, "节省人工分钟": saved_minutes,
            "人工节省": saving, "月总成本": cost, "净节省": net,
            "ROI": net / cost if cost else 0.0,
            "回本(月)": platform.one_time / net if net > 0 else float("inf")}

print("\n③ ROI 三档（承接率假设不同）+ 人工 AHT 下降的附加收益")
print(f"  {'场景':<20} {'承接率':>7} {'人工节省':>11} {'月总成本':>10} "
      f"{'净节省':>11} {'ROI':>7} {'回本':>7}")
SCENARIOS = [("保守（覆盖 Top3）", 0.46), ("目标（Top5+长尾）", 0.54),
             ("乐观（全意图优化）", 0.62)]
results = {}
for label, c in SCENARIOS:
    r = roi(c, aht_reduction=0.08)         # 转人工带上下文，AHT 降 8%
    results[label] = r
    print(f"  {label:<20} {c:>7.0%} ${r['人工节省']:>10,.0f} ${r['月总成本']:>9,.0f} "
          f"${r['净节省']:>10,.0f} {r['ROI']:>6.0%} {r['回本(月)']:>6.1f}月")
print("  读法：**一次性投入必须计入分母**（否则 ROI 会被严重高估——第一版就犯了这个错）；")
print("        平台成本覆盖所有进入 agent 的话量（含转人工的），且 ROI 给的是「净节省/总成本」；")
print("        三档都必须为正、且回本周期可接受（例如 ≤ 6 个月）才签承诺")

print("\n④ 敏感性：承接率差 10 个百分点，ROI 会怎样？")
print(f"  {'承接率':>7} {'增量':>7} {'净节省':>11} {'ROI':>7} {'回本':>7} {'结论':<10}")
for c in (0.36, 0.40, 0.44, 0.48, 0.52, 0.56):
    r = roi(c, aht_reduction=0.08)
    verdict = "亏" if r["净节省"] < 0 else ("回本过慢" if r["回本(月)"] > 12 else "可接受")
    print(f"  {c:>7.0%} {r['增量承接']:>7.0%} ${r['净节省']:>10,.0f} {r['ROI']:>6.0%} "
          f"{r['回本(月)']:>6.1f}月 {verdict:<10}")
breakeven = None
for c in [x / 100 for x in range(30, 70)]:
    if roi(c, aht_reduction=0.08)["净节省"] > 0:
        breakeven = c
        break
print(f"  盈亏平衡承接率 ≈ {breakeven:.0%}（低于此值项目亏钱；现状 IVR 基线是 {base.ivr_containment:.0%}）")
print("  读法：**给出盈亏平衡点**比给一个 ROI 数字专业得多 —— 客户可以用它判断风险；")
print("        注意本例的盈亏平衡点（30%）竟**低于现状 IVR 基线（32%）**，这说明参数偏乐观：")
print("        人力成本、承接率、影子阶段的估计都可能虚高 —— 所以要主动打折再复算")

print("\n④b 主动打折：影子运行的承接率上界要下调多少才仍然成立")
def discounted(ideal: float, cut: float) -> float:
    return max(0.0, ideal - cut)
print(f"  {'理想承接率':>10} {'下调 10 点':>11} {'下调 15 点':>11} {'下调后是否仍优于基线':>20}")
for ideal in (0.46, 0.54, 0.62):
    d10, d15 = discounted(ideal, 0.10), discounted(ideal, 0.15)
    print(f"  {ideal:>10.0%} {d10:>11.0%} {d15:>11.0%} "
          f"{('是' if d15 > base.ivr_containment else '否'):>20}")
print("  读法：**影子运行给出的是上界**（人工标注「本可承接」比真实自动完成更乐观），")
print("        行业经验是实跑比影子低 10–15 个百分点 —— 承诺前必须按打折后的数字复算一遍；")
print("        本例：目标档 54% 下调 15 点后仍为 39%（高于基线 32%）——项目成立；")
print("        但**保守档 46% 下调 15 点只有 31%，低于基线** ——所以保守档不能只靠承接率，")
print("        必须把「转人工 AHT 下降」与「下班/溢出时段兜底」一起计入，否则不该签这个档位")

print("\n⑤ 分阶段放量门禁（每阶段可回退）")
@dataclass
class Phase:
    name: str
    exposure: float
    target_containment: float
    gates: List[str]
PHASES = [
    Phase("P0 影子（不接真实流量）", 0.0, 0.0, ["意图识别准确率 ≥ 95%"]),
    Phase("P1 下班 + 排队溢出", 0.15, 0.45, ["承接率 ≥ IVR 基线", "CSAT 不低于现状"]),
    Phase("P2 Top2-3 意图全量", 0.35, 0.50, ["转人工顺畅率 ≥ 99%", "错误率 ≤ 0.5%"]),
    Phase("P3 Top5-8 意图", 0.70, 0.54, ["CSAT 不下降", "人工 AHT 下降 ≥ 5%"]),
    Phase("P4 全会话", 1.00, 0.54, ["连续 4 周指标达标"]),
]
print(f"  {'阶段':<26} {'曝光':>7} {'目标承接':>9} 门禁")
for p in PHASES:
    print(f"  {p.name:<26} {p.exposure:>7.0%} {p.target_containment:>9.0%} "
          f"{'；'.join(p.gates)}")
print("  kill criteria：连续两周 CSAT 下降 > 5 个点，或错误率 > 1% -> 回退到上一阶段")
print("  读法：门禁要**可测、可回退、写进合同**；否则问题会一直拖到全量上线才暴露")

print("\n⑥ 交付时间线（10 周示例）")
TIMELINE = [("第 1-2 周", "基线采集与话量分析、目标与门禁签约"),
            ("第 3-4 周", "意图设计、话术与降级路径、系统整合"),
            ("第 5 周", "P0 影子运行与校准"),
            ("第 6 周", "P1 下班/溢出放量"),
            ("第 7-8 周", "P2/P3 扩大意图"),
            ("第 9 周", "P4 全会话 + 复盘"),
            ("第 10 周", "交接、SOP、看板与季度复盘机制")]
for w, what in TIMELINE:
    print(f"  {w:<10} {what}")
print("  读法：**第 1-2 周就必须把门禁与目标签下来** —— 这是交付可控的前提；")
print("        技术工作在 3-4 周，真正的风险在「意图覆盖是否真实」和「转人工是否顺畅」")
```

预期输出要点（实跑）：① 基线给出月话量 20 万下的**转人工 11 万通、人工 71.5 万分钟、成本约 \$39 万/月、每次接触 \$1.97**——这就是所有结论的基线；② 意图覆盖表算出 agent 承接率约 **54%**，但**只有相对现有 IVR（32%）的增量才产生节省**（22 个百分点）；③ ROI 三档在「承接率 46%/54%/62% + 转人工 AHT 降 8%」下给出净节省与 ROI，且**平台成本随承接率一起上升**（边际收益递减）；④ 敏感性给出**盈亏平衡承接率（约 30%）**——比单点 ROI 数字更有决策价值；但要注意它**低于现状 IVR 基线（32%）**，说明参数偏乐观，因此必须把影子运行的承接率上界**下调 10–15 个百分点**再复算（目标档 54% 下调 15 点后为 39%，仍高于基线；但**保守档 46% 下调后只有 31%，低于基线**，所以保守档必须把 AHT 下降与溢出兜底一起计入才成立），真正的风险落在 CSAT 与转人工体验上；⑤ 分阶段门禁把「影子→溢出→扩大→全量」写成可测、可回退的清单，并给出 kill criteria；⑥ 10 周时间线强调**前两周就要签下门禁与目标**。

## 常见追问

- **追问**：如果客户坚持「要替代 X 名坐席」怎么办？
  - 要点：把目标改写成**指标语言**（承接率、AHT、转人工率、CSAT），并解释人力节省取决于**话量分布与时段**而不是「替换几个人」；给出三档预测与盈亏平衡点，让客户自己判断。**接受「先做溢出与下班时段」这种更小的承诺**，比接受一个不可验证的大承诺更专业。
- **追问**：怎么处理「用户就是要找人工」？
  - 要点：**尊重意图**——明确说「转人工」就立即转（最多一次确认），并且**不要让用户重复叙述**；同时用回呼（callback）缓解排队。**强行挽留是 CSAT 的头号杀手。**
- **追问**：口音/噪声导致的识别问题怎么兜？
  - 要点：确认策略（复述关键字段）、可切换到按键、以及**在识别置信度低时主动转人工**；对关键业务（金额、账号）必须做二次确认。
- **追问**：怎么做影子运行（P0）？
  - 要点：用**历史录音回放**让 agent 走完整流程（不实际执行写操作），人工标注「是否正确理解 + 是否本可承接」；这能在零风险下校准意图与话术，并给出承接率的**上界估计**（实战会更低，要打折扣）。
- **追问**：整合失败（CRM 超时）时怎么办？
  - 要点：**幂等 + 重试 + 明确话术 + 转人工**四件套；对写操作要有「未确认即不生效」的语义，避免「用户以为改了但系统没改」（串 [[together-01]] 的错误语义）。
- **追问**：如何证明 CSAT 没被牺牲？
  - 要点：**分渠道分组对比**（agent 处理 vs 人工处理），并做**同一时段前后对比**；同时看投诉率、二次来电率（同一问题 48 小时内再次来电）——后者常比 CSAT 更敏感。
- **追问**：合规要评审什么？
  - 要点：录音与转写的告知与留存、AI 身份披露、数据最小化与访问审计、跨境传输、以及行业特定要求（金融/医疗）；涉及账户操作时的**强身份核验**要求。

## 相关题目

- [[elevenlabs-07]]：医院预约系统——同一类落地场景，安全边界与核验更严格。
- [[elevenlabs-04]]：实时语音 agent 的延迟预算，决定「听起来像不像在对话」。
- [[elevenlabs-01]]：流式代理与取消，是 barge-in 与「随时转人工」的实现基础。
- [[evaluation-04]]：回归门禁与红线集，本题的分阶段门禁沿用同一套思路。
- [[anthropic-32]]：客户现场诊断与沟通，与本题的交付节奏呼应。

## 参考资料与归属

- **设计实时语音 AI Agent** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent>。第 4 节语音 agent 链路与工程约束参照这篇。
- **Handling Overload（Google SRE Book 第 21 章）（延伸）** —— Google SRE：<https://sre.google/sre-book/handling-overload/>。第 3、5 节「按可量化门禁分阶段放量、并预设回退条件」的纪律参照这一章。
- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。第 1 节「先建指标与基线、再谈模型」的纪律来自这份清单。
- **为实时语音 agent 做 latency 预算（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节端到端延迟与流水线化的口径来自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（月话量 20 万、IVR 承接率 32%、转人工率 55%、AHT 6.5 分钟、人力 \$0.55/分钟、平台 \$0.06/分钟 + \$8,000 固定、agent 每通 1.6 分钟、意图占比与自动化率、三档承接率 46%/54%/62%、转人工 AHT 降 8%、各阶段曝光与门禁）都是为演示方法而构造的**示例参数与显式假设**；真实项目必须用该联络中心的实际话务与成本数据校准，并遵守当地录音与数据法规。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
