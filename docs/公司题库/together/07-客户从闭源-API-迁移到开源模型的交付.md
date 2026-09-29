---
type: question
id: together-07
company: Together AI
topic: applied
order: 7
question: 客户想从闭源前沿模型的 API 迁移到开源模型。你会如何推进这次合作？
question_en: A customer wants to migrate from a closed frontier model API to open models. How would you run this engagement?
asked_at: []
level: 高阶
tags: [迁移, 客户交付, 评估基线, 非劣性检验, 混合路由, 成本平价]
sources:
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
  - title: LiveBench: A Challenging, Contamination-Limited LLM Benchmark（延伸）
    url: https://arxiv.org/abs/2406.19314
    author: White et al. (ICLR 2025)
    published: 2024-06-27
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
  - title: Prompting、RAG 还是 fine-tuning：决策框架与成本权衡（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [together-04, together-06, finetuning-08, evaluation-04, anthropic-32]
updated: 2026-09-28
---

## 一句话答案

> 这类迁移项目的成败**不在模型选型，而在「评估口径 + 迁移顺序 + 兜底设计」**。六步推进：
> ① **先问「为什么迁移」**：成本、数据驻留/合规、延迟与可控性、可定制（微调）、供应商风险——不同动机对应完全不同的方案（例如为合规就要自托管，为成本可能混合路由就够）；
> ② **建客户自己的评估集**（不是公开榜单）：从**真实流量抽样** + 人工标注任务级判据（正确性/格式/引用/拒答边界），样本量按**非劣性检验**算（要证明「不比闭源差超过 $\delta$」，而不是「跑分接近」）；
> ③ **候选与部署形态一起定**：开源模型（能力/许可/上下文/多语言/工具调用）+ 部署形态（专用 endpoint / serverless / 自托管）——**这两件事互相约束**（自托管才有微调与数据驻留的完全控制，串 [[together-06]]）；
> ④ **补能力缺口而不是硬扛**：prompt 与结构化输出约束、检索增强、少样本、必要时 LoRA 微调；**难请求回退到闭源**（混合路由）——这是最务实的过渡形态；
> ⑤ **算平价（parity）**：质量差距 $\Delta q$ 与成本节省 $\Delta c$ 必须同时给出；**混合路由的最优点**可以算出来（把「回退闭源」的比例当作变量优化）；
> ⑥ **灰度迁移 + 兜底**：影子流量对比 → 小比例放量 → 逐步扩大；**始终保留一键回退**，并对质量回归设红线（串 [[anthropic-31]] 的分级发布）。
> 一句话判据：**用客户的流量、客户的判据、客户能接受的非劣性边界来验收**——而不是用「我们的模型在某某榜单上更高」。

## 面试官在考什么

- **是否从动机出发**：能不能把「迁移」拆成不同的目标组合，并指出**只有部分目标需要换模型**（例如纯成本目标可能只需混合路由 + 缓存 + 批处理）。
- **评估是否客户化**：能否坚持用**真实流量样本 + 任务级判据**，而不是公开榜单；以及能否说清榜单的风险（污染、与任务不匹配、指标敏感性，串 [[evaluation-06]]）。
- **统计口径**：能否用**非劣性检验**（而不是「差异不显著」）来验收；样本量怎么算；以及为什么要**配对比较**（同一个请求分别跑两边）。
- **成本模型是否完整**：不仅比较 token 单价，还要算**总拥有成本**（GPU 利用率、运维、工程投入、迁移期双跑成本、以及为达质量所需的后处理/微调成本）。
- **回退与混合路由**：能否设计「难请求走闭源」的分流规则（按任务类型、置信度、长度、失败重试），并**量化分流比例的收益**。
- **工程与合规细节**：许可（模型权重许可是否允许商用/再分发）、数据驻留、日志与审计、以及「从闭源迁移后谁来负责质量」。
- **诚实与长期**：会说「不建议迁移」的场景（用量太低、质量要求极高、团队无运维能力）；以及**给出可执行的下一步**（试点范围、时间线、验收判据）。

**常见错误答案**

- 直接推荐某个开源模型（跳过评估与动机）。
- 用公开榜单证明「开源已追平」。
- 只比 token 单价，不算工程与运维成本。
- 只要「平均质量」达标就宣布成功——不看**分群与长尾**（某些任务类型掉得厉害）。
- 没有回退方案，一次性切换。
- 忽略许可与合规（权重许可、数据处理协议）。

## 原理与推导

### 1. 六个阶段与交付物

| 阶段 | 关键动作 | 交付物 | 常见陷阱 |
| --- | --- | --- | --- |
| ① 目标澄清 | 动机排序（成本/合规/延迟/可定制/风险） | 一页目标与约束 | 目标含糊导致后期反复 |
| ② 评估基线 | 真实流量抽样 + 标注判据 + 样本量 | 客户专属评估集 + 验收判据 | 用榜单代替评估集 |
| ③ 选型与形态 | 候选模型 + 部署形态（含许可审查） | 候选清单 + 形态建议 | 忽略许可与上下文长度 |
| ④ 补缺口 | prompt/结构约束/检索/微调/回退 | 优化后的基线 + 差距清单 | 一上来就微调 |
| ⑤ 平价分析 | 质量 $\Delta q$ vs 成本 $\Delta c$ + 混合路由 | 决策表（含分流比例） | 只算 token 单价 |
| ⑥ 灰度迁移 | 影子→小比例→放量 + 回退 | 上线计划 + 监控看板 | 一次性切换 |

### 2. 验收：非劣性检验（本题的统计核心）

目标不是「证明两者一样」，而是「**证明开源不比闭源差超过 $\delta$**」（$\delta$ 为业务可接受的质量损失，例如 3 个百分点）。配对设计（同一请求跑两边）下的样本量近似：

$$n\approx\frac{(z_{1-\alpha}+z_{1-\beta})^2\,\sigma_d^2}{\delta^2}\quad(\text{配对口径})$$

其中 $\sigma_d$ 是**配对差值的标准差**（通常远小于各自的方差，这正是配对的价值）。粗口径（按两比例）$n\approx\frac{2(z_{1-\alpha}+z_{1-\beta})^2 p(1-p)}{\delta^2}$。

**纪律**：
- **预注册** $\delta$、$\alpha$、功效与判据（任务级通过标准）；
- **分群报告**（任务类型、语言、长度）——平均达标而某类崩掉是迁移事故的常见形态；
- **不要用「差异不显著」来宣称等价**（功效不足时「不显著」毫无信息量，串 [[anthropic-27]]）。

### 3. 成本平价与混合路由

- **闭源成本**：$\text{tok}\times p_{\text{closed}}$（按量，无固定成本）；
- **开源成本**：$\frac{\text{GPU}\cdot\text{h}\times p_{\text{gpu}}}{\text{吞吐}\times\text{利用率}}$ + 运维 + 工程投入摊销（串 [[together-04]]）；
- **混合路由**：把请求按「难度/风险」分流，闭源只处理 $r$ 比例：

$$\text{cost}(r)=r\cdot\text{cost}_{\text{closed}}+(1-r)\cdot\text{cost}_{\text{open}},\qquad \text{quality}(r)=r\cdot q_{\text{closed}}+(1-r)\cdot q_{\text{open}}$$

**最优点**：在质量约束 $q(r)\ge q_{\min}$ 下最小化成本——即**用闭源兜住最难的那部分请求**。工程上分流规则可以是：任务类型白名单、输出长度阈值、结构化校验失败重试、或**置信度阈值**（例如自一致性差就回退）。

### 4. 迁移顺序（按风险从低到高）

1. **旁路任务**（非用户可见：分类、抽取、摘要草稿）；
2. **内部工具**（员工用，容忍度较高）；
3. **低风险用户可见**（格式化输出、补全建议）；
4. **高风险/核心链路**（保留闭源或有严格门禁）。
**每上一级都要有回退开关与指标红线**。

### 5. 总拥有成本（容易漏算的项）

| 项 | 说明 |
| --- | --- |
| 迁移期双跑 | 两套系统并行，成本翻倍（通常 1–3 个月） |
| 工程投入 | prompt/评估/微调/路由开发的人力 |
| 运维 | 自托管要值班、扩容、故障处理；serverless 则体现在单价里 |
| 质量补偿 | 检索、后处理、重试带来的额外成本 |
| 合规与审计 | 数据驻留、日志留存、许可审查 |

### 6. 何时应当建议「不要迁移」

- 用量低于盈亏平衡（专用 endpoint 不划算，串 [[together-04]]）；
- 质量要求极高且差距无法用工程弥补（例如高风险专业领域）；
- 团队缺少运维 GPU 的能力且不接受托管形态；
- 许可不允许所需用途。
**给出「不迁移」的建议并给出替代方案（混合路由、缓存优化、批处理）**，比硬推迁移更能建立长期信任。

## 数值与代码验证

### 表 1：非劣性验收所需的样本量（粗口径，$\alpha=0.05$、功效 80%）

| 可接受损失 $\delta$ | $p\approx0.5$ | $p\approx0.8$ | $p\approx0.95$ |
| --- | --- | --- | --- |
| 5 个百分点 | 约 1,570 | 约 1,000 | 约 600 |
| 3 个百分点 | 约 4,360 | 约 2,800 | 约 1,700 |
| 1 个百分点 | 约 39,200 | 约 25,000 | 约 15,000 |

**读法**：验收口径必须**事先谈定**——$\delta=1$ 个百分点需要数万样本，工程上不可行；实践中常用 $\delta=3$–5 个百分点 + **分群检查**。

### 表 2：混合路由的成本-质量权衡（实测，质量下限 0.88）

| 闭源比例 $r$ | 成本（\$/1M） | 质量（通过率） | 相对全闭源成本 |
| --- | --- | --- | --- |
| 100% | 15.00 | 0.900 | 100.0% |
| 50% | 7.80 | 0.870 | 52.0% |
| 30% | 4.92 | 0.858 | 32.8% |
| 10% | 2.04 | 0.846 | 13.6% |
| 0% | **0.60** | 0.840 | **4.0%** |

按质量下限求解最优闭源比例：**下限 0.90 → 100%（不迁移）**；0.88 → **67%**；0.86 → **34%**。

**读法（本题最重要的两条）**：
1. **质量下限直接决定迁移是否成立**：当下限高于开源可达到的水平时，最优解就是「全用闭源」——所以第一步必须谈清 $\delta$；
2. **抬高开源质量的回报极高**：开源质量从 0.84 抬到 0.88（+4 个百分点，正好达标）时，最优闭源比例从 **67% 直接降到 0**，成本从 \$10.25 降到 \$0.60/1M（**降幅 94.1%**）。**这比优化路由比例值钱得多**——迁移项目里最该投的是评估集 + prompt/检索/微调，而不是分流规则。

### 可运行代码

```python
# 迁移交付的三个量化工具：非劣性样本量、混合路由最优点、总拥有成本对比
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 非劣性验收的样本量（配对与两比例两种口径） ----------
def z(p: float) -> float:
    """标准正态分位数（Acklam 近似）"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def n_two_prop(delta: float, p: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> int:
    return math.ceil(2 * (z(1 - alpha / 2) + z(power)) ** 2 * p * (1 - p) / delta ** 2)

def n_paired(delta: float, sd_diff: float = 0.3, alpha: float = 0.05, power: float = 0.8) -> int:
    """配对设计：用配对差值的标准差（通常远小于各自方差）"""
    return math.ceil((z(1 - alpha / 2) + z(power)) ** 2 * sd_diff ** 2 / delta ** 2)

print("① 非劣性验收所需样本量（α=0.05、功效 80%）")
print(f"  {'可接受损失 δ':>12} {'两比例 p=0.5':>14} {'两比例 p=0.8':>14} {'配对(sd=0.3)':>14}")
for delta in (0.05, 0.03, 0.01):
    print(f"  {delta:>12.0%} {n_two_prop(delta):>14,d} {n_two_prop(delta, 0.8):>14,d} "
          f"{n_paired(delta):>14,d}")
print("  读法：δ 从 5 个百分点收紧到 1 个百分点，样本量涨 25 倍；")
print("        配对设计（同一请求跑两边）能大幅省样本 —— 所以验收要做配对比较，而不是两组各抽一批")

# ---------- 2) 混合路由：质量约束下的成本最优点 ----------
@dataclass
class Route:
    """闭源 vs 开源的成本与质量（按每百万 token 归一）"""
    closed_cost: float = 15.0        # 闭源 API 输出价（$/1M token）
    open_cost: float = 0.60          # 开源部署（含利用率与运维）的等效成本
    closed_quality: float = 0.90     # 任务级通过率
    open_quality: float = 0.84
    def mix(self, r: float) -> Tuple[float, float]:
        cost = r * self.closed_cost + (1 - r) * self.open_cost
        quality = r * self.closed_quality + (1 - r) * self.open_quality
        return cost, quality
    def best_for_quality(self, q_min: float) -> Tuple[float, float, float]:
        """在质量不低于 q_min 的前提下最小化成本；返回 (最优 r, 成本, 质量)"""
        best = (1.0,) + self.mix(1.0)
        for i in range(0, 101):
            r = i / 100
            cost, quality = self.mix(r)
            if quality >= q_min and cost < best[1]:
                best = (r, cost, quality)
        return best

route = Route()
print("\\n② 混合路由：把「最难的那部分」留给闭源")
print(f"  {'闭源比例 r':>10} {'成本($/1M)':>11} {'质量(通过率)':>12} {'相对全闭源成本':>14}")
for r in (1.0, 0.5, 0.3, 0.1, 0.0):
    cost, quality = route.mix(r)
    print(f"  {r:>10.0%} {cost:>11.2f} {quality:>12.3f} {cost/route.closed_cost:>13.1%}")
for q_min in (0.90, 0.88, 0.86):
    r, cost, q = route.best_for_quality(q_min)
    print(f"  质量下限 {q_min:.2f} -> 最优闭源比例 {r:>4.0%}，成本 ${cost:>5.2f}/1M（"
          f"相对全闭源 {cost/route.closed_cost:.1%}）")
print("  读法：质量下限越松，需要回退闭源的比例越低、成本越低 ——")
print("        工程上要做的第一件事是**用 prompt/检索/微调把开源的质量抬起来**（抬高 open_quality），")
print("        而不是先去优化路由比例")

# ---------- 3) 抬高开源质量 vs 增加闭源比例的收益对比 ----------
print("\\n③ 两条路的收益对比（质量下限 0.88）")
base = Route()
r0, c0, q0 = base.best_for_quality(0.88)
print(f"  基线（open_quality=0.84）：最优闭源比例 {r0:.0%}，成本 ${c0:.2f}/1M")
for bump, label in ((0.86, "工程优化 +0.02"), (0.88, "工程优化 +0.04（达标）")):
    improved = Route(open_quality=bump)
    r1, c1, q1 = improved.best_for_quality(0.88)
    print(f"  {label:<22} open_quality={bump:.2f}: 最优闭源比例 {r1:>4.0%}，成本 ${c1:.2f}/1M "
          f"（相对基线 {(c0-c1)/c0:>5.1%} 下降）")
print("  读法：把开源质量抬 4 个百分点（正好达标）就能让最优闭源比例从 67% 降到 0 ——")
print("        **这是迁移项目里投资回报最高的动作**，远胜过优化路由比例")

# ---------- 4) 总拥有成本（把容易漏算的项加上） ----------
@dataclass
class TCO:
    name: str
    monthly_gpu: float = 0.0
    monthly_api: float = 0.0
    migration_dual_run_months: float = 2.0
    engineering_person_months: float = 6.0
    cost_per_person_month: float = 15_000
    ops_monthly: float = 4_000
    months: int = 12
    def total(self) -> Dict[str, float]:
        dual = (self.monthly_gpu + self.monthly_api) * self.migration_dual_run_months
        eng = self.engineering_person_months * self.cost_per_person_month
        run = (self.monthly_gpu + self.monthly_api) * self.months + self.ops_monthly * self.months
        return {"迁移期双跑": dual, "工程投入": eng, "运行": run,
                "合计（首年）": dual + eng + run}
print("\\n④ 总拥有成本对比（首年，含迁移期双跑与工程投入）")
SCENARIOS = [
    TCO("全闭源 API（不迁移）", monthly_api=180_000, migration_dual_run_months=0,
        engineering_person_months=0, ops_monthly=0),
    TCO("迁移到开源（专用 endpoint）", monthly_gpu=90_000, migration_dual_run_months=2,
        engineering_person_months=6, ops_monthly=4_000),
    TCO("混合路由（30% 回退闭源）", monthly_gpu=63_000, monthly_api=54_000,
        migration_dual_run_months=2, engineering_person_months=4, ops_monthly=4_000),
]
for sc in SCENARIOS:
    parts = sc.total()
    print(f"  {sc.name:<26} " + "  ".join(f"{k} ${v:>9,.0f}" for k, v in parts.items()))
print("  读法：闭源方案的成本全在运行费；迁移方案首年要额外承担双跑与工程投入 ——")
print("        所以**按首年还是按第三年比较，结论可能完全不同**，必须与客户明确口径")

# ---------- 5) 分群验收：平均值达标但某类崩掉（迁移事故的典型形态） ----------
@dataclass
class Segment:
    name: str
    closed_pass: float
    open_pass: float
    share: float
SEGS = [Segment("结构化抽取", 0.95, 0.93, 0.4),
        Segment("长文档摘要", 0.90, 0.86, 0.3),
        Segment("多轮改稿", 0.88, 0.72, 0.2),
        Segment("多语言", 0.85, 0.60, 0.1)]
avg_closed = sum(s.closed_pass * s.share for s in SEGS)
avg_open = sum(s.open_pass * s.share for s in SEGS)
print("\\n⑤ 分群验收（平均达标 ≠ 可以迁移）")
print(f"  {'分群':<12} {'占比':>6} {'闭源':>7} {'开源':>7} {'差距':>7}")
for s in SEGS:
    flag = "  ← 崩了" if s.closed_pass - s.open_pass > 0.10 else ""
    print(f"  {s.name:<12} {s.share:>6.0%} {s.closed_pass:>7.2f} {s.open_pass:>7.2f} "
          f"{s.open_pass - s.closed_pass:>+7.2f}{flag}")
print(f"  加权平均：闭源 {avg_closed:.3f} vs 开源 {avg_open:.3f}（差 {avg_open-avg_closed:+.3f}）")
print("  读法：加权平均已差 7.7 个百分点，而「多轮改稿」与「多语言」分别掉 16 与 25 个百分点 ——")
print("        如果只看平均值就上线，这两类用户会立刻投诉。**验收必须分群**，并对崩掉的分群单独定策略")
```

预期输出要点（实跑）：① 非劣性样本量表显示 $\delta$ 从 5 收紧到 1 个百分点时样本量涨约 25 倍，而**配对设计显著省样本**——所以验收要做同一请求的双边配对比较；② 混合路由表显示**质量下限越松、需回退闭源的比例越低、成本越低**；③ 收益对比给出本题最重要的结论：开源质量从 0.84 抬到 0.88（+4 个百分点，正好达标）时，**最优闭源比例从 67% 降到 0，成本降 94.1%**——**把开源质量抬起来比优化路由比例值钱得多**（+2 个百分点时降到 50%、成本降 23.9%）；④ 总拥有成本显示闭源方案成本全在运行费、迁移方案首年要额外承担双跑与工程投入——**按首年还是第三年比较，结论可能完全不同**；⑤ 分群验收显示**加权平均已差 7.7 个百分点（0.911 → 0.834），而「多轮改稿」与「多语言」两群分别掉 16 与 25 个百分点**——只看平均值就上线，这两类用户会立刻投诉。

## 常见追问

- **追问**：客户只想省钱，不想改架构，怎么办？
  - 要点：先做**不需要换模型的省钱手段**——前缀缓存（省 prefill）、批处理（提高利用率）、输出长度约束、路由到更小的同族模型、以及混合路由。这些往往能拿到 30–60% 的成本下降，**且风险远低于整体迁移**（串 [[together-04]]）。
- **追问**：评估集怎么建才可信？
  - 要点：从**真实流量分层抽样**（按任务类型/长度/语言），人工标注任务级判据（不只是「像不像」）；预注册 $\delta$ 与判据；**留出集**用于最终验收，避免对着它调 prompt（串 [[evaluation-04]]）；定期更新以防过拟合（串 [[evaluation-06]]）。
- **追问**：什么时候该微调而不是继续 prompt？
  - 要点：当差距集中在**风格/格式/领域术语**且检索与 prompt 已优化到位、且有足够标注数据时才微调；微调会带来版本管理与回滚成本（串 [[finetuning-08]] 的决策框架）。**先用 prompt 与检索把差距压到 3 个百分点以内，再决定是否微调。**
- **追问**：混合路由的分流规则怎么定？
  - 要点：从可观测信号出发——任务类型白名单、输出长度阈值、结构化校验失败、自一致性差（多次采样不一致）、或用户显式选择；并把「回退率」作为监控指标（回退率上升说明开源质量在漂移）。
- **追问**：迁移后发现质量下降怎么办？
  - 要点：**一键回退**（路由切回闭源）+ 按分群定位（哪一类掉得多）+ 用评估集复现；同时把失败样本固化进评估集（串 [[anthropic-31]] 的事故闭环）。**回退开关必须在迁移前就做好并演练过。**
- **追问**：许可与合规要检查什么？
  - 要点：权重许可（是否允许商用、是否要求署名、是否有再分发限制）、训练数据来源的合规声明、数据驻留（自托管 vs 第三方）、日志与审计要求、以及「客户数据是否会被用于训练」的条款。**这些通常比技术问题更容易卡住项目**。

## 相关题目

- [[together-04]]：专用 endpoint 的成本与定价，是迁移经济性分析的基础。
- [[together-06]]：serverless 平台与部署形态选择，决定迁移的落地方式。
- [[finetuning-08]]：prompt / RAG / 微调的决策框架，对应缺口弥补的路线选择。
- [[evaluation-04]]：回归门禁与红线集，迁移验收与放量门禁沿用同一套。
- [[anthropic-32]]：客户现场的诊断与沟通方法，与本题的交付节奏呼应。

## 参考资料与归属

- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。第 1、3 节「先建指标与基线、从简单方案起步、用真实数据验收」的纪律来自这份清单。
- **LiveBench: A Challenging, Contamination-Limited LLM Benchmark（延伸）** —— White et al. (ICLR 2025)，2024-06-27：<https://arxiv.org/abs/2406.19314>。第 2 节「公开榜单的污染风险与持续更新」的论据来自这篇。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 3 节「吞吐与成本随部署形态与批处理变化」的框架来自这篇。
- **Prompting、RAG 还是 fine-tuning：决策框架与成本权衡（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节缺口弥补路线（prompt → 检索 → 微调）的决策顺序取自本仓库同主题专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$\delta$ 取 1/3/5 个百分点、$p$ 取 0.5/0.8、配对 sd 0.3、闭源 \$15/1M 与开源 \$0.60/1M、质量 0.90 与 0.84、月成本 18 万/9 万、工程 6 人月、\$15k/人月、运营 \$4k/月、双跑 2 个月、分群通过率）都是按本仓库统一口径构造的**工程算例与显式假设**；真实项目必须用客户自己的流量、成本与判据校准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
