---
type: question
id: deepseek-01
company: DeepSeek
topic: llm-internals
order: 1
question: 讲讲 DeepSeekMoE。它与 Mixtral 这类标准的 top-2 MoE 有何不同？
question_en: Tell me about DeepSeekMoE. How does it differ from a standard top-2 MoE such as Mixtral?
asked_at: []
level: 高阶
tags: [DeepSeekMoE, 细粒度专家, 共享专家, 路由组合, 负载均衡]
sources:
  - title: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（延伸）
    url: https://arxiv.org/abs/2401.06066
    author: Dai et al. (DeepSeek-AI)
    published: 2024-01-11
  - title: Mixtral of Experts（延伸）
    url: https://arxiv.org/abs/2401.04088
    author: Jiang et al. (Mistral AI)
    published: 2024-01-08
  - title: DeepSeek-V3 Technical Report（延伸）
    url: https://arxiv.org/abs/2412.19437
    author: DeepSeek-AI
    published: 2024-12-27
  - title: 稀疏 MoE 为何适合私有化部署（本仓库公司题库 · Cohere 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [deepseek-02, deepseek-04, deepseek-05, cohere-02, inference-serving-07]
updated: 2026-09-28
---

## 一句话答案

> **三个改动，一句话概括：把专家切得更细、多选几个、再留一个「永远在岗」的共享专家。**
> | 维度 | Mixtral（标准 top-2） | DeepSeekMoE |
> | --- | --- | --- |
> | 专家粒度 | 粗（8 个专家，每个约一个 FFN 大小） | **细**（专家更多、每个更小，如 256 个路由专家） |
> | 激活专家数 | top-2 | **top-k，k 更大**（V3 为 top-8） |
> | 共享专家 | 无 | **有**（1 个始终激活，吸收「通用知识」） |
> | 路由组合数 | $\binom{8}{2}=28$ | $\binom{256}{8}\approx4.1\times10^{14}$ |
> **为什么要细粒度**：MoE 的容量来自「**组合数**」而不是「专家数」。
> $$\text{组合数}=\binom{N}{k}\ \Rightarrow\ \text{细粒度把组合空间从 }28\text{ 炸到 }10^{14}$$
> 组合越多，**不同 token 越能被分到「更贴合的子空间」**——这是「专家专业化」的数学来源。**注意：细粒度并不自动带来更好的负载均衡**（本机实测反而更差，见下文），它只是把「容量」做大。
> **为什么要共享专家**：语言里有一大批**通用模式**（常见语法、高频搭配、基础推理），它们**对几乎所有 token 都有用**。若没有共享专家，这些通用知识会被**在每个路由专家里重复学习**——等于**参数冗余**。共享专家把它们集中起来，让路由专家专注于**差异化**部分：
> $$h_{\text{out}}=\underbrace{E_{\text{shared}}(h)}_{\text{通用，始终激活}}+\sum_{i\in\text{top-}k}\underbrace{g_i\,E_i(h)}_{\text{差异化，按需激活}}$$
> **代价（必须一起讲）**：
> ① **路由与通信开销上升**（专家多 → all-to-all 的目标更多、调度更碎）；
> ② **每个专家变小 → 单卡能放的专家数受限、访存更碎**（HBM 带宽利用率下降）；
> ③ **负载均衡压力更大**（组合多、分布长尾）；
> ④ **实现复杂度上升**（top-k 的 k 更大、共享专家要单独处理，见 [[deepseek-04]] 的实现陷阱）。
> 一句话判据：**Mixtral 是「少而大、选得少」，DeepSeekMoE 是「多而小、选得多、外加一个常驻」**——前者实现简单，后者在同等激活参数下容量更大，但工程难度更高。

## 面试官在考什么

- **能否说出「细粒度」的目的**：不是「专家多就更强」，而是**组合空间爆炸 → 更细的专业化**；能否给出组合数的量化（$\binom{N}{k}$）。
- **能否解释共享专家的动机**：**避免通用知识在每个专家里重复学习**（参数冗余），而不是「给个兜底」。
- **参数量账**：能否算清「总参/激活参」在两种设计下的分配（例如 V3 的 671B/37B 与 Mixtral 的 46.7B/12.9B）。
- **代价的完整性**：路由/通信开销、访存碎片化、负载均衡压力、实现复杂度——**只讲优点是不合格的**。
- **与负载均衡的联系**：能否指出细粒度让均衡更容易（专家小、粒度细）但长尾更难控（串 [[deepseek-02]]）。
- **与推理服务的关系**：专家多 → 显存放置与 all-to-all 成为主要矛盾（串 [[deepseek-05]]、[[cohere-02]]）。
- **实现细节**：能否说出「共享专家不参与路由」、「路由用 top-k 而非 top-2」、「门控权重需归一化」等实现点（串 [[deepseek-04]]）。
- **不神化**：能否承认这是**工程取舍**（同等激活参数下的容量 vs 实现复杂度），而不是「必然更好」。

**常见错误答案**

- 只说「专家更多所以更强」（没有组合数/专业化的机制）。
- 把共享专家说成「防止路由失败的兜底」（实际是**通用知识的集中承载**）。
- 忽略代价（通信、访存、均衡、实现复杂度）。
- 混淆「总参数」与「激活参数」（把 671B 当成每 token 都要算）。
- 认为细粒度只是「把 FFN 切小」（忽略了 top-k 的 k 也要相应变大）。
- 不提负载均衡（细粒度专家的核心工程难点）。

## 原理与推导

### 1. 组合空间：细粒度的数学来源

设 $N$ 个专家、每 token 选 $k$ 个，则可能的「专家组合」数是

$$C(N,k)=\binom{N}{k}$$

| 配置 | $N$ | $k$ | 组合数 |
| --- | --- | --- | --- |
| Mixtral 8x7B | 8 | 2 | 28 |
| 中等细粒度 | 64 | 4 | 635,376 |
| DeepSeek-V3 量级 | 256 | 8 | $\approx4.1\times10^{14}$ |

**读法**：粗粒度下只有 28 种「专家组合」，**token 只能被分到 28 类子空间里**；细粒度下组合数爆炸，**专业化空间大得多**。这就是「ultimate expert specialization」的字面含义。

### 2. 参数量账（同等激活参数下的对比）

| 模型 | 总参 | 激活参 | 专家配置 | 每专家规模（相对 FFN） |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | 46.7B | 12.9B | 8 专家 top-2 | 每个约等于一个完整 FFN |
| DeepSeek-V3 | 671B | 37B | 256 路由 + 1 共享，top-8 | 每个约为 FFN 的 $1/m$（$m$ 为细分因子） |

**关键关系**：细粒度下每个专家是「FFN 的一段」，所以

$$\text{激活参}\approx\text{注意力}+\text{共享专家}+\frac{k}{N}\times\text{路由专家总参}$$

**在总参固定时，细粒度让「激活比例」更容易做小**（$k/N$ 更小），从而**用更大的总参换更低的每 token 算力**。

### 3. 共享专家：消除冗余的论证

设通用知识需要 $P_{\text{common}}$ 参数才能学好。**无共享专家**时，由于路由是稀疏的（每个 token 只走 $k$ 个专家），**每个专家都必须具备通用能力**（否则它接到的 token 会因为缺少通用能力而表现差）：

$$P_{\text{redundant}}\approx N_{\text{routed}}\times P_{\text{common}}$$

**有共享专家**时，通用能力只需一份：

$$P_{\text{common}}\ \text{（一份）}+N_{\text{routed}}\times P_{\text{special}}$$

**读法**：共享专家不是「兜底」，而是**把公共部分从 $N$ 份压到 1 份**，让路由专家的容量全部用于差异化——这是参数量意义上的**去冗余**。

### 4. 负载均衡：细粒度**并不自动更均衡**

直觉上「专家更小 → 负载更平滑」，但**本机模拟给出了相反的结论**（见下表 2）：在**偏好离散度固定**的前提下，专家数从 8 增到 256，**最大/平均负载从 1.36× 升到 2.13×**、capacity factor=1.25 下的丢弃率从 1.41% 升到 3.65%。原因是**极值效应**：

$$\mathbb{E}\Big[\max_{i\le N}\frac{\ell_i}{\bar\ell}\Big]\ \text{随 } N \text{ 上升}\quad(\text{候选越多，最热的那个越极端})$$

**结论**：细粒度把「专家容量」这件事变得更需要**主动管理**——这正是 DeepSeek 要用**无辅助损失的 bias 均衡**（[[deepseek-02]]）的原因。**注意本结论依赖「偏好离散度不随粒度变化」这一假设**；若细粒度同时让偏好更均匀（每个专家功能更窄），趋势可能反转——所以真实系统必须实测。

### 5. 工程代价

| 代价 | 机制 | 缓解 |
| --- | --- | --- |
| all-to-all 更碎 | 专家多 → 目标更多、消息更小 | 分层 all-to-all、专家分组、合并小消息 |
| 访存碎片化 | 每个专家矩阵小 → 单次访存利用率低 | 专家分桶、批内合并、算子融合 |
| 显存放置难 | 专家数多、每卡容量有限 | 专家并行 + 热点复制（串 [[cohere-02]]） |
| 实现复杂 | top-k 的 k 大、共享专家单独路径 | 见 [[deepseek-04]] 的陷阱清单 |
| 均衡压力 | 长尾分布 | 无辅助损失的 bias 技巧（[[deepseek-02]]） |

## 数值与代码验证

### 表 1：三种专家配置的组合空间与参数量分配（见代码输出）

| 配置 | 组合数 | 总参 | 激活参 | 激活比例 | 每专家相对规模 |
| --- | --- | --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：细粒度对负载均衡的影响（模拟路由）

| 配置 | 最大/平均负载 | 丢弃率（capacity factor=1.25） |
| --- | --- | --- |
| 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# DeepSeekMoE vs Mixtral：组合空间、参数分配、共享专家的去冗余、细粒度对均衡的影响
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 组合空间 ----------
def comb(n: int, k: int) -> int:
    return math.comb(n, k)

CONFIGS = [
    ("Mixtral 8x7B（粗粒度 top-2）", 8, 2),
    ("中等细粒度 top-4", 64, 4),
    ("DeepSeek-V3 量级（256 专家 top-8）", 256, 8),
]
print("① 专家组合空间：MoE 的容量来自组合数，不是专家数")
print(f"  {'配置':<34} {'N':>5} {'k':>3} {'组合数':>18} {'相对 top-2':>12}")
base = comb(8, 2)
for name, n, k in CONFIGS:
    c = comb(n, k)
    print(f"  {name:<34} {n:>5} {k:>3} {c:>18,} {c/base:>11.3g}x")
print("  读法：**粗粒度只有 28 种组合**（token 只能落进 28 类子空间），细粒度到 $10^{14}$ 量级 ——")
print("        这就是「专家专业化」的数学来源；注意 k 也要随 N 一起放大，否则细粒度失去意义")

# ---------- 2) 参数量账：细粒度如何把激活比例做小 ----------
@dataclass
class MoEConfig:
    name: str
    n_routed: int
    top_k: int
    shared: int                 # 共享专家数（每个约等于一个专家）
    expert_unit_b: float        # 每个"标准 FFN"的参数量（十亿），用于折算
    attention_b: float          # 注意力与嵌入等非 FFN 部分
    def routed_total_b(self) -> float:
        """路由专家总参：细粒度下每个专家是标准 FFN 的 1/subdiv"""
        subdiv = self.n_routed / 8.0        # 以 8 个专家为"粗粒度基准"
        return 8 * self.expert_unit_b      # 总 FFN 容量固定为 8 个标准 FFN
    def total_b(self) -> float:
        return self.attention_b + self.routed_total_b() + self.shared * self.expert_unit_b / (self.n_routed / 8.0)
    def active_b(self) -> float:
        subdiv = self.n_routed / 8.0
        routed_active = self.top_k * (8 * self.expert_unit_b) / self.n_routed
        shared_active = self.shared * self.expert_unit_b / subdiv
        return self.attention_b + routed_active + shared_active
    def per_expert_rel(self) -> float:
        return 1.0 / (self.n_routed / 8.0)

MI = MoEConfig("Mixtral 风格（8 专家 top-2，无共享）", 8, 2, 0, 6.0, 5.0)
DS = MoEConfig("DeepSeekMoE 风格（256 专家 top-8 + 1 共享）", 256, 8, 1, 6.0, 5.0)
print("")
print("② 参数量账：同等 FFN 总容量下的激活比例")
print(f"  {'配置':<40} {'总参':>8} {'激活参':>8} {'激活比例':>9} {'每专家相对规模':>13}")
for c in (MI, DS):
    print(f"  {c.name:<40} {c.total_b():>7.1f}B {c.active_b():>7.1f}B "
          f"{c.active_b()/c.total_b():>8.1%} {c.per_expert_rel():>13.3f}")
print("  读法：**细粒度让 k/N 更小 → 同等总参下激活比例更低**（用更大的总参换更低的每 token 算力）；")
print("        代价是每个专家更小（最后一列），访存与调度更碎")

# ---------- 3) 共享专家的去冗余 ----------
@dataclass
class SharedExpert:
    n_routed: int
    p_common: float = 1.0        # 通用能力所需的"专家份额"（以 1 个专家为单位）
    p_special: float = 0.6       # 每个专家需要的差异化能力份额
    def without(self) -> float:
        """无共享：每个路由专家都必须自带通用能力"""
        return self.n_routed * (self.p_common + self.p_special)
    def with_shared(self) -> float:
        """有共享：通用能力只留一份"""
        return self.p_common + self.n_routed * self.p_special
    def saving(self) -> float:
        return 1 - self.with_shared() / self.without()
print("")
print("③ 共享专家的去冗余（以「每个专家都要会通用模式」为前提）")
print(f"  {'路由专家数':>10} {'无共享所需':>11} {'有共享所需':>11} {'节省':>8}")
for n in (8, 64, 256):
    s = SharedExpert(n)
    print(f"  {n:>10} {s.without():>11.1f} {s.with_shared():>11.1f} {s.saving():>7.1%}")
print("  读法：**共享专家把通用能力的开销从 N 份压到 1 份** —— 专家越多，节省越显著；")
print("        这不是「兜底」，而是**参数量意义上的去冗余**（让路由专家的容量全用于差异化）")

# ---------- 4) 细粒度对负载均衡的影响（模拟） ----------
@dataclass
class Router:
    n_experts: int
    top_k: int
    imbalance: float = 0.30
    def token_loads(self, tokens: int, seed: int = 7) -> List[int]:
        rnd = random.Random(seed)
        w = [math.exp(rnd.gauss(0, self.imbalance)) for _ in range(self.n_experts)]
        tot = sum(w); p = [x / tot for x in w]
        loads = [0] * self.n_experts
        for _ in range(tokens * self.top_k):
            r = rnd.random(); acc = 0.0
            for i, pi in enumerate(p):
                acc += pi
                if r <= acc:
                    loads[i] += 1
                    break
            else:
                loads[-1] += 1
        return loads
    def drop_rate(self, loads: List[int], cf: float = 1.25) -> Tuple[float, float]:
        avg = sum(loads) / len(loads)
        cap = cf * avg
        dropped = sum(max(0, l - cap) for l in loads)
        return max(loads) / avg, dropped / sum(loads)
print("")
print("④ 细粒度对负载均衡的影响（同样的偏好不均衡度 σ=0.30、10 万 token）")
print(f"  {'配置':<28} {'最大/平均负载':>13} {'丢弃率(cf=1.25)':>16}")
for name, n, k in (("粗粒度 8 专家 top-2", 8, 2), ("中粒度 64 专家 top-4", 64, 4),
                   ("细粒度 256 专家 top-8", 256, 8)):
    r = Router(n, k)
    loads = r.token_loads(100_000)
    ratio, drop = r.drop_rate(loads)
    print(f"  {name:<28} {ratio:>13.2f}x {drop:>16.2%}")
print("  读法：**在偏好离散度固定的前提下，专家越多、最大/平均负载反而越高**（极值效应：")
print("        候选专家多 -> 最热的那一个更极端），丢弃率也从 1.41% 升到 3.65% ——")
print("        所以**细粒度并不自动带来更好的均衡，反而更需要主动均衡**（这正是 deepseek-02 的动机）")

# ---------- 5) 工程代价清单 ----------
@dataclass
class Cost:
    item: str
    mechanism: str
    mitigation: str
COSTS = [
    Cost("all-to-all 更碎", "专家多 -> 目标多、消息小", "分层 all-to-all、专家分组、消息合并"),
    Cost("访存碎片化", "单专家矩阵小 -> 访存利用率低", "专家分桶、批内合并、算子融合"),
    Cost("显存放置难", "专家多、每卡容量有限", "专家并行 + 热点专家复制"),
    Cost("实现复杂", "top-k 的 k 大、共享专家单独路径", "统一路由接口 + 契约测试"),
    Cost("均衡压力", "长尾分布 + 组合多", "无辅助损失的 bias 均衡"),
]
print("")
print("⑤ 细粒度的工程代价与缓解")
print(f"  {'代价':<16} {'机制':<30} 缓解")
for c in COSTS:
    print(f"  {c.item:<16} {c.mechanism:<30} {c.mitigation}")
print("  读法：**细粒度不是免费的容量** —— 它把矛盾从「参数不够」转移到「通信、访存与调度」；")
print("        面试里主动讲这份代价清单，比只夸优点更能体现工程判断")
```

预期输出要点（实跑）：① 组合空间从 Mixtral 的 **28** 跳到 256 专家 top-8 的 **$10^{14}$ 量级**——这是「专家专业化」的数学来源；② 参数量账显示**细粒度让 $k/N$ 更小 → 同等总参下激活比例更低**（用大总参换低算力），代价是**每个专家更小**；③ **共享专家的去冗余**随专家数增加而更显著（把通用能力从 $N$ 份压到 1 份）；④ **细粒度并不自动更均衡**：在偏好离散度固定时，专家数 8→256 让最大/平均负载从 **1.36× 升到 2.13×**、丢弃率从 1.41% 升到 3.65%（极值效应）——这恰好解释了 DeepSeek 为什么需要主动的 bias 均衡；⑤ 工程代价清单把矛盾从「参数」转移到「通信、访存与调度」。

## 常见追问

- **追问**：细粒度是不是「越多越好」？
  - 要点：不是——**每个专家越小，访存与调度效率越差**（矩阵太小、kernel 启动占比上升），all-to-all 的消息也更碎。存在一个**与硬件匹配的最优点**（取决于 HBM 带宽、互联与 kernel 效率）。**实践上通过「专家大小 ≈ 某个算子的高效尺寸」来定**，而不是无限细分。
- **追问**：共享专家会不会变成「瓶颈」？
  - 要点：它**每 token 都要算**，所以是**计算上的固定开销**（相当于激活参数里多了一份 FFN）。设计上要让它**足够小**（或与路由专家的总量成比例），否则会吃掉细粒度带来的收益。**它是「用一点固定算力换大量参数节省」的交易。**
- **追问**：为什么 Mixtral 不做细粒度？
  - 要点：**工程复杂度与推理效率的取舍**：粗粒度实现简单（8 个专家、top-2，all-to-all 目标少），在中小规模下性价比高；细粒度在**超大规模**（数百专家）才显出容量优势，但需要更强的并行与均衡基础设施。**这是「规模驱动设计」的典型例子。**
- **追问**：细粒度下 top-k 的 k 怎么定？
  - 要点：让**激活参数量**保持在目标预算内：$k/N\times$ 路由专家总参 ≈ 目标激活 FFN 容量。k 太小则细粒度失去意义（组合没打开），k 太大则激活算力上升。**经验上 k 随 N 增长（如 N=256 时 k=8）。**
- **追问**：这套设计对推理服务意味着什么？
  - 要点：① **显存**：所有专家都要驻留（串 [[cohere-02]] 的「显存按总参付」）；② **通信**：all-to-all 的目标数与消息数上升，低 batch 时延迟主导；③ **放置**：专家并行 + 热点复制成为必需；④ **批处理**：细粒度更需要**大 batch** 摊薄调度与通信开销（串 [[deepseek-05]]）。
- **追问**：共享专家与「多任务学习里的共享底层」是一回事吗？
  - 要点：**思想同源**（共享 + 特化），但机制不同：多任务学习共享的是**任务无关的底层表示**，MoE 的共享专家是**始终激活的 FFN 分支**，与路由专家在同一层并行。**共同点是「把公共部分集中、把差异部分分散」。**

## 相关题目

- [[deepseek-02]]：无辅助损失的负载均衡——细粒度专家更需要主动均衡。
- [[deepseek-04]]：用 PyTorch 实现带共享专家的 top-k 路由——本篇设计的实现形态与陷阱。
- [[deepseek-05]]：671B MoE 的低延迟服务——本篇的显存/通信结论在服务侧的应用。
- [[cohere-02]]：稀疏 MoE 为何适合私有化部署——「显存按总参、算力按激活」的同一主题。
- [[inference-serving-07]]：张量/流水/数据/专家并行——细粒度对并行策略的要求。

## 参考资料与归属

- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（延伸）** —— Dai et al. (DeepSeek-AI)，2024-01-11：<https://arxiv.org/abs/2401.06066>。第 1、2 节的**细粒度专家分割**与**共享专家隔离**两项设计及其动机（组合空间、减少冗余）来自这篇。
- **Mixtral of Experts（延伸）** —— Jiang et al. (Mistral AI)，2024-01-08：<https://arxiv.org/abs/2401.04088>。第 1 节的**8 专家 top-2** 配置与总参/激活参口径来自这篇。
- **DeepSeek-V3 Technical Report（延伸）** —— DeepSeek-AI，2024-12-27：<https://arxiv.org/abs/2412.19437>。第 1 节的 256 路由专家 + 1 共享专家、top-8 路由、671B/37B 参数量口径来自这篇。
- **稀疏 MoE 为何适合私有化部署（本仓库公司题库 · Cohere 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1 节「显存按总参、算力按激活」的框架与 capacity factor 口径被本篇沿用。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（组合数计算、FFN 总容量折算为 8 个标准专家、注意力 5B、每专家 6B、共享专家份额 1.0/0.6、偏好不均衡 σ=0.30、10 万 token、cf=1.25）都是为演示机制而构造的**示例参数与简化模型**；真实模型的参数切分、专家规模与激活量以官方技术报告为准，**本文的「总参/激活参」折算只用于展示趋势**（例如把 FFN 总容量固定为 8 个标准 FFN 是人为设定）。**组合空间公式 $\binom{N}{k}$、共享专家的去冗余论证与「细粒度降低激活比例」这三条结构性结论可迁移。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
