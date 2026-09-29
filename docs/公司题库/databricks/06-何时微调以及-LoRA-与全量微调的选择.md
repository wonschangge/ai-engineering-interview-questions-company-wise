---
type: question
id: databricks-06
company: Databricks
topic: finetuning
order: 6
question: 什么情况下你会选择 fine-tuning，而不是用 RAG 或 prompt engineering？如果确实要微调，是选 LoRA 还是全量微调？
question_en: When would you choose fine-tuning over RAG or prompt engineering? And if you do fine-tune, LoRA or full fine-tuning?
asked_at: []
level: 高阶
tags: [微调, RAG, LoRA, QLoRA, 成本模型, 决策框架]
sources:
  - title: LoRA: Low-Rank Adaptation of Large Language Models（延伸）
    url: https://arxiv.org/abs/2106.09685
    author: Hu et al. (ICLR 2022)
    published: 2021-06-17
  - title: QLoRA: Efficient Finetuning of Quantized LLMs（延伸）
    url: https://arxiv.org/abs/2305.14314
    author: Dettmers et al. (NeurIPS 2023)
    published: 2023-05-23
  - title: Prompting、RAG 还是 fine-tuning：决策框架与成本权衡（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 为大型商品目录设计语义搜索（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [databricks-07, databricks-02, finetuning-08, finetuning-05, rag-01]
updated: 2026-09-28
---

## 一句话答案

> **一句判据：RAG 改「知道什么」，微调改「怎么做」。**
> 具体分工：
> | 需求 | 首选 | 理由 |
> | --- | --- | --- |
> | 知识会变、要引用、要按用户权限过滤 | **RAG** | 知识在外部索引里，更新与鉴权都容易 |
> | 知识量远超权重能记住的 | **RAG** | 权重里塞不进也用不好 |
> | 风格/语气/格式稳定、术语一致 | **微调（LoRA）** | 这是「行为」而非「知识」，几百万 token 就能固化 |
> | 输出结构严格（JSON/SQL/工具调用） | **微调** | 比 prompt 更稳，且能缩短 prompt 省成本 |
> | 想把大模型能力蒸馏到小模型（降本/降延迟） | **微调（蒸馏）** | 这是「行为迁移」，RAG 做不到 |
> | 只是想让模型「知道」几条新规则 | **prompt** | 先试 prompt，最便宜 |
> **做微调前必须先有的三样东西**：① **可度量的评测集**（否则无法判断有没有变好）；② 一个**要打败的基线**（prompt / RAG / few-shot）；③ **足够且干净、有授权的数据**（风格类几百到几千条，行为类几千到几万条）。**缺任何一样，先去做那一样。**
> **LoRA vs 全量微调**：
> - **默认选 LoRA**：可训练参数少 2–3 个数量级（显存与算力都省）、**一个基座可服务多租户的多个适配器**（这是多客户场景的决定性优势）、不易灾难性遗忘、容易回滚（丢掉适配器即可）；
> - **选全量微调**：需要**大幅改变模型行为/分布**（新语言、新领域预训练、深度风格迁移）、或要**把知识与能力真正烧进权重**（蒸馏出可独立部署的小模型）、且**有资源与评测能力**；
> - **中间档**：**QLoRA**（4-bit 基座 + LoRA，单卡可微调大模型）、**继续预训练**（补领域词表/语料后再 LoRA）、**适配器合并**（推理零额外延迟）。
> **成本上还有一个常被忽略的对照项**：如果微调的动机是「**省 prompt token**」，**prompt caching 往往更便宜、更快、风险更低**（串 [[anthropic-22]]）——先算这笔账再决定微调。

## 面试官在考什么

- **是否先问动机**：能否把需求拆成「知识 vs 行为 vs 成本/延迟 vs 蒸馏」，而不是直接回答「用 RAG」或「用微调」。
- **前置条件**：能否坚持「先有评测集与基线」——**没有评测的微调是赌博**（这是最容易扣分的一点）。
- **LoRA 的核心优势**：不只是「省显存」，还有**多租户服务**（一个基座 + N 个适配器）与**易回滚**；能否说清 LoRA 的**局限**（注入新事实效果差、秩与目标模块的选择影响效果）。
- **显存算术**：全量微调的**优化器状态**是大头（Adam 约 16 字节/参数：fp32 权重副本 + 动量 + 方差），70B ≈ **1.1 TB**，必须靠 ZeRO/FSDP/分片；LoRA 只训练适配器（可小 2–3 个数量级），但**前向仍要跑整个基座**（所以激活显存与序列长度仍受限）。
- **数据与评测的工程化**：训练/验证/测试切分、防止**测试集泄漏**、以及「微调后是否回退通用能力」（用通用评测守住）。
- **成本对比的完整性**：训练成本 + **服务成本**（专用部署 vs 共享基座）+ **数据标注** + **维护**（漂移后重训）。只看训练成本会严重低估。
- **风险**：灾难性遗忘、过拟合、数据合规（客户数据能否用于训练）、以及**推理时无法引用来源**（微调学到的知识不可溯源——在合规场景是硬伤）。
- **替代方案**：prompt caching、更小的模型 + 路由、few-shot、结构化输出约束（JSON schema）、以及**RAG + LoRA 组合**（知识走检索、行为走适配器）。

**常见错误答案**

- 直接说「用 RAG 就行」或「直接微调」（不看需求）。
- 微调前没有评测集与基线（无法证明收益）。
- 认为 LoRA 一定「效果差」或全量微调一定「效果最好」（取决于任务类型）。
- 只看训练成本，不看服务成本与维护成本。
- 忽略数据合规（把客户数据直接拿去训练）。
- 想用微调「注入知识」（这是 RAG 的活，且微调注入的知识无法引用与更新）。
- 不考虑 prompt caching 这条更便宜的路径。

## 原理与推导

### 1. 决策框架（三分支）

```
需要"知道"某些东西吗？
├─ 是 → 会变？要引用？要鉴权？ → RAG（几乎总是）
└─ 否 → 需要"改变行为/风格/格式"吗？
        ├─ 是 → 先试 prompt（few-shot + 结构化约束）
        │        └─ 不够稳/成本高 → LoRA
        └─ 否 → 需要"降本/降延迟/蒸馏"吗？
                 └─ 是 → 微调（通常蒸馏到小模型）
```

**关键区分**：
- **知识**（facts）→ 外部化（RAG）：**可更新、可引用、可按权限过滤**；
- **行为**（style/format/tool-use）→ 参数化（微调）：**稳定、省 token、可离线固化**。

### 2. 显存与算力算术（LoRA vs 全量）

**全量微调**（Adam，混合精度）每参数大致需要：

| 项 | 字节/参数 |
| --- | --- |
| 模型权重（bf16） | 2 |
| 梯度（bf16/fp32） | 2–4 |
| 优化器状态（fp32 一阶 + 二阶 + fp32 主权重） | 12 |
| **合计** | **约 16–18** |

70B → **1.1–1.3 TB**（不含激活）→ 必须 **ZeRO/FSDP 分片**，至少 16–32 张 80 GB 卡才能装下。

**LoRA**：只训练低秩矩阵 $B\cdot A$（$r\ll d$），可训练参数 $\approx 2r\cdot d\cdot L_{\text{target}}$。以 $r=16$、只挂注意力投影为例，可训练参数常为**总参数的 0.1%–1%**，优化器状态同步缩小 → **单卡或数卡可做**。但注意：

$$T_{\text{step}}\approx\underbrace{\text{前向}+\text{反向（含基座）}}_{\text{与全量同量级}}+\underbrace{\text{适配器更新}}_{\text{可忽略}}$$

**所以 LoRA 省的是显存与优化器计算，不省前向/反向的主体计算**——训练时间不会按参数比例线性下降（通常快 1.5–3×，而不是 100×）。

### 3. 训练算力估算（与仓库统一口径一致）

$$\text{GPU 小时}\approx\frac{6ND}{\text{MFU}\times\text{峰值算力}\times3600}\quad(\text{全量}）$$

其中 $N$ 为参数量、$D$ 为训练 token 数。**LoRA 时 $N$ 换成「等效计算参数量」≈ 基座参数量**（因为前向/反向仍过全模型），但**显存占用按适配器算**——这个差异很关键：

- **算力（时间）**：LoRA 与全量接近（都过全模型），LoRA 略快（无需优化器状态更新）；
- **显存（能不能做）**：LoRA 显著低（这是它最大的工程价值）。

### 4. 服务成本（最容易被忽略的一环）

| 方案 | 服务形态 | 多租户可行性 | 单位成本 |
| --- | --- | --- | --- |
| prompt only | 共享基座 | ✅ | 最低（无需训练） |
| RAG | 共享基座 + 索引 | ✅ | 索引存储 + 检索延迟 |
| **LoRA** | **共享基座 + 多适配器** | ✅（**一个基座服务 N 个客户**） | 适配器切换开销很小 |
| 全量微调 | **每客户/每版本一套权重** | ❌（通常要独立部署或换权重） | GPU 成本 ×N |

**读法**：在多客户/多任务场景里，**LoRA 的经济性优势主要来自「共享基座」**，而不是训练本身便宜——这一点在 SaaS 里往往是决定性的。

### 5. 数据量与任务类型的经验区间

| 目标 | 数据量级（示例） | 备注 |
| --- | --- | --- |
| 风格/语气/格式对齐 | 数百–数千条 | 质量 > 数量；要覆盖边界 |
| 工具调用/结构化输出 | 数千–数万条 | 需要包含负例与纠错样本 |
| 领域术语与回答模式 | 数千–数万条 | 与 RAG 组合效果最好 |
| 新语言/新领域（大偏移） | 数十万–数百万条 | 通常先继续预训练再 SFT |
| 知识注入（事实） | **不建议** | 用 RAG |

### 6. 与 prompt caching 的对照（先算这笔账）

若微调动机是「**省 prompt token**」（例如每次都要塞 2,000 token 的说明）：

$$\text{节省}=\text{QPS}\times\text{节省的 prompt token}\times p_{\text{in}}\quad\text{vs}\quad \text{微调的固定成本}$$

**但 caching 也能省同样的 token**（缓存命中价通常为输入价的 10%–25%），且**零训练风险**。所以顺序是：

$$\text{prompt 优化}\to\text{caching}\to\text{RAG}\to\text{LoRA}\to\text{全量微调}$$

**每往下一级，固定成本与风险都上升一个台阶**——只有当上一级「确实做不到」时才往下走。

## 数值与代码验证

### 表 1：70B 模型的显存算术（见代码输出）

| 方案 | 权重 | 优化器状态 | 可训练参数 | 最低 GPU 数（80 GB） |
| --- | --- | --- | --- | --- |
| 全量微调（Adam, bf16+fp32 主权重） | 见输出 | 见输出 | 见输出 | 见输出 |
| LoRA（r=16，注意力投影） | 见输出 | 见输出 | 见输出 | 见输出 |
| QLoRA（4-bit 基座 + LoRA） | 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：多租户服务成本（N 个客户）

| 方案 | 基座实例数 | 适配器/权重 | 月成本倍数 |
| --- | --- | --- | --- |
| 共享基座 + prompt | 见输出 | — | 1× |
| 共享基座 + LoRA 适配器 | 见输出 | N 个（小） | 见输出 |
| 全量微调（每客户一套） | 见输出 | N 套（大） | 见输出 |

### 可运行代码

```python
# fine-tuning vs RAG vs prompt：决策打分 + 显存/成本算术 + 多租户服务成本
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

GB = 1024 ** 3

# ---------- 1) 显存算术：全量 vs LoRA vs QLoRA ----------
@dataclass
class FTPlan:
    name: str
    params_b: float                    # 基座参数量（十亿）
    trainable_ratio: float             # 可训练参数占比
    bytes_per_trainable: float         # 每可训练参数的显存（权重+梯度+优化器状态）
    base_weight_bytes: float           # 每参数的基座权重字节（bf16=2, 4-bit≈0.5）
    gpu_mem_gb: float = 80.0
    def trainable_params(self) -> float:
        return self.params_b * 1e9 * self.trainable_ratio
    def memory_gb(self) -> Dict[str, float]:
        base = self.params_b * 1e9 * self.base_weight_bytes
        opt = self.trainable_params() * self.bytes_per_trainable
        # 激活显存：与序列长度/批大小相关，这里按经验取基座权重的 25%
        act = base * 0.25
        return {"基座权重(GB)": base / GB, "优化器/梯度(GB)": opt / GB,
                "激活(GB)": act / GB, "合计(GB)": (base + opt + act) / GB}
    def min_gpus(self) -> int:
        return max(1, math.ceil(self.memory_gb()["合计(GB)"] / self.gpu_mem_gb))

PLANS = [
    FTPlan("全量微调（Adam, bf16 权重 + fp32 状态）", 70, 1.0, 16.0, 2.0),
    FTPlan("LoRA（r=16，注意力投影）", 70, 0.006, 16.0, 2.0),
    FTPlan("QLoRA（4-bit 基座 + LoRA）", 70, 0.006, 16.0, 0.5),
]
print("① 70B 模型的训练显存算术（不含激活细节，激活按基座权重 25% 粗估）")
print(f"  {'方案':<36} {'基座':>9} {'优化器':>9} {'激活':>8} {'合计':>9} {'最低卡数(80GB)':>14}")
for p in PLANS:
    m = p.memory_gb()
    print(f"  {p.name:<36} {m['基座权重(GB)']:>9.0f} {m['优化器/梯度(GB)']:>9.0f} "
          f"{m['激活(GB)']:>8.0f} {m['合计(GB)']:>9.0f} {p.min_gpus():>14}")
print("  读法：**全量微调的门槛在优化器状态**（约 16 字节/参数 → 70B 需 1.1 TB 级显存）；")
print("        LoRA 省的是这一项（可训练参数少 2–3 个数量级），但**前向/反向仍过整个基座**")

# ---------- 2) 训练算力：LoRA 不会按参数比例变快 ----------
PEAK_FLOPS = 989e12      # H100 bf16 dense（仓库统一常数）
def train_gpu_hours(params_b: float, tokens: float, mfu: float = 0.40,
                    speedup_vs_full: float = 1.0) -> float:
    """6ND 口径；LoRA 用 speedup_vs_full 表示相对全量的实际加速（通常 1.5–3×）"""
    flops = 6 * params_b * 1e9 * tokens
    hours_full = flops / (PEAK_FLOPS * mfu * 3600)
    return hours_full / speedup_vs_full
print("\n② 训练算力（6ND 口径、MFU 40%、H100 bf16 989 TFLOPs）")
print(f"  {'场景':<34} {'tokens':>10} {'GPU 小时':>10} {'卡数':>6} {'墙钟(小时)':>10}")
for name, params, tokens, speed, gpus in (
        ("全量微调 70B", 70, 1e9, 1.0, 32),
        ("LoRA 70B（前向/反向仍过基座）", 70, 1e9, 2.0, 8),
        ("LoRA 8B（小模型 + 蒸馏）", 8, 1e9, 2.0, 4),
        ("继续预训练 8B（100 亿 token）", 8, 10e9, 2.0, 8)):
    gh = train_gpu_hours(params, tokens, speedup_vs_full=speed)
    print(f"  {name:<34} {tokens/1e9:>9.0f}B {gh:>10,.0f} {gpus:>6} {gh/gpus:>10.1f}")
print("  读法：**LoRA 不会把训练时间按参数比例压下来**（前向/反向仍过全模型），实测通常快 1.5–3×；")
print("        它的决定性优势是**显存门槛**（能不能做）与**多租户服务**（下面第 ④ 节）")

# ---------- 3) 决策打分：知识 / 行为 / 成本 / 蒸馏 ----------
@dataclass
class Need:
    name: str
    is_knowledge: float        # 0-1：需求本质是"知道某些事实"
    changes_often: float       # 0-1：内容更新频率
    needs_citation: float      # 0-1：是否需要引用来源
    needs_permission: float    # 0-1：是否需要按用户权限过滤
    behavior_shift: float      # 0-1：需要改变风格/格式/行为
    cost_pressure: float       # 0-1：对 token 成本/延迟的压力
    def score_rag(self) -> float:
        return (0.35 * self.is_knowledge + 0.25 * self.changes_often
                + 0.2 * self.needs_citation + 0.2 * self.needs_permission)
    def score_ft(self) -> float:
        return (0.45 * self.behavior_shift + 0.25 * self.cost_pressure
                + 0.15 * self.is_knowledge * (1 - self.changes_often)
                + 0.15 * (1 - self.needs_citation))
NEEDS = [
    Need("企业知识问答（制度/流程会更新）", 0.9, 0.8, 0.9, 0.7, 0.2, 0.3),
    Need("客服回复风格与话术统一", 0.2, 0.2, 0.1, 0.1, 0.9, 0.6),
    Need("严格 JSON/SQL 结构化输出", 0.1, 0.1, 0.0, 0.0, 0.85, 0.7),
    Need("把大模型能力蒸馏到 8B", 0.2, 0.2, 0.1, 0.2, 0.9, 0.9),
    Need("私有代码库的问答与检索", 0.95, 0.6, 0.9, 0.8, 0.3, 0.4),
    Need("每条 prompt 带 2k token 说明、想省钱", 0.3, 0.3, 0.1, 0.1, 0.4, 0.95),
]
print("\n③ 决策打分（谁更适合 RAG / 微调）")
print(f"  {'需求':<34} {'RAG':>6} {'微调':>6} {'建议':<28}")
for n in NEEDS:
    s_rag, s_ft = n.score_rag(), n.score_ft()
    if n.cost_pressure > 0.9 and n.behavior_shift < 0.5 and n.is_knowledge < 0.5:
        advice = "先试 caching / 精简 prompt"      # ★ 成本动机先走零风险路径
    elif max(s_rag, s_ft) < 0.4:
        advice = "先优化 prompt（最便宜）"
    elif abs(s_rag - s_ft) < 0.12:
        advice = "RAG + LoRA 组合（知识走检索、行为走适配器）"
    else:
        advice = "RAG" if s_rag > s_ft else "微调（LoRA 优先）"
    print(f"  {n.name:<34} {s_rag:>6.2f} {s_ft:>6.2f} {advice:<28}")
print("  读法：**知识类需求几乎总是 RAG**（可更新、可引用、可鉴权）；**行为类需求才是微调的地盘**；")
print("        而「省 prompt token」这类需求**要先试 caching**（见第 ⑤ 节），再考虑微调")

# ---------- 4) 多租户服务成本：LoRA 的决定性优势 ----------
@dataclass
class Serving:
    customers: int = 50
    gpus_per_base: int = 8                 # 一个基座实例占 8 卡
    gpu_price_h: float = 2.0               # $/GPU·h（仓库统一口径）
    hours_per_month: float = 730
    adapter_mem_gb: float = 0.2            # 每个 LoRA 适配器显存（示例）
    full_ft_gpus: int = 8                  # 全量微调后每客户一套权重所需的卡数
    def shared_base_cost(self) -> float:
        base = self.gpus_per_base * self.gpu_price_h * self.hours_per_month
        # 适配器几乎不额外占卡（同一基座可挂多个适配器，按请求切换）
        return base * 3                        # 3 个基座实例做冗余/负载
    def per_customer_full_cost(self) -> float:
        return self.full_ft_gpus * self.gpu_price_h * self.hours_per_month
s = Serving()
print(f"\n④ 多租户服务成本（{s.customers} 个客户、\\${s.gpu_price_h}/GPU·h）")
shared = s.shared_base_cost()
per_full = s.per_customer_full_cost()
print(f"  共享基座 + LoRA 适配器：3 个基座实例 × {s.gpus_per_base} 卡 = "
      f"\\${shared:,.0f}/月（**与客户数基本无关**）")
print(f"  全量微调（每客户一套权重）：{s.customers} × {s.full_ft_gpus} 卡 = "
      f"\\${per_full * s.customers:,.0f}/月")
print(f"  成本比：{per_full * s.customers / shared:,.1f}×")
for c in (5, 20, 50, 200):
    s2 = Serving(customers=c)
    print(f"    客户数 {c:>4}：共享 \\${s2.shared_base_cost():>10,.0f} vs "
          f"全量 \\${s2.per_customer_full_cost() * c:>12,.0f}")
print("  读法：**LoRA 的经济性主要来自「共享基座」** —— 客户数越多，差距越大；")
print("        这也是 SaaS 场景几乎总选 LoRA 的原因（全量微调只在「要独立部署/深度定制」时才合理）")

# ---------- 5) 先算 caching 这笔账 ----------
def ft_break_even_qps(prompt_tokens: int, price_in: float = 3.0,
                      cache_discount: float = 0.1, ft_fixed_month: float = 20_000,
                      ft_saving_ratio: float = 1.0) -> Dict[str, float]:
    """微调的动机若是省 prompt token：比较 caching 与微调的盈亏平衡 QPS"""
    per_call_full = prompt_tokens / 1e6 * price_in
    per_call_cached = per_call_full * cache_discount
    per_call_ft = 0.0                        # 微调后 prompt 变短，视为 0
    saved_by_cache = per_call_full - per_call_cached
    saved_by_ft = per_call_full * ft_saving_ratio - per_call_ft
    return {"缓存每次省($)": saved_by_cache, "微调每次省($)": saved_by_ft,
            "微调月固定成本($)": ft_fixed_month,
            "微调的盈亏平衡月调用数": ft_fixed_month / saved_by_ft if saved_by_ft else float("inf"),
            "缓存的盈亏平衡月调用数": 0.0}
r = ft_break_even_qps(prompt_tokens=2000)
print("\n⑤ 动机是「省 prompt token」时：先算 caching 这笔账（2,000 token 说明、\\$3/百万输入）")
for k, v in r.items():
    print(f"  {k:<24} {v:>14,.4f}" if "($)" in k else f"  {k:<24} {v:>14,.0f}")
print(f"  换算成 QPS：微调要摊平 \\$20,000/月的固定成本，需要约 "
      f"{r['微调的盈亏平衡月调用数']/(30*86400):,.2f} 次/秒的持续调用")
print("  读法：**caching 是「零风险 + 立即生效」的同向手段**（命中价常为输入价的 10%–25%）；")
print("        只有当调用量足够大、且 caching 仍不够时，微调才在成本上说得通")
```

预期输出要点（实跑）：① 显存算术：全量微调 70B 合计约 **1,206 GB（至少 16 张 80 GB 卡）**，其中**优化器状态就占 1,043 GB**；LoRA 降到 **169 GB（3 卡）**；QLoRA 再降到 **47 GB（单卡可做）**；② 训练算力：同样 10 亿 token 下全量微调约 **295 GPU·小时**、LoRA 约 **147**（2× 加速，而非按参数比例的上百倍）、LoRA 8B 只要 **17** ——**LoRA 的优势在显存门槛与多租户，不在训练时长**；③ 决策打分把六类需求分开：**知识类（企业问答、私有代码库）几乎总是 RAG**，**行为类（话术统一、结构化输出、蒸馏）才是微调**，而「省 prompt token」要先试 caching；④ 多租户服务成本（50 客户）：共享基座 + LoRA 约 **\$35,040/月且与客户数基本无关**，全量微调每客户一套权重约 **\$584,000/月（16.7×）**，客户数到 200 时差距扩大到 67×；⑤ 「省 prompt token」的盈亏平衡：caching 每次省 \$0.0054、微调每次省 \$0.0060，但要摊平 \$20k/月的固定成本需要 **333 万次调用/月（约 1.29 次/秒的持续调用）**——所以**先试 caching**（零风险、立即生效）。

## 常见追问

- **追问**：LoRA 的秩 $r$ 怎么选？
  - 要点：$r$ 越大容量越强但参数与过拟合风险上升。经验起点：**风格/格式类 $r=8$–16**、**较复杂的行为迁移 $r=32$–64**；同时要选**目标模块**（通常注意力投影 + MLP 都挂效果更好，只挂 $q,v$ 最省）。**用验证集调，不要凭感觉。**
- **追问**：LoRA 能不能注入新知识？
  - 要点：**效果差且不可靠**——参数化知识**无法引用、无法按权限过滤、更新要重训**。若目标是「让模型知道 X」，用 RAG；LoRA 适合「让模型以某种方式回答」。**两者组合（检索到事实 + 适配器控制风格）是最常见的生产形态。**
- **追问**：微调后通用能力下降（灾难性遗忘）怎么办？
  - 要点：① 用 **LoRA**（冻结基座，天然缓解）；② **混入通用数据**（例如 5%–20% 的通用指令样本）；③ 降低学习率/训练轮数（1–3 epoch 通常够）；④ **用通用评测集守住**（把「通用能力不下降」设为门禁）。全量微调时这三点尤其重要。
- **追问**：怎么防止测试集泄漏？
  - 要点：训练/验证/测试**按来源切分**（不同时间段、不同客户、不同仓库），而不是随机切；对「模板化生成的数据」尤其要按模板切分；训练前做**去重与近重复检测**（相似度阈值）；并且**测试集不参与任何超参调整**。
- **追问**：客户坚持要「自己的模型」（合规/心理需求）怎么办？
  - 要点（串 [[databricks-08]]）：先**把需求翻译成可验证的目标**（质量、成本、数据边界），然后**用同一套评测做对照实验**（RAG vs LoRA vs 两者）；如果 RAG 达标而客户仍坚持微调，可以**用 LoRA 满足「专属」的感知**（每客户一个适配器）同时**保留 RAG 供知识与更新**——这是最常见的双赢解。
- **追问**：什么时候必须全量微调？
  - 要点：① **分布大偏移**（新语言、新领域、代码/数学等能力性改变）；② **蒸馏出可独立部署的小模型**（要改变的是能力而非风格）；③ 有明确的评测与充足数据，且**服务形态允许专用部署**。否则 LoRA + RAG 更划算。
- **追问**：微调的数据从哪来？
  - 要点：① **真实流量 + 人工标注**（最贵但最准）；② **强模型蒸馏**（用大模型生成/纠正，注意许可条款）；③ **合成数据 + 过滤**（需要严格的质检与去重）；④ **拒绝采样**（从多次生成中挑最好的）。**无论哪种，都要有「评测集与训练集来源隔离」。**

## 相关题目

- [[databricks-08]]：客户坚持微调而你认为 RAG 够用——本轮的决策框架在客户现场的应用。
- [[databricks-07]]：agent 从 demo 到上线的检查清单——微调/检索变更同样要走这套门禁。
- [[finetuning-08]]：prompt / RAG / fine-tuning 的决策框架（本仓库专题）。
- [[finetuning-05]]：QLoRA 的显存与量化权衡，对应本题的 QLoRA 一档。
- [[rag-01]]：RAG 的基本流程与失败模式，对应「知识走检索」一侧。

## 参考资料与归属

- **LoRA: Low-Rank Adaptation of Large Language Models（延伸）** —— Hu et al. (ICLR 2022)，2021-06-17：<https://arxiv.org/abs/2106.09685>。第 2 节「低秩适配器、可训练参数少 2–3 个数量级、冻结基座缓解遗忘」的性质来自这篇。
- **QLoRA: Efficient Finetuning of Quantized LLMs（延伸）** —— Dettmers et al. (NeurIPS 2023)，2023-05-23：<https://arxiv.org/abs/2305.14314>。第 1、2 节「4-bit 量化基座 + LoRA」的显存口径来自这篇。
- **Prompting、RAG 还是 fine-tuning：决策框架与成本权衡（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1 节「RAG 改知道什么、微调改怎么做」的框架与第 6 节的升级顺序取自该专题文档。
- **为大型商品目录设计语义搜索（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节「检索侧成本与服务形态」的对照口径取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（70B、LoRA 可训练占比 0.6%、每可训练参数 16 字节、4-bit 基座 0.5 字节/参数、激活按基座 25% 粗估、989 TFLOPs/40% MFU、1e9/10e9 token、加速 2×、50 客户/8 卡基座/\\$2 per GPU·h/730 小时、2,000 token 提示与 \\$3/百万、caching 折扣 0.1、微调固定成本 \\$20k/月）都是为演示取舍而构造的**示例参数与显式假设**；真实的显存/算力/价格强烈依赖具体框架（DeepSpeed/FSDP/Unsloth 等）、精度与序列长度，必须用自己的实测替换。**注意：文中「优化器状态 16 字节/参数」是 Adam + 混合精度的常用估算，不同实现会有差异。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
