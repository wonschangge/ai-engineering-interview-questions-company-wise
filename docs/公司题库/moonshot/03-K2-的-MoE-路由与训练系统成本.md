---
type: question
id: moonshot-03
company: Moonshot AI（Kimi）
topic: inference-serving
order: 3
question: Kimi K2 是一个 1T 参数的 MoE，每个 token 激活约 32B 参数，拥有数百个专家。请解释其路由机制以及训练它的系统成本。
question_en: Kimi K2 is a 1T-parameter MoE activating about 32B parameters per token, with hundreds of experts. Explain its routing mechanism and the system cost of training it.
asked_at: []
level: 高阶
tags: [K2, MoE 路由, 专家并行, 训练成本, all-to-all]
sources:
  - title: Kimi K2: Open Agentic Intelligence（延伸）
    url: https://arxiv.org/abs/2507.20534
    author: Kimi Team (Moonshot AI)
    published: 2025-07-28
  - title: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（延伸）
    url: https://arxiv.org/abs/2401.06066
    author: Dai et al. (DeepSeek-AI)
    published: 2024-01-11
  - title: MegaBlocks: Efficient Sparse Training with Mixture-of-Experts（延伸）
    url: https://arxiv.org/abs/2211.15841
    author: Gale et al.
    published: 2022-11-29
  - title: 用 PyTorch 实现带共享专家的 top-k MoE 路由（本仓库公司题库 · DeepSeek 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [moonshot-02, moonshot-04, moonshot-08, deepseek-04, deepseek-09]
updated: 2026-09-28
---

## 一句话答案

> **K2 的公开口径**（来自其技术报告摘要，本机已核实）：**1T 总参数、32B 激活参数、MoE 架构、MuonClip 优化器、15.5T tokens 预训练、零损失尖峰**。
> **路由机制**：**数百个路由专家 + top-k 稀疏激活 + 一个始终激活的共享专家**（共享专家的作用见 [[deepseek-01]] 的去冗余论证）：
> $$y=\underbrace{E_{\text{shared}}(x)}_{\text{权重 1}}+\sum_{i\in\text{top-}k}g_iE_i(x),\qquad
> g_i=\frac{a_i}{\sum_{j\in\text{top-}k}a_j},\quad a=\sigma(\text{router}(x))$$
> **要点**：① **sigmoid 亲和度 + 在选中的 k 个内归一化**（不是 softmax，理由见 [[deepseek-04]]：softmax 的门控梯度是稠密的）；② **负载均衡用「无辅助损失的 bias 调整」**（与 DeepSeek 同源，避免辅助损失带来的专业化损失，串 [[deepseek-02]]）；③ **共享专家不进路由**。
> **训练系统成本的三个账，按主导性排序**：
> | 账 | 按什么算 | K2 量级（本机算例） |
> | --- | --- | --- |
> | **算力** | **激活参数（32B）** | $6\times32\text{B}\times15.5\text{T}=2.98\times10^{24}$ FLOPs → **2.09M GPU·h ≈ \$4.2M**；**稠密 1T 要 31 倍** |
> | **显存** | **总参数（1T）** | 参数 bf16 2 TB + 主权重 fp32 4 TB + **Adam 状态 8 TB** + 梯度 2 TB = **16 TB**；256 卡 → **62.5 GB/卡** |
> | **通信** | **每 token 的激活专家数 × 层数** | 每 token 每 MoE 层 **112 KiB** → 58 层 **6.7 MB/token**；batch 4096、EP=32 → 每卡 0.85 GB/步（约 17 ms） |
> **核心结论（这道题的考点）**：
> $$\text{MoE 的账是「分裂」的：}\quad\underbrace{\text{算力}\propto P_{\text{active}}}_{\text{省 31×}}\quad\text{vs}\quad\underbrace{\text{显存与通信}\propto P_{\text{total}}}_{\text{一点不省}}$$
> **所以「1T 参数、32B 激活」意味着：算力按 32B 付、显存与通信按 1T 付。** 这就是 MoE 训练的系统成本本质——**它把「算力瓶颈」换成了「显存与通信瓶颈」**。
> **专家并行的耦合**（本机算例，384 专家）：
> | EP 度 | 每卡专家数 | 每卡专家显存（FP8） | 跨节点跳数 |
> | --- | --- | --- | --- |
> | 8 | 48 | **125 GB（放不下）** | 1（节点内） |
> | **32** | **12** | **31.2 GB** | 2 |
> | 128 | 3 | 7.8 GB | 3+ |
> **读法**：**EP 度越高越省每卡显存，但 all-to-all 的跨节点跳数增加**——这是「数百专家」模型的核心系统取舍。
> 一句话判据：**「1T 总参 / 32B 激活」不是「省 31 倍」，而是「算力省 31 倍、显存与通信付 1T 的价」**——把这句话说出来，这道题就答到点上了。

## 面试官在考什么

- **能否区分「总参数」与「激活参数」分别决定什么**：算力按激活、显存与通信按总参——**这是本题的第一考点**。
- **是否知道路由的三个细节**：sigmoid 亲和度、**在 top-k 内归一化**、共享专家不进路由（串 [[deepseek-04]]）。
- **负载均衡的方案**：能否说出「无辅助损失的 bias 调整」及其动机（辅助损失会造成专业化损失，串 [[deepseek-02]]）。
- **显存账的完整性**：能否列出**参数 + 主权重 + 优化器状态 + 梯度**（Adam 状态往往是最大单项），而不是只算参数。
- **通信量的公式**：能否给出「每 token 每 MoE 层 $2kdb$ 字节」并算出 6.7 MB/token 这个量级。
- **专家并行的取舍**：EP 度与每卡显存、跨节点跳数的关系（**EP 越大越省显存但通信拓扑更差**）。
- **MoE 的系统本质**：能否指出 MoE **把算力瓶颈换成了显存/通信瓶颈**——这正是它在训练侧「不是免费午餐」的原因。
- **训练稳定性的关联**：能否提到 **MuonClip 的 QK-clip**（1T 规模下 attention logits 爆炸是真实问题，串 [[moonshot-08]]）。
- **诚实**：承认具体数字（专家数、层数）**以官方报告为准**，本机的量级算例只用于说明结构。

**常见错误答案**

- 说「1T 参数但只激活 32B，所以训练很便宜」（**忽略显存与通信按 1T 付**）。
- 只算参数显存，漏掉**优化器状态**（Adam 的 m+v 在 fp32 下是参数的 8 倍字节数）。
- 用 softmax 描述 MoE 路由（**K2/DeepSeek 用 sigmoid + top-k 内归一化**）。
- 把共享专家算进 top-k 候选（破坏「始终激活」的语义）。
- 忽略 all-to-all 的通信量（**数百专家必然要 EP，EP 必然要 all-to-all**）。
- 认为 EP 度越大越好（**跨节点跳数上升，固定延迟无法摊薄**）。
- 不提训练稳定性（1T 规模的 loss spike 与 attention logits 爆炸是真实的工程问题）。

## 原理与推导

### 1. 路由机制：四个细节

**① 亲和度用 sigmoid（不是 softmax）**：

$$a_i=\sigma(s_i)\quad\text{vs}\quad a_i=\frac{e^{s_i}}{\sum_j e^{s_j}}$$

**理由**（串 [[deepseek-04]] 的实测）：softmax 的**分母把所有专家耦合起来**，导致门控梯度**稠密**（本机实测每行 **214.8/256** 个非零梯度），而 sigmoid 只有 **8.0**（恰为 top-k）——**梯度带宽差约 27 倍**。此外 sigmoid 让各专家的亲和度**独立**（不互相竞争），更符合「多个专家同时对某 token 有用」的语义。

**② 在 top-k 内归一化**：

$$g_i=\frac{a_i}{\sum_{j\in\text{top-}k}a_j}$$

**必须**——否则门控和不是 1（本机实测：softmax 下约 0.19、sigmoid 下约 6.25），**输出尺度会被系统性压缩或放大**。

**③ 共享专家不进路由**：它**始终激活、权重为 1**，是**独立的一条稠密分支**（串 [[deepseek-01]]：共享专家承担「通用知识」，让路由专家专注「专精知识」，从而去冗余）。

### 2. 算力账：按激活参数

$$C_{\text{FLOPs}}=6\,P_{\text{active}}\,T_{\text{tokens}}$$

**本机算例**（$P_{\text{active}}=32$B、$T=15.5$T）：

$$6\times32\times10^9\times15.5\times10^{12}=2.98\times10^{24}\ \text{FLOPs}$$

40% MFU 下 **2.09M GPU·h ≈ \$4.2M**（按 \$2/GPU·h）。**对照：稠密 1T 需要 31 倍**——这就是 MoE 的算力优势。

### 3. 显存账：按总参数（MoE 一点不省）

| 项 | 公式 | K2 量级 |
| --- | --- | --- |
| 参数（bf16） | $2P$ | 2 TB |
| **主权重（fp32）** | $4P$ | 4 TB |
| **Adam 状态（m+v，fp32）** | $8P$ | **8 TB** |
| 梯度（bf16） | $2P$ | 2 TB |
| **合计** | $16P$ | **16 TB** |

**关键**：**优化器状态（8 TB）是最大单项**——这就是为什么大规模训练必须做**优化器状态分片**（ZeRO 系列）。分片后：

| 卡数 | 每卡显存 | 能否放入 80 GB |
| --- | --- | --- |
| 128 | 125 GB | ✗ |
| **256** | **62.5 GB** | ✓（含激活与缓冲余量有限） |
| 512 | 31.2 GB | ✓ |
| 1024 | 15.6 GB | ✓ |

**读法**：**稠密 1T 也是这 16 TB**——**MoE 在显存上没有任何优势**。这正是「算力省 31 倍」的代价所在。

### 4. 通信账：all-to-all

每 token 每 MoE 层要把隐藏状态发给 $k$ 个专家并收回结果：

$$\text{bytes/token/layer}\approx2\,k\,d\,b$$

**本机算例**（$k=8$、$d=7168$、FP8 $b=1$）：$2\times8\times7168=112$ KiB/层 → **58 层 = 6.7 MB/token**。

| batch | 全系统流量 | EP=32 每卡 | IB 传输时间（50 GB/s） |
| --- | --- | --- | --- |
| 512 | 3.41 GB | 0.11 GB | 2.1 ms |
| **4096** | **27.25 GB** | **0.85 GB** | **17.0 ms** |
| 32768 | 217.97 GB | 6.81 GB | 136.2 ms |

**读法**：**通信量随 batch 线性上升**（不像访存那样可摊薄），所以**大 batch 不会「免费」**——这正是为什么 MoE 训练必须做**计算-通信重叠**（串 [[deepseek-09]]）与**分层 all-to-all**。

### 5. 专家并行的耦合：EP 度是核心旋钮

| EP 度 | 每卡专家数 | 每卡专家显存（FP8） | 跨节点跳数 |
| --- | --- | --- | --- |
| 8 | 48 | **125 GB（放不下）** | 1（节点内） |
| **32** | **12** | **31.2 GB** | 2 |
| 128 | 3 | 7.8 GB | 3+ |
| 384 | 1 | 2.6 GB | 3+ |

**读法**：**EP 度决定「每卡放多少专家」**——384 个专家、1T 总参下，**每卡专家显存 = (384/EP) × 2.6 GB**。EP 小则显存放不下，EP 大则**跨节点跳数上升、固定延迟无法摊薄**（串 [[moonshot-01]] 的同一机制：小 batch 下固定延迟不可隐藏）。**所以 EP 度是被「显存下限」与「通信上限」夹出来的。**

### 6. 训练稳定性：MuonClip 与 QK-clip

**K2 报告的公开要点**：**15.5T tokens 预训练、零损失尖峰**，靠的是 **MuonClip**——在 **Muon**（一种利用梯度矩阵正交化的优化器，token 效率更高）之上加 **QK-clip**：

$$\text{若 } \max|q^\top k|>\tau\ \Rightarrow\ \text{把 }W_Q,W_K\text{ 按比例缩小}$$

**为什么需要**：**attention logits 爆炸**会让 softmax 进入饱和区 → 梯度消失/数值溢出 → 训练失稳（串 [[moonshot-08]] 的专题）。**QK-clip 是一种「直接约束病态量」的手法**——比调小学习率更精准。

## 数值与代码验证

### 表 1：三个账的量化（见代码输出）

| 账 | 主导量 | K2 量级 |
| --- | --- | --- |
| 见输出 | 见输出 | 见输出 |

### 表 2：专家并行的显存/通信耦合

| EP 度 | 每卡专家数 | 每卡专家显存 | 跨节点跳数 |
| --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# K2 量级 MoE 的训练系统成本：算力（按激活参）、显存（按总参）、通信（all-to-all）与 EP 耦合
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

GB, TB = 1e9, 1e12
PEAK, MFU = 989e12, 0.40        # H100 SXM5、40% MFU（仓库统一常数）
HBM, IB = 3.35e12, 50e9         # HBM 带宽、每卡跨节点有效带宽（示例）

@dataclass
class K2:
    """K2 量级配置（总参/激活参/tokens 来自公开报告摘要）"""
    total: float = 1.0e12
    active: float = 32e9
    tokens: float = 15.5e12
    layers: int = 61
    moe_layers: int = 58
    d_model: int = 7168
    top_k: int = 8
    experts: int = 384            # 数百个（以官方报告为准）
    gpu_hour_usd: float = 2.0

K = K2()
print("① 算力账：按【激活参数】算 —— MoE 的核心优势")
flops = 6 * K.active * K.tokens
gpu_hours = flops / (PEAK * MFU) / 3600
print(f"  6 × {K.active/1e9:.0f}B × {K.tokens/1e12:.1f}T = {flops:.3e} FLOPs")
print(f"  {MFU:.0%} MFU 下 = **{gpu_hours/1e6:.2f}M GPU·h** ≈ ${gpu_hours*K.gpu_hour_usd/1e6:.1f}M")
dense = 6 * K.total * K.tokens
print(f"  对照：若为稠密 1T = {dense:.3e} -> **MoE 省 {dense/flops:.0f}x 算力**")

print("")
print("② 显存账：按【总参数】算 —— MoE 一点不省")
rows = [("参数 bf16", 2 * K.total), ("主权重 fp32", 4 * K.total),
        ("Adam 状态 m+v (fp32)", 8 * K.total), ("梯度 bf16", 2 * K.total)]
total_mem = sum(v for _, v in rows)
for name, v in rows:
    print(f"  {name:<24} {v/TB:>6.1f} TB")
print(f"  {'合计':<24} {total_mem/TB:>6.1f} TB（不含激活）—— **稠密 1T 也是这个数**")
print(f"  {'分片后的每卡显存':<24}")
print(f"  {'卡数':>6} {'每卡':>10} 结论")
for g in (128, 256, 512, 1024):
    per = total_mem / g
    note = "放不下（>80 GB）" if per > 80 * GB else "可放入 80 GB 卡"
    print(f"  {g:>6} {per/GB:>8.1f}G {note}")
print("  读法：**优化器状态（8 TB）是最大单项** —— 所以大规模训练必须做优化器状态分片（ZeRO 系列）；")
print("        **MoE 在显存上没有优势**，这正是「算力省 31x」的代价")

print("")
print("③ 通信账：all-to-all（每 token 每 MoE 层 2·k·d·bytes）")
per_layer = 2 * K.top_k * K.d_model * 1        # FP8 = 1 字节
print(f"  每 token 每层 {per_layer/1024:.1f} KiB -> {K.moe_layers} 层 = "
      f"{per_layer*K.moe_layers/1e6:.1f} MB/token")
print(f"  {'batch':>7} {'全系统流量':>12} {'EP=32 每卡':>12} {'IB 时间':>10}")
for b in (512, 4096, 32768):
    total = b * per_layer * K.moe_layers
    per_gpu = total / 32
    print(f"  {b:>7} {total/GB:>10.2f}G {per_gpu/GB:>10.2f}G {per_gpu/IB*1e3:>9.1f}ms")
print("  读法：**通信量随 batch 线性上升，不像访存那样可摊薄** —— 所以大 batch 不「免费」，")
print("        必须靠计算-通信重叠与分层 all-to-all 来隐藏")

print("")
print(f"④ 专家并行的耦合：EP 度是核心旋钮（{K.experts} 个专家）")
per_exp = K.total / K.experts
print(f"  {'EP 度':>6} {'每卡专家数':>10} {'每卡专家显存(FP8)':>18} {'跨节点跳数':>11}")
for ep in (8, 32, 128, 384):
    ne = K.experts / ep
    hops = "1（节点内）" if ep <= 8 else ("2" if ep <= 64 else "3+")
    print(f"  {ep:>6} {ne:>10.0f} {ne*per_exp/GB:>16.1f}G {hops:>11}")
print("  读法：**EP 越大越省每卡显存，但跨节点跳数上升、固定延迟越难摊薄** ——")
print("        EP 度是被「显存下限」与「通信上限」夹出来的")

print("")
print("⑤ 路由机制的四个细节（与实现对照）")
@dataclass
class RouteDetail:
    item: str
    correct: str
    why: str
DETAILS = [
    RouteDetail("亲和度函数", "sigmoid（不是 softmax）",
                "softmax 的分母耦合所有专家 -> 门控梯度稠密（实测 214.8/256 vs 8.0）"),
    RouteDetail("归一化位置", "在 top-k 内重新归一化",
                "否则门控和不是 1（softmax 下约 0.19、sigmoid 下约 6.25）"),
    RouteDetail("共享专家", "不进路由候选、权重恒为 1",
                "否则它会占掉 top-k 名额（实测高分数时 100% 的 token 被占）"),
    RouteDetail("负载均衡", "无辅助损失的 bias 调整",
                "辅助损失会带来专业化损失（实测约 30%），bias 只改「选谁」不改输出权重"),
]
print(f"  {'细节':<14} {'正确做法':<26} 理由")
for d in DETAILS:
    print(f"  {d.item:<14} {d.correct:<26} {d.why}")
print("  读法：**这四条都是「看起来对但会出问题」的地方** —— 具体实测见 deepseek-02/04")
```

预期输出要点（实跑）：① **算力按激活参数**：$6\times32\text{B}\times15.5\text{T}=2.98\times10^{24}$ FLOPs → **2.09M GPU·h ≈ \$4.2M**，**稠密 1T 要 31 倍**；② **显存按总参数**：参数 2 TB + 主权重 4 TB + **Adam 8 TB** + 梯度 2 TB = **16 TB**，**稠密 1T 也是这个数**（128 卡放不下、**256 卡 62.5 GB/卡**）；③ **通信**：每 token 每 MoE 层 112 KiB → 58 层 **6.7 MB/token**，batch 4096、EP=32 时每卡 0.85 GB（约 **17 ms**），且**随 batch 线性上升**；④ **EP 耦合**：384 专家下 EP=8 时每卡 48 个专家要 **125 GB（放不下）**，EP=32 时 12 个专家 31.2 GB（2 跳），EP≥128 要 3+ 跳；⑤ 路由的四个细节与实测依据。

## 常见追问

- **追问**：为什么说 MoE 把「算力瓶颈」换成了「显存与通信瓶颈」？
  - 要点：**因为两本账的计价单位不同**——**算力 $\propto P_{\text{active}}$（32B）**，而**显存与通信 $\propto P_{\text{total}}$（1T）**。本机算例：算力省 31 倍，但显存 16 TB 一点不省、每 token 通信 6.7 MB 全靠 EP + 重叠来扛。**所以 MoE 的收益是「有条件的」**：只有在**算力是瓶颈、而显存与网络能跟上**的场景才划算（串 [[moonshot-01]] 的同一框架）。
- **追问**：384 个专家为什么需要专家并行？
  - 要点：**算一下就知道**：1T/384 ≈ 2.6B 参数/专家，FP8 下 2.6 GB；**EP=8 时每卡 48 个专家要 125 GB，放不下 80 GB 卡**。所以**必须把专家分散到更多卡上**（EP=32 → 31.2 GB/卡）。**代价是 all-to-all 要跨节点**（2 跳），固定延迟与带宽都变差。
- **追问**：共享专家到底解决什么问题？
  - 要点：**去冗余**（串 [[deepseek-01]]）。若没有共享专家，**每个路由专家都要重复学「通用知识」**，导致参数浪费与专业化不足。本机在 [[deepseek-01]] 的算例显示：8 专家时共享专家可省 **54.7%** 的「通用知识」参数，256 专家时省 **62.3%**。**同时它让路由专家专注专精**，所以是「结构性的质量提升 + 参数节省」。
- **追问**：负载均衡为什么不用辅助损失？
  - 要点：**辅助损失会扭曲路由偏好**（把 token 推向「均衡」而非「最合适」的专家）。本机在 [[deepseek-02]] 的算例：辅助损失式均衡带来约 **30% 的专业化损失**，而且**不可恢复**（永久改变了专家分工）。**bias 调整只改「选谁」、不改输出权重**，所以能在不扭曲专家的前提下均衡（实测把最大/平均负载从 8.75× 压到 1.41×）。
- **追问**：训练 1T 模型最容易在什么地方翻车？
  - 要点：① **显存**（16 TB 的分片与重计算策略）；② **通信**（all-to-all 的固定延迟 + 带宽，串 [[deepseek-09]]）；③ **数值稳定性**（attention logits 爆炸 → 需要 QK-clip 这类手段，串 [[moonshot-08]]）；④ **负载不均**（热点专家导致某些卡成为瓶颈）；⑤ **数据与评测污染**（串 [[moonshot-10]] 的 agentic 评估）。**这五条都是「工程性」的，而不是「算法性」的**——这正是大模型训练的现实。
- **追问**：15.5T tokens 与「零损失尖峰」意味着什么？
  - 要点：① **数据量**：15.5T 是当前前沿量级（Chinchilla 最优对 32B 激活而言远小于此，说明**推理侧收益驱动了过度训练**）；② **零尖峰**：说明**优化器与数值稳定性的工程做到位了**（MuonClip 的 QK-clip 是直接手段）。**「零尖峰」是一个很强的工程声明**——因为规模越大，偶发尖峰越难避免（本机在 [[deepseek-07]] 的算例：尖峰概率 1%/步就会吃掉一半训练进度）。

## 相关题目

- [[moonshot-02]]：MLA 与 GQA 的 KV 对比——K2 的注意力侧选择。
- [[moonshot-04]]：把 8K–32K 扩到 128K+——K2 的长上下文能力从哪来。
- [[moonshot-08]]：attention logits 爆炸与 MuonClip——本篇训练稳定性的专题展开。
- [[deepseek-04]]：MoE 路由的实现与陷阱——本篇路由细节的实测依据。
- [[deepseek-09]]：DualPipe 的计算-通信重叠——本篇通信账的解法。

## 参考资料与归属

- **Kimi K2: Open Agentic Intelligence（延伸）** —— Kimi Team (Moonshot AI)，2025-07-28：<https://arxiv.org/abs/2507.20534>。**"1T 总参、32B 激活、15.5T tokens、MuonClip（Muon + QK-clip）、零损失尖峰"**这些事实来自该报告摘要（本机通过 arXiv API 核实）；benchmark 数字（τ²-Bench 66.1、SWE-Bench Verified 65.8 等）同样来自摘要，**本仓库未复现**。
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（延伸）** —— Dai et al. (DeepSeek-AI)，2024-01-11：<https://arxiv.org/abs/2401.06066>。第 1 节共享专家的「去冗余 + 专业化」论证来自这篇。
- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts（延伸）** —— Gale et al.，2022-11-29：<https://arxiv.org/abs/2211.15841>。第 4 节的稀疏专家计算（不做 token dropping、按块做 GEMM）的工程思路来自这篇。
- **用 PyTorch 实现带共享专家的 top-k MoE 路由（本仓库公司题库 · DeepSeek 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1 节的 sigmoid/归一化/共享专家三个细节与其实测数字直接来自该篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$P_{\text{total}}=1$T、$P_{\text{active}}=32$B、$T=15.5$T、61 层、58 个 MoE 层、$d=7168$、top-8、**384 个专家**、MFU 40%、H100 989 TFLOPs 与 3.35 TB/s、IB 50 GB/s、\$2/GPU·h、Adam fp32 状态、bf16 参数与梯度、FP8 通信）都是**示例参数与显式假设**；其中**总参/激活参/tokens 来自公开报告摘要**，而**专家数（384）、层数与 $d_{\text{model}}$ 未在摘要中出现**——**应以官方技术报告为准**。此外「算力省 31 倍」是**按 6ND 公式的理论值**，实际受 MFU、通信开销与负载不均影响，**不等于端到端加速 31 倍**（串 [[deepseek-09]] 的「同等资源约束下比较」原则）。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
