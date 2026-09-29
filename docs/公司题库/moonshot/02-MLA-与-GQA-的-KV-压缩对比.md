---
type: question
id: moonshot-02
company: Moonshot AI（Kimi）
topic: llm-internals
order: 2
question: Kimi K2 使用了 Multi-head Latent Attention (MLA)。请解释它的作用，以及相比 GQA，它在减少 KV cache 方面表现如何。
question_en: Kimi K2 uses Multi-head Latent Attention (MLA). Explain what it does, and how it compares with GQA at reducing the KV cache.
asked_at: []
level: 高阶
tags: [MLA, GQA, MQA, KV cache, 权重吸收]
sources:
  - title: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（延伸）
    url: https://arxiv.org/abs/2405.04434
    author: DeepSeek-AI
    published: 2024-05-07
  - title: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints（延伸）
    url: https://arxiv.org/abs/2305.13245
    author: Ainslie et al. (Google)
    published: 2023-05-22
  - title: Fast Transformer Decoding: One Write-Head is All You Need（延伸）
    url: https://arxiv.org/abs/1911.02150
    author: Shazeer (Google)
    published: 2019-11-06
  - title: 上下文从 8K 推到几十万时最先崩什么（本仓库公司题库 · Moonshot AI 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [moonshot-01, moonshot-03, deepseek-05, inference-serving-08, llm-internals-05]
updated: 2026-09-28
---

## 一句话答案

> **MLA 的作用**：不缓存完整的 K/V，而是把每个 token 的 K/V **压成一个低秩潜向量** $c_t\in\mathbb{R}^{d_c}$（外加一个小的 RoPE 分量），
> $$c_t=W^{DKV}h_t,\qquad k_t=W^{UK}c_t,\quad v_t=W^{UV}c_t$$
> **推理时可以把 $W^{UK}$ 吸收进 $W^Q$、把 $W^{UV}$ 吸收进输出投影**——于是**根本不需要物化完整的 K/V**，注意力直接在潜空间里算。
> **KV cache 的对比（同一配置：64 个 query 头、$d_{\text{head}}=128$、同精度、同层数）**：
> | 方案 | 每层每 token 的 KV 元素 | 相对 MHA | **相对 GQA-8** | 注意力每对算力 |
> | --- | --- | --- | --- | --- |
> | MHA | 16384 | 1× | 8× | 1× |
> | **GQA-8** | 2048 | 1/8 | 1× | 1× |
> | MQA | 256 | 1/64 | **0.125×** | 1× |
> | **MLA**（$d_c{+}d_{\text{rope}}=576$） | **576** | **1/28.4** | **0.281×（即 3.6× 更小）** | 朴素 1× / **吸收 4.5×** |
> **所以对这道题的关键回答是**：
> - **相对 MHA，MLA 的 KV 小约 28 倍**——这是「长上下文可行」的量级；
> - **相对 GQA-8，MLA 只小约 3.6 倍**（本机算例：80 层下 **90 KiB/token vs 320 KiB/token**）——**不是数量级差异**；
> - **而且 MQA 的 KV 比 MLA 还小 2.25 倍**——但 MQA 会让所有头共享同一份 K/V，**质量损失明显**；MLA 的潜向量是**学习出来的**，论文报告的质量**不低于甚至优于 MHA**。
> **MLA 的代价（必须一起讲）**：
> ① **吸收形式把注意力每对算力放大 4.5 倍**（潜空间维度 576 vs $d_{\text{head}}$ 128）——本机算例：128K 预填充的注意力算力从 **34.3 → 154.5 PFLOPs**；
> ② **所以真实工程是「两种形式混用」**：**解码（访存受限）用吸收形式**（省显存带宽）、**预填充（算力受限）用朴素形式**（物化 K/V，算力与 MHA 相同 + 可忽略的投影开销）；
> ③ **实现复杂度**：权重吸收会改变计算图，需要专门的 kernel；训练时低秩瓶颈也要学。
> 一句话判据：**"MLA 比 GQA 省 3.6 倍 KV、比 MHA 省 28 倍；它真正的价值在于'用小得多的 KV 换来不低于 MHA 的质量'，而不是'比 GQA 省一个数量级'"**——把这两个倍数说清楚，这道题就答对了。

## 面试官在考什么

- **是否知道 MLA 的机制**：低秩潜向量 $c_t$ + 解耦的 RoPE 分量 + **权重吸收**（这是它区别于 GQA 的本质）。
- **能否给出正确的对比基准**：**对 MHA 是约 28 倍，对 GQA-8 只有约 3.6 倍**——**把 GQA 当成基准时夸大倍数是最常见的错误**。
- **是否知道 MQA 更小**：MQA 的 KV 只有 MLA 的 1/2.25，但**质量代价大**；MLA 的关键是「**压缩但不掉质量**」。
- **能否指出吸收形式的算力代价**（4.5× 注意力算力）以及**解码/预填充用不同形式**这一工程实践。
- **是否理解 GQA 的优势**：**零额外算力、与现有 kernel 完全兼容、实现极简**——所以它仍是主流选择。
- **能否把 KV 与「长上下文可行性」挂钩**：解码是访存受限的，**KV 读取时间直接进单步延迟**（本机算例：128K 下 MHA 读 KV 要 **641 µs**，MLA **23 µs**）。
- **能否量化到「每 token 多少字节」**：给出 90 KiB vs 320 KiB 这类可直接用于显存规划的数字。
- **诚实**：MLA 不是「免费的压缩」——它有算力与实现代价，且**相对 GQA 的收益没有想象中大**。

**常见错误答案**

- 把 MLA 说成「另一种 GQA」（忽略**低秩潜向量 + 权重吸收**这两个本质区别）。
- 声称 MLA 比 GQA 省一个数量级（**实际约 3.6 倍**）。
- 认为 MQA/GQA/MLA 会减少**注意力算力**（GQA/MQA 的算力与 MHA 相同；MLA 吸收形式反而更大）。
- 不提 RoPE 的解耦分量（MLA 必须把位置信息从潜向量里分出来，否则压缩会破坏位置编码）。
- 不提吸收形式的算力代价。
- 认为「KV 越小越好」（MQA 更小但质量差；**质量与压缩要一起看**）。
- 忽略「解码访存受限、预填充算力受限」这个分界（它决定了用哪种形式）。

## 原理与推导

### 1. MLA 的机制：低秩潜向量 + 解耦 RoPE

**MHA/GQA/MQA 的压缩方式是「减少头的数量」**：

$$\text{KV}=2\times n_{\text{kv}}\times d_{\text{head}}\quad(\text{MHA}:n_{\text{kv}}=n_h;\ \text{GQA}:n_{\text{kv}}<n_h;\ \text{MQA}:n_{\text{kv}}=1)$$

**MLA 换了思路：不压缩头数，而是压缩「每个头的表示」**：

$$c_t=W^{DKV}h_t\in\mathbb{R}^{d_c},\qquad
k_t^{C}=W^{UK}c_t,\qquad v_t=W^{UV}c_t,\qquad
k_t^{R}=\text{RoPE}(W^{KR}h_t)$$

**关键点**：**只缓存 $c_t$ 与 $k_t^{R}$**（$d_c+d_{\text{rope}}$ 个元素），而 $k^C$、$v$ 都可以**从 $c$ 现算**。

**为什么 RoPE 要解耦**：RoPE 是**位置相关**的旋转，如果把它作用在会被压缩的 $k^C$ 上，压缩与位置就耦合在一起（同一个 $c$ 在不同位置需要不同的 K）。所以 MLA 把位置信息分到一个**单独的小分量** $k^R$（$d_{\text{rope}}=64$）上，$c$ 只承载**内容信息**。

### 2. 权重吸收：为什么可以「不物化 K/V」

$$q_t^\top k_s=\big(W^Qh_t\big)^\top\big(W^{UK}c_s\big)=\big(\underbrace{W^{UK\top}W^Q}_{W^{Q,\text{abs}}}h_t\big)^\top c_s$$

**读法**：把 $W^{UK}$ **预先乘进 $W^Q$**，查询就在**潜空间**（$d_c$ 维）里和 $c_s$ 做点积——**不需要把 $c_s$ 还原成 $k_s$**。同理 $W^{UV}$ 可以吸收进输出投影，$v$ 也不必物化。

**收益**：① **KV cache 只存潜向量**（省显存）；② **解码时读取的 KV 更少**（省带宽，这是解码的瓶颈）。

### 3. 四种方案的定量对比（同精度、同层数）

**配置**：$n_h=64$、$d_{\text{head}}=128$、MLA 的 $d_c=512$、$d_{\text{rope}}=64$。

| 方案 | KV 元素/层/token | bf16 字节 | 相对 MHA | 相对 GQA-8 |
| --- | --- | --- | --- | --- |
| MHA | 16384 | 32.0 KiB | 1× | 8× |
| GQA-8 | 2048 | 4.0 KiB | 1/8 | 1× |
| MQA | 256 | 0.5 KiB | 1/64 | 0.125× |
| **MLA** | **576** | **1.125 KiB** | **1/28.4** | **0.281×** |

**整模型**（bf16）：

| 方案 | 61 层（V3 量级） | 80 层（70B 量级） |
| --- | --- | --- |
| MHA | 1952 KiB/token | 2560 KiB/token |
| **GQA-8** | 244 KiB/token | **320 KiB/token**（仓库统一常数） |
| MQA | 30.5 KiB/token | 40 KiB/token |
| **MLA** | **68.6 KiB/token** | **90 KiB/token** |

**读法**：**MLA 对 MHA 是 28 倍、对 GQA-8 是 3.6 倍**——而 MQA 比 MLA 还小。**「MLA 比 GQA 省一个数量级」是不成立的**（这正是这道题的考点）。

### 4. 算力代价：吸收形式的 4.5×

吸收后每对点积的维度是 $d_c+d_{\text{rope}}=576$，而 MHA/GQA 是 $d_{\text{head}}=128$：

$$\frac{576}{128}=4.5\times$$

**本机算例**（128K 预填充、61 层）：注意力算力 **34.3 PFLOPs（朴素）→ 154.5 PFLOPs（吸收）**。

**所以真实实现是两种形式混用**：

| 阶段 | 瓶颈 | 用哪种形式 | 理由 |
| --- | --- | --- | --- |
| **解码** | 访存 | **吸收** | 读取的 KV 更少（28× vs MHA），算力不是瓶颈 |
| **预填充** | 算力 | **朴素（物化 K/V）** | 算力与 MHA 相同（+ 可忽略的投影），避免 4.5× 放大 |

**读法**：**「MLA 省显存」与「MLA 费算力」是同一枚硬币的两面**——用哪种形式取决于**哪个资源是瓶颈**（串 [[moonshot-01]] 的同一分析框架）。

### 5. 与 GQA 的取舍（为什么 GQA 仍是主流）

| 维度 | GQA / MQA | MLA |
| --- | --- | --- |
| KV 压缩 | 8× / 64×（相对 MHA） | **28×** |
| 注意力算力 | **与 MHA 相同** | 朴素相同 / **吸收 4.5×** |
| 实现复杂度 | **低**（改头数即可，现有 kernel 直接支持） | **高**（权重吸收、解耦 RoPE、专用 kernel） |
| 质量 | GQA 接近 MHA；MQA 有损失 | 论文报告**不低于甚至优于 MHA** |
| 训练改动 | 小 | 中（低秩瓶颈与解耦 RoPE 都要学） |
| 与 PagedAttention 等 | 天然兼容 | 需要适配（潜向量是每 token 一个，分页更简单） |

**读法**：**GQA 的性价比来自「零算力代价 + 零实现风险」**；MLA 的价值来自「**在极小 KV 下不掉质量**」。**两者不是替代关系**——预算够、实现能力强、且长上下文是核心卖点时选 MLA；否则 GQA 是稳妥选择。

## 数值与代码验证

### 表 1：四种方案的 KV 与算力对比（见代码输出）

| 方案 | KV 元素/层/token | 相对 MHA | 相对 GQA-8 | 注意力算力 |
| --- | --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：解码 KV 读取时间（128K、FP8、61 层）

| 方案 | KV 总量 | 读取时间 |
| --- | --- | --- |
| 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# MLA vs GQA/MQA/MHA：KV cache 与注意力算力的同精度、同层数对比
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

HBM_BW = 3.35e12          # H100 SXM5（仓库统一常数）
N_HEADS, D_HEAD = 64, 128
D_C, D_ROPE = 512, 64     # MLA：潜向量 + 解耦 RoPE 分量

@dataclass
class Attn:
    name: str
    kv_per_layer: int          # 每层每 token 的 KV 元素数
    attn_cost_factor: float    # 注意力"每对"算力相对 MHA 的倍数
    note: str
VARIANTS = [
    Attn("MHA", 2 * N_HEADS * D_HEAD, 1.0, "完整 K/V，基线"),
    Attn("GQA-8", 2 * 8 * D_HEAD, 1.0, "8 个 KV 头共享给 64 个 query 头"),
    Attn("MQA", 2 * 1 * D_HEAD, 1.0, "所有头共享 1 份 K/V（质量代价大）"),
    Attn("MLA(朴素)", D_C + D_ROPE, 1.0, "只缓存潜向量，预填充时物化 K/V"),
    Attn("MLA(吸收)", D_C + D_ROPE, (D_C + D_ROPE) / D_HEAD, "权重吸收进 W_Q，潜空间做点积"),
]
BASE_KV = 2 * N_HEADS * D_HEAD
print("① 每层每 token 的 KV 与注意力算力（64 个 query 头、d_head=128）")
print(f"  {'方案':<12} {'KV 元素/层':>11} {'bf16':>9} {'相对 MHA':>9} {'相对 GQA-8':>11} "
      f"{'注意力算力':>10}")
for v in VARIANTS:
    print(f"  {v.name:<12} {v.kv_per_layer:>11} {v.kv_per_layer * 2 / 1024:>8.1f}K "
          f"{v.kv_per_layer / BASE_KV:>9.3f} {v.kv_per_layer / (2 * 8 * D_HEAD):>11.3f} "
          f"{v.attn_cost_factor:>9.2f}x")
print("  读法：**MLA 对 MHA 是 28.4x 更小，但对 GQA-8 只有 3.6x** —— 而 MQA 比 MLA 还小 2.25x；")
print("        所以「MLA 比 GQA 省一个数量级」是不成立的（这正是本题考点）")

print("")
print("② 整模型 KV/token（bf16、两种深度）")
print(f"  {'方案':<12} {'61 层(V3 量级)':>15} {'80 层(70B 量级)':>16}")
for v in VARIANTS:
    print(f"  {v.name:<12} {v.kv_per_layer * 61 * 2 / 1024:>13.1f}K "
          f"{v.kv_per_layer * 80 * 2 / 1024:>14.1f}K")
print("  读法：**GQA-8 在 80 层下正是 320 KiB/token（仓库统一常数）**；MLA 是 90 KiB/token ——")
print("        这个量级可以直接用于显存规划（串 moonshot-01 的并发账）")

print("")
print("③ 解码一步的 KV 读取时间（128K 上下文、FP8、61 层、3.35 TB/s）")
print(f"  {'方案':<12} {'KV 总量':>10} {'读取时间':>10} {'相对 MLA':>10}")
mla_us = None
for v in VARIANTS:
    total = 131072 * v.kv_per_layer * 1        # FP8 = 1 字节
    t = total / HBM_BW
    if v.name == "MLA(朴素)":
        mla_us = t * 1e6
    ratio = "—" if mla_us is None else f"{t * 1e6 / mla_us:.1f}x"
    print(f"  {v.name:<12} {total / 1e6:>8.1f}MB {t * 1e6:>9.0f}us {ratio:>10}")
print("  读法：**解码访存受限，KV 读取时间直接进单步延迟** —— MHA 在 128K 下单是读 KV 就要 641us，")
print("        MLA 只要 23us；这就是 MLA 让长上下文「可服务」的直接原因")

print("")
print("④ 预填充的注意力算力（128K、61 层）")
base_flops = 4 * 131072 * 131072 * N_HEADS * D_HEAD * 61
print(f"  {'方案':<12} {'注意力算力':>12} {'相对 MHA':>10}")
for v in VARIANTS:
    print(f"  {v.name:<12} {base_flops * v.attn_cost_factor / 1e15:>10.1f}P "
          f"{v.attn_cost_factor:>9.2f}x")
print("  读法：**吸收形式把每对算力放大 4.5x**（576 vs 128）——所以真实工程混用两种形式：")
print("        解码用吸收（省带宽）、预填充用朴素（省算力）；这是「省显存」与「费算力」的同一枚硬币")

print("")
print("⑤ 取舍对比")
@dataclass
class Tradeoff:
    dim: str
    gqa: str
    mla: str
ROWS = [
    Tradeoff("KV 压缩（相对 MHA）", "8x（GQA-8）/ 64x（MQA）", "28x"),
    Tradeoff("注意力算力", "与 MHA 相同", "朴素相同 / 吸收 4.5x"),
    Tradeoff("实现复杂度", "低（改头数，kernel 直接支持）", "高（权重吸收 + 解耦 RoPE + 专用 kernel）"),
    Tradeoff("质量", "GQA 接近 MHA；MQA 有损失", "论文报告不低于甚至优于 MHA"),
    Tradeoff("训练改动", "小", "中（低秩瓶颈与 RoPE 分量都要学）"),
    Tradeoff("长上下文服务", "可行，但 KV 仍是主要压力", "**KV 压力最小且不掉质量**"),
]
print(f"  {'维度':<22} {'GQA / MQA':<34} MLA")
for r in ROWS:
    print(f"  {r.dim:<22} {r.gqa:<34} {r.mla}")
print("  读法：**GQA 的性价比来自「零算力代价 + 零实现风险」；MLA 的价值来自「极小 KV 下不掉质量」** ——")
print("        两者不是替代关系，选哪个取决于长上下文是不是核心卖点、以及团队的 kernel 能力")
```

预期输出要点（实跑）：① **MLA 的 KV 是 MHA 的 1/28.4，但只是 GQA-8 的 0.281 倍（3.6× 更小）**，而 **MQA 比 MLA 还小 2.25 倍**；② 整模型 bf16 下 **GQA-8 在 80 层正好是 320 KiB/token（仓库统一常数），MLA 是 90 KiB/token**；③ 解码读 KV（128K、FP8、61 层）**MHA 641 µs、GQA-8 80 µs、MLA 23 µs、MQA 10 µs**——这就是 MLA 让长上下文可服务的直接原因；④ **吸收形式把注意力每对算力放大 4.5×**（128K 预填充 34.3 → **154.5 PFLOPs**），所以真实工程**解码用吸收、预填充用朴素**；⑤ 取舍表说明 **GQA 的性价比来自零算力代价与零实现风险，MLA 的价值来自极小 KV 下不掉质量**。

## 常见追问

- **追问**：MLA 为什么要把 RoPE 解耦出来？
  - 要点：**RoPE 是位置相关的旋转**。如果把它作用在会被压缩的 $k^C$ 上，那么**同一个潜向量在不同位置需要对应不同的 K**——压缩与位置就耦合了，$c$ 无法复用。MLA 因此把位置信息分到一个**单独的小分量** $k^R$（64 维）上，$c$ 只承载**内容**。**代价是每 token 多存 64 个元素**（占 MLA KV 的 11%），但换来了可压缩性。
- **追问**：MLA 相比 GQA 到底值不值？
  - 要点：**看三个条件**：① **长上下文是不是核心卖点**（KV 是主要瓶颈 → 值）；② **团队有没有 kernel 能力**（权重吸收需要专门实现 → 没有就不值）；③ **是否在意那 3.6 倍**（如果 GQA 已经够用，额外的复杂度不划算）。**本机算例的 3.6 倍不是数量级差异**——所以「值不值」要具体算，不能一概而论。
- **追问**：MQA 的 KV 比 MLA 还小，为什么不用 MQA？
  - 要点：**因为质量**。MQA 让**所有头共享同一份 K/V**，表达能力被严重压缩（论文里通常有明显质量损失）；GQA 是「折中」（8 个头），MLA 是「**用低秩压缩代替头共享**」——**压缩发生在「每个 token 的表示」层面，而不是「头」层面**，所以能保住质量。**这也是 MLA 论文报告质量不低于 MHA 的原因。**
- **追问**：KV 量化与 MLA 能叠加吗？
  - 要点：**能，而且很自然**——MLA 的潜向量是**每 token 每层一个向量**（不是每头一份），**量化与分页都更简单**（粒度统一）。叠加 KV 量化（FP8/INT8）可再省约 2×，但要注意**量化误差会直接影响注意力分数**，需要实测质量。**注意别把 MLA 的 28× 与量化的 2× 直接相乘去宣传**——两者作用在不同层面，且都有质量代价。
- **追问**：MLA 的训练要注意什么？
  - 要点：① **低秩瓶颈的维度** $d_c$ 是超参（太小会掉质量，太大省不了 KV）；② **解耦 RoPE 的分量** $d_{\text{rope}}$ 要够（否则位置信息表达不足）；③ **权重吸收只在推理时做**（训练时保持朴素形式，否则反向传播会变复杂）；④ **初始化与归一化**（低秩投影的缩放要调，否则训练不稳）。**这些细节正是「实现复杂度高」的具体含义。**
- **追问**：MLA 与 PagedAttention 的关系？
  - 要点：**互补**。PagedAttention 解决**显存碎片与共享**（分页管理 KV），MLA 解决**每 token KV 的绝对大小**。**MLA 让「每 token 的 KV 更小」，PagedAttention 让「这些 KV 被更高效地管理」**——两者叠加才让长上下文高并发服务可行（串 [[moonshot-06]] 的前缀共享）。

## 相关题目

- [[moonshot-01]]：长上下文的四条限制——本篇「KV 是硬墙」的上游分析。
- [[moonshot-03]]：K2 的 MoE 路由与训练系统成本——同为 K2 的架构选择。
- [[deepseek-05]]：显存受限下的低延迟服务——MLA 在服务账里的位置（**注意该篇的 9.3× 是「MLA-FP8 vs GQA-bf16 且层数不同」的口径，本篇按同精度同层数给出 3.6×**）。
- [[inference-serving-08]]：显存估算——KV 与权重的预算方法。
- [[llm-internals-05]]：注意力变体的演进（MHA → MQA → GQA → MLA）。

## 参考资料与归属

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（延伸）** —— DeepSeek-AI，2024-05-07：<https://arxiv.org/abs/2405.04434>。第 1、2 节的 **MLA 定义（低秩潜向量 $c_t$、解耦 RoPE 分量 $k^R$、权重吸收）**、$d_c$ 与 $d_{\text{rope}}$ 的取值、以及「质量不低于 MHA」的报告结论来自这篇。
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints（延伸）** —— Ainslie et al. (Google)，2023-05-22：<https://arxiv.org/abs/2305.13245>。第 1 节 GQA 的定义（$n_{\text{kv}}$ 个 KV 头共享给 $n_h$ 个 query 头）与「质量接近 MHA」的结论来自这篇。
- **Fast Transformer Decoding: One Write-Head is All You Need（延伸）** —— Shazeer (Google)，2019-11-06：<https://arxiv.org/abs/1911.02150>。第 1 节 MQA 的定义与「KV 最小但质量代价」的结论来自这篇。
- **上下文从 8K 推到几十万时最先崩什么（本仓库公司题库 · Moonshot AI 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节「解码访存受限、预填充算力受限」的分界被本篇用于论证「两种形式混用」。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$n_h=64$、$d_{\text{head}}=128$、$d_c=512$、$d_{\text{rope}}=64$、层数 61/80、KV 精度 bf16/FP8、上下文 128K、HBM 3.35 TB/s）都是为对比而选的**示例配置**；其中 **GQA-8 在 80 层下恰好复现仓库统一常数 320 KiB/token**（这说明该常数就是 LLaMA-3-70B 的 GQA-8 口径）。**「MLA 对 GQA 是 3.6 倍」这一结论依赖 $d_c+d_{\text{rope}}=576$ 与 $d_{\text{head}}=128$ 的取值**——若换模型配置（例如 $d_{\text{head}}=64$ 或 $d_c$ 更大），倍数会变，**所以应按自家配置重算，不要外推**。④ 的算力倍数（4.5×）只计「每对点积的维度」，未计入 kernel 效率与物化开销。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
