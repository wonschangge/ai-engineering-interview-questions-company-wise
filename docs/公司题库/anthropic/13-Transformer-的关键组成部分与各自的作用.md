---
type: question
id: anthropic-13
company: Anthropic
topic: llm-internals
order: 13
question: Transformer 模型的关键组成部分有哪些，为什么每一部分都很重要？
question_en: What are the key components of a Transformer model, and why does each matter?
asked_at: []
level: 进阶
tags: [Transformer, 注意力, 位置编码, 归一化, 前馈网络]
sources:
  - title: Attention Is All You Need（延伸）
    url: https://arxiv.org/abs/1706.03762
    author: Vaswani et al. (NeurIPS 2017)
    published: 2017-06-12
  - title: Math Behind RoPE (Rotary Position Embedding)
    url: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding
    author: Amit Shekhar (Outcome School)
    published: 
  - title: RMSNorm (Root Mean Square Layer Normalization)
    url: https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization
    author: Amit Shekhar (Outcome School)
    published: 
  - title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）
    url: https://arxiv.org/abs/2205.14135
    author: Dao et al. (NeurIPS 2022)
    published: 2022-05-27
related: [llm-internals-01, llm-internals-02, llm-internals-04, anthropic-15, anthropic-16]
updated: 2026-09-28
---

## 一句话答案

> 按「数据流」讲比按清单讲更清楚。一个 decoder-only Transformer 的一条样本依次经过六件事：
> ① **词嵌入 + 位置信息**——把 token id 变成向量，并注入「谁在前谁在后」（RoPE 是把位置变成对 $q,k$ 的旋转，因此它天然只影响注意力分数，不影响数值范围）；
> ② **因果自注意力**——让每个位置聚合此前位置的信息；多头是为了在不同子空间里学不同的关系；GQA/MQA 是为了把 KV cache 变小（推理成本的关键杠杆）；
> ③ **前馈网络（FFN）**——逐位置的两层 MLP（现代常用 SwiGLU 门控），**它承载了大部分参数与「知识」**；
> ④ **残差连接**——让梯度能穿过几十上百层，是深网络可训练的前提；
> ⑤ **归一化**——稳定激活尺度；现代模型多用 RMSNorm 且放在子层之前（pre-norm），因为它在深层网络里更稳、更省；
> ⑥ **输出头 + 权重共享**——把最后一层隐状态映射回词表分布（常与嵌入矩阵共享权重，省一大块参数）。
> **为什么每一部分都重要**——判据是「去掉它会发生什么」：没有位置信息 → 变成词袋；没有残差 → 深层训不动；没有归一化 → 训练不稳定且对学习率极敏感；把注意力换成均值池化 → 长程依赖与精确检索能力丧失；FFN 变窄 → 参数化知识容量下降。**面试里能这样「反证」的，才算真理解。**

## 面试官在考什么

- **能否按数据流讲清形状**：$[B,T,d]$ → QKV 投影 → 注意力 → FFN → logits $[B,T,V]$；每一跳的张量形状与参数量分布能算出来（串 [[anthropic-15]]）。
- **参数分布**：以 LLaMA 式配置为例，FFN 约占每层参数的 **2/3**（门控 FFN 的 $3\times d\times d_{ff}$），注意力约占 1/3；能报出量级说明你算过。
- **注意力的变体与动机**：MHA → MQA → GQA 的演进是为了**KV cache**（推理显存与带宽），而不是为了训练更快；MLA 走的是低秩压缩路线（串 [[llm-internals-04]]）。
- **归一化的位置与种类**：post-LN（原始论文）vs pre-LN（现代主流）的差别；LayerNorm vs RMSNorm；为什么 pre-norm 训练更稳（残差路径上没有被归一化「掐住」）。
- **位置编码的选型**：可学习 vs 正弦 vs RoPE vs ALiBi；RoPE 的外推问题与 YaRN 式扩展（串 [[llm-internals-01]]）。
- **训练与推理的视角差异**：训练时并行处理整段序列（算力受限），推理 decode 时逐 token、KV cache 主导（带宽受限）——同一个架构在两端的瓶颈完全不同。

**常见错误答案**

- 只背组件清单（「有 attention、有 FFN、有 LayerNorm」）而说不出各自的作用与去掉的后果。
- 把 GQA 说成「为了训练更快」——它主要省的是推理时的 KV cache。
- 认为注意力是参数主体——**FFN 才是**。
- 把位置编码说成「加在输入上就完事」——RoPE 是作用在 $q,k$ 上的旋转，且影响外推行为。

## 原理与推导

### 1. 逐步数据流与形状

设 batch $B$、序列长度 $T$、隐藏维 $d$、头数 $h$、head_dim $d_h=d/h$、FFN 中间维 $d_{ff}$：

| 步骤 | 操作 | 形状 | 参数量（每层） |
| --- | --- | --- | --- |
| 嵌入 | 查表 | $[B,T,d]$ | $V\cdot d$（全模型共享） |
| QKV 投影 | 三个线性层 | $[B,T,d]\to[B,T,d]$ ×3 | $3d^2$（GQA 下 K/V 更小） |
| 注意力分数 | $QK^\top/\sqrt{d_h}$ | $[B,h,T,T]$ | 0 |
| 加权求和 | $\mathrm{softmax}\cdot V$ | $[B,h,T,d_h]$ | 0 |
| 输出投影 | 线性层 | $[B,T,d]$ | $d^2$ |
| FFN（SwiGLU） | $\mathrm{down}(\mathrm{silu}(\mathrm{gate}(x))\odot \mathrm{up}(x))$ | $[B,T,d]\to[B,T,d_{ff}]\to[B,T,d]$ | $3\,d\,d_{ff}$ |
| 归一化 | RMSNorm | 逐位置 | $2d$（可忽略） |
| 输出头 | 线性层 | $[B,T,V]$ | $V\cdot d$（常与嵌入共享） |

取 $d=8192$、$d_{ff}=28672$（LLaMA-3-70B 量级）：FFN $=3\times8192\times28672\approx7.05\times10^{8}$；注意力侧（$Q$、$O$ 各 $d^2$，K/V 因 GQA 缩小）合计约 $1.5\times10^{8}$。**在 GQA 配置下 FFN 占每层参数约 82%**，这是「知识主要存在 FFN 里」这一常见说法的参数依据（严格说：注意力负责路由与聚合，FFN 负责变换与存储）。

### 2. 每一部分「去掉会怎样」（反证法）

| 组件 | 去掉后的后果 | 原因 |
| --- | --- | --- |
| 位置信息 | 变成置换不变的词袋模型 | 注意力对输入顺序本来不敏感 |
| 因果掩码 | 训练时能「看到答案」（泄漏） | 自回归目标要求只看历史 |
| 残差 | 深层梯度消失/爆炸，几乎训不动 | 梯度需沿恒等路径回传 |
| 归一化 | 激活尺度漂移，学习率极敏感 | 各层输入分布不稳定 |
| FFN | 表达能力骤降（每层只剩加权平均） | 非线性变换主要靠它 |
| 多头 | 只能学单一关系类型 | 不同子空间捕捉不同模式 |
| 输出头权重共享 | 参数量增加 $V\cdot d$ | 嵌入与反嵌入语义对偶 |

### 3. 注意力的三个变体（为什么这么演进）

| 变体 | KV 头数 | KV cache（每 token 每层） | 代价 |
| --- | --- | --- | --- |
| MHA | $h$ | $2\cdot h\cdot d_h\cdot 2\ \text{B}=4d$ 字节 | 最大 |
| GQA | $g$（$1<g<h$） | $4d\cdot g/h$ | 略损质量 |
| MQA | 1 | $4d/h$ | 质量损失更明显 |

以 $d=8192$、80 层、fp16 计：MHA 每 token KV $=2\times80\times8192\times2=2.6\ \text{MiB/token}$；GQA（$g/h=1/8$）降到约 **0.33 MiB/token**——长上下文服务里这是能不能跑起来的差别（串 [[llm-internals-02]]）。

### 4. 训练 vs 推理：两套瓶颈

- **训练/prefill**：整段序列并行，矩阵乘密集 → **算力受限**（FLOPs 主导），因此关心 $6ND$ 与 MFU；
- **推理/decode**：逐 token 生成，每步要把权重与 KV 从显存搬到计算单元 → **带宽受限**（bytes 主导），因此关心 batch 大小与 KV 布局。
**同一架构、不同瓶颈**：这就是为什么训练优化（并行策略、混合精度）与推理优化（continuous batching、PagedAttention、投机解码）几乎是两套独立的技术栈。

### 5. 一个容易被忽略的组件：初始化与缩放

- 残差分支的输出通常按层数做缩放（$1/\sqrt{2L}$ 类做法），否则深层激活方差累积；
- 注意力分数除 $\sqrt{d_h}$ 是为了让 softmax 输入方差与 $d_h$ 无关；
- 输出头的 logit 缩放（soft cap）与温度在采样阶段配合。
这些「小细节」决定了大模型能否稳定训练——面试里提到其中一两个，说明你读过实现而不是只读论文图。

## 数值与代码验证

### 表 1：LLaMA-3-70B 量级的每层参数拆解（$d=8192$，$d_{ff}=28672$）

| 组件 | 算式 | 参数量 | 占每层比例 |
| --- | --- | --- | --- |
| Q 投影 | $d\times h d_h=d^2$ | $6.71\times10^{7}$ | 7.8% |
| K/V 投影（GQA $g=8$） | $2\times d\times g d_h$ | $1.68\times10^{7}$ | 2.0% |
| 注意力输出投影 | $d^2$ | $6.71\times10^{7}$ | 7.8% |
| FFN（SwiGLU 三项） | $3\,d\,d_{ff}$ | $7.05\times10^{8}$ | **82.4%** |
| 归一化（2 个 RMSNorm） | $2d$ | $1.64\times10^{4}$ | ~0% |
| **合计（每层）** | | $\approx8.56\times10^{8}$ | 100% |

口径：$d=8192$、$d_{ff}=28672$、$h=64$、$d_h=128$、GQA 的 KV 头 $g=8$；嵌入与输出头共享、按全模型计一次。**读法**：在 GQA 配置下 FFN 占每层参数约 **82%**（若把 KV 换回 MHA，占比降到 72%）——这解释了为什么「把 FFN 稀疏化」（MoE）是扩大参数规模最常用的手段，以及为什么注意力侧的结构改动（GQA/MQA/MLA）主要影响的是**推理显存**而不是参数量。

### 表 2：每 token KV cache（fp16，80 层、$d=8192$）

| 变体 | 头配置 | 每 token KV（精确值） | 128K 上下文 |
| --- | --- | --- | --- |
| MHA | $h=64$ | $2\times80\times64\times128\times2=2{,}621{,}440$ B = **2.5 MiB** | **320 GiB** |
| GQA $g=8$ | 8 组 | $2\times80\times8\times128\times2=327{,}680$ B = **320 KiB** | **40 GiB** |
| MQA | 1 | $2\times80\times1\times128\times2=40{,}960$ B = **40 KiB** | **5 GiB** |

口径：$L=80$ 层、$d_h=128$、fp16（2 字节），K 与 V 各计一次。**注意 GQA $g=8$ 算出来正好是 320 KiB/token——这就是本仓库统一常数「LLaMA-3-70B 每 token KV 320 KiB」的来源**；128K 上下文 = 每 token × 131072。

### 可运行代码

```python
# 参数量与 KV cache：逐组件算账（可直接跑）
def layer_params(d, d_ff, h, d_head, g_kv=None, swiglu=True):
    g = g_kv or h
    q = d * h * d_head
    kv = 2 * d * g * d_head
    o = d * d
    ffn = 3 * d * d_ff if swiglu else 2 * d * d_ff
    norm = 2 * d
    total = q + kv + o + ffn + norm
    return dict(Q=q, KV=kv, O=o, FFN=ffn, Norm=norm, total=total)

d, d_ff, h, d_head = 8192, 28672, 64, 128
for label, g in (("MHA（g=64）", 64), ("GQA（g=8）", 8), ("MQA（g=1）", 1)):
    p = layer_params(d, d_ff, h, d_head, g_kv=g)
    print(f"{label:<14} 每层 {p['total']/1e9:.3f}B  "
          f"[Q {p['Q']/1e9:.3f}B | KV {p['KV']/1e9:.3f}B | O {p['O']/1e9:.3f}B | "
          f"FFN {p['FFN']/1e9:.3f}B ({p['FFN']/p['total']:.1%})]")

print("\n每 token KV cache（fp16 = 2 字节，L=80 层）")
L, BYTES = 80, 2
for label, g in (("MHA", 64), ("GQA g=8", 8), ("MQA", 1)):
    per_tok = 2 * L * g * d_head * BYTES          # K 和 V 各 g*d_head 维
    print(f"  {label:<8} {per_tok/1024/1024:8.4f} MiB/token   "
          f"128K 上下文 {per_tok*131072/1024**3:7.1f} GiB")

# 注意力与 FFN 的 FLOPs 对比（prefill 口径，含 2NP 与 2N^2 d）
def flops(d, d_ff, T, L, g=8, h=64, d_head=128):
    proj = 2 * T * (d*d + 2*d*g*d_head + d*d)              # QKV + O
    ffn = 2 * T * 3 * d * d_ff
    attn_scores = 2 * T * T * d                            # QK^T + softmax*V 近似
    return {"投影": proj*L, "FFN": ffn*L, "注意力矩阵": attn_scores*L}
print("\nprefill FLOPs 随序列长度的变化（每层、单位 GFLOPs）")
for T in (1024, 8192, 32768):
    f = flops(d, d_ff, T, 1)
    print(f"  T={T:>6,d}: 投影 {f['投影']/1e9:8.1f}  FFN {f['FFN']/1e9:8.1f}  "
          f"注意力 {f['注意力矩阵']/1e9:9.1f}  -> 注意力占比 "
          f"{f['注意力矩阵']/(f['投影']+f['FFN']+f['注意力矩阵']):.1%}")
print("读法：序列越长，注意力的二次项占比越高；短序列时 FFN 主导（这就是长上下文要专门优化的原因）")
```

预期输出要点（实跑）：每层参数量里 **FFN 占约 2/3**（GQA 让 KV 从 0.134B 降到 0.017B，对总参数影响很小、但对推理显存影响巨大）；每 token KV 从 MHA 的 2.5 MiB 降到 GQA(g=8) 的 0.3125 MiB、MQA 的 0.039 MiB，128K 上下文对应 320 GiB → 40 GiB → 5 GiB；prefill FLOPs 表显示**注意力占比随 $T$ 上升但远比直觉低**（$T=1024$ 时 1.0%、$T=8192$ 时 7.3%、$T=32768$ 时 23.9%）——即使在 32K 上下文，FFN 仍占大头，这解释了为什么长上下文优化既要管注意力（二次项）也不能忽略 FFN（线性项主导绝对量）。

## 常见追问

- **追问**：为什么用 pre-norm 而不是原始论文的 post-norm？
  - 要点：post-norm 把归一化放在残差相加之后，会反复缩放残差路径的信号，深层训练需要精细的学习率预热；pre-norm 让残差路径保持恒等，深层更稳（代价是需要额外的输出归一化）。
- **追问**：RMSNorm 相比 LayerNorm 省在哪？
  - 要点：省去均值中心化与偏置，只做均方根缩放——计算更少、参数更少，实践中质量相当。
- **追问**：SwiGLU 为什么比 ReLU/GELU 的 FFN 好？
  - 要点：门控结构提供了输入相关的乘法交互（$\mathrm{silu}(W_1x)\odot W_3x$），表达力更强；代价是三项矩阵（参数增加 50%），所以 $d_{ff}$ 通常相应调小（例如 $8d/3$ 而不是 $4d$）。
- **追问**：注意力真的是瓶颈吗？
  - 要点：短序列时不是（FFN 与投影主导）；长序列（>8K）与 decode 阶段才是（KV 读取带宽 + 注意力矩阵）。要按阶段与序列长度分开讨论。
- **追问**：位置编码为什么选 RoPE？
  - 要点：相对位置性质（内积只依赖相对距离）、可外推（配合插值/温度调整）、实现上只是对 $q,k$ 做旋转、且不与 KV cache 冲突（缓存的 K 已带位置旋转）。
- **追问**：如果要减少参数量，你会动哪里？
  - 要点：优先 FFN（占 2/3）——稀疏化（MoE）、低秩分解、或降 $d_{ff}$；注意力侧用 GQA/MLA 主要省的是推理显存而非参数。要给出质量-成本的权衡证据（不能只凭直觉）。

## 相关题目

- [[llm-internals-01]]：Transformer 基础组件与自注意力的推导，本题是它的「为什么」版本。
- [[llm-internals-02]]：KV cache 的内存推导，本题表 2 与之同口径。
- [[llm-internals-04]]：MLA 的低秩压缩路线，是注意力变体演进的另一支。
- [[anthropic-15]]：本模型的矩阵操作与 roofline 分析，直接接在本题的数据流上。
- [[anthropic-16]]：batching 推理系统，解释了 KV cache 规模如何决定服务架构。

## 参考资料与归属

- **Attention Is All You Need（延伸）** —— Vaswani et al. (NeurIPS 2017)，2017-06-12：<https://arxiv.org/abs/1706.03762>。第 1 节的六步数据流、多头注意力、位置编码与残差/归一化的原始设计来自这篇。
- **Math Behind RoPE (Rotary Position Embedding)** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding>。第 1 节「RoPE 以旋转方式注入相对位置」的机制表述转述自这篇。
- **RMSNorm (Root Mean Square Layer Normalization)** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization>。第 2 节与追问里「RMSNorm 省去均值中心化」的对比来自这篇。
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）** —— Dao et al. (NeurIPS 2022)，2022-05-27：<https://arxiv.org/abs/2205.14135>。第 4 节「注意力是显存/带宽问题而不只是 FLOPs 问题」的依据来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$d=8192$、$d_{ff}=28672$、$h=64$、$d_h=128$、$L=80$、fp16 2 字节、128K token）都是**按 LLaMA-3-70B 量级配置的假设值**推算，不是论文原文数字；配置细节请以各模型 `config.json` 为准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
