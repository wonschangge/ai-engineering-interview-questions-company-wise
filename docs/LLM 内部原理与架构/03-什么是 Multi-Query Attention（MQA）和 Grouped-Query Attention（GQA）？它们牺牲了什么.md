---
type: question
id: llm-internals-03
topic: LLM 内部原理与架构
order: 3
question: 什么是 Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）？它们牺牲了什么？
question_en: What are Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), and what do they trade away?
asked_at: [Meta, Mistral AI]
level: 进阶
tags: [attention, gqa, mqa, kv-cache]
sources:
  - title: What is Grouped Query Attention (GQA) and Why Do LLMs Use It?
    url: https://outcomeschool.com/blog/grouped-query-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-22
related: [llm-internals-02, llm-internals-01]
updated: 2026-09-28
---

## 一句话答案

> MQA 让所有 attention head 共享同一份 K、V，只保留各自的 Q；GQA 把 head 分组，组内共享一份 K、V，每个 head 仍保留自己的 Q。它们牺牲的是 attention 模式的多样性——MQA 牺牲得多，输出质量可能退化、训练还可能不稳定；GQA 只牺牲一点点，质量接近 MHA。换来的是 KV cache 显存成倍下降，以及更高的并发吞吐。因为 KV cache 只存 K、V 而不存 Q，所以「只共享 K、V」才是真正省显存的那一刀。

## 面试官在考什么

- 是否真的理解 KV cache 的存储结构，而不只是背「KV cache 很占显存」这句话。要能说清「每个 head 一套 K/V × 序列长度 × 层数」这三个乘数各自来自哪里。
- 是否知道 MQA 与 GQA 是同一个设计轴上的两个点，而不是两个互不相干的技巧。能用「组数」把 MHA / MQA 统一成 GQA 的特例，是明显的加分项。
- 能否回答题眼：**牺牲了什么**。只答「省显存、速度快」是半答案；必须说清被牺牲的是 head 之间看输入视角的多样性，并说明 MQA 与 GQA 在这个代价上的量级差别。
- 工程落地细节：是否见过 `num_attention_heads` / `num_key_value_heads`，能否从一份真实 config 里立刻反推出组数与每组头数。
- 是否知道 GQA 能被快速采用的关键工程原因——uptraining，而不必从头重训。

**常见错误答案**

- 「MQA/GQA 是为了减少参数量。」省参数只是附带效果（K/V 投影矩阵变小），真正的收益在推理期的 KV cache 显存与带宽。
- 「GQA 的 Q 也共享了，只是共享得少一点。」Q 从不共享，每个 head 都有自己的 Q。共享 Q 等于把 head 数直接砍掉，那是另一个模型了。
- 「GQA 省的是训练显存。」训练时 activation 与优化器状态才是大头；GQA 的主要战场是 decode 阶段的 KV cache 与显存带宽。

## 原理与推导

### 复习 MHA：每个 head 一套 Q、K、V

多头注意力并行跑 $H$ 次 attention，第 $i$ 次叫一个 head，每个 head 有**独立的**三组权重矩阵：

$$
Q_i = XW_i^Q,\qquad K_i = XW_i^K,\qquad V_i = XW_i^V,\qquad i = 1,\dots,H
$$

head 的输出拼接后再过一层输出投影：

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_H)\,W^O,\qquad
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_{head}}}\right)V_i
$$

取 $H=8$：head 1 用 $Q_1,K_1,V_1$，head 2 用 $Q_2,K_2,V_2$，……一直到 head 8。每个 head 从不同角度审视同一段输入，这是 MHA 质量的来源。

### 为什么这会撑大 KV cache

自回归 decode 时每生成一个 token，都要拿当前 Q 去和**所有历史 token** 的 K、V 做 attention。历史 K、V 若每步重算就太慢，于是缓存起来——这就是 KV cache。关键在于：缓存的是 $\{K_i,V_i\}$，**不是** $Q_i$（当前步只需一个 Q）。所以每个 head 各留一套 K、V：

$$
\text{KV cache 元素数} = 2 \times L \times H \times S \times d_{head}
$$

其中 $L$ 是层数、$H$ 是 head 数、$S$ 是已缓存序列长度、$d_{head}$ 是单 head 维度，系数 2 表示 K 与 V 各一份。换算成字节再乘 dtype 位宽 / 8。

$H=8$、缓存了 1000 个 token 时：K 有 $8 \times 1000 = 8000$ 个向量，V 同样 8000 个，共 16000 个向量——**这只是单层**。真机上的总额还要乘以 $L$。head 数 64、序列 100k 的模型，KV cache 会直接吃掉大块 GPU 显存。

回到「8 个侦探」的类比：每个侦探各拿一本笔记本，记下见过的每条线索。侦探越多、线索越长，纸（显存）就爆得越快。

### MQA：所有 head 共享一份 K、V

MQA 把 K/V 的套数压到 1 套，Q 仍然每个 head 一份：

$$
Q_i = XW_i^Q\ (i=1..H),\qquad K = XW^K,\qquad V = XW^V
$$

代入 $H=8$：

- head 1：$Q_1,\ K_{shared},\ V_{shared}$
- head 2：$Q_2,\ K_{shared},\ V_{shared}$
- ……
- head 8：$Q_8,\ K_{shared},\ V_{shared}$

KV cache 从 8 套降到 1 套，缩小 $H=8$ 倍。一般地，令 $H_{kv}$ 为 KV 套数：

$$
\text{压缩倍数} = \frac{H}{H_{kv}},\qquad
H_{kv}=1 \Rightarrow \text{缩小 } H \text{ 倍}
$$

### GQA：分组，组内共享 K、V

GQA 在「每 head 一套」与「全局一套」之间取中间态：把 $H$ 个 head 分成 $G$ 组，每组 $H/G$ 个 head，**组内共享一份 K、V**，每个 head 仍保留自己的 Q。取 $H=8$、$G=2$（每组 4 个 head）：

- Group 1 = head 1–4，共享 $K_{g1},V_{g1}$
- Group 2 = head 5–8，共享 $K_{g2},V_{g2}$

于是 head 1 用的是 $Q_1$ 配 $K_{g1},V_{g1}$；head 5 用的是 $Q_5$ 配 $K_{g2},V_{g2}$。K/V 从 8 套降到 2 套，KV cache 缩小 $8/2 = 4$ 倍：

$$
\text{压缩倍数} = \frac{H}{G} = \frac{H}{H_{kv}},\qquad
G=2 \Rightarrow 4\times
$$

「8 个侦探分 2 组、每组共用一本笔记本」：纸少用了 4 倍，但仍有两套不同线索，比 MQA 的一本强。

### 三联图：MHA / MQA / GQA 并排

```text
MHA (8 query heads, 8 KV sets - one per head):

  [Q1]  [Q2]  [Q3]  [Q4]  [Q5]  [Q6]  [Q7]  [Q8]
   |     |     |     |     |     |     |     |
   v     v     v     v     v     v     v     v
  [K1]  [K2]  [K3]  [K4]  [K5]  [K6]  [K7]  [K8]
  [V1]  [V2]  [V3]  [V4]  [V5]  [V6]  [V7]  [V8]

MQA (8 query heads, 1 KV set - shared by all heads):

  [Q1]  [Q2]  [Q3]  [Q4]  [Q5]  [Q6]  [Q7]  [Q8]
    \     \     \    |     |    /     /     /
     +-----+-----+---+-----+---+-----+-----+
                       |
                       v
                  [K_shared]
                  [V_shared]

GQA (8 query heads, 2 groups - 1 KV set per group):

  [Q1] [Q2] [Q3] [Q4]       [Q5] [Q6] [Q7] [Q8]
    \   |    |   /             \   |    |   /
     +--+----+--+                +-+----+-+
            |                          |
            v                          v
       [K_group1]                 [K_group2]
       [V_group1]                 [V_group2]
```

顺着图读：MHA 是 8 条竖线全独立；MQA 是 8 条线汇成一个扇入；GQA 是汇成两簇。**Q 那一行在三种结构里都完全一样**——8 个 head 各有各的 Q，这是必须看清楚的一点。

### 为什么只共享 K、V，不共享 Q

两个理由，一个是质量，一个是账本：

1. **Q 决定「从什么角度看输入」。** head 的差异化几乎全部体现在 Q 上：K、V 回答「输入里有什么信息」，Q 回答「我现在要找什么信息」。共享 K、V 相当于让几个 head 查同一份资料柜，但它们各自的检索问题（Q）不同，仍能问出不同的东西，因此 attention 模式只是被限制、没有被抹平。一旦连 Q 也共享，同一组内 head 的 $Q_i$ 相同、$K,V$ 也相同，$\mathrm{head}_i$ 的输出**完全相同**——这组 head 就退化成 1 个 head，多头结构名存实亡。
2. **KV cache 里没有 Q。** decode 阶段只缓存历史 K、V，Q 只是当前 token 的临时量。所以省显存的唯一着力点就是 K、V；共享 Q 一分钱显存都省不到，却要付掉全部质量代价。

### GQA 是 MHA 与 MQA 的泛化

同一个式子 $H_{kv}=G$ 就涵盖了三种结构：

- $G = H$（组数 = head 数，每组 1 个 head）→ 每个 head 独占一套 K/V → **退化为 MHA**
- $G = 1$（所有 head 挤在一组）→ 全局共享一套 K/V → **退化为 MQA**
- $1 < G < H$ → **GQA**

所以 MHA 与 MQA 不是与 GQA 并列的两种方案，而是这条轴的**两个端点**。面试时把这句话说出来，等于把三张图收缩成一个旋钮。

## 数值与代码验证

### 三种结构的对比

| 对比项 | MHA | MQA | GQA |
| --- | --- | --- | --- |
| K/V 套数 | 每 head 一套（$H$） | 全体一套（$1$） | 每组一套（$G$） |
| KV cache 大小 | 最大 | 最小 | 居中 |
| 输出质量 | 最好 | 可能退化 | 接近 MHA |
| 推理显存占用 | 最高 | 最低 | 居中 |
| 推理速度 | 最慢 | 最快 | 快 |
| 8 head、2 组时的 KV 套数 | 8 套 | 1 套 | 2 套 |

相对 MHA，$H=8$、$G=2$ 时 GQA 的 KV cache 是 $2/8 = 1/4$（缩小 4 倍），MQA 是 $1/8$（缩小 8 倍）。倍数关系自始至终只有这三档：MHA 记为 $1\times$，GQA 为 $4\times$，MQA 为 $8\times$。

### 算一遍 LLaMA 2 70B 的单 token KV cache

配置：$L=80$ 层、$H=64$ 个 Q head、$H_{kv}=8$ 个 KV head、$d_{head}=128$、fp16（2 字节）。

$$
\text{per token} = 2 \times L \times H_{kv} \times d_{head} \times 2\ \text{B}
= 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{B} \approx 320\ \text{KB}
$$

若换回 MHA（$H_{kv}=64$），同一个式子是 $2 \times 80 \times 64 \times 128 \times 2 \approx 2.56\ \text{MB/token}$，正好 $64/8 = 8$ 倍。按 $S=4096$ 的上下文算：GQA 约 $1.25\ \text{GiB}$，MHA 约 $10.0\ \text{GiB}$——后者在单卡上几乎无法用这种 batching 服务用户。

### 代码：验证参数量与 K/V 权重平均

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

H, G, d_model, d_head = 8, 2, 512, 64   # 8 个 Q head，2 个 KV 组
mha_kv = nn.Linear(d_model, H * d_head, bias=False)
gqa_kv = nn.Linear(d_model, G * d_head, bias=False)
print("MHA K/V 投影参数:", mha_kv.weight.numel())   # 8 * 64 * 512 = 262144
print("GQA K/V 投影参数:", gqa_kv.weight.numel())   # 2 * 64 * 512 = 65536
print("KV cache 压缩倍数:", H / G)                  # 4.0

# uptraining 第一步：把 MHA 的 K 权重按组取平均，得到 GQA 的 K 权重
W = mha_kv.weight.detach().view(G, H // G, d_head, d_model)  # [组, 组内 head, d_head, d_model]
W_gqa = W.mean(dim=1)                                        # 组内平均
assert W_gqa.shape == (G, d_head, d_model)

# 另一种初始化：组内取第 1 个 head（论文里 mean 略优于选单个，两者都可用）
W_pick = W[:, 0]
print(torch.allclose(W_gqa, W_pick))  # False：两种初始化不同，需靠 utraining 收敛
```

**参数量到底省了多少？** 以 LLaMA 2 7B 的配置（$d_{model}=4096$、32 层）实算：MHA 每层 $W^K$ 与 $W^V$ 各按 $H=32$ 计算，合计 $2 \times 4096 \times 32 \times 128 = 33.6\text{M}$；GQA 改成 8 个 KV head 后每层只有 $2 \times 4096 \times 8 \times 128 = 8.4\text{M}$。32 层累计省下约 $0.81\text{B}$ 参数，占 6.5B 主干权重的约 12%（K/V 参数本身从约占 17% 降到约 4%）。这个数字不小，但**收益的大头仍在推理期**：KV cache 小 8 倍，decode 每步要读的 K/V 字节数也小 8 倍，而 decode 是显存带宽受限的，省下的带宽直接变成吞吐。参数量变小只是顺带的红利。

## 常见追问

- **追问**：为什么不把 Q 也共享，省得更多？
  - 要点：共享 Q 且共享 K/V 后，组内各 head 的 $Q,K,V$ 全同，输出完全一致，等于把这组 head 合成 1 个 head，多头结构失去意义。而且 KV cache 只存 K、V，共享 Q 对显存没有任何帮助，纯亏质量。
- **追问**：GQA 到底会掉多少质量？
  - 要点：掉得很少，属于「接近 MHA」的量级，这也是它被称为甜点的原因。MQA 的退化明显更大，并且可能出现训练不稳定；GQA 通过保留多个 K/V 视角把这一损失压回去。注意中间态：$G$ 越小省得越多、质量风险越大，$G$ 的选择就是在曲线上挑点。
- **追问**：GQA 主要省在训练还是推理？
  - 要点：主要省推理。训练时 MHA 本身工作得很好，显存瓶颈是激活值与优化器状态，GQA 减少约 12% 主干参数（也就少了对应的优化器状态）算是顺带好处，但不是它被采用的理由。推理 decode 阶段要逐步回看全部历史 token，KV cache 随 $S$ 线性增长且每步都要整块读一遍，GQA 把它按 $H/H_{kv}$ 缩小，直接换来更长上下文、更高并发与更低单 token 延迟。
- **追问**：拿到一份模型 config，怎么看出 GQA 的组数？
  - 要点：看 `num_attention_heads`（Q head 数）与 `num_key_value_heads`（KV 套数 = 组数）。每组头数 $=\texttt{num\_attention\_heads} / \texttt{num\_key\_value\_heads}$。例如 32 个 Q head + 8 个 KV head = 8 组、每组 4 个 head，KV cache 比 MHA 小 4 倍。很多框架里这个字段的默认值等于 `num_attention_heads`，即默认 MHA。
- **追问**：LLaMA 2 是不是所有尺寸都用 GQA？
  - 要点：不是。LLaMA 2 里 34B 与 70B 用 GQA（8 个 KV 组，70B 有 64 个 Q head，因此 KV cache 比 MHA 小 8 倍；34B 在论文中描述过但权重未公开发布），而 7B 与 13B 仍是标准 MHA。到 LLaMA 3 才全尺寸使用 GQA。

## 公司变体

- **Meta（超级智能实验室、FAIR、Llama）**：这题基本是 Meta 的主场，因为它直接对应 Llama 系列的架构决策。偏工程实现：会追问 LLaMA 2 哪些尺寸用 GQA、70B 的 64 个 Q head 对应多少个 KV 组、KV cache 因此小多少倍，以及 uptraining 怎么把已有 MHA 模型低成本转成 GQA。
- **Mistral AI**：偏工程取舍与部署视角。Mistral 7B 用 GQA（32 个 Q head、8 个 KV 组）并搭配 sliding window attention，常见问法是「为什么两个都要」「它们各自省的是什么」——GQA 省每 token 的 K/V 体积，sliding window 限制注意的历史长度，两者作用在不同的乘数上。

## 相关题目

- [[llm-internals-02]]：KV cache 显存公式的完整推导，是本题的前置。先能手推 $2 \times L \times H \times S \times d_{head} \times \text{bytes}$，再谈 GQA 把哪个乘数改掉。
- [[llm-internals-01]]：attention 的数学本身。Q、K、V 各自的角色决定了「为什么可以共享 K/V 而不能共享 Q」。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《What is Grouped Query Attention (GQA) and Why Do LLMs Use It?》，2026-04-22，<https://outcomeschool.com/blog/grouped-query-attention>

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
