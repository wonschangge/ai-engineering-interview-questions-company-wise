---
type: question
id: llm-internals-01
topic: LLM 内部原理与架构
order: 1
question: 解释 scaled dot-product attention，以及为什么 1/sqrt(d_k) 缩放因子很重要。
question_en: Explain scaled dot-product attention and why the 1/sqrt(d_k) scaling factor matters.
asked_at: []
level: 进阶
tags: [attention, softmax, 方差, 数值稳定性]
sources:
  - title: Why Do We Scale Attention by √dₖ? The Math Behind the Scaling Factor
    url: https://outcomeschool.com/blog/scaling-dot-product-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-05
  - title: Math behind Attention - Q, K, and V
    url: https://outcomeschool.com/blog/math-behind-attention-qkv
    author: Amit Shekhar (Outcome School)
related: [llm-internals-02, llm-internals-03]
updated: 2026-09-28
---

## 一句话答案

> scaled dot-product attention 就是 $\mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V$。除以 $\sqrt{d_k}$ 不是为了「让数字小一点」，
> 而是因为：当 Q、K 的每一维独立、均值 0、方差 1 时，点积 $q\cdot k$ 的方差**恰好等于 $d_k$**（精确值，不是近似）。
> 方差随维度线性增长，会把 softmax 推到饱和区，输出退化成 one-hot，梯度随之消失。
> 把点积除以常数 $c$ 时方差变成 $d_k/c^2$，令它等于 1 解得 $c=\sqrt{d_k}$ —— 这是唯一能让方差与维度无关的因子。

## 面试官在考什么

- 你是否知道缩放解决的是**训练动力学**问题（softmax 饱和 → 梯度消失），而不是笼统的「数值变大」。
- 你能不能把方差推导讲出来：为什么点积的方差是 $d_k$，为什么除以 $c$ 方差要除以 $c^2$，为什么解出来是 $\sqrt{d_k}$。
- 你是否知道「$1/\sqrt{d_k}$」的出处是 *Attention Is All You Need*，以及它的适用边界（$d_k$ 很小时几乎没有影响）。
- 你能否区分三件常被混为一谈的事：**缩放**（改变分布形状）、**max-subtraction**（防 exp 溢出）、**温度参数**（推理时二次调温）。
- 常见错误答案：
  - 「缩放是为了防止数值溢出」——溢出靠 softmax 减最大值解决；缩放即使在没有溢出风险的数值范围内也仍然必要。
  - 「因为点积会随维度变大，所以要除以维度的平方根」——这是循环论证，说不出方差为什么等于 $d_k$。

## 原理与推导

### 1. 公式与形状

$$ \mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$

| 符号 | 含义 | 形状（单头、单个序列） |
| --- | --- | --- |
| $Q$ | Query 矩阵，当前 token 想「问什么」 | $n \times d_k$ |
| $K$ | Key 矩阵，历史 token 能「答什么」 | $m \times d_k$ |
| $V$ | Value 矩阵，历史 token 携带的信息 | $m \times d_v$ |
| $QK^\top$ | 注意力打分（未缩放） | $n \times m$ |
| $d_k$ | 每个头的 Key/Query 维度，常见 64 或 128 | 标量 |

缩放发生在 softmax **之前**、点积**之后**。注意 $d_k$ 是**单个头**的维度，不是模型的隐藏维度：头数越多，每个头的 $d_k$ 越小。

### 2. 不缩放会发生什么

softmax 把打分变成概率：

$$ \mathrm{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}} $$

而 $e^x$ 增长极快，这是问题根源：

| $x$ | 1 | 5 | 10 | 20 | 50 |
| --- | --- | --- | --- | --- | --- |
| $e^x$ | 2.718 | 148.413 | 22,026.5 | 4.85×10⁸ | 5.18×10²¹ |

给一组打分 $[50, 10, 5]$（模拟大 $d_k$ 下未缩放的点积量级），softmax 输出是 $[1.0,\ 0.0,\ 0.0]$ —— 完全 one-hot。
也就是说，模型把所有注意力压在一个位置上，其余位置的信息被彻底丢弃。

对照一组温和的打分 $[5, 1, 0.5]$，softmax 输出 $[0.9714,\ 0.0178,\ 0.0108]$：仍然是第一名占优，但其它位置没有被清零，梯度依然可以流过去。

one-hot 分布带来两个直接后果：

1. **梯度消失**：softmax 在饱和区（某一项远大于其它项）的雅可比趋于 0，反向传播传回的信号接近 0，参数几乎不再更新。
2. **容量浪费**：注意力的意义是「从多个相关位置聚合信息」，分布退化成 one-hot 之后，多头、多层堆叠带来的聚合能力被浪费。

### 3. 点积的方差为什么恰好是 $d_k$

前提假设（也是这个推导唯一的假设）：$q$ 与 $k$ 的每一维独立同分布，均值 0、方差 1。真实模型里输入会经过 LayerNorm 或类似的归一化，这个假设是合理的。

**单项**：对单个维度 $i$，

$$ \mathrm{Var}(q_i k_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]-\left(\mathbb{E}[q_i]\mathbb{E}[k_i]\right)^2 = 1\cdot 1-0=1 $$

（用到 $q_i \perp k_i$，所以乘积的期望可以拆开。）

**求和**：点积是 $d_k$ 个相互独立项的和，方差可加：

$$ \mathrm{Var}(q\cdot k)=\sum_{i=1}^{d_k}\mathrm{Var}(q_i k_i)=d_k $$

**用 $(a+b+c)^2$ 展开做直观验证**（以 $d_k=3$ 为例，令 $a=q_1k_1,\ b=q_2k_2,\ c=q_3k_3$）：

$$ (a+b+c)^2=\underbrace{a^2+b^2+c^2}_{\text{3 项，每项期望 }1}+\underbrace{2(ab+ac+bc)}_{\text{6 项，每项期望 }0} $$

- 平方项：$\mathbb{E}[a^2]=\mathbb{E}[q_1^2]\mathbb{E}[k_1^2]=1$，共 $d_k$ 项，贡献 $d_k$。
- 交叉项：$\mathbb{E}[ab]=\mathbb{E}[q_1]\mathbb{E}[k_1]\mathbb{E}[q_2]\mathbb{E}[k_2]=0$，共 $d_k^2-d_k$ 项，贡献 0。

所以 $\mathbb{E}[(q\cdot k)^2]=d_k$；又因为均值为 0，方差就等于它：$\mathrm{Var}(q\cdot k)=d_k$。这个结论对任意 $d_k$ 都成立，且是精确等式。

**结论**：$d_k$ 越大，点积的取值范围越宽（标准差是 $\sqrt{d_k}$），softmax 越容易被推到饱和区。这就是「为什么 $d_k$ 大了会出问题」的定量解释。

### 4. 为什么缩放因子恰好是 $\sqrt{d_k}$

先记住方差的一个基本性质：**除以常数 $c$，方差除以 $c^2$**（因为方差度量的是平方偏差）。

例子：$[2,4,6]$ 方差为 $2.67$；全部除以 $c=2$ 得到 $[1,2,3]$，方差为 $0.67=2.67/4$。

用它作用到点积上：

$$ \mathrm{Var}\!\left(\frac{q\cdot k}{c}\right)=\frac{d_k}{c^2} $$

我们希望缩放后的打分**方差与 $d_k$ 无关**，取最自然的归一化目标 1：

$$ \frac{d_k}{c^2}=1 \;\Longrightarrow\; c^2=d_k \;\Longrightarrow\; c=\sqrt{d_k} $$

于是 $\mathrm{Var}(q\cdot k/\sqrt{d_k})=d_k/d_k=1$：无论 $d_k$ 是 64 还是 1024，送进 softmax 的打分都处在同样的数值尺度上，softmax 不再被迫饱和。这也解释了为什么不是除以 $d_k$：除以 $d_k$ 会把方差压到 $1/d_k$，$d_k$ 大时分布过平，注意力被摊薄，同样学不好。

### 5. 与温度参数的关系

带温度的 softmax 写作 $\mathrm{softmax}(z/\tau)$。scaled dot-product attention 相当于把温度固定成 $\tau=\sqrt{d_k}$。推理时再调 temperature，等于在缩放之后再乘一个因子，是同一个旋钮家族里的第二次调节——这也是为什么不同模型/不同实现里的「温度」不可直接横向比较。

## 数值与代码验证

### 数值对比（$d_k=64$，缩放因子 $\sqrt{64}=8$）

假设三个位置的未缩放打分是 $[14, 10, 12]$：

| | 打分 | softmax 输出 | 分布形态 |
| --- | --- | --- | --- |
| 未缩放 | $[14, 10, 12]$ | $[0.8668,\ 0.0159,\ 0.1173]$ | 第一个位置吃掉 86.7%，第二个几乎被忽略 |
| 缩放后 | $[1.75, 1.25, 1.5]$ | $[0.4192,\ 0.2543,\ 0.3265]$ | 41.9% / 25.4% / 32.7%，三个位置都参与 |

打分之间只差 2~4，未缩放时却已经非常极端；缩放后分布立刻变得可用。注意打分本身并没有被「改变排序」，改变的是 softmax 感受到的**尺度**。

### 小维度时缩放几乎无影响

$d_k=4$ 时 $\sqrt{d_k}=2$：一个打分 1 缩放后是 0.5，两者都在 softmax 的温和区间。所以**缩放的重要性随 $d_k$ 增长**，这也是为什么把 512 维拆成 8 个头（每头 $d_k=64$）时，每个头单独看仍然需要缩放。

### 代码验证

```python
import torch

torch.manual_seed(0)
d_k = 64
q = torch.randn(1, 1, d_k)          # 单个 query
k = torch.randn(1, 512, d_k)        # 512 个 key

scores = q @ k.transpose(-2, -1)    # [1, 1, 512]，未缩放打分
print(scores.var().item())          # ≈ 64，即 d_k
print(torch.softmax(scores, -1).max().item())   # 接近 1：分布已经很尖

scaled = scores / d_k ** 0.5        # 除以 sqrt(d_k) = 8
print(scaled.var().item())          # ≈ 1
print(torch.softmax(scaled, -1).max().item())   # 明显更平缓
```

用纯 Python 复现方差结论（每维标准正态取样，多次实验取样本方差）：

| $d_k$ | 8 | 64 | 512 |
| --- | --- | --- | --- |
| 模拟得到的点积方差 | 8.22 | 63.91 | 510.61 |
| 理论值 $d_k$ | 8 | 64 | 512 |

样本方差在 $d_k$ 较小时略高于理论值（估计噪声更大），趋势与推导一致：**方差随 $d_k$ 线性增长**。

## 常见追问

- **追问**：$d_k$ 很小的时候还需要缩放吗？
  - 要点：需要，但影响可以忽略不计。$d_k=4$ 时打分从 1 变成 0.5，两者都远离 softmax 饱和区；$d_k$ 越大缩放越关键。
- **追问**：为什么是 $\sqrt{d_k}$，而不是 $d_k$、$\log d_k$ 或别的常数？
  - 要点：只有 $\sqrt{d_k}$ 能把方差精确拉回 1。除以 $d_k$ 会让方差变成 $1/d_k$（分布过平、注意力被摊薄），除以固定常数则无法适应不同维度。
- **追问**：这跟 softmax 的数值稳定实现（减最大值）是一回事吗？
  - 要点：不是。max-subtraction 是防 `exp` 上溢的实现技巧，数学上不改变 softmax 的输出；缩放改变的是分布形状和梯度大小。
- **追问**：多头注意力里每个头的 $d_k$ 更小，缩放还有意义吗？
  - 要点：有。缩放按每个头自己的 $d_k$ 计算。另外这也是 MQA/GQA 的讨论聚焦在 **KV 头数**上的原因：它们不动 $d_k$，动的是 K/V 的份数。
- **追问**：FlashAttention 改变了这个缩放因子吗？
  - 要点：没有。FlashAttention 只改变计算顺序（分块 + online softmax），数学与缩放因子完全不变，省的是显存读写而不是 FLOPs。
- **追问**：如果不用 softmax 而用别的归一化（如 sigmoid attention），缩放还需要吗？
  - 要点：缩放的必要性来自 softmax 的指数敏感性与方差随维度的增长；换归一化要重新分析对应统计量，不能照搬 $\sqrt{d_k}$。

## 相关题目

- [[llm-internals-02]] —— KV cache 与显存公式：缩放决定注意力打得多散，KV cache 决定这份注意力要花多少显存。
- [[llm-internals-03]] —— MQA / GQA：不动 $d_k$，而是压缩 K/V 的份数。

## 参考资料与归属

- Amit Shekhar, *Why Do We Scale Attention by √dₖ? The Math Behind the Scaling Factor*, Outcome School, 2026-04-05 —— <https://outcomeschool.com/blog/scaling-dot-product-attention>
- Amit Shekhar, *Math behind Attention - Q, K, and V*, Outcome School —— <https://outcomeschool.com/blog/math-behind-attention-qkv>
- 原始出处：Vaswani et al., *Attention Is All You Need*, 2017（缩放因子的来源论文）。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
