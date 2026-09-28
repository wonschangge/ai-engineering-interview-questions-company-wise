---
type: question
id: llm-internals-07
topic: LLM 内部原理与架构
order: 7
question: 什么是 transformer 中的位置编码？它是如何演进的（sinusoidal → learned → RoPE → ALiBi）？
question_en: What is positional encoding in transformers, and how has it evolved?
asked_at: []
level: 进阶
tags: [位置编码, rope, alibi, 长上下文]
sources:
  - title: Positional Embeddings in LLMs
    url: https://outcomeschool.substack.com/p/positional-embeddings-in-llms
    author: Amit Shekhar (Outcome School)
  - title: Math Behind RoPE (Rotary Position Embedding)
    url: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding
    author: Amit Shekhar (Outcome School)
  - title: Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation（延伸）
    url: https://arxiv.org/abs/2108.12409
    author: Ofir Press et al.
    published: 2021-08-24
related: [llm-internals-08, llm-internals-01, llm-internals-16]
updated: 2026-09-28
---

## 一句话答案

> self-attention 对输入是**置换等变**的：把 token 的顺序打乱，输出只是跟着打乱，模型本身分不出「猫追狗」和「狗追猫」，所以必须显式注入位置信息。演进的主线是**位置信息放在哪里、以及它是不是只依赖相对距离**：sinusoidal 把一个固定向量加在 embedding 上（绝对位置，理论上可外推但实测不行）；learned 直接学一张 $[L_{max}, d]$ 的表（表达自由，超出训练长度就没有 embedding）；RoPE 不加法而是按位置**旋转** Q、K，让 $\mathbf{q}_m^\top\mathbf{k}_n$ 自动只依赖 $n-m$，且不引入任何可学参数；ALiBi 更极端，彻底不加位置嵌入，直接在 attention logit 上减去一个与距离成正比的线性惩罚。今天的实践结论是 RoPE 一统主流（LLaMA、Qwen、DeepSeek、Mistral、Gemma），ALiBi 代表「零成本相对偏置」这条支线。

## 面试官在考什么

- 是否会从**置换等变性**（permutation equivariance）出发解释「为什么需要位置编码」，而不是背一句「transformer 并行处理所以没有顺序」。能写出 $\mathrm{Attn}(P X) = P\,\mathrm{Attn}(X)$ 的人，和只会举例的人不在一个层次。
- 是否分得清「绝对位置 / 相对位置」「加在 embedding 上 / 加在 attention 上 / 乘在 Q、K 上」这两组正交的设计轴。RoPE 既不是加法也不在底层，这是最容易答错的地方。
- 能否说清 sinusoidal 的**频率结构**（不同维度是不同波长的正弦波）与那个常被引用的线性关系 $\mathrm{PE}(pos+k) = M_k\,\mathrm{PE}(pos)$，以及「理论可外推 ≠ 实测可外推」。
- 是否知道 ALiBi 的具体形式：$-m\cdot|i-j|$、每个 head 一个固定斜率、斜率取几何序列，以及它为什么是**零运行时开销**（并进 causal mask）。
- 能否给出演进对比表并报出代表模型（原始 Transformer、BERT、GPT-2、T5、ALiBi、LLaMA 系），最后落到「为什么现在是 RoPE」。

**常见错误答案**

- 「RoPE 是把旋转后的位置向量加到 token embedding 上。」RoPE 是在每一层对 Q、K 做旋转（乘性、复数乘法），不碰 embedding，也不碰 V。
- 「sinusoidal 能外推，所以它比 learned 好很多。」原论文确实以「可能外推到更长序列」为理由选了 sinusoidal，但 ALiBi 论文的实测是：sinusoidal 模型训 512 只能稳定外推几十个 token（约到 $L+20$ 到 $L+50$）就开始退化。理论性质成立，工程收益不成立。
- 「ALiBi 是相对位置编码，所以它也有一个相对位置的参数表。」ALiBi 没有任何可学参数，斜率 $m$ 是人手设定的常数，且实验表明改成可学反而外推更差。

## 原理与推导

### 为什么 self-attention 天生不知道顺序

一个 attention head 做的事是：对输入 $\{x_1,\dots,x_n\}$ 先线性投影得到 $q_i,k_i,v_i$，再按 $\mathrm{softmax}(q_i^\top k_j/\sqrt{d})$ 加权求和 $\sum_j a_{ij} v_j$。整个计算里**没有一处引用下标 $i,j$**——只有 $x_i$ 与 $x_j$ 的内容。所以对任意置换矩阵 $P$：

$$
\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)
$$

（严格说，这里指的是不带因果掩码的 attention；带上因果掩码后顺序信息并非完全为零，见「常见追问」里的 NoPE 一条。）

也就是说「猫追狗」和「狗追猫」在模型眼里是同一个多重集合，输出只差一个同样的置换。位置信息必须由外部注入。三条技术路线分别是：改输入（加在 embedding 上）、改 attention 的计算（加在 logit 上）、改 Q/K 本身（乘性旋转）。

### sinusoidal：固定频率、绝对位置、加在 embedding 上

原始 Transformer 用一组不同频率的正余弦波，给位置 $pos$ 生成一个 $d$ 维向量，第 $2i$、$2i+1$ 维分别是：

$$
\mathrm{PE}(pos,2i)=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
\mathrm{PE}(pos,2i+1)=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

这个向量与 token embedding **逐元素相加**后送进第一层。三个要点：

1. **不同维度 = 不同波长。** 第 $i$ 对维度（$i=0,1,\dots,d/2-1$，对应第 $2i$、$2i+1$ 维）的角频率是 $\omega_i = 10000^{-2i/d}$，波长 $\lambda_i = 2\pi/\omega_i = 2\pi\cdot 10000^{2i/d}$。$i=0$ 时 $\omega_0 = 1$（base 10000 的指数为 0），波长 $\approx 6.28$，能分辨相邻 token；$i$ 越大波长越长，最后一对维度的波长可达 $2\pi\times 10^4$ 量级（$d=128$ 时约 $5.44\times10^4$）。很像二进制计数：低位快、高位慢。
2. **相对位置可线性表示。** 记 $\phi_i = pos\cdot\omega_i$，对任意固定偏移 $k$：

$$
\begin{aligned}
\mathrm{PE}(pos+k,2i)   &= \sin(\phi_i + k\omega_i) = \sin\phi_i\cos(k\omega_i)+\cos\phi_i\sin(k\omega_i)\\
\mathrm{PE}(pos+k,2i+1) &= \cos(\phi_i + k\omega_i) = \cos\phi_i\cos(k\omega_i)-\sin\phi_i\sin(k\omega_i)
\end{aligned}
$$

写成矩阵就是 $\mathrm{PE}(pos+k)=M_k\,\mathrm{PE}(pos)$，其中 $M_k$ 是块对角正交矩阵，第 $i$ 个 2×2 块是 $\begin{pmatrix}\cos k\omega_i & \sin k\omega_i\\ -\sin k\omega_i & \cos k\omega_i\end{pmatrix}$（即在该 2 维子空间上旋转 $-k\omega_i$），$M_k$ **与 $pos$ 无关**。所以「位置 $pos$ 和 $pos+k$ 的关系」对任意 $pos$ 都是同一个线性映射，模型原则上可以学一次、用到处——这正是原论文认为它能外推的依据。
3. **但实测外推很弱。** 上面只说明存在这样的线性映射，没说模型的其它权重（Q、K 投影）会对没见过的位置区间保持有效。$M_k$ 里的相位 $k\omega_i$ 在 $k$ 超过训练长度后进入模型从未见过的取值组合，加上加法注入把位置和语义混在同一批维度里，实测性能在 $L+20$ 到 $L+50$ 之后就走平再退化（ALiBi 论文对 $L=512$ 与 $L=1024$ 两组都测到这个现象）。

### learned absolute：把位置当成词一样去学

BERT 与 GPT-2 的做法是把位置 id 也送进一张 embedding 表：$W_{pos}\in\mathbb{R}^{L_{max}\times d}$，输出与 token embedding 相加。BERT 的 $L_{max}=512$、GPT-2 是 1024。

- 好处：表达完全自由，位置之间的相似性由数据决定，不预设任何波形。
- 代价：表是**有限且稠密占用参数**的；$pos \ge L_{max}$ 时查表直接越界，模型连一个「凑合」的向量都拿不到。外推能力为零，这是结构性的，不是调参能救的。
- 与 sinusoidal 一样，位置是**加法**注入的：位置信息与语义共享同一批维度，两者在同一向量里竞争。

### 相对位置：T5 bias 与 ALiBi 把位置挪到 attention 里

第三类方案不再改输入，而是改 attention score 的计算，并且让偏置只依赖距离 $i-j$。

**T5 bias**：给每个距离 $k$ 配一个**可学习**的标量 bias（距离超过一定值后共享同一个桶），加在 logit 上。它被证明能外推（$L=512$ 时可外推到 $k=600$ 左右），但代价是训练至少慢一倍：ALiBi 论文复现时的对照是 $L=1024$ 的 sinusoidal 模型 28.5k words/s，而 $L=512$ 的 T5 bias 模型只有 14.4k words/s。

**ALiBi**（Attention with Linear Biases）把「可学习的 bias 表」换成一条**固定的直线**：

$$
\mathrm{softmax}\!\left(\mathbf{q}_i\mathbf{K}^\top + m\cdot[-(i-1),\dots,-2,-1,0]\right)
$$

即在第 $i$ 个 query 对第 $j$ 个 key 的 logit 上加 $m\cdot(j-i) = -m|i-j|$（因果掩码下 $j \le i$，所以是纯惩罚）。要点：

- **每个 head 一个斜率 $m$**，训练前设定、不学习。头与头的区别变成「惩罚增长多快」：$m$ 大的头几乎只看最近几个 token，$m$ 小的头能看很远。这就是论文说的 recency inductive bias。
- **斜率取几何序列**：$n$ 个头时从 $2^{-8/n}$ 开始、以同一比值为公比，即 $m_k = 2^{-8k/n},\ k=1..n$。8 头得 $\{2^{-1},2^{-2},\dots,2^{-8}\} = \{0.5,0.25,\dots,0.0039\}$；16 头得 $\{2^{-0.5},2^{-1},\dots,2^{-8}\}$，相当于对 8 头那组斜率取相邻两项的几何平均。取 $d_k$ 缩放**不**作用于这个 bias（论文脚注明确说明）。
- **零运行时开销**：实现上就是把 bias 直接加进 causal mask 矩阵里（mask 从 $L\times L$ 变成 $n\times L\times L$），不增加任何算子；相对同长度的 sinusoidal 基线，显存开销 0–0.7%，论文里最大观察到约 100 MB 的额外 mask 显存（$L=1024$ 与 $3072$ 那组设定）。
- **不学参数反而更好**：论文试过把斜率设为可训练，外推结果明显变差（还慢 3%），而且直接用指数分布随机采样斜率在某些情况下也能用（只是方差较大），说明该方法的鲁棒性来自「线性 + 单调」这个先验，而非某组精确数值。

### RoPE：不加、不改 logit，而是旋转 Q 和 K

RoPE 的思路与前两类都不同：位置不进向量、不进 logit，而是**乘在 Q、K 上**。先把 $d$ 维向量按 2 维一组切开（共 $d/2$ 组），第 $i$ 组用角频率

$$
\theta_i = 10000^{-2(i-1)/d},\qquad i=1,\dots,d/2
$$

在位置 $m$ 上对第 $i$ 组做 2 维旋转：

$$
R_m^{(i)}=\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}
$$

于是 $\tilde{\mathbf{q}}_m = R_m\mathbf{q}_m$、$\tilde{\mathbf{k}}_n = R_n\mathbf{k}_n$（$R_m$ 是块对角矩阵）。attention score 变成：

$$
\tilde{\mathbf{q}}_m^\top\tilde{\mathbf{k}}_n
= (R_m\mathbf{q}_m)^\top(R_n\mathbf{k}_n)
= \mathbf{q}_m^\top R_m^\top R_n\,\mathbf{k}_n
= \mathbf{q}_m^\top R_{n-m}\,\mathbf{k}_n
$$

第二步用了 $R$ 是正交矩阵（$R_m^\top = R_m^{-1} = R_{-m}$，旋转矩阵的转置等于反向旋转）。最后结果只含 $n-m$，绝对位置 $m$、$n$ 被消掉——相对位置是**乘性旋转的副产品**，不需要显式构造相对位置表。

三个必须记住的细节：

- **不加在 embedding 上，也不加在 V 上。** V 只被 attention 权重加权求和，不参与 score 计算；位置信息若混进 V，反而会让每层输出带上绝对位置痕迹，ALiBi 论文推测「把位置隔离在 Q、K 里」对长度外推有利。RoPE 每一层都对 Q、K 重做旋转，而不是只在底层注入一次。
- **不增加可学参数。** 频率是固定公式，角度由位置决定，参数量与 sinusoidal 一样是 0。
- **配对方式不影响数学。** 原论文按相邻维配对 $(q_1,q_2),(q_3,q_4),\dots$；LLaMA、Mistral 等实现按前后半配对 $(q_i,q_{i+d/2})$。两者只差一个维度置换，等价。
- **频率 base 是长上下文技术的主旋钮。** 因为 $\theta_1=b^{0}=1$ 与 base 无关，最快那一对维度的波长恒为 $2\pi$；base 拉大的只是低频端——最慢的一对维度波长 $2\pi\,b^{(d-2)/d}$ 在 $b=10000$、$d=128$ 时约 $5.44\times10^4$，换成 LLaMA 3 用的 $b=500000$（Qwen2.5 系用 $10^6$）后变成约 $2.56\times10^6$。低频周期变长，远距离的相位不再绕圈混叠、长距离分辨率提高；代价是这些维度的相邻位置相位差变小（局部区分变弱），而且与旧 base 下学到的频率分布不再匹配，通常要配合继续预训练或外推微调。YaRN、NTK-aware scaling、Position Interpolation 都是在动这组频率（见 [[llm-internals-08]]）。

### 演进对比

| 方案 | 加在哪 | 可学参数 | 相对位置 | 超出训练长度 | 代表模型 |
| --- | --- | --- | --- | --- | --- |
| sinusoidal | 与 embedding 相加（仅底层） | 无 | 线性可表示，非直接 | 理论可外推，实测几十 token 后退化 | 原始 Transformer、早期 fairseq 系 |
| learned absolute | 与 embedding 相加（仅底层） | $L_{max}\times d$ 一张表 | 无 | 结构性失效，$pos\ge L_{max}$ 越界 | BERT（512）、GPT-2（1024）、GPT-3 |
| T5 relative bias | attention logit 加可学 bias | 每层每 head 一张距离 bias 表 | 直接（按距离分桶） | 可外推（$L=512$ 到 $k\approx600$），但训练慢一倍 | T5 |
| ALiBi | attention logit 加 $-m\lvert i-j\rvert$ | 无（斜率固定） | 直接（线性） | 强，WikiText-103 上到 $3L$ 仍在改善，$L=512$ 时 $L_{valid}>12$k 仍有改善 | BLOOM、MPT 等 |
| RoPE | 每层对 Q、K 做旋转 | 无 | 直接（内积只含 $n-m$） | 原版一般，配 YaRN/NTK 后可大幅扩展 | LLaMA 系、Qwen、DeepSeek、Mistral、Gemma |

## 数值与代码验证

### 复算 1：sinusoidal 的两条性质

$d=128$、base 10000 时，$i=0$（第一对维度）的 $\omega=10000^{0}=1$、波长 $2\pi\approx6.28$；最后一对 $i=63$ 的 $\omega=10000^{-2\times63/128}=10000^{-0.9844}\approx1.155\times10^{-4}$，波长 $\approx5.44\times10^4$。这个量级差带来完全不同的行为：第一个平面每前进一个 token 相位就走 $1$ rad（$\approx57.3^\circ$），走满四分之一波长只需 $\approx1.57$ 个 token，所以它能区分每一个相邻位置；第 63 个平面每前进一个 token 相位只走 $1.155\times10^{-4}$ rad，要积累到四分之一波长需要 $\approx1.36\times10^4$ 个 token，因此在几十到几千 token 的尺度上它几乎不变，只编码宏观位置。

线性关系逐项验证（取 $k\omega_i=\pi/4$）：$\sin(\phi+\pi/4)=0.7071(\sin\phi+\cos\phi)$、$\cos(\phi+\pi/4)=0.7071(\cos\phi-\sin\phi)$，与旋转矩阵 $M_k$ 的展开一致，且式中不含 $pos$。

### 复算 2：ALiBi 的斜率序列

按 $m_k=2^{-8k/n}$ 算：

| head 数 $n$ | 首项 | 公比 | 末项 | 序列 |
| --- | --- | --- | --- | --- |
| 8 | $2^{-1}=0.5$ | 0.5 | $2^{-8}=0.00390625$ | 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625 |
| 16 | $2^{-0.5}\approx0.70711$ | $\approx0.70711$ | 0.00390625 | 0.70711, 0.5, 0.35355, 0.25, 0.17678, 0.125, 0.08839, 0.0625, 0.04419, 0.03125, 0.02210, 0.01563, 0.01105, 0.00781, 0.00552, 0.00391 |

16 头序列里每隔一项取出的子序列（第 2、4、6… 项：0.5, 0.25, 0.125, …）与 8 头序列逐项重合，这正是「把 8 头相邻两项做几何平均、再插进中间」的预期结果。注意**两端都被钉住**：$n$ 变化时首项 $\to 1$、末项恒为 $2^{-8}$，所以换模型规模不需要重新调斜率，这是 ALiBi 论文强调的「一套斜率跨数据集、跨模型尺寸通用」。

### 复算 3：RoPE 的数值例子

取 $d=2$ 的最简情形，$\mathbf{q}=\mathbf{k}=[1,0]$、$\theta=\pi/4$。$R_m\mathbf{q}=[\cos m\theta,\sin m\theta]$，于是 score $=\cos m\theta\cos n\theta+\sin m\theta\sin n\theta=\cos((n-m)\theta)$，直接只依赖距离。实测：

- $(m,n)=(1,2)$：score $=0.707107$
- $(m,n)=(2,3)$：score $=0.707107$（绝对位置整体平移 1，score 不变）
- $(m,n)=(5,17)$：score $=-1.000000$，因为 $(n-m)\theta=3\pi$

与源文的 $(1,2)$、$(2,3)$ 两组都得 $0.707$ 一致。

```python
import math, torch

# 1) RoPE：二维情形下 score 只依赖相对距离
def rot(m, theta):
    c, s = math.cos(m * theta), math.sin(m * theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

th = math.pi / 4
q = torch.tensor([1.0, 0.0], dtype=torch.float64)
k = torch.tensor([1.0, 0.0], dtype=torch.float64)
for m, n in [(1, 2), (2, 3), (5, 17)]:
    lhs = (rot(m, th) @ q) @ (rot(n, th) @ k)   # (R_m q)·(R_n k)
    rhs = q @ (rot(n - m, th) @ k)              # q·(R_{n-m} k)
    assert torch.allclose(lhs, rhs), (m, n)
    print(m, n, f"{lhs.item():.6f}", f"cos((n-m)θ)={math.cos((n-m)*th):.6f}")

# 2) RoPE 不改变向量长度（正交性）
v = torch.randn(64, dtype=torch.float64)
theta = 10000.0 ** (-2 * torch.arange(32, dtype=torch.float64) / 64)
ang = 137 * theta                      # 位置 137、每对维度一个角度
c, s = torch.cos(ang), torch.sin(ang)
x, y = v[0::2], v[1::2]
rotated = torch.stack([x * c - y * s, x * s + y * c], dim=1).flatten()
print("||v|| =", f"{v.norm():.6f}", " ||RoPE(v)|| =", f"{rotated.norm():.6f}")

# 3) sinusoidal 的线性关系 PE(pos+k) = M_k PE(pos)
pos, k, dm = 7, 13, 128
wi = 10000.0 ** (-2 * torch.arange(dm // 2, dtype=torch.float64) / dm)
pe = lambda p: torch.stack([torch.sin(p * wi), torch.cos(p * wi)], dim=1).flatten()
ck, sk = torch.cos(k * wi), torch.sin(k * wi)
mk_pe = torch.stack([pe(pos)[0::2] * ck + pe(pos)[1::2] * sk,
                     pe(pos)[1::2] * ck - pe(pos)[0::2] * sk], dim=1).flatten()
print("max |PE(pos+k) - M_k PE(pos)| =", f"{(pe(pos + k) - mk_pe).abs().max():.2e}")

# 4) ALiBi 斜率：n 头时从 2^{-8/n} 起、同值为公比
def alibi_slopes(n):
    r = 2.0 ** (-8.0 / n)
    return [r ** (i + 1) for i in range(n)]
print("8 heads :", [f"{m:.6f}" for m in alibi_slopes(8)])
print("16 heads:", [f"{m:.6f}" for m in alibi_slopes(16)])
```

### 复算 4：论文的关键实验数字（延伸来源）

ALiBi 论文（1.3B 参数、CC100+RoBERTa 语料）报出的对照：

| 结论 | 数字 | 口径 |
| --- | --- | --- |
| ALiBi 训 1024 vs sinusoidal 训 2048 | 两者在 2048 长度上 perplexity 相同 | 测试序列长度 2048 |
| ALiBi 的效率优势 | 训练快 11%、省显存 11% | 相同 perplexity 下的对照 |
| ALiBi 相对同长度 sinusoidal 的开销 | 显存 +0–0.7%，运行时无额外算子 | 主要来自 $n\times L\times L$ 的 mask |
| WikiText-103 上 $L=512$ 的 ALiBi | 18.40（外推到 $L_{valid}=3072$）| 非重叠推理口径；优于 $L=3072$ 的 sinusoidal（18.67±0.24）|
| 训练效率对比 | $L=512$ 的 ALiBi 训练吞吐是 $L=3072$ 的 sinusoidal 的 1.84 倍 | 训练速度（words/s），perplexity 反而更好 |
| 外推上限 | WikiText-103 上至少到 $\approx3L$ 仍在改善（$L=512$ 的模型在 $L_{valid}>12$k 时仍改善）；CC100+RoBERTa 上最佳点在 $\approx2L$（$L=512$ 在 1012、$L=1024$ 在 2024 取得最优 perplexity）| 与 sinusoidal 的 $L+20 \sim L+50$、rotary 的 $L+100 \sim L+200$ 对比 |

这里要区分口径：WikiText-103 的 perplexity 是「训 512、在 3072 长度上评测」的**外推**结果，而 1.3B 的 11% 是「达到同等 perplexity 时的训练成本」对比，两组数字不能混着说。

## 常见追问

- **追问**：RoPE 和 sinusoidal 都用了 10000 这个常数，它们是一回事吗？
  - 要点：频率集合相同（同一组 $10000$ 的等比次幂，只是下标起点不同），但用法完全不同：sinusoidal 是把 $\sin/\cos$ 组成向量**加到 embedding** 上，只做一次；RoPE 是把同一组频率当作**旋转角**去乘 Q、K，每一层都做。因此 sinusoidal 的 score 里位置是以加法形式进入的，需要模型额外学出「距离」的概念；RoPE 的 score 直接就是 $R_{n-m}$ 的二次型，相对距离是硬编码在结构里的。
- **追问**：既然 RoPE 这么好，为什么还有 ALiBi？
  - 要点：两者都是相对位置、都零可学参数，差别在表达能力和成本。ALiBi 的偏置是**固定的单调线性**，只有 $n$ 个自由度（斜率），表达力弱但先验强、结构上不可能越界，外推最稳；RoPE 的相位是**周期性**的，远距离会出现相位混叠（两个距离差一个波长时旋转量相同），所以原版 RoPE 的天然外推确实弱于 ALiBi，但它更灵活，能通过改频率（YaRN/NTK）把外推做上去——这是 ALiBi 难以做到的。选型上：要极致简单和训练稳定性选 ALiBi，要长上下文可调、要生态兼容选 RoPE。
- **追问**：为什么 ALiBi 的斜率不能学？
  - 要点：论文试过，可学斜率外推明显变差（还慢 3%）。原因是外推需要的先验是「惩罚随距离单调增长且量级合适」，让模型在训练长度内自由拟合会把每个距离的 bias 调成对训练分布最优的任意形状，一旦超出训练长度就没有可依据的规律。固定几何序列把「线性 + 单调 + 多尺度」硬编码进去，才换来跨长度、跨数据集的通用性。
- **追问**：把 RoPE 的频率 base 从 10000 提到 500000 到底改了什么？
  - 要点：改的是**低频端**的波长。$b=10000$、$d=128$ 时最慢的一对维度波长约 $5.44\times10^4$，$b=500000$ 时约 $2.56\times10^6$；最快的一对维度波长恒为 $2\pi$（$\theta_1=b^0=1$），完全没动。低频周期拉长后，远距离的相位不再快速绕圈（混叠变慢），所以能撑更长上下文；代价是这些维度的相邻位置相位差变小、局部区分变弱，并且与旧 base 下学到的频率分布不匹配，一般要配合继续预训练或外推微调。PI 是另一条路：把位置索引整体压回训练区间，等价于所有 $\theta_i$ 乘 $1/s$，一刀切压缩全部频率；只有 NTK-aware 才像改 base 那样几乎不动高频、只压低频。
- **追问**：不加任何位置编码行不行？
  - 要点：因果掩码本身泄漏了一点顺序信息——位置 $i$ 的 token 能看到的历史长度是 $i+1$，而位置 $i+1$ 能看到 $i+2$ 个，这个「可见集合」的大小是唯一的，因此 decoder-only 模型在理论上可以借因果掩码隐式推断绝对位置。近年的实验（NoPE）报告了 transformer 在部分任务上不加显式位置编码也能学到位置、并且长度外推表现不错。这是**实验性结论**：它有任务与规模依赖，训练动态也更难调，目前不能当作「位置编码可以省掉」的一般性结论，工程上仍然默认用 RoPE。
- **追问**：这道题的演进顺序是线性的吗？
  - 要点：不是严格线性，而是两条轴同时演进。一条轴是**位置信息放在哪里**：输入侧加法（sinusoidal / learned）→ attention logit（T5 bias / ALiBi）→ Q、K 本身（RoPE）。另一条轴是**绝对 vs 相对**：绝对（sinusoidal / learned）→ 相对（T5、ALiBi、RoPE）。RoPE 之所以胜出，是因为它同时占住了两条轴的终点，并且零参数、只做逐元素旋转，与 KV cache 和现有 attention kernel 天然兼容（动态缩放下的 cache 细节见 [[llm-internals-08]]）。

## 相关题目

- [[llm-internals-08]]：RoPE 的完整推导与长上下文外推技巧（Position Interpolation、NTK-aware、YaRN）。本题只讲到「内积只依赖相对位置」，扩展方法全在那道题。
- [[llm-internals-01]]：scaled dot-product attention 的数学。位置编码改的是 logit 加在哪一步，得先清楚 $q^\top k/\sqrt{d_k}$ 的方差性质。
- [[llm-internals-16]]：一次前向传播的全流程。位置编码在流程里的插入点（底层加性 vs 每层乘性）在那道题里有完整的张量形状视角。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《Positional Embeddings in LLMs》，2026-01-08，<https://outcomeschool.substack.com/p/positional-embeddings-in-llms>
- Amit Shekhar (Outcome School)，《Math Behind RoPE (Rotary Position Embedding)》，2026-04-23，<https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding>
- Ofir Press, Noah A. Smith, Mike Lewis，《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》（延伸），2021-08-24（v2 2022-04-22），<https://arxiv.org/abs/2108.12409>
- 补充来源：Ashish Vaswani 等，《Attention Is All You Need》，2017，<https://arxiv.org/abs/1706.03762>（sinusoidal 公式与选择理由）
- 补充来源：Jianlin Su 等，《RoFormer: Enhanced Transformer with Rotary Position Embedding》，2021，<https://arxiv.org/abs/2104.09864>（RoPE 原始论文）

第 3 节的 RoPE 定义与相对位置推导、第 4 节复算 3 的数值例子来自 Outcome School 的 RoPE 博客（另一篇只提供「为什么需要位置编码」的动机）；sinusoidal 公式与它的两条性质来自 Attention Is All You Need，复算 1 的波长与相位差按公式自行复算；复算 2 的斜率几何序列、复算 4 的 ALiBi 数字、外推对比与「不学习斜率」的实验结论来自 arXiv:2108.12409，正文已在该处标注。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
