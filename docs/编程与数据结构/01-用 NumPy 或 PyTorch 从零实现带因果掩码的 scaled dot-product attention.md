---
type: question
id: coding-01
topic: 编程与数据结构
order: 1
question: 用 NumPy 或 PyTorch 从零实现带因果掩码的 scaled dot-product attention。
question_en: Implement causal-masked scaled dot-product attention from scratch in NumPy or PyTorch.
asked_at: [Anthropic, Google DeepMind, Amazon（AWS）]
level: 进阶
tags: [attention, 实现题, 数值稳定, 因果掩码]
sources:
  - title: 注意力背后的数学：Q、K 与 V
    url: https://outcomeschool.com/blog/math-behind-attention-qkv
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Attention 中的因果掩码
    url: https://outcomeschool.com/blog/causal-masking-in-attention
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Attention Is All You Need（延伸）
    url: https://arxiv.org/abs/1706.03762
    author: Vaswani et al. (NeurIPS 2017)
    published: 2017-06-12
  - title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）
    url: https://arxiv.org/abs/2205.14135
    author: Dao et al. (NeurIPS 2022)
    published: 2022-05-27
related: [coding-02, coding-03, llm-internals-01, llm-internals-04, llm-internals-05]
updated: 2026-09-28
---

## 一句话答案

> 三步：先算 $S = QK^\top / \sqrt{d_k}$，再把 $j > i$ 的位置在 **softmax 之前**置为负无穷，最后做 softmax 并乘 $V$。
> 形状按 $(B, H, T, d)$ 组织，输出 $(B, H, T, d_v)$，权重是 $(B, H, T, T)$——带上 batch 与 head 维度不是炫技，而是为了能和 `F.scaled_dot_product_attention` 逐元素对齐，也为了让「转错了轴」这类错误直接暴露成形状不匹配。
> 两个必须写对、也最常被问的细节：softmax 要减行最大值（数值稳定），掩码要在 softmax 之前（放到之后乘 0 会破坏归一化，行和不再等于 1）。

## 面试官在考什么

- **是不是先把契约说清楚。** 实现题的第一步不是写代码，是说清形状、dtype、是否返回权重、$d_k$ 与 $d_v$ 是否必须相等。开口就问「你的输入形状是什么」的人，通常真写过。
- **掩码的位置。** 掩码在 softmax 前还是后，是这类题区分度最高的一点。答「在 softmax 之后乘个 0/1 矩阵」的人，写出来的东西能跑、数值也「看起来像注意力」，但每一行都不再是概率分布。
- **数值稳定与负无穷的取舍。** 知不知道为什么减最大值；知道 `-inf` 在「整行都被掩掉」时会产生 NaN，而有限大负数会产生**静默错误**（整行变成均匀分布），并能说清两种失效模式哪个更好排查。
- **能不能算清楚代价。** 时间 $O(T^2 d)$、显存 $O(T^2)$ 要能当场报数：$T=8192$、$H=32$、$B=1$、bf16 时那个 $T \times T$ 矩阵是 4 GiB，正好是单层 KV cache（同序列 32 MiB 量级）的上百倍。
- **怎么证明自己写对了。** 不是「我跑通了」，而是「我和 `F.scaled_dot_product_attention` 对齐到 $3.6 \times 10^{-7}$，并且断言了行和为 1、上三角全 0」。这是本题真正的工程分。

常见错误答案：

- 「注意力就是 $QK^\top$ 然后 softmax，掩码就是把未来位置乘 0。」——乘 0 那一步如果发生在 softmax 之后，行和立刻不等于 1，输出被整体缩小。
- 「除以 $\sqrt{d_k}$ 是为了防止梯度消失。」——缩放的作用是控制 softmax 输入的方差（回到 1 量级），避免分布过度 one-hot；把因果掩码写成上三角、把缩放写成 $1/d_k$ 属于同一类「看过但没写过」的错误。

## 原理与推导

### 1. 接口契约

```python
def sdpa(q, k, v, causal=True, mask_value=None, return_weights=True, q_offset=0):
    """q: (B,H,Tq,dk)  k: (B,H,Tk,dk)  v: (B,H,Tk,dv)
    返回 out: (B,H,Tq,dv)，attn: (B,H,Tq,Tk)
    q_offset: 本批 query 的起始绝对位置（带 KV cache 的 decode 时 = Tk - Tq）"""
```

- **为什么默认带 $B$ 与 $H$。** 真实调用点是每层每头一次，PyTorch 的 `scaled_dot_product_attention`、`nn.MultiheadAttention` 内部都是 $(B, H, T, d)$。用二维矩阵写「自注意力」在 $T$ 很小的时候也能过测试，一旦换成多头就会分成两种错：$K^\top$ 真把两维转错，$(B, H, T, d)$ 与 $(B, T, H, d)$ 在 $H \ne T$ 时形状对不上、当场抛错；真正难查的是该 `transpose` 的地方用 `reshape`/`view` 顶替——$(B, T, H, d)$ 直接 `reshape` 成 $(B, H, T, d)$ 元素总数不变、不报错，只是按另一种内存顺序静默重解释数据，数字全错。这是实现题里最常见的隐性 bug。
- **$d_k$ 与 $d_v$ 不必相等。** $q,k$ 必须同 $d_k$，$k,v$ 必须同 $T_k$；输出最后一维是 $d_v$。工程上都取 $d_k = d_v = d_{\text{head}}$，但契约里分开写能避免「$V$ 的维度跟着 $K$ 走」的思维定势。
- **$T_q$ 与 $T_k$ 可以不等。** 训练时 $T_q = T_k = T$；带 cache 的 decode 时 $T_q = 1$、$T_k$ 是已生成长度；$T_q > 1$ 的并行生成则要传 $q_offset$，否则掩码会错位。

### 2. 三步推导

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

其中 $M_{ij} = 0$（$j \le i$）、$M_{ij} = -\infty$（$j > i$）。逐项展开：

1. **打分**：$S_{ij} = q_i \cdot k_j / \sqrt{d_k}$，表示「第 $i$ 个 query 有多想看第 $j$ 个 key」。
2. **掩码**：把 $j > i$ 的 $S_{ij}$ 置为 $-\infty$。选择加性掩码而不是乘性掩码，正是因为乘性掩码只能作用于 softmax 之后的概率。
3. **归一化与加权**：$P_{i:} = \text{softmax}(S_{i:})$，$o_i = \sum_j P_{ij} v_j$。

### 3. 为什么除以 $\sqrt{d_k}$

设 $q, k$ 各维独立、均值 0、方差 1，则 $q \cdot k = \sum_{l=1}^{d_k} q_l k_l$ 的方差是 $d_k$（每项方差 1，$d_k$ 项独立相加）。实测：$d_k = 128$ 时 20 万个点积的经验方差是 127.525。方差为 $d_k$ 意味着标准差 $\sqrt{d_k}$，$d_k$ 越大分数越极端，softmax 越接近 one-hot、梯度越小。除以 $\sqrt{d_k}$ 把方差拉回 1，与 $d_k$ 解耦。

### 4. 掩码为什么必须在 softmax 之前

softmax 是**逐行归一化**：$P_{ij} = e^{S_{ij}} / \sum_{j'} e^{S_{ij'}}$。若先在全体 $j$ 上归一化、再把未来位置乘 0，第 $i$ 行的和变成 $\sum_{j \le i} P_{ij} < 1$，输出被整体缩小，且缩小倍数随 $i$ 变化（第 1 行只剩 1 项，损失最大）。负数写在 softmax 之前时，$e^{-\infty} = 0$ 精确地把被掩位置从分母里剔除，分母恰好是所有**可见**位置的指数和。

### 5. 两个数值细节

- **减行最大值**：$e^{S_{ij} - m_i}$，$m_i = \max_j S_{ij}$，指数永远 $\le 0$，不会溢出；softmax 对行内整体平移不变，所以结果严格等价。分数加 1000 时权重与不加相差 $2.0 \times 10^{-14}$（float64）。
- **$-\infty$ 还是 $-10^9$**：有可见 key 时两者等价（都被 softmax 压成精确的 0）。区别只在**整行被掩掉**（padding 后的 query 位置、掩码方向写反）：`-inf` 会让 $m_i = -\infty$、$S_{ij} - m_i$ 变成 NaN（响亮地失败），有限大负数会让整行 $e^{0} = 1$ 归一化成均匀分布（静默地失败）。生产上宁可选 $-\infty$ 并在训练循环里断言没有 NaN，也不要一个「看起来正常」的均匀注意力。PyTorch 侧的额外坑：`masked_fill(~mask, -1e9)` 在 fp16 上直接抛 `RuntimeError: value cannot be converted to type c10::Half without overflow`，bf16 会静默舍入成 $-998244352$，所以按 dtype 取 `torch.finfo(dtype).min` 最稳。

### 6. 复杂度与显存

- 时间：$QK^\top$ 与 $PV$ 各 $2T^2 d$ FLOPs，合计 $4T^2 d$；对 $T$ 是 $O(T^2 d)$ 的二次项，这是长上下文的核心成本。
- 显存：标准实现必须**物化** $(B,H,T,T)$ 的分数矩阵与概率矩阵，$O(T^2)$。$T = 8192$、$H = 32$、$B = 1$ 时元素数是 $32 \times 8192^2 = 2{,}147{,}483{,}648$，bf16 正好 4 GiB，fp32 是 8 GiB。
- roofline：同一规模下 $d = 128$、bf16 的算术强度是 $1.0995\ \text{TFLOP} / 16.125\ \text{GiB} \approx 63.5$ FLOP/byte，低于 H100 SXM5 的拐点 $989/3.35 \approx 295$ FLOP/byte 约 4.65 倍——attention 是带宽一侧的算子，这正是 FlashAttention 用分块 + online softmax 把 $T \times T$ 矩阵留在 SRAM、从不落 HBM 的原因（[[llm-internals-05]]）。

### 7. 推理时到底要不要掩码

`is_causal=True` 在训练和并行生成（一次前向算 $T_q > 1$ 个位置）时必需；**单步 decode 且 $T_q = 1$ 时，这一个 query 就是最后一位，未来位置不存在，掩码是恒等操作**。带 KV cache 时若 $T_q > 1$，掩码必须按 $q\_offset = T_k - T_q$ 生成偏移下三角，不能用从 0 开始的 `tril`。这也是 [[coding-03]] 里最容易写错的一处。

## 数值与代码验证

以下所有数字都在本机真实运行取得（NumPy 2.2.6 / PyTorch 2.12.0，CPU 32 核 OpenBLAS 0.3.29 + RTX 5070 Laptop 8 GiB；脚本在 `.work/`）。

```python
import math
import numpy as np

def sdpa(q, k, v, causal=True, mask_value=None, return_weights=True, q_offset=0):
    B, H, Tq, dk = q.shape
    Bk, Hk, Tk, dk2 = k.shape
    assert (B, H, dk) == (Bk, Hk, dk2), f"q/k 形状不匹配: {q.shape} vs {k.shape}"
    assert v.shape[:3] == (B, H, Tk), f"v 的形状应为 (B,H,Tk,dv)，实际 {v.shape}"
    assert Tq <= Tk and q_offset + Tq <= Tk, "query 位置不能越过 key 长度"

    # ① 打分 + 缩放。用 @ 而不是 einsum：后者默认不走 BLAS（实测差 4 倍）
    s = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dk)

    # ② 因果掩码：在 softmax 之前把不可见位置置为负无穷
    if causal:
        qi = np.arange(q_offset, q_offset + Tq)[:, None]
        kj = np.arange(Tk)[None, :]
        neg = -np.inf if mask_value is None else mask_value
        s = np.where(kj <= qi, s, neg)

    # ③ 数值稳定 softmax（减行最大值）+ 加权求和
    m = s.max(axis=-1, keepdims=True)
    p = np.exp(s - m)
    p = p / p.sum(axis=-1, keepdims=True)
    out = p @ v
    return (out, p) if return_weights else out
```

固定种子的 $B=2$、$H=4$、$T=8$、$d=16$ 实跑：

```text
q/k/v   : (2, 4, 8, 16)
output  : (2, 4, 8, 16)   attn: (2, 4, 8, 8)
batch0/head0 前 3 行权重：
[[1.     0.     0.     0.     0.     0.     0.     0.    ]
 [0.5756 0.4244 0.     0.     0.     0.     0.     0.    ]
 [0.3807 0.5983 0.0209 0.     0.     0.     0.     0.    ]]
每行权重之和: min=1.000000000 max=1.000000000
严格上三角的最大绝对值: 0.000e+00
```

断言（全部通过）：输出形状 $(2,4,8,16)$；每行权重和 $= 1$（`atol=1e-6`）；上三角恒等于 0；第 1 行恰为 one-hot $[1,0,\dots]$；$T=1$ 时权重为 1 且输出严格等于 $V$；$T_q = 3$、$T_k = 8$、$q\_offset = 5$ 时每行可见 key 数是 6/7/8，且与完整序列前向的最后 3 行逐元素一致（差 $< 10^{-12}$）。

随机输入的「第二行」不一定是 0.5/0.5，因为它取决于 $q_1 \cdot k_0$ 与 $q_1 \cdot k_1$ 谁大。想看严格可检验的性质，把 $q,k$ 全取 1：所有分数相同，权重退化为 $1/i$，这是可以手算的解析基准。

| 行 $i$ | 权重 | 输出（$V$ 第 0 维取 $0 \ldots T-1$） |
| --- | --- | --- |
| 1 | `[1, 0, 0, 0, 0]` | 0.0000 |
| 2 | `[0.5, 0.5, 0, 0, 0]` | 0.5000 |
| 3 | `[1/3, 1/3, 1/3, 0, 0]` | 1.0000 |
| 4 | `[0.25, 0.25, 0.25, 0.25, 0]` | 1.5000 |
| 5 | `[0.2, 0.2, 0.2, 0.2, 0.2]` | 2.0000 |

第 2 行的两项**严格等于** `0.5`（不是近似），输出就是前 $i$ 个 $V$ 的均值。

与 PyTorch 对照（32 个随机张量元素级最大绝对误差，同一 $B=2,H=4,T=8,d=16$）：

| 路线 | 最大绝对误差 | 说明 |
| --- | --- | --- |
| float64 手写 vs torch float64 | $6.7 \times 10^{-16}$ | 浮点舍入量级，实现正确 |
| float32 手写 vs torch float32 | $3.6 \times 10^{-7}$ | 两者都是 fp32，差异来自算子顺序 |
| float16 手写 vs torch float16 | $2.0 \times 10^{-3}$ | fp16 eps 是 $2^{-10} \approx 9.8 \times 10^{-4}$ |
| fp32 手写 vs torch bf16 | $8.6 \times 10^{-3}$ | bf16 eps 是 $2^{-8} \approx 3.9 \times 10^{-3}$ |
| $q$ 放大 100 倍后的 bf16 差异 | $6.5 \times 10^{-2}$ | logits 变大后 softmax 接近 one-hot，bf16 的舍入被放大 |

**结论口径：fp32 对齐到 $10^{-7}$（与算子顺序有关的舍入），bf16 只能对齐到 $10^{-2}$ 量级——不是实现错了，是 dtype 的粒度。** 门禁里如果用一个绝对误差阈值同时卡 fp32 与 bf16，一定会误报，正确做法是按 dtype 分别设阈值（[[evaluation-04]] 的分层思路）。

三种错误实现的实测后果（同一批随机输入）：

| 错误 | 实测现象 |
| --- | --- |
| 掩码写在 softmax 之后（乘 0/1） | 上三角确实全 0，但前 3 行行和变成 `[0.0523, 0.0749, 0.1874]`，整批最小行和 0.00188；输出与正确实现最大差 2.178 |
| 掩码方向写反（留上三角） | 不报错，上三角有 224 个非零权重，下三角全 0，输出范数 21.31，数字「看起来正常」 |
| 缩放写成 $1/d_k$ | 权重最大差 0.505，分布明显更平坦（缩放不足），且随 $d_k$ 增大而恶化 |

显存实测（$B=1$、$H=8$、$d=64$、fp32，独立子进程读 `ru_maxrss`）：

| $T$ | 权重矩阵理论值 | 朴素实现峰值增量 | 分块实现峰值增量 | 耗时（朴素 / 分块） |
| --- | --- | --- | --- | --- |
| 1024 | 32 MiB | 96.4 MiB | 19.3 MiB | 79 / 153 ms |
| 2048 | 128 MiB | 393.0 MiB | 19.4 MiB | 310 / 574 ms |
| 4096 | 512 MiB | 1569.1 MiB | 17.4 MiB | 1202 / 2286 ms |

朴素实现的实际占用是矩阵本身的约 3 倍（分数、掩码后副本、概率各一份）；分块实现（online softmax，与 [[llm-internals-05]] 同一套递推）峰值与 $T$ 无关，$T=4096$ 时省 90 倍显存。但在这个 CPU 上分块反而更慢——把大 GEMM 拆成 $B_r \times B_c$ 的小 GEMM 后，OpenBLAS 的每次调用线程同步成本盖过了收益：$T=4096$ 时 1/4/8/32 线程下朴素耗时是 5213/1666/1213/1255 ms，分块是 5980/1889/3226/4284 ms，线程越多分块越差。**这是 CPU 的特性，不是 attention 的结论**：GPU 上瓶颈是 HBM 流量（H100 拐点 295 FLOP/byte，而 attention 只有 63.5），把 $T \times T$ 矩阵留在片上才划算。

GPU 实测（RTX 5070 Laptop 7.5 GiB，bf16，朴素实现与 `F.scaled_dot_product_attention` 的 flash 后端对照）：

| 规模（$B=1$、$d=128$） | 朴素实现（物化 $T \times T$） | SDPA flash 后端 | 显存比 / 耗时比 |
| --- | --- | --- | --- |
| $T=4096$、$H=8$ | 600 MiB、12.11 ms | 80 MiB、0.94 ms | 7.5× / 12.9× |
| $T=8192$、$H=8$ | 2280 MiB、54.97 ms | 136 MiB、4.45 ms | 16.7× / 12.4× |
| $T=8192$、$H=32$ | **OOM**（单个 4 GiB 张量分配失败） | 369 MiB、16.72 ms | 朴素实现根本跑不起来 |

同一台机器上 SDPA 的 `MATH` 后端（会把中间量升到 fp32）正好展示了同一件事的另一面：$T=8192$、$H=8$ 时峰值 5120 MiB、耗时 123.5 ms（比 flash 慢 27 倍），$T=8192$、$H=32$ 时申请 8.00 GiB 直接 OOM——8 GiB 正是 $32 \times 8192^2 \times 4$ B。

同一输入上，朴素实现与三种 SDPA 后端的最大差都是 $1.56 \times 10^{-2}$；后端之间也彼此不一致（$T=4096$、$H=8$、bf16 下 flash vs efficient 是 $3.9 \times 10^{-3}$，即恰好一个 bf16 ulp，flash vs math 与 efficient vs math 都是 $7.8 \times 10^{-3}$）。**同一个数学式、不同 kernel，bf16 下逐位对不齐是常态**，所以金标准测试必须用 fp32/fp64 参考值。

## 常见追问

- **追问**：为什么除以 $\sqrt{d_k}$，不是 $d_k$ 也不是别的？
  - 要点：$q \cdot k$ 的方差是 $d_k$（各维独立、方差 1 时），除以 $\sqrt{d_k}$ 让标准差回到 1，softmax 的输入不随 $d_k$ 漂移。$d_k = 128$ 时未缩放分数的标准差约 11.3，除以 128 会把分布压得过度平坦，除以 $\sqrt{128}$ 才回到 1 量级。实测 20 万个点积：$d_k=16/64/128$ 的经验方差是 16.09/63.89/127.53。
- **追问**：softmax 的数值稳定具体怎么写？
  - 要点：减行最大值再取指数，分母是同一基准下的指数和，数学上与朴素 softmax 恒等。不减最大值的代价是具体的：float32 的 `exp` 在自变量超过约 88.7 时溢出成 `inf`，实测在 float32、单头 $T=8$（下三角共 36 项）上把分数整体加 100（原本就有这种量级的 logits），这 36 项全变 `inf`、归一化后整行 NaN，而减最大值的那一版行和仍然是 1。掩码值上，`-inf` 在有可见 key 时与 $-10^9$ 等价，在整行被掩时产生 NaN（可检测），有限大负数则给出均匀分布（不可检测）。
- **追问**：不物化那个 $T \times T$ 矩阵，怎么算 attention？
  - 要点：分块 + online softmax。外层切 query 块、内层切 key/value 块，维护每行的 running max $m$、running sum $l$ 与未归一化累加量 $\tilde{O}$；$m$ 变大时把历史累加量整体乘 $e^{m_{\text{旧}} - m_{\text{新}}}$。实测 $T=96$、`block=32` 时与朴素实现的差是 $5.6 \times 10^{-16}$，$T=4096$ 时峰值内存从 1569 MiB 降到 17.4 MiB。
- **追问**：单步 decode 还需要掩码吗？
  - 要点：$T_q = 1$ 时不需要，那个 query 就是最后一个位置；训练和 $T_q > 1$ 的并行生成必须加。带 cache 时掩码要按 $q\_offset = T_k - T_q$ 生成偏移下三角，用 `tril(Tq, Tk)` 会错位。
- **追问**：`einsum` 和 `@` 有区别吗？
  - 要点：有，而且是数倍。同一个 $QK^\top$（$B=1,H=8,T=2048,d=64$，fp32）：`np.einsum('bhqd,bhkd->bhqk')` 671 ms / 6.4 GFLOPS，`optimize=True` 558 ms / 7.7 GFLOPS，`q @ k.transpose(0,1,3,2)` 166 ms / 25.9 GFLOPS。`einsum` 默认走的是朴素 C 循环而不是 BLAS。
- **追问**：这段代码直接上线还差什么？
  - 要点：至少差五层。(1) **不物化 $T \times T$**：$T=8192$、$H=32$ 时那个矩阵是 4 GiB，实测在 7.5 GiB 的卡上直接 OOM，生产必须走 FlashAttention / SDPA 的 flash 或 mem-efficient 后端。(2) **dtype 与累加精度**：bf16 输入要在 fp32 里累加，掩码值按 `finfo(dtype).min` 取，误差门禁按 dtype 分层设阈值。(3) **KV cache 与掩码偏移**：真实 decode 是 $T_q=1$ + 变长 cache + padding，掩码得按每个请求的实际长度生成（[[coding-03]]、[[inference-serving-03]]）。(4) **融合与后端选择**：`torch.compile` / 各家 kernel 会把 mask、softmax、dropout 融进一个算子，`is_causal=True` 让框架省掉物化掩码矩阵；不同后端在 bf16 下结果本就不同，所以金标准测试要用 fp32 或 fp64 的参考值。(5) **训练侧的额外约束**：dropout、反向对 $S$ 的重算、以及「不许有 NaN」的断言，都要一起进 CI。

## 公司变体

- **Anthropic**：偏工程实现与数值细节。公开研究输出里大量涉及 attention 内部结构（可解释性方向的 attention head 分析、attention sink 现象），所以常见问法会从「写出来」推到「你怎么知道中间量是对的」「掩码值取 $-\infty$ 还是有限负数、上线时怎么发现整行被掩」。准备时把断言与参考实现对齐这两件事讲成流程，比背公式有效。
- **Google DeepMind**：偏数学推导 + 结构实现。Transformer 出自 Google，公开模型卡与论文（如 Gemma 系列）里连 attention logit soft-capping 这类细节都会写出来，所以会追问 $1/\sqrt{d_k}$ 的方差来源、$q\_offset$ 掩码偏移、以及多头形状怎么和已有实现对齐。
- **Amazon（AWS）**：偏工程落地与硬件适配。公开技术输出集中在自研加速芯片与托管推理服务（Neuron SDK、SageMaker 一侧），因此更可能问「这段代码怎么在 bf16 下保持精度」「怎么和框架自带算子对齐误差」「上生产前用什么测试卡住回归」。

以上是依据该公司公开技术输出的侧重判断，不是对具体面试流程的描述。

## 相关题目

- [[llm-internals-01]]：scaled dot-product attention 与 $1/\sqrt{d_k}$ 的方差推导，是本题的数学前提。
- [[llm-internals-05]]：FlashAttention 的 IO 复杂度、online softmax 递推与分块大小，是本题「不物化 $T \times T$」那一问的展开。
- [[coding-02]]：把这里的两步走成多头，并改造成 GQA。
- [[coding-03]]：KV cache 与单步 decode，掩码偏移就出在这一题。
- [[llm-internals-04]]：MLA 用低秩 latent 压缩 KV，是同一族显存问题的另一条路。
- [[inference-serving-03]]：PagedAttention 管 KV cache 的显存分配，与 attention 算子内部的访存是两件事。
- [[evaluation-04]]：回归门禁的分层阈值思路，用来卡住本题的数值误差。

## 参考资料与归属

- [注意力背后的数学：Q、K 与 V](https://outcomeschool.com/blog/math-behind-attention-qkv)，Amit Shekhar（Outcome School）：$S = QK^\top$ 逐步手算、除以 $\sqrt{d_k}$ 的作用、softmax 后各行之和为 1 的口径，以及「缩放不改变分数的相对顺序」这一说法。
- [Attention 中的因果掩码](https://outcomeschool.com/blog/causal-masking-in-attention)，Amit Shekhar（Outcome School）：掩码矩阵、$-\infty$ 在 softmax 后恰好变成 0、掩码必须在 softmax 之前、以及不加掩码会造成信息泄漏的论述。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Vaswani et al.，2017-06-12（延伸）：scaled dot-product attention 与多头注意力的原始形式；「完全基于注意力、不含循环与卷积」的架构论断，以及 WMT 2014 En-De 28.4 BLEU（比当时最好结果含 ensemble 高 2 BLEU 以上）、En-Fr 41.8 BLEU、8 卡训练 3.5 天这些论文口径的数字。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，Dao et al.，2022-05-27（延伸）：tiling 减少 HBM 读写、IO 复杂度对一段 SRAM 容量最优的结论，以及实测 15%（BERT-large，序列 512，对 MLPerf 1.1 记录）、3×（GPT-2，序列 1K）、2.4×（Long Range Arena，1K–4K）的加速口径。
- H100 参数取自 NVIDIA 官方规格（bf16 dense 989 TFLOP/s、HBM3 3.35 TB/s、80 GB），拐点 295 FLOP/byte、4 GiB 权重矩阵、63.5 FLOP/byte 算术强度、320 KiB/token 的 LLaMA-3-70B KV cache 口径与 [[llm-internals-02]] 一致，均为按正文声明口径的复算结果；文中所有误差、耗时与峰值内存数字来自本机实跑（NumPy 2.2.6 / PyTorch 2.12.0）。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
