---
type: question
id: mistral-04
company: Mistral AI
topic: inference-serving
order: 4
question: 估算服务 Mistral 7B 所需的 KV cache 显存，并设计 sliding-window attention 所支持的滚动缓冲区缓存。
question_en: Estimate the KV-cache memory for serving Mistral 7B, and design the rolling-buffer cache that sliding-window attention enables.
asked_at: []
level: 进阶
tags: [数值推导, 系统设计, kv-cache, sliding-window-attention, gqa]
sources:
  - title: LLM 中的 KV Cache 是什么？
    url: https://outcomeschool.com/blog/kv-cache-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-03-27
  - title: Sliding Window Attention 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-sliding-window-attention-work
    author: Amit Shekhar (Outcome School)
    published: 2026-09-05
  - title: What is Grouped Query Attention (GQA) and Why Do LLMs Use It?
    url: https://outcomeschool.com/blog/grouped-query-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-22
  - title: Paged Attention in LLMs
    url: https://outcomeschool.com/blog/paged-attention-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-03-29
  - title: KV Cache Compression
    url: https://outcomeschool.com/blog/kv-cache-compression
    author: Amit Shekhar (Outcome School)
    published: 2026-09-12
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
related: [inference-serving-08, llm-internals-02, llm-internals-03, inference-serving-03]
updated: 2026-09-29
---

## 一句话答案

> 先把 config 钉死：Mistral 7B 是 32 层、32 个 Q head、8 个 KV head（GQA-8，即每组 4 个 Q head）、head_dim 128、hidden 4096、bf16。KV cache 每 token 是 $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = 128 KiB，即每 GiB 显存装 8192 个 token；滑动窗口 $W=4096$ 下单条序列的占用是常数 $2LH_{kv}d_{head}Wp = 512$ MiB，与上下文长度无关。
> 显存按四块报：权重 7.24B × 2 B ≈ 14.5 GB（13.5 GiB）、KV 池、每卡 1–3 GiB 工作区、5%–10% 的碎片与调度水位。一块 80 GiB 的卡扣掉权重 13.5、工作区 2.5、碎片 8 GiB 后剩约 56 GiB，能跑 112 条窗口满载的序列；同样 56 GiB 若不做窗口、保留 32k 全长（4 GiB/条）只剩 14 条。
> 滚动缓冲的落法是每层每个 KV head 一个长度 $W$ 的环形缓冲，绝对位置 $i$ 写进槽位 $i \bmod W$；掩码要同时挡未来 token、已滑出窗口的陈旧槽位和 prefill 早期没写过的槽位，而 K 必须先用绝对位置做 RoPE 再入槽。

## 面试官在考什么

- 能不能先把 config 钉死再算，并且用 KV 头数 $H_{kv}$ 而不是 Q 头数 $H_q$——这题的分差常常出在口径，不在推导。
- 会不会报四块互不重叠的账（权重 / KV cache / 工作区 / 碎片与水位），再用「可切给 KV 的池子 ÷ 单条占用」换成并发，而不是只报一个 KV 数字。
- 能不能把 sliding window attention 从「省显存」这个概念落成数据结构：环形缓冲的形状与总量、写指针、槽位到绝对位置的映射、读侧掩码的条件。
- 是否分得清两个乘数：GQA 压的是每 token 体积（4×），滑窗压的是注意的历史长度（32k 上是 8×），合计 32×；以及这两刀各自牺牲了什么。
- 能不能把窗口接回服务栈：chunked prefill 的写入顺序、PagedAttention 的块粒度、prefix caching 的复用边界，以及用带宽侧的数字把容量结论闭环。

常见错误答案：

- 拿 MHA 口径 $2LH_qd_{head}$ 或 $2LSd_{model}$ 去算 cache，结果大 4 倍（Mistral 7B 只有 8 个 KV head，$d_{model}=H_{kv}d_{head}$ 对它不成立）。
- 只报 KV 不报权重与工作区；或者把滑窗说成「不缓存」——滑窗只是把 cache 的上限钉在 $W$，prefill 与 decode 照旧依赖它。
- 按「cache 里装的都还在窗口内」做掩码，或者拿槽位号给 K 做 RoPE：张量形状与显存占用全都正常，输出静默变差。

## 原理与推导

### 1. 先把 config 钉死

| 字段 | 值 | 影响哪一项 |
| --- | --- | --- |
| `num_hidden_layers` $L$ | 32 | KV cache 线性项、权重 |
| `num_attention_heads` $H_q$ | 32 | $d_{head}=d_{model}/H_q$、算力 |
| `num_key_value_heads` $H_{kv}$ | 8 | **KV cache 的单价**（每组 4 个 Q head） |
| `head_dim` $d_{head}$ | 128 | 等于 $4096/32$，不是 $4096/8$ |
| `hidden_size` / `intermediate_size` | 4096 / 14336 | $d_{model}=H_qd_{head}$ 自洽；SwiGLU 三个矩阵 |
| `vocab_size` / `max_position_embeddings` | 32000 / 32768 | 权重、窗口上限 |
| `sliding_window` / dtype | 4096 / bf16 | 本题的主角；所有字节数 |

两个口径提醒：`tie_word_embeddings=false`，输入 embedding 与 lm_head 各算一份，逐项加总才是 7.24B；滑动窗口是**这一代模型的设计而非永久默认**——v0.1 的 config 里 `sliding_window` 是 4096，v0.3 起该字段已是 `null`、词表换成 32768，答这题先说版本再谈窗口。

### 2. KV cache 的单价

从张量形状往上乘，每一步只加一个维度（完整推导见 [[llm-internals-02]]）：① 一层一个 KV 头一个位置，K 是 $d_{head}$ 个标量、V 也是 → $2d_{head}$；② 一层 $H_{kv}$ 个 KV 头 → $2H_{kv}d_{head}$；③ $L$ 层 → $2LH_{kv}d_{head}$；④ $S$ 个位置、$b$ 条序列、每元素 $p$ 字节：

$$M_{kv} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b \cdot p$$

代入 Mistral 7B：$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = **128 KiB/token**，$2^{30}/131072 = 8192$ token/GiB。三种口径对照：

| 口径 | 每 token | 相对真值 | 32k 单条 |
| --- | --- | --- | --- |
| 正确：$2LH_{kv}d_{head}p$ | 128 KiB | 1× | 4 GiB |
| 用 Q 头数（MHA 口径）$2LH_qd_{head}p$ | 512 KiB | 4× | 16 GiB |
| $2Ld_{model}p$ | 512 KiB | 4× | 16 GiB |

后两种错法在 MHA 模型上恰好等价于正确式（那时 $H_{kv}=H_q$ 且 $d_{model}=H_qd_{head}$），换成 GQA 就是把 cache 算大 $H_q/H_{kv}=4$ 倍——[[llm-internals-03]] 讲的正是这个乘数。

### 3. 四块账与并发

**权重**逐项加总（不 tie）：embedding 与 lm_head 各 $32000 \times 4096 = 131.07$M；每层 $4096^2 + 2\times4096\times1024 + 4096^2 + 3\times4096\times14336 = 218.10$M，32 层共 6.979B；加上这两份 embedding/lm_head 的 0.262B 后可报 $N = 7.2415$B（RMSNorm 的 0.0003B 忽略不计）。bf16 权重 $7.2415\text{B} \times 2 = 14.48$ GB = **13.49 GiB**（标称 7B 口算的 14.0 GB 是同一量级的粗算）。单卡 80 GiB，沿用 [[inference-serving-08]] 的记账口径：

| 项 | 计算式 | GiB |
| --- | --- | --- |
| 权重 bf16 | $7.2415\text{B} \times 2$ | 13.49 |
| 工作区（激活 / 通信 / 框架） | 每卡 1–3 GiB，取 2.5 | 2.5 |
| 碎片与调度水位 | 总容量约 10% | 8.0 |
| **KV 池** | 80 − 13.49 − 2.5 − 8.0 | **56.01** |

并发就是池子除以单条占用：窗口缓冲 $W=4096$ 是 512 MiB/条 → **112 并发**；不做窗口、32k 全长是 4 GiB/条 → 14 并发；MHA 口径 + 32k 全长是 16 GiB/条 → 3 并发。三个口径要一起说清：80 GB 是标称、物理是 80 GiB（nvidia-smi 报 81920 MiB 量级）；GB 与 GiB 差 7.4%；这 112 是**池子上限**，块粒度、1% 调度水位、每条序列实际只用到几百个 token 都会把真实并发压低。上线后要用引擎打印的「block 数 × block_size × 128 KiB」反推池子跟这张表对账。

### 4. 滑窗把 cache 从线性项变成常数

对绝对位置 $i$ 的 query，滑窗把可见的 key 集合从 $[0, i]$ 截成 $(i-W, i]$，恰好 $W=4096$ 个位置，于是每层每个 KV head 只需保存最近 $W$ 个 token 的 K/V：

$$M_{kv}^{\text{window}} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot \min(S, W) \cdot b \cdot p \;\le\; 512\ \text{MiB} \times b$$

把两个乘数拆开看（32k 上下文）：

| 配置 | 每 token | 单条 cache | 相对 |
| --- | --- | --- | --- |
| MHA + 全长 | 512 KiB | 16 GiB | 1× |
| GQA-8 + 全长 | 128 KiB | 4 GiB | 4×（GQA 的功劳） |
| GQA-8 + 窗口 4096 | 128 KiB | 512 MiB | 32× = 4× × 8× |

8× 来自 $S/W = 32768/4096$。算力侧同样受益：因果全注意力的二次项是 $2LH_qd_{head}S^2$（$QK^\top$ 与 $\text{softmax}\cdot V$ 各一份、掩码折半），换成窗口后 32k prompt 的 pair 数从 $S^2/2 \approx 537$M 降到 $SW - W(W-1)/2 \approx 126$M，attention FLOPs 从 281.5 TFLOPs 降到 66.0 TFLOPs（约 4.3×，渐近值 $S/(2W)$；只有不带因果三角形的口径才是 8×）。代价是窗口外的信息只能**间接**到达：堆 $L$ 层以后，第 $L$ 层的位置 $i$ 能间接看到 $i-LW$，所以理论跨度是 $W \times L = 4096\times32 \approx 131$k 个 token。它的含义是「信息可以经由中间 token 逐层接力」，不等于「能精确回忆窗口外某个 token 的原文」——这是两道完全不同的题。源文还提醒，窗口把最前面的 token 也滑出去之后质量可能崩，因为最前几位是 attention sink；Mistral 7B 用的是不带 sink 的纯滑窗，它在训练窗口内自洽（模型就是在同一掩码下训出来的），但要把这套机制搬到「远超训练长度还要稳定」的流式场景时，sink 问题会回来，那时要么显式保留最前几个 token，要么把窗口外内容压成摘要或走检索（[[llm-internals-02]]）。

### 5. 滚动缓冲的数据结构

每条序列、每层、每个 KV head 维护一个长度 $W$ 的环形缓冲：

```text
K, V: bf16[L=32, H_kv=8, W=4096, d_head=128]   # 每条序列一份
写入：K[l, h, pos % W] = k_new ; pos += 1      # pos 是下一条要写的绝对位置
读取：query 在绝对位置 i；槽位 s 对应的 token 位置是 j = i - ((i - s) % W)
```

单条序列两份张量（K 与 V）合计 $2 \times 32 \times 8 \times 4096 \times 128 \times 2 = 512$ MiB，**与 S 无关**；$S \le W$ 时只用到 $S \times 128$ KiB。prefill 按 token 顺序批量写入，$S > W$ 之后新数据自然覆盖最旧的槽位，不需要任何显式的搬移或压缩——这就是「滚动」的全部含义。decode 每步写一格、写指针前进一步，读侧从环里取 K/V 做 attention，键值集合就是「环上仍然有效的槽位」，有效条件是开区间 $i-W < j \le i$。

### 6. 掩码与位置编码：三类屏蔽

对绝对位置 $i$ 的 query，槽位 $s$ 有效当且仅当

$$j = i - \big((i - s) \bmod W\big), \qquad \max(0,\ i-W+1) \le j \le i$$

三类槽位必须挡掉，少一类就是 bug：① **未来 token**（$j > i$），同一步 chunk 里位置靠后、已经写进环的 K/V；② **陈旧槽位**（$j \le i-W$），被新数据覆盖之前那里仍是旧 token 的 K/V，看着「有数据」但不该被看到；③ **未初始化槽位**（$j < 0$ 或从未写过），prefill 第一个 chunk 里环还是 0 或脏数据，不挡就会分走 softmax 的概率。

位置编码的顺序同样关键：$k = \mathrm{RoPE}(xW_K,\ i)$ 用的是**绝对位置 $i$**，算完再写进槽位，槽位号只是物理下标。两个线上最常见的静默事故：**用槽位号重转 RoPE**，位置信息变成周期性的错误值，输出分布偏移（复算里 max|Δ| ≈ 0.83），但张量形状、显存占用、吞吐指标全都正常；**把「从第 0 格到写指针」当成因果顺序**，环回绕以后 key 的顺序错乱，等价于按槽位号做因果掩码。

### 7. 读和算是两回事

- **读**：decode 一步只读位置 $(i-W, i]$ 的 K/V，更旧的 K/V 再也不会被这条序列读到，可以直接覆盖或释放——这是容量收益的来源。
- **算**：位置 $i$ 的深层 K/V 依赖于前 $L \cdot W \approx 131$k 个 token（层 1 看窗口内，层 2 看这些 token 的层 1 表示，逐层归纳）。所以前缀复用的匹配边界是**整段前缀**，不是窗口。

这两句话合起来解释了缓存设计里的取舍：滑出窗口的块对在线序列已经没有用（可以立刻还池子），但它们同时是「另一个同前缀请求」的复用机会。prefix caching 的 hash 链因此必须覆盖整段前缀（含窗口外的块），否则会在「前缀某处不同」的请求上静默复用错的 KV；至于保留多少滑出的块，就是在并发和命中率之间取舍。

### 8. 与服务栈的交互

- **chunked prefill 的写入顺序**：块内 query 需要的 key 有两段——块前的窗口和块内的因果。若先把整块 $C$ 个 token 的 K/V 全写进环再对整块做 attention，块内靠后的写入会覆盖掉靠前 query 还需要的历史槽位，结果只有块内最后一个 query 是对的。正确做法是顺序推进（等价 $C=1$），或在 kernel 里把「块内 K/V + 块前 $W-C$ 个历史槽位」拼成 key 集合。稳妥的工程约束是 $C \le W$，具体取多少由 TBT SLO 决定，[[inference-serving-14]] 里 Mistral-7B 的 100 ms P99 TBT 档取 $\tau=512$。
- **PagedAttention**：把连续环换成「物理块 + block table」是同一件事的分页版本，掩码仍按绝对位置算。窗口按 token 滑、块按块回收，所以每条序列最多有一个块（15 个 token ≈ 1.9 MiB）的陈旧 K/V——它只影响容量记账、不影响正确性，**前提是掩码按绝对位置算**，而不是假定「cache 里装的都在窗口内」。以公开实现为例，引擎的滑窗支持把序列的 block table 限成 $\lceil W/B \rceil$ 个块的定长队列，滑出去的块直接释放或留给前缀缓存（[[inference-serving-03]]）。
- **prefix caching**：复用边界随窗口移动（第 7 节）；hash 要覆盖整段前缀，位置必须对齐——RoPE 下同一个 token 在不同绝对位置是不同的向量，跨位置复用就是错的。
- **推理窗口不得大于训练窗口**：把 $W$ 调到 8192 会让模型看到训练时从未见过的注意力范围，质量掉；调小则是白白丢掉上下文。这个旋钮在训练时就定死了，服务层只能选「用不用」。
- **KV 量化可以叠加，带宽侧也要闭环**：fp8/int8 让 512 MiB/条减半到 256 MiB、int4 再减半，K 的离群通道要用 per-channel / per-head 粒度，否则先崩的是长上下文召回（[[inference-serving-06]]）。decode 每步要读权重 14.48 GB 加 $b \times 512$ MiB 的窗口 cache，单卡 H100（3.35 TB/s）在 $b=64$、窗口满载时 $(14.48 + 34.36)/3.35 \approx 14.6$ ms/step，聚合约 4.4k token/s，而算力侧只要 0.94 ms（6%）——彻底的内存带宽受限（[[inference-serving-01]]）。反过来看容量：64 条 32k 全量 cache 要 256 GiB，一块卡根本开不出来，滑窗换来的是「并发不再随 $S$ 线性下滑」。

## 数值与代码验证

下面的脚本按 Mistral-7B-v0.1 的 config 复算本节全部数字，并用一个小规模算例（$S=24$、$W=5$）把滚动缓冲与稠密参考逐元素对齐：

```python
import numpy as np

GB, GiB, MiB, KiB = 10**9, 2**30, 2**20, 2**10
L, HQ, HKV, D, I, VOCAB = 32, 32, 8, 4096, 14336, 32000    # Mistral-7B-v0.1 config
S_MAX, W, BYTES, DH = 32768, 4096, 2, D // HQ              # head_dim = 128

# 1) 权重（embedding 与 lm_head 各一份）与 KV 单价
N = L * (D * D + 2 * D * (HKV * DH) + D * D + 3 * D * I) + 2 * VOCAB * D
kv = 2 * L * HKV * DH * BYTES
assert N == 7_241_465_856
print(f"N={N/GB:.4f}B 权重 {N*BYTES/GiB:.2f} GiB | KV/token = {kv} B = {kv/KiB:.0f} KiB "
      f"| {GiB/kv:.0f} token/GiB | MHA 口径 {2*L*HQ*DH*BYTES/KiB:.0f} KiB（4×）")

# 2) 单卡 80 GiB：池子与并发（记账口径同 inference-serving-08）
pool = (80.0 - N * BYTES / GiB - 2.5 - 8.0) * GiB
print(f"池 {pool/GiB:.2f} GiB | 窗口并发 {int(pool//(kv*W))} | 32k 全长并发 "
      f"{int(pool//(kv*S_MAX))} | block=16 记账 {int(pool)//(16*kv)} 块 × 2 MiB"
      f"（{int(pool)//(16*kv)//(W//16)} 并发） | "
      f"S=4k/8k/32k 全长 {[int(kv*s/MiB) for s in (4096, 8192, 32768)]} MiB")

# 3) 滚动缓冲 vs 稠密参考
rng, S, WT = np.random.default_rng(0), 24, 5
q, k, v = (rng.normal(size=(S, 8)) for _ in range(3))

def dense(q, k, v, w):                                     # 绝对位置 + 窗口掩码
    i = np.arange(len(q)); sc = q @ k.T / np.sqrt(q.shape[-1])
    ok = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - w)
    sc = np.where(ok, sc, -np.inf)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v

def ring(q, k, v, w, chunk=1, init_mask=True):             # 槽位 i % W 的环形缓冲
    ck, cv, out = np.zeros((w, q.shape[-1])), np.zeros((w, q.shape[-1])), np.zeros_like(q)
    for a in range(0, len(q), chunk):
        b = min(a + chunk, len(q))
        for i in range(a, b):                              # 写：槽位 i % W
            ck[i % w], cv[i % w] = k[i], v[i]
        for i in range(a, b):                              # 读：只取环上仍有效的槽位
            slot = np.arange(w); pos = i - (i - slot) % w  # 槽位 → 绝对位置
            ok = (pos <= i) & (pos > i - w)                # 挡未来 token + 陈旧槽位
            if init_mask: ok &= pos >= 0                   # 挡还没写过的槽位
            sc = np.where(ok, q[i] @ ck.T / np.sqrt(q.shape[-1]), -np.inf)
            p = np.exp(sc - sc.max())
            out[i] = (p / p.sum()) @ cv
    return out

ref = dense(q, k, v, WT)
print(f"顺序写/读 {np.abs(ring(q,k,v,WT)-ref).max():.2e} | 先写整块再读 "
      f"{np.abs(ring(q,k,v,WT,chunk=3)-ref).max():.2e} | 漏掉初始化掩码 "
      f"{np.abs(ring(q,k,v,WT,init_mask=False)-ref).max():.2e}")

# 4) RoPE：K 用绝对位置 vs 用槽位号
def rope(x, pos, theta=10000.0):
    d = x.shape[-1]; ang = pos * theta ** (-np.arange(0, d, 2) / d)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.concatenate([x1*np.cos(ang) - x2*np.sin(ang),
                           x1*np.sin(ang) + x2*np.cos(ang)], -1)

ka = np.stack([rope(k[i], i) for i in range(S)])           # 绝对位置（对）
ks = np.stack([rope(k[i], i % WT) for i in range(S)])      # 槽位号（错）
print(f"RoPE 用槽位号 {np.abs(dense(q,ka,v,WT) - dense(q,ks,v,WT)).max():.2f}")

# 5) 带宽闭环（H100 SXM：3.35 TB/s、989 TFLOPs bf16 稠密）
for b in (1, 64):
    rd = N*BYTES + b*kv*W
    t = rd / 3.35e12
    print(f"b={b:3d}: 每步读 {rd/GB:5.2f} GB（KV 占 {b*kv*W/rd:.0%}）→ {t*1e3:5.2f} ms，"
          f"聚合 {b/t:5.0f} token/s，算力 {2*N*b/989e12*1e3:.2f} ms")
```

运行结果：

| 输出 | 值 |
| --- | --- |
| 参数量 / 权重 | 7.2415 B；14.48 GB = 13.49 GiB |
| KV 单价 | 131 072 B = 128 KiB = 131.072 KB；8192 token/GiB；MHA 口径 512 KiB（4×） |
| 窗口缓冲 | S=4k / 8k / 32k 全长是 512 / 1024 / 4096 MiB，窗口缓冲都是 512 MiB |
| 80 GiB 卡的池子 | 56.01 GiB → 窗口并发 112，32k 全长并发 14，block=16 记账 28678 块 × 2 MiB（112 并发） |
| 滚动缓冲 vs 稠密参考 | 顺序写/读 max\|Δ\| = 3.33e-16；先写整块再读 2.31；漏掉初始化掩码 0.77 |
| K 用槽位号做 RoPE | max\|Δ\| = 0.83 |
| 带宽闭环 | b=1：15.02 GB/step → 4.48 ms（223 token/s）；b=64：48.84 GB/step → 14.58 ms（4390 token/s，算力占 6.4%） |

四个可以直接背的结论：**112 与 14** 是同一张卡、同一个模型的两种窗口策略，差 8 倍，这就是滑窗买到的东西；**先写整块再读**的错误顺序只让块内最后一个 query 正确，其余全错；**漏掉未初始化槽位**（0.77）和**用槽位号做 RoPE**（0.83）的偏差量级一样大，但前者是 prefill 早期的边界问题、后者从第一个 token 起就错；**b=64 时 KV 已占每步读取量的 70%**（权重 14.48 GB vs 窗口 cache 34.36 GB），KV 读取量追平权重的临界 batch 是 27。

## 常见追问

- **追问**：窗口为什么是 4096，能不能调到 8k 或 16k？
  - 要点：窗口决定了训练时的注意力掩码，推理期只能 ≤ 训练值。调大让模型看到训练中从未出现的注意力范围，权重分布失配、质量掉；调小则白丢上下文，而且省不下多少——512 MiB/条已经是常数，$W$ 减半只省一半容量却砍掉一半历史。Mistral 7B 就是 $W=4096$、$L=32$ 的组合（理论跨度 131k），后续版本（v0.3 的 config 里 `sliding_window` 已是 `null`）改回全注意力，说明这是随代际变化的设计选择。
- **追问**：滑窗下 prefill 的 attention 怎么切？chunk 取多大？
  - 要点：每个 query 的注意范围只有 $\min(i, W)$ 个 key，所以 prefill 的 attention 算力从 $O(S^2)$ 降到 $O(SW)$；但写入与读取必须顺序化——块内 K/V 一旦先落盘就会覆盖块内靠前 query 还需要的历史槽位。工程上取 $C \le W$，实际值由 TBT SLO 反推（Mistral-7B 的 100 ms P99 TBT 档取 512），并注意跨 chunk 重读会带来 $(N-1)/2$ 倍的 KV 读取量（[[inference-serving-14]]）。
- **追问**：滚动缓冲和 PagedAttention 怎么共存？
  - 要点：两条路。①连续环形张量：形状固定、无分配器、写指针一动就完成，适合固定 batch 槽位或自研 kernel，代价是序列进出要占满整块 512 MiB、不好共享前缀；②物理块 + block table 的定长队列（长度 $\lceil W/B \rceil$）：块按需分配、滑出即回收、可跨请求共享，代价是多一层映射、每条序列最多一个块的陈旧 K/V（15 token ≈ 1.9 MiB）。两种实现的掩码都必须按绝对位置算。
- **追问**：窗口丢信息，产品上怎么兜？
  - 要点：先承认边界——窗口外的 token 无法被精确回忆，$W\times L \approx 131$k 只是层间可传递跨度。适合聊天、摘要、分类这类以近期上下文为主的负载；需要精确回忆长文时，要么把窗口外内容压成摘要或检索回填进 prompt，要么换长上下文版本，而不是偷偷把 $W$ 调大。
- **追问**：KV 量化叠加滑窗，容量还能再翻几倍？
  - 要点：滑窗已经砍了 8×（32k 口径），量化再叠 fp8/int8 让 512 MiB/条 → 256 MiB、int4 → 128 MiB，池子不变则并发翻倍；代价是量化误差在长上下文召回上最先暴露，K 的离群通道要用 per-channel / per-head 粒度（[[inference-serving-06]]）。
- **追问**：客户要 128k 上下文，你怎么答？
  - 要点：容量不是主要问题——窗口缓冲仍是 512 MiB/条，128k 时的并发与 4k 时一样；真正的问题是能力：v0.1 的 RoPE 以 $10^4$ 为底、训练窗口 4096，直接外推到 128k 会崩，需要位置插值 / YaRN 这类长上下文版本（[[llm-internals-08]]），而监督信号里窗口外的召回能力也未必训到。先对齐「128k 上下文」是容量承诺还是能力承诺。

## 相关题目

- [[inference-serving-08]]：70B 模型的四块显存账与带宽闭环，本题的记账口径与 GB/GiB、标称/物理三个口径直接沿用。
- [[llm-internals-02]]：KV cache 公式 $2LH_{kv}d_{head}Sbp$ 的逐步推导，以及「滑窗是唯一能止住无限增长的手段」这一结论。
- [[llm-internals-03]]：GQA 把每个 KV 头摊给 4 个 Q head，是本题「4×」这一刀的来源。
- [[inference-serving-03]]：PagedAttention 的 block、block table 与共享，是滚动缓冲在引擎里的分页版本。
- [[inference-serving-14]]：chunked prefill 的 chunk 大小与 TBT SLO 的关系，本题第 8 节的 $C \le W$ 由它收口。
- [[inference-serving-05]]：prefix caching 的 block 粒度命中与位置对齐，与滑窗的复用边界一起看；[[inference-serving-01]] 的 prefill/decode 瓶颈划分是第 8 节带宽闭环的前提。

## 参考资料与归属

1. [LLM 中的 KV Cache 是什么？](https://outcomeschool.com/blog/kv-cache-in-llms)，Amit Shekhar（Outcome School），2026-03-27。提供 KV cache 的定义、逐 token 生成里的重复计算、只缓存 K/V 因而公式有因子 2、显存换时间的 trade-off，以及「保留最近一段窗口 + 开头几个 attention sink」的定位。
2. [Sliding Window Attention 是如何工作的？](https://outcomeschool.com/blog/how-does-sliding-window-attention-work)，Amit Shekhar（Outcome School），2026-09-05。提供窗口逐 token 滑动的算例、成本从平方降到线性、信息靠层堆叠逐层传递（4 词 × 10 层 ≈ 40 词），以及「窗口滑过最前几个 token 后质量可能崩」这个 attention sink 提醒。
3. [What is Grouped Query Attention (GQA) and Why Do LLMs Use It?](https://outcomeschool.com/blog/grouped-query-attention)，Amit Shekhar（Outcome School），2026-04-22。提供 MQA/GQA 的共用与分组机制、压缩倍数 $H_q/H_{kv}$，是本题 4× 的来源。
4. [Paged Attention in LLMs](https://outcomeschool.com/blog/paged-attention-in-llms)，Amit Shekhar（Outcome School），2026-03-29。提供 block 与 block table、内部/外部碎片的划分、最后一个 block 是唯一的浪费来源，用于第 8 节的分页版本。
5. [KV Cache Compression](https://outcomeschool.com/blog/kv-cache-compression)，Amit Shekhar（Outcome School），2026-09-12。提供压缩手段的分类（丢 token、降低每元素比特数、改结构），用于追问里的量化叠加。
6. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（延伸），Kwon et al.（vLLM, SOSP 2023），2023-09-12。第 3、8 节的分页记账与「浪费被限制在一个 block 之内」沿用该论文的口径。

config 的每个字段都核对过模型公开的 `config.json`（[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)、[Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json)），其中 v0.3 的 `sliding_window` 已是 `null`、词表为 32768。正文的参数量、KV 单价、池子与并发、环形缓冲正确性、带宽闭环都是按这些 config 与 H100 SXM 公开规格自行复算的（脚本见正文），来源资料没有给出 Mistral 7B 的这些算例；与来源出现分歧时以复算为准，口径已在正文标明。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
