---
type: question
id: coding-02
topic: 编程与数据结构
order: 2
question: 实现 multi-head attention，再把它改造成 grouped-query attention。
question_en: Implement multi-head attention, then convert it to grouped-query attention.
asked_at: [Google DeepMind, Mistral AI, 阿里巴巴（Qwen）]
level: 进阶
tags: [mha, gqa, mqa, kv-heads]
sources:
  - title: Transformer 中的 Multi-Head Attention
    url: https://outcomeschool.com/blog/multi-head-attention-in-transformers
    author: Amit Shekhar (Outcome School)
    published: 
  - title: 分组查询注意力（GQA）
    url: https://outcomeschool.com/blog/grouped-query-attention
    author: Amit Shekhar (Outcome School)
    published: 
  - title: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints（延伸）
    url: https://arxiv.org/abs/2305.13245
    author: Ainslie et al. (EMNLP 2023)
    published: 2023-05-22
  - title: Fast Transformer Decoding: One Write-Head is All You Need（MQA）（延伸）
    url: https://arxiv.org/abs/1911.02150
    author: Shazeer
    published: 2019-11-06
related: [coding-01, coding-03, llm-internals-04, inference-serving-03, inference-serving-07]
updated: 2026-09-28
---

## 一句话答案

> MHA 是 $H$ 组独立的 $Q/K/V$ 投影加一次输出投影：$W_q$、$W_k$、$W_v$、$W_o$ 都是 $d_{model} \times d_{model}$，注意力部分共 $4d_{model}^2$ 个参数，$d_{model}=4096$ 时是 67.11M。
> 改成 GQA 只动一个数：K/V 的头数从 $H$ 变成 $H_{kv}$。$H_{kv}=H$ 就是 MHA，$H_{kv}=1$ 就是 MQA，介于两者之间才是 GQA；$Q$ 的头数一个不改，输出投影也不改。KV cache 每 token 每层是 $2H_{kv}d_{head}$ 个元素，LLaMA-3-70B 口径（80 层、$H=64$、$d_{head}=128$、bf16）下 GQA-8 是 320 KiB/token，若同形状用 MHA（$H_{kv}=64$）则是 2.5 MiB/token，正好 8 倍。
> 实现上最容易错的两处：分头必须先 `view` 成 $(B,T,H,d_{head})$ 再 `transpose` 成 $(B,H,T,d_{head})$，一步 `view` 到位切出来的是错的；K/V 扩容时 `repeat_interleave` 是真复制、`expand` 是零拷贝，但 eager 里两条路都比不上直接调 `F.scaled_dot_product_attention(..., enable_gqa=True)`——实测后者快 7.8–9.3 倍（中位数 / 最小值口径），因为真正的收益要靠 kernel 支持，不是靠 tensor 形状技巧。

## 面试官在考什么

- **契约与形状**：输入 $(B,T,d_{model})$、四个投影的形状、输出 $(B,T,d_{model})$、注意力权重 $(B,H,T,T)$，以及分头/合并这四步的形状变化。这题的分水岭是「能不能一边写一边说形状」，而不是背出公式。
- **能不能当场算两个数**：MHA 的 $4d_{model}^2$（$d_{model}=4096$ 时 67.11M）与 KV cache 的 $2L H_{kv} d_{head} \times$ bytes。两个数都直接决定线上成本，报不出来说明没做过容量规划。
- **是否把 GQA 理解成泛化**：$H_{kv}=H$ 退化为 MHA、$H_{kv}=1$ 是 MQA，并能用**逐元素数值对齐**证明自己的实现（而不是「跑通了、形状对」）。$H_{kv}=H$ 时与一份独立的 MHA 实现差 0.0e+00，是最有说服力的正确性证据。
- **工程细节**：$H$ 必须能被 $H_{kv}$ 整除；分头用 `reshape` 还是 `view`（非连续张量会抛异常）；`repeat_interleave` 在长序列上的显存；导出到推理引擎需要 GQA kernel 支持（[[inference-serving-11]]）；张量并行下 KV 头怎么切（[[inference-serving-07]]）。
- **是否知道收益为什么在 decode**：decode 的算术强度约 $H_q/H_{kv}$ FLOP/byte，GQA-8 把这个数从 1 提到 8，而 H100 的 roofline 拐点是 $989/3.35 \approx 295$ FLOP/byte，仍在带宽一侧——所以缩小 KV 直接变成更快的 decode（[[llm-internals-02]]）。

常见错误答案：

- **把 KV cache 按 query 头数算**。LLaMA-3-70B 是 8 个 KV 头而不是 64 个，代错就是 8 倍误差，这是这题最高频的失分点。
- **说 GQA 减少了 FLOPs 所以更快**。$QK^\top$ 的 FLOPs 一点没变（$Q$ 还是 $H$ 个头），省的是 K/V 投影参数与 decode 每步要读的字节。

## 原理与推导

### 1. 契约与形状

| 量 | 形状 | 说明 |
| --- | --- | --- |
| 输入 $x$ | $(B, T, d_{model})$ | |
| $W_q$ | $(d_{model}, d_{model})$ | $Q$ 永远是 $H$ 个头，每头 $d_{head} = d_{model}/H$ |
| $W_k, W_v$ | $(d_{model}, H_{kv} d_{head})$ | 形状按 $y=xW$ 的约定，`nn.Linear` 的 `weight` 是它的转置；GQA **只在这里变小** |
| $W_o$ | $(d_{model}, d_{model})$ | 把 $H$ 个头拼回的 $d_{model}$ 再混一次 |
| 注意力权重 | $(B, H, T, T)$ | 与第 1 题同一契约，只用于调试，生产不开 |

分头两步，顺序不能反：投影输出 $(B,T,H d_{head})$ → `view(B,T,H,d_head)` 得到「每个头一段」→ `transpose(1,2)` 得到 $(B,H,T,d_{head})$。合并时反过来：`transpose(1,2)` 得到 $(B,T,H,d_head)$，此时张量不连续，只能用 `reshape(B,T,H*d_head)`——用 `view` 会直接抛 `view size is not compatible with input tensor's size and stride`（实测报错见「数值与代码验证」一节，脚本输出里为排版做了截断）。

### 2. 参数量

MHA 的四个投影各 $d_{model}^2$，合计 $4d_{model}^2$；GQA 只把 $W_k$、$W_v$ 换成 $d_{model} \times H_{kv}d_{head} = d_{model}^2 \cdot H_{kv}/H$，于是

$$N_{attn} = 2d_{model}^2 + 2d_{model}^2 \cdot \frac{H_{kv}}{H}$$

$d_{model}=4096$、$H=8$ 时：$4 \times 4096^2 = 67{,}108{,}864 = 67.11$M（MHA）；$H_{kv}=2$ 降到 41.94M；$H_{kv}=1$ 降到 37.75M。注意 $W_o$ 与 $W_q$ 不随 $H_{kv}$ 变，所以 GQA 省的参数比例远小于 KV cache 省的比例。按 LLaMA-3-70B 的形状（$d_{model}=8192$、$H=64$、$H_{kv}=8$、$d_{head}=128$）复算：单层注意力参数从 MHA 的 268.4M 降到 151.0M，每层省 117.4M，80 层共省 9.40B 参数。这是「同一 $d_{model}$ 下换成 GQA 结构」的差，真实 70B 模型从预训练起就是 GQA，不存在这 9.4B。

### 3. 一个开关，两种扩容写法

把 K/V 从 $H_{kv}$ 个头扩到 $H$ 个头，数学上是「同一份 K/V 被 $G = H/H_{kv}$ 个 query head 复用」：

- **`repeat_interleave(G, dim=1)`**：把每个 KV 头复制 $G$ 份，得到 $(B,H,T,d_{head})$，**真实占用显存**，是 compact 张量的 $G$ 倍。分组顺序是 $[0,0,\dots,1,1,\dots]$，即第 $h$ 个 query head 用第 $\lfloor h/G \rfloor$ 个 KV 头，与逐头下标写法一致。
- **广播**：把组维插进来得到 $(B,H_{kv},G,T,d_{head})$ 对 $(B,H_{kv},1,T,d_{head})$，`expand` 本身零拷贝（`storage` 不变、`is_contiguous()` 为 False）。但只要下游有一个算子要求连续——`reshape`、`view`，或者 eager 里某些 matmul 实现——这份副本就会被悄悄物化，省显存的前提立刻消失。

实测（$(1,8,16384,128)$ bf16）：`repeat_interleave` 峰值显存增量 256.0 MiB，`expand` 是 0.0 MiB，但对 `expand` 的结果做一次 `reshape` 就立刻变成 256.0 MiB。**这也是为什么生产代码里不自己扩容**：推理引擎的 GQA kernel 直接按 $h \to \lfloor h/G \rfloor$ 读 compact 的 K/V，一份副本都不产生。

### 4. KV cache 记账（GQA 存在的全部理由）

每 token 每层的元素数是 $2H_{kv}d_{head}$（K 与 V 各 $H_{kv}d_{head}$），乘 dtype 字节数与层数：

$$M_{KV}(S) = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot \text{bytes} \cdot S$$

LLaMA-3-70B（$L=80$、$H=64$、$H_{kv}=8$、$d_{head}=128$、bf16）复算：$2 \times 8 \times 128 \times 2 = 4096$ B/层，$\times 80$ 层 $= 327{,}680$ B $= 320$ KiB/token——与本仓库其它专题的常数一致（[[llm-internals-02]]、[[inference-serving-08]]）。同形状 MHA 是 2.5 MiB/token，MQA 是 40 KiB/token。单条 8k 上下文的占用因此是 2.50 GiB（GQA-8）对 20.00 GiB（MHA）。

### 5. 为什么收益在 decode，而不在 prefill

decode 每个 step 只处理 1 个 token，却要把历史 K/V 全部读一遍，读数 $\propto H_{kv}$，而 FLOPs 仍然是 $H$ 个头的量，于是算术强度约 $H_q/H_{kv}$ FLOP/byte：MHA 是 1，GQA-8 是 8，都远低于 295 FLOP/byte 的拐点（[[llm-internals-02]]）。带宽受限时，把 K/V 缩 8 倍几乎直接换成 8 倍速度。prefill 完全不同：主导项是投影 GEMM 与 $T \times T$ 的分数矩阵，$Q$ 与 $W_o$ 的参数量没变、分数矩阵的大小没变，所以 $H_{kv}$ 变小只影响其中的一部分。实测 $T=2048$、$d_{model}=8192$ 时 MHA 64.6 ms、GQA-8 47.2 ms，只有 1.37×——这正好说明「GQA 是为了 decode 的内存带宽而生的」，把它当成 prefill 加速手段是方向性错误。

### 6. 论文口径与 uptrain

MQA 的动机（Shazeer, 2019）：multi-head attention 的训练很快，但**增量推理**（无法按序列长度并行）常常很慢，原因是反复加载大 K/V 张量的**内存带宽成本**；把 K/V 在所有 head 间共享能大幅缩小这些张量，从而降低增量解码的带宽需求，原文的实验结论是解码可以快很多、质量只有轻微下降（论文口径）。GQA（Ainslie 等, EMNLP 2023）：一是给出一条 uptrain 配方，用**原始预训练算力的约 5%** 把已有的 multi-head checkpoint 继续训练成 MQA（摘要里这个 5% 是就 MQA uptrain 给出的数字）；二是提出 GQA，用**多于 1、少于 query 头数**的 KV 头数在质量与速度间取折中，论文结论是 uptrained GQA 达到接近 MHA 的质量、同时具有接近 MQA 的速度。具体做法是把每个组内各 head 的 K/V 投影权重做平均（等价于 mean-pool），再用这 5% 的算力继续训练，让模型适应共享后的 K/V。这也是 GQA 被迅速采纳的原因：已有 MHA checkpoint 不必推倒重来。

公开使用情况（来源：GQA 博客）：LLaMA 2 的 34B 与 70B 用 8 组 GQA（70B 有 64 个 query head，KV cache 因此小 8 倍），7B/13B 仍是 MHA；LLaMA 3 全尺寸使用 GQA；Mistral 7B 是 32 个 query head、8 组 KV。

## 数值与代码验证

下面这段是可直接运行的参考实现（CPU 即可，约 2 秒；输出形状一致、参数量公式、$H_{kv}=H$ 与逐头 MHA 的数值对齐都写成了 `assert`，跑不过会直接抛异常），完整脚本在仓库根 `.work/coding02_doc.py`（下面代码块与它逐字节一致），逐配置报告版在 `.work/coding02_ref.py`：

```python
# coding-02：MHA -> GQA -> MQA 的最小可运行参考实现（CPU 即可跑）：python3 coding02_doc.py；逐配置报告版 coding02_ref.py，GPU 微基准 coding02_bench.py。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
NEG = -1e9  # 掩码用有限大负数：-inf 在全掩码行上会 softmax 出 NaN


def attention(q, k, v, causal=True, mask_value=NEG, return_attn=False):
    """q,k,v 形状 (..., T, d)；k/v 的头维可以是 1，此时靠广播完成 GQA 的共享。"""
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])     # (..., Tq, Tk)
    if causal:
        Tq, Tk = scores.shape[-2:]
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
        scores = scores.masked_fill(~mask, mask_value)              # 必须在 softmax 之前
    w = torch.softmax(scores, dim=-1)
    return (w @ v, w) if return_attn else w @ v


def merge_heads(x):                       # (B,H,T,d) -> (B,T,H*d)：只能用 reshape
    B, H, T, d = x.shape
    return x.transpose(1, 2).reshape(B, T, H * d)


class Attention(nn.Module):
    """num_kv_heads == num_heads 是 MHA；== 1 是 MQA；介于两者之间是 GQA。"""

    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert d_model % num_heads == 0 and num_heads % num_kv_heads == 0, '必须整除'
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = d_model // num_heads
        self.groups = num_heads // num_kv_heads         # 一个 KV 头被几个 query head 共享
        kv_dim = num_kv_heads * self.head_dim           # 只有 K/V 的输出维度变小
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True, mode='repeat', return_attn=False):
        B, T, _ = x.shape
        H, Hkv, G, d = self.num_heads, self.num_kv_heads, self.groups, self.head_dim
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if G == 1 or mode == 'repeat':
            # 分头两步：(B,T,H*d) -> (B,T,H,d) -> (B,H,T,d)；一次 reshape 到位是错的
            qh = q.view(B, T, H, d).transpose(1, 2)
            kh = k.view(B, T, Hkv, d).transpose(1, 2)
            vh = v.view(B, T, Hkv, d).transpose(1, 2)
            if G > 1:                                   # 真复制：显存 x G
                kh, vh = kh.repeat_interleave(G, 1), vh.repeat_interleave(G, 1)
            o, w = attention(qh, kh, vh, causal=causal, return_attn=True)
            out = merge_heads(o)
        else:                                           # 零拷贝：(B,Hkv,G,T,d) 广播 (B,Hkv,1,T,d)
            q5 = q.view(B, T, Hkv, G, d).permute(0, 2, 3, 1, 4)
            k5 = k.view(B, T, Hkv, d).transpose(1, 2).unsqueeze(2)
            v5 = v.view(B, T, Hkv, d).transpose(1, 2).unsqueeze(2)
            o5, w5 = attention(q5, k5, v5, causal=causal, return_attn=True)
            out, w = merge_heads(o5.reshape(B, H, T, d)), w5.reshape(B, H, T, T)
        return (self.o_proj(out), w) if return_attn else self.o_proj(out)


def ref_attn(x, m):                       # 独立对照：逐头循环，组下标显式写成 h // G
    B, T, _ = x.shape
    H, Hkv, d = m.num_heads, m.num_kv_heads, m.head_dim
    q = F.linear(x, m.q_proj.weight).view(B, T, H, d)
    k = F.linear(x, m.k_proj.weight).view(B, T, Hkv, d)
    v = F.linear(x, m.v_proj.weight).view(B, T, Hkv, d)
    outs = []
    for h in range(H):
        g = h // (H // Hkv)
        s = q[:, :, h] @ k[:, :, g].transpose(-1, -2) / math.sqrt(d)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), NEG)
        outs.append(torch.softmax(s, -1) @ v[:, :, g])
    return F.linear(torch.cat(outs, -1), m.o_proj.weight)


if __name__ == '__main__':
    md = lambda a, b: (a.double() - b.double()).abs().max().item()
    B, T, D, H = 2, 16, 64, 8
    x = torch.randn(B, T, D)
    for name, kv in (('MHA  ', 8), ('GQA-2', 2), ('MQA  ', 1)):
        m = Attention(D, H, kv).eval()
        with torch.no_grad():
            o, w = m(x, return_attn=True)
            r = ref_attn(x, m)
        assert tuple(o.shape) == (B, T, D) and md(o, r) < 1e-6, (name, o.shape, md(o, r))
        print(f'{name} H_kv={kv} G={H//kv}: out={tuple(o.shape)} w={tuple(w.shape)} '
              f'行和={w.sum(-1).min().item():.6f} 上三角最大={w.triu(1).abs().max().item():.1e} '
              f'vs 逐头对照 max|diff|={md(o, r):.1e}')
    m8, m2, m2b = Attention(D, H, 8).eval(), Attention(D, H, 2).eval(), Attention(D, H, 2).eval()
    m2b.load_state_dict(m2.state_dict())
    with torch.no_grad():
        print(f'H_kv=H 的统一实现 vs 逐头 MHA   max|diff|={md(m8(x), ref_attn(x, m8)):.1e}')
        print(f'broadcast vs repeat_interleave  max|diff|={md(m2(x), m2b(x, mode="repeat")):.1e}')
        q = m2.q_proj(x).view(B, T, H, m2.head_dim).transpose(1, 2)
        k = m2.k_proj(x).view(B, T, 2, m2.head_dim).transpose(1, 2)
        v = m2.v_proj(x).view(B, T, 2, m2.head_dim).transpose(1, 2)
        sdpa = m2.o_proj(merge_heads(F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)))
        print(f'GQA-2 vs SDPA(enable_gqa=True)   max|diff|={md(m2(x), sdpa):.1e}')
        assert md(m8(x), ref_attn(x, m8)) < 1e-6 and md(m2(x), m2b(x, mode='repeat')) < 1e-6
    for kv in (8, 2, 1):
        n = sum(p.numel() for p in Attention(4096, H, kv).parameters())
        assert n == 2*4096**2 + 2*4096**2*kv//H, n      # 形状一致时参数量只由 H_kv 决定
        print(f'd_model=4096 H_kv={kv}: 注意力参数 {n/1e6:.2f}M  (MHA 即 4*d_model^2 = {4*4096**2/1e6:.1f}M)')
    L, dh, byt, H70 = 80, 128, 2, 64        # LLaMA-3-70B：80 层、H=64、head_dim 128、bf16
    for name, h in (('MHA  ', 64), ('GQA-8', 8), ('MQA  ', 1)):
        per_tok = 2 * L * h * dh * byt
        print(f'{name} H_kv={h:2d}: {per_tok:>9,} B/token = {per_tok/1024:6.1f} KiB, '
              f'8k 上下文 {per_tok*8192/2**30:.2f} GiB  相对 MHA 缩小 {H70//h:>2d}x')
```

实测输出（Python 3.10、PyTorch 2.12、float32、`torch.manual_seed(0)`）：

```text
MHA   H_kv=8 G=1: out=(2, 16, 64) w=(2, 8, 16, 16) 行和=1.000000 上三角最大=0.0e+00 vs 逐头对照 max|diff|=0.0e+00
GQA-2 H_kv=2 G=4: out=(2, 16, 64) w=(2, 8, 16, 16) 行和=1.000000 上三角最大=0.0e+00 vs 逐头对照 max|diff|=0.0e+00
MQA   H_kv=1 G=8: out=(2, 16, 64) w=(2, 8, 16, 16) 行和=1.000000 上三角最大=0.0e+00 vs 逐头对照 max|diff|=0.0e+00
H_kv=H 的统一实现 vs 逐头 MHA   max|diff|=0.0e+00
broadcast vs repeat_interleave  max|diff|=0.0e+00
GQA-2 vs SDPA(enable_gqa=True)   max|diff|=1.8e-07
d_model=4096 H_kv=8: 注意力参数 67.11M  (MHA 即 4*d_model^2 = 67.1M)
d_model=4096 H_kv=2: 注意力参数 41.94M  (MHA 即 4*d_model^2 = 67.1M)
d_model=4096 H_kv=1: 注意力参数 37.75M  (MHA 即 4*d_model^2 = 67.1M)
MHA   H_kv=64: 2,621,440 B/token = 2560.0 KiB, 8k 上下文 20.00 GiB  相对 MHA 缩小  1x
GQA-8 H_kv= 8:   327,680 B/token =  320.0 KiB, 8k 上下文 2.50 GiB  相对 MHA 缩小  8x
MQA   H_kv= 1:    40,960 B/token =   40.0 KiB, 8k 上下文 0.31 GiB  相对 MHA 缩小 64x
```

读法：三种配置的输出形状都是 $(2,16,64)$、权重都是 $(2,8,16,16)$，每行权重和为 1、上三角严格为 0；**$H_{kv}=H$ 的统一实现与逐头循环的 MHA 参考实现差 0.0e+00**，`broadcast` 与 `repeat_interleave` 两条扩容路径也差 0.0e+00——同一批权重、同一套数学，走不同代码路径得到逐位相同的结果，这比「形状对得上」强得多。与 `F.scaled_dot_product_attention(enable_gqa=True)` 的差是 1.8e-07（float32 舍入量级，float64 下是 1.7e-16，见 `coding02_ref.py`），说明分组下标映射没有错位。参数量三个数字（67.11M / 41.94M / 37.75M）与公式 $2d^2 + 2d^2 H_{kv}/H$ 一致，KV 账本给出的 320 KiB/token 也与仓库统一口径对上，脚本最后三行还把相对 MHA 的缩小倍数直接打了出来（1× / 8× / 64×）。GPU 侧微基准（`.work/coding02_bench.py`，设备是 RTX 5070 Laptop 8 GB，不是 H100；绝对值不可外推，只看同一台机器上的相对关系）：

```text
设备: NVIDIA GeForce RTX 5070 Laptop GPU
== K/V 扩容：repeat_interleave（真复制） vs expand（零拷贝） ==
compact K (1, 8, 16384, 128) = 32.0 MiB
repeat_interleave -> (1, 64, 16384, 128) 256.0 MiB  峰值显存增量 256.0 MiB
expand           -> (1, 8, 8, 16384, 128) 256.0 MiB  峰值显存增量 0.0 MiB  (storage 32.0 MiB, contiguous False)
对 expand 结果做 reshape/view 立刻物化 -> storage 256.0 MiB
非连续张量上 view 报错: view size is not compatible with input tensor's size and stride (at least one dimension spans ac
== decode 单步（Tq=1，历史 S=16384，H=64, d=128, bf16） ==
每步要读的 K+V：MHA 512.0 MiB, GQA-8 64.0 MiB，比值 8.0x
MHA  H_kv=64         中位  2.041 ms / 最小  1.991 ms  峰值显存    4.0 MiB  等效带宽  269.6 GB/s
GQA-8 broadcast      中位  8.277 ms / 最小  7.770 ms  峰值显存  260.0 MiB  等效带宽   77.7 GB/s
GQA-8 repeat_interleave 中位  9.132 ms / 最小  5.876 ms  峰值显存  516.0 MiB  等效带宽  102.8 GB/s
GQA-8 SDPA(enable_gqa) 中位  0.262 ms / 最小  0.214 ms  峰值显存    0.3 MiB  等效带宽  313.7 GB/s
== prefill（T=2048, d_model=8192, H=64, d=128, bf16） ==
MHA   H_kv=64 mode=broadcast 中位   64.6 ms / 最小   63.9 ms  峰值显存 1156.0 MiB  注意力参数  268.4M
GQA-8 H_kv= 8 mode=broadcast 中位   47.2 ms / 最小   46.5 ms  峰值显存 1132.0 MiB  注意力参数  151.0M
GQA-8 H_kv= 8 mode=repeat    中位   45.5 ms / 最小   44.5 ms  峰值显存 1164.0 MiB  注意力参数  151.0M
MQA   H_kv= 1 mode=broadcast 中位   43.4 ms / 最小   43.0 ms  峰值显存 1093.0 MiB  注意力参数  136.3M
```

三点结论：① `expand` 真的不占显存（0.0 MiB），但 `reshape` 一步就物化 256.0 MiB；② decode 单步读 512.0 MiB 的 MHA 用 2.041 ms（等效 269.6 GB/s，典型的带宽受限），GQA-8 只读 64.0 MiB，交给 `SDPA(enable_gqa=True)` 是 0.262 ms（等效 313.7 GB/s），加速 7.8×（按最小值算是 9.3×，正好在 8 倍数据量比的两侧）；③ 手写扩容在 eager 里是**负优化**：`broadcast` 8.277 ms、`repeat_interleave` 9.132 ms，比 MHA 还慢，因为前者被内部物化、后者多了一次真复制，等效带宽只有 77.7 / 102.8 GB/s。GQA 的 8 倍收益必须由 fused kernel 兑现，这一段是「知道 GQA 是什么」与「能把 GQA 跑快」的分界线。prefill 那一组数字（$T=2048$、$d_{model}=8192$）里，MHA 64.6 ms、GQA-8 47.2 ms、MQA 43.4 ms，峰值显存都在 1.1 GiB 上下且几乎不随 $H_{kv}$ 变——因为此时峰值由 $T \times T$ 的分数矩阵占着（$64 \times 2048^2 \times 2$ B $= 512$ MiB 量级）。这正是标准实现的 $O(T^2)$ 显存问题，也是长上下文必须上 FlashAttention 一类 kernel 的原因（[[llm-internals-05]]）。GPU 那四行的等效带宽口径是「该行计入的 K/V 字节 ÷ 最小耗时」：MHA 记 512 MiB，两条手写扩容路记 compact 的 64 MiB 加展开后的 512 MiB 共 576 MiB，`SDPA` 只记 compact 的 64 MiB——269.6 / 77.7 / 102.8 / 313.7 GB/s 都能用这几组数字重算。

## 常见追问

- **追问**：为什么 GQA 不像 MQA 那样掉点？
  - 要点：MQA 把所有 head 压成一组 K/V，等于强制所有 head 共用同一套「视角」；GQA 保留 $H_{kv}>1$ 组，头数本身就是一个可调的超参，训练时模型仍有多个不同的 K/V 子空间。论文的口径是 uptrained GQA 质量接近 MHA、速度接近 MQA，而 MQA 会有质量下降。
- **追问**：uptrain 具体怎么做？
  - 要点：对每个组，把组内各 head 的 $W_k$、$W_v$ 按输出维做平均（mean-pool）得到共享的 K/V 投影，$W_q$、$W_o$ 原样保留；然后用约 5% 的原始预训练算力继续训练（论文口径，摘要里的 5% 是给 MQA uptrain 的），让模型适应共享后的 K/V 表示。
- **追问**：为什么 KV cache 会成为推理显存的主要占用？
  - 要点：权重是 batch 内共享的一份，KV cache 是每条序列各自一份且随上下文线性增长：$2 L H_{kv} d_{head} \times$ bytes $\times S$，再乘并发数。70B bf16 权重约 130 GiB，8 张 80 GB 卡扣掉权重只剩约 466 GiB；GQA-8 每 GiB 能装约 3,277 个 token 的 cache，MHA 只能装 1/8（[[llm-internals-02]]、[[inference-serving-08]]）。
- **追问**：$H$ 不能被 $H_{kv}$ 整除怎么办？
  - 要点：标准做法是不允许——组边界必须整齐。若真要更细，就退到「按 KV 头数取能整除的值」或改用 MLA 一类把 K/V 压成 latent 的方案，而不是让组大小不均（kernel 的下标写成 $\lfloor h/G \rfloor$ 依赖整除）。这条约束在张量并行下更硬：KV 头要按 TP 度切，TP 度超过 $H_{kv}$ 时只能复制 cache（[[inference-serving-07]]）。
- **追问**：既然 prefill 不赚，为什么所有新模型仍然用 GQA？
  - 要点：一是长上下文服务的显存容量，二是 decode 阶段的带宽（输出越长、并发越高，decode 占比越大），三是 KV 头数少还让 PagedAttention 的 block 更小、同样显存装更多块、prefix 共享的粒度更细（[[inference-serving-03]]）。训练侧的收益有限，GQA 是「为推理而设计」的结构。

## 公司变体

`asked_at` 记录的是这三家问过这道题（依据仓库 `README.zh-CN.md` 里每题下的「出现于」列表），公开可查的只有这一点；下面按各自的公开技术产出说明倾向，不是面试记录。

- **Google DeepMind**：本题两条延伸来源都出自 Google——MQA 的作者 Shazeer、GQA 的作者 Ainslie 等。偏**论文口径 + 数学推导**：会追问 uptrain 的 5% 是怎么来的、为什么组内平均 K/V 权重是合理初始化、以及 $H_{kv}$ 取多少是质量与速度的折中点（论文在 8 组附近给的就是这个折中）。
- **Mistral AI**：Mistral 7B 是 32 个 query head + 8 组 GQA，并配 sliding window attention（来源：GQA 博客）。偏**工程实现与推理成本**：更可能让你现场写出分头/合并与 KV cache 的字节账，并追问长上下文下 cache 与窗口机制怎么配合。
- **阿里巴巴（Qwen）**：Qwen 系列在 HuggingFace config 里同样用 `num_attention_heads` 与 `num_key_value_heads` 两个字段表达头数，不同尺寸的组数不同。偏**大模型服务化**：追问点通常落在引擎侧的 GQA kernel 支持、KV cache 显存与并发、以及张量并行下 KV 头的切分（[[inference-serving-07]]、[[inference-serving-11]]）。

## 相关题目

- [[coding-01]]：本题调用的 `attention` 就是那里的 scaled dot-product attention，掩码与 softmax 数值稳定的细节都在那一题。
- [[coding-03]]：把这里的实现接上 KV cache 与单步 decode，就能看到 GQA 省下的字节真正体现为吞吐。
- [[llm-internals-04]]：GQA 之外的另一条压缩路线（MLA），以及三者（GQA、量化、latent）的层次差异。
- [[inference-serving-03]] 与 [[inference-serving-11]]：KV 头数变少如何影响 PagedAttention 的 block 与共享粒度，以及什么时候必须依赖引擎自带的 GQA kernel 而不是自己写扩容。
- 理论对照：MQA/GQA 的取舍与 `num_key_value_heads` 的读法见 [[llm-internals-03]]；KV cache 的完整推导见 [[llm-internals-02]]；FlashAttention 的动机见 [[llm-internals-05]]。

## 参考资料与归属

- Amit Shekhar (Outcome School)，*Transformer 中的 Multi-Head Attention*，<https://outcomeschool.com/blog/multi-head-attention-in-transformers>：multi-head attention 的分头、并行、拼接与输出投影，以及 $d_k = d_{model}/h$ 的口径。
- Amit Shekhar (Outcome School)，*分组查询注意力（GQA）*，<https://outcomeschool.com/blog/grouped-query-attention>：GQA/MQA 的定义与对比表、$H_{kv}=H$ 与 $H_{kv}=1$ 的两个退化情形、`num_key_value_heads` 术语、uptrain 与 5% 的说法、以及 LLaMA 2/3 与 Mistral 7B 的公开使用情况。
- Joshua Ainslie 等，*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*（延伸），EMNLP 2023，2023-05-22，<https://arxiv.org/abs/2305.13245>：uptrain 配方与 5% 原始预训练算力、GQA 用多于 1 少于 query 头数的 KV 头数、以及「uptrained GQA 质量接近 MHA、速度接近 MQA」的结论。
- Noam Shazeer，*Fast Transformer Decoding: One Write-Head is All You Need*（延伸），2019-11-06，<https://arxiv.org/abs/1911.02150>：MQA 的动机——增量推理慢在反复加载大 K/V 张量的内存带宽成本上，共享 K/V 头可大幅缩小这些张量，实验上解码更快而质量只有轻微下降。

MQA 与 GQA 论文只用于引用动机与结论口径（带宽瓶颈、5% uptrain、GQA 的定义与折中定位）；参数量、KV cache 账本、数值误差、显存与耗时全部是按本仓库统一常数（LLaMA-3-70B：80 层、$H=64$、$H_{kv}=8$、$d_{head}=128$、bf16 即 320 KiB/token；H100 SXM5 bf16 稠密 989 TFLOPs、HBM3 3.35 TB/s、roofline 拐点 295 FLOPs/byte）自行复算与实测的结果：CPU 部分的数值来自 `.work/coding02_doc.py` 与 `.work/coding02_ref.py`，GPU 耗时与显存来自 `.work/coding02_bench.py`（原始输出见第 4 节的两个输出块）；测量设备为 RTX 5070 Laptop 8 GB，GPU 耗时与显存数字不可外推到 H100。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
