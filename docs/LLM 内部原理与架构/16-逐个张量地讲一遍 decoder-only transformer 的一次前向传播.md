---
type: question
id: llm-internals-16
topic: LLM 内部原理与架构
order: 16
question: 请逐个张量地讲一遍 decoder-only transformer 一次前向传播中发生的过程。
question_en: Walk me through what happens, tensor by tensor, in one forward pass of a decoder-only transformer.
asked_at: [Anthropic]
level: 高阶
tags: [前向传播, 张量形状, 综合题]
sources:
  - title: Decoding Transformer Architecture
    url: https://outcomeschool.com/blog/decoding-transformer-architecture
    author: Amit Shekhar (Outcome School)
    published: 2026-04-07
  - title: LLaMA 2：Open Foundation and Fine-Tuned Chat Models（延伸）
    url: https://arxiv.org/abs/2307.09288
    author: Hugo Touvron et al. (Meta AI)
    published: 2023-07-18
  - title: RoFormer：Enhanced Transformer with Rotary Position Embedding（延伸）
    url: https://arxiv.org/abs/2104.09864
    author: Jianlin Su et al.
    published: 2021-04-20
  - title: Root Mean Square Layer Normalization（延伸）
    url: https://arxiv.org/abs/1910.07467
    author: Biao Zhang, Rico Sennrich
    published: 2019-10-16
  - title: GLU Variants Improve Transformer（延伸）
    url: https://arxiv.org/abs/2002.05202
    author: Noam Shazeer
    published: 2020-02-09
related: [llm-internals-01, llm-internals-14, llm-internals-15, inference-serving-01, inference-serving-09]
updated: 2026-09-28
---

## 一句话答案

> 一次前向传播是一条形状流水线：token id $[B,S]$ 查表成 embedding $[B,S,d]$，穿过 $L$ 层「RMSNorm → GQA 注意力 → 残差 → RMSNorm → SwiGLU FFN → 残差」，最后 final norm + lm_head 得到 logits $[B,S,V]$。
> 全程只有注意力分数 $[B,H,S,S]$ 是 $S^2$ 量级，其余算子都是逐 token 的 1×1 映射；真正的形状难点只有一处——把 $[B,S,H,d_{head}]$ 正确 reinterpret 成 $[B,H,S,d_{head}]$。
> 训练对 $[B,S,V]$ 与右移一位的标签算交叉熵；推理 prefill 只取最后位置得 $[B,V]$ 采样，decode 则把 $S=1$ 喂进同一套算子并把新 K/V 追加进 cache。

60 秒口头版，照这 8 步讲：

1. $[B,S]$ 整数 id 查表成 $[B,S,d]$，位置信息此刻还没有。
2. 进层：RMSNorm → 三路投影得 Q $[B,S,H d_{head}]$、K/V $[B,S,H_{kv} d_{head}]$。
3. reshape/transpose 成 $[B,H,S,d_{head}]$，对 Q、K 施加 RoPE。
4. $QK^\top/\sqrt{d_{head}}$ 得 $[B,H,S,S]$，加因果掩码后 softmax。
5. 乘 V 得 $[B,H,S,d_{head}]$，合并头后过 $W_o$，残差相加。
6. RMSNorm → SwiGLU：up/gate 升到 $[B,S,d_{ff}]$，逐元素乘后 down 回 $[B,S,d]$，再残差相加。
7. 32 层重复第 2–6 步，最后一次 final norm。
8. lm_head 出 $[B,S,V]$：训练算交叉熵，推理取最后位置 $[B,V]$ 采样；decode 时 $S=1$、位置取 cache 长度。

这 8 步真正考的不是背架构图，而是能不能把模型当成一张会流动的形状图来读：每个算子先说清输入输出形状，再说清它跨不跨 token。

## 面试官在考什么

- **是不是真的读过 / 写过模型代码**：形状错一位（head 维与 seq 维互换、$d_{ff}$ 写成 $4d$）立刻暴露只背过架构图。这是整个专题的综合验收，任何一环含糊都会被抓。
- **能不能区分「架构」与「一次具体运行」**：架构是静态计算图，前向传播是它在给定 $(B,S)$ 下的一次实例化，所有中间张量都必须带 batch 维和 seq 维。
- **是否知道哪些算子跨 token**：attention 是唯一的 token 间通信算子，FFN / RMSNorm / 残差全部逐位置独立。这一句能推出后续一串结论——FFN 是参数与 FLOPs 的大头、$O(S^2)$ 只出在注意力、KV cache 只缓存注意力的中间态。
- **形状之外的数值细节**：掩码为什么加 $-\infty$、logits 为什么 fp32 累加、final norm 为什么不能省、weight tying 省了多少参数。这是「跑通过」与「读过论文」的分界。
- **prefill 与 decode 是不是两个形状问题**：$S=1024$ 是矩阵乘矩阵（compute-bound），$S=1$ 是矩阵乘向量（memory-bound）。同一份代码，瓶颈完全不同。

常见错误答案：

- 「输入过 embedding，然后 attention 和 FFN，最后 softmax 出词」——没有任何形状与中间张量名，等于没说。
- 把注意力分数说成 $[B,S,S]$ 而丢掉 head 维，或把 $W_k, W_v$ 写成 $d \times d$（GQA 下是 $d \times H_{kv} d_{head}$，这正是省参数的位置）。
- 认为掩码是 softmax 之后把上三角置零——那样每行概率之和不再为 1，等于悄悄缩小了整行注意力。

## 原理与推导

### 阶段 → 形状 → 关键算子

| # | 阶段 | 输出形状 | 关键算子 |
| --- | --- | --- | --- |
| 0 | tokenizer | $[B,S]$ int32 | BPE，无梯度 |
| 1 | embedding 查表 | $[B,S,d]$ | $W_{emb}[V,d]$ 索引 |
| 2 | pre-RMSNorm | $[B,S,d]$ | 逐 token 归一 × $\gamma[d]$ |
| 3 | QKV 投影 | Q $[B,S,H d_{head}]$；K/V $[B,S,H_{kv} d_{head}]$ | 三个 matmul，无 bias |
| 4 | reshape/transpose | $[B,H,S,d_{head}]$（K/V 为 $[B,H_{kv},S,d_{head}]$） | view，只动布局 |
| 5 | RoPE | 同形状 | 按位置旋转，只动 Q/K |
| 6 | scores | $[B,H,S,S]$ | $QK^\top/\sqrt{d_{head}}$ |
| 7 | 因果掩码 + softmax | $[B,H,S,S]$ | 上三角加 $-\infty$，沿 key 维归一 |
| 8 | 加权求和 ×V | $[B,H,S,d_{head}]$ | 每个 query 位置一个输出 |
| 9 | 合并头 + $W_o$ | $[B,S,d]$ | transpose + reshape + matmul |
| 10 | 残差相加 | $[B,S,d]$ | 加回 norm 之前的输入 |
| 11 | post-attn RMSNorm + SwiGLU | up/gate $[B,S,d_{ff}]$ → down $[B,S,d]$ | 两个 matmul + 逐元素乘 |
| 12 | 残差相加（第 2 次） | $[B,S,d]$ | 回到残差流宽度 |
| 13 | final norm + lm_head | $[B,S,V]$ | $[d,V]$ 投影，与 $W_{emb}$ 常共享 |
| 14 | 训练 / 推理分叉 | 损失标量 / $[B,V]$ | 交叉熵 / 取最后位置采样 |

### 贯穿全程的配置

用 LLaMA-2 7B 的主体超参走完，每个数字都可复算：$B=2$、$S=1024$、$d=4096$、$H=32$、$d_{head}=128$、$L=32$、$V=32000$、$d_{ff}=11008$、bf16，只把 KV 头数取成 $H_{kv}=8$ 的 GQA——7B/13B 本身是 32 个 KV 头的 MHA，8 头 GQA 是 34B/70B 与 LLaMA-3 8B 的设置（两个口径的参数账见本节末尾）。注意 $H \cdot d_{head} = 32 \times 128 = 4096 = d$：投影矩阵是方阵，这是刻意设计。

### 第 1–4 步：embedding、RMSNorm 与 QKV 投影

`input_ids` 是 $[B,S]$ 整数张量，embedding 是查表而非矩阵乘：$x = W_{emb}[\text{ids}]$。$W_{emb}$ 自身 $V \times d = 32000 \times 4096 = 131{,}072{,}000$ 参数。查表不注入任何位置信息——$[B,S,d]$ 的任意两行可交换，顺序由 RoPE 补上。

pre-norm 的含义是 norm 放在子层入口：第 $l$ 层注意力吃的是 $\text{RMSNorm}(x)$，而残差相加用的是**未归一化的** $x$。这保证残差流是一条近似恒等的直通路，梯度可以无阻回传，才堆得起 32 层甚至上百层。

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma, \qquad \gamma \in \mathbb{R}^{d}$$

与 LayerNorm 的差别是不减均值、也没有 bias，只有一个 $\gamma$；省掉均值统计意味着少一次归约，对张量并行更友好。

投影的形状要写准，GQA 省参数就省在 K/V 上：

$$W_q \in \mathbb{R}^{d \times H d_{head}}, \quad W_k, W_v \in \mathbb{R}^{d \times H_{kv} d_{head}}, \quad W_o \in \mathbb{R}^{d \times d}$$

代入配置：$W_q$ 与 $W_o$ 各 $4096 \times 4096$，$W_k, W_v$ 各 $4096 \times 1024$（$8 \times 128 = 1024$）。实现上三路通常拼成一个 $W_{qkv} \in \mathbb{R}^{d \times (H + 2H_{kv}) d_{head}}$ 一次算完再 `split`，参数量不变，少两次 kernel 启动。若 $H_{kv} = H$，四路即各 $d \times d$，正是常说的「attention 每层 $4d^2$ 参数」。

reshape 到 $[B,H,S,d_{head}]$ 时顺序不能反：先把最后一维切成 $(H, d_{head})$ 再交换 $S$ 与 $H$；反过来会把 $S$ 和 $H$ 混进同一维，这类 bug 不报错，只让 loss 卡住。转置后内存不连续，习惯上 `.contiguous()`，或让 kernel 直接吃 strided 输入（FlashAttention 走后者）。

### 第 5 步：RoPE 施加在 Q/K 上

$d_{head}=128$ 被切成 64 个二维子空间，第 $i$ 组用频率 $\theta_i = 10000^{-2i/d_{head}}$、按位置 $m$ 旋转 $m\theta_i$：

$$\tilde{q}_m = R_{\Theta,m} q_m, \qquad \tilde{k}_n = R_{\Theta,n} k_n, \qquad \tilde{q}_m^\top \tilde{k}_n = q_m^\top R_{\Theta,n-m} k_n$$

第二个等号来自 $R$ 的正交性与 $R_a^\top R_b = R_{b-a}$：内积只依赖相对位置 $n-m$，正是注意力分数想要的形式。旋转对每个 head 独立施加，各头共用同一套 $\theta_i$，位置信息写进 K 的每个分量。GQA 里 32 个 query 头共享 8 组 K/V 与 RoPE 无关，那是为了砍掉 K/V 的投影参数与 cache 带宽：每个 KV 头被 4 个 query 头复用（实现上 `repeat_interleave`，或让 kernel 按 $h \to \lfloor h/4 \rfloor$ 直接取组）。

两个实现坑：`rotate_half` 有「相邻两维配对」与「前后半段配对」两种约定，数学等价（差一次固定置换），但换约定必须重排权重才能加载已有 checkpoint；decode 时位置 $m$ 必须取 cache 的绝对长度，不能从 0 重数——形状完全正确，输出会错（第 4 节给出可复现的偏差）。

### 第 6–8 步：注意力，唯一跨 token 的算子

$$\text{scores} = \frac{Q K^\top}{\sqrt{d_{head}}} \in \mathbb{R}^{B \times H \times S \times S}, \qquad \text{attn} = \text{softmax}\big(\text{scores} + M\big) V$$

$1/\sqrt{d_{head}}$ 把内积方差压回 $O(1)$：$q,k$ 各分量独立、均值 0 方差 1 时 $q^\top k$ 的方差是 $d_{head}$，除以 $\sqrt{d_{head}}$ 后标准差回到 1，softmax 不会进饱和区（推导见 [[llm-internals-01]]）。

掩码 $M$ 是 $[S,S]$ 下三角矩阵（$j \le i$ 为 0，否则 $-\infty$），广播到 $[B,H,S,S]$。**必须在 softmax 之前加**：$\exp(-\infty)=0$ 才会同时从 softmax 的分子和分母中消失，行和精确为 1；若在 softmax 之后把上三角置零，分母里未来的那份质量已经被算进去了，剩下位置的概率之和小于 1。用 0 代替 $-\infty$ 加到 score 上也不行（$\exp(0)=1$ 会变成对未来的均匀注意力），实践中用 `-inf` 或 dtype 极小值（bf16 下常用 `-1e30`）。

一条实现级约束：用 `-inf` 时每行至少要有一个可见位置。因果掩码保证 $j=0$ 永远可见，所以不会出问题；但 padding mask 与 causal mask 叠加、整行被 padding 掉时，该行全 $-\infty$ 会让 softmax 出 NaN，自写实现通常把全掩码行手动置 0。

softmax 沿 key 维做，输出每行和为 1，再乘 $V \in [B,H,S,d_{head}]$ 得到 $[B,H,S,d_{head}]$。scores 张量在 $B=2, H=32, S=1024$ 时有 $2 \times 32 \times 1024^2 \approx 6.7 \times 10^7$ 个元素，bf16 下 128 MiB、fp32 下 256 MiB，**每层每头都要物化一份**；$S=8192$ 就是 8 GiB。FlashAttention 用 online softmax 在 SRAM 里分块累加，从不把 $[B,H,S,S]$ 写回 HBM，数学上完全等价。

### 第 9–10 步：合并头、输出投影、残差

$[B,H,S,d_{head}]$ 转置回 $[B,S,H,d_{head}]$ 再 reshape 成 $[B,S,d]$：元素顺序不变（因为 $H d_{head} = d$），只是转置后内存不连续，要么 `.contiguous()`、要么让 kernel 直接吃 strided 输入；再乘 $W_o$ 得到 $[B,S,d]$。$W_o$ 负责把 32 个头的子空间结果线性混合回 4096 维，多头的作用在这里收口。

然后 `x = x + attn_out`，加的是**进入本层注意力之前**的 $x$，不是 norm 之后的张量——这是 pre-norm 在代码上的唯一标志，也是抄错代码的重灾区。

### 第 11–12 步：SwiGLU FFN 与残差相加

$$\text{FFN}(x) = W_{down}\big(\text{SiLU}(W_{gate} x) \odot W_{up} x\big)$$

$W_{gate}, W_{up} \in \mathbb{R}^{d \times d_{ff}}$，$W_{down} \in \mathbb{R}^{d_{ff} \times d}$，逐元素乘要求两个中间张量都是 $[B,S,d_{ff}]$。$\text{SiLU}(z) = z \cdot \sigma(z)$，去掉门控就是普通 MLP。这里 $11008 / 4096 = 2.6875 \approx \frac{8}{3}$（$\frac{8}{3}d = 10922.67$，向上取到 256 的倍数正好是 11008）而非 $4$：GLU 变体比标准 FFN 多一个矩阵，取 $\frac{2}{3} \times 4d$ 可以把参数与 FLOPs 拉回与 $4d$ 普通 FFN 相当的水平，是有意对齐预算，不是随手取的数。

FFN 是整层参数的大头：$3 \times 4096 \times 11008 = 135{,}266{,}304$，而注意力四路投影合计 $16{,}777{,}216 + 4{,}194{,}304 + 4{,}194{,}304 + 16{,}777{,}216 = 41{,}943{,}040$，前者是后者的 3.2 倍。$W_{down}$ 的输出再与残差流相加（第 2 次残差），一层的形状流水线到此闭环。

### 第 13–14 步：final norm、lm_head 与分叉

循环结束后还要过一次 RMSNorm。理由有两条：残差流累加 32 层后尺度会漂移，而 lm_head 直接吃它；且 pre-norm 下每层子层看到的都是归一化输入，唯独最后一层的输出从未归一化，这一步是补齐。

lm_head 是 $[d, V] = 131{,}072{,}000$ 参数，输出 $[B,S,V]$，即 2×1024×32000 ≈ 6550 万个 logits。weight tying 取 $W_{lm\_head} = W_{emb}^\top$：省掉 1.31 亿参数，并把输入侧与输出侧的语义空间绑成同一个几何结构，代价是两边梯度互相干扰。整个前向里最需要 fp32 的就是这一层——$V=32000$ 量级的归约在 bf16 下尾数只有 8 位，logits 通常上采到 fp32 再算 softmax 与采样。

训练与推理在此分叉：训练把 $[B,S,V]$ 与右移一位的标签算交叉熵，$S$ 个位置同时有监督信号；推理 prefill 同样算满 $[B,S,V]$ 但只取最后位置得 $[B,V]$，其余位置是填 cache 的副产物；decode 的 $S=1$，输出 $[B,1,V]$，只算新 token 的 Q/K/V 并追加入 cache。

### 参数量与 FLOPs 的量级

| 组件 | 单层 | ×32 层 |
| --- | --- | --- |
| 注意力四路投影（GQA） | 41,943,040 | 1,342,177,280 |
| FFN（SwiGLU 三矩阵） | 135,266,304 | 4,328,521,728 |
| 两个 RMSNorm gamma | 8,192 | 262,144 |
| 小计 | 177,217,536 | 5,670,961,152 |

加上 $W_{emb} = V \times d = 131{,}072{,}000$，与 lm_head 共享时总计 5,802,033,152 ≈ 5.80 B；把 lm_head 当独立张量另算一份（朴素计数法），总计 5,933,105,152 ≈ 5.93 B。业界常引用的 7B ≈ 6.7 B 是「MHA + 输入输出各算一份」的口径：$32 \times (4d^2 + 3 d d_{ff} + 2d) + d + 2Vd = 6{,}738{,}415{,}616 \approx 6.74$ B，比 5.93 B 多出的 805,310,464（≈0.81 B）正是 32 层 GQA 省下的 K/V 投影（$32 \times 25{,}165{,}824 = 805{,}306{,}368$）再加一个 final norm 的 $d = 4096$。全篇一律以可逐项复算的 5.80 B / 5.93 B 为准，口径写明「上表逐张量相加」。

FLOPs 上，每个参数每 token 一次乘加 = 2 FLOPs，权重矩阵部分即 $2 N_{params} S$（$N$ 取上表 5.80 B、$S=1024$ 时约 11.9 TFLOP，摊到每 token 是 11.6 GFLOP）。逐算子复算还要补上没有参数的两项——$QK^\top$ 与 $\text{softmax}\cdot V$（各 $2 B H S^2 d_{head}$，下面的式子按 $B=1$ 的每序列口径）：每 token 每层 $4 H S d_{head} = 4 \times 32 \times 1024 \times 128 \approx 1.68 \times 10^7$ FLOPs，32 层合计 ≈0.54 GFLOP，于是每 token ≈12.1 GFLOP、每序列 ≈12.4 TFLOP：

$$\text{FLOPs} \approx \underbrace{2 N_{params} S}_{\text{所有权重矩阵}} + \underbrace{4 L H S^2 d_{head}}_{\text{每层 } QK^\top + \text{softmax}\cdot V}$$

两项之比 $2 H d_{head} S/(4d^2 + 3 d d_{ff}) = 8192S/202{,}375{,}168 \approx S/24{,}704$：$S=1024$ 时第二项只占约 4%，FFN 与投影占 96% 以上；要到 $S$ 约 2.5 万，注意力的 $S^2$ 项才与权重矩阵项持平（只把 attention 的 $4d^2$ 放进分母会算出 8192，那是漏掉 FFN 的口径）。

### prefill 与 decode：同一条流水线，两种形状

| 维度 | prefill | decode |
| --- | --- | --- |
| $S$ / 主要算子 | 1024，矩阵 × 矩阵 | 1，矩阵 × 向量（GEMV） |
| scores 形状 | $[B,H,S,S]$ | $[B,H,1,S_{cache}]$ |
| 算术强度 | 约 $S$ 量级 FLOPs/Byte | 约 1 FLOP/Byte |
| 瓶颈 | compute-bound | memory-bandwidth bound |
| KV cache | 写入 $S$ 个位置 | 读取 + 追加 1 个位置 |

算术强度的算法：decode 每步读 $5.93\text{B} \times 2\,\text{B} = 11.9$ GB 权重、做约 11.9 GFLOP，比值约 1 FLOP/Byte；prefill 读同样的权重、做 $S$ 倍 FLOPs，比值约 1000 FLOP/Byte。H100 的 bf16 峰值约 990 TFLOPS、HBM 带宽约 3.35 TB/s，脊点约 295 FLOP/Byte，于是 prefill 落在脊点右侧（吃算力），decode 落在最左边（吃带宽）——11.9 GB 权重必须完整过一遍 HBM 才产出 1 个 token，这正是单请求 decode 吞吐上不去、必须靠 batching 摊薄权重读取的原因（详见 [[llm-internals-02]]）。

## 数值与代码验证

上面每一步写成可运行的 PyTorch 骨架（`rope` 只留接口，其余形状全可执行）；参数放在 meta device 上，不真分配显存：

```python
import torch, torch.nn as nn, torch.nn.functional as F

B, S, d, H, Hkv, dh, L, V, dff = 2, 1024, 4096, 32, 8, 128, 32, 32000, 11008

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)      # torch>=2.4；否则手写
        self.wq = nn.Linear(d, H * dh, bias=False)
        self.wk = nn.Linear(d, Hkv * dh, bias=False)         # GQA：KV 投影更窄
        self.wv = nn.Linear(d, Hkv * dh, bias=False)
        self.wo = nn.Linear(H * dh, d, bias=False)
        self.wgate, self.wup = nn.Linear(d, dff, bias=False), nn.Linear(d, dff, bias=False)
        self.wdown = nn.Linear(dff, d, bias=False)

    def forward(self, x, pos, cache=None):
        b, s = x.shape[:2]
        h = self.n1(x)
        q = self.wq(h).view(b, s, H, dh).transpose(1, 2)          # [B,H,S,dh]
        k = self.wk(h).view(b, s, Hkv, dh).transpose(1, 2)        # [B,Hkv,S,dh]
        v = self.wv(h).view(b, s, Hkv, dh).transpose(1, 2)
        q, k = rope(q, pos), rope(k, pos)                         # 只动 Q/K
        if cache is not None:                                     # decode：追加再读全量
            cache["k"] = torch.cat([cache["k"], k], dim=2)
            cache["v"] = torch.cat([cache["v"], v], dim=2)
            k, v = cache["k"], cache["v"]
        k = k.repeat_interleave(H // Hkv, dim=1)                  # 8 -> 32 头，逻辑上复制
        v = v.repeat_interleave(H // Hkv, dim=1)
        scores = q @ k.transpose(-1, -2) / dh ** 0.5              # [B,H,S,Scache]
        sq, sk = scores.shape[-2], scores.shape[-1]
        keep = torch.ones(sq, sk, dtype=torch.bool).tril(sk - sq)
        attn = scores.masked_fill(~keep, float("-inf")).softmax(-1) @ v    # 掩码先于 softmax
        x = x + self.wo(attn.transpose(1, 2).reshape(b, s, d))    # 残差加 n1 的输入
        n = self.n2(x)
        return x + self.wdown(F.silu(self.wgate(n)) * self.wup(n))

def rope(t, pos):                                                 # [B,H,S,dh]，旋转构造见第 3 节
    ...

with torch.device("meta"):
    emb = nn.Embedding(V, d)
    layers = nn.ModuleList([Layer() for _ in range(L)])
    head = nn.Linear(d, V, bias=False)
    head.weight = emb.weight                                      # weight tying
n_attn = sum(p.numel() for p in [layers[0].wq.weight, layers[0].wk.weight,
                                 layers[0].wv.weight, layers[0].wo.weight])
n_ffn = sum(p.numel() for p in [layers[0].wgate.weight, layers[0].wup.weight, layers[0].wdown.weight])
n_norm = sum(p.numel() for p in [layers[0].n1.weight, layers[0].n2.weight])
body, n_emb = L * (n_attn + n_ffn + n_norm), emb.weight.numel()
print("单层 attn", n_attn, "ffn", n_ffn, "norm", n_norm)
print("全部层", body, "emb", n_emb, "tied 总计", body + n_emb, "untied 总计", body + 2 * n_emb)
print("4*d*d 对照", 4 * d * d, " 3*d*dff", 3 * d * dff)

# 行为校验：cache 路径应与「一次算完」等价；位置从 0 重数会出错
fresh = lambda: {"k": torch.zeros(1, Hkv, 0, dh), "v": torch.zeros(1, Hkv, 0, dh)}
torch.manual_seed(0)
with torch.device("cpu"):
    m, x, pos = Layer().eval(), torch.randn(1, 8, d), torch.arange(8)[None]
    with torch.no_grad():
        full = m(x[:, :4], pos[:, :4], None)                      # 一次算完 4 个位置
        kv = fresh()
        m(x[:, :3], pos[:, :3], kv)                               # prefill 3 个位置
        step = m(x[:, 3:4], pos[:, 3:4], kv)                      # decode 第 4 个
        kv = fresh()                                              # 换一条干净的 cache
        m(x[:, :3], pos[:, :3], kv)
        bad = m(x[:, 3:4], pos[:, :1], kv)                        # 位置从 0 重数
    print("cache vs 全量", (full[0, 3] - step[0, 0]).abs().max().item(),
          "| 位置重数偏差", (full[0, 3] - bad[0, 0]).abs().max().item())
```

实测输出（前两行与上表逐项一致；行为校验随随机初始化与硬件浮动）：

```text
单层 attn 41943040 ffn 135266304 norm 8192
全部层 5670961152 emb 131072000 tied 总计 5802033152 untied 总计 5933105152
4*d*d 对照 67108864  3*d*dff 135266304
cache vs 全量 8.940696716308594e-07 | 位置重数偏差 0.09869331121444702
```

第四行值得单独讲：两条路径的输入张量完全相同，差别只在「走 cache」还是「一次算完」，偏差 1e-6 量级，纯属浮点重结合；而把 decode 的位置从 0 重数，形状毫无变化，同一 token 的输出却偏了 0.099。必须复算而不是凭印象的数字：

| 量 | 计算式 | 结果 |
| --- | --- | --- |
| 单层 attention（GQA） | $d^2 + 2 H_{kv} d_{head} d + d^2$ | 41,943,040（≈0.42 亿） |
| 「4d²」口径 | $4 \times 4096^2$ | 67,108,864（比 GQA 多 60%，是 MHA 的数） |
| 单层 FFN | $3 \times 4096 \times 11008$ | 135,266,304（≈1.35 亿） |
| 全部层 | $32 \times 177{,}217{,}536$ | 5,670,961,152（≈56.7 亿） |
| tie 省下的参数 | $V \times d$ | 131,072,000（≈1.31 亿） |
| 每序列 FLOPs（权重矩阵项） | $2 \times 5.80\text{e}9 \times 1024$ | ≈11.9 TFLOP（再加注意力的 0.55 TFLOP 才是全量） |
| 每 token FLOPs（权重矩阵项） | $2 \times 5.80\text{e}9$ | ≈11.6 GFLOP（全量 ≈12.1 GFLOP） |
| 每层 $QK^\top$（$B{=}2$） | $2 \times 2 \times 32 \times 1024^2 \times 128$ | 17.18 GFLOP |
| KV cache（$B{=}2$，bf16） | $2LBH_{kv}d_{head}S \times 2$ B | 256 MiB |
| 单 token KV（每序列） | $2 \times 32 \times 8 \times 128 \times 2$ B | 128 KiB |
| eager scores 张量（$B{=}2$） | $2 \times 32 \times 1024^2$ 个元素 | 128 MiB（bf16）/ 256 MiB（fp32） |

最后一行是长上下文的分水岭：$S=1024$ 时 scores 要物化 128 MiB，32 层每层一次；$S$ 翻到 8192 就是 8 GiB。这个 $S^2$ 比参数量、比 KV cache（都是 $S$）都长得快，所以换 attention kernel 不是优化选项，而是能否跑起来的前提。

## 常见追问

- **追问**：为什么掩码必须在 softmax 之前加，而不是把 softmax 之后的权重置 0？
  - 要点：softmax 的分母对所有位置求和。先置 0 再归一化等于从分母里凭空删掉未来那份质量，剩下位置的概率之和小于 1；只有先把分数压到 $-\infty$，$\exp(-\infty)=0$ 才同时从分子和分母中消失，行和仍精确为 1。
- **追问**：最后一层之后为什么还要 final norm？去掉会怎样？
  - 要点：残差流是 32 层累加的结果，尺度可能漂移，而 lm_head 直接吃它；且 pre-norm 下每层子层看到归一化输入，只有最后一层的输出没被归一化，这一步是补齐。去掉后短序列也许能跑，长序列或训练早期容易出现 logits 溢出。
- **追问**：lm_head 与 embedding 共享权重吗？取舍是什么？
  - 要点：weight tying 让 $W_{lm\_head} = W_{emb}^\top$，省下 $V \times d$（本题 1.31 亿参数），并让输入输出共享语义几何。代价是输入侧（要求 embedding 区分 token）与输出侧（要求 logits 线性可分）的梯度互相干扰；词表占比大的小模型通常 tie，超大模型倾向解绑。
- **追问**：logits 为什么要用 fp32 累加？
  - 要点：lm_head 要在 $d=4096$ 上归约出 32000 个 logit，bf16 尾数只有 8 位，几千项累加的舍入误差足以改变排序（尤其尾部 token 的采样概率），后面还要做 $\exp$ 把误差放大。常规做法是权重 bf16、累加与 softmax 用 fp32。
- **追问**：KV cache 插在哪一步？decode 时哪些张量换了形状？
  - 要点：插在 RoPE 之后、$QK^\top$ 之前——cache 里存的是旋转后的 K。decode 时 Q 是 $[B,H,1,d_{head}]$，K/V 是 $[B,H_{kv},S_{cache}+1,d_{head}]$，只算新 token 的投影再 `cat` 追加；位置 $m$ 取 cache 的绝对长度（详见 [[llm-internals-02]]、[[llm-internals-14]]）。
- **追问**：GQA 的形状差别具体落在哪几行代码？
  - 要点：只在 $W_k, W_v$ 的输出维度（$H_{kv} d_{head}$ 而非 $H d_{head}$）与 head 维的 `repeat_interleave`；Q/O 与整个 FFN 不变。参数量上 K/V 投影从 $2d^2$ 降到 $2 H_{kv} d_{head} d$，本题每层省 $2 \times 4096^2 - 2 \times 1024 \times 4096 \approx 2517$ 万（详见 [[llm-internals-03]]）。

## 公司变体

`asked_at` 里只有 Anthropic。从公开的岗位描述与技术材料看，这类岗位的面试同时要求把形状推到底和把「为什么这样设计」讲清楚，原理推导与工程实现大致各占一半：

- **偏原理推导**：$1/\sqrt{d_{head}}$ 的方差论证、RoPE 为什么保持相对位置、pre-norm 为什么让深网络可训、final norm 与 weight tying 的取舍。给出公式再加一句「为什么」，比罗列形状更能体现水平。
- **偏工程实现**：$[B,S,H,d_{head}]$ 与 $[B,H,S,d_{head}]$ 的转置时机、掩码加在 softmax 之前、fp32 累加、prefill 与 decode 的形状分叉、KV cache 的插入点与位置编码的绝对性。这些是「自己跑过模型」与「只读过论文」的分界。
- **常见形式**：从 $[B,S]$ 开始让你一路写形状，中途把 $S$ 改成 1（decode）或把 $H_{kv}$ 改小（GQA），看你能不能就地指出哪些张量变了、哪些没变。

以上是该岗位公开面试取向的概括，不代表任何具体的面试流水。

## 相关题目

- [[llm-internals-01]]：$1/\sqrt{d_{head}}$ 的方差推导与 scaled dot-product attention 的定义。
- [[llm-internals-02]]：KV cache 的显存公式、prefill/decode 的瓶颈分析与 PagedAttention。
- [[llm-internals-03]]：MQA / GQA 的取舍，对应本题 $W_k, W_v$ 的形状差异。
- [[llm-internals-14]]、[[llm-internals-15]]：位置编码与归一化、激活函数的深入版本。
- [[inference-serving-09]]：把这条流水线的形状与瓶颈延伸到线上服务的吞吐与延迟指标。

## 参考资料与归属

- Amit Shekhar (Outcome School)，*Decoding Transformer Architecture*，2026-04-07，<https://outcomeschool.com/blog/decoding-transformer-architecture>：整体数据流的分步叙述（tokenize → embedding → attention → FFN → 残差与归一化 → 堆叠层 → 输出投影与采样），以及 encoder-only / decoder-only / encoder-decoder 三种变体的定位。
- Hugo Touvron et al. (Meta AI)，*LLaMA 2: Open Foundation and Fine-Tuned Chat Models*，2023-07-18，<https://arxiv.org/abs/2307.09288>（延伸）：本题贯穿配置 $d=4096$、$H=32$、$L=32$、$V=32000$、$d_{ff}=11008$ 的来源，以及 RMSNorm pre-normalization 与 34B/70B 上 8 个 KV 头的 GQA（论文表 1 写明 7B/13B 不用 GQA，仍是 32 个 KV 头的 MHA）等结构选择。
- Jianlin Su et al.，*RoFormer: Enhanced Transformer with Rotary Position Embedding*，2021-04-20，<https://arxiv.org/abs/2104.09864>（延伸）：RoPE 的旋转构造与相对位置性质。
- Biao Zhang, Rico Sennrich，*Root Mean Square Layer Normalization*，2019-10-16，<https://arxiv.org/abs/1910.07467>（延伸）：RMSNorm 公式与去掉均值中心化的动机。
- Noam Shazeer，*GLU Variants Improve Transformer*，2020-02-09，<https://arxiv.org/abs/2002.05202>（延伸）：SwiGLU 的定义与 $d_{ff}$ 的预算对齐。

参考源 *Decoding Transformer Architecture* 覆盖的是架构级数据流（各组件做什么、encoder 与 decoder 如何协作），没有给出逐张量形状、参数量与 FLOPs 推导。第 3 节的形状贯穿、参数拆解、prefill/decode 算术强度对比，第 4 节的代码与数值，均由按标准实现的独立推导与复算给出，并用上述延伸来源交叉核对。**本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。**
