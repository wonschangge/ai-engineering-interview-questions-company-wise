---
type: question
id: inference-serving-01
topic: 推理、服务与 GPU 性能
order: 1
question: 解释 prefill 和 decode 两个阶段。为什么 prefill 是计算受限的，而 decode 是内存带宽受限的？
question_en: Explain the prefill and decode phases. Why is prefill compute-bound and decode memory-bandwidth-bound?
asked_at: [Moonshot AI, NVIDIA, Together AI]
level: 进阶
tags: [prefill, decode, roofline, 算术强度]
sources:
  - title: "Prefill vs Decode: LLM Inference Optimization"
    url: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [inference-serving-02, inference-serving-10, inference-serving-14, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> prefill 一次吞下整段 prompt（$S$ 个 token），同一份权重被 $S$ 个 token 复用，算术强度约等于 $S$，按 LLaMA-3-70B 的真实形状复算，$S=1024$ 时约 870 FLOPs/byte，远高于 H100 的 roofline 拐点 295 FLOPs/byte，所以受算力限制；decode 每个 step 只处理 1 个新 token，却要把整份权重从 HBM 读一遍，算术强度约 1 FLOP/byte（bf16），所以受带宽限制。
> 换成时间：70B bf16 权重 140 GB，H100 带宽 3.35 TB/s，单卡读一遍至少 41.8 ms，这就是单请求 decode 的 token 间隔下界（23.9 token/s）；tensor parallel 到 8 卡后每卡只读 17.5 GB，上界升到约 191 token/s。
> 两个阶段的差别全部来自那个 $S$。batch 里的 $B$ 条序列共用同一次权重读取，所以 batch 把 decode 的算术强度往计算侧拉（短上下文下约等于 $B$），这正是 continuous batching 的收益来源，也是收益出现拐点的原因。

## 面试官在考什么

- 能否从张量形状（矩阵×矩阵 vs 矩阵×向量）推到瓶颈，而不是只背「prefill 快、decode 慢」这个结论。
- 会不会算 roofline：算术强度 = FLOPs / bytes，再与硬件的算力/带宽比（拐点）作比较，判断落在哪一侧。
- 能否给出可复算的数字：权重字节数、每 token 的 KV 字节数、token/s 上界，并主动交代口径（稠密 bf16、十进制 GB、是否含 KV）。
- 是否知道两个阶段各自决定哪个指标（TTFT 与 TPOT/throughput），以及优化手段为什么天然分成两组。
- 能否解释 batch 的收益为什么会饱和——每多一条序列，就多读一份它自己的 KV cache。

常见错误答案：

- 「decode 慢是因为自回归必须串行」。串行只解释了无法并行，没有解释数学单元为什么闲着。真正的下界是每步必须搬的字节数除以带宽。
- 「prefill 计算受限是因为 attention 是 $O(S^2)$」。$S=1024$ 时二次项只占约 1%（因果掩码口径），主导项是形状规整的线性投影 GEMM；$O(S^2)$ 到 32k 以上才成为主角。
- 只说「decode 算术强度低」而给不出数值。面试官通常会立刻追问「低到多少、拐点是多少、你按哪个口径算的」。

## 原理与推导

### 1. 两个阶段只差一个 $S$

一次请求的时间线是「1 次 prefill + $M$ 次 decode」。模型、权重、kernel 全都一样，差别只在每个 step 送进去几个 token：

| 维度 | prefill | decode |
| --- | --- | --- |
| 每步输入 | $[S, d]$，整段 prompt | $[1, d]$，上一个 token |
| 线性层形状 | $[S,d]\times[d,d']$，矩阵×矩阵（GEMM） | $[1,d]\times[d,d']$，矩阵×向量（GEMV） |
| 步数 | 1 | $M$（输出长度） |
| KV cache 动作 | 写入 $S$ 个位置 | 读全部历史，追加 1 个位置 |
| 每步运算量 | $2NS + 2S^2 d_{model} L$ 量级 | $2N$ |
| 每步必须读的字节 | $2N$（权重）+ 新写入的 KV | $2N$（权重）+ 全部历史 KV |

表中 $N$ 指参数量（约 $70\times10^9$）、$M$ 指输出 token 数、$d_{model}=8192$、$L=80$，即 70B 级的常用口径；第 3、4 小节会把这两个数字分别代进去。

顺带一个容易被忽略的后果：$M$ 次 decode step 每一次都要完整读一遍权重，所以一次请求的总访存量是 $O(M\cdot N)$ 量级，而 prefill 只读一遍权重。输出越长，decode 越主导端到端时间和服务成本。

### 2. roofline：算术强度落在拐点的哪一侧

kernel 的执行时间下界取「算」与「搬」中更慢的那个：

$$T \ge \max\left(\frac{\text{FLOPs}}{P_{\text{peak}}}\;,\;\frac{\text{Bytes}}{BW}\right)$$

记算术强度 $I = \text{FLOPs}/\text{Bytes}$、硬件拐点 $I^{*} = P_{\text{peak}}/BW$，则 $I > I^{*}$ 时算力受限，$I < I^{*}$ 时带宽受限。H100 SXM 按厂商数据表口径（bf16 稠密 989 TFLOPs，不含 2:4 稀疏的 1979；HBM3 3.35 TB/s）：

$$I^{*} = \frac{989\times10^{12}}{3.35\times10^{12}} \approx 295\ \text{FLOPs/byte}$$

这个拐点对结论不敏感：同一代 A100 80GB 是 $312/2.04 \approx 153$；若按「有效算力」口径，比如 prefill 实跑 50% MFU（表 1 的口径），H100 的拐点降到 $494/3.35 \approx 148$，实跑 40% MFU 则是约 118。几种口径给出的拐点都在 118–295 之间，而下面要算的两个阶段分别是约 1 和 870 以上：decode 比拐点低两个数量级以上，prefill 高出 3–12 倍，所以结论不依赖口径选择。

### 3. decode：一次权重读取只换来一个 token

一个线性层在 decode 时做的是权重矩阵 $W\in\mathbb{R}^{d'\times d}$ 乘一个向量：

$$\text{FLOPs} = 2dd',\qquad \text{Bytes} \approx \underbrace{2dd'}_{\text{权重}} + \underbrace{O(d)}_{\text{激活}} \;\Longrightarrow\; I \approx 1\ \text{FLOP/byte}$$

激活只占 $O(d)$ 字节，相对 $O(dd')$ 的权重可以忽略。于是整个模型在 decode 时每 token 的浮点运算量与必须读的权重字节数在数值上相同（都是 $2N$ 这个量级），$I\approx1$，比拐点低两个数量级以上。

所以 decode 每步的时间下界就是把权重从 HBM 搬进 SM 的时间：

$$t_{\text{step}} \ge \frac{W_{\text{bytes}}}{BW},\qquad \text{tokens/s} \le \frac{BW}{W_{\text{bytes}}}$$

70B bf16 的 $W_{\text{bytes}}=140$ GB，单张 H100 给出 $140\times10^9 / 3.35\times10^{12} = 41.8$ ms，即 23.9 token/s。注意 140 GB 本来就装不进一块 80 GB 的卡，这个数只是「单卡上界」；真实部署必须切分权重。tensor parallel 让每张卡各读自己那一片，聚合带宽变成 卡数 × 单卡带宽：

| TP 度 | 每卡权重 | 每步读权重的下界 | token/s 上界 |
| --- | --- | --- | --- |
| 1 | 140 GB | 41.8 ms | 23.9 |
| 2 | 70 GB | 20.9 ms | 47.9 |
| 4 | 35 GB | 10.5 ms | 95.7 |
| 8 | 17.5 GB | 5.2 ms | 191 |

这就是 decode 的加速主要来自「把权重摊到更多卡上」的原因，也是 tensor parallel 在 decode 阶段比在 prefill 阶段更有存在感的直觉来源（代价是每层两次 all-reduce 的通信与延迟）。

第二个字节来源是 KV cache。按 LLaMA-3-70B 的配置（$L=80$、8 个 KV 头、head_dim 128、bf16），每 token 的 KV 字节数为

$$m_{kv} = 2 \times 2 \times 80 \times 8 \times 128 = 327{,}680\ \text{B} = 320\ \text{KiB}$$

8k 上下文下是 2.7 GB，只占权重的 1.9%，可以忽略；128k 上下文下是 42.9 GB，占 30.7%，此时上界要按 $W_{\text{bytes}} + m_{kv}S$ 重算：$140/(140+42.9) \approx 0.76$，token/s 只剩权重口径的约 76%（掉约 23%）。

### 4. prefill：一份权重喂给 $S$ 个 token

prefill 的线性层是 $[S,d]\times[d,d']$：

$$\text{FLOPs} = 2Sdd',\qquad \text{Bytes} \approx 2dd' + 4Sd \;\Longrightarrow\; I \approx \frac{S}{1 + 2S/d}$$

权重主导（$S \ll d$ 时激活项可略）的情况下，算术强度就是 $S$ 的量级。按 LLaMA-3-70B 逐层相加、把权重与每层激活的读写都算进去复算整模型：$S=1024$ 时约 870 FLOPs/byte，$S=8192$ 时约 3500 FLOPs/byte（只计线性项则分别是 861 与 3258），都高出拐点 3–12 倍。所以 prefill 的时间下界由第一项决定，瓶颈在算力。

代价是 TTFT 随 $S$ 增长。按 8×H100 聚合峰值 7.91 PFLOPs 口径（线性项 + 因果掩码下的 attention 项）：

| prompt $S$ | 总 FLOPs | 峰值下的下界 | 40% MFU 口径 |
| --- | --- | --- | --- |
| 1k | 145 TFLOPs | 18 ms | 46 ms |
| 8k | 1.23 PFLOPs | 156 ms | 390 ms |
| 128k | 40.9 PFLOPs | 5.2 s | 12.9 s |

attention 的二次项确实存在：每层约 $4S^2d_{model}$（无掩码）或 $2S^2d_{model}$（因果掩码），占总 FLOPs 的比例（因果口径，与代码同用 $N=70\times10^9$）随 $S$ 从约 1%（1k）、7.1%（8k）、23.5%（32k）升到 55.1%（128k）。所以「超长 prompt 的 prefill 越来越像 attention 受限」成立，但把 $S\le8$k 的 prefill 说成 $O(S^2)$ 是错的。

### 5. batch 如何把 decode 往计算侧拉

$B$ 条序列的 decode step 共享同一次权重读取：字节数是 $2N + B\,m_{kv}S$，运算量是 $2NB$，于是

$$I(B) = \frac{2NB}{2N + B\,m_{kv}S}$$

- 上下文短、KV 项可忽略时 $I \approx B$，$B$ 接近拐点 295 时 decode 重新变成计算受限。
- 8k 上下文时 $I$ 的渐近上限只有 $2N/(m_{kv}S) = 140/2.68 \approx 52$ FLOPs/byte——分母里的 KV 项会主导，**任何 batch 都回不到计算侧**。能抬高这个上限的只有压 KV（GQA、KV 量化、稀疏注意力）；压权重字节（fp8/int8）让 $W$ 减半、小 batch 区间的 token/s 上界翻倍，但由 KV 决定的渐近上限不变。

| batch $B$ | 每步读取字节 | 算术强度 | 每步时间（TP=8） | 聚合吞吐 | 每条流的 TPOT |
| --- | --- | --- | --- | --- | --- |
| 1 | 143 GB | 1.0 | 5.3 ms | 188 token/s | 5.3 ms |
| 8 | 161 GB | 6.9 | 6.0 ms | 1328 token/s | 6.0 ms |
| 64 | 312 GB | 28.7 | 11.6 ms | 5501 token/s | 11.6 ms |
| 256 | 827 GB | 43.3 | 30.9 ms | 8294 token/s | 30.9 ms |
| 1024 | 2889 GB | 49.6 | 107.8 ms | 9500 token/s | 107.8 ms |

这张表同时给出了收益拐点的来源：聚合吞吐从 $B=64$ 到 1024 只涨 1.7 倍，而每条流自己的 token 间隔从 11.6 ms 恶化到 107.8 ms。权重读取被摊薄的那部分收益很快用完，剩下每次都要多读的 KV 成了纯增量成本，延迟与吞吐开始互相拉扯。短上下文、$B$ 接近 295 时 decode 会重新变成计算受限，这正是 [[inference-serving-10]] 那道手算题要判断的分界。

### 6. 两个阶段各自对应什么指标、什么手段

- 指标映射：$\text{TTFT} \approx$ prefill 时间 $+$ 1 个 decode step；$\text{TPOT/ITL} \approx$ decode step 时间；端到端 $= \text{TTFT} + (M-1)\cdot\text{TPOT}$。取 TTFT 400 ms、输出 200 token、TPOT 25 ms，端到端约 5.4 s，其中 decode 占 93%——所以「prefill 便宜、decode 贵」在时间账上是常态。
- prefill 侧手段：chunked prefill 把长 prompt 切成块、在块之间插入别人的 decode step，它不减少总 FLOPs，改的是尾延迟与其他流量的 ITL（[[inference-serving-14]]）；prefix caching 直接跳过重复前缀的 prefill，省的就是这份算力。
- decode 侧手段：continuous batching 用 $B$ 摊薄权重读取（[[inference-serving-02]]）；KV 量化、GQA、稀疏注意力削减第二个字节来源；fp8/int8 权重让 $W_{\text{bytes}}$ 减半、权重读取口径的 token/s 上界直接翻倍；speculative decoding 让一次权重读取验证多个候选 token，把带宽受限的空隙用满。
- 两阶段的资源需求相反，所以可以分池：prefill 池堆算力、decode 池堆带宽与显存容量（P/D 分离），代价是 KV cache 要走网络——8k 上下文 2.7 GB，NVLink 4 单向约 450 GB/s（公开规格口径）时约 6 ms，400 Gb/s 的 NDR InfiniBand 约 54 ms，后者已经和 prefill 本身同量级，这也是分离式部署必须配高速互联的原因。
- 收口一句：prefill 与 decode 的差异不是实现问题，而是矩阵形状决定的算术强度差异；chunked prefill、continuous batching、P/D 分离这些手段，处理的都是这两个阶段互相冲突的资源需求。

## 数值与代码验证

口径先声明清楚，这几条是面试时最容易被追问的部分：

- H100 SXM：bf16 稠密 989 TFLOPs（稀疏口径 1979，roofline 用稠密值）、HBM3 3.35 TB/s、80 GB，拐点 295 FLOPs/byte。换 SKU 或换代际，拐点会变（A100 80GB 约 153），但结论不变。
- 70B 级模型按 LLaMA-3-70B 的 `config.json`：80 层、$d_{model}=8192$、64 个注意力头、8 个 KV 头、head_dim 128、FFN 28672；bf16 权重 140 GB（约 130 GiB），每 token KV 320 KiB。
- FLOPs 口径：矩阵乘按「乘加 = 2 FLOPs」，每 token 前向 $2N$（$N$ 为参数量，代码里写作 `P` 以免与层数 `L` 混淆），不计 attention 的 $S^2$ 项、softmax 与各类归一化；attention 项单独列出时用因果掩码口径（无掩码则是两倍）。
- 字节单位：GB 为 $10^9$ 字节，GiB 为 $1024^3$，两者分开写。
- 表里的下界忽略了 kernel 启动、TP all-reduce、HBM 实测效率（通常只有标称的 80–90%）与调度开销，因此是「不可能超过」的上界，不是预期值。

**表 1：硬件的拐点**

| 硬件 | bf16 稠密算力 | HBM 带宽 | 拐点 |
| --- | --- | --- | --- |
| H100 SXM（本题口径） | 989 TFLOPs | 3.35 TB/s | 295 FLOPs/byte |
| H100 SXM（50% MFU 有效算力） | 494 TFLOPs | 3.35 TB/s | 148 FLOPs/byte |
| A100 80GB | 312 TFLOPs | 2.04 TB/s | 153 FLOPs/byte |

**表 2：两阶段的算术强度对照**

| 阶段 | 每步运算量 | 每步字节（8k 上下文） | 算术强度 | 落点 |
| --- | --- | --- | --- | --- |
| prefill $S=1024$ | 145 TFLOPs | 140 GB（权重）+ 26.5 GB（激活） | 约 870 | 计算受限，高出拐点约 3 倍 |
| prefill $S=8192$ | 1.23 PFLOPs | 140 GB + 212 GB | 约 3500 | 计算受限，高出拐点 12 倍 |
| decode $B=1$ | 0.14 TFLOPs | 143 GB | 1.0 | 带宽受限，只有拐点的 1/295 |
| decode $B=64$ | 9.0 TFLOPs | 312 GB | 28.7 | 带宽受限 |
| decode $B=1024$ | 143 TFLOPs | 2889 GB | 49.6 | 带宽受限（渐近上限 52） |

下表把 batch 的收益与代价放在一起，是容量规划和调参时最常引用的那张表：

| batch $B$ | 读字节 | 强度 | 每步 ms | 聚合 token/s | TPOT |
| --- | --- | --- | --- | --- | --- |
| 1 | 143 GB | 1.0 | 5.3 | 188 | 5.3 ms |
| 8 | 161 GB | 6.9 | 6.0 | 1328 | 6.0 ms |
| 32 | 226 GB | 19.8 | 8.4 | 3796 | 8.4 ms |
| 64 | 312 GB | 28.7 | 11.6 | 5501 | 11.6 ms |
| 128 | 484 GB | 37.1 | 18.0 | 7094 | 18.0 ms |
| 256 | 827 GB | 43.3 | 30.9 | 8294 | 30.9 ms |
| 512 | 1514 GB | 47.3 | 56.5 | 9061 | 56.5 ms |
| 1024 | 2889 GB | 49.6 | 107.8 | 9500 | 107.8 ms |

**表 3：attention 二次项占总 FLOPs 的比例（因果掩码口径，LLaMA-3-70B）**

| $S$ | 1k | 2k | 8k | 32k | 128k |
| --- | --- | --- | --- | --- | --- |
| 二次项占比 | 0.95% | 1.9% | 7.1% | 23.5% | 55.1% |

下面的片段把上面的关键数字一次算完，输出与表格一致：

```python
# roofline 速算：prefill / decode 的瓶颈与上界（H100 SXM 口径）
PEAK, BW, TP = 989e12, 3.35e12, 8      # bf16 稠密算力、HBM3 带宽、tensor parallel 度
P, BYTES = 70e9, 2                     # 70B 参数、bf16
L, KVH, HD = 80, 8, 128                # 层数、KV 头数、head_dim
W = P * BYTES                          # 权重字节数
KV1 = 2 * BYTES * L * KVH * HD         # 每 token 的 KV 字节数
print(f"权重 {W / 1e9:.0f} GB，每 token KV {KV1 / 1024:.0f} KiB，拐点 {PEAK / BW:.1f} FLOPs/byte")
# 权重 140 GB，每 token KV 320 KiB，拐点 295.2 FLOPs/byte


def prefill(S):
    flops = 2 * P * S + 2 * S * S * 8192 * L       # 线性项 + 因果掩码下的 attention 项
    return flops, flops / (PEAK * TP) * 1e3        # FLOPs, 8 卡峰值下界（ms）


for S in (1024, 8192, 131072):
    flops, ms = prefill(S)
    print(f"prefill S={S:6d}: {flops / 1e12:8.0f} TFLOPs，峰值下界 {ms:7.0f} ms，"
          f"权重口径强度 ≈ {S} FLOPs/byte")
# prefill S=  1024:      145 TFLOPs，峰值下界      18 ms，权重口径强度 ≈ 1024 FLOPs/byte
# prefill S=  8192:     1235 TFLOPs，峰值下界     156 ms，权重口径强度 ≈ 8192 FLOPs/byte
# prefill S=131072:    40868 TFLOPs，峰值下界    5165 ms，权重口径强度 ≈ 131072 FLOPs/byte


def decode_step(B, ctx=8192):
    bytes_ = W + KV1 * ctx * B
    flops = 2 * P * B
    t = bytes_ / (BW * TP)                         # 每步秒数（忽略 TP 通信与 kernel 开销）
    return bytes_, flops / bytes_, t, B / t


for B in (1, 8, 64, 256, 1024):
    b, inten, t, tps = decode_step(B)
    print(f"decode B={B:5d}: 读 {b / 1e9:6.0f} GB，强度 {inten:5.1f}，"
          f"每步 {t * 1e3:6.1f} ms，聚合 {tps:6.0f} tok/s")
# decode B=    1: 读    143 GB，强度   1.0，每步    5.3 ms，聚合    188 tok/s
# decode B=    8: 读    161 GB，强度   6.9，每步    6.0 ms，聚合   1328 tok/s
# decode B=   64: 读    312 GB，强度  28.7，每步   11.6 ms，聚合   5501 tok/s
# decode B=  256: 读    827 GB，强度  43.3，每步   30.9 ms，聚合   8294 tok/s
# decode B= 1024: 读   2889 GB，强度  49.6，每步  107.8 ms，聚合   9500 tok/s
```

与来源对照：

- 源文给的「prefill 时数学单元忙碌程度可到 90% 上下、decode 只有 20–40%」是工程经验值，和这里的 MFU 口径不是一回事。按 roofline 复算，decode 的算力占比上界只有 $26.8/7912 \approx 0.34\%$（TP=8、bf16、$B=1$）；0.34% 是「有用浮点运算 ÷ 峰值算力」，源文的百分数描述的是 GPU 忙碌程度的粗粒度观感，中间差着访存请求占空、kernel launch 间隙与 SM 占用率。面试时把口径说出来，比背一个百分数有用。
- 源文「prompt 从 100 翻到 200，attention 那部分工作量变成约 4 倍」对应二次项的正确缩放，但 $S=100\to200$ 这个量级上二次项占总量不到 0.1%，不能用它解释 prefill 的整体瓶颈。
- 延伸来源给的两个实测锚点正好是本题结论在真实系统里的形态：PaLM 540B 上用 int8 权重量化、2048 上下文、低 batch 生成时是 29 ms/token（约 34.5 token/s），要在 29 ms 内读完约 540 GB 权重需要约 18.6 TB/s 的聚合带宽，只能靠多芯片分摊；而大 batch 处理输入 token 时达到 76% MFU，说明 prefill 确实能把算力吃到七成以上。

## 常见追问

- **追问**：为什么不靠 L2 cache 或共享内存把 decode 变快？
  - 要点：H100 的 L2 是 50 MB 量级，70B bf16 权重是 140 GB，差三个数量级，而且每个 step 都要完整过一遍，容量上不可能缓存住，只能靠带宽。片上存储能放下的是单个 tile，这正是 FlashAttention 那类 IO-aware kernel 的思路，不是「把模型放进 cache」。
- **追问**：长上下文下 decode 还是带宽受限吗？强度差多少？
  - 要点：还是，而且更严重。KV 读取把算术强度继续往下压：渐近上限是 $2N/(m_{kv}S)$，8k 上下文约 52 FLOPs/byte，32k 约 13，128k 约 3.3，而拐点是 295。所以长上下文服务里「压 KV」的优先级高于「加 batch」。
- **追问**：batch 越大 decode 就越可能变计算受限吗？
  - 要点：短上下文下会，临界 batch 就是拐点约 295；长上下文下不会，因为 KV 项让 $I(B)$ 有上限（8k 时约 52）。这解释了为什么长短上下文的最优策略不同：短上下文靠 continuous batching 摊薄权重，长上下文还得叠 KV 量化、GQA、稀疏注意力。
- **追问**：speculative decoding 为什么能加速 decode？
  - 要点：一次权重读取的字节数不变，但用这次读取并行验证多个候选 token，被接受的部分相当于「免费」的 token；加速比约等于平均接受长度，代价是草稿模型的开销与额外显存。它的收益来源正是 decode 的带宽受限特性——带宽已经付了，算力闲着。
- **追问**：为什么 prefill 和 decode 要用不同的并行策略？
  - 要点：prefill 的计算能被 $S$ 个 token 填满，算力受限，优先把 GEMM 喂大（大 batch、长 chunk），张量并行的 all-reduce 是纯开销；decode 每卡只有 1 个 token，算力闲置、带宽受限，张量并行几乎线性地减少每卡要读的字节，通信换带宽是划算的。P/D 分离把这条差异推到部署形态上。
- **追问**：只允许优化一个阶段，先优化哪个？
  - 要点：看指标违约在哪。长输出、多轮对话里 decode step 数远大于 1，端到端时间九成以上在 decode，单位成本也几乎全在 decode；但 RAG、长文档、长 system prompt 场景 TTFT 由 prefill 支配。工程上的顺序是先量 TTFT 与 TPOT 谁先违约，再用 [[inference-serving-10]] 那种 roofline 手算确认瓶颈在哪一侧。

## 公司变体

- **Moonshot AI**：偏工程落地。公开技术输出集中在长上下文与分离式推理基础设施，常见问法是从「长 prompt 的 TTFT 和吞吐怎么同时满足」「prefill/decode 分池之后 KV 怎么传」出发，要把本题的瓶颈分析落到部署形态与容量规划上。
- **NVIDIA**：偏实现与量化口径。kernel 与框架视角（TensorRT-LLM 的 in-flight batching、chunked prefill、KV cache 管理），常要求现场用 roofline 算 token/s 上界，并说清 dense/sparse、MFU、实测带宽这几个口径的区别。
- **Together AI**：偏服务与成本。问法常从「每块 GPU 多少 token/s、batch size 的收益拐点在哪、量化与投机解码各能省多少」切入，需要把算术强度和集群成本直接连起来。

以上是依据各家公开技术输出的侧重判断，不是对具体面试流程的描述。

## 相关题目

- [[inference-serving-02]]：continuous batching 把 $B$ 条序列的 decode 合成一次权重读取，是本题第 5 小节结论的直接应用。
- [[inference-serving-10]]：同一套 roofline 的手算题（batch=1、一块 H100、70B 模型每秒多少 token），可以拿本题的数字互验。
- [[inference-serving-14]]：chunked prefill 只切分 prefill、不改总 FLOPs，如何改善混合流量下的尾延迟。
- [[llm-internals-02]]：KV cache 的显存公式与规模化影响，是本节 $m_{kv}$ 这个第二字节来源的推导出处。

## 参考资料与归属

1. [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)，Amit Shekhar（Outcome School），2026-07-05（页面标注的发布时间）。提供 prefill/decode 两阶段的定义、KV cache 作为两阶段桥梁的角色、TTFT/TPOT/throughput 与阶段的对应、「数学单元忙碌程度」的工程经验值，以及「哪种优化对应哪个阶段」的分类；原理与推导第 6 小节的阶段—手段映射沿用该文的分类。
2. [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)（延伸），Reiner Pope 等（Google），2022-11-09。数值与代码验证里引用的实测锚点（PaLM 540B 上 int8 权重量化、低 batch 生成 29 ms/token，大 batch 处理输入 token 达 76% MFU，2048 上下文，多查询注意力支持 32× 更长的上下文）来自该文摘要；正文用它们说明「prefill 高 MFU / decode 低 MFU」在真实系统中的形态，其余公式与推导未取自该文。
3. 正文所有算例（权重字节、每 token KV 字节、拐点、两阶段时间下界与算术强度、batch 扫描、attention 占比）都是按 LLaMA-3-70B 与 H100 SXM 的公开规格自行复算的，两篇来源没有给出这些算例；与来源出现分歧时以复算为准，口径已在「数值与代码验证」开头声明。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
