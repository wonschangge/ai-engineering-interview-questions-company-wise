---
type: question
id: meta-10
company: Meta（超级智能实验室、FAIR、Llama）
topic: llm-internals
order: 10
question: 解释 Llama 级别模型中的架构选择：为什么用 grouped-query attention、RoPE 和 SwiGLU，而不是 2017 年的原版 Transformer？
question_en: Explain the architectural choices in a Llama-class model: why grouped-query attention, RoPE and SwiGLU instead of the vanilla 2017 Transformer?
asked_at: []
level: 进阶
tags: [llama, gqa, rope, swiglu, 架构取舍, kv-cache]
sources:
  - title: What is Grouped Query Attention (GQA) and Why Do LLMs Use It?
    url: https://outcomeschool.com/blog/grouped-query-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-22
  - title: Math Behind RoPE (Rotary Position Embedding)
    url: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding
    author: Amit Shekhar (Outcome School)
    published: 2026-04-23
  - title: Feed-Forward Networks in LLMs
    url: https://outcomeschool.com/blog/feed-forward-networks-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-04-13
related: [llm-internals-03, llm-internals-08, llm-internals-15, llm-internals-02, llm-internals-14, llm-internals-07, llm-internals-16]
updated: 2026-09-28
---

## 一句话答案

> 2017 年原版 Transformer 的四项配置——post-LN 归一化、加在 embedding 上的 sinusoidal 绝对位置编码、每个 head 独占一套 K/V 的 MHA、ReLU FFN 且隐藏维取 $4d$——在「拿同样的算力与显存换更多 token 质量与更高并发」这个目标下各自成为瓶颈。Llama 级模型用 pre-LN + RMSNorm、RoPE、GQA、SwiGLU 分别换掉它们，把架构改动拆成了四条互不替代的线。
> 题面点名的三条落在三个坐标上：GQA 改 KV cache 公式里的乘数 $H_{kv}$，决定 decode 每步要重读多少字节与并发上限；RoPE 改位置信息的注入方式，从「加法注入绝对位置」变成「旋转 Q、K，打分只依赖 $n-m$」，并留下 base 这个频率旋钮供长度扩展；SwiGLU 改 FFN 的参数花法，在同一参数预算下把隐藏维从 $4d$ 收到 $8d/3$，换一条被当前输入调制的门控支路。
> 三者正交、可以同时成立，收益分别落在推理带宽、长度扩展与同参数量下的质量上；分布式并行与训练稳定性属于同批第 11 题，不在本篇范围。

## 面试官在考什么

- **这道题的位置与考法。** 它挂在 [题库总导读](../../../README.zh-CN.md) 的 Meta 一节「LLM 内部原理与架构」下，落在 onsite 的 ML 领域/广度轮：面试官要的不是三个名词的定义，而是一次架构决策复盘——拿 2017 年原版 Transformer 当基线，逐个交代 GQA、RoPE、SwiGLU 各改掉基线的哪一处、换来什么、付了什么，并能随手报出 Llama 的真实配置数字。
- 能否把基线摆清楚再谈改动：post-LN 的 LN 在残差加法之后（[[llm-internals-14]]）、sinusoidal 是加在 embedding 上的绝对编码且实测只能外推 $L+20$ 到 $L+50$ 个 token（[[llm-internals-07]]）、MHA 每个 head 独占 K/V（[[llm-internals-03]]）、ReLU FFN 隐藏维取 $4d$（[[llm-internals-15]]）。四项都要点名，缺一项就变成背三个名词。
- 三条改动是否被讲成三个互不替代的坐标，而不是三张并列的名片：GQA 动 $H_{kv}$、RoPE 动位置注入、SwiGLU 动参数花法。面试官会追问「它们彼此正交吗」，答案是可以同时成立、收益落在三本不同的账上。
- GQA 的数字口径能否复算：只共享 K/V 不共享 Q；$H_{kv}=H$ 是 MHA、$H_{kv}=1$ 是 MQA、中间才是 GQA，压缩倍数就是 $H/H_{kv}$；LLaMA-3-70B 每 token 320 KiB、32k 上下文 10 GiB，回退 MHA 是 8 倍；以及 uptraining 这条低成本采用路径。
- RoPE 是否给得出推导而不是背结论：旋转矩阵正交 ⇒ 打分只依赖 $n-m$；外推崩溃要用波长与训练长度的比值解释；PI、NTK-aware、NTK-by-parts、YaRN 要串成有因果的链；YaRN 温度在实现里并进旋转表后，等价放大是 $1/t\approx1.459$ 而不是 $1.208$。
- SwiGLU 的等参数账是否算得清：$3\,d\,d_{ff}=2\,d\cdot4d$ 推出 $d_{ff}=8d/3$，11008 与 22016 是怎么来的，70B 的 28672 为什么是反例；以及它不是一种激活函数，而是 GLU 家族的门控 FFN。
- 能否补出同波现代化里题面没写但常被顺手问到的部分：pre-LN 与 RMSNorm、去掉线性层 bias、去掉 dropout。

常见错误答案：

- 说「GQA 是为了减少参数量」或「GQA 省的是训练显存」。参数量与优化器状态是附带收益；训练显存的大头是激活值与优化器状态，GQA 的主战场是推理期 decode 的 KV cache 与显存带宽。
- 说「RoPE 是相对位置编码，所以天然能外推」，或把 PI 说成「把位置乘以 $s$」。波长大于训练长度的那批平面在训练中实际充当绝对位置编码；PI 是把位置索引压缩成 $m/s$，方向说反就全错。

## 原理与推导

### 1. 基线：2017 年那一版到底长什么样

| 组件 | 原版 Transformer（2017） | Llama 级模型 | 换掉的理由 | 详见 |
| --- | --- | --- | --- | --- |
| 归一化 | post-LN：$x_{l+1}=\mathrm{LN}(x_l+F_l(x_l))$ | pre-LN + RMSNorm | 残差主轴上不再有归一化，梯度有恒等通路，可去 warmup、训更深；RMSNorm 只做 re-scaling，参数从 $2d$ 降到 $d$ | [[llm-internals-14]] |
| 位置信息 | sinusoidal 绝对编码，与 embedding 相加，只在底层注入一次 | RoPE：每层对 Q、K 做逐元素旋转 | 加法把位置与语义混在同一批维度，实测外推只有 $L+20$ 到 $L+50$；旋转让打分只依赖 $n-m$，且零可学参数 | [[llm-internals-07]] |
| 注意力 | MHA：每个 head 一套 K/V | GQA：$H_{kv}$ 组共享一套 K/V | KV cache 是 $2LH_{kv}d_{head}S\cdot\text{bytes}$，decode 每步要把整条序列重读一遍 | [[llm-internals-03]] |
| FFN | ReLU，$d_{ff}=4d$，两个矩阵 | SwiGLU 门控，$d_{ff}\approx8d/3$ | 等参数预算下 perplexity 更好（经验结论） | [[llm-internals-15]] |

原版是 encoder-decoder 结构，Llama 是 decoder-only，差异不止这四处；但面试官要的基线就是这张表，先说清「加了绝对位置、post-LN、每 head 一套 K/V、4d 的 ReLU FFN」，后面三条改动才有落点。

### 2. 三条改动在三个坐标系里，互不替代

| 改动 | 动的对象 | 收益落在哪本账 | 代价 |
| --- | --- | --- | --- |
| GQA | KV cache 公式里的 $H_{kv}$ | decode 每步重读字节数与并发上限 | 组内 head 共享 K/V，K/V 视角的多样性下降，质量是接近 MHA 而非无损 |
| RoPE | 位置信息的注入方式 | 相对位置性质 + 可扩展的频率旋钮 | 原生外推弱，要靠 PI / NTK / YaRN 把低频周期拉长 |
| SwiGLU | FFN 的参数分配（$3\,d\,d_{ff}$ 预算下取 $d_{ff}$） | 同参数量下每 token 的质量 | 三个矩阵的融合 kernel、tensor parallel 切分、MoE 中逐专家重复 |

三条改动落在三个不同的对象上：GQA 不碰位置编码，RoPE 不改变 cache 大小，SwiGLU 不改变 attention 结构，所以它们可以同时成立、收益互不挤占。买的东西也不一样：GQA 买的是 decode 的带宽与并发上限（它本身不提升质量），RoPE 买的是长度可扩展性与相对位置性质，SwiGLU 买的是同一参数预算下的质量。

### 3. GQA：改掉的是 $H_{kv}$ 这一个乘数

把 KV 套数记作 $H_{kv}$，三种结构就是同一个旋钮的三档：$H_{kv}=H$ 是 MHA，$H_{kv}=1$ 是 MQA，介于两者之间才是 GQA，压缩倍数恒为 $H/H_{kv}$（[[llm-internals-03]]）。

**为什么共享的是 K/V 而不是 Q。** decode 只需缓存历史 token 的 K、V，新的 Q 是当前 token 的临时量，不进 cache；共享 Q 一分显存都省不到，反而让组内各 head 的 $Q,K,V$ 全同、输出一致，等于把这组 head 合成 1 个 head。

**Meta 主场的四组数字。** ① LLaMA-3-70B（80 层、64 个 Q head、8 个 KV head、head_dim 128、bf16）每 token 320 KiB，32k 上下文 10 GiB；同一模型回退 MHA（64 个 KV head）是 2.5 MiB/token，32k 要 80 GiB，正好 8 倍，128k 时是 40 GiB 对 320 GiB，后者单张 80 GB 卡放不下。② 8 张 80 GB 卡约 596 GiB，扣掉 70B 权重（bf16 约 130 GiB）后，32k 会话的并发上限从约 46 条掉到约 6 条，差别全在 $H_{kv}$。③ LLaMA 2 只有 34B/70B 用 GQA（70B 是 64 个 Q head 配 8 个 KV head，cache 比 MHA 小 8 倍），7B/13B 仍是 MHA，到 LLaMA 3 才全尺寸上 GQA。④ 参数量只是附带收益：按 7B 的 $d_{model}=4096$、32 层反事实实算，K/V 投影从每层 33.6M 降到 8.4M，32 层累计省约 0.81B，占 6.5B 主干的约 12%。

**为什么省字节等于更快。** decode 一步里 K/V 那部分流量的算术强度约 $H_q/H_{kv}=64/8=8$ flops/byte（bf16 口径，即每个 K/V 元素被组内 8 个 Q head 各用一次；整步还要把权重读取算进去，见 [[inference-serving-01]]），远低于 H100 bf16 的 $989/3.35\approx295$ flops/byte 拐点，瓶颈在显存带宽；每 token 少读 8 倍的 K/V，直接换成并发路数与 TPOT（[[llm-internals-02]]）。

**配置怎么读。** 看 `num_attention_heads` 与 `num_key_value_heads` 两个字段：32 配 8 就是 8 组、每组 4 个 head、cache 小 4 倍；后者默认等于前者，即默认 MHA。`uptraining` 是把已有 MHA 模型的 K/V 权重按组取平均（或取组内第一个 head）作初始化再继续训练，这是 GQA 能被低成本采用的关键工程原因；具体省多少算力取决于配方，不要报来源里没有的百分比。

### 4. RoPE：改掉的是位置信息的注入方式

角频率 $\theta_i=\text{base}^{-2i/d}$（base 取 10000），把 $d$ 维拆成 $d/2$ 个二维平面，每层对 Q 和 K 各乘一个旋转矩阵，V 不参与；没有任何可学参数，cos/sin 表按位置预计算一次即可复用，这是后面所有扩展技巧零开销的前提。配对方式有两种：原论文用相邻对，LLaMA/HF 实现用相差 $d/2$ 的两维，数学等价、只差一个维度置换。

**相对位置性质要推导。** 旋转矩阵正交，$R_m^{\top}=R_{-m}$ 且 $R_{-m}R_n=R_{n-m}$，于是

$$\langle R_m q,\;R_n k\rangle=q^{\top}R_m^{\top}R_nk=\langle q,\;R_{n-m}k\rangle$$

绝对位置在打分里被消掉，只剩 $n-m$。注意 $R_m$ 本身仍只由 $m$ 决定，所以「编码绝对位置、打分只看相对距离」两句话同时成立，这个区分是后面所有长度扩展讨论的支点（完整推导与扩展链见 [[llm-internals-08]]）。

**外推为什么崩，按波长讲。** 第 $i$ 个平面的波长 $\lambda_i=2\pi\,\text{base}^{2i/d}$：$d=128$ 时 $\lambda_0\approx6.28$，最后一个平面 $\lambda_{63}\approx5.44\times10^4$。用 $r(i)=L/\lambda_i$ 判断（$L$ 为训练长度）：$r\gg1$ 的平面转了很多圈，是纯相对位置；$r<1$ 的平面在训练全程连一圈都没转完，$m$ 与相位一一对应，实际充当绝对位置编码——把位置推到训练区间外，这批相位立刻失控。

**扩展链按因果讲**（数字与实测口径引自 [[llm-internals-08]]）：PI 把位置索引压成 $m/s$，等价于所有频率乘 $1/s$，论文观察 $s\approx8$ 是可恢复上限，$s$ 越大越伤高频；NTK-aware 改 base，$\text{base}'=\text{base}\cdot s^{|D|/(|D|-2)}$（$|D|=128$ 时指数约 1.0159，$s=8$ 对应 82685），高频几乎不动、压力摊到低频，但 $s$ 只能靠实验定，Code Llama 手工把 base 调到 1M，反解 $s\approx93$；NTK-by-parts 按 $r(i)$ 分三区，Llama 系推荐 $\alpha=1$、$\beta=32$：$L=4096$、$d=128$ 时 64 个平面里，不插值 21 个（32.8%）、ramp 过渡 25 个（39.1%）、全插值 18 个（28.1%）；YaRN 再加一个 attention 温度，$\sqrt{1/t}=0.1\ln s+1$（$s=8$ 时 $1.208$、$t\approx0.685$），实现上并进旋转表，因此 Q、K 各被放大一次，等价于 logits 放大 $1/t\approx1.459$，而不是 $1.208$——这个口径答错会被抓。实测口径：论文在 Llama 2 7B 上 $s=16$ 时 128k 的 sliding window perplexity 已经大于 10，换 $s=32$ 才降到 2.37；passkey 检索 128k 上 99.4%（$s=32$）、64k 上 96.3%（$s=16$），属于 train short, test long。

**两个实现坑。** 动态缩放下必须缓存未施加 RoPE 的 K，否则历史 token 的旋转表与当前 $s$ 不一致；以及长上下文不等于有效上下文，关键信息放在中间仍可能被忽略（[[llm-internals-13]]）。

### 5. SwiGLU：改掉的是 FFN 的参数花法

$$\mathrm{SwiGLU}(x)=\bigl(\mathrm{Swish}(xW_{gate})\odot xW_{up}\bigr)W_{down}$$

Swish 在 $\beta=1$ 时就是 SiLU。先纠正两个高频错误：它不是一种激活函数，而是带门控的 FFN 结构，与 ReGLU、GEGLU 同属 GLU 家族，门控给的是输入相关的乘性选择能力；这是经验结论（Shazeer 在 T5 上做等参数量、等计算量的 perplexity 对比，论文自述结果有噪声、也不解释原因），不能说成「Swish 比 ReLU 好所以换了」。

**参数账是这题最能拉分的地方。** 三个矩阵让参数量变成 $3\,d\,d_{ff}$，要在同一预算下公平比较就得解 $3\,d\,d_{ff}=2\,d\cdot4d$，得 $d_{ff}=\tfrac{2}{3}\cdot4d=8d/3\approx2.67d$。LLaMA 再向上取到 256 的倍数：$d=4096$ 时 $8d/3=10922.67$，取 $43\times256=11008$（正是 LLaMA 2 7B 的 `intermediate_size`，$=2.69d$）；$d=8192$ 时取 $86\times256=22016$（LLaMA 1 65B）。反例要主动交代：LLaMA 2 70B 取 28672（$=3.5d$），并没有严格守住 $2.67d$。

**三件套在这里串起来。** GQA 把 attention 参数压小后，FFN 在 block 里的占比继续上升：$d=4096$、$H_{kv}=8$ 时 attention 是 $2\times4096\times4096+2\times4096\times1024=41.94\text{M}$，FFN 是 135.27M，占比从基线的约三分之二升到约 76.3%。

**工程侧说到点子上。** gate 与 up 能拼成一个矩阵做一次 GEMM，tensor parallel 下按输出维切这个拼接矩阵、$W_{down}$ 是行并行、输出需要一次 all-reduce；不做融合 kernel 时损失在访存（$2d_{ff}$ 维中间激活要写回 HBM 再读回来）而不是 FLOPs，而 decode 本就带宽受限，所以 `silu_and_mul` 这类融合算子是默认写法。MoE 里每个专家同样用 SwiGLU，参数账与特化方向可以接着问（[[llm-internals-10]]）。

### 6. 合起来看：同一个 block 上的现代配置

| 组件 | 2017 原版 | Llama 级 | 关键数字 |
| --- | --- | --- | --- |
| 归一化 | post-LN | pre-LN + RMSNorm | 参数 $2d\to d$ |
| 位置编码 | 加法 sinusoidal（绝对） | RoPE（旋转，打分只看 $n-m$） | 可学参数 0；base 是旋钮 |
| 注意力 | MHA，$H_{kv}=H$ | GQA，$H_{kv}=8$ | 70B 每 token 320 KiB（MHA 的 $1/8$） |
| FFN | ReLU，$d_{ff}=4d$ | SwiGLU，$d_{ff}\approx8d/3$ | 11008 / 22016；等参数误差 +0.78% |

不算在题面里、但属于同一波现代化的还有：pre-LN 加 RMSNorm（[[llm-internals-14]]）、去掉线性层 bias、去掉 dropout。逐张量前向与参数量口径见 [[llm-internals-16]]；把训练从 8 卡扩到数千卡的并行与稳定性问题见同批第 11 题（`meta-11`）。

## 数值与代码验证

单位统一 1024 进制。LLaMA-3-70B 取 $L=80$、$H_q=64$、$H_{kv}=8$、$d_{head}=128$、bf16（2 B/元素），与 [[llm-internals-02]] 的口径一致。

**表 1：每 token / 每序列的 K/V 占用（$M=2LH_{kv}d_{head}S\cdot\text{bytes}$）**

| 模型 | 每 token | 8k | 32k | 128k |
| --- | --- | --- | --- | --- |
| LLaMA-3-70B（GQA-8） | **320 KiB** | 2.5 GiB | **10 GiB** | **40 GiB** |
| LLaMA-3-70B（回退 MHA-64） | **2.5 MiB** | 20 GiB | **80 GiB** | **320 GiB** |
| LLaMA-3-8B（32 层、8 个 KV head） | **128 KiB** | 1 GiB | 4 GiB | 16 GiB |

回退 MHA 就是把 $H_{kv}$ 从 8 换成 64，比值恰好 $64/8=8$；128k 的 320 GiB 已经超出单张 80 GB 卡的容量。

**表 2：8×H100（8 × 80 GB ≈ 596 GiB）上 32k 会话的并发上限**

| 项 | GQA-8 | MHA-64 |
| --- | --- | --- |
| 权重（70B、bf16） | 约 130 GiB | 约 130 GiB |
| 每条 32k 会话的 cache | 10 GiB | 80 GiB |
| 可留给 cache | 约 466 GiB | 约 466 GiB |
| 并发上限（未取整 46.6 / 5.8，未计激活与碎片） | 约 46 | 约 6 |

**表 3：decode 的带宽下界估算（H100 3.35 TB/s，$S=32\text{k}$，权重读一遍）**

| 场景 | 每步至少读 | 步耗时下界 | 吞吐下界 |
| --- | --- | --- | --- |
| $b=1$、GQA-8 | 130 GiB 权重 + 10 GiB K/V | 45.0 ms | 22.2 tok/s |
| $b=1$、MHA-64 | 130 GiB 权重 + 80 GiB K/V | 67.4 ms | 14.8 tok/s |
| $b=8$、GQA-8 | 130 GiB + 8 × 10 GiB | 67.4 ms | 118.6 tok/s |
| $b=8$、MHA-64 | 130 GiB + 8 × 80 GiB | 246.9 ms | 32.4 tok/s |

这是理想峰值带宽下的读字节数除法，不含计算时间与调度开销，但足以说明趋势：$b=1$ 时 130 GiB 权重仍是主项，GQA 只把每步从 67.4 ms 压到 45.0 ms（1.5×）；$b=8$ 时 8 条序列的 K/V 累计成主项，差距拉到 246.9 对 67.4 ms（约 3.7×）。这正是「GQA 在长上下文、高并发场景才成为卖点」的数字依据。

**表 4：RoPE 的频率分区与扩展参数**（$d=128$、base 10000、$L=4096$，$\alpha=1$、$\beta=32$）

| 量 | 公式 | 数值 |
| --- | --- | --- |
| 最快平面波长 | $\lambda_0=2\pi$ | 6.28 |
| 最慢平面波长 | $\lambda_{63}=2\pi\cdot10000^{126/128}$ | $5.44\times10^4$ |
| 三区平面数（不插值 / ramp / 全插值） | $r>\beta$；$\alpha<r<\beta$；$r<\alpha$ | 21 / 25 / 18 |
| NTK-aware base（$s=8$） | $10000\cdot8^{128/126}$ | 82685 |
| Code Llama base 1M 反解 $s$ | $(10^6/10^4)^{126/128}$ | 93.1 |
| YaRN 温度（$s=8$） | $\sqrt{1/t}=0.1\ln8+1$ | 1.208，$t=0.685$ |
| 并进旋转表后的实际放大 | $1/t$ | 1.459 |

**表 5：SwiGLU 的等参数预算（$d=4096$）**

| 方案 | $d_{ff}$ | 参数量 | 相对 4d ReLU 基线 |
| --- | --- | --- | --- |
| 精确 $8d/3$ | 10922.67 | 134.22 M | $1.000\times$ |
| LLaMA 对齐值 | 11008 | 135,266,304 | $1.008\times$（+0.78%） |
| 基线 ReLU | 16384 | $2\,d\cdot4d=134,217,728$ | $1.000\times$ |

```python
L, Hq, Hkv, dh, B = 80, 64, 8, 128, 2          # LLaMA-3-70B, bf16
KiB, GiB = 1024, 1024 ** 3
per_tok_gqa = 2 * L * Hkv * dh * B             # 327680 B
per_tok_mha = 2 * L * Hq * dh * B              # 2621440 B
assert per_tok_gqa == 320 * KiB and per_tok_mha == 2.5 * 1024 * 1024
assert per_tok_mha // per_tok_gqa == 8
print('32k: GQA', per_tok_gqa * 32768 / GiB, 'GiB; MHA', per_tok_mha * 32768 / GiB, 'GiB')
# 32k: GQA 10.0 GiB; MHA 80.0 GiB

cards, weights = 8 * 80e9 / GiB, 70e9 * 2 / GiB      # 596.05, 130.39 GiB
free = cards - weights                                # 465.66 GiB
print('并发', free / (per_tok_gqa * 32768 / GiB), free / (per_tok_mha * 32768 / GiB))
# 46.57  5.82  -> 约 46 条 与 约 6 条

d, layers = 4096, 32
saved = (2 * d * d - 2 * d * 8 * dh) * layers         # 805306368
print('省参数', saved / 1e9, 'B，占 6.5B 主干', 100 * saved / 6.5e9, '%')
# 0.805 B，12.4%

swiglu, base = 3 * d * 11008, 2 * d * 16384
print(swiglu, base, swiglu / base)                    # 135266304 134217728 1.0078125
attn_gqa = 2 * d * d + 2 * d * 1024                   # W_Q,W_O + W_K,W_V
print('attention', attn_gqa / 1e6, 'M, FFN 占比', 100 * swiglu / (swiglu + attn_gqa), '%')
# 41.94 M, 76.3%

import math
lam = lambda i: 2 * math.pi * 10000 ** (2 * i / 128)
zones = {'不插值': 0, 'ramp': 0, '全插值': 0}
for i in range(64):
    r = 4096 / lam(i)
    zones['不插值' if r > 32 else '全插值' if r < 1 else 'ramp'] += 1
print(zones)                                          # 21 / 25 / 18
print(10000 * 8 ** (128 / 126), (1e6 / 1e4) ** (126 / 128))
# 82684.6  -> 82685 ；93.06  -> 93.1
sq = 0.1 * math.log(8) + 1
print(sq, 1 / sq ** 2, sq ** 2)                       # 1.208  0.685  1.459
```

**表 6：必须覆盖的错误答案**

| 错误说法 | 正确口径 |
| --- | --- |
| GQA 是为了减少参数量 | 参数量只是附带（省约 0.81B、约 12%）；主战场是 decode 的 KV cache 与显存带宽 |
| GQA 省的是训练显存 | 训练大头是激活值与优化器状态；GQA 主要省推理 |
| GQA 连 Q 也共享 | Q 从不共享；共享 Q 既不省显存，又让组内 head 退化成 1 个 head |
| RoPE 是相对位置编码所以天然能外推 | 波长大于训练长度的平面在训练中充当绝对位置编码，外推即越界 |
| PI 是把位置乘以 $s$ | 方向说反：PI 是把位置索引压缩成 $m/s$，等价于频率乘 $1/s$ |
| SwiGLU 是一种激活函数，或多一个矩阵所以更贵 | 它是门控 FFN 结构；等参数预算下总参数与每 token MACs 与基线基本持平 |
| 只答「省显存、更快、效果更好」 | 要落到 $H_{kv}$、波长分区、$d_{ff}$ 这三个具体乘数上 |

数字纪律：全文只用本仓库既有口径（70B 每 token 320 KiB、32k 为 10 GiB、MHA 回退 8 倍、8B 为 128 KiB、11008 / 22016 / 28672、$8d/3$、YaRN 温度公式），引用不到出处的倍数与百分比一律不写；凡涉及并行的结论按所选推理栈的实现现场核对（KV 头数与 tensor parallel 度数的整除关系、分页调度与 GQA 的对齐方式），不要凭印象报具体阈值。

## 常见追问

- **追问**：LLaMA 2 各尺寸到底用不用 GQA？70B 的 64 比 8 怎么读？
  - 要点：LLaMA 2 只有 34B 与 70B 用 GQA，7B 与 13B 仍是 MHA；到 LLaMA 3 才全尺寸上 GQA。70B 的 64 是 `num_attention_heads`、8 是 `num_key_value_heads`，即 8 组、每组 8 个 Q head 共享一套 K/V，cache 比 MHA 小 $64/8=8$ 倍。
- **追问**：uptraining 具体怎么做，为什么它重要？
  - 要点：把已训练 MHA 模型的 K/V 权重按组重塑成 $[G,H/G,d_{head},d]$，组内取平均（或取第一个 head）作初始化，再继续训练收敛；论文里 mean 略优于 pick single。它把 GQA 从「重训一遍」变成一次短程继续训练，这是 GQA 被快速采用的关键。不要报来源里没有的算力百分比。
- **追问**：为什么 7B 用 11008、65B 用 22016，而 70B 是 28672？
  - 要点：11008 与 22016 来自 $8d/3$ 再向上取 256 的倍数（$4096\to43\times256$、$8192\to86\times256$），目的是在等参数预算下保住质量；70B 的 28672 是 $3.5d$，说明这条比例在大尺寸上并没有被严格执行。
- **追问**：RoPE 的 base 和长上下文是什么关系？
  - 要点：base 是频率旋钮，$\theta_0=1$ 与 base 无关，拉大 base 只拉长低频端的波长。LLaMA 2 用 10000、LLaMA 3 用 500000；更大的 base 提高远距离相位分辨率，代价是低频的局部区分变弱，且与旧 base 学到的频率分布不匹配，通常要配合继续预训练或外推微调——PI、NTK-aware、YaRN 动的都是这组频率（[[llm-internals-08]]）。
- **追问**：长上下文就够了吗？上下文写满 128k 就一定能用？
  - 要点：不够。位置编码不越界只是前提，注意力是否真落在远处关键 token 上要用 passkey、RULER 这类探针和 lost-in-the-middle 分别看（[[llm-internals-13]]）；YaRN 论文也提醒 perplexity 不是有效上下文的充分指标。
- **追问**：MoE 变体里专家的 FFN 是否照抄 dense 配置？
  - 要点：每个专家同样是 SwiGLU 三矩阵结构，但专家更窄、数量更多，$d_{ff}$ 不再按 dense 的 $8d/3$ 对齐，要按总激活参数与路由负载一起定；参数账与特化方向见 [[llm-internals-10]]。

## 相关题目

- [[llm-internals-03]]：MHA / MQA / GQA 的三联图、为什么只共享 K/V、uptraining 的组内权重平均，是本题 GQA 部分的前置；本篇只补它与 RoPE、SwiGLU 的合力。
- [[llm-internals-02]]：KV cache 公式的逐项推导与 320 KiB / 10 GiB / 40 GiB 口径，以及并发上限 46 与 6 的算法都来自这篇。
- [[llm-internals-08]]：RoPE 的构造、相对位置推导、波长分区、PI / NTK / YaRN 的完整链与实测表，本篇只做面试口径的压缩复述。
- [[llm-internals-15]]：GLU 家族、$8d/3$ 参数账与 FFN 占比；本篇沿用它 $d=4096$、$H_{kv}=8$ 的同一组配置复算 41.94M 与 76.3%，把三件套串成一条线。
- [[llm-internals-14]]、[[llm-internals-07]]：pre-LN / RMSNorm 与位置编码演进，用来把 2017 基线补全。
- [[llm-internals-16]]：逐张量前向与参数量口径，适合对照本题的三条改动在 forward 里的实际位置。
- 专题导读：[LLM 内部原理与架构](../../LLM 内部原理与架构/README.md)；同批第 11 题（`meta-11`，从 8 卡扩到数千卡）与本篇互补——那篇讲系统侧的并行与稳定性，本篇只讲单模型内部的架构取舍。

## 参考资料与归属

1. [What is Grouped Query Attention (GQA) and Why Do LLMs Use It?](https://outcomeschool.com/blog/grouped-query-attention)，Amit Shekhar（Outcome School），2026-04-22。提供 MHA / MQA / GQA 是同一轴上三个点的定性、只共享 K/V 的理由、uptraining 的组内平均做法，以及 GQA 换来并发与带宽收益的结论。
2. [Math Behind RoPE (Rotary Position Embedding)](https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding)，Amit Shekhar（Outcome School），2026-04-23。提供 RoPE 的二维旋转构造、只作用于 Q/K、相对位置性质的直觉解释与「长上下文仍需外推技巧」的定位。
3. [Feed-Forward Networks in LLMs](https://outcomeschool.com/blog/feed-forward-networks-in-llms)，Amit Shekhar（Outcome School），2026-04-13。提供 expand-then-contract 的 FFN 结构、SwiGLU 三矩阵写法、LLaMA 的 $2.67d$ 经验比例与 FFN 占比随 GQA 上升的观察。
4. 延伸来源：本篇的 PI / NTK-aware / NTK-by-parts / YaRN 公式与实测数字（base $=82685$、$s\approx93$、三区 21 / 25 / 18、$\sqrt{1/t}=1.208$、128k perplexity 与 passkey 口径）承自 [[llm-internals-08]] 所引的 RoFormer 与 YaRN 论文；LLaMA 的 11008 / 22016 / 28672、320 KiB 与 10 GiB 等配置数字承自 [[llm-internals-15]]、[[llm-internals-02]]、[[llm-internals-03]] 的既有复算。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
