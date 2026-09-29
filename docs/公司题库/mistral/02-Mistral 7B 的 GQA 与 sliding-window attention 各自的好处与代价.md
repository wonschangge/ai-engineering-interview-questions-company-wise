---
type: question
id: mistral-02
company: Mistral AI
topic: llm-internals
order: 2
question: Mistral 7B 发布时采用了 grouped-query attention 和 sliding-window attention。两者各自带来什么好处，又各自付出什么代价？
question_en: Mistral 7B shipped with grouped-query attention and sliding-window attention. What does each buy you, and what does each cost?
asked_at: []
level: 进阶
tags: [gqa, sliding-window-attention, kv-cache, 长上下文, 数值推导]
sources:
  - title: 分组查询注意力（GQA）
    url: https://outcomeschool.com/blog/grouped-query-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-22
  - title: Sliding Window Attention 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-sliding-window-attention-work
    author: Amit Shekhar (Outcome School)
    published: 2026-09-05
  - title: KV Cache in LLMs
    url: https://outcomeschool.com/blog/kv-cache-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-03-27
  - title: KV Cache Compression
    url: https://outcomeschool.com/blog/kv-cache-compression
    author: Amit Shekhar (Outcome School)
    published: 2026-09-12
  - title: Mistral 7B 论文（延伸）
    url: https://arxiv.org/abs/2310.06825
    author: Albert Q. Jiang 等（Mistral AI）
    published: 2023-10-10
  - title: Mistral-7B-v0.1 config.json（延伸）
    url: https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json
    author: Mistral AI（Hugging Face）
    published: 2023-09-27
  - title: Mistral-7B-Instruct-v0.2 Model Card（延伸）
    url: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    author: Mistral AI（Hugging Face）
    published: 2023-12-11
related: [llm-internals-03, llm-internals-02, llm-internals-13, inference-serving-03]
updated: 2026-09-29
---

## 一句话答案

> GQA 动的是 KV cache 账本里的 $H_{kv}$：Mistral 7B 用 32 个 Q head 配 8 个 KV 组，每个 token 的 K/V 从 MHA 的 512 KiB 降到 128 KiB（4×）。decode 是带宽受限的，省下来的读取量直接换成并发路数与 TPOT；代价是组内 4 个 head 共用一份 K/V，牺牲 K/V 视角的多样性，质量只是接近 MHA 而不是无损，而且它是预训练前的结构决策，推理期没有开关。
> SWA 动的是同一个公式里的位置数：每层只回看最近 $W=4096$ 个位置，滚动缓冲区把 cache 与上下文长度解耦，固定在 512 MiB/序列，同一形状的全注意力在 32k 下要 4 GiB。代价是窗口外的信息只能靠层叠间接传播，$L \times W \approx 131$k 是理论上界而不是有效召回，需要精确回看远处 token 的任务会掉分。
> 两者作用在公式里不同的乘数上，收益是乘法叠加的（$4 \times 8 = 32\times$），所以只上其中一个都不够：只上 GQA，cache 仍随序列长度线性增长；只上 SWA，每个位置仍然贵 4 倍。

## 面试官在考什么

- 能不能把两个机制落到同一本 KV cache 账上：$M = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b \cdot \text{bytes}$（[[llm-internals-02]]）。GQA 改的是 $H_{kv}$（每个位置存多少字节），SWA 改的是 $S$（把位置数封顶到 $W$），两个乘数互不相干，因此可以相乘。
- GQA 的数字是否真的算过：Mistral 7B 是 32 个 Q head、8 个 KV 组、$d_{head}=128$，所以每 token 128 KiB、相对 MHA 的 512 KiB 是 4×，不是 8×（8× 是 LLaMA-3-70B 那种 64 个 Q head 的模型）。
- 是否理解为什么「cache 更小」等于「生成更快」：decode 每步要把整条序列的 K/V 读一遍，算术强度只有 $H_q/H_{kv}$ flops/byte，远低于 H100 约 295 flops/byte 的拐点，所以瓶颈在带宽而不在算力。
- SWA 的收益要能拆开讲：缓存容量的 8×（32k/4096，精确）与注意力计算量的下降（口径不同结论不同）是两笔账，不能混着报。
- 是否知道两个机制在服务栈里的落地代价：窗口掩码 kernel、chunked prefill 的跨 chunk 可见性、prefix caching 的 block 复用裁决、GQA 的 KV 组数对 tensor parallel 分片的约束，以及「serving 配置必须与训练配置一致」。

常见错误答案：

- 说成「GQA 省算力、SWA 省显存」。GQA 不减少 FLOPs，省的显存与带宽都来自每位置字节数；SWA 也不只省显存，它封住位置数的同时把 attention pair 数从 $O(S^2)$ 降到 $O(S \cdot W)$。两者作用在同一个公式的不同乘数上，不是「一个省算力、一个省显存」的分工。
- 把 SWA 的 8× 到处套用。8× 是滚动缓冲区在全注意力 32k 口径下的缓存比；注意力计算量按 causal 口径在 32k 下只降约 4.3×，端到端 prefill FLOPs 只降约 1.4×，论文实测的 2× 加速是 16k 序列上 attention kernel 的口径。

## 原理与推导

### 1. 两个机制作用在同一个公式的不同乘数上

KV cache 的单序列占用是（推导见 [[llm-internals-02]]）：

$$M = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b \cdot \text{bytes}$$

| 机制 | 动的乘数 | 一句话 |
| --- | --- | --- |
| GQA | $H_{kv}$：32 个 Q head 只保留 8 套 K/V | 降低「每个位置存多少字节」 |
| SWA | $S$：$\min(S, W)$，$W=4096$ | 把「存多少个位置」封顶 |

代入 Mistral-7B-v0.1 的真实 config（$L=32$、$H_q=32$、$H_{kv}=8$、$d_{head}=4096/32=128$、`sliding_window=4096`、bf16）：

- 每 token：$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = **128 KiB**（MHA 是 512 KiB）
- 每序列：$128\ \text{KiB} \times 4096 = $ **512 MiB**，与上下文长度无关

两者同时开，账本变成 $512\ \text{MiB} \times b$；如果都不开（MHA + 全注意力）在 32k 下是 $16\ \text{GiB} \times b$，差 32 倍。

### 2. 逐格填满：MHA / GQA / MQA 与 full attention / SWA

| 方案 | 作用在公式哪一项 | 收益 | 代价 |
| --- | --- | --- | --- |
| MHA | 基线：$H_{kv}=H_q=32$ | K/V 视角最全，质量最好，没有额外结构约束 | 每 token 512 KiB；32k 单序列 16 GiB |
| GQA（Mistral 7B） | $H_{kv}$：32 → 8，每位置字节数 ÷4 | 每 token 128 KiB；decode 每步重读量与并发上限同比例改善；质量接近 MHA；可用 uptraining 事后转换 | 组内 4 个 head 共用一份 K/V，视角多样性下降；组数必须在预训练前定死 |
| MQA | $H_{kv} \to 1$，再 ÷8 | cache 与带宽比 GQA 再小 8× | 质量退化明显、训练可能不稳，同规模模型多选 GQA 而不是它 |
| full attention | $S$ 取满上下文长度 | 每个位置直接可见全部历史，精确长程召回最好；mask 与 kernel 最简单 | prefill 注意力 pair 数 $O(S^2)$；cache 随 $S$ 线性增长 |
| SWA（$W=4096$） | $\min(S, W)$ | pair 数降到 $O(S \cdot W)$；cache 与 $S$ 解耦；训练算力与 activation 同向下降 | 窗口外只能靠层叠间接传播，有效召回受限；需要窗口掩码 kernel 与调度配合 |

### 3. GQA 买到什么

**每位置字节数 ÷4。** 组数 $G = H_{kv} = 8$：32 个 Q head 分成 8 组，每组 4 个 head 共用一套 K/V，共享倍数 $H_q / H_{kv} = 32/8 = 4$，K/V 的存储与读取量都变成 $1/4$。

**decode 每步重读量 ÷4。** decode 每生成一个 token，就要把本条序列 32 层的 K/V 全部读一遍（[[llm-internals-02]]）。$S=4096$ 时 GQA 是 $128\ \text{KiB} \times 4096 = 512$ MiB/条，MHA 是 2 GiB/条；$S=32768$ 时分别是 4 GiB 与 16 GiB。

**算术强度从 1 提到 4。** 一个 decode step、一层、一条序列：读 $4 H_{kv} d_{head} S$ 字节，做 $4 H_q d_{head} S$ flops，相除得 $H_q/H_{kv}$ flops/byte。MHA 是 $32/32 = 1$，Mistral 7B 的 GQA 是 $32/8 = 4$，仍然远低于 H100 bf16 的 $989/3.35 \approx 295$ flops/byte 拐点——所以省下来的带宽直接变成吞吐与 TPOT，而不是算力。（同一公式下 LLaMA-3-70B 是 $64/8 = 8$，因为它的 Q head 更多，不是矛盾。）

**容量 ÷4。** 权重只有一份、cache 每序列一份，所以并发上限基本由 cache 决定。80 GB 卡扣掉 13.49 GiB 权重后约 61.0 GiB 可用：GQA + SWA 能开约 122 路，换成 MHA + SWA 只有约 30 路。

**顺带的好处。** K/V 投影参数从每层 33.6M 降到 8.4M，32 层共省约 0.81B 参数（占 7.24B 的 11.1%）。这不是主因，但会一起减少优化器状态与权重读取量。

### 4. GQA 付出什么

**牺牲 K/V 视角的多样性。** head 的差异化主要体现在 Q 上（「我现在要找什么」），K/V 回答的是「输入里有什么」。共享 K/V 相当于让 4 个 head 查同一个资料柜，各自的问题仍不同，注意力模式只是被限制而不是被抹平，所以质量是「接近 MHA」。这就是为什么不能连 Q 一起共享：组内 $Q$、$K$、$V$ 全同时，4 个 head 的输出完全一致，等于退化成 1 个 head，而 KV cache 里本来就不存 Q，共享 Q 一分显存都省不到。

**MQA 是这条轴的端点，不是等价选项。** $H_{kv}=1$ 时 cache 再小 8×，但质量退化明显、训练还可能不稳定，Mistral 7B 因此选了 8 组而不是 1 组。

**它改不动。** 组数是预训练结构决策，推理期没有开关。手里已经有一个训练好的 MHA checkpoint，只有两条路：换权重，或者 uptraining——把组内各 head 的 K/V 权重取平均做初始化，再用很短的训练恢复质量（公开口径约 5% 的预训练算力）。这也是 GQA 当年被快速采用的原因，但它仍然是一次训练动作，不是部署开关。

**它还约束并行度。** tensor parallel 按 KV head 切分，$H_{kv}=8$ 意味着 TP 度数超过 8 就要复制 KV head（或改用其它并行方式），否则分不匀。

### 5. SWA 买到什么

**每层只看最近 $W$ 个位置。** 第 $k$ 层位置 $i$ 的隐状态只 attend 上一层 $[i-W, i]$ 区间；配合 causal 掩码，每个位置最多对 $W$ 个 key 打分。

**计算量降到 $O(S \cdot W)$。** 32k 序列上，causal full attention 的 pair 数约 537M，$W=4096$ 的 SWA 约 126M，比值 4.27×（渐近值 $S/(2W)$）。如果按「不带 causal 三角形」的 $O(S^2) \to O(S \cdot W)$ 口径，这个比值就是 $S/W = 8\times$——两个口径都对，但必须说清是哪一个，端到端 prefill FLOPs 只从 756 TFLOPs 降到 541 TFLOPs（1.40×），因为投影与 MLP 的 $2PS$ 才是大头。论文给出的实测口径是：16k 序列、$W=4096$，改过的 FlashAttention / xFormers 相对 vanilla attention 有 2× 加速，与 16k 下 attention FLOPs 比 2.29× 基本吻合。

**缓存容量固定。** 滚动缓冲区把位置 $i$ 的 K/V 写进槽位 $i \bmod W$，$i \ge W$ 时覆盖最旧的一项，于是 cache 大小变成 $W$ 而不是 $S$：512 MiB/序列。论文的结论是 32k 序列下 cache 减少 8×，且不影响质量。这一笔是精确的 8×（$32768/4096$），也是 SWA 最值钱的收益——显存不再是上下文长度的函数，容量规划从「随长度涨」变成「常数」。

**训练侧同向受益。** 注意力 pair 数下降意味着 score 矩阵规模与算力同比例下降；用 FlashAttention 时不 materialize score 矩阵，省的是算力，用朴素实现时连 activation 一起省。

### 6. SWA 付出什么

**窗口外只能间接传播。** 信息每过一层最多向前传 $W$ 个位置，堆 $L$ 层后理论上能跨 $L \times W = 32 \times 4096 = 131{,}072$ 个 token，这就是论文说的约 131K 理论 attention span。注意它是**上界**：它只说明存在一条传递路径，不说明模型真的把信息保住了。每一跳都是一次加权平均与非线性，远距离信息被反复混合、稀释，实际召回率远低于这个上界。可验证的做法是 needle-in-a-haystack 与 [[llm-internals-13]] 的 lost-in-the-middle 口径：把关键事实放在窗口外，观察准确率随距离衰减的曲线，而不是拿 131k 当结论。

**任务适配要判断，不能只说「省显存更快」。** 适合：聊天、多轮对话、摘要、代码补全、局部模式识别——这些任务的有效依赖大多落在窗口内。不适合：整篇文档的精确定位与引用（「第 40 页的电话号码」）、长链依赖的数学推导、必须逐字回看早期的结构化抽取。这类任务要么把关键内容放进窗口（重排 prompt、检索增强），要么换全注意力模型或更长的窗口。

**窗口是训练期语义。** 对已经用 $W=4096$ 训练好的权重，推理期把窗口调大或调小都不是无损操作：调小等于让模型看到与训练分布不同的注意力模式，调大则会让它在训练时从未直接看过的距离上做直接注意力。窗口大小要跟训练配置一致。

### 7. 生产坑一：配置必须与训练一致，也别把 SWA 当成推理期的 KV 裁剪

**不同版本的 Mistral 7B 对 SWA 的取舍不同，一律以模型 config 为准。** 已核对过的字段：`Mistral-7B-v0.1` 的 config 里 `sliding_window=4096`、`rope_theta=10000`；而 `Mistral-7B-Instruct-v0.2` 与 `Mistral-7B-v0.3` 的 config 里 `sliding_window=null`、`rope_theta=1000000`，model card 明确写了 v0.2 相对 v0.1 的三点变化是 32k 上下文、Rope-theta = 1e6、No Sliding-Window Attention。也就是说同一个名字下的模型，缓存行为可能从「固定 512 MiB」变成「随 $S$ 线性增长」。serving 栈如果按印象配 `--sliding-window`，要么让本该固定大小的 cache 继续增长，要么给全注意力模型硬套一个窗口，两种都是静默掉质量。

**把窗口当作对全注意力模型的推理期裁剪切不可行。** 那是另一个语义：全注意力模型在训练时学会了把多余的注意力丢给最前面几个 token（attention sink），一旦这些 token 滑出窗口，输出会直接崩掉，所以 StreamingLLM 这类方案必须显式保留开头若干个 token（参见 [[llm-internals-02]] 的滑窗 + attention sinks 结论）。而 SWA 是把窗口写进训练分布的模型结构，每层都只看窗口内，没有「窗口外曾经看过、现在不看了」的落差。两件事不要互相套用。

### 8. 生产坑二：kernel、调度、并行度，以及 on-prem 换算

**kernel 要支持窗口掩码。** 需要 block-sparse / xFormers 或带 local mask 的 FlashAttention；纯 causal kernel 套上去会算错或退化。

**chunked prefill 要显式处理可见性。** 论文的做法是把 chunk size 取成窗口大小，于是每个 chunk 的掩码由三块组成：chunk 内纯 causal（距离必然不超过 $W$）、对 cache 的滑动窗口、以及窗口外完全不看。如果实现里只留了 chunk 内 causal，跨 chunk 的上下文就丢了；如果掩码退化成全 causal，模型又会看到训练时不该直接看到的距离，同一 prompt 分块与否会给出不同结果。chunk 大于 $W$ 时，chunk 内部也必须加窗。

**prefix caching 要按窗口裁决复用范围。** 相对最新位置已经落在窗口外的 block，对后续 token 不再被直接 attend，可以安全回收；但「能回收」不等于「任意命中都等价」——命中判定与 block 回收策略要一起改，否则缓存的收益与正确性都会走样（block 级分配与共享见 [[inference-serving-03]]）。

**tensor parallel 受 KV 组数约束。** $H_{kv}=8$：TP 取 1/2/4/8 都能整除；再往上就要复制 KV head，多出来的卡只重复同一份计算。

**落到 Mistral 的语境做换算。** open-weight + 客户 on-prem、数据不出境，意味着不能靠「换更大的托管集群」解决问题，只能在客户那张 40–80 GB 卡上做预算。以 bf16、权重 13.49 GiB 计：

| 结构（32k 上下文） | 每 token | 单序列 cache | 40 GB 卡 | 80 GB 卡 |
| --- | --- | --- | --- | --- |
| MHA + full attention | 512 KiB | 16 GiB | 1 路 | 3 路 |
| GQA + full attention | 128 KiB | 4 GiB | 5 路 | 15 路 |
| MHA + SWA | 512 KiB | 2 GiB | 11 路 | 30 路 |
| GQA + SWA（v0.1） | 128 KiB | 512 MiB | 47 路 | 122 路 |

（未计激活、碎片与 kernel 开销，是容量上界；口径与 [[llm-internals-02]] 的表 3 一致。）这张表就是面试官想听到的答案形态：机制 → 公式里哪一项 → 客户卡上开几路、能撑多长上下文。

## 数值与代码验证

所有数字由 `Mistral-7B-v0.1` 的 config 复算：`num_hidden_layers=32`、`num_attention_heads=32`、`num_key_value_heads=8`、`hidden_size=4096`（$d_{head}=128$）、`sliding_window=4096`、`max_position_embeddings=32768`、`torch_dtype=bfloat16`。单位统一 1024 进制（1 KiB = 1024 B，1 GiB = 1024 MiB）。

**表 1：单 token K/V 的逐层放大（GQA，8 组）**

| 层级 | 元素数 | 字节数 |
| --- | --- | --- |
| 一层、一个 KV head 的 K | $128$ | 256 B |
| 一层、一个 KV head 的 K + V | $2 \times 128$ | 512 B |
| 一层、全部 8 个 KV head | $2 \times 8 \times 128$ | 4 KiB |
| 全部 32 层 | $2 \times 32 \times 8 \times 128 = 65{,}536$ | **128 KiB**（131,072 B） |

MHA（$H_{kv}=32$）同一算式是 512 KiB/token，比值 4×；MQA（$H_{kv}=1$）是 16 KiB/token，比 MHA 小 32×。

**表 2：不同上下文长度下单序列 cache（GiB）**

| $S$ | GQA + full | MHA + full | GQA + SWA（滚动缓冲区） |
| --- | --- | --- | --- |
| 4k（$=W$） | 0.5 | 2.0 | **0.5**（同上，不再增长） |
| 8k | 1.0 | 4.0 | 0.5 |
| 32k | 4.0 | 16.0 | **0.5** |

SWA 列的 0.5 GiB 就是 128 KiB × 4096；论文「32k 下 cache 减少 8×」即 $4.0 / 0.5$。

**表 3：prefill 计算量的三种口径（$W=4096$）**

| 口径 | 16k | 32k | 说明 |
| --- | --- | --- | --- |
| 注意力 pair 数（causal 实算） | 2.29× | 4.27× | 537M → 126M pair @32k |
| $O(S^2) \to O(S \cdot W)$ 的 $S/W$ | 4× | **8×** | 不带 causal 三角形时的渐近比 |
| 端到端 prefill FLOPs（$2PS$ + attention） | 1.15× | 1.40× | 投影与 MLP 才是大头 |

论文 16k 的实测 2× 加速落在第一行（2.29×）的量级上，而不是第二、三行——引用 8× 时要说清是缓存还是算力、是哪个口径。

**表 4：decode roofline（H100，bf16 峰值 989 TFLOPs、带宽 3.35 TB/s，带宽受限口径）**

| 场景 | GQA | MHA | 差异 |
| --- | --- | --- | --- |
| $b=1$、$S=4096$ | 4.48 ms/step（223 tok/s） | 4.96 ms/step（201 tok/s） | 1.11× |
| $b=8$、$S=32768$ | 14.6 ms/step（549 tok/s） | 45.4 ms/step（176 tok/s） | 3.11× |
| $b=40$、$S=32768$ | 55.6 ms/step（719 tok/s） | 209.5 ms/step（191 tok/s） | 3.77× |

读法：短上下文、小 batch 时每步要读的权重（13.49 GiB）压过 K/V，GQA 的带宽优势几乎看不出来；上下文与并发一上来，K/V 读取成为主项，GQA 才有 3× 以上的实际差距。这也是「为什么只在长上下文高并发场景才把 GQA 当成卖点」的数字依据。

```python
# Mistral-7B-v0.1 的真实 config 字段（sliding_window / num_hidden_layers 已核对）
CFG = dict(num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
           hidden_size=4096, sliding_window=4096, max_position_embeddings=32768)
L, H_q, H_kv = CFG['num_hidden_layers'], CFG['num_attention_heads'], CFG['num_key_value_heads']
d_head = CFG['hidden_size'] // H_q          # 128
W = CFG['sliding_window']                   # 4096
S = CFG['max_position_embeddings']          # 32768
B = 2                                       # bf16
KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3


def kv_bytes(h_kv, seq, batch=1):
    """2 * L * H_kv * d_head * S * b * bytes"""
    return 2 * L * h_kv * d_head * seq * batch * B


print('d_head', d_head)                                   # 128
print('GQA 每 token', kv_bytes(H_kv, 1) // KiB, 'KiB')     # 128
print('MHA 每 token', kv_bytes(H_q, 1) // KiB, 'KiB')      # 512
print('MHA/GQA', kv_bytes(H_q, 1) / kv_bytes(H_kv, 1))     # 4.0
print('滚动缓冲区/序列', kv_bytes(H_kv, W) / MiB, 'MiB')    # 512.0
print('全注意力 32k/序列', kv_bytes(H_kv, S) / GiB, 'GiB')  # 4.0
print('缓存倍数 S/W', S / W)                               # 8.0
print('decode 算术强度 H_q/H_kv', H_q / H_kv)              # 4.0

# 注意力 pair 数：causal full vs SWA
pairs_full = S * (S + 1) // 2
pairs_swa = sum(min(W, i + 1) for i in range(S))
print('pairs', pairs_full, pairs_swa, round(pairs_full / pairs_swa, 2))
# pairs 536887296 125831168 4.27

# 滚动缓冲区：位置 i 写进槽位 i % W，超过 W 后覆盖最旧的
buf = [None] * 4
for i in range(7):
    buf[i % 4] = i
    print('pos', i, '-> slot', i % 4, buf)
# pos 4 覆盖 pos 0 的槽位，pos 5 覆盖 pos 1 …… cache 长度恒为 4

# SWA 掩码（W=3，n=6）：第 i 行只有 [i-W, i]
n, w = 6, 3
for i in range(n):
    print(''.join('X' if 0 <= i - j <= w else '.' for j in range(n)))

# 层叠传播：第 k 层最多把信息向前带 k*W 个位置（论文 Figure 1 的结论）


def reach(i, layers):
    lo = i
    for _ in range(layers):
        lo = max(0, lo - W)
    return i - lo


print([reach(40 * W, k) for k in (1, 2, 4, 32)])
# [4096, 8192, 16384, 131072]  -> 32 层、W=4096 的理论 span 上界 131072
```

## 常见追问

- **追问**：如果只允许改一个，你会选 GQA 还是 SWA？
  - 要点：看场景。要压「单条序列的显存上限」、让长对话不随长度涨（on-prem 单卡多路、无限长会话），选 SWA，它把 $S$ 变成常数；要提高「单位显存的吞吐与 TPOT」、让长上下文服务更便宜（云端高并发、长 prompt 的 prefill），选 GQA，它把每位置字节数变成 $1/4$，且不牺牲可见范围。真实答案通常是都要：GQA 不改变每条序列的 cache 增长趋势，SWA 不降低单位位置的代价。
- **追问**：为什么 SWA 的 131k 不能当成「能处理 131k 上下文」？
  - 要点：它是路径存在性的上界，不是召回率。$L$ 层、每层窗口 $W$，信息最多前移 $L \times W$，但每一跳都做加权平均，远距离信息被反复混合与稀释。要用 needle-in-a-haystack、lost-in-the-middle 这类实验按距离测准确率，而不是用乘法算出的数字。短窗口 + 少层的模型会明显漏掉长程依赖。
- **追问**：Mistral 7B v0.2 / v0.3 为什么又把 SWA 去掉了？
  - 要点：v0.2 的 model card 写着 32k 上下文、Rope-theta = 1e6、No Sliding-Window Attention，config 里 `sliding_window=null`。可以这样解释：既然要把上下文拉到 32k 并让远处 token 直接可见，固定窗口反而成为限制；代价是 cache 重新变成 $S$ 的函数（32k 下 4 GiB/条），即用显存换长程精确性。这也说明窗口大小与上下文目标是一起决定的，不是越省越好。
- **追问**：能不能给已有的全注意力模型在推理期加上 SWA 省显存？
  - 要点：不行，会撞上 attention sink。全注意力模型会把多余注意力丢给最前面几个 token，这些 token 滑出窗口后输出会崩。可行的近似是保留开头若干个 sink token + 最近窗口（StreamingLLM 一类），但要接受信息永久丢失，且需要按任务验证掉分幅度。它与「训练时就带窗口的 SWA」不是同一件事。
- **追问**：GQA 的组数还能不能再小？质量怎么验证？
  - 要点：能，$H_{kv}=1$ 就是 MQA，cache 再小 8×，但质量退化更明显、训练可能不稳；这条轴的两个端点就是 MHA 与 MQA，$G$ 越小省得越多、风险越大。验证方式是同一训练配方下对比 benchmark 与长上下文任务，而不是只看 perplexity。
- **追问**：这两个机制分别会让服务栈的哪些部分变复杂？
  - 要点：GQA 影响 kernel 的 head 映射与 tensor parallel 切分（$H_{kv}=8$ 是分片上限），除此之外基本免费；SWA 影响 kernel（需要 local mask）、prefill 分块（跨 chunk 可见性）、prefix caching（窗口外 block 的复用与回收裁决）、以及容量规划（cache 变常数，调度器要按固定大小预留而不是按最大长度预留）。后者的工程量明显更大，这也是「有 SWA 的模型不一定在每一步都跑满它的收益」的原因。

## 相关题目

- [[llm-internals-02]]：KV cache 公式、prefill/decode 的 compute- 与 memory-bound 划分、算术强度 $H_q/H_{kv}$ 与 H100 的 295 flops/byte 拐点都来自这篇；本题只改了两个乘数。注意口径差异：那篇算 LLaMA-3-70B 的 GQA 是 8 flops/byte（64/8），Mistral 7B 是 4（32/8）。
- [[llm-internals-03]]：MHA / MQA / GQA 的分组细节、为什么只共享 K/V、uptraining 的组内权重平均做法，是本题 GQA 部分的前置（约 5% 预训练算力的口径见本题参考资料 1）。
- [[llm-internals-13]]：lost-in-the-middle。窗口外信息的有效召回要用它的实验口径来测，才能说明 $L \times W$ 只是上界。
- [[inference-serving-03]]：PagedAttention 的 block 分配与前缀共享。SWA 下「窗口外的 block 能不能复用」要在这一层裁决。

## 参考资料与归属

1. [分组查询注意力（GQA）](https://outcomeschool.com/blog/grouped-query-attention)，Amit Shekhar（Outcome School），2026-04-22。提供分组共享 K/V 的机制、MHA/MQA/GQA 是同一轴上的点、Mistral 7B 的 32 个 Q head / 8 个 KV 组的公开口径，以及 uptraining 用约 5% 预训练算力恢复质量的结论。
2. [Sliding Window Attention 是如何工作的？](https://outcomeschool.com/blog/how-does-sliding-window-attention-work)，Amit Shekhar（Outcome School），2026-09-05。提供滑动窗口的逐位置走查、层叠传播的直觉（每层看 4 个词，10 层约 40 个词）、收益与取舍清单，以及「第一个 token 滑出窗口会让质量崩掉」的 attention sink 警告。
3. [KV Cache in LLMs](https://outcomeschool.com/blog/kv-cache-in-llms)，Amit Shekhar（Outcome School），2026-03-27。提供 KV cache 随 token 数与并发增长的基本账本与 Llama-2 级模型的算例，本题的 $M$ 公式与它同源。
4. [KV Cache Compression](https://outcomeschool.com/blog/kv-cache-compression)，Amit Shekhar（Outcome School），2026-09-12。提供四类压缩路线的分类，以及「跨 head 共享 K/V 必须在训练前设计、不能事后加装」与 token eviction 必须保留开头 sink token 的定位。
5. [Mistral 7B](https://arxiv.org/abs/2310.06825)（延伸），Albert Q. Jiang 等（Mistral AI），2023-10-10。第 2 节的架构表（$L=32$、$H_q=32$、$H_{kv}=8$、$d_{head}=128$、窗口 4096）、SWA 的定义与约 131K 的理论 attention span、rolling buffer cache 的 $i \bmod W$ 与 32k 下 8× 的结论、16k 序列上 2× 的实测加速、以及 pre-fill and chunking 的三段掩码，均取自该论文。
6. [Mistral-7B-v0.1 config.json](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json)（延伸），Mistral AI，2023-09-27。本文全部算术的输入：`sliding_window=4096`、`num_hidden_layers=32`、`num_attention_heads=32`、`num_key_value_heads=8`、`hidden_size=4096`、`torch_dtype=bfloat16`。
7. [Mistral-7B-Instruct-v0.2 Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)（延伸），Mistral AI，2023-12-11。生产坑一的直接证据：v0.2 相对 v0.1 的三点变化是 32k 上下文、Rope-theta = 1e6、去掉 sliding window attention；v0.2/v0.3 的 config 中 `sliding_window` 为 `null`。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
