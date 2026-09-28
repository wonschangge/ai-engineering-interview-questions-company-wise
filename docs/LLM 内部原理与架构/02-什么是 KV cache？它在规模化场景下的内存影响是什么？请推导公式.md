---
type: question
id: llm-internals-02
topic: LLM 内部原理与架构
order: 2
question: 什么是 KV cache？它在规模化场景下的内存影响是什么？请推导公式。
question_en: What is the KV cache, and what are its memory implications at scale? Derive the formula.
asked_at: [OpenAI, xAI, Mistral AI, Amazon (AWS), Apple, NVIDIA, Together AI, Character.AI]
level: 进阶
tags: [kv-cache, inference, 显存, gqa]
sources:
  - title: KV Cache in LLMs
    url: https://outcomeschool.com/blog/kv-cache-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-03-27
  - title: KV Cache Compression
    url: https://outcomeschool.com/blog/kv-cache-compression
    author: Amit Shekhar (Outcome School)
    published: 2026-09-12
  - title: Mastering LLM Techniques：Inference Optimization（延伸）
    url: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
    author: Shashank Verma, Neal Vaidya (NVIDIA Technical Blog)
    published: 2023-11-17
related: [llm-internals-01, llm-internals-03, inference-serving-01]
updated: 2026-09-28
---

## 一句话答案

> KV cache 把每一层注意力的 Key 和 Value 按 token 存下来，让 decode 每一步只需要为最新 token 计算 K/V，把重复计算从 $O(S^2)$ 降到每步一次，代价是一份随序列长度和并发数线性增长的显存。
> 单条序列的占用是 $M = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot \text{bytes}$，并发 $b$ 条再乘 $b$。以 LLaMA-3-70B（80 层、8 个 KV 头、head_dim 128、bf16）为例，每 token 320 KiB，32k 上下文需要 10 GiB；如果同一模型用 MHA（64 个 KV 头），每 token 是 2.5 MiB，32k 需要 80 GiB，单张 80 GB 卡都放不下。
> 所以 KV cache 省的是算力而不是显存；在规模化服务里它常常比权重更早耗光显存，这正是 PagedAttention / vLLM 要解决的问题。

## 面试官在考什么

- 是否理解自回归生成里「重复计算」的本质，而不是只背下「KV cache 能加速推理」这个结论。
- 能否从张量形状一步步推出显存公式，并知道 GQA / MQA 改变的是公式里的哪一项。
- 是否清楚 prefill 与 decode 的瓶颈差异（compute-bound 与 memory-bandwidth-bound），以及 cache 在两者之间的桥梁角色。
- 能否把单请求的显存换算到并发规模（乘 $b$），并解释为什么它决定服务的吞吐上限。
- 是否知道压缩手段的分类：丢 token、降低每元素比特数、以及改模型结构这三种，各自能不能事后加装。

常见错误答案：

- 说 KV cache 是「省显存」的技术。它恰恰要多占显存，省的是算力。
- 认为缓存里存的是 Q、K、V 三份，或者把公式里的因子 2 漏掉、把 bytes 当成 1。
- 拿 MHA 的简化式 $H_{kv}\cdot d_{head} = d_{model}$ 去算 GQA 模型，把 LLaMA-3-70B 的 KV 头数当成 64，结果显存算大 8 倍。

## 原理与推导

### 1. 逐 token 生成带来的重复计算

LLM 是自回归生成的：每预测一个新 token，都要看全部历史 token。以 "I love teaching AI" 为例：

| 步骤 | 当前输入序列 | 需要算 K/V 的 token | 其中是重复计算 |
| --- | --- | --- | --- |
| 1 | I love | I, love | — |
| 2 | I love teaching | I, love, teaching | I, love |
| 3 | I love teaching AI | I, love, teaching, AI | I, love, teaching |

"I" 的 K/V 在第 1 步就算出来了，第 2、3 步又各算了一遍。而某个 token 在某一层的 K/V 是确定的：它取决于该 token 的表示、它的绝对位置（RoPE）以及该层固定的投影权重 $W_K, W_V$，因果掩码保证它永远看不到后面的 token，所以**一个 token 的 K/V 一旦算出就不会再变**。重算是纯粹的浪费，而且序列越长浪费越大。

### 2. 为什么只缓存 K、V，不缓存 Q

看注意力的计算形式 $\text{softmax}\!\left(QK^{\top}/\sqrt{d_{head}}\right)V$：

- **Q 只服务当前 token**：第 $t$ 步只用到第 $t$ 个位置的 query 向量，它和全部历史 K 做点积得到注意力分数，取完 V 之后这个 Q 就再也用不到了。
- **K、V 会被所有未来步复用**：第 $t+1, t+2, \dots$ 步的 query 都要和「包含第 $t$ 个 token 在内」的全部 K 做点积，并据此读取对应的 V。

这就是公式里那个因子 2 的来源：每个历史 token 要存 K 和 V 两份。因为不存 Q，这个因子不是 3。

### 3. 计算量对比：约 50 倍

以初始 prompt 2 个 token、最终生成到序列长度 100 为例。

无 cache 时，第 $t$ 步要为当前整段序列重算 K/V，累加起来是：

$$T_{\text{no-cache}}(S) = 2 + 3 + \cdots + S = \frac{S(S+1)}{2} - 1 = \Theta(S^2)$$

有 cache 时，prefill 一次性算完开头 2 个 token，之后每个 decode step 只为 1 个新 token 计算：

$$T_{\text{cache}}(S) = 2 + (S-2)\cdot 1 = \Theta(S)$$

代入 $S=100$：无 cache 是 $2+3+\cdots+100 = 5049$ 次 K/V 计算；有 cache 时源文按「100 个生成步」的口径记作 $2+99 = 101$ 次，比值 $5049/101 \approx 49.99$。两种口径只差一个常数：若严格按同一个终长 $S=100$ 计，cache 侧是 100 次，比值 $5049/100 = 50.49$。结论不变——**约 50 倍**，而且比值随 $S$ 线性增长，源文那句「序列越长省得越多」就是这个 $\Theta(S^2)$ 与 $\Theta(S)$ 的差。

### 4. 显存公式推导

从最小的存储单元往上乘，每一步只加一个维度：

1. 一个 KV 头、一层、一个位置上，Key 向量是 $d_{head}$ 个标量；bf16/fp16 每个元素占 $\text{bytes}=2$ → $d_{head}\cdot \text{bytes}$
2. 一层里有 $H_{kv}$ 个 KV 头 → $H_{kv}\cdot d_{head}\cdot \text{bytes}$
3. Key 和 Value 各存一份 → 乘 2
4. 一共 $L$ 层 → 乘 $L$
5. 序列长度 $S$ 个位置 → 乘 $S$
6. 并发 $b$ 条序列，每条序列的 cache 互相独立 → 乘 $b$

$$M_{kv} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b \cdot \text{bytes}$$

| 符号 | 含义 | LLaMA-3-70B 取值 |
| --- | --- | --- |
| 2 | K 与 V 两份 | 2 |
| $L$ | Transformer 层数 | 80 |
| $H_{kv}$ | KV 头数（MHA 时等于注意力头数，GQA 时更小，MQA 时为 1） | 8 |
| $d_{head}$ | 每个头的维度 | 128 |
| $S$ | 序列长度（prompt + 已生成 token） | 32k = 32768 |
| $b$ | batch size / 并发序列数 | 1 |
| bytes | 每个元素的字节数（fp16、bf16 为 2） | 2 |

两点常被追问的推广：

- 注意力头数 $H_q$ 与模型宽度满足 $d_{model} = H_q \cdot d_{head}$。令分组数 $G = H_q / H_{kv}$（MHA 时 $G=1$，MQA 时 $G=H_q$），公式可以改写成与 $d_{model}$ 挂钩的形式：

$$M_{kv} = \frac{2 \cdot L \cdot S \cdot b \cdot d_{model} \cdot \text{bytes}}{G}$$

- 很多资料写的 $2 \cdot L \cdot d_{model} \cdot S \cdot b \cdot \text{bytes}$ 就是 $G=1$ 的 MHA 特例，用 GQA 模型时必须补上这个 $1/G$。
- 只关心「每 token 多少字节」时去掉 $S$：$m = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot \text{bytes}$。位置编码不改变向量维度，RoPE 旋转后的 K 直接存下来即可，公式不受影响。

### 5. 与 prefill / decode 两阶段的关系

- **prefill**：整段 prompt 一次性喂进去，K/V 投影是矩阵×矩阵，高度并行、算力受限（compute-bound），输出第一个 token 的同时把 cache 建立起来，它决定 TTFT。
- **decode**：每步只喂上一个新 token，矩阵×向量，算力利用率低、带宽受限（memory-bound），每步读完整段 cache 再追加一格，它决定 TPOT 与吞吐。

cache 就是这两个阶段的接口：prefill 建立它，decode 依赖并延长它。

decode 为什么必然是带宽受限，可以用算术强度算一遍。一个 decode step、一层、一条序列：

- 需要读的字节数：K 和 V 各有 $H_{kv} \cdot S \cdot d_{head}$ 个元素，共 $4\,H_{kv} d_{head} S$ 字节
- 需要做的浮点运算：$QK^{\top}$ 与 $PV$ 各 $2 H_q d_{head} S$ flops，共 $4 H_q d_{head} S$ flops

两者相除得到算术强度 $H_q / H_{kv}$ flops/byte：MHA 是 1，GQA-8 是 8。而 H100 的 bf16 峰值算力约 989 TFLOPs、HBM 带宽 3.35 TB/s，拐点约 $989/3.35 \approx 295$ flops/byte。8 远低于 295，所以 decode 稳定地落在带宽一侧——这也解释了为什么「cache 更小」不只是省显存，还直接让生成更快。

### 6. trade-off 与规模化后果

一句话：**用显存换时间**。省下来的是每步重算历史 K/V 的 $\Theta(S^2)$ 计算，付出的是 $\Theta(b\cdot S)$ 的显存，以及每步额外读取 cache 的带宽。

规模化的关键在于这个乘法：权重是整个 batch 共享的一份，而 KV cache 是每条序列各自一份。并发越高、上下文越长，cache 就越可能成为主要占用。以 70B bf16 模型为例，权重约 140 GB（约 130 GiB），而 GQA 版每 GiB 显存只能装 3276.8 个 token 的 cache；换成 MHA 结构，同样的显存只能装 1/8 的 token。8 张 80 GB 卡（8 × 80 GB ≈ 596 GiB）扣掉权重后剩约 466 GiB：GQA-8 能放下约 46 条 32k 会话，MHA 只能放下约 6 条（未取整为 46.6 与 5.8，正好 8 倍）——同样参数量、同样硬件，差别全在公式里的 $H_{kv}$。

要提升这个上限，除了压缩 cache 本身，还有服务层的显存管理问题：按 `max_seq_len` 为每个请求预分配一整块连续显存，会让预留但没用到的部分和碎片一起浪费掉；vLLM 的 PagedAttention 把 cache 切成固定大小的 block、按需分配、用 block table 做映射，同时支持前缀共享。它的动机直接来自本节推导出的 $b \cdot S$，细节留给 [[inference-serving-01]]。

### 7. KV cache 压缩的四条路线

| 路线 | 做法 | 需要重训 | 丢 token | 典型收益 | 代表 |
| --- | --- | --- | --- | --- | --- |
| 量化 | 每个元素用 8/4 bit 存储，K 与 V 分别定标 | 否 | 否 | 2×–4× | KIVI |
| 丢弃 token | 只保留最近窗口 + 最前面几个 sink token，其余按注意力累计分数保留 heavy hitter | 否 | 是 | cache 变成固定大小 | StreamingLLM、H2O |
| 跨头共享 K/V | 减少 $H_{kv}$，query 头仍然各自独立 | 是 | 否 | 4×–8×（MQA 可到 $H_q$ 倍） | GQA、MQA |
| 低秩压缩 | 每层每 token 只存一个 latent 向量，用时再展开 | 是 | 否 | 10× 以上 | MLA（DeepSeek-V2/V3） |

- **滑窗 + attention sinks** 是唯一能真正止住无限增长的方案：只保留最近窗口加最前面几个 token，cache 大小固定。必须保留开头那几个 token，是因为 softmax 的概率要归一化到 100%，当历史里没有真正有用的 token 时，模型学会了把多余的注意力丢给最前面的位置，它们成了 attention sink。一旦把第一个 token 淘汰掉，输出会直接崩掉。
- 代价是信息永久丢失：适合聊天、摘要，不适合「找出第 40 页提到的电话号码」这类精确回忆。
- 剩下的三条都不丢 token：量化最容易事后加装，只是低于 4 bit 后误差上升；低秩压缩把 $H_{kv}\cdot d_{head}$ 换成更小的 latent 维度，效果最好但必须在训练前设计进去。
- GQA / MQA 属于「改模型结构」这一类压缩，它改的就是公式里的 $H_{kv}$ 项，分组方式与质量权衡留给 [[llm-internals-03]]。

## 数值与代码验证

以下数字都用 LLaMA-3-70B 的 `config.json` 参数复算：`num_hidden_layers=80`、`num_attention_heads=64`、`num_key_value_heads=8`、`hidden_size=8192`（因此 $d_{head}=8192/64=128$），精度 bf16。单位统一按 1024 进制：1 KiB = 1024 B，1 GiB = 1024 MiB。

**表 1：单 token cache 的逐层放大过程（GQA-8）**

| 层级 | 元素数 | 字节数 |
| --- | --- | --- |
| 一层、一个 KV 头的 K | $128$ | 256 B |
| 一层、一个 KV 头的 K + V | $2 \times 128$ | 512 B |
| 一层、全部 8 个 KV 头 | $2 \times 8 \times 128$ | 4 KiB |
| 全部 80 层 | $2 \times 80 \times 8 \times 128 = 163{,}840$ | **320 KiB**（327,680 B） |

**表 2：不同上下文长度下单条序列的 cache（GiB）**

| 上下文 $S$ | GQA-8（$H_{kv}=8$，320 KiB/token） | MHA（$H_{kv}=64$，2.5 MiB/token） |
| --- | --- | --- |
| 8k | 2.5 GiB | 20 GiB |
| 32k | **10 GiB** | **80 GiB** |
| 128k | 40 GiB | 320 GiB |

**表 3：8×H100（8 × 80 GB ≈ 596 GiB）上 32k 会话的并发上限**

| 项 | GQA-8 | MHA |
| --- | --- | --- |
| 权重（70B bf16） | 约 140 GB（约 130 GiB） | 同左 |
| 可留给 KV cache（596 GiB − 130 GiB 权重） | 约 466 GiB | 约 466 GiB |
| 每条 32k 会话占用 | 10 GiB | 80 GiB |
| 并发上限（未计激活与碎片） | 约 46 | 约 6 |

**表 4：K/V 计算次数对比（源文口径）**

| 口径 | 无 cache | 有 cache | 比值 |
| --- | --- | --- | --- |
| 源文的 100 步口径 | 5049 | 101 | 49.99 |
| 同一终长 $S=100$ | 5049 | 100 | 50.49 |

```python
KiB = 1024
MiB = 1024**2
GiB = 1024**3


def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, batch=1, bytes_per_elem=2):
    """2 * L * H_kv * d_head * S * b * bytes"""
    return 2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_elem


# LLaMA-3-70B：80 层、64 个注意力头、8 个 KV 头、head_dim 128、bf16
gqa = kv_cache_bytes(80, 8, 128, 1)
mha = kv_cache_bytes(80, 64, 128, 1)
print(gqa / KiB, "KiB/token")   # 320.0
print(mha / MiB, "MiB/token")   # 2.5

for seq in (8_192, 32_768, 131_072):
    print(seq, f"GQA {kv_cache_bytes(80, 8, 128, seq) / GiB:.1f} GiB",
          f"MHA {kv_cache_bytes(80, 64, 128, seq) / GiB:.1f} GiB")
# 8192   GQA 2.5 GiB    MHA 20.0 GiB
# 32768  GQA 10.0 GiB   MHA 80.0 GiB
# 131072 GQA 40.0 GiB   MHA 320.0 GiB

# 每 GiB 能装多少 token —— 容量规划时的口算常数
print(GiB / gqa)                # 3276.8 tokens/GiB

# 计算量对比
print(sum(range(2, 101)), 2 + 99, sum(range(2, 101)) / (2 + 99))
# 5049 101 49.99009900990099
```

与源文对照：源文用「32 层、32 头、$d_{head}=128$、2 字节」的 Llama-2 级模型举例，单 token 是 $2\times32\times32\times128\times2 = 524{,}288$ 字节，记作 0.5 MB；4000 token 约 2 GB，100k token 约 50 GB。按 1024 进制，同一个数是 512 KiB/token、1.95 GiB 和 48.8 GiB，差异只是十进制与二进制单位的口径（1 GiB = 1.074 GB），源文的计算本身没有错。

## 常见追问

- **追问**：KV cache 能跨请求复用吗？
  - 要点：能，前提是前缀完全相同。这就是 prompt caching / prefix caching：system prompt、few-shot 示例、多轮对话的历史前缀都可以命中，命中后省掉的是 prefill 的算力和 TTFT，decode 每一步的开销不变。因果注意力下 K/V 依赖全部前序 token，前缀只要有一位不同，从该位往后的 cache 全部失效。
- **追问**：长对话怎么防止 cache 无限增长？
  - 要点：$M \propto S$，不设上限迟早撞到显存。可选手段是滑窗 + attention sinks（固定大小）、按注意力分数淘汰 token、KV 量化、把窗口外的历史做成摘要或检索、直接限制上下文长度。取舍是「精确记忆」换「固定显存」：丢 token 的方案在需要精确回忆的任务上会失败。
- **追问**：为什么说 KV cache 是省算力而不是省显存？
  - 要点：它不减少任何必需的存储，反而新增一份与 $b\cdot S$ 同阶的显存。省下的是每步重算历史 K/V 的开销：生成到长度 $S$ 时，K/V 投影次数从无 cache 的 $\Theta(S^2)$ 降到 $\Theta(S)$。副作用是瓶颈位置从算力搬到了显存容量和带宽上。
- **追问**：batch 变大时显存怎么变？
  - 要点：对 cache 而言严格线性，$M \propto b$，因为每条序列的 cache 互相独立，只有权重是共享的一份。所以 batch 上限通常由 KV cache 而不是权重决定；增大 batch 摊薄了权重读取、提升吞吐，同时线性推高 cache 占用和每步的 cache 读取量。这正是 continuous batching 与 PagedAttention 的直接动机。
- **追问**：把 cache 量化到 int8 / fp8 会怎样？
  - 要点：公式里的 bytes 从 2 变成 1，显存和读取带宽都线性减半；4 bit 再减半。难点是 Key 上存在少数幅度极大的通道，per-tensor 量化误差很大，KIVI 这类方案对 K 用 per-channel、对 V 用 per-token 的粒度。量化同时省容量和带宽，所以通常连每步延迟一起降。
- **追问**：FlashAttention 是不是也在压缩 KV cache？
  - 要点：不是。FlashAttention 用 tiling 减少 $S\times S$ 注意力分数矩阵在 HBM 上的读写，是 kernel 级的 IO 优化，不改变 cache 的形状和大小，两者正交可叠加。PagedAttention 同样不是压缩，它解决的是显存分配与碎片。真正缩小 cache 的只有量化、token eviction、GQA/MQA 和 MLA。

## 公司变体

`asked_at` 覆盖的八家公司在公开材料/技术博客里对这道题的侧重方向不同：

- **OpenAI、xAI、Mistral AI、Together AI**：推理栈与服务方向，偏工程。常从「给定 GPU 数量能服务多少并发、长上下文成本如何」出发，落到 prompt caching / prefix 复用带来的成本变化。
- **Amazon (AWS)**：托管服务（Bedrock、SageMaker）视角，偏容量规划与成本：实例选型、显存与带宽比、多租户下的 cache 隔离与配额。
- **Apple**：端侧与私有云计算，偏内存预算和能效，关注小 batch、低比特量化、固定长度窗口这类约束下的取舍。
- **NVIDIA**：框架与 kernel 视角（TensorRT-LLM），偏实现细节：in-flight batching、PagedAttention 的 block 管理、MQA/GQA 的 kernel 支持、prefill 与 decode 的 compute/memory-bound 划分。
- **Character.AI**：大规模高并发对话场景，偏「KV cache 与内存带宽」这条线，量化 cache、固定窗口是最常被追问的工程手段。

以上是基于各家公开技术材料与岗位方向的侧重判断，具体题目以实际面试轮次为准。

## 相关题目

- [[llm-internals-01]]：scaled dot-product attention 与 $1/\sqrt{d_k}$ 缩放，解释 K/V 从哪来、为什么是这个形状。
- [[llm-internals-03]]：MHA / MQA / GQA 的分组方式与质量权衡，决定本节公式里 $H_{kv}$ 的取值。
- [[inference-serving-01]]：PagedAttention 与 vLLM，把 $b \cdot S$ 的显存需求变成可调度、无碎片的 block 分配。

## 参考资料与归属

1. [KV Cache in LLMs](https://outcomeschool.com/blog/kv-cache-in-llms)，Amit Shekhar（Outcome School），2026-03-27。提供逐 token 生成与 "I love teaching AI" 的重复计算例子、只缓存 K/V 的理由、5049 与 101 的计算量对比、显存换时间的 trade-off，以及滑窗 + attention sinks 的定位。
2. [What is KV Cache Compression?](https://outcomeschool.com/blog/kv-cache-compression)，Amit Shekhar（Outcome School），2026-09-12。提供 Llama-2 级模型的单 token 显存算例、量化与 token eviction（H2O、StreamingLLM、attention sink）、跨头共享（GQA/MQA）、低秩压缩（MLA）四条路线的对比与「何时用哪一种」。
3. [Mastering LLM Techniques：Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（延伸），Shashank Verma、Neal Vaidya（NVIDIA Technical Blog），2023-11-17。第 3 节显存公式中「每 token 字节数 = 2 × 层数 × 头数 × head_dim × 精度字节数」及其带 batch、sequence length 的总量形式取自该文，prefill/decode 的 compute-bound 与 memory-bound 划分、MQA/GQA 与 PagedAttention 在推理优化中的位置也取自该文；第 3、4 节另外用 LLaMA-3-70B 的参数把公式复算成了 320 KiB/token 与 10 GiB@32k 的具体数字。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
