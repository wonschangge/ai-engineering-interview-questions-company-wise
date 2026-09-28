---
type: question
id: inference-serving-10
topic: 推理、服务与 GPU 性能
order: 10
question: 做一下 roofline 计算：在 batch size 为 1 时，一块 H100 服务 70B 模型每秒能产出多少 token？
question_en: Do the roofline maths: how many tokens per second can one H100 produce for a 70B model at batch size 1?
asked_at: [NVIDIA, Together AI]
level: 高阶
tags: [roofline, 算术强度, 带宽, 容量规划]
sources:
  - title: Prefill vs Decode：LLM 推理优化
    url: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [inference-serving-01, inference-serving-06, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> batch=1 的 decode 每生成一个 token 都要把参与计算的全部权重从 HBM 读一遍，算术强度只有约 1 FLOP/byte，远低于 H100 SXM5 的 roofline 拐点 989 TFLOPs ÷ 3.35 TB/s ≈ 295 FLOPs/byte，所以这题是纯带宽题：**tokens/s ≈ 有效带宽 ÷ 每 token 要读的字节数**。
> 但 70B 的 bf16 权重是 140 GB，一块 80 GB 的 H100 装不下，所以必须先声明前提，三种口径的答案分别是：FP8 权重 70 GB → 上界约 48 tokens/s，扣除 20%–40% 的实现损耗后落在 **30–40 tokens/s**；INT4 权重 35 GB → 上界约 96，实际 **60–70 tokens/s**；bf16 走 8 卡 TP=8、每卡 17.5 GB → 单卡上界约 191，但每层两次 all-reduce 会吃掉相当一部分，端到端实际 **100–150 tokens/s**。
> 答案依赖精度与并行度，先讲清前提再给数字，比直接背一个数更重要。

## 面试官在考什么

- 会不会查/背硬件常数，并且**说清口径**：989 TFLOPs 是 bf16 稠密（不含 2:4 sparsity）、3.35 TB/s 是 SXM5 的 HBM3、80 GB 是容量。把 H100 PCIe 的数（HBM2e 2 TB/s、bf16 稠密 378 TFLOPs，厂商表上的 756 是含稀疏口径）和 SXM5 的混着用，整道题的数字就全错了。
- 能不能**自己发现题目里的矛盾**：70B bf16 = 140 GB > 80 GB，单卡根本放不下，所以「一块 H100 服务 70B」这句话本身需要先被澄清成量化或张量并行。
- 会不会把「读权重的字节数」和「做多少 FLOPs」放在同一个坐标系里比 —— 也就是先算算术强度、再和 roofline 拐点比，而不是直接背「decode 是 memory bound」这个结论。
- 是否区分**上界与实测**：带宽理论值要先扣 HBM 效率，再扣 KV cache 读取、kernel 启动与融合质量、采样开销。
- 能不能顺着同一个模型推到 batch>1：算术强度随 batch 线性上升，在 batch≈295 处转为计算受限，吞吐曲线在这里弯折 —— 这正是 continuous batching 的收益来源。

常见错误答案：

- 拿 140 GB 去除 3.35 TB/s 得 24 tokens/s，却不说清这是「bf16 权重、单卡且假设装得下」的口径；更常见的是直接忽略掉 140 GB > 80 GB、单卡装不下这个前提。
- 用 **FP8 稀疏口径的 3958 TFLOPs**（稠密是 1979）或 bf16 含稀疏的 1979（稠密 989）去除某个 FLOPs 数，把两个精度、两种稀疏口径混在一起，得出一个毫无意义的巨大吞吐。
- 把 prefill 的结论套到 decode 上：prefill 是算力受限，于是以为 decode 也受算力限制，进而去找 FLOPS 峰值而不是带宽。batch=1 的 decode 每一层都是矩阵乘向量，算力利用率极低，瓶颈在带宽侧。

## 原理与推导

### 1. 先把硬件常数与权重账摆出来

| 项 | H100 SXM5 | 说明 |
| --- | --- | --- |
| HBM 容量 | 80 GB | HBM3 |
| HBM 带宽 | 3.35 TB/s | 以下所有带宽计算的分母 |
| bf16 稠密算力 | 989 TFLOPs | fp16 相同；厂商表里的 1979 是**含 2:4 sparsity** 的口径 |
| FP8 稠密算力 | 1979 TFLOPs | 厂商表的 3958 是**含 2:4 sparsity** 口径 |
| roofline 拐点 | $989 / 3.35 \approx 295$ | 单位 FLOPs/byte，即 bf16 的 machine balance |

权重账：70B 参数 × 2 字节（bf16）= **140 GB**，一块 80 GB 的卡连权重都放不下，更不用说还要留 KV cache 和 activations。所以题干必须被翻译成三选一：

1. **降精度**：FP8/INT8 权重 70 GB（单卡刚好装得下），INT4 权重约 35 GB（含 scale 大约再多 1%–3%）。
2. **张量并行**：8 卡 bf16，每卡 17.5 GB。
3. 两者叠加：8 卡 FP8，每卡 8.75 GB。

下面三种都算一遍，这是这题的标准答法。

### 2. 为什么 decode 每生成一个 token 要读一遍全部权重

自回归 decode 每个 step 的输入是 1 个 token。看一层里的主投影：

- 注意力部分的 $W_Q, W_K, W_V, W_O$，形状 $d_{model} \times d_{model}$ 量级；
- MLP 部分的 $W_{gate}, W_{up}, W_{down}$，形状约 $d_{model} \times 4d_{model}$。

新 token 的激活是一个 $d_{model}$ 维的**向量**，所以每个 step 是「矩阵 × 向量」（GEMV），不是「矩阵 × 矩阵」（GEMM）。矩阵的每一个元素都必须参与这一次乘法，因此**这个 step 必须把该矩阵的全部权重字节从 HBM 搬进 SM**。下一层同样，直到最后一层。

FLOPs 与字节数：

- FLOPs：一个权重参与一次乘加（MAC）。按「1 次 MAC = 2 FLOPs」的惯例，$N$ 个参数的模型前向一次约 $2N$ FLOPs，70B 即 **约 140 GFLOPs/token**。
- 字节数：bf16 下 1 个权重 2 字节，于是 $2N$ FLOPs 对应 $2N$ 字节 —— 算术强度约 **1 FLOP/byte**。
- 这个「$2N$ FLOPs 对 $2N$ 字节」的巧合只发生在 bf16 上；FP8 是 2 FLOPs/byte，INT4 是 4 FLOPs/byte。

1、2、4 全部远低于拐点 295，所以 batch=1 的 decode 稳定落在 roofline 的带宽一侧。这同时解释了另外两件事：

- **prefill 为什么不是带宽受限**：prompt 有 $S$ 个 token，权重只读一遍却被用了 $S$ 次，算术强度变成 $S$ 倍，长 prompt 下迅速越过 295，转到算力侧。
- **KV cache 的价值**：不做 cache 的话每个 step 要重算全部历史 token 的 K/V，算力浪费的量级是 $O(S^2)$；做了 cache 只多读 $O(S)$ 的字节。

### 3. 上界公式

带宽受限意味着时间由「搬了多少字节」决定：

$$t_{\text{step}} \approx \frac{\text{每 token 需要读的字节数}}{\text{有效带宽}}, \qquad \text{tokens/s} \approx \frac{BW_{\text{eff}}}{W_{\text{per-token}} + KV_{\text{per-token}}}$$

batch=1、短上下文时 $KV_{\text{per-token}}$ 可以忽略，$W_{\text{per-token}}$ 就是「本卡负责的那部分权重」。代入三种配置（$BW$ 取 3.35 TB/s 的理论值）：

| 配置 | 每卡权重 | 每 token 读取 | 理论上界 | 说明 |
| --- | --- | --- | --- | --- |
| FP8/INT8 单卡 | 70 GB | 70 GB | $3.35\times10^{12}/(70\times10^{9}) \approx 47.9$ | 恰好装进 80 GB |
| INT4 单卡 | 35 GB | 35 GB | $3.35\times10^{12}/(35\times10^{9}) \approx 95.7$ | 剩余空间给 KV cache |
| bf16 TP=8 | 17.5 GB | 17.5 GB | $3.35\times10^{12}/(17.5\times10^{9}) \approx 191.4$ | 每卡口径，需扣通信 |

注意这两步：

- 用 **FLOPs 侧交叉验证**：计算侧的上界是 $989\times10^{12}/(140\times10^{9}) \approx 7064$ tokens/s。两个上界取小得到 47.9，再次确认是带宽受限，且带宽侧比计算侧紧 $7064/47.9 \approx 147$ 倍 —— 这个 147 就是第 6 节要用的 batch 临界点（FP8 权重配 bf16 算力口径这一套）。
- 「上界」的含义是 HBM 利用率 100% 且 kernel 零开销。真实 decode 拿不到这个数：GEMV 的访存模式、权重反量化、kernel 启动间隙、采样与 logits 计算、调度与组批开销都在抢时间。工程上按 60%–80% 的 HBM 有效利用率估：

| 配置 | 理论上界 | ×0.6 | ×0.7 | ×0.8 | 建议报的区间 |
| --- | --- | --- | --- | --- | --- |
| FP8 单卡 | 47.9 | 28.7 | 33.5 | 38.3 | **30–40 tokens/s** |
| INT4 单卡 | 95.7 | 57.4 | 67.0 | 76.6 | **60–70 tokens/s**（乐观到 75） |
| bf16 TP=8 | 191.4 | 114.9 | 134.0 | 153.1 | 再扣 TP 通信后 **100–150 tokens/s** |

### 4. 长上下文时 KV cache 不再是零

KV cache 的每 token 字节数（推导见 [[llm-internals-02]]）：

$$KV_{\text{per-token}} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot \text{bytes}$$

以 LLaMA-3-70B 的 GQA 配置（$L=80$、$H_{kv}=8$、$d_{head}=128$、bf16）代入：$2\times80\times8\times128\times2 = 327{,}680$ 字节 = **320 KiB/token**。

decode 每个 step 要读**当前长度 $S$ 的全量 cache**，所以 $S$ 一大这笔账就不能忽略：

| $S$ | KV cache | 相对 INT4 权重 35 GB | 相对 FP8 权重 70 GB |
| --- | --- | --- | --- |
| 4k | 1.25 GiB | 3.8% | 1.9% |
| 32k | **10 GiB** | **30.7%** | 15.4% |

$S=32\text{k}$、INT4 单卡时的每 token 读取量变成 $35\ \text{GB} + 10\ \text{GiB} \approx 45.7$ GB，上界掉到 $3.35\times10^{12}/(45.7\times10^{9}) \approx 73$ tokens/s，乘 0.7 是 **约 51 tokens/s** —— 比短上下文时的 67 少了四分之一。这就是「同一份权重、同一块卡，长对话流式更慢」的定量原因。

### 5. 为什么 TP=8 拿不到 8 倍的 191

张量并行把每一层的权重切成 8 份，但**每一层都要做两次 all-reduce**：注意力输出投影后一次、MLP 输出投影后一次。70B 的 $L=80$，于是一个 token 要串行经过 $80\times2 = 160$ 次集合通信。

- 单次通信量：$d_{model}=8192$ 个元素，bf16 下 16 KiB；按 NVLink 900 GB/s 的聚合带宽算，这 16 KiB 的传输时间只有约 0.02 μs，但集合通信的固定延迟在微秒量级（$\alpha + \beta n$ 里的 $\alpha$ 项），160 次串起来就是毫秒级。
- 对照基准：每卡读 17.5 GB 权重的时间是 $17.5/3.35 \approx 5.2$ ms。通信一旦进入同一量级，191 就不可达。

结论性的三点，面试里说清楚就够：

1. TP 减少的是**每卡要读的字节数**（一个 token 每卡只读 1/8 权重），但单 token 延迟的改善达不到 8 倍：通信延迟落在关键路径上，且不像算力那样能被 batch 摊薄。191 的理想上界落到 100–150，差的 20%–50% 就是这 160 次 all-reduce。
2. TP 的通信延迟不随 batch 变化，所以 **batch 越大 TP 越划算** —— 这正是「低延迟小 batch 用 TP，高吞吐大 batch 用更大 TP/更多卡」的由来。
3. 8 卡 bf16 的 **100–150 tokens/s 是端到端单请求速率**；它与参考源「单请求 decode 时算力单元只有 20%–40% 忙」的观察自洽 —— 同一批卡如果齐刷刷跑 batched decode，总吞吐远高于把请求一条条串起来，这正是 [[inference-serving-01]] 与 [[inference-serving-02]] 的主题。

### 6. batch>1：算术强度线性上升，在约 295 处弯折

batch=$b$ 时，一次权重读取被 $b$ 个 token 共享，所以每 token 摊到的权重字节变成 $W/b$，算术强度变成：

$$AI(b) = \frac{2Nb}{W_{\text{bytes}}} = b \cdot AI(1)$$

bf16 下 $AI(1)=1$ FLOP/byte，于是 $AI(b)=b$。令它等于拐点：

$$b^{*} = \frac{\text{ridge}}{AI(1)} = \frac{989 \times 10^{12}/(3.35 \times 10^{12})}{2N/W_{\text{bytes}}} = 295 \quad (\text{bf16 权重，bf16 算力口径})$$

- $b \ll 295$：带宽受限，吞吐 $\approx b \cdot BW/W$，随 batch 线性上升。
- $b = b^{*}$：带宽上界与计算上界相等，即 $W_{\text{bytes}}/BW = 2Nb/(989\times10^{12})$，解得 $b^{*} = \text{ridge}/AI(1)$；bf16 权重下 $AI(1)=1$，正好落在 295，FP8 权重配 bf16 算力口径则是 148。
- $b \gg 295$：计算受限，吞吐停在 $989\times10^{12}/(140\times10^{9}) \approx 7064$ tokens/s 附近，不再随 batch 增长。

临界点随「权重精度」与「算力口径」是否配套而变，因为 ridge 与 $AI(1)$ 两个量都在动：算力固定按 bf16 稠密 989 TFLOPs 时，bf16 权重 $b^*\approx295$、FP8 权重 $\approx148$、INT4 权重 $\approx74$；若 GEMM 本身跑在 FP8 Tensor Core 上（稠密峰值同比翻倍到 1979），算术强度与峰值同时翻倍，FP8 权重的临界点又回到 $\approx295$。所以 H100 上的临界点落在 74–295 之间，报数前先声明用哪套口径。

这条曲线就是吞吐-延迟曲线的形状来源：batch 从 1 涨到 295，吞吐近似线性、TPOT 基本不变甚至略降（权重读取被摊薄）；越过 295 后吞吐封顶而 TPOT 继续变大，继续加 batch 只赔延迟不赚吞吐。把这条曲线画在面试白板上，continuous batching 的必要性就不需要额外解释了 —— 单请求 batch=1 只用到 1/295 的算力预算，服务端要做的事就是把这个乘数用满，见 [[inference-serving-02]]。

### 7. 面试答题模板

1. **写常数**：H100 SXM5 80 GB / 3.35 TB/s / bf16 稠密 989 TFLOPs（注明不含 sparsity）。
2. **算拐点**：$989/3.35 \approx 295$ FLOPs/byte。
3. **算算术强度**：decode 每 token 读全部权重、做约 $2N$ FLOPs，bf16 下约 1 FLOP/byte（FP8 是 2，INT4 是 4）。
4. **判瓶颈**：1 < 295，带宽受限，因此用 $BW/W$ 而不是 FLOPS 峰值。
5. **发现前提问题**：70B bf16 需 140 GB > 80 GB，先声明是量化还是 TP。
6. **算上界**：70 GB → 47.9；35 GB → 95.7；17.5 GB → 191.4。
7. **扣开销**：HBM 效率 60%–80%、KV cache（$S=32\text{k}$ 时 10 GiB，占 INT4 权重的 30%）、TP 的 160 次 all-reduce。
8. **给区间加前提**：「FP8 单卡、短上下文约 35 tokens/s；INT4 约 65；8 卡 bf16 TP 约 100–150」。

## 数值与代码验证

以下每个数字都在下面这段脚本里复算过。单位口径：容量、带宽、算力、权重用**十进制** GB/TB（厂商标称口径），KV cache 用**二进制** GiB 并单独标注 —— 1 GiB = 1.0737 GB，混用会让 KV cache 那几项差 7%。

```python
GB, GiB, TB = 1e9, 1024**3, 1e12
BW = 3.35 * TB              # H100 SXM5 HBM3
PEAK = 989e12               # bf16 稠密；含 2:4 sparsity 是 1979e12
N = 70e9                    # 参数量
FLOP_PER_TOKEN = 2 * N      # 一次乘 + 一次加，约 140 GFLOPs

# 1) roofline 拐点
ridge = PEAK / BW                      # 295.2238805970149 FLOPs/byte
assert abs(ridge - 295.2) < 0.05

# 2) 权重账：三种精度
for name, bits in [("bf16", 16), ("fp8", 8), ("int4", 4)]:
    print(name, N * bits / 8 / GB, "GB", N * bits / 8 / GiB, "GiB")
# bf16 140.0 GB    fp8 70.0 GB    int4 35.0 GB

# 3) 算术强度（batch=1，只算权重读取）
for name, bits in [("bf16", 16), ("fp8", 8), ("int4", 4)]:
    ai = FLOP_PER_TOKEN / (N * bits / 8)
    print(name, ai, "FLOPs/byte")      # 1.0 / 2.0 / 4.0，全部 << 295

# 4) 带宽上界（batch=1，短上下文）
for name, wbytes in [("fp8", N), ("int4", N / 2), ("bf16 TP=8", N * 2 / 8)]:
    print(name, BW / wbytes)           # 47.857 / 95.714 / 191.429
comp_ceiling = PEAK / FLOP_PER_TOKEN   # 7064.28 tokens/s，比带宽侧松 147 倍
print(BW / N / comp_ceiling)           # 0.006774... -> 带宽侧紧 147.6 倍

# 5) batch 临界点：AI(b) = b * 2N/W_bytes = ridge（算力固定按 bf16 稠密 989）
for name, bytes_per_weight in [("bf16", 2), ("fp8", 1), ("int4", 0.5)]:
    print(name, ridge * bytes_per_weight / 2)   # 295.2 / 147.6 / 73.8

# 5b) 若 GEMM 也跑在 FP8 Tensor Core（稠密 1979），AI(1) 与峰值同时翻倍，
#     FP8 权重的临界点回到 1979/3.35/2 ≈ 295
print((1979e12 / BW) / 2)                       # 295.4

# 6) KV cache（LLaMA-3-70B GQA：80 层 / 8 KV 头 / head_dim 128 / bf16）
def kv(S, L=80, h=8, d=128, eb=2):
    return 2 * L * h * d * S * eb

print(kv(1))                # 327680 B/token = 320 KiB
print(kv(32768) / GiB)      # 10.0 GiB
print(kv(32768) / (N / 2))  # 0.3068 -> 长上下文时 KV 已是 INT4 权重的 30.7%

# 7) 长上下文（S=32k）下 INT4 单卡的上界
total = N / 2 + kv(32768)
print(BW / total, BW / total * 0.7)    # 73.2 -> 51.3 tokens/s
```

与源文对照：所用参考源只给出了定性结论（decode 是 memory-bandwidth-bound、算力单元在单请求 decode 下只有 20%–40% 忙、prompt 长度增长时 attention 部分按平方增长）和「比较了精度与实际利用率」的框架，没有给出本题的具体 token/s。本节的 295、47.9、95.7、191.4、7064、295.2/147.6/73.8、320 KiB、10 GiB 全部由上式自行复算，比例关系的自洽性也在脚本里断言过。延伸来源给出的是「以解析模型做推理效率的划分决策、以及 int8 权重下 540B 模型低 batch 延迟 29 ms/token」这一同构口径的参考点：29 ms/token 对应约 34 tokens/s，与本节「INT8/FP8 单卡 30–40 tokens/s」的区间同一量级，但两者模型规模、硬件（TPU v4）与上下文都不同，只能作为量级校验，不能直接对比。

## 常见追问

- **追问**：为什么用 989 而不是 1979 TFLOPs？
  - 要点：1979 是 bf16 **含 2:4 structured sparsity** 的口径，需要权重本身满足结构化稀疏才能吃到；稠密 GEMM 的有效峰值是 989。同样的坑还有 FP8：厂商表里 3958 是稀疏，稠密是 1979。本题带宽受限，用哪个都不影响结论，但口径写错会被立刻抓住。
- **追问**：batch=1 时 H100 的算力利用率是多少？
  - 要点：用「计算侧吞吐 ÷ 峰值」估：$47.9/7064 \approx 0.68\%$。这与参考源说的「单请求 decode 时算力单元只有 20%–40% 忙」不是同一口径 —— 后者是算力单元（math units）在单请求 decode 下「忙」的时间占比，其中已经混进了大量等数据的时间；前者是纯 TFLOPs 利用率，量级差异来自分母不同。回答时说明自己用的是哪一个。
- **追问**：那如果换成 H100 PCIe 呢？
  - 要点：PCIe 版是 HBM2e 2 TB/s，bf16 稠密 378 TFLOPs（厂商表的 756 与 SXM 的 1979 一样都是含稀疏口径）。拐点 $378/2 = 189$，仍是带宽受限；FP8 权重 70 GB 的上界掉到 $2\times10^{12}/(70\times10^{9}) \approx 28.6$ tokens/s，实际约 18–23。**带宽差 1.7 倍，吞吐就差 1.7 倍**，这正是这题要建立的直觉。
- **追问**：MoE 模型（比如总参 70B、激活 8B）怎么算？
  - 要点：公式不变，但 $W$ 换成**每个 token 实际激活的专家权重**。batch=1 时激活约 8B 参数：bf16 下要读 16 GB，上界 $3.35\times10^{12}/(16\times10^{9}) \approx 209$ tokens/s；权重压到 FP8 则读 8 GB，上界约 419。代价是显存里仍要驻留全部专家的权重（总参 70B bf16 仍是 140 GB），而且 batch=1 时专家负载无法摊薄，路由不均会让某些卡空转。
- **追问**：投机解码能不能突破这个上界？
  - 要点：能提高有效 tokens/s，但机制不是提高带宽。一次验证 $k$ 个 draft token 仍然只读一遍权重，一次权重读取平均换回 $k_{\text{accept}}$ 个 token，所以摊到每个 token 的**有效**权重读取降到 $W/k_{\text{accept}}$。它改的是「一次权重读取换几个 token」这个乘数，不是带宽本身，也不改变带宽上界对单个 token 的结论。
- **追问**：规格表里的 3.35 TB/s 实测能拿到多少？
  - 要点：纯拷贝 benchmark 通常能到 85%–90%，但 decode 是 GEMV + 反量化 + 归约混在一起，访存不连续、算子碎片化，60%–80% 是更现实的区间 —— 这就是上界要乘 $\eta$ 的原因。想知道自己系统上的 $\eta$，量一下「每秒实际读取字节数 ÷ 3.35 TB/s」即可，不需要跑微基准。

## 公司变体

- **NVIDIA**：偏工程实现与 kernel。公开面试信息显示其 GPU 性能轮次会围绕 roofline 展开，追问方向通常是「你的 kernel 拿到了多少 HBM 带宽、GEMV 为什么打不满、TensorRT-LLM 里哪些融合能减少权重读取」，而不是让你停在 47.9 这个数字上。也可能反问「同样的模型换成 FP8 kernel，为什么实测提升不到 2 倍」——答案是权重量减半但 KV cache 与激活的读取没减、反量化增加了指令开销。
- **Together AI**：偏容量规划与成本，关注点从「一块卡多少 token/s」迅速转到「一个 8 卡节点能服务多少并发、每百万 token 的 GPU 成本是多少」。因为他们是推理服务商，通常会把 batch=1 的答案当作延迟下界，然后立刻问 batch 拉到多少吞吐开始不涨（本题的 295），以及 FP8/INT4 在哪些模型上能直接上线。

以上是基于两家公开技术材料与岗位方向的侧重判断，不代表具体面试流程。

## 相关题目

- [[inference-serving-01]]：prefill 与 decode 的瓶颈差异，是本题「为什么用带宽而不是 FLOPS」的前置结论。
- [[inference-serving-06]]：FP16/BF16/FP8/INT8/INT4/FP4 的对比，决定本题 $W_{\text{bytes}}$ 取什么值、以及量化能拿回多少吞吐。
- [[inference-serving-02]]：continuous batching，把本题 batch 从 1 推到 295 那个拐点的工程手段。
- [[llm-internals-02]]：KV cache 的字节公式，本题第 4 小节长上下文修正项的来源。

## 参考资料与归属

1. [Prefill vs Decode：LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)，Amit Shekhar（Outcome School）。提供 prefill 计算受限 / decode 内存带宽受限的划分、单请求 decode 时算力单元 20%–40% 忙的量级、decode 每步要「流式读取全部权重加增长的 KV cache」这一成本表述、KV cache 的读写角色，以及 batch 变大可以在几乎不增加单步成本的前提下提高吞吐（continuous batching 的动机）。本文第 2、3 节的瓶颈判定与第 6 节的 batch 讨论以该文的结论为出发点。
2. [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)（延伸），Reiner Pope、Sholto Douglas、Aakanksha Chowdhery、Jacob Devlin、James Bradbury、Anselm Levskaya、Jonathan Heek、Kefan Xiao、Shivani Agrawal、Jeff Dean（Google），2022-11-09。该文的解析模型用「给定划分下的算力/带宽权衡」做多维度划分决策，并给出 int8 权重下 PaLM 540B 在低 batch 时 29 ms/token、大 batch 处理输入时 76% MFU 的参考点。题面所需的具体常数（H100 989 TFLOPs / 3.35 TB/s、70B 权重 140 GB、295 拐点、320 KiB/token、三种配置的 tokens/s）该文未覆盖，均为按本文第 3 节公式自行计算；第 4、5 节的 KV cache 修正项、TP 的 all-reduce 开销与 60%–80% 有效利用率同样为自行推导与工程估计，非该文结论。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
