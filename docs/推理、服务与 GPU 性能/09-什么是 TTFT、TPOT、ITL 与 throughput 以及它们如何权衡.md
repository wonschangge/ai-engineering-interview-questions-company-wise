---
type: question
id: inference-serving-09
topic: 推理、服务与 GPU 性能
order: 9
question: 什么是 TTFT、TPOT、ITL 和 throughput？它们之间如何相互权衡？
question_en: What are TTFT, TPOT, ITL and throughput, and how do they trade off against each other?
asked_at: [Microsoft, Apple, Perplexity]
level: 进阶
tags: [SLO, 延迟指标, goodput, 容量规划]
sources:
  - title: Prefill vs Decode: LLM Inference Optimization
    url: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: 
  - title: The First-Token Latency Problem in LLMs
    url: https://www.youtube.com/watch?v=XD8DD4cEHu0
    author: Outcome School
    published: 
  - title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving（延伸）
    url: https://arxiv.org/abs/2401.09670
    author: Zhong et al. (OSDI 2024)
    published: 2024-01-18
related: [inference-serving-01, inference-serving-14, inference-serving-13]
updated: 2026-09-28
---

## 一句话答案

> TTFT 是第一个 token 什么时候到（排队 + prefill + 采样），TPOT 是首 token 之后平均每个 token 的耗时（decode 决定），ITL 是同一个量的逐 token 间隔**分布**——TPOT 是它的均值，p99 才对应「卡不卡」；throughput 是整机每秒产出的 token 数或请求数。
> 四者由 $E2E \approx TTFT + TPOT \times (O-1)$ 串起来：10 token 的短输出里 TTFT 占 E2E 的 77%，2000 token 的长文里只占 1.6%，所以优化目标必须跟着长度分布走。
> 取舍的根子是 prefill 吃算力、decode 吃显存带宽：batch 变大只摊薄权重那一项，吞吐先在中小 batch 近似线性上涨、之后趋于饱和；每多一条并发序列就多读一份 KV，TPOT 以固定斜率变长。验收要用 goodput——满足 TTFT/TPOT 约束的那部分吞吐，超 SLO 的吞吐是无效吞吐。

## 面试官在考什么

- 能否把四个缩写落到算式与口径上：TTFT 里含排队，TPOT 的分母是 $O-1$，ITL 是分布而不是均值，throughput 要区分「输入 + 输出 token」与「仅输出 token」。
- 是否记得指标 → 相位 → 硬件瓶颈的三角对应（TTFT→prefill→算力，TPOT/ITL→decode→带宽），这是后面全部取舍的唯一依据，记住它就不需要背结论。
- 会不会做容量规划：由 TTFT/TPOT 的 SLO 反推并发上限与 batch 上限，并说清是哪一条约束先绑住。
- 有没有 goodput 意识：吞吐只有落在 SLO 之内才是有效产能，超出 SLO 的那部分吞吐在验收里等于零。
- 压测与观测的工程素养：分位数、真实长度分布、冷启动、流式 chunk 合并对 ITL 测量的污染。

常见错误答案：

- 把 TPOT 与 ITL 讲成两个并列指标，或只报平均 TPOT 就宣称延迟达标。两者的关系是均值与分布，而 SLA 一般写在 p99 上。
- 只答「batch 越大吞吐越高、延迟越差」，给不出斜率。decode 每多一条并发序列的边际代价是 $\frac{2 L H_{kv} d_{head} S \cdot \text{bytes}}{BW}$，是常数；被摊薄的是权重那一项。以 LLaMA-3-70B、4k 上下文、8×H100 计，这条斜率是 0.050 ms/step。
- 把吞吐与延迟当成只能二选一，忽略 chunked prefill、prefix caching、KV 量化、speculative decoding 这些能整体平移 Pareto 前沿的手段。

## 原理与推导

### 1. 四个指标的定义与口径

| 指标 | 定义 | 归属相位 / 资源 | 常用单位 | 谁最在意 |
| --- | --- | --- | --- | --- |
| TTFT | 请求发出到第一个输出 token 到达 | 排队 + prefill + 首 token 采样（算力） | ms | 交互式对话、检索问答 |
| TPOT | 首 token 之后平均每个 token 的耗时 | decode（显存带宽） | ms/token | 长输出、流式朗读 |
| ITL | 相邻 token 到达间隔 $t_i - t_{i-1}$ 的分布 | decode + 调度 | ms，报 p50/p95/p99 | 「卡不卡」的体感、SLA |
| throughput | 单位时间整机产出的 token 数或请求数 | batch × 每步耗时 | tok/s、req/s | 成本、容量、采购 |

$$TTFT = t_{queue} + t_{prefill} + t_{sample}, \qquad TPOT = \frac{E2E - TTFT}{O-1} = \frac{1}{O-1}\sum_{i=2}^{O} ITL_i$$

$$E2E = TTFT + TPOT \times (O - 1)$$

四处容易踩的口径：

- **TPOT 的分母**。上面这版不含首 token（只统计首 token 之后的 $O-1$ 个间隔），是多数服务框架与压测工具的常见口径；另有 $\text{TPOT}' = E2E/O$ 的写法，把首 token 也算进平均，比上式大 $\text{TTFT}/O$。两者都能用，报数时必须写明是哪一种。
- **throughput 统计的是哪种 token**。把 prefill token 也算进去时，长 prompt 负载的数字会虚高：真正稀缺的资源是 decode 的 slot，而 prefill token 与输出 token 不是同一种产能。
- **ITL 的量测**。服务端把 $n$ 个 token 合并进一个 SSE chunk 时，量到的间隔是真实间隔的 $n$ 倍，分位数会整体好看；用网关侧测量时必须先确认这一点。
- **throughput 的粒度**。要同时报单请求级与系统级：单流吞吐就是 $1/\text{TPOT}$（TPOT 25 ms 即 40 tok/s），系统吞吐约为 $b/\text{TPOT}$，因为 batch 内各流共享同一份权重读取。第 4 节把这两列并排放，混用口径会把同一个系统的吞吐说成相差上百倍。

### 2. 三个派生量：E2E、goodput、cost per 1M tokens

$E2E$ 的拆解直接决定优化方向。取三组典型数字：

| 负载 | TTFT | 输出长度 $O$ | TPOT | E2E | TTFT 占比 |
| --- | --- | --- | --- | --- | --- |
| 分类 / JSON 输出 | 300 ms | 10 | 10 ms | 0.39 s | 76.9% |
| 对话 | 400 ms | 200 | 25 ms | 5.38 s | 7.4% |
| 长文生成 | 500 ms | 2000 | 15 ms | 30.5 s | 1.6% |

结论是硬性的：$O \lesssim 30$ 时 TTFT 占 E2E 的一半以上，主导体验；$O = 2000$ 时把 TTFT 从 500 ms 优化到 300 ms 只省掉 E2E 的 0.66%，而同样这 200 ms 在 $O = 500$ 时是 2.5%。

还有一层不能忽略：$E2E$ 对 $O$ 是线性的，而线上流量的 $O$ 往往横跨两个数量级（分类 10 token、长文 2000 token），把两种流量混在一起算平均 E2E，得到的是一个在真实分布里并不存在的请求。均值会被长尾样本拖着走，所以必须按输出长度分桶后再看 p95/p99；TPOT 的 p99 同理，只在同一长度桶内才可比。

**goodput** 是「在满足 TTFT 与 TPOT 约束前提下」的吞吐。DistServe 把它定义为每张 GPU 上同时满足 TTFT 与 TPOT 约束的最大请求速率，并用它替代吞吐做验收：在多种模型、应用与延迟要求下，相比当时的最优服务系统可以服务 7.4 倍的请求数、或把 SLO 收紧 12.6 倍，同时让 90% 以上的请求落在延迟约束内。**cost per 1M tokens** 则由 goodput 直接推出：

$$\text{cost per 1M} = \frac{\text{节点每小时成本}}{\text{goodput (tok/s)} \times 3600} \times 10^{6}$$

### 3. 为什么必然互相拉扯：prefill 吃算力，decode 吃带宽

用算术强度（FLOP/Byte）把两个相位放到同一把尺子上。前向每 token 的算力是 $2N$ 加注意力项；因果掩码下 prefill 的平均 token 只看一半的 key，是 $2Ld_{model}P$，而 decode 每步要看全部 $S$ 个 key，是 $4Ld_{model}S$。

$$\text{INT}_{\text{prefill}} = \frac{\left(2N + 2Ld_{model}P\right) \cdot P}{W}, \qquad \text{INT}_{\text{decode}} = \frac{b\left(2N + 4Ld_{model}S\right)}{W + b\,kS}, \qquad k = 2LH_{kv}d_{head}\cdot\text{bytes}$$

硬件平衡点 = 峰值算力 / 带宽。H100 SXM5 是 $989/3.35 = 295$ FLOP/Byte；把 40% MFU 打进去的有效平衡点是 $3.165\,\text{PFLOP/s} \div 26.8\,\text{TB/s} = 118$ FLOP/Byte（8 卡节点口径）。40% 是端到端 MFU 的保守取值：源文说的「大 prompt 下数学单元忙到 90% 以上」是算力单元的瞬时占用率，kernel 间隙、集合通信与 attention 的读写还要再吃掉一部分。

| 场景 | 每 token FLOPs | 每 token 字节 | 算术强度 | 判定 |
| --- | --- | --- | --- | --- |
| prefill，$P=4096$ | 145 GFLOP | 34.2 MB（权重按 $P$ 摊） | 4,253 FLOP/B | 远在平衡点之上，算力受限 |
| decode，$b=48$，$S=4096$ | 151 GFLOP | 4.26 GB | 35 FLOP/B | 远在平衡点之下，带宽受限 |
| decode，$b=384$，$S=4096$ | 151 GFLOP | 1.71 GB | 88 FLOP/B | 仍在平衡点之下 |

decode 即使把并发堆到显存允许的上限（384 条，见第 5 节），算术强度也只有 88 FLOP/B，没有翻到算力侧；而 prefill 高出平衡点 36 倍。两者要的资源不同，这就是「提高并发几乎不影响总吞吐成本、却必然拉长每条序列的 TPOT」以及「长 prefill 会打断别人的流式输出」的共同来源。

### 4. batch：吞吐的来源，TPOT 的代价

一步 decode 必须把全部权重和 batch 内全部 KV 从 HBM 读一遍，所以

$$t_{step}(b, S) \approx \frac{W + b\,kS}{BW}, \qquad TPOT(b) \approx t_{step}, \qquad \Theta(b) = \frac{b}{t_{step}} = \frac{b \cdot BW}{W + b\,kS} \xrightarrow{b \to \infty} \frac{BW}{kS}$$

分母里 $W$ 被 $b$ 摊薄，$b\,kS$ 不被摊薄，于是吞吐先线性涨、再拐弯趋向 $BW/(kS)$，而 TPOT 的边际成本恒定。

| $b$ | 每步字节 | TPOT $=t_{step}$ | 单流 tok/s | 系统 tok/s |
| --- | --- | --- | --- | --- |
| 1 | 141.3 GB | 5.27 ms | 190 | 190 |
| 8 | 150.7 GB | 5.62 ms | 178 | 1,422 |
| 32 | 182.9 GB | 6.83 ms | 146 | 4,688 |
| 64 | 225.9 GB | 8.43 ms | 119 | 7,593 |
| 128 | 311.8 GB | 11.63 ms | 86 | 11,002 |
| 256 | 483.6 GB | 18.04 ms | 55 | 14,187 |
| 384 | 655.4 GB | 24.46 ms | 41 | 15,702 |

（LLaMA-3-70B、bf16、8×H100、TP=8、$S=4096$，$k=320$ KiB/token 为全模型口径，下表同。batch 内每条序列每步推进一个 token，所以这一步的耗时就是该批所有请求的 ITL，也就是 TPOT：$b=1$ 时 5.27 ms（190 tok/s），$b=384$ 时 24.46 ms（41 tok/s），TPOT 随 $b$ 以固定斜率（$kS$ 字节 ÷ 带宽 = 0.050 ms/条）上升，系统吞吐涨了 82 倍。）

### 5. 上下文长度：并发上限与吞吐一起下降

KV 显存决定并发上限 $b_{\max}(S) \approx M_{KV}/(kS/\text{TP})$。每卡留 60 GiB 给 KV（80 GiB 减去 16.3 GiB 权重与约 3 GiB 激活/工作区）时：

| 上下文 $S$ | $b_{\max}$ | $t_{step}$ | 单流 tok/s | 系统 tok/s |
| --- | --- | --- | --- | --- |
| 4k | 384 | 24.46 ms | 41 | 15,702 |
| 8k | 192 | 24.46 ms | 41 | 7,851 |
| 16k | 96 | 24.46 ms | 41 | 3,926 |
| 32k | 48 | 24.46 ms | 41 | 1,963 |

这张表有一条容易被忽略的规律：**显存被 KV 占满时，每步要读的 KV 总量等于 KV 显存容量本身，与上下文长度无关**，所以 $t_{step}$ 全是 24.46 ms；但每步产出的 token 数等于并发数，而并发数 $\propto 1/S$，于是吞吐 $\propto 1/S$。长文档问答的吞吐只有短对话的 1/8，成本却是 8 倍，这个结论不需要任何实测就成立。

同一条回答内部也会变慢：固定 $b=48$，上下文从 4k 长到 32k 的过程中 ITL 从 7.63 ms 涨到 24.46 ms（3.2 倍）。这是 ITL 分布天然右偏的机制之一——**同一段回答的后半段比前半段慢**。

### 6. prefill 插队：ITL 尾延迟尖刺

colocate 部署下，一个长 prefill 会和一堆 decode 排进同一个 batch。若它作为不可分割的一步执行，这一步的耗时由算力决定，同批所有 decode 请求都要等它（下表按「算力与带宽可在同一批内重叠、该步耗时取两者较大值」估算，$b=48$、$S=4096$ 的基线是 7.63 ms；算力时间只算了 chunk 自身的 $2NP$ 与 chunk 内部注意力，没算它回看已缓存上下文的开销，所以表中的台阶是偏乐观的下限）：

| prefill chunk | 该 chunk 算力时间 | 同批 decode 显存时间 | 该步实际耗时 |
| --- | --- | --- | --- |
| 256 token | 11.35 ms | 7.63 ms | ≈ 11.4 ms |
| 512 token | 22.76 ms | 7.63 ms | ≈ 22.8 ms |
| 2048 token | 92.33 ms | 7.63 ms | ≈ 92.3 ms |

不做 chunked prefill、按整段 prompt 一步算完时，$P=4096$ 的请求会让同批所有请求的 ITL 出现一个 188 ms 的台阶；换成 2048 token 的 chunk 是 92 ms，512 token 是 23 ms。**chunk 大小直接决定 p99 ITL 的下限**，代价是长 prompt 自己要等更多步（TTFT 略增）。这正是 chunked prefill 用「一点点 TTFT」换「ITL 平稳」的量化形式。

### 7. 由 SLO 反推容量，并用 goodput 验收

给定 $S=4096$、TTFT ≤ 1 s、TPOT ≤ 50 ms：

- 由 TPOT 反推：$(0.050 \times 26.8\,\text{TB/s} - W)/(kS) = 894$ 条；
- 由显存反推：$b_{\max} = 384$ 条；
- 取 $\min$ 得 $b^* = 384$，此时 TPOT 24.46 ms 达标，goodput 15,702 tok/s。

把 SLO 收紧到 TPOT ≤ 20 ms，带宽允许 295 条、显存允许 384 条，$b^* = 295$，goodput 14,752 tok/s——**此时多放 89 条并发，原始吞吐从 14,752 涨到 15,702 tok/s（+6%），超 SLO 的那 89 条全部变成无效产能**。这就是 goodput 与 throughput 的分野：约束宽松时两者接近，约束一紧，吞吐曲线还在涨而 goodput 已经掉头。

请求级的 goodput 还要再除一次输出长度：15,702 tok/s ÷ 200 = 78 req/s；同样一台机器服务 2000 token 的长文，只剩 7.8 req/s，即使 token 级吞吐相同。

### 8. TP 度：降低延迟地板，增加通信

batch=1 时每步只读权重，TP 度决定地板（忽略 KV 与通信）：

| TP | 权重/卡 | 每步地板 | 单流上界 |
| --- | --- | --- | --- |
| 2 | 70.0 GB | 20.90 ms | 48 tok/s |
| 4 | 35.0 GB | 10.45 ms | 96 tok/s |
| 8 | 17.5 GB | 5.22 ms | 191 tok/s |

TP 提高会把单请求延迟地板按 $1/\text{TP}$ 压低，但每步的集合通信次数不变（Megatron 式 TP 每层前向约 2 次 all-reduce，80 层即 160 次）：按每次 5 µs 估算约 0.8 ms，在 5.22 ms 的地板上占 15%；这是乐观假设，把通信延迟算得更保守时这一项会吃掉单流上界的 20% 以上。关键在于这笔开销在 $b=1$ 时完全无法摊薄，所以 TP 度换来的是单流延迟，不是聚合吞吐——同一条序列无论切给几张卡，要读的权重和 KV 总量不变。

## 数值与代码验证

上面的表格由这段代码复算，常数口径写死在开头：

```python
N, L, D_MODEL, H_KV, D_HEAD, BYTES = 70e9, 80, 8192, 8, 128, 2   # LLaMA-3-70B, bf16
BW, BF16_PEAK, MFU, NODE = 3.35e12, 989e12, 0.4, 8               # H100 SXM5 稠密算力
W = BYTES * N                    # 140.0 GB
K = 2 * L * H_KV * D_HEAD * BYTES  # 327680 B = 320 KiB/token（全模型口径）
EFF, BWN = BF16_PEAK * MFU * NODE, BW * NODE                     # 3.165 PFLOP/s, 26.8 TB/s

step_ms = lambda b, S: (W + b * K * S) / BWN * 1e3
for b in (1, 8, 32, 64, 128, 256, 384):                          # 第 4 节
    t = step_ms(b, 4096)
    print(b, f"{t:6.2f} ms", f"{1000/t:6.1f} tok/s/流", f"{b/t*1000:8.0f} tok/s 系统")

KV_BUDGET = 60 * 1024**3         # 每卡留给 KV 的显存
for S in (4096, 8192, 16384, 32768):                             # 第 5 节
    b = int(KV_BUDGET / (K / NODE * S))
    print(S, b, f"{step_ms(b, S):6.2f} ms", f"{b/step_ms(b,S)*1000:8.0f} tok/s 系统")

prefill_flops = lambda P: 2 * N * P + 2 * L * D_MODEL * P * P     # 因果注意力
for P in (256, 512, 1024, 2048, 4096, 8192, 32768):              # 第 3、6 节
    print(P, f"{prefill_flops(P)/EFF*1e3:8.1f} ms @MFU0.4")      # 11.4 22.8 45.7 92.3 188.1 390.2 1894.2

# 校验：算术强度与硬件平衡点
print(BF16_PEAK / BW, EFF / BWN)                                  # 295.2  118.1
print(prefill_flops(4096) / 4096 / (W / 4096))                    # 4253 FLOP/B
print((2*N + 4*L*D_MODEL*4096) / ((W + 48*K*4096) / 48))          # 35 FLOP/B  @b=48
```

与源文对照：源文给的是「TTFT 主要由 prefill 决定、TPOT 由 decode 决定」的定性映射，以及 $E2E = TTFT + (O-1)\times TPOT$ 与 400 ms + 199 × 25 ms ≈ 5.4 s 的算例，本节的 5.38 s 与它一致（源文明确把首 token 计入 TTFT、TPOT 只覆盖其后的 $O-1$ 个 token，与本节口径相同）。源文说单请求 decode 时 GPU 数学单元利用率约 20%–40%、prefill 在大 prompt 下可达 90% 以上，本节用算术强度给出了同一结论的另一种算法：decode 在 $b=48$ 时是 35 FLOP/B，相对 118 FLOP/Byte 的有效平衡点意味着算力侧大量闲置。第 5 节的「吞吐 $\propto 1/S$」、第 6 节的 chunk 尖刺表、第 7 节的 goodput 反推都超出了源文范围，是按同一套常数自行推导的。

## 常见追问

- **追问**：为什么不只报 TPOT，还要看 ITL？
  - 要点：TPOT 是均值，均值掩盖长尾。用户感知的是连续两个 token 之间的停顿：均值 25 ms、p99 200 ms 时，一段 1000 token 的回答里会出现约 10 次 200 ms 的卡顿，体感与「稳定 25 ms」完全不同。SLA 通常就写在 p99 ITL 上，所以观测系统必须落分布。补齐口径才是完整答案：均值看产能，p99 看体验。
- **追问**：p99 ITL 与均值差一个数量级，钱花在哪了？
  - 要点：三个机制叠加。① 调度抖动：batch 组成每步都在变，新请求加入、老请求结束，同一请求在不同步里分摊到的 KV 读取量不同。② prefill 插队：一条 4k prompt 不切块会让同批所有请求多等 188 ms（第 6 节）。③ 自身上下文增长：同一条回答从 4k 长到 32k 时 ITL 涨 3.2 倍（第 5 节）。再叠加网络与流式 chunk 合并的测量噪声。定位顺序是先看 ITL 分位数的时间序列，再看同一时刻的 batch 组成与 prefill 队列。
- **追问**：不加卡，怎么同时改善 TTFT 和 TPOT？
  - 要点：先分清能「平移前沿」和只能「换位置」的手段。平移的有：prefix caching（省掉重复 prefill，直接砍 TTFT）、KV 量化（$k$ 减半，同时降显存占用与每步读取量，TPOT 与并发上限一起改善）、speculative decoding（一次校验的内存流量与生成 1 个 token 相当，接受率高时 TPOT 可降到 1/2–1/3，接受率低时收益归零）、更好的 attention kernel。只能换位置的有：batch 大小、chunk 大小、TP 度。真正同时恶化两者的只有「上下文变长」这一类工作负载变化。
- **追问**：为什么常见的 TPOT 门槛落在 20–50 ms/token？
  - 要点：不是「跟上阅读」——人类阅读约 250 词/分，折合约 5.6 token/s，而 25 ms/token 是 40 token/s，约为阅读速度的 7 倍，50 ms/token 也仍有 3.6 倍。门槛来自另外两件事：长回答的 E2E（2000 token 的答案，25 ms/token 要 50 s，50 ms/token 要 100 s），以及停顿感（工程上把单次超过约 100 ms 的间隔当作会被明确感知为停顿的经验阈值，所以 p99 比均值更值得守）。因此门槛应该由输出长度分布和感知阈值共同定，而不是照抄一个数字。
- **追问**：压测怎么设计才算数？
  - 要点：① 固定输入/输出长度分布（真实分布，不是固定 128/128），固定解码参数；② 用开环到达（泊松）压 SLO，用闭环固定并发测延迟曲线，两者结论不同，不要混用；③ 分别扫并发而不是扫 batch——batch 由调度器决定，你控制不了；④ 记录 TTFT 与 ITL 的分位数曲线，而不是一个平均吞吐数字；⑤ 区分冷启动（首次编译、CUDA graph 捕获、cache 为空）与稳态；⑥ 长跑至少几分钟，覆盖调度器的稳态行为。
- **追问**：goodput 怎么定义、怎么验收？
  - 要点：口径要与 SLO 绑定——「在 TTFT ≤ X 且 TPOT ≤ Y 的前提下，每张 GPU 每秒完成多少请求」，统计时只计满足约束的请求（或按满足比例折算）。验收时同时给出达标率（如 90% 以上请求落在约束内），否则可以通过牺牲尾部来刷指标。容量规划时用 goodput 反推并发上限，用 cost per 1M tokens 汇报单位成本。

## 公司变体

`asked_at` 里的三家在公开技术材料与岗位方向上侧重不同：

- **Microsoft**：平台与服务视角，偏工程。这条线的公开工作（Azure/DeepSpeed 系）关心的是「给定 SLO，每张卡能服务多少请求」，Microsoft Research 的 Splitwise 与 DistServe 同期提出了按相位拆机的容量规划。所以这道题在微软语境里通常落到 goodput、容量规划、多租户配额与成本上，公式推导只作为结论的依据。
- **Apple**：端侧与私有云计算（Private Cloud Compute）视角，偏「受限硬件下的取舍」。显存预算小、batch 小、几乎没有 TP 空间，于是 TTFT 更容易被模型加载与预热支配、TPOT 更容易被带宽支配，讨论常落在「多小的模型、多低的比特能同时守住 TTFT 与 TPOT」，以及固定硬件池下的排队策略。
- **Perplexity**：检索问答的产品侧，偏工程与体验。prompt 里要塞检索到的长文档，TTFT 直接决定「搜索感觉快不快」，因此 prefix caching、prompt 长度控制、p99 是高频话题，通常要求候选人从真实的长度分布出发给容量数字。

以上是基于公开技术材料与岗位方向的侧重判断，具体题目以实际面试轮次为准。

## 相关题目

- [[inference-serving-01]]：prefill 与 decode 为什么一个是计算受限、一个是带宽受限。本节第 3 节把它变成了可计算的算术强度对比。
- [[inference-serving-14]]：chunked prefill 如何改善混合流量下的尾延迟。第 6 节的 chunk 尖刺表就是它的量化动机。
- [[inference-serving-13]]：p99 latency 翻倍的排查过程。第 4–6 节列出的机制（batch 组成、上下文增长、prefill 插队）就是那份排查清单的候选原因。

## 参考资料与归属

1. [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)，Amit Shekhar（Outcome School）。提供 prefill/decode 的相位划分与瓶颈归属、TTFT/TPOT/throughput/端到端延迟的定义、$E2E = TTFT + (O-1)\times TPOT$ 与 400 ms + 199 × 25 ms ≈ 5.4 s 的算例、batch 变大同时抬高吞吐与延迟的定性结论，以及 chunked prefill、prefix caching、disaggregation、speculative decoding 的手段地图。
2. [The First-Token Latency Problem in LLMs](https://www.youtube.com/watch?v=XD8DD4cEHu0)，Outcome School。该视频页面无法抓取正文，第 1 节把 TTFT 拆成「排队 + prefill + 首 token 采样」时参考了它的主题（首 token 延迟是交互体验的瓶颈），未使用其中的任何具体数字。
3. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)（延伸），Zhong et al.，OSDI 2024，2024-01-18。第 2 节的 goodput 定义（每张 GPU 上同时满足 TTFT 与 TPOT 约束的最大请求速率）、7.4 倍请求数 / 12.6 倍 SLO 收紧 / 90% 以上请求满足约束这三个数字（口径是与当时最优服务系统对比、多种模型与应用下的最好结果），以及「prefill 与 decode 互相干扰、资源分配被耦合」的问题陈述取自该文摘要；第 7 节由 SLO 反推 $b^*$ 的算例是按其口径、用同一套常数复算的。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
