---
type: question
id: inference-serving-15
topic: 推理、服务与 GPU 性能
order: 15
question: 解释 disaggregated prefill/decode 服务，以及它在什么情况下才划算。
question_en: Explain disaggregated prefill/decode serving and when it pays off.
asked_at: [Moonshot AI, Groq]
level: 高阶
tags: [p-d-分离, distServe, goodput, 集群]
sources:
  - title: Prefill-Decode Disaggregation in LLM Inference
    url: https://outcomeschool.com/blog/prefill-decode-disaggregation
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving（延伸）
    url: https://arxiv.org/abs/2401.09670
    author: Zhong et al. (OSDI 2024)
    published: 2024-01-18
related: [inference-serving-01, inference-serving-14, inference-serving-07, inference-serving-09]
updated: 2026-09-28
---

## 一句话答案

> prefill 与 decode 的资源画像几乎相反：prefill 是算力受限、要一次性并行处理整段 prompt，最优并行策略偏向更激进的切分；decode 是显存带宽受限、每步只挪一个 token，既需要大量显存装 KV cache，又需要大 batch 才能摊薄权重读取。把两者放在同一批 GPU 上，prefill 会插队抬高别人的 ITL，而且并行策略与资源配比只能取一套折中，最终为了满足两个 SLO 而过度配置。
> 分离（disaggregation）把集群切成 prefill 池与 decode 池，prefill 池算完 prompt、产出首 token 与 KV cache，再把 KV cache 通过网络送到 decode 池继续生成。两个阶段于是能各自选型、各自按自己的 SLO 扩缩容，优化目标也从裸吞吐换成 goodput——满足 TTFT/TPOT 约束的最大请求速率。DistServe 在 90% 以上请求满足约束的条件下报告：最多可服务 7.4× 更多请求，或把 SLO 收紧 12.6×。
> 划算的前提是「TTFT 与 TPOT 同时被严格要求 + prompt 足够长 + 池间有 NVLink/IB 级互联 + 规模到多机多租户」；短上下文、慢网络、小流量或 SLO 宽松时，传输开销与运维复杂度会吃掉全部收益，此时单机的 continuous batching + chunked prefill + prefix caching 是更优解。

## 面试官在考什么

- 能不能把「为什么要分离」讲成资源需求冲突（算力 vs 带宽、并行策略最优解不同、SLO 指标不同），而不是背一张架构图。
- 是否知道分离引入的新成本项是 **KV cache 跨节点传输**，并能把它换算成 GB 与 Gbps，判断在给定互联下是否可接受。
- 是否理解 goodput 与 throughput 的区别，能否说出论文结论的实验条件（模型、输入输出长度、SLO attainment 门槛）。
- 能否给出「不划算」的判据：短上下文、低带宽、负载小、流量形状均匀、SLO 宽松、运维成本，而不是笼统地说「规模大就上」。
- 是否能把分离放进一条正确的优化顺序里：单机调度（continuous batching、chunked prefill、prefix caching）→ 并行策略 → 最后才是跨机分离。

常见错误答案：

- 只答「prefill 和 decode 分开跑更快」，说不出代价项，也答不出失效条件。
- 说分离后 TTFT 和 TPOT「都变好」，忽略了 KV 传输直接落在 TTFT 的关键路径上，互联差时反而更糟。
- 认为分离能省显存或省卡：两个池各自持有一份完整权重副本，显存不但没省，容量规划还多了一层。

## 原理与推导

### 1. 两个阶段的资源画像为什么冲突

| 维度 | prefill | decode |
| --- | --- | --- |
| 瓶颈 | 矩阵乘算力（compute-bound） | HBM 带宽与显存容量（memory-bound） |
| 每步处理的 token 数 | 整段 prompt，$S$ 个 | 1 个 |
| 算术强度 | $\approx S$ FLOP/byte | $\approx 1$（batch 1），随 batch 近似线性上升 |
| 决定哪个指标 | TTFT | TPOT / ITL |
| 并行策略偏好 | 更大的 TP（intra-op）缩短执行时间 | 吞吐靠 PP/复制线性扩展；只有 TPOT 很严时才必须 TP |
| 显存占用 | 激活 + 一份 KV cache | 权重 + 长期驻留、随并发与长度增长的 KV cache |
| batch 偏好 | 小 batch；单条长 prompt 就能打满算力，再加只会拉长延迟 | 大 batch；否则权重读取被浪费，吞吐上不去 |

算术强度可以手算出来。单序列 prefill 处理 $S$ 个 token 时，每个权重元素被复用 $S$ 次，所以强度约 $S$ FLOP/byte；decode 每步只处理 1 个 token，强度约等于 batch size $b$（还没算 KV cache 读取）。H100 的 bf16 稠密算力约 989 TFLOPs、HBM3 带宽 3.35 TB/s，拐点是 $989/3.35 \approx 295$ FLOP/byte；A100 80GB 是 $312/2.039 \approx 153$。于是单序列 prefill 在 $S$ 到几百个 token 时就跨过拐点变成算力受限，而 decode 要靠 batch 才能爬上去——DistServe 实测 13B 模型处理 512-token prompt 已让 A100 接近算力饱和，与 153 这个拐点量级一致。

共置的问题有两层：

1. **批内干扰**。一个 prefill 步骤远长于一个 decode 步骤，混在同一批里，decode 必须等 prefill 算完，ITL 出现尖刺；反过来 decode 占着槽位也会拉长 prefill。论文的测量结论是：往 decode 批里塞进一个 prefill，TTFT 与 TPOT 同时变差，prompt 越长越明显。
2. **资源与并行策略耦合**。共置时只能选一套并行方案和资源配比，而它必须按 TTFT、TPOT 里更严的那一个来配，另一个阶段就被迫过度配置。分离后每个阶段变成独立的排队系统：prefill 近似 M/D/1，平均 TTFT 为

$$\text{Avg\_TTFT} = D + \frac{RD^2}{2(1-RD)}$$

其中 $D$ 是执行时间、$R$ 是到达率，第二项是排队延迟。低到达率时第一项主导、intra-op 并行更划算；高到达率时第二项主导、inter-op（流水线）更好——这就是「同一套并行策略不可能同时最优」的定量说法。

### 2. chunked prefill 缓解了什么、没解决什么

chunked prefill 把长 prefill 切块，与 decode 步骤拼进同一批，避免单次长 prefill 长时间阻塞别人。它有两个内在代价：

- **分块重读 KV**。算第 $k$ 块时要用到前面所有块的 KV cache，把它们重新从 HBM 载入 SRAM。切成 $N$ 块的总读取量是 $N+(N-1)+\cdots+1 = N(N+1)/2 = O(N^2)$，而不分块是 $O(N)$：$N=4$ 是 2.5×，$N=8$ 是 4.5×，$N=16$ 是 8.5×。上下文越长、块越多，这份额外带宽越贵。
- **折中而非消除**。块太小，prefill 与 decode 抢算力、执行时间被拉长（TTFT 变差）；块大到接近饱和算力，就没剩多少槽位给 decode（增益消失）。本质上它是在 TTFT 与 TPOT 之间挪动痛苦，而不是让两者同时变好。

分离则把「阶段间干扰」从根上去掉：decode 池里根本不跑 prefill，长 prompt 不可能插它的队。单机调度与跨机分离不是替代关系，分离之后每个池内部照样要做 continuous batching 与分块。

### 3. 分离架构与一次请求的完整路径

四个部件：router（决定请求走哪个 prefill 实例、哪个 decode 实例）、prefill 池、decode 池、KV 传输通道。

```text
请求 → [router] ──选 prefill 实例 + 预留 decode 实例
                     │
                     ▼
             [prefill worker]  整段 prompt 并行计算
                     │         ├─ 产出首 token（立即回给用户，压低 TTFT）
                     │         └─ 产出 KV cache
                     ▼
        (KV cache 经 RDMA/NVLink/IB 传输，可与 prefill 计算重叠)
                     │
                     ▼
             [decode worker]   载入 KV cache，逐 token 生成
                     │
                     ▼
              流式返回用户（token 2, 3, 4, …）
```

工程上要点：

- **两个池各持一份完整权重**（按实例粒度）。分离消除的是干扰与错配，不是显存开销；同样的卡数下，可用 KV 显存反而更少。
- **选型可以不同**。prefill 池挑算力密、互联好的卡（TP 度开大），decode 池挑显存容量与带宽大的卡，甚至可以便宜一档；但两侧的权重量化格式与 KV 布局必须兼容。
- **放置是拓扑问题**。KV 传输量随上下文线性增长，placement 要尽量让配对的两个实例落在同一 NVLink 域/同一交换机域内，否则收益直接被网络吃掉。
- **KV 的切分方式要一致**。KV cache 在 TP 维度上按 KV 头切分；两侧 TP 度不同就必须重切分（reshuffle），这要求 KV 头数在两端都可整除。这是分离方案里最容易在实现阶段踩到的约束。

### 4. 新增成本：KV cache 跨节点传输

KV cache 的体量用 [[llm-internals-02]] 的公式算：

$$V = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot \text{bytes}$$

它随 $S$ 线性增长，长上下文时可达几十 GB 甚至几百 GB（MHA 大模型）：

| 模型 / 精度 | 每 token | 512 token | 8k | 32k | 128k |
| --- | --- | --- | --- | --- | --- |
| OPT-66B，MHA-72，fp16 | 2.25 MiB | 1.125 GiB（1.21 GB） | 18 GiB（19.3 GB） | 72 GiB（77.3 GB） | 288 GiB（309 GB） |
| LLaMA-2/3-70B，GQA-8，bf16 | 320 KiB | 160 MiB | 2.5 GiB（2.68 GB） | 10 GiB（10.7 GB） | 40 GiB（42.9 GB） |

OPT-66B 的原生上下文只有 2048，表里 8k 以上的三列是按公式做的线性外推，只用于标定 MHA 架构的 KV 体量量级。

把这些字节搬过一个网络要多久：

| 载荷 | 100 Gbps 以太网（12.5 GB/s） | 400 Gbps IB（50 GB/s） | NVLink 4 单向（450 GB/s） |
| --- | --- | --- | --- |
| 1.21 GB（512 token，OPT-66B） | 97 ms | 24 ms | 2.7 ms |
| 2.68 GB（8k，70B GQA） | 215 ms | 54 ms | 6.0 ms |
| 10.7 GB（32k，70B GQA） | 859 ms | 215 ms | 24 ms |
| 42.9 GB（128k，70B GQA） | 3.4 s | 859 ms | 95 ms |

论文给出的门槛也是同一量级：单条 512-token 的 OPT-66B 请求 KV cache 约 1.13 GB，在 10 rps 到达率下需要 11.3 GB/s ≈ 90 Gbps 才能让传输「隐形」，所以分离要靠 InfiniBand（论文提到 800 Gbps 级）或节点内 NVLink（论文给 A100 的 600 GB/s）。三个手段常一起用：

- **传输与计算重叠**：分块传输，prefill 算完一块就发一块，别等整段算完。
- **分层缓存**：把 KV 落到远端 DRAM/SSD 或专门的缓存池，下一轮请求直接复用，而不是每次重算重传。Moonshot AI 公开的 Mooncake 就是 KVCache-centric 的分池架构，把集群里闲置的 CPU/DRAM/SSD 组织成 KV 的存储层。
- **压缩后再传**：KV 量化到 fp8/int8 让传输量直接减半；只传 decode 后续真会读到的部分——例如滑窗注意力的窗口外 KV 不必传。

### 5. 收益：goodput 与独立扩缩容

分离的收益不是「吞吐变高」，而是 **goodput 变高**——在 SLO attainment 门槛（例如 90% 的请求满足约束）下，每张 GPU 能服务的最大请求速率。裸吞吐可以靠牺牲延迟堆上去，goodput 不能。

DistServe 的实验条件与结论（13B 模型，输入 512 / 输出 64，单张 A100 80GB，90% attainment）：

| 配置 | 每 GPU goodput |
| --- | --- |
| 现有共置系统 | 1.6 rps |
| 只跑 prefill | 5.6 rps |
| 只跑 decode | 10 rps |
| 2 prefill + 1 decode（3 卡合计 10 rps） | 3.3 rps，约为共置的 2.1× |

整体上，论文报告在多种模型、应用（chatbot / 编程助手 / 文档摘要）与延迟要求下，最多可服务 **7.4× 更多请求**，或在同等请求量下把 SLO **收紧 12.6×**，同时 >90% 的请求满足约束。

另一项收益是**容量比由流量形状决定**，可以按需配比。以 70B、TP=8、prompt 8k、输出 200、并发 32 为例（忽略通信开销，MFU 按 45% 估）：

| 项 | 计算 | 结果 |
| --- | --- | --- |
| prefill | $2 \cdot 70\text{e}9 \cdot 8192$ FLOPs ÷ (8 卡 × 989 TFLOPs × 0.45) | 322 ms × 8 卡 = **2.58 GPU·s/请求** |
| decode 单步 | 每卡读 17.5 GB 权重 + 32 × 8300 token × 40 KiB KV = 28.4 GB，÷ 3.35 TB/s | 8.5 ms/步 |
| decode 全程 | 8.5 ms × 200 步 × 8 卡 ÷ 32 并发 | **0.42 GPU·s/请求** |

两者相差约 6 倍：长 prompt、短输出的流量需要明显更大的 prefill 池。反过来，代码补全、思维链这类长输出短输入的流量要让 decode 池更大。这个比例只有分离之后才能独立调节；共置时只能整机一起扩，池间比例永远是最贵的那种流量说了算。

### 6. 什么情况下不划算

1. **上下文短或互联差**。传输开销是收益的抵扣项：上下文短，KV 小到不值得为它建一套传输与调度；互联差（跨可用区甚至公网，1 Gbps 级链路）时，8k prompt 的 2.68 GB 要 21 s 才传完，比重新算一遍 prefill 还慢。判据是把「KV 传输时间」与「prefill 执行时间」放在一起比：同域 NVLink/IB 是百分之几，跨区就是灾难。
2. **负载太小或形状均匀**。请求少、prompt 长度方差小的时候，本来就没有明显的 prefill 插队问题；分离只是把单进程变成两个池 + 一个 router，多出来的组件不产生任何收益。单机、单卡部署直接排除。
3. **SLO 宽松**。goodput 的收益来自「在严格 TTFT/TPOT 下少浪费算力」。如果 TTFT 有几秒余量、TPOT 只要快过阅读速度，单机的 continuous batching + chunked prefill（见 [[inference-serving-14]]）+ prefix caching 已经能满足，分离是过度设计。
4. **运维复杂度**。两个池的容量规划与配额、两套扩缩容信号、路由与重试语义、版本一致性、故障域都变成两份。最典型的是 decode worker 中途崩溃：它手上的 KV 丢失，请求只能重做 prefill，重试要能从 router 层重新走一遍流程。
5. **放置与路由错配**。没有拓扑感知的放置与在线调度，会出现「prefill 池空闲、decode 池排队」这种错配，实际 goodput 反而低于共置。分配方案需要按 workload 特征与 SLO 搜出来，不是拍脑袋定个比例。

决策规则可以压成一句话：**TTFT 与 TPOT 同时被严格要求、prompt 中位数在几千 token 以上、池间有 NVLink/IB 级互联、集群已经多机多租户**，这四条同时成立才值得上分离；缺任何一条，先把单机调度做满（continuous batching → chunked prefill → prefix caching → 并行策略），到瓶颈再考虑跨机分离。

## 数值与代码验证

以下数字均可复算；单位区分二进制（GiB = 1024³ B）与十进制（GB = 10⁹ B），与论文数字对照处显式标注口径。

```python
GiB = 1024 ** 3


def kv_bytes(layers, kv_heads, head_dim, seq_len, batch=1, elem=2):
    """V = 2 * L * H_kv * d_head * S * b * bytes"""
    return 2 * layers * kv_heads * head_dim * seq_len * batch * elem


# 1) 三个规模：DistServe 的 OPT-66B 算例 + 70B GQA-8
opt66 = kv_bytes(64, 72, 128, 512)          # MHA：H_kv = 72（OPT-66B 的注意力头数）
llama70 = kv_bytes(80, 8, 128, 1)           # GQA-8
print(opt66 / GiB, opt66 / 1e9)             # 1.125 GiB  1.208 GB（论文写 1.13GB）
print(llama70 / 1024, "KiB/token")          # 320.0
for s in (8_192, 32_768, 131_072):
    print(s, round(kv_bytes(80, 8, 128, s) / GiB, 2))   # 2.5 / 10.0 / 40.0


# 2) 传输时间（有效载荷速率，未计协议开销）
def xfer(byts, bits_per_s):
    return byts * 8 / bits_per_s * 1e3      # ms


for name, bw in [("100GbE", 100e9), ("400G IB", 400e9), ("NVLink4", 3600e9)]:
    print(name, round(xfer(kv_bytes(80, 8, 128, 8192), bw), 1), "ms")
    # 100GbE 214.7 / 400G IB 53.7 / NVLink4 6.0（450 GB/s 单向 = 3600 Gbit/s）
print([round(xfer(opt66, b), 1) for b in (100e9, 400e9, 3600e9)])   # 96.6 / 24.2 / 2.7 ms（OPT-66B 512 token）

# 3) roofline 拐点 = bf16 稠密算力 / HBM 带宽
print(round(989e12 / 3.35e12), round(312e12 / 2039e9))    # 295  (H100)  153 (A100 80G)

# 4) 池容量比：70B、TP=8、prompt 8k、输出 200、并发 32
flops = 2 * 70e9 * 8192
t_prefill = flops / (8 * 989e12 * 0.45)                   # 8 卡 MFU 45%
print(round(t_prefill * 1000), "ms", round(t_prefill * 8, 2), "GPU·s/请求")   # 322 ms 2.58

w_per_gpu = 70e9 * 2 / 8                                  # 17.5 GB
kv_per_gpu = 8_300 * (2 * 80 * 1 * 128 * 2)               # GQA-8 在 TP=8 下每卡 1 个 KV 头
t_step = (w_per_gpu + 32 * kv_per_gpu) / 3.35e12
print(round(t_step * 1000, 1), "ms/step", round(t_step * 200 * 8 / 32, 2), "GPU·s/请求")
# 8.5 ms/step  0.42 GPU·s/请求
```

与源文对照，三处口径差异要主动说明：

- **KV cache 体积**。论文给「OPT-66B 单条 512-token 请求约 1.13 GB」。按 OPT-66B 的真实配置（64 层、72 个注意力头、$d_{head}=128$）复算，$2 \times 64 \times 72 \times 128 \times 512 \times 2$ B = 1,207,959,552 B = 1.125 GiB = 1.21 GB；1.125 GiB 四舍五入正好是 1.13，可见论文这里是 GiB 口径而不是十进制 GB。后文的传输时间与容量估算一律用公式值，并同时标注 GiB 与十进制 GB 两种口径。
- **带宽门槛**。论文说 10 rps 下需要 11.3 GB/s ≈ 90 Gbps，这是把 1.13 当作十进制 GB 算出来的。用公式值复算是 12.1 GB/s ≈ 96.6 Gbps，比论文高约 7%，即论文这个门槛偏保守；两者同量级，不影响结论。
- **goodput 倍数**。7.4×（更多请求）与 12.6×（更紧 SLO）是论文在多种模型/应用/延迟要求下、>90% 请求满足约束的结果，不是任意配置下的保证；单点例子是 13B / 512 in / 64 out 的 1.6 → 3.3 rps/GPU（2.1×）。引用时必须带上这些条件。

第 5 节的池容量比是一阶估计：忽略了 TP 的 all-reduce 通信、KV 传输时间、调度开销，并把「8 卡 prefill 实例」与「8 卡 decode 实例」当作同构单元。它的用处是说明池间比例由 workload 决定（8k in / 200 out 时约 6:1），不是精确的容量规划结果。

## 常见追问

- **追问**：KV cache 具体怎么传？
  - 要点：走 RDMA（GPUDirect 一类）或 NVLink，避免经过主机内存拷贝；分块传输，prefill 算完一块发一块，用传输与计算重叠把延迟藏起来；KV 量化到 fp8/int8 让字节数减半；只传 decode 侧真正会读到的部分（滑窗注意力窗口外的 KV、已被淘汰掉的 token 的 KV 都不必传）；把 KV 落到分层缓存（远端 DRAM/SSD）让后续请求直接命中。
- **追问**：两个池分别用什么信号扩缩容？
  - 要点：prefill 池看 TTFT 的 p99 与 prefill 队列等待时间（排队延迟是主要成分）；decode 池看 TPOT 的 p99、running batch 是否上不去、KV cache 显存占用率（占用打满就意味着 batch 被显存卡住，要扩）。这两条曲线形状不同，所以共置系统的扩缩容信号必然是两难；分离后各自独立，扩缩容的判定也更干净。
- **追问**：分离和 prefix caching 会互相影响吗？
  - 要点：会。命中前缀缓存省掉的是 prefill 算力，但 KV 仍然要出现在 decode 实例的显存里，所以还要按命中长度把 KV 传过去——命中只省算力，不省带宽。另一种做法是把会话钉在同一个 decode 实例上、让 decode 侧自己保留 KV，多轮对话的第二轮就能完全不经过 prefill（代价是会话亲和性与显存驻留策略）。两种做法在真实系统里常混合使用。
- **追问**：既然有 chunked prefill，为什么还要分离？
  - 要点：chunked prefill 是单机内的调度手段，它把长 prefill 切碎后仍然和 decode 抢同一张卡，只能在 TTFT 与 TPOT 之间挪动折中，还引入 $O(N^2)$ 的 KV 重读；分离消除的是阶段间干扰本身，并让两个阶段各自选型与扩容。两者可以叠加：分离后的 prefill 池内部照样分块、decode 池内部照样 continuous batching。
- **追问**：分离之后 TTFT 会不会变差？
  - 要点：首 token 之前多了一次 KV 传输，TTFT 多出一项 $V/B$（$V$ 是 KV 字节数、$B$ 是有效带宽）；把传输与 prefill 计算重叠后，这一项可以大部分被隐藏。收益端是 decode 不再被长 prefill 插队，TPOT 的尾部明显改善。互联差时 $V/B$ 会超过收益，这正是「什么情况下不划算」的第一条。
- **追问**：什么规模才值得上分离？
  - 要点：经验门槛是多机、多租户、TTFT 与 TPOT 同时有硬约束、池间有 NVLink/IB 级互联、prompt 长度分布有明显长尾。单机 8 卡以内先做单机调度；只有当「同一批 GPU 上两个阶段的最优并行策略与资源配比明显不一致」成为主要瓶颈时，分离才开始划算。

## 公司变体

`asked_at` 的两家在公开材料里对这道题的侧重不同：

- **Moonshot AI**：长上下文在线服务方向，偏工程与容量规划。公开的 Mooncake 架构本身就是 KVCache-centric 的分离式设计——prefill 与 decode 分池，并把集群里闲置的 CPU/DRAM/SSD 组织成 KV 的分层缓存。因此更可能追问：长上下文下 KV 传输与落盘的分层策略、prefill 池与 decode 池的配比怎么定、缓存命中率如何影响 TTFT 与成本，而不是停留在架构名词上。
- **Groq**：自研加速器与确定性执行路线，公开材料强调编译期静态调度、片上 SRAM 与稳定的 TPOT。侧重点因此偏向「调度与硬件约束」：跨池搬运 KV 对确定性时延目标的影响、没有大容量 HBM 时 KV 如何驻留与分层，以及扩容时如何保持时延曲线不变。它不太会顺着「TP 度搜索」提问，而更可能从系统调度与容量规划切入。

以上是基于两家公开技术材料与岗位方向的侧重判断，具体题目以实际面试轮次为准。

## 相关题目

- [[inference-serving-01]]：prefill 与 decode 两阶段的定义，以及为什么一个算力受限、一个带宽受限——分离动机的起点。
- [[inference-serving-14]]：chunked prefill 如何在单机内缓解混合流量下的 tail latency，以及它为什么不能替代分离。
- [[inference-serving-07]]：tensor / pipeline / data 并行与 expert 并行的取舍，决定两个池各自的最优并行方案。
- [[inference-serving-09]]：TTFT、TPOT、ITL 与 throughput 之间的权衡，以及 goodput 这个目标函数是怎么来的。

## 参考资料与归属

1. [Prefill-Decode Disaggregation in LLM Inference](https://outcomeschool.com/blog/prefill-decode-disaggregation)，Amit Shekhar（Outcome School），2026-09-13（页面标注日期）。提供 router / prefill worker / decode worker / KV 传输四部件架构、一次请求的逐步流程、prefill 与 decode 的资源画像差异、共置与分离的对比表、「何时适用、何时过度设计」的清单，以及 decode worker 崩溃导致请求重做的故障分析。
2. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)（延伸），Yinmin Zhong 等（OSDI 2024），2024-01-18。提供 prefill-decoding 干扰与资源/并行耦合的论证、M/D/1 排队分析（式 1–3）、两个阶段的并行策略偏好、chunked prefill 的 $O(N^2)$ KV 重读、KV 传输量门槛（1.13 GB / 90 Gbps / A100 NVLink 600 GB/s）、placement 算法，以及 7.4× / 12.6× 与 Fig. 1 的 1.6 / 5.6 / 10 rps 数据。第 4、5 节的关键数字均出自该文，第 3 节的架构描述出自第 1 条来源。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
