---
type: question
id: inference-serving-13
topic: 推理、服务与 GPU 性能
order: 13
question: 一次没有改动模型的部署之后，你的 p99 latency 翻了一倍。请讲一遍排查过程。
question_en: After a deployment that did not change the model, your p99 latency doubled. Walk through the debugging process.
asked_at: [OpenAI, Amazon, Databricks, Perplexity]
level: 高阶
tags: [排障, 尾延迟, 可观测性, 容量]
sources:
  - title: LLM Inference Optimization
    url: https://outcomeschool.com/blog/llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Mastering LLM Techniques：Inference Optimization（延伸）
    url: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
    author: Shashank Verma, Neal Vaidya (NVIDIA Technical Blog)
    published: 2023-11-17
related: [inference-serving-09, inference-serving-14, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> p99 翻倍几乎从不是「模型变慢」，而是**队列、缓存、流量结构、配置**这四类里有一类变了。排查顺序固定：先把「翻倍」定义清楚（哪个指标、哪一段、哪个子集、什么时候开始），再把请求路径切段（网关 → 排队 → prefill → decode → 流式返回），用敏感度系数 $\partial L/\partial TTFT = 1$、$\partial L/\partial ITL = N_{out}$ 决定优先看哪一段，然后对着变更清单二分到单个变量。
> 「模型没改」不等于「什么都没改」：batch 上限、显存水位、block size、prefix cache 的 hash 规则、镜像里的 kernel 版本，任何一项动了都足以让 p99 翻倍。
> 为什么不是小波动：p99 由尾部 1% 定义，而排队延迟在利用率附近的放大是超线性的——M/M/1 下利用率从 0.8 升到 0.9，p99 排队时间就从 21.9 个服务时间涨到 45.0 个，正好 2.05 倍。

## 面试官在考什么

- **分位数意识**：知不知道 p99 是「尾部 1% 的系统」，与均值无关；知不知道几千样本窗口里 p99 抖动 20%–30% 是统计噪声，先要确认「翻倍」是真的。
- **分层能力**：能不能把一条请求拆成可独立测量的段，而不是在「服务慢」这个粒度上猜。LLM 服务至少要拆成 排队 / prefill(TTFT) / decode(ITL) / 通信 / 客户端 五段。
- **假设的质量**：能不能给出按先验概率排序、且每条都能被一个具体观测证伪的假设清单，而不是「可能是网络问题」。
- **二分与回滚的纪律**：会不会用同流量影子、金丝雀、同输入重放、按实例/region/租户分组对比把变量收敛，一次只动一个变量。
- **工程收尾**：修复之后能不能把它变成版本化配置、SLO 告警和发布前回归压测，而不是「重启一下好了」。

常见错误答案：

- 「GPU 利用率不高，所以不是容量问题」「模型没改，所以一定是网络」——用结论代替分层，直接跳到修复。
- 一上来调 `max_num_seqs`、调显存水位、加副本、重启实例；或者只盯着均值、把不同实例的 p99 再平均一次。
- 把客户端测得的 E2E 与服务端 TTFT 当作同一个指标，于是永远分不清瓶颈在边界内还是边界外。

## 原理与推导

### 1. 第 0 步：先把「翻倍」定义清楚

p99 不是一个能直接比较的数，它由指标口径、聚合维度、时间窗和样本量共同决定。四个问题必须先回答：

1. **哪个指标？** E2E、TTFT、ITL 是三个不同的随机变量。E2E 的近似分解是

$$L_{E2E} \approx TTFT + N_{out} \cdot ITL$$

对两边求偏导得到**敏感度系数**：$\partial L_{E2E}/\partial TTFT = 1$，$\partial L_{E2E}/\partial ITL = N_{out}$。含义很直接——TTFT 每多 1 ms，E2E 就多 1 ms；ITL 每 token 多 1 ms，E2E 要多 $N_{out}$ ms。输出 250 token 的聊天式负载，ITL 从 20 ms 涨到 40 ms，E2E 就从 5.37 s 变成 10.37 s（1.93 倍）；而同样负载下 prefill 时间翻倍只让 E2E 涨 1.7%。**先看敏感度，再决定先查哪一段**，这一步能省掉一半的排查时间。（更严格的口径是 $L_{E2E} = TTFT + (N_{out}-1)\cdot ITL$，这里用 $N_{out}$ 是便于口算的上界，误差一个 ITL，对下文所有倍数的影响不超过 0.01。）

2. **哪个子集？** 按 model / region / instance / tenant / endpoint / 输入长度桶分组对比 p99。全部实例一起涨说明是全局变量（配置、缓存、流量）；只有部分实例涨说明是拓扑、混部或数据面问题。

3. **什么时候开始？** 把 p99 曲线和发布记录、配置变更、自动扩缩容事件、上游 QPS 曲线画在同一条时间轴上。延迟是阶跃还是缓慢抬升，直接区分「变更引入」与「容量被慢慢吃掉」。

4. **样本够不够？** 窗口内 $n$ 个样本中，落在真实 p99 以上的样本数服从 $\mathrm{Binomial}(n, 0.01)$，其相对标准差是 $1/\sqrt{0.01n}$：$n=1000$ 时 32%，$n=10^4$ 时 10%，$n=10^5$ 时 3.2%。流量低的端点上「p99 翻倍」很可能只是噪声，先确认它不是。

### 2. 第 1 步：切段，找出增长在哪一段

| 段 | 要看的指标 | 典型工具/来源 | 增长时的判据 |
| --- | --- | --- | --- |
| 客户端与边缘 | 客户端 E2E、连接复用率、DNS/TLS 耗时、重试次数 | 客户端埋点、RUM | 服务端 TTFT/ITL 不变，只有客户端 E2E 涨 |
| 网关/鉴权/限流 | 排队请求数、429/503 比例、LB 队列深度 | 网关指标、LB 日志 | 网关排队时间而非 GPU 时间上涨 |
| 调度排队 | 排队等待时长、waiting/running 序列数、实际 batch size | 引擎自带的调度指标（各引擎都暴露 waiting 与 running 口径） | waiting 队列变长而每步耗时不变 |
| 前处理 | tokenizer / 多模态预处理 P99、CPU 利用率 | 服务指标、CPU profile | CPU 打满、GPU 空转 |
| prefill（TTFT） | TTFT 分位、输入长度分布、prefix cache 命中率 | 服务指标、引擎日志 | TTFT 涨而 ITL 不变 |
| decode（ITL） | ITL 分位、KV cache 使用率与**驱逐率**、显存水位、SM occupancy / MFU | 引擎指标、DCGM、Nsight Systems | ITL 涨且 batch 变小或显存接近水位 |
| 并行通信 | TP/EP all-reduce 耗时、NVLink/PCIe 流量、NUMA 绑定 | NCCL 日志、Nsight Systems | 计算 kernel 变短但每步总时长变长 |
| 流式返回 | chunk 间隔、首字节到末字节、网络 RTT | 网关/客户端 | 服务端 token 间隔正常，客户端间隔异常 |

切完段只要回答一个问题：**是排队变慢了，还是计算变慢了？** 判据是三组指标的组合：

- waiting 队列/并发变长，而每步 GPU 耗时不变 → 排队问题（容量、配置、流量）。
- 每步耗时变长，且 GPU 利用率/MFU 同时升高 → 单位请求变贵了（重算、长请求、kernel 回退）。
- 每步耗时变长，但 GPU 利用率反而下降 → 卡在通信、CPU 或显存（换页/驱逐/重分配）。

这一刀决定了后面所有方向，所以要先做。

### 3. 第 2 步：列出「这次部署到底改了什么」

「没有改动模型」只说明权重没变。同一次发布里可能一起变的还有：推理引擎版本与默认参数（batch 上限、`max_num_tokens`、显存水位、block size、chunked prefill 开关）、CUDA/driver 与 kernel 选择、tokenizer 或 prompt 模板版本、网关与鉴权配置、自动扩缩容策略、限流阈值、日志/追踪采样率、上游客户端版本（重试与连接池）。把发布 diff 当作假设来源，而不是当作「没改」的证明。

### 4. 第 3 步：按先验概率排序的假设清单

1. **配置漂移**：`max_num_seqs` / `max_num_batched_tokens` / `gpu_memory_utilization` / `kv_cache_free_gpu_memory_fraction` / block size 被改小或改了默认值，调度器变保守 → batch 变小 → 吞吐下降、排队变长。
   - 验证：对比新旧实例的生效配置（打印启动参数与运行时快照），看 batch size 与 waiting 队列的分位。
   - 修复：配置纳入版本控制与灰度，发布时 diff 生效配置而不只是 diff 代码。
2. **缓存失效**：prefix cache 命中率掉了（prompt 模板改动把时间戳/请求 ID 放到前面、hash 规则或 block size 变化、多副本部署让缓存被打散），共享前缀被重复 prefill。机制见 [[inference-serving-05]]。
   - 验证：命中率（cached tokens / prompt tokens）、平均 prefill token 数、TTFT 与 prefill 段算力占比。
   - 修复：把动态内容后置并写成模板规范；缓存亲和路由或共享缓存层。
3. **长请求混入**：上游把长上下文或长输出请求放进同一池子，长 prefill 独占 GPU iteration，抬高所有在跑请求的 ITL。机制与修复见 [[inference-serving-14]]。
   - 验证：TTFT 与 ITL 的相关性、输入长度分布的分位、单次 iteration 时长的时间线。
   - 修复：长请求隔离池、按输入长度分桶路由、开启 chunked prefill、对超长请求限流。
4. **资源与拓扑变化**：GPU 型号混部（SXM 与 PCIe 的带宽/算力不同）、NUMA/PCIe 拓扑变化、TP/EP 度变化导致通信变慢、CPU 侧 tokenizer 或网关成为瓶颈。
   - 验证：按实例分组对比 p99；看通信时间占比；固定输入重放对比不同节点。
   - 修复：同构资源池化、拓扑亲和调度、把 CPU 侧前处理拆成独立服务。
5. **依赖与网络**：tokenizer/embedding/向量库/限流代理/DNS 变慢，超时与重试把有效 QPS 放大。
   - 验证：下游调用 P99、超时与重试计数、重试后的到达率曲线。
   - 修复：重试预算 + 指数退避 + 抖动，超时不大于上游 SLO 的余量。
6. **kernel 与编译回退**：换了镜像或 CUDA/driver 版本，FlashAttention 或量化 kernel 走到慢路径、CUDA graph 未命中（batch shape 不在捕获的桶里）而回落到 eager。
   - 验证：Nsight Systems 时间线里 kernel 名字与耗时、每步 CPU 侧 launch 开销。
   - 修复：锁定镜像与 kernel 版本、发布前跑固定形状的 kernel 基准、补齐 cudagraph 桶。
7. **观测与埋点偏差**：采样率、超时截断（把慢请求算成失败而不计入延迟）、重试让同一次请求被记两次、客户端连接复用变化，让「测出来的 p99」变了而不是服务变了。
   - 验证：用固定负载从集群内直连压测，和线上口径对不上就是观测问题。
   - 修复：统一延迟口径（含排队、不含客户端渲染）、把超时请求按上限计入而非丢弃。
8. **流量结构变化**：输入/输出长度分布、并发分布、租户结构变化，即使 QPS 不变也会推高单位成本。
   - 验证：输入/输出 token 长度分位的时间序列、每请求 FLOPs 估算、每请求 KV 占用。
   - 修复：容量按 token 而非按请求规划，把长度分布纳入容量模型。

### 5. 第 4 步：二分与回滚，每次只动一个变量

- **对照实验**：把上一个版本的镜像/配置与原版本同时起在同一批硬件上，用同流量影子（shadow）或小流量金丝雀（canary）分流，比较同一时刻的同分位指标。
- **同输入重放**：录制线上请求（固定 prompt、固定 `max_tokens`、固定采样参数），对两个版本各重放一遍。差异存在 → 服务端；差异消失 → 客户端、网络或流量结构。
- **分组对比**：按实例、节点、region、租户、输入长度桶切分指标。所有分组同时涨 → 全局变量；单分组涨 → 该分组的局部变量。
- **时间线对齐**：在同一时刻取 GPU 时间线与引擎指标，判断是排队增长（waiting 队列 + iteration 时长不变）还是计算增长（kernel 变长）。
- **一次一个变量**：配置项、镜像、上游版本逐个回滚，回滚后指标复位的那一项就是根因；同时改多项等于重新做一次无法归因的实验。

不要做的事：不要在 p99 上做跨实例平均（会把一个坏实例稀释掉）；不要一上来调参或加副本（会把根因掩盖成容量问题）；不要忽略客户端侧排队（边界外的延迟你永远修不到）；不要为了「先恢复」而重启，重启会销毁现场证据，顺序应该是先抓快照再回滚。

### 6. 第 5 步：修复与防回归

- 配置与镜像**版本化 + 灰度**，发布门禁里加「生效配置 diff」。
- 给 TTFT、ITL 的分位数、排队等待时间、prefix cache 命中率、KV 抢占/驱逐计数建 SLO 与告警；这些指标比 GPU 利用率更早暴露回归。
- 发布前跑**固定负载回归压测**：固定输入/输出长度分布、固定并发阶梯，比较新旧版本的 p50/p99 与每 token 成本。
- 长请求隔离池或限流；prompt 模板中动态内容一律后置，作为模板评审项。
- 客户端侧：重试预算、退避与抖动、连接池复用，避免故障时把负载放大成重试风暴。

## 数值与代码验证

以下换算统一口径：算力与带宽按十进制（989 TFLOPS $= 989\times10^{12}$ FLOP/s，3.35 TB/s $= 3.35\times10^{12}$ B/s），显存容量按厂商标称容量（每卡 80 GB、8 卡 640 GB），KV 字节数用精确值 327,680 B/token（即 320 KiB）；凡写成 GiB/KiB 的格子才是 1024 进制，所以表 3 的 KV 容量是 436 GB ÷ 327,680 B，不是 436 GiB ÷ 327,680 B。H100 口径为 **SXM 版、bf16 稠密算力 989 TFLOPS、HBM3 带宽 3.35 TB/s**，需要说明的是 PCIe 版是 2 TB/s 与 756 TFLOPS（稠密），所以混部本身就会拉高 p99——这也是假设清单第 4 条要分组对比的原因。MFU 按 40% 估算（明确假设，不是厂商数字）。

**表 1：E2E 分段与敏感度（TTFT = 排队 200 ms + prefill（表 3 的纯算力值）+ 首 token 与调度 80 ms；分段按表头取值）**

| 场景 | 段 | 基线 p99 | 变化 | 新 E2E | 倍数 |
| --- | --- | --- | --- | --- | --- |
| A 聊天式：prompt 2k / output 250 | TTFT 371 ms + decode 250×20 ms | 5371 ms | ITL 20 → 40 ms | 10371 ms | 1.93× |
| A | 同上 | 5371 ms | queue 200 → 1000 ms | 6171 ms | 1.15× |
| A | 同上 | 5371 ms | prefill 90.6 → 181 ms | 5461 ms | 1.02× |
| B RAG 短输出：prompt 8k / output 40 | TTFT 642 ms + decode 40×20 ms | 1442 ms | queue 200 → 800 ms | 2042 ms | 1.42× |
| B | 同上 | 1442 ms | ITL 20 → 40 ms | 2242 ms | 1.55× |
| B | 同上 | 1442 ms | queue ×4 且 ITL ×2 | 2842 ms | 1.97× |

表 1 的三个结论：① 输出越长，E2E 对 ITL 越敏感（A 场景下只靠 TTFT 让 E2E 翻倍，需要它从 371 ms 涨到 5741 ms，即 15.5 倍，实际不可能）；② 要凑出「正好翻倍」，通常是两个段同时退化（B 场景 queue ×4 且 ITL ×2 = 1.97×），所以不要只盯一个指标；③ 单请求的 prefill 翻倍在长输出场景只值 1.7%，它真正的杀伤力来自表 4 的聚合效应——重复 prefill 抬高整机利用率，再经表 2 放大排队。

**表 2：M/M/1 排队尾延迟随利用率的变化**（服务时间以均值为单位，$\mu=1$，$t_{99}=\ln(100\rho)/(\mu-\lambda)$）

| 利用率 $\rho$ | 0.45 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| 平均排队 $E[W]$ | 0.82 | 1.50 | 2.33 | 4.00 | 9.00 | 19.00 |
| p99 排队 $t_{99}$ | 6.92 | 10.24 | 14.16 | **21.91** | **45.00** | 91.08 |

- $0.80 \to 0.90$：p99 排队时间从 21.91 涨到 45.00 个服务时间，×2.05；平均排队从 4.00 涨到 9.00，×2.25。若平均服务时间 20 ms，p99 排队从 438 ms 涨到 900 ms。**利用率涨 10 个百分点就足以让 p99 翻倍**——这正是「模型没改、延迟翻倍」最常见的机制。
- 尾部与均值的绝对距离本来就大：M/M/1 下 $t_{99}/E[W]$ 在 $\rho=0.8$ 时是 $21.91/4.00 = 5.48$，$\rho=0.9$ 时是 $45.00/9.00 = 5.00$。均值上多出 5 个服务时间，传到 p99 就是多出 23 个服务时间，所以**评估容量必须看尾部分位，不能拿平均排队外推**。
- 重试把有效负载翻倍（$\rho$ 0.45 → 0.90）：p99 排队 ×6.50。
- 服务时间方差的影响用 Kingman 公式看：$E[W] \approx \dfrac{\rho}{1-\rho}\cdot\dfrac{c_a^2+c_s^2}{2}\cdot E[S]$。$c_s^2$ 从 1 涨到 4（长请求混入），$\rho=0.8$ 时平均排队从 4.00 涨到 10.00（×2.5）。**方差和均值一样重要**。

M/M/1 假设泊松到达与指数服务时间，真实推理负载有突发到达和重尾的服务时间，所以表中绝对值偏乐观，它的用途是给出「利用率越接近 1、尾部放大越快」这个量级判断；Kingman 公式里的 $c_a^2$、$c_s^2$ 就是用来修正这一偏差的。这类模型适合定方向，不应该当成容量数字直接写进承诺。

**表 3：70B 在 8×H100 上的关键常数（bf16，GQA-8：80 层 / 8 KV 头 / head_dim 128）**

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 权重 | $70\times10^9 \times 2$ B | 140 GB |
| TP=8 每卡权重读取下界 | $17.5\ \text{GB} / 3.35\ \text{TB/s}$ | 5.22 ms / decode step |
| KV cache | $2 \times 80 \times 8 \times 128 \times 2$ B | 320 KiB / token |
| 32k 上下文单序列 KV | $320\ \text{KiB} \times 32768$ | 10 GiB |
| KV 总容量（util 0.9） | $(576-140)$ GB | 约 1.33 M token（32k 会话约 41 条，4k 会话约 325 条） |
| 2k prompt prefill 算力 | $2 \times 70\times10^9 \times 2048$ | 0.287 PFLOP（节点 90.6 ms） |
| 8k prompt prefill 算力 | $2 \times 70\times10^9 \times 8192$ | 1.147 PFLOP（节点 362 ms，单卡 2899 ms） |
| 节点有效算力 | $8 \times 989 \times 0.4$ TFLOPS | 3.165 PFLOPS |
| roofline 拐点 | $989/3.35$ | 295 FLOP/byte |

**表 4：prefix cache 命中率掉光的代价**（每请求 2k token 共享前缀，重算需要 0.287 PFLOP）

| 到达率 | 额外算力需求 | 占节点有效算力（3.165 PFLOPS） |
| --- | --- | --- |
| 1 QPS | 0.29 PFLOPS | 9% |
| 5 QPS | 1.43 PFLOPS | 45% |
| 10 QPS | 2.87 PFLOPS | 91% |

重复 prefill 抢的是同一批 GPU 的 iteration 时间，于是利用率被抬高，再按表 2 的规律放大排队 p99——这解释了为什么「只是少了一点缓存命中」能变成尾延迟翻倍。

**表 5：长 prefill 对 ITL 的干扰**（8k prompt 独占节点 362 ms，到达率 0.5/s）

| 调度方式 | 单次阻塞时长 | 时间线被独占比例 | ITL p99 |
| --- | --- | --- | --- |
| 整段 prefill 插队 | 362 ms | 18% | 撞上阻塞的请求 ITL ≈ 362 ms |
| chunked prefill，chunk=512 | 22.6 ms（16 块） | 同左，但每块仅占一个 iteration | ≈ 23 ms |

18% 远大于 p99 需要的 1%，所以几乎每个请求都会撞上尖刺；切成 512 token 的 chunk 后单次侵入降到 22.6 ms。

```python
import math

# ---- 1) M/M/1 排队：P(W > t) = rho * exp(-(mu - lam) * t)，取 q=0.99 解出 t ----
def mm1_p99_wait(rho, q=0.99):
    """返回以平均服务时间为单位的 p99 排队时间（mu=1, lam=rho）。"""
    return math.log(rho / (1 - q)) / (1 - rho)

for rho in (0.45, 0.80, 0.90):
    print(rho, round(mm1_p99_wait(rho), 2))          # 6.92 21.91 45.0
print(round(mm1_p99_wait(0.9) / mm1_p99_wait(0.8), 2))   # 2.05  -> 利用率 0.8->0.9，p99 翻倍
print(round(mm1_p99_wait(0.9) / mm1_p99_wait(0.45), 2))  # 6.5   -> 重试把负载翻倍
print(round(mm1_p99_wait(0.8) * 20, 1), "ms")            # 438.2 ms（平均服务时间 20 ms 时）

# ---- 2) Kingman：服务时间方差对排队的影响 ----
kingman = lambda rho, ca2, cs2: rho / (1 - rho) * (ca2 + cs2) / 2
print(kingman(0.8, 1, 1), kingman(0.8, 1, 4))     # 4.0 10.0 -> 方差 x4，平均排队 x2.5

# ---- 3) 70B / H100：decode 下界、prefill 算力与阻塞时长 ----
BW, BF16, N, MFU = 3.35e12, 989e12, 70e9, 0.4
print(N * 2 / 1e9, "GB weights")                          # 140.0 GB
print(N / 8 * 2 / BW * 1e3, "ms/step at TP=8")            # 5.22 ms 权重读取下界
node = 8 * BF16 * MFU
for T in (2048, 8192):
    fl = 2 * N * T
    print(T, f"{fl/1e15:.3f} PFLOP", f"{fl/node*1e3:.1f} ms on 8xH100@40%MFU")
# 2048 0.287 PFLOP 90.6 ms / 8192 1.147 PFLOP 362.4 ms
print(2 * N * 2048 * 5 / node)                            # 0.45 -> 命中率掉光时 5QPS 多吃 45% 算力

# ---- 4) p99 的统计噪声：窗口内超过 p99 的样本数 ~ Binomial(n, 0.01) ----
for n in (1000, 10_000):
    print(n, f"{1 / math.sqrt(0.01 * n) * 100:.1f}%")     # 31.6% 10.0%

# ---- 5) 表 1 的 E2E 分解：TTFT = 排队 + prefill（表 3）+ 固定开销 80 ms ----
e2e = lambda q, pf, n_out, itl: q + pf + 80 + n_out * itl
A = e2e(200, 90.6, 250, 20)
print(round(A, 1), "ms")                                  # 5370.6
print(round(e2e(1000, 90.6, 250, 20) / A, 2))             # 1.15  queue 200 -> 1000 ms
print(round(e2e(200, 181.2, 250, 20) / A, 2))             # 1.02  prefill 翻倍
print(round(e2e(200, 90.6, 250, 40) / A, 2))              # 1.93  ITL 翻倍
B = e2e(200, 362, 40, 20)
print(round(B, 1), "ms", round(e2e(800, 362, 40, 40) / B, 2))   # 1442.0 ms 1.97（两段同时退化）
```

## 常见追问

- **追问**：怎么快速区分「排队变慢」和「计算变慢」？
  - 要点：看三组指标的组合。waiting 队列/并发变长而每步耗时不变 → 排队；每步耗时变长且 SM 利用率/MFU 升高 → 单位请求变贵（重复 prefill、长请求混入、kernel 回退）；每步耗时变长但利用率下降 → 通信、CPU 或显存（驱逐、重分配）。再拿 GPU 时间线对齐同一时刻确认，别只靠单一指标下结论。
- **追问**：为什么 p50 没变，只有 p99 翻倍？
  - 要点：p99 由最慢 1% 定义，中位数几乎不受尾部影响。常见机制是**子群污染**：若只有 $f$ 比例的请求走了更慢的路径（超长输入输出、缓存未命中、跨节点、重试），混合分布的 $q$ 分位在 $q < 1-f$ 时等于快值、在 $q > 1-f$ 时才跳到慢值。表 5 里 18% 的时间线被整段 prefill 占住，于是约 18% 的 token 间隔落在阻塞里：p50 与 p80 完全不动，p82 以上整体跳到 362 ms 的尖刺值——尖刺占比只要超过 1% 就会改写 p99，占比越大 p99 越贴近尖刺值（本例直接从 20 ms 抬到 362 ms，而不是恰好 2 倍）。另一条机制是队列本身的放大：$\rho$ 从 0.8 到 0.9 时平均排队 ×2.25，而 p99 相对均值本来就有约 5 倍的距离（21.91 对 4.00），均值上的一点变化在 p99 上是数量级更大的绝对增量。
- **追问**：怎么排除是客户端或网络的问题？
  - 要点：同一时间窗内并列比较客户端 E2E 与服务端 TTFT/ITL 的分位；从集群内直连服务做同输入重放，绕开公网与网关；查客户端重试计数、连接复用率、DNS/TLS 耗时。若服务端分位不变，问题在边界外，改服务端参数不会有效果。
- **追问**：prefix cache 命中率只是掉了一点，为什么能让 p99 翻倍？
  - 要点：因为算力是共享的。表 4 显示 5 QPS、每请求 2k 共享前缀时，重算要多吃 1.43 PFLOPS，相当于节点有效算力的 45%；利用率被抬上去之后，排队 p99 按表 2 的规律超线性放大。另外重算的 prefill 会和 decode 抢同一个 iteration，直接变成 ITL 尖刺。
- **追问**：回滚之后指标恢复了，这件事算结束了吗？
  - 要点：没有。回滚只证明「是这次变更引入的」，还没定位到单个变量。要继续做「一次一个变量」的二分，把根因收敛到一个配置项或一个 kernel，然后补上回归压测、配置 diff 门禁与对应告警，否则同一个变更下次还会以同样方式回归。
- **追问**：只有黑盒 API、没有 profiler 权限怎么办？
  - 要点：用可控输入的差分实验替代内省：固定 prompt 只变输出长度、固定长度只变并发、固定并发只变前缀共享比例、换 region 对比。用二阶代理指标（ttft/tpot 是否随并发阶跃、429 比例、cache 命中相关字段、token 速率稳定性）定位到段，再向服务方要对应段的指标。

## 公司变体

`asked_at` 覆盖的四家公司在公开材料与岗位方向上的侧重不同：

- **OpenAI**：自研推理栈与大规模 GPU 集群，偏工程实现与系统指标口径。追问会集中在「在不能改模型、也未必能改内核的前提下如何归因」、容量模型与调度策略，以及怎么把 p99 与成本放在一起权衡。
- **Amazon (AWS)**：托管推理服务视角，偏运维与容量工程。会更关注 CloudWatch 这类指标与告警、自动扩缩容与配额、多租户隔离，以及「怎么和客户沟通降级方案、怎么回滚」这类流程性问题。
- **Databricks**：多租户数据平台的模型服务，偏工程与资源治理。关注 GPU 池化与排队公平性、与其他工作负载争抢资源、成本归因，以及每个租户各自的尾延迟 SLO 怎么保证。
- **Perplexity**：面向用户的在线问答产品，偏端到端工程。p99 里包含检索、rerank 与流式返回的成分，追问会强调用户可见指标（TTFT 与流式卡顿）、快速迭代下的回归防护。

以上是依据各家公开技术材料与产品形态的侧重判断，具体面试形式以实际轮次为准。

## 相关题目

- [[inference-serving-09]]：TTFT、TPOT、ITL 与 throughput 的定义与权衡，先把指标口径对齐才谈得上「翻倍」。
- [[inference-serving-14]]：chunked prefill 如何把长 prefill 的侵入切成小片，对应第 4 节假设清单第 3 条。
- [[inference-serving-02]]：continuous batching 的调度机制，理解 waiting 队列与 batch size 为什么会变。
- [[inference-serving-05]]：prefix caching 的命中条件与失效原因，对应假设清单第 2 条。

## 参考资料与归属

1. [LLM Inference Optimization](https://outcomeschool.com/blog/llm-inference-optimization)，Amit Shekhar（Outcome School），2026-06-14。提供 prompt caching / prefix 复用的定位、continuous batching 与 PagedAttention 在服务栈中的位置、prefill 与 decode 两阶段的差异，第 2 节的分段口径与假设清单第 2、3 条基于这些材料整理。
2. [Mastering LLM Techniques：Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（延伸），Shashank Verma、Neal Vaidya（NVIDIA Technical Blog），2023-11-17。提供每 token KV cache 公式、prefill 为 compute-bound / decode 为 memory-bound 的划分、in-flight batching 与 KV cache 内存随 batch 与序列长度线性增长的分析；表 3 的 KV 与带宽口径、以及「decode 的权重读取下界」用它给出的公式配合 H100 规格复算。

表 2 的排队公式、表 3 与表 4 的算力数字、表 5 的阻塞时长均为自行推导与复算，H100 规格取厂商公开口径（SXM：bf16 稠密 989 TFLOPS、HBM3 3.35 TB/s），MFU 40% 是明示的估算假设而非实测值。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
