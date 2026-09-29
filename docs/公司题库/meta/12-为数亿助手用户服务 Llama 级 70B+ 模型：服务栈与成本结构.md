---
type: question
id: meta-12
company: Meta（超级智能实验室、FAIR、Llama）
topic: inference-serving
order: 12
question: 你需要为数亿助手用户提供 Llama 级别的 70B+ 模型服务。服务栈长什么样，钱又花在哪里？
question_en: You need to serve a Llama-class 70B+ model to hundreds of millions of assistant users. What does the serving stack look like and where does the money go?
asked_at: []
level: 高阶
tags: [服务栈, 成本模型, 容量规划, goodput, 前缀缓存, 模型路由, Llama]
sources:
  - title: LLM 推理优化
    url: https://outcomeschool.com/blog/llm-inference-optimization
    author: Amit Shekhar (Outcome School)
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving（延伸）
    url: https://arxiv.org/abs/2401.09670
    author: Zhong et al. (OSDI 2024)
    published: 2024-01-18
  - title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）
    url: https://arxiv.org/abs/2403.02310
    author: Agrawal et al.
    published: 2024-03-04
related: [inference-serving-08, inference-serving-09, inference-serving-12, inference-serving-11, system-design-10]
updated: 2026-09-28
---

## 一句话答案

> 服务栈分三层：**六跳链路**（客户端 → 边缘/BFF → LLM gateway → 会话与记忆 → 检索与工具 → 推理集群，安全护栏横切、计费遥测异步出关键路径）、**两个物理池**（prefill 吃算力、decode 吃带宽）、**一本账**（每百万输出 token 的美元数，分母是 goodput）。
> 账本钉在三个显式假设上：3 亿 DAU × 6 条/天 = 18 亿请求/天（日均 20,833 QPS，峰值取 5 倍即 10 万 QPS）；LLaMA-3-70B 每 token KV 320 KiB、bf16 权重 141.1 GB；每条 1.2k 输入 / 300 输出。最反直觉的结论是 prefill 吃掉约 74% 的 GPU·s（2.8 : 1），不是 decode。
> 钱的两笔大数：峰值满载 \$1.08/百万输出 token、时间平均 \$5.18/百万输出 token，4.8 倍差值就是「为峰值付钱」；TPOT ≤ 20 ms 的 SLO 下带宽只允许 295 条并发、显存允许 384 条，多放的 89 条让原始吞吐只涨 6.4%（14,752 → 15,702 token/s）却全部违约——报吞吐必须先说分母。
> 降本按风险排序：缓存复用 → 调度与批处理 → 精度 → 模型路由 → 解码层 → 容量采购 → 产品层；能相乘的只有作用在不同相位上的刀（前缀缓存只动 prefill、输出上限只动 decode），batching 与 FP8 抢的是同一个带宽上限。

## 面试官在考什么

- **先钉规模再谈钱**：DAU、人均消息数、峰值倍数、输入输出长度、SLO 五项不显式给出，后面每一个美元都不可复算；峰值倍数（这里取 5 倍）是最需要辩护、也最容易被追问的一条。
- **知不知道 prefill 与 decode 必须分开算**：这题的分水岭是「长 prompt 短输出」的助手流量落在 prefill 一侧，而不是靠直觉把成本押在输出 token 上。
- **分母意识**：goodput 而不是裸吞吐、峰值满载而不是时间平均、自有集群口径而不是租用 API 口径，三个口径混用可以把同一个系统说成相差 5 倍。
- **能不能落到可验证的数字**：KV 单价 320 KiB/token、8×H100 的池子与并发、TPOT 的边际代价 0.050 ms/序列、batch 被显存与 SLO 一起钉在 384。
- **生产意识**：前缀缓存的 exact-prefix 规则与失效洪峰、取消穿透、过载阶梯、相关性故障、量化与路由的回归门禁。
- Meta 语境下还有三跳特性必须答出：护栏双重计费（输入侧加 token、输出侧逐 chunk）、记忆与社交上下文的注入位置决定缓存命中、模型版本随 Llama 发布节奏整体切换。

常见错误答案：

- 只报一个「每百万 token 几美元」：不说分母是峰值还是均值、是 goodput 还是裸吞吐、算不算护栏与检索的算力，这个数字就没有意义。
- 把成本当成 decode 的账，优化全押在解码层与投机解码上；实际在 1.2k/300 的流量形状里 prefill 是 2.8 倍的大头，缓存与 chunked prefill 的杠杆更大。
- 把降本清单当乘法表乘出 10 倍，忽略 batching 与 FP8 在抢同一个带宽上限、前缀缓存只作用于 prefill。

## 原理与推导

### 1. 规模与假设：三行可复算的数

| 假设 | 取值 | 依据 / 影响 |
| --- | --- | --- |
| DAU | 3 亿 | 消费级助手规模 |
| 人均消息数 | 6 条/天 | 18 亿请求/天 → 日均 20,833 QPS |
| 峰值倍数 | 5× | 10 万 QPS；消费级日周期常见 3–10 倍，取 5 是常规口径，所有下游数字随它线性变化 |
| 输入 / 输出 token | 1,200 / 300 | 系统提示 + 裁剪后的历史 + 本轮问题；回答长度 |
| SLO | TTFT p95 < 1 s、TPOT ≤ 25 ms（goodput 算例另用 20 ms）、可用性 99.95% | 交互式对话的常规门槛 |
| 硬件 | H100 SXM5，80 GB HBM3，bf16 稠密 989 TFLOPs、FP8 1,979 TFLOPs、3.35 TB/s | 与仓库其它专题同一套常数 |
| 模型 | LLaMA-3-70B 级：80 层、GQA-8、head_dim 128、bf16 | 每 token KV 320 KiB |

由假设直接得峰值 token 需求：输入 $10^5 \times 1200 = 1.2\times10^8$ token/s，输出 $10^5 \times 300 = 3.0\times10^7$ token/s。这两个数加上 SLO，就是后面容量与成本的唯一入口。

### 2. 链路：逐跳预算与失败模式

```text
客户端（SSE/WebSocket，带取消通道）
  → 边缘 / BFF（鉴权、设备指纹、L7 限流、区域路由）+5～15 ms
  → LLM gateway（模型路由、前缀缓存查询、配额与预算）+10～30 ms
  → 会话与记忆服务（历史裁剪/摘要、画像与社交上下文）+20～60 ms
  → 检索 / 工具（可并行、独立超时预算）+50～300 ms
  → 推理集群（排队 → prefill → decode，TTFT 主导项）→ 流式回传（逐 chunk）
横切：安全与合规（输入护栏前置、输出护栏逐 chunk 判定）；计费与遥测（异步，出关键路径）
```

p95 预算示例：边缘 15 ms + gateway 30 ms + 会话组装 60 ms + 排队 200 ms + prefill 150 ms + 首 token 20 ms ≈ 475 ms，余量留给启用时的检索与护栏。排队 200 ms 是预算项而不是意外——峰值没有排队预算的设计必然违反 TTFT SLO。

Meta 语境下三跳必须单独讲：

1. **安全分类器（Llama Guard 一类）**：输入侧把策略文本与对话一起送进分类器，等于在 prompt 里多花 token；输出侧要逐 chunk 判定，不能等整段生成完。8B 分类器读 1.2k prompt 是 $2\times8.03\text{B}\times1200 = 1.93\times10^{13}$ FLOPs ≈ 主模型 prefill 的 11.3%，串行挂在 TTFT 上。所以它是成本与延迟的双重项：单独报耗时占比、单独给配额，并且别把它偷偷算进主模型账里当作推理成本。
2. **记忆与社交上下文的注入位置决定前缀缓存命中**：画像、社交关系、本轮检索结果这类每次都变的内容一旦拼在前缀开头，命中率直接归零。做法是把前缀固定成「系统提示 + 安全策略 + 工具定义」，可变部分全部后置。
3. **模型版本跟着 Llama 发布节奏整体切换**：缓存命名空间、灰度、回滚都按模型版本隔离。灰度期新旧版本共享同一前缀缓存，会读到布局不兼容的 KV。

四件必须在第一跳就定下来的事：流式是协议层的默认；取消要穿透到推理集群并释放 batch slot 与 KV block；会话状态与 KV cache 分离（前者是真相源、后者是可重建的派生物）；失败分级（纯文本对话 fail-open 并记录，涉及账户或工具操作 fail-closed）。

### 3. 钱的公式与三个口径

$$C_{1M} = \frac{p_{GPU\cdot h}}{3600 \cdot T_{eff}} \times 10^6, \qquad T_{eff} = T_{roofline}\cdot \eta_{MBU}\cdot \eta_{util}/(1+w)$$

四个因子对应四类手段：采购单价 $p$、精度与模型结构 $T_{roofline}$、batching 与缓存 $\eta_{MBU}$、浪费 token 与重试 $1/(1+w)$。报数统一成「美元/百万输出 token」与「美元/百万请求」两个单位。三个口径必须在开头声明：

1. **分母是 goodput 不是裸吞吐**（见第 6 节）：同一台机器显存允许 384 条、SLO 只允许 295 条，两个分母给出两个成本。
2. **峰值满载与时间平均相差 4.8 倍**：峰值满载口径是 \$1.08/百万输出 token（集群正跑在容量上限、没有闲置卡），时间平均口径是 \$5.18（把「按峰值买卡、整晚闲置」折进单位成本），比值就是 10 万 QPS ÷ 20,833 QPS。对外成本基准用后者。
3. **Meta 是自有集群加自建数据中心**：单价要换成折旧 + 电 + 机房 + 运维的自有成本口径（下面统一按每 GPU·h \$2 含电与折旧）。这一点和租用 API 的团队结论相反——他们的闲置是「买贵了、可以缩容」，自有集群闲置卡的现金成本不会随负载下降，只会摊进单位成本。

还要主动交代护栏与检索的算力算不算在这个账里：主账只算主模型推理的 GPU·s，护栏（第 2 节）与检索单独列账，否则「成本降了」可能只是把算力挪到了护栏。

### 4. prefill 与 decode 必须分开算

两条盘出来的账本，口径都写死：

| 账本 | prefill | decode | 结论 |
| --- | --- | --- | --- |
| 4,000 prompt / 500 输出，70B FP8，8×H100，prefill 按 40% MFU | 88.4 ms（$2\times70\text{B}\times4000 \div 6.33\ \text{PFLOP/s}$，即 15.83 PFLOP/s 的 40% MFU） | 53.3 ms（500 步 × 3.41 ms 的批次时间摊到 32 条并发） | prefill 占 62.4%，1.66 : 1 |
| 助手形状 1.2k 输入 / 300 输出，bf16，$b=384$ | 0.429 GPU·s | 0.153 GPU·s | prefill 占 73.8%，**2.8 : 1** |

第二条账里的 decode 时间 = 300 token × (每卡 17.5 GB 权重 + 384 条平摊后的 KV) ÷ 每卡 3.35 TB/s，即 19.11 ms 的节点时间 = 0.153 GPU·s。第一条账里 decode 的每步读数是 FP8 权重 70.55 GB + KV $32\times4000\times163840$ B = 91.5 GB，除以节点聚合带宽 26.8 TB/s 得 3.41 ms——把 prompt 写成 4,096 会让 prefill 变成 90.6 ms，量级不变。

机制上两者的算术强度差一个数量级以上（约 15 倍）：

| 相位 | FLOPs | 每步读数 | 算术强度 | 对拐点 295 的位置 |
| --- | --- | --- | --- | --- |
| prefill（1.2k prompt） | $2NP + 2Ld_{model}P^2 \approx 1.70\times10^{14}$ | 权重 140 GB | ≈ 1,210 FLOPs/byte | 4.1 倍之上，**算力受限** |
| decode 单步（$b=384$、4k） | $2Nb = 5.4\times10^{13}$ | 140 + 515 = 655 GB | ≈ 82 FLOPs/byte | 0.28 倍之下，**带宽受限** |

两池的最优 batch、并行度、扩缩容信号完全不同，这就是 P/D 分离的直接动机。代价与边界要一起说：KV 跨节点传输一条 8k 会话是 2.5 GiB，PCIe 单程在 40 ms 量级；而重算这条会话的 prefill 在 FP8 与 40% MFU 下要 181 ms，所以传输打得过重算。但收益依赖「两相位比例稳定」——端到端 goodput 的常见收益区间是 1.2–1.5×（工程估算口径，随流量形状与 SLO 变化）。引用论文要带前提：DistServe 是在满足 TTFT 与 TPOT 约束、90% 以上请求落在约束内的条件下，相比当时最优系统可服务 7.4 倍请求或收紧 12.6 倍 SLO。

看板必须分开报 prefill token/s、decode token/s 与两个相位各自的 GPU 小时占比；只报一个「总吞吐」会把两笔账搅成一本糊涂账。

### 5. 显存与带宽闭环

权重按 config 逐项加总是 70.55B（官方口径 70B，差 0.8%，因为不 tie embedding，输入 embedding 与 lm_head 各算一份）：

| 精度 | 计算式 | 大小 |
| --- | --- | --- |
| bf16 | $70.55\text{B} \times 2$ | 141.1 GB = 131.4 GiB |
| FP8 | $70.55\text{B} \times 1$ | 70.6 GB |
| INT4 + g128 fp16 scale | 0.5 B/参数 + 量化元数据 | 36.4 GB（元数据 1.10 GB，占 3.0%，不是 35 GB） |

KV 用 $M_{kv} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot S \cdot b \cdot p$：80 层 × 8 个 KV 头（不是 64 个 query 头）× head_dim 128 × bf16 → 每 token 327,680 B = 320 KiB = 3,276.8 token/GiB；8k 一条 2.5 GiB、32k 10 GiB、128k 40 GiB。把 KV 头数代成 query 头数就是 2,560 KiB/token，8 倍误差——这是这题最常被抓的口径。

8×H100 的显存账（标称 8 × 80 GB 按十进制读是 596.0 GiB；权重用 bf16 口径）：

| 项 | 计算式 | GiB |
| --- | --- | --- |
| 标称容量 | 8 × 80 GB | 596.0 |
| 权重 bf16 | 70.55B × 2 | −131.4 |
| 工作区 / 激活 / 通信 | 每卡 2.5 GiB × 8 | −20.0 |
| 碎片与水位 | 总量的 5%–10% | −10.0 |
| 全量切池（预算 B） | 596.0 − 131.4 − 20 − 10 | 434.6 |
| 生产配置（预算 A，再留 14% 抢占重算与突增水位） | 400 GB | 372.5 |

并发 = 池子 ÷ 单条占用：

| 上下文 | 单条 KV | 预算 A：372.5 GiB | 预算 B：434.6 GiB |
| --- | --- | --- | --- |
| 4k | 1.25 GiB | 298 | 348 |
| 8k | 2.5 GiB | **149** | **174** |
| 32k | 10 GiB | 37 | 43 |
| 128k | 40 GiB | 9 | 11 |

物理容量提醒：H100 的「80 GB」物理上是 80 GiB（nvidia-smi 每卡 8 万 MiB 量级），按物理 640 GiB 重算同一条账本会多出 44 GiB；统一按「8 × 80 GB = 596.0 GiB」的十进制标称口径报数，两者差 6.9%，说清用哪个即可。

带宽侧闭环（每步要读完整份权重加 batch 内全部 KV，节点聚合带宽 26.8 TB/s）：

| 场景（4k 上下文） | 每步读数 | 每步时间 | 每卡 token/s |
| --- | --- | --- | --- |
| batch=1，bf16 纯权重 | 140 GB | 41.8 ms | 23.9（精确 23.7） |
| batch=1，FP8 纯权重 | 70 GB | 20.9 ms | 47.9（精确 47.5） |
| b=32，bf16 | 183 GB | 6.83 ms | 586 |
| b=128，bf16 | 312 GB | 11.63 ms | 1,375（每卡要留 20 GiB 给 KV） |
| b=32，FP8 权重 + FP8 KV | 91.5 GB | 3.41 ms | 1,172 |

公布的 187.5 token/s/卡只有 roofline 586 的 32%，这正是 batching 的空间（推到 58% MBU 是 1.81×）。延迟侧的闭环同样用带宽算：TPOT 的边际代价是每卡每序列 KV ÷ 每卡带宽 = 40 KiB/token × 4,096 ÷ 3.35 TB/s = 0.050 ms/序列，固定项是每卡 17.5 GB 权重 ÷ 3.35 TB/s = 5.22 ms，于是 $b=384$ 时 TPOT 24.46 ms，已经贴住 25 ms 的 SLO——显存上限与延迟 SLO 一起把 batch 钉死在 384。

最后是带宽结构的结论：池子满载时 KV 读取 400 GB 已经超过权重 141.1 GB（2.8 倍），长上下文的瓶颈来自 cache 而不是权重。这解释了 KV 量化为什么同时改善容量与速度，也解释了为什么长上下文会话要先做摘要与分层缓存，而不是指望单卡显存。

### 6. goodput：分母决定结论

4k 上下文、TPOT ≤ 20 ms 的口径下，每步读数是每卡 17.5 GB 权重加 $b \times 4,096 \times 40$ KiB 的 KV，约束 $17.5\ \text{GB} + b\times0.168\ \text{GB} \le 20\ \text{ms} \times 3.35\ \text{TB/s}$（0.168 GB = 0.156 GiB，右边是 67 GB）给出 $b \le 295$，原始吞吐 14,752 token/s。显存侧按物理容量口径允许 383 条（478.6 GiB ÷ 1.25 GiB，即第 5 节物理容量提醒里的 640 GiB 口径），取整为 384 条；TPOT ≤ 25 ms 的延迟上限是 394 条，两条约束一起把 batch 钉在 384。

$b=384$ 时原始吞吐 15,702 token/s，比 295 条只高 6.4%，但多放的 89 条全部超出 20 ms 的 SLO：**被 SLO 判定为无效的 batch slot 占 23.2%**。同一台机器、同一个模型，「多放一点」的收益是 6.4%，代价是这批请求全部违约。所以报吞吐必须带分母，报成本必须用 goodput 反推的容量。

### 7. 生产里一定会踩的坑

1. **前缀缓存的 exact-prefix 规则**：前缀必须从第一个 token 起逐 token 一致；时间戳、随机 ID、每次各不相同的检索结果拼在开头会让命中率归零。系统提示或安全策略一改版，全量缓存同时失效并产生 prefill 洪峰 → 缓存按模型版本与租户命名空间隔离、改版灰度、TTL 随机化。收益上限是「prefill 占总 GPU 时间的比例 × 命中率」= 74% × 60% → 成本乘数 1.8×；两笔账要分开算：全体共享的同一个前缀近乎免费（10 个 2,000 token 变体合计 6.1 GiB），而按 3 亿用户各存一份 2,000 token 历史是 196.6 PB——而且命中别人的前缀既是正确性问题也是数据泄漏。
2. **取消要穿透到底**：用户关页面、改问题、点停止，都必须释放 batch slot 与 KV block，否则废弃请求继续占最稀缺的资源，这是过载雪崩的起点。
3. **过载阶梯按顺序启用**：配额与准入（429 加 retry-after）→ 优先级队列加提前拒绝（预测在 SLO 内完不成就立刻拒，比排队 30 秒后超时便宜，也不会诱发重试风暴）→ 降级（8B 兜底、输出从 300 降到 150 token、关工具与检索）→ 熔断与隔离 → 重试纪律（幂等、抖动、预算上限、用 request id 去重）。盯的指标是拒绝率而不是排队长度。
4. **相关性故障**：整点流量脉冲、TTL 对齐导致同一秒集体失效、灰度期新旧模型共享同一前缀缓存导致 KV 布局不兼容；解法分别是加抖动、TTL 随机化、按版本隔离命名空间。
5. **量化与路由必须有独立回归门禁与一键回退**：FP8 需要 Hopper 这一代的原生 tensor core 路径（FP4 属 Blackwell 一代），走不到原生 kernel 反而更慢；回退路径的最坏成本是「小模型浪费 + 大模型重试」，回退率不进单位成本账就会算错。

### 8. 降本清单：排序与为什么不能相乘

| 顺序 | 手段 | 量级 | 前提 |
| --- | --- | --- | --- |
| ① | 缓存与复用（前缀缓存、KV 复用） | 1.3–2×，重复问答可达 10× | 无损，但只作用于 prefill，且受 exact-prefix 约束 |
| ② | 调度与批处理（continuous batching、chunked prefill） | 2–4×（按 MBU 32% → 58% 取 1.8×） | 需要队列与准入配合 |
| ③ | 精度（FP8 权重 + FP8 KV） | 1.6–1.9×（上限 2×，见下） | 需要原生 kernel 与回归门禁 |
| ④ | 模型与路由（8B 兜底） | 换 8B 近 9×；路由 $G_{route}$ 2.13× | 质量门槛与回退率 |
| ⑤ | 解码层（max_tokens、stop 序列、结构化输出） | 1.05–1.3× | 消灭解析失败重试，只作用于 decode |
| ⑥ | 容量与采购 | 1.5–3× | 周期以月计 |
| ⑦ | 产品层（Batch API、上下文上限、按需降级） | 1.05–2× | 需要产品配合 |

路由的收益写成 $G_{route} = 1/(\alpha/r + (1-\alpha))$：$\alpha$ 是可路由到小模型的比例，$r$ 是大小模型成本比（70B 对 8B 是 $70/8 \approx 8.75$）。$\alpha = 0.6$ 时 $G_{route} = 2.13$，$\alpha = 0.7$ 时 2.63。

**为什么不能相乘**：batching 与 FP8 在抢同一个带宽上限——baseline 只有 roofline 的 32%（187.5 对 586 token/s/卡），batching 先把它推到 58% 左右，FP8 再抬高上限本身，两者叠加不是 $1.8 \times 2$。精度项的边际收益也随 batch 变化：权重 FP8、KV 仍 bf16、batch ≈ 80 时每步读数比是 1.39×，$b=32$ 且 KV 一起量化接近 2.0×，$b=384$ 的权重-only 只有 1.12×（KV 主导了读数）。而前缀缓存只作用于 prefill、输出上限只作用于 decode，这两项互不重叠，才是真正可以相乘的。

链式复算：$1.8 \times 1.8 \times 1.39 \times 2.13 \times 1.08 = 10.3$ 名义，乘 SLO 余量 0.8 与缓存命中波动 0.9 后是 7.4×。两个高杠杆旋钮的边际收益：路由比例 0.6 → 0.7（+23.3%）→ 折扣后 9.2×；命中率 0.6 → 0.75（缓存项 1.79 → 2.24，+25%）→ 折扣后 9.3×；两刀叠加 1.54× → 名义 15.9×、折扣后 11.5×。若把缓存项按「含 KV 写入与调度收益」的口径取 2.0×，名义就是 11.5×、折扣后 8.3×，此时路由单刀即 10.2×。结论是：**越过 10× 靠的是「哪些请求可以不用大模型答」（路由比例与命中率），不是更底层的 kernel**。

两个常被写错的取舍：投机解码是延迟手段（低 batch 带宽受限时顺带提吞吐，大 batch 算力饱和时被拒的 draft 是纯浪费），不能写进省钱清单；MoE 按激活参数算每 token 字节、按总参数算显存（全部专家要常驻），小 batch 下 expert 通信可能吃掉收益。chunked prefill 的量化形式要能当场给：4k prompt 不切块会给同批所有 decode 一个约 183–190 ms 的 ITL 台阶（prefill 有效吞吐 $2.24\times10^4$ token/s，约合 40% MFU），切到 2048 与 512 分别降到 92 ms 与 23 ms，代价是长 prompt 自己的 TTFT 略增。

### 9. 引擎选型、容量口径与利用率

选型上，Llama 权重开源、模型版本相对稳定，编译式栈（kernel 融合、kernel auto-tuning、CUDA graph）与自研调度拿到正收益的前提是「负载明显偏离通用引擎的假设 × 规模足够大」，判断式就是换引擎收益 = 负载偏离度 × 规模。自研的 break-even 要当场算：相对开源引擎提升 20%、集群 64 卡只等效省下 12.8 张卡（\$25.6/h，约 \$224k/年），还要再付一条与上游持续 merge 的分支的维护成本，而通用引擎覆盖的场景里自研只会更慢（上游每周都在提交 kernel 与调度改进）。

选型流程：定 SLO → 刻画负载（输入输出长度的 p50/p95/p99、共享前缀占比、结构化输出比例、多 LoRA 需求）→ 同 trace、同硬件、同精度压测 → 比 goodput 与每百万 token 的 GPU 小时 → 加上工程成本（迁移工时、发布流程、排障能力）→ 小流量灰度（两个引擎在温度 0 下也不一定逐 token 一致，kernel 与 batching 会改变数值累加顺序）。

容量与利用率上，两笔最容易漏掉的钱：容量必须按「缓存随时可能全失效」的无缓存口径配，不能把命中率写进容量基线；时间平均利用率只有 20.8%（峰值与均值差 4.8 倍），要靠错峰、离线 Batch API、夜间下线一批池子与弹性扩缩把它吃回来。弹性扩容要提前到 15 分钟量级，因为模型加载与 CUDA graph 捕获要几分钟到十几分钟，做不到秒级。

收尾用一张看板：美元/百万输出 token（按租户、模型、路由分支拆开）、prefix cache 命中率、prefill 与 decode 的 GPU 小时占比、TTFT p95 与 ITL p99、路由比例与回退率、重试率与拒绝率、安全护栏耗时占比。治理规则是任何优化都在固定负载加固定 SLO 下做 A/B、同时看成本与 p99；量 ITL 之前先确认服务端没有把 n 个 token 合并进一个 SSE chunk（合并后量到的间隔是真实值的 n 倍，分位数会整体偏好看）。

## 数值与代码验证

下面的脚本复算本节全部数字，常数口径写在开头（权重用逐项加总的 70.55B，吞吐与成本用官方 70B 口径，两者差 0.8%）：

```python
GB, GiB, KiB = 10**9, 2**30, 2**10
L, D, H_KV, D_HEAD, I_FFN, V = 80, 8192, 8, 128, 28672, 128256
N_CFG = L * (2*D*D + 2*D*H_KV*D_HEAD + 3*D*I_FFN) + 2*V*D     # 逐项加总（不 tie embedding）
N = 70e9                                                      # 服务主体账用官方 70B 口径（差 0.8%）
BW1, BF16, FP8, NODE, MFU, PRICE = 3.35e12, 989e12, 1979e12, 8, 0.40, 2.0
KNEE, BW_NODE, P_EFF = BF16/BW1, BW1*NODE, BF16*NODE*MFU
KV_TOK = 2 * L * H_KV * D_HEAD * 2                            # 327,680 B = 320 KiB/token
print(f"N 逐项 {N_CFG/GB:.2f}B | bf16 {N_CFG*2/GB:.1f} GB = {N_CFG*2/GiB:.1f} GiB | FP8 {N_CFG/GB:.1f} GB"
      f" | 拐点 {KNEE:.1f} FLOPs/byte | 节点带宽 {BW_NODE/1e12:.1f} TB/s")
print(f"KV {KV_TOK} B = {KV_TOK/KiB:.0f} KiB/token | {GiB/KV_TOK:.1f} token/GiB | 8k {8192*KV_TOK/GiB:.2f} GiB"
      f" | 32k {32768*KV_TOK/GiB:.0f} | 128k {131072*KV_TOK/GiB:.0f}"
      f" | MHA 口径 {2*L*64*D_HEAD*2/KiB:.0f} KiB（8 倍）")

POOL_B = 8*80/1.073741824 - N_CFG*2/GiB - 20 - 10             # 596.0 − 131.4 − 20 − 10
POOL_A = 400 * GB / GiB                                       # 生产配置：400 GB
print(f"KV 池 B {POOL_B:.1f} GiB、池 A {POOL_A:.1f} GiB（{POOL_A*GiB/GB:.0f} GB）"
      f" | 8k 并发 A {POOL_A*GiB/(8192*KV_TOK):.0f} / B {POOL_B*GiB/(8192*KV_TOK):.0f}"
      f" | 32k A {POOL_A*GiB/(32768*KV_TOK):.0f} | 128k A {POOL_A*GiB/(131072*KV_TOK):.0f}")

for b, w, kv in [(32, 140e9, KV_TOK), (128, 140e9, KV_TOK), (32, 70e9, KV_TOK//2)]:
    t = (w + b*4096*kv) / BW_NODE
    print(f"b={b:3d} 每步 {t*1e3:5.2f} ms → 每卡 {b/t/NODE:6.0f} token/s（每卡 KV {b*4096*kv/NODE/GiB:4.1f} GiB）")
print(f"batch=1 纯权重：bf16 {BW1/140e9:.1f}、FP8 {BW1/70e9:.1f} token/s/卡"
      f" | 公布 187.5 = roofline 586 的 {187.5/586:.0%}，推到 58% MBU 是 {586*0.58/187.5:.2f}×")

base, marg = (140e9/NODE)/BW1*1e3, KV_TOK/NODE*4096/BW1*1e3
print(f"TPOT = {base:.2f} + {marg:.4f}×b ms；b=384 → {base+marg*384:.2f} ms"
      f" | 上限：20 ms → {int((20-base)/marg)} 条、25 ms → {int((25-base)/marg)} 条")

preflops = lambda P: 2*N*P + 2*L*D*P*P
t_pre = preflops(1200) / P_EFF
node_pre, node_dec = 1200/t_pre, 384/((140e9 + 384*4096*KV_TOK)/BW_NODE)
PEAK, PIN, POUT = 1e5, 1200, 300
g_pre, g_dec = t_pre*NODE, POUT/node_dec*NODE
n_pre, n_dec = PEAK*PIN/node_pre, PEAK*POUT/node_dec
print(f"prefill {t_pre*1e3:.1f} ms/请求、节点 {node_pre:,.0f} tok/s；decode 节点 {node_dec:,.0f} tok/s")
print(f"峰值 1e5 QPS：{round(n_pre):,} + {round(n_dec):,} = {round(n_pre)+round(n_dec):,} 节点"
      f"（{(round(n_pre)+round(n_dec))*NODE:,} 张卡）"
      f" | GPU·s/请求 {g_pre:.3f}+{g_dec:.3f}={g_pre+g_dec:.3f} | prefill 占 {g_pre/(g_pre+g_dec):.1%}")
r = (g_pre+g_dec)*PRICE/3600
print(f"峰值满载 {r*1e6:,.0f} 美元/1M 请求、{r/POUT*1e6:.2f} 美元/1M 输出 token"
      f" | 时间平均（{20833/PEAK:.1%}）{r*1e6/(20833/PEAK):,.0f}、{r/POUT*1e6/(20833/PEAK):.2f}"
      f" | 倍差 {1/(20833/PEAK):.1f}×")
print(f"AI：prefill {preflops(1200)/140e9:,.0f}、decode {2*N*384/(140e9+384*4096*KV_TOK):.0f} FLOPs/byte"
      f" | 8k 会话重算 prefill {2*N*8192/(FP8*NODE*0.40)*1e3:.0f} ms vs PCIe 单程 40 ms"
      f" | 8B 护栏 = 主模型 prefill 的 {2*8.03e9*1200/preflops(1200):.1%}")
for b in (295, 384):
    t = (140e9 + b*4096*KV_TOK)/BW_NODE
    print(f"goodput：b={b} → {t*1e3:.2f} ms/步、{b/t:,.0f} token/s")

G = lambda a: 1/(a/(70/8) + (1-a))
f_pre = g_pre/(g_pre+g_dec)
cache, nominal = 1/(1-f_pre*0.60), None
nominal = cache*1.8*1.39*G(0.6)*1.08
print(f"缓存项 {cache:.2f}×（prefill {f_pre:.1%} × 命中 60%）| G(0.6)={G(0.6):.2f}、G(0.7)={G(0.7):.2f}"
      f"（+{G(0.7)/G(0.6)-1:.1%}）| 名义 {nominal:.1f}× → 折扣 0.72 → {nominal*0.72:.1f}×")
print(f"命中 75% → {nominal*(1/(1-f_pre*0.75))/cache*0.72:.1f}×；路由 70% → {nominal*G(0.7)/G(0.6)*0.72:.1f}×"
      f"；两刀 {G(0.7)/G(0.6)*(1/(1-f_pre*0.75))/cache:.2f}× → "
      f"{nominal*G(0.7)/G(0.6)*(1/(1-f_pre*0.75))/cache*0.72:.1f}×")
print(f"共享前缀 10 × 2,000 token {10*2000*KV_TOK/GiB:.1f} GiB；3 亿用户各一份 {3e8*2000*KV_TOK/1e15:.1f} PB"
      f" | 池子满载 KV {POOL_A*GiB/GB:.0f} GB 对权重 {N_CFG*2/GB:.1f} GB = {POOL_A*GiB/(N_CFG*2):.1f} 倍")
```

输出与口径对照：

| 输出 | 值 | 口径 |
| --- | --- | --- |
| 参数量与 KV 单价 | 70.55B；141.1 GB = 131.4 GiB；FP8 70.6 GB；KV 327,680 B = 320 KiB/token、3,276.8 token/GiB | config 逐项加总、不 tie embedding；KV 头数 8（MHA 口径会大 8 倍） |
| KV 池与并发 | 池 A 372.5 GiB（400 GB）→ 8k 149、32k 37、128k 9；池 B 434.6 GiB → 8k 174 | 596.0 − 131.4 − 20 − 10，池 A 再留 14% 水位 |
| 带宽闭环 | batch=1 是 23.9（bf16）/47.9（FP8）；b=32 是 586；b=128 是 1,375；FP8 是 1,172 | 每卡 token/s，节点聚合 26.8 TB/s |
| TPOT 模型 | $5.22 + 0.050 \times b$ ms；$b=384$ → 24.46 ms | bf16、4k、TP=8 |
| goodput 边界 | 20 ms → 295 条、25 ms → 394 条；295 条是 14,752 token/s，384 条是 15,702 token/s（+6.4%） | 带宽与 SLO 共同给的上限 |
| 两池与节点 | prefill 5,368 + decode 1,911 = 7,279 节点（58,232 张卡）；GPU·s/请求 0.429 + 0.153 = 0.582 | 峰值 10 万 QPS、bf16、40% MFU |
| 三个成本口径 | \$323/百万请求、\$1.08/百万输出 token（峰值满载）；\$1,553、\$5.18（时间平均，利用率 20.8%）；倍差 4.8× | 每 GPU·h \$2 含电与折旧 |
| 算术强度 | prefill 约 1,210 FLOPs/byte（含注意力项）；decode 82 FLOPs/byte | 拐点 295 |
| 前缀缓存 | 缓存项 1.79×；共享前缀 6.1 GiB；按用户缓存 196.6 PB | prefill 占比 73.8% × 命中 60% |
| 降本链 | 名义 10.3× → 折扣 0.72 → 7.4×；路由 70% → 9.2×；命中 75% → 9.3×；两刀 1.54× → 11.5× | 缓存项严格只算 prefill |

四个可以直接背的结论：**prefill 是 2.8 倍的大头**（1.2k/300 形状下 73.8% 的 GPU·s），所以第一优先级的降本是缓存与 chunked prefill，不是解码层；**同一台机器的两个分母差 4.8 倍**（峰值满载 \$1.08 对时间平均 \$5.18），混用会把同一个系统说成差五倍；**batch 被显存与 SLO 一起钉住**（显存 383–384 条、20 ms SLO 只给 295 条，多放的 89 条让原始吞吐涨 6.4% 却全部违约）；**KV 读取在高并发下超过权重 2.8 倍**（400 GB 对 141.1 GB），这是长上下文与 KV 量化的全部理由。

## 常见追问

- **追问**：为什么不是 decode 最贵？输出 token 明明是最贵的单价。
  - 要点：单价贵与总账占比是两件事。decode 的边际成本是每步读权重与 KV 的带宽，1.2k 输入/300 输出的形状下 prefill 占 73.8% 的 GPU·s（2.8 : 1），因为 prompt 里的系统提示、历史与检索结果都要重算一遍；输出侧 300 token 只对应 300 个带宽受限的步。所以降本顺序是缓存 → 调度 → 精度 → 路由，而不是先做解码层。
- **追问**：goodput 和 throughput 到底差在哪，面试里怎么用一句话说清？
  - 要点：throughput 是「每秒产出多少 token」，goodput 是「在 TTFT 与 TPOT 约束内每秒产出多少 token」。$b=295$（TPOT ≤ 20 ms）与 $b=384$ 的成本差 6.4% 的吞吐、23.2% 的超 SLO batch slot——用 throughput 当分母，所有超 SLO 的算力都算成了收益。
- **追问**：前缀缓存的收益上限怎么估？
  - 要点：上限 = prefill 占总 GPU 时间的比例 × 命中率。本例是 74% × 60% → 成本乘数 1.8×；同一份账要分开看存储：共享前缀 6.1 GiB 近乎免费，按 3 亿用户各存 2,000 token 历史是 196.6 PB。落地约束是 exact-prefix（差一个 token 就不能复用）、缓存命名空间按模型版本与租户隔离、改版灰度，否则一次策略更新就是一次 prefill 洪峰。
- **追问**：这题里最容易被忽略的成本项是什么？
  - 要点：三项。① 峰值与均值的 4.8 倍差——自有集群的闲置卡现金成本不随负载下降，只会摊进单位成本；② 护栏与检索的算力：8B 分类器读 1.2k prompt 就相当于主模型 prefill 的 11.3%，输出侧还要逐 chunk 判定，不单独列账就会把成本挪走；③ 回退与重试：小模型回退的浪费加大模型重试，回退率不进单位成本账就必然算错。
- **追问**：$b=384$ 这个数是怎么来的，为什么不能更大？
  - 要点：两个约束同时到顶。显存侧物理容量口径允许 383–384 条（4k 上下文、1.25 GiB/条）；延迟侧 TPOT $= 5.22 + 0.050\times b$ ms，$b=384$ 时 24.46 ms 已贴住 25 ms SLO（再往上 394 条就越线）。而 20 ms 的更严 SLO 只给 295 条。所以「batch 能开多大」是显存、SLO、上下文长度三个数一起定的，单独报一个 batch 没有意义。
- **追问**：自研推理引擎什么时候划算？
  - 要点：判断式是换引擎收益 = 负载偏离度 × 规模。相对开源引擎提升 20% 在 64 卡集群上只等效省 12.8 张卡（约 \$224k/年），却要多养一条持续与上游 merge 的分支；而通用引擎覆盖的场景（标准注意力、常规调度）自研只会更慢。可验证的流程是：同 trace、同硬件、同精度压测 → 比 goodput 与每百万 token 的 GPU 小时 → 加上工程成本 → 小流量灰度，注意两个引擎在温度 0 下也不保证逐 token 一致。

## 相关题目

- [[inference-serving-08]]：70B 的四块显存账与并发口径，第 5 节的池子与预算直接沿用它的记账方式。
- [[inference-serving-09]]：TTFT/TPOT/ITL/throughput 与 goodput 的定义，第 6 节的分母之争建立在它之上。
- [[inference-serving-12]]：降本 10 倍的手段清单与排序，第 8 节是它在助手流量形状下的落点。
- [[inference-serving-11]]：vLLM / SGLang / TensorRT-LLM / 自研栈的选型，第 9 节的 break-even 与选型流程与它互为一体。
- [[system-design-10]]：消费级聊天助手的完整服务栈，本题是它的成本与容量视角；两篇共用同一套假设与常数。
- 专题导读：[推理、服务与 GPU 性能](../../推理、服务与%20GPU%20性能/README.md)、[AI 系统设计](../../AI%20系统设计/README.md)。

## 参考资料与归属

1. [LLM 推理优化](https://outcomeschool.com/blog/llm-inference-optimization)，Amit Shekhar（Outcome School）。提供 KV cache 与压缩、PagedAttention、FlashAttention、continuous batching、prefix caching、prefill/decode 分离、speculative decoding 的手段地图，以及「真正的瓶颈是显存/内存而不是算力」这一判断；第 4、8 节的手段分类与取舍依据来自它。
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（延伸），Kwon et al.（vLLM, SOSP 2023），2023-09-12。第 8 节 batching 的量级与「把浪费限制在一个 block 之内」的分页记账口径取自该文（同等延迟下吞吐提升 2–4 倍，与 FasterTransformer、Orca 对比）。
3. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)（延伸），Zhong et al.（OSDI 2024），2024-01-18。第 4 节的 7.4 倍请求数 / 12.6 倍 SLO 收紧 / 90% 以上请求满足 TTFT 与 TPOT 约束，取自摘要原文；两相位相互干扰的问题陈述同样来自该文，prefill : decode = 2.8 : 1 是按第 1、5 节声明的常数自行复算的结果。
4. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)（延伸），Agrawal et al.，2024-03-04。第 8 节 chunked prefill 的机制与「切成近似等长的块、构造无停顿调度」取自该文；第 8 节的 ITL 台阶（4k 约 183–190 ms、2048 约 92 ms、512 约 23 ms）是按第 4 节的 prefill 有效吞吐复算的，不是论文数字。

正文章节里的规模假设（3 亿 DAU、6 条/天、5 倍峰值、1.2k/300 token）、8×H100 的显存账、TPOT 模型、goodput 边界、三个成本口径与降本链全部是按第 1、5 节声明的常数自行推导的，脚本见正文；KV 单价 320 KiB/token、H100 的 989 TFLOPs、1,979 TFLOPs、3.35 TB/s 与 roofline 拐点 295 FLOPs/byte 与仓库其它专题保持同一套常数。与来源出现分歧时以复算为准，口径已在正文标明。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
