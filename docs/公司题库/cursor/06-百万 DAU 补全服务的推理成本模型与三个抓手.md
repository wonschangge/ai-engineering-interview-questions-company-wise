---
type: question
id: cursor-06
company: Cursor（Anysphere）
topic: inference-serving
order: 6
question: 要把自研补全模型服务给数百万 DAU：请讲讲推理成本模型，以及你最优先的三个抓手。
question_en: Serving a custom completion model to millions of DAU: walk me through the inference-cost model and your top three levers.
asked_at: []
level: 高阶
tags: [推理成本, prefix caching, 容量规划, 连续批处理, 量化, 路由, goodput]
sources:
  - title: How does Prompt Caching work?
    url: https://outcomeschool.com/blog/how-does-prompt-caching-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Lost in the Middle: How Language Models Use Long Contexts（延伸）
    url: https://arxiv.org/abs/2307.03172
    author: Liu et al. (TACL 2024)
    published: 2023-07-06
  - title: Cursor 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-cursor-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Prefill vs Decode：LLM 推理优化
    url: https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: 
  - title: RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems（延伸）
    url: https://arxiv.org/abs/2306.03091
    author: Liu et al.
    published: 2023-06-05
related: [inference-serving-12, inference-serving-05, inference-serving-09, inference-serving-08, inference-serving-10, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> 先把成本写成公式：$C_{\text{1M}} = p_{\text{GPU·h}}/(3600 \cdot T_{\text{eff}}) \times 10^6$，$T_{\text{eff}} = T_{\text{roofline}} \cdot \eta_{\text{MBU}} \cdot \eta_{\text{util}}/(1+w)$，四个因子分别对应采购、精度与模型、batching 与缓存、浪费 token；补全这种「超长 prompt + 极短输出」的负载不能按 token 单价报价，要按次报价。
> 同一账本（4000 token prompt + 30 token 输出、70B FP8、8×H100、prefill 按 40% MFU）是 88.4 ms prefill 加 3.2 ms decode，节点墙钟 91.6 ms，即 0.733 GPU·秒，折 \$5.09×10⁻⁴/次、**\$509/1M 次补全**；prefill 占 96.5%，所以 decode 侧的一切（更大 batch、投机解码、TPOT 调参）最多只能动剩下的 3.5%。
> 三个抓手按「先无损、再低损、最后动质量」排：① prefix caching 加缓存亲和路由，命中 3000/4000 前缀后单次降到 25.3 ms（3.62×），完全无损；② 精度与模型路由，FP8 让 prefill 的算力上限翻倍、实测折扣后 1.6–1.9×，60% 流量走 8B 补全模型再加 2.13×，但必须按代码域指标设回归门禁；③ 调度与利用率，chunked prefill 把长 prefill 对同批 decode 的 ITL 冲击从 181 ms 压到 23 ms，利用率运营点停在 0.6–0.7。
> 分母侧（吞吐 × 利用率）上限 5–10×，分子侧（单价）只有 1.5–3× 且以月计；10× 的落点在「哪些请求可以不用大模型答」，不在更底层的 kernel。

## 面试官在考什么

- 有没有先把成本写成公式再谈抓手，并且开口就说清口径：91.6 ms 是**一个 8 卡节点**的墙钟时间，换算成钱要乘 tensor parallel 的卡数；报的是「每次补全」还是「每百万 token」。
- 知不知道补全的流量结构和对话完全不同：prompt 4000 token、输出 30 token，prefill 占 96.5% 的 GPU 时间；4000/500 的对话或 agent 请求只有 62.4%。这决定三个抓手打在 prefill 侧。
- 抓手排序的依据：无损（缓存与复用）→ 低损（调度、精度）→ 动质量（路由、蒸馏）→ 采购。倍数大但动质量、或者要等 kernel 就绪的，都得往后排。
- 补全的浪费项定义得对不对：不是「解析失败重试」，而是「用户继续打字导致的请求取消」与「预取但从未展示的补全」，必须单独埋点。
- 验收是否用 goodput 而不是裸吞吐，指标有没有按补全的流量结构切：命中率 × 命中长度、取消率、路由比例与回退率、按输出长度分桶的 p99 TTFT。

常见错误答案：

- 按 token 单价报价。4000 token prompt 的 prefill 成本被摊到 30 个输出 token 上，同一账本折成 \$17.0/1M 输出 token；只报 token 单价会让团队去优化只占 3.5% 的 decode。
- 把节点墙钟当成一张卡的占用：91.6 ms 直接乘 \$2.5/GPU·h 得到 \$63.6/1M 次，恰好把 8 张卡少算 8 倍。成本模型题的第一个坑永远是单位。
- 把各抓手倍数直接相乘（batching 2.0 × FP8 1.8 × 缓存 1.39 × 路由 2.13 × 输出上限 1.08 ≈ 11.5×），忽略它们作用在同一份带宽与算力上限上，也忽略 SLO 余量与命中波动。

## 原理与推导

### 1. 成本模型：四个因子，以及补全必须按次报价

按 token 计价时，单位成本等于「单位 GPU 小时价格 ÷ 每卡每秒交付的有效 token 数」：

$$C_{\text{1M}} = \frac{p_{\text{GPU·h}}}{3600 \cdot T_{\text{eff}}} \times 10^6, \qquad T_{\text{eff}} = T_{\text{roofline}} \cdot \eta_{\text{MBU}} \cdot \eta_{\text{util}} \cdot \frac{1}{1+w}$$

四个因子对应四类手段：$p_{\text{GPU·h}}$ 是采购与合约（上限 1.5–3×，以月计）；$T_{\text{roofline}}$ 由精度与模型决定（FP8、INT4、GQA/MLA、换成小模型）；$\eta_{\text{MBU}}$ 是实际达到峰值带宽或算力的比例（batching、chunked prefill、kernel 质量）；$\eta_{\text{util}}$ 是 GPU 真正做有用功的时间占比（调度、抢占、缓存命中把 prefill 整段省掉）；$w$ 是浪费 token 比例。

补全要换一个写法。每次补全占用的是一段时间，而这段时间里整个节点都在为它服务：

$$C_{\text{次}} = \frac{p_{\text{GPU·h}} \cdot N_{\text{卡}} \cdot t_{\text{节点}}}{3600}, \qquad t_{\text{节点}} = t_{\text{prefill}} + t_{\text{decode}}$$

$N_{\text{卡}}$ 就是 TP 的卡数（本题 8）。这一项在口算里最容易被漏掉，漏掉后整条成本链会差 8 倍。

**补全的 $w$ 是什么。** 对话场景的浪费是重试、解析失败、过度生成；补全场景是「用户继续打字，请求在流式输出中被取消」和「预取但从未展示的补全」。因为成本 96.5% 在 prefill，而 prefill 在首个 token 之前就付掉了，取消几乎意味着全额沉没：若 20% 的请求在 prefill 后被取消、另有 8% 的预取从未展示，则 $w \approx 0.20 \times 0.965 + 0.08 = 0.27$，有效成本要乘 1.27，账面吞吐相应折掉 21%。日志里必须区分「取消」「预取」「已展示并接受」三类，否则账面上会漏掉一整块付费算力。

### 2. 补全为什么是 prefill-bound

用同一个账本比较两种负载（70B FP8、8×H100、FP8 KV、decode 按 batch 32、每步 3.41 ms 节点墙钟、prefill 按 40% MFU）：

| 负载 | prefill | decode | prefill 占比 | decode 侧优化的天花板 |
| --- | --- | --- | --- | --- |
| 补全：4000 prompt / 30 输出 | 88.4 ms | 3.2 ms | **96.5%** | 3.5% |
| 对话或 agent：4000 prompt / 500 输出 | 88.4 ms | 53.3 ms | 62.4% | 37.6% |

结论很硬：在补全请求上，continuous batching 的槽位填充、投机解码、TPOT 调参加起来最多只能动 3.5%。真正的大头是那 4000 个 token 的 prefill——所以三个抓手全在 prefill 侧：让它别重复算（缓存）、让它算得便宜（精度与模型）、让它别堵住别人且把卡填满（调度与利用率）。

体验侧同理。端到端时间 $E = \text{TTFT} + (O-1) \cdot \text{TPOT}$，取 TTFT = 300 ms、TPOT = 10 ms：

| 输出长度 $O$ | decode 段 | E2E | TTFT 占比 |
| --- | --- | --- | --- |
| 10 | 90 ms | 0.39 s | 76.9% |
| 30 | 290 ms | 0.59 s | 50.8% |

补全的延迟预算要同时写「TTFT 主导」和「TPOT 不可忽略」：$O=30$ 时 decode 段已经涨到 TTFT 的 96.7%，两者基本打平，照抄对话场景的 SLO 会两头都守不住。

### 3. 抓手一：prefix caching + 缓存亲和路由（无损，先做）

**机制数字。** 块粒度 block=16 时，64 token 的前缀命中情况是：改第 0 个 token → 命中 0；改第 40 个 token → 命中 32（前两个块）；末尾追加 → 命中全部 64。重算 4096 token 前缀要 $2NL = 0.573$ PFLOPs，单卡 bf16 满 MFU 要 0.58 s、40% MFU 要 1.45 s；把它的 1.25 GiB KV 读回来只要 0.40 ms——**缓存命中把一段计算换成一次内存读取**，比值随口径在 1450×（100% MFU）到 3600×（40% MFU）之间。

按厂商计价口径（写 1.25×、读 0.1×），同一前缀被用 $k$ 次：不缓存是 $k$，缓存是 $1.25 + 0.1(k-1)$，解出盈亏平衡 $k^{\star} = 1.28$——**同一前缀被用到第二次就省钱**；$k=5$ 省 67%，$k=10$ 省 78.5%。

**Cursor 语境的四个坑。**

1. **prompt 顺序**：稳定内容（仓库结构、文件开头）放最前，光标附近的可变内容与检索片段放最后。顺序写反、或把动态时间戳拼在开头，命中率直接归零。
2. **负载均衡**：随机把同一文件的前缀打散到不同副本，显存里会全是只写不读的块。要么按 file/session 做亲和路由，要么做集群级共享缓存。
3. **BPE 边界**：用户每次按键改的是前缀尾部，但分词会把改动扩散到 token 边界之外，失配点比肉眼看到的更靠前。命中判定要按 token 序列的公共前缀长度算，不能按字符位置估。
4. **缓存块与 batch 抢显存**：缓存占的显存就是 batch 的上限，LRU 驱逐抖动会直接打到 p99。要给缓存池设配额、给 batch 留保底。

**收益复算。** 命中 3000/4000 前缀：prefill 从 88.4 ms 降到 22.1 ms，单次 25.3 ms，3.62×。按实测命中率 60%、可复用前缀占 75% 折算：1.77×。缓存作用于 prefill、量化作用于算力上限与带宽，两者不抵消，可以叠加。

### 4. 抓手二：精度与模型/路由（低损到动质量）

**FP8 对补全的价值主要在 prefill 侧。** decode 带宽受限时权重减半近乎直接翻倍吞吐，实测区间 1.6–1.9×；但在 prefill 占 96.5% 的补全负载上，更值钱的是 FP8 tensor core 把 prefill 的算力上限抬高 2×（按同样的实测折扣落到 1.6–1.9×），端到端接近这个倍数。KV 量化则额外买两样东西：长上下文下的显存与更大的 batch。

**KV 这笔账要背熟。** 70B GQA-8（80 层、8 个 KV 头、head_dim 128、bf16）是 $2 \times 80 \times 8 \times 128 \times 2 = 320$ KiB/token；4k 单序列 1.25 GiB；32 条并发各存一份是 40 GiB，而共享同一份前缀块只要 1.25 GiB（FP8 KV 再减半到 640 MiB/条）。这既是缓存的收益，也是缓存的显存成本。

**路由。** 小模型先答、置信度不足再回退大模型，收益为

$$G_{\text{route}} = \frac{1}{\alpha/r + (1-\alpha)}$$

$\alpha$ 是路由到小模型的流量比例、$r$ 是大模型与小模型的单请求成本比（按参数量近似 70B/8B = 8.75）：$\alpha = 0.6$ 时 2.13×。8B 补全模型的名义优势是 8.75×，但兑现不了——它必须用代码域指标设回归门禁（补全接受率、编辑距离、任务成功率、格式合法率），不能只看 PPL。回退路径的最坏成本是「小模型浪费 + 大模型重试」：回退率 $f = 0.1$ 就把 2.13× 吃成 1.89×，$f$ 必须算进单位成本。补全有一个有利点：质量是逐 token 可验证的（用户接受、拒绝、改写），「小模型 + 校验回退」比开放对话更容易落地。

### 5. 抓手三：把利用率与调度做实，再谈采购

**continuous batching 在补全上收益收缩。** static batching 的槽位利用率是 $1/H_S$（长度服从指数分布时）：$S=8$ 是 36.8%、$S=32$ 是 24.6%，对应 2.73×/4.06× 的实测加速。但补全输出短、长度分布偏斜小，$\mathbb{E}[\max]/\,\mathbb{E}[L]$ 接近 1，这项收益明显缩水；补全上它的主要价值变成「长 prefill 不阻塞 decode」和把 prefill 算力填满，也就是下面这件事。

**chunked prefill 的量化形式。** 8×H100、bf16、40% MFU 口径下单 token 的 prefill 成本约 44 µs，因此 4096 token 一步算完会给同批 decode 一个 181 ms（≈188 ms）的 ITL 台阶；切成 2048 或 512 的 chunk 后分别降到 91 ms 与 23 ms，代价是长 prompt 自己的 TTFT 略增。对补全这种「TTFT 主导」的负载，chunk 要按 TTFT 预算反推，而不是照搬对话场景的 token budget。

**利用率运营点。** M/M/1 口径下，$\rho$ 从 0.3 提到 0.6 单位成本减半（1.00× → 0.50×），但平均排队时间从 0.43× 涨到 1.50× 服务时间，$\rho = 0.8$ 是 4.00×。补全的 SLO 只有百毫秒级，运营点停在 0.6–0.7；再往上要加卡，不是继续压榨。

### 6. 排序依据与两个陷阱

**排序**：无损（缓存与复用）→ 低损（调度、精度）→ 动质量（路由、蒸馏）→ 采购。分母侧（吞吐 × 利用率）上限 5–10×，分子侧（单价）只有 1.5–3×，且换合约的周期以月计。

**陷阱一：把倍数直接相乘。** 名义 2.0（batching）× 1.8（FP8）× 1.39（缓存）× 2.13（路由）× 1.08（输出上限）≈ 11.5×，乘上 SLO 余量 0.8 与命中波动 0.9 后只有 8.3×；而 batching 那一项在 30 token 输出的补全上还要再缩（长度浪费本来就少），整条链按端到端口径折算落到 7–8×。batching 与量化本质上在抢同一个带宽与算力上限，收益不可能各拿满分。

**陷阱二：把投机解码和 P/D 分离当省钱手。** 投机解码增加 FLOPs，只在低 batch 带宽受限时顺带提吞吐，大 batch 下被拒绝的 draft 是纯浪费；P/D 分离收益 1.2–1.5×，且要先付 KV 跨节点传输的代价，对 30 token 输出的补全请求几乎没有产品价值。两者首先都是延迟手段。

**10× 的落点取决于流量结构。** 路由比例 60% → 70%：名义 14.2×、折扣后 10.2×；只把缓存命中率 60% → 75%：名义 12.8×、折扣后 9.2×。越过 10× 靠的是「哪些请求可以不用大模型答」，不是更底层的 kernel。

### 7. 验收：goodput 与按流量结构切分的看板

同一台机器在 4k 上下文、TPOT ≤ 20 ms 下，带宽允许 295 条并发、显存允许 384 条。多放那 89 条只让原始吞吐从 14,752 涨到 15,702 token/s（+6.4%），却全部超出 SLO——有效产能反而下降。所以验收口径是 goodput，不是裸吞吐。

补全要看的看板：\$ / 1M 次补全；命中率 × 命中长度；取消率与预取未展示率；prefill 与 decode 各自的 GPU 小时占比；路由比例与回退率；按输出长度分桶的 p99 TTFT。量 ITL 前先确认服务端没有把 $n$ 个 token 合并进一个 SSE chunk，否则量到的间隔是真实值的 $n$ 倍、分位数整体偏好看。治理规则：量化、路由、蒸馏各自有独立评测门禁与一键回退，任何优化都在固定负载 + 固定 SLO 下 A/B，同时看成本与 p99——「省了钱但 p99 爆了、接受率掉了」在这种登录态产品里会直接反映到留存。

## 数值与代码验证

口径：H100 SXM5（bf16 稠密 989 TFLOPs、FP8 稠密 1979 TFLOPs、HBM3 3.35 TB/s、80 GB、\$2.5/GPU·h）；一个节点 = 8×H100（TP=8）；模型 70B FP8，形状按 LLaMA-3-70B（80 层、8 个 KV 头、head_dim 128）；prefill 按 40% MFU，decode 按 batch 32、4k 上下文；内存用 1024 进制。

**表 1：单次补全的 GPU 时间账（4000 prompt / 30 输出）**

| 项 | 计算 | 节点墙钟 | GPU·秒（×8 卡） |
| --- | --- | --- | --- |
| prefill 4000 token | $5.60\times10^{14} \div 6.33\times10^{15}$ | 88.4 ms | 0.707 |
| decode 30 步 | $30 \times 3.41 \div 32$ | 3.2 ms | 0.026 |
| 合计 | | **91.6 ms** | **0.733** |
| 单次成本 | $0.733 \times 2.5 \div 3600$ | | \$5.09×10⁻⁴ |
| 每百万次 | | | **\$509** |
| 折算输出 token 单价 | $509 \div 30$ | | \$17.0/1M 输出 token |

decode 每步 3.41 ms 的独立复算：每卡读 70 GB / 8 = 8.75 GB 权重 + 32 × 4096 × 160 KiB / 8 = 2.68 GB KV，合计 11.43 GB ÷ 3.35 TB/s = 3.41 ms。

对照基线：按 187.5 output token/s/卡、\$2.5/GPU·h 得到 \$3.70/1M 输出 token（口径是 decode-only、不含 prefill）。补全按次报价折出的 \$17.0/1M 输出 token 之所以贵 4.6 倍，正是因为它把 4000 token 的 prefill 摊进了 30 个输出 token——这也是「补全不能按 token 单价报价」的数字依据。

**表 2：口径核对——同一个 91.6 ms 怎么算成钱**

| 口径 | 单次记账 | \$/1M 次补全 | 说明 |
| --- | --- | --- | --- |
| 按卡计价（TP=8，8 张卡都算钱） | 0.733 GPU·秒 | \$509 | 与 \$2.5/GPU·h 一致 |
| 把节点墙钟当 1 GPU·秒（漏乘 8） | 0.0916 | \$63.6 | 恰好少算 8 倍，账单差一个数量级 |

**表 3：容量复算（3M DAU × 300 次/天 = 900M 次/天 = 10,417 次/s）**

| 场景 | 每次 GPU·秒 | $\rho=1$ 需求 | $\rho=0.6$ 需求 | \$/天（\$2.5/GPU·h） |
| --- | --- | --- | --- | --- |
| 无缓存 | 0.733 | 7,636 卡（954 节点） | 12,727 卡（1,591 节点） | \$764k |
| 命中 3000/4000 前缀（100% 命中） | 0.202 | 2,109 卡（264 节点） | 3,515 卡（439 节点） | \$211k |
| 命中率 60% × 可复用 75%（1.77×） | 0.414 | 4,320 卡（540 节点） | 7,199 卡（900 节点） | \$432k |
| 同一账本漏乘 8 卡 | — | 954 | 1,591 | \$95k |

缓存那一列的收益是精确的：25.3 / 91.6 = 0.276，\$764k × 0.276 ≈ \$211k；漏乘 8 的版本就是 \$95k → \$26k。

**表 4：缓存的盈亏平衡（厂商计价口径：写 1.25×、读 0.1×）**

| 前缀被用 $k$ 次 | 不缓存 | 缓存 $1.25 + 0.1(k-1)$ | 节省 |
| --- | --- | --- | --- |
| 1 | 1.00 | 1.25 | −25% |
| 1.28 | 1.28 | 1.28 | 0（盈亏平衡） |
| 2 | 2.00 | 1.35 | 32.5% |
| 5 | 5.00 | 1.65 | 67.0% |
| 10 | 10.00 | 2.15 | **78.5%** |

**表 5：利用率与排队（M/M/1 口径）**

| $\rho$ | 相对单位成本 | 平均排队时间（× 服务时间） |
| --- | --- | --- |
| 0.3 | 1.00× | 0.43 |
| 0.5 | 0.60× | 1.00 |
| 0.6 | 0.50× | 1.50 |
| 0.7 | 0.43× | 2.33 |
| 0.8 | 0.37× | 4.00 |

**表 6：goodput 与裸吞吐（bf16、4k 上下文、TP=8、TPOT ≤ 20 ms）**

| 并发上限 | 原始吞吐 | 实际 TPOT | 结论 |
| --- | --- | --- | --- |
| 295（带宽允许） | 14,752 token/s | 20.0 ms | 达标 |
| 384（显存允许） | 15,702 token/s | 24.5 ms | 全部违约，有效产能下降 |

```python
# 口径：H100 SXM5（bf16 989 TFLOPs、FP8 1979 TFLOPs、HBM3 3.35 TB/s），节点 = 8 卡（TP=8），单价 2.5 美元/GPU·h
N, G, P8, P16, BW, P_GPU = 70e9, 8, 1979e12, 989e12, 3.35e12, 2.5
KV16 = 2 * 80 * 8 * 128 * 2                     # 327,680 B = 320 KiB/token（bf16）
SP, SO = 4000, 30                               # 补全请求：4000 prompt + 30 输出

t_pre = 2 * N * SP / (G * P8 * 0.4)             # prefill 40% MFU
step = (N / G + 32 * 4096 * KV16 / 2 / G) / BW  # FP8 KV、batch 32、4k 上下文
t_dec = SO * step / 32
node_s, gpu_s = t_pre + t_dec, (t_pre + t_dec) * G
print(round(t_pre * 1e3, 1), round(t_dec * 1e3, 2), round(gpu_s, 3), round(t_pre / node_s * 100, 1))
# 88.4 3.2 0.733 96.5

cost = gpu_s * P_GPU / 3600                     # 每次补全的美元成本
print(round(cost * 1e6, 1), round(cost / SO * 1e6, 2), round(node_s * P_GPU / 3600 * 1e6, 1))
# 509.0（$/1M 次）  16.97（$/1M 输出 token）  63.6（漏乘 TP=8 的同一账本）

t_hit = 2 * N * 1000 / (G * P8 * 0.4)           # 命中 3000/4000 前缀
print(round((t_hit + t_dec) * 1e3, 1), round(node_s / (t_hit + t_dec), 2))
# 25.3 3.62
print(round(node_s / ((1 - 0.6 * 0.75) * t_pre + t_dec), 3), round(1 - (1.25 + 0.1 * 9) / 10, 3))
# 1.768（命中率 60% × 可复用 75%）  0.785（k=10 省 78.5%）

rps = 3e6 * 300 / 86400                         # 3M DAU × 300 次/天
print(round(rps), [round(rps * s / 0.6) for s in (gpu_s, (t_hit + t_dec) * G)])
# 10417 [12726, 3515]   ρ=0.6 需要的卡数：无缓存 12,727；命中 3000/4000 后 3,515

print(round(1 / (0.6 / 8.75 + 0.4), 2), round(1 / (0.6 * (0.9 / 8.75 + 0.1 * (1 + 1 / 8.75)) + 0.4), 2))
# 2.13（α=0.6、r=8.75）  1.89（回退率 0.1 时）
```

与来源对照：参考源给的是机制与手段清单（Cursor 的补全与索引工作方式、prompt caching 的复用条件、推理优化手段），成本公式、per-request 账本、容量与排队数字没有现成口径，全部是按本节声明的参数自算；KV 公式与 320 KiB/token 与 [[inference-serving-12]]、[[inference-serving-08]] 保持一致。

## 常见追问

- **追问**：三个抓手只能做一个，选哪个？
  - 要点：缓存。它无损、不依赖 kernel 与评测门禁，收益只由流量结构决定，而且做完能拿到「哪些前缀真的重复、重复几次」这份数据，它决定路由与上下文裁剪的收益上限。代价是要改客户端拼 prompt 的顺序和网关的亲和路由，属于纯工程改动。
- **追问**：为什么不能按 token 单价报价？
  - 要点：4000/30 的负载把 prefill 摊到 30 个输出 token 上，折成 \$17.0/1M 输出 token；只报这个数会让团队去优化只占 3.5% 的 decode。对外统一报「每百万次补全」，对内同时报 prefill/decode 的 GPU 小时占比，两个口径一起看才不会误判优化方向。
- **追问**：缓存命中率上不去，先查什么？
  - 要点：① prompt 里有没有把动态内容（时间戳、光标附近的编辑区、随机 few-shot）拼在开头；② 网关是不是随机负载均衡，把同一文件的前缀打散到不同副本；③ 命中判定是不是按 token 序列公共前缀（BPE 会把失配点往前扩）。三项都是工程问题，不是模型问题。
- **追问**：缓存块和 batch 抢显存，怎么取舍？
  - 要点：给缓存池设配额、给 batch 留保底；驱逐策略按「命中收益 × 剩余 token 数」而不是纯 LRU（长前缀块即使很久没被用，复用时的收益也更大）。p99 抖动通常来自驱逐风暴，先看驱逐速率再看命中率。
- **追问**：路由的质量门禁怎么设？
  - 要点：用代码域指标（补全接受率、编辑距离、任务成功率、格式合法率），把「接受率不低于基线的 99%」写成硬 SLO，并做灰度与一键回退；回退率要算进单位成本（$f = 0.1$ 就把 2.13× 吃成 1.89×）。只比 PPL 会放过真实的补全质量退化。
- **追问**：prefill 占 96.5%，那缩短 prompt 不是更直接？
  - 要点：是，而且这是产品级抓手：减少检索片段、按需裁剪仓库上下文、精简 system prompt，砍 $S_p$ 的收益线性且大部分无损。但它同时动质量与用户预期，要有评测门禁；工程上它比换 kernel 见效快得多。
- **追问**：3M DAU 的账单怎么算给老板听？
  - 要点：10,417 次/s × 每次 GPU·秒 = 卡数，再乘 \$2.5/GPU·h 与 24 小时。务必说清乘不乘 TP 的卡数（本题差 8 倍），并给出 $\rho=0.6$ 的运营点、缓存命中率与回退率三个假设；不同假设下的账单差 3 倍以上，先把假设摆上桌再谈结论。

## 相关题目

- [[inference-serving-12]]：成本公式与四因子、10× 的标准组合清单；本题把同一套公式落到「超长 prompt + 极短输出」的补全负载上，并补上 TP 卡数与「按次报价」两个口径。
- [[inference-serving-05]]：prefix caching 的块级命中、共享前缀的显存账、重算与读回的数量级对比，是抓手一的前置。
- [[inference-serving-09]]：TTFT/TPOT/ITL 的定义与权衡，本题的延迟预算与 SSE chunk 合并陷阱依赖它的口径。
- [[inference-serving-08]]：70B 的权重与 KV 显存账，抓手二的 320 KiB/token、40 GiB 与 1.25 GiB 与它同源。
- [[inference-serving-10]]：batch=1 的 roofline 手算与 295 FLOPs/byte 拐点，本题 decode 步长时间的来源。
- [[inference-serving-02]]：static 与 continuous batching 的槽位利用率 $1/H_S$、2.73×/4.06× 的模拟结果与失效条件。
- 专题导读：[推理、服务与 GPU 性能](../../推理、服务与 GPU 性能/README.md)。

## 参考资料与归属

- **How does Prompt Caching work?** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-prompt-caching-work>。
- **Lost in the Middle: How Language Models Use Long Contexts（延伸）** —— Liu et al. (TACL 2024)，2023-07-06：<https://arxiv.org/abs/2307.03172>。
- **Cursor 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-cursor-work>。编辑器补全产品的延迟与成本约束背景。
- **Prefill vs Decode：LLM 推理优化** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization>。prefill 与 decode 两阶段的开销差异（成本模型主项）。
- **RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems（延伸）** —— Liu et al.，2023-06-05：<https://arxiv.org/abs/2306.03091>。仓库级补全的质量-成本评测口径。

- **延伸来源说明**：文中的复杂度对照、内存账与实测算例，是按本仓库统一口径自行推导与实测的工程算例，不是上述来源的原文数字；来源仅用于机制、算法与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
