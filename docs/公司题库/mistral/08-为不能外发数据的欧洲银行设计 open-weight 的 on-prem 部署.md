---
type: question
id: mistral-08
company: Mistral AI
topic: system-design
order: 8
question: 为一家不能把数据发往任何外部 API 的欧洲银行，设计 open-weight 模型的 on-prem 部署方案。
question_en: Design an on-prem deployment of an open-weight model for a European bank that cannot send data to any external API.
asked_at: []
level: 高阶
tags: [系统设计, on-prem 部署, 数据合规, open-weight, 容量规划]
sources:
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（vLLM）（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (SOSP 2023)
    published: 2023-09-12
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
  - title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（延伸）
    url: https://arxiv.org/abs/2401.09670
    author: Zhong et al. (OSDI 2024)
    published: 2024-01-18
  - title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）
    url: https://arxiv.org/abs/2403.02310
    author: Agrawal et al.
    published: 2024-03-04
  - title: SGLang: Efficient Execution of Structured Language Model Programs（延伸）
    url: https://arxiv.org/abs/2312.07104
    author: Zheng et al.
    published: 2023-12-12
  - title: RouteLLM: Learning to Route LLMs with Preference Data（延伸）
    url: https://arxiv.org/abs/2406.18665
    author: Ong et al.
    published: 2024-06-26
  - title: FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance（延伸）
    url: https://arxiv.org/abs/2305.05176
    author: Chen, Zaharia, Zou
    published: 2023-05-09
related: [system-design-01, system-design-09, system-design-10, inference-serving-08]
updated: 2026-09-29
---

## 一句话答案

> 先把边界钉死：被禁止的不是「没有零保留承诺的 API」，而是**外部处理方本身**，所以架构上要把外部处理者从链路里删掉，而不是层层加合同——权重、tokenizer、embedding 与 reranker、向量库、脱敏与 guardrail 模型、评测集、日志与 trace、制品库全部留在行内，容器网络默认 deny egress，推理引擎与框架的匿名用量上报显式关闭，权重先镜像进内部制品库再上机。这正是 [[system-design-09]] 的反命题：那里的 gateway 靠跨供应商路由与故障转移创造价值，这里的「供应商」只有行内自己的集群，gateway 退化成配额、审计与降级阶梯的控制面。
> 容量账（5 万员工 × 6 次/天 = 30 万次/天 ≈ 3.5 QPS 均值，峰值 30 QPS 即 8.6×，1.2k 输入 / 300 输出）：8×H100 节点上 prefill 22,355 输入 tok/s、decode 15,702 输出 tok/s，峰值只要 $1.61 + 0.57 \approx 2.2$ 个节点，取整加冗余买 **3 个 8 卡节点 = 24 张 H100 ≈ 21 kW**；真正的代价不在卡数而在利用率：日均 3.5 QPS 对 41–56 QPS 的产能只有 6–8% 利用率，每 1M 输出 token 的等效成本从满载口径的 1.08 美元涨到 13–17 美元。模型落点用 Mistral 的 open-weight：70B 级 bf16 权重 141 GB，**必须 TP=8（或至少 TP=4）**——TP=2 时每卡权重 70 GB、KV 只剩 11 GiB，单节点 4k 并发 71 条，接不住峰值并发 240 条；7B 级（Mistral 7B 的 GQA + sliding window 4096，每 token KV 128 KiB、每序列封顶 512 MiB）单卡可放，用来做分流与降级兜底；Mixtral 8×7B 这类稀疏 MoE 按 47B 买显存、按 13B 算算力。没有外部 API 就没有「切云端大模型」这张保底牌，降级阶梯只剩小模型 → 限流排队 → 关检索与工具 → 返回缓存答案。

## 面试官在考什么

- **边界感与假设**：能不能第一句就把「数据不得外发」翻译成架构约束（无外部处理者、deny egress、关遥测），并把人数、人均次数、token 长度、SLO、存量硬件这些没写在题面上的量先澄清成一张假设表。这题对应客户场景轮（`README.zh-CN.md` 里 Mistral AI 的应用类岗位有客户场景轮），澄清能力就是区分度。
- **两笔账都要算**：容量账（prefill 吃算力、decode 吃带宽 → 节点数 → 功耗与机房条件）与利用率账（买卡按峰值、用卡按均值 → 单位 token 成本涨一个数量级）。只报「买 3 台 8 卡服务器」而不说利用率与单位成本，等于没答 on-prem 的核心矛盾。
- **显存是硬约束，不是优化项**：权重先吃显存，KV 决定并发；能说清 TP 度、量化、KV 量化三者的取舍，并用峰值并发（Little 定律）反查显存是否装得下。
- **气隙工程与无兜底降级**：权重与引擎镜像怎么进来、怎么验签与出 SBOM、遥测与自动更新怎么关（这类开关必须点名，不能含糊）、蓝绿要同时容纳两份权重、无外网也要能跑完离线评测；没有外部 API 时降级阶梯必须自建且可审计，容量冗余（N+1）与限流排队不是「优化」，是可用性的一部分。
- **合规叙事能不能落地**：GDPR（数据最小化与外部处理者的消除）、EU AI Act（高风险系统的日志、技术文档与人工监督）、DORA（ICT 第三方风险与退出计划）要各自对应到具体的工程动作，而不是背条款。

常见错误答案：

- 答成「混合云 + 外部 API 兜底」，或退一步说「用承诺零保留（ZDR）的 API 就行」。被禁的是外部处理方这个角色本身——数据一旦离开行内边界就产生了 GDPR 第 28 条的处理者关系、跨境传输评估与 DORA 的第三方集中度问题，这些不靠合同条款消除。
- 只谈模型与卡数，不谈运维：权重冻结后没有上游补丁通道，CVE 响应、许可证合规与长期支持要自己承担；也没有把「换模型 = 重大变更」的流程成本算进方案。

## 原理与推导

### 1. 边界：全栈 in-perimeter，以及 gateway 为什么会退化

| 层 | 行内必须自持的组件 | 对应的外发风险 |
| --- | --- | --- |
| 模型 | 权重与 tokenizer（多语言 byte-fallback BPE）、embedding、reranker、脱敏与内容策略分类器 | 任何一次推理调用、任何一次 embedding 调用都是外发 |
| 数据 | 向量库与索引、检索服务、会话存储、审计库 | 检索托管服务会把文档与查询一起带出去 |
| 平台 | 推理引擎与框架、容器 registry 与制品库、监控与 trace 后端 | 镜像拉取、遥测、自动更新、崩溃上报都是静默出口 |
| 流程 | 离线评测集与回归门禁、审批与工单、许可证与 SBOM 档案 | 无外网时「质量回归只能靠用户投诉发现」 |

**默认 deny egress** 是这一节的操作核心：容器网络只放行内部制品库与内部监控端点，DNS 只解析内网，时间源用内部 NTP。三个必须显式关掉的开关（否则气隙只是口号）：推理引擎的匿名用量统计——vLLM 默认开启，官方文档给的关闭方式是把 `VLLM_NO_USAGE_STATS=1`（或 `DO_NOT_TRACK=1`）写进镜像环境，或创建 `~/.config/vllm/do_not_track` 文件，并用 `tail ~/.config/vllm/usage_stats.json` 核对开关生效后该文件不再增长；框架侧的离线模式与遥测开关——`HF_HUB_OFFLINE=1`（禁止任何对 Hub 的 HTTP 调用）、`HF_HUB_DISABLE_TELEMETRY=1`（或 `DO_NOT_TRACK=1`）、`HF_HUB_DISABLE_UPDATE_CHECK=1`（不发 PyPI 请求）；以及权重下载路径——先在隔离区下载、验签、出 SBOM，再推入内部 registry 与制品库，推理节点只从内部拉取。

gateway 在这一版里仍然要留，但职责缩水成控制面：配额与优先级（按部门/业务线）、审计留痕、降级阶梯的执行、模型版本与灰度的路由。它的「路由」不再有跨供应商含义（见 [[system-design-09]] 第 3 节的分层路由），只剩「同一套权重谱系里选大/小模型 + 选优先级队列」。同时要立一条硬规矩：gateway **不得**在任何路径上兜底到外部 API，包括「只把失败样本发出去看看」这类运维需求的变体。open-weight 在这里的价值是可替换性：Mistral 7B 与 Mixtral 8×7B 的权重都以 Apache 2.0 发布（模型卡口径），可自托管、可再分发、可离线冻结，行内可以同时持有多个镜像版本并把「换供应商」降级为「换 registry 里的一个 tag」；但许可证要逐模型过法务——同一家的 open-weight 家族并不全是 Apache 2.0（Codestral 系列走非商用许可），商用条款与再分发限制必须在选型前定下来。

### 2. 假设与流量

| 假设 | 取值 | 说明 |
| --- | --- | --- |
| 员工数 / 人均次数 | 5 万 × 6 次/天 | 30 万次/天 |
| 输入 / 输出 token | 1,200 / 300 | 系统提示 + 政策片段 + 本轮问题；输出为摘要与答复 |
| 均值 / 峰值 QPS | 3.47 / 30（8.6×） | 银行流量集中在 09:00–17:00，午间与月末结账还有次级尖峰 |
| SLO | TTFT p95 ≤ 1 s、TPOT ≤ 25 ms | 与 [[system-design-10]] 同口径 |
| 硬件 | 8×H100 SXM5（80 GiB、bf16 稠密 989 TFLOPs、HBM3 3.35 TB/s） | MFU 0.40 |
| 模型 | 70B 级 dense 为主，7B–8B 级做分流与兜底 | 权重与 KV 口径见第 4 节 |

流量形状决定了一件事：**峰值与均值差 8.6 倍，而这个差距买卡是买不完的**。夜间与周末的 24 张卡会大段闲置，所以方案里必须写清「闲置算力干什么」（见第 6 节），否则 3 台服务器就是一笔只买到「峰值可用性」的固定成本。

### 3. 容量账：两条产能、节点数、机房条件

**prefill 吃算力。** 单请求 FLOPs 约 $2NP + 2Ld_{model}P^2$，代入 $N = 7\times10^{10}$、$P = 1200$、$L = 80$、$d_{model} = 8192$ 得 $1.68\times10^{14} + 1.9\times10^{12} \approx 1.70\times10^{14}$ FLOPs；8 卡节点在 MFU 0.40 下有效算力 $989\times8\times0.40 = 3.165$ PFLOP/s，于是单请求 53.7 ms、节点 prefill 吞吐 22,355 输入 tok/s。

**decode 吃带宽。** 每步要读完整份权重加 batch 内全部 KV。每卡留 60 GiB 给 KV 时（[[inference-serving-08]] 的口径）$b_{\max} = 384$，单节点每步读 $140 + 384\times4096\times320\ \text{KiB} = 140 + 515 = 655$ GB，聚合带宽 26.8 TB/s 得 $t_{step} = 24.46$ ms，即 TPOT 24.46 ms（贴住 25 ms 的 SLO）与 15,702 输出 tok/s。

| 相位 | 峰值需求 | 节点产能 | 需要节点 |
| --- | --- | --- | --- |
| prefill | $30\times1200 = 36{,}000$ tok/s | 22,355 tok/s | 1.61 |
| decode | $30\times300 = 9{,}000$ tok/s | 15,702 tok/s | 0.57 |

两相位相加 2.18 个节点，取整买 3 个节点（24 张 H100）。这里要主动交代冗余口径：3 个节点在任一台故障后剩 2 台，覆盖 2.18 个节点的 92% 峰值需求，缺口交给降级阶梯（小模型分流 + 排队）；若要严格做到「任一台宕机仍 100% 满足峰值」，就是 4 个节点 32 张卡。银行场景建议按 3 台买、把缺口写成明文的降级预案，而不是假装 N+1 成立。机房条件要先进规划：24 × 700 W 的 GPU 是 16.8 kW，节点整机（CPU、内存、网卡、风扇与 PSU 损耗）约 7 kW/台，3 台约 **21 kW**；按 PUE 1.3 折成约 27 kW 制冷负荷，需要 10 kW 级机柜、双路供电与电池续航——这是 on-prem 与「租 API」在采购流程上最现实的差别：卡到位之前先要有机房、配电与散热。

### 4. 显存账：权重先吃，KV 定并发

70B 级 bf16 权重 140 GB（按 config 逐项加总 141 GB），单卡 80 GiB 放不下，必须切分；切分度决定 KV 池，KV 池决定并发：

| 落点 | 权重/卡 | 每卡 KV 池 | 4k 并发/卡 | 节点 4k 并发 | 节点 decode 产能 |
| --- | --- | --- | --- | --- | --- |
| bf16 TP=2 | 70 GB | 11.1 GiB | 18 | 71 | 2.9k tok/s |
| bf16 TP=4 / FP8·INT8 TP=2 | 35 GB | 43.7 GiB | 140 / 70 | 279 | 11.4k tok/s |
| bf16 TP=8 | 17.5 GB | 60.0 GiB | 384 | 384 | 15.7k tok/s |
| INT4 W4A16 单卡 | 36 GB | 42.6 GiB | 34 | 273 | 11.2k tok/s |

假设表的峰值并发用 Little 定律反查：单请求时长约 $0.5 + 300\times25\ \text{ms} = 8.0$ s，峰值并发 $\approx 30\times8.0 = 240$ 条，4k 上界下需要 $240\times1.25 = 300$ GiB KV（70B 每 token 320 KiB，4k 每序列 1.25 GiB）。**TP=2 的 141 GB 权重把每卡 KV 压到 11 GiB，单节点只装 71 条，直接否决**——这就是「想省卡反而落不了地」的具体算术；TP=4 最低可用，TP=8 在并发与产能上都最优（384 条、15.7k tok/s），所以本题默认 TP=8。量化的取舍要说清口径：FP8/INT8 权重把 141 GB 压到 70 GB，让 TP=2 也能承接峰值并发（279 条），但 8 卡节点的解码产能只有 11.4k tok/s——与 bf16 TP=4 相同、低于 bf16 TP=8 的 15.7k：TP=2 意味着节点上要放 4 份权重副本，每步读的权重总量反而是 bf16 TP=8 的两倍（4×70 GB vs 140 GB），KV 池也从 60 GiB 降到 43.7 GiB；量化的价值在单卡显存（让 2 卡甚至单卡起得来），不在节点解码吞吐。INT4（AWQ/GPTQ 一类 W4A16）再减半、单卡就能放下 70B（产能 11.2k tok/s），但**必须过行内任务级评测集**——困惑度不是证据，信贷摘要与要素抽取的字段级正确率才是。Pope et al. 在 540B 上做 int8 权重量化时报告低 batch 29 ms/token、大 batch 76% MFU，那是 TPU v4、2,048 token 上下文的自有实验，不能直接外推到 H100 与本题的负载。

KV 要当一等公民：INT8 KV 把每 token 320 KiB 砍到 160 KiB，等于同 TPOT 下并发翻倍（TP=8 时 384 → 768 条），代价是 attention 的精度风险要单独评测。另一个常被忽略的驱动是长上下文：一条 32k 的序列是 10 GiB，240 条并发里只要有 5% 是长文，这 12 条就要 120 GiB，比按 4k 估的 15 GiB 多出 105 GiB，所以长文流程要限并发或单独走小池子。

存量硬件要单独给一条路：如果客户手上是 **2×A100-80G 且无 NVLink**，跨卡 TP 的通信代价很高，落点是「量化权重 + 单卡放得下的 7B 级模型」。以 Mistral 7B 为例（32 层、8 个 KV 头、head_dim 128、sliding window 4096）：bf16 权重 14.5 GB、每 token KV 128 KiB、SWA 让每序列 KV 封顶 $4096\times128\ \text{KiB} = 512$ MiB，于是单卡 KV 池 62.8 GiB 可放 125 条；A100-80G 的 2.039 TB/s 下，batch 68（TPOT 25 ms）时单卡 2,719 输出 tok/s，跑满 KV 池的 batch 125（TPOT 40 ms）也只有 3,124 tok/s——**滚动的滑动窗口让并发不再随长度增长，但吞吐曲线很快就平了，因为瓶颈在 KV 读取而不是权重**。prefill 侧更紧：7B 在 A100 上 1,200 token 要 142 ms，单卡 8,434 输入 tok/s，2 张卡只接住峰值输入的 47%，要承接 70% 分流（25,200 tok/s）需要 3 张。MoE 的账要分开算：Mixtral 8×7B 的 46.7B 总参数在 bf16 下占 93 GB（87 GiB）显存，每 token 只激活 12.9B，单请求 prefill FLOPs 只有 70B 的 18%（便宜 5.4 倍）——省的是算力与带宽，不是显存，采购按 47B 定显存、按 13B 估算力。8 卡节点上 TP=8 的 Mixtral 每卡只用 11.7 GB 权重、KV 池 65.4 GiB（每 token 128 KiB，只有 70B 的 40%），节点 decode 产能约 42.8k tok/s，是 70B 节点的 2.7 倍；若为了省卡把它按 TP=2 装成 4 份副本，权重涨到 46.7 GB/卡、KV 池压到 32.8 GiB，产能掉到 21.4k（1.37×）。代价是没有现成的 70B 级质量口径，必须自己建评测。

### 5. 固定卡数下提产能：四个机制各治一种病

| 机制 | 治什么病 | 论文口径 | 本题的量化收益 |
| --- | --- | --- | --- |
| continuous batching + PagedAttention | 静态批处理的空转与 KV 碎片 | KV 碎片降到接近零、支持请求内与跨请求共享，同延迟下吞吐 2–4×，序列越长、模型越大收益越明显 | decode 显存按实际占用分配，$b_{\max} = 384$ 才成立 |
| chunked prefill | 长 prompt 独占 iteration 造成的 ITL 尖刺 | 按 token 预算切块并无停顿调度；Sarathi-Serve 在多种模型上把服务容量提到 2.6–5.6× | 4k prompt 不切块留 188 ms 台阶，切到 512 token 只剩 22.8 ms；本题均值的 1.2k prompt 是 53.7 ms → 22.8 ms |
| prefix 复用（radix/block） | 系统提示与政策库被反复 prefill | RadixAttention + 压缩有限状态机，多类负载吞吐最高 6.4× | 800 token 共享前缀、40% 命中 → 每条省 320 prefill token（输入侧的 26.7%）、9.5 GPU·h/天 |
| P/D 分离 | 两个相位的资源错配与相互干扰 | 90% 以上请求满足延迟约束时可服务 7.4× 请求，或把 SLO 收紧 12.6× | 本题 prefill:decode = 2.81:1，但 3 节点规模上不划算（见下） |

prefix 复用的口径要写清：[[system-design-10]] 的「2,000 token 前缀 + 40% 命中 → 每条省 800 token、省下约 49% 总算力」是那份 3 亿 DAU / 18 亿请求/天算例的结论，前提是长 prompt 流量；银行这边平均输入只有 1.2k，放不下 2,000 token 的共享前缀，按 800 token（系统提示 + 政策要点 + 工具定义）与 40% 命中算，省的是每条 320 个 prefill token，prefill 需求从 1.61 降到 1.18 个节点、全天总需求 1.75 个节点——**不足一个节点，改变不了 3 台的采购决定，但它把峰值余量从 37% 抬到 71%**。缓存本身的存储代价可以忽略（10 个提示变体 × 800 token × 320 KiB ≈ 2.4 GiB），前提是命名空间必须按部门/租户与模型版本隔离——命中别人的前缀在银行里就是数据泄漏，也就是 [[system-design-09]] 第 4 节说的「缓存串租户」。P/D 分离在本题是「知道但先不用」：prefill 占 74% 的 GPU·s（53.7/72.8），比例确实失衡，但 3 个节点的粒度太粗——拆成 1 个 decode 节点 + 2 个 prefill 节点后，单节点故障会打掉 50% 的 prefill 产能，而 KV 跨节点传输又要求同级互联。在单机内用 chunked prefill + prefix 复用就能把 ITL 尖刺压到 23 ms 量级，分离留到流量涨 5–10×、或 TTFT 与 TPOT 同时收紧时再上（判据见 [[inference-serving-15]]）。结构化抽取（金额、日期、账户字段）这类任务用带压缩 FSM 的引擎（SGLang 一类）而不是自由文本解码，论文在多个负载上报了最高 6.4× 的吞吐提升。

### 6. 利用率、降级阶梯与路由

**利用率是最便宜的优化。** 24 张卡每天供给 576 GPU·h，实际算力消耗只有 48.5 GPU·h（8.4%）。夜间与周末窗口（约 9 h × 24 卡 ≈ 216 GPU·h/夜）应该排满对账、报表生成、离线评测、语料标注与蒸馏实验，把闲置算力变成质量与技术债的偿还能力；这比任何单点推理优化都便宜。

**降级阶梯（没有外部兜底版）**，越靠前越优先：

1. 小模型分流：8B 级单卡 batch 64 时 4,390 输出 tok/s，8 卡节点 35.1k tok/s，是 70B 节点的 2.24 倍，同样的卡能接住更多的峰值流量。
2. 限流排队：按部门与业务线给配额，返回可行动的 `Retry-After`；排队只对可等待流量开放。
3. 关闭检索与工具：退化成纯问答（质量下降但链路完整），并在响应里标注降级级别。
4. 返回缓存或模板答案：只对答案稳定的政策类问答开放，必须记录命中来源与版本。
5. 拒绝并给人工通道：高风险决策（信贷、风控）不因降级而跳过人工复核。

每一次降级都要留痕：对银行来说「降级」本身是合规事件，必须能回答「哪一级被触发、影响了哪些请求、谁批准的」。**路由与级联在 on-prem 里省的是卡，不只是钱。** [[system-design-09]] 表 2 给了 API 价格口径下的规则路由 70% 分流 3.0×、级联（25% 升级）3.4×；换成节点·秒口径重算要小心——银行的分流目标是 8B 级模型，它的单位产能比 70B 节点高 5.1×（不是 API 价格比 22.5×），所以：全量 70B 是 72.8 ms 节点·秒/请求 → 2.18 个节点；规则路由 70% 分流降到 31.7 ms → 0.95 个节点；级联（全走 8B、25% 升级）32.4 ms → 0.97 个节点（2.25×）。**结论是采购可以从 3 个节点降到 2 个（16 张卡，省 8 张卡约 1.15 万美元/月）**，代价是 8B 必须通过被分流流量的质量门禁，而且单一 70B 节点在峰值跑到 65% 利用率、没有突发余量。FrugalGPT 的「最多降 98% 成本或同成本下 +4% 准确率」与 RouteLLM 的「部分设置下 2× 以上」都是各自论文的 benchmark 与计费口径，不能当线上收益承诺。

### 7. 气隙里的变更管理：换模型是重大变更

- **供应链**：权重、引擎镜像、依赖 wheel 与模型卡一起走「隔离区下载 → 病毒与漏洞扫描 → 签名与哈希校验 → 出 SBOM → 推入内部 registry」，推理节点只从内部拉取；权重文件与许可证存档，作为 DORA 退出计划的证据。
- **蓝绿要双份权重**：显存与存储都要同时容纳新旧两版（70B bf16 141 GB、Mixtral 93 GB、7B 14.5 GB），3 个节点里要留出一个完整节点做新版本池，升级窗口必须提前算进容量。
- **三个版本号独立灰度与一键回滚**：模型权重、prompt 模板、引擎版本分开推进，任一项可单独回退（[[evaluation-08]]）。行内要有一套**无外网也能跑完**的离线评测集与回归门禁，否则质量回归只能靠用户投诉发现（[[evaluation-04]]）。
- **没有上游补丁通道**：权重冻结后 CVE 响应、许可证合规与长期支持由行内承担，要订阅离线 CVE 源、把镜像重建写成流水线，并把漏洞响应 SLA 与供应商依赖一起写进风险清单。

### 8. 合规落地：欧洲银行的四件事

| 要求 | 工程动作 |
| --- | --- |
| 数据最小化 | 上下文只带必要字段；PII 在进推理前用行内模型脱敏；检索结果先做权限裁剪（[[rag-08]]） |
| 可审计 | 请求级 trace 串起模型/权重版本、prompt 版本、引擎版本、检索引用与审批记录、降级级别（[[evaluation-07]]） |
| 日志最小化 | 原始 prompt 与输出默认只存哈希与长度，全文按采样加密留存并设保留期——日志与 trace 是 PII 最常见的出口（[[safety-07]]） |
| 人工监督 | 信贷与风控属 EU AI Act 高风险，自动化拒贷必须有可解释依据、人工复核与申诉通道，模型输出不直接写进决策系统 |

合规叙事可以一句话串起来：GDPR 关注的是「数据是否被外部处理者处理」，把外部处理方从架构里删掉比签一份数据处理协议更彻底；EU AI Act 关注的是高风险系统的日志、技术文档与人工监督，正好落在上表的第二、三行；DORA 关注 ICT 第三方风险与退出计划，而 **open-weight 的可自托管与可替换性本身就是退出计划的答案**——但要把版本冻结、SBOM、漏洞响应 SLA 与「换模型的窗口期」写成可检查的条款，否则「我们能自己跑」只是一句话。

## 数值与代码验证

常数与仓库其它专题一致：H100 SXM5 bf16 稠密 989 TFLOPs、HBM3 3.35 TB/s（拐点 295 FLOPs/byte）、2 美元/GPU·h；A100-80G 312 TFLOPs、2.039 TB/s；LLaMA-3-70B 每 token KV 320 KiB；Mistral 7B 与 Mixtral 8×7B 都是 32 层 / 8 KV 头 / head_dim 128（每 token 128 KiB，Mistral 7B 另有 sliding window 4096）；Mixtral 8×7B 46.703B 总参数 / 12.880B 激活。

**表 1：产能与采购**

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 请求/天 | $50{,}000\times6$ | 300,000（均值 3.47 QPS） |
| 峰值倍数 | $30/3.47$ | 8.64× |
| prefill 单请求 | $2NP + 2LdP^2$ | $1.70\times10^{14}$ FLOPs → 53.7 ms → 22,355 tok/s |
| decode 单步 | $(140 + 384\times4096\times320\text{KiB})/26.8$ TB/s | 24.46 ms → 15,702 tok/s |
| 峰值节点 | $36{,}000/22{,}355 + 9{,}000/15{,}702$ | 1.61 + 0.57 = 2.18 → 买 3 个节点（24 卡） |
| 功耗 | $24\times0.7$ kW + 节点开销 | 16.8 kW（GPU）→ 约 21 kW 整机 → 27 kW 制冷（PUE 1.3） |

**表 2：利用率与单位成本**（同一个系统的三种口径，混用会把成本说差 16 倍）

| 口径 | 产能分母 | 利用率 | 每 1M 输出 token |
| --- | --- | --- | --- |
| 峰值满载（持续跑满） | — | 100% | 1.08 美元（0.582 GPU·s/请求） |
| 理论产能（prefill 与 decode 完全重叠） | 55.9 QPS | 6.2% | 17.4 美元 |
| 实测产能（两相位墙钟相加） | 41.2 QPS | 8.4% | 12.8 美元 |
| 整机分摊（3 节点 34,560 美元/月 ÷ 实际 2,700M 输出 token） | — | — | 12.80 美元 |

**表 3：显存与并发**（70B 级、4k、每卡 80 GiB；峰值并发 240 条来自 Little 定律）

| 配置 | 每卡 KV 池 | 节点 4k 并发 | 是否接住 240 条 |
| --- | --- | --- | --- |
| bf16 TP=2 | 11.1 GiB | 71 | 否 |
| bf16 TP=4 / FP8·INT8 TP=2 | 43.7 GiB | 279 | 是（量化要多一道验收） |
| bf16 TP=8 | 60.0 GiB | 384 | 是（默认落点） |
| INT4 W4A16 单卡 | 42.6 GiB | 273 | 是（产能 11.2k tok/s，最低） |

**表 4：机制收益（本题口径）**

| 项 | 计算 | 结果 |
| --- | --- | --- |
| chunked prefill | 1.2k / 4k prompt 整段 vs chunk 512 | 53.7 → 22.8 ms；188.1 → 22.8 ms |
| prefix 复用 | 800 token 前缀 × 40% 命中 | 每条省 320 token（26.7% 输入）、9.5 GPU·h/天 |
| 8B 分流 70% / 级联 25% 升级 | 0.3 × 72.8 + 0.7 × 14.2 ms；0.75 × 14.2 + 0.25 × 86.9 ms | 31.7 ms → 0.95 节点；32.4 ms → 0.97 节点（2.25×） |
| 存量 A100-80G | 7B、TPOT 25 ms → batch 68 | 2,719 输出 tok/s/卡；prefill 8,434 输入 tok/s/卡（2 张 = 47% 峰值输入） |

```python
# 仓库口径：H100 SXM5 bf16 稠密 989 TFLOPs、HBM3 3.35 TB/s、MFU 0.40、2 美元/GPU·h
BASE, BW1, MFU, NODE, P_GPU, GB, KW = 989e12, 3.35e12, 0.40, 8, 2.0, 1e9, 1024 ** 3
P_EFF = BASE * MFU * NODE                                    # 3.165 PFLOP/s
L, D, H_KV, DH, B = 80, 8192, 8, 128, 2                      # LLaMA-3-70B 形状
N70, W70, K70 = 70e9, 140e9, 2 * 80 * 8 * 128 * 2            # 140 GB 权重、320 KiB/token
N7, L7, D7, H7 = 7.241e9, 32, 4096, 8                        # Mistral 7B（SWA 4096）
W7, K7 = 2 * N7, 2 * L7 * H7 * DH * B                        # 14.5 GB 权重、128 KiB/token
pre = lambda p, n=N70, l=L, d=D: 2 * n * p + 2 * l * d * p * p

# 表 1、表 2：峰值 30 QPS、1.2k 输入 / 300 输出，按 3 个 8 卡节点算
I, O, QPS, PK = 1200, 300, 300_000 / 86400, 30
t_pre = pre(I) / P_EFF                                       # 53.7 ms
node_pre = I / t_pre                                         # 22,355 输入 tok/s
b_max = int(60 * KW / (K70 / NODE * 4096))                   # 每卡留 60 GiB → 384 条
t_step = (W70 / NODE + b_max * K70 / NODE * 4096) / BW1      # 17.5 GB 权重 + 64.4 GB KV → 24.46 ms
node_dec = b_max / t_step                                    # 15,702 输出 tok/s
node_s = t_pre + O / node_dec                                # 72.79 ms 节点·秒/请求
gpu_s, unit = node_s * NODE, node_s * NODE * P_GPU / 3600    # 0.582 GPU·s、1.08 美元/1M 输出 token
print(f"节点 {PK*I/node_pre + PK*O/node_dec:.2f}；峰值 {PK/QPS:.2f}×")
print(f"理论 {3/t_pre:.1f} QPS / {QPS/(3/t_pre)*100:.1f}% / {unit*1e6/O/(QPS/(3/t_pre)):.1f} 美元；"
      f"实测 {3/node_s:.1f} QPS / {QPS/(3/node_s)*100:.1f}% / {unit*1e6/O/(QPS/(3/node_s)):.1f} 美元")
print(f"月成本 {NODE*3*24*30*P_GPU:,.0f} 美元；实际算力 {QPS*86400*gpu_s/3600:.1f} GPU·h/天（供给 {NODE*3*24}）")
# 节点 2.18；8.64× ／ 理论 55.9 QPS、6.2%、17.4 美元 ／ 实测 41.2 QPS、8.4%、12.8 美元
# 34,560 美元/月；48.5 GPU·h/天（供给 576 → 8.4%）

# 表 3：显存与并发（每卡 80 GiB，扣权重、2.5 GiB 工作区与 1.25 GiB 碎片）
def plan(w_total, tp, per_tok=K70, ctx=4096):
    bud = 80 - w_total / GB / tp / 1.074 - 2.5 - 1.25        # 每卡 KV 池（GiB）
    seqs = bud / (ctx * per_tok / KW / tp)                   # 该卡装得下的 4k 序列数
    t = (w_total / tp + bud * KW) / BW1                      # 每卡每步读取（权重分片 + KV）
    return bud, seqs * NODE / tp, seqs * NODE / tp / t
for tag, w, tp in [("bf16", W70, 2), ("bf16", W70, 4), ("bf16", W70, 8),
                   ("int8", 70e9, 2), ("int4", N70 * 0.516, 1)]:
    bud, conc, thr = plan(w, tp)
    print(f"{tag} TP={tp}：KV {bud:.1f} GiB、并发 {conc:.0f}、产能 {thr:,.0f} tok/s")
# bf16 TP=2 11.1/71/2,899；TP=4 43.7/279/11,432；TP=8 60.0/384/15,700；int8 TP=2 43.7/279/11,432；int4 TP=1 42.6/273/11,159

# 表 4：机制、分流与存量硬件
print(f"chunked prefill：1.2k {pre(I)/P_EFF*1e3:.1f} → {pre(512)/P_EFF*1e3:.1f} ms；4k {pre(4096)/P_EFF*1e3:.1f} → {pre(512)/P_EFF*1e3:.1f} ms")
print(f"prefix 800×40%：省 320 token、{QPS*86400*pre(320)/P_EFF*NODE/3600:.2f} GPU·h/天；10 个变体 KV {10*800*K70/KW:.2f} GiB")
node_8b = (pre(I, N7, L7, D7) / (BASE * MFU) + O / (64 / ((W7 + 64 * 4096 * K7) / BW1))) / NODE
thr8 = NODE * 64 / ((W7 + 64 * 4096 * K7) / BW1)
print(f"8B：单卡 {64/((W7 + 64*4096*K7)/BW1):,.0f}、节点 {thr8:,.0f} tok/s（{thr8/node_dec:.2f}×）；"
      f"节点·秒 {node_8b*1e3:.2f} ms（{node_s/node_8b:.2f}×）；分流 {(0.3*node_s + 0.7*node_8b)*PK:.2f} 节点、"
      f"级联 {(0.75*node_8b + 0.25*(node_8b + node_s))*PK:.2f} 节点")
A100, BWA = 312e12, 2.039e12
print(f"A100 7B：KV 池 {80 - W7/GB/1.074 - 2.5 - 1.25:.1f} GiB；batch 68 {68/((W7 + 68*0.5*KW)/BWA):,.0f}、"
      f"batch 125 {125/((W7 + 125*0.5*KW)/BWA):,.0f} tok/s；prefill {I/(pre(I, N7, L7, D7)/(A100*MFU)):,.0f} 输入 tok/s"
      f"（2 卡 = {2*I/(pre(I, N7, L7, D7)/(A100*MFU))/36000:.0%} 峰值）")
Wmix = 2 * 46.703e9
print(f"Mixtral：权重 {Wmix/GB:.1f} GB；prefill 按激活参数 {pre(I, 12.88e9, 32, 4096)/pre(I):.2f}×；TP=8 节点 {plan(Wmix, 8, K7)[2]:,.0f} tok/s、TP=2 四副本 {plan(Wmix, 2, K7)[2]:,.0f} tok/s")
# 表 4：53.7 → 22.8 ms、188.1 → 22.8 ms；9.47 GPU·h/天、2.44 GiB；8B 4,390/35,118（2.24×）、14.15 ms（5.14×）、0.95/0.97 节点；A100：62.8 GiB、2,719、3,124、8,434 tok/s（2 卡 47%）；Mixtral：93.4 GB、0.18×、42,800、21,447 tok/s
```

与来源对照：PagedAttention 的「同延迟下吞吐 2–4×、支持跨请求 KV 共享」、chunked prefill 的「切块 + 无停顿调度、2.6–5.6× 服务容量」、P/D 分离的「90% 达标下 7.4× 请求或收紧 12.6× SLO」、SGLang 的「压缩 FSM、多负载最高 6.4×」、RouteLLM 的「部分设置下 2× 以上」与 FrugalGPT 的「最多 98% 降本或同成本 +4% 准确率」都来自论文摘要口径，各自的实验条件不同，不能互相折算，也不能直接当线上收益。Pope et al. 的 int8 权重结果（540B、低 batch 29 ms/token、大 batch 76% MFU）是 TPU v4 与 2,048 token 上下文的实验，本题只借用「权重可以压到 8 bit」这个结论，不借用数字。表 1 至表 4 的节点数、并发、美元与功耗全部按表内假设自行复算：源文没有银行规模的流量模型，也没有把利用率折成单位成本的口径。

## 常见追问

- **追问**：为什么不能用一个承诺零保留（ZDR）的外部 API？合同上写清不落盘不就行了吗？
  - 要点：被禁止的是「外部处理方」这个角色，不是「保留行为」。数据离开行内边界就产生 GDPR 第 28 条的处理者关系（需要处理者协议、子处理者清单与审计权）、跨境传输评估，以及 DORA 下的第三方集中度与退出计划义务。这些是**架构与治理问题**，合同只能约束行为、不能消除角色。所以方案里连「只把失败样本发出去排障」这种运维变体都不能有。
- **追问**：量化之后怎么证明质量没下降？
  - 要点：任务级评测集优先——信贷摘要的字段级准确率、抽取任务的 schema 合法率、拒答与人工升级率、以及红线集（合规话术、注入、越权检索）。困惑度与通用 benchmark 只能当烟雾报警器，不能当验收证据（[[evaluation-04]]）。流程必须是「量化 → 离线评测 → 影子流量比对 → 金丝雀 → 全量」，任一步不过就回滚到 bf16，而 bf16 的显存账在采购时就要留出位置。
- **追问**：客户手上只有 2 张 A100-80G，且卡间没有 NVLink，怎么办？
  - 要点：不要跨卡做张量并行——无 NVLink 时每层两次 all-reduce 会吃掉大部分收益。落点是「单卡能放下的 7B 级模型 + 量化权重」，用副本扩吞吐而不是用 TP 扩单卡容量：Mistral 7B bf16 权重 14.5 GB，sliding window 4096 让每序列 KV 封顶 512 MiB，单卡 4k 可跑 100 条以上；吞吐上按 TPOT SLO 定 batch（25 ms 时 68 条、2.7k tok/s/卡）。要更大模型就只能上量化 70B 单卡（INT4 约 36 GB、产能约 11k tok/s/节点），并接受质量验收。
- **追问**：峰值是均值的 8.6 倍，为什么不多买卡或者干脆按峰值配？
  - 要点：按峰值配就是这题的成本陷阱——买卡按峰值、用卡按均值，利用率只有 6–8%，单位 token 成本涨一个数量级。正确做法是三件事一起上：按峰值 + 冗余买（本题 3 节点）、用分流与限流把峰值削下来、把夜间闲置算力排给批处理与评测。真正的杠杆是利用率，不是卡数。
- **追问**：气隙环境里怎么升级模型或打安全补丁？
  - 要点：走变更流程而不是运维顺手：隔离区带入并验签、出 SBOM、内部 registry 打 tag；蓝绿要同时容纳两份权重（存储与显存都双份，3 个节点里留 1 个做新版本池）；model / prompt / engine 三个版本号分别灰度与一键回滚；离线评测集与回归门禁必须在无外网时能跑完；CVE 响应订阅离线源并 SLA 化，因为权重冻结后没有上游补丁通道。
- **追问**：为什么强调缓存命名空间？银行内部不都是同一个数据域吗？
  - 要点：银行的「同一个机构」内部分成部门、业务线、法人与权限域，前缀命中是**文本级复用**，而授权是**人的属性**。缓存 key 必须带租户/部门、权限域、模型与权重版本、prompt 版本；系统提示这类与用户无关的共享前缀可以全局复用，用户历史与检索结果只能命中自己的命名空间，否则就是 [[system-design-09]] 第 4 节的缓存串租户——在银行里这是一起数据保护事件，不是性能 bug。

## 相关题目

- [[system-design-09]]：LLM gateway 的路由、故障转移、缓存与限流。本题是它的反命题——供应商只剩自己时，gateway 退化成配额、审计与降级控制面。
- [[system-design-10]]：消费级聊天助手的服务栈。prefill/decode 两条产能、$b_{\max}$ 与利用率口径全部沿用该题；差别在规模（10 万 QPS vs 30 QPS）与「有没有外部兜底」。
- [[system-design-01]]：企业 RAG 的权限隔离与容量账。行内检索、ACL 前置裁剪与评测分层直接搬到本题。
- [[inference-serving-08]]：70B 的显存拆解（权重、KV、工作区、碎片）。第 4 节的每卡 KV 池与并发上限由它推导。
- [[inference-serving-05]]、[[inference-serving-14]]、[[inference-serving-15]]：prefix caching 的 exact-prefix 规则、chunked prefill 的 token budget 与 P/D 分离的适用条件，共同决定「固定卡数下能多接多少请求」，也是 3 节点规模上先不分离的判据。
- [[inference-serving-06]]、[[inference-serving-11]]：量化选型与引擎选型（vLLM / SGLang / TensorRT-LLM / 自研）在行内自持约束下的取舍；[[evaluation-04]]、[[evaluation-08]]：离线回归门禁与 prompt/模型版本回滚，是气隙里唯一的质量防线；[[rag-08]]、[[safety-07]]：权限感知检索与 PII 脱敏——检索是权限出口，日志与 trace 是数据出口。

## 参考资料与归属

1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（延伸），Kwon et al.（SOSP 2023），2023-09-12。摘要口径：PagedAttention 借鉴虚拟内存分页，把 KV cache 碎片降到接近零并支持请求内与跨请求共享，vLLM 在同延迟下把吞吐提升 2–4 倍。第 5 节的 continuous batching 与 KV 共享收益来自该文；第 4 节的并发上限也依赖这类分页实现。
2. [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)（延伸），Pope et al.，2022-11-09。摘要与正文口径：在 540B 模型上用 int8 权重量化，低 batch 达到 29 ms/token、大 batch 达到 76% MFU；多查询注意力让上下文长度可扩大 32 倍。实验环境是 TPU v4 与 2,048 token 上下文，第 4 节只借用「权重可压到 8 bit」的结论并明确不外推数字。
3. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)（延伸），Zhong et al.（OSDI 2024），2024-01-18。摘要口径：以 goodput 替代吞吐做验收，在 90% 以上请求满足延迟约束时可服务 7.4 倍请求，或把 SLO 收紧 12.6 倍。第 5 节 P/D 分离的收益与其适用条件来自该文。
4. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)（延伸），Agrawal et al.，2024-03-04。摘要口径：chunked prefill 把长 prompt 切块并与 decode 混批，配合无停顿调度；2.6×（Mistral-7B 单 A100）、3.7×（Yi-34B 双 A100）、5.6×（Falcon-180B 流水并行）的服务容量提升。第 5 节的 ITL 台阶与 token budget 来自该文。
5. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（延伸），Zheng et al.，2023-12-12。RadixAttention 做 KV 复用、压缩有限状态机加速结构化输出，多类负载吞吐最高 6.4×。第 5 节的结构化抽取与 prefix 复用机制来自该文。
6. [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665)（延伸），Ong et al.，2024-06-26。摘要口径：用偏好数据训练路由器在强/弱模型间动态选择，部分设置下把成本降低 2 倍以上且不牺牲质量，并具备换模型对的迁移性。第 6 节的学习式路由与迁移性来自该摘要，属论文 benchmark 口径。
7. [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)（延伸），Chen、Zaharia、Zou，2023-05-09。摘要口径：把降本手段分为提示适配、模型近似、LLM 级联三类；级联可匹配最优单模型表现并把成本最多降低约 98%，或同成本下提升约 4% 准确率。第 6 节的级联结构与数字来自该摘要。

工程细节另参考三处可核验的官方文档：vLLM 的 [Usage Stats Collection](https://docs.vllm.ai/en/stable/usage/usage_stats/)（匿名用量统计默认开启，可用 `VLLM_NO_USAGE_STATS=1`、`DO_NOT_TRACK=1` 或 `~/.config/vllm/do_not_track` 关闭，并用 `~/.config/vllm/usage_stats.json` 核对）；huggingface_hub 的[环境变量文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)（`HF_HUB_OFFLINE`、`HF_HUB_DISABLE_TELEMETRY`、`DO_NOT_TRACK`、`HF_HUB_DISABLE_UPDATE_CHECK`）；Mistral 的模型卡与 config（[Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)、[Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)）：Apache 2.0 许可证、Mistral 7B 的 GQA 与 sliding window 4096、Mixtral 的 8 专家 top-2 配置，第 1、4 节的许可证与架构常数来自这些模型卡。

除上述来源外，表 1 至表 4 的节点数、显存、并发、美元与功耗均为按表内假设自行计算：来源论文没有银行规模的流量模型，也没有把峰值采购与均值负载折算成单位成本的算例。本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
