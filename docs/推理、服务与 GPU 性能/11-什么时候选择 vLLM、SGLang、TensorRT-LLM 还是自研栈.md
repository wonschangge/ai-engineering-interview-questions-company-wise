---
type: question
id: inference-serving-11
topic: 推理、服务与 GPU 性能
order: 11
question: 什么时候你会选择 vLLM、SGLang、TensorRT-LLM，还是自研技术栈？
question_en: When would you choose vLLM, SGLang, TensorRT-LLM, or build your own stack?
asked_at: [NVIDIA, Together AI]
level: 进阶
tags: [vllm, sglang, tensorrt-llm, 选型]
sources:
  - title: How does vLLM work?
    url: https://outcomeschool.com/blog/how-does-vllm-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: How does SGLang work?
    url: https://outcomeschool.com/blog/how-does-sglang-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: How does TensorRT-LLM work?
    url: https://outcomeschool.com/blog/how-does-tensorrt-llm-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: SGLang: Efficient Execution of Structured Language Model Programs（延伸）
    url: https://arxiv.org/abs/2312.07104
    author: Zheng et al. (RadixAttention)
    published: 2023-12-12
related: [inference-serving-02, inference-serving-03, inference-serving-05]
updated: 2026-09-28
---

## 一句话答案

> 三者不是同一个引擎的三个品牌，而是三种不同的优化着力点：vLLM 把显存分配与服务调度做成通用能力（PagedAttention + continuous batching + prefix caching），模型与硬件覆盖最广；SGLang 把「LLM 程序」当成一等公民（RadixAttention 做 token 级前缀树复用、压缩有限状态机做结构化输出），在多轮、agent、JSON 密集的负载上占优；TensorRT-LLM 走编译路线，把模型编成针对特定 GPU 的 engine，单机性能上限通常最高，代价是构建时间、模型覆盖与调试难度。
> 默认从 vLLM 起步；负载有强前缀复用或大量结构化输出时评估 SGLang；模型固定、全 NVIDIA、单位成本决定生死时上 TensorRT-LLM；只有当负载形态或硬件组合明显偏离通用引擎的假设、并且手里有 kernel 与调度团队时，才在开源引擎上二次开发，完全自研是最后的选项。

## 面试官在考什么

- 能否把三个引擎的**机制差异**讲清楚，而不是背「vLLM 快、SGLang 更快、TensorRT-LLM 最快」这种榜单式结论。
- 是否知道每个机制的**收益边界**：同一个引擎在什么负载上收益大、在什么负载上几乎没有收益。
- 有没有真实的选型方法论：先定义 SLO 与负载画像，再用同一套压测脚本比 goodput 与 p99，而不是拿公开 benchmark 当结论。
- 是否理解自研的**成本结构**：需要哪些能力、收益从哪来、什么时候为正、维护成本如何吃掉性能红利。
- 能否承认不确定性：新模型上线速度、社区迭代节奏、license 与商业支持，这些非性能因素在真实决策里权重很高。

常见错误答案：

- 只按「吞吐排名」回答，说不出 SGLang 的 6.4× 与 vLLM 的 2–4× 各自的对比口径，把论文里特定任务上的最大倍数当成通用倍数。
- 把 TensorRT-LLM 的优势归结为「NVIDIA 的库所以快」，讲不出编译期 kernel 融合、自动调优与 in-flight batching 这三件事分别省掉了什么。
- 一上来就说自研，「可以更好地控制性能」，但说不出自研要在哪一项上比开源引擎强、强多少、值多少张卡。

## 原理与推导

### 1. 技术定位：三种优化着力点

| 维度 | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| 核心机制 | PagedAttention（block 级 KV 管理）+ continuous batching + prefix caching | RadixAttention（radix tree 前缀复用）+ 压缩 FSM 结构化输出 + 前端 DSL | 编译式：全图融合 + kernel 自动调优 + 自定义 attention kernel + in-flight batching + CUDA graph |
| 抽象层次 | 通用推理引擎 | 「LLM 程序」的执行运行时 | 面向 NVIDIA GPU 的编译产物 + 服务框架 |
| 硬件范围 | 多厂商（NVIDIA、AMD、以及多种加速器） | 多厂商，NVIDIA 上的工程投入最大 | 仅 NVIDIA |
| 模型覆盖速度 | 最快，新架构通常数天内进主干 | 快，主流开源架构跟进及时 | 主流架构覆盖好，冷门/最新架构可能要自己适配 |
| 交付形态 | 直接加载权重即可服务，OpenAI 兼容 API | 直接加载权重即可服务，OpenAI 兼容 API + DSL | 经典路径先 build 出 engine，再经内置 server 或 Triton / NIM 提供服务 |
| 主要代价 | 单个 kernel 的极限性能不如专门编译的产物 | 生态与生产案例比 vLLM 短 | 构建时间、产物与 GPU/精度/长度上限绑定、调试链路长 |

机制差异中最值得讲的是「复用的粒度」。prefix caching 在 vLLM 里以 block（默认 16 token）为粒度，不足一个 block 的尾部无法命中；RadixAttention 直接在 token 序列上建树、按 LRU 从最久未用的叶子开始逐层淘汰（祖先节点只要仍被共享就不会回收），因此长对话里「每一轮只多几句话」这种增量式共享能被完整捕获。反过来说，当所有请求共享一段完全相同的长 system prompt 时，两者的差别只有一个 block 之内，几乎可以忽略——这是判断「该不该为前缀复用换引擎」的关键分界线，量化见「数值与代码验证」表 3。

结构化输出两边都支持，差别在实现效率。SGLang 用压缩有限状态机把语法约束编译成每步允许的 token 集合，遇到语法已经确定的片段可以直接跳过采样，而不是逐个 token 生成。JSON、regex、固定 schema 占比高的负载（信息抽取、工具调用、agent 的输出解析）因此受益，收益量级取决于输出里「可跳过片段」的比例，而不是一个固定倍数。

### 2. 从机制推到「什么时候有效」

- **vLLM 的收益来自显存与调度，因此随并发数 × 序列长度放大**。请求短、并发低时它和朴素实现差别不大；一旦进入长上下文、高并发，显存碎片与 over-reservation 就成了吞吐上限，收益随之显现。论文口径：相比 FasterTransformer、Orca 这类当时的 SOTA 系统，在同等延迟水平下吞吐提升 2–4×，且序列越长、模型越大、解码算法越复杂（beam search 等）提升越明显。
- **SGLang 的收益来自复用与约束解码，因此随「共享前缀占 prompt 的比例」和「结构化输出占比」放大**。论文口径：在 agent control、逻辑推理、few-shot、JSON decoding、RAG pipeline、多轮对话这些任务上，跨多种语言与多模态模型，最高达到 6.4× 吞吐提升，对比基线是当时的 SOTA 推理系统。这是这些任务上的最大倍数，不是任意负载的通用倍数。
- **TensorRT-LLM 的收益来自编译期信息**。构建时它能看到整张图，于是可以把连续的小算子融合成一个 kernel（数字只进出 HBM 一次）、在真实 GPU 上把多个候选 kernel 各跑一遍留最快的那个、并按 GPU 代际选择 attention kernel。这些优化与负载无关，属于「同样硬件上把天花板抬高一点」，所以它通常在单机极限性能和单位成本上领先；代价是提前编译、产物与硬件绑定。
- **量化路径的可用性也是选型项**。FP8 从 Hopper 起就有原生 Tensor Core 路径，H100 的 FP8 稠密算力是 bf16 的 2 倍；FP4 属于 Blackwell 一代的原生格式。更早的卡上做 4 bit 通常走 weight-only INT4（AWQ/GPTQ），在 kernel 里反量化回 bf16 计算，省的是显存与带宽而不是算力。同一个模型在不同引擎上支持的 dtype 集合不同，这会直接改变「一张卡能放多少并发」。

### 3. 选型判断维度

| 维度 | 要问的问题 | 偏向 |
| --- | --- | --- |
| 模型支持与更新速度 | 新架构上线要几天？要不要自己写 modeling 代码？ | 迭代快 → vLLM |
| 硬件 | 只有 NVIDIA，还是混插多厂商？ | NVIDIA 专用 → TensorRT-LLM |
| 结构化输出与控制流 | JSON/regex 占比多少？是否需要多步程序编排？ | 密集 → SGLang |
| 多 LoRA / 多租户 | 一个基座要挂多少个适配器？是否需要按租户隔离？ | 适配器多 → vLLM / SGLang |
| 量化与 dtype | 需要 FP8？FP4？还是只能 weight-only INT4？ | 最激进的低比特 → TensorRT-LLM |
| 编译与冷启动 | 能否接受发布流程里多一个 build 步骤？扩缩容要不要等编译？ | 不能等 → vLLM / SGLang |
| 可观测性与调试 | 出问题时能否下钻到 kernel 级？日志与指标是否够用？ | 排障深度 → TensorRT-LLM + Nsight，日常排障 → vLLM |
| 基础设施集成 | 现有网关、K8s、Triton、监控栈是什么？ | 已有 Triton → TensorRT-LLM |
| 人力与社区 | 团队有几个人能读 kernel？出问题找谁？ | 人力薄 → 选生态最大的 |
| 商业支持 | 生产事故的 SLA 归谁？ | 需要兜底 → 有商业支持的发行版 |

### 4. 决策规则

1. **通用聊天 / API 服务、模型迭代快** → vLLM 起步。它是默认项：上手成本最低，模型覆盖最快，出问题时能搜到的资料最多。
2. **大量共享前缀（多轮对话、few-shot、agent 固定提示）、结构化输出密集、需要跨请求的前缀树复用** → SGLang。判断依据是「共享前缀 token 数 / 总 prompt token 数」这个比值，以及输出里受语法约束的比例。
3. **模型固定、全栈 NVIDIA、追求单机极限性能与最低单位成本、能接受编译与工程投入** → TensorRT-LLM。判断依据是「同一个模型会不会半年不变」和「GPU 规模是否大到 10% 的性能差就等于若干张卡」。
4. **有特殊需求（自定义 attention、非标准架构、与自研调度/框架深度耦合、极致成本控制）** → 先看开源引擎的插件接口与自定义 kernel 能力，在其上做二次开发；完全自研放在最后。

这四条规则背后是同一个判断：**换引擎的收益 = 负载相对通用假设的偏离程度 × 规模**。偏离小、规模小，收益就被迁移与运维成本吃掉。

### 5. 自研的门槛与收益来源

自研需要的能力清单，任何一项缺失都会让「自研更快」变成空话：

- **kernel 层**：CUDA / Triton 编程，attention（含变长、paged、chunked）、GEMM 与融合算子的实现与调优。
- **显存管理**：block 分配器、前缀复用、水位与抢占策略、碎片控制。
- **调度**：continuous batching、chunked prefill、优先级与超时、prefill/decode 配比。
- **量化**：校准流程、per-channel/per-group 粒度、低比特 kernel 与精度回归。
- **分布式**：TP / PP / EP 的切分与通信重叠、跨机拓扑感知。
- **质量体系**：压测、goodput 与 p99 回归、精度回归、灰度与回滚。

收益来自哪里：只有当你**必须**做通用引擎假设之外的事时才有正收益，例如自定义 attention 结构、非标准 KV 布局、专用硬件上的定制 kernel、或者把推理引擎与自家调度/路由深度耦合以避免一次网络跳转。反之，通用引擎已经覆盖的场景里自研只会更慢——上游每周都在提交 kernel 与调度改进，自研团队要跑赢的是整个社区。

中间路线几乎总是更优：**fork 或扩展开源引擎**。vLLM 与 SGLang 都提供自定义 kernel、自定义模型注册、插件式调度钩子，把精力集中在真正的差异点上，其余部分继续吃上游红利。代价是要维护一条与上游持续 merge 的分支，这笔账要在立项时就算清楚。

### 6. 可验证的选型流程

1. **定义 SLO**：TTFT、TPOT、p99、可用性各是多少。没有 SLO 就没有「够快」。
2. **刻画负载**：输入/输出长度分布（不是均值，要 p50/p95/p99）、并发曲线、共享前缀占比、输出结构化比例、多模态比例、是否需要多 LoRA。
3. **固定变量压测**：同一批 trace（生产回放或合成）、同一硬件、同一模型与精度、同一 SLO 判定标准，在候选引擎上跑。
4. **比 goodput 而不是峰值吞吐**：在满足 p99 SLO 的前提下，每秒完成多少请求 / 多少 token，再折算成「每百万 token 需要多少 GPU 小时」。
5. **加上工程成本**：迁移工时、发布流程变化、排障能力、模型上线延迟、license 与支持合同。
6. **小流量灰度后再全量**：两个引擎的输出在温度 0 下不一定逐 token 一致（kernel 与 batching 会改变数值累加顺序），灰度期间要盯质量指标。

需要强调：公开 benchmark 的结论在别人的模型、别人的长度分布、别人的 GPU 上成立，只能用来筛掉明显不合适的候选，不能用来做最终决策。

## 数值与代码验证

以 LLaMA-3-70B（80 层、64 个 attention 头、8 个 KV 头、head_dim 128、bf16）与 H100 SXM（80 GB HBM3、3.35 TB/s、bf16 稠密 989.4 TFLOP/s、FP8 稠密 1978.9 TFLOP/s）为参照复算。容量统一按二进制单位：1 GiB = 1024 MiB。

**表 1：选型时先要记住的常数**

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 70B bf16 权重 | $70\times10^9 \times 2$ B | 140 GB = 130.4 GiB |
| H100 单卡容量 | $80\times10^9$ B | 74.5 GiB |
| 单卡能否放下 70B bf16 | 140 GB vs 80 GB | 放不下，至少要 TP=2 |
| 每 token KV（GQA-8） | $2\cdot 80\cdot 8\cdot 128\cdot 2$ B | 320 KiB |
| 32k 上下文单条会话 | $32768 \times 320$ KiB | 10 GiB |
| batch=1 decode 单步时间下限（只算读取权重） | $140\text{ GB} / 3.35$ TB/s | 41.8 ms/token，吞吐上限约 23.9 token/s |
| roofline 拐点 | $989.4 / 3.35$ | 295 flops/byte |
| FP8 相对 bf16 峰值 | $1978.9 / 989.4$ | 2.0× |

**表 2：2×H100（TP=2）上 32k 会话的并发上限**

| 项 | 数值 |
| --- | --- |
| 两卡总容量 | 149.0 GiB |
| 扣掉 70B bf16 权重 | 剩 18.6 GiB |
| 可容纳 KV token 总数 | 61,035 |
| 折算 32k 会话数 | 约 1.9 条（未计激活与碎片） |

这张表说明「引擎的显存管理不是优化项而是准入项」：2 卡 TP 下全部余量只够一条半的 32k 会话，over-reservation 与碎片会直接把可用并发压到 1 条。

**表 3：两个引擎机制的量化边界**

| 场景 | 计算 | 结果 |
| --- | --- | --- |
| 按 max_len=2048 预留、实际输出 50 token | $(2048-50)/2048$ | 浪费 97.6% |
| 按 block=16 分配、实际 50 token | 分配 64 槽 | 内部碎片 21.9% |
| 按 block=16 分配、实际 1000 token | 分配 1008 槽 | 内部碎片 0.79% |
| block 粒度前缀复用的最坏损失 | $16-1$ | 最多 15 个 token 的 prefill 无法复用 |
| 1500 token 前缀的 KV，单份 | $1500 \times 320$ KiB | 468.75 MiB |
| 100 条并发各自一份 vs 共享一份 | $100 \times 468.75$ MiB − 468.75 MiB | 省 45.3 GiB（99%） |

表 3 的最后一行就是「要不要为前缀复用换引擎」的判据：SGLang 相对 block 粒度方案多省下来的那部分，只对应「不足一个 block 的尾部」，也就是每次命中边界最多 15 个 token；而共享前缀本身带来的 45.3 GiB 收益，vLLM 的 prefix caching 同样拿得到。**真正决定收益的是业务有没有可共享前缀，不是选了哪个引擎。** 反过来，如果负载是频繁的多轮增量对话，15 token 的损失会在每一轮重复一次，累积起来才值得计入。

**表 4：自研的 break-even 口径**

| 项 | 计算 | 结果 |
| --- | --- | --- |
| 自研相对开源引擎提升 20%，集群 64 卡 | $64 \times 0.20$ | 等效省下 12.8 张卡 |
| 编译器路线要维护的 engine 组合 | 3 GPU 型号 × 2 精度 × 2 长度上限 | 12 份产物 |
| LoRA rank 16（全部线性层）适配器 | 约 207 M 参数、bf16 | 约 414 MB，占基座 0.30% |

表 4 第一行是立项门槛：只有这 12.8 张卡的年化成本（含折旧、电、机房）超过自研团队的年成本，性能红利才可能为正；还没算上质量回归、灰度与持续 merge 上游的投入。第二行说明编译器路线的隐性成本随硬件与精度矩阵组合增长。第三行说明适配器很便宜，多租户场景不应该成为放弃通用引擎的理由。

```python
GB, GiB, KiB, MiB = 10**9, 1024**3, 1024, 1024**2

H100_BW = 3.35e12          # H100 SXM HBM3 带宽
H100_BF16 = 989.4e12       # H100 SXM bf16 稠密 Tensor Core
H100_FP8 = 1978.9e12       # H100 SXM fp8 稠密 Tensor Core
CAP = 80 * GB              # 80 GB HBM3（十进制）

# LLaMA-3-70B：80 层、64 个 attention 头、8 个 KV 头、head_dim 128、ffn 28672
L, HQ, HKV, D, HID, FFN = 80, 64, 8, 128, 8192, 28672
W = 70e9 * 2                                   # bf16 权重字节数
KV = 2 * L * HKV * D * 2                       # 每 token KV 字节数

print(W / GB, W / GiB)                         # 140.0  130.39
print(CAP / GiB, W < CAP)                      # 74.51  False -> 单卡放不下
print(KV / KiB)                                # 320.0 KiB/token
print(KV * 32768 / GiB)                        # 10.0 GiB @32k

free = 2 * CAP - W                             # 2 卡 TP 的余量
print(free / GiB, free / KV, free / KV / 32768)
# 18.61 GiB, 61035 token, 1.86 条 32k 会话

print(W / H100_BW * 1000, H100_BF16 / H100_BW)
# batch=1 时 41.79 ms/token -> 23.9 token/s; roofline 拐点 295 flops/byte

# 前缀复用的显存账：1500 token 前缀, 100 并发
one = 1500 * KV
print(one / MiB, 100 * one / GiB, 99 * one / GiB)
# 468.75 MiB / 份, 45.78 GiB / 100 份, 省下 45.32 GiB

# block 粒度前缀复用的最坏损失
print(16 - 1)                                  # 15 个 token 的 prefill

# LoRA rank 16 挂在全部线性层上：q/k/v/o + gate/up/down
lora_attn = 2 * 16 * (HID + HID) + 2 * 16 * (HID + HKV * D)
lora_mlp = 3 * 16 * (HID + FFN)
lora = L * (lora_attn + lora_mlp)
print(lora / 1e6, lora * 2 / GB, lora * 2 / W)
# 207.09 M 参数; bf16 0.414 GB, 占 70B 基座的 0.30%

# 自研 break-even
print(64 * 0.20)                               # 等效省下 12.8 张卡
```

与源文对照：三篇博客给的是机制与定性结论，没有给出本节的容量与 roofline 数字，这里按 LLaMA-3-70B 与 H100 SXM 的公开参数自行复算；判断「该不该换引擎」的量级结论（表 3 与表 4）也是自行推导。论文口径的数字（vLLM 2–4×、SGLang 最高 6.4×）只引用摘要原文，不改写口径、不外推。

## 常见追问

- **追问**：TensorRT-LLM 在某些场景为什么更快？
  - 要点：三件事。编译期能看到整图，把连续小算子融合成一个 kernel，数字只进出 HBM 一次，省的是带宽而不是算力；构建时在真实 GPU 上给每个算子试跑多个候选 kernel 并计时留最快的，这是运行期框架做不到的；attention 由 NVIDIA 按自家 GPU 代际手写，prefill 与 decode 各有一套。此外 CUDA graph 把整段 decode 序列录制成一次提交，消除了 launch 开销与 CPU 侧的间隙。这些收益与负载无关，所以它抬高的是天花板。
- **追问**：什么情况下换引擎收益最大？
  - 要点：三类负载最值得迁移。长上下文（显存管理成为准入条件，表 2 那种「只够一条半会话」的处境）、高共享前缀（多轮对话、agent、few-shot，表 3 的 45.3 GiB）、结构化输出密集（JSON、工具调用，压缩 FSM 可跳过确定片段）。反过来说，短请求、低并发、输出自由文本的服务，换引擎的收益通常小于迁移成本。
- **追问**：多引擎并存的运维代价有多大？
  - 要点：成本不在多部署一个二进制，而在每一层都要乘二。指标语义要对齐（goodput 的定义、TTFT 的起点、p99 的统计窗口），量化与校准流程各一套，灰度和回滚各一套流程，值班手册要覆盖两套失败模式，升级节奏要与上游两个 release 周期对齐。此外路由层必须知道「哪些请求可以发给哪个引擎」，否则前缀亲和性被打散，前缀复用的收益直接归零。
- **追问**：vLLM 和 SGLang 现在功能越来越像，边界在哪？
  - 要点：两者都在补齐对方的能力（vLLM 有 prefix caching 与 guided decoding，SGLang 有 continuous batching 与 OpenAI 兼容 API）。可复现的差异只剩三点：前缀复用粒度（block vs token 级 radix tree）、结构化解码的实现效率（压缩 FSM vs 通用语法约束）、以及 SGLang 的前端 DSL 对多步程序的表达能力。选择时按这三点对负载的重要性排序，其余维度按生态与人力选。
- **追问**：什么时候真的该自研？
  - 要点：只在「开源引擎的结构性假设与你的负载冲突」时。典型信号：自定义 attention 或 KV 布局、非标准模型结构、专用硬件、要把引擎嵌进自研调度器以避免额外跳转、或者单位成本已经由推理决定而 10% 的性能就是财务问题。即便这时也优先 fork 与插件化，保住持续 merge 上游的能力。
- **追问**：选型时怎么避免被 benchmark 误导？
  - 要点：先确认口径——对比的是峰值吞吐还是 SLO 内的 goodput，是单卡还是整机，精度与量化配置是否相同，长度分布是否与自己一致，是否开了前缀缓存，输出是否受语法约束。然后用生产 trace 回放，在满足 p99 的前提下比 goodput，最后折算成每百万 token 的 GPU 小时。

## 公司变体

- **NVIDIA**：偏工程实现与系统深度。TensorRT-LLM 出自该公司，公开材料集中在 engine 构建、kernel 融合与自动调优、in-flight batching、FP8/FP4 的低比特路径、CUDA graph 以及经 Triton / NIM 的部署形态；回答时把「编译期能做什么、运行期做不到什么」讲透，比背对比结论有效。
- **Together AI**：偏工程与单位成本。以多引擎、多硬件的推理服务运营为背景，关注同一模型在不同引擎/硬件上的 goodput 与成本曲线、按负载切换引擎、前缀缓存命中率对成本的影响，以及大规模集群下的调度与容错。回答时给出「每百万 token 的 GPU 小时」这类可比较的口径更容易对齐。

以上依据两家公开技术材料与岗位方向的侧重判断，不代表具体面试流程。

## 相关题目

- [[inference-serving-02]]：continuous batching 的原理，解释三个引擎共有的调度基础，也是「换引擎收益」的公共部分。
- [[inference-serving-03]]：PagedAttention 与 block 级显存管理，解释前缀复用粒度为 16 token 这一约束的来源。
- [[inference-serving-05]]：prefix caching 与 prompt caching，直接决定表 3 那 45.3 GiB 收益能不能落到自己负载上。

## 参考资料与归属

1. [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work)，Amit Shekhar（Outcome School）。提供 PagedAttention 的 block 分配与 block table、continuous batching 与 static batching 的对比、over-reservation 与碎片的定义、OpenAI 兼容 API 的定位，以及「需要多少显存才能服务多少并发」的问题框架。
2. [How does SGLang work?](https://outcomeschool.com/blog/how-does-sglang-work)，Amit Shekhar（Outcome School）。提供 RadixAttention 的 radix tree 共享与「最久未用的分支先淘汰」、前端 DSL 的多步程序写法、constrained decoding（只允许保持格式合法的 token）与「已确定的片段可以快速填充」，以及 vLLM 与 SGLang 在复用粒度与生态成熟度上的对比；「compressed finite state machine」这一术语与实现出自下一条论文条目。
3. [How does TensorRT-LLM work?](https://outcomeschool.com/blog/how-does-tensorrt-llm-work)，Amit Shekhar（Outcome School）。提供编译式路线（build → engine）、kernel 融合与 kernel auto-tuning、CUDA graph、in-flight batching、低比特量化（8 bit / 4 bit）与校准、TP/PP 切分、Triton 与 NIM 的服务路径，以及 PyTorch backend 对经典 build 路径的替代。

4. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（延伸），Woosuk Kwon 等（vLLM，SOSP 2023）。「原理与推导」中「从机制推到什么时候有效」一节的 vLLM 论文口径（相同延迟下相比 FasterTransformer、Orca 提升 2–4×，且长序列、大模型、复杂解码算法下更明显）与「近零 KV 显存浪费、支持请求内与跨请求共享」的结论取自该摘要。
5. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（延伸），Lianmin Zheng 等。「原理与推导」中「从机制推到什么时候有效」一节的 SGLang 论文口径（RadixAttention + 压缩有限状态机，在 agent control、逻辑推理、few-shot、JSON decoding、RAG、多轮对话等任务上跨多种模型最高 6.4× 吞吐）取自该摘要。

表 1 至表 4 的容量、roofline 与成本数字均按 LLaMA-3-70B 与 H100 SXM 公开参数自行复算，未取自上述资料；「FP8 自 Hopper 起有原生 Tensor Core 路径、FP4 属于 Blackwell 一代」属于硬件代际的通用事实，三篇博客只讲到「8 bit / 4 bit 等更小的数值格式」，未指定具体代际与格式名。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
