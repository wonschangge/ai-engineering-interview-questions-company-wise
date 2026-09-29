---
type: question
id: anthropic-17
company: Anthropic
topic: inference-serving
order: 17
question: 为 Claude 级别的 LLM API 设计服务栈。在不破坏 p99 latency 的前提下最大化 GPU 利用率。
question_en: Design the serving stack for a Claude-class LLM API. Maximise GPU utilisation without breaking p99 latency.
asked_at: []
level: 高阶
tags: [系统设计, 服务栈, p99, 利用率, 准入控制, 降级]
sources:
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）
    url: https://arxiv.org/abs/2403.02310
    author: Agrawal et al. (Microsoft, OSDI 2024)
    published: 2024-03-04
  - title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving（延伸）
    url: https://arxiv.org/abs/2401.09670
    author: Zhong et al. (OSDI 2024)
    published: 2024-01-18
  - title: SGLang：Efficient Execution of Structured Language Model Programs（延伸）
    url: https://arxiv.org/abs/2312.07104
    author: Zheng et al. (RadixAttention)
    published: 2023-12-12
  - title: How does Prompt Caching work?
    url: https://outcomeschool.com/blog/how-does-prompt-caching-work
    author: Amit Shekhar (Outcome School)
    published: 
related: [anthropic-16, anthropic-15, system-design-10, inference-serving-05, system-design-09]
updated: 2026-09-28
---

## 一句话答案

> 这道题的核心矛盾是一句话：**利用率与 p99 是对立的**——把 GPU 喂饱需要大 batch 与长队列，而 p99 要求小 batch 与短队列。**可解的方式不是折中，而是分层与准入**：
> ① **接入层**：鉴权、配额、按模型/长度/优先级**路由**、多区域就近接入、请求去重与幂等；
> ② **调度层**：**准入控制（admission control）**——队列长度与预估等待超过阈值就**快速拒绝或降级**，而不是让请求排到超时；优先级队列保证交互式请求先走；
> ③ **推理层**：连续批处理 + 分页 KV + chunked prefill（串 [[anthropic-16]]）、投机解码、以及**按长度分池**（短上下文池与长上下文池的 KV 压力差一个数量级，混在一起会互相拖累）；
> ④ **缓存层**：前缀缓存（系统提示/多轮对话的公共前缀）、KV 分层（HBM → DRAM → NVMe/远端）、结果级语义缓存（只对确定性请求）；
> ⑤ **观测与降级**：TTFT/TPOT 分位数、队列时长、KV 占用、批内序列数；降级链是**缩 max_tokens → 切小模型 → 排队 → 明确拒绝**（每一步都要让调用方知道）。
> 一句话判据：**用「利用率」做优化目标、用「p99」做约束**（或反过来），并行地靠**分层隔离**让两类流量各自跑到自己的最优点——而不是指望一个统一配置同时满足两者。

## 面试官在考什么

- **是否把矛盾讲成可解的**：直接说「提高利用率就会伤 p99」是初级；能给出**分层（长短上下文分池、交互与批处理分池）+ 准入（拒绝优于排队）+ 降级（有损但可控）**三件套才是高分。
- **排队论的直觉**：$W_q\propto \rho/(1-\rho)$——利用率 $\rho$ 从 70% 推到 95%，排队时间涨数倍，p99 首当其冲。所以**「把利用率打到 95%」本身就是一个危险的 OKR**；正确的目标是「在给定 p99 下的最大 goodput」（可引用的口径：goodput 是满足 SLO 的有效吞吐，DistServe 正是以 goodput 为优化目标）。
- **瓶颈会移动**：低负载时瓶颈是算力；长上下文时是 KV 显存；高并发短请求时可能是调度/网络/CPU（tokenize、采样、序列化）。能否说清「你的系统此刻瓶颈在哪、怎么判断」。
- **KV 显存与并发的算术**：给出 $B_{\max}$ 与 $T$ 的关系，并说明分页/量化/前缀缓存各自放大多少（串 [[anthropic-16]] 的表 2）。
- **成本与容量的账**：按 QPS、平均输入/输出 token 算出所需 GPU 数与每百万 token 成本；并说明「预留 headroom」为何必要（突发 + 故障 + 长尾请求）。
- **SLO 的定义与分级**：TTFT 与 TPOT 是两个独立的 SLO，交互式与批处理的容忍度完全不同；能否给出分级的 SLO 表并让调度器按它工作。

**常见错误答案**

- 「加更多 GPU」——不解决 p99（反而可能因调度更复杂而更差），也没说清瓶颈。
- 「把 batch 开到最大」——TPOT 与 p99 直接崩，且 KV 显存先爆。
- 「排队就行，反正会轮到」——排队会让 p99 无限增长；**必须有准入与拒绝**。
- 「缓存所有请求」——只对确定性/共享前缀有效；对随机生成的请求做结果缓存会带来正确性问题（温度采样、时效性）。
- 忽略 tokenize/采样/网络这些「非 GPU」环节——在高 QPS 下它们会成为真实瓶颈。

## 原理与推导

### 1. 五层栈与每层的职责

| 层 | 职责 | 关键指标 | 失效模式 |
| --- | --- | --- | --- |
| 接入 | 鉴权、配额、路由、去重、多区域 | 认证延迟、路由正确率 | 单点、区域故障 |
| 调度 | 优先级队列、准入控制、超时预算、抢占 | 队列时长、拒绝率 | 无准入 → 排队雪崩 |
| 推理 | 连续批处理、分页 KV、chunked prefill、投机 | TTFT/TPOT、批内序列数、KV 占用 | 长请求顶高 TPOT |
| 缓存 | 前缀缓存、KV 分层、语义缓存 | 命中率、KV 复用率 | 缓存穿透、失效抖动 |
| 观测/降级 | 分位延迟、错误分类、降级链 | p99、降级率 | 无降级 → 全面超时 |

### 2. 利用率与 p99 的定量关系

把推理实例看成一个排队系统：到达率 $\lambda$、服务率 $\mu$、利用率 $\rho=\lambda/\mu$。M/M/1 的排队等待：

$$W_q=\frac{\rho}{\mu(1-\rho)}\ \Rightarrow\ \text{p99 等待}\approx\frac{\ln(100)}{\mu(1-\rho)}\ \text{(指数尾近似)}$$

$\rho=0.7$ → $1/(1-\rho)=3.3$；$\rho=0.95$ → $20$。**利用率每往上推 10 个百分点，尾延迟的放大因子就接近翻倍**。这就是为什么「最大化利用率」必须以「给定 p99」为约束，并且要靠**隔离**（不同 SLO 的流量不共享同一个队列）来避免尾延迟互相污染。

### 3. 准入控制：为什么拒绝优于排队

设超时预算 $D$（例如 TTFT 10 s），当前预估等待 $\hat W$（可由队列长度与近步吞吐估计）。准入规则：

$$\text{accept}\iff \hat W + \hat S \le D\quad(\hat S\ \text{为预估服务时间})$$

否则**立即返回 429/503 并给出 retry-after**。理由有三：① 排队到超时的请求浪费了 GPU 却没有任何价值（还挤占了本可成功的请求）；② 客户端可以在别处重试（多区域/多模型）；③ 明确的拒绝比「慢慢失败」更容易被上层处理（可引用的实践口径：`Retry-After` 与指数退避是分布式系统的标准配合，串 [[anthropic-08]]）。

### 4. 为什么必须按长度/优先级分池

| 池 | 负载特征 | 最优配置 |
| --- | --- | --- |
| 短上下文（<8K） | 高 QPS、低 KV 压力 | 大 batch、高并发、强前缀缓存 |
| 长上下文（>32K） | 低 QPS、KV 主导 | 小 batch、KV 量化、独立实例 |
| 交互式 | 严 TTFT/TPOT | 优先级高、batch 上限低 |
| 批处理（离线） | 可容忍分钟级 | 大 batch、可利用空闲容量与竞价实例 |

**混池的代价**：一个 128K 请求的 KV 会挤掉上百个短请求的容量（KV 是按 token 计价的），导致短请求的 p99 被长请求随机拖垮。分池后每池各自跑到最优利用率，**整体 goodput 反而更高**。

### 5. 成本与容量

$$\text{所需 GPU 数}\approx\frac{\lambda\cdot(\bar N_{\text{in}}+\bar N_{\text{out}})\cdot 2P}{F_{\text{peak}}\cdot\text{MFU}\cdot(1-\text{headroom})}$$

以及 KV 约束：$\text{并发}\times T\times \text{KV/token}\le \text{可用显存}$。**两个约束取更大者**。headroom 通常留 20–30%（突发、故障转移、长尾请求）。

## 数值与代码验证

### 表 1：利用率与尾延迟放大（M/M/1 近似）

| 利用率 $\rho$ | 平均排队 $\rho/(1-\rho)$ | p99 排队 $4.6\rho/(1-\rho)$ | 相对 $\rho=0.5$ 的放大 | 实测（服务 40 ms） |
| --- | --- | --- | --- | --- |
| 0.50 | 1.0 | 4.6 | 1.0× | 平均 40 ms / p99 184 ms |
| 0.70 | 2.3 | 10.7 | 2.3× | 平均 93 ms / p99 430 ms |
| 0.85 | 5.7 | 26.1 | 5.7× | 平均 227 ms / p99 1,044 ms |
| 0.95 | 19.0 | 87.4 | **19×** | 平均 760 ms / p99 3,500 ms |

**读法**：把利用率从 50% 推到 95%，平均排队涨 **19 倍**、p99 涨 **19 倍**（$4.6\rho/(1-\rho)$ 里 $\rho/(1-\rho)$ 是唯一的放大因子）。**这就是「最大化利用率」不能单独作为目标的定量依据**——利用率每往上推 10 个百分点，尾延迟就再放大一截。

### 表 2：容量规划示例（Claude 级 70B、8×H100 单实例、fp16，QPS=400、输入 1K/输出 500 token）

| 口径 | 算式 | 结果 | 可信度 |
| --- | --- | --- | --- |
| **实测吞吐口径**（推荐） | 单实例 19,000 token/s（[[anthropic-16]] 的 batch=100、T=4K 推算）÷ 500 token/请求 = 38 req/s ⇒ 400/38 | **≈11 实例**（+25% headroom → 14） | 高（来自带宽受限区的推算） |
| FLOPs 口径 | $400\times1500\times2\times70\text{e}9/(989\text{e}12\times0.40\times8)$ | **≈26.5 实例** | **低——高估产能** |
| KV 约束 | 400 GB ÷ (320 KiB × 4K) | 每实例约 320 并发 | 用于交叉验证 |

**两个口径差 2.4 倍，原因值得记住**：FLOPs 口径假设 MFU 40%，但 decode 是**带宽受限**的（串 [[anthropic-15]]），实际 MFU 远低于 40%。所以**容量规划必须用实测/带宽口径，而不是 FLOPs 上界**；反过来，prefill 主导的负载（长输入、短输出）才适合用 FLOPs 口径估算。工程做法：两种都算，取**更保守**的那个，并用压测校准。

### 可运行代码

```python
# 1) 利用率-尾延迟曲线 + 准入控制的效果（离散事件模拟）
import math, random, statistics
def mm1_p99_queue(rho, service_ms=40):
    """M/M/1 的等待时间分位数：P(W>t)=rho*exp(-(1-rho)*mu*t)"""
    mu = 1.0 / (service_ms / 1000)
    return -math.log(1 - 0.99) / (mu * (1 - rho)) * 1000 * rho

print(f"{'利用率':>8} {'平均排队(ms)':>13} {'p99 排队(ms)':>13}")
for rho in (0.5, 0.7, 0.85, 0.95):
    wq = rho / (1 - rho) * 40
    print(f"{rho:>8.2f} {wq:>13.1f} {mm1_p99_queue(rho):>13.1f}")

def simulate(arrival_qps, service_ms, duration_s=60, timeout_ms=None, seed=1, admit=True):
    """单服务台排队：返回完成数、超时数、拒绝数、p50/p99 端到端延迟"""
    rnd = random.Random(seed)
    t, events = 0.0, []
    while t < duration_s:
        t += rnd.expovariate(arrival_qps)
        events.append(t)
    busy_until, lat, rejected, timed_out = 0.0, [], 0, 0
    for a in events:
        start = max(a, busy_until)
        est_wait = max(0.0, busy_until - a) * 1000
        if admit and timeout_ms and est_wait + service_ms > timeout_ms:
            rejected += 1
            continue
        finish = start + service_ms / 1000
        busy_until = finish
        e2e = (finish - a) * 1000
        if timeout_ms and e2e > timeout_ms:
            timed_out += 1
        else:
            lat.append(e2e)
    lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else float("nan")
    return len(lat), rejected, timed_out, p(0.5), p(0.99)

print("\n准入控制的效果（到达 24 QPS、单请求 40 ms、超时预算 500 ms —— 利用率约 96%）")
for admit in (False, True):
    ok, rej, to, p50, p99 = simulate(24, 40, admit=admit, timeout_ms=500)
    print(f"  准入={'开' if admit else '关'}: 成功 {ok:>4}  拒绝 {rej:>4}  超时 {to:>3}  "
          f"p50 {p50:6.1f} ms  p99 {p99:7.1f} ms")
print("读法：关闭准入时请求排队到超时（既浪费容量又无交付）；打开准入后 p99 被守住，")
print("      代价是明确的拒绝（上层可重试/降级）——这正是「拒绝优于排队」")

# 2) 容量规划：吞吐约束 vs KV 约束
PEAK, MFU, BW = 989e12, 0.40, 3.35e12
def capacity(qps, in_tok, out_tok, params=70e9, gpus_per_instance=8,
             kv_per_token=320*1024, kv_avail_gb=400, ctx=4096, headroom=0.25):
    tokens_per_s = qps * (in_tok + out_tok)
    flops_per_s = tokens_per_s * 2 * params
    instances_tp = flops_per_s / (PEAK * MFU * gpus_per_instance)
    kv_concurrency = kv_avail_gb * 1024**3 / (kv_per_token * ctx)
    out_per_s_instance = 19000                      # 来自 anthropic-16 的口径
    instances_kv = qps / (kv_concurrency / (out_tok / (out_per_s_instance/ kv_concurrency))) if False else qps * out_tok / out_per_s_instance
    need = max(instances_tp, instances_kv) * (1 + headroom)
    return instances_tp, instances_kv, need, kv_concurrency

for qps, in_tok, out_tok, ctx in ((400, 1000, 500, 4096), (100, 8000, 800, 32768)):
    a, b, need, kv = capacity(qps, in_tok, out_tok, ctx=ctx)
    print(f"\nQPS={qps}, 输入 {in_tok}, 输出 {out_tok}, 上下文 {ctx}")
    print(f"  吞吐约束需要 {a:5.1f} 实例；KV 约束需要 {b:5.1f} 实例；"
          f"取大者并留 {25}% headroom -> {need:.1f} 实例（每实例 KV 可容纳 {kv:.0f} 并发）")

# 3) 分池 vs 混池：长请求对短请求 p99 的污染（模拟 KV 被挤占）
def mixed_pool(long_share, total_concurrency=320, long_kv_multiplier=8):
    """长请求占用相当于 8 个短请求的 KV 槽位"""
    slots_long = total_concurrency / (long_share * long_kv_multiplier + (1 - long_share))
    return slots_long * (1 - long_share), slots_long * long_share
print(f"\n{'长请求占比':>10} {'短请求可用并发(混池)':>20} {'分池后短请求并发':>18}")
for share in (0.02, 0.10, 0.30):
    short_slots, long_slots = mixed_pool(share)
    print(f"{share:>10.0%} {short_slots:>20.0f} {320:>18}")
print("读法：即使长请求只占 2%，混池也会让短请求的可用并发下降（KV 被吃掉）；")
print("      分池让短请求池始终拥有完整的 320 个槽位 —— 这就是「按长度分池」的量化理由")
```

预期输出要点（实跑）：利用率从 0.5 → 0.95，平均排队与 p99 都放大 **19 倍**（实测 40 ms → 760 ms / 184 ms → 3,500 ms）；**准入控制开启后成功数反而更多（1,392 vs 1,100，约 +27%）且零超时**，代价只是 27 次明确拒绝（关闭时有 319 个请求排队到超时——既占容量又无交付）；这一条是本题最有说服力的量化结论；容量规划给出**两个口径差 2.4 倍**（实测吞吐口径 11 实例 vs FLOPs 口径 26.5 实例），差异来自 decode 的 MFU 远低于 40%，说明容量必须用带宽口径校准；分池模拟显示**即使长请求只占 2%，混池也会把短请求的可用并发从 320 压到 275**（10% 时长请求压到 169），这是按长度分池的量化依据。

## 常见追问

- **追问**：怎么给请求定优先级？
  - 要点：按 SLO 分级（交互式 > 批处理）、按租户配额（付费等级）、以及按「已等待时间」做老化（避免低优先级饿死）。优先级要进入**准入与调度**两处，而不只是排队顺序。
- **追问**：抢占怎么做？
  - 要点：分块抢占（chunked prefill 天然支持按块让出）、KV 换出到 DRAM/远端（Mooncake 式的 KV 分层）、或直接取消低优先级请求并告知；抢占会带来重算成本，要计入预算。
- **追问**：前缀缓存命中率怎么提升？
  - 要点：把稳定内容（系统提示、工具定义、长文档）放在请求前缀、避免在前缀里插入随机内容（时间戳、随机 id）、按模型/租户分桶（缓存块不能跨权限共享）；监控「可复用块比例」（串 [[inference-serving-05]]）。
- **追问**：多区域怎么保证 p99？
  - 要点：就近接入 + 区域内独立容量 + 跨区域故障转移（注意 KV 不能跨区迁移，只能重算）；路由要按「当前区域水位」而非静态权重。
- **追问**：如何判断瓶颈从算力移到了别处？
  - 要点：看四组指标的相对变化——GPU 利用率（SM occupancy）、KV 占用率、队列时长、CPU 侧（tokenize/采样/序列化）。若 GPU 利用率下降而队列变长，瓶颈在 CPU 或调度；若 KV 占用接近上限，瓶颈在显存。
- **追问**：成本怎么降？
  - 要点：① 提高前缀缓存命中（省 prefill）；② KV 量化（提升并发）；③ 长度分池 + 小模型路由（短请求用便宜模型）；④ 批处理走竞价/闲时容量；⑤ 投机解码降低每 token 成本——每一项都要用「每百万 token 成本」核算收益。

## 相关题目

- [[anthropic-16]]：batching 与 KV 显存约束，是本题推理层与容量规划的直接基础。
- [[anthropic-15]]：roofline 分析，解释了为什么瓶颈会在算力/带宽/显存之间移动。
- [[system-design-10]]：面向数亿用户的消费级聊天助手，本题是它在「平台 API 形态」下的版本。
- [[inference-serving-05]]：前缀缓存与 KV 复用，是本题缓存层的技术细节。
- [[system-design-09]]：LLM gateway 的路由、预算与降级链，对应本题接入与调度层。

## 参考资料与归属

- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 1 节推理层的分页 KV 与前缀共享机制来自这篇。
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）** —— Agrawal et al. (Microsoft, OSDI 2024)，2024-03-04：<https://arxiv.org/abs/2403.02310>。第 1 节 chunked prefill 与「平抑 TPOT 尖峰」的做法来自这篇。
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving（延伸）** —— Zhong et al. (OSDI 2024)，2024-01-18：<https://arxiv.org/abs/2401.09670>。第「面试官在考什么」里「以 goodput（满足 SLO 的有效吞吐）为优化目标」以及 prefill/decode 分离的动机来自这篇。
- **SGLang：Efficient Execution of Structured Language Model Programs（延伸）** —— Zheng et al. (RadixAttention)，2023-12-12：<https://arxiv.org/abs/2312.07104>。第 4 节前缀/基数树式的 KV 复用与结构化程序执行的思路来自这篇。
- **How does Prompt Caching work?** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-prompt-caching-work>。第 4 节前缀缓存的命中条件与收益表述转述自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（服务时间 40 ms、超时预算 500 ms、70B、8×H100、989 TFLOPs、MFU 40%、KV 320 KiB/token、实例吞吐 19k token/s 取自 [[anthropic-16]] 的推算）都是按本仓库统一口径构造的工程算例与显式假设；真实系统的调度器更复杂。M/M/1 近似用于给出量级与单调性，不作为精确预测。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
