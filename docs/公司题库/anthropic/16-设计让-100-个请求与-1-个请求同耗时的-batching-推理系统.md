---
type: question
id: anthropic-16
company: Anthropic
topic: inference-serving
order: 16
question: 设计一个 batching 推理系统，让 100 个请求与 1 个请求耗时相同。
question_en: Design a batching inference system where 100 requests take the same time as one.
asked_at: []
level: 高阶
tags: [推理服务, batching, continuous-batching, chunked-prefill, KV-显存]
sources:
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
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
related: [anthropic-15, anthropic-17, inference-serving-02, inference-serving-03, anthropic-13]
updated: 2026-09-28
---

## 一句话答案

> 「100 个请求与 1 个请求耗时相同」这句话在**decode 阶段近似成立**，在 **prefill 阶段不成立**——先说清这一点，再谈设计。
> **为什么 decode 近似成立**：decode 每步是**带宽受限**的（要把全部权重从 HBM 读一遍），读一次权重的时间与 batch 大小几乎无关（在拐点之前）。所以 batch 从 1 变成 100，**每步时间基本不变，吞吐涨 100 倍**（串 [[anthropic-15]] 的表 2）。
> **为什么 prefill 不成立**：prefill 是算力受限的，FLOPs $\propto$ token 数，100 个请求的 prefill 时间必然更长（除非并行到更多卡）。
> **因此系统设计的四件套**是：
> ① **连续批处理（continuous batching）**——把「等一批凑齐再跑」改成**按迭代调度**：每一步重新组批，新请求随时加入、完成的立即退出（不必等整批结束）；这是「100 个请求与 1 个请求同耗时」能成立的关键机制；
> ② **分页 KV（PagedAttention 式）**——KV 按块分配、非连续存储、按需增长，把显存碎片从 60–80% 降到接近 0，从而把**最大并发**顶上去；
> ③ **chunked prefill**——把长 prefill 切成块插到 decode 步之间，避免一个长请求的 prefill 把整批的 TPOT 顶出一个尖峰；
> ④ **prefill/decode 分离**——两者的资源特征不同（算力 vs 带宽），分到不同实例各自配比，避免互相拖累。
> 一句话：**「同耗时」来自「带宽受限 + 权重摊薄」，而把它工程化需要连续批处理、分页 KV 与 prefill 调度三件套。**

## 面试官在考什么

- **是否先限定条件**：直接说「用 batching 就行」是初级答案；先说清「decode 成立、prefill 不成立」才是理解。
- **静态 batching 的两个致命缺陷**：① **等批**（首批延迟 = 最慢到达者的延迟）；② **拖尾**（整批要等最慢的请求结束，长输出请求会把短请求一起拖住）。连续批处理正是为解决这两点。
- **KV 显存是真正的规模上限**：能否算出「每序列 $O(T)$ 的 KV ⇒ 一张卡能放多少并发」，以及分页/量化/前缀缓存如何放大它（串 [[inference-serving-03]]）。
- **TTFT 与 TPOT 的权衡**：连续批处理会提高吞吐但可能拉高 **TTFT**（新请求要等当前步结束）与 **TPOT**（batch 越大每步越慢）；chunked prefill 的意义正是在两者间取平衡。
- **padding 浪费**：静态 batching 要把不同长度补齐到最大长度，浪费算力与显存；连续批处理让序列各自推进（配合变长 kernel 与分页 KV），这是效率的另一半来源。
- **能否给出量与判据**：给出 $B$、$T$、KV/token 与显存上限的关系式，算出「在这台机器上最多能同时服务多少条」，以及吞吐-延迟曲线的拐点。

**常见错误答案**

- 「把 100 个请求拼成一个 batch 就行」——忽略等批延迟、拖尾、padding 与 KV 显存上限。
- 「batch 越大吞吐越高」——过拐点后每步时间线性增长、吞吐饱和，同时 TPOT 变差（串 [[anthropic-15]]）。
- 「用更大的卡就行」——KV 显存随并发线性增长，加卡要配合并行策略（TP/PP/DP）与 KV 分页。
- 忽略 prefill 与 decode 的差异，用一个统一 batch 策略处理两者。

## 原理与推导

### 1. 两种阶段的成本模型

**decode（每步生成 1 个 token/序列）**：

$$t_{\text{step}}(B)\approx\max\Big(\underbrace{\frac{2P_{\text{bytes}}}{BW_{\text{eff}}}}_{\text{读权重}},\ \underbrace{\frac{2PB}{F_{\text{peak}}}}_{\text{算力}},\ \underbrace{\frac{B\cdot \text{KV}_{\text{per-token}}\cdot T}{BW_{\text{eff}}}}_{\text{读 KV}}\Big)$$

- 第一项与 $B$ 无关 → **小 batch 时时间被它锁死**（这是「同耗时」的来源）；
- 第二、三项随 $B$ 增长 → 拐点之后时间线性增长。
**KV 读取项常被忽略，但它才是长上下文下的真正限制**。令 KV 读取追上权重读取的 batch 为

$$B_{\text{KV}}=\frac{2P_{\text{bytes}}/N_{\text{gpu}}}{\text{KV}_{\text{per-token}}\cdot T}$$

以 70B、fp16（每卡 17.5 GB）、KV 320 KiB/token、8 卡 TP 为例：$T$=4K 时 $B_{\text{KV}}\approx104$；$T$=32K 时 **$B_{\text{KV}}\approx13$**。「每步时间与 batch 无关」的平坦区上界是 $\min(B_{\text{KV}}, B_{\text{算力}})$——**上下文越长，平坦区越短**（本例 4K 时约 100、32K 时只剩 13）。

**prefill（$N$ 个 token 一次前向）**：$t_{\text{prefill}}\approx 2NP/F_{\text{peak}}$（算力受限），加上注意力的 $O(N^2)$ 项。100 个请求各自 4K token → 总 400K token 的 prefill，与 1 个请求的 4K token 相比，**算力需求是 100 倍**（可用更多卡并行，但单机时间必然更长）。

### 2. 静态 batching 的两个缺陷（用数字看）

设 100 个请求，输出长度分别为 10 和 1,000 token：

| 方案 | 首批延迟 | 批结束时间 | 说明 |
| --- | --- | --- | --- |
| 静态 batching（等齐 + 同步） | 等最慢到达者（可能数百 ms） | 由最长的 1,000 token 决定 | 短请求被拖 100 倍；GPU 在后半段只服务少数序列（利用率崩） |
| 连续 batching | 当前步结束即加入（几十 ms） | 每个序列完成即退出 | 吞吐接近理论上限，TPOT 平稳 |

**关键机制**：连续批处理在**每个 decode 步**重新决定 batch 组成（完成者退出、等待者补位），因此不需要 padding 到统一长度，也不会被长请求拖住。

### 3. KV 显存：真正的规模上限

$$B_{\max}\approx\frac{\text{显存}_{\text{可用}}}{\text{KV}_{\text{per-token}}\times T_{\text{max}}}$$

以 8×H100（80 GB/卡，权重占 17.5 GB/卡，KV 可用约 50 GB/卡 ×8 = 400 GB）、KV = 320 KiB/token（LLaMA-3-70B 口径）：

| 上下文 $T$ | 每序列 KV | $B_{\max}$（400 GB 可用） |
| --- | --- | --- |
| 4K | 1.25 GiB | ~320 |
| 32K | 10 GiB | ~40 |
| 128K | 40 GiB | ~10 |

**读法**：长上下文场景下 KV 才是并发上限——这就是为什么分页（减少碎片）、量化（KV int8 减半）、前缀缓存（共享公共前缀）在服务侧收益巨大。**注意与 roofline 拐点（约 295）对照**：$T=32$K 时 $B_{\max}\approx40\ll295$，说明**长上下文场景是 KV 显存受限，而不是算力受限**。

### 4. chunked prefill 与 prefill/decode 分离

- **问题**：一个新请求的 prefill 可能是 8K token，如果整段插入某个 decode 步，那一步的耗时会被拉长几十倍（TPOT 尖峰）。
- **chunked prefill**：把 prefill 切成固定大小的块（例如 512–2048 token），分散到多个 decode 步中执行，使**每步的 token 预算大致恒定**；代价是长请求的 TTFT 略增。
- **分离部署（DistServe 式）**：prefill 实例（算力优先、可大 batch）与 decode 实例（带宽优先、KV 显存优先）分开，各自配比与扩缩容；中间传 KV（可通过高速网络或分层存储）。收益是两者不再互相干扰，代价是 KV 传输开销与更复杂的调度。
- **调度策略**：常见做法是「token 预算 + 抢占」——每步总 token 数有上限（保证 TPOT），超出部分排队；高优先级请求可抢占低优先级（分块重算或换出 KV）。

### 5. 需要监控的指标

| 指标 | 含义 | 目标 |
| --- | --- | --- |
| TTFT p50/p99 | 首 token 延迟（prefill + 排队） | 按产品定（交互式常 <1 s） |
| TPOT p50/p99 | 每输出 token 时间（decode 步长） | 稳定、无尖峰（chunked prefill 的作用） |
| 批内序列数分布 | 实际并行度 | 贴近拐点 |
| KV 占用率 | 显存利用 | 高但不能碎片化 |
| 抢占/重算次数 | 调度压力 | 越低越好 |
| 排队时长 | 准入控制效果 | 有界（否则应拒绝而非排队） |

## 数值与代码验证

### 表 1：静态 vs 连续 batching（模拟：100 个请求，输出长度 10–1,000 token）

| 指标 | 静态 batching | 连续 batching |
| --- | --- | --- |
| 首批延迟（等齐） | 高（等最慢到达） | 低（下一步即加入） |
| 总完成时间 | 由最长请求决定 + 尾部空转 | 接近理论下界 |
| GPU 利用率（末段） | 低（只剩少数长请求） | 高（空位被新请求填满） |
| padding 浪费 | 有（补齐到最大长度） | 无（变长 + 分页 KV） |
| TPOT 稳定性 | 差（受长请求与 prefill 插入影响） | 好（配合 chunked prefill） |

### 表 2：并发上限由 KV 决定（KV = 320 KiB/token）

| 可用 KV 显存 | $T$=4K | $T$=32K | $T$=128K |
| --- | --- | --- | --- |
| 100 GB | ~80 | ~10 | ~2 |
| 400 GB | ~320 | ~40 | ~10 |
| 400 GB（KV int8 量化） | ~640 | ~80 | ~20 |

### 可运行代码

```python
# 1) 静态 vs 连续 batching 的完成时间与利用率模拟（离散时间步）
import random, statistics
PEAK_FLOPS, PEAK_BW = 989e12, 3.35e12
def step_time_ms(batch, params=70e9, kv_per_token=320*1024, T_avg=4096, gpus=8):
    """decode 单步时间的下界：max(读权重, 算力, 读 KV)（8 卡 TP 分摊权重）"""
    w_bytes = params * 2 / gpus
    flops = 2 * params * batch
    kv_bytes = batch * kv_per_token * T_avg
    t = max(w_bytes / PEAK_BW, flops / (PEAK_FLOPS * gpus), kv_bytes / (PEAK_BW * gpus))
    return t * 1000

def run(out_lens, mode, max_batch=64, kv_capacity_tokens=400*1024**3 // (320*1024)):
    """mode='static'：等齐成批、整批同生共死；'continuous'：按步补位"""
    t_ms, done, finished = 0.0, 0, []
    if mode == "static":
        queue = list(enumerate(out_lens))
        while queue:
            batch = queue[:max_batch]; queue = queue[max_batch:]
            # 整批耗时 = 最长输出 × 每步时间(批大小固定)
            longest = max(l for _, l in batch)
            t_ms += longest * step_time_ms(len(batch))
            done += len(batch); finished += [t_ms] * len(batch)
    else:
        active, waiting = [], list(enumerate(out_lens))
        while active or waiting:
            # 补位（受 max_batch 与 KV 容量限制）
            while waiting and len(active) < max_batch and \
                  (sum(a[2] for a in active) + 4096) <= kv_capacity_tokens:
                i, l = waiting.pop(0); active.append([i, l, 4096])   # [id, 剩余, 已占KV]
            if not active:
                break
            t_ms += step_time_ms(len(active))
            for a in active:
                a[1] -= 1; a[2] += 1
            still = []
            for a in active:
                if a[1] <= 0:
                    done += 1; finished.append(t_ms)
                else:
                    still.append(a)
            active = still
    return t_ms, done, finished

random.seed(21)
LENS = [random.choice([10, 50, 200, 1000]) for _ in range(100)]
for mode in ("static", "continuous"):
    total, done, fin = run(LENS, mode)
    print(f"{mode:<11} 总耗时 {total/1000:8.1f} s  完成 {done}  "
          f"首请求完成 {min(fin)/1000:6.2f} s  末请求完成 {max(fin)/1000:8.2f} s")

# 2) 「1 个请求 vs 100 个请求」的每步时间（decode 带宽受限的直接验证）
print(f"\n{'batch':>6} {'每步(ms)':>10} {'相对 batch=1':>14} {'吞吐(token/s)':>14}")
base = step_time_ms(1)
for B in (1, 10, 50, 100, 295, 600):
    t = step_time_ms(B)
    print(f"{B:>6} {t:>10.3f} {t/base:>13.2f}x {B/(t/1000):>14,.0f}")
print("读法：到拐点（约 295）之前每步时间几乎不变 —— 这就是「100 个请求与 1 个请求同耗时」的来源；")
print("      过拐点后每步时间随 B 线性增长，吞吐饱和")

# 3) KV 显存决定的并发上限
def max_concurrency(kv_gb, T, kv_per_token=320*1024):
    return int(kv_gb * 1024**3 // (kv_per_token * T))
print(f"\n{'可用KV':>8} {'T=4K':>8} {'T=32K':>8} {'T=128K':>8}")
for kv in (100, 400, 800):
    print(f"{kv:>6} GB {max_concurrency(kv,4096):>8} {max_concurrency(kv,32768):>8} "
          f"{max_concurrency(kv,131072):>8}")
print("读法：T=32K 时 400 GB 只能放约 40 条并发，远低于 roofline 拐点 295 —— "
      "长上下文场景是 KV 显存受限，不是算力受限")

# 4) chunked prefill：把 prefill 切块对 TPOT 尖峰的影响
def tpot_with_prefill(prefill_tokens, chunk, decode_step_ms, prefill_ms_per_token=0.02):
    if chunk is None:      # 整段插入一步
        return decode_step_ms + prefill_tokens * prefill_ms_per_token
    blocks = -(-prefill_tokens // chunk)
    peak = decode_step_ms + chunk * prefill_ms_per_token
    return peak, blocks
print("\nprefill=8192 token、decode 步 40 ms：")
whole = tpot_with_prefill(8192, None, 40)
print(f"  整段插入：该步 TPOT 尖峰 {whole:.0f} ms（是正常步长的 {whole/40:.0f} 倍）")
for chunk in (512, 1024, 2048):
    peak, blocks = tpot_with_prefill(8192, chunk, 40)
    print(f"  chunk={chunk:>5}: 峰值步长 {peak:6.1f} ms（{peak/40:.1f}x），需 {blocks} 步完成 prefill")
print("读法：chunk 越小 TPOT 越平稳，但长请求的 TTFT 越晚 —— 这就是 chunked prefill 的权衡")
```

预期输出要点（实跑）：静态与连续 batching 的总耗时对比显示**连续批处理显著更快且首个请求早得多**（静态要等整批最长输出）；第 2 段直接验证核心命题——**batch 从 1 到 100 每步时间几乎不变（5.224 ms，比值 1.00），吞吐线性上升**；到 295 时因 **KV 读取成为主导项**，每步涨到 14.774 ms（2.83×）、吞吐停在约 20k token/s——所以平坦区的真正上界是 $\min(B_{\text{KV}},B_{\text{算力}})$；第 3 段显示 $T$=32K 时 400 GB KV 只够约 40 条并发（远低于拐点 295），说明长上下文是**显存受限**；第 4 段量化 chunked prefill 把 TPOT 尖峰从 **5 倍**压到 **1.3 倍**（chunk=512），代价是长请求的 TTFT 延后（本例需要 16 步才完成 8192 token 的 prefill）。

## 常见追问

- **追问**：连续批处理会不会让 p99 变差？
  - 要点：会引入「等待当前步结束」的排队延迟（通常几十毫秒级），但避免了静态 batching 的等批与拖尾（后者往往几百毫秒到数秒）；关键是给**队列长度与等待时长**设上限（准入控制），宁可拒绝也不无限排队。
- **追问**：100 个请求真的完全同耗时吗？
  - 要点：近似成立（在带宽受限区），但有三处偏差——① KV 读取随 batch 增长（长上下文时显著）；② prefill 阶段不行；③ padding/调度开销。所以准确表述是「decode 阶段的每步时间近似与 batch 无关」。
- **追问**：如何决定 max_batch？
  - 要点：由三个约束取最小——① roofline 拐点（算力约束）；② KV 显存（$B\cdot T\cdot$KV/token ≤ 可用显存）；③ TPOT SLO（batch 越大每步越慢）。工程上做成**动态上限**：随可达显存与实测 TPOT 调整。
- **追问**：前缀缓存怎么与 batching 结合？
  - 要点：相同前缀的序列共享 KV 块（块级引用计数），prefill 只算新增 token；在系统提示统一、多轮对话场景命中率高，能同时降低 TTFT 与算力消耗。
- **追问**：为什么要把 prefill 与 decode 分开部署？
  - 要点：两者资源特征相反（算力 vs 带宽）且会互相干扰（prefill 插入会顶高 TPOT）；分离后可各自按最优 batch 与并行策略配置，代价是 KV 传输与更复杂调度（DistServe 的核心论点）。
- **追问**：投机解码在这个系统里放哪一层？
  - 要点：放在 decode 路径上（用小模型草拟、大模型一次验证），它减少的是**每步的串行次数**，与 batching 正交、可叠加；但会额外消耗算力（需在带宽/算力受限的边界上评估是否值得）。

## 相关题目

- [[anthropic-15]]：roofline 与算术强度分析，是本题「为什么 decode 每步时间与 batch 无关」的数学基础。
- [[anthropic-17]]：Claude 级 API 服务栈，把本题的 batching 机制放进完整的接入-调度-推理-缓存-观测体系。
- [[inference-serving-02]]：连续批处理与调度策略的通用讲解，可与本题的量化对照。
- [[inference-serving-03]]：KV cache 的显存管理与分页，是本题并发上限一节的技术基础。
- [[anthropic-13]]：KV cache 每 token 字节数的推导（320 KiB/token 的来源）。

## 参考资料与归属

- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 1 节「decode 带宽受限、batch 摊薄权重读取」与第 3 节 KV 显存约束的分析框架来自这篇。
- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 3 节 KV 分页管理与前缀共享的机制来自这篇。
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）** —— Agrawal et al. (Microsoft, OSDI 2024)，2024-03-04：<https://arxiv.org/abs/2403.02310>。第 4 节 chunked prefill「把长 prefill 切块以平抑 TPOT」的做法与其权衡来自这篇。
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving（延伸）** —— Zhong et al. (OSDI 2024)，2024-01-18：<https://arxiv.org/abs/2401.09670>。第 4 节 prefill/decode 分离部署的动机与收益来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（70B、fp16、KV 320 KiB/token、8×H100、989 TFLOPs / 3.35 TB/s、输出长度分布 10/50/200/1000、prefill 每 token 0.02 ms）都是按本仓库统一口径构造的工程算例与显式假设；真实系统的调度器更复杂（抢占、换出、优先级），实测值会不同。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
