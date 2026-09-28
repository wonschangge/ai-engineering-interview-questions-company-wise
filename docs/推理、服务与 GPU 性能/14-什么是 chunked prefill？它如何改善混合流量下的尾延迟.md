---
type: question
id: inference-serving-14
topic: 推理、服务与 GPU 性能
order: 14
question: 什么是 chunked prefill？为什么它能在混合流量下改善 tail latency？
question_en: What is chunked prefill and why does it improve tail latency under mixed traffic?
asked_at: []
level: 高阶
tags: [chunked-prefill, 调度, 尾延迟, sarathi]
sources:
  - title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）
    url: https://arxiv.org/abs/2403.02310
    author: Agrawal et al. (Microsoft, OSDI 2024)
    published: 2024-03-04
related: [inference-serving-01, inference-serving-02, inference-serving-09, inference-serving-15]
updated: 2026-09-28
---

## 一句话答案

> chunked prefill 把一个长 prompt 的 prefill 拆成若干个近似等长的 chunk，每个 iteration 只计算其中一个 chunk，并把这些 prefill token 与所有正在 decode 的请求放进**同一个 batch**，使单次 iteration 的 token 总数不超过预设的 token budget $\tau$。
> 混合流量下的 ITL 长尾来自「整段 prefill 独占 iteration」：7B 模型在单张 H100 上，8k prompt 的完整 prefill 约 290 ms（按 40% MFU 折算），而 decode 一步的权重读取下限只有 4.2 ms；切成 $\tau=512$ 的 chunk 后，每次 iteration 最多多花约 18 ms。
> 总计算量不变、平均 ITL 也几乎不变，被改变的是 iteration 时长的**上界**：尾延迟从「由窗口内最长的那条 prompt 决定」变成「由 token budget 决定」。

## 面试官在考什么

- 能否把「尾延迟」落到一个可调的量上：单次 iteration 的工作量上界。只会说「chunked prefill 能降尾延迟」而不给出上界表达式，说明没抓住机制。
- 是否理解 prefill 与 decode 的资源画像差异（compute-bound 与 memory-bound），以及为什么两者可以在同一个 batch 里互补（[[inference-serving-01]]）。
- 是否清楚 token budget 是双向取舍：调小保护 ITL 但损失 prefill 效率与 TTFT，调大则相反。面试官通常追问「你线上会设多少，依据是什么」。
- 是否分得清 chunked prefill、continuous batching、PagedAttention、P/D 分离各自解决的问题层次，而不是把它们混成一句「都是推理优化」。
- 能否讲实现层面的约束：跨 chunk 的 attention 要读前序 chunk 的 KV cache、attention mask 与位置编码要按绝对位置算、chunk size 与 GPU tile 量化、pipeline parallel 下 micro-batch 的均衡。

常见错误答案：

- 「chunked prefill 让 prefill 变快了」。总 FLOPs 不变，实测反而略慢（论文 Figure 14：chunk 512 时 Yi-34B 的 prefill 耗时最多增加 25%）。它优化的是延迟分布，不是 prefill 吞吐。
- 「有了 continuous batching / PagedAttention 就不需要 chunked prefill」。前者决定请求何时进出批，后者决定显存能否动态增删，两者都不限制**单个 iteration 的 token 数**。

## 原理与推导

### 1. 一次 iteration 的时长由谁决定

把一个 iteration 近似成「token 数决定」的量：批里有 $n = b + n_p$ 个 token，$b$ 个来自 decode（每个请求 1 个），$n_p$ 个来自 prefill。权重每步都要从 HBM 读一遍，于是

$$T_{\text{iter}}(n) \approx \max\left(\underbrace{\frac{W}{B_{\text{mem}}}}_{\text{权重读取下限}},\ \underbrace{\frac{2 N_{\text{eff}} \cdot n}{P_{\text{eff}}}}_{\text{算力}}\right)$$

其中 $W$ 是单卡要读的权重字节数，$N_{\text{eff}}$、$P_{\text{eff}}$ 是单卡分摊后的参数量与有效算力。

- $n$ 很小（纯 decode）时，$T_{\text{iter}}$ 由权重读取决定，与 batch 大小几乎无关。这正是 decode 能靠加大 batch 摊薄权重的物理基础，也是论文 Figure 6 里 LLaMA2-70B 的线性层在 128–512 token 区间耗时几乎不动的原因。
- 线性层的算术强度是 $2N$ FLOPs / $2N$ 字节 = $n$ FLOP/byte，两条线交于拐点 $n^{*} = P_{\text{peak}} / B_{\text{mem}}$：H100 SXM 按 bf16 稠密算力 989 TFLOPS、HBM3 3.35 TB/s 得 $n^{*} \approx 295$。
- 超过 $n^{*}$ 后 $T_{\text{iter}}$ 随 $n$ 线性增长。论文给的理论值是 A100 上约 200 token，实测要 500–600 token，差距来自固定开销；论文 Figure 4 在 Mistral-7B / 单张 A100 上还测出「1 个 decode token 的线性层耗时 ≈ 128 个 prefill token」，这个 128 就是拐点的实测版本（A100 80GB 的理论值是 153，差距即效率折损）。

「prefill-decode 干扰」于是有了精确定义：一个 8k token 的 prompt 一次性进入 iteration，$n$ 冲到 8000 以上，是拐点的约 27 倍，这一次 iteration 的时长完全由这条 prompt 的长度决定，期间所有 decode 请求停在原地。论文称之为 generation stall，并在 Figure 1(a) 里给出 vLLM 在 Yi-34B / 2×A100 上持续数秒的 stall 实例。

### 2. chunked prefill 的做法

Sarathi-Serve 的调度由两条规则组成，缺一不可。

**规则一：token budget。** 每个 iteration 处理的 token 总数有硬上限 $\tau$。decode 优先占位（每个在跑的请求 1 个 token），剩余预算才分给 prefill，且只分 chunk 不分段：

$$n_p \le \tau - b,\qquad \text{chunk size} = \min(\text{剩余 prompt},\ \tau - n_{\text{已用}})$$

**规则二：stall-free scheduling。** 新请求随时可以进入正在跑的批，不需要先排空 decode，也不需要暂停正在进行的 prefill。论文 Algorithm 3 的填充顺序就是这条规则的直接体现：先把所有 ongoing decode 放进批，再放**已经开始**的 prefill 的下一段 chunk，最后才用剩余预算接纳新请求。已有的 prefill 优先于新请求，避免长 prompt 被后来者反复插队饿死。

注意 chunk 不是截断：同一段 prompt 的 token 一个不少，只是分几次 forward 计算，每段的 KV 照常写进 cache，最终输出与整段 prefill 逐 token 一致。切分改变的是计算顺序与显存访问模式，不是语义。

| 调度策略 | 新请求何时进入 | 单 iteration 的 token 数 | ITL | 吞吐 |
| --- | --- | --- | --- | --- |
| request-level batching（FasterTransformer 一类） | 等当前批所有请求 decode 完 | 整批，通常按最长请求 padding | 低 | 差，批尾拖长 |
| iteration-level + prefill-prioritizing（Orca 为 hybrid batch，论文写作时的 vLLM 则让整段 prefill 独占 iteration） | 一有显存就把整段 prefill 算完 | 无上界：Orca 为 $b + S$，vLLM 为 $S$ | 长尾严重 | 高 |
| chunked prefill + stall-free（Sarathi-Serve） | 每个 iteration 的剩余预算里加入 | $b + \min(S_{\text{剩余}}, \tau - b) \le \tau$ | 有硬上界 | 高 |

这里有个容易被忽略的细节：**hybrid batching 本身不够**。Orca 已经支持 prefill 与 decode 混在同一个批里，但它混进去的是**整段** prefill，iteration 时长照样能到几百毫秒甚至几秒。论文 Table 4 的直接对照是：只用 hybrid batching 时 P99 TBT 为 0.68 s（openchat trace），加上 chunked prefill 后降到 0.14 s。切分才是关键的第二步。

### 3. 为什么尾延迟会改善

**(a) 上界被替换。** 设每个 prefill token 的算力成本为 $c = 2N_{\text{eff}} / P_{\text{eff}}$，则

$$\text{ITL} \le T_{\text{decode}}(b) + \tau \cdot c$$

右端与 prompt 长度无关，只与 token budget、decode batch 大小、硬件有关。不切分时上界是 $T_{\text{decode}}(b) + S_{\max}\cdot c$，而 $S_{\max}$ 是负载里**随机出现**的最长 prompt：并发越高、窗口越大，抽到的 $S_{\max}$ 越大，尾延迟随负载恶化（论文 Figure 1(b) 的曲线形状就来自这里）。换成 token budget 之后，这个随机变量从延迟表达式里消失了。

**(b) P99 与 max 受益，均值不变。** P99 是分布的高分位，重尾分布的高分位由尾部形状决定。把单次增量截断在 $\tau c$，等于给分布装了一个硬天花板；而总工作量没有减少，只是被摊到更多 iteration 上，所以平均 ITL 基本不动。这解释了为什么这类优化在「吞吐不变」的前提下能把 SLO 达标率拉上去——SLO 通常写在 P99 TBT 上。

**(c) 两个方向都不再饥饿。** prefill-prioritizing 会让 decode 饿死（ITL 尖刺），decode-prioritizing 会让新请求饿死（TTFT 爆炸）。chunked prefill 让长 prompt 与 decode 共存：新请求每个 iteration 都能拿到一部分预算，已经开始的 prefill 也不会被新来者无限推后。

**(d) iteration 时长均匀之后，可以开更大的 batch。** SLO 约束下的 batch 上限取决于「最坏的 iteration」有多长。时长方差小，调度器就能按预算规划每步工作量，而不是按最坏情况留余量；pipeline parallel 部署下 micro-batch 之间也更均衡，bubble 变少——论文 §5.3 在 Falcon-180B（TP4-PP2、跨节点 100 Gbps 以太网）上把严格 SLO 的容量提升了 3.6×（相对 vLLM 的同构 hybrid-parallel 配置）。

### 4. 代价是什么

1. **TTFT 变差。** 一条 prompt 需要 $\lceil S/\tau \rceil$ 个 iteration 才 prefill 完，而且它还要和其他请求分享预算。论文 Table 4（Yi-34B、2×A100、$\tau=1024$、128 请求）里，只用 hybrid batching 时 P50 TTFT 是 0.53 s（openchat）/ 3.78 s（arxiv），加上 chunked prefill 后变成 0.76 s / 3.90 s，即 +43% / +3%。
2. **prefill 效率下降。** chunk 小意味着算术强度低、固定开销占比高。论文 Figure 14 在 Yi-34B（TP2）上实测：chunk 512 时 prefill 总耗时最多增加约 25%，chunk 2048 时几乎可以忽略。
3. **跨 chunk 的 KV 重读。** 第 $i$ 个 chunk 的 attention 必须读到前 $i-1$ 个 chunk 的 KV cache，attention 的**算力**不变（因果掩码下的总注意力计算量与是否切分无关），但 HBM **读取量**增加：prompt 长 $S$、切成 $N$ 份时总读取量是 $\sum_{i=0}^{N-1} i \cdot c = cN(N-1)/2$，即 prompt 自身 KV 的 $(N-1)/2$ 倍。
4. **tile 量化。** GPU 的 matmul 按 tile 切分给线程块，维度不是 tile 整数倍时会有线程块做无用的计算。论文实测 chunk size 257 比 256 的 prefill 耗时高 32%，所以 chunk 通常取 2 的幂。

合成一句话：**用一点 TTFT 和一点 prefill 效率，换 ITL 的上界**（[[inference-serving-09]]）。

### 5. token budget 怎么选

由 SLO 反推：$\tau \lesssim \big(\text{P99 TBT SLO} - T_{\text{decode}}(b)\big) / c$。论文的做法是分层设定——严格 SLO（Mistral-7B 100 ms、Yi-34B 200 ms、LLaMA2-70B 与 Falcon-180B 1 s 的 P99 TBT）统一用 $\tau=512$；宽松 SLO（0.5 / 1 / 5 / 5 s）用 $\tau=2048$，其中 LLaMA2-70B 的宽松档用 1536 以减少 pipeline bubble。具体数值用模拟器（论文用 Vidur）在目标部署上扫出来，同时把 chunk 大小对 prefill 效率的影响算进去。两个额外的经验约束：chunk 取 2 的幂避开 tile 量化；PP 部署下不要取大 chunk，否则 micro-batch 时长方差变大、bubble 增多。

### 6. 与相邻技术的关系

| 技术 | 解决的问题 | 与 chunked prefill 的关系 |
| --- | --- | --- |
| continuous batching（[[inference-serving-02]]） | 请求何时进出批（批的边界） | 正交。它让请求级调度成为可能，但一个 iteration 仍可能装进一个巨大 prefill |
| PagedAttention（[[inference-serving-03]]） | KV cache 能否动态增删、无碎片 | 互补。分页让 chunk 的 KV 可以写到非连续的 block；没有它，chunked prefill 的显存管理会很别扭 |
| chunked prefill | 单个 iteration 的工作量上限 | 本题 |
| P/D 分离（[[inference-serving-15]]） | 把 prefill 与 decode 放到不同 GPU，彻底消除干扰 | 另一条路线。分离对 TTFT 更友好（prefill 可以整段跑、效率最高），代价是 KV cache 要跨机迁移；chunked prefill 在同一张卡上做细粒度调度 |
| prefix caching（[[inference-serving-05]]） | 相同前缀只算一次 | 正交但相互影响：命中前缀后待 prefill 的 token 变少，chunk 数变少、TTFT 变好；反过来，chunk 的划分粒度会影响缓存命中的边界 |

## 数值与代码验证

所有数字都自己复算过，下面每个表都标明口径。

### 复算 1：算术强度拐点

公开规格口径（bf16 **稠密**算力，不含稀疏）：

| 设备 | bf16 稠密算力 | HBM 带宽 | 拐点 $P/B$ |
| --- | --- | --- | --- |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | 295 FLOP/byte |
| H100 SXM（稀疏口径） | 1979 TFLOPS | 3.35 TB/s | 591 FLOP/byte |
| A100 80GB SXM | 312 TFLOPS | 2039 GB/s | 153 FLOP/byte |
| A100 40GB SXM | 312 TFLOPS | 1555 GB/s | 201 FLOP/byte |

口径提醒：算力取稀疏或 FP8 口径，拐点直接翻倍；论文里「理论约 200 token」与 A100 40GB 的 1555 GB/s 口径吻合（$312/1.555 \approx 201$），80GB HBM2e 口径下是 153。用于判断「一个 prefill 会不会把 iteration 变成 compute-bound」时，必须说清用的是哪一档。

### 复算 2：一次 8k prompt 的完整 prefill vs decode 一步

prefill 按线性层主导的 $2N$ FLOPs/token 估算（论文 Figure 4：即使序列很长，线性层仍占总时间 80% 以上）；decode 一步取下限——只算权重读取，忽略 KV cache 与 kernel 开销，因此是乐观值。

| 场景 | 计算 | 时间 |
| --- | --- | --- |
| 7B prefill 8k，单张 H100，100% MFU | $2 \times 7\text{e}9 \times 8192 / 989\text{e}12$ | 116 ms |
| 7B prefill 8k，单张 H100，50% MFU | 同上 / 0.5 | 232 ms |
| 7B prefill 8k，单张 H100，40% MFU | 同上 / 0.4 | **290 ms** |
| 7B prefill 8k，单张 H100，30% MFU | 同上 / 0.3 | 386 ms |
| 70B prefill 8k，TP2（2×H100）每卡 35B，40% MFU | $2 \times 35\text{e}9 \times 8192 / 395.6\text{e}12$ | 1449 ms |
| 7B decode 一步（14 GB 权重 / 3.35 TB/s） | $14\text{e}9 / 3.35\text{e}12$ | 4.18 ms |
| 70B decode 一步，TP2（每卡 70 GB） | $70\text{e}9 / 3.35\text{e}12$ | 20.9 ms |

于是「8k prefill 独占一次 iteration」相对 decode 一步的尖刺倍数是 $290 / 4.18 \approx 69\times$（40% MFU 口径），30% MFU 时是 $92\times$。常被引用的「一个 8k prompt 的 prefill 独占 GPU 几百毫秒」在 7B / 单卡 H100 上成立；换成 70B 就是秒级，与论文 Figure 1(a) 的观察一致。

### 复算 3：token budget 换算成每次 iteration 的增量

| 场景 | 每 prefill token 成本 | $\tau=256$ | $\tau=512$ | $\tau=1024$ | $\tau=2048$ |
| --- | --- | --- | --- | --- | --- |
| 7B / 1×H100，40% MFU | 35.4 µs | 9.1 ms | **18.1 ms** | 36.2 ms | 72.4 ms |
| 7B / 1×H100，50% MFU | 28.3 µs | 7.2 ms | 14.5 ms | 29.0 ms | 58.0 ms |
| 70B / 2×H100（TP2），40% MFU | 176.9 µs | 45.3 ms | 90.6 ms | 181.1 ms | 362.2 ms |

这张表就是选 $\tau$ 的依据：Mistral-7B 的严格 SLO 是 100 ms P99 TBT，$\tau=2048$ 光 prefill 增量就吃掉 72 ms，几乎没有余量；$\tau=512$ 只用 18 ms。而 LLaMA2-70B 的严格 SLO 是 1 s，$\tau=512$ 的 90 ms 增量仍在预算内。反过来在宽松档（Yi-34B 的 1 s），$\tau=2048$ 的增量可以接受，换来的是更高的 prefill 效率。复算 3 的成本按 H100 口径给出；论文的 Mistral-7B 实验跑在 A100 80GB 上，每 prefill token 约 112 µs（$2 \times 7\text{e}9 / (312\text{e}12 \times 0.4)$），$\tau=2048$ 需要 230 ms、已超出 100 ms 的严格 SLO，$\tau=512$ 约 57 ms 才留有余量。

### 复算 4：跨 chunk 的 KV 重读

以 LLaMA-3-70B 级配置（$L=80$、$H_{kv}=8$、$d_{head}=128$、bf16）为例，每 token 的 KV cache 是 $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ 字节 = 320 KiB。prompt 长 8192：

| chunk size | chunk 数 $N$ | 额外 KV 读取（token 槽） | 字节 | 相对 prompt | @3.35 TB/s |
| --- | --- | --- | --- | --- | --- |
| 256 | 32 | 126,976 | 38.75 GiB | 15.50× | 12.4 ms |
| 512 | 16 | 61,440 | 18.75 GiB | 7.50× | 6.0 ms |
| 1024 | 8 | 28,672 | 8.75 GiB | 3.50× | 2.8 ms |
| 2048 | 4 | 12,288 | 3.75 GiB | 1.50× | 1.2 ms |

对比同一 prompt 的 prefill 算力时间（70B / TP2 / 40% MFU 约 1.45 s），即使 $\tau=256$ 也只有 12.4 ms，占比不到 1%。这支持论文的结论：分块后 attention 依然是 compute-bound，KV 重读是可接受的代价，真正让 chunk 512 损失 25% 的是低算术强度与固定开销。

### 复算 5：简化调度模拟

把上面的常数塞进一个 40 行的模拟：decode 一步 4.18 ms，每 20 个 iteration 到达一条长度服从 lognormal 的 prompt（median 1730、p90 5696，对齐论文 Table 2 的 openchat_sharegpt4 trace），比较「整段 prefill 独占 iteration」与「token budget $\tau=512$」两种调度下，一条持续 decode 的请求所看到的 ITL 分布。

```python
import random
import statistics as st

CP = 35.4e-6      # s/prefill token：7B bf16 单张 H100，2N/P 按 40% MFU 折算
BASE = 4.18e-3    # s：decode-only 一步的权重读取下限（14 GB / 3.35 TB/s）
TAU = 512         # token budget
STEPS, EVERY = 4000, 20          # 每 20 个 iteration 到一个新请求

random.seed(0)
# 长度分布对齐论文 Table 2 的 openchat_sharegpt4：median 1730、p90 5696
prompts = [int(random.lognormvariate(7.456, 0.9303)) for _ in range(STEPS // EVERY)]

def run(chunked):
    arrivals = {1 + k * EVERY: L for k, L in enumerate(prompts)}
    queue, itl = [], []
    for step in range(STEPS):
        if step in arrivals:
            queue.append(arrivals[step])
        if chunked:                   # token budget：本 iteration 最多算 TAU 个 prefill token
            budget, served = TAU, 0
            for i, left in enumerate(queue):
                take = min(left, budget)
                queue[i] -= take
                budget -= take
                served += take
                if budget == 0:
                    break
        else:                         # prefill-prioritizing：整段 prefill 独占本 iteration
            served, queue = sum(queue), []
        itl.append((BASE + served * CP) * 1000)
    return itl

for chunked in (False, True):
    s = sorted(run(chunked))
    name = "chunked prefill (tau=512)" if chunked else "prefill-prioritizing (整段)"
    print(f"{name:26s} p50 {s[len(s)//2]:7.2f}  p99 {s[int(0.99*len(s))]:7.2f}  "
          f"max {s[-1]:7.2f}  mean {st.mean(s):6.2f}  (ms)")
```

输出（单位 ms）：

```text
prefill-prioritizing (整段)  p50    4.18  p99  137.25  max  606.72  mean   8.83  (ms)
chunked prefill (tau=512)  p50    4.18  p99   22.30  max   22.30  mean   8.83  (ms)
```

读法：均值完全相同（8.83 ms），因为两种调度的总 prefill 工作量一样；变化全在尾部——p99 从 137 ms 降到 22 ms（6.2×），max 从 607 ms 降到 22 ms（27×）。22.30 ms 正是 $4.18 + 512 \times 35.4\,\mu s$，也就是上界公式 $T_{\text{decode}} + \tau c$ 算出来的硬天花板；而 606.72 ms 对应模拟里最长的那条 17,021 token prompt（$17021 \times 35.4\,\mu s + 4.18\,\text{ms}$），说明不切分时尾延迟确实由「窗口里最长的那条 prompt」决定。真实系统里还要叠加排队、PP 通信、KV 读取，量级会更大，但形状不变。

### 论文的实验数字与条件

论文给的是**特定 trace、特定硬件、特定 SLO** 下的服务容量比，不是通用保证，引用时必须带上条件：

| 结论 | 条件（论文口径） |
| --- | --- |
| Mistral-7B：2.6× 服务容量 | 单张 A100 80GB，相对 vLLM，P99 TBT 约束下 |
| Yi-34B：最高 3.7× | 2×A100（TP2），严格 SLO 下比 vLLM 高 3.7×、比 Orca 高 4.0×（openchat_sharegpt4） |
| Falcon-180B：最高 5.6× | TP4-PP2、跨 2 节点 8×A100、100 Gbps 以太网；§5.3 给出严格 SLO 下 3.6×（相对 vLLM 的 hybrid-parallel 配置） |
| LLaMA2-70B：比 vLLM 高 4.3×、比 Orca 高 6.3× | TP4-PP2、8×A40 48GB（openchat_sharegpt4），收益主要来自 bubble 减少 |
| 严格 SLO 下比 vLLM 高 3.5× | Mistral-7B、100 ms P99 TBT、$\tau=512$ |
| 宽松 SLO 下比 vLLM 高 1.65× | Yi-34B、1 s P99 TBT、$\tau=2048$ |
| naive hybrid batching 使 TBT 最多涨 28.3× | 相对 decode-only batch，Figure 9 |
| chunk 512 使 prefill 耗时最多 +25%，chunk 2048 可忽略 | Yi-34B（TP2），Figure 14 |
| chunk 257 比 256 慢 32% | tile 量化，§4.3 |

注意摘要口径（2.6× / 3.7× / 5.6×）与正文分档数字（3.5× / 1.65×）并不矛盾：容量比随 SLO 松紧和 trace 长度分布变化很大，arxiv 那条 trace 的 prompt 中位数 7059 token，比 openchat 的 1730 长 4 倍，收益结构完全不同。另外，3.5× 的模型归属在论文里不统一：正文写的是 Mistral-7B（100 ms 严格 SLO、$\tau=512$），Figure 12 的图注却写成 Yi-34B，本表采用正文口径。

## 常见追问

- **追问**：chunk 大小（token budget）具体怎么定？
  - 要点：先用「P99 TBT SLO − decode 一步的时间」除以每 token 的 prefill 算力成本，得到理论上限；再把每 token 成本按 prefill 效率折损（chunk 512 时约 1.25 倍）放大后重算一遍，留出余量；然后取 2 的幂避开 tile 量化；最后在目标硬件上用离线 profile 或模拟器扫一遍，验证 P99 TBT 与容量。论文的做法是分层：严格 SLO 用 512，宽松用 2048（LLaMA2-70B 宽松档因为 PP bubble 用 1536）。
- **追问**：切碎之后，实现上有哪些状态必须小心？
  - 要点：三类。① KV cache 的写入位置要落在 PagedAttention 分好的 block 上，且 chunk 之间不能覆盖；② attention 要按**绝对位置**构造因果掩码并施加 RoPE，第 $i$ 个 chunk 的 query 只能看到前 $i$ 个 chunk 在内的全部 key，位置编码不能按 chunk 内相对位置重算；③ 每段的 token 数、序列长度元数据要在调度器与 kernel 之间保持一致，否则 batch 里混排的 prefill/decode 会算错。这就是论文要在 vLLM 上补 paged chunk prefill kernel（FlashAttention v2 / FlashInfer）的原因。
- **追问**：为什么它和投机解码会互相影响？
  - 要点：投机解码一次要验证 $\gamma+1$ 个 token，这部分算力同样吃 token budget。budget 太小会让一次验证放不下、投机收益归零；budget 太大又会让未命中的验证 token 放大 ITL。两者共用同一个预算，需要一起调。
- **追问**：为什么它和 prefix caching 会互相影响？
  - 要点：命中前缀后待 prefill 的 token 变少，chunk 数变少，TTFT 直接改善；但反过来，如果调度器为了凑满 budget 而把不同请求的 chunk 混在一起，缓存复用的边界与 block 对齐会变复杂。命中率高时 chunked prefill 的收益变小（本来就没有大 prefill 了），命中率低时它才是主要的尾延迟保护。
- **追问**：PP 部署下为什么更在意 chunk 的均匀性？
  - 要点：PP 用 micro-batch 填流水线，各 stage 依次处理同一 micro-batch。只要 micro-batch 之间的执行时长差异大，后面的 stage 就会空等（bubble）。chunked prefill 让每个 iteration 的 token 数接近常量 $\tau$，micro-batch 时长因此接近相等，bubble 显著减少——论文在 LLaMA2-70B 与 Falcon-180B 上的收益主要来自这一项。
- **追问**：有 chunked prefill 了，还需要 P/D 分离吗？
  - 要点：取决于瓶颈。chunked prefill 把干扰**压小但不消除**：同一个 batch 里仍有 prefill token，ITL 上界是 $T_{\text{decode}} + \tau c$，要更严的 SLO 就只能把 $\tau$ 调到很小，prefill 效率随之崩掉。P/D 分离直接消除干扰，prefill 可以整段跑、TTFT 最好，代价是 KV cache 跨机迁移和两类资源的独立扩缩容。混合流量、SLO 严格但集群互联一般时，chunked prefill 通常是更省事的第一选择（[[inference-serving-15]]）。

## 相关题目

- [[inference-serving-01]]：prefill 为什么 compute-bound、decode 为什么 memory-bound，以及拐点 $n^{*}$ 的来历。
- [[inference-serving-02]]：continuous batching 解决批的边界问题，是 chunked prefill 的前置条件。
- [[inference-serving-09]]：TTFT / TPOT / ITL 的定义与权衡，本题的收益与代价都落在这几个指标上。
- [[inference-serving-15]]：P/D 分离作为另一条消除 prefill-decode 干扰的路线及其适用条件。

## 参考资料与归属

1. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)（延伸），Amey Agrawal 等（Microsoft Research India / Georgia Tech，OSDI 2024），2024-03-04。chunked-prefills 与 stall-free batching 的定义、Algorithm 3 的填充顺序、token budget 的确定方式（含 Vidur 与 tile 量化）、实验条件与容量比（2.6× / 3.7× / 5.6× 等）、Table 2 的 trace 长度分布、Table 3 的 SLO 档位、Table 4 的 TTFT / P99 TBT 对照、Figure 14 的 chunking 开销，均来自该论文；摘要页、HTML 全文与 PDF 全文三处交叉核对。
2. 第 4 节「数值与代码验证」中的拐点、prefill 与 decode 耗时、token budget 换算、KV 重读量、以及那段调度模拟，是以 H100 / A100 的**公开规格**（bf16 稠密算力与 HBM 带宽）和 LLaMA-3-70B 级 KV cache 公式自行计算的，口径已在每张表前注明；论文没有给出这些换算的具体数值，两者相互独立。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
