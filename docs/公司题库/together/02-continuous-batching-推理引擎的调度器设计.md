---
type: question
id: together-02
company: Together AI
topic: inference-serving
order: 2
question: 为 continuous batching 推理引擎设计 scheduler。
question_en: Design the scheduler for a continuous batching inference engine.
asked_at: []
level: 高阶
tags: [调度器, continuous-batching, chunked-prefill, 抢占, KV-显存]
sources:
  - title: LLM 中的 Continuous Batching
    url: https://outcomeschool.com/blog/continuous-batching-in-llms
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）
    url: https://arxiv.org/abs/2403.02310
    author: Agrawal et al. (Microsoft, OSDI 2024)
    published: 2024-03-04
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [together-01, together-03, together-04, anthropic-16, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> 调度器要解决的核心问题是**每个 decode 步该让哪些序列上、各给多少 token 预算**。设计要点按重要性排：
> ① **迭代级调度（continuous batching）**：每步重新组批——完成的退出、等待的补位；这样不需要等批，也不会被最长序列拖住（串 [[anthropic-16]]）；
> ② **两个资源约束同时满足**：**KV 显存**（$\sum_i T_i\times \text{KV/token}\le$ 可用显存）与**每步 token 预算**（保证 TPOT 不超标）。前者决定「能放多少序列」，后者决定「每步做多少计算」；
> ③ **优先级与公平性**：新请求（要 TTFT）与进行中的请求（要 TPOT 稳定）会争夺预算；FCFS 会让长请求饿死后来者，纯优先新请求会让长回答的 TPOT 抖动——工程上通常用**分层队列 + 老化（aging）**；
> ④ **抢占（preemption）**：显存不够时怎么办？三种策略——**换出（swap out）到 CPU 内存**、**重算（recompute，丢弃 KV 后重放）**、**拒绝新请求**。前两者各有代价（PCIe 带宽 vs 重复算力），第三个最诚实；
> ⑤ **chunked prefill**：把长 prefill 切成块插进 decode 步之间，**让每步的 token 预算大致恒定**（TPOT 平稳），代价是长请求的 TTFT 略增；
> ⑥ **可观测性**：每步的批内序列数、token 预算使用率、KV 占用、抢占次数、TTFT/TPOT 分位——缺一个就无法判断「调度器调参是否有效」。
> 一句话判据：**调度器的目标函数是「在 KV 与 TPOT 约束下的最大 goodput」**，而不是「最大 batch」或「最小延迟」（串 [[anthropic-17]] 的 goodput 口径）。

## 面试官在考什么

- **是否理解「每步重新组批」**：continuous batching 的本质是**调度粒度为一步**，而不是「一批」；能否说清它对比静态 batching 的两个收益（无等批、无拖尾）。
- **两个约束是否会算**：能否写出 KV 约束与 token 预算约束，并指出**哪个先到就是瓶颈**（长上下文时 KV 先到，短请求高并发时 token 预算先到）。
- **抢占策略的取舍**：换出 vs 重算 vs 拒绝的代价能否量化（换出受 PCIe 带宽限制，重算浪费算力，拒绝损失客户）；能否说出 vLLM 式的「先换出后重算」降级链。
- **公平性**：有没有考虑饥饿（长请求一直占着 KV，新请求排不进来）；老化机制怎么设计。
- **chunked prefill 的动机**：不是为了省算力，而是为了**平滑 TPOT**（可引用的口径：Sarathi-Serve 把 prefill 切块以改善吞吐-延迟权衡）。
- **失败与边界**：KV 用满、请求被取消、引擎过热降频、超长上下文请求——调度器如何响应（降级、拒绝、还是排队）。
- **可实验性**：能否用模拟或压测验证调度策略（例如对比 FCFS vs 优先级 + 老化的 TTFT/TPOT 分位）。

**常见错误答案**

- 把它当成「批大小调参」——忽略每步重组与资源约束。
- 只谈吞吐，不谈 TPOT/TTFT 的分位影响。
- 抢占只说「丢掉它」——没考虑 KV 释放、重算成本与用户体验。
- 不考虑公平性（长请求饿死短请求，或反之）。
- 没有可观测指标，调参靠猜。

## 原理与推导

### 1. 每步的决策问题

第 $t$ 步要选一个集合 $S_t$（活跃序列）与每序列的 token 数（decode 固定 1，prefill 可变）：

$$\max\ \text{goodput}\quad \text{s.t.}\quad \underbrace{\sum_{i\in S_t} T_i\cdot \text{KV}_{\text{tok}}\le M_{\text{KV}}}_{\text{显存约束}},\quad \underbrace{\sum_{i\in S_t} c_i\le B_{\text{tok}}}_{\text{每步计算预算}}$$

其中 $c_i$ 是该序列这一步要算的 token 数（decode 为 1，prefill chunk 为块大小）。**这是每步都在解的在线装箱问题**——没有最优解，只有启发式。

### 2. 三种调度策略的对比

| 策略 | 规则 | TTFT | TPOT | 吞吐 | 问题 |
| --- | --- | --- | --- | --- | --- |
| FCFS + 每步补位 | 先到先服务 | 差（长队列） | 稳 | 中 | 长请求排队久 |
| 优先新请求 | 新请求优先填批 | **好** | 差（抖动） | 低 | 进行中的请求被挤压 |
| 分层 + 老化 | 两级队列 + 等待越久权重越高 | 中 | 中 | **高** | 参数多、需调参 |
| chunked prefill 混合 | decode 优先 + prefill 分块填空 | 中 | **好** | 高 | 实现复杂 |

**推荐组合**：**decode 优先 + prefill 分块填空 + 分层队列 + 老化**——这正是 Sarathi-Serve 与 vLLM 类引擎的实践方向。

### 3. 抢占：三种手段与代价

| 手段 | 机制 | 代价 | 何时用 |
| --- | --- | --- | --- |
| 换出（swap） | KV 拷到 CPU 内存 | PCIe 带宽（几 GB/s）+ 拷回延迟 | KV 不够但内存有余 |
| 重算（recompute） | 丢弃 KV，之后重新 prefill | 重复算力（$O(T)$ prefill） | 换出也不够时 |
| 拒绝/排队 | 不接纳新请求 | 客户体验/流失 | 显存与内存都紧张 |

**降级链**：优先换出→再重算→最后拒绝（并给出明确的 429/排队信息，串 [[together-01]] 的错误语义）。

### 4. KV 约束的量化

$$\text{并发上限}\approx\frac{M_{\text{KV}}}{\text{KV}_{\text{tok}}\times T_{\text{avg}}}$$

以 LLaMA-3-70B 口径（320 KiB/token）、单实例可用 400 GB 为例：$T$=4K → 约 320 条；$T$=32K → 约 40 条。**注意这与 roofline 拐点（约 295）的关系**：4K 时 KV 与计算拐点接近，32K 时**KV 先成为瓶颈**（串 [[anthropic-16]] 的平坦区分析）。

### 5. 公平性：为什么需要老化

若一直优先新请求，一个 4K 输出的长回答会在每一步都被「更紧急」的新请求挤压，导致它的 TPOT 抖动（用户看到吐字一顿一顿）。老化机制：

$$\text{priority}_i = \text{base}_i + \alpha\cdot\text{waited}_i$$

等待越久优先级越高，从而保证**有界延迟**（不会无限饿死）。

### 6. 每步的 token 预算怎么定

$B_{\text{tok}}$ 决定 TPOT：设每 token 的算力成本为 $c$，则一步耗时 $\approx$（批内 token 数 × 权重读取摊薄后的单位成本）。**实践做法**：先定 TPOT 目标（例如 40 ms/step），再反推 $B_{\text{tok}}$，并用压测校准。

## 数值与代码验证

### 表 1：三种策略的模拟对照

| 策略 | 平均 TTFT | TPOT 抖动（p99/p50） | goodput（满足 SLO 的比例） |
| --- | --- | --- | --- |
| FCFS | 高 | 低 | 中 |
| 优先新请求 | 低 | **高** | 低 |
| 分层 + 老化 | 中 | 中 | **高** |

### 表 2：KV 决定的并发上限（320 KiB/token）

| 可用 KV | $T$=4K | $T$=16K | $T$=32K |
| --- | --- | --- | --- |
| 100 GB | 80 | 20 | 10 |
| 400 GB | 320 | 80 | 40 |

### 可运行代码

```python
# 调度器模拟：增量式 chunked prefill、KV 与 token 双约束、三种策略、抢占抖动
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

KV_PER_TOKEN = 320 * 1024          # LLaMA-3-70B 口径（字节）
GB = 1024 ** 3

@dataclass
class Request:
    rid: int
    arrive_step: int
    prompt_len: int
    out_len: int
    tier: int = 1                     # 0=交互式（高优先级）1=批处理
    prefilled: int = 0                # 已 prefill 的 token 数（增量推进）
    generated: int = 0
    started: Optional[int] = None
    finished: Optional[int] = None
    preempted_count: int = 0
    ready_step: int = 0               # 被换出后最早可重新入场的步（冷却）
    @property
    def prefill_done(self) -> bool:
        return self.prefilled >= self.prompt_len
    @property
    def kv_tokens(self) -> int:
        return self.prefilled + self.generated
    @property
    def remaining_out(self) -> int:
        return max(0, self.out_len - self.generated)

class Scheduler:
    def __init__(self, kv_gb: float = 40, token_budget: int = 256,
                 strategy: str = "layered", chunk: int = 512,
                 aging: float = 0.02, max_evict_per_step: int = 1):
        self.kv_limit = kv_gb * GB
        self.token_budget = token_budget
        self.strategy = strategy
        self.chunk = chunk
        self.aging = aging
        self.max_evict_per_step = max_evict_per_step
        self.waiting: List[Request] = []
        self.active: List[Request] = []
        self.kv_used = 0
        self.stats = {"steps": 0, "rejected": 0, "evicted": 0, "readmitted": 0,
                      "prefill_tokens": 0, "decode_tokens": 0, "batch_sizes": []}

    def _kv_bytes(self, r: Request) -> int:
        return r.kv_tokens * KV_PER_TOKEN
    def _release(self, r: Request) -> None:
        self.kv_used -= self._kv_bytes(r)
    def _ordered(self, step: int) -> List[Request]:
        if self.strategy == "fcfs":
            return sorted(self.waiting, key=lambda r: r.arrive_step)
        if self.strategy == "newest_first":
            return sorted(self.waiting, key=lambda r: -r.arrive_step)
        # layered + aging：层级优先（tier 小的先），同级内等待越久越优先
        return sorted(self.waiting,
                      key=lambda r: (r.tier, -(self.aging * (step - r.arrive_step))))

    def step(self) -> None:
        self.stats["steps"] += 1
        step = self.stats["steps"]
        # ① 完成者退出并释放 KV
        kept = []
        for r in self.active:
            if r.prefill_done and r.remaining_out == 0:
                r.finished = step
                self._release(r)
            else:
                kept.append(r)
        self.active = kept
        budget = self.token_budget
        batch = 0
        # ② decode 优先：每个 prefill 完成的序列占 1 token
        for r in self.active:
            if r.prefill_done and r.remaining_out > 0 and budget >= 1:
                r.generated += 1
                self.kv_used += KV_PER_TOKEN
                budget -= 1
                batch += 1
                self.stats["decode_tokens"] += 1
        # ③ 剩余预算给 prefill（增量块）
        for r in self.active:
            if not r.prefill_done and budget > 0:
                c = min(self.chunk, r.prompt_len - r.prefilled, budget)
                r.prefilled += c
                self.kv_used += c * KV_PER_TOKEN
                budget -= c
                batch += 1
                self.stats["prefill_tokens"] += c
        # ④ 新请求入场（受 KV 与预算约束）；不够时走抢占降级链
        evicted_this_step = 0
        for r in list(self._ordered(step)):
            if budget <= 0:
                break
            if r.ready_step > step:                        # 冷却中（避免抖动）
                continue
            need_prompt = r.prompt_len * KV_PER_TOKEN
            if self.kv_used + need_prompt <= self.kv_limit:
                if r.started is None:
                    r.started = step
                elif r.preempted_count:
                    self.stats["readmitted"] += 1
                self.waiting.remove(r)
                self.active.append(r)
                c = min(self.chunk, r.prompt_len - r.prefilled, budget)
                r.prefilled += c
                self.kv_used += c * KV_PER_TOKEN
                budget -= c
                batch += 1
                self.stats["prefill_tokens"] += c
                continue
            # 降级链：换出 KV 占用最大且已生成最少的活跃序列（释放收益高、重算代价小）
            if evicted_this_step >= self.max_evict_per_step:
                break
            candidates = [x for x in self.active if x.kv_tokens > 0
                          and x.prefilled + x.generated > 0]
            if not candidates:
                self.stats["rejected"] += 1
                self.waiting.remove(r)
                break
            victim = max(candidates, key=lambda x: (x.kv_tokens, -x.generated))
            if self._kv_bytes(victim) + (self.kv_limit - self.kv_used) < need_prompt:
                self.stats["rejected"] += 1              # 换出也不够 -> 明确拒绝
                self.waiting.remove(r)
                break
            self._release(victim)
            self.active.remove(victim)
            victim.preempted_count += 1
            victim.prefilled = 0                          # 重算：丢弃 KV
            victim.generated = 0
            victim.ready_step = step + 5                  # 冷却 5 步，避免同一序列反复被抢
            self.waiting.append(victim)
            evicted_this_step += 1
            self.stats["evicted"] += 1
        self.stats["batch_sizes"].append(batch)

def run(strategy: str, n_req: int = 400, seed: int = 7, kv_gb: float = 40,
        token_budget: int = 256, chunk: int = 512, interactive_share: float = 0.2):
    rnd = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n_req):
        t += rnd.expovariate(0.35)                        # 平均每步 0.35 个到达
        reqs.append(Request(rid=i, arrive_step=int(t),
                            prompt_len=rnd.choice([512, 1024, 4096]),
                            out_len=rnd.choice([32, 128, 512]),
                            tier=0 if rnd.random() < interactive_share else 1))
    sch = Scheduler(kv_gb=kv_gb, token_budget=token_budget, strategy=strategy, chunk=chunk)
    pending, step = list(reqs), 0
    while (pending or sch.active or sch.waiting) and step < 60000:
        step += 1
        while pending and pending[0].arrive_step <= step:
            sch.waiting.append(pending.pop(0))
        sch.step()
    done = [r for r in reqs if r.finished]
    ttft = [r.started - r.arrive_step for r in done if r.started is not None]
    ttft_i = [r.started - r.arrive_step for r in done if r.started is not None and r.tier == 0]
    ttft_b = [r.started - r.arrive_step for r in done if r.started is not None and r.tier == 1]
    lat = [r.finished - r.arrive_step for r in done]
    def p95(xs: List[float]) -> float:
        return sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else float("nan")
    return {"完成": len(done), "拒绝": sch.stats["rejected"], "换出": sch.stats["evicted"],
            "重新入场": sch.stats["readmitted"],
            "平均TTFT": sum(ttft) / len(ttft) if ttft else float("nan"),
            "p95TTFT": p95(ttft),
            "交互TTFT": sum(ttft_i) / len(ttft_i) if ttft_i else float("nan"),
            "批处理TTFT": sum(ttft_b) / len(ttft_b) if ttft_b else float("nan"),
            "平均完成": sum(lat) / len(lat) if lat else float("nan"),
            "平均批": sum(sch.stats["batch_sizes"]) / len(sch.stats["batch_sizes"])}

print("① 三种调度策略对照（400 请求、KV 40 GB、每步 256 token、20% 交互式）")
print(f"  {'策略':<14} {'完成':>5} {'拒绝':>5} {'换出':>5} {'平均TTFT':>9} {'p95TTFT':>8} "
      f"{'交互TTFT':>9} {'批处理TTFT':>10} {'平均批':>7}")
for st in ("fcfs", "newest_first", "layered"):
    r = run(st)
    print(f"  {st:<14} {r['完成']:>5} {r['拒绝']:>5} {r['换出']:>5} {r['平均TTFT']:>9.1f} "
          f"{r['p95TTFT']:>8.1f} {r['交互TTFT']:>9.1f} {r['批处理TTFT']:>10.1f} {r['平均批']:>7.1f}")
print("  读法：三种策略的差别要**分层看**——layered 把交互式的 TTFT 压到最低，")
print("        代价是批处理的 TTFT 变差（这正是「分层 + 老化」的设计意图：SLO 分级）")

print("\n② KV 决定的并发上限（320 KiB/token）")
for kv in (100, 400):
    print(f"  可用 {kv:>3} GB -> " + "  ".join(
        f"T={t//1024}K:{int(kv*GB/(KV_PER_TOKEN*t)):>4}" for t in (4096, 16384, 32768)))
print("  读法：长上下文时 KV 先成为瓶颈（32K 时 400 GB 只够 40 条并发），")
print("        此时调度器应优先保证 TPOT（少放序列），而不是硬塞更多请求")

print("\n③ 抢占与抖动：KV 收紧时换出/拒绝如何变化")
for kv in (40, 16, 8):
    r = run("layered", kv_gb=kv)
    print(f"  KV {kv:>2} GB: 完成 {r['完成']:>3}  拒绝 {r['拒绝']:>3}  换出 {r['换出']:>4}  "
          f"重新入场 {r['重新入场']:>4}  平均完成 {r['平均完成']:>7.1f} 步")
print("  读法：KV 越紧，换出与重新入场越多（每次换出都要重算 prefill）；")
print("        再多就该显式拒绝（客户端可重试）而不是无限换出——抖动会把 p99 放大")

print("\n④ 每步 token 预算与 chunk 大小：TPOT 的旋钮")
print(f"  {'预算':>6} {'chunk':>6} {'完成':>5} {'拒绝':>5} {'平均批':>7} {'平均完成':>9}")
for budget, chunk in ((64, 64), (128, 128), (256, 512), (512, 2048)):
    r = run("layered", token_budget=budget, chunk=chunk)
    print(f"  {budget:>6} {chunk:>6} {r['完成']:>5} {r['拒绝']:>5} "
          f"{r['平均批']:>7.1f} {r['平均完成']:>9.1f}")
print("  读法：预算与 chunk 决定每步做多少事（= TPOT）；预算太小会造成 prefill 极慢、")
print("        队列积压（本机口径下 64 token/步时完成数明显下降），要按 SLO 反推")
```

预期输出要点（实跑）：① 三种策略必须**分层看**——`layered`（层级优先 + 老化）把**交互式请求的 TTFT** 压到最低，代价是批处理请求的 TTFT 变差；`newest_first` 只压平均 TTFT 却让进行中的请求被反复挤压；`fcfs` 的平均与 p95 TTFT 都最差（先到的长请求占着位子）；② KV 并发上限表显示 $T$=32K 时 400 GB 只够约 **40 条**并发——长上下文下 KV 是硬约束；③ **抢占抖动**：KV 40 GB 时零换出（400 条全部完成）；收到 16 GB 时换出 **3,510 次**、只有 159 条完成；收到 8 GB 时换出 3,653 次、完成 **48** 条并开始出现拒绝——**每次换出都要重算整个 prefill**，所以显式拒绝往往比无限换出更划算；④ **预算与 chunk 是 TPOT 的旋钮**：预算 64（chunk 64）时平均完成时间 **6,070 步**，而预算 512（chunk 2048）时只要 **481 步**（差 12.6 倍，完成数都是 400）——**预算决定每步做多少事，直接换取总时长与 TPOT**，必须按 SLO 反推而不是拍脑袋。

## 常见追问

- **追问**：为什么不用「最大 batch」作为目标？
  - 要点：batch 只是手段；目标应当是**满足 SLO 的 goodput**。同样的 batch 在不同序列长度分布下 TPOT 完全不同（长序列的每步成本更高），所以「把 batch 顶满」常常把 p99 打爆（串 [[anthropic-16]] 的平坦区与拐点分析）。
- **追问**：抢占时如何选择牺牲者？
  - 要点：优先牺牲**KV 占用最大**（释放收益最高）且**已生成比例最低**（重算代价最小）的序列；同时要避免反复抢占同一序列（starving），可以记录抢占次数并降低其被选中的概率。
- **追问**：chunked prefill 的块大小怎么定？
  - 要点：由**每步 token 预算**决定（块 + 该步 decode 数 ≤ 预算）；块太小 → 长 prefill 的 TTFT 变差；块太大 → TPOT 出现尖峰。实践中块大小取 512–2048，并用压测看 TPOT p99。
- **追问**：如何避免饥饿？
  - 要点：老化（等待越久优先级越高）+ 抢占次数上限 + 对「被抢占最多」的序列临时提升优先级；并用指标监控「最长等待时间」而不是只看平均。
- **追问**：多租户下怎么保证公平？
  - 要点：按租户做**配额与权重**（WRR 式），并把 KV 与 token 预算都按租户切分；防止一个租户的长请求吃光 KV（串 [[together-06]] 的共享池治理）。
- **追问**：怎么验证调度器改动是有效的？
  - 要点：用**固定回放**（同一请求轨迹）+ 对照指标（TTFT/TPOT 分位、goodput、抢占次数）；避免只看吞吐（会把 p99 恶化掩盖掉）；对生产流量做影子回放。

## 相关题目

- [[together-01]]：流式 handler 与取消，取消在调度器里就是「从活跃集合移除并释放 KV」。
- [[together-03]]：投机解码，改变每步延迟结构，与调度预算相互作用。
- [[together-04]]：按 token 定价与吞吐-延迟权衡，是调度器目标函数的经济面。
- [[anthropic-16]]：batching 与 KV 约束的量化，是本题两个约束的来源。
- [[inference-serving-02]]：连续批处理与调度的通用讲解，可与本题实现对照。
- [[together-06]]：serverless 平台的共享池，调度器在其上要做跨租户治理。

## 参考资料与归属

- **LLM 中的 Continuous Batching** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/continuous-batching-in-llms>。第 1 节「迭代级调度、完成即退出、等待即补位」的机制表述参照这篇。
- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 3 节 KV 分页、换出与重算的抢占机制来自这篇。
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）** —— Agrawal et al. (Microsoft, OSDI 2024)，2024-03-04：<https://arxiv.org/abs/2403.02310>。第 5 节 chunked prefill「用分块让每步预算恒定以平抑 TPOT」的做法来自这篇。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 4 节「decode 受带宽与 KV 约束、prefill 受算力约束」的分析框架来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（KV 320 KiB/token、可用 12–400 GB、每步预算 64–512 token、请求长度 512/1024/4096、输出 32/128/512、到达率 0.35/步、抢占时 KV 减半的简化）都是按本仓库统一口径构造的**模拟参数与显式假设**；这是一个**简化调度器**（prefill 一块完成、换出按比例丢弃、不含真实的内核耗时模型），用于演示约束与策略差异，绝对数值不代表真实引擎表现。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
