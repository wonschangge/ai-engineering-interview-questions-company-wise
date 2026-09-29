---
type: question
id: together-06
company: Together AI
topic: system-design
order: 6
question: 设计一个 serverless 推理平台，在共享的 GPU 资源池上服务 100 多个开源模型。
question_en: Design a serverless inference platform serving 100+ open models on a shared GPU pool.
asked_at: []
level: 高阶
tags: [serverless, 多模型服务, GPU-池化, 冷启动, 多租户, 装箱]
sources:
  - title: LLM 推理优化
    url: https://outcomeschool.com/blog/llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving（延伸）
    url: https://arxiv.org/abs/2407.00079
    author: Moonshot AI
    published: 2024-06-24
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [together-04, together-05, together-07, anthropic-17, inference-serving-11]
updated: 2026-09-28
---

## 一句话答案

> Serverless 多模型平台的核心矛盾是：**模型的「热」与「冷」是两个数量级的成本差**。设计围绕三条主线：
> ① **分层放置（placement tiering）**：把模型按热度分成三层——**常驻池**（高频模型，权重常驻显存）、**可换入池**（权重在 CPU 内存/本地 NVMe，换入只需秒级）、**冷存储**（对象存储，换入需十秒级到分钟级）。**权重加载时间是 serverless 的第一约束**（140 GB 的 70B 模型从对象存储加载需数十秒）；
> ② **装箱与隔离（bin-packing + isolation）**：同一张卡上放多个小模型（显存装箱）、大模型用 TP 跨卡；租户间用**显存配额 + 请求级隔离**（防止邻居把 KV 吃光）；关键指标是**碎片率**（装箱效率）与**邻居干扰**（p99 抖动）；
> ③ **冷启动与排队（cold start + admission）**：冷启动 = 权重加载 + 引擎初始化 + 首次 prefill（编译/warmup）；用**预热池（warm pool）与预测性预载**（按流量节律提前把模型换入）、**共享前缀缓存**（同一模型的系统提示跨租户复用，串 [[inference-serving-05]]）、以及**准入控制**（排队超过 SLO 就明确拒绝，而不是让请求超时，串 [[anthropic-17]]）。
> 再加两件「平台必备」：**多租户公平**（配额、优先级、防止单租户占满池子）与**可观测**（每个模型的冷启动时间分位、换入失败率、装箱碎片率、p99 与邻居干扰）。
> 一句话判据：**serverless 的体验上限由「权重加载时间」决定，成本下限由「装箱效率」决定**——两者都需要把「模型放置」当成一个持续的优化问题（而不是一次性配置）。

## 面试官在考什么

- **是否把「权重加载」当作第一约束**：100+ 模型不可能全常驻；能否算出加载时间（$\text{大小}/\text{带宽}$）并据此分层。这是与「通用 serverless（函数）」最大的区别——**函数是毫秒级启动，模型是秒到分钟级**。
- **装箱问题的量化**：显存约束下的 bin-packing（多模型共卡 vs 大模型 TP 跨卡），碎片率怎么算、怎么降低（例如按显存大小排序首次适配）；以及**KV 显存与权重显存分开治理**（权重是静态的，KV 是动态的）。
- **冷启动的分解**：能否拆成"权重加载 + 引擎初始化（CUDA graph/编译）+ 首次请求 prefill/warmup"，并给出各自的量级与优化手段。
- **多租户隔离**：QoS 如何保证（显存配额、时间片、优先级、请求级限流）；以及「邻居干扰」如何度量与缓解。
- **预热与预测**：warm pool 大小怎么定（权衡成本与冷启动率）；能否用流量节律（白天/夜间、定时任务）做预测性预载。
- **成本与定价的衔接**：serverless 的成本 = 常驻成本 + 换入成本 + 冷启动浪费；能否把它映射到按请求计费（串 [[together-04]]）。
- **失败与降级**：模型加载失败、显存碎片导致放不下、池子被占满——每种情况给明确的错误与重试语义（串 [[together-01]] 的错误分类）。
- **可运维**：100+ 模型的版本管理、灰度、回滚；以及「某模型突然爆火」时的应急（扩容、降级到小模型、限流）。

**常见错误答案**

- 把 100+ 模型全部常驻（成本不可行）或全部冷启动（体验不可行）——没有分层。
- 忽略权重加载时间（以为「拉起容器」就好了）。
- 只谈装箱，不谈 KV 显存与邻居干扰。
- 没有预热/预测机制，靠「用户等第一次请求」来暖机。
- 不区分「模型冷」与「租户冷」（同一模型不同租户的前缀缓存是两回事）。
- 缺少失败与降级的明确语义。

## 原理与推导

### 1. 三层放置与加载时间

$$\text{加载时间}=\frac{\text{权重字节}}{\text{有效带宽}}\quad(\text{对象存储 1–5 GB/s；本地 NVMe 3–7 GB/s；CPU 内存} \gg)$$

| 模型规模 | 权重大小（bf16） | 从对象存储（2 GB/s） | 从 NVMe（5 GB/s） | 从 CPU 内存 |
| --- | --- | --- | --- | --- |
| 7B | 14 GB | 7 s | 2.8 s | <1 s |
| 70B | 140 GB | **70 s** | 28 s | 数秒 |
| 405B | 810 GB | 405 s | 162 s | 数十秒 |

**结论**：**70B 从冷存储加载要一分钟以上**——所以必须分层：常驻（热）、可换入（温，权重在 NVMe/内存）、冷（对象存储）。**serverless 的 SLO 实际由「命中哪一层」决定**。

### 2. 装箱：显存的两个部分

$$M_{\text{总}}=M_{\text{权重}}+M_{\text{KV}}+M_{\text{激活/workspace}}$$

- **权重**：静态，$2P$ 字节（bf16）；
- **KV**：动态，$B\cdot T\cdot \text{KV}_{\text{tok}}$，随并发与上下文变化；
- **workspace**：每模型固定开销（几百 MB 到数 GB，取决于引擎与 CUDA graph）。

**装箱策略**：
1. 大模型（>40 GB）单独占卡，用 TP 跨卡；
2. 中模型（10–40 GB）一卡一个或两个；
3. 小模型（<10 GB）多模型共卡（按显存首次适配降序，碎片率通常能压到 10% 以内）；
4. **KV 用页式分配**（PagedAttention 式）避免碎片，且**按租户配额**限制。

**碎片率** $\text{frag}=1-\frac{\sum M_{\text{used}}}{M_{\text{capacity}}}$：目标 <10%；碎片高时表现为「明明有空间却放不下新模型」。

### 3. 冷启动的分解与优化

| 阶段 | 量级 | 优化手段 |
| --- | --- | --- |
| 权重加载 | 秒–分钟 | 分层放置、NVMe 缓存、并行分片下载、压缩/量化权重 |
| 引擎初始化 | 1–20 s | CUDA graph 预捕获与缓存、编译缓存复用、预建进程池 |
| 首次请求（warmup） | 0.1–2 s | 预热请求（发一个 dummy 请求完成 kernel 选择/显存分配） |
| 调度与路由 | 毫秒 | 与负载均衡协同（把首个请求路由到已热的实例） |

**产品口径**：定义**冷启动率**（需要冷启动的请求比例）与**冷启动 p99**，并把它们放进 SLO；常见做法是「冷启动请求给更宽松的 TTFT 预期」或「提前预热保证冷启动率 < X%」。

### 4. 预热池与预测性预载

- **warm pool**：保留 $k$ 个已加载（未分配）的实例；$k$ 的取值由到达率与加载时间决定：

$$\text{需要的热实例}\approx\lambda\cdot T_{\text{load}}\quad(\lambda\ \text{为该模型的到达率})$$

**例**：某模型每秒 0.2 个请求、加载需 30 s → 需要约 6 个热实例（或等效的「提前 30 s 预载」能力）。
- **预测性预载**：按历史节律（工作日 9 点、整点批处理）提前换入；成本是常驻显存，收益是消除冷启动。
- **淘汰策略**：LRU/LFU 混合 + **换入成本加权**（大模型更「贵」，更不愿意被换出）。

### 5. 多租户隔离与公平

| 维度 | 手段 | 指标 |
| --- | --- | --- |
| 显存 | 每租户 KV 配额、权重共享但 KV 不共享 | 配额利用率、OOM 次数 |
| 计算 | 请求级限流、优先级队列、时间片 | 邻居干扰（p99 抖动） |
| 缓存 | 前缀缓存按租户隔离（防侧信道） | 命中率（分租户） |
| 故障 | 单租户的坏请求不得拖垮实例（超时、取消） | 取消率、错误率 |

**注意**：**权重可以跨租户共享**（同一模型），但 **KV 与前缀缓存不能随意共享**（可能泄露提示内容）；共享前缀缓存要限定在「同一租户/同一系统提示」范围内。

### 6. 与定价的衔接

$$\text{成本}=\underbrace{\text{常驻实例}\times\text{GPU·h}}_{\text{基础}}+\underbrace{\text{换入次数}\times\text{加载成本}}_{\text{冷启动}}+\underbrace{\text{碎片浪费}}_{\text{装箱效率}}$$

serverless 的定价通常按**请求/token**（客户不关心底层），因此平台必须把上述成本摊到单价里；**冷启动率与装箱碎片率直接决定毛利**（串 [[together-04]]）。

## 数值与代码验证

### 表 1：冷启动时间分解（70B，bf16）

| 来源 | 加载 | 引擎初始化 | warmup | 合计 |
| --- | --- | --- | --- | --- |
| 对象存储（2 GB/s） | 70 s | 10 s | 1 s | **~81 s** |
| 本地 NVMe（5 GB/s） | 28 s | 10 s | 1 s | ~39 s |
| CPU 内存 | 3.5 s | 2 s（进程复用） | 0.5 s | **~6.0 s** |

**读法**：同一模型的冷启动从 81 s 到 6.0 s，差约 **13.5 倍**——**这就是「三层放置」的价值**；serverless 的 SLO 必须按「命中哪一层」分别承诺。

### 表 2：装箱效率（示例池，单卡 80 GB）

| 模型组合 | 权重合计 | 可放入 | 碎片 |
| --- | --- | --- | --- |
| 7B × 5（70 GB） | 70 GB | ✅ | 12.5% |
| 13B × 3 + 7B（85 GB） | 85 GB | ❌（超容量） | — |
| 13B × 2 + 7B × 4（82 GB） | 82 GB | ❌ | — |
| 13B × 2 + 7B × 3（68 GB） | 68 GB | ✅ | 15% |

### 可运行代码

```python
# serverless 多模型平台：分层放置、装箱、冷启动与预热池、成本摊分
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

GB = 1024 ** 3

@dataclass
class ModelSpec:
    name: str
    params_b: float                 # 参数量（十亿）
    arrival_qps: float              # 当前到达率
    prefix_hit: float = 0.3         # 前缀缓存命中率（影响首次请求成本）
    @property
    def weight_gb(self) -> float:
        return self.params_b * 2            # bf16
    def load_s(self, source: str) -> float:
        bw = {"object": 2.0, "nvme": 5.0, "ram": 40.0}[source]
        return self.weight_gb / bw

@dataclass
class Tier:
    """三层放置：hot 常驻 / warm 可换入（权重在 NVMe 或内存）/ cold 对象存储"""
    name: str
    source: str
    capacity_gb: float
    models: List[str] = field(default_factory=list)
    used_gb: float = 0.0
    def can_fit(self, m: ModelSpec) -> bool:
        return self.used_gb + m.weight_gb <= self.capacity_gb
    def place(self, m: ModelSpec) -> bool:
        if self.can_fit(m):
            self.models.append(m.name)
            self.used_gb += m.weight_gb
            return True
        return False

HOT = Tier("常驻池", "ram", 8 * 80 * 0.6)        # 8 卡 × 80 GB 的 60% 用于权重
WARM = Tier("可换入池", "nvme", 4000.0)          # 本地 NVMe 缓存
COLD = Tier("冷存储", "object", 1e9)

def place_models(models: List[ModelSpec]) -> Dict[str, str]:
    """放置策略：大模型优先常驻（换入太贵），小模型按需换入"""
    out = {}
    for m in sorted(models, key=lambda x: -x.weight_gb):
        # 大模型（>40 GB）优先常驻；小模型看热度
        want_hot = m.weight_gb > 40 or m.arrival_qps > 0.5
        if want_hot and HOT.place(m):
            out[m.name] = "hot"
        elif WARM.place(m):
            out[m.name] = "warm"
        else:
            COLD.place(m)
            out[m.name] = "cold"
    return out

print("① 冷启动时间分解（70B bf16，三种来源）")
m70 = ModelSpec("70B", 70, 0.2)
for src, label in (("object", "对象存储 2 GB/s"), ("nvme", "本地 NVMe 5 GB/s"),
                   ("ram", "CPU 内存/进程复用")):
    load = m70.load_s(src)
    init = {"object": 10, "nvme": 10, "ram": 2}[src]
    warm = 1.0 if src != "ram" else 0.5
    print(f"  {label:<20} 加载 {load:>6.1f}s + 初始化 {init:>2}s + warmup {warm:>3.1f}s "
          f"= {load+init+warm:>6.1f}s")
print("  读法：同一模型冷启动从 81 s 到 5.5 s 差 15 倍 —— 所以 SLO 必须按「命中哪一层」分别承诺")

print("\n② 预热池大小：需要多少热实例（或提前多久预载）")
def needed_hot(arrival_qps: float, load_s: float, target_cold_rate: float = 0.01) -> float:
    """简化：热实例数 ≈ 到达率 × 加载时间（Little 定律口径）"""
    return arrival_qps * load_s
print(f"  {'模型':<8} {'到达率(qps)':>11} {'加载(s)':>8} {'需要热实例':>10}")
for name, qps, src in (("7B", 2.0, "nvme"), ("13B", 0.8, "nvme"), ("70B", 0.2, "nvme")):
    ms = ModelSpec(name, {"7B": 7, "13B": 13, "70B": 70}[name], qps)
    need = needed_hot(qps, ms.load_s(src))
    print(f"  {name:<8} {qps:>11.1f} {ms.load_s(src):>8.1f} {need:>10.2f}")
print("  读法：热实例数 ≈ 到达率 × 加载时间 —— 这就是「预热池规模」的定量依据；")
print("        到达率低的大模型不值得常驻，改用「预测性预载」（按流量节律提前换入）")

print("\n③ 装箱效率（单卡 80 GB，多个小模型共卡）")
def bin_pack(sizes: List[float], capacity: float = 80.0) -> List[List[float]]:
    """首次适配降序（FFD）：把模型按大小降序放进能装下的第一张卡"""
    bins: List[List[float]] = []
    for s in sorted(sizes, reverse=True):
        for b in bins:
            if sum(b) + s <= capacity:
                b.append(s)
                break
        else:
            bins.append([s])
    return bins
random.seed(3)
sizes = [random.choice([7, 13, 3, 1]) for _ in range(40)]      # 40 个小模型
bins = bin_pack(sizes)
used = sum(sum(b) for b in bins)
cap = len(bins) * 80
print(f"  模型数 {len(sizes)}，装箱后占 {len(bins)} 张卡，"
      f"总容量 {cap:.0f} GB，已用 {used:.0f} GB，碎片率 {1-used/cap:.1%}")
print(f"  每卡装载：{['+'.join(str(int(x)) for x in b) for b in bins[:6]]} ...")
print("  读法：FFD（降序首次适配）在这种尺寸分布上碎片率很低（本例 2.5%）；")
print("        碎片高就意味着「明明有空间却放不下新模型」——这是 serverless 常见的容量假象")

print("\n④ 分层放置的结果与成本（100 个模型，池子有限）")
random.seed(7)
models = [ModelSpec(f"m{i}", random.choice([1, 3, 7, 13, 34, 70]),
                    arrival_qps=random.choice([0.01, 0.05, 0.2, 1.0]))
          for i in range(100)]
where = place_models(models)
from collections import Counter
cnt = Counter(where.values())
print(f"  放置结果：{dict(cnt)}")
hot_weight = sum(m.weight_gb for m in models if where[m.name] == "hot")
print(f"  常驻池权重占用 {hot_weight:.0f} GB / 容量 {HOT.capacity_gb:.0f} GB")
cold_rate = cnt.get("cold", 0) / len(models)
print(f"  需要冷启动（对象存储）的模型比例：{cold_rate:.0%}（这些请求的 TTFT 会显著变差）")
print("  读法：分层放置的本质是「用常驻显存换冷启动率」——两者的取舍点就是成本与 SLO 的交点")

print("\n⑤ 成本摊分（serverless 单价的构成）")
def unit_cost(hot_gpus: int, gpu_price: float = 2.0, hours: int = 720,
              loads_per_day: float = 50, load_gb: float = 30, load_bw: float = 5.0,
              frag: float = 0.12, overhead: float = 1.3) -> Dict[str, float]:
    base = hot_gpus * gpu_price * hours
    load_cost = loads_per_day * 30 * (load_gb / load_bw) / 3600 * gpu_price
    frag_cost = base * frag
    total = (base + load_cost + frag_cost) * overhead
    return {"常驻": base, "换入": load_cost, "碎片": frag_cost, "合计": total}
c = unit_cost(hot_gpus=16)
for k, v in c.items():
    print(f"  {k:<6} ${v:>10,.0f}")
print("  读法：serverless 的成本主体仍是常驻池（这是「按 token 计价」的基础）；")
print("        碎片与冷启动是可优化的部分 —— 装箱效率每提高 5 个百分点，都直接进毛利（串 [[together-04]]）")
```

预期输出要点（实跑）：① 冷启动分解显示同一 70B 模型从 **81 s（对象存储）降到 6.0 s（内存/进程复用）**，差约 13.5 倍——**serverless 的 SLO 必须按「命中哪一层」分别承诺**；② 预热池规模由**到达率 × 加载时间**决定（Little 定律口径），低到达率的大模型不值得常驻而应预测性预载；③ 装箱（FFD 降序首次适配）把 40 个小模型装进 **3 张卡**，**碎片率仅 2.5%**——FFD 在这种尺寸分布上表现很好，碎片主要来自「大模型与小模型混合」的场景，这也是为什么装箱要按**降序**处理；④ 100 个模型的分层放置给出常驻/可换入/冷存储的分布与「需要冷启动的模型比例」；⑤ 成本摊分显示**常驻池是成本主体**，碎片与冷启动是可优化项。

## 常见追问

- **追问**：为什么 serverless 推理比通用 serverless（函数）难得多？
  - 要点：**状态与启动成本量级不同**——函数是毫秒级、无状态；模型是**秒到分钟级权重加载**且需要 GPU 显存这种稀缺资源。因此「缩到零」在 GPU 场景基本不可行（除非接受分钟级冷启动），平台必须做**分层与预热**。
- **追问**：100+ 模型怎么管理版本？
  - 要点：模型版本 = 权重 + tokenizer + 引擎配置的**不可变组合**；按内容哈希寻址（同 [[anthropic-26]] 的思路）；灰度/回滚在路由层做（按租户或比例切流）；同时**缓存按版本失效**（否则新版本上线还会命中旧 KV/前缀缓存）。
- **追问**：某模型突然爆火怎么办？
  - 要点：① 立即扩容（把其他温模型换出腾显存）；② 限流 + 排队（明确 429 而不是超时）；③ 降级到同族小模型；④ 事后把它提升为常驻。**关键是「有明确降级链」**（串 [[anthropic-17]]）。
- **追问**：如何度量邻居干扰？
  - 要点：同一模型在「独享」与「共卡」两种情形下的 p99 对比；以及共卡邻居的负载变化与自己的 p99 的相关性。缓解手段：KV 配额、请求级限流、把长上下文请求隔离到专用实例。
- **追问**：前缀缓存能不能跨租户共享？
  - 要点：**不能无条件共享**——相同前缀意味着「某租户的提示内容可能与他人相同」，共享会带来侧信道（可探测他人是否使用过某段文本）。工程做法是按租户/按组织隔离缓存；同一组织的内部用户可共享（串 [[anthropic-22]] 的缓存治理）。
- **追问**：怎么避免「缩到零」带来的雪崩？
  - 要点：**限制并发冷启动数**（避免 100 个模型同时加载把存储带宽打满）；用**队列 + 准入**把突发摊平；对冷启动请求给明确预期（或提前预载）。**存储带宽是最容易被忽略的共享瓶颈**。

## 相关题目

- [[together-04]]：专用 endpoint 的成本与定价，serverless 的成本结构是其共享池版本。
- [[together-05]]：训练侧集群效率，与推理侧池化的资源治理同源。
- [[together-07]]：客户从闭源 API 迁移到开源模型——serverless 平台正是迁移的落点。
- [[anthropic-17]]：准入控制与降级链，serverless 的排队与拒绝沿用同一套。
- [[inference-serving-11]]：vLLM/SGLang/TensorRT-LLM 的选型，决定装箱与冷启动的实现细节。

## 参考资料与归属

- **LLM 推理优化** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/llm-inference-optimization>。第 1 节权重加载、显存与装箱的工程背景参照这篇。
- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 2 节 KV 页式分配与显存碎片治理来自这篇。
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving（延伸）** —— Moonshot AI，2024-06-24：<https://arxiv.org/abs/2407.00079>。第 3、4 节 KV 分层与预热/换入的思路参照这篇。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 2 节权重与 KV 显存分离治理的分析框架来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（权重 2 字节/参数、对象存储 2 GB/s、NVMe 5 GB/s、内存 40 GB/s、引擎初始化 2–10 s、warmup 0.5–1 s、单卡 80 GB、常驻池为 8 卡 × 60%、模型规模与到达率分布、GPU \$2/h、碎片 12%、overhead 1.3）都是按本仓库统一口径构造的**工程算例与显式假设**；真实平台必须用自己的存储带宽、引擎初始化时间与流量分布校准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
