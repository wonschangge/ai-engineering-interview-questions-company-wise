---
type: question
id: anthropic-24
company: Anthropic
topic: system-design
order: 24
question: 设计一个 10 亿文档、100 万 QPS 的分布式搜索系统。
question_en: Design a distributed search system for a billion documents at a million QPS.
asked_at: []
level: 高阶
tags: [系统设计, 分布式检索, 分片, 尾延迟, ANN, 缓存]
sources:
  - title: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs（HNSW）（延伸）
    url: https://arxiv.org/abs/1603.09320
    author: Malkov & Yashunin
    published: 2016-03-30
  - title: Billion-scale similarity search with GPUs（FAISS）（延伸）
    url: https://arxiv.org/abs/1702.08734
    author: Johnson, Douze, Jégou
    published: 2017-02-28
  - title: The Faiss library（延伸）
    url: https://arxiv.org/abs/2401.08281
    author: Douze et al.
    published: 2024-01-16
  - title: The Tail at Scale（延伸）
    url: https://research.google/pubs/the-tail-at-scale/
    author: Dean & Barroso (CACM 2013)
    published: 2013-02-01
related: [anthropic-22, anthropic-25, anthropic-17, rag-02, rag-10]
updated: 2026-09-28
---

## 一句话答案

> 这道题的答案不是「上向量数据库」，而是**把一个数量级问题拆成五层可独立伸缩的结构**：
> ① **分片（shard）**：按文档 id 哈希切成 $S$ 个分片，每个分片**自包含**（词法索引 + 向量索引 + 元数据），这样查询只需 fan-out 到全部（或目标）分片，无跨分片 join；
> ② **副本（replica）**：每个分片 $R$ 副本（例如 3），用于**读扩展 + 容错 + 就近路由**——100 万 QPS 靠的是「分片数 × 副本数」的总并行度；
> ③ **两阶段检索**：**词法（倒排）+ 向量（ANN）双路召回 → 融合 → 重排（cross-encoder）**。纯向量在精确匹配（错误码、专有名词）上弱，纯词法在语义改写上弱，双路是工程标准；
> ④ **分层缓存**：查询结果缓存、召回结果缓存、热分片副本、以及 embedding 缓存（同一 query 不重复算）；
> ⑤ **尾延迟治理**：fan-out 会让 p99 放大（$1-(1-p)^{N}$），所以必须用**对冲请求（hedged requests）+ 分片内提前终止 + 超时预算**，而不是「给所有分片加机器」。
> 容量上先把账算清（下表）：10 亿文档的原始文本、向量与索引各占多少、每节点能扛多少 QPS、需要多少节点——**算完账你会发现瓶颈通常不是算力，而是索引构建管道与尾延迟**。
> 一句话判据：**分片决定容量上限，副本决定 QPS 上限，缓存决定成本上限，尾延迟决定体验上限。**

## 面试官在考什么

- **是否先算容量账**：10 亿文档 × 平均大小 = 原始存储；× 向量维度 = 向量存储；× 索引开销 = 索引存储；再除以每节点容量 = 节点数。**没有这一步的回答都是空谈。**
- **fan-out 的尾延迟数学**：$N$ 个分片、单分片慢查询概率 $p$，则整查询至少命中一个慢分片的概率是 $1-(1-p)^{N}$——分片越多尾延迟越差。这是本题最重要的一个反直觉结论。
- **两路召回与融合**：为什么必须混合（词法 + 向量）、如何融合（RRF/加权）、重排放在哪（重排只对 top-k 做，才有算力预算）。
- **索引构建管道**：离线全量重建 + 在线增量（近实时）**双管道**；文档更新到可检索的延迟（freshness）与成本；删除怎么处理（墓碑 + 合并）。
- **ANN 的取舍**：HNSW 的图结构内存开销与召回-延迟曲线（可引用的原始工作：HNSW 以分层小世界图实现高效 ANN；FAISS 展示了十亿级相似搜索的 GPU 实现）。**它不是精确检索**，召回率要作为 SLA 指标。
- **分片策略的陷阱**：按哈希分片导致「必须查全部分片」；按范围/租户分片可以做**裁剪（pruning）**减少 fan-out（例如带过滤条件的查询只查相关分片）——但要处理热点。
- **成本与 SLO**：每查询成本、p99 预算如何分配到各阶段（召回 30 ms、重排 20 ms、融合 5 ms…）。

**常见错误答案**

- 「用向量数据库就能解决」——忽略混合检索、分片、副本、缓存与尾延迟。
- 分片后仍然要求「精确全局 top-k」——没意识到 ANN 本身就是近似的，全局精确排序代价极高。
- 不做尾延迟治理（只算平均延迟）。
- 忽略索引构建与更新的延迟与成本（一个 10 亿文档的索引重建要多久？）
- 单分片过大（重建慢、故障影响面大）或过小（fan-out 太大、元数据开销高）。

## 原理与推导

### 1. 容量账（先算再说）

| 项 | 算式 | 结果（10 亿文档、平均 2 KB 文本、768 维 fp16 向量） |
| --- | --- | --- |
| 原始文本 | $10^9\times2\text{ KB}$ | 2 TB（压缩后约 0.5 TB） |
| 向量 | $10^9\times768\times2\text{ B}$ | 1.5 TB |
| 向量索引（HNSW 图，约 1.5–2×） | 上行的 1.5–2 倍 | 2.3–3.0 TB |
| 倒排索引 | 约为原始文本的 30–50% | 0.6–1.0 TB |
| **单副本合计** | | **约 6–7 TB** |
| 3 副本 | ×3 | **约 18–21 TB** |

每节点可放 6 TB（含内存 + SSD 混合），则**单副本约 1–2 个节点不够、需要按分片切**：例如 256 分片、每分片 24 GB 索引 → 每分片一个节点，3 副本 = 768 个节点实例（可放在更少的物理机上，用多进程管理）。

### 2. QPS 账

设单节点可服务 2,000 QPS（混合检索，含重排）：

$$\text{总服务能力}=\frac{S\times R\times q_{\text{node}}}{1}\quad(\text{每次查询 fan-out 到全部 }S\text{ 个分片})$$

注意 QPS 的两套口径：**入口 QPS**（100 万）与**分片 QPS**（$100\text{万}\times S$）。若 $S=256$：

$$\text{分片 QPS}=10^6\times256=2.56\times10^8\ \Rightarrow\ \text{节点数}=\frac{2.56\times10^8}{2{,}000}\approx128{,}000\ \text{个分片实例}$$

这个数字说明：**盲目 fan-out 到全部分片是不可行的**，必须靠 ① 缓存（把绝大多数查询挡在召回之前）、② 分片裁剪（只查相关分片）、③ 副本并行（QPS 摊到副本）。这是本题最关键的量化洞察之一。

### 3. 尾延迟：fan-out 的惩罚

若单分片查询 $P(\text{慢})=p$，则一次 fan-out 到 $N$ 个分片的查询**至少遇到一个慢分片**的概率：

$$P_{\text{slow,query}}=1-(1-p)^{N}$$

| 单分片 $p$ | $N=10$ | $N=50$ | $N=256$ |
| --- | --- | --- | --- |
| 1% | 9.6% | 39.5% | 92.4% |
| 0.1% | 1.0% | 4.9% | 22.6% |
| 0.01% | 0.1% | 0.5% | 2.5% |

**结论**：要把整查询的慢概率压到 1%，当 $N=256$ 时单分片的慢概率必须 ≤ **0.004%**——这是「分片越多、尾延迟越难」的量化表达。

**治理手段**：
1. **对冲请求（hedged requests）**：向第二个副本发出重复请求，谁先返回用谁（代价：额外负载，通常只在 p95 之后触发）；
2. **分片内提前终止**：召回只要 top-k，不必算完（早停）；
3. **减少 fan-out**：分片裁剪（按过滤条件/租户/时间范围）、两级路由（先粗筛分片）；
4. **超时与降级**：超时后用部分结果返回（明确标注「结果可能不完整」）。

### 4. 两阶段检索与融合

```
query → embedding（缓存） ─┐
query → 词法解析 ─────────┴→ ① 召回：倒排 top-1000 ‖ 向量 top-1000（分片内并行）
                                     ↓ ② 融合：RRF / 加权（去重、跨分片合并）
                                     ↓ ③ 重排：cross-encoder 对 top-50
                                     ↓ ④ 截断返回 top-10 + 引用 + 分数
```

- **RRF（reciprocal rank fusion）**：$\text{score}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$，不需要调权重就能融合多路，工程上稳健；
- **重排的算力预算**：cross-encoder 的复杂度与「候选数 × 查询长度」成正比，所以只对 top-50 做（串 [[rag-10]]）；
- **分片合并**：每个分片返回自己的 top-k，协调节点做**全局归并**（$k$ 路归并 $O(Sk\log S)$），这也是 fan-out 不可避免的成本。

### 5. 索引管道：离线 + 增量

| 管道 | 触发 | 延迟 | 成本 |
| --- | --- | --- | --- |
| 全量重建 | 定期（模型升级、结构变更） | 小时级 | 高（需双倍存储做蓝绿切换） |
| 增量（近实时） | 文档变更事件 | 秒–分钟级 | 中（小段合并） |
| 删除 | 墓碑标记 + 后台合并 | 立即不可见（逻辑删除） | 低 |

**关键工程点**：全量重建必须**蓝绿切换**（新索引构建完成后原子切换别名），否则重建期间服务不可用；增量段要定期**合并（compaction）**，否则小段过多会拖慢查询。

### 6. 缓存分层

| 层 | 命中率量级 | 作用 |
| --- | --- | --- |
| 查询结果缓存（完全相同的 query） | 头部查询可达 20–40% | 直接省掉整个召回 |
| embedding 缓存 | 与 query 缓存重叠 | 省 embedding 算力 |
| 召回结果缓存（分片级） | 中 | 减少分片重复计算 |
| 热分片副本 | — | 抗热点租户 |

**注意**：缓存必须带**索引版本号**（索引进新文档后旧结果要失效），否则用户会「搜不到刚上传的文档」。

## 数值与代码验证

### 表 1：尾延迟放大（$P=1-(1-p)^N$）

见上表；结论是本文件的核心反直觉点——**分片数增加 25 倍（10→256），整查询慢概率在 $p=0.1\%$ 时从 1.0% 涨到 22.6%**。

### 表 2：100 万 QPS 的可行路径（256 分片、每实例 2,000 QPS）

| 策略 | 有效入口 QPS | 有效 fan-out | 分片 QPS | 节点实例数 |
| --- | --- | --- | --- | --- |
| 全分片 fan-out，无缓存 | 1,000,000 | 256 | $2.56\times10^{8}$ | **128,000** |
| + 结果缓存命中 30% | 700,000 | 256 | $1.79\times10^{8}$ | 89,600 |
| + 分片裁剪（平均只查 1/8） | 700,000 | 32 | $2.24\times10^{7}$ | **11,200** |

**三个必须说清的结论**：
1. **减少实例数靠的是缓存与裁剪，不是加副本**——副本只是把同样的 QPS 摊到更多实例上（3 副本 = 3 倍实例、每个扛 1/3），买的是**容错与就近路由**，不是容量；
2. 从 12.8 万降到 1.12 万实例（**一个数量级**）完全由「缓存 + 裁剪」贡献；
3. 若还想继续降，手段是**向量量化**（PQ/int8 把 1.5 TB 向量压到几百 GB → 减少存储节点数与内存压力，同时降低单查询延迟）。

### 可运行代码### 可运行代码

```python
# 分布式检索的容量与尾延迟模型
import math

def capacity(n_docs=1e9, doc_kb=2, dim=768, bytes_per=2, idx_mult=1.75,
             inv_ratio=0.4, replicas=3, node_capacity_tb=6.0):
    raw = n_docs * doc_kb * 1024
    vec = n_docs * dim * bytes_per
    ann = vec * idx_mult
    inv = raw * inv_ratio
    total = raw + vec + ann + inv
    per_copy_nodes = total / (node_capacity_tb * 1024**4)
    return dict(raw_tb=raw/1024**4, vec_tb=vec/1024**4, ann_tb=ann/1024**4,
                inv_tb=inv/1024**4, total_tb=total/1024**4,
                with_replicas_tb=total*replicas/1024**4,
                single_copy_nodes=per_copy_nodes, total_nodes=per_copy_nodes*replicas)

c = capacity()
print("① 容量账（10 亿文档、2 KB 文本、768 维 fp16、3 副本）")
for k, name in (("raw_tb", "原始文本"), ("vec_tb", "向量"), ("ann_tb", "ANN 索引"),
                ("inv_tb", "倒排索引")):
    print(f"  {name:<8} {c[k]:>7.2f} TB")
print(f"  单副本合计 {c['total_tb']:.2f} TB；3 副本 {c['with_replicas_tb']:.2f} TB")
print(f"  按每节点 6 TB 计：单副本需 {c['single_copy_nodes']:.1f} 节点，"
      f"3 副本共 {c['total_nodes']:.1f} 个节点实例")

def qps_plan(entry_qps=1e6, shards=256, qps_per_node=2000,
             cache_hit=0.30, pruning_factor=8, replicas=3):
    """返回节点实例数。注意：副本不减少实例数（3 副本 = 3 倍实例、每个扛 1/3），
    它买的是容错与就近路由；nodes_with_replicas 只是把同一份 QPS 摊到更多实例上。"""
    effective = entry_qps * (1 - cache_hit)
    fanout = shards / pruning_factor
    shard_qps = effective * fanout
    nodes = shard_qps / qps_per_node                      # 总服务能力需求（实例数）
    return dict(effective=effective, fanout=fanout, shard_qps=shard_qps,
                nodes=nodes, nodes_with_replicas=nodes, replicas=replicas)

print("\n② QPS 路径（入口 100 万 QPS、256 分片、单节点 2,000 QPS）")
stages = [
    ("全分片 fan-out，无缓存", dict(cache_hit=0.0, pruning_factor=1)),
    ("+ 结果缓存命中 30%", dict(cache_hit=0.30, pruning_factor=1)),
    ("+ 分片裁剪（只查 1/8）", dict(cache_hit=0.30, pruning_factor=8)),
]
for name, kw in stages:
    r = qps_plan(**kw)
    print(f"  {name:<22} 有效入口 {r['effective']:>10,.0f} QPS  "
          f"fan-out {r['fanout']:>6.1f}  分片 QPS {r['shard_qps']:>14,.0f}  "
          f"实例 {r['nodes']:>9,.0f}")
print("  读法：从 12.8 万降到 1.1 万实例，全靠缓存与裁剪 —— 而不是加机器；")
print("        副本不减少实例数（3 副本 = 3 倍实例、每个扛 1/3），它买的是容错与就近路由")

def slow_query_prob(p, n):
    return 1 - (1 - p) ** n

print("\n③ 尾延迟：fan-out 的惩罚（整查询至少命中一个慢分片的概率）")
print(f"  {'单分片慢概率':>12} " + " ".join(f"{'N='+str(n):>9}" for n in (10, 50, 256)))
for p in (0.01, 0.001, 0.0001):
    print(f"  {p:>12.4%} " + " ".join(f"{slow_query_prob(p, n):>9.2%}" for n in (10, 50, 256)))
need_p = 1 - (1 - 0.01) ** (1 / 256)
print(f"  要把整查询慢概率压到 1%（N=256），单分片慢概率必须 ≤ {need_p:.6%}")
print("  读法：这就是为什么大 fan-out 系统必须做对冲请求与提前终止，而不是只加机器")

def hedged(p, n, hedge_after_ms, base_ms=10, slow_ms=200):
    """简化模型：p95 后对冲，返回期望延迟的改善（教学示意）"""
    normal = (1 - p) * base_ms + p * slow_ms
    hedged_lat = (1 - p) * base_ms + p * (hedge_after_ms + base_ms)   # 对冲后取先返回者
    return normal, hedged_lat
for p in (0.01, 0.05):
    n_h, h = hedged(p, 256, hedge_after_ms=30)
    print(f"  p={p:.0%}：不对冲期望 {n_h:.1f} ms -> 对冲后 {h:.1f} ms")

print("\n④ 缓存必须带索引版本（否则新文档搜不到）")
def cache_key(query, index_version):
    return f"{query}#v{index_version}"
print("  同一 query 在索引 v1 与 v2 下是不同的缓存键：",
      cache_key("kv cache", 1), "/", cache_key("kv cache", 2))
print("  读法：索引版本参与缓存键 → 新索引上线后旧结果自动失效（避免「刚上传搜不到」）")
```

预期输出要点（实跑）：① 容量账给出单副本约 6.6 TB、3 副本约 20 TB，按每节点 6 TB 需要约 3.2 个节点实例——**说明单副本用少数节点就能装下，真正的规模压力在 QPS 与尾延迟，而不是存储**；② QPS 路径从「全 fan-out 无缓存」的约 **12.8 万**实例降到「缓存 + 裁剪」的约 **1.12 万**实例（一个数量级），并说明**副本不减少实例数**；③ 尾延迟表显示 $N=256$、$p=0.1\%$ 时整查询慢概率 **22.6%**，要压到 1% 则单分片慢概率必须 ≤ **0.0039%**；④ 对冲请求把高 $p$ 场景的期望延迟显著拉低（这正是 *The Tail at Scale* 里的核心手段）。

## 常见追问

- **追问**：为什么不按租户分片？
  - 要点：按租户可以裁剪 fan-out（只查该租户分片）并隔离噪声，但会遇到**热点租户**（大客户分片过载）；混合做法是「大租户独立分片 + 小租户哈希合并」，并按大小动态再平衡。
- **追问**：ANN 的召回率怎么保证？
  - 要点：把召回率做成**可测指标**（用精确检索在小样本上对拍，测 recall@k）；HNSW 的 `efSearch` 是召回-延迟旋钮，按 SLO 调；同时用词法路兜住精确匹配，降低对 ANN 召回的绝对依赖。
- **追问**：如何做到「刚上传的文档立刻能搜到」？
  - 要点：写路径与索引路径解耦（先落库 + 发事件），增量索引管道把文档写入小段（秒级可见），查询同时查「主索引 + 实时小段」；缓存要带索引版本（见代码第 ④ 段）。
- **追问**：全量重建怎么做才不停机？
  - 要点：**蓝绿**：新索引在独立资源上构建 → 校验（文档数、抽样召回对拍）→ 原子切换别名 → 保留旧索引一段时间以便回滚；重建期间增量事件要**双写**到新旧两边。
- **追问**：100 万 QPS 的网关层会不会成为瓶颈？
  - 要点：会——需要无状态网关 + 一致性哈希路由 + 连接复用；把「查询解析/embedding」这类可缓存计算前置；限流与优先级（串 [[anthropic-25]] 的配额设计）。
- **追问**：怎么控制成本？
  - 要点：① 提高结果缓存命中（头部查询集中）；② 分片裁剪；③ 向量量化（PQ/int8 把 1.5 TB 压到几百 GB）；④ 冷数据下沉到对象存储 + 按需加载；⑤ 重排只对 top-k 做。

## 相关题目

- [[anthropic-22]]：聊天服务的检索层，是本题在「单租户、低 QPS」场景的版本。
- [[anthropic-25]]：开发者 API 设计，负责把这里的检索能力以稳定接口暴露出去。
- [[anthropic-17]]：服务栈与准入控制，对应本题网关与限流部分。
- [[rag-02]]：检索质量与分块策略，决定召回阶段的效果上限。
- [[rag-10]]：重排与融合策略，对应本题的第二阶段。

## 参考资料与归属

- **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs（HNSW）（延伸）** —— Malkov & Yashunin，2016-03-30：<https://arxiv.org/abs/1603.09320>。第 4 节与常见追问里 ANN 的「图结构 + 召回-延迟可调」性质来自这篇。
- **Billion-scale similarity search with GPUs（FAISS）（延伸）** —— Johnson, Douze, Jégou，2017-02-28：<https://arxiv.org/abs/1702.08734>。第 1 节「十亿级向量检索的工程可行性」的背景来自这篇。
- **The Faiss library（延伸）** —— Douze et al.，2024-01-16：<https://arxiv.org/abs/2401.08281>。第 4 节里索引类型与量化（IVF/PQ 等）的取舍背景来自这篇。
- **The Tail at Scale（延伸）** —— Dean & Barroso (CACM 2013)，2013-02-01：<https://research.google/pubs/the-tail-at-scale/>。第 3 节的尾延迟放大与对冲请求（hedged requests）、提前终止等治理手段来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（10 亿文档、2 KB 文本、768 维 fp16、索引倍数 1.75、倒排占比 0.4、3 副本、每节点 6 TB / 2,000 QPS、缓存命中 30%、裁剪 1/8、对冲阈值 30 ms）都是按本仓库统一口径构造的工程算例与显式假设；真实系统必须用压测与召回对拍校准，尤其「单节点 2,000 QPS」高度依赖硬件与索引类型。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
