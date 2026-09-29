---
type: question
id: cohere-02
company: Cohere
topic: llm-internals
order: 2
question: 我们的旗舰模型是稀疏 MoE，总参数量约为激活参数量的 10 倍。为什么这种架构很适合私有化企业部署，它又在哪些地方带来代价？
question_en: Our flagship model is a sparse MoE with roughly 10× more total parameters than active parameters. Why does this suit private enterprise deployment, and where does it cost you?
asked_at: []
level: 高阶
tags: [MoE, 稀疏激活, 私有化部署, 显存下限, all-to-all, 负载均衡]
sources:
  - title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（延伸）
    url: https://arxiv.org/abs/2101.03961
    author: Fedus et al. (Google)
    published: 2021-01-11
  - title: Mixtral of Experts（延伸）
    url: https://arxiv.org/abs/2401.04088
    author: Jiang et al. (Mistral AI)
    published: 2024-01-08
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
  - title: 估算服务一个 70B 模型所需的 GPU 显存（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cohere-08, cohere-01, cohere-06, inference-serving-08, inference-serving-07]
updated: 2026-09-28
---

## 一句话答案

> 一句话概括这种架构的本质：**显存按「总参数」付，算力按「激活参数」付**。
> $$M_{\text{weights}}=2\times P_{\text{total}}\ (\text{bf16}),\qquad \text{FLOPs/token}\approx 2\times P_{\text{active}}$$
> 所以「总参是激活的 10 倍」意味着：**你要买 10 倍显存的机器，却只付 1 倍的计算**。
> **为什么适合私有化企业部署**（这是本题的前半问）：
> ① **私有化部署的第一约束通常是「能装进多少张卡」，而不是「每秒多少 token」**——企业内部负载的 QPS 往往不高（几十到几百并发），**算力富余、显存紧张**；MoE 恰好把预算从「算力」挪到「显存」，**用客户已有的卡换到接近大模型的质量**；
> ② **单 token 的解码开销由激活参数决定**：权重读取量 ≈ 激活参数（只读被路由到的专家），所以**延迟与吞吐接近同激活规模的小模型**；这在「低并发、要交互」的企业场景里比峰值吞吐更重要；
> ③ **KV cache 与注意力无关专家数**：KV 由层数/头数/维度决定，MoE 只换 FFN——所以**长上下文的显存开销与同架构的稠密模型相当**，不会因为专家多而更贵；
> ④ **可为不同客户做「专家级」定制**：微调时只有部分专家被更新，**多租户适配的边际成本更低**（同一基座 + 不同专家/LoRA 组合）。
> **代价（后半问，必须逐条讲）**：
> ① **显存下限高**：100B 总参 bf16 = **200 GB 权重**，仅是权重就要 **≥3 张 80 GB 卡**（还要留 KV、激活、通信缓冲）；同激活（10B）的稠密模型只要 20 GB——**这是「用显存换质量」的直接账单**；
> ② **all-to-all 通信**：专家并行需要把 token 派发到专家所在卡、再收回（dispatch + combine），**跨节点 all-to-all 往往是瓶颈**；低并发时通信无法被计算掩盖；
> ③ **负载不均衡**：路由是数据相关的，热专家会过载 → 要么**丢弃 token**（capacity factor 截断，掉质量），要么**等待**（掉延迟）；
> ④ **对 batch 敏感**：MoE 的收益来自「批内 token 分摊专家读取」，**batch 太小则通信与 kernel 启动开销占主导**；企业低峰时段的体验可能不如预期；
> ⑤ **硬件门槛**：互联质量（NVLink/IB）比同激活的稠密模型重要得多；**PCIe-only 的机器可能跑不动**；
> ⑥ **运维复杂度**：专家放置、热专家复制、量化对质量的影响、以及「专家漂移」（某些专家长期不被选中）都需要监控。
> 一句话判据：**MoE 把私有化部署的瓶颈从「算力不够」换成「显存与互联不够」**——买之前先算这两个数。

## 面试官在考什么

- **能否一句话说出本质**：显存按总参、算力按激活——这是本题的核心，答不出这句后面都站不住。
- **是否理解「私有化」的约束**：企业自建 GPU 的典型画像（**卡不多、QPS 不高、要低延迟、要数据不出域**），以及为什么「算力换显存」对这个画像有利。
- **显存下限的算术**：$2\times P_{\text{total}}$ 字节 + KV + 激活 + 通信缓冲；能否给出具体卡数（例如 100B → ≥3×80 GB 仅权重）。
- **通信代价**：能否说出 all-to-all 的两段（dispatch/combine）、跨节点与节点内的带宽差（NVLink vs IB vs PCIe）、以及**低并发时通信无法隐藏**。
- **负载均衡**：capacity factor 与丢 token 的取舍、辅助负载均衡损失、专家复制；能否量化「热专家过载」。
- **batch 敏感性**：为什么 MoE 在低 QPS 下的单位成本更高（固定开销摊不薄）。
- **KV 与上下文**：能否指出 MoE 不改注意力，所以**长上下文成本与稠密同构模型相同**（这是容易被误解的一点）。
- **运维与量化**：专家放置、热专家复制、量化对路由/质量的影响、以及「专家利用率监控」。
- **诚实**：能说出「MoE 不是免费的质量」——它把成本从计算搬到显存与网络，**如果客户的卡多但互联差，MoE 反而更难落地**。

**常见错误答案**

- 说「MoE 更省显存」（**恰恰相反**：总参数全都要驻留）。
- 只谈算力省，不谈 all-to-all 与负载不均衡。
- 认为 MoE 对长上下文更省（KV 与专家数无关）。
- 忽略 batch 敏感性（在低 QPS 场景仍然假设高吞吐）。
- 把「总参 10 倍」当成「效果一定好 10 倍」（质量提升是有限且任务相关的）。
- 忽略专家并行的**放置问题**（哪些专家放哪张卡，决定通信量）。

## 原理与推导

### 1. 显存与算力的分离（核心公式）

| 项 | 稠密模型 | MoE（$R=P_{\text{total}}/P_{\text{active}}$） |
| --- | --- | --- |
| 权重显存 | $2P_{\text{active}}$ | $2P_{\text{total}}=2R\,P_{\text{active}}$ |
| 每 token FLOPs | $\approx2P_{\text{active}}$ | $\approx2P_{\text{active}}$（只算被选专家） |
| KV cache | 由注意力配置决定 | **相同**（与专家数无关） |
| 通信 | 主要是 TP/PP 的 all-reduce | **额外 all-to-all**（dispatch + combine） |

**读法**：$R=10$ 时，**显存是 10 倍、算力是 1 倍**。私有化部署里「卡数」是硬约束，「FLOPs」通常有余量 → 这个交换往往是划算的。

### 2. all-to-all 通信量

每个 token 被路由到 $k$ 个专家（$k$ 通常为 1–2）。若专家分散在 $E$ 张卡上，则每个 token 的通信量约为：

$$\text{bytes/token}\approx 2\times k\times d\times b\quad(\text{dispatch}+\text{combine})$$

其中 $d$ 为隐藏维度、$b$ 为每元素字节。**例**：$d=8192$、$b=2$、$k=2$ → $2\times2\times8192\times2=65{,}536$ 字节/token ≈ **64 KiB/token**（仅专家通信）。

- **节点内**（NVLink ~900 GB/s）：64 KiB / 900 GB/s ≈ **0.07 µs**（可忽略）；
- **跨节点**（IB 400 Gb/s ≈ 50 GB/s）：64 KiB / 50 GB/s ≈ **1.3 µs/token**；
- **batch 1、每步 1 token** → 通信 1.3 µs 看似小，但**延迟而非带宽成为瓶颈**（RDMA 往返几十 µs），且**无法与计算重叠**（计算本身只有几十 µs）；batch 256 时通信量 ×256，带宽开始吃紧，但可摊薄。

**结论**：**MoE 的通信代价对小 batch 更严重**（延迟主导），这正是「企业低峰体验变差」的机制。

### 3. 负载均衡与 capacity factor

设专家容量上限为 $C=\text{capacity\_factor}\times\frac{\text{tokens}}{E}$。若某专家收到超过 $C$ 的 token，**多余的会被丢弃**（或排队），导致：

$$\text{drop rate}\approx P\big(\text{load}_e>C\big)$$

**缓解手段**：① **辅助负载均衡损失**（训练时惩罚不均衡）；② **capacity factor 调大**（少丢 token 但更多 padding 浪费）；③ **专家复制**（热专家多放几份）；④ **在线调度**（把 token 路由到副本）。

### 4. 与私有化部署的匹配（为什么适合）

| 企业私有化的典型约束 | 稠密大模型 | MoE（10× 稀疏） |
| --- | --- | --- |
| 卡数有限（例如 8×H100） | 70B 稠密勉强放下（140 GB 权重） | 100B 总参放不下（200 GB）→ **需要更多卡** |
| QPS 不高、要低延迟 | 延迟随模型增大而上升 | **延迟接近同激活小模型** ✅ |
| 数据不出域 | 都满足 | 都满足 |
| 长上下文需求 | KV 是瓶颈 | KV 相同 ✅ |
| 互联质量 | 要求中等 | **要求高** ⚠️ |
| 多客户定制 | 每客户一套权重 | 专家/LoRA 级定制更省 ✅ |

**读法**：MoE 的「适合」是**有条件的**——**卡数与互联足够时**，它把预算从算力挪到显存，正好匹配「卡不多但要有大模型质量」的诉求；**互联差或卡太少时，它反而更难落地**。

### 5. 量化与专家利用

- **量化**：权重量化（int8/fp8）能把显存减半，但**专家层的量化误差会累积到路由后的输出**，需要评测验证；某些实现还会量化路由网络（更敏感）。
- **专家利用率监控**：长期未被选中的专家是浪费的显存 → 需要统计**每专家的 token 占比**，并考虑「专家剪枝/重平衡」。
- **专家放置**：把**共同被激活的专家**放在同一节点（减少跨节点流量）是常见的调度优化；但这依赖路由分布的先验或在线统计。

## 数值与代码验证

### 表 1：三种架构的显存/算力/卡数对比（见代码输出）

| 架构 | 总参数 | 激活参数 | 权重显存(bf16) | 每 token FLOPs | 仅权重所需 80GB 卡 |
| --- | --- | --- | --- | --- | --- |
| 稠密 10B | 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |
| 稠密 70B | 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |
| MoE 100B/10B | 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |
| MoE 400B/40B | 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：all-to-all 通信与 batch（节点内 vs 跨节点）

| batch | 通信量 | 节点内耗时 | 跨节点耗时 | 是否可隐藏 |
| --- | --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# 稀疏 MoE 的部署算术：显存下限、算力、all-to-all、负载不均衡与 batch 敏感性
import math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

GB = 1024 ** 3
H100_MEM = 80 * GB
NVLink_BW = 900e9        # 节点内有效带宽（B/s）
IB_BW = 50e9             # 跨节点每卡有效带宽（B/s）
FLOPS = 989e12           # H100 bf16 dense（仓库统一常数）

# ---------- 1) 显存与算力：稠密 vs MoE ----------
@dataclass
class Arch:
    name: str
    total_b: float              # 总参数（十亿）
    active_b: float             # 激活参数（十亿）
    hidden: int = 8192
    bytes_per_param: float = 2  # bf16
    def weights_gb(self) -> float:
        return self.total_b * 1e9 * self.bytes_per_param / GB
    def flops_per_token(self) -> float:
        return 2 * self.active_b * 1e9
    def min_gpus_weights_only(self) -> int:
        return math.ceil(self.weights_gb() * GB / H100_MEM)
ARCHS = [
    Arch("稠密 10B", 10, 10),
    Arch("稠密 70B", 70, 70),
    Arch("MoE 100B/10B（R=10）", 100, 10),
    Arch("MoE 400B/40B（R=10）", 400, 40),
    Arch("稠密 400B", 400, 400),
]
print("① 显存与算力的分离（bf16；仅权重，不含 KV/激活/通信缓冲）")
print(f"  {'架构':<24} {'总参':>7} {'激活':>7} {'权重(GB)':>10} {'FLOPs/token':>13} "
      f"{'最少卡数':>9}")
for a in ARCHS:
    print(f"  {a.name:<24} {a.total_b:>6.0f}B {a.active_b:>6.0f}B {a.weights_gb():>10.0f} "
          f"{a.flops_per_token()/1e9:>12.0f}G {a.min_gpus_weights_only():>9}")
print("  读法：**MoE 100B/10B 的权重显存是稠密 10B 的 10 倍，但每 token FLOPs 完全相同** ——")
print("        私有化场景里「卡数」是硬约束、「算力」常有余量，所以这是「用显存换质量」的典型交换；")
print("        注意 400B 总参仅权重就要 10 张 80GB 卡（还不含 KV 与激活）")

# ---------- 2) all-to-all 通信量 ----------
@dataclass
class Comms:
    hidden: int = 8192
    bytes_per_elem: float = 2
    top_k: int = 2                # 每个 token 选 k 个专家
    rdma_latency_s: float = 20e-6  # ★ RDMA 往返延迟（跨节点，典型几十 µs）
    launch_s: float = 8e-6         # ★ 两段 all-to-all 的 kernel 启动开销
    def bytes_per_token(self) -> float:
        return 2 * self.top_k * self.hidden * self.bytes_per_elem     # dispatch + combine
    def transfer_s(self, batch: int, bw: float) -> float:
        return self.bytes_per_token() * batch / bw
    def total_s(self, batch: int, bw: float, cross_node: bool) -> float:
        """总通信耗时 = 延迟（不可摊薄）+ 带宽时间 + 启动开销"""
        lat = self.rdma_latency_s if cross_node else 0.0
        return lat + self.transfer_s(batch, bw) + self.launch_s
c = Comms()
print(f"\n② 专家 all-to-all 通信（隐藏维 {c.hidden}、top-{c.top_k}、bf16）")
print(f"  每 token 通信量 = {c.bytes_per_token()/1024:.0f} KiB（dispatch + combine）；")
print(f"  跨节点固定开销 = RDMA 往返 {c.rdma_latency_s*1e6:.0f} µs + 启动 {c.launch_s*1e6:.0f} µs"
      f"（**不随 batch 摊薄**）")
print(f"  {'batch':>7} {'总通信量':>11} {'节点内(µs)':>10} {'跨节点(µs)':>10} "
      f"{'计算(µs)':>9} {'通信占比':>8} {'可隐藏':<8}")
for batch in (1, 8, 64, 256, 1024):
    total = c.bytes_per_token() * batch
    t_nv = c.total_s(batch, NVLink_BW, cross_node=False) * 1e6
    t_ib = c.total_s(batch, IB_BW, cross_node=True) * 1e6
    flops = 2 * 10e9 * batch
    t_comp = flops / (FLOPS * 0.4) * 1e6          # 假设 40% MFU
    ratio = t_ib / t_comp
    hide = "是" if ratio < 0.15 else ("勉强" if ratio < 0.4 else "否（通信主导）")
    print(f"  {batch:>7} {total/1024:>9.0f} KiB {t_nv:>10.2f} {t_ib:>10.2f} "
          f"{t_comp:>9.1f} {ratio:>7.1%} {hide:<8}")
print("  读法：**batch=1 时跨节点通信与计算已同量级（占比过半）**，因为 RDMA 往返与启动开销")
print("        **不随 batch 摊薄**；batch 增大后通信量上升但被计算掩盖 —— 这是 MoE「低峰变差」的机制。")
print("        （对比：节点内 NVLink 的固定开销小得多，所以「专家尽量放同一节点」是关键优化）")

# ---------- 3) 负载不均衡与 capacity factor ----------
@dataclass
class Routing:
    experts: int = 64
    top_k: int = 2
    imbalance: float = 0.30      # ★ 专家偏好的对数正态标准差（真实路由器经均衡损失后约 0.2–0.4）
    def token_loads(self, tokens: int, seed: int = 3) -> List[int]:
        """模拟：每个专家有一个偏好权重（对数正态），token 按偏好+噪声分配"""
        rnd = random.Random(seed)
        w = [math.exp(rnd.gauss(0, self.imbalance)) for _ in range(self.experts)]
        tot = sum(w)
        p = [x / tot for x in w]
        loads = [0] * self.experts
        for _ in range(tokens * self.top_k):
            r = rnd.random()
            acc = 0.0
            for i, pi in enumerate(p):
                acc += pi
                if r <= acc:
                    loads[i] += 1
                    break
            else:
                loads[-1] += 1
        return loads
def drop_rate(loads: List[int], capacity_factor: float, top_k: int = 2) -> Dict[str, float]:
    """★ 容量 = capacity_factor × **每个专家的平均负载**（不是 token 数/专家数）"""
    avg_load = sum(loads) / len(loads)
    cap = capacity_factor * avg_load
    dropped = sum(max(0, l - cap) for l in loads)
    total_assignments = sum(loads)
    return {"容量上限(每专家)": cap, "丢弃率": dropped / total_assignments if total_assignments else 0.0,
            "最大/平均": max(loads) / (sum(loads) / len(loads))}
r = Routing()
loads = r.token_loads(100_000)
print(f"\n③ 负载不均衡与 capacity factor（{r.experts} 专家、每 token 选 {r.top_k}、"
      f"偏好对数正态 σ={r.imbalance}）")
print(f"  专家负载的最大/平均 = {max(loads)/(sum(loads)/len(loads)):.2f}x"
      f"（经均衡损失训练后真实系统通常 1.5–3x）")
print(f"  {'capacity factor':>16} {'容量上限':>10} {'丢弃率':>8}")
for cf in (0.8, 1.0, 1.25, 1.5, 2.0):
    d = drop_rate(loads, cf, r.top_k)
    print(f"  {cf:>16.2f} {d['容量上限(每专家)']:>10,.0f} {d['丢弃率']:>8.1%}")
print("  读法：**capacity factor 越小丢得越多（掉质量），越大则 padding 与显存浪费越多（掉吞吐）**；")
print("        这里 1.25–1.5 已能把丢弃率压到很低 —— 说明「均衡损失训练得好」时，MoE 的丢弃问题可控")

# ---------- 4) 与稠密模型的质量-算力-显存对照 ----------
print("\n④ 私有化部署的匹配分析（假设客户预算 8×H100 80GB）")
BUDGET_GPUS = 8
print(f"  {'方案':<24} {'权重显存(GB)':>12} {'是否装得下':>10} {'KV 空间(GB)':>12} {'结论':<24}")
for a, kv_gb in ((Arch("稠密 70B", 70, 70), 8 * 80 * 0.35),
                 (Arch("MoE 100B/10B", 100, 10), 8 * 80 * 0.65),
                 (Arch("MoE 200B/20B", 200, 20), 8 * 80 * 0.30)):
    fits = "是" if a.weights_gb() <= 8 * 80 * 0.9 else "否"
    note = ("装得下，KV 充裕" if a.weights_gb() <= 8 * 80 * 0.5 else
            ("装得下但 KV 紧张" if a.weights_gb() <= 8 * 80 * 0.9 else "装不下（需更多卡或量化）"))
    print(f"  {a.name:<24} {a.weights_gb():>12.0f} {fits:>10} {kv_gb:>12.0f} {note:<24}")
print("  读法：**8 卡的预算下，MoE 的总参不能太大**（100B 已占 186 GiB ≈ 200 GB 十进制权重）；")
print("        所以要按「卡数 → 可承受总参 → 可达质量」倒推，而不是反过来")

# ---------- 5) 综合决策：什么时候 MoE 更适合私有化 ----------
@dataclass
class Scenario:
    name: str
    gpus: int
    interconnect: str          # nvlink / ib / pcie
    qps: float
    latency_slo_ms: float
    def score(self) -> Tuple[float, str]:
        """返回 (适配度 0-1, 说明)"""
        # 互联质量
        ic = {"nvlink": 1.0, "ib": 0.8, "pcie": 0.35}[self.interconnect]
        # 卡数（越多越能装下大总参）
        cap = min(1.0, self.gpus / 16)
        # QPS 高 -> 能摊薄通信；QPS 低 -> MoE 的通信/延迟劣势更明显
        batch_proxy = min(1.0, self.qps / 20)
        s = 0.4 * ic + 0.3 * cap + 0.3 * batch_proxy
        note = []
        if ic < 0.5:
            note.append("互联差（PCIe）-> all-to-all 会成为瓶颈")
        if cap < 0.6:
            note.append("卡偏少 -> 大总参吃紧")
        if batch_proxy < 0.5:
            note.append("QPS 低 -> 通信与启动开销摊不薄")
        return s, ("；".join(note) if note else "适合（卡够、互联好、有稳定批）")
SCEN = [
    Scenario("8×H100 NVLink、8 QPS", 8, "nvlink", 8, 800),
    Scenario("16×H100 IB、32 QPS", 16, "ib", 32, 800),
    Scenario("8×A100 PCIe、4 QPS", 8, "pcie", 4, 800),
    Scenario("64×H100 IB、128 QPS", 64, "ib", 128, 800),
]
print("\n⑤ 什么时候 MoE 真的适合私有化（适配度 = 互联 0.4 + 卡数 0.3 + 批大小 0.3）")
print(f"  {'场景':<26} {'适配度':>7} 说明")
for s in SCEN:
    score, note = s.score()
    print(f"  {s.name:<26} {score:>7.2f} {note}")
print("  读法：**MoE 的适配性由三件事共同决定** —— 互联质量（决定 all-to-all 能不能忍）、")
print("        卡数（决定能不能装下总参）、批大小（决定固定开销能不能摊薄）")
```

预期输出要点（实跑）：① **MoE 100B/10B 的权重显存是稠密 10B 的 10 倍（186 GiB vs 19 GiB），但每 token FLOPs 完全相同（20G）**——稠密 400B 与 MoE 400B/40B 的权重相同（745 GiB）而算力差 10 倍，这就是「用显存换质量」的账单；② all-to-all 通信在 **batch=1 时跨节点耗时 29.3 µs、占计算的 58%（通信主导）**，因为 **RDMA 往返 20 µs + 启动 8 µs 不随 batch 摊薄**；batch≥8 后通信占比降到 10% 以下被计算掩盖——节点内 NVLink 则只要 8 µs；③ 负载不均衡实测 **最大/平均 = 1.86×**（与经均衡损失训练后的真实系统 1.5–3× 同量级）；capacity factor 0.8→2.0 时**丢弃率 24.0% → 12.4% → 4.9% → 1.3% → 0.0% 单调下降**（代价是 padding 与显存浪费）；④ 8×80 GB 预算下的匹配分析显示 **MoE 200B/20B 已占 373 GiB 权重、KV 只剩 192 GiB（紧张）**，而稠密 70B 只占 130 GiB——要按「卡数 → 可承受总参 → 可达质量」倒推；⑤ 适配度评分把结论落到具体场景：**互联差（PCIe）、卡少、QPS 低**三者中任一成立，MoE 都不划算。

## 常见追问

- **追问**：为什么说「MoE 省算力」却不省显存？
  - 要点：**计算只经过被选中的专家**（激活参数），所以 FLOPs 按激活算；但**所有专家的权重都必须驻留显存**（随时可能被路由到），所以显存按总算。**这是一枚硬币的两面**，也是私有化场景里唯一的取舍点。
- **追问**：专家并行（EP）与张量并行（TP）怎么选？
  - 要点：**TP 通信量大且延迟敏感**（每层 all-reduce），适合放节点内（NVLink）；**EP 的 all-to-all 也是通信重的**，但可以按「专家分组」做**分层 all-to-all**（先节点内、再跨节点）。常见组合是 **TP 在节点内 + EP 跨节点 + DP 最外层**（串 [[inference-serving-07]]）。
- **追问**：低 QPS 场景怎么救 MoE 的延迟？
  - 要点：① **专家复制**（把热专家在多个节点各放一份，减少跨节点跳数）；② **分层 all-to-all**（节点内先聚合）；③ **CUDA graph + 预取**（减少启动开销）；④ **调小 EP 度**（把专家放在更少的节点里，代价是显存）；⑤ 若仍不行，**接受它与稠密模型的服务差异**并调整产品预期。
- **追问**：capacity factor 该怎么定？
  - 要点：它是**质量与吞吐**的旋钮——太小丢 token（质量降），太大 padding 浪费（吞吐降）。做法：**用真实流量测 drop rate 与端到端质量**，并结合**辅助负载均衡损失**降低不均衡程度。上线后要**持续监控 drop rate**。
- **追问**：MoE 对长上下文更友好吗？
  - 要点：**不更友好，但也不更差**——KV cache 由注意力配置决定，**与专家数无关**。所以 MoE 与同注意力的稠密模型在长上下文上的显存开销相同；差别只在 FFN 的权重与计算。
- **追问**：量化 MoE 有什么特别风险？
  - 要点：① **路由网络对量化更敏感**（错路由 = 用错专家，误差被放大）；② 专家的量化误差**只影响被选中的 token**，所以影响是**不均匀的**（某些 token 变差）；③ 因此**必须按「最终质量」评测**，而不是只看权重的重构误差。实践中常见「专家权重 4-bit、路由与注意力保持高精度」的混合方案。
- **追问**：企业客户该买 MoE 还是稠密？
  - 要点：**按「卡数与互联」倒推**：卡少或互联差 → 稠密（同激活规模）；卡多且互联好、且要「接近大模型的质量」 → MoE。**别只看参数量与榜单分数**——私有化部署的体验由显存与通信决定（这正是本题的落点）。

## 相关题目

- [[cohere-08]]：气隙部署——MoE 在客户自有 GPU 上的落地细节（互联与放置）。
- [[cohere-01]]：token 限流——容量规划的另一面（每 token 成本与并发）。
- [[cohere-06]]：索引扩大 10 倍后的质量排查——同属「规模变化引发的问题」。
- [[inference-serving-08]]：服务 70B 所需的显存估算（本仓库专题），是本题显存算术的基础。
- [[inference-serving-07]]：张量/流水/数据/专家并行的对比，对应本题的 EP/TP 选择。

## 参考资料与归属

- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（延伸）** —— Fedus et al. (Google)，2021-01-11：<https://arxiv.org/abs/2101.03961>。第 3 节的 **capacity factor 与 token 丢弃**机制、以及负载均衡损失的作用来自这篇。
- **Mixtral of Experts（延伸）** —— Jiang et al. (Mistral AI)，2024-01-08：<https://arxiv.org/abs/2401.04088>。第 1 节「总参 vs 激活参数」的表述与 top-2 路由、专家规模的工程口径来自这篇。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 1 节「decode 受带宽/显存约束、prefill 受算力约束」的框架用于说明 MoE 在小 batch 下的延迟特征。
- **估算服务一个 70B 模型所需的 GPU 显存（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1 节的权重/KV 显存算术与仓库统一常数（989 TFLOPs、80 GB 卡、bf16 2 字节）取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（隐藏维 8192、top-2 路由、64 专家、Zipf 指数 1.3、NVLink 900 GB/s、IB 50 GB/s、MFU 40%、10 万 token 的模拟、8/16/64 卡场景）都是为演示取舍而构造的**示例参数与显式假设**；真实的通信耗时取决于并行策略、拓扑与实现（分层 all-to-all、专家复制等），必须用目标集群实测。**「显存按总参、算力按激活」这一结构性结论与 capacity factor 的机制是可直接使用的部分。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
