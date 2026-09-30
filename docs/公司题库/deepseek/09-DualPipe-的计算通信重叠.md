---
type: question
id: deepseek-09
company: DeepSeek
topic: inference-serving
order: 9
question: DualPipe 在训练中让计算与通信重叠。为什么在这个规模下这种重叠是决定性的，代价又是什么？
question_en: DualPipe overlaps computation and communication during training. Why is that overlap decisive at this scale, and what does it cost?
asked_at: []
level: 高阶
tags: [DualPipe, 流水线并行, 气泡, all-to-all 重叠, 训练吞吐]
sources:
  - title: DeepSeek-V3 Technical Report（延伸）
    url: https://arxiv.org/abs/2412.19437
    author: DeepSeek-AI
    published: 2024-12-27
  - title: PipeDream: Fast and Efficient Pipeline Parallel DNN Training（延伸）
    url: https://arxiv.org/abs/1806.00187
    author: Harlap et al.
    published: 2018-06-01
  - title: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（延伸）
    url: https://arxiv.org/abs/2104.04473
    author: Narayanan et al. (NVIDIA)
    published: 2021-04-09
  - title: 显存受限下低延迟服务 671B MoE（本仓库公司题库 · DeepSeek 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [deepseek-05, deepseek-07, deepseek-01, cohere-02, inference-serving-07]
updated: 2026-09-28
---

## 一句话答案

> **「决定性」来自两个数字的乘积：流水线气泡 × 未被隐藏的通信**——而在 671B 规模上，**这两个都很大**：
> ① **气泡大**：气泡占比 $\approx\frac{PP-1}{m+PP-1}$。**671B 需要大 PP**（显存），而**微批数 $m$ 受激活显存限制**——于是「大 PP + 小 m」正好落在气泡最差的区域。本机算例：**PP=16、m=8 时 1F1B 的气泡高达 65.2%**；
> ② **通信大**：MoE 的 **all-to-all**（dispatch + combine）与流水线的点对点通信，在 EP 度大时通信/计算比可达 **0.3–0.5**；若串行执行，**每步直接多花 30%–50%**。
> **DualPipe 的两个动作**：
> 1. **双向流水线**：每台设备**同时**处理两个方向（正/反）的微批 → **有效微批数翻倍**，气泡约减半：
> $$\text{气泡}_{\text{双向}}\approx\frac{PP-1}{2m+PP-1}$$
> 2. **把一个微批的 MoE 层切块，让 dispatch/combine 与计算交错** → 通信被算力掩盖（本机算例：通信/计算比 0.4 时，**串行 1.40 → 重叠后 1.08，约 1.3×**）。
> **合起来的收益**（$m=32$、通信/计算比 0.4）：**同一个 PP=16 下 0.486 → 0.750（1.54×）**；但**公平比较（同等每卡参数预算，即 PP=32 + DualPipe vs PP=16 + 1F1B）是 0.486 → 0.624，约 1.28×**——**这一条最容易被面试者忽略**。
> **代价（必须一起讲）**：
> | # | 代价 | 量化 |
> | --- | --- | --- |
> | ① | **每个设备要保存两个流水线阶段的参数** | 每卡参数显存 **2×** → 要维持同样预算，**PP 需加倍**（FP8 下从 PP=16 到 PP=32） |
> | ② | **调度复杂度大增** | 双向依赖 + 分块重叠 → 手工调度、正确性风险高 |
> | ③ | **需要足够的微批** | 双向要填满两个方向，$m$ 太小则双向退化为单向 |
> | ④ | **切块降低 GEMM 效率** | 块变小 → 单次 GEMM 更小、kernel 效率下降（与串 [[deepseek-01]] 的碎片化同源） |
> | ⑤ | **重叠有上限** | 重叠效率取决于 $\max(t_{\text{comp}},t_{\text{comm}})$：通信远大于计算时，重叠也救不回来 |
> 一句话判据：**DualPipe 用「参数显存翻倍 + 调度复杂度」换「气泡减半 + 通信隐藏」**——在 671B 这个「气泡与通信都很大」的区间里，这笔交易非常划算；在中小规模（PP 小、通信占比低）则未必值得。

## 面试官在考什么

- **能否量化气泡**：能否给出 $\frac{PP-1}{m+PP-1}$ 并指出「**大 PP + 小 m** 是气泡最差的区域」——这正是大模型训练的真实处境。
- **能否量化通信占比**：MoE 的 all-to-all 量级（串 [[cohere-02]]）与流水线点对点通信，通信/计算比 0.3–0.5 是否合理。
- **双向流水线的机制**：能否说清「每台设备同时处理两个方向」为什么能让**有效微批数翻倍**（气泡减半）。
- **分块重叠的机制**：能否说清「把 MoE 层切块，让一块的 dispatch 与另一块的计算交错」。
- **代价的完整性**：**参数显存翻倍**（最硬的一条）、调度复杂度、微批数要求、切块带来的 GEMM 效率损失、重叠上限。
- **Amdahl 视角**：能否指出「重叠只对通信部分有效」，所以**收益上限由通信占比决定**。
- **与 1F1B / ZeRO / 交错式 1F1B 的对比**：能否说出这些方案的取舍（激活显存 vs 气泡 vs 参数显存）。
- **规模依赖**：能否指出「**这个技术不是普适的**」——小规模下气泡本来就小、通信占比低，收益有限而复杂度上升。
- **诚实**：承认 DualPipe 的实现细节（调度表、依赖管理）是**工程壁垒**，不是「开个开关」。

**常见错误答案**

- 只说「DualPipe 让计算通信重叠」（不说**双向流水线**与**气泡减半**这一半收益）。
- 忽略**参数显存翻倍**这个最硬的代价。
- 认为「重叠可以无限提升」（重叠上限是 $\max(t_{\text{comp}}, t_{\text{comm}})$）。
- 不提微批数要求（双向需要足够多的微批才有意义）。
- 忽略切块带来的 GEMM 效率损失。
- 把气泡公式记错（气泡是 $\frac{PP-1}{m+PP-1}$，不是 $\frac{1}{m}$）。
- 认为该技术在所有规模都值得（小规模下收益低、复杂度高）。

## 原理与推导

### 1. 流水线气泡：为什么「大 PP + 小 m」最惨

在 1F1B（one-forward-one-backward）调度下，流水线填充与排空阶段设备会空闲：

$$\text{气泡占比}\approx\frac{PP-1}{m+PP-1}$$

其中 $PP$ 为流水线深度、$m$ 为微批数。**两个极端**：

| 场景 | 气泡 |
| --- | --- |
| $PP=4,\ m=64$ | 4.5% |
| $PP=16,\ m=32$ | **31.9%** |
| $PP=16,\ m=8$ | **65.2%** |

**为什么大模型落在坏区间**：① **显存压力要求大 PP**（参数与优化器状态切分）；② **激活显存限制了 $m$**（每个在飞微批都要存激活）。**两者夹击 → 气泡成为一阶问题。**

### 2. 双向流水线：有效微批数翻倍

DualPipe 让**每台设备同时处理两个方向的微批**：设备 $i$ 既跑正向的第 $i$ 段，也跑反向的第 $PP-1-i$ 段。于是**同一时刻在飞的微批数翻倍**：

$$\text{气泡}_{\text{双向}}\approx\frac{PP-1}{2m+PP-1}$$

**代价**：每台设备要保存**两个流水线阶段的参数**——这就是「参数显存翻倍」的来源（见下）。

### 3. 分块重叠：让 all-to-all 藏进算力里

MoE 层的一步是「dispatch → 专家计算 → combine」。把微批切成两块后：

```
块 A: dispatch(A) → compute(A) → combine(A)
块 B:              dispatch(B) → compute(B) → combine(B)
                  ↑ 与 compute(A) 重叠
```

**理想**：$T=\max(t_{\text{comp}},t_{\text{comm}})$；**实际**（重叠效率 $\eta$）：$T\approx t_{\text{comp}}+(1-\eta)t_{\text{comm}}$。

| 通信/计算比 | 串行 | 完全重叠 | 实际（$\eta=0.8$） | 加速 |
| --- | --- | --- | --- | --- |
| 0.1 | 1.10 | 1.00 | 1.02 | 1.08× |
| 0.4 | 1.40 | 1.00 | 1.08 | **1.30×** |
| 1.0 | 2.00 | 1.00 | 1.20 | 1.67× |
| 2.0 | 3.00 | 2.00 | 1.40 | 2.14× |

**读法**：**重叠的收益随通信占比上升，但上限是 $\max(t_{\text{comp}},t_{\text{comm}})$**——通信远大于计算时，重叠也救不回来（要减少通信本身：专家放置、分层 all-to-all）。

### 4. 内存代价：每卡要放两个阶段的参数

$$\text{每卡参数显存}=\frac{2\times P_{\text{total}}\times b}{PP}$$

**例**（FP8、$b=1$、671 GB）：若单卡参数预算约 60 GB（留出激活/梯度/优化器状态），则

$$PP\ge\frac{2\times671}{60}\approx22\ \Rightarrow\ PP=32$$

**读法**：**DualPipe 实际上把「每卡参数预算」减半，所以要把流水线深度加倍**（PP=16 → 32）。这是它最硬的代价：**省下的气泡是用更多卡换来的**（或者说是用更细的参数切分换来的）。

### 5. 端到端：收益与代价放在一起看

$$\text{有效利用率}\approx\underbrace{(1-\text{气泡})}_{\text{调度}}\times\underbrace{\frac{1}{1+(1-\eta)\cdot r}}_{\text{通信隐藏}},\qquad r=\frac{t_{\text{comm}}}{t_{\text{comp}}}$$

**本机算例**（$m=32$、$r=0.4$）：

| 配置 | 气泡 | 通信因子 | 有效利用率 |
| --- | --- | --- | --- |
| PP=16 + 1F1B（通信串行） | 31.9% | 0.714 | **0.486** |
| PP=16 + DualPipe | 19.0% | 0.926 | 0.750 |
| PP=32 + 1F1B（同等参数预算） | 49.2% | 0.714 | 0.363 |
| **PP=32 + DualPipe（同等参数预算）** | 32.6% | 0.926 | **0.624** |

**读法**：同一个 PP 下比较会得到 1.54×，但**DualPipe 把每卡参数预算减半，所以必须把 PP 加倍**——**在同等每卡预算下，真实收益是 0.486 → 0.624，约 1.28×**。**这是面试里最容易忽略的一点：技术的收益要在「同等资源约束」下比较。**

## 数值与代码验证

### 表 1：流水线气泡占比（1F1B vs 双向）

| PP | m | 1F1B 气泡 | 双向气泡 | 气泡减半的加速 |
| --- | --- | --- | --- | --- |
| 4 | 8 | 27.3% | 15.8% | 1.16× |
| 4 | 32 | 8.6% | 4.5% | 1.04× |
| 4 | 64 | 4.5% | 2.3% | 1.02× |
| 8 | 8 | 46.7% | 30.4% | 1.30× |
| 8 | 32 | 17.9% | 9.9% | 1.10× |
| 8 | 64 | 9.9% | 5.2% | 1.05× |
| **16** | **8** | **65.2%** | **48.4%** | **1.48×** |
| 16 | 32 | 31.9% | 19.0% | 1.19× |
| 16 | 64 | 19.0% | 10.5% | 1.10× |

### 表 2：通信重叠与内存代价

| 项 | 数值 |
| --- | --- |
| 通信/计算比 0.4 时的重叠收益 | 串行 1.40 → 实际 1.08（**约 1.30×**），上限是 $\max(\text{计算},\text{通信})$ |
| 每卡参数（PP=16 / 32 / 64，双向） | 83.9 G / 41.9 G / 21.0 G（PP=16 双向**超预算**） |
| 维持每卡 60 GB 预算所需最小 PP | 单方向 **12**、双向 **23**（工程取 2 的幂 → 16 / 32） |
| 端到端有效利用率（m=32、通信比 0.4） | PP=16+1F1B **0.486** → PP=16+DualPipe **0.750**（1.54×）；**同等每卡预算**下 PP=32+DualPipe **0.624**（约 1.28×） |

### 可运行代码

```python
# DualPipe：流水线气泡、通信重叠、内存代价与端到端收益
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

GB = 1e9
GPU_MEM = 80e9

# ---------- 1) 流水线气泡 ----------
def bubble_1f1b(pp: int, m: int) -> float:
    """1F1B：气泡占比 ≈ (PP-1)/(m+PP-1)"""
    return (pp - 1) / (m + pp - 1)

def bubble_dualpipe(pp: int, m: int) -> float:
    """双向流水线：每台设备同时处理两个方向 -> 有效微批数翻倍，气泡约减半"""
    return (pp - 1) / (2 * m + pp - 1)

print("① 流水线气泡占比（微批数 m、流水线深度 PP）")
print(f"  {'PP':>4} {'m':>5} {'1F1B 气泡':>10} {'双向气泡':>10} {'气泡减半的加速':>15}")
for pp in (4, 8, 16):
    for m in (8, 32, 64):
        b1, b2 = bubble_1f1b(pp, m), bubble_dualpipe(pp, m)
        print(f"  {pp:>4} {m:>5} {b1:>10.1%} {b2:>10.1%} {(1 - b2) / (1 - b1):>14.2f}x")
print("  读法：**PP 越大、m 越小，气泡越致命**（PP=16、m=8 时 1F1B 高达 65.2%）——")
print("        而 671B 训练正好落在「大 PP（显存）+ 小 m（激活显存）」的坏区间，所以气泡是一阶问题")

# ---------- 2) 通信与计算重叠 ----------
def overlap_time(t_comp: float, t_comm: float, eff: float = 0.8) -> float:
    """把微批切块后的实际时间：理想是 max(comp, comm)，按 eff 比例重叠"""
    return t_comp + (1 - eff) * t_comm

print("")
print("② MoE 层的 all-to-all 与计算重叠（微批切块、dispatch 与 compute 交错）")
print(f"  {'通信/计算比':>12} {'串行':>8} {'完全重叠':>10} {'实际(η=0.8)':>13} {'加速':>7}")
for ratio in (0.1, 0.3, 0.4, 0.5, 1.0, 2.0):
    serial = 1.0 + ratio
    ideal = max(1.0, ratio)
    real = overlap_time(1.0, ratio)
    print(f"  {ratio:>12.1f} {serial:>8.2f} {ideal:>10.2f} {real:>13.2f} {serial / real:>6.2f}x")
print("  读法：**重叠收益随通信占比上升，但上限是 max(计算, 通信)** ——")
print("        通信/计算比 0.4 时约 1.30x；若通信是计算的 2 倍，重叠只能把 3.00 压到 1.40")

# ---------- 3) 内存代价：每卡两个阶段的参数 ----------
@dataclass
class MemCost:
    total_params_b: float = 671.0
    bytes_per_param: float = 1.0        # FP8
    param_budget_gb: float = 60.0       # 每卡留给参数的预算（其余给激活/梯度/优化器状态）
    def per_device_gb(self, pp: int, dual: bool) -> float:
        copies = 2 if dual else 1
        return copies * self.total_params_b * self.bytes_per_param / pp
    def min_pp(self, dual: bool) -> int:
        return math.ceil(2 * self.total_params_b * self.bytes_per_param / self.param_budget_gb) \
            if dual else math.ceil(self.total_params_b * self.bytes_per_param / self.param_budget_gb)
M = MemCost()
print("")
print("③ 内存代价：DualPipe 下每卡要保存两个流水线阶段的参数")
print(f"  {'PP':>4} {'单方向 每卡参数':>16} {'双向 每卡参数':>15} {'双向是否超预算':>15}")
for pp in (16, 32, 64):
    one = M.per_device_gb(pp, dual=False)
    two = M.per_device_gb(pp, dual=True)
    ok = "否" if two <= M.param_budget_gb else "**是（需更大 PP）**"
    print(f"  {pp:>4} {one:>15.1f}G {two:>14.1f}G {ok:>15}")
print(f"  维持每卡 {M.param_budget_gb:.0f} GB 参数预算所需的最小 PP："
      f"单方向 {M.min_pp(False)}、双向 **{M.min_pp(True)}**")
print("  读法：**参数显存翻倍是最硬的代价** —— 要维持同样的每卡预算，PP 必须加倍；")
print("        也就是说 DualPipe 省下的气泡，是用更多卡（更细的参数切分）换来的")

# ---------- 4) 端到端：气泡 + 重叠一起算 ----------
def utilization(pp: int, m: int, ratio: float, dual: bool, eff: float = 0.8) -> float:
    b = bubble_dualpipe(pp, m) if dual else bubble_1f1b(pp, m)
    comm_factor = 1.0 / overlap_time(1.0, ratio, eff) if dual else 1.0 / (1.0 + ratio)
    return (1 - b) * comm_factor
print("")
print("④ 端到端有效利用率（m=32、通信/计算比 0.4）")
print(f"  {'配置':<34} {'气泡':>8} {'通信因子':>9} {'利用率':>8}")
for label, pp, dual in (("PP=16 + 1F1B（通信串行）", 16, False),
                        ("PP=16 + DualPipe", 16, True),
                        ("PP=32 + 1F1B（同等参数预算）", 32, False),
                        ("PP=32 + DualPipe（同等参数预算）", 32, True)):
    b = bubble_dualpipe(pp, 32) if dual else bubble_1f1b(pp, 32)
    cf = 1.0 / overlap_time(1.0, 0.4) if dual else 1.0 / 1.4
    print(f"  {label:<34} {b:>8.1%} {cf:>9.3f} {utilization(pp, 32, 0.4, dual):>8.3f}")
print("  读法：**公平的比较是「同等每卡参数预算」下的对比**（PP=32+DualPipe vs PP=16+1F1B：0.624 vs 0.486，约 1.28x）——")
print("        若只在同一个 PP 下比较，会高估 DualPipe 的收益（因为它偷走了每卡参数预算）")

# ---------- 5) 代价清单 ----------
@dataclass
class Cost:
    item: str
    quantified: str
    mitigation: str
COSTS = [
    Cost("每卡参数显存翻倍", "671B FP8 下 PP 需从 16 加到 32", "更深 PP / 更激进量化 / 参数卸载"),
    Cost("调度复杂度", "双向依赖 + 分块交错，手工排程", "静态调度表 + 严格正确性测试"),
    Cost("需要足够微批", "m 太小则双向退化为单向", "控制激活显存以开大 m（重计算、激活卸载）"),
    Cost("切块降低 GEMM 效率", "块变小 -> 单次 GEMM 更小", "块大小与 tensor core 对齐"),
    Cost("重叠存在上限", "上限 = max(计算, 通信)", "同时减少通信本身（放置、分层 all-to-all）"),
]
print("")
print("⑤ 代价清单")
print(f"  {'代价':<20} {'量化':<34} 缓解")
for c in COSTS:
    print(f"  {c.item:<20} {c.quantified:<34} {c.mitigation}")
print("  读法：**DualPipe 不是「免费的加速」** —— 它把成本从「时间」搬到「显存与复杂度」；")
print("        在 671B 这个气泡与通信都很大的区间里划算，在中小规模下未必值得")
```

预期输出要点（实跑）：① **气泡随 PP 增大、m 减小而急剧恶化**（PP=16、m=8 时 1F1B 高达 **65.2%**，双向仍有 48.4%）——而 671B 正好落在「大 PP + 小 m」的坏区间；② **通信重叠**在通信/计算比 0.4 时把串行的 1.40 压到 1.08（**约 1.30×**），但上限是 $\max(\text{计算},\text{通信})$；③ **参数显存翻倍**：每卡要放两个阶段的参数，维持 60 GB 预算所需的最小 PP 从 **12 变成 23**（工程上取 2 的幂 → **16 变 32**）；④ **端到端有效利用率**：同一个 PP 下 0.486 → 0.750（1.54×），但**同等每卡参数预算下是 0.486 → 0.624（约 1.28×）**——**收益必须在同等资源约束下比较**；⑤ 代价清单把「时间换显存与复杂度」的交易逐条量化。

## 常见追问

- **追问**：DualPipe 与 1F1B、交错式 1F1B（interleaved 1F1B）有什么区别？
  - 要点：**1F1B** 是最基础的调度（气泡 $\frac{PP-1}{m+PP-1}$，激活显存 $\propto PP$）；**交错式 1F1B** 让每台设备负责多个不连续的段，**减小气泡但增加通信量**；**DualPipe** 走的是**双向**路线（每台设备同时跑两个方向），**气泡减半但参数显存翻倍**。**三者的取舍维度不同：气泡 / 激活显存 / 参数显存 / 通信量。**
- **追问**：为什么微批数 $m$ 不能随便开大？
  - 要点：**激活显存 $\propto m$**（在飞的微批都要存激活）。缓解手段：**激活重计算**（用算力换显存）、**激活卸载**、**序列并行**。**而 $m$ 太小又会让气泡变严重**——这就是「气泡与激活显存」的正面冲突。
- **追问**：重叠效率为什么不是 100%？
  - 要点：① **依赖关系**（combine 必须在 compute 之后）；② **资源竞争**（通信与计算抢 SM/带宽）；③ **块粒度**（块太小则调度开销占比上升）；④ **负载不均**（某些专家的 all-to-all 更慢，等待）。**所以实际重叠效率常在 70%–90%。**
- **追问**：什么时候不值得用 DualPipe？
  - 要点：① **PP 小**（气泡本来就小，如 PP=4、m=64 时只有 4.5%）；② **通信占比低**（重叠收益小）；③ **显存极度紧张**（参数翻倍放不下）；④ **团队缺少调度实现能力**（复杂度是真实成本）。**判据：先算气泡与通信占比，再决定。**
- **追问**：这套思路能用到推理吗？
  - 要点：**部分可以**。推理侧的「重叠」更多是 **PD 分离**（预填充与解码分离，串 [[deepseek-05]]）与**计算/通信重叠**（EP 的 all-to-all 与 GEMM 交错）；但推理没有反向传播，所以**流水线气泡问题不同**（decode 阶段是逐 token 的，不存在微批填充/排空）。**别把训练调度直接搬到推理。**
- **追问**：怎么验证重叠真的生效？
  - 要点：① **计时分解**（把一步拆成 compute / all-to-all / p2p / 空闲，用 profiler 看时间线）；② **气泡可视化**（每台设备的忙碌/空闲时间线）；③ **消融**（关掉重叠看 step time 变化）；④ **端到端**（tokens/s per GPU 与 MFU）。**关键是能看见「空闲在哪」。**

## 相关题目

- [[deepseek-05]]：671B 的低延迟服务——推理侧的并行与重叠（PD 分离）。
- [[deepseek-07]]：FP8 训练——与本篇并列的两大训练吞吐/稳定话题。
- [[deepseek-01]]：细粒度 MoE——切块带来的 GEMM 碎片化在本篇再次出现。
- [[cohere-02]]：稀疏 MoE 的 all-to-all 量级——本篇通信占比的来源。
- [[inference-serving-07]]：张量/流水/数据/专家并行的完整对比。

## 参考资料与归属

- **DeepSeek-V3 Technical Report（延伸）** —— DeepSeek-AI，2024-12-27：<https://arxiv.org/abs/2412.19437>。第 1、2 节的 **DualPipe（双向流水线调度、计算-通信重叠、参数显存翻倍而激活显存与 1F1B 相当）**来自该报告。
- **PipeDream: Fast and Efficient Pipeline Parallel DNN Training（延伸）** —— Harlap et al.，2018-06-01：<https://arxiv.org/abs/1806.00187>。第 1 节流水线训练的填充/排空气泡模型（1F1B 的前身）来自这篇。
- **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（延伸）** —— Narayanan et al. (NVIDIA)，2021-04-09：<https://arxiv.org/abs/2104.04473>。第 3 节的 **1F1B 与交错式 1F1B 的气泡公式与激活显存权衡**来自这篇。
- **显存受限下低延迟服务 671B MoE（本仓库公司题库 · DeepSeek 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 2 节的显存账（671 GB 权重、80 GB 卡）被本篇用于计算「参数翻倍」的代价。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（PP 取 4/8/16/32/64、$m$ 取 8/32/64、通信/计算比 0.1–2.0、重叠效率 $\eta=0.8$、每卡参数预算 60 GB、671B FP8）都是为演示机制而构造的**示例参数与简化模型**；**气泡公式是 1F1B 的经典近似**，DualPipe 的「气泡减半」在本机用「有效微批数翻倍」近似，**真实调度的时间线更复杂**（依赖、分块交错、通信竞争）。可迁移的结论是：**（1）气泡由「大 PP + 小 m」放大，而大模型正好落在该区间；（2）重叠收益上限是 max(计算, 通信)；（3）DualPipe 的硬代价是每卡参数显存翻倍；（4）公平比较必须在同等参数预算下进行。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
