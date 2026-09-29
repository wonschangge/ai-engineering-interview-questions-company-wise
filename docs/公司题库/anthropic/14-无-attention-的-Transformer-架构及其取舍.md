---
type: question
id: anthropic-14
company: Anthropic
topic: llm-internals
order: 14
question: 解释无 attention 的 Transformer 架构及其取舍。
question_en: Explain attention-free Transformer architectures and their trade-offs.
asked_at: []
level: 高阶
tags: [架构, 状态空间模型, 线性注意力, 长上下文, 硬件效率]
sources:
  - title: Mamba: Linear-Time Sequence Modeling with Selective State Spaces（延伸）
    url: https://arxiv.org/abs/2312.00752
    author: Gu & Dao
    published: 2023-12-01
  - title: Attention Is All You Need（延伸）
    url: https://arxiv.org/abs/1706.03762
    author: Vaswani et al. (NeurIPS 2017)
    published: 2017-06-12
  - title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）
    url: https://arxiv.org/abs/2205.14135
    author: Dao et al. (NeurIPS 2022)
    published: 2022-05-27
  - title: Lost in the Middle: How Language Models Use Long Contexts（延伸）
    url: https://arxiv.org/abs/2307.03172
    author: Liu et al. (TACL 2024)
    published: 2023-07-06
related: [llm-internals-01, llm-internals-02, anthropic-13, inference-serving-04, anthropic-17]
updated: 2026-09-28
---

## 一句话答案

> 「无 attention 架构」不是一个东西，而是**三条替换注意力机制的路线**，目标都是把 $O(T^2)$ 的注意力换成 $O(T)$（或 $O(T\log T)$）的算子：
> ① **线性注意力 / 核方法**——把 $\mathrm{softmax}(QK^\top)V$ 用特征映射 $\phi$ 重写成 $\phi(Q)\,(\phi(K)^\top V)$，利用结合律先算 $d\times d$ 的「状态矩阵」，复杂度降到线性（代价：表达力受限，近似 softmax 的精度损失）；
> ② **状态空间模型（SSM）**——用 $h_t=Ah_{t-1}+Bx_t,\ y_t=Ch_t$ 这类递推把历史压进**固定大小的状态**，Mamba 的关键贡献是让 $A,B,C$ 依赖输入（selective），从而在语言建模上追平同规模 Transformer（论文口径）；
> ③ **卷积/长卷积混合**——用全局或分段卷积（FFT 实现 $O(T\log T)$）替代注意力。
> **取舍可以压缩成三句话**：
> 1. **推理成本**：线性架构的 decode 每步只需固定大小的状态（不随上下文增长），而注意力的 KV cache 随 $T$ 线性增长——长上下文服务里这是决定性优势；
> 2. **训练并行性**：注意力的训练矩阵乘非常「硬」（大 GEMM，GPU 友好）；SSM 的递推是**顺序**的，虽然可以用并行扫描（扫描算法）恢复并行度，但硬件效率仍不如大 GEMM，工程实现更依赖定制 kernel；
> 3. **能力取舍**：固定状态必然带来**精确回忆（recall）能力下降**——需要「从上下文里逐字取出某个事实」的任务上，注意力仍然更强；工程上常见的是**混合架构**（少量注意力层 + 多数线性层），因为少数注意力层就能补回大部分 recall。
> 一句话：**无 attention 换的是「每步成本与上下文无关」，付的是「状态有损 + 训练/内核更复杂」**。

## 面试官在考什么

- **是否知道这是多条路线**：只提 Mamba 而不知道线性注意力/卷积路线，说明只跟过热点；能说出三条路线各自的核心思想与代价才过关。
- **复杂度账**：$O(T^2)$ vs $O(T)$ 的差别在 $T$ 多大时才真正重要（下表给了交叉点量级）；以及「每个 token 的 FLOPs」与「每步的延迟」是两回事（decode 阶段是带宽受限）。
- **recall vs throughput 的实证直觉**：为什么 Mamba 类模型在语言困惑度上能打，但在「大海捞针」（needle-in-a-haystack）与精确复制任务上偏弱——这与固定状态的信息瓶颈直接相关（可引用的对照是 *Lost in the Middle* 揭示的长上下文信息利用问题：即使有完整注意力，位置与利用率也不理想，说明「注意力」本身也不是万灵药）。
- **训练与推理的硬件现实**：attention 用标准 GEMM 与 FlashAttention 式 IO 优化即可高效；SSM 需要**并行扫描 + 融合 kernel**，实现难度与生态成熟度是真实成本（可引用的口径来自 FlashAttention：注意力的瓶颈往往是显存 IO 而不是 FLOPs——这提醒「复杂度低」不等于「实际更快」）。
- **混合架构与选型**：什么时候用纯线性（超长序列、流式、端侧）、什么时候保留注意力（需要精确检索、RAG、代码理解）。
- **能否给出判据**：不要停在「哪个更好」，而要给出**决策判据**（序列长度分布、是否需要精确回忆、硬件与内核生态、训练成本）。

**常见错误答案**

- 「Mamba 完全取代 Transformer」——忽略 recall 短板与生态成熟度（训练框架、内核、量化、并行策略）。
- 「线性注意力就是省显存，效果一样」——省的是 KV cache，损失的是表达力（近似 softmax 的误差）。
- 「复杂度低所以一定更快」——实际速度取决于 kernel 效率与内存访问；attention 有 FlashAttention 这类高度优化实现。
- 「RNN 又回来了」——把 SSM 简单等同于 RNN 会忽略并行扫描、选择性机制与训练并行度的关键差异。

## 原理与推导

### 1. 三条路线的数学形式

**（a）线性注意力**：取特征映射 $\phi$（如 ELU+1、ReLU），

$$\text{Attn}(Q,K,V)_t=\frac{\sum_{s\le t}\phi(q_t)^\top\phi(k_s)\,v_s}{\sum_{s\le t}\phi(q_t)^\top\phi(k_s)}=\frac{\phi(q_t)^\top\underbrace{\sum_{s\le t}\phi(k_s)v_s^\top}_{S_t\in\mathbb{R}^{d\times d}}}{\phi(q_t)^\top\underbrace{\sum_{s\le t}\phi(k_s)}_{z_t\in\mathbb{R}^{d}}}$$

状态 $S_t$ 尺寸 $d\times d$（与 $T$ 无关），每步更新 $O(d^2)$、生成 $O(d^2)$。**代价**：softmax 被换成核内积，注意力分布的「尖锐选择性」变弱。

**（b）状态空间模型**：连续时间形式 $h'(t)=Ah(t)+Bx(t)$，离散化后

$$h_t=\bar A h_{t-1}+\bar B x_t,\qquad y_t=Ch_t$$

经典 SSM（S4 系）的 $A,B,C$ 与输入无关（线性时不变），可以用卷积核并行训练；**Mamba 的核心改动是让 $\bar B,\bar C,\Delta$ 依赖输入**（selective），使模型能「根据内容决定记住/遗忘什么」——论文把这一条作为在语言任务上追平 Transformer 的关键，并强调其线性时间与推理时固定状态。

**（c）卷积/长卷积**：把序列混合写成 $y=x\ast w$（$w$ 为学到的长核），FFT 实现 $O(T\log T)$；隐式长卷积（如用 SSM 参数化核）是 (b) 与 (c) 的交集。

### 2. 复杂度与内存的真实对照

| 架构 | 训练（prefill） | 推理每步 | 推理状态大小 | 训练并行性 |
| --- | --- | --- | --- | --- |
| 全注意力 | $O(T^2d)$ | $O(Td)$（读 KV） | $O(Td)$ 且随 $T$ 增长 | 高（大 GEMM） |
| 线性注意力 | $O(Td^2)$ | $O(d^2)$ | $O(d^2)$ 固定 | 高（可并行扫描） |
| SSM/Mamba | $O(Td^2)$ 级 | $O(d^2)$ | $O(d^2)$ 固定 | 中（需并行扫描 + 定制 kernel） |
| 长卷积 | $O(T\log T\,d)$ | $O(\log T\,d)$ 级 | $O(Td)$ 或分块 | 高（FFT） |

**关键推论**：线性架构的「每步成本与上下文无关」在**长上下文 + 高并发 decode** 场景是决定性优势（KV cache 显存从 $O(T)$ 变成常数，直接决定能塞多少并发）。

### 3. 为什么 recall 会退化（信息瓶颈）

固定状态 $h_t\in\mathbb{R}^{d}$ 要把**整段历史**压进去，而注意力的 KV cache 保留了全部历史（$O(Td)$）。信息论上这是有损压缩：当任务需要「逐字取回某个事实」（例如在长文档里找一个随机数）时，固定状态必须把该事实保留在状态里——而状态容量与更新规则并不保证这一点。**这与模型能力无关，是架构的容量约束**。工程共识（也是大量公开对比的结论方向）是：

- **困惑度/流畅度**：线性架构可以接近同规模 Transformer；
- **精确回忆 / 大海捞针 / 复制**：注意力仍明显更强；
- **混合架构**（每隔若干层放一层注意力）能补回大部分差距，而成本仍接近纯线性。

### 4. 训练侧的硬件现实

- **注意力**：$QK^\top$ 是标准 batched GEMM，配合 FlashAttention 的 IO-aware 分块即可高效；
- **SSM**：递推需要**并行扫描**（associative scan）才能在训练时利用并行性；即便如此，其算子形态（逐元素 + 小矩阵）比大 GEMM 更难打满 Tensor Core。**同一个 FLOPs 数字，实际 wall-clock 可能差数倍**——这是评估「线性架构更快」时必须做的实测校正。
- 生态：内核实现、量化、张量并行/流水并行的支持度，决定了能不能真的把模型训起来/服务起来。

### 5. 选型判据（面试要给出的结论）

| 场景 | 建议 | 理由 |
| --- | --- | --- |
| 长文档问答 / 精确检索（RAG） | 保留注意力（或混合） | 需要精确 recall |
| 超长序列流式处理（日志、音频、传感） | 线性/SSM | 固定状态、每步常数成本 |
| 端侧/低内存设备 | 线性/SSM | 无 KV cache 增长 |
| 通用对话、代码 | 全注意力或混合 | 生态成熟 + 能力均衡 |
| 长上下文 + 高并发服务 | 混合或注意力 + KV 量化/分页 | 成本与能力折中 |

## 数值与代码验证

### 表 1：复杂度交叉点（什么时候线性架构的 FLOPs 优势才明显）

| $T$ | 注意力项 $\propto T^2 d$ | 线性项 $\propto T d^2$ | 比值（$d=4096$） |
| --- | --- | --- | --- |
| 1,024 | $4.3\times10^{9}$ | $1.7\times10^{10}$ | 0.25（线性反而更贵） |
| 4,096 | $6.9\times10^{10}$ | $6.9\times10^{10}$ | **1.0（交叉点）** |
| 16,384 | $1.1\times10^{12}$ | $2.7\times10^{11}$ | 4.0 |
| 65,536 | $1.8\times10^{13}$ | $1.1\times10^{12}$ | 16.0 |

**读法**：当 $T\approx d$ 时两者相当；只有在 $T\gg d$（长上下文）时线性架构的 FLOPs 优势才显著。这解释了为什么「无 attention 架构」的讨论总是与长上下文绑定。

### 表 2：推理阶段的状态大小（$d=4096$、$L=32$ 层、fp16，实测算式）

| 架构 | 每层状态 | 32 层合计 | 随 $T$ 增长？ |
| --- | --- | --- | --- |
| 全注意力 KV cache | $2\\cdot T\\cdot d\\cdot 2$ B | $T$=4K → **2 GiB**；16K → **8 GiB**；32K → **16 GiB**；128K → **64 GiB** | 是（线性） |
| 线性注意力状态 | $d^2\\cdot 2$ B = 32 MiB | **1 GiB（常数）** | 否 |
| SSM 状态 | $d\\cdot d_{state}\\cdot2$ B | 与 $T$ 无关（$d_{state}$ 常取 16–128） | 否 |

**读法**：并发服务时 KV cache 决定「一张卡能放多少条会话」——32K 上下文单序列就要 16 GiB，而线性架构恒为 1 GiB；在**长上下文 + 高并发**场景这是结构性优势。**注意**：单看 decode 每步的**带宽下界**，$T$=32K 时注意力每步至少要读 16 GiB（约 5.1 ms @3.35 TB/s），线性架构只要 0.32 ms——这就是「每步成本与上下文无关」的量化含义。

### 可运行代码

```python
# 三条路线的成本对照：FLOPs 交叉点、状态大小、以及"每步延迟"的量级
def attn_flops(T, d, L=32):
    """注意力打分 + 加权求和（每层 2*2*T^2*d 的量级）"""
    return L * 4 * T * T * d
def linear_flops(T, d, L=32):
    """线性注意力/SSM：每 token 每层 O(d^2) 级"""
    return L * 2 * T * d * d

d = 4096
print(f"{'T':>8} {'注意力 FLOPs':>14} {'线性 FLOPs':>14} {'比值':>7}")
for T in (1024, 4096, 16384, 65536):
    a, l = attn_flops(T, d), linear_flops(T, d)
    print(f"{T:>8,d} {a:>14.2e} {l:>14.2e} {a/l:>7.2f}")
print("交叉点：T ≈ d（本例 4096）—— 只有 T 远大于 d 时线性架构的 FLOPs 才占优")

# 推理状态大小：KV cache vs 固定状态（fp16）
BYTES, L = 2, 32
def kv_bytes(T, d, L=L, bytes_per=BYTES): return 2 * L * T * d * bytes_per
def linear_state_bytes(d, L=L, bytes_per=BYTES): return L * d * d * bytes_per
print(f"\n{'上下文 T':>10} {'KV cache':>12} {'线性状态':>12} {'倍数':>8}")
for T in (4096, 16384, 32768, 131072):
    kv, st = kv_bytes(T, d), linear_state_bytes(d)
    print(f"{T:>10,d} {kv/1024**3:>10.2f} GiB {st/1024**3:>10.2f} GiB {kv/st:>8.1f}x")

# decode 阶段的每步成本：注意力要读全部 KV（带宽受限），线性只要读状态
BW = 3.35e12            # H100 HBM 带宽 B/s（仓库统一常数）
print(f"\ndecode 每步读取量（单序列，fp16，32 层）")
for T in (4096, 32768):
    kv = kv_bytes(T, d)
    st = linear_state_bytes(d)
    print(f"  T={T:>6,d}: 注意力需读 {kv/1024**3:5.2f} GiB -> {kv/BW*1000:7.2f} ms/步（下界）"
          f"；线性读 {st/1024**3:.2f} GiB -> {st/BW*1000:.3f} ms/步")
print("读法：带宽下界显示长上下文 decode 里注意力的每步成本随 T 增长，而线性架构恒定——")
print("      这也是为什么长上下文服务要靠分页/量化/前缀缓存来压 KV（而不是只优化 FLOPs）")

# 信息瓶颈的直觉：固定状态能"记住"多少个可区分事实（粗略上界）
import math
def bits_in_state(d, bytes_per=BYTES):
    return d * bytes_per * 8
print(f"\n固定状态 d={d}、fp16：理论容量上界 ≈ {bits_in_state(d)/8/1024:.0f} KiB"
      f"（≈ {bits_in_state(d)/math.log2(2**16):,.0f} 个 16-bit 符号）")
print("对比：KV cache 容量随 T 线性增长（T=32K 时 16 GiB）—— 精确回忆的差距来自这里")
print("注：上面算的是 SSM 单个 d 维状态向量；线性注意力的矩阵状态 S_t 是 d×d，单层 32 MiB、32 层 1 GiB")
```

预期输出要点（实跑）：FLOPs 比值在 $T=4096$ 附近交叉（与 $d$ 同阶），$T=65536$ 时线性架构的优势达 16 倍；状态大小对照显示 KV cache 随 $T$ 线性增长（4K→2 GiB、16K→8 GiB、32K→16 GiB、128K→64 GiB），而线性状态恒为 1 GiB；**decode 每步带宽下界**：$T=32$K 时注意力每步至少要读 16 GiB（约 5.1 ms @3.35 TB/s，单序列、未计并发），线性架构仅 0.32 ms——这才是「每步成本与上下文无关」的量化含义；最后一段用状态容量上界说明 recall 差距的来源。

## 常见追问

- **追问**：既然线性架构每步更便宜，为什么现在主流还是 Transformer？
  - 要点：① 能力权衡（精确 recall）；② 训练与推理的生态成熟度（内核、并行、量化、工具链）；③ 中等上下文（$T\approx d$）下 FLOPs 优势不存在；④ 混合架构已经能拿到大部分收益，工程风险更小。
- **追问**：Mamba 的 「selective」 到底改变了什么？
  - 要点：让状态转移与读出依赖输入（$\bar B,\bar C,\Delta$ 随 $x_t$ 变化），因此模型可以「按内容决定记住或遗忘」；代价是失去线性时不变性，从而不能直接用卷积形式并行训练，需要并行扫描与定制 kernel（论文把这一点作为实现上的关键挑战）。
- **追问**：并行扫描是什么？
  - 要点：递推 $h_t=\bar A h_{t-1}+\bar B x_t$ 满足结合律，可以用前缀和式的并行扫描在对数深度内算出所有 $h_t$，从而在训练时利用并行性——这是「RNN 类模型能在大规模上训练」的技术前提。
- **追问**：混合架构怎么配比？
  - 要点：经验方向是「多数层线性 + 少数层注意力」（例如每 4–8 层放一层注意力），用少量注意力层补回 recall；配比要在目标任务上实测，而不是照搬论文。
- **追问**：长上下文的最优解是换架构吗？
  - 要点：不一定。先做工程侧优化（KV 量化、分页、前缀缓存、稀疏/滑窗注意力、prefill/decode 分离），这些成熟度更高、风险更低；换架构是更大的一步（串 [[anthropic-17]]）。
- **追问**：怎么评估一个「无 attention」模型是否可用？
  - 要点：分维度评测——困惑度/通用任务（可能持平）、长上下文精确回忆（needle 类）、复制/编辑类任务（弱项暴露）、以及**实际吞吐与延迟**（必须实测 wall-clock，不能只看 FLOPs）。

## 相关题目

- [[llm-internals-01]]：标准 Transformer 的注意力推导，是本题的对照基线。
- [[llm-internals-02]]：KV cache 的内存与带宽账，是「线性架构为何在长上下文服务里有优势」的前提。
- [[anthropic-13]]：Transformer 各组件的作用（含 FFN 占比 82%），解释了「替换注意力」只改了一部分成本结构。
- [[inference-serving-04]]：投机解码等解码侧优化，与线性架构在「降低每步成本」上目标一致、手段不同。
- [[anthropic-17]]：Claude 级 API 服务栈，长上下文与高并发下的架构选择直接落在这里。

## 参考资料与归属

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces（延伸）** —— Gu & Dao，2023-12-01：<https://arxiv.org/abs/2312.00752>。第 1 节与第 2 节的状态空间递推形式、「让参数依赖输入（selective）」这一关键改动、线性时间与推理固定状态的说法，以及实现上对并行扫描/定制 kernel 的依赖，均来自这篇论文摘要。
- **Attention Is All You Need（延伸）** —— Vaswani et al. (NeurIPS 2017)，2017-06-12：<https://arxiv.org/abs/1706.03762>。作为对照基线的注意力形式与位置编码来自这篇。
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）** —— Dao et al. (NeurIPS 2022)，2022-05-27：<https://arxiv.org/abs/2205.14135>。第 4 节「注意力瓶颈常在显存 IO 而非 FLOPs、因此「复杂度低」不等于「更快」」的判据来自这篇。
- **Lost in the Middle: How Language Models Use Long Contexts（延伸）** —— Liu et al. (TACL 2024)，2023-07-06：<https://arxiv.org/abs/2307.03172>。第「面试官在考什么」里「即使有完整注意力，长上下文的信息利用也不理想」这一对照结论来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$d=4096$、$L=32$、fp16、$T$ 取值、H100 带宽 3.35 TB/s）都是按本仓库统一口径构造的工程算例与显式假设；线性/SSM 的每步复杂度按其结构取 $O(d^2)$ 量级近似，具体实现（状态维 $d_{state}$、kernel 融合）会有差异，因此**实际 wall-clock 必须以实测为准**。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
