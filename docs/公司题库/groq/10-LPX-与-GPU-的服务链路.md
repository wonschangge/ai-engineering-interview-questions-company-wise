---
type: question
id: groq-10
company: Groq
topic: system-design
order: 10
question: 我们把 LPX decode 加速器与负责 prefill 和 attention 的 NVIDIA GPU 搭配使用。请设计跨这两类机器的服务链路。
question_en: We pair an LPX decode accelerator with NVIDIA GPUs that handle prefill and attention. Design the serving pipeline across these two machine types.
asked_at: []
level: 高阶
tags: [异构服务, KV 传输, prefill/decode 分离, 供给比例, 状态位置]
sources:
  - title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（延伸）
    url: https://arxiv.org/abs/2401.09670
    author: Zhong et al.
    published: 2024-01-23
  - title: Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving（延伸）
    url: https://arxiv.org/abs/2407.00079
    author: Qin et al. (Moonshot AI)
    published: 2024-06-28
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [groq-06, groq-09, groq-11, nvidia-08, perplexity-05]
updated: 2026-09-28
---

## 一句话答案

> **一次性的 KV 搬运是可接受的（占 prefill 计算的约 1.9%）**——**但 decode 阶段的 attention 必须在 KV 所在的那一侧**：**否则每 token 都要搬整个 KV，100K 上下文下只有 1.5 tok/s。**
> **★ 量化一：KV 传输 vs prefill 计算（比值恒为 1.9%）**
> 70B INT8、KV 320 KiB/token、IB 50 GB/s、GPU 有效算力 396 TFLOPs
> | prompt 长度 | **KV 大小** | IB 传输 | **prefill 计算** | **传输/计算** |
> | --- | --- | --- | --- | --- |
> | 1,000 | 0.33 GB | 6.6 ms | **354 ms** | **1.9%** |
> | 10,000 | 3.28 GB | 65.5 ms | 3,539 ms | **1.9%** |
> | 100,000 | 32.77 GB | 655.4 ms | 35,389 ms | **1.9%** |
> **读法**：**传输/计算的比值恒为约 1.9%**（**因为两者都随 prompt 长度线性增长**）——**所以"prefill 在 GPU、decode 在 LPX"的一次性 KV 搬运是可接受的**。
> **★ 量化二：关键约束——decode 的 attention 不能跨机器**
> 若 KV 在 LPX 而 attention 在 GPU，**每个 token 都要把整个 KV 搬过去**
> | 上下文 | **KV 大小** | **每 token 传输** | **IB 上限 tok/s** | 结论 |
> | --- | --- | --- | --- | --- |
> | 1,000 | 0.33 GB | 6.6 ms | **152.6** | **可接受** |
> | 10,000 | 3.28 GB | 65.5 ms | **15.3** | **已明显变慢** |
> | **100,000** | **32.77 GB** | **655.4 ms** | **1.5** | **不可用** |
> **读法**：**100K 上下文时每 token 要搬 32.77 GB → 1.5 tok/s（完全不可用）**——**所以"attention 在 GPU"只能指"prefill 阶段"**（**那时 KV 正在被生产**）。**decode 阶段的 attention 必须在 KV 所在的地方**。**这是本题最重要的一条设计约束："状态不能每个 token 搬一次"**。
> **★ 量化三：供给比例（1K prompt 需要约 70 张 GPU 喂一个 LPX 架）**
> | prompt:输出 | prefill 时间 | decode 时间 | **GPU:LPX** |
> | --- | --- | --- | --- |
> | **1000:100** | 354 ms | 5.0 ms | **70.8:1** |
> | 1000:500 | 354 ms | 25.0 ms | **14.2:1** |
> | 10000:100 | 3,539 ms | 5.0 ms | **707.8:1** |
> | 10000:500 | 3,539 ms | 25.0 ms | **141.6:1** |
> **读法**：**1K prompt + 100 token 输出时需要约 70 张 GPU 才能喂饱一个 LPX 机架**——**因为 prefill 是"计算受限 + 每请求 354 ms"（单卡只有 2.83 请求/s）**，**而 LPX 机架的 decode 是 200 请求/s**。**所以"LPX + GPU"在聊天负载下实际是"小 LPX 架 + 大 GPU 群"**——**prefill 才是瓶颈**。
> **★ 链路的五个阶段**：
> | # | 阶段 | 在哪 | 关键点 |
> | --- | --- | --- | --- |
> | ① | **tokenize + prefill** | **GPU** | **计算受限，用 tensor core** |
> | ② | **KV 搬运** | **GPU → LPX** | **一次性，占 1.9%** |
> | ③ | **decode（含 attention）** | **LPX** | **KV 不动，权重常驻**（**串 [[groq-06]]**） |
> | ④ | **采样 + 流式返回** | **LPX** | **确定性 → p99 可承诺**（**串 [[groq-08]]**） |
> | ⑤ | **GPU 侧继续 prefill 下一个请求** | **GPU** | **跨请求流水线，避免 GPU 空转** |
> **读法**：**第 ⑤ 阶段是"让 GPU 不空转"的关键**——**因为 decode 期间 GPU 本来是闲的**（**除非它去做下一个请求的 prefill**）。
> **★ 三个必须解决的工程问题**：
> | 问题 | 解法 |
> | --- | --- |
> | **KV 的格式一致性** | **两侧用同一布局**（**否则要转换**） |
> | **KV 的生命周期** | **谁负责释放**（**串 [[nvidia-02]] 的引用计数**） |
> | **失败处理** | **LPX 失败则 KV 还在 GPU**（**可重试**） |
> 一句话判据：**"GPU 做 prefill 并一次性把 KV 交给 LPX → decode（含 attention）全在 LPX → GPU 立刻去做下一个请求的 prefill；供给比例按 prompt:输出分布定（1K:100 时约 70:1）"**。

## 面试官在考什么

- **★ 是否指出"attention 不能跨机器"**：**能否量化"100K 时 1.5 tok/s"**——**这是本题的分水岭**。
- **★ 是否算 KV 传输占比**：**能否给出"1.9%"**。
- **★ 是否算供给比例**：**能否给出"约 70:1"**。
- **是否指出"prefill 才是瓶颈"**：**能否指出 GPU 群要远大于 LPX 架**。
- **是否让 GPU 不空转**：**能否指出"跨请求流水线"**。
- **是否知道 KV 格式**：**能否指出"两侧要一致"**。
- **是否知道 KV 生命周期**：**能否指出"谁释放"**。
- **是否知道失败处理**：**能否指出"KV 还在 GPU 可重试"**。
- **是否知道为什么分离**：**能否指出 prefill 计算受限、decode 带宽受限**。
- **诚实**：**承认"分离引入了 KV 搬运这一新的失败点"**。

**常见错误答案**

- **让 GPU 做 decode 的 attention**（**每 token 搬 KV**）。
- **不算 KV 传输占比**（**不知道可不可接受**）。
- **假设"1:1 配置"**（**prefill 才是瓶颈**）。
- **让 GPU 在 decode 期间空转**。
- **忽略 KV 格式一致性**。
- **不定义 KV 的生命周期**。
- **不做失败处理**（**LPX 挂了 KV 就丢了**）。
- **认为"分离总是更好"**（**它引入了新失败点**）。

## 原理与推导

### 1. ★ 为什么分离

| 阶段 | 受限 | 适合的机器 |
| --- | --- | --- |
| **prefill** | **算力** | **GPU（tensor core + HBM）** |
| **decode** | **带宽** | **LPX（SRAM + fabric + 确定性）** |

**读法**：**"两个阶段的瓶颈不同"**——**所以用不同的机器是自然的**（**串 [[nvidia-01]] 的 roofline**）。

### 2. ★ KV 传输的占比

$$\frac{T_{\text{传输}}}{T_{\text{prefill}}}=\frac{L\times\text{KV/token}/\text{BW}_{\text{link}}}{2PL/(\text{峰值}\times\text{MFU})}$$

| 项 | 值 |
| --- | --- |
| **与 $L$ 无关** | **因为分子分母都含 $L$** |
| **本机** | **1.9%** |

**读法**：**"比值恒定"是一个很好的性质**——**它意味着"长 prompt 不会让搬运变成瓶颈"**。

### 3. ★ 状态不能每 token 搬

| 方案 | 每 token 传输 |
| --- | --- |
| **KV 在 LPX、attention 在 LPX** | **0** |
| **KV 在 LPX、attention 在 GPU** | **整个 KV** |

**读法**：**"这是'状态放在哪'的经典问题"**——**而答案是"计算必须去找状态，而不是状态去找计算"**。

### 4. ★ 供给比例

$$\text{GPU 数}=\frac{\text{LPX 的请求吞吐}}{\text{GPU 的 prefill 请求吞吐}}$$

| 配置 | 比例 |
| --- | --- |
| **1K prompt + 100 输出** | **70.8:1** |
| 10K prompt + 100 输出 | **707.8:1** |

**读法**：**"prefill 的吞吐远低于 decode"**——**所以 GPU 群要大得多**（**这与"decode 是瓶颈"的直觉相反**）。

### 5. ★ 跨请求流水线

| 时间线 | GPU | LPX |
| --- | --- | --- |
| **t0** | **prefill 请求 1** | **空闲** |
| **t1** | **prefill 请求 2** | **decode 请求 1** |
| **t2** | **prefill 请求 3** | **decode 请求 2** |

**读法**：**"GPU 与 LPX 同时工作"**——**这是分离架构的真正收益**（**而不是"各自更快"**）。

### 6. 新引入的失败点

| 失败点 | 后果 |
| --- | --- |
| **KV 传输失败** | **请求失败，但 KV 还在 GPU**（**可重试**） |
| **LPX 失败** | **同上** |
| **KV 格式不匹配** | **静默错误**（**最危险**） |

**读法**：**"分离引入了 KV 这一新的状态转移"**——**而它必须被显式管理**。

## 数值与代码验证

### 表 1：KV 传输占比、attention 位置、供给比例（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
print('① KV 传输 vs prefill 计算：比值恒定')
P=70e9; KV=320*1024; PCIE=25e9; IB=50e9; PEAK=989e12; MFU=0.40
print(f'  70B INT8 权重、KV {KV/1024:.0f} KiB/token、PCIe {PCIE/1e9:.0f} GB/s、IB {IB/1e9:.0f} GB/s')
print(f'  {"prompt 长度":>11} {"KV 大小":>10} {"PCIe 传输":>10} {"IB 传输":>9} {"prefill 计算":>12} {"传输/计算":>10}')
for L in (1000,10000,100000):
    kv=L*KV
    prefill=2*P*L/(PEAK*MFU)
    print(f'  {L:>11,} {kv/1e9:>8.2f} GB {kv/PCIE*1000:>8.1f} ms {kv/IB*1000:>7.1f} ms '
          f'{prefill*1000:>10.0f} ms {kv/IB/prefill:>9.1%}')
print('  读法：**传输/计算的比值恒为约 1.8%**（**因为两者都随 prompt 长度线性增长**）——')
print('        所以**"prefill 在 GPU、decode 在 LPX"的一次性 KV 搬运是可接受的**；')
print("        而**这个比值与「模型大小」和「KV 每 token 字节」有关**（**与长度无关**）")
print()
print('② 关键设计约束：decode 阶段的 attention 不能跨机器')
print(f'  若 KV 在 LPX、而 attention 在 GPU：**每个 token 都要把整个 KV 搬过去**')
print(f'  {"上下文":>9} {"KV 大小":>10} {"每 token 传输":>12} {"IB 上限 tok/s":>13} 结论')
for L in (1000,10000,100000):
    kv=L*KV
    t=kv/IB
    tag='**可接受**' if 1/t>100 else ('**已明显变慢**' if 1/t>10 else '**不可用**')
    print(f'  {L:>9,} {kv/1e9:>8.2f} GB {t*1000:>10.1f} ms {1/t:>13.2f} {tag}')
print('  读法：**1K 上下文时 152.6 tok/s（可接受）、10K 时 15.3（已明显变慢）、100K 时 1.5（不可用）** ——')
print("        所以**「attention 在 GPU」只能指「prefill 阶段」**（**那时 KV 正在被生产**）；")
print('        而**decode 阶段的 attention 必须在 KV 所在的地方**（**即 LPX 上**）——')
print('        这是本题最重要的一条设计约束（**"状态不能每个 token 搬一次"**）')
print()
print('③ 供给比例：多少 GPU 配一个 LPX 机架')
LPU_TOK=20000      # LPX 机架吞吐（tok/s，示例）
print(f'  {"prompt:输出":>12} {"prefill 时间":>12} {"decode 时间":>12} {"GPU:LPX":>10} 说明')
for L,out in ((1000,100),(1000,500),(10000,100),(10000,500)):
    prefill=2*P*L/(PEAK*MFU)
    decode=out/LPU_TOK
    ratio=prefill/decode
    print(f'  {str(L)+":"+str(out):>12} {prefill*1000:>10.0f} ms {decode*1000:>10.1f} ms {ratio:>9.1f}:1 '
          f'{"**prefill 主导**" if ratio>2 else ("**decode 主导**" if ratio<0.5 else "**接近平衡**")}')
print('  读法：**1K prompt + 100 token 输出时需要约 70 张 GPU 才能喂饱一个 LPX 机架** ——')
print('        因为**prefill 是"计算受限 + 每请求 354 ms"**（**单卡只有 2.83 请求/s**）——')
print('        而**LPX 机架的 decode 是 200 请求/s**（**20,000 tok/s ÷ 100 token**）；')
print("        所以**「LPX + GPU」在聊天负载下实际是「小 LPX 架 + 大 GPU 群」**（**prefill 才是瓶颈**）——")
print('        而**输出越长、GPU:LPX 的比例越低**（**500 token 输出时降到 14:1**）')
```

预期输出要点（实跑）：① **KV 传输**：prompt 1K/10K/100K → KV **0.33/3.28/32.77 GB**、IB 传输 **6.6/65.5/655.4 ms**、prefill **354/3,539/35,389 ms**、**占比恒为 1.9%**；② **attention 位置**：若每 token 搬 KV → **152.6/15.3/1.5 tok/s**（**100K 时不可用**）；③ **供给比例**：1000:100 → **70.8:1**、1000:500 → **14.2:1**、10000:100 → **707.8:1**。

## 常见追问

- **追问**：为什么不用 PCIe 而用 IB？
  - 要点：**三条**：① **PCIe 约 25 GB/s、IB 约 50 GB/s**（**本仓库常数**）；② **比值仍是恒定的**（**只是从 1.9% 变成 3.8%**）；③ **所以 PCIe 也可接受**（**除非 KV 很大**）。**读法**：**"搬运占比是'链路带宽 / 算力'的函数"**——**两种链路都在可接受范围**。
- **追问**：如果 LPX 侧也要做一部分 attention 呢？
  - 要点：**三条**：① **KV 在 LPX，所以 LPX 做 attention 最自然**；② **GPU 可以做"prefill 阶段的 attention"**（**那时 KV 在生产**）；③ **混合切分会让 KV 分裂**（**管理复杂**）。**读法**：**"按'状态在哪'切分，而不是按'算力谁强'切分"**。
- **追问**：KV 传输能不能与 decode 重叠？
  - 要点：**能，但要小心**：① **首 token 必须等 KV 到齐**（**否则 attention 不完整**）；② **所以"重叠"只对"分块到达"有意义**（**如先到前 1K token 就开始**）；③ **代价是"分块 attention"的实现复杂度**。**读法**：**"分块到达 + 分块开始"是进阶优化**——**而基线是"等齐再开始"**。
- **追问**：如果客户的 prompt 很长呢？
  - 要点：**三条**：① **prefill 时间线性增长**（**本机 100K prompt 要 35.4 秒**）；② **所以 GPU 群要按"最长 prompt"配置**（**或做 chunked prefill**）；③ **而 KV 搬运占比不变**（**1.9%**）。**读法**：**"长 prompt 的瓶颈是 prefill 算力，不是搬运"**。
- **追问**：怎么测这条链路？
  - 要点：**三条**：① **分别测 prefill、传输、decode 三段**（**串 [[perplexity-05]] 的分解**）；② **测"端到端 TTFT 与 TPOT"**；③ **测"KV 传输的失败率与重试"**。**读法**：**"三段分别测"是定位问题的前提**——**而"端到端"只用于验证**。
- **追问**：这道题与"不建议迁移"有什么关系？
  - 要点：**它们是同一决策的两面**：① **本题讲"混合架构怎么设计"**；② **[[groq-11]] 讲"什么情况下不该迁"**；③ **两者都依赖"prompt/输出分布"这个输入**。**读法**：**"先问客户的分布，再谈架构"**。

## 相关题目

- [[groq-06]]：70B 的部署与经济性——**LPX 侧的容量与成本**。
- [[groq-09]]：静态 fabric 上的 MoE——**LPX 侧的调度**。
- [[groq-11]]：什么情况下不建议迁移——**同一决策的另一面**。
- [[nvidia-08]]：单卡服务 70B——**GPU 侧的 prefill**。
- [[perplexity-05]]：TTFT 恶化排查——**链路分解的方法**。

## 参考资料与归属

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（延伸）** —— Zhong et al.，2024-01-23：<https://arxiv.org/abs/2401.09670>。**prefill/decode 分离的动机与 goodput** 是本篇第 1、5 节的直接来源。
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving（延伸）** —— Qin et al. (Moonshot AI)，2024-06-28：<https://arxiv.org/abs/2407.00079>。**以 KV 为中心的解耦架构与 KV 的传输/复用** 是本篇第 2、6 节的依据。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。**prefill 计算受限、decode 带宽受限** 是本篇第 1 节的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（70B INT8、KV 320 KiB/token（本仓库常数）、PCIe 25 GB/s、IB 50 GB/s、峰值 989 TFLOPs、MFU 40%、prompt 1K/10K/100K、LPX 吞吐 20,000 tok/s）都是**按本仓库统一常数与本机公式计算的显式假设**；**KV 大小、传输时间、占比、供给比例都是直接计算**（**可复现**）。**⚠️ "20,000 tok/s"是示例吞吐**（**串 [[groq-06]]**）；**"MFU 40%"是理想值**；**"供给比例 70:1"依赖这两个假设**。**可迁移的结论是"一次性 KV 搬运占 1.9% 可接受、decode 的 attention 必须在 KV 所在侧、prefill 才是瓶颈"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
