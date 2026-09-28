---
type: question
id: llm-internals-05
topic: LLM 内部原理与架构
order: 5
question: 解释 FlashAttention。它并没有减少 FLOPs，那为什么更快？
question_en: Explain FlashAttention. It does not reduce FLOPs, so why is it faster?
asked_at: [Together AI]
level: 进阶
tags: [flashattention, io-aware, online-softmax, kernel]
sources:
  - title: Decoding Flash Attention in LLMs
    url: https://outcomeschool.com/blog/decoding-flash-attention
    author: Amit Shekhar (Outcome School)
    published: 2026-04-11
  - title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）
    url: https://arxiv.org/abs/2205.14135
    author: Tri Dao et al.
    published: 2022-05-27
  - title: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning（延伸）
    url: https://arxiv.org/abs/2307.08691
    author: Tri Dao
    published: 2023-07-17
related: [llm-internals-01, llm-internals-02, inference-serving-01]
updated: 2026-09-28
---

## 一句话答案

> FlashAttention 一个 FLOPs 都没省：前向那两次矩阵乘一个不少，反向还要重算 $S$，整轮 fwd+bwd 的乘加量比标准实现还多约 17%。它快的原因是**把 attention 从「被显存带宽卡住」改成「几乎不搬数据」**——标准实现的瓶颈是那个 $N \times N$ 的分数矩阵要在 HBM 与 SRAM 之间来回搬 4 趟，FlashAttention 用 tiling 把它切进 SRAM、用 online softmax 在块内完成归一化，完整矩阵从不落 HBM。
> 量级：$N = 4000$、$d = 128$、bf16、单头单层时，HBM 流量从 66.0M 个元素（132.1 MB）降到 17.8M 个元素（35.6 MB），少 3.7 倍。流量降下来，GPU 才有机会把 tensor core 喂饱，论文实测端到端 2–4× 加速。

## 面试官在考什么

- **是否知道瓶颈在哪一侧。** 能把 attention 放进 roofline 框架，说出「H100 的拐点是约 295 FLOP/byte，而 attention 的算术强度只有约 62 FLOP/byte，不到拐点的四分之一，所以它在带宽一侧」。只答「FlashAttention 用了分块」不足以过。
- **是否理解 online softmax 是精确的。** 面试官会追问「分块算 softmax 是不是近似」，正确回答是「不是，数学上恒等，误差只来自浮点舍入」，并能写出 running max / running sum 的递推和 rescale 因子 $e^{m_{\text{old}} - m_{\text{new}}}$。
- **是否分得清「算法复杂度」与「IO 复杂度」。** FLOPs 仍然是 $O(N^2 d)$，HBM 访问从 $O(N^2)$ 降到 $O(N^2 d^2 / M)$，$M$ 是 SRAM 大小；训练时连 $O(N^2)$ 的 activation 都不存，显存从 $O(N^2)$ 降到 $O(N)$。
- **是否知道它和 KV cache 优化的关系。** PagedAttention / vLLM（[[inference-serving-01]]）管的是 KV cache 的分页显存，FlashAttention 管的是单次 attention 算子内部的 HBM 流量，两者互补。能说清这一点的人通常真在生产环境调过推理。
- **能否落到工程细节。** 块大小怎么选（受 SRAM 容量约束）、为什么 GPU 上「非 matmul 的 FLOPs」很贵（tensor core 只对 matmul 快）、反向为什么要重算而不是存下来。

常见错误答案：

- 「FlashAttention 减少了 attention 的计算量 / 用了稀疏化。」它是精确注意力，FLOPs 没减，前向的 matmul 一个不少，反向还多做了重算。
- 「因为分块所以快，块能塞进 cache。」只说对了一半。如果分块之后仍然把 $S$、$P$ 写回 HBM，再读出来做下一步，速度不会变——关键在于**同一块数据在 SRAM 里被连续消费掉，永远不落 HBM**。
- 「它省显存，所以更快。」省显存和变快是两个并列结果，不是因果关系；因果链是少搬数据。

## 原理与推导

### 1. 标准 attention 的 HBM 账本

记 $S = QK^\top \in \mathbb{R}^{N \times N}$，$P = \mathrm{softmax}(S)$，$O = PV$。朴素实现（以及早期 PyTorch/`einsum` 逐算子实现）把 $S$ 和 $P$ 当成普通 tensor，于是每次 attention 调用至少要在 HBM 和 SRAM 之间走这几趟：

1. 读 $Q, K$，算 $S = QK^\top$，**把 $S$ 写回 HBM**；
2. 把 $S$ **读回 SRAM** 做 softmax，**把 $P$ 写回 HBM**；
3. 把 $P$ **读回 SRAM**，配合 $V$ 算 $PV$，把 $O$ 写回 HBM。

用 $N = 4000$、$d = 128$、bf16（2 字节）、单头单层来记账：

| 动作 | 元素个数 | 流量 |
| --- | --- | --- |
| 读 Q、K | $2Nd = 1{,}024{,}000$ | 2.05 MB |
| 写 S | $N^2 = 16{,}000{,}000$ | 32.0 MB |
| 读 S（softmax） | $16{,}000{,}000$ | 32.0 MB |
| 写 P | $16{,}000{,}000$ | 32.0 MB |
| 读 P（与 V 相乘） | $16{,}000{,}000$ | 32.0 MB |
| 读 V、写 O | $2Nd = 1{,}024{,}000$ | 2.05 MB |
| 合计 | $66{,}048{,}000$ | **132.1 MB** |

四项 $N^2$ 搬运合计 128 MB，占全部流量的约 97%；而整个 attention 的矩阵乘只有 $4N^2d = 8.19$ GFLOP（单头单层），算术强度约 62 FLOP/byte，远低于下一节要算的 295 FLOP/byte 拐点。搬得多、算得少，这才是「IO-aware」这个词的由来：写 kernel 时必须把「数据在哪一层内存」当成一等公民。

源文用的是更粗的口径：标准实现只数那个 $N \times N$ 矩阵的四趟搬运，得到 $4N^2 = 64{,}000{,}000$ 个元素（约 64M）；FlashAttention 一侧把 $Q$ 块、$K/V$ 块与输出逐项相加，得到约 17M 个元素（精确是 $17{,}801{,}216$），源文概括为「少约 4 倍」。按逐项列全的口径复算，两边分别是 66.0M 与 17.8M，比值 3.71——差别只在 $Q,K,V,O$ 的 $Nd$ 这些小项上，结论一致。**下面统一用逐项列全的口径**，因为它能顺带说明 $S$、$P$ 的显存占用。

### 2. 为什么这块流量会要命：算术强度

把 $N = 4000$、$d = 128$ 的整个 attention 拿出来算：

- FLOPs $= 2N^2d\ (\text{for } QK^\top) + 2N^2d\ (\text{for } PV) = 4N^2d = 8.19$ GFLOP（单头）；
- 字节数 $= 2 \times (4N^2 + 4Nd) = 132.1$ MB（单头）；
- 算术强度 $= 8.19 \times 10^9 \big/ (132.1 \times 10^6) \approx 62.0$ FLOP/byte。

H100 SXM 的参数（官方规格，注意稀疏标志）：bf16 dense 989 TFLOP/s，HBM3 带宽 3.35 TB/s，显存 80 GB，每个 SM 的 SRAM 约 228 KB（Hopper 上共享内存与 L1 共用同一块物理存储，可配置给共享内存的上限是 227 KB）。

$$\text{拐点} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295\ \text{FLOP/byte}$$

$62.0 \ll 295$，attention 稳稳落在**带宽一侧**：这 8.19 GFLOP 在峰值算力下只需 $8.3\ \mu s$，而 132.1 MB 在峰值带宽下要 $39.4\ \mu s$。把 FLOPs 砍掉一半，时间几乎不变——总时间仍然由 39.4 $\mu s$ 的带宽下限支配。这解释了为什么那么多「近似 attention」在理论上省了计算却在墙钟时间上毫无收益，也解释了为什么 FlashAttention 明明没省 FLOPs 却能快好几倍。

反过来也能算出 FlashAttention 的天花板：一块 $256 \times 256$、$d = 128$ 的 bf16 tile，$4 \times 256^2 \times 128 = 33.6$ MFLOP，装入 $Q, K, V$ 三块共 192 KiB（每块 $256 \times 128$、64 KiB），算术强度 $170$ FLOP/byte——**仍然低于 295**。FlashAttention 并不是把 attention 变成了计算密集型算子，它只是把常数压下来并让搬运与计算重叠；这也是它只能到理论峰值 50–73%（FlashAttention-2 论文口径，A100 上）而不是 GEMM 水平的原因。

### 3. tiling：把「列」当内层循环

把 $Q$ 按行切成 $T_r = \lceil N/B_r \rceil$ 块，$K, V$ 按行切成 $T_c = \lceil N/B_c \rceil$ 块，每块能放进 SRAM。FlashAttention-1 的原始算法把 $K/V$ 块放在外层、$Q$ 块放在内层，沿列方向归约；FlashAttention-2 为了让不同 $Q$ 块之间不必共享统计量、从而能在序列维上并行，把循环顺序换成了 $Q$ 块在外层、$K/V$ 块在内层。下面与源文都采用后一种顺序：

```text
for each Q block Q_i (外层, 可并行):
    O_i = 0 ; m_i = -inf ; l_i = 0        # 每行的 running 统计量
    for each K/V block (K_j, V_j) (内层):
        S_ij = Q_i @ K_j^T                # 只在 SRAM 里, 不落 HBM
        m_new = max(m_i, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)
        corr  = exp(m_i - m_new)
        l_i   = l_i * corr + rowsum(P_ij)
        O_i   = O_i * corr + P_ij @ V_j
        m_i   = m_new
    O_i = O_i / l_i                       # 最后一次性归一化
    write O_i to HBM
```

两个 trick 值得单独强调：

- **块内消费**。$S_{ij}$ 在 SRAM 里算出来之后立刻喂给下一次 matmul；bf16 下 $256 \times 256$ 的 tile 是 128 KiB，生命期只有一块 tile 那么大，从不产生 $N \times N$ 的 HBM 写。
- **online softmax 的 rescale**。第 $j$ 块处理时可能发现一个更大的行最大值，之前用旧 $m$ 累积的 $l_i$ 和 $O_i$ 必须整体缩放 $e^{m_{\text{old}} - m_{\text{new}}}$，才能保证它们相对新基准是「正确的未归一化量」。

### 4. online softmax 的递推与精确性

对第 $i$ 行的分块序列 $s^{(1)}, s^{(2)}, \dots, s^{(T_c)}$，维护

$$m^{(j)} = \max\bigl(m^{(j-1)},\ \max(s^{(j)})\bigr), \qquad l^{(j)} = e^{\,m^{(j-1)} - m^{(j)}}\, l^{(j-1)} + \sum_{t \in \text{block } j} e^{\,s_t - m^{(j)}}$$

$$\tilde{O}^{(j)} = e^{\,m^{(j-1)} - m^{(j)}}\, \tilde{O}^{(j-1)} + \sum_{t \in \text{block } j} e^{\,s_t - m^{(j)}}\, v_t, \qquad O = \tilde{O}^{(T_c)} \,/\, l^{(T_c)}$$

一次性版本把每一项写成 $e^{s_t - m} \big/ \sum_{t'} e^{s_{t'} - m}$，其中 $m = \max_t s_t$。分块版本维护的不变量正是这一形式的部分和：

$$\tilde{O}^{(j)} \equiv \sum_{t \le j} e^{\,s_t - m^{(j)}} v_t, \qquad l^{(j)} \equiv \sum_{t \le j} e^{\,s_t - m^{(j)}}$$

当 $m^{(j)}$ 从 $m_{\text{old}}$ 涨到 $m_{\text{new}}$ 时，把旧累加量乘 $e^{m_{\text{old}} - m_{\text{new}}}$ 恰好把基准从 $m_{\text{old}}$ 换到 $m_{\text{new}}$：

$$e^{\,s_t - m_{\text{old}}} \cdot e^{\,m_{\text{old}} - m_{\text{new}}} = e^{\,s_t - m_{\text{new}}}$$

所以递推保持不变量的定义，全部块处理完后 $\tilde{O} / l = \sum_t e^{s_t - m} v_t \big/ \sum_t e^{s_t - m}$，与一次性 softmax 逐元素相等。**没有任何近似**：误差只来自浮点加法顺序不同，结果可能在最后几位不同；PyTorch 的 `scaled_dot_product_attention` 同样不保证与朴素实现 bitwise 相同，掩码或不同 backend 会走不同 kernel，差异还会更大。

rescale 也让实现天然数值安全：指数里的数永远 $\le 0$，不会溢出。

### 5. 块大小怎么定

SRAM 每 SM 约 228 KB，单个 block 最多能用 227 KB，扣掉双缓冲和流水线占用的部分，留给 $Q/K/V$ 三块 tile 的预算在 100–200 KB 量级。以 $d = 128$、bf16 为例，一块 $256 \times 128$ 的 $Q$ 块是 64 KiB，$K_j$、$V_j$ 各 64 KiB，再算上 $S_{ij}$ 的寄存器占用和双缓冲，就是块大小不能再往上翻的原因。源文取 $B = 256$ 做例子，这个值正好落在「一个 SM 装得下、又足够大到让 matmul 摊薄固定开销」的区间。

### 6. 反向传播：用重算换流量

朴素反向要拿到 $P$，而 FlashAttention 的前向根本没存 $P$。它的做法是：

- 前向只额外保存每行的 $m$ 和 $l$（以及可选的 logsumexp），共 $O(N)$ 而非 $O(N^2)$；
- 反向重新载入 $Q, K, V$ 块，用保存的 $m, l$ 在 SRAM 里重建 $S_{ij}$、$P_{ij}$，再走一遍链式法则。

代价是明确的：单头单层、$N = 4000$、$d = 128$ 时，重算 $S$ 需要一次额外的 $2N^2d = 4.10$ GFLOP。前向是两次 $2N^2d$ 量级的 matmul（$QK^\top$ 与 $PV$，4.10 + 4.10 = 8.19 GFLOP）；标准反向有四次同量级的 matmul（$dV = P^\top dO$、$dP = dO\,V^\top$、$dQ = dS\,K$、$dK = dS^\top Q$，约 16.4 GFLOP，另有 elementwise 的 $dS$），也就是反向约 2× 前向；加了重算之后反向约 20.5 GFLOP，变成约 2.5× 前向。用这多出来的一步，省掉的是每头每层重新读回 $S$ 与 $P$ 的 $4N^2$ 个元素（128 MB）。

### 7. IO 复杂度与显存

标准实现每头的 HBM 访问是 $\Omega(N^2 + Nd)$ 个元素，FlashAttention 是 $O(N^2 d^2 / M)$ 个元素（$M$ 为 SRAM 容量），论文证明后者对相当宽的一段 SRAM 容量是**最优**的。注意这个界里仍然带 $N^2$——tiling 消除的是那个张量在 HBM 里的「物化」，不是 $N^2$ 这个规模本身；$K/V$ 块会被每个 $Q$ 块重复读一遍，这就是代价所在。

显存一侧的变化更直观：$S$ 和 $P$ 的 activation 从 $O(N^2)$ 降到 $O(N)$。$N = 4000$、单头单层时 $S$ 与 $P$ 各 32 MB；单样本 32 层 32 头的模型（bf16）如果把两者都存下来是 65.5 GB，即便只存反向所需的 $P$ 也要 32.8 GB。FlashAttention 额外保存的只有每行的 $m, l$（float32 下每头每层 $2N$ 个数，32 层 32 头共约 33 MB），输出本来就要写回 HBM。

## 数值与代码验证

### 复算：$N = 4000$、$d = 128$、$B = 256$、bf16、单头单层

| 步骤 | 标准 attention | FlashAttention |
| --- | --- | --- |
| 读 Q | 512,000 元素 / 1.02 MB | 512,000 元素 / 1.02 MB |
| 读 K | 512,000 元素 / 1.02 MB | 8,388,608 元素 / 16.78 MB（16 个 Q 块各读 1 遍） |
| 读 V | 512,000 元素 / 1.02 MB | 8,388,608 元素 / 16.78 MB（同上） |
| $S$、$P$ 相关 | 64,000,000 元素 / 128.0 MB | 0（从不物化） |
| 写 O | 512,000 元素 / 1.02 MB | 512,000 元素 / 1.02 MB |
| 合计 | 66,048,000 元素 / **132.1 MB** | 17,801,216 元素 / **35.6 MB** |

比值 $132.1 / 35.6 = 3.71$。源文按「只数 $N^2$ 矩阵」的口径给出约 64M 对约 17M（比值 3.6），并且明确写了块数是 $4000 / 256 = 16$；逐项列全之后是 66.0M 对 17.8M，比值 3.7，两者结论一致。注意 FlashAttention 这一侧 94% 的流量是 $K/V$ 的重复读：$16 \times 16 \times 32{,}768 \times 2 = 16{,}777{,}216$ 个元素，因为块数 $T = \lceil N / B \rceil = 16$，每个 $Q$ 块都要把全部 $K, V$ 重读一遍，流量正比于 $T^2$。

规模外推（逐项列全口径，单头单层，bf16）：

| $N$ | 标准流量 | FlashAttention 流量 | 比值 | $S$ 矩阵本身 |
| --- | --- | --- | --- | --- |
| 1024 | 9.4 MB | 2.6 MB | 3.6× | 2.1 MB |
| 2048 | 35.7 MB | 9.4 MB | 3.8× | 8.4 MB |
| 4000 | 132.1 MB | 35.6 MB | 3.7× | 32.0 MB |
| 8192 | 545.3 MB | 138.4 MB | 3.9× | 134.2 MB |
| 32768 | 8.6 GB | 2.2 GB | 4.0× | 2.1 GB |

比值从 $N = 1024$ 的 3.6 倍缓慢升到 $N$ 上千之后的 4 倍附近并基本走平（$N = 4000$ 处是 3.7，因为 $T = 16$ 块里最后一块只装了 160 行，多读了约 2.3% 的 $K/V$），因为两边的流量都被 $N^2$ 项主导。**这说明约 4 倍是 HBM 流量的差距，不是墙钟时间的差距**：3.35 TB/s 下这 132.1 MB 对 35.6 MB 只对应 $39.4\ \mu s$ 与 $10.6\ \mu s$ 的带宽时间下限，而真实 kernel 还要受占用率、tile 启动开销和 matmul 本身限制。论文实测端到端 3×（GPT-2，序列 1K）与 2.4×（Long Range Arena，1K–4K），与「流量比约 4 倍、但只能兑现一部分」完全自洽——这正是 FlashAttention-2 要接着解决的问题。

### 代码：online softmax 与一次性 softmax 等价

```python
import torch

torch.manual_seed(0)
N, D, B = 4000, 128, 256
q, k, v = (torch.randn(N, D) for _ in range(3))

def naive(q, k, v):
    s = q @ k.T / D ** 0.5                      # N x N，显式物化
    p = torch.softmax(s, dim=-1)
    return p @ v, s

def flash_forward(q, k, v, block=256):
    """纯 PyTorch 模拟 FlashAttention 前向：外层 Q 块，内层 K/V 块。"""
    N, D = q.shape
    o = torch.empty_like(q)
    for i in range(0, N, block):
        qi = q[i:i + block]
        m = torch.full((qi.shape[0], 1), float('-inf'))   # running max
        l = torch.zeros((qi.shape[0], 1))                  # running sum
        acc = torch.zeros_like(qi)                         # 未归一化输出
        for j in range(0, N, block):
            kj, vj = k[j:j + block], v[j:j + block]
            s = qi @ kj.T / D ** 0.5                       # 只活在 SRAM 语义下
            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            corr = torch.exp(m - m_new)                    # m 变了要回缩历史
            p = torch.exp(s - m_new)
            l = l * corr + p.sum(dim=-1, keepdim=True)
            acc = acc * corr + p @ vj
            m = m_new
        o[i:i + block] = acc / l
    return o

out_naive, s = naive(q, k, v)
out_flash = flash_forward(q, k, v, block=256)
print('max |diff| =', (out_naive - out_flash).abs().max().item())
print('N x N 矩阵占用 = %.1f MiB, tile 占用 = %.1f KiB'
      % (s.numel() * 4 / 2 ** 20, 3 * 256 * D * 4 / 1024))
```

实测输出 `max |diff|` 在 $10^{-7}$ 量级（float32 下的舍入量级，不是近似误差），同时能看到朴素路径凭空多出一个 61.0 MiB 的 $N \times N$ 中间矩阵，而 FlashAttention 路径的 tile 只有 384 KiB（float32；bf16 则减半）——一个要写回 HBM 再读回来，另一个全程留在片上。

手算一个最小例子交叉验证：一行分数 $[2, 4, 6]$，切成两块 $[2,4]$ 与 $[6]$。

| 步骤 | $m$ | $l$ | 说明 |
| --- | --- | --- | --- |
| 初始 | $-\infty$ | 0 | — |
| 处理 $[2,4]$ | 4 | $e^{-2} + e^{0} = 1.135335$ | — |
| 进入 $[6]$ | 6 | 先回缩 $1.135335 \times e^{4-6} = 0.153651$，再加 $e^{0}$ | 回缩因子 $e^{m_{\text{old}} - m_{\text{new}}}$ |
| 结束 | 6 | 1.153651 | 与一次性 $e^{-4} + e^{-2} + e^{0}$ 相等 |

$l$ 两边都是 $1.153650922$（实数意义下严格相等）；最终概率 $[0.015876, 0.117310, 0.866813]$，与直接 softmax 逐元素一致。

## 常见追问

- **追问**：FlashAttention 是近似算法吗？会不会掉点？
  - 要点：不是近似。它是同一个数学表达式的重排，输出与标准 attention 相同（差异仅在浮点舍入），因此不存在精度与效果的 trade-off。近似的那一支走的是另一条路：block-sparse FlashAttention、Longformer/BigBird 这类稀疏模式，或者 Linformer 这类低秩近似，它们都在改数学式。
- **追问**：既然 FLOPs 没变甚至变多，为什么说它「便宜」？
  - 要点：GPU 上「贵」的是 HBM 访存。H100 的拐点是约 295 FLOP/byte，attention 只有约 62 FLOP/byte，处在带宽一侧，所以省 3.7 倍流量比省几倍 FLOPs 有效得多。反向重算多花 4.10 GFLOP，相对标准反向的 16.4 GFLOP 是 +25%（相对整个 fwd+bwd 是 +17%），换来的是少读 128 MB，在带宽一侧的算子里是划算的。
- **追问**：块大小是不是越大越好？
  - 要点：上界由 SRAM 容量决定（H100 每 SM 约 228 KB）。$d = 128$、bf16 时 $B = 256$ 已经让 $Q, K, V$ 三块占 192 KiB。块太小则 matmul 摊不薄固定开销、$K/V$ 重复读次数增加（流量正比于 $T_r \times T_c = \lceil N/B_r \rceil \times \lceil N/B_c \rceil$）；块太大则塞不下、双缓冲失效。
- **追问**：FlashAttention 和 PagedAttention / vLLM（[[inference-serving-01]]）是什么关系？会不会互相取代？
  - 要点：管的是两件事。FlashAttention 是 attention 算子内部的计算与访存调度，处理 $QK^\top$ 这个 $N^2$ 中间量；PagedAttention 是 KV cache 的显存分配与共享，处理 decode 阶段随请求增长的 cache。一个是算子的 IO，一个是显存管理器，工程上叠在一起用。
- **追问**：为什么 FlashAttention-2 能再快约 2 倍？
  - 要点：三条——(1) 减少非 matmul 的 FLOPs，把 rescale 从内层每次迭代推迟到外层块结束再做一次，因为 tensor core 只加速 matmul，`exp` 和乘法的开销独立于 matmul 存在；(2) 把并行维度从「batch × head」扩展到「batch × head × Q 块」，长序列小 batch 时也能占满所有 SM；(3) 块内重新划分 warp 的工作，减少共享内存读写。论文报告从峰值 25–40% 提到 50–73%，A100 上 GPT 风格模型训练可达 225 TFLOP/s（72% MFU）。
- **追问**：什么情况下 FlashAttention 帮助不大？
  - 要点：短序列（$N$ 只有几百）时 $N^2$ 项本来就小，带宽瓶颈不明显；decode 阶段单 token 的 attention 是 $1 \times N$ 与 KV cache 的乘积，瓶颈变成读 KV cache 的带宽，那是一块完全不同的优化战场（分页、量化、GQA）。此外反向的收益比前向小，因为要重算 $S$；训练时的收益主要来自 activation 显存下降，从而允许更大的 batch 或更长的序列。

## 公司变体

`asked_at` 里目前是 **Together AI**。Together AI 的核心业务是 GPU 集群上的训练与推理服务，公开的工程博客与开源工作集中在 kernel、CUDA 优化与推理吞吐上，所以这家公司问这题时通常会往**工程实现**方向压：块大小与 SRAM 预算怎么权衡、tiling 的循环顺序为什么在 FlashAttention-2 里换成 Q 在外、反向重算的成本怎么估、上线时用的是哪个版本、以及它和 PagedAttention 在同一个 serving 栈里怎么分工。准备时要能报出具体的量级——228 KB 每 SM、`B = 256` 时三块 192 KiB、流量从 132 MB 降到 36 MB 这个量级——而不只是复述概念。

同一个问题在别处的问法会偏原理：从 $S = QK^\top$ 的访存次数出发推导 IO 复杂度，或者要求现场写出 online softmax 的递推并证明它与一次性 softmax 等价。两种问法都得准备，因为这题的区分点恰恰是「能算清楚」而不是「听说过」。

## 相关题目

- [[llm-internals-01]]：scaled dot-product attention 与 $1/\sqrt{d_k}$，是本题的数学前提。
- [[llm-internals-02]]：KV cache 的显存公式；decode 阶段的瓶颈与本题的 prefill 瓶颈正好互补。
- [[inference-serving-01]]：PagedAttention / vLLM 的显存管理，与 FlashAttention 在同一套 serving 流程里分工。

## 参考资料与归属

- [Decoding Flash Attention in LLMs](https://outcomeschool.com/blog/decoding-flash-attention)，Amit Shekhar（Outcome School），2026-04-11：tiling、online softmax、反向重算，以及「$B = 256$、约 64M 对约 17M 个元素」的粗口径记账方式。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré，2022-05-27（延伸）：IO-aware 的动机、IO 复杂度与最优性、实测 3×（GPT-2，序列 1K）与 2.4×（Long Range Arena，1K–4K）。
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)，Tri Dao，2023-07-17（延伸）：非 matmul FLOPs、序列维并行、warp 级工作划分，峰值占比 25–40% 提升到 50–73%，A100 上 225 TFLOP/s（72% MFU）。
- H100 硬件参数取自 NVIDIA H100 官方规格（bf16 dense 989 TFLOP/s、HBM3 3.35 TB/s、80 GB），SRAM 每 SM 约 228 KB 取自 Hopper 架构文档；拐点 295 FLOP/byte 与各项流量、比值均按正文声明的口径复算。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
