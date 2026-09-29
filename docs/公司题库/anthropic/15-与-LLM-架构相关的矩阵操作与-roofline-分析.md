---
type: question
id: anthropic-15
company: Anthropic
topic: llm-internals
order: 15
question: 讲讲与 LLM 架构相关的矩阵操作。
question_en: Talk through the matrix operations involved in LLM architectures.
asked_at: []
level: 高阶
tags: [矩阵运算, GEMM, roofline, 算术强度, 量化]
sources:
  - title: Attention Is All You Need（延伸）
    url: https://arxiv.org/abs/1706.03762
    author: Vaswani et al. (NeurIPS 2017)
    published: 2017-06-12
  - title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）
    url: https://arxiv.org/abs/2205.14135
    author: Dao et al. (NeurIPS 2022)
    published: 2022-05-27
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [anthropic-13, anthropic-14, anthropic-16, inference-serving-01, inference-serving-03]
updated: 2026-09-28
---

## 一句话答案

> LLM 的计算可以归结为**四类算子**，判据是**算术强度**（每读一字节能做多少次浮点运算）：
> ① **投影类 GEMM**——QKV 投影、注意力输出投影、FFN 的 up/gate/down：形状是 $[B\!\cdot\!T,\,d]\times[d,\,d_{out}]$，**高算术强度、算力受限**；
> ② **注意力矩阵**——$QK^\top$ 与 $\text{softmax}\cdot V$：形状 $[B,h,T,d_h]\times[T,d_h]$，$O(T^2)$，**长序列时成为主导**；
> ③ **逐元素/归约**——softmax、RMSNorm、激活（SiLU）、残差相加、RoPE：**带宽受限**，几乎不贡献 FLOPs 但贡献大量内存访问；
> ④ **输出头**——$[B,T,d]\times[d,V]$，$V$ 可达 15 万，是单次最大的 GEMM 之一（但我们只对最后一个位置求 logits）。
> **同一个架构在两个阶段瓶颈完全不同**：prefill 时整段序列并行 → 大 GEMM → **算力受限**（$\approx2NP$ 口径）；decode 时每步只算 1 个 token → 退化成 **GEMV**（矩阵乘向量）→ 每步要读全部权重 → **带宽受限**。这就是为什么 decode 优化的第一手段是**增大 batch**（把权重读取摊薄）、第二是**量化**（减少字节数）、第三是把 KV 管好（PagedAttention 减少碎片与拷贝）。
> 一句话判据：**用算术强度决定「优化算力还是优化带宽」**——H100 的拐点在 $989/3.35\approx295$ FLOPs/byte（本仓库统一常数）。

## 面试官在考什么

- **能否把架构图翻译成矩阵形状**：$[B,T,d]$ 的每一步变换、参数量、FLOPs、以及内存访问量——这是「会不会算账」的基本功（串 [[anthropic-13]] 的参数量拆解）。
- **roofline 的直觉**：算术强度低于拐点 → 带宽受限（优化方向是减少字节：量化、融合、批处理）；高于拐点 → 算力受限（优化方向是提高利用率：更大 batch、更优 kernel、张量并行）。
- **prefill 与 decode 的分离讨论**：能否说出「为什么同一个模型在 prefill 阶段是算力题、在 decode 阶段是带宽题」，以及由此推出的两个不同优化栈（串 [[anthropic-16]]）。
- **注意力为什么会成为 IO 问题**：朴素实现会把 $T\times T$ 的分数矩阵写回 HBM（$T=32$K、$h=64$ 时是天文数字），FlashAttention 的思路是**分块 + 在线 softmax**，把中间结果留在片上（可引用的口径：FlashAttention 的核心贡献是 IO-aware 的精确注意力，而非近似）。
- **量化为什么在 decode 收益大**：带宽受限时性能几乎与字节数成反比——int8 让权重读取减半，理论上 decode 吞吐近 2×（算力受限的 prefill 收益小得多）。
- **KV cache 的访存模式**：为什么需要 PagedAttention（分页、按需分配、减少碎片与拷贝）、以及为什么前缀缓存（共享相同前缀的 KV 块）能省大量算力。
- **并行的接口在哪**：张量并行切的是 GEMM 的维度、流水并行切的是层、数据并行切的是 batch——每个都对应具体的矩阵分块方式（呼应「矩阵操作」这个题眼）。

**常见错误答案**

- 只列算子名（「有 matmul、有 softmax」）而不谈形状与瓶颈。
- 说「decode 慢是因为 FLOPs 多」——恰恰相反，decode 的 FLOPs 很少，慢在**每步读全部权重与 KV**。
- 认为 batch 增大总能提升吞吐——超过 KV 显存容量或进入算力受限区后就饱和（下表给拐点）。
- 把 FlashAttention 当成「近似注意力」——它是**精确**的，省的是 IO 不是计算量。

## 原理与推导

### 1. 四类算子的形状、FLOPs 与访存

以 $d=8192$、$d_{ff}=28672$、$h=64$、$d_h=128$、层数 $L=80$ 为例，处理 $N=B\cdot T$ 个 token：

| 算子 | 形状 | FLOPs（每层） | 读取字节（每层） | 算术强度 |
| --- | --- | --- | --- | --- |
| QKV 投影 | $[N,d]\times[d,3d]$（GQA 下 K/V 更小） | $2N\cdot d^2\cdot(1+2g/h)$ | $d^2(1+2g/h)\times2$ B（权重） | 高（∝N） |
| 注意力输出投影 | $[N,d]\times[d,d]$ | $2Nd^2$ | $2d^2$ B | 高（∝N） |
| FFN（SwiGLU） | $[N,d]\times[d,d_{ff}]$ ×3 | $6Nd\,d_{ff}$ | $6d\,d_{ff}$ B | 高（∝N） |
| 注意力分数+加权 | $[B,h,T,d_h]\times[T,d_h]$ | $4BhT^2d_h=4NTd$ | $O(Nd)$（分块后） | 中（∝T） |
| softmax/RMSNorm/SiLU/残差 | 逐元素 | $\sim c\cdot N d$ | $\sim 4Nd$ B 级 | **低（常数）** |
| 输出头 | $[N_{\text{last}},d]\times[d,V]$ | $2N_{\text{last}}dV$ | $2dV$ B | 取决于取几个位置 |

**关键观察**：投影与 FFN 的算术强度 $\propto N$——**batch/序列越大越「算力受限」**；逐元素与归一化的强度是常数——**永远是带宽受限**（这也是 kernel fusion 的收益来源：把多个逐元素操作合成一次读写）。

### 2. roofline：判断瓶颈在哪

$$\text{算术强度}\ I=\frac{\text{FLOPs}}{\text{访存字节}},\qquad I^{*}=\frac{\text{峰值算力}}{\text{峰值带宽}}=\frac{989\times10^{12}}{3.35\times10^{12}}\approx295\ \text{FLOPs/byte}$$

- $I<I^*$ → **带宽受限**，时间 $\approx \text{字节数}/\text{带宽}$；
- $I>I^*$ → **算力受限**，时间 $\approx \text{FLOPs}/\text{峰值算力}$。

**decode 的算术强度**：每生成 1 个 token 要读全部权重（70B 的 fp16 权重 ≈ 140 GB），FLOPs ≈ $2P=1.4\times10^{11}$：

$$I_{\text{decode}}=\frac{2P}{2P_{\text{bytes}}}=\frac{1.4\times10^{11}}{1.4\times10^{11}}=1\ \text{FLOP/byte}\ \ll\ 295$$

→ **极度带宽受限**（离拐点差两个数量级）。所以 decode 的每步时间下界就是「读一遍权重的时间」：$140\ \text{GB}/3.35\ \text{TB/s}\approx42$ ms（单卡口径；实际用张量并行把权重切到多卡，每卡只读自己那份）。

**加大 batch 为什么有效**：batch $B$ 时权重只读一次、FLOPs 变 $B$ 倍 → 强度变 $B$。$B=256$ 时 $I\approx256$，接近拐点；再往上就进入算力受限区（收益递减）。这解释了「decode 吞吐随 batch 先线性上升、后饱和」的曲线形状。

### 3. 注意力：为什么是 IO 问题

朴素注意力要物化 $[B,h,T,T]$ 的分数矩阵：$B=1,h=64,T=32768$、fp16 → $64\times32768^2\times2\ \text{B}=137\ \text{GB}$——**仅中间矩阵就远超显存**。FlashAttention 的做法是把 $Q,K,V$ 分块载入片上 SRAM，用**在线 softmax**（running max/sum）逐块累积，避免物化整个矩阵：**计算量不变（精确注意力），HBM 读写从 $O(T^2)$ 降到 $O(T^2/M)$（$M$ 为块大小）**。这直接说明「矩阵操作的优化常常是 IO 优化，而不是 FLOPs 优化」。

### 4. 量化与并行的矩阵视角

- **量化**：权重从 fp16 → int8 使读取字节减半，在带宽受限的 decode 上近似 2× 加速；在算力受限的 prefill 上收益小（且需要反量化开销）。**结论：量化主攻 decode**。
- **张量并行（TP）**：把 GEMM 的输出维（列并行）或输入维（行并行）切开，每卡算一部分再 all-reduce——注意**每层两次 all-reduce** 的通信量与 $B\cdot T\cdot d$ 成正比（decode 时 $T=1$，通信量小但延迟敏感）。
- **流水并行（PP）**：按层切分，中间激活在卡间传递（$B\cdot T\cdot d$ 字节/边界）。
- **数据并行（DP）**：batch 维切分，权重副本各自持有——decode 时最有效（各卡处理不同请求）。
**共同点**：每种并行都对应「切哪个矩阵维度」，并伴随一个通信量公式——能写出这个公式，才算真懂并行。

### 5. KV cache 的访存模式

decode 每步都要读该序列的全部 KV：$2L\cdot g d_h\cdot T$ 个元素（GQA）。$L=80,g=8,d_h=128,T=32\text{K}$、fp16 → $2\times80\times8\times128\times32768\times2\ \text{B}=34\ \text{GB}$/序列——**比权重还大**。这带来两个工程后果：① 必须分页管理（PagedAttention：块级分配、非连续存储、按需增长）；② 相同前缀的 KV 可以共享（前缀缓存/RadixAttention 式的块共享），在多轮对话里收益巨大。

## 数值与代码验证

### 表 1：四类算子的瓶颈判定（$d=8192$，H100 常数）

| 算子 | 每 token FLOPs | 每 token 字节（权重+激活） | 算术强度 | 瓶颈 |
| --- | --- | --- | --- | --- |
| QKV+O 投影 | $2d^2(1+2g/h)+2d^2\approx2.7\times10^{8}$ | $2.7\times10^{8}$ B 级 | ~1（decode） | 带宽 |
| FFN（SwiGLU） | $6d\,d_{ff}=1.4\times10^{9}$ | $1.4\times10^{9}$ B 级 | ~1（decode） | 带宽 |
| 注意力（$T$=8K） | $4Td=2.7\times10^{8}$ | $O(d)$ + KV 读取 | 随 $T$ 上升 | 中 |
| 逐元素/归一化 | $\sim10^{5}$ | $\sim10^{5}$ B | ~1 | 带宽（融合可救） |

**读法**：decode 阶段几乎所有算子都在「强度≈1」的带宽区——**这就是为什么 decode 的第一优化是 batch，第二是量化**，而不是换更快的算法。

### 表 2：batch 与算术强度（70B 量级、fp16、8×H100 张量并行，每卡读 17.5 GB）

| batch $B$ | 每步权重读取 | 每步 FLOPs | 算术强度 | 受限类型 | 每步时间下界 | 吞吐 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 140 GB（每卡 17.5 GB） | $1.4\times10^{11}$ | 1 | 带宽 | **5.22 ms** | 191 token/s |
| 32 | 同上 | $4.5\times10^{12}$ | 32 | 带宽 | **5.22 ms** | 6,126 token/s |
| 128 | 同上 | $1.8\times10^{13}$ | 128 | 带宽 | **5.22 ms** | 24,503 token/s |
| 256 | 同上 | $3.6\times10^{13}$ | 256 | 带宽（接近拐点 295） | **5.22 ms** | 49,006 token/s |
| 1,024 | 同上 | $1.4\times10^{14}$ | 1,024 | **算力** | 18.12 ms | 56,514 token/s |
| 2,048 | 同上 | $2.9\times10^{14}$ | 2,048 | **算力** | 36.24 ms | 56,514 token/s（**饱和**） |

**读法（三个关键结论）**：
1. 从 $B=1$ 到 $B=256$，**每步时间几乎不变（都是 5.22 ms）而吞吐涨了 256 倍**——这就是「100 个请求与 1 个请求耗时相同」的数学来源；
2. 拐点出现在 $B\approx295$（= roofline 拐点）附近：$B=1024$ 时强度过千，**每步时间开始随 $B$ 线性增长**，吞吐不再提升（本例停在 56.5k token/s）；
3. 因此 decode 的工程目标不是「batch 越大越好」，而是**在 KV 显存与 TPOT 约束下把 batch 顶到拐点附近**——这正是 continuous batching 与 chunked prefill 要解决的问题（串 [[anthropic-16]]）。

### 可运行代码

```python
# roofline 计算器：给定形状与硬件，判断瓶颈并给出时间下界
PEAK_FLOPS = 989e12        # H100 SXM5 bf16 dense（仓库统一常数）
PEAK_BW    = 3.35e12       # H100 HBM 带宽（仓库统一常数）
I_STAR = PEAK_FLOPS / PEAK_BW

def gemm(m, k, n, bytes_per=2, batch=1):
    """[batch,m,k] x [k,n] 的 FLOPs 与访存（读权重 + 读激活 + 写输出）"""
    flops = 2 * batch * m * k * n
    bytes_moved = bytes_per * (k * n + batch * m * k + batch * m * n)
    I = flops / bytes_moved
    t_bw = bytes_moved / PEAK_BW
    t_flops = flops / PEAK_FLOPS
    return flops, bytes_moved, I, max(t_bw, t_flops), ("带宽" if I < I_STAR else "算力")

print(f"H100 roofline 拐点 I* = {I_STAR:.0f} FLOPs/byte\n")

print("① decode 的单个 GEMM（每步 1 个 token，权重读取主导）")
for batch in (1, 8, 32, 128):
    f, b, I, t, bound = gemm(m=1, k=8192, n=8192, batch=batch)
    print(f"  batch={batch:>3}: FLOPs {f:>10.2e}  访存 {b/1e9:>7.2f} GB  "
          f"强度 {I:>7.2f}  {bound}受限  时间下界 {t*1000:>7.2f} ms")

print("\n② 70B 级整模型 decode（权重 140 GB，按 8 卡张量并行每卡 17.5 GB）")
P_BYTES = 140e9
for batch in (1, 8, 32, 128, 256, 1024, 2048):
    flops = 2 * 70e9 * batch
    I = flops / P_BYTES
    t = max(P_BYTES / (PEAK_BW * 8), flops / (PEAK_FLOPS * 8))   # 8 卡切权重
    print(f"  batch={batch:>5}: 强度 {I:>8.1f}  {'带宽' if I < I_STAR else '算力':<4}受限  "
          f"每步 {t*1000:>8.2f} ms  吞吐 {batch/t:>9.0f} token/s")

print("\n③ prefill（N 个 token 一次前向，2NP 口径）+ 注意力二次项")
def prefill(N, d=8192, d_ff=28672, L=80, g=8, h=64, d_head=128):
    proj = 2 * N * L * (d*d + 2*d*g*d_head + d*d)
    ffn = 2 * N * L * 3 * d * d_ff
    attn = 2 * N * N * L * d / 2          # QK^T 与 PV 合计的量级
    return proj, ffn, attn
for N in (1024, 8192, 32768):
    p, f, a = prefill(N)
    tot = p + f + a
    print(f"  N={N:>6,d}: 投影 {p/1e12:7.2f} TFLOPs   FFN {f/1e12:7.2f}  "
          f"注意力 {a/1e12:8.2f}  注意力占比 {a/tot:6.1%}  单卡时间下界 {tot/PEAK_FLOPS*1000:8.1f} ms")

print("\n④ 量化在两种阶段的收益（带宽减半 vs 算力不变）")
for label, I in (("decode（强度 1，带宽受限）", 1.0), ("prefill（强度 >1000，算力受限）", 2000.0)):
    speedup = 2.0 if I < I_STAR else 1.0
    print(f"  {label:<28} int8 理论加速 ≈ {speedup:.1f}x")

print("\n⑤ 朴素注意力 vs FlashAttention 的中间矩阵显存")
for T in (4096, 16384, 32768):
    B, h = 1, 64
    naive = B * h * T * T * 2
    print(f"  T={T:>6,d}: 朴素需物化 {naive/1024**3:8.2f} GiB 的分数矩阵"
          f"（分块后可降到 O(T) 级中间缓冲）")
print("读法：注意力优化的本质是 IO 优化（不物化 T×T 矩阵），FLOPs 并未减少 —— 这正是 FlashAttention 的定位")
```

预期输出要点（实跑）：H100 拐点 $I^*\approx295$；① decode 单 GEMM 从 batch=1 到 128，**强度从 1 涨到 128 而时间下界几乎不变**；② 70B 整模型 decode 在 batch≤256 时每步时间被权重读取锁定在 **5.22 ms**（每卡 17.5 GB ÷ 3.35 TB/s），batch 过拐点（约 295）后进入算力受限，每步时间线性增长而吞吐饱和在 56.5k token/s；③ prefill 的注意力占比随 $N$ 上升（1K 时 0.5%、8K 时 3.8%、32K 时 13.6%），FFN 始终是绝对量大头；④ int8 在 decode 上理论近 2×、在 prefill 上几乎无收益；⑤ 朴素注意力在 $T=32$K 时需要物化 128 GiB 的分数矩阵——这是 FlashAttention 存在的直接原因。

## 常见追问

- **追问**：为什么 decode 不直接用更大的 batch 把利用率打满？
  - 要点：三重约束——① KV 显存（每序列 $O(T)$）限制最大并发；② 进入算力受限区后每步时间线性增长，TTFT/TPOT 的 p99 变差；③ 请求到达不同步，等待凑批会增加排队延迟（这也是 continuous batching 的动机）。
- **追问**：张量并行与流水并行分别切哪个矩阵维度？
  - 要点：TP 切 GEMM 的权重维度（行/列并行，每层两次 all-reduce）；PP 切层（边界传激活）。TP 的通信量与激活大小成正比，因此**decode（$T=1$）时通信量小、PP 在 decode 时反而会出现气泡**。
- **追问**：量化会不会伤害质量？
  - 要点：要按权重/激活分别讨论（W8A8、W4A16 等配置），并用任务指标验证；工程上通常先在带宽受限的 decode 上收益最大（同样质量下更多并发），再评估 prefill 的收益。
- **追问**：前缀缓存为什么能省算力？
  - 要点：相同前缀的 KV 块可复用，prefill 阶段的注意力与投影计算可以直接跳过（只算新增 token）；在多轮对话/同一系统提示的场景命中率很高（串 [[inference-serving-05]]）。
- **追问**：怎么判断一个 kernel 是带宽还是算力受限？
  - 要点：算算术强度并与 $I^*$ 比；或实测——若把数据量减半（量化）耗时近似减半 → 带宽受限；若把 FLOPs 减半（稀疏/更小模型）耗时近似减半 → 算力受限。
- **追问**：MoE 对矩阵操作有什么影响？
  - 要点：把 FFN 的 $[N,d]\times[d,d_{ff}]$ 换成「按 token 路由到不同专家」的小 GEMM 集合——**每个专家的 batch 变小 → 算术强度下降 → 更容易带宽受限**；因此 MoE 的收益要靠足够大的总 batch（或 expert 并行 + all-to-all 通信）来兑现。

## 相关题目

- [[anthropic-13]]：Transformer 各组件的参数量拆解，是本题「哪些矩阵最大」的前提。
- [[anthropic-14]]：无 attention 架构的成本对照，与本题的 roofline 分析共用同一套常数。
- [[anthropic-16]]：batching 推理系统，直接建立在「decode 带宽受限、batch 摊薄权重」这一结论上。
- [[inference-serving-01]]：推理性能的指标体系与 roofline 直觉，可对照本题的计算口径。
- [[inference-serving-03]]：KV cache 与服务侧显存管理，对应本题第 5 节的访存模式。

## 参考资料与归属

- **Attention Is All You Need（延伸）** —— Vaswani et al. (NeurIPS 2017)，2017-06-12：<https://arxiv.org/abs/1706.03762>。第 1 节四类算子中投影、注意力与逐元素操作的原始形式来自这篇。
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（延伸）** —— Dao et al. (NeurIPS 2022)，2022-05-27：<https://arxiv.org/abs/2205.14135>。第 3 节「分块 + 在线 softmax、减少 HBM 读写而计算量不变」的做法与「精确注意力」的定位来自这篇。
- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 5 节 KV cache 分页管理的机制来自这篇。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 4 节「decode 是带宽受限、prefill 是算力受限，以及 batch 对算术强度的作用」的分析框架来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$d=8192$、$d_{ff}=28672$、$L=80$、GQA $g=8$、70B≈140 GB fp16、H100 989 TFLOPs / 3.35 TB/s、8 卡切分）都是按本仓库统一常数与假设配置推算的工程算例；真实实现的实测值会因 kernel、并行策略与显存布局而不同。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
