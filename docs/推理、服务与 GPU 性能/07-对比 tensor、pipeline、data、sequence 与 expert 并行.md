---
type: question
id: inference-serving-07
topic: 推理、服务与 GPU 性能
order: 7
question: 对比 tensor、pipeline、data、sequence 和 expert 并行。什么时候需要组合使用它们？
question_en: Compare tensor, pipeline, data, sequence and expert parallelism. When do you combine them?
asked_at: [Google DeepMind, Meta, Amazon, NVIDIA]
level: 高阶
tags: [并行策略, tensor-parallel, pipeline-parallel, expert-parallel]
sources:
  - title: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism（延伸）
    url: https://arxiv.org/abs/1909.08053
    author: Shoeybi et al.
    published: 2019-09-17
  - title: GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism（延伸）
    url: https://arxiv.org/abs/1811.06965
    author: Huang et al.
    published: 2018-11-16
  - title: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（延伸）
    url: https://arxiv.org/abs/1910.02054
    author: Rajbhandari et al.
    published: 2019-10-04
  - title: Reducing Activation Recomputation in Large Transformer Models（延伸）
    url: https://arxiv.org/abs/2205.05198
    author: Korthikanti et al. (Megatron sequence parallelism)
    published: 2022-05-10
  - title: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding（延伸）
    url: https://arxiv.org/abs/2006.16668
    author: Lepikhin et al. (expert parallelism)
    published: 2020-06-30
related: [inference-serving-08, inference-serving-11, llm-internals-10]
updated: 2026-09-28
---

## 一句话答案

> 五种并行是同一个问题的五种切法：TP 切层内权重矩阵，每层前向两次 all-reduce、反向再两次，通信最密，所以必须留在 NVLink 域内；PP 切层，只在 stage 边界传点对点激活，通信最少但有气泡；DP 切数据，每步做梯度 all-reduce，通信量正比于参数量，是万卡规模的基础层；SP 切序列维并复用 TP 通信组，把 LayerNorm/dropout 那些本来被复制的激活也按序列切开，激活显存约降到 $1/t$；EP 切专家，每层两次 all-to-all，通信量随 token 数增长且负载天然不均。
> 组合判据只有两条：显存决定要不要切，带宽决定在哪里切。先用 TP(+SP) 把单卡放不下的权重和激活压进单个 NVLink 域（$t \le 8$），再用 PP/DP/EP 把规模推过单机边界，并始终满足 总卡数 $= t \times p \times d \times e$。
> 最后用实测 MFU 和通信占比收口，公式只用来排除明显不可行的配置。

## 面试官在考什么

- 能不能用同一套记账法讲清五种并行：**切什么维度、通信发生在哪里、通信量随什么增长**，而不是各背一段定义。
- 知不知道每种并行的物理约束在哪里：TP 的可行性由「算力带宽比 ÷ 互联带宽」决定，PP 由气泡与 micro-batch 数决定，DP 由参数量与每步 token 数决定，EP 由 token 数与路由均衡决定。
- 能不能把「显存账」算成 bytes/param，并区分优化器状态、梯度、参数、激活四类占用谁被谁切。
- 能不能给出可执行的组合顺序（显存 → 机内 TP/SP → 跨机 PP/DP/EP → 并行度乘积 → 实测），而不是罗列名词。
- 是否知道并行度不是越大越好：分得越细，单卡 GEMM 越小、collective 次数越多、固定延迟占比越高。

常见错误答案：

- 「TP 度越大显存越省所以越大越好」。t 一旦跨出 NVLink 域，单向带宽从 450 GB/s 掉到 NDR IB 的 50 GB/s，同一份 all-reduce 的时间乘 9，再省显存也没有意义。
- 「DP 不需要通信」。70B 的 bf16 梯度是 140 GB，ring all-reduce 下每卡每步收发约 280 GB，在 400 Gb/s 链路上串行要 5.6 s，必须切片、分桶并与 backward 重叠。
- 混用气泡公式口径：GPipe 的 fill-drain 调度是 $(p-1)/(m+p-1)$，1F1B 是 $(p-1)/m$，两者对应的激活显存也不同（$O(m)$ 对 $O(p)$），不能互相代入。

## 原理与推导

### 记号

| 符号 | 含义 | 符号 | 含义 |
| --- | --- | --- | --- |
| $t$ | tensor parallel 度 | $c$ | 每个元素的字节数（bf16 = 2） |
| $p$ | pipeline parallel 度 | $N$ | 参数量（70B = 70e9） |
| $d$ | data parallel 度 | $L$ | 层数 |
| $e$ | expert parallel 度 | $h$ | hidden size |
| $s$ | 序列长度（micro-batch 内） | $a$ | attention 头数 |
| $b$ | micro-batch size | $k$ | 每个 token 激活的专家数（top-k） |
| $m$ | 一次迭代的 micro-batch 个数 | $E$ | 每层专家总数 |

### 一张表回答三个问题

| 策略 | 切什么 | 每次前向/反向的通信算子 | 每卡通信量 | 随什么增长 |
| --- | --- | --- | --- | --- |
| TP | 层内权重矩阵（列/行） | 每层 2 次 all-reduce（前向）+ 2 次（反向） | $8\frac{t-1}{t} b s h c$ 每层 | 随 $b,s,h$ 线性，随 $t$ 缓慢饱和 |
| SP | 序列维（非 TP 区） | 每层 all-gather + reduce-scatter | 与 TP 相同 | 与 TP 相同 |
| PP | 按层切 stage | stage 边界点对点传激活 | $b s h c$ 每次传递（单向） | 随 micro-batch 数与边界激活 |
| DP | 数据（batch） | 梯度 all-reduce（= RS + AG） | $2\frac{d-1}{d} N c$ 每步 | 随参数量，不随 batch |
| EP | 专家 | 每层 dispatch + combine 两次 all-to-all | $2k\frac{E-1}{E} h c$ 每 token 每层 | 随 token 数（动态） |

这张表就是答案的主干。TP 与 SP 的通信量几乎一样，差别在**激活显存**；DP 与 EP 都随模型容量增长，差别在 DP 的通信量每步固定、EP 的随 token 数走；PP 的通信只发生在 stage 边界，每个 micro-batch 每段传一次边界激活，与层数无关。

### TP：列并行加行并行，每层两次 all-reduce

MLP 的两层写成 $Y=\mathrm{GeLU}(XA)$、$Z=YB$。把 $A$ 按**列**切成 $[A_1,\dots,A_t]$，每张卡算 $Y_i=\mathrm{GeLU}(XA_i)$，输入 $X$ 被复制，因此第一次 GEMM 不需要通信；把 $B$ 按**行**切成 $[B_1;\dots;B_t]$，每张卡算 $Z_i=Y_iB_i$，而 $Z=\sum_i Z_i$，于是第二次 GEMM 之后需要一次 all-reduce。attention 同理：QKV 投影按头切（每张卡算一部分头），output projection 按行切，后面接一次 all-reduce。所以每个 transformer 层的前向恰好两次 all-reduce，反向再两次——这与 sequence parallelism 论文里的说法一致（tensor parallelism 在单个前向加反向中共 4 次 all-reduce）。

单次 all-reduce 的张量是 $b\times s\times h$，ring 算法下每卡收发 $2\frac{t-1}{t}bshc$ 字节（NCCL 报 bus bandwidth 时用的就是这个「算法通信量」口径）；一个层前向两次、反向两次，于是每层每卡：

$$B_{\text{TP, layer}} = 8\frac{t-1}{t}\,b\,s\,h\,c \qquad\text{每 token 每层} = 8\frac{t-1}{t}hc$$

通信量随 $b,s,h$ 线性增长，随 $t$ 只缓慢增长并饱和（$t=8$ 时 $(t-1)/t = 0.875$，$t\to\infty$ 也只到 1）。**真正随 $t$ 恶化的是相对开销**：同一层在这张卡上前向加反向的算力是 $6(N/L)bs/t$（前向 $2N$、反向 $4N$；纯推理只有前向的 $2N$），于是

$$r_{\text{TP}} = \frac{8\frac{t-1}{t}bshc / W}{6(N/L)bs/(tP)} = \frac{4(t-1)\,L\,h\,c\,P}{3\,N\,W} \;\propto\; (t-1)$$

其中 $P$ 是单卡算力、$W$ 是链路带宽。这个式子把「TP 越大通信越亏」量化了：70B 稠密模型（$L=80,h=8192,c=2$）在 H100 SXM（$P=989$ TFLOPS bf16 稠密，$W=450$ GB/s 单向 NVLink）上，$t=2$ 时 $r\approx 5\%$，$t=8$ 时 $r\approx 38\%$，$t=16$ 时 $r\approx 82\%$（都是「通信纯带宽下限 / 峰值算力上界」口径，实测算力利用率只有 40–55%，实际占比大致减半，同时 collective 固定延迟与小 GEMM 效率损失会把它推回去）。把 $W$ 换成 50 GB/s 的 NDR IB，这三个数要乘 9。

结论：TP 的可行半径由 $P/W$ 与链路带宽共同决定，不由显存决定。单机 8 卡 NVLink 是 H100 上唯一的合理半径，因此 $t \le 8$。

### SP：把非 TP 区也按序列切开

TP 只切了 attention/MLP 内部的矩阵乘，LayerNorm 和 dropout 仍然作用在完整的 $[s,b,h]$ 上，这些激活在每个 TP rank 上被**复制**——它们随 $s$ 线性增长（attention score 相关的 $5as^2b$ 项更是随 $s^2$ 增长），是长序列训练里最先爆掉的那块显存。sequence parallelism 论文给的每层激活显存公式（单位 bytes、fp16）把这一点写得很清楚：

$$\text{无并行：} sbh\left(34 + 5\frac{as}{h}\right),\qquad \text{只有 TP：} sbh\left(10 + \frac{24}{t} + 5\frac{as}{ht}\right),\qquad \text{TP+SP：} sbh\left(\frac{34}{t} + 5\frac{as}{ht}\right)$$

$10sbh$ 那一项就是被复制的 LayerNorm/dropout 区域，它不被 $t$ 除。SP 的做法是把这些区域沿序列维切成 $[X_1^s, X_2^s]$：进 attention/MLP 之前用 all-gather（论文里的 $g$）拼回完整序列，出块时用 reduce-scatter（论文里的 $\bar g$）把「求和 + 按序列切分」合并成一次通信。因为 ring all-reduce 本身就等于 reduce-scatter 加 all-gather，**SP 与 TP 的通信带宽完全相同**，换来的只是那些激活从「复制 $t$ 份」变成「每卡 $1/t$」。

两个注意点：SP 复用 TP 的通信组，$t=1$ 时无意义，也不额外占卡；它和长上下文推理里的 context parallel（按序列切、在 attention 内部做 ring attention）目标相似但通信组不同，前者是训练侧的显存优化，后者要解决的是单卡放不下长序列的 attention。

### PP：用 micro-batch 填气泡

把 $L$ 层切成 $p$ 段，每段一张卡，micro-batch 依次流过。GPipe 的 fill-drain 调度里，第 $i$ 个 micro-batch 在第 $j$ 段上的前向开始时间是 $i+j$，整条流水线共 $2(m+p-1)$ 个时隙，而每个 stage 实际忙 $2m$ 个时隙，于是

$$\text{气泡比例} = \frac{2(m+p-1)-2m}{2(m+p-1)} = \frac{p-1}{m+p-1}$$

$p=8,m=64$ 时是 9.9%，$m\to\infty$ 时趋近 0。要让气泡不超过 10%，取 $m \ge 9(p-1)$ 即可。1F1B 调度把反向提前，气泡变成 $(p-1)/m$（$p=8,m=64$ 时 10.9%，小 $m$ 时更差），它的真正收益是**激活显存从 $O(m)$ 降到 $O(p)$**：GPipe 必须把 $m$ 个 micro-batch 的激活留到反向（GPipe 因此依赖 rematerialization），1F1B 只需同时保留 $p$ 个在飞。把每个 rank 上的层再切成 $v$ 段（interleaved / virtual stage），气泡进一步降到约 $(p-1)/(vm)$，代价是点对点通信次数乘 $v$。

PP 的价值在于通信量小、可以跨机：每个 micro-batch 在每个 stage 边界只**单向**传一次边界激活 $bshc$，而 TP 在同样一段 stage 的每一层里单向就要发 $4\frac{t-1}{t}bshc$，两者相差 $4\frac{t-1}{t}\frac{L}{p}$ 倍（$t=8,p=8,L=80$ 时约 35 倍），所以 PP 能放在低带宽链路上。代价是它**无法减少激活显存**（要保持利用率就必须存多个 micro-batch 的激活），所以 PP 只能解决参数与优化器状态的显存问题。

### DP 与 ZeRO：16 bytes/param 与三阶段分片

PyTorch 风格的混合精度训练里，每个参数要 16 字节：bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + fp32 Adam 动量 4 + fp32 Adam 方差 4。70B 就是 $16\times 70\text{e}9 = 1.12$ TB，仅模型状态就需要至少 14 张 80 GB 卡——这就是「单卡放不下多少」的第一个答案。

DP 本身只复制模型、切数据，每步对全部梯度做一次 all-reduce，每卡收发 $2\frac{d-1}{d}Nc \approx 2Nc$。ZeRO 把这个过程改成「用通信换显存」，三阶段的每参数字节数是：

| 阶段 | 分片对象 | bytes/param | 70B、$d=64$ 时每卡 |
| --- | --- | --- | --- |
| 基线 | 无 | $2+2+12 = 16$ | 1120 GB（放不下） |
| ZeRO-1 | 优化器状态 | $4 + 12/d$ | 4.19 B → 293 GB |
| ZeRO-2 | + 梯度 | $2 + 14/d$ | 2.22 B → 155 GB |
| ZeRO-3 | + 参数 | $16/d$ | 0.25 B → 17.5 GB |

注意这张表是**纯 DP**（$t=p=1$）口径。实际配置里参数和梯度先被 $t\times p$ 切掉，优化器状态再由 ZeRO 在 DP 组内切：4096 卡取 $t=8,p=8,d=64$ 时，每卡 bf16 参数 2.19 GB + 梯度 2.19 GB + fp32 优化器状态 0.21 GB ≈ 4.6 GB，剩下的显存全部留给激活与 KV cache。ZeRO-3（FSDP）的代价是每层计算前要 all-gather 参数、反向后 reduce-scatter 梯度，通信次数变多且必须靠 prefetch 藏在计算后面。

### EP：all-to-all 与负载不均

MoE 层里 gate 先给每个 token 选 top-$k$ 个专家，然后按专家所在设备做 dispatch all-to-all，专家算完再 combine all-to-all 回来。每 token 每层的通信量是

$$B_{\text{EP}} = 2k\frac{E-1}{E}hc$$

代入 $k=2$、$E=8$、$h=4096$、bf16，得 28 KiB（$=3.5hc$）；同样条件下 TP 的每 token 每层通信量 $8\frac{t-1}{t}hc$ 在 $t=8$ 时是 56 KiB（$=7hc$），同一量级，但性质完全不同：TP 的通信是对称、静态、可预测的，EP 的 all-to-all 随 token 数增长、路由结果动态、专家负载天然不均（要靠容量因子丢 token 或 aux loss 平衡）。因此 EP 通常与 TP 组合——专家内部继续用 TP 切矩阵，专家之间用 EP，二者的通信组不重叠。MoE 的另一个好处是专家权重只被 EP 切分，dense 部分仍由 DP/TP 承担，模型容量与单 token FLOPs 解耦。

### 什么时候组合、怎么组合

1. **先算显存缺口**：参数 + 梯度 + 优化器状态 + 激活 + KV（推理时）。16 B/param 的账算完就知道纯 DP 够不够（70B 在 80 GB 卡上至少 14 张，还不含激活）。
2. **机内用 TP/SP 吃掉带宽友好的通信**：$t$ 取到单机 NVLink 域的卡数（H100 上是 8），序列很长时打开 SP 省激活；不要因为显存焦虑把 $t$ 推到跨机。
3. **跨机用 PP 和 DP/EP**：PP 走低通信量的点对点，用 $m \ge 9(p-1)$ 把气泡压到 10% 以内；DP/ZeRO 承担吞吐与优化器状态，用大 batch（每卡每步 token 数）摊薄梯度 all-reduce；MoE 用 EP 换容量。
4. **核对并行度乘积**：总卡数 $= t\times p\times d\times e$（SP 复用 TP 组不占卡；若用 context parallel 才额外占卡）。4096 卡典型分解是 $8\times 8\times 64$，1024 卡是 $8\times 8\times 16$。
5. **实测收口**：看 MFU、看 Nsight Systems 里 NCCL kernel 的时间占比、用 `nccl-tests` 量 all-reduce 的实际 bus bandwidth 是否接近链路峰值。Megatron-LM 在 512 卡上训 8.3B 模型拿到 15.1 PFLOPs、相对单卡基线 76% 的扩展效率，sequence parallelism 论文在 2240 张 A100 上训 530B 拿到 54.2% MFU——这些是可用于对比的现实锚点，而不是公式能给的东西。

推理侧的组合逻辑与训练侧不同：decode 是带宽受限（算力强度 1 FLOP/byte，远低于 H100 的 295 FLOP/byte 拐点），TP 的作用是把每卡的权重字节数除 $t$，从而降低单 token 延迟；DP 在推理里表现为多副本加负载均衡，用来堆吞吐；PP 在推理里代价高（每个请求多一次跨机往返、要维护多级调度），只在模型实在放不下时才用；EP 是 MoE 模型部署的必需品。

## 数值与代码验证

配置：70B 稠密量级（$L=80$、$h=8192$、$a=64$、$s=4096$、$b=1$、bf16），H100 SXM 口径（bf16 稠密算力 989 TFLOPS、HBM3 3.35 TB/s、NVLink 4.0 单向 450 GB/s、NDR IB 单向 50 GB/s）。算力带宽拐点 $989/3.35 = 295$ FLOP/byte。

下面两张表的通信量都按 NCCL 的**算法通信量**计（一次 all-reduce 每卡收发 $2\frac{t-1}{t}\times$张量大小；TP 每层 4 次 all-reduce，DP 每步 1 次梯度 all-reduce），算力按训练迭代的模型 FLOPs 计（每 token 前向 $2N$ + 反向 $4N$）；推理侧只有前向，所以那一节的算力是 $2N$/token。

TP 的逐层账（每次 all-reduce 的张量 $bshc$ = 64 MiB，每层 4 次，每卡收发 $8\frac{t-1}{t}bshc$）：

| $t$ | 每层每卡收发 | NVLink 时间 | 该层每卡算力（前向+反向） | 峰值时间 | 通信/计算（峰值） | 通信/计算（50% MFU） |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 256 MiB | 0.60 ms | 10.75 TFLOP | 10.87 ms | 5.5% | 2.7% |
| 4 | 384 MiB | 0.89 ms | 5.38 TFLOP | 5.44 ms | 16.5% | 8.2% |
| 8 | 448 MiB | 1.04 ms | 2.69 TFLOP | 2.72 ms | 38.4% | 19.2% |
| 16 | 480 MiB | 1.12 ms | 1.34 TFLOP | 1.36 ms | 82.3% | 41.2% |

同一张表把 $W$ 换成 50 GB/s 的 IB，$t=8$ 时通信/计算（50% MFU）从 19.2% 变成 173%——这是「TP 不出机」最硬的量化理由。

DP 的梯度 all-reduce（70B bf16 梯度 = 140 GB，$d$ 很大时每卡收发 $\approx 2Nc = 280$ GB，IB 串行 5.6 s；算力同样按 $6N$/token 计）：

| 每卡每步 token 数 | 4,096 | 16,384 | 65,536 | 262,144 | 1,048,576 |
| --- | --- | --- | --- | --- | --- |
| 通信/计算（50% MFU，IB） | 1.61 | 0.40 | 0.10 | 0.025 | 0.0063 |

与 TP 的表对照可以看出关键区别：**TP 的相对开销与 batch 无关（$\propto t-1$），DP 的相对开销与每卡每步 token 数成反比**。所以大 batch 与梯度累积能救 DP，救不了 TP。

PP 气泡（GPipe fill-drain，$(p-1)/(m+p-1)$）：

| $p$ | $m=8$ | $m=16$ | $m=32$ | $m=64$ | $m=128$ |
| --- | --- | --- | --- | --- | --- |
| 4 | 27.3% | 15.8% | 8.6% | 4.5% | 2.3% |
| 8 | 46.7% | 30.4% | 17.9% | 9.9% | 5.2% |
| 16 | 65.2% | 48.4% | 31.9% | 19.0% | 10.5% |

SP 的激活账（论文 Table 2 的公式直接代入上面的配置，单位 bytes、fp16，每层每卡）：

| $t$ | 只有 TP | TP+SP | 比值 | 扣掉 attention score 项（$5as^2b/t$，$t=8$ 时 671 MB/层）后的比值 |
| --- | --- | --- | --- | --- |
| 4 | 1.88 GB | 1.63 GB | 1.15× | $16sbh$ vs $8.5sbh$ = 1.88× |
| 8 | 1.11 GB | 0.81 GB | 1.36× | $13sbh$ vs $4.25sbh$ = 3.06× |
| 16 | 0.72 GB | 0.41 GB | 1.77× | $11.5sbh$ vs $2.13sbh$ = 5.41× |

不并行时是 $sbh(34+5as/h)$，在本配置下 6.51 GB/层，其中 $5as^2b$ 项占 5.37 GB——这也是论文主张「选择性重算 + SP」而不是继续全量重算的原因。论文在 22B/175B/530B/1T 四档配置上的口径是：单独用 SP 或选择性重算各能把激活显存砍掉近一半，两者合起来相对**不做重算、激活全部按 TP 切分保留的基线**（论文 Eq. 2）降到 20% 以下，即约 5×；而全量重算只相当于该基线的 10%（约 10×），代价是 30–40% 的执行时间开销——论文用「SP + 选择性重算」把这部分开销压到 4%（22B 单层实验里 39% → 4%，530B/1T 上 36% → 2%），摘要里的「激活显存降低 5×、重算的执行时间开销减少 90% 以上」就是这两个口径。要特别注意：5× 是相对 TP 基线、且含选择性重算的整体评估值，不等于单层公式里 TP 与 TP+SP 的直接比值（本表 $t=8$ 时只有 1.36×）。530B 模型在 2240 张 A100 上 MFU 从 42.1% 提到 54.2%（快 29%）。

把上面的账写成可运行的脚本，换配置只需改参数：

```python
"""并行策略选型计算器：显存 -> TP/SP -> PP/DP/EP -> 通信量复核。纯 Python，无需 GPU。"""
GB = 1e9

# H100 SXM 口径：bf16 稠密算力 / HBM3 带宽 / NVLink 4.0 单向 / NDR IB 单向
PEAK, HBM, NVL, IB = 989e12, 3.35e12, 450e9, 50e9

def report(params=70e9, layers=80, hidden=8192, seq=4096, micro=1,
           tp=8, pp=8, dp=64, ep=1, elem=2):
    gpus = tp * pp * dp * ep
    share = params / (tp * pp * ep)              # 每卡本地参数量
    zero0 = share * (2 + 2 + 12) / GB            # 优化器状态留在本地
    zero1 = share * (2 + 2 + 12 / dp) / GB       # ZeRO-1：优化器状态按 DP 切
    payload = micro * seq * hidden * elem        # 一次 all-reduce 的张量字节
    tp_layer = 8 * (tp - 1) / tp * payload       # 每层每卡收发：前向 2 次 + 反向 2 次 all-reduce
    m = max(9 * (pp - 1), 1)                     # 取 m 让 GPipe 气泡 <= 10%
    bubble = (pp - 1) / (m + pp - 1)
    dp_step = 2 * (dp - 1) / dp * params * elem  # 每卡每步梯度收发
    ep_tok = 2 * 2 * hidden * elem * (ep - 1) / ep if ep > 1 else 0.0
    flops = 6 * params * micro * seq / (tp * pp)  # 每卡每 micro-batch 前向+反向算力
    print(f"总卡 {gpus} = TP{tp} x PP{pp} x DP{dp} x EP{ep}")
    print(f"  每卡模型状态 ZeRO-0 {zero0:7.2f} GB | ZeRO-1 {zero1:7.2f} GB")
    print(f"  每卡算力 {flops/1e12:7.2f} TFLOP -> 峰值 {flops/PEAK*1e3:7.2f} ms"
          f" | 50% MFU {flops/(0.5*PEAK)*1e3:7.2f} ms")
    print(f"  TP 每层收发 {tp_layer/2**20:7.2f} MiB -> NVLink {tp_layer/NVL*1e3:6.2f} ms"
          f" | IB {tp_layer/IB*1e3:6.2f} ms")
    print(f"  PP 气泡 p={pp}, m={m} -> {bubble*100:5.1f}%")
    print(f"  DP 每步每卡梯度 {dp_step/GB:7.1f} GB -> IB {dp_step/IB:5.2f} s")
    if ep > 1:
        print(f"  EP 每 token 每层 all-to-all {ep_tok:7.0f} B")

report()                                  # 4096 卡训练
report(tp=8, pp=1, dp=8, ep=1)            # 8 卡小规模：看 ZeRO 前后的差距
```

输出（复算值）：4096 卡 = TP8×PP8×DP64 时，每卡模型状态 ZeRO-0 17.50 GB / ZeRO-1 4.58 GB；每卡每 micro-batch 算力 26.88 TFLOP（峰值 27.18 ms）；TP 每层收发 448 MiB（NVLink 1.04 ms，IB 9.40 ms）；PP 气泡 $m=63$ 时 10.0%；DP 每步每卡梯度 275.6 GB（IB 5.51 s）。

推理侧的两个交叉验证：decode batch=1 时每 token 算力 $2N = 140$ GFLOP，$t=8$ 时每卡 17.5 GFLOP（峰值 17.7 µs），但每卡要读 17.5 GB 权重，HBM 时间 5.22 ms——算力强度 1 FLOP/byte，所以单序列上限约 190 token/s，瓶颈完全在带宽，TP 在这里买到的是延迟。KV cache 用 GQA 口径（80 层、8 个 KV 头、head_dim 128、bf16）是 320 KiB/token，32k 上下文 10 GiB、128k 上下文 40 GiB；$t=8$ 正好按 KV 头切分，32k 时每卡 1.25 GiB。

## 常见追问

- **追问：为什么 TP 通常不超过 8？**
  - 要点：三个原因叠加。① 相对开销 $r_{\text{TP}} \propto (t-1)$，70B 在 H100 上 $t=8$ 已经是 38%（峰值口径）/ 19%（50% MFU）；② 跨出 NVLink 域后带宽 450 → 50 GB/s，直接 ×9；③ 每卡矩阵变小，GEMM 效率与 collective 固定延迟占比同时恶化。8 卡 NVLink 域是硬件给定的事实上界。
- **追问：ZeRO 和 DP 是什么关系？**
  - 要点：ZeRO 是 DP 的显存优化版本，通信换显存。ZeRO-1/2/3 分别分片优化器状态（12 B/param）、梯度、参数，每参数占用从 16 B 降到 $4+12/d$、$2+14/d$、$16/d$。ZeRO-3 额外引入每层 all-gather 参数与前向重算的取舍，必须靠 prefetch 与计算重叠。
- **追问：EP 和 DP 有什么区别？MoE 为什么不用 TP 切专家？**
  - 要点：DP 每个 rank 持有完整模型副本、切数据；EP 每个 rank 只持有部分专家、切参数，token 靠 all-to-all 流动。用 TP 切专家意味着每个 token 都要经过所有 TP rank 的专家计算，通信量与激活量都不可接受；EP 让每个 token 只去 $k$ 个专家所在的卡，容量与 FLOPs 解耦。
- **追问：什么时候 PP 比 TP 划算？**
  - 要点：跨机、模型层数多、能拿到足够大的 $m$ 时。PP 的通信是按 stage 边界的点对点：每个 micro-batch 每个边界只传一份 $bshc$ 边界激活，与层数无关，总量远低于 TP 的每层 all-reduce，天然适合低带宽链路；代价是气泡 $\frac{p-1}{m+p-1}$ 与无法降低激活显存。反过来，延迟敏感的在线推理不喜欢 PP，因为每个请求多一次跨机往返。
- **追问：SP 和 context parallel 是一回事吗？**
  - 要点：不是。SP 复用 TP 通信组、只在 LayerNorm/dropout 这类非 TP 区切序列，通信量等价于 TP 的 all-reduce，不额外占卡；CP（ring attention 一类）用独立的序列并行组把 attention 的计算也切开，要额外占卡，用于单卡放不下超长序列的推理与训练。
- **追问：怎么判断当前配置是被通信还是被算力卡住？**
  - 要点：看 MFU 与 Nsight Systems 里 NCCL kernel 的占比，再用 `nccl-tests` 量 all-reduce 的 bus bandwidth 是否接近链路峰值。通信占比高且 bus bandwidth 已达峰值，说明该换并行切法；bus bandwidth 远低于峰值，说明是消息太小或没重叠，先调 bucket 与 overlap。

## 公司变体

- **Google DeepMind**：公开技术栈是 TPU + XLA + GSPMD/GShard/Pathways，回答这类问题时更偏「分片如何被编译器推导」与专家并行的负载均衡——GShard 论文本身就是把 MoE 扩到 600B+ 参数、2048 张 TPU v3 上训 4 天的工作，所以追问容易落在 all-to-all 的通信量与路由均衡策略，也会要求写出气泡、通信量的公式。
- **Meta**：公开的 Llama 训练栈是 Megatron 风格 TP+PP+DP 组合加 ZeRO/FSDP，问题更偏工程实现：集群拓扑怎么排、$t/p/d$ 怎么配、MFU 掉点怎么查、故障恢复与 checkpoint 怎么设计。
- **Amazon**：公开资料集中在 Trainium/Neuron 与 SageMaker 的 model parallel 库，偏工程实现与部署约束（成本、实例拓扑、跨机带宽），也常把训练侧并行与推理侧多副本部署放在一起问。
- **NVIDIA**：Megatron-LM/NeMo 与 NCCL 的作者方，问题偏硬件口径与实现细节——NVLink 与 IB 的具体带宽、ring/tree all-reduce 的差别、MFU 的定义与测量方式，欢迎你把 $P/W$ 这类量的数量级直接算出来。

## 相关题目

- [[inference-serving-08]]：把 70B 服务的显存账算细（权重、KV cache、激活、碎片化），是「原理与推导」里组合顺序第 1 步（先算显存缺口）在推理侧的展开。
- [[inference-serving-11]]：并行切法与 serving 框架选型的关系，vLLM/SGLang/TensorRT-LLM 各自对 TP/EP/PD 分离的支持差异。
- [[llm-internals-10]]：MoE 的路由、容量因子与负载均衡，是理解 EP 通信量为什么随 token 数增长的前置知识。

## 参考资料与归属

- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism（Shoeybi et al.，2019-09-17）[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)：列并行/行并行切分、每层两次 all-reduce、8.3B 模型在 512 卡上 15.1 PFLOPs 与 76% 扩展效率。
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism（Huang et al.，2018-11-16）[arXiv:1811.06965](https://arxiv.org/abs/1811.06965)：batch splitting 流水线调度、气泡公式、rematerialization。
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（Rajbhandari et al.，2019-10-04）[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)：16 bytes/param 的分解与三阶段分片。
- Reducing Activation Recomputation in Large Transformer Models（Korthikanti et al.，2022-05-10）[arXiv:2205.05198](https://arxiv.org/abs/2205.05198)：每层激活显存公式（Table 2）、sequence parallelism 的 $g/\bar g$ 推导、激活显存 5×（相对 TP 基线）与重算开销 90%+ 的评估结果。
- GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding（Lepikhin et al.，2020-06-30）[arXiv:2006.16668](https://arxiv.org/abs/2006.16668)：expert parallelism 与 all-to-all 的大规模实践。
- 上述五条均标注为「延伸」：题目属于推理与服务专题，而并行策略的原始定义、激活显存公式与气泡推导来自训练侧的这五篇论文；文中的服务侧组合建议（decode 为什么用 TP、PP 在推理里的代价）是基于这些结论的推论，没有单独的 serving 来源，所有数值均为在 H100 口径下的重新计算。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
