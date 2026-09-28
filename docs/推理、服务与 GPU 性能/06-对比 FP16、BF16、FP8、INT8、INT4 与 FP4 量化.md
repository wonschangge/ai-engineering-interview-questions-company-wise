---
type: question
id: inference-serving-06
topic: 推理、服务与 GPU 性能
order: 6
question: 对比用于 serving 的 FP16、BF16、FP8、INT8、INT4 和 FP4。每往下降一档会损失什么？
question_en: Compare FP16, BF16, FP8, INT8, INT4 and FP4 for serving. What do you lose at each step down?
asked_at: [Mistral AI, Apple, NVIDIA, Together AI, Character.AI]
level: 进阶
tags: [量化, 精度, 显存, kernel]
sources:
  - title: How does Model Quantization work?
    url: https://outcomeschool.com/blog/how-does-model-quantization-work
    author: Amit Shekhar (Outcome School)
    published: 2026-07-15
  - title: FP8 Formats for Deep Learning（延伸）
    url: https://arxiv.org/abs/2209.05433
    author: Micikevicius et al.
    published: 2022-09-12
  - title: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（延伸）
    url: https://arxiv.org/abs/2210.17323
    author: Frantar et al. (ICLR 2023)
    published: 2022-10-31
  - title: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration（延伸）
    url: https://arxiv.org/abs/2306.00978
    author: Lin et al. (MLSys 2024)
    published: 2023-06-01
  - title: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models（延伸）
    url: https://arxiv.org/abs/2211.10438
    author: Xiao et al. (ICML 2023)
    published: 2022-11-18
related: [inference-serving-08, inference-serving-10, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> 这六个格式分两族：FP16/BF16/FP8/FP4 是浮点，动态范围由指数位提供，误差是**相对的**、与数值大小无关（E4M3 的单值相对误差上界是 6.25%）；INT8/INT4 是定点，动态范围完全由 scale 提供，误差是**绝对的**、由该 scale 覆盖范围内的最大绝对值决定。
> 所以每降一档丢的东西并不一样：FP16→BF16 丢 3 位尾数（单值误差大 8 倍）换回整个 FP32 指数范围，推理几乎无损；BF16→FP8 尾数只剩 3 位，必须靠 per-tensor/per-channel scaling 把数值搬进规格化区间，做 W8A8 需要校准；FP8→INT8 位宽不变但动态范围塌缩成一个标量，激活 outlier 会撑爆整个张量的 scale（同一个 0.01，在有 60 的 outlier 时 INT8 被压成 0，E4M3 只损失 4.6%）；INT8→INT4 只剩 16 个格点，只能做权重量化，靠 per-group(128) scale 加 GPTQ/AWQ 的误差补偿；INT4→FP4 连组内动态范围都要靠 block(16/32) scale 重建，而且格点非等距，长尾任务最先掉点。
> 报「损失」时要把四笔账一起给：数值精度、显存（含 scale 元数据）、速度（有没有对应 dtype 的 tensor core 与融合 kernel）、工程复杂度（校准数据、kernel 支持、评测基线）。

## 面试官在考什么

- 能否把「位宽」拆成符号位 / 指数位 / 尾数位，说清每档具体丢的是哪一位，而不是只答「精度下降」。
- 是否理解浮点与定点在误差性质上的根本差别：浮点给相对误差上界，定点给绝对误差上界。因此同一个张量内的宽动态范围对定点是致命的，对浮点只是次正规区间的问题。
- 能否把损失拆成精度 / 显存 / 速度 / 工程复杂度四类，并知道后三类的判断依据：tensor core 支持哪些 dtype、kernel 有没有把反量化融进去、校准与元数据的代价。
- 是否清楚权重与激活的可量化性差异，以及 W4A16 与 W8A8 各自解决什么：前者省字节（带宽与容量），后者省算力（tensor core 吞吐）。
- 有没有「先建评测基线、按任务分层测」的意识，而不是拿一个 perplexity 数字当验收标准。

常见错误答案：

- 「同为 8 bit，INT8 和 FP8 差不多」。位宽相同但误差性质完全不同；在 H100 上两者的稠密算力也确实相同，选 FP8 的理由是动态范围与 scaling 的简单性，不是算力。
- 「量化总是更快」。W4A16 在 batch 1 的 decode 上是带宽收益，在计算受限的 prefill 上几乎没有算力收益；如果 kernel 每次都要把权重解包回 FP16，收益还可能被反量化开销吃掉。

## 原理与推导

### 1. 六个格式的静态属性

| 格式 | 符号-指数-尾数 | 最大有限值 | 最小规格化正数 | 值约为 1 处的间隔 / 有效十进制位 | 备注 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 1-5-10 | 65504 | $6.10\times10^{-5}$ | $2^{-10}=9.77\times10^{-4}$ / 3.31 位 | 超过 65504 溢出为 inf |
| BF16 | 1-8-7 | $3.39\times10^{38}$ | $1.18\times10^{-38}$ | $2^{-7}=7.81\times10^{-3}$ / 2.41 位 | 指数位与 FP32 完全相同 |
| FP8 E4M3 | 1-4-3 | 448 | $2^{-6}=0.015625$ | $2^{-3}=0.125$ / 1.20 位 | 不表示 inf，只有一个 NaN 编码 |
| FP8 E5M2 | 1-5-2 | 57344 | $6.10\times10^{-5}$ | $2^{-2}=0.25$ / 0.90 位 | 遵循 IEEE 754，有 inf/NaN，留给梯度 |
| FP4 E2M1 | 1-2-1 | 6 | 1 | 0.5 / 0.60 位 | 正值只有 0、0.5、1、1.5、2、3、4、6 |
| INT8 | 定点 | 由 scale 决定 | — | 对称时 $\max/127$ | 256 个等距格点 |
| INT4 | 定点 | 由 scale 决定 | — | 对称时 $\max/7$ | 16 个等距格点 |

几个容易答错的点：E4M3 的最大值是 448 而不是 512，因为指数域全 1 时尾数被限制到 6，7 号编码留给 NaN；E5M2 的范围与 FP16 同量级（同样是 5 位指数）但尾数只有 2 位，所以它在推理里几乎只用于梯度或对精度极不敏感的中间量；E2M1 的格点是非等距的，靠近 0 密、远离 0 疏，在 1.0 附近相邻值差 0.5，到 4 和 6 之间差 2。有效十进制位按 $\log_{10}(2^{m+1})$ 算：从 FP16 往下，单值相对误差上界依次是 FP16 的 8 倍、128 倍、256 倍、512 倍。这些是**相对**误差，定点给的是**绝对**误差，两者不能直接比——INT8 per-tensor 在 $\max=1$ 时步长 0.007874、绝对误差上界 0.003937，比 FP16 在 1 附近的半个步长（$2^{-11}=4.88\times10^{-4}$）大 8 倍；换成量级 0.01 的值，同一个绝对误差对应的相对误差上界就是 39%。

### 2. 动态范围从哪来：指数位 vs scale

浮点的值写成 $v = (-1)^{s}\cdot 2^{E-b}\cdot(1+M/2^{m})$：相邻可表示值的间隔与当前值的量级成正比，所以**相对**舍入误差有与数值大小无关的上界 $\lvert v-\hat v\rvert/\lvert v\rvert\le 2^{-(m+1)}$。指数位决定能表示多大、多小：FP16 的指数域覆盖约 $2^{30}$ 倍的范围，BF16 约 $2^{254}$，E4M3 约 $2^{15}$，E2M1 只有约 $2^{2.6}$（算上最小次正规数约 $2^{3.6}$）。

定点的映射是 $q = \mathrm{round}(x/s)+z$ 与 $\hat x = (q-z)\cdot s$：格点等距，所以**绝对**误差有上界 $s/2$，而能覆盖的范围完全由 $s$ 决定。对称 per-tensor 量化取 $s=\max\lvert x\rvert/(2^{b-1}-1)$：INT8 是 $\max/127$，INT4 是 $\max/7$。

两条推论是这道题的主干：

- 同一个张量里动态范围越宽，定点越吃亏，因为 $s$ 被最大值撑大，小值被压到同一个格点上；浮点的相对误差不变，只是量级极小的值会掉进次正规区间。
- 定点给不出相对精度保证。一个 0.01 的数和一个 1.0 的数在同一个 INT8 张量里共享同一个绝对步长。

### 3. FP16 到 BF16：丢 3 位尾数，换回整个指数范围

尾数从 10 位降到 7 位（含隐含 1 是 11 位对 8 位），值在 1 附近的间隔从 $9.77\times10^{-4}$ 变成 $7.81\times10^{-3}$，单值相对误差上界大 8 倍；有效十进制位从 3.3 位降到 2.4 位。换来的是 3.39e38 的最大值，任何不会写爆 FP32 的量都不会写爆 BF16。这一档在 serving 里基本不算损失：7 位尾数带来的 0.4% 量级相对误差，远小于权重压到 8 bit 时引入的误差，多数评测里 FP16 与 BF16 的差距落在评测噪声内——要有确定结论必须在自己的评测集上跑。真正的风险在 FP16 那一侧：65504 的上限意味着某个超大激活或 attention logit 会被舍入成 inf，然后污染整条链路；BF16 没有这个问题。这就是「bf16 训练更稳」的机制根源（FP16 训练必须引入 loss scaling 来避免梯度下溢，BF16 不需要，因为它的指数范围与 FP32 一致）。所以 serving 默认选 BF16，除非某些 kernel 只有 FP16 路径。

### 4. BF16 到 FP8：尾数只剩 3 位，靠 scaling 与误差平均救回来

E4M3 只有 3 位尾数，单值相对误差上界 6.25%。直接把权重 cast 成 FP8 是不可用的，能用的原因有三条：

- **per-tensor / per-channel scaling**：$W' = W/s$，取 $s=\max\lvert W\rvert/448$ 把张量映射到 FP8 的可用区间。要注意 scale 在这里的作用与定点不同——浮点的相对误差与缩放无关，scale 只决定数值是否落在规格化区间内（避免下溢到次正规数）。所以对浮点来说，per-channel 比 per-tensor 好的地方在于：量级小的通道按自己的 scale 归一化后不会掉进次正规区，而不是「步长变小了」。
- **误差在归约维上部分抵消**：矩阵乘 $y=\sum_k x_k w_k$ 有 $K$ 项，若每项误差近似独立零均值，误差按 $\sqrt K$ 增长而信号按 $K$ 增长，相对误差约按 $1/\sqrt K$ 缩小。这是 3 位尾数在 matmul 里还能工作的统计原因；它对随机误差成立，对系统性偏差和 outlier 不成立。
- **累加仍在 FP32**：tensor core 的 accumulate 精度不变，误差来自输入量化而不是求和。

与 INT8 的差别是这一档最值得讲的地方：两者位宽相同，H100 上 FP8 与 INT8 的稠密算力相同（都是 bf16 的 2 倍），但 FP8 用 4~5 位指数换来 15~30 个二进制数量级的动态范围，激活里的 outlier 不再撑爆整个张量的 scale。FP8 论文的结论之一正是：一些用 16 位格式训练、对定点 int8 量化不友好的模型，改用 FP8 做 post-training quantization 效果明显更好。代价是校准与 scale 管理：每层每个张量至少一个 scale，要统计 amax，还要在延迟缩放与实时缩放之间做选择（见「常见追问」）。E5M2 范围大、精度更低，用于梯度；推理里的权重与激活基本都用 E4M3。

### 5. FP8 到 INT8：位宽不变，动态范围塌缩成一个标量

对称 per-tensor INT8 只有一个 scale：$\max\lvert x\rvert$ 里出现一个 outlier，全部小值一起遭殃。拿源文的 outlier 设定量化一遍（绝大多数激活落在 $[-1,1]$，其中一个值是 60），要保护的小值取 0.01：

| 方案 | scale | 0.01 的量化结果 | 相对误差 |
| --- | --- | --- | --- |
| INT8 per-tensor，无 outlier（max = 1） | 0.007874 | 0.00787 | 21.3% |
| INT8 per-tensor，有 outlier（max = 60） | 0.472441 | 0（被压没） | 100% |
| FP8 E4M3 per-tensor，有 outlier（max = 60） | 0.133929 | 0.01046 | 4.6% |

这是「权重好压、激活难压」的定量版：定点的误差是绝对的，一个 outlier 让所有小值失去分辨率；浮点的误差是相对的，outlier 只把整条数轴平移了一个指数，小值仍然规格化。

工程上只有两条出路：一是提高 scale 粒度（per-channel 对权重很有效，对 per-token 激活也行），二是把量化难度搬走。SmoothQuant 走的是第二条：基于「权重好量化、激活不好量化」这个观察，做一个数学等价的变换

$$Y = \left(X\,\mathrm{diag}(s)^{-1}\right)\left(\mathrm{diag}(s)\,W\right),\qquad s_j = \frac{\max\lvert X_j\rvert^{\alpha}}{\max\lvert W_j\rvert^{\,1-\alpha}},\ \alpha\approx 0.5$$

$s_j$ 是输入通道 $j$ 的迁移系数，$\alpha$ 控制从激活搬到权重的比例。变换本身不改变计算结果，只改变哪一侧承担量化误差，把两侧的量化难度拉平，W8A8 才能在掉点可忽略的前提下跑通（论文口径：最高 1.56 倍加速、2 倍显存下降，单节点可服务 530B 模型）。

### 6. INT8 到 INT4：只剩 16 个格点，靠方法学兜底

INT4 对称 per-tensor 的步长是 $\max\lvert x\rvert/7$，误差上界 7.1%（max 为 1 时）。per-tensor W4A16 直接上线会崩，所以必须补两件事：

- **per-group / per-channel scale**：GPTQ 用 group size 128 分组，每组一个 scale 与 zero-point，按 FP16 存，折算下来是 4.25 bit/权重（只存 scale 是 4.125 bit）。分组把 $\max\lvert x\rvert$ 限制在一小组权重内，步长随组内幅度收缩，这是最直接的收益来源。
- **误差补偿**：GPTQ 用近似二阶信息逐列量化，每量化一列就用剩余权重回补它引入的误差，把 175B 模型压到 3~4 bit 大约需要 4 个 GPU 小时，相对未压缩基线精度损失可忽略（论文口径）。AWQ 的出发点是「并非所有权重同等重要，只保护约 1% 的显著通道就能大幅降低量化误差」，而这些通道要看激活分布而不是权重本身；它用等价的通道缩放来保护显著通道，不做混合精度、不依赖反向传播或重建，因此不容易过拟合校准集，也能推广到指令微调与多模态模型。

主流形态是 **W4A16**：权重 4 bit、激活留在 16 bit。原因是激活难压，而权重已经占了模型字节数的大头。损失的特征要说清：perplexity 这类通用指标常常看起来很稳，但在数学、代码、长链推理、多语言上掉点更明显——AWQ 论文专门把 coding 与 math 基准、指令微调模型拿出来评测，正是因为它比语言建模困惑度更能暴露退化。更细的一层是概率分布被扰动：top-1 往往不变，但排在后面的候选 token 相对次序会变，温度一高差异就被放大成「输出质量」问题。具体掉多少必须按自己的任务分层测，不同模型、group size、校准集差别很大。

### 7. INT4 到 FP4：block-wise scale 成为必需品

E2M1 只有 16 个编码，正值是 0、0.5、1、1.5、2、3、4、6。落在某个区间（例如 $[1,2)$）里的格点只有 2 个，所以一组元素必须共享一个 scale 才能用满网格，而且这个组要足够小。两套主流方案：

| 方案 | 元素格式 | 共享 scale | 有效位宽 | 说明 |
| --- | --- | --- | --- | --- |
| MXFP4 | E2M1 | block = 32 个元素共享一个 E8M0（8 位纯指数，2 的幂） | 4.25 bit | OCP 微缩放格式族，同族的 MXFP8 是 E4M3 + 同样的 block 32 scale |
| NVFP4 | E2M1 | block = 16 个元素共享一个 FP8 E4M3 scale，再乘一个 per-tensor 的 FP32 二级 scale | 4.5 bit | Blackwell 的原生 FP4 路径 |

组越大，越容易被组内 outlier 撑爆；block 16/32 的存在就是为了把 $\max$ 控制在小范围内。FP4 的用武之地是权重（Blackwell 的 tensor core 原生支持 FP4 matmul）与 KV cache 这类「元素多、单个元素重要性低」的张量；激活仍然最难，FP4 训练通常要配合 QAT（前向用 FP4、反向保留高精度）。长尾任务的掉点在这一档最明显：尾数比 INT4 更少，而且网格非等距，小量级的值分辨率更粗，需要精确概率排序的数学/代码/多语言任务最先出问题。

### 8. 为什么权重好压、激活难压

| 维度 | 权重 | 激活 |
| --- | --- | --- |
| 是否随输入变化 | 固定，训练完就冻结 | 每条请求都不同 |
| 分布 | 稳定，跨通道的幅度差异可以用 per-channel 隔离 | 少数通道幅度极大，且位置随模型规模出现 |
| 量化粒度 | per-channel / per-group 都容易做，元数据可离线打包进权重 | 在线做 per-token / per-channel 需要 kernel 内 reduction |
| 难度 | 低，INT4 都能做到可接受 | 高，INT8 per-tensor 就会掉点 |

KV cache 遵循同样的规律，而且更细一层：K 在少数通道上幅度远大于其它通道，适合 per-channel（按 head_dim 的通道）量化；V 的分布在 token 维上更均匀，适合 per-token 量化。这就是 KIVI 一类方案对 K 和 V 用不同粒度的原因，也是「K 比 V 难压」这句话的出处。工程上最常见的折中是 INT8/FP8 KV cache：每个元素 2 字节变 1 字节，容量与读取带宽同时减半，风险可控（见 [[llm-internals-02]]）；再往下的 FP4 KV cache 需要 per-channel 或 block scale，收益更大但掉点风险明显上升。

### 9. 把损失拆成四类，并据此决策

| 档位 | 主要损失 | 必须配套的工程动作 |
| --- | --- | --- |
| FP16 → BF16 | 单值相对误差大 8 倍，换来不溢出的指数范围 | 无；确认 kernel 有 BF16 路径 |
| BF16 → FP8 | 尾数只剩 3 位；需要校准与 scale 管理 | per-tensor/per-channel scale、amax 统计、E4M3 用于权重与激活 |
| FP8 → INT8 | 动态范围塌缩成一个 scale，激活 outlier 致命 | 提高 scale 粒度，或 SmoothQuant 类迁移；接受校准流程 |
| INT8 → INT4 | 16 个格点，激活不能再压 | per-group(128) scale、GPTQ/AWQ、按任务分层评测 |
| INT4 → FP4 | 网格非等距，长尾任务最先掉 | block(16/32) scale、指定硬件与 kernel、QAT（若训练侧也要压） |

四笔账要分开报：

1. **数值精度**：perplexity 会掩盖退化，必须加数学、代码、长链推理、多语言和你自己的业务集，并且固定采样参数对比。
2. **显存**：权重按 bit/权重 折算，别忘元数据（group 128 + FP16 scale/zero-point 是 +0.125~0.25 bit，NVFP4 的 block 16 是 +0.5 bit）；KV cache 按精度线性缩放。
3. **速度**：先看硬件有没有对应 dtype 的 tensor core（INT8 从 Turing 起、FP8 从 Hopper/Ada 起、FP4 从 Blackwell 起），再看 kernel 有没有把反量化融进 GEMM。W4A16 通过 FP16 tensor core 做反量化后的矩阵乘（Marlin 这类 kernel），收益来自带宽；要拿到算力收益必须上 W8A8 或 FP8。
4. **工程复杂度**：校准数据要代表性、per-group 元数据要能被 kernel 高效读取、与 TP/EP 并行策略要兼容、上线后要有精度回归监控。

决策顺序：显存吃紧就先上 W4A16（AWQ/GPTQ）加 INT8/FP8 KV cache，这是收益最大、改动最小的一步；追求吞吐且硬件是 Hopper 及以上就上 FP8 W8A8，prefill 与 decode 都受益；精度敏感场景（长链推理、代码、数学、多语言）停在 FP8，不要为了省显存降到 INT4；端侧用 INT4/FP4 加融合 kernel；无论选哪档，先建立分层评测基线，再逐档往下试。

## 数值与代码验证

口径说明：权重与显存用十进制 GB（$1\ \mathrm{GB}=10^{9}$ 字节）和 GiB（$1\ \mathrm{GiB}=2^{30}$ 字节）两种写法，KV cache 沿用 [[llm-internals-02]] 的 1024 进制；算力与带宽取 NVIDIA H100 SXM5 的规格口径（HBM3 3.35 TB/s，bf16 稠密 989 TFLOPS，FP8/INT8 稠密均为其 2 倍，即 1979 TFLOPS/TOPS）。

**表 1：70B 权重的字节数（含 scale 元数据）**

| 精度 | bit/权重 | 权重总量 | 相对 FP16 | H100 单序列 decode 带宽上限 |
| --- | --- | --- | --- | --- |
| FP16 / BF16 | 16 | 140.0 GB（130.4 GiB） | 1× | 41.8 ms/token，23.9 tok/s |
| FP8 / INT8 | 8 | 70.0 GB（65.2 GiB） | 2× | 20.9 ms/token，47.9 tok/s |
| MXFP8（block 32 + E8M0） | 8.25 | 72.2 GB（67.2 GiB） | 1.94× | — |
| INT4 纯 4 bit | 4 | 35.0 GB（32.6 GiB） | 4× | 10.4 ms/token，95.7 tok/s |
| INT4 + group 128 + FP16 scale | 4.125 | 36.1 GB（33.6 GiB） | 3.88× | — |
| INT4 + group 128 + scale 与 zero-point（GPTQ 口径） | 4.25 | 37.2 GB（34.6 GiB） | 3.76× | 11.1 ms/token，90.1 tok/s |
| MXFP4（block 32 + E8M0） | 4.25 | 37.2 GB（34.6 GiB） | 3.76× | — |
| NVFP4（block 16 + E4M3 scale） | 4.5 | 39.4 GB（36.7 GiB） | 3.56× | — |

decode 上限那列只算「把全部权重从 HBM 读一遍」这一项，属于纯带宽上界，实际还要加 KV cache 读取、反量化与 kernel 开销，因此它是乐观值；它同时说明为什么 weight-only 量化对 batch 1 的 TPOT 有效：这一档的瓶颈就是字节数。

**表 2：KV cache（LLaMA-3-70B，80 层、8 个 KV 头、head_dim 128）**

| 精度 | 每 token | 8k 上下文 | 32k 上下文 | 128k 上下文 |
| --- | --- | --- | --- | --- |
| FP16 / BF16 | 320.0 KiB | 2.50 GiB | 10.00 GiB | 40.0 GiB |
| FP8 / INT8 | 160.0 KiB | 1.25 GiB | 5.00 GiB | 20.0 GiB |
| FP4 | 80.0 KiB | 0.625 GiB | 2.50 GiB | 10.0 GiB |

**表 3：8 张 H100（640 GB）上 32k 会话的并发上限（未计激活与碎片）**

| 权重精度 | KV FP16 | KV FP8 |
| --- | --- | --- |
| FP16 权重（140 GB） | 46.6 条 | 93.1 条 |
| FP8 权重（70 GB） | 53.1 条 | 106.2 条 |
| INT4 权重（37.2 GB） | 56.1 条 | 112.3 条 |

把权重从 FP16 压到 INT4 只把并发从 46.6 提到 56.1，压 KV cache 却直接翻倍——长上下文场景里 KV cache 才是主导项，量化的收益要先算清楚压的是谁（细节见 [[inference-serving-08]]）。

```python
"""inference-serving-06 的数值复算（只依赖标准库）"""
GiB = 1024 ** 3


def fp8_extremes(e4m3=True):
    """枚举全部编码，返回 (最大有限值, 最小正规格化值)"""
    exp_bits, man_bits = (4, 3) if e4m3 else (5, 2)
    bias = (1 << (exp_bits - 1)) - 1
    finite, normals = [], []
    for bits in range(256):
        e = (bits >> man_bits) & ((1 << exp_bits) - 1)
        m = bits & ((1 << man_bits) - 1)
        if (e4m3 and (e, m) == (0b1111, 0b111)) or (not e4m3 and e == 0b11111):
            continue                          # E4M3FN 的 NaN、E5M2 的 inf 与 NaN
        if e == 0:
            v = 2 ** (1 - bias) * (m / 2 ** man_bits)        # 次正规数
        else:
            v = 2 ** (e - bias) * (1 + m / 2 ** man_bits)    # 规格化数
            normals.append(v)
        finite.append(v)
    return max(finite), min(normals)


print(fp8_extremes(True), fp8_extremes(False))
# (448.0, 0.015625) (57344.0, 6.103515625e-05)

N, BW = 70e9, 3.35e12                                # 70B 参数；H100 SXM5 HBM3 带宽
for bits in (16, 8, 4 + 32 / 128, 4 + 8 / 16):       # 含 scale 元数据的有效位宽
    by = N * bits / 8
    print(f"{bits:>6.3f} bit -> {by / 1e9:>6.1f} GB，单序列 {by / BW * 1e3:>5.2f} ms/token")

L, H_kv, d_head = 80, 8, 128                         # LLaMA-3-70B
per_token = 2 * L * H_kv * d_head                    # 每 token 的元素数（K 与 V 两份）
for ctx in (8192, 32768, 131072):
    print(ctx, [round(per_token * b * ctx / GiB, 2) for b in (2, 1, 0.5)])  # FP16/FP8/FP4


def e4m3_round(x):
    """把正数舍入到最近的 E4M3FN 可表示值"""
    cand = [2 ** (1 - 7) * (m / 8) if e == 0 else 2 ** (e - 7) * (1 + m / 8)
            for e in range(16) for m in range(8) if (e, m) != (0xF, 0x7)]
    return min(cand, key=lambda v: abs(v - x))


x = 0.01                                             # 源博文 outlier 设定下要被保护的小值（本篇自选）
for xmax in (1.0, 60.0):
    q8 = round(x / (xmax / 127)) * (xmax / 127)
    qf = e4m3_round(x / (xmax / 448)) * (xmax / 448)
    print(f"max={xmax:>4}: INT8 {q8:.5f} ({abs(q8 - x) / x:.1%}) "
          f"E4M3 {qf:.5f} ({abs(qf - x) / x:.1%})")
# max= 1.0: INT8 0.00787 (21.3%)  E4M3 0.01004 (0.4%)
# max=60.0: INT8 0.00000 (100.0%) E4M3 0.01046 (4.6%)
```

与源文对照：源博文用「7B 权重、FP32 28 GB / INT8 7 GB / INT4 3.5 GB」举例，换算成 70B 就是 140 GB / 70 GB / 35 GB，与本篇一致；它的 scale 例子是 $[-1,1]$ 映射到 0~255 的非对称口径（$2/255=0.00784$、zero-point 128），本篇表里用的是对称口径 $\max/127=0.007874$，两者相差 0.4%，量级相同，差异来自 255 步与 127 步的映射方式。源博文没有给出格式的位域细节、FP4 的 block 布局与各档的具体评测结论：位域细节与 FP8/INT8 的对比来自延伸来源（见末节），MXFP4/NVFP4 的 block 与 scale 布局来自 OCP 与 NVIDIA 的公开格式定义，不在上述来源范围内。

## 常见追问

- **追问**：为什么 bf16 在训练里比 fp16 稳？
  - 要点：BF16 有 8 位指数，动态范围与 FP32 相同，梯度、激活、loss 都不会因为超出 65504 上界变成 inf；FP16 只有 5 位指数，训练时必须用 loss scaling 把梯度放大到可表示区间再缩回来，否则下溢成 0。代价是 BF16 尾数少 3 位，单值精度低，靠优化器的随机性和误差平均盖过去。推理侧这个差异小得多，因为不存在反向传播带来的极端小值。
- **追问**：per-tensor、per-channel、per-group 的区别与代价是什么？
  - 要点：粒度越细，$\max\lvert x\rvert$ 越贴近局部，误差越小，代价是元数据与 kernel 复杂度。per-tensor 一个 scale，最省但最脆；per-channel 按输出/输入通道各一个 scale，权重上几乎免费（离线打包）；per-group 每 128 个权重一个 scale 与 zero-point，是 INT4 的默认选择，代价约 0.125~0.25 bit/权重，且需要专门的重排 kernel 才能高效读取。激活侧做 per-token/per-channel 需要在 kernel 内做 reduction，会引入同步开销。
- **追问**：量化省显存，为什么不一定省时间？
  - 要点：省时间的前提是瓶颈在字节数或算力上，且 kernel 支持。decode 是带宽受限的，权重字节减半就接近减半时间；prefill 是计算受限的，W4A16 的矩阵乘仍然在 FP16 tensor core 上跑，只是多了一步反量化，算力没有变多，收益来自少了读取字节。反过来，如果 kernel 需要反复解包 4 bit、做非连续访存，反量化开销可能吃掉全部收益，这也是 W4A16 早期「权重显存降到 1/4、速度却没变」的原因，后来靠 Marlin 这类重排 + 融合 kernel 才补上。
- **追问**：FP8 的 scaling 怎么做，延迟缩放和实时缩放有什么区别？
  - 要点：延迟缩放（delayed scaling）用过去若干步的 amax 历史来定 scale，好处是不用等当前张量的归约结果，不阻塞流水线，代价是对分布突变的张量反应慢；实时缩放（current scaling）用当前张量自己的 amax，精度更好但要一次 reduction，可能引入同步。更细的粒度（per-token per-128-channel 的激活、128×128 的权重块）是把 FP8 用到训练/大规模服务的关键手段。E4M3 的 scale 要把 amax 映射到 448 以内，E5M2 映射到 57344。
- **追问**：为什么 KV cache 里 K 比 V 难量化？
  - 要点：K 的少数通道幅度远大于其它通道（与 RoPE 和注意力结构有关），per-tensor 或 per-token 的 scale 会被这些通道撑大；V 的分布沿 token 维更均匀。所以常见做法是对 K 用 per-channel、对 V 用 per-token。KV 量化的收益是双份的：容量和每步读取带宽同时下降，因此通常连 TPOT 一起降。
- **追问**：怎么验收一次量化？
  - 要点：先固定解码参数（温度、top-p、max tokens）与 prompt 集合，再分层评测：语言建模困惑度、数学、代码、长链推理、多语言、指令遵循，以及业务自己的集；同时记录输出长度分布（量化会改变停止行为）与拒答率。只看 perplexity 会漏掉长尾退化，只跑一个基准会被噪声骗。

## 公司变体

`asked_at` 覆盖的五家公司在公开材料里的侧重不同：

- **NVIDIA**：格式与 kernel 的定义方（TensorRT-LLM、Transformer Engine、CUTLASS）。偏底层实现：tensor core 支持哪些 dtype、scale 放在哪一级、per-block scale 对 kernel 与带宽的额外代价、FP8 的延迟缩放与 FP4 的 block 16 布局。这家的追问最容易落到「scale 存在哪、kernel 怎么读」。
- **Mistral AI**：自研推理栈与开源权重。偏工程取舍：同一个模型发布哪些量化权重、用哪种格式换多少吞吐、长上下文下 KV cache 用什么精度，属于「选型与落地」而不是推导。
- **Apple**：端侧与私有云计算场景，内存预算与能耗是硬约束。偏工程与 kernel 实现：4 bit 权重加融合 kernel、按设备能力选择量化档位、权重的分组与打包格式。
- **Together AI**：多模型托管推理服务。偏工程与评测：一套 kernel 要覆盖多少种量化格式、每美元 token 数怎么变、不同格式上线前的精度回归怎么做。
- **Character.AI**：大规模高并发对话。偏工程：单卡能放多少并发由权重字节与 KV cache 共同决定，因此低比特权重与 KV cache 量化直接换成成本，讨论通常围绕吞吐与显存的取舍。

以上是按各家公开技术材料与业务方向做的侧重判断，具体题目以实际面试轮次为准。

## 相关题目

- [[inference-serving-08]]：70B 模型的完整显存预算（权重、KV cache、激活、碎片），本篇的表 1 与表 2 是那道题的输入。
- [[inference-serving-10]]：batch 1 的 roofline 计算，量化改变的就是那条公式里「每 token 读多少字节」这一项。
- [[llm-internals-02]]：KV cache 的显存公式与四条压缩路线，量化是其中最容易事后加装的一条。

## 参考资料与归属

1. [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)，Amit Shekhar（Outcome School），2026-07-15。提供量化/反量化的 scale 与 zero-point 机制、对称与非对称量化的取舍、per-tensor 与 per-channel 的区别、PTQ 与 QAT 的对比、激活 outlier 的例子（$[-1,1]$ 中出现 60）、以及 GPTQ / AWQ / bitsandbytes / GGUF 的定位与「4 bit 是当前甜点」的结论。
2. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)（延伸），Micikevicius et al.，2022-09-12。E4M3 与 E5M2 的位域定义，E4M3 通过不表示 inf、只保留一个 NaN 编码来扩展动态范围；该文在 CNN、RNN、Transformer（最大 175B）上匹配了 16 位训练的精度，并指出 FP8 的 post-training quantization 对抵抗定点 int8 量化的模型更有效。本篇第 4 节的 E4M3/E5M2 属性、FP8 与 INT8 的对比来自该文。
3. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)（延伸），Frantar et al.（ICLR 2023），2022-10-31。基于近似二阶信息的 one-shot 权重量化：175B 模型约 4 GPU 小时压到 3~4 bit，精度损失可忽略，压缩收益相对早先的 one-shot 方法翻倍以上。本篇第 6 节的 GPTQ 描述来自该文；4.25 bit 是按该文 group 128 加 FP16 scale/zero-point 折算出的有效位宽。
4. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)（延伸），Lin et al.（MLSys 2024），2023-06-01。只保护约 1% 的显著权重通道即可大幅降低量化误差，显著通道要依据激活分布而不是权重来挑，用等价缩放实现、不做混合精度、不依赖反向传播或重建；对 coding/math 与指令微调、多模态模型给出评测。本篇第 6 节的 AWQ 描述与「按任务分层评测」的依据来自该文。
5. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)（延伸），Xiao et al.（ICML 2023），2022-11-18。训练无关的 W8A8 方案：用数学等价变换把量化难度从激活迁移到权重，$\alpha$ 控制迁移强度；论文口径为最高 1.56 倍加速、2 倍显存下降，可在单节点服务 530B 模型。本篇第 5 节的迁移公式与收益数字来自该文。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
