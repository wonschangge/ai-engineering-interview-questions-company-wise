---
type: question
id: finetuning-05
topic: 微调、后训练与对齐
order: 5
question: QLoRA 是如何降低显存占用的？它需要做哪些 quantization 权衡？
question_en: How does QLoRA reduce memory, and what quantization tradeoffs does it require?
asked_at: [Hugging Face]
level: 进阶
tags: [qlora, 量化, nf4, 显存]
sources:
  - title: How does Model Quantization work?
    url: https://outcomeschool.com/blog/how-does-model-quantization-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization（视频）
    url: https://www.youtube.com/watch?v=lnfWvX66FUk
    author: Outcome School
    published: ""
  - title: QLoRA: Efficient Finetuning of Quantized LLMs（延伸）
    url: https://arxiv.org/abs/2305.14314
    author: Dettmers et al. (NeurIPS 2023)
    published: 2023-05-23
  - title: bitsandbytes 文档（延伸）
    url: https://huggingface.co/docs/bitsandbytes/index
    author: Hugging Face
    published: ""
related: [finetuning-04, finetuning-09, inference-serving-06, finetuning-06]
updated: 2026-09-28
---

## 一句话答案

> QLoRA = **冻结的 4-bit 量化底座 + bf16 的 LoRA 适配器**：前向与反向都把 NF4 权重反量化成 bf16 参与矩阵乘，
> 但只对适配器算权重梯度，梯度穿过量化权重回传到 $L_1,L_2$。它把权重一项从 16 bit/参数压到
> $4 + 8/64 + 32/(64\cdot 256) = 4.127$ bit/参数，即 **3.9×** 的底座压缩，再叠加 LoRA 本来就省掉的梯度与优化器状态，
> 论文口径是在**单张 48GB GPU** 上微调 65B 并保持 16-bit 全量微调的任务表现。
> 代价是五点权衡：反量化带来的吞吐损失、量化误差、只有适配器可训练、超参不可迁移、部署路径要额外选。

## 面试官在考什么

- 你能否把「省显存」拆成**两个正交的机制**：4-bit 压缩的是权重，LoRA 压缩的是梯度与优化器状态。只答「4 bit 省 8 倍」的人会被立刻追问到崩。
- 你能否听懂三个技术点各自解决**哪个具体问题**：NF4 解决「桶怎么分」，双重量化解决「量化常数自己的开销」，分页优化器解决「长序列的显存尖峰」。
- 你能否把 NF4 与 INT4 的差别讲成**分桶规则**的差别（等概率质量 vs 等距），而不是「NF4 更准」。
- 你是否知道这套东西的**收益边界**：QLoRA 让「原本放不下」变成「放得下」，而不是让「放得下」变成「更快」。底座还在显存里（只是压到 4-bit），激活一分没省。
- 你能否诚实说出量化误差到底被谁吸收了、什么任务上吸收不掉。

常见错误答案：

- 「QLoRA 把模型压到 4 bit，所以显存降 8 倍」。权重确实降约 3.9×，但优化器状态、激活、临时缓冲的账完全不同：固定开销口径的比值是 29–30×，把两边都逃不掉的激活算进来后端到端峰值只有 10–20×（见「数值与代码验证」）。
- 「4-bit 训练不掉点，因为反量化误差可以忽略」。反量化误差不是可忽略的小量，它被低秩适配器的可学习空间补偿掉了；换成全量微调或换掉适配器就没这个保证。
- 「QLoRA 是用 GPTQ/AWQ 量化后训练」。GPTQ/AWQ 是离线权重量化、面向**部署**，其误差结构假设推理时才成立；QLoRA 走的是 bitsandbytes 的 NF4 kernel 与分页优化器。

## 原理与推导

### 0. 先把两块显存分开：量化管权重，LoRA 管状态

训练显存可以写成两项之积：

$$M \;=\; \underbrace{N \cdot b_{\text{weight}}}_{\text{底座，所有参数都要存}} \;+\; \underbrace{N_{\text{train}} \cdot b_{\text{state}}}_{\text{可训练参数的状态}} \;+\; \underbrace{M_{\text{act+tmp}}}_{\text{激活、临时、碎片}}$$

其中 $N$ 是总参数量，$N_{\text{train}}$ 是可训练参数量。几个配置的 $(b_{\text{weight}}, b_{\text{state}})$：

| 配置 | $b_{\text{weight}}$ | $b_{\text{state}}$ | 谁在起作用 |
| --- | --- | --- | --- |
| bf16 + Adam 全量微调 | 2 B | 14 B（梯度 2 + master 4 + 动量 8） | 总量 $16N$ |
| LoRA on bf16 底座 | 2 B | 16 B（仅适配器） | LoRA：$N_{\text{train}} \ll N$ |
| **QLoRA** | **0.516 B** | 16 B（仅适配器） | 量化压权重 + LoRA 压状态 |

两条机制互不替代，所以才值得一起用。bf16 全量微调的 16 B/参数来自 [[finetuning-09]]；LoRA 的可训练参数量 $r(d+k)$ 来自 [[finetuning-04]]。

### 1. 底座：4-bit 量化、逐 block 常数、NF4

块状量化的定义为（论文式 1、2）：把张量 $\mathbf{X}$ 展平成 $n=(b\times h)/B$ 个连续块，第 $i$ 块用自己的常数 $c_i$ 做绝对最大值归一化：

$$\mathbf{X}^{\text{Int8}}_i=\operatorname{round}\!\left(\frac{127}{\operatorname{absmax}(\mathbf{X}_i)}\mathbf{X}_i\right)=\operatorname{round}(c_i\,\mathbf{X}_i),\qquad
\operatorname{dequant}(c_i,\mathbf{X}_i)=\frac{\mathbf{X}_i}{c_i}$$

为什么必须逐块而不是逐张量：一整个张量里只要出现一个幅值很大的离群权重，per-tensor 的 scale 就被撑大，其余权重的有效位数被压扁。块越小越稳，但每块的量化常数本身要占字节——这就是双重量化的来由。

**NF4 的分桶规则。** 论文的构造（式 4）是对标准正态做分位数量化，取相邻分位的中点作为桶中心：

$$q_i=\tfrac12\Big(Q_X\!\big(\tfrac{i}{2^k+1}\big)+Q_X\!\big(\tfrac{i+1}{2^k+1}\big)\Big)$$

再把这张表归一化到 $[-1,1]$，并为了能精确表示 0（padding、mask 等零值必须零误差）把正负两半拼起来、去掉重复的零点。$k=4$ 时得到 16 个值，论文附录 E 给出的完整表是：

```text
[-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
 -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
 0.7229568362236023, 1.0]
```

**它和均匀 INT4 的差别在间距分布，不在位宽。** 上面这张表相邻值的间距不均：中间最窄 $0.0796$，两端最宽 $0.3038$（差 3.8×）；同样覆盖 $[-1,1]$ 的均匀 INT4 间距恒为 $2/15=0.1333$。也就是说 NF4 在零附近比均匀桶密 1.67×，在两端稀 2.28×——它把「分辨率」花在概率密度最高的地方。（代码里用的是标准 int4 码点 $(-8,\ldots,7)/8$：间距 $0.125$、正端只到 $0.875$；$2/15$ 是让 16 个码点对称铺满 $[-1,1]$ 的解析口径。）

一句话概括两种假设：**均匀 INT4 假设数值在一个区间里均匀出现；NF4 假设数值近似零均值正态、且先验已知**。第二种假设下每个桶装到等量的概率质量，码本的熵最大，这是 NF4 说自己「信息论最优」的确切含义——最优性绑定在正态先验上。

### 2. 双重量化：把量化常数自己再量化一次

块大小 64 时，第一级常数的开销是：

$$b_{\text{const}} = \frac{32\ \text{bit}}{64} = 0.5\ \text{bit/param}$$

在 4.5 bit/参数 的总预算里这占了 **11%**，比把块从 64 放大到 128 省下的还多。双重量化的做法是把第一级的 FP32 常数 $c_2$ 当成输入再做一次量化：$c_2$ 恒为正，所以先减均值做对称量化，用 **FP8、块大小 256** 得 $c_2^{\text{FP8}}$ 与第二级常数 $c_1^{\text{FP32}}$：

$$b_{\text{const}}' = \frac{8}{64} + \frac{32}{64\times 256} = 0.125 + 0.00195 = 0.127\ \text{bit/param}$$

于是每参数从 $4.5$ bit 降到 $4 + 0.127 = 4.127$ bit，**省下 0.373 bit/参数**——这就是论文里的 0.37。代进 65B：$0.373/8 \times 65\times10^9 = 3.03$ GB，对应论文的「约 3 GB」。这个节省随块大小变大而衰减：块 64 省 0.373、128 省 0.187、256 省 0.093。论文对块大小的措辞是：权重用块 64 换「更高的量化精度」，$c_2$ 用块 256 换「省内存」——小块的常数开销正是 DQ 要解决的问题。

### 3. QLoRA 的完整前向：一个存储类型 + 一个计算类型

论文式 5、6 把单层写成：

$$\mathbf{Y}^{\text{BF16}}=\mathbf{X}^{\text{BF16}}\,\operatorname{doubleDequant}(c_1^{\text{FP32}},c_2^{\text{FP8}},\mathbf{W}^{\text{NF4}})+\mathbf{X}^{\text{BF16}}\mathbf{L}_1^{\text{BF16}}\mathbf{L}_2^{\text{BF16}}$$

$$\operatorname{doubleDequant}(c_1,c_2,\mathbf{W}^{4\text{bit}})=\operatorname{dequant}\big(\operatorname{dequant}(c_1,c_2),\mathbf{W}^{4\text{bit}}\big)=\mathbf{W}^{\text{BF16}}$$

关键在于 $c_1,c_2,\mathbf{W}$ 都只是**工作集**：矩阵乘本身在 bf16 上做，反量化出来的 $\mathbf{W}^{\text{BF16}}$ 是临时结果，不进显存常驻。所以显存里长期躺着的是 4-bit 权重与常数，计算精度仍然是 16-bit。

### 4. 梯度怎么走：为什么只训适配器还能"补偿"量化误差

参数更新只需要 $\partial E/\partial \mathbf{L}_i$，不需要 $\partial E/\partial \mathbf{W}$。但求 $\partial E/\partial \mathbf{L}_i$ 必须算 $\partial \mathbf{X}/\partial \mathbf{W}$，而这一步要走同一套反量化、在 bf16 里算导数。于是失败模式很清晰：

$$\mathbf{Y} = \mathbf{X}\,(\mathbf{W}^{\text{NF4}\to\text{BF16}} + \Delta\mathbf{W}),\qquad \Delta\mathbf{W}=\tfrac{\alpha}{r}\mathbf{L}_1\mathbf{L}_2$$

量化误差 $\mathbf{W}-\mathbf{W}^{\text{NF4}\to\text{BF16}}$ 是**固定的、与输入无关的**加性扰动，而 $\Delta\mathbf{W}$ 是可学的。训练时适配器的梯度里就包含了补偿这项扰动所需的方向，所以误差被低秩子空间吸收一部分——这才是「4-bit 不掉点」的真实机制。它有三个隐含前提：量化误差足够小、任务所需的更新确实低秩（[[finetuning-04]]）、适配器挂得足够多。论文的消融正好印证第三点：**只挂 $W_q,W_v$ 复现不了全量微调，必须挂到所有线性层**。

### 5. 分页优化器：解决尖峰而不是解决均值

梯度检查点的显存曲线不是平的：反向时按层重算，遇到长序列的那一个 micro-batch 会突然抬高。分页优化器用 NVIDIA unified memory 给优化器状态做分页，显存不够时自动换出到主机内存、更新时换回：

- 它换的是**优化器状态**，不是激活，也不是权重；
- 代价是换页延迟，所以论文只声称「分页不频繁时与普通优化器同速」；论文给出 65B 在 48GB 上 batch size 16 时两者训练速度相同，并把「什么情况下会变慢」留给后续工作；
- 论文把它归为让 33B/65B 装进 24GB/48GB 单卡的必要条件，而不是提速手段。

### 6. 量化权衡（题干第二问）

| 权衡 | 具体表现 | 工程判断 |
| --- | --- | --- |
| 速度 | 每次用到权重都要反量化到 bf16，是额外 kernel 与访存；收益是权重读取的带宽降到 $1/3.9$ | 论文的对照对象是 16-bit 全量微调，口径是「不降低 runtime 与预测表现」；在底座本来就放得下的场景，吞吐通常不及 bf16 LoRA，需要自己实测 |
| 精度 | 量化误差是固定扰动，主要由低秩适配空间吸收 | 长链推理、代码、多语言这类对数值敏感的任务必须用任务集实测，别只看 perplexity |
| 可训练范围 | 底座永远冻结，只有适配器可学 | 学全新能力受低秩容量限制，见 [[finetuning-06]] |
| 超参 | 量化底座上的最优学习率/秩与 bf16 底座不完全可比 | 论文在 NF4 底座上扫出的最优学习率是 7B/13B 用 $2\times10^{-4}$、33B/65B 用 $1\times10^{-4}$（同时把 batch size 从 16 加大到 32/64）；换底座要重扫 |
| 部署路径 | 产物是适配器，推理时要么合并回 bf16 底座，要么直接在量化底座上跑 | 前者精度更稳，后者省显存；量化部署侧的取舍见 [[inference-serving-06]] |

一句话总结权衡的方向：QLoRA 用**吞吐与可训练容量**换**可行性**，它解决的是「放不下」，不是「不够快」。

## 数值与代码验证

### 每参数字节与双重量化（自算，与论文口径对照）

| 项 | 位宽 | bit/param | B/param | 占 65B 的显存 |
| --- | --- | --- | --- | --- |
| 第一级：NF4 权重 | 4 | 4 | 0.5 | 32.5 GB |
| 第一级：FP32 常数（块 64） | 32/64 | 0.5 | 0.0625 | 4.06 GB |
| 小计（无 DQ） | — | **4.5** | 0.5625 | 36.56 GB |
| 第二级：FP8 常数（块 256） | 8/64 | 0.125 | 0.015625 | 1.02 GB |
| 第二级：FP32 常数（$64\times256$） | 32/16384 | 0.00195 | 0.000244 | 0.016 GB |
| **合计（NF4 + DQ）** | — | **4.127** | **0.51587** | **33.53 GB** |

省下 $0.373$ bit/参数，65B 上 $3.03$ GB，与论文的 0.37 bit 与「约 3 GB」一致。口径：十进制 GB（$10^9$），常量项与权重项都按同一 $N$ 摊。

### 底座显存与适配器状态

$N$ 按 config 逐项加总（embedding + lm_head + 每层 attention/FFN/layernorm + final norm，即 Llama 的不 tie 口径）。适配器状态按 16 B/可训练参数（bf16 权重 2 + bf16 梯度 2 + fp32 master 4 + Adam 动量 8）：

| 模型 | $N$ | bf16 底座 | NF4+DQ 底座 | LoRA $r{=}16$（q/k/v/o） | LoRA $r{=}64$（全部线性层） |
| --- | --- | --- | --- | --- | --- |
| Llama-7B（$L{=}32,d{=}4096$） | 6.74 B | 13.48 GB | **3.48 GB** | 16.78 M → 0.27 GB | 159.9 M → 2.56 GB |
| Llama-13B（$L{=}40,d{=}5120$） | 13.02 B | 26.03 GB | **6.71 GB** | 26.21 M → 0.42 GB | 250.4 M → 4.01 GB |
| Llama-2-70B（$L{=}80,d{=}8192$） | 68.98 B | 137.95 GB | **35.58 GB** | 65.54 M → 1.05 GB | 828.4 M → 13.25 GB |

形状取自 Llama-1/Llama-2 config：7B 的 $d_{ff}{=}11008$、13B 的 $d_{ff}{=}13824$、70B 的 $d_{ff}{=}28672$（GQA，8 个 KV head），$V{=}32000$；GB 一律十进制。

注意最后一行：70B 在 4-bit 下底座就要 $35.58$ GB（$33.1$ GiB），加 $r{=}64$ 全部线性层的适配器状态 $13.25$ GB 已经 $49$ GB，一张 80GB 卡勉强，两张就宽裕。**「4-bit 让 70B 单卡可训」成立，「让 70B 单卡随便训」不成立。**

### 7B 的峰值加总（$b{=}1$，梯度检查点）

梯度检查点只存每层输入（2 个 layernorm 输入 $\times\, s\,b\,d\times 2$ B）：

| 激活/状态 | 公式 | $s{=}512$ | $s{=}1024$ | $s{=}2048$ | $s{=}4096$ |
| --- | --- | --- | --- | --- | --- |
| NF4+DQ 底座 | $N\times 0.51587$ B | 3.48 | 3.48 | 3.48 | 3.48 |
| 适配器 + 梯度 + 优化器状态 | $16.78\text{M}\times16$ B | 0.27 | 0.27 | 0.27 | 0.27 |
| 检查点激活 | $2L\,s\,b\,d\times2$ B | 0.27 | 0.54 | 1.07 | 2.15 |
| 临时/碎片（经验值） | — | 1.5 | 1.5 | 1.5 | 1.5 |
| **合计** | | **5.51** | **5.78** | **6.32** | **7.39** |

对上论文：论文实测 7B 的 4-bit 底座占 5,048 MB（$b{=}1$ 的训练实测口径，含运行时与反量化缓冲，比纯权重账的 3.48 GB 高约 1.6 GB），并按 $b{=}1,s{=}512$ + 梯度检查点画显存分解图；上表 $s{=}512$ 一列的 5.51 GB 与它同量级，差值来自「底座单算」与「底座 + 激活 + 临时」的口径不同。

**「8–12 GB 量级」从哪来**：上表是 $b{=}1$（论文图 6 的口径）。论文 Table 9 给的是 batch size 16（7B/13B）、32（33B）、64（65B），并没有梯度累积这一项。一旦真的把 16 条 $s{=}512$ 样本放进一个 batch，最后那一次 lm_head 投影就要 $16\times512\times32000\times2\ \text{B}=0.52$ GB（0.49 GiB）的 logits（若损失在 fp32 里算则翻倍），激活也随之线性增长，峰值直接进 8–12 GB 区间。所以正确的说法是：**QLoRA 的固定开销不到 4 GB，峰值由 batch、序列长度与 logits 口径决定**。

### 与 bf16 全量微调的对照（固定开销口径）

| 模型 | bf16 + Adam 全量（$16N$） | QLoRA（$N\times0.51587$ B + 适配器状态） | 比值 |
| --- | --- | --- | --- |
| 7B | 108 GB | 3.74 GB | 28.8× |
| 13B | 208 GB | 7.13 GB | 29.2× |
| 70B | 1104 GB | 36.63 GB | 30.1× |

表里的 $N$ 取 config 精确值（7B 是 $6.738$ B；名义 7B 的常用口径 $16\times7\times10^9=112$ GB 见 [[finetuning-09]]，与上表的 $108$ GB 差 3.9%）。论文侧的对照是「65B 常规 16-bit 微调需要 >780 GB，QLoRA 降到 <48 GB」，以及「33B 需要分页优化器才装进 24GB」；$780$ GB 正好是 $65\text{B}\times12$ bytes/参数（bf16 权重与梯度各 2、Adam 两个动量各 4），比上面的 $16N$ 少了 fp32 主权重那 4 bytes/参数（$65\text{B}\times16=1040$ GB），两个口径都不含激活，结论方向一致。

### 代码：量化与反量化、以及 NF4 与均匀 INT4 的对比

```python
import numpy as np
from statistics import NormalDist

NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])                      # 论文附录 E
UNI16 = np.array([-8 + i for i in range(16)]) / 8.0  # 均匀 INT4 的 16 个码点

def quantize_block(w, table, block=64):
    """逐 block：absmax 归一化 -> 最近码点 -> 乘回常数（式 1 的 NF4 版）。"""
    q = np.empty_like(w)
    for s in range(0, len(w), block):
        blk = w[s:s + block]
        c = np.abs(blk).max()                      # 量化常数 c（每 block 一个）
        if c == 0:
            q[s:s + block] = 0
            continue
        r = np.clip(blk / c, -1, 1)                # 归一化到码表定义域
        q[s:s + block] = table[np.abs(r[:, None] - table[None, :]).argmin(1)] * c
    return q

def bits_per_param(block=64, dq=True):
    b = 4 + 32 / block                             # 第一级常数
    return 4 + 8 / block + 32 / (block * 256) if dq else b

rng = np.random.default_rng(0)
w = rng.normal(0, 1, 2_000_000)                    # 正态：论文的权重先验
w_heavy = w.copy()
idx = rng.choice(len(w), len(w) // 2000, replace=False)
w_heavy[idx] *= rng.uniform(5, 20, len(idx))       # 加 0.05% 离群
for name, x in (("正态", w), ("正态+离群", w_heavy)):
    m_nf4 = ((x - quantize_block(x, NF4)) ** 2).mean()
    m_int4 = ((x - quantize_block(x, UNI16)) ** 2).mean()
    print(f"{name:9s} NF4 MSE {m_nf4:.6f}  均匀 INT4 MSE {m_int4:.6f}  INT4/NF4 = {m_int4/m_nf4:.2f}x")

for name, x in (("正态", w), ("t(4)", rng.standard_t(4, len(w))), ("均匀", rng.uniform(-1, 1, len(w)))):
    r = ((x - quantize_block(x, UNI16)) ** 2).mean() / ((x - quantize_block(x, NF4)) ** 2).mean()
    print(f"{name:6s} 均匀 INT4/NF4 = {r:.2f}x")   # 换数据分布，码表与块大小不变

print(f"无 DQ: {bits_per_param(dq=False):.4f} bit/param -> 有 DQ: {bits_per_param():.4f} bit/param, "
      f"省 {(bits_per_param(dq=False)-bits_per_param())*65e9/8/1e9:.2f} GB（65B）")
```

自算结果（块 64、每块独立 absmax 归一化、200 万样本）：正态权重上均匀 INT4 的 MSE 是 NF4 的 **1.17×**，叠加 0.05% 离群后拉到 **1.40×**；把数据分布换掉（码表与块大小不变）：$t$ 分布（自由度 4）**1.77×**、$[-1,1]$ 上的均匀分布 **0.57×**（即均匀 INT4 反超约 **1.75×**）。这些是合成数据上的复算，不是论文数字，但方向与论文一致：**NF4 的收益来自正态先验与重尾稳健性，不来自「4 bit 里 NF4 天然更强」**。论文自己的经验对照是 Pile Common Crawl 上的平均 perplexity（125M–13B 的 OPT/BLOOM/LLaMA/Pythia 平均）：Int4 34.34、Float4(E2M1) 31.07、Float4(E3M0) 29.48、NF4+DQ 27.41；论文附录 F 用 Shapiro-Wilk 检验发现 7B LLaMA 上 7.5% 的隐藏单元不是正态分布——先验大体成立但有例外。

## 常见追问

- **追问**：NF4 与 INT4 在什么分布假设下差异最大？
  - 要点：两者都是 16 个码点，差别只在分桶规则——NF4 等概率质量（零附近密、两端稀），INT4 等距。数据越偏离正态（重尾、离群多），NF4 越占优；数据本身接近均匀分布时均匀 INT4 反而更好。所以「用 NF4」隐含了「我的权重近似正态」这个赌注。
- **追问**：双重量化为什么能省？省的是哪部分？
  - 要点：省的**不是权重本身的 4 bit，而是量化常数自己的 0.5 bit/参数**。块 64 时 FP32 常数占 4.5 bit 预算的 11%，把常数用 FP8（块 256）再量化一次，$32/64$ 变成 $8/64+32/(64\cdot256)=0.127$，省 0.373 bit/参数、65B 上省约 3 GB。它买的是「能塞进 24/48GB 卡」的那一点余量，精度收益很小。
- **追问**：为什么 4-bit 训练还能不掉点？
  - 要点：量化误差是固定的加性扰动，而低秩适配器是可学的，它的梯度里天然包含补偿方向。前提三条：误差足够小、任务更新确实低秩、适配器挂得够多（论文发现只挂 $W_q,W_v$ 复现不了全量微调，必须挂到所有线性层）。
- **追问**：什么情况下反而不要用 QLoRA？
  - 要点：三类。底座本来就放得下时，bf16 LoRA 省掉反量化开销、吞吐更好、没有量化误差；需要学新能力或改数值敏感行为时，只有适配器可训练是硬上限；需要最高质量或最短训练时间时，别拿量化底座去赌。QLoRA 的目标场景始终是「不量化就放不下」。
- **追问**：QLoRA 和 GPTQ、AWQ 是什么关系？
  - 要点：QLoRA 是量化**训练**，权重在训练全程保持 4-bit、适配器在 bf16 上更新；GPTQ/AWQ 是量化**部署**，离线做完权重量化、激活保持高精度，用来让推理更省。两者可以串起来（QLoRA 训练 → 合并 → 量化部署），但假设与误差结构不同，不能互相替代。
- **追问**：分页优化器是不是省显存的主力？
  - 要点：不是。它换出的是优化器状态，只在长序列 micro-batch 造成尖峰时触发；论文明确说分页「只在处理长序列时发生，很少见」，并给出 65B 在 48GB、batch size 16 时与普通优化器同速。真正省显存的是 4-bit 底座、LoRA 与梯度检查点（[[finetuning-09]]）。

## 公司变体

`asked_at` 只有 Hugging Face。这家公司的问法**偏工程实现而非数学推导**：NF4 与 `bitsandbytes` 的 kernel、`BitsAndBytesConfig` 的 `load_in_4bit` / `bnb_4bit_quant_type` / `bnb_4bit_use_double_quant` 三个旋钮、`peft` 的 `LoraConfig(target_modules=...)` 与 `prepare_model_for_kbit_training`、以及训练产物怎么存、怎么合并、怎么上 Hub，都是它的主场。依据是 bitsandbytes 与 peft 都由 Hugging Face 维护，QLoRA 的参考实现也直接集成进了 `transformers`（论文明确写了这一点）。

因此这题如果只讲清 NF4 的分位数构造、却说不清 `bnb_4bit_use_double_quant=True` 到底动了什么、`target_modules` 挂漏了会怎样，是拿不到分的。反过来说，分位数推导卡住但能把显存账算清、把三个旋钮讲明白，通常也能过关。

## 相关题目

- [[finetuning-04]]：LoRA 的分解与秩选择——本篇的 $N_{\text{train}}=r(d+k)$ 与「更新低秩」前提都从那里来。
- [[finetuning-09]]：7B 全量微调与 LoRA 的显存账——本篇的对照基准（16 B/参数）与激活口径在那篇。
- [[finetuning-06]]：LoRA、prefix tuning、prompt tuning 与全量微调的选型——决定「该不该上 QLoRA」的上一层决策。
- [[inference-serving-06]]：FP16/BF16/FP8/INT8/INT4/FP4 的逐档损失——量化部署侧的另一半，和 QLoRA 的量化训练互补。

## 参考资料与归属

- Amit Shekhar (Outcome School)，*How does Model Quantization work?*，<https://outcomeschool.com/blog/how-does-model-quantization-work>：scale 与 zero-point、对称/非对称量化、per-tensor 与 per-channel、PTQ 与 QAT、权重量化与激活量化、离群值问题，以及 GPTQ/AWQ/bitsandbytes/GGUF 的定位，另有 7B 在 FP32/INT8/INT4 下约 28 GB / 7 GB / 3.5 GB 的对照。
- Outcome School，*AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization*（视频），<https://www.youtube.com/watch?v=lnfWvX66FUk>。
- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer，*QLoRA: Efficient Finetuning of Quantized LLMs*（NeurIPS 2023），<https://arxiv.org/abs/2305.14314>（2023-05-23）：论文摘要在单张 48GB GPU 上微调 65B、保持 16-bit 全量微调表现；三个技术点定义（NF4、双重量化省 0.37 bit／65B 约 3 GB、分页优化器）；式 1、2、4、5、6；4.5→4.127 bit 的推导；附录 E 的 NF4 完整表；附录 F 的 Shapiro-Wilk 正态性检验（7.5% 隐藏单元非正态）；附录 G 的显存分解（$b{=}1,s{=}512$ + 梯度检查点）；第 3 节的 7B 4-bit 底座 5,048 MB 实测；Table 2 的 125M–13B 平均 perplexity；Table 9 的 batch size 与学习率；65B 常规 16-bit 微调 >780 GB、33B 需分页优化器才进 24GB；batch size 16 时分页优化器与普通优化器同速；NF4 需挂到所有线性层才复现全量微调。
- Hugging Face，*bitsandbytes* 文档，<https://huggingface.co/docs/bitsandbytes/index>：8-bit 优化器、LLM.int8() 与 QLoRA（4-bit 量化 + 低位适配器）三个特性的官方描述，是工程落地的实现参考。
- 延伸来源说明：NF4 的分位数公式与完整码表、双重量化的 0.373 bit 推导、显存分解与正态性检验来自上面的论文（作业单标注为「延伸」）；NF4 与均匀 INT4 的 MSE 对照、各规模显存与适配器参数的表格是本文按公式自算的复算结果，口径已在正文逐项写明。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
