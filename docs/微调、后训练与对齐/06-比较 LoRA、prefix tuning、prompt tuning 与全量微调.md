---
type: question
id: finetuning-06
topic: 微调、后训练与对齐
order: 6
question: 比较 LoRA、prefix tuning、prompt tuning 和 full fine-tuning。分别在什么场景下选择它们？
question_en: Compare LoRA, prefix tuning, prompt tuning and full fine-tuning. When do you pick each?
asked_at: [Sarvam AI, Apple]
level: 进阶
tags: [peft, prefix-tuning, prompt-tuning, 全量微调]
sources:
  - title: How does Fine-Tuning work?
    url: https://outcomeschool.com/blog/how-does-fine-tuning-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: How does Prefix Tuning work?
    url: https://outcomeschool.com/blog/how-does-prefix-tuning-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Prefix-Tuning: Optimizing Continuous Prompts for Generation（延伸）
    url: https://arxiv.org/abs/2101.00190
    author: Li & Liang
    published: 2021-01-01
  - title: P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks（延伸）
    url: https://arxiv.org/abs/2110.07602
    author: Liu et al. (ACL 2022)
    published: 2021-10-14
related: [finetuning-04, finetuning-05, finetuning-07, finetuning-08]
updated: 2026-09-28
---

## 一句话答案

> 四个方案不是并列的四选一，而是一棵树：**全量微调**更新全部参数；**参数高效微调（PEFT）** 冻结底座，分两条路线——**权重空间**（LoRA 学低秩增量 $\Delta W = BA$，训练后能合并回权重，推理零额外延迟）与**激活空间 / 软提示**（prefix tuning 往每层 attention 的 K/V 前拼可学习前缀，prompt tuning 只在输入 embedding 前拼可学习向量）。
> 选择规则是：改行为、数据少 → 软提示或小 $r$ LoRA；注新知识、分布外 → 全量微调或大 $r$ LoRA；多租户/多适配器 → LoRA；显存不够 → QLoRA；单任务极致质量且预算充足 → 全量微调；要快速并行试很多方向 → PEFT。
> 工程上 LoRA 已经是事实标准，最直接的原因是它**推理路径与底座完全一致**：合并后既不加延迟，也不和 prefix cache、KV cache 管理打架；再叠加适配器只占几十 MiB、部署工具链支持最全。

## 面试官在考什么

- 能否先分类再比较，而不是把四个名词平铺成一张「优缺点表」。分类维度是「改哪里的参数」：权重空间 vs 激活空间。
- 能否给出每一种方法的**可训练参数量公式**，并代入具体模型算出数字（软提示的参数量口径和 LoRA 完全不同，前者是 $T\cdot 2LH$，后者是 $r\cdot(d_{in}+d_{out})$ 逐模块累加）。
- 是否清楚**推理期开销**的差别：LoRA 合并后为零，软提示多出 $T$ 个虚拟 token，注意力与 KV cache 都要为它付费。
- 能否把「表达能力上限」和「抗灾难性遗忘」放在同一个坐标里权衡：参数越少越不容易遗忘，也越难注入新知识。
- 能不能给**部署视角**的答案：一个底座挂多个适配器怎么路由、软提示要随请求携带意味着什么、为什么软提示和 prefix cache 天然冲突。

常见错误答案：

- 说「LoRA 也是 prompt，只是拼在权重上」。LoRA 改的是权重，不是激活；它和软提示是两条不同的路线。
- 说「PEFT 一定更快」。它省的是**显存与存储**，不是时间：反向传播仍要穿过全部层，计算量基本不变，甚至因为多出的矩阵乘与 kernel 效率而略慢；同源的错误还包括把 prefix tuning 和 prompt tuning 混为一谈、把 P-Tuning v2 说成「换了个 prompt 模板」。

## 原理与推导

### 1. 分类树：参数改在哪里

| 分支 | 机制 | 代表方法 |
| --- | --- | --- |
| 全量微调 | 对每个权重矩阵 $W$ 求梯度、全部更新 | full fine-tuning |
| PEFT · 权重空间 | 冻结 $W_0$，只学一个加性增量 $\Delta W$ | LoRA、Adapter（瓶颈层） |
| PEFT · 激活空间 | 冻结全部权重，只学注入到激活/attention 的向量 | prefix tuning、prompt tuning、P-Tuning v2 |

关键区别在「学习信号作用在哪里」：权重空间的增量是可持久化的参数，训练完能落盘成一个小文件、甚至合并进底座；激活空间的增量是**输入的一部分**，每次前向都必须把它送进去。

### 2. LoRA：低秩增量，可合并

冻结 $W_0 \in \mathbb{R}^{d \times k}$，把它要学的更新约束成低秩：

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA,\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)$$

$\frac{\alpha}{r}$ 是缩放因子（$\alpha$ 是常数超参），让改变 $r$ 时不必重新调学习率。可训练参数量：

$$N_{\text{LoRA}} = r \cdot (d + k) \quad \text{（每个被适配的矩阵）}$$

初始化上 $A$ 取高斯随机、$B$ 置零，训练起点 $\Delta W = 0$，第一步的函数与底座完全一致；这两处初始化各自解决什么问题见 [[finetuning-04]]。

推理时可以精确合并，因为矩阵乘法对加法分配：

$$(W_0 + BA)x = W_0 x + B(Ax)$$

合并后是**一次** $d\times k$ 的矩阵乘，和原始模型的前向完全一致，因此额外延迟为 0。注意 $B$ 的秩最多为 $r$，但 $W_0$ 通常是满秩的，所以 $W_0 + BA$ 依然满秩——低秩约束限制的是**变化量的大小**，不是最终权重的表达力。

### 3. prefix tuning：每层 attention 的 K/V 前拼前缀

在每一层 $\ell$，把 $T$ 个可学习的「虚拟 token」拼到 key 和 value 序列前面：

$$K' = [\,K_{\text{prefix}}^{(\ell)};\, K\,],\qquad V' = [\,V_{\text{prefix}}^{(\ell)};\, V\,],\qquad Q \text{ 不变}$$

前缀**只提供 K、V，不提供 Q**，因为「提问」永远来自真实 token。注意力因此变成：

$$\text{Attn} = \text{softmax}\!\left(\frac{Q [K_{\text{prefix}}; K]^{\top}}{\sqrt{d_{head}}}\right)[V_{\text{prefix}}; V]$$

每层都有一套**独立训练**的前缀参数（不是把同一个前缀逐层传下去），所以总可训练参数量是：

$$N_{\text{prefix}} = T \cdot 2 \cdot L \cdot H = 2TLH$$

因子 2 来自 key 和 value 各一份。原文用一个小 MLP 从更小的向量重新参数化出 K/V（提高训练稳定性），训练完丢掉 MLP、只保留最终的 K/V，对应 `peft` 里的 `prefix_projection`（默认关闭）。

论文口径（GPT-2 做 table-to-text、BART 做 summarization）：只学约 **0.1%** 的参数，全量数据下性能与全量微调相当，低资源场景优于全量微调，对训练中未出现的主题外推更好。

### 4. prompt tuning：只在输入端拼软提示

只在前向的入口拼 $T$ 个可学习 embedding，之后所有层都不再注入任何东西：

$$H^{(0)} = [\,E_{\text{soft}};\, E_{\text{tokens}}\,],\qquad N_{\text{prompt}} = T \cdot H$$

$H$ 是 hidden size。它比 prefix tuning 小 $2L$ 倍——差的是「每层都要一份 K/V」。代价是引导力弱：软提示只能通过整个网络间接影响每一层，梯度路径长。

原文（Lester et al., EMNLP 2021）的结论是「规模越大越有效」：在 T5 上，模型超过百亿参数后 prompt tuning 才追平全量微调，中小编号模型上明显落后。P-Tuning v2（Liu et al., ACL 2022）把软提示做成「**深层 prompt**」——即把前缀注入每一层（就是 prefix tuning 的 NLU 版本）——在 NLU 任务上以 **0.1%–3%** 的可调参数做到与全量微调可比，跨模型规模与任务都成立，包括序列标注这类难任务。

### 5. 全量微调：上限最高，代价也最真实

所有参数都要梯度、优化器状态和一份可回滚的副本。按最常用的混合精度 Adam 口径（ZeRO 的 $2\Psi + 2\Psi + K\Psi$，$K = 12$，完整分解见 [[finetuning-09]]），**每参数 16 bytes**：

$$\underbrace{2}_{\text{bf16 权重}} + \underbrace{2}_{\text{bf16 梯度}} + \underbrace{4+4}_{\text{Adam } m,v \text{ fp32}} + \underbrace{4}_{\text{fp32 master 权重}} = 16\ \text{bytes/param}$$

此外还有激活值、临时缓冲与碎片化。Llama-3-8B 按官方 config 逐项加总是 8.03B 参数（$L=32$、$d=4096$、$d_{ff}=14336$、$V=128256$，embedding 与 lm_head 不共享），单是这一项就是 $8.03\times10^9 \times 16 \approx 128$ GB $= 120$ GiB，单张 80 GB 卡放不下。全部权重都被梯度覆盖，也是灾难性遗忘最严重的方案。

## 数值与代码验证

### 对比表（全部为自己的复算，口径写在表下）

| 维度 | full fine-tuning | LoRA | prefix tuning | prompt tuning |
| --- | --- | --- | --- | --- |
| 改什么 | 全部 $W$ | 每模块 $r(d+k)$ | 每层 K/V 前缀 $2TLH$ | 输入软提示 $TH$ |
| 在 Llama-3-8B（$d=4096$、$L=32$、共 8.03B）上的复算值 | $8.03\times10^9$（100%） | $1.68\times10^7$（0.209%，$r=16$ 适配 4 个投影） | $5.24\times10^6$（0.065%，$T=20$） | $8.19\times10^4$（0.0010%，$T=20$） |
| 训练显存：可训练参数的 16 B/param | $\approx 120$ GiB | 256 MiB | 80 MiB | 1.25 MiB |
| 推理额外延迟 | — | **0**（合并后） | $T$ 个虚拟 token 的 attention 与 KV cache | 同左，且只在前缀长度上 |
| 多任务切换 | 换整个模型（几十 GiB） | 换适配器（几十 MiB），可合并 | 换前缀（MiB 级），需随请求路由 | 同左 |
| 表达能力上限 | 最高 | 高（受 $r$ 限制） | 中（只动注意力 K/V） | 最低（只动输入） |
| 抗灾难性遗忘 | 最弱 | 强（底座冻结） | 强 | 强 |
| 工具链成熟度 | 高 | **最高**（PEFT / vLLM / SGLang 原生支持） | 中（`peft` 支持，服务端支持少） | 中 |

口径说明：8.03B 是 Llama-3-8B 的总参数（8,030,261,248，含每层 RMSNorm 与 final norm 的 0.27M；其中 embedding 与 lm_head 各约 0.53B，非嵌入部分约 6.98B），表里的占比都以它为分母。训练显存那一行只把**可训练参数**按 16 bytes/param 折算（bf16 权重 2 + bf16 梯度 2 + Adam $m,v$ fp32 8 + fp32 master 4），不含激活值与碎片；PEFT 的底座权重仍要按推理精度常驻显存，这部分（8.03B 按 bf16 约 15 GiB）在四种方案里是一样的，所以表里不重复计。全量微调还有一个 PEFT 没有的成本：要额外存一份完整权重才能回滚。

### 复算一：软提示参数量与论文的 0.1% 口径

按资料给的例子（$H=2048$、$L=24$、$T=20$、模型 1.3B）：

| 量 | 计算 | 结果 | 占 1.3B |
| --- | --- | --- | --- |
| prefix tuning | $20 \times 2 \times 2048 \times 24$ | 1,966,080 | 0.1512% |
| prompt tuning | $20 \times 2048$ | 40,960 | 0.0032% |
| 两者之比 | $2L$ | 48.0× | — |

`peft` 在 GPT-2（$H=768$、$L=12$）上打印的 `trainable params: 368,640 || all params: 124,808,448 || trainable%: 0.2954` 与公式完全吻合：$20\times2\times768\times12 = 368{,}640$。分母 124,808,448 = gpt2 自身 124,439,808 + 前缀 368,640——`peft` 把新加的适配器也算进 `all params`，引用这个百分比时口径要写清。

把 1.3B 模型的可训练预算压到论文说的 0.1%，反解 $T$：$T = \frac{0.001 \times 1.3\times10^9}{2\times2048\times24} \approx 13.2$，即约 **13 个虚拟 token** 就够到这个量级。论文的 0.1% 也能复算出来：按 GPT-2 large（$H=1280$、$L=36$、774M）取 $T=10$，$10\times2\times36\times1280 = 921{,}600$，占 $0.119\%$。所以「0.1%」不是特殊设计，而是「亿级模型 + $T$ 取十几个」的自然结果；换成 1.3B 的算例就是 0.15%。prefix tuning 原文做 summarization 时最优 $T$ 约 200；任务越难越需要更长的前缀，而且 $T$ 不是越大越好，超过某个点质量会回落。

### 复算二：LoRA 参数量，以及为什么它便宜得几乎免费

| 配置 | 计算 | 可训练参数 | 占 8.03B |
| --- | --- | --- | --- |
| $r=16$，q/k/v/o 四个投影 | $16\times(4096+4096)\times32\times4$ | 16,777,216 | 0.209% |
| $r=8$，只适配 q、v | $8\times(4096+4096)\times32\times2$ | 4,194,304 | 0.052% |

按 bf16 存储，最贵的那一档适配器也只有 32 MiB，优化器状态 256 MiB；而全量微调的优化器状态是 120 GiB。**约 480 倍的差距**，而且全量微调要额外存一份完整模型才能回滚。

### 复算三：软提示对 KV cache 与上下文的影响

前缀 $T=100$、Llama-3-8B（$L=32$、$H_{kv}=8$、$d_{head}=128$、bf16）下：

$$M_{\text{prefix}} = T \cdot 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot \text{bytes} = 100\times2\times32\times8\times128\times2 = 13{,}107{,}200\ \text{B} = 12.5\ \text{MiB}$$

| 场景 | 每 token KV | 8k 上下文的 KV cache | 加 100 token 前缀后 | 增幅 |
| --- | --- | --- | --- | --- |
| Llama-3-8B（32 层） | 128 KiB | 1.00 GiB | 1.01 GiB | +1.22% |
| Llama-3-70B（80 层） | 320 KiB | 2.50 GiB | 2.53 GiB | +1.22% |

增幅恰好是 $T/S = 100/8192 = 1.22\%$，与模型无关——前缀的代价就是**从上下文预算里切掉 $T$ 个位置**，同时让注意力多算 $T$ 列。长上下文场景（$S$ 大）里这个比例不显著，短上下文高并发场景里它直接变成吞吐损失。

更要紧的是它和 prefix caching 的冲突：prefix cache 的命中依赖「**token id 序列完全相同的前缀**」（见 [[inference-serving-05]]），而软提示是 embedding 空间里的向量，没有对应的 token id。每个适配器各有一套前缀，缓存键就得带上适配器标识，`peft` 之外的服务端基本不认这套——这就是「把软提示换成一个普通 LoRA，部署链路会立刻变简单」的具体原因。

### 代码验证：合并恒等式与参数量

```python
import numpy as np

rng = np.random.default_rng(0)
d, k, r = 512, 512, 8
W0 = rng.normal(0, 0.02, (d, k))
B  = rng.normal(0, 0.02, (d, r))
A  = rng.normal(0, 0.02, (r, k))
alpha_over_r = 16 / r           # LoRA 缩放因子
x  = rng.normal(0, 1, (k, 4))   # 4 个 token

# 未合并：y = W0 x + (alpha/r) B (A x)
y_unmerged = W0 @ x + alpha_over_r * (B @ (A @ x))
# 已合并：W' = W0 + (alpha/r) B A，然后一次矩阵乘
W_merged   = W0 + alpha_over_r * (B @ A)
y_merged   = W_merged @ x

print("max |y_unmerged - y_merged| =", np.abs(y_unmerged - y_merged).max())
print("rank(W0) =", np.linalg.matrix_rank(W0), " rank(B@A) =", np.linalg.matrix_rank(B @ A),
      " rank(W') =", np.linalg.matrix_rank(W_merged))
```

输出（本机复跑）：

```text
max |y_unmerged - y_merged| = 2.220446049250313e-15
rank(W0) = 512  rank(B@A) = 8  rank(W') = 512
```

（误差的具体数值会随 BLAS 实现与随机种子漂移，量级始终在 fp64 机器精度 $2.2\times10^{-16}$ 的十几倍以内。）

三点结论：① 合并是**数值精确**的，误差只有浮点舍入量级，所以「LoRA 合并后不掉点」不是近似；② $\Delta W$ 秩为 8，但合并后的 $W'$ 秩仍是 512——低秩约束限制的是增量，不是最终权重的秩；③ 因此适配器的表达力上限由 $r$ 决定：想让模型学会一个底座完全没有的新能力，$r$ 太小会学不动，这属于 [[finetuning-04]] 的秩选择问题。

## 常见追问

- **追问**：prefix tuning 和 prompt tuning 的差别到底在哪一层？
  - 要点：prompt tuning 只改**输入 embedding**（第 0 层入口），之后所有层看到的就是普通激活；prefix tuning 在**每一层**的 attention 里注入独立的可学习 K/V。参数差 $2L$ 倍，引导强度差一个数量级。
- **追问**：为什么 prompt tuning 在小模型上明显差？P-Tuning v2 的动机是什么？
  - 要点：Lester et al. 的消融结论是「随规模变大才追平」——模型超过百亿参数后 prompt tuning 才与全量微调可比，中小模型上差距明显。原因是输入端的 $TH$ 个参数要通过整个网络间接影响每层行为，容量与梯度路径都不够。P-Tuning v2 的答案是把软提示变成**深层 prompt**（每层都注入，即 prefix tuning 的 NLU 版本），并以 0.1%–3% 的参数在跨规模、跨任务上做到与全量微调可比。
- **追问**：Adapter（瓶颈层）和 LoRA 的推理延迟差异？
  - 要点：Adapter 在层间插入下投影—非线性—上投影，**串在计算路径上**，除非把权重合并（有非线性，通常做不到精确合并），否则每层都要多算；LoRA 是线性加性增量，可以精确合并回 $W_0$，推理路径与底座逐比特一致。这是 LoRA 在服务端胜出的直接原因。
- **追问**：软提示的「前缀占用上下文」在什么情况下是实打实的成本？
  - 要点：两个口径——上下文预算被切掉 $T$ 个位置（$T=100$、8k 窗口就是 1.22%）；KV cache 多 $T$ 个位置（同样 1.22%，与模型规模无关）。高并发短上下文时是吞吐损失；长上下文时占比小但注意力仍多算 $T$ 列。再加上与 prefix cache 不兼容，实际损失通常大于这个比例。
- **追问**：PEFT 能不能和 RLHF / DPO 叠加？
  - 要点：可以，`peft` 与 TRL 的 `DPOTrainer` / PPO 流程是兼容的，做法是给策略模型挂适配器、参考模型用「关掉适配器」的同一个底座或另一个冻结适配器。坑有三处：DPO 的隐式参考模型来自底座本身，实践中 $\beta$ 要和适配器缩放 $\alpha/r$ 一起调（[[finetuning-02]]）；PPO 的四个模型里如果策略挂了 LoRA，value head 通常仍是新加的满秩头；RL 阶段奖励信号弱、梯度噪声大，$r$ 太小时更容易训不动，实践中 $r$ 往往比 SFT 阶段更大。
- **追问**：什么时候 PEFT 反而不该选？
  - 要点：需要注入大量新知识、新语言或全新能力（分布外）时，冻结底座把可学习空间限制得太小，$r$ 调大也只是逼近全量微调，不如直接全量微调或换更大的底座；数据量足够（十万条量级）且算力充足时，全量微调的上限更高。判断框架见 [[finetuning-08]]。

## 公司变体

- **Sarvam AI**：这家公司做的是印度语言与语音模型，公开材料里最常被强调的是**低资源语言适配**：某种语言可能只有几千句干净文本。因此这道题在这里偏「有限数据下的适配策略」——soft prompt / 小 $r$ LoRA 在低资源下比全量微调更稳，因为冻结底座天然抑制灾难性遗忘，不会把高资源语言的能力一起带崩；也要能讨论 tokenizer 的 fertility 如何影响适配难度、以及多语言共享底座时「一个语言一个适配器」的路由与存储成本。偏工程与数据约束，而不是纯推导。
- **Apple**：端侧与私有云计算视角，关键词是**内存与功耗预算**。同一底座要支撑十几个功能（摘要、改写、回复建议、语气调整），每个功能挂一个适配器，而端侧内存与存储首先要留给底座本身，于是「多适配器共用一个底座、按需加载或合并」几乎是唯一解，问题会落在适配器怎么按请求路由、能不能在加载时就合并掉、以及 4-bit 底座上挂 LoRA 的精度与延迟取舍（[[finetuning-05]]）。对软提示的追问通常也最直接：多出来的前缀要占多少 KV cache、端侧还能不能保住上下文长度。
- 两家的共同点：都不会停在「LoRA 便宜」这个结论上，一定会追一个具体数字——适配器多大、前缀吃多少上下文、显存账怎么算。

## 相关题目

- [[finetuning-04]]：LoRA 的分解数学、秩 $r$ 与缩放 $\alpha$ 怎么选——本篇只用到结论。
- [[finetuning-05]]：QLoRA 如何在 4-bit 底座上挂 LoRA，以及量化带来的取舍。
- [[finetuning-07]]：灾难性遗忘的机制与缓解手段，PEFT 只是其中一层。
- [[finetuning-08]]：Prompting、RAG 还是 fine-tuning 的决策框架与成本权衡。
- [[inference-serving-05]]：prefix caching 的命中条件，以及它为什么和软提示天然冲突。
- [[inference-serving-11]]：多适配器并发、按请求路由在 vLLM / SGLang / TensorRT-LLM 里的支持现状。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《How does Fine-Tuning work?》，[链接](https://outcomeschool.com/blog/how-does-fine-tuning-work)：全量微调与 LoRA 的流程、逐步数值例子、「数据质量优先于数量」的实践建议。
- Amit Shekhar (Outcome School)，《How does Prefix Tuning work?》，[链接](https://outcomeschool.com/blog/how-does-prefix-tuning-work)：prefix tuning 的逐层 K/V 注入机制、虚拟 token 的双份计数、1.3B 与 GPT-2 的参数量算例、与 prompt tuning 的对比、前缀长度与注意力开销的讨论。
- Xiang Lisa Li, Percy Liang，《Prefix-Tuning: Optimizing Continuous Prompts for Generation》(2021)，[arXiv:2101.00190](https://arxiv.org/abs/2101.00190)（延伸）：0.1% 参数、全量数据下可比全量微调、低资源更优、未见主题外推更好的论文结论，以及 reparameterization 训练技巧。
- Xiao Liu et al.《P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks》(ACL 2022)，[arXiv:2110.07602](https://arxiv.org/abs/2110.07602)（延伸）：prompt tuning 在正常规模模型上表现不佳、难任务缺乏通用性，以及深层 prompt tuning 以 0.1%–3% 参数追平全量微调的结论。
- 延伸来源补充的是论文口径的结论与机制。其中 prompt tuning「随规模变大才追平」这一条来自延伸检索到的 Lester et al.《The Power of Scale for Parameter-Efficient Prompt Tuning》(EMNLP 2021)，[arXiv:2104.08691](https://arxiv.org/abs/2104.08691)，不在作业单的参考源里，此处单独标注；第 3 节的公式推导与第 4 节的 P-Tuning v2 数字来自上面两条 arXiv 延伸来源。本篇中所有参数量、显存与 KV cache 数字均为按公式自行复算，口径已在正文标明。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
