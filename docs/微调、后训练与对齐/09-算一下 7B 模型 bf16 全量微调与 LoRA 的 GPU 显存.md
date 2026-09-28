---
type: question
id: finetuning-09
topic: 微调、后训练与对齐
order: 9
question: 算一下用 Adam 以 bf16 全量 fine-tuning 一个 7B 模型所需的 GPU 显存。换成 LoRA 呢？
question_en: Compute the GPU memory for bf16 full fine-tuning of a 7B model with Adam. What about LoRA?
asked_at: [Mistral AI, Hugging Face]
level: 进阶
tags: [显存, adam, zero, lora]
sources:
  - title: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（延伸）
    url: https://arxiv.org/abs/1910.02054
    author: Rajbhandari et al.
    published: 2019-10-04
  - title: LoRA: Low-Rank Adaptation of Large Language Models（延伸）
    url: https://arxiv.org/abs/2106.09685
    author: Hu et al.
    published: 2021-06-17
  - title: Reducing Activation Recomputation in Large Transformer Models（延伸）
    url: https://arxiv.org/abs/2205.05198
    author: Korthikanti et al. (Megatron)
    published: 2022-05-10
related: [finetuning-04, finetuning-05, inference-serving-07, inference-serving-08]
updated: 2026-09-28
---

## 一句话答案

> 按五块算：权重、梯度、优化器状态、激活、临时与碎片。前三块在混合精度 Adam 下是固定开销，每参数 $2 + 2 + 4 + 4 + 4 = 16$ bytes，7B 就是 **112 GB**（104.3 GiB），与 batch、序列长度无关——单卡 80 GB 只算这一块就已经超了。
> 加上激活（LLaMA-7B 形状、$b=1$、$s=2048$、选择性重算后 $34sbhL \approx 9.1$ GB），单卡全量微调约 124 GB；要在 80 GB 卡上真正训起来只能分片：8 卡 ZeRO-3 把固定开销摊成 2 bytes/param（14 GB/卡），合计约 26 GB/卡。
> 换 LoRA：底座 bf16 14 GB 冻结，无梯度、无优化器状态；$r=16$ 挂在 32 层的 q/k/v/o 上只有 16.8M 可训练参数，状态约 0.27 GB，合计约 26 GB，单卡 80 GB 宽裕；换成 NF4 底座（QLoRA）约 16 GB，24 GB 卡可跑。
> 区分点是这句：**LoRA 省的是梯度与优化器状态，不是权重**——底座无论 16-bit 还是 4-bit 都必须留在显存里。

## 面试官在考什么

- 能不能把「显存」拆成互不重叠的五块，并指出哪几块与 batch/序列长度无关。只报一个笼统数字的人会被立刻追问到崩。
- 记不记得混合精度 Adam 的 16 bytes/param，以及它的出处口径：ZeRO 把参数与优化器状态叫 model states（$2\Psi$ 参数 + $2\Psi$ 梯度 + $K\Psi$ 优化器状态，$K = 12$），激活、临时缓冲与碎片叫 residual states。
- 会不会算激活：$34sbh + 5as^2b$，尤其是知不知道 $5as^2b$ 是 softmax、softmax dropout 与 attention dropout 输出这三项，随 $s^2$ 增长。
- 知不知道降显存的手段与代价：梯度检查点（把 $L$ 层分段，激活从 $O(N)$ 降到约 $O(\sqrt{N})$，多一次前向）、选择性重算（只重算注意力，激活降 65%–70%、FLOPs 只加 1.6%–2.7%）、FlashAttention（不物化 $s^2$ 矩阵）、ZeRO 三阶段的通信代价、8-bit Adam 与 CPU offload。
- LoRA 的账要算到「可训练参数量 × bytes/param」，并且能说清省的是哪一块、为什么实测收益远小于 8 倍。

常见错误答案：

- 「7B × 2 字节 = 14 GB 就够了」。把 bf16 当成「每参数 2 字节」是这题最常见的错：梯度、fp32 主权重、Adam 一阶/二阶动量都不见了。
- 「LoRA 省权重显存，所以 LoRA 能让 70B 单卡微调」。LoRA 不减少冻结权重的占用，也不减少激活；让 70B 进单卡的是 4-bit 量化底座（[[finetuning-05]]）。
- 「激活和 batch 无关」或者「attention 的 $s^2$ 中间张量不重要」。$s=2048$、$d_{head}=128$ 时它已经是其余激活之和的 2.35 倍。

## 原理与推导

### 0. 五块骨架与三个口径

ZeRO 把训练显存分成 model states（参数、梯度、优化器状态）与 residual states（激活、临时缓冲、碎片）。这里把激活单独拉出来，因为它是唯一随负载增长的一块：

| 块 | 随什么变 | 7B 全量（bf16 + Adam） | 7B LoRA（$r=16$，q/k/v/o） |
| --- | --- | --- | --- |
| 权重 | $N$ | 14 GB | 14 GB（冻结，仍占显存） |
| 梯度 | 可训练参数量 | 14 GB | 0.03 GB |
| 优化器状态 | 可训练参数量（fp32） | 84 GB（主权重 28 + $m$ 28 + $v$ 28） | 0.20 GB |
| 激活 | $b, s, L, h$，朴素实现还有 $s^2$ | 9.1 GB（$b=1, s=2048$，选择性重算） | 9.1 GB（完全相同） |
| 临时与碎片 | 框架开销、并行度 | 2–4 GB | 2–3 GB |
| **合计** | | **约 124 GB** | **约 26 GB** |

三个必须先钉住的口径：

| 口径 | 取法 | 差多少 |
| --- | --- | --- |
| GB 与 GiB | 参数量 × 字节数用十进制 GB（$10^9$）；`nvidia-smi` 与分配器用 GiB（$2^{30}$） | 112 GB = 104.3 GiB，7.4% |
| 名义 7B 与真实参数量 | 按 LLaMA-7B config（$L=32$、$d=4096$、$d_{ff}=11008$、$V=32000$、不 tie embedding）逐项加总是 6.738B | 固定开销 112 GB 对 107.8 GB，3.9% |
| 是否含 fp32 主权重 | ZeRO 口径含（$K=12$）；只算 bf16 参数/梯度 + fp32 Adam 状态是 12 bytes/param | 少 28 GB |

第三个口径是最容易被追问的：bf16 只有 8 位尾数，直接用 bf16 参数做更新会丢精度，所以主流实现（DeepSpeed、HF Trainer）都保留 fp32 主权重。**报 112 GB 还是 84 GB 都可以，但必须说清是哪个口径**；拿「7B bf16 微调只要 14 GB」当结论是错的。

### 1. 固定开销：16 bytes/param 的来历

$$M_{\text{states}} = 2N + 2N + 4N + 4N + 4N = 16N\ \text{bytes}$$

五项依次是 bf16 权重、bf16 梯度、fp32 主权重、Adam 一阶动量 $m$ 与二阶动量 $v$，各 2、2、4、4、4 bytes/param。代 $N = 7\times 10^9$：$16 \times 7\times 10^9 = 1.12\times 10^{11}$ B = **112 GB = 104.3 GiB**。ZeRO 论文给的对照是：标准 DP 下 7.5B 模型要 120 GB，GPT-2 1.5B 要 24 GB（而 fp16 权重只有 3 GB）。$K = 12$ 就是「主权重 4 + 动量 4 + 方差 4」，$2\Psi + 2\Psi + K\Psi = 16\Psi$。

这一块的三个性质要一起说：**与 batch、序列长度、上下文长度完全无关**；**只随参数量线性增长**；**A100/H100 的 80 GB 级别单卡放不下 7B 的全量微调**（104.3 GiB > 80 GiB，换哪种单位都不改变结论）。再补一句量级感：70B 的同一笔账是 1120 GB，所以 70B 全量微调必须多机多卡加 ZeRO-3。

### 2. 激活：$34sbh + 5as^2b$ 与三种降显存手段

Megatron 论文按「前向中产生、反向中需要」的口径逐张量记账（$s$ 序列长、$b$ micro-batch、$h$ 隐藏维、$a$ 头数、$L$ 层，元素 2 字节、dropout mask 1 字节）：

- attention 块 $= 11sbh + 5as^2b$：QKV 共享输入 $2sbh$、$QK^\top$ 要存的 Q 与 K $4sbh$、输出投影输入 $2sbh$、attention dropout mask $sbh$、V $2sbh$，这五项是 $11sbh$；softmax 输出 $2as^2b$、softmax dropout mask $as^2b$、attention over V 的 dropout 输出 $2as^2b$，这三项合起来是 $5as^2b$。
- MLP 块 $= 19sbh$：两个线性层输入 $2sbh + 8sbh$、GeLU 输入 $8sbh$、dropout mask $sbh$。
- 两个 layer norm 的输入 $= 4sbh$。

合计每层 $sbh(34 + 5as/h)$，其中 $5as/h$ 可以改写成：

$$5\frac{as}{h} = \frac{5s}{d_{head}}$$

只要 $s > \frac{34}{5}d_{head} \approx 6.8\,d_{head}$，注意力项就超过其余所有激活之和。$d_{head}=128$ 时阈值只有 870 个 token。$s=2048$ 时 $5as/h = 80$，注意力项占全部激活的 70.2%；$s=8192$ 时占 90.4%。这就是「长上下文训练的显存瓶颈是注意力」的定量版本。

三种手段对应三个不同的项：

1. **梯度检查点（全量重算）**：只存每层输入，把 $L$ 层按 $\sqrt{L}$ 分段、每段留一个检查点，激活从 $O(N)$ 降到约 $O(\sqrt{N})$（ZeRO 论文的措辞是「约等于总激活的平方根」，$N$ 指不重算时的激活量），代价是多一次前向：理论 FLOPs +33%，Megatron 实测执行时间开销 **30%–40%**。
2. **选择性重算**：只重算 attention 块里那三张 $s^2$ 矩阵，per-layer 从 $sbh(34 + 5as/h)$ 降到 $34sbh$，与 $a$ 无关、对 $s$ 线性；论文给的效果是激活降 5×、重算的时间开销回收 90% 以上，GPT-3（$a=96$、$s=2048$、$h=12288$）省 70% 激活只加 2.7% FLOPs，MT-NLG 省 65% 只加 1.6%。530B 模型在 2240 张 A100 上 MFU 从 42.1% 提到 54.2%。
3. **FlashAttention**：从实现上就不物化 $s^2$ 矩阵（[[llm-internals-05]]），等价于把 $5as^2b$ 这一项直接删掉。现代微调栈的默认组合是 FlashAttention + 选择性重算，而不是全量重算。

口径提醒：Megatron 的 34 是按 $4h$ 的 GeLU MLP 推的，而且每一项都含 dropout mask。LLaMA 是 SwiGLU（$d_{ff} = 2.6875h$）且 dropout 为 0：把三个 $d_{ff}$ 宽的中间张量（gate 预激活、up 输出、down 的输入）逐项相加是 $2sbh + 6sbh\frac{d_{ff}}{h} = 18.1sbh$，比 $19sbh$ 少的那一项正是 LLaMA 里不存在的 dropout mask。两边口径统一后再比，含 mask 的 SwiGLU 是 $19.1sbh$、总系数 34.1，与 34 只差 0.4%——SwiGLU 并没有把系数降下来，「总系数 33.1」是混着 mask 口径比出来的假精度。用 34 代入 LLaMA 形状不会低估：对无 dropout 的 LLaMA（真实系数约 32.1）它反而多算约 6%，所以本节的 9.1 GB 是偏保守的量级。

### 3. ZeRO 分片：把 16 bytes/param 摊到 $N_d$ 张卡

ZeRO 的三个阶段依次切优化器状态、梯度、参数（$\Psi$ 是参数量）：

$$16\Psi \;\xrightarrow{P_{os}}\; 4\Psi + \frac{K\Psi}{N_d} \;\xrightarrow{P_{os+g}}\; 2\Psi + \frac{14\Psi}{N_d} \;\xrightarrow{P_{os+g+p}}\; \frac{16\Psi}{N_d}$$

通信代价的论文口径：ZeRO-1 与 ZeRO-2 与基线 DP 相同，**ZeRO-3 是基线 DP 的 1.5×**（梯度 reduce-scatter 2 bytes/param + 前向 all-gather 2 + 反向 all-gather 2 ≈ 6 bytes/param，对应 7B 每卡每步约 42 GB）。关键补充：**ZeRO-3 只切 model states，不切激活**——激活要靠 ZeRO-R 的分片式激活检查点或序列并行（[[inference-serving-07]]），所以下一节的 9.1 GB 在 ZeRO-3 下仍然每卡一份。

| 配置（7B，$b=1$、$s=2048$） | 固定 B/param | 固定 GB/卡 | +激活 9.1 +缓冲 3 | 80 GB |
| --- | --- | --- | --- | --- |
| 单卡不分片 | 16 | 112 | 124.1 | 放不下 |
| 2 卡 ZeRO-1 | 10 | 70 | 82.1 | 差 2 GB |
| 2 卡 ZeRO-2 | 9 | 63 | 75.1 | 可以 |
| 2 卡 ZeRO-3 | 8 | 56 | 68.1 | 可以 |
| 4 卡 ZeRO-3 | 4 | 28 | 40.1 | 宽裕 |
| 8 卡 ZeRO-1 | 5.5 | 38.5 | 50.6 | 宽裕 |
| 8 卡 ZeRO-2 | 3.75 | 26.25 | 38.4 | 宽裕 |
| 8 卡 ZeRO-3 | 2 | 14 | 26.1 | 宽裕 |

「8 卡 ZeRO-3 大约 2 bytes/param/卡、7B 约 14 GB/卡」就是这题要的那个数：$16/8 = 2$。剩下的 9.1 GB 激活与 2–4 GB 缓冲决定了它离 80 GB 还有多远。这里 3 GB 是自估的预留：CUDA context 与 cuBLAS/cuDNN handle 约 0.5–1 GB、ZeRO-3 的 all-gather 缓冲（单层参数 $7\text{B}/32 \times 2$ 字节 ≈ 438 MB，prefetch 两层接近 0.9 GB）、reduce-scatter 的 bucket 与碎片。

### 4. LoRA：参数量、状态，以及省下的到底是哪一块

对 $W_0 \in \mathbb{R}^{d_{out}\times d_{in}}$，LoRA 加 $A \in \mathbb{R}^{r\times d_{in}}$、$B \in \mathbb{R}^{d_{out}\times r}$，可训练参数是：

$$|\Theta| = r\,(d_{in} + d_{out})$$

$r=16$、挂 $L=32$ 层的 q/k/v/o（4 个 $4096\times4096$）：$4 \times 32 \times 16 \times (4096 + 4096) = 16{,}777{,}216 \approx 16.8\text{M}$，占 7B 的 0.24%。按同样的 16 bytes/param 口径，这部分的状态是 $16.8\text{M} \times 16 = 268$ MB。把 SwiGLU 的 gate/up/down 也挂上（$d_{ff}=11008$）是 40.0M（0.57%）→ 0.64 GB；$r=64$ 只挂 q/k/v/o 是 67.1M（0.96%）→ 1.07 GB。**LoRA 的超参怎么调都不会让这一块变成主要矛盾**，这就是结论。

于是 LoRA 的账是：底座 14 GB（bf16，冻结）+ 0.27 GB（adapter 的权重、梯度、fp32 Adam 状态）+ 9.1 GB（激活，与全量微调一模一样）+ 缓冲 ≈ **26 GB**。

省下的那一块要说准：全量微调的固定开销 112 GB 里，有 $14\Psi = 98$ GB 是梯度与优化器状态，LoRA 把它们变成了 0.27 GB；**冻结的 14 GB 底座一点没少，9.1 GB 激活也一点没少**。所以：

- 只算固定开销，$112 / 14.27 = 7.85\times$。
- 把激活算进去，$b=1$、$s=2048$ 时是 $124.1 / 26.4 = 4.7\times$；$b=8$ 时是 $188 / 90.3 \approx 2.1\times$——激活一旦主导，LoRA 的收益就被摊平。
- LoRA 论文自己的口径印证了这一点：它只说「最多降低 2/3」（GPT-3 175B 实测 VRAM 从 1.2 TB 到 350 GB，也就是 3.4×），并把原因写成「不需要为冻结参数存优化器状态」。350 GB 正好是 175B × 2 字节的底座权重——这个下界永远在。

**盈亏平衡点**（自算）：激活追上固定开销所需的 micro-batch，全量微调是 $112 / 9.13 = 12.3$ 条 2048-token 序列，而 LoRA 只要 $14.27 / 9.13 = 1.56$ 条。也就是说 LoRA 从第二个 micro-batch 起就进入「激活主导」区间，全量微调在 80 GB 卡上永远到不了那个点。这解释了为什么「LoRA 省 8 倍显存」这句话在实际训练里从来兑现不了。

### 5. 其他可以砍的块

- **8-bit Adam**：把优化器状态从 12 bytes/param 压到 6（fp32 主权重 4 + int8 $m$ 1 + int8 $v$ 1），7B 省 42 GB，固定开销 112 → 70 GB。配合梯度检查点与短序列，单张 80 GB 卡理论上能贴边跑 7B 全量微调，但没有任何余量。
- **CPU offload**：把 12 bytes/param（7B 是 84 GB）的优化器状态放到 host 内存。按「每步把状态搬上卡、更新完再搬回去」的粗口径，往返 168 GB，即使 Gen5 x16 实测约 50 GB/s 也要 3.4 s/step；ZeRO-Offload 一类实现让状态常驻 host、只搬梯度（2 bytes/param 下行）与更新后的参数（2 bytes/param 上行），单步流量降到 28 GB 上下，但仍然只适合离线任务。
- **QLoRA**：底座换成 NF4，4 bit 主体 0.5 B/param 加上每 64 参数一个 fp32 absmax（0.0625 B/param）≈ 0.5625 B/param → 3.94 GB；双层量化把量化常数也量化后还能再小一点。加上 adapter 状态与激活约 **16 GB**（[[finetuning-05]]）。

## 数值与代码验证

下面这段脚本把上面每一张表的数字都复算一遍，配置是 LLaMA-7B 口径（$L=32$、$d=4096$、$d_{ff}=11008$、$a=32$、$d_{head}=128$、$V=32000$），激活用 Megatron 的公式。

```python
GB, GiB = 10**9, 2**30
N = 7e9                                    # 名义 7B；按 config 逐项加总是 6.738B
L, d, ffn, a = 32, 4096, 11008, 32
d_head = d // a                            # 128

# 1) 固定开销：混合精度 Adam 的 model states（ZeRO 口径，bytes/param）
states = {"bf16 w": 2, "bf16 g": 2, "fp32 master": 4, "Adam m": 4, "Adam v": 4}
bpp = sum(states.values())
print(f"[1] 固定 {bpp} B/param = {bpp*N/GB:.0f} GB ({bpp*N/GiB:.1f} GiB)；冻结底座 {2*N/GB:.0f} GB")

# 2) ZeRO 分片：4Y + K*Y/Nd -> 2Y + 14Y/Nd -> 16Y/Nd
print("[2] ZeRO 每卡固定开销 GB：", {
    f"ZeRO-{st}@{nd}": round({0: 16.0, 1: 4 + 12/nd, 2: 2 + 14/nd, 3: 16/nd}[st] * N / GB, 2)
    for nd in (2, 8) for st in (0, 1, 2, 3)})

# 3) 激活：先算每层每 (token×样本) 的字节数 h*(34 + 5*a*s/d)，再乘 s*b*L；选择性重算去掉 5*a*s^2*b 项
def act_gb(s, b, t=1, selective=True):
    per_layer = d * (34 if selective else 34 + 5 * a * s / d)
    return per_layer * s * b * L / t / GB

print("[3] 激活 b=1（朴素, 选择性重算）GB：",
      {f"s={s}": (round(act_gb(s, 1, selective=False), 1), round(act_gb(s, 1), 2))
       for s in (512, 2048, 8192)})

# 4) LoRA 可训练参数：r*(d_in+d_out)；qkvo 是 4 个 d x d
def lora_params(r, mats):
    qkvo = L * 4 * r * 2 * d
    return qkvo if mats == "qkvo" else qkvo + L * 3 * r * (d + ffn)

print("[4] LoRA（参数量 M, 状态 GB @16B/param）：", {
    f"r={r},{m}": (round(lora_params(r, m) / 1e6, 2), round(lora_params(r, m) * 16 / GB, 3))
    for r in (16, 64) for m in ("qkvo", "all")})

# 5) 合成与盈亏平衡
act = act_gb(2048, 1)
lora_fixed = 2 * N / GB + lora_params(16, "qkvo") * 16 / GB
print(f"[5] 全量单卡 {bpp*N/GB + act:.0f} GB；LoRA {lora_fixed + act:.1f} GB；"
      f"QLoRA {0.5625*N/GB + lora_params(16,'qkvo')*16/GB + act:.1f} GB（均不含缓冲）")
print(f"[6] 盈亏平衡 micro-batch：全量 {bpp*N/GB/act:.1f} 条 vs LoRA {lora_fixed/act:.2f} 条；"
      f"固定开销比 {bpp*N/GB/lora_fixed:.2f}x")
```

运行结果：

| 输出 | 值 |
| --- | --- |
| 固定开销 | 16 B/param = 112 GB = 104.3 GiB；冻结底座 14 GB |
| ZeRO-3（2 / 8 卡） | 56.0 / 14.0 GB 每卡；ZeRO-1（8 卡）38.5 GB |
| 激活 $b=1$：$s=512$ | 朴素 3.6 GB → 选择性重算 2.28 GB |
| 激活 $b=1$：$s=2048$ | 朴素 30.6 GB → 选择性重算 9.13 GB |
| 激活 $b=1$：$s=8192$ | 朴素 380.1 GB → 选择性重算 36.51 GB |
| LoRA 参数量 / 状态 | r=16 只挂 qkvo：16.78M / 0.268 GB；r=16 挂全部 7 个投影：39.98M / 0.64 GB；r=64 只挂 qkvo：67.11M / 1.074 GB |
| 合成（不含缓冲） | 全量单卡 121 GB；LoRA 23.4 GB；QLoRA 13.3 GB |
| 盈亏平衡 | 全量 12.3 条 vs LoRA 1.56 条；固定开销比 7.85× |

几个需要对照源文的地方：Megatron 论文给的是 GPT-3 与 MT-NLG 的配置，7B 的激活数字是按它的公式代入 LLaMA-7B 形状得到的，不是论文原文数字；$s=8192$ 那一行的 380 GB 说明「不重算 + 不换注意力实现」在长序列下完全不可行，36.51 GB 才是可用的量级。另外 $b=1$、$s=2048$ 的 9.13 GB 是**每卡**的量：ZeRO-3 不切激活，所以每张卡在 14 GB 固定开销之外还要再加上这 9.13 GB。

## 常见追问

- **追问**：梯度检查点到底慢多少？
  - 要点：多一次前向，理论 FLOPs +33%（前向 : 反向 ≈ 1 : 2），Megatron 实测全量重算的执行时间开销是 30%–40%。代价随重算粒度变化：只重算 attention 的 $5as^2b$，GPT-3 上只加 2.7% FLOPs、MT-NLG 上 1.6%，这才是「选择性重算」的价值。工程上说 +20%–40%、并强调「可以用 overlap 与更好的重算粒度压到 5% 以内」比背一个数更稳。
- **追问**：序列长度翻倍，哪一块涨得最快？
  - 要点：固定开销完全不动；逐 token 的隐藏维激活 $\propto s$，翻倍就是 2×；朴素 attention 的三张 $s^2$ 矩阵 $\propto s^2$，翻倍就是 4×，占比从 70.2%（$s=2048$）涨到 90.4%（$s=8192$）。顺序是：先上 FlashAttention 干掉 $s^2$ 项，再用选择性重算把 $34sbh$ 压下来，最后才考虑减小 micro-batch 或换更短的 packing。
- **追问**：ZeRO-1/2/3 分别分片什么？通信量怎么变？
  - 要点：ZeRO-1 切优化器状态（$4\Psi + K\Psi/N_d$，通信量与 DP 相同、显存降 4×）；ZeRO-2 再切梯度（$2\Psi + 14\Psi/N_d$，通信量仍与 DP 相同、显存降 8×）；ZeRO-3 再切参数（$16\Psi/N_d$，通信量升到 DP 的 1.5×）。注意三者都不切激活。切得越深，all-gather 的次数越多、单次消息越小，跨节点时通信会吃掉收益（[[inference-serving-07]]）。
- **追问**：bf16 与 fp16 在显存账上有差别吗？
  - 要点：字节数完全一样（都是 2 字节），差别在数值与工程细节：bf16 8 位指数与 fp32 同范围，不需要 loss scaling；fp16 5 位指数需要 GradScaler 与动态 loss scale，溢出时会跳过 optimizer step。显存上唯一的差别来自实现选择——保留 fp32 主权重（16 B/param）还是省掉（12 B/param），以及 Adam 状态用 fp32 还是 8-bit。
- **追问**：为什么 LoRA 省不了激活？
  - 要点：反向传播仍要穿过全部层，链式法则需要的中间量一个都不能少；冻结权重只是不需要计算和保存**权重**的梯度。所以 LoRA 的显存曲线与全量微调在激活项上完全重合，也是它「收益随 batch 增大而被摊平」的原因。
- **追问**：只有一张 24 GB 卡，怎么把 7B 训起来？
  - 要点：全量微调没有可能（固定开销 112 GB），只有 LoRA/QLoRA 两条路。bf16 LoRA 在 $b=1$、$s=2048$ 时要 26 GB，得把序列降到 1024 左右（激活 4.56 GB，合计约 21.8 GB）才进得去；QLoRA 在同样配置下约 16.3 GB，24 GB 卡有近 8 GB 余量。24–48 GB 这个区间对 7B 是偏保守的说法——QLoRA 论文里 48 GB 对应的是 65B 模型。再要压就上 gradient checkpointing、packing 到更短的序列、paged optimizer，代价都是显存换时间。

## 公司变体

`asked_at` 覆盖 Mistral AI 与 Hugging Face 两家。

- **Hugging Face**：偏**工程实现与生态栈**。这题在他们语境里通常落在「用现成的 `transformers` / `peft` / `trl` / `bitsandbytes` / `accelerate` 怎么把这笔账配平」：`per_device_train_batch_size × gradient_accumulation_steps × 卡数` 与显存的关系、`gradient_checkpointing=True`、`optim="adamw_bnb_8bit"`、`bf16=True`、`load_in_4bit` 与 `prepare_model_for_kbit_training`、以及 ZeRO-2/ZeRO-3 的 DeepSpeed config 该开哪一级。Hub 上的模型显存估算器与 PEFT 的实践文档让这些问题有标准答案，所以回答要给可落地的配置组合，而不只是公式。
- **Mistral AI**：偏**原理推导与训练栈取舍**。这家自己从零训 7B 级模型，关注点是 MFU、吞吐与并行配置的组合，追问方向更可能在「16 bytes/param 里哪几项还能砍、砍完对收敛有什么影响」「ZeRO-3 的 1.5× 通信在什么规模会吃掉收益」「为什么大规模训练里 LoRA 不是默认选择、它在什么场景才值」。要求把 $34sbh + 5as^2b$ 与 $\sqrt{N}$ 检查点这类公式推到位，并能解释每个数字的假设。

以上是基于公开技术材料与岗位方向的侧重判断，不代表具体面试流程；实际题面以面试轮次为准。

## 相关题目

- [[finetuning-04]]：LoRA 的分解与秩的选择。本题只用了参数量公式 $r(d_{in}+d_{out})$，秩为什么够用、挂哪些层在那题展开。
- [[finetuning-05]]：QLoRA 与量化权衡。本题只给 NF4 底座约 3.94 GB 的粗账，量化误差与 paged optimizer 的细节在那题。
- [[inference-serving-07]]：tensor/pipeline/data/sequence/expert 并行。ZeRO 与并行策略的关系、通信量随分片度的变化在那题。
- [[inference-serving-08]]：推理侧 70B 的显存估算。同一套记账法在权重 + KV cache 上的版本，可与本题的「固定开销 vs 随负载增长」对照。
- [[llm-internals-05]]：FlashAttention。它消掉的正是本题激活项里的 $5as^2b$。

## 参考资料与归属

1. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)（延伸），Rajbhandari et al.，2019-10-04。提供 model states 与 residual states 的划分、$2\Psi + 2\Psi + K\Psi = 16\Psi$（混合精度 Adam 的 $K=12$）与「7.5B 标准 DP 需 120 GB、GPT-2 1.5B 需 24 GB」的对照、$4\Psi + K\Psi/N_d$ → $2\Psi + 14\Psi/N_d$ → $16\Psi/N_d$ 三阶段公式与「前两级通信量与 DP 相同、第三级 1.5×」的口径、激活显存 $\propto 12 \times h \times b \times s \times L$ 的粗估与「检查点把激活降到约 $\sqrt{N}$、代价 33% 重算」的说法，以及 ZeRO-R 的三项 residual 优化（分片式激活检查点、定长缓冲、碎片整理）。
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（延伸），Hu et al.，2021-06-17。提供可训练参数与 $r(d_{in}+d_{out})$ 的构造、「可训练参数降到 1/10000、显存降到 1/3」的摘要结论、正文「最多降低 2/3，因为不需要为冻结参数存优化器状态」的措辞，以及 GPT-3 175B 实测 1.2 TB → 350 GB 的对照。
3. [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)（延伸），Korthikanti et al.（Megatron），2022-05-10。提供激活的逐张量记账与 per-layer 公式 $sbh(34 + 5as/h)$、序列并行与选择性重算后的 $34sbhL/t$（对 $s$ 线性、与 $a$ 无关）、「激活降 5×、重算时间开销回收 90% 以上」、GPT-3 省 70% 激活加 2.7% FLOPs / MT-NLG 省 65% 加 1.6%、全量重算 30%–40% 执行时间开销，以及 530B 模型 MFU 42.1% → 54.2% 的例子。

本题的 7B 数字全部是按上述公式代入 LLaMA-7B 形状（$L=32$、$d=4096$、$d_{ff}=11008$、$a=32$、$V=32000$）自行复算的：三篇论文都没有给 7B 的显存表，$b=1$、$s=2048$ 的激活、ZeRO 各阶段的每卡占用、LoRA 参数量与状态、盈亏平衡点、以及「2–4 GB 临时与碎片」的经验预留都是自行推导与假设，正文已逐处标明口径。QLoRA 的 65B/48 GB 与 NF4 双层量化属于 [[finetuning-05]] 的范围，这里只用它做量级对照，未引用其原文数字以外的结论。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
