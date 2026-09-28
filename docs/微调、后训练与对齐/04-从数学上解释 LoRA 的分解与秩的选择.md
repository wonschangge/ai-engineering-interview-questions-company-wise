---
type: question
id: finetuning-04
topic: 微调、后训练与对齐
order: 4
question: 从数学上解释 LoRA 的分解。它为什么有效，以及如何选择秩 r？
question_en: Explain LoRA's decomposition mathematically. Why does it work, and how do you choose the rank r?
asked_at: []
level: 进阶
tags: [lora, 低秩, peft, 秩]
sources:
  - title: LoRA - Low-Rank Adaptation of LLMs
    url: https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-04-24
  - title: LoRA: Low-Rank Adaptation of Large Language Models（延伸）
    url: https://arxiv.org/abs/2106.09685
    author: Hu et al. (ICLR 2022)
    published: 2021-06-17
related: [finetuning-05, finetuning-06, finetuning-09, llm-internals-15]
updated: 2026-09-28
---

## 一句话答案

> LoRA 把全量微调要学的增量 $\Delta W$ 参数化成两个小矩阵的乘积：$W = W_0 + \dfrac{\alpha}{r}BA$，
> 其中 $W_0 \in \mathbb{R}^{d\times k}$ 冻结，$B \in \mathbb{R}^{d\times r}$、$A \in \mathbb{R}^{r\times k}$、$r \ll \min(d,k)$，
> 可训练参数从 $d\cdot k$ 降到 $r(d+k)$。它有效是因为微调真正需要的**权重更新本身就是低秩的**
> （论文的经验发现），而把可表达的更新限制在秩不超过 $r$ 的矩阵里，本身又相当于一层强正则。
> 秩 $r$ 没有理论最优解：从 8 或 16 起步扫描，用**验证集**找到收益饱和点即可。

## 面试官在考什么

- 你能不能写出**形状正确**的分解，并说清 $B$、$A$ 各自的行列：$\Delta W = BA$ 必须是 $d\times k$，所以 $B$ 是 $d\times r$、$A$ 是 $r\times k$，顺序不能反。
- 你会不会算参数账：$d\cdot k$ 对比 $r(d+k)$，以及为什么这个式子在 $r \ll \min(d,k)$ 时才划算。
- 你是否理解 $\alpha/r$ 缩放、$A$ 随机初始化 + $B$ 置零这两处设计**各自解决什么问题**，而不是背下来。
- 你能否把「为什么有效」拆成经验事实、归纳偏置、优化视角三层，而不是只说「参数少所以快」。
- 你能否给出**可操作的选秩流程**（扫描区间、饱和判据、配套旋钮、验证集口径），而不是报一个数字。
- 常见错误答案：
  - 「LoRA 省显存所以能省训练时间」——可训练参数少了不等于 FLOPs 少了，$W_0$ 的前向/反向照样要算，见 [[finetuning-09]]。
  - 「$r$ 越大效果越好，所以直接取 $r=64$」——容量不是瓶颈时增大 $r$ 只增加过拟合风险与显存占用，收益会饱和。
  - 「$A$ 和 $B$ 都该随机初始化」——那样训练起点 $\Delta W \ne 0$，模型一开始就被随机扰动破坏。

## 原理与推导

### 1. 从全量微调到低秩增量

全量微调对某个线性层做的是 $h = W_0 x$ 变成 $h = (W_0 + \Delta W)x$，其中 $\Delta W$ 与 $W_0$ 同形，$d\times k$ 个自由度全部可学。
LoRA 冻结 $W_0$，只把 $\Delta W$ 重参数化：

$$ \Delta W = \frac{\alpha}{r}\,BA,\qquad B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k} $$

前向传播变成：

$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\,BAx,\qquad W_0\ \text{冻结},\; A,B\ \text{可训练} $$

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| $W_0$ | 预训练权重，冻结 | $d \times k$ |
| $A$ | 降维矩阵（先把 $x$ 投到 $r$ 维） | $r \times k$ |
| $B$ | 升维矩阵（再从 $r$ 维投回 $d$ 维） | $d \times r$ |
| $BA$ | 低秩增量，秩不超过 $r$ | $d \times k$ |
| $x$ | 该层的输入 | $k \times 1$ |
| $\alpha$ | 缩放超参，与 $r$ 配对使用 | 标量 |

注意训练时 $\Delta W$ **不需要显式构造**：先算 $Ax$（$r$ 维），再算 $B(Ax)$，额外开销是 $O(r(d+k))$ 而不是 $O(dk)$——训练时总 FLOPs 反而比直接用一张 $d\times k$ 的矩阵略多。
合并只在推理前做一次：训练时若每步都显式构造 $W_0 + \frac{\alpha}{r}BA$，会多出一份 $d\times k$ 的张量与它的反向图开销，而 $W_0$ 依旧冻结、梯度只流经 $A$ 与 $B$，收益为零。

### 2. 秩的硬约束：可表达集合恰好是秩不超过 $r$ 的矩阵

对任意矩阵做 SVD，$M = U\Sigma V^\top = \sum_{i=1}^{\min(d,k)} \sigma_i u_i v_i^\top$。$d\times r$ 的 $B$ 乘 $r\times k$ 的 $A$ 必然满足 $\operatorname{rank}(BA)\le r$；反过来任何秩不超过 $r$ 的 $d\times k$ 矩阵都能做这样的满秩分解。所以 LoRA 的可表达集合**恰好**是「所有秩不超过 $r$ 的矩阵」——这是精确的硬约束，不是近似假设。

Eckart–Young 定理进一步给出这个集合的误差下界：在 Frobenius 范数下最优的秩 $r$ 近似误差是 $\|\Delta W - BA\|_F^2 = \sum_{i>r}\sigma_i^2$，在谱范数下是 $\sigma_{r+1}$。由此有两条必须记住的边界：

- 低秩是**硬约束**：如果真实需要的更新矩阵秩为 $r^{\ast}$，那么 $r < r^{\ast}$ 时存在不可约的近似误差，加多少训练步数都补不回来。
- 参数量 $r(d+k)$ 与 $d\cdot k$ 的比值是 $r\left(\frac{1}{d}+\frac{1}{k}\right)$：只有当 $r \ll \min(d,k)$ 时压缩才显著。$d=k=4096$ 时 $r$ 要远小于 4096，$r=8$ 对应比值 $1/256$。

### 3. 两处初始化的作用

论文的初始化是 $A \sim \mathcal{N}(0,\sigma^2)$、$B = 0$。此时：

$$ \Delta W = \frac{\alpha}{r}BA = 0 \;\Rightarrow\; h = W_0x $$

训练**从预训练模型的函数出发**，第一步的损失与原始模型完全相同，增量沿着梯度方向逐渐长出来。若两侧都随机初始化，起点就是一个方差非零的随机扰动，训练初期要先把这个扰动「修回去」，小数据集上尤其致命；若两侧都置零，$\partial\mathcal{L}/\partial A$ 与 $\partial\mathcal{L}/\partial B$ 会同时为零，参数永远不动。所以硬性要求是**至少有一侧为零**，保证起点 $\Delta W = 0$。

$B=0$ 的代价是 $A$ 在第一步拿不到梯度：$\dfrac{\partial \mathcal{L}}{\partial A} = B^\top \dfrac{\partial \mathcal{L}}{\partial (BAx)}\,x^\top$ 里含 $B^\top = 0$ 的因子。这不影响收敛——$\dfrac{\partial \mathcal{L}}{\partial B} = \dfrac{\partial \mathcal{L}}{\partial (BAx)}\,x^\top A^\top$ 非零，$B$ 先走一步，从第二步起两侧同时更新。反过来把 $A$ 置零、$B$ 随机在数学上是对称的（第一步只有 $A$ 拿到梯度，随后 $B$ 也被更新），两种写法都训得动；论文选了「$A$ 随机、$B$ 置零」这一侧，硬性要求只是起点 $\Delta W = 0$。

### 4. $\alpha/r$ 缩放与秩稳定化

论文的做法是把 $\Delta Wx$ 乘上 $\alpha/r$，目的是**让扫 $r$ 时不必重调超参**：论文的说法是，在 Adam 下若初始化尺度也相应缩放，调 $\alpha$ 与调学习率大致等价，所以干脆把 $\alpha$ 固定为第一次尝试的 $r$。$\alpha/r$ 是经验归一化，不是从某个方差公式推出来的。

尺度分析能看出它压得偏狠。$(\Delta W)_{ij} = \dfrac{\alpha}{r}\sum_{l=1}^{r} B_{il}A_{lj}$ 是 $r$ 个独立乘积之和：在 $A$ 的元素服从 $\mathcal{N}(0,\sigma^2)$、$B$ 的元素量级固定的前提下，这个和的量级按 $\sqrt r$ 增长，于是

$$ \operatorname{std}\big((\Delta W)_{ij}\big) \;\propto\; \frac{\alpha\sigma}{r}\cdot\sqrt{r} \;=\; \frac{\alpha\sigma}{\sqrt r} $$

即 $\alpha/r$ 会随 $r$ 增大把更新量级按 $\sqrt r$ 压小。rank-stabilized LoRA（rsLoRA）因此改用 $\alpha/\sqrt{r}$，让更新量级与 $r$ 大致无关。两种系数相差 $\sqrt r$ 倍：$r=8$ 时 2.8 倍，$r=64$ 时 8 倍——秩越大，越要重新调学习率，或者直接换 rsLoRA。

还有一处容易踩的坑：$\Delta W$ 关于 $\alpha$ 与 $A$ 的初始化标准差 $\sigma$ 都是线性的，两者只能定出一个整体尺度。扫 $\alpha$ 时必须固定 $\sigma$（或把 $\alpha\sigma$ 当一个整体调），否则不同实验之间不可比。

实践结论：**$\alpha$ 与 $r$ 必须成对调**。「$\alpha = 2r$」（即系数 $\alpha/r$ 固定为 2）是常见起点，它固定的是系数而不是 $\Delta W$ 的实际量级。

### 5. 为什么有效：三层解释

**① 经验事实（论文的核心观察）。** 论文在 GPT-3 175B 上做了谱分析：WikiSQL 上只挂 $\{W_q,W_v\}$，$r=1$ 就有 73.4 的验证准确率、$r=8$ 是 73.8，几乎持平；把 $r=8$ 与 $r=64$ 学到的适配矩阵分别做 SVD，两者的 top-1 奇异子空间（用 Grassmann 距离定义的归一化子空间相似度，取值 $[0,1]$）重叠度超过 0.5，而两个随机高斯矩阵之间几乎没有重叠。结论是 $\Delta W$ 的**有效秩远低于矩阵维度**：学一个新任务不需要在 $d\times k$ 维空间里自由移动，只需要少数几个方向的组合——这也解释了为什么把秩卡到 8 或 16 还能追平甚至超过全量微调。

**② 归纳偏置（正则视角）。** 把假设空间从「所有 $d\times k$ 矩阵」收缩到「秩不超过 $r$ 的矩阵」，等于给优化加了一层强正则：
参数只有全量微调的 $r(d+k)/(dk)$（$d=k=4096$、$r=8$ 时是 $1/256$），容量被显式限制，小数据集上更不容易过拟合。这是**统计学习**层面的收益，与计算量无关。

**③ 优化视角（训练动力学）。** $W_0$ 冻结后，训练只更新 $r(d+k)$ 个参数、$\Delta W$ 始终落在秩不超过 $r$ 的集合里：需要维护优化器状态的参数少了，
梯度更新的噪声也更小；底座的知识被完整保留，灾难性遗忘比全量微调弱得多（见 [[finetuning-07]]）。
代价同样来自这里：**表达能力的上限就是 $r$**，任务越偏离预训练分布，这个上限越可能不够用。

### 6. 如何选 $r$：可操作的流程

1. **从 $r=8$ 或 $16$ 起步**，在这条基线上把其余超参调好，再决定要不要加大。
2. **扫描 $r \in \{4, 8, 16, 32, 64\}$**，其余变量固定（数据、步数、`target_modules`、随机种子），只比较验证集指标。
3. **按任务难度排序预期**：输出格式对齐、语气/风格迁移这类「小改动」用 $r=4\sim 8$ 通常够；
   注入新领域知识、新语言、新任务范式则要显著加大，因为需要的更新有效秩更高。
4. **找饱和点**：$r$ 翻倍后验证集指标提升落在噪声范围内（用固定种子的多次运行估计噪声），就停在上一个 $r$。
   越过饱和点后继续加 $r$ 只会增加显存与过拟合风险。
5. **配套旋钮一起看**：$\alpha$（与 $r$ 成对的系数，如 $\alpha=2r$，或固定 $\alpha$ 只扫 $r$）、`lora_dropout`、
   `target_modules`（只挂 q/v，还是 q/k/v/o，还是连 FFN 的线性层一起挂）、学习率。挂载范围往往比秩更影响结果：
   从 q/v 扩到所有线性层，可训练参数的增幅比把 $r$ 从 8 提到 16 大得多（见下一节数字）。
6. **口径要写清楚**：用验证集而不是训练损失选 $r$；比较时固定随机种子与训练步数；报告可训练参数量与峰值显存，方便复现。

## 数值与代码验证

### 参数账（自算）

单个 $d\times k$ 权重矩阵，全量微调需要学 $dk$ 个参数，LoRA 只需要 $r(d+k)$ 个。取 $d=k=4096$：

| $r$ | 全量 $dk$ | LoRA $r(d+k)$ | 压缩倍数 | 适配器 fp16 大小 |
| --- | --- | --- | --- | --- |
| 4 | 16,777,216 | 32,768 | 512× | 64 KiB |
| 8 | 16,777,216 | 65,536 | **256×** | 128 KiB |
| 16 | 16,777,216 | 131,072 | **128×** | 256 KiB |
| 32 | 16,777,216 | 262,144 | 64× | 512 KiB |
| 64 | 16,777,216 | 524,288 | 32× | 1 MiB |

$16{,}777{,}216 / 65{,}536 = 256$ 是精确值，不是约等于；$r=8$ 时「16.8M → 65.5K」与源文的 65,536 一致。

按 $L=32$、$d=4096$、FFN 隐层 $11008$（Llama-7B 的层配置，不含 embedding 与 lm_head）粗算，每层 4 个 attention 投影加 3 个 FFN 矩阵，共约 $6.48\times10^9$ 个参数：

| 挂载范围 | $r$ | 可训练参数 | 占 6.48 B 比例 |
| --- | --- | --- | --- |
| 仅 $W_q, W_v$ | 8 | 4.19 M | 0.06% |
| 仅 $W_q, W_v$ | 16 | 8.39 M | 0.13% |
| 全部线性层 | 8 | 20.0 M | 0.31% |
| 全部线性层 | 16 | 40.0 M | 0.62% |

把 $r$ 从 8 提到 16 只翻一倍参数；而从 q/v 扩到全部 7 个线性层是近 5 倍（4.19 M → 20.0 M）。
分母要说清：本节用被统计的 $6.48\times10^9$ 个线性层参数，[[finetuning-09]] 用 7B 总量，同一个 40.0 M 在那里的占比是 0.57%。

### 论文的极端口径

论文以 GPT-3 175B 为设定报告：相对用 Adam 做全量微调，训练显存从 1.2 TB 降到 350 GB（约 3 倍）；
只挂 $W_q,W_v$、$r=4$（96 层共约 19M 可训练参数）时，**可训练参数减少约 10,000 倍**、checkpoint 从 350 GB 变成 35 MB，
并且在 RoBERTa、DeBERTa、GPT-2、GPT-3 上质量持平或更好，同时**不引入额外推理延迟**（合并后与原模型结构一致）。
这里的 10,000 倍是「175B 全量可训练参数」对「只挂少数几层的适配器参数」的极端口径，不能当成任意模型上的通用倍数；
显存降 3 倍则是 LoRA 相对 Adam 全量微调（Adam 需要额外的梯度与优化器状态）的口径。

### 显存到底省在哪：区分「可训练参数」与「总显存」

Adam + bf16 全量微调的常用口径是每参数约 16 bytes：bf16 权重 2 + bf16 梯度 2 + fp32 master 权重 4 + 两个 fp32 动量 8。
7B 模型对应：

| 组成 | 每参数 | 7B 规模 |
| --- | --- | --- |
| bf16 权重 | 2 B | 13.0 GiB |
| bf16 梯度 | 2 B | 13.0 GiB |
| fp32 master 权重 | 4 B | 26.1 GiB |
| Adam 两个动量 | 8 B | 52.2 GiB |
| 合计（不含激活） | 16 B | 104.3 GiB |

LoRA 冻结底座，消掉的是后面三项：梯度不再为 $W_0$ 保存（省 2 B/参数）、master 权重与动量也不需要（省 12 B/参数），
合计省掉 14 B/参数 = 98 GB ≈ 91 GiB，剩下的 2 B/参数就是底座自己的 bf16 权重（13.0 GiB）。但**底座权重仍要放进显存**，
前向激活与反向传播照旧，所以「LoRA 显存 = 全量的几分之一」只在「可训练状态」这个口径下成立：
16 B/参数里省掉的 14 B 是梯度与优化器状态，激活那一块一点没省，量化底座（QLoRA）才进一步压缩那 13.0 GiB。完整的显存账见 [[finetuning-09]]。

### $\alpha/r$ 的尺度效应（自算）

$d=k=256$、$A$ 的元素服从 $\mathcal{N}(0,0.01^2)$、$B$ 的元素服从 $\mathcal{N}(0,1)$ 且量级不随 $r$ 变，每个 $r$ 用 20 组独立随机矩阵取平均，统计 $\operatorname{std}\big(\frac{1}{r}BA\big)$：

| $r$ | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| 实测均值（相对 $r=4$） | 1.00 | 0.70 | 0.50 | 0.35 | 0.25 |
| $\sqrt{4/r}$ | 1.00 | 0.71 | 0.50 | 0.35 | 0.25 |

系数取 $1/r$ 时更新量级按 $1/\sqrt r$ 衰减（相对 $r=4$，$r=64$ 时缩到约 $1/4$；单次抽样会在几个百分点内偏离这条规律），换成 $1/\sqrt r$ 就与 $r$ 无关——这就是 rank-stabilized 变体的依据。

### 代码验证：分解、起点与合并等价性

```python
import torch, torch.nn as nn

torch.manual_seed(0)
d, k, r, alpha = 4096, 4096, 8, 16

W0 = torch.randn(d, k) * 0.02                # 冻结的预训练权重
A = nn.Parameter(torch.randn(r, k) * 0.01)   # 论文：高斯初始化
B = nn.Parameter(torch.zeros(d, r))          # 论文：零初始化
x = torch.randn(k, 4)
scale = alpha / r

def forward(x):
    return W0 @ x + scale * (B @ (A @ x))

base = W0 @ x
with torch.no_grad():
    assert torch.allclose(forward(x), base, atol=1e-6)   # 训练起点 = 预训练模型

loss = forward(x).pow(2).mean()
loss.backward()
print(A.grad is not None, B.grad is not None, W0.grad if isinstance(W0, nn.Parameter) else None)
print(A.grad.abs().max().item())    # 0.0：B=0 时 ∂L/∂A 里含 B^T=0，第一步 A 不动

opt = torch.optim.SGD([A, B], lr=0.05)
opt.step()                          # 走一步后 ΔW 才离开零

print(torch.linalg.matrix_rank((B @ A).detach()).item())   # <= r；x 只有 4 列，实测 4

W_merged = W0 + scale * (B @ A).detach()
print(torch.allclose(W_merged @ x, forward(x), atol=1e-4))
# 期望依次输出：True True None / 0.0 / 4 / True
```

测「有效秩」必须指定阈值，奇异值谱本身不会给出一个整数秩：

```python
import torch

torch.manual_seed(0)
Q, _ = torch.linalg.qr(torch.randn(512, 512))            # 正交基，保证奇异值就是设定值
M = Q @ torch.diag(torch.logspace(0, -6, 512)) @ Q.T     # 奇异值从 1 衰减到 1e-6
s = torch.linalg.svdvals(M)

for thresh in (1e-2, 1e-3):
    print(thresh, int((s > thresh * s[0]).sum()))        # 阈值 1e-2 → 171，1e-3 → 256
print(torch.linalg.matrix_rank(M).item())                # 359：矩阵库默认浮点容差给出的秩
```

同一个 $\Delta W$，阈值 $10^{-2}$ 给 171、$10^{-3}$ 给 256、`matrix_rank` 的浮点容差给 359。论文没有给这种无阈值的整数秩，
而是用归一化子空间相似度 $\phi = \|(U_{r=8}^{i})^{\top}U_{r=64}^{j}\|_F^2 / \min(i,j) \in [0,1]$ 说明秩亏缺：$r=8$ 与 $r=64$ 的 top-1 方向相似度超过 0.5，随机高斯矩阵之间则接近 0。

## 常见追问

- **追问：LoRA 应该挂在哪几层？**
  - 要点：论文的消融方向是**在有限预算下优先适配 $W_q$ 与 $W_v$**：在 18M 参数预算下（GPT-3 175B，96 层），$\{W_q,W_v\}$ 在 WikiSQL 上是 73.7、MultiNLI 是 91.3，
    同样预算下换成 $\{W_q,W_k,W_v,W_o\}$ 是 73.7 与 91.7——基本持平（MultiNLI 上四矩阵版还高 0.4），而只挂 $W_q$ 或只挂 $W_k$ 分别掉到 70.4 与 70.0。
    论文据此的结论是：与其把单个矩阵的秩加大，不如把预算分给更多值得适配的矩阵。
    `target_modules` 通常比 $r$ 更值得先调：先扫挂载范围，再扫秩；扩到 FFN 线性层时参数增幅最大。
- **追问：为什么 $A$ 随机初始化、$B$ 置零，反过来行不行？**
  - 要点：目标只是让 $\Delta W = BA = 0$，两侧零其一即可。$B=0$ 时 $\partial\mathcal{L}/\partial A$ 含 $B^\top=0$ 的因子，
    第一步 $A$ 拿不到梯度；反过来 $A=0$ 时 $\partial\mathcal{L}/\partial B$ 含 $A^\top=0$ 的因子，第一步 $B$ 拿不到梯度。
    两种取法对称、都会从第二步起正常更新，所以「$A$ 随机、$B$ 置零」是论文选的一侧而不是唯一解；
    真正不能做的是两侧同时随机（起点 $\Delta W \ne 0$）或两侧同时置零（梯度恒为零）。
- **追问：怎么判断秩选够了？$r$ 加大后效果反而变差又是为什么？**
  - 要点：饱和有两把尺子。一是**验证集曲线**：固定其它变量，把 $r$ 翻倍，指标提升落入噪声区间即为饱和。
    二是**奇异值谱**：对训练后的 $BA$ 或对全量微调得到的 $\Delta W$ 做 SVD，看多少奇异值超过指定阈值；
    若全量微调的 $\Delta W$ 有效秩是 $r^{\ast}$，适配器的 $r$ 就不该低于它。
  - 要点：$r$ 变大后变差通常是三件事：容量上来后小数据集更容易过拟合（提高 dropout、减小学习率、缩短训练）；
    $\alpha/r$ 没跟着调导致有效更新尺度过小；最优学习率本身随 $r$ 变化，扫秩时必须重新扫一遍学习率，否则结论无效。
- **追问：多个任务能不能共用一个底座？**
  - 要点：可以。适配器不合并时按请求加载对应的 $A, B$，就是多租户/热切换的基础；合并进 $W_0$ 是单向操作，
    之后无法再切回其他适配器。这是 LoRA 相对 prefix tuning 一类方法的工程优势，见 [[inference-serving-11]]。
- **追问：为什么 LoRA 不一定省训练时间？**
  - 要点：省的是优化器状态与可训练参数的梯度，**不省** $W_0 x$ 的前向与反向 FLOPs；序列长、batch 大时激活显存甚至可能成为瓶颈。
    要同时省算力得配合量化底座、梯度检查点或更小的模型。见 [[finetuning-09]]。
- **追问：LoRA 与全量微调的差距在什么任务上最明显？**
  - 要点：需要注入大量**新知识**或大幅改变输出分布的任务（新语言、新领域的持续预训练式目标），
    秩不超过 $r$ 的约束装不下这些变化；只做格式/风格/指令跟随时差距很小。见 [[finetuning-06]]。

## 相关题目

- [[finetuning-05]]：QLoRA 如何进一步压缩底座显存，以及量化带来的权衡。
- [[finetuning-06]]：LoRA、prefix tuning、prompt tuning 与全量微调的横向比较。
- [[finetuning-07]]：灾难性遗忘与持续学习——冻结底座为什么更抗遗忘。
- [[finetuning-08]]：Prompting、RAG 还是 fine-tuning 的决策框架。
- [[finetuning-09]]：Adam 全量微调的显存账与 LoRA 的对比口径。
- [[inference-serving-11]]：多适配器服务与 adapter 热切换的工程实现。
- [[llm-internals-15]]：线性层的低秩分解与 SVD 在模型压缩中的其他用法。

## 参考资料与归属

- [LoRA - Low-Rank Adaptation of LLMs](https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms)，Amit Shekhar（Outcome School），2026-04-24：
  分解形式 $h = W_0x + \frac{\alpha}{r}BAx$、$B$ 置零初始化、合并回 $W_0$、$4096\times4096$ 矩阵上 $r=8$ 的参数对比与「rank 8–64 实用」的经验结论。
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（延伸），Hu et al.，ICLR 2022，arXiv:2106.09685，2021-06-17：
  低内在秩（rank deficiency）的经验观察（$r=1$ 即可、不同 $r$ 与不同种子的子空间重叠、18M 参数预算下不同挂载矩阵与不同秩的对照）、GPT-3 175B 设定下「可训练参数约少 10,000 倍、显存约降 3 倍、无额外推理延迟」的结论。
  正文第 3 节的两处初始化推导、第 4 节 $\alpha/r$ 与 rank-stabilized 缩放的尺度（$\sqrt r$）分析、第 5 节的三层有效性与第 6 节的选秩流程，
  是本文在原始论文与源文之上补充的推导与工程整理；「数值与代码验证」一节的参数量表、显存表与尺度表均为本文按 $r(d+k)$、$16$ bytes/param 等口径复算或实测，未取自源文的未经核对数值。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
