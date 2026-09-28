---
type: question
id: llm-internals-15
topic: LLM 内部原理与架构
order: 15
question: 解释 SwiGLU，以及为什么现代 LLM 的 MLP 模块中用门控激活取代了 ReLU/GELU。
question_en: Explain SwiGLU and why gated activations replaced ReLU/GELU in modern LLM MLP blocks.
asked_at: [Meta]
level: 进阶
tags: [swiglu, ffn, 激活函数, llama]
sources:
  - title: Feed-Forward Networks in LLMs
    url: https://outcomeschool.com/blog/feed-forward-networks-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-04-13
  - title: GLU Variants Improve Transformer（延伸）
    url: https://arxiv.org/abs/2002.05202
    author: Noam Shazeer
    published: 2020-02-12
related: [llm-internals-14, llm-internals-16, llm-internals-10]
updated: 2026-09-28
---

## 一句话答案

> SwiGLU 是把 FFN 的第一个线性层拆成两条支路、再用 Swish(=SiLU) 门控其中一条的 GLU 变体：$\mathrm{SwiGLU}(x)=\big(\mathrm{Swish}(xW_{gate})\odot xW_{up}\big)W_{down}$。它比 ReLU/GELU FFN 好，是因为逐元素乘积给了网络**输入相关**的乘性选择能力（门可以按 token 关掉整条通道），而 Swish 平滑、非单调的形状比 ReLU 的硬截断更好优化。代价是每层多一个矩阵乘；为守住参数预算，隐藏维从 $4d$ 降到约 $\tfrac{2}{3}\cdot 4d \approx 2.67d$，所以它并不是「更贵」，而是「同样的参数花在门控上」。

## 面试官在考什么

- 能不能写出 **GLU 家族的统一形式**，并把 ReLU/GELU/Swish 代进去得到 ReGLU/GEGLU/SwiGLU。只记住 SwiGLU 一条公式而不能归纳，说明是背的。
- 能不能说清「门控」到底带来了什么表达能力：乘性交互 = 输入相关的特征选择，而 ReLU/GELU 只是逐元素的固定非线性。
- **参数账是否算得清**：三个矩阵、$d_{ff}$ 要按 $2/3$ 缩放、LLaMA 的 $8d/3$ 与 256 对齐是怎么来的。这一条最能区分「读过论文」和「读过博客」。
- 是否知道这是**经验结论**：Shazeer 的结论来自同参数量下的 perplexity 对比，不是定理；不要把它讲成「门控在数学上必然更强」。
- 工程侧：gate/up 能否合并成一次 GEMM、融合 kernel 省的是显存带宽还是 FLOPs、MoE 的专家为何也用 SwiGLU。

**常见错误答案**

- 「SwiGLU 是一种新的激活函数。」它不是激活函数，是一个**带门控的 FFN 结构**。Swish 才是激活函数，SwiGLU 是「Swish 门控的线性单元」。
- 「SwiGLU 多了一个矩阵，所以参数更多、更慢。」保持 $d_{ff}=4d$ 才会多 50% 参数；实践里 $d_{ff}$ 会缩到 $8d/3$ 左右，总参数与 FLOPs 与原始 FFN 基本持平。
- 「Swish 比 ReLU 好，所以换了。」单换激活函数收益很小；这题的主角是**门控**，Swish 只是选来当门的那把刀。

## 原理与推导

### 标准 FFN 与它承担的角色

Transformer 每个 block 里 attention 负责跨 token 搬运信息，FFN 负责**逐 token** 的非线性变换。位置无关、参数共享，因此同一组权重作用在每个 token 上：

$$
\mathrm{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2,\qquad W_1 \in \mathbb{R}^{d_{ff}\times d},\ W_2 \in \mathbb{R}^{d\times d_{ff}}
$$

原论文取 $d_{ff} = 4d$，形状是「先升维再降维」：升维给出足够宽的空间做特征组合，降维把结果压回残差流需要的维度。因为参数量 $\approx 2\,d\,d_{ff} = 8d^2$，而 MHA 的 $W^Q,W^K,W^V,W^O$ 合计约 $4d^2$，所以 FFN 占了 block 参数的三分之二——模型的「知识」主要存在这里。

### GLU 家族：用一条支路去门控另一条

Gated Linear Unit 的核心改动是**逐元素相乘**：

$$
\mathrm{GLU}(x) = (W_1 x) \odot \sigma(W_2 x)
$$

$\odot$ 是 Hadamard 积。$\sigma(W_2x)$ 取值在 $(0,1)$，相当于给 $W_1x$ 的每一维配了一个**由当前输入算出来的**软开关。把 $\sigma$ 换成别的函数就得到一整族：

$$
\mathrm{ReGLU} = \mathrm{ReLU}(xW_g)\odot xW_u,\quad
\mathrm{GEGLU} = \mathrm{GELU}(xW_g)\odot xW_u,\quad
\mathrm{SwiGLU} = \mathrm{Swish}(xW_g)\odot xW_u
$$

再接一个输出投影 $W_{down}$，就是完整的 FFN：

$$
\mathrm{SwiGLU}(x) = \Big(\mathrm{Swish}\big(xW_{gate}\big)\odot xW_{up}\Big)W_{down}
$$

其中 $\mathrm{Swish}(z) = z\cdot\mathrm{sigmoid}(\beta z)$；$\beta=1$ 时即 SiLU（Sigmoid Linear Unit），现代实现都用 $\beta=1$，所以 Swish 与 SiLU 在这里指同一个函数。三矩阵形状：$W_{gate},W_{up}\in\mathbb{R}^{d_{ff}\times d}$，$W_{down}\in\mathbb{R}^{d\times d_{ff}}$（多数实现省略 bias）。

注意同一个 $x$ 要分别过 $W_{gate}$ 与 $W_{up}$ 做两次投影（$W_{down}$ 吃的是门控之后的 $d_{ff}$ 维激活），这就是「多一个矩阵乘」的来源。

### 为什么门控更强：乘性交互与条件化

ReLU/GELU 分支的能力上限是「固定的逐元素非线性 + 两层线性」，而两层线性之间没有数据依赖的耦合。门控引入乘法后，表达能力直接上一个台阶：

- **按输入选择通道。** 门值可以随 token 变化：同一个通道在 token A 上打开、在 token B 上关闭。ReLU 的稀疏性是**输入相关但单维**的（只看自己那一维的符号），门控则允许一路信息去调制另一路，形成特征之间的交互。
- **乘积项即条件化的线性变换。** 门控后的第 $i$ 维是 $\mathrm{Swish}\big(\sum_k v_{ik}x_k\big)\cdot\sum_j w_{ij}x_j$，即两个关于 $x$ 的线性型相乘，等价于一个**权重由输入决定**的线性映射 $W(x)\,x$。这才是「门控 = 输入相关的特征选择」的准确含义。
- **乘积关系写进了函数形式。** 两层 ReLU MLP 是通用逼近器，理论上当然也能逼近 $x_1x_2$：门控的优势不是「ReLU 网络做不到」，而是把两个线性型相乘直接摆进前向表达式，不必让优化器在连续空间里把它搜出来（手工装配权重的例子见下一节）。
- **Swish 的形状更友好。** $\mathrm{Swish}'(z)=\mathrm{sigmoid}(z)\,(1+z(1-\mathrm{sigmoid}(z)))$ 处处连续，而 ReLU 在 0 处导数跳变；Swish 还非单调——在 $z\approx-1.2785$ 处取最小值约 $-0.2785$，负输入区有小的负输出，梯度不会整片归零。

**这是经验结论，不是定理。** Shazeer 的做法是把 T5 的 FFN 子层换成上述变体，在 C4 上预训练；为保持参数量与计算量相同，$d_{ff}$ 从 3072 缩到 2048（乘 $2/3$）。正文的结论是 "The GEGLU and SwiGLU variants produce the best perplexities"，摘要写的是 some of them yield quality improvements over the typically-used ReLU or GELU activations。论文同时承认结果有噪声，也明说不解释这些结构为何有效：既没有「门控必然更优」的证明，也没有覆盖所有规模与数据分布。LLaMA、PaLM、Mistral、Qwen 采用它，是同参数量下的经验胜出加工程可行性共同作用的结果。

### 参数账：$d_{ff}$ 为什么必须缩到 $8d/3$

SwiGLU 有三个矩阵，参数量 $3\,d\,d_{ff}$、每个 token 的 MACs 同样 $3\,d\,d_{ff}$——**在 dense FFN 里参数量与计算量成正比**。分两种情形：

- 保持 $d_{ff} = 4d$：参数与 FLOPs 都变成原来的 $3/2$。多 50% 的代价换来一点 perplexity，通常不划算，也不会被采用。
- 让参数量持平：解 $3\,d\,d_{ff} = 2\,d\cdot 4d$ 得

$$
d_{ff} = \frac{2}{3}\cdot 4d = \frac{8}{3}d \approx 2.67\,d
$$

这正是 LLaMA 系列 `intermediate_size` 的来历：LLaMA 论文的原话是 "We use a dimension of $\frac{2}{3}4d$ instead of $4d$ as in PaLM"，工程上再把它向上取到 256 的倍数（多半出于切分与 kernel 对齐的考虑，论文本身没写这条规则）。$d=4096$ 时 $8d/3 = 10922.67$，向上取 256 的倍数得 $43\times256=11008$，正是 LLaMA 2 7B 的取值（$=2.69d$）；$d=8192$ 时 $8d/3=21845.33$，同样向上取整得 $86\times256=22016$，正是 LLaMA 1 65B 的 `intermediate_size`。反例是 LLaMA 2 70B：它取 28672（$=3.5d$），比 $2.67d$ 大一截，说明大尺寸上这条经验比例并没有被严格守住。

### 为什么门控的代价在大模型里可以接受

「多一个矩阵乘」听上去是纯亏，但放到大模型的实际瓶颈里看：prefill 是计算受限，SwiGLU 与同参数 ReLU FFN 的 FLOPs 几乎一样（见下表），没有额外损失；decode 是显存带宽受限，每步都要把 FFN 的全部权重从 HBM 读一遍，参数量持平意味着读的字节数也持平。**被换掉的只是「同一个参数预算怎么花」，而不是「花多少」。** 更值得一提的是 gate 与 up 可以拼成一个矩阵做一次 GEMM：

$$
W_{gateup} = \begin{bmatrix} W_{gate} \\ W_{up} \end{bmatrix} \in \mathbb{R}^{2d_{ff}\times d}
$$

一次 GEMM 出 $2d_{ff}$ 维结果，切开做 Swish 与逐元素乘，再喂给 $W_{down}$。tensor parallel 下按**输出维度**切分 $W_{gateup}$（列并行：每 rank 只算 $2d_{ff}/t$ 个输出通道，拿到的是权重矩阵对应的行块），各 rank 在本地独立完成门控；$W_{down}$ 是行并行，它的输出需要一次 all-reduce——这是 Megatron 类实现的标准做法。不融合的话，Swish 的输出要写回 HBM 再读出来做乘法，中间激活的访存是纯浪费；融合 kernel（`silu_and_mul` 这类把激活与逐元素乘写在一起的算子）把这段压成一次读写。

MoE 里每个专家同样用 SwiGLU（Mixtral、DeepSeek、Qwen 的 MoE 变体都是）：专家本质就是小 FFN，同一套质量与参数结论直接适用，而且三个矩阵给了路由器更多可被特化的方向。细节见 [[llm-internals-10]]。

回到题干：不是 ReLU/GELU 本身不行（GELU 至今仍是很多模型的默认），而是在同一参数预算下，把预算花在「门控 + 平滑激活」上更划算，代价只是一个额外的矩阵乘——在大模型里这是划算的交换。

## 数值与代码验证

取 $d = 4096$（LLaMA 2 7B 的宽度），忽略 bias。

| 配置 | $d_{ff}$ | 矩阵数 | 每层 FFN 参数 | 相对基线 | 每 token MACs |
| --- | --- | --- | --- | --- | --- |
| 基线 ReLU/GELU FFN | 16384 ($4d$) | 2 | 134.22 M | $1.000\times$ | 134.22 M |
| SwiGLU 保持 $d_{ff}=4d$ | 16384 | 3 | 201.33 M | $1.500\times$ | 201.33 M |
| SwiGLU 等参数预算 $d_{ff}=8d/3$ | 10922.67 | 3 | 134.22 M | $1.000\times$ | 134.22 M |
| SwiGLU 实际对齐值 | 11008 | 3 | 135.27 M | $1.008\times$ | 135.27 M |

复算过程：$2\times 4096\times 16384 = 134{,}217{,}728$；$3\times 4096\times 16384 = 201{,}326{,}592$（比值恰为 $1.5$）；等预算那一行要用精确分数代入，$3\cdot d\cdot\tfrac{8}{3}d = 8d^2 = 134{,}217{,}728$，与基线完全相同（表里的 $10922.67$ 只是四舍五入的显示值，直接用它会算出 $134{,}217{,}769$，差 41 个参数）；$3\times 4096\times 11008 = 135{,}266{,}304$，比基线多 $1{,}048{,}576$ 个参数，即 $+0.78\%$。也就是说 LLaMA 的 11008 相对完美的 $8d/3$ 只多花了 0.8% 的参数。

**FFN 在 block 里的占比**（attention 按 MHA 的 $4d^2$ 计）：

$$
\text{ReLU FFN 占比} = \frac{8d^2}{8d^2+4d^2} = \frac{2}{3},\qquad
\text{SwiGLU 等预算占比} = \frac{3d\cdot\tfrac{8}{3}d}{3d\cdot\tfrac{8}{3}d+4d^2} = \frac{8d^2}{12d^2} = \frac{2}{3}
$$

两者都是 66.7%——$8d/3$ 这个取值的意义就是把「三分之二」这个比例原样保住。但真实模型用 GQA，attention 参数会小于 $4d^2$：取 $d=4096$、32 个 Q head、8 个 KV head、$d_{head}=128$（$W_Q,W_O$ 各 $4096\times4096$，$W_K,W_V$ 各 $4096\times1024$），attention 参数为 $2\times 4096\times 4096 + 2\times 4096\times 8\times 128 = 41.94\text{M}$，于是 FFN 占比升到 $135.27/(135.27+41.94) \approx 76.3\%$。这解释了源文那句「GQA 会让 FFN 占比超过 66%」。

### 对比表

| 对比项 | ReLU FFN | GELU FFN | SwiGLU FFN |
| --- | --- | --- | --- |
| 公式 | $\mathrm{ReLU}(xW_1)W_2$ | $\mathrm{GELU}(xW_1)W_2$ | $(\mathrm{Swish}(xW_g)\odot xW_u)W_{down}$ |
| 权重矩阵数 | 2 | 2 | 3 |
| 同 $d_{ff}=4d$ 的参数 | $8d^2$ | $8d^2$ | $12d^2$（$1.5\times$） |
| 等参数预算下的 $d_{ff}$ | $4d$ | $4d$ | $\approx 2.67d$（实际 $8d/3$） |
| 逐元素乘法 | 无 | 无 | 有（门控） |
| 同预算 perplexity（C4 上预训练） | 基线 | 与 ReLU 同被论文列为对照 | 论文报告最好（GEGLU/SwiGLU） |
| 代表模型 | 原版 Transformer、GPT-2 | GPT-3、BERT | LLaMA 全系、PaLM、Mistral、Qwen |

表中的 perplexity 一行引自 Shazeer 的论文：他在 T5 base 上以**相同参数量与计算量**对比各变体（$d_{ff}$ 从 3072 缩到 2048），正文结论是 GEGLU/SwiGLU 的 perplexity 最好；论文同时提醒结果有噪声、且不解释原因。具体数值随规模与训练步数变化，引用时以论文原文为准，不要背具体小数。

### PyTorch 验证：等价改写与参数量

```python
import torch, torch.nn as nn, torch.nn.functional as F

d, d_ff = 4096, 11008

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))   # SiLU == Swish(beta=1)

m = SwiGLU(d, d_ff)
n = sum(p.numel() for p in m.parameters())
print(n, n / (2 * d * 4 * d))       # 135266304  1.0078125：与 4d 基线几乎等参

# gate/up 合并成一次 GEMM，结果完全一致
x = torch.randn(2, 8, d)
W = torch.cat([m.gate.weight, m.up.weight], dim=0)         # [2*d_ff, d]
h = F.linear(x, W)                                          # 一次 GEMM
g, u = h.chunk(2, dim=-1)
assert torch.allclose(m(x), m.down(F.silu(g) * u), atol=1e-5)

# Swish 非单调：最小值约 -0.2785，出现在 z ≈ -1.278
z = torch.linspace(-4, 4, 100001, requires_grad=False)
s = F.silu(z)
print(s.min().item(), z[s.argmin()].item())                 # -0.278465 -1.2788（网格点；连续最小点在 -1.2785）
print((F.relu(z) - s).abs().max().item())                   # 0.278465：差异集中在负半轴，ReLU 恒为 0 而 SiLU 最低到 -0.2785
```

```python
# 门控是「结构性」拿到乘积，而不是靠学出来：手算权重组装即得 AND，无需训练
import torch, torch.nn.functional as F

x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
b_gate = torch.tensor([-50.])           # 偏置把门推离 0.5
w_gate = torch.tensor([[100., 100.]])   # sigmoid(100*(x1+x2) - 50) ≈ 1 当 x1+x2>=1，≈0 当 x1=x2=0
w_up   = torch.tensor([[1.], [1.]])     # x1 + x2
gate = torch.sigmoid(x @ w_gate.T + b_gate)
out  = gate * (x @ w_up)
print([round(v, 6) for v in gate.flatten().tolist()])   # [0.0, 1.0, 1.0, 1.0]
print([round(v, 6) for v in out.flatten().tolist()])    # [0.0, 1.0, 1.0, 2.0] —— 门 + 线性读出即 AND / 计数

# 同一结构也能直接给出 x1*x2 = AND：门分支判 x1+x2>=1.5，up 分支取 x2
w_gate2 = torch.tensor([[100., 100.]])
b_gate2 = torch.tensor([-150.])         # sigmoid(100*(x1+x2) - 150)：仅 x1=x2=1 时门为 1
w_up2   = torch.tensor([[0.], [1.]])    # 0*x1 + 1*x2，配合门即得 AND
print([round(v, 6) for v in (torch.sigmoid(x @ w_gate2.T + b_gate2) * (x @ w_up2)).flatten().tolist()])
# [0.0, 0.0, 0.0, 1.0]：x1*x2 在布尔域上就是 AND
```

这段代码要说明的不是「ReLU 网络算不出 AND」——AND 在布尔域上线性可分，一个阈值单元就够——而是门控允许把乘积关系**手工装配**进权重里。要在 ReLU MLP 上得到同样的行为，得靠梯度下降在连续空间里搜；论文的比较也是在同参数预算下看**训练后**的 perplexity，而不是比结构能表达什么。

## 常见追问

- **追问**：既然门控更强，为什么不把 $d_{ff}$ 保持 $4d$ 硬上？
  - 要点：那会让 FFN 参数与 FLOPs 都变成 $1.5$ 倍。训练成本、显存、推理延迟都要按这个倍数付，而 perplexity 的收益只对应很小一部分。$8d/3$ 是在「等参数预算」这条公平线上比较，Shazeer 论文的比较也是在同一参数预算下做的。
- **追问**：Swish 和 SiLU 是同一个东西吗？
  - 要点：$\mathrm{Swish}(z)=z\cdot\mathrm{sigmoid}(\beta z)$，$\beta=1$ 时即 SiLU，所以现代 LLM 里两个名字指同一函数。注意 GELU 常写成 sigmoid 近似 $x\,\mathrm{sigmoid}(1.702x)$，它和 Swish 形状相近但不是一个函数，写 SwiGLU 时别把两者弄混。
- **追问**：换成 SwiGLU 之后，FFN 的「知识存储」性质变了吗？
  - 要点：没变。FFN 依然逐 token 独立作用、依然持有 block 里约三分之二的参数，只是多了一条可被输入调制的通道。可解释性工作里那些「key-value memory」式的 FFN 分析判据仍然适用，只是写回残差流的那 $d_{ff}$ 维激活变成了「门控之后的组合」，单条通道不再等价于一次纯线性读出，分析时要把它当成 Swish 门与 $xW_{up}$ 的乘积来看。
- **追问**：为什么是 Swish 当门，而不是 sigmoid 或 ReLU？
  - 要点：原始 GLU 用 sigmoid 门，输出被压在 $(0,1)$ 且均值约 0.5，信号被系统性衰减；ReLU 门（ReGLU）是硬截断，负半轴全丢；Swish 允许小幅负门值（最小值约 $-0.2785$），既能抑制也能**反相**调制，并且导数连续。论文只报告 GEGLU/SwiGLU 的 perplexity 最好，并强调结果有噪声，并没有说平滑门在所有任务上全面胜出。
- **追问**：MoE 里每个专家也用 SwiGLU，为什么？
  - 要点：专家本质就是小 FFN，同一套质量/参数结论直接适用；而且三个矩阵给了路由器更多可特化方向——专家可以只在 gate 分支上分化。代价是每个专家都带三份权重，稀疏激活下每 token 的访存变大，所以专家数、top-k 与 $d_{ff}$ 要一起调。
- **追问**：不做融合 kernel 会损失多少？
  - 要点：损失在访存而非 FLOPs。分开写时，$2d_{ff}$ 维的中间激活要写回 HBM 再读回来做逐元素乘；decode 阶段本就带宽受限，这段多余读写会直接吃掉一部分吞吐。融合成一次 GEMM + 一次 elementwise 后，中间结果留在寄存器/SRAM 里。这也解释了为什么 gate/up 合并（`gate_up_proj`）几乎成了默认写法。

## 公司变体

- **Meta（FAIR / Llama）**：这题在 Meta 是架构决策题，偏工程实现而非数学推导。LLaMA 论文本身只写了一句话——用 SwiGLU 替换 ReLU，维度取 $\frac{2}{3}4d$；而公开 config 里 `hidden_act: silu` 与 `intermediate_size` 是摆明面的（7B 11008、65B 22016、70B 28672）。所以常见追问是「11008 是怎么来的」「为什么不是 16384」「GQA 之后 FFN 占多少参数」「tensor parallel 怎么切 $W_{gate}$ 与 $W_{up}$」，以及 MoE 变体里专家的 FFN 是否照抄 dense 的配置。准备时把 4096/11008 与 8192/28672 两组数算熟，比背论文结论有用。

## 相关题目

- [[llm-internals-14]]：FFN 在 Transformer block 里的定位与「知识存储」性质，是本题的前置。
- [[llm-internals-16]]：RoPE 等位置编码改动同样属于「现代 LLM 相对原版 Transformer 的架构更新」，两题常一起问。
- [[llm-internals-10]]：MoE 用多个 FFN 替换单个 FFN，专家内部仍是 SwiGLU，参数账与稀疏激活的权衡可以接着这题问。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《Feed-Forward Networks in LLMs》，2026-04-13，<https://outcomeschool.com/blog/feed-forward-networks-in-llms>（FFN 结构、expand-then-contract、SwiGLU 的三矩阵写法、LLaMA 2 7B 的 11008 与 $2.69\times$、FFN 占比约 66%、GQA 会把这个占比推得更高、MoE 用多个专家 FFN 替换单个 FFN）
- Noam Shazeer，《GLU Variants Improve Transformer》（延伸），2020-02-12，<https://arxiv.org/abs/2002.05202>（GLU 变体的统一形式与 SwiGLU 定义、为等参数把 $d_{ff}$ 乘 $2/3$（T5 base 3072→2048）、同参数量与计算量下 GEGLU/SwiGLU 取得最好 perplexity 的经验结论）

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。源文只给了 $2.7\times$ 这个结论与 SwiGLU 的三矩阵写法；GLU 家族的统一形式、$2/3$ 缩放因子、$8d/3$ 与 256 对齐的推导来自 Shazeer 的论文（延伸来源），LLaMA 的配置取值（11008、22016、28672）核对自公开的 HF config。文中的参数与占比数字（$134.22\text{M}$、$135.27\text{M}$、$1.5\times$、$8d/3$、$66.7\%$、$76.3\%$）均为按 $d=4096$ 自行复算，与源文的 $2.7\times$ 口径一致但更精确。
