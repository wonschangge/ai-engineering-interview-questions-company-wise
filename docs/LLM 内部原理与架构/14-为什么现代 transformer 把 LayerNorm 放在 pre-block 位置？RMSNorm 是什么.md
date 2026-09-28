---
type: question
id: llm-internals-14
topic: LLM 内部原理与架构
order: 14
question: 为什么现代 transformer 把 LayerNorm 放在 pre-block 位置？RMSNorm 是什么？
question_en: Why is LayerNorm placed pre-block in modern transformers, and what is RMSNorm?
asked_at: []
level: 进阶
tags: [layernorm, rmsnorm, pre-ln, 训练稳定性]
sources:
  - title: RMSNorm (Root Mean Square Layer Normalization)
    url: https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: On Layer Normalization in the Transformer Architecture（延伸）
    url: https://arxiv.org/abs/2002.04745
    author: Xiong et al.
    published: 2020-02-11
  - title: Root Mean Square Layer Normalization（延伸）
    url: https://arxiv.org/abs/1910.07467
    author: Zhang, Sennrich
    published: 2019-10-16
related: [llm-internals-15, llm-internals-16]
updated: 2026-09-28
---

## 一句话答案

> post-LN 写 $x_{l+1}=\mathrm{LN}\big(x_l+F_l(x_l)\big)$，pre-LN 写 $x_{l+1}=x_l+F_l\big(\mathrm{LN}(x_l)\big)$，差别只在 norm 位于残差加法的哪一侧。把 norm 挪进子层之后，残差主干上没有任何归一化，梯度有一条恒等通路从顶层直通底层，初始化时靠近输出层的参数梯度不再被局部 $\sigma$ 反复调制，于是可以去掉 warmup、训到更深。代价是主干表示尺度随深度累积，必须在最后补一个 final norm 才能投影到词表。RMSNorm 再砍一刀：只除以均方根，不减均值、不加 $\beta$，参数从 $2d$ 降到 $d$、少一次跨维度 reduction，质量与原论文中的 LayerNorm 基本持平，LLaMA 系、Gemma、Qwen、DeepSeek 因此基本全面换用。

## 面试官在考什么

- 能不能把两个公式写对。post-LN 的 LN 在残差加法**之后**，pre-LN 的 LN 在子层**内部**；把 post-LN 写成 $x+F(\mathrm{LN}(x))$ 是最常见的错误，那已经是 pre-LN。
- 是不是真的理解「主干上不归一化」等于「梯度有恒等通路」，以及这个选择要付什么代价。能写出每层雅可比、指出连乘里那条纯恒等项，再接上「表示尺度累积 → 需要 final norm → 有效步长的口径变了」，才算答到点上。
- LayerNorm 的统计维度：在**特征维**（每个 token 自己的 $d$ 个通道）上算，与 batch 大小、与序列长度都无关；能顺手对比 BatchNorm 的 running statistics 更好。
- RMSNorm 的公式与不变性：$Jx=0$ 给出尺度不变性，$J\mathbf 1\neq 0$ 说明平移不变性被放弃——「re-centering 可省、re-scaling 必须留」这句话要有数学支撑，不能只是复述结论。
- 工程细节：norm 的参数量、add+norm 的 kernel 融合、QK-norm 与 DeepNorm 各自解决什么问题。

**常见错误答案**

- 「post-LN 就是 $x+\mathrm{Sublayer}\big(\mathrm{LN}(x)\big)$。」这是 pre-LN 的公式，写错等于整道题的前提塌了。
- 「RMSNorm 是为了省参数、牺牲一点质量换速度。」省下的参数是 0.27M 量级、占 7B 模型的 0.004%（见数值一节），而且原论文的结论是质量可比。

## 原理与推导

### 两种摆放：只差 norm 在加法哪一侧

记第 $l$ 个子层（attention 或 FFN）为 $F_l$，其输入为 $x_l\in\mathbb{R}^d$（逐 token 独立），两种摆放是：

$$
\text{post-LN：}\quad x_{l+1}=\mathrm{LN}\big(x_l+F_l(x_l)\big)
\qquad\qquad
\text{pre-LN：}\quad x_{l+1}=x_l+F_l\big(\mathrm{LN}(x_l)\big)
$$

post-LN 是原始 Transformer 的写法，每层输出都先加残差、再整体过一遍 LN，所以**残差主干上每一站都有归一化**。pre-LN 把 LN 塞进子层内部，子层看到的是归一化后的输入，而主干 $x_l\to x_{l+1}$ 的加法本身不做任何归一化。

```text
post-LN block                          pre-LN block
   x_l                                    x_l
    |                                      |
    +--> [ Sublayer F ] --+                +--> [ LN ] --> [ Sublayer F ] --+
    |                     |                |                                |
    +-------------------->(+)              +------------------------------->(+)
                          |                                                |
                      [ LN ]                                            x_{l+1}
                          |
                       x_{l+1}
```

顺着图看：post-LN 的主干线（左图竖线）被一个 LN 盒子截断；pre-LN 的主干是右下那条绕过 LN、绕过整个子层的直线，LN 只出现在支路入口。

### LayerNorm 在哪个维度上做

对每个 token 的 $d$ 维特征向量 $x$：

$$
\mu=\frac1d\sum_{i=1}^d x_i,\qquad \sigma^2=\frac1d\sum_{i=1}^d (x_i-\mu)^2,
\qquad \mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

$\gamma,\beta\in\mathbb{R}^d$ 是可学习参数（逐通道缩放与平移）。统计量只来自**同一个 token 自己的 $d$ 个特征**：与 batch 里其它样本无关，与序列长度无关。这带来三个工程后果：batch 大小不影响数值；变长序列不必为 padding 位置做统计 mask；训练与推理走同一套计算，没有 running statistics 需要在导出发版时固定。

对 $\mathrm{LN}$ 求雅可比可以把它做什么讲清楚。记中心化投影 $\Pi=I-\frac1d\mathbf 1\mathbf 1^\top$，中心化向量 $u=x-\mu\mathbf 1=\Pi x$，则

$$
J_{\mathrm{LN}}=\frac{\partial\,\mathrm{LN}(x)}{\partial x}=\frac{1}{\sqrt{\sigma^2+\epsilon}}\Big(\Pi-\frac{uu^\top}{d(\sigma^2+\epsilon)}\Big)\xrightarrow[\epsilon\to0]{}\frac{1}{\sigma}P,\quad P=I-\frac{\mathbf 1\mathbf 1^\top}{d}-\frac{uu^\top}{u^\top u}
$$

$P$ 是到 $\mathrm{span}\{\mathbf 1,u\}$ 的正交补上的投影（$\mathbf 1$ 与 $u$ 正交，两项各是一个秩 1 投影），秩 $d-2$、谱范数 1。因此当 $\epsilon\to0$：

$$
J_{\mathrm{LN}}\mathbf 1=0,\qquad J_{\mathrm{LN}}x=0,\qquad \|J_{\mathrm{LN}}\|_2=\frac{1}{\sigma}
$$

前两式就是两条不变性——平移不变（$\mathrm{LN}(x+c\mathbf 1)=\mathrm{LN}(x)$）来自减均值，尺度不变（$\mathrm{LN}(kx)=\mathrm{LN}(x)$，$k>0$）来自除标准差。第三式是本答案后面反复用到的关键：**LN 的雅可比整体带一个 $1/\sigma$ 因子**。

### pre-LN 为什么更稳：主干上的恒等通路

顶层表示对第 $l$ 层输入的雅可比按链式法则逐层连乘：

$$
\frac{\partial x_L}{\partial x_l}=\prod_{k=l}^{L-1}\frac{\partial x_{k+1}}{\partial x_k}
$$

- post-LN 的每层因子是 $J_{\mathrm{LN}}(y_k)\,(I+J_{F_k})$，其中 $y_k=x_k+F_k(x_k)$。整个因子被 $J_{\mathrm{LN}}$ 左乘：**无论走哪条路径，从第 $l$ 层到第 $L$ 层都要穿过 $L-l$ 次 LN**，每次都乘上一个带 $1/\sigma_k$ 的投影。
- pre-LN 的每层因子是 $I+J_{F_k}\big(\mathrm{LN}(x_k)\big)J_{\mathrm{LN}}(x_k)$。把连乘展开，取每层的 $I$ 就得到一条纯恒等路径 $I^{L-l}=I$：底层的信号与梯度可以不被任何归一化改写地直达顶层；剩下的交叉项则提供子层贡献。

再叠加上一节的 $\|J_{\mathrm{LN}}\|_2=1/\sqrt{\sigma^2+\epsilon}$：post-LN 的梯度尺度被每一层的局部 $\sigma_k$ 依次调制，$\sigma_k$ 偏小就放大、偏大就压小，而初始化时各层的激活尺度并不整齐。这就是 Xiong et al.（ICML 2020）用 mean field theory 得到的结论：**post-LN 在初始化时靠近输出层的参数期望梯度很大**，大学习率在这些梯度上会直接不稳定，warmup 是绕过这个问题的实用手段；相反 **pre-LN 在初始化时梯度行为良好**，可以去掉 warmup，实验中去掉 warmup 的 pre-LN 能达到与调好的 post-LN 基线可比的结果，同时训练时间与调参成本显著下降（具体数值见论文表格）。warmup 的直觉作用也在这里：模型刚初始化时先走极小步长，避免在高曲率、尺度未定的区域被大梯度带跑。

需要划清的边界：论文讲的是「输出端梯度尺度大、需要 warmup」，不是「post-LN 的梯度随深度指数衰减」。下面的实测显示两个变体的 $\|\partial x_L/\partial x_0\|_2$ 都随深度增长，只是 pre-LN 增长得更快；把「更稳」简化成「梯度更大或更小」会答偏。

### pre-LN 的代价：主干变成只加不减的累加器

pre-LN 的递推展开是

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l\big(\mathrm{LN}(x_l)\big)
$$

主干是一个累加器，子层输出只加不减。初始化时各子层输出尺度相当且近似不相关，于是 $\mathrm{Var}(x_L)\approx\mathrm{Var}(x_0)+L\cdot v$，主干范数大致按 $\sqrt L$ 增长（实测见下一节）。两个后果：

1. **必须有 final norm。** 表示尺度随深度漂移，如果直接拿最后一层的 $x_L$ 去过词表投影，logits 的尺度会随深度变化，loss 与梯度都难以标定。所以 pre-LN 模型在输出投影前再放一个 LayerNorm/RMSNorm——LLaMA、Qwen 一类权重里那个单独的 `model.norm` 就是这个 final norm。post-LN 的每层输出已经被钉在单位方差附近，不需要这一步。
2. **有效步长的口径变了。** 由于 $J_{\mathrm{LN}}\propto1/\sigma$，参数的有效更新幅度和表示的尺度耦合；从 post-LN 换到 pre-LN，同一个 learning rate 的含义并不相同，超参需要重扫，不能照搬。

### RMSNorm：只保留 re-scaling

RMSNorm 的定义（$\epsilon$ 在根号内）：

$$
\mathrm{RMS}(x)=\sqrt{\frac1d\sum_{i=1}^d x_i^2+\epsilon},\qquad \mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\mathrm{RMS}(x)}
$$

没有 $\mu$、没有 $\beta$。记 $s=\mathrm{RMS}(x)$，对 $y=\gamma\odot x/s$ 求导：

$$
\frac{\partial y_i}{\partial x_j}=\frac{\gamma_i}{s}\Big(\delta_{ij}-\frac{x_ix_j}{d\,s^2}\Big),\qquad \frac{\partial s}{\partial x_j}=\frac{x_j}{d\,s}\quad\Longrightarrow\quad J_{\mathrm{RMSNorm}}\,x=\frac{\gamma\odot x}{s}\Big(1-\frac{\sum_j x_j^2}{d\,s^2}\Big)=\frac{\gamma\odot x\,\epsilon}{s^3}\xrightarrow[\epsilon\to0]{}0
$$

注意最后一步用了 $\sum_j x_j^2=d(s^2-\epsilon)$：$\epsilon$ 在根号内，所以严格说只有 $\epsilon\to0$ 时 $Jx$ 才等于 0，$\epsilon>0$ 时它只是一个 $O(\epsilon)$ 的残差（下表 RMSNorm 列的 $2.0\times10^{-6}$ 正是它，与 $\epsilon=10^{-6}$ 相符）。也就是说，输入整体放大 $k$ 倍、输出逐元素不变，这正是原论文所说的 re-scaling invariance。反过来 $J_{\mathrm{RMSNorm}}\mathbf 1\neq0$（实测范数 2.32），RMSNorm **不**具有平移不变性——没有减均值，也就没有东西去抵消平移。当 $\epsilon\to0$ 时 $J_{\mathrm{RMSNorm}}=\frac1s\big(I-\frac{xx^\top}{x^\top x}\big)$，即 $\frac1s$ 乘上到 $x^\perp$ 的投影（秩 $d-1$），$\|J\|_2=1/s$。

两个直接推论：

- **为什么减均值可以不要。** 原论文的假设是 LayerNorm 的 re-centering invariance 可以省掉，只保留 re-scaling invariance 与随之而来的隐式学习率自适应。放到 transformer 的语境里：$J\propto1/s$ 意味着输入尺度 $s$ 变小时回传梯度按比例变大、$s$ 变大时被压小，等于对每个 token 的更新幅度做了自适应；而均值偏移在残差加法与线性投影里可以被自己吸收，不需要专门一层来做平移。真正需要被钉死的是**尺度**。
- **参数更少、算子更简单。** 每层 norm 的参数从 $2d$（$\gamma,\beta$）降到 $d$（只有 $\gamma$）；计算上只需要一趟 sum of squares，省掉了求均值以及加 $\beta$ 的广播。原论文还给了一个 pRMSNorm 变体，只用 $p\%$ 的输入分量估计 RMS 而不破坏上述性质，说明这个统计量本身相当宽容。

### 变体与工程细节

- **DeepNorm**（[DeepNet](https://arxiv.org/abs/2203.00555)）：把残差项放大再归一化，$x_{l+1}=\mathrm{LN}\big(\alpha x_l+F_l(x_l)\big)$（$\alpha>1$，并配套一套按深度推导的初始化），目标是同时拿到 post-LN 的表达力与 pre-LN 的稳定性，论文用它把 transformer 训到 1,000 层。
- **QK-norm**：在 attention 内部对每个 head 的 Q、K 各做一次 RMSNorm（再施加 RoPE、再算点积），直接约束 attention logits 的尺度。这一手法先在视觉 transformer 上被用来压住 attention logits 的增长（[ViT-22B](https://arxiv.org/abs/2302.05442)），后在 LLM 里普及。Raschka 的 [QK-Norm 综述](https://sebastianraschka.com/llm-architecture-gallery/qk-norm/) 记录：OLMo 2 的消融把「pre→post 的位置改动」与「加 QK-norm」一起用来压 loss spike，两项贡献难以分离；Qwen3-Next 用 zero-centered 变体、MiniMax M2 用 per-head 变体；也有反向案例，Cohere 的 Tiny Aya 明确去掉 QK-norm，理由是与长上下文行为冲突。
- **fused residual 与 kernel**：pre-LN 的形状让「残差加法 + 下一次 norm 的统计」可以融进一个 kernel（读一遍主干、写一遍），减少一次显存往返。PyTorch 也把 `torch.nn.RMSNorm` 与 `torch.nn.functional.rms_norm` 收进了官方 API（本仓库环境 torch 2.12 实测存在），这类算子级实现正是融合的对象。所以「norm 放哪、用什么 norm」既是数学问题，也直接决定实测吞吐。

## 数值与代码验证

### 手算一遍 LayerNorm 与 RMSNorm

取 $x=[2,4,4,8]$、$d=4$、$\gamma=1$、$\beta=0$：

| 量 | 值 | 口径 |
| --- | --- | --- |
| $\mu$ | 4.5 | $(2+4+4+8)/4$ |
| $\sigma^2$ | 4.75 | $\big(6.25+0.25+0.25+12.25\big)/4$ |
| $\sigma$ | 2.1794 | $\sqrt{4.75}$ |
| mean of squares / RMS | 25 / 5 | $(4+16+16+64)/4$、$\sqrt{25}$ |
| LayerNorm 输出 | $[-1.1471,\,-0.2294,\,-0.2294,\,1.6059]$ | 均值 0、范数 2.0 |
| RMSNorm 输出 | $[0.4,\,0.8,\,0.8,\,1.6]$ | 均值 0.9、范数 2.0 |

口径说明：源文只给了 RMSNorm 这一列（由 $\mathrm{RMS}=5$ 得到 $[0.4,0.8,0.8,1.6]$），我复算一致；LayerNorm 两行与范数列是按同一个向量补算的。两者的输出范数都等于 $\sqrt d=2$（$\gamma=1$），因为都做了 re-scaling；差别在方向——LayerNorm 的输出均值被强制为 0，RMSNorm 把原来的均值偏移留了下来（均值 0.9）。

```python
import torch
x = torch.tensor([[2., 4., 4., 8.]])
mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
ln = (x - mu) / torch.sqrt(var + 1e-5)                     # [-1.1471, -0.2294, -0.2294, 1.6059]
rms = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)  # [0.4, 0.8, 0.8, 1.6]
print(ln.norm().item(), rms.norm().item(), rms.mean().item())  # 2.0  2.0  0.9
```

### 不变性实测：雅可比

同一个随机向量（`torch.manual_seed(0)`，$d=8$，$\sigma=1.1836$；LayerNorm 取 PyTorch 默认 $\epsilon=10^{-5}$，RMSNorm 取 $\epsilon=10^{-6}$）上算 $J$ 的谱：

| 指标 | LayerNorm | RMSNorm | 含义 |
| --- | --- | --- | --- |
| $\|Jx\|$ | $2.0\times10^{-5}$ | $2.0\times10^{-6}$ | 两者都有尺度不变性（$\epsilon>0$ 时是 $O(\epsilon)$ 残差） |
| $\|J\mathbf 1\|$ | $3.3\times10^{-16}$ | 2.32 | 只有 LayerNorm 有平移不变性 |
| $\|J\|_2$ | 0.8449 | 0.8330 | 都等于 $1/\text{尺度}$ |

$\epsilon\to0$ 时 LayerNorm 雅可比的秩是 $d-2=6$，这 6 个奇异值全部等于 $0.844909=1/\sqrt{\sigma^2+\epsilon}$；$d=8$ 下多出来的第 7 个由 $\epsilon$ 引起、量级 $6\times10^{-6}$，所以脚本按 $10^{-10}$ 阈值统计会打印 7（RMSNorm 同理打印 8 个，最小的一个约 $6\times10^{-7}$）。RMSNorm 在 $\epsilon\to0$ 时的谱是 $1/s$ 乘一个秩 $d-1$ 的投影，被清零的那个方向正是 $x$ 自身，即 $Jx=0$。

```python
import torch
torch.manual_seed(0); d = 8
x = torch.randn(1, d, dtype=torch.double)
ln  = lambda t: torch.nn.functional.layer_norm(t, (d,))
rms = lambda t: t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
for name, fn in [("LayerNorm", ln), ("RMSNorm", rms)]:
    J = torch.autograd.functional.jacobian(fn, x).squeeze()
    sv = torch.linalg.svdvals(J)
    one = torch.ones(d, dtype=torch.double)
    print(f"{name:>10} ||Jx||={(J @ x.squeeze()).norm():.2e}  ||J1||={(J @ one).norm():.3f}  "
          f"||J||2={sv[0]:.6f}  非零奇异值个数={int((sv > 1e-10).sum())}")
```

### 深度与主干梯度：pre-LN 的 $\sqrt L$ 累积

在 4 头、$d_{model}=128$（每头 32 维）、随机初始化的小合成栈上，用 power iteration 估输入到输出的雅可比谱范数 $\|\partial x_L/\partial x_0\|_2$（`post` = 每个子层后做 LN；`pre` = 只在子层入口做 LN，主干完全不归一化；两边都不额外加 final norm，这样比较的是主干自身的传播）：

| 层数 $L$ | post-LN | pre-LN | $\sqrt{L/2}$（参考） |
| --- | --- | --- | --- |
| 2 | 1.6949 | 1.7950 | 1.00 |
| 4 | 2.1268 | 2.3504 | 1.41 |
| 8 | 2.5639 | 3.1664 | 2.00 |
| 16 | 3.2132 | 4.5625 | 2.83 |
| 32 | 3.5060 | 5.9290 | 4.00 |
| 64 | 4.3671 | 9.1258 | 5.66 |

读法：pre-LN 的 $\|\partial x_L/\partial x_0\|_2$ 从 1.80 长到 9.13，比例 5.1，与 $\sqrt{64/2}=5.66$ 同量级，符合「主干是只加不减的累加器、各层贡献独立叠加」的推断；post-LN 因为每层输出被归一化，累积被压住（1.69 → 4.37）。**同时注意：这条曲线解释的是主干信号的累积，不是 post-LN 需要 warmup 的原因**——后者来自每层雅可比带 $1/\sigma$ 因子导致的输出端梯度尺度失控（Xiong et al. 的理论结果）。绝对数值依赖初始化、宽度与是否加 final norm，只有趋势可迁移。

```python
# 复算上表：pre-LN 与 post-LN 的 ||dx_L/dx_0||_2
import torch, torch.nn as nn
class Block(nn.Module):
    def __init__(s, d, h, post):
        super().__init__(); s.post = post
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        if s.post:                      # x_{l+1} = LN(x + F(x))
            x = s.n1(x + s.att(x, x, x, need_weights=False)[0])
            return s.n2(x + s.mlp(x))
        y = s.n1(x)                     # x_{l+1} = x + F(LN(x))
        x = x + s.att(y, y, y, need_weights=False)[0]
        return x + s.mlp(s.n2(x))

def jac_norm(L, post, d=128, h=4, B=4, S=8, V=64, iters=30, seed=0):
    torch.manual_seed(seed)
    emb = nn.Embedding(V, d)
    blocks = nn.ModuleList([Block(d, h, post) for _ in range(L)])
    x0 = emb(torch.randint(0, V, (B, S))).detach().requires_grad_(True)
    v = torch.randn_like(x0); v = v / v.norm()
    x = x0
    for b in blocks:
        x = b(x)
    for _ in range(iters):              # power iteration
        g = torch.autograd.grad(x, x0, grad_outputs=v, retain_graph=True)[0]
        v = g / g.norm()
    return g.norm().item()

for L in (2, 4, 8, 16, 32, 64):
    print(L, round(jac_norm(L, True), 4), round(jac_norm(L, False), 4))
```

### 参数量与算量

以 LLaMA-2 7B 的骨架（$d_{model}=4096$、32 层、每层 2 个 norm + 1 个 final norm = 65 个 norm）为例：

| 口径 | 每个 norm 的参数 | 65 个 norm 合计 |
| --- | --- | --- |
| LayerNorm（$\gamma,\beta$） | $2\times4096=8192$ | 532,480 |
| RMSNorm（只有 $\gamma$） | 4096 | 266,240 |

差值是 266,240 ≈ 0.27M，占 6.74B 的 **0.004%**。所以「省参数」在 7B 级模型上完全不构成理由，真正的收益在算子：少一次跨维度 reduction（求均值）与一次逐元素广播（加 $\beta$），kernel 更简单也更易融合。

时间收益要带口径读。原论文报告的是运行时间下降 **7%~64%**，但这是 2019 年在 RNN 等模型上的口径——LayerNorm 在那些模型的总计算里占比很高。在 transformer LLM 里 norm 的 FLOPs 与访存占比小得多，端到端差异会明显低于这个区间，而且强烈依赖 kernel 是否把 add 融合进去、序列形状与 dtype。如果看到「RMSNorm 快约 10%」这类说法，先问是哪一级口径：算子级（单次 norm 调用、或 add+norm 融合前后）可以看到这个量级，端到端会被矩阵乘摊薄；两者不能混用，也不能把论文的 7%~64% 直接搬到 LLM 上。

## 常见追问

- **追问**：pre-LN 与 post-LN 的公式差别到底是什么？
  - 要点：只差 LN 在残差加法的哪一侧。post-LN 是 $\mathrm{LN}(x+F(x))$，主干每站都被归一化；pre-LN 是 $x+F(\mathrm{LN}(x))$，主干不做归一化，LN 只在子层入口。写公式时必须写对，这是整题的立足点。
- **追问**：为什么 pre-LN 一定需要 final norm？
  - 要点：主干是累加器，$\mathrm{Var}(x_L)\approx\mathrm{Var}(x_0)+Lv$，范数随深度增长（实测 $L=2\to64$ 时 $\|\partial x_L/\partial x_0\|_2$ 从 1.80 到 9.13）。不过 final norm 的话，词表投影前 logits 的尺度随深度漂移。post-LN 因为每层输出都被归一化，主干尺度恒定，不需要这一步。
- **追问**：RMSNorm 减掉了均值，为什么质量不掉？
  - 要点：原论文的假设是 LayerNorm 的 re-centering invariance 可以省，只保留 re-scaling invariance 与隐式学习率自适应；$Jx=0$ 保证尺度不变性，而均值偏移可以被残差加法与线性投影吸收。真正难控的是尺度，不是平移。
- **追问**：pre-LN 就万能吗？还有什么坑？
  - 要点：不是。表示尺度累积（要 final norm）、与 post-LN 的有效步长口径不同（超参要重扫）、极深时主干仍可能失控（需要残差缩放、DeepNorm、μP 一类手段）。post-LN 的表达力在部分任务上更好，OLMo 2 就回到 post-norm 并用 QK-norm 压 spike。
- **追问**：LayerNorm 是在哪个维度上做的？换成 batch 维会怎样？
  - 要点：特征维——每个 token 的 $d$ 个通道。换成 batch 维就是 BatchNorm：统计量依赖 batch 组成、变长序列要小心 padding、推理要用 running statistics，且 batch 很小时统计噪声大。LayerNorm 的这套统计与 batch 无关，所以序列模型里更稳。
- **追问**：$\epsilon$ 放在哪、影响什么？
  - 要点：放在方差或均方根内部（$\sqrt{\sigma^2+\epsilon}$ / $\sqrt{\mathrm{mean}(x^2)+\epsilon}$），避免除零并给零向量一个下限。它同时决定了 LN 雅可比在 $\epsilon\to0$ 时被精确清零的方向：$\epsilon>0$ 时 $J_{\mathrm{LN}}$ 沿 $u$ 方向的奇异值不再是 0，而是 $O(\epsilon)$ 量级（$d=8$、$\epsilon=10^{-5}$ 时实测 $6\times10^{-6}$）。

## 相关题目

- [[llm-internals-15]]：SwiGLU 与 MLP 结构。transformer block 的另一半，常与本题连问「现代 block 相比原版改了哪几处」。
- [[llm-internals-16]]：一次前向传播的全流程。norm 在流水线里的位置、final norm 与词表投影的衔接在那一题里串起来。
- [[llm-internals-08]]：RoPE 与长上下文外推。QK-norm 的施加顺序在 RoPE 之前，两题一起看才能把 attention 入口的尺度控制讲完整。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《RMSNorm (Root Mean Square Layer Normalization)》，<https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization>：RMSNorm 的定义、LayerNorm 对比、数值例（$x=[2,4,4,8]$）、7%~64% 的口径、pre-norm 的放置说明。
- Xiong et al.，《On Layer Normalization in the Transformer Architecture》（延伸），ICML 2020，<https://arxiv.org/abs/2002.04745>：post-LN 初始化时输出端梯度大、warmup 的必要性、pre-LN 可去掉 warmup 的 mean field 论证与实验。
- Zhang, Sennrich，《Root Mean Square Layer Normalization》（延伸），NeurIPS 2019，<https://arxiv.org/abs/1910.07467>：RMSNorm 与 pRMSNorm 的定义、re-centering invariance 可省的假设、隐式学习率自适应。
- 正文中 DeepNorm（<https://arxiv.org/abs/2203.00555>）与 QK-norm（<https://sebastianraschka.com/llm-architecture-gallery/qk-norm/>）两段属于背景补充，链接已在正文给出，不在上面的绑定来源内。数值一节的手算、雅可比谱、深度曲线与参数量均由本仓库复算（PyTorch 环境实测），与源文数字不一致处以复算为准并已在正文标注口径。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
