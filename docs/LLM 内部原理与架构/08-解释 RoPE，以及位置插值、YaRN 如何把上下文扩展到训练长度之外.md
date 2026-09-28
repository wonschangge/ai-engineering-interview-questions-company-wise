---
type: question
id: llm-internals-08
topic: LLM 内部原理与架构
order: 8
question: 解释 RoPE，以及位置插值 / YaRN 如何把上下文扩展到训练长度之外。
question_en: Explain RoPE and how position interpolation / YaRN extend context beyond the trained length.
asked_at: [Meta, Moonshot AI, 阿里巴巴]
level: 高阶
tags: [rope, yarn, 长上下文, 外推]
sources:
  - title: Math Behind RoPE (Rotary Position Embedding)
    url: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding
    author: Amit Shekhar (Outcome School)
    published: 2026-04-23
  - title: RoFormer: Enhanced Transformer with Rotary Position Embedding（延伸）
    url: https://arxiv.org/abs/2104.09864
    author: Jianlin Su et al.
    published: 2021-04-20
  - title: YaRN: Efficient Context Window Extension of Large Language Models（延伸）
    url: https://arxiv.org/abs/2309.00071
    author: Bowen Peng et al.
    published: 2023-08-31
related: [llm-internals-07, llm-internals-13, llm-internals-16]
updated: 2026-09-28
---

## 一句话答案

> RoPE 不对 embedding 做加法，而是按位置把 Query/Key 的每两个维度当成一个二维平面旋转，第 $i$ 个平面的角频率是
> $\theta_i=10000^{-2i/d}$。因为旋转矩阵正交，$\langle R_m q,\;R_n k\rangle=\langle q,\;R_{n-m}k\rangle$——
> 打分只依赖相对距离 $n-m$。超出训练长度会崩，是因为低频（波长 $>L$）那批维度在训练中退化成绝对位置编码、长程相位直接跑到训练分布之外；
> 长度扩展的全部技巧，都是在「保住高频局部信息」和「把低频周期拉长」之间做分配：PI 一律压缩、NTK-aware 改 base、
> YaRN 按波长分三区并补一个 attention 温度。

## 面试官在考什么

- 你是否能写出 RoPE 的构造：$d$ 维拆成 $d/2$ 个二维平面、每对乘 $2\times2$ 旋转矩阵、角频率 $\theta_i=\text{base}^{-2i/d}$。
- 你是否能给出 $\langle R_mq,R_nk\rangle=\langle q,R_{n-m}k\rangle$ 的推导，并解释「编码绝对位置、attention 只看到相对位置」这句话的准确含义。
- 你是否知道长上下文失效的物理图像是**波长与训练长度的关系**，而不是笼统的「位置没训过」。
- 你能否把 PI → NTK-aware → NTK-by-parts → YaRN 讲成一条有因果的演进链，而不是并列罗列四个名词。
- 你是否知道 YaRN 除了插值还改了 softmax 温度（$1/\sqrt{t}=0.1\ln s+1$），以及为什么需要它。
- 常见错误答案：
  - 「RoPE 是相对位置编码，所以天然可以外推」——不成立，波长 $>L$ 的那批维度在训练中充当了绝对位置编码，外推时它们给出的是没见过的绝对位置。
  - 「PI 就是除以 $s$，把位置缩小」——方向说反了。PI 是把位置索引压缩成 $m/s$，等价于把频率乘以 $1/s$，角度才落回训练范围。

## 原理与推导

### 1. RoPE 的构造

设单头维度为 $d$（偶数），把 $d$ 维向量按相邻两维切分成 $d/2$ 个二维平面。第 $i$ 个平面（$i=0,\dots,d/2-1$）的角频率定义为

$$ \theta_i=\text{base}^{-\frac{2i}{d}},\qquad \text{base}=10000 $$

位置 $m$ 的 Query 在第 $i$ 个平面上旋转 $m\theta_i$：

$$ R_{\Theta,m}^{(i)}=\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix},\qquad
R_{\Theta,m}=\mathrm{diag}\!\left(R_{\Theta,m}^{(0)},\dots,R_{\Theta,m}^{(d/2-1)}\right) $$

完整的 RoPE 是对 Q、K 各乘一个块对角正交矩阵：

$$ q_m=R_{\Theta,m}W_qx_m,\qquad k_n=R_{\Theta,n}W_kx_n $$

用复数写更紧凑：把 $(x_{2i},x_{2i+1})$ 看成 $x_{2i}+\mathrm{i}x_{2i+1}$，则 $f(x,m)=e^{\mathrm{i}m\theta_i}x$，内积取实部。

三个必须说清的细节：

- **只作用于 Q、K，不作用于 V。** 相对位置性质来自 $q^\top k$，V 只是被 softmax 权重聚合的内容，加位置没有收益。
- **没有可学参数。** $\theta_i$ 由公式给定，$\cos/\sin$ 表按 $m$ 预计算一次、所有前向复用，这是后面所有扩展方法「零额外开销」的前提。
- **配对方式有两种。** 原论文用相邻对 $(x_{2i},x_{2i+1})$；LLaMA/HF 实现用 $(x_i,x_{i+d/2})$。两者只差一个维度置换，数学等价。

### 2. 相对位置性质

把旋转矩阵代入打分：

$$ \langle R_mq,\,R_nk\rangle=q^\top R_m^\top R_nk $$

$R_m^\top=R_{-m}$（每个二维块是正交矩阵，转置等于反向旋转），而 $R_{-m}R_n=R_{n-m}$（同平面的旋转角度可加）。于是

$$ \boxed{\ \langle R_mq,\,R_nk\rangle=\langle q,\,R_{n-m}k\rangle\ } $$

两个位置的绝对信息在打分里被消掉，只剩 $n-m$。注意「RoPE 编码的是绝对位置」这个说法也成立：$R_m$ 只由 $m$ 决定；只是**打分**对绝对位置不敏感。这个区分是后面所有讨论的支点。

### 3. 波长：为什么外推会崩

第 $i$ 个平面走完一整圈需要的 token 数是

$$ \lambda_i=\frac{2\pi}{\theta_i}=2\pi\,\text{base}^{\frac{2i}{d}} $$

$i=0$ 时 $\lambda_0=2\pi\approx6.28$：每 6 个 token 转一圈，纯局部信息。$i$ 增大波长指数增长，最后一个平面在 $d=128$、base $=10000$ 时 $\lambda_{63}=5.44\times10^4$，比 4k 训练长度长一个量级。

把 $L$ 记作预训练长度，定义**每个平面在训练窗口内转了多少圈**：

$$ r(i)=\frac{L}{\lambda_i}=\frac{L}{2\pi\,\text{base}^{2i/d}} $$

- $r(i)\gg1$（波长远短于 $L$）：转了很多圈，模型只能看到相对相位差 → 纯相对位置。
- $r(i)<1$（波长比 $L$ 还长）：训练全过程连一圈都没转完，$m$ 与 $r$ 一一对应 → 这批维度实际充当**绝对位置编码**（YaRN 论文的观察）。

外推失败的机制在第二类维度：训练时 $m\theta_i$ 只覆盖 $[0,L\theta_i)\subset[0,2\pi)$，直接取 $m=2L$ 就把角度推到 $\ge2\pi$，落到模型从未见过的相位上，attention logits 失控。第一类维度看似安全，但当相对距离远大于训练中出现的距离时，它们同样在振荡（第 4 节有数字）。

### 4. 位置扩展的四种做法

| 方法 | $g(m)$ | $h(\theta_i)$ | 谁被动 | 需要微调 | 外推能力 | 短距离损失 |
| --- | --- | --- | --- | --- | --- | --- |
| PI | $m/s$ | $\theta_i$ | 全部平面一起压缩 $s$ 倍 | 需要（相对更贵） | 论文观察 $s\approx8$ 是可恢复上限 | 有，$s$ 越大越明显 |
| NTK-aware | $m$ | $\text{base}'^{-2i/d}$ | 高频几乎不动、低频多插值 | 无需或极少 | 有限，且 $s$ 与真实倍率不严格对应 | 小 |
| NTK-by-parts | $m$ | 按 $r(i)$ 分区插值 | 只有低频被插值 | 需要（效果最好） | 好 | 小 |
| YaRN | $m$ | 同 NTK-by-parts | 同上，另加 attention 温度 | 需要 | 论文报告 4k 模型可外推到 128k 量级 | 小，官方基准接近原模型 |

**PI（Position Interpolation）**：把位置索引压回训练范围，$f'(x_m,m)=f(x_m,mL/L')=f(x_m,m/s)$。等价于所有 $\theta_i$ 乘 $1/s$，最大角度重新落回训练区间。代价是所有频率一律被压缩，高频的局部区分能力被削平——YaRN 论文用「PI 移除了 RoPE 的高频成分，$s$ 越大越不可恢复」来解释它为什么在 $s\approx8$ 之后即使微调也会退化。

**NTK-aware**：不做统一压缩，而是改 base。校准条件是「最慢的那一维按 $s$ 倍插值，最快的一维保持不变」：

$$ b'^{\,1-\frac{2}{|D|}}=s\,b^{\,1-\frac{2}{|D|}}\ \Longrightarrow\ b'=b\cdot s^{\frac{|D|}{|D|-2}} $$

在 base 变化下 $\theta_i'=b'^{-2i/|D|}$，$i=0$ 处 $\theta_0'=\theta_0=1$ 完全不动，$i$ 越大被压得越多——压力被摊到各维度上，而不是平均分摊。它的毛病是「$s$ 该取多少」只能靠试：论文明确指出最优 base 通常要靠实验确定，Code Llama 的 RoPE ABF 直接手工把 base 调到 1M 就是这种经验做法。

**NTK-by-parts**：不再隐式调节，按 $r(i)$ 显式分三区，用 ramp 平滑过渡（$\alpha=1,\ \beta=32$ 是论文给出的 Llama 系推荐值）：

$$ \gamma(r)=\begin{cases}0 & r<\alpha\\ 1 & r>\beta\\ \dfrac{r-\alpha}{\beta-\alpha} & \text{otherwise}\end{cases}\qquad
h(\theta_i)=\bigl(1-\gamma(r(i))\bigr)\frac{\theta_i}{s}+\gamma(r(i))\,\theta_i $$

$\gamma=1$ 不插值、$\gamma=0$ 全插值：短波长（$r>\beta$，纯相对位置、决定邻近 token 次序）原样保留，长波长（$r<\alpha$，承担绝对位置）完全插值避免外推，中间带线性过渡。

**YaRN**：NTK-by-parts 加上一个 attention 温度。插值把所有角度压小，相对距离的信息量随之下降，attention 分布会变平、熵升高（更接近均匀分布）。YaRN 在 softmax 上用温度 $t$ 补偿：

$$ \mathrm{softmax}\!\left(\frac{q_m^\top k_n}{t\sqrt{|D|}}\right),\qquad \sqrt{1/t}=0.1\ln s+1 $$

这条公式是论文在 LLaMA 7B/13B/33B/65B 上按最低 perplexity 拟合出来的，实测 $s=8$ 时 $\sqrt{1/t}\approx1.208$。实现上把它并进 $\cos/\sin$ 表（把旋转表整体乘 $\sqrt{1/t}$）即可，零额外开销，也不改动 attention 代码。

**Dynamic Scaling**：训练长度 $L$、目标长度 $L'$ 固定时，短序列也在按 $L'$ 的口径插值，短文本会掉点；改成每次前向按当前序列长度取 $s=\max(1,l'/L)$，序列变长时平滑退化而不是突然崩。与 NTK-aware 组合就是工程里常见的 dynamic NTK（Qwen 早期版本用过），与 YaRN 组合是 Dynamic-YaRN（论文报告不微调也能有 2 倍以上扩展）。注意 KV cache：使用动态 $s$ 时必须缓存**未施加 RoPE** 的 K，否则历史 token 的旋转表会与当前 $s$ 不一致。

### 5. 一句话收束

RoPE 的数学给了两个性质：绝对位置可旋转、打分只看相对距离。长度扩展的本质，是在不破坏高频（局部次序）信息的前提下，把低频（长程、近似绝对位置）的周期拉长——PI 靠统一压缩，NTK-aware 靠改 base 摊压力，YaRN 靠按波长分区加温度补偿。

## 数值与代码验证

以下数字全部按上面的公式复算，单头维度 $d=128$（64 个二维平面）、base $=10000$。索引口径统一用**平面序号 $i=0,\dots,63$**（每对相邻两维算一个平面；RoFormer/YaRN 按复数分量编号 $\theta_d=b^{-2d/|D|}$、$d=0,\dots,|D|/2-1$，与这里的 $i$ 一一对应）。

**波长与分区边界（$L=4096$，YaRN 论文取值 $\alpha=1,\beta=32$）**

| 判据 | 临界平面 | 含义 |
| --- | --- | --- |
| $r(i)=L/\lambda_i=\beta=32$ | $i\approx20.9$ | $i\le20$ 完全不插值 |
| $r(i)=1$ | $i\approx45.0$ | $r(i)<1$ 首次出现在 $i=46$，$i\ge46$ 全插值 |
| $\lambda_i=L$ | $i\approx45.0$ | 波长超过训练长度的平面从 $i=46$ 起 |

按 $r(i)$ 三区归类（单位是 **64 个平面**）：**不插值 21 个（32.8%，$i\le20$）、ramp 过渡 25 个（39.1%，$i=21\dots45$）、全插值 18 个（28.1%，$i\ge46$）**；$L=2048$ 时对应 17 / 24 / 23。被保护的高频平面接近三分之一，而 PI 把 64 个平面全部按同一因子压缩——这是它明显损伤局部信息的直接原因。

| 平面 $i$ | $\theta_i$ | $\lambda_i$ | $L/\lambda_i$（$L=4096$） | 处理 |
| --- | --- | --- | --- | --- |
| 0 | 1 | 6.28 | $6.5\times10^2$ | 不插值 |
| 16 | 0.1 | 62.8 | 65.2 | 不插值 |
| 24 | 0.0316 | 198.7 | 20.6 | ramp |
| 32 | 0.01 | 628.3 | 6.52 | ramp |
| 40 | 0.00316 | 1987 | 2.06 | ramp |
| 48 | 0.001 | 6283 | 0.652 | 全插值 |
| 56 | $3.16\times10^{-4}$ | $1.99\times10^4$ | 0.206 | 全插值 |
| 63 | $1.15\times10^{-4}$ | $5.44\times10^4$ | 0.075 | 全插值 |

**base 调节（$\text{base}'=\text{base}\cdot s^{|D|/(|D|-2)}$，$|D|=128$，指数 $128/126\approx1.0159$）**

| $s$ | base$'$ | 最慢维度插值倍率 $\lambda'/\lambda$ | $\sqrt{1/t}=0.1\ln s+1$ |
| --- | --- | --- | --- |
| 4 | 40,890 | 4.000 | 1.139 |
| 8 | 82,685 | 8.000 | 1.208 |
| 16 | 167,199 | 16.00 | 1.277 |
| 32 | 338,097 | 32.00 | 1.347 |

（单头维度 64 时指数是 $64/62\approx1.0323$，同一 $s$ 下 base 涨得更快。）Code Llama 手工把 base 设成 1M；用 $b'=b\cdot s^{|D|/(|D|-2)}$ 反解，$|D|=128$ 时 $s\approx93$（论文 Table 4 把 Code Llama 7B 记作 NTK-aware 的 $4k\times88.6$）——论文特意说明这是经验取值，没有解析解。

**未插值外推时高频分量的相位**（平面序号），相对距离 $n-m=4096$ 时：$i=0$ 的平面相位是 $4096\theta_0=4096$ rad，$i=16$ 是 $409.6$ rad，$i=32$ 是 $40.96$ rad。单对分量的点积会被相位 $\phi$ 乘上 $\cos\phi$：$\phi=0.5$ rad 时只剩 0.878，$\phi=\pi/2$ 时为 0，$\phi=\pi$ 时翻成 $-1$。所以外推时这批平面的打分是「随机符号 + 随机幅度」，attention 失去结构。

**温度补偿的效果**（演示用 logits，$s=8$ 时 $\sqrt{1/t}\approx1.208$，即 $t=1/1.208^2\approx0.685$）：$[20,22]$ 经 softmax 得 $[0.1192,0.8808]$；按 $\mathrm{softmax}(q^\top k/(t\sqrt{|D|}))$ 把 logits 除以 $t$ 得 $[29.18,32.10]$，softmax 后是 $[0.0513,0.9487]$——分布变尖，抵掉插值带来的熵增。实现上把旋转表乘 $\sqrt{1/t}$ 会让 Q、K 各放大一次，等价于 logits 放大 $1/t=1.459$，而不是只放大 $1.208$。

**YaRN 的实测口径（论文 Table，源文口径）**：Llama 2 7B，$s=16$、400 步、全局 batch 64、64k 数据 + sliding window perplexity（$S=256$）在 10 篇 128k Proof-pile 文档上：8k $3.51$ / 16k $2.99$ / 32k $2.65$ / 64k $2.42$ / 128k $>10^1$；从 $s=16$ checkpoint 再训 200 步得到 $s=32$ 模型（仍是 64k 训练数据）：$3.56/3.04/2.70/2.45/\mathbf{2.37}$。**只有 $s=32$ 那档真正外推到 128k，$s=16$ 在 128k 上已经崩了**；论文强调 $s=32$ 只用 64k 数据训过，属于「train short, test long」。13B 同口径对应 $3.29/2.83/2.53/2.31/2.24$。成本面：7B 从 2k 扩到 32k 用 128 A100-hours，而 PI 的 2k×8→16k 要 640 A100-hours；摘要口径是比此前方法少 10 倍 token、少 2.5 倍训练步数，微调数据量不到预训练数据的约 0.1%。

**检索任务上的表现（论文 passkey retrieval 口径）**：把五位数字 passkey 随机插进 8k~128k 的上下文里，YaRN 微调后的 7B 在 64k（$s=16$）上平均准确率 96.3%、在 128k（$s=32$）上 99.4%；13B 两档分别是 97.5% 与 99.4%，论文认为 $s=16$ 那档在检索任务上可能训练不足。$s=32$ 的模型只见过 64k 训练数据，却能在 128k 上取到 passkey，这是「train short, test long」最直接的证据。论文同时提醒 perplexity 不是有效上下文的充分指标：Code Llama 13B 在 128k 上 perplexity 已明显上升，passkey 仍能取到。

```python
import math

D = 128                      # 单头维度
BASE = 10000.0
thetas = [BASE ** (-2 * i / D) for i in range(D // 2)]

# 相对位置不变性：任意平移 (m, n) -> (m+c, n+c)，打分不变
def rope_2d(x, m):
    out = []
    for i, th in enumerate(thetas):
        c, s = math.cos(m * th), math.sin(m * th)
        a, b = x[2 * i], x[2 * i + 1]
        out += [a * c - b * s, a * s + b * c]
    return out

q = [1.0, 0.3, -0.7, 0.2] * (D // 4)
k = [0.5, -0.9, 0.4, 0.8] * (D // 4)

def score(m, n):
    return sum(a * b for a, b in zip(rope_2d(q, m), rope_2d(k, n)))

s0 = score(0, 2)
print(f"{s0:.12f}", [f"{score(m, m + 2):.12f}" for m in (5, 100, 4096, 2 ** 20)])
# 五个值全部等于 s0（最后一个差 3e-11 量级，是 1e4 量级三角函数参数下的浮点误差）：
# 打分只依赖 n-m

# YaRN 的三区划分（d=128 共 64 个平面，L=4096 训练长度）
L, alpha, beta = 4096.0, 1.0, 32.0
lam = [2 * math.pi / th for th in thetas]          # 每个平面的波长
r = [L / l for l in lam]                           # 训练窗口内转过的圈数
gamma = [0.0 if x < alpha else 1.0 if x > beta else (x - alpha) / (beta - alpha) for x in r]
print("不插值平面数:", sum(g == 1.0 for g in gamma), "/ 64")   # 21
print("全插值平面数:", sum(g == 0.0 for g in gamma), "/ 64")   # 18
print("ramp 平面数:", sum(0.0 < g < 1.0 for g in gamma), "/ 64")  # 25

# YaRN 温度（s=32）
s = 32
print("1/sqrt(t) =", 0.1 * math.log(s) + 1, "-> t =", 1 / (0.1 * math.log(s) + 1) ** 2)
```

## 常见追问

- **追问**：RoPE 为什么不作用于 V？
  - 要点：相对位置性质来自 $q^\top k$ 的内积；V 只按 softmax 权重做加权求和，不参与打分，旋转它既不增加相对位置信息，也会改变输出的数值尺度。
- **追问**：相邻配对 $(x_{2i},x_{2i+1})$ 和半维配对 $(x_i,x_{i+d/2})$ 有区别吗？
  - 要点：只差一个固定的维度置换，不改变点积、不改变表达力。差别只在实现（半维配对在 KV cache 里是两次切片，写 kernel 更顺）。
- **追问**：用了 dynamic NTK，KV cache 要注意什么？
  - 要点：$s$ 随序列长度变化时每个历史 token 的旋转角都会变，所以必须缓存**未旋转**的 K（或缓存位置本身），不能缓存已经乘过 RoPE 的 K。
- **追问**：为什么插值之后 attention 会变平，加个温度就能救？
  - 要点：插值把所有相位差压小，不同 key 的打分差异变小、softmax 熵升高，分布趋于均匀；温度等价于把 logits 整体放大，抵消熵增。论文用 896 篇 16k 文档验证了 $1/\sqrt{t}$ 对 perplexity 的影响与数据、位置都近似无关，所以能拟合成 $0.1\ln s+1$ 这种只依赖 $s$ 的形式。
- **追问**：把上下文扩到 128k，等于模型真的能用好 128k 吗？
  - 要点：扩展只保证位置编码不再越界；「注意力是否真落在远处关键 token 上」是另一个问题，需要 passkey retrieval、RULER 这类探针和 lost-in-the-middle 的评估分别看。
- **追问**：PI 说的「少量微调」和 YaRN 的「0.1%」是什么口径？
  - 要点：都是**扩展阶段**的额外预算，不是重训。PI 论文的口径是「比原预训练少几个数量级的 token」（原文举例是几十亿量级），复现时是千步量级（论文附录的对照实验里，同一目标 4k×2 下 PI 用 1000 步、YaRN 只用 400 步就取得接近的 perplexity）；YaRN 的口径是 $s=16$ 用 400 步 × 全局 batch 64 × 64k 序列，摘要里以「比此前方法少 10 倍 token、少 2.5 倍训练步数」和「不到原预训练数据约 0.1%」概括。三个数字口径不同，不能互相换算。
- **追问**：YaRN 是「训练无关」的方法吗？
  - 要点：不是。NTK-by-parts 阶段需要微调才能达到论文报告的效果（7B 扩 32k 约 128 A100-hours），成本优势来自步数少（400 步量级）和可复用前序 checkpoint（$s=32$ 只再训 200 步）；只有 Dynamic-YaRN 那种按序列长度动态取 $s$ 的推理期用法才不微调，且扩展倍数有限（论文报告 2 倍以上）。
- **追问**：RoPE 之外还有别的路吗？
  - 要点：ALiBi 用与距离成正比的线性 bias，外推更好但表达力受限；NoPE 一类做法直接去掉显式位置编码，靠因果 mask 隐式提供顺序信息。它们与 RoPE 的取舍点都在「局部次序 vs 长程外推」。

## 公司变体

- **Meta**：偏工程实现与训练配方。LLaMA 系列全部使用 RoPE，讨论常落在 base 取值、实现细节（半维配对）与「延长预训练比事后插值更划算」这条路线——Llama 3.1 的 config 用的是 `rope_type: llama3`、`rope_theta: 500000`、`factor: 8`、`original_max_position_embeddings: 8192`，即按频率分段的缩放策略（`low_freq_factor=1.0`、`high_freq_factor=4.0`），而不是原样套 YaRN。
- **Moonshot AI**：偏工程落地，YaRN 是配置项而不是要现场推导的公式。Kimi K2 的 config 是 `rope_scaling.type: yarn`、`factor: 32.0`、`original_max_position_embeddings: 4096`、`rope_theta: 50000`、`max_position_embeddings: 131072`——正好是 4k → 128k 的 $s=32$ 口径，容易被问到「这几个字段各自控制什么、改错了会怎样」。
- **阿里巴巴**：偏调用与调参口径。Qwen 早期版本用过 Dynamic NTK（YaRN 论文点名 Qwen 7B），Qwen2.5 的 config 则是 `rope_theta: 1000000`、`max_position_embeddings: 32768`，即走「直接放大 base + 原生长上下文预训练」，而不是推理期插值。

三家都不太会要求你在白板上默写完整推导；按公开模型 config 与论文推断，差异在于 Meta 更容易被追问「为什么不直接用 YaRN 而要延长预训练」，Moonshot 更容易被追问配置字段与部署细节，阿里更容易被追问 base 调大与插值的取舍。以上依据是各家公开 config 与相关论文，不代表具体面试流程。

## 相关题目

- [[llm-internals-07]] —— 位置编码的演进主线：从 sinusoidal、learned 到 RoPE、ALiBi，RoPE 在这一支里的位置。
- [[llm-internals-13]] —— lost-in-the-middle：位置编码不越界不等于模型能用好中间的信息。
- [[llm-internals-16]] —— 一次前向传播的逐张量流程：RoPE 发生在哪一步、$\cos/\sin$ 表在哪里被复用。

## 参考资料与归属

- Amit Shekhar, *Math Behind RoPE (Rotary Position Embedding)*, Outcome School, 2026-04-23 —— <https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding>
- Jianlin Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864, 2021-04-20 —— <https://arxiv.org/abs/2104.09864>
- Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole, *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv:2309.00071, 2023-08-31 —— <https://arxiv.org/abs/2309.00071>
- 模型配置事实来源：`meta-llama/Llama-3.1-8B`（经镜像仓库读取）、`moonshotai/Kimi-K2-Instruct`、`Qwen/Qwen2.5-7B-Instruct` 的 `config.json`。

延伸来源说明：参考博客给出 RoPE 的构造、旋转矩阵与相对位置性质的直觉；PI、NTK-aware、NTK-by-parts、YaRN 的分区公式、$\sqrt{1/t}=0.1\ln s+1$、训练配方与实验数字来自 RoFormer 与 YaRN 两篇论文（含其附录）。波长分区边界、base 数值与温度数值均按公式自行复算，与论文口径不一致处已在正文标注。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
