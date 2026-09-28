---
type: question
id: llm-internals-09
topic: LLM 内部原理与架构
order: 9
question: Chinchilla 缩放定律说了什么？它与更早的缩放直觉有何不同？
question_en: What do the Chinchilla scaling laws say, and how do they differ from earlier scaling intuitions?
asked_at: [Anthropic]
level: 进阶
tags: [scaling-law, chinchilla, 训练]
sources:
  - title: Scaling Laws for Neural Language Models（延伸）
    url: https://arxiv.org/abs/2001.08361
    author: Kaplan et al. (OpenAI)
    published: 2020-01-23
  - title: Training Compute-Optimal Large Language Models（延伸）
    url: https://arxiv.org/abs/2203.15556
    author: Hoffmann et al. (DeepMind)
    published: 2022-03-29
  - title: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws（延伸）
    url: https://arxiv.org/abs/2401.00448
    author: Sardana et al.
    published: 2023-12-31
  - title: Llama 3 Model Card（延伸）
    url: https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
    author: AI@Meta
    published: 2024-04-18
  - title: Llama 2: Open Foundation and Fine-Tuned Chat Models（延伸）
    url: https://arxiv.org/abs/2307.09288
    author: Touvron et al. (Meta AI)
    published: 2023-07-18
  - title: LLaMA: Open and Efficient Foundation Language Models（延伸）
    url: https://arxiv.org/abs/2302.13971
    author: Touvron et al. (Meta AI)
    published: 2023-02-27
related: [llm-internals-10, llm-internals-11]
updated: 2026-09-28
---

## 一句话答案

> Chinchilla 缩放定律回答的是「给定训练算力 $C$，参数量 $N$ 和训练 token 数 $D$ 该怎么分」：两者应当**同比例增长**，即 $N_{opt}\propto C^{0.5}$、$D_{opt}\propto C^{0.5}$，折算成经验比例约 20 tokens/参数。更早的 Kaplan(2020) 给出的是相反的建议——算力增加时大头给参数量（$N\propto C^{0.73}$、$D\propto C^{0.27}$），数据量只要慢慢涨，模型可以训练到远未收敛就停下。Chinchilla 用同算力下 70B/1.4T 全面超过 280B/300B 的 Gopher 验证了修正后的结论，也说明当时的大模型普遍训练不足。但 20:1 只是**训练算力**口径的最优点；把推理成本写进目标函数后，故意让 8B 模型吃下 15T tokens 反而是更划算的工程选择。

## 面试官在考什么

- 是否把 scaling law 理解成**算力分配问题**：约束是 $\mathrm{FLOPs}(N,D)=C$，要解的是 loss 在约束下的最小值，而不是「模型越大越好」这类口号。
- 能否说清 Kaplan 与 Chinchilla 的差别落在**指数**上（0.73/0.27 对 0.5/0.5），并推出同算力下的实际含义：算力涨 10 倍时，前者让参数涨 5.4 倍、数据只涨 1.9 倍，后者要求两者都涨 3.16 倍。
- 能否写出参数化损失 $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$，讲清三项各自的含义，并用拉格朗日法推出 $a=\beta/(\alpha+\beta)$、$b=\alpha/(\alpha+\beta)$。
- 是否掌握证据链而非只记结论：400+ 个模型的实验规模、三条互相独立的拟合路线、同算力同预算的对照实验（70B 对 280B）。
- 是否知道定律的边界：只对 loss/perplexity 外推有效；数据质量、重复率、tokenizer、MoE 的参数口径都会改变曲线；推理成本一旦进入目标函数，最优解就离开 Chinchilla 点。

**常见错误答案**

- 「Chinchilla 说模型越小越好、数据越多越好。」它说的是**同比例**。固定算力下 loss 关于 $N$ 有明确的谷底，模型小过头同样吃亏。
- 「20 tokens/参数 是上限，训多了就是浪费。」20:1 只是「训练算力最优」这一特定目标函数下的解；推理需求足够大时，刻意超出这个比例能降低总成本。
- 「Kaplan 被推翻了，所以幂律是错的。」被修正的是分配指数与实验方法；幂律形式 $L\propto N^{-\alpha}$、$L\propto D^{-\beta}$ 本身被完整继承下来。

## 原理与推导

### 把问题写成带约束的优化

给定训练算力预算 $C$，目标是让最终预训练 loss 最小：

$$\min_{N,D}\;L(N,D)\quad \text{subject to}\quad \mathrm{FLOPs}(N,D)=C$$

对稠密 transformer，每个 token 的前向计算量约为 $2N$ FLOPs（每个参数一次乘加按 2 FLOPs 计），反向约为前向的两倍，因此

$$\mathrm{FLOPs}(N,D)\approx 6ND$$

Chinchilla 论文在拟合效率前沿时用的就是 $6ND$ 这个近似。口径需要交代清楚：$N$ 是非嵌入参数量（Kaplan 的约定），$6ND$ 忽略 attention 的序列平方项、激活重算等开销，只做量级估计。按 $6ND$ 复算 Gopher 得 $5.04\times10^{23}$ FLOPs；论文正文沿用的 Gopher 算力 $5.76\times10^{23}$ 来自 Gopher 论文，比 $6ND$ 高约 12%；附录 F 用含 embedding 与 attention 项的更完整记账重算得 $6.3\times10^{23}$，并说明完整记账与 $6ND$ 的差异在这些模型上很小（表 A4 的比值 0.99–1.10）。引用 Gopher 算力时要写明口径。

### Kaplan 2020：把算力主要投给参数量

Kaplan 等人先分别拟合三条单变量幂律（$N$ 为不含 embedding 的参数量，$D$ 为 token 数，$C_{\min}$ 为最小训练算力）：

$$L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\quad \alpha_N\approx0.076,\qquad L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D},\quad \alpha_D\approx0.095,\qquad L(C_{\min})=\left(\frac{C_c}{C_{\min}}\right)^{\alpha_C},\quad \alpha_C\approx0.050$$

读法：参数翻倍，loss 乘以 $2^{-0.076}\approx0.949$；数据翻倍，loss 乘以 $2^{-0.095}\approx0.936$。两个指数都远小于 1，意味着单纯堆规模只能换来缓慢的改善。把 $N$ 与 $D$ 联合起来，过拟合程度由一个组合量控制：

$$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}$$

论文给出的过拟合判据是组合量 $N^{0.74}/D$（同一个 0.74 也出现在经验关系 $D\propto N^{0.74}$ 里）：模型放大 8 倍，数据只需放大 $8^{0.74}\approx4.7$ 倍（论文表述为「约 5 倍」）就能避免过拟合惩罚。再与 $L(N,S)$、$C\approx6NBS$ 联立（$B$ 为 batch size，$S$ 为优化步数），得到固定算力下的最优分配：

$$N\propto C^{0.73},\qquad B\propto C^{0.24},\qquad S\propto C^{0.03},\qquad D=B\cdot S\propto C^{0.27}$$

注意 $0.73+0.27=1$，与 $C\approx6ND$ 自洽。它的工程含义是：算力涨 10 倍，模型放大 $10^{0.73}\approx5.4$ 倍，数据只增加 $10^{0.27}\approx1.9$ 倍。Chinchilla 论文在复述这条结论时写作「模型 5.5 倍、数据 1.8 倍」，对应的是 $10^{0.74}$、$10^{0.26}$ 这组取整指数，与 0.73/0.27 是同一结论的不同取整口径。配套的定性结论是「训练非常大的模型并显著早停」：大模型样本效率更高，没收敛就停下来反而更划算。

### Chinchilla：重做实验，指数变成对半

Hoffmann 等人指出 Kaplan 的实验设置有两个系统性偏差，都会把结论推向「堆参数」：

1. **学习率 schedule 被固定**。Kaplan 对所有模型用同一条 cosine schedule；Chinchilla 发现 schedule 长度应当匹配实际训练 token 数。当 schedule 固定到 130B tokens 时，训练量少于此长度的中间 checkpoint 的 loss 被高估，于是「多喂数据」的收益被系统性低估。
2. **模型规模区间不同**。Kaplan 的多数 run 小于 100M 参数；Chinchilla 覆盖 70M–16B（多数大于 500M），并在 FLOP-loss 前沿上观察到轻微曲率，从小模型区间外推会放大偏差。

新实验规模是 400+ 个模型、$N$ 从 70M 到 16B、$D$ 从 5B 到 500B tokens，每个配置训练 4 个不同长度的 horizon。三条互相独立的拟合路线给出几乎一致的答案：

| 拟合方法 | $N_{opt}\propto C^{a}$ | $D_{opt}\propto C^{b}$ |
| --- | --- | --- |
| Approach 1：固定模型规模、变 token 数 | $a=0.50$ | $b=0.50$ |
| Approach 2：IsoFLOP 剖面（固定算力扫模型规模，找 loss 谷底） | $a=0.49$ | $b=0.51$ |
| Approach 3：拟合参数化损失函数 | $a=0.46$ | $b=0.54$ |

$a\approx b\approx0.5$ 的直白读法就是：**模型规模翻倍，训练 token 数也翻倍**。

### 参数化损失与闭式最优解

Approach 3 假设最终 loss 具有下面的形式：

$$\hat{L}(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$

三项各有明确含义：

- $E$：数据分布本身的不可约熵，即理想生成过程能到达的 loss 下限；
- $A/N^{\alpha}$：**容量不足**项。参数有限的 transformer 即使在无限数据上训练，也达不到理想生成过程；
- $B/D^{\beta}$：**训练不足**项。优化步数有限、只见过数据集的一个样本，模型没有收敛。

参数 $(A,B,E,\alpha,\beta)$ 通过最小化 $\log\hat{L}$ 与观测 $\log L$ 之间的 Huber loss（$\delta=10^{-3}$）来拟合，用 L-BFGS，并从多组初值中取最优以避免局部极小。

在 $6ND=C$ 约束下解这个优化问题。写拉格朗日函数 $\mathcal{L}=E+AN^{-\alpha}+BD^{-\beta}+\lambda(6ND-C)$，对 $N$ 与 $D$ 求偏导并置零：

$$\alpha A N^{-\alpha-1}=6\lambda D,\qquad \beta B D^{-\beta-1}=6\lambda N$$

两式相除消去 $\lambda$，得到最优点的判据：

$$\alpha\frac{A}{N^{\alpha}}=\beta\frac{B}{D^{\beta}}$$

即两项「欠拟合损失」在最优点的对数导数贡献相等。代入 $D=C/(6N)$ 整理：

$$N^{\alpha+\beta}=\frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}\ \Longrightarrow\ N_{opt}(C)=G\left(\frac{C}{6}\right)^{a},\qquad G=\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\qquad a=\frac{\beta}{\alpha+\beta}$$

对称地有

$$D_{opt}(C)=G^{-1}\left(\frac{C}{6}\right)^{b},\qquad b=\frac{\alpha}{\alpha+\beta}$$

这就是论文的闭式效率前沿。两个可直接使用的推论：

- **指数只由 $\alpha$ 与 $\beta$ 的比值决定**。只有 $\alpha=\beta$（容量项与数据项衰减得一样快）才有 $a=b=0.5$，也就是参数与数据同比例增长。论文三条路线拟合出的 $b$ 分别等于 0.50、0.51、0.54，都接近 0.5。
- **tokens/参数 这个比例与算力无关**。由上式 $D_{opt}/N_{opt}=G^{-2}(C/6)^{b-a}$，当 $a=b$ 时退化为常数 $G^{-2}=(B/A)^{1/\alpha}$，与 $C$ 无关。这正是「多少 tokens/参数」能当成经验法则的原因——它描述的是效率前沿的方向，不是某个算力点上的巧合。

Chinchilla 自身的配置是 70B 参数配 1.4T tokens，比值恰好 20，这也是「约 20 tokens/参数」这一说法的来源。它是**该论文数据分布与 tokenizer 口径下的经验值**，换分布就要重新拟合。

### 关键证据：同算力 A/B 对照

Gopher 的训练算力是 $5.76\times10^{23}$ FLOPs。Chinchilla 用同一量级的算力（$6ND\approx5.9\times10^{23}$），把参数从 280B 降到 70B（4 倍），token 从 300B 提到 1.4T（4.7 倍）：

| 模型 | 参数 $N$ | tokens $D$ | tokens/参数 | $C\approx6ND$ |
| --- | --- | --- | --- | --- |
| Gopher | 280B | 300B | 1.07 | $5.0\times10^{23}$（论文正文沿用 $5.76\times10^{23}$） |
| Chinchilla | 70B | 1.4T | 20.0 | $5.9\times10^{23}$ |

结果：Chinchilla 在大范围下游任务上超过 Gopher(280B)、GPT-3(175B)、Jurassic-1(178B) 与 MT-NLG(530B)；MMLU 平均准确率 67.5%，按论文原话比 Gopher 提升「超过 7 个百分点」。论文还专门指出，更小的模型同时降低了微调与推理算力——这条观察在两年后变成了「过训练」策略的依据。

对 Gopher 算力预算，论文 Approach 3 外推出的最优模型规模约为 40B；团队最终选了 70B／1.4T 作为可落地的折中点。这说明拟合给出的是前沿上的区间，不是必须命中的靶心。

### 为什么后来的模型不再卡在 20:1

LLaMA 系列沿着「参数与数据同比例增长」这条线继续加数据：LLaMA-1 的 7B 用 1.0T tokens（约 143 tokens/参数），Llama 2 全系列用 2T tokens（7B 约 286 tokens/参数），Llama 3 的 8B/70B 用 15T+ tokens（8B 约 1875 tokens/参数，约为 Chinchilla 比例 20 的 94 倍）。

原因是最优点被**总成本**而不是训练算力决定。服务阶段每生成一个 token 约需 $2N$ FLOPs，服务 $T$ 个 token 就是 $2NT$，于是目标变成

$$\text{cost}\approx 6ND+2NT$$

$T$ 越大，$N$ 在成本里的权重越高，最优解沿着「更小的 $N$、更大的 $D$」移动。Sardana 等人的分析给出定量结论：推理需求到十亿次请求量级时，最优模型应当比 Chinchilla-optimal 更小、训练更久；他们用 47 个模型验证，质量在 tokens/参数 高到 10,000 时仍在改善。同一篇工作也提醒：只用常见比例区间（约 20:1 附近）拟合出来的曲线，会高估极端区间里额外 token 的收益。

代价与前提要说清：过训练买到的是「同等质量下更便宜的推理」，不是更高的质量上限；如果数据本身不够或重复率过高，多喂 token 的边际收益会明显衰减。

### 定律的边界

- **只保证 loss/perplexity 的平滑外推**。下游 benchmark 的指标是 loss 的非线性函数（accuracy 带阈值），loss 平滑下降可以表现为某个 benchmark 上的「突然涌现」。涌现是真实的能力突变还是度量方式的假象，学界仍有争论——用 scaling law 预测下游能力时必须说明用的是哪种指标。
- **数据分布与质量会改变曲线**。重复数据、数据配比、多语言比例、tokenizer 都改变 $D$ 的有效口径。把英文配方直接外推到多语言场景，$B/D^{\beta}$ 的系数不再成立；数据质量提升等价于放大了有效 $D$，这也是「同样 15T tokens，效果差很多」的原因之一。
- **参数口径必须交代**。$N$ 是不含 embedding 的非嵌入参数还是总参数、MoE 是总参数还是激活参数，会直接改变 tokens/参数 的读数。MoE 的 scaling 还需要额外的「专家数」维度，把总参数直接代入 $6ND$ 会低估算力 → [[llm-internals-10]]。
- **外推的是趋势，不是点估计**。系数依赖模型规模区间与学习率 schedule 的设置，换区间、换 schedule 口径，指数就会漂移；这也是 Kaplan 与 Chinchilla 分歧的主要来源。

## 数值与代码验证

### 表 1：公开报告里的 tokens/参数（按 $D/N$ 复算）

| 模型 | $N$ | $D$ | $D/N$ | $6ND$ |
| --- | --- | --- | --- | --- |
| LaMDA | 137B | 168B | 1.23 | $1.4\times10^{23}$ |
| GPT-3 | 175B | 300B | 1.71 | $3.2\times10^{23}$ |
| Jurassic-1 | 178B | 300B | 1.69 | $3.2\times10^{23}$ |
| Gopher | 280B | 300B | 1.07 | $5.0\times10^{23}$ |
| MT-NLG 530B | 530B | 270B | 0.51 | $8.6\times10^{23}$ |
| Chinchilla | 70B | 1.4T | 20.00 | $5.9\times10^{23}$ |
| LLaMA-1 7B | 7B | 1.0T | 142.86 | $4.2\times10^{22}$ |
| Llama 2 7B | 7B | 2.0T | 285.71 | $8.4\times10^{22}$ |
| Llama 3 8B | 8B | 15T | 1875.00 | $7.2\times10^{23}$ |

前六行的参数与 token 数取自 Chinchilla 论文 Table 1，后三行取自 LLaMA、Llama 2、Llama 3 的公开报告与模型卡。$6ND$ 一列是按 $6ND$ 口径复算的，与论文引用的算力数字并不相同：Gopher 复算 $5.04\times10^{23}$，论文正文沿用 Gopher 论文报告的 $5.76\times10^{23}$（$6ND$ 比它低约 12%），附录 F 的更完整记账则为 $6.3\times10^{23}$（与 $5.76\times10^{23}$ 相差不到 10%）。所以这一列只用于横向比较量级，引用具体算力数字时要注明口径。这张表最值得记住的是 $D/N$ 一列：2022 年以前的主流大模型都在 0.5–1.7 之间，Chinchilla 把它推到 20，Llama 3 的 8B 则推到 1875。

### 表 2：算力涨 10 倍时如何分配

| 口径 | 参数量 $N$ | 数据量 $D$ |
| --- | --- | --- |
| Kaplan 2020（$0.73/0.27$） | $10^{0.73}\approx5.4\times$ | $10^{0.27}\approx1.9\times$ |
| Chinchilla 论文复述 Kaplan 时的口径（$0.74/0.26$） | $10^{0.74}\approx5.5\times$ | $10^{0.26}\approx1.8\times$ |
| Chinchilla（$0.5/0.5$） | $10^{0.5}\approx3.16\times$ | $10^{0.5}\approx3.16\times$ |

Chinchilla 口径下 $N\cdot D$ 正好涨 10 倍，与 $C\approx6ND$ 一致；Kaplan 口径下 $5.4\times1.9\approx10.2$，同样自洽——两种口径都满足算力约束，区别只在把新增算力投向哪一边。

### 代码：自洽性检查与指数换算

```python
import math

# 1) 复算 Chinchilla 配置：C ≈ 6ND 与 D/N ≈ 20 互相自洽
C = 6 * 70e9 * 1.4e12
print(f"C = {C:.2e}")          # 5.88e+23，与 Gopher 的 5.76e23 同量级
r = 20.0                       # 目标 tokens/参数
N = math.sqrt(C / (6 * r))     # 由 N*D = C/6 与 D = r*N 联立解出
D = math.sqrt(r * C / 6)
print(f"N = {N:.3e}, D = {D:.3e}")   # 7.000e+10, 1.400e+12
```

```python
# 2) 从论文报告的 a、b 反推拟合出的 alpha/beta 比值（b/a = alpha/beta）
for name, a, b in [("Approach 1", 0.50, 0.50), ("Approach 2", 0.49, 0.51), ("Approach 3", 0.46, 0.54)]:
    print(f"{name}: a={a:.2f} b={b:.2f} -> alpha/beta={b/a:.3f}")
# Approach 1 -> 1.000（参数与数据同比例）
# Approach 2 -> 1.041
# Approach 3 -> 1.174（alpha 更大，最优点偏向多喂数据）

# 3) 10 倍算力下的分配对比
for name, a, b in [("Kaplan", 0.73, 0.27), ("Chinchilla", 0.50, 0.50)]:
    print(f"{name}: N x{10**a:.2f}, D x{10**b:.2f}")
# Kaplan: N x5.37, D x1.86
# Chinchilla: N x3.16, D x3.16
```

第 2 段用到的恒等式由 $a=\beta/(\alpha+\beta)$、$b=\alpha/(\alpha+\beta)$ 直接相除得到，不需要知道 $A$、$B$ 的绝对值——这也是面试里被追问「拟合系数是多少」时的正确答法：系数依赖数据分布与 tokenizer，可迁移的是形式与指数关系。

## 常见追问

- **追问**：为什么同算力下「更小的模型 + 更多数据」反而 loss 更低？
  - 要点：loss 是 $N$ 与 $D$ 的联合函数，最优时两项欠拟合损失满足 $\alpha A/N^{\alpha}=\beta B/D^{\beta}$。算力固定意味着 $N\cdot D$ 固定，若 $D$ 相对不足，$B/D^{\beta}$ 项主导，多喂数据的边际收益高于加大模型；Kaplan 的过拟合判据 $N^{0.74}/D$ 说的也是同一件事。IsoFLOP 剖面上，固定算力下的 loss 关于 $N$ 有明显谷底；论文据此判断同 Gopher 算力下最优模型应比 280B 的 Gopher 小约 4 倍。
- **追问**：两次拟合的差异是数据问题还是方法问题？
  - 要点：方法。Kaplan 用固定 token 数与固定 cosine schedule（到 130B tokens），使训练量更少的中间 checkpoint 的 loss 被高估，从而低估「多喂数据」的收益；Chinchilla 让 schedule 长度匹配 token 数后，结论反转。此外 Kaplan 的 run 多数小于 100M 参数，外推到 100B+ 会放大偏差。做任何自己的拟合，前提都是学习率 schedule 与训练长度匹配、数据分布一致。
- **追问**：20 tokens/参数 能当铁律用吗？
  - 要点：不能。它是「训练算力最优」这一目标函数下的经验值，依赖数据分布、tokenizer 与拟合区间。换目标函数（把推理与微调算力、数据获取成本算进去）最优点就移动；论文自己也倾向于给前沿区间而非单点。
- **追问**：把推理成本算进去，最优配置怎么变？
  - 要点：服务 $T$ 个 token 的推理算力约 $2NT$，总成本写成 $6ND+2NT$，$T$ 越大越偏向小 $N$、大 $D$。极端区间（数千到上万 tokens/参数）质量仍在改善，但用常见比例区间拟合出的曲线会高估该区间的收益，所以「过训练」的收益要按自己的推理量单独测算。
- **追问**：MoE 上怎么套 Chinchilla？
  - 要点：算力口径要用**激活参数**（$6N_{act}D$），而容量与记忆由**总参数**决定，两者不能混。MoE 的 scaling 需要额外的专家数维度，已有工作发现专家带来的收益随总规模变大而递减，所以把稠密模型的 20:1 直接搬到 MoE 上会算错预算 → [[llm-internals-10]]。
- **追问**：scaling law 能预测下游能力吗？
  - 要点：loss/perplexity 可以平滑外推，下游 metric 不行——accuracy 之类的指标是 loss 的非线性（带阈值）函数，同一条 loss 曲线可以在某个 benchmark 上呈现「突然涌现」，换一个更平滑的 metric 涌现就消失。安全评估、指令遵循这类能力尤其难外推，必须实测。

## 公司变体

本题标注的出处是 Anthropic。从公开材料看，这类团队问这道题时偏**原理推导与决策口径**，而不是让你复述某家的数据配方：

- 公式本身会被追问：$L(N,D)$ 三项各自代表什么，$a=\beta/(\alpha+\beta)$ 怎么推出来，$6ND$ 漏掉了哪些开销。
- 外推的可靠性是重点：哪些量能从小规模实验外推（loss、算力最优前沿），哪些不能（下游能力、安全评估），外推需要什么前提（schedule 与 token 数匹配、数据分布一致）。
- 也会落到工程取舍：给定真实的数据总量与推理需求，参数量与 token 数怎么定，该比 Chinchilla-optimal 更小还是更大。

回答时把「结论 + 推导 + 边界」三段都给出。只背 20:1 和「70B 打败 280B」这两个结论，追问一层就会露。

## 相关题目

- [[llm-internals-10]]：MoE 如何在不增加 FLOPs 的前提下扩展容量——总参数与激活参数的口径之分，正是 scaling law 里 $N$ 该怎么取的问题。
- [[llm-internals-11]]：预训练、SFT 与偏好优化的分工——预训练阶段的数据与算力预算怎么定，是这道题的直接下游。
- [[llm-internals-02]]：KV cache 的显存公式——推理成本的结构化来源，也是「宁可过训练小模型」的量化依据。
- [[llm-internals-06]]：BPE 与 tokenization——$D$ 的单位是 token，分词方案一变，tokens/参数 的读数就变。
- [[llm-internals-05]]：FlashAttention——FLOPs 不等于 wall-clock，$6ND$ 只是算力口径的近似。

## 参考资料与归属

1. **Scaling Laws for Neural Language Models** — Kaplan et al. (OpenAI)，2020-01-23，[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)（延伸来源）：$L(N)$、$L(D)$、$L(C_{\min})$ 三条幂律的指数与常数、$L(N,D)$ 的过拟合形式、$N\propto C^{0.73}$ 的分配结论与「8 倍模型对 5 倍数据」的判据。
2. **Training Compute-Optimal Large Language Models** — Hoffmann et al. (DeepMind)，2022-03-29，[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)（延伸来源）：400+ 模型实验、三种拟合路线与指数（0.50/0.50、0.49/0.51、0.46/0.54）、$L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$ 与闭式最优解、$6ND$ 约束、Chinchilla 与 Gopher 的算力/数据/MMLU 对比、Gopher 算力下约 40B 的外推结果。
3. **Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws** — Sardana et al.，2023-12-31，[arXiv:2401.00448](https://arxiv.org/abs/2401.00448)（延伸来源）：「把推理成本写进目标函数后最优模型更小、训练更久」的结论、1B 请求量级的场景设定、质量改善延伸到 10,000 tokens/参数 的 47 模型实验，以及极端区间收益被高估的提醒。
4. **Llama 3 Model Card** — AI@Meta，2024-04-18，[MODEL_CARD.md](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)（延伸来源）：8B/70B 使用 15T+ tokens 预训练。
5. **Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al. (Meta AI)，2023-07-18，[arXiv:2307.09288](https://arxiv.org/abs/2307.09288)（延伸来源）：全系列 2T tokens 的预训练数据量。
6. **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al. (Meta AI)，2023-02-27，[arXiv:2302.13971](https://arxiv.org/abs/2302.13971)（延伸来源）：7B/13B 使用 1.0T tokens 的训练配置。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
