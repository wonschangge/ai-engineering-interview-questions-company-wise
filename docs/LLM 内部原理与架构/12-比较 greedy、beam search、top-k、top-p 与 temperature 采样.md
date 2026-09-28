---
type: question
id: llm-internals-12
topic: LLM 内部原理与架构
order: 12
question: 比较 greedy、beam search、top-k、top-p 与 temperature 采样。它们各自在什么情况下会失败？
question_en: Compare greedy, beam search, top-k, top-p and temperature sampling. When does each fail?
asked_at: [Google DeepMind, Apple, Perplexity]
level: 进阶
tags: [解码, 采样, temperature, top-p]
sources:
  - title: How does Temperature control LLM output?
    url: https://outcomeschool.com/blog/how-does-temperature-control-llm-output
    author: Amit Shekhar (Outcome School)
    published: 2026-09-15
  - title: How do Top-k and Top-p Sampling work?
    url: https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work
    author: Amit Shekhar (Outcome School)
    published: 2026-09-19
  - title: The Curious Case of Neural Text Degeneration（延伸）
    url: https://arxiv.org/abs/1904.09751
    author: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
    published: 2019-04-22
  - title: Six Challenges for Neural Machine Translation（延伸）
    url: https://arxiv.org/abs/1706.03872
    author: Philipp Koehn, Rebecca Knowles
    published: 2017-06-12
  - title: Google's Neural Machine Translation System（延伸）
    url: https://arxiv.org/abs/1609.08144
    author: Yonghui Wu et al. (Google)
    published: 2016-09-26
  - title: Turning Up the Heat：Min-p Sampling for Creative and Coherent LLM Outputs（延伸）
    url: https://arxiv.org/abs/2407.01082
    author: Minh Nhat Nguyen, Andrew Baker, Clement Neo, Allen Roush, Andreas Kirsch, Ravid Shwartz-Ziv
    published: 2024-07-01
  - title: Competition-Level Code Generation with AlphaCode（延伸）
    url: https://arxiv.org/abs/2203.07814
    author: Yujia Li et al. (DeepMind)
    published: 2022-02-08
  - title: CTRL：A Conditional Transformer Language Model for Controllable Generation（延伸）
    url: https://arxiv.org/abs/1909.05858
    author: Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, Richard Socher
    published: 2019-09-11
  - title: transformers 的采样实现（generation/utils.py）（延伸）
    url: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
    author: Hugging Face
related: [llm-internals-16, llm-internals-11]
updated: 2026-09-28
---

## 一句话答案

> 五种策略可以放进同一条流水线：logits $z$ 先做温度缩放 $z/T$，再按 top-k / top-p（或 min-p）截断候选集，softmax 归一化后按概率采样。greedy 与 beam search 是确定性解码，相当于把「截断 + 采样」换成 argmax 或序列级搜索；temperature、top-k、top-p 是随机采样。
> 失效点各不相同：greedy 会重复退化，beam search 有长度偏置、短视、算力随 beam 线性增长，temperature 过高会语义漂移、过低会套话打转，top-k 的固定 $k$ 不适应分布形状，top-p 的累积阈值在模型不确定时会吞下长尾。
> 选型一句话：代码、抽取、工具调用用低温配 greedy 或小 top-p；创意写作提高温度并放宽 top-p；任何评测都必须固定并公开这套解码参数。

## 面试官在考什么

- 能否把解码写成「缩放 → 截断 → 归一化 → 采样」四步，并说清 temperature 作用在 logits 上、top-k / top-p 作用在缩放后的分布上，顺序反过来结果就不同。
- 是否真的理解 temperature 的数学——除以 $T$ 等于把 logit 之间的差放大 $1/T$ 倍——而不是把它当成一个「随机性旋钮」。
- 是否知道 top-k 与 top-p 的失效形状相反：一个是固定候选数不适应分布，一个是累积阈值与单个 token 的质量脱钩。
- 是否理解 beam search 优化的是序列似然而不是文本质量，能说出长度偏置、短视、算力线性增长这三个具体缺陷。
- 能否把参数映射到任务，并意识到「评测口径」也是这题的一部分：pass@$k$ 与 greedy 不兼容，解码参数不固定则两次实验的指标不可比。

常见错误答案：

- 说「temperature 越低越准确」。它不增加任何知识，只改变从已有分布里抽样的方式；$T$ 甚至不改变 token 的排序。
- 说 top-p 是「取概率大于 $p$ 的 token」。阈值作用在累积概率上，被保留的单个 token 概率可以远小于 $p$。

## 原理与推导

### 1. 统一记号与流水线

设模型在某一层输出的 logits 为 $z \in \mathbb{R}^{|V|}$（未归一化分数），温度为 $T>0$，候选集为 $\mathcal{C}\subseteq\{1,\dots,|V|\}$：

$$p_i = \frac{\exp(z_i/T)}{\sum_{j\in\mathcal{C}}\exp(z_j/T)},\quad i\in\mathcal{C}$$

四步的责任划分是这题的骨架：温度缩放 $z/T$ 只改分布的锐度、不改排序；惩罚类 logits 变换改的是个别 token 的分数；top-k / top-p / min-p 决定候选集 $\mathcal{C}$，不改变保留 token 之间的相对概率；softmax 只在 $\mathcal{C}$ 内归一化并抽样。

五种策略在这条流水线上的差别：

| 策略 | 决策规则 | 随机 | 与温度的关系 | 每步开销 |
| --- | --- | --- | --- | --- |
| greedy | $\arg\max_i z_i$ | 否 | 无关（$T$ 不改排序） | $1\times$ |
| beam search | 保留 $B$ 条最高序列分的假设 | 否 | 无关（除 beam-sample） | 约 $B\times$ |
| temperature | 按 $z/T$ 缩放后在全词表采样 | 是 | 就是它本身 | $1\times$ |
| top-k | 固定保留概率最高的 $k$ 个再采样 | 是 | 先温后截时耦合 | $1\times$ |
| top-p | 保留累积概率达到 $p$ 的最小集合再采样 | 是 | 先温后截时耦合 | $1\times$ |

### 2. temperature 的推导

温度只在 softmax 之前做一次除法，把 logit 的差整体乘以 $1/T$：

$$\frac{z_i}{T}-\frac{z_j}{T}=\frac{z_i-z_j}{T}$$

softmax 只依赖 logits 之间的差（分子分母同乘一个常数不改变结果），由此得到三个直接推论：

1. $T=1$ 时不做任何改变，模型原始分布即训练得到的分布。
2. $T<1$ 放大差距，分布更尖锐，最高分 token 的概率被推高，输出更确定、更容易重复；$T>1$ 缩小差距，分布更平坦，长尾 token 拿到真实的概率质量。
3. $\arg\max$ 在正数缩放下不变，所以 **$T$ 不改变排序**；$T\to 0^+$ 时 $\exp$ 的差趋于无穷，分布退化成 one-hot，等价于 greedy。实现上 $T=0$ 不做除法，而是直接跳过采样取 argmax。

平坦程度可以定量：熵 $H(p)=-\sum_i p_i\ln p_i$，有效候选数 $\exp(H)$ 约等于「这一步实际参与竞争的有几个 token」。下面用 4 个 logits 的小例子算一遍。

### 3. top-k：固定大小的候选集

保留概率最高的 $k$ 个 token，其余概率置零后重新归一化：

$$p_i' = \frac{p_i}{\sum_{j\in\text{top-}k}p_j},\quad i\in\text{top-}k$$

实现上不需要真的先 softmax 再重归一化——直接在保留的 $k$ 个 logits 上做 softmax，结果完全相同（源文给出的 torch 片段就是这个技巧）。两个边界：$k=1$ 退化成 greedy，$k=|V|$（或实现里的 $k=0$）等于纯采样。

$k$ 是人为设定的常数，与当前分布的形状无关，这就是它两种相反的失效方式：

- 分布本来很尖锐时 $k$ 相对太宽：源文的例子是 "The capital of France is"，Paris 占 95%，$k=3$ 会把 the（2%）、a（1%）强行拉进候选并重归一化，偶尔生成 "The capital of France is the"。分布很平时 $k$ 又相对太窄：源文的例子是 "My favourite colour is"，blue 12%、red 11%、green 11%、black 10%、purple 10% 基本等价，$k=3$ 直接砍掉了同样合理的 black、purple、yellow。

### 4. top-p：固定概率质量

核采样（nucleus sampling）按概率从高到低累加，保留累积概率首次达到 $p$ 的那个最小集合 $\mathcal{C}_p=\text{top-}m$，其中

$$m=\min\Big\{m'\;\Big|\;\sum_{i=1}^{m'}p_{(i)}\ge p\Big\}$$

$p_{(i)}$ 是降序排列后的概率。同样在集合内重新归一化，再采样。候选数 $m$ 由分布自己决定：模型确信时 $m=1$（"The capital of France is" 里 Paris 95% 已经越过 0.9），模型犹豫时 $m$ 可以到几十上百。工程上常见取值是 $p=0.9$ 或 $0.95$；$p=1.0$ 等于关闭该过滤器，$p\to 0$ 退化成 greedy。

两者的对照可以压成一句话：top-k 固定候选个数（常见 10–50），top-p 固定累积概率（常见 0.9–0.95）；模型确信时 top-k 仍要塞进 $k$ 个弱候选，top-p 可能只剩 1 个；模型犹豫时 top-k 会砍掉同样合理的候选，top-p 自动放宽。

### 5. 顺序与组合：先温度还是先截断

两种口径都会在真实系统里出现，结果不同：

- **先温度、后截断**：截断看到的分布已经是 $z/T$ 的 softmax，所以候选集大小本身随 $T$ 变化——同样的 top-p，低温时候选更少、高温时候选更多。transformers 就是这样实现的。
- **先截断、后温度**：候选集只由未缩放的分布决定，与 $T$ 无关，但截断依据的分布没有考虑调用方真正想要的锐度。也有推理栈把温度排在采样链的截断步骤之后，两种口径都存在，读实现时要看准。

用源文 "The cat sat on the" 的分布（mat 0.40、floor 0.25、sofa 0.15、bed 0.10、roof 0.05，再加一个「其余 token 合计 0.05」的桶，6 项才凑满 1.0）算一遍 top-p = 0.9 的候选数：

| 温度 $T$ | 分布（降序） | 累积到 0.9 的位置 | 候选数 | 重归一化后 |
| --- | --- | --- | --- | --- |
| 0.5 | 0.615 / 0.240 / 0.087 / … | 第 3 项 | 3 | 65.3% / 25.5% / 9.2% |
| 1.0 | 0.40 / 0.25 / 0.15 / 0.10 / … | 第 4 项 | 4 | 44.4% / 27.8% / 16.7% / 11.1% |
| 2.0 | 0.277 / 0.219 / 0.170 / 0.139 / 0.098 / … | 第 5 项 | 5 | 30.7% / 24.3% / 18.8% / 15.4% / 10.9% |

同一个 `top_p=0.9`，低温下等价于更激进的截断，高温下几乎不截断——所以「温度调高、top-p 保持不变」实际是把两个旋钮一起拧了。这也是不要把 temperature 和 top-p 同时调的经验规则背后的机制。

同一位置还会挂其它 logits 变换，它们都在 softmax 之前生效，顺序如下：

$$z \;\to\; \text{惩罚类变换（repetition / presence / frequency）} \;\to\; z/T \;\to\; \text{top-}k \;\to\; \text{top-}p \;\to\; \text{min-}p \;\to\; \text{softmax} \;\to\; \text{采样}$$

- **repetition penalty**（Keskar 等，2019，CTRL）：对已经出现过的 token 改 logit。论文原式是在 softmax 里把它除一次 $\theta$，即 $p_i\propto\exp\big(x_i/(T\theta)\big)$（$\theta>1$，论文在 greedy 下用 $\theta\approx1.2$）；但负 logit 除以 $\theta$ 会朝 0 移动、反而抬高重复 token 的概率，所以 transformers 这类实现改成按符号处理——正 logit 除以 $\theta$、负 logit 乘以 $\theta$（默认值 1.0，即不惩罚）。它是有状态的，跨步累积。
- **presence / frequency penalty**（OpenAI 风格的接口）：加性而非乘性，presence 对出现过的 token 加一个常数，frequency 按出现次数线性累加。
- **min-p**：把阈值改成相对量 $\tau=p_{\max}\cdot p_{\min}$，只保留 $p_i\ge\tau$ 的 token。transformers 的文档给出的典型区间是 0.01–0.2，语义上对应 top-p 的 0.99–0.8。它与温度的方向刚好和 top-p 相反：温度升高使 $p_{\max}$ 变小、阈值随之变小，候选集自动放宽，所以它是为「高温 + 保持连贯」设计的；transformers 的源码注释也明确写了 min-p 必须在温度缩放之后应用。

### 6. 失败场景逐条

**greedy：重复与退化。** 每步都取最大概率，等价于在整条序列上做一步贪心搜索，没有回退。源文给的现象是 "I think that I think that I think that" 这类自我循环；Holtzman 等（2019）把「用似然作为解码目标」的后果总结为平淡且异常重复，并指出这与模型质量本身无关——同一套权重换解码策略，文本质量差异很大。第二个问题是确定性：同一个 prompt 只有一条输出，任何靠「采样多个候选再筛选」（自洽性投票、单元测试过滤、best-of-$n$ 重排）的手段都用不上。

- **beam search 的长度偏置**：$\log P(Y|X)=\sum_t \log p(y_t|\cdot)$ 每一步都是负数，序列越长总分越低。GNMT 的原话是，不加长度归一化时，常规 beam search 平均会偏向更短的结果，因为每一步都在往上加一个负的 log 概率。用「每步平均条件概率都是 0.8」这个口径复算：长度 5 时 $\log P=-1.116$，长度 20 时 $\log P=-4.463$，差的 3.35 nats 完全来自长度而不是质量。GNMT 的修正是
  $$s(Y,X)=\frac{\log P(Y|X)}{lp(Y)}+cp(X;Y),\qquad lp(Y)=\frac{(5+|Y|)^{\alpha}}{(5+1)^{\alpha}}$$
  其中 $cp$ 是 coverage penalty，$\alpha$ 控制长度归一化的强度。$\alpha$ 的口径要注意：$\alpha\in[0.6,0.7]$ 是更早那个「分数除以 $|Y|^{\alpha}$」的简化归一化在开发集上调出的经验区间，而上面这个打分函数用的是 $\alpha=0.2$、$\beta=0.2$——论文按 Google 内部数据调出、并作为全文实验的统一设置；论文也指出 RL 精调之后长度归一化与 coverage penalty 的收益明显变小（Table 3）。同一组修正把纯 ML 模型在 WMT'14 En→Fr 开发集上的 BLEU 从 30.3 提到 31.4；上面这对 5 步与 20 步的序列在 $\alpha=0.6$ 下从相差 3.35 nats 缩到 1.07 nats，偏置被压低但没有消失。
- 短视：beam 只在每一步保留 $B$ 条前缀最优的假设，无法保证全局最优。构造一个两步玩具分布：第一步 A 0.6 / B 0.4，A 之后最好的续写是 X（条件概率 0.5），B 之后是 Z（0.9）。greedy 走 A→X 得到 $0.6\times0.5=0.30$；$B=2$ 的 beam 同时留着 B→Z $=0.4\times0.9=0.36$，于是找到更高的序列分。反过来说，beam 找到的是「模型打分的最高序列」，不是「最好的文本」——两个目标在开放式生成里并不一致。
- 成本与收益不成比例：每步要扩展 $B$ 条候选，算力与 KV cache 显存都按 $B$ 放大（每条 beam 各持一份 cache，即 [[llm-internals-02]] 公式里的 $b$ 项），而 GNMT 的经验是常用 8–12 条、退到 4 条甚至 2 条对 BLEU 只有轻微损失。更糟的是搜索空间放大本身有害：Koehn & Knowles（2017）的结论是 beam search 只在窄 beam 下提升翻译质量，暴露在更大的搜索空间里反而变差。这就是它在开放式生成里普遍让位给采样的原因。

**temperature：两端都坏。** 偏高时尾部 token 拿到真实概率，输出开始语义漂移、出现幻觉，而自回归会把一次错误选择当成后续所有步的前提——源文的说法是一个坏 token 就能毁掉整段回答，且越写越离谱。偏低时分布过于尖锐，退化成 greedy 的重复与套话，像 "I think that I think that" 一样打转，多样性趋近于零。需要强调的是 $T$ 不改变排序，所以如果正确答案压根不在分布前列，调温度只会让它以更低概率被抽到，不会让它变成首位。

**top-k：$k$ 不自适应。** 上面两种反例（尖锐时太宽、平坦时太窄）来自同一个根因：$k$ 是常数，而每一步的分布形状都在变。

**top-p：累积阈值与单个 token 的质量脱钩。** top-p 只关心「凑够 $p$ 的概率质量」，不关心被凑进来的 token 有多差，因此在分布平坦或温度偏高时会系统性吞下长尾。给一个可复算的例子：8 个颜色合计 80%（最高 12%），长尾是 200 个各占 0.1% 的 token；`top_p=0.9` 必须先收下 8 个颜色，再从长尾里继续收 100 个才越过 0.9，候选集共 108 个，其中 100 个是单个概率 0.001 的垃圾。同样这条分布上 `min_p=0.1` 的阈值是 $0.1\times0.12=0.012$，长尾全部被切掉，只留 8 个。「top-p 在尖锐分布下也会截进低质量 token」这句话要拆开看：若 $p_{\max}$ 已经不小于 $p$，top-p 退化成只留 1 个，行为等同 greedy，并没有多截；真正会出问题的是 $p_{\max}$ 略小于 $p$ 的情况——比如 $p_{\max}=0.85$、$p=0.9$，还差 0.05 必须从尾部补，如果尾部有 50 个各 0.002 的 token，就得再收 25 个，候选集 26 个里有 25 个的单个概率只有 $2\times10^{-3}$。min-p 用相对阈值替代绝对累积阈值，正是为了消除这种「为了凑概率而凑数」的行为；min-p 论文的动机也明确写成「top-p 在高温度下难以兼顾质量与多样性」。

**所有采样方法的共同边界。** 解码策略只能改变从分布里取样的方式，不能把缺失的知识变出来。任何温度、任何 top-p，都救不了一个在第一步就没把正确答案放进候选的模型。

### 7. 场景化建议

| 场景 | 建议配置 | 理由 |
| --- | --- | --- |
| 代码、数学、结构化抽取 | greedy 或 $T\le0.2$ + 小 top-p | 只有一个正确答案，不确定性只带来错误 |
| 工具调用 / JSON 输出 | $T=0$（或 0.1 以内），宽 top-p 或不截断 | 格式稳定性优先，采样只会增加解析失败率 |
| 摘要、翻译 | $T=0.2$–$0.5$ | 要贴着原文，但允许措辞变化 |
| 通用对话 | $T=0.7$–$1.0$ | 自然度与稳定性的平衡点 |
| 创意写作、头脑风暴 | $T=0.8$–$1.2$ + top-p 0.95 或 min-p | 需要多样性；超过 1.5 左右很快变成胡话 |
| 需要多候选后筛选（rerank、自洽性、单测过滤） | $T>0$ + top-p，采 $n$ 条 | greedy 只有一条路径，筛选无从谈起 |

两个工程约束：一是**评测必须固定解码参数**。pass@$k$ 需要采样 $n$ 个候选再估计无偏指标，$T=0$ 时 $k$ 个候选完全相同，pass@$k$ 直接退化成 pass@1；反过来，同一个模型在 $T=0.2$ 与 $T=0.8$ 下的分数不可比，报指标时必须写明参数。二是**参数要成组管理**，因为先温后截意味着它们互相耦合：固定 top-p 调温度、或固定温度调 top-p，都比同时调两个更容易复现。

## 数值与代码验证

**表 1：logits [4, 3, 1, 0] 在不同温度下的分布（复算值，4 位小数）**

| $T$ | blue | clear | falling | banana | 熵 $H$ | 有效候选数 $e^{H}$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 1.00 |
| 0.5 | 0.8786 | 0.1189 | 0.0022 | 0.0003 | 0.383 | 1.47 |
| 0.7 | 0.7957 | 0.1907 | 0.0110 | 0.0026 | 0.563 | 1.76 |
| 1.0 | 0.6964 | 0.2562 | 0.0347 | 0.0128 | 0.773 | 2.17 |
| 2.0 | 0.5089 | 0.3087 | 0.1136 | 0.0689 | 1.138 | 3.12 |

源文同一组 logits 给的是两位小数：$T=0.5$ 为 0.88 / 0.12 / 0.00 / 0.00，$T=1$ 为 0.70 / 0.26 / 0.03 / 0.01，$T=2$ 为 0.51 / 0.31 / 0.11 / 0.07，与上表四舍五入后完全一致，源文 $T=0.1$ 行的 1.00 / 0.00 / 0.00 / 0.00 也一致。有效候选数这一列是额外算的：即使 $T=2$，实际参与竞争的也只有约 3.1 个 token——「高温让所有 token 都有机会」在 4 个 token 的玩具上是错觉，在 10 万词表上才接近事实。

**表 2：验证「先温度后截断」的耦合（top_p = 0.9）**

| 温度 | 候选数 | 被截断的 token | 采样分布 |
| --- | --- | --- | --- |
| 0.5 | 3 | bed、roof、其余桶 | 65.3% / 25.5% / 9.2% |
| 1.0 | 4 | roof、其余桶 | 44.4% / 27.8% / 16.7% / 11.1% |
| 2.0 | 5 | 其余桶 | 30.7% / 24.3% / 18.8% / 15.4% / 10.9% |

表 2 的 $T=1$ 一行与源文一致：源文用 mat 40% / floor 25% / sofa 15% / bed 10% / roof 5% 说明 top-p=0.9 在第 4 项越过阈值，重归一化后 40/90 = 44.4%、25/90 = 27.8%、15/90 = 16.7%、10/90 = 11.1%。**口径说明**：源文这 5 行合计只有 95%，因为它把剩下的 5% 写成「其余 token 合计」；要让温度可重新缩放，分布必须凑满 1.0，所以这里把「其余」当成第 6 个桶（0.05），$T=1$ 时能精确还原源文的 5 行数字。

**表 3：平坦分布下 top-p 与 min-p 的候选集（8 个颜色 + 200 个各 0.1% 的长尾 token）**

| 方法 | 候选数 | 入选内容 |
| --- | --- | --- |
| top_p = 0.9 | 108 | 8 个颜色 + 100 个概率 0.001 的长尾 token |
| min_p = 0.1（阈值 0.012） | 8 | 只有 8 个颜色 |

这里的 108 与 8 来自同一份构造分布，可以手算复核：8 个颜色合计 0.80，还差 0.10 才到 0.9，长尾每个 0.001，需要 100 个。**口径说明**：长尾的形状源文没有给，0.001 × 200 是为了让「凑够 0.9」这一步可计算而设的假设；真实词表的尾部更接近幂律，凑数需要的 token 只会更多。

```python
import math

def softmax(logits, T=1.0):
    scaled = [z / T for z in logits]          # 温度只改锐度，不改排序
    m = max(scaled)                           # 减最大值仅为数值稳定
    total = sum(math.exp(z - m) for z in scaled)
    return [math.exp(z - m) / total for z in scaled]


def nucleus(probs, p):
    acc, kept = 0.0, 0
    for x in probs:
        if acc >= p - 1e-12:                  # 容差抵消浮点累加误差，否则会多收一个 token
            break
        acc, kept = acc + x, kept + 1
    return kept, [x / acc for x in probs[:kept]]      # (候选数, 重归一化后的概率)


# 表 1：不同温度下的分布与有效候选数 exp(H)
for T in (0.1, 0.5, 0.7, 1.0, 2.0):
    q = softmax([4.0, 3.0, 1.0, 0.0], T)
    H = -sum(x * math.log(x) for x in q if x > 0)
    print(f"T={T}: " + " ".join(f"{x:.4f}" for x in q) + f"  有效候选={math.exp(H):.2f}")

# 表 2：源文 "The cat sat on the"，第 6 桶是「其余 token 合计 5%」
base = [math.log(x) for x in (0.40, 0.25, 0.15, 0.10, 0.05, 0.05)]    # 由概率反推 logits
for T in (0.5, 1.0, 2.0):
    kept, renorm = nucleus(softmax(base, T), 0.9)
    print(f"T={T}: top_p=0.9 保留 {kept} 个 -> " + " ".join(f"{x * 100:.1f}%" for x in renorm))

# 表 3：平坦分布下 top-p 会凑进长尾，min-p 不会
full = [0.12, 0.11, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08] + [0.001] * 200
print("top_p=0.9 候选数", nucleus(full, 0.9)[0], "| min_p=0.1 候选数", sum(x >= 0.1 * max(full) for x in full))

# beam 短视与长度偏置：每步平均条件概率 0.8，长度归一化用 GNMT 公式、α=0.6
after_A, after_B = {"X": 0.5, "Y": 0.3, "W": 0.2}, {"Z": 0.9, "W": 0.1}
print("greedy A->X", 0.6 * after_A["X"], "| beam=2 最优 B->Z", 0.4 * after_B["Z"])
for L in (5, 20):
    raw, lp = math.log(0.8 ** L), (5 + L) ** 0.6 / 6 ** 0.6
    print(f"L={L}: logP={raw:.4f} 归一化后={raw / lp:.4f}")
```

## 常见追问

- **追问**：`temperature=0` 的输出是完全可复现的吗？
  - 要点：不是。$T=0$ 在实现上是跳过采样直接取 argmax，语义确定，但浮点归约顺序、kernel 选择、batch 内其它请求、MoE 路由等都可能带来微小数值差异，源文的措辞是「几乎总是相同」，不是「保证相同」。要严格复现就缓存输出或固定 kernel 与 batch 组成。
- **追问**：为什么 beam search 在机器翻译上曾经是标配，在开放式生成里却打不过采样？
  - 要点：翻译的参考译文接近「最可能的序列」，似然是可用的代理指标，且长度归一化与 coverage penalty 能修掉长度偏置和漏译；开放式生成里「最可能的序列」往往是平淡、重复、套话化的文本（Holtzman 等，2019），而且搜索空间放大后质量还会下降（Koehn & Knowles，2017）。加上成本按 $B$ 倍增长，收益为负。
- **追问**：把 top_p 设成 1.0、top_k 设成 0 分别意味着什么？
  - 要点：都是关闭对应的过滤器，回到纯采样。这条连续谱的两个端点很清楚：$k=1$ 或 $p\to 0$ 是 greedy，$k=|V|$ 或 $p=1$ 是纯采样，top-k 与 top-p 只是取中间某个位置。
- **追问**：为什么评测报告必须写解码参数？
  - 要点：pass@$k$ 依赖采样，$T=0$ 时 $k$ 个候选完全相同，指标退化为 pass@1；高温能刷高 pass@$k$（多样性换正确率）却拖低 pass@1（格式与稳定性）。同一个模型在不同 $T$、top-p 下的分数不可比，所以论文与榜单都要求把 temperature、top-p、$n$、max_tokens 一起写清楚。

## 公司变体

`asked_at` 三家在这道题上的侧重不同：

- **Google DeepMind**：偏数学推导与搜索本身。这题的经典材料一半出自 Google 自家的 GNMT（长度归一化与 coverage penalty，Wu et al., 2016），一半出自 NMT 社区对 beam search 的系统性反思（Koehn & Knowles, 2017，Johns Hopkins）。DeepMind 公开的代码生成系统 AlphaCode 走的是另一条路——大规模采样后用程序行为过滤到少量提交——正好是「为什么不用 beam search」的现成对照。追问通常落在「长度归一化改的是什么」「beam 为什么短视」。
- **Apple**：偏工程约束。端侧与私有云计算场景下，解码要在固定内存与延迟预算内完成，于是更关心 on-device 上 top-k / top-p 的排序与累积开销、greedy 在不同设备上的一致性，以及为什么有限算力下通常不上 beam search。产品侧常见问法是「怎样让同一个 prompt 的输出稳定」。
- **Perplexity**：偏工程与产品形态。搜索与 RAG 场景要求答案 grounded 到检索结果并带引用，解码侧倾向低温配保守截断来压制幻觉与格式抖动；API 又把 temperature / top_p 这类参数直接交给调用方，所以需要能解释参数与事实性、引用一致性、延迟之间的关系，而不是文学性。

以上是依据各家公开技术材料与产品方向的侧重判断，不代表具体的面试轮次或固定题面。

## 相关题目

- [[llm-internals-16]]：decoder-only transformer 一次前向传播的逐张量过程，说明 logits 从哪里来、为什么是 $|V|$ 维。
- [[llm-internals-11]]：pre-training、SFT 与 preference optimisation，解释「正确答案在不在分布里」是解码策略无法弥补的部分。
- [[llm-internals-02]]：KV cache 的显存公式，其中的 $b$ 项就是 beam search 与多候选采样的成本来源。

## 参考资料与归属

1. [How does Temperature control LLM output?](https://outcomeschool.com/blog/how-does-temperature-control-llm-output)，Amit Shekhar（Outcome School），2026-09-15。提供 temperature 除以 logits 的机制、$T=1/0.5/0.1/2$ 的四 token 数值表与两位小数结果、低温与高温的输出特征、$T=0$ 即 greedy、$T=0$ 也「几乎总是相同」而非严格可复现，以及「不要把 temperature 与 top-p 一起调」的工程建议；表 1 与该表四舍五入后一致。
2. [How do Top-k and Top-p Sampling work?](https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work)，Amit Shekhar（Outcome School），2026-09-19。提供 greedy 的重复退化与纯采样的长尾问题、top-k 的固定 $k$ 在「Paris 95%」与「喜欢的颜色」两种分布下的反例、top-p 的逐步累积与重归一化示例、两者的对照表、以及「温度改锐度、top-k/top-p 只截断」的定位。
3. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)（延伸），Ari Holtzman 等，2019-04-22。第 6 节 greedy 退化的判断（以似然为解码目标导致平淡且重复的文本）、以及「同一模型换解码策略质量差异巨大」的结论取自该文；nucleus sampling 即 top-p 也是该文提出。
4. [Six Challenges for Neural Machine Translation](https://arxiv.org/abs/1706.03872)（延伸），Philipp Koehn、Rebecca Knowles，2017-06-12。第 6 节「beam search 只在窄 beam 下提升质量、搜索空间放大后变差」这条结论取自该文摘要与正文的总结列表。
5. [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144)（延伸），Yonghui Wu 等（Google），2016-09-26。第 6 节的长度偏置表述、长度归一化公式 $lp(Y)=(5+|Y|)^{\alpha}/(5+1)^{\alpha}$ 与打分函数 $s(Y,X)$、coverage penalty、$\alpha\in[0.6,0.7]$（简化启发式）与打分函数所用 $\alpha=\beta=0.2$ 的口径差异、RL 精调后这两项收益变小（Table 3）、BLEU 30.3→31.4、以及常用 8–12 条 beam 的经验都取自该文第 7 节；第 6 节里 5 步与 20 步序列的归一化数值是用该公式自行复算的。
6. [Turning Up the Heat：Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082)（延伸），Minh Nhat Nguyen 等，2024-07-01（ICLR 2025）。第 4、6 节的 min-p 定义（以最高 token 概率为缩放系数的动态阈值）、「top-p 在高温度下难以兼顾质量与多样性」的动机、以及它已被 transformers、vLLM 等框架采纳的事实取自该文。
7. [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)（延伸），Yujia Li 等（DeepMind），2022-02-08。第 7 节「大规模采样 + 按程序行为过滤」这条替代 beam search 的路线取自该文摘要中列出的三个关键组件。
8. [transformers 的采样实现（generation/utils.py）](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)（延伸），Hugging Face，无发布日期（按 2026-09-28 的 main 分支核对）。第 5 节的执行顺序（temperature → top-k → top-p → min-p → softmax → multinomial）、「采样参数只在 `do_sample=True` 时生效」、`min_tokens_to_keep` 下限、以及 `configuration_utils.py` 的默认值（字段默认留 `None`，生成时由 `GenerationConfig._get_default_generation_params()` 填成 `temperature=1.0`、`top_k=50`、`top_p=1.0`、`do_sample=False`、`repetition_penalty=1.0`）都来自该实现；min-p 的典型取值区间 0.01–0.2 与其对应 top-p 0.99–0.8 的说明来自同目录的 `configuration_utils.py` 文档字符串。
9. [CTRL：A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858)（延伸），Nitish Shirish Keskar 等，2019-09-11。第 5 节 repetition penalty 的出处与论文原式（对已出现的 token 取 $p_i\propto\exp(x_i/(T\theta))$、greedy 下 $\theta\approx1.2$）来自该文；按符号处理（正 logit 除以 $\theta$、负 logit 乘以 $\theta$）的变体来自 transformers 的实现。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
