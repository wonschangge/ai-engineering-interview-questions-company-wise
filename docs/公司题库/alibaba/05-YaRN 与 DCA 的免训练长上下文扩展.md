---
type: question
id: alibaba-05
company: 阿里巴巴（Qwen）
topic: llm-internals
order: 5
question: Qwen2.5 借助 YaRN 加 Dual Chunk Attention 把上下文扩展到 128K（Turbo 约 1M），且基本无需训练。请解释其原理，以及为什么事后扩展如此有吸引力。
question_en: Qwen2.5 extends context to 128K (and ~1M for Turbo) using YaRN plus Dual Chunk Attention, mostly training-free. Explain how, and why post-hoc extension is attractive.
asked_at: []
level: 高阶
tags: [rope, yarn, DCA, 长上下文外推, 免训练]
sources:
  - title: RoPE（旋转位置编码）背后的数学
    url: https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding
    author: 
    published: 
  - title: RoFormer: Enhanced Transformer with Rotary Position Embedding（延伸）
    url: https://arxiv.org/abs/2104.09864
    author: Jianlin Su et al.
    published: 2021-04-20
  - title: YaRN: Efficient Context Window Extension of Large Language Models（延伸）
    url: https://arxiv.org/abs/2309.00071
    author: Bowen Peng et al.
    published: 2023-08-31
  - title: RULER: What's the Real Context Size of Your Long-Context Language Models?（延伸）
    url: https://arxiv.org/abs/2404.06654
    author: Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg
    published: 2024-04-09
related: [llm-internals-08, llm-internals-13, llm-internals-02, alibaba-03]
updated: 2026-09-28
---

## 一句话答案

> Qwen2.5 的扩展是两件事拼起来：**static YaRN 只换 $\cos/\sin$ 表**（按波长分三区插值，再叠一个只依赖长度比的 attention 温度），**DCA 只换 query 侧的位置集合**（三段式重写相对位置矩阵），权重和 KV cache 的数据结构都不动，所以「基本无需训练」成立。
> DCA 的硬不变式是：任何 query–key 的相对距离都被压回 $[0,\,c-1]$（$c$ 为预训练窗口）。Qwen 官方把长上下文退化的主因归到「query 与 key 之间出现训练中未见过的大相对位置距离」，这条被直接消掉；代价是远距离只剩粗粒度分辨率。
> 事后扩展的吸引力在成本与风险轮廓：训练步数为 0（DCA 是一段推理期 monkey patch），与「长数据从哪来」解耦，训练长度内的行为完全不变（官方原话），因此可以按长度开关、随时回退。
> 但「免训练」不等于免费：KV 显存（7B 每 token 56 KiB、128K 单序列 7 GiB）与 prefill 的 $O(S^2)$ 算力一分不省，有效上下文仍要靠 RULER 这类多针评测另算。

## 面试官在考什么

- 能否把 RoPE 的三条前提说全：旋转矩阵正交 ⇒ $\langle R_mq,\,R_nk\rangle=\langle q,\,R_{n-m}k\rangle$，打分只依赖 $n-m$；只作用于 Q/K、不作用于 V；没有可学参数，$\cos/\sin$ 表按位置预计算一次、所有前向复用——第三条正是后面所有扩展方法「零额外开销」的来源。
- 是否知道外推崩溃的物理图像是**波长与训练窗口的关系**（$r(i)=L/\lambda_i<1$ 的那批平面在训练中充当绝对位置编码），而不是笼统的「位置没训过」。
- 能不能把 DCA 的三段式写成公式并给出边界：Intra 的距离落在 $[0,s-1]$、Succ 落在 $[1,c-1]$、Inter 落在 $[c-s,c-1]$，三者合起来把全部相对距离压进 $[0,c-1]$。
- 是否分得清 YaRN 的两半：NTK-by-parts 频率分区在论文里是需要微调的，attention 温度 $\sqrt{1/t}=0.1\ln s+1$ 才是纯推理期的那一半，也正是 Qwen2.5-1M 报告明确说自己一直在用的那一半。
- 能不能把「免训练」讲成有边界的结论：省掉的是训练，不是推理账单；并用 RULER 的 128K 档对照说清扩窗与训长各自值多少分。

**常见错误答案**

- 把 DCA 当成稀疏注意力或滑窗（MInference、StreamingLLM 那一类），或说成「加一套位置编码」「把 base 调大」。DCA 不改变注意力的稀疏性，也不动位置嵌入本身，只重写相对位置矩阵；论文明确避免线性缩小位置索引与放大 base 频率这两种做法。它与稀疏注意力正交，而且在 Qwen2.5-1M 里直接组合还会互相干扰。
- 把 YaRN 说成完全训练无关，或把「扩到 128K/1M」等同于「能用好 128K/1M」。前者漏掉频率分区的微调口径，后者被 RULER 的多针、多跳追踪与 lost-in-the-middle 直接反驳。

## 原理与推导

### 1. RoPE 的三条前提

把单头维度 $d$ 按相邻两维切成 $d/2$ 个二维平面，第 $i$ 个平面的角频率 $\theta_i=\text{base}^{-2i/d}$，位置 $m$ 的 Query 在平面上旋转 $m\theta_i$：

$$ \theta_i=\text{base}^{-\frac{2i}{d}},\qquad
\langle R_mq,\,R_nk\rangle=q^\top R_m^\top R_nk=\langle q,\,R_{n-m}k\rangle $$

- **相对位置性质来自正交性**：$R_m^\top=R_{-m}$、$R_{-m}R_n=R_{n-m}$，两个绝对位置在打分里被消掉。注意「RoPE 编码绝对位置」这句话同样成立，只是**打分**对绝对位置不敏感。
- **只作用于 Q/K**：位置信息来自 $q^\top k$，V 只被 softmax 权重聚合，旋转它没有收益。
- **没有可学参数**：$\cos/\sin$ 表按位置预计算、所有前向复用。任何「只改位置编码、不改权重」的方案都因此不增加前向开销。
- base 的取值是配置项：经典实现取 $10^4$；Qwen2.5-7B/14B-Instruct 的 config 是 `rope_theta=1e6`，1M 版（7B/14B-Instruct-1M）是 `rope_theta=1e7`，两档的 `rope_scaling` 都是 `None`。这个取值是训练期定的（报告里的 ABF，Adaptive Base Frequency），不是推理期可以随手调的旋钮。

### 2. 外推崩溃的物理图像：波长与圈数

第 $i$ 个平面走完一圈需要的 token 数是 $\lambda_i=2\pi\,\text{base}^{2i/d}$；把每个平面在训练窗口 $L$ 内转过的圈数记作

$$ r(i)=\frac{L}{\lambda_i}=\frac{L}{2\pi\,\text{base}^{\frac{2i}{d}}} $$

- $r(i)\gg1$ 的平面转了很多圈，模型只能看到相对相位差，是纯相对位置编码。
- $r(i)<1$ 的平面（波长比 $L$ 还长）训练全过程连一圈都没转完，$m$ 与相位一一对应，**实际充当绝对位置编码**（YaRN 论文的观察）。直接取 $m>L$，就是把相位推到从未见过的取值上。

$d=128,\ \text{base}=10^4,\ L=4096$（仓库统一口径）下：$r(i)=\beta=32$ 的临界在 $i^*=20.94$，$r(i)=1$ 的临界在 $i^*=45.03$。取整后 $i\le20$ 不插值、$i=21\dots45$ 走 ramp、$i\ge46$ 的 18 个平面全部插值——这 18 个平面占 64 个平面的 28.1%，就是「越界」的那一批。

把 base 从 $10^4$ 提到 $10^6$（Qwen2.5-128K 档）并不能消掉这批平面：按 $\lambda_i>L$ 复算，$L=32768$ 时仍有 24 个平面（37.5%，$i\ge40$）的波长超过训练窗口；1M 档（base $=10^7$、训练到 256K）是 21 个（32.8%，$i\ge43$）。**调大 base 只是把这批平面挪到别的位置，外推仍然要在推理期解决。**

### 3. DCA 的三段式：一套公式，三条边界

记号：chunk size $s$、预训练长度 $c$、局部窗口 $w=c-s$；序列长度 $l$ 按 $s$ 切成 $n=l/s$ 个 chunk。论文给的取法是 $s$ 取 $\tfrac34 c$，Llama2（$c=4096$）对应 $s=3072,\ w=1024$。DCA 的关键设计是**键侧位置不动**：

$$ P_k=[0,1,\dots,l-1]\bmod s $$

也就是 $[0,1,\dots,s-1]$ 在每个 chunk 内循环，最大值恒为 $s-1$；这样键的旋转角与它在哪个 chunk 无关，KV cache 可以按原样复用。改动全在 query 侧，按 chunk 差分三段：

$$ P_q^{\text{Intra}}=P_k,\qquad
P_q^{\text{Succ}}=\bigl[\underbrace{s,s+1,\dots,s+w-1}_{w},\ \underbrace{c-1,\dots,c-1}_{\text{所有 chunk 相同}}\bigr],\qquad
P_q^{\text{Inter}}=[\underbrace{c-1,\dots,c-1}_{l}] $$

相对位置矩阵按 chunk 差选位置：$M[i][j]=P_q[i]-P_k[j]$，chunk 差为 $0$ 用 Intra、为 $1$ 用 Succ、大于 $1$ 用 Inter。三条边界可以逐条推出来：

- **Intra（同 chunk，$0\le j\le i<s$）**：距离落在 $[0,s-1]$，与预训练分布完全一致，局部次序完整保留。
- **Succ（相邻 chunk）**：$P_q^{\text{Succ}}[i]=s+t$（$t=i\bmod s$，仅当 $t<w$），对前一个 chunk 里局部下标为 $j$ 的键，距离是 $s+t-j$——**恰好等于原始绝对距离**，所以局部窗口内的短程信息无损；最近的前驱键距离下界是 1（当前 query 是 chunk 首 token 时恰为 1），$w=c-s$ 个最近邻的最大距离是 $t+w\le c-1$。$t\ge w$ 时位置压成常数 $c-1$，退化成 Inter 的处理。整段距离落在 $[1,c-1]$。
- **Inter（跨多个 chunk）**：距离恒为 $c-1-P_k[j]\ge c-s$，全部落在 $[c-s,c-1]$，是训练见过的区间。

**不变式一句话收口：任何 query–key 的相对距离都被压回 $[0,c-1]$，位置编码不需要任何微调。** 代价是用位置分辨率换外推能力——跨 chunk 的距离只由「键在自己 chunk 内的局部下标」决定，与它属于哪个 chunk 无关；所有非相邻 chunk 的 query 位置又都是同一个常数 $c-1$，所以远距离上既区分不出键来自几个 chunk 之前，也区分不出 query 自己的远近。

### 4. 为什么是三段而不是两段：论文那个 12 token 的算例

论文用 $c=10,\ s=6,\ w=4,\ l=12$ 走了一遍（$12$ 个 token 超过预训练窗口 $10$，切成 2 个 chunk）：

- $P_k=[0,1,2,3,4,5,0,1,2,3,4,5]$，$P_q^{\text{Inter}}=[9,\dots,9]$（12 个），$P_q^{\text{Succ}}=[6,7,8,9,9,9,6,7,8,9,9,9]$。
- 对相邻的 $q_6$ 与 $k_5$：用 Intra 会得到 $P_q^{\text{Intra}}[6]-P_k[5]=0-5=-5$（负距离，违反 $i\ge j$ 时 $P_q[i]\ge P_k[j]$ 的因果前提）；用 Inter 会得到 $9-5=4$——两个明明相邻的 token 被说成相距 4，论文原话是这会破坏模型维持 locality 的能力；只有 Succ 得到 $6-5=1$。
- 论文正文另举了一对 $q_s=q_6$ 与 $k_1$：Intra 给 $0-1=-1$（这就是论文里那个负距离的标准例子），Succ 给 $6-1=5$，恰好等于它们的真实距离，而 Inter 给 $9-1=8$。

三个数字说明三段的**分工互不替代**：Intra 保同 chunk 的分辨率，Inter 给出跨 chunk 的粗粒度可达性，Succ 专门修复相邻 chunk 的局部性。缺掉 Succ，最近邻的距离会被夸大（1 变成 4）；只用 Intra，跨 chunk 的相对距离会变成负数。论文的消融实验（Figure 4）也是按这三段分别拆开验证的。

### 5. 只改 query 侧：为什么能直接复用 KV cache 与 FlashAttention

- 键侧 $P_k$ 完全不变，所以 KV cache 的布局、分页（PagedAttention）、prefix cache 全部可以照用；DCA 不改 KV 的字节数，也不改注意力的稀疏性。
- 位置只进旋转表，因此三段式只是一次「按 chunk 差选 $\cos/\sin$ 表」的分支，可以并进 FlashAttention 的分块计算。论文明确说 DCA 可与 FlashAttention 2 无缝结合，ChunkLlama 的实现只是 monkey patch 掉 `LlamaAttention` 的推理代码；这也是它能在生产环境落地、而重新训练一份权重做不到的关键差别。
- 与 PI／NTK 的关系是正交而非替代：DCA 论文明确写的是——**避免线性缩小位置索引或放大 base 频率**，改为复用预训练模型的位置嵌入、重新设计相对位置矩阵的构造。已支持 32k 的模型再叠 DCA 可以到 192k，passkey 准确率与 PPL 都保持得住。

### 6. YaRN 拆成两半：频率分区要微调，attention 温度不要

**第一半是频率分区（NTK-by-parts）。** 按 $r(i)$ 分三区并用 ramp 平滑过渡（YaRN 推荐 $\alpha=1,\ \beta=32$）：

$$ \gamma(r)=\begin{cases}0 & r<\alpha\\ 1 & r>\beta\\ \dfrac{r-\alpha}{\beta-\alpha} & \text{otherwise}\end{cases}\qquad
h(\theta_i)=\bigl(1-\gamma(r)\bigr)\frac{\theta_i}{s}+\gamma(r)\,\theta_i $$

$d=128$、base $=10^4$、$L=4096$ 时，64 个平面分成不插值 21 个（32.8%）、ramp 25 个（39.1%）、全插值 18 个（28.1%）。被保护的高频平面接近三分之一，而 PI 一刀切压缩全部 64 个平面——这就是 PI 明显损伤局部信息的原因。**这一半在论文里是需要微调的**：7B 从 2k 扩到 32k 用 400 步、全局 batch 64、64k 数据，约 128 A100-h；作为对照，PI 的 2k×8 → 16k 约 640 A100-h。摘要口径是比此前方法少 10 倍 token、少 2.5 倍训练步数、微调数据不到预训练数据的约 0.1%。

**第二半是 attention 温度，纯推理期。** 插值把所有相位差压小，打分差异变小、softmax 分布变平、熵升高；温度把 logits 整体放大以抵消熵增：

$$ \mathrm{softmax}\!\left(\frac{q^\top k}{t\sqrt{d}}\right),\qquad \sqrt{1/t}=0.1\ln s+1 $$

$s$ 是推理长度与训练长度之比：$s=4\to1.139$、$8\to1.208$、$16\to1.277$、$32\to1.347$。实现上把它并进 $\cos/\sin$ 表即可，零额外开销、不改 attention 代码。Qwen2.5-1M 报告用的正是这一半，原话是**总是把 YaRN 的 attention scaling 与 DCA 一起用**，并声明**这两种外推方法都不改变模型处理短序列时的行为**——这句「不改变训练长度内行为」是事后扩展之所以「几乎免费」的核心论据。

### 7. Qwen 的两档配置：128K 走静态 YaRN，1M 走推理框架

- **128K 档（Qwen2.5-7B/14B-Instruct）**：config 是 `max_position_embeddings=32768`、`rope_theta=1e6`、`rope_scaling=None`；官方给出的 128K 路径是在 config 里叠加**静态** YaRN，即写入 `rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}`。$32768\times4=131072$，模型卡口径是「完整 131,072 上下文、生成 8192 token」（7B 共 28 层、28 个 Q 头、4 个 KV 头，14B 是 48 层、8 个 KV 头）。模型卡同时提醒：vLLM 目前只支持 static YARN，缩放因子恒为 4、与输入长度无关，**会损伤短文本**，建议只在确实需要长上下文时才加这段配置。这条不只是坑，也是 static 与 dynamic 取舍的官方依据。
- **1M 档（Qwen2.5-7B/14B-Instruct-1M、API 侧的 Qwen2.5-Turbo）**：config 是 `max_position_embeddings=1010000`、`rope_theta=1e7`、`rope_scaling` 仍然是 `None`。**外推（DCA）在推理框架里，不在权重里**；而 1010000 只是声明值，训练只到 256K——这是本题最容易被误读的口径。
- 一句话对比：static YaRN 是长度无关的（短序列也被整体压缩，所以有短文本代价），DCA 是分段相关、训练窗口内的位置完全不变；两者互补——YaRN 温度治「注意力被摊平」，DCA 治「相对位置越界」。而「免训练」这个标签只对 attention 温度那一半与 DCA 成立。

### 8. 事后扩展为什么有吸引力，以及它的边界

**为什么吸引人：**

- **成本量级**：DCA 是 0 训练步，论文口径就是一段替换 `LlamaAttention` 推理代码的 monkey patch。对照量级：PI 的 2k×8 → 16k 约 640 A100-h，YaRN 的 7B 2k → 32k 约 128 A100-h／400 步，DCA 论文为做对照跑的长对话微调是 7B 约 40 GPU-h、13B 约 60 GPU-h（16k 步、batch 1、16k 输入）。几十到几百 A100-h 与「重做一遍长上下文持续预训练」差着数量级。
- **数据侧解耦**：长文档语料稀缺且贵。事后扩展把「修位置编码」与「投长数据」拆开——先保证位置不越界，再把预算花在长数据上；位置修复可以立即上线，长数据投入可以慢慢来。
- **风险轮廓好**：DCA 在训练窗口内不改位置（chunk 差为 0 的 Intra 段与原始 RoPE 逐项一致），YaRN 温度只依赖 $s$，所以短文本质量基本不动；可以按请求长度或开关启用，改动小、可回退、能按流量分级。反过来 static YaRN 会伤短文本、必须按需开启。一个旋钮比一个新 checkpoint 好运营得多。

**边界在哪里（必须一起给）：** 事后扩展只买到「位置不越界」，不买到长上下文能力。Qwen2.5-1M 报告的 RULER 表给出可复算的对照（口径是 RULER 在 128K 长度上的分数）：7B 从 31.4 提到 55.1，14B 从 53.0 提到 78.1，32B 从 57.7 提到 82.0，72B 从 67.0 提到 88.4；而真正在长数据上训过的 Qwen2.5-7B-Instruct-1M 是平均 91.8、128K 档 84.4，14B-Instruct-1M 平均 95.7。同一批数字说明两件事：**扩展能救回一大截，但救不回「没训过」的那部分。**

由此得到选型结论：不是「扩窗」与「训长」二选一，而是两者相乘——报告里 72B 加 DCA+YaRN 在 LV-Eval 上全线胜过 14B-Instruct-1M，说明基座规模与扩展手段要一起看。「无需训练」的准确边界是「不需要为位置编码本身训练」：Qwen2.5-1M 的真实路线是先做渐进式长上下文训练到 256K，再用 DCA 外推到 1M；只在 32K 上训过的 7B/14B 加 DCA 就能在 1M passkey 上做到 80% 以上，但在多针、多值 NIAH 上差距会显出来（报告的结论是训到 256K 显著提升了继续外推的能力）。评测口径也要一起给：用 RULER 的多针、多跳追踪与聚合，不要拿单针 NIAH 当结论，否则「事后扩展」会被高估。

## 数值与代码验证

以下数字全部按上面的公式与仓库统一常数复算（脚本见 `.work/lane-b/alibaba-05-recompute.py`）。KV 单价 $k=2\,L_{\text{layer}}H_{kv}d_{\text{head}}\cdot\text{bytes}$，bf16 即每元素 2 B，$d_{\text{head}}=128$。

### 表 1：RoPE 平面分区与临界值（$d=128$，64 个平面，$\alpha=1,\ \beta=32$）

| 判据 | 复算结果 | 含义 |
| --- | --- | --- |
| $r(i)=\beta=32$，$L=4096$、base $=10^4$ | $i^*=20.94$ | $i\le20$ 不插值（21 个，32.8%）|
| $r(i)=1$（$\lambda_i=L$），同口径 | $i^*=45.03$ | $i\ge46$ 的 18 个平面波长超过 $L$（28.1%）|
| ramp 过渡 | $i=21\dots45$ | 25 个（39.1%）|
| $\lambda_i>L$ 的平面数，base $=10^6$、$L=32768$ | 24 个（37.5%，$i\ge40$）| 128K 档调大 base 后仍有这批平面 |
| $\lambda_i>L$ 的平面数，base $=10^7$、$L=262144$ | 21 个（32.8%，$i\ge43$）| 1M 档同理 |

### 表 2：DCA 三段位置集合与论文算例（$c=10,\ s=6,\ w=4,\ l=12$）

| 项 | 数值 |
| --- | --- |
| $P_k$ | $[0,1,2,3,4,5,0,1,2,3,4,5]$ |
| $P_q^{\text{Intra}}$ | $=P_k$，距离 $\in[0,5]$ |
| $P_q^{\text{Succ}}$ | $[6,7,8,9,9,9,6,7,8,9,9,9]$，距离 $\in[1,9]$ |
| $P_q^{\text{Inter}}$ | $[9,9,9,9,9,9,9,9,9,9,9,9]$，距离 $\in[4,9]$ |
| 相邻对 $(q_6,k_5)$ | Intra $-5$ / Succ $+1$ / Inter $+4$（真值 1）|
| 跨 chunk 对 $(q_6,k_1)$ | Intra $-1$ / Succ $+5$ / Inter $+8$（真值 5）|
| 全部 $M[i][j]$ 取值范围 | $[0,\,9]$，上界恰为 $c-1$ |

### 表 3：KV cache 单价与总量（Qwen 两档 + 仓库统一锚点）

| 模型 | 层数 / KV 头 | 每 token KV | 128K 单序列 | 1M（$10^6$）|
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 28 / 4 | $2\times28\times4\times128\times2=57{,}344$ B = **56 KiB** | 7.00 GiB | 53.4 GiB |
| Qwen2.5-14B | 48 / 8 | $2\times48\times8\times128\times2=196{,}608$ B = **192 KiB** | 24.00 GiB | 183.1 GiB |
| LLaMA-3-8B（统一锚点）| 32 / 8 | **128 KiB** | 16.00 GiB | 122.1 GiB |
| LLaMA-3-70B（统一锚点）| 80 / 8 | **320 KiB** | 40.00 GiB | 305.2 GiB |

按 $2^{20}$ 计，7B 的 1M 是 56.0 GiB；口径不同时必须在正文写明用的是 $10^6$ 还是 $2^{20}$。7B/14B 是 Qwen 自己的配置，70B/8B 是仓库统一锚点，两组的来源不能混着说。128K 档走静态 YaRN 时序列反而不长于 131072，**KV 的账完全由长度与结构决定，与是否外推无关**。

### 表 4：激活、分块预填充与算力交叉点（Qwen2.5-7B，$d_{\text{model}}=3584$、$d_{ff}=18944$、$L=28$ 层、$N\approx7\times10^9$）

| 项 | 算式与结果 |
| --- | --- |
| 1M 输入单层 MLP 激活 | $2\,S\,d_{ff}\cdot2=2\times10^6\times18944\times2=7.58\times10^{10}$ B = 75.8 GB = 70.6 GiB（报告记作 71 GB，即同一笔账按 GiB 记）|
| chunked prefill，chunk $=32768$ | 75.8 GB $\times(32768/10^6)$ = 2.48 GB，降幅 96.72%（报告 96.7%）|
| chunked prefill，chunk $=131072$（官方启动推荐值）| 9.93 GB，降幅 86.89% |
| 算力比值 $2SLd_{\text{model}}/N$ | $2.87\times10^{-5}S$：$S=32768$ 时 0.94、$S=131072$ 时 3.76、$S=10^6$ 时 28.67 |
| 同式换 $N=7.61\times10^9$（模型卡参数量）| 0.86 / 3.46 / 26.37，交叉点仍在 32K 附近 |

表 4 的 $S$ 一律取 $10^6$，这就是报告「1 million tokens」的口径——75.8 GB 恰等于 70.6 GiB、对上报告的 71 GB，降幅 96.72% 也对上报告的 96.7%。若改用 $S=2^{20}=1048576$，同一组量变成 79.5 GB（74.0 GiB）、chunk 降幅 96.875% 与 87.5%、1M 算力比值 30.06；两套口径不能混着引用。

$4S^2Ld_{\text{model}}$ 是 $L$ 层注意力的量（$H d_{\text{head}}=d_{\text{model}}$，两次矩阵乘共 4 倍），$2NS$ 是参数项；两者在 $S\approx32\text{K}$ 附近相等（比值 0.94），1M 时注意力量是参数项的约 29 倍。报告口径与之一致：1M 时 attention 占前向时间超过 90%，所以稀疏注意力是必需项而不是可选项（MInference 把算力与访存降约 10 倍，摘要口径是 1M 场景 prefill 提速 3–7 倍，恢复精度的 NIAH 实验里保留约 4 倍）。

### 表 5：RULER 上的扩窗 vs 训长（报告 Table 4，口径为 128K 长度上的分数与全档平均）

| 模型 | 原始 RoPE（128K 档）| DCA+YaRN（128K 档）| 提升 |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 31.4 | 55.1 | +23.7 |
| Qwen2.5-14B-Instruct | 53.0 | 78.1 | +25.1 |
| Qwen2.5-32B-Instruct | 57.7 | 82.0 | +24.3 |
| Qwen2.5-72B-Instruct | 67.0 | 88.4 | +21.4 |
| Qwen2.5-7B-Instruct-1M（训到 256K）| — | 平均 91.8，128K 档 84.4 | 平均比 7B+DCA+YaRN 的 85.4 高 6.4 |
| Qwen2.5-14B-Instruct-1M（训到 256K）| — | 平均 95.7 | — |

读法：DCA+YaRN 把 128K 档整体抬升 21–25 分，但只有真在长数据上训过的 1M 版才能把 128K 档做到 84–92 分区间；基座越大，同样的扩展手段越值钱（72B 的 88.4 高于 14B 的 78.1）。

### 表 6：DCA 论文的源文实测口径（未复算，标清出处为源文）

| 口径 | 源文数字 |
| --- | --- |
| 4k 窗口模型无训练外推到 >32k | PPL 只增加 0.02 |
| Llama2 70B 无持续训练 | 支持 >100k；PG19 上 4k→96k 的 PPL 从 5.18 升到 5.59（+0.56）|
| 训练无关的 70B 对比闭源 | 达到 gpt-3.5-16k 的 94% |
| 与 PI／NTK 正交 | 已支持 32k 的模型再叠 DCA 到 192k，passkey 准确率与 PPL 都保持 |

### 代码：分区、温度、DCA 位置与三笔账

```python
import math
D, BASE, L, ALPHA, BETA = 128, 10000.0, 4096.0, 1.0, 32.0
theta = [BASE ** (-2 * i / D) for i in range(D // 2)]
lam = [2 * math.pi / t for t in theta]              # 每个平面的波长
r = [L / l for l in lam]                            # 训练窗口内转过的圈数
print('不插值', sum(x > BETA for x in r), 'ramp', sum(ALPHA <= x <= BETA for x in r),
      '全插值', sum(x < ALPHA for x in r), '/ 64')
print('r=32 临界 i* =', round((D / 2) * math.log(L / (2 * math.pi * BETA)) / math.log(BASE), 2))
print('r=1  临界 i* =', round((D / 2) * math.log(L / (2 * math.pi)) / math.log(BASE), 2))
print('温度 sqrt(1/t):', {s: round(0.1 * math.log(s) + 1, 3) for s in (4, 8, 16, 32)})

for base, tr in ((1e6, 32768), (1e7, 262144)):      # Qwen 两档：波长超过训练窗口的平面数
    rr = [tr / (2 * math.pi * base ** (2 * i / D)) for i in range(D // 2)]
    print(f'base={base:.0e} L={tr}: 波长>L 的平面 {sum(x < 1 for x in rr)} 个')

c, s, w, l = 10, 6, 4, 12                           # DCA 论文算例
Pk = [i % s for i in range(l)]
Pq = {'Intra': Pk[:], 'Inter': [c - 1] * l,
      'Succ': [(s + i % s) if (i % s) < w else (c - 1) for i in range(l)]}
print('Pk', Pk, '| Succ', Pq['Succ'], '| Inter', Pq['Inter'])
for i, j in ((6, 5), (6, 1)):
    print(f'(q_{i},k_{j}) Intra={Pk[i]-Pk[j]:+d} Succ={Pq["Succ"][i]-Pk[j]:+d} '
          f'Inter={Pq["Inter"][i]-Pk[j]:+d}')
dca = lambda i, j: (Pk[i] - Pk[j]) if i // s == j // s else (
    Pq['Succ'][i] - Pk[j] if i // s - j // s == 1 else Pq['Inter'][i] - Pk[j])
allM = [dca(i, j) for i in range(l) for j in range(i + 1)]
print('DCA 距离范围', min(allM), max(allM), '（上界 c-1 =', c - 1, '）')

KiB, GiB, GB = 1024, 1024 ** 3, 10 ** 9
for name, nl, nkv in (('Qwen2.5-7B', 28, 4), ('Qwen2.5-14B', 48, 8),
                      ('LLaMA-3-8B', 32, 8), ('LLaMA-3-70B', 80, 8)):
    kb = 2 * nl * nkv * 128 * 2
    print(f'{name:12s} {kb/KiB:5.0f} KiB/token | 128K {kb*131072/GiB:6.2f} GiB'
          f' | 1e6 {kb*1e6/GiB:6.1f} GiB')
S, d_ff = 10 ** 6, 18944                   # 报告的 1M token 口径（十进制）
act = 2 * S * d_ff * 2
print(f'1M 单层 MLP 激活 {act/GB:.1f} GB；chunk=32768 后 {act*32768/S/GB:.2f} GB'
      f'（降 {1-32768/S:.2%}）')
print('算力比值 2*S*L*d/N (N=7e9):',
      [round(2 * 28 * 3584 / 7e9 * x, 2) for x in (32768, 131072, 10 ** 6)])
```

```text
不插值 21 ramp 25 全插值 18 / 64
r=32 临界 i* = 20.94
r=1  临界 i* = 45.03
温度 sqrt(1/t): {4: 1.139, 8: 1.208, 16: 1.277, 32: 1.347}
base=1e+06 L=32768: 波长>L 的平面 24 个
base=1e+07 L=262144: 波长>L 的平面 21 个
Pk [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5] | Succ [6, 7, 8, 9, 9, 9, 6, 7, 8, 9, 9, 9] | Inter [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
(q_6,k_5) Intra=-5 Succ=+1 Inter=+4
(q_6,k_1) Intra=-1 Succ=+5 Inter=+8
DCA 距离范围 0 9 （上界 c-1 = 9 ）
Qwen2.5-7B      56 KiB/token | 128K   7.00 GiB | 1e6   53.4 GiB
Qwen2.5-14B    192 KiB/token | 128K  24.00 GiB | 1e6  183.1 GiB
LLaMA-3-8B     128 KiB/token | 128K  16.00 GiB | 1e6  122.1 GiB
LLaMA-3-70B    320 KiB/token | 128K  40.00 GiB | 1e6  305.2 GiB
1M 单层 MLP 激活 75.8 GB；chunk=32768 后 2.48 GB（降 96.72%）
算力比值 2*S*L*d/N (N=7e9): [0.94, 3.76, 28.67]
```

表 1 到表 4 的数字与上面两段输出逐项一致；表 5 与表 6 是源文口径，不参与复算。表 6 之外的硬结论必须落在正文：**YaRN 与 DCA 都不改变 FLOPs 与 KV 字节数**，它们只把位置语义拉回训练分布——事后扩展省掉的是训练，不是推理账单。

## 常见追问

- **追问**：DCA 是不是一种稀疏注意力或滑窗？
  - 要点：不是。DCA 不改注意力的稀疏性（每个 query 照样看全部前驱 key），它只重映射相对位置矩阵，把距离压回 $[0,c-1]$。它与稀疏注意力正交：MInference 那类是「少算一些 key」，DCA 是「换一种位置语义」。二者组合时还会互相干扰（见下一条）。
- **追问**：为什么 Qwen2.5-1M 里 DCA 和稀疏注意力会打架，怎么修的？
  - 要点：DCA 的相对位置在 chunk 边界不连续，会破坏 MInference 依赖的 Vertical-Slash 模式——未修时在超过 400K 的上下文上 NIAH 检索准确率会掉到 60% 或更低。修法是**只在挑选关键 token 的阶段恢复连续相对位置，最终注意力计算仍用 DCA 的不连续位置**；同时用 FlashAttention 的 `softmax_lse` 定义 attention recall $\exp(\text{lse}_{\text{sparse}}-\text{lse}_{\text{full}})$，在 1M 校准集上重新搜稀疏配置（原始 MInference 的离线搜索通常保持在 32k 以下，对 1M 不合适）。修完后恢复大部分性能，并保留约 4 倍 prefill 提速。
- **追问**：用了 static YaRN 要注意什么？dynamic 版本呢？
  - 要点：static YaRN 与输入长度无关，短序列也被整体压缩，所以会损伤短文本质量（模型卡原文口径），只应在确实需要长上下文时开启。dynamic 取 $s=\max(1,l/L)$ 时每个历史 token 的旋转角都会随长度变化，**必须缓存未施加 RoPE 的 K（或缓存位置本身）**，否则 KV cache 里已旋转的 K 会和当前 $s$ 不一致——这条与 [[llm-internals-08]] 的口径一致。
- **追问**：开启 rope_scaling 或换 DCA 的 chunk size 之后，工程上还要做什么？
  - 要点：锁框架版本；在短上下文基准上回归（static YaRN 的代价正好落在那里）；prefix cache／KV cache 的 key 必须包含 rope 配置（type、factor、original_max_position_embeddings，以及 DCA 的 chunk size），改因子或改 chunk size 都要让旧缓存失效，否则会拿到用错位置算出来的 K。
- **追问**：1M 上下文部署时显存怎么估？
  - 要点：只按权重估会严重低估。7B 的 KV 是 53.4 GiB/请求（$10^6$ 口径）到 56.0 GiB（$2^{20}$ 口径），单层 MLP 激活在 1M 输入下就有 75.8 GB，超过权重与 KV 的总和；官方启动口径是 `--tensor-parallel-size 4`（7B）／8（14B）、`--max-model-len 1010000`、`--enable-chunked-prefill --max-num-batched-tokens 131072`、`--max-num-seqs 1`，需要装 vLLM 的 `dev/dual-chunk-attn` 分支，并给出 1M 序列的显存门槛 7B ≥120 GB、14B ≥320 GB（多卡总和）。chunked prefill 取 32768 一档能把激活降 96.875%。
- **追问**：把「扩到 1M」当成「能用 1M」，会错在哪？
  - 要点：错在三层。第一层是位置：RULER 评测的 17 个宣称至少 32K 窗口的模型里，只有一半在 32K 上还能保持可接受表现。第二层是内容：lost-in-the-middle 的 U 形曲线说明「放得下」不等于「用得上」——20／30 文档设置下 GPT-3.5-Turbo 最差位置的准确率低于它自己 56.1% 的 closed-book 成绩，把检索文档从 20 篇加到 50 篇只涨约 1.5%（见 [[llm-internals-13]]）。第三层是本模型自己的对照：DCA+YaRN 能把 128K 档抬 21–25 分，但 128K 档最好的训练外推结果（7B 的 55.1）仍远低于训到 256K 的 1M 版（平均 91.8）；只在 32K 上训过的模型加 DCA 能在 1M passkey 上过 80%，换到多针、多值 NIAH 就会露出差距。评测要用 RULER 的多针、多跳追踪与聚合，不能只看单针 NIAH。

## 相关题目

- [[llm-internals-08]]：RoPE 的完整推导与 PI／NTK／NTK-by-parts／YaRN 的演进链，是本题的前置；这里只取它的「波长分区 + attention 温度」两条结论，在此之上补 DCA 与 Qwen 的两档配置。
- [[llm-internals-13]]：lost-in-the-middle 的 U 形曲线与「宣称窗口 vs 有效窗口」的量化口径，回答「扩窗之后为什么还是用不上」。
- [[llm-internals-02]]：KV cache 的显存公式与带宽口径；表 3 的 56／192／128／320 KiB 与它的记账方式一致。
- [[alibaba-03]]：同公司题，Qwen 长上下文与思考预算的服务侧口径，含 YaRN factor 4.0 那一档的部署取舍。

## 参考资料与归属

- **RoPE（旋转位置编码）背后的数学** —— Outcome School 博客（该条来源未署个人作者与日期）：<https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding>。旋转构造、平面切分与「打分只依赖相对距离」的直觉来自这篇。
- **RoFormer: Enhanced Transformer with Rotary Position Embedding（延伸）** —— Jianlin Su et al.，2021-04-20：<https://arxiv.org/abs/2104.09864>。旋转矩阵的正交性与 $\langle R_mq,\,R_nk\rangle=\langle q,\,R_{n-m}k\rangle$ 的一手出处。
- **YaRN: Efficient Context Window Extension of Large Language Models（延伸）** —— Bowen Peng et al.，2023-08-31：<https://arxiv.org/abs/2309.00071>。频率三区划分、$\gamma(r)$ 的 ramp、$\sqrt{1/t}=0.1\ln s+1$、以及 7B 2k→32k 的 400 步／128 A100-h 与 PI 对照 640 A100-h 的口径都来自这篇。
- **RULER: What's the Real Context Size of Your Long-Context Language Models?（延伸）** —— Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg，2024-04-09：<https://arxiv.org/abs/2404.06654>。「宣称窗口 vs 有效窗口」的评测口径、以及 17 个宣称至少 32K 的模型里只有一半在 32K 上保持可接受表现的结论来自这篇。
- **延伸来源说明**：DCA 论文（*Training-Free Long-Context Scaling of Large Language Models*，arXiv:2402.17463）给出三段式的公式、$c=10/s=6/w=4$ 的算例、$P_k$ 保持不变以复用 KV cache、与 FlashAttention 2 的无缝结合、「避免线性缩小位置索引或放大 base 频率」的定位、192k 的正交性结果、4k 无训练扩到 32k 只涨 0.02 PPL、70B 从 5.18 升到 5.59、训练无关的 70B 达到 gpt-3.5-16k 的 94%，以及作为对照的长对话微调成本（7B 约 40 GPU-h、13B 约 60 GPU-h）；Qwen2.5-1M 技术报告（arXiv:2501.15383）给出「主因是训练中未见过的大相对位置距离」、总是把 YaRN attention scaling 与 DCA 一起用、「都不改变训练长度内的行为」、RULER 的 31.4→55.1 等对照、只训到 32K 的模型加 DCA 在 1M passkey 上过 80%、MInference 与 DCA 组合的 60% 精度问题与两处修法、1M 时 attention 占前向时间超过 90%、以及 71 GB 激活与 96.7% 降幅；Qwen2.5 技术报告（arXiv:2407.10671）、Qwen 官方博客与模型卡（<https://qwenlm.github.io/blog/qwen2.5-1m/>）提供两档 config 事实（`max_position_embeddings` 32768 与 1010000、`rope_theta` 1e6 与 1e7、`rope_scaling` 均为 `None`）、128K 档的静态 YaRN 配置（factor 4.0、original 32768）与短文本提醒、以及 1M 的启动参数与 120 GB／320 GB 门槛。以上均不在上面的来源数组里，正文已标注哪条数字来自哪篇。表 1 到表 4 与代码输出的数字是按公开公式自行复算的结果（脚本见 `.work/lane-b/alibaba-05-recompute.py`），表 5 与表 6 是源文口径。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
