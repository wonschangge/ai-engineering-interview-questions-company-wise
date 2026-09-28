---
type: question
id: finetuning-03
topic: 微调、后训练与对齐
order: 3
question: 解释 GRPO，以及为什么在规模化场景下去掉 value network 很关键。
question_en: Explain GRPO and why dropping the value network matters at scale.
asked_at: [DeepSeek]
level: 高阶
tags: [grpo, rl, value-network, 推理]
sources:
  - title: Group Relative Policy Optimization (GRPO)
    url: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo
    author: Amit Shekhar (Outcome School)
    published:
  - title: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（延伸）
    url: https://arxiv.org/abs/2402.03300
    author: Shao et al. (GRPO 原始出处)
    published: 2024-02-05
  - title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（延伸）
    url: https://arxiv.org/abs/2501.12948
    author: DeepSeek-AI
    published: 2025-01-22
related: [finetuning-01, finetuning-02, finetuning-10, inference-serving-09]
updated: 2026-09-28
---

## 一句话答案

> GRPO 是 PPO 的变体：对同一个 prompt 采样一组 $G$ 个完整回答，用奖励函数给每个回答打分 $r_i$，用**组内**均值和标准差把奖励标准化成优势 $\hat A_i = (r_i - \mathrm{mean}(r))/\mathrm{std}(r)$，再把它广播到该回答的每个 token，套 PPO 式的 clip 目标加 KL 项更新策略。基线从「一个学出来的 value network」换成「同一组 $G$ 个样本的平均分」（$\bar r$ 里含该样本自身，不是 RLOO 式的 leave-one-out），所以 critic 及其优化器状态整份消失。
> 规模化下这一步很关键，是因为 critic 通常与策略同规模：按混合精度 Adam 的 16 bytes/param 口径（[[finetuning-09]]），7B 策略占 112 GB，再加一份 7B critic 又是 112 GB。去掉它，权重相关的固定开销从 252 GB 降到 140 GB；若奖励来自规则验证器而根本不需要 RM，则降到 126 GB，正好一半。省下的显存直接换成更大的 rollout batch、更长的上下文或更大的模型。
> 同时少了一个要在长序列上拟合期望回报的不稳定目标，而「组内相对」天然适配 0/1 的可验证奖励。代价是每个 prompt 要多采 $G$ 次、奖励方差大时优势噪声大、组内奖励全同（全对或全错）时优势为 0、该 prompt 不产生梯度。

## 面试官在考什么

- 会不会**先算 PPO 的账再讲 GRPO 的设计**。题面问的是「为什么去掉很关键」，答不出 value network 的成本就答不了这题：显存、算力、信用分配、拟合误差四条成本必须落到数字上（[[finetuning-01]]、[[finetuning-09]]）。
- 能不能把「组内相对」讲成一个统计设计，而不是一句「GRPO 不需要 value」。要能说出基线为什么能降方差、为什么减基线不改变梯度期望、组内标准化带来的平移/缩放不变性意味着什么。
- 会不会写目标函数：importance ratio、clip 的上下界、advantage 广播到 token、KL 项放在 loss 里的位置及其估计器。
- 是否理解它与**可验证奖励**（RLVR）的耦合：0/1 的 verifier 只提供序信息，组内对比恰好只需要序（[[finetuning-10]]）。
- 有没有规模化视角：显存与并行策略、rollout 引擎、采样与训练的吞吐配比，而不是只谈收敛曲线（[[inference-serving-09]]）。

常见错误答案：

- 「GRPO 去掉了 reward model」。去掉的是 value model（critic）；reward 无论是 RM 还是规则验证器都必须存在，否则没有学习信号。这两个模型在 PPO 里的分工不同，混为一谈会被直接追问。
- 把「组内标准化」讲成「减 batch 均值」。那是 batch 级基线，跨 prompt 混在一起算均值毫无意义——同一批里既有简单题又有难题，均值既不是任何一道题的 $V$，还会把题间难度差当信号。**分组是按 prompt 分的**，这是 GRPO 名字里 Group 的全部含义。
- 只讲省显存、不讲代价。面试官通常紧接着问「那你为什么不无脑把 $G$ 开到 64」「全对全错怎么办」。

## 原理与推导

### 1. 先算 PPO 的成本：value network 到底贵在哪

策略梯度要的是优势 $\hat A_t$：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot \hat A_t\right]
$$

最朴素的 REINFORCE 直接用整条轨迹的回报 $R$ 当 $\hat A_t$，无偏但方差极大。方差缩减的标准手段是减一个**只依赖状态、不依赖动作**的基线 $b(s_t)$，依据是

$$
\mathbb{E}_{a_t \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, b(s_t)\right] = b(s_t)\sum_{a_t}\nabla_\theta \pi_\theta(a_t\mid s_t) = b(s_t)\,\nabla_\theta 1 = 0
$$

所以减基线不改变梯度期望，只降方差；最优基线是 $V(s_t) = \mathbb{E}[R_t \mid s_t]$。PPO 用学出来的 $\hat V_\phi$ 干这件事，于是多出一个回归目标

$$
\mathcal{L}_{value}(\phi) = \mathbb{E}\left[\left(\hat V_\phi(s_t) - \hat R_t\right)^2\right]
$$

这个额外的目标带来四条成本，缺一条都讲不完整：

1. **一整套可训练参数与优化器状态**。源文口径是「the value model is almost the same size as the policy model」，critic 因此需要自己的 16 bytes/param（bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + Adam 一阶 4 + 二阶 4），与策略完全对等。
2. **每步多一次前向（以及反向）**。rollout 阶段每个 token 要预测一次 value；训练阶段 critic 要前向 + 反向更新。
3. **长序列的信用分配本身很难**。奖励只在序列末尾给一次，critic 要在几千个 token 上拟合「生成到这里为止的期望回报」，越长的回答这项回归越难。
4. **它是一个会失效的拟合目标**。value 有偏，优势就有偏，而优势直接乘在策略梯度上；奖励尺度随任务变化、奖励稀疏时训练早期尤其难拟合。这是训练不稳定的主要来源之一。

### 2. GRPO 的做法：把基线换成同组样本的平均分

四步：

1. **采样一组**。对同一个 prompt $q$，用旧策略 $\pi_{\theta_{old}}$ 采样 $G$ 个**完整**回答 $\{o_i\}_{i=1}^{G}$。常见 $G = 8 \sim 64$。
2. **逐个打分**。$r_i = R(q, o_i)$。$R$ 可以是规则/验证器（数学答案是否等价、代码是否通过单测），也可以是在人类偏好上训好的 RM。
3. **组内标准化**，得到该回答的优势：

$$
\hat A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

论文里的默认口径是 **outcome supervision**：同一个回答的**所有 token 共享同一个** $\hat A_i$，不做任何 token 级的价值估计——省掉 critic 的位置就在这里。（论文另有一个 process supervision 版本，用 PRM 给每一步打分，代价是要额外训一个过程奖励模型；默认版本不需要。）

4. **用 PPO 式的目标更新策略**，把 $\hat A_i$ 广播到 token 上：

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q,\ \{o_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left[\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\,\hat A_i\right] - \beta\,\mathbb{D}_{KL}\left[\pi_\theta \,\|\, \pi_{ref}\right]\right)\right]
$$

其中 $\rho_{i,t} = \pi_\theta(o_{i,t}\mid q,o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})$ 是 importance ratio，$1/|o_i|$ 是按回答长度取平均，$\pi_{ref}$ 通常是 SFT 后的模型。KL 用论文给的 k3 估计器（逐 token 在 $o_{i,t}$ 上取值）：

$$
\mathbb{D}_{KL}\left[\pi_\theta \| \pi_{ref}\right] = \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1 \;\ge\; 0
$$

记 $u = \pi_{ref}/\pi_\theta$，$u - \log u - 1$ 在 $u>0$ 上恒非负、$u=1$ 时取 0，且 $\mathbb{E}_{\pi_\theta}[u] = 1$、$\mathbb{E}_{\pi_\theta}[-\log u] = \mathrm{KL}$，所以它是 KL 的无偏低方差估计。**KL 项加在 loss 里，不是加在 reward 里**——这一点常被答错，它决定了梯度里多出一项对 $\pi_\theta$ 的直接约束，而不是通过奖励间接生效。

### 3. 「组内相对」为什么成立，以及它换来了什么

三个可以直接写出来的性质：

- **平移不变**：$\sum_i (r_i - \bar r) = 0$，奖励整体加常数不改变任何 $\hat A_i$。推论是 verifier 只需要给出**序**，不需要校准过的分数——这正是 0/1 规则奖励能直接用的原因（[[finetuning-10]]）。
- **缩放不变**：再除以 $\mathrm{std}(r)$ 后，$r \to ar + b\ (a>0)$ 完全不影响 $\hat A$。好处是不同 prompt 的奖励尺度自动对齐（数学题 0/1、代码题带部分分可以混着训）；代价见下一节——**组内方差很小时分母很小，噪声被放大**。
- **它是一个经验基线**：$\bar r$ 是 $V(q)$ 的蒙特卡洛估计，但只用同组 $G$ 个样本，而且 $r_i$ 自己参与了均值，所以严格讲不是无偏基线（自举项带来 $O(1/G)$ 量级的偏差），$G$ 越大越接近无偏。答辩时说「近似无偏的基线估计」比说「无偏」稳。

一句话概括设计：同一个 prompt 的多个回答构成天然对照组，把「绝对分数」翻译成「相对好坏」。这个翻译对**只要结果对错**的任务几乎无损，而对「哪一步错了」不做任何承诺——序列级信用分配是这套方法主动付出的代价。

### 4. 逐条回答「为什么在规模化下关键」

1. **显存与算力真省**。少一个与策略同规模的模型及其完整优化器状态，等于把权重相关的固定开销砍掉一份。省下来的显存不会闲置：换成更大的 rollout batch（rollout 是 memory-bound 的 decode，吞吐直接吃 KV cache 容量）、更长的上下文，或者干脆把参数量提上去。这条对应论文摘要里的原话——GRPO 是「a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO」（[[inference-serving-09]] 的容量视角）。
2. **少一个不稳定的拟合目标**。value 的偏差会直接污染优势估计，而优势乘在策略梯度上；长序列上这个回归任务还特别难。去掉 critic，训练里就少了一条会独立发散、需要单独调学习率与 batch 的曲线。
3. **与可验证奖励天然契合**。答案对错是 0/1，组内对比直接给出优势信号：只要组内不全同，得分高于组均值的回答拿正优势、低于的拿负优势。不需要 value 去预测期望回报，而 0/1 恰恰是 value 最难拟合准的稀疏信号（[[finetuning-10]]）。
4. **实现简单、超参更少**。少一个模型、少一个 optimizer、少一套 GAE 式的优势计算与 value clip，代码路径短很多，扩容时出问题的面也小。

### 5. 代价必须一起讲

- **每个 prompt 要采 $G$ 次**。生成的 token 数按 $G$ 倍增长，采样成本上升数倍（数值见下节）。所幸采样是 memory-bound、训练是 compute-bound，用高吞吐推理引擎跑 rollout 时这段算力相对便宜；但它绝不是零成本。
- **奖励方差大时优势噪声大**。$\bar r$ 的标准误是 $\sqrt{p(1-p)/G}$ 量级，$G$ 小的时候基线本身就在抖。
- **组内奖励全同时优势为 0**。全对或全错 → $\mathrm{std}(r)=0$ → 分子分母同时为 0，该 prompt 不产生任何梯度，纯浪费一次 rollout。这决定了必须做难度均衡的 prompt 采样或动态过滤。
- **信用分配变粗**。整条回答共享一个优势，长回答里错误的中间步骤也被一起加强，长链推理时这个粗糙度会显现。
- **除标准差会把低方差组放大**。$p \to 1$ 时少数错误样本承担几乎全部梯度（下节给数），这也是后来一批变体在「要不要除 $\mathrm{std}$、要不要除以长度」上做文章的原因。

### 6. 与 DPO 的边界

- DPO 是 **offline**：偏好数据固定，训练时不再采样，目标是解析出的分类式损失，没有 reward model、没有 rollout、没有 KL 调节旋钮（KL 已经烘进损失推导里）。
- GRPO 是 **online**：策略自己采样，需要奖励函数，能持续看到自己当前的分布并纠偏，因此可以学到离线数据里没有的行为（例如更长的思维链）。
- 判断规则很干脆：**有验证器就优先 online RL，只有偏好对就用 DPO**。两者也可以串起来，SFT → DPO → RLVR 是公开配方里常见的顺序（[[finetuning-02]]、[[finetuning-10]]）。

### 7. 出处与实证口径（以下按论文口径陈述）

- GRPO 由 **DeepSeekMath** 提出，论文摘要把它明确写成 PPO 的变体，动机之一是「在提升数学推理能力的同时优化 PPO 的显存占用」。同一篇摘要里的能力口径：DeepSeekMath 7B 在不使用外部工具、不做投票的情况下在竞赛级 MATH 上达到 51.7%，self-consistency 采样 64 次达到 60.9%；论文把数学推理能力归因于两点——公开网页数据上的数据筛选管线，以及 GRPO。
- **DeepSeek-R1** 进一步展示：推理能力可以通过**纯 RL** 被激励出来，不需要人类标注的推理轨迹；训练过程里涌现出自我反思、验证与动态策略调整等推理模式；在数学、代码竞赛、STEM 这类可验证任务上超过用人类示范做监督学习的同类模型；这些涌现出的推理模式还能被系统性地用来提升小模型。
- 这两条都是论文口径，不是复现实测结果；下文所有显存、概率与 FLOPs 数字都按「数值与代码验证」一节声明的假设自行复算。

## 数值与代码验证

口径声明：显存按 16 bytes/param（可训练参数，混合精度 Adam）与 2 bytes/param（冻结、只前向，bf16 权重）两个档位；容量用十进制 GB（$10^9$ 字节），与 [[finetuning-09]] 的 7B ≈ 112 GB 对齐；假设 critic 与 RM 都与策略同规模、critic 独立于策略（不共享 trunk）。表内不含激活、KV cache 与通信缓冲。

表 1：7B 训练的权重相关显存

| 组件 | 可训练 | 口径 | 7B 显存 |
| --- | --- | --- | --- |
| 策略 $\pi_\theta$ | 是 | 16 B/param | 112 GB |
| 参考模型 $\pi_{ref}$ | 否 | 2 B/param | 14 GB |
| 奖励模型 RM | 否 | 2 B/param | 14 GB |
| PPO 的 critic / value network | 是 | 16 B/param | 112 GB |
| **PPO 合计** | | | **252 GB** |
| **GRPO 合计（带 RM）** | | | **140 GB** |
| **GRPO 合计（规则验证器，无 RM）** | | | **126 GB** |

去掉 critic 省 112 GB，占 PPO 的 44.4%；再省掉 RM 是 126 GB，正好一半。放到 8×80 GB 上：PPO 的权重相关开销占 39.4%，GRPO 占 19.7%，中间差出的 126 GB 可以换算成约 63B 参数的 bf16 推理副本容量——对 rollout 引擎来说这就是更大的并发 batch。

表 2：复算源文的 4 答案例子（$r = [1, 0, 1, 0]$）

| 算法 | 结果 |
| --- | --- |
| $\bar r$ | 0.5 |
| 只减均值 $r_i - \bar r$ | $[+0.5, -0.5, +0.5, -0.5]$ |
| 除总体标准差（$\div G$，$\mathrm{std}=0.5$） | $[+1, -1, +1, -1]$ |
| 除样本标准差（$\div (G-1)$，$\mathrm{std}=0.5774$） | $[\pm 0.866]$ |

源文的示例只做到了「减均值」这一步（它随后补了一句实践中还会除以标准差），所以源文那个 $\pm 0.5$ 与论文公式的 $\pm 1$ 只差一个常数因子。分母用 $G$ 还是 $G-1$ 是纯全局缩放，等价于改学习率，不改变梯度方向；**要不要除 $\mathrm{std}$ 才是有实质影响的选择**，因为它改变的是不同 prompt 之间的相对权重。

表 3：伯努利奖励下的优势幅度。$p$ 为组内单样本答对概率，标准化后正确样本的优势是 $|A_+|=\sqrt{(1-p)/p}$，错误样本是 $|A_-|=\sqrt{p/(1-p)}$

| $p$ | $\lvert A_+\rvert$ | $\lvert A_-\rvert$ | 不对称倍数 |
| --- | --- | --- | --- |
| 0.50 | 1.000 | 1.000 | 1.00× |
| 0.70 | 0.655 | 1.528 | 2.33× |
| 0.90 | 0.333 | 3.000 | 9.00× |
| 0.95 | 0.229 | 4.359 | 19.00× |
| 0.99 | 0.101 | 9.950 | 99.00× |

加权和恒为 0（$p|A_+| = (1-p)|A_-| = \sqrt{p(1-p)}$），所以组内优势是严格零和的；但**难度失衡时梯度几乎全部来自少数样本**：模型已经 90% 会做的题，那 10% 的错误样本拿到 9 倍于正确样本的优势幅度。这就是「难题太少时训练信号又稀又尖」的定量版本，也是难度均衡采样的理由。

表 4：零梯度概率 $P(\text{组内全同}) = p^{G} + (1-p)^{G}$

| $p$ 与 $G$ | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 12.50% | 0.78% | 0.00% | 0.00% | 0.00% |
| 0.70 | 24.82% | 5.77% | 0.33% | 0.00% | 0.00% |
| 0.80 | 41.12% | 16.78% | 2.81% | 0.08% | 0.00% |
| 0.90 | 65.62% | 43.05% | 18.53% | 3.43% | 0.12% |
| 0.95 | 81.45% | 66.34% | 44.01% | 19.37% | 3.75% |
| 0.99 | 96.06% | 92.27% | 85.15% | 72.50% | 52.56% |

这张表给出两条工程结论：① 加大 $G$ 对**简单题**的浪费是有效的解药（$p=0.9$、$G=8$ 时 43% 的 prompt 白采，$G=64$ 时降到 0.12%），但它按 $G$ 倍加大采样成本——$G$ 的选择本质是拿算力换信号密度。② 对 $p=0.99$ 这种题，$G=64$ 仍有超过一半的 prompt 零梯度，靠加大 $G$ 是治不好的，只能把这类题从训练集里降权或过滤掉（难度均衡采样 / 动态过滤的动机）。

表 5：采样量与每 token 算力（$N = 7\times10^9$；前向 $2N$ FLOPs/token，前向+反向 $6N$）

| 视角 | PPO | GRPO |
| --- | --- | --- |
| rollout 每 token | $2N$（策略）+ $2N$（critic 前向）= $4N$ | $2N$（策略） |
| 训练每 token | $6N$（策略）+ $6N$（critic）= $12N$ | $6N$（策略） |
| 合计 | $16N$ | $8N$ |

在「critic 独立且每步都更新」的假设下，单位 token 的算力约为 PPO 的一半。但这个比较必须带上采样量的抵消项：同样 256 个 prompt、回答长度 1024 token，$G=8$ 时一次 rollout 生成 $256\times 8\times 1024 \approx 2.10$M token，$G=16$ 时约 4.19M token，而 PPO 口径下每个 prompt 只生成 1 条。**GRPO 省的是「模型数量」，不是「采样量」**——它用 $G$ 倍采样换掉一个与策略同规模的模型，这笔交易的划算程度取决于 critic 的实现方式（是否共享 trunk）、$G$ 和回答长度。

下面这段脚本复算表 2 到表 5 的数字，只用标准库：

```python
import math
from statistics import mean, pstdev, stdev

# 表 2：源文的 4 答案例子
r = [1.0, 0.0, 1.0, 0.0]
m = mean(r)
print("mean=%.4f pstdev=%.4f stdev=%.4f" % (m, pstdev(r), stdev(r)))
print("只减均值 ->", [x - m for x in r])
print("除总体std ->", [round((x - m) / pstdev(r), 4) for x in r])

# 表 3：伯努利奖励下的优势幅度与零和性
for p in (0.5, 0.7, 0.9, 0.95, 0.99):
    a_pos, a_neg = math.sqrt((1 - p) / p), math.sqrt(p / (1 - p))
    assert abs(p * a_pos - (1 - p) * a_neg) < 1e-12   # 加权和恒为 0
    print("p=%.2f |A+|=%.3f |A-|=%.3f 不对称 %.2fx" % (p, a_pos, a_neg, a_neg / a_pos))

# 表 4：组内奖励全同（优势为 0）的概率
for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    print("p=%.2f " % p + " ".join("G=%d:%6.2f%%" % (G, 100 * (p ** G + (1 - p) ** G))
                                   for G in (4, 8, 16, 32, 64)))

# 表 5：显存与采样量（16 B/param 可训练，2 B/param 冻结，十进制 GB）
N, gb = 7e9, 1e9
train, frozen = N * 16 / gb, N * 2 / gb
ppo, grpo_rm, grpo_v = 2 * train + 2 * frozen, train + 2 * frozen, train + frozen
print("单份可训练 %.0f GB 单份冻结 %.0f GB" % (train, frozen))
print("PPO %.0f GB | GRPO+RM %.0f GB | GRPO+验证器 %.0f GB" % (ppo, grpo_rm, grpo_v))
print("省 %.1f%% / %.1f%%" % (100 * (ppo - grpo_rm) / ppo, 100 * (ppo - grpo_v) / ppo))
for G in (1, 8, 16):
    print("G=%2d -> 256 prompt x 1024 token 的 rollout = %.2fM token"
          % (G, 256 * G * 1024 / 1e6))
```

## 常见追问

- **追问**：组大小 $G$ 怎么选？
  - 要点：$G$ 是「基线估计精度」与「采样成本」的交换。$G$ 越大，$\bar r$ 的标准误按 $1/\sqrt{G}$ 降，零梯度概率按 $p^G$ 降（表 4）；成本按 $G$ 线性涨。公开实践常见 $8\sim64$，且要和回答长度一起看：回复越长，单条样本越贵，$G$ 要相应调小。可操作的选法是从 $G=8$ 起，观察「零梯度 prompt 占比」与「优势的组间方差」，按目标信号密度往上加。
- **追问**：组内奖励全同（全对或全错）怎么办？
  - 要点：$\mathrm{std}=0$ → 优势为 0 → 该 prompt 贡献零梯度，白采一次。三个手段：① 难度均衡的 prompt 采样（按当前模型的历史通过率分层，把 $p$ 控制在 $0.2\sim0.8$）；② **动态过滤**——一个 group 内奖励全同时直接丢弃该样本并重采（DAPO 一类实现里的 dynamic sampling 就是这个思路），代价是采样量上升；③ 过滤后如果全批都被丢掉，说明该 batch 的题对当前模型没有信息量，应该换数据而不是硬训。
- **追问**：GRPO 里 KL 项还要不要？
  - 要点：要分清两个作用。KL 约束的是「不要离参考模型太远」，防的是奖励 hacking 与能力遗忘（[[finetuning-11]]）。它在 GRPO 中依然存在（论文默认带 $\beta$ 项），但一批后续变体选择直接去掉 KL、或把它从 loss 挪进 reward：去掉的前提是奖励本身足够难 hack（规则验证器）且有其他手段控长度。面试里稳妥的答法是「默认保留，去掉要有理由与监控指标」，而不是「GRPO 不需要 KL」。
- **追问**：它和 RLOO、Reinforce++、DAPO 这些变体差在哪？
  - 要点：按四条轴对比。① **基线怎么算**：GRPO 用组均值；RLOO 用 leave-one-out（对第 $i$ 个样本用其余 $G-1$ 个的均值当基线），两者都不需要 critic。② **要不要除组内标准差**：除以 std 让优势尺度无关，但低方差组会被放大（表 3）；只减均值是一个真实存在的设计选择。③ **归一化粒度**：先按序列长度取平均再按组平均，会让长回答里每个 token 的权重变小；改成 token 级归一化（长序列不被稀释）是一类改动方向，对应的长度偏置分析也是近两年讨论的重点。④ **clip 与超长控制**：把 clip 上下界解耦（放宽上界以鼓励低概率的好 token）、对超长回答做奖励塑形，都是这一族算法的工程补丁。判断标准只有一条：改动是否改变了「谁拿到多大梯度」。
- **追问**：为什么说它和可验证奖励特别配？
  - 要点：verifier 给的是 0/1，绝对数值没有意义（不同题的 0/1 不可比），而组内标准化把绝对分数变成组内相对位置，并且对奖励做平移/缩放都不变。反过来，如果奖励是一个需要校准的连续分数（如 RM 打分），组内标准化会丢掉「这一组整体都很好」这个信息——这既是优点也是损失，取决于任务（[[finetuning-10]]）。
- **追问**：GRPO 能不能用在开放式任务（写作、对话）上？
  - 要点：能，但优势变小。它的信用分配是序列级的，开放式任务里「整条回答好」与「某个句子好」差别很大，且奖励只能靠 RM 或 LLM judge，而这类奖励更容易被 hack（[[finetuning-11]]）。源文的口径也是：GRPO 最适合有可验证奖励的推理类任务，开放式任务上 PPO 更合适；DPO 系在开放式对齐里也常见，但那属于离线路线。

## 公司变体

这题标注在 **DeepSeek**。GRPO 出自 DeepSeekMath，DeepSeek-R1 又把它推到了推理模型的中心位置，所以这家公司对这道题的期待通常偏**原理推导 + 自家论文口径**，而不是泛泛的算法介绍：

- 期望你能现场把优势公式写出来，并说清「组内」的粒度是 prompt，不是 batch。
- 期望你复述论文的动机原话方向：GRPO 是 PPO 的变体，在提升数学推理能力的同时优化 PPO 的显存占用；以及 R1 的「纯 RL 激励推理能力、涌现自我反思与验证」这一口径。把动机说成「省掉 reward model」是典型扣分点。
- 期望你给出显存与采样量的量化账，而不是只说「更省」。省的是一个与策略同规模的模型及其优化器状态，采样反而增加 $G$ 倍——能主动说出这句权衡的候选人，会被认为读过论文也跑过训练。
- 工程侧的追问通常落在 rollout 怎么组织、$G$ 与回答长度怎么配、全对全错的组怎么处理、KL 项留不留；算法侧的追问会延伸到与 DPO、RLVR 的分工边界。

## 相关题目

- [[finetuning-01]]：RLHF 的端到端流程。value network、reward model、KL 惩罚在 PPO 流水线里的各自位置，是理解这道题的前置。
- [[finetuning-02]]：DPO 与 online RL 的边界。判断规则「有验证器走 online RL、只有偏好对走 DPO」是这道题的标准收尾。
- [[finetuning-10]]：RLVR 与可验证奖励。GRPO 组内标准化只需要奖励的序信息，这正是它与规则验证器天然契合的原因。
- [[inference-serving-09]]：容量规划视角。省下的一份模型显存如何换算成 rollout batch、KV cache 与吞吐，属于那一节的账。
- [[finetuning-09]]：16 bytes/param 的显存口径与 ZeRO 分片，本节的显存表直接沿用它的口径。
- [[finetuning-11]]：reward hacking。KL 项与奖励设计要防的就是这件事。

## 参考资料与归属

1. [Group Relative Policy Optimization (GRPO)](https://outcomeschool.com/blog/group-relative-policy-optimization-grpo)，Amit Shekhar（Outcome School）。提供 PPO 中 policy model 与 value model 的分工、reward model 与 value model 的区别（前者给整条回答打分、后者预测每一步的期望回报）、GRPO 的六步流程、组大小 $8\sim64$ 的实践区间、组内奖励全同时优势为 0 的失效模式、KL 惩罚防 reward hacking 的作用、以及 4 答案分组示例（奖励 $[1,0,1,0]$）。该文未给出显存、概率或 FLOPs 数字，正文表 1、表 3、表 4、表 5 与表 2 的标准化部分均为按本节声明的口径自行复算。
2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)（延伸），Shao、Wang、Zhu、Xu、Song、Bi、Zhang、Zhang、Li、Wu、Guo，arXiv 2024-02-05。GRPO 的原始出处：提供「PPO 的变体、在增强数学推理能力的同时优化 PPO 的显存占用」这一动机口径、组内标准化优势（outcome supervision）与 process supervision 两个版本、含 clip 与 KL 项的目标函数形式、k3 KL 估计器，以及 DeepSeekMath 7B 在 MATH 上 51.7%（无工具、无投票）与 self-consistency 64 采样 60.9% 的能力口径。正文第 2、3 节的目标函数与估计器按该文重述。
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)（延伸），DeepSeek-AI，arXiv 2025-01-22（Nature 645, 633–638, 2025）。提供纯 RL 激励推理能力（不需要人类标注推理轨迹）、自我反思/验证/动态策略调整等推理模式的涌现、在数学与代码竞赛与 STEM 等可验证任务上超过用人类示范监督训练的同类模型、以及这些模式可被用来提升小模型这四条结论口径。正文未引用其中的具体分数或训练成本数字。

表 1 至表 5 的全部数字、优势幅度公式 $|A_+| = \sqrt{(1-p)/p}$、零梯度概率 $p^G + (1-p)^G$、以及「PPO $16N$ / GRPO $8N$」的每 token FLOPs 比较，均按「数值与代码验证」一节声明的假设自行复算，不属于上述来源的结论。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
