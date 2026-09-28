---
type: question
id: finetuning-01
topic: 微调、后训练与对齐
order: 1
question: 请端到端讲一遍 RLHF：reward model、策略优化、KL 惩罚。
question_en: Walk through RLHF end to end: reward model, policy optimization, KL penalty.
asked_at: []
level: 进阶
tags: [rlhf, ppo, reward-model, kl]
sources:
  - title: Reinforcement Learning from Human Feedback (RLHF)
    url: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Proximal Policy Optimization (PPO)
    url: https://outcomeschool.com/blog/proximal-policy-optimization-ppo
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Training language models to follow instructions with human feedback（延伸）
    url: https://arxiv.org/abs/2203.02155
    author: Ouyang et al. (OpenAI)
    published: 2022-03-04
  - title: Proximal Policy Optimization Algorithms（延伸）
    url: https://arxiv.org/abs/1707.06347
    author: Schulman et al.
    published: 2017-07-20
related: [finetuning-02, finetuning-03, finetuning-11, llm-internals-11]
updated: 2026-09-28
---

## 一句话答案

> RLHF 是三段串行：① 用人工示范做监督微调（SFT），得到会按助手格式作答的初始策略；② 让模型对同一 prompt 生成多条回答、由人排序，用 Bradley-Terry 似然训一个标量 reward model $r_\phi$；③ 把 $r_\phi$ 当奖励，用 PPO 更新策略，同时逐 token 惩罚它与 SFT 参考模型的 KL 偏离。
> 目标是 $\max_\theta\ \mathbb{E}_{y\sim\pi_\theta}\big[r_\phi(x,y)\big]-\beta\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_{SFT}\big)$：KL 项把策略锚在参考分布上，限制它能吃到多少 reward model 的分布外误差，$\beta$ 是「多拿奖励」与「别走远」之间的兑换比率。
> 代价是四个模型同场（policy / value / reward / reference）、人类标注与在线 rollout 的工程复杂度，以及 RM 一旦有偏就会被策略主动放大的风险。

## 面试官在考什么

- 能否给出三阶段各自的输入、产出与数据量级（示范 $10^4$ 条、偏好对 $10^5$ 量级、RL 阶段只需要未标注 prompt），而不是复述 pipeline 名称。
- 能否写出 Bradley-Terry 损失，并解释为什么只有**分数差**有辨识性——这直接决定了偏好数据必须成对采集、奖励进 RL 之前要做尺度与零点对齐。
- 能否写出带 KL 的目标函数，并把 KL 讲成「限制策略移动半径、压制 reward hacking」的约束，而不是「防止过拟合」；能否给出 $\beta$ 的量级与调参现象。
- 能否讲出 PPO 的机制细节：概率比 $\rho$、clip 目标里 $\min$ 的作用、优势估计（GAE）与 value network 的职责（降方差），以及「同一批 rollout 做多轮更新」这一样本效率来源。
- 能否把工程代价落到数字上：RM 的头怎么改、序列级奖励怎么分配到 token、长度偏置、超参敏感、四模型的显存账。

常见错误答案：

- 把 reward model 说成「回归人类打的绝对分」。RM 的分数整体加减常数损失不变，能学的只有差值；用绝对分还会把标注者的松紧差异当成真实信号学进去。
- 把 KL 惩罚说成「防止过拟合训练集」或「让训练稳定」。它约束的是策略相对参考模型走了多远，机制是压缩 RM 外推误差的可利用空间；InstructGPT 的消融里 $\beta$ 放大到默认值的 100 倍也修不回 DROP、SQuAD 上的回退，只让验证奖励掉下来。

## 原理与推导

### 1. 三阶段骨架：每段的输入与产出

| 阶段 | 输入 | 产出 | InstructGPT 口径（论文设定） |
| --- | --- | --- | --- |
| SFT | 人工写的 (prompt, 示范回答) 对 | 会听指令的初始策略，同时冻成后面的参考模型 $\pi_{SFT}$ | 12,725 条训练 prompt（论文写 "about 13k"），训 16 epoch |
| RM | 同一 prompt 的 $K$ 条回答被人工排序，拆成 $\binom{K}{2}$ 个偏好对 | 打分函数 $r_\phi(x,y)\in\mathbb{R}$ | 33,207 条 prompt，$K=4\!\sim\!9$，只训 1 epoch |
| RL | 未标注 prompt + 冻结的 $r_\phi$ 与 $\pi_{SFT}$ | 对齐后的策略 $\pi_\theta$ | 256k episodes，约 31k 唯一 prompt，batch 512 / minibatch 64 |

三段的成本结构完全不同：示范最贵（要写），偏好次之（要选，但一次排序能翻出 $\binom{K}{2}$ 个训练对），RL 的输入 prompt 本身不需要标注，贵在算力与工程。顺序不能跳：没有 SFT，采样出来的回答只是网页续写，标注者比较的是两段噪声，RM 学不到有用信号，参考模型也是坏的锚点。

### 2. reward model：从偏好的 log odds 到 pairwise 损失

**Bradley-Terry 假设**：存在潜变量 $r^*(x,y)$ 表示「人类对这个回答的偏好强度」，则

$$P(y_w \succ y_l \mid x) = \sigma\big(r^*(x,y_w) - r^*(x,y_l)\big),\qquad \sigma(z)=\frac{1}{1+e^{-z}}$$

用参数化的 $r_\phi$ 去拟合它，负对数似然就是 RM 的损失（InstructGPT 按 $\binom{K}{2}$ 归一，因为同一 prompt 的比较高度相关）：

$$\mathcal{L}_{\text{RM}}(\phi) = -\frac{1}{\binom{K}{2}}\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]$$

两个必须记住的性质：

- **平移不变性**：$r_\phi \to r_\phi + c$ 不改变损失，所以只有差值有辨识性，绝对数值没有量纲意义。进 RL 之前要做尺度对齐——InstructGPT 的做法是用一个 bias 把示范回答的平均分归到 0。
- **差值是 log odds**：$r_\phi(x,y_w)-r_\phi(x,y_l)=\log\frac{P(y_w\succ y_l)}{P(y_l\succ y_w)}$，于是 margin 可以直接翻译成概率：差 1 对应 73.1%，差 3 对应 95.3%（见数值表）。

实现细节：从 SFT 模型出发去掉最后的 unembedding 层，换成一个输出标量的投影头。同一个 prompt 的 $\binom{K}{2}$ 个对放进**同一个 batch element**，一次前向得到 $K$ 个分数——若把它们当独立样本打散，一个 completion 会被 $K-1$ 个梯度反复推，单遍数据就过拟合；论文因此只训 1 个 epoch。

**为什么用成对比较而不是绝对打分**：跨标注者的绝对分不可比（有人宽松有人严格），而「A 比 B 好」的噪声小得多；一次排序还能产出 $\binom{K}{2}$ 个训练对，$K=9$ 时是 36 个。这也解释了 RM 精度的天花板：InstructGPT 的标注者之间一致率是 72.6%±1.5%（留出标注者之间 77.3%±1.3%），而它训出的 6B RM 在训练标注者上 72.4%±0.4%、在留出标注者组上 69.6%±0.9%——在训练标注者上已经贴着人类一致率，此时继续加 RM 容量收益有限，该改的是标注指南与数据分布。

### 3. RL 目标与 KL 惩罚

$$\max_\theta\ \mathbb{E}_{x\sim D,\ y\sim \pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big] - \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{SFT}(\cdot\mid x)\big)$$

**逐 token 的实现口径**。整段 KL 在实践中不这么算，而是折进每一步的奖励：

$$r_t = -\beta\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{SFT}(y_t\mid x,y_{<t})}\quad (t<T),\qquad r_T = -\beta\log\frac{\pi_\theta(y_T\mid\cdot)}{\pi_{SFT}(y_T\mid\cdot)} + r_\phi(x,y)$$

即序列级 KL 取「逐 token KL 之和」，序列级奖励只在终止步给一次。这个写法是单样本估计：期望上 $\mathbb{E}_{\pi_\theta}\big[-\log\frac{\pi_\theta}{\pi_{SFT}}\big]=\mathrm{KL}(\pi_\theta\|\pi_{SFT})$，但方差大；工程上常用方差更小的 $r-1-\log r$（$r=\pi_{SFT}/\pi_\theta$），它非负，且两个分布相同时 $r\equiv1$、逐样本即为 0（数值段给出这两项的验算）。

**KL 项的三层作用**：

1. **分布锚点**：偏好数据只覆盖参考分布附近的文本。离开这个区域，$r_\phi$ 的输出等于外推，完全不可信；不加约束时策略会漂到重复、乱码、语言混杂的文本上，而这些区域恰好是 RM 会给偶然高分的地方。
2. **压制 reward hacking**：RM 是人类偏好的代理，会系统性偏好某些表层特征（更长、更自信、更多「我很乐意帮忙」、宁可拒答）。策略正是往 RM 梯度方向做分布优化的机器，会把这些特征一路放大，直到整段输出崩塌成骗过 RM 的文本（机制与各家的对策见 [[finetuning-11]]）。KL 让「跑远」的边际收益递减，从目标函数层面掐住这条路径。
3. **保持多样性**：KL 惩罚的是整条分布的形状变化，不只是抬高 top-1；没有它，策略熵会快速塌缩，采样多样性下降，靠 temperature 也救不回来。

**$\beta$ 是强度旋钮，但必须和奖励尺度一起定**。既然 RM 只有差值有辨识性，$r_\phi$ 的绝对尺度就是任意的，$\beta$ 单独没有意义：换一个训练更久、输出更「陡」的 RM，同一个 $\beta$ 的实际约束强度就变了。量级参照：InstructGPT 用 $\beta=0.02$，消融显示 $\beta=0$ 与 $\beta=2$ 都明显变差，最优在 0.01–0.02。

**KL 不能替代能力保持**：把 $\beta$ 提到 2.0 仍然修不回 DROP、SQuAD 上的回退，要保住预训练能力得靠混预训练梯度，见第 6 节。

### 4. PPO 在做什么

RLHF 把生成写成一个退化的 MDP：state 是 prompt + 已生成 token，action 是下一个 token，episode 到 EOS 结束；过程奖励只有 KL 惩罚，终止奖励才是 $r_\phi$ 给的标量。PPO 在此之上做三件事。

**（1）概率比 + clip：限制单次更新的幅度。** 记 $\rho_t(\theta)=\dfrac{\pi_\theta(y_t\mid s_t)}{\pi_{\theta_{old}}(y_t\mid s_t)}$，代理目标是

$$L^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_t\Big[\min\big(\rho_t(\theta)\hat A_t,\ \mathrm{clip}(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon)\hat A_t\big)\Big]$$

$\min$ 让它成为性能的悲观估计（下界）：$\hat A_t>0$ 时目标随 $\rho$ 增大，但 $\rho>1+\epsilon$ 之后 clip 项变成常数，再推高概率没有梯度；$\hat A_t<0$ 时目标随 $\rho$ 减小，$\rho<1-\epsilon$ 之后同样被截断。也就是「朝有利方向走，超出 $\epsilon$ 就没有奖励」，从而不需要 TRPO 的二阶约束。$\epsilon=0.2$ 是论文默认值；其 Table 1 的消融（7 个连续控制环境、共 21 次运行的平均归一化分数）显示不 clip 会崩到 −0.39，$\epsilon=0.2$ 最好（0.82），0.1 与 0.3 分别是 0.76 和 0.70。

**（2）GAE：把优势估出来。** 定义 TD 残差与截断的广义优势估计：

$$\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t),\qquad \hat A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots + (\gamma\lambda)^{T-t-1}\delta_{T-1}$$

value network 的职责是提供 baseline 以降方差，它不改变最优解，只决定能不能训得动。RLHF 里这件事特别难：奖励稀疏（只有终止步有 $r_\phi$），InstructGPT 还明确**不做折扣**（$\gamma=1$），于是 value 必须估「剩余 KL 罚分的累加 + 可能的终局奖励」。value 估不准时，优势的噪声会直接进策略梯度，这是 PPO 调不动的常见根因。InstructGPT 用 6B value，并**从 RM 初始化**——两者都是「prompt + response → 标量」，先学会打分再学当 baseline。

**（3）同一批 rollout 做多轮更新。** 这是 PPO 相对朴素策略梯度的样本效率来源，也是 RLHF 里最贵的一环（生成 rollout 要跑完整解码）。代价是每多跑一轮，$\rho$ 就离 1 更远，clip 会逐渐把梯度掐掉，所以 epoch 数本身要调；InstructGPT 的选择更保守：一个 batch（512 条）拆 8 个 minibatch，只做 1 个 inner epoch。

### 5. 四个模型同场：显存账与工程复杂度

| 角色 | 是否训练 | 为什么必须在场 | 显存（16 B/param 训练态、2 B/param 推理态） |
| --- | --- | --- | --- |
| policy | 是 | 被优化的策略，要生成 rollout 并反向 | 16 B/param |
| value | 是 | 优势估计的 baseline，同样要反向 | 16 B/param |
| reward | 否 | 给终止奖励，冻结 | 2 B/param |
| reference | 否 | 逐 token 算 KL，冻结 | 2 B/param |

按 bf16 权重 2 + bf16 梯度 2 + fp32 master 4 + Adam 一阶 4 + 二阶 4 = 16 B/param 的口径，一个 7B policy 与同规模 value 各占 112 GB，加上 14 GB 的 reward 与 14 GB 的 reference，权重与优化器态合计 **252 GB**：要四张 80 GB 卡才装得下，激活、KV cache 与通信缓冲只能挤剩下的空间。三个缓解手段：

- **把 RM 与 value 做小**：InstructGPT 对 1.3B / 6B / 175B 三档 policy 都用同一个 6B RM 与 6B value，让这部分开销与 policy 规模解耦（论文理由是 175B RM 训练更不稳定，也更不适合拿来初始化 value）。这里有个常见误传需要纠正：论文摘要里被引用最多的是「1.3B 的 InstructGPT 输出比 175B GPT-3 更受偏好」，它说的是**被对齐后的 1.3B policy 在人类偏好上超过大 100 倍的基座模型**，不是「用 1.3B 的 RM 去对齐 175B」——RM 与 value 都是 6B，policy 三档都训了。
- **去掉 value**：用组内相对优势替代 baseline（GRPO），直接省掉一份 16 B/param，见 [[finetuning-03]]。
- **去掉在线 RL**：把 KL 约束目标的闭式最优策略反解成一个分类损失（DPO），模型数从 4 降到 2，见 [[finetuning-02]]。

### 6. 工程细节与坑

- **奖励是序列级标量，要分配到 token**：如上文的 $r_t$，只有终止步带 $r_\phi$，其余步骤靠 GAE 用 $\lambda$ 指数衰减摊回去（数值段给出例子：终止的 +1 在 $\lambda=0.95$ 时对第 1 个 token 只贡献 0.857）。
- **长度偏置**是这类方法最经典的坑：偏好数据里更长的回答更容易被选中，RM 把「长」当成质量的代理特征，PPO 于是把输出越写越长，而 KL 只约束分布、不约束长度。对策要在数据与奖励两侧做：偏好数据长度平衡、奖励里加长度惩罚、把输出长度与拒答率、重复率一起作为必看监控曲线。
- **超参极敏感**：$\beta$、clip $\epsilon$、policy/value 学习率、batch 与序列长度、采样温度要一起调，任何一项错了都表现为「RM 分数涨、人评掉」。InstructGPT 的可复现设定：$\beta=0.02$、clip 0.2、采样温度 1、优势估计不打折扣、常数学习率 + 前 10 次迭代 warmup、权重 EMA 衰减 0.992。
- **对齐税（能力回退）**：RL 只在很窄的助手分布上优化，公开 NLP 数据集指标会掉。PPO-ptx 在同一个 minibatch 里先算 PPO 梯度、再算预训练梯度并累积进同一个梯度缓冲，预训练梯度乘系数 $\gamma=27.8$，每个 RL episode 配 8 倍预训练样本（论文设定：$\gamma$ 在 1.3B 上扫出来，$\ge 20$ 才能恢复 DROP、SQuAD）。代价是每个 minibatch 多一次预训练前向。
- **成本口径**：InstructGPT 的 175B SFT 用 4.9 petaflop/s-days，175B PPO-ptx 用 60 petaflop/s-days，相对 GPT-3 预训练的 3,640 只有约 1.8%；但这是算力账，不含人工标注成本，也不含四个模型并行的工程投入。

### 7. 为什么 RLHF 有效，代价是什么

**有效性来自把偏好压成一个可优化的标量**。「有用」「无害」「语气」「什么时候该拒答」都写不成可微损失，但人能比较两份回答。RLHF 先把这层比较关系压成标量 $r_\phi$，再把最大化它变成一次梯度上升。SFT 只能模仿示范（只有正例，且示范覆盖不全），RL 阶段能在同一个 prompt 下把更优一侧的概率抬起来——这是「取舍」能力，不是「模仿」能力。

**代价**：① 人类标注量大且一致性有限（论文口径：标注者之间一致率 72.6%±1.5%，6B RM 对训练标注者的预测精度 72.4%±0.4%，后者就是精度的现实上限）；② $r_\phi$ 只是代理，它一偏，整套目标就偏，而且策略会主动去搜索这个偏差；③ 训练不稳定、超参敏感、可比性差（换一个 RM 或换一版偏好数据，结果不能直接对比）；④ 在线 rollout + 四模型并行，工程复杂度远高于 SFT。

**什么时候不选它**：奖励可以被规则验证（数学答案、代码单测、格式检查）时，用可验证奖励的 RL 更省也更稳；目标是「一次性、低频、要求高」的生成时，best-of-n 只做推理侧重排就够了——不需要 PPO，但推理成本 ×$n$，而且它只是在冻结分布里挑样本，不能把概率质量搬到 RM 偏好的区域，收益会随 $n$ 很快被 RM 的排序误差吃平。RLHF 的价值在于把偏好写进权重，让推理时零额外成本。

## 数值与代码验证

| 量 | 计算 | 结果 | 口径说明 |
| --- | --- | --- | --- |
| BT 损失 | $-\log\sigma(\Delta)$，$\Delta=0/0.5/1/3$ | 0.6931 / 0.4741 / 0.3133 / 0.0486 | margin 越大损失越小，$\Delta=0$ 退化成 0.6931 |
| margin 的概率含义 | $\sigma(1)$、$\sigma(3)$ | 73.1%、95.3% | 差值即 log odds |
| 序列 KL 的预算 | $\sum_t \mathrm{KL}_t = 10$ nats，$\beta=0.02$ | $0.02\times10 = 0.2$ | 200 token 回答、逐 token KL 之和 |
| 同上放大到 50 nats | $0.02\times50$ | 1.0 | 惩罚项与 RM 分差同量级时，$\beta$ 才开始主导 |
| clip 区间 | $[1-\epsilon,\,1+\epsilon]$，$\epsilon=0.2$ | [0.8, 1.2] | $\rho$ 超出后该方向梯度为 0 |
| 终止奖励的摊回 | $\lambda=0.95$、4 步 episode、只看终局 +1 | 0.8574 / 0.9025 / 0.9500 / 1.0000 | 即 $\lambda^{T-1-t}$，$\gamma=1$ |
| 训练态显存 | 2+2+4+4+4 | 16 B/param | bf16 权重/梯度 + fp32 master + Adam 一二阶；论文自身口径是 fp16 权重与激活 + fp32 master |
| 7B 四模型合计 | $16N+16N+2N+2N$ | 112+112+14+14 = 252 GB | 同规模 policy/value + 推理态 RM/reference |
| SFT 算力占比 | $4.9/3640$ | 0.13% | 相对 GPT-3 预训练的 petaflop/s-days |
| PPO-ptx 算力占比 | $60/3640$ | 1.65% | 同上，两者合计 1.78% |
| episode 复用率 | $256\text{k}/31\text{k}$ | 8.3 次/prompt | 同一批 prompt 被反复采样 |
| 单 batch 比较数上限 | $64\times\binom{9}{2}$ | 2,304 | batch 内 64 个 prompt、$K=9$ 时的上限 |
| $\beta$ 消融跨度 | $2.0/0.02$ | 100 倍 | 论文口径：$\beta$ 放大 100 倍仍修不回 DROP/SQuAD |

```python
import math

# 1) Bradley-Terry：只看分数差，差值是 log odds
for gap in (0.0, 0.5, 1.0, 3.0):
    print(f"gap={gap:<4} sigma={1/(1+math.exp(-gap)):.4f} loss={-math.log(1/(1+math.exp(-gap))):.4f}")

# 2) KL 的两种写法：定义式 E_pi[log(pi/ref)] 与估计量 r-1-log(r)（r = ref/pi）
pi = [0.4, 0.35, 0.25]
ref = [0.5, 0.30, 0.20]
exact = sum(p * math.log(p / r) for p, r in zip(pi, ref))
est1 = sum(p * math.log(p / r) for p, r in zip(pi, ref))          # 单样本估计量 -log(pi/ref) 的期望，与定义式同式
est2 = sum(p * ((r / p) - math.log(r / p) - 1.0) for p, r in zip(pi, ref))  # 低方差版本
same = sum(p * ((p / p) - math.log(p / p) - 1.0) for p in pi)     # 分布一致时 r=1，逐样本为 0
print(f"KL={exact:.6f}  E[log pi/ref]={est1:.6f}  E[r-1-log r]={est2:.6f}  ref=pi 时={same:.6f}")

# 3) PPO clip：min 把「朝有利方向走太远」的那一侧梯度掐掉
eps = 0.2
for adv in (1.0, -1.0):
    for rho in (0.5, 1.5):
        clipped = min(max(rho, 1 - eps), 1 + eps)
        print(f"adv={adv:+.0f} rho={rho}  unclipped={rho*adv:+.2f}  clipped={clipped*adv:+.2f}  min={min(rho*adv, clipped*adv):+.2f}")

# 4) GAE：终局才给的标量奖励如何摊回每个 token（gamma=1, lambda=0.95, beta=0.02）
beta, lam, kl_t = 0.02, 0.95, [0.1, 0.2, 0.3, 0.4]
r = [-beta * k for k in kl_t]
r[-1] += 1.0                                     # 终止步加上 reward model 的标量
adv = [sum(lam ** (l - t) * r[l] for l in range(t, len(r))) for t in range(len(r))]
print("rewards", [round(x, 4) for x in r])
print("GAE    ", [round(x, 4) for x in adv])

# 5) 四模型显存账：bf16 训练态 16 B/param，纯推理 2 B/param
N = 7e9
GB = lambda bytes_per_param: bytes_per_param * N / 1e9
print(f"policy {GB(16):.0f} + value {GB(16):.0f} + reward {GB(2):.0f} + reference {GB(2):.0f} = {GB(36):.0f} GB")

# 6) InstructGPT 口径换算
print(f"SFT 4.9/3640={4.9/3640*100:.2f}%  PPO-ptx 60/3640={60/3640*100:.2f}%  合计 {(4.9+60)/3640*100:.2f}%")
print(f"256k episodes / 31k prompts = {256000/31000:.1f} 次/prompt；单 batch 比较数上限 64*C(9,2) = {64*36}")
```

```text
gap=0.0  sigma=0.5000 loss=0.6931
gap=0.5  sigma=0.6225 loss=0.4741
gap=1.0  sigma=0.7311 loss=0.3133
gap=3.0  sigma=0.9526 loss=0.0486
KL=0.020481  E[log pi/ref]=0.020481  E[r-1-log r]=0.020481  ref=pi 时=0.000000
adv=+1 rho=0.5  unclipped=+0.50  clipped=+0.80  min=+0.50
adv=+1 rho=1.5  unclipped=+1.50  clipped=+1.20  min=+1.20
adv=-1 rho=0.5  unclipped=-0.50  clipped=-0.80  min=-0.80
adv=-1 rho=1.5  unclipped=-1.50  clipped=-1.20  min=-1.50
rewards [-0.002, -0.004, -0.006, 0.992]
GAE     [0.8393, 0.8856, 0.9364, 0.992]
policy 112 + value 112 + reward 14 + reference 14 = 252 GB
SFT 4.9/3640=0.13%  PPO-ptx 60/3640=1.65%  合计 1.78%
256k episodes / 31k prompts = 8.3 次/prompt；单 batch 比较数上限 64*C(9,2) = 2304
```

第 3 段（clip）要看两个方向的差别：$\hat A>0$ 且 $\rho=1.5$ 时 $\min$ 取到被 clip 的 1.2，梯度被掐掉；$\hat A<0$ 且 $\rho=0.5$ 时 $\min$ 同样取到被 clip 的 −0.8。也就是说两个方向都只在「对目标有利」的那一侧被拦住，这正是 PPO 不需要二阶约束就能当信赖域用的原因。第 2 段里前两列同式，所以相等是构造出来的；真正要看的是第三列：$r-1-\log r$ 的期望也等于同一个 KL，而参考分布与策略一致时它逐样本为 0，不会把「已经对齐」误判成偏离。

## 常见追问

- **追问**：KL 系数怎么调？调大调小分别发生什么？
  - 要点：$\beta$ 小 → 策略敢跑远，RM 分数涨得快，但很快进入外推区域：变长、套话多、过度拒答，最终语言质量崩塌；$\beta$ 大 → 策略几乎不动，SFT 的毛病原样保留，人评不涨。InstructGPT 的消融里 $\beta=0$ 与 $\beta=2$ 都差，最优在 0.01–0.02；由于 RM 只有差值有辨识性，$\beta$ 的绝对强度取决于 $r_\phi$ 的尺度，换 RM 或改奖励归一化后必须重扫。实践上不能只盯奖励：同时看 KL 曲线、输出长度、拒答率，用 KL 当方向盘。
- **追问**：RM 在分布外为什么不可靠？
  - 要点：RM 只在偏好数据覆盖的分布上拟合过 pairwise 排序，离开这个分布就是外推；而 RL 是主动把策略推向「当前 RM 打分最高」的优化器，等于在优化一个代理目标的最坏情形。三个具体后果：长度偏置、讨好式话术、过度拒答都会被放大。对策是 KL 约束 + 多轮重采偏好数据重训 RM + 用留出的人工评测判断进展（不要用 RM 分数判断对齐是否成功）。
- **追问**：为什么不直接用 RM 做 best-of-n，而要跑 RL？
  - 要点：best-of-n 不改权重，只是从冻结分布里挑样本，概率质量搬不到 RM 偏好的区域，收益随 $n$ 很快被 RM 的排序误差吃平，而且推理成本 ×$n$；RL 把偏好写进权重，推理时零额外成本，等价于在 KL 约束下把整条分布往 $r_\phi$ 的方向搬。反过来，低频、高要求的场景 best-of-n 性价比更高，所以线上常常两者并用。
- **追问**：PPO 和 DPO 是什么关系？
  - 要点：同一个 KL 约束目标的两种解法。$\max \mathbb{E}[r]-\beta\,\mathrm{KL}$ 的最优策略是 $\pi^*\propto\pi_{SFT}\exp(r/\beta)$；反解出 $r=\beta\log(\pi^*/\pi_{SFT})+\beta\log Z$ 再代回 Bradley-Terry，就得到只含策略与参考模型的 logistic 损失，模型数从 4 降到 2，也不需要 rollout。差别在数据分布：DPO 用固定的离线偏好对，没有探索，吃不到「策略自己生成、再被人标注」的那部分信息。展开见 [[finetuning-02]]。
- **追问**：奖励可以被规则验证时还需要 RM 吗？
  - 要点：不需要。数学答案、代码单测、格式约束这类可验证奖励不容易被 hack，还省掉 RM 的训练与推理；代价是只适用于有标准答案的任务，而且要防奖励结构的漏洞（只看最终答案不看推理，会鼓励瞎猜与走捷径）。这条路线通常配去掉 value network 的 GRPO，见 [[finetuning-03]] 与 [[finetuning-11]]。
- **追问**：为什么偏好数据要成对或排序，不能直接打 1–7 分？
  - 要点：绝对分跨标注者不可比，标注者的松紧差异会被当成真实信号；BT 模型只依赖差值，正好匹配「只有顺序」的数据。效率上也是排序更高：一个 $K=9$ 的排序就是 36 个训练对。论文里的 1–7 Likert 分数只用在人评环节（评测），不用在 RM 训练。

## 相关题目

- [[finetuning-02]]：DPO 把第 3 节的 KL 约束目标闭式反解成分类损失，去掉 RM 与 PPO。两题连着看，才能把「在线 RL」与「离线偏好优化」的取舍说清。
- [[finetuning-03]]：GRPO 用组内相对优势替代 value network，直接回答第 5 节「四个模型怎么变成三个」的问题。
- [[finetuning-11]]：reward hacking 是 KL 项存在的根本原因；那题给出各实验室的具体对策与失败案例。
- [[llm-internals-11]]：pre-training / SFT / preference optimisation 的横向对比，把三阶段放回整条后训练流水线，对比监督信号与算力的分配。

## 参考资料与归属

- [Reinforcement Learning from Human Feedback (RLHF)](https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf) — Amit Shekhar (Outcome School)。三阶段流程、偏好数据的采集方式、四个模型的角色、KL 惩罚的直观解释与 $\beta$ 的取值范围。
- [Proximal Policy Optimization (PPO)](https://outcomeschool.com/blog/proximal-policy-optimization-ppo) — Amit Shekhar (Outcome School)。clip 的直觉与 $\epsilon=0.2$ 的含义、同一批数据多轮更新带来的样本效率，以及 PPO 在 RLHF 中的位置与缺点（超参敏感、需要 value model）。
- [Training language models to follow instructions with human feedback（延伸）](https://arxiv.org/abs/2203.02155) — Ouyang et al. (OpenAI)，2022-03-04。全部 InstructGPT 具体数字与设定的来源：12,725 / 33,207 / 31,144 条 prompt，$K=4\!\sim\!9$ 与 $\binom{K}{2}$ 的 batch 组织方式，6B RM 与 6B value（value 从 RM 初始化），$\beta=0.02$、clip 0.2、无折扣、256k episodes、batch 512 / minibatch 64 / 单 inner epoch，$\gamma=27.8$ 与 8 倍预训练样本，4.9 / 60 / 3,640 petaflop/s-days，72.6%±1.5% 的标注者间一致率与 72.4% / 69.6% 的 RM 精度，以及 $\beta$、$\gamma$ 的消融口径。
- [Proximal Policy Optimization Algorithms（延伸）](https://arxiv.org/abs/1707.06347) — Schulman et al.，2017-07-20。clip 代理目标（式 7）与 $\epsilon=0.2$、截断 GAE（式 11–12），以及 Table 1 中 $\epsilon$ 取值的消融数据。
- **延伸来源说明**：第 2、4、6 节的公式与全部 InstructGPT 具体数字来自上面两条标注「（延伸）」的论文；两篇 Outcome School 文章提供三阶段骨架、KL 惩罚的直观解释与工程坑的框架。第 7 节 best-of-n 与 RL 的对比、第 3 节两类 KL 估计量的取舍属于基于公式的推理，未引用具体文献；第 5 节的四模型显存账按公开的 bf16 + Adam 训练态口径自行推算（论文自身写的是 fp16 权重与激活 + fp32 master）。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
