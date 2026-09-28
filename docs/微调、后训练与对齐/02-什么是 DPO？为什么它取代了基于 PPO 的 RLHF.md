---
type: question
id: finetuning-02
topic: 微调、后训练与对齐
order: 2
question: 什么是 DPO？为什么它在许多实验室里取代了基于 PPO 的 RLHF？什么情况下 online RL 仍然更好？
question_en: What is DPO, why did it replace PPO-based RLHF in many labs, and when is online RL still better?
asked_at: [Hugging Face, Scale AI]
level: 高阶
tags: [dpo, rlhf, 偏好优化, offline]
sources:
  - title: Direct Preference Optimization (DPO)
    url: https://outcomeschool.com/blog/direct-preference-optimization-dpo
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model（延伸）
    url: https://arxiv.org/abs/2305.18290
    author: Rafailov et al. (NeurIPS 2023)
    published: 2023-05-29
related: [finetuning-01, finetuning-03, finetuning-11, llm-internals-11]
updated: 2026-09-28
---

## 一句话答案

> DPO 把「KL 约束下的奖励最大化」重参数化了一次：该目标的最优策略有闭式解 $\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$，反解出 $r=\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z(x)$，代回 Bradley-Terry 偏好似然后配分项 $Z(x)$ 在成对比较中相消——**奖励模型被隐式地表示成策略与参考模型的对数比**，RL 问题退化成二分类式的监督学习。
> 取代 PPO 的理由是工程性的：不需要单独的 RM、不需要在线 rollout、不需要 value network，常驻模型从四份降到两份，训练稳定且只有几十行代码。
> 代价是它**离线**：策略不去探索，数据集没覆盖的偏好就学不到，因此有可验证奖励的任务（数学、代码）上 online RL 仍然更好。

## 面试官在考什么

- 能否**推出** DPO 损失而不是背下来。考点在第 2–4 节四步：闭式最优策略 → 反解奖励 → 代回 BT → $Z(x)$ 相消。$\pi^*$ 怎么来的、$Z(x)$ 为什么能扔，最能区分「读过论文」和「理解论文」。
- 能否把「为什么取代 PPO」落到具体资源上：少一个 RM、少一个 value network、少一次在线采样循环，而不是笼统地说「更简单更快」。
- 是否清楚 DPO 的**边界**：offline、参考模型固定、只用静态偏好集，所以发现不了数据集外的错误；以及长度/风格偏置从哪来。
- 能否给出「online RL 何时更好」的**判据**：奖励可自动验证、需要探索与自我纠错的场景，在线闭环才有信息增益。

常见错误答案：

- 说 DPO「不需要 reward model」就结束。准确说法是**不再训练显式 RM**，但策略本身被重参数化成了隐式 RM；它仍是相对参考模型定义的，无法脱离 $\pi_{\text{ref}}$ 单独存在。
- 说 DPO 与 PPO「数学上等价」而不加限定。二者共享同一最优解族，但数据分布不同：PPO 的样本来自当前策略，DPO 的 pair 来自冻结的 $\pi_{\text{ref}}$。「等价」只在同一份离线数据的覆盖范围内成立。
- 说 DPO 省显存所以训练更便宜就收尾。省的是显存与工程复杂度，不是数据成本——偏好标注这一项两者一样贵，只是从 RL 循环挪到了数据侧。

## 原理与推导

### 1. 起点：带 KL 约束的 RLHF 目标

$$\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\Big[r(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\Big]$$

逐 token 写开就是 $\sum_t \beta\log\frac{\pi_\theta(y_t|\cdot)}{\pi_{\text{ref}}(y_t|\cdot)}$，即整条序列的对数比，在 $y\sim\pi_\theta$ 下取期望才是序列级 KL；目标的来龙去脉见 [[finetuning-01]]。

### 2. 第一步：闭式最优策略

固定 $x$，把问题写成带归一化约束的拉格朗日量：

$$\mathcal{J}(\pi)=\sum_y \pi(y)\Big[r(x,y)-\beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\Big]+\lambda\Big(1-\sum_y\pi(y)\Big)$$

对 $\pi(y)$ 求偏导并令零，得 $r(x,y)-\beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}-\beta-\lambda=0$，解出

$$\pi^*(y|x)=\pi_{\text{ref}}(y|x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)\cdot\underbrace{\exp\!\Big(-\frac{\lambda}{\beta}-1\Big)}_{\text{与 }y\text{ 无关，即 }1/Z(x)}=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)$$

其中 $Z(x)=\sum_y \pi_{\text{ref}}(y|x)\exp\!\big(r(x,y)/\beta\big)$。两个极端：$\beta\to 0$ 时策略把所有质量压到 $r$ 最高的 $y$ 上（KL 形同虚设）；$\beta\to\infty$ 时 $\exp(r/\beta)\to 1$，$\pi^*\to\pi_{\text{ref}}$。所以 $\beta$ 就是「允许偏离参考模型多少」的刻度。

### 3. 第二步：反解奖励

把上式取对数反解出 $r$：

$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$$

这是整条推导的枢纽，结论很反直觉：**在 KL 约束的假设下，「奖励」与「策略对参考模型的对数比」一一对应**。奖励不再是外部打分的黑箱，而是策略自身携带的信息。$\beta\log Z(x)$ 是只依赖 prompt 的偏移量，同一 prompt 下所有回答共享。

### 4. 第三步：代回 Bradley-Terry，配分项相消

偏好数据形如 $(x,y_w,y_l)$，$y_w$ 是被人偏好的回答。用 Bradley-Terry 把两个奖励的差转成概率：$p(y_w\succ y_l|x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$。代入第 3 步的表达式，因为 $Z(x)$ **只依赖 $x$**，同一个 prompt 的两个回答做差时被完整消掉：

$$r(x,y_w)-r(x,y_l)=\beta\log\frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

$Z(x)$ 需要对整个回答空间求和，本来不可计算；成对比较把不可算的全局归一化变成可算的局部比值。**前提是 chosen 与 rejected 共享同一个 prompt**，否则两个 $Z(x)$ 不同，相消不成立。

### 5. 第四步：DPO 损失

把 $\pi^*$ 换成待训练的策略 $\pi_\theta$，取负对数似然：

$$\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\Big]$$

定义**隐式奖励** $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，则 $\mathcal{L}_{\text{DPO}}$ 就是把 $\hat r_\theta$ 塞进 reward model 用的同一个 BT 损失。「先训 RM、再用 PPO 最大化 RM」的两阶段流程被压成一个 logistic 回归形式的单阶段目标。

### 6. 梯度与「只约束差」的性质

记 margin $z=\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$，利用 $\frac{d}{dz}\big[-\log\sigma(z)\big]=-\sigma(-z)$：

$$\nabla_\theta\mathcal{L}_{\text{DPO}}=-\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{权重}\ w}\Big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\Big)\Big]$$

- **权重 $w$ 是「当前把顺序排错的概率」**。排序正确（$z\gg0$）时 $w\to0$，梯度消失，不再浪费更新；排序错误时 $w\to1$，重点纠正。DPO 因此天然是难例优先的，训练后期梯度自动变小。
- **方向**是提高 $y_w$ 的 log 概率、压低 $y_l$ 的 log 概率，强度由 $\beta$ 缩放。
- **损失只约束「差」**：给两条回答的 log 概率同时加常数 $c$，margin 与 loss 一字不变。所以损失下降**不代表**模型整体更像助手，只说明排序被拉开了；极端情况下两者的绝对概率可以一起下降（对两者都更不自信），流畅度已经退化。必须另外监控 chosen 的绝对 log 概率、隐式奖励的 margin 分布、输出长度与拒答率，相关失效模式见 [[finetuning-11]]。

### 7. 为什么取代 PPO：逐条工程理由

| 维度 | PPO-based RLHF | DPO |
| --- | --- | --- |
| 需要训练的模型 | reward model + policy（+ value network） | 只有 policy |
| 常驻显存里的模型 | policy / reference / reward / value 四份 | policy / reference 两份 |
| 在线 rollout | 是，每轮都要用当前策略生成回答 | 否，只用固定的离线 pair |
| 优势估计与 clip | 需要 GAE 与 clip 比率，超参多 | 无，只有一个 logistic 损失 |
| 主要超参 | lr、KL 系数、clip 范围、batch、rollout 规模 | $\beta$、lr |
| 实现量级 | 需要 RL 框架与 rollout 基础设施 | 几十行，普通训练循环 |
| 训练稳定性 | 对超参敏感，value 估计噪声大 | 稳定，可复现性好 |
| 收敛速度 | 需要多轮采样—更新 | 通常 1–3 个 epoch |

四条结论性理由：

1. **少一个 RM 和它那一轮训练**。RM 的训练与调参、训练时常驻的显存都省掉了；偏好数据的标注成本两者相同（见上文「常见错误答案」），省下的不是数据成本。
2. **不需要在线 rollout**。这是最大的一笔工程开销：PPO 每轮更新前都要用当前策略采样一批回答、前向 RM 打分、维护 rollout buffer；DPO 只遍历一个固定数据集。
3. **不需要 value network**。PPO 的 critic 与 policy 同量级，要额外一份权重 + 优化器状态 + 前向反向。
4. **稳定、超参少、实现短**。人力有限时，可复现性的差距比峰值效果更重要；DPO 的失败模式通常表现为「学得不够」而不是「训崩了」，$\beta$ 仍需调，但不必同时调 KL 系数、clip 范围与 rollout 规模。

效果口径要引用准确：Rafailov et al. 的报告是 DPO 在**情感控制**上超过 PPO-based RLHF，在**摘要与单轮对话**上持平或更好，同时实现与训练显著更简单——是摘要结论，不是「全面超越」。

### 8. 代价与边界：offline 的四条后果

1. **学不到数据集外的偏好**。pair 是 $\pi_{\text{ref}}$ 采样并被人标注的，$\pi_\theta$ 训练中不探索；偏好集里没出现过的错误，DPO 没有任何信号去修正。
2. **对数据质量、覆盖与偏置极度敏感**。噪声 pair 会被当成正确排序拟合；标注者一致的风格偏好（更长、更客套、markdown 更多）会被直接学成「好答案」的定义。**长度偏置**是经典症状。
3. **没有试错—反馈闭环**。只在给定 pair 上重排概率，不会去发现「比选中答案更好的答案」，所以在需要探索的任务（长链推理、工具使用、多步约束满足）上收益有限。
4. **参考模型固定**。训练越久，$\pi_\theta$ 离 $\pi_{\text{ref}}$ 越远，隐式奖励的分母变成一个越来越陈旧的锚点，容易过拟合偏好集。

DPO 也**不能替代 SFT**：它需要模型已经会按助手格式作答、且 chosen 与 rejected 有可比的先验概率。在 base model 上直接跑 DPO，策略连「怎么回答」都还没有，偏好信号无从附着。

### 9. 什么情况下 online RL 仍然更好

判据是**环境能否对策略自己产出的样本给出可靠反馈**。满足时，在线闭环每轮都在提供离线数据集里不存在的新信息：

- **有可验证奖励（RLVR）**。数学可判定对错，代码可跑单测，格式与约束可程序化检查。此时让同一 prompt 采样 $G$ 条、用组内相对分数做优势（GRPO）就能反复纠错并越过示范的天花板，见 [[finetuning-03]] 与 [[finetuning-10]]。DPO 在这里帮不上：它没有「自己试出来的正确答案」可学。
- **需要探索与自我纠错的长链推理**。一条几千 token 的 CoT，正确答案可能来自策略自己的一条非典型路径；offline pair 只能告诉它「这两条里哪条好」，无法奖励一条从未被采样过的更优轨迹。
- **偏好数据稀缺但可自动评估**。标注预算不够做偏好集，但有规则奖励或单元测试时，online RL 的数据成本近似为零。
- **需要迭代式偏好收集**。真实偏好会随模型能力提升而漂移；online/iterative DPO、RLAIF 用当前策略产样本、重新标注、再训一轮，才能持续逼近。
- **多目标互相冲突**。有用性与无害性常由不同 RM 打分，在线优化可以用动态权重或约束形式在冲突面上取舍；DPO 的单一 pair 只能表达「这条比那条好」。

**公开配方通常是混合路线**：SFT 建立指令遵循 → DPO 做偏好取舍 → RLVR 在可验证任务上冲刺。Tulu 3 把这三段作为公开可复现的后训练配方发布；Llama 3 的路线是 RM → 拒绝采样 → SFT → DPO 多轮迭代，并说明不用更复杂的 RL 算法。

### 10. 实践要点

- **$\beta$ 是核心旋钮**，同时控制 KL 约束强度和隐式奖励尺度（$\hat r_\theta$ 与 $\beta$ 成正比）。常见起点 0.1。取小 → 允许大步移动、隐式奖励尺度小、更易过拟合偏好集；取大 → 贴近参考模型、学到的差异不明显。因为 $\hat r$ 带 $\beta$，**换 $\beta$ 等于换奖励尺度**，跨实验比较 margin 时要固定 $\beta$。
- **参考模型一般就用 SFT 模型**，不放梯度、可用推理模式或量化省显存，它与 PPO 里的 reference 扮演同一角色——防止策略跑出参考分布导致语言质量崩塌。
- **数据质量远重于数量**。标注一致性差的那部分会被当成噪声拟合；宁可少而干净。偏好数据的构造与检查见 [[finetuning-11]]。训练预算通常 1–3 个 epoch，再多就是过拟合偏好集。
- **监控症状指标**：输出平均长度（长度偏置）、拒答率（过度安全）、chosen 的绝对 log 概率（分布塌陷）、hold-out 通用能力集（对齐税）。

## 数值与代码验证

以下数字全部由 `.work/dpo_verify.py` 与 `.work/dpo_code_trim.py` 复算得到，口径写在各表里。

### 1. 闭式解、反解与 $Z(x)$ 相消

玩具分布：$\beta=0.5$，真实奖励 $r=[1.0,\,2.0,\,0.0]$，参考分布 $\pi_{\text{ref}}=[0.5,\,0.3,\,0.2]$。

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 配分项 | $Z=\sum_y\pi_{\text{ref}}(y)e^{r(y)/\beta}$ | $20.273973$ |
| 最优策略 | $\pi^*(y)=\pi_{\text{ref}}(y)e^{r(y)/\beta}/Z$ | $[0.182230,\ 0.807905,\ 0.009865]$ |
| 比值检验 | $\pi^*(y)/\pi_{\text{ref}}(y)$ vs $e^{r(y)/\beta}/Z$ | $[0.364460,\ 2.693017,\ 0.049324]$，两组完全一致 |
| 反解残差 | $\max_y\lvert\beta\log\frac{\pi^*}{\pi_{\text{ref}}}+\beta\log Z-r(y)\rvert$ | $0$（浮点精确） |
| $Z$ 相消 | $\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}-r(y)$ 的取值 | 三个 $y$ 上恒为 $-\beta\log Z=-1.504669$ |

最后一行即第 4 节的相消：$\log(\pi^*/\pi_{\text{ref}})$ 在每个 $y$ 上相差的正是与 $y$ 无关的常数，成对做差时它消失了。

### 2. 损失尺度、只约束差、梯度闭式

$\mathcal{L}=-\log\sigma(z)$，$z$ 是隐式奖励的差：

| margin $z$ | $\sigma(z)$ | 损失 | 梯度权重 $w=1-\sigma(z)$ |
| --- | --- | --- | --- |
| $+1.0$ | $0.7311$ | $0.3133$ | $0.2689$ |
| $0.0$ | $0.5000$ | $0.6931$ | $0.5000$ |
| $-1.0$ | $0.2689$ | $1.3133$ | $0.7311$ |

$z=0$ 时损失 $0.6931=\ln 2$，与二分类交叉熵在随机猜测处相同——DPO 的损失尺度就是「一个二分类问题」。另外三项验证：

| 检验 | 计算 | 结果 |
| --- | --- | --- |
| margin 与 loss | $\beta=0.1$，两条回答的 log 比值差为 $5$ | $z=0.5$，loss $=0.4741$ |
| 平移不变性 | 两条回答的 log 概率同时 $+3$ | $z=0.5$，loss $=0.4741$，**一字不变** |
| 梯度闭式 vs 数值 | $\partial\mathcal{L}/\partial\log\pi_\theta(y_w)$ | 闭式 $-\beta(1-\sigma(z))=-0.03775407$；中心差分同值 |

### 3. $\beta$ 同时缩放隐式奖励

固定 log 比值、扫描 $\beta$：

| $\beta$ | margin（log 比 = 1） | margin（log 比 = 2） |
| --- | --- | --- |
| $0.01$ | $+0.010$ | $+0.020$ |
| $0.5$ | $+0.500$ | $+1.000$ |
| $5.0$ | $+5.000$ | $+10.000$ |

对照上表：损失从 $z=0.01$ 时的约 $0.688$ 降到 $z=1.0$ 时的 $0.313$。**同样的策略改变，$\beta=0.01$ 时几乎不产生损失信号，$\beta=5$ 时已经饱和**——这就是「换 $\beta$ 等于换奖励尺度」的具体含义。

### 4. 常驻显存账：四个模型 vs 两个模型

7B 模型，bf16 权重与梯度、Adam 用 fp32 两矩 + fp32 主权重（$16$ bytes/param 的可训练开销）：

| 组成 | DPO | PPO-based RLHF |
| --- | --- | --- |
| policy（权重 $2$ + 梯度 $2$ + Adam $8$ + 主权重 $4$） | $104.3$ GiB | $104.3$ GiB |
| value network（与 policy 同量级，同样要训） | — | $104.3$ GiB |
| reference（bf16 权重，冻结、推理模式） | $13.0$ GiB | $13.0$ GiB |
| reward model（$7$B 量级，bf16 权重、冻结） | — | $13.0$ GiB |
| 合计（不含激活与 KV cache） | $\approx 117$ GiB | $\approx 235$ GiB |

口径说明：$13.0\text{ GiB}\times8=104.3\text{ GiB}$（$7\times10^9\times16/2^{30}$）。这是「全参数用 Adam 训练」的口径；若用 8-bit Adam 或 LoRA 只训适配器，这一项大幅缩小、比值随之改变，但「PPO 多一个 value network 和一个 reward model」的结构性差别不变。激活值随序列长度与 batch 变化很大，未计入。上表是量级估算，不是基准测试数字。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta):
    """单条 pair 的 DPO 损失：logp_* 是策略的 log 概率，ref_* 是参考模型的。"""
    z = beta * ((logp_w - ref_w) - (logp_l - ref_l))
    return -math.log(sigmoid(z)), z

# 1) 闭式最优策略 pi* = pi_ref * exp(r/beta) / Z，并验证反解 r = beta*log(pi*/pi_ref) + beta*logZ
beta, r, pref = 0.5, [1.0, 2.0, 0.0], [0.5, 0.3, 0.2]
Z = sum(p * math.exp(ri / beta) for p, ri in zip(pref, r))
pi = [p * math.exp(ri / beta) / Z for p, ri in zip(pref, r)]
print("Z =", round(Z, 6), " pi* =", [round(x, 6) for x in pi])
lr = [math.log(a / b) for a, b in zip(pi, pref)]          # 反解出的 log 比值
print("beta*log(pi*/ref) - r =", [round(beta * x - ri, 6) for x, ri in zip(lr, r)])  # 恒为 -beta*logZ
print("residual of inverse:", max(abs(beta * x + beta * math.log(Z) - ri)
                                  for x, ri in zip(lr, r)))

# 2) 损失尺度：z = 0 处等于 ln 2；权重 w = 1 - sigmoid(z) 是「排错的概率」
for z in (1.0, 0.0, -1.0):
    print(f"z={z:+.1f}  loss={-math.log(sigmoid(z)):.4f}  w={1 - sigmoid(z):.4f}")
print("ln 2 =", round(math.log(2), 4))

# 3) 只约束差：两条回答的 log 概率同时 +3，margin 与 loss 不变
b = 0.1
loss1, z1 = dpo_loss(-18.0, -25.0, -20.0, -22.0, b)
loss2, z2 = dpo_loss(-15.0, -22.0, -20.0, -22.0, b)
print("shift:", (round(z1, 4), round(loss1, 4)), (round(z2, 4), round(loss2, 4)))

# 4) 梯度闭式 dL/dlogp_w = -beta*(1-sigmoid(z)) 与中心差分对照
h = 1e-6
num = (dpo_loss(-18.0 + h, -25.0, -20.0, -22.0, b)[0]
       - dpo_loss(-18.0 - h, -25.0, -20.0, -22.0, b)[0]) / (2 * h)
print("dL/dlogp_w  numeric =", round(num, 8), " closed =", round(-b * (1 - sigmoid(z1)), 8))

# 5) 常驻显存账（GiB；bf16 权重/梯度 + fp32 Adam 两矩 + fp32 主权重 = 16 B/param）
GiB, P = 2 ** 30, 7e9
trainable, frozen = (2 + 2 + 8 + 4) * P / GiB, 2 * P / GiB   # policy 那一份 / 冻结模型只放 bf16 权重
print("policy(trainable)=%.1f  frozen=%.1f  DPO=%.1f  PPO=%.1f"
      % (trainable, frozen, trainable + frozen, trainable * 2 + frozen * 2))
```

```text
Z = 20.273973  pi* = [0.18223, 0.807905, 0.009865]
beta*log(pi*/ref) - r = [-1.504669, -1.504669, -1.504669]
residual of inverse: 0.0
z=+1.0  loss=0.3133  w=0.2689
z=+0.0  loss=0.6931  w=0.5000
z=-1.0  loss=1.3133  w=0.7311
ln 2 = 0.6931
shift: (0.5, 0.4741) (0.5, 0.4741)
dL/dlogp_w  numeric = -0.03775407  closed = -0.03775407
policy(trainable)=104.3  frozen=13.0  DPO=117.3  PPO=234.7
```

## 常见追问

- **追问**：DPO 的隐式奖励能不能拿来给别的模型打分？
  - 要点：形式上可以，$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 在任意 $(x,y)$ 上都可算，实践中有人用它做 best-of-n 或数据筛选。但它是**相对参考模型的**：换掉 $\pi_{\text{ref}}$ 数值全变，只能与同一 $\pi_{\text{ref}}$ 下的分数比较；它还是 $\beta$ 的线性函数，跨实验比较要固定 $\beta$。它是策略参数化的副产品，不是独立训练的回归器，外推行为没有保证。
- **追问**：DPO 与 PPO 在什么假设下等价？
  - 要点：在「KL 约束的奖励最大化 + 偏好由 Bradley-Terry 生成 + 数据分布相同」三条下共享同一最优解族——DPO 就是 RLHF 目标的重参数化，第 2–4 节的推导即证明。差异全在数据分布：PPO 用当前策略的在线样本，DPO 用 $\pi_{\text{ref}}$ 的离线 pair。所以「等价」指最优解的形式，不是训练动力学或实际效果。
- **追问**：为什么 DPO 也会长度偏置？
  - 要点：$\hat r$ 是**序列级**的对数比。若标注者系统性偏好更长的回答，$\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$ 就在奖励「更长」，而 $\pi_{\text{ref}}$ 的训练分布本身带长度先验，偏差经隐式奖励被放大成策略层面的啰嗦。缓解手段是构造长度平衡的 pair、按长度配对 chosen 与 rejected、或把长度作为显式特征剔除；同时监控平均输出长度。
- **追问**：KTO / ORPO / SimPO 这些变体各自在改什么？
  - 要点：**KTO** 去掉「成对」要求，用「这条好/不好」的二元标签配合前景理论（prospect theory）式的非对称加权效用函数，数据门槛低（不必凑 pair），代价是信息量本来就比成对比较少。**ORPO** 把 SFT 与偏好对齐合成一步，在交叉熵上叠加 odds-ratio 惩罚项，省掉单独 SFT 阶段与参考模型。**SimPO** 改隐式奖励的定义，去掉参考模型项、换成序列平均 log 概率并加目标 margin $\gamma$，省掉参考模型前向。共同方向都是「去掉一个模型或一种数据形式」，代价是在特定数据条件下不如原版 DPO。
- **追问**：为什么不能只用 DPO，跳过 SFT？
  - 要点：$\pi_{\text{ref}}$ 是损失里的锚点，且 chosen/rejected 需要在合理的助手分布上有可比的先验概率。base model 面对「指令 + 两个答案」时给出的 log 概率本身没有助手语义，反解出的隐式奖励不是「哪个回答更好」的信号。顺序是先 SFT 建立指令遵循与输出格式，再用 DPO 做偏好取舍，见 [[llm-internals-11]]。
- **追问**：如果坚持用 PPO，什么条件下值得？
  - 要点：奖励可自动验证且策略需要探索时（数学、代码、约束满足），或需要同时优化多个冲突目标、动态调整奖励权重时。收益来自在线闭环带来的新信息，而不是 PPO 这个算法本身；把 PPO 换成 GRPO 去掉 value network 往往更划算，见 [[finetuning-03]]。

## 公司变体

- **Hugging Face**：这家公司是 DPO 工程化的主要推动者之一——TRL 库的 `DPOTrainer`（建在 `transformers` 之上）让 DPO 成为 HF 生态里的默认偏好优化选项，实现细节公开可查。问法偏工程实现与配方：参考模型怎么放（同进程还是另起、要不要量化、能不能用 PEFT 适配器、把关掉适配器后的基座当隐式参考）、chosen/rejected 的 loss mask 怎么算（只在回答 token 上求和还是整段平均）、$\beta$ 与学习率怎么配、1–3 epoch 的预算怎么定、显存够不够放下两个 7B。也会追问实践症状：输出变长怎么查、拒答率上升怎么办、怎么和 `SFTTrainer` 串成流水线。原理层面会考「隐式奖励是什么」「为什么 $Z(x)$ 能消掉」，通常不要求手推拉格朗日量。
- **Scale AI**：业务是数据标注与评估，偏好数据的生产与质量就是主场。问法偏数据侧与评测侧：偏好对的标注指南怎么写才能让「哪个更好」可复现、标注者一致性怎么度量、噪声 pair 在 DPO 损失里会造成什么后果（哪个梯度项被放大）、怎么构造长度平衡的 pair 以避免长度偏置、怎么用 hold-out 人工评测早期发现「loss 降但模型更差」、什么信号说明该从 offline DPO 切到 online 数据收集。也会反过来问「给定一份有偏的偏好数据集，你会先清洗还是先训」。
- 两家的共同点：都不会停在「DPO 更简单」这个结论上，一定会追一个具体取舍（$\beta$、epoch、显存、pair 数量），或者追问一个失效症状怎么诊断。

## 相关题目

- [[finetuning-01]]：本题第 1 节直接复用了那道题的 KL 约束目标；先能写出 RLHF 的三段式与目标函数，DPO 的推导才有落点。
- [[finetuning-03]]：GRPO 用组内相对优势去掉 value network，与 DPO 去掉 value network 的动机不同——前者为在线 RL 降本，后者干脆不做在线 RL。两题对照能说清「省掉一个模型」的两种路径。
- [[finetuning-11]] 与 [[llm-internals-11]]：前者是 DPO 过拟合偏好数据表层特征（长度、格式、讨好）时的 reward hacking 视角，后者给出 post-training 的阶段划分与「不能跳步」的原因。

## 参考资料与归属

- [Direct Preference Optimization (DPO)](https://outcomeschool.com/blog/direct-preference-optimization-dpo) — Amit Shekhar (Outcome School)。DPO 的直觉解释、偏好数据的三段结构、损失函数的通俗写法（`Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))`）、$\beta$ 的常见取值 0.1、与 PPO 的逐项对比表，以及「参考模型是安全锚点」「DPO 不能替代 SFT」两条实践结论。
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model（延伸）](https://arxiv.org/abs/2305.18290) — Rafailov et al. (NeurIPS 2023)，2023-05-29。隐式奖励的重参数化、闭式最优策略、$Z(x)$ 在成对比较中相消、logistic 损失形式，以及第 7 节引用的效果口径（情感控制上超过 PPO-based RLHF，摘要与单轮对话上持平或更好，实现与训练更简单）。
- **延伸来源说明**：第 2–4 节的推导与第 6 节的梯度闭式取自上述论文的公式（论文摘要只给结论，正文给推导）；第 9 节里「混合路线」与 RLVR 的判据来自公开后训练配方的通行做法，具体表述与数字以 [[finetuning-10]] 的来源为准，此处不引用额外链接。第 7 节的四条工程理由与第 8 节的 offline 边界来自 2026-05-17 的 Outcome School 博客。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
