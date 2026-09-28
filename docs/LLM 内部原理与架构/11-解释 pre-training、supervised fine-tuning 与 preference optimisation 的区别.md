---
type: question
id: llm-internals-11
topic: LLM 内部原理与架构
order: 11
question: 解释 pre-training、supervised fine-tuning 与 preference optimisation 之间的区别。
question_en: Explain the difference between pre-training, supervised fine-tuning and preference optimisation.
asked_at: [Meta, Scale AI]
level: 进阶
tags: [预训练, sft, rlhf, dpo]
sources:
  - title: Decoding InstructGPT
    url: https://outcomeschool.com/blog/decoding-instructgpt
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Reinforcement Learning from Human Feedback (RLHF)
    url: https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Training language models to follow instructions with human feedback（延伸）
    url: https://arxiv.org/abs/2203.02155
    author: Ouyang et al. (OpenAI)
    published: 2022-03-04
  - title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model（延伸）
    url: https://arxiv.org/abs/2305.18290
    author: Rafailov et al.
    published: 2023-05-29
related: [llm-internals-12, llm-internals-09]
updated: 2026-09-28
---

## 一句话答案

> 三个阶段的区别是**监督信号从哪来**：pre-training 用文本自身做下一 token 预测（自监督，正例是“下一个真实 token”），产出只会续写的 base model；SFT 用人工写的 (指令, 回答) 对做监督，loss 只算在回答 token 上，产出会听指令但不分好坏的助手；preference optimisation 用“A 比 B 好”的**比较**信号，先训 reward model 再用 PPO 最大化分数并加 KL 约束（RLHF），或把偏好对直接写成一个 logistic 分类损失（DPO）。
> 三者是串行而非替代：base 提供能力，SFT 提供接口，偏好优化提供取舍标准。监督信号的数据量从前到后骤降（$10^{12}$ 级 token → $10^4$ 条示范；偏好对虽在 $10^5$ 量级，每一对只贡献一个二元比较信号），增量算力却很小：InstructGPT 口径下 SFT 只占 GPT-3 预训练的 0.135%，RLHF 阶段（PPO-ptx 口径）占 1.6%。

## 面试官在考什么

- 能否用「监督信号」这条主线把三个阶段切开，而不是背 pipeline 名称：自监督的下一 token 预测 / 人工示范的正例 / 人工比较的相对偏好。
- 知不知道 SFT 的 loss 只算在回答 token 上，以及为什么必须 mask 掉 prompt；这是最能区分“调过模型”和“看过文章”的细节。
- 能否写出 RLHF 的目标函数并说清 KL 项的机制：参考模型是锚点，限制策略移动半径，从而限制它利用 reward model 分布外误差的程度。
- 是否理解 DPO 与 RLHF 的数学关系（隐式 reward、闭式最优策略），以及取舍在哪里：更简单稳定，但仍是离线数据、仍需偏好标注。
- 能否给量级并交代口径：参数比、算力占比、偏好对数量，以及哪些数字是“论文转述”而不是可复算的。

常见错误答案：

- 把 preference optimisation 说成“再用分数做一次监督微调”。它优化的是**相对偏好**（pair 上的 margin），不是拟合某一条“正确答案”。
- 说 KL 项是为了“防止过拟合”或“让训练更稳”。它约束的是策略相对 SFT 分布能走多远，机制是限制 reward hacking 的可利用空间。
- 认为 SFT 数据越多越好：InstructGPT 只用了约 13k 条示范就完成格式对齐，覆盖度与一致性比条数更关键。

## 原理与推导

### 1. 一张主表：三个阶段各自回答什么

| 阶段 | 目标函数 | 数据形式 | 典型规模 | 算力占比 | 产出 |
| --- | --- | --- | --- | --- | --- |
| pre-training | 全部位置的下一 token 交叉熵 | 无标注文本，标签是文本自身右移一位 | $10^{12}$–$10^{13}$ token | 100%（GPT-3 175B：3,640 petaflop/s-days） | base model：会续写，不会听指令 |
| SFT | 只在回答 token 上的交叉熵 | 人工写的 (指令, 回答) 对 | $10^4$–$10^5$ 条（InstructGPT 约 13k prompt） | 0.135%（4.9 petaflop/s-days） | 会按助手格式作答的 SFT model |
| RLHF（PPO） | reward model 分数 $-\ \beta\cdot\mathrm{KL}$ | 偏好对 $(x,y_w,y_l)$，标注成本是“排序” | RM 数据 33k prompt × $K{=}4\!\sim\!9$ | 1.6%（PPO-ptx 口径 60 petaflop/s-days） | 对齐后的 policy，外加一个 RM |
| DPO | 隐式 reward 差的 logistic 损失 | 同一批离线偏好对，不采样 | 同上 | 与一次 SFT 同量级 | 对齐后的 policy，无独立 RM |

三个问题分别对应：**目标函数**决定模型被推向什么，**数据形式**决定信号里含多少信息，**解决什么问题**决定它在流水线里的位置。pre-training 解决“什么都不知道”，SFT 解决“知道但不用”，偏好优化解决“用了但不知道轻重”。

### 2. pre-training：自监督的下一 token 预测

$$\mathcal{L}_{\text{PT}}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}_{\text{pre}}}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

- **自监督的含义**：数据不需要人工标注，序列右移一位就是标签，所以语料规模可以到万亿 token 级。这一步把词法、句法、事实、世界知识和一部分推理能力压进权重。
- **base model 的行为特征**：它建模的是“互联网文本的下一个 token”，不是“助手该怎么回答”。给它 `Explain the moon landing to a 6 year old in a few sentences.`，它很可能再续写几条同类指令，因为这种句式在网页上经常成组出现——这是完全合理的下一 token 预测，也是完全没用的回答。
- **与 SFT 的形式关系**：目标函数是同一种交叉熵，差别在 mask 位置（PT 覆盖全部位置）和数据分布（PT 是网页/代码/书籍，SFT 是示范回答）。

### 3. SFT：把 loss 移到回答上

$$\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{\text{SFT}}}\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,y_{<t})$$

求和只覆盖回答 $y$ 的位置，指令 $x$ 只作为条件。实现上就是把 prompt 段的 label 设成 `-100`（PyTorch 的 `ignore_index`）不计损失。Llama 3 的技术报告把这个写法明确写成了 "standard cross entropy loss on the target tokens (while masking loss on prompt tokens)"。

**为什么必须 mask**：prompt 是用户分布，回答是助手分布。对 prompt 算交叉熵等于顺带训练模型生成用户风格的提问，这部分梯度对目标任务没有收益，还会稀释回答部分的信号。量级上：prompt 150 token、回答 50 token 时，mask 后只有 $50/200=25\%$ 的位置产生梯度。

SFT 实际教到的东西：格式（markdown、列表）、语气、长度习惯、拒答模板、工具调用格式、思考与答案的分段。InstructGPT 的做法是约 13k 条示范、训 2 个 epoch，并混入 10% 预训练数据（论文给的理由是对后续 PPO 训练有帮助）。

这个阶段**质量与多样性比数量重要**：示范太少（几千条以下，且任务类型单一）模型只会一种腔调；示范太单一（全部出自同一批标注者的措辞习惯）输出会高度趋同，这个偏差还会被后面的偏好数据继承。真正要覆盖的是任务类型、输出格式和拒答边界，而不是把同一种问答复制十万条。

**为什么 SFT 不够**：

- **只有正例**：交叉熵对每条示范的每个 token 一律“提高概率”，它不知道这条是“最优”还是“勉强可用”，更表达不了“两个都能用，但 A 更好”。
- **示范分布窄**：写比选贵得多。InstructGPT 的示范只来自约 40 名承包商，风格与覆盖面天然受限，也无法为每个 prompt 都写一条。
- **结构不匹配**：偏好是**序列级、成对**的信号，需要以 pair 为输入的损失；token 级交叉熵没有这种结构，所以格式和风格能被 SFT 对齐，“取舍标准”不能。

### 4. 偏好优化（一）：RLHF 三段式

**4.1 把排序变成训练对。** 用 SFT 模型对同一 prompt 采样 $K=4\!\sim\!9$ 条回答，让标注者排序（只给顺序，不给绝对分数）。一条 $K$ 的排序能拆出 $\binom{K}{2}$ 个 $(y_w,y_l)$ 对：$K=4$ 得 6 对，$K=9$ 得 36 对，并列样本丢弃。排序比写答案容易得多，一个人的一次排序就产出几十个训练对。InstructGPT 的 RM 数据集是 33k prompt，按 $K$ 的上下界算，pair 总数约 $2.0\times10^5$ 到 $1.2\times10^6$。

**4.2 reward model（Bradley-Terry）。** 在 SFT 模型上去掉最后的 unembedding 层，换一个输出标量的头：

$$\mathcal{L}_{\text{RM}}(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]$$

- 只有**分数差**有辨识性：给所有 $r$ 加上同一常数不改变损失，所以 RM 的绝对值没有量纲意义。
- 数值感受：$r_w-r_l=1$ 时 $\sigma=0.7311$，损失 0.3133；差 3 时损失 0.0486；打平时损失 0.6931。
- 规模选择的理由：InstructGPT 用 6B RM。不是 175B 训不出来，而是 175B RM 训练更不稳定，且 PPO 阶段要把 RM 常驻显存，代价高。
- 工程细节：同一个 prompt 的 $\binom{K}{2}$ 个 pair 放进同一个 batch element，一次前向得到 $K$ 个分数（省掉 $\binom{K}{2}$ 次前向），且只训 1 个 epoch——多 epoch 会迅速过拟合。

**4.3 PPO + KL。** 把 LLM 当策略，最大化

$$\max_\theta\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\Big[r_\phi(x,y)-\beta\,\mathrm{KL}\big(\pi_\theta(y\mid x)\,\|\,\pi_{ref}(y\mid x)\big)\Big]$$

$\pi_{ref}$ 是冻住的 SFT 模型。实现上 KL 是逐 token 折进 reward 的：$r_t=r_\phi(x,y)\cdot\mathbb{1}[t=T]-\beta\log\frac{\pi_\theta(y_t\mid\cdot)}{\pi_{ref}(y_t\mid\cdot)}$，即“整条序列的 KL 之和”这个口径。

**KL 项的作用**：reward model 只是人类偏好的代理，它只在偏好数据覆盖的分布内准确。对分布外文本（越长越自信、堆砌“我很乐意帮忙”、无意义的长答案、过度拒绝）它会给出乐观的错误高分。策略一旦撞上这个方向就会一路放大它（reward hacking），最终分布崩塌成能骗过 RM 的怪文本。KL 把策略锚在 SFT 分布附近，限制移动半径，从而限制它能吃到多少 RM 的外推误差。它约束的是探索半径，不是泛化意义上的正则。

**β 的量级**：InstructGPT 用 $\beta=0.02$，消融显示 0.01–0.02 最好；$\beta=0$（无约束）和 $\beta=2$（约束过强）都明显变差。

**代价**：PPO 同时要驻留 4 个模型——policy、value/critic（与 policy 同量级）、reward model、reference model。四份权重的显存问题见 [[llm-internals-02]] 与 [[llm-internals-03]]。InstructGPT 的 PPO-ptx 跑 256k episodes（约 31k 唯一 prompt），batch 512、minibatch 64。

**对齐税**：RLHF 会挤掉一部分预训练能力（公开 NLP 数据集上的回退）。PPO-ptx 在 PPO 梯度里混入预训练梯度（每个 minibatch 用 8 倍于 RL episodes 的预训练样本，系数 $\gamma=27.8$），把大部分回退补回来，代价是训练时间翻倍。

### 5. 偏好优化（二）：DPO

RLHF 的目标是 KL 约束下的奖励最大化，这个问题的**最优策略有闭式解**，于是可以把 reward 消掉，直接在策略上做分类。

**第一步**，固定 $x$，对每个 $y$ 求带归一化约束的驻点，得到

$$\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{ref}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big),\qquad Z(x)=\sum_{y}\pi_{ref}(y\mid x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)$$

直觉：$\beta\to 0$ 时趋向“只挑 reward 最高的回答”，$\beta\to\infty$ 时退化成 $\pi_{ref}$，即 $\beta$ 就是“允许偏离参考模型多少”的旋钮。

**第二步**，反解出 reward：

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}+\beta\log Z(x)$$

**第三步**，代回 Bradley-Terry。因为 $Z(x)$ 只依赖 $x$，在差值 $r_w-r_l$ 里被消掉：

$$p(y_w\succ y_l\mid x)=\sigma\!\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\Big)$$

**第四步**，取负对数就是 DPO 损失：

$$\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\Big)\Big]$$

- **隐式 reward**：把 $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$ 定义为隐式 reward，DPO 损失就是把 $\hat r_\theta$ 塞进 RM 的同一套 BT 损失。策略本身就是 reward model，这是它最漂亮的地方。
- **梯度**：$\nabla_\theta\mathcal{L}=-\beta\,\mathbb{E}\big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\big(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\big)\big]$。权重是“当前把顺序排错的概率”：已经排对时权重趋近 0（没有梯度），排错时权重很大（重点纠正），方向上提高 preferred、压低 rejected。不需要 reward model，也不需要在线 rollout。
- **工程差别**：模型数从 4 降到 2（policy + 冻住的 reference），没有 value 估计、没有 clip 比率、没有 rollout，超参数只剩 $\beta$ 和学习率。Meta 的公开技术报告就是这条路线的代表：Llama 3 的 post-training 用 reward model → 拒绝采样 → SFT → DPO 多轮迭代，并明确说不用更复杂的 RL 算法，因为它们“更不稳定、更难规模化”。注意 DPO 并不消灭 reward model，Llama 3 里 RM 还在，用来给拒绝采样打分、筛出更好的 SFT 数据。
- **取舍**：① 仍然需要偏好数据（人工或 AI 标注），成本只是从 RL 循环挪到了数据侧；② 离线——pair 是 $\pi_{ref}$ 采样并被人标注过的，策略没有探索，发现不了“比示范更好”的答案；③ 损失只约束两个 log 比的**差**，对两条回答概率同时下降不敏感（把 $\log\pi_\theta(y_w\mid x)$ 和 $\log\pi_\theta(y_l\mid x)$ 同加一个常数，loss 完全不变），所以要看 chosen 的 log 概率与隐式 reward 的 margin，不能只看 loss；④ 偏好数据有限且有偏，模型会过拟合表层特征（长度、格式、讨好），这是 DPO 版的 reward hacking。

### 6. GRPO：去掉 value network 的 RL 变体

GRPO 保留 KL 约束与 PPO 式 clip 比率，但把优势估计换掉：同一 prompt 采样 $G$ 条回答，得分 $\{r_1,\dots,r_G\}$，用组内标准化分数做优势

$$A_i=\frac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$$

组内均值就是 baseline，于是**不需要 value network**，比 PPO 少一个与 policy 同量级的模型，显存和工程复杂度各降一档（DeepSeek-R1 的目标函数里仍保留到 reference 的 KL 惩罚）。

它在推理模型上流行的原因有两个：长 CoT 的 value 估计本来就难（一条轨迹几千 token、奖励稀疏且只在末尾），组内相对优势不需要学 value；数学/代码这类有可验证答案的任务可以直接用规则奖励（答案正确性 + 格式）替代神经 reward model，DeepSeek-R1-Zero 就没用 neural RM，又省掉一个模型和一轮 RM 训练。

### 7. 串起来看

- **监督信号**：文本自身 → 人工示范（正例）→ 人工比较（相对偏好）。
- **成本结构**：token 几乎免费但要万亿级 → 示范最贵在“写” → 偏好对贵在“标注一致性”，但比写便宜，且一个排序能翻出几十个 pair。
- **能力边界**：base 什么都能续但不会听 → SFT 会听但不分轻重 → 偏好优化分轻重。
- **不要跳步**：没有 SFT，采样出来的回答离助手分布太远，标注者面对的是乱续写，比较数据本身没有信息量，RM 学不到有用信号，DPO 的 reference 也是个坏锚点。

## 数值与代码验证

以下数字全部按公开口径复算过，口径不同的地方单独标注。

| 量 | 计算 | 结果 | 口径说明 |
| --- | --- | --- | --- |
| 模型参数比 | $175\text{B}/1.3\text{B}$ | $134.6\approx135$ 倍 | 论文 §1 的主要发现写 "over 100x fewer"（摘要作 "100x fewer"），取的是下界；精确比值 134.6（≈135）倍 |
| SFT 算力占比 | $4.9/3640$ | $0.135\%$ | 分母是 GPT-3 预训练的 3,640 petaflop/s-days |
| PPO-ptx 算力占比 | $60/3640$ | $1.65\%\approx1.6\%$ | 与源文“约 1.6%”一致，按原文 petaflop/s-days 口径复算确认 |
| SFT + PPO 合计 | $64.9/3640$ | $1.78\%$ | 整个对齐流程不到预训练的 2% |
| 一个排序的 pair 数 | $\binom{4}{2},\binom{9}{2}$ | 6 对，36 对 | 源文的 $K=4$、$K=9$ 口径 |
| RM 单 batch 上限 | $64\times\binom{9}{2}$ | 2,304 对 | 论文附录：batch 含 64 个 prompt |
| BT 损失 | $\sigma(1),\sigma(3),\sigma(0)$ | 损失 0.3133 / 0.0486 / 0.6931 | 差值越大损失越小 |
| 幻觉率下降 | $21\%/41\%$ | $0.512\approx$ 一半 | closed-domain 任务，InstructGPT vs GPT-3 |
| 毒性输出 | 约少 25% | — | 要求“礼貌作答”时的口径 |
| SFT mask 后的有效位置 | $50/200$ | $25\%$ | prompt 150 + 回答 50 的举例 |

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)                                    # 固定随机 logits，使输出可复现

# 1) SFT：只在回答 token 上算交叉熵，prompt 段 label = -100
logits = torch.randn(1, 6, 10)                          # (B, T, V)
labels = torch.tensor([[-100, -100, -100, 3, 5, 2]])    # 前 3 个位置是 prompt
sft = F.cross_entropy(logits.view(-1, 10), labels.view(-1), ignore_index=-100)
print("sft loss", round(sft.item(), 4))

# 2) reward model：Bradley-Terry 成对损失，只看分数差
bt = lambda gap: -F.logsigmoid(torch.tensor(gap)).item()
print("BT", [(g, round(bt(g), 4)) for g in (1.0, 3.0, 0.0)])

# 3) DPO：beta 乘 log 比之差，与上面的 BT 损失同形
beta = 0.1
lw, ref_w = -18.0, -20.0     # policy / reference 给 preferred 回答的 log 概率
ll, ref_l = -25.0, -22.0     # 给 rejected 回答的 log 概率
margin = beta * ((lw - ref_w) - (ll - ref_l))
print("dpo margin", round(margin, 4), "loss", round(bt(margin), 4))
shift = beta * (((lw + 3) - ref_w) - ((ll + 3) - ref_l))  # 两条同时 +3
print("shifted margin", round(shift, 4))                  # 不变：只约束差

# 4) GRPO：组内相对优势，替代 value network
r = torch.tensor([1.0, 0.0, 2.0, 1.0])
adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
print("grpo advantage", [round(x, 3) for x in adv.tolist()])
```

```text
sft loss 2.6604
BT [(1.0, 0.3133), (3.0, 0.0486), (0.0, 0.6931)]
dpo margin 0.5 loss 0.4741
shifted margin 0.5
grpo advantage [0.0, -1.414, 1.414, 0.0]
```

第三段代码验证的是 DPO 的“只约束差”性质：两条回答的 log 概率同时加 3，margin 与 loss 一字不变。这也提醒实践里必须另外监控 chosen 回答的绝对 log 概率。

## 常见追问

- **追问**：为什么不能让标注者直接打分，非要用排序？
  - 要点：绝对分数在不同标注者之间没有可比性（有人宽松有人严格），一致性和可复现性差；比较的噪声小得多。而且 BT 模型只依赖分数差，天然匹配“只有顺序、没有绝对分”的数据。成本上，一次 $K=9$ 的排序翻出 36 个 pair。
- **追问**：KL 项拿掉会怎样？
  - 要点：策略会朝 RM 的外推误差方向狂奔——越长越自信、堆砌套话、过度拒绝，分布逐渐崩塌成骗过 RM 的文本。InstructGPT 的消融里 $\beta=0$ 明显变差，$\beta=2$ 也差，最优在 0.01–0.02。KL 的作用是给探索设半径。
- **追问**：DPO 和 RLHF 是等价的吗？
  - 要点：数学上 DPO 是 RLHF 目标的重参数化——KL 约束下的最优策略与 reward 有闭式关系，反解出的隐式 reward 代回 BT 就得到 DPO 损失，所以二者共享同一个最优解族。差异在数据分布：RLHF 在线采样、DPO 用固定离线 pair，因此 DPO 不具备 RL 的探索能力，“等价”只在同一份偏好数据的覆盖范围内成立。
- **追问**：既然 DPO 更简单，为什么还要 PPO/GRPO？
  - 要点：在线 RL 能越过示范与离线 pair 的天花板，尤其在奖励可验证的推理任务上（数学、代码）收益明显；DPO 只能在给定 pair 上重排概率，且容易过拟合偏好数据的表层特征。反过来，通用对话对齐里 DPO 的性价比通常更高，这也是 Llama 3 的选择。
- **追问**：对齐税从哪来，怎么缓解？
  - 要点：post-training 的数据分布很窄（助手风格），梯度把权重拉离预训练分布，通用能力回退，可以看作轻量版的灾难性遗忘。InstructGPT 的 PPO-ptx 在 RL 梯度里混入预训练梯度（8 倍样本、$\gamma=27.8$）补回大部分回退，代价是训练时间翻倍；通用做法还有混入通用指令数据、限制训练步数、用 hold-out 通用评测集持续监控。
- **追问**：reward model 容易怎么坏？
  - 要点：① 过拟合——同一 prompt 的多个回答高度相关，若把 $\binom{K}{2}$ 个 pair 当独立样本打散，一个 completion 会被 $K-1$ 个梯度更新反复推，InstructGPT 的对策是把同一 prompt 的所有 pair 放一个 batch element 且只训 1 个 epoch；② 继承标注偏差——标注指南含糊或标注者意见分裂时，RM 学到互相矛盾的信号，再被策略放大成 reward hacking；③ 分布漂移——策略每更新一轮，RM 面对的输入分布就旧一分，所以要迭代收集新偏好数据重训 RM。

## 公司变体

- **Meta**：偏工程实现与流水线决策。公开的 Llama 3 技术报告把 post-training 写成 reward model → 拒绝采样 → SFT → DPO 的多轮迭代，并明确说不用更复杂的 RL 算法（更不稳定、更难规模化），SFT 部分写明用交叉熵且 mask 掉 prompt 上的 loss。因此这家公司更可能问“给定算力和数据预算，你怎么排这三段”“为什么在这个规模上用 DPO 而不是 PPO”“偏好数据从哪来、怎么迭代”“RM 在 DPO 流程里还留着做什么”；因为有大模型推理基础设施背景，也会追问 PPO 阶段四个模型的显存与吞吐代价。原理层面一般停在“KL 为什么必要”“DPO 与 RLHF 的关系”，不会要求手推 PPO 的 clip 目标。
- **Scale AI**：这家公司的业务就是数据标注与评估——InstructGPT 论文里写明标注团队是“通过 Upwork 和 ScaleAI 招募的约 40 名承包商”。问法因此偏数据与评测工程：标注指南怎么写才能让“哪个更好”可复现、排序如何转成 pair（$K=4\!\sim\!9$ 与 $\binom{K}{2}$）、标注者一致性怎么衡量、标注偏差如何不被 reward model 放大、怎么用 hold-out 人工评测早发现 reward hacking。也会反过来问“SFT 数据或偏好数据质量差时，整条流水线会具体坏在哪一步”。
- 两家的共同点：不会满足于“三个阶段分别是什么”的复述，一定会追一个具体数字或一个取舍判断。

## 相关题目

- [[llm-internals-09]]：Chinchilla 缩放定律回答的是预训练阶段参数与 token 怎么配比；偏好优化阶段是另一个配比问题（13k 示范 + 约 1.6% 算力换来可用性）。两题连着看才是完整的“算力该花在哪”。
- [[llm-internals-12]]：temperature / top-p 是在偏好优化之后的分布上做取舍；SFT 改的是分布形状，偏好优化移动的是分布的质量重心，采样策略决定从这个分布里取哪一段。
- [[llm-internals-02]] 与 [[llm-internals-03]]：PPO 要同时驻留 policy、value、reward、reference 四个模型，KV cache 与 GQA 的显存账决定了这套流程能不能放进同一批卡。

## 参考资料与归属

- [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt) — Amit Shekhar (Outcome School)，2026-08-13。三阶段流程、排序转 pair、KL 惩罚、对齐税与 PPO-ptx 的通俗串讲。
- [Reinforcement Learning from Human Feedback (RLHF)](https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf) — Amit Shekhar (Outcome School)，2026-05-14。RLHF 三段式、四个模型、$\beta$ 的取值范围与常见的工程坑。
- [Training language models to follow instructions with human feedback（延伸）](https://arxiv.org/abs/2203.02155) — Ouyang et al. (OpenAI)，2022-03-04。InstructGPT 的全部具体数字来源：3,640 / 4.9 / 60 petaflop/s-days、13k / 33k / 31k、$K=4\!\sim\!9$ 与 $\binom{K}{2}$、6B RM、$\beta=0.02$、$\gamma=27.8$、21% vs 41%、毒性少 25%，以及文中第 3、4、6 节的推导细节。
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model（延伸）](https://arxiv.org/abs/2305.18290) — Rafailov et al.，2023-05-29。DPO 的闭式最优策略、隐式 reward 与 logistic 损失形式。
- **延伸来源说明**：第 6 节的 GRPO 组内相对优势与“可验证奖励替代 neural reward model”来自公开文献（[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)）；公司变体一节里 Llama 3 的 post-training 路线（RM → 拒绝采样 → SFT → DPO、SFT 对 prompt 段 mask）来自 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)。这两处不在上面四条来源内，正文已按原文口径标注。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
