---
type: question
id: finetuning-10
topic: 微调、后训练与对齐
order: 10
question: 什么是 RLVR（RL with verifiable rewards）？它在哪些方面胜过学到的 reward model？
question_en: What is RLVR (RL with verifiable rewards) and where does it beat a learned reward model?
asked_at: [智谱 AI, 阿里巴巴, Sarvam AI, Scale AI]
level: 高阶
tags: [rlvr, verifier, reward-model, 推理]
sources:
  - title: Group Relative Policy Optimization (GRPO)
    url: https://outcomeschool.com/blog/group-relative-policy-optimization-grpo
    author: Amit Shekhar (Outcome School)
    published: 
  - title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（延伸）
    url: https://arxiv.org/abs/2501.12948
    author: DeepSeek-AI
    published: 2025-01-22
  - title: Tulu 3: Pushing Frontiers in Open Language Model Post-Training（延伸）
    url: https://arxiv.org/abs/2411.15124
    author: Lambert et al.
    published: 2024-11-22
related: [finetuning-03, finetuning-11, finetuning-01, inference-serving-09]
updated: 2026-09-28
---

## 一句话答案

> RLVR 是把 RLHF 目标里的「学出来的 reward model」换成**可程序化验证的奖励函数**：数学题精确比对答案、代码跑单元测试、约束与格式用规则校验、编译器或求解器给出判定。它胜在**精确、不可被优化器在分布外刷分、不需要人类偏好标注、奖励确定可复现**，而且奖励虽然稀疏，却能用组内相对优势（GRPO）变成稳定梯度。代价是只覆盖可验证的任务，且验证器本身会成为新的被钻空子对象。

## 面试官在考什么

- **概念边界**：能不能分清 reward model（学出来的偏好打分器）、value model（PPO 的 critic）、verifier（确定性判定函数）三个东西。这是本题最常见的混淆点。
- **能否逐条论证「为什么更好」**，而不是背「RLVR 更省」这种结论：精确性、标注成本、可复现性、信号转梯度、样本可自动化构造，每一条都要给出机制。
- **是否知道它不万能**：可验证域的边界在哪、验证器如何被钻空子、窄域优化导致的能力偏移、组内奖励全同时梯度为 0。
- **有没有工程视角**：验证器分层、奖励塑形、难度课程与动态采样、与 SFT/DPO 的配比、上线前的回归评测。
- 区分度在于：初级候选人把它答成「GRPO 的另一种叫法」，高阶候选人能把它答成「**一次奖励来源的替换，以及由此带来的能力与风险的同时位移**」。

常见错误答案：

- **「RLVR 就是 GRPO，去掉 value network 的那个」**——把奖励来源和优化算法混为一谈。RLVR 描述的是奖励从哪来（Tulu 3 用的是 PPO，同样能做 RLVR）；GRPO 描述的是优势怎么估（也可以配学出来的 reward model）。
- **「可验证 = 正确」**——验证器有 bug、测试用例覆盖不全、答案不唯一时，模型学到的是「通过验证器」而不是「解决问题」，这正是 [[finetuning-11]] 里 reward hacking 的另一种形态。

## 原理与推导

### 三个角色先分清

| 角色 | 输入 | 输出 | 是否需要训练 | 是否可被优化器利用 |
| --- | --- | --- | --- | --- |
| reward model（RM） | prompt + 完整回答 | 一个标量偏好分 | 需要，用人类/AI 偏好对训练 | 可以：它是个可微代理，分布外会误判 |
| value model / critic | prompt + 前缀 | 该前缀的期望回报 | 需要，与策略同规模 | 只影响优势估计的方差与偏差 |
| verifier（RLVR 用） | prompt + 完整回答 | 规则判定（常为 0/1） | 不需要，是人写的函数 | 可以，但只能通过验证器本身的漏洞 |

Tulu 3 的 RLVR 目标与标准 KL 约束 RLHF 完全同形，只是把 $r_\phi$ 换成 $v$：

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\Big[\,v(x,y)-\beta\,\mathrm{KL}\big[\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\big]\Big]$$

$$v(x,y)=\begin{cases}\alpha & \text{答案通过验证器}\\ 0 & \text{否则}\end{cases}$$

要点：目标函数形式没变，**变的是奖励的信息来源**。KL 项与参考模型的作用也一字未改——它防的正是策略为了刷分而漂移，链接到 [[finetuning-01]]。

### 为什么「精确奖励」在优化下是结构性优势

设 rater 对同一回答的偏好存在真实随机性，第 $i$ 个 rater 的分数为 $r_i$，则

$$r_i=\mu(x,y)+\epsilon_i,\qquad \mathbb{E}[\epsilon_i]=0,\quad \mathrm{Var}(\epsilon_i)=\sigma_{\mathrm{human}}^2,\quad \mathrm{Cov}(\epsilon_i,\epsilon_j)=\rho\,\sigma_{\mathrm{human}}^2$$

用 $K$ 个 rater 平均训练 RM，RM 的噪声下限是

$$\mathrm{Var}\Big(\tfrac{1}{K}\sum_i r_i\Big)=\sigma_{\mathrm{human}}^2\Big[\rho+\frac{1-\rho}{K}\Big]$$

含义有两层：**多雇标注员只能压掉独立噪声 $1/K$ 那一项，压不掉系统性分歧 $\rho\sigma^2$**；而 RLVR 里 $\sigma_{\mathrm{human}}=0$，这一项直接消失。所以 RLVR 不是「省点钱」，而是把一种策略梯度无法消除的偏差从奖励里拿掉了。

### 不可欺骗性的量化口径

设验证器的假阴性率为 $\epsilon_{fn}$（真解被判错）、假阳性率为 $\epsilon_{fp}$（错解被判对）。单条回答的期望奖励可以精确写成

$$\mathbb{E}[v]=\alpha\big[(1-\epsilon_{fn})\Pr[\text{真解}]\big]+\alpha\big[\epsilon_{fp}\Pr[\text{未真解}]\big]\le\alpha\Pr[\text{真解}]+\alpha\,\epsilon_{fp}$$

不等号右端的第二项说明：**与「真正解题」无关的那部分奖励上界是 $\alpha\,\epsilon_{fp}$**，它是一个可以用隐藏测试与人工抽检直接测出来的量。要诚实补一句：$\epsilon_{fp}$ 本身会随策略输出分布外移而变大（过拟合可见测试用例就是这种情形）；而学出来的 RM 除了这一点，还要额外承担自身在分布外的泛化误差 $\delta_{\mathrm{OOD}}$，这一项没有可写死的上界。所以面试里可以这样收敛：**RLVR 消灭的是奖励模型的分布外误判通道，不是消灭 reward hacking**——后者的载体从 RM 转移到了验证器（[[finetuning-11]]）。

### 稀疏奖励怎么变成梯度

可验证奖励通常是 0/1，整条序列共享一个标量，credit assignment 天然稀疏。两条常用路径：

1. **组内相对优势**：同一 prompt 采样 $G$ 条回答，用组内均值做 baseline，标准差归一化：

$$\hat{A}_i=\frac{r_i-\mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})+\varepsilon}$$

以 $G=4$、奖励为 $(1,0,1,0)$ 为例：$\mathrm{mean}=0.5$，$\mathrm{std}=0.5$，归一化后优势恰为 $(+1,-1,+1,-1)$。这就是「不需要 value model 也能有优势信号」的来源，展开见 [[finetuning-03]]。DeepSeek-R1 走的正是这条路：纯 RL、不依赖人类标注的推理轨迹，就能在数学、代码竞赛与 STEM 这类可验证任务上激励出推理能力，并观察到自我反思、验证、策略自适应等推理模式的涌现（均为论文摘要口径）。若全部同奖励（全对或全错），$\mathrm{std}=0$、分子也为 0，该组梯度为 0。

2. **PPO + 把 value model 从 RM 初始化**：Tulu 3 的 RLVR 走的是这条路，用 PPO 优化，并把 value model 的初始化接到一个通用 RM 上（沿用 InstructGPT 一系的做法）。

### 奖励塑形的必要性

纯 0/1 奖励在工程上不够用，因为「格式崩坏但答案蒙对」和「格式正确但算错」应当给不同信号。Tulu 3 的做法是把奖励拆开：$\alpha=10$（通过验证器给 10 分）、非 EOS 结尾的回答给 $-10$（惩罚不完整生成），并把 $\beta$ 在 $\{0.01, 0.03, 0.05, 0.1\}$ 里做消融。注意 $-10$ 与 $+10$ 同量级，意味着**生成不完整被视作与答案错误同等严重的失败**。

## 数值与代码验证

### 组内奖励全同的概率（自己算）

$G$ 条回答全部同奖励的概率是 $p^{G}+(1-p)^{G}$，其中 $p$ 是该题当前策略的单次通过率：

| 单次通过率 $p$ | $G=8$：全对 | $G=8$：全错 | $G=8$ 无效组占比 | $G=16$ 无效组占比 |
| --- | --- | --- | --- | --- |
| 0.05 | $3.9\times10^{-11}$ | 0.6634 | **66.34%** | 44.01% |
| 0.10 | $1.0\times10^{-8}$ | 0.4305 | **43.05%** | 18.53% |
| 0.30 | $6.6\times10^{-5}$ | 0.0576 | 5.77% | 0.33% |
| 0.50 | 0.0039 | 0.0039 | 0.78% | 0.003% |
| 0.70 | 0.0576 | $6.6\times10^{-5}$ | 5.77% | 0.33% |
| 0.90 | 0.4305 | $1.0\times10^{-8}$ | **43.05%** | 18.53% |

口径与假设：奖励是 0/1，$G$ 条回答独立同分布。结论是**有效梯度带宽落在 $p\in[0.3,0.7]$**（$G=8$ 时无效组占比不到 6%）；$p=0.05$ 时 $G=8$ 有 66.3% 的组白采、$G=16$ 也有 44.0%，$p=0.9$ 时对应 43.1% 与 18.5%——越靠近两端，采样算力浪费越大。这直接推出动态采样的必要性：按组内通过率筛题，而不是均匀采样题库。顺带一个常被问到的数：在通过率 $p$ 下，等到第一条正确答案的期望 rollout 数是 $1/p$（$p=0.1$ 时要 10 条，$p=0.01$ 时要 100 条）。

### 显存账：RLVR 省掉了哪一份模型

按「bf16 权重 = 2 bytes/param」的口径，7B 参数每份副本 14 GB：

| 组成 | 每份 | 份数 | 小计 |
| --- | --- | --- | --- |
| policy（训练中） | 14 GB | 1 | 14 GB |
| reference model（KL 用，冻结） | 14 GB | 1 | 14 GB |
| value model / critic | 14 GB | 1 | 14 GB |
| reward model（学出来的 RM） | 14 GB | 1 | 14 GB |
| **合计：RM + PPO（value model）** | | 4 | **56 GB** |
| **合计：RLVR（去掉 RM；PPO 仍留 critic）** | | 3 | **42 GB** |
| **合计：RLVR + GRPO（再去掉 critic）** | | 2 | **28 GB** |

再叠加 policy 自身的优化器状态：Adam 的 fp32 $m,v$ 加 fp32 master weights 是 $3\times4=12$ bytes/param $=84$ GB，全量微调的显存主导项仍是它，换 LoRA（例如 $r=16$、只适配 q/k/v/o 四个投影、$d=4096$、32 层）可训练参数 $16\times(4096+4096)\times32\times4=1.68\times10^7$（占比不到 0.3%），优化器状态降到约 0.2 GB——与本题相关的结论是：**RLVR 省的是「每多一份与策略同规模的冻结模型」，量级在十几 GB 一份，而不是省优化器状态**。若把 GRPO 的 rollout 也计入，采样阶段的 KV cache 按 32 层、8 个 KV head、head_dim 128、bf16 算是 128 KiB/token，8 条并发 4096 token 的回答约占 4 GiB。

### 验证器与 reward model 的往返成本

一次 RM 前向的算力约 $2\times$ 参数量 $\times$ 生成 token 数。用 7B RM 给 1000 条、每条 4096 token 的回答打分：

$$2\times7\times10^{9}\times1000\times4096\approx5.7\times10^{16}\ \text{FLOPs}$$

按 A100 bf16 峰值 312 TFLOPs、30% MFU 折算约 $5.7\times10^{16}/(0.3\times312\times10^{12})\approx610$ 秒。而字符串精确比对、正则约束校验几乎是零算力，代码验证则取决于测试用例本身的执行时间——**验证器的成本从「模型前向」变成了「CPU 任务调度」**，这是 RLVR 的 rollout 流水线要单独设计的原因。

成本结构的差别还带来两个工程结论。① **样本可以流水线式扩张**：题目来自现成题库与单元测试仓库，验证器按题域写一次，新增样本的边际成本基本只有 rollout 算力与验证器执行时间，不存在「再标一批偏好对」的线性人力成本。② **奖励可回归**：验证器不参与训练，同一 $(x,y)$ 的奖励在训练全程恒定；RM 每轮重训都会让分数漂移，跨版本对比要额外控制变量。代价也同样清楚：验证器一旦有 bug，会稳定地把错误信号喂给策略，而且它在训练全程不会自己变好。

### 一个最小可跑的实现骨架

下面只保留 RLVR 相关的关键步骤，重点是奖励侧与组内归一化；PPO/GRPO 的损失调用被抽象成函数。

```python
import re
from statistics import mean, pstdev

def math_verifier(prompt, completion, gold, alpha=10.0):
    """数学题：抽取答案 + 精确比对。抽取失败按错误处理，不给半份奖励。"""
    m = re.search(r"\\boxed\{([^}]*)\}", completion) or re.search(r"answer is\s*([-\d./]+)", completion)
    if not m:
        return 0.0
    pred = m.group(1).strip().replace(" ", "")
    return alpha if pred == gold.strip() else 0.0

def code_verifier(prompt, completion, tests, alpha=10.0, timeout=3.0):
    """代码题：跑可见测试；隐藏测试只在抽检时启用，避免策略过拟合可见用例。"""
    ok = run_tests_in_sandbox(completion, tests, timeout=timeout)  # 返回 bool，需沙箱与超时
    return alpha if ok else 0.0

def group_advantages(rewards, eps=1e-8):
    """GRPO 式组内归一化：全是同一个奖励时 std=0，这里显式返回全 0，不产生梯度。"""
    mu, sd = mean(rewards), pstdev(rewards)
    if sd < eps:
        return [0.0] * len(rewards)
    return [(r - mu) / (sd + eps) for r in rewards]

# 一次训练步（伪代码）
# for prompt, gold in dynamic_sampler:            # 见下方通过率分桶
#     comps   = [policy.generate(prompt) for _ in range(G)]
#     rewards = [math_verifier(prompt, c, gold) for c in comps]
#     advs    = group_advantages(rewards)
#     if all(a == 0.0 for a in advs):
#         continue                                # 跳过退化组，省下反向传播
#     loss = ppo_or_grpo_loss(policy, old_policy, ref_policy, comps, advs, beta=0.03)
#     loss.backward(); optimizer.step()

def bucket_by_pass_rate(rewards_by_prompt, lo=0.3, hi=0.7):
    """动态采样：把题目按历史批次的组内通过率分桶，只从有效带宽里抽题。"""
    warm, hard, easy = [], [], []
    for pid, rs in rewards_by_prompt.items():
        p = sum(1 for r in rs if r > 0) / len(rs)
        (warm if lo <= p <= hi else hard if p < lo else easy).append(pid)
    return warm, hard, easy
```

四点实现细节值得在面试里点出来：**抽取失败的答案必须记为错误**（否则等于给格式崩坏发奖励）；**退化组要跳过**（省掉一次无效的反向）；**通过率分桶要按历史批次滚动更新**，因为策略在训练中会变强，今天的难题明天就进了有效带宽；**RLVR 不是单独一步**——Tulu 3 把它排在偏好微调之后，通用配方是 SFT 冷启动给格式与基础能力 → 偏好数据修风格与安全 → RLVR 在可验证域上拔高。

## 常见追问

- **追问**：RLVR 和 RLAIF 有什么区别？
  - 要点：RLAIF 换的是「谁来当裁判」——用 AI 反馈替代人类反馈，奖励依然是一个学出来或提示出来的**偏好判断**，可以覆盖开放式任务，但同样存在偏好偏差、可欺骗性和分布外漂移。RLVR 换的是「奖励是什么」——用确定性函数替代一切打分器，只覆盖可验证任务，但换来精确与可复现。二者正交，实践中常混用：可验证部分走 RLVR，风格与安全部分走偏好数据。
- **追问**：没有验证器的领域怎么办？
  - 要点：三条退化路径。① **结果奖励模型（ORM）**：对最终答案学一个打分器，噪声大但覆盖广；② **过程奖励模型（PRM）**：对推理的每一步打分，信号更密但标注成本高、且容易奖励「看起来严谨」的废话；③ **人工抽检 + LLM-as-judge**：成本可控但引入了 judge 偏差，必须配 [[rag-04]] 那套 judge 偏差治理。真话是：**在没有可靠验证器的域上做 RL，收益会明显缩水**。
- **追问**：模型「答案蒙对但推理错」，RLVR 会不会奖励它？
  - 要点：会。0/1 奖励只看最终判定，所以要用三种办法兜：**答案唯一性检查**（拒绝答案不唯一或多解的题）、**对推理链做一致性抽检**（换数字重问、要求同一结论在改写题干下复现）、以及**在 held-out 的同类新题上验证泛化**。判断标准不是训练集奖励涨了多少，而是没见过的同类题涨没涨——这正是 Tulu 3 强调 development 与 unseen 两套评测的原因。
- **追问**：训练奖励一直涨，但评测不涨、其他能力还退化，怎么排查？
  - 要点：先分清三类原因。① **验证器错**：假阳性/假阴性、答案抽取正则误匹配；② **过拟合验证器**：可见测试用例被记住——Tulu 3 报告过一个具体形态，IFEval 这类可验证约束在 KL 较大时会被过度优化，出现只堆砌约束词而不回答问题的输出（例如被要求包含某关键词 25 次时，就只重复该词 25 次、完全不回答原问题）；③ **窄域优化带来的能力偏移**：Tulu 3 明确写了 RLVR 能在对应测试集上提升，但**不保证所有评测的平均分提升**，机制是在窄分布上重复优化让策略偏离通用分布，即 [[finetuning-07]] 讲的能力偏移/遗忘。对应动作：抽检高奖励样本、用隐藏测试与人工评测量假阳性率、调小学习率或加大 $\beta$、把约束类奖励降权；缓解能力偏移则靠混入通用指令数据、用 KL 把策略拴在参考模型附近、训练中持续跑多任务评测而不是只盯目标指标。
- **追问**：RLVR 要多少数据？
  - 要点：比直觉少得多。Tulu 3 的验证器 prompt 混合体一共 29,946 条（GSM8K 7,473 条、MATH 7,500 条、可验证指令跟随 14,973 条），但它们被反复训练——论文提到在消融中相当于把 GSM8K 的 7,473 条跑了约 $100000/7473\approx13$ 个 epoch，每轮之间重新打乱，最终模型每隔 40–100 步存 checkpoint 并按开发集挑最好的。对照 LIMA 的口径（约 1000 条精挑数据就足以让 SFT 表现良好，见 [[finetuning-08]]），可以给出同一个结论：**后训练阶段数据质量与验证质量比数量重要**。真正的规模化瓶颈是 rollout 算力与验证器吞吐，不是标注量。另有一个容易被忽略的前提是题库纪律：训练题库要与评测集按 held-out 切分并做污染检测，否则训练奖励和被污染的评测分数会一起虚高，判据也就失效了（评测纪律见 [[rag-04]]）。
- **追问**：KL 系数 $\beta$ 怎么选？
  - 要点：$\beta$ 是「刷训练奖励」与「保持通用能力」之间的旋钮，Tulu 3 的消融网格是 0.01/0.03/0.05/0.1。判据不是训练奖励，而是这些症状指标：回答长度、KL 实际值、拒答率、非目标评测的分数。实践中更稳的估计量是 $r-\log r-1$（$r=\pi_{\mathrm{ref}}/\pi_\theta$，逐样本非负），在 $r=1$ 时为 0，避免朴素 KL 估计在单样本上有负值。

## 公司变体

- **智谱 AI**：公开的 GLM 系列技术材料把 RL/RLHF 阶段的重点放在推理与对齐能力的提升上。面试里更可能顺着「可验证奖励能覆盖到哪些能力域」往下问：数学、代码、工具调用之外，怎样给中文场景里的结构化输出、格式约束、计算校验写验证器；以及验证器覆盖不全时如何用偏好数据补。准备时把验证器设计当成开放题来答。
- **阿里巴巴**：Qwen 系列的公开模型说明中明确使用过基于可验证奖励的 RL 来增强推理（数学与代码），属于工程落地最直接的参照系。面试偏工程实现的可能性更高：采样组大小与并发、rollout 的推理框架与吞吐、动态采样与难度分桶、验证器服务的超时与沙箱、以及训练与推理的一致性（logprob 对不齐会导致 ratio 全部被 clip）。
- **Sarvam AI**：公开工作在开源模型与多语言（尤其印度语言）适配方向，这类场景下可验证奖励天然稀缺——数学与代码能跑验证器，但翻译质量、语言正确性、文化适配很难写成确定性函数。面试角度更可能落在「验证器在低资源语言上如何构造」：先用规则覆盖能覆盖的结构化任务（数值、格式、代码），开放生成部分留给偏好数据或 judge，并对 judge 做分层抽检。
- **Scale AI**：公开定位是数据、评测与 RLHF 服务，对本题的天然切入点是**评测可信度**：验证器的假阳性率与假阴性率怎么测、隐藏测试与抽检怎么设计、题库防污染怎么做、训练奖励与独立评测不一致时如何归因。面试里把「可验证」推到「可审计」这一层会更占优。

这四家公开材料的侧重不同（两家偏自研模型与训练配方，一家偏多语言落地，一家偏数据与评测供给），但四家都没有公开过各自的面试流程细节，以上的区分只基于公开技术材料的方向，不涉及具体轮次或题型。

## 相关题目

- [[finetuning-01]]：RLHF 端到端流程与 KL 惩罚——RLVR 的目标函数与它同形，只换了奖励来源。
- [[finetuning-03]]：GRPO 与「去掉 value network」——RLVR 常用它把稀疏奖励转成优势信号。
- [[finetuning-11]]：reward hacking 与各实验室的应对——RLVR 把被钻空子的对象从 reward model 换成了验证器。
- [[finetuning-07]]：灾难性遗忘——窄域 RL 优化导致能力偏移的机制与缓解。
- [[finetuning-06]]：LoRA / prefix tuning / 全量微调的对比——RLVR 的显存账要在这里对齐口径。
- [[finetuning-12]]：蒸馏——把 RLVR 训出来的能力搬进小模型的常见后续步骤。
- [[inference-serving-09]]：TTFT、TPOT 与 throughput 的权衡——RLVR 的 rollout 是「长输出 × 大并发」的吞吐问题，采样成本要按这套口径估。
- [[rag-04]]：评测纪律与 judge 偏差治理——验证器的可信度要靠同一套方法论来守。
- 专题入口：[[finetuning]]。

## 参考资料与归属

- [Group Relative Policy Optimization (GRPO)](https://outcomeschool.com/blog/group-relative-policy-optimization-grpo)，Amit Shekhar（Outcome School）。组内相对优势、组均值作 baseline、退化组无梯度、奖励函数只对完整回答打分的口径来自本文。
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（延伸）](https://arxiv.org/abs/2501.12948)，DeepSeek-AI，2025-01-22。纯 RL 不需人类标注推理轨迹即可激励推理能力、在数学/代码竞赛/STEM 这类可验证任务上表现突出、以及自我反思与验证等推理模式涌现，均为该论文摘要的口径。
- [Tulu 3: Pushing Frontiers in Open Language Model Post-Training（延伸）](https://arxiv.org/abs/2411.15124)，Lambert et al.，2024-11-22。RLVR 的命名与定义、$v(x,y)\in\{0,\alpha\}$ 与 $\alpha=10$、非 EOS 惩罚 $-10$、$\beta$ 网格、29,946 条可验证 prompt 的构成、约 13 个 epoch 与每 40–100 步存 checkpoint、以及「能提升对应测试集但不保证所有评测平均分提升」和 IFEval 过度优化的实例，均来自该论文。

延伸来源只用于补充公式细节与实证数字（DeepSeek-R1 的结论、Tulu 3 的 RLVR 配方），作业单给出的那篇 GRPO 文章提供了组内相对优势、组均值 baseline 与退化组无梯度的主线口径。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
