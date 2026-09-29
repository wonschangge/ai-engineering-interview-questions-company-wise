---
type: question
id: gdm-11
company: Google DeepMind 与 Google AI
topic: evaluation
order: 11
question: 什么情况下你会选择 Q-learning 而不是 policy gradient，反之又如何？
question_en: When would you choose Q-learning over policy gradients, and vice versa?
asked_at: []
level: 高阶
tags: [强化学习, off-policy, on-policy, 样本效率, 动作空间]
sources:
  - title: Reinforcement Learning: An Introduction（延伸）
    url: http://incompleteideas.net/book/the-book-2nd.html
    author: Sutton & Barto
    published: 2018-01-01
  - title: Human-level control through deep reinforcement learning（延伸）
    url: https://www.nature.com/articles/nature14236
    author: Mnih et al. (DeepMind)
    published: 2015-02-25
  - title: Proximal Policy Optimization Algorithms（延伸）
    url: https://arxiv.org/abs/1707.06347
    author: Schulman et al. (OpenAI)
    published: 2017-07-20
related: [gdm-10, gdm-05, agents-05, coding-31, evaluation-06]
updated: 2026-09-28
---

## 一句话答案

> **判据是三条**：**① 动作空间是否连续/高维**（**连续 → 策略梯度**）；**② 能否复用历史数据**（**能复用 → Q-learning**）；**③ 是否需要随机策略**（**需要 → 策略梯度**）。
> | 维度 | **Q-learning（值方法）** | **策略梯度** |
> | --- | --- | --- |
> | **数据来源** | **off-policy（可复用历史）** | **on-policy（要新样本）** |
> | **动作空间** | **离散、低维** | **连续、高维** |
> | **策略形式** | 确定性（**+ 探索**） | **随机策略（天然）** |
> | **样本效率** | **高** | 低 |
> | **稳定性** | 可能发散（**致命三要素**） | **较稳（但方差大）** |
> | **代表算法** | DQN、Double DQN | REINFORCE、PPO、SAC |
> **读法**：**"动作是否连续"与"数据能否复用"是两个最实用的判据**——**记住这两条就能答对大多数场景**。
> **★ 量化一：off-policy 的本质（本机 5×5 gridworld 实验）**
> **用"均匀随机策略"采集 4000 条转移，然后反复遍历这份固定数据集**：
> | 遍历次数 | **贪心策略步数** | 说明 |
> | --- | --- | --- |
> | 1 | **8** | **已学到最优（8 步 = 曼哈顿距离）** |
> | 30 | **8** | 稳定 |
> **读法**：**只用随机策略产生的数据，Q-learning 就学到了最优策略**——**因为它只需要 $(s,a,r,s')$ 四元组，不要求数据来自当前策略**。**这是 off-policy 的核心价值**：**数据可以来自任何行为策略**（**包括人类演示、旧策略、甚至随机探索**）。
> **★ 量化二：on-policy 的硬约束（同一份数据给 REINFORCE）**
> **用"均匀随机策略"采集 600 条完整轨迹，反复遍历**：
> | 遍历次数 | **贪心策略步数** | 结果 |
> | --- | --- | --- |
> | 1 | **未学会** | —— |
> | 60 | **未学会** | **分布不匹配** |
> **读法**：**同一批"随机策略的轨迹"反复喂给 REINFORCE，学不到最优**——**因为策略改进后，"好动作"的样本在这批老数据里几乎没有**（**on-policy 要求样本来自当前策略**）。
> **★ 量化三：阳性对照（证明上面的失败是"分布"而非"实现"）**
> | 设置 | 结果 |
> | --- | --- |
> | **REINFORCE + 每次新采样** | **第 113 集达标**（**移动均值 < 10**） |
> | 最终贪心策略步数 | **8（最优）** |
> **读法**：**同一算法、同样超参，只把"固定数据"换成"每次新采样"就能学会**——**这证明量化二的失败原因是"分布不匹配"**（**而不是实现 bug**）。**这是本机实验设计的完整性所在**。
> **★ 什么时候选哪个（决策表）**：
> | 场景 | 选择 | 理由 |
> | --- | --- | --- |
> | **离散动作、样本预算紧** | **Q-learning / DQN** | **样本效率高** |
> | **连续控制（机器人、驾驶）** | **策略梯度（DDPG/SAC/PPO）** | **$\arg\max_a Q$ 在连续空间不可行** |
> | **有大量离线数据** | **离线 RL（CQL/IQL）** | **off-policy 才能用** |
> | **需要多模态策略** | **策略梯度** | **随机策略能表达"两种都行"** |
> | **需要稳定训练** | **PPO** | **信赖域约束** |
> | **动作维度极高（如推荐列表）** | **策略梯度 / 结构化方法** | **$Q$ 表不可行** |
> **读法**：**"连续动作"与"离线数据"是最常见的两个决定性因素**——**它们分别排除了 Q-learning 与策略梯度**。
> 一句话判据：**"先问动作空间（连续 → PG）→ 再问数据能否复用（能 → Q-learning）→ 再问是否需要随机策略（需要 → PG）→ 最后看稳定性要求"**。

## 面试官在考什么

- **★ 是否给出"动作空间"这条判据**：**连续/高维动作 → 策略梯度**（**因为 $\arg\max_a Q$ 不可行**）。
- **★ 是否给出"off-policy vs on-policy"**：**能否量化"固定数据能否学习"**（本机：**Q-learning 1 遍就学到 8 步最优，REINFORCE 60 遍学不会**）。
- **★ 是否知道"随机策略"的必要性**：**能否指出"多模态场景"**（**如"向左或向右都行，但不能直行"**）。
- **样本效率**：**能否指出 off-policy 复用数据 → 样本效率高**。
- **稳定性**：**能否指出"致命三要素"（函数逼近 + bootstrapping + off-policy）**。
- **算法谱系**：**能否列举 DQN/Double/Dueling 与 REINFORCE/PPO/SAC**。
- **是否知道 SAC 是"两者结合"**：**actor-critic = 策略 + 值函数**。
- **探索机制**：**Q-learning 用 $\epsilon$-greedy；策略梯度用熵正则**。
- **实验意识**：**能否用"阳性对照"证明结论**（**这是本机实验最值得学的一点**）。
- **诚实**：**承认"两者都有超参敏感问题"**。

**常见错误答案**

- **只说"Q-learning 更简单"**（**没有判据**）。
- **不知道连续动作下 Q-learning 不可行**（**$\arg\max$ 要在连续空间优化**）。
- **把"on-policy"说成"只能用一个样本"**（**其实是一批，但必须是当前策略的**）。
- **不提"随机策略"的必要性**。
- **不知道 actor-critic**（**把两者对立起来**）。
- **忽略稳定性问题**（**DQN 的致命三要素**）。
- **不做对照实验**（**只给结论不给证据**）。
- **认为"策略梯度一定更好"**（**它样本效率低得多**）。

## 原理与推导

### 1. ★ 三条判据

| 判据 | Q-learning | 策略梯度 |
| --- | --- | --- |
| **动作空间** | **离散、可枚举** | **连续、高维** |
| **数据复用** | **能（off-policy）** | **不能（on-policy）** |
| **策略形式** | **确定性**（**+ 探索噪声**） | **随机（概率分布）** |
| **样本效率** | 高 | 低 |
| **方差** | **TD 误差小** | **回报的方差大** |

**读法**：**"$\arg\max_a Q(s,a)$ 在连续空间上是一个优化问题"**——**这是"连续动作排除 Q-learning"的根本原因**（**不是"效果差"，而是"不可行"**）。

### 2. ★ off-policy 与 on-policy（本机实验）

| 方法 | 数据 | 结果 |
| --- | --- | --- |
| **Q-learning** | **固定 4000 条转移**（随机策略采） | **1 遍即学到 8 步最优** |
| **REINFORCE** | **固定 600 条轨迹**（随机策略采） | **60 遍仍未学会** |
| **REINFORCE（对照）** | **每次新采样** | **第 113 集达标，最终 8 步** |

**读法**：**"off-policy 能用任何行为策略的数据"**——**而"on-policy 的梯度只对当前策略的样本无偏"**。**数学原因**：

$$\nabla_\theta J=\mathbb E_{\tau\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a|s)\cdot G\big]$$

**读法**：**期望的分布是 $\pi_\theta$**——**如果样本来自 $\pi_{\text{old}}$，这个期望就是有偏的**（**除非做重要性采样修正，那又会带来方差爆炸**）。

### 3. ★ 稳定性：致命三要素

| 要素 | 含义 | 单独出现 | 三者同时 |
| --- | --- | --- | --- |
| **函数逼近** | 用神经网络表示 $Q$ | 可接受 | —— |
| **Bootstrapping** | 用 $Q$ 的估计更新 $Q$ | 可接受 | **可能发散** |
| **Off-policy** | 用行为策略的数据 | 可接受 | —— |

**读法**：**"三者同时出现会发散"**——**这就是 DQN 需要"目标网络 + 经验回放"的原因**（**目标网络冻结 bootstrapping、回放改变数据分布**）。**而策略梯度天然 on-policy，所以不受这一条影响**（**但它的方差问题需要 baseline/GAE 来缓解**）。

### 4. Actor-Critic：两者不是对立的

| 组件 | 作用 |
| --- | --- |
| **Actor（策略）** | **输出动作分布** |
| **Critic（值函数）** | **估计 $V$ 或 $Q$，给 actor 提供低方差的优势估计** |

**读法**：**"actor-critic 用值函数降低策略梯度的方差"**——**所以现代算法（PPO/SAC/A3C）几乎都是 actor-critic**（**"选哪个"这个问题在实践中变成了"用哪种 actor-critic"**）。

### 5. 探索机制

| 方法 | 探索方式 | 特点 |
| --- | --- | --- |
| **Q-learning** | **$\epsilon$-greedy / Boltzmann** | **显式、可调** |
| **策略梯度** | **策略本身的随机性 + 熵正则** | **内在、可学习** |
| **SAC** | **最大熵目标** | **把探索写进目标函数** |

**读法**：**"熵正则是策略梯度的探索机制"**——**它鼓励策略保持随机性**（**而 Q-learning 要靠外部噪声**）。

### 6. 场景对照

| 场景 | 选择 | 理由 |
| --- | --- | --- |
| **Atari（离散动作）** | DQN 或 PPO | 两者都行，DQN 样本效率更高 |
| **机器人控制（连续）** | **SAC / PPO** | **连续动作** |
| **围棋/棋类** | **MCTS + 值网络** | **巨大离散空间，用搜索** |
| **推荐系统** | **策略梯度 / 离线 RL** | **动作空间巨大** |
| **LLM 对齐（RLHF）** | **PPO / GRPO** | **序列生成 = 巨大动作空间** |
| **有离线数据集** | **离线 RL（CQL/IQL）** | **off-policy** |

**读法**：**"RLHF 用 PPO 而不是 DQN"的原因正是"动作空间是词表（连续/巨大）"**——**这是把经典 RL 与大模型连起来的一句话**（**串 [[agents-05]]**）。

## 数值与代码验证

### 表 1：off-policy 从固定数据学习、on-policy 的失败、阳性对照（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
import random, math
rng=random.Random(0)
N=5; GOAL=(N-1,N-1); ACT=[(-1,0),(1,0),(0,-1),(0,1)]
def step(s,a):
    r,c=s; dr,dc=ACT[a]; nr,nc=r+dr,c+dc
    if not (0<=nr<N and 0<=nc<N): return s,-1.0,False
    if (nr,nc)==GOAL: return (nr,nc),0.0,True
    return (nr,nc),-1.0,False
# 固定数据集 A：转移（供 off-policy 方法）
def collect_transitions(n=4000):
    data=[]; s=(0,0)
    for _ in range(n):
        a=rng.randrange(4); ns,r,done=step(s,a)
        data.append((s,a,r,ns,done)); s=(0,0) if done else ns
    return data
# 固定数据集 B：完整轨迹（供 on-policy 方法，回报有定义）
def collect_trajectories(n_ep=600, limit=200):
    trajs=[]
    for _ in range(n_ep):
        s=(0,0); tr=[]
        for _ in range(limit):
            a=rng.randrange(4); ns,r,done=step(s,a)
            tr.append((s,a,r)); s=ns
            if done: break
        trajs.append(tr)
    return trajs
TR=collect_transitions(); TJ=collect_trajectories()
print('① off-policy：Q-learning 从「固定转移集」学（不要求数据来自当前策略）')
def q_from(data, passes, alpha=0.3, gamma=0.95):
    Q={(r,c):[0.0]*4 for r in range(N) for c in range(N)}
    for _ in range(passes):
        for s,a,r,ns,done in data:
            tgt = r if done else r+gamma*max(Q[ns]); Q[s][a]+=alpha*(tgt-Q[s][a])
    return Q
def greedy(Q, limit=100):
    s=(0,0)
    for k in range(limit):
        if s==GOAL: return k
        a=max(range(4),key=lambda i:Q[s][i]); s,_,d=greedy_step(s,a)
        if d: return k+1
    return None
def greedy_step(s,a):
    r,c=s; dr,dc=ACT[a]; nr,nc=r+dr,c+dc
    if not (0<=nr<N and 0<=nc<N): return s,-1.0,False
    if (nr,nc)==GOAL: return (nr,nc),0.0,True
    return (nr,nc),-1.0,False
print(f'  {"遍历次数":>8} {"贪心策略步数":>12} 说明')
for p in (1,3,10,30):
    g=greedy(q_from(TR,p))
    tag = "**最优（8 步）**" if g == 8 else ("接近" if g and g < 20 else "未学会")
    print(f'  {p:>8} {str(g):>12} {tag}')
print('  读法：**随机策略采的转移 + 反复遍历 -> 学到最优策略（8 步）** -> 这是 off-policy 的核心价值')

print('')
print('② on-policy：REINFORCE 用「同一批完整轨迹」反复训练（回报有定义，但分布不匹配）')
def reinforce_fixed(trajs, passes, lr=0.05, gamma=0.95):
    theta={(r,c):[0.0]*4 for r in range(N) for c in range(N)}; base=0.0
    for _ in range(passes):
        for tr in trajs:
            G=0.0; rets=[]
            for _,_,r in reversed(tr):
                G=r+gamma*G; rets.append(G)
            rets.reverse()
            for (s,a,_),Gt in zip(tr,rets):
                z=theta[s]; m=max(z); e=[math.exp(v-m) for v in z]; Z=sum(e); p=[v/Z for v in e]
                adv=Gt-base
                for k in range(4): theta[s][k]+=lr*adv*((1.0 if k==a else 0.0)-p[k])
                base=0.9*base+0.1*Gt
    return theta
def greedy_theta(th, limit=100):
    s=(0,0)
    for k in range(limit):
        if s==GOAL: return k
        a=max(range(4),key=lambda i:th[s][i]); s,_,d=greedy_step(s,a)
        if d: return k+1
    return None
print(f'  {"遍历次数":>8} {"贪心策略步数":>12} 说明')
for p in (1,5,20,60):
    g=greedy_theta(reinforce_fixed(TJ,p))
    tag2 = "已学会" if g == 8 else ("接近" if g and g < 20 else "**未学会（分布不匹配）**")
    print(f'  {p:>8} {str(g):>12} {tag2}')
print('  读法：**同一批随机策略的轨迹反复喂给 REINFORCE，学不到最优** ->')
print('        因为策略改好后，"好动作"的样本在这批老数据里几乎没有（**on-policy 要求新样本**）')

print('')
print('③ 阳性对照：同一算法换成「每次新采样」就能学会')
def reinforce_fresh(episodes, lr=0.05, gamma=0.95):
    theta={(r,c):[0.0]*4 for r in range(N) for c in range(N)}; base=0.0; hist=[]
    for _ in range(episodes):
        s=(0,0); tr=[]; steps=0
        while True:
            z=theta[s]; m=max(z); e=[math.exp(v-m) for v in z]; Z=sum(e); p=[v/Z for v in e]
            x=rng.random(); acc=0.0; a=3
            for i,pi in enumerate(p):
                acc+=pi
                if x<=acc: a=i; break
            ns,r,done=step(s,a); tr.append((s,a,r)); s=ns; steps+=1
            if done or steps>200: break
        G=0.0; rets=[]
        for _,_,r in reversed(tr):
            G=r+gamma*G; rets.append(G)
        rets.reverse()
        for (st,a,_),Gt in zip(tr,rets):
            z=theta[st]; m=max(z); e=[math.exp(v-m) for v in z]; Z=sum(e); p=[v/Z for v in e]
            adv=Gt-base
            for k in range(4): theta[st][k]+=lr*adv*((1.0 if k==a else 0.0)-p[k])
            base=0.9*base+0.1*Gt
        hist.append(steps)
    return hist, theta
hist, th = reinforce_fresh(400)
def solved(hist, thresh=10.0, w=50):
    for i in range(w,len(hist)):
        if sum(hist[i-w:i])/w < thresh: return i
    return None
print(f'  REINFORCE + 每次新采样：400 集内移动均值 < 10 的集数 = {solved(hist)}')
print(f'  最终贪心策略步数 = {greedy_theta(th)}（最优 8）')
print('  读法：**同一算法、同样超参，只把「固定数据」换成「每次新采样」就能学会** ->')
print('        这证明 ② 的失败原因是「分布不匹配」，而不是实现 bug')

print('')
print('④ 判据：什么时候用哪个')
print(f'  {"判据":<22} {"选 Q-learning":<26} {"选策略梯度":<26}')
for a,b,c in (('动作空间','离散、低维','**连续/高维**'),
              ('能否复用历史数据','**能（off-policy）**','不能（on-policy）'),
              ('需要随机策略','不需要','**需要（如探索/多模态）**'),
              ('样本预算','**紧**','充足'),
              ('稳定性要求','可接受调参','**要求稳（PPO 类）**')):
    print(f'  {a:<22} {b:<26} {c:<26}')
print('  读法：**"动作是否连续「与」数据能否复用"是两个最实用的判据**')
```

预期输出要点（实跑）：① **off-policy**：Q-learning 在固定 4000 条转移上遍历 1/3/10/30 遍 → **贪心策略步数都是 8（最优）**；② **on-policy**：REINFORCE 在同一批 600 条轨迹上遍历 1/5/20/60 遍 → **始终未学会**；③ **阳性对照**：REINFORCE 每次新采样 → **第 113 集达标、最终贪心 8 步（最优）**；④ 四条判据表。

## 常见追问

- **追问**：为什么连续动作下 Q-learning 不行？
  - 要点：**因为要 $\arg\max_a Q(s,a)$**：① **离散动作只需枚举**（**$O(|A|)$**）；② **连续动作要在每步解一个优化问题**（**代价高且不精确**）；③ **折中方案**：**DDPG 用一个"actor 网络"来近似 $\arg\max$**（**于是它就变成了 actor-critic**）。**读法**：**"DDPG 是'用策略网络解决 $\arg\max$ 的 Q-learning'"**——**这句话能说清两者的关系**。
- **追问**：重要性采样能救 on-policy 吗？
  - 要点：**能但要付代价**：① **用 $\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$ 加权**（**修正分布**）；② **但比值的方差随轨迹长度指数增长**（**长序列下不可用**）；③ **所以有"截断重要性采样"（PPO 的 clip）**——**它牺牲无偏性换稳定性**。**读法**：**"PPO 的 clip 就是'受控的重要性采样'"**——**这是理解 PPO 的关键**。
- **追问**：样本效率差多少？
  - 要点：**量级差异**：① **DQN 在 Atari 上通常需要 $10^7$ 帧**；② **而 off-policy 方法（如 Rainbow）在同样帧数下分数更高**；③ **在连续控制上，SAC 通常比 PPO 样本效率高 5–10 倍**（**但 PPO 更稳、更易调**）。**读法**：**"样本效率与稳定性是一对取舍"**——**SAC 高效但难调、PPO 稳但费样本**。
- **追问**：为什么 RLHF 用 PPO？
  - 要点：**三个原因**：① **动作空间是词表**（**巨大离散 → 值方法不可行**）；② **序列生成是"一步动作、稀疏奖励"**（**只有最后有奖励信号**）；③ **PPO 的 clip 提供了稳定性**（**这在"策略与奖励模型都不稳"的场景里很关键**）。**读法**：**"RLHF 的算法选择完全由'动作空间'与'稳定性'决定"**——**与经典 RL 的判据一致**。
- **追问**：怎么判断该用离线 RL？
  - 要点：**两个信号**：① **有大量历史数据但无法在线交互**（**如医疗、金融**）；② **在线探索代价高或危险**。**难点**：**分布外动作的 $Q$ 值被高估**（**"外推误差"**）——**所以离线 RL 的核心是"保守"（CQL 惩罚 OOD 动作的 $Q$）**。**读法**：**"离线 RL 的敌人是外推误差"**——**一句话说清它的技术核心**。
- **追问**：这道题的"陷阱"在哪？
  - 要点：**把两者当成"对立的技术"**：① **现代算法几乎都是 actor-critic**（**两者结合**）；② **"选哪个"实际是"选哪种 actor-critic 与探索机制"**；③ **而判据仍然是"动作空间 + 数据可用性 + 随机性需求"**。**读法**：**"不要把 Q-learning 与策略梯度对立"**——**面试时说"实践中是 actor-critic，判据是这三条"会显得更成熟**。

## 相关题目

- [[gdm-10]]：抛硬币的期望——**同一家公司的另一道概率题**。
- [[gdm-05]]：bias-variance 权衡——**RL 里的偏差-方差取舍（TD 偏差 vs 蒙特卡洛方差）**。
- [[agents-05]]：agent 的权限与授权——**RL 在 agent 系统里的位置**。
- [[coding-31]]：随机化算法与期望分析——**期望与方差的分析工具**。
- [[evaluation-06]]：不确定性与区间——**"训练曲线波动"的量化**。

## 参考资料与归属

- **Reinforcement Learning: An Introduction（延伸）** —— Sutton & Barto，2018-01-01：<http://incompleteideas.net/book/the-book-2nd.html>。**off-policy/on-policy 的定义、TD 学习与策略梯度定理**是本篇第 1、2 节的依据（**第 3、13 章**）。
- **Human-level control through deep reinforcement learning（延伸）** —— Mnih et al. (DeepMind)，2015-02-25：<https://www.nature.com/articles/nature14236>。**DQN 的经验回放与目标网络（以及"致命三要素"的工程解法）**是本篇第 3 节的依据。
- **Proximal Policy Optimization Algorithms（延伸）** —— Schulman et al. (OpenAI)，2017-07-20：<https://arxiv.org/abs/1707.06347>。**PPO 的截断目标与稳定性**是本篇"重要性采样"追问的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（5×5 gridworld、每步 −1、最大 200 步、4000 条转移、600 条轨迹、$\alpha{=}0.3$、$\gamma{=}0.95$、$\epsilon{=}0.1$、学习率 0.05、遍历次数 1–60、400 集阳性对照）都是**本机实跑结果**（**可复现**）；**"8 步 = 最优"是曼哈顿距离的直接计算**。**⚠️ 本机是 tabular（表格型）实验**——**深度 RL 的结论会受函数逼近与超参影响**（**"Q-learning 样本效率更高"在深度 RL 里仍然成立，但差距会变化**）；**"REINFORCE 学不会"依赖"数据完全来自随机策略"这一极端设置**（**若数据来自一个还不错的策略，结论会更温和**）。**可迁移的结论是"off-policy 能从固定数据学习、on-policy 不能"**，**不是具体的集数**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
