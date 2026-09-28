---
type: question
id: inference-serving-04
topic: 推理、服务与 GPU 性能
order: 4
question: 什么是 speculative decoding？为什么输出质量能够保持？什么情况下它没有帮助？
question_en: What is speculative decoding? Why is output quality preserved, and when does it not help?
asked_at: [NVIDIA, Together AI]
level: 高阶
tags: [投机解码, draft-model, 拒绝采样, 延迟]
sources:
  - title: Speculative Decoding
    url: https://outcomeschool.com/blog/speculative-decoding
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Fast Inference from Transformers via Speculative Decoding（延伸）
    url: https://arxiv.org/abs/2211.17192
    author: Leviathan et al. (Google)
    published: 2022-11-30
  - title: Accelerating Large Language Model Decoding with Speculative Sampling（延伸）
    url: https://arxiv.org/abs/2302.01318
    author: Chen et al. (DeepMind)
    published: 2023-02-02
related: [inference-serving-01, inference-serving-10, llm-internals-12]
updated: 2026-09-28
---

## 一句话答案

> speculative decoding（投机解码）让一个小 draft 模型自回归地猜 $\gamma$ 个候选 token，再让 target 模型用**一次前向**并行验证这 $\gamma$ 个位置：从前往后逐个接受，第一个被拒绝的位置按残差分布 $\max(0, p-q)$ 归一化后重新采样，其后候选全部丢弃；全部接受时额外白得一个 bonus token。
> 质量不变不是经验现象而是数学性质：接受概率取 $\min(1, p(x)/q(x))$ 恰好保留了 $\min(p,q)$ 那部分概率质量，拒绝后的残差分布 $\propto \max(0, p-q)$ 恰好补上差额，两者逐项相加等于 $p(x)$，因此输出**精确服从 target 模型的分布**，不是近似，temperature / top-p 也照常生效。
> 收益来自 decode 阶段闲置的算力：batch=1 跑 70B bf16，每步读 140 GB 权重却只做约 1 FLOP/byte，H100 上的算力利用率约 0.34%，而多验证 $\gamma$ 个候选几乎不增加这一步的耗时。代价是 target 的浮点运算量被放大 $(\gamma+1)/(\mathbb{E}[A]+1)$ 倍，所以在算力已经饱和的大 batch、或接受率过低时，收益趋零甚至转负。

## 面试官在考什么

- 动机是否落在 decode 的带宽受限本质上，而不是「小模型快，所以两个模型一起用更快」这类空话。
- 五个步骤能否完整复述，尤其是「第一个被拒绝的位置用残差分布重采样」和「全部接受时的 bonus token」这两个最容易漏的细节。
- 能否证明 $\min(1,p/q)$ 加残差重采样恰好等于精确采样。多数候选人只会说「猜对了就留下」，这题的分水岭就在这里。
- 是否理解收益的三个变量：接受率 $\alpha$、猜测长度 $\gamma$、draft 相对 target 的代价比 $c$，以及验证带来的 FLOPs 放大。
- 是否知道什么时候**不该**用：高 batch 算力饱和、draft 与 target 分布差距大、时延由 prefill 主导、显存紧张。

常见错误答案：

- 把算法描述成「draft 猜对就用、猜错就从 target 重新采样那一个 token」。后半句漏掉了残差分布：接受步骤已经按 $\min(p,q)$ 保留了一份质量，再用完整的 $p$ 重采样会把它重复计一次，凡是 $q>p$ 的 token 都被系统性放大，输出分布不再是 $p$。
- 把「质量保持」说成接受率足够高时的经验性质。它是无条件的精确等价，与 $\alpha$ 无关；$\alpha$ 低只是慢，不是差。
- 用参数量比估算加速比（1B draft vs 70B target 就以为 draft 只花 1/70 的时间）。draft 要跑 $\gamma$ 次**串行**前向，每次都有 kernel 启动与调度开销，小模型的 MFU 很低，真实代价比远大于参数量比。

## 原理与推导

### 1. 收益从哪来：decode 每步只产出一个 token，算力却大量闲置

设 target 有 $N$ 个参数，bf16 下每步要读 $2N$ 字节的权重，做 $2N$ 次浮点运算（每个参数一次乘加），于是算术强度是

$$\frac{2N\ \text{FLOPs}}{2N\ \text{bytes}} = 1\ \text{FLOP/byte}$$

H100 SXM 的 bf16 稠密算力约 989 TFLOP/s、HBM3 带宽约 3.35 TB/s，roofline 拐点在 $989/3.35 \approx 295$ FLOPs/byte。算术强度 1 只有拐点的 $1/295$，所以 batch=1 的 decode 稳定落在带宽一侧：这一步的算力利用率只有约 0.34%，时间是权重的读取时间 $2N/\text{BW}$。取 $N=70\text{B}$：$140\ \text{GB}/3.35\ \text{TB/s} \approx 41.8$ ms，也就是每 token 约 41.8 ms。roofline 的完整推导见 [[inference-serving-10]]，prefill 与 decode 的瓶颈差异见 [[inference-serving-01]]。

投机解码就是去花这 99.7% 的闲置算力：验证 $\gamma+1$ 个位置需要读同一份权重和同一段 KV cache，权重读取被摊薄到 $\gamma+1$ 个位置上，decode 的算术强度被抬高约 $\gamma+1$ 倍（$1 \to \gamma+1$）。串行步数因此减少，而每步的耗时几乎不变。

### 2. 算法：draft 猜 $\gamma$ 个，target 一次验证

记 draft 的分布为 $q$、target 的分布为 $p$，已确认前缀为 $s$。

```text
输入: 已确认前缀 s, 猜测长度 gamma
1. draft 自回归采样: for i in 1..gamma: x_i ~ q_i(.|s, x_<i), 记录 q_i(x_i)
   —— q_i 是 draft 在第 i 个位置上的完整分布, 不止是被采样的那个 token
2. target 一次前向: 输入 [s, x_1..x_gamma]
   —— causal mask 保证第 i 个位置的输出只依赖 s, x_<i,
      与串行生成时的条件分布完全一致, 所以可以并行算
   —— 得到 gamma+1 个位置的分布 p_1..p_gamma (验证用) 和 p_{gamma+1} (bonus 用)
3. for i in 1..gamma:
       以概率 min(1, p_i(x_i) / q_i(x_i)) 接受 x_i
       若拒绝: break
4. 若第 i 个位置被拒绝:
       从残差分布 norm(max(0, p_i - q_i)) 采样一个 token 顶替 x_i
       丢弃 x_{i+1..gamma} 及其对应的 target 输出 (条件前缀变了, 全部作废)
5. 若 gamma 个全部接受:
       从 p_{gamma+1} 采样一个 bonus token (这一步的分布本来就算出来了, 不额外花时间)
输出: 接受的前缀 + (被拒绝位置的替代 token 或 bonus token)
```

两个容易漏的点：

- **残差分布的归一化因子恰好是拒绝概率**。$\sum_x \max(0, p(x)-q(x)) = 1 - \sum_x \min(p(x),q(x))$，右边正是「$x\sim q$ 之后被拒绝」的概率 $\beta$，所以 $\text{norm}(\max(0,p-q))$ 是良定义的分布。
- **每轮至少前进一个 token**。即使第一个候选就被拒，第 4 步也一定产出一个服从 target 分布的 token，进度不会为零，因此最坏情况只是慢，不会卡住。

### 3. 为什么输出分布不变（这是本题的题眼）

**命题**：设 draft 分布为 $q$、目标分布为 $p$，按「$x\sim q$，以 $\min(1, p(x)/q(x))$ 接受，否则从 $\text{norm}(\max(0,p-q))$ 重新采样」的规则产出 token $y$，则 $y$ 的分布恰好是 $p$。

**证明**。先算拒绝概率：

$$\beta = 1 - \sum_x q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right) = 1 - \sum_x \min(p(x), q(x))$$

再对任意 token $y$ 展开全概率：

$$\Pr(y) = \underbrace{q(y)\min\!\left(1,\frac{p(y)}{q(y)}\right)}_{\text{接受路径}} + \underbrace{\beta \cdot \frac{\max(0, p(y)-q(y))}{\beta}}_{\text{拒绝后重采样}} = \min(p(y),q(y)) + \max(0, p(y)-q(y))$$

最后一步用到归一化因子的恒等式，而对每个 $y$ 逐项有

$$p(y) = \min(p(y),q(y)) + \max(0,\,p(y)-q(y))$$

（$p\ge q$ 时右边是 $q + (p-q)$，$p<q$ 时是 $p+0$）。于是 $\Pr(y)=p(y)$，证毕。

**直觉**：接受步骤只保住了 $\min(p,q)$ 这份「两个模型都认可」的质量，并且是按 $q$ 的比例保住的；缺口 $p - \min(p,q)$ 正好是 target 比 draft 更想要的那部分，残差分布就是把这个缺口语义化的结果。用 $p$ 直接重采样会把这个缺口算两遍，用 $q$ 重采样则缺口的符号会错——只有残差分布能逐项补齐。

**推广到 $\gamma$ 个位置**：对接受的前缀做归纳。第 1 个位置的输出服从 $p_1$；在「$x_1$ 已被接受」的条件下，第 2 个位置的接受判定与残差分布都是相对 $p_2, q_2$ 定义的，同一个论证给出它服从 $p_2$；如此递推到第 $\gamma$ 个位置以及 bonus 位置。所以整段序列的联合分布等于 target 模型逐步自回归采样的联合分布。这也解释了为什么第 4 步必须丢弃被拒绝位置之后的全部候选：那些分布的条件前缀已经作废。

**等价性的前提**（面试里主动说出来能加分）：

- **共享词表**：$p$ 与 $q$ 必须定义在同一支持集上，$\min(p,q)$ 与残差才有意义。这是 draft 通常选同族模型（tokenizer 相同）的原因，跨 tokenizer 需要先做词表映射。
- **采样参数一致且作用在两个分布上**：temperature、top-k、top-p 的截断与重归一化要同时作用于 $p$ 与 $q$，且残差必须相对「最终的目标分布」计算。只对最终采样做 top-p、却对残差分布不做截断，等价性立刻失效。
- **数值精度**：结论在精确算术下成立，浮点舍入会让它在最后几位上有偏差，DeepMind 那篇的表述是 preserves the distribution "within hardware numerics"。
- **随机性独立**：接受判定用的均匀随机数要与 draft 采样用的随机数独立，复用同一个随机数会引入相关性。
- **正确性与 $q$ 的质量无关**：$q$ 再差也不改变输出分布，只降低速度。所以「draft 模型会不会让输出变差」这个问题的答案是不用担心质量，只需要担心吞吐。

**温度=0 的退化**：$p,q$ 都变成 one-hot。若两个模型 argmax 相同，$p/q = 1$ 必然接受；不同则 $p(x)=0$ 必然拒绝，残差恰好是 target 的 one-hot，所以贪心解码下投机解码退化为「候选与 target 的 argmax 逐位比较」，仍然是精确的。采样相关的参数细节见 [[llm-internals-12]]。

### 4. 收益的量化条件

把单个位置的接受概率记作 $\alpha$，它是两个分布的重叠面积：

$$\alpha = \sum_x \min(p(x),q(x)) = 1 - \mathrm{TV}(p,q)$$

设各位置独立同分布（近似，实际上逐位置的条件分布不同），被接受的候选数 $A$ 满足 $A \ge i$ 当且仅当前 $i$ 个候选都被接受，于是

$$\mathbb{E}[A] = \sum_{i=1}^{\gamma} \Pr(A \ge i) = \sum_{i=1}^{\gamma} \alpha^i = \frac{\alpha(1-\alpha^{\gamma})}{1-\alpha}$$

每轮产出的 token 数是 $\mathbb{E}[A]+1$（被拒绝位置的替代 token 或 bonus token 都算一个）：

$$\mathbb{E}[\text{tokens}] = 1 + \mathbb{E}[A] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$

设 draft 单步耗时是 target 单步的 $c$ 倍，一轮耗时是 $\gamma c + 1$ 个 target 步，于是

$$\text{speedup} \approx \frac{\mathbb{E}[\text{tokens}]}{1 + \gamma c} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(1+\gamma c)}$$

三个可调的旋钮就是 $\alpha$、$\gamma$、$c$：

- $\alpha$ 由 draft 与 target 的接近程度决定，工程上只能换 draft 模型或蒸馏 draft，不能事后调。
- $\gamma$ 可以调，且存在最优点：$\gamma$ 越大 $\mathbb{E}[\text{tokens}]$ 越接近上界 $1/(1-\alpha)$，但分母 $1+\gamma c$ 线性增长。对 $\alpha=0.8,c=1/70$，最优 $\gamma \approx 12$，加速比约 $4.0\times$；对 $\alpha=0.7,c=0.1$，最优 $\gamma=4$，约 $2.0\times$。
- $c$ 是容易低估的量：权重的字节数之比只是它的下界。按同一个带宽受限口径，1B draft（2 GB）对 70B target（140 GB）是 $1/70$，7B draft（14 GB）是 $1/10$；但 draft 要跑 $\gamma$ 次**串行**前向，kernel 启动、采样与调度开销都不随参数量线性缩小，小模型的 MFU 明显更低，所以实测 $c$ 一般比字节数之比更大。表 3 的两组 $c$ 取值就是这两个字节数之比：$1/70$ 对应 1B draft，$0.1$ 对应 7B draft。

还有一个必须看见的代价：一轮里 target 要处理 $\gamma+1$ 个位置，只为产出 $\mathbb{E}[\text{tokens}]$ 个 token，所以 target 的 FLOPs 被放大

$$\text{amplification} = \frac{\gamma+1}{\mathbb{E}[\text{tokens}]} = \frac{(\gamma+1)(1-\alpha)}{1-\alpha^{\gamma+1}}$$

$\alpha=0.7,\gamma=4$ 时是 $1.80\times$，$\alpha=0.5,\gamma=8$ 时是 $4.51\times$。这就是「低接受率 + 大 $\gamma$ 会亏」的定量来源：闲置算力是免费额度，额度花完了就变成净支出。

## 数值与代码验证

**表 1：H100 SXM（80 GB HBM3，3.35 TB/s，bf16 稠密 989 TFLOP/s）上 batch=1 的 decode**

| 项 | 数值 | 说明 |
| --- | --- | --- |
| 70B bf16 权重 | 140 GB（130.4 GiB） | $2\times70\text{B}$ 字节 |
| 每步读写 | 约 140 GB | 权重占绝对多数，KV cache 在短上下文下可忽略 |
| 每步 FLOPs | 140 GFLOP | $2N$，每参数一次乘加 |
| 算术强度 | 1.00 FLOP/byte | 拐点 295 FLOPs/byte，差 295 倍 |
| 算力利用率 | 约 0.34% | $1/295$ |
| 每步耗时下界 | 41.8 ms | $140\ \text{GB}/3.35\ \text{TB/s}$ |
| 吞吐上界 | 23.9 token/s | 把整个 70B 放在一张卡上的理想口径 |

口径说明：140 GB 的 bf16 权重装不进一张 80 GB 卡，上面一行只为给出 roofline 上界。真实部署用 TP 分片，每卡只读自己的分片：TP=2 时每卡约 70 GB，步时下界 20.9 ms、上界约 47.9 token/s，且还要扣掉每层 all-reduce 的通信；实测通常明显低于这个上界。

**表 2：每轮期望产出的 token 数 $\mathbb{E}[\text{tokens}]=(1-\alpha^{\gamma+1})/(1-\alpha)$**

| $\gamma$ | $\alpha=0.5$ | $\alpha=0.6$ | $\alpha=0.7$ | $\alpha=0.8$ | $\alpha=0.9$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.500 | 1.600 | 1.700 | 1.800 | 1.900 |
| 2 | 1.750 | 1.960 | 2.190 | 2.440 | 2.710 |
| 4 | 1.938 | 2.306 | **2.773** | 3.362 | 4.095 |
| 6 | 1.984 | 2.430 | 3.059 | 3.951 | 5.217 |
| 8 | 1.996 | 2.475 | 3.199 | 4.329 | 6.126 |
| 12 | 2.000 | 2.497 | 3.301 | 4.725 | 7.458 |

$\alpha=0.5$ 时无论 $\gamma$ 多大都逼近 2 个 token，这是 $\alpha$ 太低时 $\gamma$ 加长没有意义的原因：多猜的候选第一轮就被拒，纯粹浪费 draft 的算力。

**表 3：加速比 $(\mathbb{E}[\text{tokens}])/(1+\gamma c)$**

| $\gamma$ | $c=0.1,\alpha=0.6$ | $c=0.1,\alpha=0.7$ | $c=0.1,\alpha=0.8$ | $c=1/70,\alpha=0.6$ | $c=1/70,\alpha=0.7$ | $c=1/70,\alpha=0.8$ |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1.63× | 1.82× | 2.03× | 1.91× | 2.13× | 2.37× |
| 4 | 1.65× | **1.98×** | 2.40× | 2.18× | 2.62× | 3.18× |
| 8 | 1.37× | 1.78× | 2.40× | 2.22× | 2.87× | 3.88× |
| 12 | 1.13× | 1.50× | 2.15× | 2.13× | 2.82× | **4.03×** |

$c=0.1$ 时最优 $\gamma$ 在 3–6 之间，$c=1/70$ 时推到 6–12：draft 越便宜，越值得多猜。盈亏平衡的接受率（speedup = 1）用同一公式反解：$c=0.1,\gamma=4$ 需要 $\alpha > 0.287$；$c=1/70,\gamma=4$ 只需要 $\alpha > 0.054$。

**表 4：投机解码把「算力拐点」提前到更小的 batch（70B bf16，TP=1，上下文 $S=2048$，KV 320 KiB/token）**

| 每步位置数 | 算力受限的临界 batch |
| --- | --- |
| 1（无投机） | 不出现：$S=2048$ 时 KV 读取的算术强度 208 FLOPs/byte 仍低于拐点 |
| 3（$\gamma=2$） | 约 186 |
| 5（$\gamma=4$） | 约 82 |
| 9（$\gamma=8$） | 约 39 |

同一份权重与 KV 读取服务 $\gamma+1$ 个位置，把 decode 从带宽侧推向算力侧；一旦 batch 超过这个临界值，验证的 FLOPs 就不再免费，吞吐随之下降。口径：每步读 $2N = 140$ GB 权重加 $B\cdot S\cdot KV_{\text{token}}$ 的 KV 前缀，做 $B(\gamma+1)\cdot 2N$ 次浮点运算，令算术强度等于拐点解出 $B$。$KV_{\text{token}}$ 沿用 [[inference-serving-10]] 的 LLaMA-3-70B GQA 口径（80 层、8 个 KV head、head_dim 128、bf16）：$2\times80\times8\times128\times2 = 327{,}680$ 字节 = 320 KiB/token，$S=2048$ 时每序列 640 MiB。这里的算力按 989 TFLOP/s 峰值计，实际可达算力只有峰值的五到七成，真实拐点比表中更靠左。

```python
def exp_tokens(alpha, gamma):
    """每轮期望产出的 token 数 (1 - alpha**(gamma+1)) / (1 - alpha)"""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):
    """c = draft 单步耗时 / target 单步耗时"""
    return exp_tokens(alpha, gamma) / (1 + c * gamma)


print(exp_tokens(0.7, 4))          # 2.7731
print(speedup(0.7, 4, 0.1))        # 1.9808
print(speedup(0.8, 12, 1 / 70))    # 4.0336
print((4 + 1) / exp_tokens(0.7, 4))  # 1.8031  FLOPs 放大倍数

# 源文口径复核: target 50 ms/步, draft 5 ms/步 (c=0.1), gamma=4
for tokens in (5, 4, 1):
    print(tokens, f"{tokens * 50 / 70:.2f}x")   # 3.57x / 2.86x / 0.71x

# 残差分布验证: 源文的 everyday / today / tomorrow 例子
q = {"everyday": 0.2, "today": 0.5, "tomorrow": 0.3}
p = {"everyday": 0.5, "today": 0.1, "tomorrow": 0.4}
print(round(sum(min(p[k], q[k]) for k in p), 3))              # 0.6 = 接受概率
res = {k: max(0.0, p[k] - q[k]) for k in p}
Z = sum(res.values())
print(Z, {k: round(v / Z, 4) for k, v in res.items()})
# 0.4 {'everyday': 0.75, 'today': 0.0, 'tomorrow': 0.25}
print({k: round(min(p[k], q[k]) + max(0.0, p[k] - q[k]), 4) for k in p})
# {'everyday': 0.5, 'today': 0.1, 'tomorrow': 0.4} == p, 逐项复原
```

与源文对照：源文用 50 ms / 5 ms / $\gamma=4$ 给出的三个数字 3.57×、2.86×、0.71× 分别是「5 个 token」「4 个 token」「1 个 token」除以 70 ms，与上面代码的输出一致，公式口径也对得上；源文的残差例子（归一化后 0.75 / 0 / 0.25）与复算一致。源文只给了 $\gamma=4$ 的定值算例，没有给期望接受长度的公式，表 2、表 3 与最优 $\gamma$ 是补齐的部分。

论文口径（不是复算值，作为量级参考）：Leviathan 等在 T5-XXL 上报告 2×–3× 加速且输出完全一致，Chen 等在 70B 的 Chinchilla 上报告分布式的 2–2.5× 解码加速；源文另外转述 DeepSeek-V3 用 Multi-Token Prediction 作为内置 drafter 得到约 1.8× 的生成加速。

## 常见追问

- **追问**：被拒绝时为什么不直接用 target 分布 $p$ 重采样？
  - 要点：接受步骤已经按 $\min(p,q)$ 保留了一份质量，再用完整的 $p$ 重采样会把它重复计一次，凡是 $q>p$ 的 token 都被系统性放大。正确做法是只补差额：$\text{norm}(\max(0,p-q))$。这是「拒绝采样 + 残差修正」与「猜对留下、猜错重来」的本质区别。
- **追问**：验证 $\gamma$ 个位置，为什么这一步的开销几乎没有增长？
  - 要点：batch=1 时这一步的耗时由权重读取决定，1 个位置和 $\gamma+1$ 个位置读的是同一份权重；causal mask 保证每个位置的分布只依赖它的前缀，所以与串行生成的条件分布一致，可以并行算。增长的部分是 KV 读取（每序列多 $\gamma$ 个位置的 cache）与投影的 FLOPs，在带宽受限区间里相对权重读取可以忽略；但如果这一步本来就在算力侧（大 batch），验证就会显著变贵。
- **追问**：temperature 或 top-p 改了，等价性还在吗？
  - 要点：在，只要把后处理后的分布当作 $p$ 与 $q$，并且对残差分布使用同一套截断规则。温度本身影响的是 $\alpha$，不是正确性。$\alpha$ 关于温度不是单调的：$T\to0$ 时两个模型都变成 one-hot（argmax 一致则 $\alpha\to1$，不一致则 $\to0$），$T\to\infty$ 时两者都趋近均匀分布、$\alpha$ 又回到 1，所以中间一定存在一个最差点。最差点落在哪个温度取决于两个模型 logit 的尺度（分布越尖锐越靠后），不能一概而论；工程做法是把 temperature、top-p 一起纳入 $\alpha$ 的实测，接受长度变短就相应调小 $\gamma$。
- **追问**：draft 模型选得差会不会影响输出质量？
  - 要点：不会。上面的证明对任意 $q$ 成立，$q$ 只决定接受率，因此只影响速度。质量唯一的风险来自实现：词表不一致、残差没有按目标分布归一化、或者只对最终采样做 top-p 而漏掉残差分布。
- **追问**：什么情况下投机解码会让服务变慢？
  - 要点：三类。一是算力饱和：高 QPS / 大 batch 下验证的 $(\gamma+1)/(\mathbb{E}[\text{tokens}])$ 倍 FLOPs 直接变成耗时，吞吐下降；二是接受率太低（draft 与 target 不同族、领域偏移大、温度很高），$\gamma$ 个候选第一轮就被拒，白跑 draft；三是收益被摊薄：输出只有几个 token、或时延由 prefill 决定（TTFT 主导），decode 省下的那几步微不足道。另外 draft 模型自身要占显存与其 KV cache（例如 1B bf16 权重约 2 GB，约占 80 GB 卡的 2.5%），在显存吃紧的部署里会挤压 batch 上限。
- **追问**：Medusa、EAGLE、n-gram 这些变体和它是什么关系？
  - 要点：都是换掉「draft 从哪来」这一步。Medusa 在 target 上加若干解码头、一次预测多个位置，再用树形注意力一次验证多条候选路径，省掉独立 draft 模型；EAGLE 在特征空间上做自回归 draft，比 token 空间的 draft 更准，接受长度更高；n-gram / 检索式 draft 完全不要模型，把上下文里出现过的后缀直接当候选，在摘要、代码改写这类会复述输入的场景里 $c\approx 0$，但在没有重复片段的场景完全失效；DeepSeek-V3 的 MTP 把 drafter 训练进模型本身，推理时不需要额外模型。它们的接受/拒绝规则仍然是同一套，等价性证明不需要改动。
- **追问**：它和 continuous batching 怎么相互影响？
  - 要点：verification 让每条序列每步占用 $\gamma+1$ 个 token 的计算位置，等于把有效 batch 放大 $\gamma+1$ 倍，会吃掉 continuous batching 用来填满算力的 batch 预算，所以「延迟换吞吐」的取舍在调度器里是显式冲突：同一张卡上开投机通常提升单请求速度、降低总吞吐。工程做法是按 QPS 与实测接受长度动态开关与调节 $\gamma$（vLLM 的 `num_speculative_tokens` 就是这类旋钮），低负载时打开换延迟，高负载时关掉换吞吐。continuous batching 见 [[inference-serving-02]]。

## 公司变体

`asked_at` 覆盖的两家公司在公开技术材料里的侧重不同：

- **NVIDIA**：偏工程实现与 kernel 视角。常从 TensorRT-LLM / GPU 架构出发，追问 draft 与 target 如何在同一个 batch 里排队、量化过的 draft（INT8/FP8）会怎么改变 $c$ 与 $\alpha$、Medusa / EAGLE / lookahead 这类树形验证的 kernel 怎么写、以及在 H100 的带宽算力比下 batch 从多大开始收益反转。答这类问题要把 $(\gamma+1)/\mathbb{E}[\text{tokens}]$ 这个 FLOPs 放大倍数和表 4 的临界 batch 讲清楚。
- **Together AI**：偏多租户 serving 与成本。常从「同一张卡上开投机，吞吐和 p50/p99 latency 各自怎么变」出发，追问接受率的监控与 $\gamma$ 的自适应、draft 与 target 的配对（是否同族、tokenizer 是否一致）、在 continuous batching 的调度里怎么和 chunked prefill 抢预算、以及低负载与高峰时段是否用不同的策略。

以上是基于两家公开技术材料与岗位方向的侧重判断，具体题目以实际面试轮次为准。

## 相关题目

- [[inference-serving-01]]：prefill 与 decode 的瓶颈差异，解释为什么投机解码只对 decode 有效。
- [[inference-serving-10]]：batch=1 时 H100 服务 70B 的 roofline 计算，给出「闲置算力」的具体数字。
- [[llm-internals-12]]：temperature、top-k、top-p 与 greedy 的行为，决定接受率 $\alpha$ 与温度=0 时的退化形式。

## 参考资料与归属

1. [Speculative Decoding](https://outcomeschool.com/blog/speculative-decoding)，Amit Shekhar（Outcome School）。提供两模型分工的整体框架、五步走查、$\min(1,p/q)$ 接受规则与残差分布的数值例子、50 ms / 5 ms / $\gamma=4$ 的 3.57× / 2.86× / 0.71× 算例、共享 tokenizer 与显存开销等 trade-off，以及 Medusa / EAGLE / n-gram / DeepSeek MTP 的定位与 1.8× 的转述。
2. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)（延伸），Leviathan、Kalman、Matias（Google），2022-11-30。提供投机解码的正式提出、无需重训或改结构的表述、以及在 T5-XXL 上 2×–3× 加速且输出一致的结论；第 3 节的分布不变性证明与第 4 节的期望接受长度推导对应这篇与下一篇的采样规则。
3. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)（延伸），Chen、Borgeaud、Irving、Lespiau、Sifre、Jumper（DeepMind），2023-02-02。提供修改版拒绝采样「在硬件数值精度内保持目标分布」的表述，以及 70B Chinchilla 上 2–2.5× 的解码加速数据。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
