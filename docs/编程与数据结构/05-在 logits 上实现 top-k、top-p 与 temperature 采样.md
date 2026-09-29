---
type: question
id: coding-05
topic: 编程与数据结构
order: 5
question: 在 logits 向量上实现 top-k、top-p 与 temperature 采样。
question_en: Implement top-k, top-p and temperature sampling over a logits vector.
asked_at: [Google DeepMind, Apple]
level: 进阶
tags: [采样, top-p, temperature, 实现题]
sources:
  - title: Temperature 如何控制 LLM 的输出？
    url: https://outcomeschool.com/blog/how-does-temperature-control-llm-output
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Top-k 与 Top-p 采样是如何工作的？
    url: https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: The Curious Case of Neural Text Degeneration（nucleus sampling）（延伸）
    url: https://arxiv.org/abs/1904.09751
    author: Holtzman et al. (ICLR 2020)
    published: 2019-04-22
related: [llm-internals-12, coding-04, inference-serving-09, evaluation-01]
updated: 2026-09-28
---

## 一句话答案

> 把采样拆成四个可独立测试的纯函数再串成一条流水线：$z \to z/T \to \text{top-}k \to \text{top-}p \to \text{softmax} \to \text{采样}$，随机数生成器由调用方注入。
> 三个必须写死的约定：$T=0$ 不做除法、直接走 argmax；top-p 保留累积概率首次达到 $p$ 的那个最小集合（**包含**跨过阈值的那一个 token，并加浮点容差）；任何过滤之后都要在保留集内重新归一化——用 $-inf$ 掩码 logits 再做一次 softmax 就等价于精确重归一化。
> 复杂度：top-k 是期望 $O(V)$ 的选择（`torch.topk` / `np.argpartition`），top-p 需要 $O(V\log V)$ 的全排序。$V=128{,}256$、batch 64 时，向量化路径在 GPU 上实测 6.1 ms，峰值显存 380 MiB——比 $B\times V$ 的 logits 本体大 12 倍。

## 面试官在考什么

- 能不能把「采样」拆成有明确契约的纯函数：`apply_temperature` / `top_k_filter` / `top_p_filter` / `sample`，并且每个都有边界用例（$T=0$、$k\ge V$、$p=1$、全部 logits 相等、含 NaN）。
- 是否理解 temperature 改的是 logit 之间的**差**（锐度），不改排序；$T\to0$ 的极限是 one-hot，实现上是跳过采样而不是除零。
- top-p 的语义是否精确到边界：最小的 $m$ 使降序累积 $\ge p$，$m$ 由分布自己决定；以及为什么「按累积概率过滤后不重归一化」是能跑通但分布错的经典 bug。
- 是否知道两条截断的失效形状相反：top-k 的 $k$ 是常数、不适应分布形状，top-p 的累积阈值与单个 token 的质量脱钩。
- 工程面：全词表上的选择 vs 排序开销、batch 内每行 $k$ 与 $p$ 不同时的向量化、低精度下累积判定对候选集的影响、以及「记录 seed 与参数才能复现一次生成」。

常见错误答案：

- 把 top-p 说成「保留概率大于 $p$ 的 token」。阈值作用在**累积**概率上，保留的单个 token 概率可以远小于 $p$。
- 过滤后直接在原概率上采样、不除以保留部分的和；或者以为 temperature 与截断的先后顺序是无关紧要的细节——两种顺序在同一组参数下会给出不同的候选集。
- 实现里写 $z/T$ 而不处理 $T=0$：会得到 `inf`/`nan`，`nan` 一旦进入 softmax 会污染整行（实测 `torch.softmax([1.0, nan]) = [nan, nan]`）。

## 原理与推导

### 1. 契约

```python
sanitize_logits(logits) -> (logits, degraded)          # NaN/±Inf -> -inf；整行不可选时退化为均匀
softmax(logits, T=1.0) -> probs                        # 减最大值；T 必须 > 0
apply_temperature(logits, T) -> logits                 # z/T，T>0；T=0 由 sample 走 argmax
top_k_filter(logits, k) -> (logits, threshold, kept)   # 保留第 k 大及其并列项
top_p_filter(logits, p) -> (logits, kept_indices)      # 最小的 m 使降序累积 >= p
sample_token(logits, temperature, top_k, top_p, rng) -> (token_id, degraded)
```

三条不变量：① 过滤只把某些位置置成 $-inf$，**不改变**保留位置之间的相对大小；② `softmax` 之前一定先减最大值（数值稳定）；③ `rng` 必须显式传入——默认参数里藏一个全局种子，会让评测无法复现（见 [[evaluation-04]] 对门禁可复现性的要求）。

### 2. temperature

温度只在 softmax 前做一次除法，它把 logit 差整体放大 $1/T$ 倍：$\frac{z_i}{T}-\frac{z_j}{T}=\frac{z_i-z_j}{T}$，而 softmax 只依赖差，于是：$T=1$ 不变；$T<1$ 差距放大、分布更尖；$T>1$ 差距缩小、分布更平。

$\arg\max$ 在正数缩放下不变，所以温度**不改排序**，$T\to0^+$ 的极限是 one-hot，等价 greedy；$T=0$ 在实现里是「不做除法、直接 argmax」，这一步必须与采样路径分开写。

### 3. top-k

保留最大的 $k$ 个，其余置 $-inf$。不需要全排序：只要拿到**第 $k$ 大的值**当阈值，再扫一遍即可，期望 $O(V)$（`np.argpartition`、`torch.topk`，或 `heapq.nlargest` 的 $O(V\log k)$ 近似）。

一个必须写进文档的语义选择：阈值法写成 `z >= threshold` 时，**并列**的 token 会一起保留，候选数可能大于 $k$；按索引序取前 $k$ 的实现（`torch.topk`）则只留一个。实测 `[1.0, 1.0, 0.5]` 取 $k=1$：阈值法保留 2 个，`torch.topk(-, 1)` 返回索引 `[0]`。两种都对，但不能在同一套服务里混用。

### 4. top-p（nucleus）

按概率降序累加，保留最小的 $m$ 使 $\sum_{i=1}^{m}p_{(i)}\ge p$：

$$m=\min\Big\{m'\;\Big|\;\sum_{i=1}^{m'}p_{(i)}\ge p\Big\},\qquad \mathcal{C}_p=\text{top-}m$$

三个实现细节决定了正确性：**跨过阈值的那一个必须保留**（把 `cumsum >= p` 当删除掩码是常见错误，它会把让累积首次达标的那一个也删掉；正确写法是只删「不含自己就已经 $\ge p$」的位置，也就是按 `cumsum - prob < p` 保留）；**必须带浮点容差**（概率 $0.3/0.3/0.3/0.1$、$p=0.9$ 时最小核是 3 个，但 float64 里 $0.3+0.3+0.3=0.8999999999999999$，无容差实现会多收一个 token）；**$p\ge1$ 直接短路返回不过滤**（浮点累积和常常到不了精确的 1.0）。

两个退化方向都反直觉：$p=1$ 等于不过滤；而「全部 logits 相等」**不会**退化成只剩一个——均匀分布下最小核是 $\lceil pV\rceil$ 个（$V=5$、$p=0.9$ 时是 5 个；$V=128{,}256$、$p=0.9$ 时是 115,431 个）。真正退化成 1 个的是另一种情况：最高概率自己就 $\ge p$（如 logits `[10,0,0,0,0]` 的 $p_{\max}=0.99982>0.9$），此时 top-p 行为等同 greedy。

### 5. 顺序，以及为什么必须写死

生产框架里两种顺序都真实存在，结果不同：**先温度后截断**时，截断看到的分布是 $z/T$ 的 softmax，候选集大小随 $T$ 变化（同一个 top-p，低温时候选更少、高温时更多）；**先截断后温度**时候选集只由未缩放的分布决定，与 $T$ 无关。

实测（logits `[2.0, 1.0, 0.5, 0.1, -1.0]`、$T=0.5$、$k=3$、$p=0.9$）：先温度保留 2 个（0.8808 / 0.1192），先截断保留 3 个（0.8438 / 0.1142 / 0.0420）。这不是谁对谁错，而是**必须与文档、代码、日志里的参数一起固化**的实现约定，否则同一组 `(T, k, p)` 在两个栈上行为不同。

### 6. 过滤与重归一化

过滤后保留集的总概率不再等于 1，必须除掉保留部分的和。用 $-inf$ 掩码 logits 再做一次 softmax 与「先全词表 softmax，再除以保留部分之和」数学上恒等（实测最大绝对差 $1.1\times10^{-16}$），但前者更省一次全量除法，且在低精度下误差更小（bf16 下最大绝对误差 $4.2\times10^{-4}$ 对 $5.4\times10^{-4}$）。

注意保留集的和通常**大于** $p$：$p=0.9$ 的例子实测和是 0.972192，因为跨过阈值的那一个 token 也留了下来。度量「截断掉了多少质量」时不能拿 $p$ 当答案。

## 数值与代码验证

以下实现写在仓库根的 `.work/coding05/ref_sampling.py`（另有 `perf_sampling.py`、`extra_checks.py`、`penalties.py` 三个基准脚本），用 `python3` 实跑，输出为真实运行结果。

```python
import heapq, math, random

NEG_INF, TOL = float("-inf"), 1e-12

def sanitize_logits(logits):
    cleaned = [z if math.isfinite(z) else NEG_INF for z in logits]
    if all(z == NEG_INF for z in cleaned):
        return [0.0] * len(cleaned), True          # 整行不可选 -> 均匀兜底
    return cleaned, False

def softmax(logits, T=1.0):
    if not T > 0:
        raise ValueError("temperature 必须 > 0；T=0 走 argmax")
    scaled = [z / T for z in logits]
    m = max(scaled)
    if m == NEG_INF:
        raise ValueError("所有 logits 都是 -inf，无法归一化")
    exps = [0.0 if z == NEG_INF else math.exp(z - m) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]

def apply_temperature(logits, T):
    if not T > 0:
        raise ValueError("temperature 必须 > 0")
    return [z / T for z in logits]

def top_k_filter(logits, k):
    V = len(logits)
    if k is None or k <= 0 or k >= V:
        return list(logits), NEG_INF, V
    threshold = max(logits) if k == 1 else heapq.nlargest(k, logits)[-1]  # 第 k 大，不必全排序
    masked = [z if z >= threshold else NEG_INF for z in logits]           # >=：并列一起留
    return masked, threshold, sum(1 for z in masked if z != NEG_INF)

def top_p_filter(logits, p, tol=TOL):
    V = len(logits)
    if p is None or p >= 1.0:
        return list(logits), list(range(V))          # 短路：p=1 等于不过滤
    if p <= 0.0:
        zmax = max(logits)
        return [z if z == zmax else NEG_INF for z in logits], \
               [i for i, z in enumerate(logits) if z == zmax]
    probs = softmax(logits)
    order = sorted(range(V), key=lambda i: probs[i], reverse=True)
    acc, keep = 0.0, []
    for i in order:
        keep.append(i)
        acc += probs[i]
        if acc >= p - tol:                           # 保留跨过阈值的那一个
            break
    kept = set(keep)
    return [z if i in kept else NEG_INF for i, z in enumerate(logits)], keep

def sample_token(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    if rng is None:
        raise ValueError("必须显式传入 rng，否则生成不可复现")
    logits, degraded = sanitize_logits(logits)
    if degraded:
        return rng.randrange(len(logits)), True
    if temperature == 0.0:                           # 契约：T=0 等价 greedy
        zmax = max(logits)
        return next(i for i, z in enumerate(logits) if z == zmax), False
    z = apply_temperature(logits, temperature) if temperature != 1.0 else list(logits)
    z, _, _ = top_k_filter(z, top_k)
    z, _ = top_p_filter(z, top_p)
    probs = softmax(z)                               # 掩码 + 一次 softmax = 精确重归一化
    u, acc = rng.random(), 0.0
    for i, q in enumerate(probs):
        acc += q
        if u < acc:
            return i, False
    return len(probs) - 1, False
```

**表 1：固定 logits `[2.0, 1.0, 0.5, 0.1, -1.0]` 上的七组设置（保留集内重归一化后的概率，手工可核对）**

| 设置 | 候选数 | 重归一化后的概率 |
| --- | --- | --- |
| $T=1.0$ | 5 | 0.558545 / 0.205477 / 0.124628 / 0.083541 / 0.027808 |
| $T=0.5$ | 5 | 0.826465 / 0.111850 / 0.041147 / 0.018489 / 0.002049 |
| $T=2.0$ | 5 | 0.371917 / 0.225579 / 0.175681 / 0.143836 / 0.082986 |
| $T=1.0$、$k=2$ | 2 | 0.731059 / 0.268941 |
| $T=1.0$、$p=0.9$ | 4 | 0.574522 / 0.211355 / 0.128193 / 0.085930 |
| $T=0.5$、$k=3$、$p=0.9$ | 2 | 0.880797 / 0.119203 |
| $T=2.0$、$k=3$、$p=0.9$ | 3 | 0.481024 / 0.291756 / 0.227220 |

两个可手算的锚点：$T=1$、$k=2$ 时只剩 $\exp(2)$ 与 $\exp(1)$，比值给出 $1/(1+e^{-1})=0.731059$；$T=0.5$、$k=3$、$p=0.9$ 时截断先砍掉第 3 名，剩下 $\exp(4)$ 与 $\exp(2)$，比值是 $\text{sigmoid}(2)=0.880797$。同一组 logits 用 `torch.topk(logits, 3)` 再 `softmax` 得到 `[0.628532, 0.231224, 0.140244]`，与「保留集内 softmax」逐位一致（最大差 $1.1\times10^{-16}$）。

**表 2：边界用例（全部有断言，跑通 `.work/coding05/ref_sampling.py` 时零失败）**

| 用例 | 预期 | 实测 |
| --- | --- | --- |
| $k=1$ | 等于 argmax | 1000 次采样全部命中 token 0 |
| $p=1.0$；$p\to0$ 与 $p=10^{-6}$ | 不过滤；退化为 argmax | 保留 5 个；只保留 1 个 |
| 全部 logits 相等、$V=5$ | $\lceil pV\rceil$ 个 | $p=0.5\to3$；$p=0.9/0.99/1.0\to5$ |
| $p_{\max}\ge p$（logits `[10,0,0,0,0]`，$p_{\max}=0.99982$） | 只剩 1 个 | 保留 1 个，行为等同 greedy |
| 并列最大、$k=1$ | 语义相关 | 阈值法保留 2 个；`torch.topk` 只返回索引 `[0]` |
| `[1.0, nan, 0.5, inf, -inf]` | 排除 NaN 与 ±Inf | 清理为 `[1.0, -inf, 0.5, -inf, -inf]`，概率 `0.622459 / 0.377541`，候选 `[0, 2]` |
| 整行 NaN；整行 `-inf` | 不崩；明确报错 | 退化为均匀兜底（`degraded=True`）；`softmax` 抛 `ValueError` 而不是返回 `nan` |

**表 3：top-p 的两个边界反例（同一输入，不同实现给出不同答案）**

| 反例 | 输入 | 规范实现 | 错误实现 |
| --- | --- | --- | --- |
| 浮点累加 | 概率 $0.3/0.3/0.3/0.1$，$p=0.9$ | 保留 3 个（带 $10^{-12}$ 容差） | 无容差 `cumsum > p` 保留 4 个，多收一个 token |
| mask off-by-one | 概率 $0.5/0.25/0.25$，$p=0.75$ | 保留 2 个 | `cumsum >= p` 全删只保留 1 个，过度截断 |

**大样本经验频率（目标分布为表 1 中 $T=1$、$p=0.9$ 的 4 个候选，`numpy` 逆变换采样）**：采样 $10^3$ / $10^4$ / $10^5$ / $10^6$ 次时，经验频率与目标概率的最大绝对偏差分别是 0.013930 / 0.005578 / 0.001185 / 0.000296，与 $1/\sqrt{N}$ 同量级下降——这是证明采样器正确最直接的方式。同一份实现在纯 Python 下（$V=5$，每次调用重算 softmax）实测：不过滤（$k=0$、$p=1$）约 4–10 µs/次，带 top-p（$T=0.8$、$p=0.95$）约 14–20 µs/次，再叠 top-k（$k=4$）约 15–23 µs/次——同一份代码只因参数不同就差 3 倍，放到 10 万词表上更不可接受，必须走张量实现。

**表 4：全词表规模上的开销（$V=128{,}256$，LLaMA-3 的词表大小；CPU 为 32 核，GPU 为 RTX 5070 Laptop 8 GB；同一脚本多次运行的每次中位值区间，机器共享、绝对耗时随负载浮动）**

| 操作（单行、float32） | CPU | GPU |
| --- | --- | --- |
| `np.argpartition(-z, 50)` / `np.sort(z)` | 1.36–1.72 ms / 2.88–3.48 ms | — |
| `torch.topk(z, 50)` | 0.22–0.33 ms | 0.063–0.075 ms |
| `torch.sort(z)` | 10.8–11.6 ms | 0.056–0.059 ms |
| `torch.kthvalue(z, V-50)` | 1.69–1.84 ms | 0.29 ms |

选择确实比全排序便宜（numpy 上约 2.1 倍；torch CPU 上 35–49 倍，与 `topk` 走并行、`sort` 在这个 build 上偏慢有关，属于「选型必须实测」的典型例子）；到 $V=12.8$ 万的 GPU 上两者都降到 0.06–0.07 ms，**排序不再是瓶颈，内存流量才是**。

**表 5：batch 64 的向量化对比与显存（每行 $k$ 与 $p$ 都不同；同样是多次运行的每次中位值区间）**

| 路径 | 耗时 | 备注 |
| --- | --- | --- |
| numpy 循环 / numpy 向量化 | 中位 647–1029 ms / 473–711 ms | 二者结果逐位一致；循环最快一次 637 ms |
| torch 向量化 CPU | 105–152 ms | |
| torch 循环 GPU / 向量化 GPU | 7.4–10.3 ms / 6.13–6.14 ms | 循环要付 64 次 kernel 启动；向量化峰值显存 379.9 MiB |

$B\times V$ 的 fp32 logits 本体只有 31.3 MiB，采样路径的峰值却是它的 12 倍：`torch.sort` 要同时持有值（31.3 MiB）和 int64 下标（62.6 MiB），再叠加 softmax、cumsum、两级掩码与散射的临时张量。GPU 上向量化比逐行快 1.2–1.7 倍（向量化一侧稳定在 6.14 ms，波动全在逐行循环上），收益主要来自把 64 次 kernel 启动合成一次——这也是采样必须与 [[inference-serving-02]] 的 continuous batching 调度合在一起设计的原因。

**表 6：低精度下「累积判定」会改变候选集（$B=8$、$V=128{,}256$、$p=0.9$，fp64 为参考；每行候选约 30,000 个）**

| dtype | 降序累积最大绝对误差 | 候选集与 fp64 不同的 token 数/行 | 分布最大绝对误差（屏蔽 logits / 先 softmax） |
| --- | --- | --- | --- |
| float32 | $8.4\times10^{-7}$ | 0 | $3.2\times10^{-9}$ / $3.0\times10^{-9}$ |
| float16 | $2.0\times10^{-4}$ | 约 75–79（另一路线约 97–164） | $2.7\times10^{-5}$ / $2.7\times10^{-5}$ |
| bfloat16 | $1.0\times10^{-3}$ | 约 729–803 | $4.2\times10^{-4}$ / $5.4\times10^{-4}$ |

结论：**累积判定必须在 fp32（或更高）里做**。fp16 下约 30,000 个候选里有 0.26% 的边界被挪动，bf16 下是 2.6%；这两者都不是「最后一位的舍入」，而是系统性的候选集偏移。fp16 的动态范围也是真问题：把 logits 拉开到跨度 30 时，fp16 softmax 会把 128,255 个尾部 token 的概率全部压成 0（它们合计的 fp64 质量是 $8.5\times10^{-8}$，本例里不影响结果，但尾部概率一旦落在 fp16 的 subnormal 区间 $<6.1\times10^{-5}$，相对误差就会放大到 0.2% 量级）。

**可复现性（`torch.multinomial` + `Generator`）**：CPU 上 seed=7 连跑两次都得到 `[0,1,0,1,3,3,1,0,0,0]`，换成 seed=8 得到 `[2,2,0,1,0,0,1,0,1,0]`；CUDA 上 seed=7 连跑两次都得到 `[0,1,2,0,0,0,2,3,1,0]`。同 seed 的 CPU 与 CUDA 序列不同——generator 是按设备独立的流。更容易踩的坑是把 `random.Random(42)` 写在**每次调用内部**：20 次采样会退化成 20 个相同的 token（实测全为 `0`）。RNG 必须作为状态跨步复用，并和 seed 一起写进日志，才有 [[evaluation-07]] 意义上的可追溯性。

**惩罚类 logits 变换（在采样之前、softmax 之前实现），token 0（原 logit 2.0）与 token 4（原 logit −1.0）都算作已出现：**

| 规则 | 惩罚后的 logits | $p_0$ | $p_4$ |
| --- | --- | --- | --- |
| 无惩罚 | 2.000 / −1.000 | 0.5585 | 0.0278 |
| CTRL 论文原式：正负都除以 $\theta=1.2$ | 1.667 / −0.833 | 0.4727 | **0.0388**（反而升高） |
| transformers 风格：正数除以 $\theta$、负数乘以 $\theta$ | 1.667 / −1.200 | 0.4784 | 0.0272 |
| presence penalty：已出现 token 减 1.0 | 1.000 / −2.000 | 0.3265 | 0.0163 |

负 logit 除以 $\theta$ 会朝 0 移动，等于**提高**了重复 token 的概率。另外乘性惩罚与温度可交换（实测两种顺序结果完全相同），加性惩罚与温度不可交换（`[2.0, 0.0, -1.0, -1.8, -4.0]` 对 `[3.0, 1.0, 0.0, -0.8, -3.0]`），所以「惩罚与温度谁先谁后」必须按惩罚类型分别写死。

### 生产化的差距

这段代码是单请求、单机、单精度的教学实现，直接上线缺四块：① **张量形状与并行**——tensor parallel 下 logits 是按词表分片的（[[inference-serving-08]]），top-p 需要全局累积，采样器要么 all-gather logits，要么让各卡报局部候选再汇总定阈值，这段代码假设 $V$ 完整可见；② **与调度器的耦合**——continuous batching 每个 iteration 的 batch 组成都在变，每行 $k$、$p$、$T$ 都不同，还要维护 per-request 的 RNG 状态、已生成 token（惩罚项依赖它）与 `min_tokens` 之类的约束；③ **数值与硬件**——必须显式把 softmax 与累积升到 fp32，避免热路径里的 `.item()` 同步，`torch.multinomial` 与手写逆变换在不同 dtype 下的性能与精度都要实测（表 4、表 6）；④ **约束解码的交互**——JSON schema、grammar 会把非法 token 的 logit 直接置 $-inf$，此时归一化与「整行被屏蔽」的兜底路径必须有明确行为，`-inf` 数量大时减最大值的策略也要复核。

测试同样不够：这份自测覆盖了边界与统计正确性，但没有覆盖 bf16 下的候选集回归、多设备 RNG 一致性、以及「同一 prompt 两次生成的 token 序列完全一致」这类端到端断言。

## 常见追问

- **追问**：为什么生产里默认用 top-p 而不是 top-k？
  - 要点：$k$ 是常数，与当前分布形状无关——模型确信时（$p_{\max}=0.95$）仍要塞进 $k-1$ 个弱候选，模型犹豫时又会砍掉同样合理的候选；top-p 的候选数由分布自己决定（$\lceil pV\rceil$ 到 1 都可能），所以它是更安全的默认值。这也是几乎所有 API 都暴露 top-p、而 top-k 不一定有的原因。
- **追问**：temperature 与 top-p 同时用时谁先谁后？
  - 要点：没有唯一正确答案，但必须写死。先温度后截断时，同一个 $p$ 在低温下等价于更激进的截断、高温下几乎不截断（本仓库 [[llm-internals-12]] 用源文 `The cat sat on the` 的 6 项分布在 $T=0.5/1/2$ 上复算过同一个 $p=0.9$，候选数是 3/4/5），这也是「不要同时拧两个旋钮」的经验规则背后的机制。
- **追问**：为什么贪心解码在长文本上容易退化？
  - 要点：每步取 argmax 等价于整条序列上的一步贪心，没有回退；同一上下文反复触发同一个最高概率 token，就会进入 `I think that I think that` 式的循环。Holtzman 等（2019）把「用似然当解码目标」的后果总结为平淡且异常重复，并指出这与模型质量无关——同一份权重换解码策略，文本质量差异很大（论文口径）。greedy 的第二个问题是确定性：任何「采多条再筛选」的手段（自洽性投票、单测过滤、best-of-$n$）都用不上。
- **追问**：重复惩罚与频率惩罚怎么实现，有什么副作用？
  - 要点：都在 softmax 之前改 logits。repetition penalty 是乘性、按「是否出现过」判断；presence / frequency penalty 是加性、按出现次数累加。副作用有三个：对负 logit 用论文原式会把重复 token 的概率推高（实测 $p_4$ 从 0.0278 升到 0.0388，所以要按符号处理）；加性惩罚与温度不可交换，顺序必须写死；惩罚是有状态的、跨步累积，会破坏「同一分布独立采样」的假设，让 top-p 的候选集随历史变化。
- **追问**：top-p 在什么情况下也会截进垃圾 token？
  - 要点：$p_{\max}$ 略小于 $p$ 时，为了凑够质量必须从尾部补数。复算：8 个颜色合计 0.80，$p=0.9$ 还差 0.10，长尾每个 0.001，于是要再收 100 个——候选集 108 个里 92.6% 是长尾。min-p 用相对阈值（$p_{\max}\cdot p_{\min}$）替代绝对累积阈值，正是为了消除这种「为凑概率而凑数」。

## 公司变体

`asked_at` 两家在这道题上的侧重不同：

- **Google DeepMind**：偏数学与算法本身。这题的经典材料一半来自解码策略的研究脉络（nucleus sampling 出自 Holtzman 等 2019 的工作，动机是似然解码产生的文本平淡且重复），一半来自「采样在大规模服务里怎么落地」。追问常落在：$T\to0$ 的极限为什么是 argmax、top-p 的最小核怎么定义、候选集大小与熵的关系、以及在 $V$ 十万级的词表上如何避免全排序。
- **Apple**：偏工程约束与端侧/私有云场景。8 GB 级设备上，采样要和 KV cache 抢显存（[[llm-internals-02]] 的 320 KiB/token 口径），于是更关心采样路径的临时张量有多大、能否原地操作、低精度下候选集是否稳定、以及同一个 prompt 在不同设备上能否给出可复现的输出。产品侧常见问法是「怎样让同一个 prompt 的输出稳定」。

以上是依据各家公开技术材料与产品方向的侧重判断，不代表具体的面试轮次或固定题面。

## 相关题目

- [[llm-internals-12]]：五种解码策略的统一流水线与失效场景，是这道实现题的理论版本。
- [[coding-04]]：同专题的编程题，可以对照「纯函数 + 边界断言」的写法。
- [[inference-serving-09]]：TTFT / TPOT / ITL / throughput 的权衡，用来定位采样开销在延迟预算里的占比；[[inference-serving-02]] 的 continuous batching 决定采样要写成 per-row 参数的向量化实现。
- [[evaluation-01]]：LLM-as-judge 的偏差与纠正，解释为什么评测必须固定解码参数；[[evaluation-04]] 的回归门禁与 [[evaluation-07]] 的可观测性，说明 seed 与参数为什么要入档、进 trace。

## 参考资料与归属

1. [Temperature 如何控制 LLM 的输出？](https://outcomeschool.com/blog/how-does-temperature-control-llm-output)，Amit Shekhar（Outcome School）。提供 temperature 除以 logits 的机制、$T=0.1/0.5/1/2$ 的四 token 数值表、$T=0$ 即 greedy 与「$T=0$ 也几乎总是相同而非严格可复现」、以及「不要同时调 temperature 与 top-p」的工程建议；表 1 的温度行与该表按同一口径复算一致。
2. [Top-k 与 Top-p 采样是如何工作的？](https://outcomeschool.com/blog/how-do-top-k-and-top-p-sampling-work)，Amit Shekhar（Outcome School）。提供 greedy 的重复退化、纯采样的长尾问题、top-k 在「Paris 95%」与「喜欢的颜色」两种分布下的反例、top-p 的逐步累积与重归一化示例（40/25/15/10 → 44.4/27.8/16.7/11.1）、两者对照表，以及「在保留的 $k$ 个 logits 上做 softmax 等价于先 softmax 再重归一化」这个技巧。
3. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)（延伸），Ari Holtzman、Jan Buys、Li Du、Maxwell Forbes、Yejin Choi，2019-04-22（ICLR 2020）。第 4 节的动机（以似然为解码目标导致平淡且重复的文本、人类文本与机器文本存在分布差异、nucleus sampling 从动态概率核中采样以在截断不可靠尾部的同时保持多样性）取自该文摘要，属论文口径。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
