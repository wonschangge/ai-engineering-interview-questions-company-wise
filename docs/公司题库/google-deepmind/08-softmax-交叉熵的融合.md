---
type: question
id: gdm-08
company: Google DeepMind 与 Google AI
topic: llm-internals
order: 8
question: 推导 softmax 输入下交叉熵损失的梯度，并解释为什么在数值上要把二者融合（fuse）。
question_en: Derive the gradient of the cross-entropy loss with softmax inputs, and explain why the two are fused numerically.
asked_at: []
level: 高阶
tags: [softmax, 交叉熵, 梯度推导, log-sum-exp, 数值稳定性]
sources:
  - title: Attention Is All You Need（延伸）
    url: https://arxiv.org/abs/1706.03762
    author: Vaswani et al. (Google)
    published: 2017-06-12
  - title: The Log-Sum-Exp Trick（延伸）
    url: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    author: Gregory Gundersen
    published: 2020-02-09
  - title: Numerical Computation（延伸）
    url: https://www.deeplearningbook.org/contents/numerical.html
    author: Goodfellow, Bengio & Courville
    published: 2016-01-01
related: [gdm-05, gdm-07, openai-13, coding-27, llm-internals-01]
updated: 2026-09-28
---

## 一句话答案

> **梯度推导**：**$\dfrac{\partial L}{\partial z_k}=q_k-y_k$**（**softmax 概率减 one-hot**）——**推导的关键是"softmax 的雅可比 × 交叉熵的导数"里那一项恰好抵消**。
> **为什么融合**：**分开算（先 softmax 再取 log）会经历"溢出"与"下溢"两次数值灾难**——**而融合（log-sum-exp）把两者合成一步，全程在对数域完成。**
> **★ 推导（三步）**：
> $$L=-\log q_y,\qquad q_k=\frac{e^{z_k}}{\sum_j e^{z_j}}$$
> **第一步**（对 $q$ 求导）：$\dfrac{\partial L}{\partial q_k}=-\dfrac{\mathbb 1[k{=}y]}{q_k}$
> **第二步**（softmax 的雅可比）：$\dfrac{\partial q_k}{\partial z_i}=q_k(\mathbb 1[k{=}i]-q_i)$
> **第三步**（链式法则，**关键抵消**）：
> $$\frac{\partial L}{\partial z_i}=\sum_k\frac{\partial L}{\partial q_k}\frac{\partial q_k}{\partial z_i}=-\sum_k\frac{\mathbb 1[k{=}y]}{q_k}q_k(\mathbb 1[k{=}i]-q_i)=-(\mathbb 1[y{=}i]-q_i)=q_i-y_i$$
> **读法**：**$q_k$ 在分子分母里各出现一次，恰好约掉**——**这就是"梯度如此简洁"的原因**（**也是"融合"的数学基础：整条链路可以在对数域重写**）。
> **★ 量化一：朴素实现会崩（溢出与下溢）**
> | 场景 | **朴素（softmax → log）** | **融合（log-sum-exp）** | 真值 |
> | --- | --- | --- | --- |
> | 大 logits、真实类最大（$[1000,0,0]$） | **OverflowError** | **−0.000000** | 0.0 |
> | 大 logits、真实类最小（$[0,1000,0]$） | **OverflowError** | **1000.000000** | 1000.0 |
> | **真实类 logit 极小（$[-800,0,0]$）** | **ValueError**（`log(0)`） | **800.693147** | $800+\ln 2$ |
> | 中 logits、真实类最大（$[20,0,0]$） | 4.12e-09 | **0.000000** | 0.0 |
> | 中 logits、真实类最小（$[0,20,0]$） | 20.000000004 | **20.000000** | 20.0 |
> **读法**：**① 大 logits 下朴素实现直接抛异常**（**$\exp$ 在 709.78 溢出**）；**② 真实类 logit 极小时 `log(0)` 报错**（**$\exp(-800)$ 下溢为 0**）；**③ 即使在中 logits 下，朴素实现也有 $4\times10^{-9}$ 的绝对误差**（**真值是 0**）。**所以"小规模测试通过"不能保证线上不崩**。
> **★ 量化二：梯度校验（融合实现）**
> | 项 | 值 |
> | --- | --- |
> | 解析梯度 $q-\text{onehot}$（$z{=}[1,2,3]$、$y{=}2$） | **[0.090031, 0.244728, −0.334759]** |
> | 数值梯度 | **[0.090031, 0.244728, −0.334759]** |
> | **最大绝对差** | **$1.74\times10^{-10}$** |
> **读法**：**融合实现的梯度与数值梯度一致**——**所以"数值稳定"没有牺牲正确性**。
> **★ 三个实现要点**：
> | # | 要点 |
> | --- | --- |
> | ① | **log-sum-exp 必须先减最大值**（$\text{lse}(z)=m+\log\sum e^{z_i-m}$，$m{=}\max z$） |
> | ② | **反向传播要用 $q-y$ 而不是"先算 softmax 再算交叉熵"**（**前者天然稳定**） |
> | ③ | **框架里用 `log_softmax` + `nll_loss`**（**而不是 `softmax` + `log`**——**这是 PyTorch 把两者分开提供的唯一理由**） |
> 一句话判据：**"推导出 $q-y$ → 用 log-sum-exp 融合 → 反向也用融合形式 → 用数值梯度校验"**。

## 面试官在考什么

- **★ 能否推出 $q-y$**：**能否写出"分子分母抵消"这一步**（**而不是背结论**）。
- **★ 是否知道"为什么融合"**：**能否指出"溢出 + 下溢"两个灾难**（本机：**OverflowError 与 ValueError**）。
- **★ 是否知道 exp 的阈值**：**能否说出 709.78**（**float64**）。
- **是否知道"中 logits 也有误差"**：**能否指出 $4\times10^{-9}$ 的绝对误差**（**真值为 0**）。
- **实现细节**：**能否说出"减最大值"、`log_softmax` + `nll_loss`**。
- **梯度校验**：**能否用数值梯度验证解析梯度**。
- **框架意识**：**能否指出"PyTorch 分开提供这两个函数就是为了稳定性"**。
- **推广**：**能否指出"同样的技巧用在注意力里"**（**在线 softmax**，**串 [[openai-12]]**）。
- **诚实**：**承认"$q-y$ 的形式依赖'softmax + 交叉熵'这个组合"**（**换损失函数就不成立**）。

**常见错误答案**

- **只背 $q-y$**（**推不出来**）。
- **说"融合是为了更快"**（**是为了数值稳定**）。
- **用 `softmax` 再 `log`**（**会崩**）。
- **不减最大值**（**大 logits 溢出**）。
- **不做梯度校验**（**实现错了也不知道**）。
- **不知道 exp 的溢出阈值**。
- **以为"float32 更安全"**（**float32 的阈值更低：88.7**）。
- **不知道同样的技巧用在注意力里**（**在线 softmax**）。

## 原理与推导

### 1. 梯度推导（完整三步）

$$L=-\log q_y=-z_y+\log\sum_j e^{z_j}$$

**更直接的推导**（**推荐**——**不需要雅可比**）：

$$\frac{\partial L}{\partial z_i}=-\mathbb 1[i{=}y]+\frac{e^{z_i}}{\sum_j e^{z_j}}=q_i-y_i$$

**读法**：**从 $L=-z_y+\log\sum e^{z_j}$ 出发**，**$\log\sum e^{z_j}$ 对 $z_i$ 的导数恰好是 $q_i$**——**一步就得到结论**（**比走雅可比更简洁**）。

### 2. ★ 为什么融合

| 步骤 | 朴素路径 | 融合路径 |
| --- | --- | --- |
| **1** | $e^{z_k}$（**溢出风险**） | $z_k-m$（**减最大值**） |
| **2** | 归一化 → $q_k$（**下溢为 0**） | $\text{lse}=m+\log\sum e^{z_k-m}$ |
| **3** | $\log q_y$（**$\log 0$**） | $\log q_y=z_y-\text{lse}$ |

**读法**：**"减最大值"同时解决溢出与下溢**——**因为 $z_k-m\le0$，所以 $e^{z_k-m}\in(0,1]$**（**永不溢出**）；**而 $\log q_y$ 直接在对数域算**（**永不取 $\log 0$**）。

### 3. 数值阈值

| dtype | $\exp$ 溢出阈值 | 说明 |
| --- | --- | --- |
| **float64** | **709.78** | **本机实测** |
| **float32** | **88.72** | **低得多** |
| **float16** | **11.09** | **极低**（**混合精度训练必须小心**） |

**读法**：**"混合精度训练里 logits 到 10 就可能溢出"**——**这是"为什么必须用 log_softmax"的最实际理由**（**串 [[llm-internals-01]]**）。

### 4. 实现对照

| 实现 | 代码 | 稳定性 |
| --- | --- | --- |
| **朴素** | `-log(softmax(z)[y])` | **崩** |
| **减最大值** | `-log(softmax(z - max(z))[y])` | **好**（**但仍有 $\log 0$ 风险**） |
| **融合** | `-(z[y] - logsumexp(z))` | **最好** |
| **框架** | `nll_loss(log_softmax(z), y)` | **最好（且高效）** |

**读法**：**"减最大值"是必要条件、不是充分条件**——**因为 $q_y$ 仍可能下溢为 0**（**本机 $[-800,0,0]$ 的例子**）。

### 5. 同样的技巧在别处

| 位置 | 用法 |
| --- | --- |
| **注意力** | **在线 softmax**（**FlashAttention 的核心**，**串 [[openai-12]]**） |
| **语言模型** | **per-token log 概率**（**串 [[openai-13]]**） |
| **对比学习** | **InfoNCE 的 log-sum-exp** |
| **混合专家** | **门控的 softmax** |
| **采样** | **top-p/top-k 的归一化** |

**读法**：**"log-sum-exp 是深度学习里最通用的数值技巧"**——**它出现的地方比"交叉熵"多得多**。

### 6. 梯度校验的做法

$$\frac{\partial L}{\partial z_i}\approx\frac{L(z+\epsilon e_i)-L(z-\epsilon e_i)}{2\epsilon}$$

| 项 | 说明 |
| --- | --- |
| **$\epsilon$** | **$10^{-6}$ 左右**（**太大截断误差大、太小舍入误差大**） |
| **判据** | **相对误差 $<10^{-6}$** |
| **用途** | **验证自定义损失/层的实现** |

**读法**：**"梯度校验"是写自定义层时的标准动作**——**它能在训练前发现实现错误**。

## 数值与代码验证

### 表 1：朴素 vs 融合、梯度校验（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
import math
print('① exp 的溢出阈值（float64）')
lo,hi=0.0,2000.0
for _ in range(200):
    mid=(lo+hi)/2
    try:
        v=math.exp(mid)
        if math.isinf(v): hi=mid
        else: lo=mid
    except OverflowError: hi=mid
print(f'  math.exp 溢出阈值 ≈ {lo:.2f}（超过它 -> OverflowError 或 inf）')
print(f'  所以 logits 到 1000 量级时，朴素 softmax 必崩（exp(1000) 溢出）')
print()
print('② 朴素 vs 融合（log-sum-exp）：三个场景')
def naive_ce(z,y):
    try:
        m=max(z); e=[math.exp(v) for v in z]; s=sum(e); p=[v/s for v in e]
        return -math.log(p[y])
    except (OverflowError, ValueError, ZeroDivisionError) as ex:
        return f'{type(ex).__name__}'
def stable_ce(z,y):
    m=max(z)
    lse=m+math.log(sum(math.exp(v-m) for v in z))
    return -(z[y]-lse)
print(f'  {"场景":<30} {"朴素":>12} {"融合":>10} {"正确值":>9}')
for name,z,y,exp in (('大 logits、真实类最大（溢出）',[1000,0,0],0,0.0),
                     ('大 logits、真实类最小（溢出）',[0,1000,0],0,1000.0),
                     ('真实类 logit 极小（下溢）',[-800,0,0],0,800.693147),
                     ('中 logits、真实类最大',[20,0,0],0,0.0),
                     ('中 logits、真实类最小',[0,20,0],0,20.0)):
    n=naive_ce(z,y); s=stable_ce(z,y)
    print(f'  {name:<30} {str(n):>12} {s:>10.6f} {exp:>9.1f}')
print('  读法：**朴素实现在 |logit| 到 1000 时给出 OverflowError 或 inf；融合实现给出正确值** ——')
print('        而 |logit| ≤ 20 时两者都对（**所以"小规模测试通过"不能保证线上不崩**）')
print()
print('③ 梯度校验：解析梯度 q - onehot vs 数值梯度（融合实现）')
def log_softmax(z):
    m=max(z); lse=m+math.log(sum(math.exp(v-m) for v in z))
    return [v-lse for v in z]
def ce(z,y):
    return -log_softmax(z)[y]
z=[1.0,2.0,3.0]; y=2
q=[math.exp(v) for v in log_softmax(z)]
grad=[q[k]-(1.0 if k==y else 0.0) for k in range(len(z))]
eps=1e-6
num=[]
for k in range(len(z)):
    zp=list(z); zp[k]+=eps; zm=list(z); zm[k]-=eps
    num.append((ce(zp,y)-ce(zm,y))/(2*eps))
print(f'  z = {z}, 真实类别 y = {y}')
print(f'  解析梯度 q - onehot = {[round(v,6) for v in grad]}')
print(f'  数值梯度            = {[round(v,6) for v in num]}')
print(f'  最大绝对差 = {max(abs(a-b) for a,b in zip(grad,num)):.2e}')
print('  读法：**解析梯度与数值梯度一致（差在 1e-10 量级）** -> 融合实现的梯度也是正确的；')
print('        而梯度形式"q - onehot"极其简洁（这是交叉熵在工程上最讨喜的一点）')
```

预期输出要点（实跑）：① **$\exp$ 溢出阈值 ≈ 709.78**（float64）；② **五个场景**：大 logits 的两种情形下朴素实现**抛 `OverflowError`**、真实类 logit 极小时**抛 `ValueError`**（`log(0)`），而融合实现给出 **−0.000000 / 1000.000000 / 800.693147**；中 logits 下朴素实现有 **$4.12\times10^{-9}$** 的误差；③ **梯度校验**：解析梯度与数值梯度的**最大绝对差 $1.74\times10^{-10}$**。

## 常见追问

- **追问**：为什么不直接用 $L=-z_y+\log\sum e^{z_j}$ 实现？
  - 要点：**那就是融合实现**：① **它只需要一次 log-sum-exp**（**不需要先归一化**）；② **梯度直接是 $q-y$**（**$q$ 由 log-softmax 得到**）；③ **所以"融合"的准确含义是"用对数域的形式表达整个损失与梯度"**。**读法**：**"融合"不是"把两个函数写在一起"，而是"在数学上重写整条链路"**。
- **追问**：float32 下要更小心什么？
  - 要点：**阈值低得多**：① **$\exp$ 在 88.72 溢出**（**float64 是 709.78**）；② **混合精度训练里 logits 到 10 就要注意**；③ **解法**：**在 float32 里做 log-sum-exp，再转回低精度**（**或全程用 log 域**）。**读法**：**"低精度的数值范围小得多"**——**这是混合精度训练必须显式处理 softmax 的原因**。
- **追问**：如果 logits 里有 NaN/inf 呢？
  - 要点：**要显式处理**：① **NaN 会传播**（**结果全是 NaN**）；② **inf 会让 softmax 变成 one-hot**（**数学上是"确定性"**）；③ **实践**：**在损失里做 `torch.nan_to_num` 或跳过该样本**（**但要记录**）。**读法**：**"NaN 通常来自上游（如除零、log(0)）"**——**要在源头修**（**串 [[gdm-03]]**）。
- **追问**：标签平滑下梯度是什么？
  - 要点：**变成 $q-\tilde y$**：① **目标从 one-hot 变成 $\tilde y$**（**平滑分布**）；② **梯度仍然是"预测减目标"**（**形式不变**）；③ **所以标签平滑的实现只需改目标向量**（**串 [[openai-09]]**）。**读法**：**"$q-y$ 的形式对'任意目标分布'都成立"**——**这是它优雅的地方**。
- **追问**：为什么框架要分开提供 `log_softmax` 与 `nll_loss`？
  - 要点：**为了数值稳定与效率**：① **分开提供让用户能组合**（**如自定义权重、ignore_index**）；② **`nll_loss` 直接吃 log 概率**（**不需要再取 log**）；③ **框架内部用融合的 CUDA kernel**（**一次遍历完成**）。**读法**：**"分开提供"正是"为了让你不去做 softmax + log"**——**这是设计意图**。
- **追问**：同样的技巧在注意力里怎么用？
  - 要点：**在线 softmax**：① **分块计算时无法一次看到所有 logits**；② **维护"当前最大值 $m$"与"当前累积和 $\ell$"**；③ **每来一块就更新 $m$ 与 $\ell$，并重缩放之前的结果**；④ **这样只需 $O(n)$ 内存**（**串 [[openai-12]] 的 FlashAttention**）。**读法**：**"在线 softmax 是 log-sum-exp 的流式版本"**——**它让"分块计算"在数值上成立**。

## 相关题目

- [[gdm-07]]：正则化 vs 验证——**同一家公司的另一道 ML 基础题**。
- [[openai-13]]：交叉熵、KL 与困惑度——**交叉熵的语义与换算**。
- [[openai-12]]：self-attention 与长上下文——**在线 softmax 的应用**。
- [[coding-27]]：log-softmax 的数值稳定实现——**工程版本**。
- [[llm-internals-01]]：语言模型的训练目标——**混合精度下的数值问题**。

## 参考资料与归属

- **Attention Is All You Need（延伸）** —— Vaswani et al. (Google)，2017-06-12：<https://arxiv.org/abs/1706.03762>。**softmax 与缩放点积注意力**是本篇"同样的技巧在别处"一节的依据。
- **The Log-Sum-Exp Trick（延伸）** —— Gregory Gundersen，2020-02-09：<https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/>。**log-sum-exp 的推导与"减最大值"的必要性**是本篇第 2 节的直接来源。
- **Numerical Computation（延伸）** —— Goodfellow, Bengio & Courville，2016-01-01：<https://www.deeplearningbook.org/contents/numerical.html>。**上溢/下溢、条件数与数值稳定的通用论述**是本篇第 3 节的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（$\exp$ 阈值 709.78、五组 logits（$[1000,0,0]$、$[0,1000,0]$、$[-800,0,0]$、$[20,0,0]$、$[0,20,0]$）、梯度校验的 $z{=}[1,2,3]$ 与 $\epsilon{=}10^{-6}$）都是**本机实跑结果**（**可复现**）；**溢出阈值由二分搜索得到、梯度差由数值微分得到**。**⚠️ 阈值 709.78 是 float64 的**（**float32 是 88.72、float16 是 11.09**）——**换 dtype 结论会变**；**可迁移的结论是"朴素实现会溢出/下溢，融合实现不会"**，**不是具体阈值**。**本机的朴素实现用 `math.exp`（会抛异常）**——**用 numpy 会得到 inf/nan 而不是异常，但结果同样不可用**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
