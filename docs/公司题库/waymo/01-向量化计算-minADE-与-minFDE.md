---
type: question
id: waymo-01
company: Waymo
topic: coding
order: 1
question: 在 NumPy 中，为多模态轨迹预测计算 minADE 和 minFDE，且 ground truth 长度可变。不允许使用任何 Python 循环。
question_en: In NumPy, compute minADE and minFDE for multimodal trajectory prediction with variable-length ground truth. No Python loops allowed.
asked_at: []
level: 进阶
tags: [minADE, minFDE, 向量化, 掩码, 广播]
sources:
  - title: nuScenes: A Multimodal Dataset for Autonomous Driving（延伸）
    url: https://arxiv.org/abs/1903.11027
    author: Caesar et al.
    published: 2019-03-26
  - title: MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction（延伸）
    url: https://arxiv.org/abs/1910.05449
    author: Chai et al. (Waymo)
    published: 2019-10-12
  - title: NumPy: Array Programming for Python（延伸）
    url: https://numpy.org/doc/stable/
    author: NumPy Developers
    published: 2024-01-01
related: [waymo-02, waymo-03, waymo-07, coding-01, applied-01]
updated: 2026-09-28
---

## 一句话答案

> **核心技巧是"广播 + 掩码 + take_along_axis"**：**pred (N,K,T,2) 减 gt (N,T,2) 得到 (N,K,T,2)，用掩码把超出真值长度的步清零、再按长度取最后一步**——**而最容易错的是"用 ADE 选模态再报 FDE"（约 51% 的情况下两个最优模态不同、平均高估 2.216）。**
> **★ 量化一：单样本与掩码口径**
> K=6、预测 80 步、真值 60 步
> | 项 | **值** |
> | --- | --- |
> | **minADE** | **1.742** |
> | **minFDE** | **2.984** |
> **读法**：**minADE 取"所有模态里平均位移最小的"、minFDE 取"终点位移最小的"**——**所以两个 min 是独立的，它们可能来自不同模态**。**而"真值短于预测时必须掩掉多出的步"，否则口径不一致**。
> **★ 量化二：两个 min 来自不同模态的频率（51.4%）**
> 1,000 次随机试验
> | 项 | **值** |
> | --- | --- |
> | **argmin(ADE) ≠ argmin(FDE) 的比例** | **51.4%** |
> | **不一致时"用 ADE 选模态再报 FDE"平均高估** | **2.216** |
> | 最大高估 | **12.129** |
> **读法**：**约 51% 的情况下两个最优模态不同**——**所以"用 ADE 选模态再报 FDE"会系统性高估（平均 2.216）**。**而正确做法是两个指标各自独立取 min，这是实现里最常见的错**。
> **★ 量化三：变长真值的两种口径（方向不一致）**
> | 真值长度 | **掩码** | **补齐** | **差** |
> | --- | --- | --- | --- |
> | 20 | **1.035** | 0.932 | **-0.104** |
> | 40 | 0.823 | **1.630** | **+0.807** |
> | 60 | 1.816 | 2.183 | **+0.367** |
> | **80** | **2.574** | **2.574** | **+0.000** |
> **读法**：**两种口径的差值方向"不一致"（20 步时补齐更低、40/60 步时更高）**——**所以"补齐"既可能高估也可能低估，取决于"预测是否继续运动"**。**而"真值 80 步时两者完全一致"，所以口径差异只在"变长"时出现**。
> **★ 量化四：向量化 vs 循环（批量决定胜负）**
> float32、K=6、T=80
> | 批量 N | **向量化** | **循环** | **加速** |
> | --- | --- | --- | --- |
> | **64** | 1.61 ms | **1.18 ms** | **0.73×（循环更快）** |
> | 1,024 | 18.70 ms | 17.94 ms | 0.96× |
> | **16,384** | **160.96 ms** | 202.94 ms | **1.26×** |
> | **65,536** | **632.25 ms** | 820.65 ms | **1.30×** |
> **读法**：**小批量下循环更快（numpy 每次调用的开销占主导）、大批量下向量化才赢**——**所以"无循环"不是绝对更快**。**而"无循环"的真正价值是"可移植到 GPU"，循环版本无法直接搬，这是它的本质优势**。
> **★ 量化五：中间张量的内存**
> | 项 | **数值** |
> | --- | --- |
> | pred（1024×6×80×2，float64） | **7.9 MB** |
> | diff/dist（同样大小） | **7.9 MB** |
> | **合计** | **15.7 MB** |
> | K=100 时 | **约 260 MB** |
> **读法**：**一批 1,024 个样本的中间张量约 16 MB，不大，所以"一次性算完"是可行的**——**而如果 K=100（多模态预测的极端）就是 260 MB，那时要分块**。**要按 K 估内存**。
> **★ 实现的三步**：
> | 步 | 代码要点 |
> | --- | --- |
> | **① 广播** | **`pred - gt[:, None, :, :]`** |
> | **② 掩码** | **`dist * (steps < L[:, None])`** |
> | **③ 取最后一步** | **`take_along_axis(dist, (L-1)[:, None, None], axis=2)`** |
> 一句话判据：**"广播出 (N,K,T,2) → 用掩码清零超长步 → 按长度 take 最后一步 → 两个指标各自独立取 min → 小批量下循环反而更快、但向量化能上 GPU"**。

## 面试官在考什么

- **★ 是否用"广播 + 掩码"处理变长**：**能否给出三步**——**这是本题的分水岭**。
- **★ 是否指出"两个 min 独立"**：**能否给出"51.4% 的情况下不同"**。
- **★ 是否算"用 ADE 选模态"的高估**：**能否给出"平均 2.216"**。
- **★ 是否用 `take_along_axis`**：**能否给出这个函数**。
- **是否指出"掩码必须用掩码均值"**：**能否给出 `sum/mask.sum`**。
- **是否指出"两种口径方向不一致"**：**能否给出"20 步时 -0.104、40 步时 +0.807"**。
- **是否算内存**：**能否给出"15.7 MB / 260 MB"**。
- **是否诚实说"小批量下循环更快"**：**能否给出这个反直觉的实测**。
- **是否指出"无循环可上 GPU"**：**能否给出这个本质优势**。
- **诚实**：**承认"minADE/minFDE 本身有固有局限"**（**串 [[waymo-03]]**）。

**常见错误答案**

- **用循环**（**违反题目要求**）。
- **不掩码**（**口径不一致**）。
- **用 ADE 选模态再报 FDE**（**平均高估 2.216**）。
- **掩码后用 `mean` 而不是 `sum/mask.sum`**（**被零稀释**）。
- **不知道 `take_along_axis`**（**用花式索引也行但要广播**）。
- **断言"向量化一定更快"**（**小批量下不是**）。
- **不考虑内存**（**K=100 时 260 MB**）。
- **只报 minADE 不报 minFDE**。

## 原理与推导

### 1. ★ 广播

| 张量 | 形状 |
| --- | --- |
| **pred** | **(N, K, T, 2)** |
| **gt** | **(N, T, 2)** |
| **diff** | **(N, K, T, 2)** |

**读法**：**"`gt[:, None, :, :]` 插入模态维"**——**这就是"无循环"的关键**。

### 2. ★ 掩码

$$\text{ADE}=\frac{\sum_t \text{dist}_t\cdot m_t}{\sum_t m_t}$$

| 项 | 作用 |
| --- | --- |
| **$m_t$** | **`steps < L`** |
| **分母** | **`mask.sum`（不是 T）** |

**读法**：**"分母必须是有效步数"**——**否则短真值被零稀释**。

### 3. ★ 两个 min

| 指标 | 取 min 的对象 |
| --- | --- |
| **minADE** | **`ade.min()`** |
| **minFDE** | **`fde.min()`** |

**读法**：**"两次独立取 min"**——**而"用 ADE 的 argmin 索引 FDE"是错的**。

### 4. ★ 口径

| 口径 | 适用 |
| --- | --- |
| **掩码** | **只用有效步** |
| **补齐** | **假设静止** |

**读法**：**"两者结果不同且方向不一致"**——**所以要在文档里写清用哪个**。

### 5. ★ 性能

| 批量 | 谁快 |
| --- | --- |
| **64** | **循环** |
| **65,536** | **向量化（1.30×）** |

**读法**：**"向量化的价值在大批量与 GPU"**——**而不在小批量**。

### 6. ★ 内存

| K | 内存 |
| --- | --- |
| **6** | **15.7 MB** |
| **100** | **260 MB** |

**读法**：**"按 K 估内存"**——**而 K 是预测器的设计参数**。

## 数值与代码验证

### 表 1：单样本、模态不一致、口径、性能、内存（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
import numpy as np, time
rng=np.random.RandomState(0)

def min_ade_fde(pred, gt, gt_len=None):
    """pred: (K,T,2) 或 (N,K,T,2)；gt: (T,2) 或 (N,T,2)；gt_len: (N,)；全部广播，无 Python 循环。"""
    pred=np.asarray(pred,float); gt=np.asarray(gt,float)
    if gt.ndim==2:
        T_gt = gt.shape[0] if gt_len is None else int(gt_len)
        d = np.linalg.norm(pred[:, :T_gt, :] - gt[None, :T_gt, :], axis=-1)
        return d.mean(axis=1).min(), d[:, -1].min()
    N,K,T,_=pred.shape; L=np.asarray(gt_len)
    steps=np.arange(T)[None,:]; mask=steps<L[:,None]
    diff=pred-gt[:,None,:,:]
    dist=np.linalg.norm(diff,axis=-1)*mask[:,None,:]
    ade=dist.sum(axis=2)/mask.sum(axis=1)[:,None]
    idx=(L-1)[:,None,None]*np.ones((1,K,1),dtype=int)
    fde=np.take_along_axis(dist,idx,axis=2)[:,:,0]
    return ade.min(axis=1), fde.min(axis=1)

print('① 单样本与掩码口径')
K,T=6,80
pred=rng.randn(K,T,2).cumsum(axis=1)*0.3
a,f=min_ade_fde(pred, rng.randn(60,2).cumsum(axis=0)*0.3)
print(f'  K={K}、预测 {T} 步、真值 60 步 → **minADE = {a:.3f}、minFDE = {f:.3f}**')
print("  读法：**minADE 取「所有模态里平均位移最小的」、minFDE 取「终点位移最小的」** ——")
print('        所以**两个 min 是独立的**（**它们可能来自不同模态**）-> 见本机②；')
print('        而**"真值短于预测时必须掩掉多出的步"**（**否则口径不一致**）-> 见本机③')
print()
print('② 两个 min 来自不同模态的频率（跑 1000 次）')
K,T=6,80; diff_cnt=0; gap=[]
for _ in range(1000):
    p=rng.randn(K,T,2).cumsum(axis=1)*0.5
    g=rng.randn(T,2).cumsum(axis=0)*0.5
    d=np.linalg.norm(p-g[None,:,:],axis=-1)
    ia,iff=int(d.mean(axis=1).argmin()), int(d[:,-1].argmin())
    if ia!=iff:
        diff_cnt+=1; gap.append(d[ia,-1]-d[iff,-1])
print(f'  **argmin(ADE) ≠ argmin(FDE) 的比例 = {diff_cnt/1000:.1%}**（1000 次里 {diff_cnt} 次）')
print(f'  **不一致时"用 ADE 选模态再报 FDE"平均高估 {np.mean(gap):.3f}**（最大 {np.max(gap):.3f}）')
print('  读法：**约 {:.0%} 的情况下两个最优模态不同** ——'.format(diff_cnt/1000))
print('        所以**"用 ADE 选模态再报 FDE"会系统性高估**（**平均高估 {:.3f}**）；'.format(np.mean(gap)))
print('        而**正确做法是两个指标各自独立取 min**（**这是实现里最常见的错**）-> 关键')
print()
print('③ 变长真值的两种口径：掩码 vs 补齐')
K,T=6,80
print(f'  {"真值长度":>8} {"掩码":>9} {"补齐":>9} {"差":>9} 说明')
for L in (20,40,60,80):
    pred=rng.randn(K,T,2).cumsum(axis=1)*0.3
    gt=rng.randn(L,2).cumsum(axis=0)*0.3
    a_m,_=min_ade_fde(pred,gt)
    gt_pad=np.vstack([gt,np.repeat(gt[-1:],T-L,axis=0)])
    d=np.linalg.norm(pred-gt_pad[None,:,:],axis=-1); a_p=d.mean(axis=1).min()
    note='**口径一致**' if L==T else ('**补齐更差**' if a_p>a_m else '**补齐反而更好**')
    print(f'  {L:>8} {a_m:>9.3f} {a_p:>9.3f} {a_p-a_m:>+9.3f} {note}')
print('  读法：**两种口径的差值方向"不一致"**（**20 步时补齐更低、40/60 步时更高**）——')
print("        所以**「补齐」既可能高估也可能低估**（**取决于「预测是否继续运动」**）；")
print("        而**「真值 80 步时两者完全一致」**（**因为不需要补**）-> 所以口径差异只在「变长」时出现")
print()
print('④ 向量化 vs 循环：批量大小决定胜负')
print(f'  {"批量 N":>8} {"向量化(ms)":>11} {"循环(ms)":>10} {"加速":>8} 说明')
for N in (64,1024,16384,65536):
    K,T=6,80
    pb=rng.randn(N,K,T,2).astype(np.float32).cumsum(axis=2)*0.3
    gb=rng.randn(N,T,2).astype(np.float32).cumsum(axis=1)*0.3
    lens=rng.randint(30,T+1,size=N)
    t0=time.perf_counter()
    for _ in range(2): a_b,f_b=min_ade_fde(pb,gb,lens)
    tv=(time.perf_counter()-t0)/2
    def looped():
        oa=np.empty(N); of=np.empty(N)
        for i in range(N):
            L=int(lens[i])
            dd=np.linalg.norm(pb[i,:,:L,:]-gb[i,:L,:][None,:,:],axis=-1)
            oa[i]=dd.mean(axis=1).min(); of[i]=dd[:,-1].min()
        return oa,of
    t0=time.perf_counter()
    for _ in range(2): a_l,f_l=looped()
    tl=(time.perf_counter()-t0)/2
    print(f'  {N:>8} {tv*1000:>11.2f} {tl*1000:>10.2f} {tl/tv:>7.2f}x '
          f'{"**向量化更快**" if tl/tv>1.2 else ("**差不多**" if tl/tv>0.8 else "**循环更快**")}')
print('  读法：**小批量下循环更快（numpy 每次调用的开销占主导）、大批量下向量化才赢** ——')
print('        所以**「无循环」不是绝对更快**（**它在大批量下才赢**）；')
print("        而**「无循环」的真正价值是「可移植到 GPU」**（**循环版本无法直接搬**）-> 这是它的本质优势")
```

预期输出要点（实跑）：① **单样本**：K=6、真值 60 步 → **minADE 1.742、minFDE 2.984**；② **模态不一致**：**51.4%** 的情况下 argmin 不同、**平均高估 2.216**（最大 12.129）；③ **口径**：真值 20/40/60/80 步 → 掩码 **1.035/0.823/1.816/2.574**、补齐 **0.932/1.630/2.183/2.574**；④ **性能**：N=64/1024/16384/65536 → **0.73×/0.96×/1.26×/1.30×**；⑤ **内存**：**15.7 MB**（K=6）/ **260 MB**（K=100）。

## 常见追问

- **追问**：为什么"用 ADE 选模态再报 FDE"这么常见？
  - 要点：**三条**：① **因为"minADE 与 minFDE 来自同一个模态"听起来更自然**；② **而实际上它们是"两个独立的假设"**；③ **所以"独立取 min"才是定义**。**读法**：**"minADE/minFDE 的定义就是'各自最优'"**——**而不是"同一模态的两个指标"**。
- **追问**：`take_along_axis` 怎么用？
  - 要点：**三条**：① **索引要与被取张量同维**（**所以 `(L-1)[:, None, None]` 广播到 (N,K,1)**）；② **`axis=2` 指定时间维**；③ **也可以用"掩码后取 max 位置"的等价写法**。**读法**：**"索引要广播到同形状"**——**这是这个 API 的唯一坑**。
- **追问**：如果真值长度是"每个模态不同"呢？
  - 要点：**三条**：① **那就要按模态掩码**（**`L` 变成 (N,K)**）；② **而"minADE"的含义变成"每个模态用各自的长度"**；③ **这在实际评测里很少见**（**因为真值只有一条**）。**读法**：**"真值只有一条，所以长度是样本级的"**——**模态级长度是另一种设定**。
- **追问**：怎么处理"真值里有缺失帧"？
  - 要点：**三条**：① **用"有效性掩码"而不是"长度"**（**因为缺失可能在中间**）；② **分母变成"有效帧数"**；③ **而"最后一步"要取"最后一个有效帧"**。**读法**：**"长度是掩码的特例"**——**所以通用实现应该用掩码**。
- **追问**：为什么要报内存？
  - 要点：**三条**：① **因为 K 是"预测器的设计参数"**（**串 [[waymo-03]]**）；② **而 K=100 时中间张量是 260 MB**；③ **所以"评测脚本本身会成为瓶颈"**。**读法**：**"评测也要算内存"**——**这在"全车队档案评测"里尤其重要**（**串 [[waymo-06]]**）。
- **追问**：这道题与"行为预测的指标"有什么关系？
  - 要点**两条**：① **[[waymo-03]] 讲"该用哪些指标"**；② **本题讲"怎么算 minADE/minFDE"**；③ **两者的连接点是"minADE 的固有局限"**（**它会奖励撒网**）。**读法**：**"实现正确"与"指标正确"是两件事**——**都要讲**。

## 相关题目

- [[waymo-02]]：模块化 vs 端到端——**预测模块在架构中的位置**。
- [[waymo-03]]：行为预测的输出表示与指标——**minADE 的局限**。
- [[waymo-07]]：全车队档案的相似片段检索——**大规模评测的内存**。
- [[coding-01]]：数组编程与向量化——**通用技巧**。
- [[applied-01]]：模仿学习的基本框架——**轨迹预测的背景**。

## 参考资料与归属

- **nuScenes: A Multimodal Dataset for Autonomous Driving（延伸）** —— Caesar et al.，2019-03-26：<https://arxiv.org/abs/1903.11027>。**minADE/minFDE 的评测口径** 是本篇第 1 节的直接来源。
- **MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses（延伸）** —— Chai et al. (Waymo)，2019-10-12：<https://arxiv.org/abs/1910.05449>。**K 条模态 + 概率的输出表示** 是本篇第 3 节的依据。
- **NumPy 文档（延伸）** —— NumPy Developers，2024-01-01：<https://numpy.org/doc/stable/>。**广播与 `take_along_axis`** 是本篇第 1、2 节的实现依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（K=6、T=80、真值 60 步、1,000 次试验的 51.4%、平均高估 2.216、口径对比的四组、批量 64–65,536 的计时、内存 15.7/260 MB）都是为演示"向量化 minADE/minFDE"而构造的**示例参数与实测结果**；**计时是本次实跑**（**可复现**，**但依赖机器**）。**⚠️ "51.4%"与"2.216"依赖随机种子与数据分布**——**它们说明的是"两个 min 会不同"这个定性事实**；**"计时"依赖硬件**。**可迁移的结论是"广播 + 掩码 + take_along_axis、两个 min 独立、口径要写清、向量化的价值在大批量与 GPU"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
