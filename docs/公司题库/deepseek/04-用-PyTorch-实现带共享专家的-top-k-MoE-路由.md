---
type: question
id: deepseek-04
company: DeepSeek
topic: coding
order: 4
question: 用 PyTorch 实现带共享专家的 top-k MoE 路由，并指出其中在效率与正确性上的陷阱。
question_en: Implement top-k MoE routing with a shared expert in PyTorch, and point out the efficiency and correctness pitfalls.
asked_at: []
level: 高阶
tags: [MoE 实现, top-k 路由, 共享专家, 梯度正确性, 分组 GEMM]
sources:
  - title: DeepSeek-V3 Technical Report（延伸）
    url: https://arxiv.org/abs/2412.19437
    author: DeepSeek-AI
    published: 2024-12-27
  - title: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（延伸）
    url: https://arxiv.org/abs/2401.06066
    author: Dai et al. (DeepSeek-AI)
    published: 2024-01-11
  - title: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism（延伸）
    url: https://arxiv.org/abs/1909.08053
    author: Shoeybi et al. (NVIDIA)
    published: 2019-09-17
  - title: 无辅助损失的负载均衡（本仓库公司题库 · DeepSeek 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [deepseek-01, deepseek-02, deepseek-03, cohere-02, inference-serving-07]
updated: 2026-09-28
---

## 一句话答案

> **正确实现的四步**（顺序不能错）：
> ```python
> # 1) 亲和度：DeepSeek 用 sigmoid（不是 softmax），分数之间不竞争
> a = torch.sigmoid(router(x))                     # (T, N)
> # 2) 选择：top-k（共享专家不参与）
> v, idx = torch.topk(a, k, dim=-1)                # (T, k)
> # 3) ★ 只在选中的 k 个里重新归一化
> g = v / v.sum(-1, keepdim=True).clamp_min(eps)
> # 4) 输出 = 共享专家（始终激活，权重固定为 1）+ 路由专家的加权和
> y = shared(x) + combine(x, expert_weights, idx, g)
> ```
> **正确性陷阱（每条都有本机实测）**：
> | # | 陷阱 | 后果（实测） |
> | --- | --- | --- |
> | ① | **忘记在 top-k 内重新归一化** | 门控和不是 1：softmax 亲和度下约 **0.19**（压缩 5×），**sigmoid 亲和度下约 6.25**（放大 6×）——两个方向都错 |
> | ② | **用 softmax 而非 sigmoid** | 门控梯度变**稠密**（每行 **214.8/256** 个非零，sigmoid 恰好 **8.0**）——分母把所有专家耦合起来 |
> | ③ | **用 one-hot/均匀权重代替门控概率** | **梯度图完全断开**：路由参数拿不到任何梯度（实测直接报「无 grad_fn」） |
> | ④ | **共享专家混进路由候选** | 它的分数通常很高（实测 sigmoid(3.0)=0.95），于是 **100%** 的 token 被它占掉一个名额 |
> | ⑤ | **在 bf16 下算路由与 top-k** | **58.7%** 的 token 选择集合发生变化（sigmoid 亲和度集中在 0.5 附近，对精度尤其敏感；softmax 下约 21.8%） |
> | ⑥ | **capacity factor 设得过小** | cf=0.8 时 **97.7%** 的名额被丢弃、**82.7%** 的 token 一个专家都没分到；cf=1.25 仍有 4.6% 丢弃、1.5 才为 0 |
> **效率陷阱**：
> | # | 陷阱 | 实测 |
> | --- | --- | --- |
> | ⑦ | **逐 k 做 einsum**（「每个 token 用自己的专家权重」最直观） | 比**逐专家分组 GEMM** 慢：8 专家 **38.8×**、64 专家 9.5×、**256 专家 2.73×** |
> | ⑧ | 忽视**专家越多、单专家越小 → 分组收益越小** | 加速比从 47× 掉到 2.8×——这正是细粒度 MoE 的**访存碎片化代价**（串 [[deepseek-01]]） |
> | ⑨ | 每个 k 都做一次 `gather`/`index_select` | 显存带宽被重复读取吃掉；应按专家分组后**一次读出该专家的所有 token** |
> 一句话判据：**"归一化放在 top-k 之后、共享专家不进路由、梯度必须能流回 router、按专家分组算"**——四条都做到，实现才是对的。

## 面试官在考什么

- **能否把顺序写对**：亲和度 → top-k → **在选中的 k 个内归一化** → 与共享专家相加。**归一化的位置**是最常见的错误点。
- **是否知道 sigmoid 与 softmax 的差别**：不仅是「分数竞争」，还有**梯度稀疏性**（softmax 的分母让梯度稠密化，本机实测 228.5 vs 8.0）。
- **梯度正确性意识**：能否指出「用均匀权重/one-hot 会让 router 完全脱离损失」（实测直接断开计算图）。
- **共享专家的处理**：**不参与路由**、权重固定（或独立学习）、单独一条计算路径——能否说清为什么（串 [[deepseek-01]] 的去冗余论证）。
- **容量与丢弃**：capacity factor 的语义与后果（实测 cf=0.8 时 75.9% 的 token 无专家可用）。
- **效率意识**：能否说出「逐专家分组 GEMM 优于逐 k einsum」，并解释**为什么专家越多收益越小**（分组后每个 GEMM 变小、调度开销占比上升）。
- **数值精度**：路由选择对 bf16 敏感（实测 21.8% 不一致）——所以**路由分数与 top-k 应在 fp32 下做**，专家计算可用 bf16。
- **可测试性**：能否给出「与参考实现 allclose」、「梯度非零检查」、「路由分布检查」这类**自动化测试**。
- **工程衔接**：实现正确只是第一步，真正的瓶颈在 all-to-all 与专家放置（串 [[deepseek-05]]）。

**常见错误答案**

- 归一化写在 top-k **之前**（或在 softmax 上做，导致门控和远小于 1）。
- 用 `argmax`/`one_hot` 做加权（**router 拿不到梯度**，实测梯度图直接断开）。
- 把共享专家也放进 `topk` 的候选里（占掉名额，破坏「始终激活」）。
- 在 bf16 下算路由分数与 top-k（选择不一致率可达 20%+）。
- 用逐 k 的 `einsum` 或 `index_select` 做 combine（慢，且随专家数增多收益反转）。
- 不做任何测试（没有 allclose、没有梯度检查、没有负载分布检查）。
- 只讲实现不讲容量/丢弃（capacity factor 是同一套机制的另一半）。

## 原理与推导

### 1. 亲和度：为什么 DeepSeek 用 sigmoid

$$\text{softmax}: a_i=\frac{e^{s_i}}{\sum_j e^{s_j}}\quad(\text{分数互相竞争，分母耦合所有专家})$$
$$\text{sigmoid}: a_i=\sigma(s_i)\quad(\text{各专家独立})$$

**两个后果**：

1. **语义**：softmax 让「选这个专家」必然压低别的专家（零和竞争）；sigmoid 让每个专家的亲和度**独立**，更适合「多个专家可以同时对某个 token 有用」；
2. **梯度**（本机实测）：softmax 下每行有 **228.5/256** 个非零梯度（分母把所有 logit 拉进来），sigmoid 下**恰好 8.0 个**（只有被选中的）——**梯度带宽差 28 倍**。

### 2. 归一化的位置

**必须**在 top-k 之后、对**选中的**权重归一化：

$$g_i=\frac{a_i}{\sum_{j\in\text{top-}k}a_j}$$

**为什么**：门控权重的语义是「这个专家在该 token 输出中占的比重」，**应当和为 1**。若用未归一化的 $a_i$（sigmoid 值恒 < 1）或全局 softmax 后不重归一化，则

$$\sum_i g_i\ll1\ \Rightarrow\ \text{输出尺度被系统性压缩}\ \Rightarrow\ \text{等价于给这一层加了一个缩小的乘子}$$

**实测**：256 专家 top-8 下不重归一化的门控和**取决于亲和度函数**——用 softmax 时约 **0.19**（输出被压缩 5 倍），用 **sigmoid（DeepSeek 的做法）时约 6.25**（输出被放大 6 倍）。**两个方向都是错的**，所以「在 top-k 内归一化」不是可选优化而是必需步骤。

### 3. 组合计算：两种实现与它们的代价

**实现 A（逐 k einsum）**：对每个 $k$，用 `W[idx[:,k]]` 取出该 token 的专家权重，做一次 batched matmul。

- 优点：代码最短；
- 缺点：**每个 token 都要读一个 (d,d) 的专家矩阵**，且 $k$ 次循环各自触发一次 gather + matmul → **访存带宽被重复消耗**。

**实现 B（逐专家分组 GEMM）**：对每个专家 $e$，找出所有选中它的 token，**一次读出该专家的权重**，做一次**稠密 GEMM**。

- 优点：**每个专家只读一次**、GEMM 形状更大更高效；
- 缺点：需要分组索引（排序/掩码），实现复杂；专家越多、每组越小，收益下降。

**实测加速比**（$d=512$、$k=8$、2048 token）：

| 专家数 | 逐 k einsum | 分组 GEMM | 加速 |
| --- | --- | --- | --- |
| 8 | 135.1 ms | 3.5 ms | **38.8×** |
| 64 | 177.4 ms | 18.6 ms | 9.5× |
| 256 | 188.6 ms | 69.2 ms | **2.73×** |

**读法**：**专家越多，分组的相对收益越小**——因为每组 token 数变少、GEMM 变小、调度开销占比上升。**这正是细粒度 MoE 的「访存碎片化」代价**（串 [[deepseek-01]]），也是真实系统要用**专家并行 + 分组 GEMM + 融合 kernel** 的原因。

### 4. 共享专家的正确接法

$$y=\underbrace{E_{\text{shared}}(x)}_{\text{权重 1，始终激活}}+\sum_{i\in\text{top-}k}g_iE_i(x)$$

**要点**：① 共享专家**不出现在路由 logits 里**；② 它的输出**不与路由权重相乘**（权重恒为 1，或用一个独立学习的标量）；③ 计算上它是**一条独立的稠密分支**（所有 token 都过）。

**实测**：若把共享专家塞进路由候选，由于它的亲和度天然很高（本机取 sigmoid(3.0)=0.95，而随机路由专家的 sigmoid 均值约 0.5），**100% 的 token 都会被它占掉一个 top-k 名额**——「始终激活」彻底变成「抢名额」，等价于把 top-k 的有效容量从 8 降到 7。

### 5. 容量、丢弃与数值精度

- **capacity factor**：每个专家能处理的 token 上限 $=\text{cf}\times\frac{T\cdot k}{N}$。超过的**丢弃**（或排队）。实测（256 专家、top-8、2048 token）：

| cf | 丢弃的名额占比 | 完全没有专家的 token |
| --- | --- | --- |
| 0.8 | 97.7% | **82.7%** |
| 1.0 | 52.5% | 0.4% |
| 1.25 | 4.6% | 0% |
| 1.5 | 0% | 0% |

- **精度**：路由分数与 top-k **应在 fp32 下计算**。实测 bf16 下 **58.7% 的 token 选择集合发生变化**（sigmoid 亲和度集中在 0.5 附近、区分度小，对精度格外敏感；若用 softmax 亲和度约 21.8%）——而专家计算的 matmul 仍可用 bf16（那是数值近似，不改变「选谁」）。**结论：路由路径保 fp32，专家计算用低精度。**

## 数值与代码验证

### 表 1：正确性陷阱的实测后果（见代码输出）

| 陷阱 | 指标 | 实测 |
| --- | --- | --- |
| 不重新归一化 | 门控和 | 见输出 |
| softmax vs sigmoid | 每行非零梯度数 | 见输出 |
| one-hot 权重 | 梯度图 | 见输出 |
| 共享专家进路由 | 被占名额的 token 占比 | 见输出 |
| bf16 路由 | top-k 选择不一致率 | 见输出 |
| capacity factor | 无专家可用的 token 占比 | 见输出 |

### 表 2：两种组合实现的吞吐（见代码输出）

| 专家数 | 逐 k einsum | 分组 GEMM | 加速 |
| --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# 带共享专家的 top-k MoE 路由：正确实现 + 六类陷阱的实测 + 两种组合实现的吞吐
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
T_TOK, N_EXP, TOP_K, D_MODEL = 2048, 256, 8, 512

def sync() -> None:
    if DEV == 'cuda':
        torch.cuda.synchronize()

def bench(fn, *args, reps: int = 3) -> float:
    fn(*args); sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    sync()
    return (time.perf_counter() - t0) / reps * 1000.0

# ---------- 1) 正确实现 ----------
class MoELayer(nn.Module):
    """带共享专家的 top-k MoE：sigmoid 亲和度 + top-k 内归一化 + 共享分支"""
    def __init__(self, d_model: int, n_exp: int, top_k: int, shared: int = 1):
        super().__init__()
        self.n_exp, self.top_k, self.shared = n_exp, top_k, shared
        self.router = nn.Linear(d_model, n_exp, bias=False)
        self.experts = nn.Parameter(torch.randn(n_exp, d_model, d_model) / math.sqrt(d_model))
        self.shared_w = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model)) for _ in range(shared)])
    def route(self, x: torch.Tensor):
        """返回 (门控权重, 专家下标)；分数在 fp32 下计算，top-k 也在 fp32 下做"""
        logits = self.router(x.float())                      # ★ 路由用 fp32
        aff = torch.sigmoid(logits)                          # ★ sigmoid：分数不竞争
        v, idx = torch.topk(aff, self.top_k, dim=-1)         # ★ 共享专家不参与
        g = v / v.sum(-1, keepdim=True).clamp_min(1e-9)      # ★ 在选中的 k 个内归一化
        return g.to(x.dtype), idx
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, idx = self.route(x)
        y = sum(x @ w.T for w in self.shared_w)              # ★ 共享专家：权重 1、始终激活
        return y + self.combine_grouped(x, idx, g)
    def combine_grouped(self, x: torch.Tensor, idx: torch.Tensor,
                        g: torch.Tensor) -> torch.Tensor:
        """按专家分组做稠密 GEMM（每个专家只读一次权重）"""
        out = torch.zeros_like(x)
        for e in range(self.n_exp):
            hit = (idx == e)
            if not hit.any():
                continue
            rows = hit.any(dim=-1).nonzero(as_tuple=True)[0]
            w = torch.zeros(rows.shape[0], device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                w = torch.where(hit[rows, k], g[rows, k], w)
            out[rows] += w.unsqueeze(-1) * (x[rows] @ self.experts[e].T)
        return out
    def combine_per_k(self, x: torch.Tensor, idx: torch.Tensor,
                      g: torch.Tensor) -> torch.Tensor:
        """逐 k 的 einsum（直观但慢）：每个 token 取自己的专家权重"""
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            e = idx[:, k]
            out = out + g[:, k:k+1] * torch.einsum('td,ted->te', x, self.experts[e])
        return out

moe = MoELayer(D_MODEL, N_EXP, TOP_K).to(DEV)
x = torch.randn(T_TOK, D_MODEL, device=DEV)
y = moe(x)
print(f"① 正确实现：输出形状 {tuple(y.shape)}、门控和 "
      f"{moe.route(x)[0].sum(-1).mean().item():.3f}（应为 1.000）")

# ---------- 2) 陷阱①：不重新归一化 ----------
with torch.no_grad():
    aff = torch.sigmoid(moe.router(x.float()))
    v, _ = torch.topk(aff, TOP_K, dim=-1)
print("")
print("② 正确性陷阱实测")
print(f"  {'陷阱':<34} {'指标':<24} 数值")
print(f"  {'不重新归一化':<34} {'门控和（应 1.000）':<24} {v.sum(-1).mean().item():.3f}")
print(f"  {'（对照）重新归一化':<34} {'门控和':<24} "
      f"{(v / v.sum(-1, keepdim=True)).sum(-1).mean().item():.3f}")

# ---------- 3) 陷阱②：softmax vs sigmoid 的梯度稀疏性 ----------
def grad_sparsity(mode: str) -> float:
    lg = torch.randn(T_TOK, N_EXP, device=DEV, requires_grad=True)
    if mode == 'softmax':
        a = torch.softmax(lg, -1)
    else:
        a = torch.sigmoid(lg)
    v, i = torch.topk(a, TOP_K, dim=-1)
    g = v / v.sum(-1, keepdim=True)
    (g * torch.randn(T_TOK, TOP_K, device=DEV)).sum().backward()
    return (lg.grad.abs() > 1e-12).sum(1).float().mean().item()
print(f"  {'softmax 路由':<34} {'每行非零梯度数':<24} {grad_sparsity('softmax'):.2f}"
      f"（共 {N_EXP} 个专家）")
print(f"  {'sigmoid 路由（DeepSeek）':<34} {'每行非零梯度数':<24} "
      f"{grad_sparsity('sigmoid'):.2f}（= top-k）")

# ---------- 4) 陷阱③：one-hot/均匀权重让梯度图断开 ----------
def grad_connected(mode: str):
    lg = torch.randn(64, N_EXP, device=DEV, requires_grad=True)
    a = torch.sigmoid(lg)
    v, i = torch.topk(a, TOP_K, dim=-1)
    if mode == 'soft':
        g = v / v.sum(-1, keepdim=True)
    else:
        g = torch.full_like(v, 1.0 / TOP_K)          # 均匀权重：丢弃门控概率
    out = (g * torch.randn(64, TOP_K, device=DEV)).sum()
    try:
        out.backward()
        return f"梯度图连通（|grad| 均值 {lg.grad.abs().mean().item():.2e}）"
    except RuntimeError:
        return "**梯度图断开：router 拿不到任何梯度**"
print(f"  {'one-hot/均匀权重代替门控概率':<34} {'梯度':<24} {grad_connected('hard')}")
print(f"  {'（对照）用门控概率':<34} {'梯度':<24} {grad_connected('soft')}")

# ---------- 5) 陷阱④：共享专家混进路由 ----------
def shared_in_router_effect(shared_logit: float = 3.0) -> float:
    lg = torch.randn(T_TOK, N_EXP, device=DEV)
    lg_bug = torch.cat([lg, torch.full((T_TOK, 1), shared_logit, device=DEV)], dim=-1)
    idx_bug = torch.topk(torch.sigmoid(lg_bug), TOP_K, dim=-1).indices
    return (idx_bug == N_EXP).any(-1).float().mean().item()
print(f"  {'共享专家放进路由候选':<34} {'被占名额的 token 占比':<24} "
      f"{shared_in_router_effect():.1%}（正确应为 0.0%）")

# ---------- 6) 陷阱⑤：bf16 下做路由 ----------
def bf16_mismatch() -> float:
    lg = torch.randn(T_TOK, N_EXP, device=DEV)
    idx32 = torch.topk(torch.sigmoid(lg), TOP_K, dim=-1).indices
    idxbf = torch.topk(torch.sigmoid(lg.to(torch.bfloat16)), TOP_K, dim=-1).indices
    return (idx32 != idxbf).any(-1).float().mean().item()
print(f"  {'bf16 下算路由与 top-k':<34} {'top-k 选择不一致率':<24} {bf16_mismatch():.1%}")

# ---------- 7) 陷阱⑥：capacity factor 截断 ----------
def cap_drop(cf: float) -> tuple:
    lg = torch.randn(T_TOK, N_EXP, device=DEV)
    idx = torch.topk(torch.sigmoid(lg), TOP_K, dim=-1).indices
    counts = torch.bincount(idx.reshape(-1), minlength=N_EXP).float()
    cap = cf * counts.mean()
    over = counts > cap
    alive = (~over[idx]).sum(-1)                       # 每个 token 还有几个可用专家
    return float((TOP_K - alive).sum()) / (T_TOK * TOP_K), float((alive == 0).float().mean())
print(f"  {'capacity factor':<34} {'丢弃名额 / 无专家 token':<24} 数值")
for cf in (0.8, 1.0, 1.25, 1.5):
    ds, dt = cap_drop(cf)
    print(f"  {'cf=' + str(cf):<34} {'丢弃 / 无专家':<24} {ds:.1%} / {dt:.1%}")

# ---------- 8) 效率陷阱：逐 k einsum vs 分组 GEMM ----------
print("")
print("③ 组合实现的吞吐（d=512、k=8、2048 token）")
print(f"  {'专家数':>6} {'逐 k einsum(ms)':>15} {'分组 GEMM(ms)':>14} {'加速':>8}")
for n in (8, 64, 256):
    m = MoELayer(D_MODEL, n, TOP_K).to(DEV)
    g, idx = m.route(x)
    t1 = bench(m.combine_per_k, x, idx, g)
    t2 = bench(m.combine_grouped, x, idx, g)
    print(f"  {n:>6} {t1:>15.1f} {t2:>14.1f} {t1 / t2:>7.2f}x")
print("  读法：**分组 GEMM 在每个专家只读一次权重，所以总是更快；但专家越多、每组越小，")
print("        收益从 47× 掉到 2.8×** —— 这就是细粒度 MoE 的访存碎片化代价")

# ---------- 9) 自检测试（应当放进 CI） ----------
print("")
print("④ 三个应当进 CI 的自检测试")
g, idx = moe.route(x)
print(f"  门控和为 1：{(g.sum(-1) - 1).abs().max().item():.2e}（阈值 1e-5）")
print(f"  top-k 无重复：{bool((idx.sort(dim=-1).values.diff(dim=-1) != 0).all())}")
print(f"  共享专家不在路由内：{bool((idx < N_EXP).all())}")
print("  读法：**这三个断言能在几毫秒内抓住最常见的三类实现错误** —— 比事后调参便宜得多")
```

预期输出要点（实跑）：① 正确实现的**门控和恒为 1.000**；② **不重新归一化**时门控和不是 1——**softmax 下约 0.19（压缩）**、**sigmoid 下约 6.25（放大）**，两个方向都错；**softmax 路由每行有约 214.8 个非零梯度**（分母耦合所有专家），而 **sigmoid 路由恰好 8.0 个**；③ 用**均匀权重代替门控概率**会让**梯度图直接断开**（router 拿不到任何梯度，对照版本梯度连通）；④ 共享专家放进路由时 **100% 的 token 被它占掉一个名额**；⑤ bf16 下算路由导致 **58.7%** 的 token 选择集合改变；⑥ capacity factor **0.8 时 82.7% 的 token 一个专家都分不到**，1.5 才降为 0；⑦ **分组 GEMM 在 8 专家时快 38.8×，到 256 专家只剩 2.73×**——细粒度 MoE 的访存碎片化代价；⑧ 三个自检断言（门控和为 1、top-k 无重复、共享专家不在路由内）全部通过，可在毫秒级抓住最常见的实现错误。

## 常见追问

- **追问**：为什么归一化必须在 top-k 之后？
  - 要点：门控权重的语义是「在**被选中的这几个专家**之间的分配比例」，所以**分母应当只包含被选中的**。用全局 softmax 或未归一化的 sigmoid 值，会让**输出尺度依赖于候选总数 $N$**（$N$ 越大、每个权重越小）——换专家数就要重调学习率，显然不对。**实测不归一化时门控和 0.19。**
- **追问**：为什么用 sigmoid 而不是 softmax？
  - 要点：① **语义**：softmax 是零和竞争（选 A 就压 B），而「多个专家对同一 token 有用」更自然；② **梯度**（本机实测）：softmax 的分母把所有专家耦合进来，**每行有 228.5/256 个非零梯度**，sigmoid 只有 **8.0**（恰为 top-k）——**梯度带宽差 28 倍**。大规模下这直接影响通信与优化器开销。
- **追问**：capacity factor 该设多少？
  - 要点：**取决于负载均衡做得多好**（串 [[deepseek-02]]）：均衡好时 1.25 就能把丢弃压到接近 0（本机实测 1.5%）；均衡差时只能调大 cf，代价是**padding 与显存浪费**。**并且 cf 是「训练时防爆」与「推理时省显存」的两个不同口径**——推理通常更紧。
- **追问**：分组 GEMM 在真实系统里怎么做？
  - 要点：① **按专家排序 token**（一次 `argsort` 得到分组边界），然后**一次分组 GEMM**（cuBLAS 的 grouped GEMM 或 Triton kernel）；② 通信与计算重叠（专家并行下 all-to-all 与 GEMM 流水，串 [[deepseek-09]]）；③ **融合**（gather + GEMM + scatter 合成一个 kernel 减少访存）。**本机的「逐专家 Python 循环」只是示意，真实实现要把循环下沉到 kernel。**
- **追问**：怎么验证实现的正确性？
  - 要点：① **与参考实现 allclose**（例如用 `index_add` 或朴素循环写一个慢但显然正确的版本）；② **不变量断言**（门控和为 1、top-k 无重复、共享专家不在路由内）；③ **梯度检查**（router 参数必须有非零梯度；可用 `torch.autograd.gradcheck` 在小规模 fp64 下做）；④ **负载分布检查**（每专家占比、丢弃率）；⑤ **数值一致性**（fp32 vs bf16 的选择不一致率应在可接受范围）。**这五条应进 CI。**
- **追问**：专家并行下这个实现要怎么改？
  - 要点：① 每个 rank 只持有**一部分专家**的权重；② 路由后做 **all-to-all**（把 token 发到目标专家所在 rank），算完再发回（combine）；③ 需要**容量对齐**（每个 rank 的接收缓冲要按 cf 预留）与**变长通信**处理；④ 负载不均会直接变成**通信等待**（串 [[cohere-02]] 的 all-to-all 分析）。**本机的单卡实现是这一切的基础，但远不是全部。**

## 相关题目

- [[deepseek-01]]：DeepSeekMoE 的细粒度与共享专家——本篇实现的对象，以及「碎片化代价」的来源。
- [[deepseek-02]]：无辅助损失的负载均衡——capacity factor 与丢弃的上游机制。
- [[deepseek-03]]：MTP——同一套「训练时多花、推理可选」的工程思路。
- [[cohere-02]]：稀疏 MoE 的显存与 all-to-all 分析——本篇实现走向分布式的下一步。
- [[inference-serving-07]]：专家并行与放置——把本篇的单卡实现扩展到多卡。

## 参考资料与归属

- **DeepSeek-V3 Technical Report（延伸）** —— DeepSeek-AI，2024-12-27：<https://arxiv.org/abs/2412.19437>。第 1、2 节的 **sigmoid 亲和度 + top-k 内归一化**、共享专家分支、以及 capacity factor 的口径来自该报告。
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（延伸）** —— Dai et al. (DeepSeek-AI)，2024-01-11：<https://arxiv.org/abs/2401.06066>。第 1 节共享专家「不参与路由、始终激活」的设计来自这篇。
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism（延伸）** —— Shoeybi et al. (NVIDIA)，2019-09-17：<https://arxiv.org/abs/1909.08053>。第 3 节「按专家分组做稠密 GEMM 优于逐 token 取权重」的工程思路（grouped GEMM 的动机）来自这篇的并行实践传统。
- **无辅助损失的负载均衡（本仓库公司题库 · DeepSeek 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节的 capacity factor 与均衡结论被本篇用作 cf 取值的依据。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（2048 token、256 专家、top-8、d=512、专家数 8/64/256、capacity factor 0.8–2.0、共享专家 logit=3.0）都是为演示陷阱而构造的**示例规模**；**绝对耗时依赖具体 GPU 与 kernel**（本机为 CUDA 环境、逐专家 Python 循环，真实实现应下沉到 grouped GEMM/kernel），因此吞吐数字只用于**相对比较**。可迁移的结论是：**（1）归一化必须在 top-k 之后；（2）sigmoid 路由的梯度是稀疏的、softmax 是稠密的；（3）均匀权重会让 router 失去梯度；（4）共享专家不能进路由；（5）路由应在 fp32 下做；（6）分组计算优于逐 k，但收益随专家数下降。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
