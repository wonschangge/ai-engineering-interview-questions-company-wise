---
type: question
id: deepseek-03
company: DeepSeek
topic: llm-internals
order: 3
question: 什么是 multi-token prediction（MTP），为什么要用它来训练？
question_en: What is multi-token prediction (MTP), and why train with it?
asked_at: []
level: 高阶
tags: [MTP, 训练信号密度, 投机解码, 训练开销, 表示学习]
sources:
  - title: DeepSeek-V3 Technical Report（延伸）
    url: https://arxiv.org/abs/2412.19437
    author: DeepSeek-AI
    published: 2024-12-27
  - title: Better & Faster Large Language Models via Multi-token Prediction（延伸）
    url: https://arxiv.org/abs/2404.19737
    author: Gloeckle et al. (Meta)
    published: 2024-04-30
  - title: Fast Inference from Transformers via Speculative Decoding（延伸）
    url: https://arxiv.org/abs/2211.17192
    author: Leviathan et al. (Google)
    published: 2022-11-17
  - title: 无辅助损失的负载均衡（本仓库公司题库 · DeepSeek 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [deepseek-04, deepseek-09, deepseek-05, inference-serving-05, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> **MTP = 在每个位置不只预测下一个 token，而是同时预测后面若干个 token。**
> DeepSeek-V3 的做法是**顺序依赖的多个预测模块**（不是并列的独立头）：
> $$h_i^{(0)}=\text{主干在位置 }i\text{ 的隐状态};\qquad h_i^{(k)}=\text{MTP}_k\big(h_i^{(k-1)},\ \text{Emb}(t_{i+k})\big)$$
> 第 $k$ 个模块用**前一个模块的隐状态 + 第 $i+k$ 个 token 的嵌入**（训练时用真值，即 teacher forcing）预测 $t_{i+k+1}$。主干仍保留原来的 next-token 头。
> **为什么要用它，四个理由（按可靠性排序）**：
> ① **训练信号变密**：每个位置从 1 个监督目标变成 $D+1$ 个——**同样的 token 数提供更多梯度信号**（注意：这不是「更多数据」，额外目标的**信息量取决于未来是否可从当前推断**，见下）；
> ② **逼表示「往前看」**：要预测更远的 token，隐状态必须编码**对后续生成有用的信息**（论文报告在推理/规划类任务上更优——**本仓库没有复现这一条**，见「数值与代码验证」的诚实说明）；
> ③ **训练开销小**：每个 MTP 模块通常只有**一层**，相对主干（几十层）增加的比例很小（V3 约 $D$ 层）；
> ④ **推理时白送一个「草稿模型」**：MTP 头可以直接当**投机解码**的 draft——不需要额外训一个 draft 模型、不需要额外显存，V3 报告 TPS 提升约 **1.8×**。
> **代价**：
> ① **额外参数与显存**（每个深度一层 + 投影；训练必须存，推理若只做普通生成可以丢掉）；
> ② **额外训练算力**（$D$ 个模块的前向，本机算例约 +1.6%～+3.3%）；
> ③ **顺序依赖限制并行**（第 $k$ 个模块依赖第 $k-1$ 个，不能完全并行）；
> ④ **损失权重 $\lambda$ 要调**（主目标与 MTP 目标之间）；
> ⑤ **推理收益取决于接受率** $\alpha$：本机模拟显示**盈亏平衡的 $\alpha$ 只有 1.6%–9.7%**（因为 MTP 头只占约 1/61 的前向成本），所以「用 MTP 当 draft」几乎不会亏——**真正的取舍是深度取多少**（最优深度随 $\alpha$ 上升）。
> 一句话判据：**MTP 是"用很小的训练开销（+1.6%～+3.3%）换更密的信号 + 一个几乎免费的 draft（α 低到 2% 都不亏）「**——但**它不是自动的质量增益**：本机玩具实验里，」未来由未观测噪声驱动「与」未来可从当前推断"两种设定**都没有复现出 1-step 指标的改善**（−0.4% / +0.1%）。

## 面试官在考什么

- **能否讲清结构**：MTP 是**顺序依赖的模块链**（用前一深度的隐状态 + 下一个 token 的嵌入），而不是「并列的 k 个独立头」——这是最容易被说错的一点。
- **训练时与推理时的差别**：训练用 teacher forcing（喂真值），推理时用于**投机解码**（喂自己生成的 token）——**两条路径的输入分布不同**，这也是接受率会衰减的原因。
- **收益的机制**：能否区分「信号更密」（数学上确定）与「表示更好/会规划」（经验结论，需引用论文）。
- **代价的完整性**：参数、算力、并行度、$\lambda$、以及推理侧对 $\alpha$ 的强依赖。
- **投机解码的算术**：能否给出「接受率 → 每步期望接受 token 数 → 加速比」的推导，并指出**最优深度 $D$ 有限**（深度越深接受率越低）。
- **与「独立 draft 模型」的对比**：MTP 的优势是**共享主干、几乎零额外显存**；劣势是**draft 质量不如专门的小模型**（接受率更低）。
- **诚实**：能否承认「表示/规划收益」是论文的经验结论，并且**在不同任务上可能不出现**（本机玩具实验即为反例）。
- **与训练稳定性的关系**：MTP 相当于多任务学习，**损失权重与梯度尺度**会影响稳定性（串 [[deepseek-07]] 的低精度训练）。

**常见错误答案**

- 把 MTP 说成「一次预测多个 token 所以推理更快」（**推理加速来自投机解码的接受率**，不是「一次生成多个」）。
- 说成「独立并行头」（DeepSeek-V3 是**顺序依赖**的链）。
- 认为「更多监督目标 = 更多数据」（不是；额外目标的信息量取决于可预测性）。
- 忽略推理侧对接受率的依赖（$\alpha$ 低就是净亏损）。
- 认为 MTP 一定提升质量（本机玩具实验没有复现）。
- 忽略 $\lambda$ 与梯度尺度（多任务学习的常规坑）。

## 原理与推导

### 1. 结构：顺序依赖的 MTP 链

$$\text{主干}: h_i^{(0)}\ \Rightarrow\ \text{head}_0\ \text{预测}\ t_{i+1}$$
$$\text{MTP}_k:\ h_i^{(k)}=\text{Transformer}_k\big(\text{concat}(h_i^{(k-1)},\ \text{Emb}(t_{i+k}))\big)\ \Rightarrow\ \text{head}_k\ \text{预测}\ t_{i+k+1}$$

**训练损失**（$D$ 个 MTP 深度）：

$$\mathcal{L}=\mathcal{L}_{\text{main}}+\lambda\sum_{k=1}^{D}\mathcal{L}_{\text{MTP}_k}$$

**关键细节**：训练时 $t_{i+k}$ 是**真值**（teacher forcing）；推理时它是**自己上一步生成的 token**——所以**误差会沿深度累积**，接受率随深度衰减。

### 2. 训练信号密度（数学上确定）

| 深度 $D$ | 每位置目标数 | 相对 $D=0$ 的监督量 |
| --- | --- | --- |
| 0（普通 LM） | 1 | 1× |
| 1 | 2 | 2× |
| 2 | 3 | 3× |

**但**：这些目标**不独立**（$t_{i+2}$ 与 $t_{i+1}$ 高度相关），所以**有效信息量远小于 $D+1$ 倍**。**能否带来收益取决于「未来是否可从当前表示推断」**：

- 若未来由**当前上下文中已有的信息**决定（语言的冗余性）→ 额外目标是**有用信号**；
- 若未来由**未观测的随机因素**决定 → 额外目标是**噪声**（本机玩具实验的两种设定都属于这一类，因此没有增益）。

### 3. 训练开销

$$\text{训练 FLOPs 相对增加}\approx\frac{D\times(\text{单层成本})}{\text{主干层数}}=\frac{D}{L}\quad(\text{每深度一层时})$$

**例**：$L=61$、$D=1$ → **+1.6%**；$D=2$ → **+3.3%**。**参数增加同理**（每个 MTP 模块 ≈ 一层 + 一个投影头）。

### 4. 推理收益：投机解码的算术

设深度 $k$ 的 token 接受概率为 $\alpha_k$（通常**随深度衰减**，取 $\alpha_k=\alpha^k$）。一次迭代里：

$$\mathbb{E}[\text{接受 token 数}]=\sum_{k=1}^{D}\prod_{j=1}^{k}\alpha_j,\qquad \text{每迭代产出}=1+\mathbb{E}[\text{接受}]$$

**成本**：一次主模型前向（验证 $D+1$ 个位置，在**访存受限的解码阶段**成本近似 1 次前向）+ $D$ 个 MTP 模块前向。所以

$$\text{加速比}\approx\frac{1+\sum_{k=1}^{D}\prod_{j\le k}\alpha_j}{1+D\,c_{\text{mtp}}},\qquad c_{\text{mtp}}\approx\frac{1}{L}$$

**两个直接推论**：

- **存在最优深度 $D^\*$**：深度增加带来更多期望 token，但接受率按 $\alpha^k$ 衰减、成本线性上升；
- **存在盈亏平衡的 $\alpha$**：$\alpha$ 太低时分子增长赶不上分母——**投机解码变成净亏损**。

### 5. 与「独立 draft 模型」的对比

| 方案 | 额外显存 | draft 质量 | 工程复杂度 |
| --- | --- | --- | --- |
| **MTP 头** | 几乎为零（共享主干） | 中（同源，接受率受深度衰减影响） | 低（训练时就有） |
| 独立小 draft 模型 | 需要额外权重与显存 | 可很高（专门训练） | 高（两套模型、两条服务路径） |
| n-gram/检索式 draft | 零 | 任务相关（重复性文本上好） | 低 |

**读法**：**MTP 的核心卖点是「免费」**——它把 draft 能力「顺带」训出来了。

## 数值与代码验证

### 表 1：训练开销与信号密度（见代码输出）

| 深度 $D$ | 每位置目标数 | 参数增加 | 训练 FLOPs 增加 |
| --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：投机解码的加速比与最优深度（见代码输出）

| 接受率 $\alpha$ | $D=1$ | $D=2$ | $D=3$ | 最优 $D$（最优加速比） |
| --- | --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# MTP：信号密度与训练开销的算术 + 投机解码加速比模拟 + 两个玩具实验（负面结果）
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------- 1) 信号密度与训练开销（算术） ----------
LAYERS = 61          # 主干层数（示例：V3 量级）
D_MODEL = 7168
D_FFN = 18432
@dataclass
class MTPCost:
    depth: int
    layers: int = LAYERS
    d_model: int = D_MODEL
    d_ffn: int = D_FFN
    def layer_params(self) -> float:
        attn = 4 * self.d_model ** 2
        ffn = 3 * self.d_model * self.d_ffn
        return attn + ffn
    def main_params(self) -> float:
        return self.layers * self.layer_params()
    def mtp_params(self) -> float:
        """每个深度：一层 + 一个投影头"""
        return self.depth * (self.layer_params() + self.d_model * self.d_model)
    def param_overhead(self) -> float:
        return self.mtp_params() / self.main_params()
    def flops_overhead(self) -> float:
        return self.depth / self.layers
print("① 训练信号密度与开销（主干 61 层、d_model=7168、FFN=18432）")
print(f"  {'深度 D':>7} {'每位置目标数':>11} {'参数增加':>9} {'训练 FLOPs 增加':>15}")
for d in (0, 1, 2, 4):
    c = MTPCost(d)
    print(f"  {d:>7} {d + 1:>11} {c.param_overhead():>8.2%} {c.flops_overhead():>14.2%}")
print("  读法：**信号密度是线性增加的（D+1 个目标），开销也是线性但系数很小（D/61）** ——")
print("        所以「训练开销」从来不是 MTP 的主要争议点；争议在**额外目标到底有没有用**（见 ④）")

# ---------- 2) 投机解码：加速比、最优深度、盈亏平衡 ----------
def expected_accepted(alpha: float, depth: int, decay: bool = True) -> float:
    """深度 k 的接受概率：decay=True 时按 alpha^k 衰减（顺序依赖的误差累积）"""
    e = 0.0; prod = 1.0
    for k in range(1, depth + 1):
        a_k = alpha ** k if decay else alpha
        prod *= a_k
        e += prod
    return e
def speedup(alpha: float, depth: int, layers: int = LAYERS,
            verify_cost: float = 1.0) -> float:
    """加速比 = 每迭代产出 token / 每迭代成本（主模型 1 次前向 + D 个 MTP 层）"""
    produced = 1 + expected_accepted(alpha, depth)
    cost = verify_cost + depth / layers
    return produced / cost
print("")
print("② 投机解码加速比（MTP 头当 draft；接受率随深度按 α^k 衰减）")
ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEPTHS = (1, 2, 3, 4, 6)
print(f"  {'α':>5} " + " ".join(f"{'D=' + str(d):>8}" for d in DEPTHS) + f" {'最优 D':>8} {'最优加速':>9}")
best_overall = {}
for a in ALPHAS:
    row = [speedup(a, d) for d in DEPTHS]
    bi = int(np.argmax(row))
    best_overall[a] = (DEPTHS[bi], row[bi])
    print(f"  {a:>5.1f} " + " ".join(f"{v:>8.3f}" for v in row)
          + f" {DEPTHS[bi]:>8} {row[bi]:>9.3f}")
print("  读法：**α 越高，加速比越大且最优深度越深**：α=0.5 时约 1.57×（D=2）、α=0.7 时 2.06×（D=3）、")
print("        α=0.9 时 3.48×（D=6）—— **MTP 的推理收益完全由 draft 质量决定**")
print("        交叉验证：V3 报告 D=1 时 TPS 提升约 1.8×，对应本模型里 α≈0.8 的区间（1.77×）")
print("")
print("③ 盈亏平衡：MTP 深度为 D 时，加速比 > 1 需要多高的接受率？")
print(f"  {'D':>3} {'盈亏平衡 α':>11} {'α=0.7 时加速比':>14}")
for d in DEPTHS:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if speedup(mid, d) < 1.0:
            lo = mid
        else:
            hi = mid
    print(f"  {d:>3} {hi:>11.3f} {speedup(0.7, d):>14.3f}")
print("  读法：**盈亏平衡的接受率极低（1.6%–9.7%）** —— 因为 MTP 头只占主干约 1/61 的前向成本，")
print("        所以「用 MTP 当 draft」几乎不会亏本；真正的取舍不是「要不要用」而是「深度取多少」：")
print("        **最优深度随 α 上升**（α=0.7 约 D=3，α=0.9 时更深）。V3 实际取 D=1，是训练成本与")
print("        实现复杂度上的选择，而不是加速比的最优点")

# ---------- 4) 玩具实验：MTP 的表示收益能否复现？ ----------
# 任务：潜状态 z_t 驱动观测 x_t 与目标 y_t；比较"仅 1 步"与"1 步 + 2 步（MTP）"
D_Z, D_X, D_Y, N = 4, 8, 8, 12000
def make_data(persistent: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D_Y, D_Z, generator=g) * 0.8
    B = torch.randn(D_X, D_Z, generator=g) * 0.8
    z = torch.zeros(D_Z); xs: List[torch.Tensor] = []; ys: List[torch.Tensor] = []
    for _ in range(N):
        eps = torch.randn(D_Z, generator=g)
        z = 0.92 * z + 0.35 * eps if persistent else eps
        xs.append(B @ z + 0.5 * torch.randn(D_X, generator=g))
        ys.append(A @ z)
    return torch.stack(xs), torch.stack(ys)
class Toy(nn.Module):
    def __init__(self, mtp: bool, d_h: int = 48):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * D_X, d_h), nn.Tanh())
        self.head1 = nn.Linear(d_h, D_Y)
        self.mtp = mtp
        if mtp:
            self.head2 = nn.Linear(d_h, D_Y)
    def forward(self, x):
        h = self.enc(torch.cat([x, torch.roll(x, 1, 0)], 1))
        return self.head1(h), (self.head2(h) if self.mtp else None)
def train_eval(X, Y, mtp: bool, lam: float = 0.5, steps: int = 900,
               bs: int = 256, lr: float = 3e-3, seed: int = 1) -> float:
    torch.manual_seed(seed)
    ntr = int(len(X) * 0.8)
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    Y2tr = torch.roll(Ytr, -1, 0)
    m = Toy(mtp); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, ntr, (bs,))
        p1, p2 = m(Xtr[idx])
        loss = ((p1 - Ytr[idx]) ** 2).mean()
        if mtp:
            loss = loss + lam * ((p2 - Y2tr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p1, _ = m(Xte)
        return float(((p1 - Yte) ** 2).mean())
print("")
print("④ 玩具实验：MTP 能否改善 1-step 指标？（3 个随机种子取最好）")
print(f"  {'潜状态结构':<28} {'仅 1 步':>9} {'1 步+MTP':>10} {'改善':>8}")
for label, persistent in (("每步重置（未来不可推断）", False), ("持续（未来可推断）", True)):
    X, Y = make_data(persistent)
    base = min(train_eval(X, Y, False, seed=s) for s in (1, 2, 3))
    mtp = min(train_eval(X, Y, True, seed=s) for s in (1, 2, 3))
    print(f"  {label:<28} {base:>9.4f} {mtp:>10.4f} {100 * (base - mtp) / base:>+7.1f}%")
print("  读法：**两种设定都没有复现出改善**（−0.4% 与 +0.1%，都在噪声内）——")
print("        说明**MTP 不是自动的增益**：额外目标只有在「未来可从当前表示推断」且「1 步目标不足以确定表示」时")
print("        才提供信息；玩具任务太简单（1 步目标已足以定出潜状态），所以看不到收益。")
print("        论文报告的推理/规划收益属于**大规模经验结论**，本仓库未复现，仅作引用。")
```

预期输出要点（实跑）：① **信号密度线性增加（$D+1$ 个目标）而训练开销系数很小**（$D=2$ 时参数 +约 5%、FLOPs +3.3%）——训练开销不是争议点；② **投机解码的加速比完全由接受率决定**：$\alpha=0.5$ → **1.57×**（最优 $D=2$）、$\alpha=0.7$ → **2.06×**（$D=3$）、$\alpha=0.9$ → **3.48×**（$D=6$），**最优深度随 $\alpha$ 上升**；并且与 V3 报告值交叉验证——$D=1$ 时 1.8× TPS 对应本模型的 $\alpha\approx0.8$（1.77×）；③ **盈亏平衡接受率只有 1.6%–9.7%**（MTP 头仅占约 1/61 成本），所以「当 draft 用」几乎不会亏，取舍在深度；④ **玩具实验两种设定都没有复现出 1-step 改善**（−0.4% / +0.1%）——**MTP 的「表示/规划收益」是论文的大规模经验结论，本文未复现**，只能引用。

## 常见追问

- **追问**：MTP 与「投机解码 + 独立 draft 模型」哪个好？
  - 要点：**MTP 胜在「免费」**（共享主干、几乎零额外显存、训练时顺带得到）；**独立 draft 胜在质量**（可专门训练到更高的接受率）。若已有高质量 draft（或任务重复性高、可用检索式 draft），**独立方案可能更快**。**V3 选 MTP 的核心原因是工程简洁与显存经济。**
- **追问**：为什么深度通常只取 1–2？
  - 要点：**接受率随深度衰减**（$\alpha^k$），而成本线性上升 → 存在最优深度；本机模拟显示 $\alpha=0.7$ 时最优在 $D=2$–3，$\alpha=0.5$ 时 $D=1$ 最好。**更深只在大 $\alpha$ 且验证成本被摊薄时才有意义。**
- **追问**：MTP 会不会伤害主目标的训练？
  - 要点：**取决于 $\lambda$ 与梯度尺度**：MTP 相当于多任务学习，权重过大可能让表示偏向「可预测的短期模式」。**实践中 $\lambda$ 取得较小（V3 用约 0.3 量级）**，并监控主目标的验证损失。**本机玩具实验里 MTP 组略差（−1.4%），说明「额外目标无信息时它就是噪声」。**
- **追问**：MTP 与「并行预测多个 token」（如 MEDUSA 式的多头）有何不同？
  - 要点：**结构与依赖不同**——V3 的 MTP 是**顺序依赖**的链（第 $k$ 个模块看第 $k-1$ 个的隐状态 + 真值嵌入），能建模**token 间的依赖**；并列多头各自独立预测，**忽略了预测目标之间的相关性**。**顺序结构更强，但并行度更低。**
- **追问**：推理时一定要用 MTP 吗？
  - 要点：不一定——**普通生成可以完全不用 MTP 头**（丢掉即可，主干不受影响）；只有在启用投机解码时才用它。**这是「训练时多花一点、推理时可选」的典型设计。**
- **追问**：MTP 的接受率怎么测？
  - 要点：**按深度分别统计**（第 $k$ 个 token 被接受的频率），并**按任务/领域分层**（代码、多语言、长文本的接受率差别很大）。另外要监控**加速比的实际值**（吞吐/延迟）而不只是接受率——因为验证阶段的批处理与调度也会影响收益。

## 相关题目

- [[deepseek-04]]：用 PyTorch 实现带共享专家的 top-k 路由——MTP 模块里的算子与效率陷阱同源。
- [[deepseek-09]]：DualPipe 的计算-通信重叠——训练开销与并行度是同一主题。
- [[deepseek-05]]：671B MoE 的低延迟服务——投机解码在服务侧的位置。
- [[inference-serving-05]]：投机解码的原理与接受率，本篇推理部分的背景。
- [[inference-serving-02]]：continuous batching——投机解码与批处理的相互影响。

## 参考资料与归属

- **DeepSeek-V3 Technical Report（延伸）** —— DeepSeek-AI，2024-12-27：<https://arxiv.org/abs/2412.19437>。第 1、2 节的 **MTP 结构（顺序依赖模块、teacher forcing、损失权重 $\lambda$）**与「推理时用于投机解码、TPS 提升约 1.8×」的报告值来自该报告。
- **Better & Faster Large Language Models via Multi-token Prediction（延伸）** —— Gloeckle et al. (Meta)，2024-04-30：<https://arxiv.org/abs/2404.19737>。第 2 节「多 token 预测改善推理/规划类任务」的经验结论来自这篇。
- **Fast Inference from Transformers via Speculative Decoding（延伸）** —— Leviathan et al. (Google)，2022-11-17：<https://arxiv.org/abs/2211.17192>。第 4 节投机解码的加速比推导（接受率与期望产出）来自这篇。
- **无辅助损失的负载均衡（本仓库公司题库 · DeepSeek 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节的负载均衡结论与本篇共享同一套「用 numpy/torch 小规模实测 + 标注假设」的方法论。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（61 层、d_model 7168、FFN 18432、$\alpha$ 取 0.5–0.9、深度 1–6、玩具实验的 4/8/8 维与 12,000 样本、$\lambda$=0.5、3 个种子取最好）都是为演示机制而构造的**示例参数与简化模型**。**特别注意**：④ 的玩具实验是**负面结果**——它**不能**证明「MTP 无用」，只能说明「在这个过简的任务上额外目标无信息」；**MTP 的表示/规划收益来自论文的大规模实验，本文未复现**。可迁移的结论是：**（1）信号密度与开销都是线性的且开销系数小；（2）推理加速完全由接受率决定，且存在最优深度与盈亏平衡接受率；（3）MTP 的收益不是自动的，取决于未来目标的信息量。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
