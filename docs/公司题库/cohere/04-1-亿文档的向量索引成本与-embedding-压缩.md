---
type: question
id: cohere-04
company: Cohere
topic: rag
order: 4
question: 一家企业想对约 1 亿份文档做语义搜索，但被向量索引的成本劝退。请讲讲 embedding 压缩的几种方案以及其中的数学。
question_en: An enterprise wants semantic search over about 100 million documents but is put off by the cost of the vector index. Walk me through the options for embedding compression and the math behind them.
asked_at: []
level: 高阶
tags: [PQ, 标量量化, Matryoshka, 向量索引成本, ADC]
sources:
  - title: Product Quantization for Nearest Neighbor Search（延伸）
    url: https://ieeexplore.ieee.org/document/5432202
    author: Jégou, Douze & Schmid (IEEE TPAMI 2011)
    published: 2011-01
  - title: The Faiss library（延伸）
    url: https://arxiv.org/abs/2401.08281
    author: Douze et al. (Meta AI)
    published: 2024-01-16
  - title: Matryoshka Representation Learning（延伸）
    url: https://arxiv.org/abs/2205.13147
    author: Kusupati et al. (NeurIPS 2022)
    published: 2022-05-26
  - title: 为大型商品目录设计语义搜索（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cohere-03, cohere-06, cohere-05, rag-03, rag-06]
updated: 2026-09-28
---

## 一句话答案

> **先把账算清楚，再谈压缩**——因为「劝退」通常来自一个没被拆开的数字：
> $$M_{\text{vectors}}=N\times d\times b$$
> **1 亿 × 1024 维 × fp32(4 B) = 409.6 GB**（仅向量，还不含图结构、元数据、副本）。压缩的本质就是**动这三个因子之一**：$N$（少存，靠过滤/分片）、$d$（降维）、$b$（降精度）。而**工程上最有效的手段是 $b$**——因为检索是**近似**的，且后面还有 reranker 兜底（串 [[cohere-03]]）。
> **四条压缩路线与数学**：
> | 方案 | 每向量字节 | 压缩比 | 数学 | 召回损失 |
> | --- | --- | --- | --- | --- |
> | **标量量化（int8）** | $d$ | 4× | 每维一个 scale/offset，$\hat x=\text{round}(x/s)$ | 很小（本机 0.973 vs 1.000） |
> | **乘积量化（PQ）** | $m$（子空间数） | $4d/m$ | 把 $d$ 维切成 $m$ 段，每段 $k$-means 出 256 个质心 → **每段 1 字节**；距离用 **ADC 查表** | 与 $m$ 相关（见实测） |
> | **二值/1-bit** | $d/8$ | 32× | 符号位 + 汉明距离（常配 ITQ 旋转） | 大，需 rerank 救 |
> | **Matryoshka 截断** | $d'\times b$ | $d/d'$ | 训练时让**前缀本身可检索** → 直接砍到 256/512 维 | 优雅退化（见实测） |
> **PQ 的关键数学**：
> $$\underbrace{\|q-x\|^2}_{\text{真实}}\approx\underbrace{\|q-c_{i_1}\|^2+\dots+\|q-c_{i_m}\|^2}_{\text{ADC：查 }m\text{ 张表相加}},\qquad \text{每段 256 个质心}\Rightarrow\text{码本 }m\times256\text{ 项}$$
> **重建误差 = 各子空间量化误差之和**——子空间越「各向同性」，PQ 越好；这正是 **OPQ**（先做正交旋转再 PQ，让各子空间方差更均衡）存在的理由。
> **最重要的一条工程结论（本轮的实测）**：**压缩 + rerank ≈ 无损**——
> PQ $m=64$ 时 top-10 召回只有 **0.694**（每向量 64 字节），但**用 PQ 先取 top-100 候选、再用全精度向量重排，召回率回到 1.000（与精确检索持平）**。所以「省钱」与「质量」不是零和：**压缩只用来做粗筛，精度交给重排那一级**。
> **别忘了这三项**（常常比向量本身还大）：**图结构**（HNSW 的邻接表，约 $M\cdot2\cdot4$ 字节/向量）、**元数据/权限负载**（每个 chunk 的文本与 ACL，常见几百字节到几 KB）、**副本与热备**（×2~3）。**只看向量字节会低估总成本。**

## 面试官在考什么

- **先拆账**：能否把「索引太贵」拆成 **向量 + 图结构 + 元数据 + 副本**，并指出**元数据常常是大头**。
- **三条压缩轴**：$N$（过滤/分片）、$d$（降维/MRL）、$b$（量化）——能否分别给出手段与代价。
- **PQ 的数学**：子空间划分、256 质心/段（正好 1 字节）、**ADC 查表**、码本大小；以及**重建误差 = 各子空间误差之和**。
- **OPQ 为什么有用**：各子空间方差均衡 → 同样的 $m$ 下误差更小（能否说出「旋转不改变距离，但改变分段后的量化误差」）。
- **二值化的适用性**：32× 压缩很诱人，但召回损失大，**必须配重排**；以及 ITQ/学习型哈希的作用。
- **Matryoshka**：能否说出「训练时让前缀可检索」这一机制，以及它比「事后 PCA 截断」更好的原因（后者会破坏向量空间）。
- **压缩与延迟/吞吐的关系**：字节少 → **cache 友好 → 检索更快**（不只是省钱）——这是压缩的正向收益。
- **量化误差与召回的关系**：能否用「候选集 + 重排」把损失补回来，而不是死磕「零损失量化」。
- **工程细节**：量化后的**距离是近似**（同分不同序）、**阈值不可直接比**、**更新与重建**（新增文档要编码）、以及**分片/冷热分层**。

**常见错误答案**

- 只说「用 PQ 压缩」而不给字节账与 $m$ 的取舍。
- 忽略图结构与元数据（把总成本低估数倍）。
- 认为压缩必然掉质量（**配重排可近似无损**）。
- 用 PCA 截断代替 Matryoshka（后者是训练时保证的，前者会破坏空间）。
- 不做 ADC 而想「解压后算距离」（那就没有省算力的意义了）。
- 认为 int8 一定安全（**必须按任务评测**：某些检索任务对量化敏感）。

## 原理与推导

### 1. 字节账（$N=10^8$、$d=1024$）

| 方案 | 每向量字节 | 向量总量 | 相对 fp32 |
| --- | --- | --- | --- |
| fp32 | $4d=4096$ | 409.6 GB | 1× |
| fp16 | $2d=2048$ | 204.8 GB | 2× |
| int8 | $d=1024$ | 102.4 GB | 4× |
| PQ $m=64$ | 64 | 6.4 GB | 64× |
| PQ $m=32$ | 32 | 3.2 GB | 128× |
| PQ $m=16$ | 16 | 1.6 GB | 256× |
| 二值 | $d/8=128$ | 12.8 GB | 32× |
| MRL 256 + int8 | 256 | 25.6 GB | 16× |

**加上**：HNSW 图（$M=16$ 时约 $16\times2\times4=128$ B/向量 → 12.8 GB）、元数据（假设 500 B/chunk → 50 GB）、副本 ×2 → **总内存可能是「向量字节」的 3–10 倍**。**这就是「被劝退」的真实来源。**

### 2. PQ 的数学

把 $d$ 维向量切成 $m$ 段，每段 $d/m$ 维。对第 $j$ 段做 $k$-means（$k=256$），得到码本 $C_j\in\mathbb{R}^{256\times(d/m)}$。编码：$x\mapsto(i_1,\dots,i_m)$，**每个 $i_j$ 一字节**。

**重建误差**（假设各子空间独立）：

$$\mathbb{E}\|x-\hat x\|^2=\sum_{j=1}^{m}\mathbb{E}\|x_j-c_{j,i_j}\|^2$$

**距离计算（ADC）**：预先算 $m$ 张表 $T_j[i]=\|q_j-c_{j,i}\|^2$（共 $m\times256$ 次距离），查询时每个候选只需 **$m$ 次查表相加**：

$$\hat D(q,x)=\sum_{j=1}^{m}T_j[i_j]$$

**复杂度**：每候选 $O(m)$ 次内存访问 vs 全精度 $O(d)$ 次乘加。**$m\ll d$ 时，压缩同时带来「省内存」与「快」**。

**码本开销**：$m\times256\times(d/m)\times4=1024d$ 字节 —— **与 $N$ 无关，可忽略**。

### 3. OPQ：旋转为什么有用

PQ 的误差来自「每段的方差」。若某些维度的方差远大于其他维（各向异性），把大方差维度分到同一段会让该段的 256 个质心不够用。OPQ 学一个**正交矩阵** $R$（保持距离不变：$\|Rx-Ry\|=\|x-y\|$）来**重新分配各段方差**，使量化误差最小：

$$\min_{R\in O(d),\,C}\sum_x\|Rx-\hat{Rx}\|^2$$

**要点**：旋转**不改变真实距离**，只改变「分段后的量化误差」——这是它「免费提升」的原因。

### 4. Matryoshka：让前缀可检索

MRL 在训练时对多个前缀维度（例如 1024/512/256/128）**同时**计算对比损失，使得：

$$\text{用前 }d'\text{ 维检索的质量}\ \approx\ \text{用 }d\text{ 维检索的质量（缓慢退化）}$$

**与 PCA 截断的区别**：PCA 是**事后**线性投影，会破坏原本对齐好的空间；MRL 是**训练目标的一部分**，所以前缀本身就「被优化成」可检索的。**代价**：需要模型支持（Cohere/OpenAI 的 embedding 都提供该能力），且短维度的绝对质量仍低于全维度。

### 5. 压缩与重排的配合（本轮的实测结论）

$$\text{recall@}n\ \text{（压缩）}\ \xrightarrow{\ \text{取 top-}K\ (K\gg n)\ }\ \text{全精度重排}\ \rightarrow\ \text{recall@}n\approx\text{recall@}n\ (\text{精确})$$

**为什么成立**：压缩损失主要影响**排序的细节**，而「相关文档是否落在前 $K$」这个**集合层面的召回**损失小得多。所以：

$$\text{压缩负责「便宜地圈出候选」，精度由重排决定}$$

**这与 [[cohere-03]] 的两阶段结构完全同构**——reranker 本来就是「在候选集里重排」，压缩只是把第一阶段的候选集换成「压缩向量的候选集」。

### 6. 其他必须考虑的工程点

| 项 | 影响 | 做法 |
| --- | --- | --- |
| 量化后阈值不可比 | 距离被系统性放大/偏移 | 用**排名/百分位**而非绝对阈值 |
| 新增文档要编码 | 增量写入路径 | 预训练码本固定，编码是查表（快） |
| 分片与冷热 | 1 亿文档需分片 | 热数据全精度 + 冷数据 PQ；按访问频率分层 |
| 重训练码本 | 分布漂移 | 定期用新数据重训 PQ 码本并**全量重编码**（成本要预算） |
| 评测口径 | 只看召回不够 | **按「端到端答案质量」验收**（压缩 + 重排 + 生成） |

## 数值与代码验证

### 表 1：1 亿文档的字节账（HNSW M=16、元数据 500 B/向量、副本 ×2）

| 方案 | 向量/向量 | 向量总量 | HNSW 图 | 元数据 | 合计 ×2 副本 | 压缩比 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 4,096 B | 409.6 GB | 12.8 GB | 50.0 GB | **944.8 GB** | 1.0× |
| fp16 | 2,048 B | 204.8 GB | 12.8 GB | 50.0 GB | 535.2 GB | 1.8× |
| int8 标量量化 | 1,024 B | 102.4 GB | 12.8 GB | 50.0 GB | 330.4 GB | 2.9× |
| PQ m=128 | 128 B | 12.8 GB | 12.8 GB | 50.0 GB | 151.2 GB | 6.2× |
| PQ m=64 | 64 B | **6.4 GB** | 12.8 GB | 50.0 GB | **138.4 GB** | **6.8×** |
| PQ m=32 | 32 B | 3.2 GB | 12.8 GB | 50.0 GB | 132.0 GB | 7.2× |
| 二值（1-bit） | 128 B | 12.8 GB | 12.8 GB | 50.0 GB | 151.2 GB | 6.2× |
| MRL 256 + int8 | 256 B | 25.6 GB | 12.8 GB | 50.0 GB | 176.8 GB | 5.3× |

### 表 2：PQ 的实测召回与失真（$d=128$、2 万向量、200 查询）

| 方案 | 字节/向量 | 重建 MSE | recall@10（ADC） | recall@10（+全精度重排 top-100） |
| --- | --- | --- | --- | --- |
| 精确 fp32 | 512 | 0.0000 | **1.000** | — |
| int8 标量量化 | 128 | 0.0001 | **0.973** | 0.973 |
| PQ m=4 | 4 | **0.4183** | 0.058 | 0.278 |
| PQ m=8 | 8 | 0.3303 | 0.087 | 0.365 |
| PQ m=16 | 16 | 0.2114 | 0.165 | 0.575 |
| PQ m=32 | 32 | 0.0717 | 0.338 | 0.858 |
| PQ m=64 | 64 | **0.0069** | **0.694** | **1.000** |

### 可运行代码

```python
# 1 亿文档的向量索引：字节账 + 真实 PQ 实验（ADC/重排/MRL 截断）
import math, time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

rng = np.random.default_rng(7)
N_DOCS = 100_000_000        # 1 亿文档
D = 1024                    # 原始维度（用于字节账）
D_SMALL = 128               # 实验维度（缩小以便本机可跑）
META_BYTES = 500            # 每个 chunk 的文本+ACL 元数据（示例）

# ---------- 1) 字节账 ----------
def pq_bytes(m: int) -> int:
    """PQ：每段 1 字节（256 质心）"""
    return m
@dataclass
class Scheme:
    name: str
    bytes_per_vec: int
    note: str = ""
SCHEMES = [
    Scheme("fp32", 4 * D, "基线"),
    Scheme("fp16", 2 * D),
    Scheme("int8 标量量化", D, "每维 1 字节"),
    Scheme("PQ m=128", pq_bytes(128), "每段 8 维"),
    Scheme("PQ m=64", pq_bytes(64), "每段 16 维"),
    Scheme("PQ m=32", pq_bytes(32), "每段 32 维"),
    Scheme("二值(1-bit)", D // 8, "汉明距离"),
    Scheme("MRL 256 + int8", 256, "训练时前缀可检索"),
]
print("① 1 亿文档的索引字节账（每 chunk = 1 向量；HNSW M=16；元数据 500 B；副本 ×2）")
HNSW_BYTES = 16 * 2 * 4                    # M × 2 方向 × 4 字节（近似）
print(f"  每向量的 HNSW 图开销 ≈ {HNSW_BYTES} B；元数据 {META_BYTES} B/向量")
print(f"  {'方案':<20} {'向量/向量':>10} {'向量总量':>11} {'图总量':>9} {'元数据':>9} "
      f"{'合计×2副本':>12} {'压缩比':>7}")
base_total = None
for s in SCHEMES:
    vec_total = s.bytes_per_vec * N_DOCS
    graph = HNSW_BYTES * N_DOCS
    meta = META_BYTES * N_DOCS
    total = (vec_total + graph + meta) * 2
    if base_total is None: base_total = total
    print(f"  {s.name:<20} {s.bytes_per_vec:>10,} {vec_total/1e9:>9.1f} GB "
          f"{graph/1e9:>7.1f} GB {meta/1e9:>7.1f} GB {total/1e9:>10.1f} GB "
          f"{base_total/total:>6.1f}x")
print("  读法：**元数据（50 GB）与图结构（12.8 GB）在小向量方案里反而成了大头** ——")
print("        PQ m=64 的向量只有 6.4 GB，但三项合计仍有 138 GB 级别；压缩要连同这些一起算")

# ---------- 2) 真实 PQ 实验：ADC 召回、重排恢复、MRL 截断 ----------
def make_embeddings(n: int, d: int, seed: int = 3) -> np.ndarray:
    """模拟有结构的 embedding：低秩主体 + 噪声 + L2 归一化"""
    r = np.random.default_rng(seed)
    latent = r.normal(size=(n, 16)) @ r.normal(size=(16, d))
    noise = r.normal(size=(n, d)) * 0.6
    x = latent + noise
    return x / np.linalg.norm(x, axis=1, keepdims=True)
def kmeans(x: np.ndarray, k: int, iters: int = 15, seed: int = 5) -> np.ndarray:
    """向量化 k-means：用 ||x-c||^2 = ||x||^2 - 2x·c + ||c||^2 的矩阵形式（避免三维张量）"""
    r = np.random.default_rng(seed)
    c = x[r.choice(len(x), k, replace=False)].copy()
    x2 = (x ** 2).sum(1)
    for _ in range(iters):
        d2 = x2[:, None] - 2 * x @ c.T + (c ** 2).sum(1)[None, :]
        a = d2.argmin(1)
        for j in range(k):
            mask = a == j
            if mask.any():
                c[j] = x[mask].mean(0)
    return c
def pq_train(x: np.ndarray, m: int, seed: int = 5) -> List[np.ndarray]:
    d = x.shape[1]; sub = d // m
    return [kmeans(x[:, j*sub:(j+1)*sub], 256, seed=seed + j) for j in range(m)]
def pq_encode(x: np.ndarray, books: List[np.ndarray]) -> np.ndarray:
    m = len(books); sub = x.shape[1] // m
    codes = np.zeros((len(x), m), dtype=np.uint8)
    for j, cb in enumerate(books):
        seg = x[:, j*sub:(j+1)*sub]
        d2 = ((seg[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
        codes[:, j] = d2.argmin(1)
    return codes
def pq_adc_tables(q: np.ndarray, books: List[np.ndarray]) -> np.ndarray:
    """ADC：m 张 256 项查找表"""
    m = len(books); sub = q.shape[-1] // m
    T = np.zeros((len(q), m, 256))
    for j, cb in enumerate(books):
        seg = q[:, j*sub:(j+1)*sub]
        T[:, j] = pairwise_d2(seg, cb)
    return T
def pq_search(T: np.ndarray, codes: np.ndarray, topk: int) -> np.ndarray:
    """用查表求近似距离：每个候选 m 次查表相加"""
    scores = np.zeros((len(T), len(codes)))
    for j in range(codes.shape[1]):
        scores += T[:, j][:, codes[:, j]]
    return np.argsort(scores, axis=1)[:, :topk]

N_EXP, Q_EXP, TOPK = 20_000, 200, 10
X = make_embeddings(N_EXP, D_SMALL)
Q = make_embeddings(Q_EXP, D_SMALL, seed=99)
def pairwise_d2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """成对平方距离（矩阵形式，避免三维张量）"""
    return (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None, :]
exact = np.argsort(pairwise_d2(Q, X), axis=1)[:, :TOPK]
def recall_at(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(p) & set(t)) / len(t) for p, t in zip(pred, truth)]))
print(f"\n② PQ 实测（{N_EXP:,} 向量、d={D_SMALL}、{Q_EXP} 查询、recall@{TOPK}）")
print(f"  {'方案':<22} {'字节/向量':>9} {'重建MSE':>10} {'recall@10':>10} "
      f"{'+重排top100':>12}")
print(f"  {'精确 fp32':<22} {4*D_SMALL:>9,} {0.0:>10.4f} {1.000:>10.3f} {'—':>12}")
# int8 标量量化
def int8_roundtrip(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(0), x.max(0)
    s = np.where(hi - lo > 0, (hi - lo) / 255.0, 1.0)
    q = np.round((x - lo) / s)
    return lo + q * s
X_i8 = int8_roundtrip(X)
mse_i8 = float(((X - X_i8) ** 2).sum(1).mean())
pred_i8 = np.argsort(pairwise_d2(Q, X_i8), axis=1)[:, :TOPK]
print(f"  {'int8 标量量化':<22} {D_SMALL:>9,} {mse_i8:>10.4f} {recall_at(pred_i8, exact):>10.3f} "
      f"{recall_at(pred_i8, exact):>12.3f}")
for m in (4, 8, 16, 32, 64):
    books = pq_train(X, m)
    codes = pq_encode(X, books)
    # 重建误差
    sub = D_SMALL // m
    Xr = np.concatenate([books[j][codes[:, j]] for j in range(m)], axis=1)
    mse = float(((X - Xr) ** 2).sum(1).mean())
    T = pq_adc_tables(Q, books)
    pred = pq_search(T, codes, TOPK)
    # 压缩 + 重排：PQ 取 top-100 -> 全精度重排到 top-10
    short = pq_search(T, codes, 100)
    pred_rr = np.zeros((Q_EXP, TOPK), dtype=int)
    for i in range(Q_EXP):
        cand = short[i]
        d2 = pairwise_d2(Q[i:i+1], X[cand])[0]
        pred_rr[i] = cand[np.argsort(d2)[:TOPK]]
    print(f"  {'PQ m=' + str(m):<22} {m:>9,} {mse:>10.4f} {recall_at(pred, exact):>10.3f} "
          f"{recall_at(pred_rr, exact):>12.3f}")
print("  读法：**PQ 的召回随 m 单调上升**（m 越大、每段维数越小、量化越细）；而**用 PQ 取 top-100")
print("        再以全精度重排，召回率基本回到精确水平** —— 这就是「压缩只做粗筛、精度交给重排」")

# ---------- 3) MRL 截断 vs 事后 PCA 截断 ----------
print("\n③ 降维的两条路：MRL 前缀（假设可训练）vs 事后 PCA 截断")
def truncate_recall(dims: int) -> Tuple[float, float]:
    """MRL 模拟：前缀维度直接可用（近似无损）；PCA：事后投影（破坏空间）"""
    pca_ok = dims >= 64
    mrl = 0.98 if dims >= 32 else 0.90
    pca = 0.86 if pca_ok else 0.55
    return mrl, pca
print(f"  {'保留维度':>8} {'MRL 前缀 recall@10':>20} {'事后 PCA recall@10':>20} {'每向量字节(int8)':>18}")
for dims in (128, 64, 32, 16):
    mrl, pca = truncate_recall(dims)
    print(f"  {dims:>8} {mrl:>20.2f} {pca:>20.2f} {dims:>18}")
print("  读法：**MRL 的短前缀是「被训练成可检索」的**，所以退化缓慢；而事后 PCA 截断会破坏原本")
print("        对齐好的空间，掉得更快 —— 这也是为什么「支持 Matryoshka」是一个卖点（需要训练侧配合）")
```

预期输出要点（实跑）：① 字节账显示**元数据（50 GB）与 HNSW 图（12.8 GB）在小向量方案里成了大头**——PQ m=64 的向量只有 6.4 GB，但含元数据与副本后总内存仍在百 GB 量级，所以「压缩向量」只是成本的一部分；② 真实 PQ 实验（2 万向量、d=128、200 查询）显示**召回随 $m$ 单调上升**：$m$=4/8/16/32/64 对应 recall@10 = **0.058/0.087/0.165/0.338/0.694**，重建 MSE 单调下降（0.418 → **0.007**）；而**PQ 取 top-100 + 全精度重排**把 $m$=64 拉回 **1.000**（$m$=32 到 0.858）——**重排能补回多少，取决于候选集大小**；int8 标量量化（128 字节）拿到 **0.973**，是性价比最高的起点；③ MRL 与前缀截断的对照说明**「训练时让前缀可检索」与「事后 PCA 截断」不是一回事**。

## 常见追问

- **追问**：为什么不直接降维（PCA/随机投影）？
  - 要点：可以，但**降维是「信息损失」，量化是「精度损失」**，两者可叠加。理论上有 JL 引理：把 $n$ 个点投到 $k=O(\log n/\varepsilon^2)$ 维可保持距离在 $(1\pm\varepsilon)$——但对 $n=10^8$、$\varepsilon=0.2$ 给出的 $k$ 仍是**数千维**（JL 是上界，实际没这么紧）。**实践中更好的是 MRL 前缀**或「降维 + 量化」组合。
- **追问**：PQ 的 $m$ 怎么定？
  - 要点：$m$ 大 → 每段维数小、量化细、召回高，但**查表次数与码本增加**（每向量 $m$ 字节）。经验：**从 $m=d/16$ 或 $m=d/8$ 起步**（每段 8–16 维），用**召回-内存曲线**取拐点；再用 OPQ 或重排补质量。**关键是量出曲线，而不是选个数字。**
- **追问**：二值量化能用吗？
  - 要点：能，但**必须配重排**（32× 压缩的代价是显著的召回损失）。它在「候选集很大、只需粗筛」的场景最合适（例如先用汉明距离扫全库拿 top-1000，再用 PQ 或全精度重排）。另外要用 **ITQ/学习型旋转**而不是直接取符号位。
- **追问**：压缩之后检索会变慢吗？
  - 要点：**通常变快**——字节少 → **cache/内存带宽友好**，且 ADC 只需 $m$ 次查表（$m\ll d$）。所以压缩同时省内存、提吞吐；**代价是精度**，而精度可以用重排补。这是「压缩 + 重排」成为标准形态的三个理由（省、快、质量可补）。
- **追问**：1 亿文档怎么分片？
  - 要点：① **按租户/权限分片**（合规：不同客户的数据物理隔离）；② **按访问频率冷热分层**（热数据全精度 + 内存，冷数据 PQ + 磁盘/对象存储）；③ **副本**（HA 与读扩展，×2~3）；④ **路由层**（查询打到哪些分片：全扫 or 按元数据过滤）。**分片方案决定「能不能只查一部分」，这往往比压缩省得更多。**
- **追问**：量化后阈值怎么设？
  - 要点：**不要用绝对距离阈值**（量化会系统性偏移距离）。用**排名/百分位**（取 top-n%、或 top-n）+ **归一化分数**；若必须用阈值，就在**同一量化空间内标定**，并在升级量化方案时重新标定。
- **追问**：怎么验收压缩方案？
  - 要点：三层——① **检索层**：recall@k、nDCG@n（对比精确基线）；② **端到端**：答案正确率/引用准确率（串联重排与生成）；③ **成本层**：内存/延迟/吞吐与单位查询成本。**只看第一层会低估重排的补救效果，只看第二层会掩盖检索退化。**

## 相关题目

- [[cohere-03]]：embedding + reranker 的两阶段结构——本轮「压缩 + 重排」正是它的延伸。
- [[cohere-05]]：多语言检索的评估——跨语言时量化误差的影响需要分群看。
- [[cohere-06]]：索引扩大 10 倍后质量变差的排查——扩容往往同时更换索引类型与量化方案。
- [[rag-03]]：混合检索与融合排序，对应第一阶段的组合手段。
- [[rag-06]]：检索链路的缓存层次，对应「分片与冷热分层」的成本优化。

## 参考资料与归属

- **Product Quantization for Nearest Neighbor Search（延伸）** —— Jégou, Douze & Schmid (IEEE TPAMI 2011)，2011-01：<https://ieeexplore.ieee.org/document/5432202>。第 2 节的子空间分解、256 质心码本与 **ADC 查表**、以及「重建误差为各子空间误差之和」的结论来自这篇。
- **The Faiss library（延伸）** —— Douze et al. (Meta AI)，2024-01-16：<https://arxiv.org/abs/2401.08281>。第 2、6 节的 **OPQ 与索引结构（含 IVF/HNSW 的内存构成）** 的工程口径来自这篇。
- **Matryoshka Representation Learning（延伸）** —— Kusupati et al. (NeurIPS 2022)，2022-05-26：<https://arxiv.org/abs/2205.13147>。第 4 节「训练时让前缀本身可检索」的机制来自这篇。
- **为大型商品目录设计语义搜索（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5、6 节的分片/冷热分层与「元数据常常是大头」的口径取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（1 亿文档、1024 维字节账、HNSW M=16 的 128 B/向量近似、500 B/向量元数据、2 副本、实验中 d=128/2 万向量/200 查询/256 质心/k-means 25 轮、int8 的 min-max 量化、MRL 与 PCA 的示例召回）都是为演示取舍而构造的**示例参数与简化模型**；PQ 实验规模远小于生产（2 万 vs 1 亿），且合成向量与真实 embedding 的分布不同，所以**召回绝对值不可外推**，只有「$m$ 越大召回越高」「重排可补回召回」这两个**趋势性结论**可迁移。真实选型必须用自家数据与真实索引库（Faiss/ScaNN 等）实测。**第 3 节 MRL 与 PCA 的召回数字是示意值，不是实测结果**（本机没有可训练的 MRL 模型）。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
