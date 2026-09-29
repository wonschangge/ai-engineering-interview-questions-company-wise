---
type: question
id: coding-11
topic: 编程与数据结构
order: 11
question: 在 embedding 上实现余弦相似度检索，然后解释为什么你不会把它上线。
question_en: Implement cosine-similarity retrieval over embeddings, then explain why you would not ship it as-is.
asked_at: []
level: 进阶
tags: [余弦相似度, 检索, 实现题, ann]
sources:
  - title: 向量数据库是如何工作的？
    url: https://outcomeschool.com/blog/how-does-a-vector-database-work
    author: Amit Shekhar (Outcome School)
    published:
  - title: Billion-scale similarity search with GPUs（FAISS）（延伸）
    url: https://arxiv.org/abs/1702.08734
    author: Johnson, Douze, Jégou
    published: 2017-02-28
  - title: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs（HNSW）（延伸）
    url: https://arxiv.org/abs/1603.09320
    author: Malkov & Yashunin
    published: 2016-03-30
related: [rag-09, rag-02, rag-03, multimodal-10, coding-10]
updated: 2026-09-28
---

## 一句话答案

> 参考实现只有三步：**入库时**把每条向量归一化并持久化，**检索时**把 query 归一化后做一次 $(N,d)\cdot(d,)$ 的矩阵-向量乘，再用 `argpartition` 从 $N$ 个分数里选出 top-$k$。归一化之后余弦就是内积，所以整条链路上只有一次 GEMM 和一次 $O(N)$ 的选择（$N=10^8$ 时实测 0.96 s 对全排序的 17.7 s）。它不能上线的原因不在代码里，而在账上：$10^8$ 条 768 维 fp32 向量是 307 GB，每次查询都要读一遍；算术强度只有 0.5 FLOP/byte，比 H100 的 roofline 拐点 295 低 590 倍，所以即使在 3.35 TB/s 上只读一遍也要 92 ms，单卡带宽只够 10.9 QPS。能上线的版本是「ANN 索引 + 用量化换内存 + 元数据过滤 + 混合召回 + 重排 + 召回率验证」的组合，这段几十行的代码只是它在上线前的 ground truth 生成器。

## 面试官在考什么

- 会不会写向量化实现：一次 GEMM 加一次选择，而不是 for 循环里逐条算余弦；并且知道范数在入库时算一次就够，不能在每次查询里重算。
- 是否理解「归一化把余弦退化成内积」这个等价变形，以及它是后面所有优化（量化、ANN、batch 检索）的前提。
- 知不知道 `argpartition` 与 `argsort` 的区别：前者是 $O(N)$ 的选择、后者是 $O(N\log N)$ 的排序，$k\ll N$ 时前者的常数与复杂度都更好。
- 能不能把「不会上线」讲成一份工程清单，而不是只说一句「数据量大了会慢」。要落到带宽与算力（谁是瓶颈）、延迟 SLO、内存、过滤、多租户与更新、检索质量这七件事上。
- 是否知道 ANN 的取舍结构：暴力检索精度 100% 但不可扩展，ANN 用 recall 换延迟，乘积量化用量化误差换内存；而且 recall 曲线只能在自己的数据和查询分布上扫出来。

常见错误答案：

- 「上向量数据库就不慢了」。说不出慢在哪一侧（带宽还是算力）、也不知道向量库内部就是这里要写的 flat 扫描加上一层 ANN 索引。
- 「余弦相似度要除以两个模长」，于是在每次查询里对全库重算范数——多花一次 $O(Nd)$ 的读取，等于把检索成本翻倍。
- 把 recall@k 当成可以照抄的公开数字。本节实测：同一份库向量、同样 $n_{list}=256$，在三种查询构造下 $n_{probe}=8$ 的 recall@10 都是 1.000；而 [[rag-09]] 记录的同规模实验在 $n_{probe}=8$ 处是 0.427。两处的库规模、维度与桶数一致，差别只能来自数据分布与查询构造（真实近邻是否跨桶），所以参数只能在自己的数据上扫出来。

## 原理与推导

### 1. 余弦相似度、等价变形与契约

$$\cos(q,x)=\frac{q\cdot x}{\lVert q\rVert\,\lVert x\rVert},\qquad \hat x=\frac{x}{\lVert x\rVert}\ \Longrightarrow\ \cos(q,x)=\hat q\cdot\hat x$$

归一化只在**入库**做一次（$O(Nd)$ 的读写，可离线、可批量、可并行），之后每次查询只剩 $Nd$ 次乘加。同时要把 $\lVert x\rVert$ 存下来或直接存归一化结果，否则每次查询都要重新算 $N$ 个范数。函数契约要写死三件事：输入是「已归一化的库矩阵」还是「原始矩阵」（前者意味着调用方负责入库归一化）、$k\le N$、返回的索引是否保证降序。零向量要单独处理：它的余弦没有定义，实现里应该保持零行并让它恒得 0 分，而不是产出 NaN 把整个 top-$k$ 污染掉。

### 2. 三种实现与复杂度

| 实现 | 复杂度 | 瓶颈 | 用途 |
| --- | --- | --- | --- |
| 纯 Python 循环 | $O(Nd)$ | 解释器每元素几十 ns | 只用于对照，证明向量化的价值 |
| 归一化 + GEMM + `argsort` | $O(Nd)+O(N\log N)$ | 排序是额外的一遍全量读写 | 需要看全序时（例如算阈值、算分位数） |
| 归一化 + GEMM + `argpartition` | $O(Nd)+O(N)$ | 内存带宽 | 生产口径：只要 top-$k$ |

`argpartition` 只保证「前 $k$ 个是最大的 $k$ 个」，不保证它们内部有序，所以要对这 $k$ 个元素再做一次小排序（$k=10$ 时是 10 个元素的排序）。这一步经常被漏掉，结果返回的 top-10 顺序是随机的。

### 3. 规模账：为什么瓶颈是带宽

取 $N=10^8$、$d=768$、fp32、$k=10$：

| 项 | 数值 | 口径 |
| --- | --- | --- |
| 向量本体 | 307.2 GB | $4Nd$，fp32 |
| FLOPs | $1.536\times10^{11}$（153.6 GFLOP） | $2Nd$，一乘一加 |
| A100 bf16 算力下限 | 0.49 ms | 312 TFLOPs（bf16 dense） |
| A100 读一遍 | 150.7 ms | 2.039 TB/s HBM2e |
| H100 读一遍 | 91.7 ms | 3.35 TB/s HBM3 |
| 算术强度 | 0.5 FLOP/byte | fp32；bf16 是 1.0，int8 是 2.0 |
| roofline 拐点 | A100 153、H100 295 FLOPs/byte | 算力除以带宽 |

算术强度比拐点低 306～590 倍，所以这是彻底的带宽题：算力下限 0.49 ms 与带宽下限 150.7 ms 差两个数量级，优化方向只能是**少读字节**（量化、分片、ANN），而不是少算乘加。再乘上业务指标：100 QPS 全量扫描需要 30.7 TB/s 聚合带宽，等于 9.2 张 H100 的全部带宽；单张 H100 只干这件事的上限是 10.9 QPS。排序也要算进预算：$10^8$ 个分数上跑一次 `argpartition` 在这台开发机上是 0.96 s，与扫描本身同量级。

### 4. 不会上线的七条理由（这题真正的考点）

1. **规模**：如上，每次查询扫全库在带宽上就不可行；瓶颈是带宽不是算力（与 [[inference-serving-01]] 的 roofline 思路同源，索引侧的三者对比见 [[rag-09]]）。
2. **延迟 SLO**：检索预算通常是几十毫秒，而且要和 embedding 编码、重排、生成抢同一个预算；全量扫描加排序在这个预算里连一次都跑不完。
3. **内存**：fp32 全量常驻不可行（307 GB 只是向量，没有算 ID、元数据、图边和索引结构）。要么量化（fp16 减半、int8 再减半、PQ 到 96 B/向量），要么分片加多副本，二者都会引入新的取舍。
4. **没有索引**：flat 是精确基线，没有任何结构可以跳过数据；只有加了 ANN 结构（HNSW / IVF / 量化码）才谈得上把扫描量降下来。
5. **过滤**：真实检索几乎总带权限、时间、类别的过滤条件。纯相似度不够用，而且过滤放在召回前还是召回后是两套不同的工程（post-filter 会把 $k$ 条结果过滤到近乎为空，pre-filter 会破坏图遍历），详见 [[rag-08]]。
6. **多租户与更新**：索引要支持增量写入、删除与版本切换。HNSW 的删除多是墓碑、IVF 需要重训质心、蓝绿切换要考虑双份内存——这些都不是这段代码能表达的（见 [[rag-11]]）。
7. **质量**：余弦相似度只是「语义相近」的代理指标，不是任务指标；embedding 模型本身的领域适配（见 [[rag-07]]）和召回是否覆盖了答案所在的 chunk，都要用标注集验证（见 [[rag-04]]）。一个 recall@10 很高但 top-1 常常是错的系统，对下游生成毫无价值。

### 5. ANN 替代方案与两篇原始论文的口径

- **HNSW**：论文口径是「完全基于图、不需要额外的粗搜结构」，用多层嵌套子集构成层次图，节点出现的最高层由**指数衰减分布**随机决定，从而把链接按特征距离尺度分离；从上层开始搜索加上这种尺度分离，使复杂度达到**对数级**；此外「用于选择近邻的启发式在高 recall 与**高度聚集的数据**上显著提升性能」，并且结构与跳表相似、易于做均衡的分布式实现。本节实测也印证了聚集数据的特殊性：在 64 个强簇的合成数据上，单层精确 k-NN 图裂成 **64 个连通分量**，贪心搜索的 recall@10 是 0.000～0.010；每点加 2 条随机长链（层级结构的最小替代品）后连通分量变成 1，recall@10 升到 0.633～0.926，而距离计算次数只有 399～729 次（flat 需要 20000 次）。
- **FAISS 与 GPU 暴力/量化**：论文口径是把 k-selection 优化到**理论峰值的 55%**，使最近邻实现**比此前 GPU 最好结果快 8.5 倍**，并在暴力、近似与基于乘积量化的压缩域三类场景里都大幅超过当时最好结果；举例的规模数字是在 Yfcc100M 的 9500 万张图上 35 分钟构建高精度 k-NN 图、在 4 张 Maxwell Titan X 上**不到 12 小时构建连接 10 亿向量的图**（注意这是建图时间，不是查询延迟）。
- **取舍落点**：暴力检索精度 100% 但不可扩展；ANN 用 recall 换延迟（旋钮是 `efSearch` / `nprobe`）；乘积量化用量化误差换内存（$d=768$ 时 $m=96$ 段 8 bit 的码是 96 B/向量，比 fp32 小 32 倍，1 亿条从 307 GB 降到 9.6 GB）。三者可以叠加，但每一层都会往下游传递误差，所以最后必须用原始向量回表重排。

### 6. 检索之外的工程

① 入库流水线：归一化、量化、批量编码、模型版本与索引版本绑定，任何一环变了都要重建或双写；② 混合检索：纯向量在精确匹配（错误码、专有名词、ID）上会输给 BM25，需要用 RRF 之类的融合（见 [[rag-02]]）；③ 重排：向量只负责宽召回，窄精排交给 cross-encoder（见 [[rag-03]]）；④ 缓存：query 向量与热门查询的结果都可以复用（见 [[inference-serving-05]]）；⑤ 评测：recall@k 是过程指标、端到端任务是结果指标，两者要同时看（见 [[rag-04]]）。

## 数值与代码验证

环境：AMD Ryzen 9 8945HX（32 线程）、62 GiB DDR5、numpy 2.2.6 + OpenBLAS 0.3.29；本节全部数字都在 CPU 上单进程跑出——机器另有一块 8 GB 的 RTX 5070 Laptop GPU（torch 2.12 可见 CUDA），但它没有参与任何实验，faiss 也未安装（ANN 部分为手工实现）。机器是共享环境：性能项取多次重复的最小值，纯 Python 循环在多次运行间有 40～80 µs/向量的波动，IVF 表因 k-means 的线程调度在运行间还有 ±0.01 量级的 recall 波动。参考实现与断言如下，输出为真实运行结果。

```python
import math
import numpy as np

def normalize(X):
    """入库时调用一次：把 (N, d) 逐行归一化，零向量保持零行。"""
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def cosine_topk(q, Xn, k=10):
    """检索：Xn 是已归一化的 (N, d) 库矩阵，q 是 (d,) 查询向量。
    返回 (idx, score)：idx 是 (k,) int64 按余弦降序，score 是对应余弦值。
    契约：k <= N；不复制库矩阵，但 top-k 选择本身要 O(N) 的额外内存（分数向量、
    取负副本与 argpartition 的 int64 索引，实测 16 B/条，N=10^8 时 1.6 GB）。"""
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    q = q / max(np.linalg.norm(q), 1e-12)
    sims = Xn @ q                                       # 归一化后：余弦 == 内积
    idx = np.argpartition(-sims, k - 1)[:k]              # O(N) 选出 top-k
    idx = idx[np.argsort(-sims[idx], kind="stable")]     # 只对 k 个元素精排
    return idx, sims[idx]

def cosine_topk_argsort(q, Xn, k=10):
    """对照组：把 O(N) 的选择换成 O(N log N) 的全排序。"""
    q = q / max(np.linalg.norm(q), 1e-12)
    return np.argsort(-(Xn @ q), kind="stable")[:k]

def cosine_topk_loop(q, X_list, k=10):
    """对照组：纯 Python 逐行循环，用来说明向量化值多少。"""
    q, qn = list(q), math.sqrt(sum(v * v for v in q))
    out = []
    for i, row in enumerate(X_list):
        dot = xn = 0.0
        for a, b in zip(q, row):
            dot += a * b
            xn += b * b
        out.append((dot / (qn * math.sqrt(xn)) if xn and qn else 0.0, i))
    out.sort(key=lambda t: -t[0])
    return [i for _, i in out[:k]]
```

```python
# 断言：与 sklearn 的 cosine 暴力实现逐位对齐；归一化前后等价
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(20260928)
N, d, k = 20_000, 768, 10
X = rng.standard_normal((N, d), dtype=np.float32)
q = rng.standard_normal(d, dtype=np.float32)
Xn = normalize(X)
idx, sc = cosine_topk(q, Xn, k)
nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X)
dist, idx_ref = nn.kneighbors(q.reshape(1, -1))
assert np.array_equal(idx, idx_ref[0])                   # top-k 索引与 sklearn 逐位一致
assert np.abs(sc - (1 - dist[0])).max() < 1e-6           # 余弦值误差在浮点容差内
assert np.array_equal(cosine_topk_argsort(q, Xn, k), idx_ref[0])
assert cosine_topk_loop(q, X[:3000].tolist(), k) == \
    NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X[:3000]) \
    .kneighbors(q.reshape(1, -1), return_distance=False)[0].tolist()
cos_direct = (X @ q) / (np.linalg.norm(X, axis=1) * np.linalg.norm(q))
err = np.abs(cos_direct - Xn @ (q / np.linalg.norm(q))).max()
assert err < 1e-6                                        # 归一化后内积 == 余弦
print("断言全部通过：top-10 索引与 sklearn 逐位一致，余弦最大绝对误差 %.3e"
      % np.abs(sc - (1 - dist[0])).max())
print("归一化后内积与余弦的最大误差 %.3e；每行模长偏离 1 最多 %.3e"
      % (err, np.abs(np.linalg.norm(Xn, axis=1) - 1).max()))
print("top-10 索引 =", idx.tolist())
print("top-10 余弦 =", [round(float(v), 6) for v in sc])
```

```text
断言全部通过：top-10 索引与 sklearn 逐位一致，余弦最大绝对误差 5.960e-08
归一化后内积与余弦的最大误差 4.657e-08；每行模长偏离 1 最多 1.192e-07
top-10 索引 = [11891, 19486, 17541, 19236, 13084, 8689, 7538, 6115, 11846, 12068]
top-10 余弦 = [0.140385, 0.131655, 0.130634, 0.12792, 0.122742, 0.122593, 0.121826, 0.119707, 0.118723, 0.118274]
```

$N=10^4$、$d=768$、$k=10$ 的三种实现耗时，以及 $N$ 扫描（fp32，$k=10$，每格是 5 次重复的最小值）：

| N | 纯循环 | argsort | argpartition | 纯 GEMM | GEMM 有效带宽 | log-log 斜率（对上一行） |
| --- | --- | --- | --- | --- | --- | --- |
| $10^4$ | 434.4 ms | 2.98 ms | 2.95 ms | 2.99 ms | 10.3 GB/s | — |
| $10^5$ | 外推 4.3 s | 17.04 ms | 8.89 ms | 5.97 ms | 51.5 GB/s | 0.48 |
| $10^6$ | 外推 43 s | 185.11 ms | 78.55 ms | 58.40 ms | 52.6 GB/s | 0.95 |
| $3\times10^6$ | — | 492.07 ms | 226.25 ms | 170.92 ms | 53.9 GB/s | 0.96 |

读法：① 纯 Python 循环在 $N=10^4$ 就要 434 ms，比向量化版本慢 146 倍；按 43.4 µs/向量外推，$N=10^6$ 要 43 s，$N=10^8$ 要 72 分钟。② $N\ge10^5$ 之后 log-log 斜率 0.95～0.96，即耗时随 $N$ **线性**增长，符合「读的字节数线性增长、带宽恒定」的预期；$10^4\to10^5$ 段斜率只有 0.48，是因为小 $N$ 时 3 ms 里大半是 numpy/BLAS 的调用与线程唤醒开销（这 30 MB 的数据按实测带宽读一遍只要 0.5～1 ms）。③ GEMM 的有效带宽 52.6 GB/s 与独立测得的机器读带宽 34.4 GB/s（`np.dot` 读 1.07 GB 用 31.3 ms）同量级，说明向量化实现已经贴在带宽上；把线程数从 32 降到 1，同一次 $N=10^6$ 查询从 75 ms 变慢到 138 ms（22 GB/s），而 4 线程就已经能跑满同样的带宽（72 ms），说明这条路径吃的是内存并行度——也说明单进程跑满 32 线程去服务一个查询在服务端是浪费：它换来的只是分摊到一次查询上的带宽，吞吐并不会因此提高。

`argsort` 与 `argpartition` 的差距只在 $N$ 大了以后才显现（下表只对分数向量做选择、不含 GEMM；$N=10^4$ 时端到端两者都在 2.9 ms 上下、看不出差距，$N=10^8$ 时选择本身就相差 18 倍）：

| N | argsort | argpartition | 比值 |
| --- | --- | --- | --- |
| $10^6$ | 84.6 ms | 8.5 ms | 10.0× |
| $10^7$ | 1107.2 ms | 93.9 ms | 11.8× |
| $10^8$ | 17674.6 ms | 960.8 ms | 18.4× |

「为什么用 `argpartition`」这个问题只有在规模上才有答案，小数据上两者没有区别。

IVF 的 recall-latency 实测（$N=2\times10^5$、$d=128$、64 簇高斯混合、$n_{list}=256$、200 条查询、$k=10$、ground truth 为全量精确 top-10；查询为库内向量的扰动，$n_{probe}$ 是唯一旋钮）：

| $n_{probe}$ | 算距离的向量数 | 占全库 | recall@10 | 单次耗时 |
| --- | --- | --- | --- | --- |
| 1 | 851 | 0.43% | 0.410 | 0.19 ms |
| 2 | 1698 | 0.85% | 0.666 | 0.28 ms |
| 4 | 3218 | 1.61% | 0.957 | 1.21 ms |
| 8 | 6367 | 3.18% | 1.000 | 3.48 ms |
| 16 | 12661 | 6.33% | 1.000 | 5.44 ms |
| 32 | 25279 | 12.64% | 1.000 | 8.40 ms |
| 256（=全库） | 200000 | 100% | 1.000 | 62.88 ms |

flat 精确检索是 8.25 ms/查询、$\lvert A_k\cap G_k\rvert/k=1.000$。这张表的作用不是这几个数字，而是它的形状：**扫描量降到 3.18% 就能拿回全部 recall，但降到 1.61% 就丢 4.3 个点、降到 0.43% 只能拿回 41%**。曲线在哪里弯，由「真实近邻是否落在被探测的桶里」决定，也就是由数据分布与查询构造决定——[[rag-09]] 记录的同规模实验在 $n_{probe}=8$ 处只有 0.427（库规模、维度、$n_{list}$ 都相同，差别来自数据与查询构造，本节的三种查询构造恰好都对 IVF 友好）。另外要说明：这里的 IVF 是 Python 逐查询循环，$n_{probe}=8$ 只扫 3.18% 的数据却只快 2.4 倍，差的那些时间全花在解释器与 numpy 调用开销上，所以有意义的量是「算距离的向量数」这一列；C++/GPU 实现（FAISS）才能按比例把它兑现成延迟。这张表的实验核心只有二十行——先 k-means 出质心，再按 $n_{probe}$ 拼出候选、只在候选上算距离（`exact` 是全量精确检索，用来生成 ground truth 与算 recall）：

```python
km = KMeans(n_clusters=256, n_init=1, max_iter=25, random_state=0).fit(Xn)
C = km.cluster_centers_.astype(np.float32)
C /= np.linalg.norm(C, axis=1, keepdims=True)
order = np.argsort(km.labels_, kind="stable")            # 倒排表：同桶的向量连续存放
bs = np.searchsorted(km.labels_[order], np.arange(256))  # 每桶的起止
be = np.searchsorted(km.labels_[order], np.arange(256), side="right")

def exact(Q, k=10):                                      # 精确基线 = ground truth
    return [np.argpartition(-(Xn @ q), k - 1)[:k] for q in Q]

def ivf(Q, nprobe, k=10):
    out, dist_calls = [], 0
    for q in Q:
        probes = np.argpartition(-(C @ q), nprobe - 1)[:nprobe]      # 只比质心
        cand = np.concatenate([order[bs[p]:be[p]] for p in probes])  # 拼候选
        dist_calls += cand.size
        s = Xn[cand] @ q                                             # 只在候选上算距离
        out.append(cand[np.argpartition(-s, k - 1)[:k]])
    return out, dist_calls / len(Q)
```

图检索那一组实验做的是同一件事的两个变体：用精确 k-NN 建图（`argpartition` 取每行的前 $M$ 个）后用 HNSW 论文 Algorithm 2 的 best-first 搜索，另一组把每条边的若干邻居换成随机长链，再用 `scipy.sparse.csgraph.connected_components` 数连通分量，得到上表里「recall 0.000 对 0.926」的对照。

量化的实测（同一批库向量，每行先归一化再量化，检索时不再重归一化）：

| 存储格式 | 常驻内存 | recall@10（只用量化向量） | 回表用 fp32 重排后 |
| --- | --- | --- | --- |
| fp32 | 0.102 GB | 1.000 | — |
| fp16 | 0.051 GB | 0.994 | — |
| int8 标量（每向量 1 个 scale） | 0.026 GB | 0.933 | 1.000（shortlist 50/100/200/500 都是 1.000） |

结论：① int8 把 0.102 GB 压到 0.026 GB（4 倍），单用量化分数会丢 6.7 个点，但只要先用它召一个 50 条的 shortlist、再用 fp32 原向量重排，recall@10 就回到 1.000——这就是 PQ 在生产里的标准用法「宽召回 + 窄精排」，也是 [[rag-03]] 里 cross-encoder 重排的同构结构；② 在这台 CPU 上 fp16/int8 的**耗时没有变快**（25.11 / 24.91 / 24.21 ms），因为要先上转成 fp32 才能进 BLAS 点积：量化省的是内存，要把它兑现成延迟必须靠支持低精度点积的硬件与 kernel。

单层图检索为什么不能想当然（$N=2\times10^4$、$d=128$、$M=16$、100 条查询、用 HNSW 论文 Algorithm 2 的 best-first 搜索）：

| 数据集 | 图 | 连通分量 | ef | recall@10 | 距离计算次数/查询 |
| --- | --- | --- | --- | --- | --- |
| 64 个强簇 | 精确 k-NN 图 | 64 | 32 | 0.000 | 178 |
| 64 个强簇 | 精确 k-NN 图 | 64 | 128 | 0.010 | 255 |
| 64 个强簇 | 每点加 2 条随机长链 | 1 | 32 | 0.633 | 399 |
| 64 个强簇 | 每点加 2 条随机长链 | 1 | 128 | 0.926 | 729 |
| 各向同性高斯 | 精确 k-NN 图 | 1 | 32 | 0.528 | 608 |
| 各向同性高斯 | 精确 k-NN 图 | 1 | 128 | 0.756 | 1914 |

在聚集数据上纯 k-NN 图裂成 64 个分量，贪心搜索永远出不了入口所在的那个簇，recall 是 0；补上长链之后连通分量变成 1，用 729 次距离计算拿到 0.926（flat 要 20000 次）。这正是 HNSW 论文里「指数衰减的层级把链接按距离尺度分离、并在高度聚集的数据上依赖启发式选边」要解决的问题：**图的导航性来自长链，不是来自近邻本身**。维度灾难的量化：在 4000 个随机库向量上测「最近距离 / 最远距离」的对比度，维度越高，最近与最远越难区分。$d=768$ 时两者只差 17%、$d=4096$ 时只差 6.8%：暴力检索在数学上依然正确，但「谁是第一」这件事对噪声（量化误差、维度截断、上游 embedding 的抖动）越来越敏感，这也是 ANN 的 recall 在真实数据上比在低维 benchmark 上更难做高的原因。

| d | 2 | 8 | 64 | 768 | 4096 |
| --- | --- | --- | --- | --- | --- |
| $d_{\min}/d_{\max}$ | 0.0097 | 0.1692 | 0.5778 | 0.8539 | 0.9365 |
| $(d_{\max}-d_{\min})/d_{\min}$ | 102.2 | 4.91 | 0.731 | 0.171 | 0.068 |

## 常见追问

- **追问**：为什么用余弦而不是欧氏距离？
  - 要点：对归一化后的向量两者**排序完全等价**，因为 $\lVert x-y\rVert^2=2-2\,x\cdot y$（单位向量）。差别在语义与数值：余弦只看方向，不看长度，多数文本 embedding 模型在训练时就用余弦或归一化内积作目标，长度因此不携带稳定语义，余弦把这部分噪声直接丢掉；欧氏距离在图像与空间数据上更自然（位置本身就是含义）。如果库里的向量没归一化，两者会给出不同的 top-$k$，这时必须先决定长度要不要参与排序。
- **追问**：为什么很多系统直接用内积？
  - 要点：归一化之后内积就等于余弦，而内积省掉了每次查询里的除法与开方，也能直接用上为 GEMM 优化的 BLAS/tensor core 路径（MIPS）。代价是把「长度」这一维信息彻底丢掉了：如果上游 embedding 没归一化，内积会把长向量系统性地排到前面，这也是很多「同一个模型换了个向量库结果变差」的真实原因——先确认库端到底按什么度量建索引。
- **追问**：维度越高，检索越难吗？
  - 要点：复杂度上不难（$2Nd$ FLOPs 与 $d$ 线性），难的是**区分度**：上表的对比度从 $d=2$ 的 102 掉到 $d=768$ 的 0.171，最近邻与稍远的点分数挤在一起，于是量化误差、ANN 的近似误差、上游模型的微小变化都会改变排名。工程上的对策不是「降维到 2」（会丢语义），而是提高量化精度、加深候选池与重排，并在自己的数据上验 recall。
- **追问**：量化之后还要不要回表取原向量重排？
  - 要点：要，量化分数只能决定谁进 shortlist，最终名次必须用原向量算。上表的实测是 int8 直出 0.933、回表后 1.000；这同时说明回表深度不是越大越好——shortlist 从 50 加到 500 没有额外收益，只是多读字节。真实系统里回表成本 = 深度 × 每条原始字节数（$d=768$、fp32 时 3072 B/条），所以原始向量常放 NVMe 或对象存储、只把这几百条读回来（见 [[rag-03]] 与 [[rag-09]]）。
- **追问**：入库时要固定哪些东西，才能保证检索可复现？
  - 要点：四件事必须一起版本化——embedding 模型与权重版本、归一化方式（是否 L2、零向量怎么办）、量化方式与码本、索引参数。任何一项变了，旧向量与新查询就处在不同的空间里，表现为「recall 悄悄下降」而不是报错。换模型必须全量重编码（见 [[rag-07]]），不能只重新编码新增文档。

## 相关题目

- [[rag-09]]：HNSW / IVF-PQ / flat 的对比与索引参数扫描，是本题「不会上线」之后的正解。
- [[rag-02]]：稀疏与稠密检索的选择；纯向量在精确匹配上的失败模式决定了必须做混合召回。
- [[rag-03]]：reranker 的位置与开销；量化回表重排和 cross-encoder 精排是同一个「宽召回 + 窄精排」结构。
- [[rag-08]]：权限感知 retrieval；过滤必须发生在召回阶段，纯相似度无法表达 ACL。
- [[rag-11]]：索引新鲜度；增量写入、删除与重建是 flat 实现完全覆盖不到的部分。
- [[rag-04]]：检索质量评估；recall@k 与端到端任务指标的关系决定 recall 目标定在哪里。
- [[inference-serving-01]]：prefill 与 decode 的 roofline 分析，与本题「带宽而非算力是瓶颈」是同一套推理方式。
- [[multimodal-10]]：在一个索引里做多模态检索；跨模态向量共享同一套相似度与索引取舍。

## 参考资料与归属

1. [向量数据库是如何工作的？](https://outcomeschool.com/blog/how-does-a-vector-database-work)，Amit Shekhar（Outcome School），2026-06-06。提供三种相似度度量的公式与算例（余弦的取值范围与长度无关性、内积受长度影响、欧氏距离的算例）、暴力检索的代价直觉（1 亿条 × 1000 维 = 一次查询 1000 亿次计算）、ANN 与 recall 的口径、HNSW/IVF/PQ 的直观解释（IVF 的分桶与探测、PQ 的 128 维 512 B 压到 8 B 的 64 倍压缩例）、pre-filter 与 post-filter 的取舍、以及 FAISS 的 `IndexFlatL2` 代码骨架。
2. [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)（延伸），Jeff Johnson、Matthijs Douze、Hervé Jégou，2017-02-28。第 5 节引用的三个口径均取自论文摘要：k-selection 达到理论峰值性能的 55%、最近邻实现比此前 GPU 最好结果快 8.5 倍、在暴力/近似/乘积量化压缩域三类场景上都大幅超过当时最好结果，以及 Yfcc100M 的 9500 万张图 35 分钟建图、4 张 Maxwell Titan X 上 12 小时内构建 10 亿向量的图。
3. [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)（延伸），Yu. A. Malkov、D. A. Yashunin，2016-03-30（v4 2018-08-14）。第 5 节的 HNSW 口径取自论文摘要：完全基于图、不需要额外粗搜结构，最高层由指数衰减分布随机决定从而把链接按特征距离尺度分离，从上层开始搜索带来对数级复杂度，启发式选边在高 recall 与高度聚集数据上显著提升性能，结构与跳表相似因而易于分布式实现。
4. 超出上述三条的部分为自行推导与实测：「原理与推导」第 3 节的规模账按 $2Nd$ FLOPs、$4Nd$ 字节与题面给定的硬件常数（A100 bf16 312 TFLOPs / 2.039 TB/s、H100 SXM5 bf16 989 TFLOPs / 3.35 TB/s HBM3、roofline 拐点 295 FLOPs/byte、LLaMA-3-70B 每 token 320 KiB KV）复算；「数值与代码验证」的全部表格由本次运行产生，脚本在仓库根 `.work/` 下（`c11_doc_final.py` 参考实现与断言、`c11_bench.py` 性能与带宽、`c11_doc_ivf.py` IVF recall-latency、`c11_quant.py` 量化与回表、`c11_nsw3.py` 图检索与连通分量、`c11_scale.py` 规模账），数据为合成数据，绝对数值不可外推到真实 embedding，可外推的是复杂度、线性增长与 recall-latency 的取舍形状。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
