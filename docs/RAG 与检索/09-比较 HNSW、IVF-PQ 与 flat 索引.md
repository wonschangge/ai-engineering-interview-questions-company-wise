---
type: question
id: rag-09
topic: RAG 与检索
order: 9
question: 比较 HNSW、IVF-PQ 和 flat 索引。你如何选择？recall@k 在 latency 上的代价是什么？
question_en: Compare HNSW, IVF-PQ and flat indexes. How do you choose, and what does recall@k cost in latency?
asked_at: []
level: 进阶
tags: [ann, hnsw, ivf-pq, 召回率]
sources:
  - title: How does Approximate Nearest Neighbor (ANN) search work?
    url: https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work
    author: Amit Shekhar (Outcome School)
    published:
  - title: How does a Vector Database work?
    url: https://outcomeschool.com/blog/how-does-a-vector-database-work
    author: Amit Shekhar (Outcome School)
    published:
related: [rag-02, rag-11, rag-01]
updated: 2026-09-28
---

## 一句话答案

> flat 是精确基线：每条 query 要把 $N$ 条向量全部读一遍，$2Nd$ FLOPs、$4Nd$ 字节，recall@k 恒等于 1，瓶颈是显存带宽而不是算力。HNSW 用多层可导航图把「读全部」换成「沿图跳几十到几百个点」，内存充裕时给出最好的 recall-latency 曲线，代价是每向量 $8M$ 字节的图开销、更长的建库时间与难做的删除。IVF-PQ 先用 k-means 分桶（$n_{probe}$ 决定扫多少），再把每条向量压成 $m$ 字节的码，用「读得更少」换 recall，是十亿级规模能塞进内存的主要形态。
> recall 与 latency 是同一根旋钮的两端：HNSW 调 $efSearch$，IVF 调 $n_{probe}$，PQ 调分段数与重排深度。延迟近似随「扫描的候选数 × 每条字节数」增长，而 recall 随参数上升的斜率完全由数据分布决定——所以参数只能在自己的 query 集上扫出来，公开 benchmark 的数字不能当结论。

## 面试官在考什么

- 三种索引的机理能否讲清：flat 的全量精确、IVF 的粗量化加倒排桶、PQ 的分段量化加查表距离、HNSW 的层次图加贪心 best-first。只会说「一个准一个快」区分不出水平。
- 会不会把取舍量化：内存（$4Nd$、$4d+8M$、$m$ 字节/向量）、每次查询的扫描量（$N$、$\frac{n_{probe}}{n_{list}}N$、$efSearch$ 量级的节点访问）、以及两种不同的延迟模型——flat 是流式带宽受限，HNSW 是随机访问受限。
- 是否知道 recall@k 的口径与上界：$|A_k \cap G_k| / k$，ground truth 必须用 flat 生成；IVF 的 recall 被「真实近邻是否落在被探测的桶里」卡死，PQ 的 recall 被量化噪声卡死，两者要用不同的手段补。
- 有没有工程边界感：删除与更新（HNSW 墓碑、IVF 重训）、过滤与多租户（post-filter 塌陷、pre-filter 破坏图遍历）、建库时间、内存预算、小数据量下 flat 反而更优。
- 会不会做评测：扫参数画 recall-latency 曲线、报 p50/p95、先定 recall 目标再压延迟，而不是先选索引再解释。

常见错误答案：

- 「HNSW 又快又准，默认选它」。漏掉内存放大与删除成本：$d=768$ 时 $M=32$ 的图要多占 256 MB/百万条，1 亿条时光图边就是 25.6 GB；多数实现的删除是标记删除，长期不 compaction 会让 recall 悄悄下降。
- 「IVF-PQ 精度低所以不能用」或「recall 不到 100% 就不能上线」。PQ 的正确用法是候选生成器：过采样 + 回表用原始向量重排（实测见「数值与代码验证」：重排深度从 10 加到 300，recall@10 从 0.142 回到 0.406），把损失压到可接受范围。

## 原理与推导

### 1. flat：精确基线与它真正的瓶颈

flat 不做任何结构，query 与全部 $N$ 条向量逐一算距离，$O(Nd)$ 次乘法-累加：

$$T_{\text{flat}} \approx \frac{4Nd}{BW}, \qquad \text{FLOPs} = 2Nd, \qquad \text{算术强度} = \frac{2Nd}{4Nd} = 0.5\ \text{FLOP/B}$$

取 $N=10^6$、$d=768$、fp32：内存 $4Nd = 3.07$ GB，$1.54$ GFLOP/query，算术强度 0.5 FLOP/B，而 A100 的 fp32 平衡点是 $19.5/2.04 = 9.6$ FLOP/B——差 19 倍，所以 flat 是彻底的带宽受限：整块索引流经一次就要 1.51 ms（A100 2.04 TB/s），纯算力只要 79 µs。这条推论解释了后面所有优化的方向：**要快就得少读字节**，而不是少算乘加。$N=10^5$ 时索引 307 MB，一次全扫约 151 µs，这也是「10 万条以内直接 flat」的量化依据。

### 2. IVF：用粗量化缩小扫描范围

k-means 把 $N$ 条向量分到 $n_{list}$ 个桶，查询时先算 query 到 $n_{list}$ 个质心的距离，再只扫最近的 $n_{probe}$ 个桶：

$$\text{扫描量} = N\cdot\frac{n_{probe}}{n_{list}}, \qquad \text{粗量化成本} = n_{list}\cdot d\ \text{次 MAC}$$

$n_{probe}=1$ 时 recall 会很差，因为真实近邻常常落在相邻（甚至不相邻）的桶里。这里有一个精确的上界，值得单独记住：

$$\text{recall@}k \le \frac{1}{k}\sum_{j=1}^{k}\mathbb{1}\big[\text{cell}(g_j)\in \text{probed}\big] \qquad (g_j\ \text{是第 } j \text{ 个真实近邻})$$

桶内若用精确距离，这个上界取等号——「数值与代码验证」的实测中两者每一位小数都相同（0.205/0.324/0.427/0.649 对应 cell bound 0.205/0.324/0.427/0.649）。所以 IVF 调参的本质是**让真实近邻落进被探测的桶**，而不是让桶内的排序更准。

$n_{list}$ 的取舍有两个相互拉扯的项：粗量化成本 $\propto n_{list}$ 线性增长；而桶越小，近邻越容易跨桶，要保持同样的 recall 就必须让 $n_{probe}$ 跟着涨（扫描量 $= N n_{probe}/n_{list}$ 又不降）。经验上 $n_{list}\approx\sqrt N$（SIFT1M 用 1024 就是这条），公开的工程指南给的范围是 $4\sqrt N \sim 16\sqrt N$（$N<10^6$ 时，换算成每桶 62～250 条），并给出训练样本量要求 30～256 条/质心。$N$ 继续变大时每条桶里的向量会多到没法用：$N=10^8$ 时 $\sqrt N=10^4$ 个桶意味着每桶 $10^4$ 条，探 0.2% 的桶（20 个）只能扫 0.2% 的数据、recall 早就崩了，要保持 recall 得探几百个桶（扫 2%～10% 的数据）。于是工程上把 $n_{list}$ 提到 $2^{16}\sim2^{20}$，代价落在粗量化本身：$10^5$ 个桶、$d=768$ 时一次查询要 $7.7\times10^7$ 次 MAC（相当于 $10^8$ 规模下扫 0.1% 数据的成本），因此大 $n_{list}$ 必须改用 HNSW 之类的图索引做粗量化，把这一项从 $O(n_{list})$ 压到 $O(\log n_{list})$。

### 3. PQ：把每条向量压到 $m$ 字节

乘积量化把 $d$ 维切成 $m$ 段（每段 $d/m$ 维），每段用 $2^{b}$ 个码字的码本近似（通常 $b=8$）：

$$\text{码长} = m\cdot\frac{b}{8}\ \text{字节}, \qquad \text{压缩比} = \frac{4d}{m b/8} = \frac{32d}{mb}\ \xrightarrow{b=8}\ \frac{4d}{m}$$

$d=768$：$m=96$（每段 8 维）→ 96 B/向量，压缩 32×；$m=48$ → 48 B，64×；$m=384$ → 384 B，仅 8×。码本自身的内存是 $2^{b}\cdot d\cdot 4$ 字节（与 $m$ 无关），$b=8$、$d=768$ 时 768 KB，可以忽略。

查询用 ADC（asymmetric distance computation）：先为这条 query 建 $m\times 2^{b}$ 的距离查找表，成本 $2^{b}d = 1.97\times10^5$ 次 MAC，与 $m$ 无关；然后每个候选只要 $m$ 次查表相加（$m=96$ 时 96 次加法 vs 精确距离的 768 次 MAC），字节数从 3072 B 降到 96 B。**量化误差进的是排序**：$\|x-\hat x\|^2=\sum_j\|x_j-c_j\|^2$ 是各段误差之和，两条候选的分数差如果小于这个噪声量级，名次就会被随机化——这类数据上 recall@10 会断崖式下跌（同一套 $m=16$、8 bit 的 PQ，在两种合成分布上的全量扫描 recall@10 分别是 0.091 与 0.187，压缩率都是 32×）。补救方式只有一个方向：过采样后用原始向量重排。

打分的口径要写清楚。码本按 L2 训练，而检索常用内积/余弦，两者排序并不自动等价：

$$\|q-\hat x\|^2 = \|q\|^2 - 2\,q\cdot\hat x + \|\hat x\|^2 \;\Longrightarrow\; \text{L2 一致排序用}\ q\cdot\hat x - \tfrac{1}{2}\|\hat x\|^2;\quad \text{直接内积排序用}\ q\cdot\hat x$$

漏掉 $\|\hat x\|^2/2$ 相当于把重建向量的范数混进相似度。两种口径在生产代码里都有人用，实测 recall 会差几个百分点（同一批数据上是 0.163 与 0.187，本例反而是不带修正项的内积口径更好），所以打分函数必须和码本训练目标、度量一起在自己的数据上验证。

### 4. HNSW：层次图 + 贪心 best-first

层的分配服从几何分布，第 $l$ 层保留的节点比例是 $M^{-l}$（论文取 $mL=1/\ln M$）：

$$P(\text{level}\ge l)=M^{-l}, \qquad \mathbb{E}[\text{每节点层实例数}] = \sum_{l\ge0}M^{-l} = \frac{M}{M-1}$$

第 0 层每条向量的度上限 $M_0=2M$，上层上限 $M$；上层链的期望条数 $M/(M-1)\approx1.03$，可以忽略。于是内存（工程上通用的口径）：

$$\text{bytes/vector} = 4d + 2M\cdot 4 = 4d + 8M$$

| $M$ | 第 0 层边 | B/向量 | 100 万条 | 相对 flat | 1 亿条 |
| --- | --- | --- | --- | --- | --- |
| 16 | 32 | 3200 | 3.20 GB | +4.2% | 320 GB |
| 32 | 64 | 3328 | 3.33 GB | +8.3% | 333 GB（其中图边 25.6 GB） |
| 64 | 128 | 3584 | 3.58 GB | +16.7% | 358 GB |

查询自顶向下贪心：上层大步跳，落到第 0 层后用 best-first 遍历，结果堆只保留最近的 $efSearch$ 个候选，直到「候选队列里最近的候选已经比结果堆中最远的那个更远」才停。距离计算的次数落在 $[efSearch,\ efSearch\cdot 2M]$ 这个包络里（去重后每个节点最多算一次），$efSearch$ 就是 recall 旋钮。

召回为什么不是 100%：① 贪心 + 提前终止会停在局部最优；② 建图时的启发式剪枝为了多样性主动丢边，图不是精确的 kNN 图；③ 删除是标记删除，墓碑越多有效连通性越差；④ $efSearch$ 给得太小。这四点里只有第四点是查询时可调的。

延迟模型和 flat 完全不同。顺序读一条 3072 B 向量在 2.04 TB/s 下只要 1.5 ns，而一次随机访问要 ~100 ns，差约 66 倍：HNSW 的延迟约等于「访问节点数 × 随机访问延迟」，而不是「总字节数 / 带宽」。$efSearch=128$、$M=32$ 时包络是 13～819 µs（128～8192 次距离计算 × 100 ns）。这也解释了为什么 HNSW 在批量查询下吞吐提升明显、单查询延迟却几乎不降。

### 5. 选择：先定 recall 目标，再选索引

| 索引 | 内存/百万条（$d=768$） | 每次查询的扫描量 | recall 旋钮 | 训练 / 更新 |
| --- | --- | --- | --- | --- |
| flat | 3.07 GB | $N$ 条精确距离 | 无，恒为 1 | 无需训练；append 直接可用，删除要墓碑或重建 |
| IVF-Flat | 3.07 GB + 质心 12.6 MB（$n_{list}=4096$）+ id | $(n_{probe}/n_{list})N$ 条精确距离 | $n_{probe}$ | 需要训练质心；插入入桶，删除墓碑，分布漂移要重训 |
| IVF-PQ | 96 MB 码（$m=96$）+ id 8 B/条 + 质心 | 同样候选数，但每条只读 96 B | $n_{probe}$ + 重排深度 | 需要训练质心与码本；重排要另存原始向量 |
| HNSW | 3.33 GB（$M=32$） | 访问 $efSearch$ 量级节点，最多 $efSearch\cdot 2M$ 次距离计算 | $efSearch$（$M$、$efConstruction$ 离线定型） | 无需训练，增量插入快；删除是墓碑，部分实现不支持删除 |

决策顺序（不是先选索引）：

1. 需要精确答案，或 $N<10^5$，或要用它生成 ground truth → flat。10 万条 × 768 维 = 307 MB，全扫约 151 µs，图索引在这个规模上省不下时间，还多出建库与内存开销。
2. 内存够（索引能常驻显存/内存）、要求高 recall 与低延迟、写入不频繁 → HNSW。$M$ 定图的质量与内存，$efSearch$ 定查询的 recall-latency 工作点，$efConstruction$ 定建库成本（建库的距离计算次数 $\propto N\cdot efConstruction\cdot M$，是离线一次性支出，别只看查询延迟）。
3. 内存受限、数据量到千万～十亿、可以接受量化损失 → IVF-PQ（或 HNSW 配标量量化、DiskANN 这类磁盘驻留的图索引）。关键是**别同时常驻原始向量**：1 亿条 × 768 维 fp32 是 307 GB，而 $m=96$ 的码只有 9.6 GB；需要重排时把原始向量放 NVMe 或对象存储，只把重排的那几百条读回来。
4. 频繁删除/更新 → 见「常见追问」中关于删除与更新的那一条，三种索引的代价完全不同。

**过滤与多租户是这道题的隐藏考点。** 设过滤选择率为 $s$：post-filter 先检索再过滤，$k=10$ 的结果里期望只剩 $10s$ 条，$s=1\%$ 时平均 0.1 条，等于没有结果——要拿到 10 条就得把候选深度放大到 $10/s=1000$，延迟随之上一个量级。pre-filter 在 IVF 上相对好做（桶内按位图跳过不合规的点），但在 HNSW 上会破坏遍历：一个节点的邻居可能全部被过滤掉，搜索无法前进，需要过滤感知的遍历或分区索引。工程上的常规折中是**按租户/时间分区成多个小索引 + 查询时路由**，并接受小分区上 flat 反而更优的事实——单租户 5 万条时 flat 全扫约 75 µs，比维护一张 HNSW 图划算。

**评测口径。** 用 flat 在**全量语料**上生成 top-$k$ ground truth（$k$ 与线上取回的 $k$ 一致），扫 $efSearch$ / $n_{probe}$ / 重排深度，画 recall@k 对 p50、p95 延迟的曲线，同时记录内存、建库时间、增量写入吞吐；最后固定业务可接受的 recall（例如 recall@10 ≥ 0.95）去压延迟。语料和 query 分布都在变，所以这条曲线要定期重测（与 [[rag-11]] 的索引新鲜度、[[rag-04]] 的评估口径是同一套流程）。

## 数值与代码验证

统一口径：$d=768$、fp32、$N=10^6$；A100 80G 带宽 2.04 TB/s、fp32 19.5 TFLOPS，H100 SXM 3.35 TB/s、fp32 66.9 TFLOPS；随机访问延迟按 ~100 ns 估算。

**flat 的 roofline**：内存 $4Nd = 3.072\times10^9$ B = 3.07 GB（2.86 GiB）；$N d = 7.68\times10^8$ 次 MAC，$1.54\times10^9$ FLOPs = 1.54 GFLOP；算术强度 0.5 FLOP/B，远低于 A100 fp32 的 9.6 FLOP/B 平衡点。

| 规模 | 内存 | 一次全扫（A100） | 一次全扫（H100） | 纯算力下界（A100） |
| --- | --- | --- | --- | --- |
| $10^5$ | 307 MB | 151 µs | 92 µs | 7.9 µs |
| $10^6$ | 3.07 GB | 1.51 ms | 0.92 ms | 79 µs |
| $10^7$ | 30.7 GB | 15.1 ms | 9.2 ms | 0.79 ms |

**IVF / IVF-PQ 的扫描量**（$N=10^6$，$n_{list}=4096\approx4\sqrt N$，每桶平均 244 条）：

| $n_{probe}$ | 候选数 | 占全库 | IVF-Flat 读取 | IVF-PQ 读取（$m=96$） | 精确 MACs |
| --- | --- | --- | --- | --- | --- |
| 4 | 977 | 0.10% | 3.0 MB | 0.09 MB | $7.5\times10^5$ |
| 8 | 1953 | 0.20% | 6.0 MB | 0.19 MB | $1.5\times10^6$ |
| 32 | 7812 | 0.78% | 24.0 MB | 0.75 MB | $6.0\times10^6$ |

粗量化本身也不是免费的：$n_{list}=4096$ 时 $3.1\times10^6$ 次 MAC（一次 flat 全扫的 0.41%），$n_{list}=65536$ 时升到 $5.0\times10^7$（6.55%）——这就是把 $n_{list}$ 从 $10^3$ 量级提到 $2^{16}$ 量级时必须换成图索引做粗量化的原因。ADC 建表成本 $2^{8}\times768=1.97\times10^5$ 次 MAC，约为 $n_{list}=4096$ 粗量化成本的 6%。

**合成数据实测**（自行运行，脚本用 numpy；数据为 64 簇高斯混合，$d=128$、$N=2\times10^5$、100 条 query、$K=10$、$n_{list}=256$，向量 L2 归一化后用内积，ground truth 用 float64 全量精确计算）。绝对数值不能外推到真实 embedding——这批合成数据的近邻分数间距很小，是 IVF 与 PQ 都不友好的困难情形；能外推的是下面三条结构性结论，而不是具体数值。

| $n_{probe}$ | 扫描候选 | IVF-Flat recall@10（= cell bound） | IVF-PQ：ADC 分数 | +回表重排 100 | +回表重排 300 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1426 | 0.205 | 0.094 | 0.193 | 0.204 |
| 4 | 3836 | 0.324 | 0.127 | 0.279 | 0.316 |
| 8 | 7297 | 0.427 | 0.142 | 0.343 | 0.406 |
| 32 | 26241 | 0.649 | 0.165 | 0.446 | 0.572 |

- 结论一：IVF-Flat 的 recall 与 cell bound **完全相等**（上表每一位小数都一致），桶内精确距离不会引入额外损失；要提 recall 只能多探桶。
- 结论二：$m=16$、8 bit 的 PQ 把候选的排序打乱得很厉害——ADC 的 recall@10 只有 0.142，而同一候选集的理论上限是 0.427；回表重排到 300 条深度才把差距从 0.285 压到 0.021。重排深度是真实成本：按 $d=768$、$m=96$ 的口径换算，300 条 × 3072 B = 0.92 MB，与扫 7297 条 96 B 码（0.70 MB）同量级。
- 结论三：同一套 PQ 参数（$m=16$、8 bit、32× 压缩）在另一种分布上——簇间距远大于簇内散布，同簇 top-10 的分数几乎并列——全量扫描 recall@10 只有 0.091；换成噪声占比更大、近邻分数差更明显的分布后升到 0.187，该分布上的平均相对重建误差 $\|x-\hat x\|^2/\|x\|^2$ 是 0.32。两次运行的码长与码本配置完全相同（$m=16$ 段、每段 256 个码字），差别只在训练数据与分布。决定 recall 的因此不是重建误差的绝对值，而是「真实近邻的分数差 / 量化噪声」这个比值。

```python
# 内存与每次查询的扫描量（口径同上，单位 B / MAC）
N, d, fp32 = 1_000_000, 768, 4
print("flat      ", f"{N*d*fp32/1e9:.2f} GB", f"{2*N*d/1e9:.2f} GFLOP/query")
for M in (16, 32, 64):
    per = d * fp32 + 8 * M                 # 4d + 2M*4：向量 + 第 0 层图边
    print(f"HNSW M={M:2d} ", f"{N*per/1e9:.2f} GB", f"+{(per-d*fp32)/(d*fp32)*100:.1f}%",
          f"图边 {N*8*M/1e6:.0f} MB")
for m in (48, 96, 192, 384):
    print(f"PQ   m={m:3d}", f"{m} B/vec", f"{4*d/m:.0f}x", f"1M -> {N*m/1e6:.0f} MB",
          f"1B -> {10**9*m/1e9:.0f} GB")
nlist = 4096
print("粗量化", f"{nlist*d/1e6:.1f} M MAC", f"{nlist*d/(N*d)*100:.2f}% of flat scan")
for nprobe in (4, 8, 32):
    cand = N * nprobe / nlist
    print(f"nprobe={nprobe:2d}", f"扫 {cand:.0f} 条",
          f"fp32 {cand*d*fp32/1e6:.1f} MB", f"PQ96 {cand*96/1e6:.2f} MB")
```

生产里扫参数时用索引串构造索引，再逐个改属性（FAISS 的 `ParameterSpace` 就是把 `nprobe=8,efSearch=64` 这类字符串解析成这些属性，并递归到 `RFlat` 这类包装索引的子索引上）。索引串一眼能看出用了哪几种手段（`IVF4096,PQ96` 是「4096 桶 + 96 字节码」，`HNSW32` 是「$M=32$ 的图」，`RFlat` 表示最后用原始向量重排）：

```python
import faiss, numpy as np

xb = np.random.rand(1_000_000, 768).astype("float32")   # 已归一化则内积=余弦
xq = np.random.rand(200, 768).astype("float32")
ps = faiss.ParameterSpace()

for factory in ("Flat", "IVF4096,Flat", "IVF4096,PQ96", "HNSW32",
                "IVF4096,PQ96,RFlat"):
    index = faiss.index_factory(768, factory, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        index.train(xb)                       # IVF 系需要训练；Flat/HNSW 不需要
    index.add(xb)
    for key in ("nprobe", "efSearch", "k_factor", "k_factor_rf"):
        for v in (4, 8, 32, 128):
            try:
                ps.set_index_parameters(index, f"{key}={v}")   # 按名字递归设置
            except RuntimeError:
                break                         # 这个索引没有该参数（如 Flat 的 nprobe）
            D, I = index.search(xq, 10)       # 与 flat 的 top-10 比，算 recall@10
    # 曲线：横轴 p50/p95 延迟，纵轴 recall@10；再记内存与建库时间
```

## 常见追问

- **追问**：为什么 HNSW 的召回不是 100%？
  - 要点：四个机制叠加。贪心 + 提前终止（候选队列里最近的候选比结果堆中最远的那个更远就停）会在局部最优停下；建图时的启发式剪枝为多样性主动丢边，图不是精确 kNN 图；删除是标记删除，墓碑降低有效连通性；$efSearch$ 太小。只有最后一项是查询时可调的，前 3 项要靠更大的 $M$、更大的 $efConstruction$ 或定期重建解决。
- **追问**：PQ 的量化误差如何影响排序？为什么必须回表重排？
  - 要点：PQ 的分数误差是各段量化误差之和，量级固定在「段内重建误差」上；当真实近邻的分数差小于它时名次被随机化——实测 ADC 的 recall@10 = 0.142，而同一候选集上限是 0.427。回表重排把「量化分数」换成「原始向量精确分数」，因此只要真实近邻出现在过采样的候选里，就能找回来（重排 300 条 → 0.406）。代价是过采样深度 × 每条原向量字节数，以及必须另存原始向量。
- **追问**：为什么 IVF 的 $n_{list}$ 常取 $\sqrt N$？$N$ 很大时为什么不继续这么取？
  - 要点：两项拉扯——粗量化成本 $\propto n_{list}$ 线性涨，而桶越小近邻越容易跨桶、$n_{probe}$ 必须跟着涨。$N=10^6$ 时 $\sqrt N=1000$，实用上取 2 的幂 1024（SIFT1M 的经典设置），公开工程指南给的范围是 $4\sqrt N\sim16\sqrt N$，并建议每质心至少 30～256 条训练样本，否则质心质量崩掉。$N$ 到 $10^8$ 时按这条规则要 $10^4$ 个桶、每桶 $10^4$ 条，探 0.2% 的桶根本保不住 recall，得探几百个桶（扫 2%～10% 的数据）；于是把 $n_{list}$ 提到 $10^5\sim10^6$ 量级，而线性粗量化这时一次要 $10^5\times768\approx7.7\times10^7$ 次 MAC（在 $10^8$ 规模上相当于扫 0.1% 数据的成本，且每条 query 都固定发生），所以改用 HNSW 做粗量化，把这项从线性压到对数级。
- **追问**：过滤条件很多（权限、时间、类别）时怎么办？
  - 要点：选择率 $s$ 决定一切。post-filter 要把候选深度放大到 $k/s$（$s=1\%$、$k=10$ 时是 1000），延迟上一个量级；pre-filter 在 IVF 上只是桶内跳点，在 HNSW 上会让遍历断掉，需要过滤感知遍历或分区索引。常规折中是按租户/时间分区建小索引加查询路由，小分区用 flat。相关设计见 [[rag-08]]。
- **追问**：量化后的向量还能用于 rerank 吗？
  - 要点：可以用于「粗排」，不能用于最终决策。量化分数只决定谁进短名单，短名单里的名次必须用原始向量（或更高精度的表示）重算；如果想要更省，可以在重排阶段用标量量化（SQ8）的原向量，精度损失远小于 PQ。这也是 [[rag-03]] 里 cross-encoder 精排的同构结构：召回宽、精排窄。
- **追问**：语料持续变化、需要频繁删除和更新时，三种索引各有什么代价？
  - 要点：flat 删除是墓碑或整体重建；IVF 插入便宜（算一次质心距离入桶），删除是墓碑，分布漂移后必须重训质心；HNSW 增量插入便宜、但多数实现不支持真正删除（只能墓碑 + 重建），$efConstruction$ 越大重建越贵。所以「写多」的场景要么接受墓碑并定期 compaction，要么用蓝绿索引切换（见 [[rag-11]]）。

## 相关题目

- [[rag-02]]：稀疏与稠密检索的选择、hybrid 融合。召回阶段的候选质量决定了这里讨论的索引参数值不值得继续调。
- [[rag-01]]：chunking 决定 $N$ 与每条向量的内容，是索引规模的源头；chunk 变大意味着 $N$ 变小但每条更长。
- [[rag-03]]：cross-encoder reranker 与「宽召回 + 窄精排」的两阶段漏斗，和 PQ 的过采样加回表重排是同一个结构。
- [[rag-11]]：索引新鲜度、删除与重建；「原理与推导」第 5 节里的「删除是墓碑」在那里展开成完整的写入链路。
- [[rag-08]]：权限感知 retrieval；这里的 pre-filter / post-filter 选择率分析是它的一半内容。
- [[rag-04]]：评估口径；recall@k、nDCG 与端到端答案正确率的关系决定 recall 目标定在哪里。
- [[rag]]：专题导读，索引选型在整条 RAG 流水线里的位置。

## 参考资料与归属

1. [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)，Amit Shekhar（Outcome School）。提供暴力检索的代价直觉（10 亿条 × 1000 维，一次查询 $10^{12}$ 次乘法-累加；按这里的口径换成 FLOPs 要乘 2）、recall 的定义与速度-精度取舍、KD-Tree/LSH/IVF/HNSW 四类方法的机理与对照表（IVF 的 `nprobe` 旋钮、HNSW 层次图与「内存更多、建库更慢」的代价）、以及 FAISS 的 IVF 代码骨架（10 万条、128 维、100 簇、`nprobe=10`）。
2. [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)，Amit Shekhar（Outcome School）。提供三种相似度度量的公式与算例、ANN 与 recall 的口径、HNSW/IVF/PQ 的直观解释、IVF 的算例（1 亿条分 100 簇、探 3 簇 = 300 万条 = 3%）、PQ 的压缩算例（128 维 × 4 B = 512 B → 8 字节码，64×）、pre-filter 与 post-filter 的取舍，以及 DiskANN、ScaNN 等索引方向。
3. 超出上述两条的部分为自行推导与实测：「原理与推导」里的复杂度与内存公式（$4d+8M$、$M^{-l}$ 的层级分布、ADC 成本 $2^{b}d$）、$n_{list}$ 的工程区间与训练样本量要求、大 $N$ 场景改用图索引做粗量化，来自 HNSW 原论文与 FAISS 索引选择指南的公开结论，本次未列延伸链接；「数值与代码验证」的实测表与全部数值计算由本次运行得到，合成数据设置与 $d=768$ 的换算口径已在文中写明，未引用任何 benchmark 数字。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
