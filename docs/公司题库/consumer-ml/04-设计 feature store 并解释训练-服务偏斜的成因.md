---
type: question
id: consumer-ml-04
company: 面向消费者的规模化 ML 公司（Uber、Netflix、LinkedIn、Airbnb、Pinterest、Spotify）
topic: ml-fundamentals
order: 4
question: 你会如何设计 feature store？造成 training/serving skew 的原因是什么？
question_en: How do you design a feature store, and what causes training/serving skew?
asked_at: []
level: 进阶
tags: [特征存储, feature-store, 训练-服务偏斜, point-in-time-correctness, 在线特征, 新鲜度-SLO, 数据泄漏]
sources:
  - title: Machine Learning Operations (MLOps): Overview, Definition, and Architecture（延伸）
    url: https://arxiv.org/abs/2205.02302
    author: Kreuzberger et al.
    published: 2022-05-04
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
  - title: Evidently 文档（延伸）
    url: https://docs.evidentlyai.com/
    author: Evidently AI
    published: 
related: [system-design-04, rag-11, evaluation-09, evaluation-05, evaluation-07, consumer-ml-02]
updated: 2026-09-29
---

## 一句话答案

> feature store 是「特征定义 → 物化 → 取数」这条流水线的平台化，固定五件套：特征注册中心（命名、实体键、口径、所有者、版本、血缘、PII 与保留期）、转换层（**同一份定义**编译出批与流两条执行路径）、离线 store（append-only 双时态 lakehouse，支持 as-of join 与回填）、在线 store（KV/宽列，最新值 + TTL，按实体键批量点查，p99 5–10 ms）、两个入口（`get_historical_features` 必须显式带 `event_time`，`get_online_features` 必须显式带 `feature_versions`）。
> training/serving skew 按「怎么修」分五类：逻辑重复实现、时间语义错、时点错、缺省与降级不同、类型单位精度版本不一致；双写竞态与分布漂移性质不同，不能混在里面当 skew 修。
> 结构上消除 skew 只有一个手段：serving 侧记特征快照日志，训练直接读日志而不是重算（log-and-wait），代价是新特征要等日志回填一到数天。

## 面试官在考什么

- **先澄清再画图**。实体是什么（rider/driver、account/profile、viewer×member、guest→member、playlist/artist）、特征数与基数、在线 QPS 与 p99 预算、新鲜度分档、是否需要时间旅行回填与合规删除。跳过澄清直接说「用 Redis + Flink」是这道题最典型的失分方式。
- **有没有算过容量账**。1 亿实体 × 500 个 double 特征是 400 GB/副本、双副本 800 GB，而向量侧同量级只有 102.4 GB（HNSW int8）——「哪些特征值得进在线 store」是第一个真取舍，不是细节。
- **是否理解 point-in-time correctness**。as-of join 的定义、append-only 双时态、无时间参数的读取应当默认拒绝，以及「最新值回填」如何把 AUC 抬到线上达不到的高度。
- **skew 的分类是否可执行**。每一类要能给出判据与修法，而不是一句「离线在线分布不一样」；还要能说清 drift 与 skew 的分界。
- **检测与验收**。三层对账（计数 / 内容抽样 / canary 探针）、serving 快照日志、PSI 前先量噪声底、固定的归因顺序。

常见错误答案：

- 把 feature store 讲成「一个 Redis 或一张宽表」，没有注册中心、没有版本与血缘、没有 as-of 语义，等于只答了五件套里的一件。
- 把 skew 当成「分布不一样」，于是去模型侧加权或重训；真实原因常常是管道 bug（漏去重、时区、窗口端点）或降级路径不同，重训只会把 bug 学得更牢。

## 原理与推导

### 1. 先澄清：这六件事决定架构

| 要澄清 | 影响 |
| --- | --- |
| 实体与键 | 一个 key 混多个实体（guest 与 member、设备与账号）是最隐蔽的 skew 来源 |
| 特征数与基数 | 10^8 实体 × 10^2–10^3 特征；每加一个 double 特征就是 0.8 GB/副本 |
| 在线 QPS 与 p99 | 整条链路 p99 在 100 ms 量级（[[system-design-04]] 的商品搜索漏斗全部预算只有 46 ms 量级、p99 留 150 ms），在线取数只能占 5–10 ms |
| 新鲜度分档 | 秒级（CDC）/ 分钟级（流式聚合）/ 小时–天级（蓝绿重建 + 别名切换） |
| 时间旅行与回填 | 能否 as-of 查询任意历史时刻；新特征上线是否要跑历史回填 |
| 删除与权限收回 | 删除延迟 SLO、tombstone 语义、是否要物理删除 |

### 2. 五件套骨架

```text
                 ┌── 特征注册中心：命名/实体键/口径/所有者/版本/血缘/PII/保留期
一份特征定义 ────┤
                 ├─ 批执行  → 离线 store（lakehouse，append-only + 双时态）
                 │              └─ as-of join → get_historical_features(event_time=…)
                 └─ 流执行  → 在线 store（KV/宽列，最新值 + TTL，p99 5–10 ms）
                                └─ 批量点查  → get_online_features(feature_versions=…)
serving 侧 → 特征快照日志（值 + feature_version + fallback 标记 + 实验分桶）
                                └─ log-and-wait → 训练集（不重算，结构上消除 skew）
```

关键约束只有一句：**在线路径只能做「按实体键批量点查」**，禁止在线跨表 join 与串行 RPC fan-out。

### 3. 容量先算账再选型

1 亿实体 × 500 特征 × 8 B = $4\times10^{11}$ B = 400 GB/副本，双副本 800 GB；换成 float32 是 200 GB。按「热点特征（秒/分钟级）× 全量特征（天级）」分层：50 个热点特征只占 40 GB/副本（全量的 10%）。

对照同规模向量侧（[[system-design-04]] 口径，10^8 条）：HNSW int8 是 102.4 GB，IVF-PQ（m=96）只有 9.6 GB。特征 store 是 int8 HNSW 的 3.91 倍、IVF-PQ 码的 41.67 倍；把 768 维 fp16 embedding 塞进在线 KV 是 153.6 GB，是 IVF-PQ 码的 16 倍——**大 embedding 放向量库，不进在线 KV**。

内存型实例按 0.005 USD/GB·小时、730 小时/月计：800 GB 要 11 台 96 GB 机型（按 80% 可用内存），\$2920/月；只放 50 个热点降到 2 台、\$292/月。这 10 倍差就是「哪些特征值得在线」的价格标签。

点查量同样要算：5×10^6 请求/天约 57.9 QPS，每请求取 200 个键就是 1.16×10^4 次 key 点查/秒；峰值 10^4 QPS 时是 2×10^6 次/秒，按 200 键一次批量 MGET 归约成 10^4 op/s——所以在线 store 的分片依据是容量与批量点查吞吐，不是 raw QPS。顺便对照：LLaMA-3-70B 每 token 的 KV cache 是 320 KiB，一个 4k 上下文请求就是 1.25 GiB，特征 store 烧的是内存与网络预算，和显存是两本账。

### 4. 双执行：一份定义编译两条路径

离线 SQL/Pandas 与在线 Java/C++ 各写一遍同一聚合，两侧必然在去重、空值、浮点归约顺序、时区上发散。正确做法是特征 DSL 编译到批（Spark/SQL）与流（Flink/增量）两条执行路径，在线侧只读物化值、不做二次计算。

为什么禁止在线二次计算：串行 fan-out 的尾延迟不是叠加而是放大。单次调用「1 ms 固定开销 + 指数尾巴，p99 = 5 ms」时，串行 8 次的均值 14.95 ms、p99 21.90 ms（放大 4.38 倍），串行 20 次 p99 47.66 ms（9.53 倍）——46 ms 的漏斗预算一次就花光。带宽反而不是瓶颈：一次批量取 200 个特征的 payload 是 4000 B，1 Gbps 下只要 32 µs，**成本在 RTT 与串行化上**。

一致性用 CI 断言，而不是文档纪律：同一份输入跑两条路径，数值特征相对误差 ≤ $10^{-3}$、类别特征不一致率 ≤ $10^{-3}$。

### 5. 时间语义：as-of join 是分水岭

离线 store 保留 append-only 的 `(entity_key, feature_name, event_time, write_time, value)`，查询必须显式传 `as_of`：每一行样本 `(entity, event_time, label)` 取的是 event_time 之前最后一次物化的值，而不是「今天的最新值」。

纪律三条：① 无时间参数的读取默认拒绝；② 迟到数据用 watermark 界定重算窗口，超窗值视为不可变，并把「重算会影响哪些训练集」写进流水线，否则同一份训练集会因重跑时间不同而不可复现；③ 回填是一等公民——新特征先跑历史回填，用回填结果与线上日志交叉验证，不要「先上线、以后补」。

代价也要说清：1 亿实体每特征每变更一行是 $\{8+8+8\}$ B = 24 B，每天一版就是 2.40 GB/版本/特征，一年 876 GB/特征，500 个特征是 438 TB/年——所以**只对高 churn 特征做逐变更双时态，稳定特征走日快照 + 分区裁剪**。

### 6. 新鲜度是三个可验收 SLO

把「新鲜度」这个形容词拆成三个量（[[rag-11]] 的口径）：变更→可被取到的延迟 $T_{visible}$、旧值→不可再取到的延迟 $T_{retire}$、删除/权限收回生效的延迟 $T_{delete}$。三者是三条链路，多数团队只做第一条，结果线上读到已被修正的旧值、被删除用户的特征还躺在 KV 里。

写路径的机制固定为「至少一次投递 + 幂等 + 版本单调 = 可见结果等价于恰好一次」：事件携带 `(entity_key, feature_version, event_time)`，只接受版本更大的写入；先写新值再退休旧值（窗口期是「新旧都取得到」= 排序问题，不是「都取不到」= 可用性问题）；删除走 tombstone 而非立即物理删。没有幂等键时，$10^{-3}$ 的重复投递率就足以让约 $10^5$ 个实体的累加特征被双倍计入。

调新鲜度是拿写吞吐换可见延迟：Elasticsearch 默认每 1 秒 refresh 一次，且只对最近 30 秒内收到过搜索请求的索引生效；把间隔从 1 s 放宽到 30 s，可见延迟差 30 倍，写放大只从 5.9× 降到 4.5×（省 23.7%）——这条曲线明显递减（[[rag-11]]）。另外缓存 key 必须带 `feature_version` 与 embedding 模型版本，否则「特征已更新、线上仍命中旧值」会持续到缓存自然过期。

### 7. skew 的五类成因（外加两类性质不同的）

① **逻辑重复实现**：两侧在去重、空值、窗口端点上发散。复算：离线自然日 7 天 vs 在线滑动 7×24 h 让 99.7% 的实体取值不同、相对误差中位 4.5%（其中 6.5% 是来自 t 之后事件的纯泄漏）；漏按 event_id 去重影响 84.1% 的实体、相对误差中位 6.4%；在线缺值填 0 而离线 dropna，会让 1.1% 的实体在训练集里整行消失。三处叠加后中位相对误差 7.2%，是 $10^{-3}$ 阈值的 72 倍。

② **时间语义错**：事件时间 vs 处理时间；窗口是 $[t-7d, t)$ 还是 $(t-7d, t]$；自然日切还是滑动 7×24 h；UTC 还是本地时区/夏令时。UTC 自然日与本地自然日（UTC-7）之间，每天 29.2% 的事件落进相邻日桶、窗口整体平移 7 小时。Uber 的跨时区城市、Netflix 的跨区域会员、Spotify 的家庭账号会把「边界差一天」放大成节假日与冷启动期的特征失真。

③ **时点错**：用最新值回填等于把曝光之后才产生的行为喂给模型。复算：as-of 特征 AUC 0.7440，最新值/全历史特征 AUC 0.8160（虚高 +0.0720，泄漏项单独就有 0.8101）；日更特征在 90 天窗口下，98.9% 的样本时刻之后至少还有一次变更，也就是这些行携带的值与 event_time 不一致。

④ **缺省与降级不同**：在线取数超时或 key 缺失时静默填 0 或全局均值，离线则 dropna 或填中位数。0.5% 的 fallback 率在 500 万请求/天里是 25000 条/天、90 天 225 万条，训练集里却完全不存在这些行，模型学到的是「理想特征」。正确做法是 fallback 必须打标并落日志、监控比例，超阈值即当作特征不可用而不是继续服务。

⑤ **类型/单位/精度/版本**：float64→float32 截断、毫秒 vs 秒、分 vs 元、枚举新增码值的 unseen 处理、embedding 模型版本不一致（训练用 v1、在线用 v2 等于把同一条 query 放进两套语义空间，[[rag-11]] 的「索引版本绑定 `(model_id, dim, normalize, chunker_version)`」是同一纪律，[[system-design-04]] 的「query 与文档必须用同一个 embedding 模型」同源）。

另两类常被混进来，必须单独说：**双写竞态**（离线与在线两条写入一成功一失败，或训练吃全量历史而在线只有 TTL 窗口内的值——这是工程缺陷，要修管道）；**分布漂移**（用户与内容构成变了，实现没错——要走重训判据与 drift 监控，不能当成 skew 去改代码，[[evaluation-05]] 的原因 ① 与 ⑥）。

### 8. 检测与验收：三层，且不重训就能定位

① **同一时刻对账**：同一实体键、同一 event_time，离线重算 vs 在线日志里实际取到的值逐特征比对，输出 skew 率看板。沿用 [[rag-11]] 的三层结构——计数对账按小时、抽样 1% 内容对账、每分钟更新的 canary 探针断言端到端可见延迟。阈值与最小样本量事前写死：要在 $p=10^{-3}$ 的不一致率下有 95% 概率至少抓到一条，需要 2995 条对账样本；要把 $p$ 估到 ±20% 相对精度需要 95944 条。按 1% 抽样、500 万请求/天是 5 万条/天，攒够约 2 天。

② **serving 侧特征快照日志（log-and-wait）**：把线上真正用到的特征值 + `feature_version` + fallback 标记 + 实验分桶一起落库，训练直接读这份日志而不是重算——这是结构上消除 skew 的唯一手段，代价是「新特征要等日志回填」（通常一到数天）。体量可控：200 个特征 × 4 B = 800 B/请求，500 万请求/天是 4.0 GB/天原始，同一实体重复取值下 zstd 实测 8.5×，压实后 0.5 GB/天；对照 [[evaluation-09]] 的全量请求日志口径是 30 GB/天。

③ **影子/双算 + 统计监控**：影子链路用离线口径重算并与在线值比对（[[evaluation-09]] 的 L1 影子流量），特征分布按实体分桶看 PSI/KL、缺失率与分位点。但**先量噪声底再定阈值**：A/A 下 PSI 的 95 分位在 n=1000 时是 0.0576、n=10000 时 0.0041、n=10^5 时 0.00076——同一份数据切两半就能造出 0.005 量级的「漂移」。经验阈值只能当起点（PSI < 0.1 无明显偏移、0.1–0.25 中等、> 0.25 显著，且它很钝：数值特征整体上移 25% 才 0.089，类别分布挪 8% 是 0.176）。归因顺序固定：先查工程与数据管道（版本、上游断流、超时降级），再查训练-服务一致性，最后才怀疑模型与指标（[[evaluation-05]] 的诊断流程、[[consumer-ml-02]] 的裁决树在本题直接复用）。每次「特征/管道/模型」变更都要保留 1% 长期 holdback 与可回滚的版本指针，日志必须带版本标注，否则线上变差只能靠猜（[[evaluation-07]]）。

### 9. 六家平台把同一套骨架落到各自的实体与时间尺度

Uber 的 ETA 与派单是 < 100 ms 的同步链路，司机位置与供给量秒级、涌价与需求预测分钟级，特征按城市（市场）分区，训练窗口必须与线上窗口同源，否则跨时区的早高峰特征整体错位。Netflix 的播放埋点来自 T+1 离线与在线会话两条来源，实体必须区分 account 与 profile，新片冷启动没有行为特征只能靠内容侧兜底（[[system-design-04]] 的冷启动追问同一口径）。LinkedIn 的特征是 viewer×member 对级（连接度、互动历史），基数与稀疏性极大，在线只允许点查 + 本地缓存，图谱类特征走离线预计算。Airbnb 是双边市场，价格/库存/可订状态秒级、供需与季节性小时级，任何特征上线都要同时看需求侧与供给侧指标，否则一侧被挤压会被大盘掩盖。Pinterest 的曝光流基数无界、短时兴趣窗口短，特征键的哈希与去重口径不一致会直接变成「同一次曝光在训练与线上不是同一个特征」。Spotify 的播放、播放列表与音频 embedding 多模态混合，长尾内容特征稀疏、家庭账号共享设备导致实体归因错。六家的共同点：**特征必须在「请求时刻」可获取且可解释**，任何依赖离线全量数据的特征不得进入在线路径——这也是各家把特征平台做成平台级基础设施的原因。

### 10. 取舍三组与反模式

- **预计算 vs 实时计算**：消费级规模默认绝大多数特征离线/流式预计算后物化、在线只读；只有真需要「秒级变化且影响排序」的特征（库存、可用司机/座位、涌价、限时促销）走在线计算，因为每次 fan-out 都在 p99 上收税。算力不是瓶颈——H100 bf16 dense 989 TFLOPs、HBM 3.35 TB/s 下 roofline 拐点是 295 FLOPs/byte，而特征聚合的算术强度只有个位数 FLOPs/byte，落在带宽一侧，瓶颈永远是 I/O 与带宽。
- **单 store vs 双 store（离线 + 在线）**：双 store 是 skew 的结构性来源，用「同一份定义编译两条路径 + 在线只读物化值 + serving 快照日志」把可比性做成工程约束，而不是靠文档纪律。
- **log-and-wait vs 重算**：前者结构一致但新特征要等日志回填（一到数天），后者上线快但要长期养一套一致性校验。默认值应写死为：**排序/出价等直接决定收益的特征走 log-and-wait，探索性特征允许重算但必须挂对账看板**。

反模式逐条写死：把离线数仓当在线 store（p99 不可控、批任务一抖动就打垮线上）；在线做跨表 join 或串行 RPC 算特征；特征口径写在 wiki 而不是代码里；「先上线再补回填」；一个 key 混多个实体；把 drift 当 skew 修；把 skew 当「模型不稳」去调参；以及最贵的一条——**没有 serving 侧特征日志**，于是每次「线上效果变差」都只能靠猜。验收标准一句话：对任意一次线上请求，能回答「它用了哪个版本的哪些特征、值是多少、是不是降级值、离线按同一时刻重算是否一致」。

## 数值与代码验证

口径：以下数字由 `.work/consumer04_calc.py` 与 `.work/consumer04_doc.py` 在本机复算（AMD Ryzen 9 8945HX、单进程、pandas 2.3.3 / pyarrow 18.1.0 / NumPy 2.2.6），绝对值不可外推到生产集群；H100 与 KV cache 等常数沿用仓库统一口径。

**容量与延迟预算**

| 项 | 计算 | 结果 |
| --- | --- | --- |
| 在线 store 单副本 | $10^8 \times 500 \times 8$ B | 400 GB（372.5 GiB） |
| 每加一个 double 特征 | $10^8 \times 8$ B | 0.80 GB/副本 |
| float32 化 / 双副本 | $10^8 \times 500 \times 4$ B；×2 | 200 GB；800 GB |
| 热点 50 个特征 | $10^8 \times 50 \times 8$ B | 40 GB/副本（全量 10%） |
| 内存成本（0.005 USD/GB·h × 730 h） | 800 GB vs 80 GB | \$2920/月（11 台 96 GB）vs \$292/月（2 台） |
| 向量侧对照 | $10^8 \times 1024$ B / $\times 96$ B | 102.4 GB / 9.6 GB（特征侧是 3.91× / 41.67×） |
| 在线取数预算 | 5 ms 与 10 ms | 46 ms 漏斗的 10.9% / 21.7%，p99 150 ms 的 3.3% / 6.7% |
| 串行 8 次 RPC | 单次均值 1.87 ms、p99 5 ms | 均值 14.95 ms，p99 21.90 ms（4.38×） |

**训练集体积（压缩比是本机实测后外推，口径写在左列）**

| 列构成 | 实测压缩比 | 10^9 行 × 300 float32 外推 |
| --- | --- | --- |
| 按 (entity, event_time) 聚集 + zstd | 4.59× | 261 GB |
| 按 (entity, event_time) 聚集 + snappy | 3.63× | 330 GB |
| 混合列、未排序 + snappy / zstd | 1.21× / 1.34× | 990 GB / 894 GB |
| 全 float 与高熵 id 宽表 + snappy | 0.87×（反而膨胀） | 1.38 TB |

原始字节是 $10^9 \times 300 \times 4$ B = 1.2 TB。**压缩比不是常数，取决于列构成与是否按实体聚集**——简报里「压缩后仍在百 GB 量级」对应的是按实体聚集 + zstd 的那两档（261–330 GB）；若不排序、列又全是连续 float，Parquet 只能压到 0.87×，比原始还大。纯 I/O 下界：1.2 TB 按 2 GB/s 顺序读写是 10 分钟/遍（单机，未算解码与 join）。as-of join 本身实测 1.08 M 样本/s（4000 万版本行全驻内存），外推 10^9 行只要 0.3 小时算力，所以**重算训练集的成本由分区流式 I/O 与解码主导，不是 join**。

**skew 复算（合成数据，20000 实体 / 8 万事件 / 9 天）**

| 口径差 | 不一致面 | 量级 |
| --- | --- | --- |
| 离线自然日 7 天 vs 在线滑动 7×24 h | 99.7% 实体 | 相对误差中位 4.5%、p90 11.9%，其中 6.5% 是 t 之后的纯泄漏 |
| 漏按 event_id 去重（5% 重复投递） | 84.1% 实体 | 相对误差中位 6.4% |
| 在线填 0 vs 离线 dropna | 1.1% 实体 | 这些实体在训练集里整行消失 |
| 三处叠加 | 99.1% 实体 | 中位相对误差 7.2% = $10^{-3}$ 阈值的 72 倍 |
| 最新值 join vs as-of join | 全部样本 | AUC 0.8160 vs 0.7440（虚高 +0.0720） |
| 日更特征用最新值回填 | 90 天窗口 | 98.9% 的样本携带与 event_time 不一致的值 |

**监控阈值前先量噪声底**

| 场景 | PSI |
| --- | --- |
| 数值特征整体上移 1% / 5% / 25% / 50% | 0.00018 / 0.00436 / 0.08914 / 0.28428 |
| 类别分布把 1% / 5% / 8% 质量换桶 | 0.00201 / 0.05493 / 0.17578 |
| A/A 噪声底 95 分位（n=1000 / 10^4 / 10^5 / 10^6） | 0.0576 / 0.0041 / 0.00076 / 0.00006 |

读法：n=10^4 时的噪声底 0.0041 已经盖住「整体上移 5%」的 0.00436，所以不先量噪声底就定阈值，等于把采样波动当 drift；反过来经验阈值 0.1 极钝，整体上移 25% 都只有 0.089。

**一段可运行的 as-of 检查（对应第 5、7 节）**

```python
import numpy as np
import pandas as pd

# 特征表：append-only，带 event_time；迟到数据由 watermark 界定重算窗口
feat = pd.DataFrame({
    "entity": [0, 0, 0, 0, 1, 1, 1, 2, 2],
    "event_time": [1, 2, 7, 9, 3, 3, 5, 4, 6],
    "event_id": ["a1", "a2", "a3", "a4", "b1", "b1", "b2", "c1", "c2"],
    "value": [1.0, 1.0, 1.0, 100.0, 2.0, 2.0, np.nan, np.nan, np.nan],
})
sample = pd.DataFrame({"entity": [0, 1, 2], "event_time": [7, 5, 8], "label": [0, 1, 0]})
W = 7.0

def agg(dedup=True, fill0=False, closed_right=False):
    """同一份定义的一种执行路径：只允许在去重 / 缺值 / 窗口端点三处有口径差"""
    e = feat.drop_duplicates("event_id") if dedup else feat
    out = []
    for _, r in sample.iterrows():
        t, em = r["event_time"], e.entity == r["entity"]
        lo = (e.event_time >= t - W) if closed_right else (e.event_time > t - W)
        hi = (e.event_time <= t) if closed_right else (e.event_time < t)
        v = e[em & lo & hi].value
        out.append(0.0 if fill0 and (v.isna().all() or v.empty) else
                   (np.nan if v.isna().all() or v.empty else v.sum()))
    return np.array(out)

base = agg()
print("as-of 基准      ", base)                       # [2. 2. nan]
print("漏去重          ", agg(dedup=False))            # [2. 4. nan]  entity 1 被双倍计
print("缺值填 0        ", agg(fill0=True))             # [2. 2. 0.]   离线 dropna 丢样本
print("窗口右闭(含 t)  ", agg(closed_right=True))      # [3. 2. nan]  把 t 时刻的事件也算进来
```

实际输出：

```text
as-of 基准       [ 2.  2. nan]
漏去重           [ 2.  4. nan]
缺值填 0         [2. 2. 0.]
窗口右闭(含 t)   [ 3.  2. nan]
```

三个变体各自对应一种最常见的 skew：实体 1 因 `event_id` 重复被计两次（相对误差 1.0，超 $10^{-3}$ 阈值）；实体 2 在离线因 dropna 丢掉整行样本、线上照常服务；实体 0 的窗口右闭会把 $t$ 时刻（曝光时刻）的事件也算进来（此处 $t=7$ 恰好有一条 value=1.0 的事件）。若把同一张表按「最新值/全历史」聚合，实体 0 会拿到 103.0 而 as-of 只有 2.0——放大 51.5 倍，这就是「离线 AUC 虚高、线上无收益」的最小可复现版本。

## 常见追问

- **追问**：为什么不让在线直接查数仓，省掉一套在线 store？
  - 要点：p99 不可控且批任务抖动会直接打垮线上；10^9 行的训练集重算一次就是 1.2 TB 原始字节的 I/O，在线逐请求回算历史特征在这类规模下必然超时。在线 store 的存在理由是「把计算从请求路径上挪走」，代价是要养一致性校验。
- **追问**：至少要一次投递，怎么保证特征不重复计？
  - 要点：事件带 `(entity_key, feature_version, event_time)` 三元组，写入只接受版本更大者，读取按版本去重；存储侧保证「先写新值、再退休旧值」，于是窗口期是「新旧都取得到」（排序问题），不是「都取不到」（可用性问题）。删除走 tombstone + 后台压实，物理删除按合规窗口执行。
- **追问**：缓存里怎么保证不会一直命中旧特征？
  - 要点：缓存 key 必须含 `feature_version`，embedding 类特征还要含模型版本与归一化参数；否则「特征已更新、线上仍读旧值」会持续到自然过期。这和 [[rag-11]] 的 `index_version` 进检索缓存 key 是同一条纪律。
- **追问**：怎么判断线上变差是 skew、drift 还是模型问题？
  - 要点：顺序固定——先对账（同一 event_time 离线重算 vs 线上实取值，skew 率超阈值就是 skew），再看特征分布 PSI/缺失率/fallback 比例（先量噪声底），最后才看模型指标与业务指标。skew 可修复、drift 要重训、模型退化要回滚，三者的动作完全不同（[[consumer-ml-02]]）。
- **追问**：新特征上线流程是什么？
  - 要点：注册中心登记口径与所有者 → 同一份定义编译两条路径 → CI 双路径一致性断言 → 历史回填并交叉验证 → 在线 store 物化 → 影子/小流量验证 → 打开 serving 快照日志的采样率 → 进训练集。缺回填或快照日志的一步就是「先上线、以后补」，事后必然要重算。
- **追问**：为什么大 embedding 不放在线 KV？
  - 要点：768 维 fp16 是每条 1536 B，1 亿条就是 153.6 GB，是 IVF-PQ（m=96）码 9.6 GB 的 16 倍；向量库还能顺带做 ANN 检索与量化，KV 只能做精确点查。特征侧的取舍同理：只有热点特征配得上内存成本。

## 相关题目

- [[system-design-04]]：同量级目录的语义搜索漏斗给出了 46 ms 量级预算、150 ms p99、HNSW int8 与 IVF-PQ 的字节账，本题的在线预算与向量侧对照直接复用它的口径。
- [[rag-11]]：三档新鲜度的现实映射、`refresh_interval` 的写放大曲线、三层对账与「索引版本绑定模型版本」——本题的新鲜度 SLO 与对账结构来自这里。
- [[evaluation-09]]：四级阶梯（离线回归 → 影子 → A/B → 全量）、serving 日志的字段与体量口径、版本四元组，本题的 log-and-wait 与快照日志是它在特征侧的版本。
- [[evaluation-05]]：离线涨、线上不涨的七类原因与诊断流程，用于把 skew 与 drift、代理指标失败区分开。
- [[evaluation-07]]：生产可观测性要记录哪些字段，本题「日志必须带版本标注」的出处。
- [[consumer-ml-02]]：裁决树与测量错觉，用来回答「线上变差先怀疑谁」。

## 参考资料与归属

- **Machine Learning Operations (MLOps): Overview, Definition, and Architecture（延伸）** —— Kreuzberger et al.，2022-05-04：<https://arxiv.org/abs/2205.02302>。特征平台/训练-服务管线的架构与角色划分。
- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。特征与训练-服务一致性的工程规则。
- **Evidently 文档（延伸）** —— Evidently AI：<https://docs.evidentlyai.com/>。漂移与数据质量监控的工具形态参考。

- **延伸来源说明**：第 3–5 节的偏斜分类与存储/延迟算例，是按本仓库统一口径（H100 bf16 dense 989 TFLOPs、每 token KV 按模型规模取 128 KiB / 320 KiB、\$2/GPU 小时等）自行推算的工程算例，不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
