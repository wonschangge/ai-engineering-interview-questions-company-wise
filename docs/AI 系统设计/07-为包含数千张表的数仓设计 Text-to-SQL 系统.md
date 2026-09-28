---
type: question
id: system-design-07
topic: AI 系统设计
order: 7
question: 为包含数千张表的数仓设计一个 Text-to-SQL 系统。
question_en: Design a text-to-SQL system for a warehouse with thousands of tables.
asked_at: [Databricks, Palantir]
level: 高阶
tags: [系统设计, text-to-sql, schema-检索, 自纠错]
sources:
  - title: Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task（延伸）
    url: https://arxiv.org/abs/1809.08887
    author: Yu et al. (EMNLP 2018)
    published: 2018-09-24
  - title: Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs（BIRD）（延伸）
    url: https://arxiv.org/abs/2305.03111
    author: Li et al. (NeurIPS 2023)
    published: 2023-05-04
  - title: Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows（延伸）
    url: https://arxiv.org/abs/2411.07763
    author: Lei et al. (ICLR 2025)
    published: 2024-11-12
  - title: CHESS: Contextual Harnessing for Efficient SQL Synthesis（延伸）
    url: https://arxiv.org/abs/2405.16755
    author: Talaei et al.
    published: 2024-05-27
  - title: DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction（延伸）
    url: https://arxiv.org/abs/2304.11015
    author: Pourreza & Rafiei (NeurIPS 2023)
    published: 2023-04-21
related: [rag-09, agents-06, evaluation-10, system-design-06, agents-02]
updated: 2026-09-28
---

## 一句话答案

> 这题的真难点不是「写 SQL」而是 **schema linking**：把数千张表、数万列裁剪成每请求 2–5k token 的子模式，再生成、执行、自纠错。规模算账给出结论——全量 schema 120 万 token 直接不可行（每请求 366 GiB KV、预填充 53.5 s）；压到「表名 + 列名 + 类型」的 7.68 万 token 仍要每请求 23 GiB KV 与 3.4 s 预填充（8×H100）、每卡只放得下 2 条并发；裁到 3k token 后才降到 0.9 GiB 与 0.13 s、54 条并发；而整个系统的 LLM 推理成本只有约 \$47/天（2 万提问/天），真正的成本大头是数仓扫描与人工维护的语义层。因此顺序是：**语义层/指标平台优先路由 → schema 检索与裁剪 → 多候选生成与自纠错 → 只读执行护栏 → 结果校验与口径解释**。

## 面试官在考什么

- **是否先澄清需求**：表的规模、用户是分析师还是业务人员、查询复杂度（单表聚合 / 多表 join / 窗口函数）、有没有语义层（dbt / 指标平台）、方言、权限模型（行级 / 列级）、失败容忍度（报错可重试，**查错表得出错误数字是事故**）。
- **是否意识到这题的重心是检索而不是生成**：数千张表里找对那几张、并写对语义口径。答不出这一层，后面所有细节都没有立足点。
- **能否把规模算成数字**：schema token 数、KV cache、预填充时间、并发槽位、每问题 token 账、每天 GPU 小时。面试官用这一节分辨「背过架构图」和「真的做过容量规划」。
- **生成与验证策略是否成套**：一次性生成 vs 分解式生成 vs 多候选投票 vs 执行回灌自纠错，以及各自什么时候值得花这份算力。
- **执行侧安全护栏是否想全**：只读连接、扫描量上限、AST 改写、行列权限在生成期与执行期双重生效、审计日志、PII 不能落进索引和 trace。
- **评测口径**：学术基准（Spider 91.2% → BIRD 73.0% → Spider 2.0 21.3%，Spider 2.0 论文口径，均基于其 agent 框架）与企业内部评测的差别，以及为什么主指标必须是执行结果正确性而不是 SQL 文本相似度。

**常见错误答案**：

- 上来就讨论「怎么让模型写出更好的 SQL」，把 schema 检索当成一个可有可无的前置步骤。
- 认为「模型上下文够长，把整库 DDL 塞进去就行」。上下文装得下 ≠ 服务得起：见第 2 节的 prompt 体积、KV 与预填充三笔账。
- 权限只答「数据库有权限」。生成期不裁剪 schema 时，无权表名会出现在生成的 SQL 里，等于把表名、列名、甚至样例值泄漏给无权用户。

## 原理与推导

### 1. 需求澄清与假设

先钉死一组可讨论的口径：

- **规模**：2,000 张表、平均 30 列 → 6 万列；每列的名称 + 类型 + 描述 + 少量样例值按 20 token 计。
- **元数据**：有 table/column 注释与分区/clustering 信息；外键只在部分表上有声明（真实数仓常见，只能从 dbt 或血缘里推）。
- **用户**：分析师（能读改 SQL）与业务人员（只看数字）混合。
- **延迟与成本**：交互式，p95 端到端 10–30 s 可接受；允许每次提问花几分钱，但**不允许一次全表扫描悄悄烧掉几十美元**。
- **权限**：分析师与业务人员可见的表/列不同，行级与列级权限必须在生成期（schema 裁剪）与执行期（数据库自身权限）两侧都生效，见 [[rag-08]]。

### 2. 三个规模公式

**schema 体积**：设 $N_c$ 为列数、$t_c$ 为每列元数据 token 数，全量喂入的 prompt 长度是
$$S_{\text{full}} = N_c \cdot t_c = 60{,}000 \times 20 = 1{,}200{,}000\ \text{token}$$

**上下文占用**：LLaMA-3-70B 的 KV cache 是每 token 320 KiB（80 层 × 8 个 KV 头 × head_dim 128 × 2（K/V）× 2 B = 327,680 B）。于是单个请求的 KV 是

$$M_{kv} = S \times 320\ \text{KiB}, \qquad M_{kv}(76.8\text{k}) = 23.4\ \text{GiB}$$

76.8k token 是「只保留表名 + 列名 + 类型」的压缩版：6 万列平均每列 1.28 token（列名与类型都取最省的写法，去掉注释、描述与样例值），是全量版的 1/15.6——**这是乐观下界**，按每列 4–5 token 的正常分词就是 24–30 万 token、73–92 GiB KV，单卡同样放不下。即便按乐观口径，7.68 万 token 也仍然超过 32k 的常用服务窗口，只有在 128k 窗口下才装得进。每卡 80 GB（74.5 GiB）显存扣掉约 16.4 GiB 权重分片与工作区后按 50 GiB 给 KV 池：这个 76.8k 压缩版每卡只能放 **2 个**并发请求，而 3k token 的精筛版能放 **54 个**。

**预填充时间**：70B 模型每个 token 的前向约 $2 \times 70.55 \times 10^{9} \approx 1.4 \times 10^{11}$ FLOP，8 卡 H100、bf16 稠密 989 TFLOPs/卡、MFU 0.4：

$$T_{\text{pre}} = \frac{2 N_{\text{param}} S}{\text{TFLOPS} \times \text{TP} \times \text{MFU}} = \frac{2 \times 70.55\text{e}9 \times S}{989\text{e}12 \times 8 \times 0.4}$$

- 全量 schema（120 万 token）：$T_{\text{pre}} = 53.5$ s，**首 token 之前就超时了**；
- 压缩版（76.8k）：3.42 s；
- 精筛版（3k）：0.134 s。

三个数字合起来就是本题的核心论证：**上下文窗口能装下不等于服务得起**。裁剪不是省钱技巧，而是可行性的前提。第 4 节讲怎么做，容量与成本的总账在「数值与代码验证」一节。

### 3. 架构与数据流

**离线层**（T+1 刷新，是全系统准确率的地基）：元数据目录（information_schema / Hive Metastore / Unity Catalog / dbt manifest 里的表、列、类型、注释、分区与聚簇键、外键与血缘）、数据画像、多粒度向量索引、查询日志索引、血缘/ER 图、语义层。逐项说明：

1. **数据画像**：每列的唯一值数（cardinality）、空值率、min/max、Top-k 频次、少量样例值。它同时用于消歧（「status 有哪几种取值」）和护栏（哪张表是十亿行的事实表）。
2. **多粒度向量索引**：表级摘要向量 + 列级向量 + **样例值向量**（值级别的精确/近邻匹配往往比列名匹配更可靠；BIRD 正是靠强调「数据库取值」立论——脏数据、自然语言问题与数据库取值之间的外部知识、以及大规模数据库上的 SQL 效率，并给出当时最好的 ChatGPT 只有 40.08% 执行准确率、人类 92.96% 的对照，论文口径）。上例 6.2 万条 1024 维 fp32 向量只有 254 MB，单机内存足够装下。
3. **查询日志索引**：历史 SQL 的规范化文本 + 表列使用集合，用于「问相似问题的人用了哪些表」的协同信号。
4. **血缘/ER 图**：选中一张表后把 join 路径上的邻居一起带进候选，避免「选对了事实表却漏了维度表」。
5. **语义层**：canonical 指标定义（口径、过滤条件、时间粒度、维度）。

**在线链路**：

```
问题 → ① 路由（语义层命中？口径是否已定义？）
     → ② 候选召回：向量(表/列/样例值) ⊕ BM25 ⊕ 历史 SQL ⊕ 血缘邻居 → RRF 融合 → top-k
     → ③ 精筛：LLM schema selector 在压缩候选上选到最终子模式（表 + 必要列 + join 键）
     → ④ 生成：分解式生成 / 多候选 / 自洽投票
     → ⑤ 校验：AST 解析 → 只读改写（强制 LIMIT、分区过滤、拒绝 DDL/DML）→ 权限与成本预检
     → ⑥ 执行：只读副本，超时与扫描量上限，返回行数上限
     → ⑦ 结果校验：语义单元测试 + 行数合理性 +（高风险问题）对账查询
     → ⑧ 呈现：数字 + 口径/时间范围/过滤条件说明 +「模型生成的查询」或「已注册口径」标注
     └─ 失败 → 把 DB 报错或单元测试失败证据连同子模式回灌重写（≤3 轮）
```

关键设计判断：**口径已有定义时不要自由生成**。指标平台里的 `gross_margin` 与模型每次现写的 `revenue - cost` 在边界条件上几乎不可能一致，一致性比单次准确率更重要；自由生成只作为语义层覆盖不到的长尾兜底，并显式标注。

### 4. schema linking：本题的核心

候选打分是多信号的加权融合，而不是单靠向量相似度：

$$\text{score}(c) = w_1 \cdot \cos(e_q, e_c) + w_2 \cdot \text{BM25}(q, c) + w_3 \cdot \text{match}(q, \text{values}(c)) + w_4 \cdot \text{cooccur}(c \mid \text{log}) + w_5 \cdot \text{lineage}(c)$$

两条工程纪律：

- **样例值是第一等公民**。「上个月 ACME 的退款率」里的 `ACME` 只可能出现在数据里，列名里没有这个词；反过来，`status` 这种列名要靠 Top-k 取值才能判断是不是「退款状态」——CHESS 的信息检索器之一就是数据库取值的 LSH 索引。表级与列级要分开召回：join 键的正确性由表级关系决定，聚合与过滤条件由列级决定。
- **召回负责不丢，精筛负责不吵**。如果候选召回阶段没把正确表放进 top-k，后续 LLM 无论多强都救不回来——这是**不可恢复的损失**，所以 `recall@k`（gold 表是否落在候选集中）是离线必测的守门指标。CHESS 的 schema selector 在 4,337 列的合成 schema 上把 Pass@1 从 59% 提到 61%、Pass@5 从 61% 提到 63%（约 +2 个百分点，论文口径），同时把 token 用量降到约 1/5；它的三个工具（filter_column / select_tables / select_columns）都只做列级过滤，输入 schema 里没有的表不会被补进来。负向信号同样可用：被频繁 join 却从未与该表同时出现的组合、画像里 cardinality = 1 的常量列，都用于压分。

### 5. 生成策略对比

| 策略 | 做法 | 代价 | 适用 |
| --- | --- | --- | --- |
| 一次性生成 | 问题 + 子模式 → 一条 SQL | 1 次调用 | 简单单表聚合 |
| 分解式生成 | 先分类（简单/嵌套/复杂）→ 拆子问题 → 逐项生成 → 拼装 | 3–5 次调用 | 多表 join、嵌套聚合 |
| 多候选 + 自洽 | 生成 N 条候选，按执行结果一致性投票 | N 倍生成 + 执行成本 | 高风险问题、指标类查询 |
| 执行回灌自纠错 | 把 DB 报错/Dry-run 结果回灌重写 | 1–3 轮 | 语法错、列名错、类型错 |

论文口径的参照：DIN-SQL 把生成拆成子问题并把子问题答案喂回模型，在三个 LLM 上把朴素 few-shot 稳定提升约 10%，在 Spider holdout 上把执行准确率从 79.9 推到 85.3，在 BIRD 上达到 55.9%（论文自报 SOTA）。CHESS 在高算力预算设置下对每题生成 20 条候选、再用 10 条自然语言单元测试打分，取到 BIRD 测试集 71.10% 执行准确率，LLM 调用次数比同档方法少约 83%（论文口径）。

**投票的判据要选对**：比较 SQL 文本相似度会奖励「写法一样的错答案」，比较执行结果集合的一致性才能捕捉真正的语义分歧；对聚合类查询，结果一致但 SQL 完全不同恰恰是最常见的正确情形。

### 6. 执行安全与护栏

- **只读连接 + 资源上限**：独立只读账号、副本或只读端点，语句超时、扫描字节上限、返回行数上限、并发配额。禁止 DDL/DML，禁止无分区过滤的全表扫描。
- **AST 校验与改写**：解析成 AST 后（`sqlglot` 一类的方言感知解析器）再判断，正则匹配注定漏判。改写规则：强制加 `LIMIT`、按画像补分区过滤、拒绝非白名单函数、拒绝无 join 条件的笛卡尔积、把 `SELECT *` 展开成显式列（便于列级权限裁剪）。
- **权限双层**：生成期用 schema 过滤（无权表不进候选、无权列不进子模式），执行期用数据库自身的行级/列级权限兜底。只做后者会导致无权对象的名称出现在模型输出里；只做前者则一旦 LLM 幻觉出一个表名就成了越权尝试。这与 retrieval 的权限设计是同一个模式，见 [[rag-08]]。
- **成本护栏**：用数仓的 dry-run / 查询计划估算扫描量，超阈值就拦下或降级（先给抽样结果 + 提示「全量需扫描 X TB，预计 $Y」），思路与 [[inference-serving-12]] 的单位成本治理一致。
- **审计**：记录提问者、原问题、召回的候选集、最终子模式、生成的 SQL、计划成本、实际执行耗时与行数、是否走了语义层。审计日志本身是敏感资产——schema 描述和样例值可能含 PII，日志与 trace 都要脱敏，见 [[agents-10]]。

### 7. 写对语义：口径、时间与对账

SQL 语法正确、执行无报错、返回一个数字——这条路径完全可能是错的。企业场景里最常见的四类语义错误：

1. **同名不同源**：`orders` 在 ODS、DWD、ADS 三层各有一张，口径完全不同。解法是把分层与数据域写进表描述并参与检索打分，同时在提示里强制模型声明所用层级。
2. **时间语义**：`created_at` vs `paid_at` vs `shipped_at`；自然语言里的「上个月」是自然月、滚动 30 天还是财务月；时区是 UTC 还是业务时区。解法是把时间字段的业务含义写进列描述，并要求输出显式的时间过滤条件供用户核对。
3. **状态与口径**：有效订单是否排除退款、是否含税、金额单位是分还是元。解法是优先路由到语义层。
4. **隐式过滤**：内部测试账号、员工订单、压测数据。解法是把它做成语义层的默认过滤条件，而不是指望模型每次记得加。

配套的是**结果校验**：用 LLM 从原问题生成几条自然语言单元测试（「结果必须只含 2025 年 Q1」「每个客户一行」），对候选结果逐条判定；再叠加行数合理性（返回 1 行还是 200 万行）与量级检查。高风险问题追加对账查询（与语义层的同口径指标比对），差异超阈值就不直接呈现数字，改为提示「与指标平台不一致，需人工确认」。任务级评测的做法见 [[evaluation-10]]，缺 ground truth 时的评测集构建见 [[evaluation-04]]。

### 8. 评测与迭代

学术基准的演进正好说明难度的跃迁，但注意三者口径不同、不可直接横比：

| 基准 | 规模（论文口径） | 报告数字 | 它没覆盖的东西 |
| --- | --- | --- | --- |
| Spider（EMNLP 2018） | 10,181 问 / 5,693 条复杂 SQL / 200 库 / 138 领域，要求在未见过的库上泛化 | 当时最好模型在 database split 上仅 12.4% 精确匹配 | 脏数据、大 schema、真实企业工作流 |
| BIRD（NeurIPS 2023） | 12,751 对数据 / 95 库 / 33.4 GB / 37 个专业领域 | 最好的模型 ChatGPT 执行准确率 40.08%，人类 92.96% | 跨源、数仓方言、文档与元数据检索 |
| Spider 2.0（ICLR 2025） | 632 个真实企业工作流问题，库常含 1000+ 列，分布在 BigQuery/Snowflake 等系统，单条 SQL 常超 100 行 | 基于 o1-preview 的 agent 框架只解决 21.3%；同框架在 Spider 1.0 上 91.2%、BIRD 上 73.0% | 内部口径与权限 |

**这张表的用法**：Spider 1.0 的 91.2% 与 Spider 2.0 的 21.3% 出自同一篇论文的同一套框架，说明**学术基准上的高分不代表企业可用**。所以内部评测必须用自己的数仓问题集，按复杂度分层（单表聚合 / 多表 join / 窗口与嵌套 / 需口径澄清），逐层报通过率并分别设阈值——混在一起的总体通过率会被简单的单表查询拉高。上线路径同理由此确定：影子模式（只生成不呈现，人工比对）→ 10% 灰度（只对分析师开放、强制展示 SQL 与口径）→ 全量但高成本查询需确认；回归门禁见 [[evaluation-04]]。

主指标是**执行结果正确性**（结果集或聚合值一致），SQL 文本相似度只作为辅助信号。同时必须单独监控：

- **schema linking 的召回率**：候选集未命中 gold 表的比例。它决定了系统的上限，是最该先优化的指标。
- **澄清触发率与误触发率**：问题本身有歧义时才反问，且最多一次，用「假设句式 + 默认解释」给出临时答案，而不是把皮球踢回给用户，见 [[agents-11]]。
- **护栏拦截率**：被拦下的危险查询里，有多少是真正的误拦（这是体验损失）、多少是正确拦截（这是收益）。
- **单位成本**：每次提问的 LLM token、GPU 小时、数仓扫描字节与美元。

## 数值与代码验证

**容量与成本账**（全部按上文口径自算；H100 常数与 KV 口径与仓库其它专题一致）：

| 方案 | 送入 prompt | KV/请求 | 单请求预填充（8×H100, MFU 0.4） | 每卡可并发（50 GiB KV 池） |
| --- | --- | --- | --- | --- |
| 全量 schema | 1,200,000 token | 366 GiB | 53.5 s | 不可行（超出窗口与单卡显存） |
| 表名 + 列名 + 类型（乐观下界） | 76,800 token | 23.4 GiB | 3.42 s | 2 |
| 粗筛 top-20 表（共约 715 列 × 20 token） | 14,300 token | 4.4 GiB | 0.64 s | 11 |
| 精筛子模式 | 3,000 token | 0.92 GiB | 0.13 s | 54 |

从 76.8k 到 3k 是 25.6 倍 token 压缩，换来 27 倍并发槽位（50 GiB 池下 2 个 → 54 个）：这就是「检索即容量」的量化含义；注意 7.68 万 token 已是乐观下界，按每列 4 token 分词就是 24 万 token、73 GiB KV，单卡同样放不下——结论只会更强。

**全系统每天的成本**（假设：2 万提问/天，每问题 12 次 LLM 调用，平均每次 5,000 输入 + 300 输出 token）：

| 项 | 数值 | 算式 |
| --- | --- | --- |
| 每天 input token | 1.20 B | $20{,}000 \times 12 \times 5{,}000$ |
| 每天 output token | 72 M | $20{,}000 \times 12 \times 300$ |
| 预填充吞吐（8 卡 70B，MFU 0.4） | 22.4k token/s | $989\text{e}12 \times 8 \times 0.4 / (2 \times 70.55\text{e}9)$ |
| 预填充时间 | 14.86 GPU·h/天 | $1.2\text{e}9 / 22{,}429 / 3600$ |
| 解码时间（batch 32，权重读取主导） | 4.02 GPU·h/天 | 每步 $(141.1/8 + 3.93)\ \text{GB} / 3.35\ \text{TB/s} = 6.4$ ms，即 $32/6.4\ \text{ms} \approx 5.0\text{k}$ token/s（batch 32 的算术强度约 32 FLOP/byte，计入 KV 读取只会更低，远低于 H100 的 295 FLOP/byte 拐点，所以吃带宽不吃算力） |
| 合计 | 18.9 GPU·h/天 ≈ **0.79 个 8 卡节点** | |
| GPU 成本（\$2.5/GPU·h） | **\$47/天，\$1,417/月**，即每次提问 \$0.0024 | $18.89 \times 2.5$ |
| 若全部按 token 计价（input \$2.5、output \$10 / 1M，示例价） | \$3,720/天 | $1{,}200 \times 2.5 + 72 \times 10$ |

两点必须主动说明：**（a）** 预填充占约 79% 的时间，所以「少喂 schema」与 prefix caching 是同一件事的两面；**（b）** 自建 GPU 与按 token 计价差约 79 倍，报成本必须说清口径。真正的大头不在 LLM：一次数十 TB 的全表扫描在按量计费的云数仓上要几美元到几十美元，比该问题的全部 LLM 开销高三个数量级——护栏的重点是**别让模型决定扫多少数据**。

**容量账复算脚本**（纯标准库，运行结果与上表逐项一致）：

```python
GiB, GB = 1024**3, 1000**3
KV_TOKEN = 2 * 80 * 8 * 128 * 2          # LLaMA-3-70B bf16：327,680 B = 320 KiB/token
W_BYTES  = 70.55e9 * 2                   # 70B bf16 权重（十进制 GB）
PEAK, BW, TP = 989e12, 3.35e12, 8        # H100 SXM5 bf16 dense、HBM3、张量并行度
MFU = 0.4

# 1) 上下文占用与并发槽位
for S in (3000, 14300, 76800, 1200000):
    kv = KV_TOKEN * S
    print(f"S={S:>8}: KV={kv/GB:8.1f} GB = {kv/GiB:7.2f} GiB, 每卡 50 GiB 池 -> {50*GiB//kv} 槽")

# 2) 单请求预填充时间
for S in (3000, 14300, 76800, 1200000):
    print(f"S={S:>8}: prefill = {2*70.55e9*S/(PEAK*TP*MFU):7.3f} s")

# 3) 每天 GPU 小时与美元
Q, CALLS, T_IN, T_OUT = 20_000, 12, 5_000, 300
IN_D, OUT_D = Q*CALLS*T_IN, Q*CALLS*T_OUT
R_PRE = PEAK*TP*MFU / (2*70.55e9)                    # 8 卡预填充 tok/s
B, S = 32, 3000
STEP = (W_BYTES/TP + KV_TOKEN/TP*S*B) / BW           # 每步每卡读权重分片 + 本批 KV（与仓库 decode 口径一致）
R_DEC = B / STEP                                     # 整机 decode tok/s
h = IN_D/R_PRE/3600 + OUT_D/R_DEC/3600
print(f"R_PRE={R_PRE:,.0f} tok/s, R_DEC={R_DEC:,.0f} tok/s, {h:.2f} GPU·h/天, ${h*2.5:,.0f}/天")
```

**SQL 校验与改写的形状**（`pip install sqlglot` 后可跑；真实实现同样要走方言感知的 AST 解析，不要用正则）：

```python
import sqlglot
from sqlglot import exp

BANNED = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter)

def guard(sql: str, dialect: str, allowed_tables: set[str], max_rows: int = 1000):
    tree = sqlglot.parse_one(sql, read=dialect)
    if any(tree.find(k) for k in BANNED):
        return None, "拒绝写操作"
    for t in tree.find_all(exp.Table):
        if t.name not in allowed_tables:                # 生成期权限过滤的兜底
            return None, f"越权或幻觉表名：{t.name}"
    if any(not j.args.get("on") and not j.args.get("using") for j in tree.find_all(exp.Join)):
        return None, "拒绝无连接条件的笛卡尔积"
    if not tree.args.get("limit"):
        tree = tree.limit(max_rows)                     # 强制行数上限
    return tree.sql(dialect=dialect), None
```

## 常见追问

- **追问**：为什么不直接把全部 schema 塞进长上下文？现在的模型动辄 128k–1M 窗口。
  - 要点：三笔账一起算——全量 120 万 token 根本超窗口；即便压到 76.8k 能装下，单请求 KV 23.4 GiB 使每卡只能有 2 个并发，且预填充 3.42 s 直接吃掉 TTFT 预算；token 计费口径下 1.2B token/天就是每天数千美元。CHESS 的数字是 token 降到约 1/5 并 +2 个百分点准确率（4,337 列合成 schema，论文口径）。**上下文变长降低的是「装不下」的门槛，不改变「噪声会降低精度、KV 与预填充会吃掉并发」这两件事**。
- **追问**：用户的问题本身有歧义（「上个月的销售」指哪个销售）怎么办？
  - 要点：先做歧义检测（多候选 schema 分数接近、缺时间字段、指标名有多个口径都不是高置信信号），命中就澄清。设计约束：最多反问一次、一次只问一个关键问题、同时给出「按 A 口径的临时答案 + 假设说明」，而不是只回一句「请问您指哪个」；用户回答后写入会话上下文并作为该会话后续查询的默认口径，见 [[agents-11]]。
- **追问**：怎么防止「SQL 语法正确、跑得出来，但查错了表」？
  - 要点：这是本题最危险也最容易被忽略的失败模式，BIRD 的误差分析里 **wrong schema linking 占 41.6%、misunderstanding database content 占 40.8%**（ChatGPT 错误样本 500 例，论文口径）——八成以上的错误在 schema 层而不是 SQL 语法层。工程手段是四层：离线守住候选召回率、生成期强制声明所用表与口径、执行后用 LLM 自然语言单元测试判定、高风险查询与语义层同口径指标对账，不一致就不直接给数字，见 [[evaluation-04]]。
- **追问**：多轮对话里用户说「那再按地区拆一下」，怎么处理？
  - 要点：把会话状态显式化成结构化对象（时间范围、过滤条件、指标、维度、已确认的口径），只把**变更的槽位**喂给生成阶段，而不是把历史 SQL 全文拼进 prompt；同时保留上一轮的子模式与澄清结果作为上下文。这既省 token，也避免模型被上一轮的表达习惯带偏。
- **追问**：方言与跨源怎么处理？
  - 要点：方言差异（BigQuery 的 `EXTRACT`/`UNNEST`、Snowflake 的 `QUALIFY`、Spark SQL 的窗口与类型转换）会让「语义正确」的 SQL 执行失败。做法是把方言作为生成阶段的一等条件（提示、few-shot 与 AST 校验都按目标引擎选），而不是事后字符串替换；跨源查询交给联邦查询引擎或先物化中间表，不要指望模型处理跨引擎的 join。
- **追问**：审计与合规要做到什么程度？
  - 要点：可回放（问题 → 候选 → 子模式 → SQL → 计划成本 → 结果摘要 → 使用者）比「存一条 SQL」重要得多，出事后要能回答「这个数字当时是怎么来的」；审计数据本身常含 PII，入日志前脱敏与最小化，见 [[agents-10]]。

## 公司变体

- **Databricks**：偏工程与平台集成。公司主业是 Lakehouse，问这题会落到 Unity Catalog 的元数据与行/列级权限、Delta 表的统计信息与分区裁剪、以及「怎么把自然语言接口做进 SQL 编辑器/Notebook 的既有工作流」。准备重点是权限模型与元数据来源的具体接口，以及成本护栏（`EXPLAIN` / 扫描量预估）怎么接进平台；语义层与指标视图的沉淀也是加分项。
- **Palantir**：偏前置部署与治理。公司做的是把模型嵌进客户既有系统，所以追问常围绕本体（ontology）与语义层的建模、多租户下的权限与审计、客户数仓注释质量差时的冷启动路径，以及「错误数字进入客户决策流程」这类风险的管控。答案里显式区分「已验证结果」与「模型生成结果」、并给出人工复核通道，比堆技术栈更贴合。

两家共同点是**都会追问权限与审计**，且都不满足于「数据库有权限」这一句。

## 相关题目

- [[rag-09]]：把向量索引换成 HNSW 还是 IVF-PQ——schema 与样例值索引的选型直接复用这道题的结论。
- [[rag-02]]：稀疏检索与稠密检索的选择。表名/列名的字面匹配（BM25）与语义匹配（向量）在 schema linking 里是互补的，融合方式沿用这里的做法。
- [[rag-08]]：权限感知的 retrieval。生成期 schema 过滤 + 执行期数据库权限兜底的双层设计与之同构。
- [[agents-02]]：agent 循环中的错误回灌与重试。SQL 报错回灌重写、以及「重写 SQL 而不是重跑检索」的选择属于同一类问题。
- [[agents-11]]：human-in-the-loop 审批。歧义澄清与高风险查询的人工确认都落在这道题的设计空间里。
- [[agents-06]]：多 agent 编排的失效模式。CHESS 式的四 agent 分工（检索/精筛/生成/单测）要不要拆、拆了会不会互相甩锅，参照这道题。
- [[evaluation-10]]：任务级评测。执行结果正确性为主指标、分层报通过率的结构从这里展开。
- [[evaluation-04]]：回归门禁。新模型或新 prompt 上线前的守门规则。
- [[system-design-06]]：同为「先做信息抽取再结构化」的大规模 pipeline，可对比离线画像与在线生成的职责划分。
- [[inference-serving-12]]：单位成本治理。成本护栏与 prefix caching 的排序沿用该题的框架。
- [[system-design-09]]：LLM gateway 的路由思路，对应本题「口径命中走语义层、未命中才自由生成」的路由判断。

## 参考资料与归属

- Yu et al. (EMNLP 2018)，*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*：https://arxiv.org/abs/1809.08887 。第 8 节的 10,181 问 / 5,693 条 SQL / 200 库 / 138 领域与 12.4% 精确匹配取自该论文摘要。
- Li et al. (NeurIPS 2023)，*Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs（BIRD）*：https://arxiv.org/abs/2305.03111 。第 8 节的 12,751 对 / 95 库 / 33.4 GB / 37 领域、ChatGPT 40.08% 与人类 92.96% 取自摘要；错误分析里 wrong schema linking 41.6%、misunderstanding database content 40.8%、misunderstanding knowledge evidence 17.6%（500 例抽样）取自正文 6.6 节，第 3 节与常见追问引用该口径。
- Lei et al. (ICLR 2025)，*Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows*：https://arxiv.org/abs/2411.07763 。第 8 节的 632 个问题、1000+ 列、BigQuery/Snowflake、单条超 100 行、o1-preview 框架 21.3% / Spider 1.0 91.2% / BIRD 73.0% 取自摘要。
- Talaei et al.，*CHESS: Contextual Harnessing for Efficient SQL Synthesis*：https://arxiv.org/abs/2405.16755 。第 4、5 节引用的四 agent 分工、schema selector 降低约 5 倍 token 与 +2 个百分点、BIRD 测试集 71.10% 与 LLM 调用少约 83%、以及 4,337 列合成 schema 的 Pass@1 59%→61% / Pass@5 61%→63% 取自该论文正文 4.2.2 与 4.2.1/摘要；「生产级 schema 常含数千列、远超 BIRD」的论述亦出自该文。
- Pourreza & Rafiei (NeurIPS 2023)，*DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction*：https://arxiv.org/abs/2304.11015 。第 5 节的「提升约 10%」、Spider holdout 79.9 → 85.3、BIRD 55.9% 取自摘要；自纠错模块的 generic / gentle 两种 prompt 取自正文 4.4 节。
- 延伸来源说明：作业单给出的五条来源均为论文，容量与成本算例（KV cache、预填充时间、并发槽位、每天 GPU 小时与美元）是按 H100 SXM5 官方规格（bf16 稠密 989 TFLOPs、HBM3 3.35 TB/s）与 LLaMA-3-70B 配置（80 层、8 个 KV 头、head_dim 128、bf16，即 320 KiB/token）自行复算的：prefill 口径 $2NS/(\text{峰值} \times \text{卡数} \times \text{MFU})$、decode 单步按「每卡读权重分片 + 本批 KV 除以单卡带宽」（与 [[inference-serving-15]] 一致），与本仓库其它专题的常数相同；标为「示例价」的 token 单价只用于口径对比，不代表任何厂商当前定价。数仓规模（2,000 张表、6 万列、2 万提问/天）与每列 token 数是算账假设，非来自上述论文。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
