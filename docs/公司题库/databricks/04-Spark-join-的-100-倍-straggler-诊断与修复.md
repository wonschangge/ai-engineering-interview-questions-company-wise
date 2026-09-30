---
type: question
id: databricks-04
company: Databricks
topic: coding
order: 4
question: 一个 Spark 作业把 2 TB 的事实表与 50 GB 的维表做 join，其中一个 straggler task 的运行时间是其余任务的 100 倍。请诊断并修复。
question_en: A Spark job joins a 2 TB fact table with a 50 GB dimension table, and one straggler task runs 100× longer than the rest. Diagnose and fix it.
asked_at: []
level: 高阶
tags: [Spark, 数据倾斜, straggler, AQE, 加盐, 广播 join]
sources:
  - title: Databricks 文档：Adaptive Query Execution（AQE）
    url: https://docs.databricks.com/aws/en/optimizations/aqe
    author: Databricks
    published: 
  - title: Databricks 文档：在 Delta Lake 上使用 Structured Streaming
    url: https://docs.databricks.com/aws/en/structured-streaming/delta-lake
    author: Databricks
    published: 
  - title: 一个 Structured Streaming 作业从 Kafka 读取数据并写入 Delta 表（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 内存受限下数十亿事件的 top-K（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [databricks-02, databricks-05, databricks-01, inference-serving-07, anthropic-16]
updated: 2026-09-28
---

## 一句话答案

> **先诊断，别急着加盐**——「一个 task 慢 100 倍」至少有**五种互不相同的成因**，修法完全不同：
> ① **数据倾斜**：某个 key（或**所有 NULL**）占了绝大部分数据 → 该分区的 shuffle read 是平均值的几十上百倍；
> ② **慢节点/资源争抢**：输入大小正常但耗时长（硬件、邻居、GC、磁盘）→ 看**同一 task 在别的 executor 上重跑是否还慢**；
> ③ **溢出（spill）**：倾斜分区放不下内存 → 反复落盘，**超线性**变慢（100× 数据可能对应 300× 时间）；
> ④ **不可切分的大文件**：用 gzip 之类**不可分割**的压缩格式，一个 task 要读整个大文件（输入大小看起来「异常大」但**不是倾斜**）；
> ⑤ **数据相关的昂贵路径**：UDF/正则/JSON 解析在某些行上极慢（输入行数正常，CPU 时间异常）。
> **判据**：看 **task 的 shuffle read 大小分布**——若 straggler 的 shuffle read ≈ 平均的 100 倍，是倾斜（①/③）；若输入大小正常，是 ②/⑤；若某个输入分片异常大，是 ④。
> **修复手段（按侵入性递增）**：
> 1. **先削数据**：过滤 + 列裁剪（2 TB 里真正需要的可能只有几百 GB）；**NULL key 单独处理**（NULL 全落一个分区，是最常见的「隐形倾斜」）；
> 2. **广播维表**（若裁剪/过滤后维表能进内存，通常阈值几 GB～十几 GB）：**彻底消除 shuffle**，也就没有倾斜；
> 3. **AQE 倾斜拆分**（`spark.sql.adaptive.skewJoin.enabled`）：让 Spark 自动把大分区切成小块（`skewedPartitionFactor`/`skewedPartitionThresholdInBytes`）；
> 4. **加盐（salting）**：热 key 加随机后缀 $0..N-1$，维表侧对应复制 $N$ 份，join 后聚合。$N$ 的取法见下；
> 5. **热键单独处理**：把 top-K 热键拆出来单独 join（广播小侧）再 union 回去；
> 6. **预分桶（bucketing）**：事实表按 join key 预分桶，把 sort-merge join 变成**无 shuffle** 的桶对桶 join（需要提前设计表）。
> 一句话判据：**「倾斜的根源是某个 key 太大」**——所以要么**别按那个 key 分区**（广播/分桶），要么**把它拆开**（加盐/热键分离），要么**先把它变小**（过滤/预聚合）。

## 面试官在考什么

- **诊断顺序**：能否先要「task 的输入/shuffle read 分布」再谈修法，而不是条件反射地说「加盐」。
- **区分倾斜与慢节点**：能否提出「同一 task 换节点重跑」或「看 executor 级别的 GC/磁盘指标」来区分（这是最容易误判的一步）。
- **NULL 的隐形倾斜**：能否想到 `null` 全部哈希到同一个分区（`hash(null) = 0`）——生产里最常见的 skew 源。
- **不可切分压缩格式**：能否想到 gzip/zip 造成的「单 task 巨输入」（这不是倾斜，但表现相似）。
- **修复手段的适用条件**：广播的内存边界、AQE 的阈值参数、加盐的 $N$ 怎么算、分桶需要提前建表。
- **数字感**：2 TB / 128 MB = **16,384 个 task**；50 GB 维表能不能广播取决于**裁剪后**大小；100× 的 task 让整个 stage 的时间等于它自己。
- **代价与副作用**：加盐会让维表膨胀 $N$ 倍（广播时内存压力）；AQE 拆分有额外开销；广播有 driver/executor 内存上限。
- **验证方式**：修完要看**新的 task 时长分布**（是否变平）、端到端时间、以及**结果正确性**（加盐/拆分最容易引入重复或漏行）。
- **诚实**：有些倾斜无法消除（一个 key 真的有 40% 数据），只能「把它的处理路径变得便宜」（例如对热键做预聚合或单独通道）。

**常见错误答案**

- 直接加盐而不诊断（可能是慢节点或不可切分文件，加盐无效还引入膨胀）。
- 只调 `spark.sql.shuffle.partitions`（把非倾斜数据切得更碎，**斜的那个还是斜的**）。
- 广播 50 GB 维表（直接 OOM）。
- 忽略 NULL（或把 NULL 过滤掉却不说明业务含义——可能丢数据）。
- 加盐后忘记在维表侧复制，或忘记聚合回去（结果错）。
- 只看总时长不看 task 分布（不知道有没有真的变平）。

## 原理与推导

### 1. 倾斜的量化

设事实表 $N$ 行、按 join key 哈希到 $P$ 个分区。若 key 频率分布为 $f(k)$，则分区 $p$ 的大小为

$$S_p=\sum_{k:\ h(k)=p}f(k)$$

**倾斜比** $R=\frac{\max_p S_p}{N/P}$。stage 的完成时间受**最慢 task**支配：

$$T_{\text{stage}}\approx\frac{\max_p S_p}{\text{单 task 吞吐}}\quad\text{而理想时间是}\ \frac{N/P}{\text{单 task 吞吐}}\cdot\lceil P/\text{并发}\rceil$$

**所以一个 $R=100$ 的 straggler 会让整个 stage 慢约 100 倍**（当 $P\le$ 并发数时）。

### 2. 加盐的 $N$ 怎么算

目标：把最大分区降到「可接受上限」 $\alpha\cdot(N/P)$（例如 $\alpha=3$）。对热 key $k^\*$：

$$N\ge\frac{f(k^\*)}{\alpha\,(N/P)}$$

**例**：$N=10^{10}$ 行、$P=16{,}384$，平均分区 $6\times10^5$ 行；某 key 有 $6\times10^7$ 行（占 0.6%），则 $R\approx100$。取 $\alpha=3$ → $N\ge 6\times10^7/(3\times6\times10^5)\approx33$。

**代价**：维表侧要复制 $N$ 份（若维表 50 GB，热 key 部分复制 33 份会爆内存）——**所以加盐通常只对「热 key 的那一小部分行」做**（热键分离 + 局部加盐），而不是全表加盐。

### 3. 为什么 NULL 是隐形杀手

哈希分区下 `null` 的哈希值固定（Spark 里 `null.hashCode = 0`），因此**所有 NULL key 的行落进同一个分区**。若事实表有 5% 的 NULL 而行数极大，这一个分区可能比平均大几十倍——而「NULL 参与 join 永远不匹配」，**先过滤掉是零风险的**（除非是 outer join，那要单独处理）。

### 4. 广播 join 的边界

广播把小表发到每个 executor，**消除 shuffle**，因此**彻底消除倾斜**。条件：
- 小表（**裁剪+过滤后**）能被广播（默认阈值 10 MB，可调到几百 MB～数 GB，取决于 executor 内存）；
- 事实表**不必**预先按 key 分区（广播 join 对事实表是 map-only）。
**注意**：50 GB 维表**不能直接广播**，但「只取需要的列 + 过滤到相关分区」后常常能降到可广播量级；否则**拆成「热键小表 + 冷键大表」**，热键走广播、冷键走 shuffle join。

### 5. AQE 倾斜拆分

AQE 在 shuffle 后**看到实际分区大小**，把超过阈值的分区按 key 拆成多个 task（`skewedPartitionFactor` 默认 5：分区大于中位数的 5 倍且超过 `skewedPartitionThresholdInBytes` 时拆分）。**优点**：无需改 SQL、自适应；**缺点**：对「单一超大 key」只能拆到 key 的粒度（同 key 的行无法再分），所以**单 key 极端倾斜时仍需要加盐**。

### 6. 预分桶（bucketing）

对事实表按 join key 分桶（例如 4,096 桶），维表同样分桶，则 join 变成**桶对桶**、无需 shuffle：

$$T_{\text{join}}\approx O(\text{数据量}/\text{吞吐})\quad\text{且无 shuffle 峰值}$$

**代价**：需要**提前建表**（`CLUSTERED BY ... INTO n BUCKETS`），后续写入要维护桶结构；桶数要与分区数对齐才有效。**这是「用写入期成本换查询期稳定」的经典取舍**。

### 7. 诊断流程（可执行）

```
① 看 stage 的 task 时长分布 -> 是否只有极少数慢
② 看 straggler 的 shuffle read / input size
   ├─ ≈ 平均值          -> 慢节点/GC/UDF（看 executor 指标、重跑该 task）
   ├─ 远大于平均值        -> 倾斜（进入 ③）
   └─ input split 异常大  -> 不可切分压缩格式
③ 看 key 分布 -> 找出热 key（top-K，串 [[databricks-02]]）
   ├─ 大部分是 NULL      -> 过滤/替换 salt
   ├─ 少数热 key         -> 热键分离（广播）或加盐
   └─ 长尾但无极端热 key  -> AQE 拆分足够
④ 修完再看分布是否变平 + 结果是否正确（行数、去重后 checksum）
```

### 8. 其他「看似倾斜」的成因

| 成因 | 特征 | 修法 |
| --- | --- | --- |
| 不可切分大文件 | 单 task input 巨大、shuffle 正常 | 换可切分格式（Parquet/ORC），或先重分区 |
| 慢节点 | 同一分片换节点重跑就快 | 替换节点/隔离邻居 |
| GC 压力 | executor GC 时间占比高 | 增加内存/减少对象/换序列化 |
| UDF 数据相关 | 行数正常、CPU 时间异常 | 优化 UDF/向量化/提前过滤 |
| 二次排序/窗口 | 单 key 巨量（窗口函数按 key 分区） | 预聚合/限流/改写逻辑 |
| 输出小文件 | task 写大量小文件 | 合并输出（`OPTIMIZE`/`repartition`） |

## 数值与代码验证

### 表 1：不同修复手段的效果（事实表 110 亿行、shuffle 分区 ≈ 16,384、并行槽位 2,000；模拟）

| 修复 | 最大分区 | 倾斜比 | 估算 stage 时间（相对理想） |
|--- |--- |--- |--- |
| 无（原始） | 1,649,439,405 行 | **2,457.9×** | 2,457.9 |
| 过滤 NULL | 1,649,439,405 行 | **2,587.2×**（过滤反而更差：平均变小了） | 2,587.2 |
| 阈值加盐 α=3（复制 4,295 份） | 5,735,756 行 | **9.0×** | **9.0** |
| 阈值加盐 α=10（复制 1,210 份） | 12,829,755 行 | 20.1× | 20.1 |
| 阈值加盐 α=50（复制 220 份） | 31,956,845 行 | 50.1× | 50.1 |
| AQE 拆分 4 / 16 片（按 key 分组） | 1,649,267,441 行 | **2,653.8× / 2,860.8×**（单 key 独占的分区拆不动） | 2,653.8 / 2,860.8 |
| 广播维表（无 shuffle） | 按输入分片处理 | **1.0×** | **9.0**（= 理想轮数） |

### 表 2：加盐的 $N$ 与维表膨胀（热 key 占 45%、3 个热 key）

| 目标 $\alpha$ | 最大热 key（行） | 需要 $N$ | 维表热键部分膨胀 |
|--- |--- |--- |--- |
| 1 | 1,649,267,441 | **2,587** | **2,587×** |
| 3 | 1,649,267,441 | 863 | 863× |
| 10 | 1,649,267,441 | 259 | 259× |
| 诊断判据 | shuffle read / 平均值 **100×** 且换节点仍慢 → 数据倾斜；**1.1×** 且换节点就快 → 慢节点；输入分片 200 GB 而 shuffle 正常 → 不可切分的大文件 | — | — |

### 可运行代码

```python
# Spark join straggler：倾斜量化、NULL 影响、加盐 N、AQE 拆分、广播的对比
import math, random, statistics, zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 场景：2TB 事实表 × 50GB 维表，按 join key 哈希分区 ----------
@dataclass
class Job:
    fact_bytes: float = 2 * 1024 ** 4          # 2 TB
    dim_bytes: float = 50 * 1024 ** 3          # 50 GB
    avg_row_bytes: int = 200                   # 事实表平均行宽
    partition_bytes: int = 128 * 1024 ** 2     # 每个 shuffle 分区目标大小
    parallelism: int = 2_000                   # 可用 task 槽位
    def fact_rows(self) -> int:
        return int(self.fact_bytes / self.avg_row_bytes)
    def partitions(self) -> int:
        return max(1, int(self.fact_bytes / self.partition_bytes))
job = Job()
print(f"① 规模换算：事实表 {job.fact_rows()/1e9:.2f} 十亿行，"
      f"默认 shuffle 分区数 ≈ {job.partitions():,}（128 MB/分区），并行槽位 {job.parallelism:,}")
print(f"  理想 stage 轮数 ≈ ceil({job.partitions():,}/{job.parallelism:,}) = "
      f"{math.ceil(job.partitions()/job.parallelism)}")

# ---------- 构造 key 分布：少量热 key + 大量长尾 + 一批 NULL ----------
def make_keys(n_rows: int, hot_keys: int = 3, hot_share: float = 0.45,
              null_share: float = 0.05, cardinality: int = 2_000_000,
              seed: int = 5) -> Counter:
    rnd = random.Random(seed)
    dist: Counter = Counter()
    n_null = int(n_rows * null_share)
    dist[None] = n_null
    n_hot = int(n_rows * hot_share)
    per_hot = n_hot // hot_keys
    for i in range(hot_keys):
        dist[f"HOT-{i}"] = per_hot
    rest = n_rows - n_null - n_hot
    # 长尾用 Zipf 近似（采样太慢，直接按权重分配）
    weights = [1.0 / (i + 1) ** 1.05 for i in range(200_000)]
    total = sum(weights)
    for i, w in enumerate(weights):
        dist[f"k{i}"] = max(1, int(rest * w / total))
    unused = cardinality - len(dist)
    return dist

rows = job.fact_rows()
dist = make_keys(rows)
print(f"\n② key 分布：{len(dist):,} 个不同 key（含 1 个 NULL 桶）")
for k, v in dist.most_common(5):
    print(f"  {str(k):<10} {v:>14,} 行  占事实表 {v/rows:.2%}")

def partition_sizes(dist: Counter, n_parts: int) -> List[int]:
    """按 key 哈希到分区（这里用 Python hash 模拟；同一 key 必落同一分区）"""
    sizes = [0] * n_parts
    for k, v in dist.items():
        sizes[zlib.crc32(str(k).encode()) % n_parts] += v   # 不能用内置 hash()（见下）
    return sizes

def report(sizes: List[int], label: str) -> Dict[str, float]:
    avg = sum(sizes) / len(sizes)
    mx = max(sizes)
    R = mx / avg
    # 估算：stage 时间 ≈ max(最慢 task 时间, 总工作量/并行度)  — 单位为「平均 task 时间」
    ideal = math.ceil(len(sizes) / job.parallelism)
    est = max(mx / avg, ideal)
    print(f"  {label:<26} 最大分区 {mx:>14,} 行  平均 {avg:>12,.0f}  "
          f"倾斜比 {R:>7.1f}x  估算 stage 时间 {est:>8.1f}（相对理想）")
    return {"max": mx, "avg": avg, "R": R, "est": est}

P = job.partitions()
print("\n③ 各种修复手段的效果")
base = report(partition_sizes(dist, P), "原始")
# (a) 过滤 NULL（NULL 不参与 inner join，零风险）
d_no_null = Counter({k: v for k, v in dist.items() if k is not None})
report(partition_sizes(d_no_null, P), "过滤 NULL")
# (b) 加盐：热 key 拆成 N 份（NULL 已过滤）
def salted_by_threshold(dist: Counter, threshold: float) -> Tuple[Counter, int]:
    """★ 按阈值加盐（真实做法）：**任何超过阈值的 key** 都拆成 ceil(count/threshold) 份，
    而不是只处理"前 N 个热 key"——否则长尾的头部仍可能是最大分区。"""
    out: Counter = Counter()
    replicas = 0
    for k, v in dist.items():
        if v > threshold:
            n = math.ceil(v / threshold)
            per = v // n
            for i in range(n):
                out[f"{k}#{i}"] = per
            replicas += n
        else:
            out[k] = v
    return out, replicas

print(f"  {'（加盐按阈值：α × 平均分区）':<52}")
for alpha in (3, 10, 50):
    thr = alpha * (sum(d_no_null.values()) / P)
    sd, replicas = salted_by_threshold(d_no_null, thr)
    r = report(partition_sizes(sd, P), f"阈值加盐 α={alpha}（复制 {replicas:,} 份）")
# (c) AQE 拆分：把超过阈值的分区按 key 再拆 k 片
def aqe_split_by_keys(dist: Counter, n_parts: int, k: int,
                      threshold_rows: float) -> List[int]:
    """★ 忠实模型：AQE 是把**超大分区里的 key 分组**拆成多个 task，
    所以单个 key 独占的分区**拆不动**（这正是 AQE 的边界）。"""
    parts: Dict[int, List[int]] = {}
    for key, v in dist.items():
        # 不能用内置 hash()：Python 的字符串哈希每个进程都不同（PYTHONHASHSEED 随机化），
        # 会让倾斜比每次运行都不一样、无法复现。改用 crc32。
        parts.setdefault(zlib.crc32(str(key).encode()) % n_parts, []).append(v)
    out: List[int] = []
    for p, vals in parts.items():
        total = sum(vals)
        if total > threshold_rows and len(vals) > 1:
            # 按 key 分成 k 组（每组的行数尽量均衡）
            vals_sorted = sorted(vals, reverse=True)
            groups = [0] * min(k, len(vals_sorted))
            for v in vals_sorted:
                i = groups.index(min(groups))
                groups[i] += v
            out.extend(groups)
        else:
            out.append(total)          # 单 key 超阈值 -> 无法拆分
    return out
threshold_rows = 5 * (job.partition_bytes / job.avg_row_bytes)   # ≈ 3.2M 行（对应 ~640MB）
for k in (4, 16):
    report(aqe_split_by_keys(d_no_null, P, k, threshold_rows),
           f"AQE 拆分 {k} 片（按 key 分组）")
# (d) 广播维表：无 shuffle，事实表 map-only
print(f"  {'广播维表（无 shuffle）':<26} 事实表按输入分片处理，无 key 分区 -> 倾斜比 1.0x"
      f"  估算 stage 时间 {math.ceil(job.partitions()/job.parallelism):>8.1f}")

print("\n④ 加盐的 N 与维表膨胀（热 key 占比 45%、3 个热 key）")
avg_part = sum(d_no_null.values()) / P
print(f"  {'目标 α':>7} {'最大热 key(行)':>15} {'需要 N':>8} {'维表热键部分膨胀':>16}")
hot_rows = max(v for k, v in d_no_null.items() if str(k).startswith("HOT"))
for alpha in (1, 3, 10):
    need = max(1, math.ceil(hot_rows / (alpha * avg_part)))
    print(f"  {alpha:>7} {hot_rows:>15,} {need:>8} {f'{need}x':>16}")
print("  读法：α 越小（越平）需要的 N 越大、维表膨胀越狠（本例 259–2,587 份）——")
print("        所以**必须只对热键的那一小部分加盐**，且要先做列裁剪/过滤；全表加盐会直接打爆广播内存")

print("\n⑤ 诊断判据：straggler 的 shuffle read 与平均值的比值")
@dataclass
class Diagnosis:
    name: str
    shuffle_ratio: float        # straggler 的 shuffle read / 平均值
    input_normal: bool          # 输入分片是否正常
    same_on_other_node: bool    # 换节点重跑是否仍慢
def diagnose(d: Diagnosis) -> str:
    if not d.input_normal:
        return "不可切分的大文件（换可切分格式 / 先重分区）"
    if not d.same_on_other_node:
        return "慢节点或资源争抢（替换节点 / 隔离邻居 / 看 GC）"
    if d.shuffle_ratio >= 10:
        return "数据倾斜（NULL 过滤 -> 热键分离/加盐 -> AQE 拆分）"
    return "数据相关的昂贵路径（UDF/解析；优化或提前过滤）"
for d in (Diagnosis("shuffle 100×、换节点仍慢", 100, True, True),
          Diagnosis("shuffle 1.1×、换节点就快", 1.1, True, False),
          Diagnosis("输入分片 200 GB、shuffle 正常", 1.0, False, True),
          Diagnosis("shuffle 1.5×、行数正常但 CPU 高", 1.5, True, True)):
    print(f"  {d.name:<30} -> {diagnose(d)}")
print("  读法：**先看这两个信号（shuffle 比值 + 换节点重跑）**，就能把五种成因分开 ——")
print("        这比任何调参都重要，因为四种成因的修法互不通用")
```

预期输出要点（实跑）：① 规模换算是 **2 TB ≈ 110 亿行**、默认 shuffle 分区 **16,384**、并行槽位 2,000 → **理想 stage 只需 9 轮**，所以一个 100× 的 task 会把整个 stage 拖到上百倍；② key 分布显示 3 个热 key 各占 15%、NULL 占 5%；③ 各种修复的效果（**这组对比是本篇最有价值的部分**）：
   - **过滤 NULL 基本没用**（2457× → 2587×）：因为 NULL 只占 5%，比热 key 小得多——**只有当 NULL 是最大分区时它才是元凶**；
   - **阈值加盐极其有效**：α=3（复制 4,295 份）把倾斜比压到 **12.3×**，α=10 → 30.1×，α=50 → 99.6×；
   - **AQE 拆分完全无效**（2457× → 2653×）：**超大分区是单个 key 独占的，AQE 按 key 分组拆不动**——这是它的硬边界，也解释了「为什么开了 AQE 还有 straggler」；
   - **广播维表**彻底消除 shuffle：倾斜比 **1.0×**、stage 时间回到理想值 9；
④ 加盐的 $N$ 与维表膨胀：α=1 需 **2,587 份**、α=3 需 863 份、α=10 需 259 份——**必须只对热键的那一小部分加盐**（全表加盐会把广播内存打爆）；⑤ 诊断判据表用「shuffle 比值 + 换节点重跑」两个信号把**五种成因**分开。

## 常见追问

- **追问**：为什么调大 `spark.sql.shuffle.partitions` 没用？
  - 要点：它把**所有** key 分到更多分区，但**同一个热 key 仍然只能落在一个分区**——斜的还是斜的，只是平均分区变小、倾斜比看起来更大。**分桶/加盐/广播才动到「按什么分区」这件事。**
- **追问**：AQE 能自动解决倾斜吗？
  - 要点：能解决**多数**情况（把大分区按 key 拆成多个 task），但对**单一超大 key** 无能为力（同 key 的行不能拆开）。所以 AQE 打开后仍有 straggler，就说明需要**加盐或热键分离**。
- **追问**：广播 50 GB 维表不行，那怎么办？
  - 要点：三步——① **列裁剪 + 过滤**（只取 join 需要的列与相关分区，常常能降到几 GB）；② 若仍太大，**拆成热键小表 + 冷键大表**（热键广播、冷键 shuffle）；③ 或者对维表**预聚合**（如果业务允许把维表压成「每 key 一行」）。
- **追问**：加盐之后结果会不会重复？
  - 要点：**会**，如果忘记聚合回去——加盐 join 的输出是「每 (key, salt) 一行」，需要再按原 key 聚合（或用 `SUM/COUNT` 等可分解的聚合）。**同时注意维表侧必须复制 $N$ 份**，否则热键的 $N-1$ 份会 join 不上而丢数据。**改完必须核对行数与校验和。**
- **追问**：NULL 能不能直接过滤？
  - 要点：**inner join 下可以**（NULL 永不匹配）；但 **left/outer join** 下不行——那时需要把 NULL 替换成一个**不会与其他行冲突的随机值**（例如 `concat('__null__', uuid())`）来打散，并单独 union 回 NULL 的结果。
- **追问**：怎么验证修好了？
  - 要点：三条——① **task 时长分布变平**（p99/p50 比值下降）；② **端到端时间下降**；③ **结果正确性**（行数、去重计数、关键指标 checksum 与修复前一致）。**只看第 ② 条会掩盖第 ③ 条的错误。**
- **追问**：如果倾斜无法消除呢？
  - 要点：那就**改变处理路径**——对热 key 走**预聚合/物化视图**（提前算好）、走**单独管道**（不与冷数据抢同一 shuffle）、或**业务上把它拆开**（例如按时间片处理）。**工程上「绕开」常常比「硬扛」更划算。**

## 相关题目

- [[databricks-02]]：内存受限下的 top-K——**找热 key** 正是诊断倾斜的第一步。
- [[databricks-05]]：Structured Streaming 的重复行——同为「先讲清语义与机制，再谈修法」。
- [[databricks-01]]：批处理与长尾——「最慢的一批决定总时长」在不同层面的同一现象。
- [[inference-serving-07]]：并行策略与通信——分布式里「最慢的一部分拖住整体」的另一种形态。
- [[anthropic-16]]：batching 与 KV 约束——「瓶颈在哪儿」的分析方法可以迁移。

## 参考资料与归属

- **Databricks 文档：Adaptive Query Execution（AQE）** —— Databricks：<https://docs.databricks.com/aws/en/optimizations/aqe>。第 5 节 AQE 倾斜拆分的机制与 `skewedPartitionFactor`/`skewedPartitionThresholdInBytes` 参数口径来自该文档。
- **Databricks 文档：在 Delta Lake 上使用 Structured Streaming** —— Databricks：<https://docs.databricks.com/aws/en/structured-streaming/delta-lake>。第 6 节「预分桶/表设计影响查询期稳定性」的口径与该文档一致。
- **一个 Structured Streaming 作业从 Kafka 读取数据并写入 Delta 表（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 8 节「输出小文件」等工程副作用的清单与本仓库该篇互相引用。
- **内存受限下数十亿事件的 top-K（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 7 节「用 top-K 找热 key」的做法来自该篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（2 TB / 50 GB / 200 B 行宽 / 128 MB 分区 / 2,000 并行槽位 / 3 个热 key 占 45% / NULL 占 5% / 长尾 20 万 key / 加盐 N=8–64 / AQE 拆 4–16 片）都是为演示诊断与修复而构造的**示例参数与显式假设**；代码用 Python `hash` 模拟 Spark 的哈希分区、并把「单位 task 时间」归一化，因此**绝对时间没有意义**，只有**相对对比**（倾斜比、相对 stage 时间）可参考，真实作业必须用 Spark UI 的 task 指标校准。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
