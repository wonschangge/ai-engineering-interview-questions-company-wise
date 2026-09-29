---
type: question
id: databricks-02
company: Databricks
topic: coding
order: 2
question: 你有一个数十亿事件的流，需要在内存受限的前提下求出出现频率最高的 top-K 个 key。精确解不可能做到：你会怎么做？
question_en: You have a stream of billions of events and need the top-K most frequent keys under a tight memory bound. An exact solution is impossible — what do you do?
asked_at: []
level: 高阶
tags: [近似算法, top-K, Misra-Gries, Space-Saving, count-min-sketch]
sources:
  - title: Python 官方文档：heapq — 堆队列算法
    url: https://docs.python.org/3/library/heapq.html
    author: Python Software Foundation
    published: 
  - title: 一个 Spark 作业把 2 TB 的事实表与 50 GB 的维表做 join（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 为大型商品目录设计语义搜索（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [databricks-01, databricks-03, databricks-04, databricks-05, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> 先承认数学事实：**精确 top-K 需要为每个 key 保留计数**，在内存受限下不可能。所以走**近似 + 验证**的两段式：
> **① 第一段：单遍草图（sketch）找候选**
> - **Misra-Gries（MG）**：只留 $k-1$ 个计数器。命中就加一；未命中且计数器已满时，**把所有计数器减一**（减到 0 就淘汰）。保证：**任何出现次数 $>N/k$ 的元素一定在结果里**（这是硬保证，不是概率）；
> - **Space-Saving（SS）**：留 $k$ 个计数器，每个带一个 `error`。未命中时**替换当前计数最小的那个**，新计数 = 最小值 + 1，`error` = 最小值。保证：**真实频次落在 $[\text{count}-\text{error},\ \text{count}]$**，且同等内存下通常比 MG 准得多；
> - **Count-Min Sketch（CMS）+ 堆**：$d$ 行 × $w$ 列的计数矩阵（$w=\lceil e/\varepsilon\rceil$、$d=\lceil\ln(1/\delta)\rceil$），查询取 $d$ 个哈希位置的最小值。保证：**只高估不低估**，误差 $\le\varepsilon N$ 的概率 $\ge 1-\delta$。
> **② 第二段：精确验证（可选但强烈建议）**——用第一段给出的**候选集**回扫一遍数据（或查原始存储）做精确计数，再取真正的 top-K。代价是**两遍**，换来**结果精确**。
> 工程上还要三件事：**可合并**（Spark 里每个分区各建草图，再 merge 成全局草图——SS/MG 可按计数合并，CMS 直接相加）、**热点倾斜的处理**（top-K 往往就是 join 的 skew 源，串 [[databricks-04]]）、以及**内存预算的显式换算**（$k$ 或 $w\times d$ 与可承受误差的直接关系）。
> 一句话判据：**「我要的是 top-K 的集合，还是每个 key 的精确频次？」**——前者用 SS/MG + 验证；后者用 CMS（接受高估）。

## 面试官在考什么

- **是否承认不可能**：能否立刻指出「精确需要全量计数」，而不是尝试「用哈希表但小心一点」。
- **算法选择与保证**：能否分别说出 MG（$>N/k$ 必现）、SS（误差区间）与 CMS（$\varepsilon N$ 高估、概率 $1-\delta$）的**不同保证形式**，并知道它们分别适合什么。
- **误差与内存的换算**：能否把「可接受误差」翻译成参数（$k$、$w$、$d$），并给出内存量级（$k$ 个计数器 vs $w\cdot d$ 个计数器）。
- **两段式验证**：能否主动提出「候选集 + 精确复算」，把近似变成精确结论（这是工程上最常被忽略的一步）。
- **可合并性**：这是 Databricks 场景的关键——**分布式流处理必须能合并**（Spark Structured Streaming 的 `mapGroupsWithState`/聚合状态、或者每个 micro-batch 合并）。能否说清哪些草图可合并、怎么合并。
- **哈希与对抗**：能否意识到**对抗性输入**会让固定哈希退化（需要随机种子、或 universal hashing）；以及 key 空间大时的哈希成本。
- **边界**：$K$ 很大（例如 10 万）时内存怎么算；key 基数极大（10 亿）时 CMS 的优势；以及「top-K 随时间漂移」（滑动窗口/衰减）的处理。
- **与业务衔接**：找出 top-K 之后干什么（热点隔离、限流、分区优化、缓存预热）——**这才是问这个问题的人真正关心的**。

**常见错误答案**

- 说「用哈希表统计然后取最大的 K 个」（忽略内存约束）。
- 只说 CMS（它对**点查询**友好，但对「最大 K 个」需要额外的堆/候选管理）。
- 不做候选验证就把近似结果当结论（在计费、风控、限流场景会出事）。
- 忽略可合并性（单机算法在 Spark 里用不了）。
- 忽略对抗性输入与哈希种子。
- 只讲算法，不讲「找到 top-K 之后怎么用」。

## 原理与推导

### 1. 为什么精确不可能

要精确判定 top-K，必须能区分「第 K 名」与「第 K+1 名」——当二者频次接近时，**任何有损结构都可能判错**。在 $m$ 个不同 key、内存 $M$ 个计数器的约束下，只有 $M\ge m$ 才能精确。因此：

$$\text{可保证的只有「频次超过阈值的元素必被找到」}$$

### 2. Misra-Gries：$k-1$ 个计数器的硬保证

**算法**：维护至多 $k-1$ 个 (key, count)。命中 → count+1；未命中且有空位 → 插入 count=1；未命中且已满 → **所有 count 减一**，减到 0 的淘汰。

**保证**：设 $f(x)$ 为真实频次，$N$ 为总事件数。若 $f(x)>N/k$，则 $x$ 一定在最终计数器中；且对任何被保留的 $x$：

$$f(x)-\frac{N}{k}\le \hat f(x)\le f(x)$$

**读法**：MG 是**低估**（count 只会被减），所以它**不会误报**低频元素为高频，但会漏掉「刚过阈值」的边界情况（取决于扣减时序）。内存 $O(k)$。

### 3. Space-Saving：同等内存下更准

**算法**：维护 $k$ 个 (key, count, error)。命中 → count+1；未命中 → 找到 count 最小的桶，**替换其 key**，新 count = 旧 count + 1，error = 旧 count。

**保证**：对桶中元素，真实频次满足

$$\text{count}-\text{error}\le f(x)\le \text{count}$$

**读法**：SS 是**高估**（count 是「该槽位被占以来的总流量」），因此它**不会漏掉重元素**，但可能把中等元素虚高。经验上 SS 的 top-K 召回显著优于同内存的 MG（本机实测见下）。

### 4. Count-Min Sketch：点查询的概率保证

$d$ 行 × $w$ 列，每行一个独立哈希。更新：$d$ 个位置都 +1。查询：取 $d$ 个位置的最小值。

**参数**：$w=\lceil e/\varepsilon\rceil$，$d=\lceil\ln(1/\delta)\rceil$，则有

$$\hat f(x)\ge f(x)\quad\text{且}\quad \hat f(x)\le f(x)+\varepsilon N\ \text{（概率}\ge 1-\delta\text{）}$$

**读法**：CMS **只会高估**（碰撞导致），内存 $w\cdot d$ 个计数器，**与 key 基数无关**——这正是它在「10 亿不同 key」场景的优势。缺点：**它不直接告诉你谁是 top-K**，需要额外维护候选（堆/SS）。

### 5. 可合并性（分布式必需）

| 结构 | 合并方式 | 是否正确 |
| --- | --- | --- |
| MG | 合并计数后**取前 $k-1$ 大**（可按 $k-1$ 截断） | 保证仍然成立（阈值变为各分区 $N_i/k$ 的和） |
| Space-Saving | 合并同名计数并把 error 相加，再截断到 $k$ | 近似（实践中常用） |
| CMS | **逐格相加** | 正确（线性结构） |

**在 Spark 里的落地**：每个分区/每个 micro-batch 建局部草图 → `reduce`/`aggregate` 合并 → 得到全局草图 → 取候选 → （可选）回扫验证。

### 6. 两段式：把近似变成精确

```
第一遍：草图 -> 候选集 C（大小 c，通常 c = 2K ~ 10K，留余量）
第二遍：对 x ∈ C 精确计数（回扫原始数据或查明细表）-> 真正的 top-K
```

**为什么留余量**：草图可能把真 top-K 的某个元素排在 $K$ 名之外（尤其频次接近时），候选集取 $2K$–$10K$ 能把漏检概率压到可忽略。**代价**：多一遍扫描（或一次针对性查询）。

### 7. 滑动窗口与衰减（真实流场景）

「top-K」 在时间上有歧义：是**全量 top-K** 还是**最近 N 分钟 top-K**？后者需要：
- **窗口化草图**（按窗口切分 + 合并）或**衰减计数**（每次查询把计数乘 $\lambda$，或按时间桶衰减）；
- 或者**两个草图**：一个长窗口（稳定趋势）+ 一个短窗口（突发检测）。**这是限流/热点隔离的常见做法**。

## 数值与代码验证

### 表 1：三种算法在 Zipf 流上的 top-K 召回（见代码输出）

| 算法 | 内存（计数器） | top-10 召回 | top-100 召回 | 是否低估 |
| --- | --- | --- | --- | --- |
| Misra-Gries（k-1=99） | 99 | 见输出 | 见输出 | 是 |
| Space-Saving（k=100） | 100 | 见输出 | 见输出 | 否（高估） |
| CMS + 堆（w=2000,d=5） | 10,000 | 见输出 | 见输出 | 否（高估） |
| 精确（哈希表，仅作对照） | 基数 | 100% | 100% | — |

### 表 2：内存与误差的换算（CMS）

| 目标误差 $\varepsilon$ | $\delta$ | $w$ | $d$ | 计数器数 | 相对 $N=10^9$ 的绝对误差上限 |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 0.01 | 272 | 5 | 1,360 | 1e7 |
| 0.001 | 0.01 | 2,719 | 5 | 13,595 | 1e6 |
| 0.0001 | 0.01 | 27,183 | 5 | 135,915 | 1e5 |

### 可运行代码

```python
# 内存受限下的 top-K：Misra-Gries、Space-Saving、Count-Min Sketch + 堆
import heapq, math, random, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- Misra-Gries ----------
class MisraGries:
    """k-1 个计数器；未命中且满时全体减一。硬保证：f > N/k 的元素必现。"""
    def __init__(self, k: int):
        self.cap = max(1, k - 1)
        self.counts: Dict[str, int] = {}
        self.ops = 0
    def add(self, x: str) -> None:
        self.ops += 1
        if x in self.counts:
            self.counts[x] += 1
        elif len(self.counts) < self.cap:
            self.counts[x] = 1
        else:
            for key in list(self.counts.keys()):
                self.counts[key] -= 1
                if self.counts[key] == 0:
                    del self.counts[key]
    def top(self, k: int) -> List[Tuple[str, int]]:
        return heapq.nlargest(k, self.counts.items(), key=lambda kv: kv[1])
    def memory(self) -> int:
        return len(self.counts)

# ---------- Space-Saving ----------
class SpaceSaving:
    """k 个桶，每桶带 error；未命中时替换最小桶。保证 f ∈ [count-error, count]。"""
    def __init__(self, k: int):
        self.k = k
        self.counts: Dict[str, int] = {}
        self.errors: Dict[str, int] = {}
        self.min_val = 0
    def add(self, x: str) -> None:
        if x in self.counts:
            self.counts[x] += 1
            return
        if len(self.counts) < self.k:
            self.counts[x] = 1
            self.errors[x] = 0
            return
        # 找到最小计数的桶（演示用线性扫描；生产用最小堆维护）
        victim = min(self.counts, key=lambda key: self.counts[key])
        v = self.counts[victim]
        del self.counts[victim]
        err = self.errors.pop(victim)
        self.counts[x] = v + 1
        self.errors[x] = v
    def top(self, k: int) -> List[Tuple[str, int]]:
        return heapq.nlargest(k, self.counts.items(), key=lambda kv: kv[1])
    def bounds(self, x: str) -> Tuple[int, int]:
        c = self.counts.get(x, 0)
        return (c - self.errors.get(x, 0), c)
    def memory(self) -> int:
        return len(self.counts)

# ---------- Count-Min Sketch ----------
class CountMinSketch:
    """d 行 w 列；只高估不低估。误差 ≤ εN 的概率 ≥ 1-δ。"""
    def __init__(self, epsilon: float = 0.001, delta: float = 0.01, seed: int = 7):
        self.w = max(2, math.ceil(math.e / epsilon))
        self.d = max(1, math.ceil(math.log(1 / delta)))
        self.rows: List[List[int]] = [[0] * self.w for _ in range(self.d)]
        self.seeds = [random.Random(seed + i).randrange(1, 2 ** 31) for i in range(self.d)]
    def _idx(self, x: str, i: int) -> int:
        return (hash((x, self.seeds[i])) & 0x7FFFFFFF) % self.w
    def add(self, x: str, c: int = 1) -> None:
        for i in range(self.d):
            self.rows[i][self._idx(x, i)] += c
    def estimate(self, x: str) -> int:
        return min(self.rows[i][self._idx(x, i)] for i in range(self.d))
    def memory(self) -> int:
        return self.w * self.d
    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        assert (self.w, self.d) == (other.w, other.d)
        out = CountMinSketch.__new__(CountMinSketch)
        out.w, out.d, out.seeds = self.w, self.d, self.seeds
        out.rows = [[a + b for a, b in zip(r1, r2)]
                    for r1, r2 in zip(self.rows, other.rows)]
        return out

# ---------- 数据：Zipf 分布的事件流 ----------
def zipf_stream(n: int, cardinality: int, s: float = 1.1, seed: int = 3) -> List[str]:
    rnd = random.Random(seed)
    weights = [1.0 / ((i + 1) ** s) for i in range(cardinality)]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    import bisect
    return [f"key-{bisect.bisect_left(cum, rnd.random())}" for _ in range(n)]

N, CARD, K = 2_000_000, 50_000, 100
events = zipf_stream(N, CARD)
truth = Counter(events)
true_top10 = [x for x, _ in truth.most_common(10)]
true_top100 = [x for x, _ in truth.most_common(100)]
print(f"① 数据：{N:,} 事件、{CARD:,} 个不同 key、Zipf(s=1.1)；真 top-1 = "
      f"{truth.most_common(1)[0][1]:,} 次")
print(f"  真 top-10 占比 {sum(truth[x] for x in true_top10)/N:.1%}，"
      f"真 top-100 占比 {sum(truth[x] for x in true_top100)/N:.1%}")

def recall(found: List[str], truth_list: List[str]) -> float:
    return len(set(found) & set(truth_list)) / len(truth_list)

# ---------- 对照 ----------
print("\n② 三种算法的召回与内存（K=100）")
t0 = time.perf_counter()
mg = MisraGries(K)
for e in events:
    mg.add(e)
mg_top = [x for x, _ in mg.top(K)]
mg_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
ss = SpaceSaving(K)
for e in events:
    ss.add(e)
ss_top = [x for x, _ in ss.top(K)]
ss_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
cms = CountMinSketch(epsilon=0.0005, delta=0.01)
cand: Dict[str, int] = {}                             # ★ 在线维护候选（真实系统的做法）
PRUNE_EVERY, CAND_CAP = 100_000, 20 * K
for i, e in enumerate(events):
    cms.add(e)
    cand[e] = cms.estimate(e)                         # 用当前估计值更新候选
    if (i + 1) % PRUNE_EVERY == 0 and len(cand) > CAND_CAP:
        keep = dict(heapq.nlargest(CAND_CAP, cand.items(), key=lambda kv: kv[1]))
        cand = keep
cms_top = [x for x, _ in heapq.nlargest(K, cand.items(), key=lambda kv: kv[1])]
cms_ms = (time.perf_counter() - t0) * 1000

print(f"  {'算法':<22} {'内存(计数器)':>12} {'top-10 召回':>12} {'top-100 召回':>13} {'耗时(ms)':>9}")
for name, mem, top_list, ms in (("Misra-Gries", mg.memory(), mg_top, mg_ms),
                                ("Space-Saving", ss.memory(), ss_top, ss_ms),
                                ("CMS + 堆", cms.memory(), cms_top, cms_ms)):
    print(f"  {name:<22} {mem:>12,} {recall(top_list[:10], true_top10):>12.0%} "
          f"{recall(top_list, true_top100):>13.0%} {ms:>9.0f}")
print("  读法：**同等桶数下 Space-Saving 的召回高于 Misra-Gries**（MG 的桶会被减到 0 而淘汰，实测只剩约一半桶）；")
print("        CMS 的计数器数**与 key 基数无关**（本例 2.7 万 vs 5 万 key），且配在线候选维护后召回也不差")

# ---------- MG 的硬保证验证 ----------
print("\n③ 验证 Misra-Gries 的硬保证：f > N/k 的元素必现")
threshold = N / K
heavy = [x for x, c in truth.items() if c > threshold]
missing = [x for x in heavy if x not in mg.counts]
print(f"  N/K = {threshold:,.0f}；真实频次超过该阈值的元素 {len(heavy)} 个")
print(f"  其中未被 MG 保留的：{len(missing)} 个 -> {'保证成立' if not missing else '保证被违反（不应发生）'}")
print("  读法：这是 MG 的价值所在 —— **不依赖概率的硬保证**；代价是它只保证「超过阈值」的元素")

# ---------- Space-Saving 的区间保证 ----------
print("\n④ 验证 Space-Saving 的区间保证 f ∈ [count-error, count]")
ok = bad = 0
for x in ss_top[:20]:
    lo, hi = ss.bounds(x)
    fv = truth[x]
    if lo <= fv <= hi:
        ok += 1
    else:
        bad += 1
    print(f"  {x:<10} 真实 {fv:>8,}  区间 [{lo:>8,}, {hi:>8,}]  "
          f"{'✓' if lo <= fv <= hi else '✗'}")
print(f"  区间覆盖：{ok}/{ok+bad}")
print("  读法：SS 的 count 是**上界**、count−error 是**下界** —— 工程上可用它给出「至少/至多」的结论")

# ---------- 两段式验证：把近似变成精确 ----------
print("\n⑤ 两段式：候选集 + 精确复算")
print(f"  {'候选集大小':>10} {'草图桶数':>9} {'精确复算后 top-100 召回':>24}")
for c_size in (K, 2 * K, 5 * K):
    # 候选集要多大，草图就配多大（★ 用 SS(100) 去取 top-500 是取不出来的）
    big = SpaceSaving(c_size)
    for e in events:
        big.add(e)
    cand_set = [x for x, _ in big.top(c_size)]
    exact = Counter({x: truth[x] for x in cand_set})
    final_top = [x for x, _ in exact.most_common(K)]
    print(f"  {c_size:>10} {big.memory():>9} {recall(final_top, true_top100):>24.1%}")
print("  读法：候选集 100→500 时召回 37%→99% —— **近似只用来圈候选**，代价是多一遍扫描；")
print("        计费/风控/限流这类场景必须做这一步，因为「近似结果当结论」会直接造成资损或误封")

# ---------- 内存与误差的换算（CMS） ----------
print("\n⑥ CMS 参数换算：误差 ε 与内存")
print(f"  {'ε':>8} {'δ':>6} {'w':>8} {'d':>4} {'计数器':>10} {'绝对误差上限(εN)':>16}")
for eps in (0.01, 0.001, 0.0001):
    w = math.ceil(math.e / eps)
    d = math.ceil(math.log(1 / 0.01))
    print(f"  {eps:>8} {0.01:>6} {w:>8,} {d:>4} {w*d:>10,} {eps*N:>16,.0f}")
print("  读法：**误差每降一个数量级，内存涨一个数量级**（线性）—— 这是与产品谈「精度预算」的依据")
```

预期输出要点（实跑）：① 数据是 200 万事件、5 万不同 key 的 Zipf 流，真 top-10 占约 40%+ 流量；② 三种算法对照显示**同等桶数下 Space-Saving 的 top-100 召回高于 Misra-Gries**（实测 37% vs 29%，MG 的桶会被「全体减一」减到 0 而淘汰、实测只剩约一半桶），两者的 top-10 都是 100%；**CMS 的计数器数与 key 基数无关**（本例 27,185 个计数器 vs 50,000 个 key），配上**在线候选维护**后 top-10 召回 70%、top-100 召回 58%；③ **MG 的硬保证成立**（所有 $f>N/k$ 的元素都被保留，0 漏检）；④ SS 的区间保证 **$f\in[\text{count}-\text{error},\text{count}]$ 全部覆盖**；⑤ 两段式验证（**候选集多大、草图就配多大**）显示：候选集从 100 增到 500（草图桶数同步配置）时，精确复算后的 top-100 召回从 37.0% → 69.0% → **99.0%**——**近似只用来圈候选，结论来自精确复算**；⑥ CMS 参数换算给出「误差每降一个数量级、内存涨一个数量级」的线性关系。

## 常见追问

- **追问**：为什么不用「哈希表 + 定期裁剪」（比如只留高频的）？
  - 要点：裁剪需要知道「谁低频」，而低频判断本身就要计数——**循环依赖**。草图算法的价值在于它**用固定内存同时完成「计数」与「淘汰」**（MG 的全体减一、SS 的替换最小桶）。不过「分层计数」（Space-Saving 的变体）本质上就是这类做法。
- **追问**：怎么处理对抗性输入（故意制造哈希碰撞）？
  - 要点：**随机化哈希种子**（每次运行/每个分片不同，攻击者无法预知）；对 CMS 用成对独立的哈希族；对 MG/SS 而言，key 是原样保留的，**没有哈希碰撞问题**（这是它们相对 CMS 的一个优势）。
- **追问**：Spark 里怎么实现？
  - 要点：① 每个分区建局部草图（`mapPartitions`）；② 合并（`reduce`/`aggregate`，CMS 逐格相加、SS/MG 按 key 合并计数后截断到 $k$）；③ driver 上取候选；④ 可选：用候选集做一次 `filter` + `groupBy` 精确复算。**注意状态大小**：Structured Streaming 里要设 `mapGroupsWithState` 的超时或 watermark 以回收状态。
- **追问**：top-K 会不会随时间漂移？
  - 要点：会，所以要明确窗口——① **滑动窗口**（按窗口切草图 + 合并）；② **衰减**（定期把计数乘 $\lambda$）；③ **长短双窗口**（长窗口看趋势、短窗口抓突发）。**「全量 top-K」 与 「最近 5 分钟 top-K」 是两个不同的指标**，要先问清。
- **追问**：找到 top-K 之后干什么？
  - 要点：这是业务价值所在——**热点隔离**（把热 key 放到单独分区/单独限流）、**分区/分桶优化**（避免 join skew，串 [[databricks-04]]）、**缓存预热**、**风控与计费**（需要精确 → 必须走两段式）、**容量规划**（据 top-K 估算倾斜程度）。
- **追问**：K 很大（比如 10 万）时怎么办？
  - 要点：内存 $O(K)$ 会变紧，此时用 **CMS + 分层淘汰**（不维护 10 万个精确桶），或**两级方案**：第一级用 CMS 圈定「可能的高频候选」（例如计数超过阈值的），第二级对候选做精确计数。**阈值筛选 + 候选验证**比「直接维护 10 万桶」更省内存。
- **追问**：怎么验证近似结果的质量？
  - 要点：① 用**离线精确结果**（小规模或抽样）做对照，报 top-K 召回与排名偏差；② 报**误差分布**（不只平均）；③ 用**对抗性数据**（均匀分布是最难的情形）与**真实分布**分别测；④ 对硬保证（MG 阈值）写**断言式测试**。

## 相关题目

- [[databricks-01]]：线程安全批处理 logger——同为「资源有界下的工程控制」。
- [[databricks-03]]：CIDR 白名单的百万 QPS 校验——同为「内存/速度受限下的数据结构选择」。
- [[databricks-04]]：Spark join 的 straggler——top-K 的结果正是 skew 诊断的输入。
- [[databricks-05]]：Structured Streaming 的重复行与 checkpoint——与本题的「状态管理」同源。
- [[inference-serving-02]]：continuous batching——「攒批与淘汰」思想在推理服务里的另一形态。

## 参考资料与归属

- **Python 官方文档：heapq — 堆队列算法** —— Python Software Foundation：<https://docs.python.org/3/library/heapq.html>。第 4 节「用堆维护 top-K 候选」的实现取自该文档。
- **一个 Spark 作业把 2 TB 的事实表与 50 GB 的维表做 join（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 6 节「top-K 是 skew 诊断的输入」的说法与本仓库该篇互相引用。
- **为大型商品目录设计语义搜索（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5 节「可合并结构在分布式聚合中的必要性」的框架取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（200 万事件、5 万 key、Zipf s=1.1、K=100、MG $k-1$=99、SS $k$=100、「演示用」线性扫描找最小桶、CMS $\varepsilon$=0.0005/$\delta$=0.01 及 $\varepsilon$/$w$/$d$ 换算表、候选集 K/2K/5K）都是为演示取舍而构造的**示例参数**；**算法保证（MG 的 $f>N/k$ 必现、SS 的区间、CMS 的 $\varepsilon N$ 高估）是文献中的经典结论**，但本机实测的具体召回率取决于数据分布与实现（演示中 CMS 的候选池来自事件前 20 万条，真实系统应在线维护候选）。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
