---
type: question
id: microsoft-01
company: Microsoft
topic: coding
order: 1
question: 在一个大规模查询日志上实现“top-k 最高频搜索查询”，然后说说当日志变成跨多台机器的无界流时，哪些地方会出问题。
question_en: Implement "top-k most frequent search queries" over a large query log, then discuss what breaks when the log becomes an unbounded stream spread across many machines.
asked_at: []
level: 高阶
tags: [top-k, Count-Min Sketch, 流式算法, 分布式聚合, 时间衰减]
sources:
  - title: An Improved Data Stream Summary: The Count-Min Sketch and its Applications（延伸）
    url: https://dl.acm.org/doi/10.1016/j.jalgor.2003.12.001
    author: Cormode & Muthukrishnan
    published: 2005-04-01
  - title: Efficient Computation of Frequent and Top-k Elements in Data Streams（延伸）
    url: https://dl.acm.org/doi/10.1007/978-3-540-30570-5_27
    author: Metwally, Agrawal & El Abbadi
    published: 2005-01-05
  - title: Mining of Massive Datasets（延伸）
    url: http://www.mmds.org/
    author: Leskovec, Rajaraman & Ullman
    published: 2014-11-01
  - title: web 规模语料的流式去重（本仓库公司题库 · xAI 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [microsoft-02, microsoft-03, xai-08, coding-01, inference-serving-03]
updated: 2026-09-28
---

## 一句话答案

> **单机部分：精确计数不可行，必须用近似结构。**
> $$\text{精确哈希表：}10^8\ \text{个不同查询}\times50\ \text{B}=\mathbf{5\ GB}\qquad\text{Count-Min Sketch（CMS）：}\mathbf{5.3\ KB}\sim530\ \text{KB}$$
> **CMS 的两个参数**（宽度 $w$、深度 $d$）：
> $$w=\left\lceil\frac{e}{\varepsilon}\right\rceil,\qquad d=\left\lceil\ln\frac{1}{\delta}\right\rceil$$
> **误差保证**：以 $1-\delta$ 的概率，对任意项 $\hat c\ge c$ 且 $\hat c\le c+\varepsilon N$——**只高估、不低估**。
> **★ 关键判据：$\varepsilon$ 必须按「第 $k$ 名的相对频率」来定，而不是拍脑袋**
> $$\varepsilon N\ll c_k\quad\Longleftrightarrow\quad\varepsilon\ll p_k=\frac{c_k}{N}$$
> **本机算例**（$N=10^9$、第 $k$ 名计数 $c_k=5\times10^6$ 即 $p_k=0.5\%$）：
> | $\varepsilon$ | $\varepsilon N$（误差上界） | 与 $c_k$ 的关系 | top-k 可靠？ |
> | --- | --- | --- | --- |
> | 0.01 | $10^7$ | **是 $c_k$ 的 2 倍** | **不可靠**（任何低频项都可能挤进来） |
> | 0.001 | $10^6$ | $c_k$ 的 20% | 边界 |
> | **0.0005** | $5\times10^5$ | $c_k$ 的 10% | **可靠** |
> | **0.0001** | $10^5$ | $c_k$ 的 2% | **可靠** |
> **读法**：**若 $\varepsilon N$ 大于第 $k$ 名的计数，top-k 就没有意义**（一个只出现一次的新查询也可能被估成「高频」）——**这是 CMS 用于 top-k 时最容易搞错的一点**。
> **★ 实现陷阱（本机实测踩到）**：**哈希函数不能用 `(a·q+b) mod w`**——**$q$ 与 $q+w$ 会在每一行都撞进同一个桶**（造成「假重尾」）：
> | 哈希 | $\varepsilon$ | top-100 召回 | 假阳数 |
> | --- | --- | --- | --- |
> | **线性哈希** | 0.0001 | **25.0%** | 75 |
> | **splitmix64 混淆** | 0.0001 | **100.0%** | **0** |
> | **splitmix64** | 0.001 | **99.0%** | **0** |
> **读法**：**同一个算法，换个哈希函数就从「不可用」变成「完美」**——**这是「实现细节决定算法成败」的典型例子**。
> **「跨多台机器的无界流」会坏在哪（六条）**：
> | # | 问题 | 量化 / 对策 |
> | --- | --- | --- |
> | ① | **合并草图会累积误差** | $m$ 个草图合并后 $\delta'\approx m\delta$；要保持置信度，$d$ 按 $\ln(m/\delta)$ 增长（本机：$m{=}100$ 时 $d$ 从 5 涨到 **10**，内存 106 → **212 KB**） |
> | ② | **「无界」意味着「全时段 top-k」没有意义** | 需要**时间衰减**（滑动窗口 / 指数衰减）：否则三年前的爆款永远占榜 |
> | ③ | **重复计数**（至少一次投递） | 同一事件被算两次 → top-k 偏移；需要**幂等**（事件 ID 去重，串 [[xai-08]]） |
> | ④ | **数据倾斜**：单机看到的热点 ≠ 全局热点 | 按 key 分区会让热点集中；**两级聚合**（本地 top-k → 全局归并）能大幅降带宽 |
> | ⑤ | **时钟偏移与迟到数据** | 窗口边界不一致 → 需要**水位线（watermark）**与容错窗口 |
> | ⑥ | **协调成本 vs 精度** | 集中式精确计数（带宽 $\propto$ 去重基数）vs 分布式草图（带宽 $\propto$ 草图大小）——**后者的成本是「精度预算」** |
> **推荐架构（两级）**：
> ```
> 每台机器：CMS（或 Space-Saving）→ 每 T 秒把草图/候选集上报
> 聚合层：把草图逐元素相加（CMS 可合并）→ 维护全局 CMS → 定期输出 top-k
> 时间维度：多分辨率窗口（1min/1h/1d）或指数衰减计数器
> ```
> 一句话判据：**"精确不可行 → CMS（只高估）→ ε 按第 k 名的频率定 → 哈希必须混淆 → 分布式合并要补 d → 无界流要加时间衰减 → 重复投递要幂等"**。

## 面试官在考什么

- **是否知道精确计数不可行**：能否给出量级（本机：$10^8$ 个不同查询 = **5 GB**）并转向近似结构。
- **CMS 的参数与误差界**：能否写出 $w=\lceil e/\varepsilon\rceil$、$d=\lceil\ln(1/\delta)\rceil$ 与「只高估不低估」的保证。
- **★ top-k 的可靠性判据**：**能否指出 $\varepsilon N$ 必须远小于第 $k$ 名的计数**——**这是本题最核心的考点**（多数人只背公式，不检查这一点）。
- **哈希函数的坑**：能否指出**线性取模哈希会让 $q$ 与 $q+w$ 系统性碰撞**（本机：召回从 100% 崩到 25%）——**能说出这一点的人极少**。
- **替代方案**：能否提到 **Space-Saving / Misra-Gries**（$O(k)$ 内存、确定性保证），并说明与 CMS 的取舍。
- **分布式合并**：能否指出 **CMS 可按元素相加合并**（关键性质），但 **$\delta$ 会累积**（需要补 $d$）。
- **「无界」的含义**：能否指出**必须加时间衰减**——「全时段 top-k」在产品上没有意义。
- **工程细节**：重复投递（幂等）、倾斜（两级聚合）、时钟偏移（水位线）、迟到数据。
- **成本视角**：能否比较「精确集中式」（带宽 $\propto$ 基数）与「分布式草图」（带宽 $\propto$ 草图）的取舍。
- **诚实**：承认**近似结构会给出假阳**（top-k 里混入低频项），所以**要么留余量、要么加一步精确校验**（对候选做二次精确计数）。

**常见错误答案**

- 直接说「用哈希表统计然后排序」（**没有考虑内存与流式**）。
- 背了 CMS 公式但**不检查 $\varepsilon N$ 与第 $k$ 名的关系**（top-k 结果不可信）。
- **用线性取模哈希**（系统性碰撞，本机实测召回 25%）。
- 忽略**合并导致的 $\delta$ 累积**（多机合并后置信度崩塌）。
- 把「无界流」当成「更大的一批数据」（**忽略时间维度**）。
- 忽略重复投递（至少一次语义下会重复计数）。
- 只做单层聚合（**热点机器成为瓶颈**）。
- 认为「近似 = 不可用」（**应该说明假阳率与校验手段**）。

## 原理与推导

### 1. 为什么精确不可行

$$\text{内存}=D\times(\text{key 长度}+\text{计数器})\quad(D=\text{不同查询数})$$

**本机算例**：$D=10^8$、每项 50 字节 → **5 GB**——**单机放不下**（而且随日志增长**无界**）。

**三条出路**：① **采样**（只统计一部分，但 top-k 的尾部会失真）；② **分片 + 精确**（每片精确，归并时带宽大）；③ **近似草图**（固定内存、单遍、可合并）。

**读法**：**「大规模 + 无界」这两个约束直接排除精确方案**——所以问题变成「选哪个近似结构」。

### 2. Count-Min Sketch：参数与保证

**结构**：$d$ 行、每行 $w$ 个计数器；每个项用 $d$ 个哈希映射到每行一个桶。

| 操作 | 复杂度 |
| --- | --- |
| 更新 | $O(d)$ |
| 查询 | $O(d)$（取 $d$ 个桶的**最小值**） |
| 内存 | $w\times d$ 个计数器 |

**参数**：

$$w=\left\lceil\frac{e}{\varepsilon}\right\rceil,\qquad d=\left\lceil\ln\frac{1}{\delta}\right\rceil$$

**保证**：以 $1-\delta$ 的概率，$\hat c\ge c$（**只高估**）且 $\hat c\le c+\varepsilon N$。

**本机算例**（内存）：

| 配置 | $w$ | $d$ | 内存 |
| --- | --- | --- | --- |
| $\varepsilon{=}0.01,\delta{=}0.01$ | 272 | 5 | **5.3 KB** |
| $\varepsilon{=}0.001,\delta{=}0.01$ | 2,719 | 5 | 53.1 KB |
| $\varepsilon{=}0.0005$ | 5,437 | 5 | 106.2 KB |
| $\varepsilon{=}0.0001$ | 27,183 | 5 | 530.9 KB |

**读法**：**CMS 只要几十到几百 KB**——**这就是流式场景的标准选择**。**注意 $d$ 只随 $\ln(1/\delta)$ 增长**（所以提高置信度很便宜），**$w$ 随 $1/\varepsilon$ 线性增长**（提高精度更贵）。

### 3. ★ top-k 的可靠性判据

**CMS 的 top-k 为什么可能出错**：CMS 只保证「高估不超过 $\varepsilon N$」。所以一个真实计数为 $c$ 的项，估计值可能是 $c+\varepsilon N$。**若 $\varepsilon N$ 与第 $k$ 名的计数同量级，那么任何项都可能被推进 top-k。**

$$\text{可靠条件：}\ \varepsilon N\ll c_k\quad\Longleftrightarrow\quad\varepsilon\ll p_k=\frac{c_k}{N}$$

**本机算例**（$N=10^9$、$c_k=5\times10^6$）：

| $\varepsilon$ | $\varepsilon N$ | $c_k$ 的倍数 | 结论 |
| --- | --- | --- | --- |
| 0.01 | $10^7$ | **2.0×** | **完全不可靠** |
| 0.001 | $10^6$ | 0.2× | 边界 |
| **0.0005** | $5\times10^5$ | **0.1×** | **可靠** |
| **0.0001** | $10^5$ | **0.02×** | **可靠** |

**读法**：**「$\varepsilon$ 取 0.01」这种拍脑袋的取值会让 top-k 变成随机名单**。**实践做法**：**先用一小段数据估计 $p_k$（第 $k$ 名的相对频率），再取 $\varepsilon\approx p_k/10$**。**代价**：$\varepsilon$ 小 10 倍 → $w$ 大 10 倍（**内存线性上升**）——**但绝对值仍然很小**（530 KB）。

### 4. ★ 实现陷阱：哈希函数（本机实测踩到）

**错误做法**：

```python
bucket = (a * q + b) % w          # ✗ 线性取模
```

**为什么错**：**$q$ 与 $q+w$ 映射到同一个桶**（因为 $a(q+w)+b\equiv aq+b \pmod w$）——**而这对所有行都成立**（每行的 $a,b$ 不同，但 $w$ 相同）。于是 $w$ 的倍数项与 $q{=}0$ 在**每一行都碰撞**，CMS 取最小值后它们**被估成与最热项相同的计数**——**「假重尾」**。

**本机实测**（Zipf 流、$10^6$ 条日志、$10^5$ 个不同查询、top-100）：

| 哈希 | $\varepsilon$ | 召回 | 假阳 |
| --- | --- | --- | --- |
| **线性取模** | 0.0001 | **25.0%** | 75 |
| **splitmix64** | 0.0001 | **100.0%** | **0** |
| splitmix64 | 0.001 | 99.0% | 0 |
| splitmix64 | 0.01 | 46.0% | 54 |

**读法**：**同一个 CMS、同一个 $\varepsilon$，换个哈希函数就从「不可用」变成「完美」**。**正确做法**：用**混淆函数**（splitmix64 / MurmurHash / xxHash 的 finalizer）把 $q$ 打散，**再做取模**：

```python
def mix(x):                       # splitmix64 的 finalizer
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)
bucket = mix(q + seed_r) % w
```

**注意**：**「用随机数当哈希系数」不是解**——**线性结构本身**才是问题（本机实测：$\varepsilon{=}0.01$ 时召回 0%、假阳 100%）。

### 5. 替代方案：Space-Saving（$O(k)$ 内存）

**思路**：只维护 $k$ 个计数器；新项若不在表中且表未满 → 加入；表满且新项不在表中 → **替换计数最小的那个**，并把它的计数作为新项的初始计数。

**保证**：**任何真实计数 $>N/k$ 的项都会被报告**；报告的计数**高估不超过 $N/k$**。

**与 CMS 的取舍**：

| 维度 | CMS | Space-Saving |
| --- | --- | --- |
| 内存 | $w\times d$（可调精度） | $O(k)$（只跟踪 $k$ 个） |
| 保证 | 任意项的误差界 | **重尾项一定被报告** |
| 合并 | **逐元素相加**（简单） | 需要合并候选集并重新计数（较复杂） |
| 适用 | 需要「任意项计数」 | **只要 top-k** |

**读法**：**如果目标就是 top-k，Space-Saving 更省内存**（$O(k)$ 而不是 $O(1/\varepsilon)$）；**如果还要查任意项的计数，用 CMS**。**实践中常用「CMS + 堆」或「Space-Saving + 全局归并」**。

### 6. ★ 无界流 + 多机：六条会坏的地方

#### ① 合并草图会累积误差

CMS 可合并（**逐元素相加**），但置信度会退化：**$m$ 个独立草图合并后 $\delta'\approx m\delta$**。要保持目标 $\delta'$，每个草图的 $d$ 要按 $\ln(m/\delta')$ 增长：

| 机器数 $m$ | 合并后 $\delta'$ | 需要的 $d$ | 内存（$\varepsilon{=}0.0005$） |
| --- | --- | --- | --- |
| 1 | 0.01 | 5 | 106 KB |
| 10 | 0.10 | 7 | 149 KB |
| **100** | **0.63** | **10** | **212 KB** |
| 1000 | 1.00 | 12 | 255 KB |

**读法**：**合并的成本不是带宽，而是「精度预算」**——$m$ 每涨 10 倍，$d$ 涨约 2（**因为 $d\propto\ln m$**）。**这是好消息**：**内存增长是对数的**。

#### ② 「无界」意味着必须加时间衰减

**「全时段 top-k」在产品上没有意义**（三年前的爆款会永远占榜）。**两种做法**：

| 做法 | 机制 | 内存 |
| --- | --- | --- |
| **多分辨率窗口** | 同时维护 1 min / 1 h / 1 d 的草图，定期滚动 | $\times$ 窗口数 |
| **指数衰减** | 每 $T$ 时间把所有计数器**乘以 $\lambda$**（如 0.5） | 与单草图相同 |

**读法**：**指数衰减最省内存**（只多一个「定期减半」的操作），**但衰减因子与周期要按产品语义定**（「最近一小时的热搜」 vs 「今天的热搜」）。

#### ③ 重复投递 → 必须幂等

**至少一次（at-least-once）语义下，同一事件可能被算两次**。**影响**：top-k 偏移（**尤其对「恰好跨过第 $k$ 名」的项**）。**对策**：① **事件 ID 去重**（在源头或聚合层，串 [[xai-08]]）；② **精确一次（exactly-once）语义**（代价大）；③ **接受小误差**（如果重复率低）。

#### ④ 数据倾斜：本地热点 ≠ 全局热点

**按 key 分区**会让「某查询的所有事件」落到同一台机器 → **热点机器**。**对策**：**两级聚合**——① 每台机器先算**本地 top-k**（$O(k)$ 而不是全量）；② 只把**候选集**上报（**带宽 $\propto$ 候选数**，而不是事件数）；③ 全局层归并候选并**重新精确计数**（候选集小，可以精确）。

**读法**：**两级聚合是「分布式 top-k」的标准架构**——**它把带宽从「事件量级」降到「候选量级」**（差几个数量级）。

#### ⑤ 时钟偏移与迟到数据

窗口聚合依赖时间，而**各机器的时钟不一致**、**数据会迟到**。**对策**：① **事件时间（event time）而不是处理时间**；② **水位线（watermark）**：等迟到数据到某个界限后再关闭窗口；③ **允许窗口重算**（把迟到数据并入旧窗口并修正输出）。

#### ⑥ 协调成本 vs 精度

| 方案 | 带宽 | 精度 |
| --- | --- | --- |
| **集中式精确**（所有事件发到一处） | $\propto$ 事件量 | 精确 |
| **分片精确 + 全量归并** | $\propto$ 去重基数 | 精确 |
| **分布式草图**（本地 CMS → 合并） | $\propto$ 草图大小 × 机器数 | 有界误差 |

**读法**：**「精确」的带宽成本随基数增长，而「草图」是固定成本**——**代价是精度预算**。**选择依据**：**top-k 的 $k$ 有多大、精度要求多高、事件量多少**。

## 数值与代码验证

### 表 1：CMS 的规模、可靠性判据、哈希陷阱、分布式合并（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| 精确 vs 近似的内存（$10^9$ 条日志、$10^8$ 个不同查询） | 精确哈希表 **5.0 GB**（每项 50 字节）；CMS eps=0.01 **5.3 KB**（w=272, d=5）；0.001 53.1 KB（w=2719）；0.0005 106.2 KB（w=5437）；0.0001 530.9 KB（w=27183） |
| top-k 的可靠性判据（$N=10^9$、$c_k=5\times10^6$） | eps 0.01 → eps·N **10,000,000（c_k 的 2.00 倍）不可靠**；0.001 → 1,000,000（0.20）边界；0.0005 → 500,000（0.10）可靠；0.0001 → 100,000（0.02）可靠——**eps 要按 $p_k/10$ 来取** |
| 哈希陷阱（线性取模 vs splitmix64；Zipf 流 $10^6$ 条、$10^5$ 个不同查询、top-100） | 线性取模：召回 **0.0% / 2.0% / 25.0%**、假阳 100 / 98 / 75；**splitmix64：46.0% / 99.0% / 100.0%、假阳 54 / 0 / 0**——实现细节决定算法成败 |
| 分布式合并（机器数 1 / 10 / 100 / 1000） | 合并后 delta **0.01 / 0.10 / 0.63 / 1.00**；需要的 d **5 / 7 / 10 / 12**；内存 106.2 / 148.7 / 212.4 / 254.9 KB——**d 只随 ln(m) 增长** |
| 无界流的六个问题 | 合并累积误差；无界导致全时段无意义（要时间衰减）；重复投递；数据倾斜（两级聚合）；时钟偏移与迟到（事件时间 + 水位线） |

### 可运行代码

```python
# top-k 高频查询：精确 vs CMS 的内存、top-k 可靠性判据、哈希陷阱、分布式合并的精度代价
import bisect
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

MASK = (1 << 64) - 1

def mix(x: int) -> int:
    """splitmix64 的 finalizer：把 key 打散后再取模（★ 不能用线性取模）"""
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)

print("① 精确 vs 近似：内存对比（10^9 条日志、10^8 个不同查询）")
print(f"  {'方案':<24} {'内存':>10} 说明")
print(f"  {'精确哈希表':<24} {10**8 * 50 / 1e9:>8.1f} GB 每项 50 字节")
for eps in (0.01, 0.001, 0.0005, 0.0001):
    w = math.ceil(math.e / eps)
    d = math.ceil(math.log(1 / 0.01))
    print(f"  {'CMS eps=' + str(eps):<24} {w * d * 4 / 1024:>6.1f} KB w={w} d={d}")
print("  读法：**CMS 只要几十到几百 KB**（vs 精确的 5 GB）—— 流式场景必须用近似结构")

print("")
print("② top-k 的可靠性判据：eps*N 必须远小于第 k 名的计数")
N, C_K = 10**9, 5e6
print(f"  假设 N={N:.0e}、第 k 名计数 c_k={C_K:.0e}（即 p_k={C_K/N:.2%}）")
print(f"  {'eps':>8} {'eps*N':>14} {'c_k 的倍数':>11} 结论")
for eps in (0.01, 0.001, 0.0005, 0.0001):
    ratio = eps * N / C_K
    verdict = "可靠" if ratio <= 0.1 else ("边界" if ratio <= 0.2 else "**不可靠**")
    print(f"  {eps:>8} {eps*N:>14,.0f} {ratio:>11.2f} {verdict}")
print("  读法：**eps 要按 p_k/10 来取** —— 否则任何低频项都可能被高估进 top-k（top-k 变成随机名单）")

print("")
print("③ 哈希陷阱：线性取模 vs 混淆函数（Zipf 流、10^6 条日志、10^5 个不同查询、top-100）")
def cms_topk(n_events: int = 10**6, n_distinct: int = 10**5, eps: float = 0.0001,
             delta: float = 0.01, k: int = 100, linear: bool = False,
             seed: int = 0) -> Tuple[float, int, int, int]:
    rng = random.Random(seed)
    w = math.ceil(math.e / eps)
    d = math.ceil(math.log(1 / delta))
    weights = [1.0 / (i + 1) for i in range(n_distinct)]
    total = sum(weights)
    cum: List[float] = []
    acc = 0.0
    for x in weights:
        acc += x
        cum.append(acc / total)
    table = [[0] * w for _ in range(d)]
    seeds = [rng.randrange(1, 2 ** 61) for _ in range(d)]
    lin = [(rng.randrange(1, 2 ** 61), rng.randrange(1, 2 ** 61)) for _ in range(d)]
    def bucket(r: int, q: int) -> int:
        return (lin[r][0] * q + lin[r][1]) % w if linear else mix(q + seeds[r]) % w
    exact: Counter = Counter()
    for _ in range(n_events):
        q = bisect.bisect_left(cum, rng.random())
        exact[q] += 1
        for r in range(d):
            table[r][bucket(r, q)] += 1
    def est(q: int) -> int:
        return min(table[r][bucket(r, q)] for r in range(d))
    true_top = [q for q, _ in exact.most_common(k)]
    ranked = sorted(((est(q), q) for q in range(n_distinct)), reverse=True)
    cms_top = [q for _, q in ranked[:k]]
    recall = len(set(true_top) & set(cms_top)) / k
    ck = exact[true_top[-1]]
    fp = sum(1 for q in cms_top if exact[q] < ck * 0.5)
    return recall, fp, w, d
print(f"  {'哈希':<12} {'eps':>8} {'w':>7} {'d':>3} {'召回':>7} {'假阳':>6}")
for linear in (True, False):
    for eps in (0.01, 0.001, 0.0001):
        r, fp, w, d = cms_topk(eps=eps, linear=linear)
        print(f"  {'线性取模' if linear else 'splitmix64':<12} {eps:>8} {w:>7} {d:>3} "
              f"{r:>6.1%} {fp:>6}")
print("  读法：**线性取模让 q 与 q+w 在每一行都撞同一桶** -> 召回崩到 25%、假阳 75；")
print("        换成混淆函数后 **eps=0.0001 时召回 100%、假阳 0** —— 实现细节决定算法成败")

print("")
print("④ 分布式合并：delta 会累积（需要更大的 d，但只按 ln m 增长）")
w = math.ceil(math.e / 0.0005)
print(f"  {'机器数 m':>9} {'合并后 delta':>12} {'需要的 d':>9} {'内存':>10}")
for m in (1, 10, 100, 1000):
    delta_merged = 1 - (1 - 0.01) ** m
    d_need = max(1, math.ceil(math.log(m / 0.01)))
    print(f"  {m:>9} {delta_merged:>12.2f} {d_need:>9} {w * d_need * 4 / 1024:>8.1f} KB")
print("  读法：**合并的成本是「精度预算」而不是带宽**；好消息是 d 只随 ln(m) 增长（内存对数上升）")

print("")
print("⑤ 无界流的六个问题与对策")
@dataclass
class Issue:
    name: str
    why: str
    fix: str
ISSUES = [
    Issue("合并累积误差", "m 个草图合并后 delta 约 m*delta", "按 ln(m/delta) 补 d（内存对数增长）"),
    Issue("无界 -> 全时段无意义", "三年前的爆款永远占榜", "时间衰减：多分辨率窗口或指数衰减"),
    Issue("重复投递", "至少一次语义下重复计数", "事件 ID 去重 / 精确一次 / 接受小误差"),
    Issue("数据倾斜", "本地热点 != 全局热点", "两级聚合：本地 top-k -> 候选归并"),
    Issue("时钟偏移与迟到", "窗口边界不一致、数据迟到", "事件时间 + 水位线 + 允许窗口重算"),
    Issue("协调成本 vs 精度", "精确归并的带宽随基数增长", "分布式草图（固定带宽，代价是精度预算）"),
]
print(f"  {'问题':<20} {'原因':<28} 对策")
for i in ISSUES:
    print(f"  {i.name:<20} {i.why:<28} {i.fix}")
print("  读法：**这六条就是「无界 + 多机」相对「单机一批」多出来的全部问题**")

print("")
print("⑥ 替代方案：CMS vs Space-Saving")
@dataclass
class Algo:
    dim: str
    cms: str
    ss: str
ROWS = [
    Algo("内存", "w*d（精度可调）", "O(k)（只跟踪 k 个）"),
    Algo("保证", "任意项误差 <= eps*N", "真实计数 > N/k 的项必被报告"),
    Algo("合并", "逐元素相加（简单）", "合并候选集并重新计数（较复杂）"),
    Algo("适用", "需要查任意项计数", "只要 top-k"),
]
print(f"  {'维度':<8} {'Count-Min Sketch':<26} Space-Saving")
for r in ROWS:
    print(f"  {r.dim:<8} {r.cms:<26} {r.ss}")
print("  读法：**只要 top-k 时 Space-Saving 更省内存**；**要查任意项计数时用 CMS**")
```

预期输出要点（实跑）：① **精确 5 GB vs CMS 5.3 KB–530 KB**；② **可靠性判据**：$\varepsilon{=}0.01$ 时 $\varepsilon N$ 是第 $k$ 名计数的 **2.0 倍**（top-k 完全不可靠），$\varepsilon{=}0.001$ 降到 0.2 倍（边界），$\varepsilon{=}0.0005$ 到 **0.1 倍**（可靠）、$\varepsilon{=}0.0001$ 到 **0.02 倍**；③ **哈希陷阱**：线性取模在 $\varepsilon{=}0.0001$ 时召回 **25.0%**、假阳 **75**，换成 splitmix64 后 **100.0% / 0**；④ **合并代价**：$m{=}100$ 时 $\delta$ 从 0.01 涨到 **0.63**，$d$ 从 5 涨到 **10**（内存 106 → **212 KB**）；⑤ 无界流的六个问题与对策；⑥ CMS 与 Space-Saving 的四维对比。

## 常见追问

- **追问**：为什么 CMS 取最小值而不是平均值？
  - 要点：**因为碰撞只会让计数变大（只高估）**。① **每个桶的值 = 该桶内所有项的真实计数之和** $\ge$ 目标项的真实计数；② **取最小值**给出最紧的上界（哪一行碰撞最少就用哪一行）；③ **平均值没有这个保证**（碰撞多的行会把估计拉高，而平均值无法排除它）。**这是 CMS 的核心设计**：**用 $d$ 个独立哈希降低「全部行都被严重碰撞」的概率**。
- **追问**：如果 top-k 里混进了假阳怎么办？
  - 要点：**两步**：① **对候选做精确二次计数**（候选集只有几百到几千项，**精确计数完全可行**）——**这是最干净的做法**；② **留余量**（取 top-$2k$ 再筛，或把 $\varepsilon$ 取得更小）。**实践架构**：**草图负责「筛出候选」，精确计数负责「定榜」**——**这正是「两级聚合」的思路**（串本机 ④ 的对策）。
- **追问**：为什么不用「按 key 哈希分片 + 每片精确」？
  - 要点：**可以，但带宽与内存都随基数增长**：① **每片精确**需要每片保存**该片的全部不同 key**（内存 $\propto D/m$，仍然无界）；② **归并时要传输全部 key**（带宽 $\propto D$）。**对比**：**草图方案传输的是固定大小的草图**（几十到几百 KB × 机器数）。**判据**：**若 $D$ 可控（例如只有几百万），分片精确更准且实现简单；若 $D$ 无界，必须用草图。**
- **追问**：时间衰减怎么做最省？
  - 要点：**指数衰减（decayed counters）**：① **机制**——每 $T$ 时间把所有计数器乘以 $\lambda$（如 0.5，即半衰期 $T$）；② **成本**——一次 $O(w\times d)$ 的乘法（**可以后台做**）；③ **效果**——「最近的热度」自然占优，**内存与单草图相同**。**注意**：① **衰减因子要与产品语义对齐**（「最近 1 小时」→ 半衰期约 20 分钟）；② **整数计数器做衰减要小心精度**（可以用浮点或周期性右移）；③ **多个时间尺度需要多个草图**（1 min / 1 h / 1 d 各一份，**内存 × 3**）。
- **追问**：怎么验证 top-k 的实现是对的？
  - 要点：**四步**：① **与精确实现对拍**（小数据量下用哈希表精确计数，比较 top-k 的重合度）——**本机的模拟就是这么发现哈希陷阱的**；② **合成分布测试**（Zipf 分布有已知的 top-k，**可以直接验证召回**）；③ **检查哈希函数**（**单独测哈希的分布均匀性**，尤其是「$q$ 与 $q+w$ 是否撞同一桶」这种结构性碰撞）；④ **注入倾斜与重复**（验证两级聚合与幂等）。**关键**：**「召回率」与「假阳率」要分开测**——本机的线性哈希在 $\varepsilon{=}0.01$ 时召回 0% 但**假阳 100%**，只看召回会以为「参数没调好」。
- **追问**：如果要求「精确 top-k」呢？
  - 要点：**那就必须付出带宽或内存**：① **集中式精确**（所有事件发到一处，带宽 $\propto$ 事件量——**通常不可行**）；② **分片精确 + 全量归并**（带宽 $\propto$ 基数）；③ **两级：草图筛候选 + 精确定榜**（**候选集精确计数**，这是**最实用的「精确 top-k」方案**）。**注意**：**「精确」的代价随基数增长，而「近似 + 校验」是固定成本**——**所以绝大多数生产系统选后者**。

## 相关题目

- [[microsoft-02]]：agent host 的工具调用层设计——同一家公司的 low-level design 题。
- [[microsoft-03]]：Copilot 的 3 秒 p95 预算——延迟与成本的分解方法。
- [[xai-08]]：流式去重——**同一套「固定内存 + 单遍 + 可合并」的近似结构**（Bloom/MinHash）。
- [[coding-01]]：LRU 缓存——另一种「有界内存下的数据结构」设计。
- [[inference-serving-03]]：多租户与优先级调度——聚合层的负载与公平性。

## 参考资料与归属

- **An Improved Data Stream Summary: The Count-Min Sketch and its Applications（延伸）** —— Cormode & Muthukrishnan，2005-04-01：<https://dl.acm.org/doi/10.1016/j.jalgor.2003.12.001>。**CMS 的定义、$w=\lceil e/\varepsilon\rceil$ 与 $d=\lceil\ln(1/\delta)\rceil$ 的参数选择、以及「只高估」的误差保证**来自这篇。
- **Efficient Computation of Frequent and Top-k Elements in Data Streams（延伸）** —— Metwally, Agrawal & El Abbadi，2005-01-05：<https://dl.acm.org/doi/10.1007/978-3-540-30570-5_27>。**Space-Saving 算法（$O(k)$ 内存、替换最小计数项、「真实计数 $>N/k$ 必被报告」）**来自这篇。
- **Mining of Massive Datasets（延伸）** —— Leskovec, Rajaraman & Ullman，2014-11-01：<http://www.mmds.org/>。第 4 章的**流式算法与衰减窗口（decayed counters）**是本题「无界流要加时间衰减」的依据。
- **web 规模语料的流式去重（本仓库公司题库 · xAI 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。该篇的**Bloom/MinHash 的「固定内存 + 单遍 + 可合并」框架**与本篇的 CMS 是同一类设计，其**假阳/假阴取舍**也被本篇沿用。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（$10^9$ 条日志、$10^8$ 个不同查询、每项 50 字节、$\varepsilon$ 取 0.01–0.0001、$\delta{=}0.01$、$N{=}10^9$、$c_k{=}5\times10^6$、Zipf 流的 $10^6$ 条日志与 $10^5$ 个不同查询、top-100、机器数 1–1000）都是为演示机制而构造的**示例参数与显式假设**；**$w$/$d$ 公式、误差界、内存与合并的 $\delta$ 累积都是解析结论**（可复现），**CMS 模拟的召回/假阳是本机实跑结果**（依赖 Zipf 参数与随机种子，**绝对值会变，但「线性哈希系统性碰撞」这一结论是结构性的**）。**「第 $k$ 名的相对频率 $p_k$」必须按自家数据实测**——本机用的 0.5% 是示意。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
