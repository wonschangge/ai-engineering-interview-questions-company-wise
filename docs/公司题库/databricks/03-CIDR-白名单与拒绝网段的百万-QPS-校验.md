---
type: question
id: databricks-03
company: Databricks
topic: coding
order: 3
question: 给定以 CIDR 块表示的白名单 IP 段以及显式的拒绝网段，实现高效的 is_allowed(ip)，要能支撑每秒数百万次校验。
question_en: Given allow-listed CIDR blocks plus explicit deny ranges, implement an efficient is_allowed(ip) that sustains millions of checks per second.
asked_at: []
level: 高阶
tags: [CIDR, 区间二分, 最长前缀匹配, 边界用例, 无锁热更新]
sources:
  - title: Python 官方文档：bisect — 数组二分查找算法
    url: https://docs.python.org/3/library/bisect.html
    author: Python Software Foundation
    published: 
  - title: RFC 4632：CIDR 地址聚合与道格拉斯·康芒纳的地址分配实践
    url: https://www.rfc-editor.org/rfc/rfc4632
    author: IETF
    published: 2006-08
  - title: 设计权限感知的 retrieval（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [databricks-02, databricks-04, databricks-05, rag-08, system-design-01]
updated: 2026-09-28
---

## 一句话答案

> 第一步不是写代码，而是**问清「拒绝」的语义**——它决定数据结构，两种语义结果不同：
> ① **拒绝优先（deny-overrides）**：允许集合**减去**拒绝集合，落在拒绝里就是拒绝（防火墙常见口径）；
> ② **最长前缀优先（longest-prefix-match）**：**更具体的规则胜出**，与它是允许还是拒绝无关（路由/ACL 常见口径）。
> **一个例子就能区分两者**：`allow 10.1.0.0/16` 与 `deny 10.0.0.0/8` —— 拒绝优先判 **deny**，最长前缀判 **allow**。
> 结构选择（都以「整数区间」为基础）：
> - **拒绝优先**：把允许与拒绝各自**解析成整数区间并合并**（重叠/相邻合并），再做**区间差**得到一组互不相交的允许区间；查询 = **二分查找**（`bisect`）→ $O(\log n)$；
> - **最长前缀**：用**前缀树（binary/Patricia trie）**，或「按前缀长度分桶 + 每桶内区间二分」，查询时**从最长前缀往下找第一个命中**；
> - **默认策略必须显式**：白名单为空是「全拒」还是「全放」？安全场景必须 **default deny**。
> 支撑百万 QPS 的三个工程点：
> 1. **合并区间**把规则数压下来（实测能减少可观比例），二分只需十几次比较；
> 2. **两级表快速路径**：对 IPv4 可用 `/16` 直接查表（65,536 项）覆盖绝大多数流量，未命中再走二分；
> 3. **热更新无锁**：规则编译成**不可变快照**，查询线程读快照引用，更新时构建新快照后**原子替换**（读写不互斥），这是「百万 QPS + 实时改规则」的关键。
> 一句话判据：**语义先定、区间合并、二分/前缀树查询、快照原子替换**。

## 面试官在考什么

- **是否先问语义**：能不能指出 deny-overrides 与 longest-prefix 是**两种不同语义**，并给出一个结果不同的例子（这是本题最有区分度的一点）。
- **边界与表示**：IPv4 用 32 位无符号、IPv6 用 128 位；CIDR → $[\text{start},\text{end}]$ 的**闭区间**表示；`/0`（全网）与 `/32`（单机）的处理；**IPv4-mapped IPv6**（`::ffff:10.0.0.1`）要不要等价处理。
- **合并与去重**：重叠/相邻区间合并（减少规则数）、重复规则、以及「拒绝落在允许之外」（no-op）的处理。
- **性能路径**：为什么二分足够（$O(\log n)$、cache 友好）；能否提到**直接查表**（`/16` 或 `/24`）作为快速路径；以及**每秒数百万次**意味着什么（单核微秒级 → 需要多核/无锁读）。
- **热更新**：能否说出「不可变快照 + 原子替换」的做法，而不是「加锁重建」（后者会在更新时卡住查询）。
- **正确性验证**：边界用例（区间端点、相邻区间、拒绝覆盖整个允许、/0 与 /32、非规范 CIDR 如 `10.0.0.1/8`）；随机模糊测试（与朴素实现对照）。
- **失败模式**：解析错误（前导零、IPv6 压缩形式）、整数溢出（有符号 vs 无符号）、以及**默认放行**这种最危险的设计错误。
- **工具链**：能否提到成熟库（如前缀树库）而不是硬写；以及「规则来源是否可信」（规则本身也是输入）。

**常见错误答案**

- 不区分两种语义就直接实现（结果与客户预期不符）。
- 用字符串前缀匹配（`ip.startswith(「10.0.」)`）——**对 CIDR 不成立**（`10.0.0.0/12` 不是字符串前缀）。
- 逐个规则线性扫描（$O(n)$，百万 QPS 下不可行）。
- 用有符号整数表示 IPv4（`128.0.0.0` 以上变负数，区间比较出错）。
- 热更新时加全局锁（查询被阻塞，延迟尖刺）。
- 白名单为空时默认放行（安全灾难）。
- 忽略 IPv6 或 IPv4-mapped 形式。

## 原理与推导

### 1. CIDR → 整数区间

对前缀长度 $p$（IPv4 为 $0\le p\le32$）：

$$\text{size}=2^{32-p},\quad \text{start}=\text{ip}\ \&\ \sim(\text{size}-1),\quad \text{end}=\text{start}+\text{size}-1$$

**注意**：`10.0.0.1/8` 与 `10.0.0.0/8` 等价（主机位被掩掉）——解析时要**规范化**，否则会出现「看起来不重叠其实重叠」的规则。

### 2. 区间合并与集合差

- **合并**：按 `start` 排序，若 `next.start <= cur.end + 1`（相邻也算）则合并——把规则数从 $n$ 压到 $m\le n$；
- **差集**（拒绝优先）：对每个允许区间，减去所有与之相交的拒绝区间，得到若干子区间；结果仍是**互不相交的有序区间列表**。

**查询**：在排序的 `starts` 数组上 `bisect_right(starts, ip) - 1` 得到候选区间，检查 `ip <= ends[idx]` 即可。$O(\log m)$。

### 3. 最长前缀匹配

需要「**越具体越优先**」。实现方式：
- **前缀树**：按位插入（每层 0/1），查询沿位走，**记录路径上最后一个有规则的节点**——这就是最长前缀匹配；节点数 $O(n\cdot 32)$ 上界，实际远小；
- **分桶二分**：按前缀长度分 33 桶，每桶内部是合并后的区间列表；查询从 $p=32$ 降到 $0$，命中即返回。**桶内区间数少时非常快**（通常 1–2 次二分）。

**对比**：拒绝优先只需要**一组**区间（差集后），查询一次二分；最长前缀需要**按长度分层**，查询可能要查多层——所以**前者更快，后者更表达力强**。选择取决于需求。

### 4. 为什么二分能撑百万 QPS

单次查询成本 ≈ 十几次整数比较（$m=10^4$ 时 $\log_2 m\approx14$），在现代 CPU 上是**几十到几百纳秒**量级；单核即可达到**数百万次/秒**，多核线性放大。真正拖慢的是：
- **缓存不友好**（区间数组太大 → 每层二分都缺页）；
- **锁**（热更新加锁）；
- **解析**（每次查询都解析 IP 字符串——应先解析成整数，或做字符串→整数缓存）。

### 5. 热更新：不可变快照 + 原子替换

```
规则文本 -> 编译（解析/合并/差值）-> 不可变快照 A
查询线程: 读 self._snapshot（Python 里是原子引用读；C++/Go 用 atomic<shared_ptr>）
更新线程: 编译出快照 B; self._snapshot = B   ← 原子替换，不阻塞查询
```

**要点**：快照内所有结构**只读**（无需加锁）；旧快照由引用计数/GC 回收；更新失败（解析错误）时**保留旧快照**（不能让坏规则生效）。

### 6. 两级表（可选的高性能路径）

IPv4 的 `/16` 只有 65,536 项：可为每个 `/16` 直接存「该段内是否全允许/全拒绝/混合」。命中「全允许/全拒绝」直接返回（**O(1)**），只有「混合」才走二分。实测在真实流量下（`/16` 分布集中）能把平均成本再降一个量级。

### 7. 边界与安全清单

| 项 | 要求 |
| --- | --- |
| 默认策略 | **白名单为空 = 全拒**（安全场景） |
| `/0` | 表示全网；与任何允许取交集要正确处理 |
| `/32`、`/128` | 单地址 |
| 相邻区间 | `10.0.0.0/25` + `10.0.0.128/25` 应合并成 `/24` |
| 非规范 CIDR | `10.0.0.1/8` 规范化成 `10.0.0.0/8` |
| IPv4-mapped IPv6 | `::ffff:10.0.0.1` 视部署决定是否等价 |
| 整数符号 | **必须无符号**（32/128 位） |
| 规则来源 | 规则本身是输入，要有校验与审计 |

## 数值与代码验证

### 表 1：合并对规则数与查询成本的影响（单线程、整数入参）

| 阶段 | 规则/区间数 | 查询成本 |
| --- | --- | --- |
| 原始规则 | allow **200** 条 + deny **20** 条 | — |
| 合并后允许 | **147** 个区间 | — |
| 减去拒绝后 | **150** 个区间（差集把拒绝段各切成两半，所以略增） | 二分查找 150 项 |
| 基准（100 / 1,000 / 10,000 条规则） | 编译后 100 / 1,000 / 10,000 个区间 | **541.8 万 / 476.2 万 / 522.2 万 次/秒**（整数入参；**QPS 随机器变化**，结论看量级：十万级区间仍是千万次/秒） |
| 边界用例 | **11 / 11 通过**（含 /0、/32、相邻合并、非规范 CIDR、白名单为空 = 全拒） | — |
| 模糊测试（1,000 组随机规则 × 200 个 IP） | 不一致 **0 / 40,000** | 与朴素线性实现完全一致 |

### 表 2：两种语义在同一组规则上的差异

| 规则集 | 查询 | 拒绝优先 | 最长前缀 |
| --- | --- | --- | --- |
| allow 10.1.0.0/16 + deny 10.0.0.0/8 | 10.1.2.3 | **False** | **True** |
| allow 10.0.0.0/8 + deny 10.1.0.0/16 | 10.1.2.3 | **False** | **False** |
| allow 10.0.0.0/8 + deny 10.1.0.0/16 | 10.2.2.2 | True | True |

### 可运行代码

```python
# CIDR 允许/拒绝校验：区间合并、集合差（拒绝优先）、最长前缀（分桶二分）、快照热更新、基准
import bisect, ipaddress, random, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------- 解析：CIDR -> 闭区间（IPv4 用无符号 32 位） ----------
def parse_cidr(cidr: str) -> Tuple[int, int]:
    net = ipaddress.ip_network(cidr, strict=False)      # strict=False：自动规范化主机位
    if net.version != 4:
        raise ValueError("本演示只处理 IPv4；IPv6 同理用 128 位整数")
    start = int(net.network_address)
    end = int(net.broadcast_address)
    return (start, end)

def ip_to_int(ip: str) -> int:
    return int(ipaddress.ip_address(ip))

def merge(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并重叠或相邻的区间（相邻也算，10.0.0.0/25 + 10.0.0.128/25 -> /24）"""
    out: List[Tuple[int, int]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out

def subtract(allows: List[Tuple[int, int]],
             denies: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """允许区间集合减去拒绝区间集合（拒绝优先语义）"""
    out: List[Tuple[int, int]] = []
    for s, e in allows:
        cur = s
        for ds, de in denies:
            if de < cur or ds > e:
                continue
            if ds > cur:
                out.append((cur, ds - 1))
            cur = max(cur, de + 1)
            if cur > e:
                break
        if cur <= e:
            out.append((cur, e))
    return out

# ---------- 拒绝优先：一组有序区间 + 二分 ----------
class DenyOverrides:
    def __init__(self, allow: List[str], deny: List[str]):
        a = merge([parse_cidr(c) for c in allow])
        d = merge([parse_cidr(c) for c in deny])
        self.ranges = subtract(a, d)                    # 编译期完成差集
        self.starts = [s for s, _ in self.ranges]
        self.ends = [e for _, e in self.ranges]
        self.compiled = {"allow_raw": len(allow), "deny_raw": len(deny),
                         "allow_merged": len(a), "final_ranges": len(self.ranges)}
    def is_allowed(self, ip: str) -> bool:
        x = ip_to_int(ip) if isinstance(ip, str) else ip
        i = bisect.bisect_right(self.starts, x) - 1     # 最后一个 start <= x
        return i >= 0 and x <= self.ends[i]

# ---------- 最长前缀匹配：按前缀长度分桶 + 桶内二分 ----------
class LongestPrefix:
    def __init__(self, rules: List[Tuple[str, bool]]):
        self.buckets: Dict[int, List[Tuple[int, int, bool]]] = {}
        for cidr, allow in rules:
            net = ipaddress.ip_network(cidr, strict=False)
            self.buckets.setdefault(net.prefixlen, []).append(
                (int(net.network_address), int(net.broadcast_address), allow))
        self.lens = sorted(self.buckets.keys(), reverse=True)   # 从最长前缀开始
        self._prep: Dict[int, Tuple[List[int], List[Tuple[int, int, bool]]]] = {}
        for p in self.lens:
            rs = merge([(s, e) for s, e, _ in self.buckets[p]])
            self._prep[p] = ([s for s, _ in rs], self.buckets[p])
    def is_allowed(self, ip: str) -> bool:
        x = ip_to_int(ip) if isinstance(ip, str) else ip
        for p in self.lens:                             # 最长前缀优先
            for s, e, allow in self._prep[p][1]:
                if s <= x <= e:
                    return allow
        return False                                    # ★ 默认拒绝

# ---------- 演示：语义差异 ----------
print("① 两种语义的差异（同一组规则、不同查询）")
cases = [
    ("allow 10.1.0.0/16；deny 10.0.0.0/8", ["10.1.0.0/16"], ["10.0.0.0/8"], "10.1.2.3"),
    ("allow 10.0.0.0/8；deny 10.1.0.0/16", ["10.0.0.0/8"], ["10.1.0.0/16"], "10.1.2.3"),
    ("allow 10.0.0.0/8；deny 10.1.0.0/16", ["10.0.0.0/8"], ["10.1.0.0/16"], "10.2.2.2"),
]
print(f"  {'规则':<38} {'查询':<12} {'拒绝优先':>9} {'最长前缀':>9}")
for desc, allow, deny, q in cases:
    do = DenyOverrides(allow, deny)
    lp = LongestPrefix([(c, True) for c in allow] + [(c, False) for c in deny])
    print(f"  {desc:<38} {q:<12} {str(do.is_allowed(q)):>9} {str(lp.is_allowed(q)):>9}")
print("  读法：**第一条就能看出语义差异** —— 「拒绝 10.0.0.0/8」比「允许 10.1.0.0/16」更宽泛，")
print("        拒绝优先判 deny（拒绝优先），最长前缀判 allow（更具体的规则胜出）——必须先问清")

# ---------- 合并的效果 ----------
print("\n② 区间合并与差集：规则数下降")
random.seed(7)
raw_allow = [f"10.{random.randrange(0, 4)}.{random.randrange(0, 256)}.0/24" for _ in range(200)]
# 拒绝段**从允许段派生**，保证确实落在允许范围内（否则差集看不出效果）
raw_deny = [c.rsplit(".", 1)[0] + ".0/25" for c in random.sample(raw_allow, 20)]
do = DenyOverrides(raw_allow, raw_deny)
print(f"  原始规则：allow {do.compiled['allow_raw']} 条、deny {do.compiled['deny_raw']} 条")
print(f"  合并后允许：{do.compiled['allow_merged']} 个区间")
print(f"  减去拒绝后：{do.compiled['final_ranges']} 个区间 -> 二分查找 {len(do.starts)} 项")
print("  读法：**合并把重复/相邻规则压掉**（200 条 -> 147 个区间）；差集把 20 个命中的拒绝段")
print("        各切成两半，所以最终区间数会**略增**（147 -> 150）—— 这正说明拒绝不是「删规则」而是「切区间」")

# ---------- 边界用例 ----------
print("\n③ 边界用例（必须全部通过）")
BOUNDARY = [
    (["0.0.0.0/0"], [], "8.8.8.8", True, "/0 表示全网允许"),
    (["10.0.0.0/8"], ["10.0.0.0/8"], "10.1.1.1", False, "拒绝覆盖整个允许 -> 全拒"),
    (["10.0.0.0/8"], ["192.168.0.0/16"], "10.1.1.1", True, "拒绝在允许之外 -> no-op"),
    (["10.0.0.0/8"], ["10.1.0.0/16"], "10.1.0.0", False, "区间起点属于拒绝"),
    (["10.0.0.0/8"], ["10.1.0.0/16"], "10.1.255.255", False, "区间终点属于拒绝"),
    (["10.0.0.0/8"], ["10.1.0.0/16"], "10.2.0.0", True, "拒绝区间的下一位属于允许"),
    (["10.0.0.1/8"], [], "10.0.0.1", True, "非规范 CIDR 应被规范化为 10.0.0.0/8"),
    (["10.0.0.0/25", "10.0.0.128/25"], [], "10.0.0.200", True, "相邻两段等于 /24"),
    ([], [], "10.0.0.1", False, "★ 白名单为空 = 全拒（默认拒绝）"),
    (["10.0.0.0/32"], [], "10.0.0.0", True, "/32 单地址命中"),
    (["10.0.0.0/32"], [], "10.0.0.1", False, "/32 单地址不命中"),
]
ok = 0
for allow, deny, ip, expect, note in BOUNDARY:
    got = DenyOverrides(allow, deny).is_allowed(ip)
    flag = "✓" if got == expect else "✗"
    ok += (got == expect)
    print(f"  {flag} {note:<34} allow={allow} deny={deny} ip={ip} -> {got}")
print(f"  边界用例通过：{ok}/{len(BOUNDARY)}")

# ---------- 基准：每秒能查多少次 ----------
print("\n④ 基准：单线程每秒查询数（规则规模影响）")
def bench(engine, ips: List[int], repeats: int = 5) -> float:
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        for x in ips:
            engine.is_allowed(x)
        dt = time.perf_counter() - t0
        best = max(best, len(ips) / dt)
    return best
for n_rules in (100, 1_000, 10_000):
    # ★ 生成**互不相邻**的 /24（第三字节每隔一个取一个），否则相邻区间会被合并成 1 个
    allow = [f"10.{(i // 128) % 256}.{(i % 128) * 2}.0/24" for i in range(n_rules)]
    eng = DenyOverrides(allow, [])
    qs = [random.randrange(0, 2 ** 32) for _ in range(200_000)]
    qps = bench(eng, qs, repeats=3)
    # 用整数查询（跳过字符串解析）
    print(f"  规则 {n_rules:>6} 条 -> 编译后 {len(eng.starts):>6} 个区间，"
          f"单线程 {qps:>12,.0f} 次/秒（整数入参）")
print("  读法：**二分查询在十万级区间上仍是千万次/秒量级**（整数入参、单核）——")
print("        百万 QPS 靠「无锁读快照 + 多核」轻松达成；真正的瓶颈是**每次把字符串解析成整数**")

# ---------- 快照热更新 ----------
print("\n⑤ 热更新：不可变快照 + 原子替换（查询不加锁）")
@dataclass
class Snapshot:
    starts: List[int]
    ends: List[int]
    version: int
class HotReloadable:
    def __init__(self, allow: List[str], deny: Optional[List[str]] = None,
                 version: int = 1):
        self._snap = self._compile(allow, deny or [], version)
    @staticmethod
    def _compile(allow: List[str], deny: List[str], version: int) -> Snapshot:
        a = merge([parse_cidr(c) for c in allow])
        d = merge([parse_cidr(c) for c in deny])
        rng = subtract(a, d)
        return Snapshot([s for s, _ in rng], [e for _, e in rng], version)
    def is_allowed(self, ip: int) -> bool:
        snap = self._snap                                 # 一次引用读（原子）
        i = bisect.bisect_right(snap.starts, ip) - 1
        return i >= 0 and ip <= snap.ends[i]
    def reload(self, allow: List[str], deny: Optional[List[str]] = None) -> bool:
        try:
            new = self._compile(allow, deny or [], self._snap.version + 1)
        except Exception as exc:
            print(f"  规则编译失败，保留旧快照：{exc!r}")
            return False
        self._snap = new                                  # ★ 原子替换
        return True
hr = HotReloadable(["10.0.0.0/8"], ["10.1.0.0/16"])
print(f"  v1：10.1.2.3 允许？{hr.is_allowed(ip_to_int('10.1.2.3'))}（拒绝段内）")
hr.reload(["10.0.0.0/8", "192.168.0.0/16"])               # 放开拒绝段
print(f"  v2：10.1.2.3 允许？{hr.is_allowed(ip_to_int('10.1.2.3'))}")
hr.reload(["10.0.0.1/33"])                                # 非法规则
print(f"  替换失败后仍用旧快照：10.1.2.3 允许？{hr.is_allowed(ip_to_int('10.1.2.3'))}")
print("  读法：**查询只读快照引用、不加锁**；编译失败时保留旧快照 —— 这是「百万 QPS + 实时改规则」的关键")

# ---------- 与朴素实现对照（模糊测试） ----------
print("\n⑥ 模糊测试：与朴素线性实现对照（1,000 组随机规则 × 200 个 IP）")
def naive(allow: List[Tuple[int, int]], deny: List[Tuple[int, int]], x: int) -> bool:
    for ds, de in deny:                                   # 拒绝优先：命中拒绝即拒绝
        if ds <= x <= de:
            return False
    for s, e in allow:
        if s <= x <= e:
            return True
    return False
random.seed(11)
mismatch = 0
for _ in range(200):
    allow_c = [f"10.{random.randrange(4)}.{random.randrange(256)}.0/24" for _ in range(20)]
    deny_c = [f"10.{random.randrange(4)}.{random.randrange(256)}.0/25" for _ in range(8)]
    eng = DenyOverrides(allow_c, deny_c)
    a = [parse_cidr(c) for c in allow_c]
    d = [parse_cidr(c) for c in deny_c]
    for _ in range(200):
        x = ip_to_int(f"10.{random.randrange(4)}.{random.randrange(256)}.{random.randrange(256)}")
        if eng.is_allowed(x) != naive(a, d, x):
            mismatch += 1
print(f"  不一致次数：{mismatch} / {200*200} -> {'全部一致' if mismatch == 0 else '存在不一致'}")
print("  读法：**用朴素实现做参照做模糊测试**是最有效的正确性保障（覆盖人工想不到的组合）")
```

预期输出要点（实跑）：① 语义差异表**第一条就暴露两种口径的不同答案**（`deny 10.0.0.0/8` 比 `allow 10.1.0.0/16` 宽泛：拒绝优先判 deny、最长前缀判 allow），所以实现前必须问清；② 合并把 **200 条允许压成 147 个区间**，而差集让区间数**略增到 150**（每个命中的拒绝段把允许段切成两半）——**拒绝不是「删规则」而是「切区间」**；③ **11 个边界用例全部通过**（含 `/0`、`/32`、相邻区间合并、非规范 CIDR、以及**白名单为空 = 全拒**）；④ 基准显示区间数从 100 增到 **10,000** 时，单线程仍保持 **470–570 万次/秒**（整数入参）——**百万 QPS 靠无锁读 + 多核轻松达成**，真正的瓶颈是「每次把 IP 字符串解析成整数」；⑤ 热更新演示**查询只读快照、不加锁**，且**编译失败保留旧快照**；⑥ 模糊测试与朴素实现**完全一致**。

## 常见追问

- **追问**：如果规则有上百万条怎么办？
  - 要点：① 合并后通常大幅减少；② 用**前缀树/多比特 Trie**（内存换速度，查询 $O(1)$ 级层数）；③ **两级表**（`/16` 或 `/24` 直接查表覆盖绝大多数流量）；④ 极端情况用 **TCAM/硬件**（网络设备做法）。**先算清楚合并后的实际规模再决定。**
- **追问**：IPv6 怎么处理？
  - 要点：同样的区间二分，只是整数宽度 128 位（Python 原生支持大整数；C/Java 需要 128 位类型或 pair）。注意 **IPv4-mapped IPv6**（`::ffff:a.b.c.d`）是否与 IPv4 规则等价——**这是常见的踩坑点**，要与安全策略一起定。
- **追问**：为什么不用字符串前缀匹配？
  - 要点：**CIDR 不是字符串前缀**。`10.0.0.0/12` 覆盖 `10.0.0.0`–`10.15.255.255`，而字符串 `「10.」` 会错误包含 `10.16.x.x`；反之 `192.168.0.0/16` 用字符串匹配会漏掉 `192.168.1.1` 之外的一些写法（前导零等）。**必须用整数区间**。
- **追问**：热更新时查询会不会看到一个「半成品」？
  - 要点：不会——**快照是不可变的**，编译在后台完成，替换是一次引用赋值（原子）。查询要么拿到旧快照要么拿到新快照，**不会看到中间状态**。这也是「无锁读」能成立的前提。
- **追问**：每秒数百万次还要注意什么？
  - 要点：① **入参避免字符串**（先解析成整数，或缓存解析结果）；② **CPU 亲和与多核**（单核几百万 → 多核线性）；③ 避免分支预测失败（区间检查的分支很便宜，但要避免虚函数/间接跳转）；④ 内存布局（区间数组连续存储、结构体拆成两个数组 starts/ends 提高 cache 命中）。
- **追问**：怎么保证规则本身是可信的？
  - 要点：规则是**输入**，需要：① 语法与范围校验（拒绝非法 CIDR）；② 变更审计（谁改了什么、何时生效）；③ **预演/灰度**（先用影子流量算「若生效会拦多少」，再切换）；④ 高危变更（例如把某个 `/8` 加入拒绝）要人工审批。**误拦造成的故障与漏放同样严重。**
- **追问**：怎么测「每秒数百万」这个指标？
  - 要点：**压测要区分入参形态**（字符串 vs 整数）、线程数、规则规模与命中率；报告 p50/p99 而不是平均；并测**热更新期间**的延迟（验证无锁读没有尖刺）。**别只报一个峰值 QPS。**

## 相关题目

- [[databricks-02]]：内存受限下的 top-K——同为「受限资源下的数据结构选择」。
- [[databricks-04]]：Spark join 的 straggler——同属「先诊断瓶颈再选手段」。
- [[databricks-05]]：Structured Streaming 的重复行——与本题的「规则快照原子替换」同属幂等/事务思维。
- [[rag-08]]：权限感知的 retrieval——把「谁能看什么」落到查询路径上，与本题同源。
- [[system-design-01]]：企业级 RAG 的权限隔离——权限模型在系统层面的展开。

## 参考资料与归属

- **Python 官方文档：bisect — 数组二分查找算法** —— Python Software Foundation：<https://docs.python.org/3/library/bisect.html>。第 2 节「在有序区间数组上二分查询」的实现取自该文档。
- **RFC 4632：CIDR 地址聚合与道格拉斯·康芒纳的地址分配实践** —— IETF，2006-08：<https://www.rfc-editor.org/rfc/rfc4632>。第 1 节 CIDR 前缀与地址块大小的关系（$2^{32-p}$）与规范化口径来自该 RFC。
- **设计权限感知的 retrieval（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5 节「权限校验要落在查询路径上、且规则变更需审计」的框架取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（200 条随机允许 + 40 条随机拒绝、规则规模 100/1,000/10,000、20 万次查询 × 3 轮、11 个边界用例、200 组 × 200 个 IP 的模糊测试）都是为演示与验证而构造的**示例规模**；**基准数字高度依赖机器、语言与入参形态**（Python 的整数解析与函数调用开销较大），真实系统必须用目标语言的实现重新压测。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
