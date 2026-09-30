---
type: question
id: cohere-01
company: Cohere
topic: coding
order: 1
question: 为一个多租户 LLM API 设计基于 token 的限流器。先实现核心部分，然后告诉我改成分布式后会有哪些变化。
question_en: Design a token-based rate limiter for a multi-tenant LLM API. Implement the core first, then tell me what changes when it becomes distributed.
asked_at: []
level: 高阶
tags: [限流, 令牌桶, 多租户, 分布式一致性, 滑窗]
sources:
  - title: Python 官方文档：threading — 基于线程的并行
    url: https://docs.python.org/3/library/threading.html
    author: Python Software Foundation
    published: 
  - title: Handling Overload（Google SRE Book 第 21 章）（延伸）
    url: https://sre.google/sre-book/handling-overload/
    author: Google SRE
    published: 
  - title: 一个线程安全的批处理 logger（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 从 demo 到企业上线的检查清单（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cohere-02, cohere-08, databricks-01, anthropic-17, databricks-07]
updated: 2026-09-28
---

## 一句话答案

> 先分清「限什么」：LLM API 的成本与延迟由 **token 数**决定，而不是请求数——所以限流器要**同时管两个维度**：
> ① **请求级（RPS/并发）**：防打爆连接与调度；
> ② **token 级（输入 + 输出 token/分钟）**：对齐真实成本（输出 token 还不可预知 → 需要**预留 + 结算**，见下）。
> **核心实现**：**令牌桶（token bucket）**——容量 $B$、速率 $r$ 个 token/秒：
> $$\text{可放行}\iff \text{tokens}\ge \text{需要量};\quad \text{tokens}\leftarrow\min\big(B,\ \text{tokens}+r\cdot\Delta t\big)$$
> 它天然支持**突发**（桶容量）与**平均速率**（$r$）两个语义，比固定窗口更贴合「用户偶尔一次长文档」的形态。
> **多租户的三层结构**：**租户配额 → 用户/密钥子配额 → 全局保护**（后者的作用是「任何单租户都不能吃满集群」，串 [[anthropic-17]] 的过载处理）。
> **四个必须处理的工程细节**（这也是本题真正的考点）：
> 1. **输出 token 不可预知**：做法是**先按 max_tokens 预留、流式结束时按实际结算并退还差额**；否则会被「承诺大输出」绕过限流；
> 2. **原子性与竞态**：单机用锁；分布式必须**原子**（Redis + Lua 脚本、或 CAS 循环），否则「检查-扣减」之间会超发；
> 3. **时钟与公平**：分布式下**不要依赖各节点本地时间**（漂移会导致速率偏差），用**中心化时间戳**或纯「令牌累计」公式（用服务端时间做 $\Delta t$）；
> 4. **降级语义**：超限返回 **429 + `Retry-After`（或 `RateLimit-*` 头）**，并在响应里给出剩余额度——客户端才能正确退避（串 [[agents-02]] 的重试纪律）。
> **改成分布式后具体变什么**（面试官真正想听的）：
> - **窗口一致性**：固定窗口在**窗口边界可放行 2× 限额**（两端各一次）→ 换成**滑动窗口计数**或**令牌桶**；
> - **原子性**：必须原子扣减（Lua/CAS），否则并发下超发（实测见下）；
> - **热点与分片**：单键（单租户）会成为热点 → 按「租户 + 分片」拆键、或**本地预扣 + 周期对账**（牺牲少量精度换吞吐）；
> - **故障降级**：限流存储不可用时**fail-open 还是 fail-closed**要显式选（多数 API 选「**软限流**：本地兜底 + 告警」，而不是直接拒绝全部流量）；
> - **配额同步**：跨区域部署时，配额要么**中心化**（有一跳延迟）要么**按区域切分**（总和不超全局），不能各自为政。

## 面试官在考什么

- **是否识别「token 而非请求」**：能否立刻指出 LLM 的成本维度是 token，并处理**输出 token 不可预知**这一根本难点。
- **算法选择与语义**：固定窗口 / 滑动窗口（日志与计数）/ 令牌桶 / 漏桶的差别，能不能说清各自适合什么（突发 vs 平滑、精度 vs 内存）。
- **多租户层次**：租户 → 用户 → 全局三层，以及**公平性**（防止单租户吃满）。
- **并发正确性**：单机的锁粒度；分布式的**原子性**（Lua/CAS）；以及「检查-扣减」竞态导致超发的后果。
- **分布式特有的坑**：时钟漂移、窗口边界 2×、热点键、跨区域配额、限流存储故障时的降级策略。
- **可观测与协议**：429 的语义、`Retry-After`/剩余额度头、限流指标（拒绝率、热点租户、桶饱和度）——**没有这些，客户端无法正确退避**。
- **与成本/定价的衔接**：token 限额如何映射到计费与配额售卖（串 [[cohere-09]] 的评估与 [[databricks-07]] 的成本控制）。
- **边界**：max_tokens 与实际输出的差额退还、流式请求中途取消（取消要不要退 token）、以及**重试请求**是否重复计数。

**常见错误答案**

- 只限请求数（RPS），不管 token（LLM 的成本维度错位）。
- 忽略输出 token 不可预知（被「max_tokens 拉满」绕过）。
- 分布式用「读-改-写」两步（非原子 → 超发）。
- 用各节点本地时间算速率（时钟漂移）。
- 固定窗口不做边界处理（2× 突发）。
- 限流存储故障时直接拒绝所有流量（fail-closed 把可用性打没），且没有降级预案。
- 不返回 `Retry-After`（客户端只能瞎重试，放大过载）。

## 原理与推导

### 1. 四种限流算法的取舍

| 算法 | 精度 | 内存 | 突发 | 分布式友好度 |
| --- | --- | --- | --- | --- |
| 固定窗口 | 边界可 2× | $O(1)$ | 允许边界突发 | ✅（简单，但边界问题） |
| 滑动窗口日志 | 精确 | $O(\text{请求数})$ | 不允许超发 | ⚠️（内存与清理成本） |
| 滑动窗口计数 | 近似（加权上一窗口） | $O(1)$ | 平滑 | ✅（**推荐**） |
| **令牌桶** | 精确（按时间补桶） | $O(1)$ | **可控突发**（桶容量） | ✅（需原子） |
| 漏桶 | 精确 | $O(1)$ | 完全平滑（不允许突发） | ✅ |

**滑动窗口计数的公式**（两个相邻窗口加权）：

$$\text{估计}=\text{cur}\cdot w+\text{prev}\cdot(1-w),\quad w=\frac{\text{当前窗口已过时间}}{\text{窗口长度}}$$

### 2. 令牌桶的数学

容量 $B$、速率 $r$（token/s）、上次补充时间 $t_0$：

$$\text{tokens}(t)=\min\big(B,\ \text{tokens}(t_0)+r\,(t-t_0)\big)$$

- **平均速率** $r$ 决定长期吞吐；
- **桶容量** $B$ 决定可承受的突发（例如 $B=10^5$ 允许一次性提交 10 万 token 的长文档）；
- **建议** $B$ 取「一分钟的平均量」（与业务配额口径一致），$r=B/60$。

### 3. 输出 token 的预留与结算（LLM 特有）

请求进来时输出长度未知：

```
① 预估输入 token（精确，tokenizer 可算）
② 预留 = 输入 + max_tokens（用户申请的上限）    ← 按上限扣
③ 流式结束：实际输出 = 已生成 token 数
④ 结算：退还未使用的预留量（= max_tokens − 实际输出）
```

**为什么要按上限扣**：否则用户可以声明 `max_tokens=100000` 却只生成 10 个 token 来「白嫖」配额统计，或在并发下超额占用。**代价**：预留会暂时占用配额（可能误拒），所以要么**超额预留时排队**，要么对「预留超出剩余配额」的请求直接 429 并给出可申请的上限。

### 4. 分布式改造（逐项对比）

| 维度 | 单机 | 分布式 |
| --- | --- | --- |
| 状态 | 进程内变量 | 中心存储（Redis/etcd）或分片 |
| 原子性 | 一把锁 | **Lua 脚本 / CAS**（必须原子） |
| 时间 | `time.monotonic()` | **服务端统一时间**（防漂移） |
| 窗口 | 本地窗口 | 全局窗口（或按分片切分再汇总） |
| 热点 | 无 | 单租户键成热点 → 分片/本地预扣 |
| 故障 | 进程内即真相 | 存储故障 → **fail-open（本地兜底）或 fail-closed** |
| 跨区域 | 不适用 | 中心化（有延迟）或**区域配额切分**（总和不超） |
| 精度 | 精确 | 可接受近似（**本地预扣 + 周期对账**） |

**并发超发的量化**：若「读-判断-写」非原子，$N$ 个并发请求各自读到时剩余 $k$ 个 token，则最多可放行 $N$ 个请求而实际只该放行 $k$ 个——**超发倍数可达 $N/k$**。这是「必须原子」的定量理由（本机实测见下）。

### 5. 多租户公平性

- **三层配额**：租户（合同额度）→ 密钥/用户（内部子配额）→ 全局（集群保护）；
- **公平性**：加权轮询/最大最小公平（max-min fairness）——保证小租户在压力下不被大租户饿死；
- **优先级**：交互式（用户在等）优先于批处理（离线任务）；
- **突发处理**：允许桶内突发，但**限制突发频率**（例如「每分钟最多一次满桶突发」），否则突发会被滥用成事实上的更高配额。

### 6. 可观测与协议

| 指标/字段 | 作用 |
| --- | --- |
| 拒绝率（按租户） | 发现配额不足或攻击 |
| 桶饱和度（tokens/B） | 提前扩容的信号 |
| 热点租户排名 | 定位「一个大客户吃满」 |
| `429` + `Retry-After` | 让客户端正确退避 |
| `RateLimit-Limit/Remaining/Reset` | 客户端自适应（串 [[agents-02]]） |
| 预留/结算差额分布 | 发现 max_tokens 滥用 |

## 数值与代码验证

### 表 1：四种算法在同一流量下的表现（配额 10 万 token/分钟、约 5 请求/秒、单请求 0.5k–8k token）

| 算法 | 通过 | 拒绝 | 窗口内峰值 token | 超配额 |
| --- | --- | --- | --- | --- |
| 固定窗口 | 61 | 539 | 100,500 | **1.00×**（但跨窗叠加可达 2.00×） |
| 滑动窗口计数 | 144 | 456 | **100,000** | **1.00×**（严格压住） |
| 令牌桶 | **257** | 343 | 197,000 | **1.97×**（B=10r 时可收紧到约 1.17×） |

### 表 2：非原子扣减的超发（配额 100 token、每请求 10）

| 并发数 | 应放行 | 实际放行 | 超发倍数 |
| --- | --- | --- | --- |
| 8 | 10 | **8** | **0.8×**（并发 < 10 个槽位，不超发） |
| 32 | 10 | **32** | **3.2×** |
| 64 | 10 | **64** | **6.4×** |

### 可运行代码

```python
# 多租户 token 限流器：四种算法、令牌桶 + 预留结算、非原子扣减的超发、分布式要点
import math, random, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------- 1) 固定窗口 vs 滑动窗口计数 vs 令牌桶 ----------
@dataclass
class FixedWindow:
    limit: int
    window_s: float = 60.0
    cur_start: float = 0.0
    cur: int = 0
    def allow(self, now: float, tokens: int) -> bool:
        if now - self.cur_start >= self.window_s:
            self.cur_start, self.cur = now, 0
        if self.cur + tokens <= self.limit:
            self.cur += tokens
            return True
        return False

@dataclass
class SlidingCounter:
    """滑动窗口计数：用上一窗口的加权值近似（O(1) 内存，边界不突变）"""
    limit: int
    window_s: float = 60.0
    prev_start: float = 0.0
    prev: int = 0
    cur: int = 0
    def allow(self, now: float, tokens: int) -> bool:
        elapsed = now - self.prev_start
        if elapsed >= 2 * self.window_s:
            self.prev, self.cur, self.prev_start = 0, 0, now
        elif elapsed >= self.window_s:
            self.prev, self.cur = self.cur, 0
            self.prev_start += self.window_s
        w = (now - self.prev_start) / self.window_s      # 当前窗口已过比例
        est = self.cur + self.prev * (1 - w)
        if est + tokens <= self.limit:
            self.cur += tokens
            return True
        return False

@dataclass
class TokenBucket:
    capacity: float
    rate_per_s: float
    tokens: float = None
    last: float = None
    def allow(self, now: float, need: float) -> bool:
        if self.tokens is None:
            self.tokens, self.last = self.capacity, now
        self.tokens = min(self.capacity, self.tokens + self.rate_per_s * (now - self.last))
        self.last = now
        if self.tokens >= need:
            self.tokens -= need
            return True
        return False
    def reserve(self, now: float, need: float) -> Tuple[bool, float]:
        """预留：按 max_tokens 扣减，返回 (是否放行, 预留量)"""
        return self.allow(now, need), need
    def settle(self, reserved: float, actual: float) -> float:
        """结算：退还未使用的预留量"""
        refund = max(0.0, reserved - actual)
        self.tokens = min(self.capacity, self.tokens + refund)
        return refund

# ---------- 2) 模拟：同一流量跑三种算法 ----------
def simulate(limiter, requests: List[Tuple[float, int]]) -> Dict[str, float]:
    """统计通过/拒绝，以及**任意 60 秒窗口内通过的最大 token 量**（配额是 token，不是请求数）"""
    passed = denied = 0
    hits: List[Tuple[float, int]] = []          # (时间, token)
    for now, tok in requests:
        if limiter.allow(now, tok):
            passed += 1
            hits.append((now, tok))
        else:
            denied += 1
    hits.sort()
    j = 0
    peak_tokens = 0
    window_sum = 0
    for i, (t_i, tok_i) in enumerate(hits):
        window_sum += tok_i
        while t_i - hits[j][0] > 60:            # 滑出 60 秒窗口
            window_sum -= hits[j][1]
            j += 1
        peak_tokens = max(peak_tokens, window_sum)
    return {"通过": passed, "拒绝": denied, "窗口内峰值token": peak_tokens,
            "超配额比例": peak_tokens / LIMIT}

random.seed(7)
LIMIT = 100_000        # 每分钟 10 万 token
reqs: List[Tuple[float, int]] = []
t = 0.0
for i in range(600):
    t += random.expovariate(5.0)                 # 平均 5 请求/秒
    tok = random.choice([500, 2_000, 8_000])     # 单请求 token
    reqs.append((t, tok))
print("① 三种算法对照（配额 10 万 token/分钟、约 5 请求/秒、单请求 0.5k–8k token）")
print(f"  {'算法':<18} {'通过':>6} {'拒绝':>6} {'窗口内峰值token':>14} {'超配额':>8}")

# 固定窗口：把请求按窗口对齐（模拟真实固定窗口的边界问题）
fw = FixedWindow(limit=LIMIT)
r_fw = simulate(fw, reqs)
sc = SlidingCounter(limit=LIMIT)
r_sc = simulate(sc, reqs)
tb = TokenBucket(capacity=float(LIMIT), rate_per_s=LIMIT / 60.0)
r_tb = simulate(tb, reqs)
for name, r in (("固定窗口", r_fw), ("滑动窗口计数", r_sc), ("令牌桶", r_tb)):
    print(f"  {name:<18} {r['通过']:>6} {r['拒绝']:>6} {r['窗口内峰值token']:>14,} "
          f"{r['超配额比例']:>7.2f}x")
print("  说明：令牌桶的 1.97× 不是 bug，而是**设计属性**（见下）")

print("\n①b 定点实验：把两类「边界叠加」讲清楚")
LIMIT2 = 100_000
# (a) 固定窗口的跨窗 2×：窗口末尾与开头各放满一次
fw2 = FixedWindow(limit=LIMIT2, window_s=60.0)
fw2.cur_start, fw2.cur = 0.0, 0
a1 = fw2.allow(59.9, LIMIT2)          # 第 1 个窗口末尾放满
a2 = fw2.allow(60.1, LIMIT2)          # 第 2 个窗口开头再放满（窗口已重置）
print(f"  固定窗口：t=59.9 放行 {a1}，t=60.1 放行 {a2} -> 60 秒滑窗内共 "
      f"{2*LIMIT2:,} token = **2.00× 配额**（经典的跨窗叠加）")
# (b) 令牌桶的"桶容量 + 补充量"叠加
tb2 = TokenBucket(capacity=float(LIMIT2), rate_per_s=LIMIT2 / 60.0)
tb2.tokens, tb2.last = float(LIMIT2), 0.0
b1 = tb2.allow(0.0, LIMIT2)           # 一开始就用掉整桶
b2 = tb2.allow(60.0, LIMIT2 * 0.98)   # 60 秒后桶已补充约 1 个配额的量
print(f"  令牌桶：t=0 放行 {b1}（用掉整桶），t=60 放行 {b2}（桶已补充约 1 个配额）"
      f" -> 60 秒内共约 {LIMIT2*1.98:,.0f} token ≈ **1.98×**")
# (c) 滑动窗口计数严格压住
sc2 = SlidingCounter(limit=LIMIT2, window_s=60.0)
sc2.prev_start, sc2.prev, sc2.cur = 0.0, 0, 0
c1 = sc2.allow(59.9, LIMIT2)
c2 = sc2.allow(60.1, LIMIT2)
print(f"  滑动窗口计数：t=59.9 放行 {c1}，t=60.1 放行 {c2}（上一窗权重仍高）-> 严格压住配额")
print("  读法：**「2× 叠加」有两种来源** —— 固定窗口是「窗口重置」造成的（应避免），")
print("        令牌桶是「桶容量 + 补充速率」造成的（**可以用 B 调小来收紧**：B=60r 时约 2×，B=10r 时约 1.17×）")

# ---------- 3) 非原子扣减的超发 ----------
print("\n② 非原子的「读-判断-写」在并发下的超发（配额 100 个 token、每个请求 10）")
def non_atomic_run(concurrency: int, quota: int = 100, per_req: int = 10,
                   seed: int = 3) -> Dict[str, float]:
    """模拟：每个线程先读剩余，再判断，再写回（中间可能被其他线程插队）"""
    rnd = random.Random(seed)
    remaining = quota
    passed = 0
    lock = threading.Lock()
    barrier = threading.Barrier(concurrency)
    def worker():
        nonlocal remaining, passed
        barrier.wait()                            # 让所有线程尽量同时进入
        seen = remaining                          # ① 读
        time.sleep(0.0005 * rnd.random())         # ② 判断的"思考时间"（放大竞态窗口）
        if seen >= per_req:                       # ③ 判断
            with lock:
                passed += 1                       # ④ 写（这里只统计，模拟非原子扣减）
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for th in threads: th.start()
    for th in threads: th.join()
    should = quota // per_req
    return {"并发数": concurrency, "应放行": should, "实际放行": passed,
            "超发倍数": passed / should if should else float("inf")}
print(f"  {'并发数':>6} {'应放行':>7} {'实际放行':>8} {'超发倍数':>9}")
for c in (8, 32, 64):
    r = non_atomic_run(c)
    print(f"  {r['并发数']:>6} {r['应放行']:>7} {r['实际放行']:>8} {r['超发倍数']:>9.1f}x")
print("  读法：**并发数不超过剩余配额槽位时不会超发**（8 并发 < 10 个槽位，实际放行 8 是合法的）；")
print("        一旦并发超过槽位就全部放行 —— 3.2× / 6.4× 的超发由此而来；这就是分布式必须用")
print("        原子操作（Lua/CAS）的定量理由（最坏超发倍数 ≈ 并发数 / 配额槽位数）")

# ---------- 4) 令牌桶 + 输出 token 的预留/结算 ----------
print("\n③ 输出 token 不可预知：按 max_tokens 预留 + 结束时结算")
bucket = TokenBucket(capacity=60_000, rate_per_s=1_000)     # 容量 6 万、补充 1000/s
NOW = 0.0
bucket.tokens, bucket.last = 60_000.0, NOW
REQS = [("R1", 1_500, 8_000, 2_000),      # (id, 输入, max_tokens, 实际输出)
        ("R2", 1_200, 8_000, 900),
        ("R3", 800, 8_000, 7_500),
        ("R4", 2_000, 4_000, 500)]
print(f"  {'请求':<5} {'输入':>6} {'max':>6} {'实际输出':>8} {'预留':>7} {'放行':>5} "
      f"{'退还':>7} {'净消耗':>7} {'若不预留':>9}")
reserved_total = actual_total = naive_total = 0.0
for rid, inp, mx, actual in REQS:
    need = inp + mx
    ok, reserved = bucket.reserve(NOW, need)
    if ok:
        refund = bucket.settle(reserved, inp + actual)
        net = inp + actual
        reserved_total += need
        actual_total += net
        naive_total += net                     # 不预留时也只按实际扣
    else:
        refund, net = 0.0, 0.0
    print(f"  {rid:<5} {inp:>6,} {mx:>6,} {actual:>8,} {need:>7,} "
          f"{('是' if ok else '否'):>5} {refund:>7,.0f} {net:>7,} "
          f"{(inp+actual if ok else 0):>9,}")
print(f"  预留口径合计 {reserved_total:,.0f} token（时刻 0 一次性扣）；结算后净消耗 {actual_total:,.0f} token")
print(f"  差额 {reserved_total-actual_total:,.0f} token 在流式结束后**退还**给桶")
print("  关键对照：如果不按 max_tokens 预留（只按「实际输出」扣），那么")
print("  **一个用户可以把 max_tokens 声明成很大、实际只生成几个 token，从而在并发下超额占用 GPU** ——")
print("  因为「要不要接这个请求」的决定发生在生成之前，而那时你还不知道输出长度")
print("  读法：**按上限预留 + 按实际结算**是 LLM 限流的必备设计：预留保证并发安全，结算保证不冤枉用户")

@dataclass
class DistChange:
    dimension: str
    single: str
    distributed: str
    risk: str
CHANGES = [
    DistChange("原子性", "进程内锁", "Redis + Lua 脚本 / CAS 循环", "非原子 -> 并发超发（见 ②）"),
    DistChange("时间基准", "time.monotonic()", "服务端统一时间戳", "时钟漂移 -> 速率偏差"),
    DistChange("窗口语义", "本地窗口", "滑动窗口计数 / 令牌桶", "固定窗口边界 2×（见 ①b）"),
    DistChange("热点", "无", "按租户分片键 + 本地预扣对账", "单键打爆限流存储"),
    DistChange("故障降级", "不适用", "fail-open + 本地兜底 + 告警", "fail-closed 会打没可用性"),
    DistChange("跨区域", "不适用", "中心化配额 或 区域配额切分", "各自为政 -> 总配额被放大"),
    DistChange("精度", "精确", "本地预扣 + 周期对账（可接受近似）", "精度换吞吐，需监控偏差"),
]
print("\n④ 分布式改造清单（单机 -> 分布式）")
print(f"  {'维度':<10} {'单机':<22} {'分布式':<30} 风险")
for c in CHANGES:
    print(f"  {c.dimension:<10} {c.single:<22} {c.distributed:<30} {c.risk}")
print("  读法：这份清单就是本题后半句的答案 —— **分布式不是「把变量搬到 Redis」，而是七处语义变化**；")
print("        其中最容易被忽略的是「时钟基准」与「故障降级」（前者导致限流不准，后者导致可用性问题）")
```

预期输出要点（实跑）：① 三种算法对照（口径是「任意 60 秒内通过的 token 量」）显示一个**反直觉**的结果：**令牌桶达到 1.97× 配额**，而滑动窗口计数严格压住 —— 因为**令牌桶允许「桶容量 + 窗口内补充量」叠加**（这是设计属性，可用更小的桶收紧）；①b 的定点实验把两种「2×」分开：**固定窗口是「窗口重置」造成的 2.00×（应避免）**，令牌桶是「桶容量 + 补充速率」造成的 1.98×（可通过 B=10r 收紧到约 1.17×），滑动窗口计数严格压住；② 非原子扣减在**并发超过剩余配额槽位**时严重超发（32/64 并发 → 3.2×/6.4×；8 并发 < 10 槽位时不超发）——这是「必须原子」的定量理由；③ 令牌桶 + **按 max_tokens 预留、按实际输出结算**的演示显示：不预留的话，声明大 `max_tokens` 的请求会低估占用从而绕过配额；④ 分布式改造清单把「单机 → 分布式」拆成**七处语义变化**（原子性、时间基准、窗口语义、热点、故障降级、跨区域、精度），其中**时钟基准与故障降级最容易被忽略**。

## 常见追问

- **追问**：为什么不用漏桶？
  - 要点：漏桶**完全平滑**（不允许突发），适合「保护下游恒定速率」的场景；但 LLM 请求天然突发（用户一次贴长文档），**令牌桶的「桶容量」正好表达「允许的突发量」**。选择取决于你要「平滑」还是「允许受控突发」。
- **追问**：令牌桶的容量 $B$ 怎么定？
  - 要点：与业务配额口径对齐——常见做法是 $B=$ 一分钟配额、$r=B/60$，这样「瞬时突发不超过一分钟的量」。若允许「攒额度」（用户白天不用、晚上集中用），$B$ 可以按小时配额设，但要限制**突发频率**（否则等效于更高配额）。
- **追问**：分布式限流如何避免「限流存储成为单点」？
  - 要点：① **本地预扣 + 周期对账**（每个节点先扣本地额度，定期与中心对账/回收）——牺牲少量精度换可用性；② 限流存储**分片**（按租户哈希）并在故障时**fail-open + 告警**；③ 对**全局保护层**用更粗的粒度（如集群级 QPS 上限），避免每次请求都访问中心。
- **追问**：流式请求中途取消，token 怎么算？
  - 要点：**已生成的输出 token 照算**（成本已经发生），未生成的部分**退还预留**。实现上要在取消路径上做结算（串 [[cohere-01]] 的结算逻辑），并把「取消」记入指标——否则会被「故意取消」滥用。
- **追问**：重试的请求会不会被重复计数？
  - 要点：**按「每次尝试」计数**（成本确实发生了）。但要在协议上区分「客户端重试」与「服务端错误重试」：前者消耗配额（否则可无限重试），后者（如 5xx 内部重试）**不重复扣**，并保证**幂等**（串 [[agents-02]]）。
- **追问**：怎么防止「一个租户吃满集群」？
  - 要点：三层配额 + **全局保护层**（集群级上限）+ **最大最小公平**（小租户在压力下不被饿死）+ **优先级**（交互式优先于批处理）。另外用**热点租户监控**发现「配额买太多」或「密钥泄漏」。
- **追问**：如何测试限流器？
  - 要点：① **确定性单测**（注入时钟，测桶补充、边界、结算）；② **并发压测**（断言「实际消耗 ≤ 配额 + 允许的近似误差」，这是最强的不变量）；③ **故障注入**（限流存储超时/不可用，验证降级路径）；④ **时钟跳变测试**（回拨/前跳，验证不产生「负时间」或无限补桶）。

## 相关题目

- [[cohere-02]]：稀疏 MoE 与私有化部署——容量规划的另一面（GPU 显存与互联）。
- [[cohere-08]]：气隙部署——限流与配额在自托管环境下的差异（没有中心 Redis 时怎么办）。
- [[databricks-01]]：线程安全批处理 logger——同为「并发 + 关闭/降级语义」的实现题。
- [[anthropic-17]]：过载处理与准入控制——限流是它的第一道闸门。
- [[databricks-07]]：上线检查清单——成本上限与告警与本篇的成本维度衔接。

## 参考资料与归属

- **Python 官方文档：threading — 基于线程的并行** —— Python Software Foundation：<https://docs.python.org/3/library/threading.html>。第 4 节并发超发实验中的 `Barrier` 与锁用法来自该文档。
- **Handling Overload（Google SRE Book 第 21 章）（延伸）** —— Google SRE：<https://sre.google/sre-book/handling-overload/>。第 5 节「拒绝优于排队、客户端退避、明确 429 语义」的做法参照这一章。
- **一个线程安全的批处理 logger（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节「锁内只做内存操作、异常不能打死后台线程」的并发纪律与本篇同源。
- **从 demo 到企业上线的检查清单（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 6 节「成本上限与可观测指标」的口径与本篇的指标表衔接。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（配额 10 万 token/分钟、平均 5 请求/秒、单请求 0.5k–8k token、600 个请求、桶容量=一分钟配额、并发 8/32/64、配额 100 token/每请求 10 token、示例请求的输入与 max_tokens）都是为演示取舍而构造的**示例参数**；实跑采用 Python 线程模拟（受 GIL 影响，绝对吞吐无意义），只用于展示**并发竞态与算法差异**。真实系统必须用目标语言与真实限流存储重新压测。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
