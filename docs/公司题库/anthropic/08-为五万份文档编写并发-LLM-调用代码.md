---
type: question
id: anthropic-08
company: Anthropic
topic: coding
order: 8
question: 你需要对 50,000 份文档执行 LLM 调用。该 API 允许约 100 个并发请求，并且偶尔会返回 429 和超时。请写出 Python 代码。
question_en: You need to run an LLM call over 50,000 documents. The API allows ~100 concurrent requests and occasionally returns 429s and timeouts. Write the Python.
asked_at: []
level: 高阶
tags: [实现题, 并发, 429, 退避抖动, 断点续跑, 自适应并发]
sources:
  - title: Coroutines and Tasks（Python asyncio 文档）（延伸）
    url: https://docs.python.org/3/library/asyncio-task.html
    author: Python Software Foundation
    published: 
  - title: Synchronization Primitives（Python asyncio 文档）（延伸）
    url: https://docs.python.org/3/library/asyncio-sync.html
    author: Python Software Foundation
    published: 
  - title: Exponential Backoff And Jitter（延伸）
    url: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    author: Marc Brooker (AWS Architecture Blog)
    published: 2015-03-04
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
related: [anthropic-07, anthropic-09, anthropic-05, inference-serving-09, anthropic-03]
updated: 2026-09-28
---

## 一句话答案

> 正确写法由**六件事**组成，缺一件都会在 5 万这个量级上出事：
> ① **有界并发**：`asyncio.Semaphore(100)`（不要直接把 5 万个任务一次性 `gather`）。
> ② **错误分类**：429 / 5xx / 超时 / 连接重置 → 可重试；400 / 422 / 内容策略拒答 → **不可重试**（重试只会浪费配额）；超长文档 → 先切分或走另一条路径。
> ③ **指数退避 + full jitter**：$d_n=\min(d_{\max}, d_0 2^{n-1})$，实际等待 $U(0,d_n)$；429 若带 `Retry-After` 就听它的（可引用的口径来自 AWS 那篇：抖动的作用是打散重试、避免同步化）。
> ④ **超时分层**：连接超时、读取超时、以及整个任务的总预算，三者都要有（只设一个「总超时」会导致长文档永远失败）。
> ⑤ **断点续跑**：每完成一条就追加写结果（`jsonl`）+ 幂等键（文档 id + 内容哈希），重启时跳过已完成——**5 万条任务必然会被中断，这是设计前提而不是异常**。
> ⑥ **自适应并发**：遇到 429 就下调并发（例如 ×0.7），连续成功再缓慢上调（+1）；这比固定 100 更接近真实配额（串 [[anthropic-07]] 的分布式配额讨论）。
> 一句话原则：**把「并发上限」当成需要主动管理的资源，而不是一个写死的常数**。

## 面试官在考什么

- **是否一次性 `gather`**：这是最常见的失分点。5 万个协程同时创建会瞬间打满连接池与内存，且无法施加背压。
- **错误是否分类**：会不会把 400 也拿去重试；会不会对「内容策略拒答」重试 5 次浪费 5 倍配额（这在真实账单上很贵）。
- **退避是否带抖动**：只说「指数退避」不够，要说出**为什么需要抖动**（大量请求同时失败时，固定退避会让它们同时重试，把下游再次打爆）。
- **是否考虑成本与时间**：5 万份文档 × 平均 3k token 的输入，按每百万 token 计价能算出总成本与总时长量级；能报出量级说明真的想过规模化。
- **幂等与续跑**：能否说清「幂等键 = 文档 id + 内容哈希」以及「结果落盘用追加写（append-only）而不是覆盖写」（覆盖写在中途崩溃时会丢已完成结果）。
- **可观测性**：进度、成功率、重试次数、429 比例、p50/p99 延迟——这些指标缺一个，就无法判断「是 API 变慢了还是我的并发太高」。
- **是否提到优先级与分片**：5 万条里通常有关键文档与可延后文档；先跑关键的子集可以更早拿到可用结果（这也是产品判断）。

**常见错误答案**

- `asyncio.gather(*[call(doc) for doc in docs])`——无并发上限、无背压。
- 固定 `sleep(1)` 重试——重试同步化。
- 只 catch `Exception` 然后无脑重试——把不可重试错误也重试。
- 没有断点续跑——跑到 3 万条被中断，只能从头开始。
- 用固定并发 100 从头跑到尾——429 率上升后不降速，导致整体吞吐反而下降（见下表）。

## 原理与推导

### 1. 吞吐上限：先算清楚理论上限

由 Little's law：$\text{吞吐}=\text{并发}/\text{平均延迟}$。并发 100、单次延迟 2 s → 理论上限 **50 请求/秒**；5 万条 → 约 **1,000 s ≈ 17 分钟**（纯理想情况）。这给出了「该等多久」的基线，任何显著慢于它的实现都说明有额外开销（重试、限速、本地 CPU 瓶颈）。

**成本量级**：5 万 × 3k 输入 token = 1.5 亿输入 token；按 \$3/百万计 ≈ **\$450**（不含输出）。若把 400 类错误也重试 3 次，最坏情况成本 ×4 ≈ \$1,800——这就是「错误分类」值钱的地方。

### 2. 退避与抖动

| 策略 | 第 n 次等待 | 问题 |
| --- | --- | --- |
| 固定 1 s | 1 s | 重试同步化（雷群） |
| 指数 | $d_0 2^{n-1}$ | 后期等待过长；仍会同步 |
| 指数 + full jitter | $U(0,\ d_0 2^{n-1})$ | 推荐：期望等待减半，且天然错峰 |
| 尊重 `Retry-After` | 服务端指定 | 429 场景优先 |

**抖动为什么有效**：若无抖动，同一时刻失败的 $N$ 个请求会在同一时刻重试，形成周期性尖峰；抖动把重试时间摊平成均匀分布，下游看到的是平滑负载。

### 3. 自适应并发（AIMD 式）

```
on 429:        concurrency = max(1, floor(concurrency * 0.7))     # 乘性下降
on success×k:  concurrency = min(cap, concurrency + 1)            # 加性上升
```

这与 TCP 拥塞控制同构：**快速退让、缓慢试探**。效果是让并发自动停在「刚好不触发 429」的位置——比人工调参更稳，也能适应 API 配额的动态变化（例如你的组织配额被别人占用时）。

### 4. 断点续跑的正确形态

- **幂等键**：`id = hash(doc_id + content_hash)`；同一文档内容未变则永不重算。
- **结果落盘**：追加写 `jsonl`（append-only），每条带 `id/status/latency/attempts/usage`；崩溃后重新读取已完成集合。
- **写入原子性**：单行 `jsonl` 写入在多数文件系统上是原子的（小于 `PIPE_BUF` 时），更大的结果建议「先写临时文件再 rename」。
- **去重与重放**：重跑时跳过已成功的 id；对失败的用**独立的重试轮次**（第二轮只跑失败集合），避免每轮都从零开始。

### 5. 与并发相关的「数据变更」问题（面试官常追问）

任务执行期间源数据可能变化（文档被更新、删除，或新增）。处理原则：

- **快照语义**：任务开始时冻结输入清单（记录每条的 `content_hash`），保证结果可解释；
- **变更检测**：任务结束后对比源数据版本，对变化的条目**增量重跑**（而不是全量重跑）；
- **幂等写下游**：结果写下游时用 upsert（按幂等键），避免重复插入。

## 数值与代码验证

### 表 1：固定并发 vs 自适应并发（配额 25，429 与在途并发相关；500 条实测）

| 策略 | 耗时 | 成功 | 429 | **超配额调用** | 有效吞吐 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 固定 10（保守） | 0.62 s | **491/500** | 14 | **0** | 794 条/秒 | 稳定、零超配额，但浪费余量 |
| 固定 50 | 0.44 s | 457/500 | 691 | 914 | 1,041 条/秒 | 长期超配额，**丢 43 条** |
| 固定 100（激进） | **0.38 s** | 457/500 | 771 | **1,020** | **1,214 条/秒** | 最快，但丢 43 条且白烧 1,020 次超配额调用 |
| 自适应 4→100 | 1.31 s | **491/500** | 14 | **0** | 376 条/秒 | 成功率与保守策略并列最高、零超配额；收敛需要足够长的时间跨度 |

**读法（三个必须说出来结论）**：
1. **吞吐不是目标**——激进策略吞吐最高，代价是 8.6% 的文档最终失败（重试耗尽）与上千次无效调用；
2. **自适应不保证更快**——在本例这种「总量小、收敛慢」的场景里它反而更慢；它的价值在**长任务**（5 万条、跑十几分钟）上体现为「成功率不降、配额不浪费」；
3. **要在「成功率约束下的吞吐」这个目标上比较策略**——先定成功率下限（例如 99%），再在其中选吞吐最高的配置。

### 表 2：错误分类与重试策略

| 错误 | 可重试 | 退避 | 备注 |
| --- | --- | --- | --- |
| 429 Too Many Requests | 是 | 尊重 `Retry-After`，否则指数+抖动 | 触发降并发 |
| 5xx | 是 | 指数 + 抖动（上限 3 次） | 服务端故障 |
| 超时 / 连接重置 | 是 | 指数 + 抖动 | 需要分层超时 |
| 400 / 422 | **否** | — | 请求格式问题，重试无意义 |
| 内容策略拒答 | **否** | — | 记入 `rejected` 分类，不重试 |
| 输入超长 | 否（先处理） | — | 切分或换模型，不是重试 |

### 可运行代码

```python
# 5 万条文档 + 有界并发 + 错误分类 + 抖动退避 + 断点续跑 + 自适应并发（可离线跑）
import asyncio, hashlib, json, os, random, tempfile, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------- 假的 LLM API：注入 429 / 超时 / 不可重试错误 ----------
class FakeLLM:
    """429 与「在途并发」相关：超过配额并发才明显触发（这样自适应才有意义）"""
    def __init__(self, quota=25, base429=0.02, over429=0.75, r_timeout=0.05,
                 r_bad=0.01, latency=0.02, seed=11):
        self.quota, self.base429, self.over429 = quota, base429, over429
        self.r_timeout, self.r_bad, self.latency = r_timeout, r_bad, latency
        self.rnd = random.Random(seed)
        self.calls = 0
        self.inflight = 0
        self.over_quota_calls = 0
    async def complete(self, doc_id: str, text: str) -> str:
        self.calls += 1
        self.inflight += 1
        over = self.inflight > self.quota
        try:
            await asyncio.sleep(self.latency)
            r = self.rnd.random()
            p429 = self.over429 if over else self.base429
            if over:
                self.over_quota_calls += 1
            if r < p429:
                raise TooManyRequests(over_quota=over)
            if r < p429 + self.r_timeout:
                raise asyncio.TimeoutError()
            if r < p429 + self.r_timeout + self.r_bad:
                raise BadRequest("malformed document")
            return f"summary:{len(text)}"
        finally:
            self.inflight -= 1

class TooManyRequests(Exception):
    def __init__(self, retry_after: Optional[float] = None, over_quota: bool = False):
        self.retry_after = retry_after
        self.over_quota = over_quota          # 关键：区分「我超配额了」与「服务端基础限流噪声」

class BadRequest(Exception): pass
class ContentRejected(Exception): pass

@dataclass
class Stats:
    ok: int = 0; retried: int = 0; rejected: int = 0; bad: int = 0
    r429: int = 0; timeouts: int = 0
    concurrency: int = 0; peak: int = 0
    latencies: List[float] = field(default_factory=list)

class Runner:
    """有界并发 + 自适应并发 + 抖动退避 + 断点续跑"""
    def __init__(self, api: FakeLLM, out_path: str, concurrency: int = 10,
                 cap: int = 100, max_attempts: int = 4, timeout_s: float = 5.0):
        self.api, self.out_path = api, out_path
        self.concurrency, self.cap = concurrency, cap
        self.max_attempts, self.timeout_s = max_attempts, timeout_s
        self.stats = Stats()
        self._sem = asyncio.Semaphore(concurrency)
        self._done: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._success_streak = 0
        self._last_decrease = 0.0

    # ---------- 断点续跑：读取已完成 ----------
    def load_done(self) -> None:
        if not os.path.exists(self.out_path):
            return
        with open(self.out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") in {"ok", "rejected", "bad"}:
                    self._done[rec["id"]] = rec

    def _append(self, rec: dict) -> None:
        with open(self.out_path, "a", encoding="utf-8") as f:      # append-only，崩溃不丢已完成
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def key(doc_id: str, text: str) -> str:
        return hashlib.sha256(f"{doc_id}\x00{text}".encode()).hexdigest()[:16]

    async def _adjust(self, ok: bool, was_429: bool, min_c: int = 2,
                      decrease_cooldown: float = 0.05) -> None:
        """AIMD：乘性下降必须**按时间窗口限流**，否则 100 个在途请求同时 429 会把并发连乘到 1"""
        now = time.monotonic()
        if was_429:
            self._success_streak = 0
            if now - self._last_decrease >= decrease_cooldown:
                self._last_decrease = now
                # 只有「超配额」才说明是自己并发太高；基础限流噪声不该让我们降速
                if was_429 == "over":
                    new = max(min_c, int(self.concurrency * 0.7))
                    if new != self.concurrency:
                        self.concurrency = new
                        self._sem = asyncio.Semaphore(new)
        elif ok:
            self._success_streak += 1
            if self._success_streak >= 10 and self.concurrency < self.cap:
                self.concurrency += max(1, self.concurrency // 10)      # 加性增长（约 10%）
                self._sem = asyncio.Semaphore(self.concurrency)
                self._success_streak = 0

    async def one(self, doc_id: str, text: str) -> dict:
        k = self.key(doc_id, text)
        if k in self._done:
            return self._done[k]
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            was_429 = False                # False / "noise" / "over"
            try:
                async with self._sem:
                    self.stats.peak = max(self.stats.peak, self.concurrency)
                    out = await asyncio.wait_for(self.api.complete(doc_id, text), self.timeout_s)
                self.stats.latencies.append(time.perf_counter() - t0)
                rec = {"id": k, "doc": doc_id, "status": "ok", "attempts": attempt, "out": out}
                self._append(rec); self._done[k] = rec; self.stats.ok += 1
                await self._adjust(True, False)
                return rec
            except TooManyRequests as e:
                was_429 = "over" if e.over_quota else "noise"
                self.stats.r429 += 1
                if attempt == self.max_attempts:
                    break
                self.stats.retried += 1
                d = e.retry_after if e.retry_after is not None else min(2.0, 0.05 * 2 ** (attempt - 1))
                await asyncio.sleep(random.uniform(0, d))          # full jitter
            except asyncio.TimeoutError:
                self.stats.timeouts += 1
                if attempt == self.max_attempts:
                    break
                self.stats.retried += 1
                d = min(2.0, 0.05 * 2 ** (attempt - 1))
                await asyncio.sleep(random.uniform(0, d))
            except ContentRejected:
                rec = {"id": k, "doc": doc_id, "status": "rejected", "attempts": attempt}
                self._append(rec); self._done[k] = rec; self.stats.rejected += 1
                return rec                                              # 不重试
            except BadRequest:
                rec = {"id": k, "doc": doc_id, "status": "bad", "attempts": attempt}
                self._append(rec); self._done[k] = rec; self.stats.bad += 1
                return rec                                              # 不重试
            finally:
                await self._adjust(False, was_429)
        rec = {"id": k, "doc": doc_id, "status": "failed", "attempts": self.max_attempts}
        self._append(rec); self._done[k] = rec
        return rec

    async def run(self, docs: List[Tuple[str, str]]) -> Stats:
        self.load_done()
        todo = [(d, t) for d, t in docs if self.key(d, t) not in self._done]
        await asyncio.gather(*[asyncio.create_task(self.one(d, t)) for d, t in todo])
        return self.stats

# ---------- 演示：500 条（真实是 5 万，按比例缩放） ----------
tmp = tempfile.mkdtemp()
out = os.path.join(tmp, "results.jsonl")
DOCS = [(f"doc{i:04d}", f"content-{i}" * 20) for i in range(500)]

async def main():
    api = FakeLLM(quota=25, r_timeout=0.04, r_bad=0.01, latency=0.01)
    r = Runner(api, out, concurrency=10, cap=100)
    t0 = time.perf_counter()
    st = await r.run(DOCS)
    dt = time.perf_counter() - t0
    print(f"第一轮：成功 {st.ok}，429 {st.r429}，超时 {st.timeouts}，不可重试 {st.bad}，"
          f"重试 {st.retried}，耗时 {dt:.2f}s，最终并发 {r.concurrency}，"
          f"超配额调用 {api.over_quota_calls}/{api.calls}")
    # 断点续跑：重跑同一批（应全部命中缓存，不再调用 API）
    calls_before = api.calls
    st2 = await r.run(DOCS)
    print(f"第二轮（断点续跑）：新增 API 调用 {api.calls - calls_before} 次，"
          f"（结果文件 {sum(1 for _ in open(out, encoding='utf-8'))} 行）")
    # 追加 50 条新文档：只跑新增
    more = [(f"new{i:04d}", f"content-new-{i}") for i in range(50)]
    calls_before = api.calls
    await r.run(DOCS + more)
    print(f"第三轮（新增 50 条）：实际调用 {api.calls - calls_before} 次（只跑新增，幂等键生效）")

    # 并发策略对比：固定 100 vs 自适应
    print(f"\n{'策略':<18} {'耗时(s)':>8} {'成功':>5} {'429':>5} {'超配额调用':>10} "
          f"{'吞吐(条/秒)':>12} {'最终并发':>8}")
    for label, c0, cap in (("固定 10（保守）", 10, 10), ("固定 50", 50, 50),
                           ("固定 100（激进）", 100, 100), ("自适应 4→100", 4, 100)):
        api2 = FakeLLM(quota=25, r_timeout=0.04, r_bad=0.01, latency=0.01, seed=7)
        out2 = os.path.join(tmp, f"r-{label}.jsonl")
        rr = Runner(api2, out2, concurrency=c0, cap=cap)
        t0 = time.perf_counter(); s2 = await rr.run(DOCS); dt2 = time.perf_counter() - t0
        eff = s2.ok / dt2
        print(f"{label:<18} {dt2:>8.2f} {s2.ok:>5} {s2.r429:>5} {api2.over_quota_calls:>10} "
              f"{eff:>12.1f} {rr.concurrency:>8}")

asyncio.run(main())

# ---------- 5 万条的规模推算（不实际跑） ----------
N, TOK_PER_DOC, PRICE_PER_M, LAT_S = 50_000, 3_000, 3.0, 2.0
for conc in (20, 100):
    tput = conc / LAT_S
    print(f"\n推算（并发 {conc}、延迟 {LAT_S}s）：吞吐 {tput:.0f} 条/秒，"
          f"{N} 条需 {N/tput/60:.1f} 分钟；输入 token 成本 ≈ ${N*TOK_PER_DOC/1e6*PRICE_PER_M:.0f}")
```

预期输出要点（本机实跑）：第一轮在负载相关的 429/超时下把 500 条处理完（成功 495，超配额调用 0），**第二轮新增 API 调用为 0**（断点续跑生效），**第三轮只处理新增的 50 条**（幂等键生效；多出的少量调用来自重试）。策略对比给出表 1 的结论：最激进的固定并发最快但**丢 8.6% 的文档**并产生上千次超配额调用；自适应版本成功率最高且零超配额，但在本例这种短任务上更慢。两个实现细节值得单独强调：**AIMD 的乘性下降必须按时间窗口限流**（否则上百个在途请求同时收到 429 会把并发连乘到 1——这是我第一版代码的真实缺陷）；**必须区分「超配额 429」与「配额内基础限流噪声」**（对后者降速只会白白损失吞吐，这也是第二版代码的缺陷）。

## 常见追问

- **追问**：为什么不用线程池 + `requests`？
  - 要点：可以，但 5 万条任务的线程栈与上下文切换成本更高；异步在**高并发 I/O** 下内存占用低一个量级。若已有同步 SDK 且不想改，用 `asyncio.to_thread` + 有界线程池也可行，但要把线程池当成并发上限。
- **追问**：怎么处理「输入超长」的文档？
  - 要点：先分类——可切分的切分后汇总、不可切分的改走长上下文模型或人工；关键是**不要让它们进入重试循环**（重试永远不会成功）。
- **追问**：结果要写数据库，怎么做幂等？
  - 要点：按幂等键 upsert，并用批次事务；写库失败只重放该批（结果文件是唯一真相源，可重放）。
- **追问**：怎么知道「是 API 变慢还是我的并发太高」？
  - 要点：看四个指标的组合——429 比例、p99 延迟、成功吞吐、以及「本地队列等待时长」。429 上升 + 吞吐下降 → 并发过高；429 不变但延迟上升 → 服务端变慢（此时应降低并发而不是加大）。
- **追问**：配额是按组织共享的，怎么避免和其他任务抢？
  - 要点：把并发上限做成**可配置 + 动态**（读取配额服务的当前余量）、给不同任务分优先级与配额份额、并在 429 时优先让高优先级任务重试。
- **追问**：如果 5 万条里有 100 条一直失败怎么办？
  - 要点：把失败集合单独落盘并做**有限轮次**重试（例如第二轮只跑失败项，最多 3 轮）；仍未成功的进入人工队列并附带错误上下文——**不要无限重试**。

## 相关题目

- [[anthropic-07]]：限流器的算法与配额管理，是本题并发上限与自适应策略的基础。
- [[anthropic-09]]：并行化的设计空间（分片、幂等、合并），是本题的上位框架。
- [[anthropic-05]]：爬虫的异步改造与重试，与本题共享同一套并发控制与退避机制。
- [[inference-serving-09]]：在线服务侧的配额与调度治理，解释了 429 从何而来。
- [[anthropic-03]]：任务调度器的失败重试与并发上限，可对照两题的调度抽象。

## 参考资料与归属

- **Coroutines and Tasks（Python asyncio 文档）（延伸）** —— Python Software Foundation：<https://docs.python.org/3/library/asyncio-task.html>。第 1 节「任务组/取消/`wait_for` 超时」的用法来自这份文档。
- **Synchronization Primitives（Python asyncio 文档）（延伸）** —— Python Software Foundation：<https://docs.python.org/3/library/asyncio-sync.html>。第 1 节 `Semaphore` 限并发的实现依据来自这份文档。
- **Exponential Backoff And Jitter（延伸）** —— Marc Brooker (AWS Architecture Blog)，2015-03-04：<https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/>。第 2 节退避与抖动（含 full jitter 的动机：避免重试同步化）来自这篇。
- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。第 1 节「先算基线（吞吐/成本）再优化」的工程取向参照这份清单。
- **延伸来源说明**：表 1、表 2 的数值、以及可运行代码中的全部参数（429 10%、超时 6%、不可重试 1%、延迟 20 ms、并发 10→100、每文档 3k token、\$3/百万 token）都是按本仓库统一口径构造的工程算例与显式假设，用于演示机制；5 万条的推算见代码输出，不是任何来源的原文数字。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
