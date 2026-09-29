---
type: question
id: coding-08
topic: 编程与数据结构
order: 8
question: 编写一个针对 API 的异步批处理器，支持并发上限、带 jitter 的重试和错误隔离。
question_en: Write an async batcher for an API with a concurrency cap, jittered retries, and error isolation.
asked_at: [Anthropic, Perplexity]
level: 高阶
tags: [异步, 批处理, 重试, 并发]
sources:
  - title: Coroutines and Tasks（Python asyncio 文档）（延伸）
    url: https://docs.python.org/3/library/asyncio-task.html
    author: Python Software Foundation
    published: ""
  - title: Exponential Backoff And Jitter（延伸）
    url: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    author: Marc Brooker (AWS Architecture Blog)
    published: 2015-03-04
related: [agents-02, inference-serving-09, system-design-09, coding-07, evaluation-07]
updated: 2026-09-28
---

## 一句话答案

> 批处理器要同时交付四件事：按「大小阈值或时间窗口先到者」攒批，用 `Semaphore` 把**在途 API 调用**钉在 $N$ 以内，只对可重试错误做 $d_n = \min(d_{\max}, d_0 2^{n-1})$ 的指数退避、等待时间取 $U(0, d_n)$（full jitter），最后让每个子请求各自结算成 `Outcome`——一个 400 不能拖垮整批。
> 错误隔离不能靠 `TaskGroup`：它的语义是「一个子任务失败就取消同组其余任务」，本机 3.11 实测 5 个子任务里 1 个失败，另外 4 个全部被连带取消；正确做法是每个 batch 一个独立 `Task`、逐项 `set_result`。两个旋钮的实测代价：300 个请求（平均 5ms 到达）用 20ms 窗口是 68–69 批、平均批量 4.4、P99 32ms；换 100ms 窗口只剩 16–17 批（调用数少约 4 倍），平均批量 17.65–18.75，P99 涨到 112ms。

## 面试官在考什么

- **先定契约再写代码**：`handler(items)` 必须等长返回、顺序一致；抛异常 = 整批失败（网络/429/5xx），元素是异常 = 该子请求单独失败。契约不定死，错误隔离无从谈起。
- **攒批的两个触发条件**：批量到阈值、或窗口超时，先到者触发；并能说清窗口从**本批第一个请求**到达时开始计，不是每个请求各自计时。
- **并发上限约束的是什么**：同时在途的 API 调用数，不是待处理请求数。二者混为一谈的答案一上量就露馅。
- **重试是否分类且有预算**：429/5xx/超时可重试，4xx 参数错误重试无意义；`Retry-After` 优先于自己算的退避；总预算在提交时刻定死，退避加一次尝试塞不进预算就立刻失败而不是硬等。
- **错误隔离的机制而非口号**：说得出 `TaskGroup` 会连带取消、调用方取消不能让同批其他请求失败、部分失败只重试失败项。
- **背压的两道闸门**：队列（还没攒成批）与批槽位（已攒成批、在等信号量或在途）。只做一个，等待任务会无界增长吃光内存。

常见错误答案：

- 「`asyncio.gather(*calls, return_exceptions=True)` 就是错误隔离」。它只解决「不让异常中断聚合」，既不限并发，也没有重试、预算与背压。
- 「失败就重试三次、指数退避」。不问错误类别、不尊重 `Retry-After`、不设总预算，还有把退避 `sleep` 写在信号量里面、连并发额度一起睡掉的写法。

## 原理与推导

### 1. 契约

```python
async def handler(items: Sequence[T]) -> Sequence[R | BaseException]:
    """等长返回、顺序一致。抛异常 = 整批失败；元素是异常 = 该子请求失败。"""
```

三条必须写进文档的约定：① `len(返回值) == len(items)`，长度不符按不可重试的协议错误处理；② 抛异常与「返回异常元素」语义不同，前者整批按退避重试，后者只把失败项带进下一轮；③ `handler` 自己不做重试与限流，策略只存在于批处理器一处，否则退避会在两层叠乘。

### 2. 为什么要攒批

批处理把三类固定成本摊到更多请求上：一次 HTTP/RPC 往返、一次服务端调度、一次权重读取。最后一项是 GPU 场景的主因：decode 每步都要把整份权重从 HBM 读一遍，按 [[inference-serving-02]] 的口径（8B、上下文 1024），$B=1$ 的 step 是 4.82 ms、$B=8$ 是 5.10 ms——step 只涨 6%，吞吐接近 8 倍。这也解释了 decode 为什么带宽受限：bf16 下算术强度约 1 FLOP/byte，远低于 989 TFLOPs ÷ 3.35 TB/s ≈ 295 FLOPs/byte 的 roofline 拐点，批越大越接近拐点。

批量的上限不是拍出来的：LLaMA-3-70B 每 token KV cache 320 KiB（GQA 8 个 KV 头、head_dim 128、80 层、bf16），32 条 8k 上下文就是 $32 \times 8192 \times 320\ \text{KiB} = 80\ \text{GiB}$，单张 80 GB 卡已经放不下。客户端攒批要为服务端的显存与排队留余量，而不是把 `max_batch_size` 顶到 API 上限。

### 3. 攒批结构

收集器只做一件事：从有界队列取第一个请求，然后要么在窗口内攒到 `max_batch_size`，要么等窗口超时，谁先到就发车。窗口必须兜底，否则低峰期一个孤零零的请求永远等不到同批伙伴。攒好的批交给独立 `Task`（`create_task`）执行，调用方取消与超时都不会传染；批任务再用 `asyncio.Semaphore(N)` 圈住真正的在途调用，退避 `sleep` 放在信号量**外面**，否则重试会把并发额度一起睡掉。

### 4. 退避与 jitter

第 $n$ 次失败后的基准等待 $d_n = \min(d_{\max}, d_0 2^{n-1})$，实际等待按抖动模式取：无抖动 $d_n$、等抖动 $d_n/2 + U(0, d_n/2)$、全抖动 $U(0, d_n)$。无抖动的问题是同步重试：所有客户端在同一失败时刻用同一公式算出同一个 $d_n$，于是同一时刻再次撞向服务端，形成周期性惊群与二次拥塞；抖动把这一批重试摊平成近似恒定的到达率。AWS 那篇的模拟口径（OCC 竞争、100 个客户端、网络延迟均值 10 ms 方差 4 ms）给出的结论是：加抖动后调用次数减少一半以上、完成时间显著改善，Equal Jitter 略差于 Full Jitter。

预算要同时约束三个量：$\text{总预算} \ge \text{排队窗口} + \sum_{n<A} d_n + A \times \text{单次尝试超时}$。$d_0=0.05$、$d_{\max}=2$、$A=4$ 时退避合计只有 $0.05+0.1+0.2=0.35$ s，单次尝试超时 5 s 却意味着最坏 20 s——所以超时必须逐层递减（总预算 → 单次尝试 → 连接），每次退避前检查剩余预算。成功率很直接：单次成功概率 $p$ 时期望尝试 $1/p$；单次失败率 0.3、允许 4 次时整次成功率 $1 - 0.3^4 = 99.19\%$。

### 5. 错误隔离

- 每个子请求带自己的 `Future` 与总预算，结算前先看 `fut.done()`：调用方已取消/超时就丢弃结果，批任务继续跑完（在途请求收不回来，这是异步批处理的固有代价）。
- 部分失败只重试失败项：一轮结算后 `pending` 收窄成失败子集，下一轮批更小、更便宜；代价是重试项与已成功项解耦——子请求之间若有顺序依赖（后一项要吃前一项的结果），只能由调用方在 `handler` 外面自己串起来，批处理器只保证同一批内互不影响。
- 熔断与重试预算要一起用：连续失败到阈值就快速失败一段时间，否则每个请求各自重试到预算耗尽，正好在下游故障期间把它再打一遍。加上熔断器与 `overflow` 策略的变体实测（同机同参数）：40 个请求连续失败后的错误分布是 `{ServerError: 8, CircuitOpen: 32}`，下游总共只被调用 6 次；队列上限 4、批槽位 2 时，`raise` 策略接纳 4 个、以 `QueueFull` 拒绝 36 个，`block` 策略全部接纳但墙钟 503–507 ms（10 批 ÷ 并发 2 × 100 ms 的理论下限是 500 ms）。
- 复杂度：提交与结算摊还 $O(1)$；内存上界 $O(\text{queue\_limit} + \text{批槽位} \times \text{max\_batch\_size})$；期望调用次数 $O(1/p)$，最坏 $O(A)$。

### 6. 生产化的差距

这段代码不能直接上线，差在五处：① 超时预算要逐层递减并把剩余预算下传（[[agents-09]] 的时间预算口径）；② 限流与重试是一个闭环——退避参数必须和 [[coding-07]] 的令牌桶对齐，否则「重试预算」只是自我安慰；③ 可观测性：每批大小分布、成功率、重试率、P99 延迟、`Outcome.attempts` 直方图都要进 trace（[[evaluation-07]]）；④ 公平性：FIFO 加固定窗口会让老请求在大批到来时被反复挤到下一轮，需要按到达时间加权或限制单请求被跳过的次数；⑤ 跨进程与跨机器：本地攒批只能合并同一实例的请求，真正的大 batch 要靠上层聚合（[[system-design-09]] 的 gateway 做请求收集与路由），并且要处理「一个 400 把整批带走」的坏苹果问题——生产上用二分拆分定位并隔离它。

## 数值与代码验证

参考实现（下面这份就是实测所用的完整文件，Python 3.11.0rc1 与 3.10.12 都跑过；3.10 走 `wait_for` 兼容分支）：

```python
"""coding-08 参考实现：攒批 + 并发上限 + full-jitter 重试 + 错误隔离（本文实测所用版本）。"""
import asyncio, random, sys, time
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence
class ApiError(Exception):
    """下游错误基类：retryable 决定是否重试，retry_after 来自 Retry-After 头。"""
    retryable, status, retry_after = False, None, None
    def __init__(self, msg="", *, retryable=False, status=None, retry_after=None):
        super().__init__(msg)
        self.retryable, self.status, self.retry_after = retryable, status, retry_after
class BadRequest(ApiError):
    """4xx：重试一百次还是同样的错误。"""
class RateLimited(ApiError):
    def __init__(self, msg="rate limited", *, retry_after=None):
        super().__init__(msg, retryable=True, status=429, retry_after=retry_after)
class ServerError(ApiError):
    def __init__(self, msg="server error", *, status=503):
        super().__init__(msg, retryable=True, status=status)
class Outcome(NamedTuple):
    """单个子请求的独立结果：成功值或异常，加尝试次数与这段耗时。"""
    ok: bool
    value: Any = None
    error: Optional[BaseException] = None
    attempts: int = 0
    latency: float = 0.0
def backoff_delay(attempt, *, base=0.05, cap=2.0, jitter="full", rng=None, retry_after=None):
    """d_n = min(cap, base*2^(n-1))：none = d_n，equal = d_n/2+U(0,d_n/2)，full = U(0,d_n)。"""
    if retry_after is not None:
        return max(0.0, float(retry_after))          # 服务端说了等多久，不要再叠随机
    d = min(cap, base * 2.0 ** (attempt - 1))
    if jitter == "none":
        return d
    if jitter == "equal":
        return d / 2 + rng.uniform(0, d / 2)
    if jitter == "full":
        return rng.uniform(0, d)
    raise ValueError(jitter)
class AsyncBatcher:
    """把窗口内的请求攒成 batch；handler 的契约见 _run_batch 的注释。"""
    def __init__(self, handler: Callable[[Sequence], Awaitable[Sequence]], *,
                 max_batch_size=32, max_wait_ms=20.0, max_concurrency=4, max_attempts=4,
                 base_delay=0.05, max_delay=2.0, jitter="full", request_timeout=10.0,
                 attempt_timeout=5.0, queue_limit=256, inflight_batches=None,
                 rng=None, sleep=asyncio.sleep, clock=time.monotonic):
        self.handler = handler
        self.max_batch_size, self.max_wait = max_batch_size, max_wait_ms / 1000
        self.max_concurrency, self.max_attempts = max_concurrency, max_attempts
        self.base_delay, self.max_delay, self.jitter = base_delay, max_delay, jitter
        self.request_timeout, self.attempt_timeout = request_timeout, attempt_timeout
        self._rng, self._sleep, self._clock = rng or random.Random(0), sleep, clock
        self._sem = asyncio.Semaphore(max_concurrency)                  # 在途调用上限
        self._pending = asyncio.Queue(maxsize=queue_limit)              # 背压闸门①
        self._slots = asyncio.Semaphore(inflight_batches or 2 * max_concurrency)  # 闸门②
        self._tasks, self._collector, self._closed = set(), None, False
    def start(self):
        if self._collector is None:
            self._collector = asyncio.create_task(self._collect())
    async def __aenter__(self):
        self.start()
        return self
    async def __aexit__(self, *exc):
        self._closed = True
        await self._pending.put(None)                # 唤醒可能卡在窗口等待的收集器
        await asyncio.gather(self._collector, return_exceptions=True)
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
    async def try_call(self, item) -> Outcome:
        if self._closed:
            raise RuntimeError("batcher is closed")
        self.start()
        fut = asyncio.get_running_loop().create_future()
        req = (item, fut, self._clock() + self.request_timeout)   # 总预算在提交时刻定死
        await self._pending.put(req)                 # 队列满则挂起 —— 这就是背压
        return await fut
    async def _collect(self):
        """大小阈值与时间窗口先到者触发；窗口从本批第一个请求到达时开始计。"""
        while True:
            first = await self._pending.get()
            if first is None:
                return
            batch, deadline = [first], self._clock() + self.max_wait
            while len(batch) < self.max_batch_size:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._pending.get(), remaining)
                except (asyncio.TimeoutError, TimeoutError):
                    break                            # 攒不满也要发，否则永远等下去
                if nxt is None:
                    await self._spawn(batch)
                    return
                batch.append(nxt)
            await self._spawn(batch)
    async def _spawn(self, batch):
        await self._slots.acquire()                  # 批槽位满则收集器挂起（背压）
        task = asyncio.create_task(self._run_batch(batch))   # 独立 Task：调用方取消不传染
        self._tasks.add(task)
        task.add_done_callback(lambda t: (self._tasks.discard(t), self._slots.release()))
    async def _run_batch(self, batch):
        """handler(items) 返回等长序列：元素是异常 = 该子请求失败，其余 = 成功值。"""
        pending, attempts, t0 = list(batch), 0, self._clock()
        deadline, last_err = min(r[2] for r in batch), None
        while pending:
            budget = deadline - self._clock()
            if budget <= 0 or attempts >= self.max_attempts:
                break
            attempts += 1
            err, values = None, None
            async with self._sem:                    # 只圈住在途调用，退避睡在信号量外
                try:
                    items = [r[0] for r in pending]
                    timeout = min(self.attempt_timeout, budget)
                    if sys.version_info >= (3, 11):              # 3.11+：asyncio.timeout
                        async with asyncio.timeout(timeout):
                            values = await self.handler(items)
                    else:                                       # 3.10 兼容：wait_for
                        values = await asyncio.wait_for(self.handler(items), timeout)
                except asyncio.CancelledError:
                    raise
                except (TimeoutError, asyncio.TimeoutError, OSError) as exc:
                    err = ServerError(f"transport: {exc!r}")     # 超时/连接错误必须可重试
                except BaseException as exc:                     # 整批失败（429 / 5xx / 4xx）
                    err = exc
            if err is None and (values is None or len(values) != len(pending)):
                err = BadRequest("handler returned wrong length")
            if err is None:
                pending, last_err = self._settle(pending, values, attempts, t0)
                if not pending:
                    return                               # 全部落袋
            else:
                last_err = err
                if not getattr(err, "retryable", False):
                    break                                # 4xx：重试无意义
            if attempts >= self.max_attempts:
                break                                    # 没有下一次机会，不必再睡
            delay = backoff_delay(attempts, base=self.base_delay, cap=self.max_delay,
                                  jitter=self.jitter, rng=self._rng,
                                  retry_after=getattr(last_err, "retry_after", None))
            if self._clock() + delay >= deadline:
                break                                    # 预算不够，把剩余时间还给调用方
            await self._sleep(delay)
        last_err = last_err or TimeoutError("request budget exhausted")  # 首轮之前预算就耗尽
        for item, fut, _ in pending:                     # 剩下的以最后一个错误结算
            self._finish(fut, Outcome(False, error=last_err, attempts=attempts,
                                      latency=self._clock() - t0))
    def _settle(self, pending, values, attempts, t0):
        """结算一轮：可重试的单项失败留到下一轮（只重试失败项），其余各自落袋。"""
        retry, last_err = [], None
        for (item, fut, dl), val in zip(pending, values):
            if isinstance(val, BaseException) and getattr(val, "retryable", False):
                retry.append((item, fut, dl))
                last_err = val
                continue
            bad = isinstance(val, BaseException)
            self._finish(fut, Outcome(not bad, value=None if bad else val,
                                      error=val if bad else None,
                                      attempts=attempts, latency=self._clock() - t0))
        return retry, last_err
    def _finish(self, fut, outcome):
        if fut.done():                    # 调用方已超时/取消：丢弃结果，批任务继续跑完
            return
        fut.set_result(outcome)
```

验证用的假 API（记录调用次数、下游自己观测到的在途峰值与每批大小；按调用序号注入故障）：

```python
class FakeApi:
    def __init__(self, latency=0.02, fail_plan=None):
        self.latency, self.fail_plan = latency, dict(fail_plan or {})
        self.calls = self.inflight = self.peak = 0
        self.batch_sizes = []
    async def __call__(self, items):
        self.calls += 1
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)     # 下游自己数在途峰值，比自计更可信
        self.batch_sizes.append(len(items))
        try:
            await asyncio.sleep(self.latency)
            plan = self.fail_plan.get(self.calls)     # 注入 429 / 5xx / 超时
            if plan is not None:
                raise plan() if isinstance(plan, type) else plan
            return list(items)
        finally:
            self.inflight -= 1
```

① **并发上限严格生效**（200 个请求、`max_batch_size=8`、下游延迟 20ms；3.11 上 20 多次重跑的区间）：

| `max_concurrency` | 批次数 | 下游在途峰值 | 最大批量 | 墙钟 ms |
| --- | --- | --- | --- | --- |
| 1 | 25 | 1 | 8 | 507–524 |
| 4 | 25–26 | 4 | 8 | 144–158 |
| 16 | 25–26 | 16 | 8 | 42–51 |

断言 `api.peak <= max_concurrency` 与 `sum(batch_sizes) == 200`（无丢失、无重复）全部通过；3.10.12 上同一份代码（走 `wait_for` 分支）实测区间与 3.11 重叠（13 次重跑为 507–510 / 145–148 / 44–45 ms），差异主要来自机器负载与频率，而不是 3.11 的 `asyncio.timeout` 分支。批次数稳定在 25（负载高时偶发 26）、与并发上限无关，说明攒批行为与并发闸解耦；墙钟大致按 $1/N$ 下降——25 批在 $N=1/4/16$ 下分别要 25/7/2 轮，与 507/144/42 ms 的量级一致。

② **批大小分布与延迟取舍**（300 个请求、串行到达、间隔 $\sim\text{Exp}$(均值 5ms)、`max_batch_size=64`、到达序列用 `random.Random(7)` 固定；3.11 与 3.10 各 5 次重跑）：

| `max_wait` | 批次数 | 平均批量 | P50 延迟 ms | P99 延迟 ms | 最大批量 |
| --- | --- | --- | --- | --- | --- |
| 20ms | 68–69 | 4.35–4.41 | 22.8–23.7 | 31.7–32.1 | 9–11 |
| 100ms | 16–17 | 17.65–18.75 | 64.9–67.9 | 111.3–111.8 | 27–28 |

延迟是调用方视角的端到端耗时（含背压等待）。100ms 窗口把 API 调用数压到 1/4 左右，代价是 P50 涨约 2.9 倍、P99 涨约 3.5 倍；批次数在 68 与 69、16 与 17 之间跳是窗口边界效应——最后几个请求落进上一批还是另起一批，正好差一批。突发场景是另一个极端：500 个请求同时提交、`max_batch_size=32` 时是 16 批、平均 31.2，前 15 批全是满的 32。

③ **退避符合公式**（`max_attempts=5`、$d_0=0.05$s、$d_{\max}=0.5$s，下游一直 503）：

```console
    实测等待序列 [0.05, 0.1, 0.2, 0.4]
    理论 d_n=min(0.5, 0.05*2^(n-1)) [0.05, 0.1, 0.2, 0.4]   总尝试 = 5   最终错误 = ServerError
    429 + Retry-After=0.2 -> 实测等待 0.2000s（覆盖 full jitter），第 2 次成功 = True
    full jitter 采样 400 次： 均值/d_n = 0.4992 / 0.5053 / 0.4945 / 0.5179（n=1..4）
                             最大/d_n = 0.9944 / 0.9936 / 0.9996 / 0.9993
    单次调用读超时（attempt_timeout=0.02）-> ok=True, attempts=2
```

full jitter 的实测均值落在 $d_n/2$ 的 ±4% 内、上界不超过 $d_n$，与 $U(0,d_n)$ 一致。最后一个边界是易漏项：单次调用读超时被映射成可重试错误后重试成功（`attempts=2`）——超时若不映射，最该重试的一类错误反而不重试。

④ **jitter 实证**：100 个客户端同时收到 429，服务端是漏桶（容量 5 req/s、桶深 5），$d_0=0.5$s、$d_{\max}=8$s，200 个随机种子取均值：

| 抖动 | 总尝试次数 | 其中 429 浪费 | 全部成功耗时 s | 首轮重试 std | 100ms 窗口峰值到达 |
| --- | --- | --- | --- | --- | --- |
| none | 1107.0 | 1007.0 | 135.50 | 0.000 | 95.0 |
| equal | 553.9 | 453.9 | 25.69 | 0.072 | 46.0 |
| full | 612.8 | 512.8 | 22.36 | 0.143 | 26.9 |

首轮重试时间的标准差与「区间宽度 $/\sqrt{12}$」吻合（equal 0.072、full 0.144），无抖动时恒为 0——这就是惊群的来源：95 个客户端在同一时刻重试。加抖动后总尝试从 1107 降到 554–613、完成时间从 135.5s 降到 22–26s、瞬时峰值从 95 降到 27–46。一处与 AWS 口径的差异要如实说明：在**立即拒绝型**漏桶里，full jitter 那些接近 0 的等待会落在令牌为 0 的时刻，总尝试数略高于 equal（613 vs 554），但完成时间最短、拥塞峰值最低（26.9 vs 46）。AWS 的模拟是 OCC 竞争模型（每轮只有一个写者成功），结论是 Full Jitter 比 Equal 做的功更少；模型不同、方向一致：抖动远好于不抖动，full 与 equal 的取舍取决于限流器是「排队」还是「立即拒绝」。

⑤ **错误隔离**（一个必然 400 的坏请求 + 一个先 429 后成功的请求 + 8 个正常请求）：

```console
      bad: ok=False attempts=1 error=BadRequest
    flaky: ok=True  attempts=2 value=ok:flaky
       r0..r7: ok=True  attempts=1
    两轮的批内容 = [['bad', 'flaky', 'r0' ... 'r7'], ['flaky']]
    调用方取消：canceled=1，同批其余 = ['ok:1', 'ok:2', 'ok:3']，被取消的 future 已取消=True
```

坏请求只失败自己且 `attempts=1`（4xx 不重试）；429 那一项第二轮单独重试（`attempts=2`，第二轮批里只有它）；8 个正常请求不受影响；调用方取消 1 个后同批另外 3 个照常返回。对照实验：把 5 个子任务直接放进 `asyncio.TaskGroup`、让第 3 个在 20ms 抛 `BadRequest`，结果是成功 `[]`、被连带取消 `[0, 1, 3, 4]`——同组兄弟全部被取消，这正是不能用 `TaskGroup` 承载「每个子请求独立」语义的原因。

## 常见追问

- **追问**：为什么用 full jitter，而不是固定抖动或等抖动？
  - 要点：目标是打散同步重试的到达时刻，full 把等待摊在整个 $[0,d_n]$ 上，去同步最快；等抖动保留 $d_n/2$ 下限，在立即拒绝型限流器下浪费的调用略少但完成更慢（实测 25.69s vs 22.36s）。固定抖动（$d_n$ 上加一点随机）不解决问题，客户端仍在同一量级时刻到达。
- **追问**：批量大小与延迟怎么权衡？
  - 要点：批越大摊薄越多、调用数越少，但每个请求要多等一个窗口（实测 P50 23ms → 65–68ms、P99 32ms → 112ms）。这是吞吐-尾延迟曲线，不能两全；把窗口绑到调用方的延迟预算上（预算 200ms 时窗口不超过 50ms），高峰再按队列深度自适应放大。
- **追问**：如何避免重试风暴？
  - 要点：三件事一起做——重试预算（次数加总时限，且逐层递减，3 层各 3 次就是 27 倍放大）、熔断（实测 40 个请求只放 6 次到下游）、jitter。再叠加服务端的 `Retry-After`，它比任何客户端公式都准。
- **追问**：批内某个请求超时，会不会取消整个 batch？
  - 要点：不会。批跑在自己的 `Task` 里，单次尝试的超时由 `attempt_timeout` 与剩余总预算取小后控制，超时映射成可重试的 `ServerError` 而非取消信号；调用方自己的超时只影响它的 `Future`（结果被丢弃），不影响同批其他请求。
- **追问**：队列满了该阻塞还是拒绝？怎么保证请求不丢不重？
  - 要点：阻塞（背压）保住吞吐但把排队延迟转嫁给调用方，拒绝保住尾延迟但要求上游有降级路径；实测 `raise` 拒 36/40、`block` 全接纳但墙钟 503–507ms（下限 500ms），生产上按优先级分流。不丢不重靠 `submit` 与 `Future` 一一对应、只 `set_result` 一次，断言 `sum(batch_sizes) == 提交数`；真正的重复风险在服务端——调用超时无法判断对方是否已处理，所以写操作必须带幂等键（[[agents-02]] 的 at-least-once 加幂等消费）。

## 公司变体

- **Anthropic**：偏工程实现与错误语义。关注契约是否清晰（等长返回、异常即整批失败）、错误分类与幂等、`Retry-After` 的处理、以及「一个坏请求不能拖垮整批」的具体机制；追问常落在取消语义（调用方超时后批任务怎么办）与超时预算分层上。
- **Perplexity**：偏峰值流量下的取舍。关注突发到达时的攒批行为、并发上限与限流参数的配合、尾延迟与吞吐的权衡、队列积压时的公平性；追问常是「峰值时你先牺牲谁」这类取舍题，而不是纯 API 细节。

## 相关题目

- [[agents-02]]：五类错误、幂等与熔断，重试的错误分类直接沿用那一套。
- [[agents-09]]：循环终止与成本、步数上限，时间预算下传的口径与本题的请求预算同源。
- [[inference-serving-02]]：continuous batching 解释「为什么攒批能提高 GPU 利用率」，是第 2 小节的依据。
- [[coding-07]]：令牌桶限流器，服务端参数决定客户端重试预算怎么定。
- [[evaluation-07]]：可观测性 schema，批大小、成功率、重试率与 P99 要落进 trace。
- [[system-design-09]]：LLM gateway 的请求收集与路由，跨实例批量要在这层做。

## 参考资料与归属

1. [Coroutines and Tasks（Python asyncio 文档）](https://docs.python.org/3/library/asyncio-task.html)（延伸），Python Software Foundation。提供本文用到的原语语义：`TaskGroup` 中「第一个非 `CancelledError` 的失败会取消组内其余任务」并把未处理异常聚合成 `ExceptionGroup`；`asyncio.timeout` 把内部的 `CancelledError` 转成 `TimeoutError`（因此只能在上下文管理器外面捕获）；`wait_for` 超时会取消被等待对象；以及 `Semaphore`、`Queue` 的用法。数值与代码验证 ⑤ 的 TaskGroup 反例与实现里的超时分支出自该文档；文档没有给批处理器设计，攒批、预算与结算逻辑为自行设计。
2. [Exponential Backoff And Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)（延伸），Marc Brooker（AWS Architecture Blog），2015-03-04。提供抖动的三种口径（Full / Equal / Decorrelated）与模拟结论：在 100 个竞争客户端、网络延迟均值 10 ms 方差 4 ms 的 OCC 模拟里，加抖动把调用次数减少一半以上并显著改善完成时间，Equal Jitter 比 Full Jitter 略差，Decorrelated 与 Full 的取舍不明显；退避不改变冲突工作的 $N^2$ 性质，只显著减少实际工作量。原理与推导第 4 小节的三种抖动公式与上述结论引自该文；本机复算的差异（立即拒绝型漏桶下 full 的总尝试数略高）已在数值与代码验证 ④ 注明。

除上述来源外的全部内容——批处理器实现、假 API、断言脚本、表 1 至表 4、5 段实测输出、熔断与背压两组数字——都是在本机（AMD Ryzen 9 8945HX、`nproc` = 32；Python 3.11.0rc1 与 3.10.12）真实运行得到：文档中贴出的就是实测文件，验证驱动单独运行。时间类数字受机器负载与频率影响（表中给出多次重跑的区间），统计类数字由固定随机种子决定，可复现。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
