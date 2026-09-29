---
type: question
id: coding-07
topic: 编程与数据结构
order: 7
question: 实现一个 token-bucket 限流器，然后把它改造成分布式的。
question_en: Implement a token-bucket rate limiter, then make it distributed.
asked_at: [Anthropic, OpenAI, xAI, Cohere]
level: 高阶
tags: [限流, token-bucket, 分布式, 实现题]
sources:
  - title: A Single Rate Three Color Marker（RFC 2697）（延伸）
    url: https://www.rfc-editor.org/rfc/rfc2697
    author: Heinanen & Guerin (IETF)
    published: 1999-09
  - title: Counting a lot of different things（Cloudflare 工程博客）（延伸）
    url: https://blog.cloudflare.com/counting-things-a-lot-of-different-things/
    author: Cloudflare
    published: 2017-06-19
related: [system-design-09, coding-06, inference-serving-09, system-design-10]
updated: 2026-09-28
---

## 一句话答案

> 单机版只有一个状态和一条更新式：$tokens \leftarrow \min\big(b,\ tokens + (t - t_{last})\cdot r\big)$，只在请求到达时懒补充，绝不设定时器；$r$ 是补充速率，$b$ 是突发上限（桶深）。分布式版有三条路线：中心化存储（Redis 里一次 EVAL 完成读-改-写，时间取服务端 `TIME`）、局部额度加周期同步（每实例分 $r/N$，用精度换延迟）、租借（一次租 $L$ 个，RPC 降为 $1/L$，代价是需求不均时漏发）。三条路线都在「一致性、延迟、精度」这个三角里取舍，底线只有两条：判定必须原子，时间必须来自同一个时钟源。

## 面试官在考什么

- **能不能真写出可运行的实现**：懒补充、$\min(b,\cdot)$、单请求 cost、`retry_after` 秒数，这几个细节一眼就能看出是背的还是写过的。
- **知不知道为什么不设定时器**：定时器要后台线程、会 tick 抖动、测试不可控；懒补充把结算推迟到请求到达，结果只依赖时间差。
- **分布式部分是否落在正确性上**：原子性（Lua / 单命令）、统一时钟（服务端 `TIME` 而不是实例本地时间）、限流键的设计、拒绝策略与 `Retry-After`。
- **会不会算**：租借的额度滞留与漏发、RPC 放大倍数、Little's law（在途请求数 = RTT × QPS）。
- **常见错误答案**：起定时器补 token；先 `GET` 再 `SET`；用实例本地时间做懒补充；把 token bucket 与漏桶混为一谈（漏桶是恒定输出速率，不允许突发）；只写固定窗口就说自己实现了限流。

## 原理与推导

### 两个参数与一条更新式

$r$（token/秒）是长期平均速率，$b$ 是桶容量，也就是「允许的突发上限」：桶满时瞬间能放行 $b$ 个请求，之后被压回 $r$。桶空到满需要 $b/r$ 秒，这个量在分布式设计里反复出现（键的 TTL、租借的回收周期都用它）。

设 $t_{last}$ 是上次结算时刻，$tokens$ 是结算后的余量，请求到达时：

$$tokens \leftarrow \min\big(b,\ tokens + (t - t_{last})\cdot r\big),\qquad t_{last} \leftarrow t$$

**为什么这是对的**：桶以速率 $r$ 连续补充，两次结算之间补充量就是时间差乘 $r$，与「每秒加 $r$ 次 1」的定时器版本相比只是改变了结算时刻，额度按时间差累加（可加性），总额度序列完全一致。**为什么必须取 min**：不取 min，桶空闲 1 小时就攒下 $3600r$ 个 token，下一个请求能把整段空闲换成一个无限大的突发——实测中 $r=1/\text{s}$、$b=30$ 的桶空闲 3600 s 后仍只有 30 个 token。

**为什么不能用实例本地时间**：`elapsed` 是两次结算的时间差，只要写入状态的时间戳来自不同时钟源，时间差就会在切换点被「重复记账」或「漏记」。实测（两个实例交替、两实例时钟差 1 ms、$r=100/\text{s}$、$b=10$、1 s 内 6000 次请求）通过 359 次，而理论上限只有 110 次；时钟差 50 ms 时 6000 次请求全部放行。一次 NTP 回拨 1 s 更直接：把更早的时间写回 `last` 的幼稚写法在 $r=100/\text{s}$ 下会凭空多补 100 个 token。

另外两个实现细节：`elapsed <= 0` 时既不补充也不回退 `last`（时钟冻结或回拨时的唯一安全动作）；单请求 cost 必须满足 $0 < cost \le b$，$cost > b$ 的请求永远无法通过，这类配置错误要在构造/入口就抛出来，而不是静默拒绝。

复杂度：每次请求 $O(1)$ 时间、$O(1)$ 空间（状态只有 `tokens` 与 `last` 两个浮点数）。实测单机 0.7–1.8 M ops/s（571–1463 ns/次，5 次重复测量的区间，单线程、含一把互斥锁，机器见验证一节）。

### 与其他算法的区别

| 算法 | 状态 | 突发 | 窗口边界 | 备注 |
| --- | --- | --- | --- | --- |
| 固定窗口计数 | 1 个计数 | 允许 2× | 有 | 实测 2 s 内放行 100 个，而限额是 50 |
| 滑动窗口日志 | 窗口内每次请求时间戳 | 精确 | 无 | 内存与请求数成正比 |
| 滑动窗口计数 | 2 个计数 | 近似 | 显著缓解 | 实测第二段 50 个请求全部拒绝 |
| 漏桶 | 1 个计数（队列） | 恒定输出，不允许突发 | 无 | nginx `limit_req` 用的是它 |
| token bucket | 2 个数（tokens、ts） | 允许 $b$ | 无 | 本题 |
| GCRA / 虚拟调度 | 1 个时间戳 | 允许 $\tau/T + 1$ | 无 | 很多 CDN 与网关的实现 |

滑动窗口计数是 Cloudflare 2017 年那篇工程博客采用的口径：用上一窗口与当前窗口加权外推，$\text{rate} = \text{prev}\cdot\frac{T - elapsed}{T} + \text{cur}$。他们给的例子是限额 50/min、上一窗口 42 次、当前窗口已过 15 s 且已有 18 次，估计速率 $42 \times \frac{60-15}{60} + 18 = 49.5$，再来一个就拒。同一篇文章给出的效果口径是：在 4 亿次请求（27 万个不同来源）的分析中，0.003% 的请求被错误放行或错误限流，估计速率与真实速率的平均差距 6%，被放行的 3 个来源实际只超阈值不到 15%，没有来源在低于阈值时被限流。这篇文章还交代了他们的取舍：leaky bucket 精度更好但需要多个非原子操作（他们当年只能用 memcached 的 `GET`/`SET`/`INCR`），于是选了滑动窗口近似——两个数字即可、`INCR` 一条命令可写，并把计数改成异步执行、只把「是否开始限流」这一个比特放到请求路径上。

GCRA 值得单独讲，因为它把状态从两个数压到一个时间戳（$T$ 为发送间隔 $1/r$，$\tau = (b-1)/r$ 为容忍的到达时间抖动）：

$$\text{放行} \iff t \ge TAT - \tau,\qquad TAT \leftarrow \max(t, TAT) + T$$

实测在 $r=20,b=5$、$r=1.5,b=8$、$r=500,b=1$ 三组参数下各 20000 次随机到达，GCRA 与 token bucket 的判定**不一致 0 次**——两者是同一语义的两种坐标表示，工程上选哪个只看要不要多一个可读的「剩余额度」。RFC 2697 的单速率三色标记（srTCM）是「桶 + 突发」这套模型在标准化文本里的先例：CIR、CBS、EBS 三个参数，两个共享 CIR 的桶 C 与 E 初始全满，每秒钟给 C 加一次、C 满了给 E 加；占用不超过 CBS 标记 green，不超过 EBS 标记 yellow，否则 red。要标注清楚语境：这是 Diffserv 里的网络流量整形与标记，且 RFC 2697 的状态是 Informational（并非 Internet Standard），它明确规定「实际实现不必按上述形式化规格建模」——这正是懒补充在标准层面被允许的依据。

### 分布式：三条路线与一个三角

| 路线 | 一致性 | 每请求延迟 | 精度 | 主要失效模式 |
| --- | --- | --- | --- | --- |
| 中心化 Redis + 单条原子脚本 | 强（单键串行执行） | 多一次 RTT | 精确 | 热点键、中心故障成为全局单点 |
| 局部额度 + 周期同步 | 弱 | 本地，0 额外延迟 | 配置的 $N$ 与实际实例数不符就错 | 扩容/滚动发布期超发 |
| 租借 lease | 中（额度守恒，但会滞留） | 每 $L$ 个请求付一次 RTT | 租借越大漏发越多 | 进程死亡时未用完的额度丢失 |

中心化路线的关键是**一次 EVAL 完成全部读-改-写**，并用服务端 `TIME` 取时间。`MULTI`/`EXEC` 也能让一组命令原子执行，但事务里拿不到中间结果做分支（要分支就得 `WATCH` 加重试），所以带判定的读-改-写只能写成 Lua 脚本或单条命令。先 `GET` 再算再 `SET` 一定会在并发下超发：实测 16 线程 × 20 次请求、全局额度 100，原子脚本放行 100 次（超出 0），`GET`-再-`SET` 放行 320 次（超出 220）——额度守恒完全失效。中心化的代价是每请求一次 RPC：注入 0.2 ms 往返时单连接只能做 3.6k–4.0k req/s（253–280 µs/次），8 连接合计 2.5 万–3.1 万 req/s；注入 1 ms 时降到约 0.9k / 7.0k–7.6k req/s（5 次测量的区间）。按 Little's law，要在 0.5 ms RTT 下支撑 10 万 QPS 的中心化限流，平均在途请求数就是 $0.0005 \times 100000 = 50$ 个连接，这是连接池与超时预算要预留的量。

局部额度路线的错误是系统性的：每实例分 $r/N$ 并且本地判定，一旦实际实例数与配置不一致，全局出力就跟着错。实测配置 $N=8$：滚动发布期间有 12 个实例在跑时全局上限变成 150/s（+50%），只剩 3 个实例时变成 37.5/s（−62%）。它适合「允许少量超发、但绝不能因为限流器挂掉而拒绝用户」的场景，并且必须把实例数做成带租约的注册表，让配额随真实实例数收敛。

租借路线是两者的折中：实例本地有余量就直接放行，用完再向中心原子地租 $L$ 个。它把中心 RPC 从每请求一次压到每 $L$ 个请求一次，代价有三块：一是**粒度漏发**——中心剩 90 个 token 而 $L=100$ 时一个都租不出去（实测放行 0/90）；$L=25$ 时只能租出 3 份，实测放行 75/90（83.3%）。二是**需求不均**——需求随机分布在 8 个实例上、总量 200、额度 90，2000 次蒙特卡洛的平均放行率：$L=1$ 与 $L=10$ 均为 100%，$L=25$ 降到 78.1%（中心零头 15、实例滞留 4.7），$L=50$ 降到 28.8%，$L=100$ 降到 0%。三是**进程死亡**——租出去没用完的额度要靠租约 TTL 回收，否则永久泄漏。反过来，租借不会超发：token 只是从中心搬到实例，总量守恒。

三条路线的取舍可以收成一句话：**要强一致就付 RTT 与热点风险，要低延迟就让出精度，要两者兼得就用租借，用「一笔额度的等待时间」买「一条 RPC 的摊销」**。

### 生产化的差距：这段代码为什么不能直接上线

- **限流键与层级**。键不是「用户 id」这么简单：网关层要按 `租户 / API key / 路由 / 模型` 组合，并且常常需要多级同时约束（全局 → 租户 → 用户 → 路由），任一级先超限就拒。键的基数决定内存：1 亿个活跃键、每键 40 字节左右的状态，就是几 GB 量级，必须有 TTL 与惰性清理。
- **超发与漏发哪个更糟**。保护下游（数据库、GPU 池）时宁可严格，漏发可以接受；保护用户体验时宁可宽松。同一套代码里这两种取向要通过「是否预留余量、租借大小、超限后是否排队」来体现，而不是靠拍参数。
- **限流的目的是分层的**：防滥用（按来源、按指纹）、公平性（按租户份额，见 [[system-design-09]] 的预算与公平份额）、成本控制（按 token 计费量）、保护下游（按并发与队列深度）。用同一个 RPM 数字覆盖这四种目的，是生产事故的常见起点。
- **按请求限流 ≠ 按成本限流**。LLaMA-3-70B 每 token 的 KV cache 是 320 KiB，一个 4096 token 上下文的请求就占 $4096 \times 320\ \text{KiB} \approx 1.25$ GiB 显存；而 decode 是带宽受限的（H100 SXM5 的 bf16 dense 算力 989 TFLOPs、HBM 3.35 TB/s，roofline 拐点在 $989/3.35 \approx 295$ FLOPs/byte，decode 落在带宽一侧），所以真正稀缺的是 token 而不是请求数。对外限流要用 RPM + TPM 双轴，并且输入/输出分开计（长上下文的输入会把带宽与显存吃掉一大块）。
- **可观测性**。必须能回答「现在被限流的比例是多少、Top 被限流主体是谁、429 集中出现在哪个租户/路由」——这属于 [[evaluation-07]] 的指标体系；没有这个，限流参数就只能靠猜。
- **与重试的相互作用**。被限流的客户端会重试，重试会加剧拥塞，所以拒绝时必须给可行动的 `Retry-After`，客户端必须配合退避与抖动：全抖动（full jitter，AWS 在 2015 年的 Exponential Backoff And Jitter 里给出的口径是 `sleep = random(0, min(cap, base * 2^n))`，把期望等待压到上限的一半）是这里的最低要求，细节见 [[coding-08]]。
- **别把限流和排队混为一谈**。排队只对可等待流量开放，利用率 $\rho$ 超过 0.7 后排队时间按 $\rho/(1-\rho)$ 上升，用排队掩盖容量不足会把 p99 变成事故；网关层与应用层的分担方式见 [[inference-serving-09]]。

## 数值与代码验证

环境：Python 3.10.12（CPython，单进程）；机器 AMD Ryzen 9 8945HX（16 核 32 线程）/ Linux x86_64，测量期间机器上还有别的负载，所以吞吐与 RPC 这类墙钟数字给的是多次重复测量的区间；`numpy 2.2.6`、`torch 2.12.0` 可用但本题用不到，参考实现是纯标准库；本机没有 `redis-server` 二进制，用 `fakeredis 2.38.0` + `lupa 2.8` 提供的真实 Lua 解释器执行 `EVAL`（Redis 命令语义、脚本原子性、TTL 都是真的，网络不是）。脚本：`.work/rl_token_bucket.py`（单机与对照算法）、`.work/rl_distributed.py`（分布式三条路线）、`.work/rl_redis_lua.py`（Lua 脚本）。

### 单机参考实现（可运行，已跑通全部断言）

```python
import threading, time


class TokenBucket:
    """r = 每秒补充 token 数；b = 桶容量（突发上限）。无后台线程。"""

    def __init__(self, rate, capacity, clock=time.monotonic, locked=True):
        if rate < 0:
            raise ValueError("rate 必须 >= 0")
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        self.rate, self.capacity, self.clock = float(rate), float(capacity), clock
        self.tokens = float(capacity)          # 初始满桶：允许一次 b 的突发
        self.last = clock()
        self.granted = self.rejected = 0
        self._lock = threading.Lock() if locked else None

    def _refill(self, now):
        elapsed = now - self.last
        if elapsed <= 0:                       # 时钟冻结/回拨：不补、也不回退 last
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)  # 必须取 min
        self.last = now

    def _try_acquire(self, cost, now):
        if not 0 < cost <= self.capacity:
            raise ValueError("cost 必须落在 (0, capacity]：cost > b 的请求永远无法通过")
        self._refill(self.clock() if now is None else now)
        if self.tokens >= cost:
            self.tokens -= cost
            self.granted += 1
            return True
        self.rejected += 1
        return False

    def try_acquire(self, cost=1.0, now=None):          # now 可注入，测试用可控时钟
        if self._lock is None:
            return self._try_acquire(cost, now)
        with self._lock:
            return self._try_acquire(cost, now)

    def retry_after(self, cost=1.0, now=None):          # 单位：秒，直接写进 Retry-After
        now = self.clock() if now is None else now
        if self._lock is not None:                     # 读-改-写，必须和 try_acquire 抢同一把锁
            with self._lock:
                return self._retry_after(cost, now)
        return self._retry_after(cost, now)

    def _retry_after(self, cost, now):
        self._refill(now)
        return max(0.0, (cost - self.tokens) / self.rate) if self.rate > 0 else float("inf")
```

手算用例（$r=2/\text{s}$、$b=5$，逐条断言通过）：$t=0$ 连发 5 个全过、第 6 个拒；$t=0.4$ 拒（只补回 0.8 个）；$t=0.6$ 过一个、再一个拒；$t=1.1$ 过一个、再一个拒；$t=10$ 又连过 5 个——17 条用例全部符合，`tokens` 始终不超过 $b$。

| 验证项 | 参数 | 实测结果 |
| --- | --- | --- |
| 突发上限 | $b=30$、$r=1/\text{s}$、冻结时钟连发 40 次 | 通过 **30** 次；空闲 10 s 后再发 40 次新增通过 **10** 次；空闲 1 h 后补充量被 $b$ 截在 30 个额度（不取 min 会攒下 3600 个） |
| 速率精度 | $r=100/\text{s}$、$b=20$、5 ms 步长跑 10 s（2000 次请求） | 通过 1019 次，理论 $b + rT = 1020$，偏差 **−1**（−0.098%） |
| 并发正确性（加锁） | 冻结时钟、16 线程 × 200 次、额度 100 | 通过 **100**，超出额度 **0** |
| 并发正确性（去掉锁，把读-改-写拆开） | 同上 | 通过 **1247–1595**（5 次测量），超出额度 **+1147…+1495** |
| GCRA 等价性 | 3 组参数 × 20000 次随机到达 | 与 token bucket 判定不一致 **0** 次 |
| 窗口边界（限额 50/min） | 窗口末 50 个 + 下一窗口初 50 个 | 固定窗口放行 **100**；滑动日志 **50**；滑动窗口近似 **50** |
| 单机吞吐 | 1,000,000 次 `try_acquire` | 冻结时钟 **0.7–1.8 M ops/s**（571–1463 ns/次）；真实时钟 **0.7–1.3 M ops/s**（785–1493 ns/次），均为 5 次测量的区间 |

### 分布式：Redis 里的 Lua 脚本（真实执行过）

```lua
-- KEYS[1]=限流键；ARGV[1]=rate；ARGV[2]=capacity；ARGV[3]=cost；ARGV[4]='server' 或时间戳
local rate = tonumber(ARGV[1])
local cap  = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now
if ARGV[4] == 'server' then
  local t = redis.call('TIME')                       -- 服务端时间，不信任调用方的表
  now = tonumber(t[1]) + tonumber(t[2]) / 1000000
else
  now = tonumber(ARGV[4])                            -- 反例路径，仅用于对照实验
end

local rec = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens, ts = tonumber(rec[1]), tonumber(rec[2])
if tokens == nil then tokens = cap; ts = now end     -- 惰性创建：第一次请求就是满桶

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end                  -- 时间不前进或回拨时不倒扣
tokens = math.min(cap, tokens + elapsed * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HSET', KEYS[1], 'tokens', string.format('%.6f', tokens),
                            'ts', string.format('%.6f', ts + elapsed))
local ttl = 60000
if rate > 0 then ttl = math.ceil(cap / rate * 1000) + 1000 end   -- 空桶补满的时间 + 余量
redis.call('PEXPIRE', KEYS[1], ttl)
return allowed
```

要点：整段逻辑在服务端一次执行完，读-改-写之间不可能插入别的命令；时间由 `TIME` 提供；浮点数一律 `string.format` 成字符串再存（不要依赖 Lua number 的隐式转换——实测把 Lua 计算出的浮点数直接作为返回值时会被截断成整数）；`PEXPIRE` 的 TTL 至少是「空桶补满时间」$b/r$。

| 验证项 | 参数 | 实测结果 |
| --- | --- | --- |
| `SCRIPT LOAD` + `EVALSHA` 原子性 | 16 线程 × 20 次、额度 100、`rate=0` | 通过 **100**，超出额度 **0**（上面这段脚本就是被测脚本：1112 字节，sha1 36f934c8…） |
| 反例：客户端 `GET` 再 `SET` | 同上 | 通过 **320**，超出额度 **+220** |
| 实例数无关性 | $r=100/\text{s}$、$b=50$、1 s、实例数 1/2/4/8 | 均通过 **149** 次（理论上限 $b + rT = 150$），全局配额不随实例数变化 |
| 时钟口径 | $r=100/\text{s}$、$b=10$、1 s 内 6000 次请求 | 服务端时间 **109**（上限 110）；本地时间差 0.2 ms → **119**；1 ms → **359**；5 ms → **1559**；50 ms → **6000**（全放行） |
| 时钟回拨 | NTP 回拨 1 s，$r=100/\text{s}$ | 不回退 `last` → 正确补 10 个；幼稚写法 `last = now` → 补 **110** 个（凭空多 100 个） |
| 额度守恒 | $r=200/\text{s}$，跑 200 ms，核对 $b + rT - \text{余量}$ | $b=1$：预期 40.1–40.8、实测 28–39（残差 +1.1…+12.8）；$b=20$：预期 59.1–60.3、实测 59–60（残差 < 1），6 次测量 |
| TTL 陷阱 | $b=100$、$r=100/\text{s}$，耗尽后 0.15 s 再申请 50 个 | 正确 TTL 2000 ms → **拒绝**（只补回约 15 个）；错误 TTL 100 ms → 键已过期、桶被当满桶重建 → **放行** |
| `INCR` + `EXPIRE` 写法 | 限额 3/100 ms，每 20 ms 一次共 15 次 | 只在 `n == 1` 设 TTL → 通过 **9** 次；每次请求都刷新 TTL → 通过 **3** 次（计数器被刷活，永久卡死） |
| 脚本自身开销 | 进程内 `EVAL` 1 万次 | 约 **0.19–0.48 ms/次**（6 次测量）——这是 fakeredis 的 Python 开销，不代表真实 redis-server，只说明脚本逻辑本身可忽略 |

额度守恒里的残差不是实现误差：$\min(b,\cdot)$ 在桶顶截掉了「超过 $b$ 的那部分补充」。$b=1$ 时每次放行都撞在桶顶上，被截掉的量约等于「两次尝试之间补充的 token 数」，请求来得越稀（循环越慢）残差越大，实测就在 +1.1 到 +12.8 之间摆动；$b=20$ 时只有桶真正满的瞬间才截断，残差稳定在 1 个 token 以内。想让这个数字稳定，就得用可控时钟按固定步长推进，而不是跑墙钟。

### 分布式：租借的客户端与 RPC 代价

```python
class LeaseLimiter:
    """一次向中心租 lease 个 token，本地用完再租。"""

    def __init__(self, store, key, rate, capacity, lease):
        self.store, self.lease = store, lease
        self.script = make_lease_script(key, rate, capacity, lease)   # 与上面同构的原子脚本
        self.local = 0
        self.leases_ok = self.leases_denied = 0

    def acquire(self):
        if self.local <= 0:                       # 只有租借时才有一次 RPC
            got = self.store.script(self.script)
            if not got:
                self.leases_denied += 1           # 中心凑不出一个整租
                return False
            self.local = got
            self.leases_ok += 1
        self.local -= 1
        return True
```

8 个实例、全局额度 90（冻结时钟，隔离出租借本身的误差）：无限需求下 $L=1$ 与 $L=10$ 都放行 90（100%），$L=25$ 只放行 75（83.3%，中心剩 15 个凑不出整租），$L=100$ 放行 **0**；需求随机、总量 200 的 2000 次蒙特卡洛下，$L=50$ 的平均放行率掉到 28.8%。另外注意被拒租借的次数与请求数同阶（$L=1$ 时 8 个实例各发 100 次请求，800 次请求对应 710 次被拒的中心调用）——被拒后必须有本地冷却窗口，否则「省 RPC」会被拒绝路径吃回去。

## 常见追问

- **追问**：为什么不用本地时间做懒补充？
  - 要点：`elapsed` 依赖两次结算的时间戳，只要时间戳来自不同时钟源，跨实例切换就会重复记账。实测两实例时钟差 1 ms 时 1 s 内放行 359 次（上限 110），差 50 ms 时 6000 次全放行。正确做法是让中心提供时间（Lua 里 `TIME`），或让 Redis 这类单键串行存储成为唯一写入者。
- **追问**：令牌桶与漏桶怎么选？
  - 要点：桶允许突发（上限 $b$），漏桶恒定输出速率、不允许突发；bucket 更适合「平均速率 + 容忍突发」的 API 限流，漏桶适合平滑下行流量与整形。两者状态都是常数大小，区别只在语义。
- **追问**：网关层与应用层怎么分担？
  - 要点：网关层做粗粒度、高基数的防护（按 IP/租户/路由，$O(1)$ 状态、无 I/O），应用层做与业务成本相关的精细限流（按 token 预算、按并发、按下游容量），两层用不同的键与参数，不要用同一个数字；容量与延迟的权衡见 [[inference-serving-09]]。
- **追问**：请求一次性要 1000 个 token，但 $b$ 只有 100 怎么办？
  - 要点：要么把 cost 建模成多级（把大请求拆成多次 acquire，失败则回滚成「要么全给要么不给」的两阶段），要么把 $b$ 调到 $cost_{max}$，要么对这类请求直接拒（配置错误应当在启动时校验并报警，而不是静默永久拒绝）。
- **追问**：被限流之后呢？
  - 要点：返回 429 与可行动的 `Retry-After`（用 `retry_after` 算出来的秒数），客户端按 full jitter 退避；网关同时要防止「重试放大」把限流变成雪崩，见 [[coding-08]]。
- **追问**：多级限流怎么保证不会互相打架？
  - 要点：从粗到细依次判定，任一级先拒绝就短路返回；每级用独立的键与原子脚本；级别之间用不同的时间尺度（秒级对突发、分钟/天级对配额），并统一在 429 响应里说明是哪一级、什么时候恢复。

## 公司变体

四家都以 API 形式对外出售模型能力，限流是它们对外契约的一部分，所以这题在这四家都偏**工程实现与生产细节**，而不是数学推导：写出能跑的代码、解释为什么必须原子、给出 429 与 `Retry-After` 的语义。

- **Anthropic**：对外限制是 RPM 与 ITPM/OTPM（输入/输出 token 每分钟）双轴，响应头会回传剩余额度与恢复时间——正好对应「按请求」与「按成本」两种限流对象，回答时最好主动区分。
- **OpenAI**：公开的 rate limit 文档以 `x-ratelimit-*` 系列响应头暴露剩余请求数/token 数与重置时间，429 是标准拒绝信号；面试里通常追问「限流头怎么设计、客户端怎么用」。
- **xAI**：Grok API 同样按模型与档位给出 RPM/TPM 限制，问题更偏「多租户下怎么分档、怎么保证突发不互抢」。
- **Cohere**：限流按 API key 的档位区分（试用与生产不同），追问往往落在「按 key 计数」的键设计与配额如何随档位热更新。

## 相关题目

- [[coding-06]]：同样考数据结构与常数级复杂度，但 LRU 的难点是并发与淘汰，这题的难点是原子性与时钟。
- [[coding-08]]：被限流后的重试、并发上限与错误隔离，是限流器在客户端一侧的对偶问题。
- [[system-design-09]]：gateway 的预算与限流策略，把这题的单点实现放回系统里。
- [[system-design-10]]：消费级聊天助手的服务栈，量级放大后限流键与热点问题的来源。
- [[inference-serving-09]]：吞吐与延迟的权衡，决定限流参数该按什么口径设。
- [[evaluation-07]]：被限流比例、Top 被限流主体的可观测性指标。

## 参考资料与归属

- Heinanen & Guerin (IETF)，《A Single Rate Three Color Marker》（RFC 2697），1999-09：https://www.rfc-editor.org/rfc/rfc2697 ——第 3 节的 CIR/CBS/EBS、两个共享速率的桶与 green/yellow/red 标记来自此文，「实现不必按形式化规格建模」也是原文表述；注意语境是 Diffserv 网络流量整形，且状态为 Informational。本文未复现该文的多色标记逻辑。
- Cloudflare，《Counting a lot of different things》（How we built rate limiting capable of scaling to millions of domains），2017-06-19：https://blog.cloudflare.com/counting-things-a-lot-of-different-things/ ——第 3 节的滑动窗口近似公式、42/18/15 s 的算例、4 亿请求与 27 万来源的 0.003%、6%、3 个来源的统计口径，以及「leaky bucket 需要非原子多操作所以改用近似计数」的取舍，均出自此文；文中未标注该日期以外的版本信息。
- AWS 的 full jitter 口径（Exponential Backoff And Jitter，2015）与 H100 SXM5 的 989 TFLOPs dense、3.35 TB/s、LLaMA-3-70B 每 token 320 KiB KV cache 等常数沿用本仓库其它专题的一致口径，未在作业单来源中列出。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
