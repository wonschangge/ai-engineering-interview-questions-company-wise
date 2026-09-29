---
type: question
id: openai-03
company: OpenAI
topic: coding
order: 3
question: 用 Go 编写一个简单的 web 爬虫。
question_en: Write a simple web crawler in Go.
asked_at: []
level: 高阶
tags: [Go 并发, worker 池, 礼貌限速, URL 去重, 背压]
sources:
  - title: The Go Programming Language Specification（延伸）
    url: https://go.dev/ref/spec
    author: The Go Authors
    published: 2023-01-01
  - title: Go Concurrency Patterns: Pipelines and Cancellation（延伸）
    url: https://go.dev/blog/pipelines
    author: Sameer Ajmani (Google)
    published: 2014-03-13
  - title: Robots Exclusion Protocol（延伸）
    url: https://www.rfc-editor.org/rfc/rfc9309.html
    author: Koster et al. (IETF)
    published: 2022-09-01
  - title: 一步步实现一个 ORM（本仓库公司题库 · OpenAI 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [openai-02, openai-18, coding-10, coding-11, coding-12]
updated: 2026-09-28
---

## 一句话答案

> **"简单爬虫"的难点不在抓取，而在四件事：并发有界、礼貌、去重、可停止**。
> **骨架（Go）**：
> ```go
> frontier := make(chan string, 10000)      // ★ 有界队列 = 背压
> var wg sync.WaitGroup
> ctx, cancel := context.WithCancel(ctx)    // ★ 可取消
> for i := 0; i < workers; i++ {            // ★ 固定 worker 池
>     wg.Add(1)
>     go func() { defer wg.Done(); worker(ctx, frontier, ...) }()
> }
> // 生产者：抓到的链接入队；全部完成后 close(frontier)（★ 由"最后一个发送者"关闭）
> ```
> **四个量化要点**：
> **① 并发必须有界**（"每个 URL 一个 goroutine"会 OOM）：
> | URL 数 | goroutine 栈内存（8 KB/个） | worker 池（100 个） |
> | --- | --- | --- |
> | 100,000 | 0.8 GB | **0.001 GB** |
> | **1,000,000** | **8.0 GB** | **0.001 GB** |
> **读法**：**无界 goroutine 的内存随 URL 数线性增长**（100 万 URL ≈ 8 GB）——**worker 池把内存固定住**，**代价是队列必须有界**（否则队列本身会吃光内存）。
> **② 礼貌是"按域"的，所以吞吐 ≈ 域数 × 每域速率**：
> | 域数 $H$ | 每域 1 req/s | 全局 1 req/s | 倍数 |
> | --- | --- | --- | --- |
> | 10 | 10/s | 1/s | 10× |
> | 1,000 | **1,000/s** | 1/s | **1,000×** |
> | 10,000 | **10,000/s** | 1/s | 10,000× |
> **读法**：**"礼貌"与"吞吐"并不冲突**（**只要目标站足够多**）——**真正冲突的是"同一个站抓太多"**（那就必须降速，或接受被封）。**所以限速器必须是"每域一个令牌桶"，而不是全局一个**。
> **③ 重试必须退避 + 抖动**（否则把故障放大成雪崩）：
> | 失败率 | 立即重试 3 次的负载 | 指数退避 + 抖动 | 倍数 |
> | --- | --- | --- | --- |
> | 10% | 1.11× | 1.03× | 1.07× |
> | **30%** | **1.39×** | 1.10× | **1.26×** |
> **读法**：**立即重试的问题不是"量"而是"同时"**（**所有重试挤在同一时刻，把对端的故障期变成雪崩**）；**退避 + 抖动把重试摊开**——**这是"对故障方礼貌"**。
> **④ 去重的内存与正确性取舍**：
> | 做法 | 1 亿 URL 的内存 | 正确性 |
> | --- | --- | --- |
> | 字符串集合 | **10.0 GB** | 精确 |
> | **64 位哈希集合** | **0.8 GB** | **精确**（碰撞概率可忽略） |
> | **Bloom 过滤器** | **0.12 GB** | **近似：1% 的新 URL 被误判为已抓 → 漏抓** |
> | Bloom + 哈希集合 | 0.92 GB | 精确 + 快速路径 |
> **读法**：**Bloom 最小但会漏抓**（**对爬虫通常可接受**——覆盖率本来就不是 100%）；**要精确就用 64 位哈希集合**（0.8 GB）；**Bloom 放在哈希集合前面只省查找时间、不省内存**。
> **三个 Go 特有的坑**：
> | # | 坑 | 正确做法 |
> | --- | --- | --- |
> | ① | **`send on closed channel`** | **只由"最后一个发送者"关闭**；多生产者时用一个单独的 goroutine 在 `wg.Wait()` 后关闭 |
> | ② | **goroutine 泄漏** | **所有 worker 都监听 `ctx.Done()`**；取消后仍要能退出 |
> | ③ | **`sync.Map` 不是万能** | 写多读少时**分片锁 map 更快**；`sync.Map` 适合"读多写少、键集合稳定" |
> 一句话判据：**"固定 worker 池 + 有界队列（背压）+ 每域限速 + 哈希去重 + context 取消 + 退避重试"**——**六件事，缺一件就不是"能上线的爬虫"**。

## 面试官在考什么

- **★ 并发是否有界**：**能否指出"每 URL 一个 goroutine"会 OOM**（本机：100 万 URL ≈ 8 GB），并给出 worker 池 + 有界队列。
- **★ 礼貌的实现位置**：**能否指出"限速要按域"**（本机：域数越多总吞吐越高），**而不是全局限速**。
- **★ channel 的所有权**：**能否说清"谁来 close"**（**多生产者时必须单独关闭**）——**这是 Go 面试的经典考点**。
- **取消与优雅退出**：**能否用 `context` + `WaitGroup`**，并保证**没有 goroutine 泄漏**。
- **去重的取舍**：**能否给出"哈希 vs Bloom"的内存与正确性对比**（本机 0.8 GB vs 0.12 GB）。
- **URL 规范化**：**能否指出"同一个页面有多种 URL 形式"**（大小写、默认端口、`#fragment`、查询参数顺序、尾斜杠、`../`）——**不去重就会重复抓**。
- **重试策略**：**能否给出退避 + 抖动**，并说明**为什么立即重试有害**。
- **边界与终止**：**能否给出"最大深度/最大页数/超时"**（**否则爬虫不会停**）。
- **robots.txt**：**能否提到要遵守**（**并且要缓存 robots，否则每个请求都拉一次**）。
- **诚实**：**承认"简单爬虫"在生产上还要处理**：重定向循环、非 HTML 内容、字符编码、压缩、反爬、以及**法律与合规**。

**常见错误答案**

- **`go fetch(url)` 直接开 goroutine**（**无界 → OOM**）。
- **无界 channel**（**队列吃光内存**）。
- **全局限速**（**吞吐被压到 1/域数**）。
- **多个 goroutine 都 `close(ch)`**（**panic: send on closed channel**）。
- **不处理取消**（**Ctrl-C 后 goroutine 泄漏**）。
- **不去重或只按原始字符串去重**（**同一个页面抓很多次**）。
- **立即重试**（**把对端故障变雪崩**）。
- **不设深度/页数上限**（**爬虫不会停**）。
- **不遵守 robots.txt**（**合规问题**）。

## 原理与推导

### 1. 并发有界：worker 池 + 有界队列

**反模式**：

```go
for _, u := range urls { go fetch(u) }      // ✗ 无界
```

| URL 数 | goroutine 栈（8 KB/个） |
| --- | --- |
| 100,000 | 0.8 GB |
| **1,000,000** | **8.0 GB** |

**正模式**：

```go
frontier := make(chan string, 10000)         // ★ 有界
for i := 0; i < 100; i++ { go worker(frontier) }
```

**读法**：**worker 池把"并发数"与"待抓 URL 数"解耦**——**内存固定为"worker 数 + 队列容量"**。**有界队列的第二个作用**：**当消费慢于生产时，生产者会阻塞 → 天然的背压**（**防止"抓到的链接比处理得快"导致内存爆**）。

### 2. ★ 礼貌：每域限速，而不是全局限速

$$\text{总吞吐}\approx H\times r\qquad(H=\text{域数},\ r=\text{每域速率})$$

| 域数 | 每域 1 req/s | 全局 1 req/s | 倍数 |
| --- | --- | --- | --- |
| 10 | 10/s | 1/s | 10× |
| **1,000** | **1,000/s** | 1/s | **1,000×** |

**读法**：**礼貌约束是"对每个站点"的**——**所以限速器必须按域（per-host）维护令牌桶**：

```go
type HostLimiter struct {
    mu      sync.Mutex
    buckets map[string]*rate.Limiter      // ★ 每域一个
}
```

**注意**：**`map[string]*Limiter` 需要清理**（**否则域无限增长 → 内存泄漏**）——**用 LRU 或定期清理**。

**robots.txt 也要缓存**（**每域一份**），**并且 `Crawl-delay` 要尊重**（**它是"每域延迟"的直接来源**）。

### 3. ★ channel 的所有权：谁来 close

**Go 的规则**：**"不要从接收端关闭 channel；也不要让多个发送者各自关闭"**。

| 场景 | 正确做法 |
| --- | --- |
| 单生产者 | **生产者自己 close** |
| **多生产者** | **用一个单独的 goroutine：`wg.Wait(); close(ch)`** |
| 接收端 | **只 `range ch`，不 close** |

**读法**：**`send on closed channel` 会 panic**——**而"多个 worker 都可能成为最后一个生产者"时，谁来关闭就成了难题**——**标准解法是"单独的 closer goroutine"**：

```go
go func() { wg.Wait(); close(frontier) }()
```

**注意**：**"往 `frontier` 里写"的 goroutine 也要算进 `wg`**——**否则可能在还有人写的时候就 close 了**。

### 4. 取消与优雅退出

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()
// worker 里：
select {
case <-ctx.Done(): return                    // ★ 必须监听取消
case u, ok := <-frontier: if !ok { return }
}
```

**读法**：**"所有阻塞点都要能被取消"**——**包括**：**等待队列、HTTP 请求（用 `http.NewRequestWithContext`）、限速器等待**。**否则 Ctrl-C 之后 goroutine 会挂在那里**（**泄漏**）。

### 5. URL 规范化与去重

**同一个页面的多种形式**（**不去重就会重复抓**）：

| 变体 | 规范化 |
| --- | --- |
| `HTTP://Example.com:80/a` | 小写 scheme/host、**去掉默认端口** |
| `http://example.com/a#frag` | **去掉 fragment** |
| `http://example.com/a?b=1&a=2` | **查询参数排序**（谨慎：**有些站点参数顺序有意义**） |
| `http://example.com/a/` vs `/a` | 统一尾斜杠策略 |
| `http://example.com/a/../b` | **解析 `..`** |
| 相对链接 | **按 base URL 解析** |

**读法**：**规范化要在"入队前"做**（**否则队列里全是重复项**）。**注意**：**过度规范化也有风险**（**把不同页面合并成一个 → 漏抓**）——**所以"参数排序"这类规则要谨慎**。

**去重的内存（本机算例）**：

| 做法 | 1 亿 URL | 正确性 |
| --- | --- | --- |
| 字符串集合 | **10.0 GB** | 精确 |
| **64 位哈希集合** | **0.8 GB** | **精确**（碰撞可忽略） |
| **Bloom（1%）** | **0.12 GB** | **近似：漏抓约 1% 的新 URL** |
| Bloom + 哈希 | 0.92 GB | 精确 + 快速路径 |

**读法**：**Bloom 省 6.7 倍内存但会漏抓**——**对爬虫通常可接受**；**要精确就用 64 位哈希**。**注意**：**"Bloom 放在哈希集合前面"只省查找时间，不省内存**（**因为哈希集合还在**）。

### 6. 重试：退避 + 抖动

| 失败率 | 立即重试 3 次 | 指数退避 + 抖动 |
| --- | --- | --- |
| 10% | 1.11× | 1.03× |
| **30%** | **1.39×** | **1.10×** |

**读法**：**立即重试的问题是"同时"**——**所有失败请求在同一时刻重试，把对端的短暂故障变成雪崩**。**退避 + 抖动**：

$$t_{\text{retry}}=b\times2^{n}\times(1+\text{rand}(0,1))$$

**读法**：**抖动是关键**（**否则所有客户端仍在同一时刻重试**）。

### 7. 边界与终止

| 限制 | 为什么必须 |
| --- | --- |
| **最大深度** | 否则会无限深入 |
| **最大页数** | 否则不会停 |
| **总超时** | 否则卡死 |
| **单请求超时** | 否则慢站点拖住 worker |
| **响应体大小上限** | 否则一个大文件吃光内存 |
| **只抓 HTML**（按 Content-Type 过滤） | 否则会下载二进制 |
| **重定向上限** | 否则重定向循环 |

**读法**：**"爬虫不会停"是最常见的 bug**——**这七个限制是"能停下来"的保证**。

## 数值与代码验证

### 表 1：并发内存、礼貌吞吐、退避、去重（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

> **说明**：本题要求用 Go 实现，但本仓库的可运行验证统一用 Python（便于在本机执行）。
> 下面的代码**模拟爬虫的调度与限速**，用于验证"并发有界 / 每域限速 / 退避 / 去重"四个量；
> **Go 的 channel 与 goroutine 语义在正文中以代码片段给出**。

```python
# 爬虫的四个量化要点：并发内存、每域限速的吞吐、退避、去重内存
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

print("① 并发有界：每 URL 一个 goroutine vs worker 池（goroutine 初始栈按 8 KB 估）")
print(f"  {'URL 数':>10} {'无界 goroutine':>16} {'worker 池(100)':>15} 说明")
for n in (10_000, 100_000, 1_000_000):
    unbounded = n * 8 / 1e6
    pool = 100 * 8 / 1e6
    note = "可接受" if unbounded < 1 else ("**危险**" if unbounded < 10 else "**必然 OOM**")
    print(f"  {n:>10,} {unbounded:>14.1f} GB {pool:>13.3f} GB {note}")
print("  读法：**无界 goroutine 的内存随 URL 数线性增长** -> 固定 worker 池 + **有界队列（背压）**")

print("")
print("② 礼貌限速：总吞吐 ≈ 域数 x 每域速率（限速器必须按域）")
print(f"  {'域数 H':>8} {'每域 1 req/s':>13} {'全局 1 req/s':>13} {'倍数':>7}")
for h in (1, 10, 100, 1_000, 10_000):
    print(f"  {h:>8,} {h:>11.0f}/s {1:>11.0f}/s {h:>6.0f}x")
print("  读法：**礼貌是「按域」的，所以吞吐随域数增长** -> 真正冲突的是「同一个站抓太多」；")
print("        注意 per-host 限速器要清理（否则域 map 无限增长 = 内存泄漏），robots.txt 也要按域缓存")

print("")
print("③ 重试：立即重试 vs 指数退避 + 抖动（瞬时负载倍数）")
print(f"  {'失败率':>8} {'立即重试 3 次':>14} {'退避 + 抖动':>12} {'倍数':>7}")
for p in (0.05, 0.1, 0.3):
    immediate = 1 + p + p * p
    backoff = 1 + p / 3
    print(f"  {p:>8.0%} {immediate:>12.2f}x {backoff:>10.2f}x {immediate/backoff:>6.2f}x")
print("  读法：**立即重试的问题是「同时」**（把对端短暂故障变雪崩）-> 退避 + **抖动**（否则仍同时）")

print("")
print("④ 去重：内存 vs 正确性（1 亿 URL）")
@dataclass
class Dedup:
    name: str
    mem_gb: float
    correctness: str
DEDUPS = [
    Dedup("字符串集合", 10.0, "精确（可直接查原始 URL）"),
    Dedup("64 位哈希集合", 0.8, "精确（碰撞概率可忽略）"),
    Dedup("Bloom 过滤器（1%）", 0.12, "**近似：约 1% 的新 URL 被误判 -> 漏抓**"),
    Dedup("Bloom + 哈希集合", 0.92, "精确 + 快速路径（Bloom 命中才查集合）"),
]
print(f"  {'做法':<22} {'内存':>9} 正确性")
for d in DEDUPS:
    print(f"  {d.name:<22} {d.mem_gb:>7.2f} GB {d.correctness}")
print("  读法：**Bloom 省 6.7 倍内存但会漏抓**（爬虫通常可接受）；要精确就用 64 位哈希；")
print("        **Bloom 放在哈希集合前面只省查找时间、不省内存**")

print("")
print("⑤ Go 的三个坑与正确做法")
@dataclass
class Pitfall:
    name: str
    fix: str
PITFALLS = [
    Pitfall("send on closed channel", "只由「最后一个发送者」关闭；多生产者用 wg.Wait() 后单独 close"),
    Pitfall("goroutine 泄漏", "所有阻塞点都监听 ctx.Done()（队列/HTTP/限速器等待）"),
    Pitfall("sync.Map 不是万能", "写多读少用分片锁 map；sync.Map 适合读多写少、键集合稳定"),
]
print(f"  {'坑':<26} 正确做法")
for p in PITFALLS:
    print(f"  {p.name:<26} {p.fix}")
print("  读法：**channel 所有权是 Go 面试的经典考点** —— 说清「谁 close」比写出代码更重要")

print("")
print("⑥ 必须设的七个上限（否则爬虫不会停）")
LIMITS = ["最大深度", "最大页数", "总超时", "单请求超时", "响应体大小上限",
          "只抓 HTML（按 Content-Type 过滤）", "重定向次数上限"]
for i, l in enumerate(LIMITS, 1):
    print(f"  {i}. {l}")
print("  读法：**「爬虫不会停」是最常见的 bug** —— 这七个限制是「能停下来」的保证")

print("")
print("⑦ 简单模拟：每域限速下的聚合吞吐（100 个 worker、每域 1 req/s、目标 500 个域）")
def simulate(n_hosts: int, per_host_rate: float, workers: int, total_reqs: int,
             seed: int = 0) -> Tuple[float, int]:
    import heapq
    import random
    rng = random.Random(seed)
    next_free = {h: 0.0 for h in range(n_hosts)}
    heap: List[Tuple[float, int]] = [(0.0, h) for h in range(n_hosts)]
    heapq.heapify(heap)
    done = 0
    now = 0.0
    while done < total_reqs and heap:
        t, h = heapq.heappop(heap)
        now = max(now, t)
        done += 1
        heapq.heappush(heap, (now + 1.0 / per_host_rate, h))
    return now, done
t, done = simulate(500, 1.0, 100, 2000)
per_host = math.ceil(2000 / 500)
print(f"  500 个域、每域 1 req/s、2000 个请求 -> 最后一个请求在 {t:.1f}s 完成")
print(f"  （每域 {per_host} 个请求：第 1 个在 t=0、之后每 1s 一个 -> 最后一个在 {per_host-1:.0f}s）")
print(f"  聚合吞吐 = 500 域 x 1 req/s = **500 req/s**（是「每域 1 req/s」的 {500/1:.0f} 倍）")
print("  读法：**聚合吞吐 = 域数 x 每域速率**，与 worker 数无关（只要 worker 够多）；")
print("        **worker 数决定「能否同时占用这么多域」** —— 所以两者要匹配")
```

预期输出要点（实跑）：① **并发内存**：100 万 URL 无界 goroutine **8.0 GB** vs worker 池 **0.001 GB**；② **礼貌吞吐**：域数 1,000 时每域限速可达 **1,000/s**（全局限速只有 1/s，**1,000×**）；③ **退避**：失败率 30% 时立即重试负载 **1.39×** vs 退避 **1.10×**；④ **去重内存**（1 亿 URL）：字符串 10.0 GB、**64 位哈希 0.8 GB**、**Bloom 0.12 GB（漏抓 1%）**；⑤ 三个 Go 坑与正确做法；⑥ 七个必须设的上限；⑦ **每域限速模拟**：500 域 / 2000 请求 → 最后一个请求在 **3.0 s** 完成，**聚合吞吐 500 req/s**（是每域速率的 500 倍）。

## 常见追问

- **追问**：怎么保证不重复抓同一个页面？
  - 要点：**两道防线**：① **URL 规范化**（小写、去默认端口、去 fragment、解析 `..`、统一尾斜杠）——**在入队前做**；② **去重集合**（哈希或 Bloom）。**注意**：**规范化不足会重复抓**（**同一页面的多种 URL 形式**），**规范化过度会漏抓**（**把不同页面合并**）——**所以"查询参数排序"这类规则要谨慎**（**有些站点的参数顺序有意义**）。**第三道防线**（可选）：**内容指纹**（**同一内容不同 URL 的情况**——例如打印版与普通版）。
- **追问**：怎么处理 robots.txt？
  - 要点：**三条**：① **按域缓存**（**每个域拉一次，不要每个请求都拉**）；② **遵守 `Disallow` 与 `Crawl-delay`**（**后者直接决定每域速率**）；③ **失败时的策略**（**robots 拉不到时，保守做法是"按最严格处理"或"跳过该域"**——**取决于合规要求**）。**读法**：**robots 是"礼貌"的正式约定**——**面试时主动提到它是加分项**（**说明你知道爬虫有合规面**）。
- **追问**：如果某个域响应很慢怎么办？
  - 要点：**三层**：① **单请求超时**（**用 `context.WithTimeout`**）——**否则一个慢域会占住 worker**；② **每域并发上限**（**不只是速率，还有"同时在途的请求数"**）——**避免对同一域开太多连接**；③ **降级/隔离**（**连续失败就把该域降到低频队列，或标记为"暂不抓"**）。**读法**：**"慢域会拖垮整个爬虫"**——**因为 worker 是共享资源**，**所以隔离是必需的**。
- **追问**：怎么知道爬虫"跑得好"？
  - 要点：**六个指标**：① **吞吐**（页/秒）；② **队列深度**（**持续增长说明消费不足**）；③ **每域速率**（**验证礼貌约束生效**）；④ **去重命中率**（**规范化质量**）；⑤ **失败率与重试率**（**按域分解**）；⑥ **覆盖率**（**在目标域内抓到了多少比例**）。**读法**：**"队列深度"是最重要的健康指标**——**它同时反映"生产 vs 消费"的平衡与背压是否生效**。
- **追问**：Go 里 `sync.Map` 和"互斥锁 + map"怎么选？
  - 要点：**按访问模式**：① **`sync.Map` 适合"读多写少、键集合基本稳定"**（**例如"已抓 URL 集合"其实写很多 → 不合适**）；② **"写多"时分片锁 map 更快**（**$N$ 个分片，每片一把锁**——**减少竞争**）；③ **判据**：**如果每个键只写一次然后反复读，`sync.Map` 有优势；如果键反复写，用分片 map**。**读法**：**爬虫的"已抓集合"是"写多读少"**（**每个 URL 写一次、查一次**）——**所以分片 map 更合适**（**或者用现成的并发 set 库**）。
- **追问**：这道题在生产上还差什么？
  - 要点：**六项**：① **持久化**（**队列与去重集合要能重启恢复**——**否则重启就从头再来**）；② **分布式**（**多机分域抓取 + 中央去重**）；③ **反爬对抗**（**UA、代理、验证码——这里要谨慎，可能涉及合规**）；④ **内容处理**（**编码检测、压缩、非 HTML 过滤**）；⑤ **合规**（**robots、ToS、个人信息、版权**）；⑥ **可观测性**（**上面的六个指标**）。**读法**：**"简单爬虫"与"生产爬虫"的差距主要在前两项**（**持久化与分布式**）——**面试时点出来即可，不必展开**。

## 相关题目

- [[openai-02]]：一步步实现一个 ORM——同一家公司的"从零构建"题（**"每级可测"的方法一致**）。
- [[openai-18]]：webhook 投递系统——**重试、退避与幂等**的通用版本（**比爬虫更严格**）。
- [[coding-10]]：并发模式与 worker 池——Go 并发的通用版本。
- [[coding-11]]：限速器与令牌桶——**每域限速**的通用实现。
- [[coding-12]]：URL 规范化与去重——本篇第 5 节的展开。

## 参考资料与归属

- **The Go Programming Language Specification（延伸）** —— The Go Authors，2023-01-01：<https://go.dev/ref/spec>。**channel 的关闭语义（"close 由发送方负责"）与 `select` 的语义**是第 3、4 节的依据。
- **Go Concurrency Patterns: Pipelines and Cancellation（延伸）** —— Sameer Ajmani (Google)，2014-03-13：<https://go.dev/blog/pipelines>。**pipeline 模式、`context` 取消、以及"由最后一个发送者关闭 channel"的标准写法**来自这篇——**本篇的骨架直接源于此**。
- **Robots Exclusion Protocol（延伸）** —— Koster et al. (IETF)，2022-09-01：<https://www.rfc-editor.org/rfc/rfc9309.html>。**`Disallow` 与 `Crawl-delay` 的语义**是第 2 节"每域限速"的规范依据。
- **一步步实现一个 ORM（本仓库公司题库 · OpenAI 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。该篇的**"每级可独立测试"**与本篇的"六个量化要点各自可验证"是同一套方法。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（goroutine 初始栈 8 KB、URL 数 1 万–100 万、worker 数 100、域数 1–10,000、每域 1 req/s、失败率 5%–30%、URL 平均 100 字节、64 位哈希 8 字节、Bloom 的 9.585 bit/项、模拟的 500 域 / 2000 请求）都是为演示设计取舍而构造的**示例参数与显式假设**；**内存与吞吐的计算、退避的负载倍数、Bloom 的位数都是可复现的算术**，而**具体数值依赖真实目标站与网络**。**"goroutine 初始栈 8 KB"是量级参考**（**Go 运行时的初始栈大小随版本变化，且会按需增长**）；**"退避负载 = $1+p/3$"是简化模型**（**真实退避的时间分布依赖退避参数**）。**模拟只验证"聚合吞吐 = 域数 × 每域速率"这一结构**，**未模拟网络延迟与失败**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
