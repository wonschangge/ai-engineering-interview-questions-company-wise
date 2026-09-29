---
type: question
id: together-01
company: Together AI
topic: coding
order: 1
question: 为流式 token 生成编写服务端 handler，并正确处理客户端断开连接的情况。
question_en: Write a server-side handler for streaming token generation that handles client disconnects correctly.
asked_at: []
level: 进阶
tags: [流式, SSE, 断连处理, 取消, 幂等]
sources:
  - title: Token Streaming 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-token-streaming-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: Handling Overload（Google SRE Book 第 21 章）（延伸）
    url: https://sre.google/sre-book/handling-overload/
    author: Google SRE
    published: 
related: [together-02, together-03, anthropic-17, inference-serving-02, together-06]
updated: 2026-09-28
---

## 一句话答案

> 流式 handler 的难点**不在「把 token 发出去」，而在「客户端走了之后怎么办」**。七个必须处理的点：
> ① **断开检测**：客户端断开在服务端表现为**写失败**（`BrokenPipe`/`ConnectionReset`）或框架的 `request.is_disconnected()`——**必须检测并向上游传播取消**，否则 GPU 会继续为一个没人看的回答烧算力；
> ② **取消传播**：取消要一路传到推理引擎（停止 decode、释放 KV cache）。**没有取消的流式服务 = 按接受量付费却按生成量烧钱**；
> ③ **背压**：客户端读得慢（手机弱网）时不能让服务端无限缓冲——要有**有界队列**，满了要么丢帧（不可接受）要么暂停生成（正确做法）；
> ④ **心跳**：长回答中间可能几十秒没有 token（首 token 前的排队/prefill），中间层（LB/代理）会按空闲超时断连——必须发 `ping` 事件保活；
> ⑤ **结束语义**：`done` 事件必须必达（含 `finish_reason` 与 usage）；仅靠「连接关闭」作为结束信号会让客户端无法区分「正常结束」与「断线」；
> ⑥ **续传与幂等**：每个 chunk 带单调 `seq`；客户端重连时上报已收到的最大 `seq`，服务端从那里续（或明确告知不可续，让客户端重放）；
> ⑦ **计费与账务**：已生成的 token 仍要计费（与用户是否看到无关），但**取消后的 token 应当少收或不收**——这是产品决策，必须在协议与文档里写清。
> 一句话判据：**流式的正确性 = 取消是否真的停住了 GPU + 结束是否可区分 + 慢客户端是否被背压住**。

## 面试官在考什么

- **断开检测是否真的实现**：很多人只写 `for token in stream: yield token`——断连时循环继续跑，GPU 白烧。能否写出「检测 + 取消 + 释放」的完整链路是本题的分水岭。
- **取消的语义**：取消是「停止生成」还是「暂停」？已生成部分要不要保留？KV cache 立即释放还是等超时？能否说清代价（重算 vs 显存）。
- **背压**：是否意识到「客户端慢」会反压到服务端内存；有界队列 + 暂停生成 vs 丢弃的区别。
- **SSE 协议细节**：`event:`/`data:` 格式、心跳、`Content-Type: text/event-stream`、禁用缓冲（`X-Accel-Buffering: no` 或等价的代理配置）。
- **异常路径**：上游推理失败如何在一个已经开始流的响应里表达（不能改 HTTP 状态码了——必须用**流内错误事件**）。这是最容易被忽略、也最能体现经验的一点。
- **资源与计费**：取消后 KV 释放、连接池回收、以及「已生成 token 是否计费」的产品口径。
- **可测试性**：怎么测断连（客户端中途关闭、慢读、代理超时）——能否给出可复现的测试方法。

**常见错误答案**

- 只写生成循环，不检测断开。
- 用 `Content-Length` 或缓冲整个响应再发（那就不是流式了）。
- 流已经开始后还用 HTTP 状态码表达错误（已发送 200 就无法改）。
- 无界缓冲（弱网客户端把服务端内存吃光）。
- 不发心跳，导致长 prefill 被 LB 断开。
- 把「连接关闭」当作正常结束信号。

## 原理与推导

### 1. 状态机（服务端视角）

```
ACCEPTED → PREFILLING → DECODING → DONE
                │            │
                └────────────┴──→ CANCELLED（客户端断开 / 超时 / 用户取消）
                                   └─ 释放 KV、停止 decode、记录已生成 token 数
```

**关键**：`CANCELLED` 是一个**正常终态**（不是异常），要打点、要释放资源、要按策略计费。把取消当异常处理会导致资源泄漏。

### 2. 断开检测的三种技术路径

| 路径 | 做法 | 适用 |
| --- | --- | --- |
| 写失败 | 捕获 `BrokenPipeError` / `ConnectionResetError` | 通用（HTTP/1.1） |
| 框架钩子 | `request.is_disconnected()`（如 Starlette） | ASGI 框架 |
| 心跳探活 | 定期发 `ping`，写失败即判定断开 | 长静默期（长 prefill） |

**注意**：TCP 断开不一定立刻可见（半开连接）；因此**心跳 + 写失败**要一起用，并且给「静默超过 N 秒」兜底。

### 3. 取消传播到推理引擎

推理引擎需要暴露**按请求取消**的接口（vLLM 类引擎的 `abort/stop` 语义）。取消时要做三件事：
1. 从调度器的活跃序列里移除（停止后续 decode）；
2. 释放该序列的 KV 块（否则显存泄漏，最终导致整机 OOM）；
3. 记录指标（取消率、取消时的已生成 token 数）——**取消率是容量规划的重要输入**（高取消率意味着有效吞吐低于名义吞吐）。

**代价**：如果用户只是「网络抖动」后重连，取消意味着**重算 prefill**；所以产品上常见两种策略：
- **立即取消**（省算力，用户体验差）；
- **保活窗口**（例如断连后保留 KV 30 秒，重连可续）——显存换体验，需要显式配置上限。

### 4. 背压：慢客户端的正确处理

$$B_{\text{buffer}}\ \text{有限};\quad \text{队列满}\Rightarrow\text{暂停生成（而非丢弃）}$$

丢弃 token 会让输出**语义错乱**（用户看到跳字的回答），所以正确做法是**让生成等一下**：把「发送」变成拉取式（生成器 yield 前等待队列有空间）。但**暂停不能无限**——如果客户端长时间不读，就回到取消路径（断开它）。

### 5. 流内错误：已经开始流之后怎么报错

HTTP 状态码在第一个字节发出后就不能改了，所以必须定义**流内错误事件**：

```
event: error
data: {"code": "upstream_failed", "message": "...", "retryable": true, "seq": 128}
```

**要求**：① 错误事件后必须**关闭流**（否则客户端会等）；② 必须带 `seq`，让客户端知道已收到多少内容；③ 对**不可重试**错误（如内容策略）与可重试错误（上游超时）区分 `retryable`（串 [[anthropic-28]] 的错误分类思路）。

### 6. 续传的两种策略

| 策略 | 服务端成本 | 客户端复杂度 | 适用 |
| --- | --- | --- | --- |
| 服务端缓冲最近 N 个 token | 内存（每连接 N 个 token） | 低（上报 seq 即可） | 交互式产品 |
| 不支持续传（重放整个请求） | 无 | 中（要处理重复内容） | 短回答 |

**注意**：续传必须**幂等**——同一次生成的续传不能导致重复计费（用请求 id + seq 记录已计费区间）。

## 数值与代码验证

### 表 1：断连处理的成本量级（示例）

| 场景 | 无取消 | 有取消 |
| --- | --- | --- |
| 用户开始生成后 2 秒关闭页面，模型仍生成 500 token | 白烧 500 token 的算力 | 停止，仅已生成的 ~40 token 计费 |
| 100 万请求/日、取消率 8%、平均浪费 300 token | 每天浪费 2,400 万 token | 接近 0 |
| 折算成本（\$3/百万 token 口径） | 约 \$72/日 ≈ \$2.2 万/年 | — |

### 表 2：三种断开检测路径的覆盖

| 路径 | 能发现问题 | 延迟 | 备注 |
| --- | --- | --- | --- |
| 写失败 | 大多数断开 | 下一个 token（毫秒级） | 依赖真的去写 |
| 框架钩子 | 优雅关闭 | 轮询间隔 | 与框架耦合 |
| 心跳 | 半开连接、代理超时 | 心跳间隔（如 15 s） | 保活与检测一体 |

### 可运行代码

```python
# 流式 handler 的四个关键行为：断连取消、有界背压、心跳保活、流内错误
import asyncio, time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

class ClientGone(Exception):
    """客户端断开（真实实现里由写失败/框架钩子触发）"""

class UpstreamFailed(Exception):
    pass

@dataclass
class Engine:
    """模拟推理引擎：暴露按请求取消（真实实现里是 abort/stop）"""
    total_tokens: int = 500
    token_ms: float = 12.0
    generated: dict = field(default_factory=dict)
    cancel_calls: List[str] = field(default_factory=list)     # 记录引擎收到的取消
    fail_after: Optional[int] = None                          # 在第 N 个 token 后抛错
    prefill_s: float = 0.0                                    # 首 token 前的静默期（prefill/排队）
    def __post_init__(self):
        self._cancelled = set()
    async def stream(self, req_id: str) -> AsyncIterator[str]:
        if self.prefill_s:                                        # 模拟首 token 前的静默
            await asyncio.sleep(self.prefill_s)
        for i in range(self.total_tokens):
            if req_id in self._cancelled:
                return                                        # 停止 decode（真实实现还要释放 KV）
            await asyncio.sleep(self.token_ms / 1000 / 200)    # 演示用加速 200 倍
            if self.fail_after is not None and i == self.fail_after:
                raise UpstreamFailed("upstream_timeout")
            self.generated[req_id] = i + 1
            yield f"tok{i} "
    def cancel(self, req_id: str) -> None:
        # ★ 关键：取消必须由引擎自己记录（不能依赖生成器下次被调度，因为它可能已被 cancel）
        self._cancelled.add(req_id)
        self.cancel_calls.append(req_id)

@dataclass
class Sender:
    """模拟客户端写：可注入断开时刻与慢读"""
    disconnect_after_s: Optional[float] = None
    slow_after_s: Optional[float] = None
    slow_factor: float = 30.0
    buffer_capacity: int = 8
    sent: int = 0
    t0: float = field(default_factory=time.perf_counter)
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
    async def write(self, chunk: str) -> None:
        if self.disconnect_after_s is not None and self.elapsed() > self.disconnect_after_s:
            raise ClientGone()
        self.sent += 1
        if self.slow_after_s is not None and self.elapsed() > self.slow_after_s:
            await asyncio.sleep(0.02 * self.slow_factor)       # 慢客户端：每次写都卡

async def serve_stream(engine: Engine, req_id: str, sender: Sender,
                       heartbeat_s: float = 0.05, max_silence_s: float = 0.5) -> dict:
    stats = {"sent": 0, "cancelled": False, "heartbeats": 0, "error": None,
             "max_queue": 0}
    queue: asyncio.Queue = asyncio.Queue(maxsize=sender.buffer_capacity)
    async def producer():
        try:
            async for tok in engine.stream(req_id):
                await queue.put(("tok", tok))   # 有界队列：满则暂停生成（背压）
            await queue.put(("done", None))
        except asyncio.CancelledError:
            pass
        except Exception as exc:             # ★ 上游异常必须显式传给消费者
            await queue.put(("error", exc))
    prod = asyncio.create_task(producer())
    last_activity = time.perf_counter()
    try:
        while True:
            stats["max_queue"] = max(stats["max_queue"], queue.qsize())
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                await sender.write(": ping\n\n")          # 心跳：长静默期保活
                stats["heartbeats"] += 1
                if time.perf_counter() - last_activity > max_silence_s:
                    raise ClientGone()
                continue
            if kind == "done":
                await sender.write("event: done\ndata: {}\n\n")   # 结束事件必达
                break
            if kind == "error":
                raise payload                                # 交给下面的异常分支处理
            await sender.write(f"data: {payload}\n\n")
            stats["sent"] += 1
            last_activity = time.perf_counter()
    except ClientGone:
        stats["cancelled"] = True
        engine.cancel(req_id)                                # ★ 取消传播到引擎
    except Exception as exc:
        stats["error"] = type(exc).__name__
        try:
            await sender.write(f'event: error\ndata: {{"code": "{type(exc).__name__}",'
                               f' "retryable": true, "seq": {stats["sent"]}}}\n\n')
        except ClientGone:
            stats["cancelled"] = True
            engine.cancel(req_id)
    finally:
        prod.cancel()
        try:
            await prod
        except BaseException:
            pass
    return stats

async def main():
    print("① 客户端断开 -> 取消传播（引擎收到取消，生成提前停止）")
    eng = Engine()
    st = await serve_stream(eng, "r1", Sender(disconnect_after_s=0.05))
    print(f"  发送 {st['sent']} 个 chunk，取消={st['cancelled']}，"
          f"引擎已生成 {eng.generated.get('r1')} 个 token（上限 500）")
    print(f"  引擎收到的取消请求：{eng.cancel_calls}")
    print("  说明：真实实现里引擎收到取消还要释放该序列的 KV 块，否则显存会被泄漏的序列吃光")

    print("\n② 正常完成 -> done 事件必达")
    eng2 = Engine(total_tokens=40)
    st2 = await serve_stream(eng2, "r2", Sender())
    print(f"  发送 {st2['sent']} 个 chunk，取消={st2['cancelled']}，"
          f"已生成 {eng2.generated.get('r2')} 个 token（完整 40）")

    print("\n③ 慢客户端 -> 有界队列造成背压（生成被暂停而不是丢弃）")
    eng3 = Engine(total_tokens=200)
    t0 = time.perf_counter()
    st3 = await serve_stream(eng3, "r3", Sender(slow_after_s=0.01, slow_factor=30))
    dt = (time.perf_counter() - t0) * 200                     # 折算回真实时间
    print(f"  发送 {st3['sent']} 个 chunk（无丢弃），队列峰值深度 {st3['max_queue']}"
          f"（上限 {Sender().buffer_capacity}）")
    print(f"  真实口径耗时约 {dt:.0f} ms（远长于不限速的 {200*12:.0f} ms）"
          f"，心跳 {st3['heartbeats']} 次")
    print("  读法：慢客户端被背压拖慢而不是丢 token（丢 token 会让语义错乱）；")
    print("        但沉默超过上限仍要走取消路径，否则连接会永久占一个槽位")

    print("\n④ 上游失败 -> 流内 error 事件（HTTP 状态码已经发出去了）")
    st4 = await serve_stream(Engine(fail_after=1), "r4", Sender())
    print(f"  结果：sent={st4['sent']}  error={st4['error']}  cancelled={st4['cancelled']}")
    print("  读法：错误事件必须带 seq 与 retryable，客户端据此决定续传还是重放")

    print("\n⑤ 长 prefill（首 token 前静默）-> 心跳保活，避免被中间层按空闲超时断开")
    st5 = await serve_stream(Engine(total_tokens=5, prefill_s=0.2), "r5", Sender(),
                             heartbeat_s=0.05, max_silence_s=0.5)
    print(f"  静默期发出心跳 {st5['heartbeats']} 次，最终发送 {st5['sent']} 个 chunk，"
          f"取消={st5['cancelled']}")
    print("  读法：没有心跳，长 prefill 会被 LB/代理当成空闲连接断开，用户看到的是「请求失败」")

    print("\n⑥ 成本对照：有取消 vs 无取消（100 万请求/日、取消率 8%、平均浪费 300 token）")
    daily_requests, cancel_rate, wasted = 1_000_000, 0.08, 300
    for label, tokens in (("无取消", daily_requests * cancel_rate * wasted), ("有取消", 0)):
        print(f"  {label:<6} 每日浪费 {tokens:>12,.0f} token  "
              f"≈ ${tokens/1e6*3:>9,.0f}/日  ≈ ${tokens/1e6*3*365:>11,.0f}/年")
    print("  读法：取消不是体验优化，是单位经济性的一部分")

asyncio.run(main())
```


预期输出要点（实跑）：① 客户端断开后**引擎收到取消**（`cancel_calls` 里有该请求 id）且**生成提前停止**——引擎侧只生成了远少于 500 个 token；② 正常路径下 `done` 事件必达、引擎完整生成 40 个；③ 慢客户端触发**有界队列背压**：200 个 chunk **一个都没丢**，队列峰值深度被限制在容量内（8），代价是耗时被拖长——这正是「慢客户端不能丢 token、但必须被背压住」的实证；④ 上游失败走进**流内 error 事件**路径（`error=UpstreamFailed`、`sent=1`），因为 HTTP 状态码已经发出去了；⑤ 长 prefill 的静默期**发出 4 次心跳**并最终正常完成——这正是「没有心跳就会被 LB 当空闲连接掐断」的反证；⑥ 成本对照显示 8% 取消率下无取消机制每天白烧约 **2,400 万 token**（约 \$72/日、\$2.6 万/年）。

## 常见追问

- **追问**：如何区分「客户端断线」与「网络抖动」？
  - 要点：无法在服务端 100% 区分。工程做法是**保活窗口**：断连后保留 KV $T$ 秒（例如 30 s），窗口内重连可续，超时则取消并释放。窗口大小是显存与体验的取舍，要可配置并监控。
- **追问**：取消后 KV cache 什么时候释放？
  - 要点：立即从活跃序列移除（停止 decode）是可选的，但**KV 块的释放可以延迟**到保活窗口结束（因为要支持续传）；关键是**有上限**（每序列、每租户），否则高取消率下显存会被「等待重连」的序列吃光。
- **追问**：为什么不用 WebSocket？
  - 要点：SSE 足够（单向流）且更简单（HTTP 语义、易过代理、易做鉴权）；WebSocket 适合双向交互（语音 agent 的双向音频、工具调用的中途确认）。选型看是否需要**双向**（串 [[elevenlabs-04]] 的语音场景）。
- **追问**：怎么测试断连处理？
  - 要点：① 单元级——注入「写失败」的假 sender，断言引擎收到取消；② 集成级——用真实 HTTP 客户端中途 `close()`，断言引擎侧指标（活跃序列数归零）；③ 慢读测试——限速客户端，断言内存与队列不增长；④ 代理超时测试——让静默期长于 LB 空闲超时，断言心跳保活生效。
- **追问**：流式与计费怎么对齐？
  - 要点：**按已生成 token 计费**（无论客户端是否收到）；取消时可选「部分计费」策略，但必须在文档与协议里写清；对**续传**要用请求 id + seq 去重，避免重复计费。
- **追问**：如果推理引擎不支持取消怎么办？
  - 要点：至少做到**停止发送 + 丢弃后续输出**（省网络与内存），但与「停止生成」相比仍浪费 GPU；此时应把取消能力作为选型要求，或在上层做限额（限制最大生成长度）来兜底。

## 相关题目

- [[together-02]]：continuous batching 调度器——取消在调度器里表现为「从活跃序列移除并释放 KV」。
- [[together-03]]：投机解码——它改变的是每步延迟，与流式的取消/背压正交但会叠加。
- [[anthropic-17]]：服务栈的准入控制与降级，与本题的取消/背压同属资源治理。
- [[inference-serving-02]]：连续批处理与调度策略，是本题推理侧的基础。
- [[together-06]]：serverless 推理平台，取消与配额在其上要跨租户治理。
- [[anthropic-28]]：错误分类与重试语义，与本题「流内 error 事件」同一套设计思路。

## 参考资料与归属

- **Token Streaming 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-token-streaming-work>。第 1 节 SSE 分帧、心跳与结束事件的工程背景参照这篇。
- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 3 节「取消时要释放 KV 块」的依据来自这篇的显存管理机制。
- **Handling Overload（Google SRE Book 第 21 章）（延伸）** —— Google SRE：<https://sre.google/sre-book/handling-overload/>。第 4 节「背压与过载时宁可拒绝也不要无限排队」的取向来自这一章。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（500 token 上限、12 ms/token、断连时刻 0.15 s、队列容量 16、心跳 0.2 s、沉默上限 1 s、100 万请求/日、取消率 8%、浪费 300 token、\$3/百万 token）都是按本仓库统一口径构造的**演示参数与显式假设**；代码为了便于运行把时间**加速了 50 倍**，因此输出中的绝对耗时不是真实延迟，只用来看行为关系。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
