---
type: question
id: elevenlabs-01
company: ElevenLabs
topic: coding
order: 1
question: 写一个把流式 TTS 代理到浏览器的服务，并能在用户离开页面时干净利落地取消。
question_en: Write a service that proxies streaming TTS to a browser and cancels cleanly when the user leaves the page.
asked_at: []
level: 进阶
tags: [流式, TTS, 二进制音频, 背压, 取消, 浏览器播放]
sources:
  - title: 设计实时语音 AI Agent
    url: https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent
    author: Amit Shekhar (Outcome School)
    published: 
  - title: 解释流式 TTS 的 chunking 与 jitter buffer 大小设定（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 为实时语音 agent 做 latency 预算（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [together-01, multimodal-08, multimodal-03, elevenlabs-02, elevenlabs-04]
updated: 2026-09-28
---

## 一句话答案

> 与「代理 token 流」相比，代理**音频流**多了四件事，这也是本题的四个考点：
> ① **二进制帧而不是文本事件**：音频是连续字节流（例如 PCM 16-bit/24 kHz 单声道 = **48 KB/s**），用 `Transfer-Encoding: chunked` 或 WebSocket 传二进制，**不要塞进 SSE 的 base64**（base64 膨胀 33% 且增加编解码延迟）；
> ② **首包要够「可播」**：浏览器不能播 5 ms 的碎片——**首块通常要 100–200 ms 音频**（一个可解码的帧/子句），所以 TTS 侧要有「最小首块」策略，代理侧不能把首块切得更碎；
> ③ **播放侧缓冲（jitter buffer）**：网络抖动 + TTS 生成抖动都要靠客户端缓冲吸收；**缓冲大小是抖动分布的分位数**（例如 p99），不是拍一个常数——缓冲太小会卡顿，太大则首音变慢；
> ④ **取消要「三处同时干净」**：浏览器停止播放并清空缓冲、代理**关闭到 TTS 的上游连接**、TTS 侧停止合成（否则白烧 GPU）。用户离开页面时触发的是 `pagehide`/`visibilitychange` 或 WebSocket 关闭——**服务端必须把它当成正常终态而不是异常**。
> 一句话判据：**代理音频流的正确性 = 首块够大可播 + 抖动被缓冲吸收 + 取消真的传导到 TTS**。

## 面试官在考什么

- **是否区分文本流与音频流**：能否指出音频需要**二进制帧、固定采样率与帧长**，以及「时间轴」是音频协议的隐含部分（每个 chunk 对应多少毫秒）。
- **首块策略**：能否说出「为什么不能把首包切碎」（解码器需要最小样本数、播放器需要可调度缓冲），以及**首块大小与 TTFA 的权衡**（首块越大 TTFA 越晚，但越不容易卡）。
- **背压与缓冲**：客户端读不过来时（弱网、后台标签页）代理要不要缓冲？缓冲多少？**有界 + 暂停上游**比无限堆积正确（与 [[together-01]] 同源，但音频还多一个「播放时钟」约束）。
- **取消链路**：能否写出「浏览器 → 代理 → TTS」三级的取消传播，以及**取消时的资源释放**（连接、缓冲、GPU）。
- **协议选择**：SSE（文本）vs chunked HTTP（二进制）vs WebSocket（双向，例如 barge-in 需要）；能否说出各自的适用场景与中间层（代理、CDN）的缓冲坑（`X-Accel-Buffering` 之类）。
- **浏览器播放实现**：MediaSource Extensions、WebAudio（`AudioBufferSourceNode`）或 `<audio>` + MSE 的取舍；**自动播放策略**（需要用户手势）与**标签页后台时的时钟漂移**。
- **可测性**：怎么测取消（页面关闭、网络断开、切后台）与卡顿（注入抖动）。

**常见错误答案**

- 用 SSE + base64 传音频（膨胀 33%，且事件边界与音频帧边界混淆）。
- 把首块切得和后续块一样小（播放器起不来，TTFA 反而更差）。
- 无界缓冲（弱网客户端把服务端内存吃光）。
- 只在客户端停止播放，不通知上游（TTS 继续合成，白烧 GPU）。
- 把 `pagehide` 当异常处理（不释放资源、记录成错误）。
- 忽略自动播放策略（首次播放被浏览器拦截）。

## 原理与推导

### 1. 音频的「带宽与时间」账

| 格式 | 码率 | 1 分钟音频 |
| --- | --- | --- |
| PCM 16-bit / 24 kHz / 单声道 | 384 kbps（48 KB/s） | 2.8 MB |
| PCM 16-bit / 44.1 kHz / 立体声 | 1,411 kbps（176 KB/s） | 10.6 MB |
| Opus 64 kbps | 64 kbps（8 KB/s） | 0.48 MB |
| MP3 128 kbps | 128 kbps（16 KB/s） | 0.96 MB |

**结论**：**原始 PCM 的带宽是文本的几百倍**——所以生产里常见「上游 PCM、下行 Opus」的转码策略（省带宽但增加几毫秒编码延迟），或者干脆下行也走 PCM（局域网/高质量场景）。**这个取舍要显式说出来**。

### 2. 首块（first chunk）与 TTFA

$$\text{TTFA}=\underbrace{t_{\text{文本}\to\text{首块}}}_{\text{TTS 生成}}+\underbrace{t_{\text{传输}}}_{\text{首块字节/带宽}}+\underbrace{t_{\text{解码+调度}}}_{\text{浏览器}}$$

- **首块越大**：$t_{\text{生成}}$ 越长（要等更多音频），但解码与调度更稳；
- **首块越小**：理论上 TTFA 更短，但**低于解码器/播放器的最小可播单元就播不出来**，实际体验更差。
**工程做法**：首块取「一个完整韵律单元」（子句）或固定 100–200 ms，后续块可以更小（20–100 ms）以保持连续。

### 3. jitter buffer 的大小

设端到端抖动的分布为 $J$（标准差 $\sigma$），要保证不卡顿的缓冲为

$$B\approx z_{1-q}\cdot\sigma\ (\text{目标分位数 }q),\quad \text{再加一个安全余量}$$

**例**：$\sigma=15$ ms、取 p99（$z=2.33$）→ $B\approx35$ ms；若 $\sigma=40$ ms 则 $B\approx93$ ms。**注意**：缓冲直接加到 TTFA 上（用户听到第一声的时间 = 首块到达 + 缓冲），所以「抗抖动」与「快」是同一条预算里的两件事（口径见 [[multimodal-08]] 与 [[multimodal-03]]）。

### 4. 背压：有界队列 + 暂停合成

$$\text{队列满}\Rightarrow\text{暂停上游（pull 模式）}\quad\text{而不是丢弃音频帧}$$

**为什么不能丢**：音频帧有严格时间顺序，丢帧 = 爆音/跳字；而丢弃**未合成的文本**是安全的（可以在恢复后继续）。所以背压要作用在**文本/合成请求**层面，而不是音频帧层面。

### 5. 三级取消传播

```
浏览器：pagehide / visibilitychange(hidden) / 用户点"停止" / WebSocket close
  ↓（关闭下游连接 / 发 cancel 帧）
代理：检测写失败或收到 cancel → 关闭到 TTS 的上游连接
  ↓（HTTP 连接关闭 / 显式 cancel API）
TTS：停止合成、释放音频缓冲与（若有）GPU 会话
```

**要点**：
- 代理必须**真的关闭上游连接**（而不是只停转发）——很多 TTS 服务在连接关闭后才停止合成；
- 用户切到后台不一定要取消（可能只是暂时不可见）——**产品决策**：切后台继续生成（回来能听）还是取消（省成本）。推荐：**切后台继续、页面关闭取消**；
- 取消是**正常终态**：要打点（取消率、取消时已生成音频秒数），用于容量与成本核算。

### 6. 协议选择

| 协议 | 适用 | 注意 |
| --- | --- | --- |
| chunked HTTP（二进制） | 单向播放（最常见） | 代理/网关要禁用缓冲，否则首块被攒住 |
| SSE（文本事件） | 需要带元数据的事件流 | 音频要 base64（膨胀 33%），不推荐 |
| WebSocket | 需要双向（barge-in、实时参数调整） | 需要心跳与重连语义 |

**中间层坑**：反向代理（Nginx 等）默认可能缓冲响应；必须显式关闭（否则「流式」变成「一次性」）。这一点在音频场景里尤其致命——**缓冲会让 TTFA 从 300 ms 变成 2 s**。

## 数值与代码验证

### 表 1：首块大小与 TTFA/卡顿的权衡（示例）

| 首块 | 生成等待 | 传输（48 KB/s） | 可播性 | 评价 |
| --- | --- | --- | --- | --- |
| 20 ms | 20 ms | 1 ms | 差（低于常见解码/调度单元） | 易卡 |
| **120 ms** | 120 ms | 6 ms | 好 | **推荐首块** |
| 500 ms | 500 ms | 25 ms | 很好 | TTFA 偏慢 |

### 表 2：缓冲与抖动（缓冲直接加到 TTFA）

| 抖动 $\sigma$ | p95 缓冲（$z$=1.64） | p99 缓冲（$z$=2.33） | TTFA 增量 |
| --- | --- | --- | --- |
| 10 ms | 16 ms | 23 ms | 小 |
| 25 ms | 41 ms | 58 ms | 中 |
| 40 ms | 66 ms | 93 ms | 明显 |

### 可运行代码

```python
# 流式 TTS 代理的四件事：二进制分帧、首块策略、有界背压、三级取消
# 用统一的虚拟时钟做确定性模拟（不依赖真实睡眠，耗时与阈值同量纲）
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SAMPLE_RATE = 24_000
BYTES_PER_SAMPLE = 2
BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE        # 48 KB/s

def ms_to_bytes(ms: float) -> int:
    return int(BYTES_PER_SEC * ms / 1000)

@dataclass
class Clock:
    """虚拟时钟：所有阈值与耗时都在同一时间轴上"""
    now_ms: float = 0.0
    def advance(self, ms: float) -> None:
        self.now_ms += ms

@dataclass
class TTSEngine:
    total_ms: float = 3000.0
    rtf: float = 0.15                     # 生成 1 s 音频需 0.15 s 计算（≈6.7x 实时）
    generated_ms: Dict[str, float] = field(default_factory=dict)
    cancel_calls: List[str] = field(default_factory=list)
    def __post_init__(self):
        self._cancelled = set()
    def generate(self, req_id: str, want_ms: float, clock: Clock) -> Optional[bytes]:
        """按 RTF 推进虚拟时间并产出一块音频；被取消则返回 None"""
        if req_id in self._cancelled:
            return None
        produced = self.generated_ms.get(req_id, 0.0)
        if produced >= self.total_ms:
            return None
        clock.advance(want_ms * self.rtf)
        got = min(want_ms, self.total_ms - produced)
        self.generated_ms[req_id] = produced + got
        return bytes(ms_to_bytes(got))
    def cancel(self, req_id: str) -> None:
        self._cancelled.add(req_id)
        self.cancel_calls.append(req_id)

@dataclass
class BrowserClient:
    leave_at_ms: Optional[float] = None
    slow_factor: float = 1.0              # >1 表示播放/读取比实时更慢（弱网）
    received_bytes: int = 0
    chunks: int = 0
    first_audio_ms: Optional[float] = None
    def write(self, data: bytes, clock: Clock) -> bool:
        """返回 False 表示连接已断（用户离开页面）"""
        if self.leave_at_ms is not None and clock.now_ms > self.leave_at_ms:
            return False
        self.received_bytes += len(data)
        self.chunks += 1
        if self.first_audio_ms is None:
            self.first_audio_ms = clock.now_ms
        clock.advance(len(data) / BYTES_PER_SEC * 1000 * self.slow_factor)
        return True

def proxy_tts(engine: TTSEngine, req_id: str, client: BrowserClient,
              first_chunk_ms: float = 120, chunk_ms: float = 60,
              queue_capacity: int = 8, jitter_sigma_ms: float = 40.0,
              target_quantile: float = 0.99) -> dict:
    """模拟代理：有界队列 + 首块策略 + 取消传播。返回统计。"""
    z = {0.95: 1.645, 0.99: 2.326}[target_quantile]
    buffer_ms = z * jitter_sigma_ms
    clock = Clock()
    clock.advance(buffer_ms)                    # jitter buffer 直接加到首音延迟上
    queue: List[bytes] = []
    stats = {"buffer_ms": buffer_ms, "max_queue": 0, "cancelled": False,
             "blocked_ms": 0.0}
    produced_ms = 0.0
    first = True
    # 生产/消费分离：生产者尽快填满队列，消费者按「播放速率」推进时钟
    while produced_ms < engine.total_ms:
        # 生产者：把队列填到上限（受 RTF 限制）；队列已满则等待消费者（背压）
        while len(queue) < queue_capacity and produced_ms < engine.total_ms:
            want = first_chunk_ms if first else chunk_ms
            data = engine.generate(req_id, want, clock)
            if data is None:
                break
            queue.append(data)
            stats["max_queue"] = max(stats["max_queue"], len(queue))
            produced_ms += want
            first = False
        filled_to_cap = len(queue) >= queue_capacity      # 队列是否被填满（背压触发条件）
        if not queue:
            break
        # 消费者：播放一块需要「音频时长 × 慢速因子」的真实时间
        chunk = queue.pop(0)
        if not client.write(chunk, clock):
            stats["cancelled"] = True
            engine.cancel(req_id)               # ★ 取消传播到 TTS
            break
        if filled_to_cap:
            # 队列满 => 生产者必须等这一块被播完（背压的量化形式）
            stats["blocked_ms"] += chunk_ms * max(0.0, client.slow_factor - 1.0)
    return stats

print("① 首块大小与首音延迟（RTF=0.15、抖动 40 ms、p99 缓冲 93 ms、48 KB/s）")
print(f"  {'首块(ms)':>8} {'首音延迟(ms)':>12} {'发送字节':>10} {'chunks':>7}")
for first in (20, 60, 120, 250, 500):
    eng, cli = TTSEngine(total_ms=2000), BrowserClient()
    proxy_tts(eng, f"r{first}", cli, first_chunk_ms=first)
    print(f"  {first:>8} {cli.first_audio_ms:>12.0f} {cli.received_bytes:>10,} {cli.chunks:>7}")
print("  读法：首音延迟 ≈ 缓冲 + 首块生成时间（首块 × RTF）+ 传输；")
print("        20 ms 首块看似最快，但低于播放器/解码器的可播单元就会卡 —— 120 ms 是常见折中")

print("\n② 用户离开页面 -> 三级取消（浏览器 → 代理 → TTS）")
eng = TTSEngine(total_ms=5000)
cli = BrowserClient(leave_at_ms=1500)
st = proxy_tts(eng, "leave", cli)
print(f"  客户端离开时已收 {cli.chunks} 块 / {cli.received_bytes:,} 字节，"
      f"取消={st['cancelled']}")
print(f"  TTS 侧已生成 {eng.generated_ms.get('leave', 0):.0f} ms 音频"
      f"（总长 5000 ms），TTS 收到的取消：{eng.cancel_calls}")
print("  读法：取消传到 TTS 后合成立刻停止 —— 不这么做，GPU 会继续为没人听的音频烧算力")

print("\n③ 背压：客户端读得慢（弱网，播放速度 1.8×）")
eng = TTSEngine(total_ms=2000)
cli = BrowserClient(slow_factor=1.8)
st = proxy_tts(eng, "slow", cli, queue_capacity=4)
print(f"  发送 {cli.chunks} 块（无丢弃），队列峰值 {st['max_queue']}（上限 4），"
      f"生产者被阻塞 {st['blocked_ms']:.0f} ms")
print("  读法：音频帧不能丢（会爆音），所以背压作用在合成侧 —— 表现为「整体被拖慢」")

print("\n④ jitter buffer 直接加到首音延迟上")
print(f"  {'抖动σ(ms)':>10} {'p95 缓冲':>9} {'p99 缓冲':>9} {'首块120ms 时的首音(p99)':>25}")
for sigma in (10, 25, 40):
    b99 = 2.326 * sigma
    print(f"  {sigma:>10} {1.645*sigma:>9.0f} {b99:>9.0f} {b99 + 120*0.15:>25.0f}")
print("  读法：抗抖动与低延迟是同一条预算里的两件事；缓冲按抖动分位数取，不是拍常数")

print("\n⑤ 成本：取消率对合成算力的影响（按音频秒占用算力）")
daily_sessions, cancel_rate, avg_ms, rtf = 200_000, 0.12, 6000, 0.15
wasted_ms = daily_sessions * cancel_rate * avg_ms
print(f"  {'策略':<22} {'每日合成量(音频小时)':>20} {'GPU·h':>9} {'$/日':>9}")
for label, ms in (("无取消", wasted_ms), ("有取消（省 80%）", wasted_ms * 0.2)):
    audio_h = ms / 1000 / 3600
    gpu_h = audio_h * rtf
    print(f"  {label:<22} {audio_h:>20,.0f} {gpu_h:>9,.0f} {gpu_h*2:>9,.0f}")
print("  读法：12% 取消率下每天白烧的合成量是可观成本 —— 取消既是体验问题也是成本问题")
```

预期输出要点（实跑，虚拟时钟模拟，确定性可复现）：① 首音延迟随首块增大而上升（缓冲 93 ms 固定 + 首块 × RTF），**20 ms 首块看似最快但低于可播单元就会卡，120 ms 是常见折中**；② 用户在第 1.5 s 离开页面后，**代理把取消传到 TTS，TTS 立刻停止合成**（只生成了约 1.5 s 音频而非 5 s），且 `cancel_calls` 中有该请求——这是本题最核心的评分点；③ 弱网客户端（播放 1.8× 慢）让**队列被填满（峰值 4 = 上限）并迫使生产者累计等待 1,440 ms**：chunk 一个不丢，表现为整体拖慢而不是跳帧；④ jitter buffer 按抖动的 p99 取值（$\sigma$=40 ms → 93 ms），并**直接加到首音延迟上**；⑤ 12% 取消率下每天白烧的合成量换算成 GPU·h 与美元——**取消也是成本问题**。

## 常见追问

- **追问**：为什么推荐 chunked HTTP 而不是 WebSocket？
  - 要点：单向播放场景 chunked HTTP 更简单（易过代理、易鉴权、天然支持 HTTP 缓存与重试语义）；**需要双向时**（barge-in、实时改参数、客户端上传音频）才用 WebSocket。选型看是否需要双向（串 [[elevenlabs-04]] 的 barge-in）。
- **追问**：首块该怎么定？
  - 要点：取「一个完整韵律单元」（子句/短语边界）或固定 100–200 ms；关键是**不要在子句中间切**（会破坏韵律且增加下游拼接噪声）；实测方法是看 TTFA 与卡顿率的联合曲线。
- **追问**：用户切到后台标签页怎么办？
  - 要点：浏览器会限制后台标签页的定时器与音频调度；**推荐策略是「切后台继续生成、暂停播放」**（用户回来可继续听），并在产品上明确；若要省成本则取消，但要接受「回来听不到」。
- **追问**：怎么测「取消是否真的干净」？
  - 要点：① 单元测试——注入写失败，断言 TTS 侧收到 cancel 且生成停止；② 集成测试——真实浏览器关闭页面，断言上游连接数与 GPU 会话数回到基线（**看的是资源曲线，而不是日志**）；③ 长跑测试——反复开关页面，确认无连接/显存泄漏。
- **追问**：下行要不要转码（PCM→Opus）？
  - 要点：省带宽（48 KB/s → 8 KB/s，约 6 倍）但要付出编码延迟与 CPU；**短音频/局域网不必转**，长音频/移动网络值得；转码还会引入额外的缓冲（编解码器帧长），要计入 TTFA。
- **追问**：怎么避免「中间层把流式缓冲成一次性」？
  - 要点：显式关闭代理缓冲（例如 Nginx 的 `proxy_buffering off`）、设置合适的 `X-Accel-Buffering`、并在**上线前用 curl 逐块观察到达时间**来验证（不要只看功能是否可用）。

## 相关题目

- [[together-01]]：流式 token 生成与断连处理，是本题的文本版本；音频多了二进制帧与播放时钟。
- [[multimodal-08]]：流式 TTS 的 chunking 与 jitter buffer，本题的缓冲口径来自它。
- [[multimodal-03]]：实时语音 agent 的延迟预算，本题只覆盖其中「TTS→浏览器」这一段。
- [[elevenlabs-02]]：TTS 服务的容量问题与文本 LLM 有何不同，解释代理背后的算力成本。
- [[elevenlabs-04]]：实时语音 agent 的端到端延迟预算与 TTFA vs TTFT。

## 参考资料与归属

- **设计实时语音 AI Agent** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent>。第 1 节语音链路各段的延迟构成与流水线化的工程背景参照这篇。
- **解释流式 TTS 的 chunking 与 jitter buffer 大小设定（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 3 节 jitter buffer 按抖动分位数取值的口径取自本仓库同主题专题文档。
- **为实时语音 agent 做 latency 预算（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 2 节首音延迟的六段构成与「隐性成本」清单取自本仓库同主题专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（24 kHz/16-bit/单声道 = 48 KB/s、RTF 0.15、音频总长 2–5 s、首块 20–500 ms、后续块 60 ms、队列上限 4–8、抖动 10–40 ms、目标分位 p99、20 万会话/日、取消率 12%、平均 6 s、\$2/GPU·h）都是按本仓库统一口径构造的**工程算例与显式假设**；代码为演示把时间**加速了 50 倍**，输出中的绝对耗时不是真实延迟。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
