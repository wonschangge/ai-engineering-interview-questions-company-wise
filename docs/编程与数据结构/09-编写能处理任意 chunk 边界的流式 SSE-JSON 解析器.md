---
type: question
id: coding-09
topic: 编程与数据结构
order: 9
question: 编写一个流式 SSE/JSON 解析器，能处理任意 chunk 边界。
question_en: Write a streaming SSE/JSON parser that handles arbitrary chunk boundaries.
asked_at: [Cohere]
level: 高阶
tags: [sse, 流式解析, 实现题, 增量-json]
sources:
  - title: Token Streaming 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-token-streaming-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: HTML Living Standard: Server-sent events（延伸）
    url: https://html.spec.whatwg.org/multipage/server-sent-events.html
    author: WHATWG
    published: ""
related: [inference-serving-09, coding-08, multimodal-08, evaluation-07, agents-02]
updated: 2026-09-28
---

## 一句话答案
> **核心不是「切分」而是「状态机」**：维护一个只保存「尚未遇到行尾的那一行」的 buffer，收到 chunk 就追加、按行取走完整的行、把残行留下，任何时刻都不假设一个 chunk 里有一整行或一整个事件；行尾必须同时认 LF、CRLF 与单独 CR（还要处理跨 chunk 的 CRLF）。
> 输入必须是 bytes 并用**有状态的增量 UTF-8 解码器**，否则切在多字节字符中间就会解出乱码——这一条最能区分「写过」和「背过」；事件只在空行处派发，所以多行 `data` 要用 LF 拼接、注释行（心跳）只计数、流结束时残留事件按规范丢弃。
> JSON 层默认走「每个 `data` 就是完整 JSON」的快路径，只有服务端把一个大 JSON 分片发送时才需要括号配平累积器。

## 面试官在考什么

- 能否写出**真正的增量状态机**，而不是「每次 chunk 到来就把整个 buffer 重新切一遍」：后者在跨 chunk 事件上直接错，在大 JSON 分片下退化成 $O(n^2)$。
- 协议细节是否扎实：三种行尾、冒号后只吃一个空格、多行 `data` 用 LF 拼接、空行才派发、`:` 开头是注释、结尾残留事件丢弃、UTF-8 含 BOM 的口径，都能与 WHATWG 的 ABNF 逐条对上。
- 是否意识到**字节边界与字符边界不是一回事**：一个 UTF-8 汉字 3 字节、emoji 4 字节，TCP 与 HTTP chunked 完全不保证不在中间切断。
- 有没有**可验证的测试意识**：对同一段流用 1/2/3/随机长度切分，断言事件序列逐字节一致，这是证明「能处理任意 chunk 边界」的唯一硬证据；以及上线意识——读超时与心跳怎么区分、中断怎么传导到上游生成、`Last-Event-ID` 断点续传、buffer 上界与背压、首事件延迟的可观测性。

常见错误答案：

- 「用 `buffer.split('\n\n')` 取完整段、剩下的留到下次」——CRLF 流里根本没有 `\n\n`（空行是 `\r\n\r\n`），末尾事件又永远等不到空行；实测这两种写法分别丢掉全部事件与最后一个事件。
- 「按 chunk 直接 `decode()` 再找 `data: ` 前缀」——多字节字符被切开时中文变成 U+FFFD，CRLF 还会在值尾留下一个 `\r`。

## 原理与推导

### 1. 线格式：ABNF 与两个易错点
响应头 `Content-Type: text/event-stream` 说明这是事件流；流是 UTF-8 文本、按行组成、事件之间用空行分隔。

| 元素 | 规范口径 | 工程含义 |
| --- | --- | --- |
| 行尾 | CRLF、单个 LF、或单个 CR | 只找 `\n\n` 会同时漏掉 CRLF 与单独 CR |
| 字段 | 第一个 `:` 前是字段名，冒号后**只**吃一个空格 | `data:  x` 的值是 ` x`；`event`/`id`/`retry` 分别是事件类型、断点续传标识、重连毫秒数（只认 ASCII 数字） |
| `data` / 空行 | 每个值后接 LF、派发时去掉末尾 LF；空行才派发，`data` 为空则只重置 | 多行内容等价于 $D = v_1 \Vert \texttt{LF} \Vert \dots \Vert v_n$；心跳、只带 `id` 的事件都不产出内容 |
| `id` 的细节 | 含 NUL 的值忽略；派发时提交、不随事件重置 | 重连时放进 `Last-Event-ID` 请求头 |
| 注释 / 流结束 | `:` 开头整行忽略；未以空行结束的残留事件**不派发** | 注释是心跳（15 秒左右一次可防代理掐连接）；残留要显式决定丢弃还是补派发 |

两个最常被忽略的点：**单独 CR 也是合法行尾**；**事件只在空行处派发**，所以「收到 `data:` 就等于收到内容」是错的，`data: [DONE]` 同样要等空行。

### 2. 增量状态机：buffer、行切分与「待吞 LF」
状态只有四个：残行 buffer、扫描游标、一个「刚见过 CR」标志、以及事件级的 `data`/`event`/`id` 缓冲；收到 chunk 后循环取行，直到取不出整行。CR 是唯一需要前瞻的地方：可以「先不切、等下一个 chunk 看它后面是不是 `\n`」，但这样最后一行要等到下一个 chunk 或 `close()` 才派发；更好的写法是见到 `\r` 立刻当行尾切并记下 `skip_lf`，下一字符若是 LF 就吞掉。两者结果一致，后者少一次等待——实测 20001 个事件按 4096 B 分块喂完后**全部已经派发，`close()` 补派发 0 个**。buffer 上界就是最长的一行，所以给行长设上限（例如 $2^{20}$ 个字符，即 `max_line_chars`）并对超长行报错，是防住故障服务端吃光内存的最低成本手段。

### 3. UTF-8：必须在字节层做，且必须用增量解码器
`chunk.decode()` 在切开的字符上直接抛错或产生 U+FFFD；`codecs.getincrementaldecoder('utf-8-sig')()` 把不完整的字节序列留在解码器内部，下次喂进来再补齐，顺带按规范吃掉流开头的一个 BOM。实测把「中文字符🙂ünïcode」按 1/2/3 字节切分喂入，还原结果完全一致。

### 4. 参考实现之一：SSE 字节级解析器
```python
import codecs

class SSEError(Exception):
    """协议级错误：超长行、UTF-8 解码失败等。"""

class SSEParser:
    """WHATWG「解析事件流」的字节级增量实现：输入 bytes，buffer 里只留未结束的那一行。"""

    def __init__(self, max_line_chars: int = 1 << 20) -> None:
        self._dec = codecs.getincrementaldecoder("utf-8-sig")()
        self._buf, self._scan = "", 0    # 残行 buffer / 扫描游标（避免每个 chunk 重扫）
        self._skip_lf = False            # 刚见过 CR：紧接着的 LF 属于同一个 CRLF
        self.max_line_chars, self.retry_ms, self.closed = max_line_chars, None, False
        self._data: list[str] = []       # data 字段值列表（派发时用 LF 连接）
        self._event_type = self._id_buf = self.last_event_id = ""   # 事件类型 / id 缓冲
        self.events_dispatched = self.comments_seen = 0   # 派发计数 / 心跳计数
        self.discarded = self.dropped_data = ""           # close() 丢弃的残行与 data

    def feed(self, chunk: bytes) -> list[dict]:
        if self.closed:
            raise SSEError("parser already closed")
        self._buf += self._dec.decode(chunk)      # 不完整的多字节序列会被解码器留在内部
        return self._drain()

    def close(self) -> list[dict]:
        """流结束：返回最后能派发的事件；未以空行结束的残留按规范丢弃。"""
        if self.closed:
            return []
        self._buf += self._dec.decode(b"", final=True)   # 触发「结尾半个字符」报错
        events = self._drain()
        self.closed = True
        self.discarded = self._buf                      # 未以行尾结束的残行
        self.dropped_data = "\n".join(self._data)       # 已缓冲但没等到空行的 data
        self._buf, self._event_type = "", ""
        self._data = []
        return events

    def _next_line(self):
        """从 buffer 头切出一行并消费掉；拿不到整行返回 None（用「待吞 LF」代替 CR 前瞻）。"""
        if self._skip_lf:                       # 上一个 CR 后面跟的 LF：吞掉
            if self._buf[:1] == "\n":
                self._buf, self._scan = self._buf[1:], 0
            elif self._buf:
                self._skip_lf = False           # 后面不是 LF，CR 已经单独作数
        buf, i, n = self._buf, self._scan, len(self._buf)
        while i < n:
            c = buf[i]
            if c == "\n" or c == "\r":
                self._buf, self._scan = buf[i + 1:], 0
                self._skip_lf = c == "\r"
                return buf[:i]
            i += 1
        self._scan = i                          # 记住扫描位置：下一个 chunk 从这里继续
        return None

    def _drain(self) -> list[dict]:
        events: list[dict] = []
        while (line := self._next_line()) is not None:
            ev = self._process_line(line)
            if ev is not None:
                events.append(ev)
        if len(self._buf) > self.max_line_chars:   # buffer 里只剩未结束的那一行
            raise SSEError(f"line exceeds max_line_chars={self.max_line_chars}")
        return events

    def _process_line(self, line: str):
        if line == "":                      # 空行 = 派发
            return self._dispatch()
        if line.startswith(":"):            # 注释行：心跳保活
            self.comments_seen += 1
            return None
        colon = line.find(":")
        field, value = (line, "") if colon == -1 else (line[:colon], line[colon + 1:])
        if value.startswith(" "):           # 冒号后只吃掉一个空格（第二个空格是数据）
            value = value[1:]
        if field == "event":
            self._event_type = value
        elif field == "data":
            self._data.append(value)
        elif field == "id" and "\x00" not in value:          # 含 NUL 的 id 忽略
            self._id_buf = value
        elif field == "retry" and value.isascii() and value.isdigit():
            self.retry_ms = int(value)                        # retry 只认 ASCII 数字
        return None                                           # 未知字段忽略

    def _dispatch(self):
        self.last_event_id = self._id_buf    # 规范：先提交 id，即使不派发事件
        if not self._data:                   # 没有 data 字段 → 不派发
            self._event_type = ""
            return None
        ev = {"event": self._event_type or "message", "id": self.last_event_id,
              "data": "\n".join(self._data)}   # 多行 data 以 LF 连接（等价于末尾 LF 去掉）
        self._data, self._event_type = [], ""
        self.events_dispatched += 1
        return ev
```

复杂度：**扫描**是每字节 $O(1)$ 摊还（游标只前进、行尾判定 $O(1)$），但每取出一行都要把剩余 buffer 切片复制一次，所以单次 `feed()` 实际是 $O(C \cdot k)$（$C$ 为 chunk 字节数、$k$ 为其中完整行数）而不是 $O(C)$——实测这一项不是瓶颈（见「数值与代码验证」的切片 vs 偏移量对照，比值 0.84×–1.36×，方向不稳定）；空间是残行 buffer $\le L_{\max}$ 加一个事件的 `data` 行。作为对照，「每个 chunk 重扫整个 buffer」在跨 chunk 的大 JSON 分片下是 $O(n^2)$，本实现不会重扫。

### 5. 参考实现之二：括号配平的增量 JSON
三种做法按优先级排：① **每个 `data` 是完整 JSON** → 直接 `json.loads`，覆盖绝大多数 LLM API，成本最低，优先这样做；② **一个 JSON 被拆到多个事件或多个 `data` 行** → 累积到顶层括号配平再解析；③ 逐 token 增量解析（SAX 风格）→ 只在「必须先看到某个字段才能决策」时才做，SSE 本身已经提供了增量粒度，通常不必。②的关键是**字符串状态**：字符串里的 `{`、`}` 与转义引号都不参与配平，只看括号计数会让状态机提前结束（实测在错误位置解析并抛 `JSONDecodeError`）或在字符串内的 `{` 上永久等待（实测一个值都产不出）。

```python
class JSONStreamAccumulator:
    """把文本片段拼成完整 JSON 值（顶层为 {} 或 []）；字符串与转义内的括号不参与配平。"""

    def __init__(self, max_chars: int = 8 << 20) -> None:
        self.max_chars, self._buf = max_chars, ""
        self._stack: list[str] = []
        self._in_str = self._esc = self._started = False

    @property
    def buffered(self) -> str:
        return self._buf

    def feed(self, text: str) -> list:
        out: list = []
        for ch in text:
            if not self._started:
                if ch in " \t\r\n":
                    continue
                if ch not in "{[":
                    raise ValueError(f"JSON 必须以 {{ 或 [ 开头，实际是 {ch!r}")
                self._started = True
            self._buf += ch
            if len(self._buf) > self.max_chars:
                raise ValueError(f"单个 JSON 超过 max_chars={self.max_chars}")
            if self._in_str:                      # 字符串内部：只关心转义与收尾引号
                if self._esc: self._esc = False
                elif ch == "\\": self._esc = True
                elif ch == '"': self._in_str = False
                continue
            if ch == '"':
                self._in_str = True
            elif ch in "{[":
                self._stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not self._stack or self._stack.pop() != ch:
                    raise ValueError(f"括号不匹配：遇到 {ch!r}")
                if not self._stack:                 # 顶层配平 → 立刻产出一个完整值
                    out.append(json.loads(self._buf))
                    self._buf, self._started = "", False
        return out
```

### 6. 从参考实现到生产：还差什么
- **读超时与心跳**：把「只收到注释」和「真卡住」分开——注释行说明服务端活着、只是模型还没吐字。实测同一段流：服务端静默 3 秒时，无心跳的客户端在 1.0 秒读超时上抛 `TimeoutError`；每 300 ms 一个心跳时，同样 1 秒超时的客户端一直读到 3.0 秒的 `[DONE]`，共 11 次心跳（建连 1 次 + 静默期 10 次）、静默期 0 个内容事件、全程无异常，最终收到 4 个事件（3 个内容 + `[DONE]`）。
- **取消**：用户中断要关闭连接**并停止解析**，还要把取消传导到上游生成（[[multimodal-04]] 的打断语义）。实测客户端读满 50 个事件后强制 RST：服务端被写失败打断前已多写出若干事件（作者环境 99 个、约 2 倍；本次重跑 51 个、约 1 倍），异常是 `ConnectionResetError`——这些事件对应的 GPU 时间就是白烧的，取消要尽早触发而不是等下游发现没人读。这个倍数是竞态量，差别只来自 socket 缓冲与调度，别把它当固定值。
- **重连与断点续传**：`id` 在派发时提交为 `last_event_id`，重连时按规范放进 `Last-Event-ID` 请求头，`retry` 给出重连毫秒数；服务端据此从断点继续，但「已派发而客户端没收到」的事件会重复，消费端要有幂等键。
- **中途断流与背压**：已收到的内容**保留并标记未完成**，把 `close()` 的 `discarded`/`dropped_data` 上报为指标而不是静默丢弃；下游渲染慢时 buffer 不能无限增长，上游要按「未消费事件数」流控或丢弃，思路同 [[coding-08]]。
- **可观测性**：首字节延迟、首事件延迟、事件间隔分布、心跳间隔、中断率、丢弃字节数（[[evaluation-07]]）。另有一个纯工程陷阱：**`read(n)` 会凑满 n 字节才返回**，实测用 37 字节缓冲读同一条流时，最后不足 37 字节的尾巴永远留在缓冲里（客户端超时），换成 `read1(n)`/`recv` 立刻正常——SSE 的读必须用「有多少取多少」的接口。

## 数值与代码验证
原理与推导里第 4、5 小节的清单就是实测用的实现（Python 3.10.12，纯标准库），下面是终端输出（按主题整理，逐条内容未改动）。

**① 规范样例与边界用例逐条对上**（WHATWG 解释算法里的四段流 + 三种行尾、CR 跨 chunk、BOM、含 NUL 的 `id`、坏 `retry`、超长行、非法 UTF-8）：

```text
  [PASS] 例1 三个事件 message:'first event'@1 | message:'second event'@ | message:' third event'@（冒号后只吃一个空格）
  [PASS] 例2 只派发两个事件（末块未以空行结束被丢弃）message:''@ | message:'\n'@；例3 三行 data 用 LF 连接 'YHOO\n+2\n10'
  [PASS] 例4 末尾无空行 → 残留不派发；例5 心跳计数 comments_seen=2
  [PASS] 行尾 LF/CRLF/单独 CR 都切出 ['A','B','C']；CRLF 切在 CR|LF 之间也对；含 NUL 的 id 被忽略
  [PASS] UTF-8 多字节按 1/2/3 字节切分都还原正确（中文字符🙂ünïcode）；超长行抛 SSEError；非法 UTF-8 抛 UnicodeDecodeError
```
全部断言通过：50 条（Python 3.10.12）。

**② 切分压力测试**（本题最有说服力的一条）：同一段 283 B 的 LF 流（含心跳、`event:` 字段、中文、emoji、`\n` 与 `\u00e9` 转义、空 content、`[DONE]`；CRLF 与单独 CR 的跨 chunk 切分由第 ① 组的边界用例覆盖），用 12 种固定长度与 2000 次随机长度喂入，断言事件序列完全一致：

```text
  [PASS] 固定切分 [1, 2, 3, 4, 5, 7, 8, 16, 63, 64, 4096, 283] 全部一致  事件数=7，stream=283 B
  [PASS] 2000 次随机切分（1–11 B）全部一致 不一致 0/2000；逐字节喂入（LF 与中文/emoji 的每个字节都被切开）一致
  [PASS] 还原文本 'The cat 坐 🐈sat\né'
     事件序列（节选）： message:'{"delta":{"content":"The"}}'@ | message:'{"delta":{"content":" 🐈"}}'@ | ... | message:'[DONE]'@
```

**③ 朴素实现的失败对照**（证明状态机不是过度设计）：

```text
  [PASS] 反例1：CRLF 流被 \n\n 切分 → 一个事件都切不出来 []; 末尾事件无空行 → 永久留在 buffer 里 ['data: A']
  [PASS] 反例2：逐字节独立 decode → 中文变成 U+FFFD spec=['{: 中}'] naive=['{: ���}']；CRLF 留下尾随 \r ['A\r','B\r']
  [PASS] 反例3：字符串里的 } 让括号计数提前归零 → 在错误位置解析：JSONDecodeError；字符串里的 { 让它永不归零 → 卡死
```

**④ 性能与线开销**（20001 个事件的 OpenAI 形状流，193.2 B/事件，共 3.86 MB；表中 3 B/1 B 两行只喂前 2000 个事件，共 0.386 MB，所以 MB/s 按 0.386 MB 算）。这台机器上还跑着别的负载（32 核、load average ≈ 46），wall 时间抖动可达数倍，所以耗时一律按 **CPU 时间**（`process_time`）取 9 次最小值：

| chunk 粒度 | 事件数 | CPU 耗时 | µs/事件 | MB/s |
| --- | --- | --- | --- | --- |
| 16 KiB | 20001 | 237.60 ms | 11.88 | 16.3 |
| 4 KiB | 20001 | 228.67 ms | 11.43 | 16.9 |
| 64 B | 20001 | 260.79 ms | 13.04 | 14.8 |
| 3 B | 2000 | 106.65 ms | 53.33 | 3.6 |
| 1 B | 2000 | 259.92 ms | 129.96 | 1.5 |

1 B 一切的成本是 16 KiB 一切的 10.9 倍——**分块越碎，单次 `feed()` 的固定开销越占主导**。分层成本：SSE 状态机 11.97 µs/事件、`json.loads` 2.41 µs/事件、括号配平累积器（慢路径）28.81 µs/事件，是 `json.loads` 的 12.0 倍（本次重跑 28.64 vs 2.92 µs，即 9.8 倍），端到端约为快路径（11.97 + 2.41 ≈ 14.4 µs）的 2 倍，所以「每个 `data` 是完整 JSON」时不要上累积器。同一实现的同一项在三次运行里给过 11.88 / 13.74 / 23.66 µs/事件，把每行切片换成「偏移量 + 定期压缩」则给过 1.36× / 0.96× / 0.85×——**方向都不稳定**，纯 Python 里字节码开销占主导，这类微优化只有在 C++/Rust 或更大 chunk 上才有意义；共享机器上的性能数字只能看比例，不能当容量规划的依据。

线开销（同一段流，`ensure_ascii=False`）：全字段 chunk 193.2 B/事件、其中内容只有 4.16 B，**放大 46.4 倍**；只留 `delta.content` 时 36.2 B/事件、放大 8.7 倍。作为对照，LLaMA-3-70B（GQA-8、bf16）每 token 的 KV cache 是 320 KiB = 327680 B（[[llm-internals-02]] 的口径），线开销约为它的 1/1696——**流式协议的开销不在带宽上，而在「每个事件一次读写唤醒」的固定成本上**，这也是把多个 token 合并成一个事件的价值。末尾处理同样有实测：末尾带空行时派发 20001 个事件，砍掉最后一个换行只剩 20000 个（`[DONE]` 按规范被丢弃）。

**⑤ 真实 HTTP 端到端**（本地 `ThreadingHTTPServer` 输出 `text/event-stream`，客户端每次只读 37 B）：

```text
[LF + 2 ms 间隔] read() 1233 次（min 3 B / max 37 B），心跳 7 次，事件 301 个，首事件延迟 1.25 ms
[CRLF + 无间隔] read() 8024 次，事件 2001 个，心跳 41 次，首事件延迟 0.84 ms
[LF + 8 KiB 大块读] read() 229 次，事件 2001 个，首事件延迟 0.68 ms；三种读法都 EOF=False、残留=''
[缓冲陷阱] resp.read(37) → TimeoutError（剩余字节卡在缓冲里）；resp.read1(37) → 读满 21 个事件
[读超时] 服务端发 3 个事件后静默 3 s，客户端 1 s 读超时
  无心跳：3 个事件、心跳 1 次、TimeoutError、1001 ms ｜ 每 300 ms 心跳：4 个事件、心跳 11 次、无异常、3004 ms
[取消] 客户端解析到 50 个事件后断开；服务端被写失败打断前已写出 99 个事件（≈2 倍），
       异常 ConnectionResetError，从客户端 close 到服务端写失败 0 ms
```
CRLF 流用 37 B 小块读需要 8024 次 `read`、8 KiB 大块读只要 229 次（分块越细，边界落在 chunk 中间的概率越高；`read` 次数由 TCP 投递时机决定，本次重跑分别为 7982 与 153 次，看量级即可）；心跳让同一条流「卡 3 秒」却不触发超时；取消时服务端已经多生成 1–2 倍内容。

## 常见追问
- **追问**：为什么不直接对整个 buffer 找连续两个换行？
  - 要点：三个理由。① CRLF 流的空行是 `\r\n\r\n`，找 `\n\n` 一个事件都切不出来（实测 `[]`）；② 末尾事件后面可能没有空行，会永远留在 buffer 里（实测丢掉最后一个事件）；③ 把整段流缓存下来再切，会把首事件延迟从毫秒级推到整段生成完毕，与流式的目的相反。
- **追问**：SSE 和 WebSocket 怎么选？
  - 要点：SSE 单向（服务端 → 客户端）、跑在普通 HTTP 上、浏览器 `EventSource` 自带重连；WebSocket 双向、要单独握手升级、重连得自己写。LLM 的 token 流是纯单向推送，SSE 更简单；需要双工（实时语音、协作编辑、Agent 实时控制通道）才上 WebSocket（[[multimodal-03]]、[[multimodal-08]]）。
- **追问**：怎么边收边渲染又不破坏 Markdown 格式？
  - 要点：把累积的原始文本当唯一真相，每来一个增量就整段重新渲染并做**增量修复**：未闭合的代码围栏补临时闭合、未完成的表格行先不渲染、行内反引号与粗体标记计数为奇数时补一个临时符号、链接只渲染到 `](` 之前。渲染必须幂等，否则光标与滚动会跳；流结束时删掉所有临时补丁。
- **追问**：`[DONE]` 之外的终止方式要不要都支持？
  - 要点：要按三个来源分别处理：业务终止标记（`data: [DONE]`）、`finish_reason` 之类的字段、TCP/HTTP 层结束。任何单一信号都不能假设一定到达，所以解析器要有 `close()` 路径，业务层要有「没收到终止标记就标记为不完整」的兜底。

## 公司变体

Cohere 是模型 API 平台（Chat / Embed / Rerank），公开文档里的 Chat 接口提供流式输出，因此这题在这条产品线上更可能要求写成**可运维的工程实现**而不是协议背诵：解析器要吃任意 chunk 边界、要处理带 `event:` 的 SSE 帧、要交代读超时与重连、以及客户端取消时如何停止上游生成与计费。它对外同时提供 tokenizer 与多语言能力，「多字节字符被切开」「非 ASCII 内容的事件体积」这类细节也容易被追问。稳妥的答法是先给状态机与三条硬不变量（永不假设 chunk 边界、行尾认三种写法、事件只在空行派发），再用切分压力测试证明它，最后讲清生产缺的那几块（超时、取消、断点续传、背压、可观测性）。（这里只谈公开的产品形态与 API 能力面，不涉及任何具体面试流程。）

## 相关题目
- [[coding-08]]：异步批处理器——流式解析的下游消费与背压控制，取消与限流是同一套语义。
- [[inference-serving-09]]：TTFT/TPOT/ITL 口径——首事件延迟就是客户端能观测到的 TTFT，服务端分块策略要与它对账。
- [[evaluation-07]]：可观测性——首字节/首事件/事件间隔分布、心跳间隔、中断率与丢弃字节数都是这条链路的核心指标。

## 参考资料与归属

- Amit Shekhar (Outcome School)，*Token Streaming 是如何工作的？*：[https://outcomeschool.com/blog/how-does-token-streaming-work](https://outcomeschool.com/blog/how-does-token-streaming-work)。事件流的用途与动机（token streaming、time to first token、自回归逐 token 生成）、`Content-Type: text/event-stream` 的建连方式、`data:` 行加空行的消息形状、`[DONE]` 终止标记、以及 SSE 与 WebSocket 在方向/建连/重连三方面的对比取自该文；第 3 节第 1 小节的字段语义与第 6 小节的生产化条目是在此基础上的工程展开。
- WHATWG，*HTML Living Standard: Server-sent events*（延伸）：[https://html.spec.whatwg.org/multipage/server-sent-events.html](https://html.spec.whatwg.org/multipage/server-sent-events.html)。ABNF（`stream`/`event`/`field`/`end-of-line` 与 `name-char`、`any-char` 的定义）、三种行尾、`data` 追加 LF 与派发时去掉末尾 LF、`id` 的 NUL 规则与派发时提交、`retry` 只认 ASCII 数字、注释行忽略、`Last-Event-ID` 请求头、UTF-8 decode 吃掉开头 BOM、以及「流结束时未派发的残留事件丢弃」全部来自该规范的「Parsing an event stream」与「Interpreting an event stream」两节；第 3 节第 4、5 小节的代码逐条实现的就是这些规则，第 1 小节的表格也以规范口径为准。
- 「数值与代码验证」一节里的性能数字、线开销与端到端结果由本地脚本实测（20001 事件 / 3.86 MB 的 OpenAI 形状流、37 B 小块读的真实 HTTP 端点；耗时取 9 次最小 CPU 时间），适合看量级与比例，不适合当绝对基准。LLaMA-3-70B 每 token 320 KiB 的 KV cache 口径沿用仓库内其它专题（GQA-8、bf16、8 个 KV 头、head_dim 128）；除已列出的两个来源外未引入其它引用，所有实测数字与断言结果均可在本地复现。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
