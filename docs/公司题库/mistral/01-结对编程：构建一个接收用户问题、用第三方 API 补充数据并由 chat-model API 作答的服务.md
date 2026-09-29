---
type: question
id: mistral-01
company: Mistral AI
topic: coding
order: 1
question: 结对编程：构建一个服务，接收用户问题，用第三方 API 的数据做补充，再通过 chat-model API 给出回答。你会如何组织它？
question_en: Pair-programming: build a service that takes a user question, enriches it with data from a third-party API, and answers via a chat-model API. How do you structure it?
asked_at: []
level: 进阶
tags: [结对编程, 实现题, 系统设计, 错误隔离, 流式输出]
sources:
  - title: AI Agent 循环
    url: https://outcomeschool.com/blog/ai-agent-loop
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Building Effective Agents（延伸）
    url: https://www.anthropic.com/engineering/building-effective-agents
    author: Anthropic
    published: 2024-12-19
  - title: ReAct: Synergizing Reasoning and Acting in Language Models（延伸）
    url: https://arxiv.org/abs/2210.03629
    author: Yao et al. (ICLR 2023)
    published: 2022-10-06
  - title: Coroutines and Tasks（Python asyncio 文档）（延伸）
    url: https://docs.python.org/3/library/asyncio-task.html
    author: Python Software Foundation
    published: ""
  - title: Exponential Backoff And Jitter（延伸）
    url: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    author: Marc Brooker (AWS Architecture Blog)
    published: 2015-03-04
  - title: Token Streaming 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-token-streaming-work
    author: Amit Shekhar (Outcome School)
    published: ""
related: [coding-12, coding-08, coding-09, agents-02]
updated: 2026-09-28
---

## 一句话答案

> 分四层：**入口层**只做鉴权、限流与请求校验；**编排层**是唯一实现 `answer(question) -> {answer, data_used, usage, trace}` 的地方；**数据适配层**把第三方 API 包成结构化观察（错误分类、逐层超时、截断信封、幂等键、TTL 缓存）；**模型客户端**用同一个 `chat(messages, tools, deadline)` 契约，按配置在托管 chat-model API（la Plateforme / OpenAI 兼容）与 VPC 内自托管的 open-weight 端点之间切换。这样切的收益是每一层都能用假实现单独跑测试。
> 两个取舍必须显式说：数据补充走「已知字段预取」还是「function calling 让模型自己查」；后者第 $t$ 步要重发完整历史，$I_t = S_0 + (t-1)\bar{s}$、累计 $T S_0 + T(T-1)\bar{s}/2 = O(T^2)$，所以步数上限要从 token/费用预算反推，不能拍 10。
> 三条底线：第三方失败一律降级成观察回灌、按「部分结果 + 原因」返回而不是抛异常；回灌前先序列化再截断并显式带 `truncated`、`total_chars`、`next_offset`；PII 只发最小必要字段、结果进带来源与时间戳的 TTL 缓存，数据驻留受限时把模型侧整体换成 VPC 内端点。

## 面试官在考什么

- **是不是先定接口再写实现**：这题对应 Mistral 的结对编程轮（README.zh-CN.md 的公开信息：技术面之后现场构建一个小型 LLM 服务，随后深挖 transformer/serving 内部原理），所以 `answer()` 的返回结构（答案、用到的数据、用量、轨迹）、入口层与编排层的职责边界、模型客户端「一个契约两个后端」，都要在写第一行代码前说清。
- **数据补充路线是不是显式取舍**：预取（一次往返、延迟低、可能白取）与 function calling（灵活、每步重发全历史）各自的代价，以及步数上限是否从预算反推。
- **外部依赖的失败分类与降级**：429/5xx/超时才是可重试；4xx 参数错误不重试、把可操作信息回灌给模型；空结果与无权限当事实；幻觉工具名回报可用清单；退避带 full jitter、尊重 `Retry-After`，且 sleep 不能占并发额度。
- **幂等与缓存是否落到机制**：幂等键在第一次尝试前生成、跨重试不变；读结果缓存带 TTL、来源与抖动；single-flight 合并并发重复请求。
- **流式与可观测**：SSE 解析是状态机（三种行尾、bytes 层增量 UTF-8）、首事件延迟就是客户端观测到的 TTFT、取消要传导到上游停止生成；轨迹要能回放一次失败决策。

常见错误答案：

- 写成「校验 → 调第三方 → 塞进 prompt → 调 chat API」四步就结束：第三方挂了只会整条请求 500，或者用 `except Exception: data = {}` 把故障伪装成「没有数据」。
- 默认走 function calling 却不设预算：步数拍 10、结果不限长、超时只设一个 30 s，等于把成本与尾延迟都交给模型决定。

## 原理与推导

### 1. 分层与接口契约

数据流是 `request → 入口层 → 编排层 → {数据适配层, 模型客户端} → 流式响应`。划分依据是「谁决定什么」：入口层决定收不收，编排层决定怎么组合，适配层决定怎么跟外部世界打交道，模型客户端只负责一次模型调用。契约先定死，四层才能平行开发、单独替换、单独测试；接口只有三个：`DataClient.get(endpoint, params, *, deadline, idem_key)`、`ChatClient.chat(messages, tools, *, deadline)`、`answer(question, *, data, model, cfg)`。

| 层 | 只做 | 不做 | 失败语义 |
| --- | --- | --- | --- |
| 入口层 | 鉴权、租户配额、令牌桶限流、请求 schema 校验、注入 trace_id | 不碰 prompt、不调外部 API | 401/403/429/422 快速失败 |
| 编排层 | 预取或循环、预算判决、上下文组装、轨迹记录 | 不知道 HTTP，不知道数据源字段 | 「部分结果 + 原因」，不抛异常 |
| 数据适配层 | 超时分层、错误分类、退避、截断、幂等键、缓存 | 不判断结果够不够用（那是编排层的判断） | 一律转成结构化观察 |
| 模型客户端 | 一次 `chat()`、流式解析、usage 归一化 | 不做重试编排、不拼 prompt | 只在「明确未成功」时有限重试 |

最容易被忽略的是观察对象的形状：成功给 `content`、`total_chars`、`truncated`，失败给 `error.kind`、`error.message`、`error.retryable`、`error.attempts`。错误对象化之后，「一个坏请求只失败自己」才有可能实现——编排层拿到的是数据，不是异常。

### 2. 数据补充的两条路线与步数预算

预取的适用条件是「查什么在服务端已知」：一个订单号换状态、一个城市换天气。一次往返、一次模型调用，延迟最低；代价是可能白取（问的是通用知识时这次调用纯浪费），并且字段一旦写死就无法应对模型换个角度追问。function calling 把「查什么」交给模型：把数据源的字段与枚举做成工具 schema，模型在循环里决定调谁、传什么，观察回灌后再决定下一步；灵活性的价格是每步重发完整历史：

$$I_t = S_0 + (t-1)\bar{s}, \qquad \sum_{t=1}^{T} I_t = T S_0 + \frac{T(T-1)}{2}\bar{s} = O(T^2)$$

$S_0$ 是每步重发的固定前缀（system prompt + 工具定义 + 问题），$\bar{s}$ 是每步新增的助手输出加观察。按仓库 [[coding-12]] 的实测口径复算（$S_0 = 41$、$\bar{s} \approx 241.2$ token，token 用 chars/4 代理），$T=10$ 的累计输入是 11,264、$T=20$ 是 46,648、$T=40$ 是 189,776（闭式解；该文实测 11,263 / 46,650 / 189,800，差 ≤24 token）——累计除以 $T^2$ 收敛到 $\bar{s}/2 \approx 120.6$，平方项是能实测到的，不是理论担忧。

所以步数上限要反解 $\frac{\bar{s}}{2}T^2 + (S_0 - \frac{\bar{s}}{2})T \le B_{\text{token}}$。把 0.05 美元的输入预算按 3 美元/百万 token 换成 16,667 token，解出 $T=12$；0.10 美元给 $T=16$；0.50 美元给 $T=37$。硬编码 10 在 0.05 美元预算下只用掉 67.6%，而 40 步是预算的 11.4 倍。按同一口径加上输出 60 token/步与第三方每次 0.002 美元，单请求成本从 $T=1$ 的 0.003 美元涨到 $T=12$ 的 0.084 美元（输入侧占 58.6%）、$T=40$ 的 0.685 美元（输入侧占 83.1%）：循环越长模型输入越主导，第三方调用反而从 66.2% 降到 11.7%。**上限由预算反推、耗尽时返回部分结果与原因**，这两句要一起说（[[agents-09]]）。

### 3. 外部依赖：分类、重试与降级

同一个 HTTP 状态码在不同类别里的正确动作完全不同，先分类再谈处置（完整版见 [[agents-02]]）：

| 类别 | 信号 | 重试 | 处置 | 回灌给模型 |
| --- | --- | --- | --- | --- |
| 瞬时可重试 | 429、5xx、连接超时、读超时 | 有限次 | $d_n = \min(d_{\max}, d_0 2^{n-1})$ 配 full jitter，有 `Retry-After` 就听它的 | `kind`、`attempts`、`retry_after_s` |
| 参数错误 | 缺必填、类型错、枚举越界 | 否 | 本地或下游都能判对错，直接回灌可操作信息 | 哪个字段、收到什么、合法取值、一个正确示例 |
| 语义失败 | 200 但空结果、无匹配、无权限 | 否 | 当事实，换关键词、换数据源、缩小范围 | 空结果本身加一条下一步建议 |
| 依赖不可用 | 连续超时、错误率飙升 | 熔断后快速失败 | 降级阶梯：缓存快照 → 备用源 → 缩小范围 → 明确告知 | 主源不可用、已降级到某个时间点的快照 |
| 模型侧 | 幻觉工具名、输出不可解析 | 一次兜底 | 结构化输出从源头消除（[[agents-03]]） | 可用工具清单、解析错误位置 |

抖动不是可选项。没有抖动时，同一时刻失败的请求用同一个公式算出同一个 $d_n$，会在同一时刻再次撞向下游。full jitter 取 $U(0, d_n)$，期望是 $d_n/2$、标准差是 $d_n/\sqrt{12}$。本机口径（100 个客户端同时收到 429、漏桶 5 req/s 桶深 5、$d_0=0.5$、$d_{\max}=8$）下，首轮重试时间的标准差从 0（95 个客户端同一时刻重试）变成 0.145，100 ms 窗口内的首轮峰值从 95 降到 27，全部完成时间从 135.5 s 降到 22.2 s；equal jitter 的标准差是它的一半（0.072），总尝试数略少但完成更慢（25.8 s）。取舍取决于限流器是排队还是立即拒绝，两档都要能说（[[coding-08]]）。

超时必须逐层递减并把剩余预算下传：总预算 → 单次尝试 → 连接，每次调用取 $T_{\text{call}} = \min(T_{\text{attempt}}, t_{\text{left}})$；退避等待若会吃掉剩余预算，就放弃这次重试而不是把 deadline 拖爆。降级必须是有阶梯且对模型显式可见的，并且降级结果要带来源与时间戳，绝不能伪装成实时数据。

### 4. 并发、幂等与缓存

并发要落到机制：用 `Semaphore(max_inflight)` 把「在途第三方调用数」钉在 $N$ 以内；退避的 `sleep` 必须放在信号量**外面**，否则重试会把并发额度一起睡掉（本机 16 个任务各 3 次尝试、并发 4：睡在里面 1,404 ms、睡在外面 602 ms，差 2.33 倍）；每个提交带自己的 Future，一个坏请求只失败自己。明确反例：不要用 `asyncio.TaskGroup` 承载「每个子请求独立」的语义，它的语义是「第一个非 `CancelledError` 的失败取消组内其余任务」——本机 5 个子任务里 1 个抛参数错误，2 个已完成、2 个被连带取消（[[coding-08]]）。背压要有两道闸门：待处理队列与在途槽位；只做一道，等待任务会无界增长（实测提交 100 个、队列上限 8、并发 4：入队峰值 8、在途峰值 4、拒绝 56 个）。

幂等分两侧。对第三方**写**操作，幂等键要在第一次尝试前生成、跨所有重试复用、由归一化参数哈希而成（工具名 + 排序后的参数 JSON + 会话或步号），并且下游真的落了唯一约束才算数；超时是「结果未知」而不是失败，正确动作是按幂等键对账而不是重放。对**读**操作，幂等的落地形态是缓存加 single-flight：同一指纹的并发重复请求只打一次上游——实测 400 个并发请求（100 个不同问题各重复 4 次）上游只调 100 次、省 4 倍，其中 300 个在途重复被合并；稍后来的相同请求走 TTL 缓存，第二波 100 个请求全部命中、上游 0 次调用。缓存的三个决策也要显式说出：**键**是什么（规范化后的请求指纹，不是原始问题文本）、**TTL** 多长（时效敏感的数据 30–300 s，来源与抓取时间要跟值一起存，否则答案无法解释出处）、**过期如何打散**（批量写入的缓存项若同一秒过期就是自己制造的雪崩：实测 10,000 条、TTL 300 s，无抖动时每秒过期峰值 10,000 条，加 $U(0, 30\text{s})$ 抖动后峰值 398 条）。命中路径实测 0.01 ms、未命中含排队 330 ms，所以命中率本身就是一等指标；缓存用 cache-aside（先写库再失效），命中时把来源透传进 `data_used`——「基于 10 分钟前的快照」是可接受的产品行为，「基于不知道什么时候的数据」不是（[[coding-06]]、[[system-design-09]]）。

### 5. 流式输出、取消与上下文预算

chat-model 一侧默认走流式。SSE 解析必须是状态机：只保留未遇到行尾的那一行、同时认 LF/CRLF/单独 CR、在 bytes 层用有状态增量 UTF-8 解码器（否则多字节字符被切开就出 U+FFFD）、事件只在空行处派发；`read(n)` 会凑满 n 字节才返回，尾包卡在缓冲里只能靠 `read1`/`recv` 或读超时暴露（[[coding-09]]）。首事件延迟就是客户端观测到的 TTFT，本机毫秒级；切分越碎，单次 `feed` 的固定开销越占主导（1 B 一切比 16 KiB 一切贵 10.9 倍），所以「多 token 合并成一个事件」是服务端可调的延迟旋钮。线开销同样不小：OpenAI 形状的 chunk 全字段 193.2 B/事件而内容只有 4.16 B（放大 46.4 倍），只留 `delta.content` 降到 36.2 B/事件（放大 8.7 倍）。

取消要尽早：用户断开必须关闭连接并把取消传导到上游生成。客户端断开后服务端仍会继续生成（[[coding-09]] 的竞态口径是 1–2 倍）；本机用无界缓冲复现时，客户端收满 50 个事件后服务端仍一路生成到 100 个（浪费 1.00 倍），换成有界队列加 `close()` 与 `cancel()` 传导后降到 1 个——那部分 GPU 时间就是白烧的。回灌进上下文前必须先序列化再截断，并显式带上 `truncated`、`total_chars`、`next_offset`；悄悄截断等于让模型基于残片推理：截断信封固定 15 个字符，500 字符上限下 `content` 是 515 字符、2,000 字符上限下是 2,015 字符，本机一个 268,900 字符的结果只走信封（整个信封折合约 169 token）。错误消息也要面向模型裁剪，异常栈既不具可操作性又可能泄露内部结构（[[agents-02]]、[[safety-06]]）。

### 6. 数据边界与部署取舍

Mistral AI 的客户场景以欧洲企业与 on-prem 为主，所以场景约束要落在数据边界上：发给第三方 API 的字段最小化、PII 出网前脱敏与假名化、日志只留脱敏后的键与摘要、缓存条目带来源与抓取时间；受数据驻留约束时把 chat-model 从托管 API（按 token 计费、无固定成本，但数据出网需要合同与 DPA 兜底）换成 VPC 内的 open-weight 推理端点（数据不出网，但版本、量化、容量与升级都自己管）——接口不变，只换实现与配置，这正是第 1 节把 `ChatClient` 抽成契约的收益。

自托管要落到具体口径：Mistral 7B 每 token 的 KV cache 是 $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = 128 KiB（[[mistral-04]]），80 GiB 卡扣掉权重 13.49 GiB、工作区 2.5 GiB 与碎片 8 GiB 后剩 56.01 GiB，滑窗 $W=4096$ 下单条序列恒为 512 MiB，因此并发上限 112 条；按 [[mistral-04]] 的带宽口径（$b=64$、H100 3.35 TB/s）每 step 14.6 ms，聚合约 4,400 token/s。租用 2.5 美元/GPU·h 时单位成本是 0.158 美元/百万 token，break-even 利用率就是它除以 API 单价：API 0.20 美元/百万 token 要 79%、0.50 要 32%、1.00 要 16%。结论不是「自托管更便宜」，而是「利用率上不去就别自建」——把合规需求与成本放进同一张表讨论，才是这题要的答案。

### 7. 评测与可观测

`trace` 与服务端 span 是同一份数据的两个视角：一次请求一条 trace，span 覆盖入口（鉴权、限流）→ 校验 → 预取或工具调用（每次尝试一条）→ prompt 组装 → 模型调用 → 后处理，每个 span 带模型版本、prompt 版本、数据源版本、缓存命中标记、usage 与错误码；只记工具名与耗时无法回放一次失败决策（[[evaluation-07]]）。指标要闭环到动作：缓存命中率与回源率、错误分类分布、重试率、部分结果率、每请求成本、P50/P95/P99 延迟、每次答案用到的数据条数。本机 1,000 请求仿真里 P95 与 P99 直接落在超时天花板（2 s）上，而部分结果率是 5.3%——**尾延迟指标不告诉你有多少答案是残缺的**，部分结果率必须单独告警。

上线门槛是「用假实现离线跑满全部失败路径」：假第三方按脚本回答（429 后成功、4xx、超时、超大结果、无权限），假模型按脚本产出工具调用或最终答案，断言每条路径的 `status`、`reason`、`data_used`、`usage` 与轨迹条数；这套断言本机约 0.3 s 跑完，跑通之后才有资格谈回归门禁（[[evaluation-04]]）与在线采样（[[evaluation-09]]）。现实预期要摆正：τ-bench 口径下当时最好的 function calling agent 按域加权平均成功率只有 48.2%（τ-retail 61.2%、τ-airline 35.2%），τ-retail 的 pass^8 在 25% 以下（[[agents-09]]），所以服务要按「一定会失败」设计：部分结果 + 原因 + 可回放轨迹，而不是把异常抛给用户。

## 数值与代码验证

下面是本机实跑文件 `.work/mistral01_compact_block.py`（文档里的代码块与它逐字一致）与 `.work/mistral01_verify2.py` 的节选，仅标准库，Python 3.10.12 与 3.11 都跑过。

```python
"""mistral-01 节选骨架：入口层 → 编排层 → 数据适配层 / 模型客户端（仅标准库）。"""
import asyncio, json, random, time
RETRYABLE = {"rate_limited", "server_error", "read_timeout", "connect_timeout"}

class DataError(Exception):
    """HTTP 状态码 → 类型化错误；observation() 是回灌给模型的唯一形状。"""
    def __init__(self, kind, message, retry_after=None):
        super().__init__(message)
        self.kind, self.message, self.retry_after = kind, message, retry_after

    def observation(self, attempts):
        return {"ok": False, "retryable": self.kind in RETRYABLE, "attempts": attempts,
                "error": {"kind": self.kind, "message": self.message}}

def envelope(value, cap):
    """先序列化再截断：截断必须显式告知模型，否则它以为看到的就是全部。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= cap: return {"content": text, "total_chars": len(text), "truncated": False}
    head = text[:cap // 2]
    return {"content": head + "\n…[truncated]…\n" + text[-(cap - cap // 2):],
            "total_chars": len(text), "truncated": True, "next_offset": len(head)}

class DataClient:
    """第三方适配层：并发上限、逐层超时、错误分类、full jitter、截断信封、幂等键。"""
    def __init__(self, transport, cfg, rng=None, clock=time.monotonic):
        self.t, self.cfg, self.clock = transport, cfg, clock
        self.rng, self.sem = rng or random.Random(0), asyncio.Semaphore(cfg["max_inflight"])
        self.inflight = self.peak_inflight = 0
        self.waits = []

    async def get(self, endpoint, params, *, deadline, idem_key=None):
        for attempt in range(1, self.cfg["max_attempts"] + 1):
            if deadline - self.clock() <= 0:
                return DataError("deadline_exceeded", "总预算已耗尽").observation(attempt - 1)
            async with self.sem:                 # 只圈住「在途调用」
                try:
                    self.inflight += 1
                    self.peak_inflight = max(self.peak_inflight, self.inflight)
                    value = await asyncio.wait_for(self.t(endpoint, params, idem_key),
                                                   min(self.cfg["attempt_s"], deadline - self.clock()))
                    err = None
                except asyncio.TimeoutError:
                    err = DataError("read_timeout", f"{endpoint} 超过单次尝试上限")
                except DataError as exc:
                    err = exc
                finally:
                    self.inflight -= 1
            if err is None:
                return {"ok": True, "endpoint": endpoint,
                        **envelope(value, self.cfg["max_result_chars"])}
            if err.kind not in RETRYABLE or attempt == self.cfg["max_attempts"]:
                return err.observation(attempt)  # 参数错误/无权限：不重试，直接回灌
            cap = min(self.cfg["dmax_s"], self.cfg["d0_s"] * 2 ** (attempt - 1))
            delay = err.retry_after or self.rng.uniform(0, cap)      # Retry-After 优先
            if delay > deadline - self.clock():
                return err.observation(attempt)  # 退避会吃掉剩余预算：放弃而不是拖爆
            self.waits.append(round(delay, 4))
            await asyncio.sleep(delay)           # 睡在信号量外，不吃并发额度
        return err.observation(self.cfg["max_attempts"])

async def answer(question, *, data, model, cfg, tokens=lambda s: max(1, len(str(s)) // 4)):
    """编排层唯一入口：返回 {answer, data_used, usage, trace}；失败也返回，不抛异常。"""
    t0, trace, used = time.monotonic(), [], []
    usage = {"in": 0, "out": 0, "usd": 0.0, "steps": 0}
    messages = [{"role": "system", "content": "先用工具取数再回答；数据不足就说不知道"},
                {"role": "user", "content": question}]
    deadline = t0 + cfg["total_s"]
    for step in range(1, cfg["max_steps"] + 1):
        if time.monotonic() > deadline or usage["usd"] >= cfg["max_usd"]:
            return partial("budget", step - 1, used, usage, trace)   # 预算耗尽：硬出口
        reply = await model.chat(messages, cfg["tools"], deadline=deadline)
        inp, out = (tokens(json.dumps(x, ensure_ascii=False, default=str)) for x in (messages, reply))
        usage.update({"in": usage["in"] + inp, "out": usage["out"] + out, "steps": step,
                      "usd": usage["usd"] + inp * cfg["price_in"] + out * cfg["price_out"]})
        trace.append({"step": step, "backend": model.name,
                      "calls": len(reply.get("tool_calls", []))})
        if reply.get("final") is not None:
            return {"status": "ok", "reason": "model_final", "answer": reply["final"],
                    "data_used": used, "usage": usage, "trace": trace}
        for call in reply.get("tool_calls", []):     # 失败不是异常，而是下一条观察
            obs = await data.get(call["name"], call.get("args", {}), deadline=deadline,
                                 idem_key=call.get("id"))
            used.append({"endpoint": call["name"], "ok": obs["ok"],
                         "truncated": obs.get("truncated", False)})
            messages.append({"role": "tool", "name": call["name"],
                             "content": json.dumps(obs, ensure_ascii=False, sort_keys=True)})
    return partial("step_limit", cfg["max_steps"], used, usage, trace)

def partial(reason, steps, used, usage, trace):
    """预算是硬出口：返回已有证据与原因，不抛异常、不无限重试。"""
    return {"status": "partial", "reason": reason, "steps": steps, "data_used": used,
            "answer": f"未完成（{reason}），已查到 {len(used)} 条数据",
            "usage": usage, "trace": trace}
```

表 1：步数上限从预算反推（$S_0 = 41$、$\bar{s} = 241.2$、输入 3 美元/百万 token）

| 输入预算 | 预算 token | 反推 $T$ | 实际累计输入 | 占预算 |
| --- | --- | --- | --- | --- |
| 0.05 美元 | 16,667 | 12 | 16,411 | 98.5% |
| 0.10 美元 | 33,333 | 16 | 29,600 | 88.8% |
| 0.50 美元 | 166,667 | 37 | 162,156 | 97.3% |
| 对照：硬编码 10 步 / 40 步 | 16,667 | — | 11,264 / 189,776 | 只用 67.6% / 超预算 11.4 倍 |

表 2：抖动三档（100 客户端同时 429、漏桶 5 req/s 桶深 5、$d_0=0.5$、$d_{\max}=8$，200 个随机种子取均值）

| 抖动 | 总尝试 | 其中 429 | 完成耗时 s | 首轮重试 std | 100ms 窗口首轮峰值 |
| --- | --- | --- | --- | --- | --- |
| 无 | 1,107 | 1,007 | 135.50 | 0.000 | 95.0 |
| equal | 555 | 455 | 25.77 | 0.072 | 46.0 |
| full | 613 | 513 | 22.19 | 0.145 | 27.0 |

理论值是 full 的 std $= d_0/\sqrt{12} = 0.1443$、equal $= d_0/(2\sqrt{12}) = 0.0722$、无抖动 $= 0$：实测与公式吻合，也复现了 [[coding-08]] 的同口径结果（0.143 / 26.9 / 22.36 s）。full 的总尝试比 equal 多 10.5%，但完成时间短 13.9%、首轮峰值低 41.3%。

表 3：幂等、缓存与 TTL（第一波 400 个并发请求 = 100 问 × 4 重复；第二波 100 个顺序请求、间隔 2 ms）

| 指标 | 实测 |
| --- | --- |
| 上游调用（第一波） | 100 次（无 single-flight 是 400 次，省 4.0 倍），300 个在途重复被合并 |
| 墙钟（第一波） | 635 ms，与「100 次 ÷ 并发 4 × 25 ms = 625 ms」的下限一致 |
| 缓存（第二波） | 100/100 命中，上游新增 0 次；命中 0.01 ms，未命中含排队 330 ms |
| TTL 过期峰值 | 无抖动 10,000 条/s，$U(0,30\text{s})$ 抖动后 398 条/s（降 25.1 倍），std 8.66 s |

同一份骨架的失败路径实跑输出（假第三方按脚本回答、假模型按脚本产出工具调用，`flaky+bad` 表示先 429 后参数错误）：

```console
flaky+bad     status=ok      reason=model_final  steps=3 in=208 usd=0.00116 数据=2 退避=[0.02]
slow          status=ok      reason=model_final  steps=2 in=92  usd=0.00059 数据记录 ok=False
              退避=[0.0422, 0.0758]（重试 3 次后把 read_timeout 观察回灌，模型用已有证据作答）
huge          status=ok      reason=model_final  steps=2 数据=1（truncated=True，content 515 字符）
max_usd=0.0002 status=partial reason=budget     steps=1   ← 硬出口，返回已有证据
max_steps=1    status=partial reason=step_limit steps=1
12 并发请求    峰值在途=4（上限 4），12 个全部 status=ok
幂等键透传     [('flaky','c0',1), ('flaky','c0',2), ('bad','c1',3)]  ← 两次尝试复用同一个键
空结果/幻觉工具名/4xx：上游各被调用 1 次（不重试），kind 依次 ok / not_found / invalid_args
编排层自身开销：两种后端（hosted 与 on-prem-open-weight）同一契约各跑一遍，墙钟 3.3 ms（假 transport）
```

对照组（同机实测，证明这些机制不是过度设计）：退避睡在信号量里 1,404 ms vs 睡在外面 602 ms（2.33 倍）；`TaskGroup` 里 5 个子任务 1 个失败会连带取消尚未完成的 2 个，换成独立 Future 后 0 个被连带取消；提交 100 个、队列上限 8、并发 4 时入队峰值 8、在途峰值 4、拒绝 56 个；客户端收满 50 个事件断开后，无界缓冲下服务端仍生成到 100 个，有界队列加取消传导后只多生成 1 个。

## 常见追问

- **追问**：为什么不让模型直接调第三方 API？
  - 要点：三个理由。凭据不能进模型上下文（等于把 key 交给不可信输入）；服务端无法做限流、预算与 schema 校验；观测、缓存与幂等全部失效——流量绕过适配层后，你既不知道花了多少，也无法回放。正确做法是适配层持有凭据并把它做成窄接口（只用得到的字段与枚举）。
- **追问**：第三方限流（429）时怎么办？
  - 要点：这是闭环而不是一次重试：`Retry-After` 优先，其次 $d_n = \min(d_{\max}, d_0 2^{n-1})$ 加 full jitter（100 客户端场景下首轮重试 std 从 0 到 0.145、完成时间从 135.5 s 到 22.2 s），连续失败到阈值就熔断并走降级阶梯，退避参数要与自己的令牌桶对齐（[[coding-07]]、[[coding-08]]）。
- **追问**：用户重复提问或客户端重试，会不会重复调用第三方与重复计费？
  - 要点：会，除非幂等键在第一次尝试前生成并被下游消费。读路径的落地形态是缓存加 single-flight（实测 400 个并发重复请求压到 100 次上游调用）；写路径靠唯一约束加对账：超时是未知结果，先查幂等键再决定重放；计费侧按幂等键去重（[[agents-02]]）。
- **追问**：流式还是非流式？取消怎么做？
  - 要点：交互式场景默认流式，因为首事件延迟就是用户感知的 TTFT；解析器必须是状态机（三种行尾、增量 UTF-8、空行派发），取消要把关闭连接的信号传导到上游生成，否则白烧 GPU（[[coding-09]]）。非流式只留给内部批处理与需要完整 JSON 的评测路径。
- **追问**：生产化还差什么？
  - 要点：三处最容易踩的坑——超时预算没有逐层递减（总预算 → 单次尝试 → 连接，且要下传剩余预算）；`read(n)` 凑满 n 字节才返回，尾包卡在缓冲里只能靠 `read1`/`recv` 或读超时暴露；Python 线程池超时杀不掉任务，超时判定必须绑定「真的开始执行」，否则回灌给模型的是假观察，会把它引向无意义的重试（[[coding-12]]、[[coding-09]]）。

## 相关题目

- [[coding-12]]：最小化 agent loop 的预算、截断信封与无进展检测，本文的步数上限与幂等键口径来自那里；[[coding-08]]：异步批处理器，并发上限、full jitter 与错误隔离的完整实测；[[coding-09]]：SSE 流式解析器，取消传导、线开销与 `read(n)` 陷阱。
- [[agents-02]]：工具调用错误分类、幂等与超时预算的完整版；[[agents-03]]：结构化输出从源头消除参数错误；[[agents-09]]：终止条件、成本与步数限制以及 τ-bench 口径；[[system-design-09]]：LLM gateway 的路由、缓存、预算与限流。
- [[coding-06]]：TTL 缓存与淘汰策略；[[coding-07]]：令牌桶限流；[[evaluation-07]]、[[evaluation-09]]、[[evaluation-04]]：遥测 schema、在线采样与回归门禁；[[mistral-04]]：Mistral 7B 的 KV cache 与自托管容量口径；[[mistral-06]]：function calling 的生产可靠性；[[mistral-08]]：不能外发数据的 on-prem 部署；[[safety-06]]：工具权限与数据外泄。

## 参考资料与归属

- *AI Agent 循环*（原文 *What is an AI Agent Loop?*）— Amit Shekhar（Outcome School）：<https://outcomeschool.com/blog/ai-agent-loop>。提供 think–act–observe 循环结构、约 20 行的循环骨架、两条停机条件（模型自报完成与步数上限）与四类循环失败；本文的分层契约、错误分类、预算反推、代码与全部数字均为按该口径自行实现与测量，源文没有给出量级数字。
- *Building Effective Agents* — Anthropic，2024-12-19：<https://www.anthropic.com/engineering/building-effective-agents>（延伸来源：workflow 与 agent 的区分、优先最简单可控的方案、把工具定义当作接口来投入）；*ReAct: Synergizing Reasoning and Acting in Language Models* — Yao 等（ICLR 2023），arXiv 2022-10-06：<https://arxiv.org/abs/2210.03629>（延伸来源：推理与行动交错、观察回灌修正下一步，以及 ALFWorld 与 WebShop 上 34% 与 10% 的绝对成功率提升）。
- *Coroutines and Tasks*（Python asyncio 文档）— Python Software Foundation：<https://docs.python.org/3/library/asyncio-task.html>（延伸来源：`TaskGroup` 在第一个非 `CancelledError` 失败时取消组内其余任务、`asyncio.timeout` 与 `wait_for` 的取消语义、`Semaphore` 与 `Queue` 的用法；第 4 节的 TaskGroup 反例与超时分支出自该文档）。
- *Exponential Backoff And Jitter* — Marc Brooker（AWS Architecture Blog），2015-03-04：<https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/>（延伸来源：Full / Equal / Decorrelated 三种抖动口径与模拟结论；第 3 节的公式与表 2 的复算以该文口径为准）。*Token Streaming 是如何工作的？* — Amit Shekhar（Outcome School）：<https://outcomeschool.com/blog/how-does-token-streaming-work>（流式动机、SSE 消息形状与 `[DONE]` 终止标记；具体测量沿用站内 [[coding-09]]）。
- 表 1 至表 3、`console` 输出与对照组数字（并发闸门、TaskGroup 连带取消、背压拒绝、取消后的多余生成、TTL 过期峰值、缓存与幂等命中）都在本机（AMD Ryzen 9 8945HX 级机器、Python 3.10.12 与 3.11）用 `.work/mistral01_math.py`、`.work/mistral01_service.py`、`.work/mistral01_verify2.py`、`.work/mistral01_compact_block.py` 与 `.work/mistral01_doc_driver.py` 真实运行得到，时间类数字受负载影响、看比值即可；Mistral 7B 的 KV cache 与 H100 带宽口径沿用站内 [[mistral-04]]。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
