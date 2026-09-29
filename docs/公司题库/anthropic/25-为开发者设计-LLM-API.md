---
type: question
id: anthropic-25
company: Anthropic
topic: system-design
order: 25
question: 为开发者设计 LLM API（面向外部开发者的接口、版本与配额）。
question_en: Design an LLM API for developers (interface, versioning, quotas).
asked_at: []
level: 高阶
tags: [系统设计, API-设计, 版本兼容, 配额, 幂等]
sources:
  - title: RFC 9110：HTTP Semantics（延伸）
    url: https://www.rfc-editor.org/rfc/rfc9110.html
    author: IETF (Fielding, Nottingham, Reschke)
    published: 2022-06
  - title: AIP-158：Pagination（Google API 改进提案）（延伸）
    url: https://google.aip.dev/158
    author: Google
    published: 
  - title: AIP-180：Backwards compatibility（Google API 改进提案）（延伸）
    url: https://google.aip.dev/180
    author: Google
    published: 
  - title: Handling Overload（Google SRE Book 第 21 章）（延伸）
    url: https://sre.google/sre-book/handling-overload/
    author: Google SRE
    published: 
  - title: Exponential Backoff And Jitter（延伸）
    url: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    author: Marc Brooker (AWS Architecture Blog)
    published: 2015-03-04
related: [anthropic-17, anthropic-24, anthropic-08, system-design-09, anthropic-23]
updated: 2026-09-28
---

## 一句话答案

> 面向开发者的 LLM API，设计目标不是「把模型能力暴露出去」，而是**让开发者能可靠地构建产品**。七条设计原则：
> ① **接口面窄而正交**：`messages`（对话）、`tools`（工具调用）、`files`（文件）、`batches`（批处理）、`models`（能力与定价）、`usage`（用量）；能用一个接口表达的就不要新增（这正是 [[anthropic-19]] 工具面原则在 API 上的翻版）；
> ② **版本策略明确**：**加字段/加枚举值不破坏兼容**，删字段/改语义才需要新版本（可引用的口径来自 AIP-180：向后兼容的变更要显式定义并测试）；版本号放在日期或路径里，**弃用要有公告期与迁移窗口**；
> ③ **错误可编程处理**：每个错误都有稳定 `type`（`invalid_request` / `rate_limit` / `overloaded` / `context_length` / `content_filter`）+ **是否可重试** + `retry_after`（若适用）；客户端只需按 `type` 分支（可引用的口径：HTTP 语义把 4xx 与 5xx 的区别、以及 `Retry-After` 的语义标准化了）；
> ④ **配额可预测**：按维度限流（RPM 请求数、TPM token 数、并发数），响应头回传当前余量与重置时间；超限返回 429 + `Retry-After`（配合 [[anthropic-08]] 的退避抖动）；
> ⑤ **幂等与安全重试**：写操作支持客户端幂等键，重复请求返回同一结果而不重复计费；
> ⑥ **流式是一等公民**：SSE 分帧、`event:` 类型明确、结束事件必达、可中断、可续传（带 `seq`）；
> ⑦ **可观测与可预算**：用量可查、成本可估（`count_tokens` 式的预检）、有沙箱/测试模式、有状态页与变更日志。
> 一句话判据：**API 的质量取决于「开发者在出错时能不能自助解决」**——错误分类、配额余量、令牌计数、迁移指南，这四样决定支持成本。

## 面试官在考什么

- **是否以「开发者体验」为轴**：把 API 当产品而不是函数集合——文档、SDK、错误、配额、迁移路径都是设计对象。
- **兼容性纪律**：什么算破坏性变更（删字段、改类型、改默认值语义、收紧校验）；如何用「加字段 + 忽略未知字段」实现演进；如何做弃用（时间线、双跑、告警）。参考口径：AIP-180 明确列出兼容与不兼容变更类型及测试要求。
- **错误分类的可编程性**：能否给出稳定的错误码表，并区分**客户端可修**（参数错）、**可重试**（限流/过载）、**不可重试**（内容策略、上下文超长需换策略）。
- **限流设计**：为什么按 token 而不是只按请求数（一个 100K token 的请求与一个 10 token 的请求成本差 4 个数量级）；如何做**分层配额**（组织/项目/密钥）；并发限制与速率限制的区别。
- **幂等**：为什么生成类接口也需要幂等键（网络重试会重复计费）；如何实现（键 → 结果映射 + 过期）。
- **流式的协议细节**：SSE 的心跳、断线重连、`Last-Event-ID` 式续传、以及**取消**（客户端断开后服务端应停止生成以省算力）。
- **可迁移性**：API 变更如何通知（Deprecation 头、邮件、SDK 提示）；如何做**影子流量/双跑**让开发者验证。

**常见错误答案**

- 只设计一个 `POST /completions`，不区分流式/工具/批处理/文件。
- 错误只返回 `{「error」: 「something went wrong」}`——不可编程处理。
- 只按请求数限流——大 token 请求会打爆后端。
- 版本号靠 URL 里的 `v1` 却在该路径下做破坏性变更（`v1` 变成「永远的最新版」）。
- 没有幂等键：客户端超时重试导致重复计费与重复副作用。
- 忽略取消：客户端断开后服务端还在生成（白烧 GPU）。

## 原理与推导

### 1. 接口面（六个资源）

| 资源 | 方法 | 说明 |
| --- | --- | --- |
| `POST /v1/messages` | 创建对话补全（`stream: true/false`） | 核心接口；支持 `tools`、`system`、`max_tokens` |
| `POST /v1/messages/count_tokens` | 预检 token 数 | 让开发者能预算成本与上下文 |
| `POST /v1/files` + `GET /v1/files/{id}` | 上传/查询文件 | 用于批量与长文档 |
| `POST /v1/batches` | 提交批处理任务 | 高吞吐、低单价、可容忍分钟级延迟 |
| `GET /v1/models` | 能力与定价元数据 | 让客户端能自适应 |
| `GET /v1/usage` | 用量与计费明细 | 支持按项目/密钥维度 |

**为什么窄**：每个资源都要维护版本、配额、SDK、文档与兼容性测试；资源越多，长期维护成本越高（对齐 [[anthropic-19]] 的「少而正交」）。

### 2. 兼容性：什么能改、什么不能

| 变更 | 兼容？ | 处理 |
| --- | --- | --- |
| 新增可选请求字段 | ✅ | 直接发布 |
| 新增响应字段 | ✅ | 客户端必须忽略未知字段（写进文档并测试） |
| 新增枚举值 | ⚠️ | 需要客户端有 `default` 分支；公告期 |
| 改默认值语义 | ❌ | 新版本或显式参数 |
| 删字段/改类型 | ❌ | 新版本 + 迁移窗口 |
| 收紧校验（原本接受现在拒绝） | ❌ | 视为破坏性变更（最常见的「意外破坏」） |

**工程做法**：① 契约测试（对每个兼容性声明写测试）；② **双跑**：新旧行为并行一段时间，用 `Deprecation` 响应头告警；③ 弃用时间线（公告 → 告警 → 只读 → 下线）至少跨越数个发布周期。

### 3. 错误分类（可编程处理）

| HTTP | `type` | 可重试 | 客户端应有行为 |
| --- | --- | --- | --- |
| 400 | `invalid_request` | ❌ | 修参数（错误信息给合法取值） |
| 401/403 | `authentication_error` / `permission_error` | ❌ | 修凭证/权限 |
| 404 | `not_found` | ❌ | 检查 id |
| 413 | `request_too_large` | ❌ | 拆分或换文件接口 |
| 422 | `context_length_exceeded` | ❌ | 截断/摘要/换长上下文模型 |
| 429 | `rate_limit_error` | ✅（尊重 `Retry-After`） | 退避 + 抖动（串 [[anthropic-08]]） |
| 500 | `api_error` | ✅ | 退避重试 |
| 529 | `overloaded_error` | ✅ | 退避、降级到小模型或排队 |
| 200 + `stop_reason: content_filter` | — | ❌ | 换措辞或走人工（不是重试） |

**关键**：`stop_reason`（`end_turn` / `max_tokens` / `tool_use` / `content_filter`）必须**显式**返回——否则客户端无法区分「答完了」与「被截断了」（这是最常见的集成 bug 来源）。

### 4. 配额：多维度 + 分层

| 维度 | 作用 | 典型粒度 |
| --- | --- | --- |
| RPM（请求/分钟） | 防小请求洪水 | 密钥/项目/组织 |
| TPM（token/分钟） | 防大请求打爆算力 | 同上 |
| 并发请求数 | 防长请求占满 KV | 项目 |
| 每日/每月预算 | 成本控制 | 组织 |

**为什么必须按 token 限流**：一个请求的成本 ∝ 输入 + 输出 token；若只限 RPM，攻击者/误配置可以发 128K token 的请求把后端打满。**响应头**回传 `x-ratelimit-remaining-{rpm,tpm}` 与 `reset` 时间，让客户端能自适应（这也是「开发者能自助」的关键）。

### 5. 幂等与取消

- **幂等键**：客户端生成 UUID 放在请求头；服务端在事务里建立「键 → 结果」映射（TTL 例如 24 h）；重复请求直接返回原结果，**不重复计费**。
- **取消**：客户端断开 SSE 连接 → 服务端应停止生成（省算力）。协议上要定义「断开即取消」，并在文档里说明**已生成的 token 仍会计费**（否则用户会以为取消=免费）。这一条非常容易被忽略，却是成本与体验的交汇点。
- **超时预算**：文档要给出推荐值——连接超时、首 token 超时、总超时；SDK 内置合理的默认值与自动重试（仅对可重试类型）。

### 6. 流式协议（SSE）规范

```
event: message_start      data: {"id": "...", "model": "...", "usage": {"input_tokens": 1234}}
event: content_block_delta data: {"index": 0, "delta": {"type": "text_delta", "text": "He"}}
event: ping                data: {}                      ← 心跳，防中间层断连
event: message_delta       data: {"stop_reason": "end_turn", "usage": {"output_tokens": 312}}
event: message_stop        data: {}                      ← 结束事件必达
```

**要求**：① 事件类型稳定且可扩展（客户端忽略未知事件）；② 心跳（否则 LB 会在空闲时断连）；③ 结束事件必达（否则客户端要等超时）；④ `usage` 在结束时给出（用于计费对账）。

### 7. 支持成本：把问题挡在工单之前

- **可自助**：`count_tokens`（预算）、`usage`（对账）、状态页（区分「我的问题」与「平台问题」）、错误信息含**可操作建议**与请求 id（便于排查）；
- **可验证**：发布前用**契约测试 + 兼容性测试矩阵**（SDK × 版本 × 流式/非流式）；
- **可演进**：变更日志 + 迁移指南 + 弃用告警（响应头 + 邮件 + 控制台横幅）。

## 数值与代码验证

### 表 1：限流维度与典型配置（示例组织）

| 维度 | 值 | 说明 |
| --- | --- | --- |
| RPM | 4,000 | 平均 66 req/s |
| TPM | 2,000,000 | 平均 33K token/s |
| 并发 | 100 | 长请求（128K 上下文）的关键约束 |
| 日预算 | \$5,000 | 超出后拒绝或降级 |

**一致性检查**：若平均请求 500 token（输入+输出），RPM 4,000 → 2M token/分钟 = **恰好打满 TPM**；若平均请求 2,000 token，则 RPM 4,000 会需要 8M TPM——**说明两个维度的配置必须与真实请求画像匹配**，否则会出现「RPM 没到但 TPM 已满」的困惑（这也是最常见的配额工单来源）。

### 表 2：错误处理矩阵（客户端应实现的分支）

| 场景 | 重试 | 退避 | 其他动作 |
| --- | --- | --- | --- |
| 429 | ✅ | 尊重 `Retry-After`，否则指数+抖动 | 降低并发、切小模型 |
| 529 | ✅ | 指数+抖动（上限 3–5 次） | 排队或降级 |
| 5xx | ✅ | 指数+抖动 | 记录请求 id 上报 |
| `context_length_exceeded` | ❌ | — | 截断/摘要/换模型 |
| `content_filter` | ❌ | — | 换措辞或人工 |

### 可运行代码

```python
# LLM API 的三件开发者基础设施：token 预算、配额一致性检查、错误分类重试策略
from dataclasses import dataclass
from typing import Optional
import math, random

# 1) token 估算（用于 count_tokens 式的预检；真实实现用 tokenizer，这里给出可核算的近似）
def estimate_tokens(text: str, per_token_chars: float = 4.0) -> int:
    """英文约 4 字符/token，中文约 1.5 字符/token；这里按可配置口径估算"""
    return max(1, math.ceil(len(text) / per_token_chars))

def price(input_tokens: int, output_tokens: int, cache_hit_tokens: int = 0,
          price_in=3.0, price_out=15.0, cache_discount=0.1):
    """每百万 token 计价（示意）"""
    return (input_tokens - cache_hit_tokens) / 1e6 * price_in \
         + cache_hit_tokens / 1e6 * price_in * cache_discount \
         + output_tokens / 1e6 * price_out

prompt = "请用三句话解释 KV cache 的显存开销" * 30        # 约 600 字符
print("① token 预检与成本估算")
tin = estimate_tokens(prompt)
print(f"  估算输入 {tin} token；输出上限 1,024 token 时最坏成本 "
      f"${price(tin, 1024):.5f}（按 \\$3/\\$15 每百万 token）")
print(f"  其中若 80% 输入命中前缀缓存：${price(tin, 1024, cache_hit_tokens=int(tin*0.8)):.5f}")
print("  读法：count_tokens 让开发者在下发请求前就能预算 —— 这是减少成本类工单的最有效手段")

# 2) 配额一致性检查：RPM 与 TPM 必须匹配真实请求画像
@dataclass
class Quota:
    rpm: int
    tpm: int
    concurrency: int
    def implied_avg_tokens(self) -> float:
        return self.tpm / self.rpm
    def check(self, avg_in: int, avg_out: int, peak_concurrency: int) -> dict:
        per_req = avg_in + avg_out
        need_tpm = self.rpm * per_req
        return {"每请求 token": per_req, "RPM 打满时需要的 TPM": need_tpm,
                "TPM 足够": need_tpm <= self.tpm,
                "建议 RPM（按 TPM 反推）": int(self.tpm / per_req),
                "并发足够": peak_concurrency <= self.concurrency}
q = Quota(rpm=4000, tpm=2_000_000, concurrency=100)
print(f"\n② 配额一致性（RPM={q.rpm}, TPM={q.tpm:,}, 并发={q.concurrency}）")
print(f"  该配额隐含的平均请求规模 = TPM/RPM = {q.implied_avg_tokens():.0f} token/请求")
for avg_in, avg_out, conc in ((400, 100, 60), (1600, 400, 90), (8000, 2000, 120)):
    r = q.check(avg_in, avg_out, conc)
    print(f"  画像 输入 {avg_in:>5}/输出 {avg_out:>5}: 每请求 {r['每请求 token']:>5} token, "
          f"TPM 够={r['TPM 足够']}, 建议 RPM={r['建议 RPM（按 TPM 反推）']:>5}, "
          f"并发够={r['并发足够']}")
print("  读法：同一份配额在不同请求画像下含义完全不同 —— 文档必须给出「隐含平均请求规模」，")
print("        否则开发者会遇到「RPM 没用满却被 429」的困惑")

# 3) 错误分类与重试策略（客户端应实现的分支）
@dataclass
class ApiError(Exception):
    status: int
    type: str
    retryable: bool
    retry_after_s: Optional[float] = None

ERRORS = {
    400: ("invalid_request", False), 401: ("authentication_error", False),
    403: ("permission_error", False), 404: ("not_found", False),
    413: ("request_too_large", False), 422: ("context_length_exceeded", False),
    429: ("rate_limit_error", True), 500: ("api_error", True), 529: ("overloaded_error", True),
}
def classify(status: int) -> ApiError:
    t, r = ERRORS.get(status, ("api_error", status >= 500))
    return ApiError(status, t, r, retry_after_s=1.0 if status == 429 else None)

def call_with_retry(fail_sequence, max_attempts=5, seed=7):
    """按给定失败序列模拟重试；完整 API 可能返回 partial（部分结果）或 stop_reason"""
    rnd = random.Random(seed)
    delay = 0.05
    for attempt in range(1, max_attempts + 1):
        status = fail_sequence[attempt - 1] if attempt <= len(fail_sequence) else 200
        if status == 200:
            return {"attempts": attempt, "ok": True}
        err = classify(status)
        if not err.retryable:
            return {"attempts": attempt, "ok": False, "type": err.type, "action": "修请求，不要重试"}
        wait = err.retry_after_s if err.retry_after_s is not None else min(2.0, delay * 2 ** (attempt - 1))
        wait = rnd.uniform(0, wait)                      # full jitter
        delay = wait if err.retry_after_s is None else delay
    return {"attempts": max_attempts, "ok": False, "type": "retries_exhausted"}

print("\n③ 错误分类与重试")
for seq, label in (([529, 529, 200], "两次过载后成功"),
                   ([422, 200], "上下文超长（不应重试）"),
                   ([429, 429, 429, 429, 429], "持续限流（重试耗尽）")):
    r = call_with_retry(seq)
    print(f"  {label:<22} -> {r}")
print("  读法：客户端只应按 type 分支；对不可重试类型立刻失败并给出可操作提示，")
print("        对可重试类型用「尊重 Retry-After + full jitter」；重试预算必须有限")

# 4) 弃用时间线：给开发者足够迁移窗口
def deprecation_timeline(announce_day=0, warn_days=30, readonly_days=60, sunset_days=180):
    return {
        "公告": announce_day,
        "响应头告警开始": announce_day + warn_days,
        "只读（拒绝新写入）": announce_day + readonly_days,
        "下线": announce_day + sunset_days,
        "迁移窗口天数": sunset_days - announce_day,
    }
print("\n④ 弃用时间线（示例策略）")
for k, v in deprecation_timeline().items():
    print(f"  {k:<16} {v}")
print("  读法：窗口要给足（半年量级），并且每个阶段都要有「可观测的告警」——")
print("        只发邮件不算通知，响应头 + 控制台 + SDK 警告才算")
```

预期输出要点（实跑）：① `count_tokens` 式预检让开发者在下发前算出最坏成本，缓存命中后成本显著下降；② 配额一致性检查显示**同一份配额（4,000 RPM / 2M TPM）隐含平均请求 500 token**：请求画像变成 2,000 token 时 TPM 会先被打满（建议 RPM 降到 1,000），变成 10,000 token 时更早触发——这就是「RPM 没用满却被 429」的根因；③ 错误分类演示了「可重试/不可重试/重试耗尽」三种结局与各自的动作；④ 弃用时间线给出分阶段（告警 → 只读 → 下线）的窗口，**迁移窗口约 180 天**且每阶段都要有可观测告警。

## 常见追问

- **追问**：为什么不直接做 `v2` 而在 `v1` 上兼容演进？
  - 要点：版本爆炸会让 SDK、文档、配额、测试矩阵成倍增长；**优先「加字段 + 忽略未知 + 公告期」**，只在语义必须改变时才发新版本（参考 AIP-180 的兼容性分类）。
- **追问**：如何让开发者知道配额快用完？
  - 要点：响应头回传余量与重置时间 + 控制台可视化 + 阈值告警（webhook/邮件）；并把「配额接近上限」做成**可编程**的信号，让客户端能提前降速或排队（而不是等到 429）。
- **追问**：批处理接口与实时接口怎么共用一套语义？
  - 要点：同一份请求结构（messages/tools），批处理只改**交付语义**（异步、结果落文件、可容忍分钟级）与**计价**；复用同一套错误码与幂等键，减少学习成本。
- **追问**：如何防滥用（盗刷 key、爬取模型）？
  - 要点：多层——密钥作用域与轮换、按组织的异常检测（速率突变、内容模式）、信用额度与预付费、以及对「模型蒸馏式」高频同构请求的识别与限制；同时要保证正常开发者的误报率可接受。
- **追问**：SDK 要做什么、不该做什么？
  - 要点：该做——重试（仅可重试类型）、超时、流式解析、幂等键、token 计数、类型定义；不该做——隐藏错误（吞掉 `stop_reason`）、自动无限重试、隐式改变参数语义。**SDK 的默认行为会成为事实标准**。
- **追问**：怎么验证 API 变更没有破坏兼容性？
  - 要点：契约测试 + 多版本 SDK 的兼容性矩阵 + 影子流量双跑；对「收紧校验」这类隐性破坏做专项测试（用历史请求样本回放）。

## 相关题目

- [[anthropic-17]]：服务栈与准入控制，是本题配额与 429 背后的服务端机制。
- [[anthropic-24]]：大规模检索系统，是本题在「平台能力」上的一个具体下游。
- [[anthropic-08]]：并发调用与退避抖动，是本题错误分类在客户端侧的落地。
- [[system-design-09]]：LLM gateway 的路由与预算，对应本题的接入与配额层。
- [[anthropic-23]]：会话内的多路流式，与本题的流式协议设计互补。

## 参考资料与归属

- **RFC 9110：HTTP Semantics（延伸）** —— IETF (Fielding, Nottingham, Reschke)，2022-06：<https://www.rfc-editor.org/rfc/rfc9110.html>。第 3 节的 4xx/5xx 语义、`Retry-After` 与状态码用法来自这份标准。
- **AIP-158：Pagination（Google API 改进提案）（延伸）** —— Google：<https://google.aip.dev/158>。第 1 节列表接口的分页约定（`page_size`/`page_token`）参照这份提案。
- **AIP-180：Backwards compatibility（Google API 改进提案）（延伸）** —— Google：<https://google.aip.dev/180>。第 2 节「兼容与破坏性变更的分类与测试要求」来自这份提案。
- **Handling Overload（Google SRE Book 第 21 章）（延伸）** —— Google SRE：<https://sre.google/sre-book/handling-overload/>。第 3 节「过载时应拒绝而非排队、并给客户端可重试信号」的取向来自这一章。
- **Exponential Backoff And Jitter（延伸）** —— Marc Brooker (AWS Architecture Blog)，2015-03-04：<https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/>。表 2 与代码第 ③ 段的退避抖动策略来自这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（4,000 RPM、2M TPM、并发 100、\\$3/\\$15 每百万 token、缓存折扣 0.1、请求画像 500/2,000/10,000 token、弃用窗口 30/60/180 天、token 估算 4 字符/token）都是按本仓库统一口径构造的工程算例与显式假设，不代表任何真实平台的配额或定价。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
