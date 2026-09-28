---
type: question
id: evaluation-07
topic: 评估与可观测性
order: 7
question: 生产环境 LLM 系统需要哪些可观测性：traces、spans、成本、反馈？
question_en: What observability does a production LLM system need: traces, spans, cost, feedback?
asked_at: []
level: 进阶
tags: [可观测性, trace, 成本, opentelemetry]
sources:
  - title: AI Agent Observability
    url: https://outcomeschool.com/blog/ai-agent-observability
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: OpenTelemetry: Generative AI semantic conventions（延伸）
    url: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    author: OpenTelemetry
    published: ""
  - title: Arize Phoenix 文档（延伸）
    url: https://docs.arize.com/phoenix
    author: Arize
    published: ""
related: [evaluation-09, evaluation-04, agents-10, inference-serving-09]
updated: 2026-09-28
---

## 一句话答案

> 骨架是四层：metrics、logs、traces 是通用三层，LLM 系统必须再加第四层「质量与成本信号」——judge 结论、检索命中、引用覆盖率、token 成本、cache 命中率。一次请求是一条 trace，span 覆盖 网关/鉴权 → 预处理 → 检索 → prompt 组装 → 模型调用 → 后处理 →（agent 场景）工具调用与观察，每个 span 都带模型版本、prompt 版本、索引版本、代码版本与成本/延迟分解，没有版本标注的日志无法归因。指标分系统、成本、质量、业务四类，全部要能按版本与租户切分；用户反馈必须回连 trace id；敏感内容脱敏后靠分层采样控制成本。与普通后端的根本差别是输出非确定、质量无法由 HTTP 状态码表达，所以质量是一等遥测数据，不是事后补的报表。

## 面试官在考什么

- **可观测性与评估的分工**：observability 回答「实际发生了什么」，evaluation 回答「结果好不好」，两者靠 trace id 与 trace 数据集打通（[[evaluation-04]]、[[evaluation-09]]）。只会报工具名、说不出这条分工的，通常没在真生产里跑过。
- **traces/spans 的字段级设计**：能不能说出 span 树每一层要记什么，尤其是检索 doc id 与分数、prompt 模板版本、usage token、TTFT/TPOT、finish_reason、重试与错误码。
- **版本轴就是可归因性**：模型版本、prompt 版本、检索/索引版本、代码版本。线上「效果变差」的第一问是「哪个版本变了」，四个轴说不全就拿不到区分分。
- **指标是否闭环到动作**：四类指标各自的告警、排查路径与切分维度（版本、租户、语言、渠道）是什么。
- **工程约束**：全量 trace 的存储量、PII 脱敏、多租户隔离、采样与保留期、长时 agent 任务的异步上报，能不能给出量级而不只是形容词。

常见错误答案：

- 「接一个 tracing 平台就有可观测性了」——工具只提供容器，字段与版本语义要自己定义；schema 不统一，换工具时历史数据直接作废。
- 「错误率和延迟正常说明系统健康」——LLM 的多数故障是静默失败：HTTP 200、延迟正常、答案是错的，只有质量信号能看见。

## 原理与推导

### 1. 为什么 LLM 系统要多一层遥测

通用可观测性建立在两个前提上：同一输入得到同一输出；失败会以错误码或异常暴露。LLM 系统两条都不成立。

| 维度 | 普通后端 | LLM 系统 |
| --- | --- | --- |
| 输出 | 确定 | 同输入可得到不同输出、不同执行路径 |
| 失败形态 | 崩溃、错误码、超时 | 静默失败：格式对、内容错、HTTP 200 |
| 成本结构 | CPU/内存/带宽按时间计 | token 计费 + GPU 时间，随输出长度与步数变化 |
| 外部依赖 | 数据库、缓存 | 检索索引、reranker、外部模型、工具，每层都能拖垮质量 |
| 关键问题 | 「跑通了吗」 | 「跑通了，答案对吗，花了多少」 |

所以要在 metrics/logs/traces 之外把**质量**与**成本**本身当成遥测数据：一次请求结束时同时产出「用了多少 token、花了多少钱、检索命中了什么、答案有没有引用、抽样 judge 给几分」。没有这层，回归只能靠用户投诉发现。

### 2. trace 与 span：一次请求的骨架

一次用户请求 = 一个 trace；每个阶段 = 一个 span，span 用 `parent_span_id` 组成树。典型链路：

```text
trace req_8f3c                     tenant=acme  route=/chat  release=2026.09.28-3
├─ span gateway.auth               api_key_id, quota_remaining, model_route
├─ span preprocess.rewrite         lang=zh, rewrite_model, rewritten_query
├─ span retrieval.search           index_version=kb-2026-09-21, top_k=20
│   └─ attr docs[]                 [{doc_id, chunk_id, score}, ...]
├─ span retrieval.rerank           rerank_model, top_n=5, kept_doc_ids
├─ span prompt.build               template=rag-answer@41, tokens=1800, truncated=false
├─ span chat gpt-4.1               usage.in=1800 usage.out=350 cache_read=1200
│                                  ttft=0.42s tpot=12ms finish=stop retries=0
└─ span postprocess.citations      citations=3 verified=3 redactions=1
```

agent 场景把「模型调用」换成循环：每一次 think → tool call → observation 都是一个 span，父子关系表达「第几轮、属于哪个子任务」，工具参数与返回值要落库但正文按脱敏规则处理（[[agents-10]]）。会话（session）是多条 trace 的容器，用 `session_id` 把多轮串起来。

必须有地方落库的字段（trace 级属性，或对应阶段 span 的属性），按可归因性排序：

| 字段 | 示例 | 作用 |
| --- | --- | --- |
| model_version | `gpt-4.1-2025-04-14`、`qwen3-32b@sha256:…` | 同名模型的静默升级会让「模型变差」无法解释 |
| prompt_version | `rag-answer@41` | 与 prompt 发布、回滚记录对齐（[[evaluation-08]]） |
| index_version | `kb-2026-09-21` | 把检索变更与模型变更拆开 |
| release / git_sha | `2026.09.28-3` | 定位代码层回归 |
| tenant / user_hash | `acme`、`u_9c1f` | 切分与隔离；存哈希不存明文 |
| tokens_in / tokens_out / cache_read | 1800 / 350 / 1200 | 成本归因；以 provider 返回的 usage 为准，不要自己估 |
| ttft_ms / tpot_ms / queue_ms | 420 / 12 / 8 | 区分排队、prefill、decode 三段瓶颈（[[inference-serving-09]]） |
| finish_reason / error_code / retries | `stop` / `rate_limit` / 1 | 区分「模型主动拒绝」与「系统失败」 |
| retrieved doc_id + score | 20 召回 → 5 入 prompt | 定位幻觉时先看检索层，而不是先怪模型 |
| quality signals | citation_coverage=1.0、judge=4/5（抽样） | 静默失败的唯一可见途径 |

这四个版本字段不要只写进日志文本，要做成可检索的属性（结构化字段），否则按版本聚合就退化成全文检索。

### 3. 四个版本轴

线上效果变差时，模型、prompt、索引、代码任一变更都会改变输出分布。做法是让四个版本随请求写入 span 或 trace 级属性，看板默认按这四个维度切分；同时给每次请求打上实验分桶（A/B 组、流量来源），否则线上对比会被流量结构变化污染。发布与回滚流程本身属于 prompt 治理（[[evaluation-08]]），可观测性只负责**把版本带进数据**。

### 4. 不能记什么

原始 PII、密钥、完整隐私正文不进 trace。可落地的处理：入口做字段级脱敏（手机号、证件号、卡号、邮箱、地址），正文以引用 id 指向加密存储、trace 只留 id + 长度 + 哈希，`api_key` 只在网关 span 里留末四位或哈希；保留期按租户与合规要求配置，并对 trace 平台自身的查询做访问审计（权限模型与 [[rag-08]] 的检索侧同源）。折中方案是「加密短存 + 采样」：正文加密后保留数天供排障，长期只留结构化字段与哈希。

### 5. 指标清单：四类

| 类别 | 落地指标（3–5 个） | 用法 |
| --- | --- | --- |
| 系统 | QPS、TTFT p50/p95/p99、TPOT 与 E2E p95、错误率与超时率、重试率 | 按 route × model 分维度；TTFT 与 TPOT 分开告警，两者优化手段不同（[[inference-serving-09]]） |
| 成本 | 每请求 input/output token、USD/1M tokens、prefix cache 命中率、每会话成本、GPU 利用率与 goodput | 单位成本环比与预算比例告警；成本下降要确认不是质量换来的（[[inference-serving-12]]） |
| 质量 | 检索命中率、最高相似度 p10、引用覆盖率、抽样 judge 均分、拒答率、幻觉代理指标 | 看滑窗趋势与版本对比，不用绝对阈值（[[evaluation-03]]） |
| 业务 | 会话成功率、建议采纳率、追问/重述率、转人工率、留存 | 与在线实验口径对齐（[[evaluation-09]]） |

硬性要求：**每个指标都要能按 `model_version × prompt_version × tenant × 语言/渠道` 切分**。否则「总体好评率没变」可能掩盖「中文流量跌 8 个点、英文流量涨回来」这类结构性变化，平均值会把反向变化抵消掉。

### 6. 标准与工具：先有 schema，再有工具

字段命名不要自创方言。OpenTelemetry 的 Generative AI 语义约定已经把模型、token、工具调用标准化（该页面现已迁移到独立的 GenAI 约定仓库维护，属性与指标仍标注为 development，处于演进中），可直接复用的部分：

- span 名约定 `{gen_ai.operation.name} {gen_ai.request.model}`，常用操作名 `chat`、`text_completion`、`embeddings`、`retrieval`、`execute_tool`、`invoke_agent`、`create_agent`；
- 模型与用量：`gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.response.finish_reasons`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.usage.cache_read.input_tokens`、`gen_ai.response.time_to_first_chunk`；
- 工具与 agent：`gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.call.arguments`、`gen_ai.tool.call.result`、`gen_ai.agent.name`、`gen_ai.conversation.id`；
- 质量字段：`gen_ai.evaluation.name`、`gen_ai.evaluation.score.value`、`gen_ai.evaluation.explanation`，即 judge 结论本身也走同一套约定；
- 指标名：`gen_ai.client.operation.duration`、`gen_ai.client.operation.time_to_first_chunk`、`gen_ai.client.operation.time_per_output_chunk`、`gen_ai.execute_tool.duration`、`gen_ai.invoke_agent.duration`。

价值是 instrumentation 一次、后端可换；代价是约定仍在演进（属性名曾从 `gen_ai.system` 迁移到 `gen_ai.provider.name`），所以要在采集层加一层归一化映射：规范里已有的量（TTFT、TPOT、token 用量）映射到规范名，规范没覆盖的业务字段（租户、索引版本、实验分组）放独立命名空间（上文表格为可读性用了 `ttft_ms` 这类短名，落库时按这条规则展开），升级约定时不动业务侧代码。

平台形态可参考 Arize Phoenix：trace 树视图、数据集与实验管理、与 eval/judge 联动、prompt 版本管理、数据保留与访问控制（RBAC），也就是把「观测 → 取样 → 评测 → 发布」放进同一条流水线。选型时先问三件事：是否 OTLP 原生（能否换后端）、能否按租户与版本切分、保留期与自托管是否满足合规。

### 7. 采样、保留与长时任务

全量 trace 的存储随流量线性增长（下一节有量级），分层采样是默认做法：

- **必采**：错误与超时、发生重试、judge 低分、用户负反馈、命中内容过滤或拒答的请求；
- **按比例采**：正常请求，比例由要看的最小差异反推；质量指标需要的采样率高于系统指标；
- **聚合替代明细**：QPS、延迟分位数、token 用量由 metrics 承载，不要从 trace 明细现算；
- **保留分级**：明细 7–30 天，脱敏后的结构化字段 6–12 个月，聚合指标长期保留。

用户反馈（点赞点踩、改写、采纳、投诉、转人工）必须带 trace id 落库，才能和当时的模型版本、检索结果、judge 分对齐（[[evaluation-09]]）。agent 这类长时任务（分钟级到小时级）不能等 trace 结束再上报：span 完成即异步上报，用同一 trace id 与父子 span 串起跨进程步骤，避免进程被杀后整条链路丢失；有副作用的工具调用另写审计轨迹（[[agents-10]]）。

### 8. 落地顺序

1. 请求级 trace + 四个版本轴：先让「一次失败请求能完整复盘」成立；
2. 成本与延迟看板：token、缓存命中、延迟分解落到 route/model/租户维度，设预算与分位数告警；
3. 质量抽样与告警：抽样 judge、检索命中、引用覆盖率，先做趋势与版本对比，再谈绝对阈值；
4. 线上数据回流离线 eval：把负反馈与低分 trace 固化成回归集，接入上线门禁，形成「观测 → 取样 → 回归 → 发布」闭环（[[evaluation-04]]）。

跳过第 1 步直接做第 3 步是最常见的返工来源：没有 trace 与版本标注，judge 分再准也只能说明「今天比昨天差」，说不清差在哪一层。

## 数值与代码验证

下面数字是按明确假设的复算，用于量级判断，不是厂商报价。

存储量（假设每请求 8 个 span、每 span 1.5 KB payload，即 12 KB/请求；20 万请求/天；下表按 1 KB = 1024 B、1 GB = 1024 MB 换算）：

| 采样率 | 每天 | 每月（30 天） |
| --- | --- | --- |
| 100% | 2.29 GB | 68.7 GB |
| 5% | 117 MB | 3.43 GB |
| 1% | 23.4 MB | 0.69 GB |
| 0.2% | 4.7 MB | 0.14 GB |

span payload 里 prompt 与 response 正文占大头：正文改存对象存储、trace 只留指针与哈希，可以再降一个量级；而「错误全采」在个位数错误率下只增加几个百分点的量，几乎不影响总成本。

成本归因（示例假设单价 input 3 USD/1M tokens、output 15 USD/1M tokens、cache read 0.3 USD/1M tokens；单请求 1800 input + 350 output，其中 1200 input 命中前缀缓存）：

$$\text{cost} = (n_{in} - n_{cache})\,p_{in} + n_{cache}\,p_{cache} + n_{out}\,p_{out}$$

无缓存时 $0.01065$ USD/请求，命中 1200 token 缓存后 $0.00741$ USD/请求，降 30.4%；20 万请求/天对应 2130 → 1482 USD/天。这类等式必须用 provider 返回的 usage 计算，按字符数估算会系统性偏差。

延迟分解（TTFT 0.42 s、TPOT 12 ms、输出 350 token、后处理 0.18 s）：

$$\text{E2E} \approx \text{TTFT} + (n_{out}-1)\times\text{TPOT} + t_{post}$$

代入得 $0.42 + 349 \times 0.012 + 0.18 = 4.79$ s，其中 decode 占 87.5%、TTFT 占 8.8%、后处理占 3.8%。含义是这条链路的延迟预算在 decode：优化 TTFT 只改善首字体验，不会改善 E2E。

分位数不能跨实例平均。两个 pod 各 5000 个合成延迟样本（固定随机种子），p99 分别为 2.25 s 与 6.01 s，两者平均 4.13 s；把样本合并后真实的全局 p99 是 5.49 s，简单平均低估 24.8%。所以 p99 只能在聚合层由直方图或草图（histogram、t-digest、CKMS）计算，不能对实例 p99 取平均。

抽样精度：单组比例在 95% 置信度下的区间半宽为 $\delta$ 时，

$$n \approx \frac{z^2 p(1-p)}{\delta^2}, \quad z = 1.96$$

错误率基线 2% 时，要分辨 ±0.2pp 需要约 1.88 万个可判样本，分辨 ±1pp 只需约 750 个；基线 10% 时分辨 ±1pp 需要约 3500 个。比较两组差异时样本量还要再乘约 2。含义：高频错误可以靠低采样率告警，低频质量指标（幻觉、严重错误）必须提高采样率或直接全采，否则看板的波动全是采样噪声。

```python
"""trace 侧的三件事：成本、分位数、分层采样。Python 3.10+"""
import math, random
from dataclasses import dataclass

P_IN, P_OUT, P_CACHE = 3.0, 15.0, 0.30   # USD / 1M tokens，示例假设

@dataclass
class Span:
    name: str
    model: str
    prompt_version: str
    index_version: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    error: str | None = None

def request_cost(spans):
    tin = sum(s.tokens_in for s in spans)
    tcache = sum(s.cache_read for s in spans)
    tout = sum(s.tokens_out for s in spans)
    return ((tin - tcache) * P_IN + tcache * P_CACHE + tout * P_OUT) / 1e6

def p99(xs):                      # 必须在同一个数据集内计算
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.99 * len(xs)) - 1)]

def should_sample(span, trace_flags, rate=0.01):
    """分层采样：错误 / 慢 / 负反馈全采，其余按比例"""
    if span.error or trace_flags.get("slow") or trace_flags.get("thumbs_down"):
        return True
    return random.random() < rate

spans = [Span("chat", "gpt-4.1", "rag-answer@41", "kb-2026-09-21",
              tokens_in=1800, tokens_out=350, cache_read=1200,
              ttft_s=0.42, tpot_s=0.012)]
print(f"cost/req = {request_cost(spans):.5f} USD")

rng = random.Random(7)
pod_a = [max(0.05, rng.gauss(0.9, 0.6)) for _ in range(5000)]
pod_b = [max(0.05, rng.gauss(1.6, 1.9)) for _ in range(5000)]
avg = (p99(pod_a) + p99(pod_b)) / 2
print(f"p99 A={p99(pod_a):.2f}s B={p99(pod_b):.2f}s 平均={avg:.2f}s 全局={p99(pod_a + pod_b):.2f}s")
```

输出：`cost/req = 0.00741 USD`；`p99 A=2.25s B=6.01s 平均=4.13s 全局=5.49s`，与上文的成本与分位数结论一致；存储量级由上面的假设直接算出，不依赖这段代码。

## 常见追问

- **追问**：脱敏和可调试性冲突时怎么取舍？
  - 要点：分层处理。trace 默认只留结构化字段（长度、哈希、分数、版本、错误码）；正文加密短存（数天）且只对授权角色可见，配访问审计；需要长期留存的做实体级替换（保留结构与长度、替换敏感实体），代价是引用校验类问题会失真。复现能力的下限是「输入哈希 + 版本 + 检索快照」，这三样不脱敏也能定位大部分回归。
- **追问**：多租户如何隔离 traces？
  - 要点：写入侧把 tenant 设为必填属性并在采集管道校验，缺 tenant 的 span 直接拒绝或进隔离区；存储侧按租户分索引/分区 + 字段级权限；查询侧把租户过滤做成强制默认条件，防止越权查询；加密密钥与保留期按租户配置；跨租户只保留不可反推的聚合统计。
- **追问**：agent 的 trace 与普通请求有什么不同？
  - 要点：一条 trace 跨越分钟到小时、跨进程与跨服务，必须异步上报并传播 context；span 数量与 token 消耗高一个量级，步数本身是最重要的成本与健康指标；工具调用有副作用，要记录「谁在什么授权下改了什么」并写入审计日志；还要能检测循环（重复工具 + 重复参数）与终止原因（[[agents-10]]）。
- **追问**：采样率怎么定？错误全采会不会丢掉正常基线？
  - 要点：错误全采解决「看见异常」，正常流量按比例采解决「有可比基线」；比例由要分辨的最小差异反推（上面公式），质量指标基线低、方差大，需要比系统指标更高的采样率；样本要随机化并固定时间窗与版本，用「最近 N 条」会引入时段偏差。
- **追问**：怎么把线上 trace 变成回归集与门禁？
  - 要点：三类来源——负反馈、judge 低分、人工标注；先清洗去重（按输入语义聚簇，避免同义样本刷屏），再按任务类型分层抽样；每条样本固化输入、期望要点（不一定要唯一参考答案）、当时的检索快照与版本；上线前跑对比实验，按任务类型设通过门槛（[[evaluation-04]]、[[evaluation-09]]）。
- **追问**：可观测性本身的成本怎么算？
  - 要点：三块——存储（采样率 × payload 大小）、写入链路（采集代理、队列、采样器的 CPU 与网络）、以及对被观测系统的影响（同步上报会污染 TTFT，必须异步批量上报）。把观测成本计入单位请求成本，和模型成本放在同一张看板上，超阈值就降 payload 或调采样率。

## 相关题目

- [[evaluation-09]]：在线评估记录什么日志、采样什么、对什么做 A/B——反馈与 trace id 的对接。
- [[evaluation-04]]：回归门禁；把线上低分 trace 固化成门禁用例。
- [[evaluation-08]]：prompt 版本与回滚；trace 里的 `prompt_version` 与发布记录对齐。
- [[evaluation-03]]：生产 RAG 的幻觉检测；质量指标与代理指标的细节。
- [[agents-10]]：agent 操作的审计轨迹；长时任务与副作用记录。
- [[inference-serving-09]]：TTFT/TPOT/ITL 口径；延迟分解用的指标定义。
- [[inference-serving-12]]：降本手段与排序；成本看板要能验证每一项是否真的生效。
- [[rag-08]]：权限感知检索；trace 存储的脱敏与权限模型同源。

## 参考资料与归属

- **AI Agent Observability**，Amit Shekhar（Outcome School），2026-05-27：[链接](https://outcomeschool.com/blog/ai-agent-observability)。三层信号（logs/metrics/traces）与 session-trace-span 层级、traces 对 agent 最重要、可观测性和评估的分工、控制记录量与脱敏等实践来自这篇；分层采样「错误与慢请求全采、正常按比例采」的具体分层方式是按这些实践给出的落地做法。
- **OpenTelemetry: Generative AI semantic conventions（延伸）**，OpenTelemetry：[链接](https://opentelemetry.io/docs/specs/semconv/gen-ai/)。span 命名约定、`gen_ai.*` 属性与 GenAI 指标名来自这里；正文中提到的迁移与 development 稳定性也来自该页面的现状说明。
- **Arize Phoenix 文档（延伸）**，Arize：[链接](https://docs.arize.com/phoenix)。trace 视图、数据集与实验、eval 联动、prompt 管理、保留期与 RBAC 等平台形态来自这里。

存储量、成本、延迟分解、分位数与采样精度的数字是按正文给出的假设自行复算的结果，不是上述来源提供的数据。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
