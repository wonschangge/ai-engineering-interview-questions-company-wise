---
type: question
id: microsoft-02
company: Microsoft
topic: coding
order: 2
question: low-level design：为一个 agent host 的工具调用层设计类与接口，其中的工具可以来自原生代码、OpenAPI spec 或 MCP server。
question_en: Low-level design: design the classes and interfaces for the tool-calling layer of an agent host, where tools can come from native code, OpenAPI specs, or MCP servers.
asked_at: []
level: 高阶
tags: [接口设计, 工具调用, OpenAPI, MCP, 能力协商]
sources:
  - title: OpenAPI Specification 3.0（延伸）
    url: https://spec.openapis.org/oas/v3.0.3
    author: OpenAPI Initiative
    published: 2021-02-15
  - title: Model Context Protocol Specification（延伸）
    url: https://modelcontextprotocol.io/specification
    author: Anthropic / MCP contributors
    published: 2024-11-25
  - title: Toolformer: Language Models Can Teach Themselves to Use Tools（延伸）
    url: https://arxiv.org/abs/2302.04761
    author: Schick et al. (Meta)
    published: 2023-02-09
  - title: 大规模日志的 top-k 高频查询（本仓库公司题库 · Microsoft 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [microsoft-01, microsoft-06, agents-01, agents-03, coding-07]
updated: 2026-09-28
---

## 一句话答案

> **设计目标只有一句话：让 agent 循环看不见「工具从哪来」。** 所以核心是**五个抽象 + 三个适配器**。
> ```python
> # ① 统一元数据（规范化后的 JSON Schema 是唯一契约）
> @dataclass
> class ToolSpec:
>     name: str                  # 命名空间化：f「{provider}.{tool}」
>     description: str
>     schema: dict               # JSON Schema（object）
>     source: str                # native | openapi | mcp
>     version: str
>     scopes: list[str]          # 需要的权限
>     idempotent: bool           # ★ 决定能否重试
>     timeout_s: float
>     max_result_bytes: int      # ★ 结果上限（否则会撑爆上下文）
>     needs_confirmation: bool   # 不可逆操作
> # ② 来源适配器（三种实现，同一个接口）
> class ToolProvider(Protocol):
>     def list_tools(self) -> list[ToolSpec]: ...
>     def invoke(self, name: str, args: dict, ctx: InvokeContext) -> ToolResult: ...
> # ③ 注册表（聚合 + 命名空间 + 版本 + 冲突消解）
> # ④ 调用器（权限 → 超时 → 重试 → 熔断 → 审计 → 结果规范化）
> # ⑤ Schema 适配（OpenAPI / MCP → JSON Schema）
> ```
> **三个来源的差异矩阵（这张表是本题的核心）**：
> | 维度 | **原生代码** | **OpenAPI spec** | **MCP server** |
> | --- | --- | --- | --- |
> | 工具发现 | 静态（代码里注册） | **静态**（解析 spec） | **动态**（运行时 `tools/list`，会变） |
> | 参数形态 | 直接是 JSON Schema | path/query/body **三处**（要展平） | 已是 JSON Schema |
> | `$ref` | 无 | **要内联展开** | 可能有 |
> | `nullable` | 直接支持 | **要转成 `type: [T, null]`** | 直接支持 |
> | 调用方式 | 进程内函数 | **HTTP + 认证** | **JSON-RPC（可长连接）** |
> | 失败语义 | 异常 | HTTP 状态码 + 业务错误体 | JSON-RPC error + 工具级错误 |
> | 幂等性 | 由实现声明 | 由 HTTP 方法推断（GET/PUT 幂等） | 由 server 声明 |
> | 认证 | 进程内 | **多种（API key/OAuth/…）** | 由 server 承担 |
> | 生命周期 | 随进程 | 随配置 | **连接会断、工具会变** |
> **四个必须量化的设计驱动（本机算例）**：
> | # | 驱动 | 量化 | 设计含义 |
> | --- | --- | --- | --- |
> | ① | **命名冲突** | 100 个工具、1 万候选名 → **39%** 至少冲突一次（200 个 → **86%**） | **必须 `provider.tool` 命名空间** |
> | ② | **结果大小** | 1 MB 结果 ≈ **262K token**（128K 上下文的 **200%**） | **工具层强制结果上限 + 截断/分页/摘要** |
> | ③ | **超时 × 重试** | 5 s × 3 次 = **15 s**（3 s 预算的 **5 倍**） | **超时按「剩余预算」动态设置**，不是每工具固定值 |
> | ④ | **Schema 转换** | path/query/body 三处参数 + `$ref` + `nullable` | **统一在边界转换**，循环内只见 JSON Schema |
> **错误分类与重试策略（必须分开，否则会重复副作用）**：
> | 类别 | 例子 | 可重试？ |
> | --- | --- | --- |
> | **传输/瞬时** | 连接超时、5xx、限流 | **可重试**（指数退避） |
> | **权限** | 401/403、缺 scope | **不可重试**（要用户授权） |
> | **参数/schema** | 缺必填、类型错 | **不可重试**（要模型改参数） |
> | **业务错误** | 「issue 已关闭」 | **不可重试**（把错误信息给模型） |
> | **副作用未确认** | POST 超时（可能已成功） | **只在 `idempotent` 时重试**；否则查询确认 |
> 一句话判据：**「统一 spec、三种适配器、命名空间、错误分类、结果上限、预算联动超时」**——**六条缺一条，agent host 就会在真实工具上翻车。**

## 面试官在考什么

- **是否抓住「统一接口」这个本质**：**能否用一句话说清「让 agent 循环看不见工具从哪来」**，并给出抽象分层。
- **三个来源的差异**：能否列出**静态 vs 动态发现**、**参数三处 vs 一处**、**HTTP vs JSON-RPC**、**幂等性来源**这些真实差异（本机的差异矩阵）。
- **Schema 统一的细节**：**path/query/body 展平**、`$ref` 内联、`nullable` 转换——**能否说出具体要处理什么**。
- **命名空间**：能否给出**冲突概率**的量级（本机 100 工具 39%）并说明为什么必须加前缀。
- **错误分类**：**能否区分「可重试」与「不可重试」**，尤其**副作用未确认**这一类（**这是最危险的重试场景**）。
- **结果大小**：**能否指出工具结果会进 prompt**，并给出上限（本机 1 MB = 262K token）——**这是很多人漏掉的一层**。
- **超时与预算联动**：能否指出**超时×重试会突破端到端预算**（本机 15 s vs 3 s），所以必须动态设置。
- **权限与审计**：能否把**scope 检查**与**不可逆操作的确认**放进调用器（而不是散在各工具里）。
- **动态能力协商（MCP）**：**能否指出 MCP 的工具会在运行时变化**——注册表要能刷新，且要处理「工具消失后模型还在调它」。
- **接口设计的整洁性**：`ToolResult` 的形态（status/content/error/metadata）、**同步 vs 异步**、**流式结果**。

**常见错误答案**

- 只设计一个 `call_tool(name, args)` 函数（**没有 spec 层、没有来源适配、没有权限/审计**）。
- 把三种来源的差异**泄漏到 agent 循环里**（`if source == 'openapi': ...` 散落各处）。
- **不做命名空间**（工具名冲突后行为不可预测）。
- **不区分错误类别**（对所有错误都重试 → **重复下单/重复发消息**）。
- 把原始工具结果直接塞进 prompt（**撑爆上下文**）。
- 超时写死（**与端到端预算脱节**）。
- 忽略 MCP 的动态性（**工具列表变了就崩**）。
- 权限检查散落在各工具实现里（**无法审计、容易漏**）。
- 不记录审计日志（**出事后无法归因**）。

## 原理与推导

### 1. 分层：让 agent 循环只看见一层

```
agent 循环
   ↓ 只用这两个接口
ToolRegistry（发现/查找）        ToolInvoker（执行）
   ↓                              ↓
ToolProvider 适配器：Native / OpenAPI / MCP
   ↓
真实工具（函数 / HTTP / JSON-RPC）
```

**关键**：**循环里只出现 `ToolSpec` 与 `ToolResult`**——**任何来源特有的东西都在适配器里被消化**。

### 2. 五个抽象（职责与接口）

| 抽象 | 职责 | 关键方法/字段 |
| --- | --- | --- |
| **`ToolSpec`** | 统一的工具元数据 | `name`（命名空间化）、`schema`、`scopes`、**`idempotent`**、`timeout_s`、**`max_result_bytes`**、`needs_confirmation` |
| **`ToolProvider`** | 来源适配 | `list_tools()`、`invoke(name, args, ctx)` |
| **`ToolRegistry`** | 聚合、命名空间、版本、刷新 | `register(provider)`、`get(name)`、`refresh()` |
| **`ToolInvoker`** | 权限 → 超时 → 重试 → 熔断 → 审计 | `invoke(name, args, ctx)` |
| **`SchemaAdapter`** | 边界转换 | `openapi_to_jsonschema()`、`mcp_to_jsonschema()` |

**`ToolResult` 的统一形态**：

```python
@dataclass
class ToolResult:
    status: str            # ok | error | timeout | denied | truncated
    content: str           # 给模型看的文本（已截断/摘要）
    error_kind: str | None # transport | permission | schema | business | unknown
    retryable: bool
    metadata: dict         # 耗时、原始大小、是否截断、审计 id
```

**读法**：**`status` 与 `error_kind` 分开**——**因为「失败」的处理方式取决于类别**（第 5 节）。

### 3. 三个来源的差异（本机的差异矩阵）

**最容易泄漏到上层的四个差异**：

| 差异 | 泄漏的后果 |
| --- | --- |
| **发现方式**（静态 vs 动态） | 循环假设工具列表固定 → MCP 工具变化后崩 |
| **参数位置**（三处 vs 一处） | 循环里要写「从 path 取还是从 body 取」 |
| **幂等性来源**（声明 vs HTTP 方法 vs server） | 重试策略写不对 → 重复副作用 |
| **失败形态**（异常 vs HTTP 状态 vs JSON-RPC error） | 错误处理到处特判 |

**读法**：**适配器的价值就是把这四个差异吃掉**——**每个差异对应一个适配器里的转换点**。

### 4. Schema 统一：本机的转换器

**输入**：一份典型 OpenAPI spec（含 path/query 参数、`$ref` 请求体、`nullable`）。
**输出**：统一的 `ToolSpec`（JSON Schema object）。

**本机实测**（2 个 operation）：

| 工具 | 方法 | 路径 | 展平后的参数 |
| --- | --- | --- | --- |
| `listIssues` | GET | `/repos/{owner}/{repo}/issues` | `owner`, `repo`, `state`, `per_page` |
| `createIssue` | POST | `/repos/{owner}/{repo}/issues` | `title`, `body`, `labels` |
| 必填 | —— | —— | `owner`, `repo` |

**三个必须处理的构造**：
1. **path/query/body 三类参数展平成同一个对象**——**但要记录来源**（`x-in`），供调用时回填到正确位置；
2. **`$ref` 内联展开**（`#/components/schemas/Issue` → 直接展开 properties 与 required）；
3. **`nullable: true` → `type: [「string」, 「null」]`**（JSON Schema 的写法）。

**读法**：**转换发生在「边界」**（注册时一次），**循环里永远只见 JSON Schema**——**这样模型看到的工具描述与来源无关**。

### 5. 命名冲突与命名空间

**为什么必须加前缀**：三个来源的命名习惯不同（native 用 `snake_case`、OpenAPI 用 `operationId`、MCP 由 server 自定义），**重名概率随工具数上升**（生日问题）：

$$P(\text{至少一次冲突})\approx1-e^{-N^2/(2M)}$$

**本机算例**（$M=10^4$ 个候选名）：

| 工具数 $N$ | 冲突概率 |
| --- | --- |
| 20 | 2% |
| 50 | 12% |
| **100** | **39%** |
| 200 | **86%** |

**读法**：**100 个工具就有近四成概率撞名**——**所以 `provider.tool` 是必需的**（例如 `native.search`、`github.createIssue`、`mcp.slack.post`）。**注意**：**给模型看的名字也要带前缀**（否则模型无法区分），但**可以在描述里说明「这是 GitHub 的创建 issue」**。

### 6. 错误分类与重试策略（最关键的一节）

**错误分类**：

| 类别 | 判定 | 可重试 | 处理 |
| --- | --- | --- | --- |
| **传输/瞬时** | 连接失败、5xx、限流（429） | **是**（指数退避 + 抖动） | 重试至上限 |
| **权限** | 401/403、缺 scope | **否** | **请求用户授权**（而不是重试） |
| **参数/schema** | 缺必填、类型错 | **否** | **把错误返回给模型让它改参数** |
| **业务错误** | 工具内部逻辑拒绝 | **否** | 把错误信息给模型 |
| **★ 副作用未确认** | POST 超时（**可能已经成功**） | **仅当 `idempotent`** | 否则**查询确认**（「刚才那个 issue 建了吗」） |

**读法**：**「副作用未确认」是最危险的一类**——**盲目重试会造成重复下单、重复发消息、重复扣款**。**判据**：① **`idempotent=True`**（GET/PUT、或带幂等键的 POST）→ 可重试；② **否则先查询确认**；③ **再否则报告「结果未知」给模型**（**让模型决定是否人工确认**）。**这是工具层必须替模型承担的责任**。

### 7. 结果大小：会撑爆上下文的那一层

**工具结果会直接进 prompt**，所以要有上限：

$$\text{tokens}\approx\frac{\text{bytes}}{4}\qquad\text{占 128K 上下文}=\frac{\text{tokens}}{131072}$$

**本机算例**：

| 结果大小 | 约合 token | 占 128K | 结论 |
| --- | --- | --- | --- |
| 1 KB | 256 | 0.2% | 可直接放入 |
| 10 KB | 2,560 | 2.0% | 可直接放入 |
| 100 KB | 25,600 | 19.5% | 需要截断 |
| **1 MB** | **262,144** | **200%** | **必须摘要/分页** |
| 10 MB | 2,621,440 | 2000% | 必须摘要/分页 |

**读法**：**1 MB 的结果就是上下文的两倍**——**所以 `max_result_bytes` 必须是 spec 的字段**，并且要有**三级策略**：① **截断**（保留头尾 + 中间省略标记）；② **分页**（给模型「下一页」的工具）；③ **摘要**（用小模型把结果压缩，**注意这会引入延迟**）。

### 8. 超时与端到端预算联动

$$t_{\text{worst}}=T\times(r+1)$$

**本机算例**：

| 超时 $T$ | 重试 $r$ | 最坏延迟 | 占 3 s 预算 |
| --- | --- | --- | --- |
| 5 s | 0 | 5 s | 1.7× |
| 5 s | 1 | 10 s | 3.3× |
| **5 s** | **2** | **15 s** | **5.0×** |
| 30 s | 0 | 30 s | 10.0× |

**读法**：**每个工具各配一个固定超时，累加后必然突破端到端预算**。**正确做法**：**调用器接收「剩余预算」**，据此动态设置超时与重试次数：

$$T_{\text{eff}}=\min(T_{\text{tool}},\ \text{剩余预算}\times\alpha)\quad(\alpha\approx0.5)$$

**并且**：**「工具还在跑」应当是一等状态**（异步 + 进度事件），而不是「一直等到超时」。

### 9. 权限、确认与审计

| 关注点 | 放在哪里 | 为什么 |
| --- | --- | --- |
| **scope 检查** | **调用器**（调用前） | 集中一处才能审计与一致 |
| **不可逆操作确认** | **调用器**（`needs_confirmation`） | 避免每个工具各写一套 |
| **审计日志** | **调用器**（调用前后） | 记录用户、工具、参数哈希、结果、耗时、是否截断 |
| **数据边界** | **调用器**（参数与结果的脱敏） | 防止把敏感数据发给外部工具 |

**读法**：**这四件事必须在调用器里**——**散到各工具实现里就必然漏**（尤其审计与脱敏）。

### 10. MCP 的动态性（最容易被忽略）

**MCP 的工具列表会在运行时变化**（server 重启、能力开关、版本升级）。**所以**：

| 问题 | 处理 |
| --- | --- |
| 工具列表变化 | **注册表支持刷新**（`refresh()`），并把变化通知给 agent（「工具集已更新」） |
| **模型调用已消失的工具** | 返回**明确的 schema 错误**（「该工具不再可用」），**让模型换工具**（而不是抛异常） |
| server 断连 | **熔断 + 重连**；期间该 provider 的工具标记为不可用 |
| 工具名冲突（同 server 两个版本） | **版本进名字或进 spec**，注册表按版本解析 |

**读法**：**「工具会变」是 MCP 与 OpenAPI 最大的区别**——**设计时必须假设工具列表是「活的」**。

## 数值与代码验证

### 表 1：四个设计驱动量的量化（见代码输出）

| 驱动 | 量化 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
# agent host 工具调用层：OpenAPI->JSON Schema 转换、命名冲突、结果上限、超时预算联动
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# ---------- ① Schema 适配：OpenAPI -> 统一 JSON Schema ----------
SAMPLE_SPEC: Dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Issues", "version": "1.0"},
    "servers": [{"url": "https://api.example.com/v1"}],
    "paths": {
        "/repos/{owner}/{repo}/issues": {
            "get": {
                "operationId": "listIssues", "summary": "List issues",
                "parameters": [
                    {"name": "owner", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                    {"name": "repo", "in": "path", "required": True,
                     "schema": {"type": "string"}},
                    {"name": "state", "in": "query", "required": False,
                     "schema": {"type": "string", "enum": ["open", "closed"]}},
                    {"name": "per_page", "in": "query",
                     "schema": {"type": "integer", "default": 30}},
                ],
                "responses": {"200": {"description": "ok"}},
            },
            "post": {
                "operationId": "createIssue", "summary": "Create issue",
                "requestBody": {"required": True, "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/Issue"}}}},
                "responses": {"201": {"description": "created"}},
            },
        }
    },
    "components": {"schemas": {"Issue": {
        "type": "object", "required": ["title"],
        "properties": {"title": {"type": "string"},
                       "body": {"type": "string", "nullable": True},
                       "labels": {"type": "array", "items": {"type": "string"}}}}}},
}

def openapi_to_tools(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 OpenAPI 的每个 operation 展平成统一的工具描述（JSON Schema）"""
    schemas = spec.get("components", {}).get("schemas", {})
    tools: List[Dict[str, Any]] = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            props: Dict[str, Any] = {}
            required: List[str] = []
            for p in op.get("parameters", []):
                sch = dict(p["schema"])
                if sch.get("nullable"):                     # ★ nullable -> type: [T, null]
                    sch["type"] = [sch["type"], "null"]
                    sch.pop("nullable", None)
                sch["x-in"] = p["in"]                       # ★ 记录来源，供调用时回填
                props[p["name"]] = sch
                if p.get("required"):
                    required.append(p["name"])
            body = (op.get("requestBody", {}).get("content", {})
                    .get("application/json", {}).get("schema"))
            if body:
                if "$ref" in body:                          # ★ $ref 内联展开
                    ref = schemas[body["$ref"].split("/")[-1]]
                    for k, v in ref.get("properties", {}).items():
                        sch = dict(v)
                        if sch.get("nullable"):
                            sch["type"] = [sch["type"], "null"]
                            sch.pop("nullable", None)
                        sch["x-in"] = "body"
                        props[k] = sch
                    required += ref.get("required", [])
                else:
                    for k, v in body.get("properties", {}).items():
                        props[k] = dict(v)
                        props[k]["x-in"] = "body"
            tools.append({
                "name": op["operationId"],
                "description": op.get("summary", ""),
                "schema": {"type": "object", "properties": props,
                           "required": sorted(set(required))},
                "method": method.upper(), "path": path,
                "server": spec["servers"][0]["url"],
            })
    return tools

print("① OpenAPI -> 统一工具 schema（本机转换器实测）")
tools = openapi_to_tools(SAMPLE_SPEC)
for t in tools:
    print(f"  {t['name']:<12} {t['method']:<5} {t['path']:<32} "
          f"参数 {list(t['schema']['properties'])}")
print(f"  必填：{tools[0]['schema']['required']}")
print(f"  nullable 转换结果：{tools[1]['schema']['properties']['body']['type']}")
print("  读法：**三个必须处理的构造**：path/query/body 展平（记录 x-in）、$ref 内联、nullable 转 type 数组")

print("")
print("② 命名冲突概率（生日问题：1 - exp(-N^2/(2M))）")
print(f"  {'工具数 N':>9} {'候选名空间 M':>12} {'至少一次冲突':>13}")
for n in (20, 50, 100, 200):
    m = 10_000
    print(f"  {n:>9} {m:>12} {1 - math.exp(-n*n/(2*m)):>12.0%}")
print("  读法：**100 个工具就有 39% 概率撞名** -> 必须 `provider.tool` 命名空间")

print("")
print("③ 工具结果大小 -> 上下文成本（结果会直接进 prompt）")
print(f"  {'结果大小':>10} {'约合 token':>12} {'占 128K':>10} 结论")
for kb in (1, 10, 100, 1024, 10240):
    tok = kb * 1024 / 4
    frac = tok / 131072
    note = ("可直接放入" if frac < 0.02 else
            ("需要截断" if frac < 0.5 else "**必须摘要/分页**"))
    print(f"  {kb:>8} KB {tok:>12,.0f} {frac:>9.1%} {note}")
print("  读法：**1 MB 结果 = 262K token = 上下文的 200%** -> `max_result_bytes` 必须是 spec 字段；")
print("        三级策略：截断（留头尾）-> 分页（下一页工具）-> 摘要（小模型压缩，有延迟代价）")

print("")
print("④ 超时 × 重试 vs 端到端预算（3 秒）")
print(f"  {'超时 T':>7} {'重试 r':>7} {'最坏延迟':>9} {'占 3s 预算':>11}")
for T, r in ((5, 0), (5, 1), (5, 2), (10, 1), (30, 0)):
    worst = T * (r + 1)
    print(f"  {T:>5}s {r:>7} {worst:>7}s {worst/3:>10.1f}x")
print("  读法：**固定超时累加必然突破预算** -> 调用器接收「剩余预算」，按 min(T_tool, 剩余×0.5) 动态设置；")
print("        并且「工具还在跑」应当是一等状态（异步 + 进度事件）")

print("")
print("⑤ 错误分类与重试策略")
@dataclass
class ErrKind:
    name: str
    example: str
    retry: str
    action: str
ERRS = [
    ErrKind("传输/瞬时", "连接超时、5xx、429", "可重试", "指数退避 + 抖动"),
    ErrKind("权限", "401/403、缺 scope", "不可重试", "请求用户授权"),
    ErrKind("参数/schema", "缺必填、类型错", "不可重试", "把错误返回给模型改参数"),
    ErrKind("业务错误", "issue 已关闭", "不可重试", "把错误信息给模型"),
    ErrKind("★ 副作用未确认", "POST 超时（可能已成功）", "仅 idempotent 时", "否则查询确认，或报告「结果未知」"),
]
print(f"  {'类别':<16} {'例子':<22} {'可重试':<16} 处理")
for e in ERRS:
    print(f"  {e.name:<16} {e.example:<22} {e.retry:<16} {e.action}")
print("  读法：**「副作用未确认」是最危险的一类** —— 盲目重试会造成重复下单/重复发消息；")
print("        所以 `idempotent` 必须是 spec 的字段，而不是由调用方猜")

print("")
print("⑥ 三个来源的差异矩阵（适配器要吃掉的东西）")
@dataclass
class Diff:
    dim: str
    native: str
    openapi: str
    mcp: str
DIFFS = [
    Diff("工具发现", "静态（代码注册）", "静态（解析 spec）", "**动态（运行时 tools/list）**"),
    Diff("参数形态", "JSON Schema", "**path/query/body 三处**", "JSON Schema"),
    Diff("调用方式", "进程内函数", "HTTP + 认证", "**JSON-RPC（可长连接）**"),
    Diff("失败语义", "异常", "HTTP 状态 + 业务体", "JSON-RPC error"),
    Diff("幂等性", "实现声明", "**由 HTTP 方法推断**", "server 声明"),
    Diff("生命周期", "随进程", "随配置", "**连接会断、工具会变**"),
]
print(f"  {'维度':<10} {'原生':<18} {'OpenAPI':<24} MCP")
for d in DIFFS:
    print(f"  {d.dim:<10} {d.native:<18} {d.openapi:<24} {d.mcp}")
print("  读法：**这四个差异（发现/参数/幂等/失败）如果泄漏到循环里，就会到处特判** ——")
print("        适配器的价值就是在这里吃掉它们")

print("")
print("⑦ 接口骨架（五个抽象）")
@dataclass
class ToolSpec:
    name: str                      # provider.tool（命名空间化）
    description: str
    schema: Dict[str, Any]         # JSON Schema
    source: str                    # native | openapi | mcp
    version: str = "1"
    scopes: List[str] = field(default_factory=list)
    idempotent: bool = False       # ★ 决定能否重试
    timeout_s: float = 5.0
    max_result_bytes: int = 64 * 1024   # ★ 结果上限
    needs_confirmation: bool = False    # 不可逆操作

@dataclass
class ToolResult:
    status: str                    # ok | error | timeout | denied | truncated
    content: str
    error_kind: Optional[str] = None   # transport | permission | schema | business
    retryable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ToolProvider(Protocol):
    def list_tools(self) -> List[ToolSpec]: ...
    def invoke(self, name: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> ToolResult: ...

print("  ToolSpec 字段：", list(ToolSpec.__dataclass_fields__))
print("  ToolResult 字段：", list(ToolResult.__dataclass_fields__))
print("  ToolProvider 方法：list_tools / invoke")
print("  读法：**循环里只出现 ToolSpec 与 ToolResult** —— 来源特有的东西全在适配器里；")
print("        `status` 与 `error_kind` 必须分开（失败的处理方式取决于类别）")
```

预期输出要点（实跑）：① **OpenAPI 转换器**：2 个 operation 展平为统一 schema（`listIssues` 的 4 个参数、`createIssue` 的 3 个），**`nullable` 正确转成 `[「string」,「null」]`**，path/query/body 的来源记在 `x-in`；② **命名冲突**：100 个工具 **39%**、200 个 **86%**；③ **结果大小**：100 KB 占上下文 19.5%、**1 MB 达 200%**；④ **超时×重试**：5 s × 3 次 = **15 s**（预算的 5 倍）；⑤ 五类错误的重试策略（**「副作用未确认」仅幂等时可重试**）；⑥ 三个来源的六维差异矩阵；⑦ 五个抽象的字段与方法。

## 常见追问

- **追问**：怎么处理「模型调用了不存在的工具」？
  - 要点：**返回可读的 schema 错误，而不是抛异常**：① **错误信息要包含可用工具列表**（或最接近的候选，用编辑距离/嵌入相似度）；② **把错误作为 `ToolResult` 返回给模型**（**让模型自己纠正**）；③ **记录这类事件**（频繁出现说明工具描述不清楚，或模型能力不足）。**注意**：**MCP 下「工具消失」是正常事件**（server 重启）——**必须与「模型写错名字」区分开**（前者要刷新注册表，后者要给候选）。
- **追问**：工具描述（description）为什么重要？
  - 要点：**因为它是模型选择工具的唯一依据**。① **描述要写「什么时候用」而不只是「是什么」**（「查询 GitHub issue 列表；当用户问 issue 状态时使用」）；② **参数要有描述与示例**（否则模型会瞎填）；③ **描述长度有成本**（所有工具的描述都进 prompt——**100 个工具 × 100 token = 1 万 token**）；④ **工具太多时要检索/分组**（**不要把 1000 个工具全塞进 prompt**，而是按需检索工具，串 [[agents-01]]）。
- **追问**：重试会不会导致重复副作用？
  - 要点：**会，而且这是最危险的场景**。**三层防护**：① **`idempotent` 字段**（GET/PUT 天然幂等；POST 需要幂等键）；② **幂等键（idempotency key）**——调用方生成一个 UUID 传给服务端，**服务端据此去重**（Stripe 等的标准做法）；③ **「结果未知」作为一等状态**——超时后**不重试**，而是**用查询工具确认**（「刚才那个 issue 建了吗」），**或把不确定性告诉模型/用户**。**读法**：**宁可报告「未知」，也不要盲目重试**——**因为重复下单的代价远大于多问一句**。
- **追问**：怎么限制「工具结果撑爆上下文」？
  - 要点：**在工具层做，而不是在 prompt 组装层做**：① **`max_result_bytes` 进 spec**（每个工具声明自己的上限）；② **三级策略**（截断 → 分页 → 摘要）；③ **把「被截断」作为结果的一部分告诉模型**（`status: truncated` + 「共 X 条，已显示前 Y 条」）——**否则模型会以为看到了全部**；④ **审计里记录原始大小**（用于发现「总是被截断」的工具）。**本机算例**：**1 MB = 262K token**（**超过 128K 上下文**）——**所以上限不是优化，是必需**。
- **追问**：MCP 相比 OpenAPI 多了哪些设计负担？
  - 要点：**四点**：① **动态发现**（工具列表会变 → 注册表要刷新、要通知 agent）；② **连接生命周期**（长连接会断 → 熔断与重连）；③ **能力协商**（server 声明支持什么 → 要做能力检查）；④ **信任边界**（**MCP server 是第三方** → 参数与结果的脱敏、scope 检查、以及「server 可能返回恶意内容」的防护，串 [[microsoft-08]] 的提示注入）。**读法**：**OpenAPI 是「静态契约」，MCP 是「活的能力」**——**后者要求工具层是「运行时可变」的**。
- **追问**：怎么测试这个工具层？
  - 要点：**四层测试**：① **Schema 转换的单测**（用真实 spec 覆盖 `$ref`/`nullable`/三处参数）；② **契约测试**（用 mock server 验证调用参数被回填到正确位置——path/query/body）；③ **故障注入**（超时、5xx、429、部分成功）**验证重试与「结果未知」路径**；④ **端到端**（agent 循环里跑真实工具，**尤其验证「工具消失/名字写错」的处理**）。**关键**：**故障注入必须覆盖「POST 超时」这一场景**——**它是唯一会因重试造成真实损失的路径**。

## 相关题目

- [[microsoft-01]]：top-k 高频查询——同一家公司的编码题（近似结构与工程陷阱）。
- [[microsoft-06]]：电子表格 agent——工具层之上的一层（编排与失败处理）。
- [[agents-01]]：工具选择与描述设计——模型侧如何看待工具集。
- [[agents-03]]：agent 的可靠性与错误恢复——「结果未知」这一状态的展开。
- [[coding-07]]：接口与协议设计——同一类抽象设计题。

## 参考资料与归属

- **OpenAPI Specification 3.0（延伸）** —— OpenAPI Initiative，2021-02-15：<https://spec.openapis.org/oas/v3.0.3>。第 4 节的 **path/query/body 参数位置、`$ref`、`nullable`** 等构造来自该规范；**本机的转换器只实现了其中一小部分**（未处理 `oneOf`/`allOf`、header/cookie 参数、多 content-type）。
- **Model Context Protocol Specification（延伸）** —— Anthropic / MCP contributors，2024-11-25：<https://modelcontextprotocol.io/specification>。第 3、10 节的 **`tools/list` 动态发现、JSON-RPC 调用、能力协商**来自该规范。
- **Toolformer: Language Models Can Teach Themselves to Use Tools（延伸）** —— Schick et al. (Meta)，2023-02-09：<https://arxiv.org/abs/2302.04761>。第 1 节的**「工具调用由模型发起、结果回填上下文」**这一基本形态来自这篇。
- **大规模日志的 top-k 高频查询（本仓库公司题库 · Microsoft 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。该篇的「实现细节决定成败」（哈希函数的系统性碰撞）与本篇的「错误分类与幂等」是同一类工程判断。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（2 个 operation、10000 个候选名、$N$ 取 20–200、结果大小 1 KB–10 MB、token 按 4 字节/token 估、128K 上下文、超时 5–30 s、重试 0–2 次、3 s 端到端预算）都是为演示设计驱动量而构造的**示例参数与显式假设**；**冲突概率、token 估算、超时上界都是直接计算**（可复现），而**具体数值依赖真实工具集与业务**。**转换器是「最小可用版本」**——真实 OpenAPI 转换需要处理 `oneOf`/`allOf`、参数序列化风格、多 content-type、安全方案（security schemes）等，**本机未覆盖**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
