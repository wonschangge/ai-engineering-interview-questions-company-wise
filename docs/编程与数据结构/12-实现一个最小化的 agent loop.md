---
type: question
id: coding-12
topic: 编程与数据结构
order: 12
question: 实现一个最小化的 agent loop，包含工具分发、错误处理和步数预算。
question_en: Implement a minimal agent loop with tool dispatch, error handling and a step budget.
asked_at: [Cognition（Devin、Windsurf）]
level: 高阶
tags: [agent-loop, 实现题, 工具分发, 预算]
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
related: [agents-01, agents-09, agents-02, coding-08, agents-10]
updated: 2026-09-28
---

## 一句话答案

> agent loop 是「调模型 → 有工具调用则分发执行 → 观察回灌 → 重复；模型给出最终答案则结束」的运行时，最小可用版本 165 行，由工具注册表、分发器、预算与轨迹记录四部分组成。决定质量的是模型出错时循环怎么处置：参数错误回灌可操作信息、瞬时错误按 full jitter 有限重试、幻觉工具名回报可用清单、结果过大截断并给分页游标、重复动作与无进展提前退出。
> 所有失败都必须是可恢复的观察：预算耗尽返回「部分结果 + 原因」，绝不抛异常或无限重试。

## 面试官在考什么

- 会不会**先定契约再写循环**：`run(task) -> {status, reason, steps, usage, done, answer, trajectory}` 一次说清，失败状态是一等公民而不是异常。
- **分发器是不是安全边界**：白名单查表而不是 `getattr`/`eval`、schema 校验、结果截断、写操作门禁、逐次超时与剩余预算下传。
- 错误是否**按类型分支**：可重试的才重试，参数错误回灌让模型自改，语义失败当事实，工具不存在报可用清单；预算与终止是否**可推导**——步数上限从 token 预算反推，四类预算各自在哪一层检查，超了返回什么。
- **可测试与可观测**：假模型加假工具就能离线跑满全部路径，轨迹足以回放与评测；这一点决定了整套实现能不能在 CI 里回归。

常见错误答案：

- 只写 `while True:` 调模型、最多 10 步：没有校验、超时、截断与失败出口，工具返一次 200 KB 就把上下文吃光。
- 用 `getattr(module, call.name)` 或 `eval` 构造调用，并把工具异常直接抛出终止循环：前者是工具名注入的入口，后者让一次网络抖动毁掉整条任务。

## 原理与推导

循环体只有四步：取模型输出 → 有工具调用则分发 → 把观察回灌进消息历史 → 判断能否终止。模型是无状态的，每轮都要重发完整历史；调用工具的是循环而不是模型，模型只决定调谁、传什么。这个「推理与行动交错」的结构来自 ReAct（ICLR 2023）的口径：推理轨迹负责归纳、跟踪与更新行动计划并处理异常，行动负责从外部环境取回信息；在 ALFWorld 与 WebShop 上相对模仿学习与强化学习基线分别高出 34 与 10 个百分点的绝对成功率，且只用了一两个 in-context 示例。
Anthropic 的取向是取舍原则：先找最简单可控的方案，只在确有收益时增加复杂度；agent 就是「LLM 基于环境反馈在循环中使用工具」，实现通常很直接，真正要投入的是工具集与文档（ACI），并建议保持简洁、把规划步骤显式展示、像对待 HCI 一样投资 ACI。

| 组件 | 职责 | 关键设计点 |
| --- | --- | --- |
| 注册表 | 名字 → schema + handler + 副作用声明 | 只登记显式注册的名字；工具定义每步重发，是固定的输入 token 开销 |
| 分发器 | 查找 → 校验 → 执行 → 序列化 | 每类失败都转成结构化观察；超时只放弃等待，不向外抛 |
| 循环控制与轨迹 | 调模型、回灌、终止判决，并逐步记录模型与 prompt 版本、参数、耗时、错误、截断标记 | 四类预算 + 完成信号 + 重复动作 + 无进展；轨迹 JSON 可序列化，能离线回放成同一串观察 |

分发器是最容易写出「能跑但危险」的地方，五点必须落实：① 白名单查找——工具名由模型生成，等价于用户输入，绝不能用它做属性访问、模块导入或代码求值（见 [[safety-06]]）；② schema 校验必填、类型、枚举与未知参数，失败时回灌「缺什么、期望什么、可用什么」，让模型自改而不是终止循环（见 [[agents-02]]；这类格式错误的正解是用结构化输出从源头消除，见 [[agents-03]]）；③ 结果先序列化再截断，显式带上 `truncated`、`total_chars`、`next_offset`，悄悄截断等于让模型基于残片推理；④ 副作用声明——只读工具直接执行，写操作要幂等或走审批（见 [[agents-11]]）；⑤ 每次调用取 `min(工具超时, 剩余墙钟)`，退避等待若会吃掉剩余预算就放弃重试，而不是把 deadline 拖爆。

| 错误类型 | 处置 | 回灌给模型的内容 |
| --- | --- | --- |
| 瞬时可重试（429、5xx、连接超时） | 有限次指数退避 + full jitter | 最终失败时带 `kind`、`attempts`、`retry_after_s` |
| 参数错误（缺必填、类型错、枚举越界） | 不重试，立即回灌 | 可操作的校验消息 + schema |
| 语义失败（空结果、无权限、无匹配） | 不重试，当作事实 | 空结果本身 + 换关键词或换工具的建议 |
| 模型侧错误（幻觉工具名、输出不可解析） | 不重试；解析问题用结构化输出从源头消除，只留一次兜底 | 可用工具清单 / 解析错误位置与期望格式 |

full jitter 的口径来自 AWS 架构博客的退避实践（见 [[coding-08]] 的引用）：等待时间取 $U(0, \min(d_{\max}, d_0 2^n))$，比固定退避更能打散同步重试造成的拥塞峰值；服务端给了 `Retry-After` 就听它的。终止有七层，本实现落地前六层：完成信号、步数上限、单步工具超时、预算上限（墙钟、token、费用）、重复动作检测（动作指纹 `hash(name + canonical_json(args))`）、无进展检测，第七层是人工兜底（见 [[agents-11]]）。
步数上限要从预算反推：第 $t$ 步输入 $I_t = S_0 + (t-1)\bar{s}$，累计输入 $T S_0 + T(T-1)\bar{s}/2 = O(T^2)$，代入单价与总预算 $B$ 解出的 $T$ 才是上限，耗尽时按「部分结果 + 原因」返回（见 [[agents-09]]）。无进展优先用结构化 diff（出现新事实或新状态变更才算进展），非结构化结果再用 3-gram Jaccard 兜底。

## 数值与代码验证

参考实现（仅标准库，完整可跑；下面所有数字都来自这份代码，验证脚本追加在同一文件底部）：

```python
"""最小化 agent loop：工具注册表 / 分发器 / 预算 / 轨迹记录（仅标准库，可直接运行）。"""
import hashlib, json, random, re, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

READ, WRITE, RETRYABLE = "read", "write", {"transient"}        # 只重试瞬时可恢复的错误
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
         "array": list, "object": dict}

@dataclass
class Tool:                      # 注册表条目：名字 + schema + handler + 副作用声明
    name: str
    schema: dict
    handler: object
    side_effect: str = READ
    requires_approval: bool = False

class Registry:                  # 白名单：只查表，绝不用 getattr / eval 构造调用
    def __init__(self): self._t = {}
    def register(self, tool): self._t[tool.name] = tool
    def get(self, name): return self._t.get(name)
    def names(self): return sorted(self._t)
    def specs(self):             # 工具定义每步重发，是输入 token 的固定开销
        return [{"name": t.name, "input_schema": t.schema} for t in self._t.values()]

def validate(schema, args):      # JSON Schema 子集：返回可操作的错误清单，空列表表示通过
    if not isinstance(args, dict):
        return [f"arguments 必须是 object，实际是 {type(args).__name__}"]
    props, errs = schema.get("properties", {}), []
    for key in schema.get("required", []):
        if key not in args: errs.append(f"缺少必填参数 {key!r}")
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            errs.append(f"未知参数 {key!r}，可用参数：{sorted(props)}")
        elif not isinstance(value, TYPES.get(spec.get("type"), object)) or (
                isinstance(value, bool) and spec.get("type") in ("integer", "number")):
            errs.append(f"参数 {key!r} 期望 {spec.get('type')}，实际是 {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            errs.append(f"参数 {key!r} 必须是 {spec['enum']} 之一")
    return errs

def serialize(value, cap):       # 截断必须显式告知模型，否则它以为看到的就是全部
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= cap: return {"content": text, "total_chars": len(text), "truncated": False}
    head, tail = text[:cap // 2], text[-(cap - cap // 2):]
    return {"content": head + "\n…[truncated]…\n" + tail, "total_chars": len(text),
            "truncated": True, "next_offset": len(head),
            "hint": "结果已截断；带上 offset=next_offset 重新调用可分页读取"}

def action_key(name, args):      # 重复检测与幂等键共用同一个规范化指纹
    return hashlib.sha256((name + json.dumps(args, sort_keys=True, ensure_ascii=False,
                                             default=str)).encode()).hexdigest()[:16]

def similarity(a, b):            # 无进展的兜底判据：3-gram Jaccard，O(len)
    gram = lambda s: {s[i:i + 3] for i in range(max(1, len(s) - 2))}
    sa, sb = gram(re.sub(r"\s+", " ", a.lower())), gram(re.sub(r"\s+", " ", b.lower()))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0

@dataclass
class Budget:                    # 四类预算 + 门禁参数：随请求传入，不写死在代码里
    max_steps: int = 8; wall_clock_s: float = 30.0; max_input_tokens: int = 60_000
    max_usd: float = 0.5; tool_timeout_s: float = 2.0; max_result_chars: int = 4_000
    max_attempts: int = 3; backoff_cap_s: float = 1.0
    max_same_action: int = 2; max_no_progress: int = 2; similarity_threshold: float = 0.85
    approved: set = field(default_factory=set)
    def usd(self, tin, tout): return tin * 3e-6 + tout * 15e-6

class Dispatcher:
    def __init__(self, registry, budget, sleep=time.sleep, rng=None):
        self.reg, self.b, self.sleep = registry, budget, sleep
        self.rng, self.pool = rng or random.Random(0), ThreadPoolExecutor(max_workers=4)
    def _fail(self, kind, message, **extra):
        return {"ok": False, "error": {"kind": kind, "message": message,
                                       "retryable": kind in RETRYABLE, **extra}}
    def call(self, name, args, remaining_s=None):
        tool = self.reg.get(name)
        if tool is None:                                   # 幻觉工具名：告知清单后继续
            return self._fail("not_found", f"未知工具 {name!r}；可用工具：{self.reg.names()}")
        errs = validate(tool.schema, args)
        if errs:                                           # 参数错误：回灌信息，不抛异常终止
            return self._fail("invalid_args", "参数校验失败：" + "；".join(errs), schema=tool.schema)
        if tool.side_effect == WRITE and tool.requires_approval \
                and action_key(name, args) not in self.b.approved:
            return self._fail("denied", f"{name} 是写操作，需要人工审批",
                              approval_key=action_key(name, args))
        timeout = min(self.b.tool_timeout_s, remaining_s or self.b.tool_timeout_s)
        try:
            value = self.pool.submit(tool.handler, args).result(timeout=max(0.001, timeout))
        except FutureTimeout:                              # 线程杀不掉：只能放弃等待
            return self._fail("transient", f"{name} 超过 {timeout:.2f}s 未返回")
        except Exception as exc:                           # 工具内部异常一律转成观察
            return self._fail(getattr(exc, "kind", "internal"), f"{name} 失败：{exc}",
                              retry_after_s=getattr(exc, "retry_after", None))
        return {"ok": True, "tool": name, **serialize(value, self.b.max_result_chars)}
    def call_with_retry(self, name, args, remaining_s=None):
        waits, obs = [], None
        for attempt in range(self.b.max_attempts):
            obs = self.call(name, args, remaining_s)
            obs["attempts"] = attempt + 1
            if obs["ok"] or not obs["error"]["retryable"] or attempt == self.b.max_attempts - 1:
                return obs, waits
            cap = min(self.b.backoff_cap_s, 0.05 * 2 ** attempt)
            delay = obs["error"].get("retry_after_s") or self.rng.uniform(0, cap)   # full jitter
            if remaining_s is not None and delay > remaining_s:
                return obs, waits                          # 退避会吃掉剩余预算：放弃而非拖爆
            waits.append(round(delay, 4)); self.sleep(delay)
        return obs, waits

class AgentLoop:
    def __init__(self, model, registry, dispatcher, budget, clock=time.perf_counter,
                 tokens=lambda s: max(1, len(str(s)) // 4)):   # token 用 chars/4 代理，非真实分词
        self.m, self.reg, self.d, self.b = model, registry, dispatcher, budget
        self.clock, self.tokens = clock, tokens
    def _end(self, status, reason, steps, done, answer=None):
        return {"status": status, "reason": reason, "steps": steps, "usage": self.usage,
                "trajectory": self.traj, "done": done, "answer": answer if answer is not None else
                f"未完成（{reason}）；中间结果：{json.dumps(done, ensure_ascii=False)}"}
    def run(self, task):
        b, self.traj = self.b, []
        traj = self.traj
        messages, usage = [{"role": "user", "content": task}], {"in": 0, "out": 0, "usd": 0.0}
        self.usage, hits, last_obs, stuck, done = usage, {}, None, 0, []
        rec = lambda step, kind, **kw: traj.append({"step": step, "kind": kind, **kw})
        t0 = self.clock()
        for step in range(1, b.max_steps + 1):
            if self.clock() - t0 > b.wall_clock_s:                              # ① 墙钟预算
                return self._end("budget_exhausted", "wall_clock", step - 1, done)
            if usage["in"] >= b.max_input_tokens or usage["usd"] >= b.max_usd:  # ② 费用预算
                return self._end("budget_exhausted", "cost", step - 1, done)
            reply = self.m(messages, self.reg.specs())
            prompt = json.dumps(messages, ensure_ascii=False) + json.dumps(self.reg.specs(),
                                                                          ensure_ascii=False)
            tin = self.tokens(prompt)                                           # 每步重发全历史
            tout = self.tokens(reply.get("text") or json.dumps(reply.get("tool_calls") or ""))
            usage["in"] += tin; usage["out"] += tout; usage["usd"] += b.usd(tin, tout)
            rec(step, "model", text=reply.get("text", ""), tool_calls=reply.get("tool_calls", []),
                usage={"in": tin, "out": tout})
            if reply.get("final") is not None:                                  # ③ 完成信号
                rec(step, "final", answer=reply["final"])
                return self._end("ok", "model_final", step, done, reply["final"])
            blob = []
            for c in reply.get("tool_calls", []):
                obs, waits = self.d.call_with_retry(c["name"], c.get("args", {}),
                                                    b.wall_clock_s - (self.clock() - t0))
                key = action_key(c["name"], c.get("args", {}))
                hits[key] = hits.get(key, 0) + 1
                if hits[key] > 1:                                               # ④ 重复动作提示
                    obs["repeat_hint"] = f"该动作与参数已第 {hits[key]} 次执行，请换动作或给答案"
                rec(step, "tool", tool=c["name"], args=c.get("args", {}), action_key=key,
                    hits=hits[key], ok=obs["ok"], error=(obs.get("error") or {}).get("kind"),
                    attempts=obs["attempts"], truncated=obs.get("truncated", False), waits_s=waits)
                messages.append({"role": "tool", "name": c["name"],
                                 "content": json.dumps(obs, ensure_ascii=False, sort_keys=True)})
                blob.append(messages[-1]["content"])
                if obs["ok"]: done.append({"tool": c["name"], "args": c.get("args", {})})
                if hits[key] > b.max_same_action:                               # ⑤ 循环检测
                    return self._end("partial", "repeated_action", step, done)
            sim = similarity("\n".join(blob), last_obs) if last_obs is not None else 0.0
            stuck = stuck + 1 if sim >= b.similarity_threshold else 0
            rec(step, "progress", similarity=round(sim, 4), stuck=stuck)
            if stuck >= b.max_no_progress:                                      # ⑥ 无进展检测
                return self._end("partial", "no_progress", step, done)
            last_obs = "\n".join(blob)
        return self._end("budget_exhausted", "step_limit", b.max_steps, done)   # ⑦ 步数上限
```

验证用一个按脚本回答的假模型加五个假工具（正常查询、巨大结果、缓存命中、分页、瞬时失败）；脚本项还可以写成 `messages -> dict` 的函数，用来断言模型看到的观察长什么样。`python3 agent_loop.py` 的真实输出：

```text
① 正常路径                ok                model_final       steps=3 轨迹=8条 usd=0.00205
② 参数错误后自我修正           ok                model_final       steps=3 轨迹=8条 usd=0.00219
③ 重复动作提前退出            partial           repeated_action   steps=3 轨迹=8条 usd=0.01112
④ 步数预算耗尽              budget_exhausted  step_limit        steps=3 轨迹=9条 usd=0.00369
⑤ 超大结果被截断             ok                model_final       steps=2 轨迹=5条 usd=0.00156
⑥ 观察高度相似判无进展          partial           no_progress       steps=3 轨迹=9条 usd=0.00242
⑦ 瞬时错误重试              ok                model_final       steps=2 轨迹=5条 usd=0.00107
⑧ 轨迹可回放               ok                model_final       steps=2 轨迹=5条 usd=0.00114
⑦ 实测退避序列(s): [0.0162, 0.0151]
```

八个边界用例的断言（全部通过，整套约 0.05 s，CPython 3.10；下面写的就是脚本里的断言）：

①② 正常路径与自我修正：`(status, reason, steps) == ("ok", "model_final", 3)`，轨迹 8 条（model/tool/progress ×2 + model/final）；参数非法那次工具步的 `error` 序列为 `["invalid_args", None]`，错误消息含 `缺少必填参数 'date'`，模型第 2 步改用合法参数成功。
③④ 重复动作与预算耗尽：`hits == [1, 2, 3]` 时 `reason == "repeated_action"`；`(status, reason, len(done)) == ("budget_exhausted", "step_limit", 3)` 时返回部分结果与原因。
⑤⑥⑦⑧ 截断、无进展、重试与回放：模型侧断言 `truncated and total_chars == 200000 and "offset" in hint` 成立（500 字符上限下 content 只有 515 字符）；相似度序列 `[0.0, 1.0, 1.0]` 之后 `reason == "no_progress"`；`attempts == 3` 且两次等待 0.0162 s、0.0151 s 都落在 $[0, \text{cap}]$ 内；轨迹序列化再反序列化后重新分发，`action_key` 一致且结果 `ok`。

每步重发完整历史的代价（观察 800 字符、1 个工具定义，token 用 chars/4 代理，只用于看增长阶）：

| 步数 $T$ | 首步输入 | 末步输入 | 累计输入 | 累计 $/T^2$ |
| --- | --- | --- | --- | --- |
| 10 | 41 | 2,212 | 11,263 | 112.6 |
| 20 | 41 | 4,624 | 46,650 | 116.6 |
| 40 | 41 | 9,449 | 189,800 | 118.6 |

累计 $/T^2$ 收敛到 $\bar{s}/2 \approx 120$（$T = 40$ 时 118.6；$S_0 = 41$、$\bar{s} \approx 241.2$，由末步输入 9,449 反推），与 $T S_0 + T(T-1)\bar{s}/2$ 的理论值 189,776 只差 24 个 token：平方项不是理论担忧，40 步就能实测到。长轨迹的硬件代价同样真实：LLaMA-3-70B 每 token KV cache 320 KiB，10 万 token 的上下文就是约 30.5 GiB KV，按 H100 SXM5 的 3.35 TB/s HBM 带宽读一遍要 9.8 ms，**每个 decode step 都要付这笔钱**；prefill 的算术强度约 $8.1\times10^{4}$ FLOPs/byte（10 万 token 的 14 PFLOPs ÷ 权重 140 GB 与 KV 写入 30.5 GiB 合计约 173 GB 的流量），远高于 295 FLOPs/byte 的 roofline 拐点（989 TFLOPs dense ÷ 3.35 TB/s），单卡峰值下界 14.2 s——这正是 prefix caching 与轨迹压缩存在的理由（见 [[agents-07]]）。

以下耗时都在本机测得（AMD Ryzen 9 8945HX、CPython 3.10，多次运行取最小值；绝对耗时随负载浮动，结论看比值）。分发器单次调用：白名单查表 0.09 µs，`validate` 0.62 µs，handler 直调 0.17 µs，`Dispatcher.call` 全程 14.4 µs，参数错误路径 1.0 µs——**线程池的 submit/result 比参数校验贵 20 倍以上**，快速工具走线程池是纯损耗，生产实现应让 I/O 走事件循环或异步，线程池只留给真正的阻塞调用。截断信封固定 170–172 字符：上限取 500 / 2,000 / 8,000 / 32,000 字符时，content 长 515 / 2,015 / 8,015 / 32,015，整体折合 171 / 546 / 2,046 / 8,046 token（chars/4）。退避抖动（cap = 1.0 s，1,000 次采样）：无抖动 mean 1.0000 s、std 0.0000 s；full jitter mean 0.4817 s、std 0.2948 s（理论值 0.5000 / 0.2887）——std 从 0 到 0.29 就是同步重试被抹平的幅度。无进展判据的实测：3-gram Jaccard 在「完全相同」「只差时间戳」「字段顺序不同」上给出 1.0000 / 0.6333 / 1.0000，在「多一个真事实」上给出 **0.8406**，离 0.85 的阈值只差 0.01——字符级相似度不区分「新事实」与「换个说法」。同一批样本改用键路径级 diff（忽略 `fetched_at` 这类元数据）后，「只差时间戳」是 0 个新增内容叶子，「多一个真事实」是 2 个（`flights[2].id`、`flights[2].price`），判定立刻干净。编排自身的开销：39 步循环 11.4 ms（291 µs/步，含假模型与记录），观察为空时同样 39 步只要 4.2 ms（107 µs/步）——差出来的 184 µs 几乎全是 `similarity()`。这个函数的常数很大且随长度线性增长：970 字符 176 µs、1,000 字符 229 µs、10,000 字符 4.2 ms、100,000 字符 25.3 ms，也就是说**一次 100 KB 的观察会让每步多付 25 ms**。无进展检测因此必须限制参与比较的文本长度（取字段摘要或指纹），或改用结构化 diff 与增量摘要。

生产化的差距（把玩具升级成系统）：① 并发——一轮内无依赖的只读调用应并行，但要限并发、保持回灌顺序与调用顺序一致、做错误隔离（见 [[coding-08]]）；② 人机交互——写操作前暂停等审批，并把审批 id 记进轨迹（见 [[agents-11]]）；③ 可观测与审计——模型版本、prompt 版本、原始工具参数、成本、截断与重试都要按步记录，否则事后无法复现一次失败（见 [[evaluation-07]]、[[agents-10]]）；④ 评测——用任务成功率与 pass^k 而不是平均值做上线门槛（见 [[evaluation-10]]）；⑤ 上下文管理——轨迹变长要压缩与摘要，且压缩时保护目标与约束（见 [[agents-07]]）；⑥ 现实预期——τ-bench 口径下当时最好的 function calling agent 在真实领域任务上按域加权平均成功率也只有 48.2%（τ-retail 61.2%、τ-airline 35.2%），τ-retail 的 pass^8 低到 25% 以下，所以循环必须按「一定会失败」来设计（见 [[agents-09]]）。另外一条最容易被忽略：本实现用线程池做超时，而 Python 线程杀不掉，超时后工具还在跑，真正的隔离要靠子进程或沙箱；线程池只有 4 个 worker 还有第二个后果——4 个卡住的调用就会让后续调用排队，实测第 5 个本该瞬时返回的调用被记成「超过 0.40 s 未返回」，这个超时是假的，回灌给模型的假观察会把模型引向无意义的重试（超时判定必须绑定「真的开始执行」，或者每次调用独立执行单元）。

## 常见追问

- **追问**：为什么用最大步数，而不是一直跑到模型说完成？
  - 要点：模型自报完成不可单独采信（过早停止与自信的错误结论都很常见），而循环每步的边际成本随历史线性上涨、累计随步数平方上涨。正确做法是「完成信号 + 外部验证器」判正确性，用从预算反推的步数上限判上限，超限时返回部分结果与原因（见 [[agents-09]]）。
- **追问**：无进展检测怎么定义，阈值怎么定？
  - 要点：优先做结构化 diff（新事实、新状态变更才算进展，时间戳一类字段进忽略列表），非结构化结果再用 3-gram Jaccard 兜底，并且要求「连续 N 步」而不是单步相似。实测 0.85 阈值对「多一个真事实」只留 0.01 余量，所以阈值必须用线上轨迹回放标定，不能照抄。
- **追问**：流式输出下怎么让循环可取消？
  - 要点：把可安全中断点定义在工具调用之间，流式层只缓冲可丢弃的草稿；取消信号要向下传到 LLM 请求、工具调用与退避睡眠，已产生的副作用走补偿（见 [[coding-09]]、[[agents-10]]）。

## 公司变体

Cognition（Devin、Windsurf）：题面来自做长时程 coding agent 的公司，其公开工程材料的取向是**上下文工程优先于多 agent 编排**，因此更看重实现在压力下的行为——步数与 token 预算、观察裁剪与压缩、失败出口，以及靠测试作为外部验证器的终止判据。这类题通常要求现场写代码而不是讲概念，评审会盯三处：工具名是否白名单查表（不是 `getattr`）、错误是否作为观察回灌（不是抛异常）、超预算时返回什么。追问容易落到「长任务怎么不跑偏」，答轨迹压缩 + 目标复述 + 定期用测试验证，比答「换个更强的模型」得分高。

## 相关题目

- [[agents-01]]：ReAct 模式与 chain-of-thought 的区别；[[agents-02]]：工具调用错误、超时与重试（五类错误的完整版）；[[agents-09]]：循环终止与成本步数限制；[[agents-10]]：操作可逆与可审计；[[agents-11]]：human-in-the-loop 审批。
- [[agents-03]]：结构化输出与 function calling；[[agents-07]]：长任务记忆与压缩；[[agents-12]]：漂移诊断。
- [[coding-08]]：异步批处理器——并发上限、错误隔离与 full jitter 的完整实现；[[coding-09]]：流式 SSE 解析器——取消与断流语义；[[safety-06]]：工具权限与数据外泄；[[evaluation-07]]、[[evaluation-10]]：可观测性与 agent 评测。

## 参考资料与归属

- *AI Agent 循环*（原文 *What is an AI Agent Loop?*）— Amit Shekhar（Outcome School），页面标注 2026-05-28：<https://outcomeschool.com/blog/ai-agent-loop>。提供 think–act–observe 循环结构、约 20 行的循环骨架、并行工具调用、两条停机条件（模型自报完成与步数上限）与四类循环失败（死循环、重复同一动作、上下文溢出、过早停止）；本文的四组件划分、分发器要点、错误分类、预算推导、全部代码与实测数字均为按该口径自行实现与测量，源文没有给出任何量级数字。
- *Building Effective Agents* — Anthropic，2024-12-19：<https://www.anthropic.com/engineering/building-effective-agents>（延伸来源：workflow 与 agent 的区分、优先最简单方案、简洁与透明两条原则、ACI 与工具文档的投入）。
- *ReAct: Synergizing Reasoning and Acting in Language Models* — Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao（ICLR 2023），arXiv 提交 2022-10-06：<https://arxiv.org/abs/2210.03629>（延伸来源：推理轨迹与行动的交替生成、观察回灌修正推理，以及 ALFWorld 与 WebShop 上 34% 与 10% 的绝对成功率提升、仅用一两个 in-context 示例）。
- τ-bench 的 61.2% / 35.2% / 按域加权 48.2% / pass^8 <25% 与 AWS 架构博客的 full jitter 口径，经站内 [[agents-09]]、[[coding-08]] 引用，此处不另附链接。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
