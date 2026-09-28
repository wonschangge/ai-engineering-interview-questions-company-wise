---
type: question
id: agents-09
topic: Agent 与工具调用
order: 9
question: 什么能让 agent 循环正确终止？你如何限制成本和步数？
question_en: What makes an agent loop terminate correctly, and how do you bound cost and steps?
asked_at: []
level: 高阶
tags: [终止条件, 预算, pass-k, 可靠性]
sources:
  - title: AI Agent Loop
    url: https://outcomeschool.com/blog/ai-agent-loop
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Fixing Infinite Loops in AI Agents
    url: https://www.linkedin.com/posts/pallavi-shekhar_ai-aiagents-machinelearning-share-7440257380707364864-5Ycc
    author: Pallavi Shekhar
    published: ""
  - title: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains（延伸）
    url: https://arxiv.org/abs/2406.12045
    author: Yao et al.
    published: 2024-06-17
related: [agents-02, agents-12, agents-06, inference-serving-09]
updated: 2026-09-28
---

## 一句话答案

> 正确终止 = 模型给出显式完成信号，**且**外部验证器确认目标状态达成；同时每一层预算（步数、token、墙钟、费用）都有硬上限，耗尽时按「部分完成 + 原因」的契约正常返回，而不是继续重试。
> 步数上限不能拍一个 10，要从累计 token 预算反推：第 $t$ 步输入约 $S_0+(t-1)\bar{s}$，累计输入 token 随步数平方增长，所以「能跑多少步」是预算推导的结果。
> 可用性门槛看 pass^k（同一任务重复 $k$ 次全部成功）而不是平均成功率——平均成功率会把「偶尔能成功」掩盖成「基本可用」。

## 面试官在考什么

- 会不会**先分类再开药**：不终止至少有六种成因（任务不可达、判据不满足、工具持续失败、多 agent 互相等待、上下文污染、判据本身有歧义），它们的修法完全不同，只会答「加 max_steps」说明只处理了症状。
- 是否理解**模型自报完成不可单独采信**：终止信号必须来自模型之外的验证器（测试、schema 校验、数据库终态比对），这是 τ-bench 这类评测判定成功的依据，也是生产系统与 demo 的分界线。
- 预算是不是**推导出来的**：能不能把 token 预算换算成步数上限、把墙钟 deadline 向下传递、把「步数上限 / p99 步数」的余量说清楚，而不是背一个经验数字。
- 有没有**失败出口**：任务不可达时返回什么？预算耗尽时返回什么？答案里必须出现「部分结果 + 原因 + 可续跑」，否则系统在压力下只会无限重试。
- 用**一致性指标**还是平均值做上线决策，是否知道 pass^k 与 pass@k 的区别，以及为什么「偶尔成功」在生产里等于不可用。

常见错误答案：

- 只答「加个 `max_steps=10`，超了就停」。上限本身不解决正确性，反而会把「任务需要 25 步」误判成失败；截断时的返回约定才是关键。
- 认为换个更强的模型、把上下文窗口开大就能消掉循环。步数上限之外的问题（判据歧义、工具错误信息不可用、多 agent 协议缺默认分支）与模型能力无关。

## 原理与推导

### 1. 先分类：循环为什么不终止

| 失效模式 | 现场症状 | 对症手段 |
| --- | --- | --- |
| 任务不可达但模型不承认 | 工具/权限根本不具备，模型换着写法反复试 | 给出**失败出口**：允许最终答案声明「无法完成 + 缺什么」，并在系统提示里写明能力边界外必须报失败 |
| 观察始终不满足模型的内部判据 | 反复改写查询、阈值永远差一点 | 把软判据换成**外部可验证谓词**（测试通过、schema 合法、目标状态 diff 为空） |
| 工具持续返回空或错误 | 同一调用重复 N 次，模型不知道错在哪 | 结构化错误 + 重试上限与退避，见 [[agents-02]] |
| 两个 agent 互相等待对方先动 | 双方都等对方先产出一个块，都不动 | 每个委派都要有超时与「无输入也推进」的默认分支，见 [[agents-06]] |
| 上下文被无关信息污染后忘了目标 | 转向无关子任务，逐步背离原始约束 | 目标与约束常驻（置顶摘要），压缩时保护目标，见 [[agents-12]]、[[agents-07]] |
| 判据本身有歧义 | 「把数据整理好」永远无法判定完成 | 落成验收清单/测试/终态断言；这一条只能在任务定义阶段解决，运行时兜不住 |

诊断顺序建议从第 6 条往前查：判据歧义和缺失败出口是设计缺陷，改 prompt 或加步数都无效；工具与协议问题看日志就能定位。

### 2. 终止条件的七层设计

1. **显式完成信号 + 外部验证器（最可靠的一层）**。模型输出 final answer 后不直接返回，先跑验证器；τ-bench 的做法就是对话结束后比对「数据库终态 vs 标注目标状态」，奖励 $r = r_{\text{action}} \times r_{\text{output}} \in \{0,1\}$：$r_{\text{action}}$ 判终态数据库是否等于唯一的目标状态，$r_{\text{output}}$ 判回答里是否给全了必要信息——模型自述「完成」本身不构成奖励。工程上等价于：单元测试、HTTP 断言、DB diff、schema 校验。验证失败就把 diff 当观察回灌，允许有限次修复——**自己给自己发通行证**是这一层最常见的漏洞。
2. **步数上限**。按 token 预算反推（见第 4 节），并用线上步数分布的 p99 留 1.5–2 倍余量。业界常见的 `max_steps=10` 是产品侧经验默认值，不是推导值。
3. **单步超时与整体墙钟超时**。工具调用、模型调用、整个请求各有一层 deadline，并把剩余时间向下传递（deadline propagation）——子步骤不能再申请超过父级剩余时间的时间片，见 [[agents-02]]。
4. **token/费用预算**。累计计费上限，用真实计费口径（含重试、含被截断的输出），超过即停。
5. **无进展检测**。连续 N 步观察的**信息增量**低于阈值就提前退出。判据要看「是否出现新事实/新实体/新状态变更」，不能只看长度：模型复述同一段文字会让长度上涨但信息量为零。
6. **循环检测**。对动作做规范化哈希，`hash(tool_name + canonical_json(args))`；同一工具同一参数重复出现超过 2 次时中断并注入提示「已执行过该动作且结果如下，请换动作或给出最终答案」。注意保留合法轮询：等待异步任务完成时动作会重复，但**观察在变**，因此判据要加上「且观察也未变」。
7. **人工兜底**。高风险操作、低置信度、预算接近耗尽时升级给人，见 [[agents-11]]。

第 1 层决定「完成得对不对」，第 2–6 层决定「跑不跑得完、花多少钱」，第 7 层决定「出事谁来兜」。只做其中一层都不成立：只有第 1 层，任务不可达时会无限重试；只有第 2 层，会在半成品处静默截断。

### 3. 失败出口：把「没做完」当成一等结果

终止条件必须和返回契约一起设计。统一的 result envelope：

```text
status: ok | partial | needs_human | unreachable | budget_exhausted
reason: 一句话原因（哪一层预算耗尽、验证器报了哪条差异）
steps / tokens / usd: 实际消耗
done: 已完成产物的引用（文件、记录 id、中间结论）
next_cursor: 可续跑的游标，供人工或下一次调用接着做
```

三个硬性约定：

- **重试只针对可重试错误**（超时、429、5xx）；参数错误与权限错误重试没有意义。验证失败也不能原样重试——必须把 diff 回灌并限定修复次数（第 1 层），否则只是在烧预算。
- **不静默截断**。`max_steps` 触发时用户必须看到「这是部分结果」，否则下游会把半成品当成品用。
- **可取消、可观测**。暴露进度（当前步数/预算消耗）与取消入口，长任务尤其需要；取消后已产生的副作用要按 [[agents-10]] 的可逆性设计处理。

### 4. 预算从哪来：累计成本是平方的

循环每轮把完整历史重发一次，所以输入长度随步数线性增长，累计计费 token 随步数**平方**增长。设 $S_0$ 为每步固定开销（系统提示 + 工具定义），$\bar{s}$ 为每步新增的观察与输出：

$$I_t = S_0 + (t-1)\bar{s}, \qquad \sum_{t=1}^{T} I_t = T S_0 + \frac{T(T-1)}{2}\bar{s} = O(T^2)$$

给定单任务预算 $B$（输入价 $p_{in}$、输出价 $p_{out}$、每步输出 $\bar{o}$）：

$$\left(T S_0 + \frac{T(T-1)}{2}\bar{s}\right) p_{in} + T \bar{o}\, p_{out} \le B$$

解出 $T$ 就是**从预算反推的步数上限**。两个直接推论：第 $T$ 步的边际成本是 $I_T p_{in} + \bar{o} p_{out}$，随 $T$ 线性上涨，所以「最后几步最贵」；prefix caching 把重复前缀按折扣计费，是把平方压回近乎线性最直接的手段（见 [[inference-serving-05]]），代价是要求前缀逐字节稳定——把时间戳、随机 id 放进系统提示会直接打掉缓存命中；把上下文本身压到有界（第 5 节的压缩）也能达到同样的量级效果。工具结果缓存则作用于 $\bar{s}$，减少每步新增。

### 5. 成本控制的杠杆（按性价比排序）

| 杠杆 | 作用点 | 代价与前提 |
| --- | --- | --- |
| 模型分层：规划/验证用强模型，执行与抽取用便宜模型 | 单 token 单价 | 需要稳定的任务分解；路由错了会返工 |
| 上下文压缩：只保留关键观察与结论 | $I_t$ 的增长率 | 压缩有丢信息风险，目标与约束必须保护，见 [[agents-07]] |
| 工具结果截断与分页 | $\bar{s}$ | 截断会移除模型定位错误所需的信息，要保留字段名与错误码 |
| 并行执行独立步骤 | 墙钟（不省 token） | 只对无依赖步骤有效；并发写同一资源要加锁 |
| 缓存：工具结果缓存 + prefix cache | $\bar{s}$ 与重复前缀 | 需要注意 TTL、幂等性与缓存键，见 [[inference-serving-05]] |
| 预算事前分配：每个子任务一个小上限，父任务留储备 | 尾部风险 | 需要父子预算记账 |
| 减少不必要的工具调用 | 步数 | 依赖「该不该调工具」的判断质量，见 [[agents-08]] |
| 输出约束：max_tokens、stop 序列、结构化输出 | $\bar{o}$ 与重试率 | 过紧会截断结构化输出导致解析失败，反而更贵 |

排序依据是「省下的比例 ÷ 实施风险」。模型分层的收益最大也最需要评测兜底；截断与分页是纯工程改动，通常最先做。

### 6. 一致性口径：pass^k，以及为什么平均值不够

τ-bench 提出的 pass^k 定义在 $k$ 次 i.i.d. 试验上：**全部 $k$ 次都成功**的概率，按任务平均。若某任务跑 $n$ 次有 $c$ 次成功，无偏估计量是

$$\hat{\text{pass}}^k = \mathbb{E}_{\text{task}}\left[\binom{c}{k} \middle/ \binom{n}{k}\right], \qquad \hat{\text{pass}}@k = 1 - \mathbb{E}_{\text{task}}\left[\binom{n-c}{k} \middle/ \binom{n}{k}\right]$$

两者方向相反：pass@k 是「$k$ 次里至少一次成功」（探索能力，随 $k$ 上升），pass^k 是「$k$ 次全部成功」（一致性，随 $k$ 下降）。同一份试验数据在两个指标下会给出相反结论，复算见「数值与代码验证」一节。

论文给出的关键事实（Table 2 与第 5.1 节）：gpt-4o 这类 function calling agent 在 τ-retail 上 $\text{pass}^1 = 61.2\%$、τ-airline 上 $35.2\%$，表里 avg 列是 $48.2\%$，该列注释明确「按域加权、不按任务加权」；τ-retail 的 $\text{pass}^8$ 掉到 $<25\%$（正文原话是 as low as ~25%）。**引用口径要注意**：摘要里「succeed on <50% of the tasks」指的就是这个两域平均，并不是单指零售——把它转述成「零售域成功率低于 50%」与正文的 61.2% 冲突。

一个容易被忽略的推论：如果按独立同分布把 $\text{pass}^8$ 当成 $(\text{pass}^1)^8$，$0.61^8 \approx 1.9\%$，远低于论文报的 $<25\%$。差距来自任务维度的异质性——难任务恒难、易任务恒易，同一任务内的多次试验在整体上正相关；按 Jensen 不等式 $\mathbb{E}[p^k] \ge (\mathbb{E}[p])^k$，异质性只会把 $\text{pass}^k$ 抬到 $(\text{pass}^1)^k$ 之上，不会压低它。所以 pass^k 不能被简化成 $(\text{pass}^1)^k$，但它随 $k$ 单调下降的性质不变：把 $k$ 写进上线门槛（例如「连续 5 次一致」），就能把「偶尔成功」挡在门外。

同类结论也来自 AgentBench 的口径：在 8 个交互环境中评测，长期推理、决策与指令遵循是主要障碍——说明瓶颈在长程执行稳定性，而不是单点知识。

### 7. 工程落地：配置、可观测、回归

- **预算与终止条件是随请求传递的配置**，不是全局常量：每个租户、每个接口、每种任务风险等级一套，随请求进入运行时并记录到 trace。
- **监控三类分布**：步数分布、终止原因分布、单任务成本分布。终止原因分布里 `budget_exhausted` 与 `no_progress` 的占比上升，通常意味着任务难度或判据定义变了，而不是模型变笨了。
- **回归测试要固定任务集与预算**，同时报告 pass^1 与 pass^k；只报平均成功率会让「一致性变差、平均值不变」的改动蒙混过关。

## 数值与代码验证

### 每步正确率的复利（口径：各步独立同分布）

| 每步正确率 $p$ | 5 步 | 10 步 | 20 步 | 50 步 | 平均首次失败步数 $1/(1-p)$ |
| --- | --- | --- | --- | --- | --- |
| 0.90 | 0.590 | 0.349 | 0.122 | 0.005 | 10 |
| 0.95 | 0.774 | 0.599 | **0.358** | 0.077 | 20 |
| 0.98 | 0.904 | 0.817 | 0.668 | 0.364 | 50 |
| 0.99 | 0.951 | 0.904 | 0.818 | 0.605 | 100 |

$0.95^{20} = 0.3585$：每步 95% 正确、20 步走完，整体只有 35.9% 的概率全程无错。独立同分布是这里最乐观的假设，真实系统里错误往往聚簇（一次环境抖动连带多步失败），所以该表用来解释「为什么必须缩短步数或加中间验证」，不能直接当线上预测。

### 步数上限从预算反推（口径：示意参数，用于演示算法）

$S_0 = 7{,}500$ token（系统提示与政策 3,000 + 30 个工具定义 × 150）；每步新增 $\bar{s} = 1{,}100$（观察 800 + 输出 300）；示意价 3 美元/百万输入 token、15 美元/百万输出 token——**这两个单价只是算例，不代表任何厂商当前定价**。

| 步数 $T$ | 累计输入 token | 第 $T$ 步的边际成本 | 任务成本 | 命中 prefix cache 后（重复前缀按 10% 计费） |
| --- | --- | --- | --- | --- |
| 5 | 48,500 | 0.040 美元 | 0.168 美元 | 0.069 美元 |
| 10 | 124,500 | 0.057 美元 | 0.418 美元 | 0.129 美元 |
| 20 | 359,000 | 0.090 美元 | 1.167 美元 | 0.274 美元 |
| 30 | 703,500 | 0.123 美元 | 2.245 美元 | 0.452 美元 |

第 20 步的输入 28,400 token，是第 1 步 7,500 token 的 3.8 倍——这就是「最后几步最贵」的量化版本，也是不能把步数上限与预算分开定的原因。同一个 0.50 美元预算：无缓存只能跑 **11 步**（0.478 美元），命中重复前缀能跑 **32 步**（0.492 美元）。缓存折扣按 10% 是演示假设，不同实现差异很大，但量级结论稳定：**步数上限与缓存策略必须一起定**。

### 工具定义的固定成本

| 工具数 | 每步定义开销（按 150 token/工具估） | 20 步定义累计输入 | 该工具集的 20 步累计输入 | 定义占比 | 折合成本 |
| --- | --- | --- | --- | --- | --- |
| 5 | 750 | 15,000 | 284,000 | 5.3% | 0.045 美元 |
| 10 | 1,500 | 30,000 | 299,000 | 10.0% | 0.090 美元 |
| 30 | 4,500 | 90,000 | 359,000 | 25.1% | 0.270 美元 |
| 100 | 15,000 | 300,000 | 569,000 | 52.7% | 0.900 美元 |

每个工具定义按 150 token 是量级假设（name + description + 2–3 个参数的 JSON schema，实际长度取决于 description 措辞，应以 tokenizer 实测为准）。表里的累计输入按各自的 $S_0 = 3{,}000 + 150n$ 重算，不是共用同一个分母：工具数本身会抬高每步固定开销，所以 100 个工具时定义占了累计输入的一半以上，而不是简单按比例外推。30 个工具时，定义在第 20 步的 28,400 token 输入里占 15.8%，累计贡献 25% 的输入 token 与 23% 的任务成本——工具定义是**每步重发的固定开销**，随步数线性累加，所以「工具太多」既是能力问题（模型选错工具）也是预算问题。

### 代码：带预算与循环检测的循环骨架

```python
import hashlib, json, time

def action_key(call):
    payload = call.name + json.dumps(call.args, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def partial(status, reason, steps, cost, deps):
    return {"status": status, "reason": reason, "steps": steps, "cost": cost,
            "done": deps.artifacts(), "next_cursor": deps.cursor()}

async def run(goal, tools, cfg, deps, clock=time.monotonic):
    """cfg 携带全部预算：max_steps / wall_clock_s / input_token_budget / usd_budget
    以及阈值 max_same_action / max_no_progress / similarity_threshold。"""
    t0, cost = clock(), {"in": 0, "out": 0, "usd": 0.0}
    messages, hits, last_obs, stuck = [user(goal)], {}, None, 0

    for step in range(1, cfg.max_steps + 1):
        if clock() - t0 > cfg.wall_clock_s:
            return partial("budget_exhausted", "wall_clock", step, cost, deps)   # ③ 墙钟
        if cost["in"] >= cfg.input_token_budget or cost["usd"] >= cfg.usd_budget:
            return partial("budget_exhausted", "cost", step, cost, deps)         # ④ 费用

        resp = await call_llm(messages, tools, deadline=t0 + cfg.wall_clock_s)
        cost["in"] += resp.usage.input_tokens
        cost["out"] += resp.usage.output_tokens
        cost["usd"] += usd(resp.usage)

        if resp.is_final:
            ok, diff = await verify(resp.answer)                          # ① 外部验证器
            if ok:
                return {"status": "ok", "answer": resp.answer, "steps": step, "cost": cost}
            obs = "验证未通过，差异：" + diff                              # 回灌后有限次修复
        else:
            obs = await call_tool(resp.tool_call, timeout=cfg.tool_timeout)  # ③ 单步超时
            key = action_key(resp.tool_call)                              # 只有工具调用有动作键
            hits[key] = hits.get(key, 0) + 1
            if hits[key] > cfg.max_same_action:                           # ⑥ 循环检测
                obs += "（同一动作与参数已重复，请换动作或给出最终答案）"
        stuck = stuck + 1 if info_gain(obs, last_obs) < cfg.similarity_threshold else 0
        if stuck >= cfg.max_no_progress:                                  # ⑤ 无进展检测
            return partial("partial", "no_progress", step, cost, deps)
        last_obs = obs
        messages.append(observe(obs))

    return partial("partial", "step_limit", cfg.max_steps, cost, deps)     # ② 步数上限
```

`verify`、`call_tool`、`info_gain`、`usd`、`deps` 为运行时提供的组件，此处只体现调用位置与顺序：验证器在返回之前、预算检查在 LLM 调用之前、循环与无进展检测在观察入历史之前。循环检测只对工具调用算动作键，最终答案分支不参与（验证失败时走 diff 回灌修复，不计入重复计数）。`partial()` 的第一个参数就是 envelope 里的 `status`，`reason` 只补充是哪一层触发的。

### 代码：pass^k 与 pass@k 的估计量

```python
from math import comb
from statistics import mean

def pass_hat_k(n, c_list, k):
    """n: 每任务试验次数；c_list: 每任务成功次数；k: 要求全部成功的次数。"""
    return mean(comb(c, k) / comb(n, k) for c in c_list)

def pass_at_k(n, c_list, k):
    """comb 在 k > n 时返回 0，两个估计量都不需要额外分支。"""
    return mean(1 - comb(n - c, k) / comb(n, k) for c in c_list)

c_list = [8, 7, 6, 5, 4]          # 5 个任务，各跑 8 次
print(pass_hat_k(8, c_list, 1), pass_hat_k(8, c_list, 4), pass_hat_k(8, c_list, 8))
print(pass_at_k(8, c_list, 4))
```

输出为 `0.75 0.36 0.2` 与 pass@4 的 `0.997`：单次成功率 75% 的任务集，连续 4 次全部成功的平均概率降到 36%，连续 8 次只剩 20%——5 个任务里只有那个 8/8 的任务能贡献一次「8 次全成功」，其余任务的 $c<8$ 使 $\binom{c}{8}=0$。同一份数据里 pass@4 高达 0.997，两个指标给出完全相反的结论，这正是「平均值掩盖不一致」的量化版本。

## 常见追问

- **追问**：怎么区分「需要更多步」与「陷入循环」？
  - 要点：看观察的**信息增量**，不看步数。出现新事实、新状态变更、新错误码就是真进展（哪怕慢）；连续 3 步以上只有措辞变化、无可验证进展，就按循环处理——提前退出并带上原因，比再多给 10 步更划算。
- **追问**：pass^k 与 pass@k 的差别，什么时候用哪个？
  - 要点：pass@k 是 $k$ 次里至少一次成功，用于衡量探索上限（采样、多次生成、代码候选）；pass^k 是 $k$ 次全部成功，用于衡量可交付的一致性。面向用户的生产接口看 pass^k，离线挖掘能力看 pass@k。
- **追问**：流式输出下如何优雅终止与回滚？
  - 要点：终止决策要在**下一个可安全中断的边界**执行（工具调用之间、而不是生成句子中间）；已经流给用户的内容不可撤回，所以要在流式层做缓冲与「可丢弃的草稿」标记；已产生的外部副作用走补偿事务，见 [[agents-10]]。
- **追问**：不同风险等级的任务如何设不同预算？
  - 要点：按副作用分级——只读查询给宽预算（失败可重试）；可逆写操作给中等预算 + 自动补偿；不可逆写操作给小预算、单步验证、必要时审批，见 [[agents-11]]。同一套运行时，靠随请求传入的配置切换。
- **追问**：预算耗尽时该抛错还是返回部分结果？
  - 要点：返回部分结果并显式标注 `budget_exhausted`，附已完成清单与可续跑游标；抛错会让上游重试整条链路，浪费已经花掉的预算。但绝不能为了「做完整」而放宽验证器——那等于用正确性换完成率。
- **追问**：为什么不能只用「模型说完成了」作为终止条件？
  - 要点：过早停止（premature stopping）与自信的错误结论都是常见失效，指令越模糊越容易发生；外部验证器把「完成」从模型的主观判断变成可执行断言，这是 τ-bench 用数据库终态比对来判定成功的理由。

## 相关题目

- [[agents-02]]：工具调用错误、超时与重试——终止条件里「可重试 vs 不可重试」的判定基础。
- [[agents-06]]：多 agent 编排如何运作、什么情况下会失效——互相等待导致的死锁。
- [[agents-12]]：长时间运行后的漂移诊断——目标遗忘与判据歧义的表现。
- [[agents-07]]：长任务的记忆设计——压缩时如何保护目标与约束。
- [[agents-08]]：调用工具还是依据自身知识回答——减少不必要调用直接省步数。
- [[agents-10]]：操作可逆与可审计——终止、取消、回滚的工程前提。
- [[agents-11]]：human-in-the-loop 审批——人工兜底这一层的设计。
- [[inference-serving-05]]：prefix caching 与 prompt caching——把平方级输入成本压回近线性。
- [[inference-serving-09]]：TTFT、TPOT、ITL 与 throughput——墙钟预算与 token 预算的换算关系。

## 参考资料与归属

- *AI Agent Loop* — Amit Shekhar（Outcome School），页面标注 2026-05-28：<https://outcomeschool.com/blog/ai-agent-loop>（循环的 think-act-observe 结构、自然停止与安全停止、四类循环失效）。
- *Fixing Infinite Loops in AI Agents* — Pallavi Shekhar，LinkedIn 帖文（未标注日期）：<https://www.linkedin.com/posts/pallavi-shekhar_ai-aiagents-machinelearning-share-7440257380707364864-5Ycc>（硬上限约 10 次工具调用、超限返回部分结果、同工具同参数重复超过 2 次即中断并注入提示、结构化错误与 planner-executor）。
- *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains* — Yao, Shinn, Razavi, Narasimhan，2024-06-17：<https://arxiv.org/abs/2406.12045>（延伸来源：pass^k 定义与无偏估计量、以数据库终态比对的奖励口径 r_action × r_output、Table 2 的 61.2% / 35.2% 与按域加权的 avg 48.2%、τ-retail pass^8 <25%）。
- AgentBench（Liu et al., ICLR 2024）关于「8 个交互环境中长期推理、决策与指令遵循是主要障碍」的结论按题面口径引用，此处不另附链接。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
