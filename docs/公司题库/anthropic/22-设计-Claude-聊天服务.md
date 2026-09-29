---
type: question
id: anthropic-22
company: Anthropic
topic: system-design
order: 22
question: 设计 Claude 聊天服务。
question_en: Design the Claude chat service.
asked_at: []
level: 高阶
tags: [系统设计, 聊天服务, 会话记忆, 流式, 安全层, 容量规划]
sources:
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: How does Prompt Caching work?
    url: https://outcomeschool.com/blog/how-does-prompt-caching-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: 深入 ChatGPT：按下回车之后会发生什么
    url: https://outcomeschool.substack.com/p/inside-chatgpt-what-happens-after
    author: Amit Shekhar (Outcome School)
    published: 
  - title: SGLang：Efficient Execution of Structured Language Model Programs（延伸）
    url: https://arxiv.org/abs/2312.07104
    author: Zheng et al. (RadixAttention)
    published: 2023-12-12
related: [anthropic-17, anthropic-23, system-design-10, inference-serving-05, anthropic-24]
updated: 2026-09-28
---

## 一句话答案

> 聊天服务 = **一个会话状态系统 + 一个模型服务平台 + 一层安全与计费**。按请求路径讲最清楚：
> ① **接入**：鉴权、配额、请求去重（客户端重试不能重复计费）、安全前置检查（输入侧分类器）；
> ② **会话层**：把「用户消息 + 历史 + 附件 + 记忆」组装成模型输入——**这是产品差异化的核心**，也是成本的主要来源；
> ③ **检索层**：附件与知识库的分块、索引与召回（有引用就必须能溯源）；
> ④ **推理层**：continuous batching + 分页 KV + 前缀缓存（系统提示与会话前缀高度可复用）+ 流式输出（SSE）；
> ⑤ **安全层**：输出侧分类与拦截（流式下必须**边生成边判**，否则只能事后撤回）；
> ⑥ **计费与观测**：按 token 计量（输入/输出/缓存命中分开计价）、TTFT/TPOT 分位、每会话成本。
> **三个必须主动指出的设计约束**：
> - **成本随会话长度超线性增长**：历史全带上 → 每轮输入 token 线性增长，而**缓存命中能把重复前缀的算力几乎归零**，所以「上下文组装策略」直接决定单位经济性；
> - **流式与安全冲突**：边生成边推送意味着「已发出的内容」无法收回，因此高风险类别必须**先判后发**（缓冲若干 token）或用更保守的策略；
> - **多设备一致性**：会话是用户可见的状态，必须能跨设备同步、支持重放与编辑重发（编辑历史 → 分叉出新的分支，而不是覆盖）。

## 面试官在考什么

- **是否把「聊天」当状态系统而不是无状态 API**：会话、分支、附件、记忆、配额——这些是聊天服务的真正复杂度所在（模型推理反而是可以外包的能力）。
- **上下文组装策略**：全量历史 vs 滑窗 vs 摘要 vs 检索式记忆；各自的成本与质量影响；**缓存友好性**（前缀稳定才能命中）是否被考虑。
- **成本结构**：能否给出「每轮成本 = 输入 token × 单价 + 输出 token × 单价」并指出**输入侧随轮数增长**，以及缓存命中把该项降到多少。
- **流式与安全**：SSE 的工程细节（分片、重连、幂等续传）与安全冲突（已输出的内容不可收回）。
- **容量规划**：DAU → 并发会话 → QPS → 实例数；KV 显存决定并发（串 [[anthropic-17]]）。
- **失败与降级**：模型超时/超额时如何返回部分结果、如何提示重试、如何避免重复计费（幂等键）。
- **能说清「哪些不做」**：例如不做端到端加密（否则服务端无法做安全与检索），但要做数据保留策略与用户可见的删除。

**常见错误答案**

- 只画「客户端 → 网关 → 模型 → 返回」，没有会话状态、检索、安全与计费。
- 把安全当成一次性过滤（「输入输出各过一次分类器」），忽略流式下的时序问题。
- 忽略缓存：每轮都把全部历史原样重算，成本随轮数平方增长（输入侧线性 × 轮数）。
- 忽略多设备与编辑重发的语义（覆盖 vs 分支）。
- 无容量与成本数字。

## 原理与推导

### 1. 端到端请求路径

```
客户端（多设备）
  │  ① POST /messages（幂等键 + 会话 id）
  ▼
接入层：鉴权 → 配额检查 → 输入安全分类 → 去重（幂等键命中则复用结果）
  │
  ▼
会话层：加载会话状态（历史/分支/附件/记忆）→ 组装上下文
  │        ├─ 稳定前缀（系统提示 + 工具定义 + 长期记忆）→ 走前缀缓存
  │        └─ 动态部分（最近 N 轮 + 本轮检索片段）
  ▼
检索层：附件/知识库 → 分块检索（混合检索）→ 重排 → 带引用注入
  │
  ▼
推理层：continuous batching + 分页 KV + 流式生成（SSE/WebSocket）
  │        └─ 输出安全：边生成边判（缓冲窗口）或先判后发
  ▼
计费与观测：token 计量（输入/输出/缓存命中分开）→ 归档 → 指标
  │
  ▼
客户端：流式渲染 + 引用锚点 + 可编辑重发
```

### 2. 上下文组装：四种策略的成本

设第 $n$ 轮，系统提示与工具定义为 $S$（稳定）、每轮新增用户消息与回答平均 $u$ token：

| 策略 | 第 n 轮输入 token | 总输入（N 轮） | 缓存友好 | 质量 |
| --- | --- | --- | --- | --- |
| 全量历史 | $S+n\cdot u$ | $NS+\frac{uN(N+1)}{2}$（**平方**） | 好（前缀稳定） | 最佳 |
| 滑窗（最近 k 轮） | $S+k\cdot u$ | $N(S+ku)$（线性） | 中（窗口滑动，前缀被破坏） | 丢失远期信息 |
| 摘要 + 最近 k 轮 | $S+k u+m$（$m$ 为摘要） | $N(S+ku+m)$ | 中 | 折中 |
| 检索式记忆 | $S+ku+\text{top-}r$ 片段 | 线性 | 差（检索结果不稳定） | 依赖检索质量 |

**关键工程点**：**前缀缓存要求前缀稳定**——滑窗与检索都会改变前缀，导致缓存命中率下降。因此产品化做法常常是「**分层**」：稳定前缀（系统提示 + 长期记忆摘要）放在最前（永远命中缓存），动态部分放最后（缓存未命中但量小）。这一条能把输入成本压到接近线性（串 [[inference-serving-05]]）。

### 3. 流式与安全的时序问题

流式下有三种策略，各有代价：

| 策略 | 做法 | 延迟代价 | 风险 |
| --- | --- | --- | --- |
| 全缓冲后审 | 生成完整 → 输出安全判定 → 一次性返回 | TTFT 变成总时长（不可接受） | 无 |
| 窗口缓冲 | 缓冲 $w$ 个 token，判定通过就放行 | 增加 $w$ 个 token 的延迟 | 判定滞后 $w$ |
| 边发边判（事后撤回） | 立即推送，发现违规再撤回 | 无 | **已发出的内容无法收回** |

**工程折中**：对高风险类别（自伤、武器、CSAM 等）用「窗口缓冲 + 更严格分类器」，对普通类别用「边发边判 + 撤回 + 记录」；同时客户端要支持「撤回」语义（明确告知内容被移除）。

### 4. 会话状态的建模

| 概念 | 作用 | 关键设计 |
| --- | --- | --- |
| 会话（conversation） | 用户可见的连续对话 | 有序消息列表 + 元数据 |
| 消息（message） | 用户/助手/工具的一条记录 | 内容寻址、不可变、带 token 计量 |
| 分支（branch） | 编辑重发产生新分支 | 父指针 + 分支版本（不覆盖历史） |
| 附件（attachment） | 文件与其索引 | 内容哈希去重 + 索引状态机 |
| 记忆（memory） | 跨会话的长期信息 | 显式条目 + 用户可见可删 |
| 配额（quota） | 计费与限流 | 按 token/请求数，幂等扣减 |

**多设备一致性**：会话用**单一权威写入点**（服务端）+ 客户端乐观更新 + 服务端序列号（客户端按序应用）；编辑重发走分支而不是覆盖，避免「两端各改一半」的冲突。

### 5. 容量与成本模型

$$\text{并发会话}\approx\frac{\text{DAU}\times\text{每活跃用户并发会话数}}{\text{会话时长}}\times\text{平均会话时长}=\text{DAU}\times\text{并发比例}$$

- **推理容量**：由吞吐与 KV 显存共同决定（串 [[anthropic-17]] 的表 2）；
- **成本**：$\text{cost}=\sum_{\text{turns}}\big(\text{in}_t\cdot p_{\text{in,miss}}+\text{in}_t^{\text{cache}}\cdot p_{\text{in,cache}}+\text{out}_t\cdot p_{\text{out}}\big)$；
- **单位经济性指标**：每活跃用户日成本、每千轮成本、缓存命中带来的节省比例。

## 数值与代码验证

### 表 1：上下文策略的成本对比（$S$=2,000、$u$=400 token、20 轮、缓存命中价按输入价 10% 计）

| 策略 | 20 轮总输入 token | 无缓存成本（相对） | 有缓存命中 | 说明 |
| --- | --- | --- | --- | --- |
| 全量历史 | 124,000 | 1.00 | 0.16 | 平方增长，但前缀稳定、缓存收益最大 |
| 滑窗（k=4） | 72,000 | 0.58 | 0.35 | 线性，但前缀滑动导致命中率低 |
| 摘要 + 滑窗 | 84,000 | 0.68 | 0.37 | 折中方案 |
| 分层（稳定前缀 + 滑窗） | 84,000 | 0.68 | **0.13** | **最优**：前缀永远命中 |

（相对值按「全部输入按未命中价」为 1.0 计算；表内为可运行代码的实测输出。**注意一个反直觉结论**：全量历史虽然 token 最多，因为有缓存，成本反而**低于**滑窗与摘要策略——省 token 的常规做法若不考虑缓存友好性，可能更贵。）

### 表 2：容量与成本量级（示例：100 万 DAU、并发比例 2%、每会话 8 轮）

| 项 | 算式 | 结果 |
| --- | --- | --- |
| 并发会话 | $10^6\times0.02$ | 20,000 |
| 每轮平均输入/输出 | 见上表 | 6.2K / 0.5K token |
| 峰值 QPS（每会话每 20 s 一轮） | $20{,}000/20$ | 1,000 req/s |
| 输出 token 吞吐 | $1{,}000\times500$ | 500K token/s |
| 所需推理实例 | 19K token/s/实例（[[anthropic-16]] 口径） | **约 26 个**（+headroom → 33） |
| 日输入 token | $10^6\times8\times6.2\text{K}$ | $4.96\times10^{10}$（约 500 亿） |
| 日成本（缓存命中 80%、\$3/\$15 每百万 token） | 输入 \$41.7K + 输出 \$60.0K | **约 \$10.2 万/日** |
| 单位经济性 | $\text{日成本}/\text{DAU}$ | **约 \$0.10/活跃用户/日** |

### 可运行代码

```python
# 1) 上下文组装策略：token 成本与缓存收益
def cost_of_strategy(turns=20, S=2000, u=400, window=4, summary=600, cache_price=0.1):
    """返回 (总输入 token, 无缓存相对成本, 有缓存相对成本)"""
    strategies = {}
    # 全量历史：第 n 轮输入 = S + n*u
    full = sum(S + n * u for n in range(1, turns + 1))
    # 全量 + 前缀缓存：稳定的 S + 历史中不变部分可命中，只有新增一轮未命中
    full_cached = sum(S * cache_price + (n - 1) * u * cache_price + u for n in range(1, turns + 1))
    strategies["全量历史"] = (full, full, full_cached)
    # 滑窗：永远只有最近 window 轮；前缀随窗口滑动，命中率设为 0（保守）
    win = sum(S + window * u for _ in range(turns))
    strategies["滑窗(k=4)"] = (win, win, win * 0.6)      # 60% 视为部分命中
    # 摘要 + 滑窗
    summ = sum(S + summary + window * u for _ in range(turns))
    strategies["摘要+滑窗"] = (summ, summ, summ * 0.55)
    # 分层：稳定前缀（S + 摘要）永远命中，只有新增一轮未命中
    layered_uncached = summ
    layered_cached = sum((S + summary) * cache_price + (window - 1) * u * cache_price + u
                         for _ in range(turns))
    strategies["分层(前缀+滑窗)"] = (summ, layered_uncached, layered_cached)
    return strategies

base = None
print(f"{'策略':<18} {'总输入token':>12} {'无缓存(相对)':>13} {'有缓存(相对)':>13}")
for name, (tok, unc, cac) in cost_of_strategy().items():
    if base is None:
        base = unc
    print(f"{name:<18} {tok:>12,d} {unc/base:>13.2f} {cac/base:>13.2f}")
print("读法：全量历史看似最贵（平方增长），但前缀稳定 → 缓存后反而最便宜；")
print("      分层策略把「稳定前缀」与「动态尾部」分开，是成本最优的工程形态")

# 2) 流式安全的三种策略：延迟与风险
def stream_safety(total_tokens, window, tpot_ms=25, classifier_ms=15):
    """返回 (TTFT, 最坏暴露 token 数)"""
    full_buffer = (total_tokens * tpot_ms + classifier_ms, 0)
    windowed = (window * tpot_ms + classifier_ms, 0)          # 缓冲 window 个 token
    immediate = (tpot_ms, window)                              # 判定滞后 window
    return {"全缓冲后审": full_buffer, "窗口缓冲": windowed, "边发边判": immediate}

print(f"\n流式安全策略（总输出 500 token、每 token 25 ms）：")
for name, (ttft, exposure) in stream_safety(500, window=40).items():
    print(f"  {name:<10} TTFT 额外 {ttft:>6.0f} ms，最坏暴露 {exposure:>3} token")
print("读法：全缓冲把 TTFT 拉到十几秒（不可接受）；窗口缓冲用几十 token 的延迟换「零暴露」；")
print("      边发边判延迟最低但已发出的内容不可收回 —— 高风险类别必须用前两种")

# 3) 容量与成本
def plan(dau=1_000_000, concurrency=0.02, turns=8, in_tok=6200, out_tok=500,
         sec_per_turn=20, inst_tps=19000, price_in=3.0, price_out=15.0, cache_hit=0.8):
    concurrent = dau * concurrency
    qps = concurrent / sec_per_turn
    out_tps = qps * out_tok
    instances = out_tps / inst_tps
    daily_in = dau * turns * in_tok
    daily_out = dau * turns * out_tok
    cost_in = daily_in / 1e6 * price_in * ((1 - cache_hit) + cache_hit * 0.1)
    cost_out = daily_out / 1e6 * price_out
    return dict(concurrent=concurrent, qps=qps, instances=instances, daily_in=daily_in,
                cost_in=cost_in, cost_out=cost_out, per_user=(cost_in + cost_out) / dau)
r = plan()
print(f"\n容量：并发会话 {r['concurrent']:,.0f}，峰值 {r['qps']:,.0f} req/s，"
      f"需要约 {r['instances']:.0f} 个推理实例（+25% headroom → {r['instances']*1.25:.0f}）")
print(f"日输入 token {r['daily_in']:.2e}；日成本 ≈ ${r['cost_in']+r['cost_out']:,.0f}"
      f"（输入 ${r['cost_in']:,.0f} + 输出 ${r['cost_out']:,.0f}）")
print(f"单位经济性：每活跃用户日成本 ≈ ${r['per_user']:.4f}；缓存命中 80% 让输入成本降到 "
      f"{(1-0.8)+0.8*0.1:.0%}")
print("读法：输出 token 单价高且不可缓存 → 控制输出长度（max_tokens、提示约束）比压输入更值钱")
```

预期输出要点（实跑）：① 全量历史的**总输入 token 最高（12.4 万）**，但前缀稳定使其**缓存后成本很低（相对 0.16）**；滑窗与摘要策略省了 token 却因前缀不稳定而**缓存收益小（0.35/0.37）**；**分层策略**（稳定前缀 + 小滑窗）在有缓存时最低（相对 **0.13**）——这就是「上下文组装即单位经济性」的量化形态；② 流式安全三策略的对照显示全缓冲把 TTFT 拉到十几秒（不可用）、窗口缓冲以几十 token 的延迟换零暴露、边发边判延迟最低但有暴露风险；③ 容量规划给出并发会话 2 万、峰值 1,000 req/s、约 26 个推理实例，以及日成本结构与「每用户日成本」——并指出**输出 token 不可缓存，控制输出长度比压输入更值钱**。

## 常见追问

- **追问**：记忆怎么存与怎么用？
  - 要点：显式记忆条目（用户可见可删）+ 检索式注入；写入要经过「值得记吗」的判定（否则噪声累积）；注入位置放在稳定前缀里（利于缓存）；隐私上要支持查看/导出/删除（合规要求）。
- **追问**：附件检索怎么做？
  - 要点：上传即异步分块索引（内容哈希去重）、检索用混合（词法 + 向量）+ 重排、生成时必须带**可点击的引用锚点**（溯源是产品要求而非可选）；长文档要做分层摘要以避免上下文爆炸。
- **追问**：多设备同步冲突怎么办？
  - 要点：服务端单一权威写入 + 序列号；客户端乐观更新并回滚；编辑重发产生**分支**而非覆盖；离线操作在重连后按序重放（幂等键去重）。
- **追问**：怎么防止重复计费？
  - 要点：请求级幂等键（客户端生成）+ 服务端在事务里扣减配额并记录结果；重试命中已有结果则直接返回而不重复扣费。
- **追问**：如何处理编辑重发（regenerate/edit）？
  - 要点：把它建模为**从某个节点分叉**：原消息保留，新分支记录父指针；计费按新分支的实际 token；UI 上可切换分支（这也是「用户可见的状态」设计的一部分）。
- **追问**：延迟不达标怎么排查？
  - 要点：分段观测——接入（鉴权/配额）、**排队**、检索、prefill（TTFT）、decode（TPOT）、后处理、网络；再对照 [[anthropic-17]] 的四组指标判断瓶颈在算力/显存/CPU/队列。

## 相关题目

- [[anthropic-17]]：API 侧的服务栈与利用率-p99 权衡，是本题推理层的直接基础。
- [[anthropic-23]]：单会话内多问题的调度与上下文共享，是本题会话层的深入。
- [[system-design-10]]：面向数亿用户的消费级聊天助手，与本题是同一条产品线的另一种问法。
- [[inference-serving-05]]：前缀缓存与 KV 复用，决定本题上下文策略的经济性。
- [[anthropic-24]]：大规模检索系统，是本题检索层在极端规模下的展开。

## 参考资料与归属

- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 1 节推理层的分页 KV 与跨请求前缀共享机制来自这篇。
- **How does Prompt Caching work?** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-prompt-caching-work>。第 2 节「前缀稳定才能命中缓存」的机制与收益表述转述自这篇。
- **SGLang：Efficient Execution of Structured Language Model Programs（延伸）** —— Zheng et al. (RadixAttention)，2023-12-12：<https://arxiv.org/abs/2312.07104>。第 2 节分层上下文与基数树式 KV 复用的思路来自这篇。
- **深入 ChatGPT：按下回车之后会发生什么** —— Amit Shekhar (Outcome School)：<https://outcomeschool.substack.com/p/inside-chatgpt-what-happens-after>。第 1 节端到端请求路径的产品侧背景参照这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（$S$=2,000、$u$=400、20 轮、缓存价 = 输入价 10%、100 万 DAU、并发比例 2%、8 轮/会话、6.2K 输入/500 输出、19K token/s/实例、\$3/\$15 每百万 token）都是按本仓库统一口径构造的工程算例与显式假设；真实产品的定价与流量特征不同，容量必须用压测校准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
