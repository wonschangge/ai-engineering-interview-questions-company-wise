---
type: question
id: cognition-05
company: Cognition（Devin、Windsurf）
topic: system-design
order: 5
question: 为数千个并发的云端 coding agent 设计执行环境。它必须能在 agent 等待 CI 长达四十分钟的情况下依然稳定运行。
question_en: Design the execution environment for thousands of concurrent cloud coding agents. It must stay stable while agents wait up to forty minutes for CI.
asked_at: []
level: 高阶
tags: [执行环境, 沙箱, 挂起恢复, 资源池化, CI-等待, 准入控制]
sources:
  - title: Handling Overload（Google SRE Book 第 21 章）（延伸）
    url: https://sre.google/sre-book/handling-overload/
    author: Google SRE
    published: 
  - title: Don't Build Multi-Agents（Cognition 博客）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: 在生产系统中如何让 agent 的操作可逆或至少可审计（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 为数千个并发的云端 coding agent 设计执行环境（本仓库公司题库 · Anthropic 篇的相关题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cognition-04, cognition-08, cognition-02, system-design-01, anthropic-17]
updated: 2026-09-28
---

## 一句话答案

> 题面里的「**等待 CI 长达四十分钟**」就是全部题眼：**等待的 agent 不能占着独占资源**。所以核心设计是一句话：
> $$\text{把「等待」做成状态保存 + 资源释放 + 事件唤醒，而不是「占着容器 sleep」}$$
> 六层结构：
> ① **控制面（scheduler/admission）**：维护会话状态机（`pending → running → waiting_ci → running → done`），负责分配/回收沙箱，并做**准入控制**（池子满了排队，而不是超卖）；
> ② **会话存储（session store）**：工作区的**增量快照**（overlay 上层 + 元数据）+ agent 的**推理状态**（对话/计划/工作集）。**这两样都必须持久化**，否则唤醒后 agent 会「失忆」；
> ③ **热池（warm pool）**：预热的镜像/容器，缩短冷启动；池子大小由「到达率 × 启动时间」决定；
> ④ **沙箱运行时**：每 agent 一个隔离沙箱（microVM/容器 + overlayfs），**用完即还**；镜像按仓库/语言预构建；
> ⑤ **CI 集成**：**webhook 优先**（推送式唤醒），轮询只作兜底并**指数退避**；等待有上限与取消语义；重连必须幂等；
> ⑥ **成本与稳定性杠杆**：$70\%$ 的 agent 在等 CI 时，只需为 $30\%$ 准备算力——**这就是「能扛住 40 分钟等待」的定量原因**。
> 一句话判据：**把「等待」从「占用资源」变成「占用一行数据库记录」**；谁把 sleep 留在容器里，谁的成本模型就是错的。

## 面试官在考什么

- **是否抓住「释放资源」这一点**：能否主动提出挂起/快照 + 唤醒，而不是「给容器加长超时」。
- **状态的两部分**：能否区分**工作区状态**（文件、git 状态）与**agent 推理状态**（对话、计划、已确认结论）——两者都要持久化，缺一就会「醒来失忆」。
- **CI 等待的具体机制**：webhook vs 轮询（退避）、超时与重试、幂等重连、以及「CI 失败后如何恢复上下文」。
- **准入与背压**：池子耗尽时排队还是拒绝？公平性（多租户）怎么保证？能否引 SRE 的过载处理口径（串 [[anthropic-17]]）。
- **隔离与安全**：客户仓库写权限 + CI 凭证 + 网络访问——沙箱隔离、出网白名单、按任务下发临时凭证、审计（串 [[cognition-08]] 的威胁模型）。
- **快照/恢复的工程细节**：增量（overlay 上层）、去重、存储位置与**恢复延迟预算**；以及失败处理（快照坏了怎么办）。
- **可观测性**：池饱和度、排队时长 p95、冷启动 p95、快照/恢复成功率、僵尸沙箱数、每个 agent 的资源画像。
- **成本量级**：$\text{沙箱小时}\times\text{单价}$；能否算出「不挂起」与「挂起」的成本差（通常是数倍）。
- **诚实**：会说「挂起/恢复有代价（快照时间、恢复失败率）」，并给出权衡（例如短等待直接留着，长等待才挂起）。

**常见错误答案**

- 让 agent 在容器里 `sleep` 等 CI（成本随等待时间线性上升）。
- 只保存文件，不保存 agent 的推理状态（醒来后重复劳动）。
- 用轮询（每秒查一次）而不是 webhook（打爆 CI API，也浪费资源）。
- 没有准入控制（池子耗尽后全部超时失败，雪崩）。
- 忽略「同一仓库热点」（几千个 agent 同时 clone 同一个大仓库 → 存储与网络打满）。
- 沙箱跨租户复用而不清洗（数据泄漏）。
- 没有快照/恢复的成功率与延迟指标（出问题无法定位）。

## 原理与推导

### 1. 会话状态机（把「等待」变成一等状态）

```
pending ──分配沙箱──▶ running ──触发 CI──▶ waiting_ci ──CI 完成(webhook)──▶ running
   ▲                     │                    │                              │
   │                     │                    └── 快照并归还沙箱 ─────────────┘
   │                     │                                                   │
   └── 排队（池满）───────┘                    超时/取消 ──▶ failed/cancelled ◀┘
```

**关键**：`waiting_ci` 状态下**不持有沙箱**，只持有会话记录。唤醒时按需重新分配沙箱并恢复快照。

### 2. 容量模型（本题的核心算术）

设并发 agent 数 $N$，其中处于 `waiting_ci` 的比例为 $w$：

$$\text{需要的沙箱数}\approx N(1-w)\times(1+\text{余量})$$

| 场景 | $N$ | $w$ | 需要沙箱 | 不挂起时（$w=0$） |
| --- | --- | --- | --- | --- |
| 常态 | 10,000 | 0.70 | **3,000** | 10,000 |
| 高峰 | 10,000 | 0.55 | 4,500 | 10,000 |
| CI 大面积变慢 | 10,000 | 0.85 | 1,500 | 10,000 |

**读法**：**「等待」是免费的容量来源**——只要能挂起，$w$ 越高需要的沙箱越少。这也是「必须扛住 40 分钟等待」的正面解读：**40 分钟不消耗沙箱，只消耗一行记录与一份快照**。

### 3. 快照与恢复

| 项 | 做法 | 目标 |
| --- | --- | --- |
| 快照对象 | 工作区（overlay 上层 + git 状态） | 增量、可去重 |
| 存储 | 对象存储（廉价）+ 本地缓存（热仓库） | 成本与延迟平衡 |
| 恢复 | 拉取快照 → 挂载 → 恢复依赖缓存 | **p95 恢复 ≤ 数秒** |
| 失败处理 | 校验 + 重试 + 从 git 重建（兜底） | 恢复成功率 ≥ 99.9% |
| 推理状态 | 会话记录（对话/计划/工作集） | 数据库 + 版本号 |

**恢复延迟预算**（示例）：调度 100 ms + 快照拉取 1–3 s（增量）+ 容器挂载 200 ms + 依赖缓存 500 ms ≈ **2–4 s**。**若恢复要 30 s，那「挂起」就失去意义**（用户感知为卡顿）。

### 4. CI 集成：推送式唤醒

| 方式 | 机制 | 评价 |
| --- | --- | --- |
| **webhook** | CI 完成后回调控制面 | ✅ 首选（零轮询成本、延迟低） |
| 轮询 + 退避 | 1s → 2s → 4s → …上限 30s | 兜底（webhook 丢失时） |
| 长连接/事件流 | 控制面订阅 CI 事件 | 可用（需处理断线重连） |

**必须有**：等待上限（例如 60 分钟）、取消语义（用户点了停止）、**幂等重连**（重复唤醒不得重复执行副作用）、以及 CI 失败后的**上下文恢复**（把失败日志作为新观察喂回 agent）。

### 5. 准入控制与公平性

- **背压**：池子利用率 > 85% 时开始**排队**（不是继续接单）；
- **公平性**：按租户/用户加权（防止单租户占满池子）；
- **优先级**：交互式（用户在场）优先于批处理；waiting_ci 唤醒优先于新任务（因为已投入成本）；
- **过载策略**：明确拒绝（429 + 重试语义）优于「接了但做不完」（串 [[anthropic-17]]）。

### 6. 隔离与安全（云端 agent 的必需项）

| 风险 | 机制 |
| --- | --- |
| 跨租户数据泄漏 | 沙箱不复用（或复用前强制清洗 + 校验）；工作区按租户加密 |
| 凭证泄漏 | **按任务下发短期凭证**（token broker），沙箱内不落地长期密钥 |
| 出网滥用 | 出网代理 + 白名单（包源、CI、模型 API）+ 流量审计 |
| 恶意代码 | microVM/gVisor 级隔离、资源限额（CPU/内存/磁盘/进程数） |
| 审计 | 每次工具调用与 git 操作留痕（谁、何时、哪个会话） |

### 7. 失败模式与对策

| 失败模式 | 症状 | 对策 |
| --- | --- | --- |
| 僵尸沙箱 | 池子被占满但无人在用 | 心跳 + TTL 回收；会话状态机对账 |
| 快照损坏 | 恢复失败 | 校验和 + 从 git 重建兜底 |
| CI API 限流 | 轮询被拒 | webhook 优先 + 退避 + 配额管理 |
| 仓库热点 | 同仓库 clone 打满网络 | 本地镜像缓存/共享克隆（只读层）+ 限速 |
| 唤醒风暴 | CI 同时完成 → 同时抢沙箱 | 排队 + 批量唤醒的速率限制 |
| 长时间运行泄漏 | 磁盘/内存缓慢增长 | 每会话资源画像 + 定期回收 |

## 数值与代码验证

### 表 1：挂起 vs 不挂起的成本（10,000 并发 agent、70% 在等 CI、等 CI 25 分钟）

| 策略 | 需要沙箱 | 沙箱小时/日 | 计算 \$/日 | 存储 \$/日 | 合计 \$/日 | 每 agent |
| --- | --- | --- | --- | --- | --- | --- |
| 不挂起（容器内等待） | 12,500 | 300,000 | 75,000 | 0.00 | **75,000** | 7.500 |
| 挂起（等待即释放） | **3,750** | **25,714** | 6,429 | 9.33 | **6,438** | **0.644** |

### 表 2：热池大小与冷启动（到达率 × 启动时间）

| 到达率（沙箱/分钟） | 启动时间 | 需要热池 | 说明 |
| --- | --- | --- | --- |
| 20 | 20 s | **约 6.7** | 预热池下限 |
| 50 | 20 s | **约 16.7** | 预热池下限 |
| 20 | 60 s | **20.0** | 预热池下限 |
| 100 | 30 s | 50.0 | 需要分片预热 |

### 可运行代码

```python
# 云端 coding agent 的执行环境：挂起/恢复的容量与成本、热池、准入排队、快照延迟
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Fleet:
    agents: int = 10_000              # 并发 agent 数
    wait_ci_share: float = 0.70       # 处于"等 CI"的比例
    avg_task_min: float = 35.0        # 平均任务时长（含等待）
    ci_wait_min: float = 25.0         # 其中等 CI 的时间
    sandbox_price_h: float = 0.25     # 沙箱（CPU 型）每小时单价
    margin: float = 1.25              # 余量（突发 + 恢复并发）
    snapshot_gb: float = 2.0          # 单次快照增量大小
    storage_price_gb_month: float = 0.02
    def active_sandboxes(self, suspend: bool) -> float:
        if suspend:
            return self.agents * (1 - self.wait_ci_share) * self.margin
        return self.agents * self.margin

def daily_cost(f: Fleet, suspend: bool) -> Dict[str, float]:
    boxes = f.active_sandboxes(suspend)
    # 每个 agent 每天完成的"任务数"：按活跃占比折算沙箱小时
    tasks_per_agent_per_day = 24 * 60 / f.avg_task_min
    compute_min = f.avg_task_min - (f.ci_wait_min if suspend else 0)
    sandbox_hours = boxes * 24 * (compute_min / f.avg_task_min) if suspend else boxes * 24
    compute_cost = sandbox_hours * f.sandbox_price_h
    # 挂起时：等待期间占用存储（快照）
    wait_agents = f.agents * f.wait_ci_share if suspend else 0
    storage_gb = wait_agents * f.snapshot_gb
    storage_cost = storage_gb * f.storage_price_gb_month / 30
    return {"沙箱数": boxes, "沙箱小时/日": sandbox_hours,
            "计算成本/日": compute_cost, "存储成本/日": storage_cost,
            "合计/日": compute_cost + storage_cost,
            "每 agent 每日成本": (compute_cost + storage_cost) / f.agents}

f = Fleet()
print("① 挂起 vs 不挂起（10,000 并发 agent、70% 在等 CI、等 CI 25 分钟）")
print(f"  {'策略':<22} {'沙箱数':>8} {'沙箱小时/日':>12} {'计算$/日':>10} "
      f"{'存储$/日':>9} {'合计$/日':>10} {'每agent':>8}")
for suspend in (False, True):
    r = daily_cost(f, suspend)
    label = "挂起（等待即释放）" if suspend else "不挂起（容器内等待）"
    print(f"  {label:<22} {r['沙箱数']:>8,.0f} {r['沙箱小时/日']:>12,.0f} "
          f"{r['计算成本/日']:>10,.0f} {r['存储成本/日']:>9,.2f} {r['合计/日']:>10,.0f} "
          f"{r['每 agent 每日成本']:>8.3f}")
base = daily_cost(f, False)["合计/日"]
sus = daily_cost(f, True)["合计/日"]
print(f"  读法：挂起把沙箱数从 {daily_cost(f,False)['沙箱数']:,.0f} 降到 "
      f"{daily_cost(f,True)['沙箱数']:,.0f}，日成本从 ${base:,.0f} 降到 ${sus:,.0f} "
      f"（省 {1-sus/base:.0%}）——")
print("        关键：**等待只花存储（几美分级别），不花沙箱**")

print("\n② 等 CI 比例对成本的影响（挂起策略下）")
print(f"  {'等 CI 比例':>10} {'沙箱数':>8} {'合计$/日':>10} {'相对不挂起节省':>14}")
for w in (0.30, 0.50, 0.70, 0.85):
    f2 = Fleet(wait_ci_share=w)
    r = daily_cost(f2, True)
    r_nosus = daily_cost(f2, False)
    print(f"  {w:>10.0%} {r['沙箱数']:>8,.0f} {r['合计/日']:>10,.0f} "
          f"{1-r['合计/日']/r_nosus['合计/日']:>14.0%}")
print("  读法：等 CI 比例越高，挂起的收益越大；但即使只有 30% 在等，也能省下可观成本 ——")
print("        所以「挂起」不是优化项，而是**云端 agent 的基础设施形态**")

print("\n③ 热池大小：到达率 × 启动时间（Little 定律口径）")
def warm_pool(arrival_per_min: float, start_s: float, target_wait_s: float = 5.0) -> Dict[str, float]:
    lam = arrival_per_min / 60.0        # 每秒到达
    in_flight = lam * start_s           # 正在启动的实例数
    return {"需要热实例": in_flight, "启动期间到达数": in_flight,
            "目标等待(s)": target_wait_s}
print(f"  {'到达(个/分)':>11} {'启动(s)':>8} {'需要热池':>9} {'说明':<20}")
for arr, st in ((20, 20), (50, 20), (20, 60), (100, 30)):
    r = warm_pool(arr, st)
    print(f"  {arr:>11} {st:>8} {r['需要热实例']:>9.1f} "
          f"{'预热池下限' if r['需要热实例'] < 30 else '需要分片预热':<20}")
print("  读法：热池大小 ≈ 到达率 × 启动时间；启动慢（60 s）就必须多预热或**边恢复边预热**（串 ②）")

print("\n④ 准入控制：池子利用率与排队时长")
def erlang_c(c: int, a: float) -> float:
    """Erlang C：到达 a erlang、c 个服务台时"到达即排队"的概率（对数空间递推，避免下溢）"""
    logB = 0.0                                  # B(0) = 1
    for k in range(1, c + 1):
        laB = math.log(a) + logB
        denom = k + (math.exp(laB) if laB > -700 else 0.0)
        logB = laB - math.log(denom)
    B = math.exp(logB) if logB > -700 else 0.0
    rho = a / c
    if B <= 0:
        return 0.0
    return B / (1 - rho * (1 - B))

def queue_wait_mmc(util: float, service_min: float = 30.0,
                   servers: int = 3000) -> Tuple[float, float]:
    """返回 (平均排队分钟, Erlang C)。M/M/c：W_q = C / (c·μ − λ)"""
    mu = 1 / service_min
    a = servers * util                          # 到达强度（erlang）
    lam = a * mu
    C = erlang_c(servers, a)
    denom = servers * mu - lam
    if denom <= 1e-12:
        return float("inf"), C
    return C / denom, C

print(f"  {'利用率':>7} {'Erlang C':>10} {'平均排队(min)':>13} {'p95 排队(min)':>13} {'结论':<14}")
for util in (0.50, 0.70, 0.85, 0.90, 0.95, 0.99):
    w, C = queue_wait_mmc(util)
    p95 = w * 2.2                               # 粗略：p95 ≈ 2.2× 均值（指数尾）
    verdict = ("理想模型下可忽略" if w < 0.1 else
               "理想模型下仍短" if w < 1 else "需排队")
    print(f"  {util:>7.0%} {C:>10.2e} {w:>13.3f} {p95:>13.3f} {verdict:<14}")
print("  读数注意：在 3,000 个服务台的规模下，M/M/c 的排队在 ρ≤0.9 时几乎为零（大数定律），")
print("  这是**理想化**结果；真实系统还要叠加**任务时长重尾**与**到达突发**，排队会明显提前")
print("  读法：理想 M/M/c 下排队到 ρ≈0.95 才抬头，但**任务时长重尾 + 到达突发**会让拐点前移 ——")
print("        所以运维口径仍是「把稳态利用率压在 ~85% 以下，并在 85% 开始排队/拒绝」（串 [[anthropic-17]]）")

print("\n⑤ 快照与恢复的延迟预算（决定「挂起」是否值得）")
@dataclass
class Resume:  # noqa: D101
    schedule_ms: float = 100          # 调度决策
    pull_ms: float = 1800             # 拉取增量快照
    mount_ms: float = 200             # 挂载 overlay
    deps_ms: float = 500              # 依赖缓存挂载
    warmup_ms: float = 300            # 进程/工具预热
    def total_s(self) -> float:
        return (self.schedule_ms + self.pull_ms + self.mount_ms
                + self.deps_ms + self.warmup_ms) / 1000
r = Resume()
print(f"  调度 {r.schedule_ms:.0f} + 快照 {r.pull_ms:.0f} + 挂载 {r.mount_ms:.0f} + "
      f"依赖 {r.deps_ms:.0f} + 预热 {r.warmup_ms:.0f} = {r.total_s():.1f} s")
print(f"  对照：等 CI 25 分钟 = {25*60:.0f} s —— 恢复只占 {r.total_s()/(25*60):.2%}")
print("  读法：只要**恢复延迟 ≪ 等待时长**，挂起就是净收益；反之（例如恢复要 30 s）")
print("        短等待不如直接留着容器 —— 所以要设一个「最小挂起时长」阈值")

print("\n⑥ 阈值策略：多长的等待才值得挂起")
def should_suspend(wait_min: float, snapshot_s: float = 1.8, resume_s: float = 2.9,
                   sandbox_price_h: float = 0.25, io_cost: float = 0.0002) -> Dict[str, float]:
    hold_cost = wait_min / 60 * sandbox_price_h
    suspend_cost = io_cost * 2 + resume_s / 3600 * sandbox_price_h  # 快照+恢复+占用
    return {"等待时长(min)": wait_min, "不挂起成本($)": hold_cost,
            "挂起成本($)": suspend_cost, "是否值得": 1.0 if suspend_cost < hold_cost else 0.0}
print(f"  {'等待(min)':>9} {'不挂起($)':>10} {'挂起($)':>9} {'值得挂起':>9}")
for w in (0.5, 2, 5, 15, 40):
    r = should_suspend(w)
    print(f"  {w:>9.1f} {r['不挂起成本($)']:>10.4f} {r['挂起成本($)']:>9.4f} "
          f"{('是' if r['是否值得'] else '否'):>9}")
print("  读法：存在一个**最小挂起阈值**（本例约几分钟）——短等待留着更省事，长等待必须挂起；")
print("        40 分钟的 CI 等待远在阈值之上，所以题面场景的正确做法就是挂起")
```

预期输出要点（实跑）：① 挂起把沙箱数从 12,500 降到约 3,750、日成本大幅下降（等待只花存储的几分钱），**这就是「能扛住 40 分钟等待」的定量原因**；② 等 CI 比例越高挂起收益越大（即使只有 30% 在等也值得）；③ 热池大小 ≈ 到达率 × 启动时间；④ 准入控制用**精确的 Erlang C 递推**计算 M/M/c 排队：在 3,000 个服务台的规模下 ρ≤0.9 时排队几乎为零（大数定律），但**任务时长重尾与到达突发会让拐点前移**——所以运维口径仍是把稳态利用率压在 85% 以下；⑤ 恢复延迟（约 2.9 s）只占 25 分钟等待的 0.2%，**恢复 ≪ 等待**是挂起成立的前提；⑥ 存在**最小挂起阈值**，40 分钟远在其上。

## 常见追问

- **追问**：为什么不干脆给容器加长超时？
  - 要点：成本随等待时间**线性上升**，而等待期间**没有任何计算价值**。几千个 agent 各等 40 分钟，等于白付数千沙箱小时/日。挂起把这段成本从「沙箱」降到「存储」，差两三个数量级。
- **追问**：快照包含什么？会不会很大？
  - 要点：用 overlayfs 的**上层（增量）**：只包含本会话真正改动的文件；对「读取多、写入少」的 agent 任务（典型情况）增量很小；再配合对象存储去重与压缩。**不要每次全量打包整个仓库。**
- **追问**：agent 的「记忆」怎么持久化？
  - 要点：分两部分——**工作区**（文件/git 状态，走快照）与**推理状态**（对话、计划、工作集、已确认结论，走数据库/会话记录）。唤醒时先恢复推理状态，再按需恢复工作区。
- **追问**：CI 失败后怎么继续？
  - 要点：把**结构化失败信息**（失败的测试名、断言、日志片段）作为新观察喂回上下文，让 agent 继续修；同时保留「已尝试方案」集合避免重复（串 [[agents-09]]）。
- **追问**：怎么防止僵尸沙箱吃满池子？
  - 要点：**心跳 + TTL**（会话无心跳即回收）、状态机对账（定期扫描「无会话却持有沙箱」的实例）、以及回收时的强制清理；指标上监控「僵尸率」。
- **追问**：仓库热点（几千个 agent 同时 clone 同一大仓库）怎么办？
  - 要点：**共享只读基座**（同一仓库的只读层共享，每个会话只挂自己的可写上层）+ 本地镜像缓存 + clone 限速/排队；这也是 overlay 方案的自然收益。
- **追问**：多租户隔离到什么程度？
  - 要点：沙箱**不复用**（或复用前强制清洗并校验）；网络出白名单；凭证按任务临时下发；存储按租户加密与访问控制；审计留痕（串 [[cognition-08]]）。

## 相关题目

- [[cognition-04]]：harness 里的 RL——rollout 就跑在这套环境上，成本与稳定性直接决定训练可行性。
- [[cognition-08]]：拥有仓库写权限 + CI 凭证 + 网络访问的威胁模型，是本题第 6 节的展开。
- [[cognition-02]]：多 agent 与写入单线程——执行环境的并发模型要与此一致。
- [[system-design-01]]：企业级 RAG 助手（权限隔离），与本题的租户隔离同源。
- [[anthropic-17]]：过载处理与准入控制，对应本题第 5 节。

## 参考资料与归属

- **Handling Overload（Google SRE Book 第 21 章）（延伸）** —— Google SRE：<https://sre.google/sre-book/handling-overload/>。第 5 节「利用率阈值、排队与明确拒绝优于接了做不完」的过载处理纪律来自这一章。
- **Don't Build Multi-Agents（Cognition 博客）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 1 节「上下文必须共享、动作携带隐式决策」的原则用于说明为什么**唤醒后必须恢复推理状态**，否则 agent 会重做已决策的部分。
- **在生产系统中如何让 agent 的操作可逆或至少可审计（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 6 节「按任务下发临时凭证 + 全量审计」的做法取自该专题文档。
- **为数千个并发的云端 coding agent 设计执行环境（本仓库公司题库 · Anthropic 篇的相关题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 3、5 节「沙箱池化与隔离」的框架在本仓库其他公司篇中亦有交叉印证。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（10,000 并发 agent、70% 等 CI、任务 35 分钟含 25 分钟等待、沙箱 \$0.25/小时、余量 1.25、快照增量 2 GB、存储 \$0.02/GB·月、恢复各段 100/1800/200/500/300 ms、M/M/c 简化式、排队阈值 85%）都是为演示容量与成本结构而构造的**示例参数与显式假设**；真实系统必须用自己的到达率、任务时长分布、CI 等待分布与集群价格校准。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
