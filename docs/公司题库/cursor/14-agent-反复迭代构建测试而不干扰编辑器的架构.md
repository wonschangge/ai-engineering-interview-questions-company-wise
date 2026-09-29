---
type: question
id: cursor-14
company: Cursor（Anysphere）
topic: system-design
order: 14
question: 一个 agent 需要在代码上反复迭代（运行构建、测试、lint），同时不干扰用户在编辑器里看到的内容。请设计这样的架构。
question_en: An agent needs to iterate on code (run builds, tests, lint) without disturbing what the user sees in the editor. Design that architecture.
asked_at: []
level: 高阶
tags: [系统设计, 沙箱, 工作区隔离, 构建缓存, 迭代循环]
sources:
  - title: Harness Engineering in AI
    url: https://outcomeschool.com/blog/harness-engineering-in-ai
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Claude Code 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-claude-code-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）
    url: https://arxiv.org/abs/2310.06770
    author: Jimenez et al. (ICLR 2024)
    published: 2023-10-10
  - title: Building Effective Agents（延伸）
    url: https://www.anthropic.com/engineering/building-effective-agents
    author: Anthropic
    published: 2024-12-19
related: [cursor-08, cursor-12, cursor-13, cursor-15, system-design-02]
updated: 2026-09-28
---

## 一句话答案

> 架构的核心是**双工作区 + 单向同步**：agent 在**隔离工作区**里改代码、跑构建与测试；用户编辑器看到的是**自己的工作区**，二者只通过「变更集」发生关系（agent 产出 → 差异审查 → 原子应用，串 [[cursor-13]]）。
> 具体四块：
> ① **隔离层**——用 `git worktree`/目录副本/容器把 agent 的工作区与用户的工作区分开（提供文件系统级隔离）；容器再加网络与资源隔离。
> ② **执行层**——沙箱容器里跑构建/测试/lint，带**依赖与构建缓存**（否则每次迭代都要重装依赖、全量编译，迭代周期从秒级退化到分钟级）、资源限额（CPU/内存/磁盘/时长）、端口隔离（不占用用户的 dev server 端口）。
> ③ **同步层**——agent 的中间态**不写用户工作区**；完成后生成变更集，用户审阅后原子应用；也支持「实时跟随」模式（用户能看到 agent 在做什么），但仍然是**只读预览**，不是直接改用户的缓冲区。
> ④ **反馈层**——把构建/测试输出结构化成 agent 可用的信号（哪个文件哪一行、期望 vs 实际），并驱动下一轮迭代；失败的中间态可丢弃（容器是「用完即弃」的）。
> 一句话原则：**agent 的时间和用户的时间必须解耦**——它可以慢，但不能让用户等、不能让用户看到半成品、不能占用用户的资源与端口。

## 面试官在考什么

- **隔离层次的取舍**：能否比较「同目录 + stash」「git worktree」「目录副本」「容器」「远程沙箱」在启动时间、隔离强度、成本与可用性上的差异，并按场景选择。
- **「不干扰」的三层含义**：① 视图（用户不看到中间态）；② 资源（CPU/内存/磁盘/端口不与用户争抢）；③ 状态（不污染用户的 git 状态、不改变其未提交修改）。
- **迭代速度的工程手段**：增量编译、测试选择（只跑受影响的测试）、依赖与构建缓存复用、并行执行——能否给出「不做这些会慢多少」的量级。
- **同步语义**：变更集的原子性与冲突处理（用户可能同时在改同一文件），以及「何时把结果给用户」（完成即给、关键里程碑给、还是实时跟随）。
- **失败与清理**：容器崩溃/超时/失控循环的处理（幂等重试、硬超时、资源回收），避免「沙箱泄漏」（僵尸容器占用磁盘）。
- **与评测的连接**：这种架构正是 SWE-bench 式评测的运行环境——在隔离副本里跑真实测试来判定补丁是否解决问题（并且要检查是否破坏既有测试）。

**常见错误答案**

- 「让 agent 直接在用户工作区改，改完再撤销」——用户会看到抖动、git 状态被污染、失败时留下半成品。
- 「每次迭代重新拉镜像装依赖」——迭代周期从秒级变成分钟级，agent 的成本与体验都崩。
- 「把构建产物和用户共享」——缓存冲突、端口冲突、资源争抢。
- 「不设超时与配额」——一个死循环的测试就能把机器打满，用户编辑器卡死。

## 原理与推导

### 1. 隔离方案对比

| 方案 | 启动时间 | 隔离强度 | 成本 | 适用 |
| --- | --- | --- | --- | --- |
| 同目录 + stash | 0 | 无（污染用户状态） | 0 | 不推荐 |
| `git worktree` | ~100 ms | 文件系统级（独立工作树、共享对象库） | 低 | 本地 agent、轻量迭代 |
| 目录副本 | 秒级（大仓库更慢） | 文件系统级 | 低（磁盘×2） | 简单场景 |
| 本地容器 | 1–10 s（镜像预热后 <1 s） | 文件系统 + 进程 + 网络 | 中 | 需要跑不可信命令 |
| 远程沙箱/微虚机 | 秒级（池化） | 最强（内核级） | 高（算力成本） | 多租户、强隔离要求 |

**关键工程点**：容器**池化预热**（保持若干空闲容器，命中即用）把启动时间从秒级降到百毫秒级，这是体验的分水岭。

### 2. 迭代循环与其加速

一次迭代 = 编辑 → 构建 → 测试 → lint → 反馈 → 再编辑。若不加速：

| 环节 | 未优化 | 优化后 | 手段 |
| --- | --- | --- | --- |
| 依赖安装 | 60–300 s | 0（命中缓存） | 依赖缓存层（按 lockfile 哈希） |
| 构建 | 120 s（全量） | 3–10 s（增量） | 增量编译 + 远程/本地构建缓存 |
| 测试 | 300 s（全量） | 5–30 s（受影响测试） | 测试选择（test impact analysis） |
| lint/类型检查 | 30 s（全量） | 1–3 s（增量） | 语言服务的增量模式 |
| 合计 | **8–13 min** | **10–45 s** | — |

**量级差异**：同样的 agent 步数预算，优化后能迭代 10–20 次而不是 1–2 次——这直接决定任务成功率。

### 3. 同步策略：三种模式

| 模式 | 用户看到什么 | 优点 | 风险 |
| --- | --- | --- | --- |
| 完成后交付（默认） | 最终 diff | 不干扰、可审查 | 反馈慢 |
| 里程碑交付 | 每完成一个子任务给一次 diff | 平衡 | 需要清晰的任务分解 |
| 实时跟随（只读预览） | agent 正在改的文件的**只读视图** | 透明、可中断 | 实现复杂；不能是用户缓冲区本身 |

**实时跟随的实现要点**：预览必须是**独立视图**（例如虚拟文档/只读标签页），不能把 agent 的中间态写进用户的真实缓冲区——否则就是串 [[cursor-12]] 描述的并发破坏。

### 4. 资源与安全隔离

- **配额**：CPU（核数）、内存、磁盘（含构建产物上限）、单次命令时长、总任务时长；超限即杀并回收。
- **端口**：容器内端口空间独立，避免与用户 dev server 冲突；需要暴露时用随机高位端口。
- **网络**：默认**无外网**或白名单（包仓库、文档站）；确需私有源则用凭据代理，绝不把用户凭据注入沙箱。
- **文件系统**：只挂载工作区副本；用户主目录、SSH 密钥、云凭据不可见。
- **清理**：任务结束/超时/崩溃后强制回收容器与临时卷；磁盘水位监控（僵尸沙箱是常见事故源）。

### 5. 与用户编辑的并发

agent 的基线是**启动时的快照**；用户可能同时在改。合并策略：

1. 变更集应用前做快照检查（内容哈希）；
2. 不冲突 → 直接应用；
3. 冲突 → 三方合并，或把冲突文件交给 agent 基于**最新内容**重做（只重做冲突文件）；
4. 全程不覆盖用户输入（同 [[cursor-12]] 的三类冲突处理）。

## 数值与代码验证

### 表 1：迭代周期对比（含 agent 侧推理时间）

| 配置 | 构建+测试 | LLM 推理/步 | 每步总耗时 | 20 步任务总时长 |
| --- | --- | --- | --- | --- |
| 无缓存、全量 | 10 min | 10 s | ~10.2 min | **~3.4 h** |
| 依赖缓存 + 增量构建 | 20 s | 10 s | 30 s | ~10 min |
| 再叠加测试选择 | 8 s | 10 s | 18 s | ~6 min |

读法：**同一任务、同一模型**，仅靠构建/测试侧优化，总时长差 30 倍以上——这解释了为什么「sandbox 工程」是 agent 产品的核心竞争力之一。

### 表 2：资源限额示例

| 资源 | 限额 | 超限动作 |
| --- | --- | --- |
| CPU | 4 核 | 限流（不影响用户） |
| 内存 | 8 GiB | OOM 杀进程 + 上报 |
| 磁盘 | 5 GiB（含构建缓存配额） | 清理缓存或终止 |
| 单命令时长 | 10 min | 硬超时 + 输出部分日志 |
| 总任务时长 | 60 min | 停止并交付中间成果 |
| 网络 | 白名单 | 拒绝并记录 |

### 可运行代码

```python
# 1) 迭代周期的时间账：缓存/增量/测试选择各自贡献多少
def iteration_time(deps_cached, incremental_build, test_selection, llm_s=10.0):
    deps   = 0.0 if deps_cached else 180.0
    build  = 5.0 if incremental_build else 120.0
    tests  = 10.0 if test_selection else 300.0
    lint   = 2.0
    return deps + build + tests + lint + llm_s

configs = [
    ("全量（无缓存）",        dict(deps_cached=False, incremental_build=False, test_selection=False)),
    ("仅依赖缓存",            dict(deps_cached=True,  incremental_build=False, test_selection=False)),
    ("依赖缓存 + 增量构建",    dict(deps_cached=True,  incremental_build=True,  test_selection=False)),
    ("三件套（+ 测试选择）",   dict(deps_cached=True,  incremental_build=True,  test_selection=True)),
]
print(f"{'配置':<24} {'每步(s)':>9} {'20 步总时长':>13}")
for name, kw in configs:
    t = iteration_time(**kw)
    print(f"{name:<24} {t:>9.1f} {t*20/60:>12.1f} min")

# 2) 沙箱池化：预热容器把启动时间降到百毫秒级
import random
random.seed(127)
def cold_start():  return random.gauss(6.0, 1.5)      # 冷启动：拉镜像/初始化
def warm_start():  return max(0.05, random.gauss(0.25, 0.08))
N = 5000
cold = [max(0.3, cold_start()) for _ in range(N)]
warm = [warm_start() for _ in range(N)]
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))]
print(f"\n沙箱启动：冷 {pct(cold,0.5):.2f}s / p95 {pct(cold,0.95):.2f}s   "
      f"池化 {pct(warm,0.5):.2f}s / p95 {pct(warm,0.95):.2f}s")
print("每次迭代都要启动沙箱时，这段时间会被乘以步数 —— 池化是体验的分水岭")

# 3) 变更集应用：用户并发改动时的合并决策（与 cursor-13 同一套判定）
import hashlib
def h(s): return hashlib.sha256(s.encode()).hexdigest()[:12]
def sync(snapshot, user_now, agent_new, path):
    if h(user_now) == snapshot[path]:
        return "直接应用"
    if user_now == agent_new:
        return "内容一致，无需应用"
    return "冲突 → 只重做冲突文件（以最新内容为基线）"
snapshot = {"a.py": h("x=1\n"), "b.py": h("y=1\n")}
print()
print(" 用户未改动 ->", sync(snapshot, "x=1\n", "x=2\n", "a.py"))
print(" 用户已改动 ->", sync(snapshot, "x=99\n", "x=2\n", "a.py"))
```

预期输出要点：迭代周期从「全量」的每步十几分钟降到三件套的约 17 秒（20 步任务从 3.4 小时降到 6 分钟）；沙箱冷启动 p95 与池化后对比说明为什么要预热池；同步判定演示用户并发改动时的三种结论。

## 常见追问

- **追问**：依赖安装很慢（私有源、编译型依赖）怎么办？
  - 要点：把依赖层做成**缓存镜像/卷**（按 lockfile 哈希索引）、预热常用依赖集合、以及**复用上一次成功的环境**（容器保留 + 增量更新依赖）。
- **追问**：沙箱需要访问私有包仓库怎么办？
  - 要点：通过代理注入短期凭据（沙箱内不落地长期凭据）、限制目标域名、并审计请求；绝不能把用户凭据直接注入。
- **追问**：如何防止沙箱逃逸？
  - 要点：用内核级隔离（微虚机/gVisor 类方案）、非 root 运行、seccomp/AppArmor、只读根文件系统 + 可写工作卷、禁用特权设备；定期更新运行时补丁。
- **追问**：用户希望实时看到 agent 在做什么，怎么实现而不干扰？
  - 要点：只读预览视图（虚拟文档）+ 可中断；真实缓冲区只在用户确认或 agent 完成后由变更集原子应用。
- **追问**：agent 看到的是最新代码吗？
  - 要点：基线是启动快照；提供「同步」动作在检测到用户改动时刷新工作区，并在应用阶段做冲突合并（否则会出现基于旧代码的改动）。
- **追问**：怎么衡量这套架构的效果？
  - 要点：迭代周期（每步耗时）、任务成功率与步数、沙箱启动 p95、资源利用率、以及事故类指标（僵尸容器、端口冲突、磁盘打满）。

## 相关题目

- [[cursor-08]]：harness 的隔离/验证/回滚，本题是其执行环境层面的展开。
- [[cursor-12]]：并发写入与冲突处理，本题的同步层复用同一套判定。
- [[cursor-13]]：补丁应用的可靠性，决定变更集能否安全落地。
- [[cursor-15]]：评估方案，需要这种隔离环境来跑真实测试判定。
- [[system-design-02]]：代码助手的整体设计，本题是其「沙箱执行」子系统的细节。

## 参考资料与归属

- **Harness Engineering in AI** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/harness-engineering-in-ai>。第 1 节与第 4 节 harness 的执行与验证环节转述自这篇。
- **Claude Code 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-claude-code-work>。第 3 节「工具化执行与隔离」的工程取向参照这篇。
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）** —— Jimenez et al. (ICLR 2024)，2023-10-10：<https://arxiv.org/abs/2310.06770>。第「面试官在考什么」里「在隔离副本里跑真实测试判定补丁」的评测环境设计来自这篇。
- **Building Effective Agents（延伸）** —— Anthropic，2024-12-19：<https://www.anthropic.com/engineering/building-effective-agents>。第 1 节「只增加必要的复杂度」与工具设计的取向参照这篇。
- **延伸来源说明**：表 1、表 2、以及三段可运行代码中的全部数值（依赖 180 s、全量构建 120 s、全量测试 300 s、LLM 每步 10 s、冷启动 6 s vs 池化 0.25 s、资源限额取值）都是按本仓库统一口径构造的工程算例与显式假设，不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
