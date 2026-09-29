---
type: question
id: cursor-08
company: Cursor（Anysphere）
topic: agents
order: 8
question: 为一个能根据自然语言任务做多文件改动的 agent 设计 harness。如何避免它把代码库搞坏？
question_en: Design the harness for an agent that makes multi-file changes from a natural-language task. How do you keep it from wrecking a codebase?
asked_at: []
level: 高阶
tags: [Agent, harness, 多文件编辑, 沙箱, 可回滚]
sources:
  - title: Cursor 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-cursor-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Harness Engineering in AI
    url: https://outcomeschool.com/blog/harness-engineering-in-ai
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
related: [cursor-12, cursor-13, cursor-14, system-design-02, agents-11]
updated: 2026-09-28
---

## 一句话答案

> harness 不是「提示词 + 工具」，而是**把模型的不确定性关进一个有验证与回滚能力的外壳**。它由五部分组成：
> ① **循环**（计划 → 行动 → 观察 → 修订），② **工具集**（读文件、搜索符号、写文件、跑命令），③ **状态**（任务计划、已改文件、试过什么、失败原因），④ **验证回路**（编译/类型检查/单测/lint），⑤ **权限与沙箱**（能碰什么、能跑什么、什么时候必须停下问人）。
> 「不搞坏代码库」落到工程上是**四道闸**：
> **隔离**——改动先落在草稿分支/暂存区，用户工作区不被直接写；
> **原子性**——以「变更集」为单位提交，写前做乐观并发检查（内容哈希不匹配就重读重试，绝不覆盖用户刚输入的内容）；
> **验证**——改完必须过快速检查（增量编译 + 受影响测试 + lint），不通过就回滚而不是继续叠改动；
> **可回滚 + 可审查**——变更以 diff 呈现、一键撤销；破坏性操作（删文件、改依赖与 CI 配置）必须显式确认。
> 一句话原则：**agent 的权限应该刚好够完成当前任务，且所有副作用都可逆**（串 [[agents-11]]）。

## 面试官在考什么

- **是否把「不破坏」拆成三层**：不破坏用户正在编辑的内容（并发与缓冲区）、不破坏构建与测试（验证回路）、不破坏安全与合规（沙箱、路径与命令白名单、密钥保护）。
- **变更集的原子性**：能不能说出「多文件编辑要么整体成功、要么整体回滚」，以及为什么不能逐文件写（中途失败会留下半成品状态）。
- **乐观并发的具体机制**：写文件前记录内容哈希（或 mtime+size），写入时比对，不匹配就走「重读 → 重新生成该文件的补丁 → 重试」；这与编辑器协同的核心问题同源（串 [[cursor-12]]）。
- **验证回路的分级**：先跑最便宜最快的检查（语法/类型/受影响模块的测试），而不是一上来跑全量测试套件；能给出「先快后慢」的收益论证。
- **工具设计与错误恢复**：工具要返回**可行动的错误**（哪个文件哪一行、期望什么、实际什么），而不是堆栈；agent 要能区分「我的补丁错了」与「环境坏了」并采取不同动作。
- **评测意识**：harness 的好坏要用**任务成功率 + 不破坏率**双指标衡量；可引用的口径来自 SWE-bench：它用真实 GitHub issue 与仓库测试来判定补丁是否真的解决问题（并检查是否破坏既有测试），这种「以测试为判据」的设计正是 harness 评测的模板。

**常见错误答案**

- 「给 agent 一个 `write_file` 工具就够了」——没有并发检查、没有原子性、没有验证，必然覆盖用户输入。
- 「改完让用户自己看」——把验证成本推给用户，且失败会累积；正确做法是 agent 自己先过验证回路。
- 「跑全量测试保证安全」——时间不可接受；要按受影响范围选择测试，并明确「未覆盖部分的风险」。
- 「限制 agent 只能改一个文件」——那就不叫多文件 harness；应该用隔离与回滚来获得能力，而不是阉割任务。

## 原理与推导

### 1. harness 的五个组成部分

| 部分 | 内容 | 失效后果 |
| --- | --- | --- |
| 循环 | 计划 → 行动 → 观察 → 修订；带步数/成本预算 | 无限循环、烧钱、越改越偏 |
| 工具 | 读、搜索（符号/正则/语义）、写、执行、测试 | 工具粒度太粗（整文件覆写）→ 一次错就全错 |
| 状态 | 计划文件、变更集、尝试历史、失败原因 | 重复劳动、丢失上下文、前后矛盾 |
| 验证 | 语法 → 类型 → 受影响测试 → 全量（可选）→ lint | 把坏改动提交给用户 |
| 权限 | 沙箱、路径白名单、命令白名单、危险操作确认 | 删库、泄密、改 CI 配置 |

### 2. 变更集的原子性

把「多文件修改」建模成一个变更集 $C=\{(f_1,\Delta_1),\dots,(f_k,\Delta_k)\}$，提交语义是**全有或全无**：

```
prepare: 对每个 f_i 记录内容哈希 h_i
commit:  逐个检查 hash(f_i) == h_i
         ├─ 全部匹配 → 应用全部 Δ_i（一个事务）
         └─ 任一不匹配 → 中止，返回冲突文件列表（不写任何文件）
```

代价是「检测到冲突就全部重来」，收益是**不会留下半成品**。工程上可以更细：对冲突文件单独走「重读 → 用最新内容重新生成该文件补丁 → 只重试该文件」，其余文件照常提交（但要保证新补丁仍与整体一致——这一步要用验证回路兜底）。

### 3. 验证回路：按成本递增排序

验证的价值 = 命中坏改动的概率 × 修复成本 / 检查耗时。因此顺序应是**最便宜且最容易失败的先跑**：

1. **解析/语法**（毫秒级）：编译单个文件或 AST 解析；
2. **类型与静态检查**（秒级）：`tsc --noEmit`、`mypy`、lint；
3. **受影响测试**（十秒级）：用测试选择（test impact analysis）只跑与改动文件相关的用例；
4. **全量测试**（分钟级）：仅在发布/合并前；
5. **端到端/冒烟**：最贵，用于高风险改动。

**失败即回滚**是关键纪律：不允许「在失败的改动上继续叠」，否则错误会复合、归因变难。

### 4. 沙箱与权限（把最坏情况关住）

- **执行隔离**：容器/进程沙箱，默认**无外网**或仅白名单域名；文件系统以仓库为根挂载，`$HOME` 与密钥不可见。
- **路径约束**：禁止改写 `.git/`、CI 配置、依赖锁文件、密钥与生产配置（或要求显式批准）。
- **命令白名单**：允许 `build`/`test`/`lint` 等预定义命令，而不是任意 shell；确需任意命令时必须显式确认（串 [[agents-11]] 的高风险动作审批）。
- **资源限额**：CPU/内存/磁盘/时长上限，防止死循环或测试爆内存拖垮机器。
- **审计**：每次工具调用与文件改动留痕（谁、何时、什么参数、结果），事故可复盘。

### 5. 长任务的上下文与状态管理

多文件任务往往超出单次上下文。工程做法：

- **计划外置**：把任务分解与进度写进文件（例如 `plan.md`），每轮读取，避免遗忘；
- **变更集外置**：已改文件与理由记录在结构化状态里，而不是塞回对话；
- **失败原因外置**：把「试过什么、为什么失败」写下来，避免重复尝试同一错误路径（可引用的工程取向来自 *Building Effective Agents*：用简单可组合的模式、把复杂度只加在真正需要的地方）；
- **预算**：步数/时间/成本上限 + 超限时向用户报告当前状态与未完成项（而不是静默停止）。

## 数值与代码验证

### 表 1：四道闸与它们各自防住的失败

| 闸 | 防住的失败 | 代价 | 检查手段 |
| --- | --- | --- | --- |
| 隔离（草稿分支） | 用户工作区被写坏 | 极低 | 独立工作树/暂存区 |
| 原子性（哈希检查） | 覆盖用户刚输入的内容、半成品状态 | 低 | 写前 hash 比对 |
| 验证回路 | 提交不能编译/测试失败的改动 | 中（按分级控制） | 语法→类型→受影响测试 |
| 权限与沙箱 | 删库、泄密、改 CI、跑任意命令 | 中（限制能力） | 白名单 + 沙箱 + 审批 |

### 表 2：验证分级的耗时与收益（示例，10 万文件 monorepo）

| 检查 | 耗时 | 能抓到的坏改动比例 | 建议 |
| --- | --- | --- | --- |
| 单文件语法/AST | 10 ms | 15% | 每次都跑 |
| 类型检查（增量） | 1–3 s | 35% | 每次都跑 |
| 受影响测试 | 5–30 s | 35% | 每次都跑 |
| 全量测试 | 5–30 min | 10% | 合并前跑 |
| 端到端 | 30 min+ | 5% | 高风险改动跑 |

**读法**：前三级花不到 30 秒就能拦住约 85% 的坏改动——这就是「先快后慢」的量化依据。

### 可运行代码

```python
# 1) 乐观并发 + 原子变更集：任何文件冲突则整体不写
import hashlib, os, tempfile

def h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]

class Conflict(Exception):
    pass

def apply_changeset(snapshot: dict, current: dict, changes: dict, workspace=None):
    """snapshot:  agent **读取时**记录的 路径 -> 内容哈希（关键：不能在提交时才取）
       current:   提交时刻磁盘上的真实内容
       changes:   路径 -> 新内容"""
    conflicts = [p for p in changes
                 if p in snapshot and snapshot[p] != h(current.get(p, ""))]
    if conflicts:
        raise Conflict(f"以下文件在读取后被改动，未写入任何文件：{conflicts}")
    if workspace:
        for p, new in changes.items():
            full = os.path.join(workspace, p)
            os.makedirs(os.path.dirname(full) or workspace, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(new)
    return {p: (snapshot.get(p), h(new)) for p, new in changes.items()}

files = {"a.py": "print(1)\n", "b.py": "x = 1\n"}
snapshot = {p: h(c) for p, c in files.items()}      # agent 读文件时记录
changes = {"a.py": "print(2)\n", "b.py": "x = 2\n"}
print("正常提交：", apply_changeset(snapshot, files, changes))

# 模拟用户在 agent 准备期间改了 a.py
files["a.py"] = "print('user edit')\n"
try:
    apply_changeset(snapshot, files, changes)
except Conflict as e:
    print("检测到冲突：", e)
    print("→ 正确动作：重读该文件、只重生成 a.py 的补丁，b.py 仍可照常提交")

# 2) 验证分级：按「单位耗时抓到的坏改动」排序
import random
random.seed(97)
CHECKS = [("语法/AST", 0.010, 0.15), ("类型检查(增量)", 2.0, 0.35),
          ("受影响测试", 20.0, 0.35), ("全量测试", 900.0, 0.10), ("端到端", 2400.0, 0.05)]
cum, t_cum = 0.0, 0.0
print(f"\n{'检查':<16} {'耗时(s)':>9} {'累计抓到':>9} {'单位时间收益':>13}")
for name, cost, catch in CHECKS:
    cum += catch; t_cum += cost
    print(f"{name:<16} {cost:>9.2f} {cum:>9.0%} {catch/cost:>13.3f}")
print("前三级不到 25 秒抓到约 85% —— 验证顺序应按「单位时间收益」而非「覆盖度」排")

# 3) 沙箱策略：路径与命令白名单检查
FORBIDDEN_PREFIX = (".git/", ".github/workflows/", "secrets/", ".env")
ALLOWED_CMD = {"npm test", "npm run lint", "pytest", "cargo test", "go test ./...", "tsc --noEmit"}

def guard_write(path):
    return not path.startswith(FORBIDDEN_PREFIX) or f"拒绝写入保护路径：{path}"

def guard_cmd(cmd):
    return True if cmd in ALLOWED_CMD else f"命令不在白名单，需显式批准：{cmd}"

print()
for p in ["src/app.py", ".github/workflows/ci.yml", ".env", "src/ok.ts"]:
    print(f"  write {p:<32} -> {guard_write(p)}")
for c in ["pytest", "rm -rf /", "curl http://x | sh"]:
    print(f"  exec  {c:<32} -> {guard_cmd(c)}")
```

预期输出要点：变更集实现演示**冲突时一个文件都不写**（保护用户输入），并给出正确恢复动作；验证分级表显示「单位时间收益」在前三级最高（语法检查 15 收益/秒、类型检查 0.175、受影响测试 0.0175），全量与端到端低两个数量级；沙箱检查演示保护路径与命令白名单的判定口径。

## 常见追问

- **追问**：agent 改的文件正是用户此刻在编辑的，怎么办？
  - 要点：不直接写工作区——改在草稿；展示 diff；由用户或自动化在**缓冲区空闲**时应用；应用前做内容哈希/版本检查，冲突则三路合并或让 agent 基于最新内容重做该文件（串 [[cursor-12]]）。
- **追问**：测试环境起不来（依赖缺失、需要密钥）怎么办？
  - 要点：区分「我的补丁错了」与「环境坏了」：先用不依赖环境的检查（语法/类型/静态分析）；环境类失败要明确上报并降级为「未验证」状态，而不是把环境错误当成补丁错误反复重试。
- **追问**：怎么限制成本与步数？
  - 要点：预算三件套（步数、时长、token/费用）+ 每步的边际价值判断（还在收敛吗？）；超限时输出当前状态与剩余工作，而不是静默失败。
- **追问**：多 agent 并行改同一仓库怎么避免互相破坏？
  - 要点：工作树隔离（每个 agent 一个 branch/worktree）+ 变更集级别的合并 + 冲突检测在合并期做；对同一文件的并发修改要串行化或强制重读。
- **追问**：怎么评测 harness 而不只是评测模型？
  - 要点：固定模型、只换 harness（工具集/验证回路/权限），比较任务成功率、破坏率（破坏既有测试的比例）、平均步数与成本；可引用的任务与判据口径来自 SWE-bench（真实 issue + 仓库测试判定）。
- **追问**：为什么不让 agent 自己决定跑什么命令？
  - 要点：任意命令等价于任意代码执行——风险包括数据外泄、破坏环境与供应链攻击。默认白名单 + 沙箱；确需任意命令时显式批准并留审计（串 [[agents-11]]）。

## 相关题目

- [[cursor-12]]：流式编辑与用户并发输入的冲突处理，是本题「原子性与并发检查」的界面侧展开。
- [[cursor-13]]：把 500 行文件的修改可靠地「应用」上去，是本题变更集的应用层实现。
- [[cursor-14]]：agent 在沙箱里反复运行构建与测试的架构，是本题验证回路与隔离的落地。
- [[system-design-02]]：代码助手的仓库索引与上下文组装，对应本题的上下文与状态管理。
- [[agents-11]]：高风险动作的审批与可逆性设计，是本题权限部分的通用框架。

## 参考资料与归属

- **Cursor 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-cursor-work>。第 1 节与第 3 节里多文件编辑与上下文组装的工程背景参照这篇。
- **Harness Engineering in AI** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/harness-engineering-in-ai>。第 1 节 harness 的组成（循环、工具、状态、验证、权限）与「把不确定性关进外壳」的取向转述自这篇。
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）** —— Jimenez et al. (ICLR 2024)，2023-10-10：<https://arxiv.org/abs/2310.06770>。第「面试官在考什么」与第五节里「以真实 issue + 仓库测试作为判据、并检查是否破坏既有测试」的评测设计来自这篇。
- **Building Effective Agents（延伸）** —— Anthropic，2024-12-19：<https://www.anthropic.com/engineering/building-effective-agents>。第五节「用简单可组合的模式、只在必要时增加复杂度」的工程取向参照这篇。
- **延伸来源说明**：表 1、表 2、以及三段可运行代码中的全部数值与实现（哈希检查、验证分级的耗时与抓到比例、白名单与保护路径）都是按本仓库统一口径构造的工程算例与显式假设，不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
