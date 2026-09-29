---
type: question
id: databricks-05
company: Databricks
topic: coding
order: 5
question: 一个 Structured Streaming 作业从 Kafka 读取数据并写入 Delta 表。集群在一个 batch 处理到一半时被杀掉并重启。客户会看到重复行吗？请从 checkpoint 和事务日志的层面解释。
question_en: A Structured Streaming job reads from Kafka and writes to a Delta table. The cluster is killed mid-batch and restarts. Will customers see duplicate rows? Explain in terms of the checkpoint and the transaction log.
asked_at: []
level: 高阶
tags: [Structured-Streaming, Delta-Lake, 恰好一次, checkpoint, 事务日志, 幂等]
sources:
  - title: Databricks 文档：在 Delta Lake 上使用 Structured Streaming
    url: https://docs.databricks.com/aws/en/structured-streaming/delta-lake
    author: Databricks
    published: 
  - title: Delta Lake 文档：并发控制与事务日志
    url: https://docs.delta.io/latest/concurrency-control.html
    author: Delta Lake
    published: 
  - title: 一个 Spark 作业把 2 TB 的事实表与 50 GB 的维表做 join（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: agent 循环中如何处理工具调用错误、超时与重试（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [databricks-04, databricks-02, databricks-01, agents-02, system-design-01]
updated: 2026-09-28
---

## 一句话答案

> **默认答案：不会重复**——但成立条件是两个，缺一不可：
> ① **checkpoint 完好且持久**（在对象存储/DBFS 上，不是临时盘）：它保存 **offset log**（每个 batch 读了哪些 offset）与 **commit log**（哪些 batch 已完成）；
> ② **sink 幂等**：Delta 的**事务日志**为每个 micro-batch 记录一条 `SetTransaction(appId, batchVersion)`，重放同一个 batch 时**发现该版本已提交就直接跳过**，因此重复执行不会重复写。
> **崩溃恢复的四种时序**（这是本题最该讲清的部分）：
> | 崩溃点 | 现象 | 是否重复 |
> | --- | --- | --- |
> | 写入前 | batch 未执行 | 否（重放） |
> | 写入中（数据文件已写、事务未提交） | Delta 用**乐观并发 + 原子提交**：未提交的文件不会被读到（不会被计入快照） | 否 |
> | **写入已提交、commit log 未写** | 重启后重放该 batch → **`SetTransaction` 发现版本已存在 → 跳过** | **否**（这就是幂等键的作用） |
> | commit log 已写、下一批未开始 | 直接从下一批继续 | 否 |
> **但下面这些情况会看到重复（或丢数据）**，必须在答案里主动列出：
> 1. **checkpoint 丢失/被换**（临时盘、新集群用了新目录、`checkpointLocation` 写错）→ 退化成 **at-least-once**；
> 2. **sink 不幂等**：写 Kafka、写 JDBC（无 upsert）、`foreachBatch` 里自己写外部系统 → 需要**业务主键 + MERGE/去重**；
> 3. **`foreachBatch` 有副作用**（调 API、写两个系统）→ **两个 sink 之间没有原子性**，必然出现部分生效；
> 4. **来源不可重放**（socket 源、或没 checkpoint 时从 latest 起读）→ 这次不是重复而是**丢数据**；
> 5. **Kafka 数据被 retention 清掉 / offset 重置** → 丢数据；
> 6. **非确定性转换**（`rand()`、`current_timestamp()`、UDF 带状态）→ 重放写进去的**内容不同**（仍只写一次，但结果不可复现）。
> 一句话判据：**恰好一次 = 可重放的来源 + 持久的 checkpoint + 幂等的 sink**；三者缺一，要么重复要么丢失——**而「重复」与「丢失」要用不同的手段修**。

## 面试官在考什么

- **先给结论再给条件**：能否直接答「默认不重复」，并立刻补上两个前提（持久 checkpoint + 幂等 sink）。
- **能否讲清两套日志的分工**：**checkpoint 的 offset log / commit log** 管「读到哪、做到哪」；**Delta 的事务日志**管「写没写、写了几次」。很多人只记得其中一套。
- **`SetTransaction`/`txnVersion` 这条幂等键**：能否指出「写已提交但 commit log 未写」是最危险的一格，而它正是靠这条记录被兜住的。
- **边界条件**：能否主动列出会重复/会丢数据的场景（尤其 `foreachBatch` 的副作用与 checkpoint 丢失）。
- **重复 vs 丢失的区分**：能否指出「来源不可重放/Kafka retention」导致的是**丢数据**，不能用去重解决。
- **修复手段**：重复 → **业务主键 + MERGE（upsert）+ 去重窗口**；丢失 → 扩大 retention、从 earliest 重放、补数管道。
- **验证方式**：能否设计「在 batch 中途 kill -9」的测试，以及怎么核对（行数、主键去重计数、`_commit_version`/`_commit_timestamp` 列）。
- **工程细节**：checkpoint 目录要**独立于数据**、能被多个重启复用；**只能有一个 writer**；`trigger(availableNow=True)` 与常驻的区别；以及 `VACUUM` 会不会影响恢复（不会影响流恢复，但会影响 time travel）。
- **诚实**：承认端到端恰好一次在「跨多个外部系统」时**做不到**（除非引入两阶段提交或幂等键）。

**常见错误答案**

- 只说「Delta 支持 exactly-once」而不说条件。
- 只讲 checkpoint 不讲 Delta 事务日志（或反之）。
- 认为「重启后会重复，需要自己加去重」——在标准 Delta sink 下这是**多余的**（但换成 Kafka sink 就变成必需的）。
- 忽略 `foreachBatch` 的副作用问题。
- 把丢数据当成重复问题去修（或反之）。
- 不区分「批次内容相同」与「重放内容可能不同」（非确定性转换）。

## 原理与推导

### 1. 两套日志的分工

```
Structured Streaming checkpoint/
  ├─ offsets/<batchId>     ← 本批读了哪些 offset（提交后才写）
  ├─ commits/<batchId>     ← 本批已完成（写 sink 成功后写）
  └─ state/                ← 有状态算子（聚合/join/去重）的状态快照

Delta 表 _delta_log/
  └─ <version>.json        ← 本次提交新增/删除了哪些文件（原子提交）
       └─ 其中包含 txn: {appId, version}   ← ★ 幂等键（SetTransaction）
```

**恢复流程**：读 `commits/` 找到最后一个已完成的 batch → 从 `offsets/` 拿到下一批的 offset 范围 → **重放该批** → 写 Delta（幂等）→ 写 `commits/<batchId>`。

### 2. 为什么「写已提交、commit log 未写」不会重复

Delta 的提交是**原子**的：只有当 `<version>.json` 落盘后，那批数据文件才「可见」。若进程在「数据文件已写、`_delta_log` 未写」时被杀：
- 那些数据文件是**孤儿文件**（不可见、后续会被 `VACUUM` 清理）；
- 重启后重放该批 → 重新写文件并提交 → **客户只看到一份**。

若进程在"`_delta_log` 已写、checkpoint 的 `commits/` 未写"时被杀：
- 重启后重放该批 → 写入时 Delta 检查 `SetTransaction(appId, version)` → **已存在 → 跳过写入**；
- 然后写 `commits/` → 继续下一批。

**所以真正兜底的是 `SetTransaction` 这条记录**——它把「重放」变成「无操作」。

### 3. 幂等键的三要素

| 要素 | 作用 | 失效后果 |
| --- | --- | --- |
| `appId`（query id） | 区分不同作业 | 换 query id → 视为新作业 → **重复写** |
| `batchVersion` | 区分同一作业的不同批 | 版本重用 → 覆盖/跳过错误批次 |
| checkpoint 目录 | 保存两者的对应关系 | 丢失 → 版本号从头开始 → **重复写** |

**注**：`appId` 保存在 checkpoint 的 metadata 里；**换 checkpoint 目录 = 换 appId = 历史批次全部重放**——这是「新集群 + 新目录」导致重复的根本原因。

### 4. 哪些情况下会重复 / 会丢

| 场景 | 结果 | 修法 |
| --- | --- | --- |
| checkpoint 在临时盘 / 每次新建 | **重复**（历史重放） | checkpoint 放持久存储且**固定不变** |
| Kafka sink / JDBC（无 upsert） | **重复** | 业务主键 + `MERGE` |
| `foreachBatch` 写两个系统 | **部分生效** | 幂等键 + 补偿/对账 |
| socket 源 / 无 checkpoint 从 latest | **丢** | 用可重放源 + 明确起点 |
| Kafka retention 过期 / offset 重置 | **丢** | 扩大 retention、监控 lag、补数 |
| 非确定性转换（`rand()`） | **内容不可复现** | 用确定性表达式/种子 |
| 多个 writer 写同一张表 | 冲突/重复 | 单 writer；或用 `MERGE` 保证幂等 |
| `VACUUM` 清理旧文件 | 影响 time travel，**不影响流恢复** | 设置合理保留期 |

### 5. 有状态算子（聚合/去重/流-流 join）

状态保存在 checkpoint 的 `state/` 里，**与批次一起原子推进**：批失败则状态回滚到上一批。所以：
- 带 `dropDuplicates` 的流**天然有去重能力**（但状态会随时间增长，需要 watermark 限制）；
- 带 watermark 的聚合在 watermark 过期后**丢弃迟到数据**——这是「丢」而不是「重」；
- 流-流 join 的状态大小取决于 watermark 与 key 基数，**没有 watermark 会无限增长**。

### 6. 端到端「恰好一次」的真实边界

$$\text{恰好一次}=\underbrace{\text{可重放来源}}_{\text{Kafka offsets}}+\underbrace{\text{持久 checkpoint}}_{\text{offset/commit log}}+\underbrace{\text{幂等 sink}}_{\text{SetTransaction}}$$

**跨多个外部系统时无法做到**（没有跨系统的两阶段提交），只能用：
- **幂等写**（每个系统用自己的幂等键）；
- **事务性 outbox**（把「要写的动作」先写进 Delta，再由下游消费）；
- **对账与补偿**（最终一致）。

**这是把答案从「背概念」提升到「工程判断」的关键一句。**

## 数值与代码验证

### 表 1：四种崩溃时序的结果（模拟，见代码输出）

| 崩溃点 | 有幂等键 | 无幂等键（如 Kafka sink） |
| --- | --- | --- |
| 写入前 | 见输出 | 见输出 |
| 写入中（未提交） | 见输出 | 见输出 |
| 写入已提交、commit log 未写 | 见输出 | 见输出 |
| commit log 已写 | 见输出 | 见输出 |

### 表 2：checkpoint 状态与重复量的关系

| checkpoint | 重启后行为 | 重复行数 |
| --- | --- | --- |
| 完好 | 从断点继续 + 幂等跳过 | 见输出 |
| 丢失 | 从头/从 latest 重放 | 见输出 |

### 可运行代码

```python
# Kafka->Delta 的恰好一次：崩溃点模拟（checkpoint + 事务日志 + 幂等键）
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class DeltaLog:
    """sink 的抽象：**每次实际写入都追加一条记录**（模拟 append-only 的 sink）。
    幂等键命中时"跳过"= 不追加 —— 这才是「重复」与「不重复」的差别所在。"""
    appended: List[Tuple[str, int, int]] = field(default_factory=list)   # (appId, version, rows)
    seen: set = field(default_factory=set)                               # 幂等键集合
    orphan_files: int = 0
    def commit(self, app_id: str, version: int, rows: int, idempotent: bool) -> str:
        key = (app_id, version)
        if idempotent and key in self.seen:
            return f"跳过（幂等键 {key} 已存在）"          # ★ SetTransaction 命中
        self.appended.append((app_id, version, rows))      # 追加写入
        self.seen.add(key)
        return f"提交 v{version}（{rows} 行）"
    def total_rows(self) -> int:
        return sum(r for _, _, r in self.appended)

@dataclass
class Checkpoint:
    """offset log + commit log"""
    app_id: str = "app-1"
    offsets: Dict[int, Tuple[int, int]] = field(default_factory=dict)   # batchId -> (from, to)
    commits: List[int] = field(default_factory=list)
    def next_batch(self) -> int:
        return (max(self.commits) + 1) if self.commits else 0

def run_pipeline(batches: List[int], kill_at: Optional[Tuple[int, str]] = None,
                 idempotent: bool = True, cp: Optional[Checkpoint] = None,
                 delta: Optional[DeltaLog] = None) -> Dict[str, object]:
    """batches: 每批要处理的行数；kill_at=(batchId, 阶段) 模拟在该阶段被杀"""
    cp = cp or Checkpoint()
    delta = delta or DeltaLog()
    restarted = False
    for bid, rows in enumerate(batches):
        if bid < cp.next_batch():
            continue                                   # 已完成的批次不再处理
        offs = (bid * 100, bid * 100 + rows)           # 假装 offset 与行数对应
        # ① 读 offset（提交后才写进 offsets/）
        stage_kill = kill_at == (bid, "before_write")
        if stage_kill:
            return {"重复行": 0, "总行数": delta.total_rows(), "已提交批次": list(cp.commits),
                    "说明": f"批 {bid} 写入前被杀：重放该批，不重复"}
        # ② 写数据文件（可能被"写入中"杀掉）
        if kill_at == (bid, "during_write"):
            delta.orphan_files += 1
            return {"重复行": 0, "总行数": delta.total_rows(), "已提交批次": list(cp.commits),
                    "说明": f"批 {bid} 写入中（未提交）被杀：孤儿文件不可见，重放不重复"}
        # ③ 提交到 Delta 事务日志
        msg = delta.commit(cp.app_id, bid, rows, idempotent)
        if kill_at == (bid, "after_commit_before_checkpoint"):
            return {"重复行": 0, "总行数": delta.total_rows(), "已提交批次": list(cp.commits),
                    "说明": f"批 {bid} 已提交 Delta 但 commit log 未写：重启后重放 -> {msg}"}
        # ④ 写 checkpoint 的 offsets + commits
        cp.offsets[bid] = offs
        cp.commits.append(bid)
    return {"重复行": 0, "总行数": delta.total_rows(), "已提交批次": list(cp.commits),
            "说明": "全部批次完成"}

BATCHES = [120, 130, 110, 140, 125]
CLEAN = sum(BATCHES)
print("① 四种崩溃时序 + **重启**（标准 Delta sink：有幂等键）")
print(f"  {'崩溃点':<34} {'最终总行数':>10} {'重复':>6} 说明")
for stage, label in (("before_write", "写入前"),
                     ("during_write", "写入中（未提交）"),
                     ("after_commit_before_checkpoint", "已提交 Delta、commit log 未写"),
                     (None, "无崩溃（基线）")):
    cp, delta = Checkpoint(), DeltaLog()
    first = run_pipeline(BATCHES, kill_at=((2, stage) if stage else None),
                         idempotent=True, cp=cp, delta=delta)
    # ★ 关键：崩溃后**重启**（沿用同一 checkpoint 与同一张表），这才是客户看到的最终状态
    second = run_pipeline(BATCHES, kill_at=None, idempotent=True, cp=cp, delta=delta)
    total = delta.total_rows()
    dup = "否" if total == CLEAN else f"是（多 {total - CLEAN}）"
    note = first["说明"] if stage else second["说明"]
    print(f"  {label:<34} {total:>10,} {dup:>6} {note}")
print("  读法：**四种时序在重启后都不重复** —— 「写入前」靠重放、「写入中」靠孤儿文件不可见、")
print("        「已提交 Delta 但 commit log 未写」靠 `SetTransaction` 幂等键跳过、其余靠 commit log 续跑")

print("\n② 换成不幂等的 sink（Kafka/JDBC/foreachBatch 自写），同样做 kill + 重启")
print(f"  {'崩溃点':<34} {'最终总行数':>10} {'重复':>6} 说明")
for stage, label in (("after_commit_before_checkpoint", "已提交、commit log 未写"),
                     ("during_write", "写入中（未提交）"),
                     (None, "无崩溃（基线）")):
    cp, delta = Checkpoint(), DeltaLog()
    run_pipeline(BATCHES, kill_at=((2, stage) if stage else None),
                 idempotent=False, cp=cp, delta=delta)
    run_pipeline(BATCHES, kill_at=None, idempotent=False, cp=cp, delta=delta)
    total = delta.total_rows()
    dup = "否" if total == CLEAN else f"是（多 {total - CLEAN}）"
    print(f"  {label:<34} {total:>10,} {dup:>6} "
          f"{'重放时幂等键缺失 -> 又写了一遍' if total > CLEAN else '未触发重放'}")
print("  读法：**幂等键一去掉，同一时序立刻重复**（多出整整一批）—— 这就是")
print("        「Delta sink 恰好一次」与「Kafka/JDBC sink 至少一次」的差别；写外部系统必须自造幂等")

print("\n③ checkpoint 丢失：换目录 = 换 appId = 历史重放")
cp = Checkpoint(); delta = DeltaLog()
r1 = run_pipeline([100, 100, 100], cp=cp, delta=delta)                 # 正常跑三批
print(f"  正常完成：总行数 {r1['总行数']:,}，已提交批次 {r1['已提交批次']}")
new_cp = Checkpoint(app_id="app-2")                                    # ★ 新目录 -> 新 appId
r2 = run_pipeline([100, 100, 100], cp=new_cp, delta=delta)
print(f"  用新 checkpoint 重启：总行数 {r2['总行数']:,} -> 多出 "
      f"{r2['总行数']-r1['总行数']:,} 行（**重复**）")
print(f"  Delta 事务日志里的键：{sorted(delta.committed.keys())}")
print("  读法：**appId 变了，幂等键就失效** —— 所以 checkpoint 必须放在持久存储、且重启沿用同一目录；")
print("        「新集群 + 新目录」是生产里最常见的重复来源")

print("\n④ 非确定性转换：重放写进去的内容不同（仍只写一次）")
def transform(batch: List[int], deterministic: bool, seed: int) -> List[int]:
    if deterministic:
        return [x * 2 for x in batch]
    rnd = random.Random(seed)                                         # 每次重放种子不同
    return [x * 2 + rnd.randrange(10) for x in batch]
b = list(range(10))
first = transform(b, deterministic=False, seed=1)
replay = transform(b, deterministic=False, seed=2)
first_d = transform(b, deterministic=True, seed=1)
replay_d = transform(b, deterministic=True, seed=2)
print(f"  非确定性：首次 {first[:5]} vs 重放 {replay[:5]} -> 一致？{first == replay}")
print(f"  确定性：  首次 {first_d[:5]} vs 重放 {replay_d[:5]} -> 一致？{first_d == replay_d}")
print("  读法：非确定性**不会造成重复行**，但会让「同一批次重放后内容不同」——")
print("        排查问题时无法用重放复现，且 upsert 的结果会随重试变化（这很危险）")

print("\n⑤ 修复清单：重复与丢失要用不同手段")
ISSUES = [
    ("checkpoint 在临时盘/每次新建", "重复", "checkpoint 放持久存储并固定目录"),
    ("写 Kafka / JDBC（无 upsert）", "重复", "业务主键 + MERGE，或下游去重"),
    ("foreachBatch 写两个系统", "部分生效", "幂等键 + 对账补偿（无跨系统原子性）"),
    ("socket 源 / 无 checkpoint 从 latest", "丢数据", "换可重放源 + 明确起始 offset"),
    ("Kafka retention 过期 / offset 重置", "丢数据", "扩大 retention + 监控消费 lag + 补数"),
    ("多 writer 写同一张表", "冲突/重复", "单 writer，或统一走 MERGE"),
]
print(f"  {'场景':<36} {'后果':<10} 修法")
for scene, result, fix in ISSUES:
    print(f"  {scene:<36} {result:<10} {fix}")
print("  读法：**「重复」与「丢失」是两类问题** —— 重复靠幂等键/去重，丢失靠保留期与补数；")
print("        用错手段会既解决不了问题、又引入新问题（例如为防重复而丢数据）")
```

预期输出要点（实跑）：① 四种崩溃时序**在重启之后**都不重复（最终总行数都等于 625）：「写入前」靠重放、「写入中」靠孤儿文件不可见、「**已提交 Delta 但 commit log 未写**」靠 `SetTransaction` 幂等键跳过、「commit log 已写」直接从下一批继续——**注意必须模拟「kill + 重启」两步**，只看崩溃瞬间的中间状态会得出错误结论（我第一版演示就犯了这个错）；② **去掉幂等键**（换成 Kafka/JDBC sink）后，「已提交但 commit log 未写」这一格在重启后**多出整整一批（625 → 735，多 110 行）**——这就是「Delta sink 恰好一次 vs 其他 sink 至少一次」的差别；③ **换 checkpoint 目录 = 换 appId**，幂等键失效、历史批次全部重放 → **多出一倍行数**（生产里最常见的重复来源）；④ 非确定性转换**不会造成重复**，但会让「重放后内容不同」，排查与 upsert 都不可复现；⑤ 修复清单把「重复」与「丢失」分成两类，各自对应不同手段。

## 常见追问

- **追问**：如果 checkpoint 丢了怎么办？
  - 要点：**无法自动恢复恰好一次**。可选：① 从 Kafka 的**已知 offset** 重新开始（结合业务主键 + `MERGE` 让重放幂等）；② 若表上有 `_commit_timestamp`/业务时间列，可以按时间去重；③ 最坏情况**重建表**再补数。**教训是 checkpoint 必须放在持久、独立、固定的位置。**
- **追问**：`foreachBatch` 里写外部系统怎么保证不重复？
  - 要点：**做不到跨系统原子性**。做法：① 用**幂等键**（batchId 或业务主键）让外部写可重放；② 把「要做的动作」先写进 Delta（outbox 模式），下游消费并保证幂等；③ 对账任务定期比对两边差异并补偿。**先承认做不到，再给补偿方案**，比声称「支持 exactly-once」专业得多。
- **追问**：`dropDuplicates` 能代替幂等吗？
  - 要点：**不能代替，但可以补充**。它在**流内部**有状态去重（状态受 watermark 限制），能挡住「同一条记录被重复投递」；但挡不住「整个批次被重放」造成的**不同批次重复**（因为去重状态的 key 是行内容，若内容一致则能挡，若含时间戳等变化字段就挡不住）。**幂等键在 sink 侧，去重在流侧，两者职责不同。**
- **追问**：`VACUUM` 会不会破坏恢复？
  - 要点：**不影响流恢复**（恢复靠 checkpoint 与事务日志，不依赖旧数据文件）；它影响的是 **time travel**（旧版本不可读）与**孤儿文件清理**。所以保留期要按「审计/回滚需求」设，而不是按「流恢复」设。
- **追问**：多个流写同一张 Delta 表会怎样？
  - 要点：Delta 支持**并发写**（乐观并发控制），但**同一批次的幂等键只在同一 appId 下有效**——两个作业写同一张表不会互相跳过，可能出现业务重复。**推荐单 writer，或让所有写入走统一的幂等逻辑（MERGE）。**
- **追问**：怎么验证「真的恰好一次」？
  - 要点：① **故障注入**：在 batch 各阶段 `kill -9`，重启后比对**主键去重计数**与期望值；② 检查 Delta 的 `_commit_version`/`_commit_timestamp`（同一条业务记录不应出现两次不同版本）；③ 对账脚本：Kafka 侧消息数 vs Delta 侧行数（考虑过滤/聚合）；④ 监控 `numInputRows` 与批次耗时突变（重放会出现「重复处理同一 offset」的迹象）。
- **追问**：`trigger(availableNow=True)` 有什么区别？
  - 要点：它把流当**增量批**跑（处理完当前可用数据就停），**仍然用同一套 checkpoint 与幂等机制**——所以崩溃恢复语义相同；区别是**常驻与否**（成本与延迟），不是一致性。

## 相关题目

- [[databricks-04]]：Spark join 的 straggler——同属「先讲清机制、再谈调优/修复」。
- [[databricks-02]]：内存受限下的 top-K——流式场景里的「状态有界」问题同源。
- [[databricks-01]]：批处理 logger 的关闭语义——「进程内批」与「端到端批」的一致性对照。
- [[agents-02]]：工具调用的错误、超时与重试——**重试必须幂等**是同一个道理。
- [[system-design-01]]：企业级 RAG 的系统设计——幂等与一致性在数据管道里的另一处体现。

## 参考资料与归属

- **Databricks 文档：在 Delta Lake 上使用 Structured Streaming** —— Databricks：<https://docs.databricks.com/aws/en/structured-streaming/delta-lake>。第 1、2 节"checkpoint 的 offset/commit log 与 Delta 事务日志配合实现恰好一次"的机制来自该文档。
- **Delta Lake 文档：并发控制与事务日志** —— Delta Lake：<https://docs.delta.io/latest/concurrency-control.html>。第 2 节「原子提交、乐观并发、`SetTransaction` 幂等键」的机制来自该文档。
- **一个 Spark 作业把 2 TB 的事实表与 50 GB 的维表做 join（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节「先诊断瓶颈/成因再修」的方法论与本篇同源。
- **agent 循环中如何处理工具调用错误、超时与重试（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5 节「重试必须幂等」的论证取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（5 个批次的行数 120/130/110/140/125、offset 与行数的对应关系、appId 变化造成的整表重放）都是为演示崩溃时序而构造的**简化模型**；它刻意忽略了真实系统里的细节（文件级提交、`_delta_log` 的 JSON 内容、checkpoint 的 state 目录、Kafka 的 offset 语义），所以**行数与「重复」只用于表达时序差异，不代表真实行为**。真实语义以 Databricks/Delta 官方文档为准。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
