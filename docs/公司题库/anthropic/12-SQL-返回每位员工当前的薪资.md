---
type: question
id: anthropic-12
company: Anthropic
topic: coding
order: 12
question: SQL：在 ETL 错误导致每年都插入一条新薪资行之后，返回每位员工当前的薪资。
question_en: "SQL: return each employee's current salary after an ETL error inserted a new salary row every year."
asked_at: []
level: 进阶
tags: [SQL, 窗口函数, 时态数据, 数据修复, 去重]
sources:
  - title: Window Functions（SQLite 文档）（延伸）
    url: https://www.sqlite.org/windowfunctions.html
    author: SQLite Consortium
    published: 
  - title: SELECT（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_select.html
    author: SQLite Consortium
    published: 
  - title: WITH clause（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_with.html
    author: SQLite Consortium
    published: 
related: [anthropic-11, anthropic-10, consumer-ml-13, anthropic-08, anthropic-09]
updated: 2026-09-28
---

## 一句话答案

> 「当前薪资」= **按生效日期取最新一条**，但 ETL 错误让这件事复杂在两处：
> ① **重复行**（同一年/同一天插入多条，值可能相同也可能冲突）——所以排序键要包含**去重规则**；
> ② **语义歧义**：「当前」是按生效日期（`effective_date`）还是按插入时间（`created_at`）？两条可能不一致（补录历史、回填数据），**必须先问清**。
> 标准写法（窗口函数，SQLite 3.25+ / PostgreSQL / MySQL 8+ 均支持）：
> ```sql
> SELECT employee_id, salary, effective_date
> FROM (
>   SELECT s.*,
>          ROW_NUMBER() OVER (PARTITION BY employee_id
>                             ORDER BY effective_date DESC, id DESC) AS rn
>   FROM salaries s
> ) t
> WHERE rn = 1;
> ```
> `ORDER BY effective_date DESC, id DESC` 里的 **`id DESC` 是平局打破键（tie-breaker）**——没有它，同一 `effective_date` 的多行会让结果不确定（不同执行计划给出不同答案，这在审计场景是灾难）。
> 若引擎版本不支持窗口函数：用**相关子查询**（`WHERE NOT EXISTS (更晚的一条)`）或 `GROUP BY employee_id` + 自连接回表取该最大日期对应的行。

## 面试官在考什么

- **是否先定义「当前」**：生效日期 vs 插入时间 vs 两者结合（业务上通常「生效日期最晚 + 同日期取最新插入」）。能不能主动提出这个歧义，是本题第一道分水岭。
- **平局处理**：同一日期多条记录时怎么办（取最大 id / 取最大 `created_at` / 报冲突让人工介入）——**必须显式选一个**，否则查询结果不可复现。
- **能否给出多种写法**：窗口函数（首选）、相关子查询（兼容旧引擎）、`GROUP BY + JOIN`（可读性差但常见）；并说明各自的复杂度与执行计划差异。
- **数据质量视角**：ETL 错误插入了重复行——**查询只是止血**，真正的修法是 ① 加唯一约束 `(employee_id, effective_date)`；② 写修数脚本去重（保留哪一条要有规则）；③ 加数据质量断言防止复发。
- **时态查询的推广**：「某一天的薪资」/「薪资历史变化」/「涨薪幅度的分布」——能否把「当前值」扩展到「任意时间点的值」（`effective_date <= d` 取最大）。
- **NULL 与边界**：`salary` 为 `NULL`（缺失）时算不算「当前薪资」？`effective_date` 为未来日期（提前录入）算不算当前？

**常见错误答案**

- `SELECT employee_id, MAX(salary) FROM salaries GROUP BY employee_id`——返回的是**历史最高薪资**，不是当前薪资（这是最经典的错误）。
- `ORDER BY effective_date DESC LIMIT 1`——只返回**一个员工**的结果（忘记分组）。
- 用 `MAX(effective_date)` 但回表时不带平局键——同日期多行时返回随机一行。
- 直接 `SELECT DISTINCT employee_id, salary`——把不同年份的薪资都留下了。

## 原理与推导

### 1. 「当前」的三种定义

| 定义 | 排序键 | 适用 |
| --- | --- | --- |
| 生效日期最晚 | `effective_date DESC, id DESC` | 标准做法（业务语义：某日起生效） |
| 插入时间最晚 | `created_at DESC, id DESC` | 修正补录场景（「最新录入的才作数」） |
| 二者结合 | `effective_date DESC, created_at DESC, id DESC` | 审计严格场景（既看生效也看录入顺序） |

**为什么要写出来**：如果 ETL 先补录了 2024 年的薪资、后补录 2025 年的，两种定义会给出不同结果；面试官问「哪个对」时，正确答案是「取决于业务，所以我先问」。

### 2. 三种实现的对比

| 写法 | 复杂度 | 可移植性 | 备注 |
| --- | --- | --- | --- |
| 窗口函数 `ROW_NUMBER()` | 排序 $O(n\log n)$ | SQLite 3.25+ / MySQL 8+ / PG | 首选；可直接带 tie-breaker |
| 相关子查询 `NOT EXISTS` | 最坏 $O(n^2)$（有索引时接近 $O(n\log n)$） | 全兼容 | 旧引擎兜底 |
| `GROUP BY` + 自连接回表 | 两次扫描 + join | 全兼容 | 平局时需再次去重，易错 |

### 3. 平局（同一 `effective_date` 多行）的三种业务规则

1. **取最新插入**：`ORDER BY effective_date DESC, id DESC`（最常见）；
2. **取薪资最高/最低**：把 salary 加进排序键（业务上通常不合理，但要能说出为什么）；
3. **报冲突**：`GROUP BY employee_id, effective_date HAVING COUNT(*) > 1` 单独列出，交人工裁决——审计场景更稳妥。

### 4. 从「查询」到「修复」

```sql
-- ① 找出冲突（同一员工同一生效日期多行）
SELECT employee_id, effective_date, COUNT(*) AS n, COUNT(DISTINCT salary) AS distinct_salaries
FROM salaries GROUP BY employee_id, effective_date HAVING COUNT(*) > 1;

-- ② 去重（保留最新 id），并在事务里执行
DELETE FROM salaries WHERE id NOT IN (
  SELECT MAX(id) FROM salaries GROUP BY employee_id, effective_date
);

-- ③ 防止复发（SQLite/MySQL 用唯一索引；PostgreSQL 还可用排他约束做区间级防重）
CREATE UNIQUE INDEX ux_salary_emp_date ON salaries(employee_id, effective_date);
```

**注意**：如果重复行的 `salary` 值不同（真正的冲突），**不能盲目 `MAX(id)`**——要先导出冲突清单让人确认（谁改了谁？哪个是权威来源？），否则修数会把正确值删掉。

### 5. 时态查询的推广

「某一天的薪资」：

```sql
SELECT employee_id, salary FROM (
  SELECT s.*, ROW_NUMBER() OVER (PARTITION BY employee_id
                                 ORDER BY effective_date DESC, id DESC) AS rn
  FROM salaries s WHERE effective_date <= :as_of
) WHERE rn = 1;
```

「薪资变化序列」（每次涨薪的幅度）：用 `LAG(salary) OVER (PARTITION BY employee_id ORDER BY effective_date)` 计算差值与涨幅百分比——这是「当前值」的自然延伸，也是面试常见的追问方向。

## 数值与代码验证

### 表 1：示例数据（含 ETL 重复与同日期冲突）

| id | employee | salary | effective_date | 说明 |
| --- | --- | --- | --- | --- |
| 1 | e1 | 100 | 2024-01-01 | 历史 |
| 2 | e1 | 110 | 2025-01-01 | 历史 |
| 3 | e1 | 120 | 2026-01-01 | **当前** |
| 4 | e2 | 200 | 2025-06-01 | 历史 |
| 5 | e2 | 210 | 2025-06-01 | **同日期冲突（ETL 重复）** |
| 6 | e2 | 205 | 2026-01-01 | **当前（按生效日期）** |
| 7 | e3 | 300 | **2027-03-01** | 未来生效（提前录入） |
| 8 | e3 | 290 | 2025-01-01 | 历史 |

期望（`effective_date DESC, id DESC`）：e1→120、e2→205、e3→300（未来日期也算「最新」——**要不要排除未来日期是业务问题**，代码会同时给出两种口径）。

### 表 2：三种「当前薪资」口径的差异

| 口径 | e1 | e2 | e3 | 说明 |
| --- | --- | --- | --- | --- |
| 错误写法 `MAX(salary)` | 120 | 210（**错**，历史最高） | 300 | 最经典错误 |
| 生效日期最新 + id 最大 | 120 | 205 | 300 | 标准答案 |
| 排除未来生效日期 | 120 | 205 | 290 | `WHERE effective_date <= date('now')` |

### 可运行代码

```python
# sqlite3 实跑：三种写法 + 平局 + 未来日期口径 + 冲突检测 + 修数
import sqlite3
con = sqlite3.connect(":memory:"); cur = con.cursor()
cur.executescript("""
CREATE TABLE salaries(id INTEGER PRIMARY KEY, employee_id TEXT, salary INT, effective_date TEXT);
INSERT INTO salaries VALUES
 (1,'e1',100,'2024-01-01'),(2,'e1',110,'2025-01-01'),(3,'e1',120,'2026-01-01'),
 (4,'e2',200,'2025-06-01'),(5,'e2',210,'2025-06-01'),   -- 同日期冲突（ETL 重复）
 (6,'e2',205,'2026-01-01'),
 (7,'e3',300,'2027-03-01'),(8,'e3',290,'2025-01-01');    -- e3 有「未来生效」行（用足够远的日期，避免依赖运行当天）
""")

print("① 窗口函数版（当前薪资，平局取最大 id）")
for r in cur.execute("""
SELECT employee_id, salary, effective_date FROM (
  SELECT s.*, ROW_NUMBER() OVER (PARTITION BY employee_id
                                 ORDER BY effective_date DESC, id DESC) AS rn
  FROM salaries s) WHERE rn = 1 ORDER BY employee_id
"""): print("   ", r)

print("\n② 对照：错误的 MAX(salary) 写法（返回历史最高，不是当前）")
for r in cur.execute("SELECT employee_id, MAX(salary) FROM salaries GROUP BY employee_id ORDER BY 1"):
    print("   ", r)

print("\n③ 兼容旧引擎：NOT EXISTS 相关子查询（同口径，应得到相同结果）")
for r in cur.execute("""
SELECT s.employee_id, s.salary, s.effective_date FROM salaries s
WHERE NOT EXISTS (
  SELECT 1 FROM salaries t
   WHERE t.employee_id = s.employee_id
     AND (t.effective_date > s.effective_date
          OR (t.effective_date = s.effective_date AND t.id > s.id)))
ORDER BY s.employee_id
"""): print("   ", r)

print("\n④ 平局：同一 effective_date 多条")
for r in cur.execute("""
SELECT employee_id, effective_date, COUNT(*) n, COUNT(DISTINCT salary) distinct_salaries
FROM salaries GROUP BY employee_id, effective_date HAVING COUNT(*) > 1
"""): print("   ", r)

print("\n⑤ 排除未来生效日期（业务口径二）")
for r in cur.execute("""
SELECT employee_id, salary, effective_date FROM (
  SELECT s.*, ROW_NUMBER() OVER (PARTITION BY employee_id
                                 ORDER BY effective_date DESC, id DESC) AS rn
  FROM salaries s WHERE effective_date <= date('now')) WHERE rn = 1 ORDER BY employee_id
"""): print("   ", r)

print("\n⑥ 时态查询：2025-07-01 当天的薪资")
for r in cur.execute("""
SELECT employee_id, salary, effective_date FROM (
  SELECT s.*, ROW_NUMBER() OVER (PARTITION BY employee_id
                                 ORDER BY effective_date DESC, id DESC) AS rn
  FROM salaries s WHERE effective_date <= '2025-07-01') WHERE rn = 1 ORDER BY employee_id
"""): print("   ", r)

print("\n⑦ 涨薪幅度序列（LAG）")
for r in cur.execute("""
SELECT employee_id, effective_date, salary,
       LAG(salary) OVER (PARTITION BY employee_id ORDER BY effective_date, id) AS prev_salary,
       salary - LAG(salary) OVER (PARTITION BY employee_id ORDER BY effective_date, id) AS delta
FROM salaries ORDER BY employee_id, effective_date, id
"""): print("   ", r)

print("\n⑧ 修数：先看冲突，再去重（保证幂等与可回滚——事务里执行）")
cur.execute("BEGIN")
cur.execute("""DELETE FROM salaries WHERE id NOT IN (
                 SELECT MAX(id) FROM salaries GROUP BY employee_id, effective_date)""")
cur.execute("CREATE UNIQUE INDEX ux_salary_emp_date ON salaries(employee_id, effective_date)")
cur.execute("COMMIT")
print("    去重后行数 =", cur.execute("SELECT COUNT(*) FROM salaries").fetchone()[0], "（原 8 行）")
print("    唯一索引已建立：再插入同日期行会失败")
try:
    cur.execute("INSERT INTO salaries VALUES (99,'e2',999,'2026-01-01')")
except sqlite3.IntegrityError as e:
    print("    IntegrityError:", e)
con.close()
```

预期输出要点（实跑）：① 窗口函数版给出 `e1→120 / e2→205 / e3→300`；② 错误的 `MAX(salary)` 给出 `e2→210`（**历史最高**，与 ① 不同，这正是要指出来的差异）；③ 相关子查询版结果与 ① **完全一致**（可移植性验证）；④ 平局检测报出 `(e2, 2025-06-01, 2 行, 2 个不同值)`——**值不同说明是真冲突**，不能盲目取 `MAX(id)`；⑤ 排除未来生效日期后 `e3→290`（两种业务口径的差异——注意示例特意用了足够远的 `2027-03-01`，避免结果随运行日期变化）；⑥ 时态查询给出 2025-07-01 的薪资；⑦ `LAG` 给出每次涨薪的差值与幅度；⑧ 修数后行数从 8 降到 7，且唯一索引让重复插入直接报 `IntegrityError`（**从「查出来」升级到「防得住」**）。

## 常见追问

- **追问**：为什么不直接删掉重复行？
  - 要点：先看重复行的 `salary` 是否一致——一致则删（保留最新 id 或最小 id 均可）；**不一致则是真冲突**，必须导出清单让人确认权威来源，否则会删掉正确值。
- **追问**：如果表有几十亿行，窗口函数会不会很慢？
  - 要点：窗口函数需要按 `(employee_id, effective_date)` 排序；建 `(employee_id, effective_date DESC, id DESC)` 索引可让排序走索引。更快的做法是维护一张**当前值物化表**（每次写入时 upsert），查询退化为一次点查（这是典型的「读多写少」优化）。
- **追问**：怎么防止 ETL 再次插重复？
  - 要点：唯一索引（本例）；ETL 侧改为 **upsert（`INSERT ... ON CONFLICT DO UPDATE`）**而不是纯插入；再加上数据质量断言（每日校验无重复、无未来日期异常）。
- **追问**：薪资变更要不要审计？
  - 要点：要——append-only 变更日志（谁、何时、旧值、新值、原因），当前值可由日志派生或加版本列；这比「原地更新」更适合合规场景。
- **追问**：`effective_date` 用字符串存会不会有问题？
  - 要点：ISO-8601 的 `YYYY-MM-DD` 字符串在字典序与时间序一致时可以安全比较（本例依据）；但一旦涉及时间与时区，就要用带时区的类型。混用日期与时间戳会导致边界日误判（串 [[anthropic-11]]）。
- **追问**：怎么测试这条查询？
  - 要点：构造覆盖用例——单条、多条、同日期重复（值同/值异）、未来日期、NULL 薪资；对「当前薪资」断言具体值，并对不同实现（窗口/子查询）做**结果对拍**。

## 相关题目

- [[anthropic-11]]：区间重叠与半开语义，与本题同属「时态数据 + 先定口径」的题型。
- [[anthropic-10]]：自连接与去重的另一类 SQL 题。
- [[consumer-ml-13]]：数据质量与漂移监控，本题的重复行与未来日期都可以做成断言。
- [[anthropic-08]]：大规模数据处理的并发与幂等（修数脚本同样要幂等）。
- [[anthropic-09]]：分片并行的通用方法，适用于几十亿行的薪资表处理。

## 参考资料与归属

- **Window Functions（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/windowfunctions.html>。第 1 节 `ROW_NUMBER()`/`LAG()`/`PARTITION BY` 的语义与可用版本来自这份文档。
- **SELECT（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_select.html>。第 2 节相关子查询与 `GROUP BY` 的写法来自这份文档。
- **WITH clause（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_with.html>。CTE 在改写复杂查询时的用法来自这份文档。
- **延伸来源说明**：表 1、表 2 的数据（e1/e2/e3 共 8 行，含同日期冲突与未来生效行）与修数脚本都是为说明本仓库口径而构造的示例，不代表任何真实薪资数据；唯一索引/排他约束的可移植性差异属于各数据库特性说明。来源仅用于 SQL 语义与函数行为的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
