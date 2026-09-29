---
type: question
id: anthropic-11
company: Anthropic
topic: coding
order: 11
question: SQL：判断是否存在任何用户的订阅日期区间相互重叠。
question_en: "SQL: determine whether any user has overlapping subscription date ranges."
asked_at: []
level: 进阶
tags: [SQL, 区间重叠, 半开区间, 自连接, 时态数据]
sources:
  - title: SELECT（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_select.html
    author: SQLite Consortium
    published: 
  - title: Date And Time Functions（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_datefunc.html
    author: SQLite Consortium
    published: 
  - title: WITH clause（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_with.html
    author: SQLite Consortium
    published: 
related: [anthropic-10, anthropic-12, consumer-ml-13, anthropic-08, coding-09]
updated: 2026-09-28
---

## 一句话答案

> 区间重叠的判定式只有一条，但**语义（开闭区间）必须先说清**：
> 对两个区间 $[s_1,e_1)$ 与 $[s_2,e_2)$（**半开区间**，即「结束日当天不再订阅」），重叠当且仅当
> $$s_1 < e_2\ \land\ s_2 < e_1$$
> 用**严格小于**是关键：它让「A 结束的那天 B 开始」**不算重叠**（连续续订是正常业务，不是数据错误）。若业务语义是闭区间 $[s,e]$（结束日当天仍有效），则判定式变成 $s_1\le e_2\land s_2\le e_1$——**同一条 SQL 在两种语义下答案不同**，所以第一句话必须是「先确认口径」。
> 骨架（找出所有重叠对，而不是只回答「有/没有」）：
> ```sql
> SELECT a.user_id, a.id AS id_a, b.id AS id_b
> FROM subs a JOIN subs b
>   ON a.user_id = b.user_id
>  AND a.id < b.id                    -- 只比一次，排除自配对与顺序重复
>  AND a.start_date < b.end_date      -- 两个不等式 = 区间重叠
>  AND b.start_date < a.end_date
> ORDER BY a.user_id, id_a, id_b;
> ```
> 若只要回答「是否存在」：外层套 `SELECT EXISTS(...)`（或 `LIMIT 1`）——**别把全表结果都算出来再判断**。

## 面试官在考什么

- **先问口径**：半开还是闭区间？`NULL` 结束日代表「至今有效」还是「未知」？同一天开始算不算重叠？时区与日期类型（`DATE` vs `TIMESTAMP`）？
- **重叠判定的正确性**：能否写出 $s_1<e_2\land s_2<e_1$，并解释为什么不是「某区间包含另一区间」（包含只是重叠的特例）。
- **`a.id < b.id` 的作用**：既排除自配对，又把无序对归一化（否则每对出现两次，且自己和自己必然「重叠」）。
- **NULL 的处理**：开放式区间（`end_date IS NULL` 表示持续有效）是真实场景的标准形态，必须显式处理（用 `COALESCE(end_date, '9999-12-31')` 或加 `IS NULL` 分支），否则 `NULL` 参与比较会得到 `NULL`（不是 `TRUE`）而**静默漏报**。
- **性能**：朴素自连接是 $O(n^2)$/用户；正确做法是**按用户分区并按开始时间排序**，只需与前一个区间比较——这就是 `LAG()` 窗口函数的用途（线性复杂度）。
- **数据质量的视角**：重叠往往是上游写入缺陷（缺唯一约束、并发插入），所以除了查询，还要给出**约束与修数方案**（能否用排他约束/触发器阻止未来重叠）。

**常见错误答案**

- `WHERE a.start BETWEEN b.start AND b.end`——只覆盖了一种相对位置，漏掉「a 包含 b」与「a 在 b 之前开始但在其内结束」等情形。
- 忘记 `a.id < b.id`，结果每对出现两次并包含自配对。
- 用 `COALESCE(end_date, now())` 处理开放区间——「至今有效」应当用**未来日期**（如 `'9999-12-31'`）而不是「今天」，否则同一查询在不同日期结果不同。
- 只输出「存在/不存在」，不给重叠对——面试官通常想看你能否给出可修复的证据。

## 原理与推导

### 1. 为什么是两个不等式

两个区间不重叠只有两种相对位置：$e_1\le s_2$（1 在 2 之前）或 $e_2\le s_1$（2 在 1 之前）。取反即得重叠条件（半开区间）：

$$\text{overlap}\iff \lnot(e_1\le s_2\lor e_2\le s_1)\iff s_1<e_2\ \land\ s_2<e_1$$

**闭区间的差别**：$e_1=s_2$ 时闭区间**共享端点**（算重叠），所以严格小于要放宽为 $\le$。

### 2. 排序 + 相邻比较：从 $O(n^2)$ 到 $O(n\log n)$

按用户分区、按 `start_date` 升序后，任意重叠对必然在**排序序列中相邻**（或更准确：只需检查「当前区间的开始是否小于此前所有区间的最大结束日」）。窗口函数版本：

```sql
WITH s AS (
  SELECT user_id, id, start_date, COALESCE(end_date, '9999-12-31') AS e,
         MAX(COALESCE(end_date,'9999-12-31')) OVER (
           PARTITION BY user_id ORDER BY start_date, id
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_end
  FROM subs
)
SELECT user_id, id, start_date, prev_max_end FROM s
WHERE prev_max_end IS NOT NULL AND start_date < prev_max_end;
```

**要点**：`prev_max_end` 是「此前所有区间的最大结束日」（用 `ROWS ... 1 PRECEDING` 的前缀最大值），只要当前开始日小于它就必然重叠——这个写法比自连接快，而且能直接给出**违规的那一行**。

### 3. NULL 与开放区间

| 语义 | 表示 | 处理 |
| --- | --- | --- |
| 至今有效 | `end_date IS NULL` | 视为 `'9999-12-31'`（未来），而不是 `now()` |
| 未知 | `end_date IS NULL` 但含义是缺失 | 不能当有效区间参与判定，应单列出来人工处理 |
| 单日订阅 | `start = end`（半开） | 长度为 1 天的区间；`start=end` 在闭区间下长度为 0，要按业务决定 |

**SQL 三值逻辑的坑**：`NULL < '2026-01-01'` 的结果是 `NULL`（既非真也非假），在 `WHERE` 里等同于假 —— **漏报就是这么来的**。任何参与比较的列都要先 `COALESCE` 或显式判空。

### 4. 从「查出来」到「防住」

- **数据库约束**：PostgreSQL 的排他约束（`EXCLUDE USING gist (user_id WITH =, daterange(start,end) WITH &&)`）能在写入时就阻止重叠；MySQL/SQLite 没有等价能力，只能用触发器或在应用层加锁。
- **并发写入**：两个并发插入各自看不到对方 → 都会成功。只有唯一约束/排他约束/串行化隔离级别能真正防住，应用层「先查再插」存在竞态（TOCTOU）。
- **修数**：把重叠对导出 → 人工/规则决定保留哪一条（通常保留最新）→ 用 upsert 重写 → 回放校验。

## 数值与代码验证

### 表 1：六种区间关系与判定结果（半开区间）

| 关系 | 示例（A=[10,20)） | $s_A<e_B\land s_B<e_A$ | 重叠？ |
| --- | --- | --- | --- |
| 完全分离（A 在前） | B=[5,10) | 10<10 假 | 否（**相邻不算**） |
| 部分重叠 | B=[15,25) | 10<25 真 ∧ 15<20 真 | 是 |
| B 包含 A | B=[5,25) | 真 ∧ 真 | 是 |
| A 包含 B | B=[12,15) | 真 ∧ 真 | 是 |
| 相同区间 | B=[10,20) | 真 ∧ 真 | 是 |
| 完全分离（B 在前） | B=[25,30) | 20<30 真 ∧ 25<10 假 | 否 |

**读法**：只有第一行与最后一行是「不重叠」，且**第一行正是连续续订**（A 结束当天 B 开始）——它在半开语义下合法。

### 表 2：示例数据（本文件可复现）

| id | user | start | end | 说明 |
| --- | --- | --- | --- | --- |
| 1 | u1 | 2026-01-01 | 2026-02-01 | 正常 |
| 2 | u1 | 2026-02-01 | 2026-03-01 | **相邻续订**（不应报重叠） |
| 3 | u1 | 2026-02-15 | 2026-03-15 | **与 id=2 重叠** |
| 4 | u2 | 2026-01-01 | NULL | 至今有效 |
| 5 | u2 | 2026-05-01 | 2026-06-01 | **与 id=4 重叠**（开放区间） |
| 6 | u3 | 2026-01-01 | 2026-07-01 | 单条，无重叠 |

期望：u1 报出 (2,3)；u2 报出 (4,5)；u3 无。

### 可运行代码

```python
# sqlite3 实跑：自连接版 vs 窗口函数版，并对比「半开 vs 闭」的结论差异
import sqlite3
con = sqlite3.connect(":memory:"); cur = con.cursor()
cur.executescript("""
CREATE TABLE subs(id INT, user_id TEXT, start_date TEXT, end_date TEXT);
INSERT INTO subs VALUES
 (1,'u1','2026-01-01','2026-02-01'),
 (2,'u1','2026-02-01','2026-03-01'),
 (3,'u1','2026-02-15','2026-03-15'),
 (4,'u2','2026-01-01',NULL),
 (5,'u2','2026-05-01','2026-06-01'),
 (6,'u3','2026-01-01','2026-07-01');
""")

OPEN = "'9999-12-31'"      # 开放区间视为未来日期（不要用 now()）

print("① 自连接版（半开区间：相邻续订不算重叠）")
for row in cur.execute(f"""
SELECT a.user_id, a.id, b.id
FROM subs a JOIN subs b
  ON a.user_id = b.user_id AND a.id < b.id
 AND a.start_date < COALESCE(b.end_date, {OPEN})
 AND b.start_date < COALESCE(a.end_date, {OPEN})
ORDER BY a.user_id, a.id, b.id
"""):
    print("   ", row)

print("\n② 闭区间版（把严格小于放宽为 <=，相邻续订会被误报为重叠）")
for row in cur.execute(f"""
SELECT a.user_id, a.id, b.id
FROM subs a JOIN subs b
  ON a.user_id = b.user_id AND a.id < b.id
 AND a.start_date <= COALESCE(b.end_date, {OPEN})
 AND b.start_date <= COALESCE(a.end_date, {OPEN})
ORDER BY a.user_id, a.id, b.id
"""):
    print("   ", row)

print("\n③ 只要「是否存在」：EXISTS 短路，不物化全部重叠对")
for row in cur.execute(f"""
SELECT EXISTS(
  SELECT 1 FROM subs a JOIN subs b
    ON a.user_id=b.user_id AND a.id<b.id
   AND a.start_date < COALESCE(b.end_date, {OPEN})
   AND b.start_date < COALESCE(a.end_date, {OPEN}))
"""):
    print("    存在重叠 =", bool(row[0]))

print("\n④ 窗口函数版（O(n log n)，并直接给出违规行与「此前最大结束日」）")
for row in cur.execute(f"""
WITH s AS (
  SELECT id, user_id, start_date, COALESCE(end_date, {OPEN}) AS e,
         MAX(COALESCE(end_date, {OPEN})) OVER (
           PARTITION BY user_id ORDER BY start_date, id
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_end
  FROM subs)
SELECT user_id, id, start_date, prev_max_end
FROM s WHERE prev_max_end IS NOT NULL AND start_date < prev_max_end
ORDER BY user_id, id
"""):
    print("   ", row)

print("\n⑤ NULL 陷阱：不做 COALESCE 时，开放区间会静默漏报")
for row in cur.execute("""
SELECT COUNT(*) FROM subs a JOIN subs b
  ON a.user_id=b.user_id AND a.id<b.id
 AND a.start_date < b.end_date AND b.start_date < a.end_date
"""):
    print("    漏报后的重叠对数 =", row[0], "（正确应为 2：u2 那对被 NULL 吞掉了）")

print("\n⑥ 修数演示：保留每个用户「开始最晚」的一条，把被它包含的旧区间截断")
cur.executescript("""
CREATE TEMP TABLE fixed AS
SELECT id, user_id, start_date, end_date,
       LEAD(start_date) OVER (PARTITION BY user_id ORDER BY start_date, id) AS next_start
FROM subs;
UPDATE fixed SET end_date = next_start
 WHERE next_start IS NOT NULL AND (end_date IS NULL OR end_date > next_start);
""")
for row in cur.execute("SELECT id,user_id,start_date,end_date FROM fixed ORDER BY user_id,id"):
    print("   ", row)
con.close()
```

预期输出要点（实跑）：① 自连接版精确报出 `(u1,2,3)` 与 `(u2,4,5)`，**相邻续订 (u1,1,2) 不被报出**；② 闭区间版会额外把 (u1,1,2) 报出来——同一份数据、两种口径、不同结论；③ `EXISTS` 返回 1（存在）；④ 窗口函数版给出同样的违规行，并显示 `prev_max_end` 便于定位；⑤ **不做 `COALESCE` 时重叠对数从 2 掉到 1**——u2 那对因为 `end_date IS NULL` 参与比较得到 `NULL`（在 `WHERE` 中等同假）而被静默吞掉，这就是三值逻辑导致的漏报；⑥ 修数演示把被后续区间覆盖的旧区间截断（`end_date` 收到 `next_start`），输出一张无重叠的表。

## 常见追问

- **追问**：为什么不用 `BETWEEN`？
  - 要点：`BETWEEN` 表达「包含」，只能覆盖「一个区间包含某点」，无法表达两个区间的关系；而且闭区间语义会误判相邻续订。
- **追问**：PostgreSQL 里怎么从根上防止重叠？
  - 要点：排他约束 `EXCLUDE USING gist (user_id WITH =, daterange(start_date, end_date, '[)') WITH &&)`——它在**写入时**拒绝重叠，天然解决并发竞态；MySQL/SQLite 只能用触发器或应用层锁（仍有竞态）。
- **追问**：如果表有几个亿行，这条查询还能跑吗？
  - 要点：自连接版本不能；用窗口函数版（按用户分区 + 有序扫描）并建 `(user_id, start_date)` 索引；进一步可按用户哈希分片并行（串 [[anthropic-09]]）。
- **追问**：怎么处理时区？
  - 要点：区间比较要求**同一时间基准**——统一存 UTC，或统一存「业务本地日」的 DATE 类型；混用会导致边界日误判（尤其是跨时区的续订）。
- **追问**：重叠一定是错误吗？
  - 要点：不一定——家庭套餐、升级/降级并存、试用与正式订阅并存都可能合法。所以要**先定义业务规则**（例如「同一 user 同一 product 不允许同时有两个 active 订阅」），再把它写成约束。
- **追问**：如何做「某一天的有效订阅」这类时态查询？
  - 要点：`WHERE start_date <= :day AND COALESCE(end_date,'9999-12-31') > :day`；若要按天展开可用日历表 join——这是时态数据建模的常规手法。

## 相关题目

- [[anthropic-10]]：自连接 + 归一化的另一类题型（组合计数），可对照两题的「先定语义再写 SQL」。
- [[anthropic-12]]：ETL 脏数据下的「当前值」，与本题的开放区间/重复行同属数据质量问题。
- [[consumer-ml-13]]：数据质量与新鲜度监控，重叠检测可以做成一条数据质量断言。
- [[anthropic-08]]：大规模扫描（几亿行）的并行化手段。
- [[coding-09]]：区间与序列处理的通用技巧。

## 参考资料与归属

- **SELECT（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_select.html>。第 1 节与代码里的 `JOIN`/`EXISTS`/窗口函数语法来自这份文档。
- **Date And Time Functions（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_datefunc.html>。示例中日期以 `TEXT`（ISO-8601）存储并可直接比较的依据来自这份文档。
- **WITH clause（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_with.html>。窗口函数版使用 CTE 的写法与作用域来自这份文档。
- **延伸来源说明**：表 1 的六种关系、表 2 的示例数据（u1/u2/u3 共 6 条）与修数演示都是为说明本仓库口径而构造的；排他约束的写法属于 PostgreSQL 特性（未列入来源数组，仅作机制说明）。来源仅用于 SQL 语义与函数行为的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
