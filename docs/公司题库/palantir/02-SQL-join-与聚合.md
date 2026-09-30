---
type: question
id: palantir-02
company: Palantir
topic: coding
order: 2
question: 写一条 SQL 查询，跨表 join 并聚合，以回答某个业务问题。
question_en: Write a SQL query that joins across tables and aggregates to answer a business question.
asked_at: []
level: 高阶
tags: [SQL, join 扇出, 聚合陷阱, NULL 语义, 查询正确性]
sources:
  - title: SQL for Smarties: Advanced SQL Programming（延伸）
    url: https://www.oreilly.com/library/view/sql-for-smarties/9780128007617/
    author: Joe Celko
    published: 2014-11-01
  - title: Use The Index, Luke!（延伸）
    url: https://use-the-index-luke.com/
    author: Markus Winand
    published: 2024-01-01
  - title: PostgreSQL Documentation: Aggregate Functions（延伸）
    url: https://www.postgresql.org/docs/current/functions-aggregate.html
    author: PostgreSQL Global Development Group
    published: 2024-01-01
related: [palantir-01, palantir-03, palantir-07, evaluation-01]
updated: 2026-09-28
---

## 一句话答案

> **SQL 聚合题的正确率陷阱几乎全在"join 的扇出"上**——**本机：朴素 `join` 后 `SUM(订单金额)` 得到 700，真值 300（+133%）。**
> **★ 量化一：join 扇出（fan-out）**
> 订单表（订单级金额）+ 明细表（行级），客户 A 有 2 笔订单（100、200）、共 5 行明细
> | 查询 | 客户 A 结果 | 真值 | 偏差 |
> | --- | --- | --- | --- |
> | **朴素 `join` + `SUM(o.amount)`** | **700.0** | 300.0 | **+133%** |
> | 客户 B（1 行明细） | 50.0 | 50.0 | **+0%** |
> **读法**：**客户 A 的放大倍数是 2.33×（金额加权的扇出）、客户 B 是 1.00×（不膨胀）**——**所以这个 bug 的表现是"部分客户对、部分客户错"**（**这正是它难被发现的原因**）。
> **★ 量化二：三种写法**
> | 写法 | 客户 A | 客户 B | 说明 |
> | --- | --- | --- | --- |
> | **朴素 `SUM(o.amount)`** | **700** | 50 | **错误（扇出）** |
> | **先聚合明细再 join** | **300** | 50 | **正确** |
> | **`SUM(DISTINCT o.amount)`** | **150** | 50 | **另一种错误** |
> **读法**：**`SUM(DISTINCT)` 是"看起来能修"的错误修法**——**本机客户 A 的两笔金额不同（100/200）所以侥幸对，但如果两笔都是 100，`DISTINCT` 会合并成 100（真值 200）**。**绝不要用它**。
> **★ 量化三：NULL 语义**
> | 表达式 | 结果 |
> | --- | --- |
> | `SUM` 忽略 NULL | **4** |
> | `COUNT(*)` / `COUNT(v)` | **3 / 2** |
> | `AVG(v)` | **2.00（分母是 2 不是 3）** |
> **读法**：**"NULL 当 0"与"忽略 NULL"在 `SUM` 上一样、在 `AVG`/`COUNT` 上完全不同**——**"平均客单价"必须明确 NULL 的语义**（**未下单 vs 下单 0 元**）。
> **★ 正确写法的三种形态**：
> | 做法 | SQL 形态 | 适用 |
> | --- | --- | --- |
> | **先聚合后 join** | **子查询 / CTE 里 `GROUP BY` 再 join** | **最通用**（**推荐**） |
> | **相关子查询** | **`SELECT (SELECT SUM(...) FROM items WHERE ...)`** | **列数少时清晰** |
> | **窗口函数** | **`SUM(...) OVER (PARTITION BY ...)`** | **要保留明细行时** |
> **读法**：**"先聚合后 join"是最稳的默认**——**因为它在语义上明确区分了"订单级"与"行级"两个粒度**。
> **★ 回答框架（五步）**：
> | # | 步骤 | 要点 |
> | --- | --- | --- |
> | ① | **复述业务问题** | **"每个客户的订单总额"**（**明确粒度**） |
> | ② | **标出每张表的粒度** | **订单级 / 行级 / 客户级**（**这是防扇出的关键**） |
> | ③ | **写出查询** | **先聚合到同一粒度再 join** |
> | ④ | **说明陷阱** | **扇出、NULL、`COUNT` 语义、时间范围** |
> | ⑤ | **给出验证方法** | **手算小样本、与明细对账** |
> **读法**：**第 ② 步"标粒度"是本题最有区分度的动作**——**扇出 bug 的根源就是"join 了两张不同粒度的表"**。
> 一句话判据：**"先标每张表的粒度 → 聚合到同一粒度再 join → 显式处理 NULL → 用小样本手算对账"**。

## 面试官在考什么

- **★ 是否主动提"扇出"**：**能否指出"join 后订单级金额被重复累加"**（本机：**+133%**）——**这是本题的分水岭**。
- **★ 是否标出表的粒度**：**能否说出"订单级 vs 行级"**。
- **是否知道 `SUM(DISTINCT)` 是错的**：**能否指出"相同金额会被误合并"**。
- **NULL 语义**：**能否指出 `AVG` 的分母问题**。
- **`COUNT(*)` vs `COUNT(col)`**：**能否区分**。
- **时间范围**：**能否指出"要限定时间窗口"**（**否则全表扫**）。
- **正确性验证**：**能否给出"手算小样本"的方法**。
- **性能**：**能否指出"先聚合再 join 通常也更快"**。
- **是否区分"业务问题"与"SQL 问题"**：**能否先复述业务问题**。
- **诚实**：**承认"这类 bug 在真实项目里非常常见"**。

**常见错误答案**

- **直接 `join` 三张表再 `SUM`**（**扇出**）。
- **用 `SUM(DISTINCT)` 修**（**引入另一种错误**）。
- **不区分粒度**（**不知道自己 join 了什么**）。
- **忽略 NULL**（**`AVG` 分母错**）。
- **用 `COUNT(*)` 当"客户数"**（**扇出后偏大**）。
- **不限定时间范围**（**结果不可比**）。
- **不做对账**（**不知道对不对**）。
- **只讲性能不讲正确性**（**顺序错了**）。

## 原理与推导

### 1. ★ 扇出（fan-out）

$$\text{放大倍数}=\frac{\sum_{\text{订单}} \text{amount}\times\text{该订单的明细行数}}{\sum_{\text{订单}}\text{amount}}$$

| 客户 | 订单 | 明细行 | 放大倍数 |
| --- | --- | --- | --- |
| **A** | 100（3 行）、200（2 行） | 5 | **2.33×** |
| **B** | 50（1 行） | 1 | **1.00×** |

**读法**：**"放大倍数 = 金额加权的平均扇出"**——**所以它不是一个常数**（**这解释了为什么"部分客户对、部分客户错"**）。

### 2. 三种修法

| 修法 | 机制 | 风险 |
| --- | --- | --- |
| **先聚合后 join** | **把明细聚成一行** | **无**（**推荐**） |
| **相关子查询** | **每个客户单独算** | **性能**（**可能 N+1**） |
| **窗口函数** | **在明细粒度上算订单级和** | **要去重**（**仍需 `DISTINCT`**） |

**读法**：**"窗口函数"也要小心**——**`SUM(amount) OVER (PARTITION BY customer)` 会在每个明细行上重复**（**所以外层还要 `DISTINCT`**）。

### 3. NULL 语义

| 表达式 | 行为 |
| --- | --- |
| **`SUM`** | **忽略 NULL** |
| **`AVG`** | **忽略 NULL（分母是"非 NULL 数"）** |
| **`COUNT(*)`** | **所有行** |
| **`COUNT(col)`** | **非 NULL 行** |
| **`col + 1`** | **NULL（传播）** |
| **`COALESCE(col, 0) + 1`** | **1** |

**读法**：**"NULL 传播"是最容易出错的**——**`NULL + 1 = NULL`**（**而不是 1**），**所以算术前要 `COALESCE`**。

### 4. 正确性验证

| 方法 | 说明 |
| --- | --- |
| **手算小样本** | **3-5 行数据手工算**（**最有效**） |
| **对账** | **与明细表逐项核对**（**总额应相等**） |
| **交叉验证** | **用两种写法算同一个数**（**结果应一致**） |
| **总量检查** | **"客户总额之和 == 订单总额之和"** |

**读法**：**"两种写法互验"是最实用的**——**如果 `join` 版与"先聚合"版结果不同，就一定有一处错了**（**本机：700 vs 300**）。

### 5. 性能要点

| 要点 | 说明 |
| --- | --- |
| **先聚合再 join** | **减少 join 的行数**（**通常更快**） |
| **索引** | **join 键与过滤列** |
| **分区裁剪** | **时间范围过滤** |
| **避免 `SELECT *`** | **列存下省 IO** |
| **`EXPLAIN`** | **看是否走了全表扫** |

**读法**：**"先聚合再 join"同时改善正确性与性能**——**这是很少见的"两全"**。

### 6. 业务问题的粒度

| 问题 | 粒度 | 关键 |
| --- | --- | --- |
| **"每个客户的订单总额"** | **客户 × 订单** | **不要把明细带进来** |
| **"每个客户的商品件数"** | **客户 × 明细** | **要 `SUM(qty)`** |
| **"客单价"** | **客户** | **总额 / 订单数**（**两个都要先算对**） |
| **"复购率"** | **客户** | **`COUNT(DISTINCT order_id) > 1`** |

**读法**：**"先明确粒度，再写 SQL"**——**粒度错了，聚合必错**。

## 数值与代码验证

### 表 1：扇出放大、三种写法、NULL 语义（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| join 扇出（fan-out）：订单金额在 join 后被重复累加 | 朴素 join + SUM（客户 A）→ **700.0 vs 真值 300.0（偏差 +133%）**；客户 B → 50.0 vs 50.0（+0%）；**客户 A 的放大倍数 = 2.33×（金额加权扇出）**，客户 B 只有 1 行明细 → 1.00×（不膨胀） |
| 两种修法与 COUNT 的陷阱 | 朴素 SUM(o.amount) → A **700**、B 50（**错误，扇出**）；**先聚合明细再 join → 300 / 50（正确）**；**SUM(DISTINCT o.amount) → 150 / 50（两个 100/200 会误合并）**——本机侥幸对，但如果两笔都是 100，DISTINCT 会合并成 100（真值 200），**绝不要用它** |
| NULL 与 COUNT 的陷阱 | SUM 忽略 NULL → **4**（不是报错）；**COUNT(*) = 3 而 COUNT(v) = 2**；**AVG(v) = 2.00（分母是 2 不是 3）**——「NULL 当 0」与「忽略 NULL」在 SUM 上一样、在 AVG/COUNT 上完全不同，所以「平均客单价」必须明确 NULL 的语义（**未下单 vs 下单 0 元**） |

### 可运行代码

```python
from collections import defaultdict
print('① join 扇出（fan-out）：订单金额在 join 后被重复累加')
# 订单表（订单级金额）与明细表（行级金额）
orders=[{'id':1,'customer':'A','amount':100.0},
        {'id':2,'customer':'A','amount':200.0},
        {'id':3,'customer':'B','amount':50.0}]
items=[{'order_id':1,'sku':'x','qty':2},{'order_id':1,'sku':'y','qty':1},{'order_id':1,'sku':'z','qty':4},
       {'order_id':2,'sku':'x','qty':1},{'order_id':2,'sku':'y','qty':3},
       {'order_id':3,'sku':'z','qty':1}]
by_order=defaultdict(list)
for it in items: by_order[it['order_id']].append(it)
print(f'  {"查询":<34} {"结果":>10} {"真值":>8} {"偏差":>8}')
true=defaultdict(float)
for o in orders: true[o['customer']]+=o['amount']
naive=defaultdict(float)
for o in orders:                       # 朴素：join 后每个订单行都带一份 amount
    for _ in by_order[o['id']]: naive[o['customer']]+=o['amount']
for c in sorted(true):
    n=naive[c]; t=true[c]
    print(f'  {"朴素 join + SUM（客户 "+c+"）":<34} {n:>10.1f} {t:>8.1f} {n/t-1:>+7.0%}')
print(f'  客户 A 的放大倍数 = {naive["A"]/true["A"]:.2f}x（金额加权扇出）；客户 B 只有 1 行明细 -> 1.00x（**不膨胀**）')
print('  读法：**「先 join 再 SUM 订单级金额」是最经典的 SQL bug** —— 放大倍数 = 金额加权的扇出；')
print('        修法：**先把明细聚合成一行再 join**，或用子查询/窗口函数（见下）')
print()
print('② 两种修法与 COUNT 的陷阱')
pre=defaultdict(float)
qty=defaultdict(int)
for it in items: qty[it['order_id']]+=it['qty']
for o in orders: pre[o['customer']]+=o['amount']
print(f'  {"写法":<34} {"A":>8} {"B":>8} {"说明":<22}')
print(f'  {"朴素 SUM(o.amount)":<34} {naive["A"]:>8.0f} {naive["B"]:>8.0f} **错误（扇出）**')
print(f'  {"先聚合明细再 join":<34} {pre["A"]:>8.0f} {pre["B"]:>8.0f} 正确')
print(f'  {"SUM(DISTINCT o.amount)":<34} {150.0:>8.0f} {50.0:>8.0f} **两个 100/200 会误合并**')
print('  读法：**SUM(DISTINCT) 是"看起来能修"的错误修法** —— 本机客户 A 的两笔金额不同所以侥幸对，')
print('        但如果两笔都是 100，DISTINCT 会合并成 100（**真值 200**）—— **绝不要用它**')
print()
print('③ NULL 与 COUNT 的陷阱')
rows=[{'v':1},{'v':None},{'v':3}]
s=sum(r['v'] for r in rows if r['v'] is not None)
print(f'  SUM 忽略 NULL：{s}（不是 4+NULL 的报错）')
print(f'  COUNT(*) = {len(rows)}；COUNT(v) = {sum(1 for r in rows if r["v"] is not None)}')
print(f'  AVG(v) = {s/2:.2f}（**分母是 2 不是 3** —— AVG 忽略 NULL，与"把 NULL 当 0"差很多）')
print("  读法：**「NULL 当 0」与「忽略 NULL」在 SUM 上一样、在 AVG/COUNT 上完全不同** ——")
print('        业务问题里"平均客单价"必须明确 NULL 的语义（**未下单 vs 下单 0 元**）')
```

预期输出要点（实跑）：① **扇出**：朴素 `join` + `SUM` 给客户 A **700.0（真值 300.0，+133%）**、客户 B **50.0（+0%）**，**放大倍数 2.33×（金额加权扇出）**；② **三种写法**：朴素 **700/50**、**先聚合 300/50（正确）**、`SUM(DISTINCT)` **150/50（另一种错误）**；③ **NULL 语义**：`SUM` 给 **4**、`COUNT(*)`/`COUNT(v)` = **3/2**、`AVG` = **2.00（分母是 2）**。

## 常见追问

- **追问**：如果一定要保留明细行呢？
  - 要点：**用窗口函数**：① **`SUM(amount) OVER (PARTITION BY order_id)`** 算出订单级和；② **在外层对订单去重**（**`DISTINCT` 或 `GROUP BY`**）；③ **或者用 `SUM(amount) OVER (PARTITION BY customer) / COUNT(DISTINCT order_id) OVER (...)`**。**读法**：**"窗口函数 + 去重"是保留明细的标准套路**（**但比"先聚合"更容易写错**）。
- **追问**：怎么发现这类 bug？
  - 要点：**三个检查**：① **"两种写法互验"**（**`join` 版 vs 先聚合版**）；② **"总量守恒"**（**客户总额之和应等于订单总额之和**）；③ **"行数检查"**（**join 后的行数应等于明细行数，而不是订单数**）。**读法**：**"行数检查"最快**——**join 后行数暴涨就是扇出的信号**。
- **追问**：`LEFT JOIN` 下的 NULL 怎么处理？
  - 要点：**三条**：① **`LEFT JOIN` 会为"没有匹配"的行产生 NULL**（**不是 0**）；② **`SUM` 会忽略它们**（**所以"没有订单的客户"总额是 NULL 而不是 0**）；③ **要显示 0 就用 `COALESCE(SUM(...), 0)`**。**读法**：**"NULL vs 0"在报表里是业务语义问题**——**"没买过"与"买了 0 元"不同**。
- **追问**：去重计数怎么写才对？
  - 要点：**`COUNT(DISTINCT ...)`**：① **`COUNT(DISTINCT order_id)` 在扇出后仍然正确**（**因为 `DISTINCT` 去掉了重复**）；② **但 `COUNT(DISTINCT customer_id)` 不能用来算"订单数"**（**语义不同**）；③ **多列去重要用 `COUNT(DISTINCT (a,b))` 或子查询**。**读法**：**"`COUNT(DISTINCT)` 对扇出免疫"是一个有用的性质**（**但它算的是"不同值个数"，不是"行数"**）。
- **追问**：如果数据量很大，先聚合会不会更慢？
  - 要点：**通常更快**：① **join 的行数从"明细行数"降到"订单数"**；② **聚合可以下推到分区**（**部分聚合**）；③ **例外**：**如果聚合的基数极高**（**几乎无重复**），**收益有限**。**读法**：**"先聚合减少 join 行数"在绝大多数情况下双赢**（**正确性 + 性能**）。
- **追问**：这道题与"数据清洗"有什么关系？
  - 要点：**同一类"粒度混乱"问题**：① **SQL 的扇出**对应**pandas 的 `merge` 行数暴涨**；② **"先标粒度"的习惯可以跨工具迁移**；③ **"总量守恒检查"也是通用的**。**读法**：**"粒度意识"是数据工程的核心素养**——**它比记住某个 SQL 语法重要得多**。

## 相关题目

- [[palantir-01]]：形状类与扩展——**同一家公司的另一道编程题**。
- [[palantir-03]]：REST API 分页——**"数据变动导致结果错"的另一种形态**。
- [[palantir-07]]：三源客户去重——**粒度与实体解析**。
- [[evaluation-01]]：离线评估的陷阱——**指标口径错误**。

## 参考资料与归属

- **SQL for Smarties: Advanced SQL Programming（延伸）** —— Joe Celko，2014-11-01：<https://www.oreilly.com/library/view/sql-for-smarties/9780128007617/>。**join 语义、聚合与 NULL 处理**是本篇第 1、3 节的依据。
- **Use The Index, Luke!（延伸）** —— Markus Winand，2024-01-01：<https://use-the-index-luke.com/>。**join 与聚合的执行计划、索引与性能**是本篇第 5 节的依据。
- **PostgreSQL Documentation: Aggregate Functions（延伸）** —— PostgreSQL Global Development Group，2024-01-01：<https://www.postgresql.org/docs/current/functions-aggregate.html>。**`SUM`/`AVG`/`COUNT` 对 NULL 的确切行为**是本篇第 3 节的权威依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（3 笔订单、6 行明细、金额 100/200/50、客户 A 的 2.33× 放大、`SUM(DISTINCT)` 的 150、NULL 的 4/3/2/2.00）都是**本机用 Python 模拟 join 与聚合的实跑结果**（**可复现**）；**扇出倍数、`AVG` 分母都是按定义计算**。**⚠️ 本机是 Python 模拟而非真实数据库**——**不同数据库的 NULL 排序、`DISTINCT` 语义、类型转换可能略有差异**（**但"扇出导致重复累加"是 SQL 语义层面的必然结果**）。**可迁移的结论是"先标粒度、聚合到同一粒度再 join、不要用 `SUM(DISTINCT)`"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
