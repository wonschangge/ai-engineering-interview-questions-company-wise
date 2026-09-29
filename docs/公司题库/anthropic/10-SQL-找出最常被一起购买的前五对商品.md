---
type: question
id: anthropic-10
company: Anthropic
topic: coding
order: 10
question: SQL：写一个查询，找出最常被一起购买的前五对商品。
question_en: "SQL: write a query to find the top five pairs of products most frequently purchased together."
asked_at: []
level: 进阶
tags: [SQL, 自连接, 关联规则, 组合计数, 性能]
sources:
  - title: SELECT（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_select.html
    author: SQLite Consortium
    published: 
  - title: WITH clause（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_with.html
    author: SQLite Consortium
    published: 
  - title: Core Functions（SQLite 文档）（延伸）
    url: https://www.sqlite.org/lang_corefunc.html
    author: SQLite Consortium
    published: 
related: [anthropic-11, anthropic-12, rag-01, anthropic-08, coding-09]
updated: 2026-09-28
---

## 一句话答案

> 标准解法是**在同一张订单明细表上做自连接**，并且用两个技巧：
> ① **`a.product_id < b.product_id`**——它同时完成「去重同一对的不同顺序」与「排除自己与自己配对」，比 `DISTINCT + LEAST/GREATEST` 更省事；
> ② **每个订单内先去重商品**——同一订单里同一商品出现多行（数量 > 1）会让计数虚高，所以要先 `SELECT DISTINCT order_id, product_id`。
> 骨架：
> ```sql
> WITH items AS (SELECT DISTINCT order_id, product_id FROM order_items)
> SELECT a.product_id AS p1, b.product_id AS p2, COUNT(*) AS pair_count
> FROM items a JOIN items b
>   ON a.order_id = b.order_id AND a.product_id < b.product_id
> GROUP BY a.product_id, b.product_id
> ORDER BY pair_count DESC, p1, p2      -- 平局要有确定性顺序
> LIMIT 5;
> ```
> 进阶要点（ interviewer 通常会追问）：**支持度/置信度**（关联规则里「一起买」的频率要按订单总数或单品频次归一化）、**高频单品会霸榜**（可先过滤长尾/头部）、**性能**（索引与组合爆炸）、以及**平局的确定性排序**。

## 面试官在考什么

- **是否处理「同订单同商品多行」**：这是最常见的脏数据陷阱；不 `DISTINCT` 会让「买 3 瓶牛奶 + 1 个面包」的计数变成 3。
- **是否用 `a < b` 而非 `a != b`**：`!=` 会同时产生 (A,B) 与 (B,A) 两组（计数各一半），或者需要额外 `DISTINCT/GROUP BY` 归一化，容易写错。
- **平局与确定性**：`ORDER BY pair_count DESC` 在并列时不保证顺序——加上 `p1, p2` 才能得到稳定结果（评审与回归测试都需要确定性）。
- **归一化口径**：只说「一起出现的次数」通常不够，要能说出 support（出现次数 / 订单总数）与 confidence（$P(B\mid A)$）的区别与用途（推荐场景看 confidence，热门商品场景看 lift）。
- **性能意识**：自连接是 $O(\sum_i k_i^2)$（$k_i$ 为订单内商品种类数）；大篮子订单（几百个 SKU）会让组合数爆炸，需要阈值或采样。
- **能否区分「一起买」与「因果关系」**：一起买是相关性；促销、品类结构、场景（例如电池与手电筒）都会制造共现。

**常见错误答案**

- `SELECT a.product, b.product, COUNT(*) FROM items a JOIN items b ON a.order_id = b.order_id WHERE a.product != b.product`——顺序重复计数，且会把自己与自己排除不干净。
- 忘记 `DISTINCT`，让数量字段污染计数。
- 只给 `LIMIT 5` 而没有确定性排序。
- 用 `MAX(count)` 或先取 top-1 再循环——把问题做成多次查询（面试里会被问「能不能一条 SQL 解决」）。

## 原理与推导

### 1. 组合计数的定义与顺序归一化

设订单集合 $\mathcal{O}$，订单 $o$ 的商品集合 $I_o$（**集合**而非多重集——这是 `DISTINCT` 的语义依据）。商品对 $(x,y)$ 的共现次数：

$$\text{count}(x,y)=\big|\{o\in\mathcal{O}: x,y\in I_o\}\big|,\qquad (x,y)\ \text{按字典序固定}$$

`a.product_id < b.product_id` 正是把「无序对」映射到唯一有序表示，避免 $2\times$ 重复与自配对。

### 2. 关联规则的三个标准量

| 量 | 定义 | 用途 |
| --- | --- | --- |
| support | $\text{count}(x,y)/|\mathcal{O}|$ | 过滤噪声（太罕见不可信） |
| confidence | $\text{count}(x,y)/\text{count}(x)$ | 推荐「买了 X 的人还买了 Y」 |
| lift | $\text{confidence}/\text{support}(y)$ | 去掉热门商品的天然高频（lift>1 才有正向关联） |

**为什么必须提 lift**：热门商品（例如「塑料袋」）与任何东西的共现都高，confidence 也不低，但 lift≈1 说明**没有额外信息量**。面试里主动提 lift，说明你理解「共现 ≠ 关联」。

### 3. 性能：组合爆炸与索引

- **计算量**：$\sum_o \binom{k_o}{2}$；若某些订单有 500 个 SKU，单订单就产生 12 万对。
- **控制手段**：① 限制订单内商品数（超出则采样或跳过）；② 先按 support 剪枝（只考虑出现次数 ≥ 阈值的单品，即 Apriori 的剪枝思想）；③ 对高频商品做分层统计。
- **索引**：`(order_id, product_id)` 复合索引让自连接按订单分段扫描；若表很大，先物化 `DISTINCT` 的明细到临时表（CTE 在多数引擎里会被优化，但物化更可控）。

### 4. 边界与陷阱

| 陷阱 | 后果 | 处理 |
| --- | --- | --- |
| 同订单同商品多行 | 计数虚高 | `SELECT DISTINCT` |
| 退单/取消的订单 | 计入无效共现 | 先按状态过滤 |
| 同一订单跨天拆单 | 拆成多单后共现丢失 | 用「购物篮 id」而不是订单号 |
| 平局 | 结果不稳定 | `ORDER BY count DESC, p1, p2` |
| NULL product_id | 产生伪配对 | `WHERE product_id IS NOT NULL` |

## 数值与代码验证

### 表 1：三种写法的对比（示例数据：6 个订单）

| 写法 | 是否顺序去重 | 是否排除自配对 | 是否需要额外 `DISTINCT` | 结果正确性 |
| --- | --- | --- | --- | --- |
| `a != b` | ✗（(A,B) 与 (B,A) 各计一次） | ✓ | 需要（再归一化） | 易错 |
| `a < b` | ✓ | ✓ | 不需要 | **推荐** |
| `LEAST/GREATEST` + `DISTINCT` | ✓ | ✓ | 需要 | 可读但更啰嗦 |

### 表 2：本文件的示例数据（可复现）

| 订单 | 商品 |
| --- | --- |
| 1 | A, B, C |
| 2 | A, B |
| 3 | A, B, D |
| 4 | B, C |
| 5 | A, C |
| 6 | A, B |
| 7 | A, A, B（**脏数据**：同一订单同一商品重复行） |

共 **7 个订单**。期望结果：`(A,B)=5`、`(A,C)=2`、`(B,C)=2`、`(A,D)=1`、`(B,D)=1`——其中 `(A,C)` 与 `(B,C)` 并列，靠 `p1,p2` 决定顺序。注意 (A,B) 出现在订单 1/2/3/6/7 共 **5** 次（这正是手算容易漏掉订单 7 的地方）。

### 可运行代码

```python
# 用 sqlite3 实跑：建表 -> 造数 -> 查询 -> 带 support/confidence/lift 的扩展版本
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE order_items(order_id INT, product_id TEXT);
CREATE INDEX idx_oi ON order_items(order_id, product_id);
INSERT INTO order_items VALUES
 (1,'A'),(1,'B'),(1,'C'),
 (2,'A'),(2,'B'),
 (3,'A'),(3,'B'),(3,'D'),
 (4,'B'),(4,'C'),
 (5,'A'),(5,'C'),
 (6,'A'),(6,'B'),
 (7,'A'),(7,'A'),(7,'B');      -- 脏数据：同一订单同一商品重复行
""")

BASE = """
WITH items AS (SELECT DISTINCT order_id, product_id FROM order_items)
SELECT a.product_id AS p1, b.product_id AS p2, COUNT(*) AS pair_count
FROM items a JOIN items b
  ON a.order_id = b.order_id AND a.product_id < b.product_id
GROUP BY a.product_id, b.product_id
ORDER BY pair_count DESC, p1, p2
LIMIT 5
"""
print("① 基础版（a < b + DISTINCT）：")
for row in cur.execute(BASE):
    print("   ", row)

print("\n② 对照：用 a != b 会怎样（同一对出现两次、计数各半）：")
for row in cur.execute("""
WITH items AS (SELECT DISTINCT order_id, product_id FROM order_items)
SELECT a.product_id, b.product_id, COUNT(*) AS c
FROM items a JOIN items b ON a.order_id=b.order_id AND a.product_id <> b.product_id
GROUP BY a.product_id, b.product_id ORDER BY c DESC, 1, 2 LIMIT 6
"""):
    print("   ", row)

print("\n③ 脏数据检验：不做 DISTINCT 时 (A,B) 会被多算几次")
for row in cur.execute("""
SELECT a.product_id, b.product_id, COUNT(*) AS c
FROM order_items a JOIN order_items b
  ON a.order_id=b.order_id AND a.product_id < b.product_id
GROUP BY 1,2 HAVING a.product_id='A' AND b.product_id='B'
"""):
    print("   未去重 (A,B) 计数 =", row[2])
for row in cur.execute("""
WITH items AS (SELECT DISTINCT order_id, product_id FROM order_items)
SELECT COUNT(*) FROM items a JOIN items b
  ON a.order_id=b.order_id AND a.product_id<b.product_id
 WHERE a.product_id='A' AND b.product_id='B'
"""):
    print("   去重后   (A,B) 计数 =", row[0])

print("\n④ 扩展：support / confidence / lift（推荐场景的口径）")
for row in cur.execute("""
WITH items AS (SELECT DISTINCT order_id, product_id FROM order_items),
     n AS (SELECT COUNT(DISTINCT order_id) AS total FROM order_items),
     pairs AS (SELECT a.product_id p1, b.product_id p2, COUNT(*) c
               FROM items a JOIN items b ON a.order_id=b.order_id AND a.product_id<b.product_id
               GROUP BY 1,2),
     single AS (SELECT product_id, COUNT(*) c FROM items GROUP BY product_id)
SELECT p.p1, p.p2, p.c,
       ROUND(1.0*p.c/(SELECT total FROM n), 3)                AS support,
       ROUND(1.0*p.c/s1.c, 3)                                 AS confidence_1to2,
       ROUND((1.0*p.c/s1.c)/(1.0*s2.c/(SELECT total FROM n)), 3) AS lift
FROM pairs p
JOIN single s1 ON s1.product_id = p.p1
JOIN single s2 ON s2.product_id = p.p2
ORDER BY p.c DESC, p.p1, p.p2
LIMIT 5
"""):
    print("   ", row)
con.close()
```

预期输出要点（实跑）：① 基础版给出 `(A,B)=5, (A,C)=2, (B,C)=2, (A,D)=1, (B,D)=1`（并列用 `p1,p2` 定序）；② `a != b` 的版本让每一对**出现两次**（`(A,B)` 与 `(B,A)` 各计 5），顺序未归一化；③ **脏数据检验**：订单 7 里 A 出现两次，不做 `DISTINCT` 会把 (A,B) 从 **5 抬到 6**——这一行就是「为什么必须 DISTINCT」的实证；④ 扩展版给出 support/confidence/lift（订单总数 7）：热门单品（A 出现 6 次）让 `(A,B)` 的 lift 只有 0.97（**小于 1，说明共现里没有额外信息量**），而稀有的 `(A,D)` lift 反而 1.17。

## 常见追问

- **追问**：如果还要「前五对」以外的信息，比如每个商品的 top-3 搭档？
  - 要点：用窗口函数 `ROW_NUMBER() OVER (PARTITION BY p1 ORDER BY c DESC)` 取每组前 3（SQLite 支持窗口函数，见对应文档）。
- **追问**：数据量到十亿行怎么办？
  - 要点：先剪枝（只保留 support ≥ 阈值的单品）、按订单分段并行（MapReduce 式：map 出对、reduce 计数）、或改用专用频繁项集算法（Apriori/FP-Growth）；纯 SQL 自连接在超大表上不可行。
- **追问**：怎么处理「同一商品在不同订单里被拆成不同 SKU」？
  - 要点：先做商品归一化（映射到标准品 id）再计数，否则共现被稀释——这是数据建模问题，不是 SQL 问题。
- **追问**：`LIMIT 5` 在并列很多时结果不稳定，怎么保证可复现？
  - 要点：显式加次级排序键（`p1, p2`），或先算完整排名再过滤（`ROW_NUMBER`）。
- **追问**：时间衰减怎么加（最近一起买更重要）？
  - 要点：把计数换成加权和（`SUM(exp(-Δdays/τ))` 或分段权重），并在 SQL 里用订单时间做权重——注意此时 `COUNT(*)` 不再适用。
- **追问**：怎么验证查询正确？
  - 要点：构造小数据集手算 + 断言（本文件的表 2）；再用「反向检查」（用 `a>b` 的版本应得到相同计数）。

## 相关题目

- [[anthropic-11]]：区间重叠检测，与本题同属「自连接 + 条件归一化」的题型。
- [[anthropic-12]]：ETL 脏数据下的「当前值」查询，与本题的「脏数据先清洗」同一主题。
- [[rag-01]]：检索里的切分与去重，与本题「集合语义 vs 多重集语义」的区分相通。
- [[anthropic-08]]：大规模批处理的并发与幂等，对应本题「十亿行怎么办」的分布式解法。
- [[coding-09]]：字符串/序列处理与去重的一般手法。

## 参考资料与归属

- **SELECT（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_select.html>。第 1 节 `JOIN`/`GROUP BY`/`ORDER BY`/`LIMIT` 的语义与执行顺序来自这份文档。
- **WITH clause（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_with.html>。可运行代码里 CTE（`WITH items AS (...)`)的写法与作用域来自这份文档。
- **Core Functions（SQLite 文档）（延伸）** —— SQLite Consortium：<https://www.sqlite.org/lang_corefunc.html>。扩展查询里 `ROUND` 等函数的语义来自这份文档。
- **延伸来源说明**：表 1、表 2 以及示例数据（6+1 个订单、商品 A–D、脏数据订单 7）都是为演示本仓库口径而构造的数据集，support/confidence/lift 的定义是关联规则分析的标准定义（未逐条引用原始论文）；来源仅用于 SQL 语义与函数行为的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
