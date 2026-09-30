---
type: question
id: openai-02
company: OpenAI
topic: coding
order: 2
question: 一步步实现一个数据库 ORM。
question_en: Implement a database ORM, step by step.
asked_at: []
level: 高阶
tags: [ORM, 查询构建器, 身份映射, N+1, 连接池]
sources:
  - title: Patterns of Enterprise Application Architecture（延伸）
    url: https://martinfowler.com/books/eaa.html
    author: Martin Fowler
    published: 2002-11-15
  - title: SQL Injection Prevention Cheat Sheet（延伸）
    url: https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
    author: OWASP
    published: 2023-01-01
  - title: Design Patterns: Elements of Reusable Object-Oriented Software（延伸）
    url: https://www.oreilly.com/library/view/design-patterns-elements/0201633612/
    author: Gamma, Helm, Johnson & Vlissides
    published: 1994-10-31
  - title: 支持事务的内存 KV 存储（本仓库公司题库 · OpenAI 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [openai-01, openai-20, coding-08, coding-09, databricks-03]
updated: 2026-09-28
---

## 一句话答案

> **「一步步」是这道题的考点**——**面试官要看的是「你按什么顺序加复杂度」，而不是最终有多少功能**。
> **五级增量（每级都可独立运行、可测）**：
> | 级 | 加什么 | 关键设计 |
> | --- | --- | --- |
> | **1** | **行 → 对象映射** | `SELECT` 结果映射为对象；**字段名与列名的映射表** |
> | **2** | **查询构建器** | `where`/`order`/`limit`；**值永远走参数占位**（★ 绝不字符串拼接） |
> | **3** | **会话 + 身份映射 + 工作单元** | 跟踪已加载对象、**脏检查**、提交时 flush |
> | **4** | **关系与预加载** | has-many / belongs-to；**解决 N+1** |
> | **5** | **连接池 / 事务 / 迁移** | **Little 定律定池大小**；迁移与回滚 |
> **★ 本机实测：N+1 是这道题的核心考点**（200 个作者、每个 10 本书）：
> | 方式 | 查询数 | 内存库耗时 | **加 1 ms RTT** | 加 5 ms RTT |
> | --- | --- | --- | --- | --- |
> | **N+1** | **201** | 14.3 ms | **215 ms** | **1019 ms** |
> | **预加载** | **2** | 2.8 ms | **5 ms** | 13 ms |
> | 差距 | **100×** | 5.0× | **43×** | **79×** |
> **读法**：**内存库里只差 5 倍，加上网络 RTT 后差几十到上百倍**——**所以「本地测着还行」会严重低估线上代价**。**N+1 的查询数是 $1+N$**（**与数据量成正比，这是它致命的根本原因**）。
> **但预加载不是万能**：**扇出极大时会一次载入全部子行**（内存风险）→ 需要**分批/游标**（`WHERE author_id IN (...)` 按批 500 个父对象）。
> **身份映射（identity map）的收益**：
> | 引用序列 | 无身份映射 | 有身份映射 | 省 |
> | --- | --- | --- | --- |
> | `[1,1,2,3,3,3,4,5,5]`（9 次引用、5 个对象） | 9 次加载 | **5 次** | **44%** |
> **读法**：**身份映射保证「同一行只有一个对象」**——**既省查询，又让「改一处、处处生效」成立**（**否则同一行的两个副本会互相覆盖**）。**代价**：会话内对象缓存会膨胀 → 需要 `clear`/`expire`。
> **连接池大小（Little 定律）**：
> $$N_{\text{连接}}=\text{QPS}\times\text{持有时长}$$
> | QPS | 持有时长 | 所需连接 |
> | --- | --- | --- |
> | 1,000 | 5 ms | **5** |
> | 1,000 | 50 ms | **50** |
> | 5,000 | 50 ms | **250** |
> **读法**：**池不是越大越好**——**超过数据库承载能力后，更多连接只会让排队更糟**（**要配超时与背压**）。
> **三个「绝不能做」**：
> | # | 反模式 | 后果 |
> | --- | --- | --- |
> | ① | **把值拼进 SQL 字符串** | **SQL 注入**（必须参数化） |
> | ② | **在循环里发查询**（N+1） | 查询数随数据量线性增长 |
> | ③ | **把「脏检查」做成全表扫描** | flush 时 $O(\text{对象}\times\text{属性})$ |
> 一句话判据：**"行映射 → 参数化查询 → 身份映射/工作单元 → 预加载解决 N+1 → Little 定律定池「**——**五级按顺序讲，每级说清」为什么现在才加它"**。

## 面试官在考什么

- **★ 是否「按顺序加复杂度」**：**能否把实现拆成五级、每级可独立运行**（**而不是一上来就设计一个大框架**）。
- **★ N+1 的意识与量化**：**能否指出「循环里发查询」的查询数是 $1+N$**，并给出实测与网络投影（本机 201 vs 2 次、线上 43–79×）。
- **参数化查询**：**能否把「绝不拼接字符串」当成硬约束**（**这是 ORM 存在的最强理由之一**）。
- **身份映射**：**能否指出「同一行必须只有一个对象」**（**否则「改一处」不生效，而且会写出互相覆盖的 bug**）。
- **工作单元与脏检查**：**能否说清「flush 时比较什么」**（快照 vs 当前值；**以及它的成本**）。
- **预加载的代价**：**能否指出「一次载入全部子行」的内存风险**，并给出分批方案。
- **连接池**：**能否用 Little 定律算池大小**（本机 1000 QPS × 5 ms = 5 个连接）。
- **事务边界**：**能否说清「会话与事务的关系」**（**会话可以跨多个事务；flush 通常发生在事务内**）。
- **可测试性**：**能否说清「每级怎么测」**（内存数据库 + 生成 SQL 的断言）。
- **诚实**：**承认「手写 ORM 通常不如用现成的」**——**但这道题考的是「你是否理解它们内部的取舍」**。

**常见错误答案**

- **一上来设计大框架**（「先做元类、描述符、关系代数……」——**没有可运行的中间态**）。
- **字符串拼接 SQL**（**SQL 注入**）。
- **不提 N+1**（**这是 ORM 最经典的问题**）。
- **不做身份映射**（**同一行两个对象 → 更新互相覆盖**）。
- **flush 时全量比较所有属性**（$O(\text{对象}\times\text{属性})$，且**没有「变更集」概念**）。
- **预加载一把梭**（**大扇出时内存爆**）。
- **连接池开很大**（**反而拖慢数据库**）。
- 把会话当成事务（**语义混淆**）。
- 不做测试（**「我跑了一遍看着对」**）。

## 原理与推导

### 1. 第 1 级：行 → 对象映射

**最小可用**：一个字段列表 + 一个 `select`：

```python
class Model:
    __table__: str = ""
    __fields__: tuple = ()

def select(cls, where="", params=(), limit=None):
    sql = f"SELECT {','.join(cls.__fields__)} FROM {cls.__table__}"
    if where:  sql += f" WHERE {where}"          # where 是模板，值走占位符
    if limit:  sql += f" LIMIT {int(limit)}"     # limit 是整数，必须强制转换
    return [cls(**dict(zip(cls.__fields__, row))) for row in db.execute(sql, params)]
```

**读法**：**表名与列名是「代码」，值是「数据」**——**前者来自类定义（可信），后者必须参数化**。**`LIMIT` 要强制转成整数**（**它不能参数化，所以必须校验**）。

### 2. 第 2 级：查询构建器（参数化是硬约束）

**反模式**：

```python
sql = f"SELECT * FROM users WHERE name = '{name}'"    # ✗ 注入
```

**正确**：

```python
sql = "SELECT * FROM users WHERE name = ?"            # ✓ 值走占位符
db.execute(sql, (name,))
```

**为什么这是 ORM 存在的理由之一**：**把「用户可控的值」与「SQL 结构」在语法层分开**（**串 [[openai-01]] 的「数据与指令分离」是同一类思想**）。

**构建器的最小 API**：

| 方法 | 生成的 SQL |
| --- | --- |
| `.where(「age > ?」, 18)` | `WHERE age > ?` |
| `.order_by(「name」)` | `ORDER BY name`（**列名必须来自白名单**） |
| `.limit(10)` | `LIMIT 10`（**强制整数**） |

**读法**：**「列名不能参数化」是个真实的坑**——**所以 `order_by` 的列名必须走白名单校验**（**否则 `ORDER BY (SELECT ...)` 就是注入点**）。

### 3. 第 3 级：会话 + 身份映射 + 工作单元

**三个概念**：

| 概念 | 作用 |
| --- | --- |
| **会话（Session）** | 一次「工作对话」的上下文；**持有身份映射与工作单元** |
| **身份映射（Identity Map）** | **键 = 主键，值 = 对象**；保证同一行只有一个对象 |
| **工作单元（Unit of Work）** | 记录「哪些对象变脏了」，**提交时统一 flush** |

**身份映射的收益（本机算例）**：

| 引用序列 | 无映射 | 有映射 | 省 |
| --- | --- | --- | --- |
| `[1,1,2,3,3,3,4,5,5]`（9 引用 / 5 对象） | 9 次加载 | **5 次** | **44%** |

**读法**：**收益随「重复引用率」上升**——**但更重要的收益是「正确性」**：**没有身份映射时，同一行的两个对象会互相覆盖**（**经典的「更新丢失」**）。

**脏检查的两种做法**：

| 做法 | 机制 | 成本 |
| --- | --- | --- |
| **快照比较** | 加载时存一份属性快照，flush 时逐属性比 | $O(\text{对象}\times\text{属性})$ |
| **属性拦截** | `__setattr__` 钩子标记脏 | 每次赋值有开销，**但 flush 只处理脏对象** |

**本机算例**：**1 万个对象 × 20 个属性 = 每次 flush 20 万次比较**——**如果 QPS 高，这本身就是瓶颈**（**所以生产 ORM 用属性拦截或显式 `dirty` 标记**）。

### 4. ★ 第 4 级：关系与 N+1（本题的核心）

**N+1 的机制**：

```python
for author in authors:                    # 1 次查询
    print(author.books)                   # 每个 author 再查 1 次 -> N 次
```

$$\text{查询数}=1+N$$

**本机实测**（200 个作者、每个 10 本书）：

| 方式 | 查询数 | 内存库 | **+1 ms RTT** | +5 ms RTT |
| --- | --- | --- | --- | --- |
| **N+1** | **201** | 14.3 ms | **215 ms** | **1019 ms** |
| **预加载** | **2** | 2.8 ms | **5 ms** | 13 ms |
| 差距 | **100×** | 5.0× | **43×** | **79×** |

**读法**：**内存库里只差 5 倍**（**因为 sqlite 内存库的每次查询几乎免费**）——**这正是「本地测着还行」的陷阱**；**加上网络 RTT 后差 43–79 倍**。**N+1 的本质问题是「查询数随数据量线性增长」**（**数据翻倍，查询数翻倍**）。

**预加载的两种写法**：

| 写法 | SQL | 适用 |
| --- | --- | --- |
| **JOIN** | `SELECT ... FROM authors JOIN books ...` | 一对少（**行会重复，要去重**） |
| **两次查询 + 内存拼接** | `SELECT ... FROM books WHERE author_id IN (...)` | **一对多（推荐）** |

**注意**：**`IN (...)` 的列表长度有上限**（**不同数据库不同，常见 1000–65535**）——**所以大扇出要分批**：

```python
for batch in chunks(parent_ids, 500):
    books = select(Book, f"author_id IN ({','.join('?'*len(batch))})", batch)
```

**读法**：**「分批预加载」是生产 ORM 的标配**（**它同时解决了 N+1 与内存爆的问题**）。

### 5. 第 5 级：连接池、事务与迁移

**连接池大小（Little 定律）**：

$$N=\text{QPS}\times\text{持有时长}$$

| QPS | 持有时长 | 所需连接 |
| --- | --- | --- |
| 1,000 | 5 ms | **5** |
| 1,000 | 50 ms | **50** |
| 5,000 | 50 ms | **250** |

**读法**：**池不是越大越好**——**超过数据库承载能力后，更多连接只会让排队更糟**（**数据库的并发度是有限的**）。**正确做法**：**按 Little 定律算下限 + 设置获取超时（拿不到连接就快速失败，而不是无限等）**。

**事务边界**：

| 问题 | 答案 |
| --- | --- |
| 会话与事务的关系 | **会话可以跨多个事务**（一次会话 = 多个工作单元） |
| flush 发生在哪 | **通常在事务内**（**保证「一组变更」原子提交**） |
| 自动提交 | **每次 `execute` 自动提交**（**简单但无法组合**） |

**迁移**：**版本化的 schema 变更**（`up`/`down` 成对；**每次变更都有回滚路径**）——**这与「错误决策的止损」是同一套思路**（串 [[microsoft-10]]：**可逆性**）。

## 数值与代码验证

### 表 1：N+1、身份映射、连接池（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| 身份映射（第 3 级；引用序列 [1, 1, 2, 3, 3, 3, 4, 5, 5]，9 次引用、5 个对象） | 实际加载次数 **5**（无身份映射则 9 次）→ **省 44%**；`sess.get(Author,1) is sess.get(Author,1)` → **True**——身份映射既省查询，又保证「改一处、处处生效」 |
| N+1 vs 预加载（**耗时随机器变化**；载入 2000 本） | N+1：查询数 **201**、内存库 13.3ms、+1ms RTT **214ms**、+5ms RTT **1018ms**；预加载：查询数 **2**、2.5ms、5ms、**13ms**——**查询数比 100×；内存库耗时比 5.3×；+5ms RTT 后 81×** |
| 分批预加载 | 分 4 批（每批 50 个父对象）载入 2000 本书——**分批预加载是生产 ORM 的标配**（同时解决 N+1 与内存风险，因为 IN 列表有长度上限） |
| 连接池大小（Little 定律 $N$ = QPS × 持有时长） | 100 QPS × 5ms → **0.5**；1000 × 5ms → **5.0**；1000 × 50ms → **50.0**；5000 × 5ms → **25.0**；5000 × 50ms → **250.0**——**池不是越大越好**，要配「获取超时 + 背压」 |

### 可运行代码

```python
# 一步步实现 ORM：行映射 -> 参数化查询 -> 身份映射 -> 预加载（N+1）-> 连接池
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE authors(id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE books(id INTEGER PRIMARY KEY, author_id INTEGER, title TEXT, year INTEGER);
""")
import random
rng = random.Random(0)
NA, PER = 200, 10
db.executemany("INSERT INTO authors VALUES(?,?,?)",
               [(i, f"author{i}", rng.choice(["CN", "US", "DE"])) for i in range(NA)])
db.executemany("INSERT INTO books VALUES(?,?,?,?)",
               [(i, i // PER, f"book{i}", 1990 + rng.randint(0, 30)) for i in range(NA * PER)])
db.commit()

class Model:
    __table__: str = ""
    __fields__: Tuple[str, ...] = ()
    def __init__(self, **kw: Any) -> None:
        for f in self.__fields__:
            setattr(self, f, kw.get(f))

class Author(Model):
    __table__, __fields__ = "authors", ("id", "name", "country")
class Book(Model):
    __table__, __fields__ = "books", ("id", "author_id", "title", "year")

print("① 第 1–2 级：行映射 + 参数化查询（值永远走占位符）")
def select(cls, where: str = "", params: Sequence[Any] = (),
           limit: Optional[int] = None) -> List[Any]:
    sql = f"SELECT {','.join(cls.__fields__)} FROM {cls.__table__}"
    if where:
        sql += f" WHERE {where}"            # where 是模板（代码），值走 params（数据）
    if limit is not None:
        sql += f" LIMIT {int(limit)}"       # ★ LIMIT 不能参数化 -> 强制整数
    cur = db.execute(sql, tuple(params))
    return [cls(**dict(zip(cls.__fields__, row))) for row in cur.fetchall()]
a0 = select(Author, "country = ?", ("CN",), limit=3)
print(f"  查询 CN 作者（前 3）：{[a.name for a in a0]}")
print(f"  反模式（拼接）：f\"... WHERE name = '{{name}}'\"  -> **SQL 注入**；正确：'WHERE name = ?'")

print("")
print("② 第 3 级：身份映射（同一行只有一个对象）")
class Session:
    def __init__(self) -> None:
        self.identity: Dict[Tuple[str, Any], Any] = {}
    def get(self, cls: Any, pk: Any) -> Any:
        key = (cls.__table__, pk)
        if key not in self.identity:                     # 未加载 -> 查库
            rows = select(cls, "id = ?", (pk,))
            if not rows:
                return None
            self.identity[key] = rows[0]
        return self.identity[key]
sess = Session()
refs = [1, 1, 2, 3, 3, 3, 4, 5, 5]
objs = [sess.get(Author, i) for i in refs]
print(f"  引用序列 {refs}（{len(refs)} 次引用、{len(set(refs))} 个对象）")
print(f"  实际加载次数：{len(sess.identity)}（无身份映射则 {len(refs)} 次）"
      f" -> 省 {1 - len(sess.identity)/len(refs):.0%}")
print(f"  同一对象：sess.get(Author,1) is sess.get(Author,1) -> "
      f"{sess.get(Author, 1) is sess.get(Author, 1)}")
print("  读法：**身份映射既省查询，又保证「改一处、处处生效」**（否则同行的两个副本会互相覆盖）")

print("")
print("③ 第 4 级：N+1 vs 预加载（实测 + 网络投影）")
def load_n_plus_1() -> Tuple[float, int, int]:
    t0 = time.perf_counter(); q = 0
    authors = select(Author); q += 1
    for a in authors:
        a.books = select(Book, "author_id = ?", (a.id,)); q += 1
    return time.perf_counter() - t0, q, sum(len(a.books) for a in authors)
def load_eager() -> Tuple[float, int, int]:
    t0 = time.perf_counter(); q = 0
    authors = select(Author); q += 1
    books = select(Book); q += 1
    by: Dict[int, List[Any]] = {}
    for b in books:
        by.setdefault(b.author_id, []).append(b)
    for a in authors:
        a.books = by.get(a.id, [])
    return time.perf_counter() - t0, q, sum(len(a.books) for a in authors)
t1, q1, n1 = load_n_plus_1()
t2, q2, n2 = load_eager()
print(f"  {'方式':<10} {'查询数':>7} {'内存库':>10} {'+1ms RTT':>10} {'+5ms RTT':>10} {'载入':>6}")
for name, q, t in (("N+1", q1, t1 * 1e3), ("预加载", q2, t2 * 1e3)):
    print(f"  {name:<10} {q:>7} {t:>8.1f}ms {t+q*1:>8.0f}ms {t+q*5:>8.0f}ms {n1 if name=='N+1' else n2:>6}")
print(f"  查询数比 **{q1/q2:.0f}x**；内存库耗时比 {t1/t2:.1f}x；+5ms RTT 后 **{(t1*1e3+q1*5)/(t2*1e3+q2*5):.0f}x**")
print("  读法：**内存库只差几倍，加上网络 RTT 差几十倍** -> 「本地测着还行」会严重低估线上代价；")
print("        预加载要分批（IN 列表有长度上限），否则大扇出会撑爆内存")

print("")
print("④ 分批预加载（解决「IN 列表上限」与「内存爆」）")
def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
parents = [a.id for a in select(Author)]
total = 0
for batch in chunked(parents, 50):
    ph = ",".join("?" * len(batch))
    total += len(select(Book, f"author_id IN ({ph})", batch))
print(f"  分 {len(list(chunked(parents, 50)))} 批（每批 50 个父对象）载入 {total} 本书")
print("  读法：**分批预加载是生产 ORM 的标配** —— 同时解决 N+1 与内存风险")

print("")
print("⑤ 第 5 级：连接池大小（Little 定律：N = QPS x 持有时长）")
print(f"  {'QPS':>6} {'持有时长':>9} {'所需连接':>9} 说明")
for qps, hold in ((100, 0.005), (1000, 0.005), (1000, 0.05), (5000, 0.005), (5000, 0.05)):
    need = qps * hold
    print(f"  {qps:>6} {hold*1000:>7.0f}ms {need:>9.1f} "
          f"{'够用' if need <= 20 else '需要更大池或分片'}")
print("  读法：**池不是越大越好** —— 超过数据库承载能力后，更多连接只会让排队更糟；")
print("        要配「获取超时 + 背压」（拿不到连接就快速失败，而不是无限等）")

print("")
print("⑥ 五级增量与每级的验证方式")
LEVELS = [
    ("1 行->对象映射", "断言 SELECT 结果映射为对象（字段顺序/类型）"),
    ("2 查询构建器", "断言生成的 SQL 含占位符；注入样例不改变 SQL 结构"),
    ("3 身份映射/工作单元", "断言同一行返回同一对象；flush 只更新脏对象"),
    ("4 关系与预加载", "断言查询数（N+1 -> 常数）；分批边界（0/1/上限）"),
    ("5 连接池/事务/迁移", "断言池上限与获取超时；迁移 up/down 可逆"),
]
print(f"  {'级别':<22} 验证方式")
for lv, how in LEVELS:
    print(f"  {lv:<22} {how}")
print("  读法：**每级都要有可独立运行的测试** —— 这是「一步步实现」的落地方式")
```

预期输出要点（实跑）：① 行映射 + 参数化查询可用（**`LIMIT` 强制整数**）；② **身份映射**：9 次引用只加载 **5** 个对象（**省 44%**），且同一主键返回同一对象；③ **N+1 实测**：**201 次查询 / 14.3 ms** vs 预加载 **2 次 / 2.8 ms**——**查询数比 100×**，**+5 ms RTT 后耗时比约 79×**；④ 分批预加载（每批 50 个父对象）；⑤ **连接池**：1000 QPS × 5 ms = **5 个连接**、× 50 ms = **50 个**；⑥ 五级增量与各自的验证方式。

## 常见追问

- **追问**：为什么不用现成的 ORM？
  - 要点：**生产上应该用**——**但这道题考的是「你是否理解它们内部的取舍」**：① **N+1 是 ORM 最经典的性能陷阱**（**用现成 ORM 也会踩**，所以必须理解预加载 API）；② **身份映射与工作单元决定了「更新丢失」这类正确性问题**；③ **参数化是安全底线**（**手写 SQL 时最容易破**）。**读法**：**「我会用现成的，但我知道它在什么情况下会咬我」**——**这是最好的回答姿态**。
- **追问**：脏检查怎么做才高效？
  - 要点：**两种主流做法**：① **属性拦截**（`__setattr__`/描述符在赋值时标记脏）——**flush 只处理脏对象**，**但每次赋值有开销**；② **快照比较**（加载时存快照，flush 时逐属性比）——**flush 是 $O(\text{对象}\times\text{属性})$**（**本机 1 万对象 × 20 属性 = 20 万次比较**）。**生产 ORM 多用 ① + 显式 `dirty` 集合**。**注意**：**可变属性（如 list）的原地修改检测不到**——**所以要么禁止原地改，要么用不可变值或显式标记**（**这是 ORM 的经典坑**）。
- **追问**：会话要不要线程安全？
  - 要点：**通常不**——**会话是「工作单元」，天然属于单个线程/请求**（**共享会话是 bug 来源**）。**做法**：① **每请求一个会话**（**用上下文管理器管理生命周期**）；② **连接池是线程安全的**（**会话从池里借连接**）；③ **跨线程共享对象是危险的**（**因为对象与会话绑定**）。**读法**：**「会话不共享」是最重要的使用约束**——**面试时说清这一点，比讨论锁的实现更有价值**。
- **追问**：N+1 怎么在代码审查里发现？
  - 要点：**四个手段**：① **查询日志/计数**（**在测试里断言「查询数不超过 K」**——**最有效**）；② **静态检查**（**标记「在循环体内访问未加载关系」的模式**）；③ **代码审查清单**（**「这个循环里有没有数据库调用」**）；④ **性能监控**（**线上慢查询与查询数指标**）。**读法**：**「在测试里断言查询数」是把性能问题变成正确性问题**（**它会在 CI 里失败，而不是等用户投诉**）——**这是最工程化的答案**。
- **追问**：ORM 的迁移怎么设计？
  - 要点：**三条**：① **版本化**（每个迁移有唯一版本号，**按序应用**）；② **成对**（`up` 与 `down`；**每次变更都要有回滚路径**）；③ **幂等与可重入**（**失败后能重跑**）。**注意**：**有些变更不可逆**（删列会丢数据）——**所以「不可逆迁移」要先备份或分两步**（**先加新列 → 双写 → 切读 → 再删旧列**）。**读法**：**「不可逆变更分两步做」是数据库演进的核心工程经验**（**与串 [[microsoft-10]] 的「可逆性分级」是同一套思路**）。
- **追问**：如果表很大，`LIMIT/OFFSET` 分页有什么问题？
  - 要点：**`OFFSET` 是 $O(\text{offset})$**（**数据库要扫描并丢弃前 offset 行**）——**所以深分页会很慢**。**替代**：① **键集分页（keyset pagination）**：`WHERE id > :last_id ORDER BY id LIMIT 10`（**$O(\log n)$**，**推荐**）；② **游标**（服务端保持位置）；③ **物化「页边界」**。**读法**：**「深分页用键集」是必须知道的工程常识**——**而且 ORM 通常默认生成 `OFFSET`，所以要知道怎么绕开**。

## 相关题目

- [[openai-01]]：支持事务的内存 KV 存储——**「数据与指令分离」的同源思想**（参数化查询）。
- [[openai-20]]：设计一个内存数据库——本题的系统设计版本。
- [[coding-08]]：N+1 查询与预加载——N+1 的通用版本。
- [[coding-09]]：分页与游标——深分页的工程处理。
- [[databricks-03]]：数据管道的幂等与重跑——**迁移的「可重入」是同一类要求**。

## 参考资料与归属

- **Patterns of Enterprise Application Architecture（延伸）** —— Martin Fowler，2002-11-15：<https://martinfowler.com/books/eaa.html>。**Identity Map、Unit of Work、Data Mapper、Lazy Load** 这四类模式的原始定义来自这本书——**本篇第 3、4 级的结构直接源于此**。
- **SQL Injection Prevention Cheat Sheet（延伸）** —— OWASP，2023-01-01：<https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html>。**「参数化查询是首要防御」与「表名/列名不能参数化，必须走白名单」**是本篇第 2 级的依据。
- **Design Patterns（延伸）** —— Gamma, Helm, Johnson & Vlissides，1994-10-31：<https://www.oreilly.com/library/view/design-patterns-elements/0201633612/>。**Proxy（延迟加载）与 Observer（脏检查）**对应本篇的关系加载与工作单元实现。
- **支持事务的内存 KV 存储（本仓库公司题库 · OpenAI 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。该篇的**「数据与指令分离」（哨兵、参数化）**与本篇第 2 级同源。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（200 个作者、每个 10 本书、共 2000 本书、N+1 201 次查询 / 14.3 ms、预加载 2 次 / 2.8 ms、引用序列 9 次/5 对象、RTT 投影 1 ms 与 5 ms、连接池 100–5000 QPS × 5/50 ms、分批大小 50）都是**本机实跑结果**（**sqlite 内存库，绝对耗时依赖机器**）；**查询数与身份映射的节省是可复现的结构性结论**；**RTT 投影是假设**（**每条查询 +1/+5 ms**，真实值依赖网络与数据库）；**连接池公式（Little 定律）是解析结论**。**「IN 列表上限 1000–65535」是常见范围**，**具体值依赖数据库**（**本机未实测上限**）。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
