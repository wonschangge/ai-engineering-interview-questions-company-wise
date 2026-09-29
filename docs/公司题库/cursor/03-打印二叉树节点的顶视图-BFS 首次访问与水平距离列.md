---
type: question
id: cursor-03
company: Cursor（Anysphere）
topic: coding
order: 3
question: 打印二叉树节点的顶视图。
question_en: Print the top view of nodes in a binary tree.
asked_at: []
level: 入门
tags: [二叉树, 树的遍历, BFS, 水平距离, 实现题, 增量维护]
sources:
  - title: Tree traversal（维基百科）（延伸）
    url: https://en.wikipedia.org/wiki/Tree_traversal
    author: Wikipedia
    published: 
  - title: collections — deque 与 OrderedDict（Python 标准库文档）（延伸）
    url: https://docs.python.org/3/library/collections.html
    author: Python Software Foundation
    published: 
related: [coding-06, coding-12, coding-10, coding-11, coding-09]
updated: 2026-09-28
---

## 一句话答案

> 顶视图 = 每一列（水平距离）中**深度最小的那个节点**，按列升序输出；二叉树的列就是「根 0、左孩子 $-1$、右孩子 $+1$」。BFS 按深度非降序出队，所以每列**首次访问到的节点必然是该列最浅的**，一趟 $O(n)$、额外空间 $O(k)$（$k \le \min(n, 2h+1)$）；DFS 前序没有这个性质，靠「首次访问就写下来」在 7 个节点的小树上就已经输出错。
> 契约要先于代码写死四件事：列函数、深度定义、并列裁决、输出顺序。并列必须指定「同列取 BFS 从左到右先访问到的那个」：9 个节点的退化树里第 $+2$ 列在深度 4 上有两个坐标完全重叠的候选（路径 L,R,R,R 与 R,L,R,R），不写死就会出现两个各自都说得通的答案。
> 还有两处必须显式：BFS 首次访问的列序是 $0,-1,+1,-2,+2,\dots$，不是从左到右，输出前必须 `sorted()`（33 列排序 0.66 µs，同树整趟 BFS 23.25 ms）；视图大小是 $O(\min(n, 2h+1))$ 的两端，$10^6$ 个节点的左链就是 $10^6$ 列、序列化约 485 万 token，所以返回必须带 `truncated` / `total_columns` / `next_offset`，而不是悄悄截断。

## 面试官在考什么

- **契约是否先于代码**：列函数、深度、并列裁决、输出顺序这四项有没有在写代码前定死；退化树上的并列候选是最容易被口头带过的地方。
- **能不能说清「为什么 BFS 可以、DFS 不行」的前提**：BFS 的出队深度非降（分层遍历的定义性质），DFS 的访问顺序按子树走。
- **是否把容器顺手给的顺序当成业务顺序**：Python 3.7+ 的 dict 保插入序，而 BFS 首次访问序是 $0,-1,+1,-2,+2,\dots$；这一条与 LRU/TTL 题里 dict 没有 `move_to_end` 是同一类坑。
- **两种遍历的爆炸点是否分得清**：BFS 峰值 = 最宽一层（完美树取到 $2^h$），DFS 峰值 = 树高，而 Python 默认 `sys.getrecursionlimit()` 只有 1000。
- **复杂度与产品账**：时间 $O(n)$、额外空间 $O(k)$；编辑器里真正的成本是每按键的重复遍历，而不是单次遍历；真实树是 n-ary，列函数必须进契约。

常见错误答案：

- DFS 前序 + 首次访问就写下来。7 个节点的树里第 $+2$ 列会被深度 4 的 6 占住，真正的顶是深度 2 的 7。
- 把 `dict` 的值列表直接当输出（那是 BFS 访问序，不是列升序），或者默认「视图总是很小」，在左链上返回 $n$ 列把整棵树搬进上下文。

## 原理与推导

### 1. 契约先行：四件事在写代码前定死

| 契约项 | 二叉树的默认取值 | 不写死会发生什么 |
| --- | --- | --- |
| 列函数 | 根 0，左孩子 $-1$，右孩子 $+1$ | n-ary 树（AST、文件树、符号表）没有 $\pm 1$；按子节点序号还是按字符偏移，同一棵树会给出不同视图 |
| 深度定义 | 根为 0，向下每层 $+1$ | 「顶」的口径漂移，评测基线与缓存键对不上 |
| 并列裁决 | 同列取 BFS 从左到右先访问到的那个 | 9 节点退化树上 $+2$ 列有两个深度相同、坐标重叠的候选，两个实现给出不同值 |
| 输出顺序 | 按列升序 | dict 只保插入序，而 BFS 首次访问序是 $0,-1,+1,-2,+2,\dots$ |

前两项决定「答案是什么」，后两项决定「两个都对的实现是否给出同一个答案」。面试现场先把这张表说出来，再开始写。

### 2. BFS 的「首次访问即答案」是结构保证

BFS 队列不变式：任一时刻队内节点的深度只有两种取值，$d$ 与 $d+1$（$d$ 为队首深度），所以**出队顺序的深度非降**。

反证：设 $v$ 是列 $c$ 上第一个被记下的节点，深度 $d(v)$。若同列存在 $w$ 且 $d(w) < d(v)$，则 $w$ 的出队时刻早于 $v$，列 $c$ 会先被 $w$ 占住，与假设矛盾。因此每个列第一次被写入时拿到的一定是最小深度，实现里只需要一次 $O(1)$ 的「这个列出现过吗」判断，**不需要任何深度比较**。

同一深度内的先后由入队顺序决定（左孩子先入队即从左到右），这正是并列裁决可以写成「先访问到的那个」的原因。

### 3. DFS 前序没有这个性质：两个最小反例

**反例一（7 个节点）**：根 1 的左孩子 2 一路向右到 6（$1 \to 2 \to 4 \to 5 \to 6$），右孩子 3 向右到 7。前序（先左后右）在 $+2$ 列先遇到深度 4 的 6，而真正的顶是深度 2 的 7。

| 列 | BFS 首次访问 | 深度 | DFS 前序首次访问 | 深度 |
| --- | --- | --- | --- | --- |
| $-1$ | 2 | 1 | 2 | 1 |
| $0$ | 1 | 0 | 1 | 0 |
| $+1$ | 3 | 1 | 5 | 3 |
| $+2$ | **7** | 2 | **6** | 4 |

**反例二（9 个节点，并列）**：路径 L,R,R,R 到节点 9、路径 R,L,R,R 到节点 5，两条路径的列都是 $+2$、深度都是 4，而 $+2$ 列在深度 2 上没有节点。此时「深度最小」本身不定解，必须补上「从左到右先访问到的」：BFS 在深度 4 先遇到左子树里的 9，若 DFS 先走右子树则会给出 5。同一棵树里 DFS 还会把 $+1$ 列写成深度 3 的 8，正确答案是深度 1 的 2。

要保留 DFS，唯一的修正是同时维护「每列当前最小深度」并在访问时比较：多一次 dict 查找与比较，实测比 BFS 慢 15%–28%（见下表）。两侧的差距不在常数，而在 BFS 侧根本不需要这个判断——这就是「为什么 BFS 可以、DFS 不行」的完整答案。

| h | n | 列数 | BFS 首次访问 | ns/node | DFS + 每列最小深度 | ns/node | 慢 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | 65,535 | 31 | 11.27 ms | 172 | 14.47 ms | 221 | 28% |
| 16 | 131,071 | 33 | 23.25 ms | 177 | 28.81 ms | 220 | 24% |
| 17 | 262,143 | 35 | 41.83 ms | 160 | 50.92 ms | 194 | 22% |
| 18 | 524,287 | 37 | 89.24 ms | 170 | 102.49 ms | 196 | 15% |

### 4. 输出顺序必须显式排序

深度 $d$ 上可能出现的列是 $\{-d, -d+2, \dots, d\}$（与 $d$ 同奇偶），所以 BFS 第一次遇到新列的顺序是 $0, -1, +1, -2, +2, -3, +3, \dots$：先按 $|c|$ 递增，同 $|c|$ 先负后正。Python 3.7+ 的 dict 保插入序，于是 `list(seen.values())` 给出的是 BFS 访问序，不是答案要求的列升序。

`sorted()` 的成本可以忽略：33 列实测 0.66 µs，同一棵 h=16 的树整趟 BFS 23.25 ms，排序占 0.0028%。没有任何理由在这里省一次排序。

这与仓库 LRU/TTL 那道题同源：3.7+ 的普通 dict 保插入序，却没有 `move_to_end`（本机实测 `hasattr({}, 'move_to_end')` 为 `False`），把容器顺手给的顺序当业务顺序，最后都会在某个边界上翻车（见 [[coding-06]]）。

### 5. 复杂度与两个峰值

- **时间 $O(n)$**：每个节点出队、入队各一次，列判断是 $O(1)$。
- **视图 $O(k)$，$k \le \min(n, 2h+1)$**：上界 $2h+1$ 来自「深度 $d$ 的列落在 $[-d, d]$ 内」；$n$ 那一头来自左链，每层一列。
- **BFS 峰值 = 最宽一层**：完美二叉树 h=16 实测队列长度峰值 65,536（$=2^{16}$），tracemalloc 峰值 4.09 MB（含 seen dict 与队列元组）；h=20 就是 1,048,576 槽位。一般二叉树的层宽不超过 $\min(n, 2^d)$，退化树反而更小（左链峰值为 1）。
- **DFS 峰值 = 树高**：递归版在左链上撞 `sys.getrecursionlimit()`（默认 1000）。二分实测：998 个节点还能跑通，999 个即 `RecursionError`。把 limit 调到 $10^6$ 去跑 30 万深的链不是修复——进程直接段错误（exit 139），C 栈先崩。
- **现场直接写迭代版**：显式队列（BFS）或显式栈（DFS）。同一条 $10^6$ 深的左链，迭代 BFS 实测 239.4 ms（239 ns/node）跑通，队列峰值 1。

### 6. 视图大小的两端，以及 Cursor 语境下的收尾口径

| 形状 | n | 列数 k | k/n |
| --- | --- | --- | --- |
| 完美二叉树 h=18 | 524,287 | 37（$=2h+1$） | 0.0071% |
| 完美二叉树 h=20 | 2,097,151 | 41 | 0.0020% |
| 左链 | 1,000,000 | 1,000,000 | 100% |

平衡树上视图小到可以忽略，退化树上视图等于整棵树：左链序列化「列:值」是 14,777,778 字符，按代码口径 3.05 字符/token（`tiktoken` 的 `o200k_base`，与 [[coding-10]] 一致）约 4,845,173 token。按「视图总是很小」做假设，等于把整棵树搬进上下文。

因此返回结构里要有 `truncated` / `total_columns` / `next_offset`（截断标记的口径与 [[coding-09]]、[[coding-12]] 一致），并按 [[coding-10]] 的流式口径逐页产出：同一份语料实测峰值内存从 4.1 MB（全部块驻留）降到 2.8 MB（只消费当前块）。注意列升序输出不能真流式——BFS 首次访问序不是升序，只能先收齐列集再分页，或者按「列区间 + 深度游标」两段式产出。

### 7. 增量维护：编辑器里真正的成本是重复遍历

单次遍历在 $10^5$ 量级只有十毫秒级，成本来自**每按键都重算**：n=65,535（h=15）全量重算 11.3 ms/次（同进程另一次取样 10.7 ms），按 8 键/秒输入就是每个用户 9.0% 单核；到 $10^6$ 节点（177 ns/node）就是每次按键 177 ms。

顶视图不能只靠「列 → 节点」的 map 做增量：用户折叠或删掉一个浅节点后，同列更深、之前被遮住的节点必须重新露出来，所以要么每列保留按深度有序的结构，要么全量重扫。可直接复用的口径来自 [[coding-06]] 的 TTL 实现：每列一个最小堆（本机实测 `(expire_at, seq, key)` 元组 heappush 0.24 µs @ 百级堆、0.47 µs @ 20 万级，heappop 0.44 µs → 1.27 µs），堆不支持删任意元素就用自增版本号惰性作废，作废条目必须当指标暴露（该文实测 145,334 条），且有界清理是硬要求（20 万条同时失效一次清完 575.3 ms，单次最多 64 条只要 0.27 ms）——按键路径上不能出现这种长尾。单列增量 0.24–0.28 µs 对全量 11.3 ms 差约 $4\times10^4$ 倍，这就是必须做 dirty path 的理由。但先证明瓶颈确实是重复遍历，而不是先上复杂结构（[[coding-11]] 的口径：上不了线的原因常在账上）：$10^5$ 量级的朴素 BFS + map 已经够用。

接到 Cursor 的产品面还有三件事要进契约：

1. **n-ary 的列函数**：AST、文件树、符号表都没有 $\pm 1$，列函数必须显式给出（按字符偏移/缩进，或按子节点序号）。同一棵树因枚举顺序不同给出不同视图，会污染缓存键与评测基线。
2. **确定性**：不要用内置 `hash()` 做列键的持久化或分片——`str` 的 hash 每进程加盐，本机实测两次运行的 `hash('x')` 不同（与 [[coding-06]] 的结论一致）；列键用整数或显式排序键，输出永远显式排序。
3. **并发与回灌**：只读遍历可以并行，但合并必须保持列序与「同列取先访问」的确定性；视图进 agent 上下文后成本按 $O(T^2)$ 放大（[[coding-12]] 实测 T=40 步累计输入 189,800 token），合并前先按列预算裁剪。

## 数值与代码验证

口径：Python 3.10.12，`time.process_time()` 取样取最小值（避开共享机器的调度噪声）、`gc` 关闭，树用数组/`__slots__` 节点表示；共享机器上的绝对数只能看比例与量级。复算脚本在 `.work/` 下，命令为 `python3 .work/lane-d/bench4.py`。

**表 1：h=16 的口径细节**

| 指标 | 实测 | 说明 |
| --- | --- | --- |
| 队列长度峰值 | 65,536 | $=2^{16}$，最宽一层的节点数 |
| tracemalloc 峰值 | 4.09 MB | seen dict + 队列元组 |
| 33 列 `sorted()` | 0.66 µs | 占整趟 BFS（23.25 ms）的 0.0028% |
| 递归前序可跑通的最长左链 | 998 节点 | 999 节点即 `RecursionError`（默认 limit 1000） |
| limit 调到 $10^6$ 跑 30 万深链 | exit 139 | 段错误，C 栈先崩 |
| $10^6$ 左链的迭代 BFS | 239.4 ms（239 ns/node） | 队列峰值 1、列数 $10^6$ |

**表 2：每按键的成本账（h=15，n=65,535）**

| 路径 | 单次成本 | 8 键/秒的单个用户 | 相对 |
| --- | --- | --- | --- |
| 全量重算（BFS + 排序） | 11.3 ms | 9.0% 单核 | $1\times$ |
| 单列增量（最小堆 push，百级堆） | 0.24 µs | 可忽略 | 约 $4\times10^4$ 倍更快 |
| 全量重算（$10^6$ 节点，按 177 ns/node 外推） | 177 ms | — | 每次按键 177 ms |

**参考实现与复现**（列契约 + BFS 首次访问 + 显式排序 + 预算与分页；后半段是反例与断言，仅标准库）：

```python
from collections import deque


class Node:
    __slots__ = ('val', 'left', 'right')

    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right


def top_view(root, child_col=None, max_nodes=10 ** 6, max_columns=10 ** 5):
    """返回 (视图, meta)。契约：列 = 水平距离（根 0 / 左 -1 / 右 +1）；顶 = 该列深度最小的
    节点；并列取 BFS 从左到右先访问到的那个；输出按列升序；截断时 meta 给出原因。"""
    if child_col is None:
        child_col = lambda pc, i, ch: pc + (-1 if i == 0 else +1)   # n-ary 由调用方覆盖
    top, q, visited = {}, deque([(root, 0, 0)]), 0
    while q:
        if visited >= max_nodes:
            return _pack(top, 'node_budget')
        node, col, depth = q.popleft()
        visited += 1
        if col not in top:                       # BFS 出队深度非降 → 首次访问即最浅
            if len(top) >= max_columns:
                return _pack(top, 'column_budget')
            top[col] = (node.val, depth)
        for i, ch in enumerate((node.left, node.right)):
            if ch is not None:
                q.append((ch, child_col(col, i, ch), depth + 1))
    return _pack(top, None)


def _pack(top, reason):
    cols = sorted(top)                           # BFS 首次访问序是 0,-1,+1,-2,…，必须显式排序
    view = [(c, top[c][0], top[c][1]) for c in cols]
    return view, {'total_columns': len(cols), 'truncated': reason is not None,
                  'reason': reason, 'next_offset': 0}


def page(view, offset=0, limit=2):
    """消费方凭 next_offset 续读；截断永远显式。"""
    chunk = view[offset:offset + limit]
    end = offset + len(chunk)
    return {'columns': chunk, 'next_offset': end, 'total_columns': len(view),
            'truncated': end < len(view)}


def perfect(h):                                  # 完美二叉树，层序编号：n = 2^(h+1) - 1
    nodes = [Node(i) for i in range(1, 2 ** (h + 1))]
    for i in range(len(nodes) // 2):
        nodes[i].left, nodes[i].right = nodes[2 * i + 1], nodes[2 * i + 2]
    return nodes[0]


def first_order(root):                           # BFS 首次访问的列序（排序前）
    seen, order, q = {}, [], deque([(root, 0)])
    while q:
        u, c = q.popleft()
        if c not in seen:
            seen[c] = u.val
            order.append(c)
        for ch, dc in ((u.left, -1), (u.right, +1)):
            if ch is not None:
                q.append((ch, c + dc))
    return order


def dfs_first_view(root):                        # 前序首次访问就写下来——经典错解
    """栈是 LIFO，先压右孩子才能先弹出左孩子。"""
    seen, st = {}, [(root, 0, 0)]
    while st:
        u, c, d = st.pop()
        if c not in seen:
            seen[c] = (u.val, d)
        for ch, dc in ((u.right, 1), (u.left, -1)):
            if ch is not None:
                st.append((ch, c + dc, d + 1))
    return [(c, *seen[c]) for c in sorted(seen)]


print(top_view(perfect(3))[0])
# [(-3, 8, 3), (-2, 4, 2), (-1, 2, 1), (0, 1, 0), (1, 3, 1), (2, 7, 2), (3, 15, 3)]
print(first_order(perfect(3)))
# [0, -1, 1, -2, 2, -3, 3]      ← BFS 首次访问序，不是升序

n1, n2, n6, n7, n8, n9 = Node(1), Node(2), Node(6), Node(7), Node(8), Node(9)
n3, n4, n5 = Node(3), Node(4), Node(5)
n1.left, n1.right = n6, n2
n6.right, n7.right, n8.right = n7, n8, n9
n2.left, n3.right, n4.right = n3, n4, n5
print(top_view(n1)[0])
# [(-1, 6, 1), (0, 1, 0), (1, 2, 1), (2, 9, 4)]   ← +1 列 = 深度 1 的 2；+2 列 = 先访问到的 9
print(dfs_first_view(n1))
# [(-1, 6, 1), (0, 1, 0), (1, 8, 3), (2, 9, 4)]   ← +1 列被深度 3 的 8 占住（7 节点反例同理）

chain = cur = Node(1)
for v in range(2, 60):
    cur.left = Node(v)
    cur = cur.left
vc, mc = top_view(chain, max_columns=10)
print(vc[0], mc['total_columns'], mc['reason'])
# (-9, 10, 9) 10 column_budget      ← 截断显式，不返回残缺的「完整视图」
print(page(vc, offset=0, limit=2))
# {'columns': [(-9, 10, 9), (-8, 9, 8)], 'next_offset': 2, 'total_columns': 10, 'truncated': True}
```

## 常见追问

- **追问**：为什么 BFS 首次访问就是答案，DFS 差在哪？
  - 要点：BFS 出队深度非降，所以每列第一个被写入的节点深度最小（反证见第 2 小节）；DFS 的访问顺序由子树结构决定，与深度无关。7 节点反例里 DFS 把 $+2$ 列写成深度 4 的 6。DFS 想用就必须每列比较当前最小深度，实测慢 15%–28%。
- **追问**：输出为什么必须再排一次序？
  - 要点：BFS 首次访问列序是 $0,-1,+1,-2,+2,\dots$，值来自 dict 的插入序。33 列排序 0.66 µs，占整趟 BFS 的 0.0028%。这与 dict 没有 `move_to_end` 是同一类问题：容器的顺序是容器的，不是业务的（[[coding-06]]）。
- **追问**：递归写法行不行？平衡树和退化树的资源曲线差多少？
  - 要点：默认 `sys.getrecursionlimit()` = 1000，实测 998 个节点的左链能跑通、999 个就 `RecursionError`；把 limit 调到 $10^6$ 跑 30 万深的链直接段错误（exit 139）。峰值也分两种口径：BFS 峰值 = 最宽一层（h=16 是 65,536 槽位、tracemalloc 4.09 MB，h=20 是 $2^{20}$），DFS 峰值 = 树高；左链的队列峰值恒为 1，但视图有 $10^6$ 列、约 485 万 token。
- **追问**：如果节点是 n-ary 的（AST、文件树）怎么定义列？
  - 要点：$\pm 1$ 不存在了，必须显式给出列函数：按字符偏移/缩进（最贴近编辑器语义），或按子节点序号（第 $i$ 个子节点偏移 $i$）。列函数进契约，否则同一棵树因子节点顺序不同给出不同视图，缓存键与评测基线一起被污染。此时 BFS 的「深度非降」结论不变。
- **追问**：编辑器里每次按键都重算吗？
  - 要点：不。n=65,535 全量重算 11.3 ms，8 键/秒即 9.0% 单核/用户，$10^6$ 节点是每次按键 177 ms。折叠或删除浅节点后同列更深的节点要重新露出，所以要每列一个按深度有序的最小堆 + 版本号惰性作废（[[coding-06]] 口径：push 0.24–0.47 µs，作废条目必须当指标，清理必须有界）。增量对全量约 $4\times10^4$ 倍。
- **追问**：视图很大时怎么返回给上层？
  - 要点：显式截断而非悄悄裁剪：`truncated` / `total_columns` / `next_offset`（[[coding-09]]、[[coding-12]]），并按 [[coding-10]] 的流式口径逐页产出（4.1 MB → 2.8 MB）。列升序输出不能真流式，先收齐列集再分页，或按「列区间 + 深度游标」两段式产出。

## 相关题目

- [[coding-06]]：LRU cache + TTL。dict 保插入序但没有 `move_to_end` 的实测、堆 + 版本号惰性作废、作废条目当指标、有界清理的长尾口径（575.3 ms 对 0.27 ms）都来自这篇，增量维护一节的数字直接复用。
- [[coding-12]]：最小 agent loop。步数上限从 token 预算反推、超限返回「部分结果 + 原因」而不是抛异常，是这道题截断口径的上位写法；$O(T^2)$ 的上下文成本（T=40 累计 189,800 token）是第 7 小节第 3 条的依据。
- [[coding-10]]：带重叠的分块器。流式产出与峰值内存（4.1 MB → 2.8 MB）、代码口径 3.05 字符/token 的换算来自这篇，这里用它把列数换算成 token。
- [[coding-11]]：余弦检索为什么不上线。瓶颈在账上而不在代码里的口径，用来判断「该不该上增量结构」。
- [[coding-09]]：流式 SSE/JSON 解析器。`truncated` / `total_columns` / `next_offset` 这类显式标记的用法与它同源。
- [编程与数据结构](../../编程与数据结构/README.md)：这道题所属专题的 12 道手写实现题地图。

## 参考资料与归属

- **Tree traversal（维基百科）（延伸）** —— Wikipedia：<https://en.wikipedia.org/wiki/Tree_traversal>。层序遍历与深度优先遍历的定义与复杂度。
- **collections — deque 与 OrderedDict（Python 标准库文档）（延伸）** —— Python Software Foundation：<https://docs.python.org/3/library/collections.html>。层序遍历所需的双端队列原语。

- **延伸来源说明**：水平距离的推导、边界用例与实测算例，是按本仓库统一口径自行推导与实测的工程算例，不是上述来源的原文数字；来源仅用于机制、算法与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
