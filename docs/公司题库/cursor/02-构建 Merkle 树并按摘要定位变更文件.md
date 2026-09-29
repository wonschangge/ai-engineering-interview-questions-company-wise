---
type: question
id: cursor-02
company: Cursor（Anysphere）
topic: coding
order: 2
question: 给定一个仓库快照（路径 → 内容），构建一棵 Merkle 树，并编写函数返回两个快照之间发生变更的文件，且不比较每个文件的内容。
question_en: Given a repository snapshot (path → content), build a Merkle tree and write the function returning which files changed between two snapshots without comparing every file's content.
asked_at: []
level: 进阶
tags: [merkle-tree, 增量索引, 内容寻址, 哈希, 仓库快照]
sources:
  - title: Merkle tree（维基百科）（延伸）
    url: https://en.wikipedia.org/wiki/Merkle_tree
    author: Wikipedia
    published: 
  - title: hashlib — 安全哈希与消息摘要（Python 标准库文档）（延伸）
    url: https://docs.python.org/3/library/hashlib.html
    author: Python Software Foundation
    published: 
  - title: Git Internals — Packfiles（Pro Git）（延伸）
    url: https://git-scm.com/book/en/v2/Git-Internals-Packfiles
    author: Scott Chacon, Ben Straub
    published: 
  - title: Merkle tree（维基百科）（延伸）
    url: https://en.wikipedia.org/wiki/Merkle_tree
    author: Wikipedia
    published: 
  - title: hashlib — 安全哈希与消息摘要（Python 标准库文档）（延伸）
    url: https://docs.python.org/3/library/hashlib.html
    author: Python Software Foundation
    published: 
  - title: Git Internals — Packfiles（Pro Git）（延伸）
    url: https://git-scm.com/book/en/v2/Git-Internals-Packfiles
    author: Scott Chacon, Ben Straub
    published: 
related: [coding-03, coding-06, coding-09, coding-10, rag-01, coding]
updated: 2026-09-28
---

## 一句话答案

> 叶子摘要只绑内容与长度（`blob\0` + 长度 + 字节），内部摘要绑子节点个数与顺序（`tree\0` + 个数 + 子摘要），这两处域分隔是整棵树成立的前提：没有它，空文件、空目录列表与某个短内容会互相冒充，树可以被人为构造出相同的根。
> 整棵树必须按路径的**字节序**确定性构建：靠 dict 迭代序、靠内置 `hash()`（每进程随机加盐）、或不做 CRLF 与 Unicode 归一化，都会让同一快照在两台机器上算出不同的根，索引服务端多进程共享时等于每轮全量重建。
> 比较先比根：相同直接返回空集（1 次 32 B 比较，这就是「不比较每个文件的内容」的落点）；不同才从根同时下潜，只进入摘要不同的孩子，代价是 $O(k \log n)$ 而不是 $O(n)$——n = 10 万、k = 3 时实测访问 51 个节点、读 102 个摘要 = 3264 B = 3.19 KiB，而朴素 diff 要读两侧 1.526 GiB。
> 第二问的落点是形状对插入/删除的稳定性：按排序位置切分的树在重命名或插入一个文件后几乎全部节点作废（实测 2.0 × 10⁵ 个节点的摘要变化），按路径前缀分桶后只重建一个桶（782 个摘要）；接口返回 added / modified / deleted 三类，叶子不绑路径，纯移动才不会触发重新分块与重新 embedding，rename 由 digest → paths 反向表配对识别。

## 面试官在考什么

- **先立契约再写代码**：叶子与内部节点是否域分隔、是否长度绑定、排序键是否确定。这三条任缺一条，树在功能上还能跑，但正确性已经没了——而且是静默的。
- **复杂度是否真是 $O(k \log n)$**：根摘要短路、只下潜摘要不同的孩子、下潜用显式栈（内存 $O(\text{depth})$）而不是把两侧叶子全部展开比较。
- **是否理解 $O(k \log n)$ 的前提**：它只对「路径集不变、仅内容变化」成立。路径集一变，位置分片的树就退化成 $O(n)$，这是第二问真正区分人的地方。
- **增量重建的失效边界**：stat 快路径会漏检、文件正在写时哈希会读到半截内容、旧向量留在索引里会让模型给出错误 grounding。要能给出「便宜捷径 + 兜底复核 + 指标」的成套方案，而不是只讲省了多少次哈希。
- **边界是否写全**：三类返回、大小写不敏感卷、Unicode 归一化、空目录不产生叶子、大文件与二进制走内容定义分块、叶子集要显式排除 `node_modules` 与构建产物。

常见错误答案：

- 把「不比较每个文件的内容」理解成「不读内容就能建树」。建树必须读一遍内容算叶子摘要；省掉的是**比较阶段的第二次全量读**。讲清这一点才能对上账：建成后每次比较是 3.19 KiB 的摘要读取，而不是 1.526 GiB。
- 直接说「Merkle 树比较是 $O(\log n)$」而不区分情形。另外照抄 Git 也不等于拿到 $O(\log n)$：Git 的 tree 对象是**目录内平铺**的条目列表，比较同一个目录的两个 tree 是 $O(\text{该目录条目数})$；要拿到 $\log n$ 必须让结构层本身平衡或分桶。

## 原理与推导

### 1. 结构契约：两种节点，域分隔加长度绑定

```
leaf  = H("blob" || 0x00 || len(content) || content)
inner = H("tree" || 0x00 || count(children) || children[0] || children[1] || ...)
```

| 朴素写法 | 实测后果 | 契约要求 |
| --- | --- | --- |
| `leaf = H(content)`、`inner = H(left + right)` | 单叶子树 `H(H('a') + H('b'))` 与双叶子树 `H(H('a'), H('b'))` 的根**相同**（实测碰撞成立）；`H('')` 同时是空文件与空目录列表的摘要 | 叶子与内部节点必须用不同域前缀；空子树要有自己的常量摘要 |
| 子摘要直接首尾相接、不绑个数与名字 | 长度不定的拼接允许重排与截断构造出相同输入 | 内部节点绑子节点个数，Git 的做法是绑 `<mode> <name>\0<20B hash>` |
| 只哈希文件内容，不绑目录结构 | 两个目录互换同名文件、或把 `a/b` 与 `a` 的归属改动，树看不出差别 | 目录条目必须把模式与名字一起哈希，这正是 Git tree 对象的序列化格式 |

Git 的 tree 对象就是内部节点契约的参考实现：把所有条目按名称排序，序列化成 `<mode> <name>\0<20B hash>` 再整体哈希。名字进哈希意味着「同一个内容放在不同路径」在结构层是不同的节点；模式进哈希意味着可执行位的变化也会被 diff 捕获。

### 2. 确定性：同一快照只能有一个根

| 写法 | 为什么会破坏 | 正确做法 |
| --- | --- | --- |
| 按 dict 的插入序建树 | dict 保持插入序（Python 3.7 起是语言保证），两个进程遍历同一个目录得到不同顺序，根就不同（实测成立） | 显式按路径排序，不依赖容器行为 |
| 用内置 `hash(path)` 当排序键或摘要 | CPython 对 str 的哈希按进程随机加盐，实测同一表达式三次运行得到 `-4064319862613324564`、`3280391631705228031`、`6982149393400456965` | 排序键与摘要都用 SHA-256 等稳定算法（另见 [[coding-06]] 对缓存键的同一提醒） |
| 不归一化行尾 | 同一文件 CRLF 与 LF checkout 的字节不同，摘要不同（实测成立），两个 checkout 会建出两棵不同的树 | 归一化要么全链路统一做（把归一化后的字节当唯一真源），要么统一不做；两端混用会得到两个都「正确」的根 |
| 不归一化大小写与 Unicode | macOS/Windows 上 `A.py` 与 `a.py` 是同一个文件；HFS+ 会把文件名写成 NFD，APFS 保留原样，同一路径的字节序不同 | 路径先做 NFC 归一化、分隔符统一成 `/`、按需小写化，再编码成 UTF-8 排序 |

### 3. 比较：先比根，再同向下潜

1. 设叶子数 $n$、树高 $d = \lceil \log_2 n \rceil$、变更叶子数 $k$。比较两侧根摘要：相同即返回空集，这一步读 32 B，与 $n$ 无关，是「不比较每个文件的内容」的字面落点。
2. 不同则从根同时下潜，对每个节点只比较它的两个孩子的摘要，只把摘要不同的孩子压栈。
3. 只有变更叶子的祖先路径上才会出现摘要不同的节点，访问量约为路径并集 $k \cdot \log_2(n/k) + 2k$；$n = 10^5$、$k = 3$ 时为 51.1，实测 51 个节点（48 个内部节点 + 3 个叶子），读 102 个摘要 = 3264 B。
4. 显式栈里最多同时有 $O(d)$ 个节点，内存是 $O(\log n)$ 而不是 $O(n)$，也不会因稀疏 Merkle 的 64 层深度而爆栈。

一次完整刷新与一次比较是两笔不同的账：建树要把两侧内容各读一遍并哈希（$2 \times 781.25\ \text{MiB} = 1.526\ \text{GiB}$），而此后每次比较只需 3.19 KiB 的摘要读取，二者相差 $5 \times 10^5$ 倍。索引常驻服务端的场景里，重建是偶发、比较是高频，收益来自这个频次结构。

### 4. 形状必须对插入与删除稳定

| 结构 | 3 个文件内容变更 | 重命名或插入 1 个路径 | 说明 |
| --- | --- | --- | --- |
| 按排序下标对半切分 | 51 个摘要（$3 \times (17+1) = 54$ 为上界，去重后 51） | 200,002 个节点摘要变化，展开槽位共 262,143 个，约等于全量 | 下标平移把后续所有叶子挪到新兄弟位置，最坏 $O(n)$ 作废 |
| 补齐到 $2^{17}$ 的稀疏树 + 空子树常量摘要 | 51 个摘要 | 18 个摘要（只重算该槽位到根的路径） | 槽位由路径本身决定，插入不改变别人的位置；空子树折叠后实际存储 $2n-1$ 个摘要 |
| 按路径首字节 256 桶 + 桶内平衡树 | 51 个摘要 | 782 个摘要（桶内 391 个文件：$2 \times 391-1 = 781$，加根 1 个） | 只重建受影响的那个桶，其余 255 个桶的摘要全部复用 |

这张表就是「用户按一次保存后索引多久追上」的量级来源：全量重建 $2n-1 = 199{,}999$ 个摘要，内容变更只重算 51 个（$3.9 \times 10^3$ 倍），而一个位置分片的实现在重命名时又掉回全量。生产实现的顺序是：先把形状做成路径的函数（前缀分桶或定长槽位），再谈摘要算法的吞吐。

### 5. 坑一：stat 元数据不是「没变」的证据

用 `(size, mtime, inode)` 跳过哈希，能把扫描从「读 781 MiB」降到 10 万次 stat，但两个方向都会错：

- 元数据变了内容没变（`git checkout`、复制、构建工具 `touch`）只会让一批文件白算一遍；元数据没变内容变了才是事故。常见的简化说法是「mtime 只有秒级精度」；在 ext4 上 `st_mtime` 带纳秒字段，真正的坑更宽——HFS+、FAT 与部分网络挂载只有秒级甚至 2 秒粒度；Git 的 racy-git 处理针对的是另一个问题：索引里记录的 mtime 等于索引自身的写入时间时，必须重新哈希内容，否则同一秒内的写入会被当成没变。
- 文件正在被写入时哈希，会读到半截内容，把错误版本的摘要写进树。

要害不是多算一次，而是下游的静默错误 grounding：旧代码的向量留在索引里，模型据它回答，用户看不出任何异常。所以快路径必须配三件套：定期全量复核、按比例抽样重哈希、把「未复核路径数」做成 gauge，再设阈值触发全量重建。这与 [[coding-06]] 里 TTL 的惰性删除加有界清扫是同一类写法——用便宜的近似换延迟，用后台的有界复核换正确性。

### 6. 坑二：重命名与移动

重命名符号、搬目录、跑一次格式化是编辑器场景最高频的变更，两种朴素做法都不对：只哈希内容、不绑路径时 `git mv a.py b.py` 表现为「1 删除 + 1 新增」，按路径做 key 的索引会重新分块并重新 embedding；在叶子上绑路径则让纯移动变成内容变更。

解法是分两层：结构层绑 `(mode, name, digest)` 保证路径语义，内容层的叶子摘要只绑内容，另外维护 `digest → paths` 反向表先按摘要配对出 rename。账单（n = $4 \times 10^5$、平均 12 KiB）：

| 场景 | 字节 | token（按 4.93 字符/token） |
| --- | --- | --- |
| 全量重新分块与 embedding | $4 \times 10^5 \times 12{,}288 = 4.58\ \text{GiB}$ | $9.97 \times 10^8$ |
| 移动 3 个文件（摘要未变，只有结构层改动） | $3 \times 12{,}288 = 36{,}864\ \text{B}$ | $7.5 \times 10^3$ |
| 差值 | — | $1.3 \times 10^5$ 倍 |

一个容易踩的细节：纯移动可能让**根摘要保持不变**。实测 5 个文件的快照把 `f.py` 移到 `moved/f.py`，排序后的位置与内容摘要都没变，两侧根相等；只比根就会漏报这次路径变更。所以「根相同返回空集」必须同时确认路径集相同，否则它会掩盖 rename——而 rename 正是要靠 `digest → paths` 补回来的那类变更。

### 7. 摘要算法与内存账

$n = 4 \times 10^5$ 时整棵树 $2n-1 = 799{,}999$ 个摘要节点：

| 算法 | 每节点 | 整棵树 | 取舍 |
| --- | --- | --- | --- |
| SHA-256 | 32 B | 25,599,968 B = 25.6 MB = 24.41 MiB | 默认选择；$\ge 2^{128}$ 量级的碰撞成本对新系统足够 |
| SHA-1 | 20 B | 15,999,980 B = 16.0 MB = 15.26 MiB | 省 37.5% 内存（每节点 32 B → 20 B），但 20 字节哈希的碰撞成本已不适合新系统；只为对齐 Git 口径时用 |
| BLAKE3 | 32 B | 25.6 MB | 可并行、多核 GB/s 级，适合把建树做成 CPU 密集的批量作业 |
| xxhash 等非抗碰撞哈希 | 8 B | 6.4 MB | 只能作一级筛子 |

弱哈希的使用方向是单向的：「弱哈希不同 $\Rightarrow$ 一定进候选」成立；「弱哈希相同 $\Rightarrow$ 文件没变」不成立。一旦反向使用，误判就等于索引静默漏更新。分层口径与 [[coding-06]]、[[coding-09]] 的增量状态机一致：近似层负责筛，权威层负责判定。

吞吐口径：本机 `hashlib` 单核实测 1.30–1.38 GB/s（1 MiB、64 KiB、8 KiB 三档一致），带 SHA-NI 的现代 x86 按 2 GB/s 计。1.526 GiB 的纯哈希耗时在 0.82 s（2 GB/s）到 1.26 s（1.3 GB/s）之间，而磁盘按 500 MB/s 要 3.28 s——真实场景里瓶颈是 I/O 与序列化，不是哈希。

### 8. 接口与边界

- **三类返回**：只有单侧存在的叶子算新增或删除，两侧都有但摘要不同算修改。
- **大小写不敏感卷**：macOS/Windows 上 `A.py` 与 `a.py` 是同一个文件，排序必须用归一化后的字节序，否则两台机器建出不同的树；Unicode 同理（NFC/NFD）。
- **空目录不产生叶子**：Git 不跟踪空目录；需要目录视图就单独存目录条目，别让空目录持有与空文件相同的摘要。
- **大文件与二进制**：整文件哈希会让「改一行格式化」失效整个文件，改走内容定义分块（CDC，平均块 32–64 KiB）作为二级 Merkle，失效范围收敛到一个块。
- **叶子集要显式过滤**：只有源码 4 万个文件时，摘要数 $2n-1 = 79{,}999$、深度 16；把 20 万个 `node_modules` 文件算进来，$n$ 变成 24 万，摘要数 479,999（6 倍），深度从 16 涨到 18。树规模与重建时间由「哪些路径进叶子集」决定，不由 diff 算法决定——这一条同时决定了引用与高亮所需的字符区间元数据要不要随块一起存（[[coding-10]]、[[rag-01]]）。

## 数值与代码验证

口径：$n = 100{,}000$ 个文件、平均 8 KiB、SHA-256、1024 进制（1 MiB = 1,048,576 B，1 GiB = 1024 MiB）；摘要固定 32 B。所有数字由下面的代码在本机复算，代码输出以注释形式附在行末。

**表 1：一次朴素比较与 Merkle 比较**

| 项 | 计算 | 结果 |
| --- | --- | --- |
| 单快照内容量（两侧为朴素 diff 的读取量） | $100{,}000 \times 8192$ | 819,200,000 B = 781.25 MiB；两侧 1.526 GiB |
| 仅哈希耗时（2 GB/s 口径） | $1.6384 \times 10^9\ \text{B} \div (2 \times 10^9\ \text{B/s})$ | 0.82 s |
| 仅哈希耗时（本机实测 1.3 GB/s） | $1.6384 \times 10^9\ \text{B} \div (1.3 \times 10^9\ \text{B/s})$ | 1.26 s |
| 磁盘下界（500 MB/s） | $1.6384 \times 10^9\ \text{B} \div (500 \times 10^6\ \text{B/s})$ | 3.28 s |
| Merkle 比较读摘要 | $51 \times 2 \times 32$ | 3264 B = 3.19 KiB |
| 比值 | $1.526\ \text{GiB} \div 3264\ \text{B}$ | $5.0 \times 10^5$ |

**表 2：重建与比较的摘要数（n = 10 万，2n − 1 = 199,999）**

| 动作 | 摘要数 | 相对全量 |
| --- | --- | --- |
| 3 个文件内容变更（独立路径上界 54，去重实测 51） | 54 / 51 | 1 / 3704（去重 1 / 3922） |
| 全量重建（含补齐槽位展开 262,143） | 199,999 | 1 |
| 重命名 1 个文件（位置分片实现，展开槽位 262,143 个） | 200,002 个节点摘要变化 | 约等于全量 |
| 256 桶方案：重建受影响的一个桶 | $781 + 1 = 782$ | 1 / 256 |

分叉大小是同一个取舍的另一面（n = 10 万）：二叉时深度 17、单个文件重算 18 个摘要、每次下降比较 64 B；换到 16 叉是 5 / 6 / 512 B，256 叉是 3 / 4 / 8 KiB。$k = 3$ 时二叉树的 102 个摘要（3.19 KiB）比 256 叉的 7 × 8 KiB 更省，只有 $k$ 很大时才值得换成大分叉——别只报深度、不报每层宽度。

```python
import hashlib, unicodedata

LEAF_TAG, NODE_TAG = b'blob\x00', b'tree\x00'
EMPTY = hashlib.sha256(NODE_TAG + (0).to_bytes(4, 'big')).digest()   # 空子树常量摘要

def blob_digest(data):
    """叶子：绑内容与长度，不绑路径（移动文件后摘要不变）。"""
    return hashlib.sha256(LEAF_TAG + len(data).to_bytes(8, 'big') + data).digest()

def node_digest(children):
    """内部：域分隔 + 子节点个数；子摘要定长，顺序即位置绑定。"""
    m = hashlib.sha256(); m.update(NODE_TAG); m.update(len(children).to_bytes(4, 'big'))
    for d in children:
        m.update(d)
    return m.digest()

def normalize(path):
    """卷归一化后再编码：大小写不敏感卷上 A.py 与 a.py 是同一文件。"""
    return unicodedata.normalize('NFC', path).replace('\\', '/').lower().encode('utf-8')

class MerkleTree:
    def __init__(self, files):                       # files: 路径 -> 内容(bytes)
        self.paths = sorted(files, key=normalize)    # 字节序，跨机一致
        self.blobs = {p: blob_digest(files[p]) for p in self.paths}
        n = len(self.paths)
        depth = max(0, (n - 1).bit_length())         # ceil(log2 n)；n=1e5 -> 17
        level = [self.blobs[p] for p in self.paths] + [EMPTY] * ((1 << depth) - n)
        self.levels = [level]
        while len(level) > 1:
            level = [node_digest(level[i:i + 2]) for i in range(0, len(level), 2)]
            self.levels.append(level)

    @property
    def root(self):
        return self.levels[-1][0]

    def descent(self, other):
        """路径集相同时：从根同时下潜，只进入摘要不同的孩子。"""
        changed, stack, visited = [], [(len(self.levels) - 1, 0)], 0
        while stack:
            lv, i = stack.pop()
            visited += 1
            if lv == 0:
                if self.levels[0][i] != other.levels[0][i]:
                    changed.append(self.paths[i])
                continue
            for t in (2 * i, 2 * i + 1):             # 只压入两侧摘要不同的孩子
                if self.levels[lv - 1][t] != other.levels[lv - 1][t]:
                    stack.append((lv - 1, t))
        return sorted(changed), visited

def changed_files(old, new):
    """不读内容，返回 (added, modified, deleted)。"""
    if old.root == new.root and old.paths == new.paths:
        return [], [], []                            # 1 次 32 B 比较
    added   = [p for p in new.paths if p not in old.blobs]
    deleted = [p for p in old.paths if p not in new.blobs]
    modified = [p for p in new.paths if p in old.blobs and old.blobs[p] != new.blobs[p]]
    return added, modified, deleted

def renames(old, new):
    """digest -> paths 反向表：纯移动配成 rename，不重新分块。"""
    rev = {}
    for p, d in old.blobs.items():
        rev.setdefault(d, set()).add(p)
    return [(q, p) for p in new.paths for q in rev.get(new.blobs[p], ())
            if q != p and q not in new.blobs]

a = {'a.py': b'1', 'b.py': b'2', 'c/d.py': b'3', 'c/e.py': b'4', 'f.py': b'5'}
b = {'a.py': b'1X', 'b.py': b'2', 'c/d.py': b'3', 'c/e.py': b'4', 'g.py': b'5'}
moved = {('moved/f.py' if p == 'f.py' else p): c for p, c in a.items()}
ta, tb, tm = MerkleTree(a), MerkleTree(b), MerkleTree(moved)
print(changed_files(ta, tb))                  # (['g.py'], ['a.py'], ['f.py'])
print(changed_files(ta, tm), ta.root == tm.root)   # (['moved/f.py'], [], ['f.py']) True
print(renames(ta, tm))                        # [('f.py', 'moved/f.py')]

naive = lambda *p: hashlib.sha256(b''.join(p)).digest()
print(naive(naive(b'a'), naive(b'b')) == naive(naive(b'a') + naive(b'b')))  # True：无域前缀即碰撞
print(EMPTY == blob_digest(b''))              # False：空子树的摘要不冒充空文件

n, blk = 100_000, 8192
big = {'src/f%06d.py' % i: i.to_bytes(8, 'big') * (blk // 8) for i in range(n)}
t1, big2 = MerkleTree(big), dict(big)
for p in ('src/f000003.py', 'src/f050000.py', 'src/f099999.py'):
    big2[p] += b'!'
got, visited = t1.descent(MerkleTree(big2))
print(len(t1.levels) - 1, 2 * n - 1)          # 17 层, 摘要 199999
print(len(got), visited, 2 * visited * 32)    # 3, 51 个访问节点, 3264 B

big3 = {('src/moved_f000003.py' if p == 'src/f000003.py' else p): v for p, v in big.items()}
t3 = MerkleTree(big3)
print(sum(1 for la, lb in zip(t1.levels, t3.levels) for x, y in zip(la, lb) if x != y))  # 200002
```

域分隔的效果由代码块末尾两行复算：无前缀时 `H(H('a') + H('b'))` 同时是「叶子 `'a'`、`'b'` 的内部节点」与「内容为 64 字节的单个叶子」的摘要，两条完全不同的树撞到同一个根；加上 `blob` / `tree` 前缀后不再碰撞，`EMPTY` 也不再等于 `blob_digest(b'')`。

## 常见追问

- **追问**：根相同就能断定两个快照没有差异吗？
  - 要点：只有在摘要抗碰撞且路径集也相同这两个条件同时成立时才行。实测的 5 文件例子把 `f.py` 移到 `moved/f.py` 后根摘要相同，但路径集变了，`added` / `deleted` 都不为空。所以短路条件是「根相同且路径集相同」，rename 必须靠 `digest → paths` 而不是根比较。
- **追问**：大文件、二进制文件怎么处理？
  - 要点：整文件哈希的失效粒度是文件，改一行格式化就让整文件重算与重新分块。走内容定义分块（CDC，平均块 32–64 KiB）把失效范围收敛到一个块，代价是分块器本身要在两侧跑一遍、块边界要确定性（否则同一内容有两种切法）。
- **追问**：同一文件在短时间内被连续保存十次怎么办？
  - 要点：入口做 debounce 合并窗口，按路径聚合；对同一个桶加 single-flight，避免十个请求同时重建同一段祖先路径；摘要缓存的键是（规范化路径 + 内容摘要），未变的分块直接复用。合并窗口的长度是「索引新鲜度」与「重建放大」的显式取舍，要能从指标上看到被合并的请求数。
- **追问**：换成 BLAKE3 或 xxhash 能不能行？
  - 要点：吞吐可以，语义要分清。BLAKE3 抗碰撞且可并行，直接替换 SHA-256 没有语义风险；xxhash 这类非抗碰撞哈希只能当一级筛子，「不同 ⇒ 进候选」成立，「相同 ⇒ 没变」不成立，反向使用就等着索引静默漏更新。
- **追问**：stat 快路径的漏检怎么量化、怎么兜底？
  - 要点：漏检率 = 静默未更新的路径数 ÷ 实际变更路径数，用「定期全量复核 + 抽样重哈希 + 与 VCS 状态对账」测出来。指标上要暴露未复核路径数、复核周期与触发全量重建的次数；把快路径的收益（10 万次 stat 对 781 MiB 读取）与漏检率的实测值一起报，而不是只报收益。

## 相关题目

- [[coding-03]]：KV cache 与单步 decode。「cache 不是 $O(1)$、改一个字后面全废」的失效边界与本题的增量重算同构。
- [[coding-06]]：LRU cache 加 TTL。稀有路径用惰性删除加有界清扫，与本题 stat 快路径加定期全量复核是同一类「正确性与成本分离」的写法；内置 `hash()` 按进程加盐的实测也在那篇。
- [[coding-09]]：任意 chunk 边界的流式 SSE-JSON 解析器。增量状态机的口径与「部分输入先落地、稍后复核」的处理方式可直接搬过来。
- [[coding-10]]：带重叠且不切开语义单元的文本分块器。4.93 字符/token 的实测口径与「失效粒度由分块粒度决定」的结论来自这篇。
- [[rag-01]]：面向大型技术文档语料库的 chunking 策略。内容定义分块与引用区间元数据要不要随块存，与本题第 8 节的叶子集口径配套。
- [[coding]]：整个专题的十二道实现题，导读见 [编程与数据结构](../../编程与数据结构/README.md)。

## 参考资料与归属

- **Merkle tree（维基百科）（延伸）** —— Wikipedia：<https://en.wikipedia.org/wiki/Merkle_tree>。
- **hashlib — 安全哈希与消息摘要（Python 标准库文档）（延伸）** —— Python Software Foundation：<https://docs.python.org/3/library/hashlib.html>。
- **Git Internals — Packfiles（Pro Git）（延伸）** —— Scott Chacon, Ben Straub：<https://git-scm.com/book/en/v2/Git-Internals-Packfiles>。
- **Merkle tree（维基百科）（延伸）** —— Wikipedia：<https://en.wikipedia.org/wiki/Merkle_tree>。Merkle 树的结构、根摘要与包含证明。
- **hashlib — 安全哈希与消息摘要（Python 标准库文档）（延伸）** —— Python Software Foundation：<https://docs.python.org/3/library/hashlib.html>。内容摘要的计算与编码。
- **Git Internals — Packfiles（Pro Git）（延伸）** —— Scott Chacon, Ben Straub：<https://git-scm.com/book/en/v2/Git-Internals-Packfiles>。内容寻址 + 打包的工业实例。

- **延伸来源说明**：文中的复杂度对照、内存账与实测算例，是按本仓库统一口径自行推导与实测的工程算例，不是上述来源的原文数字；来源仅用于机制、算法与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
