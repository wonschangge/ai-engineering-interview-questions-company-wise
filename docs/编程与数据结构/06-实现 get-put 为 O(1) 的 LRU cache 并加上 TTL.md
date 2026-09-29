---
type: question
id: coding-06
topic: 编程与数据结构
order: 6
question: 实现一个 get/put 为 O(1) 的 LRU cache，然后加上 TTL。
question_en: Implement an LRU cache with O(1) get/put, then add TTL.
asked_at: [OpenAI, xAI, 阿里巴巴（Qwen）]
level: 进阶
tags: [lru, ttl, 数据结构, 缓存]
sources:
  - title: collections — OrderedDict（Python 标准库文档）（延伸）
    url: https://docs.python.org/3/library/collections.html
    author: Python Software Foundation
    published: ""
  - title: Key eviction（Redis 文档）（延伸）
    url: https://redis.io/docs/latest/develop/reference/eviction/
    author: Redis
    published: ""
related: [system-design-09, inference-serving-05, coding-07, system-design-10]
updated: 2026-09-28
---

## 一句话答案

> 两个结构分工：哈希表把「按键定位节点」做成 $O(1)$，双向链表把「摘下来、挂到头」做成 $O(1)$，链表从头到尾的顺序就是 LRU 顺序；再配一对哨兵头尾节点，边界判断就全消失了。TTL 加的不是一个时间戳，而是三个决策——过期语义（写入时固定还是读时续期）、内存回收方式（惰性删除保正确性 + 有界批量清扫保内存）、过期项算不算占容量（不算，且淘汰顺序必须是「先清过期、再按 LRU」）。离上线还差三样：按字节而不是按条数记账、命中率与过期率可观测、并发下分片而不是一把大锁。

## 面试官在考什么

- **双结构的必然性**：能讲清「哈希表只管定位、链表只管顺序」，缺一个就退化——单链表摘节点要前驱指针，`list` 删中间元素要搬移（实测容量 ×4 时单次 get 耗时 ×4.0，即 $O(n)$）。
- **指针操作写不写全**：四个变换（命中摘除、命中挂头、新增挂头、超容摘尾）有没有漏；**哨兵节点**是区分「真写过链表」与「背过题解」的分水岭，面试官会先扫这里。
- **两个实现与取舍**：手写双向链表（功底）与 `OrderedDict`（工程判断）都要能写；复杂度口径是**摊还** $O(1)$，不是「绝对 O(1)」。
- **TTL 的三个决策与 LRU 的交互**：只会在节点上加一个 `expire_at` 的答案停在第一层；能说出惰性删除 + 主动清扫、按 `expire_at` 的最小堆、过期项不占容量，才进第二层。

常见错误答案：

- 「`dict` 在 3.7+ 保插入序，拿它记访问顺序就是 $O(1)$ LRU」——`dict` 没有 `move_to_end`，命中后把它移到头部做不到；改用 `list` 记顺序，`remove` 又是 $O(n)$。
- 「TTL 就是节点里加个时间戳，读的时候比一下」——漏了主动回收与容量记账：实测按常驻条数记账时，常驻条目里 80.3% 是尸体，并且多淘汰了 125,644 次活数据。

## 原理与推导

### 1. 为什么必须两个结构

| 结构 | 能提供 | 缺什么 |
| --- | --- | --- |
| 哈希表 `key -> node` | 按键 $O(1)$ 定位、$O(1)$ 增删 | 不记录访问先后，不知道该淘汰谁 |
| 双向链表 | $O(1)$ 摘任意节点、$O(1)$ 头尾插删，顺序即新鲜度 | 按键定位要 $O(n)$ 遍历 |

两者互补：哈希表存**节点引用**（不是值），链表按「最近使用 → 最久未使用」排列；拿 `list` 记顺序则是命中路径上就要 $O(n)$ 搬移。容易漏的细节是节点里必须存 `key`——淘汰尾部节点时要用它反向删哈希表项。

### 2. 四个指针变换与哨兵

约定 `head` 哨兵一侧是最近使用、`tail` 哨兵一侧是最久未使用，所有操作都归结为两个原语：`_unlink`（摘掉节点）与 `_push_front`（挂到 head 之后）。

| 操作 | 哈希表 | 链表 |
| --- | --- | --- |
| `get` 命中 | 不动 | `_unlink(node)` + `_push_front(node)` |
| `put` 已存在 | 不动 | 改值 + `_unlink` + `_push_front` |
| `put` 新键 | 插入 | `_push_front`；超容量则 `_unlink(tail.prev)` 并删表项 |

`get` 未命中只查一次表、不动链表。哨兵的价值：没有哨兵要特判「链表为空」「删的是头」「删的是尾」「删的是唯一节点」四种情况；有哨兵后 `node.prev`、`node.next` 永远非空，`_unlink` 里一个 `if` 都不需要。

**复杂度口径。** `get`/`put` 各做常数次字典操作与常数次指针赋值：字典是均摊 $O(1)$（rehash 摊还），链表原语严格 $O(1)$，所以整体**摊还 $O(1)$、与容量无关**——实测手写版 get 在 5 万 / 20 万 / 100 万条目下为 392.7 / 390.5 / 393.0 ns，规模涨 20 倍而单次耗时不变；put 为 534.4 / 643.2 / 860.2 ns，那 1.6 倍增长来自节点分配与 cache 局部性，是常数因子而不是渐近项。

`OrderedDict` 内部就是「哈希表 + 双向链表」（C 实现），映射关系一一对应：`move_to_end(key)`（默认 `last=True`）等价于 `_push_front`，`popitem(last=False)` 等价于摘 `tail.prev`。两个坑：`popitem()` 在空表上抛 `KeyError`，要先判 `len`；3.7+ 的普通 `dict` 保插入序但**没有** `move_to_end`（本机实测 `hasattr({}, 'move_to_end')` 为 `False`），不能拿它顶替。生产代码优先用库：实测 get 快约 2.4 倍（165.0 ns vs 390.5 ns）、put 快约 4.0 倍（161.4 ns vs 643.2 ns，都是 20 万条目下的中位数），且少一类指针 bug。OrderedDict 版的全部实现就是两处调用：命中后 `self._d.move_to_end(key)`，超容量时 `self._d.popitem(last=False)`。

### 3. TTL 的三个决策

- **过期语义**：读路径惰性删除（保证「读到的值一定没过期」）+ 写路径或后台有界批量清扫（保证内存能回收），两者都要。滑动 TTL（读命中续期）是 session 超时语义，每次读都要刷新过期时间并往堆里塞新条目——实测 1000 个键读 100 万次会把堆从 1000 条撑到 1,001,000 条（约 95 MiB）。
- **过期索引**：按 `expire_at` 排序的最小堆。堆性质给出关键剪枝「堆顶未过期 ⇒ 其余全部未过期」，于是清理就是从堆顶连续弹出。堆不支持删除任意元素，键被重新 `put` 或淘汰后，堆里旧条目用一个自增 `seq` 版本号作废（弹出时比对 `node.seq != seq` 即跳过），代价是作废条目占内存，必须当指标暴露（实测 145,334 条）。代价还有复杂度：堆的 `heappush` / `heappop` 是 $O(\log n)$，所以 TTL 版的 `put` 不再是 $O(1)$ 而是 $O(\log n)$（`get` 仍是摊还 $O(1)$，`sweep(batch)` 最坏 $O(\text{batch} \cdot \log n)$）；要严格 $O(1)$ 且不作废可换时间轮，代价是精度受槽粒度限制。
- **过期与容量的交互**：淘汰顺序必须是「先清过期项、再按 LRU」，且过期项**不算**占容量。按常驻条数记账就会出现「看起来满了、其实大半是尸体」：同一负载下错误记账的版本平均常驻 982.1 条而活数据只有 193.4 条（尸体 80.3%）；正确记账的版本常驻 193.4 条、全是活的、淘汰活数据 0 次。清扫还必须**有界**：20 万条同时过期时一次清完 575.3 ms，而单次最多清 64 条只要 0.27 ms——把清尾工作切成有界批次（$O(\text{batch} \cdot \log n)$）才不会在请求路径上砍出长尾。

**为什么工业实现敢用近似 LRU。** 精确 LRU 的代价是每个条目多一组链表指针（实测相对裸 `dict` 每条多 80.0 B），并且每次访问都要改全局有序结构（并发下就是锁或原子操作）。Redis 改成随机采样若干候选、淘汰其中「最久未访问」的那个，用 `maxmemory-samples` 控制采样数（默认 5，文档称调到 10 时非常接近精确 LRU，power-law 访问下差异很小）；同时把**过期**与**内存淘汰**做成两套正交机制：`volatile-*` 只淘汰带 TTL 的键，`allkeys-*` 处理全部键。实测也说明近似够用：热冷混合负载下精确 LRU 73.37%、采样 5 个 71.77%、FIFO 64.89%。

## 数值与代码验证

环境 Python 3.10.12 / AMD Ryzen 9 8945HX（16 核 32 线程，跑批时 load average 约 25–30，故绝对 ns 数偏保守），实现只用标准库（numpy 2.2.6 仅用于生成 Zipf 负载）。实现脚本 `.work/coding06_lru_ttl.py`、测试与基准脚本 `.work/coding06_run.py`、断言片段 `.work/coding06_snippet.py`、计时复核脚本 `.work/coding06_bench2.py`（交错采样、5 次取中位数；都在仓库根 `.work/` 下，未纳入版本控制），命令 `python3 .work/coding06_run.py`，30 条断言全部通过。内存、命中率、TTL 记账这类与调度无关的数字两次运行逐位一致，计时类数字随负载浮动，正文一律以 `.work/coding06_bench2.py` 的中位数为准。节点 `_Node` 用 `__slots__` 只留 6 个槽位（`key`、`value`、`prev`、`next`、`expire_at`、`seq`），两个实现共用它；为省篇幅，代码块省略了 import、`_Node`、`TTLCache.__init__`（签名 `TTLCache(capacity, default_ttl=None, clock=time.monotonic, count_expired_in_capacity=False, sweep_every=256, sweep_batch=64)`）与 `__len__`、`keys_mru_to_lru`、`_now`、`_expired`、`_drop`、`_maybe_sweep`，完整可运行版本见 `.work/coding06_lru_ttl.py`；断言片段就是把文档里这两段代码补齐上述省略成员后实跑通过的。

### 参考实现 1：哈希表 + 带哨兵的双向链表

```python
class LRUCache:
    """哈希表 + 双向链表。get/put 均为 O(1)（摊还，无 rehash 之外的分支）。

    链表约定：head 哨兵一侧是「最近使用」，tail 哨兵一侧是「最久未使用」。
    两个哨兵让 _unlink / _push_front 永不出现 None 判断。
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self.capacity = capacity
        self._map: Dict[Hashable, _Node] = {}
        self._head = _Node()          # 哨兵
        self._tail = _Node()          # 哨兵
        self._head.next = self._tail
        self._tail.prev = self._head
        self.hits = 0
        self.misses = 0
        self.evicted = 0

    # --- 链表原语：只有这两处会改指针 ---

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _push_front(self, node: _Node) -> None:
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    # --- 对外契约 ---

    def get(self, key, default=None):
        node = self._map.get(key)
        if node is None:
            self.misses += 1
            return default
        self._unlink(node)
        self._push_front(node)
        self.hits += 1
        return node.value

    def put(self, key, value) -> None:
        node = self._map.get(key)
        if node is not None:                 # 已存在：更新值并提到头部
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        node = _Node(key, value)             # 新键：插入头部 + 登记哈希表
        self._map[key] = node
        self._push_front(node)
        if len(self._map) > self.capacity:   # 超容量：摘尾部（最久未使用）
            victim = self._tail.prev
            self._unlink(victim)
            del self._map[victim.key]
            self.evicted += 1
```

### 参考实现 2：TTL 的惰性删除、有界清扫与过期优先淘汰

```python
# TTLCache 的三个方法（_unlink / _push_front 与实现 1 完全相同）
    def sweep(self, now: Optional[float] = None, max_remove: Optional[int] = None) -> int:
        """从最小堆顶部弹出已过期且仍有效的条目。返回删除条数。"""
        if now is None:
            now = self._now()
        removed = 0
        heap = self._heap
        while heap:
            if max_remove is not None and removed >= max_remove:
                break
            expire_at, seq, key = heap[0]
            if expire_at > now:
                break                     # 堆顶未过期 → 其余都没过期
            heapq.heappop(heap)
            node = self._map.get(key)
            if node is None or node.seq != seq:
                self.stats["stale_heap"] += 1
                continue                  # 已被删或已被重新 put，作废
            self._drop(key, "expired_swept")
            removed += 1
        return removed

    # --- 对外契约 ---

    def get(self, key, default=None):
        node = self._map.get(key)
        if node is None:
            self.stats["misses"] += 1
            return default
        if self._expired(node, self._now()):
            self._drop(key, "expired_lazy")   # 惰性删除：先删再报 miss
            self.stats["misses"] += 1
            return default
        self._unlink(node)
        self._push_front(node)
        self.stats["hits"] += 1
        return node.value

    def put(self, key, value, ttl: Optional[float] = None) -> None:
        now = self._now()
        if ttl is None:
            ttl = self.default_ttl
        expire_at = None if ttl is None else now + ttl
        self._writes += 1

        node = self._map.get(key)
        if node is not None:                      # 更新：换值、换过期时间、换 seq
            node.value = value
            node.expire_at = expire_at
            self._seq += 1
            node.seq = self._seq
            if expire_at is not None:
                heapq.heappush(self._heap, (expire_at, node.seq, key))
            self._unlink(node)
            self._push_front(node)
        else:
            self._seq += 1
            node = _Node(key, value, expire_at, self._seq)
            self._map[key] = node
            self._push_front(node)
            if expire_at is not None:
                heapq.heappush(self._heap, (expire_at, node.seq, key))

        if not self.count_expired_in_capacity:
            self.sweep(max_remove=max(self.sweep_batch, len(self._map) - self.capacity))
        while len(self._map) > self.capacity:     # 仍然超容量 → 按 LRU 淘汰活数据
            victim = self._tail.prev
            self._drop(victim.key, "evicted_lru")
        self._maybe_sweep()
```

`_maybe_sweep()` 每 `sweep_every` 次写调用一次 `sweep(max_remove=sweep_batch)`，把「一次清很多」摊平成「每次清一点」；`_now()` 就是 `self._clock()`，`_expired(node, now)` 判断 `node.expire_at is not None and node.expire_at <= now`，`_drop(key, reason)` 从哈希表摘掉节点并累加 `stats[reason]`，`_unlink`/`_push_front` 与实现 1 完全相同。除上面的断言外，跑批还验证了：同一时刻插入 8 项（`ttl=5`）会在到期瞬间被 `live_len()` 清空、`ttl=None` 在时钟前进 10^9 后仍命中、错误记账版本会淘汰活数据并留下过期尸体、`sweep` 单次耗时 0.015 / 0.266 / 15.282 / 575.345 ms（对应 `max_remove` 1/64/4096/200000）。

### 边界用例与断言（`python3 .work/coding06_snippet.py` 的输出为 `all assertions passed`）

```python
from coding06_lru_ttl import LRUCache, TTLCache

now = [1000.0]                      # 时钟依赖注入：显式推进，不 sleep
clock = lambda: now[0]

c = LRUCache(2)
c.put(1, "a"); c.put(2, "b")
assert c.get(1) == "a"              # 命中：1 变成最近使用
c.put(3, "c")                       # 超容量：淘汰最久未使用的 2
assert c.get(2) is None and c.keys_mru_to_lru() == [3, 1]

t = TTLCache(capacity=2, default_ttl=10.0, clock=clock, sweep_every=0)
t.put("k", "v")
now[0] += 9.999
assert t.get("k") == "v"            # 到期前 1 ms 仍命中
now[0] += 0.001                     # 恰好等于 expire_at → 失效
assert t.get("k") is None and t.stats["expired_lazy"] == 1

print("all assertions passed")
```

| 单次 put / get（ns，5 次取中位数） | 20 万条目 | 100 万条目 |
| --- | --- | --- |
| 手写 LRU put / get | 643.2 / 390.5 ns | 860.2 / 393.0 ns |
| OrderedDict put / get | 161.4 / 165.0 ns | 185.8 / 166.1 ns |

反例（`list` 记顺序）单次 get 在 N=2000/8000/32000 时为 10.9/43.6/190.9 us，容量 ×4 耗时 ×4.0、×4.4，符合 $O(n)$：线性外推到 100 万条目约 6.0 ms/次，同样 100 万次 get 要 1.7 小时，而手写 LRU 只要 0.39 s。tracemalloc 实测每条目内存：裸 `dict` 80.4 B、OrderedDict 133.3 B、手写链表 160.4 B、TTL 版 284.3 B（N=10 万），所以容量要按字节给而不是按条数给——用 KV cache 举例（口径与仓库其它专题一致，LLaMA-3-70B：80 层、8 个 KV 头、$d_h=128$、bf16）：

$$\text{KV per token} = 2 \times 80 \times 8 \times 128 \times 2\ \text{B} = 327{,}680\ \text{B} = 320\ \text{KiB}$$

32k token 上下文就是 $32768 \times 320\ \text{KiB} = 10\ \text{GiB}$；反过来 1 GiB 只放得下 3,276 个 token 的 KV（$1073741824 / 327680 = 3276.8$）。命中率对照（10 万键、20 万次访问、容量 1000 = 键空间 1%，原始打印）：

```text
  负载              精确 LRU      FIFO      随机淘汰     采样 LRU(s=5)
  Zipf(1.2)       12.41%    12.34%    12.41%    12.38%
  热冷混合            73.37%    64.89%    65.03%    71.77%
  顺序扫描             0.00%     0.00%    20.21%     0.50%
```

Zipf(1.2) 负载下（最热 1% 的键只占 18.1% 访问）随机淘汰几乎不输精确 LRU（12.41% vs 12.41%），说明「策略选对」比「精确」更重要；热冷混合下 LRU 明显赢 FIFO（73.37% vs 64.89%），这是 LRU 的收益来源；顺序扫描时 LRU 归零（0.00%），因为扫描把缓存整个冲了一遍——生产里要专门防扫描；同理，一批键同时过期就是自己制造的雪崩，TTL 必须加随机抖动，本地缓存与共享层的分层取舍见 [[system-design-09]]。并发（8 线程 × 2 万次迭代、容量 1000、5 次取中位数，原始打印）：

```text
  UnsafeLRU : 完成 74714/160000 次迭代, 最终 size=1002 (期望 <= 1000), 异常 6 个：AttributeError('NoneType' object has no attribute 'next')
  分片  1 把锁:  71,402 ops/s（5 次中位数）, 最终 size=1000, 异常 无
  分片 16 把锁: 299,187 ops/s（5 次中位数）, 最终 size=1000, 异常 无
  单线程基线：987,425 ops/s
```

无锁版不只是「结果可能不准」：它直接抛异常打断工作线程（8 个线程只完成 74,714/160,000 次迭代，本次调度下是 `AttributeError`，链表指针已被写坏，换个调度也可能是 `KeyError`），最终 `size` 还比容量大。分片到 16 把锁后吞吐约为单锁的 4.2 倍（4 把锁时 148,418 ops/s，即 2.1 倍），但仍只有单线程基线的 30%，这是 CPython GIL 加锁开销的实证。分片哈希必须用稳定哈希（crc32/murmur），不能用内置 `hash()`——它对 `str` 按进程随机加盐（本机实测两次运行的 `hash('x')` 不同），多进程按同一键路由会落到不同分片。

## 常见追问

- **为什么不用 `functools.lru_cache`**：key 是函数参数元组，无法按业务键失效；没有 TTL、没有字节记账、不暴露条目级指标，`maxsize=None` 还是无界常驻内存。它是纯函数记忆化，不是业务缓存。
- **LFU 与 LRU 怎么选**：LRU 抗不住扫描（实测扫描负载命中率 0.00%）；LFU 抗扫描但要计数器与衰减（Redis 用 Morris 近似计数 + `lfu-decay-time`，计数器饱和在 255），新键还有冷启动劣势，所以要能说出「LFU 没有衰减机制、历史热点会永远赖着不走」。通用场景 LRU 更稳。
- **Redis 为什么用近似 LRU**：精确 LRU 要额外元数据（实测 80.0 B/条）并让每次访问都改全局结构；采样在 power-law 下几乎等价，还能用 `maxmemory-samples` 调精度与 CPU 的平衡。
- **并发、大对象与一致性怎么办**：先分片、每片一把锁、锁内只做指针操作不做 IO，注意容量静态切分后键分布倾斜会出现一片淘汰另一片空闲；值大小差异大时按字节记账并给单条上限（超限绕过缓存），反过来小对象的元数据开销不可忽略——80.0 B/条的 LRU 指针在 100 万条时约 76 MiB；缓存与数据库的一致性走 cache-aside（先写库、再失效缓存），不做双写，本地 LRU 只做一级、跨进程与跨机交给共享层（[[coding-07]]）。

## 公司变体

题库记录这题出现于 OpenAI、xAI、阿里巴巴（Qwen），三家都是「编码轮里边写边加需求」的形式而非笔试题：先把 $O(1)$ LRU 写对，再当场加 TTL。

- **OpenAI / xAI**：追问偏工程实现——指针有没有一次写对、哨兵用没用、现场加需求（改成线程安全、改成按字节淘汰）时结构还撑不撑得住；xAI 允许自选语言，选 Python 就要准备回答 GIL 与锁的代价。依据是同一份题库清单里 coding-06 与 [[coding-07]]、[[coding-08]] 并列，都是「先实现、再加需求」的编码题。
- **阿里巴巴（Qwen）**：追问偏缓存生产经验——命中率与容量指标、热点 key、本地缓存与分布式缓存分层，以及这题与 KV cache / prefix cache 的联系（[[inference-serving-05]]）。
- 口径说明：题库只记录了「出现于哪些公司」，没有公司专属题干变体，上面按公司给出的追问方向是常见考法的推断，不是官方口径。

## 相关题目

- [[system-design-09]]：LLM gateway 的缓存、限流与预算，本文的分片与锁粒度、缓存击穿在那里是主线。
- [[inference-serving-05]]：prefix caching 用 KV block 做 LRU，正是「大对象按字节记账」的现实版本（320 KiB/token）。
- [[evaluation-07]]：命中率、淘汰率、过期率要作为一等指标暴露，这一节给了埋点清单。

## 参考资料与归属

- [collections — OrderedDict（Python 标准库文档）](https://docs.python.org/3/library/collections.html)（延伸）：`move_to_end(key, last=True)` 与 `popitem(last=True)` 的语义与 $O(1)$ 口径。文中相关行为（空表 `popitem` 抛 `KeyError`、普通 `dict` 无 `move_to_end`、`popitem(last=False)` 返回最早插入项）均在本机 Python 3.10.12 上实跑确认。
- [Key eviction（Redis 文档）](https://redis.io/docs/latest/develop/reference/eviction/)（延伸）：`maxmemory`、`allkeys-*` 与 `volatile-*` 两套策略的正交分工、近似 LRU 的采样做法与 `maxmemory-samples` 默认值、LFU 的 Morris 计数器与衰减参数、用 `keyspace_hits/keyspace_misses` 与 `evicted_keys`、`expired_keys` 判断命中率与过期情况。
- 实测数据（字节数、命中率、TTL 记账、堆长度）来自本机运行 `.work/coding06_run.py`，ns/op 与 ops/s 来自 `.work/coding06_bench2.py`（交错采样、5 次取中位数）；LLaMA-3-70B 的 KV cache 口径（320 KiB/token）与本仓库其它专题一致。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
