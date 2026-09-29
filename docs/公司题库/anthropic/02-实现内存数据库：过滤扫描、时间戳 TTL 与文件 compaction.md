---
type: question
id: anthropic-02
company: Anthropic
topic: coding
order: 2
question: 实现一个内存数据库：先支持 SET/GET/DELETE，再加上带过滤的扫描，然后是带时间戳的 TTL，最后是文件 compaction。
question_en: Build an in-memory database: SET/GET/DELETE first, then filtered scans, then TTL with timestamps, then file compaction.
asked_at: []
level: 高阶
tags: [内存数据库, kv-store, ttl, compaction, 实现题]
sources:
  - title: Key eviction（Redis 文档）（延伸）
    url: https://redis.io/docs/latest/develop/reference/eviction/
    author: Redis
    published: 
  - title: Log-structured merge-tree（维基百科）（延伸）
    url: https://en.wikipedia.org/wiki/Log-structured_merge-tree
    author: Wikipedia
    published: 
  - title: decimal — 十进制定点与浮点运算（Python 标准库文档）（延伸）
    url: https://docs.python.org/3/library/decimal.html
    author: Python Software Foundation
    published: 
related: [coding-06, coding-07, coding-11, rag-11]
updated: 2026-09-29
---

## 一句话答案

> 四层要一次规划、分层落地：第 1 层先把契约与不变量冻结（未命中返回哨兵、删除幂等、写入时拷贝、删除必须同时摘掉所有索引），并把四个扩展点预留在第一层——可注入时钟、结构化谓词接口、删除的墓碑语义、唯一写入口 `append(op, key, value, expire_at)`；第 2 层的 scan 用有序索引下推 prefix，并明确「先过滤再截断」；第 3 层 TTL 用单调时钟加读路径惰性删除加写路径有界清扫，且过期项不占容量；第 4 层把内存表换成追加段加墓碑，compaction 走「写新段 → fsync → 原子 rename → 删旧段」，只在后台限速摊销。真正的风险不是多写 50 行，而是第一层把删除写成原地物理删除、把时钟写死成 `time.time()`——那样第 3、4 层必须推翻重来。

## 面试官在考什么

- **契约能力**：黑盒评测器只认契约。未命中返回什么、删不存在的键返回什么、到期边界用 `<` 还是 `<=`、scan 的游标语义，这些不先定，第二层起每加一条需求都要返工。
- **分层与扩展点**：一次只加一层，且上一层的断言不退化（同专题 [[coding-07]] 考的正是「需求持续加码时如何不失控」）。
- **过滤的成本意识**：说清「先过滤再截断」和选择性 $s$ 带来的 $1/s$ 候选放大，而不是把过滤写成全表物化后的一个 `if`。
- **时间语义与工程闭环**：TTL 从哪一刻起算、时钟从哪来、谁负责收内存、过期项算不算容量；段与墓碑、崩溃恢复、空间放大与写放大、compaction 的触发与限速。
- 常见错误答案：一是「TTL 就是在节点里加个时间戳，读的时候比一下」——没有回收、没有容量记账、没有时钟口径；二是「compaction 就是定期把文件重写一遍」——不说墓碑、不说 rename 原子性、不报空间放大与写放大。

## 原理与推导

### 先冻结契约与不变量

| 方法 | 签名 | 未命中与边界 | 不变量 |
| --- | --- | --- | --- |
| `set` | `set(k, v, ttl=None) -> bool` | 覆盖写返回 True；`ttl<0` 等于写入即过期 | 写入时拷贝值；TTL 从写入时刻起算（固定窗口） |
| `get` | `get(k) -> bytes \| MISS` | 返回哨兵 `MISS`，不抛 KeyError | 命中 ⇒ 值一定没过期 |
| `delete` | `delete(k) -> bool` | 键不存在返回 False（幂等）；逻辑已过期也返回 False | 删除后 scan 一定不含该键 |
| `scan` | `scan(prefix, predicate, limit, cursor) -> (items, next_cursor)` | 空库返回 `([], None)` | 顺序是 key 字节序；先过滤再截断；不含过期项 |

四个「只改一处就能加进来」的扩展点，必须在第 1 层就留下：**时钟注入**（构造时传 `clock`，而不是在方法体里调 `time.time()`）、**结构化谓词**（`predicate(key, value) -> bool`，不是字符串表达式，后者的解析、类型检查与注入问题会一起进来）、**删除语义**（从第一层起 delete 就是逻辑删除：摘索引、留一个可被合并的墓碑，到第 4 层墓碑才落盘）、**唯一写入口**（所有变更加 `expire_at` 后收敛到一次 `append`，第 1 层它是内存写，第 4 层它变成段追加）。

### 第 1 层：数据结构与内存记账

哈希表存值、另建有序索引供扫描，删除要同时摘干净：只摘哈希表就出现「删了但还能扫出来」。写入时把 `bytearray`、`memoryview` 转成 `bytes`，否则调用方复用 buffer 会污染库内数据——一行代码，但评测器会专门测。容量按**字节**记账，不按条数。下表是同一台机器的 tracemalloc 实测（N=10 万，每次快照里由被测结构自己新建键值对象，覆盖容器与所存对象，3 次取中位数）：

| 每条目内存（N=10 万） | 实测 |
| --- | --- |
| 裸 `dict` / `OrderedDict`（int→int，复现 [[coding-06]] 口径） | 80.4 B / 133.3 B |
| 本题 MemDB（bytes 键 + 64 B 值，无 TTL） | 190.1 B |
| 本题 MemDB（bytes 键 + 64 B 值，带 TTL） | 286.1 B |

注意这不代表 int 键可用：`set` 里的 `len(k) + len(v)` 与 `scan` 的 `k.startswith(prefix)` 都要求键和值是 bytes 系，传 `int` 键会直接抛 `TypeError`；上表的 `dict`/`OrderedDict` 两行只是 [[coding-06]] 的基线口径。比裸 `dict` 贵的那部分不在哈希表上，而在另外两份结构：`_keys` 有序索引（Python list，等于把键引用再存一遍；生产要换跳表或 LSM 的有序 run，见上文注释）与每条一个 `(value, expire_at, write_seq)` 三元组。TTL 再加一份索引，单价 **96 B/条**（286.1 − 190.1，等于一个三元组加堆里的一项，与 [[coding-06]] 在链表结构上量到的 96 B 相合）：在 64 B 的小值上等于 +150%，在 LLaMA-3-70B 一个 token 的 KV cache（仓库统一常数：80 层、8 个 KV 头、$d_h=128$、bf16，$2\times80\times8\times128\times2=327{,}680$ B = 320 KiB）上只占 0.03%（LLaMA-3-8B 是 128 KiB）。同一条元数据对不同大小的值含义完全不同，所以记「条数」迟早算错。

并发只需一句话：现在单线程评测，生产要把键空间分片、每分片单写者、读走快照，分片哈希用稳定哈希；[[coding-06]] 实测加锁分片后吞吐只有单线程基线的 30%。

### 第 2 层：带过滤的扫描

三件事先定死：**顺序稳定**（按 key 字节序，否则结果不可复现）、**先过滤再截断**、**游标失效策略二选一并且写明**。策略一是快照隔离：游标里带 epoch，开启扫描之后新写入的键对本次扫描不可见（不重复、可能遗漏，因为中途被删的键会直接消失）；策略二是允许重复与遗漏、由调用方幂等重放。两者都可以，不能的是不说。

成本模型：要凑满 $k$ 条结果，在选择性 $s$ 下期望要检查 $k/s$ 行，成本 $\approx (k/s)\cdot c$，其中 $c$ 是每行谓词求值成本。本机实测（$10^6$ 条，Python 逐行谓词，$c\approx 0.53\sim0.63\ \mu s$）：$s=1\%$、$k=10$ 要 477 µs 走 901 行；$s=0.1\%$ 要 5.7 ms 走 9001 行——成本正好随 $1/s$ 放大；同口径的 [[rag-09]] 给过另一组算例：$k=10$、$s=1\%$ 时期望只剩 0.1 条，要凑满 10 条得把候选深度放大到 $k/s=1000$。而「先 `limit=10` 再过滤」的写法只要 8 µs，返回 1 条（正确应 10 条），因为命中的行不在前 10 行里——**它又快又错，是这道题最常见的一处翻车**。过滤要下推到扫描过程本身：prefix 走有序索引（本例中前缀一旦越过 `prefix` 区间就直接收工），等值/范围条件高频时值得建二级索引把 $s$ 提到 100%。同时扫描不要全量物化：用游标逐页返回。仓库同口径的代价参考是 [[coding-11]] 的 $10^6$ 条全扫 58.4 ms（向量化，52.6 GB/s）与纯 Python 逐条 43.4 µs/条（外推 $10^8$ 要 72 分钟），全量物化加排序还会把内存峰值抬到库大小的两倍。最后一个反直觉坑：顺序扫描会冲掉缓存热点，[[coding-06]] 实测扫描负载下 LRU 命中率归零（0.00%，热冷混合下精确 LRU 73.37%、采样 5 个 71.77%），所以一旦接了查询缓存或缓冲池，scan 必须旁路缓存或只按需回填。

### 第 3 层：TTL 的三个决策

**① 过期语义**：写入时固定（TTL 从 SET 起算）还是读时续期（滑动过期）。选前者：可复现、内存可预测、不会「越热的键越不过期」，而且滑动过期要在 get 里写状态，与段文件的追加写路径冲突。到期边界取 `expire_at <= now` 即失效（恰好等于到期时刻算过期），并把这个选择写进断言。**② 回收方式**：两个都要——读路径惰性删除保证「读到的值一定没过期」，写路径或后台的有界清扫保证内存真的降下来。过期索引用小根堆：堆顶未到期则其余全未到期；被覆盖写的键在堆里留下作废条目（当指标暴露，[[coding-06]] 实测 145,334 条），代价是 `put` 从 $O(1)$ 变成 $O(\log n)$。**③ 容量记账**：过期项不算容量，淘汰顺序是先清过期再按 LRU/FIFO；按常驻条数记账的版本里 80.3% 是尸体（常驻 982.1 条、活数据 193.4 条），多淘汰了 125,644 次活数据，而正确记账版本淘汰活数据 0 次（均为 [[coding-06]] 实测）。

清扫的实测成本（20 万条同时过期，本机中位数）：只弹堆不维护索引 248 ms（1.24 µs/条），墓碑加一次重建有序索引 369 ms（1.85 µs/条），而「有序数组 + bisect 删除」4.08 s（20.4 µs/条），其中索引维护占 94%。所以第一个该改的不是批量大小，是索引的删除复杂度。单批 64 条的清扫成本还随规模线性上涨（6.5 → 37.4 µs/条，n 从 2.5 万到 20 万），因为删除是 $O(n)$ memmove：按 $8\times n^2/2$ 算搬移 2.44 → 159.5 GB、等效 26–45 GB/s，正是 memmove 的量级——这就是「删除要同时从所有索引摘干净」的账单，对照 [[coding-06]] 的链表实现（单批 64 条 0.266 ms、20 万条一次清完 575.3 ms）。

**时钟与抖动**必须在第一层就注入：进程内用单调时钟（`clock=time.monotonic`，测试可推进、不怕 NTP 回拨），跨进程或落盘用同一时钟源的墙钟并做单调性防护，`elapsed <= 0` 的唯一安全动作是既不补充也不回退。[[coding-07]] 的实测：两个实例时钟差 1 ms、$r=100/\text{s}$ 时 1 s 内 6000 次请求放行 359 次（理论上限 110），差 50 ms 时 6000 次全部放行。同一批键同时过期等于自己制造雪崩，TTL 要加随机抖动、清扫要限批。

### 第 4 层：段文件与 compaction

形态：内存表 + 追加段（每条记录 = 19 B 头 + key + value，头里带 crc32 与 `expire_at`）+ 墓碑；删除写墓碑而不是原地改；读路径按段号升序重放、新段覆盖旧段；`get` 走内存索引，冷读才回落到段。compaction 协议顺序不能变：① 合并已封存段，丢掉墓碑与已过期记录；② 写 `.tmp` 并 fsync；③ `os.replace` 原子改名是唯一切换点；④ fsync 目录、删旧段。三种崩溃都能收敛：改名之前崩 = 什么都没发生（旧段完好，`.tmp` 不参与重放）；改名之后、删旧段之前崩 = 新旧段并存，重放时新段号更大、结果一致；最后一条记录写了一半 = crc32 校验失败、截断丢弃。触发策略看段数、墓碑占比、层数三个旋钮：同一个库实测（2 万键、覆盖写一遍、删 30%、2 千条 TTL 过期）有 17 个段、盘上 3.95 MB 对应 1.04 MB 活载荷，空间放大 3.81×、墓碑占合并记录的 27.3%（超过 20% 阈值才压），合并后 2 个段、1.30 MB、空间放大 1.26×，丢掉 6,000 条墓碑与 2,000 条过期记录。自洽检查：合并后残余的空间放大正好等于记录头开销，93 B/条 ÷ 74 B/条 = 1.26×，与实测一致，说明既没有残余垃圾也没有重复记录。写放大 = 累计写盘 ÷ 最终活载荷 = 5.07×（含两遍全量写、墓碑与合并重写）；外部口径 [[rag-11]] 的 $W\approx 1+\log_{10}S$（$S$ 为每日新增段数）：1 s refresh 下 $S=86{,}400$、$W\approx 5.9\times$，51.2 GB 载荷落盘 302 GB、NVMe 2 GB/s 下 151 s，对照 2.87 GPU-h（10,332 s）的 embedding 只占 1.5%——该优化的是「什么时候压」，不是「压得多干净」。TTL 与 compaction 必须闭环：合并时把已过期键直接丢弃、墓碑也能被合并清掉，否则盘上永远留着逻辑上早死的数据。也不要在线压：段合并的算术强度接近 0，H100 的 roofline 拐点 $989\times10^{12}/(3.35\times10^{12})\approx295$ FLOPs/byte 说明它远在带宽侧、算力帮不上忙，[[inference-serving-03]] 里 PagedAttention 论文也明确否决了对数十 GB KV cache 做在线 compaction；本机 Python 逐条合并 3.13 µs/条（22,000 条 68.9 ms，fsync 只占 8.0 ms），换算到 51.2 GB 是上万秒 CPU——生产要用编译语言或列式批处理把它压到磁盘带宽，并限速分批，否则 compaction 自己就是 p99 长尾的来源。

**先写断言再写实现**——边界清单（每条都值得一个断言）：空库的 `get`/`delete`/`scan`；覆盖写；删不存在的键；连删两次；调用方改 buffer；TTL 剩余大于 0、恰好等于到期时刻、负 TTL、`ttl=None` 加时钟冻结或回拨；扫描的 prefix 边界（空 prefix、无匹配）、游标分页不重不漏、游标之后写入的键不可见；过期项不出现在 scan 里、也不占容量（容量压力下不淘汰活数据）；compaction 期间的读一致性；写完新段还没 rename 就断电；rename 之后没删旧段就断电；截断的尾记录；以及最后一条——**compaction 完成后目录字节数真的下降**，否则只是把垃圾搬到了新段。

## 数值与代码验证

三段代码在本机跑通（Python 3.10、单线程），所含断言全部通过；第三段给出段格式与 compaction 协议，其完整实现（含追加写、封段、崩溃恢复）承担下表的验算。数字是 3–5 次取中位数，机器负载会造成 10%–20% 波动。

```python
import bisect, heapq, time
MISS = object()                          # 未命中哨兵：不抛 KeyError，也不与空值混淆

class MemDB:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._kv = {}                    # key -> (value, expire_at, write_seq)
        self._keys = []                  # 有序扫描索引（生产：跳表或 LSM 的有序 run）
        self._heap = []                  # TTL 最小堆 (expire_at, write_seq, key)
        self._seq = self._writes = self._bytes = 0
        self.stats = dict(expired_lazy=0, expired_swept=0, deleted=0, evicted=0, stale_heap=0, examined=0)
    def set(self, k, v, ttl=None):
        if isinstance(v, (bytearray, memoryview)):
            v = bytes(v)                                      # 写入时拷贝
        exp = None if ttl is None else self._clock() + ttl     # TTL 从写入时刻起算
        old = self._kv.get(k)
        if old is None:
            bisect.insort(self._keys, k)
        else:
            self._bytes -= len(k) + len(old[0])                # 覆盖写：先退旧记账
        self._seq += 1; self._kv[k] = (v, exp, self._seq)
        self._bytes += len(k) + len(v)
        if exp is not None:
            heapq.heappush(self._heap, (exp, self._seq, k))
        self._writes += 1
        if self._writes % 64 == 0:
            self.sweep(64)                                     # 写路径：有界清扫
    def get(self, k):
        e = self._kv.get(k)
        if e is None: return MISS
        if e[1] is not None and e[1] <= self._clock():          # 恰好到期即过期
            self._drop(k, 'expired_lazy')                       # 读路径：惰性删除
            return MISS
        return e[0]
    def delete(self, k):
        e = self._kv.get(k)
        if e is None:
            return False                                        # 幂等
        if e[1] is not None and e[1] <= self._clock():
            self._drop(k, 'expired_lazy')
            return False                                        # 逻辑上早已不存在
        self._drop(k, 'deleted')
        return True
    def _drop(self, k, why):
        e = self._kv.pop(k); self._bytes -= len(k) + len(e[0])
        i = bisect.bisect_left(self._keys, k)
        if i < len(self._keys) and self._keys[i] == k:
            self._keys.pop(i)                                   # 从所有索引摘干净
        self.stats[why] += 1
    def sweep(self, batch):
        now, done = self._clock(), 0
        while done < batch and self._heap:
            exp, seq, k = self._heap[0]
            if exp > now:
                break                                           # 堆顶未到期 ⇒ 其余全未到期
            heapq.heappop(self._heap)
            e = self._kv.get(k)
            if e is None or e[2] != seq:                         # 被覆盖或被删的作废条目
                self.stats['stale_heap'] += 1
                continue
            self._drop(k, 'expired_swept')
            done += 1
        return done
    def scan(self, prefix='', predicate=None, limit=None, cursor=None):
        epoch, start = cursor if cursor else (self._seq, None)
        i = 0 if start is None else bisect.bisect_right(self._keys, start)
        out, examined = [], 0
        while i < len(self._keys):
            k, i, examined = self._keys[i], i + 1, examined + 1
            if prefix and not k.startswith(prefix):
                if k[:len(prefix)] > prefix:
                    break                                       # 越过 prefix 区间即可收工
                continue
            e = self._kv[k]
            if e[2] > epoch:                                    # 游标之后新写的键：快照不可见
                continue
            if e[1] is not None and e[1] <= self._clock():
                continue                                        # 扫描屏蔽过期项
            if predicate is not None and not predicate(k, e[0]):
                continue
            out.append((k, e[0]))
            if limit is not None and len(out) >= limit:
                break
        self.stats['examined'] += examined
        return out, (None if i >= len(self._keys) else (epoch, self._keys[i - 1]))
```

```python
now = [1000.0]
db = MemDB(clock=lambda: now[0])
assert db.get('x') is MISS and db.delete('x') is False and db.scan() == ([], None)
buf = bytearray(b'v1'); db.set('a', buf); buf[0:2] = b'XX'; assert db.get('a') == b'v1'
db.set('a', b'v2'); assert db.get('a') == b'v2'
assert db.delete('a') is True and db.delete('a') is False
db.set('t', b'x', ttl=5.0); now[0] += 4.999; assert db.get('t') == b'x'
now[0] += 0.001                              # 恰好等于 expire_at：失效
assert db.get('t') is MISS and db.stats['expired_lazy'] == 1
db.set('perm', b'x'); now[0] += 1e9; assert db.get('perm') == b'x'
for i in range(20): db.set(f'p:{i:02d}', str(i).encode())
page1, cur = db.scan(prefix='p:', limit=5)
db.set('p:05b', b'99')                       # 游标之后新写的键
page2, _ = db.scan(prefix='p:', limit=9, cursor=cur)
assert 'p:05b' not in [k for k, _ in page2]  # 快照语义：新键对本次扫描不可见
dead = MemDB(clock=lambda: now[0]); base = dead._bytes
for i in range(50): dead.set(f'dead{i:02d}', b'y' * 20, ttl=1.0)
now[0] += 2.0
assert dead.sweep(64) == 50 and dead._bytes == base   # 过期项归零：不占容量
print('layers 1-3 ok')
```

```python
import os, struct, time, zlib
OP_PUT, OP_DEL = b'P', b'D'
HDR = struct.Struct('>cHId')                 # op, klen, vlen, expire_at（0 表示永久）

def encode(op, k, v, exp):                   # 每条：crc32(4B) + 头(19B) + key + value
    kb = k.encode(); body = HDR.pack(op, len(kb), len(v), exp or 0.0) + kb + v
    return struct.pack('>I', zlib.crc32(body) & 0xffffffff) + body

def records(buf):                            # 截断或校验失败的尾记录一律丢弃
    off = 0
    while off + 4 + HDR.size <= len(buf):
        crc = struct.unpack_from('>I', buf, off)[0]
        op, klen, vlen, exp = HDR.unpack_from(buf, off + 4)
        end = off + 4 + HDR.size + klen + vlen
        if end > len(buf) or zlib.crc32(buf[off + 4:end]) & 0xffffffff != crc:
            return
        body = buf[off + 4:end]
        yield op, body[HDR.size:HDR.size + klen].decode(), body[HDR.size + klen:], exp
        off = end

rec = encode(OP_PUT, 'k', b'v', None)
assert list(records(rec + rec)) == [(OP_PUT, 'k', b'v', 0.0)] * 2
assert list(records(rec + rec[:7])) == [(OP_PUT, 'k', b'v', 0.0)]    # 截断的尾记录被丢弃

class SegStore:
    """__init__ 按段号升序把目录里的段重放进内存索引，再开一个活跃段；写路径 append
    追加记录、段满则 fsync 封段；删除只写墓碑；get 走内存索引，冷读才回落段文件。"""

    def compact(self):
        """写新段 → fsync → 原子 rename → 删旧段；rename 之前崩溃等于什么都没发生。"""
        if len(self._segs) < 2:
            return None                                          # 触发策略：段数不够不压
        now, merged = self._clock(), {}
        for no in sorted(self._segs):                            # 段号升序 ⇒ 新段优先
            with open(self._path(no), 'rb') as f:
                for op, k, v, exp in records(f.read()):
                    merged[k] = (op, v, exp)
        before = sum(os.path.getsize(self._path(no)) for no in self._segs)
        keep = {k: r for k, r in merged.items()
                if r[0] == OP_PUT and not (r[2] and r[2] <= now)}     # 墓碑与过期一起丢
        buf = b''.join(encode(op, k, v, exp) for k, (op, v, exp) in keep.items())
        tmp = os.path.join(self.dir, '.compact.tmp')
        with open(tmp, 'wb') as f:
            f.write(buf); f.flush(); os.fsync(f.fileno())         # ① 新段落盘
        os.replace(tmp, self._path(self._next))                   # ② 唯一切换点
        for no in self._segs:
            os.unlink(self._path(no))                             # ③ 删旧段
        self._segs, self._next = [self._next], self._next + 1
        self.bytes_written += len(buf)
        return dict(before=before, after=len(buf), merged=len(merged), live=len(keep))
```

| 验算（口径） | 实测 |
| --- | --- |
| 全表游标扫描 $10^6$ 行（limit=10 万/页，Python 逐行） | 811 ms，811 ns/行 |
| 选择性 $s=1\%$ / $0.1\%$、`limit=10`（先过滤再截断） | 10 条，477 µs / 5.7 ms，检查 901 / 9001 行（529 / 632 ns/行） |
| 先 `limit=10` 再过滤（错误顺序） | 1 条（正确 10 条），8 µs |
| 20 万条同时过期：只弹堆 / 墓碑加一次重建 / 有序数组 bisect 删除 | 248 ms（1.24 µs/条）/ 369 ms（1.85 µs/条）/ 4.08 s（20.4 µs/条，索引维护占 94%） |
| 单批 64 条清扫成本（n=2.5 万 → 20 万） | 0.42 → 2.39 ms，每条 6.5 → 37.4 µs（$O(n)$ memmove，等效 26–45 GB/s） |
| compaction：17 段 → 2 段，盘上 3.95 MB → 1.30 MB，活载荷 1.04 MB | 空间放大 3.81× → 1.26×，写放大 5.07×，97 ms（合并 68.9 + 编码 12.1 + fsync 8.0）；丢 6,000 条墓碑（27.3%）与 2,000 条过期记录 |

外部口径复算：$W=1+\log_{10}86{,}400=5.94\approx5.9\times$，$51.2\times5.9=302.1$ GB，$302.1/2=151$ s，$151/10{,}332=1.46\%$；H100 拐点 $989/3.35=295.2\approx295$ FLOPs/byte；LLaMA-3-70B 每 token 320 KiB、LLaMA-3-8B 128 KiB 与仓库统一常数一致。

## 常见追问

- **追问**：未命中为什么返回哨兵而不是抛异常？
  - 要点：miss 是正常路径，异常在高频路径上又贵又要包 try；哨兵还能和「值就是空字节串」区分开。若语言没有便宜的哨兵，就返回 `(ok, value)` 二元组，语义一样。
- **追问**：SET 为什么必须拷贝值？
  - 要点：调用方复用 `bytearray` 缓冲时会污染库内数据，且这种 bug 只在写后改 buffer 时出现，很难复现。代价是大值多一次拷贝，可用不可变类型或所有权转移规避。
- **追问**：TTL 只做惰性删除够不够？
  - 要点：不够。惰性只保证读语义，内存回收要靠有界清扫；两者叠加，且过期项不占容量、淘汰顺序先清过期再按 LRU。
- **追问**：scan 的游标在并发写之后怎么算？
  - 要点：显式二选一。快照语义（epoch）不重复、允许遗漏；要保证不漏就得固定一份键快照或按 key 版本重放，代价是内存或延迟。游标是「最后返回的键加 epoch」，不是偏移量——偏移量在删除之后会跳行。
- **追问**：为什么 compaction 不能放在请求路径上？
  - 要点：它是 IO 密集、延迟不可控且有写放大；本机 Python 逐条 3.13 µs/条，51.2 GB 规模要上万秒 CPU。必须后台限速分批，否则 p99 长尾就是它自己。

## 相关题目

- [[coding-06]]：O(1) 的 LRU 与 TTL——本题第 1、3 层的容量记账、惰性删除、有界清扫在那里有更细的实测（含 80.3% 尸体与 125,644 次误淘汰）。
- [[coding-07]]：需求持续加码时如何不失控；时钟差与单调性防护的口径直接来自它。
- [[coding-11]]：全扫成本与「不会上线」的工程理由；本题第 2 层的扫描口径与它对齐。
- [[rag-11]]：段式索引的 refresh 与写放大公式；第 4 层的 $W\approx1+\log_{10}S$、302 GB、151 s、1.5% 取自它。
- [[inference-serving-03]]：PagedAttention 为什么否决对数十 GB KV cache 做在线 compaction。

## 参考资料与归属

- **Key eviction（Redis 文档）（延伸）** —— Redis：<https://redis.io/docs/latest/develop/reference/eviction/>。内存键值存储的过期与淘汰语义（TTL 的工业参考实现）。
- **Log-structured merge-tree（维基百科）（延伸）** —— Wikipedia：<https://en.wikipedia.org/wiki/Log-structured_merge-tree>。写入合并与 compaction 的经典结构（LSM 树）。
- **decimal — 十进制定点与浮点运算（Python 标准库文档）（延伸）** —— Python Software Foundation：<https://docs.python.org/3/library/decimal.html>。时间戳与金额类数据的精度处理。

- **延伸来源说明**：过滤扫描的索引设计、TTL 的惰性/主动过期取舍、以及 compaction 的写放大算例，是按本仓库统一口径自行推导与实测的工程算例，不是上述来源的原文数字；来源仅用于机制、算法与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
