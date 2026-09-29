---
type: question
id: consumer-ml-01
company: 面向消费者的规模化 ML 公司（Uber、Netflix、LinkedIn、Airbnb、Pinterest、Spotify）
topic: coding
order: 1
question: 用有界内存的 sketch 实现流式 top-k；实现一个滑动窗口速率计数器。
question_en: Implement a streaming top-k with a bounded-memory sketch; implement a sliding-window rate counter.
asked_at: []
level: 高阶
tags: [流式算法, sketch, top-k, 滑动窗口, 速率限制, 实现题]
sources:
  - title: Counting a lot of different things（Cloudflare 工程博客）（延伸）
    url: https://blog.cloudflare.com/counting-things-a-lot-of-different-things/
    author: Cloudflare
    published: 2017-06-19
  - title: A Single Rate Three Color Marker（RFC 2697）（延伸）
    url: https://www.rfc-editor.org/rfc/rfc2697
    author: Heinanen & Guerin (IETF)
    published: 1999-09
  - title: Key eviction（Redis 文档）（延伸）
    url: https://redis.io/docs/latest/develop/reference/eviction/
    author: Redis
    published: 
related: [coding-07, coding-06, coding-05, evaluation-07, system-design-09]
updated: 2026-09-29
---

## 一句话答案

> 两半各定容器，但先钉内存预算：**精确 top-k** 是 hash map 加大小 $k$ 的最小堆，哈希查找摊还 $O(1)$、堆调整 $O(\log k)$，每个不同 key 约 80 B——1 亿个不同 key 就是 8.04 GB，基数无界时必须换 sketch：Space-Saving 用 $k$ 个槽给出 $f \le \hat f \le f + N/k$，把相对误差压到 $\varepsilon$ 用 Misra-Gries 的 $1/\varepsilon$ 个槽（只低估），Count-Min 用 $w=\lceil e/\varepsilon \rceil$、$d=\lceil \ln(1/\delta)\rceil$（$\varepsilon=0.001,\ \delta=0.01$ 时是 $2719 \times 5$ 个 int32，53.1 KiB）但只高估、不可枚举，只能当**候选生成器**，再对候选做精确复核。
> **速率计数器**分三种：固定窗口 1 个计数但边界放 $2\times$（限额 50/min 实测放行 100）；滑动窗口日志精确，内存却与请求数成正比；滑动窗口计数只留 2 个计数、按 $\text{rate}=\text{prev}\cdot\frac{T-\text{elapsed}}{T}+\text{cur}$ 外推，是消费级的默认（同一算例实测放行 50）。要显式允许突发就上 token bucket/GCRA（状态是两个数/一个时间戳）。
> 分布式把三条硬约束写在最前面：判定原子、窗口按 $\lfloor t/T \rfloor$ 对齐、时间取服务端。对外接口一律返回 `(item, count, err)` 三元组并把误差上界画进看板——近似值只能进候选与告警，不能直接接计费、分成或封禁。

## 面试官在考什么

- **预算是否先于容器**：基数是多少（Uber 的行程 id、Netflix 的播放埋点、Pinterest 的曝光流都是无界的）、每条多少字节、超预算时的降级动作是什么。答不出数字就等于没答。
- **误差界是否写成 API 契约**：sketch 的误差是绝对量并随总事件数 $N$ 线性增长，返回 `(item, count, err)` 而不是 `(item, count)`，才让下游知道 CTR、分成、告警阈值里有多少是噪声。
- **是否知道 sketch 的能力边界**：不可枚举、不可回放、单向偏（Space-Saving/Misra-Gries 只低估或只高估），所以 CM 必须两级组合、跨分片必须两阶段 merge。
- **三种窗口实现各自的失效形状**，以及能不能给出一条明确选择和代价，而不是三种都念一遍。
- **分布式三约束是否落在机制上**：一次原子读-改-写、统一窗口对齐、服务端时间；以及热点键上的锁粒度（有界内存不等于有界 CPU）。

常见错误答案：

- 「用最小堆维护 top-k，$O(\log k)$」——只答了数据结构，没答内存随基数增长；或者「CM 直接取估计值最大的 $k$ 个 key」，把只高估的计数器当精确频次用。
- 「Redis 的 `INCR` + `EXPIRE` 就是滑动窗口」，或者先 `GET` 再 `SET`、用实例本地时间起窗口。

## 原理与推导

### 先钉内存预算，再选容器

精确 top-k 的最小实现是一张计数表加一个大小 $k$ 的最小堆：计数表给每个不同 key 精确频次，堆让「当前第 $k$ 名」是 $O(1)$ 可读的；每事件是一次哈希查找加一次 $O(\log k)$ 的堆调整，每 key 约 80 B。本仓库同专题 coding-06 用 tracemalloc 量的口径是裸 `dict` 80.4 B/条、`OrderedDict` 133.3 B/条、手写双向链表 160.4 B/条、带 TTL 284.3 B/条（N=10 万）。按 80.4 B/条外推：

$$\text{1e8 keys} \times 80.4\ \text{B} = 8.04 \times 10^9\ \text{B} = 8.04\ \text{GB} = 7.49\ \text{GiB}$$

一天 10 亿次曝光/播放事件（均值 11,574 事件/s）就能撞上这个量级。所以「有界内存」要落到预算上：能给多少字节、超了是采样、按 key 前缀折叠、还是只对最近窗口保候选。

sketch 把内存换成「与精度目标成正比」：Space-Saving 是 $k$ 个槽（$k=1000$ 时按每槽 item 8 B + count 4 B + err 4 B 的紧凑布局是 15.6 KiB），Count-Min 是 $w \times d$ 个计数器。顺带一个工程事实：53.1 KiB 只有 LLaMA-3-70B 一个 token 的 KV cache（320 KiB，GQA-8、bf16、80 层）的 1/6.03，也是 LLaMA-3-8B 的 128 KiB/token 的 1/2.41——**有界内存同时是有界工作集**，sketch 常驻 L2、随机访存便宜，而 8 GB 的计数表每次更新都是一次跨 DRAM/TLB 的随机访问。

反过来，候选集合已知且有限时（城市、套餐档位、错误码枚举、词表）不要用 sketch：一趟选择即可，`np.argpartition` 期望 $O(V)$、`heapq.nlargest` 是 $O(V\log k)$（coding-05 的口径）。只有基数无界且内存有界时，sketch 才是唯一解。

### Space-Saving 与 Misra-Gries 的机制和误差界

Space-Saving（Stream-Summary）维护 $k$ 个 `(item, count, err)` 槽，用计数桶链表做到找最小项 $O(1)$ 摊还（或最小堆 $O(\log k)$）。表满时顶替最小项，并把被顶替项的计数当作新项的 `err` 起点、新计数继承后加 1。两条保证：

1. 槽内计数之和恒等于 $N$，所以任意时刻的最小槽计数都 $\le N/k$；被顶替项的计数不小于它的真实频次，因此新项继承的这个计数就是过估计的上限，对表内任何项有 $f \le \hat f \le f + N/k$。
2. 若某项真实频次 $f > N/k$，它在任何时刻都不可能是最小项，因此一定留在表内。

Misra-Gries 是 $k-1$ 个计数器、未命中且表满时全体减 1。每次「全体减 1」至少消耗 $k$ 个事件，所以总减量 $D \le N/k$；计数器只在命中时加 1，于是 $f - N/k \le \hat f \le f$——只低估、不虚报，适合做「至少这么多」的判定与告警，不适合直接排名。

误差是**绝对量**：$N=10^9$、$k=1000$ 时 $N/k = 10^6$，意味着频次差在 $10^6$ 以内的两项不可分辨。本机实测（Zipf(1.1)、$N=10^6$、$k=1000$、$N/k=1000$）把这个形状坐实了：83 个 $f > N/k$ 的重头项 SS 与 MG 各漏 0 个，SS 过估计最大 1、MG 低估 476–477（都在 1000 的界内）；但换上真值全部约 500 的均匀流（$V=2000$）后，表内 key 的真值是 440–585，SS 的估计值却挤在 999–1004（就是 $N/k$ 本身），与真实 top-20 的交集是 **0/20**。所以看板要画误差上界，名次只在远超 $N/k$ 的头部才有意义。

### Count-Min：只高估、参数、两级组合

Count-Min 是 $d \times w$ 个计数器，$d$ 个两两独立（实践上独立）的哈希，更新是 $d$ 次加一，查询取 $d$ 行最小值：

$$w = \left\lceil \frac{e}{\varepsilon} \right\rceil, \qquad d = \left\lceil \ln\frac{1}{\delta} \right\rceil, \qquad \hat f \ge f,\quad \Pr\big[\hat f \le f + \varepsilon N\big] \ge 1-\delta$$

取最小值保证只高估；$\varepsilon=0.001$、$\delta=0.01$ 时 $w=2719$、$d=5$、$w\cdot d = 13{,}595$ 个 int32 = 54,380 B = 53.1 KiB。注意 $\delta$ 是**单次查询**的失败概率：这次实验有 87,828 个不同 key、每个都要排一次序，用联合界看预期越界次数就是 $87828 \times 0.01 \approx 878$——所以 CM 的精度必须用实测的最大过估计来校准，而不是拿 $\varepsilon N$ 当保证。本机实测（$N=10^6$、$V=200{,}000$、87,828 个不同 key、$\varepsilon N=1000$）：过估计均值 68.7、最大 664，全部 87,828 个 key 都至少高了 1，只出现 1 次的 key 估计值最高 665——**不是「偶尔撞一次」，而是「几乎必然高估一点，但被 min-of-$d$ 压在界内」**。

CM 的两个工程性质来自这个式子的线性：更新是逐行相加，所以两个 sketch 可以逐格相加后合并，这正是 map 端预聚合（combiner）能把 shuffle 压下一到两个数量级的原因；保守更新（conservative update）能显著压低过估计，但更新不再线性、合并语义要另外写清。另一条硬要求是**行间哈希必须真正独立**：实测用 $(a_r\cdot i + b_r) \bmod w$ 这种线性模哈希、而 key 的取值空间与 $w$ 有算术结构（相差 $w$ 整数倍的 key）时，两个 key 会在 5 行里全部相撞，min-of-5 退化成一行的效果——最大过估计 131,349（上界是 1000），候选 top-20 精确度 5%；换成混合哈希后最大过估计 664、top-20 精确度 95%。

CM 不能枚举、也不能自己给出 top-k，所以是两级：sketch 当候选生成器，再对候选做精确复核（同窗口精确计数，或把明细落盘/回放）。实测候选质量的形状是「头部准、长尾崩」：CM 估计值 top-20 与真值 top-20 交集 95%、top-100 是 98%，但 top-1000 掉到 69.4%，且 1000 个候选里有 917 个真实频次不超过 $N/k$——**top-k 里冒出只来过几次的 key 是 sketch 的固有形状，不是 bug**。复核通道只要 $1000/87828 = 1.14\%$ 的不同 key，这就是「近似 + 精确复核」比「全量精确」便宜的地方。

### 滑动窗口速率计数器：三种实现与各自的失效形状

| 实现 | 状态 | 窗口边界 | 内存 | 失效形状 |
| --- | --- | --- | --- | --- |
| 固定窗口计数 | 1 个计数 | 允许 $2\times$ 突发 | 常数，最便宜 | 限额 50/min 时边界实测放行 100 |
| 滑动窗口日志 | 窗口内每次请求的时间戳 | 精确 | $\propto$ 请求数 | 单键 10 万 QPS、60 s 窗口就是 600 万个时间戳 |
| 滑动窗口计数 | 2 个计数 | 近似、被铺开 | 常数 | 估计值误差；实例时间轴不对齐就整体漂移 |
| token bucket / GCRA | 2 个数 / 1 个时间戳 | 允许 $b$ 的受控突发 | 常数 | 参数选错要么常态拒绝要么形同虚设 |

滑动窗口计数的公式是 $\text{rate} = \text{prev}\cdot\frac{T-\text{elapsed}}{T} + \text{cur}$，只在窗口内请求路径上做一次加法和比较。Cloudflare 的算例：限额 50/min、上一窗口 42 次、当前已过 15 s 且已有 18 次，估计速率 $42\times\frac{45}{60}+18 = 49.5$，再来一个（$50.5 > 50$）就拒；同一篇文章的线上口径是 4 亿次请求、27 万个不同来源下 0.003% 的判定错误、估计速率与真实速率平均差 6%，并且他们把计数改成异步、只把「是否开始限流」一个比特放进请求路径。

**那个 $2\times$ 并没有消失，只是被铺开了**（下面是自己实测的补充口径）：让窗口 1 末尾来 50 个请求、窗口 2 每秒来 1 个，固定窗口在 $[T-1, T+59)$ 这个 60 s 区间里放行 100 个；滑动窗口计数放行 99 个（窗口 2 每秒 1 个共放行 49 个，因为额度按 $\text{prev}\cdot t/T$ 线性放开）；滑动窗口日志只有 50 个（它精确记住那 50 个时间戳还没滑出去）。也就是说，加权外推把「边界后瞬间的 100 个」摊成「一整个窗口匀速补回来的 99 个」——保护下游时这是本质差别，但对外承诺「任意 60 s 不超过 50」依然不成立，这类契约要么改成「每分钟窗口配额」，要么换 token bucket 用 $b$ 显式表达突发；token bucket 与 GCRA 是同一条判定式的两种坐标表示（coding-07 实测：$r=20,b=5$、$r=1.5,b=8$、$r=500,b=1$ 三组参数各 20000 次随机到达，判定不一致 0 次），选哪个只看要不要多一个可读的「剩余额度」。额外一条专坑：所有实例必须按 $\lfloor t/T \rfloor$ 对齐到同一时间轴，不能各按「本实例的第一个请求」起窗，否则等效限额随实例数漂移（coding-07 的局部额度路线实测：配置 $N=8$ 在 12 实例时上限 +50%、只剩 3 实例时 −62%）。

### 分布式三条硬约束与对抗面

- **判定原子**。先 `GET` 再算再 `SET` 在并发下必然超发：实测 16 线程 × 20 次请求、全局额度 100，一次 `EVAL` 的原子脚本放行 100 次（超出 0），`GET`-then-`SET` 放行 320 次（超出 220）。
- **窗口对齐**。滑动窗口计数的键必须按 $\lfloor t/T \rfloor$ 切，键的 TTL 不小于窗口长度、且不能每次请求都刷新——coding-07 实测：把 TTL 设成 100 ms 会让键过期、桶被当满桶重建而误放行；每次请求都刷新 TTL 会把计数器刷活，限额 3/100 ms 的对照里正确写法通过 9 次、刷活写法只通过 3 次且不再恢复。
- **时钟同源**。用实例本地时间做窗口或懒补充，等于让每个实例各算一套：两实例时钟差 1 ms、$r=100/\text{s}$、$b=10$、1 s 内 6000 次请求，本地时间口径放行 359 次，而服务端时间口径是 109 次（理论上限 $b+rT = 110$）。RFC 2697 的单速率三色标记（CIR/CBS/EBS、两个共享速率的桶、green/yellow/red）是「桶 + 突发」这套模型在标准文本里的先例，注意它的语境是 Diffserv 网络整形、状态为 Informational，且原文明确说实现不必按形式化规格建模——这正是懒补充/事件驱动结算被允许的依据。
- **跨分片两阶段**。每个 shard 出本地 top-k 再 merge，$k$ 小所以合并很便宜（100 分片 × $k=1000$ 就是 $10^5$ 个候选）；但近似值不能跨实例平均——分位数与比率必须由可合并的 sketch 在聚合层算，复用 evaluation-07 的实测口径：两个 pod 各 5000 个延迟样本，p99 分别为 2.25 s 与 6.01 s，简单平均 4.13 s，合并后真实的全局 p99 是 5.49 s、低估 24.8%。窗口要按事件时间加 watermark 切分，否则迟到数据会让同一窗口重算两次。
- **对抗面**。所有 sketch 的保证都建立在「频次分布非对抗」上：刷量可以用海量一次性 key 把 Space-Saving 的 $k$ 个槽全部换成伪造项、每项还被高估到约 $N/k$，从而污染榜单与反作弊输入。上榜链路前面必须有 key 级去重与限速、误差上界监控和精确复核通道，sketch 输出只能作为候选与告警输入。
- **有界内存不等于有界 CPU**。计数器落在热点键上要做读-改-写，锁粒度往往比数据结构更早决定上限：同构结构分片实测（coding-06，8 线程 × 2 万次迭代、容量 1000）单线程基线 987,425 ops/s、1 把锁 71,402 ops/s、16 片锁 299,187 ops/s（仍是基线的 30%）；分片哈希必须用稳定哈希（crc32/murmur），内置 `hash()` 对 str 按进程随机加盐，会让同一个键漂到不同分片；sketch 侧同理，每事件 $d$ 次随机访存落在同一行上会互相争用，所以消费级埋点流的常见形态是客户端/SDK 先本地预聚合、再周期性合并上传——省 CPU 也省网络，代价是判定延迟与重复计，合并前要靠幂等键或按窗口去重。这条链在 CPU 摄入层，与 GPU 侧的 roofline 无关（H100 SXM5 bf16 dense 989 TFLOPs ÷ 3.35 TB/s ≈ 295 FLOPs/byte 是模型侧的拐点），把 top-k 挪到 GPU 上反而要先付每事件一次 H2D 拷贝。

## 数值与代码验证

环境：Python 3.10.12（CPython）、numpy 2.2.6；机器 AMD Ryzen 9 8945HX / Linux x86_64，与其他任务共享负载，故墙钟数字的可外推部分是量级与倍数。内存用 `tracemalloc` 的 current 口径（键对象在 `start` 之前创建，只计表结构与值）；哈希实测用 splitmix64 混合哈希，坏哈希对照用线性模哈希。脚本：`.work/consumer01_run.py`（参数复算、内存、误差、速率实验、微基准）、`.work/consumer01_doc.py`（正文代码的完整驱动，17 条断言）、输出在 `.work/consumer01_run.out`。

| 复算项 | 口径 | 结果 |
| --- | --- | --- |
| Count-Min 参数 | $\varepsilon=0.001$、$\delta=0.01$，$e/\varepsilon = 2718.28$、$\ln(1/\delta)=4.605$ | $w=2719$、$d=5$、13,595 个 int32 = 54,380 B = **53.1 KiB** |
| 与模型侧对照 | LLaMA-3-70B 320 KiB/token（GQA-8、bf16、80 层）、LLaMA-3-8B 128 KiB/token | 53.1 KiB 是 1/6.03 与 1/2.41 |
| 绝对误差 | $N=10^9$、$k=1000$；$\varepsilon N$ | $N/k = 10^6$；$\varepsilon N = 10^6$（两者同级，别混用） |
| 精确计数表 | 80.4 B/条 × $10^8$ 个 key | 8.04 GB = 7.49 GiB（`OrderedDict` 133.3 B/条则 12.41 GiB） |
| 吞吐 | 一天 10 亿事件 | 11,574 事件/s 均值 |

| 内存实测（N=10 万条） | tracemalloc | 1e8 条外推 |
| --- | --- | --- |
| `dict` str→int 计数表 | 8,042,904 B（80.4 B/条） | 8.04 GB |
| `dict` + 大小 1000 的最小堆 | 5,307,480 B（53.1 B/条，含堆槽） | — |
| CM `array('i')` 13,595 格 | 57,868 B（含对象头） | 53.1 KiB |
| CM `list[int]` 同尺寸（计数 >256） | 489,420 B（8.5×） | 列表每格多一个 int 对象，sketch 必须用 `array`/numpy |

| 流式 top-k 实测（Zipf(1.1)、$N=10^6$、$V=2\times10^5$、87,828 个不同 key、$k=1000$、$N/k=1000$） | 结果 |
| --- | --- |
| $f > N/k$ 的 83 个重头项漏检 | Space-Saving 0/83、Misra-Gries 0/83 |
| Space-Saving 过估计 | 最大 1（界 1000），top-20 与真值交集 95% |
| Misra-Gries 低估 | 476–477（界 1000） |
| Count-Min 过估计（混合哈希） | 均值 68.7、最大 664（界 1000）；估计值 top-20 精确度 95%、top-100 98%、top-1000 69.4% |
| Count-Min（线性模哈希的坏对照） | 最大过估计 131,349 = 上界的 131 倍，top-20 精确度 5% |
| 均匀流（$V=2000$，真值 440–585，全在 $N/k$ 之下） | SS 估计值 999–1004；top-20 与真值交集 **0/20** |

| 速率计数器实测（限额 50 / 60 s） | 边界 100 个请求放行 | 窗口 1 末尾 50 个 + 窗口 2 每秒 1 个，区间 $[T-1,T+59)$ 合计 |
| --- | --- | --- |
| 固定窗口 | **100** | 100 |
| 滑动窗口日志 | **50** | 50（窗口 2 放行 0） |
| 滑动窗口计数 | **50** | 99（窗口 2 放行 49，额度按线性放开：$t=0$ 拒绝、$t=6$ s 估计 45 放行、$t=30$ s 估计 27、$t=59$ s 估计 3.8） |
| Cloudflare 算例 | $42\times45/60+18 = 49.5$，再加一个拒 | — |
| 估计误差（随机到达 $\lambda=35$/min、6 个窗口） | 与真实 60 s 计数平均绝对差 1.60、平均相对差 4.7%（样本 25） | 与源文 6% 同量级 |

正文代码（核心实现摘自 `.work/consumer01_doc.py`；该脚本连同参数复算、三种窗口对照共 17 条检查全部通过，这里只保留核心实现与 4 条关键断言）：

```python
import heapq, math, zlib
from array import array


class SpaceSaving:
    """k 个槽的 Stream-Summary；count 是真实频次的上界，err 是继承来的误差起点。"""

    def __init__(self, k):
        self.k, self.slots, self.heap = k, {}, []          # heap: 惰性 (count, item)

    def add(self, x, c=1):
        s = self.slots.get(x)
        if s is not None:
            s[0] += c
            heapq.heappush(self.heap, (s[0], x))
            return
        if len(self.slots) < self.k:                       # 表未满：直接占槽
            self.slots[x] = [c, 0]
            heapq.heappush(self.heap, (c, x))
            return
        while self.heap[0][0] != self.slots[self.heap[0][1]][0]:
            heapq.heappop(self.heap)                       # 清掉已过期的堆项
        cnt, victim = self.heap[0]                         # 当前最小项
        del self.slots[victim]
        self.slots[x] = [cnt + c, cnt]                     # 继承最小计数，再 +c
        heapq.heapreplace(self.heap, (cnt + c, x))


class CountMin:
    """w x d 计数器；查询取 d 行最小值（只高估），不能枚举。"""

    def __init__(self, eps, delta):
        self.w = math.ceil(math.e / eps)
        self.d = math.ceil(math.log(1.0 / delta))
        self.rows = [array('i', bytes(4 * self.w)) for _ in range(self.d)]

    def _h(self, r, x):
        return zlib.crc32(('%d:%s' % (r, x)).encode()) % self.w   # 各行独立种子

    def add(self, x, c=1):
        for r in range(self.d):
            self.rows[r][self._h(r, x)] += c

    def est(self, x):
        return min(row[self._h(r, x)] for r, row in enumerate(self.rows))


class SlidingWindowCounter:
    """两个计数 + 上一窗口的加权外推；窗口按 floor(t / T) 对齐。"""

    def __init__(self, limit, window):
        self.limit, self.T, self.idx, self.prev, self.cur = float(limit), float(window), -1, 0, 0

    def _roll(self, t):
        i = math.floor(t / self.T)                         # 对齐到全局时间轴
        if i != self.idx:
            self.prev = self.cur if i == self.idx + 1 else 0   # 中间空窗必须清零
            self.cur, self.idx = 0, i
        return t - i * self.T

    def estimate(self, t):
        elapsed = self._roll(t)
        return self.prev * (self.T - elapsed) / self.T + self.cur

    def allow(self, t):
        if self.estimate(t) + 1 <= self.limit:
            self.cur += 1
            return True
        return False


c = SlidingWindowCounter(50, 60)                           # 42 / 15 s / 18 -> 49.5
c.idx, c.prev, c.cur = 0, 42, 18
assert c.estimate(15.0) == 49.5 and c.allow(15.0) is False
ss = SpaceSaving(3)                                        # k=3、10 个事件、N/k=3.33
for x in 'aaaabcdef':
    ss.add(x)
assert len(ss.slots) <= 3 and 4 <= ss.slots['a'][0] <= 4 + 10 // 3   # 只高估、不超界
cm = CountMin(0.001, 0.01)
assert (cm.w, cm.d) == (2719, 5) and len(cm.rows) * cm.w * 4 == 54380
for x in ['hot'] * 7 + ['cold']:
    cm.add(x)
assert cm.est('hot') >= 7 and cm.est('cold') >= 1          # 只高估，不是精确值
print('snippet assertions passed: 4')
```

微基准（$2\times10^5$ 次 update、5 次取中位数，单线程；两次运行的区间）：精确 `dict` 97–116 ns/事件，CM（$d=5$、预算好的 int 下标）0.95–1.26 µs，CM（str key + 每行 f-string + crc32）2.31–2.84 µs，Space-Saving $k=1000$（int key）0.94–1.00 µs。结论：**sketch 的每事件成本是 $d$ 次随机访存加哈希，比一次 dict 更新贵 8–13 倍（预算好下标）到 20–29 倍（含字符串哈希）**，热点键上要按分片摊开，且不要把哈希成本花在字符串拼接上（生产里用预算好的 key id + murmur/xxhash）。

## 常见追问

- **追问**：为什么不能把 Count-Min 的估计值直接排序取 top-k？
  - 要点：CM 只高估、不能枚举，取估计值最大的 $k$ 个会把长尾里被碰撞抬起来的 key 排进来（实测 top-1000 候选里 917 个真实频次不超过 $N/k$，top-1000 精确度 69.4%）。正确形态是 sketch 当候选生成器 + 精确复核（复核 1000 个候选只占 1.14% 的不同 key），或者直接用 Space-Saving 这类自带误差界的摘要。
- **追问**：$N/k$、$\varepsilon N$ 这些绝对误差怎么对外暴露？
  - 要点：接口返回 `(item, count, err)`，把 `err` 上界（Space-Saving 是入库时继承的最小计数、CM 是 $\varepsilon N$）画进看板；下游用频次做阈值判断时要求 $f > N/k$ 或 $\hat f - \text{err} > \text{阈值}$；分位数/比率不进这条链路，用可合并直方图在聚合层算（evaluation-07：p99 简单平均低估 24.8%）。
- **追问**：$k$ 和 $\varepsilon$ 怎么定？
  - 要点：从预算反推，不要从精度空想。要保证「频次超过 $N/k$ 的项不漏」就把 $k$ 取成最大可容忍的槽数再算 $N/k$ 是否小于业务分辨力；要「相对误差 $\varepsilon$」就用 Misra-Gries 的 $1/\varepsilon$ 个槽（$\varepsilon=0.001$ 是 1000 个槽）。CM 的 $w$ 只和 $\varepsilon$ 有关（$w = \lceil e/\varepsilon\rceil$）、$d$ 只和 $\delta$ 有关，加行比加宽便宜但每事件多 $1$ 次访存。
- **追问**：为什么不用滑动窗口日志，明明它精确？
  - 要点：内存与窗口内请求数成正比：单键 10 万 QPS、60 s 窗口就是 600 万个时间戳，而按来源/租户/路由乘出来的键基数会让它爆炸；滑动窗口计数只用 2 个计数。精确性只在「限额很低且必须逐次可解释」时才值得付这个内存。
- **追问**：限流用固定窗口 + 2× 突发为什么不行？
  - 要点：因为突发是瞬间的双倍：限额 50/min 时窗口末 50 个加下一窗口初 50 个实测放行 100，下游拿到的是 2 倍峰值。固定窗口适合「配额型、边界不敏感」的场景（按天/月的调用量），保护下游要用 token bucket/GCRA 把 $b$ 显式写进契约。
- **追问**：多分片/多实例下 top-k 与被限流主体怎么汇总？
  - 要点：top-k 两阶段（每片出本地 top-k 再 merge，100 片 × $k=1000$ 只有 $10^5$ 个候选，合并是 $O(Sk\log k)$）；限流判定必须一次原子读-改-写（实测 `GET`-then-`SET` 超发 220）、窗口按 $\lfloor t/T \rfloor$ 对齐、时间取服务端；被限流主体的统计要按「判定来源等级」下钻，而不是把实例级比例平均。

## 相关题目

- [[coding-07]]：token bucket 与分布式限流线的完整口径，本文件的原子性、时钟、TTL 实测都来自那里。
- [[coding-06]]：同一套字节记账与分片锁粒度实测（80.4 B/条、987,425 → 71,402 → 299,187 ops/s）。
- [[coding-05]]：候选集已知且有限时的选择（`argpartition` 期望 $O(V)$、`nlargest` 是 $O(V\log k)$）。
- [[evaluation-07]]：误差上界、被限流比例与分位数不可跨实例平均，是本题对外暴露精度的落点。
- [[system-design-09]]：gateway 的缓存、限流与预算，把单键计数器放回多级限流的系统里。
- [[coding-08]]：被限流后的重试与并发上限，是速率计数器在客户端一侧的对偶问题。

## 参考资料与归属

- **Counting a lot of different things（Cloudflare 工程博客）（延伸）** —— Cloudflare，2017-06-19：<https://blog.cloudflare.com/counting-things-a-lot-of-different-things/>。近似计数与布隆/sketch 类结构在流式统计中的取舍。
- **A Single Rate Three Color Marker（RFC 2697）（延伸）** —— Heinanen & Guerin (IETF)，1999-09：<https://www.rfc-editor.org/rfc/rfc2697>。令牌桶与突发额度的标准化模型。
- **Key eviction（Redis 文档）（延伸）** —— Redis：<https://redis.io/docs/latest/develop/reference/eviction/>。按时间与容量两套机制淘汰的工业实现参考。

- **延伸来源说明**：第 3–5 节的误差界、内存账与计数器实测算例，是按本仓库统一口径（H100 bf16 dense 989 TFLOPs、每 token KV 按模型规模取 128 KiB / 320 KiB、\$2/GPU 小时等）自行推算的工程算例，不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
