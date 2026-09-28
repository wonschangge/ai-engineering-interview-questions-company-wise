---
type: question
id: inference-serving-03
topic: 推理、服务与 GPU 性能
order: 3
question: PagedAttention 是如何工作的？它解决了 KV cache 碎片化的什么问题？
question_en: How does PagedAttention work and what KV cache fragmentation problem does it solve?
asked_at: [NVIDIA, Together AI]
level: 进阶
tags: [pagedattention, vllm, 显存管理, 碎片]
sources:
  - title: Paged Attention in LLMs
    url: https://outcomeschool.com/blog/paged-attention-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-03-29
  - title: How does vLLM work?
    url: https://outcomeschool.com/blog/how-does-vllm-work
    author: Amit Shekhar (Outcome School)
    published: 2026-06-17
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
related: [inference-serving-02, inference-serving-08, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> PagedAttention 把每个请求的 KV cache 切成固定大小的 block（vLLM 默认 16 个 token），block 之间不必连续，再用一张 block table 记录「逻辑块 → 物理块」的映射，attention kernel 按表取值做分块 softmax。
> 它消除的是外部碎片（所有 block 等大，任何空闲块都能被任何请求使用），并把内部碎片压到只剩每个序列的最后一个 block：长度 512 的序列平均浪费 1.4%，最坏 2.85%。
> 论文口径下，已有系统真正存 token 的显存只占 20.4%–38.2%，而 block 级按需分配把浪费压到接近零，于是同样显存能并发的序列数变多、batch 变大，同延迟下吞吐提升 2–4×。

## 面试官在考什么

- **能不能把「浪费」量化**：不是泛泛说「碎片很严重」，而是能说出朴素实现按 `max_seq_len` 预分配连续显存，产生预留槽位、内部碎片、外部碎片三类浪费，并给出论文的 20.4%–38.2% 这个口径（这是「KV cache 有效利用率」而非「碎片率」，必须说清）。
- **能不能讲清机制而不是背名词**：逻辑块、物理块、block table、分块 softmax、waste ≤ block size − 1 的推导。只会说「像操作系统的虚拟内存」是不够的，要说清「像在哪一步」。
- **能不能把显存收益换算成业务指标**：显存 → 并发序列数 → batch size → 吞吐/成本。这是 serving 岗位真正关心的一条因果链，也是 [[inference-serving-02]] 的前置条件。
- **知不知道代价**：block 引入间接寻址与 kernel 复杂度，论文实测 attention kernel 延迟比 FasterTransformer 高 20%–26%；block size 是碎片与并行度的权衡；这个优化只在「显存容量受限」的负载上成立，不是普适加速。
- **能不能延伸到共享与调度**：copy-on-write 共享（parallel sampling、beam search、共享 prefix）、all-or-nothing 抢占、recompute 与 swap 的取舍。这几条是 PagedAttention 从「省显存」变成「省显存 + 能调度」的关键。

常见错误答案：

- 「PagedAttention 让 KV cache 变小了。」——它不改变 KV cache 本身的字节数，只改变存放方式；压缩 KV cache（量化、MLA、驱逐）是另一类技术，见 [[llm-internals-02]]、[[llm-internals-04]]。
- 「碎片化没了，所以显存利用率 100%。」——还剩最后一个 block 的尾部浪费、block table 元数据、以及引擎为调度留的水位线。
- 「block 不连续，所以 attention 算错了/需要重排。」——分块 softmax 在数学上与原式等价，只是一次把 softmax 的归约分母拆成多块累加。

## 原理与推导

### 第一步：先算清 KV cache 有多贵

对 $L$ 层、$H_{kv}$ 个 KV 头、每头维度 $d_{head}$、精度 $b$ 字节的模型，单 token 的 KV cache 是

$$\text{bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot b$$

以 LLaMA-3-70B（$L=80$、$H_{kv}=8$、$d_{head}=128$、bf16 即 $b=2$）为例：$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ 字节 $= 320$ KiB/token。于是 2k 上下文要 640 MiB，8k 上下文要 2.5 GiB，32k 上下文要 10 GiB——**单个请求**。论文用的是 OPT-13B（$L=40$、hidden 5120、MHA）：$2 \times 5120 \times 40 \times 2 = 800$ KiB/token，2048 token 上限对应 1.6 GB/请求。显存里权重是常数，激活只占小头，所以「能同时服务多少个请求」几乎完全由 KV cache 的管理方式决定。

### 第二步：朴素实现的三类浪费

多数深度学习框架要求张量连续存放，早期 serving 系统（FasterTransformer、Orca）因此给每个请求预分配一段**连续**显存，长度按请求的最大可能序列长度（例如 2048）算，因为输出长度事先未知。论文把浪费拆成三类：

1. **预留（reserved）**：为未来 token 预留、最终会被用到的槽位。它不算永久浪费，但在整个请求生命周期内占着位置，把别的请求挡在外面。
2. **内部碎片（internal fragmentation）**：实际生成长度远小于最大长度，超出部分是纯浪费。2048 token 的预留、50 token 的输出 → 97.6% 的槽位是纯浪费（$1 - 50/2048$）。
3. **外部碎片（external fragmentation）**：预分配尺寸各不相同（2048、512、128……），buddy allocator 留下的空洞既装不下一个新请求，又无法被利用。

论文的 profiling 结论是：这些系统里真正存 token 状态的 KV cache 显存只占 **20.4%–38.2%**（SOSP 2023 版 §1、§3.1，图 2；这是论文在 ShareGPT/Alpaca 负载上的实测口径，不是理论最坏值）。

论文同时否决了「用 compaction 收拾碎片」这条路：KV cache 动辄数十 GB，在性能敏感的在线服务里搬移它不现实，而且预分配的连续 chunk 本身就堵死了共享的可能。

### 第三步：block、block table 与分块 softmax

PagedAttention 把每个序列的 KV cache 切成固定大小 $B$ 个 token 一块的逻辑块（logical KV block），GPU 上预先切好等大的物理块池（physical KV block），块内连续、块间无所谓。每个请求维护一张 **block table**，第 $i$ 项给出第 $i$ 个逻辑块对应的物理块号以及该块已填入的位置数。类比是直白的：块 = 页，token = 字节，请求 = 进程。

attention 的形式不变。原式是

$$a_{ij}=\frac{\exp(q_i^\top k_j/\sqrt{d})}{\sum_{t=1}^{i}\exp(q_i^\top k_t/\sqrt{d})},\qquad o_i=\sum_{j=1}^{i}a_{ij}v_j$$

把 key/value 按块分组，记第 $j$ 块为 $K_j=(k_{(j-1)B+1},\dots,k_{jB})$、$V_j$ 同理，则

$$A_{ij}=\frac{\exp(q_i^\top K_j/\sqrt{d})}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_i^\top K_t\mathbf{1}/\sqrt{d})},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_j A_{ij}^\top$$

分母的求和范围从「所有已见 token」换成「所有已见块」，**数学上完全等价**：softmax 的归约是可结合的，分块只是把一次大归约拆成若干次并累加（这正是 FlashAttention 在线 softmax 的同一套办法，见 [[llm-internals-05]]）。kernel 侧要做的额外工作只有三件：按 block table 取块、处理变长序列、处理块内未填满的位置。工程上 vLLM 把这几件事做成三个融合 kernel：reshape+block write 融合、block read 与 attention 融合（每个 warp 负责读一个块以保证合并访存）、block copy 批量融合（服务于 copy-on-write）。

对一个长度为 $N$ 的序列，块数是 $\lceil N/B \rceil$，浪费恰是最后一块的空槽：

$$W(N)=\lceil N/B\rceil \cdot B - N \le B-1$$

内部碎片的相对占比是 $\dfrac{W(N)}{N+W(N)}$。$B=16$、$N=512$ 时最坏 15/527 ≈ 2.85%、平均 1.44%；$N=2048$ 时最坏 0.73%。$B$ 越大碎片越大：$B=128$、$N=128$ 时最坏接近 50%。这就是「block size 取小」的动机。

### 第四步：块大小为什么是 16

论文的 ablation 给出双向约束：block 太小，kernel 无法充分并行地读 KV（每块 token 数少，访存与计算都不好铺开）；block 太大，内部碎片回升、可共享的粒度变粗。实测在 ShareGPT（长对话）上 block size 16–128 表现都很好；在 Alpaca（短指令）上 16、32 正常，更大就明显退化，因为序列长度已经短于 block。论文的结论是 16 既能喂饱 GPU、又不产生显著内部碎片，因此 vLLM 默认 16。

### 第五步：共享——从「省显存」变成「省显存 + 省算力」

块一旦等大，「同一段内容只存一份、多个请求各指向它」就变成改指针的操作。三种场景收益最大：

- **共享 prefix**：system prompt、few-shot 示例、agent 的工具说明在多请求间逐字节相同，其 KV 只依赖前缀本身（因果 attention 的性质），可以只算一次、只存一份。
- **parallel sampling**：一个 prompt 采 $n$ 个回答，prompt 段完全共享。
- **beam search**：$k$ 个 beam 共享到分叉点，分叉后才各占块。

后两者需要写时复制（copy-on-write）：某个序列要往共享块里写新 token 时，先把该块复制成私有块再写；引用计数为 1 的块可以直接原地追加。论文用 `fork`（复制 block table 并增加引用计数）、`append`（追加新 token）、`free`（引用计数归零则释放）三个原语统一实现了这些解码算法。

论文实测的显存节省（图 15，Alpaca 轨迹 OPT-13B）：parallel sampling 6.1%–9.8%，beam search 37.6%–55.2%；换成 ShareGPT 轨迹则分别是 16.2%–30.5% 与 44.3%–66.3%。共享 prefix 场景（WMT16 翻译，LLaMA-13B）：共享 1 个 80 token 示例时吞吐是 Orca(Oracle) 的 1.67×，共享 5 个共 341 token 示例时 3.58×。共享 prefix 的收益同时体现在两处：省显存，以及省掉重复 prefill 的算力（对应 TTFT，见 [[inference-serving-05]]）。

### 第六步：抢占与恢复

显存总会被吃满，引擎必须能抢占。因为一个请求的所有 token 状态必须同时在 GPU 上才能继续算，vLLM 采用**全有或全无（all-or-nothing）**的驱逐策略，并且把同一请求的多个序列（beam 的候选）作为 sequence group **成组调度**（gang scheduling），因为组内可能有共享块。恢复被驱逐的块有两条路：

- **swap**：把块拷到 CPU 内存，需要时拷回。开销受 PCIe 带宽限制，小块会产生大量小传输、拉低有效带宽。
- **recompute**：直接丢掉块，恢复时把「prompt + 已生成 token」重新 prefill 一遍。不占用 block，开销与 block size 无关。

论文的 ablation（图 19）结论是：block 小时 recompute 更划算，block 大时 swap 更划算，两者在 16–64 之间端到端相当；而 **recompute 的开销从未超过 swap 延迟的 20%**。这条结论要按今天的上下文长度重新审视：论文的序列上限只有 2048 token，而重算的算力开销是 $2 N_{\text{params}} N$——70B 模型重算 100k token 的 prefill，仅权重项就是 $2 \times 70\text{e}9 \times 1\text{e}5 \approx 14$ PFLOPs，单卡要 35 s 量级；而 100k token 的 KV 共 30.5 GiB（TP=4 时每卡 7.6 GiB），换出去不到 0.2 s（复算见「数值与代码验证」）。上下文越长，recompute 越吃亏。

### 第七步：收益如何变成吞吐

显存里能放的序列数 $n$ 由 $\text{KV 预算} / \text{每序列 KV}$ 决定，而 batch size 基本被 $n$ 卡住，吞吐又随 batch 上升（把 decode 从带宽受限往计算受限推，见 [[inference-serving-01]]）。把 20.4%–38.2% 的有效利用率换成接近 100%，并发上限提升约 2.5×–4.8×；论文在端到端测量里给出的数字是：同延迟下吞吐 2–4×，且序列越长、模型越大、解码算法越复杂收益越明显（§6.2：ShareGPT 上相对 Orca(Oracle) 可承受 1.7×–2.7× 的请求速率，相对 Orca(Max) 2.7×–8×，相对 FasterTransformer 最高 22×）。

代价也要说清：论文实测 PagedAttention 的 attention kernel 延迟比高度优化的 FasterTransformer **高 20%–26%**（访问 block table、额外的分支、变长序列处理）。它只影响 attention 算子、不影响 Linear 等算子，且换来的端到端吞吐远超这点损失。论文也在 Discussion 里明确：这套方法只在「动态分配 + 显存容量受限」的负载上成立；训练（张量形状静态）或非 LLM 的 compute-bound 推理上引入分页反而会因为间接寻址变慢。

## 数值与代码验证

以下数字均在本地用公式复算过，口径写明。

**单 token KV cache 与单请求占用（LLaMA-3-70B 架构：80 层、8 KV 头、d_head=128、bf16）**

| 序列长度 | 单请求 KV cache | 需要的 block 数（B=16） | block table 大小（8 B/项） | 表占 KV 的比例 |
| --- | --- | --- | --- | --- |
| 2 048 | 640 MiB | 128 | 1 KiB | 0.00015% |
| 8 192 | 2.50 GiB | 512 | 4 KiB | 0.00015% |
| 32 768 | 10.0 GiB | 2 048 | 16 KiB | 0.00015% |

元数据可以忽略，这解释了为什么「用间接寻址换连续性」在显存上是稳赚的。

**尾部碎片（内部碎片）占比**

| 序列长度 $N$ | 平均浪费 | 最坏浪费（$B=16$） | 最坏浪费（$B=64$） | 最坏浪费（$B=128$） |
| --- | --- | --- | --- | --- |
| 128 | 5.54% | 10.49% | 32.98% | 49.80% |
| 512 | 1.44% | 2.85% | 10.96% | 19.87% |
| 2 048 | 0.36% | 0.73% | 2.98% | 5.84% |

平均列按「最后一个块的填充位置在 $1..B$ 上均匀」取 $(B-1)/2$ 个空槽计算；最坏列取 $B-1$ 个空槽。这就是「$B=16$ 时内部碎片是百分之几」的来源，也是 Alpaca 这类短序列负载上大 block 会退化的原因。

**并发数：同一份显存，两种分配方式**

配置：LLaMA-3-70B bf16 权重 140 GB ≈ 130.4 GiB，跑在 4×H100 80 GB 上（TP=4，每卡权重 35 GB ≈ 32.6 GiB）。每卡 80 GiB 扣掉权重还剩约 47 GiB，取其 20 GiB（合计 80 GiB）作 KV 预算，其余留给激活、workspace 与调度水位。序列长度 8 192。下表按**整请求**口径计价：TP=4 时每个 8192 token 的请求在单卡上只占 640 MiB（每卡 2 个 KV 头），预算与单价同比例缩小，比值不变。

| 分配方式 | 每请求 KV（8192 token，整请求口径） | 可并发序列数 | 依据 |
| --- | --- | --- | --- |
| 朴素：按 max_len=8192 预分配连续块 | 2.50 GiB | 32 | 预留即占用，无论实际生成多长（80 GiB ÷ 2.50 GiB） |
| 朴素：叠加论文实测的有效利用率 20.4%–38.2% | 2.50 GiB 里只有 0.51–0.96 GiB 真在存 token | 6–12 | 32 × 20.4% ≈ 6.5、32 × 38.2% ≈ 12.2 |
| PagedAttention：$B=16$，尾部碎片 1.4%（平均）–2.9%（最坏） | 2.53–2.57 GiB | 31 | 80 GiB ÷ 2.57 GiB ≈ 31；论文口径的有效利用率接近 100% |

同一条因果链归一化看更干净：若平均序列长度对应的 KV 是 640 MiB（2048 token），80 GiB 预算给出 $80 \times 1024 / 640 = 128$ 个「全长序列」的容量。朴素预分配下真正有效的容量只有 $128 \times 20.4\% \approx 26$ 到 $128 \times 38.2\% \approx 49$ 个序列；分页分配下只剩尾部碎片，有效容量是 $128 / 1.029 \approx 124$ 个。**并发上限因此抬升约 2.5×–4.8×**（$124 \div 49 \approx 2.5$、$124 \div 26 \approx 4.8$；只看利用率的倒数则是 1/38.2% ≈ 2.6 到 1/20.4% ≈ 4.9，差额来自最后一个块的尾部浪费），这正是论文端到端测到的 2–4× 吞吐的量级来源。注意这个比值来自论文的实测利用率，不是「碎片率」；把它当成「分页一定带来 4.8×」是过度解读。

**论文报告值（引用，非本地复算）**

| 指标 | 数值 | 出处 |
| --- | --- | --- |
| 已有系统 KV cache 有效利用率 | 20.4%–38.2% | 论文图 2、§1、§3.1 |
| 端到端吞吐提升 | 2–4×（同延迟，vs FasterTransformer/Orca） | 论文摘要 |
| attention kernel 额外延迟 | +20%–26%（vs FasterTransformer） | 论文 §7.1 |
| 共享节省：parallel sampling / beam search | 6.1%–9.8% / 37.6%–55.2%（Alpaca）；16.2%–30.5% / 44.3%–66.3%（ShareGPT） | 论文 §6.3、图 15 |
| 共享 prefix 吞吐 | 1.67×（1 个示例，80 token）、3.58×（5 个示例，341 token） | 论文 §6.4、图 16 |
| recompute vs swap | 16–64 相当；recompute 开销 ≤ swap 的 20% | 论文 §7.3、图 19 |
| OPT-13B 单 token KV | 800 KiB（$2\times5120\times40\times2$） | 论文 §3 |

**分块 softmax 与 block 映射的等价性验证**

```python
import math

def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]

def blockwise_softmax(scores, B):
    """scores: 已经算好的 q·k/√d；按 B 分块跑在线 softmax（FlashAttention 的归约方式）。"""
    out, m_old, l_running = [0.0] * len(scores), None, 0.0
    for start in range(0, len(scores), B):
        blk = scores[start:start + B]
        m_new = max(blk) if m_old is None else max(m_old, max(blk))
        # 最大值变大时，之前块里已经 exp 过的分子要整体缩回去，归约才等价
        scale = 1.0 if m_old is None else math.exp(m_old - m_new)
        for i in range(start):
            out[i] *= scale
        l_running = l_running * scale + sum(math.exp(x - m_new) for x in blk)
        for i, x in enumerate(blk):
            out[start + i] = math.exp(x - m_new)
        m_old = m_new
    return [p / l_running for p in out]

scores = [0.1 * i - 2.0 for i in range(37)]      # 任意非整块长度
a, b = softmax(scores), blockwise_softmax(scores, B=16)
print("max abs diff:", max(abs(x - y) for x, y in zip(a, b)))   # 期望 1e-16 量级
```

**block table 上的 paged 解码（核心逻辑，教学版）**

```python
class PagedKVCache:
    """b=1、单层、单头的教学版：把 KV cache 存进等大物理块，靠 block table 定位。"""
    def __init__(self, block_size=16, n_blocks=1024):
        self.B, self.free = block_size, list(range(n_blocks))
        self.phys = {}                     # block_id -> [k/v 列表]
        self.table = []                    # 逻辑块 -> 物理块

    def _alloc(self):
        bid = self.free.pop(0)
        self.phys[bid] = []
        return bid

    def append(self, k, v):
        filled = sum(len(self.phys[b]) for b in self.table)
        if filled % self.B == 0:           # 上一块满了：按需再要一块
            self.table.append(self._alloc())
        self.phys[self.table[-1]].append((k, v))

    def gather(self):
        """按逻辑顺序拼回 K/V —— 物理上它们可以散落在任意块里。"""
        out = []
        for bid in self.table:
            out.extend(self.phys[bid])
        return out

    @property
    def waste_slots(self):
        return len(self.table) * self.B - sum(len(self.phys[b]) for b in self.table)


kv = PagedKVCache(block_size=16)
for t in range(50):                        # 50 个 token，绝对不占 64 个槽
    kv.append(k=t, v=-t)
print("blocks:", kv.table, "| waste slots:", kv.waste_slots)   # ceil(50/16)*16-50 = 14
```

（真实实现当然不会用 Python 列表；vLLM 按 `(num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim)` 预切一整块大张量，block table 是 GPU 上的整数数组，kernel 用 `pos // B` 与 `pos % B` 取块号与块内偏移，见 [[inference-serving-02]] 的调度上下文。）

**换出/重算的代价复算（用来回答追问；换出按单卡口径）**

| 场景 | 数据量 | 开销（按有效带宽折算） |
| --- | --- | --- |
| 单卡换出 10 GiB 到 CPU（PCIe 5.0 x16，实测有效约 50 GB/s） | 10 GiB | ≈ 0.2 s |
| 同样 10 GiB 走 NVLink（H100 的 900 GB/s 是双向合计，单向理论 450 GB/s） | 10 GiB | ≈ 24 ms |
| 重算 32k token 的 prefill（权重项 $2 \cdot N_{\text{params}} \cdot N$ 加 attention 二次项，按单卡 H100 bf16 稠密 989 TFLOPs 的 40% 利用率折算） | ≈ 6.0 PFLOPs | ≈ 15 s（单卡；TP=4 时约 4 s，未计调度、KV 重写与排队） |

重算的 FLOPs 是 $2 \times 70\text{e}9 \times 32768 \approx 4.59$ PFLOPs，加上 attention 的二次项 $2LHd_{head}N^2 \approx 1.4$ PFLOPs（$H=64$ 个 query 头、$d_{head}=128$、因果掩码下两个矩阵乘各占一半），合计约 6.0 PFLOPs；这个口径只算「一次额外前向」，实际恢复被抢占的请求还要重写 KV cache、重新排队。数量级上：重算 32k token（TP=4）约 4 s，而把同一个请求的 10 GiB KV 换出去（每卡 2.5 GiB，50 GB/s）只要约 50 ms。

## 常见追问

- **追问**：prefix caching 是怎么建立在 block 共享上的？
  - 要点：attention 是因果的，前 $t$ 个 token 的 KV 只依赖前 $t$ 个 token，因此「逐字节相同的前缀」其 KV 必然相同，可以按块复用。实现上做**块粒度**的最长前缀匹配（vLLM 用前缀块的 hash 命中已有物理块；SGLang 用基数树匹配任意长度的公共前缀），命中就只增引用计数、不重新 prefill。注意 RoPE 下同一 token 在不同绝对位置是不同的向量，所以缓存必须按位置对齐使用；前缀里任何一个字符变了（时间戳、用户名）都会让该位置之后全部失效。详见 [[inference-serving-05]]。
- **追问**：换出到 CPU 的代价是什么？
  - 要点：一次完整的 KV cache 拷贝，受 PCIe 带宽限制——单卡 50 GB/s 量级下 10 GiB 要 0.2 s 左右，而走 NVLink（H100 的 900 GB/s 是双向合计，单向 450 GB/s 量级）只要 20 多毫秒，所以跨卡换出远比换到 CPU 划算。同时换出会引入「被抢占的请求何时回来」的不确定性，尾延迟抖动变大。论文给了两个工程细节：block 太小时大量小传输会拉低有效 PCIe 带宽；recompute 的开销不随 block size 变化且不超过 swap 的 20%（论文的序列上限是 2048 token，长上下文要按现值重算）。
- **追问**：为什么 block size 常取 16？
  - 要点：下界由 kernel 并行度与访存效率决定（块太小，每次读的 token 太少，GPU 铺不开，而且换出时小传输拖累 PCIe）；上界由内部碎片与共享粒度决定（$B$ 大则最坏浪费 $B-1$、可共享的最细粒度变粗）。论文实测 ShareGPT 上 16–128 都行、Alpaca 上 16/32 之后明显退化，因此默认 16。
- **追问**：碎片化在真实负载下的表现是什么样？
  - 要点：真实负载的长度分布是重尾的（ShareGPT 的输入比 Alpaca 长约 8.4×、输出长约 5.8×，方差更大）。按最大长度预分配时，越重尾浪费越狠——短请求把宝贵的连续区段整段占死；分页之后浪费只与「最后一个块」和「序列长度分布」有关，与长度方差无关。所以收益最大的恰恰是长短混杂、长 prompt 多的负载（对话、agent、翻译 few-shot）。
- **追问**：GQA / MQA / MLA 之后 KV cache 变小了，PagedAttention 还有意义吗？
  - 要点：有，但收益结构会变。GQA/MQA 减少的是 KV 头数 $H_{kv}$（每个 block 变小、同样显存能装更多块），MLA 把 KV 压成低维 latent（DeepSeek-V2 每层只存 512 维 latent 加 64 维 RoPE 键，60 层合计 67.5 KiB/token，见 [[llm-internals-04]]）。分页要解决的是「动态增长 + 未知长度 + 需要共享」这三件事，跟单价的绝对值无关；单价变低后，碎片造成的绝对浪费变小，但共享 prefix 与抢占调度这两项收益依然存在，换出/重算的取舍反而更偏向 swap（数据量小、拷贝快）。
- **追问**：既然分页这么好，为什么训练里没人用？
  - 要点：训练的张量形状是静态的，显存可以提前规划、复用 buffer，分页带来的间接寻址只有成本没有收益。论文在 Discussion 里明确写了这个边界：需要「动态分配 + 显存容量受限」才划算。

## 公司变体

- **NVIDIA**：偏工程实现与 kernel 细节。NVIDIA 自家的 TensorRT-LLM 把这套机制命名为 **paged KV cache**，讨论重点通常在 block 布局（block size、每块张量形状、是否把 K/V 分成两块池）、paged attention kernel 的访存合并与 warp 分工、以及 paged KV 与 in-flight batching（continuous batching）在同一个 engine 里的配合。被追问到「为什么需要新的 kernel」时要能说出：间接寻址破坏了原本连续的访存模式，所以要么把 block read 融进 attention，要么先算好 block table 再让 kernel 按表取数；同时要能引用论文的 +20%–26% kernel 开销作为量级参考。也可能从显存账出发，要求你把 paged 与非 paged 的并发数都算出来。
- **Together AI**：偏服务系统与性价比。这是 vLLM 的主力维护方，自家也在同一套技术上运营推理服务，这条题通常落在「同样的 GPU 能服务多少并发/每百万 token 成本多少」这一类追问上：会问到 prefix caching 对多轮对话与 agent 负载的命中率、换出/重算对 p99 的影响、以及块粒度共享与请求路由（缓存亲和）怎么配合。回答时把显存 → 并发 → 吞吐 → 单位成本的链条算出来，比只讲页表结构更对味。

以上依据两家公开技术材料与岗位方向的侧重判断，不代表具体面试流程。

## 相关题目

- [[inference-serving-02]]：continuous batching——PagedAttention 腾出显存，iteration-level 调度把这些位置立刻填满，两者是配套的。
- [[inference-serving-05]]：prefix caching 与 prompt caching——块共享在跨请求场景的直接产品化。
- [[inference-serving-08]]：估算服务 70B 模型的显存——碎片与预留是那张清单里的一块。
- [[llm-internals-02]]：KV cache 的公式与规模化内存影响，本页所有数字的起点。
- [[llm-internals-03]]：MQA / GQA——把 block 变小的一条正交路径。
- [[llm-internals-04]]：MLA——把 KV cache 压成 latent，改变每块的字节数但不改变分页的必要性。
- [[llm-internals-05]]：FlashAttention——分块归约的同一套数学，PagedAttention 的 kernel 是它的近亲。

## 参考资料与归属

- [Paged Attention in LLMs](https://outcomeschool.com/blog/paged-attention-in-llms)，Amit Shekhar (Outcome School)，2026-03-29：block/block table 的直觉、酒店类比、内部与外部碎片的划分、共享 prefix 与 beam search 的动机。
- [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work)，Amit Shekhar (Outcome School)，2026-06-17：KV cache 作为服务瓶颈的定位、PagedAttention 与 continuous batching 的配合、vLLM 的整体结构。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Woosuk Kwon 等 (vLLM, SOSP 2023)，2023-09-12（延伸）：全部量化结论出自该论文——20.4%–38.2% 的有效利用率、分块 softmax 的公式（式 4）、waste 限于一个块的论述、block size ablation、recompute/swap 对比、以及 2–4× 吞吐。前两篇博客未覆盖的推导、公式与 ablation 数字来自这篇论文。
- 站内交叉核对：[[llm-internals-02]] 的 KV cache 公式与 [[inference-serving-08]] 的显存清单。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
