---
type: question
id: inference-serving-05
topic: 推理、服务与 GPU 性能
order: 5
question: 解释 prefix caching / prompt caching。什么时候应该使用它？什么会导致已缓存的 prefix 失效？
question_en: Explain prefix caching / prompt caching. When should it be used and what invalidates a cached prefix?
asked_at: [Moonshot AI, Character.AI]
level: 进阶
tags: [prefix-caching, prompt-caching, radix-attention, 成本]
sources:
  - title: How does Prompt Caching work?
    url: https://outcomeschool.com/blog/how-does-prompt-caching-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: SGLang：Efficient Execution of Structured Language Model Programs（延伸）
    url: https://arxiv.org/abs/2312.07104
    author: Zheng et al. (RadixAttention)
    published: 2023-12-12
related: [inference-serving-03, inference-serving-01, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> prefix caching（prompt caching）是跨请求复用同一段前缀已经算好的 KV：因果注意力保证前 $t$ 个 token 的 K/V 只由这 $t$ 个 token 及其绝对位置决定，和后面追加什么无关，所以新请求只要前缀逐 token 相同，就能直接引用已有的物理 KV block，跳过这段 prefill。
> 收益分两层：省掉这段 prefill 的算力与 TTFT（70B、bf16 口径下 4096 token 前缀约 $2NL = 0.57$ PFLOPs，在 H100 SXM 上 0.58 s@100% MFU），以及共享物理块带来的显存节省（32 个并发请求共享同一段 4096 token 前缀，KV 占用从 40 GiB 降到 1.25 GiB）。
> 缓存的对象是「前缀」而不是任意片段，命中要求逐 token 完全一致，所以在某个位置改了任何东西，该位置之后的缓存全部作废；判断值不值得开的标准是「命中率 × 命中长度」，而不是缓存开关本身。

## 面试官在考什么

- 能否从因果性推出「只有前缀可复用」，而不是把 prompt caching 当成一个部署开关背下来。
- 能否分清两层收益的边界：跳过 prefill 计算（省算力、省 TTFT）与共享 KV block（省显存），以及它们各自在什么负载下才成立。
- 能否说出引擎里的实现粒度：block 级最长前缀匹配、链式 hash、基数树 + LRU 驱逐。
- 能否逐条列出失效原因，尤其是「动态内容放在最前面」这个最常见的自伤写法。
- 能否给出可观测指标与取舍：命中率 × 命中长度、只写不读的浪费、缓存占显存反过来挤压 batch size、多租户隔离。

常见错误答案：

- 说 prompt caching 缓存的是「回答」或「prompt 文本」。缓存的是 KV；按语义复用最终答案是 semantic caching，两者机制与失效条件完全不同。
- 说「改一个词没关系，语义一样」或「把最常问的问题排前面」。匹配发生在 token 级，逐 token 完全相同才命中。
- 只讲省钱不讲延迟，或者反过来认为开了缓存一定省显存——在被复用之前，缓存先多占一份显存。

## 原理与推导

### 1. 为什么只有「前缀」能复用

decoder 每一层的隐状态满足 $h_i = f(x_{1:i})$：因果掩码让位置 $i$ 的输出只看到它自己和它之前的 token。K/V 是在 $h_i$ 上做线性投影再旋入位置得到的，因此

$$K_i = \mathrm{RoPE}(W_K h_i,\ i), \qquad V_i = W_V h_i$$

都只是「前 $i$ 个 token + 第 $i$ 个位置」的函数。往后面追加 token 不会改变任何 $i \le t$ 的 $K_i, V_i$，这就是前缀可以整体复用的数学依据：**可复用的部分必然是一段前缀**——前 $t$ 个位置逐 token 匹配就能整段引用；中间某个位置被改动，它之前的仍然命中，它之后的所有层、所有后续位置全部要重算。

两个推论值得在面试里点出来：

- 双向编码器（BERT 式）没有这个性质：位置 $i$ 的表示依赖整个序列，改动后缀也会改变前缀的表示，所以不存在 prefix cache。
- 位置信息如果不写进 K，而是像 ALiBi 那样直接加在 attention score 上，K/V 本身与绝对位置无关，理论上允许把某段 KV 挪到别的位置复用；主流实现（RoPE + 绝对位置对齐）不这么做，见第 4 节第 3 条。

### 2. 两个层次：prefill 复用与存储复用

| 层次 | 复用的是什么 | 省的是什么 | 典型形态 |
| --- | --- | --- | --- |
| prefill 重用 | 同一段前缀已经算好的 KV | 重复的 prefill 计算 → TTFT 与算力 | 引擎的 automatic prefix caching、厂商 API 的 prompt caching |
| 存储复用 | 同一份 KV 的物理块（可落盘、可跨实例） | 显存容量与重算次数 | block 共享、copy-on-write、KVCache 池化、跨实例缓存 |

两者常被统称 prompt caching，但优化目标不同。厂商 API 卖的主要是第一层：命中的 token 更便宜、首 token 更快，KV 存在厂商侧。自建引擎里，vLLM 的 automatic prefix caching、SGLang 的 RadixAttention 同时给两层：既跳过 prefill，又让多个请求引用同一批物理块。理解的检查点在于：**第一层的收益只取决于命中长度，第二层的收益取决于「同时存活的、前缀相同的请求数」**，后者才是显存账。

### 3. 实现：block 粒度最长前缀匹配

KV 在引擎里按固定大小的 block 分配（典型 16 个 token，见 [[inference-serving-03]]），所以前缀匹配也以 block 为单位：

1. 请求进来先算 token id 序列，按 block 切分，对每个完整块算一个键 $\text{hash}_i = H(\text{hash}_{i-1},\ \text{tokens}_i,\ \text{meta})$：输入是该块的 token id、上一块的 hash，以及所有会影响 KV 的元信息（模型与权重版本、adapter、量化方案等）。链式依赖让「改一处、整条链失效」变成 hash 表的自然行为，不需要显式失效逻辑。
2. 用这个键在缓存表里查物理块。命中就引用（引用计数 +1），未命中的部分照常 prefill；新算出的块在引用计数归零后不立即释放，而是留在空闲队列里等驱逐，这样下一个同前缀请求能命中。
3. 追加写入时用到共享块，触发 copy-on-write：复制最后一块再写，前面的块保持只读共享。
4. 驱逐用 LRU。命中率与可用显存是一对直接矛盾：留得越久命中率越高，能开的 batch 越小。

SGLang 的 RadixAttention 把这件事做成基数树：节点按 token 序列组织，多个并发请求共享同一棵子树，前缀更长的分支自然复用，论文在 agent 控制、逻辑推理、few-shot、JSON 解码、RAG、多轮对话这些负载上报告了最高 6.4× 的吞吐提升——这是 SGLang 整体优化的口径（RadixAttention 加压缩有限状态机等），不是单看 prefix caching 一项的倍数。

### 4. 什么会让已缓存的 prefix 失效

1. **前缀中任何位置的任何改动**：逐 token 匹配，改一个 token 就让该位置之后的全部失效。BPE 会把改动扩散到 token 边界之外——改一个词可能让后面若干 token 的切分同时变化，失配点比肉眼看到的更靠前。
2. **把动态内容放在最前面**：日期、时间戳、user id、随机 session id、每轮变化的会话摘要、顺序不稳定的检索结果、键序不稳定的 JSON 序列化，只要落在前缀里，整段缓存每次都是 miss。正确做法是稳定内容在前（工具定义、system 指令、大文档、few-shot 示例）、易变内容在后（用户消息、本轮检索片段），需要时用部分厂商 API 提供的显式 cache breakpoint 标注稳定段的边界。
3. **位置相关的失效**：RoPE 把绝对位置旋进了 K，同一段 token 出现在不同绝对位置就是不同的向量，所以缓存的 KV 只能当「位置对齐的连续前缀」用，不能把中间某段抽出来挪位复用（除非显式反旋转再重算，主流引擎不做）。详见 [[llm-internals-08]]。
4. **影响 KV 的元信息变了**：换模型或权重版本、换量化方案（权重或 KV 的 bit 宽变了，数值就不同）、换 tokenizer 或 chat template、换 LoRA adapter（adapter 不同则 KV 不同）、MLA 这类 latent cache 的形状变化、多模态输入的编码结果变化——这些都必须进 cache key，否则会命中错误的 KV。
5. **存储层的失效**：显存压力下被 LRU 驱逐；厂商侧 TTL 到期（源文口径默认 5 分钟，也有 1 小时档，更长 TTL 的写入计价更高）；进程重启或滚动发布清空显存中的 KV；多副本之间默认不共享缓存，负载均衡把下一轮请求路由到没有这份前缀的实例上就是 miss——这也是缓存亲和路由（cache-aware routing）的动机，集群级缓存与 prefill/decode 分离的配合见 [[inference-serving-15]]。
6. **应用层的新鲜度失效**：前缀命中了，但前缀里的文档已经更新——这不是缓存机制的问题，而是要自己把内容版本号写进前缀或 cache key，让旧前缀自然失配。

需要区分「不命中」和「失效」：换了措辞但语义相同的提问，本来就匹配不上，那是 semantic caching 该解决的层；prefix caching 只认 token。

### 5. 收益怎么算

- **算力与 TTFT**：prefill 的算力约为 $2NL$（$N$ 为参数量、$L$ 为这段 token 数，系数 2 是乘加各算一次 FLOP）。命中 $L$ 个 token 就省掉这部分，量级等于一次前向。注意这个式子只算了线性层，attention 的二次项是 $2 L_{\text{layers}} d_{\text{model}} L^2$（含因果掩码折半），对 70B、$d_{\text{model}}=8192$、80 层，两项在 $L \approx 1.07 \times 10^5$ 时同量级，典型前缀（几千到几万 token）用 $2NL$ 足够。
- **显存**：共享前缀的 $k$ 个请求只需要一份块，省下 $(k-1)\cdot L \cdot m$，其中 $m$ 是每 token 的 KV 字节数（LLaMA-3-70B GQA-8、bf16 是 320 KiB/token，见 [[llm-internals-02]]）。
- **成本**：厂商计价口径下写缓存约 1.25×、读缓存约 0.1×（相对普通输入 token 单价）。设前缀被复用 $k$ 次，无缓存花 $kL$，有缓存花 $1.25L + 0.1(k-1)L$，解出盈亏平衡 $k^{\star} = (1.25-0.1)/(1-0.1) \approx 1.28$——**同一个前缀只要被用到第二次就开始省钱**，$k=10$ 时省 78.5%。1.25× 是计费口径而非物理开销：算的部分本来就要算，写缓存额外付出的主要是把 KV 留在显存里的机会成本。
- **要看的指标**：命中率 × 命中长度，工程上等价于「所有请求命中的 prefill token 总数 / 全部 prompt token 总数」。只看命中率会被大量短前缀刷高，只看命中长度会忽略长尾请求。另外要监控只写不读的块占比（纯亏 25% 的单子）和驱逐率（命中率突然掉，多半是显存被别的负载挤了）。

### 6. 什么时候该用

| 负载 | 是否划算 | 原因 |
| --- | --- | --- |
| 多轮对话 | 很划算 | 每轮都重发全部历史，天然是最长公共前缀不断增长 |
| 固定长 system prompt / 安全策略 | 很划算 | 前缀稳定且长，跨用户复用面大 |
| few-shot 固定示例、agent 的工具定义 | 很划算 | agent 每一步都重发同一份前缀，复用次数极高 |
| RAG 的固定指令 + 检索片段 | 划算，但要排版 | 指令在前可命中；检索结果每轮不同，必须放在后面 |
| 同一模板 + 不同后缀的批处理 | 划算 | 命中长度固定且可预测 |
| 单次请求、前缀只有几十 token | 不划算 | 省下的算力小于写入溢价与管理开销 |
| 每个请求前缀都不同（如随意拼接的用户文本） | 不划算 | 命中率接近 0，还白占显存 |
| 显存紧张、并发优先 | 需权衡 | 缓存块挤压 batch size，吞吐可能反而下降 |
| 跨租户共享前缀 | 需隔离 | 正确实现下 hash 不同不会误命中，但共享物理块带来侧信道与合规风险，生产上按租户分桶或在 cache key 里加租户标识 |

## 数值与代码验证

**表 1：复用一段前缀的账（口径：LLaMA-3-70B，$N=70\times10^9$，仅线性层 $2NL$；H100 SXM bf16 稠密 989.4 TFLOPS、HBM3 3.35 TB/s；KV 按 80 层、8 个 KV 头、head_dim 128、bf16 = 320 KiB/token）**

| 前缀长度 $L$ | prefill 算力 $2NL$ | 重算耗时 @100% MFU | 重算耗时 @40% MFU | KV 占用 | 读一遍 KV @3.35 TB/s |
| --- | --- | --- | --- | --- | --- |
| 4096 | 0.573 PFLOPs | 0.58 s | 1.45 s | 1.25 GiB | 0.40 ms |
| 5000（源文例子） | 0.700 PFLOPs | 0.71 s | 1.77 s | 1.53 GiB | 0.49 ms |
| 32768 | 4.59 PFLOPs | 4.64 s | 11.6 s | 10.0 GiB | 3.2 ms |

两个可以直接引用的常数：70B 每 token 的 prefill 是 140 GFLOPs；H100 SXM 上 100% MFU 的 prefill 吞吐是 7067 token/s，乘 MFU 就是现实值。源文的例子是 5000 token 指令加 50 token 提问：无缓存时每次都要 prefill 5050 个 token，命中后只剩 50 个，prefill 时间从 1.79 s 降到 18 ms（40% MFU 口径），约 100 倍——**缓存命中把一段计算换成了内存读取**：同样 4096 token，重算是 0.58 s 的算力，把 KV 读回来只要 0.40 ms，比值约 1450。

**表 2：共享前缀的显存账（4096 token 前缀、320 KiB/token）**

| 场景 | KV 占用 |
| --- | --- |
| 32 个并发请求各存一份 | 40.0 GiB |
| 32 个请求共享同一份前缀块 | 1.25 GiB |

**表 3：块粒度命中长度（block = 16 token，缓存的前缀 64 token = 4 块）**

| 新请求与缓存前缀的差异 | 命中长度 |
| --- | --- |
| 完全相同 | 64 token（4 块） |
| 改第 0 个 token（第 0 块内） | 0 token |
| 改第 15 个 token（第 0 块末尾） | 0 token |
| 改第 40 个 token（第 2 块中间） | 32 token（前 2 块） |
| 改最后一个 token | 48 token（前 3 块） |
| 末尾追加 10 个 token | 64 token（全部命中） |

**表 4：成本盈亏平衡（源文的厂商计价口径：写 1.25×、读 0.1×，以「前缀长度 L 的普通输入价为 1」为单位）**

| 复用次数 $k$ | 无缓存 | 有缓存 | 节省 |
| --- | --- | --- | --- |
| 1 | 1.00L | 1.25L | 亏 25% |
| 2 | 2.00L | 1.35L | 32.5% |
| 10 | 10.00L | 2.15L | 78.5% |
| 100 | 100.00L | 11.15L | 88.8% |

```python
import hashlib

BLOCK = 16

def chain_hashes(tokens, block=BLOCK):
    """块粒度链式 hash：第 i 块的键依赖前一块，改一处则整条链失效。"""
    out, prev, n_full = [], "", len(tokens) - len(tokens) % block
    for i in range(0, n_full, block):
        payload = prev + "|" + ",".join(map(str, tokens[i:i + block]))
        prev = hashlib.sha256(payload.encode()).hexdigest()[:8]
        out.append(prev)
    return out

def hit_tokens(new_tokens, cached_hashes):
    hit = 0
    for a, b in zip(cached_hashes, chain_hashes(new_tokens)):
        if a != b:
            break
        hit += 1
    return hit * BLOCK          # 命中长度 = 最长公共前缀向下取整到 block

prefix = list(range(64))
base = chain_hashes(prefix)
print(hit_tokens(prefix, base))                              # 64
print(hit_tokens([999] + prefix[1:], base))                  # 0   改第一个 token
print(hit_tokens(prefix[:40] + [999] + prefix[41:], base))   # 32  改第 40 个 token
print(hit_tokens(prefix + [100, 101], base))                 # 64  尾部追加不影响
```

```python
import torch

torch.manual_seed(0)
d = 64
Wk = torch.randn(d, d) / d ** 0.5
x = torch.randn(32, d)                     # 32 个 token 的隐状态

def rope(v, pos0):
    """把绝对位置旋进 K：位置不同，同一段 token 的 K 就不同。"""
    pos = torch.arange(pos0, pos0 + v.shape[0]).float()[:, None]
    half = v.shape[-1] // 2
    freq = 10000 ** (-torch.arange(half).float() / half)
    ang = pos * freq[None, :]
    a, b = v[..., :half], v[..., half:]
    return torch.cat([a * ang.cos() - b * ang.sin(),
                      a * ang.sin() + b * ang.cos()], dim=-1)

K_full = rope(x @ Wk, 0)            # 一次 prefill 32 个 token
K_prefill8 = rope(x[:8] @ Wk, 0)    # 只 prefill 前 8 个 token
K_shift8 = rope(x[:8] @ Wk, 4)      # 同样 8 个 token，挪到位置 4..11

print(torch.equal(K_full[:8], K_prefill8))          # True  前缀 KV 只依赖前缀
print(torch.equal(K_full[:8], K_shift8))            # False 位置变了 KV 就变了
print(float((K_full[:8] - K_shift8).abs().max()))   # 4.6908
```

与源文对照：源文用 5000 token 指令 + 50 token 提问的例子说明「每次都在重算那 5000 个 token」，并给出写 1.25×、读 0.1×、默认 TTL 5 分钟的计价口径，以及「稳定在前、易变在后」的排序原则；这里的 FLOPs、KV 字节数、耗时与盈亏平衡点是用 LLaMA-3-70B 与 H100 SXM 的参数自行复算的，源文未给模型级数字。RadixAttention 与 6.4× 吞吐提升来自 SGLang 论文摘要，属于整篇论文的优化口径。

## 常见追问

- **追问**：prefix caching 和 KV cache 是什么关系？
  - 要点：KV cache 是单个请求内部、decode 每步复用自己历史的机制；prefix caching 是同一个 KV 结构在请求之间、甚至实例之间被复用。前者是后者的前提，后者多出来的是「跨请求索引 + 共享 + 驱逐」这套缓存管理，以及一套失效规则。
- **追问**：为什么只能缓存前缀，不能缓存中间片段？
  - 要点：K/V 是「左侧全部 token」的函数，改动中间位置会让它之后的所有 K/V 变化，而它之前的保持不变——可复用的部分天然是前缀。RoPE 还额外要求位置对齐，所以连「同一段前缀挪到别的偏移量」都不行。
- **追问**：多副本部署下怎么让缓存真正命中？
  - 要点：默认各实例的缓存互相独立，路由随机就会把命中率打散。常见做法是会话粘性（同一会话固定路由到同一实例）、以 prompt 前缀 hash 为键的一致性哈希路由、或者把 KV 做成集群级/分离式的共享池，并把 prefill 与 decode 的调度一起考虑（见 [[inference-serving-15]]）。代价是热点实例与负载不均，需要和吞吐目标一起调。
- **追问**：有没有开了 prefix caching 反而变差的场景？
  - 要点：命中率低的负载会白付写入溢价并多占显存；缓存块挤压 batch size 会降低整体吞吐；长 TTL 的写入计价更高；前缀匹配本身有每请求的 hash 与查表开销；短前缀（几十 token）省下的算力覆盖不了这些固定成本。
- **追问**：跨用户共享前缀有什么风险？
  - 要点：正确实现下只有 hash 完全相同才命中，不会读到别人的内容；但「谁能命中同一前缀」本身可能被推断，且用户数据会以 KV 形式留在别人可触达的缓存里，合规上通常要求按租户分桶或把租户标识写进 cache key。共享前缀还可能让一个租户淘汰另一个租户的缓存，造成性能干扰。
- **追问**：prompt caching 和 semantic caching 的区别？
  - 要点：prompt caching 复用 KV，命中条件是两个请求的前缀逐 token 相同，模型仍然要读新内容、生成新回答；semantic caching 复用最终答案，命中条件是两次提问语义相同，风险是答错。前者不改变输出质量，后者可能。
- **追问**：怎么确认缓存真的生效了？
  - 要点：看引擎指标与厂商 API 的 usage 字段——分别给出写入缓存的 token 与从缓存读取的 token；自建引擎看按块统计的命中 token 数与驱逐数。如果写入一直是正的、读取一直是零，先查前缀里是不是混进了动态内容。

## 公司变体

`asked_at` 的两家都是产品侧大规模服务的公司，这题偏工程实现，而不是数学推导——因果性那段推导两句话就够，考察重点会落在缓存管理与成本上。

- **Moonshot AI**：长期做长上下文与大规模在线服务，公开过以 KVCache 为中心的分离式服务架构（Mooncake），因此更可能追问集群尺度的问题：缓存放在哪个实例、prefill 与 decode 分离后缓存怎么迁移、怎么做缓存感知路由提高命中率、多租户与配额下缓存如何隔离与计费。
- **Character.AI**：海量高并发多轮对话场景，每个角色有稳定的 persona/system 前缀但角色数量极多，更可能追问会话级问题：多轮对话的命中率怎么维持、长会话历史如何截断与摘要而不摧毁前缀、缓存占用的显存与 batch size 如何权衡、驱逐策略抖动时怎么稳住 p99。

以上是按两家公开技术方向与题目性质推断的考察侧重，具体面试流程未见公开披露。

## 相关题目

- [[inference-serving-03]]：PagedAttention 与 block 分配，prefix caching 的匹配与共享就建立在这套块管理之上。
- [[inference-serving-01]]：prefill 与 decode 的瓶颈差异，解释为什么跳过 prefill 直接改善 TTFT。
- [[llm-internals-02]]：KV cache 的显存公式与每 token 字节数，本节共享前缀的显存账用它做单位换算。
- [[llm-internals-08]]：RoPE 与位置插值，解释为什么缓存必须位置对齐、不能挪位复用。

## 参考资料与归属

1. [How does Prompt Caching work?](https://outcomeschool.com/blog/how-does-prompt-caching-work)，Amit Shekhar（Outcome School），页面标注 2026-06-09。提供 prompt caching 的定位（复用 prefill 阶段建立的 KV）、exact-prefix 规则与「稳定在前、易变在后」的排序原则、cache write 与 cache read 的区分及 1.25×／0.1× 计价口径、默认 5 分钟 TTL 与更长 TTL 的取舍、应当放进缓存的内容清单（system 指令、大文档、工具定义、few-shot 示例）、与 semantic caching 的对比，以及 RAG 与 agent 场景的适用性。
2. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（延伸），Zheng et al.，2023-12-12。RadixAttention 用基数树做 KV cache 复用、以及「在 agent 控制、逻辑推理、few-shot、JSON 解码、RAG、多轮对话上最高 6.4× 吞吐提升」的结论取自该文摘要；第 3 节的最长前缀匹配与 LRU 驱逐以该文机制为参照，块粒度的链式 hash 与 copy-on-write 属于 vLLM 一类引擎的通行做法（见 [[inference-serving-03]]），论文外的数字（FLOPs、KV 字节、耗时、盈亏平衡点）按 LLaMA-3-70B 与 H100 SXM 参数自行复算。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
