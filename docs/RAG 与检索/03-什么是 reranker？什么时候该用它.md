---
type: question
id: rag-03
topic: RAG 与检索
order: 3
question: 什么是 reranker？什么时候该用它？cross-encoder 会给你带来什么开销？
question_en: What is a reranker, when should you use one, and what does a cross-encoder cost you?
asked_at: [Cohere, Microsoft, Perplexity]
level: 进阶
tags: [reranker, cross-encoder, 两阶段检索, 延迟]
sources:
  - title: How does a Reranker work?
    url: https://outcomeschool.com/blog/how-does-a-reranker-work
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models（延伸）"
    url: https://arxiv.org/abs/2104.08663
    author: Thakur et al. (NeurIPS 2021)
    published: 2021-04-17
related: [rag-02, rag-04, rag-01]
updated: 2026-09-28
---

## 一句话答案

> reranker 是两阶段检索的第二阶段：第一阶段用 bi-encoder（或 BM25）从全语料里捞出 50–100 个候选，reranker 用 cross-encoder 把 query 与每个候选拼成**一个**序列做一次完整前向，输出相关性分数并重新排序，最后只把 3–10 段喂给 LLM。
> 它更准是因为 token 级交叉注意力让 query 与 document 的每个词在每一层互相看见，能捕捉否定、条件、实体约束这些双塔在压缩 document 时丢掉的信号；代价是**无法预计算**——$N$ 个候选就是 $N$ 次前向。以 110M 参数、$N=100$、每对 288 token 计，一次重排约 $6.3\times10^{12}$ FLOPs，在 A100（312 TFLOPS bf16）上理论下限约 20 ms，实际几十到几百毫秒，且延迟与 $N$ 成正比，所以 $N$ 要卡在 50–100。
> 召回噪声大、top-k 分数挤在一起、上下文预算只够 3–5 段时收益最大；单轮简单查询、召回已经很准、TTFT 预算极紧时可以先不上。重排只能重排已有候选，召回上限由第一阶段决定。

## 面试官在考什么

- 能否从架构差异**推出**全部结论：bi-encoder 两塔可分离 → document 向量可离线预计算 → 能建 ANN 索引 → 快但粗；cross-encoder 输入拼接不可分离 → 每对一次前向 → 准但贵。这条因果链答不出来，后面全是背结论。
- 会不会给开销算式而不是形容词。$2 \times$ 参数量 $\times$ token 数是唯一需要的口径，要能自己代入 $N$、序列长度、模型大小和硬件算出量级。
- 是否清楚 reranker 是「精排」不是「召回」：它不扩大候选集，只重排候选集，recall@N 就是天花板。
- 有没有延迟预算意识：重排的几十到几百毫秒要和 embedding、ANN、LLM 的 TTFT 放在同一个预算里谈，并说得出什么时候该把它砍掉。
- 评估口径：nDCG@k / MRR@k 看排序，端到端看答案正确率，并且要做「有/无重排」的 A/B，不能只看离线指标。

常见错误答案：

- 「reranker 就是拿个小模型再算一遍相似度」——把 cross-encoder 说成更小的 bi-encoder。差别不是模型大小，而是 query 与 document 有没有在同一个前向里交互。
- 「加重排能提升召回」——重排不新增文档。正确答案没进第一阶段的前 $N$ 时，重排永远救不回来；这时该做的是提高 $N$、加 BM25 一路或改 chunking，而不是换更大的 reranker。

## 原理与推导

### 1. 两阶段漏斗：贵的算力只能花在少数候选上

先把「不分阶段」的代价摆出来。1M 个 chunk 的语料，如果直接用 cross-encoder 对全语料打分，每个 (query, document) 对 288 token：

$$1\times10^{6} \times 288 \times 2 \times 1.1\times10^{8} \approx 6.3\times10^{16}\ \text{FLOPs}$$

在 A100 的 312 TFLOPS 上理论下限是 203 s/query——比「只重排 100 个候选」贵四个数量级。所以架构上必须切成两段：

$$\text{cost} = \underbrace{O(N_{corpus})}_{\text{第一阶段：可预计算、ANN 次线性}} + \underbrace{O(N)}_{\text{重排：在线、对 } N \text{ 线性}},\qquad N \ll N_{corpus}$$

两段的目标函数不同，这是全部工程设计的分界线：第一阶段优化 recall@N（$N$ 取 50–100），第二阶段优化最终 top-k 的精度（$k$ 取 3–10）。只谈「哪个模型更好」而不谈这两段各自在优化什么，就是没有抓住这道题。

### 2. bi-encoder 与 cross-encoder 的差别到底在哪

**bi-encoder（第一阶段）** 把两侧分别编码再比距离：

$$s_{bi}(q,d) = \langle E_q(q),\, E_d(d)\rangle$$

关键在于 $E_d(d)$ 与 $q$ 无关，可以离线算完存进索引用 ANN 检索。代价是：模型必须在**不知道 query 是什么**的条件下把整个 document 压成一个定长向量，只能保留「对任意 query 都可能有用」的信息。

**cross-encoder（reranker）** 把两侧拼成一个序列送进模型：

$$s_{ce}(q,d) = f_\theta(q \oplus d)$$

$\oplus$ 是 token 级拼接。在每一层里，query 段的 token $i$ 与 document 段的 token $j$ 之间都有注意力：

$$a_{ij}^{(l)} \propto \exp\!\left(\frac{(W_Q h_i^{(l)})\cdot (W_K h_j^{(l)})}{\sqrt{d_k}}\right)$$

每一层都可以根据 query 重新对齐 document 的表示，模型实际建模的是 $P(\text{relevant} \mid q, d)$，条件在输入里，不必为未知的 query 留冗余。这就是它能处理下面这些信号的原因：否定（「不支持 X 的方案」）、条件与例外（「仅当 Y 发生时」）、数值与单位、型号/实体约束，以及用词完全不同但语义对应的情况。

两点容易被追问的边界：cross-encoder 的分数只在**同一个 query 的候选之间**有比较意义，跨 query 的绝对值不可比；它也不能当召回器用，因为没有任何东西可以预计算。

### 3. 开销模型：一次重排要花多少

transformer 前向的 matmul 项约等于 $2 \times$ 参数量 $\times$ token 数，于是

$$\text{FLOPs}_{\text{rerank}} \approx N \times L \times 2P,\qquad L = L_q + L_d$$

忽略的是 attention 的 $L^2$ 项（每 token 每层 $4L\,d_{model}$）：$d_{model}=768$、12 层时，$L=288$ 占 4.8%、$L=512$ 占 8.6%，到 $L=8192$ 反而超过参数项。所以短序列用 $2P$ 是够用的口径，长序列必须把 $L^2$ 项加回来。

代入一个具体例子（$N=100$、$L=32+256=288$、$P=1.1\times10^{8}$）：

$$\text{FLOPs} = 100 \times 288 \times 2 \times 1.1\times10^{8} = 6.336\times10^{12}$$

在 A100 的 312 TFLOPS bf16 上：$6.336\times10^{12} / 312\times10^{12} \approx 20.3$ ms 理论下限。同一口径下第一阶段的在线成本是 query 编码 $32 \times 2 \times 1.1\times10^{8} = 7.0$ GFLOPs 加上 1M 个 768 维向量的内积 $2\times768\times10^{6} = 1.5$ GFLOPs，合计约 8.6 GFLOPs——重排是它的约 **740 倍**。ANN 把内积那一项压到可忽略后差距更悬殊，因为在线成本几乎全在 query 编码上。

延迟与 $N$ 严格成正比，这就是「把 $N$ 控制在 50–100」的算术依据：

| 候选数 $N$ | rerank FLOPs | A100 理论下限 | 相对 $N=50$ |
| --- | --- | --- | --- |
| 20 | $1.27\times10^{12}$ | 4.1 ms | 0.4× |
| 50 | $3.17\times10^{12}$ | 10.2 ms | 1.0× |
| 100 | $6.34\times10^{12}$ | 20.3 ms | 2.0× |
| 200 | $1.27\times10^{13}$ | 40.6 ms | 4.0× |
| 1000 | $6.34\times10^{13}$ | 203.1 ms | 20× |

理论下限和实测差约 24–41 倍，必须知道差在哪，否则容量规划会错得离谱。BEIR 在 1M 篇 DBPedia 子集上实测 BM25+CE（6 层 384 维 MiniLM cross-encoder，重排 BM25 top-100）单查询 450 ms；同一模型换成 22.7M 参数、V100（125 TFLOPS fp16）复算，按 $L=300$ 是 10.9 ms、按 padding 到 512 是 18.6 ms，实测有效率只有 2.4%–4.1%。差的是 tokenize 与 Python 框架开销、batch 内长度不齐导致的 padding 浪费、以及小模型在小 batch 下 kernel 效率低。结论：$2PL$ 只用来判断数量级和「方案是否根本不可行」，容量规划必须实测。

除算力外还有两项开销：显存与 GPU 占用。110M 参数 fp16 权重 210 MiB，batch 100 × 288 token 的激活每个约 42 MiB（同时存活约 10 个则 0.41 GiB 量级），单卡放得下——瓶颈是算力和延迟，不是显存。真正的工程约束是重排服务要和 LLM 抢 GPU，因此通常独立部署、独立扩容。

### 4. 什么时候该用、什么时候不该用

| 信号 | 判断 | 动作 |
| --- | --- | --- |
| top-50 的相似度分数挤在一起（分数差 < 几个百分点） | bi-encoder 分辨不出细微差别，重排收益最大 | 上 reranker，$N$ 取 50–100 |
| 上下文预算只够 3–5 段 | 排序错一位就直接把错文档喂给 LLM | 上 reranker，$k$ 取 3–5 |
| 多路召回要融合（BM25 + dense + 字段） | 手工加权难以调，cross-encoder 是统一的裁判 | 先 RRF 合并再重排，见 [[rag-02]] |
| 单轮简单查询、head query 命中率已经很高 | 重排只是给本来就对的结果换顺序 | 先做 A/B，收益 < 1% 就不上 |
| TTFT 预算 < 100 ms 且重排要占一半以上 | 延迟挤不进去 | 降 $N$、换蒸馏小模型，或直接砍掉重排 |
| 候选文档很长（>2k token） | 截断会切掉答案所在的位置 | 先按 [[rag-01]] 切 chunk 再重排 |

一个必须说清的边界：**重排不改变召回上限**。recall@100 是天花板，$k$ 再小也越不过它。所以「先量 recall@N 再决定要不要加重排」是正确顺序；如果 recall@100 只有 0.6，先修召回。

评估要分三层报，口径不能混：recall@N 衡量第一阶段给的天花板，nDCG@k / MRR@k 衡量重排的排序质量，重排后 top-3 的答案正确率衡量端到端收益；再配一组「有/无重排」的线上 A/B。三层指标分开看的原因和做法见 [[rag-04]]。

### 5. 工程实现要点

- **漏斗参数**：hybrid 召回 100（BM25 50 + dense 50，RRF 融合）→ rerank → top-5。$N$ 和 $k$ 是两个独立旋钮：$N$ 买召回上限，$k$ 买上下文预算。
- **批处理与分桶**：100 个 pair 分批前向（示例代码里 batch 16）；按 token 长度分桶（bucket）能明显减少 padding 浪费——批次内长度差越大，补出来的 pad token 白吃的算力越多。截断一定用 `truncation="only_second"`（只截 document），否则长 query 会被截掉一半。
- **降级路径**：给重排设 80 ms 之类的超时，超时直接返回第一阶段顺序。重排是精度优化项，不该成为可用性的单点。
- **小模型与蒸馏**：把 cross-encoder 当 teacher 蒸馏到 bi-encoder 是标准做法；反过来 reranker 本身也该用蒸馏过的 6 层模型（22.7M vs 110M，前向算力降到约 1/4.85），离线 nDCG 掉 1–2 个点时通常仍然划算。
- **LLM 做 listwise 重排**：一次把候选编号喂给 LLM 让它排序，成本是「$N \times$ 每段 token」的一次 prefill。用 8B LLM 重排 20 个 256-token 候选约 $8.2\times10^{13}$ FLOPs，是上面 cross-encoder 重排 100 个候选的 **12.9 倍**，还有位置偏置和上下文长度限制。所以它只在 $N$ 很小（<20）、或需要模型给出排序理由时才用。
- **缓存与复用**：归一化后的 query 做键缓存重排结果，对重复 query（客服、搜索的 head 流量）直接命中。

## 数值与代码验证

BEIR 在 1M 篇 DBPedia 子集上给出的单查询延迟与索引大小（V100 + CUDA 11.0，dense 用精确检索、ColBERT 用 ANN，CPU 为 8 核 Xeon 8168）：

| 系统 | 类别 | 向量维度 | GPU 延迟 | CPU 延迟 | 索引 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 稀疏词法 | – | – | 20 ms | 0.4 GB |
| TAS-B / GenQ | dense bi-encoder | 768 | 14 ms | 125 ms | 3 GB |
| DPR | dense bi-encoder | 768 | 19 ms | 230 ms | 3 GB |
| ColBERT | late interaction | 128 | 350 ms | – | 20 GB |
| BM25+CE | cross-encoder 重排 top-100 | – | 450 ms | 6100 ms | 0.4 GB |

同一份数据上的效果（18 个 zero-shot 数据集平均 nDCG@10，我按论文 Table 2 逐列复算）：

| 系统 | 平均 nDCG@10 | vs BM25（均值口径） | 论文报告值 |
| --- | --- | --- | --- |
| BM25 | 0.4234 | 基线 | 基线 |
| TAS-B（最好的 dense） | 0.4148 | -2.0% | -2.8% |
| DPR | 0.2371 | -44.0% | -47.7% |
| ColBERT | 0.4306 | +1.7% | +2.5% |
| BM25+CE | 0.4763 | +12.5% | +11% |

口径说明：论文只给了「Avg. Performance vs. BM25」的相对值，没写聚合方式。我用两种口径复算过——先求各系统 18 个数据集的平均分再比，BM25+CE 是 +12.5%；先逐数据集算相对差再平均，是 +14.3%。两者都比论文的 +11% 略高，属于聚合口径差异，不影响结论：**重排把平均排序质量抬了一个档次（+11%～+14%），代价是延迟从 20 ms 级涨到 450 ms 级（约 22–32 倍）**，而且索引不用变大（重排复用第一阶段的候选）。

FLOPs 估算器（纯 Python，可直接跑）：

```python
def rerank_flops(n_candidates, q_len, d_len, params, attn_layers=None, d_model=768):
    """单次重排的前向 FLOPs。matmul 主导项 = 2 * 参数 * token；
    可选加上 attention 的 L^2 项（每层每 token 4*L*d_model）。"""
    L = q_len + d_len
    flops = 2 * params * n_candidates * L
    if attn_layers:
        flops += attn_layers * 4 * L * L * d_model * n_candidates
    return flops

for n in (50, 100, 200):
    f = rerank_flops(n, 32, 256, 110e6)
    print(f"N={n:3d}: {f:.3e} FLOPs -> A100(312 TFLOPS bf16) 下限 {f / 312e12 * 1e3:5.1f} ms")

# N= 50: 3.168e+12 FLOPs -> A100(312 TFLOPS bf16) 下限  10.2 ms
# N=100: 6.336e+12 FLOPs -> A100(312 TFLOPS bf16) 下限  20.3 ms
# N=200: 1.267e+13 FLOPs -> A100(312 TFLOPS bf16) 下限  40.6 ms
f = rerank_flops(100, 32, 256, 110e6, attn_layers=12)
print(f"含 attention 项: {f / 312e12 * 1e3:.1f} ms")   # 21.3 ms（+4.8%）
```

两阶段漏斗的可运行实现（注意 `padding=True` 按 batch 内最长补齐而不是补到 512，以及只截 document）：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = (AutoModelForSequenceClassification
         .from_pretrained("BAAI/bge-reranker-v2-m3", torch_dtype=torch.bfloat16)
         .cuda().eval())

@torch.inference_mode()
def rerank(query, candidates, top_k=5, batch_size=16, max_len=512):
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        enc = tok([query] * len(batch), batch,
                  padding=True,                 # 补到 batch 内最长，不是 max_len
                  truncation="only_second",     # 只截 document，保住 query
                  max_length=max_len, return_tensors="pt").to("cuda")
        scores.append(model(**enc).logits.view(-1).float().cpu())
    scores = torch.cat(scores)
    order = torch.argsort(scores, descending=True)[:top_k]
    return [(candidates[i], float(scores[i])) for i in order]

# 生产上还要加：按长度分桶、80 ms 超时降级为第一阶段顺序、结果缓存
```

容量换算：按理论下限 20 ms 计，单卡约 50 QPS；工程上做到 60 ms 时约 16.7 QPS；BEIR 那种 450 ms 的实现只有 2.2 QPS。1000 QPS 的检索服务按 60 ms 计需要约 60 张卡专门做重排——这个数字必须在架构评审时摆出来，它决定了 $N$ 是 50 还是 200。重排的耗时最终要和生成阶段的 TTFT 放在同一个预算里分配，见 [[inference-serving-09]]。

## 常见追问

- **追问**：能直接用 cross-encoder 做召回吗？
  - 要点：不能。它对每个 (query, document) 都要重跑一遍前向，没有任何可预计算的部分，全语料扫描是不可行的。ColBERT 之所以能两用，是因为它保留 token 级表示、document 侧可以离线算并建索引。
- **追问**：late interaction（ColBERT）和 cross-encoder 的区别？
  - 要点：交互发生的位置不同。ColBERT 用 MaxSim $s(q,d)=\sum_i \max_j \langle E_{q_i}, E_{d_j}\rangle$，query 与 document 的每个 token 都保留独立向量、文档侧可预计算，所以比双塔准、比 cross-encoder 快；代价是索引膨胀——BEIR 同一份 1M 文档上 ColBERT 索引 20 GB，单向量 dense 只要 3 GB（约 6.7 倍），因为每个 token 都要存一个向量。
- **追问**：重排会不会把本来正确的文档排下去？
  - 要点：会。三种典型原因：训练数据域不匹配（MS MARCO 训出来的模型处理法律、医疗、代码语料时可能反向排序）、长文档截断把答案切掉、以及候选池里存在与 query 词面更像但实际不相关的干扰项。检测方法是在自己的标注集上比较重排前后的 recall@5 / nDCG@5，把掉分 case 单独捞出来看。
- **追问**：重排的分数能拿来当阈值做拒答吗？
  - 要点：默认不能。排序分数只保证同一个 query 内部有序，跨模型、跨 query 的绝对值不可比，换模型后阈值全部失效。要用就必须在自己的数据上做校准（例如按分位数定阈值），并且把「top-1 与 top-2 的分差」和「绝对分数」分开当特征。
- **追问**：$N$ 到底取多少？
  - 要点：由「recall@N 的边际收益」对「延迟的边际成本」决定，是可以算的：$N$ 从 50 提到 100 延迟翻倍（+10 ms 量级），如果 recall@100 相对 recall@50 只提升 1 个百分点，就不值。先把 recall 曲线画出来，再定 $N$。
- **追问**：候选段落特别长怎么办？
  - 要点：先按 [[rag-01]] 切成 256–512 token 的 chunk 再重排，否则只有开头一段参与打分。另一种做法是先用「标题 + 首段」做粗排的近似表示，取回全文后再进 LLM；但要注意这只是省算力，不改变重排本身的分辨率。
- **追问**：多路召回的结果怎么进重排？
  - 要点：先融合再重排。用 RRF 或加权归一化把 BM25 与 dense 的候选合成一个 50–100 的池子，再把池子交给 cross-encoder，让它替你做跨路的统一裁判——这比手调权重稳，也顺带解决了稀疏与稠密分数不可比的问题，见 [[rag-02]]。

## 公司变体

`asked_at` 里的三家都能问到这道题，但落点不同：

- **Cohere**：偏工程与产品取舍。rerank 是它的独立产品线，面试里会直接问「embedding 与 reranker 为什么都要卖、两阶段流水线怎么设计、reranker 在什么情况下值得那部分 latency」。回答要落到「什么时候不加」以及 API 形态的约束（单次请求的候选数与文档长度上限、计费粒度、超时与降级），而不是只讲注意力机制。数学部分通常停在「$2PL$ 与延迟线性于 $N$」这一层。
- **Microsoft**：偏平台与基础设施。Azure AI Search 把重排封装成产品里的 L2 ranking 阶段，直接作用于第一阶段的初始结果集、把语义上更相关的结果提到前面，所以面试会关心多租户下的容量与配额、索引与重排的部署形态、以及评测体系。另外重排模型的常用训练数据 MS MARCO 就出自微软，被追问「cross-encoder 拿什么数据训、域偏移怎么处理」的概率不低。
- **Perplexity**：偏架构选型与端到端预算。公开的岗位方向里包含 ranking & retrieval，公司题里有「cross-encoder、ColBERT、LLM 方案怎么选」和「50 个候选、只有 10 段预算怎么取舍、怎么知道选得好不好」这类问法。回答要把 ranker 的选择绑在端到端延迟预算（这类题给的口径通常是数秒内给出完整回答）与 A/B 指标上，替选型给出可证伪的判断标准。

以上是基于公开技术材料与岗位方向的侧重判断，具体题目以实际面试轮次为准。

## 相关题目

- [[rag-02]]：稀疏与稠密如何选择、hybrid 融合——重排的输入从哪来。
- [[rag-04]]：retrieval 与 generation 分开评估——重排该报什么指标。
- [[rag-01]]：chunking 策略——重排前必须先切对粒度。
- [[rag-09]]：HNSW / IVF-PQ / flat 的选择——第一阶段的 recall 与 latency 权衡，决定重排的天花板。
- [[inference-serving-09]]：TTFT / TPOT 与吞吐——重排延迟要算进同一个预算。
- [[rag]]：专题导读。

## 参考资料与归属

- [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work)，Amit Shekhar（Outcome School）。两阶段流水线、bi-encoder 与 cross-encoder 的对比、ColBERT 的定位、候选数与 top-k 要按精度/延迟权衡来调、以及「重排不能召回新文档」「分数只有序的意义」这些结论来自这篇；原文给的经验区间是召回 100–200 再取 top 5–10，正文按更保守的 50–100 → 3–10 讨论。
- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)（延伸），Thakur et al.（NeurIPS 2021）。「数值与代码验证」一节的延迟与索引表格、BM25+CE 的实现口径（重排 BM25 top-100，6 层 384 维 MiniLM cross-encoder）、以及各系统的平均 nDCG@10 来自这篇；其中的绝对平均值是我按论文 Table 2 的 18 个数据集复算的，聚合口径差异已在正文说明。$2PL$ 开销模型、A100 与 V100 的理论下限、容量与成本换算是我自己的推导，不属于上述资料。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
