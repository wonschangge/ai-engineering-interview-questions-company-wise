---
type: question
id: rag-05
topic: RAG 与检索
order: 5
question: 什么是 HyDE（假设性文档嵌入，hypothetical document embeddings）？什么时候它优于标准的稠密检索？
question_en: What is HyDE (hypothetical document embeddings), and when does it beat standard dense retrieval?
asked_at: []
level: 进阶
tags: [hyde, query-rewriting, 稠密检索]
sources:
  - title: How does HyDE work in RAG?
    url: https://outcomeschool.com/blog/how-does-hyde-work
    author: Amit Shekhar (Outcome School)
    published: 2026-07-06
related: [rag-02, rag-06, rag-01]
updated: 2026-09-28
---

## 一句话答案

> HyDE（Hypothetical Document Embeddings，假设性文档嵌入）先用 LLM 针对 query 写一段「假答案」，再把这段假答案编码成向量，用这个向量而不是 query 向量去检索。它成立的前提是检索里的 **query–document 非对称**：短问句和长文档在长度、体裁、词汇分布上都不一样，余弦相似度被系统性压低；假答案在体裁上像文档，于是「问答匹配」被换成了「文档—文档匹配」。
> 代价是每次查询多一次 LLM 生成（几十到几百毫秒、几十到上百个输出 token），而且收益强依赖数据：短查询、术语密集、零样本检索收益最大，**不是普适增益**。

## 面试官在考什么

- **能不能把问题归因到「非对称」而不是「query 太短」。** 正确的因果链是「长度/体裁/词汇分布不同 → 查询向量落在编码器训练分布的边缘 → 余弦相似度被压缩、排序不稳」，而不是「查询词少所以信息少」。
- **知不知道 HyDE 在方法谱系里的位置。** 它属于 query transformation 家族，和 rewrite / expansion / multi-query / step-back 并列；区别在于改写产物是**一段文档体裁的文本**，不是一串关键词或若干个查询。
- **算不算得清代价。** 多一次自回归生成 = 一次 prefill + $T_g$ 次 decode；batch-1 的 decode 是显存带宽受限而不是算力受限，所以延迟几乎随模型大小线性增长，并且**不会被 continuous batching 摊薄**（吞吐可以摊，单请求延迟不行）。
- **知不知道它会被谁吃掉收益。** 精确匹配需求（型号、错误码、人名）是 BM25 的主场，领域内已微调过 embedder 时收益同样被摊平；能说清优先级（先 reranker，再 hybrid，最后才是 HyDE）说明真在生产环境调过链路。
- **会不会评。** 把它做成可开关的召回策略，在 recall@k / nDCG@k（口径见 [[rag-04]]）上做 A/B，并对生成失败（空、过短、跑题）准备降级路径。

常见错误答案：

- 「HyDE 就是把 query 改写、扩写成同义词再检索。」——那是 query expansion。改写后的文本仍是查询体裁，非对称没有被解决。
- 「假设文档必须事实正确，写错了会污染向量。」——机制依赖的是体裁与领域词汇，不是事实正确性；细节错通常无害（最终上下文只来自真实 chunk）。真正致命的是**跑题**：模型把问题理解到另一个领域，假答案会把检索拖向更远的区域。

## 原理与推导

### 1. 要解决的问题：query–document 非对称

稠密检索把 query 和 chunk 编码到同一个向量空间，用余弦相似度排序。通用编码器通常在大量「短查询 ↔ 长段落」的对比学习数据上训练，学到的正是这种映射。但推理时两侧的角色是钉死的：

- 查询侧永远是用户输入，典型长度 5–20 个 token，体裁是疑问句、关键词串，甚至只有 2–3 个词；
- 索引侧是 chunk，长度由 chunking 策略决定（[[rag-01]]），典型 256–512 个 token，体裁是陈述段落。

两侧的分布不同，向量也就落在空间里不同的区域。把编码器简化成对 token 向量做均值池化 $v = \frac{1}{n}\sum_{i=1}^{n} t_i$，并假设每个 token 向量由一个共享的内容分量 $\mu$ 加独立噪声 $\epsilon_i$（$\mathrm{Var}(\epsilon) = \sigma^2$）组成，则

$$v = \mu + \frac{1}{n}\sum_{i=1}^{n}\epsilon_i, \qquad \mathrm{Std}(v - \mu) = \frac{\sigma}{\sqrt{n}}$$

内容分量不随 $n$ 变化，噪声按 $1/\sqrt{n}$ 衰减。查询 $n_q = 10$、chunk $n_d = 256$ 时，查询侧相对于 chunk 侧的噪声是 $\sqrt{256/10} \approx 5.1$ 倍。这个模型忽略了 attention、归一化、各向异性等真实因素，只能给方向和量级，但它解释了工程上都能观察到的两个现象：

- 相关 chunk 的绝对相似度被压低（查询侧的噪声项把余弦拉向背景均值，相关与不相关的差距一起变窄）；
- top-k 边界抖动——用后面代码里的 margin 指标可以量出来：同一个问题换个说法就换一批 chunk，意味着第 $k$ 名附近的分数差小于查询改写带来的扰动。跨域零样本下稠密检索的优势本就不稳固，BEIR 的口径与结论见 [[rag-02]]。

### 2. HyDE 的三步

1. **生成假设文档。** 用一个 LLM（不必是最终回答用的那个）针对 query 写一段**陈述段落**，而不是回答问题给用户看。prompt 的关键是把体裁钉住，例如 `Write a short passage that answers the question, even if you are unsure. Never repeat or restate the question.`；要显式禁止「复述问题」，否则模型倾向于把 query 换个说法抄一遍，向量又回到查询体裁。典型输出 $T_g$ = 32–128 token，并用 `max_new_tokens` 截断。
2. **编码。** 用**索引侧同一个**文档编码器、同一套归一化与维度编码 $\hat d$，得到 $v_h = E(\hat d)$。换编码器或漏掉归一化会让两侧向量不可比，这是最常见的实现错误。
3. **检索。** 用 $v_h$ 替代 $v_q$，或按权重融合后重新归一化：

$$v = \frac{\alpha v_q + (1-\alpha) v_h}{\lVert \alpha v_q + (1-\alpha) v_h \rVert_2}, \qquad \alpha \in [0, 1]$$

$\alpha = 0$ 是原始形态（完全用假答案）；$\alpha$ 取 0.3–0.5 是常见的稳妥折中，因为它保留了原查询里的实体与约束，也顺带限制了跑题时的损失。融合后的 $v$ 直接喂给原来的 ANN 索引，**索引侧一行都不用改**——这是 HyDE 相对微调 embedder 的最大优势。

### 3. 为什么有效

- **长度与体裁对齐。** $\hat d$ 的 token 数与 chunk 拉近（通常 32–128 对 256–512，不必等长），第 1 节的 $\sigma/\sqrt{n}$ 差距被显著缩小，相关 chunk 的相似度回到噪声更低的水平，margin 变宽。
- **领域词汇注入。** 假答案会自然带出真实答案才会用的词（问「笔记本为什么掉电快」，假答案里出现 `background applications`、`screen brightness`、`battery health`）。这些词不在 query 里，但出现在真实 chunk 里；对稠密检索是新的语义线索，对 hybrid 里的稀疏通道也直接加分。
- **零样本、只改查询侧。** 不需要 relevance label，不需要训练，不动索引，可以按请求开关。相比之下微调 embedder 要标注数据和重跑全量索引，代价高得多。
- **论文口径。** Gao 等（2022）报告 HyDE 在 web search、QA、fact verification 等多任务与多语言上明显优于无监督稠密检索基线 Contriever、接近有监督微调的检索器（HyDE 出处见第 8 节）；但各任务与数据集差异很大，这只说明机制方向成立，不是普适增益的证据。

事实正确性在这里不是必要条件：向量携带的是「这段话在讲哪一片区域」，只要区域对，具体数字错了也不会把检索带偏，而最终答案只由真实 chunk 生成。

### 4. 常见变体

| 变体 | 做法 | 代价与适用 |
| --- | --- | --- |
| 向量融合 | 上文 $\alpha$ 加权，或直接拼接后降维 | 几乎零额外成本，工程默认 |
| 多份假答案 | 采样 $m$ 份 $\hat d^{(j)}$，取 $\frac{1}{m}\sum_j v_h^{(j)}$ 或做 RRF | 生成成本 ×$m$，方差下降，适合离线批量 |
| 检索式改写 | 不调 LLM：先用 query 检索历史问答库的相似**问题**，把对应**答案**当假文档 | 只多一次 ANN，延迟低；依赖问答库覆盖度，冷启动差 |
| 反向 HyDE（doc2query 家族） | 离线把每个 chunk 生成「它可能回答的问题」，索引侧变成伪查询；运行时用真查询直接匹配 | 成本全部挪到离线，在线零开销；索引重建代价高、伪查询质量决定上限 |

反向那一列值得单独记住：它把 HyDE 的「在线付一次生成」换成了「离线付 N 次生成」，是同一个非对称问题的另一种解法。

### 5. 什么时候失效

- **查询本身已经像文档。** 用户粘贴 200 token 的错误日志、一段代码、一封邮件时，$n_q \approx n_d$，非对称消失，收益被摊平，还白付一次生成。
- **精确匹配是刚需。** 型号、错误码、法条编号、人名：生成模型不会凭空产生正确的 token，改写反而可能把它们磨掉。这类查询本来就该由 BM25 兜底（[[rag-02]]），这也是「hybrid 把 HyDE 收益吃掉」的机制。
- **领域漂移（drift）。** 模型对垂直领域不熟时，假答案会落在相邻但不相关的区域，此时它比原查询更糟——这是 HyDE 最典型的「负收益」路径，必须在评估里单独观察。
- **延迟/成本预算紧。** 每次查询多一次生成的链路会直接抬高 TTFT 与成本，对交互式产品和按 token 计费的 API 都不友好（[[inference-serving-09]]、[[inference-serving-12]]）。

### 6. 决策清单

| 场景 | 建议 | 理由 |
| --- | --- | --- |
| 短查询（≤ 5 词）、内部黑话、垂直术语（医疗/法律/工单） | 开 | 非对称最严重，领域词汇注入收益最大 |
| 零样本检索、没有标注数据 | 开 | 不需要训练和重建索引 |
| 离线批处理、评测集构建 | 开 | 延迟不敏感，收益可测 |
| 用户粘贴长文本、日志、代码 | 关 | 非对称已消失，还白付生成 |
| 查询含型号/错误码/编号 | 关或降 $\alpha$ | 交给 BM25 精确匹配，避免改写磨掉 token |
| 已有领域内微调过的 embedder，或 hybrid + reranker 已把 recall 做到位 | 先测，默认关 | 收益被吸收，成本却最高，优先级排在最后 |
| TTFT 敏感（如 < 100 ms） | 关，或退到检索式改写 | 见下一节的延迟账：最快的一档（1.5B、64 token）也要 235 ms |

## 数值与代码验证

### 1. 多出来的那次生成：FLOPs

decode 每个 token 的乘加量约等于全参数量（前向每参数一次乘加），所以

$$F_{\text{hyde}} \approx 2P\,(T_{in} + T_g)$$

口径：$P$ 为总参数量；忽略 attention 的 $T^2$ 项（$T$ 是几十到上百的量级，远小于隐藏维）；$T_{in} = 60$ 为模板加问题的 token 数；$T_g$ 为假设文档长度。

| 生成模型 | $T_g$ | $F_{\text{hyde}}$ |
| --- | --- | --- |
| 1.5B | 64 | $2 \times 1.5\times10^9 \times 124 = 3.72\times10^{11}$（372 GFLOP） |
| 1.5B | 128 | $5.64\times10^{11}$（564 GFLOP） |
| 8B | 64 | $1.98\times10^{12}$（1.98 TFLOP） |
| 8B | 128 | $2 \times 8\times10^9 \times 188 = 3.01\times10^{12}$（3.01 TFLOP） |

### 2. 延迟：batch-1 的 decode 是带宽受限

单个请求生成时，每输出一个 token 必须把全部权重从 HBM 读一遍，因此

$$t_{\text{token}} \gtrsim \frac{2P\ \text{bytes} + \text{KV 读取}}{BW}$$

口径：A100 80GB SXM，HBM2e 带宽 2039 GB/s；「带宽利用率 40%」是 batch-1、无投机解码时的保守取值。KV 项在 batch-1 下很小：Llama-3-8B 每 token 需读 $2 \times 8 \times 128 \times 32 \times 2 = 131$ KB，4k 上下文累计约 537 MB，约为 16 GB 权重的 3.4%，先忽略。

| 生成模型 | 权重 | 带宽上限 token/s | 按 40% | 每 token | 生成 64 | 生成 128 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5B fp16 | 3.0 GB | 680 | 272 | 3.7 ms | 235 ms | 471 ms |
| 8B fp16 | 16.0 GB | 127 | 51 | 19.6 ms | 1.26 s | 2.51 s |
| 8B int8 | 8.0 GB | 255 | 102 | 9.8 ms | 0.63 s | 1.26 s |
| 70B int8 | 70.0 GB | 29 | 12 | 85.8 ms | 5.49 s | 10.99 s |

prefill 那一侧可以忽略：8B 模型 60 token 的 prompt，在 312 TFLOP/s（bf16 dense）的 30% MFU 下约 10 ms。量化会把延迟按字节数近似减半（8B fp16 → int8 是 19.6 ms → 9.8 ms），这也是「用小模型或量化模型生成假文档」这条优化路径的依据。

**量级对照**：常被引用的「几十到几百毫秒」对应表里第一行这一档——1.5B / 64 token 是 235 ms，更小的模型或更短的输出才落到几十毫秒。如果直接用回答用的 8B 模型生成 128 token，增量是 2.5 s 量级，比一次 ANN 检索贵三个数量级，这个差距必须在设计时就摆到桌面上。

### 3. 与检索侧的成本对比

编码器按 110M 参数、8B 生成模型按 $T_g = 128$ 记账；索引按 100 万 chunk、$d = 768$：

| 动作 | FLOPs | 相对一次 query 编码 |
| --- | --- | --- |
| 编码 query（16 token） | $3.52\times10^{9}$ | 1× |
| 编码假设文档（64 token） | $1.41\times10^{10}$ | 4.0× |
| 100 万向量暴力扫描（$d = 768$） | $1.54\times10^{9}$ | 0.44× |
| HyDE 生成（1.5B / 64 token） | $3.72\times10^{11}$ | 106× |
| HyDE 生成（8B / 128 token） | $3.01\times10^{12}$ | 855× |
| 离线建索引（100 万 chunk × 256 token） | $5.63\times10^{16}$ | $1.6\times10^{7}$× |

读法：

- 一次 8B 的 HyDE 生成 ≈ **1958 倍**一次 100 万向量的暴力扫描，≈ 214 倍编码假设文档本身，≈ 855 倍编码原查询。**瓶颈完全在生成侧，检索侧的账可以忽略**；换成 1.5B 小模型生成 64 token，降到 242 倍暴力扫描，这是最划算的一档。
- 反过来看，一次 HyDE 生成只相当于重新编码 53 个 256-token chunk（1.5B 时是 6.6 个）；相比 100 万 chunk 的离线建索引（$5.63\times10^{16}$ FLOP）小 4 个数量级。所以 HyDE 的成本压力在**在线 QPS**，不在离线预算——查询量大的系统才需要认真优化它。
- API 计费侧只给公式，不写死价格：增量成本 $= T_{in}\,p_{in} + T_g\,p_{out}$。取 $T_g = 128$，输入项 $T_{in}\,p_{in}$ 按常见的 $p_{in} \approx 0.1\,p_{out}$ 只占不到 5%，可忽略，于是输出项 $= 1.28\times10^{-4}\,p_{out}$；按公开价目常见的 0.1–1 美元/百万输出 token 区间，单次查询增量约 $1.3\times10^{-5}$–$1.3\times10^{-4}$ 美元。

### 4. 与源文数字的对照

源文给了一组「closeness」示意值：查询与真实 chunk 约 0.55，假答案与真实 chunk 约 0.88（满分 1），并自注只是便于理解的示意。这两个数不能当实测引用：余弦的绝对水平取决于编码器，同一对文本在不同模型上可以差 0.2 以上，源文也没给测量条件。

可迁移的口径是**相对量**：固定编码器与语料，比较「最低相关分数 − 最高不相关分数」这个 separation 与第 $k$、$k+1$ 名的 margin，而不是绝对相似度。下面的脚本把这条口径变成可复算的指标。

### 5. 代码：排序稳健性诊断（可运行）

```python
import numpy as np


def rank_metrics(s, rel_idx, k=10):
    """s = doc_vecs @ q_vec（向量已 L2 归一化，内积即余弦）。"""
    order = np.argsort(-s)
    hit = np.zeros(len(s), dtype=bool); hit[list(rel_idx)] = True
    recall = hit[order[:k]].sum() / max(hit.sum(), 1)          # recall@k
    sep = float(s[hit].min() - s[~hit].max())                  # 最低相关 − 最高不相关
    margin = float(s[order[k - 1]] - s[order[k]])              # 第 k 名与第 k+1 名之差
    return order[:k], float(recall), sep, margin


def paraphrase_stability(query_vecs, doc_vecs, k=20):
    """同一 query 的多个改写各自检索，返回 top-k 的成对 Jaccard 均值。
    越低说明查询侧排序越不稳，改写类方法（HyDE / multi-query）的收益空间越大。"""
    tops = [set(np.argsort(-(doc_vecs @ v))[:k].tolist()) for v in query_vecs]
    js = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    return float(np.mean(js)) if js else float("nan")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d, n_rel, n_bg, k = 768, 5, 20_000, 10
    u = rng.normal(size=d)                                     # 共享"内容方向"，逐维标准差 1
    docs = np.vstack([u + 0.5 * rng.normal(size=(n_rel, d)),   # 相关 chunk：噪声小
                      rng.normal(size=(n_bg, d))])             # 背景 chunk：与 u 无关
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)

    def query(eps):                                            # eps 越大 = 查询侧噪声越大
        v = u + eps * rng.normal(size=d)
        return v / np.linalg.norm(v)

    for name, eps in (("短查询（噪声大）", 1.2), ("假设文档（噪声小）", 0.3)):
        v = query(eps)
        _, rec, sep, mar = rank_metrics(docs @ v, range(n_rel), k=k)
        stab = paraphrase_stability([query(eps) for _ in range(4)], docs, k=20)
        print(f"{name:<20} eps={eps:<4} cos(u,v)={float(u @ v / np.linalg.norm(u)):.2f} recall@{k}={rec:.2f} sep={sep:+.3f} margin={mar:.4f} Jaccard@20={stab:.2f}")
```

在这组合成向量上实跑（`numpy` 2.x，`default_rng(0)`）：查询侧 `eps=1.2` 得 `cos(u,v)=0.64`、`recall@10=1.00`、`sep=+0.402`、`margin=0.0014`、`Jaccard@20=0.15`；假设文档侧 `eps=0.3` 得 `cos(u,v)=0.96`、`recall@10=1.00`、`sep=+0.709`、`margin=0.0019`、`Jaccard@20=0.38`。三点读法，也正是把这段代码放进评估脚本的理由：

- **recall@10 在两行里都是 1.00，完全不敏感**——背景 chunk 与内容方向无关时，5 个相关 chunk 只要进了候选就都被召回。判断「要不要上 HyDE」不能只看 recall@k，要看 separation、nDCG 与边界稳定性，这是评测设计上的常见坑。
- separation 从 +0.402 涨到 +0.709：查询侧噪声变小，相关与不相关的分界被拉开，reranker 的输入质量也随之变好（[[rag-03]]）。
- Jaccard@20 从 0.15 涨到 0.38：改写后 top-20 的重合度上升，排序不再被措辞支配。一条可用的启发式（阈值需在自有语料上标定）是——同一 query 的 3–5 个改写之间 Jaccard@20 低于 0.6 说明查询侧不稳、改写类方法有空间；接近 0.9 则 HyDE 大概率只是白花一次生成。

这是一组明确标注的合成向量实验，只用来演示指标口径与单调性，不能当作真实收益证据——收益必须在自己的语料上 A/B。

### 6. 代码：HyDE 检索最小实现（可运行）

```python
import torch
import torch.nn.functional as F

PROMPT = ("Write a short passage that answers the question, even if you are unsure. "
          "Never repeat or restate the question.\nQuestion: {q}\nPassage:")


class HydeRetriever:
    """HyDE 最小实现：生成假设文档 -> 编码 -> 与原查询向量融合 -> 检索。"""

    def __init__(self, encode, generate, index_vecs, *, max_new_tokens=96, blend=0.5):
        self.encode = encode                  # List[str] -> Tensor[B, d]，已 L2 归一化
        self.generate = generate              # (prompt, max_new_tokens) -> str
        self.index_vecs = index_vecs          # Tensor[N, d]，已 L2 归一化
        self.max_new_tokens, self.blend = max_new_tokens, blend

    @torch.no_grad()
    def __call__(self, query, top_k=20):
        hypo = self.generate(PROMPT.format(q=query), self.max_new_tokens).strip()
        q_vec = self.encode([query])
        if len(hypo) < 16:                    # 护栏：生成失败/过短则退回原查询
            vec, used_hyde = q_vec, False
        else:
            h_vec = self.encode([hypo])
            vec = F.normalize(self.blend * q_vec + (1 - self.blend) * h_vec, dim=-1)
            used_hyde = True                  # blend=0 即论文的原始形态
        scores = self.index_vecs @ vec.squeeze(0)
        top = torch.topk(scores, top_k)
        return top.indices.tolist(), top.values.tolist(), hypo, used_hyde


if __name__ == "__main__":
    torch.manual_seed(0)

    def encode(texts):                        # 桩：真实场景换成索引侧同一个编码器
        return F.normalize(torch.randn(len(texts), 768), dim=-1)

    def generate(prompt, max_new_tokens):     # 桩：真实场景换成 1–8B 的生成模型
        return "Background apps, high screen brightness and old battery health reduce battery life."

    index = F.normalize(torch.randn(10_000, 768), dim=-1)
    ids, sims, hypo, used = HydeRetriever(encode, generate, index)("why is my laptop battery draining fast?")
    print("used_hyde =", used, "| top-5 ids =", ids[:5], "| top-5 cos =", [round(s, 3) for s in sims[:5]])
    ids2, _, _, used2 = HydeRetriever(encode, lambda p, n: "I don't know.", index)("q")
    print("fallback  =", used2 is False, "| top-1 =", ids2[0])
```

这段代码用桩函数跑通（`torch` 2.x），验证的是控制流而不是检索质量：`used_hyde=True` 时走融合路径，第二组输入生成结果过短，`fallback=True` 且 top-1 由原查询给出。生产里要补三件事：`max_new_tokens` 与 `blend` 配置化、生成耗时与降级次数打点、按 query 哈希缓存假设文档（同一 query 重复问时复用，能压低平均成本）。

## 常见追问

- **追问**：HyDE 与 multi-query / RAG-Fusion 的区别？
  - 要点：multi-query 生成 $m$ 个**查询**，各自检索后用 RRF（$\text{score}(d)=\sum_r 1/(k+\text{rank}_r(d))$，$k$ 常取 60）融合，目的是扩大召回覆盖、对冲单次改写失败；HyDE 生成 1 个**文档体裁的伪答案**，改的是查询侧的分布（长度、体裁、词汇），思路是把查询搬到文档流形上。两者正交互补，可以叠成「$m$ 份假答案」的形态，但代价都是 $m$ 倍生成。
- **追问**：它依赖生成质量吗？假答案写错会不会更糟？
  - 要点：依赖的是体裁与领域词汇，不是事实正确性；细节错通常无害，因为最终上下文只来自真实 chunk。风险是跑题（领域漂移）。廉价护栏有三个：生成结果过短（如 < 16 字符）、与查询的字符 n-gram 重合度过高（说明在复述问题）、与语料采样的相似度低于某个地板值时，降级回原查询，并把这个事件打点。
- **追问**：能不能不调 LLM，用现成的答案库替代生成？
  - 要点：可以，这就是检索式改写。维护历史问答对（或用 FAQ、工单库），先用 query 检索最相似的若干**问题**，把对应**答案**拼起来当假文档再编码检索：多一次 ANN，没有 LLM 延迟，也没有生成漂移。代价是依赖问答库对当前分布的覆盖度，冷启动阶段没有等价物，效果通常不如 LLM 生成的假答案。
- **追问**：为什么它对长查询没帮助？
  - 要点：收益来源是非对称。查询已经是 200 token 的日志或邮件时 $n_q \approx n_d$，噪声差距消失；而且长查询里往往含精确 token（堆栈、错误码、版本号），生成器在改写成段落时会把这些精确串磨掉，反而伤检索——这正是该交给 BM25 的场景（[[rag-02]]）。
- **追问**：假设文档写多长合适？
  - 要点：向索引侧 chunk 的长度靠拢但不必等长，实践上取 chunk 长度的 1/8–1/4（chunk 256–512 token 时取 32–128 token）即可。太短会退回「像查询」；太长则双输：生成成本线性上升，且无关 token 会把内容分量的占比稀释（$\sigma/\sqrt{n}$ 描述的是对同一内容分量的估计误差，不是让无关内容自动消失）。instruction 里必须显式禁止复述问题。
- **追问**：HyDE、hybrid、reranker 三者的优先级怎么排？
  - 要点：先加 reranker（[[rag-03]]）——它作用在已经召回的候选上，收益最稳、延迟可控、不需要改召回；再用 hybrid（[[rag-02]]）解决精确匹配与专有名词；HyDE 放在召回侧、收益最不确定，应该最后加且做成开关。三者不冲突，但要合并进同一份延迟预算：HyDE 的生成和 reranker 的前向都加在 TTFT 之前（[[inference-serving-09]]）。

## 相关题目

- [[rag-01]]：chunk 长度决定了索引侧的 $n_d$，也决定了 HyDE 的假答案该写多长。
- [[rag-02]]：稀疏与稠密的取舍；HyDE 的收益被 BM25 吃掉，就发生在这条边界上。
- [[rag-03]]：reranker 与 HyDE 解决的是链路上不同位置的问题，优先级也更高。
- [[rag-06]]：agentic RAG 会把改写、检索、判断是否需要再检索变成多轮决策，HyDE 可以退化成其中一步。
- [[inference-serving-09]]：多一次生成如何计入 TTFT / TPOT 预算。
- [[inference-serving-10]]：batch-1 的 roofline 账，与上面那张延迟表是同一套算法。

## 参考资料与归属

- How does HyDE work in RAG? — Amit Shekhar（Outcome School），2026-07-06，[原文链接](https://outcomeschool.com/blog/how-does-hyde-work)

源文覆盖了动机（问题和答案「形状」不同）、分步流程、优缺点与使用建议，以及那组自注为示意的 closeness（0.55 / 0.88）数字。本页的 FLOPs、延迟与成本数字**不来自源文**，是按 $F \approx 2P(T_{in}+T_g)$ 与 batch-1 带宽受限模型自行复算的，假设（$T_{in}=60$、$T_g=64/128$、A100 2039 GB/s、40% 带宽利用率、110M 编码器）已随表给出；合成向量实验的代码与输出也都在正文中。HyDE 的出处是 2022 年面向零样本稠密检索的论文（Luyu Gao、Xueguang Ma、Jimmy Lin、Jamie Callan，Precise Zero-Shot Dense Retrieval without Relevance Labels，arXiv:2212.10496）：第 3 节「明显优于无监督稠密检索基线 Contriever、接近有监督微调的检索器」这一口径来自该文摘要（多任务、多语言、相对无监督基线比较），具体数据集上的数字不在本页复现——来源清单里没有论文原文，因此只做定性描述。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
