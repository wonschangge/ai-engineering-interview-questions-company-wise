---
type: question
id: rag-04
topic: RAG 与检索
order: 4
question: 你会如何评估 RAG 流水线的质量：分别评估 retrieval 和 generation？
question_en: How would you evaluate a RAG pipeline: retrieval and generation separately?
asked_at: [Cohere]
level: 进阶
tags: [评估, ragas, retrieval-metrics, llm-as-a-judge]
sources:
  - title: LLM Evaluation
    url: https://outcomeschool.com/blog/llm-evaluation
    author: Amit Shekhar (Outcome School)
    published: 2026-05-24
  - title: Ragas: Automated Evaluation of Retrieval Augmented Generation（延伸）
    url: https://arxiv.org/abs/2309.15217
    author: Es, James, Espinosa-Anke, Schockaert
    published: 2023-09-26
  - title: BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models（延伸）
    url: https://arxiv.org/abs/2104.08663
    author: Thakur, Reimers, Rücklé, Srivastava, Gurevych (NeurIPS 2021)
    published: 2021-04-17
related: [rag-02, rag-03, rag-12, llm-internals-11]
updated: 2026-09-28
---

## 一句话答案

> 端到端分数只能说明「坏了」，分层指标才能说明「坏在哪」，所以评估要拆成三层。检索层用 recall@k、precision@k、MRR、nDCG@k，对着人工或 LLM 标注的相关性集合打分；生成层用 faithfulness（论断被 context 支持的比例）、answer relevance、context precision/recall，可以做成 reference-free 的 LLM judge；端到端层用真实问题集上的答案正确率、拒答率、引用正确率，再加线上 A/B 与成本延迟。最硬的约束是 recall@k 构成端到端正确率的上界——检索没召回的东西，生成侧再怎么调 prompt 也拿不回来。

## 面试官在考什么

- 会不会分层拆解：能把「RAG 质量」拆成检索层、生成层、端到端层，并说清每层的数据集从哪来、口径是什么，而不是报一个总分。
- 指标定义是否精确：recall@k 的分母是「语料里的相关 chunk 总数」还是「检索结果里的相关数」、nDCG 用指数增益还是线性增益、MRR 与 nDCG 的差别在哪。口径说不清，数字就没有意义。
- 评测集的构造能力：gold chunk 与 gold answer 怎么标、谁来标、无答案问题怎么覆盖、怎么防止把文档原句抄成问题造成污染。
- LLM-as-judge 的工程素养：知不知道 judge 的位置偏差、冗长偏差、自偏好，以及用什么可验证的方式校准它（与人工的一致率、Cohen's kappa）。
- 实验设计素养：评测集要多大才能检出想要的变化、同一份评测集上做 A/B 应该用配对检验、离线指标能不能被在线指标验证。

**常见错误答案**

- 「接上 RAGAS/DeepEval 跑一遍就有分了。」框架只提供指标实现，不解决口径问题：谁来定义相关性、分母包含什么、judge 用哪个模型，这些决定了数字能不能用。
- 「端到端正确率 82%，可以上线了。」这个数字无法区分「检索漏了」「召回了但被噪声淹没」「召回了模型没用上」三种故障，也就无法决定下一步该优化谁。
- 「faithfulness 就是正确率。」faithfulness 衡量答案是否被给定 context 支持，context 本身错了、答案照样可以很 faithful。两者必须分开报。

## 原理与推导

### 分层评估要解决的是定位问题

同一个「答案错了」的背后至少有四类互斥原因，它们分别落在不同的层，只有分层指标才能把它们分开：

| 观察到的症状 | 故障层 | 该看的指标 |
| --- | --- | --- |
| 答案完全没提到该事实 | 检索召回 | recall@k、hit rate |
| 答案里混进了别的文档的条款 | 检索排序、上下文组装 | context precision、nDCG@k |
| 片段就在 prompt 里，模型却没用 | 生成、上下文组装 | faithfulness、位置扫描（见 [[llm-internals-13]]） |
| 用对了片段，数字抄错或推理错 | 生成 | answer correctness、引用正确率 |
| 该说「不知道」时硬答 | 拒答策略、检索阈值 | 无答案子集上的正确拒答率 |

分层还带来一个可执行的验收方式：检索层的改动（换 embedding、加 reranker、改 chunk）跑检索指标，生成层的改动（换模型、改 prompt）跑生成指标，两边的回归互不干扰。

### 检索层：常用指标与它们的口径

记 query 为 $q$，人工标注的相关 chunk 集合为 $\mathrm{Rel}(q)$，检索器返回的排序列表为 $\mathrm{Top}_k(q)$。

$$
\mathrm{recall@}k = \frac{|\mathrm{Rel}(q) \cap \mathrm{Top}_k(q)|}{|\mathrm{Rel}(q)|}
\qquad
\mathrm{precision@}k = \frac{|\mathrm{Rel}(q) \cap \mathrm{Top}_k(q)|}{k}
$$

$$
\mathrm{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\mathrm{rank}_q}
\qquad
\mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k},
\quad \mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2 (i+1)}
$$

另有一个常用的 hit rate（hit@k）：top-k 里只要存在至少一条相关 chunk 就记 1，否则记 0，再对 query 集取平均。它是 recall@k 的二值弱化版，只看「有没有命中」，不反映召回了几条，也不反映排在第几位。

口径上有三个坑：第一，recall@k 的分母是「语料里与该 query 相关的 chunk 总数」，这个数实践中拿不到，只能用标注集合近似——按池化（pooling）方式用多个检索器各取 top-k 的并集定候选，未判定的按不相关处理。这个约定对检索结果里出现未判定项的系统是保守的：分子被少数几条，recall@k 与 precision@k 读出来都偏低（precision 的分母固定为 $k$，压低幅度取决于 top-k 里有几条其实相关却没被判定）；但标注集也可能漏掉语料里从未被任何系统检索到的相关 chunk，那会让 recall 的分母偏小、读数偏高，所以「下界」只在召回结果都落在判定池内时严格成立。无论哪种情况，跨系统比较都必须用同一份标注集。第二，nDCG 的增益函数有两种常见口径，$2^{rel}-1$ 会放大高等级文档的权重，$rel$ 是线性口径，两者不可混用。第三，MRR 只看第一个相关结果，多跳问题、需要多个片段才能回答的问题上它几乎不敏感，别拿它当主指标。

**为什么 recall@k 是 RAG 的第一指标**：只有当回答问题所需的片段进了 top-k，后面所有环节才有机会做对。记 $g$ 为「不依赖 gold chunk 也能蒙对」的概率，则

$$
\mathrm{Acc}_{\mathrm{e2e}} \le \mathrm{recall@}k + \bigl(1 - \mathrm{recall@}k\bigr) \cdot g
$$

在封闭域（企业文档、私有语料）里 $g \approx 0$，上界就退化成 $\mathrm{Acc}_{\mathrm{e2e}} \le \mathrm{recall@}k$。这条不等式的实际用法是：先量 recall@k，再量端到端正确率，两者之差就是生成侧和上下文组装侧的损失空间；如果 recall@10 只有 0.72，那么任何 prompt 工程的收益都被压在 0.72 以下。

### 相关性集合从哪来

- **人工分级标注**：最可信，也最贵。适合 100–500 条量级的固定回归集，标注口径要写成文档（什么算「相关」、分级怎么定）。
- **从 gold answer 反推**：先写标准答案，再让人（或让模型给候选、人确认）找出支撑它的 chunk。适合已经有客服工单、专家问答这类「答案先于检索」的场景。
- **LLM 蒸馏**：从 chunk 反向生成它能回答的问题，得到 (question, gold chunk) 对。便宜、可批量，但只能覆盖「单跳、答案就在这一段里」的形态，且必须做改写，避免把文档原句直接当问题——那等于把检索题出成了字符串匹配题。
- **公开数据集的方法论参照**：BEIR 用 18 个公开数据集、10 个检索系统做跨域零样本评测，结论是 BM25 是稳健基线，rerank 与 late-interaction 平均最好但计算代价高，稠密与稀疏模型更省算力却经常打不过它们，泛化能力仍有明显空间。它对 RAG 评估的启示不是抄分数，而是方法论：单语料上的提升要在多域上复验，只看自家语料就会把域内过拟合当成能力。

### 生成层：reference-free 三指标与 reference-based 补充

RAGAS 的思路是在没有标准答案的情况下，用 LLM 把「答案好不好」拆成三个可计算的代理量：

$$
F = \frac{|V|}{|S|}
\qquad
\mathrm{AR} = \frac{1}{n}\sum_{i=1}^{n} \mathrm{sim}(q, q_i)
\qquad
\mathrm{CR} = \frac{\text{被抽取的句子数}}{\text{context 的总句子数}}
$$

- **faithfulness $F$**：先把答案拆成一组互相独立的论断 $S$（分母），再逐条判断能否由 context 推出，被支持的记为 $V$（分子）。两步都是 LLM 调用，因此 $F$ 的数值同时受论断拆分粒度与 judge 判定的影响。
- **answer relevance $\mathrm{AR}$**：让模型根据答案反向生成 $n$ 个问题 $q_i$，再算它们与原始问题的 embedding 余弦相似度均值。答案不完整、含大量冗余时相似度会被拉低。它不看事实性，只看是否真的回答了问题。
- **context relevance $\mathrm{CR}$**：让模型从 context 里抽出回答问题所必需的句子，用比例惩罚「塞了一堆无关内容」。论文里作者自己指出这是最难的一项，context 较长时模型抽句子的表现明显更差。

这三个指标的关键假设是「judge 模型可以替代人做这些判断」。偏差要提前说清：judge 有位置偏好、冗长偏好、对自家模型输出的偏好；$F$ 是比例，同一个错误在 3 条论断的短答案里扣掉 1/3，在 30 条论断的长答案里只扣 1/30，所以长短答案的 $F$ 不可直接横向比较；论断拆分本身出错会同时污染分子和分母。这也是为什么 $F$ 只能当趋势指标，不能当发布门禁的绝对阈值。

论文在自建的 WikiEval（50 个维基页面、两名标注者，faithfulness 与 context relevance 的标注者一致率约 95%、answer relevance 约 90%）上给出了三个指标与人工成对判断的一致率：0.95、0.78、0.70；作为对照，让同一个模型直接打 0 到 10 分只有 0.72、0.52、0.63，让它从两个候选里选一个只有 0.54、0.40、0.52。faithfulness 最可信、context relevance 最难，这也是「judge 必须逐项校准」的实证依据。

有标准答案时补 reference-based 指标：answer correctness（EM、token F1，或用 LLM 做等价性判定）；检索侧的 context precision / context recall 有两种常见口径，一种把经典 precision/recall 直接套在检索结果上（召回的 chunk 里多少真正相关、语料里的相关 chunk 召回了多少），另一种站在答案侧——把标准答案拆成句子，看有多少句能被检索到的 context 支持。后一种不需要穷举语料里的全部相关 chunk，代价是只能覆盖标准答案提到的那部分证据，与经典 IR 的 recall 不是同一个量。

### 端到端与业务层

- **答案正确率**：优先用成对比较（同一个问题给 A/B 两版，选更好的那个）而不是让 judge 打 1 到 5 分：成对比较更容易与人工判断对齐，也更容易算显著性（配对检验）。但有两个前提要说清：交换 A/B 顺序各判一次以消除位置偏差；以及「排名式 judge 一定优于打分」并不成立——WikiEval 上让模型从两个候选里排序的一致率（0.54、0.40、0.52）反而低于直接打分（0.72、0.52、0.63），所以 judge 的形式要先校准再定。
- **拒答与无答案**：单独出一个「语料里没有答案」的子集，量正确拒答率与误拒率。RAG 系统最常见的线上投诉是「有资料时答得好，没资料时编得像真的」。
- **归因正确率**：答案里的每条论断能否指到真正支撑它的 retrieved span，是另一条独立的轴，见 [[rag-12]]。
- **在线指标**：会话成功率、追问率、点踩率、人工升级率，加上 TTFT、总延迟与每会话 token 成本。离线指标必须能被在线指标验证，否则会优化出一套自嗨的分数：离线涨、线上不动，说明测的东西不是用户在乎的东西。

### 评测流程

1. 建 100–500 条真实问题集，显式覆盖单跳、多跳、表格/图表、无答案、需要权限过滤这几类。
2. 每条标注 gold chunk（供检索层评测）与 gold answer（供生成层评测），并记录它属于哪一类。
3. 固定 prompt 模板与解码参数，每次改动跑全量，指标与上一次对比，形成版本化回归。
4. LLM judge 要校准：定期抽 50–100 条让人工判一遍，报一致率与 Cohen's kappa，judge 换模型或换 prompt 后重新校准。
5. 防污染：问题要改写，不要用文档原句；新文档入库后检查是否与评测集重复。

## 数值与代码验证

### 复算一：四个检索指标在一份排序上的取值

设某个 query 在语料里共有 5 条相关 chunk，等级口径为 2（单独就能回答）、1（有用但不充分）、0（无关），相关集合的等级是 $\{2,2,1,1,1\}$。两个检索器返回同一个 6 条候选集，只是顺序不同：

| 指标（口径） | 排序 A：噪声排在前面 | 排序 B：精排之后 |
| --- | --- | --- |
| recall@1 | 0/5 = 0.000 | 1/5 = 0.200 |
| recall@3 | 1/5 = 0.200 | 3/5 = 0.600 |
| recall@6 | 4/5 = 0.800 | 4/5 = 0.800 |
| precision@3 | 1/3 = 0.333 | 3/3 = 1.000 |
| MRR | 0.500 | 1.000 |
| nDCG@6（指数增益 $2^{rel}-1$） | 0.5539 | 0.9257 |
| nDCG@6（线性增益，对照口径） | 0.5726 | 0.8993 |

这张表里最重要的一行是 recall@6：两个排序完全一样，因为候选集相同。reranker 只重排固定候选池，动的是 MRR 与 nDCG；recall@k 则要看 $k$ 取多大——只有 $k$ 不小于候选池深度时它才与排序无关（上表的 recall@6），$k$ 更小时重排同样会改变 recall@k（上表的 recall@1、recall@3）。这正是「检索指标涨了端到端没涨」这个追问的算术来源（见 [[rag-03]]）。顺带说明口径差异：同一组数据换成线性增益，nDCG@6 从 0.5539 变成 0.5726，两个数都不能说错，但跨系统比较时必须统一。

```python
import math

CORPUS = [2, 2, 1, 1, 1]        # 语料里相关的 5 条 chunk 及其等级
RANK_A = [0, 1, 0, 2, 2, 1]     # 未精排：噪声排在最前
RANK_B = [2, 2, 1, 0, 0, 1]     # 精排后：同一个候选集，顺序更好

def report(name, rels, corpus=CORPUS, k=6):
    hits = [1 if r > 0 else 0 for r in rels]
    rr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    gains = [2 ** r - 1 for r in rels[:k]]
    ideal = [2 ** r - 1 for r in sorted(corpus, reverse=True)[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    lin = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    lin_ideal = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(corpus, reverse=True)[:k]))
    print(f"{name}: recall@1={sum(hits[:1])}/{len(corpus)} recall@3={sum(hits[:3])}/{len(corpus)}"
          f" recall@6={sum(hits[:6])}/{len(corpus)} precision@3={sum(hits[:3])}/3"
          f" MRR={rr:.3f} nDCG@6={dcg / idcg:.4f} nDCG@6(线性)={lin / lin_ideal:.4f}")

report("排序 A", RANK_A)
report("排序 B", RANK_B)
```

输出与上表一致：排序 A 为 `recall@1=0/5 recall@3=1/5 recall@6=4/5 precision@3=1/3 MRR=0.500 nDCG@6=0.5539 nDCG@6(线性)=0.5726`，排序 B 为 `recall@1=1/5 recall@3=3/5 recall@6=4/5 precision@3=3/3 MRR=1.000 nDCG@6=0.9257 nDCG@6(线性)=0.8993`。

### 复算二：上界与两因子分解

若 recall@10 测得 0.72、生成侧在「gold chunk 已召回」条件下的正确率是 0.90，则两因子分解给出

$$
\mathrm{Acc}_{\mathrm{e2e}} \approx 0.72 \times 0.90 = 0.648
$$

如果实测端到端正确率是 0.60，差值 0.048 就落在生成侧（相对损失 $0.048 / 0.648 \approx 7.4\%$）。这个分解假设「没有 gold chunk 时模型蒙不对」且「一个问题只需一个片段」，多跳、多片段的问题上它只是近似，但足以决定优化方向：先把 recall@10 从 0.72 往上推，还是先修生成。

### 复算三：评测集要多大

观测到的正确率本身有抽样误差。$N$ 条样本、观测正确率 $\hat p$ 时，95% Wilson 区间大致为：

| $N$ | 观测正确率 | 95% 置信区间 | 半宽 |
| --- | --- | --- | --- |
| 100 | 0.80 | [0.711, 0.867] | ±7.8 个点 |
| 200 | 0.80 | [0.739, 0.850] | ±5.5 个点 |
| 500 | 0.80 | [0.763, 0.833] | ±3.5 个点 |

按未配对的双比例检验、80% 功效、基线 0.80 计算（$n = 2(z_{\alpha/2}+z_{\beta})^2 p(1-p)/\delta^2$，方差项取基线 $p = 0.80$ 这个偏保守的口径），要检出 5 个点的提升每组需要约 1005 条，检出 2 个点每组需要约 6279 条；若第一项方差改用两组合并比例 $\bar p = (p_1+p_2)/2$，同一检验给出约 906 条与 6039 条，结论不变。所以 200 条的评测集做不到「精确排名」，只能抓大改动。

但**同一份评测集上比较两个版本是配对实验**，正确的检验是 McNemar：只看两版判定不一致的样本对 $(b, c)$，$b$ 是新版对旧版错，$c$ 反之。$N = 200$ 时：

| $b$ | $c$ | 净提升 | 双侧精确 $p$ 值 |
| --- | --- | --- | --- |
| 15 | 5 | +5.0 个点 | 0.041 |
| 20 | 10 | +5.0 个点 | 0.099 |
| 30 | 20 | +5.0 个点 | 0.203 |

净提升都是 5 个点，显著性差了 5 倍：决定功效的是不一致对的**不对称程度**，不是样本总量。所以报告里要同时给 $b$、$c$ 和净差，只写「涨了 5 个点」等于没报告。

### 复算四：judge 校准与评测成本

抽检 100 条，人工与 judge 各给一个二值判定，按 2×2 混淆矩阵算（口径：$p_o$ 为一致率，$p_e$ 为按两者边际分布算出的偶然一致率）：

| 一致率 $p_o$ | 偶然一致 $p_e$ | Cohen's $\kappa$ |
| --- | --- | --- |
| 0.90 | 0.545 | 0.780 |
| 0.75 | 0.561 | 0.430 |

只看一致率会骗人：0.75 的一致率在偶然一致率 $p_e = 0.561$ 时，扣掉偶然一致后只剩 0.43。表里两行的边际分布分别约为 65/35 与 67/33，$p_e$ 就是按各自的边际算出来的；边际越偏斜，偶然一致率越高，一致率本身就越不可比。校准口径写进流程，judge 换代时才有一致的判据。

成本也要算进评估预算。200 条样本、每条按 RAGAS 式流程跑一遍（拆论断 1 次 + 逐条验证 1 次 + 反向生成问题 1 次 + context 抽句 1 次；答案相关性取 $n = 3$，就是 3 次 embedding）：约 $200 \times 4 = 800$ 次 LLM 调用加 600 次 embedding 调用。这个量级决定了回归集不能无限扩，也解释了为什么工程上会把逐条论断验证合并成一次批量判定。

### 复算五：faithfulness 是比例，分母决定它有多敏感

记论断总数为 $N_S$、被支持数为 $N_V$：

| 论断数 $N_S$ | 被支持 $N_V$ | $F$ | 漏判 1 条论断的代价 |
| --- | --- | --- | --- |
| 2 | 1 | 0.500 | 0.500 |
| 3 | 2 | 0.667 | 0.333 |
| 6 | 5 | 0.833 | 0.167 |
| 12 | 9 | 0.750 | 0.083 |
| 30 | 27 | 0.900 | 0.033 |

同一个错误在短答案里放大、在长答案里稀释，所以「答案变详细了」这件事本身就会推动 $F$ 变化，与质量无关。工程上的对策是固定口径：固定拆分 prompt、固定每个答案的论断数上限、报告时同时给出平均论断数。

```python
"""Ragas 式 faithfulness 的最小骨架：拆论断 -> 逐条验证 -> 取比例。"""
def faithfulness(answer, context, llm, max_claims=20):
    claims = llm(
        "把下面的答案拆成一条条独立论断，每行一条，不要改写含义。\n"
        f"答案：{answer}"
    ).strip().splitlines()
    claims = [c.strip("-*0123456789. ") for c in claims if c.strip()][:max_claims]
    if not claims:
        return 0.0, []
    # 一次批量判定，比逐条调用省 max_claims-1 次往返
    verdicts = llm(
        "对每条论断判断能否由给定 context 推出，逐行只输出 yes 或 no。\n"
        f"context：{context}\n"
        + "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
    ).strip().splitlines()
    flags = [v.strip().lower().startswith("y") for v in verdicts]
    flags = (flags + [False] * len(claims))[:len(claims)]
    return sum(flags) / len(claims), list(zip(claims, flags))
```

`flags` 的补齐与截断是刻意的：judge 少输出一行或格式跑偏时，缺的按「不支持」计入，宁可低估也不高估。这类失败模式在生产里很常见，指标实现必须对 judge 的不稳定输出有明确策略。

## 常见追问

- **追问**：LLM-as-a-judge 有哪些已知偏差，怎么缓解？
  - 要点：位置偏差（成对比较时偏爱第一个或第二个）、冗长偏差（偏好更长的答案）、自偏好（偏好同族模型的输出）。缓解手段是成对比较时交换顺序各判一次取一致结果、把长度与格式作为控制变量、用与候选模型不同族的 judge、以及定期用人工抽检算一致率与 kappa 校准。judge 的分数是代理量，校准曲线是它的使用前提。
- **追问**：完全没有标注数据怎么办？
  - 要点：先做合成：从 chunk 反向生成问题，得到 (question, gold chunk) 对，能直接支撑检索层指标；再从「检索到的 context + 问题」生成参考答案，作为生成层的弱标签。风险是循环论证——用同一个模型既造题又判分，分数会自我一致地偏高。做法是造题、判分、被评模型尽量用不同模型，并保留一小份纯人工标注集作为锚点。
- **追问**：检索指标涨了，端到端没涨，怎么定位？
  - 要点：按复算一的结论，先确认涨的是哪一类指标。如果只有 nDCG/MRR 涨而 recall@k 没动，收益在排序；排序指标涨但端到端不动，问题通常在上下文组装或生成：片段被塞进中段而失效（见 [[llm-internals-13]]）、上下文超出预算被截断、prompt 没要求「只依据给定文档作答」。把 faithfulness 与答案正确率一起报，就能把「没用上」和「用错了」分开。
- **追问**：faithfulness 很高，答案还是错的，怎么解释？
  - 要点：faithfulness 只衡量答案与 context 的一致性，不衡量 context 与事实的一致性。语料本身过期、chunk 切分把条件句和结论句切开（见 [[rag-01]]）、召回的片段只覆盖了政策的一个例外分支，都会产生「很 faithful 的错答案」。所以 faithfulness 必须与 answer correctness、context recall 同时看，三者一起才能定位是语料问题、切分问题还是生成问题。
- **追问**：怎么避免「只在自家语料上自嗨」？
  - 要点：BEIR 的做法值得抄：准备多个互相独立的领域子集，分别报指标再报跨域平均与最差子集；同时保留一个公开数据集或跨域子集做外部参照。只报单一语料的平均值，会把域内过拟合读成能力提升。
- **追问**：线上怎么持续评估？
  - 要点：离线回归集负责「改动前后是否退步」，在线负责「用户是否真的更好」。线上常见口径是会话成功率、追问率、点踩率、人工升级率、以及 TTFT 与每会话 token 成本；灰度 A/B 要按用户或会话分流而不是按请求，避免同一用户在不同版本间抖动。离线指标涨、线上不动，就要回头质疑评测集与真实流量分布是否一致。

## 公司变体

- **Cohere**：落点偏工程实现。公开信息显示这家公司把 embedding 与 rerank 做成独立可调用的产品线，Chat 接口把 citations 作为一等特性；客户形态以企业私有语料与私有部署为主，公开 benchmark 无法替代自建评测集。因此这题在他们那里通常不会停在「推导 nDCG 公式」，而会往系统侧推：在企业语料上怎么用尽量少的人工标注造出可回归的评测集；rerank 带来的 nDCG@k 收益怎么换算成端到端正确率与 citation 质量；没有标准答案时 faithfulness、answer relevance 这类 reference-free 指标怎么校准到可信（与人工的一致率、judge 偏差）；以及把重排与 judge 的额外调用量算进延迟与成本预算。数学侧的追问（nDCG 增益口径、recall@k 的分母、样本量与显著性）也会出现，但落点仍是「这套评测能不能支撑上线决策」。具体面试流程不在公开信息范围内，这里只描述能力落点。

## 相关题目

- [[rag-02]]：稀疏检索器与稠密检索器的选择。选型实验的判据就是本题的检索层指标，两篇的口径要对齐（同一个标注集、同一个 $k$）。
- [[rag-03]]：reranker 与 cross-encoder 的开销。复算一说明重排只改排序质量指标，recall@k 仅在 $k$ 不小于候选池深度时才不受影响，正好解释「加了 rerank 端到端却不动」。
- [[rag-12]]：把答案的每条论断归因到 retrieved span。那是生成层里独立的一条评估轴，与 faithfulness 互补。
- [[llm-internals-11]]：pre-training、SFT 与 preference optimisation 的区别。偏好优化用成对偏好数据训练 reward model，与 LLM-as-judge 的成对比较共用同一套方法论：怎么标、怎么控制偏差、怎么验一致性。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《LLM Evaluation》，2026-05-24，<https://outcomeschool.com/blog/llm-evaluation>（三层评估的框架、RAGAS 四指标范式、LLM-as-judge 的偏差清单与最佳实践来自此文）
- Shahul Es、Jithin James、Luis Espinosa-Anke、Steven Schockaert，《Ragas: Automated Evaluation of Retrieval Augmented Generation》，2023-09-26（v2 修订于 2025-04-28），<https://arxiv.org/abs/2309.15217>（延伸来源：faithfulness、answer relevance、context relevance 的定义与公式，WikiEval 的标注者一致率与三档方法的人工一致率对照均来自此文）
- Nandan Thakur、Nils Reimers、Andreas Rücklé、Abhishek Srivastava、Iryna Gurevych，《BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models》（NeurIPS 2021），2021-04-17，<https://arxiv.org/abs/2104.08663>（延伸来源：18 个数据集、10 个检索系统、BM25 强基线、rerank 与 late-interaction 表现最好但代价高的结论来自此文）
- 第 4 节的检索指标、Wilson 置信区间、样本量估算、McNemar 精确检验 $p$ 值、Cohen's kappa 与 faithfulness 比例，除代码块直接输出的部分外，均按该节写明的口径自行复算，未引用任何未标注来源的数字。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
