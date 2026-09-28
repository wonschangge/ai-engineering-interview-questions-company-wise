---
type: question
id: rag-12
topic: RAG 与检索
order: 12
question: 你如何把生成答案中的每一条论断都归因到具体的 retrieved span？
question_en: How do you attribute every claim in a generated answer to a specific retrieved span?
asked_at: [Perplexity, Harvey, Abridge]
level: 高阶
tags: [引用, 归因, 可验证性, 幻觉]
sources:
  - title: Enabling Large Language Models to Generate Text with Citations（ALCE）（延伸）
    url: https://arxiv.org/abs/2305.14627
    author: Gao et al. (EMNLP 2023)
    published: 2023-05-24
related: [rag-04, rag-06, llm-internals-11, rag-08]
updated: 2026-09-28
---

## 一句话答案

> 归因分三层：答案级（这段回答来自哪几篇文档）、论断级（每一句话由哪个 span 支持）、span 级（落到具体段落、表格行或图注）。工程上落地的是论断级判定 + span 级定位，做法是两段式：生成时把候选 span 编号放进上下文并要求句末引用，生成后用独立的 NLI 蕴含模型校验「引用的 span 是否真的支持这句话」，不支持的句子删除或降级标注。ALCE 的口径下，把证据放进上下文直接提示的 ChatGPT 在 ELI5 上也只有 51.1% 的论断被引用完整支持（论文摘要的表述是「即使最好的模型也有 50% 的时候缺少完整的引用支持」），所以「让模型自己标引用」不等于完成归因，校验是必需环节。

## 面试官在考什么

- 是否区分归因粒度，并说清自己的产品要哪一层：只给「答案来自哪 5 篇文档」是答案级；能点开、能高亮、能审计的是论断级 + span 级。
- 能否给出至少三条实现路线并对比准确率、成本与可控性：提示内联、后验蕴含校验、约束解码；而不是只会说「让模型写 `[1][2]`」。
- 是否掌握引用质量的评测口径：论断级的 citation recall、citation 级的 citation precision，以及必须与 correctness、fluency 一起看。
- 工程闭环是否完整：span 的稳定 id 与版本、多跳论断的一句多引、表格行列级定位、无引用句子的保守处理、校验带来的延迟增量。
- 失败模式意识：幻觉引用（引用存在但不支持）、后验匹配式引用、版本错配、越权引用、长答案引用漂移。

常见错误答案：

- **「句末让模型写 [1][2] 就完成了归因」**——引用存在不等于引用支持。ALCE 在 ELI5 上测得 ChatGPT（top-5 段落、直接提示）的 citation recall 只有 51.1%。
- **「用 embedding 相似度把答案句匹配回 span 即可」**——ALCE 的 PostCite 正是这个思路（用 GTR 在 top-100 段落里找最匹配的一段作为引用），论文报告 ClosedBook + PostCite 在 ASQA 上的 citation recall 比 Vanilla 低 47%。相似度给的是「像」，归因要的是「蕴含」。

## 原理与推导

**符号约定。** 检索得到候选 span 集合 $P = \{p_1, \dots, p_n\}$，每个 span 携带定位元组 $(doc\_id, version, char\_start, char\_end)$。生成答案被切成论断 $s_1, \dots, s_m$（句子级，长句再按并列/分号切成子句）。归因要做两件事：为每条论断求引用集合 $\mathcal{C}_i \subseteq P$，并给出支持判定 $\phi(\text{concat}(\mathcal{C}_i), s_i) \in \{0, 1\}$，其中 $\phi$ 是以 span 文本为前提、论断为假设的蕴含判定器。

**先想清楚要它干什么。** 归因有四个互不替代的用途：可验证（用户点开原文逐句核对）、可追责（合规与审计要求答案能回溯到当时那一版文档）、压幻觉（模型知道每句话都要挂证据，会少编）、评测信号（citation recall 既能做线上监控，也能反过来用于选答案）。四个用途对粒度和召回率的要求不同：审计要求 span 级与版本锁定，压幻觉只要求论断级与足够高的召回，选答案只要求一个排序信号。

**三个粒度的差别在「可核对的最小单位」。**

| 粒度 | 输出形式 | 用户能做什么 | 成本 |
| --- | --- | --- | --- |
| 答案级 | 5 个文档链接 | 知道出处，但不知道哪句话来自哪篇 | 最低，检索结果直接回传 |
| 论断级 | 每句挂 1–3 个 span id | 逐句核对、逐句审计 | 需要生成时标注 + 生成后校验 |
| span 级 | id 加字符偏移、页码、表格行列 | 点击跳转并高亮到原文那几行 | 需要解析期保留偏移与结构信息 |

**路线 A：提示内联（cite-as-you-generate）。** 把候选 span 编号写进上下文（`[1] 段落内容`），指令要求「每个事实句句末标注来源编号，没有依据的句子不要写」。有效的原因是证据与生成在同一上下文里，模型可以逐句对照而不是靠参数记忆。误差来源有三类：模型对语义近似但结论不同的 span 辨识不足（把「A 导致 B」引成「B 导致 A」的段落）、指令遵循失败（长答案后段忘记标注）、长上下文里证据位置与生成位置距离过远导致错配（lost-in-the-middle 效应）。长答案另有一类引用漂移：后段论断挂到了前段论断的 span 上，而不是它自己的证据；对策是分段生成、每段独立重编号并逐段校验，让漂移在段内就被检出，不跨段累积。可控、便宜、只需一次生成，是默认起点。

**路线 B：后验归因（post-hoc verification）。** 先生成，再对每条论断独立判定。ALCE 的定义可以直接借用：

$$\text{recall}_i = \mathbb{1}\left[|\mathcal{C}_i| \ge 1 \wedge \phi(\text{concat}(\mathcal{C}_i), s_i) = 1\right], \qquad \text{citation recall} = \frac{1}{m}\sum_{i=1}^{m} \text{recall}_i$$

也就是「这条论断至少有一个引用，且所有引用拼起来能推出它」。精确率按 citation 计：引用 $c_{i,j}$ 是冗余的，当且仅当它自己不能支持该论断、而删掉它其余引用仍然能支持：

$$c_{i,j}\ \text{irrelevant} \iff \phi(c_{i,j}, s_i) = 0 \ \wedge\ \phi(\text{concat}(\mathcal{C}_i \setminus \{c_{i,j}\}), s_i) = 1$$

$c_{i,j}$ 得 1 分当且仅当 $\text{recall}_i = 1$ 且它不是冗余的，最后对所有 citation 求平均。这个定义刻意**不要求最小引用集**：冗余引用不扣分，只有当一条引用既不能单独支持、删掉它证据依然充分时才算多余——因为真实写作里人也会为关键结论多引一条增强可信度。

**为什么要用 concat 而不是逐条判定。** 多跳论断 $s$ 需要 $p_a$ 与 $p_b$ 联合支持（「先把 X 换算成 Y，再与 Z 比较」这类），单独每条都不蕴含 $s$，逐条判定会把正确的多跳引用误杀成无依据。concat 判定的副作用是把「拼接幻觉」也判成支持：两段各自无关的文本拼起来，恰好能推出一条新结论。所以生产上做两层判定——逐条 $\phi(c_{i,j}, s_i)$ 加联合 $\phi(\text{concat}(\mathcal{C}_i), s_i)$，两层结论不一致的论断标为「部分支持」，进入人工复核或升级给 LLM judge。

**路线 C：约束解码与结构化输出。** 把引用变成硬约束：用 grammar 限制引用位置的 token 只能取自候选 id 集合、用 JSON schema 强制 `{"sentence": ..., "citations": [span_id]}` 的结构，或用句末引用 token 的 logits 掩码。它消除的是「引用一个根本不存在的文档」这类错误，代价是约束会占用模型容量、可能压低流畅度与正确率，因此适合对格式要求严格的场景（医疗、法律），不适合把约束开到最紧。

**路线 D（辅助）：token 级归因。** 用 attention 权重、梯度或 token 相似度把答案 token 映射到证据 token。它只能做调试与可视化：attention 权重与输出贡献不是一一对应（attention sink、多头叠加、残差路径），开源实现里取全层全头 attention 的显存代价也很高——$L = 2048$、32 层、32 头、fp16 就要 $2048^2 \times 32 \times 32 \times 2$ 字节约 8.6 GB，$L = 8192$ 时约 137 GB，所以实践上通常只留最后一层再对 head 求平均。

**论断怎么切。** 评测与校验的单位都是 statement/claim，切分口径决定指标口径。ALCE 把 ELI5 的参考答案用 InstructGPT 拆成 sub-claims 作为评测单位（附录 A），三个数据集的人类答案规模差异也很大：ASQA 平均 65 词、ELI5 平均 131 词、QAMPARI 每个问题平均 13 个短答案（附录 B）。工程规则：按句子切，长句按并列结构再切；一条论断尽量只含一个可验证的事实主张，把「因此」「综上」这类推理连接词单独记录成推理跨度，不要让一条论断同时承担事实与推理。

## 数值与代码验证

**ALCE 在 ELI5 上的主结果**（论文的 ELI5 主表即表 6，口径：GTR 检索 top-5 段落放入上下文；fluency 用 MAUVE；correctness 是 claim recall；citation recall 按论断平均、citation precision 按 citation 平均；判定用 NLI 模型 TRUE）。

| 系统（ELI5, top-5 段落） | MAUVE | claim recall | citation recall | citation precision |
| --- | --- | --- | --- | --- |
| ChatGPT Vanilla | 57.2 | 12.0 | 51.1 | 50.0 |
| ChatGPT Vanilla + Rerank（best-of-4） | 56.1 | 11.4 | 69.3 | 67.8 |
| ChatGPT Summ（top-10） | 40.3 | 12.5 | 51.5 | 48.2 |
| ChatGPT ClosedBook + PostCite | 32.6 | 18.6 | 15.5 | 15.5 |
| GPT-4 Vanilla | 38.4 | 14.2 | 44.0 | 50.1 |
| GPT-4 Vanilla（top-20） | 41.5 | 18.3 | 48.5 | 53.4 |

三点读法。其一，论文摘要的结论「即使最好的模型也有 50% 的时候缺少完整的引用支持」正对应 ELI5 这一栏：Vanilla 的 citation recall 51.1%。其二，这里的 Rerank 不是检索端的 reranker，而是对同一问题采样 4 个回答、用自动 citation recall 选最好的那个，换来 +18.2 个点的 citation recall（51.1 → 69.3），说明「校验信号本身可以用来选答案」。其三，ClosedBook 不给引用，表中的引用分数来自 PostCite 事后补引用（论文 § 4.3 说明 PostCite 与 ClosedBook 搭配使用，ASQA 主表表 4 的表注也写明 ClosedBook 的引用由 PostCite 补出），它的 correctness 反而更高（18.6）而 citation recall 掉到 15.5——不把证据放进上下文、只做事后匹配，引用质量会塌。对照 ASQA（人类答案平均 65 词，答案更短、更贴近检索到的段落）同一套 ChatGPT Vanilla + GTR top-5 配置的 EM recall 是 40.4、citation recall 73.6、precision 72.5（论文 ASQA 主结果表 4，正确性一列即 EM Rec.）：论断越长、越需要跨段落综合，完整支持的难度越高，ELI5 的 131 词答案正是难例。再看 Summ 行（把 top-10 段落先摘要成短上下文再生成）：correctness 略升到 12.5，citation recall 与 precision 却掉到 51.5 与 48.2，论文把这归因于有损压缩——摘要丢掉了原文的限定条件，模型据此生成的句子再引回原文就指不准。所以摘要式压缩天然与引用冲突，要么禁止无引用的概括句，要么让概括句引用聚合来源并标注「本句为归纳」。

**指标要跟产品目标对齐。** 医疗与法律场景把 citation recall 放在 citation precision 之前（宁可多引一条，不能漏掉依据），开放域搜索与助手场景更在意 citation precision 与 correctness（不能让用户点到无关来源）。这也是为什么不能只报一个综合分：ALCE 三个维度分开报，正是为了让不同类型的系统（长答案综合型 vs 短答案列举型）在不同场景下各有取舍依据。自动判定的口径与误差上限也要写明：ALCE 的引用指标由 NLI 模型 TRUE 判定（§ 3.3 给出该选择），把人工标注当金标时 citation recall 的判定准确率是 85.1%、citation precision 是 77.6%，Cohen's κ 分别为 0.698（substantial）与 0.525（moderate）（§ 6 与 § G.5）；附录 A 另用 40 个输出、共 120 对人工标注检查蕴含判定器，报告准确率 80.0%——那是 correctness 侧 sub-claim 蕴含判定的误差，不能直接当成引用指标的一致性。

**一次引用校验的算力账**（口径：8 条论断 × 平均 1.5 条引用 = 12 对 (claim, cited span)；前向 FLOPs 按 $2PL$ 估，只算权重项、忽略 attention 的 $L^2$ 项；A100 80GB SXM 的 BF16 dense 算力按 312 TFLOP/s）。

| NLI 判定模型 | 定长 $L=512$ 的 FLOPs | 理论下限 | 按真实长度 $L=286$ | 理论下限 |
| --- | --- | --- | --- | --- |
| MiniLM-L6（22.7M） | $12 \times 2 \times 2.27\times10^7 \times 512 \approx 2.8\times10^{11}$ | 0.9 ms | $1.6\times10^{11}$ | 0.5 ms |
| DeBERTa-v3-base（184M） | $\approx 2.3\times10^{12}$ | 7.3 ms | $1.3\times10^{12}$ | 4.1 ms |
| DeBERTa-v3-large（435M） | $\approx 5.4\times10^{12}$ | 17.1 ms | $3.0\times10^{12}$ | 9.6 ms |

真实长度 286 token 的取法是论断 30 token 加 span 256 token。理论下限是「整批 12 对打成一次前向、GPU 打满」的理想值，小 batch 下 kernel 启动、padding 与 Python 编排会吃掉 3–8 倍余量，所以论断级校验的实测量级在几十毫秒到一两百毫秒。降本的最有效手段不是换小模型，而是减少对数：上面 12 对是「校验已有引用」的账；走路线 B 做后验归因时还要把每条论断与全部候选 span 逐对判定，那一步是 $8 \times 5 = 40$ 对，用 embedding 预筛把候选 span 从 5 个降到 2 个后变成 $8 \times 2 = 16$ 对（降到原来的 40%，代价是预筛漏掉的 span 不可能再被引用），同时把长 span 截断到 256 token。

**LLM judge 校验贵一个到两个数量级**（口径：8 条论断一次批量调用，prompt 含 5 个 span 共 1500 token，输出 200 token；prefill 按 $2PT$）。

| 判定模型 | prefill FLOPs | 理论下限 | 200 token 输出（memory-bound） |
| --- | --- | --- | --- |
| 8B | $2 \times 8\times10^9 \times 1500 \approx 2.4\times10^{13}$ | 77 ms | 单卡 50–100 tok/s，约 2–4 s |
| 70B | $2 \times 7\times10^{10} \times 1500 \approx 2.1\times10^{14}$ | 673 ms | 更慢或需要多卡 |

结论：不要对每条论断同步调一次大模型。可行的是级联——NLI 先跑，只把「部分支持」「矛盾」的论断升级给 judge，按 30% 的升级率算，judge 的调用量与同步等待时间降到原来的三成（以表中 8B 行约 2–4 s 的批量调用为基线约 0.6–1.2 s；70B 单次只会更慢，这里只做量级估算），再叠加按比例人工抽查。

**span 定位与校验的可运行代码**（10 条论断的玩具例，含 ALCE 口径的指标与保守模式裁剪）：

```python
# 引用集合：每条论断挂了哪些 span id（s4 无引用）
statements = {"s1": ["p1"], "s2": ["p1", "p3"], "s3": ["p2"], "s4": [],
              "s5": ["p1", "p4"], "s6": ["p5"], "s7": ["p6", "p7"],
              "s8": ["p8"], "s9": ["p9", "p10"], "s10": ["p11"]}

# phi(spans, claim) -> bool：spans 拼起来是否蕴含 claim。
# 生产环境换成 NLI 模型前向；这里用查表模拟，便于核对每个分支。
SUPPORT = {("p1", "s1"): True, ("p1", "s2"): False, ("p3", "s2"): False, ("p1,p3", "s2"): True,
           ("p2", "s3"): False, ("p1", "s5"): False, ("p4", "s5"): True, ("p1,p4", "s5"): True,
           ("p5", "s6"): True, ("p6", "s7"): False, ("p7", "s7"): False, ("p6,p7", "s7"): True,
           ("p8", "s8"): True, ("p9", "s9"): False, ("p10", "s9"): False, ("p9,p10", "s9"): False,
           ("p11", "s10"): True}


def phi(spans, claim):
    return SUPPORT.get((",".join(spans), claim), False)


recall = {s: bool(c) and phi(c, s) for s, c in statements.items()}
citation_recall = sum(recall.values()) / len(statements)

scores = []
for s, cites in statements.items():
    for c in cites:
        rest = [x for x in cites if x != c]
        irrelevant = (not phi([c], s)) and bool(rest) and phi(rest, s)
        scores.append(1.0 if (recall[s] and not irrelevant) else 0.0)
citation_precision = sum(scores) / len(scores)
citation_f1 = 2 * citation_precision * citation_recall / (citation_precision + citation_recall)

kept = [s for s in statements if recall[s]]      # 保守模式：删掉没有依据的句子
print(f"recall={citation_recall:.4f} precision={citation_precision:.4f} "
      f"f1={citation_f1:.4f} kept={len(kept)}/{len(statements)}")
# recall=0.7000 precision=0.6923 f1=0.6961 kept=7/10
```

这个玩具例里有三个刻意设计的点，正好对应上面的推导：`s2` 与 `s7` 是联合支持（单条引用各自不蕴含，两条一起才蕴含），按定义两条引用都拿满分；`s5` 的 `p1` 是冗余引用（`p4` 单独就够），被记 0 分，所以精确率是 $9/13 = 0.6923$ 而不是 $1.0$；`s3` 与 `s9` 引用存在但不支持，论断与它的引用一起被记为 0 分。保守模式下 10 句只剩 7 句，答案长度按每句 25 字算从约 250 字压到约 175 字——这就是「宁可少说」的代价，产品上要用「本段未在资料中找到依据」之类的可见标注替代静默删除，否则用户会以为信息不存在。

**span 定位的最小数据结构**：

```python
span = {
    "span_id": "sha1(doc_version + char_start + char_end)",  # 稳定 id：内容变了 id 必须变
    "doc_id": "contract-2024-11",
    "doc_version": "v3",
    "char_start": 18422, "char_end": 19010,   # 前端高亮用
    "page": 7, "section": "8.2 违约责任",
    "table_ref": None,                        # 表格填 {"row": 3, "col": "金额"}
}
```

## 常见追问

- **追问**：为什么不能只靠 embedding 相似度做归因？
  - 要点：相似度衡量的是字面/语义接近，蕴含衡量的是「前提能否推出假设」。否定（「不承担赔偿责任」vs「承担赔偿责任」）、数值与单位（3.2% vs 3.2 倍）、时间范围（2023 年 vs 2023 财年）、条件限定（「仅在书面同意后」）都能让高相似文本变成不支持。ALCE 的 PostCite 用 GTR 在 top-100 里选最匹配段落补引用，ClosedBook + PostCite 在 ASQA 上的 citation recall 比 Vanilla 低 47%，这就是纯匹配路线与「证据进上下文」路线的差距。成本结构也不同：相似度是双塔（bi-encoder），span 向量离线算好、在线只做一次点积，单次比较是 $O(d)$，适合当召回级预筛；蕴含判定是交叉编码（cross-encoder），每个 (论断, span) 对都要跑一次联合前向，成本随对数与序列长度增长，上面那张 FLOPs 表算的就是这一项。所以相似度用来缩小候选，蕴含判定用来下结论，两者是流水线关系而不是替代关系。
- **追问**：NLI 模型与 LLM judge 怎么取舍？
  - 要点：NLI 便宜、可批量、判定可复现，缺点是只给蕴含/中性/矛盾，对数值比较、多跳链、时序推理不敏感；ALCE 用 TRUE 模型做自动判定，在人工标注的 120 对上（sub-claim 蕴含）准确率 80.0%，引用指标自身与人工标注的一致性是 85.1% / 77.6%，这些数字既是可信度参考也是误差上限。LLM judge 能处理多跳与比较类论断，但成本高一到两个数量级（上面 70B 的账），还有位置偏差与自我一致性偏差。工程做法是级联加抽查：NLI 全量跑，partial/矛盾的升级 judge，再按 2%–5% 抽样人工复核并监控指标漂移。
- **追问**：多跳论断的引用怎么评？
  - 要点：逐条判定必然误杀多跳，所以要用联合判定（ALCE 的 concat 口径）；但联合判定会放过拼接幻觉。因此要求每个跳点各自有引用（引用数 $\ge$ 跳数）、联合判定必须成立、并单独统计「推理跨度」（一条论断的引用数与跳数），跨度大的论断进人工复核队列。产品上允许一句多引，不要为了引用数量好看强行合并引用。
- **追问**：引用会拖慢生成吗？
  - 要点：引用标记本身的 token 开销可忽略（每句 1–3 个标记）；真正的增量在校验。NLI 批量 12 对在理论下限 17 ms 量级、实测留 3–8 倍余量约 50–150 ms，相对 2 s 量级的生成延迟不足 10%，而且可以按句流式增量校验，把校验时间藏在后续句子的生成时间里。别做的是每条论断同步调一次大模型（2–4 s 量级）或者把校验放在用户点击之后（那时已经无法阻止错误答案被读走）。
- **追问**：引用覆盖率 100% 但答案仍然是错的，怎么办？
  - 要点：引用只保证「有依据」，不保证「结论正确」。多段拼接推出的新结论、忽略反例的过度概括、把两个 span 的数值算成一个新数字，都能在引用齐全的情况下出错。对策：citation 指标必须与 correctness 一起看（链 [[rag-04]]），对聚合型论断重算数值而不是判蕴含，对「唯一结论式」表述要求同时给出反例 span 或显式写「资料未涉及」。
- **追问**：span 的粒度、版本与权限怎么定？
  - 要点：引用的最小可点击单位建议取段落级，表格定位到行列、图定位到图注（链 [[rag-10]]）；id 由文档版本加字符区间（或内容哈希）生成，索引更新后旧引用仍能回溯到当时的内容快照（链 [[rag-11]]）；权限过滤必须在检索阶段完成，模型的候选引用集合只能是用户可见子集，否则一次引用就构成越权泄露（链 [[rag-08]]）。

## 公司变体

- **Perplexity**：产品形态本身就是「每段论断后面挂编号来源、点开跳到原文」，所以偏工程实现。会追问引用在流式输出里的编号稳定性（后续句子插入引用时前面的编号不能跳）、span 高亮与原文跳转的偏移精度、同一来源被多次引用的去重、抓取与索引时效导致的「引用链接打不开或内容已变」兜底、以及 citation recall 与 precision 线上怎么持续监控。原理部分一般停在 citation recall/precision 的定义与评测口径，不会要求手推 NLI 判定式。
- **Harvey**：法律场景把引用当证据链，因此偏「可审计的实现」。会追问审计留痕（哪一版文档、哪一次检索、哪一次模型输出、哪个模型版本）、无依据论断的处理策略（保守模式删除还是显式标注「未找到依据」）、引用召回优先于精确率的取舍、以及合同类长文档里的条款级定位与跨条款综合。
- **Abridge**：临床场景的证据来源是双重的——医患对话转录（谁说了什么）与医学知识库（指南怎么写），所以会追问「这条论断的依据是患者自述还是外部文献」，以及两类来源在 UI 与审计上如何区分。医疗场景对幻觉的容忍度最低，会追问保守策略的边界：宁可少写也要有据、模型不确定时必须显式声明，以及引用被人工复核推翻后的回流机制。
- 三家的共同点是都要求论断级 + span 级归因，都关心可靠性而不是规模；差别在证据类型（开放网页 / 法律文书 / 临床对话与医学文献）与审计留痕强度。以上是依据各家公开产品形态与业务方向推断的考察侧重，不涉及具体面试流程。

## 相关题目

- [[rag-04]]：引用指标属于 generation 侧的可验证性维度，必须与 correctness 一起看；只报 citation recall 会奖励「每句都引、但结论错误」的系统。
- [[rag-06]]：agentic RAG 的多轮检索下，每轮新检索到的 span 需要重新编号并保持 id 稳定，否则跨轮的引用会漂移到上一轮的候选集合。
- [[llm-internals-11]]：「先给依据再下结论」和引用格式是 post-training 学到的行为，SFT 数据里要有带引用与无依据拒答的样本，否则约束解码只能管住格式、管不住内容。
- [[rag-08]]：模型的候选引用集合必须已经过权限过滤，引用越权文档比答错更严重。
- [[rag-10]]：表格与图表需要行列级、图注级定位，否则 span 级归因在 PDF 场景会退化成整页引用。
- [[rag-11]]：索引更新后旧引用要么锁版本快照、要么显式失效，否则用户点开看到的内容与当时引用的内容不一致。
- [[rag]]：专题导读，检索、重排、生成与评测的完整链路。

## 参考资料与归属

- [Enabling Large Language Models to Generate Text with Citations（ALCE）（延伸）](https://arxiv.org/abs/2305.14627) — Gao et al. (EMNLP 2023)，2023-05-24（v2 2023-10-31，EMNLP 2023 主会论文）。题解中引用的全部论文数字与定义来自该文：三维度评测（fluency 用 MAUVE、correctness、citation quality）、论断级 citation recall 与 citation 级 citation precision 的形式化定义、用 NLI 模型 TRUE 做自动蕴含判定（附录 A 的人工标注 120 对上准确率 80.0%；引用指标与人工标注的一致性另见 § 6 与 § G.5：citation recall 85.1%、citation precision 77.6%）、ELI5 与 ASQA 的具体数值、PostCite 与 best-of-4 选择（论文中称 Rerank）的做法与差距，以及 ASQA/ELI5/QAMPARI 三个数据集的人类答案规模统计（附录 B）。
- **延伸来源说明**：作业单只提供上述一条来源并标注为延伸来源。正文里的成本账（12 对论断-引用判定的 FLOPs 与理论下限、8B/70B judge 的 prefill 与 decode 量级、attention 权重显存）都是按正文写明的口径自行推导的估算，不是论文结论；三条实现路线与失败模式清单是对公开工程实践的归纳，未引用具体链接。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
