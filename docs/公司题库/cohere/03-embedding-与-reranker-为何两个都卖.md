---
type: question
id: cohere-03
company: Cohere
topic: rag
order: 3
question: 你有一个 embedding 模型和一个 reranker。为什么两个都要卖？请设计两阶段的检索流水线，并告诉我 reranker 在什么情况下值得付出那部分 latency。
question_en: You sell both an embedding model and a reranker. Why both? Design the two-stage retrieval pipeline, and tell me when the reranker is worth its latency.
asked_at: []
level: 高阶
tags: [双塔, 交叉编码器, 两阶段检索, recall@k, 延迟预算]
sources:
  - title: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks（延伸）
    url: https://arxiv.org/abs/1908.10084
    author: Reimers & Gurevych (EMNLP 2019)
    published: 2019-08-27
  - title: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（延伸）
    url: https://arxiv.org/abs/2004.12832
    author: Khattab & Zaharia (SIGIR 2020)
    published: 2020-04-27
  - title: Passage Re-ranking with BERT（延伸）
    url: https://arxiv.org/abs/1901.04085
    author: Nogueira & Cho
    published: 2019-01-31
  - title: 为大型商品目录设计语义搜索（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cohere-04, cohere-05, cohere-06, rag-01, rag-03]
updated: 2026-09-28
---

## 一句话答案

> **因为两者的计算形态决定了它们只能各管一段**：
> | | embedding（双塔 bi-encoder） | reranker（交叉编码器 cross-encoder） |
> | --- | --- | --- |
> | 打分方式 | $f(q)$ 与 $g(d)$ **各自独立**编码，取内积/余弦 | $h(q,d)$ **联合**编码 |
> | 能否预计算 | ✅ 文档向量可离线算好并建 ANN 索引 | ❌ 必须每个 (q,d) 对现场前向 |
> | 单查询复杂度 | $O(\log N)$（ANN）或 $O(1)$ 摊薄 | $O(k)$ 次前向（$k$ 为候选数） |
> | 判别力 | 中（query 与 doc 无交互） | **高**（token 级交互） |
> **所以两阶段的分工是**：
> $$\text{embedding 负责「别漏」（recall@k）};\qquad \text{reranker 负责「排对」（precision@n）}$$
> **关键约束（本题的题眼）**：**reranker 的召回上限就是第一阶段给的候选集**——
> $$\text{recall@}n\le\text{recall@}k$$
> 换句话说：**reranker 只能重排，不能找回**。所以第一阶段的目标是「用可接受的成本把 recall@k 顶到接近 1」，第二阶段才谈精度。
> **推荐的流水线**：
> ```
> ① 过滤（权限/时间/租户/语言）        ← 先砍掉不该出现的（合规）
> ② 第一阶段：embedding ANN top-k（k=100~1000）
>    + 词法检索（BM25/精确串）做混合      ← 补语义检索漏掉的专有名词/编号
> ③ 第二阶段：reranker 对 k 个候选打分 → top-n（n=5~20）
> ④ （可选）第三阶段：LLM listwise 精排 top-20 → top-5，或直接交给生成模型
> ```
> **reranker 什么时候值得那部分延迟**（四条判据，都可测量）：
> ① **第一阶段「排不准」而不是「找不回」**：即 $\text{recall@}k-\text{recall@}n$ 这个**缺口大**——这是 reranker 的全部空间（缺口小则白花钱）；
> ② **查询复杂/歧义**（长查询、多跳、否定、稀有术语）——此时交叉编码器的 token 级交互收益最大；
> ③ **错答代价高**（法律、医疗、金融、对外回复）——比一次 rerank 的几毫秒贵得多；
> ④ **n 很小**（只给模型喂 5 条）——top-5 的排序质量直接决定答案质量。
> **什么时候不值得**：查询简单且第一阶段精度已高；延迟预算极紧（语音/联想输入）；k 必须很大（成本线性）——此时用**级联**（小 reranker 先粗筛）+ **蒸馏**，或直接**把 k 调小并接受 recall 略降**。
> 一句话判据：**先量出「缺口」（recall@k − recall@n）和「预算」（能容忍多少毫秒），再决定要不要 reranker**——而不是默认「加了就好」。

## 面试官在考什么

- **能否说出结构差异**：双塔 vs 交叉编码器，以及「能否预计算」这个决定性的工程后果。
- **是否理解 recall 上限**：能否主动指出 **reranker 不能修复第一阶段漏掉的文档**——这是本题最核心的认知。
- **两阶段设计的完整性**：过滤、混合检索（词法 + 向量）、k 的选择、n 的选择、以及可选的第三阶段。
- **延迟的可算性**：能否把总延迟拆成"查询编码 + ANN + rerank（$k\times$ 每对耗时/批并行）+ LLM 首字"并算出 $k$ 的上限。
- **成本-收益**：能否用「缺口大小 × 查询价值」来决定是否 rerank，而不是靠感觉；以及能否提出**级联**与**蒸馏**作为折中。
- **k 与 n 的取舍**：k 大 → recall 高但延迟与成本线性上升；n 大 → 模型上下文变长、噪声变多（「lost in the middle」）。
- **混合检索**：能否指出纯向量检索在**专有名词、编号、精确串**上会漏，需要 BM25/关键词兜底。
- **评估方法**：能否给出「分开量两阶段」的评估（第一阶段看 recall@k，第二阶段看 nDCG@n / precision@n），以及**端到端指标**（答案正确率）。
- **诚实的边界**：承认有些场景 reranker 收益很小（此时不该卖/不该买），并给出替代方案。

**常见错误答案**

- 说「reranker 更准所以都上」（不说收益从哪来、代价是多少）。
- 认为 reranker 能提升召回（**不能**，受 $k$ 限制）。
- 只讲质量不讲延迟预算（语音场景加 100 ms 就是灾难）。
- 忽略混合检索（纯向量在编号/专名上失手）。
- k 设得极大（成本线性爆炸）而不做级联。
- 不区分「第一阶段评估」与「第二阶段评估」，导致无法定位问题在哪一段。

## 原理与推导

### 1. 为什么双塔能建索引、交叉编码器不能

- **双塔**：$s(q,d)=\langle f(q),g(d)\rangle$。因为 $g(d)$ 与 $q$ 无关，**可以在离线把全库 doc 向量算好**，查询时只需算 $f(q)$ 再做**最近邻搜索**（ANN 把 $O(N)$ 降到近似 $O(\log N)$）。
- **交叉编码器**：$s(q,d)=h([q;d])$。分数**依赖具体的 $q$**，无法预计算 → 每次查询要对每个候选做一次前向 → $O(k)$ 次前向。**这就是它只能用在「小候选集」上的原因**。

**推论**：两者不是替代关系，而是**「召回」与「精排」的分工**。

### 2. 两阶段的信息论下界

设第一阶段返回集合 $C$（$|C|=k$），相关文档集合为 $R$：

$$\text{recall@}k=\frac{|C\cap R|}{|R|}\qquad\text{precision@}n=\frac{|\text{top-}n\cap R|}{n}$$

因 reranker 只能重排 $C$：$\text{recall@}n\le\text{recall@}k$。**reranker 的收益上界**就是：

$$\Delta=\text{recall@}k-\text{recall@}n\ (\text{不加 reranker 时})$$

**如果 $\Delta\approx0$（第一阶段已经把相关的排在最前），reranker 的收益就接近零**——这是「该不该买」的第一个量化判据。

### 3. 延迟模型

$$T_{\text{total}}=T_{\text{embed}}(q)+T_{\text{ANN}}(k)+T_{\text{rerank}}(k)+T_{\text{LLM}}$$

$$T_{\text{rerank}}(k)=\Big\lceil\frac{k}{B_{\text{batch}}}\Big\rceil\times t_{\text{forward}}(L_q+L_d)$$

**例**：每对前向 1.5 ms（cross-encoder base 级、序列 512）、批并行 16 → $k=100$ 时 ≈ $7\times1.5=10.5$ ms；$k=1000$ 时 ≈ $63\times1.5=95$ ms。**在「首字延迟预算 300 ms」的语音场景里，$k=1000$ 就已经吃掉三分之一**。

**级联（cascade）**：先用**小 reranker**（0.3 ms/对）跑 $k=1000$ → 取 top-50，再用**大 reranker**（1.5 ms/对）跑 50 → 总 ≈ $63\times0.3+4\times1.5\approx25$ ms，**远低于直接用大模型跑 1000**（95 ms），质量损失有限。

### 4. 什么时候值得（期望效用）

$$\text{收益}=V\cdot\big(P(\text{对}\mid\text{rerank})-P(\text{对}\mid\text{无 rerank})\big)\qquad\text{成本}=c_{\text{rerank}}+\text{(延迟的机会成本)}$$

其中 $V$ 是「答对一次的业务价值」。**判据**：当 **缺口大**（$\Delta$ 大）**且 $V$ 高**时值得；当 $V$ 低（闲聊、低价值查询）或 $\Delta\approx0$ 时不值得。

### 5. k 与 n 的选择

| 参数 | 影响 | 经验 |
| --- | --- | --- |
| $k$（第一阶段候选数） | ↑ 则 recall 高、延迟与 rerank 成本线性上升 | 100–1000；用**召回曲线**找拐点（recall 饱和处） |
| $n$（喂给 LLM 的条数） | ↑ 则上下文变长、噪声与「中间遗忘」风险上升 | 3–20；配合引用与去重 |
| 阈值 | 过滤低分候选 | 用**绝对阈值 + 相对阈值（top 比例）**组合 |

**找 $k$ 拐点的方法**：画 $\text{recall@}k$ 随 $k$ 的曲线，取「再增大 k 收益 < 1 个百分点」的位置——**这一步能把 k 从 1000 降到 200，延迟省 80%**。

### 6. 混合检索为什么必要

纯向量检索在以下情况会漏：
- **精确串/编号**（错误码 `E-4021`、订单号、函数名）——向量的语义相似度帮不上；
- **稀有专名**（新出现的产品名，embedding 没见过）；
- **否定与逻辑**（「不含 X 的方案」）。

所以第一阶段通常是 **BM25 + 向量 → RRF/加权融合**，再由 reranker 统一打分（串 [[rag-03]] 的混合检索与融合）。

## 数值与代码验证

### 表 1：reranker 的收益上界（模拟检索，见代码输出）

| 配置 | recall@k | recall@n（无 rerank） | 缺口 $\Delta$ | recall@n（有 rerank） |
| --- | --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：延迟预算下的 $k$ 上限（每对 1.5 ms、批并行 16）

| 目标首字预算 | 可用于 rerank | 最大 $k$（直接大模型） | 最大 $k$（级联） |
| --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# 两阶段检索：recall@k 上限、reranker 收益、延迟预算与级联、期望效用判据
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 模拟检索质量：双塔 vs 交叉编码器 ----------
@dataclass
class Corpus:
    n_docs: int = 20_000
    rel_per_query: int = 3
    noise: float = 0.22          # 双塔打分的噪声（真实同域数据上 recall@100 通常 0.7–0.95）
    ce_gain: float = 1.2         # 交叉编码器对相关性的放大（判别力更强）
    seed: int = 11
    def make(self, n_queries: int) -> List[Dict[str, object]]:
        rnd = random.Random(self.seed)
        out = []
        for _ in range(n_queries):
            rel = set(rnd.sample(range(self.n_docs), self.rel_per_query))
            docs = {}
            for d in range(self.n_docs):
                is_rel = d in rel
                base = 1.0 if is_rel else 0.0
                # 双塔：独立编码 -> 噪声大
                docs[d] = {"bi": base + rnd.gauss(0, self.noise),
                           # 交叉编码器：联合编码 -> 同样的信号 + 更小的噪声
                           "ce": base * self.ce_gain + rnd.gauss(0, self.noise * 0.45)}
            out.append({"rel": rel, "docs": docs})
        return out
def recall_at(scores: List[Tuple[int, float]], rel: set, k: int) -> float:
    top = [d for d, _ in sorted(scores, key=lambda x: -x[1])[:k]]
    return len(set(top) & rel) / len(rel)
def pipeline_quality(queries, k: int, n: int) -> Dict[str, float]:
    r_k = r_n_bi = r_n_ce = 0.0
    for q in queries:
        rel, docs = q["rel"], q["docs"]
        bi = [(d, v["bi"]) for d, v in docs.items()]
        r_k += recall_at(bi, rel, k)                       # 第一阶段 recall@k
        r_n_bi += recall_at(bi, rel, n)                    # 不做 rerank，直接取前 n
        cand = sorted(bi, key=lambda x: -x[1])[:k]          # 候选集
        ce = [(d, docs[d]["ce"]) for d, _ in cand]          # reranker 只在候选集内重排
        r_n_ce += recall_at(ce, rel, n)
    m = len(queries)
    return {"recall@k": r_k / m, "recall@n(无rerank)": r_n_bi / m,
            "recall@n(有rerank)": r_n_ce / m}
qs = Corpus().make(60)
print("① 两阶段检索的质量分解（60 个查询、每查询 3 篇相关文档、2 万文档库）")
print(f"  {'k':>6} {'n':>4} {'recall@k':>10} {'recall@n(无)':>13} {'recall@n(有)':>13} "
      f"{'缺口Δ':>8} {'reranker 补回':>13}")
for k, n in ((20, 5), (50, 5), (100, 5), (200, 5), (500, 5)):
    r = pipeline_quality(qs, k, n)
    gap = r["recall@k"] - r["recall@n(无rerank)"]
    fixed = r["recall@n(有rerank)"] - r["recall@n(无rerank)"]
    print(f"  {k:>6} {n:>4} {r['recall@k']:>10.2f} {r['recall@n(无rerank)']:>13.2f} "
          f"{r['recall@n(有rerank)']:>13.2f} {gap:>8.2f} {fixed:>13.2f}")
print("  读法：**reranker 的表现被 recall@k 卡住**（它只能重排候选集）——k 越大，它的上限越高；")
print("        同时「缺口Δ」就是它的全部空间：缺口小的时候加 reranker 收益很小")

# ---------- 2) 延迟预算与级联 ----------
@dataclass
class Latency:
    embed_ms: float = 8.0
    ann_ms: float = 12.0
    llm_ttft_ms: float = 180.0
    small_ms: float = 0.3        # 小 reranker 每对耗时
    large_ms: float = 1.5        # 大 reranker 每对耗时
    batch_par: int = 16          # 批并行度（同时前向的对数）
    def rerank_ms(self, k: int, per_pair: float) -> float:
        return math.ceil(k / self.batch_par) * per_pair
    def total_ms(self, k: int, per_pair: float) -> float:
        return self.embed_ms + self.ann_ms + self.rerank_ms(k, per_pair) + self.llm_ttft_ms
L = Latency()
print("\n② 延迟预算：不同 k 与不同 reranker 的总延迟（含 LLM 首字 180 ms）")
print(f"  {'k':>6} {'无 rerank':>11} {'小 reranker':>12} {'大 reranker':>12} "
      f"{'级联(小->大)':>13} {'级联第二级':>11}")
for k in (50, 100, 200, 500, 1000):
    base = L.total_ms(0, 0)
    small = L.total_ms(k, L.small_ms)
    large = L.total_ms(k, L.large_ms)
    cascade = (L.embed_ms + L.ann_ms + L.rerank_ms(k, L.small_ms)
               + L.rerank_ms(50, L.large_ms) + L.llm_ttft_ms)
    print(f"  {k:>6} {base:>11.1f} {small:>12.1f} {large:>12.1f} {cascade:>13.1f} "
          f"{'50 条':>11}")
print("  读法：**大 reranker 的成本随 k 线性上升**（k=1000 时增加 94.5 ms）；级联（小模型粗筛 1000")
print("        → 大模型精排 50）只增加约 25 ms，**是大模型直跑的四分之一** —— 这是延迟紧时的标准解法；")
print("        注意级联比「只用小模型」贵一点（多了第二级），但质量明显更好（小模型判别力有限）")

# ---------- 3) 期望效用：该不该上 reranker ----------
@dataclass
class Utility:
    value_correct: float = 5.0      # 答对一次的业务价值（美元）
    cost_per_query: float = 0.002   # 每次查询的 LLM/检索成本
    rerank_cost_per_1k: float = 0.05  # 每 1000 对重排的算力成本
    latency_value_per_100ms: float = 0.0  # 延迟的机会成本（可设为 0 做纯质量对照）
    def roi(self, gap: float, fix_ratio: float, k: int,
            latency_added_ms: float) -> Dict[str, float]:
        """gap: recall@k - recall@n(无)；fix_ratio: reranker 能补回缺口的多大比例"""
        gain = self.value_correct * gap * fix_ratio
        cost = k / 1000 * self.rerank_cost_per_1k + latency_added_ms / 100 * self.latency_value_per_100ms
        return {"质量收益($)": gain, "成本($)": cost, "净收益($)": gain - cost,
                "ROI": (gain - cost) / cost if cost else float("inf")}
U = Utility()
print("\n③ 期望效用（答对价值 \$5/次、每 1000 对重排 \$0.05）")
SCEN = [
    ("第一阶段已很强（闲聊）", 0.05, 0.30, 100, 6.0),
    ("通用问答", 0.20, 0.60, 200, 19.5),
    ("长查询/多跳", 0.35, 0.75, 200, 19.5),
    ("法律/医疗（错答代价高）", 0.30, 0.70, 500, 48.0),
    ("k=1000 且不做级联", 0.30, 0.60, 1000, 94.5),
]
print("  （a）只算算力成本（延迟视为免费）—— 结论一定是「都值得」，因为算力太便宜")
print(f"  {'场景':<24} {'缺口Δ':>7} {'k':>6} {'质量收益($)':>11} {'算力成本($)':>11} "
      f"{'净收益($)':>10}")
for name, gap, fix, k, lat in SCEN:
    r = U.roi(gap, fix, k, lat)
    print(f"  {name:<24} {gap:>7.2f} {k:>6} {r['质量收益($)']:>11.3f} "
          f"{r['成本($)']:>11.4f} {r['净收益($)']:>10.3f}")
print("  （b）计入**延迟机会成本**（延迟敏感场景：每 100 ms 值 \$2.00，例如语音/联想输入）")
U2 = Utility(latency_value_per_100ms=2.00)
print(f"  {'场景':<24} {'缺口Δ':>7} {'k':>6} {'延迟(ms)':>9} {'延迟成本($)':>11} "
      f"{'净收益($)':>10} {'建议':<12}")
for name, gap, fix, k, lat in SCEN:
    r = U2.roi(gap, fix, k, lat)
    adv = "上 reranker" if r["净收益($)"] > 0.2 else ("可上可不上" if r["净收益($)"] > 0 else "不值得")
    print(f"  {name:<24} {gap:>7.2f} {k:>6} {lat:>9.1f} {lat/100*2.00:>11.3f} "
          f"{r['净收益($)']:>10.3f} {adv:<12}")
print("  读法：**决定因素往往不是算力成本，而是延迟** —— 只算算力时「都值得」（重排极便宜）；")
print("        一旦按业务给延迟定价：**缺口小的场景**（闲聊，收益 \$0.075）被延迟成本吃掉而变成不值得，")
print("        **k=1000 且不做级联**（\$1.89 延迟成本 vs \$0.90 收益）同样翻负 —— 而它的正解是**级联**")
print("        （同样质量、延迟降到 25 ms）。所以判据是：「缺口 × 修正比例 × 单次价值」对上「延迟的机会成本」")

# ---------- 4) 混合检索：纯向量会漏掉的东西 ----------
print("\n④ 混合检索的必要性（专名/编号类查询）")
@dataclass
class Query:
    text: str
    kind: str          # exact / semantic
    bi_hit: bool       # 向量检索能否命中
    bm25_hit: bool     # 词法检索能否命中
QUERIES = [
    Query("错误码 E-4021 怎么解决", "exact", False, True),
    Query("订单号 SO-99182 的物流", "exact", False, True),
    Query("函数 get_user_balance 的签名", "exact", False, True),
    Query("如何申请退款", "semantic", True, False),
    Query("报销流程是什么", "semantic", True, False),
    Query("发票开错了怎么办", "semantic", True, True),
]
bi_only = sum(1 for q in QUERIES if q.bi_hit) / len(QUERIES)
bm25_only = sum(1 for q in QUERIES if q.bm25_hit) / len(QUERIES)
hybrid = sum(1 for q in QUERIES if q.bi_hit or q.bm25_hit) / len(QUERIES)
print(f"  {'查询':<32} {'类型':<9} {'向量':>5} {'词法':>5}")
for q in QUERIES:
    print(f"  {q.text:<32} {q.kind:<9} {('命中' if q.bi_hit else '漏'):>5} "
          f"{('命中' if q.bm25_hit else '漏'):>5}")
print(f"  纯向量召回率 {bi_only:.0%}；纯词法 {bm25_only:.0%}；**混合（并集）{hybrid:.0%}**")
print("  读法：**专名/编号类查询向量会漏、语义类查询词法会漏** —— 两者是互补的，")
print("        所以第一阶段的混合检索不是「锦上添花」，而是「必要项」")
```

预期输出要点（实跑）：① 质量分解显示 **reranker 的召回被 recall@k 卡住**（它只能在候选集内重排），且 **k 越大上限越高**；同时「缺口 $\Delta$」就是它的全部空间——缺口小的时候收益很小；② 延迟模型显示**大 reranker 成本随 k 线性上升**（k=1000 时约 95 ms、约占含 LLM 首字总延迟的三分之一），而**级联（小模型粗筛 1000 → 大模型精排 50）只要约 25 ms**；③ 期望效用分两栏讲清「决定因素不是算力而是延迟」：**只算算力时五个场景全部「值得」**（重排极便宜）；**按业务给延迟定价（\$2/100ms）后，「闲聊」（−\$0.05）与「k=1000 不做级联」（−\$1.04）翻负**，而长查询/多跳仍显著为正（+\$0.91）——k=1000 的正解是**级联**（同质量、延迟从 94.5 ms 降到 25 ms）；④ 混合检索的必要性用一组专名/编号查询说明**纯向量与纯词法各漏一半，并集才补齐**。

## 常见追问

- **追问**：为什么不直接用一个更强的 embedding 模型，省掉 reranker？
  - 要点：双塔的**结构上限**在那里——query 与 doc 没有 token 级交互，再强的编码器也无法在「需要细粒度匹配」的任务上追平交叉编码器（这是 ColBERT 这类「后交互」方案存在的理由：用多向量近似交互，兼顾索引与判别力）。**但反过来**：如果缺口本来很小（第一阶段已够好），增强 embedding 确实比加 reranker 更省延迟——**先量缺口再选方案**。
- **追问**：k 怎么定？
  - 要点：画 **recall@k 曲线**，取「再增大 k 收益 < 1 个百分点」的拐点。经验上 100–300 覆盖多数场景；**这一步常能把 k 从 1000 降到 200，延迟省 80% 而质量几乎不变**。另外 k 应随「查询难度」自适应（简单查询用小 k）。
- **追问**：怎么评估两阶段各自的贡献？
  - 要点：**分开量**——第一阶段看 recall@k（离线、用标注的相关集）、第二阶段看 nDCG@n/precision@n（给定候选集）；再做**消融**（去掉 reranker、去掉 BM25）看端到端答案正确率的变化。**只测端到端无法定位问题在哪一段。**
- **追问**：重排分数能不能缓存？
  - 要点：可以，但收益有限——**分数依赖具体 query**，只有「同一 query 反复出现」（热门问题、FAQ）才划算。更实用的缓存是：**embedding 查询缓存**、**最终答案缓存**、以及**文档向量（离线必然要）**。串 [[rag-06]] 的缓存层次。
- **追问**：reranker 和 LLM 精排（listwise）怎么选？
  - 要点：LLM 精排质量可能更高但**延迟与成本高一个数量级**，且**不适合打分排序的稳定性**（分数不可比）。常见做法：**reranker 做 k→20 的收窄，LLM 只在 top-20 内做一次 listwise 选择**（第三阶段），或**用 LLM 蒸馏出小 reranker**（把 LLM 的排序能力搬到便宜模型上）。
- **追问**：多语言/跨语言检索怎么办？
  - 要点：① 用**多语言 embedding**（共享向量空间）；② 注意**语言不匹配时的 recall 下降**（英文文档 + 法语查询）；③ 评估必须**按语言分群**（串 [[cohere-05]]）；④ reranker 也要选多语言的，否则跨语言重排会退化。
- **追问**：怎么把延迟再压？
  - 要点：① **级联**（小模型粗筛）；② **蒸馏**（用大 reranker 蒸馏小模型）；③ **量化/ONNX/TensorRT** 加速前向；④ **并行**（ANN 与 rerank 分块流水线化，边收边排）；⑤ **自适应 k**（简单查询直接跳过 rerank）。

## 相关题目

- [[cohere-04]]：1 亿文档的索引成本与 embedding 压缩——第一阶段的成本正是本题的前置约束。
- [[cohere-05]]：多语言检索的评估——跨语言时两阶段的表现差异与分群评估。
- [[cohere-06]]：索引扩大 10 倍后质量变差的排查——第一阶段召回变化会直接反映到端到端。
- [[rag-01]]：RAG 的基本流程与失败模式，对应本题在整体链路中的位置。
- [[rag-03]]：混合检索与融合排序，对应第一阶段的具体做法。

## 参考资料与归属

- **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks（延伸）** —— Reimers & Gurevych (EMNLP 2019)，2019-08-27：<https://arxiv.org/abs/1908.10084>。第 1 节「双塔可预计算、把检索成本降到可索引」的论证来自这篇。
- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（延伸）** —— Khattab & Zaharia (SIGIR 2020)，2020-04-27：<https://arxiv.org/abs/2004.12832>。第 1 节「后交互在索引与判别力之间折中」的框架来自这篇。
- **Passage Re-ranking with BERT（延伸）** —— Nogueira & Cho，2019-01-31：<https://arxiv.org/abs/1901.04085>。第 1、2 节「交叉编码器判别力强但只能用于小候选集」的结论来自这篇。
- **为大型商品目录设计语义搜索（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5、6 节的 k/n 取舍与延迟预算口径取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（2 万文档库、每查询 3 篇相关、双塔噪声 0.55、交叉编码器增益 1.2 与噪声系数 0.45、60 个查询、k=20–1000、n=5、每对 1.5 ms/0.3 ms、批并行 16、embed 8 ms、ANN 12 ms、LLM 首字 180 ms、\$$5 单次价值与 \$$0.05/千对成本）都是为演示取舍而构造的**示例参数**；真实收益与延迟必须用自家标注集与真实硬件实测。**「reranker 受 recall@k 上限约束」与「延迟随 k 线性」这两个结构性结论是可直接使用的部分。**
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
