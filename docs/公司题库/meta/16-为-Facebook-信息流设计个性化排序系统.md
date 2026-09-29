---
type: question
id: meta-16
company: Meta（超级智能实验室、FAIR、Llama）
topic: system-design
order: 16
question: 为 Facebook 的信息流设计一套个性化排序系统 / “下一篇帖子”逻辑。
question_en: Design a personalised news-feed ranking system / the “next post” logic for Facebook's feed.
asked_at: []
level: 高阶
tags: [系统设计, feed 排序, 多目标, 位置偏差, 实时特征]
sources:
  - title: Wide & Deep Learning for Recommender Systems（延伸）
    url: https://arxiv.org/abs/1606.07792
    author: Cheng et al. (Google)
    published: 2016-06-24
  - title: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction（延伸）
    url: https://arxiv.org/abs/1703.04247
    author: Guo et al.
    published: 2017-03-13
  - title: Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm（延伸）
    url: https://arxiv.org/abs/1809.05818
    author: Hu et al.
    published: 2018-09-16
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
related: [consumer-ml-06, consumer-ml-03, consumer-ml-13, system-design-04, meta-17]
updated: 2026-09-28
---

## 一句话答案

> Feed 排序的标准答案是**四层漏斗**：候选池（好友/关注/推荐/广告混合，$10^3\sim10^4$）→ 轻量召回打分（过滤到 $10^2$）→ 多目标精排（$10^2\to$ 几十）→ 重排/策略层（多样性、疲劳、完整性、广告插入，出最终十条）。但「下一篇帖子」这四个字才是真正的题眼，它要求三件工程上很难的事：
> ① **实时性**——用户刚划走一条长视频、刚点了赞，下一次请求（可能 2 秒后）的排序就要体现出来，所以要有**行为流 → 实时特征 → 在线打分**这条链路；
> ② **多目标不可合并成一个数就了事**——点击、停留、评论、分享、隐藏、取关、举报各自指向不同后果，要靠**多任务模型 + 权重/约束层**显式表达；
> ③ **位置偏差就在排序器自己身上**——训练数据是旧排序器产生的，位置与曝光都由它决定，所以训练要随机化位置、评估要去偏，否则模型只是在模仿过去的自己（串 [[consumer-ml-03]]）。

## 面试官在考什么

- **漏斗每一层的目标函数是否说清**：候选生成阶段只要「不漏」（优化 recall@N），精排阶段才优化序（NDCG/多目标），重排阶段处理的是**约束**而不是相关性。三层混着谈是典型减分项。
- **多目标的处理方式**：会不会提到多任务学习（共享底层 + 多头输出、MMoE/PLE 式的专家共享与门控）、以及**权重从哪里来**（业务目标 → 效用函数 → 权重，且要能解释权重变化的后果）。可引用的经典对照是 Wide & Deep 的「记忆 + 泛化」联合建模，以及 DeepFM 的「低阶与高阶交互同时建模」——都在说明**排序模型的关键是把不同性质的特征/交互放在同一模型里**。
- **实时特征链路**：能否给出「行为 → 流式聚合 → 特征存储 → 在线读取」的具体形态与延迟预算（通常要求预处理表 p99 在毫秒级、流式窗口在秒级）。
- **位置偏差的工程处理**：训练时位置随机化（把展示位置当作实验变量）、或用逆倾向加权（IPW）修正；可引用 Unbiased LambdaMART 的口径——**同时估计点击倾向与未点击倾向**，用成对方式联合学习去偏权重，并在商业搜索引擎上做过在线 A/B 验证；不处理位置偏差会让模型把「位置带来的点击」学成「内容好」。
- **负反馈与长期价值**：隐藏、取关、举报、以及「看到但没反应」都是信号；能否说明为什么**停留时长与点击会产生冲突**（诱饵式内容点击高、停留短、负反馈高）。

**常见错误答案**

- 「用点击率当唯一目标」——会直接把系统推向标题党与情绪化内容，短期 CTR 涨、长期负反馈与流失涨。
- 「离线 NDCG 涨了就上线」——日志是旧策略产生的，离线评估存在偏差；必须配在线实验与护栏（串 [[consumer-ml-02]]）。
- 「特征越实时越好」——实时特征会带来训练-服务偏斜与回填难度（在线特征在离线回放时不可得），要区分**可用实时特征**与**只能离线回填的特征**。
- 「多样性靠后处理随机化」——随机化会伤相关性；正规做法是把多样性/疲劳建成明确约束或直接进目标函数。

## 原理与推导

### 1. 漏斗与各层预算

| 层 | 输入规模 | 输出规模 | 目标 | 典型预算 |
| --- | --- | --- | --- | --- |
| 候选混合 | 好友/关注/推荐/广告/群组 | $10^3\sim10^4$ | 不漏（recall@N） | 20 ms（多路并行 + 截断） |
| 轻量打分 | $10^4$ | $10^2$ | 粗排，用少量特征 | 10 ms（小模型或双塔内积） |
| 多目标精排 | $10^2$ | 数十 | 序最优（多目标加权） | 40 ms（大模型、几百特征） |
| 重排/策略 | 数十 | 10 | 约束（多样性、疲劳、完整性、广告位） | 10 ms（规则 + 轻量优化） |

总预算落在 $80\sim120$ ms 量级（用户可感知的阈值），这与消费级 feed 的公开经验一致。

### 2. 多目标怎么变成单一排序分

最简形式是加权和：$\text{score}=\sum_k w_k\,\hat{p}_k$，其中 $\hat p_k$ 是各行为的预测概率。工程上要注意三件事：

1. **量纲**：点击概率是 0–1，停留时长是秒，分享是很小的概率。要么把长时长期望算成 $\hat p_{\text{click}}\times\hat t_{\text{dwell}}$（期望值），要么对各目标做归一化（除以其基线均值）。
2. **负向项**：负反馈（隐藏、取关、举报）应以负权重进入，或作为硬约束（预测负反馈概率超阈值直接降权/过滤）。
3. **权重不是模型的一部分**：它由业务效用决定，应该**可配置、可实验、可回滚**（这也是「把权重摆到台面上」的工程含义）。

### 3. 位置偏差：为什么它内生于排序器

日志里的交互概率可以写成

$$P(\text{click}\mid \text{item }d\text{ at position }k)=e_k\cdot r_{d}$$

$e_k$ 是位置倾向（examination propensity），$r_d$ 是内容相关性。训练时若把位置当成普通特征、又用**旧策略的位置分布**做样本，模型会把 $e_k$ 的一部分吸收进 $r_d$，于是「放到第一位的东西看起来更好」——这就是自我强化的偏差。

三种处理方式：① **训练时随机化位置**（小流量故意打乱，得到无偏样本）；② **IPW 加权**（权重 $1/e_k$，需要估计 $e_k$）；③ **成对/联合估计**（可引用 Unbiased LambdaMART 的做法：把点击与未点击的倾向放在一起联合学习，避免只校正一侧带来的偏差）。代价是方差：$e_k$ 很小时 $1/e_k$ 会放大噪声，所以要截断权重（weight clipping）。

### 4. 实时性与序列建模

「下一篇帖子」要求排序能反映**最近几秒**的行为。实现上有两条路：

- **特征路**：行为流（Kafka）→ 流式聚合（秒级窗口：最近 5 分钟点击类目分布、最近一次会话时长）→ 特征存储（毫秒级读取）。
- **序列路**：把用户最近 $L$ 次行为（例如 100–500 条）的 embedding 序列送进 Transformer，直接学「当前兴趣」，不必人工设计聚合特征。代价是每请求的计算量与特征新鲜度要求更高。

两者通常并存：序列模型提供兴趣表示，流式聚合提供近实时计数类特征。

### 5. 重排层要处理的约束

- **多样性**：同一作者/同一话题/同一媒体类型连续出现的惩罚（MMR 式贪心或直接进目标函数的分散项）。
- **疲劳**：同一内容重复曝光的衰减（按曝光次数与时间衰减）。
- **完整性（integrity）**：已被降权的内容不能因为在重排里「凑数」又被捞回来；这是安全与排序的交界（串 [[system-design-05]]）。
- **广告插入**：广告位是约束不是自由变量（位置、频率上限、体验指标），必须与自然内容联合排序而不是事后插入。

## 数值与代码验证

### 表 1：延迟预算分配（示例）

| 阶段 | 预算 | 说明 |
| --- | --- | --- |
| 多路召回 | 20 ms | 各路并行，超时即截断（用旧结果兜底） |
| 特征读取 | 30 ms | 在线特征存储 p99 目标 <10 ms/批，含近实时聚合 |
| 精排（$10^2$ 条 × 数百特征） | 40 ms | 大模型；批内并行 |
| 重排与策略 | 10 ms | 多样性、疲劳、完整性、广告位 |
| 网络与序列化 | 10 ms | 响应体只回必要字段 |

### 可运行代码

```python
# 1) 多目标加权 + 多样性重排：展示「权重变化如何改变最终十条的构成」
import random
random.seed(5)
CATS = ["朋友", "群组", "兴趣", "视频", "广告"]
items = []
for i in range(200):
    cat = random.choice(CATS)
    items.append({
        "id": i, "cat": cat,
        "p_click": random.betavariate(2, 8),
        "dwell": random.gauss(20, 8),
        "p_share": random.betavariate(1, 30),
        "p_hide": random.betavariate(1, 60),
    })

def score(it, w):
    return (w["click"] * it["p_click"]
            + w["dwell"] * (it["p_click"] * it["dwell"]) / 30.0
            + w["share"] * it["p_share"] * 5.0
            - w["hide"] * it["p_hide"] * 20.0)

def rerank(items, w, k=10, decay=0.35):
    """贪心：每选一条后，同类别候选打 decay 折扣（多样性）"""
    pool, out, seen = list(items), [], {}
    while len(out) < k and pool:
        best = max(pool, key=lambda it: score(it, w) * (decay ** seen.get(it["cat"], 0)))
        out.append(best)
        seen[best["cat"]] = seen.get(best["cat"], 0) + 1
        pool.remove(best)
    return out

cards = {
    "偏点击":  {"click": 1.0, "dwell": 0.2, "share": 0.2, "hide": 1.0},
    "偏停留":  {"click": 0.6, "dwell": 1.0, "share": 0.3, "hide": 1.0},
    "重负反馈": {"click": 0.6, "dwell": 0.6, "share": 0.5, "hide": 4.0},
}
for name, w in cards.items():
    top = rerank(items, w)
    dist = {c: sum(1 for it in top if it["cat"] == c) for c in CATS}
    exp_clicks = sum(it["p_click"] for it in top)
    exp_hides = sum(it["p_hide"] for it in top)
    print(f"{name:<6} 类别分布 {dist}  期望点击 {exp_clicks:.3f}  期望隐藏 {exp_hides:.4f}")

# 2) 位置偏差：用 IPW 校正前后对比（模拟旧策略的位置分布）
POS_PROP = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]     # e_k：位置 k 被查看的倾向
def naive_and_ipw():
    naive_num = ipw_num = ipw_den = 0.0
    for item in items[:60]:
        for k, e in enumerate(POS_PROP):
            if random.random() < e:                   # 只有被看到才可能点击
                clicked = random.random() < item["p_click"]
                r = 1.0 if clicked else 0.0
                naive_num += r                        # 朴素：直接用点击
                ipw_num += r / e                      # IPW：按倾向倒数加权
                ipw_den += 1 / e
                break
    return naive_num / 60, ipw_num / ipw_den
n, w = naive_and_ipw()
print(f"\n朴素点击率 {n:.3f}（混入位置效应）  IPW 估计 {w:.3f}（校正后）")
print("位置越靠前 e_k 越大，朴素估计会被靠前位置的高曝光拉高——这就是排序器自我强化偏差的来源")
```

预期输出要点：三个权重配置下**最终十条的类别分布明显不同**（偏点击的卡片里「视频/兴趣」占比上升，重负反馈的卡片更均衡、期望隐藏显著下降）——这就是「权重是产品决策」的量化体现。第二段显示朴素点击率与 IPW 估计的差距（模拟中朴素值系统性偏高），说明为什么必须做位置校正。

## 常见追问

- **追问**：停留时长涨了但点赞跌了，怎么判断是好事还是坏事？
  - 要点：看**构成**而不是总量——是被少数长内容拉高（可能伴随负反馈上升），还是整体互动都上升。把负反馈与长期留存作为护栏；必要时做长期 holdout 实验测留存。
- **追问**：新内容/新作者怎么冷启动？
  - 要点：内容侧特征（文本/视觉 embedding、作者历史）替代交互特征；给新内容固定的探索配额（bandit 或 ε 流量），并用「探索专用的无偏评估」衡量收益（串 [[consumer-ml-07]]）。
- **追问**：位置偏差能不能只靠把位置特征删掉解决？
  - 要点：不能。删掉位置特征并不改变样本的位置分布（$e_k$ 仍在），偏差依然存在；必须随机化位置或显式加权。
- **追问**：实时特征会不会让离线训练和在线服务不一致？
  - 要点：会，而且这是最常见的线上事故来源。要记录**特征快照**（训练时落盘当时的在线特征值）、对不可回填的实时特征做「在线-离线一致性检查」，并保留回退值（串 [[consumer-ml-04]]）。
- **追问**：怎么评估排序质量？
  - 要点：离线用 NDCG/多目标 AUC 做回归网（并做去偏），在线用 A/B 看多目标与护栏；长期用留存与满意度调查。三层缺一不可（串 [[consumer-ml-13]]）。
- **追问**：广告与自然内容为什么要联合排序？
  - 要点：分开排会导致「广告位把最优自然内容挤走、整体效用下降」；联合排序把广告的 eCPM 与自然内容的效用放在同一目标下，同时用约束保证体验（频率上限、位置规则），串 [[meta-17]]。

## 相关题目

- [[consumer-ml-06]]：两阶段候选生成与排序的通用版本，本题是它在「超大规模社交 feed」下的具体化。
- [[consumer-ml-03]]：位置偏差的生成模型与 IPW 推导，本题第 3 节直接建立在它上面。
- [[consumer-ml-13]]：上线后的漂移监控与重训触发，是 feed 排序长期运行的必备件。
- [[system-design-04]]：商品目录语义搜索的召回与排序设计，可对照「候选生成不必精排」这一分工。
- [[meta-17]]：广告排序与评估框架，与本题共用同一套漏斗但目标函数不同。

## 参考资料与归属

- **Wide & Deep Learning for Recommender Systems（延伸）** —— Cheng et al. (Google)，2016-06-24：<https://arxiv.org/abs/1606.07792>。第 2 节「记忆与泛化的联合建模」以及把宽线性部分与深度部分一起训练的动机来自这篇。
- **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction（延伸）** —— Guo et al.，2017-03-13：<https://arxiv.org/abs/1703.04247>。第 2 节「低阶与高阶特征交互同时建模、且共享输入」的设计来自这篇（CTR 预测场景）。
- **Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm（延伸）** —— Hu et al.，2018-09-16：<https://arxiv.org/abs/1809.05818>。第 3 节位置倾向 $e_k$ 与相关性 $r_d$ 相乘的生成模型、以及「同时估计点击与未点击倾向、成对联合学习去偏权重、并在商业搜索引擎上做在线 A/B」的做法来自这篇。
- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。第 1 节漏斗分层与第 4 节「先建指标与护栏、再谈模型复杂度」的工程取向参照这份清单。
- **延伸来源说明**：表 1 的延迟预算、第 1 节的各层规模、第 4 节的序列长度量级、以及两段可运行代码中的全部参数（200 条候选、Beta 分布参数、位置倾向分布 $e_k$、多样性折扣 0.35 等）都是按本仓库统一口径构造的工程算例与显式假设，不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
