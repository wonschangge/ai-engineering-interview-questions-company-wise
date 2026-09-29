---
type: question
id: meta-18
company: Meta（超级智能实验室、FAIR、Llama）
topic: system-design
order: 18
question: 为某个 Instagram Story 功能设计其背后的 ML 组件。
question_en: Design the ML components behind an Instagram Story feature.
asked_at: []
level: 高阶
tags: [系统设计, Story, 冷启动, 时效衰减, 内容理解]
sources:
  - title: Wide & Deep Learning for Recommender Systems（延伸）
    url: https://arxiv.org/abs/1606.07792
    author: Cheng et al. (Google)
    published: 2016-06-24
  - title: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction（延伸）
    url: https://arxiv.org/abs/1703.04247
    author: Guo et al.
    published: 2017-03-13
  - title: Learning Transferable Visual Models From Natural Language Supervision（CLIP）（延伸）
    url: https://arxiv.org/abs/2103.00020
    author: Radford et al. (ICML 2021)
    published: 2021-02-26
  - title: A Holistic Approach to Undesired Content Detection in the Real World（延伸）
    url: https://arxiv.org/abs/2208.03274
    author: Markov et al. (OpenAI, AAAI 2023)
    published: 2022-08-05
related: [consumer-ml-07, system-design-05, multimodal-01, meta-16, consumer-ml-10]
updated: 2026-09-28
---

## 一句话答案

> Story 的 ML 组件要拆成**四块**，因为它们的输入、目标和延迟预算都不同：
> ① **托盘排序（tray ranking）**——决定「先看谁的故事」，输入是关系亲密度、未读计数、上次观看时间、以及作者与我的互动历史，本质是一个「作者级」排序问题；
> ② **故事内排序与广告插入**——决定「先看这一段还是那一段」、以及广告放在第几位，本质是「素材级」排序 + 位置约束；
> ③ **理解层**——把素材变成可排序、可检索、可审核的表示（视觉 embedding、文字 OCR、音乐/贴纸识别），这是 Story 与其他 feed 最大的不同；
> ④ **反馈与安全**——看完/跳过/快速划走的细粒度信号、静音与屏蔽、以及上传期与展示期的双层内容审核。
> 而 Story 的**决定性特点**是 24 小时生命周期：内容新鲜度极高、新素材占比大，所以**内容侧先验特征（理解层）比交互特征更早可用**，冷启动是常态而不是例外（串 [[consumer-ml-07]]）。

## 面试官在考什么

- 能否把「设计一个 Story 功能」**拆成可建模的子问题**，并给每个子问题配信号与指标，而不是笼统说「用推荐模型排序」。
- 是否理解 **24 小时生命周期带来的连锁后果**：交互数据极少（一条 story 可能只有几十次观看）、衰减极快、模型必须靠**内容特征 + 作者特征 + 关系特征**在几小时内给出可用排序。
- **关系信号**（亲密度）与**内容信号**（素材质量）的分工：托盘排序几乎全靠关系，故事内排序更依赖内容与时效。
- **细粒度反馈**：Story 里最重要的信号之一是「快速划走」（skip within <1 s），它比「看完」更常见也更有信息量；能否把它建模成显式的负向信号，而不是只统计完成率。
- **安全与体验**：上传期（违规直接拦）与展示期（按观看者年龄/地区动态决定）的分层审核，以及未成年人与敏感内容保护（串 [[system-design-05]]）。
- 是否考虑到**广告插入的约束**：位置、频率上限、以及与自然素材的体验平衡（串 [[meta-17]]）。

**常见错误答案**

- 「把 Story 当成普通 feed 排序」——忽略 24 小时时效与关系优先的结构。
- 「冷启动用作者历史就行」——新作者、转发内容、合作发布都会让作者特征失效，必须有内容侧表示。
- 「审核只在上传时做一次」——同一条内容对不同观看者的合规要求不同（年龄、地区），展示期必须再判一次。
- 「用完成率当唯一指标」——完成率会被短素材刷高，必须配合「快速划走率」「静音率」「回复率」一起看。

## 原理与推导

### 1. 托盘排序（作者级）

候选是「有未看 story 的作者」，规模是好友/关注数（$10^2\sim10^3$），排序目标不是点击而是**观看完成与互动的组合**。核心特征：

- **关系强度**：历史互动（私信、评论、点赞、@提及）的加权计数，最好用**双向**信号（我→他 与 他→我 都要看）；
- **时效**：新发布的 story 优先，但要用**剩余时间**而不是发布时间（24 小时窗口下，剩余 20 小时和剩余 1 小时的价值完全不同）；
- **未读计数**：还有几段没看（未读多说明「值得优先」或「看不完」，要区分这两种情况）；
- **负向**：静音、上次跳过该作者、以及「上次看了但从未互动」。

排序分的一个可用骨架：

$$\text{score}(a)=\underbrace{f_{\text{aff}}(u,a)}_{\text{亲密度}}\times\underbrace{g(\Delta t_{\text{post}})}_{\text{时效衰减}}\times\underbrace{h(n_{\text{unseen}})}_{\text{未读量}}\times\underbrace{(1-\hat p_{\text{skip}}(u,a))}_{\text{不跳过的概率}}$$

可引用的模型侧依据是 Wide & Deep 的「记忆 + 泛化」联合建模与 DeepFM 的「低阶 + 高阶交互同时建模」——关系信号（低阶、稀疏、强记忆）与内容/上下文信号（高阶）放进同一个模型，比手工拼公式更稳。

### 2. 素材级排序与广告插入

故事内素材数通常 $1\sim10^2$，排序要处理：

- **素材质量**：清晰度、是否含人脸/文字、是否重复（近似重复检测）；
- **观看进度**：上一段是否看完（决定从哪开始）；
- **广告位**：通常限制在固定位置（例如两段自然素材之间），频率上限按用户/天计。广告与自然素材**联合排序**（同一目标函数下用 eCPM 与自然效用比较），而不是先排自然再塞广告。

### 3. 理解层：Story 独特性的来源

| 任务 | 输出 | 用途 |
| --- | --- | --- |
| 视觉嵌入 | 素材向量 | 近似重复聚类、相似推荐、去重 |
| OCR/文字检测 | 文本 | 排序特征、违规文本审核、可访问性 |
| 音乐/贴纸识别 | 实体标签 | 音乐推荐、版权、趋势发现 |
| 场景/物体分类 | 标签 | 安全策略（武器、裸露）、兴趣推断 |

技术选择上，**图文对比学习得到的通用嵌入（可引用 CLIP 的做法：用自然语言监督训练可迁移的视觉表示）**很适合这里：它不需要为每个标签单独标注，就能支持「文本查询 → 素材」的检索与聚类，这对每天海量新素材的冷启动很关键。

### 4. 反馈与安全

- **信号**：看完、快速划走、回复、截图、静音作者、举报、隐藏。快速划走是最丰富的负向信号（比举报常见几个数量级），要做成显式标签并防作弊（用户随手划过不代表内容差）。
- **审核分层**：**上传期**做高召回拦截（宁可误拦，代价是创作者体验受损），**展示期**按观看者属性动态判定（年龄、地区、是否有敏感历史）。可引用的工程做法来自 *A Holistic Approach to Undesired Content Detection in the Real World*：把分类器的输出与政策规则结合、并用**按危害类别分层的准确率/召回率**而不是单一总体指标来评估与发布。
- **未成年保护**：默认更严格的可见性、限制推荐扩散、对私信与评论做额外过滤；这些是策略约束，必须能在排序层强制生效。

### 5. 与 feed 排序的关键差别（答题时的对比锚点）

| 维度 | Feed（帖子） | Story |
| --- | --- | --- |
| 生命周期 | 长期可复用（累积互动） | 24 小时（数据极少） |
| 排序单元 | 帖子（内容级） | 作者（托盘）+ 素材（故事内） |
| 主要信号 | 内容与历史互动 | 关系亲密度 + 时效 |
| 反馈粒度 | 点赞/评论/停留 | 快速划走/回复/静音 |
| 冷启动 | 常见但可缓解 | **常态** |

## 数值与代码验证

### 表 1：24 小时窗口下的时效衰减（示例）

| 发布至今 | 剩余时间 | 衰减因子 $g$ | 说明 |
| --- | --- | --- | --- |
| 0.5 h | 23.5 h | 1.00 | 刚发布，最高优先 |
| 6 h | 18 h | 0.72 | 仍新鲜 |
| 12 h | 12 h | 0.50 | 一半寿命 |
| 20 h | 4 h | 0.17 | 即将过期（要抓紧曝光） |
| 23.5 h | 0.5 h | 0.02 | 基本不再展示 |

口径：$g=\left(1-\Delta t/24\right)^{2}$（二次衰减是示例选择，可用数据拟合）。**读法**：剩余时间越少越要「抢救曝光」，但也不能让即将过期的内容压过新鲜内容——所以要用剩余时间的**边际收益**而不是单调的紧迫度。

### 表 2：延迟预算（示例）

| 阶段 | 预算 | 说明 |
| --- | --- | --- |
| 托盘候选与特征 | 25 ms | 关系特征 + 未读计数（缓存友好的聚合表） |
| 托盘排序 | 15 ms | 作者级排序，规模 $10^2\sim10^3$ |
| 素材特征与排序 | 30 ms | 含理解层嵌入的查表（离线算好） |
| 审核判定（展示期） | 10 ms | 策略命中查表，复杂案例异步复核 |
| 广告插入 | 10 ms | 联合排序 + 频率约束 |

### 可运行代码

```python
# 1) 托盘排序：时效衰减与关系强度的交互，以及「快速划走」的负向作用
import math, random
random.seed(17)
authors = [{"id": i,
            "aff": random.betavariate(2, 5),          # 关系强度 0-1
            "age_h": random.uniform(0.2, 23.5),       # 发布至今小时
            "unseen": random.randint(1, 12),
            "p_skip": random.betavariate(2, 6)}       # 快速划走概率
           for i in range(300)]

def decay(age_h, tau=24.0, power=2.0):
    return max(0.0, (1 - age_h / tau)) ** power

def tray_score(a, w_skip=1.0, w_unseen=0.15):
    unseen = 1 + w_unseen * math.log1p(a["unseen"])
    skip_pen = (1 - a["p_skip"]) ** w_skip
    return a["aff"] * decay(a["age_h"]) * unseen * skip_pen

ranked = sorted(authors, key=tray_score, reverse=True)[:8]
print("托盘 top-8（含衰减与跳过硬惩罚）：")
for a in ranked:
    print(f"  作者{a['id']:>3} 亲密度{a['aff']:.2f} 发布{a['age_h']:5.1f}h 未读{a['unseen']:>2} "
          f"跳走率{a['p_skip']:.2f} -> {tray_score(a):.4f}")

# 关掉衰减看排序变化（说明时效在 Story 里有多重要）
ranked_no_decay = sorted(authors, key=lambda a: tray_score({**a, "age_h": 0.0}), reverse=True)[:8]
overlap = len({a["id"] for a in ranked} & {a["id"] for a in ranked_no_decay})
print(f"\n与「不衰减」排序的 top-8 重合度：{overlap}/8 —— 衰减改变了 {8-overlap} 个位置")

# 2) 冷启动：新素材占比与可用信号的时间线
DAILY_STORIES, VIEWS_PER_STORY_FIRST_HOUR = 500_000_000, 30
print(f"\n若每天新增 {DAILY_STORIES:,} 条 story，首小时平均只有 {VIEWS_PER_STORY_FIRST_HOUR} 次观看，"
      f"则 1 小时内能积累的交互样本只有 {DAILY_STORIES*VIEWS_PER_STORY_FIRST_HOUR:,} 条"
      f"（分摊到每条 <1 次有效负反馈）")
print("结论：排序在首小时必须靠内容与关系特征，交互特征要等几小时后才有统计意义")

# 3) 审核分层的代价：上传期高召回 vs 展示期按观看者判定
def layered_review(items, upload_recall=0.95, show_recall_extra=0.6, fp_rate=0.03):
    blocked, missed = 0, 0
    for it in items:
        if it["bad"] and random.random() < upload_recall:
            blocked += 1                      # 上传期拦下
        elif it["bad"] and random.random() < show_recall_extra:
            blocked += 1                      # 展示期拦下
        elif it["bad"]:
            missed += 1
        elif random.random() < fp_rate:
            blocked += 1                      # 误拦
    return blocked, missed

items = [{"bad": random.random() < 0.02} for _ in range(20000)]
b, m = layered_review(items)
bad_total = sum(1 for it in items if it["bad"])
print(f"\n审核分层（2% 违规）：拦下 {b:,}，漏放 {m:,}，"
      f"违规召回 {1-m/bad_total:.1%}，误拦约 {b-(bad_total-m):,}（{ (b-(bad_total-m))/len(items):.2%}）")
```

预期输出要点：托盘排序里**衰减把「发布 20 小时以上的高亲密度作者」压到后面**，与不衰减排序的 top-8 明显不同（示例中会换掉若干位置）——说明时效是 Story 排序的一等公民；冷启动段给出量级直觉：每天 5 亿条 story、首小时平均 30 次观看，意味着**交互信号在首小时基本不可用**；审核段展示分层召回（上传期 95% + 展示期补 60%）后的总体召回与误拦率，说明为什么要分两层而不是只在上传时判一次。

## 常见追问

- **追问**：Story 和 Reels/Feed 的排序能共用一套模型吗？
  - 要点：可以共享底层（用户/作者表示、内容嵌入），但**排序头必须分开**——生命周期、排序单元、反馈粒度都不同。共享底层 + 独立头是多产品场景的标准做法。
- **追问**：怎么衡量「快速划走」是内容差还是用户手滑？
  - 要点：用停留时间做门槛（例如 <0.5 s 单独一档）、与同一用户的历史行为对比（是否对所有内容都快划）、以及只把「同一作者/同一类内容被反复快划」当作强负信号。
- **追问**：24 小时过期后数据就丢了，模型怎么学？
  - 要点：把素材的**聚合统计**与**嵌入**长期保留（内容侧表示可复用），只把逐条交互日志按隐私策略处理；这样新素材可以复用「相似素材」的历史表现。
- **追问**：如何做 Story 的冷启动推荐（新用户）？
  - 要点：先用托盘的关系先验（互关/通讯录/同城），再快速用少量交互做在线更新；对完全无信号的新用户，用内容热度与多样性配额，避免只用「最热」。
- **追问**：广告插在第几位、放几条，怎么定？
  - 要点：用体验护栏（负反馈、完播率）与收入联合优化，位置与频次作为约束；做位置实验时要考虑**位置本身的偏差**（串 [[consumer-ml-03]]），不能只看原始 CTR。
- **追问**：审核误拦创作者内容怎么补偿？
  - 要点：分级处理（硬拦 / 降权 / 打标等待复核）、快速申诉通道、以及把误拦率作为审核系统的发布门禁指标（串 [[system-design-05]]）。

## 相关题目

- [[consumer-ml-07]]：内容推荐的冷启动方法（内容特征 + 探索），Story 的冷启动比它更极端。
- [[system-design-05]]：分类器与 LLM 结合的内容审核设计，是本题审核分层部分的完整版。
- [[multimodal-01]]：把图像接进 LLM 的三条路线与视觉 token 成本，理解层直接建在它上面。
- [[meta-16]]：Feed 排序的四层漏斗与位置偏差处理，可与本题的托盘/素材两级排序对照。
- [[consumer-ml-10]]：视觉相似检索（近似重复检测、去重），是 Story 素材去重与相似推荐的算法基础。

## 参考资料与归属

- **Wide & Deep Learning for Recommender Systems（延伸）** —— Cheng et al. (Google)，2016-06-24：<https://arxiv.org/abs/1606.07792>。第 1 节「关系类稀疏特征（记忆）与内容/上下文特征（泛化）联合建模」的依据来自这篇。
- **DeepFM: A Factorization-Machine based Neural Network for CTR Prediction（延伸）** —— Guo et al.，2017-03-13：<https://arxiv.org/abs/1703.04247>。第 1 节「低阶与高阶交互同时建模」的对照依据来自这篇。
- **Learning Transferable Visual Models From Natural Language Supervision（CLIP）（延伸）** —— Radford et al. (ICML 2021)，2021-02-26：<https://arxiv.org/abs/2103.00020>。第 3 节「用自然语言监督训练可迁移视觉表示、无需为每个标签单独标注」的做法与动机来自这篇。
- **A Holistic Approach to Undesired Content Detection in the Real World（延伸）** —— Markov et al. (OpenAI, AAAI 2023)，2022-08-05：<https://arxiv.org/abs/2208.03274>。第 4 节「把分类器与政策规则结合、按危害类别分层评估」的工程做法来自这篇。
- **延伸来源说明**：表 1、表 2 的数值、第 1 节的排序分骨架、以及三段可运行代码中的全部参数（300 位作者、衰减指数 2、每天 5 亿条 story、上传期 95% 与展示期 60% 召回等）都是按本仓库统一口径构造的工程算例与显式假设，不是上述来源的原文数字，也不代表该产品的真实规模；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
