---
type: question
id: meta-17
company: Meta（超级智能实验室、FAIR、Llama）
topic: system-design
order: 17
question: 为 Facebook Ads 设计推荐系统，以及一套广告排序的评估框架。
question_en: Design a recommendation system for Facebook Ads, and an evaluation framework for ads ranking.
asked_at: []
level: 高阶
tags: [系统设计, 广告排序, eCPM, 校准, 增量评估]
sources:
  - title: On Calibration of Modern Neural Networks（延伸）
    url: https://arxiv.org/abs/1706.04599
    author: Guo et al. (ICML 2017)
    published: 2017-06-13
  - title: Counterfactual Reasoning and Learning Systems（延伸）
    url: https://arxiv.org/abs/1209.2355
    author: Bottou et al.
    published: 2012-09-11
  - title: Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm（延伸）
    url: https://arxiv.org/abs/1809.05818
    author: Hu et al.
    published: 2018-09-16
  - title: Rules of Machine Learning: Best Practices for ML Engineering（延伸）
    url: https://developers.google.com/machine-learning/guides/rules-of-ml
    author: Martin Zinkevich (Google)
    published: 
related: [meta-16, consumer-ml-14, consumer-ml-13, evaluation-09, alibaba-08]
updated: 2026-09-28
---

## 一句话答案

> 广告排序与自然内容排序的结构相同（召回 → 精排 → 重排），但有三个本质差别：
> ① **目标函数里多了钱与约束**——排序分是 $\text{eCPM}=1000\times \hat p\text{CTR}\times \text{bid}$（或按转化目标换成 $\hat p\text{CVR}\times\text{CPA}$），而广告主还有**预算**，所以必须做 **pacing**（在一天/一个投放期内平滑花掉预算，而不是前两小时烧完）；
> ② **校准比排序更致命**——出价按你的预测计费，$\hat p\text{CTR}$ 系统性高估 20% 就意味着广告主多付 20%、平台补贴 20%，而且会扭曲竞价（高估的广告拿到不该拿的曝光）。可引用的实证是 *On Calibration of Modern Neural Networks*：现代深度网络（含 BatchNorm 与更深结构）普遍**置信度偏高**，温度缩放（temperature scaling）是简单有效的后处理校准手段；
> ③ **评估必须回到增量**——广告的价值是「因为投了广告而多出来的转化」，所以框架的中心是**增量口径**（holdout / ghost ads / 地理实验），而不是「看过广告的人转化率更高」这种混淆比较；这也正是反事实评估框架在广告场景里的原始动机（可引用 Bottou 等人的工作）。

## 面试官在考什么

- **结算方式与目标函数是否对得上**：CPM（按曝光）、CPC（按点击）、CPA/oCPX（按转化）三种结算下，排序分与校准要求各不相同。说得出「按 CPC 结算就要校准 pCTR，按 oCPX 就要校准 pCVR 且要有转化延迟的处理」才算过关。
- **校准问题的机制与修法**：能否说出深度网络过度自信的现象与来源（可引用 Guo et al. 的实证口径）、以及温度缩放/isotonic 回归/分桶校准的取舍；并知道**校准要在线上分布上做**（离线校准到线上会漂）。
- **预算与 pacing**：是否理解「预算约束改变了最优出价」——没有 pacing 时早晨就烧完预算，广告主拿不到全天流量，平台也损失收入。
- **评估框架的层次**：离线（AUC、校准误差、排序一致性）→ 在线（A/B：平台收入、广告主 ROI、用户体验护栏）→ 增量（holdout、ghost ads、geo 实验）→ 长期（广告主留存与预算增长）。缺一层就会被追问。
- **用户体验护栏**：广告加载率、负反馈、隐藏率、以及「广告与自然内容的相互挤压」都要有护栏指标，否则短期收入上涨会以长期体验为代价（串 [[meta-16]]）。

**常见错误答案**

- 「用 CTR 排序即可」——忽略出价与预算，等于把广告系统做成内容推荐器。
- 「AUC 高就是好模型」——AUC 只衡量排序，不衡量校准；而计费依赖校准（这是广告与推荐最本质的区别）。
- 「用看过广告用户的转化率减去没看过用户的转化率算增量」——两组用户本来就不一样（选择偏差），必须用随机化或 ghost ads 构造反事实。
- 「预算由广告主自己控制，系统不用管」——不做 pacing 的系统会让预算在早高峰耗尽，直接损害广告主效果与平台收入。

## 原理与推导

### 1. 排序分与结算

三种结算方式对应三种排序分（以曝光为排序单位）：

| 结算 | 广告主支付 | 排序分（eCPM） | 需要校准的量 |
| --- | --- | --- | --- |
| CPM | 每千次曝光固定价 | bid | 不需要（但要预估曝光质量） |
| CPC | 每次点击 | $1000\times\hat p\text{CTR}\times \text{bid}_{\text{CPC}}$ | $\hat p\text{CTR}$ |
| oCPX/CPA | 每次转化 | $1000\times\hat p\text{CTR}\times\hat p\text{CVR}\times \text{bid}_{\text{CPA}}$ | $\hat p\text{CTR}$ 与 $\hat p\text{CVR}$ |

**关键推论**：$\hat p$ 的偏差会直接变成钱。若真实 CTR 为 $p$、模型预测为 $\hat p=(1+\epsilon)p$，那么按 CPC 结算时每次点击的实际收费不变（按次计费），但**排序分会系统性高估**，于是该广告获得超出其真实价值的曝光，挤压了更高效的广告——**平台收入下降、广告主效果变差**，而 $\epsilon$ 只有校准能修。

### 2. 校准：为什么会高估，怎么修

Guo et al. 的实证口径是：现代深度网络（更深的结构、BatchNorm、更少的过拟合）比早期的小网络**校准更差**，表现为置信度系统性偏高；并且他们报告**温度缩放（在验证集上学一个标量温度 $T$）是最简单有效的方法**，同时保持准确率不变。

工程实现：
- 训练时用**负对数似然/交叉熵**（不只是 AUC 类的排序损失），让概率有语义；
- 后处理用**温度缩放**（单一参数、不易过拟合验证集）或 **isotonic 回归**（更灵活，但需要更多数据、且可能过拟合）；
- 监控**校准曲线**（分桶预测 vs 实际）与 ECE（期望校准误差），而不是只看 AUC；
- 校准必须按**细分人群/位置/版位**检查——整体校准好不代表每个细分都好（这也是分群监控的意义）。

### 3. 预算与 pacing

广告主给的是「一天 1000 元预算」，系统要决定**什么时候花**。朴素做法是按 eCPM 排序一路花，结果是早高峰耗尽预算。标准解法是对出价施加随预算消耗状态变化的调节因子：

$$\text{bid}'=\text{bid}\times \alpha(t,\ \text{spent}(t),\ \text{remaining})$$

$\alpha$ 随已花费比例上升而下降，使消耗曲线贴合全天流量曲线（probabilistic pacing 则用一个通过概率来随机丢弃竞价机会）。pacing 的评估指标是**预算消耗平滑度**与**错失机会率**，而它的副作用是可能降低竞价激烈时段的竞争力——所以要在「花完」与「花得有效」之间权衡。

### 4. 拍卖与体验约束

- **拍卖机制**：主流是广义第二价格（GSP）或类似机制，排序按 eCPM、计费按「刚好超过下一位所需的最小出价」；机制的细节会影响激励，工程师至少要理解**质量分进入排序**这件事本身就是平台对「广告相关性」的偏好表达。
- **体验约束**：广告加载率上限、频率上限（同一用户一天看到同一广告主几次）、负反馈（隐藏广告）作为惩罚项或硬约束；这些约束应当**在排序内部**表达，而不是事后删广告（事后删会破坏竞价一致性）。
- **冷启动**：新广告没有历史 CTR → 用创意特征（文本/图片 embedding）+ 广告主历史 + 探索配额；平台要给新广告固定的探索流量，否则永远拿不到数据。

### 5. 评估框架（本题的第二问）

四层，缺一层都会被追问：

| 层 | 指标 | 方法 |
| --- | --- | --- |
| 离线 | AUC/LogLoss、**校准误差 ECE**、分群一致性 | 按时间切分（不能随机切分）、按版位/人群分群 |
| 在线 | 平台：收入、eCPM、填充率；广告主：CTR/CVR、CPA；用户：负反馈、加载率 | 随机化 A/B，**预注册主指标与护栏** |
| 增量 | 转化增量、ROI、iROAS | holdout（保留一部分用户完全不投）、**ghost ads**（把自然结果当反事实对照）、geo 实验 |
| 长期 | 广告主留存、预算增长、生态健康 | 长期 holdout、季度视角 |

**增量评估的三个技术点**：① **ghost ads** 用「本来会投但被拦下」的对照来估计反事实，避免纯 holdout 的机会成本；② **转化延迟**（用户看到广告后 7 天才买）要求用回填后的数据，早期读数会系统性低估；③ **溢出与干扰**（同一用户被多个广告触达、预算在广告主之间竞争）会让简单 A/B 失真，需要 cluster/geo 随机化（串 [[evaluation-09]]）。

## 数值与代码验证

### 表 1：校准偏差对平台的实际影响（示例）

| 场景 | 真实 CTR | 预测 CTR | 相对偏差 | 后果 |
| --- | --- | --- | --- | --- |
| 校准良好 | 2.0% | 2.0% | 0% | 竞价与计费一致 |
| 高估 20% | 2.0% | 2.4% | +20% | 该广告排序分虚高、抢占更优广告的曝光 |
| 低估 20% | 2.0% | 1.6% | −20% | 好广告拿不到曝光，平台与广告主双输 |
| 分群不校准 | 整体正确，某人群 +50% | — | 分群偏差 | 该人群被过量投放，负反馈上升 |

### 表 2：pacing 对消耗曲线的影响（示例，全天预算 1000 元）

| 时段 | 流量占比 | 无 pacing 花费 | 有 pacing 花费 |
| --- | --- | --- | --- |
| 0–6 h | 10% | 100 元 | 100 元 |
| 6–12 h | 35% | **350 元（累计 450）** | 350 元 |
| 12–18 h | 30% | 300 元（预算在 14 h 耗尽） | 300 元 |
| 18–24 h | 25% | **0 元（已耗尽）** | 250 元 |

无 pacing 时晚间流量完全错过；有 pacing 时全天平滑。这就是「预算约束改变最优出价」的直观表现。

### 可运行代码

```python
# 1) 校准：温度缩放前后对比（在合成数据上复现「深度网络过度自信」）
import random, math
random.seed(9)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 合成一个过度自信的打分器：真实 logit z，预测 logit 放大 2 倍
data = []
for _ in range(20000):
    z = random.gauss(0, 1.4)
    p = sigmoid(z)
    y = 1 if random.random() < p else 0
    data.append((z, 2.0 * z, y))          # (真实 logit, 过度自信 logit, 标签)

def ece(data, idx, bins=15):
    tot, err = len(data), 0.0
    buckets = [[] for _ in range(bins)]
    for row in data:
        p = sigmoid(row[idx])
        buckets[min(bins - 1, int(p * bins))].append((p, row[2]))
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        err += len(b) / tot * abs(conf - acc)
    return err

def fit_temperature(data, lo=0.2, hi=5.0, steps=200):
    best, bestT = 1e9, 1.0
    for i in range(steps + 1):
        T = lo + (hi - lo) * i / steps
        nll = 0.0
        for _, z, y in data:
            p = min(max(sigmoid(z / T), 1e-9), 1 - 1e-9)
            nll -= y * math.log(p) + (1 - y) * math.log(1 - p)
        if nll < best:
            best, bestT = nll, T
    return bestT

print(f"校准前 ECE = {ece(data, 1):.4f}（预测 logit 被放大 2 倍 -> 过度自信）")
T = fit_temperature(data)
def calib(z):
    return z / T
data_cal = [(z, z / T, y) for z, z2, y in data]
print(f"拟合温度 T = {T:.3f}，校准后 ECE = {ece(data_cal, 1):.4f}")

# 2) 校准偏差如何改变竞价结果（两条广告争夺一个曝光位）
A = {"name": "广告A(校准好)", "p_true": 0.020, "p_pred": 0.020, "bid": 1.0}
B = {"name": "广告B(高估20%)", "p_true": 0.018, "p_pred": 0.0216, "bid": 1.0}
def ecpm(ad, key):
    return 1000 * ad[key] * ad["bid"]
rank = sorted([A, B], key=lambda a: -ecpm(a, "p_pred"))
print(f"\n按预测排序: {rank[0]['name']} 拿到曝光"
      f"（eCPM {ecpm(rank[0],'p_pred'):.2f} vs {ecpm(rank[1],'p_pred'):.2f}）")
rank_true = sorted([A, B], key=lambda a: -ecpm(a, "p_true"))
print(f"按真实价值排序: {rank_true[0]['name']} 才该拿曝光"
      f"（真实 eCPM {ecpm(rank_true[0],'p_true'):.2f} vs {ecpm(rank_true[1],'p_true'):.2f}）")
lost = ecpm(rank_true[0], "p_true") - ecpm(rank[0], "p_true")
print(f"平台每千次曝光的期望损失 = {lost:.3f}（用真实 CTR 计的 eCPM 差）")

# 3) 增量评估：为什么「看过广告的人转化更高」是错的
base_cvr, lift, p_exposed = 0.030, 0.004, 0.35
naive = base_cvr + lift                          # 只看被曝光人群
real = base_cvr + lift * p_exposed / 1.0         # 全局增量（示意：只有被曝光者才有提升）
print(f"\n朴素比较（曝光人群 CVR） {naive:.4f} vs 全局 CVR {real:.4f}"
      f" —— 差 {naive/real-1:.1%}，多出来的部分就是选择偏差")
```

预期输出要点：校准前 ECE 明显偏高，拟合温度 $T>1$（合成数据里约 2）后 ECE 大幅下降——这就是「深度网络过度自信 + 温度缩放能修」的最小复现；第 2 段演示**高估 20% 的广告抢走了曝光**，并给出平台按真实价值计的期望损失；第 3 段说明用曝光人群的 CVR 当增量会系统性高估（示例中约 12%），这就是必须用 holdout/ghost ads 的原因。

## 常见追问

- **追问**：AUC 高但收入没涨，可能是什么原因？
  - 要点：校准偏差（最可能）、排序目标与收入目标不一致（例如没有把出价与预算纳入）、以及**竞价环境变化**（你的模型变了，别人的出价也变了，均衡位置未必改善）。先看校准曲线与分群 ECE，再看竞价仿真。
- **追问**：怎么处理转化延迟？
  - 要点：训练用**延迟回填**的标签（并按时间窗截断），监控早期读数与最终读数的偏差；对 oCPX 用延迟校正模型估计「现在还没转化但最终会转化」的概率；评估时等标签成熟再下结论。
- **追问**：ghost ads 为什么比纯 holdout 更好？
  - 要点：纯 holdout 会损失真实收入（机会成本高）；ghost ads 记录「如果没有这个广告，本该发生的自然结果」，用同一用户的自然行为做反事实对照，既省流量又能控制选择偏差。
- **追问**：新广告冷启动怎么办？
  - 要点：创意 embedding + 广告主历史 + 相似广告迁移 + 固定探索配额；同时用**乐观初始化**（置信上界式探索）避免新广告被永久埋没，并给探索设置体验护栏。
- **追问**：广告加载率上升但收入涨了，怎么判断该不该继续？
  - 要点：看长期护栏——用户负反馈、会话时长、次月留存；广告收入的短期上升若伴随体验指标恶化，就要用长期 holdout 量化代价（串 [[consumer-ml-14]]）。
- **追问**：竞价机制要不要建模对手？
  - 要点：多数工程实践把它当作环境（不显式建模对手），重点放在校准与 pacing；但要知道机制变化（如从 GSP 到一价/改良机制）会改变最优出价策略，这类改动必须用仿真 + 小流量实验验证。

## 相关题目

- [[meta-16]]：自然内容 feed 排序，与本题共用漏斗结构但目标函数不同，对照读能看清「钱进入目标函数」的差别。
- [[consumer-ml-14]]：业务价值的增量归因方法（A/B 与样本量），是本题评估框架的基础。
- [[consumer-ml-13]]：模型漂移监控与重训触发，广告场景里校准漂移尤其频繁。
- [[evaluation-09]]：在线实验设计的随机化单位与护栏，geo 实验部分与本题的增量评估直接相关。
- [[alibaba-08]]：发布时如何让指标可信（去污染、口径声明），与广告里的「校准与增量」是同一类诚实性问题。

## 参考资料与归属

- **On Calibration of Modern Neural Networks（延伸）** —— Guo et al. (ICML 2017)，2017-06-13：<https://arxiv.org/abs/1706.04599>。第 2 节关于现代深度网络校准变差、置信度系统性偏高、以及温度缩放作为简单有效后处理手段的实证结论来自这篇。
- **Counterfactual Reasoning and Learning Systems（延伸）** —— Bottou et al.，2012-09-11：<https://arxiv.org/abs/1209.2355>。第 5 节广告增量评估的反事实框架（把投放系统视为与环境交互的学习系统，用因果推断预测改动后果）来自这篇。
- **Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm（延伸）** —— Hu et al.，2018-09-16：<https://arxiv.org/abs/1809.05818>。第 5 节里「曝光与点击都受旧策略影响、因此需要去偏」的论证与在线 A/B 验证口径来自这篇。
- **Rules of Machine Learning: Best Practices for ML Engineering（延伸）** —— Martin Zinkevich (Google)：<https://developers.google.com/machine-learning/guides/rules-of-ml>。第 5 节「先定义指标与护栏、按时间切分而非随机切分」的评估纪律参照这份清单。
- **延伸来源说明**：表 1、表 2 的数值、第 1 节的结算-排序分对照、第 3 节的 pacing 形式、以及三段可运行代码（温度缩放、竞价排序、增量偏差）中的全部参数与结论都是按本仓库统一口径构造的工程算例与显式假设（合成数据、费率为示意），不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
