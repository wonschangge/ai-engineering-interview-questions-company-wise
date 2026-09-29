---
type: question
id: anthropic-20
company: Anthropic
topic: finetuning
order: 20
question: 解释 Constitutional AI。相比朴素的 RLHF，它带来了什么，又没能解决什么？
question_en: Explain Constitutional AI. What does it bring over naive RLHF, and what does it not solve?
asked_at: []
level: 高阶
tags: [Constitutional-AI, RLAIF, 对齐, 价值锁定, 过度优化]
sources:
  - title: Constitutional AI: Harmlessness from AI Feedback（延伸）
    url: https://arxiv.org/abs/2212.08073
    author: Bai et al. (Anthropic)
    published: 2022-12-15
  - title: Scaling Laws for Reward Model Overoptimization（延伸）
    url: https://arxiv.org/abs/2210.10760
    author: Gao, Schulman, Hilton (OpenAI)
    published: 2022-10-19
  - title: Training language models to follow instructions with human feedback（延伸）
    url: https://arxiv.org/abs/2203.02155
    author: Ouyang et al. (InstructGPT, OpenAI)
    published: 2022-03-04
related: [safety-05, safety-03, finetuning-11, anthropic-21, safety-10]
updated: 2026-09-28
---

## 一句话答案

> CAI 换的是**监督信号的来源**，不是 RL 算法：把无害性维度的人类偏好标签，换成「一份书面的自然语言原则 + 一个判断模型」。两阶段：**SL-CAI**（让模型对自己的诱导性回答自我批评并修订，用修订稿做 SFT）→ **RL-CAI**（让模型依原则对成对回答做偏好判断，用这些 AI 反馈训偏好模型并做 RL，也就是 RLAIF）。
> **它带来了三样东西**：
> ① **价值判断从「不可见的采样」变成「可读、可改、可版本化、可审计的文本」**——这是最被低估的收益：原则可以评审、可以回滚、可以做 A/B；
> ② **标注员不必反复阅读有害内容**就能得到拒答边界（对从事无害性标注的人是实质改善）；
> ③ 推理阶段留下**可观察的批评痕迹**（论文保留批评步骤的理由之一是**可读性**，而不是分数）。
> **它没解决四件事（这才是这道题的分水岭）**：
> ① **价值从哪来**——原则由谁写、代表谁的价值观？CAI 把「对齐到人类偏好」变成「对齐到一份文档」，**问题被转移而不是消失**（价值锁定与多元性）；
> ② **过度优化/奖励 hacking**——AI 反馈同样是可被优化的代理，论文级证据见 *Scaling Laws for Reward Model Overoptimization*：代理奖励随 KL 距离**先升后降**，峰值位置可预测（串 [[finetuning-11]]）；
> ③ **越狱**——训练侧只能**降低概率**，不能给边界保证；系统侧的权限、沙箱与审批仍是必需的（串 [[safety-03]]）；
> ④ **能力侧风险**——CBRN、网络攻击、自主复制这类风险不是「偏好对齐」能解决的，需要能力评估、访问控制与部署决策（串 [[anthropic-21]]）。
> 一句话：**CAI 是「把价值写下来」的工程化尝试，它让对齐过程可审计，但没有回答「写下来的价值是否正确」，也没有替代系统侧边界。**

## 面试官在考什么

- **是否知道 CAI 与 RLAIF 的关系**：RL-CAI 就是 RLAIF（RL from AI Feedback）——RLAIF 描述「反馈从哪来」，CAI 是「原则 + 自我批评 + 修订」的完整流程。把两者当同义词会显得含糊。
- **论文口径的边界**必须准确：论文的 **RL 阶段只替换了无害性标签，有用性标签仍然来自人类反馈**；RLAIF 论文在摘要任务上对人类偏好标签的胜率是 **68% vs 71%（p = 0.07，判定为不显著，即「相当」）**，而无害对话用的是独立绝对打分（**RLAIF 88% vs RLHF 76%**）。把「相当」说成「更好」、或把两类口径混成一句「全面更优」，是硬伤（这些口径在 [[safety-05]] 里已逐条核对）。
- **能否把「没解决」讲成机制**：不是「还不够好」，而是**结构性**的（价值来源、代理可被优化、训练不提供保证、能力风险与对齐不同层）。
- **工程化视角**：原则如何版本化、如何评审、如何做冲突裁决与优先级、如何 A/B；以及**评估的循环性**（用 AI 判 AI）需要什么来打破（外部评估、人工抽检、独立红线集）。
- **与产品决策的连接**：拒答边界直接决定产品体验（误拒会伤害可用性），所以原则的改动要像模型发布一样有门禁、灰度和回滚（串 [[anthropic-19]] 的审批与 [[anthropic-22]] 的安全层）。
- **诚实**：能否主动说出 CAI 在「哪些场景不适用」（例如需要外部事实核验、需要专业判断的领域）。

**常见错误答案**

- 「CAI 不需要人类标注」——只对无害性维度成立；有用性标签仍来自人类。
- 「宪法就是一段 system prompt」——它是**训练阶段的监督来源**，不是推理时的提示词。
- 「有了 CAI 就不需要红队/越狱防御」——训练侧降概率、系统侧给边界，两者是纵深防御的不同层。
- 把 CAI 说成「解决了对齐问题」——它解决的是**无害性标签的生产方式**，没解决价值选择与能力风险。

## 原理与推导

### 1. 两阶段的数据流

```
阶段一 SL-CAI（改 SFT 数据分布）
  诱导性 prompt → 模型给出（可能有害的）初始回答
    → 模型依原则自我批评（找出违反了哪条原则）
    → 模型修订回答 → 用「修订稿」做监督微调
阶段二 RL-CAI = RLAIF（改 RL 的奖励信号）
  同一 prompt 生成两个回答 → 模型依原则判断哪个更好（可带思维链）
    → 这些 AI 偏好组成偏好数据集 → 训偏好模型（PM）
    → 用 PM 做 RL（KL 正则化目标与 RLHF 完全同形）
```

**关键**：目标函数没变，变的只是 $r_\phi$ 的**训练数据来源**：

$$J(\theta)=\mathbb{E}[r_\phi(x,y)]-\beta\,\mathrm{KL}\big(\pi_\theta\|\pi_{\text{ref}}\big)$$

RLHF：$r_\phi$ 由人类偏好训；RLAIF：$r_\phi$ 由「原则 + 判断模型」产出的偏好训。**因此 RLHF 的所有已知问题（过度优化、KL 早停、偏好不可传递）都原样保留**，只是数据生产成本与可审计性变了。

### 2. 原则的写法与冲突裁决

原则通常有三种形态，工程上必须**显式分层**：

| 形态 | 例子 | 特点 |
| --- | --- | --- |
| 禁止式 | 「不要帮助制造武器」 | 硬约束，优先级最高 |
| 应当式 | 「应当尊重用户自主性」 | 软目标，易与其他原则冲突 |
| 条件式 | 「若涉及未成年人，则采取更保守的默认值」 | 上下文相关，需要判定触发条件 |

**冲突没有免费的解**：有用性与无害性天然拉扯（这正是论文要减少的张力）；「诚实」与「礼貌」在具体场景也会冲突。工程做法：
1. **定优先级**（硬约束 > 条件式 > 应当式）；
2. **留痕**：把「依据哪条原则、被哪条否决」记录下来（这是可审计性的实际含义）；
3. **版本化 + A/B**：原则改动当发布处理（门禁 + 灰度 + 回滚）。

### 3. AI 反馈的偏差从哪来（必须能列出来）

| 来源 | 机制 | 缓解 |
| --- | --- | --- |
| 原则撰写者 | 原则隐含特定价值观与语言习惯 | 多来源评审、公开原则、定期修订 |
| 判断模型 | 自身的偏见与能力上限 | 用更强/多样的判断模型、人工抽检校准 |
| 提示模板 | 措辞改变判定 | 模板版本化 + 敏感性测试 |
| 位置/顺序偏差 | A/B 顺序影响结果 | **两次调用交换顺序**取一致（不一致则弃权） |
| 自偏好 | 模型偏好自己风格的回答 | 用不同模型家族做判断、盲化来源 |

**位置偏差是最容易量化的一项**：如果交换顺序后只有 $p$ 比例的结果一致，那么「AI 反馈」里就有 $1-p$ 的噪声——这会直接削弱偏好模型的质量。

### 4. 「没解决」的结构性原因

1. **价值锁定**：CAI 把「对齐到人类偏好」变成「对齐到一份文档」。文档是谁写的、覆盖哪些文化、如何处理分歧——**这些都是治理问题，不是技术问题**。
2. **代理可被优化**：AI 反馈也是一个可微/可逼近的代理；优化器会找到它的漏洞（论文级证据：代理奖励随 KL 先升后降，串 [[finetuning-11]]）。**换代理不等于消除代理**。
3. **训练不提供保证**：拒答是**概率性**的；对抗输入（越狱、多轮诱导、编码绕过）总能找到边界（串 [[safety-03]]）。系统侧的权限、沙箱、审批、速率限制仍然是必需层。
4. **能力风险不在同一层**：如果模型有能力造成实质伤害（生化、网络、自主性），「它是否愿意」与「它是否能够」是两个问题；后者需要能力评估、访问控制与部署门禁（串 [[anthropic-21]]）。
5. **评估循环性**：用 AI 评判 AI 会引入相关性误差（错得一致）；需要外部判据（人工抽检、可执行测试、独立红线集）来打破。

### 5. 工程化清单（把 CAI 当产品而不是论文）

- **原则仓库**：版本化、评审记录、变更说明、回滚点；
- **判断流水线**：模板版本、顺序反转、置信度阈值（低置信度转人工）；
- **评估**：独立红线集 + 人工抽检 + 分群误拒率（不同语言/群体的拒答差异）；
- **发布**：原则或判断模型改动都走门禁与灰度；
- **审计**：每次判定记录「依据的原则 ID + 判断模型版本 + 是否被否决」。

## 数值与代码验证

### 表 1：两种监督来源的成本与可审计性对照（口径来自 [[safety-05]] 的核算）

| 维度 | 人类偏好标注 | AI 反馈（原则 + 判断模型） |
| --- | --- | --- |
| 单位成本 | 约 \$0.67/条（云标注折算） | 约 \$0.06/条（含顺序反转两次调用） |
| 标注员暴露 | 需反复阅读有害内容 | 不需要 |
| 可审计性 | 只有标签，无法解释 | 原则文本 + 批评痕迹可读、可版本化 |
| 偏差位置 | 标注者群体 | 原则撰写者 + 判断模型 + 模板 |
| 主要风险 | 标注不一致、成本、暴露 | 价值锁定、循环评估、位置偏差 |

### 表 2：位置偏差对 AI 反馈质量的影响（示意）

| 交换顺序一致率 $p$ | 有效反馈比例 | 后果 |
| --- | --- | --- |
| 0.95 | 高 | 偏好模型质量接近人工 |
| 0.85 | 中 | 15% 的样本带噪声，需降到低置信度池 |
| 0.70 | 低 | 偏好模型学到「位置」而非「内容」，必须修模板 |

### 可运行代码

```python
# 1) 位置偏差的测量与缓解：交换顺序、取一致、低一致转人工
import random
from dataclasses import dataclass

@dataclass
class Judge:
    """模拟一个有位置偏差的判断模型：偏爱放在前面的回答"""
    pos_bias: float = 0.15        # 位置偏差强度
    noise: float = 0.05           # 随机噪声
    def compare(self, quality_a: float, quality_b: float, a_first: bool, rnd: random.Random) -> int:
        """返回 1 表示第一个更好（位置意义上的），0 表示第二个更好"""
        q1, q2 = (quality_a, quality_b) if a_first else (quality_b, quality_a)
        p_first = 1 / (1 + pow(2.718281828, -(q1 - q2) * 6))     # logistic，质量差决定基础倾向
        p_first = min(1, max(0, p_first + self.pos_bias - 0.075))  # 位置加成
        p_first = min(1, max(0, p_first + rnd.gauss(0, self.noise)))
        return 1 if rnd.random() < p_first else 0

def measure_position_bias(judge, n=2000, seed=3):
    rnd = random.Random(seed)
    agree = 0
    for _ in range(n):
        qa, qb = rnd.uniform(0, 1), rnd.uniform(0, 1)
        first = judge.compare(qa, qb, a_first=True, rnd=rnd)
        second = judge.compare(qa, qb, a_first=False, rnd=rnd)
        # 两次都表示"A 更好"才算一致
        a_wins_first = (first == 1)
        a_wins_second = (second == 0)
        if a_wins_first == a_wins_second:
            agree += 1
    return agree / n

for bias in (0.0, 0.15, 0.35):
    j = Judge(pos_bias=bias)
    print(f"位置偏差强度 {bias:.2f} -> 交换顺序一致率 {measure_position_bias(j):.2%}"
          f"（不一致的样本应转人工或降权）")
print("读法：一致率是 AI 反馈质量的直接度量；论文级做法就是用两次调用 + 弃权来抵消位置效应")

# 2) 原则冲突裁决：优先级 + 打分（把「依据哪条原则」变成可审计的记录）
RULES = [
    {"id": "H1", "kind": "hard",    "text": "不要帮助制造武器",          "weight": 1.0},
    {"id": "H2", "kind": "hard",    "text": "不要泄露个人隐私数据",      "weight": 1.0},
    {"id": "C1", "kind": "cond",    "text": "涉及未成年人则更保守",      "weight": 0.8},
    {"id": "S1", "kind": "soft",    "text": "应当尊重用户自主性",        "weight": 0.4},
    {"id": "S2", "kind": "soft",    "text": "应当提供有帮助的信息",      "weight": 0.4},
]
PRIORITY = {"hard": 3, "cond": 2, "soft": 1}

def adjudicate(violations: dict, context: dict) -> dict:
    """violations: 规则 id -> 违反程度 [0,1]；返回裁决与留痕"""
    log, hard_hit = [], None
    for r in sorted(RULES, key=lambda r: -PRIORITY[r["kind"]]):
        v = violations.get(r["id"], 0.0)
        if r["kind"] == "cond" and not context.get("minor", False):
            continue                                  # 条件不成立则跳过
        log.append({"rule": r["id"], "kind": r["kind"], "violation": round(v, 2)})
        if r["kind"] == "hard" and v >= 0.5:
            hard_hit = r["id"]
            break
    if hard_hit:
        return {"decision": "refuse", "reason": f"触发硬约束 {hard_hit}", "audit": log}
    score = sum(r["weight"] * (-violations.get(r["id"], 0.0)) for r in RULES
                if not (r["kind"] == "cond" and not context.get("minor", False)))
    return {"decision": "refuse" if score < -0.5 else "answer",
            "score": round(score, 3), "audit": log}

cases = [
    ("普通问题", {"S2": 0.0}, {}),
    ("涉及武器制造", {"H1": 0.9, "S2": 0.1}, {}),
    ("轻微边缘请求", {"S1": 0.3, "S2": 0.3}, {}),
    ("未成年人相关", {"C1": 0.7, "S2": 0.2}, {"minor": True}),
]
for name, viol, ctx in cases:
    r = adjudicate(viol, ctx)
    print(f"\n{name}: {r['decision']}  "
          f"({r.get('reason') or 'score=' + str(r.get('score'))})")
    print(f"   审计留痕: {r['audit']}")
print("\n读法：裁决结果必须带「依据哪条原则、被哪条否决」——这就是「可审计」在产品里的具体形态")

# 3) 过度优化的可预测性（引用 Gao et al. 的拟合式形式，系数为示意）
import math
def proxy_reward(d_kl, a=1.0, b=1.6):
    """d = sqrt(KL)；proxy = d*(a - b*d) 形式的先升后降"""
    d = math.sqrt(max(d_kl, 0))
    return d * (a - b * d)
best = None
for i in range(1, 400):
    dkl = i * 0.01
    r = proxy_reward(dkl)
    if best is None or r > best[1]:
        best = (dkl, r)
print(f"\n代理奖励峰值出现在 KL≈{best[0]:.3f}（对应 d=sqrt(KL)≈{best[0]**0.5:.2f}，代理分 {best[1]:.3f}）")
print("读法：代理奖励随 KL 先升后降 → 存在明确的早停点；把 KL 预算定在峰值之前，")
print("      就是「用可预测的过度优化曲线来定早停」——这是缩放定律对安全实践的直接影响")
```

预期输出要点（实跑）：① 位置偏差从 0 → 0.35 时，**交换顺序一致率从 71.3% 掉到 55.8%**（不一致样本必须转人工或降权，这是 AI 反馈质量的直接度量）；顺带一个真实观察：偏差 0.15 时一致率反而略高（72.5%），说明**中等位置偏差会被质量信号掩盖**——所以测量不能只测一个点位，要扫多个强度；② 原则裁决给出「裁定 + 依据的规则 ID + 是否命中硬约束」的可审计记录，并展示条件式规则（未成年人）如何按上下文启用；③ 过度优化曲线显示**代理奖励存在可预测的峰值**（本例在 $d=\sqrt{\mathrm{KL}}\approx0.31$、即 **KL≈0.10** 处达到峰值——注意峰值由 $d$ 决定，别把 $d$ 当 KL），因此可以把 KL 预算定在峰值之前作为早停策略——这直接连到下一题的「缩放定律如何影响安全评估」。

## 常见追问

- **追问**：为什么有用性标签仍然用人类反馈？
  - 要点：论文口径如此（只替换无害性标签）。工程理由也成立——**有用性更依赖细粒度的用户意图与文化语境**，AI 判断在这些维度上更容易自洽地偏；而无害性更接近「规则可判定」，适合用原则表达。
- **追问**：原则冲突怎么裁决？
  - 要点：定优先级（硬约束 > 条件式 > 应当式）+ 留痕 + 版本化；冲突本身不该被「打分平均」掩盖——把冲突显式暴露出来，才能在治理层讨论。
- **追问**：CAI 会不会让模型变得过度拒答？
  - 要点：会——这是**拒答边界的偏移**问题。指标上要看**分群误拒率**（不同语言/群体的拒答差异）与「该答未答」的比例，而不是只看有害输出率；原则收紧时必须有灰度与回滚。
- **追问**：如何评估 CAI 的效果？
  - 要点：三层——① 有害输出率与严重度分布（红线集）；② 误拒率与有用性（不能只优化一侧）；③ 一致性（同一请求在不同措辞/语言下判定是否稳定）。**只报有害率下降是片面的**。
- **追问**：用 AI 判 AI 的循环怎么打破？
  - 要点：外部判据——人工抽检（按风险分层）、可执行测试（代码/工具场景）、独立红线集与第三方评估；并把「判断模型与策略模型同源」的相关性误差显式记录为已知局限。
- **追问**：在 agent 场景 CAI 怎么用？
  - 要点：原则可以作为**行动前裁决**的依据（这个工具调用是否越界），但**不能替代权限系统**：训练侧降概率 + 系统侧最小权限 + 高危动作审批（串 [[agents-11]]）。

## 相关题目

- [[safety-05]]：CAI 与 RLAIF 的机制细节、论文口径与成本核算，本题是它在公司场景下的「收益/局限」版本。
- [[safety-03]]：越狱与对抗性 prompt，说明为什么训练侧不提供保证。
- [[finetuning-11]]：reward hacking 与过度优化的拟合式，是本题「没解决②」的技术依据。
- [[anthropic-21]]：缩放定律如何影响安全评估，接住本题「没解决④」的能力风险部分。
- [[safety-10]]：红队方案设计，是白盒评估与训练侧对齐的互补层。

## 参考资料与归属

- **Constitutional AI: Harmlessness from AI Feedback（延伸）** —— Bai et al. (Anthropic)，2022-12-15：<https://arxiv.org/abs/2212.08073>。第 1 节两阶段流程、第 2 节原则形态、以及「保留批评步骤的理由是可读性」这一表述来自这篇论文。
- **Scaling Laws for Reward Model Overoptimization（延伸）** —— Gao, Schulman, Hilton (OpenAI)，2022-10-19：<https://arxiv.org/abs/2210.10760>。可运行代码第 3 段「代理奖励随 KL 先升后降、峰值可预测」的形式来自这篇（本文件用的是**示意系数**，论文给出了各自的拟合系数）。
- **Training language models to follow instructions with human feedback（延伸）** —— Ouyang et al. (InstructGPT, OpenAI)，2022-03-04：<https://arxiv.org/abs/2203.02155>。作为「朴素 RLHF」对照的流程（人类偏好 → 偏好模型 → PPO + KL 正则）来自这篇。
- **延伸来源说明**：表 1 的单位成本口径（\$0.67/条与 \$0.06/条）、RLAIF 88% vs RLHF 76%、68% vs 71%（p=0.07）等具体数字在 [[safety-05]] 中已逐条核对并注明出处，本文复用其口径而不重复列源；表 2 与可运行代码中的位置偏差强度、规则权重、过度优化系数都是为演示而构造的**示意值**，不代表论文参数。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
