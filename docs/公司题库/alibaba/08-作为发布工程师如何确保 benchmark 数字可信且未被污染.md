---
type: question
id: alibaba-08
company: 阿里巴巴（Qwen）
topic: evaluation
order: 8
question: Qwen 发布的开放权重常居公开榜单前列。作为发布工程师，你如何确保 benchmark 数字可信且未被污染？
question_en: Qwen ships open weights that top public leaderboards. As the release engineer, how do you make the benchmark numbers trustworthy and uncontaminated?
asked_at: []
level: 高阶
tags: [benchmark-污染, 去污染, 发布工程, 评测治理]
sources:
  - title: LLM 评估
    url: https://outcomeschool.com/blog/llm-evaluation
    author: Amit Shekhar (Outcome School)
    published: 
  - title: LiveBench: A Challenging, Contamination-Limited LLM Benchmark（延伸）
    url: https://arxiv.org/abs/2406.19314
    author: White et al. (ICLR 2025)
    published: 2024-06-27
  - title: Investigating Data Contamination in Modern Benchmarks for Large Language Models（延伸）
    url: https://arxiv.org/abs/2311.09783
    author: Deng et al. (NAACL 2024)
    published: 2023-11-16
related: [evaluation-06, evaluation-05, evaluation-04, evaluation-02, alibaba-07]
updated: 2026-09-28
---

## 一句话答案

> 把「分数」当成一件**有供应链的产品**来交付：数据侧做去污染并有记录、评测侧用持续更新的公开基准加私有留出集、协议侧预注册口径并保留可复现证据、披露侧写清数据切分与去污染方法。核心心态是一句话：**默认公开榜单上的分数都可能被污染**，所以任何能力声明都必须能被自建的、私有的、定期轮换的评测复核。
> 具体到发布工程，我会交付四样东西：① 一份去污染报告（哪些基准、用什么方法、命中多少）；② 一份私有 held-out 评测（不进训练、只暴露聚合指标）；③ 一份协议声明（解码参数、样本量、置信区间、是否多次采样）；④ 一份「不引用」清单（无法复现的第三方数字不写进模型卡）。

## 面试官在考什么

- **污染的四种来源**能否分清：直接泄漏（测试样本进了预训练语料）、间接泄漏（用解析/答案训练、合成数据带入、教师模型蒸馏时背出答案）、评测过拟合（针对榜单格式调 prompt）、榜单机制偏差（私测择优与选择性披露）。只答「测试集泄漏」是初级答案。
- 是否知道**检测手段各自能证明什么**：$n$-gram 重叠只能证明表面复制；困惑度异常只是疑似；填空探针（TS-Guessing）是较强的记忆证据；扰动测试与时间切分是工程上最实用的两道筛子。
- 是否有**发布纪律**：红线集（不能拿污染集宣称能力）、口径先行（预注册）、私有留出（不参与调参）、以及「训练数据里到底有什么」的可追溯记录。
- 能否区分**「分数高」与「能力强」**：榜单是回归网不是目标函数（串 [[evaluation-05]]）；发布时的分数必须配「分群与长尾未退化」的证据。
- 是否愿意**主动暴露弱点**：模型卡里写清不擅长的场景与已知污染风险，比只放一个漂亮表格更能建立信任。

**常见错误答案**

- 「我们做了去重，所以没污染」——$n$-gram 去重挡不住改写、翻译与跨语言重述。
- 「榜单是第三方跑的，与我们无关」——开放权重一旦发布，任何人都能跑；发布方对**自己引用的数字**负责。
- 「多跑几个基准取平均更稳」——如果其中几个被污染，平均只是把噪声混合，反而更难解释。

## 原理与推导

### 1. 四种污染来源与各自的证据类型

| 来源 | 典型路径 | 能观察到什么 |
| --- | --- | --- |
| 直接泄漏 | 网页抓取时把 benchmark 原文一起收进语料 | $n$-gram/子串重叠、困惑度异常、填空探针命中 |
| 间接泄漏 | 用基准的解析与答案做训练、合成数据带入、教师模型蒸馏时把答案背出 | 扰动后分数骤降、改写版本仍高分、时间切分异常 |
| 评测过拟合 | 不改数据，针对榜单格式与偏好调 prompt/解码 | 同模型换一种问法分数掉、私有两种格式差异巨大 |
| 榜单机制偏差 | 私测多版本择优、选择性披露、数据获取不对称 | 复现失败、版本与分数对不上 |

**关键认知**：前两类是数据问题，第三类是流程问题，第四类是生态问题——**治理手段完全不同**，混在一起谈就会答成「多去重就好了」。

### 2. 检测：每种手段的证明力

1. **$n$-gram / 子串重叠**：能证明「这段文本出现过」，不能证明「模型记住了」。$n$ 越小误报越多（常见做法是 $n\ge 13$ 的词级 $n$-gram 或 50 字符以上的子串），且改写、翻译、跨语言重述会让它归零。
2. **困惑度异常**：在测试样本上异常低的困惑度是疑似证据——但短样本、模板化样本（多选题格式高度固定）本来困惑度就低，需要同类样本内的相对比较，不能跨任务比。
3. **填空探针（TS-Guessing）**：遮住多选题的错误选项或样本里的罕见词，让模型补。Deng et al. 的实验里，ChatGPT 与 GPT-4 **仅凭上下文**就能以 **52% 与 57%** 的精确匹配率猜出 MMLU 中被遮住的错误选项（论文自报口径，实验时间为 2023 年）。这是较强的记忆证据，但**这个数不是 MMLU 答题准确率**，两者不能混用。
4. **扰动测试**：改写题面、换变量名与数值、调整选项顺序。分数大幅下降通常说明模型依赖表面模式而非能力——这是工程上性价比最高的一招，因为它同时能发现「被污染」与「评测集本身太模板化」。
5. **时间切分**：只用模型训练截止之后发布的数据。局限是模型可能通过其它途径（后续语料、蒸馏、用户反馈）见过，所以它是必要不充分条件。
6. **私有 held-out**：最可靠的一类证据。要点是**它不能参与任何调参**，且要防内部泄漏（谁看过答案、是否进过任何训练集或合成数据）。

### 3. 防范：三层各自的动作

**数据侧**：① 建去污染流水线，把已知评测集（含其改写形态）从语料中剔除，并记录命中统计；② 对评测集注入 canary 串（一段唯一无意义的字符串），发布前用检索确认它没出现在语料与生成结果里；③ 训练数据来源与版本可追溯（否则事后无法回答「到底见没见过」）。

**评测侧**：① 用持续更新的公开基准作为对外信号——LiveBench 的设计口径是题目来自新发布的数学竞赛、arXiv 论文、新闻与数据集，答案有客观 ground truth 可自动判分，同时避开众包与 LLM judge 的主观偏差，**题目按月更新**，论文发布时（2024 年 6 月）报告最强模型准确率低于 70%；② 自建私有 held-out 并定期轮换（例如每季度换掉一半）；③ 把「换题后的分数」与「旧题的分数」并列报告，让读者能看到稳定性。

**流程侧**：① **预注册**：跑之前就写好要报的基准、解码参数、样本量与判分方式，避免事后挑好看的；② 报告规范：写清数据切分、去污染方法、是否多次采样与置信区间、以及所有不能复现的数字；③ 红线：**绝不用已知被污染的集合宣称能力**，也绝不把训练分布内的数据当评测集（串 [[evaluation-04]]）。

### 4. 发布工程师的检查清单（可执行）

发布前逐项打勾，任一项不通过就不发布数字：

1. 去污染报告：覆盖哪些基准、方法、命中条数、处理方式（剔除/标注）。
2. canary 检查：注入的 canary 串在语料与输出中零命中。
3. 私有评测：至少一份未进训练的 held-out，样本量与置信区间写明。
4. 扰动鲁棒性：题面改写/数值替换后分数下降在可解释范围内（例如个位数百分点），否则要么模型依赖模式、要么评测集有问题。
5. 分群与长尾：低资源语言、长尾领域、无答案场景的分数一并报告（避免总平均掩盖退化，串 [[evaluation-05]]）。
6. 口径声明：解码参数、few-shot 设置、是否用工具、是否多次采样取最优（**取最优这件事必须写明**）。
7. 可复现：随机种子与脚本可提供，或至少指明复现所需的环境与版本。
8. 模型卡披露：已知弱点与不适配场景。

### 5. 为什么「事后去重」不够：改写与语义等价

去污染的真正难点是**同一道题的等价改写**：换数字、换变量名、翻译成另一种语言、把选择题改成填空。这些形态在 $n$-gram 上完全不像，但模型只要见过语义等价的形式就可能「会做」。工程上可行的做法是三层叠加：$n$-gram 挡住直接复制；**语义去重**（对题目做嵌入聚类，把近重复簇整体剔除）挡住轻度改写；**扰动测试**兜底发现漏网的模板化依赖。三者都不能单独成立，这一点要在面试里主动说出来。

## 数值与代码验证

### 表 1：检测手段的证明力与成本

| 手段 | 证明强度 | 计算成本 | 主要漏检 |
| --- | --- | --- | --- |
| $n$-gram 重叠（$n\ge13$） | 表面复制 | 低（可哈希） | 改写、翻译、跨语言重述 |
| 困惑度异常 | 疑似 | 中（需前向） | 模板化样本、短样本 |
| TS-Guessing 填空探针 | 较强（记忆证据） | 中（需构造探针） | 非记忆型污染、自由生成任务 |
| 扰动测试 | 中（间接） | 中（要重跑评测） | 模型真的学到能力的场景（正常下降） |
| 时间切分 | 弱（必要不充分） | 低 | 通过其它途径见过 |
| 私有 held-out | 强 | 高（要建集与维护） | 内部泄漏 |

### 表 2：TS-Guessing 的口径纪律

| 数字 | 是什么 | 不能当成什么 |
| --- | --- | --- |
| 52%（ChatGPT）/ 57%（GPT-4） | MMLU 上「遮住错误选项后精确匹配」的比例 | MMLU 答题准确率；现在模型的结论 |
| LiveBench < 70% | 2024 年 6 月最强模型的准确率 | 永久阈值（题目按月更新，这条线会移动） |

### 可运行代码

```python
# 1) 最简去污染检查器：n-gram 重叠（词级，n=13），并演示它对改写的漏检
import re, hashlib
from collections import Counter

def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()

def ngrams(text, n):
    w = norm(text).split()
    return {" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))}

def overlap(train_text, bench_text, n=13):
    a, b = ngrams(train_text, n), ngrams(bench_text, n)
    return len(a & b), len(b), (len(a & b) / len(b) if b else 0.0)

TRAIN = ("The capital of France is Paris. Which of the following is the largest planet in the solar system? "
         "Options: A Mercury B Jupiter C Saturn D Earth. Answer: B. ") * 20
BENCH = ("Which of the following is the largest planet in the solar system? "
         "Options: A Mercury B Jupiter C Saturn D Earth. Answer: B.")
REWRITTEN = ("Among the planets listed below, which one is the biggest? "
             "A) Mercury B) Jupiter C) Saturn D) Earth — the answer is Jupiter.")

for name, probe in (("原文", BENCH), ("改写", REWRITTEN)):
    hit, total, ratio = overlap(TRAIN, probe)
    print(f"{name}: 命中 {hit}/{total} 个 {13}-gram，重叠率 {ratio:.1%}")

# 2) canary 注入/检索：发布前确认评测集没进语料
CANARY = "CANARY-AIEIQ-7f3c9d21-勿入训练语料"
CORPUS = "…大量语料… 其中混入了一段评测文本 " + CANARY + " …"
def canary_check(canary, corpus):
    return hashlib.sha256(canary.encode()).hexdigest()[:12], (canary in corpus)
tag, leaked = canary_check(CANARY, CORPUS)
print(f"canary {tag}: 语料中{'命中（有泄漏）' if leaked else '零命中（通过）'}")

# 3) 扰动测试的判定口径：分数下降多少算「可疑」
def perturb_delta(base, perturbed, tol=0.03):
    d = base - perturbed
    return d, ("正常波动" if d <= tol else "可疑：疑似依赖表面模式或被污染")
for base, pert in ((0.82, 0.80), (0.82, 0.51)):
    d, verdict = perturb_delta(base, pert)
    print(f"原题 {base:.2f} → 扰动后 {pert:.2f}，下降 {d:.2f} → {verdict}")
```

预期输出：原文与训练语料的 13-gram 重叠率很高（构造的重复语料），**改写版本重叠率为 0%**——这正是 $n$-gram 去污染的核心局限；canary 检查演示了「命中即泄漏」的判定；扰动测试给出「下降 ≤3 个百分点算正常波动，超过则可疑」的可执行阈值（阈值要按任务与样本量自己标定，不能照抄）。

## 常见追问

- **追问**：时间切分不是很好用吗，为什么说它必要不充分？
  - 要点：它只保证「基准发布时间晚于训练截止」，但模型可能通过后续语料、蒸馏、用户反馈、或第三方微调版本间接见过。它排除的是最省事的那条泄漏路径。
- **追问**：私有 held-out 怎么防内部泄漏？
  - 要点：访问控制（只有评测负责人能看原始样本）、不把原始样本贴进任何文档或工单、只暴露聚合指标、定期轮换、并把「谁在什么时候看过」记入审计（串 [[evaluation-07]] 的可观测思路）。
- **追问**：开放权重发布后，别人跑出更低的分数怎么办？
  - 要点：先复现差异来源（解码参数、few-shot、是否用工具、prompt 模板），再判断是污染疑云还是能力差异。模型卡里写清复现所需的全部参数，是唯一能让分歧收敛的办法。
- **追问**：怎么回应「你们是不是刷榜」这种质疑？
  - 要点：拿证据而不是态度——去污染报告、私有留出分数、扰动测试结果、以及「同一模型在不同格式下的一致性」。同时主动承认榜单是回归网不是目标函数。
- **追问**：如果内部评测与公开榜单结论相反，信哪个？
  - 要点：信内部、且要解释差异——公开榜单是别人造的分布，内部评测贴近自己的流量与用户。差得太多时先把两边样本对齐（同分布、同解码）再下结论（串 [[evaluation-05]]）。
- **追问**：合成数据会不会把 teacher 的答案带进来造成污染？
  - 要点：会。教师模型蒸馏时可能把基准答案背出来，尤其是它训练时见过的话。所以合成数据也要过去污染流水线，并在数据血缘里记录生成模型与版本（串 [[alibaba-06]] 的蒸馏链路）。

## 相关题目

- [[evaluation-06]]：污染的四来源、六检测与三层防范的通用版本，本题是它在「开放权重发布」场景下的具体化。
- [[evaluation-05]]：榜单分数与用户感受背离的七类原因，解释了为什么必须把榜单当回归网而不是目标函数。
- [[evaluation-04]]：把去污染与私有留出做成发布门禁，是本题检查清单的落地形态。
- [[evaluation-02]]：没有标注时如何构建评测集，与私有 held-out 的建设同源。
- [[alibaba-07]]：RLVR 的题库与评测集必须去重，否则就是自己给自己出题——污染问题的训练侧版本。

## 参考资料与归属

- **LLM 评估** —— Outcome School 博客（该条来源未署个人作者与日期）：<https://outcomeschool.com/blog/llm-evaluation>。评测分层、基准选择与「评测集要能复现」的总纲转述自这篇。
- **LiveBench: A Challenging, Contamination-Limited LLM Benchmark（延伸）** —— White et al.（ICLR 2025），2024-06-27：<https://arxiv.org/abs/2406.19314>。题目按月更新、答案有客观 ground truth、规避众包与 LLM judge 偏差的抗污染设计，以及发布时最强模型低于 70% 的口径来自这篇。
- **Investigating Data Contamination in Modern Benchmarks for Large Language Models（延伸）** —— Deng et al.（NAACL 2024），2023-11-16：<https://arxiv.org/abs/2311.09783>。检索式重叠检查系统、TS-Guessing 填空探针协议，以及 MMLU 缺失选项精确匹配 52%（ChatGPT）/ 57%（GPT-4）与 TruthfulQA 元数据实验的口径来自这篇。
- **延伸来源说明**：表 1 的证明力与成本评级、表 2 的口径纪律、扰动测试的 3 个百分点阈值示例，以及去污染三层叠加（$n$-gram + 语义聚类 + 扰动兜底）的做法，是本仓库基于上述资料整理的工程口径与自算结论，不是论文原文数字。Qwen 系列的公开榜单表现与模型卡披露实践来自其公开材料，本仓库**未把该材料列入来源数组，也未据此改动任何来源的 title/author/published**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
