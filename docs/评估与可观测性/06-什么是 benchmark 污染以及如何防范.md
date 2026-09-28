---
type: question
id: evaluation-06
topic: 评估与可观测性
order: 6
question: 什么是 benchmark 污染，你如何防范？
question_en: What is benchmark contamination and how do you prevent it?
asked_at: [智谱 AI, 阿里巴巴, Scale AI]
level: 进阶
tags: [污染, 数据泄漏, held-out, 评测治理]
sources:
  - title: LLM Evaluation
    url: https://outcomeschool.com/blog/llm-evaluation
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: "LiveBench: A Challenging, Contamination-Limited LLM Benchmark（延伸）"
    url: https://arxiv.org/abs/2406.19314
    author: White et al. (ICLR 2025)
    published: 2024-06-27
  - title: "Investigating Data Contamination in Modern Benchmarks for Large Language Models（延伸）"
    url: https://arxiv.org/abs/2311.09783
    author: Deng et al. (NAACL 2024)
    published: 2023-11-16
related: [evaluation-05, evaluation-02, evaluation-04, finetuning-12]
updated: 2026-09-28
---

## 一句话答案

> benchmark 污染指评测集的内容或答案以任何形式进入了模型的训练/调优链路，或者评测协议本身被针对榜单优化，使分数不再度量能力。按来源分四层：直接泄漏、间接泄漏、评测过拟合、榜单机制偏差；公开静态 benchmark 的 test split 本身就是公开文本，因此污染只能被缓解，不能被消除。
> 检测环节没有单一信号可以定案：$n$-gram 重叠只能证明表面复制，改写、翻译、跨语言重述会让它归零；困惑度异常给的是疑似证据；TS-Guessing 这类填空探针能给出较强的记忆证据——Deng et al. 的实验里 ChatGPT 与 GPT-4 仅凭上下文就能以 52% 与 57% 的精确匹配率猜出 MMLU 中被遮住的错误选项（论文自报口径）。
> 工程落点只有一条：上线决策依赖自建的、私有的、定期轮换的 held-out 集；公开榜单分数只当参考信号，并且必须连同评测协议与去污染说明一起报告。

## 面试官在考什么

- 能否分层给来源，而不是把污染等同于「测试集泄漏了」这一句话。
- 是否知道每种检测手段的检出边界：能发现什么、漏掉什么、证据强度属于「疑似」还是「可定案」。
- 能否把防范落成可执行的三层：数据侧的去污染流水线、评测侧的 held-out 建设与轮换、流程侧的报告与访问控制规范。
- 是否理解污染的代价不只是分数虚高，而是选型决策错误——把记忆当成能力，然后在错误路线上追加投入。
- 会不会把「模型见过数据」和「评测协议被针对榜单调优」分开，因为两者的证据位置与处置手段完全不同。

常见错误答案：

- 「训练前把测试集从语料里删掉就行」。删掉的只是清单内的集合，而改写、翻译、跨语言重述、蒸馏、合成数据、按榜单调 prompt 都不在那份清单里。
- 把 $n$-gram 重叠命中当成判定性证据。命中率高确实说明存在问题，但命中率为 0 什么也证明不了；反过来，模板化文本还会造成大面积误报。
- 「换一个更新的 benchmark 就干净了」。新基准在被大量引用、被写进博客与题解之后同样会流入语料；而且越新的基准越常由自动化流程从新数据里构造，题目质量与代表性都不再有保证。

## 原理与推导

### 1. 污染的四层来源

| 层 | 机制 | 典型路径 | 证据留在哪 |
| --- | --- | --- | --- |
| ① 直接泄漏 | 评测样本原文进了预训练或微调语料 | 网页抓取把 benchmark 页面、数据集镜像、题解博客、issue 讨论一起收进语料 | 语料里能查到重叠 |
| ② 间接泄漏 | 样本原文没进去，答案进去了 | 用评测集做 few-shot 示例或合成数据的种子；用教师模型的推理做 CoT 蒸馏，答案与解题过程一起写进训练数据 | 模型内部：似然、填空 |
| ③ 评测过拟合 | 数据没进训练，但评测协议针对榜单被调过 | prompt 模板与 few-shot 示例按榜单调；解码与聚合参数（majority vote、best-of-$n$、思考预算）按榜单调 | 换协议分数就掉 |
| ④ 榜单机制偏差 | 榜单的测试与披露机制本身不对称 | 私测择优、选择性披露、配对与投票采样不均衡（The Leaderboard Illusion 一文的讨论口径） | 榜单元数据与提交记录 |

①② 是「模型见过答案」，③ 是「评测让模型看起来更强」，④ 是「榜单让某些模型看起来更强」。能主动切这一刀，比背检测手段清单更能说明工程判断力。③ 与 ④ 的展开见 [[evaluation-05]]，② 的蒸馏通路的展开见 [[finetuning-12]]。

### 2. 为什么公开静态 benchmark 的污染只是时间问题

公开 benchmark 的 test split 天生是公开文本：题目、答案、解析会同时出现在论文附录、GitHub 仓库、数据集镜像、教学博客与问答社区里。爬虫按质量与域名启发式筛数据，不区分「训练集」「测试集」。多数去污染实现是「拿已知评测集清单跑一遍匹配，命中就丢」，它覆盖的范围严格等于那份清单——新基准、私有集、以及当年尚未发布的版本都不在清单里。

两个正反馈让问题随时间恶化：越被高频引用、越被当作卖点的基准，被复述的次数越多；同时它的分数越受关注，打榜动机越强。指令微调与合成数据会进一步放大泄漏效果：把答案写成「标准答案 + 完整推理」的格式，比预训练语料里零散的偶然复述更容易被模型记住。

后果有三层：分数虚高 → 选型错误 → 真实能力上限被高估。

### 3. 检测手段与各自的边界

| 手段 | 信号 | 能发现 | 漏掉 | 证据强度 |
| --- | --- | --- | --- | --- |
| 重叠检查 | 语料与评测集的 $n$-gram 或子串重合 | 原文复制、浅层词替换 | 改写、翻译、跨语言重述；清单外的基准 | 命中是强提示，未命中不能证明干净 |
| 似然 / 困惑度异常 | 测试样本上异常低的困惑度 | 被高概率背下来的样本 | 同领域风格与模板带来的天然低困惑度 | 疑似 |
| TS-Guessing | 遮住错误选项或罕见词后让模型补 | 多选题、实体型样本的记忆痕迹 | 自由生成型任务；已被改写的样本 | 论文实验层面的强证据 |
| 扰动 / 改写测试 | 换语序、换变量名与数字后的分数变化 | 依赖表面形式的分数 | 改写本身改变了难度，需要先做难度校准 | 疑似，需对照 |
| 时间切分 | 只用训练截止之后发布的数据 | 直接的记忆型泄漏 | 数据可能经转发、镜像、代理数据集回流 | 结构性缓解，不是证明 |
| 私有 held-out | 无公开痕迹的样本 | 一切依赖公开数据的污染 | 内部泄漏 | 最强，前提是没泄漏 |

困惑度由 token 级对数似然定义：

$$\mathrm{PPL}(x) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})\right)$$

它的绝对值不可比：模板化文本、与训练分布同源的文本天然困惑度低。更稳的用法是取同分布内的相对量（例如概率最低的 $k\%$ 个 token 的平均对数似然），并且必须用一批同任务、同风格但确实不在评测集里的样本作对照，否则会误报。

判据要写清口径：单一信号只能说「疑似」；至少两条独立证据（例如重叠命中 + 扰动后分数崩塌 + 填空探针答对）才下结论。检测的产出应该是「这个评测集的分数能信多少」，而不是一个布尔值。

### 4. 防范：数据侧、评测侧、流程侧

**数据侧**

- 把去污染做进数据构建流水线，而不是训练前手工跑一次：命中后丢样本或丢整篇文档，并记录每个来源的命中率与丢弃量——这两个数本身就是数据质量的监控指标。
- 记录数据来源、抓取时间与版本，给每份语料标注截止时间，让时间切分可执行。
- 在评测集发布版本里埋 canary 字符串或水印，检测模型能否续写它。这是少数能对「未知泄漏路径」报警的手段。
- 不把评测集用作 few-shot 示例、合成数据种子，或 prompt 调优的迭代集。

**评测侧**

- 用持续更新的基准作为公开信号。LiveBench 的设计口径是：题目来自新发布的数学竞赛、arXiv 论文、新闻与数据集，答案有客观 ground truth 可自动判分，同时避开众包与 LLM judge 的主观偏差，题目按月更新；论文发布时（2024 年 6 月）报告最强模型准确率低于 70%。可更新与足够难可以同时成立，这是这类基准的价值。
- 自建私有 held-out 集并定期轮换，按任务类型、难度、语言分层；轮换节奏与模型发布节奏对齐。
- 公开分数时同时给出协议：prompt、few-shot 数、解码参数、聚合方式（是否 majority vote 或 best-of-$n$）、评测集版本。

**流程侧**

- 报告规范写死：说明去污染方法与数据切分，不引用无法复现的分数。
- 内部评测与公开榜单不共用同一集合。回归门禁如果建立在被污染的集合上，会把「记忆得更多」判成「能力更强」，见 [[evaluation-04]]。
- 供应商选型要求给出可复现的评测方法，或直接用自己的私有集复测；不接受只给一个数字。
- 评测集做访问控制与审计：谁看过答案、模型是否曾以它为输入，都要留痕。

### 5. 一句可背的结论

默认公开榜单上的分数都可能被污染或被针对性优化，所以上线决策只认自建的、私有的、定期轮换的 held-out 集；公开分数用于粗筛与对外沟通，用于选型决策时必须附带协议与去污染说明。私有集的构建方法见 [[evaluation-02]]。

## 数值与代码验证

论文口径的数字，引用时必须带年份与出处：

| 结论 | 数字 | 口径 |
| --- | --- | --- |
| MMLU 上猜被遮住的错误选项，精确匹配率 | ChatGPT 52%，GPT-4 57% | Deng et al.（NAACL 2024）论文自报，arXiv:2311.09783 摘要；实验时间为 2023 年 |
| TruthfulQA 提供额外元数据后性能提升 | 论文只给定性结论，摘要未给具体数值 | 同上，论文自报 |
| 评测集与预训练语料的重叠探查 | 论文提出检索式重叠系统（方法，非数字） | 同上，论文自报 |
| 最强模型准确率 | 低于 70% | White et al.（ICLR 2025）论文自报，arXiv:2406.19314 摘要，2024-06 版本 |
| 题目更新频率 | 每月增加与更新 | 同上，论文自报 |

三点口径提醒：这些数字绑定当时的模型与基准版本，不能说成「现在的模型」的结论；LiveBench 持续更新，其榜单上 70% 这条线会随时间上移；TS-Guessing 的 52% / 57% 是「给定上下文填空」的精确匹配率，不是 MMLU 答题准确率，两个数不能混用。

下面这张表是**本方复算**，不是论文数字：用一段合成语料和 6 条评测样本，测每种改写方式在词级 $n$-gram 下的覆盖率（样本的 $n$-gram 有多大比例能在语料中找到）。

| 样本构造 | 5-gram | 8-gram | 13-gram |
| --- | --- | --- | --- |
| 逐字复制 | 1.000 | 1.000 | 1.000 |
| 只改格式（大小写、标点、空格） | 1.000 | 1.000 | 1.000 |
| 替换术语（KV cache → key value memory） | 1.000 | 1.000 | 1.000 |
| 深度改写（换语序与同义词） | 0.000 | 0.000 | 0.000 |
| 跨语言（语料是英文，样本是中文） | 0.000 | 0.000 | 0.000 |
| 对照组（语料里确实没有） | 0.000 | 0.000 | 0.000 |

与直觉相反的部分：浅层编辑几乎不影响覆盖率（术语替换后仍是 1.000，因为大部分窗口没被碰到），而一次深度改写直接把信号打到 0。用「窗口存活」建模能解释这条陡峭的边界：设每个词被改写的概率为 $p$，长度 $n$ 的窗口完全不被碰到的概率是 $(1-p)^n$。取 $p=0.2$ 时，$n=5$ 还有 0.328，$n=13$ 只剩 0.055；取 $p=0.05$ 时 $n=13$ 是 0.513。于是：

- 短 $n$ 对改写稍好，但误报陡增；
- 长 $n$ 抗误报，但对改写几乎失效；
- 不存在同时解决漏报与误报的 $n$，这正是否掉「$n$-gram 检查通过 = 没污染」的量化依据。

误报的具体形态：很多评测 harness 会给每条样本前置同一段 instruction 模板。如果爬到的语料里恰有一份带该模板的页面，那么「任意一个窗口命中就标红」的粗粒度判定会把全部样本标成污染，连语料里根本不存在的对照样本也会被标红（复算脚本见「数值与代码验证」一节，6/6 全部命中）。所以工程实现要做的是：过滤模板与公共片段、报告覆盖率而不是布尔值、并按「命中窗口数 / 总窗口数」设阈值。

```python
import re

TEMPLATE = ("You are a helpful assistant. Answer the following question carefully "
            "and put the final answer on its own line.")

CORPUS = [
    "Unrelated paragraph about GPU memory bandwidth, kept as filler",
    "PagedAttention stores the KV cache in fixed size blocks so that a sequence "
    "no longer needs one contiguous allocation",
    "PagedAttention stores the key value memory in fixed size blocks so that a "
    "request no longer needs one contiguous allocation",
    "The scheduler admits a new request as soon as the running batch has a free "
    "slot instead of waiting for the whole batch to finish",
    TEMPLATE + " Explain how PagedAttention manages memory.",
]

EVAL = {
    "verbatim": "PagedAttention stores the KV cache in fixed size blocks so that a "
    "sequence no longer needs one contiguous allocation",
    "format": "PagedAttention  stores the KV cache in fixed-size blocks,  so that a "
    "sequence no longer needs one contiguous allocation!",
    "term": "PagedAttention stores the key value memory in fixed size blocks so that "
    "a request no longer needs one contiguous allocation",
    "paraphrase": "KV cache is kept by PagedAttention inside blocks of a fixed size, "
    "which removes the need for one contiguous allocation per sequence",
    "translated": "调度器一旦发现运行中的批次有空位，就立刻接入新请求，而不必等整个批次跑完",
    "control": "Rotary position embedding applies a rotation matrix whose angle is "
    "proportional to the token position index",
}


def norm(text):
    text = text.lower()
    text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", text)  # CJK 逐字切
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return text.split()


def grams(text, n):
    t = norm(text)
    return {tuple(t[i : i + n]) for i in range(len(t) - n + 1)}


def pool(n):
    return set().union(*(grams(doc, n) for doc in CORPUS))


def coverage(item, n):
    g = grams(item, n)
    return "n/a" if not g else f"{len(g & pool(n)) / len(g):.3f}"


print(f"{'item':<12}" + "".join(f"{n:>8}" for n in (5, 8, 13)))
for name, item in EVAL.items():
    print(f"{name:<12}" + "".join(f"{coverage(item, n):>8}" for n in (5, 8, 13)))

# 误报演示：harness 给每条样本前置同一段模板，语料里恰有一份带该模板的页面
hits = sum(1 for item in EVAL.values() if grams(TEMPLATE + " " + item, 5) & pool(5))
print(f"粗粒度判定（任意窗口命中即标红）：{hits}/{len(EVAL)}")
```

```text
item               5       8      13
verbatim       1.000   1.000   1.000
format         1.000   1.000   1.000
term           1.000   1.000   1.000
paraphrase     0.000   0.000   0.000
translated     0.000   0.000   0.000
control        0.000   0.000   0.000
粗粒度判定（任意窗口命中即标红）：6/6
```

要用到生产里，这份脚本还缺三件事：语料侧的近重复索引（MinHash / SimHash）以覆盖局部改写；模板与公共片段黑名单；以及每条评测样本的版本与 hash 记录，让「这个分数对应哪个版本」可追溯。

## 常见追问

- **追问**：$n$-gram 检测为什么不够？
  - 要点：两个方向都不够。漏报方面，改写与跨语言重述会让覆盖率归零（本方复算 0.000）；误报方面，模板与公共片段会让粗粒度判定全量标红（6/6）。它只能当排除性证据：命中说明有问题，不命中不说明没问题。
- **追问**：时间切分有什么局限？
  - 要点：它只保证「基准晚于截止时间」，不保证模型没见过——数据可能通过转发、镜像、代理数据集在新版本里回流；新基准未必匹配线上分布，题目质量与代表性也参差。它适合当结构性缓解，不适合当证据。
- **追问**：私有 held-out 集怎么防内部泄漏？
  - 要点：访问控制（只有评测 owner 能看答案）、只对外暴露聚合指标、按人分片持有、答案不落到训练管线可见的存储、访问留审计日志、按季度或按模型发布节奏轮换；同时确保它从不出现在 few-shot 示例、合成数据种子与内部 prompt 调优迭代集里。
- **追问**：供应商只给分数不给方法怎么办？
  - 要点：要求协议卡（prompt、few-shot 数、解码与聚合方式、评测集版本、去污染方法）并要求可复现；不能复现就把分数当参考，用自己的私有集在小样本上复测，并把「无法复现」写进选型记录。
- **追问**：困惑度低就说明污染吗？
  - 要点：不是。同领域文本、模板化样本、高频格式都会让困惑度偏低。要用同分布对照和相对量（例如概率最低的 $k\%$ 个 token 的平均对数似然），而不是绝对值；它给的是疑似信号。
- **追问**：污染和评测过拟合怎么区分？
  - 要点：看证据位置。污染在模型内部留痕（似然异常、能填空、扰动后分数崩），过拟合只在评测协议层面可见（换 prompt 模板或换榜单就掉分）。处置也不同：前者清数据、重建评测集，后者固定协议、禁止按榜单调 prompt、报告完整协议。

## 公司变体

- **智谱 AI**：自研模型团队同时承担预训练数据治理与自建评测，这类题目通常落在工程实现一侧：被追问最多的是去污染流水线怎么落地（命中率、丢弃策略、词汇改写与跨语言造成的漏检）、中文语料里怎么发现泄漏、评测集怎么版本化与轮换。原理部分会被要求讲清 $n$-gram 与似然两类信号各自的边界。
- **阿里巴巴**：模型团队与云上模型服务都在，落点偏工程实现与流程治理：模型选型的评测协议怎么定、供应商报告怎么审、内部评测集怎么做权限与回归、公开榜单分数在选型里占多大权重。常见延伸是「供应商分数不可复现时，你的决策流程是什么」。
- **Scale AI**：数据与评测服务方，落点偏方法论与流程规范。公开的团队产出集中在榜单与评测集设计（公开讨论过如何让榜单更难被 game，以及 verifier 怎么设计），因此追问更可能是：held-out 集怎么管理、标注与评测人员的访问控制、如何交付一份第三方可复现的评测报告。
- 以上只说明侧重方向（工程实现 vs 原理推导），不代表具体轮次或流程安排。

## 相关题目

- [[evaluation-05]]：榜单分数在涨但用户体感变差，其中「榜单机制层面的偏差」与这里第 4 层来源是同一问题的两面。
- [[evaluation-04]]：回归门禁建立在哪份数据集上，直接决定门禁会不会被污染分数带偏。
- [[evaluation-02]]：没有现成 ground truth 时怎么造私有评测集，是「用私有 held-out 兜底」这一结论的前置能力。
- [[finetuning-12]]：蒸馏与合成数据是间接泄漏最主要的通路。

## 参考资料与归属

- [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)，Amit Shekhar（Outcome School）——benchmark 的三个决定性问题（饱和、数据污染、资格门槛），以及「公开基准很少匹配真实用例、应自建评测集」的实践建议。
- [LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314)，White et al.（ICLR 2025 Spotlight），2024-06-27（延伸来源）——题目按月更新、按客观 ground truth 自动判分、规避众包与 LLM judge 偏差的设计口径，以及发布时最强模型低于 70% 的结论。
- [Investigating Data Contamination in Modern Benchmarks for Large Language Models](https://arxiv.org/abs/2311.09783)，Deng et al.（NAACL 2024），2023-11-16（延伸来源）——检索式重叠系统与 TS-Guessing 协议，MMLU 缺失选项 52% / 57% 与 TruthfulQA 元数据实验。
- 「原理与推导」一节的来源分层、检测手段清单与防范分层由本篇整理；「数值与代码验证」一节的论文口径表逐条给出出处，其中的覆盖率表、窗口存活算例与代码为本方复算，非论文数字。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
