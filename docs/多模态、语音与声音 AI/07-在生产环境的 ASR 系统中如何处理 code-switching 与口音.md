---
type: question
id: multimodal-07
topic: 多模态、语音与声音 AI
order: 7
question: 在生产环境的 ASR 系统中，你如何处理 code-switching 与口音？
question_en: How do you handle code-switching and accents in a production ASR system?
asked_at: [Sarvam AI, Abridge]
level: 高阶
tags: [code-switching, 口音, mer, 数据]
sources:
  - title: Robust Speech Recognition via Large-Scale Weak Supervision（Whisper）（延伸）
    url: https://arxiv.org/abs/2212.04356
    author: Radford et al. (OpenAI)
    published: 2022-12-06
  - title: Attention-Guided Adaptation for Code-Switching Speech Recognition（延伸）
    url: https://arxiv.org/abs/2312.08856
    author: Aditya et al. (ICASSP 2024)
    published: 2023-12-14
  - title: TALCS: An Open-Source Mandarin-English Code-Switching Corpus and a Speech Recognition Baseline（延伸）
    url: https://arxiv.org/abs/2206.13135
    author: Li et al. (Interspeech 2022)
    published: 2022-06-27
  - title: Summary on the ISCSLP 2022 Chinese-English Code-Switching ASR Challenge（延伸）
    url: https://arxiv.org/abs/2210.06091
    author: Deng et al. (ISCSLP 2022)
    published: 2022-10-12
  - title: Mandarin-English Code-switching Speech Recognition with Self-supervised Speech Representation Models（延伸）
    url: https://arxiv.org/abs/2110.03504
    author: Tseng et al.
    published: 2021-10-07
related: [multimodal-06, multimodal-03, finetuning-06, safety-09]
updated: 2026-09-28
---

## 一句话答案

> 这两类问题要分开处理。**code-switching（语码转换）**的难点是语言边界没有标记、两种语言共享音素集、转写规范不统一；**口音**的难点是训练数据覆盖不足带来的**系统性退化**，而不是随机出错。手段分成三层：数据侧（真实 CS 语料 + 拼接/合成增强 + SSL 表征）、建模侧（统一多语言子词表、LID 联合训练、参数高效适配与注意力头引导）、解码侧（语言感知解码、热词偏置、术语后处理）；口音侧一律先按群量化再优化。上线前必须先把 MER/WER 的归一化与分词口径写死，否则跨版本数字不可比。

## 面试官在考什么

- 能否把两个常被混为一谈的概念拆开定义：句间/句内/标签式 code-switching 与口音/方言，各自的失败机制不同。
- 是否理解**为什么现成的多语言大模型会退化**：不只是「数据少」，还包括语言身份路由错误、词表竞争、解码时语言先验缺失。
- 数据直觉：真实 CS 语料有多稀缺、合成数据能补什么又会在哪里引入偏差。
- 建模手段的机制细节：LID 联合训练、注意力头引导、热词偏置各自改变了分布里的哪一项。
- 评估纪律：MER 与 WER 的差别、口径敏感性、分群报告、以及「整体指标没掉但某群体体验更差」的识别方法。

常见错误答案：

- 只答「多喂 data、微调一下就好了」：没有区分真实/合成数据的作用边界，也没有解释为什么通用大模型在这上面会退化。
- 把 code-switching 与口音当成同一件事，或者只谈口音分类器而完全不碰解码与评估口径。
- 声称「MER 比 WER 全面，所以只看 MER」：MER 同样不区分「同一语言内的错词」与「整段语言判断错误」，只换了一个分母口径。

## 原理与推导

### 1. 先把两类问题分开定义

**code-switching** 指同一段话语中使用两种以上语言，按切换位置分三类：

| 类型 | 形态 | 例子 |
| --- | --- | --- |
| 句间转换（inter-sentential） | 一句话一种语言，句与句之间切换 | 「今天先到这里。」「OK, let's stop here.」 |
| 句内转换（intra-sentential） | 句子内部切换，最常见也最难 | 「把这个 deploy 到 production 环境」 |
| 标签式转换（tag switching） | 插入固定短语或感叹词 | 「这个方案，怎么说呢，makes sense」 |

真正会破坏句子内部声学—语言对齐的是句内转换与标签式插入，其中句内转换最难——标签式插入的短语固定、可以穷举，工程上并入句内一起处理。它的三个难点：

1. **没有明确的语言边界**：音素序列里没有分隔符，边界只在语义层面存在。
2. **共享音素集冲突**：英文 `one` 与中文「万」在声学上高度接近（`/w/` 起头、鼻音韵尾），语义完全无关。声学证据不足时只能靠语言上下文决定，这直接引出下面的 LID 联合建模。
3. **转写规范不统一**：中文段用汉字还是拼音、英文词用原拼写还是音译、大小写与标点怎么处理，各团队各语料都不一样，而这套规范同时决定了指标的分母。

**口音**是同一语言内部的音素实现与韵律差异（辅音清浊、元音空间、声调实现、语速与重音）。它的关键性质是**系统性**：某个口音群体不是随机错几个词，而是在特定音素对上成片出错，因此整体平均指标会掩盖它。

### 2. 现成模型为什么退化：机制与证据

Whisper 在 68 万小时多语言、多任务弱监督上训练，零样本迁移就能在标准 benchmark 上接近此前全监督的结果、在准确率与鲁棒性上接近人类水平（Whisper 论文摘要口径）。但这类多语言模型在 code-switching 场景下表现不佳：Attention-Guided Adaptation 的出发点是「强大的多语言模型（如 Whisper）经常处理不好 code-switching」，该工作通过参数高效适配，只额外训练约 5.6% 的参数，把紧密表达语言身份的注意力头引导到对应语言，在中英 CS 语料上达到 14.2% 的混合错误率并超过当时的最好方法（ICASSP 2024）。

退化的机制可以拆成三条：

1. **语言身份信息存在但没有被正确路由**。以上述工作能通过「选择语言身份相关的注意力头并引导它们」来改进为前提，模型内部已经编码了语言身份；另一项自监督工作的结论与之呼应——SSL 模型的隐表征带有**帧级语言身份**信息，即使只用英语语音训练（Tseng et al. 口径）。所以问题往往出在路由与解码，而不是表征里没有信息。
2. **词表竞争与切分不一致**。多语言子词表里，同一段音素可能被中文和英文的 subword 同时竞争；英文专有名词被切成碎块后，语言模型的先验贡献被稀释。
3. **训练目标没有语言边界监督**。以单语转写为主的目标函数不会告诉模型「这里应该切回中文」，句内切换在训练分布里又是长尾。

形式化地看，联合建模就是在显式引入语言变量 $l_t$：

$$
P(y \mid x) = \sum_{l_{1:T}} P(l_{1:T} \mid x)\, P(y \mid x, l_{1:T})
$$

而纯 CTC 的目标是

$$
P(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} P(\pi_t \mid x_t)
$$

逐帧后验之间条件独立，意味着输出序列这一层没有标签侧的语言先验来打破平局（编码器帧本身带上下文，但那是声学侧的证据）：在 `one` 与「万」这类共享音素的片段上，两套语言的 token 会分摊后验概率。这正是 CTC 与 LID 联合训练、或用外部语言模型约束解码的动机。

### 3. 数据侧：真实语料、增强、自监督表征

1. **真实 CS 语料是地基**。TALCS 是约 **587 小时、16 kHz** 的普通话—英语 CS 语料，采自真实线上一对一英语教学场景，论文称其为当时最大的、标注良好的开源中英 CS ASR 数据集，并给出 ESPnet 与 WeNet 上的 MER 基线。口径提示：587 小时只有 Whisper 训练语料（68 万小时）的约 **0.086%**，这就是为什么通用模型不会自然覆盖这一现象。
2. **数据增强与合成**。三条常见路线：用单语数据做句内拼接/片段替换（需要语言学约束，避免造出不自然的切换点）；用 TTS 合成 CS 语音（覆盖面广但音色与信道单一，容易让模型过拟合到合成声学）；从高资源单语迁移的元学习/课程学习路线。合成数据只做预热与辅助任务，评测必须落在真实 held-out 集上。
3. **自监督表征**。SSL 预训练能利用大量**不含 CS** 的未标注语音：论文报告 SSL 隐表征含帧级语言身份（即使只用英语语音训练），**CTC 与 LID 联合训练**能提升 CS 识别，且**多语言语音预训练**效果最好。
4. **挑战赛与基线提供难度参照**。ISCSLP 2022 中英 CS ASR 挑战使用 TAL_CSASR 与 MagicData-RAMC 两个训练集，**40 余支队伍**参加，冠军取得 **16.70% MER**，相比基线**绝对提升 9.8%**（即基线约 26.50% MER，相对下降 37.0%）。这个数字说明：即便有专门语料和专项比赛，中英 CS 的 MER 仍在十几个百分点量级。

### 4. 建模侧：词表、LID、适配、解码

1. **统一的多语言子词词表**：中文按字或高频词、英文按 BPE 子词，共用一张表；避免同一语言在不同模型分支下切分粒度漂移，也让英文专有名词保持可拼写性。
2. **LID 联合建模与语言感知解码**：多任务目标写成 $\mathcal{L} = \mathcal{L}_{\text{CTC}} + \lambda\,\mathcal{L}_{\text{LID}}$，解码时把语言后验作为约束项，$\text{score}(y) = \log P_{\text{ASR}}(y \mid x) + \beta \sum_t \log P(l_t \mid x_t)$。$\beta$ 控制语言约束强度：过大会把双语人名、品牌名强行推向单语写法。第 2 节里提到的 SSL + CTC + LID 属于同一路线。
3. **参数高效适配与注意力头引导**：只训练少量适配参数（上文的 5.6% 量级），把语言身份相关的注意力头引导到对应语言上，在保留通用能力的同时修正路由。适配器选型（LoRA、prefix tuning、prompt tuning 与全量微调的取舍）见 [[finetuning-06]]。
4. **解码期语言模型融合与热词偏置**：shallow fusion 加术语偏置项，$\text{score} = \log P_{\text{AM}} + \alpha \log P_{\text{LM}} + \gamma \sum_i \log b(w_i)$，对「中文句子里夹英文产品名/代码标识符」最直接有效。$\gamma$ 是双刃剑：偏置过强会引入插入错误与幻觉词，必须用带负样本的集做扫描。
5. **统一的文本归一化与标点规范**：数字 ITN、大小写、标点、是否接受拼音或音译，全链路一套规则。它既是转写质量要求，也直接决定第 6 节的指标口径。

### 5. 口音侧：条件化、自适应、先量化再优化

1. **口音条件化**：把口音分类或说话人表征作为条件特征（accent embedding、FiLM 式调制、adapter 混合路由），让模型对不同音素实现使用不同的参数子集。
2. **微调与测试时自适应**：有标注口音数据做监督微调；线上用高置信伪标签做无监督自适应，配合置信筛选与回放旧数据，防止对标准口音回退。
3. **先量化再优化**：按口音/地区/母语背景分群报告 WER，并给出置信区间（方法见 [[safety-09]] 的分群审计与 [[multimodal-06]] 的分群评测）。否则会出现下面的典型陷阱。

   设两个群体：A（主流口音）WER 8%，B（口音群体）WER 24%。当流量构成是 90/10 时整体 WER 为 $0.9\times8\%+0.1\times24\%=9.60\%$；当用户构成变成 50/50（模型一个字都没改）时整体变成 $0.5\times8\%+0.5\times24\%=16.00\%$，凭空多出 6.4 个点。**整体指标的变化可能全部来自混比漂移**，所以门禁必须按群看（口径见 [[evaluation-04]]）。
4. **口音与 CS 会叠加**：非母语英语的发音会被推向母语的音素空间，让共享音素冲突更频繁；只做口音适配或只做 CS 适配都只能解决一半。

### 6. 评估口径：MER 的定义与不可比陷阱

CS 场景下最常用 MER：

$$
\text{MER} = \frac{S + D + I}{N}
$$

$S,D,I$ 是替换、删除、插入错误数，$N$ 是参考侧 token 数，这类工作里的常见做法是**中文按字、英文按词**计入同一序列。它与 WER 的差别不在公式，而在 token 化与分母：把中英混排文本当成一个序列统计编辑距离，才是「混合错误率」。由此产生三类口径分歧，必须在版本间固定：

- **标点与大小写算不算分母**；
- **拼音、音译是否视为正确**（同一段音频的同一个输出，口径不同可以差出 28 个点，见下节）；
- **数字用 ITN 前后哪种形式**。

SLT 2022 有一项专门为 CS-ASR 建立带人工判断参考基准的研究，比较多种指标与人类判断的相关性，结论是**先做转写（transliteration）处理与文本归一化**的指标与人类判断相关性最高（该工作列在 [[multimodal-06]] 的来源里）。换句话说，指标怎么定义会直接改变结论。

还有一层：MER 只统计 token 级错误，不区分「同一语言内的错词」与「整段语言判断错误」。所以生产上要配两个辅指标——**LID 准确率/语言边界 F1**（边界处允许少量帧容差）与**术语命中率**，否则一个把整句英文翻译成中文的系统可能在 MER 上看起来只是「错了一些词」。

### 7. 上线闭环

数据 → 训练 → 解码 → 评估四层各自的可交付项：真实 CS 语料与其转写规范；统一词表 + LID 联合训练 + 少量适配参数；热词表与术语后处理；按口音/语言分群的评测集与门禁。线上再加两条监控：分群 WER 漂移与**热词插入错误率**（偏置带来的新失败模式），以及低置信度片段的人工抽检回灌。

## 数值与代码验证

下表所有数字的来源与口径都写清；标「自算」的由本节脚本或等价算式得到。

| 量 | 数值 | 来源与口径 |
| --- | --- | --- |
| Whisper 训练监督规模 | 68 万小时（多语言、多任务弱监督，零样本迁移） | Whisper 论文摘要 |
| TALCS 语料 | 约 587 小时、16 kHz，真实线上一对一英语教学场景；论文给出 ESPnet 与 WeNet 基线 | TALCS 论文摘要 |
| TALCS 占 Whisper 语料比例 | $587/680000 = 0.086\%$ | 自算 |
| ISCSLP 2022 CSASR 规模与结果 | 40 余支队伍；冠军 16.70% MER，比基线绝对提升 9.8% $\Rightarrow$ 基线 26.50% MER、相对下降 37.0% | 挑战赛摘要 + 自算 |
| 注意力头引导适配结果 | 14.2% MER，仅额外训练约 5.6% 的参数 | ICASSP 2024 摘要 |
| 5.6% 折算参数量 | 以 Whisper large 约 1.55B 参数计 $\approx 8.7\times10^{7}$ | 自算；论文摘要未指明具体 checkpoint，此数仅作量级参照 |
| 共享音素例子（`one` / 「万」） | 8 个参考 token 中 1 个替换 $\Rightarrow$ 12.50% MER | 自算，见下方脚本 |
| 转写口径差异 | 同一输出：逐字对齐 28.57% MER；若规范接受拼音则 0% | 自算 |
| 归一化口径敏感度 | 80 个错误、1000 token：标点不计分母 8.00%，计入 30 个标点后 7.77%，差 0.23 个点 | 自算 |
| 口音混比漂移 | 8%/24% 两群体，流量 90/10 $\to$ 9.60%；50/50 $\to$ 16.00%，模型未变 | 自算 |
| 分群功效估算 | 检测 15%→17% WER 差（$\alpha=0.05$ 双侧、power 80%）需每组约 5271 个参考 token | 自算 |

```python
# MER / WER 与 S、D、I 分解：确认口径差异到底差在哪里
def align_errors(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

def mer(ref, hyp):
    S, D, I = align_errors(ref, hyp)
    return (S + D + I) / len(ref), (S, D, I)

# 中文按字、英文按词，混排成一个序列
ref = ["这", "批", "data", "的", "size", "是", "10", "万"]
hyp = ["这", "批", "data", "的", "size", "是", "10", "one"]   # 共享音素 /w/ 起头、鼻音韵尾
print(mer(ref, hyp))        # (0.125, (1, 0, 0)) -> 12.50% MER

# 同一段输出，转写规范不同 -> 结论完全不同
ref2 = ["我", "们", "用", "Redis", "做", "缓", "存"]
hyp2 = ["我", "们", "用", "Redis", "做", "huancun"]
print(mer(ref2, hyp2))      # (0.2857..., (1, 1, 0)) -> 若接受拼音则为 0%
```

```python
# 分群报告：功效估算 + 句级 bootstrap（token 在句内相关，不能按 token 独立算区间；依赖上一代码块定义的 align_errors）
import random
z_a, z_b = 1.959964, 0.841621          # alpha=0.05 双侧, power=80%
p1, p2 = 0.15, 0.17
n = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
TOKENS_PER_SEC = 3.0                   # 混合语速口径：约 3 个参考 token/秒
print(f"每组需要 {n:.0f} 个参考 token ≈ {n / TOKENS_PER_SEC / 60:.0f} 分钟音频")
# 组内相关 rho、每句 m 个 token 时设计效应 DE = 1 + (m-1) * rho
for rho in (0.0, 0.02, 0.05):
    de = 1 + (30 - 1) * rho
    print(f"rho={rho}: DE={de:.2f} -> {n * de / TOKENS_PER_SEC / 60:.0f} 分钟/组")

def clustered_wer_ci(utterances, n_boot=2000, seed=0):
    """utterances: [(ref_tokens, hyp_tokens)]；按句重采样，保留句内相关性"""
    rng = random.Random(seed)
    point = sum(align_errors(r, h)[0] + align_errors(r, h)[1] + align_errors(r, h)[2]
                for r, h in utterances) / sum(len(r) for r, _ in utterances)
    stats = []
    for _ in range(n_boot):
        sample = [utterances[rng.randrange(len(utterances))] for _ in utterances]
        err = sum(sum(align_errors(r, h)) for r, h in sample)
        stats.append(err / sum(len(r) for r, _ in sample))
    stats.sort()
    return point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]
```

口径提醒：上面的功效公式把每个参考 token 当成独立伯努利样本，实际同一句内的错误高度相关，所以必须用句级 bootstrap 报区间，并按设计效应放大样本量需求（`rho=0.02` 时约放大到 46 分钟/组）。

## 常见追问

- **追问**：MER 与 WER 在 CS 上的差别到底是什么？
  - 要点：公式相同，差在 token 化与分母——MER 把中英混排转写当成一个序列（中文按字、英文按词）统计 $S+D+I$ 并除以参考 token 数，WER 通常按单语词序列统计。MER 同样不区分错误归属的语言，所以要配 LID 准确率与语言边界 F1。
- **追问**：共享音素（`one` / 「万」）为什么会困扰 CTC？
  - 要点：声学证据不足以区分，CTC 的帧条件独立假设让左右上下文和语言先验都无法进入打分，两个语言的 token 平分后验。修法是引入语言变量：CTC + LID 联合训练、解码时加语言后验约束、或加外部 LM 与词表约束。
- **追问**：生产上用户说中文但夹英文专有名词，怎么保证拼对？
  - 要点：优先级是热词偏置（解码期 $\gamma$ 加权）→ 术语表后处理（正则/映射，处理大小写与连字符）→ 领域数据微调。必须同时监控插入错误率，偏置过强会把发音相近的普通词替换成热词。
- **追问**：口音适配会不会让标准口音的体验变差？
  - 要点：会，属于灾难性遗忘的一类。做法是适配参数与主干隔离（adapter 只对匹配的口音激活）、微调时回放旧数据、门禁上按群判退化而不是只看整体，并保留一键回滚。
- **追问**：合成 CS 数据能不能替代真实数据？
  - 要点：不能。拼接/TTS 合成的切换点过于规整、声学单一，模型会学到合成信道的捷径。可用作预热与辅助任务，评测必须落在真实 held-out 集上，并且真实数据永远优先。
- **追问**：要投多少数据才能确认某口音群体的改善？
  - 要点：先用功效公式定评测集规模（检测 2 个点差异约需每组 5271 个参考 token ≈ 29 分钟音频，按 3 token/s 口径），再按设计效应放大；训练数据量用增量实验定，看分群 WER 曲线的边际收益，而不是先定预算。

## 公司变体

- **Sarvam AI**：产品重心在印度语言与英语混说的语音识别与合成，公开信息里这类系统的核心难点就是多语言/多文字转写规范、罗马化与本地语言混排、以及多口音的覆盖。这题在该公司偏**工程与数据实现**：转写规范怎么定、语料怎么造与标注、分群指标怎么报、产品侧怎么落地；语言建模与指标定义层面的推导会被追问，但通常不作为主线。
- **Abridge**：产品是临床对话的转写与结构化，实际音频是医学英语术语（药物名、检查名、缩写）夹在日常口语里的长尾，说话人来自多元口音群体且录音信道（远场、听诊环境）不理想。这题在该公司偏**工程落地与评测**：热词/术语偏置、术语准确率与遗漏风险、分群审计、以及指标口径与人工复核的配合，对模型内部机制的追问通常停留在「为什么 LID 联合训练有效」这一层。

两家都不会要求现场推导 CTC 前向后向；把「为什么退化 + 怎么修 + 口径怎么写死」讲清楚比背公式更关键。以上是依据公开产品方向与岗位侧重的判断，具体题目以实际面试轮次为准。

## 相关题目

- [[multimodal-06]]：除了 WER 之外如何评估 ASR 与 TTS 质量——指标口径、分群报告与人工判断相关性的完整讨论。
- [[multimodal-03]]：为实时语音 agent 做 latency 预算——流式场景下 CS 与口音带来的额外延迟（重打分、LID 等待）。
- [[finetuning-06]]：LoRA、prefix tuning、prompt tuning 与全量微调的对比——参数高效适配的选型依据。
- [[safety-09]]：审计已部署模型在不同用户群体间的表现差异——分群标签、交叉切片与置信区间。
- [[evaluation-04]]：回归门禁——把口径、MDE 与按群判定固化进上线流程。

## 参考资料与归属

- Radford et al. (OpenAI), *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022-12-06，[arXiv:2212.04356](https://arxiv.org/abs/2212.04356)（延伸）
- Aditya et al. (ICASSP 2024), *Attention-Guided Adaptation for Code-Switching Speech Recognition*, 2023-12-14，[arXiv:2312.08856](https://arxiv.org/abs/2312.08856)（延伸）
- Li et al. (Interspeech 2022), *TALCS: An Open-Source Mandarin-English Code-Switching Corpus and a Speech Recognition Baseline*, 2022-06-27，[arXiv:2206.13135](https://arxiv.org/abs/2206.13135)（延伸）
- Deng et al. (ISCSLP 2022), *Summary on the ISCSLP 2022 Chinese-English Code-Switching ASR Challenge*, 2022-10-12，[arXiv:2210.06091](https://arxiv.org/abs/2210.06091)（延伸）
- Tseng et al., *Mandarin-English Code-switching Speech Recognition with Self-supervised Speech Representation Models*, 2021-10-07，[arXiv:2110.03504](https://arxiv.org/abs/2110.03504)（延伸）

五条来源均为延伸来源。对应关系：68 万小时训练规模与零样本迁移结论来自 Whisper 论文摘要；多语言模型在 CS 上退化、5.6% 额外参数与 14.2% MER 来自 Attention-Guided Adaptation 摘要；587 小时、16 kHz、ESPnet/WeNet 基线来自 TALCS 摘要；40 余支队伍、16.70% MER、9.8% 绝对提升来自 ISCSLP 2022 挑战赛摘要；帧级语言身份、CTC + LID 联合训练与多语言预训练结论来自 Tseng et al. 摘要。SLT 2022 的 CS-ASR 指标基准研究未列为本文来源，仅在正文中按其结论引用（该文列在 [[multimodal-06]] 的来源里）。正文中标注「自算」的数字均由文中脚本与算式复算得出。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
