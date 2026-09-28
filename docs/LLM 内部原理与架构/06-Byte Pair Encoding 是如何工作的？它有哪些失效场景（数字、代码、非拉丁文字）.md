---
type: question
id: llm-internals-06
topic: LLM 内部原理与架构
order: 6
question: Byte Pair Encoding 是如何工作的？它有哪些失效场景（数字、代码、非拉丁文字）？
question_en: How does Byte Pair Encoding work, and what are its failure modes (numbers, code, non-Latin scripts)?
asked_at: [阿里巴巴, Sarvam AI, Hugging Face]
level: 进阶
tags: [bpe, tokenizer, 多语言, 代码]
sources:
  - title: Byte Pair Encoding
    url: https://outcomeschool.com/blog/bpe-in-llms
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Neural Machine Translation of Rare Words with Subword Units（延伸）
    url: https://arxiv.org/abs/1508.07909
    author: Sennrich, Haddow, Birch
    published: 2015-08-31
  - title: SentencePiece: A simple and language independent subword tokenizer（延伸）
    url: https://arxiv.org/abs/1808.06226
    author: Kudo, Richardson
    published: 2018-08-17
related: [llm-internals-07, llm-internals-16, rag]
updated: 2026-09-28
---

## 一句话答案

> BPE 把 tokenization 当成贪心压缩问题：训练时把语料拆成符号序列，反复统计相邻符号对的频次，每轮把最高频的一对合成新符号，重复若干轮得到 merge 表与词表；编码时按 merge 的学习顺序（rank）重放，而不是在新文本上重新统计频次。它用「高频串合成长 token、罕见串退回短 token」同时避开了词级词表的 OOV 与字符级序列过长。
>
> 失效场景的根因只有一个：算法只最大化频次与压缩率，不理会语义边界与数位边界。数字被按固定宽度切块导致数位对不齐；代码里的空白、缩进与长标识符被切碎；非拉丁文字在 byte-level BPE 下从 UTF-8 字节起步，词表没给这个脚本留预算时，同一语义的 token 数可以是英文的十几倍（同一句实测：泰米尔语为英文的 2.4–13.9 倍、中文 1.0–3.4 倍，四种 tokenizer 的对比见第 3 节）。

## 面试官在考什么

- 是否把 BPE 当算法而不是当 API：能不能写出训练循环的目标（每轮按词频加权统计相邻对、合并最高频的一对），而不是只会调用 `tokenizer.encode()`。
- 编码阶段的行为是否清楚：知不知道是**按 rank 重放**；会不会误答成「编码时重新统计频次」或「在词表里做最长匹配」。
- byte-level 的边界：256 字节底表意味着永不 OOV，也意味着非 ASCII 字符天生要占 2–4 个字节，这是多语言 token 膨胀的物理来源，而不是某个实现的缺陷。
- 失效场景是否具体：数字的数位错位、代码的空白与标识符切分、非拉丁脚本的 token 放大，以及这三件事如何传导成成本、上下文占用与推理延迟。
- 词表大小的取舍与缓解手段：embedding 参数量、长尾 token 欠训练、多语言词表配比、数字按位切分的可行性。

**常见错误答案**

- 「BPE 编码时在词表里找最长的匹配子词。」那是贪心最长匹配，与重放 merge 规则不等价：rank 冲突时会切出训练中从未出现过的组合，模型侧的 embedding 也就对不上语义单元。
- 「byte-level 就是按字节切，所以中文一个字固定是 3 个 token」与「非拉丁语言 token 多，把词表调大就行」是相邻的两个误判。前者忽略了词表给中文留了多少预算：实测同一句 15 个汉字在四个 tokenizer 下分别是 34 / 28 / 16 / 10 个 token；后者忽略了词表翻倍会让 embedding 参数翻倍（Qwen3-0.6B 里 embedding 一项已占总参数 26%），而新增长尾 token 出现次数少、训练不充分。

## 原理与推导

### 词级与字符级都不行，subword 是折中

词级 tokenization 的 $|V|$ 要覆盖所有词形：常见词形几十万，加上专名、术语、拼写变体与每天都在产生的新词，没有上界；固定词表下未见词只能映射到 UNK，信息与梯度一起丢掉。Sennrich 等（2015）处理机器翻译时把这点讲得很直接——翻译本身是 open-vocabulary 问题，与其回退到词典，不如把稀有词编码成子词序列：人名靠字符复制或音译、复合词靠组合式翻译、同源词靠形态变换。

字符级 tokenization 没有 OOV，代价是序列膨胀：设平均词长约 $\bar{k}$（英文含空格约 5–6 个字符），同一段文本的字符级 token 数是词级的 $\bar{k}$ 倍；prefill 计算量按 token 数线性增长，attention 项按 token 数平方增长，训练吞吐与上下文窗口都跟着恶化。

subword 的位置就在两者之间：

$$
T_{\text{char}} \ \gg\ T_{\text{subword}} \ \gtrsim\ T_{\text{word}}, \qquad |V|_{\text{subword}} \ll |V|_{\text{word}}
$$

### 训练：每轮合并频次最高的相邻符号对

把语料写成符号序列（初始是字符，byte-level 变体下是字节），词 $w$ 的语料频次为 $f(w)$。相邻符号对 $(a,b)$ 的加权频次是

$$
c(a,b) = \sum_{w\,:\,a,b\ \text{在}\ w\ \text{的切分中相邻}} f(w)
$$

每轮取 $(a^*,b^*) = \arg\max_{a,b} c(a,b)$，在序列里把该对合并成新符号，并把这一对按顺序追加进 merge 表；重复到词表达到目标规模。词表规模的组成关系是

$$
|V| = |V_{\text{base}}| + N_{\text{merges}} + N_{\text{special}}
$$

其中 byte-level BPE 的 $|V|_{\text{base}} = 256$，字符口径下则是初始字符集大小。朴素的每轮重扫语料需要 $O(V \cdot N)$（$N$ 为语料符号数）；工程实现用增量维护的 pair 计数加优先队列，做到接近 $O(N \log N)$。

### 手算 5 步：low / lower / newest / widest

源文用的语料与词频：`low` 5 次、`lower` 2 次、`newest` 6 次、`widest` 3 次，每个词末尾加词尾标记 `_`，初始符号集 `{l, o, w, e, r, n, s, t, i, d, _}`。首轮统计里 `es`、`st`、`t_` 各 9 次并列最高（我复算的 top5：`es` 9、`st` 9、`t_` 9、`we` 8、`lo` 7），选谁由并列打破规则决定，源文与我的实现都得到先合并 `e+s`。之后的轮次：

1. `e+s → es`（9）：`newest` → `n e w es t _`
2. `es+t → est`（9）
3. `est+_ → est_`（9）：`newest` 与 `widest` 各得到一个 `est_`
4. `l+o → lo`（7）
5. `lo+w → low`（7）
6. `e+w → ew`（6）、`ew+est_ → ewest_`（6）、`n+ewest_ → newest_`（6）：整词被合成单个 token
7. `low+_ → low_`（5）、`d+est_ → dest_`（3）：继续吃掉长尾

这 10 步的完整 merge 表与首轮并列情况在第 4 节给出（那是我自己的实现输出，不是源文数字）。每次合并都会让符号序列变短、词表加一，直到预设的 $|V|$——源文给的量级是 32K–256K。

### 编码：按 merge 的 rank 重放

训练产物是**有序**的 merge 表。编码新文本时按 merge 的学习顺序逐条尝试合并，命中就改序列，直到没有可用的 merge 为止：

```python
# merges 是训练产物的有序表，下面的骨架就是编码主循环
rank = {pair: i for i, pair in enumerate(merges)}   # 训练顺序即 rank
syms = list(text)
while len(syms) > 1:
    cand = [p for p in zip(syms, syms[1:]) if p in rank]
    if not cand:
        break
    best = min(cand, key=rank.get)                   # 只看 rank，不重新统计频次
    # 在 syms 里把所有 best 合并掉后进入下一轮（完整可运行版本见第 4 节）
```

**为什么必须按 rank 而不是重新统计频次**：三个理由。其一，编码时只有一条文本，语料级频次 $c(a,b)$ 无从计算。其二，就算在当前文档上重算，合并顺序会随文档内容变化，同一段文字在不同上下文里切法不同，而模型的 embedding 是按训练时的切分学出来的，切分漂移等于输入分布漂移。其三，rank 重放是确定性的（同一文本永远同一 token 序列），可以用「pair 的 rank 作键」的优先队列实现，单次编码接近 $O(n \log n)$。

用上表验证：`lowest` → `l o w e s t` →（rank1）`l o w es t` →（rank2）`l o w est` →（rank4）`lo w est` →（rank5）`low est`，结果 `[low, est]`，与源文一致。

### byte-level BPE：永不 OOV 的代价

GPT-2 起主流做法是先把文本按正则预切分，再把每段转成 UTF-8 字节，在字节上跑 BPE。基础词表固定 256 个字节，因此任意 Unicode 文本都能编码，最坏情况退化成逐字节 token，OOV 这个概念消失。代价是字节化让非 ASCII 字符天生变贵：UTF-8 下一个汉字 3 字节、一个天城文字母 3 字节、一个泰米尔文字母 3 字节、常见 emoji 4 字节；只有当某字符对应的完整字节序列在预训练语料里高频到被合并成单个 token，它才享受「1 字符 1 token」的待遇。

| tokenizer | 学习到的 merge 数 | 底表 | 词表 `n_vocab` | 差额（special + 预留 id） |
| --- | --- | --- | --- | --- |
| GPT-2 (`gpt2`) | 50000 | 256 字节 | 50257 | 1（special 占 1 个 id） |
| GPT-4 系 (`cl100k_base`) | 100000 | 256 字节 | 100277 | 21（注册的 special 只有 5 个，最大 id 100276，其余 16 个 id 未使用） |
| GPT-4o 系 (`o200k_base`) | 199742 | 256 字节 | 200019 | 21（注册的 special 只有 2 个，最大 id 200018，其余 19 个 id 未使用） |

这张表是用本地 tiktoken 0.12.0 读出来的：`mergeable_ranks` 的长度减去 256 个字节底表就是「学习到的 merge 数」，$|V| = 256 + N_{\text{merges}} + N_{\text{special 与预留 id}}$ 逐项成立；差额里只有一部分是真正注册的 special token，其余 id 是空洞。表中数字不是源文数字。

### 失效场景一：数字

数字怎么切由「预切分正则允许几位数字成块」和「BPE 合并了哪些数字串」共同决定，结果与数位无关：

- `cl100k_base` / `o200k_base`：左起每 3 位一组。`123 → [123]`（1 个 token）、`1234 → [123][4]`、`12345 → [123][45]`、`1234567 → [123][456][7]`。我用 2677 个随机 4–9 位纯数字验证，全部符合这个规则。
- 后果不是「token 多」，而是**同一个 token 落在不同数位**：token id 4513 就是字符串 `123`，在 `1234` 里它覆盖千位到十位（值 1230），在 `1234567` 里覆盖百万位到万位（值 1230000）。token 本身不携带数位信息，模型只能靠上下文反推它在数字串中的位置，学不到「同位相加」这种归纳偏置。
- `gpt2` 没有位置规则：数字块完全取决于预训练语料里哪些数字串被合并过，1–4 位混排——`1234 → [12][34]`、`1234567 → [123][45][67]` 只是恰好成立，换几个数字就散架：`24177 → [24][177]`、`437406 → [437][406]`、`1000 → [1000]`（单个 token）、`999999 → [9999][99]`。对 2000 个随机 4–10 位数字统计它的数字 token 长度分布，得到 1 位 477 次、2 位 3668 次、3 位 1943 次、4 位 38 次，没有任何按位分组的规律。`o200k_base` 与 `cl100k_base` 在上面列出的数字上切法完全一致。
- 同一个数的不同写法切法也不同（`cl100k_base`）：`1000 → [100][0]`、`1,000 → [1][,][000]`、`1 000 → [1][ ][000]`，数值相等的字符串没有稳定的 token 表示；换 `gpt2`，`1000` 又变成单个 token、`1 000` 变成 `[1][ 000]`。

### 失效场景二：代码

- **空白与缩进靠压缩率吃饭**：4 空格缩进在 `return` 前被切成 `'   '` + `' return'`（空白附着到后一个词），tab 缩进则是单个 `'\treturn'`。cl100k 实测：`'    return a + b'` 是 5 个 token，`'\treturn a + b'` 是 4 个 token，而连续 16 个空格只占 1 个 token。缩进越深反而越省——这是压缩率的胜利，不是语法的胜利。
- **标识符切分对命名风格敏感**：camelCase 常被切在词素边界（`getUserAccountBalanceByUserId → [getUser][Account][Balance][By][UserId]`，5 个 token）；snake_case 更容易被切断（`very_long_snake_case_identifier_name → [very][_long][_sn][ake][_case][_identifier][_name]`，7 个 token，`snake` 被劈成 `_sn` + `ake`）。
- **跨语言迁移不平等**：不同编程语言的语料占比不同，tokenizer 对它们的压缩率也不同，同一段逻辑的 token 数不一样，等效上下文、推理成本、固定窗口里能放的代码量都随之变化。「模型在 Python 上强、在小众 DSL 上弱」有一部分是 tokenizer 造成的，不是模型容量造成的。

### 失效场景三：非拉丁文字

物理来源已经在 byte-level 一节说明：非 ASCII 字符至少 2 个字节，能否压成单个 token 取决于词表是否给这个脚本留了位置。同一句意思的平行句，四种 tokenizer 的实测对比（口径：tiktoken 0.12.0 的 `gpt2` / `cl100k_base` / `o200k_base`，以及 `Qwen/Qwen3-0.6B` 的 `tokenizer.json`，句义相同的六种语言各一句）：

| 语言 | 字符数 | UTF-8 字节 | gpt2 | cl100k | o200k | Qwen3-0.6B | cl100k 相对英文 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 英文 | 44 | 44 | 10 | 10 | 10 | 10 | 1.0x |
| 中文 | 15 | 45 | 34 | 28 | 16 | 10 | 2.8x |
| 印地语（天城文） | 43 | 113 | 70 | 48 | 19 | 44 | 4.8x |
| 泰米尔语 | 51 | 139 | 139 | 69 | 24 | 57 | 6.9x |
| 阿拉伯语 | 42 | 77 | 40 | 30 | 16 | 18 | 3.0x |
| 韩语 | 23 | 57 | 54 | 24 | 17 | 21 | 2.4x |

读法与结论：

- `gpt2` 时代泰米尔语句子 51 个字符直接用掉 139 个 token，恰好等于它的 UTF-8 字节数——几乎没有字节串被合并，等于退化成字节回退。
- `cl100k` 把 15 个汉字压到 28 个 token（约 1.9 token/字），泰米尔语仍是英文的 6.9 倍。
- `o200k` 靠更大的词表把泰米尔语降到 24（2.4 倍）、印地语降到 19（1.9 倍）。
- `Qwen3-0.6B` 面向中文，中文降到 10（与英文持平），但印地语仍是 44（4.4 倍）。**tokenizer 的多语言能力是词表配比的产物，不是算法的产物**；源文那句「BPE 不改算法就能支持任何语言」在「不会崩」的意义上成立，在「代价相同」的意义上不成立。
- 成本换算：设同一内容在某语言下的 token 数是英文的 $\rho$ 倍，则 prefill 计算量（约 $2NT$，$N$ 为参数量、$T$ 为 token 数）与 KV cache 字节数（$2LHd_{\text{head}}Tb$，$b$ 为每元素字节数）都按 $\rho$ 线性放大，attention 项按 $\rho^2$ 放大。中文在 `cl100k` 下 $\rho = 2.8$，同一句话的 prefill 成本与 KV cache 是英文的 2.8 倍，attention 项是 $2.8^2 = 7.8$ 倍。上下文窗口装的是 token 不是字符，所以同样的 8K 窗口在泰米尔语下能装的内容不足英文的六分之一。
- 公平性方向：tokenizer 造成的语言间不平等已有专门量化（Petrov 等，NeurIPS 2023；Ahia 等，EMNLP 2023），结论方向是同一语义在不同语言下的 token 数差异，把成本与能力差距转嫁给了非英语用户。具体倍数各家口径不同，本条只取方向。

### 与 WordPiece、Unigram 的区别

| 维度 | BPE | WordPiece | Unigram（SentencePiece） |
| --- | --- | --- | --- |
| 起点与动作 | 字符/字节起步，逐轮合并 | 字符起步，逐轮合并 | 大候选集起步，逐轮剪枝 |
| 选择标准 | 相邻对频次最高 | 合并后语料似然提升最大，约 $c(ab)/(c(a)\,c(b))$ | 删掉后语料似然损失最小 |
| 是否概率模型 | 否，只有 merge 表 | 否，只有打分表 | 是，EM 估计子词概率 |
| 输出概率 | 无（可用频次近似） | 无 | 有，可做 subword regularization 采样 |
| 空格处理 | byte-level 用 `Ġ` 表示空格，字符口径用词尾标记 `_` | 词首子词不加前缀，续接子词加 `##` | 用 `▁`（U+2581）表示空格，直接吃原始文本 |
| 代表模型 | GPT-2/GPT-4 系、Llama、Qwen | BERT | T5、mT5、ALBERT |

两点澄清。第一，SentencePiece 是实现框架（可配置 BPE 或 Unigram），不是与 BPE 并列的算法；它真正的贡献是语言无关——直接在原始句子（不预分词）上训练，因此对中文、日文这类不写空格的语言同样可用（Kudo & Richardson, 2018）。第二，subword regularization 把「一种文本只有一种切分」变成分布：Unigram 按概率采样切分做数据增强，BPE 的近似版本是 BPE-dropout（训练时以概率 $p$ 跳过某些 merge）。

### 词表大小的权衡与缓解手段

- **压缩率 vs 参数量**：词表越大，平均 token 数越少（序列更短，prefill 与 KV cache 更省），但 embedding 参数 $|V| \cdot d$ 线性增长。以 Qwen3-0.6B 的公开配置为例（`vocab_size = 151936`、`hidden_size = 1024`、`tie_word_embeddings = true`）：单份 embedding 是 $151936 \times 1024 = 155{,}582{,}464$ 个参数，我按 config 逐项相加得到总参数约 596M，embedding 占 26%；若不共享输入输出 embedding，总参数会增到约 752M。
- **长尾 token 欠训练**：新加进词表的子词在语料里出现次数少，梯度信号弱，embedding 学不好；扩词表续训时必须给新 token 合理初始化并跑够步数，只换 tokenizer 不重训等于把 embedding 随机化。
- **多语言配比**：词表预算是零和分配，某语言分到的 token 越多，其它语言越吃亏。常见做法是提高低资源语言的采样率（temperature sampling）并相应扩大词表，而不是直接扩大词表。
- **专用 tokenizer**：Qwen 系为中文与代码单独设计词表（本地 `tokenizer.json` 显示它是 byte-level BPE，151387 条 merge、26 个 added token，基础词表 $256 + 151387 = 151643$）；面向印度语言的团队则要为天城文、泰米尔文等脚本单独配置预切分与词表预算，否则同一份内容要多付几倍 token。
- **数字按位切分**：Qwen3 的预切分正则用 `\p{N}` 把每个数字位单独切开，`1234 → [1][2][3][4]`——数位对齐了，token 数从 2 涨到 4。另一条路是保留分块但从右对齐（千分位分组），让同一个 token 永远处在同一数位。代价都真实存在，不存在免费解。

## 数值与代码验证

**经典例子的 merge 序列（我实现并运行的结果）**：语料 `low`×5、`lower`×2、`newest`×6、`widest`×3，词尾标记 `_`，合并 10 轮。

| 轮次 | 合并 | 频次 | 轮次 | 合并 | 频次 |
| --- | --- | --- | --- | --- | --- |
| 1 | `e + s → es` | 9 | 6 | `e + w → ew` | 6 |
| 2 | `es + t → est` | 9 | 7 | `ew + est_ → ewest_` | 6 |
| 3 | `est + _ → est_` | 9 | 8 | `n + ewest_ → newest_` | 6 |
| 4 | `l + o → lo` | 7 | 9 | `low + _ → low_` | 5 |
| 5 | `lo + w → low` | 7 | 10 | `d + est_ → dest_` | 3 |

首轮 `es`、`st`、`t_` 并列 9（`we` 8、`lo` 7），并列打破规则因实现而异；源文选 `e+s`，我用「频次优先、同级取字典序」，同样先合并 `e+s`，并且第 3–5 轮落成 `est_`、`lo`、`low`，与源文描述的顺序一致。第 10 轮后的切分：`low_ → [low_]`、`lower_ → [low, e, r, _]`、`newest_ → [newest_]`、`widest_ → [w, i, dest_]`。

**可运行的最简实现**（训练用词频加权统计，编码只按 rank 重放）：

```python
from collections import Counter

def train_bpe(words, n_merges):
    """words: {词 + 词尾标记: 频次} -> 按 rank 排序的 merge 表"""
    seqs = {w: list(w) for w in words}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, syms in seqs.items():
            for a, b in zip(syms, syms[1:]):
                pairs[(a, b)] += words[w]           # 按词频加权，不是按出现次数
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(best)                          # 频次优先，字典序打破并列
        for w in seqs:
            syms, out, i = seqs[w], [], 0
            while i < len(syms):
                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:
                    out.append(syms[i] + syms[i + 1]); i += 2
                else:
                    out.append(syms[i]); i += 1
            seqs[w] = out
    return merges

def encode(text, merges):
    rank = {pair: i for i, pair in enumerate(merges)}
    syms = list(text)
    while len(syms) > 1:
        cand = [p for p in zip(syms, syms[1:]) if p in rank]
        if not cand:
            break
        best = min(cand, key=rank.get)               # 只看 rank，从不重算频次
        out, i = [], 0
        while i < len(syms):
            if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:
                out.append(syms[i] + syms[i + 1]); i += 2
            else:
                out.append(syms[i]); i += 1
        syms = out
    return syms

words = {"low_": 5, "lower_": 2, "newest_": 6, "widest_": 3}
merges = train_bpe(words, 10)
print([a + b for a, b in merges])
# ['es', 'est', 'est_', 'lo', 'low', 'ew', 'ewest_', 'newest_', 'low_', 'dest_']
print(encode("lowest", merges))   # ['low', 'est']
```

**数字切分（同一条数字串在四种 tokenizer 下）**：

| 数字串 | gpt2 | cl100k_base | o200k_base | Qwen3-0.6B |
| --- | --- | --- | --- | --- |
| 1234 | `[12][34]` | `[123][4]` | `[123][4]` | `[1][2][3][4]` |
| 12345 | `[123][45]` | `[123][45]` | `[123][45]` | `[1][2][3][4][5]` |
| 1234567 | `[123][45][67]` | `[123][456][7]` | `[123][456][7]` | `[1]…[7]` |
| 1000000 | `[1][000000]` | `[100][000][0]` | `[100][000][0]` | `[1][0][0][0][0][0][0]` |
| 1000 | `[1000]` | `[100][0]` | `[100][0]` | `[1][0][0][0]` |
| 24177 | `[24][177]` | `[241][77]` | `[241][77]` | `[2][4][1][7][7]` |

`cl100k_base` 与 `o200k_base` 的左起 3 位规则在 2677 个随机 4–9 位纯数字上 100% 成立（再抽 3000 个 10–12 位数字同样成立）；`gpt2` 则在同一批数字上做不到这一点：它的数字块长度 1–4 位混排（见上一节的长度分布），上表最后两行就是反例。`123` 这个 token 的 id 是 4513，在 `1234` 里覆盖千位到十位，在 `1234567` 里覆盖百万位到万位，数位对不齐的根源就在这里。`Qwen3-0.6B` 的词表里 151936 是 config 声明值（对齐填充用），`tokenizer.json` 里的实际词表是 151669（151643 + 26 个 added token）。以上 token 计数与切分全部来自本地实测，源文没有给这些数字。

## 常见追问

- **追问**：编码时为什么不重新统计一遍频次，按当前文本重算不是更「准」吗？
  - 要点：编码时只有单条文本，语料级频次 $c(a,b)$ 根本不存在；即便在当前文档上重算，合并顺序会随内容漂移，切出训练时未出现的组合，而 embedding 是按训练切分学出来的。rank 重放保证确定性、可复现，并且能用「pair 的 rank」做优先队列，单次编码接近 $O(n \log n)$。
- **追问**：byte-level BPE 为什么永不 OOV？代价是什么？
  - 要点：底表就是 256 个字节，任意 Unicode 先转 UTF-8，最坏退化成逐字节 token。代价是非 ASCII 字符占 2–4 个字节（汉字/天城文/泰米尔文 3 字节、常见 emoji 4 字节），只有高频字节串才会被合并成单 token；实测 `gpt2` 下泰米尔语句子 51 字符用掉 139 个 token，等于一个字节都没合并。
- **追问**：中文一个字是几个 token？
  - 要点：取决于词表给了中文多少预算。同一句 15 个汉字：`gpt2` 34 个 token、`cl100k_base` 28、`o200k_base` 16、`Qwen3-0.6B` 10。「中文 1 字 1–2 token」这种说法只在中文配比充足的词表上成立。
- **追问**：词表是不是越大越好？
  - 要点：词表越大压缩率越高、序列越短，但 embedding 参数按 $|V| \cdot d$ 线性增长（Qwen3-0.6B 里占 26%），新增长尾 token 还训练不充分。扩词表要配合继续预训练，否则等于给 embedding 重新随机初始化。
- **追问**：让你重新设计 tokenizer 解决数字问题，你会怎么做？
  - 要点：三条路都要说清代价。(1) 预切分层把数字位拆开（如 Qwen3 的 `\p{N}`），数位对齐，token 数翻倍；(2) 保留分块但固定位宽并从右对齐，让同一个 token 始终处在同一数位；(3) 不动 tokenizer，改数据与训练目标（数字对齐任务、统一写成带分隔符的形式）。没有免费方案。
- **追问**：subword regularization 是什么，为什么要它？
  - 要点：把「一种文本只有一种切分」变成分布。Unigram 按概率采样切分（Kudo & Richardson, 2018），BPE 的近似版本是 BPE-dropout：训练时以概率 $p$ 跳过一些 merge。作用是数据增强、降低对单一分词的过拟合；推理时仍用确定性切分。

## 公司变体

- **阿里巴巴（Qwen）**：偏工程实现与词表设计取舍。Qwen 系 tokenizer 是 byte-level BPE（本地 `tokenizer.json` 可见 `model.type = BPE`、151387 条 merge、26 个 added token），中文与代码的压缩率是明确的设计目标；问题通常落在「词表大小与多语言配比怎么定」「预切分正则为什么这么写」「数字为什么按位切」这类可验证的实现决策上，而不是 BPE 的数学推导。
- **Sarvam AI**：偏多语言 tokenizer 与成本公平性。这家公司面向印度语言做模型与语音产品，考察自然会落到天城文、泰米尔文等脚本的 token 膨胀如何影响上下文预算与单位成本，以及为低资源语言单独设计词表与预切分的取舍。公开信息只支撑到「他们做印度语言方向」，具体面试流程不宜脑补。
- **Hugging Face**：偏工具链与实现细节。`tokenizers` 库同时提供 BPE、WordPiece、Unigram 三种模型与 ByteLevel 预切分/解码器，问题常落在「怎么从零训一个 byte-level BPE」「训练与推理的 round-trip 是否一致」「added token 与 special token 怎么处理」「fast tokenizer 的 offset/alignment 有什么用」。第 4 节那两个函数正好可以对接这些细节。

以上依据各家公开的模型、tokenizer 文件与工具产出归纳，不代表任何具体的面试流水。

## 相关题目

- [[llm-internals-07]]：位置编码。token 切分决定序列长度，位置编码决定模型如何使用这些位置，讨论长上下文时两者必须一起看。
- [[llm-internals-16]]：decoder-only transformer 的一次前向传播。token → embedding → logits 的链路正是 tokenizer 与模型参数的交界，$|V|$ 直接决定 embedding 与输出层的参数量。
- [[rag]]：RAG 与检索。chunk 策略、embedding 与 API 计费都按 token 计价，非拉丁语料的 token 膨胀会直接改变 chunk 大小与成本上限（该专题尚未撰写，站点会显示为待撰写）。

## 参考资料与归属

- **Byte Pair Encoding** —— Amit Shekhar（Outcome School），源站标注 2026-03-31：<https://outcomeschool.com/blog/bpe-in-llms>。训练与编码流程、32K–256K 的词表量级、`low`/`lower`/`newest`/`widest` 例子与「编码重放 merge 规则」的表述来自这篇。
- **Neural Machine Translation of Rare Words with Subword Units（延伸）** —— Sennrich, Haddow, Birch，2015-08-31：<https://arxiv.org/abs/1508.07909>。第 3 节的 open-vocabulary 动机、subword 折中与带词尾标记的 merge 训练依据这篇（该文报告 subword 模型在 WMT15 英德、英俄任务上比词典回退基线分别高 1.1 与 1.3 BLEU）。
- **SentencePiece: A simple and language independent subword tokenizer（延伸）** —— Kudo, Richardson，2018-08-17：<https://arxiv.org/abs/1808.06226>。「直接在原始文本上训练、语言无关」的设计与 subword regularization 依据这篇；第 3 节的三方对比表与第 5 节最后一条追问由此延伸而来。
- 延伸阅读（未列入 front matter 来源，仅支撑「tokenizer 公平性」的方向性结论，未引用其中任何数字）：Petrov 等，*Language Model Tokenizers Introduce Unfairness Between Languages*，NeurIPS 2023，<https://arxiv.org/abs/2305.15425>；Ahia 等，*Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models*，EMNLP 2023，<https://aclanthology.org/2023.emnlp-main.614/>。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
