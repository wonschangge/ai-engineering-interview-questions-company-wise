---
type: question
id: alibaba-02
company: 阿里巴巴（Qwen）
topic: llm-internals
order: 2
question: Qwen 使用字节级 BPE，词表约 151K，为多语言覆盖做了增强，并把数字拆成单个字符。为什么做这些选择，各自的取舍是什么？
question_en: Qwen uses byte-level BPE with a ~151K vocabulary, augmented for multilingual coverage and with digits split into single characters. Why those choices, and what are the trade-offs?
asked_at: []
level: 进阶
tags: [bpe, tokenizer, 多语言, 数字切分, 词表大小]
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
  - title: 大型语言模型（LLM）中的 Tokenization
    url: https://www.youtube.com/watch?v=sK2s9I84EVI
    author: 
    published: 
related: [llm-internals-06, llm-internals-16, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> 三个选择各自对应一笔账。byte-level 把底表固定成 256 个字节，永不 OOV，代价是非 ASCII 字符天生占 2–4 字节，能否压成单 token 全靠词表预算；151K 词表就是把这笔预算花在中文、代码与多语言上，换来更短的序列与更省的 KV cache，代价是 embedding 参数按 $|V|d$ 线性增长（Qwen3-0.6B 里占 26%）以及长尾 token 训练不足；数字逐位切分放弃压缩率换数位对齐——token id 本身不携带数位信息，`123` 在 `1234` 与 `1234567` 里覆盖的数位完全不同，模型学不到「同位相加」。
> 判断依据是场景：token 是计费、上下文与显存的共同单位，数字密集型负载优先数位对齐，多语言服务优先词表配比，两者都要用实测 token 数收口，不能靠「字符数 ÷ 4」估。

## 面试官在考什么

- 能不能把「byte-level BPE」拆到实现层：预切分正则怎么切 chunk、ByteLevel 怎么把 chunk 变成 UTF-8 字节、编码为什么是**按 merge 表的学习顺序重放**而不是词表最长匹配或重新统计频次。停在「Qwen 用了 byte-level BPE」是背结论，不是答机制。
- 词表大小的账是否会算：$|V| = 256 + N_{\text{merges}} + N_{\text{special 与预留 id}}$ 能不能落到本地 `tokenizer.json` 的真实数字上，embedding 与输出层 logits 怎么随 $|V|$ 增长，config 里声明的 `vocab_size` 与实际词表为什么不是一回事。
- 是否理解多语言能力是**词表预算与配比**的产物，不是 BPE 算法的产物：同一套算法只换词表预算，同一句中文的 token 数能从 34 掉到 10。
- 数字切分是不是真懂机制：有没有注意到 Qwen 的预切分正则是单字符 `\p{N}`，而 cl100k 系是 `\p{N}{1,3}` 左起三位分组，以及两种切法各自破坏了什么、保住了什么。
- 取舍能否量化到成本：同一内容相对英文的 token 倍数 $\rho$ 如何线性放大 prefill 与 KV cache、平方放大 attention，数字逐位后序列变长多少。

**常见错误答案**

- 「byte-level BPE 就是按字节切，所以一个汉字固定 3 个 token。」底表是 256 个字节不假，但合并阶段会把高频字节串合成单 token：实测同一句 15 个汉字在 `gpt2` 下 34 个 token、`Qwen3` 下 10 个；常见中文短句在 Qwen3 上落在 0.45–0.67 token/字，`cl100k` 下是 1.07–1.87 token/字。
- 「数字逐位切分是为了让模型逐位算数，代价就是 token 变多、没别的。」真正的收益是数位对齐（同一个 token 永远处在同一量级），真正的代价不止长度：数值等价的不同写法切法不同（`1000` 与 `1,000`），结构化输出的数字稳定性要靠约束解码兜底。

## 原理与推导

### 1. 从文本到 token 的四步流水线

Qwen 系 tokenizer 是 byte-level BPE（本地 `tokenizer.json` 的 `model.type = BPE`），编码一条文本要走四步：

1. **预切分（pre-tokenizer）**：用正则把文本切成 chunk。Qwen 系的正则是

   ```
   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
   ```

   注意第三支 `\p{N}` 是**单字符**匹配——这就是数字逐位切分的实现位置。`cl100k_base` 与 `o200k_base` 在同一位置写的是 `\p{N}{1,3}`，`gpt2` 是 `\p{N}+`。
2. **ByteLevel**：每个 chunk 先转成 UTF-8 字节，再映射到 256 个「可见字节符号」上（空格写成 `Ġ` 这类单字节表示）。底表恒为 256，任意 Unicode 文本都能编码，最坏情况退化成逐字节 token，OOV 这个概念消失。
3. **BPE 合并**：在字节符号序列上按训练好的 merge 表重放。训练时每轮统计相邻符号对的词频加权频次 $c(a,b) = \sum_{w: a,b \text{ 在 } w \text{ 中相邻}} f(w)$，把最高的那对合成新符号并按顺序追加进 merge 表；这个顺序就是 rank。
4. **编码 = rank 重放**：新文本只按 merge 的 rank 从小到大尝试合并，命中就改序列，**绝不重新统计频次、也不做词表最长匹配**。原因是语料级频次在单条文本上不存在，而且一旦按当前文档重算，同一段文字在不同上下文里切法会漂移，embedding 是按训练时的切分学出来的，切分漂移等于输入分布漂移。rank 重放同时保证确定性与可复现：用「pair 的 rank」做键的优先队列，单次编码接近 $O(n \log n)$。

词表规模恒等式是本题的记账基础：

$$|V| = 256 + N_{\text{merges}} + N_{\text{special 与预留 id}}$$

### 2. byte-level 的物理代价与 151K 词表的动机

字节化让非 ASCII 字符天生变贵：UTF-8 下一个汉字 3 字节、一个天城文字母 3 字节、一个泰米尔文字母 3 字节、常见 emoji 4 字节。也就是说，这些字符的起跑线是 3–4 个 token，只有当它对应的完整字节序列在预训练语料里高频到被 merge 成单个 token，才享受「一字一 token」的待遇。**压缩率完全由词表给这个脚本留了多少位置决定，不是实现缺陷。**

一个规模矛盾能说明为什么必须把词表预算堆到 151K 量级：中文常用字 3000–7000 个，若要让它们接近「一字一 token」，光汉字就要吃掉词表的三分之一；再叠上代码（缩进、常见标识符、语言关键字）与其它脚本，32K 词表根本不够分。Qwen 系把词表做到 151K，是把预算按中文与代码优先、其余语言按语料占比分配的产物。

### 3. 151K 词表的参数账：词表预算是零和的

用 Qwen3-0.6B 的公开配置逐项复算（`vocab_size = 151936`、`hidden_size = 1024`、`num_hidden_layers = 28`、`tie_word_embeddings = true`）：

- 单份 embedding：$151936 \times 1024 = 155{,}582{,}464 \approx 155.6\text{M}$
- 每层 attention（$H_q=16$、$H_{kv}=8$、$d_{head}=128$）：$16 \times 128 \times 1024 + 2 \times 8 \times 128 \times 1024 + 16 \times 128 \times 1024 = 6{,}291{,}456$
- 每层 MLP（`intermediate_size = 3072`，SwiGLU 三个矩阵）：$3 \times 1024 \times 3072 = 9{,}437{,}184$
- 总参数：$155.6\text{M} + 28 \times 15.73\text{M} + 59{,}392 \approx 596\text{M}$，embedding 单独占 **26.1%**

词表从 32K 提到约 151K（$151936 / 32768 \approx 4.64\times$）买到的是压缩率：平均序列更短，prefill 计算量与 KV cache 同比例下降，上下文窗口能装更多内容。付出的代价有三项：

1. **参数量线性增长**：embedding 按 $|V|d$ 线性长；若不共享输入输出 embedding，Qwen3-0.6B 的总参数从 596M 涨到 752M，多出一整份 155.6M。
2. **输出层 logits 与 $|V|$ 线性**：每 token 约 $2|V|d = 2 \times 151936 \times 1024 \approx 3.11 \times 10^{8}$ FLOPs。decode 阶段这部分**不能被 MoE 稀疏化**——无论模型总参数多少，每个 token 都要过一次满词表投影，所以大词表对小激活模型是固定税。
3. **长尾 token 欠训练**：新加进词表的子词在语料里出现次数少、梯度弱，embedding 学不好。扩词表必须配继续预训练与合理初始化；只换 tokenizer 不重训，等于把 embedding 随机化。

词表还有一条容易忽略的对齐约束：added token 必须与 `tokenizer.json` 一致（Base 22 个、Instruct 26 个，id 紧接 151642 之后连续排布）。内容或数量错位会让 chat template 渲染出的 `<|im_start|>` 这类特殊串被切成普通子词，fast tokenizer 的 offset 与字符对齐也跟着漂，直接污染依赖 span 的训练与评测。

### 4. 词表预算怎么分配：多语言增强的真实含义

词表预算是零和分配，某语言分到的 token 越多，其它语言越吃亏。常见做法不是「把词表调大」这一件事，而是**提高低资源语言的采样率**（temperature sampling 之类的语料重加权）并相应扩词表，让新分到的 token 真的在训练里被看到。

同一句意思的平行句在四种 tokenizer 下的实测差异（口径：tiktoken 0.12.0 的 `gpt2` / `cl100k_base` / `o200k_base`，加 `Qwen3-0.6B` 的 `tokenizer.json`，六种语言各一句，原始语料与口径见 [[llm-internals-06]] 第 3 节）见下节表格。两处读法必须点出：

- `gpt2` 下泰米尔语句子 51 个字符用掉 **139 个 token，恰好等于它的 UTF-8 字节数 139**——一个字节串都没被合并，等于退化成纯字节回退。
- `Qwen3-0.6B` 面向中文，中文 10 token 与英文 10 token 持平（$\rho = 10/10 = 1.0$）；而 `cl100k` 下中文 $\rho = 28/10 = 2.8$、泰米尔语 $\rho = 69/10 = 6.9$。同一套 BPE 算法，只换词表预算，同一句中文从 34 token（`gpt2`）变 10 token，**3.4×**。

Qwen 的多语言增强由此可以精确定义：不是改算法，而是把词表预算与语料配比按目标语言重排。它的效果有边界——同一张表里 Qwen3 的印地语仍是 44 token（英文的 4.4 倍），因为天城文语料没分到同等预算。

落地到成本预算还有一条：token 数不能按「字符数 ÷ 4」折算。15 个汉字 ÷4 得 3.75，`cl100k` 实测 28（低估约 7.5×）、`Qwen3` 实测 10（低估约 2.7×）；batch 组包、上下文截断与单价估算都要按内容语言实测的 $\rho$ 算，字符数是不安全的代理量。

### 5. 数字逐位切分：放弃了什么，换到了什么

先看事实。数字怎么切由「预切分允许几位数字成块」与「BPE 合并了哪些数字串」共同决定，结果与数位无关：

- `cl100k_base` / `o200k_base`：`\p{N}{1,3}`，左起每 3 位一组。用随机数字串验证，2677 个 4–9 位与 3000 个 10–12 位纯数字上 100% 成立（[[llm-internals-06]] 第 4 节），我自己再抽 500 个 4–9 位也全部符合。
- `gpt2`：`\p{N}+`，没有位宽规则，切法完全取决于预训练语料里哪些数字串被合并过。
- `Qwen3`：预切分正则里的 `\p{N}` 是单字符分支，每个数字位单独成一个 chunk，**逐位切分**。

动机是数位对齐。token id 4513 就是字符串 `123`：在 `cl100k` 的 `1234` 里它覆盖千位到十位（数值 1230），在 `1234567` 里它覆盖百万位到万位（数值 1230000）。同一个 token 落在不同数位，token 本身不携带数位信息，模型只能靠上下文反推它在数字串里的位置，学不到「同位相加」这类归纳偏置——这对算术、财务对齐、表格数值比较都是直接损伤。

代价要量化：按左起 3 位分组口径（`cl100k` / `o200k`）对比，同一数值的 token 数从 2 涨到 4（`1234`）、从 3 涨到 7（`1000000`）；$n$ 位纯数字逐位后的长度是 3 位分组口径的 $n / \lceil n/3 \rceil$ 倍，4–10 位区间即 2.0–3.0 倍。数字密集型内容（表格、日志、时间戳、哈希、数学题）的序列长度、prefill 计算量与 KV cache 同比例放大，按 token 截断时也会吃掉更多样本，训练集与评测集必须用同一条截断规则，否则指标差异会被误读成模型能力差异。还有一条更隐蔽的代价：**逐位切分并不能保证数值等价写法切法一致**——`1000` 已经是 `[1][0][0][0]`，但 `1,000` 会切成 `[1][,][0][0][0]`，千分位、空格分隔、科学计数法都会引入额外 token。所以结构化输出里的数字稳定性要靠约束解码（grammar / JSON schema 约束）保证，不能指望 tokenizer。

### 6. 取舍收口：四条主线各自的方向

- **byte-level 的硬币两面**：「永不 OOV」与「非 ASCII 天生占 2–4 字节」是同一个设计的两面。能否压成单 token 取决于词表预算，不是实现缺陷。
- **词表预算零和**：扩词表要同时做语料重配比（temperature sampling），否则新 token 训练不充分；中文与代码的压缩率是 Qwen 明确的设计目标。
- **压缩率、参数量、长尾训练充分度三角**：词表越大序列越短，但 embedding 按 $|V|d$ 线性长，新增长尾 token 训练不足。三者不可兼得，只能按服务形态选点。
- **数字的三条路**：① 预切分逐位（Qwen 现方案），数位对齐、token 数翻倍；② 保留分块但固定位宽并从右对齐（千分位分组），让同一 token 永远处在同一数位，代价是要改预切分与对照表且对不定长数字不自然；③ 不动 tokenizer，改数据与训练目标（数字对齐任务、统一写法、让模型在推理期展开成一位一位各算）。没有免费解。

## 数值与代码验证

**词表组成（本机实测，脚本读本地 `tokenizer.json`）**

| 口径 | merges | 词表字典 | added tokens | 实际词表 | config `vocab_size` | 差额 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B-Base | 151387 | 151643 | 22 | 151665 | 151936 | 271 |
| Qwen3-0.6B（Instruct） | 151387 | 151643 | 26 | 151669 | 151936 | 267 |

读法：$256 + 151387 = 151643$ 逐项成立；added token 的 id 紧接在 151642 之后连续排布（Base 占 151643–151664，Instruct 占 151643–151668），所以实际词表是二者之和。Instruct 多出的 4 个是 chat template 需要的 `<tool_response>`、`</tool_response>`、`<think>`、`</think>`。config 声明的 `vocab_size = 151936` 是对齐填充值：**它决定 embedding 与 logits 的矩阵行数**（这才是真正的成本口径），但**不是真实词表**，拿它算 token 数会错，拿它当「词表里有 151936 个可用 token」也会错。

**多语言 token 数（口径见上一节，源表在 [[llm-internals-06]] 第 3 节）**

| 语言 | 字符数 | UTF-8 字节 | gpt2 | cl100k | o200k | Qwen3-0.6B | cl100k 相对英文 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 英文 | 44 | 44 | 10 | 10 | 10 | 10 | 1.0x |
| 中文 | 15 | 45 | 34 | 28 | 16 | 10 | 2.8x |
| 印地语（天城文） | 43 | 113 | 70 | 48 | 19 | 44 | 4.8x |
| 泰米尔语 | 51 | 139 | 139 | 69 | 24 | 57 | 6.9x |
| 阿拉伯语 | 42 | 77 | 40 | 30 | 16 | 18 | 3.0x |
| 韩语 | 23 | 57 | 54 | 24 | 17 | 21 | 2.4x |

按本地 tokenizer 另测五句常见中文（句子与上表不同，只作量级复核）：17 字 → Qwen3 11 token、cl100k 23；16 字 → 10 / 22；11 字 → 5 / 12；两句 15 字 → 8 / 21 与 7 / 16。Qwen3 落在 0.45–0.65 token/字，cl100k 落在 1.07–1.4 token/字，方向与上表一致；上表那句 15 字是 0.67（Qwen3）/ 1.87（cl100k）token/字，略高于这批抽样的上沿，量级不变。

**数字切分对照（本机实测：tiktoken 0.12.0 + 本地 Qwen3 `tokenizer.json`）**

| 数字串 | gpt2 | cl100k_base | o200k_base | Qwen3-0.6B |
| --- | --- | --- | --- | --- |
| 1234 | `[12][34]` | `[123][4]` | `[123][4]` | `[1][2][3][4]` |
| 12345 | `[123][45]` | `[123][45]` | `[123][45]` | `[1][2][3][4][5]` |
| 1234567 | `[123][45][67]` | `[123][456][7]` | `[123][456][7]` | `[1]…[7]`（七段） |
| 1000 | `[1000]` | `[100][0]` | `[100][0]` | `[1][0][0][0]` |
| 1000000 | `[1][000000]` | `[100][000][0]` | `[100][000][0]` | `[1][0][0][0][0][0][0]` |
| 24177 | `[24][177]` | `[241][77]` | `[241][77]` | `[2][4][1][7][7]` |

`gpt2` 的位宽不定性用另一批随机数字复现（抽样计数随批次波动）：2000 个随机 4–10 位数字里，数字 token 的长度计数是 1 位 515、2 位 3655、3 位 1997、4 位 28、5 位 2，2–3 位占 91%，没有任何固定位宽；同一批数字在 `cl100k` 下则严格是 2 / 3 / 4 个 token（对应 4–6 / 7–9 / 10 位），即 $\lceil n/3 \rceil$。

**代价换算成显存与算力**（用仓库统一常数，口径写明：LLaMA-3 的层数与 KV 头数，不是 Qwen 自己的配置，只为给量级）

设同一内容在某语言下的 token 数是英文的 $\rho$ 倍，则 prefill 计算量（约 $2NT$，$N$ 为参数量、$T$ 为 token 数）与 KV cache 字节数 $2LH_{kv}d_{\text{head}}Sb$ 都按 $\rho$ 线性放大，attention 项按 $\rho^2$ 放大。LLaMA-3-8B 每 token KV cache $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = 128 KiB；LLaMA-3-70B 是 $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ B = 320 KiB。

| 同一句 15 个汉字 | token 数 | 8B 每序列 KV cache | 70B 每序列 KV cache |
| --- | --- | --- | --- |
| Qwen3-0.6B 口径 | 10 | $10 \times 128\ \text{KiB} = 1.25$ MiB | $10 \times 320\ \text{KiB} = 3.125$ MiB |
| cl100k 口径 | 28 | $28 \times 128\ \text{KiB} = 3.5$ MiB | $28 \times 320\ \text{KiB} = 8.75$ MiB |

上下文窗口装的是 token 不是字符，换算成「能装多少内容」差距同样真实：按同一个 Qwen3 tokenizer 的口径，8K 窗口下泰米尔语约 $8192 \times (51/57) \approx 7.3$k 字符、英文约 $8192 \times (44/10) \approx 36$k 字符，前者只有后者的约 20%；换成跨 tokenizer 的 $\rho$ 口径，`cl100k` 下泰米尔语 $\rho = 6.9$，同一段内容要 6.9 倍 token，同一个 8K 预算装下的内容只剩英文的 $1/6.9 \approx 14.5\%$，不足六分之一。两种口径都成立但不通用，引用时必须写明用的是哪一个。

**可运行脚本**（只用 `tiktoken` 与 `tokenizers` 读本地文件，不依赖 `transformers`——本机 `transformers` import 会报错）

```python
import glob, json
import tiktoken
from tokenizers import Tokenizer

import os
snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/*/"))[0]
tok_json = json.load(open(snap + "tokenizer.json"))
print("model.type =", tok_json["model"]["type"])

# 1) 词表组成：|V| = 256 + merges + added
n_merges = len(tok_json["model"]["merges"])
vocab = len(tok_json["model"]["vocab"])
added = len(tok_json["added_tokens"])
cfg = json.load(open(snap + "config.json"))
print(f"merges={n_merges} base={256 + n_merges} vocab_dict={vocab} added={added} "
      f"actual={vocab + added} config={cfg['vocab_size']} gap={cfg['vocab_size'] - vocab - added}")

# 2) 参数账：embedding 占比与 logits FLOPs
V, d, L = cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"]
emb = V * d
per_layer = cfg["num_attention_heads"] * cfg["head_dim"] * d * 2 \
            + 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * d \
            + 3 * d * cfg["intermediate_size"]
total = emb + L * per_layer + 2 * d * (L + 1)      # 含 final norm，tie_word_embeddings=True
print(f"embedding={emb/1e6:.1f}M total={total/1e6:.1f}M share={emb/total:.1%} "
      f"logits_flops_per_token={2*V*d:.3e}")

# 3) 数字切分：Qwen3 逐位 vs 三种 tiktoken 词表
qwen = Tokenizer.from_file(snap + "tokenizer.json")
def pieces(enc, s):
    ids = enc.encode(s)
    return [enc.decode_single_token_bytes(i).decode("utf-8", "replace") for i in ids]
for s in ["1234", "1234567", "1000", "1000000", "24177"]:
    row = {n: "|".join(pieces(tiktoken.get_encoding(n), s)) for n in ("gpt2", "cl100k_base", "o200k_base")}
    q = "|".join(qwen.decode([i]) for i in qwen.encode(s).ids)
    print(f"{s:>8}  gpt2={row['gpt2']:<10} cl100k={row['cl100k_base']:<10} "
          f"o200k={row['o200k_base']:<10} qwen3={q}")
```

## 常见追问

- **追问**：151K 词表带来的 embedding 参数增长，在 Qwen3-0.6B 这种小模型上到底占多少？
  - 要点：$151936 \times 1024 = 155.6\text{M}$，占总参数约 596M 的 26.1%；若输入输出 embedding 不共享，总参数约 752M（多出一整份 155.6M）。词表从 32K 到 151K 是 4.64×，而参数量只涨这一项，所以小模型上占比才这么扎眼；大模型上分母变大，占比自然回落，但 logits 的每 token 成本不回落。
- **追问**：为什么 decode 阶段的输出层不能被 MoE 稀疏化，这件事对大词表意味着什么？
  - 要点：MoE 稀疏的是 FFN 专家，attention 与最后的满词表投影对所有 token 都要走一遍。每 token 约 $2|V|d$ FLOPs 是固定税，词表越大越贵——这就是大词表对小激活参数模型不划算的原因。
- **追问**：数字逐位切分后 token 变多了，是不是违背了「压缩率越高越好」？
  - 要点：是主动牺牲压缩率换归纳偏置。位数字是高频 token，训练充分，钱包层面代价主要是序列长度（prefill、KV cache、上下文占用）而不是长尾欠训练；换来的是同一个 token 永远对应同一量级，模型可能学到同位对齐。数字密集型场景值，纯自然语言场景不值。
- **追问**：除了逐位切分，还有什么办法解决数字对齐？
  - 要点：三条路都要说清代价。① 预切分逐位（Qwen 现方案），数位对齐但 token 数涨到 3 位分组口径的 2–3 倍（4 位 2.0×、6 位 3.0×）；② 保留分块但固定位宽、从右对齐（千分位分组），同一 token 永远在同一数位；③ 不动 tokenizer，改数据与训练目标（数字对齐任务、统一写法）。没有免费解。
- **追问**：`config.json` 的 `vocab_size = 151936` 能直接当词表大小用吗？
  - 要点：不能当「真实词表」，但必须当「成本口径」。它是矩阵行数与对齐填充值，Instruct 实际词表 151669（$256 + 151387 + 26$），差 267 行；Base 是 151665，差 271 行。算 embedding 与 logits 用 config，算 token 数与词表覆盖用 `tokenizer.json`。
- **追问**：换 tokenizer（扩词表）后直接续训，会有什么后果？
  - 要点：新 token 的 embedding 是随机初始化的，长尾 token 语料出现次数少、梯度弱，不配足够步数与合理初始化就等于把输入分布打乱重建。工程上要么冻结旧 embedding 只训新行、要么给新行按旧 token 均值初始化，再跑足够量的继续预训练。

## 相关题目

- [[llm-internals-06]]：分支与失效场景的母题，本题的数字表、多语言表与「byte-level 永不 OOV 的代价」口径都来自那一篇的第 3–4 节。
- [[llm-internals-16]]：decoder-only transformer 的一次前向传播。token → embedding → logits 的链路正是 tokenizer 与模型参数的交界，$|V|$ 同时决定这两处的参数量。
- [[llm-internals-02]]：KV cache 的显存公式。token 数 $\rho$ 倍放大直接落在这条公式上，是多语言与数字切分成本换算的共同出口。

## 参考资料与归属

- **Byte Pair Encoding** —— Amit Shekhar（Outcome School）：<https://outcomeschool.com/blog/bpe-in-llms>。BPE 训练/编码流程与「编码重放 merge 规则」的表述、32K–256K 的词表量级来自这篇。
- **大型语言模型（LLM）中的 Tokenization** —— 视频讲解：<https://www.youtube.com/watch?v=sK2s9I84EVI>。tokenizer 在训练与推理链路中的位置、词表规模与多语言的直觉说明来自这条。
- **Neural Machine Translation of Rare Words with Subword Units（延伸）** —— Sennrich, Haddow, Birch，2015-08-31：<https://arxiv.org/abs/1508.07909>。subword 折中与「开放式词表」动机的原始依据。
- **SentencePiece: A simple and language independent subword tokenizer（延伸）** —— Kudo, Richardson，2018-08-17：<https://arxiv.org/abs/1808.06226>。语言无关的分词框架与词表预算在多语言上的意义由此延伸而来。
- 词表组成、参数账、数字切分表与多语言续表为本机复算：`tokenizers` 直读 `Qwen/Qwen3-0.6B-Base` 的 `tokenizer.json`（Instruct 的 added token 数由公开 `tokenizer_config.json` 的 `added_tokens_decoder` 核对），`tiktoken` 0.12.0 提供 `gpt2` / `cl100k_base` / `o200k_base`；多语言原表口径见 [[llm-internals-06]] 第 3 节。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
