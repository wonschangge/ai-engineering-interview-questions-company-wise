---
type: question
id: coding-04
topic: 编程与数据结构
order: 4
question: 从零实现 BPE 的训练与编码。
question_en: Implement BPE training and encoding from scratch.
asked_at: []
level: 进阶
tags: [bpe, tokenizer, 实现题, unicode]
sources:
  - title: Byte Pair Encoding
    url: https://outcomeschool.com/blog/bpe-in-llms
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Neural Machine Translation of Rare Words with Subword Units（延伸）
    url: https://arxiv.org/abs/1508.07909
    author: Sennrich et al. (ACL 2016)
    published: 2015-08-31
  - title: SentencePiece: A simple and language independent subword tokenizer and detokenizer（延伸）
    url: https://arxiv.org/abs/1808.06226
    author: Kudo & Richardson (EMNLP 2018 demo)
    published: 2018-08-19
related: [llm-internals-06, coding-10, llm-internals-07, coding-05]
updated: 2026-09-28
---

## 一句话答案
> BPE 把 tokenization 当成贪心压缩：语料先预切分成词，每个词展开成符号序列（字符或 UTF-8 字节），每轮按**词频加权**统计相邻符号对的频次，把最高频的一对合成新符号并追加进 merge 表，直到用完预设的 merge 数。产物是一张**有序** merge 表加词表：$|V| = |V_{\text{special}}| + |V_{\text{base}}| + N_{\text{merges}}$。
> 编码只做一件事：按 merge 的训练顺序（rank）重放，绝不重算频次、也不用最长匹配——换一种合并顺序切分就变了，而 embedding 是按训练时的切分学出来的。底层符号表必须给表外字符留退路：字符口径要 byte fallback，否则一个制表符或 emoji 就掉进 `<unk>`、往返立刻破；字节口径用 256 个字节当底表，代价是非 ASCII 字符天生占 2–4 个字节。
## 面试官在考什么
- 训练循环写得出来吗：相邻对频次**按词频加权**而非按出现次数；并列怎么打破；什么时候停（用完 merge 预算，或没有可合并的对）。
- 编码阶段知不知道是 rank 重放，能不能说清为什么不能按频次重新合并、也不能在词表里做最长匹配。
- 复杂度与优化：朴素每轮重扫语料的 $O(kN)$ 对比增量维护 pair 计数加堆的近似 $O(N\log N)$，并给出实测加速比。
- 口径与边界：字符口径 vs 字节口径、词尾标记的作用、前缀空格的表示法、特殊 token 与 unk、encode/decode 往返是否无损；再往上就是词表大小的三方权衡与「换 tokenizer 为什么要重训」。

**常见错误答案**
- 「编码时在词表里找最长匹配子词」或「按频次从高到低把能合的合掉」。前者会切出训练中从未出现过的组合（本次实测 held-out 里 81.7% 的词与 rank 重放不同）；后者在单条文本内没有语料级频次可统计，退化成逐字符（实测 95.0% 的词不同）。
- 「字符口径永不 OOV」。只有字节口径（底表 256 字节）有这个保证；字符口径的底表是语料字符集，没见过的字符只能 `<unk>`，往返立刻破——一个制表符就够。
## 原理与推导
### subword 要解决什么
词级词表要覆盖所有词形：常见词形几十万，再加专名、术语、拼写变体与每天新增的词，没有上界，未登录词只能变 UNK，信息与梯度一起丢。字符级没有 OOV，代价是序列膨胀：设平均词长 $\bar{k}$（英文含空格 5–6 字符），同一段文本的 token 数是词级的 $\bar{k}$ 倍；prefill 计算量 $\approx 2NT$ 随 token 数线性涨，attention 项按平方涨。于是 $T_{\text{char}} \gg T_{\text{subword}} \gtrsim T_{\text{word}}$，而 $|V|_{\text{subword}} \ll |V|_{\text{word}}$。

Sennrich 等（2015）的出发点正是这点：翻译是 open-vocabulary 问题，把稀有与未登录词编码成子词序列即可——人名靠字符复制或音译、复合词靠组合式翻译、同源词与借词靠音系与形态变换；该文报告在 WMT15 英德、英俄任务上比词典回退基线分别高 1.1 与 1.3 BLEU（论文口径）。
### 训练五步
1. **预切分**：按空白、标点、数字切词（本实现用 `\s+|\d+|[^\W\d]+|[^\w\s]`，把空白附着到后一个词，等价 GPT-2 的 Ġ 约定）；正则必须覆盖全部字符，漏掉下划线这类字符会让预切分静默丢字符、往返立刻破。中文这类不写空格的语言靠把 CJK 归入 `[^\W\d]` 成串切；SentencePiece 干脆跳过预切分、直接在原始句子上训练子词模型，从而做到语言无关（Kudo & Richardson, 2018）。
2. **词 → 符号序列 + 词尾标记**：`low` → `l o w </w>`。词尾标记区分词中与词尾的同一字符——`est` 与 `est</w>` 是两种符号，模型才能学到「这里词结束了」。
3. **统计相邻对的加权频次**：$c(a,b) = \sum_{w} f(w)\cdot n_{a,b}(w)$，其中 $n_{a,b}(w)$ 是 $a,b$ 在词 $w$ 的当前切分中相邻出现的次数。
4. **合并最高频的一对** $(a^*,b^*) = \arg\max_{a,b} c(a,b)$（并列取字典序，保证可复现），追加进 merge 表，并更新所有含这一对的词。
5. **停止**：用完 `num_merges`，或没有可合并的对。字符口径的 $|V|_{\text{base}}$ 是语料字符集（可再挂 256 个字节回退 token），字节口径固定 256。
### 复杂度：$O(kN)$ 与近似 $O(N\log N)$
朴素实现每轮重扫全语料重新统计相邻对，$k$ 轮共 $O(kN)$——下面 `train_bpe_naive` 十几行就能写出来，它同时是增量版的对拍基准。工程实现只维护「受到影响的词」：全局 `counts[pair]` 加 `pair -> 含该对的词集合`，每轮从最大堆取当前最高频对（堆顶计数与当前计数不符就丢弃），合并后只重扫这些词，减去旧 pair 计数、加上新 pair 计数、把变过的 pair 重新入堆，实测接近 $O(N\log N)$。

一个必须自己踩过才会信的坑：**pair 计数下降时也要重新入堆**。只在上升时入堆，某个 pair 的旧堆项会因为计数不符被丢弃，而它的计数之后再没变过，就永远从堆里消失，堆顶于是给出一个偏小的「最大值」。这种 bug 只在多轮之后暴露（本次实测全量语料上第 27 轮才与朴素基准分叉），所以增量实现必须留一个朴素基准做对拍。
### 编码：按 rank 重放
编码不做任何统计，只按 rank 升序贪心合并。三个理由：其一，编码时只有一条文本，语料级 $c(a,b)$ 无从计算；其二，就算在当前文档上重算，合并顺序会随内容漂移，切出训练中不存在的组合；其三，rank 重放是确定性的（同一文本永远同一 token 序列），可以做得很快。下面的 `encode_word` 是最直观的 $O(n^2)$ 版本；同一份实现里还有一份双向链表 + 以 rank 为键的最小堆、带惰性失效的 $O(n\log n)$ 版本（`encode_word_fast`），两者在语料内全部 80,270 个词上输出逐 token 相同，实测差距见第 4 节第 ⑦ 条。
### 字符口径 vs 字节口径
字节口径先把词转成 UTF-8 字节，在 256 个字节上跑 BPE：任意 Unicode 都能编码，最坏退化成逐字节 token，OOV 这个概念消失。代价是 Unicode 的字节宽度直接变成 token 成本：汉字、天城文、泰米尔文字母 3 字节，常见 emoji 4 字节，只有高频字节串才会被合并成单个 token。字符口径底表小、同语言压缩率更高，但必须处理底表外字符：加 byte fallback（SentencePiece 的 `byte_fallback`，把没见过的字符退成 `<0xNN>` 字节 token），或者承认有损。

空格与词边界的表示法各家不同：GPT 系用 Ġ 表示被并进词里的前导空格，SentencePiece 用 ▁（U+2581），字符口径常用词尾标记 `</w>`；本实现把字节 token 命名成十六进制（`tok.hex()`），只是换一套可打印的名字。这件事决定 detokenize 能否无损——把空格或词尾标记丢掉是最常见的往返 bug。特殊 token（`<pad>`/`<unk>`/`<s>`/`</s>`）固定占 id 0–3、不进 merge 统计，输入侧的 special 切分要单独做（本实现只保证占位与 `<unk>` 兜底）。

规范化与大小写必须在训练与推理两侧完全一致。NFC/NFKC 会把基字符与组合音标合成单码位（第 4 节表 ④ 的 NFD 串 23 字符 / 27 字节，规范化后就变成另一条 token 序列），NFKC 还做兼容分解（全角转半角、`①` 变 `1`）；大小写处理（lowercase 或 casefold）能压英文词表，代价是丢掉大小写信息，对中文这类没有大小写的语言则无关。两侧不一致就是训练/服务偏斜：同一个词在线上切出与训练不同的片段，embedding 直接对不上——这类 bug 不会报错，只会让指标悄悄变差。
## 数值与代码验证
**语料口径**：仓库根 `README.md`、`README.zh-CN.md` 与 `docs/**/*.md`（不含 `docs/编程与数据结构/`）共 118 个文件、按文件名顺序用空行拼接成的快照，1,831,356 字符 / 3,159,391 UTF-8 字节；每 20 行留一行做 held-out，得到 train 1,739,215 字符（80,270 个不同词、573,896 次词出现、2,313,111 个符号）与 test 92,140 字符（30,503 个词、7,470 个不同词）。下面所有数字都是本机单核实测（AMD Ryzen 9 8945HX，CPython 3.10.12；纯 Python 计时重复运行有 10%–15% 的波动），被测实现就是下面这段代码（完整脚本在 `.work/coding04_audit/`）。
```python
import heapq
import re
from collections import Counter, defaultdict

END_OF_WORD = "</w>"
PAT = re.compile(r"\s+|\d+|[^\W\d]+|[^\w\s]", re.UNICODE)   # 空白|数字|字母与下划线(含 CJK)|其它符号
_SPECIALS = ("<pad>", "<unk>", "<s>", "</s>")

def split_words(text):
    """预切分：空白附着到后一个词（GPT-2 的 Ġ 约定）；只依赖字符类别，不依赖词表。"""
    words, ws = [], ""
    for piece in PAT.findall(text):
        if piece.isspace():
            ws += piece
        else:
            words.append(ws + piece)
            ws = ""
    return words + ([ws] if ws else [])

def char_symbols(word):
    return tuple(word) + (END_OF_WORD,)                  # 词尾标记：区分词中与词尾的同一字符

def byte_symbols(word):
    return tuple(bytes([b]) for b in word.encode("utf-8"))

def word_freqs(texts, to_syms):
    """语料 -> {符号元组: 词频}；to_syms 决定字符口径（词尾补标记）还是字节口径。"""
    freq = Counter()
    for text in texts:
        for word in split_words(text):
            freq[to_syms(word)] += 1
    return freq

def merge_all(syms, pair):
    """一趟扫完序列里所有不重叠的 pair（最左边的先吃），是所有合并动作的唯一原语。"""
    a, b = pair
    out, i = [], 0
    while i < len(syms):
        if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out

def train_bpe_naive(freqs, num_merges):
    """朴素基准：每轮重扫全部词统计相邻对，O(kN)；复杂度与正确性的对拍实现。"""
    seqs, merges = {w: list(w) for w in freqs}, []
    for _ in range(num_merges):
        counts = Counter()
        for w, s in seqs.items():
            for pair in zip(s, s[1:]):
                counts[pair] += freqs[w]                 # 按词频加权
        if not counts:
            break
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]   # 频次最高，并列取字典序
        merges.append(best)
        for w in seqs:
            seqs[w] = merge_all(seqs[w], best)
    return merges

def train_bpe_fast(freqs, num_merges):
    """增量版：只重扫含该 pair 的词 + 堆取最大，实测接近 O(N log N)，结果与朴素基准逐条相同。"""
    words = list(freqs)
    wf = [freqs[w] for w in words]
    seqs = [list(w) for w in words]
    local = [Counter(zip(s, s[1:])) for s in seqs]      # 每个词内部的 pair 计数（不带词频）
    counts, pair_words = Counter(), defaultdict(set)
    for wid, lc in enumerate(local):
        for pair, n in lc.items():
            counts[pair] += n * wf[wid]
            pair_words[pair].add(wid)
    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)
    merges = []
    for _ in range(num_merges):
        best = None
        while heap:                                     # 惰性失效：堆顶计数与当前不符就丢弃
            neg, pair = heapq.heappop(heap)
            if -neg == counts.get(pair, 0) and -neg > 0:
                best = pair
                break
        if best is None:
            break
        merges.append(best)
        for wid in list(pair_words.pop(best, ())):       # 只碰真正含该 pair 的词
            if local[wid].get(best, 0) == 0:
                continue
            f, old = wf[wid], local[wid]
            seqs[wid] = merge_all(seqs[wid], best)
            new = Counter(zip(seqs[wid], seqs[wid][1:]))
            local[wid] = new
            for pair in set(old) | set(new):             # 增减一起算，只要变过就重新入堆
                delta = (new.get(pair, 0) - old.get(pair, 0)) * f
                if delta:
                    counts[pair] += delta
                    if counts[pair] > 0:
                        pair_words[pair].add(wid)
                        heapq.heappush(heap, (-counts[pair], pair))
    return merges

def encode_word(syms, rank, pairs):
    """rank 重放：每轮取 rank 最小的可用 pair 全部合并，这是 O(n^2) 的直观版。"""
    syms = list(syms)
    while len(syms) > 1:
        best = min((rank[p] for p in zip(syms, syms[1:]) if p in rank), default=-1)
        if best < 0:
            break
        syms = merge_all(syms, pairs[best])
    return syms

class BPETokenizer:
    """fit 产出有序 merge 表与词表；encode 只做 rank 重放；decode 保证往返无损。"""

    def __init__(self, mode="char", byte_fallback=True):
        assert mode in ("char", "byte")
        self.mode, self.byte_fallback = mode, byte_fallback
        self.merges, self.rank, self.vocab, self.token_to_id = [], {}, [], {}

    def fit(self, texts, num_merges, trainer=train_bpe_fast):
        self.freqs = word_freqs(texts, char_symbols if self.mode == "char" else byte_symbols)
        self.merges = trainer(self.freqs, num_merges)
        self.rank = {p: i for i, p in enumerate(self.merges)}   # pair -> 合并序号
        self.pairs = list(self.merges)                          # 合并序号 -> pair（简单编码器用）
        self.base = ([bytes([b]) for b in range(256)] if self.mode == "byte"
                     else sorted(set().union(*self.freqs)))     # byte 底表 = 256 字节
        if self.mode == "char" and self.byte_fallback:          # char 底表外字符退字节
            self.base += [f"<0x{b:02X}>" for b in range(256)]
        # |V| = special + 底表 + merge；合并出的符号按 rank 追加，id 就是插入序
        self.vocab = list(_SPECIALS) + self.base + [a + b for a, b in self.merges]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        return self

    def _name(self, tok):
        return tok if self.mode == "char" else tok.hex()     # 字节 token 用十六进制命名

    def encode(self, text):
        """任意文本 -> token 列表；结果只取决于 merge 表的 rank，与文本自身的统计无关。"""
        out = []
        for word in split_words(text):
            if self.mode == "byte":
                syms = byte_symbols(word)
            else:
                syms = char_symbols(word)
                if not all(c in self.token_to_id for c in syms):      # 底表外字符：退字节或退 unk
                    out += ([f"<0x{b:02X}>" for b in word.encode("utf-8")] if self.byte_fallback
                            else ["<unk>"])
                    continue
            out += [self._name(t) for t in encode_word(syms, self.rank, self.pairs)]
        return out

    def encode_ids(self, text):
        return [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in self.encode(text)]

    def decode(self, tokens):
        """token 列表 -> 原文：字节口径按 UTF-8 还原，字符口径剥掉词尾标记。"""
        if self.mode == "byte":
            return bytes.fromhex("".join(tokens)).decode("utf-8", errors="replace")
        text, buf = [], bytearray()
        for tok in tokens:
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                buf.append(int(tok[3:5], 16))                     # byte fallback 的字节 token
            else:
                text.append(buf.decode("utf-8", errors="replace"))    # 先冲掉攒下的字节
                buf.clear()
                text.append(tok)
        text.append(buf.decode("utf-8", errors="replace"))
        return "".join(text).replace(END_OF_WORD, "")
```
**① 经典例子**（`low`×5、`lower`×2、`newest`×6、`widest`×3，词尾标记 `_`，10 轮）：merge 表为 `es, est, est_, lo, low, ew, ewest_, newest_, low_, dest_`；首轮 `es`/`st`/`t_` 并列 9 次（`we` 8、`lo` 7），按「频次优先、并列取字典序」先合并 `e+s`，第 3–5 轮落成 `est_`、`lo`、`low`，与 [[llm-internals-06]] 的手算一致；把标记换成 `</w>` 只是换符号名（逐条相等）。合并后 `low_ → [low_]`、`lower_ → [low, e, r, _]`、`newest_ → [newest_]`、`widest_ → [w, i, dest_]`；对训练中没出现过的 `lowest`，编码得到 `[low, est]`——`low`（rank 5）与 `est`（rank 2）都已学会。

**② 朴素基准 vs 增量版**（两者 merge 表逐条相同；1/32 词表 = 2,508 词 / 813,591 符号，1/8 词表 = 10,033 词 / 1,487,144 符号）：1/32 子集 200 轮 0.99 s → 0.07 s（15.0 倍）、1000 轮 3.87 s → 0.10 s（39.9 倍）；1/8 子集 200 轮 5.38 s → 0.17 s（31.3 倍）、1000 轮 27.63 s → 0.32 s（87.3 倍）。朴素版耗时与 merge 数近似线性（1/8 子集 200 → 1000 轮：5.38 s → 27.63 s，5.1 倍），增量版只有 1.9 倍。全量语料上朴素版 200 轮 68.18 s（2.93 merge/s，按线性外推 2000 轮约 11.4 分钟），增量版 char 口径 2000 轮 5.13 s（390 merge/s）、byte 口径 2000 轮 33.12 s（60 merge/s）——两个数量级。

**③ 编码器与合并顺序**：语料内全部 80,270 个不同词上，直观版与双向链表 + 堆的版本输出逐 token 相同（0 个不一致），每个 token 都落在词表内（0 个越界），逐词 decode 全部还原（0 个失败）。顺序敏感性在 held-out 的 7,470 个不同词上量化——按当前词频重算再合并，7,096 个词（95.0%）与 rank 重放不同；词表最长匹配，6,104 个词（81.7%）不同。同一个词 `Adapter`：rank 重放得 `[A, d, ap, ter</w>]`，按频次重算得 8 个单字符（单条文本内没有重复对可统计），最长匹配得 `[A, d, ap, ter]`（丢掉词尾标记，切出训练中不存在的组合）。

**④ 往返一致**（char 口径 $|V| = 4 + 1944 + 256 + 2000 = 4204$：1944 个语料字符、256 个字节回退 token、2000 条 merge；byte 口径 $|V| = 4 + 256 + 2000 = 2260$。无回退列指 `byte_fallback=False`，此时整词退成一个 `<unk>`）：
| 用例 | 字符 | UTF-8 字节 | char token | 往返 | 无回退 token | 往返 | byte token | 往返 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 纯英文 | 33 | 33 | 14 | ✓ | 14 | ✓ | 14 | ✓ |
| 多空格与制表符 | 27 | 27 | 17 | ✓ | 14 | ✗ | 15 | ✓ |
| 标点与代码（含换行） | 33 | 37 | 16 | ✓ | 16 | ✓ | 16 | ✓ |
| 中文 | 18 | 46 | 15 | ✓ | 15 | ✓ | 16 | ✓ |
| emoji 与 ZWJ 家庭 | 21 | 48 | 44 | ✓ | 17 | ✗ | 39 | ✓ |
| 组合字符（NFD） | 23 | 27 | 21 | ✓ | 17 | ✗ | 19 | ✓ |
| 泰米尔文 / 天城文 / 希腊文 | 25 | 61 | 55 | ✓ | 13 | ✗ | 56 | ✓ |
| 数字与千分位 | 26 | 40 | 21 | ✓ | 21 | ✓ | 21 | ✓ |

char（带 byte fallback）与 byte 两种口径在 8 个用例上 encode → decode 都精确还原，整段 held-out 的 92,140 字符也精确还原（逐词 decode 失败 0 / 30,503）。去掉 byte fallback 后，制表符、emoji、组合字符与非拉丁文字全部丢信息——`<unk>` 不是兜底，是数据丢失。

**⑤ 压缩率 vs 词表大小**（held-out 92,140 字符 / 30,503 词，平均词长 3.02 字符）：
| merges | char \|V\| | char token/词 | char 字符/token | byte \|V\| | byte token/词 | byte/char token 比 |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 2704 | 2.385 | 1.267 | 760 | 2.826 | 1.19 |
| 1000 | 3204 | 2.154 | 1.403 | — | — | — |
| 2000 | 4204 | 1.931 | 1.564 | 2260 | 1.991 | 1.03 |
| 4000 | 6204 | 1.735 | 1.741 | — | — | — |

两个结论：收益递减——merge 数翻 8 倍只换来 1.37 倍压缩，而 embedding 参数按 $|V|\cdot d$ 线性涨、输出层 softmax 的乘加也按 $|V|$ 涨，长尾 token 还训练不充分（公开模型取 32k–256k 就是在这条曲线的拐点附近取点）；字节口径用更小的词表达到接近的压缩率（$|V| = 2260$ 时 1.991 token/词，仅比 $|V| = 4204$ 的字符口径多 3%），因为它只花 256 个位置做底表，其余预算全给合并出的多字节序列，代价是 token 序列里可能出现半个字符的字节片段。

**⑥ 跨语言成本**（同一句话的中英文版本，对照组是本地 `tiktoken 0.12.0` 的 `cl100k_base`）：
| 句子 | 字符 | UTF-8 字节 | 我的 char token | 我的 byte token | cl100k |
| --- | --- | --- | --- | --- | --- |
| 中文（30 字） | 30 | 80 | 22 | 20 | 34 |
| 英文（95 字符） | 95 | 95 | 41 | 39 | 17 |
| 中文/英文 | — | — | 0.54x | 0.51x | 2.00x |

我这份 tokenizer 在中文上反而更省，因为训练语料是中文为主的仓库文档；cl100k 是英文中心词表，同一内容的中文版要 2.00 倍 token。**压缩率是语料配比的产物，不是算法的产物**（多语言膨胀的机理见 [[llm-internals-06]]）。成本换算：设同一内容某语言的 token 数是英文的 $\rho$ 倍，prefill 计算量 $\approx 2NT$ 与 KV cache 字节数都按 $\rho$ 放大（$N$ 为参数量、$T$ 为 token 数）；LLaMA-3-70B 每 token KV cache 320 KiB，H100 SXM5 的 3.35 TB/s HBM（bf16 dense 989 TFLOPs，roofline 295 FLOPs/byte）意味着每 token 的 KV 读取下限 320 KiB / 3.35 TB/s ≈ 97.8 µs，decode 是带宽瓶颈——$\rho = 2$ 就是每个请求多付一倍。

**⑦ 编码吞吐、优化与固定前缀缓存**：纯 Python 单核在 held-out 上 92,140 字符 / 30,503 词 → 58,762 token；预切分 0.008 s（12.27 M 字符/s），预切分加 rank 重放 0.102 s（0.90 M 字符/s、0.58 M token/s）。把那个直观版 rank 重放换成双向链表 + 堆：char 口径 0.94 → 1.17 M 字符/s（1.2 倍），byte 口径 0.36 → 0.78 M 字符/s（2.2 倍）——字节口径的符号序列长 3 倍，平方扫描的代价被放大，长序列上必须换数据结构。固定前缀场景（system prompt 20,000 字符 → 11,610 token，每请求再带 80 字符增量）：500 次请求每次重编前缀 28.29 ms，缓存前缀的 token id 后只编增量 0.12 ms（240 倍）——编码是高频操作，缓存与批量比算法常数重要得多（批量与 CPU 侧优化见 [[inference-serving-05]]）。
## 常见追问
- **追问**：BPE 与 WordPiece、Unigram 的区别？
  - 要点：差别在选择标准。BPE 从字符/字节起步、每轮合并频次最高的相邻对，只有 merge 表，不是概率模型；WordPiece 也从字符起步，但选合并后语料似然提升最大的对，续接子词加 `##`；Unigram 反过来——从大候选集起步逐轮剪枝，删掉后语料似然损失最小的子词，是真正的概率模型，还能按概率采样切分做 subword regularization。SentencePiece 是训练与切分的实现框架，BPE 与 Unigram 都能配。
- **追问**：为什么词表大小通常是 32k 到 128k？
  - 要点：三方权衡。词表越大 token 数越少（prefill 与 KV cache 省），但 embedding 与输出层参数按 $|V|\cdot d$ 涨、softmax 乘加按 $|V|$ 涨，长尾 token 还训练不充分。本次实测 8 倍 merge 只换 1.37 倍压缩，说明拐点来得很快。
- **追问**：为什么换了 tokenizer 一般要重训模型？
  - 要点：tokenizer 与权重绑定。embedding 行按 token id 索引，tokenizer 一换，id → 子词串的映射就变了，原 embedding 行对应的语义单元全部错位；输入分布（切分方式、序列长度）也变了。续训至少要重初始化新增 token 的 embedding、对齐新旧词表里语义相同的 token，并跑够步数让新 token 收敛。
- **追问**：线上要编码几万 QPS 的文本，你会怎么优化？
  - 要点：先把 tokenizer 从 Python 换到 Rust/C++（HF `tokenizers` 就是这条路；本次实测同一份 1.8 M 字符语料，byte 口径 2000 条 merge 训练要 33.1 s、char 口径 5.1 s，纯 Python 训练撑不起大词表）；再把编码换成链表 + 堆的 $O(n\log n)$ 版本（byte 口径实测 2.2 倍）；再做固定前缀与高频词的 id 缓存（实测 240 倍）、按请求批量化、把编码放进 CPU 线程池而不是主推理进程，最后用长度分桶压 padding（见 [[inference-serving-05]]）。
## 相关题目
- [[llm-internals-06]]：BPE 的失效场景（数字、代码、非拉丁文字）。本篇讲怎么实现出正确的切分，那篇讲这套切分在哪些输入上代价高昂。
- [[coding-10]]：带重叠且不切开语义单元的文本分块器。分块边界本质上依赖 token 边界，编码器的往返一致性是它的前提。
- [[llm-internals-07]]：位置编码。token 切分决定序列长度、位置编码决定模型如何使用位置，论长上下文时两者必须一起看。
- [[coding-05]]：top-k、top-p 与 temperature 采样。采样在 logits 上做，logits 的维度就是 $|V|$，词表大小同时决定输出层计算量与采样开销。
## 参考资料与归属
- **Byte Pair Encoding** —— Amit Shekhar（Outcome School），源站标注 2026-03-31：<https://outcomeschool.com/blog/bpe-in-llms>。训练五步、`low`/`lower`/`newest`/`widest` 的例子与「编码重放有序 merge 规则、不用最长匹配」的表述来自这篇，32k–256k 的词表量级也出自这篇。
- **Neural Machine Translation of Rare Words with Subword Units（延伸）** —— Sennrich, Haddow, Birch，2015-08-31（ACL 2016）：<https://arxiv.org/abs/1508.07909>。第 3 节的 open-vocabulary 动机、人名/复合词/同源词的子词可译性，以及 WMT15 英德与英俄上高 1.1 / 1.3 BLEU 的数字来自这篇（原文口径）。
- **SentencePiece（延伸）** —— Kudo & Richardson，2018-08-19（EMNLP 2018 demo）：<https://arxiv.org/abs/1808.06226>。第 3 节「直接在原始句子上训练、语言无关」与 byte fallback 的依据；论文报告英日 NMT 上直接从原始句子训练可达到与预分词相当的精度。
- 本文的代码、数字与表格全部由本机运行的实现产出（`.work/bpe_doc5.py` 与 `.work/coding04_audit/` 下的实测脚本），不是源文数字；`cl100k_base` 的 token 数由本地 `tiktoken 0.12.0` 读出。本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
