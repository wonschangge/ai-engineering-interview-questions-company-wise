---
type: question
id: gdm-04
company: Google DeepMind 与 Google AI
topic: coding
order: 4
question: 解析 bigram：从字符串中提取两词短语，用于 NLP 特征工程。
question_en: Parse bigrams: extract two-word phrases from a string for NLP feature engineering.
asked_at: []
level: 进阶
tags: [bigram, 分词, 稀疏性, 规格澄清]
sources:
  - title: Foundations of Statistical Natural Language Processing（延伸）
    url: https://mitpress.mit.edu/9780262133609/foundations-of-statistical-natural-language-processing/
    author: Manning & Schütze
    published: 1999-06-28
  - title: Speech and Language Processing（延伸）
    url: https://web.stanford.edu/~jurafsky/slp3/
    author: Jurafsky & Martin
    published: 2024-01-01
  - title: Unigram language model tokenization（延伸）
    url: https://arxiv.org/abs/1804.10959
    author: Kudo
    published: 2018-04-30
related: [gdm-03, rag-02, evaluation-02, llm-internals-01]
updated: 2026-09-28
---

## 一句话答案

> **这道题的关键不是"会不会写循环"，而是"两词短语怎么定义"**——**重叠与否、跨句与否、标点如何处理，三种选择产出的特征数差 2 倍。**
> **★ 量化一：三种定义的产出差异（本机 151 字符的语料）**
> | 定义 | bigram 数 | 唯一 | 问题 |
> | --- | --- | --- | --- |
> | **重叠（$n-1$）** | **30** | 27 | **会跨句**（"dog. The" 成为特征） |
> | **不重叠（$n/2$）** | **15** | 15 | **丢一半**（"quick brown" 之后跳过 "fox jumps"） |
> | **按句子边界切分** | **27** | 24 | **更合理**（**不跨句，且保留重叠**） |
> **读法**：**同一段文本，三种定义给出 30 / 15 / 27 个特征**——**所以"提取 bigram"必须先把定义写进规格**。
> **★ 量化二：bigram 的空间是词表的平方**
> | 项 | 值 |
> | --- | --- |
> | 词表 $|V|$ | 50,000 |
> | **可能的 bigram 空间** | **$|V|^2 = 2.5\times10^9$** |
> | 本机语料观测到 | **27** |
> **读法**：**bigram 的特征空间是词表的平方（25 亿），而真实语料只覆盖极小一部分**——**所以"提取 bigram"必然面对稀疏**：**要么按最小频次剪枝，要么用哈希/嵌入**（**串 [[rag-02]]**）。
> **★ 量化三：分词规则的影响**
> | 规则 | token 数 | 是否保留 `isn't` |
> | --- | --- | --- |
> | **仅字母** | **32** | **否（被切成 `isn` + `t`）** |
> | **字母 + 撇号** | **31** | **是** |
> | 含数字 | 31 | 是 |
> **读法**：**"字母 + 撇号"保留 `isn't` 为一个 token，"仅字母"会切成两个**——**同一段文本的 token 数不同（32 vs 31）**，**bigram 的产出也随之不同**。**所以分词规则是规格的一部分**。
> **实现要点**：
> | # | 要点 |
> | --- | --- |
> | ① | **先规范化**（**大小写、Unicode NFC**） |
> | ② | **句子边界要单独处理**（**不要跨 `.` `!` `?` `;`**） |
> | ③ | **标点归属要定义**（**"dog," 里的逗号算不算**） |
> | ④ | **停用词是否过滤**（**过滤会破坏 bigram 的连续性**——**"the quick" 变 "quick"**） |
> | ⑤ | **输出要可复现**（**顺序稳定、去重规则明确**） |
> 一句话判据：**"先问清'两词短语'的定义（重叠/跨句/标点）→ 规范化 → 按句切分 → 说明稀疏性与剪枝策略"**。

## 面试官在考什么

- **★ 是否先问定义**：**能否指出"重叠/不重叠/跨句"会给出不同结果**（本机：**30/15/27**）。
- **★ 是否有稀疏意识**：**能否指出"bigram 空间是 $|V|^2$"**（本机：**25 亿 vs 观测 27**）。
- **分词细节**：**能否指出"`isn't` 会被切成两个 token"**（**撇号处理**）。
- **句子边界**：**能否指出"不要跨句"**。
- **标点处理**：**能否指出"标点归属要定义"**。
- **停用词**：**能否指出"过滤停用词会破坏 bigram"**。
- **规范化**：**能否指出大小写与 Unicode**。
- **可复现**：**能否指出输出顺序与去重规则**。
- **工程意识**：**能否指出"大数据下要流式处理"**（**不要一次载入**）。
- **诚实**：**承认"规格不清时先问，而不是猜"**。

**常见错误答案**

- **直接写 `zip(tokens, tokens[1:])`**（**不问定义**）。
- **跨句拼接**（**产生 "dog the" 这种噪声特征**）。
- **过滤停用词后再取 bigram**（**破坏了相邻关系**）。
- **不做大小写规范化**（**"The" 与 "the" 变成两个词**）。
- **不考虑标点**（**"dog," 与 "dog" 是两个 token**）。
- **不用生成器**（**大文本内存爆掉**）。
- **不定义去重与顺序**（**结果不可复现**）。
- **不提稀疏**（**不知道特征空间有多大**）。

## 原理与推导

### 1. ★ 三种定义

| 定义 | 产出 | 适用 |
| --- | --- | --- |
| **重叠（$n-1$）** | $n-1$ 个 | **语言模型**（**每个位置都要预测下一个**） |
| **不重叠（$n/2$）** | $\lfloor n/2\rfloor$ 个 | **短语挖掘**（**避免重叠冗余**） |
| **按句切分 + 重叠** | 约 $n-\text{句数}$ 个 | **特征工程**（**推荐默认**） |

**读法**：**"语言模型用重叠、短语挖掘用不重叠"**——**而"特征工程"通常要"按句切分 + 重叠"**（**本机 27 个**）。

### 2. ★ 稀疏性

$$\text{可能的 bigram 数}=|V|^2\qquad\text{实际观测}\ll|V|^2$$

| $|V|$ | $|V|^2$ |
| --- | --- |
| 10,000 | $10^8$ |
| **50,000** | **$2.5\times10^9$** |
| 200,000 | $4\times10^{10}$ |

**读法**：**"用 bigram 做特征"意味着"在平方级的空间里取极稀疏的样本"**——**三个对策**：

| 对策 | 说明 |
| --- | --- |
| **最小频次剪枝** | **只保留出现 ≥ m 次的 bigram** |
| **哈希技巧** | **把 bigram 哈希到固定维度**（**冲突可控**） |
| **嵌入** | **用词向量组合代替 one-hot**（**串 [[rag-02]]**） |

### 3. 分词规则

| 规则 | 影响 |
| --- | --- |
| **撇号** | **`isn't` 是一个还是两个 token** |
| **连字符** | **`state-of-the-art` 是一个还是四个** |
| **数字** | **`2024` 是否算词** |
| **大小写** | **是否 casefold** |
| **Unicode** | **NFC 规范化**（**`é` 的两种表示**） |

**读法**：**这些规则都会改变 token 序列**——**从而改变 bigram**——**所以要么写进规格，要么在文档里声明**。

### 4. 停用词的陷阱

| 做法 | 后果 |
| --- | --- |
| **先过滤停用词，再取 bigram** | **"the quick brown" → "quick brown"**（**相邻关系被改变**） |
| **先取 bigram，再按需过滤** | **保留原始相邻关系** |

**读法**：**"过滤顺序"是一个容易搞错的细节**——**正确做法是"先取 bigram，再决定要不要过滤"**。

### 5. 流式实现

```python
def bigrams(text, *, overlap=True, split_sentences=True, keep_punct=False):
    ...
```

| 要点 | 说明 |
| --- | --- |
| **生成器** | **`yield` 而不是返回列表**（**大文本友好**） |
| **窗口** | **只需保留上一个 token**（**$O(1)$ 内存**） |
| **句子边界** | **遇到边界就重置窗口** |
| **可复现** | **顺序 = 出现顺序**（**不排序**） |

**读法**：**"只需保留上一个 token"是流式 bigram 的核心**——**它让内存与文本长度无关**。

## 数值与代码验证

### 表 1：三种定义的产出、稀疏性、分词规则（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
import re
from collections import Counter
from typing import Dict, List, Tuple
TEXT = ("The quick brown fox jumps over the lazy dog. The dog barks, and the fox runs away! "
        "A quick brown dog jumps over a lazy fox; the fox isn't lazy anymore.")
print('① "两词短语"的三种定义，产出数量差 2 倍')
toks=re.findall(r"[A-Za-z']+", TEXT.lower())
print(f'  文本：{len(TEXT)} 字符、分词后 {len(toks)} 个 token')
overlap=[(toks[i],toks[i+1]) for i in range(len(toks)-1)]
nonover=[(toks[i],toks[i+1]) for i in range(0,len(toks)-1,2)]
def sent_bounded(text):
    out=[]
    for s in re.split(r'[.!?;]', text):
        t=re.findall(r"[A-Za-z']+", s.lower())
        out += [(t[i],t[i+1]) for i in range(len(t)-1)]
    return out
sb=sent_bounded(TEXT)
print("  定义                          bigram 数    唯一  说明")
for name,bg in (('重叠（n-1）',overlap),('不重叠（n/2）',nonover),('按句子边界切分',sb)):
    print(f'  {name:<26} {len(bg):>10} {len(set(bg)):>7} '
          f'{"**会跨句**" if name.startswith("重叠") else ("**会丢一半**" if "不" in name else "更合理")}')
print('  读法：**重叠版产出 30 个（会跨句），不重叠版 15 个（丢一半），按句切分 27 个** ——')
print('        所以"两词短语"必须定义：是否重叠、是否跨句、标点如何处理')
print()
print('② bigram 的空间爆炸：|V|^2 可能组合 vs 实际观测')
V=50_000
print(f'  词表 |V| = {V:,}')
print(f'  可能的 bigram 空间 = |V|^2 = {V*V:,}（{V*V/1e9:.1f}e9）')
observed=len(set(overlap))
print(f'  本机语料观测到 {observed} 个（**稀疏度 {(1-observed/(V*V))*100:.6f}%**）')
print('  读法：**bigram 的特征空间是词表的平方（25 亿），而真实语料只覆盖极小一部分** ——')
print('        所以"提取 bigram"必然面对**稀疏**：要么剪枝（最小频次），要么用哈希/嵌入')
print()
print('③ 分词规则的影响（同一段文本，不同规则）')
rules=[('仅字母',r"[A-Za-z]+"),('字母 + 撇号',r"[A-Za-z']+"),('含数字',r"[A-Za-z0-9']+")]
hdr = "保留 isn't"
print(f'  {"规则":<16} {"token 数":>9} {hdr:>13} {"含数字?":>9}')
for name,pat in rules:
    t=re.findall(pat, TEXT.lower())
    kept = "是" if "isn't" in t else "否（被切成 isn + t）"
    has_digit = "是" if any(c.isdigit() for x in t for c in x) else "否"
    print(f'  {name:<16} {len(t):>9} {kept:>18} {has_digit:>9}')
print("  读法：**「字母 + 撇号」保留 isn't（一个 token），而「仅字母」会切成 isn + t（两个）——")
print('        这直接影响 bigram 的产出（**同一段文本的特征数不同**），所以分词规则要写进规格')
```

预期输出要点（实跑）：① **三种定义**：151 字符、31 个 token → **重叠 30 个（唯一 27）、不重叠 15 个（唯一 15）、按句切分 27 个（唯一 24）**；② **稀疏性**：$|V|{=}50{,}000$ → 空间 **2,500,000,000**，本机观测 **27**；③ **分词规则**：仅字母 **32** 个 token（**`isn't` 被切成两个**）、字母+撇号 **31** 个（**保留 `isn't`**）、含数字 31 个。

## 常见追问

- **追问**：bigram 与 "two-word phrase" 是一回事吗？
  - 要点：**不完全是**：① **bigram 是"相邻两个 token"**（**纯位置概念**）；② **"两词短语"可能要求"有语义的搭配"**（**需要频次/PMI 过滤**）；③ **所以规格里要写清是"所有相邻对"还是"显著搭配"**。**读法**：**"PMI/频次过滤"是把 bigram 变成"短语"的关键一步**（**串 [[evaluation-02]]**）。
- **追问**：怎么处理标点？
  - 要点：**三种策略**：① **标点作为独立 token**（**"dog ." 成为一个 bigram**——**通常不想要**）；② **标点作为边界**（**句内切分**——**推荐**）；③ **标点被剥离但不断句**（**"dog," → "dog"**——**可能把两句粘起来**）。**读法**：**"标点作边界"是最常用的默认**——**但要显式声明**。
- **追问**：多语言怎么办？
  - 要点：**三点**：① **中文等无空格语言需要分词器**（**不能按空格切**——**串 [[llm-internals-01]] 的 tokenizer**）；② **Unicode 规范化**（**NFC/NFKC**）；③ **语言检测**（**混语文本要分别处理**）。**读法**：**"按空格分词"只对空格分隔的语言成立**——**这是一个常见的隐含假设**。
- **追问**：大数据量下怎么处理？
  - 要点：**三条**：① **流式**（**生成器 + 只保留上一个 token**）；② **分片并行**（**按文档分片，注意片间边界**）；③ **增量统计**（**用 Count-Min Sketch 或 Space-Saving 统计高频 bigram**——**串 [[gdm-01]]**）。**读法**：**"片间边界"是并行 bigram 的坑**（**一个文档被切开时会产生假 bigram**）。
- **追问**：怎么评估提取质量？
  - 要点：**三个角度**：① **覆盖**（**高频 bigram 是否都被提取**）；② **噪声**（**抽检"看起来不合理"的 bigram 比例**）；③ **下游效果**（**加了这个特征，模型指标提升多少**——**串 [[evaluation-01]]**）。**读法**：**"下游效果"是最终判据**——**但代价最高，所以先用前两个做筛选**。
- **追问**：为什么不用现成的库？
  - 要点：**看需求**：① **`nltk`/`spaCy` 提供分词与 n-gram**（**省事但引入依赖**）；② **它们的默认规则可能不符合你的规格**（**尤其是撇号与标点**）；③ **面试场景下要能自己写**（**因为考的就是"你能否定义清楚"**）。**读法**：**"用库"与"自己写"的选择取决于"默认规则是否符合需求"**。

## 相关题目

- [[gdm-03]]：RMSE——**同一家公司的另一道实现题**（**边界定义**）。
- [[gdm-05]]：bias-variance 权衡——**特征数量与过拟合的关系**。
- [[rag-02]]：向量索引与倒排——**稀疏特征的存储**。
- [[evaluation-02]]：分布漂移监控——**特征空间的变化检测**。

## 参考资料与归属

- **Foundations of Statistical Natural Language Processing（延伸）** —— Manning & Schütze，1999-06-28：<https://mitpress.mit.edu/9780262133609/foundations-of-statistical-natural-language-processing/>。**n-gram 模型、稀疏性与平滑**是本篇第 2 节的依据。
- **Speech and Language Processing（延伸）** —— Jurafsky & Martin，2024-01-01：<https://web.stanford.edu/~jurafsky/slp3/>。**分词、规范化与 n-gram 的工程细节**是本篇第 3、4 节的依据。
- **Unigram language model tokenization（延伸）** —— Kudo，2018-04-30：<https://arxiv.org/abs/1804.10959>。**子词分词的动机（词表与稀疏的取舍）**是本篇"稀疏性"一节的延伸。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（151 字符的英文语料、31 个 token、三种定义的 30/15/27、$|V|{=}50{,}000$ 与 $2.5\times10^9$ 的空间）都是**本机实跑结果**（**可复现**）；**计数与空间换算都是直接计算**。**⚠️ 具体数字依赖本机构造的语料与正则**——**换一段文本会得到不同的数量**；**可迁移的结论是"三种定义的产出不同（约 2 倍差距）"与"bigram 空间是 $|V|^2$"**，**不是具体数值**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
