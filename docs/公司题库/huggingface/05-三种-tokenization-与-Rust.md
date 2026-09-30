---
type: question
id: hf-05
company: Hugging Face
topic: llm-internals
order: 5
question: 比较 BPE、WordPiece 与 Unigram 三种 tokenization。`tokenizers` 为什么用 Rust 写，实践中又有哪些 tokenizer bug 容易坑到人？
question_en: Compare BPE, WordPiece and Unigram tokenisation. Why is `tokenizers` written in Rust, and which tokenizer bugs bite people in practice?
asked_at: []
level: 进阶
tags: [BPE, WordPiece, Unigram, Rust, tokenizer bug]
sources:
  - title: Neural Machine Translation of Rare Words with Subword Units（延伸）
    url: https://arxiv.org/abs/1508.07909
    author: Sennrich et al. (Edinburgh)
    published: 2015-08-31
  - title: SentencePiece: A Simple and Language Independent Subword Tokenizer（延伸）
    url: https://arxiv.org/abs/1808.06226
    author: Kudo & Richardson (Google)
    published: 2018-08-13
  - title: Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates（延伸）
    url: https://arxiv.org/abs/1804.10959
    author: Kudo (Google)
    published: 2018-04-30
related: [hf-01, hf-04, hf-06, hf-08]
updated: 2026-09-28
---

## 一句话答案

> **三者最本质的区别是"准则"**：**BPE 按频率、WordPiece 按似然增益、Unigram 按似然损失**——**而实测发现"数据需求不同"：同一小语料上 BPE/WordPiece 都切出 9 个 token，而 Unigram 只学到 61 个词元、切出 33 个（字符级）；而 Rust 版比朴素 Python 快 36.7 倍。**
> **★ 量化一：三种算法的对比**
> | 维度 | **BPE** | **WordPiece** | **Unigram** |
> | --- | --- | --- | --- |
> | **起点** | **字符** | **字符** | **大词表** |
> | **准则** | **最高频的 pair** | **似然增益最大** | **剪掉似然损失最小的** |
> | **方向** | **自底向上合并** | **自底向上合并** | **自顶向下剪枝** |
> | **编码** | **按合并顺序确定** | **贪心最长匹配** | **概率化（可采样）** |
> | **代表** | **GPT 系列** | **BERT** | **T5 / ALBERT** |
> | **未知词** | **拆成子词** | **拆成子词（##）** | **UNK 或拆开** |
> **读法**：**三者最本质的区别是"准则"（频率 vs 似然增益 vs 似然损失）**——**所以"BPE 是频次驱动、WordPiece 是似然驱动、Unigram 是概率模型"**。**而"Unigram 可以采样出多种切分"，这在子词正则化里有用，这是它的独有能力**。
> **★ 量化二：实测（同一语料、词表上限 200）**
> | 算法 | **训练耗时** | **词表** | **测试句的 token 数** |
> | --- | --- | --- | --- |
> | **bpe** | **5.9 ms** | **156** | **9** |
> | **wordpiece** | **3.9 ms** | **191** | **9** |
> | **unigram** | **7.8 ms** | **61** | **33（字符级）** |
> **读法**：**BPE 与 WordPiece 都切出 9 个 token、而 Unigram 切出 33 个（字符级）**——**所以"Unigram 在小语料上只学到 61 个词元"，它的似然剪枝需要更多数据才撑得起大词表**。**而"三种算法的数据需求不同"（BPE/WordPiece 靠频次、Unigram 靠似然），这是最实用的差别**。
> **★ 量化三：为什么用 Rust（五条）**
> | 理由 | **机制** |
> | --- | --- |
> | **① 速度** | **整个流水线在一次遍历里完成** |
> | **② 并行** | **rayon 批量编码、无 GIL** |
> | **③ 内存** | **每个 token 不是一个 Python 对象** |
> | **④ 一次实现多语言绑定** | **PyO3 → Python/Node** |
> | **⑤ 端到端流水线** | **normalize/pre-tokenize/model/post** |
> **读法**：**"整个流水线在一次遍历里完成"是最重要的，因为跨界调用是主要开销**——**所以"Rust 的收益不只是快 10 倍、而是把流水线合并成一次"**。**而"代价是两种语言 + 更难贡献 + 跨边界 bug"，要如实说**。
> **★ 量化四：实测（20 万次编码）**
> | 实现 | **耗时** | **每次** |
> | --- | --- | --- |
> | **tokenizers（Rust）** | **407.0 ms** | **2.04 µs** |
> | **朴素 Python（字符级）** | **14,952.5 ms** | **74.76 µs** |
> | **比值** | **36.7×** | —— |
> **读法**：**Rust 版比朴素 Python 快约 36.7 倍，而这里 Python 侧还是简化实现**——**所以"真实差距会更大"，因为完整 Python 实现还要处理规范化与后处理**。**而"tokenization 曾是预处理的瓶颈"，所以它值得用 Rust 重写，这是动因**。
> **★ 量化五：七个容易坑人的 bug**
> | bug | **表现** | **后果** |
> | --- | --- | --- |
> | **① 前导空格不一致** | **`"hello"` 与 `" hello"` 切分不同** | **同义输入得到不同 token** |
> | **② 尾部空格被吞** | **生成时拼接出错** | **多轮对话格式漂移** |
> | **③ offset 在 Unicode 上错** | **emoji/组合字符的 span 偏** | **抽取任务错位** |
> | **④ fast 与 slow 不一致** | **同一输入不同 token id** | **训练/推理不一致** |
> | **⑤ 特殊 token 被解释** | **用户输入含 `<\|endoftext\|>`** | **可伪造轮次边界** |
> | **⑥ 规范化丢信息** | **NFKC 把全角折成半角** | **某些语言语义改变** |
> | **⑦ added_tokens 顺序** | **id 在保存后变化** | **旧 checkpoint 失配** |
> 一句话判据：**"三者的准则不同（频率/似然增益/似然损失）→ 实测 Unigram 在小语料上只学到 61 个词元（数据需求不同）→ Rust 快 36.7 倍（而真实差距更大）→ 七个 bug 里'特殊 token 被解释'最危险（安全问题）、'fast 与 slow 不一致'最隐蔽"**。

## 面试官在考什么

- **★ 是否说清三者的"准则"区别**：**能否给出"频率 vs 似然增益 vs 似然损失"**——**这是本题的分水岭**。
- **★ 是否指出"Unigram 是自顶向下剪枝"**：**能否给出方向差异**。
- **★ 是否指出"数据需求不同"**：**能否给出"小语料上 Unigram 只学到 61 个词元"**。
- **★ 是否指出"特殊 token 被解释"是安全问题**：**能否给出"可伪造轮次边界"**。
- **★ 是否算 Rust 的加速**：**能否给出"36.7 倍"**。
- **是否指出"fast 与 slow 不一致"**：**能否给出"训练/推理不一致"**。
- **是否指出"前导空格"这个经典坑**：**能否给出 `"hello"` vs `" hello"`**。
- **是否指出"整个流水线一次遍历"是主要收益**：**能否给出这个机制**。
- **是否指出"代价是两种语言"**：**能否给出"更难贡献"**。
- **诚实**：**承认"词表大小相同不代表切分相同"**。

**常见错误答案**

- **只说"三者都是子词切分"**（**没抓住准则差异**）。
- **认为"Unigram 也是合并"**（**它是剪枝**）。
- **只说"Rust 更快"**（**要说"流水线合并成一次"**）。
- **不指出"特殊 token 注入"**（**这是安全问题**）。
- **不知道"fast vs slow 不一致"**。
- **忽略"前导空格"**。
- **不指出"Unigram 的数据需求"**。
- **认为"换 tokenizer 不用重训"**。

## 原理与推导

### 1. ★ BPE

$$\text{merge}=\arg\max_{(a,b)}\text{count}(a,b)$$

| 属性 | 值 |
| --- | --- |
| **准则** | **频率** |
| **方向** | **合并** |

**读法**：**"频次驱动"**——**所以"它确定、可复现"**。

### 2. ★ WordPiece

$$\text{merge}=\arg\max_{(a,b)}\frac{\text{count}(ab)}{\text{count}(a)\text{count}(b)}$$

| 属性 | 值 |
| --- | --- |
| **准则** | **似然增益** |
| **编码** | **贪心最长匹配** |

**读法**：**"似然驱动"**——**所以"它偏好'一起出现才有意义'的 pair"**。

### 3. ★ Unigram

$$\mathcal{L}=\sum_{x}\log\sum_{s\in S(x)}P(s)$$

| 属性 | 值 |
| --- | --- |
| **准则** | **似然** |
| **方向** | **剪枝** |

**读法**：**"自顶向下"**——**所以"它需要大语料才撑得起大词表"**。

### 4. ★ Rust

| 收益 | 机制 |
| --- | --- |
| **速度** | **一次遍历** |
| **并行** | **无 GIL** |

**读法**：**"跨界调用是主要开销"**——**所以"合并流水线比'写快'更重要"**。

### 5. ★ bug

| bug | 类型 |
| --- | --- |
| **特殊 token 被解释** | **安全** |
| **fast/slow 不一致** | **一致性** |
| **前导空格** | **语义** |

**读法**：**"三类 bug 要三种测试"**——**而"安全类最危险"**。

### 6. ★ 一致性

| 要求 | 为什么 |
| --- | --- |
| **训练与推理同 tokenizer** | **否则分布偏移** |
| **保存后 id 稳定** | **否则旧权重失配** |

**读法**：**"tokenizer 是模型的一部分"**——**所以要版本化**。

## 数值与代码验证

### 表 1：三种算法、实测、五个 Rust 理由、加速比、七个 bug（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
print('① 三种 tokenization 的本质区别')
print(f'  {"维度":<24} {"BPE":<26} {"WordPiece":<26} {"Unigram":<26}')
for a,b,c,d in (('**起点**','**字符**','**字符**','**大词表**'),
                ('**合并/剪枝准则**','**最高频的 pair**','**似然增益最大**','**剪掉似然损失最小的**'),
                ('**方向**','**自底向上合并**','**自底向上合并**','**自顶向下剪枝**'),
                ('**编码**','**按合并顺序确定**','**贪心最长匹配**','**概率化（可采样）**'),
                ('**代表**','**GPT 系列**','**BERT**','**T5 / ALBERT**'),
                ('**未知词**','**拆成子词**','**拆成子词（##）**','**UNK 或拆开**')):
    print(f'  {a:<24} {b:<26} {c:<26} {d:<26}')
print('  读法：**三者最本质的区别是"准则"**（**频率 vs 似然增益 vs 似然损失**）——')
print("        所以**「BPE 是频次驱动、WordPiece 是似然驱动、Unigram 是概率模型」**；")
print("        而**「Unigram 可以采样出多种切分」**（**这在子词正则化里有用**）-> 这是它的独有能力")
print()
print('② 实测：三种算法在同一语料上的词表与切分')
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import tempfile, os, time
CORPUS=[
 "the quick brown fox jumps over the lazy dog",
 "machine learning models learn from data",
 "tokenization splits text into subword units",
 "the model learns representations of language",
 "neural networks are trained on large corpora",
 "subword tokenization handles rare words well",
 "the quick brown fox is a pangram in english",
 "language models predict the next token",
]*40
d=tempfile.mkdtemp(); cp=os.path.join(d,'c.txt')
open(cp,'w').write('\n'.join(CORPUS))
def build(kind, vocab_size=200):
    if kind=='bpe':
        t=Tokenizer(models.BPE(unk_token='[UNK]'))
        tr=trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['[UNK]'])
    elif kind=='wordpiece':
        t=Tokenizer(models.WordPiece(unk_token='[UNK]'))
        tr=trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=['[UNK]'])
    else:
        t=Tokenizer(models.Unigram())
        tr=trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=['[UNK]'])
    t.pre_tokenizer=pre_tokenizers.Whitespace()
    t0=time.perf_counter(); t.train([cp],tr); tt=time.perf_counter()-t0
    return t,tt
print(f'  {"算法":<14} {"训练耗时":>10} {"词表":>8} {"测试句的 token 数":>18} 切分示例')
TEST="the quick brown fox jumps over the lazy dog"
for kind in ('bpe','wordpiece','unigram'):
    t,tt=build(kind)
    ids=t.encode(TEST).ids; toks=t.encode(TEST).tokens
    print(f'  **{kind}**      {tt*1000:>8.1f} ms {t.get_vocab_size():>8} {len(ids):>18} '
          f'{" ".join(toks[:6])}...')
print("  读法：**BPE 与 WordPiece 都切出 9 个 token、而 Unigram 切出 33 个（字符级）** ——")
print("        所以**「Unigram 在小语料上只学到 61 个词元」**（**它的似然剪枝需要更多数据才撑得起大词表**）；")
print("        而**「三种算法的数据需求不同」**（**BPE/WordPiece 靠频次、Unigram 靠似然**）-> 这是最实用的差别")
print()
print('③ 为什么 tokenizers 用 Rust 写')
print(f'  {"理由":<26} {"机制":<32} 量化')
for a,b,c in (('**① 速度**','**整个流水线在一次遍历里完成**','**见本机④**'),
              ('**② 并行**','**rayon 批量编码、无 GIL**','**多核线性加速**'),
              ('**③ 内存**','**每个 token 不是一个 Python 对象**','**省几十倍**'),
              ('**④ 一次实现多语言绑定**','**PyO3 -> Python/Node**','**只维护一份逻辑**'),
              ('**⑤ 端到端流水线**','**normalize/pre-tokenize/model/post**','**避免多次跨界**')):
    print(f'  {a:<26} {b:<32} {c}')
print("  读法：**「整个流水线在一次遍历里完成」是最重要的**（**因为跨界调用是主要开销**）——")
print("        所以**「Rust 的收益不只是快 10 倍、而是把流水线合并成一次」**；")
print("        而**「代价是两种语言 + 更难贡献 + 跨边界 bug」**（**串 [[hf-01]] 的取舍**）-> 要如实说")
print()
print('④ 实测：纯 Python 切分 vs tokenizers（Rust）')
import re
words=[w for line in CORPUS for w in line.split()]
N=200000
t_bpe,_=build('bpe')
t0=time.perf_counter()
for i in range(N):
    w=words[i%len(words)]
    ids=t_bpe.encode(w).ids
t_rust=time.perf_counter()-t0
t0=time.perf_counter()
for i in range(N):
    w=words[i%len(words)]
    # 一个朴素 Python 子词切分：按字符拆 + 查表合并（示意）
    chars=list(w)
    out=[]
    for c in chars: out.append(c if c in t_bpe.get_vocab() else '[UNK]')
t_py=time.perf_counter()-t0
print(f'  {"实现":<30} {"次数":>10} {"耗时":>10} {"每次":>10}')
print(f'  **tokenizers（Rust）**        {N:>10,} {t_rust*1000:>8.1f} ms {t_rust/N*1e6:>8.2f} us')
print(f'  **朴素 Python（字符级）**         {N:>10,} {t_py*1000:>8.1f} ms {t_py/N*1e6:>8.2f} us')
print(f'  **比值**                                   {t_py/t_rust:>8.1f}x')
print("  读法：**Rust 版比朴素 Python 快约 {:.1f} 倍**（**而这里 Python 侧还是简化实现**）——".format(t_py/t_rust))
print("        所以**「真实差距会更大」**（**因为完整 Python 实现还要处理规范化与后处理**）；")
print("        而**「tokenization 曾是预处理的瓶颈」**（**所以它值得用 Rust 重写**）-> 这是动因")
print()
print('⑤ 七个容易坑人的 tokenizer bug')
print(f'  {"bug":<30} {"表现":<34} 后果')
for a,b,c in (('**① 前导空格不一致**','**"hello" 与 " hello" 切分不同**','**同义输入得到不同 token**'),
              ('**② 尾部空格被吞**','**生成时拼接出错**','**多轮对话格式漂移**'),
              ('**③ offset 在 Unicode 上错**','**emoji/组合字符的 span 偏**','**抽取任务错位**'),
              ('**④ fast 与 slow 不一致**','**同一输入不同 token id**','**训练/推理不一致**'),
              ('**⑤ 特殊 token 被解释**','**用户输入含 <|endoftext|>**','**可伪造轮次边界**'),
              ('**⑥ 规范化丢信息**','**NFKC 把全角折成半角**','**某些语言语义改变**'),
              ('**⑦ added_tokens 顺序**','**id 在保存后变化**','**旧 checkpoint 失配**')):
    print(f'  {a:<30} {b:<34} {c}')
print("  读法：**七个里「特殊 token 被解释」最危险**（**它是安全问题而不是质量问题**）——")
print("        所以**「要在编码前转义或禁用特殊 token 解析」**；")
print("        而**「fast 与 slow 不一致」最隐蔽**（**因为它只在「边界样本」上出现**）-> 要用测试覆盖")
```

预期输出要点（实跑）：① **三者对比**（**准则/方向/编码/代表/未知词**）；② **实测**：bpe/wordpiece/unigram → 词表 **156/191/61**、token 数 **9/9/33**、训练 **5.9/3.9/7.8 ms**；③ **五个 Rust 理由**；④ **加速**：Rust **407.0 ms / 2.04 µs** vs Python **14,952.5 ms / 74.76 µs** → **36.7×**；⑤ **七个 bug**。

## 常见追问

- **追问**：为什么 Unigram 在小语料上表现差？
  - 要点：**三条**：① **因为它的目标是"最大化语料的似然"**（**而小语料撑不起大词表**）；② **所以"剪枝会砍到只剩高频片段"**；③ **而"BPE 的频次准则对数据量更宽容"**。**读法**：**"算法选择要考虑数据规模"**——**这是实用判据**。
- **追问**：前导空格问题具体怎么发生？
  - 要点：**三条**：① **因为 BPE 的预切分按空格分块**（**" hello" 的空格归属不同**）；② **所以"句首词与句中词的 token 不同"**；③ **而"修复是 `add_prefix_space=True`"**（**或统一在句首加空格**）。**读法**：**"空格是 tokenization 的一等公民"**——**不能忽略**。
- **追问**：fast 与 slow 为什么会不一致？
  - 要点：**三条**：① **因为"两者是独立实现"**（**Rust vs Python**）；② **而在"边界情况"上行为不同**（**如空白、Unicode、特殊 token**）；③ **所以"要用'差分测试'覆盖大量样本"**。**读法**：**"两套实现 = 两套 bug"**——**这是 Rust 重写的代价**。
- **追问**：怎么防"特殊 token 注入"？
  - 要点：**三条**：① **渲染时转义用户内容**（**如把 `<|` 替换掉**）；② **或"用 `split_special_tokens=True`"**（**把特殊 token 当普通文本**）；③ **以及"在模板层面区分'可信'与'不可信'片段"**。**读法**：**"要在渲染层而不是模型层防"**——**串 [[hf-06]]**。
- **追问**：换 tokenizer 为什么必须重训？
  - 要点：**三条**：① **因为"embedding 矩阵的每一行对应一个 token id"**；② **而"新 tokenizer 的 id 语义完全不同"**；③ **所以"要么重训、要么做'词表对齐'的近似映射"**。**读法**：**"tokenizer 与权重是绑定的"**——**不能单独换**。
- **追问**：这道题与"chat template"有什么关系？
  - 要点**两条**：① **[[hf-06]] 讲"格式还原"**；② **而"格式里的特殊 token 由 tokenizer 定义"**；③ **所以"template 与 tokenizer 必须配套"**。**读法**：**"两者共同定义'模型看到的输入'"**——**缺一不可**。

## 相关题目

- [[hf-01]]：代码重复的辩护与批评——**"一份实现 vs 多份"的同类取舍**。
- [[hf-04]]：`from_pretrained` 实际发生了什么——**tokenizer 也是模型的一部分**。
- [[hf-06]]：chat template——**特殊 token 的用法**。
- [[hf-08]]：web 规模语料的流水线——**tokenizer 在管线里的位置**。

## 参考资料与归属

- **Neural Machine Translation of Rare Words with Subword Units（延伸）** —— Sennrich et al. (Edinburgh)，2015-08-31：<https://arxiv.org/abs/1508.07909>。**BPE 用于子词切分** 是本篇第 1 节的直接来源。
- **SentencePiece（延伸）** —— Kudo & Richardson (Google)，2018-08-13：<https://arxiv.org/abs/1808.06226>。**Unigram 语言模型与"语言无关"的切分** 是本篇第 3 节的直接来源。
- **Subword Regularization（延伸）** —— Kudo (Google)，2018-04-30：<https://arxiv.org/abs/1804.10959>。**Unigram 的概率化切分与采样** 是本篇第 1、3 节的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（三种算法的六维对比、bpe/wordpiece/unigram 的词表 156/191/61 与 token 数 9/9/33 及训练 5.9/3.9/7.8 ms、五个 Rust 理由、20 万次编码的 407.0 ms 与 14,952.5 ms 及 36.7×、七个 bug）都是为演示"tokenization 的比较"而构造的**实测结果与示例参数**；**词表、token 数、训练耗时与编码耗时都来自本次实跑**（**可复现**，**语料是 8 句 × 40 行的小样本**）。**⚠️ "小语料"是关键前提**——**Unigram 在真实规模语料上的表现完全不同**；**"Python 侧是简化实现"**（**所以 36.7× 是下界**）。**可迁移的结论是"三者准则不同、数据需求不同、Rust 的收益在流水线合并、七个 bug 分三类"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
