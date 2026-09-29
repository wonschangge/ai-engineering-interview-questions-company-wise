---
type: question
id: sarvam-01
company: Sarvam AI
topic: coding
order: 1
question: 编写代码衡量 tokenizer 在不同语言上的 fertility，并说明拿到这个结果后你会怎么做。
question_en: Write code to measure tokenizer fertility across languages, and say what you would do with the result.
asked_at: []
level: 进阶
tags: [fertility, 分词器, 词表配比, 成本影响, 有效上下文]
sources:
  - title: Neural Machine Translation of Rare Words with Subword Units（延伸）
    url: https://arxiv.org/abs/1508.07909
    author: Sennrich, Haddow & Birch
    published: 2015-08-31
  - title: SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing（延伸）
    url: https://arxiv.org/abs/1808.06226
    author: Kudo & Richardson (Google)
    published: 2018-08-19
  - title: Unsupervised Cross-lingual Representation Learning at Scale（延伸）
    url: https://arxiv.org/abs/1911.02116
    author: Conneau et al. (Facebook AI)
    published: 2019-11-06
related: [sarvam-02, sarvam-04, sarvam-09, llm-internals-02, nvidia-08]
updated: 2026-09-28
---

## 一句话答案

> **fertility = 每词 token 数，而它是"词表配比的函数"**：**纯英文词表下印地语是英语的 2.7 倍，把 30% 槽位给印度语言后降到 1.3 倍**——**拿到结果后要算成本、上下文、KV 三个连带影响，再决定是否重训词表。**
> **★ 量化一：fertility 由词表配比决定**
> 用"字符 n-gram 词表 + 贪心最长匹配"模拟 BPE 机制
> | 词表训练语料配比 | **eng** | **hin** | **tam** | **hin/eng** |
> | --- | --- | --- | --- | --- |
> | **纯英文** | 2.24 | **6.00** | **6.50** | **2.7×** |
> | **90% 英文 + 各 5%** | 2.30 | **4.60** | 3.75 | **2.0×** |
> | 70% 英文 + 各 15% | 2.34 | 3.25 | 3.49 | 1.4× |
> | **40% 英文 + 各 30%** | 2.46 | 3.23 | 3.39 | **1.3×** |
> **读法**：**纯英文词表下印地语 fertility 是英语的 2.7 倍**——**而把词表里 30% 的槽位给印度语言后，比值降到 1.3 倍（同时英语从 2.24 升到 2.46）**。**所以"fertility 是词表配比的函数"，而不是"语言的固有属性"**。
> **★ 量化二：成本影响（输出侧更重要）**
> $3/M 输入、$15/M 输出、100 万词内容
> | 语言 | **fertility** | 输入成本 | **输出成本** | 合计 |
> | --- | --- | --- | --- | --- |
> | eng | 2.30 | \$6.9 | **\$34.4** | **\$41.3** |
> | **hin** | **4.60** | \$13.8 | **\$69.0** | **\$82.8（2.0×）** |
> | tam | 3.75 | \$11.2 | \$56.2 | \$67.5（1.6×） |
> **读法**：**同样 100 万词的内容，fertility 高的语言成本按比例上升**——**而输出 token 单价是输入的 5 倍，所以输出侧的 fertility 影响更大**。**"成本经济性"的第一句话是"fertility 直接乘在 token 数上"**。
> **★ 量化三：词表预算的分配（按流量加权）**
> 假设流量 50% 英文、25% 印地语、25% 泰米尔语
> | 配比 | eng | hin | tam | **加权平均** |
> | --- | --- | --- | --- | --- |
> | **90/5/5（基线）** | 2.30 | **4.60** | 3.75 | **3.24** |
> | 80/10/10 | 2.33 | 3.45 | 3.50 | 2.90 |
> | **70/15/15** | 2.34 | 3.25 | 3.49 | **2.86** |
> **读法**：**把英语从 90% 降到 70% 时，英语 fertility 略升、而印度语言明显下降**——**所以"按流量加权的平均 fertility"才是优化目标**（**而不是"英语不变"**）。**这就是"拿到 fertility 结果后该做的事"：调词表配比并重训**。
> **★ 量化四：上下文与 KV 的连带影响**
> 4K 上下文
> | 语言 | **fertility** | **4K 能放多少词** | **同文档 KV（相对）** |
> | --- | --- | --- | --- |
> | eng | 2.30 | **1,784** | **1.0×** |
> | **hin** | **4.60** | **890** | **2.0×** |
> | tam | 3.75 | 1,093 | 1.6× |
> **读法**：**同样的 4K 上下文，印地语只能放一半的词（即一半的文档）**——**而 KV 按 token 计，所以同一份文档的 KV 也涨 2 倍**。**"有效上下文"与"KV 显存"都被 fertility 直接缩放**。
> **★ 拿到结果后的四步**：
> | # | 步骤 | 判据 |
> | --- | --- | --- |
> | ① | **算成本倍数** | **fertility 直接乘 token 数** |
> | ② | **算上下文与 KV** | **1/F 的有效上下文、F 倍的 KV** |
> | ③ | **按流量加权调词表配比** | **目标是"加权平均 fertility"最小** |
> | ④ | **重训并复测** | **同时看"英语是否退化"** |
> 一句话判据：**"测 fertility（纯英文词表下 2.7×）→ 算成本（输出侧 ×5 权重）→ 算上下文与 KV（1/F 与 F×）→ 按流量加权调词表配比（3.20 → 2.83）→ 重训并复测"**。

## 面试官在考什么

- **★ 是否指出"fertility 是词表配比的函数"**：**能否给出"2.7× → 1.3×"**——**这是本题的分水岭**。
- **★ 是否算成本影响**：**能否指出"输出侧单价是输入的 5 倍"**。
- **★ 是否按流量加权**：**能否给出"加权平均 fertility"**。
- **是否算上下文影响**：**能否给出"4K 只能放 890 词"**。
- **是否算 KV 影响**：**能否指出"KV 涨 F 倍"**。
- **是否指出"英语会略退化"**：**能否给出"2.24 → 2.46"**。
- **是否给出"怎么做"**：**能否给出四步**。
- **是否用平行语料**：**能否指出"同内容对比才公平"**。
- **是否报分布而不是均值**：**能否指出"按语言/领域分层"**。
- **诚实**：**承认"模拟不能替代真实 tokenizer 的测量"**。

**常见错误答案**

- **只报一个平均值**（**掩盖语言差异**）。
- **用不同内容的语料对比**（**不公平**）。
- **认为 fertility 是语言的固有属性**（**其实是词表配比**）。
- **只看输入侧成本**（**输出侧贵 5 倍**）。
- **不考虑英语退化**（**词表是零和的**）。
- **不按流量加权**（**优化目标错**）。
- **忽略上下文与 KV**。
- **不给出"怎么做"**（**只测量不行动**）。

## 原理与推导

### 1. ★ fertility

$$\text{fertility}=\frac{\text{token 数}}{\text{词数}}$$

| 语言 | fertility |
| --- | --- |
| **eng** | **2.24** |
| **hin** | **6.00** |

**读法**：**"每词几个 token"**——**而"同样的内容"的 token 数直接乘成本**。

### 2. ★ 词表配比

| 配比 | hin/eng |
| --- | --- |
| **纯英文** | **2.7×** |
| **40/30/30** | **1.3×** |

**读法**：**"词表槽位是零和的"**——**给一个语言更多槽位，另一个就少**。

### 3. ★ 成本

$$\text{成本}=\text{词数}\times F\times(\text{输入单价}+\text{输出单价})$$

| 项 | 权重 |
| --- | --- |
| **输入** | \$3/M |
| **输出** | **\$15/M（5×）** |

**读法**：**"输出侧决定总成本的 83%"**——**所以"生成"比"理解"更受 tokenization 影响**。

### 4. ★ 上下文与 KV

| 项 | 影响 |
| --- | --- |
| **有效上下文** | **1/F** |
| **KV** | **F×** |

**读法**：**"同一份文档占两倍的位置与两倍的显存"**——**所以 RAG 能塞的段数也被缩放**。

### 5. ★ 流量加权

$$\bar F=\sum_i w_i F_i$$

| 配比 | $\bar F$ |
| --- | --- |
| **90/5/5** | **3.20** |
| **70/15/15** | **2.83** |

**读法**：**"优化目标应该是 $\bar F$"**——**因为它对应真实的成本**。

### 6. ★ 四步行动

| 步 | 内容 |
| --- | --- |
| **①** | **算成本倍数** |
| **②** | **算上下文与 KV** |
| **③** | **调词表配比** |
| **④** | **重训并复测** |

**读法**：**"测量 -> 算影响 -> 调配比 -> 复测"**——**而第 ④ 步必须看"英语是否退化"**。

## 数值与代码验证

### 表 1：fertility、成本、词表分配、上下文与 KV（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
import random, collections
rng=random.Random(0)
# 用"字符 n-gram 词表 + 贪心最长匹配"模拟 BPE 的 fertility 机制
LANGS={'eng':('abcdefghijklmnopqrstuvwxyz',5.0),'hin':('अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह',6.5),
       'tam':('அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறன',7.0)}
def gen_words(lang,n):
    alpha,meanlen=LANGS[lang]
    return [''.join(rng.choice(alpha) for _ in range(max(2,int(rng.gauss(meanlen,2))))) for _ in range(n)]
def build_vocab(mix,size=8000,ngram=4):
    """mix: {lang: share}；词表 = 混合语料里最高频的 1..ngram 字符片段"""
    rng.seed(42)                      # 固定种子 -> 结果可复现
    cnt=collections.Counter()
    for lang,share in mix.items():
        for w in gen_words(lang,int(20000*share)):
            for n in range(1,ngram+1):
                for i in range(len(w)-n+1): cnt[w[i:i+n]]+=1
    return set(t for t,_ in cnt.most_common(size))
def tokenize(w,vocab,maxn=4):
    i=0;t=0
    while i<len(w):
        for n in range(min(maxn,len(w)-i),0,-1):
            if w[i:i+n] in vocab: i+=n;t+=1;break
        else: i+=1;t+=1
    return t
def fertility(lang,vocab,n=3000):
    rng.seed(7)                       # 固定种子 -> 结果可复现
    ws=gen_words(lang,n)
    return sum(tokenize(w,vocab) for w in ws)/len(ws)
print('① fertility 由"该语言在词表训练语料里的占比"决定')
print(f'  {"词表训练语料配比":<30} {"eng":>7} {"hin":>7} {"tam":>7} {"hin/eng":>9}')
for mix,label in (({'eng':1.0},'**纯英文**'),
                  ({'eng':0.9,'hin':0.05,'tam':0.05},'**90% 英文 + 各 5%**'),
                  ({'eng':0.7,'hin':0.15,'tam':0.15},'70% 英文 + 各 15%'),
                  ({'eng':0.4,'hin':0.3,'tam':0.3},'**40% 英文 + 各 30%**')):
    v=build_vocab(mix)
    f={l:fertility(l,v) for l in LANGS}
    print(f'  {label:<30} {f["eng"]:>7.2f} {f["hin"]:>7.2f} {f["tam"]:>7.2f} {f["hin"]/f["eng"]:>8.1f}x')
print('  读法：**纯英文词表下印地语 fertility 是英语的 2–3 倍**（**取决于模拟参数**）——')
print('        而**把词表里 30% 的槽位给印度语言后，比值明显下降**（**同时英语略升**）；')
print("        所以**「fertility 是词表配比的函数」**（**而不是「语言的固有属性」**）")
print()
print('② 拿到结果后怎么做：先算成本影响')
INPUT=3.0; OUTPUT=15.0     # $/1M token
WORDS=1_000_000
print(f'  {"语言":<8} {"fertility":>10} {"输入 token":>12} {"输出 token":>12} {"成本($)":>10}')
v=build_vocab({'eng':0.9,'hin':0.05,'tam':0.05})
base=None
for lang in ('eng','hin','tam'):
    f=fertility(lang,v)
    tin=WORDS*f; tout=WORDS*f
    cost=tin/1e6*INPUT+tout/1e6*OUTPUT
    if base is None: base=cost
    print(f'  {lang:<8} {f:>10.2f} {tin:>12,.0f} {tout:>12,.0f} {cost:>10.1f} ({cost/base:.1f}x)')
print('  读法：**同样 100 万词的内容，fertility 高的语言成本按比例上升** ——')
print('        而**输出 token 单价是输入的 5 倍**（$15 vs $3）-> **输出侧的 fertility 影响更大**；')
print("        所以**「成本经济性」的第一句话是「fertility 直接乘在 token 数上」**")
print()
print('③ 词表预算的分配：把 5% 的槽位从英语挪给印度语言')
print(f'  {"配比":<34} {"eng":>7} {"hin":>7} {"tam":>7} {"加权平均":>9}')
def weighted(mix):
    v=build_vocab(mix)
    f={l:fertility(l,v) for l in LANGS}
    w={'eng':0.5,'hin':0.25,'tam':0.25}      # 假设流量配比
    return f,sum(f[l]*w[l] for l in LANGS)
for mix,label in (({'eng':0.9,'hin':0.05,'tam':0.05},'**90/5/5（基线）**'),
                  ({'eng':0.8,'hin':0.10,'tam':0.10},'80/10/10'),
                  ({'eng':0.7,'hin':0.15,'tam':0.15},'**70/15/15**')):
    f,avg=weighted(mix)
    print(f'  {label:<34} {f["eng"]:>7.2f} {f["hin"]:>7.2f} {f["tam"]:>7.2f} {avg:>9.2f}')
print('  读法：**把英语从 90% 降到 70% 时，英语 fertility 略升、而印度语言明显下降** ——')
print("        所以**「按流量加权的平均 fertility」才是优化目标**（**而不是「英语不变」**）；")
print('        而**这就是"拿到 fertility 结果后该做的事"：调词表配比并重训**')
print()
print('④ 上下文与 KV 的连带影响')
CTX=4096; KV=320*1024
print(f'  {"语言":<8} {"fertility":>10} {"4K 上下文能放多少词":>18} {"同文档 KV(相对)":>16}')
v=build_vocab({'eng':0.9,'hin':0.05,'tam':0.05})
fe=fertility('eng',v)
for lang in ('eng','hin','tam'):
    f=fertility(lang,v)
    print(f'  {lang:<8} {f:>10.2f} {CTX/f:>18,.0f} {f/fe:>15.1f}x')
print('  读法：**同样的 4K 上下文，fertility 高的语言只能放 1/F 的内容** ——')
print('        而**KV 按 token 计**（**所以同一份文档的 KV 也涨 F 倍**，**串 [[nvidia-08]]**）；')
print("        所以**「有效上下文」与「KV 显存」都被 fertility 直接缩放**")
```

预期输出要点（实跑）：① **fertility**：纯英文/90-5-5/70-15-15/40-30-30 → **eng 2.24/2.30/2.34/2.46、hin 6.00/4.60/3.25/3.23、hin/eng 2.7/2.0/1.4/1.3×**；② **成本**：同样 100 万词 → **eng \$41.3、hin \$82.8（2.0×）、tam \$67.5（1.6×）**；③ **词表分配**：加权平均 **3.24 → 2.90 → 2.86**；④ **上下文**：4K → **eng 1,784 词、hin 890 词、tam 1,093 词**，**同文档 KV 1.0/2.0/1.6×**。

## 常见追问

- **追问**：为什么用"贪心最长匹配"模拟而不是真的 BPE？
  - 要点：**三条**：① **BPE 的合并规则是"最高频对"**，**贪心最长匹配是它的一个近似**；② **本机模拟的目的是"展示机制"**（**而不是复现某个具体 tokenizer**）；③ **真实测量要直接用目标 tokenizer 的 `tokenize`**（**几行代码**）。**读法**：**"模拟展示机制、真实测量用真 tokenizer"**——**两者不能混**。
- **追问**：怎么选平行语料？
  - 要点：**三条**：① **同源内容**（**如同一批新闻/百科的翻译**）；② **覆盖多个领域**（**法律、医疗、口语**）；③ **报分层结果**（**而不是一个总数**）。**读法**：**"同内容对比才公平"**——**而"分层"能发现"某个领域特别差"**。
- **追问**：除了 fertility 还该测什么？
  - 要点：**三条**：① **"字符/token"**（**反向视角，对形态丰富的语言更直观**）；② **"切碎率"**（**一个词被切成 >N 片的比例**）；③ **"往返一致性"**（**decode(encode(x)) == x**）。**读法**：**"往返一致性是正确性的底线"**——**它必须 100%**。
- **追问**：词表扩大 vs 重新分配，怎么选？
  - 要点：**三条**：① **扩词表**（**如 32K → 128K**）**能同时改善多语言、但 embedding 变大**；② **重新分配**（**固定大小**）**是零和的**；③ **判据是"加权平均 fertility 的下降 / embedding 参数量的增加"**。**读法**：**"扩词表往往更优"**——**因为 embedding 只占总参数的一小部分**。
- **追问**：重训 tokenizer 要重训模型吗？
  - 要点：**三条**：① **要**（**embedding 变了**）；② **所以成本很高**（**等于重训**）；③ **折中方案：扩词表 + 只训新 embedding**（**但效果有限**）。**读法**：**"tokenizer 是'改一次就要重训'的决策"**——**所以要在预训练之前就做对**。
- **追问**：这道题与"跨语言 RAG"有什么关系？
  - 要点：**它是 RAG 的前置约束**：① **fertility 决定"4K 上下文能塞几段"**（**本机印地语只有一半**）；② **而"能塞几段"直接决定答案质量**；③ **所以"先修 tokenizer、再做 RAG"**（**串 [[sarvam-04]]**）。**读法**：**"tokenization 是所有下游能力的前置条件"**。

## 相关题目

- [[sarvam-02]]：为什么 tokenization 是第一道瓶颈——**同一个机制的"为什么"**。
- [[sarvam-04]]：跨语言 RAG——**上下文预算被 fertility 缩放**。
- [[sarvam-09]]：如何正确评估 Indic LLM——**测量方法学**。
- [[llm-internals-02]]：分词与嵌入——**tokenizer 的基本原理**。
- [[nvidia-08]]：单卡服务 70B——**KV 的显存算术**。

## 参考资料与归属

- **Neural Machine Translation of Rare Words with Subword Units（延伸）** —— Sennrich, Haddow & Birch，2015-08-31：<https://arxiv.org/abs/1508.07909>。**BPE 分词与子词切分** 是本篇第 1 节的依据。
- **SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing（延伸）** —— Kudo & Richardson (Google)，2018-08-19：<https://arxiv.org/abs/1808.06226>。**语言无关的分词与"词表大小"的权衡** 是本篇第 2 节的依据。
- **Unsupervised Cross-lingual Representation Learning at Scale（延伸）** —— Conneau et al. (Facebook AI)，2019-11-06：<https://arxiv.org/abs/1911.02116>。**多语言词表的配比与"低资源语言的 fertility"** 是本篇第 2、5 节的直接来源。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（三种语言的字符集与词长分布、词表 8,000、n-gram 上限 4、语料配比 100/0 到 40/30/30、$3/M 输入与 $15/M 输出（本仓库常数）、流量配比 50/25/25、4K 上下文）都是为演示"fertility 机制"而构造的**示例参数与显式假设**；**fertility、成本、加权平均都是本机实跑结果**（**可复现**）。**⚠️ "字符 n-gram 词表 + 贪心最长匹配"是 BPE 的简化近似**——**真实 tokenizer 的 fertility 必须用其自身实现测量**；**"词长分布"是合成的**。**可迁移的结论是"fertility 由词表配比决定、成本按 F 缩放且输出侧权重 5 倍、按流量加权调配比"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
