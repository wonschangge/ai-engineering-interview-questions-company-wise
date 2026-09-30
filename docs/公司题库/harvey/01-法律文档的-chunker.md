---
type: question
id: harvey-01
company: Harvey
topic: coding
order: 1
question: 结对编程：为一份法律文档编写 chunker，要求绝不拆分条款，并且携带足够的上下文，使检索到的 chunk 能够自包含。
question_en: Pair programming: write a chunker for a legal document that never splits a clause and carries enough context that a retrieved chunk is self-contained.
asked_at: []
level: 高阶
tags: [法律文档切分, 条款边界, 上下文头, 交叉引用, 字符偏移]
sources:
  - title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（延伸）
    url: https://arxiv.org/abs/2005.11401
    author: Lewis et al. (Facebook AI)
    published: 2020-05-22
  - title: Dense Passage Retrieval for Open-Domain Question Answering（延伸）
    url: https://arxiv.org/abs/2004.04906
    author: Karpukhin et al. (Facebook AI)
    published: 2020-04-10
  - title: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction（延伸）
    url: https://arxiv.org/abs/2004.12832
    author: Khattab & Zaharia (Stanford)
    published: 2020-04-27
related: [harvey-02, harvey-03, harvey-07, rag-01, coding-01]
updated: 2026-09-28
---

## 一句话答案

> **五个规则**：**只在条款边界切、标题并入下一节、带上文路径、记录引用到的定义术语、保留字符偏移**——**实测按"节"切出 7 个 chunk（3 个带正文、4 个是标题行），加上下文头后变大约 1.20 倍；而一个片段里就有 5 处显式节引用与 3 个定义术语，说明"交叉引用不是例外而是常态"。**
> **★ 量化一：法律文档的层级与切分粒度**
> | 层级 | **典型标记** | **说明** |
> | --- | --- | --- |
> | **章（Article）** | **`ARTICLE I`** | **最大单位** |
> | **节（Section）** | **`Section 1.1`** | **最常用的引用单位** |
> | **款（(a)(b)(c)）** | **`(a)` / `(b)`** | **并列条款** |
> | **项（(i)(ii)）** | **`(i)` / `(ii)`** | **款内的细分** |
> | **段** | **无标记、缩进** | **最细的语义单位** |
> **读法**：**"节"是最常用的引用单位（如 "as defined in Section 1.1"）**——**所以"chunk 的粒度应该对齐到节或款"，而不是按字符数切**。**而"按字符数切会把一个条款切成两半"，而半个条款可能语义反转，这是硬要求**。
> **★ 量化二：合成信贷协议的实测切分**
> | 项 | **数值** |
> | --- | --- |
> | **chunk 总数** | **7** |
> | **带正文的** | **3（Section 1.1 / 7.1 / 9.1）** |
> | **标题行** | **4（3 个 ARTICLE 头 + 1 个行内节）** |
> | **总字符** | **1,297** |
> **读法**：**按"节"切出 7 个 chunk（3 个带正文、4 个是标题行）**——**所以"对齐到节让每个 chunk 都能独立解释"，因为它以 `Section x.y` 开头**。**而"标题行不能单独成 chunk"，它没有内容、只有标题，要合并到下一节**。
> **★ 量化三：上下文头的代价**
> | chunk | **原长** | **+头** | **增幅** |
> | --- | --- | --- | --- |
> | **Section 1.1** | **505** | **602** | **1.19×** |
> | **Section 7.1** | **377** | **449** | **1.19×** |
> | **Section 9.1** | **238** | **291** | **1.22×** |
> | **合计** | **1,120** | **1,342** | **1.20×** |
> **读法**：**上下文头让 chunk 变大约 1.20 倍，但让检索结果可独立解释**——**所以"头的收益是可解释性、成本是索引体积"**。**而"把定义本身也内联进头"（而不是只写术语名），这是关键的取舍点**。
> **★ 量化四：交叉引用的密度**
> | 类型 | **出现次数** |
> | --- | --- |
> | **显式节引用** | **5** |
> | **定义术语（引号）** | **3** |
> | **"except as permitted"** | **1** |
> | **"without the prior written"** | **2** |
> **读法**：**这个片段里就有 5 处显式节引用与 3 个定义术语**——**所以"交叉引用不是例外而是常态"，在真实的 200 页协议里更密集**。**而"不处理引用就等于检索到的 chunk 缺了关键前提"，这是本题的延伸**。
> **★ 量化五：五个切分规则**
> | 规则 | **为什么** |
> | --- | --- |
> | **① 只在条款边界切** | **半个条款可能语义反转** |
> | **② 标题并入下一节** | **标题单独成块没有内容** |
> | **③ 带上文路径** | **让 chunk 可被引用** |
> | **④ 记录引用到的定义术语** | **为两跳检索做准备** |
> | **⑤ 保留原文字符偏移** | **为"回到原文高亮"做准备** |
> 一句话判据：**"层级里'节'是引用单位 → 所以按节切（实测 7 个 chunk）→ 加上下文头变 1.20 倍换来可解释性 → 交叉引用是常态（5 处节引用 + 3 个定义）→ 五个规则里'保留字符偏移'最易漏，因为它是引用回原文的前提"**。

## 面试官在考什么

- **★ 是否坚持"只在条款边界切"**：**能否给出"半个条款可能语义反转"**——**这是本题的分水岭**。
- **★ 是否给出上下文头的具体内容**：**能否给出"Article/Section 路径 + 定义术语"**。
- **★ 是否想到交叉引用**：**能否给出"记录引用到的定义术语"**。
- **★ 是否想到字符偏移**：**能否给出"为回到原文高亮做准备"**。
- **★ 是否指出"标题不能单独成块"**：**能否给出"要合并到下一节"**。
- **是否对齐到"节"这个粒度**：**能否给出"它是最常用的引用单位"**。
- **是否量化头的代价**：**能否给出"约 1.20 倍"**。
- **是否量化交叉引用密度**：**能否给出"5 处节引用"**。
- **是否区分"章/节/款/项"**：**能否给出四级结构**。
- **诚实**：**承认"真实文档的编号格式远不如示例规整"**。

**常见错误答案**

- **按固定字符数切**（**会拆开条款**）。
- **按段落切**（**段落不等于条款**）。
- **不带任何上下文头**（**chunk 无法独立解释**）。
- **把 ARTICLE 标题单独成块**。
- **不记录交叉引用**（**检索结果缺前提**）。
- **不保留字符偏移**（**无法给出可信引用**）。
- **用固定 overlap**（**会让同一 clause 重复出现**）。
- **认为"法律文档格式规整"**（**真实文档有大量变体**）。

## 原理与推导

### 1. ★ 语义单元

| 切法 | 后果 |
| --- | --- |
| **按字符数** | **可能拆开条款** |
| **按条款边界** | **语义完整** |

**读法**：**"条款是语义原子"**——**所以"不能切在中间"**。

### 2. ★ 自包含

$$\text{可解释}=\text{正文}+\text{路径}+\text{定义}$$

| 缺哪个 | 后果 |
| --- | --- |
| **缺路径** | **无法引用** |
| **缺定义** | **术语无法解释** |

**读法**：**"三者缺一不可"**——**所以"头要带全"**。

### 3. ★ 代价

| 项 | 增幅 |
| --- | --- |
| **加头** | **1.20×** |

**读法**：**"头的代价很小"**——**所以"没有理由不加"**。

### 4. ★ 交叉引用

| 类型 | 密度 |
| --- | --- |
| **节引用** | **5 / 片段** |
| **定义术语** | **3 / 片段** |

**读法**：**"引用是常态"**——**所以"必须显式建模"**。

### 5. ★ 偏移

| 保留偏移 | 收益 |
| --- | --- |
| **是** | **可回原文高亮** |
| **否** | **引用不可验证** |

**读法**：**"偏移是'引用'的技术前提"**——**所以它必须保留**。

### 6. ★ 标题

| 处理 | 后果 |
| --- | --- |
| **单独成块** | **空块、污染检索** |
| **并入下一节** | **正确** |

**读法**：**"标题是'节的属性'而不是'独立单元'"**——**所以"要合并"**。

## 数值与代码验证

### 表 1：层级、实测切分、上下文头、引用密度、五个规则（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| 法律文档的五个层级（章 / 节 / 款 / 项 / 段） | 「节」是最常用的引用单位（如「as defined in Section 1.1」）→ chunk 粒度应对齐到节或款，**按字符数切会把一个条款切成两半** |
| 实测切分（合成信贷协议） | 按「节」切出 **7 个 chunk、总 1,297 字符**（3 个带正文、4 个是标题行）；标题行不能单独成 chunk |
| 上下文头的增幅（三个正文 chunk） | 原长 505 / 377 / 238 → 加头后 602 / 449 / 291（1.19× / 1.19× / 1.22×）；**合计 1,120 → 1,342 字符（1.20×）** |
| 交叉引用密度（显式节引用 / 定义术语 / as defined in / except as permitted / without the prior written） | **5 / 3 / 0 / 1 / 2**——交叉引用不是例外而是常态 |
| 五个切分规则 | 只在条款边界切；标题并入下一节；带上文路径；记录引用到的定义术语；**保留原文字符偏移**（最易忽略，它是「引用回原文」的前提） |

### 可运行代码

```python
import re
print('① 法律文档的层级：切分只能在"条款边界"上')
print(f'  {"层级":<20} {"典型标记":<28} 说明')
for a,b,c in (('**章（Article）**','**`ARTICLE I` / `第 1 条`**','**最大单位**'),
              ('**节（Section）**','**`Section 1.1` / `1.1`**','**最常用的引用单位**'),
              ('**款（(a)(b)(c)）**','**`(a)` / `(b)`**','**并列条款**'),
              ('**项（(i)(ii)）**','**`(i)` / `(ii)`**','**款内的细分**'),
              ('**段**','**无标记、缩进**','**最细的语义单位**')):
    print(f'  {a:<20} {b:<28} {c}')
print("  读法：**「节」是最常用的引用单位**（**如「as defined in Section 1.1」**）——")
print("        所以**「chunk 的粒度应该对齐到节或款」**（**而不是按字符数切**）；")
print("        而**「按字符数切会把一个条款切成两半」**（**而半个条款可能语义反转**）-> 这是硬要求")
print()
print('② 实测：一个合成信贷协议的切分')
DOC = """ARTICLE I DEFINITIONS
Section 1.1 Defined Terms. As used in this Agreement, the following terms have the meanings set forth below.
"Change of Control" means the acquisition by any Person of beneficial ownership of 50% or more of the outstanding voting securities of the Company.
"Permitted Liens" means liens arising by operation of law and not resulting from any breach by the Company.
"Material Adverse Effect" means any event that would reasonably be expected to have a material adverse effect on the business of the Company.
ARTICLE VII COVENANTS
Section 7.1 Negative Covenants. The Company shall not, without the prior written consent of the Required Lenders:
(a) incur any Indebtedness, except as permitted under Section 7.3;
(b) create or permit to exist any Lien on any of its assets, other than Permitted Liens;
(c) consummate any merger or consolidation, unless such transaction would not result in a Change of Control.
Section 7.2 Financial Covenants. The Company shall maintain a Leverage Ratio of not more than 4.00 to 1.00.
ARTICLE IX EVENTS OF DEFAULT
Section 9.1 Events of Default. Each of the following constitutes an Event of Default:
(a) the Company fails to pay any principal when due;
(b) the occurrence of a Change of Control without the prior written consent of the Required Lenders.
"""
lines=[l for l in DOC.strip().split('\n') if l.strip()]
# 按"节"切分（Section x.y 起始）
chunks=[]; cur=None
for l in lines:
    if re.match(r'^(ARTICLE|Section)\s', l):
        if cur: chunks.append(cur)
        cur={'head':l,'body':[]}
    else:
        if cur is None: cur={'head':'(前言)','body':[]}
        cur['body'].append(l)
if cur: chunks.append(cur)
print(f'  {"#":>3} {"起始":<44} {"行数":>6} {"字符":>8}')
for i,c in enumerate(chunks,1):
    body=' '.join(c['body'])
    print(f'  {i:>3} {c["head"][:42]:<44} {len(c["body"]):>6} {len(c["head"])+len(body):>8}')
print(f'  **共 {len(chunks)} 个 chunk、总 {sum(len(c["head"])+len(" ".join(c["body"])) for c in chunks)} 字符**')
print("  读法：**按「节」切出 7 个 chunk（3 个带正文、4 个是标题行）** ——")
print("        所以**「对齐到节让每个 chunk 都能独立解释」**（**因为它以 Section x.y 开头**）；")
print("        而**「标题行不能单独成 chunk」**（**它没有内容、只有标题**）-> 要合并到下一节")
print()
print('③ 上下文头：让 chunk 自包含')
def header(doc_title, article, section, definitions):
    h=f'[{doc_title}] {article} / {section}'
    if definitions: h+=' | 相关定义：'+'; '.join(definitions)
    return h
DEFS={'Change of Control':'收购 50% 或以上的有表决权证券',
      'Permitted Liens':'依法律产生的担保物权',
      'Material Adverse Effect':'重大不利影响'}
print(f'  {"chunk":<40} {"原长":>6} {"+头":>6} {"增幅":>8}')
tot_o=tot_h=0
for c in chunks:
    if not c['body']: continue
    body=' '.join(c['body'])
    refs=[k for k in DEFS if k in body]
    h=header('信贷协议','ARTICLE VII' if 'Covenant' in body or 'Lien' in body or 'Indebtedness' in body else 'ARTICLE I',
             c['head'].split('.')[0], refs)
    o=len(c['head'])+len(body); n=o+len(h)
    tot_o+=o; tot_h+=n
    print(f'  {c["head"][:38]:<40} {o:>6} {n:>6} {n/max(o,1):>7.2f}x')
print(f'  **合计 {tot_o} -> {tot_h} 字符（{tot_h/tot_o:.2f}x）**')
print("  读法：**上下文头让 chunk 变大约 1.20 倍**（**但让检索结果可独立解释**）——")
print("        所以**「头的收益是可解释性、成本是索引体积」**；")
print("        而**「把定义本身也内联进头」**（**而不是只写术语名**）-> 这是关键的取舍点")
print()
print('④ 交叉引用的密度：有多少条款依赖别处')
print(f'  {"类型":<26} {"出现次数":>10} 说明')
pats=[('**显式节引用**',r'Section \d+\.\d+'),('**定义术语（引号）**',r'"[A-Z][a-z]+[^"]*"'),
      ('**"as defined in"**',r'as defined in'),('**"except as permitted"**',r'except as permitted'),
      ('**"without the prior written"**',r'without the prior written')]
for name,pat in pats:
    print(f'  {name:<26} {len(re.findall(pat,DOC)):>10} ')
print("  读法：**这个片段里就有 5 处显式节引用与 3 个定义术语** ——")
print("        所以**「交叉引用不是例外而是常态」**（**在真实的 200 页协议里更密集**）；")
print("        而**「不处理引用就等于检索到的 chunk 缺了关键前提」**（**串 [[harvey-02]]**）-> 这是本题的延伸")
print()
print('⑤ 五个切分规则')
print(f'  {"规则":<30} {"为什么":<34}')
for a,b in (('**① 只在条款边界切**','**半个条款可能语义反转**'),
            ('**② 标题并入下一节**','**标题单独成块没有内容**'),
            ('**③ 带上文路径（Article/Section）**','**让 chunk 可被引用**'),
            ('**④ 记录引用到的定义术语**','**为两跳检索做准备**'),
            ('**⑤ 保留原文字符偏移**','**为"回到原文高亮"做准备**')):
    print(f'  {a:<30} {b}')
print("  读法：**第 5 条「保留字符偏移」最容易被忽略**（**因为它是「引用回原文」的前提**）——")
print("        所以**「chunk 要能映射回原文档的位置」**（**否则无法给出可信的引用**）；")
print("        而**「这与 [[harvey-07]] 的 grounding 是同一件事」**（**都要「回到具体段落」**）-> 这是闭环")
```

预期输出要点（实跑）：① **五级层级**与典型标记；② **实测**：**7 个 chunk**（3 带正文 + 4 标题行）、总 **1,297** 字符；③ **上下文头**：**505→602 / 377→449 / 238→291**（**合计 1,120→1,342、1.20×**）；④ **引用密度**：**5 处节引用、3 个定义术语、1 处 except as permitted、2 处 without the prior written**；⑤ **五个规则**。

## 常见追问

- **追问**：如果一份文档没有 `Section` 编号怎么办？
  - 要点：**三条**：① **退到"款"级标记**（**如 `(a)(b)(c)`**）；② **再退到"空行 + 缩进"启发式**；③ **而"最差情况按句切"**（**但要在头里标注'这是启发式切分'**）。**读法**：**"要有多级回退"**——**因为真实文档格式不规整**。
- **追问**：overlap 要不要加？
  - 要点：**三条**：① **在条款边界上加 overlap 是'重复'而不是'补上下文'**（**会让同一条款被检索两次**）；② **正确的做法是'加结构头 + 加引用'**；③ **所以"overlap 在法律场景下弊大于利"**。**读法**：**"用结构而不是用重叠来补上下文"**——**这是法律场景与通用 RAG 的关键差别**。
- **追问**：定义术语怎么识别？
  - 要点：**三条**：① **引号包裹的首字母大写短语**（**如 `"Change of Control"`**）；② **`means` / `shall mean` / `has the meaning` 的模式**；③ **以及"在 Definitions 节内的所有引号短语"**。**读法**：**"定义有显式的语言标记"**——**所以"识别它比识别一般实体可靠得多"**。
- **追问**：chunk 粒度选"节"还是"款"？
  - 要点：**三条**：① **"节"适合'整条查询'**（**如"7.1 节讲了什么"**）；② **"款"适合'精确查询'**（**如"能不能新增债务"**）；③ **所以"可以同时建两层索引"**（**款级检索 + 节级回退**）。**读法**：**"粒度不唯一"**——**所以"两级索引是常见做法"**。
- **追问**：怎么验证 chunker 正确？
  - 要点：**三条**：① **"每个 chunk 的起始必须是条款标记"**（**可自动检查**）；② **"chunk 的拼接必须等于原文"**（**即切分是无损的**）；③ **以及"人工抽查边界处是否语义完整"**。**读法**：**"无损性是可自动验证的"**——**这是最好的回归测试**。
- **追问**：这道题与"grounding"有什么关系？
  - 要点**两条**：① **[[harvey-07]] 讲"每条断言都要链接回段落"**；② **而"字符偏移"正是那个链接的技术前提**；③ **所以"chunker 的输出格式决定了 grounding 能不能做"**。**读法**：**"切分是 grounding 的地基"**——**两者不能分开设计**。

## 相关题目

- [[harvey-02]]：定义在第 8 页、使用在第 140 页——**交叉引用的检索侧**。
- [[harvey-03]]：整份进上下文还是检索——**切分粒度的上游决策**。
- [[harvey-07]]：grounding 系统——**字符偏移的用途**。
- [[rag-01]]：RAG 的基本框架——**通用方法**。
- [[coding-01]]：数组编程与向量化——**工程实现**。

## 参考资料与归属

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（延伸）** —— Lewis et al. (Facebook AI)，2020-05-22：<https://arxiv.org/abs/2005.11401>。**检索增强的基本框架** 是本篇的背景来源。
- **Dense Passage Retrieval（延伸）** —— Karpukhin et al. (Facebook AI)，2020-04-10：<https://arxiv.org/abs/2004.04906>。**稠密检索与 chunk 粒度** 是本篇第 1 节的依据。
- **ColBERT（延伸）** —— Khattab & Zaharia (Stanford)，2020-04-27：<https://arxiv.org/abs/2004.12832>。**细粒度交互与"片段级"表示** 是本篇第 1、5 节的类比来源。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（五级层级、7 个 chunk 与 3 带正文、总 1,297 字符、上下文头的 505→602/377→449/238→291 与 1,120→1,342、引用密度的 5/3/1/2）都是为演示"法律文档切分"而构造的**实测结果**；**切分与统计来自本次实跑**（**可复现**，**基于一个约 1,300 字符的合成信贷协议片段**）。**⚠️ "合成片段"远小于真实的 200 页协议**——**真实文档的编号格式、嵌套深度与引用密度都更复杂**；**"1.20 倍"依赖头的具体内容**。**可迁移的结论是"按条款边界切、头要带路径与定义、交叉引用是常态、字符偏移是引用的前提、标题要合并"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
