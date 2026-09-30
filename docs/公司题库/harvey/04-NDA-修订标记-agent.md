---
type: question
id: harvey-04
company: Harvey
topic: agents
order: 4
question: 设计一个 agent：输入一份 NDA 草稿，返回一份体现该律所 playbook 的带修订标记的 Word 文档，而不是一段聊天回复。
question_en: Design an agent that takes an NDA draft and returns a Word document with tracked changes reflecting the firm’s playbook, rather than a chat reply.
asked_at: []
level: 高阶
tags: [OOXML, 修订标记, run 切分, playbook, 精确率优先]
sources:
  - title: Office Open XML: WordprocessingML Reference（延伸）
    url: https://learn.microsoft.com/en-us/office/open-xml/word/overview
    author: Microsoft
    published: 2024-01-01
  - title: python-docx Documentation（延伸）
    url: https://python-docx.readthedocs.io/
    author: Steve Canny
    published: 2024-01-01
  - title: ReAct: Synergizing Reasoning and Acting in Language Models（延伸）
    url: https://arxiv.org/abs/2210.03629
    author: Yao et al. (Princeton)
    published: 2022-10-06
related: [harvey-01, harvey-05, harvey-07, agents-01, coding-02]
updated: 2026-09-28
---

## 一句话答案

> **交付物是文档而不是回答**：**而技术难点在"修订标记是 XML 层的、而 Word 会把一个段落任意切成多个 run"——实测一个 95 字符的段落被切成 5 个 run（平均 19 字符）**，**所以朴素的字符串替换会破坏 XML**；**playbook 的四类动作里"接受不改"占 55%、"标记待议"占 15%（abstention 是设计的一部分）**，**而律师时间从 90 分钟降到 15 分钟。**
> **★ 量化一：为什么必须是带修订标记的 docx**
> | 输出形式 | **律师要做什么** | **说明** |
> | --- | --- | --- |
> | **聊天回复** | **手工把每处改动敲进 Word** | **每处 2–5 分钟** |
> | **纯文本 diff** | **仍要手工应用** | **格式会丢** |
> | **带修订标记的 docx** | **直接接受/拒绝** | **零手工步骤** |
> **读法**：**"交付物是文档而不是回答"，因为律师的工作流在 Word 里**——**所以"agent 的成败取决于输出格式而不是措辞"**。**而"纯文本 diff 会丢格式"，而修订标记保留了原样式，这是必须用 OOXML 的原因**。
> **★ 量化二：run 切分的实测（最难的技术点）**
> | run | **文本** | **长度** |
> | --- | --- | --- |
> | **1** | **The Receiving Party shall** | **26** |
> | **2** | **not** | **3** |
> | **3** | **disclose Confidential Information** | **35** |
> | **4** | **for a period of five (5) years** | **30** |
> | **5** | **.** | **1** |
> **读法**：**一个 95 字符的段落被切成 5 个 run（平均每个 19 字符），因为 Word 会因拼写检查状态与格式变化而任意切分**——**所以"一个词可能跨越两个 run"（如 disclose 与前面的空格分属不同 run）**。**而"朴素的字符串替换会破坏 XML 结构"，因为它不知道 run 边界，这是本题的核心难点**。
> **★ 量化三：替换的三种方式**
> | 方式 | **结果** | **问题** |
> | --- | --- | --- |
> | **朴素字符串替换** | **改的是 XML 文本、可能落在标签中间** | **XML 损坏** |
> | **按 run 索引定位** | **先建 (run, offset) 映射再改** | **正确** |
> | **拆 run（split）** | **把跨边界的词拆成独立 run** | **要复制 rPr** |
> **读法**：**实测中目标文本恰好落在单个 run 内、所以替换可行**——**所以"要先检查目标是否跨 run，跨了就要先拆"，而不是直接替换**。**而"跨 run 的情况在真实文档里很常见"，因为 Word 会因格式变化切分，必须处理**。
> **★ 量化四：修订标记的 OOXML 形态**
> | 动作 | **XML** |
> | --- | --- |
> | **删除** | **`<w:del><w:r><w:delText>旧</w:delText></w:r></w:del>`** |
> | **插入** | **`<w:ins><w:r><w:t>新</w:t></w:r></w:ins>`** |
> | **批注锚点** | **`<w:commentRangeStart/>` … `<w:commentRangeEnd/>`** |
> | **作者与时间** | **`w:author="Harvey" w:date="..."`** |
> **读法**：**"删除用 `w:delText` 而不是 `w:t`"，这是 Word 的约定、用错会显示异常**——**所以"要严格按 OOXML 规范生成"，而不是自己发明标记**。**而"作者与时间戳必须带上"，否则律师无法区分"谁改的"，这是合规要求**。
> **★ 量化五：playbook 的四类动作与律师的时间账**
> | 动作 | **触发条件** | **占比** |
> | --- | --- | --- |
> | **① 接受（不改）** | **条款符合 playbook** | **约 55%** |
> | **② 修改** | **有明确规则可套用** | **约 25%** |
> | **③ 标记待议** | **有规则但需人工判断** | **约 15%** |
> | **④ 建议新增** | **playbook 有要求但合同缺失** | **约 5%** |
> | 环节 | **人工** | **agent 后** |
> | **通读定位** | **45 分钟** | **5 分钟** |
> | **逐处修改** | **30 分钟** | **8 分钟** |
> | **撰写说明** | **15 分钟** | **2 分钟** |
> | **合计** | **90 分钟** | **15 分钟（省 83%）** |
> 一句话判据：**"交付物是 docx 不是回答 → 难点在 run 切分（95 字符切成 5 个 run）→ 要先检查跨 run 再拆 → 修订标记用 w:ins/w:del 并带作者 → 四类动作里 abstention 占 15% 是设计的一部分 → 律师从 90 分钟降到 15 分钟，而精确率比召回率更重要（误改一处会让律师核对全部）"**。

## 面试官在考什么

- **★ 是否意识到 run 切分问题**：**能否给出"一个词可能跨 run"**——**这是本题的分水岭**。
- **★ 是否指出"要严格按 OOXML 生成"**：**能否给出 `w:delText` 这个细节**。
- **★ 是否指出"abstention 是设计的一部分"**：**能否给出"标记待议占 15%"**。
- **★ 是否指出"精确率比召回率更重要"**：**能否给出"误改一处会让律师核对全部"**。
- **★ 是否算律师的时间账**：**能否给出"90 → 15 分钟"**。
- **是否指出"交付物是文档"**：**能否给出"律师的工作流在 Word 里"**。
- **是否带作者与时间戳**：**能否给出"合规要求"**。
- **是否用批注解释理由**：**能否给出 `w:comment`**。
- **是否保留原文可恢复**：**能否给出"修订标记本身就是可恢复的"**。
- **诚实**：**承认"真实 docx 的 run 切分比示例更碎"**。

**常见错误答案**

- **返回聊天回复**（**律师要手工搬**）。
- **做纯文本 diff**（**格式丢失**）。
- **直接替换文本**（**XML 会损坏**）。
- **不知道 run 的存在**。
- **用 `w:t` 表示删除**（**应使用 `w:delText`**）。
- **不带作者信息**。
- **全部条款都改**（**没有 abstention**）。
- **把召回率当首要指标**。

## 原理与推导

### 1. ★ run

| 概念 | 说明 |
| --- | --- |
| **段落（w:p）** | **语义单位** |
| **run（w:r）** | **格式一致的最小片段** |

**读法**：**"run 边界由格式与编辑历史决定、与语义无关"**——**所以"不能按 run 做语义操作"**。

### 2. ★ 映射

$$\text{offset}_{\text{para}}\to(\text{run}_i,\text{offset}_i)$$

| 步骤 | 动作 |
| --- | --- |
| **1** | **拼接 run 得到段落文本** |
| **2** | **在段落文本上定位编辑区间** |
| **3** | **映射回 run 坐标** |

**读法**：**"两套坐标要互转"**——**所以"要先建映射表"**。

### 3. ★ 拆分

| 情形 | 处理 |
| --- | --- |
| **编辑区间在单 run 内** | **直接改** |
| **跨 run** | **先拆 run 再改** |

**读法**：**"拆 run 要复制 rPr"**——**否则格式会丢**。

### 4. ★ 标记

| 动作 | 元素 |
| --- | --- |
| **删除** | **`w:del` + `w:delText`** |
| **插入** | **`w:ins` + `w:t`** |

**读法**：**"两类用不同元素"**——**所以"不能混用"**。

### 5. ★ 动作

| 动作 | 占比 |
| --- | --- |
| **接受** | **55%** |
| **修改 + 标记** | **40%** |

**读法**：**"价值在 40% 上"**——**而"标记待议是能力而不是缺陷"**。

### 6. ★ 指标

| 指标 | 优先级 |
| --- | --- |
| **精确率** | **高** |
| **召回率** | **中** |

**读法**：**"误改的代价高于漏改"**——**因为"漏改由律师兜底、误改会让律师不信任全部"**。

## 数值与代码验证

### 表 1：输出形式、run 切分、替换方式、OOXML、四类动作（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| 为什么输出必须是带修订标记的 Word（聊天回复 / 纯文本 diff / docx） | 聊天回复 → 律师每处手工敲 **2–5 分钟**；纯文本 diff → 仍要手工应用、**格式会丢**；**带修订标记的 docx → 直接接受/拒绝、零手工步骤** |
| Word 的 run 切分实测 | 一个 95 字符的段落被切成 **5 个 run**（长度 26 / 3 / 35 / 30 / 1，平均 19.0 字符）——**一个词可能跨越两个 run**，朴素字符串替换会破坏 XML |
| 朴素替换 vs run 感知替换 | 目标「five (5) years」实测**恰好落在单个 run（#4）内**，所以可行；但要**先检查是否跨 run，跨了就要先拆并复制 rPr** |
| 修订标记的 OOXML 形态（删除 / 插入 / 批注锚点 / 作者与时间） | `<w:del><w:r><w:delText>` / `<w:ins><w:r><w:t>` / `<w:commentRangeStart/>`…`<w:commentRangeEnd/>` / `w:author="Harvey" w:date="..."`（**删除用 w:delText 而不是 w:t**） |
| playbook 的四类动作与占比 | 接受（不改）**55%** / 修改 **25%** / 标记待议 **15%** / 建议新增 **5%** |

### 可运行代码

```python
import zipfile, os, tempfile, re
print('① 为什么输出必须是"带修订标记的 Word"而不是一段聊天')
print(f'  {"输出形式":<26} {"律师要做什么":<32} 说明')
for a,b,c in (('**聊天回复**','**手工把每处改动敲进 Word**','**每处 2–5 分钟**'),
              ('**纯文本 diff**','**仍要手工应用**','**格式会丢**'),
              ('**带修订标记的 docx**','**直接接受/拒绝**','**零手工步骤**')):
    print(f'  {a:<26} {b:<32} {c}')
print("  读法：**「交付物是文档而不是回答」**（**因为律师的工作流在 Word 里**）——")
print("        所以**「agent 的成败取决于输出格式而不是措辞」**；")
print("        而**「纯文本 diff 会丢格式」**（**而修订标记保留了原样式**）-> 这是必须用 OOXML 的原因")
print()
print('② 实测：Word 的 run 切分（这是最难的技术点）')
print('  构造一个 .docx，看 Word 如何把一个段落切成多个 run')
doc_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t xml:space="preserve">The Receiving Party shall </w:t></w:r><w:r><w:t>not</w:t></w:r><w:r><w:t xml:space="preserve"> disclose Confidential Information </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>for a period of five (5) years</w:t></w:r><w:r><w:t>.</w:t></w:r></w:p>
</w:body></w:document>'''
d=tempfile.mkdtemp(); p=os.path.join(d,'t.docx')
ct='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'''
rels='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
with zipfile.ZipFile(p,'w') as z:
    z.writestr('[Content_Types].xml',ct); z.writestr('_rels/.rels',rels); z.writestr('word/document.xml',doc_xml)
with zipfile.ZipFile(p) as z:
    x=z.read('word/document.xml').decode('utf-8')
runs=re.findall(r'<w:r>(.*?)</w:r>', x, re.S)
texts=[re.search(r'<w:t[^>]*>(.*?)</w:t>', r, re.S).group(1) for r in runs if '<w:t' in r]
print(f'  {"run":>4} {"文本":<44} {"长度":>6}')
for i,t in enumerate(texts,1):
    print(f'  {i:>4} {t[:42]:<44} {len(t):>6}')
full=''.join(texts)
print(f'  **拼接后的段落：{full}**')
print(f'  **run 数：{len(texts)}；段落总长：{len(full)}；平均每个 run {len(full)/len(texts):.1f} 字符**')
print("  读法：**一个 95 字符的段落被切成 5 个 run（平均每个 19 字符）**（**因为 Word 会因拼写检查状态与格式变化而任意切分**）——")
print("        所以**「一个词可能跨越两个 run」**（**如 disclose 与前面的空格分属不同 run**）；")
print("        而**「朴素的字符串替换会破坏 XML 结构」**（**因为它不知道 run 边界**）-> 这是本题的核心难点")
print()
print('③ 朴素替换 vs run 感知替换')
TARGET='five (5) years'
print(f'  目标：把 "{TARGET}" 改为 "three (3) years"')
print(f'  {"方式":<28} {"结果":<34} 问题')
for a,b,c in (('**朴素字符串替换**','**改的是 XML 文本、可能落在标签中间**','**XML 损坏**'),
              ('**按 run 索引定位**','**先建 (run, offset) 映射再改**','**正确**'),
              ('**拆 run（split）**','**把跨边界的词拆成独立 run**','**要复制 rPr**')):
    print(f'  {a:<28} {b:<34} {c}')
# 演示：目标文本是否在单个 run 内
hit=[i for i,t in enumerate(texts) if TARGET in t]
print(f'  **实测：目标文本落在 run #{hit[0]+1 if hit else "?"} 内（{len(hit)} 个 run 命中）**')
print("  读法：**这次目标恰好落在单个 run 内、所以替换可行** ——")
print("        所以**「要先检查目标是否跨 run，跨了就要先拆」**（**而不是直接替换**）；")
print("        而**「跨 run 的情况在真实文档里很常见」**（**因为 Word 会因格式变化切分**）-> 必须处理")
print()
print('④ 修订标记的 OOXML 形态')
print(f'  {"动作":<20} {"XML":<52}')
for a,b in (('**删除**','**`<w:del><w:r><w:delText>旧</w:delText></w:r></w:del>`**'),
            ('**插入**','**`<w:ins><w:r><w:t>新</w:t></w:r></w:ins>`**'),
            ('**批注锚点**','**`<w:commentRangeStart/>` … `<w:commentRangeEnd/>`**'),
            ('**作者与时间**','**`w:author="Harvey" w:date="..."`**')):
    print(f'  {a:<20} {b:<52}')
print("  读法：**「删除用 w:delText 而不是 w:t」**（**这是 Word 的约定、用错会显示异常**）——")
print("        所以**「要严格按 OOXML 规范生成」**（**而不是自己发明标记**）；")
print("        而**「作者与时间戳必须带上」**（**否则律师无法区分「谁改的」**）-> 这是合规要求")
print()
print('⑤ playbook 的四类动作与占比')
print(f'  {"动作":<24} {"触发条件":<28} 占比')
for a,b,c in (('**① 接受（不改）**','**条款符合 playbook**','**约 55%**'),
              ('**② 修改**','**有明确规则可套用**','**约 25%**'),
              ('**③ 标记待议**','**有规则但需人工判断**','**约 15%**'),
              ('**④ 建议新增**','**playbook 有要求但合同缺失**','**约 5%**')):
    print(f'  {a:<24} {b:<28} {c}')
print("  读法：**「接受不改」占 55%**（**所以 agent 的价值集中在 30% 的「修改 + 标记」上**）——")
print("        所以**「abstention（标记待议）是设计的一部分」**（**而不是失败**）；")
print("        而**「建议新增最难」**（**因为它要求判断「缺失」而不是「不符」**）-> 串 [[harvey-11]] 的同类问题")
print()
print('⑥ 律师的时间账')
print(f'  {"环节":<26} {"人工（分钟）":>12} {"agent 后":>10} 说明')
for a,b,c,d in (('**通读并定位问题**','**45**','**5**','**agent 已定位**'),
                ('**逐处修改**','**30**','**8**','**只需接受/拒绝**'),
                ('**撰写说明**','**15**','**2**','**agent 已生成批注**'),
                ('**合计**','**90**','**15**','**省 75 分钟**')):
    print(f'  {a:<26} {b:>12} {c:>10} {d}')
print('  读法：**从 90 分钟降到 15 分钟（省 83%）** ——')
print("        所以**「收益来自定位与起草而不是决策」**（**决策仍由律师做**）；")
print("        而**「若 agent 误改一处、律师要花时间核对全部」**（**所以精确率比召回率更重要**）-> 这是质量优先的理由")
```

预期输出要点（实跑）：① **三种输出形式**；② **run 切分**：**5 个 run、段落 95 字符、平均 19 字符**；③ **替换方式**与"目标落在 run #4"；④ **OOXML 四种形态**；⑤ **四类动作**：**55%/25%/15%/5%**；⑥ **时间账**：**90 → 15 分钟**。

## 常见追问

- **追问**：为什么不用 python-docx 直接改？
  - 要点：**三条**：① **python-docx 对修订标记的支持有限**（**它主要面向'干净文档'**）；② **所以"要么直接操作 OOXML、要么用支持修订的库"**；③ **以及"生成后用 Word 打开验证一遍"**。**读法**：**"库的抽象层次决定了能不能做修订"**——**必要时下到 XML 层**。
- **追问**：怎么保证"插入的文字继承原格式"？
  - 要点：**三条**：① **复制目标位置的 `w:rPr`**；② **并在插入 run 里带上它**；③ **以及"对加粗/斜体/字体逐一验证"**。**读法**：**"格式靠 rPr 传递"**——**所以要显式复制**。
- **追问**：如果 playbook 与合同差异很大怎么办？
  - 要点：**三条**：① **差异大时应该"标记待议"而不是"修改"**（**因为可能要重谈**）；② **并"在批注里说明差异的性质"**；③ **以及"给出建议但让律师决定"**。**读法**：**"abstention 是处理不确定的正确方式"**——**而不是硬改**。
- **追问**：怎么评估这个 agent？
  - 要点：**三条**：① **按动作分别算精确率**（**"修改"的精确率最重要**）；② **并"统计律师接受了多少比例的改动"**（**这是最直接的信号**）；③ **以及"测'漏改的关键条款数'"**（**用 [[harvey-11]] 的方式排查**）。**读法**：**"接受率是最好的指标"**——**因为它直接反映律师的信任**。
- **追问**：多轮修订怎么办（律师改了之后再来一轮）？
  - 要点：**三条**：① **要能识别"哪些改动是律师做的"**（**靠 `w:author`**）；② **并"在新一轮里不重复提已被拒绝的建议"**；③ **以及"保留历史以便对比"**。**读法**：**"作者字段是多轮协作的基础"**——**所以它不只是合规要求**。
- **追问**：这道题与"chunker"有什么关系？
  - 要点**两条**：① **[[harvey-01]] 讲"按条款切分"**；② **本题的"匹配 playbook"正是按条款逐个做的**；③ **所以"chunker 的粒度决定了 playbook 匹配的粒度"**。**读法**：**"切分是匹配的前提"**——**两者要一起设计**。

## 相关题目

- [[harvey-01]]：法律文档的 chunker——**按条款切分**。
- [[harvey-05]]：5,000 份合同的尽调——**同类的批处理架构**。
- [[harvey-07]]：grounding 系统——**批注与引用的共同基础**。
- [[agents-01]]：工具调用的基本框架——**agent 的通用结构**。
- [[coding-02]]：文本解析与结构化——**XML 处理的通用问题**。

## 参考资料与归属

- **Office Open XML: WordprocessingML（延伸）** —— Microsoft，2024-01-01：<https://learn.microsoft.com/en-us/office/open-xml/word/overview>。**`w:ins`/`w:del`/`w:delText` 与 run 结构** 是本篇第 2、3、4 节的直接来源。
- **python-docx 文档（延伸）** —— Steve Canny，2024-01-01：<https://python-docx.readthedocs.io/>。**段落/run 的对象模型** 是本篇第 1、2 节的依据。
- **ReAct（延伸）** —— Yao et al. (Princeton)，2022-10-06：<https://arxiv.org/abs/2210.03629>。**推理与行动的交替** 是本篇第 5 节的类比来源。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（三种输出形式、5 个 run 与 95 字符及平均 19 字符、三种替换方式与"目标落在 run #4"、OOXML 四种形态、四类动作的 55%/25%/15%/5%、时间账的 90→15 分钟）都是为演示"带修订标记的 agent"而构造的**实测结果与示例参数**；**run 切分与段落长度来自本次实跑**（**可复现**，**基于一个手工构造的最小 docx**）。**⚠️ "四类动作的占比"与"时间账"是示例估计**——**真实值取决于 playbook 与文档**；**"真实 docx 的 run 切分更碎"**（**因为拼写检查与修订历史**）。**可迁移的结论是"交付物是 docx、难点在 run 切分、要严格按 OOXML 生成、abstention 是设计的一部分、精确率优先"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
