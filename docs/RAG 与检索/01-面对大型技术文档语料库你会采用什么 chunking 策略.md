---
type: question
id: rag-01
topic: RAG 与检索
order: 1
question: 面对大型技术文档语料库，你会采用什么 chunking 策略，为什么？
question_en: What chunking strategy would you use for a large technical documentation corpus, and why?
asked_at: [Glean]
level: 进阶
tags: [chunking, 预处理, 检索, tokenizer]
sources:
  - title: Chunking Strategies for RAG
    url: https://outcomeschool.com/blog/chunking-strategies-for-rag
    author: Amit Shekhar (Outcome School)
    published: 
related: [rag-02, rag-09, rag-10, llm-internals-06]
updated: 2026-09-28
---

## 一句话答案

> chunk 同时是两个单位：向量库里参与相似度比较的最小单位，以及塞进 prompt 的上下文单位。技术文档语料上，先按文档自带的结构切——Markdown 标题层级 / HTML DOM / 代码的语法边界——把 `H1 > H2 > H3` 路径写进 chunk 头部一起 embedding；代码块、表格、API 参考条目作为不可切的原子块；超长小节再在内部按段落→句子递归回退到约 256–512 token、10%–20% overlap；长小节用 small-to-big，小块检索、父块投喂。
> 理由是两条失败路径都很硬：切太碎，chunk 丢了指代与前提，召回了也答不出；切太大，embedding 变成多个主题的平均，$k$ 个互不相关的主题混在一块时它与任一主题的相似度上界只有 $1/\sqrt{k}$（k=4 时 0.50）。chunking 决定的不是「预处理好不好看」，而是检索召回的上界——这个上界一旦定死，reranker 和更强的生成模型都补不回来。

## 面试官在考什么

- **是否把 chunk 当双重身份看**：只回答「512 token 最好」是背参数；能说清「检索侧要语义尖锐、生成侧要信息完整，两者方向相反」才算理解问题结构。
- **策略谱系是否成体系**：固定长度、递归字符切分、结构感知、语义切分、small-to-big 各自解决什么、在哪里失效，能不能按语料类型选型，而不是默认调用一个 splitter。
- **题干限定词「技术文档」是否接住**：标题路径进 chunk、代码块与表格的原子性、API 参考按「一个符号一块」、版本化文档的近重复去重——这几条是把通用答案落到技术文档语料的分水岭。
- **参数有没有口径**：以 token 而不是字符计量、overlap 的冗余系数怎么算、chunk 上限与 embedding 模型 max length 的关系。
- **是否把切分当实验对象**：固定检索器与生成器、只变切分器做 A/B，先看 recall@k / MRR / nDCG@k，再看端到端正确率（详见 [[rag-04]]）；并且知道改配置要全量重建索引，权限与时效元数据要落在 chunk 级（[[rag-08]]、[[rag-11]]）。

**常见错误答案**

- 「用 LangChain 的 `RecursiveCharacterTextSplitter`，chunk_size=1000、overlap=200。」——说不出 1000 是字符还是 token，也说不出这个数从哪来。1000 字符在英文技术文本里约 200 token，在中文里约 840 token（实测口径见第 4 节），同一套常数跨语言根本不是同一个粒度；而 1000 字符若超过所用 embedding 模型的 max length，尾部会被静默截断。
- 「先切小一点，反正有 reranker。」reranker 只能在召回的候选里重排：正确句子如果被切成没有主语的碎片、或者混在大块里被稀释到进不了候选集，重排无从下手；反过来只把块切大来「让模型看全」，是拿检索指标换投喂完整度。

## 原理与推导

### 两个单位，两个相反的要求

检索侧：向量库的一行就是一个 chunk，embedding 把整块压成一个点，所以「一块一个主题」是硬约束——语义混杂的块在向量空间里是一个平均值，离任何一个具体问题都远。生成侧：prompt 里放的就是 chunk（或它回填的父块），所以「一块自带回答问题所需的前提」也是硬约束。

两个约束方向相反：

- **切太碎**：指代与条件丢失。典型例子是一句「The limit was raised to 18 days, and the doctor note requirement was removed.」——它没有出现 sick leave，也没有出现任何小节名，单独入库后与「病假上限是多少」这类问题的相似度很低，永远排不进 top-k；即使被召回，模型也答不出这条改动属于哪个政策。
- **切太大**：一个块覆盖 $k$ 个主题，embedding 被稀释（下面推导），同时 top-k 里每一条的 token 数都很大，上下文预算被无关内容占掉，还会触发 lost-in-the-middle。

### 大 chunk 为什么检索不到：稀释的推导

设一个 chunk 由 $k$ 个主题片段拼接而成，第 $i$ 个片段的语义方向记为单位向量 $e_i$，两两余弦 $e_i \cdot e_j = \rho$（$i \neq j$）。前提是 embedding 已归一化，余弦即相似度。

**第一步**，把拼接后的 chunk 向量近似为各片段方向的等权平均。依据：embedding 对整段文本做 pooling，pooling 对「文本拼接」近似线性，所以「混在一起」在向量空间里就是「取平均」——这是玩具模型，只用来判断趋势：

$$
m = \frac{1}{k}\sum_{i=1}^{k} e_i
$$

**第二步**，算内积与模长（依据内积的线性性与 $\|e_i\| = 1$），再按定义归一化，得到该 chunk 与「纯主题 1」的相似度：

$$
m \cdot e_1 = \frac{1}{k}\left(1 + (k-1)\rho\right), \qquad
\|m\| = \frac{1}{k}\sqrt{k + k(k-1)\rho}, \qquad
\cos(\hat{m}, e_1) = \frac{m \cdot e_1}{\|m\|} = \frac{1 + (k-1)\rho}{\sqrt{k + k(k-1)\rho}}
$$

$\rho = 0$（主题互不相关）时退化为 $1/\sqrt{k}$：一个块里混进 4 个互不相关的主题，它与任一主题的相似度上界就从 1.00 掉到 0.50；混进 8 个是 0.354。top-k 排序里这种量级的差距是决定性的——库里任何一个语义干净的小块都能压过它。

反过来，$\rho$ 越大稀释越轻：$\rho = 0.3$、$k = 4$ 时是 0.689。这解释了工程上的经验边界——同一小节里的定义、参数、示例（语义高度相关）拼一块是安全的，跨小节甚至跨章节拼块不是。数值验证见第 4 节，闭式与数值构造的最大相对误差 $4.4\times10^{-16}$。

### 策略谱系与各自的失败模式

| 策略 | 切点依据 | 成本 | 适用 | 失败模式 |
| --- | --- | --- | --- | --- |
| 固定长度（按 token + overlap） | 数到 N 就切 | 极低 | 原型、无结构脏文本、作为兜底 | 切在句子/代码块/表格中间，切口不落在语义边界 |
| 递归字符切分 | 段落 → 换行 → 句号 → 空格逐级回退 | 低 | 通用默认 | 跟随文本形状而非语义：一段里两个主题会被绑在一起；按字符计时中英文口径不一致 |
| 结构感知 | Markdown 标题、HTML DOM、代码语法边界 | 低 | 有结构的文档（技术文档的主路径） | 无结构文本（转录、扫描件）不适用；小节长度极不均匀，超长小节仍需内部回退 |
| 语义切分 | 相邻句 embedding 相似度骤降处 | 高（逐句 embedding） | 无标题的连续文本、转录稿 | 阈值敏感（太灵敏→碎片，太钝→巨块）；单一主题长文没有切点 |
| small-to-big（父子文档） | 小块建索引、大块投喂 | 中 | 长小节、需要前后文才能答的问题 | 两套结构要维护；prompt 变长；多个子块命中同一父块时必须去重 |

还有两类常被提到：**contextual chunking**（给每块加一行上下文说明或标题路径）不是切分策略，而是能叠加在以上任何一种之上的增强，成本低、收益直接——「技术文档语料的具体做法」里的第 ① 条就是它的廉价版本；**agentic chunking**（让模型决定切点）质量上限最高，但一次要读全文、结果不可复现，只适合少量高价值的脏文档，不适合十万篇量级的技术文档语料。

### 技术文档语料的具体做法

题干限定的是「大型技术文档语料库」，通用策略要落到这几条上：

1. **按文档结构切，标题路径进 chunk 头部**。写入向量库的文本用 `H1 > H2 > H3\n正文`。理由有两个：块被单独取出后仍知道自己在讲什么，指代词有落点；标题本身是强检索词（`Timeout`、`Sick Leave`、`fanout`），参与 embedding 后等于免费加了关键词信号。
2. **代码块、表格、列表不可切断，宁可超长**。半个表格比没有表格更糟：切开的代码丢掉签名与缩进语义，切开的表格丢掉列头。做法是先识别围栏代码块与表格为原子块，只在原子块之间切；超长的原子块单独成块并标 `oversize`，不要硬切。
3. **API 参考类文档按「一个符号 / 一个接口一块」**。这类问题几乎都是单点的（某个参数默认值、某个返回码），答案的完整跨度就是该条目；把多个接口拼成一块会让 embedding 变成一堆符号名的平均，正是稀释推导里的情况。
4. **指南 / 教程类按小节切**。小节是作者给出的语义单元；小节超长时在小节内部按段落→句子回退，并且每一块都要带小节路径。
5. **标题层级链同时以两种形式存在**：进 embedding 文本（用于匹配），也进结构化字段 `heading_path`（用于过滤与加权，例如把检索限定在某个产品版本的子树内）。只做前者会丢掉可过滤性，只做后者会让标题词无法参与相似度计算。
6. **版本化文档的近重复要去重**。同一个页面在 v1.2 与 v2.0 上的正文高度重叠，两套块相似度极高，会一起挤进 top-k，把真正有区分度的候选挤掉——这种重复靠相似度阈值分不开，因为它们在语义上本来就几乎相同。做法：每块带 `doc_id`、`version`、内容指纹，索引只保留 canonical 版本（通常是当前稳定版），历史版本按查询需要过滤；入库前用 shingle / SimHash 一类内容指纹判重，而不是靠向量距离。

### 参数怎么定

- **以 token 计量，不用字符**。字符口径跨语言不可比：第 4 节实测英文技术文本 5.04 字符/token、中文 1.19 字符/token，同样「500 字符」在英文里约 99 token、中文里约 421 token，相差 4.3 倍。根源在 tokenizer 的分词方式（[[llm-internals-06]]）：字符与 token 的换算不是常数，代码、数字、非拉丁文字上还会更碎。
- **起步区间 256–512 token，overlap 10%–20%**。这是绝大多数 retrieval 向 embedding 模型 max length 内的安全区，同时兼顾「一块一个主题」与「块数不至于让索引爆掉」。它只是起点，不是结论。
- **上限由 embedding 模型的 max length 决定**。超过 max length 的部分会被静默截断，这部分文字永远不会进入向量，也永远不会被检索到——等于没入库。所以 chunk 上限 = max length − 标题路径与上下文前缀占用的 token 数。检索向模型的常见上限是 512（也有接受 8k 的），具体以所用模型的模型卡为准。
- **调整依据**（按优先级）：① 真实问题的长度分布与标注答案的跨度——答案平均跨两段就别把块切到一句；② 文档类型——FAQ / API 参考偏小，原理与流程偏大；③ embedding 模型 max length 与训练目标（模型见过多少长文本，决定长块里的位置细节能不能被保留）；④ 生成侧上下文预算，top-k 乘 chunk 要显著小于这个预算。
- **overlap 的代价可以算**。净新增 token 只有 $C - o$，所以冗余系数 $\text{冗余} = C/(C-o)$：$C = 512$、$o = 64$（12.5%）时是 1.143，向量与存储多 14.3%；$o = 102$（20%）时是 1.25，多 25%。更贵的是检索侧：相邻两块同时命中时，重复的那 $o$ 个 token 会一起进 prompt，白占预算，还可能让模型读到两遍同一句话。overlap 是保险，但保额很小就够了。

### small-to-big：把「检索粒度」和「投喂粒度」解耦

索引只存子块向量，每个子块记 `parent_id`；检索命中子块后，按 `parent_id` 去重，把父块（通常是整节，2–4 个子块的量级）投给模型。收益是不必再找那个「唯一正确的 chunk 大小」：召回靠小块，答案靠大块。代价是两套结构要维护、prompt 变长、以及必须做父块去重——多个子块命中同一父块时重复投喂会同时浪费预算并放大该节在 prompt 里的权重。

### 切分策略要用检索指标验证

固定 embedding 模型、索引参数与生成模型，只变切分器，在同一份带标注的问题集（问题 + 正确来源 span）上比较 recall@k、MRR、nDCG@k 与答案正确率（[[rag-04]]）。两个容易踩的坑：只看端到端分数——生成器会用上下文里的旁证补上检索的漏洞，于是「端到端不变」掩盖了检索变差，而 prompt 更长、成本更高；只看平均分——按答案跨度分层（单句答案 / 跨段答案 / 跨文档答案）看，策略差异往往只出现在其中一层，平均分把它抹平了。

## 数值与代码验证

本节的数字都是自行复算或运行得到的；源文没有给出任何具体基准数值，因此下面没有引用源文的数字，全部口径写在表下。

**表 1：token 与字符不是固定换算**

| 文本 | 字符数 | token 数 | 字符/token |
| --- | --- | --- | --- |
| 英文技术段落（自撰） | 403 | 80 | 5.04 |
| 中文技术段落（自撰） | 120 | 101 | 1.19 |
| 同样 500 字符预算 | 500 | 99 / 421 | — |

口径：`tiktoken` 的 `o200k_base`，两段自撰技术文本（内容是检索服务分片与权限过滤）。这只是代理口径——embedding 模型用的是自己的 tokenizer，绝对值会不同，但「中文每字符消耗的 token 是英文的 4 倍以上」这个量级关系成立。`o200k_base` 不是 embedding 词表，所以这两行只用于论证「不能用字符数当跨语言统一的预算」，不用于推算任何模型的成本。

**表 2：语料 → chunk 数 → 向量索引占用**

| chunk 大小 C | overlap o | chunk 数 | overlap 冗余 | 384d fp32 | 768d fp32 | 768d int8 | 1024d fp32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 32 | 345,451 | 1.143 | 506 MiB | 1,012 MiB | 253 MiB | 1,349 MiB |
| 512 | 64 | 172,725 | 1.143 | 253 MiB | 506 MiB | 127 MiB | 675 MiB |
| 1024 | 128 | 86,363 | 1.143 | 127 MiB | 253 MiB | 63 MiB | 337 MiB |

口径：语料 10,000 篇 × 6,000 英文词，样本实测 6.5 字符/词（含空格）与 5.04 字符/token，得 7,738 token/篇、77.4 M token；chunk 数 = 语料 token ÷ (C − o)；向量按 4 字节/维（fp32）或 1 字节/维（int8），MiB = $2^{20}$ 字节。注意这一行直接把英文的 token 密度套到整份语料上：如果语料里中文占大头，按 0.84 token/字符重算，同样字符数的 token 量会翻几倍，chunk 数与索引同步变大。

HNSW 的图结构也要算进来：M=16 时第 0 层每节点最多 2M 条邻接（2M×4 = 128 字节），上层每节点 M 条且节点落在上层的比例约 1/M，折算下来上层邻接总量约为第 0 层的 1/(M−1) = 1/15（约 4.3 字节），合计约 132 字节/节点，对应 172,725 个块约 22 MiB——是 768 维 fp32 向量（506 MiB）的 4.3%。结论：fp32 向量本身是主要开销，图结构是零头，但量化到 int8 把向量压到 127 MiB 之后，图结构的占比就上升到约 17%，不能再忽略。

**表 3：稀释闭式 $\cos(\hat{m}, e_1)$**

| 主题数 k | $\rho = 0$（互不相关） | $\rho = 0.3$（同一小节内） |
| --- | --- | --- |
| 1 | 1.000 | 1.000 |
| 2 | 0.707 | 0.806 |
| 4 | 0.500 | 0.689 |
| 8 | 0.354 | 0.622 |

口径：按稀释推导得到的闭式 $\frac{1+(k-1)\rho}{\sqrt{k+k(k-1)\rho}}$；构造 $k$ 个单位向量使其两两余弦精确等于 $\rho$（Gram 矩阵 $(1-\rho)I + \rho J$ 做 Cholesky 分解后乘正交基），实测 $\cos(\hat{m}, e_1)$ 与闭式的最大相对误差 $4.4\times10^{-16}$。

**表 4：父子块的 prompt 预算**

| 方案 | 命中数 | 送进 prompt 的 token |
| --- | --- | --- |
| 直接投喂子块 C=512 | top-5 | 2,560 |
| small-to-big，父块 = 4 子块 = 2048 | top-5（5 个不同父块） | 10,240 |
| small-to-big，5 个命中落在 2 个父块 | 去重后 2 个父块 | 4,096 |

口径：top-k = 5，忽略标题路径与前缀的少量开销。第三行说明父块去重不是优化项而是必需项：同一父块被多次命中时不去重，等于把该节重复送进 prompt。

**表 5：结构感知切分与固定长度切分在同一段 Markdown 上的对比**

```python
import re

HEADING = re.compile(r"^(#{1,6})\s+(.*)")
FENCE = re.compile(r"^\s*(```|~~~)")
SENT = re.compile(r"(?<=[。！？；.!?;])\s+")


def atomic_blocks(md):
    """切成 (标题路径, 文本, 段落号, 是否原子块)。代码围栏与表格整块保留。"""
    blocks, path, buf, fence, para = [], [], [], None, 0

    def flush():
        nonlocal para
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf.clear()
        if not text:
            return
        if text.startswith(("```", "~~~")) or text.lstrip().startswith("|"):
            blocks.append((list(path), text, para, True))        # 原子块：宁可超长
        else:
            for s in (x.strip() for x in SENT.split(text)):
                if s:
                    blocks.append((list(path), s, para, False))   # 段落 -> 句子：overlap 粒度
        para += 1

    for line in md.splitlines():
        if fence:                                                # 围栏内部不解析标题
            buf.append(line)
            if line.strip().startswith(fence):
                fence = None
            continue
        m = FENCE.match(line)
        if m:
            flush()
            fence = m.group(1)
            buf.append(line)
            continue
        h = HEADING.match(line)
        if h:
            flush()
            path[:] = path[: len(h.group(1)) - 1] + [h.group(2).strip()]
            continue
        if line.strip():
            buf.append(line)
        else:
            flush()                                              # 空行 = 段落边界
    flush()
    return blocks


def render(items):
    out = ""
    for i, (_, text, para, _) in enumerate(items):
        if i == 0:
            out = text
        else:
            out += (" " if para == items[i - 1][2] else "\n\n") + text
    return out


def chunk_markdown(md, max_tokens, overlap_tokens, ntokens):
    chunks, cur, cur_tokens = [], [], 0
    for item in atomic_blocks(md):
        path, text, _, atomic = item
        head = " > ".join(path)
        seg = f"{head}\n{text}" if head else text                # 标题路径进 embedding 文本
        n = ntokens(seg)
        if atomic and n > max_tokens:                            # 超长代码块/表格：单独成块
            if cur:
                chunks.append(render(cur))
            chunks.append(seg)
            cur, cur_tokens = [], 0
            continue
        if cur and cur_tokens + n > max_tokens:
            chunks.append(render(cur))
            keep, kept = [], 0                                   # 尾部按句回填 overlap
            for prev in reversed(cur):
                if kept + ntokens(prev[1]) > overlap_tokens:
                    break
                keep.insert(0, prev)
                kept += ntokens(prev[1])
            cur, cur_tokens = keep, kept
        cur.append(item)
        cur_tokens += n
    if cur:
        chunks.append(render(cur))
    return chunks
```

在 22 行 Markdown（一个 H1、两个 H2、一个 Python 围栏、一张三列表格）上跑，`ntokens` 用「5 字符 = 1 token」的占位口径、C=40、overlap=12：

```text
结构感知切分：4 块，围栏均衡 True
  [1] 17 tok  The gateway rewrites the query into sparse and dense forms, ...
  [2] 34 tok  ```python ... ```  ← 整块保留，不按预算切开
  [3] 19 tok  Filtering happens before the reranker sees a candidate. ...
  [4] 36 tok  A leaked document never reaches the model. / | stage | budget | ...
              ↑ [4] 的首句是 [3] 的尾部，这是 overlap 回填的证据
固定 200 字符切分：3 块，围栏均衡 False；第 2 块开头 '   futures = [s.sear'，第 3 块
  开头 'r reaches the model.' —— 围栏被一分为二，切口落在标识符与单词中间
```

口径：占位 `ntokens` 只用于让样例足够短，真实实现换成与 embedding 模型一致的 tokenizer。断言是围栏计数为偶数、只有超长原子块允许越界——把「字符串切片」换成结构感知切分时，这两条是最该测的部分。这段样例里表格侥幸留在一块内（切点没落在它上面）；换成长度 260 字符时，表头、分隔行与前两行数据落在第 2 块、最后一行单独落在第 3 块——表格不按原子块处理，完整性就只是运气。

## 常见追问

- **追问**：chunk 大小和 embedding 模型的训练长度、max length 是什么关系？
  - 要点：max length 是硬边界，超出部分被静默截断，那部分文字永远不会进入向量，也检索不到，所以 chunk 上限 = max length − 标题路径前缀。训练长度还决定模型对长文本的表征方式：只在 512 token 上训过的模型，喂 2000 token 的块，语义会被压成主题词级别的摘要，块内的位置细节（哪个参数对应哪个默认值）会丢。检索向模型的常见上限是 512，也有接受 8k 的，具体以模型卡为准。
- **追问**：overlap 会不会让答案重复出现、模型复读？
  - 要点：会，重复的正是那 $o$ 个 token。控制手段有四层：overlap 取 10%–20% 并按句子对齐（不要切在词中间）；检索后按内容指纹或 `parent_id` 去重；prompt 里保留来源标记并允许模型引用；生成侧明确要求不复述。overlap 的作用是防止答案正好横跨切口，不是让每个块都自带前文。
- **追问**：中文、多语言语料怎么切？
  - 要点：不要用英文的 `.!?` 规则，按中文标点（。！？；）与换行切，注意全角半角混排与无空格分词；技术文档常常中英混排（正文中文、符号名英文），规则要能同时处理。计量统一用 tokenizer 的预算：中文约 0.84 token/字符，是英文的 4 倍以上，同样 512 token 装的中文信息量明显少于英文，所以「中文用更小的 chunk」往往是必要的，而不是偏好。
- **追问**：chunk 的元数据里应该带什么？
  - 要点：`doc_id` 与版本（溯源、重建索引）、`heading_path`（结构化过滤与加权）、来源 URL 与页码（引用）、`acl` 权限标签（[[rag-08]]，必须在检索侧过滤，而且要落到 chunk 级——同一文档的不同小节可能对应不同权限）、时间戳（[[rag-11]]，时效性与增量更新）、内容指纹（去重）、语言（切分规则与分词器选择）。
- **追问**：改了 chunk 大小或切分器之后，还要做什么？
  - 要点：全量重建索引——块变了所有向量都过期，只重算新文档会留下两套粒度混在一起的索引；配置（大小、overlap、切分器版本、embedding 模型版本）写进索引元数据，便于回滚与 A/B。
- **追问**：怎么判断新的 chunk 策略更好？
  - 要点：固定检索器与生成器，只变切分器；先看 recall@k / MRR / nDCG@k，再看端到端答案正确率（[[rag-04]]）。常见情况是端到端分数不动而检索指标变了——说明生成器在补偿检索的缺陷，这时更该信检索指标，同时看 prompt 长度与成本的变化。

## 公司变体

`asked_at` 只标了 Glean。Glean 做的是企业搜索与助手，公开岗位（search / ranking、ML Engineer、AI Engineer、Infrastructure）和公开材料都指向**偏工程实现**的考察方式：数据源异构（Slack 会话串、Jira 工单、Google Docs、PDF 各有各的语义单位）、权限与增量更新是系统的一部分、评估体系要能跑在真实查询上。

所以这道题在 Glean 的语境里通常会从「你的策略叫什么名字」转向「在你的连接器上具体怎么落地」：Slack 会话串按 thread 切、Jira 工单按 issue 加最新评论、Docs / Wiki 按标题层级、PDF 按版式块并处理多栏；每块要挂权限标签与更新时间，索引要能增量更新（[[rag-08]]、[[rag-11]]）。上面那种稀释推导在这里更多是解释工具——用来论证「为什么不把整个工单连同历史评论塞一块」，而不是考点本身。具体轮次以实际面试为准，这里只谈侧重。

## 相关题目

- [[rag-02]]：稀疏（BM25）与稠密检索怎么选、何时混用。chunk 粒度直接决定 BM25 的文档长度归一化与稠密向量的语义锐度，两边的参数要一起调。
- [[rag-09]]：HNSW / IVF-PQ / flat 的取舍与 recall@k 的 latency 代价。第 4 节的索引内存表就是它的输入：块数由 chunk 大小决定，量化与图结构的占比也随粒度变化。
- [[rag-10]]：表格、图表、多栏 PDF 怎么处理。这是「代码块、表格不可切断」那一条的延伸——表格是原子块，但「不为文本的表格」需要单独的表征路径。
- [[llm-internals-06]]：BPE 如何工作、为什么字符与 token 的换算不是常数。「以 token 计量」这条依据来自这里。

## 参考资料与归属

1. [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)，Amit Shekhar（Outcome School），源站标注 2026-09-10。提供策略谱系（固定长度、按句、递归、文档结构、语义、contextual、small-to-big、agentic 及各自优缺点）、overlap 取约 10%–20% 的常见设置与「overlap 不是免费的」、按答案跨度选 chunk 大小的三类经验区间、embedding max length 之外文本被静默截断的结论、表格与代码块不可切断、以及改配置必须重建索引。「技术文档语料的具体做法」那六条（标题路径前缀、原子块、API 参考一块一符号、版本化文档去重）是把源文的通用做法落到技术文档语料后的推论，其中版本化文档去重与内容指纹判重属本仓库补充的工程做法。
2. 第 4 节的全部数字——token/字符实测、语料规模到 chunk 数与索引占用、HNSW 图结构占比、稀释闭式与数值构造的对照、父块 prompt 预算、切分器输出——均由本仓库自行复算或运行得到，口径写在各表下方；源文未提供任何可供对照的基准数值，故未引用源文数字。玩具模型（chunk 向量等于片段向量的等权平均）只用于判断趋势，不代表真实 embedding 的精确行为，这一点在稀释推导与表 3 中都已标注。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
