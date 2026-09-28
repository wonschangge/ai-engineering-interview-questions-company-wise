---
type: question
id: llm-internals-13
topic: LLM 内部原理与架构
order: 13
question: 长上下文中的 lost-in-the-middle 问题是什么？该如何解决？
question_en: What is the lost-in-the-middle problem in long contexts and how do you address it?
asked_at: [Moonshot AI]
level: 进阶
tags: [长上下文, rag, 位置偏差, 评测]
sources:
  - title: The Lost in the Middle Problem in LLMs
    url: https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-08-19
  - title: Lost in the Middle: How Language Models Use Long Contexts（延伸）
    url: https://arxiv.org/abs/2307.03172
    author: Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang
    published: 2023-07-06
  - title: RULER: What's the Real Context Size of Your Long-Context Language Models?（延伸）
    url: https://arxiv.org/abs/2404.06654
    author: Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg
    published: 2024-04-09
  - title: MoBA: Mixture of Block Attention for Long-Context LLMs（延伸）
    url: https://arxiv.org/abs/2502.13189
    author: Lu et al. (Moonshot AI)
    published: 2025-02-18
related: [llm-internals-08, rag, llm-internals-02]
updated: 2026-09-28
---

## 一句话答案

> lost-in-the-middle 是一条受控实验结论：同一份答案放在长上下文的开头或结尾时准确率最高，放到中段会明显下降，整体呈 U 形曲线；窗口能装下 32k/128k 不等于模型能同等利用每个位置。信息完整地进了 prompt，模型却没用上，所以故障形态是「自信错答」而不是报错。机制上至少四层叠加：softmax 注意力被序列长度摊薄、RoPE 在远距离上相位失配、开头 token 充当 attention sink、训练序列长度远短于宣称窗口。工程上的默认动作是「少塞、精排、两端放」，再用位置扫描把弱点测在自己系统的 prompt 与长度上。

## 面试官在考什么

- 能不能把现象讲成可测量的结论：U 形曲线、两个自变量（答案位置、上下文长度）、accuracy 口径。只说「模型中间记不住」是没读完题的答案。
- 是否知道出处与实验做法（多文档 QA + 键值检索两个任务），并能说清它为什么是受控实验，而不是社区传闻。
- 机制层解释：中段为什么弱。面试官要听的是机制链条，不是「论文这么说的」。
- 能不能给出有优先级的解法，并说清每条的代价（延迟、调用次数、召回风险）。
- 评测素养：needle-in-a-haystack 单针测试只测检索；「宣称 1M 上下文」需要什么口径才能验。

**常见错误答案**

- 「窗口扩到 1M 就解决了。」窗口是容量口径，位置利用率是另一个口径；论文里同一模型的扩窗版本与基础版本曲线几乎重合。
- 「中段信息被截断了。」没有被截断：含答案的文档完整送进了 prompt，这是它和「超窗丢弃」的本质区别，也是它危险的地方。
- 「说明检索坏了。」受控实验里答案文档来自人工标注、直接放入，检索是精确的；坏的是 reader 对位置的鲁棒性。生产环境里判断错这一条，会让你去优化一个没坏的模块。

## 原理与推导

### 把「位置」变成自变量：受控实验怎么做的

- **任务一，多文档 QA**：NaturalQuestions-Open 里取「长答案是段落」的 2655 条查询；上下文 = 1 篇含答案的 Wikipedia 段落 + $k-1$ 篇不含答案的干扰段落，每篇不超过 100 token，干扰段按与问题的相关性降序排列。
- **任务二，键值检索**：输入是序列化 JSON，键和值都是随机生成的 128-bit UUID，$k \in \{75, 140, 300\}$，每档 500 例。这个任务去掉了自然语言语义，是「纯检索」的最小测试台。
- **自变量**：含答案文档的位置 $p$（对文档列表重排序得到），以及上下文长度（靠增减干扰文档 $k$ 控制）。注意 $k \in \{10, 20, 30\}$ 是三档「检索少/中/多」，不是固定 token 数。
- **因变量**：accuracy——标注答案字符串是否出现在输出里（不是 EM/F1），greedy decoding。
- **两个参照点**：closed-book（一篇文档都不给）与 oracle（只给那一篇答案文档）。没有这两个锚点，U 形曲线的高低是无从判断的。

如果模型能稳定利用长上下文里的每个位置，性能应当对位置不敏感，曲线是水平的。实测是 U 形：开头（首因）与结尾（近因）高，中段低；文档数越大，中段塌得越深。

论文里最刺眼的一个数字：在 20 文档与 30 文档设置下，GPT-3.5-Turbo 最差位置的准确率低于它 closed-book 的 56.1%——把检索到的文档给它，结果不如不给。

### 机制一：softmax 的注意力预算随长度摊薄

attention 把固定的权重总量按 softmax 分配：

$$
w_i = \frac{e^{s_i}}{\sum_{j=1}^{S} e^{s_j}}
$$

分母随上下文长度 $S$ 增长，单个位置分到的份额量级是 $1/S$。给一个最简模型：目标位置的对数几率比其他位置高 $\delta$，其余 $S-1$ 个位置同分，则

$$
w_{\text{target}} = \frac{e^{\delta}}{e^{\delta} + (S-1)}
$$

取 $\delta = 2$：$S=100$ 时 $w \approx 6.95\%$，$S=10^4$ 时降到 $0.074\%$——长度涨 100 倍，份额跌约 94 倍。这就是「塞得越多、每个位置越轻」的算术根源。

这条只解释「为什么越长越糟」，不解释「为什么偏偏中间最糟」；中间的位置劣势要靠下面三条。

### 机制二：RoPE 在远距离上相位失配

RoPE 把 $q_m$ 与 $k_n$ 按维度成对旋转 $m\theta_i$ 与 $n\theta_i$，于是注意力分数里出现 $\cos((m-n)\theta_i)$，相对距离直接进点积。若假设 Q、K 各维近似独立同分布（可视为训练前的先验），则有

$$
\mathbb{E}\left[q_m^\top k_n\right] \;\propto\; \sum_{i=0}^{d/2-1} \cos(\Delta\,\theta_i),
\qquad \theta_i = 10000^{-2i/d},\quad \Delta = m-n
$$

取 $d=128$（64 个频率对），这个均值随距离的变化是：$\Delta=1$ 时 $0.970$，$\Delta=8$ 时 $0.714$，$\Delta=64$ 时 $0.477$，$\Delta=512$ 时 $0.219$，$\Delta=2048$ 时 $0.119$。高频项的周期只有几个 token，几十个 token 之外相位已经乱了；低频项（$i=63$ 的周期约 54410 token）还能提供粗粒度位置，但幅度小、区分度低。结果是「刚出现过的 token」在分数上系统性地更容易被选中，这正是近因偏好的结构来源。

两点必要的限定：第一，这不是 RoPE 独有——论文里 MPT-30B-Instruct 用的是 ALiBi，U 形照样出现，所以更准确的说法是「多数位置编码方案都不提供远距离的强位置信号」；第二，模型可以用训练把这部分补回来，补不回来的部分才是外推问题（见 [[llm-internals-08]]）。

### 机制三：attention sink 与开头的系统性优势

softmax 要求每个 query 的权重必须归一化，于是总要有个去处。序列第一个 token 几乎不携带语义，却常常吸走一大块注意力质量，成为「注意力垃圾桶」；很多长上下文模型在实现上要显式保留起始 token，原因就在这里。叠加训练期约定把指令、系统提示、格式说明放在最前面，开头位置得到的是双重加成。而结尾靠近生成位置，是近因偏好的受益者。中间两边都不沾。

论文用一组对照排除了「这是指令微调造成的」：MPT-30B 基座与 MPT-30B-Instruct 都呈 U 形，指令微调只是把 best-worst 差从约 10% 收窄到约 4%，趋势没变。

### 机制四：有效长度远小于宣称窗口

$k$ 只是表面变量，训练时的序列长度才是里子。论文里两个模型的窗口是「后期加装」的：MPT-30B-Instruct 先在 2048 token 上预训练，再用 50B token 做 8192 的长度适应；LongChat-13B (16K) 用 condensed RoPE 把 LLaMA-13B 的 2048 扩到 16384。宣称窗口是后期适应出来的，不是原生训出来的。

最硬的证据来自 encoder-decoder：Flan-UL2 在它 2048 token 的训练长度之内，best-worst 只差 1.9%，几乎位置无关；一旦超出 2048 评测，U 形立刻出现；Flan-T5-XXL 的训练长度只有 512 token，超出之后是同一趋势。规模也是变量：Llama-2 系列里 7B 只表现出近因偏差，「记住开头」这件事要到 13B/70B 才稳定出现。

### 机制五：文本分布与人类记忆的偏好

论文把 U 形对应到心理学的 serial-position effect（首因效应 + 近因效应）：人回忆列表时对首尾记得最牢。人类写作也遵循同一分布——结论放开头、总结放结尾、细节堆中间，模型从预训练语料里学到的就是这个先验。这一条是解释性的类比，机制上仍然是上面四条在起作用。

### 解法一：少塞（优先级最高）

先把候选片段砍到少量再进 prompt。做这件事的是 reranker：对每个候选片段打「这篇是否真的回答了问题」的分，只留 top 几篇。中段变短，可丢的地方自然变少。论文第 5 节的开放域 QA 实验给了这件事的量级：检索文档从 20 篇加到 50 篇，GPT-3.5-Turbo 只涨约 1.5%，Claude-1.3 只涨约 1%，而同期 retriever recall 还没饱和——reader 早就榨不出增益了，多给的部分只贡献延迟和账单。业界说的 context rot，量化口径就来自这类实验：塞得多不等于用得上。

### 解法二：精排（零成本，但只是减损）

必须放很多片段时，用顺序把好片段推到两端。记位置 $p$（1-based，$p=1$ 与 $p=k$ 是两端强区），片段到最近强区的距离是

$$
d(p) = \min(p-1,\; k-p)
$$

- **降序排列**（第 $r$ 名放第 $r$ 位）：$d(r) = \min(r-1,\ k-r)$。
- **内折排列**：把最相关的放最前、第二相关放最后、第三放第二位、第四放倒数第二位，依次向内折；奇数名次从前往后放，偶数名次从末位起往前放，$k=20$ 时的顺序是 $1,3,5,\dots,19,20,18,\dots,4,2$。

两种排法的差别在「前几名最坏能坏到什么程度」。$k=20$ 时，降序下前十名的最大 $d$ 是 9，内折后是 4；20 个名次里有 12 个严格变好。$k=30$ 时是 14 降到 7，19/30 变好。代价也要说清：内折把最不相关的片段推进中段（第 20 名的 $d$ 从 0 变成 9），而我们恰好愿意牺牲它们。

这条有个前提：答案大概率在 top-$k$ 里。排序只重新分配损失，不创造信息，所以它不能替代「少塞」。

### 解法三：拆任务与提示层加固

- **map-reduce / 多跳**：每篇片段单独问一遍同一个问题，收集短答案，再用一个很短的 prompt 汇总。每个片段都在短上下文里被读过，中段被物理消灭。代价是调用次数和延迟成倍上升，且汇总步骤会引入新的误差。
- **先抄再答**：让模型先列出与问题相关的原句和出处，再只用抄出来的句子作答。抄写比推理容易得多，这一步把埋在中间的信息提到「新鲜且靠后」的位置。
- **位置与复述**：把问题放在 prompt 末尾（生成位置的近邻，最强位置）；把硬约束（「只依据给定文档回答」）在开头和结尾各写一遍；长对话里周期性把状态重写进摘要，压缩后重申约束。
- **结构化提示**：先给索引与结论，再给细节，让模型先有一个「该去哪里找」的地图，而不是从中段开始盲扫。

### 训练侧与评测侧

- **训练**：长度课程 + 位置均衡采样——把关键信息在序列中均匀撒开，而不是固定放开头；外推阶段配合 YaRN 一类位置缩放（见 [[llm-internals-08]]）。
- **评测**：位置扫描（depth × length 网格）并报告 **best-worst spread**。这个口径正是论文提出的判据：要声称模型能稳健使用长上下文，就要证明性能对相关信息的位置不敏感。
- **别信宣称值**：NIAH 的单针检索可以通过得很好看。RULER 在 NIAH 基础上加入多针、多跳追踪与聚合任务后，17 个宣称至少 32K 窗口的模型里只有一半能在 32K 上保持可接受表现——「支持」和「会用」是两个问题。

## 数值与代码验证

### 论文的数字（口径：多文档 QA accuracy，答案字符串包含判定）

| 模型 | closed-book | oracle | 说明 |
| --- | --- | --- | --- |
| LongChat-13B (16K) | 35.0% | 83.4% | 窗口靠 condensed RoPE 扩出来 |
| MPT-30B-Instruct | 31.5% | 81.9% | ALiBi，同样呈 U 形 |
| GPT-3.5-Turbo | 56.1% | 88.3% | 0613 版本，最差位置低于 closed-book |
| GPT-3.5-Turbo (16K) | 56.0% | 88.6% | 与 4K 版曲线几乎重合 |
| Claude-1.3 | 48.3% | 76.1% | 键值检索上接近满分 |
| Claude-1.3 (100K) | 48.2% | 76.4% | 扩窗版并未更能用上下文 |

其它可直接引用的数字：Flan-UL2 在 2048 token 内的 best-worst 差 1.9%，超出后出现 U 形；键值检索在无 query-aware contextualization 时最差 45.6%，把 query 同时放在数据前后之后，GPT-3.5-Turbo (16K) 在 300 对设置上做到全对；开放域 QA 从 20 篇加到 50 篇只涨约 1.5%（GPT-3.5-Turbo）与约 1%（Claude-1.3）；MPT-30B 到 Instruct 的 best-worst 差从约 10% 收到约 4%。

一个生产口径的换算：检索把答案排在 12/30，按归一化深度 $(12-1)/(30-1) \approx 38\%$ 计算，它正处在论文里最弱的那一段。检索指标上它是命中，生成指标上它可能等于没检索到。

### 自己算：两种排序把每个名次放在哪

```python
def fold_desc(k):
    """内折顺序：1,3,5,...,k,...,6,4,2"""
    odd = [r for r in range(1, k + 1) if r % 2 == 1]
    even = [r for r in range(k, 0, -1) if r % 2 == 0]
    return odd + even

for k in (20, 30):
    seq = fold_desc(k)
    pos = {r: i + 1 for i, r in enumerate(seq)}          # 名次 -> 位置
    naive_d = lambda r: min(r - 1, k - r)                # 降序排列下到最近强区的距离
    fold_d = lambda r: min(pos[r] - 1, k - pos[r])
    top = k // 2
    print(f"k={k}  前 6 位={seq[:6]}  后 6 位={seq[-6:]}")
    print(f"  前 {top} 名的最大距离: 降序={max(naive_d(r) for r in range(1, top + 1))}"
          f"  内折={max(fold_d(r) for r in range(1, top + 1))}")
    print(f"  内折严格更优的名次数: {sum(fold_d(r) < naive_d(r) for r in range(1, k + 1))}/{k}")
    print(f"  最不相关名次的距离: 降序={naive_d(k)}  内折={fold_d(k)}")
```

输出：$k=20$ 时前 10 名最大距离 9 → 4、12/20 更优、最差名次距离 0 → 9；$k=30$ 时 14 → 7、19/30 更优、0 → 14。文中引用的是这段代码的输出。

### 自己算：注意力摊薄与 RoPE 相位

```python
import math

# 机制一：目标位置对数几率高于其他位置 delta，其余 S-1 个位置同分
for S in (100, 1_000, 10_000, 100_000):
    for delta in (1, 2, 3):
        w = math.exp(delta) / (math.exp(delta) + S - 1)
        print(f"S={S:>7}  delta={delta}  w={100 * w:.4f}%")
print("100x 长度下的份额比:", (math.e**2 / (math.e**2 + 99)) / (math.e**2 / (math.e**2 + 9_999)))

# 机制二：Q/K 各向同性时，E[q·k] 正比于各频率对 cos(delta*theta_i) 的均值
def mean_cos(delta, d_head=128, base=10_000.0):
    return sum(math.cos(base ** (-2 * i / d_head) * delta) for i in range(d_head // 2)) / (d_head // 2)

for delta in (1, 8, 64, 512, 2_048, 8_192, 32_768):
    print(f"delta={delta:>6}  mean_cos={mean_cos(delta):+.4f}")
```

输出：$\delta=2$ 时 $w$ 为 6.9453% / 0.7342% / 0.0738% / 0.0074%，100 倍长度的份额比为 94.1；`mean_cos` 为 +0.9702 / +0.7144 / +0.4769 / +0.2194 / +0.1193 / +0.0249 / +0.0565（代码里的 `delta` 在机制一是对数几率优势 $\delta$，在机制二是距离 $\Delta$）。注意 $\Delta$ 到 8000 token 开外，均值已经在 0 附近振荡（8192 时 +0.025，32768 时 +0.057），符号不稳定——长距离上位置信号只剩噪声量级，这本身就是结论。

机制一的数字是示意模型，不是真实 attention map：真实模型有多头、多层与残差通路，单个权重不等于信息利用率，但这个模型解释了为什么「同样的信号在更长的上下文里更弱」。

### 位置扫描 harness（模板，未在本机运行）

```python
"""扫 depth x length，报告 best-worst spread。需要自备 OpenAI 兼容端点。"""
import os
from openai import OpenAI

client = OpenAI(base_url=os.environ["BASE_URL"], api_key=os.environ.get("API_KEY", "EMPTY"))
NEEDLE = "Reykjavik office internal audit code: BLUE-4471."
ASK = "What is the internal audit code of the Reykjavik office? Reply with the code only."

def build(hay: str, depth: float) -> str:
    cut = int(len(hay) * depth)                 # depth=0 在开头，1.0 在结尾
    return f"{hay[:cut]}\n{NEEDLE}\n{hay[cut:]}"

def hit(ctx: str) -> float:
    msg = client.chat.completions.create(
        model="your-model", temperature=0,
        messages=[{"role": "user", "content": f"{ctx}\n\n{ASK}"}],
    ).choices[0].message.content
    return float("BLUE-4471" in msg)

filler = open("filler.txt", encoding="utf-8").read()
for n_chars in (8_000, 64_000, 256_000):        # 英文约合 2k / 16k / 64k token
    hay = (filler * (n_chars // len(filler) + 1))[:n_chars]
    curve = {d: hit(build(hay, d)) for d in [i / 10 for i in range(11)]}
    print(n_chars, "best-worst spread =", max(curve.values()) - min(curve.values()), curve)
```

每个 depth 单次抽样方差很大（0/1 判定），论文的多文档 QA 用的是上千条查询；实际要用几十条不同样本求平均，并把多针、多跳、聚合任务各加一组，否则你测的只是 NIAH 那点检索能力。

## 常见追问

- **追问**：lost-in-the-middle 和「文本超出窗口被截断」有什么区别？
  - 要点：截断是信息在进模型前就没了，属于容量问题；lost-in-the-middle 是信息完整到达却用不上，属于位置利用率问题。前者会明显影响答案且容易通过 token 计数发现，后者没有报错、没有告警，输出照样完整流畅，所以更危险。
- **追问**：把窗口扩到 1M 是不是就解决了？
  - 要点：不解决。窗口是「接受多少 token」的容量口径，位置利用率是另一个口径。论文里 GPT-3.5-Turbo 与 16K 版、Claude-1.3 与 100K 版在窗口内曲线几乎重合；RULER 里 17 个宣称至少 32K 的模型只有一半在 32K 上还能保持可接受表现。而且窗口越长，中段越深，KV cache 与延迟成本同步上涨（见 [[llm-internals-02]]）。
- **追问**：那 needle-in-a-haystack 全绿是不是就说明没问题？
  - 要点：不是。NIAH 只测「单针检索」，也就是从长文本里找出一个孤立事实，属于长上下文能力里最浅的一层。RULER 在它之上加了多针、多跳追踪（变量追踪）与聚合（统计词频）后，几乎所有模型随长度上升都出现大幅掉分。要接近真实负载，得用多跳与聚合任务，并且扫位置。
- **追问**：为什么把问题放在 prompt 末尾有效，把关键文档放开头也有效？
  - 要点：decoder-only 的因果注意力让生成位置只能看前面的 token，紧邻生成位置的 token 天然占优（近因）；开头则是 attention sink 加指令微调数据分布的双重加成。论文的 query-aware contextualization（query 同时放在数据前和数据后）在键值检索上把最差的 45.6% 拉到接近满分，但在多文档 QA 上几乎不改变趋势——它修的是「找得到」，不是「推理得对」。
- **追问**：那到底该用长上下文还是 RAG？
  - 要点：看边际收益，不看宣称值。论文的开放域 QA 里 reader 在 20 篇附近就饱和了，加到 50 篇只值约 1.5%；如果召回还在涨、答案还在丢，说明瓶颈在排序或上下文构造，而不是「窗口不够大」。实践组合：rerank + 截断拿到大部分收益，长窗口留给确实需要跨文档推理的场景，再用 compaction 控制长对话与 agent 的上下文增长。
- **追问**：长对话里 system 规则过一会儿就失效，怎么解释？
  - 要点：规则没被删除，它滑进了中段的弱区。随着对话增长，开头与最新消息保持强势，中间那段指令的权重被摊薄。做法是把硬约束首尾各写一遍，并在每次压缩/摘要后重申，而不是把它留在历史里等模型自己想起来。
- **追问**：怎么在自己的系统里量化这个问题？
  - 要点：固定 prompt 模板与解码参数，扫「答案片段位置 × 上下文长度」网格，每个格点跑几十条样本，报告 best-worst spread 与曲线形状；同时分别用真实业务文档与合成 needle 各测一遍。论文的判据是性能对位置不敏感，你的验收标准也应该是这个，而不是「NIAH 满分」。

## 公司变体

- **Moonshot AI**：偏工程实现与长上下文系统的成本权衡。公开信息显示这家公司把长上下文作为核心产品能力（Kimi 的长文本窗口），并发布了面向长上下文的块稀疏注意力方案 MoBA——把 MoE 的思路用到 attention 上，让模型自行选择注意力块，且可以在 full attention 与稀疏 attention 之间无缝切换，已用于支撑 Kimi 的长上下文请求（<https://arxiv.org/abs/2502.13189>）。因此这题在这类公司里通常不会停在「U 形曲线长什么样」，而会往系统侧推：长上下文与 RAG 的成本/质量取舍、KV cache 与 prefix cache 的账单、位置偏差如何在自己的 prompt 模板与真实文档上测出来、以及稀疏注意力/压缩会不会引入新的位置偏差。数学侧的追问（RoPE 外推、位置编码缩放）也会出现，但落点仍在「这个方案在你的服务里值不值」。

## 相关题目

- [[llm-internals-08]]：RoPE 与 YaRN 一类外推方法。本题机制二、机制四都指向「位置编码在训练长度之外不可靠」，那篇给的是补救手段。
- [[llm-internals-02]]：KV cache 显存公式。长上下文不是免费的能力，它直接换算成每 token 的显存与带宽账单。
- [[rag]]：检索与重排专题。解题的优先级排在第一位的是「少塞、塞对的」，属于 RAG 的排序与截断问题，本题只负责说明为什么这件事值钱。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《The Lost in the Middle Problem in LLMs》，2026-08-19，<https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms>
- Nelson F. Liu 等，《Lost in the Middle: How Language Models Use Long Contexts》（TACL 2023），2023-07-06，<https://arxiv.org/abs/2307.03172>（延伸来源：全部实验设置、口径与论文数字来自此文）
- Cheng-Ping Hsieh 等，《RULER: What's the Real Context Size of Your Long-Context Language Models?》（COLM 2024），2024-04-09，<https://arxiv.org/abs/2404.06654>（延伸来源：NIAH 的局限、17 个模型与 32K 有效长度的结论来自此文）
- Enzhe Lu 等（Moonshot AI），《MoBA: Mixture of Block Attention for Long-Context LLMs》，2025-02-18，<https://arxiv.org/abs/2502.13189>（延伸来源：第 6 节里该公司的公开技术方向）
- 文中「自己算」的表格与数字由第 4 节的 Python 片段输出（排序布局、注意力摊薄、RoPE 相位均值），未引用任何未标注来源的数字。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
