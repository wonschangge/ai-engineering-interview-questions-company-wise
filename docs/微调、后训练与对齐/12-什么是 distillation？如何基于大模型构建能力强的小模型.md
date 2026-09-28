---
type: question
id: finetuning-12
topic: 微调、后训练与对齐
order: 12
question: 什么是 distillation？如何基于大模型构建一个能力强的小模型？
question_en: What is distillation and how do you build a capable small model from a large one?
asked_at: [阿里巴巴]
level: 进阶
tags: [蒸馏, kd, 小模型, 数据]
sources:
  - title: How does Knowledge Distillation work?
    url: https://outcomeschool.com/blog/how-does-knowledge-distillation-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Distilling the Knowledge in a Neural Network（延伸）
    url: https://arxiv.org/abs/1503.02531
    author: Hinton, Vinyals, Dean
    published: 2015-03-09
  - title: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter（延伸）
    url: https://arxiv.org/abs/1910.01108
    author: Sanh et al.
    published: 2019-10-02
related: [finetuning-08, finetuning-07, finetuning-06, inference-serving-12]
updated: 2026-09-28
---

## 一句话答案

> distillation（蒸馏）是拿一个强模型当 teacher、用它产出的监督信号去训练 student 小模型的统称。它至少有两条主线：**logit 蒸馏**用 teacher 的完整概率分布（软标签，配合温度 $T$）当监督；**数据/序列级蒸馏**直接拿 teacher 生成的指令、推理轨迹、偏好对当 SFT 数据——后者是今天 LLM 蒸馏的主流形态，两者可以叠加。收益是成本、延迟与可部署性；代价是学生的能力上限被 teacher 与数据质量锁死，并且会继承 teacher 的幻觉、偏见与风格。

## 面试官在考什么

- 有没有分类骨架：能否把「logit 层的软标签蒸馏」和「数据层的 teacher 生成数据 + SFT」分开讲，并说清两者为什么可以叠加。
- 能不能写出带温度的 softmax 与复合损失，并解释 $T>1$ 和 $T^2$ 两个细节各自的来历——这是区分「背过结论」和「推过一遍」的地方。
- 是否清楚 LLM 时代的主力是数据蒸馏而不是 logits：词表规模、分词器一致性与闭源 API 三重限制把 logit 蒸馏挤到了边缘。
- 能否讲透能力天花板：蒸馏传递的是输出分布与推理模式，不会凭空生成 teacher 没有的能力；teacher 的错误同样会被传递。
- 能否给出可执行的落地流程：什么时候该蒸馏、教师怎么选（能力、调用成本、许可证）、数据怎么配比、用什么口径验收。

**常见错误答案**

- 「蒸馏就是让小模型模仿大模型的答案。」这只说了数据蒸馏里的硬标签那一半，漏掉了软标签携带的类间结构（暗知识）。
- 「蒸馏能让小模型超过大模型。」把个别任务上的现象当通则；也常和量化、剪枝混为一谈——那是压缩同一个模型，蒸馏是换一个学生从头训练。
- 「温度越大暗知识越清楚，所以越大越好。」$T$ 太大时分布趋近均匀，类间差异被摊平；高温极限下蒸馏退化成「对所有 logits 做平方差匹配」，teacher 自己也没有约束好的长尾 logits 被等权对待，噪声的权重反而升高。Hinton 的实验里，把 student 容量砍到很小之后 $T$ 取 2.5–4 最好，再加大并没有更好。

## 原理与推导

### 两类蒸馏：监督信号在哪一层

| | logit 蒸馏（response-based） | 数据/序列级蒸馏 |
| --- | --- | --- |
| 监督信号 | teacher 在每个位置上的完整概率分布 $p^{\text{teacher}}_{T}$ | teacher 生成的序列（答案、推理轨迹、偏好对） |
| 单样本信息量 | 分布级：含类间相似性与不确定性 | 序列级：只含被采样出来的那一条轨迹 |
| 前提条件 | teacher/student **词表与分词器一致**，能拿到 logits | 只要能调用 teacher 生成文本即可 |
| 训练代价 | 每个样本要存/传 $V$ 维分布（见数值节） | 训练就是普通 SFT，额外代价在造数据 |
| 今天的主战场 | 同架构、同分词的「大→小」同族模型、小词表场景 | 跨厂商、闭源 teacher、推理能力迁移 |

### logit 蒸馏：温度、软标签与复合损失

带温度的 softmax 把 raw logits $z_i$ 变成软标签：

$$p_i(T)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

硬标签是 one-hot（零熵，只告诉答案），软标签是整条分布，它额外携带**类间相似性**（猫的照片里 dog 也拿到一点概率）与**不确定性**（teacher 自己也没把握的样本，标签会摊平在几个类上）。这部分额外信息就是暗知识（dark knowledge）。$T=1$ 时 teacher 常常过于自信，长尾被压到 $10^{-3}$ 以下，训练时几乎不产生梯度，所以要放大温度把它们「抬」起来。

复合损失同时学 teacher 的分布与真实标签：

$$\mathcal{L}=\alpha\cdot \text{CE}\big(\text{student}(x),\,y\big)+(1-\alpha)\cdot T^2\cdot \text{KL}\big(p^{\text{teacher}}_{T}\,\big\|\,p^{\text{student}}_{T}\big)$$

其中 $y$ 是硬标签，$\alpha$ 是硬标签权重（各实现把权重记在软标签项还是硬标签项上并不统一，含义相同）。

**为什么抬温度就能露出暗知识。** 对同一组 logits，不同温度下的分布满足

$$p_i(T)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}=\frac{\big(e^{z_i}\big)^{1/T}}{\sum_j\big(e^{z_j}\big)^{1/T}}\ \propto\ p_i(1)^{1/T}$$

即**温度缩放等价于对 $T=1$ 的概率取 $1/T$ 次幂再归一化**。取幂会把小于 1 的数整体抬向 1、把差距压缩，于是 0.95 与 0.005 之间的巨大落差被拉成 0.50 与 0.14 这种量级，长尾类终于能贡献梯度。

**为什么乘 $T^2$。** 设 $C$ 为软标签项的交叉熵，$q$ 为学生的温度分布。对 logits 求导可得

$$\frac{\partial C}{\partial z_i}=\frac{1}{T}\big(q_i-p_i\big)$$

这一步给出的 $1/T$ 是精确的。还有第二个 $1/T$：在软区间（logits 量级远小于 $T$、分布接近均匀）里，两个分布的逐点差本身也被压缩——把 $q_i,p_i$ 在均匀分布附近展开，$q_i-p_i$ 与 logits 之差成正比、与 $T$ 成反比。两个因子乘起来，$\partial C/\partial z$ 的量级 $\propto 1/T^2$。而硬标签项的学生分布跑在 $T=1$ 上，梯度是 $q^{(1)}_i-y_i$，量级 $O(1)$，不随温度缩小。若不补偿，温度一开大，软标签项在总梯度里就消失了；乘上 $T^2$ 正是把软标签项的梯度量级拉回到与硬标签项可比——这就是 Hinton 等人写 $T^2$ 的原因，它是一个**梯度尺度补偿**，不是数学上的恒等变形。

KL 与交叉熵对 student 参数等价：$\text{KL}(p\|q)=\text{CE}(p,q)-H(p)$，而 $H(p)$ 与 student 无关，梯度相同，工程上直接用 `kl_div` 或 `cross_entropy(soft_target)` 都行。

温度只是训练期的技巧：蒸馏时 teacher 与 student 都在 $T>1$ 的分布上对齐，student 部署时仍按 $T=1$ 的 softmax 推理，温度不会留在模型里。这正是必须做梯度尺度补偿的原因——$T$ 改变的是训练信号的尺度，而 student 的推理行为始终定义在 $T=1$ 上。

按 response / feature / relation 还可以把蒸馏分成三类：response-based 只对齐最后一层输出（即上面的 logit 蒸馏）；feature-based 额外对齐 hidden states（中间层或最后一层），student 与 teacher 宽度不同时要加一个投影把宽度映射过去——DistilBERT 的 teacher 与 student 隐藏维度同为 768，cosine 项直接对齐 hidden state，不需要投影；relation-based 对齐样本之间的关系（如两个样本在 teacher 表示空间里的相似度）。对 LLM，feature-based 要求 student 与 teacher 的层数/宽度有对应关系，工程上比 response-based 麻烦得多。

### LLM 时代的主力：数据与序列级蒸馏

**① 数据蒸馏（造 SFT 数据）。** 用强模型生成指令-回答对、推理轨迹、偏好对，再用它们做 SFT。Self-Instruct 的路线是**自举**：从一小组人工种子指令出发，让模型自己续写新指令、生成输入与输出，再用规则去掉与已有指令过于相似的样本，迭代扩充成几万条量级的指令集。这条路线的工程重点不在训练，而在**数据工厂**：种子覆盖度、去重与去污染、难度分档、答案可验证性。选 prompting / RAG / fine-tuning 的判断在 [[finetuning-08]] 里给了框架，蒸馏是其中「fine-tuning 的数据从哪来」这一格的标准答案。

**② 推理能力蒸馏。** DeepSeek-R1 论文的口径是：大规模模型涌现出的推理模式（长链思考、自我检查、回溯）可以被**系统性地蒸馏进小模型**，效果好于让小模型自己从零做 RL。落地差异只有一句话——**学过程，不只学答案**：训练样本里保留 teacher 的思考链，而不是只留最终答案；同时要过滤（答案错的轨迹不能进数据集）与截断（超长轨迹按预算裁剪）。

**③ 结构蒸馏。** DistilBERT 的口径：用 12 层 BERT-base 作 teacher、6 层作 student，损失是**语言建模损失 + 蒸馏损失 + cosine 距离损失**三项之和——前两项分别对齐硬标签与 teacher 分布，cosine 项对齐隐藏状态的方向（这就是 feature-based 的具体做法）。结果：参数减少约 40%、保留约 97% 的语言理解能力、推理快约 60%。数值节复算了 40% 这个数字。

**④ 用蒸馏替代 RL。** RL 阶段探索出的行为（格式、工具调用习惯、拒绝策略）一旦稳定，就可以把它**固化进 SFT 数据**，用一次便宜得多的 SFT 复现，而不是给每个新模型再跑一遍 RL。这也是「蒸馏数据 = 能力的序列化」这个说法的来源；RLVR 一侧的讨论见 [[finetuning-10]]。

**⑤ 迭代蒸馏与拒绝采样。** 更实用的形态是循环：student 采样多条 → 用规则/verifier 或更强模型打分 → 只保留通过的轨迹 → 重新 SFT。它把「teacher 的分布」换成「teacher 分布里被验证过的高分部分」，是对数据蒸馏最直接的提纯手段。

**⑥ 与量化、剪枝的分工。** 蒸馏换结构（参数更少、从头训练），量化换位宽（结构不变、权重精度降低，见 [[inference-serving-06]]），剪枝删结构。三者正交，典型组合是「蒸馏出小模型 → 量化到 int4 → 端侧部署」。端侧的显存账很直观：7B 在 bf16 下是 14 GB，int4 也要约 3.5 GB，而 1.5B bf16 只要 3 GB、0.5B 只要 1 GB——这就是为什么端侧场景里蒸馏几乎不可替代。

### 能力上限与代价

- **上限由 teacher 与数据共同决定，且误差会继承。** 幻觉、偏见、格式怪癖、对某个知识领域的系统性错误都会被蒸进 student。teacher 在评测集上 90 分并不代表 student 也能 90 分，因为 student 只能学到被采样出来的那些行为。
- **logit 蒸馏在 LLM 上不常用的三个硬约束。** ① 词表极大：每个 token 要存 $V$ 维分布，$V=128\text{k}$ 时一个 token 的 bf16 logits 就是 250 KiB，100 万 token 约 238 GiB（数值节给了完整表格）；② teacher 与 student 必须共享词表与分词器，跨厂商几乎不可能；③ 闭源 API 不返回 logits，只能拿到文本。
- **数据规模与多样性决定泛化。** 只在单任务、单风格数据上蒸馏，student 会在该任务上逼近 teacher、在别处迅速退化——这和普通 SFT 的过拟合是同一个问题，缓解手段也一样：蒸馏数据混入通用/人类数据。灾难性遗忘的机制与配比策略见 [[finetuning-07]]。
- **同质化风险。** 所有下游模型都学同一个 teacher 的分布，输出会向同一处坍缩：措辞、推理风格、甚至错误都趋同。缓解办法是保留一部分人类数据、混用多源 teacher、并保留一定比例的自采样数据。
- **不能凭空虚增能力。** 蒸馏是**传递**不是**创造**。个别任务上确实会出现「学生超过老师」：数据经过过滤与聚焦、student 容量集中在一个任务上、软标签起到标签平滑的正则作用、或者数据集里混进了比 teacher 更强的信号（如 verifier 筛出的正确答案）。这是「任务聚焦 + 数据提纯」的收益，不是学生真的比 teacher 更强。

### 落地决策流程

1. **判断该不该蒸馏。** 触发条件是降本、端侧/私有化部署、高频固定任务，或者需要把某个 teacher 的能力一次性固化下来。先算盈亏平衡：只要线上累计输出量能覆盖造数据 + 训练的一次性开销，蒸馏就开始净赚（数值节给了归一化算例，平衡点大致等于数据集本身的 token 量）。把服务成本降一个数量级的完整手段清单见 [[inference-serving-12]]。
2. **选教师。** 看四件事：能力（目标任务上 teacher 自己得先做对）、可调用成本（造数据的 token 量直接决定一次性开销）、许可证与服务条款（是否允许用其输出训练自己的模型）、可复现性（API 模型会静默更新，数据集版本要冻结）。
3. **造数据与配比。** 蒸馏数据 + 通用/人类数据混配，比例按「目标任务 vs 通用能力」的评测结果调，而不是拍一个数。同一批数据要留出验证集。
4. **训练与迭代。** 首选数据蒸馏做 SFT；同族同词表的小模型可以再加 logit 蒸馏项；有 verifier 就上拒绝采样迭代。student 一般从已有的小基座出发，而不是随机初始化。
5. **验收。** 在 held-out 的真实任务上画**质量-成本曲线**（同一条曲线里放 teacher、student、量化后的 student），而不是只看蒸馏集上的分数。评测集必须独立于蒸馏数据——用 teacher 生成的题目去考 student 是自证，评测集设计的原则见 [[rag-04]]。同时按能力维度回归（通用能力、指令跟随、安全拒答），确认没有为了单任务指标丢掉别的能力。

## 数值与代码验证

以下数字均由 `.work/` 下的脚本复算。**口径声明**：温度、梯度与参数量是精确计算；存储、算力与盈亏平衡含明示假设，属于量级估算。

### 1. 温度把长尾抬起来

以源文的 $T=1$ 分布 cat/dog/car/horse $=(0.95,0.04,0.005,0.005)$ 为起点，用 $p_i(T)\propto p_i(1)^{1/T}$ 精确换算：

| $T$ | cat | dog | car | horse | 分布熵 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9500 | 0.0400 | 0.0050 | 0.0050 | 0.2305 nats（0.33 bit） |
| 2 | 0.7406 | 0.1520 | 0.0537 | 0.0537 | 0.8229 nats |
| 4 | 0.5021 | 0.2274 | 0.1352 | 0.1352 | 1.2239 nats（1.77 bit） |
| 8 | 0.3689 | 0.2483 | 0.1914 | 0.1914 | 1.3467 nats |

dog 从 0.04 抬到 0.23（$T=4$），这就是「猫像狗、不像车」这条类间关系从不可见变可见的过程；熵从 0.23 涨到 1.22 nats，而硬标签的熵恒为 0——软标签单样本携带的信息量差在 1 nat 量级。注意源文给的 $T=4$ 示意值是 `cat=0.70 dog=0.22 car=0.05 horse=0.03`，与同一组 logits 的精确缩放不一致（精确值为 0.5021/0.2274/0.1352/0.1352）；**以复算为准**，源文那几个数是示意性的。

### 2. $T^2$ 因子的数值验证

用 PyTorch 在软区间（$|\text{logits}|\ll T$，分布接近 uniform）里验证梯度恒等式与量级：

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.3):
    """alpha 是硬标签权重；软标签项乘 T^2 补偿 1/T^2 的梯度缩放。"""
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * hard + (1 - alpha) * soft

teacher = torch.tensor([[0.30, 0.10, -0.10, -0.30]])   # |z| << T，软区间
student = torch.zeros(1, 4, requires_grad=True)
for T in (1.0, 2.0, 4.0, 8.0, 16.0):
    p = F.softmax(teacher / T, -1)
    kl = F.kl_div(F.log_softmax(student / T, -1), p, reduction="sum")
    g = torch.autograd.grad(kl, student, retain_graph=True)[0]
    q = F.softmax(student.detach() / T, -1)
    assert torch.allclose(g, (q - p) / T, atol=1e-6)   # 恒等式 dC/dz_i = (q_i - p_i)/T
    print(f"T={T:>4}  |q-p|max={(q - p).abs().max().item():.6f}"
          f"  |g|max={g.abs().max().item():.8f}  |g|max*T^2={g.abs().max().item()*T*T:.6f}")
```

实测输出：

| $T$ | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| $\max\lvert q-p\rvert$ | 0.079179 | 0.038651 | 0.019050 | 0.009452 | 0.004707 |
| 未乘 $T^2$ 的 $\max\lvert\partial C/\partial z\rvert$ | 0.079179 | 0.019326 | 0.004763 | 0.001181 | 0.000294 |
| 乘 $T^2$ 后 | 0.079179 | 0.077303 | 0.076202 | 0.075613 | 0.075310 |

未补偿时梯度从 0.0792 掉到 0.000294，比值 269 倍，接近 $16^2=256$；乘 $T^2$ 后稳定在 0.075–0.079。这解释了为什么 $T$ 一开大就必须补 $T^2$：否则软标签项在总梯度里被硬标签项淹没。

### 3. DistilBERT 的参数账（自算）

| 项 | 参数量 |
| --- | --- |
| embeddings（30522×768 + 512×768 + …） | 23,837,184 |
| 单个 transformer 层（attention 2,362,368 + FFN 4,722,432 + 2×LN 3,072） | 7,087,872 |
| BERT-base：12 层 + pooler | 109,482,240（109.5M） |
| DistilBERT：6 层 + pooler/分类头 | 66,955,008（67.0M） |
| 比例 | 61.2%，即减少 **38.8%** |

与论文「参数减少约 40%」一致。注意它**不是**减少 50%：层数减半只砍掉 12 层里的 6 层，而 input embeddings 占 BERT-base 全部参数的 21.8%（$23.84/109.5$），不随层数缩减——这正是「小模型往往被 embedding 拖住」的一般现象。

论文另外两个数字（保留约 97% 的语言理解能力、推理快约 60%）是摘要口径，此处只做引用、不复算。「快 60%」在摘要里只有一句概括：按「耗时降为 0.6 倍」读则加速比约 1.67×，低于层数减半对应的 2× 理论 FLOPs 比，差额来自不随深度缩减的部分（input embeddings、池化与分类头、固定 batch 开销）以及层数变浅后算术强度下降、更偏 memory-bound 带来的 kernel 效率损失；这个百分比到底是耗时比还是加速比，摘要没有交代，引用时要标明口径。

### 4. logit 蒸馏在 LLM 上的存储与算力账

每个 token 都要存一份 teacher 分布，$V$ 是词表大小（表中 k 按 1000 计，如 128k 即 128,000）：

| 词表 $V$ | bf16 每 token | 100 万 token | fp32 每 token | 100 万 token |
| --- | --- | --- | --- | --- |
| 32k | 62.5 KiB | 59.6 GiB | 125 KiB | 119.2 GiB |
| 128k | 250 KiB | 238.4 GiB | 500 KiB | 476.8 GiB |
| 256k | 500 KiB | 476.8 GiB | 1000 KiB | 953.7 GiB |

对照：同一份语料的原始文本约 4 bytes/token，$V=128\text{k}$ 的 bf16 logits 是它的 64,000 倍。算力反而不是瓶颈：KL 每 token 约 $5V\approx 0.64$ MFLOP，3B 学生的单 token 前向约 $2N=6$ GFLOP，占比 0.011%——**卡住 logit 蒸馏的是存储与带宽，不是 FLOPs**。常见缓解手段是只存 top-$k$ 个 logits 的 value 与 index（$k=64$ 时约几百字节/token），代价是丢掉长尾的绝对量级。

### 5. 什么时候回本（归一化口径）

设 $c_s$ 为 student 每 100 万 output token 的服务成本，teacher 是同口径的 $r$ 倍，一次性蒸馏成本为 $D$。替代前成本 $r c_s Q$、替代后 $D+c_s Q$（$Q$ 以 100 万 token 为单位），平衡点：

$$Q^{*}=\frac{D}{c_s\,(r-1)}=\frac{r\,N_d+T_0/c_s}{r-1}$$

假设：数据集 100 万条 × 平均 500 output token $=500$ 单位教师 token（$N_d=500$）；student 3B 在 1B token 上训练，$6ND=1.8\times10^{19}$ FLOPs，按 5e14 有效 FLOP/s 约 10 GPU-h；$c_s$ 取 0.02–0.07 GPU-h/百万 token，等价于 3B 单卡 4,000–14,000 token/s 的持续吞吐（batch=1 的解码被 HBM 带宽卡住，吞吐比这个区间低一个量级，这里按大 batch 连续服务摊薄后的效率取值），故 $T_0/c_s\approx143\text{–}500$。

| $r$（教师/学生成本比） | 10 | 50 | 100 |
| --- | --- | --- | --- |
| $Q^{*}$（百万 token） | 571–611 | 513–520 | 507–510 |

三种倍率下平衡点都在 $5\times10^8$ output token 量级（按每响应 500 token 约 100 万次请求）。取 $r\to\infty$ 可见 $Q^{*}\to N_d$：**平衡点大致等于数据集本身的 token 量**——教师造了多少 token 的蒸馏数据，线上差不多要跑掉同样的量才回本。$r$ 越小越不划算，端侧/私有化等非成本收益要靠别的理由支撑。

## 常见追问

- **追问**：蒸馏和量化、剪枝是什么关系？
  - 要点：三者正交。蒸馏换结构（学生是新模型、参数更少、从头训练），量化换位宽（同一个模型、权重精度降低），剪枝删结构（去掉头/层/通道，通常要再微调恢复）。可以叠加，典型链路是「蒸馏 → 量化 → 端侧部署」。细节对比在 [[inference-serving-06]]。
- **追问**：拿不到 teacher 的 logits 怎么办？
  - 要点：退到数据蒸馏，用 teacher 生成的文本做 SFT；或在同族同词表的小模型上做 logit 蒸馏（可自建 teacher）。另可只蒸馏 top-$k$ logits 降低存储，但必须有 teacher 的权重或 API 支持。
- **追问**：蒸馏数据会让模型同质化吗？
  - 要点：会，程度取决于数据集中度。所有 student 学同一个 teacher 的分布，措辞、推理风格与系统性错误都会趋同。缓解：混入人类/多源数据、保留自采样数据、对 teacher 输出做多样性与难度筛选。
- **追问**：为什么个别任务上学生能超过老师？
  - 要点：数据过滤与任务聚焦让 student 的容量集中在一处；软标签相当于标签平滑起正则作用；数据集里混入了比 teacher 更强的信号（verifier 筛出的正确答案、人类示范）。这是聚焦与提纯的收益，不是蒸馏能凭空造能力。
- **追问**：怎么证明蒸馏有效，而不是自证？
  - 要点：评测集必须独立于蒸馏数据，且不能由 teacher 生成；在 held-out 真实任务上同时画 teacher、student、量化 student 的质量-成本曲线；按能力维度回归通用能力、指令跟随与安全拒答，确认没有单任务过拟合。
- **追问**：蒸馏和「直接在目标任务上微调」怎么选？
  - 要点：数据来自 teacher 就用蒸馏（能拿到推理轨迹，比只标答案信息多）；数据来自人工标注时微调更直接。两者不冲突——蒸馏数据本身就是 SFT 数据的一种来源。选择框架见 [[finetuning-08]]。

## 公司变体

**阿里巴巴（Qwen）**：依据公开技术材料与团队方向，这题在阿里侧偏**工程实现**，面试重心通常落在数据工厂而不是数学推导——teacher 怎么选（同族更大模型 vs 外部强模型）、蒸馏数据怎么造与过滤、通用数据怎么配比以防能力回退、蒸馏与 RL 的成本账怎么算、端侧/私有化场景的部署形态。公开技术材料里，Qwen 系列的小尺寸 dense 模型走的是同族强到弱蒸馏（strong-to-weak distillation）路线，即用同系列更大的模型作 teacher，这也解释了为什么同族蒸馏在工程上更受青睐：词表与分词器天然一致，还能顺带做 logit 蒸馏。原理侧的要求到「能写出带温度的软标签损失、解释 $T$ 与 $T^2$ 的动机、说清能力天花板」为止，不会要求手推温度缩放的完整级数展开。

## 相关题目

- [[finetuning-08]]：prompting / RAG / fine-tuning 的决策框架。蒸馏回答的是「决定 fine-tune 之后，训练数据从哪来」。
- [[finetuning-07]]：灾难性遗忘。蒸馏数据是典型的窄分布数据，配比不当就会把通用能力挤掉。
- [[finetuning-06]]：LoRA / prefix tuning / prompt tuning / full fine-tuning 的对比。学生模型训练时同样要做这个选择。
- [[finetuning-10]]：RLVR。把 RL 探索到的行为固化进蒸馏数据，是替代「再跑一遍 RL」的标准做法。
- [[inference-serving-12]]：把服务成本降低 10 倍的手段排序。蒸馏是其中换模型的那一档。
- [[inference-serving-06]]：FP16/BF16/FP8/INT8/INT4/FP4 的对比。蒸馏之后接着量化，是端侧部署的常规组合。
- [[rag-04]]：RAG 流水线的质量评估。评测集必须独立于训练/蒸馏数据的道理相同。

## 参考资料与归属

1. [How does Knowledge Distillation work?](https://outcomeschool.com/blog/how-does-knowledge-distillation-work)，Amit Shekhar（Outcome School），2026-06-11。提供本题的动机叙述（teacher/student 与端侧部署）、硬标签与软标签对照、暗知识与温度的作用、复合损失的权重组装、逐步训练流程，以及 response/feature/relation 三类划分与 DistilBERT、端侧部署等实例；正文的分类骨架与「$T$ 只在训练期使用」这一点取自该文。
2. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)（延伸），Hinton、Vinyals、Dean，arXiv 2015-03-09（NIPS 2014 Deep Learning Workshop）。提供软标签与温度形式、梯度恒等式 $\partial C/\partial z_i=(q_i-p_i)/T$ 与高温近似 $\partial C/\partial z_i\approx(z_i-v_i)/(NT^2)$（$v_i$ 为 teacher logits）、软标签项乘 $T^2$ 这一原始口径，以及「压缩 ensemble 的知识供部署」的问题设定。正文「为什么乘 $T^2$」的分步解释与数值验证是对这两步的复算。
3. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)（延伸），Sanh、Debut、Chaumond、Wolf，arXiv 2019-10-02（v4 2020-03-01，NeurIPS 2019 EMC² Workshop）。提供结构蒸馏的实例与口径：预训练阶段蒸馏、语言建模 + 蒸馏 + cosine 距离三重损失、参数量减少 40%、保留约 97% 的语言理解能力、快约 60%，以及端侧可行性验证。数值节的参数量账是对其「减少 40%」的独立复算。

数值节的温度换算与熵、$T^2$ 梯度实测、logit 存储与算力账、盈亏平衡算例均为按该节声明的口径自行复算，不属于上述来源的结论；源文 $T=4$ 的示意分布与精确温度缩放不一致，正文以复算值为准并已注明。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
