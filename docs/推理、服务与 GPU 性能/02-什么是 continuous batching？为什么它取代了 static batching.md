---
type: question
id: inference-serving-02
topic: 推理、服务与 GPU 性能
order: 2
question: 什么是 continuous（in-flight）batching？为什么它取代了 static batching？
question_en: What is continuous (in-flight) batching and why did it replace static batching?
asked_at: [Anthropic, xAI, Mistral AI, NVIDIA, Together AI]
level: 进阶
tags: [batching, 调度, 吞吐, 尾延迟]
sources:
  - title: Continuous Batching in LLMs
    url: https://outcomeschool.com/blog/continuous-batching-in-llms
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
related: [inference-serving-01, inference-serving-03, inference-serving-14]
updated: 2026-09-28
---

## 一句话答案

> continuous batching（NVIDIA 的 TensorRT-LLM / FasterTransformer 里叫 in-flight batching，Orca 论文里的名字是 iteration-level scheduling）把调度单位从「一个请求」降到「一次 decode 迭代」：每个 step 结束后立刻回收已经吐出 EOS 的槽位，把排队请求的 prefill 塞进空位，于是同一批的成员逐 step 变化，长度不同的请求不必同生共死。
> static batching 的批必须等到最长的那个请求结束才能交接，槽位利用率只有 $\mathbb{E}[L]/\mathbb{E}[\max_S L]$。长度服从指数分布、$S$ 个槽位时这个比值恰好是 $1/H_S$（$S=8$ 时 36.8%，$S=32$ 时 24.6%），也就是相同的墙钟时间能多出 2.7–4.1 倍 token；按 step 级模拟实测是 2.73× 与 4.06×。
> 代价是需要一个能随时增删、按需分配的 KV cache（[[inference-serving-03]]）、一个会做抢占与重排的调度器，以及接受新请求的 prefill 插进别人的 decode 流里（[[inference-serving-14]]）。

## 面试官在考什么

- **调度粒度。** 能不能一句话说清「static 是请求级、continuous 是迭代级」，并指出每个 step 都在做两个动作：回收（谁 EOS 了就出队）与填充（队列里有谁就入队）。只会说「continuous batching 就是不等整批结束」是及格线。
- **收益归因。** 这题的区分点在于是把浪费归到「padding 白算的 FLOPs」还是归到「每步读权重的固定成本被乘到了最长请求的步数上」。前者在空槽不再参与计算时会缩水，后者不会：8B bf16 在 H100 上读一遍 16 GB 权重是 4.78 ms，而多带一个上下文 1024 的请求只多 0.04 ms，空槽省下的那点算力根本救不了被拉长的墙钟时间。
- **量级从哪来。** 能报出收益取决于长度分布的方差与队列是否常满，能至少给出 $\max/\text{mean}$ 的口径（文章里的例子是 1.82×），最好能说出指数分布下是 $H_S$（$H_8=2.72$、$H_{32}=4.06$、$H_{64}=4.74$，边际递减）。再叠上 PagedAttention 的显存收益，pipeline 整体才是 vLLM 论文那句「同延迟下 2–4×」。
- **前提与代价。** 没有分页 KV cache，新请求根本挤不进来（按 $\text{max\_len}=8192$ 预留时 8B 一个槽位就要 1 GiB）；没有抢占，长请求会把显存和 batch 位置一直占住；新请求的 prefill 会阻塞已有请求的 decode。
- **概念串联。** continuous batching（时间维度填满 GPU）、chunked prefill（把长 prefill 切碎，[[inference-serving-14]]）、PagedAttention（显存维度支持动态增删，[[inference-serving-03]]）、disaggregation（把 prefill 与 decode 拆到不同 GPU）。四者解决的是四个不同层次的浪费。

常见错误答案：

- 「continuous batching 减少了计算量 / 让前向更快」。它不改模型、不改算子，也不改单请求的每步计算；单请求的 step 时间几乎不变，把 batch 填满后还会略涨（8B、上下文 1024 时 $B=1$ 是 4.82 ms，$B=8$ 是 5.10 ms，+6%）。
- 「就是把 batch size 调大」。batch 上限由显存和延迟 SLO 决定，是另一件事；continuous batching 管的是「批内成员怎么换人」，不是「批能有多大」。
- 「空槽位浪费算力所以慢」。真正省不掉的是每个 step 都要读一遍权重这个固定成本；空槽即使完全不参与计算，读权重的时间照样要花。

## 原理与推导

### 1. static batching 的机制与病灶

static batching 的循环是：攒够 $S$ 个请求 → 整批一起 prefill → 整批一起 decode，每步给每个槽位各出一个 token → **直到批内每一个请求都结束**，才把槽位交还给下一批。实现简单、形状固定、容易做 CUDA graph，代价是批的存活时间由最长的那个请求决定：

$$T_{\text{batch}} = \max_{i \le S} L_i \cdot t_{\text{step}}, \qquad \text{有用工作量} = \sum_{i \le S} L_i \cdot t_{\text{step}}$$

槽位利用率因此是

$$\eta_{\text{static}} = \frac{\sum_i L_i}{S \cdot \max_i L_i} = \frac{\bar L}{\max_i L_i}$$

生成长度是重尾分布，$\max$ 与均值差得很远：这一项就是全部病灶。批内形状有两种处理方式，代价不同但都救不回时间：

- **padded 到等长**：把批内所有序列 padding 到批的 $\max L_i$（或直接到 $\text{max\_len}$），张量形状固定、能用 CUDA graph，但 padding 位置照样参与前向计算，白算的 FLOPs 比例等于 padding 占比。测试例子里长度 5/2/7 的三个请求，pad 到等长后 21 行里只有 14 行是真实 token（33.3% 的浪费）。
- **留空槽**：让已结束的序列退出计算、只保留形状，kernel 在空槽上几乎不花 FLOPs，但**墙钟时间不变**——批仍然要跑到最长请求结束（每步读权重的固定成本一节会说明为什么这一点致命）。

两个后果分开看：

- **时间维度**：批内先结束的槽位在剩下的步数里空转，新请求只能排队等整批结束（head-of-line blocking）。短请求「写一首五言绝句」和长请求「写一千字作文」放进同一批，短请求在第 5 步就完了，槽位却要空到第 400 步。
- **排队维度**：即使 GPU 因为空槽而不满载，队列里的请求也无法开始，TTFT 被整批的剩余时间绑定。

长度 iid、均值 $\mu$ 时，$\mathbb{E}[\max]$ 有闭式解（指数分布为例）：

$$\mathbb{E}\Bigl[\max_{i\le S} L_i\Bigr] = \int_0^\infty \Pr\bigl(\max_i L_i > x\bigr)\,dx = \int_0^\infty \Bigl(1 - \bigl(1 - e^{-x/\mu}\bigr)^S\Bigr)dx = \mu \sum_{k=1}^{S} \frac{1}{k} = \mu H_S$$

换元 $u = 1 - e^{-x/\mu}$ 之后被积函数变成 $\mu(1-u^S)/(1-u) = \mu(1 + u + \dots + u^{S-1})$，逐项积分即得第 $k$ 项 $1/k$。于是 $\eta_{\text{static}} = 1/H_S$：$S=8$ 时只有 36.8% 的槽位时间在做有用工作，$S=32$ 时 24.6%。

### 2. 每步的成本下限：为什么空槽省不掉

要算清收益，必须知道一个 decode step 的成本随 batch 怎么变。decode 每步只处理一个 token，权重读取代价与 batch 无关，KV cache 读取代价随 batch 线性增长：

$$t_{\text{step}}(B) = \max\left(\underbrace{\frac{W + B\,\bar c\,k_{\text{tok}}}{BW_{\text{hbm}}}}_{\text{带宽侧}},\ \underbrace{\frac{2NB}{F_{\text{peak}}}}_{\text{算力侧}}\right),\qquad W = 2N\ (\text{bf16})$$

$N$ 是参数量、$\bar c$ 是平均上下文长度、$k_{\text{tok}}$ 是每 token 的 KV 字节数。两项相等处就是存算拐点，记 $r = F_{\text{peak}}/BW_{\text{hbm}}$（H100 bf16 上是 $989/3.35 \approx 295$ FLOP/byte）：

$$B^\star = \frac{r}{1 - \dfrac{\bar c\,k_{\text{tok}}\,r}{2N}}$$

代入 Llama-3-8B（$N = 8\times10^9$、$\bar c = 1024$、$k_{\text{tok}} = 128$ KiB）得 $\bar c k_{\text{tok}} r / 2N = 2.47 > 1$，分母为负——**不存在拐点，这个配置在任意 batch 下都在带宽侧**。也就是说 $t_{\text{step}} \approx W/BW + B\bar c k_{\text{tok}}/BW$：$B$ 从 1 涨到 256，步长只从 4.82 ms 涨到 15.03 ms（3.1 倍），而每步产出的 token 多 256 倍，吞吐从 208 涨到 17,029 token/s（82 倍）。这正是 batching 在 decode 阶段「几乎免费」的原因，也正是 static batching 把固定成本乘到最长请求步数上会亏这么多的原因。

### 3. 两种口径的吞吐比

- **常数步长口径**（源文与多数博客隐含的假设：空槽与满槽一样贵）：static 要跑 $\mu H_S$ 步才交接一批，continuous 用满槽跑 $\mu$ 步就能产出同样多的 token，比值是 $H_S$。
- **考虑排空的真实口径**（空槽退出后步长真的变短）：记 $a = W/BW$ 为读权重的固定成本、$b = \bar c k_{\text{tok}}/BW$ 为每个请求每步的 KV 读取成本。批的存活时间是 $M$，批内第 $t$ 步的占用是 $B_t$，则 $\int_0^{M} B_t\,dt = \sum_i L_i = S\mu$ 恒成立（每个请求恰好贡献 $L_i$ 步的占用），于是

$$T_{\text{static}} = a\,\mathbb{E}[M] + b\sum_i L_i = a\mu H_S + bS\mu,\qquad T_{\text{cont}} = \mu\,(a + bS)$$

$$R = \frac{a H_S + bS}{a + bS} = \frac{H_S + bS/a}{1 + bS/a}$$

$a$ 占主导时 $R$ 逼近 $H_S$，$b$ 占主导（长上下文、大 $S$）时收益被 KV 读取代价吃掉一部分。8B、上下文 1024、$S=8$ 时 $a = 4.78$ ms、$bS = 0.32$ ms，$R = 2.61$。

### 4. continuous batching 的 step 级循环

```text
loop:
    # 1. 回收：上一步结束后，EOS / 超长 / 被抢占的请求出队，释放 KV 块
    for req in running:
        if req.finished: release_kv_blocks(req); running.remove(req)

    # 2. 填充：有空槽就从等待队列取请求，为它做 prefill（可切块，见 chunked prefill）
    while free_slots() and waiting_queue:
        req = waiting_queue.pop()
        allocate_kv_blocks(req, req.prompt_len); running.append(req)

    # 3. 一次统一的 forward：running 中每个请求各出一个 token
    #    显存不足时先抢占（recompute 或 swap 掉 KV 最小的请求）
    if kv_pool.exhausted(): preempt(victim)
    logits = model.forward(pack(running))          # 变长 packed + 块对角 causal mask
    sample_and_stream(running)
```

关键性质是**数学等价**：把若干个长度不同的请求 pack 成一条扁平序列、用块对角 causal mask 隔开，与各自单独跑的结果逐元素相同。实测（float64，长度 5/2/7 的 3 个请求，2 头、$d=16$）packed 与 solo 的最大偏差是 $4.4\times10^{-16}$，中途再塞进第 4 个请求后已有行的输出漂移同样是 $4.4\times10^{-16}$——漂移只来自浮点归约顺序。所以「插队」既不需要重新编译，也不需要重跑已经算过的 token，用户看到的输出与独占运行一致（采样随机性由各自的 RNG 决定，不受 batch 组成影响）。

调度状态机要处理的状态比 static 多：等待队列、运行集合、被抢占集合、每个请求的 KV 块表、前缀共享的引用计数。抢占有两种做法——recompute（丢掉 KV，回来时重算 prefill，适合短 prompt）与 swap（把 KV 换到 CPU 内存，适合长 prompt），选择依据是 recompute 的算力代价与 swap 的 PCIe 带宽代价之比。

### 5. 收益的边界与代价

- **长度方差越大越赚。** 方差为 0（所有请求等长）时 $\eta_{\text{static}} = 1$，continuous 一点用没有；重尾分布（对话、Agent 的多轮工具调用）里收益最大。模拟中指数分布、对数正态（$\sigma=1$）、两个指数分量的混合分布（50% $\text{Exp}(\mu=20)$ + 50% $\text{Exp}(\mu=200)$）在 $S=32$ 时分别是 4.06×、5.44×、6.10×。
- **队列必须常满。** 收益靠「空出来的槽位立刻有人填」。低 QPS 时槽位天然就空着，continuous batching 的吞吐收益消失，只剩延迟收益（源文的 4 请求玩具例子就是这样：A–D 这一批在两边都是 200 步，收益只体现在后来者 E 的等待时间上——源文里 static 总耗时 11.5 s、continuous 10.0 s）。
- **$S$ 的边际收益递减，而延迟代价线性上升。** $H_S$ 是 $\ln S + \gamma$ 量级，$S=8 \to 32 \to 64$ 只从 2.72 涨到 4.06、4.74；而每步的 KV 读取与 TPOT 随 $S$ 线性增长。延迟敏感的产品线因此宁可把 batch 上限压低换 TPOT 稳定。
- **prefill 会打断 decode。** 新请求在同一个 step 里和别人的 decode 混跑：长 prompt 的 prefill 是一整块计算密集的算子，会把同批所有请求的 TPOT 顶出一个尖峰。解法是 chunked prefill 或 disaggregation。
- **KV 显存是硬约束。** 见下一节的预分配账：不做分页，槽位数量会被 $\text{max\_len}$ 而不是真实平均长度限制住，continuous batching 也就无从谈起。

## 数值与代码验证

### 复算一：static 的槽位效率与两种口径的比值

H100 SXM 官方规格：bf16 dense 989 TFLOP/s、HBM3 3.35 TB/s、80 GB。$k_{\text{tok}}$ 用 $2 \cdot n_{\text{layer}} \cdot n_{\text{kv\_head}} \cdot d_{\text{head}} \cdot b$ 计算，bf16（$b=2$）：Llama-3-8B（32 层 / 8 个 KV 头 / $d_{\text{head}}=128$）是 131,072 B = **128 KiB/token**；Llama-3-70B（80 层 / 8 个 KV 头 / 128）是 327,680 B = **320 KiB/token**；同规模但用 MHA 的 7B（32 层 / 32 头 / 128）是 512 KiB/token，GQA 的 4× 节省直接体现在这里。

| 场景 | 长度 | 槽位 $S$ | static 步数 | 有用 token-step / 总槽位步 | 槽位利用率 | continuous 步数 | 比值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 源文 4 请求例 | 100 / 20 / 50 / 200 | 4 | 200 | 370 / 800 | 46.25% | 200 | 1.00×（没有积压，只赚延迟）|
| 源文 100 请求例 | 50 个 20 + 50 个 200 | 4 | 25 × 200 = 5000 | 11000 / 20000 | 55.0% | 2750 | 1.82× = 200/110 |
| 随机轨迹（200k 请求，$\mu=100$）| 指数分布 | 8 | 6,804,647 | — | 36.69% | 2,497,091 | **2.73×**（$H_8 = 2.72$）|
| 同上 | 指数分布 | 32 | 2,535,504 | — | 24.62% | 624,772 | **4.06×**（$H_{32} = 4.06$）|

第 2 行是源文自己的数字，按 $11000/250\text{ s} = 44$ token/s 对 $11000/137.5\text{ s} = 80$ token/s 复算得 1.82×，恰好等于 $\max/\text{mean} = 200/110$（源文的 2750 步是「总 token 数 ÷ 槽位数」的理想值；用代码一的 `continuous_makespan` 按 2 短 2 长交替到达实跑是 2840 步，差额来自收尾时队列排空）。第 3、4 行是本次模拟的结果：static 的实测槽位利用率 36.69% 与 $1/H_8 = 36.79\%$、24.62% 与 $1/H_{32} = 24.64\%$ 相差都在 0.1 个百分点以内，说明 $1/H_S$ 这个闭式解是对的。注意第 1 行：没有积压时 continuous 的 makespan 与 static 完全相同，玩具例子只能展示单请求延迟收益。

| 长度分布（20 万条请求，`numpy.random.default_rng(0)`）| $S$ | $\mathbb{E}[L]$ | $\mathbb{E}[\max_S L]$ | $H_S$ | 实测 static/continuous | static 槽位利用率 |
| --- | --- | --- | --- | --- | --- | --- |
| 指数 $\text{Exp}(\mu=100)$ | 8 | 99.87 | 272.19 | 2.72 | 2.73× | 36.69% |
| 指数 $\text{Exp}(\mu=100)$ | 32 | 99.87 | 405.68 | 4.06 | 4.06× | 24.62% |
| 对数正态（$\sigma=1$，均值 100）| 8 | 100.02 | 307.53 | 2.72 | 3.07× | 32.52% |
| 对数正态（$\sigma=1$，均值 100）| 32 | 100.02 | 544.57 | 4.06 | 5.44× | 18.37% |
| 混合：50% $\text{Exp}(20)$ + 50% $\text{Exp}(200)$ | 8 | 110.15 | 405.74 | 2.72 | 3.68× | 27.15% |
| 混合：50% $\text{Exp}(20)$ + 50% $\text{Exp}(200)$ | 32 | 110.15 | 672.96 | 4.06 | 6.10× | 16.37% |

$H_S$ 只在指数分布下等于 $\mathbb{E}[\max]/\mathbb{E}[L]$；换分布后 $\mathbb{E}[\max]$ 变大，收益随之变大，但「$S$ 越大 $\mathbb{E}[\max]$ 越大、槽位利用率越低」的趋势不变。

### 代码一：step 级调度模拟（可运行）

```python
def static_makespan(lengths, S):
    """攒够 S 个跑一批，整批同生共死。"""
    t = slot_steps = 0
    for i in range(0, len(lengths), S):
        chunk = lengths[i:i + S]
        m = max(chunk)
        t += m
        slot_steps += m * S            # 空槽照样占着 GPU 的这一步
    return t, slot_steps

def continuous_makespan(lengths, S):
    """每步结束就回收 + 填充，空槽只在队列确实空着时存在。"""
    n, t, nxt, active = len(lengths), 0, 0, []
    while nxt < n or active:
        while len(active) < S and nxt < n:
            active.append([lengths[nxt], 0]); nxt += 1
        if not active:
            break
        for a in active:
            a[1] += 1
        t += 1
        active = [a for a in active if a[1] < a[0]]   # 到 EOS 的立刻出队
    return t, sum(lengths)
```

用 `static_makespan` 复现源文的 4 请求例子得到 200 步 / 800 槽位步 / 370 有效 token-step（46.25%）；上面两张表的口径是 `numpy.random.default_rng(0)`、20 万条请求、长度取整到 token，指数分布轨迹得到 2.73× 与 4.06×。该模拟沿用源文的简化：每个 step 的花费与占用无关，所以它衡量的是「槽位时间的浪费」，不包含空槽变快带来的补偿。

### 复算二：把排空效应算进去之后的比值与队列等待

用第 2 节的 $t_{\text{step}}(B) = a + bB$（8B / 1×H100、上下文 1024、$\mu=100$ token；70B bf16 用 TP=4，所以按每卡字节数记账：权重 35 GB/卡、KV 80 KiB/token/卡、上下文 2048）：

| 配置 | $S$ | $H_S$ | 常数步长口径 | 排空感知口径 $R$ | static 最坏队列等待 |
| --- | --- | --- | --- | --- | --- |
| 8B / 1×H100 | 8 | 2.72 | 2.72× | 2.61× | 272 步 × 5.10 ms = 1.39 s |
| 8B / 1×H100 | 32 | 4.06 | 4.06× | 3.41× | 406 步 × 6.06 ms = 2.46 s |
| 70B / TP=4 | 32 | 4.06 | 4.06× | 3.65× | 406 步 × 12.05 ms = 4.89 s |
| 70B / TP=4 | 64 | 4.74 | 4.74× | 3.86× | 474 步 × 13.65 ms = 6.47 s |

「最坏队列等待」是请求恰好在批刚启动时到达、必须等整批排空的时间；continuous batching 下同一个请求在下一个 step 就能开始 decode（等待量级是几个毫秒，代价是它的 prefill 要跟别人的 decode 抢一个 step）。

decode 步长的模型值（理想峰值带宽口径，不含 TP 通信、attention kernel 与框架开销，所以是下界）：

| $B$ | 8B：步长 / 吞吐 / TPOT | 70B TP=4：步长 / 吞吐 / TPOT |
| --- | --- | --- |
| 1 | 4.82 ms / 208 tok/s / 4.82 ms | 10.50 ms / 95 tok/s/卡 / 10.50 ms |
| 8 | 5.10 ms / 1,570 tok/s / 5.10 ms | 10.85 ms / 737 tok/s/卡 / 10.85 ms |
| 32 | 6.06 ms / 5,282 tok/s / 6.06 ms | 12.05 ms / 2,656 tok/s/卡 / 12.05 ms |
| 64 | 7.34 ms / 8,719 tok/s / 7.34 ms | 13.65 ms / 4,688 tok/s/卡 / 13.65 ms |
| 256 | 15.03 ms / 17,029 tok/s / 15.03 ms | 23.27 ms / 11,002 tok/s/卡 / 23.27 ms |

两个配置在 256 的 batch 下算力侧仍只占 $4.14/15.03 = 27.5\%$ 与 $9.06/23.27 = 38.9\%$（MFU 口径），印证了 $\bar c k_{\text{tok}} r/2N > 1$ 的结论：decode 的吞吐上限由 KV 与权重的字节数决定，不由 tensor core 决定。

### 复算三：KV 显存——「随时插入」的硬件前提

| 项 | 8B（1×H100，60 GiB 留给 KV）| 70B bf16 TP=4（每卡 43 GiB 留给 KV）|
| --- | --- | --- |
| 每 token KV | 128 KiB（单卡即全模型）| 全模型 320 KiB，TP=4 后每卡 80 KiB |
| 总容量 | 491,520 token ≈ 240 × 2048 token | 每卡 563,610 token（每卡只存 1/4，4 卡合起来仍是约 56 万 token 的上下文）|
| 按 $\text{max\_len}=8192$ 预留时每槽 | 1.00 GiB | 全模型 2.50 GiB（每卡 0.625 GiB）|
| 按 $\text{max\_len}=32768$ 预留时每槽 | 4.00 GiB | 全模型 10.00 GiB（每卡 2.50 GiB）|
| 槽位数量（未分页，每卡预算 ÷ 每卡预留）| 60 槽（$\text{max\_len}=8192$；平均 1000 token 时只用掉 12.2% 的预留）| 68 槽（$\text{max\_len}=8192$）、17 槽（$\text{max\_len}=32768$）|

预留预算按 80 GB 减去权重（bf16 下 8B 是 16 GB、70B 在 TP=4 下每卡 35 GB）再扣掉 activations / CUDA graph / 通信缓冲的假设值，8B 取 60 GiB、70B 每卡取 43 GiB。

若每个槽位按 $\text{max\_len}=8192$ 预留，8B 在 60 GiB 预算下只能开 60 个槽位；分页之后同一块显存能装下 491,520 个真实 token，平均上下文 1000 时相当于约 491 个并发序列——**8.2 倍的并发差**，这才是「空槽能立刻被新请求填上」的物质基础。70B 更极端：$\text{max\_len}=32768$ 时每卡一个槽位就要 2.50 GiB（全模型 10 GiB），43 GiB 预算只够 17 个槽位，batch 上限由 $\text{max\_len}$ 而不是真实长度决定，continuous batching 也就失效。

### 代码二：packed batch 与单独运行等价（可运行）

```python
import torch
torch.manual_seed(0)
L, H, D = [5, 2, 7], 2, 16
qs = [torch.randn(a, H, D, dtype=torch.float64) for a in L]
ks = [torch.randn(a, H, D, dtype=torch.float64) for a in L]
vs = [torch.randn(a, H, D, dtype=torch.float64) for a in L]
sdpa = torch.nn.functional.scaled_dot_product_attention

def bhs(x):
    return x.transpose(0, 1)[None]                         # (T, H, D) -> (1, H, T, D)

solo = [sdpa(bhs(q), bhs(k), bhs(v), is_causal=True)[0].transpose(0, 1)
        for q, k, v in zip(qs, ks, vs)]                    # 各自独占跑

T = sum(L)
Q, K, V = torch.cat(qs), torch.cat(ks), torch.cat(vs)
ids = torch.repeat_interleave(torch.arange(len(L)), torch.tensor(L))
pos = torch.cat([torch.arange(a) for a in L])
mask = ((ids[:, None] == ids[None, :]) & (pos[:, None] >= pos[None, :]))[None, None]
packed = sdpa(bhs(Q), bhs(K), bhs(V), attn_mask=mask)[0].transpose(0, 1)   # 一个 step 混跑

print("max |packed - solo| =", max((packed[sum(L[:i]):sum(L[:i+1])] - solo[i]).abs().max().item()
                                   for i in range(len(L))))
print("mask 密度 =", mask.sum().item() / mask.numel())
```

实测两项：packed 与 solo 的最大偏差 $4.4\times10^{-16}$，加入第 4 个请求后已有行同样只漂移 $4.4\times10^{-16}$（float64 的舍入量级，不是近似）；块对角 mask 的密度只有 0.235，说明稠密 mask 写法下 76.5% 的 mask 位是空洞——生产内核因此用变长（varlen）attention 而不是把稠密 mask 喂给 kernel。

## 常见追问

- **追问**：这和 NLP 推理里的 dynamic batching 有什么区别？
  - 要点：dynamic batching（NVIDIA Triton 的 dynamic batcher 这类入口侧策略）只在**入口侧**按时间窗口凑批：窗口内到达的请求合成一批，批一旦发出仍然同生共死，窗口设小了吞吐低、设大了 TTFT 高。continuous batching 改的是**批的整个生命周期**，批内成员在每个 step 都可以换，所以能同时拿到低 TTFT 与高吞吐。
- **追问**：延迟敏感的场景为什么还要给 batch 上限？
  - 要点：吞吐 $= B/t_{\text{step}}(B)$ 随 $B$ 单调增，但 TPOT $= t_{\text{step}}(B)$ 也随之上升（8B 上 $B=1 \to 256$ 是 4.82 → 15.03 ms）。SLO 卡在 TPOT 上时，超过某个 $B$ 之后多出来的吞吐是「不能卖的吞吐」（goodput 不再增长）。实操上按 SLO 反解 $B$ 上限，再用调度器优先级/分队列保证尾延迟，而不是无脑加并发。
- **追问**：prefill 和 decode 能混在同一个 batch 里吗？
  - 要点：能，而且是 continuous batching 的必然结果（新请求入队就要在下一个 step 做 prefill）。风险是长 prompt 的 prefill 是计算密集算子，会把同 step 所有请求的 TPOT 顶出一个尖峰。工程解法是 chunked prefill：把长 prefill 切成固定预算的块，与 decode 拼成「算力预算大致恒定」的混合批；极端做法是 disaggregation。见 [[inference-serving-14]]。
- **追问**：continuous batching 会改变模型的输出吗？
  - 要点：不会。批内每个请求的 attention 被 block-diagonal mask 隔离（或 varlen 内核等价处理），数学上与独占运行逐元素相同，差异只在浮点归约顺序（实测 $10^{-16}$ 量级）。会变的是**浮点级别的确定性**：同一个请求在不同 batch 组成下可能得到最后几位不同的 logits，采样在极低概率 token 上理论上可能分叉。要求 bitwise 可复现的场景要固定 batch 组成或固定随机数流。
- **追问**：吞吐提升了，成本怎么核算？
  - 要点：分母用「有效 token/s per GPU」或「每百万 token 的成本」，而不是集群吞吐。同一个 3× 吞吐提升在 $S=8$ 与 $S=32$ 上的来源不同：前者主要来自槽位利用率，后者已经把 KV 读取代价涨上去了。核算时要把 prefill 的算力（长 prompt 是 compute-bound）与 decode 的带宽分开看，否则容易把「涨价」算成「降本」。
- **追问**：什么时候 continuous batching 帮不上忙？
  - 要点：长度方差极小（所有请求等长，$\eta_{\text{static}} \approx 1$）；QPS 太低、队列常年空着（槽位本来就没被占满）；离线批处理场景能长时间攒大批、对 TTFT 不敏感，收益只剩长度方差那一项；以及被显存掐死的情形——如果 $\text{max\_len}$ 预留就吃光显存，只能先上 PagedAttention（[[inference-serving-03]]）。此外它完全不改变每步的计算量，模型本身的瓶颈（权重太大、量化没做）要靠别的招。

## 公司变体

`asked_at` 里是 **Anthropic、xAI、Mistral AI、NVIDIA、Together AI**，这题在两类公司里的问法差别很明显。

偏工程实现的一侧：

- **NVIDIA**：TensorRT-LLM / FasterTransformer 里这套机制就叫 in-flight batching，公开文档与示例把 scheduler 的配置项（最大 batch、KV 显存占比、chunked prefill 预算、抢占策略）摆在明面上。问法通常很具体：调度器的两个动作怎么落成状态机、KV 块池耗尽时选 recompute 还是 swap、batch 上限怎么和 TPOT SLO 对齐、CUDA graph 怎么处理逐 step 变化的 batch 形状。
- **Together AI**：主业是推理平台，公开工作集中在 kernel 与吞吐优化，问法偏「给定 QPS、长度分布、SLO，你怎么定 $S$、怎么估集群规模」这类可以现场算的题，能报出 $1/H_S$ 或 $\max/\text{mean}$ 的量级很占便宜。
- **Anthropic**：公开工程内容里 prompt caching（前缀 KV 复用）是重点能力，问 continuous batching 时常会顺势问到前缀共享的引用计数、共享前缀与抢占的相互作用、以及在线服务里怎么控尾延迟。
- **Mistral AI**：开源权重 + 自研/生态推理栈，问题多落在「显存预算怎么分给权重与 KV」「同样的卡怎么把并发做上去」这类落地权衡上。

偏原理推导的一侧：

- **xAI**：公开信息更多围绕超大规模集群与自研栈，这题可能从「集群规模给定时怎么把 GPU 利用率做上去」切进来，要求你把调度机制和规模化吞吐-延迟曲线讲清楚，而不是背实现细节。

两种问法都要准备：区分点从来不是「听说过 continuous batching」，而是能不能把「为什么快、快多少、什么条件下不快」用一组自洽的数字讲完——$1/H_S$ 的闭式解、$t_{\text{step}}(B) = a + bB$ 的成本结构、以及 KV 显存这道硬门槛。

## 相关题目

- [[inference-serving-01]]：prefill 计算受限、decode 带宽受限的两阶段划分，是本题第 2 节成本模型的前提。
- [[inference-serving-03]]：PagedAttention / vLLM 的 KV 分页管理；没有它，continuous batching 的新请求无处安放。
- [[inference-serving-14]]：chunked prefill，处理本题留下的「新请求 prefill 阻塞已有 decode」问题。

## 参考资料与归属

- [Continuous Batching in LLMs](https://outcomeschool.com/blog/continuous-batching-in-llms)，Amit Shekhar（Outcome School），2026-05-11（页面标注的发布时间）：static batching 的病灶、ride-share 类比、step 级工作流程、4 请求与 100 请求两套算例（44 vs 80 token/s）、2–4× 的经验区间，以及「prefill 会打断 decode」「batch 上限受显存约束」两条注意事项。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Woosuk Kwon 等（vLLM，SOSP 2023），2023-09-12（延伸）：摘要口径为「与 FasterTransformer、Orca 等系统相比，在相同延迟水平下把吞吐提升 2–4×，序列更长、模型更大、解码算法更复杂时提升更明显」，并强调 KV cache 的近零浪费与跨请求共享。**注意这个数字是 PagedAttention + iteration 级调度的合计口径**，其对比对象 Orca 本身已经实现了 iteration-level scheduling，所以 2–4× 不能全部记在 continuous batching 账上；正文第 3 节的 $H_S$ 与排空口径比值是对该区间的独立复算，不是对论文的复现。
- H100 SXM 参数取自 NVIDIA 官方规格（bf16 dense 989 TFLOP/s、HBM3 3.35 TB/s、80 GB）；$k_{\text{tok}}$、$H_S$、步长模型、吞吐与显存账目均按正文声明的口径复算，4 请求例子（200 步 / 800 槽位步 / 46.25%）与 100 请求例子（5000 步对 2750 步、1.82×）与源文一致，随机轨迹的 2.73× / 4.06× 由 step 级模拟得到（`numpy.random.default_rng(0)`、20 万条请求，可复现）。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
