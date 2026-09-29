---
type: question
id: coding-03
topic: 编程与数据结构
order: 3
question: 实现 KV cache 与单步 decode。
question_en: Implement a KV cache and single-step decoding.
asked_at: [Moonshot AI（Kimi）]
level: 进阶
tags: [kv-cache, decode, 实现题, 增量推理]
sources:
  - title: LLM 中的 KV Cache 是什么？
    url: https://outcomeschool.com/blog/kv-cache-in-llms
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: "RoFormer: Enhanced Transformer with Rotary Position Embedding（延伸）"
    url: https://arxiv.org/abs/2104.09864
    author: Su et al.
    published: 2021-04-20
related: [coding-02, inference-serving-01, inference-serving-03, inference-serving-05, coding-01]
updated: 2026-09-28
---

## 一句话答案

> KV cache 就是把每层已经算过的 $K$、$V$ 存下来：prefill 把整段 prompt 一次前向写进 cache，decode 每步只送 1 个 token，用它算 $q$、从 cache 读出全部 $K/V$ 算注意力、再把这一步的 $k,v$ 追加进去。
> 省下的是投影的重复计算：prompt 长 $P$、生成 $T$ 个 token 时，无 cache 的 token 前向次数是 $\sum_{t=1}^{T}(P+t-1)=TP+T(T-1)/2$，有 cache 是 $P+T-1$——$P=2,T=99$ 时是 5049 对 100。但注意力的规模仍随上下文增长，decode 不是 $O(1)$。
> 最容易翻车的是位置编码：decode 时 $q$ 只有 1 个位置、$k$ 有 $T$ 个，RoPE 必须按**绝对位置**分别施加，cache 里存旋转后的 $k$（或存未旋转的、取出后按绝对位置旋转且只旋转一次）。

## 面试官在考什么

- **能不能说清省了什么**：省的是 $K/V$ 投影的重复计算，以及随之而来的权重读取与 KV 读写；不是把每步的注意力变成常数。
- **数据结构与契约**：每层一块 $(B,H_{kv},T_{\max},d_{head})$，写入位置等于当前已缓存长度；预分配 + 索引写入，而不是每步 `cat`。
- **位置编码**（本题最常翻车的地方）：RoPE 的旋转角只依赖该 token 的绝对位置，$q$、$k$ 各自旋转；cache 里存旋转后的 $k$ 还是存原始 $k$，两条路都可以，但重复施加或漏施加都是错的。
- **有没有正确性证据**：拿「不带 cache 的整段前向」当基准，断言逐元素误差小于 $10^{-5}$，而不是靠肉眼看输出通顺。
- **显存与工程账**：$2LH_{kv}d_{head}\times\text{bytes}\times T$，以及 PagedAttention、prefix cache、KV 量化这条优化线各自解决什么。

常见错误答案：

- 「KV cache 让 decode 变成 $O(1)$」。每步仍要对全部历史 $K/V$ 做注意力，这部分随上下文线性增长；省掉的是投影。
- 「把新的 $k/v$ 拼在 cache 里就行」。拼的位置错了语义就全错；而且每步 `cat` 会反复分配并拷贝整段，实测比索引写入慢 3 个数量级（见第 4 节）。

## 原理与推导

### 1. 先算清省下了什么

自回归解码每步只新增 1 个 token，但这个 token 需要全部历史的 $K/V$。设 prompt 长 $P$、要生成 $T$ 个 token，第 $t$ 步若从头重算，输入长度是 $P+t-1$：

$$\text{无 cache 的 token 前向次数}=\sum_{t=1}^{T}(P+t-1)=TP+\frac{T(T-1)}{2},\qquad \text{有 cache}=P+(T-1)$$

$P=2,T=99$ 时是 5049 对 100，约 50 倍。源文给的 5049 对 101 里，5049 完全可复现，101 是把 prefill 之外又数了 99 次单 token（按「第 1 步处理 $P$ 个位置」的口径应是 $2+98=100$），差 1 次不影响量级，但口径要在答案里交代清楚。

线性层部分的总 FLOPs 因此从 $O(T^2)$ 降到 $O(T)$；attention 部分每步仍是「1 个 query 对 $T$ 个 key」，随上下文线性增长。这就是「cache 不是 $O(1)$」的来源，也是长上下文下 decode 仍然慢的原因。

### 2. 数据布局与三步契约

每层一块预分配张量，形状 $(B,H_{kv},T_{\max},d_{head})$：$H_{kv}$ 是 KV 头数而不是 query 头数（GQA 的 KV 头是共享的，见 [[coding-02]]）。契约只有三步（签名见下面的代码块）：

1. `write(layer, k, v)`：把本步的 $k/v$ 写到 $[\text{len},\text{len}+T_{\text{new}})$；
2. `read(layer, T_new)`：返回 $[0,\text{len}+T_{\text{new}})$ 的**视图**——包含刚写入的位置，所以 attention 直接读存储，不需要 `cat`；
3. `advance(T_new)`：所有层都写完之后统一推进 `len`。

第 2 步是关键设计：让 read 覆盖到刚写进去的槽位，把「拼接」变成「按长度切片」。chunked prefill 只是同一路径上 $T_{\text{new}}>1$ 的特例：此时需要按位置掩码 `k_pos > q_pos`（[[inference-serving-14]]）；decode 时 $T_{\text{new}}=1$，因果性由「cache 里只有 $\le$ 当前位置的 $K/V$」天然保证，**不需要 mask**——一旦发现自己在 decode 里造掩码，通常说明写入位置错了。另一个不报错的错法是把新 $k/v$ 拼到 head 维：$(B,H,T,d)$ 布局下 head 是 `dim=1`、序列是 `dim=-2`，写成 `cat(dim=1)` 头数会变、位置全错，而且形状合法不会抛异常。

### 3. 位置编码：绝对位置、相对依赖

RoPE 用旋转矩阵编码绝对位置 $i$：$q_i \to R_i q_i$，$k_j \to R_j k_j$，而 $\langle R_i q_i, R_j k_j\rangle$ 只依赖 $i-j$。论文的口径正是这两句：用旋转矩阵编码绝对位置，同时在 self-attention 里引入显式的相对位置依赖，并具备序列长度灵活性与随相对距离衰减的 token 依赖（见参考资料）。

因为位置信息已经烘焙在旋转结果里，**cache 里存旋转后的 $k$ 是最省事的做法**。三种写法：

| 写法 | 正确性 | 说明 |
| --- | --- | --- |
| 存旋转后的 $k$，取出直接用 | 正确 | 本实现采用；不需要额外存位置下标 |
| 存原始 $k$，取出后按各 token 的绝对位置旋转 | 正确 | 要多存一个位置数组；适合 KV 量化/重算类优化 |
| 取出后按当前 `pos` 再旋转一次；或 $q$ 用「chunk 内位置」（decode 时恒为 0） | **错** | 前者重复施加，相对角度变成 $2i-2j$ 一类；后者破坏相对依赖，所有历史位置的分数退化成同一个值（第 4 节有实测表） |

学习式绝对位置编码（`nn.Embedding(max_len, d)`）在 decode 里同样必须传绝对 `pos`，否则新 token 永远用的是位置 0 的向量；ALiBi 则不走旋转，直接在 score 上加 $-m\cdot|i-j|$ 的线性偏置，实现更简单、外推行为不同。位置编码错了的输出**往往仍然通顺**，所以只能靠与全量前向逐元素比对来发现。

### 4. 显存账与优化方向

每层 $K$ 和 $V$ 各 $H_{kv}\cdot d_{head}$ 个元素，整体是 $2LH_{kv}d_{head}\times\text{bytes}\times T$。按 LLaMA-3-70B 口径（$L=80$、$H_{kv}=8$、$d_{head}=128$、bf16 即 2 B）：

$$2\times 80\times 8\times 128\times 2=327{,}680\ \text{B}=320\ \text{KiB/token}$$

| 上下文 | 每序列 | 8 并发 | 32 并发 | 128 并发 |
| --- | --- | --- | --- | --- |
| 8k | 2.50 GiB | 20 GiB | 80 GiB | 320 GiB |
| 32k | 10.00 GiB | 80 GiB | 320 GiB | 1280 GiB |

同形状换成 MHA（$H_{kv}=64$）是 2.5 MiB/token，MQA（$H_{kv}=1$）是 40 KiB/token，GQA-8 正好是 MHA 的 1/8；MLA 一类把 K/V 压成 latent 的方案量级更低。1 GiB 显存只能装 3276.8 个 token 的 GQA-8 cache，所以「能同时服务多少请求」几乎完全由 KV 管理方式决定（[[inference-serving-08]]）。

decode 每步要读的字节是「权重 + $b\cdot T\cdot 320$ KiB」，H100 3.35 TB/s 口径下：$b=1,T=32\text{k}$ 时 KV 项 3.2 ms，而权重 141 GB 就要 42.1 ms（权重按 $70.6\text{B}\times2$ B 计，[[inference-serving-01]] 用 140 GB 口径给 41.8 ms，差 0.7%）；$b=32$ 时 KV 项 102.6 ms，反超权重项。所以长上下文 + 大 batch 下，decode 的带宽是花在 KV 上的（[[inference-serving-01]]）。

工程优化三条线：**PagedAttention** 把 cache 切成固定大小的 block，用页表消除碎片（[[inference-serving-03]]）；**prefix cache** 复用公共前缀，命中就整段引用同一批物理块（[[inference-serving-05]]）；**KV 量化**把每个元素的字节数降下来（[[inference-serving-06]]）。三者都建立在本节这套「每层一块、按位置索引」的布局上。

### 5. 从这段代码到生产：还差什么

这段实现能跑通、能对齐误差，但离上线差着一整层引擎：它**按 $(B,H_{kv},T_{\max},d_{head})$ 静态预留**，32k 上下文 × 32 并发就是 320 GiB 的显存账，而真实在跑的并发通常远低于预留值，所以要换成「按需分配 + 分块 + 引用计数」（就是 PagedAttention 解决的事，[[inference-serving-03]]）；它**一次只服务一个请求且每步同步**，没有 continuous batching 把不同请求的 decode 拼成一批（[[inference-serving-02]]），也没有 chunked prefill 让长 prompt 与 decode 混批（[[inference-serving-14]]）；它**每步几十到上百个 kernel**，本机实测的固定开销约 7 ms/步只是这件事的极端版本，GPU 上对应的是 kernel launch，所以引擎要把整步捕获成 CUDA graph。此外它没有 prefix cache（[[inference-serving-05]]）、没有 KV 量化（[[inference-serving-06]]）、没有抢占/换出与计算-传输重叠；张量并行下 KV 头还要按 TP 度切分或复制（[[inference-serving-07]]）；bf16 与 fp32 的输出差 $4.4\times10^{-2}$，精度与容差策略必须先定；长度超过训练窗口要换 RoPE 的 base（NTK/YaRN 一类），而 cache 里存的是按旧 base 旋转的 $k$，改 base 就得整段重算。最后，`cache` 溢出必须抛错而不是静默截断（实测抛 `OverflowError: KV cache 溢出：14 > 13`），生产里对应的动作是抢占、换出到 host 或丢弃最老前缀，那是引擎层的策略，实现层至少要能检测。

## 数值与代码验证

下面的第一、三个代码块与脚本 `.work/coding03_doc.py` 逐字节一致（分别是 RoPE/KV cache 部分与驱动）；第二个是 `Attention.forward` 里与 cache 有关的那几行，省略了投影与输出投影，完整版在同一脚本里。环境：torch 2.12.0，CPU，fp32，`python3 .work/coding03_doc.py` 实测 1.45 s 跑完，重复运行输出逐行一致。模型：$d_{model}=256$、8 个 query 头、2 个 KV 头（GQA）、$d_{head}=32$、4 层，3.94M 参数。

```python
# --------------------------------------------------------------------------- RoPE

def rope_tables(d_head, max_len, base=10000.0):
    """返回 (cos, sin)，形状均为 (max_len, d_head/2)；角度 = 绝对位置 i × base^(-2j/d_head)。"""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    ang = torch.outer(torch.arange(max_len).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin, pos):
    """x: (B,H,T,d) -> 相邻两维配对旋转；pos = 这段 chunk 第一个 token 的**绝对**位置。"""
    T = x.shape[-2]
    c, s = cos[pos:pos + T][None, None], sin[pos:pos + T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


# --------------------------------------------------------------------------- Cache

class KVCache:
    """每层一块预分配张量 (B, H_kv, T_max, d_head)；契约三步走：write -> read -> advance。

    read 返回的视图**包含本步刚写入的 k/v**，所以 attention 不需要 cat，直接读存储。
    """

    def __init__(self, n_layers, batch, n_kv_heads, d_head, max_len,
                 dtype=torch.float32, device="cpu"):
        shape = (n_layers, batch, n_kv_heads, max_len, d_head)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.len, self.max_len = 0, max_len

    def write(self, layer, k, v, at=None):
        start = self.len if at is None else at         # at=0 就是「新 token 写到第 0 位」那个 bug
        end = start + k.shape[-2]
        if end > self.max_len:
            raise OverflowError(f"KV cache 溢出：{end} > {self.max_len}")
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v

    def read(self, layer, T_new=0):
        end = self.len + T_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n):
        self.len += n

    @property
    def bytes(self):
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


def repeat_kv(x, n_rep):
    """(B, H_kv, T, d) -> (B, H, T, d)，GQA 的 KV 头广播。"""
    if n_rep == 1:
        return x
    B, H, T, d = x.shape
    return x[:, :, None].expand(B, H, n_rep, T, d).reshape(B, H * n_rep, T, d)
```

注意力层里与 cache 有关的两个动作（`pos` 是绝对位置，不是 chunk 内偏移）：

```python
        if rope_on_query:
            q = apply_rope(q, self.cos, self.sin, pos)      # q 按自己的绝对位置旋转
        k = apply_rope(k, self.cos, self.sin, pos)          # k 旋转后再入 cache
        if cache is None:
            k_all, v_all, k_len = k, v, T
        else:
            cache.write(layer, k, v, at=0 if write_at_zero else None)
            k_all, v_all = cache.read(layer, T_new=T)       # 视图，含本步新写入的位置
            k_len = k_all.shape[-2]
            if double_rope:                                 # 反例 2：对已旋转的 k 再转一次
                k_all = apply_rope(k_all, self.cos, self.sin, pos)
        att = (q @ repeat_kv(k_all, self.n_rep).transpose(-1, -2)) * self.scale
        if T > 1:   # decode 时 T=1，因果性由「cache 只含 ≤ 当前位置」天然保证，无需 mask
            q_pos = torch.arange(pos, pos + T).view(T, 1)
            k_pos = torch.arange(k_len).view(1, k_len)
            att = att.masked_fill(k_pos > q_pos, float("-inf"))
```

驱动就是「prefill 一次 + 单步循环」，与全量前向做逐元素对比：

```python
    def run_decode(batch=1, seq=ids, **kw):
        """prefill P 个 + 单步 decode N 次，返回拼接后的输出与 cache。"""
        c = KVCache(cfg.n_layers, batch, cfg.n_kv_heads, cfg.d_head, cfg.max_len)
        outs = []
        with torch.no_grad():
            outs.append(model(seq[:, :P], cache=c, pos=0, **kw))
            c.advance(P)
            for t in range(N):
                outs.append(model(seq[:, P + t:P + t + 1], cache=c, pos=P + t, **kw))
                c.advance(1)
        return torch.cat(outs, dim=1), c
```

真实输出（prompt 11 个 token、生成 7 个；完整输出见 `.work/coding03_doc.out`）：

```text
配置 d_model=256 heads=8 kv_heads=2 d_head=32 layers=4 参数量=3.94M

1) 正确性：无 cache 全量前向 vs prefill + 7 步 decode
------------------------------------------------------------------------
逐元素最大绝对误差 = 9.537e-07（相对 2.667e-07）；断言 < 1e-5：PASS
cache.len = 18（应为 18）；预分配 1048576 B = 1024.0 KiB（4 层 × (K+V)）

2) 反例：三种经典写法各错到什么程度（相对同一基准）
------------------------------------------------------------------------
  新 token 写到第 0 位            最大绝对误差 4.143e-01  -> 是正确实现的 434376 倍
  对 cache 里的 k 重复施加 RoPE     最大绝对误差 1.957e-01  -> 是正确实现的 205194 倍
  decode 时 q 漏掉绝对位置          最大绝对误差 2.083e-01  -> 是正确实现的 218382 倍
  忘记 advance()               cache.len 停在 0（正确 12），该步输出差 2.076e+00

3) RoPE 的相对位置性质：<R_i q, R_j k> 只依赖 i-j（固定 j=8）
------------------------------------------------------------------------
   i   i-j              正确（q 用绝对位置 i）          bug（q 恒当位置 0）
   8     0                   3.305814               4.518574
  13     5                   1.534865               4.518574
  45    37                  -1.463701               4.518574
  同时平移 +  1：+1.698513269 -> +1.698513389，绝对差 1.19e-07
  同时平移 + 64：+1.698513269 -> +1.698512197，绝对差 1.07e-06
  同时平移 +200：+1.698513269 -> +1.698511600，绝对差 1.67e-06

4) 边界：chunked prefill / batch / bf16 / 溢出保护
------------------------------------------------------------------------
  chunk= 1 最大误差 1.192e-06
  chunk= 3 最大误差 9.537e-07
  chunk= 4 最大误差 7.153e-07
  chunk=11 最大误差 9.537e-07
  batch=3 最大误差 1.192e-06；cache.len=18
  bf16 cache vs bf16 全量（同精度）：0.000e+00
  bf16 全量 vs fp32 全量（混精度）：4.397e-02（输出量级 3.58）
  cache 字节数 fp32 1048576 B -> bf16 524288 B
  超过 max_len：抛出 OverflowError（KV cache 溢出：14 > 13）

所有断言通过。
```

三条读数：正确实现与全量前向的误差 $9.5\times10^{-7}$，而三个经典 bug 的误差是 $10^{-1}$ 量级、差 20 万倍以上——**没有这条断言，位置编码的错误根本发现不了**；chunked prefill 切 1/3/4/11 都落在同一个误差量级，说明同一条 forward 路径兼容任意切分；bf16 下 cache 路径与 bf16 全量前向逐位相同（0.000e+00），而 bf16 与 fp32 之间差 $4.4\times10^{-2}$，所以容差必须按精度分别定，不能对所有情况都写 $10^{-5}$。

耗时实验（`.work/coding03_bench.py`，同一个模型配置换成 $d_{model}=512$、6 层、23.34M 参数；单线程、取多轮 `process_time` 最小值。这台机器同时跑着别的任务，load average > 20，绝对值有 ±20% 波动，趋势可信；下面的输出块是摘录，省略了若干重复行，完整输出见 `.work/coding03_bench.out`）：

```text
1a) 单步成本随输入长度的变化（无 cache 的第 t 步要算 t+1 个位置）
    输入长度 L       单次前向    每 token
        16   14.47 ms   0.904 ms
       256  131.27 ms   0.513 ms
最小二乘拟合：单步耗时(L) ≈ 0.4882 ms/token × L + 3.72 ms（最大残差 5.01 ms）

1b) 端到端：无 cache（每步重算全长） vs 有 cache（prefill + 单步 decode）
   步数     无 cache 总耗时         每步      有 cache 总耗时         每步      实测加速比     token 前向次数比
   50       1312.0 ms    26.24 ms         402.9 ms     8.06 ms       3.3x           31.4x
  100       3926.6 ms    39.27 ms         791.3 ms     7.91 ms       5.0x           57.3x
  200      11791.4 ms    58.96 ms        1451.8 ms     7.26 ms       8.1x          107.9x
2) 微基准：追加一个 token 的成本 —— cat 重建 vs 整段 copy_ vs 单槽索引写入
   B    t(已有)       cat 重建     整段 copy_       单槽写入    cat/单槽    copy/单槽     cat 拷贝量
   1     1024      83.5 µs      90.2 µs    8.40 µs      9.9x      10.7x     4.00 MiB（单槽只碰 4 KiB）
   1    16384   31406.8 µs    5019.9 µs   25.85 µs   1215.1x     194.2x    64.00 MiB（单槽只碰 4 KiB）
  32    16384  949685.5 µs  162908.3 µs   66.26 µs  14331.8x    2458.5x  2048.12 MiB（单槽只碰 128 KiB）
```

加速比随步数单调增长（3.3x → 5.0x → 8.1x），方向与 token 前向次数比（31.4x → 57.3x → 107.9x）一致，但只有它的约 1/10。原因就在同一份实测里：无 cache 每步 26.2 ms → 59.0 ms 随 $t$ 线性上升，有 cache 每步恒定 7.3–8.1 ms，而单步耗时对 $L$ 的拟合截距是 3.72 ms、有 cache 的单 token 前向实测 7.15 ms——也就是说这个 23M 参数的玩具模型上，每步约 100 个小算子的 Python/dispatch 开销（约 7 ms）与真正的计算同量级，两臂都要按步付这笔钱，于是把计算项的差距压扁了。真实 70B 每 token 的计算量比它大 3 个数量级，同一套 forward 下 work 项才是主导——所以小模型上测不出 KV cache 的收益，是实验设计问题，不是 cache 本身的问题。

微基准那一栏是整个实现里最值得记住的数字：$t=16384$ 时 cat 一次要 31.4 ms（$B=1$）到 0.95 s（$B=32$），因为它每步都重新分配并拷贝整段；把拷贝本身（`copy_` 到预分配缓冲）单独测出来是 5.0 ms / 162.9 ms，说明成本来自 $O(t)$ 的搬运而不是分配器；而单槽索引写入恒定在 8–66 µs。所以「预分配 + 按位置写入」不是风格偏好，是 decode 能否跑到毫秒级的门槛。

## 常见追问

- **追问**：为什么 decode 是带宽受限、prefill 是计算受限？
  - 要点：prefill 一次处理 $T$ 个 token，权重被 $T$ 个 token 复用，算术强度随 $T$ 上升；decode 每步只有 1 个 token，却要读完整份权重和全部历史 KV，算术强度只有个位数 FLOPs/byte，远低于 H100 的 roofline 拐点 295 FLOPs/byte（[[inference-serving-01]]）。按 141 GB 权重与 3.35 TB/s 算，单请求每步的权重读取下界就是 42.1 ms。
- **追问**：chunked prefill 与 cache 怎么交互？
  - 要点：chunked prefill 就是把一次 prefill 拆成几段连续调用同一条 forward，每段写入 $[\text{len},\text{len}+T_{\text{chunk}})$ 并推进 `len`；段内 $T_{\text{chunk}}>1$ 所以要位置掩码，段间靠 cache。好处是长 prompt 不再独占一步、可以和 decode 混批（[[inference-serving-14]]）。注意每个 chunk 的 `pos` 必须是**全局**起始位置。
- **追问**：beam search 下 cache 怎么写？多轮对话里哪部分能复用？
  - 要点：beam 的分支会重排，cache 必须按 beam 索引 gather 重排（`cache.k[:, beam_idx]`），每条 beam 的 `len` 相同，分叉后的共享块用引用计数管理。多轮复用只认「逐 token 完全相同的前缀」，因为 KV 是「前缀 + 位置」的函数，改一个字后面全废；chat template、系统提示、工具定义、模型/tokenizer/LoRA 版本都要进 cache key（[[inference-serving-05]]）。

## 公司变体

`asked_at` 只记录了 Moonshot AI（Kimi）问过这道题（本仓库 README 的出现于列表）；下面只按公开技术产出说明倾向，不是面试记录。

- **Moonshot AI（Kimi）**：长上下文是这家产品的主线，公开的 Mooncake 论文把 KVCache 当一等资源，做了以 KV 为中心的分离式服务架构与全局 KV 池。所以追问大概率偏**工程实现**：$(B,H_{kv},T_{\max},d_{head})$ 的布局与写入位置、cache 溢出/抢占时怎么办、跨请求的前缀复用边界、长上下文下 decode 的带宽账，而不是现场推旋转矩阵。把第 4 节的数字和三条优化线准备好，比背公式更有用（[[inference-serving-05]]、[[inference-serving-15]]）。

## 相关题目

- [[coding-01]] 与 [[coding-02]]：cache 路径里的 `att = softmax(q@k^T·scale) @ v` 来自前者；$H_{kv}$ 与 `repeat_kv` 来自后者，MHA 2.5 MiB/token 与 GQA-8 320 KiB/token 的 8 倍差也在那一题算过。
- [[inference-serving-01]] 与 [[inference-serving-03]]：为什么 decode 每步的固定成本是「读一遍权重」，以及 cache 的块化管理怎么消除碎片。
- [[inference-serving-06]]、[[inference-serving-08]]、[[inference-serving-14]]：KV 量化的字节账、显存估算的统一口径、chunked prefill 的调度细节。

## 参考资料与归属

- Amit Shekhar (Outcome School)，*LLM 中的 KV Cache 是什么？*，<https://outcomeschool.com/blog/kv-cache-in-llms>：只 cache K/V 而不 cache Q 的理由、prefill/decode 两阶段与 cache 的关系、「100 个 token 约 50 倍」的对比口径、以及 cache 增长带来的显存取舍。
- Jianlin Su 等，*RoFormer: Enhanced Transformer with Rotary Position Embedding*（延伸），2021-04-20，<https://arxiv.org/abs/2104.09864>：RoPE 用旋转矩阵编码绝对位置、同时在 self-attention 中引入显式相对位置依赖，以及序列长度灵活性、token 依赖随相对距离衰减这两条性质。

源文的 5049 次 Key/Value 计算与约 50 倍收益可以复现，但「有 cache 一侧的 101 次」与 5049 的口径差 1（本文按 $P+(T-1)=100$ 计，见「原理与推导」第 1 小节）；除此之外的所有数字——误差、耗时、显存、roofline——都是按本仓库统一常数（LLaMA-3-70B：80 层、$H_{kv}=8$、$d_{head}=128$、bf16 即 320 KiB/token；H100 SXM5 bf16 稠密 989 TFLOPs、HBM 3.35 TB/s、roofline 拐点 295 FLOPs/byte）自行复算或实测的结果：数值与耗时来自 `.work/coding03_doc.py` 与 `.work/coding03_bench.py`，显存与带宽账来自 `.work/coding03_mem.py`，原始输出分别在 `.work/coding03_doc.out` 与 `.work/coding03_bench.out`。测量设备是 AMD Ryzen 9 8945HX（单线程、fp32、与其它任务共享，load average > 20），CPU 耗时不可外推到 H100；涉及 H100 的数字都是按 3.35 TB/s 带宽下界算出来的。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
