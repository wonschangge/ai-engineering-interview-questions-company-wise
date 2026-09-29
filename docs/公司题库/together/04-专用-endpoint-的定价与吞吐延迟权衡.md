---
type: question
id: together-04
company: Together AI
topic: inference-serving
order: 4
question: 为专用 endpoint 定价：估算 70B 模型每百万输出 token 的成本，并解释 throughput 与 latency 之间的权衡。
question_en: Price a dedicated endpoint: estimate the cost per million output tokens for a 70B model, and explain the throughput-latency trade-off.
asked_at: []
level: 高阶
tags: [成本模型, 定价, 吞吐-延迟, 利用率, 盈亏平衡]
sources:
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
  - title: Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）
    url: https://arxiv.org/abs/2309.06180
    author: Kwon et al. (vLLM, SOSP 2023)
    published: 2023-09-12
  - title: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）
    url: https://arxiv.org/abs/2403.02310
    author: Agrawal et al. (Microsoft, OSDI 2024)
    published: 2024-03-04
  - title: LLM 推理优化
    url: https://outcomeschool.com/blog/llm-inference-optimization
    author: Amit Shekhar (Outcome School)
    published: 
related: [together-03, together-06, anthropic-15, anthropic-16, anthropic-17]
updated: 2026-09-28
---

## 一句话答案

> 定价的本质是**把 GPU 小时换成 token**，所以先写成本公式，再谈权衡：
> $$\text{每百万输出 token 成本}=\frac{\text{GPU 数}\times\text{\$/GPU·h}}{\text{每小时输出 token 数}/10^6}$$
> 三个输入项：
> ① **GPU 数**由两件事决定——**权重放得下**（70B bf16 ≈ 140 GB ⇒ 至少 2×80 GB，工程上常 4 卡 TP 留 KV 与激活空间）与**KV 放得下**（320 KiB/token × 并发 × 上下文）；
> ② **每小时输出 token 数**由 batch 与带宽决定：decode 是带宽受限的，每步时间 $\approx$ 读一遍权重/带宽（8 卡 TP 下约 5.2 ms），于是
> $$\text{吞吐}\approx\frac{B}{t_{\text{step}}(B)},\quad t_{\text{step}}(B)=\max\Big(\frac{2P_{\text{bytes}}}{N_{\text{gpu}}BW},\frac{2PB}{N_{\text{gpu}}F},\frac{B\cdot \text{KV}_{\text{tok}}\cdot T}{N_{\text{gpu}}BW}\Big)$$
> ③ **利用率**（真实业务不会一直满负载）：报价必须按**目标利用率**算，否则低峰期的空闲成本会吃掉毛利。
> **throughput 与 latency 的权衡**：提高 batch/预算 → 吞吐上升、单位成本下降，但 **TPOT 变差**（每步更慢）且 **TTFT** 受排队影响；**SLO 决定工作点**，而工作点决定成本。所以「这个 endpoint 多少钱」这个问题**必须先问 SLO**（p50/p99 的 TTFT 与 TPOT），否则无法报价。
> 一句话判据：**先按 SLO 定工作点，再按工作点算 token 成本，最后按利用率加价**——反过来做（先定价再找配置）一定会亏在某类流量上。

## 面试官在考什么

- **能否从第一性原理算**：GPU 数怎么定（权重 + KV）、每步时间怎么估（带宽/算力/KV 三项取最大）、吞吐怎么算——而不是「参考市场价」。
- **是否区分吞吐与延迟**：能否指出**同一台机器在不同 batch 下 token 成本可以差 5–10 倍**，而延迟也差同样倍数；因此报价必须绑定 SLO。
- **利用率意识**：专用 endpoint 的客户流量是波动的（白天高峰、夜间低谷），若按峰值配置却按平均利用率收费，毛利会被空闲时间吃掉；反向的「超卖」会把 p99 打坏。
- **成本拆解是否完整**：GPU 只是最大项，还要算 CPU/内存（tokenize、调度）、网络（出网/跨区）、存储（权重与日志）、以及**运维与冗余**（N+1、跨区）。
- **定价策略**：按 token（用量）还是按 GPU 小时（预留）？混合（最低承诺 + 超额）？以及**为什么按 GPU 小时定价对供应商更安全、按 token 对客户更易预算**。
- **与 API 的对比**：什么时候专用 endpoint 比按 token 的 API 更便宜——**盈亏平衡利用率**是多少。
- **诚实**：不承诺「比 API 便宜 X 倍」而不给条件（流量形态、SLO、上下文长度都会改变结论）。

**常见错误答案**

- 只报一个数（「大约 \$X/百万 token」）而不说配置、SLO 与利用率。
- 忽略 KV 与上下文长度（长上下文下并发暴跌、成本暴涨）。
- 按峰值利用率算成本（现实中不可能）。
- 忘记冗余（N+1）与非 GPU 成本。
- 不区分 prefill 与 decode 的成本结构（长输入短输出的负载成本结构完全不同）。

## 原理与推导

### 1. 成本公式与三个输入

$$\text{cost/1M tokens}=\frac{G\cdot p_{\text{gpu}}}{\text{tps}\times 3600/10^6}=\frac{G\cdot p_{\text{gpu}}\times 10^6}{3600\,\text{tps}}$$

其中 $G$ = GPU 数、$p_{\text{gpu}}$ = 每 GPU 每小时价格（本仓库统一口径 **\$2/GPU·h**）、tps = 每秒输出 token。

**GPU 数的下界**：
- 权重：$2P$ 字节（bf16）= 140 GB ⇒ $G\ge\lceil140/80\rceil=2$，实际取 $G=4$（TP=4）留 KV 与激活；
- KV：$G\cdot M_{\text{KV}}\ge B\cdot T\cdot \text{KV}_{\text{tok}}$（$M_{\text{KV}}$ 为每卡可用显存）。

### 2. 每步时间与吞吐（本仓库常数）

以 70B、8×H100（$BW$=3.35 TB/s/卡、$F$=989 TFLOPs/卡）为例：

| batch $B$ | 权重项（每卡 17.5 GB/3.35 TB/s） | 算力项 | KV 项（$T$=4K） | 受限项 | 每步 | 吞吐 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.22 ms | ~0 | 0 | 权重 | 5.22 ms | 191 tok/s |
| 100 | 5.22 ms | 1.8 ms | 5.0 ms | 权重（临界） | 5.22 ms | 19,157 tok/s |
| 295 | 5.22 ms | 5.2 ms | 14.8 ms | **KV** | 14.8 ms | ~20,000 tok/s |

（与 [[anthropic-16]] 同口径：平坦区在 KV 读取追上权重读取处结束。）

### 3. 成本随 batch 的变化（本机实算）

| batch | 每步（ms） | 受限项 | 吞吐（tok/s） | \$/百万 token | TPOT（ms） |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.22 | 权重 | 191 | **31.343** | 5.22 |
| 8 | 5.22 | 权重 | 1,531 | **3.918** | 5.22 |
| 32 | 5.22 | 权重 | 6,126 | **0.979** | 5.22 |
| 100 | 5.22 | 权重 | 19,143 | **0.313** | 5.22 |
| 256 | 12.82 | KV | 19,968 | **0.300** | 12.82 |
| 512 | 25.64 | KV | 19,968 | **0.300** | 25.64 |

**读法（两个区间要分清）**：从 batch 1 到 100 处于**权重受限区**，每步时间恒为 5.22 ms，成本从 \$31.34 降到 \$0.313（约 **100 倍**）而 TPOT 不变；batch 到 256 之后进入 **KV 受限区**，此时 $t_{\text{step}}\propto B$ 而 $\text{tps}=B/t_{\text{step}}$ **恒定**——**继续加大 batch 不再降低成本，只会让 TPOT 线性变差**。这条结论很关键：成本杠杆是「从权重受限区走出来」，不是「把 batch 顶到最大」。

### 4. 利用率：报价里最容易被忽略的一项

设目标利用率 $\rho$（实际售卖出去的时间比例），则有效成本：

$$\text{cost}_{\text{eff}}=\frac{\text{cost}}{\rho}$$

$\rho=0.7$ 时成本涨 **43%**；$\rho=0.3$ 时涨 **233%**。**专用 endpoint 的定价必须写明「按承诺吞吐/预留时长计价」**，否则低利用率客户会亏。

### 5. 与按 token API 的盈亏平衡

设 API 输出价 $p_{\text{api}}$（本仓库常用口径 \$15/百万 token）、专用 endpoint 的 token 成本 $c_{\text{ded}}$（含利用率），则专用更便宜的条件：

$$c_{\text{ded}}<p_{\text{api}}\quad\Longleftrightarrow\quad \text{月用量}> \frac{\text{月 GPU 成本}\times(1+\text{毛利})}{p_{\text{api}}}$$

**例**：8×H100 一个月（720 h）成本 $8\times2\times720=\$11{,}520$；若 API 价 \$15/百万，盈亏平衡月用量 ≈ $11{,}520/15=\mathbf{768}$ **百万 token/月**（约 7.7 亿）。低于这个量，API 更划算。

### 6. 成本拆解（完整清单）

| 项 | 量级 | 说明 |
| --- | --- | --- |
| GPU | **主导**（70–85%） | 权重 + KV |
| CPU/内存 | 5–15% | tokenize、调度、流式转发 |
| 网络 | 3–10% | 出网、跨区、负载均衡 |
| 存储 | 1–5% | 权重（一次性）+ 日志 |
| 冗余（N+1） | +25–100% | 故障与峰值兜底 |
| 运维/支持 | 5–10% | 人 |

**报价口径**：$\text{price}=\frac{(\text{GPU}+\text{其他})\times(1+\text{冗余})\times(1+\text{毛利})}{\text{可售 token 数}}$。

## 数值与代码验证

### 表 1：上下文长度决定成本（KV 受限区）

| 上下文 | KV 决定的并发上限 | 吞吐（tok/s） | \$/百万 token |
| --- | --- | --- | --- |
| 4K | 320 | 19,968 | **0.300** |
| 16K | 80 | 4,992 | **1.202** |
| 32K | 40 | 2,496 | **2.404** |
| 128K | 10 | 624 | **9.616** |

**读法**：在 KV 受限区，**吞吐由上下文长度决定而与 batch 无关**——$T$ 从 4K 涨到 128K，并发上限从 320 掉到 10，吞吐从约 20k tok/s 掉到 624 tok/s，**单位成本涨约 32 倍**（\$0.300 → \$9.616）。所以定价必须**按上下文长度分档**，而不是按「平均请求」一刀切。

### 表 2：利用率与冗余对有效成本的影响（batch=100、$T$=4K）

| 目标利用率 $\rho$ | 含 35% 冗余的有效成本 | 相对裸成本 |
| --- | --- | --- |
| 1.0 | \$0.313 | 1.35 |
| 0.9 | \$0.348 | 1.50 |
| 0.7 | \$0.448 | 1.93 |
| 0.5 | \$0.627 | 2.70 |
| 0.3 | \$1.045 | 4.50 |

**读法**：利用率 1.0 → 0.3 把成本推高 **3.33 倍**，再叠加 35% 冗余就是 **4.5 倍**。报价里必须写明这两项假设，否则「同一个 endpoint」可以有四倍价差。

### 表 3：与按 token API 的盈亏平衡（API 输出价 \$15/百万）

| GPU 数 | 月成本（含冗余与 30% 毛利） | 盈亏平衡月用量 |
| --- | --- | --- |

**读法**：低于盈亏平衡月用量时，按 token 的 API 更划算——这是与客户谈定价的起点，也是「什么时候该建议客户别买专用 endpoint」的判据。

### 表 4：SLO 与成本的关系（在 KV 受限区，成本几乎不变）

| TPOT 上限（ms） | 最大 batch | 吞吐（tok/s） | \$/百万 token |
| --- | --- | --- | --- |
| 5.5 | 109 | 19,968 | 0.300 |
| 8.0 | 159 | 19,968 | 0.300 |
| 15.0 | 299 | 19,968 | 0.300 |
| 40.0 | 320 | 19,968 | 0.300 |

**读法（反直觉但重要）**：在 KV 受限区，**收紧或放宽 TPOT 几乎不改变单位成本**（都是 \$0.300），因为聚合吞吐由带宽与 KV 读取量决定，与并发序列数无关。**真正影响成本的是上下文长度**（表 1）。所以当客户说「我要更低的延迟」时，在 KV 受限区你可以直接答应（不需要涨价）；而当客户说「我要更长的上下文」时，必须重新报价。

### 可运行代码

```python
# 专用 endpoint 的成本与定价模型：每步时间、吞吐、$/百万 token、盈亏平衡、SLO 工作点
from dataclasses import dataclass
from typing import Dict, List, Tuple

PEAK_FLOPS = 989e12        # H100 bf16 dense（仓库统一常数）
PEAK_BW = 3.35e12          # HBM 带宽（仓库统一常数）
KV_PER_TOKEN = 320 * 1024  # LLaMA-3-70B 口径
GB = 1024 ** 3

@dataclass
class Endpoint:
    params: float = 70e9
    gpus: int = 8
    gpu_price_per_h: float = 2.0
    kv_gb_per_gpu: float = 50.0        # 每卡可用于 KV 的显存
    def weight_bytes_per_gpu(self) -> float:
        return 2 * self.params / self.gpus
    def step_ms(self, batch: int, ctx: int) -> Tuple[float, str]:
        w = self.weight_bytes_per_gpu() / PEAK_BW
        f = 2 * self.params * batch / (PEAK_FLOPS * self.gpus)
        kv = batch * KV_PER_TOKEN * ctx / (PEAK_BW * self.gpus)
        t, bound = max(w, f, kv), "权重"
        if f == t: bound = "算力"
        if kv == t: bound = "KV"
        return t * 1000, bound
    def throughput(self, batch: int, ctx: int) -> Tuple[float, str]:
        t_ms, bound = self.step_ms(batch, ctx)
        return batch / (t_ms / 1000), bound
    def cost_per_million(self, batch: int, ctx: int, util: float = 1.0,
                         overhead: float = 1.35) -> Tuple[float, str]:
        """overhead 含非 GPU 成本与冗余（默认 1.35 倍）"""
        tps, bound = self.throughput(batch, ctx)
        gpu_cost_per_h = self.gpus * self.gpu_price_per_h * overhead
        return gpu_cost_per_h / (tps * 3600 / 1e6) / util, bound
    def max_batch_by_kv(self, ctx: int) -> int:
        return int(self.gpus * self.kv_gb_per_gpu * GB / (KV_PER_TOKEN * ctx))

ep = Endpoint()
print("① 每步时间与吞吐（8×H100、70B bf16、T=4K）")
print(f"  {'batch':>6} {'每步(ms)':>9} {'受限项':>6} {'吞吐(tok/s)':>12} "
      f"{'$/1M token':>11} {'TPOT(ms)':>9}")
for b in (1, 8, 32, 100, 256, 512):
    t_ms, bound = ep.step_ms(b, 4096)
    tps, _ = ep.throughput(b, 4096)
    cost, _ = ep.cost_per_million(b, 4096)
    print(f"  {b:>6} {t_ms:>9.2f} {bound:>6} {tps:>12,.0f} {cost:>11.3f} {t_ms:>9.2f}")
print("  读法：batch 从 1 到 100，成本降约 100 倍而 TPOT 不变（带宽受限区）；")
print("        过拐点后成本几乎不再降，TPOT 却明显变差 —— 这就是「成本必须绑定 SLO」的量化原因")

print("\\n② KV 决定的并发上限与长上下文的成本")
for ctx in (4096, 16384, 32768, 131072):
    mb = ep.max_batch_by_kv(ctx)
    cost, bound = ep.cost_per_million(min(mb, 512), ctx)
    tps, _ = ep.throughput(min(mb, 512), ctx)
    print(f"  T={ctx//1024:>3}K: 最大并发 {mb:>4}  取 batch={min(mb,512):>4} 时 "
          f"吞吐 {tps:>9,.0f} tok/s  ${cost:>6.3f}/1M（受限项 {bound}）")
print("  读法：上下文越长，KV 越早成为瓶颈 —— 长上下文请求的单位成本高得多，")
print("        定价必须按上下文长度分档（或对超长上下文单独报价）")

print("\\n③ 利用率与冗余对有效成本的影响（batch=100、T=4K）")
base, _ = ep.cost_per_million(100, 4096, util=1.0, overhead=1.0)
print(f"  {'利用率':>7} {'冗余倍数':>9} {'有效成本($/1M)':>15} {'相对裸成本':>10}")
for util in (1.0, 0.9, 0.7, 0.5, 0.3):
    for ov in (1.0, 1.35):
        c, _ = ep.cost_per_million(100, 4096, util=util, overhead=ov)
        print(f"  {util:>7.1f} {ov:>9.2f} {c:>15.3f} {c/base:>9.2f}x")
print("  读法：定价要写清「按什么利用率与冗余假设」—— 同一台机器在不同假设下能差 4 倍以上")

print("\\n④ 与按 token API 的盈亏平衡（API 输出价 $15/1M）")
def breakeven_tokens_per_month(gpus: int, price_per_h: float = 2.0,
                               api_price: float = 15.0, hours: int = 720,
                               overhead: float = 1.35, margin: float = 0.3) -> float:
    monthly = gpus * price_per_h * hours * overhead * (1 + margin)
    return monthly / api_price * 1e6
for gpus in (4, 8, 16):
    be = breakeven_tokens_per_month(gpus)
    print(f"  {gpus:>2} 卡专用 endpoint：月成本(含冗余与 30% 毛利) "
          f"${gpus*2*720*1.35*1.3:>9,.0f} -> 盈亏平衡月用量 ≈ {be/1e6:>7,.0f} 百万 token")
print("  读法：低于盈亏平衡用量就用按 token 的 API 更划算 —— 这个数字是与客户谈定价的起点")

print("\\n⑤ SLO 决定工作点：给定 TPOT 上限，最大可用 batch 与成本")
def max_batch_for_tpot(ep: Endpoint, ctx: int, tpot_ms: float) -> int:
    lo, hi = 1, ep.max_batch_by_kv(ctx)
    best = 1
    for b in range(1, hi + 1):
        t, _ = ep.step_ms(b, ctx)
        if t <= tpot_ms:
            best = b
        else:
            break
    return best
print(f"  {'TPOT 上限':>10} {'最大 batch':>10} {'吞吐(tok/s)':>12} {'$/1M token':>11}")
for tpot in (5.5, 8, 15, 40):
    b = max_batch_for_tpot(ep, 4096, tpot)
    tps, _ = ep.throughput(b, 4096)
    cost, _ = ep.cost_per_million(b, 4096)
    print(f"  {tpot:>10.1f} {b:>10} {tps:>12,.0f} {cost:>11.3f}")
print("  读法：TPOT 收紧到 5.5 ms（几乎不允许批）时成本是放宽到 15 ms 时的数倍 ——")
print("        所以「这个 endpoint 多少钱」这个问题必须先问 SLO，否则无法回答")
```

预期输出要点（实跑）：① batch 从 1 到 100 时**单 token 成本从 \$31.34 降到 \$0.313（约 100 倍）而 TPOT 恒定 5.22 ms**；越过 KV 拐点后**成本不再下降（吞吐恒为约 20k tok/s）而 TPOT 线性变差**——成本杠杆是「走出权重受限区」，不是「把 batch 顶满」；② 上下文越长、并发上限越低（4K→320、128K→10），**单位成本从 \$0.300 涨到 \$9.616（约 32 倍）**——长上下文必须单独定价；③ 利用率从 1.0 降到 0.3 会把有效成本推高 **3.3 倍**，加 35% 冗余再乘 1.35——**报价必须写清这些假设**；④ 与 \$15/百万的 API 价对照，4/8/16 卡专用 endpoint（含 35% 冗余与 30% 毛利）的**盈亏平衡月用量**分别为约 **674 / 1,348 / 2,696 百万 token**；⑤ **SLO 与成本的关系分区间**：在权重受限区（小 batch），放宽 batch 能把成本降一个数量级，此时 TPOT 与成本强相关；但一旦进入 KV 受限区，**收紧或放宽 TPOT 都不改变单位成本**（实测都是 \$0.300）——**此时成本由上下文长度决定（4K→128K 涨约 32 倍）**。这修正了「延迟越紧越贵」的想当然：在 KV 受限区可以直接答应更低的 TPOT 而不涨价，但客户要求更长上下文时**必须重新报价**。

## 常见追问

- **追问**：为什么按 GPU 小时定价对供应商更安全？
  - 要点：token 成本随流量形态（上下文长度、输出长度、batch 效率）波动极大；按 GPU 小时锁定了收入下限，风险从供应商转给「用不满的客户」；混合定价（预留 + 超额按 token）能兼顾双方。
- **追问**：客户流量有峰谷，怎么定价？
  - 要点：三种做法——① 按**承诺吞吐**（预留容量）计价，闲时也不退；② 按 token 但**高价覆盖峰值**；③ **共享池**（把多个客户的峰值错开复用，串 [[together-06]]）。选择取决于客户对隔离性与 p99 的要求。
- **追问**：长上下文（128K）怎么报价？
  - 要点：单独分档：KV 占用使并发下降 30 倍以上（131K 时并发个位数），且 prefill 成本随长度线性以上增长；工程上可对长上下文用 KV 量化/分层存储降低成本（串 [[anthropic-16]]）。
- **追问**：prefill 与 decode 的成本怎么分开算？
  - 要点：prefill 是**算力受限**（$2NP$），decode 是**带宽受限**（每步读权重+KV）；长输入短输出的请求成本主要在 prefill，长输出主要在 decode。定价可以按输入/输出分别计价（这也是行业惯例的原因）。
- **追问**：怎么验证成本模型准不准？
  - 要点：拿真实流量回放，记录**实际 token/s 与 GPU 利用率**，对比模型预测；差异通常来自：实测 MFU 低于假设、KV 命中率、调度开销、以及 CPU 侧瓶颈（tokenize/转发）。**每季度用实测校准一次**。
- **追问**：什么情况下应该建议客户不要买专用 endpoint？
  - 要点：用量低于盈亏平衡点、流量极不稳定（峰谷比 >5）、对 p99 要求不高（可用 API 的重试与排队）、或需要频繁切换模型。**给出「不卖」的建议能建立长期信任**。

## 相关题目

- [[together-03]]：投机解码——它在低负载时降低 token 成本，在高负载时反而抬高，是定价模型里的一个变量。
- [[together-06]]：serverless 共享池——用多租户错峰把利用率做上去，直接改善单位成本。
- [[anthropic-15]]：roofline 与算术强度，是本题每步时间与吞吐估算的基础。
- [[anthropic-16]]：batching 与 KV 约束，给出「平坦区由 KV 读取决定」的口径。
- [[anthropic-17]]：服务栈的利用率-p99 权衡与容量规划，与本题的成本模型互补。

## 参考资料与归属

- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。第 2 节「prefill 算力受限、decode 带宽受限，且 batch 与成本强相关」的分析框架来自这篇。
- **Efficient Memory Management for Large Language Model Serving with PagedAttention（延伸）** —— Kwon et al. (vLLM, SOSP 2023)，2023-09-12：<https://arxiv.org/abs/2309.06180>。第 1 节 KV 显存决定并发上限的机制来自这篇。
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（延伸）** —— Agrawal et al. (Microsoft, OSDI 2024)，2024-03-04：<https://arxiv.org/abs/2403.02310>。第「一句话答案」里「吞吐-延迟权衡需要按 SLO 取工作点」的取向来自这篇。
- **LLM 推理优化** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/llm-inference-optimization>。第 6 节成本拆解与非 GPU 项的工程背景参照这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（8×H100、\$2/GPU·h、989 TFLOPs、3.35 TB/s、KV 320 KiB/token、每卡 50 GB KV、冗余 1.35×、毛利 30%、API 价 \$15/百万、720 h/月）都是按本仓库统一口径构造的**工程算例与显式假设**；真实成本必须用自己的实测吞吐与利用率校准（本模型忽略 CPU 侧瓶颈、KV 命中率与调度开销）。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
