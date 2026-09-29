---
type: question
id: alibaba-01
company: 阿里巴巴（Qwen）
topic: coding
order: 1
question: Qwen2.5-Coder 使用仓库级 fill-in-the-middle 训练，用到 <|fim_prefix|>、<|fim_suffix|>、<|repo_name|> 这类 token。请写出格式化一个仓库级 FIM 样本的函数，并解释为什么仓库级优于文件级。
question_en: Qwen2.5-Coder trains with repository-level fill-in-the-middle using tokens like <|fim_prefix|>, <|fim_suffix|>, <|repo_name|>. Write the function that formats a repo-level FIM example, and explain why repo-level beats file-level.
asked_at: []
level: 进阶
tags: [FIM, 仓库级补全, 特殊 token, 数据打包, 实现题]
sources:
  - title: RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems（延伸）
    url: https://arxiv.org/abs/2306.03091
    author: Liu et al.
    published: 2023-06-05
  - title: CodeRAG-Bench: Can Retrieval Augment Code Generation?（延伸）
    url: https://arxiv.org/abs/2406.14497
    author: Wang et al.
    published: 2024-06-20
  - title: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）
    url: https://arxiv.org/abs/2310.06770
    author: Jimenez et al. (ICLR 2024)
    published: 2023-10-10
  - title: Lost in the Middle: How Language Models Use Long Contexts（延伸）
    url: https://arxiv.org/abs/2307.03172
    author: Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang
    published: 2023-07-06
related: [coding-04, coding-10, system-design-02, llm-internals-13]
updated: 2026-09-28
---

## 一句话答案

> 仓库级 FIM 把一个仓库的多个文件按 `<|repo_name|>{repo}<|file_sep|>{path}\n{code}` 拼成一条序列，只挑其中一个 chunk 开洞：`{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}`，labels 只在 middle 段（含紧随其后的停止符）上有值，其余位置一律 -100。
> 它比文件级强的地方不是「上下文更多」，而是三条机制链：跨文件依赖是硬约束（被补全处的基类、类型、常量常在本文件之外）、`<|repo_name|>` 与 path 这类角色信号只有在仓库级样本里才有意义、训练目标的可达性——文件级把窗口压在 8,192 token，模型没机会学「在 32K 里找定义」。
> 代价同样真实：仓库级阶段只占 300B token（5.45%），FIM 监督密度却掉到 1.5% 量级，多出来的长上下文账单落在推理端的 KV cache 与 prefill 上。

## 面试官在考什么

- 能不能先钉契约再写代码：`files` 的顺序在函数外定死（按路径或按相关度），`max_len / fim_rate / spm_ratio / middle_mode / rng` 全部显式传参，返回值带 `middle_span` 与 `fim` 两个字段。少了 `middle_span`，离线统计「FIM 实际占比」和样本回归都做不了。
- 六条硬规则是否成体系：仓库元数据只做条件、逐 chunk 抽签且序列里只有一个 `<|fim_middle|>`、PSM 与 SPM 两种排布都要能生成、目标文件 path 行必须留在 prefix 段、切点落在 token 边界、labels 只覆盖 middle。
- 特殊 token 的原子性有没有真验证过：`<|fim_prefix|>` 在 Qwen 词表里是 1 个 id（151659），在通用编码器 o200k_base 里是 6 个片段。差这一步，控制 token 就从指令退化成「让模型猜字符串」。
- 「为什么仓库级优于文件级」能不能拆成机制链而不是形容词，并且主动补上代价（监督密度、KV cache、检索侧的独立性）。
- 工程纪律：仓库内按内容哈希去重、评测集整仓留出、训练与推理逐字同构、随机种子绑到 (repo_sha, path, epoch)。

常见错误答案：

- 只答「上下文更长，所以更好」。长度是必要条件不是充分条件：相关信息落在长上下文中段时准确率显著下降（lost-in-the-middle），检索到的上下文也可能因为词汇重叠低而召不回（CodeRAG-Bench）。
- 把 `<|repo_name|>{repo}` 和 path 行也拿去 FIM。loss 会花在猜仓库名和路径上，而这些信息在推理时本来就从环境里拿得到，属于白送的标签污染。

## 原理与推导

### 1. 先把契约钉死，再写代码

`format_repo_fim(repo_name, files, tokenizer, *, max_len=32768, fim_rate=0.5, spm_ratio=0.5, middle_mode="random", target_index=None, rng=None)` 返回 `{"input_ids", "labels", "fim", "middle_span", "meta"}`，外加给断言用的 `texts`（三段原文，验证「拼得回原文件」时必需）。`files` 是 `(path, content)` 列表，顺序在函数外定死，函数内部不许再洗牌：洗掉顺序就洗掉了复现性。`middle_span` 是 `(start, end)` 半开区间，指「被算 loss 的那一段」（middle 加停止符）；`fim` 区分 FIM 样本与纯 NTP 样本。`meta` 至少记 `draws`（逐 chunk 的抽签值）、`dropped_fim`、`truncated`、`layout`——这四个字段是事后审计「实际 FIM 率」「截断策略」「PSM/SPM 配比」的唯一依据。

六条硬规则，任何一条破了都不会报错，只会让指标悄悄掉：

1. `<|repo_name|>{repo}` 之后的仓库元数据永不参与 FIM，只当条件。
2. 逐 chunk（一文件一 chunk）独立抽签 `rng.random() < fim_rate`，每个 chunk 至多一次 FIM，一条序列里绝不出现两个 `<|fim_middle|>`。
3. PSM 与 SPM 按 `spm_ratio` 抽签，两种排布都要能生成。
4. 目标文件的 path 行必须留在 prefix 段内，它是「这是哪个文件」的条件信号，不是要生成的内容。
5. 切点必须落在 token 边界上。字符级均匀随机是论文口径：两个独立均匀切点给出的三段期望长度各占 $1/3$。
6. labels 只对 middle 段（含其后的停止符）算 loss，其余置 -100，且这条口径在训练与推理两侧必须一致。

### 2. 组装：PSM 与 SPM 两种排布

```text
PSM  <|repo_name|>{repo}<|file_sep|>{path0}\n{code0}<|file_sep|>{path_t}\n{prefix}
     <|fim_suffix|>{suffix}<|fim_middle|>{middle}<|file_sep|>{path_t+1}\n{code_t+1}<|file_sep|><|endoftext|>

SPM  <|repo_name|>{repo}<|file_sep|>{path0}\n{code0}<|fim_suffix|>{suffix}
     <|fim_prefix|><|file_sep|>{path_t}\n{prefix}<|fim_middle|>{middle}<|file_sep|>{path_t+1}\n{code_t+1}<|file_sep|><|endoftext|>
```

- PSM 的 prefix 就是天然左上下文，所以不额外发 `<|fim_prefix|>`；SPM 里 suffix 先出场，必须用 `<|fim_prefix|>` 把 prefix 段重新标出来。两种排布里目标文件的 path 行都落在 prefix 段：PSM 在 `<|fim_suffix|>` 之前，SPM 在 `<|fim_prefix|>` 之后。
- 每个文件块自带一个前导 `<|file_sep|>`，因此 middle 的停止符就是下一个文件块的前导 `<|file_sep|>`（没有后续文件时是收尾的那个）。
- `<|endoftext|>` 收尾；`<|fim_pad|>` 只在把 FIM 段补齐到定长时使用，不跟序列结束混用。

### 3. 三条机制链：仓库级为什么优于文件级

**① 跨文件依赖是硬约束，不是加分项。** 被补全处的签名、类型、常量、别名常在本文件之外：`from .x import y`、proto/schema、依赖注入注册。文件级 FIM 的 suffix 只看本文件尾部，条件信息不足。RepoBench 专门把任务切成 R（检索跨文件片段）、C（带跨文件上下文补全）、P（检索加补全流水线）三档，说明单文件基准根本评估不到这项能力；SWE-bench 的 2,294 个真实 issue 跨 12 个 Python 仓库，原文明确说解决这类问题经常需要同时理解并协同修改多个函数、类甚至文件，而当时最好的模型（Claude 2）只解掉 1.96%，差距就落在仓库级信息上。

**② 仓库级上下文给的是约定与角色，不是事实。** 命名风格、目录语义（同一段代码在 `tests/` 与 `src/` 下的下一步不同）、框架与版本、路径本身（`migrations/` 里该写 DDL）。`<|repo_name|>` 与 path 这两个条件信号只有在仓库级样本里才有意义；到了文件级样本里它们退化成同分布噪声，每个样本都带同样的前缀，在梯度里近似常数。

**③ 训练目标的可达性。** 文件级阶段把序列上限压在 8,192 token，模型没机会学「在 32K 里找定义」。Qwen2.5-Coder 的仓库级阶段把上下文从 8,192 拉到 32,768（config 实测 `max_position_embeddings=32768`）、RoPE base 从 $10^4$ 提到 $10^6$（`rope_theta=1000000.0`），再用 YaRN 外推到 131,072。没有这一步，推理端把整仓塞进 prompt 只会撞上 lost-in-the-middle：相关信息落在长上下文中段时准确率显著下降、头尾最好，所以文件顺序、目标文件位置与关键定义的位置本身就是设计变量。

**边界也要一起说清楚。** 上下文越长噪声占比越高、attention 的平方项越贵（32K 相对 8K：线性项涨 4 倍，平方项涨 16 倍，所以 prefill 的 $2NT$ 只是下界）；CodeRAG-Bench 的结论是检索到的上下文确实能提升生成，但现有 retriever 在词汇重叠低时召不回、生成器在上下文长度受限时也用不上——「训练侧见过长仓库上下文」与「推理侧做检索与重排」是两件事，不能互相替代。

### 4. 生产里按踩到概率排序的坑

1. 特殊 token 没原子化；拼各段时误开 `add_special_tokens=True`，BOS/EOS 会被插进序列中段。
2. 顺序错了。必须「选窗口 → 裁剪 → 在窗口内 FIM」：FIM 论文明确警告，长文档先做 FIM 再切分会让 prefix 或 suffix 整段被切出上下文，只剩孤立的 middle；仓库级尤其危险，断言里要强制 prefix 与 suffix 都还在窗口内。
3. 切点落在字符串、注释或 token 中间；或者把元数据当成目标，让模型猜仓库名和路径。前者的解法是叠一层语法感知，后者只要记住 path 行永远属于 prefix。
4. 仓库内重复内容让指标虚高。vendored 代码、生成文件、测试快照里常有同一函数的拷贝，模型能从另一份文件直接抄出 middle，必须按内容哈希在仓库内去重，并过滤 lock/二进制/超大文件。
5. 评测泄漏。仓库级打包把同仓其它文件一起喂进来，评测集必须整仓留出而不是只留出被补全的文件，`<|repo_name|>` 本身就是泄漏通道；Qwen 报告里 SAFIM 只取 2022-04 之后创建的文件来避开预训练语料，是同一条纪律。
6. 训练与推理不同构，以及随机性不可复现。推理端 prompt 必须逐字复刻训练排布，并给 middle 一个可停止的出口（`<|endoftext|>`、`<|file_sep|>`）；少一个 path 行、PSM/SPM 弄反都不报错，只是指标掉——RepoEval 在 Qwen2.5-Coder 报告里的评测口径可当参考答案：最大序列 8,192、跨文件上下文上限 2,048 token、函数级输出上限 256 token（RepoCoder 原文给 CodeGen 的默认口径更紧：输入加输出合计 2,048、函数级输出上限 500）。种子要绑到 (repo_sha, path, epoch)，`max_len` 必须用真实 tokenizer 计数而不是字符数，截断只许砍窗口远端、不许砍 middle。

### 5. 怎么验证这 300B 没白花

- 在同一个文件级底座上加训 300B 仓库级数据做 A/B，评测必须用仓库级基准（RepoEval 的行/API/函数体三档、CrossCodeEval、RepoBench 的 R/C/P），而不是 HumanEval——HumanEval 单文件、无仓库上下文，对仓库级训练几乎没有区分度；同时要报 FIM 基准（HumanEval-FIM、SAFIM），因为「长上下文变强、infilling 变弱」是这类阶段最容易出现的回归。
- 超参要监控而不是只写在配置里：FIM 论文的结论是 50% 的 FIM 率不损伤左到右能力、甚至到 90% 也不退化，但 FIM 能力本身强依赖超参；StarCoder2-15B 就因为实现 bug 让实际 FIM 率长期低于设定值、FIM 基准明显掉队。每个 batch 统计 `draws` 的实际命中率，就是这条监控。

## 数值与代码验证

先给复算过的三张账。口径：模型参数 $N=3.2\times10^{10}$，H100 bf16 dense 989 TFLOPs，MFU 40%，GPU 时价按 \$2 计。

| 量 | 文件级阶段 | 仓库级阶段 | 算式与口径 |
| --- | --- | --- | --- |
| 训练 token | 5.2×10¹² | 3.0×10¹¹ | Qwen2.5-Coder 报告 |
| 序列上限 | 8,192 | 32,768 | config `max_position_embeddings` |
| 序列条数 | 6.35×10⁸ | 9.16×10⁶ | token ÷ 窗口，相差 69.3 倍 |
| 训练算力 $6ND$ | 9.98×10²³ FLOPs | 5.76×10²² FLOPs | $N=3.2\times10^{10}$，$D$ 为 token 数 |
| GPU·h | 7.01×10⁵ | 4.04×10⁴ | $6ND/(989\times10^{12}\times0.4\times3600)$ |
| 成本 | ≈\$1.40M | ≈\$8.09 万 | 按 \$2/GPU·h |
| 占比 | 100% | 5.77% | 与 token 占比 5.45% 同阶 |

长上下文是买来的能力，账单在服务端（LLaMA-3-70B 每 token KV cache 320 KiB）：

| 量 | 算式 | 结果 |
| --- | --- | --- |
| 每 token KV | $2\times80\times8\times128\times2$ B | 320 KiB |
| 32K 请求 KV | $320\ \text{KiB}\times32768$ | 10 GiB |
| 128K 请求 KV | ×4 | 40 GiB |
| 读一遍 @3.35 TB/s | $10.737\times10^9/3.35\times10^{12}$ | 3.2 ms |
| decode 带宽上限 | $1/3.2\ \text{ms}$ | 312 token/s |
| roofline 拐点 | $989/3.35$ | 295 FLOPs/byte |
| 32B prefill @32K | $2NT/(989\times10^{12}\times0.4)$ | 5.30 s |
| 32B prefill @8K | 同上 | 1.33 s |

抽签占比要写成算式，不要只写结论。FIM 论文的文档级 FIM 率是 $p=0.5$，PSM 与 SPM 各分走 25%；StarCoder2 的 repo-context 变体是两级抽签，仓库以 50% 进入 FIM 候选、候选仓库里每个 chunk 再以 50% 变换，$0.5\times0.5$ 得到 25% 的 chunk 被 FIM，再按 PSM/SPM 分半即各约 12.5%（论文口径的推导，不是模型卡实测值）。

监督密度是这题最容易被忽略的取舍。一个 1,495 token 的 Python 文件按 $1/3$ 切出的 middle 约 $1495/3=498$ token，在 32,768 的窗口里监督密度只有 $498/32768=1.52\%$，只有纯 NTP 的 $1/66$（$32768/498\approx66$；8,192 窗口下是 $6.08\%$）。这就解释了仓库级阶段为何只花 5.45% 的 token：它买的是长上下文与跨文件能力，不是 token 吞吐。

另外，7 个特殊 token 的参数是 $7\times2\times5120=71{,}680$，占 32B 的 $2.24\times10^{-6}$——成本不在参数，而在「必须是单 token」。

下面这段函数就是上面契约的实现。用真实 Qwen2.5-Coder-7B 的 `tokenizer.json` 跑，正文后面是实际的输出与四条 assert。

```python
import ast, random
FIM = {"repo": "<|repo_name|>", "sep": "<|file_sep|>", "prefix": "<|fim_prefix|>",
       "suffix": "<|fim_suffix|>", "middle": "<|fim_middle|>", "pad": "<|fim_pad|>",
       "eot": "<|endoftext|>"}


def encode(tok, text):
    e = tok.encode(text, add_special_tokens=False)       # BOS/EOS 只许出现在序列两端
    return e.ids, [tuple(o) for o in e.offsets]          # offsets 用来把切点吸附到 token 边界


def snap(offs, pos):                                    # 把字符位置吸附回 token 边界
    return sum(1 for _, end in offs if end <= pos)       # BPE 的 offsets 单调，计数即切点


def ast_span(code, rng):                                # 语法感知抽块；其它语言换 tree-sitter
    kinds = (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While)
    ln = sorted({n.lineno for n in ast.walk(ast.parse(code)) if isinstance(n, kinds)})
    if len(ln) < 2:
        return None
    lo, hi = sorted(rng.sample(ln, 2))
    src = code.splitlines(keepends=True)
    return "".join(src[:lo - 1]), "".join(src[lo - 1:hi - 1]), "".join(src[hi - 1:])


def format_repo_fim(repo_name, files, tokenizer, *, max_len=32768, fim_rate=0.5,
                    spm_ratio=0.5, middle_mode="random", target_index=None, rng=None):
    """files = [(path, content)]，顺序在函数外定死（按路径或按相关度），函数内不再洗牌。"""
    rng = rng or random.Random()
    enc = lambda s: encode(tokenizer, s)[0]
    meta = {"repo": repo_name, "draws": {}, "dropped_fim": [], "truncated": 0, "layout": None}
    # 规则② 逐 chunk（一文件一 chunk）独立抽签；首个命中者当 middle，其余交给 meta 另行出样本
    hits = []
    for i, (path, _) in enumerate(files):
        if target_index is not None:
            hits += [i] if i == target_index else []
            continue
        draw = rng.random()
        meta["draws"][path] = round(draw, 4)
        if draw < fim_rate:
            hits.append(i)
    if not hits:                                         # 没抽中 → 纯 NTP 样本
        ids = enc(FIM["repo"] + repo_name + "".join(FIM["sep"] + p + "\n" + c for p, c in files)
                  + FIM["sep"] + FIM["eot"])
        return {"input_ids": ids, "labels": ids[:], "fim": False, "middle_span": None, "meta": meta}
    t, meta["dropped_fim"] = hits[0], hits[1:]
    # 规则⑤ 切点：字符级均匀两个切点（期望三段各 1/3），再吸附到 token 边界
    path, code = files[t]
    ids, offs = encode(tokenizer, code)
    n = len(ids)
    piece = ast_span(code, rng) if middle_mode == "ast" and path.endswith(".py") else None
    if piece is None:
        for _ in range(20):
            c1, c2 = sorted((int(len(code) * rng.random()), int(len(code) * rng.random())))
            k1, k2 = snap(offs, c1), snap(offs, c2)
            if 1 <= k1 < k2 <= n - 1:
                e1, e2 = offs[k1 - 1][1], offs[k2 - 1][1]
                piece = (code[:e1], code[e1:e2], code[e2:])
                break
        else:                                            # 兜底：按 token 三等分，仍不越 token 边界
            piece = (code[:n // 3], code[n // 3:2 * n // 3], code[2 * n // 3:])
    pre_txt, mid_txt, suf_txt = piece
    pre, mid, suf = enc(pre_txt), enc(mid_txt), enc(suf_txt)
    # 规则④ 目标文件 path 行留在 prefix 段；规则③ PSM / SPM 抽签
    ctx = lambda fs: sum((enc(FIM["sep"] + p + "\n" + c) for p, c in fs), [])
    head, sep, eot = enc(FIM["repo"] + repo_name), enc(FIM["sep"]), enc(FIM["eot"])
    tline, before, after = enc(FIM["sep"] + path + "\n"), ctx(files[:t]), ctx(files[t + 1:])
    spm = rng.random() < spm_ratio

    def build(before):
        if spm:     # SPM：suffix 先出场，必须用 <|fim_prefix|> 把 prefix 段重新标出来
            seq = head + before + enc(FIM["suffix"]) + suf + enc(FIM["prefix"]) + tline + pre
            start = len(head) + len(before) + 1 + len(suf) + 1 + len(tline) + len(pre)
        else:       # PSM：prefix 就是天然左上下文，不额外发 <|fim_prefix|>
            seq = head + before + tline + pre + enc(FIM["suffix"]) + suf
            start = len(head) + len(before) + len(tline) + len(pre) + 1 + len(suf)
        meta["layout"] = "SPM" if spm else "PSM"
        return seq + enc(FIM["middle"]) + mid + after + sep + eot, start + 1   # +1 跳过 middle

    seq, start = build(before)
    while len(seq) > max_len and before:                 # 超窗只砍远端上下文，不碰 middle
        before, meta["truncated"] = before[len(before) // 2:], meta["truncated"] + 1
        seq, start = build(before)
    if len(seq) > max_len:
        raise ValueError("prefix + suffix + middle 本身超窗：截断不许砍 middle")
    # 规则⑥ labels 只覆盖 middle 段 + 停在其后的 <|file_sep|>，其余 -100
    span = (start, start + len(mid) + 1)
    labels = [-100] * len(seq)
    labels[span[0]:span[1]] = seq[span[0]:span[1]]
    return {"input_ids": seq, "labels": labels, "fim": True, "middle_span": span,
            "meta": meta, "texts": (pre_txt, mid_txt, suf_txt)}
```

3 文件 mini repo，middle 落在 `src/shapes/circle.py`（它 `import` 的 `Shape` 基类定义在另一个文件里）：

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")              # 真实 Qwen2.5-Coder-7B 词表
REPO, FILES = "acme/shapes", [
    ("src/shapes/base.py", "class Shape:\n    def area(self):\n        raise NotImplementedError\n"),
    ("src/shapes/circle.py", "import math\nfrom src.shapes.base import Shape\n\n\nclass Circle(Shape):\n"
     "    def __init__(self, r):\n        self.r = r\n\n    def area(self):\n"
     "        return math.pi * self.r ** 2\n"),
    ("tests/test_circle.py", "from src.shapes.circle import Circle\n\n\ndef test_area():\n"
     "    assert Circle(2).area() > 12\n")]
res = format_repo_fim(REPO, FILES, tok, rng=random.Random(1359), fim_rate=0.5, spm_ratio=0.0)
ids, labels, span = res["input_ids"], res["labels"], res["middle_span"]
spec = {tok.encode(v, add_special_tokens=False).ids[0]: v for v in FIM.values()}
buf = []                                                 # 特殊 token 独占一行，看清排布
for i, t in enumerate(ids):
    if t in spec:
        if buf:
            print(repr("".join(buf)))
            buf = []
        print(f"{spec[t]} @ {i}")
    else:
        buf.append(tok.decode([t]))
if buf:
    print(repr("".join(buf)))
print(f"总长 {len(ids)} | middle_span {span} | 非 -100 label {sum(x != -100 for x in labels)}")
print("被算 loss 的序列 =", repr(tok.decode([x for x in labels if x != -100])))

pre, mid, suf = res["texts"]                             # 四条不变量
assert pre + mid + suf == FILES[1][1], "FIM 是重排不是改写：三段必须逐字节拼回原文件"
assert all(len(tok.encode(v, add_special_tokens=False).ids) == 1 for v in FIM.values()), "特殊 token 必须原子"
mid_id, repo_id, sep_id = (tok.encode(FIM[k], add_special_tokens=False).ids[0] for k in ("middle", "repo", "sep"))
assert ids.count(mid_id) == 1 and ids.count(repo_id) == 1 and ids.index(repo_id) < ids.index(sep_id), "唯一性"
assert sum(x != -100 for x in labels) == span[1] - span[0], "labels 非 -100 数 = middle 段长度"
print("四条不变量全部通过；三段 token 数 =", [len(tok.encode(x).ids) for x in res["texts"]])
```

实际输出（三个 chunk 的抽签值是 `base.py` 0.9466、`circle.py` 0.2464、`test_circle.py` 0.9389，只有中间那个小于 0.5）：

```text
<|repo_name|> @ 0
'acme/shapes'
<|file_sep|> @ 5
'src/shapes/base.py\nclass Shape:\n    def area(self):\n        raise NotImplementedError\n'
<|file_sep|> @ 24
'src/shapes/circle.py\nimport math\nfrom src.shapes.base import Shape\n\n\nclass Circle(Shape):\n   '
<|fim_suffix|> @ 48
' def area(self):\n        return math.pi * self.r ** 2\n'
<|fim_middle|> @ 64
' def __init__(self, r):\n        self.r = r\n\n   '
<|file_sep|> @ 80
'tests/test_circle.py\nfrom src.shapes.circle import Circle\n\n\ndef test_area():\n    assert Circle(2).area() > 12\n'
<|file_sep|> @ 110
<|endoftext|> @ 111
总长 112 | middle_span (65, 81) | 非 -100 label 16
被算 loss 的序列 = ' def __init__(self, r):\n        self.r = r\n\n   <|file_sep|>'
四条不变量全部通过；三段 token 数 = [16, 15, 15]
```

这份输出能确认四件事。① 整条序列只有一个 `<|fim_middle|>`、一个 `<|repo_name|>` 且它在首个 `<|file_sep|>` 之前，另外两个文件是纯上下文。② prefix 段里既有 `src/shapes/circle.py` 这行 path，也有前面整个 `base.py`——middle 里用到的 `Shape` 正是从那里来的，这就是跨文件依赖在样本里长什么样。③ 切点是字符级随机的（论文口径），所以 middle 从缩进中间开始、到下一行缩进结束，三段 16/15/15 token 近似三等分；要贴合「补一个函数体」的真实编辑分布就换 `middle_mode="ast"`，同一份代码抽出的 middle 是 `class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n\n`（10/20/16 token），依旧逐字节拼得回原文。④ `middle_span` 是 (65, 81)，16 个 token 里 15 个是 middle、1 个是停止符 `<|file_sep|>`，labels 的非 -100 计数与它相等。再把 `spm_ratio` 设成 1.0 跑一遍，目标文件块内的特殊 token 顺序变成 `<|repo_name|> → <|fim_suffix|> → <|fim_prefix|> → <|file_sep|>(path 行) → <|fim_middle|> → …`（前面的 `base.py` 上下文块仍自带一个前导 `<|file_sep|>`），总长 113 且四条断言同样成立；把种子换成 5 则三个 chunk 都没抽中，`fim=False` 且 `labels == input_ids`（纯 NTP 样本，不要在这条路上继续减 labels）。

关于特殊 token 的第二个断言值得单独验一遍：`<|fim_prefix|>` 在 Qwen2.5-Coder-7B 词表里编码成 1 个 id（151659，同批的 7 个 token 分别是 `<|endoftext|>` 151643、`<|fim_prefix|>` 151659、`<|fim_middle|>` 151660、`<|fim_suffix|>` 151661、`<|fim_pad|>` 151662、`<|repo_name|>` 151663、`<|file_sep|>` 151664，词表共 151,665 项）；换成通用编码器 tiktoken 的 o200k_base，同一串字符被切成 `<`、`|`、`fim`、`_prefix`、`|`、`>` 共 6 个 token。控制 token 失效就是这么发生的：模型看到的不是标记，而是一段普通文本。

## 常见追问

- **追问**：PSM 和 SPM 该选哪个？
  - 要点：PSM 与自然推理顺序一致；SPM 把 suffix 提到 prefix 之前，动机是 KV cache——追加 prefix 不会作废 suffix 已算的 KV（编辑场景里前缀在变长、后缀相对稳定），且在 infilling 基准上略有优势。联合训练让两种格式都能用，代价是格式空间翻倍、prompt 组装与回归测试都要覆盖两种。
- **追问**：middle 用随机 span 还是 AST 抽块？
  - 要点：随机 span 覆盖面广，但和真实编辑分布不同；AST/语句块贴合「补一个函数体」，却容易让模型只会补语法块、失去跨块补全能力。论文口径是字符级均匀随机，Qwen 的 FIM 指令数据用 tree-sitter 抽 AST 多层级节点当 middle；选哪条取决于要优化的编辑形态，且必须写进数据版本里。
- **追问**：文件顺序按路径排还是按相关度排？
  - 要点：按路径排序可复现、假设稳定，但与检索场景的分布不一致；按相关度排序更接近线上，却难保训练与推理一致。用 lost-in-the-middle 的结论定一条硬规则：关键定义（被引用符号的声明）不要落在上下文中段。
- **追问**：FIM 会不会损伤左到右的生成能力？
  - 要点：FIM 论文的结论是 50% 的 FIM 率不损伤左到右能力、甚至到 90% 也不退化；但 FIM 能力本身强依赖超参，StarCoder2-15B 就因为实现 bug 让实际 FIM 率长期低于设定值、FIM 基准明显掉队。所以要按 batch 记录实际抽签占比，而不是只相信配置里的数字。
- **追问**：为什么不用 `<|endoftext|>` 兼作 padding？
  - 要点：FIM 段的 padding 与「序列结束」是两种语义，混用会让模型在补全时提前吐结束符，所以词表里单独留了 `<|fim_pad|>`。同理，推理端给 middle 的停止出口也要显式约定。
- **追问**：线上补全只有几百毫秒预算，训到 128K 不是白训吗？
  - 要点：训练买的是能力，线上靠检索加重排把上下文压到可用预算（RepoEval 用 2,048 token 跨文件上下文就能测出差异），再叠 prefix caching 复用不变前缀。能力、prompt 组装与缓存三件事各自解决不同问题，不能互相替代。

## 相关题目

- [[coding-04]]：BPE 的训练与编码，特殊 token 是怎么进词表并保持原子的。
- [[coding-10]]：不切开语义单元的文本分块器，与「切点必须落在 token 边界」是同一类约束。
- [[system-design-02]]：代码助手的仓库索引与上下文组装，仓库级 FIM 的推理侧另一半。
- [[llm-internals-13]]：lost-in-the-middle，解释了为什么文件顺序与目标文件位置是设计变量。

## 参考资料与归属

- RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems（延伸）— Liu et al.，2023-06-05。[https://arxiv.org/abs/2306.03091](https://arxiv.org/abs/2306.03091)
- CodeRAG-Bench: Can Retrieval Augment Code Generation?（延伸）— Wang et al.，2024-06-20。[https://arxiv.org/abs/2406.14497](https://arxiv.org/abs/2406.14497)
- SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）— Jimenez et al. (ICLR 2024)，2023-10-10。[https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)
- Lost in the Middle: How Language Models Use Long Contexts（延伸）— Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang，2023-07-06。[https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)

四篇延伸来源支撑第 3 节的机制链与基准结论（RepoBench 的 R/C/P 三档、SWE-bench 的 2,294 个 issue 与 1.96% 解决率、lost-in-the-middle 的中段衰减、CodeRAG-Bench 的检索侧结论）；函数契约、PSM/SPM 排布、特殊 token 原子性与监督密度数字来自对 Qwen2.5-Coder-7B/32B 词表与 config 的本地实测与复算，属于本文的实现口径而非论文原文。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
