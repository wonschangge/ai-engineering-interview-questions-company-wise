---
type: question
id: cursor-11
company: Cursor（Anysphere）
topic: system-design
order: 11
question: 你会如何为包含 10 万个文件的 monorepo 建立索引，让 AI 编辑器能够检索到相关上下文，并在用户编辑时保持索引更新？
question_en: How would you index a 100k-file monorepo so an AI editor can retrieve relevant context, and keep the index fresh as the user edits?
asked_at: []
level: 高阶
tags: [系统设计, 代码索引, 增量更新, Merkle 树, 混合检索]
sources:
  - title: Cursor 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-cursor-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Claude Code 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-claude-code-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems（延伸）
    url: https://arxiv.org/abs/2306.03091
    author: Liu et al.
    published: 2023-06-05
  - title: CodeRAG-Bench: Can Retrieval Augment Code Generation?（延伸）
    url: https://arxiv.org/abs/2406.14497
    author: Wang et al.
    published: 2024-06-20
  - title: Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs（HNSW）（延伸）
    url: https://arxiv.org/abs/1603.09320
    author: Malkov & Yashunin
    published: 2016-03-30
  - title: Merkle tree（维基百科）（延伸）
    url: https://en.wikipedia.org/wiki/Merkle_tree
    author: Wikipedia
    published: 
related: [cursor-07, cursor-10, rag-01, rag-08, system-design-02]
updated: 2026-09-28
---

## 一句话答案

> 索引要分**四棵并存的索引**，因为它们服务不同的查询：
> ① **文件树/清单索引**（路径、大小、语言、mtime、git 状态）——解决「哪些文件与我有关」；
> ② **符号索引**（定义、引用、调用图、类型关系）——解决「这个函数在哪、谁在用它」（代码场景最准、最便宜、可增量）；
> ③ **词法索引**（倒排：标识符、字符串、注释）——解决「我大概记得这个名字」；
> ④ **向量索引**（片段/函数的语义嵌入）——解决「命名不同但语义相近」（可引用 HNSW 的近似最近邻结构：以分层图实现对数级检索）。
> 检索时**融合**四路候选再重排（符号优先级最高），可引用的评测口径是 CodeRAG-Bench：它系统研究了「检索增强能否提升代码生成」，并指出**检索到的东西必须真的有用**——把无关代码塞进上下文会伤害生成质量。
> 「保持更新」的关键是**变更检测 + 分层失效**：用内容哈希/Merkle 树自底向上判断哪些子树变化（可引用 Merkle 树的性质：根摘要相同即可判定整棵子树相同），只重建受影响文件的符号/词法/向量条目；用户未保存的编辑走**内存增量索引**（不能等保存）。一句话：**索引不是快照，是一条持续对账的流水线。**

## 面试官在考什么

- **是否区分索引类型**：只答「向量数据库」是初级答案；代码场景里**符号索引**（LSP/语法树/调用图）通常比向量检索更准、更便宜，也更容易增量更新。
- **增量更新的机制**：能否给出「变更检测 → 受影响范围 → 局部重建」的具体做法（文件哈希、目录级 Merkle 摘要、依赖图传播），以及**未保存编辑**（编辑器内存态）怎么进索引。
- **冷启动与预算**：10 万文件的全量索引要多久、多大、花多少钱（下一节给账）；是否设计**分层启动**（先目录 + 符号签名可用，再逐步补向量）。
- **检索融合与重排**：能不能说清多路召回的分数怎么合并（按来源加权/学习式重排），以及为什么**重排比扩大召回更重要**（无关片段会稀释注意力，串 [[cursor-07]]）。
- **规模与隔离**：多用户/多分支/多工作区共享什么、隔离什么；分支切换时如何快速复用（内容寻址天然支持跨分支共享未变子树）。
- **工程细节**：忽略规则（`.gitignore` + 二进制 + 大文件 + 生成物）、增量批处理与背压、以及索引版本与 schema 迁移。

**常见错误答案**

- 「全量建一次向量索引就行」——用户一编辑就过期；且向量检索在代码上经常输给符号检索。
- 「每次编辑重建整棵索引」——10 万文件下不可接受（下一节的账）。
- 「只索引文件内容」——忽略结构信息（调用图、目录层级、git 历史），而结构正是代码检索最重要的信号。
- 「不做忽略规则」——`node_modules`、构建产物、二进制会瞬间把索引撑爆。

## 原理与推导

### 1. 四棵索引的分工与成本

| 索引 | 粒度 | 查询类型 | 全量成本（10 万文件） | 增量成本 |
| --- | --- | --- | --- | --- |
| 文件清单 | 文件 | 过滤（语言/目录/大小） | 秒级（元数据） | 极低（单个目录项） |
| 符号索引 | 符号 | 定义/引用/调用图 | 分钟级（解析 AST） | 单文件毫秒级 + 反向依赖传播 |
| 词法倒排 | token | 关键词/正则 | 分钟级 | 单文件毫秒级 |
| 向量索引 | 片段（函数/类） | 语义相似 | **小时级**（嵌入计算） | 单片段毫秒级（嵌入）+ 索引插入 |

**重要结论**：向量索引是全量成本的大头（因为要为每个片段调用嵌入模型）；所以工程上必须**分层可用**——先让符号与词法索引可用（覆盖大多数查询），向量索引在后台补齐。

### 2. 全量索引的账（10 万文件）

设每文件 200 行、每行 8 token、每 40 行切一个片段（函数级）：

- 总行数 $2\times10^7$ 行；总 token $\approx1.6\times10^8$；
- 片段数 $\approx5\times10^5$；
- 嵌入成本：按 \$0.02/百万 token（小型嵌入模型的自建成本量级）→ $1.6\times10^8/10^6\times0.02=\$3.2$（**很便宜**）；若用大模型嵌入（\$0.13/M）则约 \$21；
- 向量存储：$5\times10^5\times768$ 维 $\times$4 字节 $\approx1.5$ GiB（float32）；用 int8 量化降到约 0.4 GiB；
- 时间：单卡嵌入吞吐按每秒 5 万个片段估计 → 约 10 秒**计算**，但受解析（AST）与 I/O 限制，实际分钟级到小时级。

**结论**：向量索引的成本瓶颈不在嵌入，而在**解析与 I/O**（要读 10 万文件、解析语法树）以及**冷启动延迟**——所以设计重点是并行解析 + 分层可用 + 缓存复用。

### 3. 增量更新：变更检测与失效传播

用「内容哈希 + 目录树摘要（Merkle 式）」做变更检测：

```
每个文件: h_file = hash(内容)
每个目录: h_dir  = hash(排序后的 (名字, h_file/h_subdir))
仓库根:   h_root = hash(顶层目录摘要)
```

比较两次 $h_{\text{root}}$：相同即整棵树未变（O(1) 判定）；不同则自顶向下只进入摘要变化的子树，定位到变化文件（可引用的性质来自 Merkle 树的结构：子树摘要相同即可判定其内容一致）。

对变化的文件：

1. **重解析**该文件的符号 → 更新符号索引；
2. **失效反向依赖**：谁 import 了它、谁调用了它改动的导出符号 → 这些文件的**依赖相关条目**需重算（但不必重嵌入全部内容）；
3. **重嵌入**该文件的片段（只算变化的片段，按片段哈希去重）；
4. **更新向量索引**（HNSW 的插入/删除是局部的，不需要重建整个图）。

**未保存编辑**（编辑器内存态）：不能等保存——做法是在编辑器进程内维护**轻量增量索引**（当前文件的符号与片段），随每次编辑更新，并把它与会话请求一起发送或按需与服务端索引做 diff 合并。这也是为什么要区分「仓库索引」（持久）与「会话索引」（易变）。

### 4. 检索：四路融合 + 重排

```
查询（当前编辑位置 + 任务描述 + 最近 diff）
  ├─ 符号路：光标所在符号的定义/引用/实现（图扩展 1–2 跳）
  ├─ 词法路：标识符/字符串匹配（倒排）
  ├─ 向量路：语义相似片段（HNSW 近似检索）
  └─ 结构路：同目录/同模块/同名测试文件
→ 融合（按来源加权或学习式重排）→ 去重 → 预算裁剪 → 按依赖顺序组装
```

**重排的必要性**：多路召回会引入大量「看起来相关但无用」的片段；可引用的实证方向来自 CodeRAG-Bench——检索增强代码生成的效果取决于**检索内容是否真的有用**，无关上下文会伤害生成。所以排序目标应是「对当前任务的信息增益」，而不是「与查询的相似度」。

### 5. 工程细节（决定能否落地）

- **忽略规则**：`.gitignore` + 二进制探测 + 大文件阈值 + 生成目录黑名单（否则 `node_modules` 会淹没一切）。
- **批处理与背压**：索引任务入队，按优先级（当前打开文件 > 当前目录 > 其余）调度，避免全量索引阻塞交互。
- **分支与工作区**：内容寻址天然支持跨分支复用未变文件；分支切换只重算差异部分。
- **版本与迁移**：索引带 schema 版本与嵌入模型版本，换模型要能**双写/影子重建**，避免用户侧突然质量下降。
- **隐私与权限**：索引落在本地还是服务端是产品决策；若上传，需按仓库权限隔离（串 [[rag-08]] 的权限感知检索）。

## 数值与代码验证

### 表 1：全量索引的规模账（10 万文件）

| 项 | 数值 | 说明 |
| --- | --- | --- |
| 文件数 / 行数 | $10^5$ / $2\times10^7$ | 每文件 200 行 |
| 总 token | $1.6\times10^8$ | 每行 8 token |
| 片段数 | $5\times10^5$ | 每 40 行一片段 |
| 嵌入成本（自建小模型 \$0.02/M） | **\$3.2** | 成本瓶颈不在这里 |
| 向量存储（768 维 float32） | 1.5 GiB | int8 量化 ≈0.4 GiB |
| 纯计算时间 | 约 10 s（50k 片段/s） | 实际受解析与 I/O 限制 |

### 表 2：增量更新的代价对比

| 场景 | 朴素做法 | 增量做法 |
| --- | --- | --- |
| 改一个文件（1 个符号） | 重建全量（分钟到小时） | 重解析该文件（毫秒）+ 反向依赖（数十文件） |
| 分支切换（差异 200 文件） | 全量重建 | 只重算 200 文件（其余按哈希复用） |
| 用户连续输入（未保存） | 无法索引 | 会话内轻量索引（每次编辑毫秒级） |

### 可运行代码

```python
# 1) Merkle 式目录摘要：O(1) 判定「整棵树没变」，并按子树定位变化
import hashlib, os, json

def h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def build_tree(files: dict):
    """files: 相对路径 -> 内容；返回 (根摘要, 逐层摘要表)"""
    file_hash = {p: h(c) for p, c in files.items()}
    dirs = {}
    for p, fh in file_hash.items():
        parts = p.split("/")
        for i in range(len(parts)):
            d = "/".join(parts[:i]) or "."
            dirs.setdefault(d, {})[p] = fh
    dir_hash = {}
    for d in sorted(dirs, key=lambda x: -x.count("/")):
        entries = []
        for p in sorted(dirs[d]):
            rest = p[len(d)+1:] if d != "." else p
            child = rest.split("/", 1)
            if len(child) == 2:                          # 子目录
                sub = (d + "/" + child[0]) if d != "." else child[0]
                entries.append((child[0] + "/", dir_hash.get(sub, "")))
            else:
                entries.append((child[0], dirs[d][p]))
        dir_hash[d] = h(json.dumps(entries))
    return dir_hash.get(".", ""), file_hash, dir_hash

def changed_files(old, new):
    """返回内容变化的文件列表（走哈希比对，不读内容）"""
    return sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))

snap1 = {"src/a.py": "print(1)\n", "src/b.py": "x=1\n", "docs/r.md": "hi\n"}
snap2 = dict(snap1); snap2["src/a.py"] = "print(2)\n"
r1, f1, d1 = build_tree(snap1)
r2, f2, d2 = build_tree(snap2)
print(f"根摘要相同？ {r1 == r2}（O(1) 判定整棵树是否变化）")
print("变化的文件：", changed_files(f1, f2))
print("变化的目录：", [k for k in d1 if d1[k] != d2.get(k)])

# 2) 增量 vs 全量的代价模型
PARSE_MS_PER_FILE, EMBED_MS_PER_CHUNK, CHUNKS_PER_FILE = 3.0, 0.02, 5
FILES = 100_000
def full_rebuild(files=FILES):
    return files*PARSE_MS_PER_FILE/1000, files*CHUNKS_PER_FILE*EMBED_MS_PER_CHUNK/1000
def incremental(changed_files, dependents=20):
    parse = (changed_files + dependents)*PARSE_MS_PER_FILE/1000
    embed = changed_files*CHUNKS_PER_FILE*EMBED_MS_PER_CHUNK/1000
    return parse, embed
fp, fe = full_rebuild()
print(f"\n全量重建：解析 {fp:,.0f} s + 嵌入 {fe:,.1f} s（串行口径）")
for c in (1, 10, 200):
    ip, ie = incremental(c)
    print(f"  变更 {c:>3} 个文件：解析 {ip:>6.3f} s + 嵌入 {ie:>6.3f} s "
          f"→ 比全量快约 {fp/max(ip,1e-9):,.0f}x（解析部分）")

# 3) 四路检索融合的简化演示（按来源加权）
CANDIDATES = [
    ("符号路", "src/auth.py::verify_token", 0.95, 0.5),
    ("词法路", "src/auth.py::verify_token", 0.80, 0.2),
    ("向量路", "src/session.py::check_session", 0.72, 0.2),
    ("结构路", "tests/test_auth.py::test_verify", 0.60, 0.1),
]
WEIGHT = {"符号路": 1.0, "词法路": 0.7, "向量路": 0.6, "结构路": 0.5}
merged = {}
for src, item, score, _ in CANDIDATES:
    merged[item] = merged.get(item, 0.0) + WEIGHT[src]*score
print("\n融合排序（同一条目被多路召回会累积权重）：")
for item, s in sorted(merged.items(), key=lambda x: -x[1]):
    print(f"  {s:>5.3f}  {item}")
print("注意：融合后要按「信息增益」重排并去重，否则无关片段会挤占预算（CodeRAG-Bench 的教训）")
```

预期输出要点：Merkle 摘要演示 O(1) 判断整树未变、并精确定位变化的文件与目录；增量代价模型显示改 1 个文件比全量重建快约 3–4 个数量级；融合排序演示多路召回如何合并（同一条目被多路召回权重累积），并提示必须重排去重。

## 常见追问

- **追问**：为什么不全用向量检索？
  - 要点：符号检索精确、便宜、可解释、易增量；向量检索擅长「不知道叫什么」的场景。代码里最常见的查询是「这个符号在哪、谁用它」，符号索引直接赢。
- **追问**：索引该放在本地还是服务端？
  - 要点：看隐私与延迟要求。本地索引延迟最低、隐私最好，但受客户端算力限制；服务端索引能集中算力与复用（跨用户共享公共库），但需要权限隔离与上传策略。实际产品常是「本地轻量 + 服务端增强」的混合。
- **追问**：如何处理 `node_modules` 与生成代码？
  - 要点：忽略规则 + 只索引被实际 import 的依赖（按需索引）、生成代码打标记不进默认检索；关键是把「用户会编辑的代码」与「只读的依赖」区分开。
- **追问**：索引一致性怎么保证（例如解析失败的文件）？
  - 要点：解析失败要降级到词法/文本索引而不是丢弃；记录索引覆盖率与失败率作为可观测指标；schema 升级用影子重建 + 双读验证。
- **追问**：10 万文件的冷启动要多久，用户能接受吗？
  - 要点：分层可用——元数据秒级、符号分钟级、向量后台补齐；并在补齐期间用符号+词法兜底。产品上要给出进度与「已可用」的明确信号。
- **追问**：怎么衡量索引的质量？
  - 要点：检索命中率（recall@k，用人工标注的「真正相关」集合）、上下文利用率（被模型实际引用的比例）、以及端到端任务成功率；可引用的评测方向来自 RepoBench（仓库级补全）与 CodeRAG-Bench（检索增强代码生成的有效性）。

## 相关题目

- [[cursor-07]]：为什么长上下文不能取代检索，本题的检索层是它的落地实现。
- [[cursor-10]]：tab 预测的延迟预算，决定索引必须在几十毫秒内返回候选。
- [[rag-01]]：chunking 策略，对应本题「片段粒度（函数级 vs 行级）」的取舍。
- [[rag-08]]：权限感知检索，是本题多用户/仓库隔离部分的方法论。
- [[system-design-02]]：代码助手的仓库索引与上下文组装的上位设计。

## 参考资料与归属

- **Cursor 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-cursor-work>。第 1 节与第 4 节代码检索分层与索引工程背景参照这篇。
- **Claude Code 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-claude-code-work>。第 4 节「按需检索 + 工具化取用」的工程取向参照这篇。
- **RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems（延伸）** —— Liu et al.，2023-06-05：<https://arxiv.org/abs/2306.03091>。第 5 节「仓库级补全的检索与评测口径」来自这篇。
- **CodeRAG-Bench: Can Retrieval Augment Code Generation?（延伸）** —— Wang et al.，2024-06-20：<https://arxiv.org/abs/2406.14497>。第「一句话答案」与第 4 节「检索内容必须真的有用、无关上下文会伤害生成」的结论来自这篇。
- **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs（HNSW）（延伸）** —— Malkov & Yashunin，2016-03-30：<https://arxiv.org/abs/1603.09320>。第 1 节向量索引「分层图 + 对数级检索」的结构依据来自这篇。
- **Merkle tree（维基百科）（延伸）** —— Wikipedia：<https://en.wikipedia.org/wiki/Merkle_tree>。第 3 节「子树摘要相同即可判定内容一致」的性质来自该条目。
- **延伸来源说明**：表 1、表 2、以及三段可运行代码中的全部数值与参数（每文件 200 行、8 token/行、40 行一片段、\$0.02/百万 token、768 维 float32、解析 3 ms/文件、融合权重）都是按本仓库统一口径构造的工程算例与显式假设，不是上述来源的原文数字；来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
