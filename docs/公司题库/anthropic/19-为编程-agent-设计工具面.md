---
type: question
id: anthropic-19
company: Anthropic
topic: agents
order: 19
question: 为编程 agent 设计工具面（tool surface）：有哪些工具、它们的 schema 长什么样、结果如何返回。
question_en: Design the tool surface for a coding agent: which tools, what their schemas look like, and how results come back.
asked_at: []
level: 高阶
tags: [Agent, 工具设计, schema, 错误信息, 上下文效率]
sources:
  - title: Building Effective Agents（延伸）
    url: https://www.anthropic.com/engineering/building-effective-agents
    author: Anthropic
    published: 2024-12-19
  - title: Claude Code 是如何工作的？
    url: https://outcomeschool.com/blog/how-does-claude-code-work
    author: Amit Shekhar (Outcome School)
    published: 
  - title: SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）
    url: https://arxiv.org/abs/2310.06770
    author: Jimenez et al. (ICLR 2024)
    published: 2023-10-10
related: [anthropic-18, cursor-08, cursor-11, cursor-13, agents-03]
updated: 2026-09-28
---

## 一句话答案

> 工具面的设计目标可以压缩成一句话：**让模型用最少的轮次、最少的 token，拿到「可行动的信息」，并且只能做安全的事**。由此推出六条设计原则：
> ① **少而正交**——工具数量不宜多（每个工具都要占据上下文与决策空间）；宁可给一个参数化的 `search` 也不给十个专用查询工具；
> ② **语义化命名与严格 schema**——名字说明「做什么」而不是「怎么实现」；参数必填/选填明确、有边界（`path` 限制在工作区内、`limit` 有上限），并给出**参数说明**（模型据此决定怎么用）；
> ③ **结果要「可行动」**——返回结构化的位置信息（文件:行、符号名）+ 必要上下文，而不是整文件转储；错误要说明**期望 vs 实际 + 下一步建议**；
> ④ **结果要「省 token」**——分页/截断（明确标注「已截断，可续读」）、按需展开（先给摘要与位置，模型再取详情）、去重（同一文件多次读只回变化部分）；
> ⑤ **安全边界内建在工具里**——路径限制、命令白名单、沙箱、危险操作需确认（不要让「安全」依赖提示词约束）；
> ⑥ **幂等与可观测**——同一调用可安全重试；每次调用留痕（谁、何时、参数、结果摘要）以便审计与回放。
> 一句话判据：**工具是模型与系统之间的 API，要按「给人用的 API」的标准设计**——清晰、最小权限、可诊断、可组合。

## 面试官在考什么

- **能否给出一个具体的最小工具集**：例如 `read_file`、`search_code`（符号/正则）、`edit_file`（结构化 hunk）、`run_command`（白名单）、`run_tests`；并说明**为什么不需要更多**（每个工具都是上下文与决策负担）。
- **schema 的细节意识**：必填与选填、范围约束、枚举值、以及**描述文本**如何影响模型行为（「描述是提示词的一部分」）。
- **结果格式的取舍**：整文件 vs 片段 vs 摘要 + 位置；行号是否稳定；大数据如何分页；二进制/超大文件如何拒绝。能否给出「结果 token 预算」的意识。
- **错误信息的质量**：区分「参数错（模型可修）」、「目标不存在（模型可换目标）」、「权限不足（模型不能修）」、「环境坏了（应上报）」——**四类错误对应四种不同的模型行为**。
- **安全边界的归属**：把限制放在工具实现里（路径规范化、白名单、沙箱）而不是系统提示里；危险操作走审批（串 [[agents-11]]）。
- **可测试性**：工具应能用**回放/模拟**测试（不依赖真实网络与仓库），并有 schema 校验与契约测试。
- **与上下文管理的关系**：工具结果直接进上下文，所以「结果太大」会挤掉其他信息——工具设计必须与上下文预算联合设计（串 [[anthropic-18]]）。

**常见错误答案**

- 给一大堆细碎工具（`get_file_content`、`get_file_lines`、`get_file_metadata`…）——每个都要模型去选，决策成本高，且容易误用。
- `write_file(path, content)` 作为唯一的写工具——无法做原子变更集与并发检查（串 [[cursor-13]]）。
- 返回整文件内容——token 浪费且稀释注意力。
- 错误只返回堆栈或 「failed」——模型无法据此修正。
- 用系统提示约束「不要读工作区外的文件」——提示不是安全边界，**工具实现才是**。

## 原理与推导

### 1. 最小工具集（5 个工具覆盖 95% 场景）

| 工具 | 作用 | 关键参数 | 返回 |
| --- | --- | --- | --- |
| `read_file` | 读文件片段 | `path`、`start_line`、`end_line`（可选） | 带行号的文本 + 总行数 + 是否截断 |
| `search_code` | 按符号/正则/语义检索 | `query`、`mode`（symbol/regex/semantic）、`limit` | 命中列表（文件:行 + 上下文 1–3 行）+ 总数 |
| `edit_file` | 结构化编辑 | `path`、`hunks[]`（锚点 + old + new） | 每个 hunk 的应用结果（applied/conflict/failed + 诊断） |
| `run_command` | 执行白名单命令 | `cmd`、`timeout_s`、`cwd` | 退出码 + stdout/stderr（截断）+ 耗时 |
| `run_tests` | 跑受影响测试 | `paths`（可选）、`pattern`（可选） | 通过/失败数 + 失败用例摘要（文件:行 + 断言差异） |

**为什么是这 5 个**：读、找、改、跑、验——覆盖 agent 循环的「行动」与「验证」；其余需求用参数表达（例如 `search_code` 用 `mode` 区分符号/正则/语义），而不是新增工具。

### 2. schema 的形状与描述的重要性

```json
{
  "name": "edit_file",
  "description": "对单个文件应用结构化编辑。每个 hunk 用锚点定位；应用前会校验文件未被他人修改。",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "工作区内的相对路径，不允许 .. 或绝对路径"},
      "hunks": {
        "type": "array", "minItems": 1, "maxItems": 20,
        "items": {
          "type": "object",
          "properties": {
            "anchor": {"type": "string", "description": "唯一可定位的上下文（含符号名或前后各 2 行）"},
            "old": {"type": "string", "description": "要替换的原文，必须与实际内容（忽略行尾空白后）一致"},
            "new": {"type": "string", "description": "替换后的内容"}
          },
          "required": ["anchor", "old", "new"]
        }
      }
    },
    "required": ["path", "hunks"]
  }
}
```

**三条经验**：
- **描述里写约束**（「不允许 `..`」、「必须与实际一致」）——模型会读描述并据此避免常见错误；
- **给数量上限**（`maxItems: 20`）——防止一次提交巨大变更，便于失败归因；
- **可选参数尽量少**——每个可选参数都是模型可能误用的入口。

### 3. 结果格式：四种形态与选择

| 形态 | 示例 | 适用 | 代价 |
| --- | --- | --- | --- |
| 结构化位置 | `src/auth.py:118: def verify_token(` | 搜索命中 | 极省 token，但模型可能要再读一次 |
| 片段 + 行号 | 前 3 行 + 命中行 + 后 3 行 | 大多数读操作 | 适中 |
| 摘要 + 指针 | 「该文件 812 行，含 14 个函数，签名为…」 | 大文件/概览 | 省 token，信息有损 |
| 完整内容 | 整文件 | 小文件精读 | 昂贵，慎用 |

**截断纪律**：任何截断都必须**显式标注**（"已显示 1–200 行，共 812 行；用 `start_line=201` 继续"），否则模型会以为自己看到了全部——这是 agent 幻觉的常见来源。

### 4. 错误的四分类与对应的模型行为

| 类别 | 例子 | 返回内容 | 期望模型行为 |
| --- | --- | --- | --- |
| 参数错（可自修） | 路径写错、`start_line > 总行数` | 合法范围/最近似路径 | 修正参数重试 |
| 目标不存在（可换目标） | 符号找不到、文件已删除 | 相似命中（模糊匹配 top-3） | 换用相似目标或搜索 |
| 权限/策略拒绝（不可自修） | 越界路径、命令不在白名单 | 明确原因 + 允许的替代做法 | 停止该路径，换方案或上报 |
| 环境故障（应上报） | 测试环境起不来、依赖缺失 | 区分为环境问题 + 建议 | 报告阻塞，而不是反复重试 |

**核心**：让模型能从错误里**学到下一步该做什么**。只返回 `「error」: 「invalid」` 等于浪费一轮。

### 5. 安全：边界属于工具实现，不属于提示词

- **路径规范化**：解析符号链接后校验仍在工作区内（防 `../`、绝对路径、symlink 逃逸）；
- **命令白名单**：只允许预定义命令（build/test/lint），任意 shell 需显式审批；
- **资源限制**：单命令超时、输出截断、内存/磁盘配额；
- **网络**：默认关闭或白名单；
- **审计**：每次调用记录参数与结果摘要（可回放）。
这与 [[cursor-08]] 的四道闸是同一套原则：**隔离、原子、验证、可回滚**。

### 6. 上下文效率：工具结果也要「预算」

设上下文预算 $C$（例如 32K token），其中工具结果占比通常最高。设计原则：

$$\text{单次工具结果}\ \le\ \frac{C}{k}\quad(k\approx10\text{，保证还能装下多次往返})$$

按此，单次 `read_file` 默认返回 ≤2K token 的片段、`search_code` 默认 ≤20 条命中、`run_command` 输出截断到 ≤4K token（保留首尾，中间省略并标注）。

## 数值与代码验证

### 表 1：工具返回形态的 token 成本（同一任务：定位并修改一个函数）

| 形态 | 返回内容 | token 量级 | 需要几轮 |
| --- | --- | --- | --- |
| 整文件 | 812 行源码 | ~8,000 | 1 轮读 + 1 轮改 |
| 片段（命中 ±3 行） | 约 30 行 | ~250 | 1 轮读 + 可能再读 1 次 |
| 结构化位置 + 摘要 | 10 个命中位置 | ~120 | 1 轮读（定位）+ 1 轮精读 + 1 轮改 |

**读法**：省 token 的形态往往**多一轮**——所以真正的判据是「**总 token 成本 × 轮数**」，而不是单次返回大小。经验上「结构化位置 + 按需精读」在长任务里总成本最低（因为上下文不被一次性污染）。

### 表 2：错误信息的质量对修正轮数的影响（示意）

| 错误返回 | 模型需要的额外轮数 | 说明 |
| --- | --- | --- |
| `「error」: 「failed」` | 2–4（先猜再试） | 基本等于没给信息 |
| `「file not found」` | 1–2（去搜索） | 有方向但不具体 |
| `"file not found; closest: src/auth/token.py (0.91)"` | 1（直接改用） | 可行动 |
| `「path escapes workspace: ../../etc/passwd」` | 0–1（换路径） | 明确不可自修的原因 |

### 可运行代码

```python
# 工具面的最小实现：schema 校验 + 路径安全 + 可行动错误 + 截断纪律（可离线跑）
import json, os, re, tempfile, textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

WORKSPACE = tempfile.mkdtemp()
os.makedirs(os.path.join(WORKSPACE, "src"), exist_ok=True)
SRC = os.path.join(WORKSPACE, "src", "auth.py")
with open(SRC, "w", encoding="utf-8") as f:
    f.write("import hmac\n\n\ndef verify_token(token, secret):\n    return hmac.compare_digest(token, secret)\n\n\n"
            "def issue_token(user, ttl=3600):\n    return f'{user}:{ttl}'\n")

class ToolError(Exception):
    """错误分四类，每类携带可供模型下一步使用的信息"""
    def __init__(self, kind: str, message: str, hints: Optional[Dict[str, Any]] = None):
        self.kind, self.message, self.hints = kind, message, hints or {}
    def to_dict(self):
        return {"ok": False, "error": {"kind": self.kind, "message": self.message, **self.hints}}

def resolve(path: str) -> str:
    """① 路径安全：规范化后必须仍在工作区内（防 ../ 与绝对路径）"""
    if os.path.isabs(path):
        raise ToolError("policy", f"不允许绝对路径：{path}", {"use": "工作区内的相对路径"})
    full = os.path.realpath(os.path.join(WORKSPACE, path))
    if not full.startswith(os.path.realpath(WORKSPACE) + os.sep):
        raise ToolError("policy", f"路径越界：{path}", {"workspace": WORKSPACE})
    return full

def tool_read_file(path: str, start_line: int = 1, end_line: Optional[int] = None,
                   max_tokens: int = 2000) -> Dict[str, Any]:
    try:
        full = resolve(path)
    except ToolError as e:
        return e.to_dict()
    if not os.path.exists(full):
        # ② 目标不存在：给出最相似的候选（可行动）
        import difflib
        cands = []
        for root, _, files in os.walk(WORKSPACE):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), WORKSPACE)
                cands.append(rel)
        close = difflib.get_close_matches(path, cands, n=3, cutoff=0.5)
        return ToolError("not_found", f"文件不存在：{path}", {"closest": close}).to_dict()
    lines = open(full, encoding="utf-8").read().split("\n")
    total = len(lines)
    if start_line > total:
        # ③ 参数错：给出合法范围
        return ToolError("bad_arg", f"start_line={start_line} 超过总行数 {total}",
                         {"valid_range": [1, total]}).to_dict()
    end = min(end_line or total, total)
    # ④ 截断纪律：估算 token（粗略按 4 字符/token），超出就截断并显式标注
    out_lines, used = [], 0
    for i in range(start_line - 1, end):
        used += len(lines[i]) // 4 + 1
        if used > max_tokens:
            return {"ok": True, "path": path, "start_line": start_line, "end_line": i,
                    "total_lines": total, "truncated": True, "next_start_line": i + 1,
                    "content": "\n".join(f"{n+1:>5}| {l}" for n, l in enumerate(out_lines, start=start_line-1))}
        out_lines.append(lines[i])
    return {"ok": True, "path": path, "start_line": start_line, "end_line": end,
            "total_lines": total, "truncated": end < total,
            "next_start_line": end + 1 if end < total else None,
            "content": "\n".join(f"{n+1:>5}| {l}" for n, l in enumerate(out_lines, start=start_line-1))}

def tool_search_code(query: str, mode: str = "regex", limit: int = 20) -> Dict[str, Any]:
    if mode not in {"regex", "symbol"}:
        return ToolError("bad_arg", f"不支持的 mode={mode}", {"allowed": ["regex", "symbol"]}).to_dict()
    pat = re.compile(re.escape(query) if mode == "symbol" else query)
    hits = []
    for root, _, files in os.walk(WORKSPACE):
        for fn in files:
            p = os.path.join(root, fn)
            for i, line in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
                if pat.search(line):
                    hits.append({"file": os.path.relpath(p, WORKSPACE), "line": i,
                                 "text": line.rstrip()[:120]})
    return {"ok": True, "mode": mode, "total_hits": len(hits),
            "returned": min(len(hits), limit), "truncated": len(hits) > limit,
            "hits": hits[:limit]}

def tool_edit_file(path: str, hunks: List[Dict[str, str]]) -> Dict[str, Any]:
    try:
        full = resolve(path)
    except ToolError as e:
        return e.to_dict()
    if not os.path.exists(full):
        return ToolError("not_found", f"文件不存在：{path}").to_dict()
    if not hunks:
        return ToolError("bad_arg", "hunks 不能为空", {"min_items": 1}).to_dict()
    text = open(full, encoding="utf-8").read()
    snapshot = text
    results = []
    for idx, h in enumerate(hunks):
        old = h.get("old", "")
        if old not in snapshot:
            # 失败诊断：给出最相近的一行（可行动）
            import difflib
            close = difflib.get_close_matches(old.split("\n")[0], snapshot.split("\n"), n=1, cutoff=0.4)
            results.append({"hunk": idx, "status": "match_failed",
                            "closest": close[0] if close else None,
                            "hint": "内容可能已被修改，请基于 closest 重新生成该 hunk"})
            continue
        snapshot = snapshot.replace(old, h["new"], 1)
        results.append({"hunk": idx, "status": "applied"})
    if any(r["status"] != "applied" for r in results):
        return {"ok": False, "atomic": True, "written": False, "results": results,
                "error": {"kind": "conflict", "message": "存在未匹配的 hunk，未写入任何内容"}}
    open(full, "w", encoding="utf-8").write(snapshot)
    return {"ok": True, "atomic": True, "written": True, "results": results}

# ---- 演示 ----
print("① 读文件（默认片段 + 行号）")
r = tool_read_file("src/auth.py", 1, 5)
print("   ", {k: r[k] for k in ("ok", "total_lines", "truncated", "next_start_line")})
print(textwrap.indent(r["content"], "      "))

print("\n② 路径越界 -> policy 错误（模型不可自修，应换方案）")
print("   ", tool_read_file("../../etc/passwd"))

print("\n③ 文件不存在 -> 给出最相似候选（可行动）")
print("   ", tool_read_file("src/authh.py"))

print("\n④ 参数错 -> 给出合法范围")
print("   ", tool_read_file("src/auth.py", start_line=9999))

print("\n⑤ 搜索（结构化位置，省 token）")
print("   ", tool_search_code("verify_token", mode="symbol"))

print("\n⑥ 编辑：hunk 匹配失败时原子不写 + 诊断")
print("   ", tool_edit_file("src/auth.py", [{"anchor": "def verify_token", "old": "return hmac.compare_digest(token, secret)",
                                            "new": "return hmac.compare_digest(str(token), str(secret))"}]))
print("   ", tool_edit_file("src/auth.py", [{"anchor": "x", "old": "这段内容不存在", "new": "y"}]))

print("\n⑦ 截断纪律：max_tokens 很小的时候")
big = os.path.join(WORKSPACE, "src", "big.py")
open(big, "w", encoding="utf-8").write("\n".join(f"line_{i} = {i}" for i in range(500)))
r7 = tool_read_file("src/big.py", 1, None, max_tokens=50)
print("   ", {k: r7[k] for k in ("ok", "start_line", "end_line", "total_lines", "truncated", "next_start_line")})
print("    读法：截断被显式标注（truncated + next_start_line），模型知道自己没看到全部")
```

预期输出要点（实跑）：① 读文件返回**带行号的片段**与 `total_lines/truncated/next_start_line`；② 越界路径返回 `kind=「policy」`（模型应放弃该路径而不是重试）；③ 不存在的文件返回 **`closest` 候选**（模型可直接改用）；④ 越界的 `start_line` 返回**合法范围**；⑤ 搜索返回结构化命中列表（文件:行 + 摘要）；⑥ 编辑在任一 hunk 匹配失败时**原子不写入**并给出 `closest` 与提示；⑦ 小 token 上限触发**显式截断**并告知续读位置。

## 常见追问

- **追问**：工具太多有什么坏处？
  - 要点：① 每个工具的 schema 都占上下文；② 模型要在更多选项里决策（误用率上升）；③ 工具之间的语义重叠会导致行为不一致。经验做法是**先做少而正交的工具集，用参数扩展能力**；新增工具要有明确的「现有工具做不到」的证据。
- **追问**：怎么让工具结果对模型最有用？
  - 要点：位置优先（文件:行/符号名）、必要上下文、明确的总量与截断、可续读的指针；错误给期望 vs 实际 + 建议。**判据是「模型下一轮能不能只靠这个结果行动」**。
- **追问**：如何测试工具面？
  - 要点：schema 契约测试（参数校验/边界）+ 行为测试（路径逃逸、白名单、原子性、截断标注）+ 回放测试（用固定工具结果集离线跑 agent 循环，保证可复现）；把历史上工具误用导致的失败固化成回归用例。
- **追问**：模型误用工具（参数写错）怎么办？
  - 要点：schema 收紧（枚举/范围/必填）→ 描述里写清约束 → 错误信息给合法取值 → 若仍频繁误用，说明工具抽象与模型直觉不符（考虑改名/合并/拆参）。
- **追问**：读操作要不要缓存？
  - 要点：要——同一文件在同一轮里被多次读取时只回一次；文件变更后缓存失效（按内容哈希）；这既省 token 也避免模型基于旧内容决策。
- **追问**：写操作如何防止破坏？
  - 要点：结构化 hunk（不整文件覆盖）+ 原子变更集 + 应用前的内容哈希检查 + 应用后的语法/类型校验 + 可回滚（串 [[cursor-13]] 与 [[cursor-08]]）。

## 相关题目

- [[anthropic-18]]：agent 循环与「模型 vs harness」，本题是其中「行动/观察」两段的展开。
- [[cursor-08]]：多文件改动的四道闸（隔离/原子/验证/回滚），是本题写工具的安全基础。
- [[cursor-11]]：仓库索引与检索，决定 `search_code` 背后能提供什么。
- [[cursor-13]]：结构化 hunk 的两级匹配与失败诊断，是本题 `edit_file` 的实现细节。
- [[agents-03]]：工具调用与 function calling 的机制，是本题 schema 设计的底层约定。

## 参考资料与归属

- **Building Effective Agents（延伸）** —— Anthropic，2024-12-19：<https://www.anthropic.com/engineering/building-effective-agents>。第 1 节「工具设计要清晰、最小、可组合；只在必要时增加复杂度」的取向来自这篇。
- **Claude Code 是如何工作的？** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/how-does-claude-code-work>。第 1 节工具集与工具结果形态的产品侧背景参照这篇。
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?（延伸）** —— Jimenez et al. (ICLR 2024)，2023-10-10：<https://arxiv.org/abs/2310.06770>。第「常见追问」里「用可执行判据与回放测试验证工具面」的思路来自这篇的评测范式。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（812 行文件、2K token 片段上限、20 条命中、4K 输出截断、四类错误的额外轮数估计）都是为演示本仓库口径而构造的示例与显式假设；schema 片段是示意而非任何产品的实际接口。来源仅用于原则与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
