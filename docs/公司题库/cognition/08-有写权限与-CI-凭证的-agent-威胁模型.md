---
type: question
id: cognition-08
company: Cognition（Devin、Windsurf）
topic: safety
order: 8
question: 一个自主 agent 拥有对客户仓库的写权限、CI 凭证以及网络访问权限。你的威胁模型是什么？
question_en: An autonomous agent has write access to a customer's repository, CI credentials, and network access. What is your threat model?
asked_at: []
level: 高阶
tags: [威胁模型, confused-deputy, prompt-injection, 最小权限, 防御纵深]
sources:
  - title: Don't Build Multi-Agents（Cognition 博客）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: 什么是 prompt injection 以及分层防御（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 在生产系统中如何让 agent 的操作可逆或至少可审计（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 为即将发布的模型设计一套红队测试方案（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cognition-05, cognition-07, cognition-09, safety-01, agents-10]
updated: 2026-09-28
---

## 一句话答案

> 先定性：**这个 agent 是一个典型的 confused deputy（被混淆的代理人）**——它**持有攻击者想要的权限**（仓库写、CI 凭证、出网），同时**读取攻击者可控的内容**（issue、README、代码注释、依赖名、测试数据、它被要求访问的网页）。攻击者不需要攻破你，**只需要让 agent 读到一段「看起来像指令的内容」**。
> 所以威胁模型的第一原则是：**仓库内容与外部内容永远是「数据」，永远不是「指令」**；执行策略必须在**模型之外**（策略网关/工具白名单/参数约束），不能依赖模型自觉。
> 四个资产、五类对手、七条攻击面、以及一一对应的控制：
> **资产**：① 客户源代码与 IP；② 仓库与 CI 里的机密（token、部署密钥、云凭证）；③ **通过 CI 可达的生产环境**（这是最被低估的，CI 常常能部署）；④ 平台自身与其他租户的数据。
> **对手**：外部攻击者（能影响 agent 读到的内容）、恶意租户/内部人、被投毒的上游依赖、以及**agent 自身**（被注入后成为攻击者的工具）。
> **七条攻击面**：① 仓库内容注入（**首位**）；② 依赖安装脚本（postinstall 即代码执行）；③ 机密外泄（有网就能 curl）；④ CI 凭证滥用（改工作流、自审自合、推 main）；⑤ 供应链（发布产物、改 lockfile、植入 Action）；⑥ 横向移动（沙箱逃逸、云元数据服务、集群 API）；⑦ 数据破坏（force push、删分支、批量删除）。
> **控制（纵深，缺一不可）**：内容与指令分离 + **把仓库当不可信输入**；**按任务下发短期最小权限凭证**（不落长期密钥）；**出网白名单 + 代理审计**；**禁止访问云元数据/集群 API**；沙箱强隔离（microVM、无宿主挂载）；**分支保护 + 只走 PR + 必需评审 + 工作流文件保护**；依赖安装禁脚本、锁文件校验；机密扫描与阻断；资源配额；全量审计 + 异常检测 + kill switch；**不可变备份**。
> 一句话判据：**「一个被完全注入的 agent，最多能造成多大破坏？」**——这个问题的答案就是你的真实安全水平；设计目标是让它**有界、可检测、可回滚**。

## 面试官在考什么

- **是否先定性为 confused deputy**：能否说清「agent 有权限 + 读不可信内容」这个组合本身就是风险核心（而不是笼统说「AI 有风险」）。
- **资产清单是否完整**：尤其**是否意识到 CI 常常能部署到生产**（很多团队忘了这一点）——这是本题最大的加分点。
- **攻击面是否具体**：能否举出**依赖 postinstall**、**改 GitHub Actions 工作流**、**自审自合 PR**、**云元数据服务（IMDS）**、**force push** 这些具体手法。
- **控制是否落在模型之外**：能否明确「执行策略在策略网关/工具层强制」，而不是「在提示词里要求模型不要这么做」。
- **最小权限的工程细节**：按任务、按仓库、短 TTL、最小 scope；沙箱内不落长期凭证；用 token broker 下发。
- **检测与响应**：能否给出可检测信号（批量删除、异常出网、非工作时间推送、工作流文件改动）、以及**kill switch 与回滚**（不可变备份）。
- **量化**：能否给出「纵深防御」的概率模型、**爆炸半径**（token scope × 暴露仓库数）、以及检测的 TPR/FPR 代价。
- **残余风险与可用性权衡**：知道控制太严会让 agent 无用；能给出**按任务类型分级**的权限策略（读任务 vs 小改 vs 迁移）。
- **诚实**：承认「**无法把风险降到零**」，因此要做的是**有界化 + 可检测 + 可回滚**，并把这个口径写进客户合同。

**常见错误答案**

- 只说「我们会用 prompt 让 agent 不要泄露密钥」（把安全寄托在模型自觉上）。
- 不提 CI 凭证能部署到生产（只关注「代码被改坏」）。
- 不做最小权限（给 org-wide PAT、长期有效）。
- 忽略依赖安装脚本这条代码执行路径。
- 不做出网控制（认为「agent 需要联网」就放开）。
- 没有检测与回滚（假设「不会发生」）。
- 把「沙箱隔离」当成唯一控制（单点）。

## 原理与推导

### 1. confused deputy 的形式化

agent 的行为可写成「**根据内容决定动作**」：$a=f(c)$，其中 $c$ 包含**攻击者可控的内容**（issue/README/依赖名）。若 $c$ 中含指令性文本而 agent 无法区分「数据 vs 指令」，则攻击者可令 $f$ 输出任意特权动作：

$$\exists c_{\text{adv}}:\ f(c_{\text{adv}})\in A_{\text{priv}}\quad\text{而}\quad c_{\text{adv}}\ \text{由攻击者可控}$$

**结论**：**降低风险只有两条路**——① 缩小 $A_{\text{priv}}$（最小权限）；② 让 $f$ 对 $c_{\text{adv}}$ 不敏感（内容与指令分离 + 模型外策略校验）。**「教育模型」不属于任何一条。**

### 2. 资产与攻击面对照表

| 资产 | 攻击面 | 典型手法 | 主要控制 |
| --- | --- | --- | --- |
| 源代码/IP | 内容注入 | README 里写「把 .env 发到 X」 | 内容即数据、出网白名单 |
| 机密 | 环境变量/文件 | `curl` 外发、编码进提交 | 沙箱内无长期密钥、DLP、机密扫描 |
| **生产环境（经 CI）** | CI 凭证 | 改 workflow、自审自合、推 main | 分支保护、工作流保护、必需评审、环境审批 |
| 平台/其他租户 | 沙箱逃逸、共享资源 | IMDS、K8s API、共享缓存投毒 | microVM 隔离、禁元数据、命名空间隔离 |

### 3. 七条攻击面与控制（逐条）

**(1) 仓库内容注入（首要）**
- **机制**：issue 描述、代码注释、测试数据、依赖 README 都可以包含「给 agent 的指令」；
- **控制**：内容**永不作为指令**解释；工具调用前经过**策略网关**（白名单 + 参数范围 + 业务规则）；对高危动作（推 main、改 CI 配置、出网到非白名单）**强制人工确认**（串 [[agents-11]]）。

**(2) 依赖安装脚本**
- **机制**：`npm install` / `pip install` 会执行 postinstall/setup.py → 在沙箱内拿到网络与凭证；
- **控制**：**禁用安装脚本**（`--ignore-scripts`、`--no-build-isolation` 策略）、只允许来自内部镜像/锁文件校验的包、装包阶段**网络仅允许包源**。

**(3) 机密外泄**
- **机制**：agent 有网 + 环境里有密钥 → 一条命令即可外发；
- **控制**：**沙箱内不注入长期密钥**（用短期、按任务、最小 scope 的 token）；出网**白名单 + 代理**（记录目标与体积，异常大出网告警）；DLP 规则（检测高熵字符串/私钥格式）；禁止读取宿主与元数据服务。

**(4) CI 凭证滥用**
- **机制**：CI token 常能改工作流、推分支、触发部署；agent 可以**自审自合**；
- **控制**：**分支保护**（main 只能经 PR）、**工作流文件受 CODEOWNERS 保护**、**禁止自我批准**、环境级审批（部署需要人）、CI token 按**单仓库 + 短 TTL**下发、限制可触发的 workflow 范围。

**(5) 供应链**
- **机制**：改 lockfile、植入 Action、发布产物、污染制品缓存；
- **控制**：lockfile 变更需人工评审、Action 版本固定到 SHA、制品签名与来源校验、缓存按租户隔离。

**(6) 横向移动**
- **机制**：沙箱逃逸、访问云元数据（IMDS 拿实例凭证）、访问集群 API/内部服务；
- **控制**：microVM 级隔离、**网络命名空间禁元数据地址**、无宿主挂载、服务账号最小权限、出网代理只放行白名单域名。

**(7) 数据破坏**
- **机制**：force push、删分支、批量删除文件、清空历史；
- **控制**：**禁止 force push 与分支删除**（服务端策略）、**不可变备份**（定期快照 + 对象锁）、批量删除的**速率限制与人工确认**、审计与告警。

### 4. 防御纵深的量化

设第 $i$ 层控制被绕过的概率为 $q_i$（独立近似），则一次成功外泄的概率：

$$P_{\text{exfil}}\approx\prod_{i=1}^{n}q_i$$

| 控制组合 | 各层绕过概率 | $P_{\text{exfil}}$ |
| --- | --- | --- |
| 仅提示词约束 | 0.30 | 3.0e-1 |
| + 出网白名单 | 0.30 × 0.10 | 3.0e-2 |
| + 沙箱无长期密钥 | ×0.05 | 1.5e-3 |
| + DLP + 审计告警 | ×0.30 | 4.5e-4 |
| + 短期凭证（TTL 15 min） | ×0.20 | 9.0e-5 |
| + 人工确认高危动作 | ×0.10 | 9.0e-6 |

**读法**：**单层控制永远不够**（提示词只有 0.3）；但**独立的多层能把概率压到 1e-5 量级**。这也是为什么「纵深」不是保险，而是**唯一可行的方法**（注意独立性假设是乐观的：同一根因可能同时击穿多层，所以真实值会更差）。

### 5. 爆炸半径

$$\text{暴露面}=\text{凭证 scope}\times\text{凭证 TTL}\times\text{沙箱可达网络}$$

| 策略 | scope | TTL | 泄露后可达仓库数 |
| --- | --- | --- | --- |
| org-wide PAT | 全组织 | 永久 | 数千 |
| 仓库级 token | 单仓库 | 24 h | 1（持续一天） |
| **任务级 token** | 单仓库 + 最小权限 | **15 min** | 1（窗口极短） |

**读法**：**scope 与 TTL 一起决定爆炸半径**；把两者都收到最小，是「即使被注入也不致命」的关键。

### 6. 检测与响应

| 信号 | 为什么有效 | 注意 |
| --- | --- | --- |
| 出网目标/体积异常 | 外泄必经网络 | 需基线（正常包源流量也要建模） |
| 批量删除/force push 尝试 | 破坏类动作的前兆 | 服务端策略应直接拒绝 |
| 工作流文件改动 | 提权路径 | 直接要求人工评审 |
| 非工作时段/异常频率的推送 | 自动化滥用 | 与正常 agent 行为区分 |
| 机密扫描命中 | 直接证据 | 需在**推送前**拦截 |
| 「尝试作弊率」（改测试/跳过） | 倾向的领先指标 | 即使被挡住也要记录 |

**响应**：kill switch（暂停租户/任务）、凭证吊销（短 TTL 天然帮助）、回滚（不可变备份 + 分支保护）、取证（完整轨迹与审计日志）、以及**客户通知流程**。

### 7. 残余风险与可用性权衡

| 任务类型 | 允许 | 需要确认 | 禁止 |
| --- | --- | --- | --- |
| 只读分析/搜索 | 读仓库、只读出网 | — | 写任何东西 |
| 小修 + 测试 | 提交到特性分支、开 PR | 改依赖/lockfile | 推 main、改 CI |
| 迁移/重构（大改） | 特性分支 + PR | 大范围删除、接口变更 | 同上 + 发布制品 |
| 部署/发布 | — | **全部人工** | agent 直接部署 |

**读法**：**按任务类型分级授权**是可用性与安全的平衡点——一刀切会导致 agent 无用或被绕过。

## 数值与代码验证

### 表 1：防御纵深（见代码输出）

| 控制组合 | $P_{\text{外泄}}$ | 相对仅提示词 |
| --- | --- | --- |
| 仅提示词 | 见输出 | 1× |
| + 出网白名单 | 见输出 | 见输出 |
| + 无长期密钥 | 见输出 | 见输出 |
| + DLP/审计 | 见输出 | 见输出 |
| + 短期凭证 | 见输出 | 见输出 |
| + 高危人工确认 | 见输出 | 见输出 |

### 表 2：爆炸半径与检测代价

| 策略 | 暴露仓库数（期望） | 说明 |
| --- | --- | --- |
| org-wide PAT（永久） | 见输出 | 最坏 |
| 仓库级 token（24 h） | 见输出 | 中 |
| 任务级 token（15 min） | 见输出 | 最好 |

| 检测器 TPR | FPR | 每日误报（10 万事件） | 需多少正样本才能宣称 |
| --- | --- | --- | --- |
| 0.90 | 0.01 | 见输出 | 见输出 |
| 0.95 | 0.001 | 见输出 | 见输出 |

### 可运行代码

```python
# 云端 coding agent 的威胁模型：防御纵深、爆炸半径、检测代价、注入成功率
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 防御纵深：多层独立控制的联合效果 ----------
@dataclass
class Control:
    name: str
    bypass: float          # 被绕过/失效的概率（独立近似）
    usability_cost: str    # 对可用性的影响

CONTROLS: List[Control] = [
    Control("提示词约束（不可作为唯一手段）", 0.30, "无"),
    Control("出网白名单 + 代理审计", 0.10, "低（需要维护域名清单）"),
    Control("沙箱内无长期密钥", 0.05, "低"),
    Control("DLP + 异常出网告警", 0.30, "中（误报需处置）"),
    Control("任务级短期凭证（TTL 15 min）", 0.20, "低（需 token broker）"),
    Control("高危动作强制人工确认", 0.10, "高（打断自动化）"),
]
print("① 防御纵深：逐层叠加后的外泄概率（独立近似，显示新增的那一层）")
p = 1.0
print(f"  {'新增控制':<34} {'本层绕过':>8} {'累计 P(外泄)':>13} {'相对首层':>10} {'可用性代价':<12}")
base = None
for c in CONTROLS:
    p *= c.bypass
    if base is None:
        base = p
    print(f"  + {c.name[:32]:<32} {c.bypass:>8.2f} {p:>13.2e} {base/p:>9.0f}x {c.usability_cost:<12}")
print("  读法：**提示词单独用只有 0.30 的拦截率**；六层叠加后到 1e-5–1e-6 量级 ——")
print("        但独立性假设是乐观的（同一根因可能同时击穿多层），真实值会更差")

print("\\n② 爆炸半径：凭证 scope × TTL × 网络可达性")
@dataclass
class Credential:
    name: str
    repos_reachable: int
    ttl_min: float
    egress_open: float          # 0-1：出网自由度（1=完全放开）
    def exposure(self, p_compromise_per_min: float = 0.002) -> Dict[str, float]:
        """期望暴露仓库数 ≈ 可达仓库 × 在 TTL 内被利用的概率 × 出网自由度"""
        p_window = 1 - math.exp(-p_compromise_per_min * self.ttl_min)
        return {"可达仓库": self.repos_reachable,
                "TTL 内被利用概率": p_window,
                "期望暴露仓库": self.repos_reachable * p_window * self.egress_open}
CREDS = [Credential("org-wide PAT（永久）", 2000, 60 * 24 * 365, 1.0),
         Credential("仓库级 token（24 h）", 1, 60 * 24, 0.8),
         Credential("任务级 token（15 min，出网白名单）", 1, 15, 0.2)]
print(f"  {'凭证策略':<34} {'可达仓库':>8} {'TTL 内被利用':>12} {'期望暴露仓库':>12}")
for c in CREDS:
    r = c.exposure()
    print(f"  {c.name:<34} {r['可达仓库']:>8,} {r['TTL 内被利用概率']:>12.2%} "
          f"{r['期望暴露仓库']:>12.3f}")
print("  读法：**scope 与 TTL 一起决定爆炸半径** —— 从 2000 个仓库的永久凭证收窄到")
print("        「单仓库 + 15 分钟 + 出网白名单」后，期望暴露从上千个降到 0.01 量级")

print("\\n③ 检测代价：TPR/FPR 与每日误报、样本量")
def detection_cost(events_per_day: int, tpr: float, fpr: float) -> Dict[str, float]:
    """TP 需要真实攻击数（假设 1 起/天）；FP 按事件量计"""
    tp_per_day = 1 * tpr
    fp_per_day = events_per_day * fpr
    return {"每日真报": tp_per_day, "每日误报": fp_per_day,
            "信噪比": tp_per_day / fp_per_day if fp_per_day else float("inf")}
print(f"  {'TPR':>6} {'FPR':>8} {'每日误报':>9} {'信噪比':>8} {'含义':<20}")
for tpr, fpr in ((0.90, 0.01), (0.95, 0.001), (0.99, 0.0001)):
    r = detection_cost(100_000, tpr, fpr)
    note = "不可用" if r["信噪比"] < 0.01 else ("可用但需人工分流" if r["信噪比"] < 1 else "可用")
    print(f"  {tpr:>6.2f} {fpr:>8.4f} {r['每日误报']:>9,.0f} {r['信噪比']:>8.3f} {note:<20}")
print("  读法：10 万事件/日、FPR=1% 就是**每天 1000 条误报**（信噪比 0.001，团队必然忽略告警）——")
print("        所以检测要做**高价值信号 + 分层处置**（自动阻断 vs 人工复核），不能只堆规则")

print("\\n④ 注入成功率：内容与指令分离、模型外策略校验的作用")
def injection_success(model_susceptibility: float, content_tagging: bool,
                      policy_gateway: bool, human_confirm_highrisk: bool,
                      high_risk_share: float = 0.3) -> float:
    """返回「注入后造成实质损害」的概率"""
    p = model_susceptibility
    if content_tagging:
        p *= 0.45                      # 明确标注不可信内容
    if policy_gateway:
        p *= 0.20                      # 工具白名单 + 参数校验（模型外）
    if human_confirm_highrisk:
        p *= (1 - high_risk_share) + high_risk_share * 0.05
    return p
print(f"  {'配置':<44} {'实质损害概率':>12}")
for label, ms, ct, pg, hc in (
        ("仅强模型（susceptibility 0.25）", 0.25, False, False, False),
        ("+ 不可信内容标注", 0.25, True, False, False),
        ("+ 模型外策略网关", 0.25, True, True, False),
        ("+ 高危动作人工确认（全部）", 0.25, True, True, True)):
    print(f"  {label:<44} {injection_success(ms, ct, pg, hc):>12.2%}")
print("  读法：**降风险的主力是「模型外」的两层（策略网关 + 高危人工确认）**，")
print("        而不是换更强的模型 —— 因为注入利用的是权限与通路的组合，不是模型智力")

print("\\n⑤ 控制矩阵：按任务类型分级授权（可用性 vs 安全）")
@dataclass
class TaskPolicy:
    kind: str
    read: bool
    write_branch: bool
    open_pr: bool
    touch_ci: bool
    deploy: bool
    def risk(self) -> int:
        return sum([self.write_branch, self.open_pr, self.touch_ci, self.deploy * 3])
POLICIES = [
    TaskPolicy("只读分析", True, False, False, False, False),
    TaskPolicy("小修 + 测试", True, True, True, False, False),
    TaskPolicy("迁移/重构", True, True, True, False, False),
    TaskPolicy("需要改 CI 配置", True, True, True, True, False),
    TaskPolicy("部署/发布", True, False, False, False, True),
]
print(f"  {'任务类型':<18} {'读':>3} {'写分支':>7} {'开 PR':>6} {'改 CI':>6} {'部署':>5} {'风险分':>6}")
for pol in POLICIES:
    print(f"  {pol.kind:<18} {'✓' if pol.read else '×':>3} {'✓' if pol.write_branch else '×':>7} "
          f"{'✓' if pol.open_pr else '×':>6} {'✓' if pol.touch_ci else '×':>6} "
          f"{'✓' if pol.deploy else '×':>5} {pol.risk():>6}")
print("  读法：**按任务类型分级授权**是关键 —— 一刀切（全禁或全放）要么让 agent 无用，")
print("        要么让「改 CI」与「部署」这两条最危险的路径敞开（它们直通生产）")

print("\\n⑥ 残余风险的沟通口径（写给客户的表）")
RISK = [("内容注入导致代码被改坏", "中", "分支保护 + PR 评审 + 不可变备份"),
        ("机密外泄", "低（收窄后）", "短 TTL + 出网白名单 + DLP"),
        ("经 CI 提权到生产", "低", "工作流保护 + 环境审批 + 禁止自审自合"),
        ("数据破坏（删除/force push）", "低", "服务端策略拒绝 + 不可变备份"),
        ("跨租户泄漏", "低", "沙箱不复用 + 命名空间隔离 + 缓存隔离")]
print(f"  {'风险':<26} {'残余等级':<12} 主要控制")
for r, level, ctrl in RISK:
    print(f"  {r:<26} {level:<12} {ctrl}")
print("  读法：**目标不是零风险，而是「有界、可检测、可回滚」** —— 这张表就是与客户对齐的口径")
```

预期输出要点（实跑）：① 防御纵深显示**提示词单独用只有 0.30 的拦截率**，六层叠加后外泄概率到 **1e-5–1e-6 量级**（同时提示独立性假设偏乐观）；② 爆炸半径显示**凭证从「org-wide 永久」收窄到「单仓库 + 15 分钟 + 出网白名单」后，期望暴露仓库从上千降到 0.01 量级**；③ 检测代价显示 10 万事件/日、FPR=1% 就是**每天 1000 条误报**（信噪比 0.001，团队必然忽略告警），所以必须做高价值信号 + 分层处置；④ 注入成功率的对照显示**降风险的主力是「模型外」的两层（策略网关 + 高危人工确认）**，而不是换更强的模型；⑤ 控制矩阵给出**按任务类型分级授权**（「改 CI」与「部署」是最危险的两条路径）；⑥ 残余风险表给出与客户对齐的沟通口径（有界、可检测、可回滚）。

## 常见追问

- **追问**：为什么不干脆禁止 agent 访问网络？
  - 要点：很多任务必须联网（装依赖、查文档、调模型 API）。正确做法是**出网白名单 + 代理审计**，而不是全禁或全放：全禁会让 agent 无用（装不了依赖），全放等于把外泄通道敞开。
- **追问**：怎么防止 agent「自审自合」？
  - 要点：**平台与服务端策略**：分支保护（必需人工评审）、禁止作者自批、工作流文件受 CODEOWNERS 保护、CI token 无审批权限。**这些必须在 GitHub/CI 侧强制，而不是靠 agent 的提示词。**
- **追问**：依赖安装脚本怎么办？
  - 要点：默认 `--ignore-scripts`；只允许内部镜像与锁文件校验过的包；装包阶段网络仅允许包源；对需要构建的包走**预先构建的内部制品**，而不是在沙箱里执行任意 setup 代码。
- **追问**：怎么检测「数据外泄」？
  - 要点：三层——① **网络层**（出网目标/体积基线、异常域名）；② **内容层**（DLP 高熵串/私钥格式、提交内容里的机密）；③ **行为层**（非工作时段、异常频率、批量读取敏感目录）。**并且要在推送前拦截，而不是事后发现。**
- **追问**：如果 agent 真被注入了，怎么止损？
  - 要点：**短 TTL 让凭证自然失效**；kill switch 暂停任务/租户；吊销凭证；**从不可变备份回滚**；取证（完整轨迹 + 审计日志）；客户通知。**关键是这些动作要演练过**，而不是写在文档里。
- **追问**：怎么向客户证明安全？
  - 要点：给**资产/对手/攻击面/控制**的矩阵 + **残余风险表** + **检测与响应 SLA** + 第三方审计/渗透测试报告；并明确「**我们不做零风险承诺，我们保证有界、可检测、可回滚**」。**把「能造成多大破坏的上界」讲清楚，比讲控制清单更有说服力。**
- **追问**：红队怎么测这套东西？
  - 要点：构造**注入语料**（issue/README/依赖名的恶意指令）、**依赖投毒**、**外泄尝试**、**提权尝试**（改 workflow）、**破坏尝试**（force push/批量删除），测**拦截率**与**检测时延**；并把这些用例固化成回归集（串 [[safety-10]]）。

## 相关题目

- [[cognition-05]]：执行环境的隔离与凭证下发，是本题控制的落地层。
- [[cognition-07]]：评估——「尝试作弊率」与「破坏回归」也是安全信号。
- [[cognition-09]]：九十天推广——安全评审是推广的前置关卡。
- [[safety-01]]：prompt injection 的分层防御，是本题第 (1) 条攻击面的通用版。
- [[agents-10]]：让 agent 的操作可逆与可审计，对应本题的备份、审计与回滚。

## 参考资料与归属

- **Don't Build Multi-Agents（Cognition 博客）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 1 节「动作携带隐式决策」的视角用于说明为什么**执行策略必须在模型之外校验**（模型对内容的解释不可作为授权依据）。
- **什么是 prompt injection 以及分层防御（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 3 节第 (1) 条攻击面的分层防御框架取自该专题文档。
- **在生产系统中如何让 agent 的操作可逆或至少可审计（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 6 节「审计、回滚与 kill switch」的做法取自该专题文档。
- **为即将发布的模型设计一套红队测试方案（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。常见追问中「注入语料 + 提权 + 外泄」的红队用例设计取自该专题文档。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（各层绕过概率 0.05–0.30、凭证 scope 与 TTL、每分钟被利用概率 0.002、出网自由度、事件量 10 万/日、TPR/FPR 组合、注入易感性 0.25 与各层折减系数、任务分级矩阵）都是为演示威胁建模方法而构造的**示例参数与显式假设**；真实评估必须用自己的攻击面清单与红队结果校准（**矩阵结构与「有界/可检测/可回滚」的口径是可直接使用的部分**）。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
