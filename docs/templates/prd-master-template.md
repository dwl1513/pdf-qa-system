---
# ── 机器可读 frontmatter（被 @teamos/shared 的 PrdFrontmatterSchema 校验）──
id: PRD                                  # 去版本号：一产品一 PRD，id 恒为 PRD（项目身份在 docs/PROJECT.md）
title: <PRD 标题>
# version 不写进 PRD：产品版本是 root package.json.version 的唯一真相（SSOT）
owner: <团队或负责人 slug>
status: draft                             # draft | active | revising | superseded
created_at: "YYYY-MM-DDT00:00:00Z"
updated_at: "YYYY-MM-DDT00:00:00Z"
adr_refs: []                              # ADR id 数组（形如 ADR-vX.Y.Z-NNNN）；被 verify 校验解析到 docs/adr/ 真实文件
code_refs: []                             # 由执行侧 sync 回填，勿手写

# ── 粒度棘轮三件套（§3.5；PRD 在自己身上跑自己定义的机器）──
granularity_level: coarse                 # 全局自评：本 PRD 整体细到哪了（coarse | refined | task_ready）
granularity_levels:                       # 按章节锚点的细化深度；prdCanReactivate 闸读这里判「粒度净增」
  # B2/§领域模型: refined
  # D3/Roadmap: coarse
open_questions: []                        # 未决项登记（每条由一个 RevisionRequest 关闭）；与 C1 表逐条对齐
  # - { id: OQ-1, text: "<一句话未决问题>", closed_by: rr-xxxx }   # closed_by 在 RR resolved 后回填
revision_log: []                          # 每次 revising→active 追加一行；记录粒度净增，勿删历史行
  # - { v: v0.1.0, date: "YYYY-MM-DD", by: <slug>, change: "<改了什么>",
  #     granularity_delta: "<哪些章节 coarse→refined→task_ready>", adr_added: [ADR-vX.Y.Z-NNNN], oq_closed: [OQ-1] }
---

<!--
═══════════════════════════════════════════════════════════════════════════════
 这是 master PRD 模板（一产品一 PRD，无 sub-PRD）。骨架分四层 + 附录：

   A 叙述(Why)   — 给人读：定位 / 愿景 / 角色动线 / 范围边界
   B 契约(What)  — 给人+机器读：术语 / 顶层架构 / 领域模型 / Feature manifest
   C 治理(迭代)  — 让 PRD 自我演化：未决项登记表 / NFR / 粒度链总览
   D 外置(索引)  — 只放指针，不放副本：ADR 索引 / 技术栈指针 / Roadmap
   附录           — Demo 剧本 / Schema 指针 / 词汇表

 富 PRD 原则：顶层设计（B1）与领域模型（B2）写厚；签名级 HOW（trait/config/
 工具 schema）一律外置为指针，不在 PRD 内留副本（见每层的红线注释）。
═══════════════════════════════════════════════════════════════════════════════

 ── 两轴标注法（每个主章节 / 领域实体头都打一个三元标）──

   ⟨治理态 · 粒度 · 实现⟩

   · 治理态（审批到哪）：draft | active | revising | superseded（或该章节的评审状态）
   · 粒度  （够细没）  ：coarse | refined | task_ready
   · 实现  （建了没）  ：✅ 已建（匹配当前代码）| ◻ 目标（待建）

   例：B2 领域模型 ⟨active · refined · ✅⟩   D3 Roadmap ⟨draft · coarse · ◻⟩

 ── 红线（违反即污染 SSOT，reviewer 直接打回）──

   1. enum_semantic（实体状态枚举语义）只写在 PRD（B2）—— 它是 What，是产品域。
   2. 物理 schema（zod 类型 / 正则 / 默认值 / 可选性）唯一 SSOT = packages/shared/src/schemas/*；
      PRD 只描述语义，不抄 zod。
   3. 签名级 HOW（trait 接口 / *.yaml 配置 / MCP 工具 input-output）外置为指针，指向 ADR / code / config；
      PRD 删副本、留一行链接。发现某决策无 ADR 承载 → 登记一条 OQ（不在正文补设计）。
-->

# <PRD 标题>

> 身份：`PRD`（去版本号；产品版本是 root package.json 的唯一真相）
> 维护者：<团队>
> 状态：draft
> 撰写日期：YYYY-MM-DD

---

## 目录

- [A. 叙述（Why）](#a-叙述why)
- [B. 契约（What）](#b-契约what)
- [C. 治理（迭代）](#c-治理迭代)
- [D. 外置索引（指针）](#d-外置索引指针)
- [附录](#附录)

---

# A. 叙述（Why）

> 〔填写说明〕本层给人读，回答「为什么做、给谁做、做到哪为止」。不下沉到接口签名。

## A1. 定位 ⟨draft · coarse · ◻⟩

> 〔填写说明〕这份 PRD 是什么 / 不是什么 / 目标读者。一句话定位放最前。

## A2. 愿景与价值主张 ⟨draft · coarse · ◻⟩

> 〔填写说明〕起源问题、价值主张、与传统形态的差异化。可用对比表。

## A3. 角色与使用者动线 ⟨draft · coarse · ◻⟩

> 〔填写说明〕角色集（含英文标识 + 继承关系）+ 每个角色的典型一天（动线）。
> MVP 可只列最小角色集，完整角色矩阵作为「预留」表附后。

## A4. 范围边界 ⟨draft · coarse · ◻⟩

> 〔填写说明〕把「必做 / 不做 / 非目标」写成结构化三块，防 scope creep。

```yaml
in_scope:        # 本里程碑必做
  - <必做项>
out_scope:       # 明确不做（附原因 + 计划版本）
  - { item: <不做项>, reason: <原因>, planned: <版本> }
non_goals:       # 永不做 / 设计上排除
  - <非目标>
```

---

# B. 契约（What）

> 〔填写说明〕本层是 PRD 的硬核：人 + 机器都要读。B2 领域模型是全仓唯一 SSOT 的语义层。
> 红线：本层只写语义与顶层结构；trait/config/工具 schema 外置（见 B1 末尾指针块）。

## B0. 概念与术语 ⟨draft · coarse · ◻⟩

> 〔填写说明〕术语速查表（中文 | 英文/缩写 | 一句话定义 | 首次详述章节）+ 概念关系图 + 命名约定。

## B1. 顶层架构总览 ⟨draft · refined · ◻⟩

> 〔填写说明〕分层总览图 + 各层核心心智（一句话）。**写厚**：这是富 PRD 的顶层设计承载点。
> 完整模块图 / 数据流 / 状态机 / 包依赖拓扑由 `docs/ARCHITECTURE.md` 权威维护，本节链接过去。

```
┌──────────── 分层总览（示意）────────────┐
│ ① <接入层>                               │
│ ② <控制面>                               │
│ ③ <内核能力>                             │
└──────────────────────────────────────────┘
```

> **HOW 指针（不在 PRD 留副本）**：
> - trait 接口（ChannelAdapter / AgentRuntime / ContextStore / …）→ `packages/shared/src/traits/*` ＋ 对应 ADR
> - 配置文件（*.yaml）→ `config/*.yaml`（文件本身即 SSOT）
> - 进程拓扑 / 端口 / 部署 → 对应 ADR
> - 完整架构图 → [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)

## B2. 领域模型（Domain Schema + 生命周期 DFA）⟨active · refined · ✅/◻ 混合⟩

> 〔填写说明〕**这是 PRD 最权威的一章**。定义分解脊柱（PRD → Feature → Task）+ ADR 横切 sidecar，
> 以及每个实体的 `字段 / enum_semantic(状态枚举语义) / lifecycle(DFA 转移) / invariants / links`。
> 每个实体头打两轴标，每个 state/边/字段标 `✅`(已实现) 或 `◻`(目标)。
> 红线：enum_semantic 写这里（What）；物理 zod schema 指向 `packages/shared/src/schemas/*`（SSOT）。

### B2.1 治理平面（git，分布式 SSOT）

> 〔填写说明〕PRD / ADR / Feature 三实体。每个按下列骨架写：
>
> ```
> **<实体名>** · <物理载体路径>
> - 字段：<field 列表，每个标 ✅/◻>
> - enum_semantic(status)：<state1 ✅> · <state2 ◻> …（指向 @teamos/shared 的转移表常量名）
> - lifecycle：<from→to：触发条件 + 闸（如有）+ ✅/◻>
> - invariants：<不变量>
> - links：→/←<关联实体>(<字段>)
> ```

### B2.2 执行平面（中央运行时，不进 git）

> 〔填写说明〕Task / Session / Ask / Approval / Project / Workspace / RunControl。同上骨架。
> 缝压在 Feature(治理末层) ↔ Task(执行首层)，`feature_id` 跨缝。

### B2.3 粒度增强回路 —— RevisionRequest（冒泡载体）

> 〔填写说明〕定义第五实体 RevisionRequest 及三条跨实体冒泡边（粒度棘轮）。
> 它把「上层没说清」结构化回灌 PRD，使粒度随迭代净增——是 C1 未决项表「长出状态」的执行平面载体。

```
Task.blocked    ──raise──▶ RevisionRequest.open
Feature.blocked ──raise──▶ RevisionRequest.open
ADR.proposed    ──gap────▶ RevisionRequest.open (+ ADR→parked)
                                │ owner triage → accepted
                                ▼
                     PRD.active→revising  (version++ / 粒度净增)
                                │ resolved
                                ▼
              Feature.blocked→specifying ; Task.blocked→defined
```

## B3. Feature manifest ⟨draft · refined · ◻⟩

> 〔填写说明〕被 `@teamos/verify` 机械解析的 feature 清单。详情留在 `docs/feature/F-<slug>.md`，本表只索引。
>
> 解析约定（verify 按列名读，忽略大小写 / 多余列 / 列序）：
> 1. 表前必须有 `<!-- @feature-manifest -->` marker。
> 2. 表头至少含 `Feature ID`、`Spec`（必需）；`Status`、`Owner`、`refines` 等列可选、verify 容忍。
> 3. `Feature ID` 匹配 `^F-v\d+\.\d+\.\d+-\d{4}$`。
> 4. `Spec` 是相对仓库根的路径，文件须存在，且其 frontmatter `feature_id` == 本行、`prd` == 本 PRD `id`。
> 5. `refines`（新增·建议）：本 feature 细化的 PRD 锚点（如 `B2/§领域模型`、`A4`）—— 让 PRD→Feature 分解链可追。

<!-- @feature-manifest -->

| Feature ID | 标题 | Status | Owner | Spec | refines | 关键依赖 |
|---|---|---|---|---|---|---|
| F-v0.1.0-0001 | <feature 标题> | proposed | unassigned | docs/feature/F-v0.1.0-0001-<slug>.md | <PRD 锚点> | — |

---

# C. 治理（迭代）

> 〔填写说明〕本层是「迭代中完善 PRD」的物理载体。C1 是粒度棘轮的治理镜像，让 PRD 从一次性写作变成数据驱动闭环。

## C1. 未决项 / 歧义登记表 ⟨active · refined · ✅⟩

> 〔填写说明〕**手维护**。这是 frontmatter `open_questions[]` 的人类可读视图，也是 `RevisionRequest`（B2.3）的**治理镜像**：
> 每条 OQ 与 frontmatter `open_questions[].id` 逐条对齐；当对应 RevisionRequest `resolved`，在 `closed_by` 回填 rr-id，并在 frontmatter 同步 `closed_by`。
> `teamos sync` **只对账告警、不覆写**本表（门不写 docs，ADR-0002）——对账实现作为后续 feature 延后。

| id | kind | targets | 描述 | raised_from | 阻塞(downstream) | severity | RR 状态 | closed_by |
|---|---|---|---|---|---|---|---|---|
| OQ-1 | ambiguity \| missing_requirement \| granularity_too_coarse \| conflict | §锚点 \| need_new_adr | <一句话未决问题> | <task/feature/adr/人> | <被它阻塞的 F/Task> | low \| high \| blocking | open \| triaged \| accepted \| resolved \| rejected | <rr-id 或空> |

## C2. 非功能性需求（NFR）⟨draft · coarse · ◻⟩

> 〔填写说明〕安全 / 可审计 / 可观测 / 性能 / 可恢复 / 兼容 / 可运维。每条：需求 | 实现 | 验证方式。

## C3. 粒度链总览图 ⟨draft · refined · ◻⟩

> 〔填写说明〕画 `PRD#§ → F-xxxx` 的 refines 实例图（取自 B3 manifest 的 refines 列）+ 冒泡回路，
> 让「哪个 feature 在细化 PRD 哪一节、缺口往哪冒泡」一图可见。

```
PRD#<§锚点> ──refines──▶ F-v0.1.0-0001
PRD#<§锚点> ──refines──▶ F-v0.1.0-0002
        ▲                      │ blocked
        └──── RevisionRequest ◀┘ (冒泡缺口回灌，粒度净增)
```

---

# D. 外置索引（指针）

> 〔填写说明〕本层**只放指针**，不放副本。技术决策正文在 ADR；技术栈细节在 code/config；本层是导航。

## D1. ADR 索引表 ⟨active · refined · ✅⟩

> 〔填写说明〕替代旧反模式「决策记录（已合并至各章节）」。技术路线 = 随版本增长的多 ADR。
> 每追加一个 ADR：本表加一行 + frontmatter `revision_log` 记 `adr_added` + `adr_refs[]` 加 id（verify 校验解析）。

| ADR | 标题 | implements_prd | status | 引入版本 |
|---|---|---|---|---|
| ADR-v0.1.0-0001 | <技术决策标题> | PRD-v0.1.0 | accepted | v0.1.0 |

## D2. 技术栈指针 ⟨draft · coarse · ◻⟩

> 〔填写说明〕一句话技术栈 + 指针。**不抄依赖清单**：以 root `package.json` / `pnpm-workspace.yaml` / `AGENTS.md` 技术栈表为准。

- 语言 / 运行时 / 包管理 / 测试 / 格式化 → 见 `AGENTS.md`「技术栈」表 + root `package.json`
- 关键依赖与分发方式 → 对应 ADR

## D3. Roadmap ⟨draft · coarse · ◻⟩

> 〔填写说明〕**单一路线图**（版本 → 时间窗 → 核心交付 → 关键能力）。远期能力以「接口预留」一句话带过，指向 trait/ADR，不展开实施细节。

| 版本 | 时间窗 | 核心交付 | 关键能力 |
|---|---|---|---|
| v0.1（MVP） | <窗口> | <交付> | <能力> |

---

# 附录

## 附录 A. 端到端 Demo 剧本 ⟨draft · coarse · ◻⟩

> 〔填写说明〕可演示的逐帧剧本（多 channel 视图）。作为验收的「可演示」判据。

## 附录 B. Schema 指针 ⟨active · task_ready · ✅⟩

> 〔填写说明〕**不留 zod 副本**。语义在 B2，物理 schema 唯一 SSOT 在 `packages/shared/src/schemas/*`。本表只做语义↔文件的导航。

| 实体 | 语义（What） | 物理 schema（SSOT，`packages/shared/src/`） |
|---|---|---|
| PRD / ADR / Feature | B2.1 | `schemas/{prd,adr,feature,feature-spec}.ts` |
| Task / Session / Ask / Approval / … | B2.2 | `schemas/{task,ask-state,approval,project,workspace,run-control}.ts` |
| RevisionRequest | B2.3 | `schemas/revision-request.ts` |

## 附录 C. 词汇表 ⟨draft · coarse · ◻⟩

> 〔填写说明〕缩写 | 全称 | 含义。
