---
feature_id: F-vX.Y.Z-NNNN                  # 形态: F-v{a.b.c}-{4 位数字}，与 PRD manifest 表行一致（版本=目标里程碑）
title: <feature 标题>
prd: PRD-vX.Y.Z                            # 该 feature 生效的 PRD 版本（版本化反链，须保留；master id 本身是去版本号的 PRD）
owner: unassigned                          # engineer slug 或 unassigned
status: proposed                           # proposed | specifying | ready | in_progress | in_review | done | blocked | dropped (§3.5)
started_at: "YYYY-MM-DD"
shipped_at: null
related_adrs: []                           # ADR id 数组（形如 ADR-vX.Y.Z-NNNN）
ac: []                                     # 带 id 的可验收条目：- { id: ac-1, text: "…" }（in_review→done 闸要求非空）
ambiguity_flags: []                        # 未决项（specifying→ready 闸要求清零；缺口走 teamos_raise_revision）
revision_refs: []                          # 关联的 RevisionRequest id（blocked→specifying 对账依据）
---

# F-vX.Y.Z-NNNN: <feature 标题>

## 概要

<!-- 一句话说明这个 feature 解决什么 -->

## 验收标准

- [ ] <验收点 1>
- [ ] <验收点 2>

## 时间线

| 日期 | 事件 | PR / commit |
|---|---|---|
| YYYY-MM-DD | proposed | — |

## 经验沉淀

<!-- 实现过程中值得记录的决策、坑、复用点 -->
