---
feature_id: F-v0.1.0-0001                  # 形态: F-v{a.b.c}-{4 位数字}，与 PRD manifest 表行一致（版本=目标里程碑）
title: 支持一次批量上传多个PDF
prd: PRD-v0.1.0
owner: ou_pdf_owner
status: ready                           # proposed | specifying | ready | in_progress | in_review | done | blocked | dropped (§3.5)
started_at: "2026-06-26"
shipped_at: null
related_adrs: []                           # ADR id 数组（形如 ADR-vX.Y.Z-NNNN）
ac:
  - { id: ac-1, text: "后端接口接收多个PDF并逐个入库,部分失败返回错误清单" }
  - { id: ac-2, text: "前端可多选PDF并显示各自上传进度,e2e通过" }                                     # 带 id 的可验收条目：- { id: ac-1, text: "…" }（in_review→done 闸要求非空）
ambiguity_flags: []                        # 未决项（specifying→ready 闸要求清零；缺口走 teamos_raise_revision）
revision_refs: []                          # 关联的 RevisionRequest id（blocked→specifying 对账依据）
---

# F-v0.1.0-0001: 支持一次批量上传多个PDF

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

