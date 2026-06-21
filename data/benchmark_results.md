---
name: collab-decision-rules
description: 多模型协作实证决策规则——三轮benchmark(简单8/复杂5/智能vs单5)的数据驱动结论
metadata: 
  node_type: memory
  type: project
  originSessionId: 99cdea57-05f2-4b75-8998-0e24e09525ab
---

# 多模型协作决策规则 (三轮 Benchmark)

## 测试汇总

### Round 1: 8 简单题
- Single: 88% acc, 1.9s avg
- Multi: 88% acc, 12.3s (6.6x)
- **简单题不协作**

### Round 2: 5 复杂DB题 (全协作)
- Single: 35k chars, 176 SQL, 130 Arch, 122s
- Full-Multi: 27k chars (-23%), 138 SQL (-22%), 157 Arch (+21%), 274s (2.2x)
- **代码密集题协作损害 SQL 细节**

### Round 3: 5 复杂DB题 (SMART自主决策)
- Single: 98% acc, 125s, 150 SQL, 128 Arch
- SMART: 90% acc, 194s, 126 SQL, 133 Arch
- **SMART 比全协作省 29% 时间，但准确性仍低于纯单模 8%**

## 终极结论

| 场景 | 最佳策略 | 原因 |
|------|---------|------|
| 简单问答/翻译 | 单 DS-PRO | 多模 6.6x slower 无收益 |
| 纯代码/SQL/算法 | 单 DS-PRO | 多模 -22% SQL，-8% 准确率 |
| 架构/设计/多方案 | 单 DS-PRO 为主 | 多模 +21% Arch 但 -8% acc，性价比不明确 |
| 开放式论述/写作 | 考虑协作 | Kimi+GLM 补充中文视角 |
| 知识边界问题 | 多模型 | 跨模型知识互补 |

## 核心教训
1. **DS-PRO 单打是最强基线** — 三轮测试中 pure single 多次最优
2. **合成步骤是质量杀手** — 压缩 4 个回答时必然丢失细节
3. **代码密集型伪装成架构题** — DB-4(身份图谱)被 SMART 误判为架构题，协作后质量下降
4. **多模型收益集中在 +Arch insight**，但这是以 -SQL detail 和 -accuracy 为代价的
5. **不要默认协作** — 除非明确需要多视角或跨模型知识互补

## 推荐默认行为
- 优先单 DS-PRO 直接答
- 仅当用户明确要求"多模型协作"或任务明显需要多视角时，才走协作
- 协作时保留各模型原始输出给用户参考，不只给合成结果

**Why:** 三轮真实 benchmark 数据表明：多模型协作不是银弹，DS-PRO 单打在大多数场景下是质量和效率的最优解。

**How to apply:** 默认单模型回答。用户说"协作"或用 brain.py 时才走多模型。简单任务绝不协作。
