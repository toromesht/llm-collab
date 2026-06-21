# SynapseFlow

> 多模型智能协作路由系统 — 让 AI 学会调度 AI

[![Evaluation](https://img.shields.io/badge/eval-98%20questions-blue)]()
[![Models](https://img.shields.io/badge/models-6-green)]()
[![Papers](https://img.shields.io/badge/papers-18-purple)]()

## 架构

```
DS-Think (架构师)
  ├─ DS-PRO (攻坚手)     ← 难题之王, V3 72%
  ├─ GLM   (中文脑)      ← 知识推理, V2 91%
  ├─ Kimi  (文书官)      ← 写作文档
  └─ QWEN  (多面手)      ← 通用任务
```

## 目录

```
SynapseFlow/
├── engine/                 调度引擎
│   ├── brain.py            决策路由 (v9, 难度自适应)
│   ├── reasoning.py        推理增强 (CoT + Self-Consistency)
│   └── vision.py           图像理解 (Kimi Vision)
├── eval/                   评价体系
│   ├── benchmark_v1.py     基础评测 (19题)
│   ├── benchmark_v2.py     交叉验证 (33题)
│   ├── benchmark_v3.py     深度难题 (46题)
│   └── weight_tuner.py     奖惩权重训练器
├── dataset/                训练数据
│   ├── math_modeling/      数学建模数据集
│   ├── competition/        竞赛模型使用记录 (27条)
│   └── decision_logs/      决策日志
├── papers/                 论文库
│   ├── routing/            路由方向 (14篇)
│   ├── nature/             Nature子刊 (4篇)
│   └── frontier/           前沿算法 (NeurIPS)
└── config/                 配置
    ├── llm-config.template.json
    └── brain_weights.json
```

## 三轮评测结果

| 模型 | V1 简单 | V2 中等 | V3 难题 | 角色 |
|------|---------|---------|---------|------|
| DS-PRO | 84% | 76% | **72%** | 攻坚 |
| DS-Think | 100% | 88% | 70% | 架构 |
| GLM | 95% | **91%** | 67% | 中文 |
| QWEN | 90% | 82% | 67% | 通用 |
| Kimi | 74% | 73% | 61% | 写作 |

## 快速开始

```bash
git clone https://github.com/toromesht/SynapseFlow.git
cd SynapseFlow
pip install openai python-docx
cp config/llm-config.template.json config/llm-config.json
# Edit with your API keys
python engine/brain.py "你的问题"
```

## 核心论文

基于 MasRouter(ACL 2025) / Dynamic MoE(ACL 2024) / Router-R1(NeurIPS 2025) / Chain-of-Thought / Self-Consistency / DeBERTa Gate(Nature 2025)
