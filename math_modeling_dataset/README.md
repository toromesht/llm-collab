# Mathematical Modeling Training Dataset

> 专门用于训练数学建模大模型的结构化数据集

## 目标
训练一个能理解复杂场景、进行数学抽象、编写求解代码并解释结果的 LLM。

## 数据结构
```
math_modeling_dataset/
├── README.md
├── data/
│   ├── optimization/      # 优化类 (LP/IP/NLP/DP)
│   ├── statistics/        # 统计预测 (回归/时序/贝叶斯)
│   ├── differential_equations/  # 微分方程
│   ├── game_theory/       # 博弈论
│   └── simulation/        # 蒙特卡洛/系统动力学
├── scripts/
│   ├── generate.py        # 合成数据生成
│   └── validate.py        # 代码可运行性验证
├── examples/              # 样本展示
└── metadata.json          # 统计元数据
```

## 数据格式 (JSONL)
每条记录包含: problem_statement, modeling_process (variables/objective/constraints/type), solution_code, solution_analysis, keywords

## 数据来源
- MCM/ICM 历年赛题和优秀论文
- CUMCM 国赛历年赛题
- 经典教材例题 (姜启源/运筹学)
- 合成数据 (规则生成+LLM润色)

## 使用协议
CC BY-SA 4.0
