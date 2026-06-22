# SynapseFlow — Multi-Model Neurosynaptic Orchestration

> **不是 MoE，不是 Agent 框架。是神经突触网络。**

[![Models](https://img.shields.io/badge/models-8-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Mechanisms](https://img.shields.io/badge/synaptic_mechanisms-11-purple)]()

---

## What makes this DIFFERENT from MoE or Agent frameworks?

| | Standard MoE | Agent Framework | **SynapseFlow** |
|---|---|---|---|
| Routing | Static Top-K softmax | LLM decides who to call | **HD编码 + SDM记忆** (Kanerva 2009) |
| Learning | Joint training (gradient) | Prompt engineering | **11种突触可塑性 在线学习** |
| Model pool | Homogeneous experts | Heterogeneous tools | **6脑区 异构模型池 专用分工** |
| Plasticity | No | No | **STDP + BCM + LTP/LTD + 修剪** |
| Forgetting | No | No | **表现驱动LTD + 时间衰减 + 自动修剪** |
| Memory | No | No | **PRP标记捕获 (Frey & Morris 1997)** |
| Modulation | No | No | **DA/5-HT神经调质 + 星形Ca²⁺** |
| Self-tuning | No | No | **SPSA + 临界态(σ≈1) 自适应** |
| Brainstem | N/A | N/A | **Fortran 64-bit 并行底层路由** |

**SynapseFlow = 神经科学 × 路由算法 × 在线学习**

每次运行，突触通路自动生长、强化、关联、遗忘。不是“调用子代理”，是“神经冲动在突触网络中传播”。

---

## Architecture

```
问题 → [M5 快捷键 O(1)] → [Fortran脑干 HD+SDM] → [6脑区并行]
    → [M8 资格迹] → [DS-PRO皮层终审] → [M9 DA/5-HT调制]
    → [M3 L-LTP硬化] → [M7 簇状共强] → [M6 PRP捕获]
    → [M10 星形Ca²⁺] → [M11 临界态自调] → [M4 LTD衰减]
    → 答案
```

### 6 大脑区 + 模型分配

| 脑区 | 生物对应 | 主模型 | 擅长 |
|------|----------|--------|------|
| 运动皮层 | M1 | DS-PRO, DS-Think | 代码/SQL/算法 |
| 顶叶 | IPS | DS-PRO, Qwen3 | 数学/数值 |
| 前额叶 | dlPFC | DS-PRO, DS-Think | 逻辑/推理/规划 |
| 颞叶 | ATL | GLM-4+, Qwen | 知识/概念 |
| 语言区 | Broca/Wernicke | GLM-4+, Kimi | 写作/中文/翻译 |
| 视觉皮层 | V1-V5 | Kimi, Qwen | 图像/多模态 |

### 11 种突触可塑性机制

| # | 机制 | 论文 |
|---|------|------|
| M1 | STDP 时序可塑性 | Song, Miller & Abbott 2000 |
| M2 | BCM 滑动阈值 | Bienenstock et al. 1982 |
| M3 | L-LTP 通路硬化 | Kandel, Science 2001 |
| M4 | 表现驱动LTD+时间衰减 | Bienenstock 1982 + Dudek 1992 |
| M5 | 元可塑性快捷键 | Abraham & Bear 1996 |
| M6 | 突触标记与捕获 | Frey & Morris, Nature 1997 |
| M7 | 簇状可塑性 | Fu et al., Nature 2012 |
| M8 | 资格迹 | Gerstner et al. 2018 |
| M9 | 神经调质 DA/5-HT | Fuxe et al. 2007 |
| M10 | 星形胶质细胞 Ca²⁺ | Perea & Araque, Science 2007 |
| M11 | 临界态雪崩 | Beggs & Plenz, J Neurosci 2003 |

---

## Quick Start

```bash
git clone https://github.com/toromesht/llm-collab.git
cd llm-collab
pip install openai numpy

# Set up API keys
python setup.py

# Neuro mode (recommended)
python engine/neuro_agent.py

# Or: classic agent mode
python engine/agent.py "Prove Lagrange's theorem"
```

### Fortran Brainstem (64-bit)

```bash
# Requires gfortran 64-bit (MinGW-w64 or Linux)
gfortran -O3 -march=native -flto -funroll-loops -ffast-math -fopenmp -m64 \
  engine/brainstem.f90 engine/brainstem_cli.f90 -o engine/brainstem_cli.exe

# Auto-detected by brainstem_wrapper.py
# Falls back to PythonBrainstem if Fortran binary not found
```

---

## Benchmark Results

| Category | Score | vs Industry |
|----------|-------|-------------|
| Math (GSM8K 0-shot) | 95% | GPT-4o 5-shot: 93% |
| Code | 100% | Claude 3.5: 96% |
| DB Design | 100% | GPT-4o est: 90% |
| Logic | 67% | All models struggle |
| **Cost** | **$0.15-0.30** | GPT-5.5: $5.00 (1/16) |

---

## Project Structure

```
engine/
  brainstem.f90          Fortran 64-bit 脑干 (Kanerva SDM/HD)
  brainstem_cli.f90      Fortran CLI (stdin/stdout)
  brainstem_wrapper.py   Python封装 (Fortran优先 → Python回退)
  brain.py               v14 核心引擎 (STDP/BCM/FEP/SPSA)
  neuro_agent.py         神经编排器 (5层全流程)
  regions.py             6脑区配置 + 区域执行器
  synapse_network.py     11突触机制 通路网络
  agent.py               经典路由代理
  contest_mode.py        竞赛自洽模式
config/
  brain_regions.json     脑区模型池配置
  synapse_config.json    突触参数 (9机制可调)
  llm-config.template.json LLM API模板
docs/
  ALGORITHM_REPORT.html  完整算法研究报告 (24篇论文)
partitions/              分区训练数据 (28条精选)
eval/                    评估套件 (3轮, 98题)
cpp/                     C++高速路由 (μs级)
```

---

## Key Papers (24 references)

See `docs/ALGORITHM_REPORT.html` for complete algorithm documentation with formulas and paper citations.

Core: Kanerva 1988/2009 (HD/SDM) | Song & Miller 2000 (STDP) | Bienenstock 1982 (BCM) | Kandel 2001 (L-LTP) | Frey & Morris 1997 (Tagging) | Fu et al. 2012 (Clustered) | Gerstner et al. 2018 (Eligibility) | Perea & Araque 2007 (Astrocyte) | Beggs & Plenz 2003 (Criticality)

---

## License

MIT — Built by [toromesht](https://github.com/toromesht)
