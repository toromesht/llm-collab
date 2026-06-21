# SynapseFlow — Multi-Model Neurosynaptic Orchestration

> An 8-model, all-free, STDP-powered intelligent routing agent.

[![Models](https://img.shields.io/badge/models-8-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

**We need testers!** See [How to Test](#-how-to-test) below.

---

## What it does

SynapseFlow automatically routes your question to the best free AI model — like a brain synapse routing signals.

```
You ask → Feature extraction → Difficulty scoring → Best model(s) selected → Answer

  8 free models: DS-V4(1.6T) | Qwen3-235B | GLM-4+ | Groq-Llama3.3 | Kimi | SJTUx3
```

## Benchmark Results

| Category | Score | vs Industry |
|----------|-------|-------------|
| Math (GSM8K 0-shot) | 95% | GPT-4o 5-shot: 93% |
| Code | 100% | Claude 3.5: 96% |
| DB Design | 100% | GPT-4o est: 90% |
| Logic | 67% | GPT-5.5 est: 95% |
| **Cost** | **$0.15-0.30** | GPT-5.5: $5.00 (1/16) |

## Quick Start

```bash
git clone https://github.com/toromesht/llm-collab.git
cd llm-collab
pip install openai numpy

# Set up your free API keys (all platforms are FREE)
python setup.py

# Ask a question
python engine/agent.py "Prove Lagrange's theorem in group theory"
```

## 🤝 How to Test

**We need help testing!** If you have 10 minutes:

1. **Clone & setup** (see above)
2. **Get free API keys** from these platforms:
   - [DeepSeek](https://platform.deepseek.com) — free
   - [Alibaba Bailian](https://dashscope.console.aliyun.com) — 1M free tokens
   - [Zhipu BigModel](https://bigmodel.cn) — 5-day free trial
   - [Groq](https://console.groq.com) — free tier (VPN needed)
3. **Run the evaluator**: `python eval/benchmark_v3.py`
4. **Try your own questions**: `python engine/agent.py "your question"`
5. **Report results**: Open an Issue with your findings

**What to test:**
- Does the router pick the right model for your question?
- Does it handle math? Code? Chinese? Logic?
- Any bugs, crashes, or weird behavior?
- How does it compare to ChatGPT/Claude on your test questions?

**Every tester helps!** Even just running `python eval/benchmark_v3.py` and posting the output is valuable.

## Architecture

```
engine/agent.py          Main agent (auto routing)
engine/brain.py           STDP + BCM + Lateral Inhib + Pruning engine
engine/api_server.py      OpenAI-compatible API (for LMSYS submission)
engine/train_router.py    STDP classifier training
cpp/router.h              High-speed C++ routing (100x Python)
R/model_analysis.R        Statistical analysis & visualization
eval/                     Evaluation suite (3 rounds, 98 questions)
```

## Paper-Driven Design

Built on: MasRouter(ACL 2025) · Dynamic MoE(ACL 2024) · Router-R1(NeurIPS 2025) · RACER(2025) · STDP(NatNeuro 2000) · BCM(JNeuro 1982) · Nature SciRep 2025

## License

MIT — use it, modify it, build on it.
