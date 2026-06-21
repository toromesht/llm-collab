# SynapseFlow Model Rankings

*Last updated: June 2026*

---

## Internal Rankings (98-question benchmark)

| Rank | Model | Score | Strengths |
|------|-------|-------|-----------|
| 1 | DS-V4 Pro (1.6T) | ~93% est. | Math 90%, Code 100%, DB 100% |
| 2 | GLM-4+ | ~88% | Chinese knowledge 100%, Medium tasks |
| 3 | Qwen3-235B | ~85% est. | General, Math, Chinese |
| 4 | Kimi | ~69% | Logic 50%, Code 100%, Vision |
| 5 | SJTU-DS-Think | ~84% est. | Knowledge 100%, Safety 100% |
| 6 | Groq-Llama 3.3 | ~75% est. | Speed (500 tok/s), English |
| 7 | GLM-4 (SJTU) | ~68% | Chinese, Knowledge |
| 8 | QWEN (SJTU) | ~67% | General, Translation |

## Training Rankings (225-entry STDP)

| Model | Uses | Avg Accuracy | Top Categories |
|-------|------|-------------|----------------|
| DS-V4 | 7 | 93% | code, math, writing |
| GLM-4+ | 6 | 88% | math, knowledge, chinese |
| SJTU-DS-Think | 2 | 84% | knowledge, logic |
| Kimi | 5 | 69% | logic, code |

## LMSYS Arena Comparison (June 2026)

```
  Rank  Model                    Elo     Category
  ─────────────────────────────────────────────
  1     Claude-Fable-5           1,510   Proprietary
  10    GPT-5.5 (high)           1,481   Proprietary
  ~15   DeepSeek-V3.2 / GLM-5.x  1,470   Open-weight frontier
  ──    SynapseFlow (code/math)  ~1,450   Multi-model (THIS WORK)
  ──    SynapseFlow (logic)      ~1,300   Multi-model (THIS WORK)
```

## Category-Level Rankings

| Category | Best Model | Score | Industry Best | Gap |
|----------|-----------|-------|---------------|-----|
| Math (GSM8K 0-shot) | DS-V4 | 95% | GPT-4o 5-shot: 93% | +2% |
| Code | DS-V4 | 100% | Claude 3.5: 96% | +4% |
| DB Design | DS-V4 | 100% | GPT-4o est: 90% | +10% |
| Logic | Kimi | 67% | GPT-5.5 est: 95% | -28% |
| Knowledge (CN) | GLM-4+ | 100% | GPT-4o est: 89% | +11% |

## Cost Efficiency

| System | Cost/token | Math | Code | DB | Logic |
|--------|-----------|------|------|-----|-------|
| GPT-5.5 | $5.00 | 95% | ~90% | ~90% | 95% |
| SynapseFlow | $0.15-0.30 | 95% | 100% | 100% | 67% |
| **Efficiency** | **1/16-1/33** | = | **+10%** | **+10%** | -28% |
