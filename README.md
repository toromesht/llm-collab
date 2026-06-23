# SynapseFlow — LLM Model Router

> **Semantic classification → single-model routing → cost savings.**
> 
> Routes user questions to the best LLM. Easy questions → cheap models. Hard questions → smart models.
> C++ router (<4μs). Fortran encoder. Python orchestration. R analysis.

`v2.2 — Semantic classifier + C++/Fortran compiled`

> ⚠️ **EARLY PROTOTYPE** — research project, not production-ready. APIs unstable.

---

## What This Does

```
User: "12个球称重找次品"
  → classify: logic (confidence: 0.85)
  → route:   ds-pro  (hard logic, need best model)
  → cost:    ~$0.004 (est.)

User: "法国首都是哪里"
  → classify: knowledge (confidence: 0.92)
  → route:   groq     (trivial, use cheapest model)
  → cost:    ~$0.0001 (est.)
```

**Saves money** by not calling expensive models for simple questions.
**Does NOT improve accuracy** over calling the best model directly (proven in benchmarks).

---

## Quick Start (Python only, 2 minutes)

### 1. Install

```bash
git clone https://github.com/toromesht/llm-collab.git
cd llm-collab
pip install openai numpy sentence-transformers
```

### 2. Configure API Keys

Create `~/.claude/tools/llm-config.json`:

```json
{
  "deepseek_pro": {
    "api_key": "sk-your-key-here",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat"
  },
  "sjtu_zhiyuan": {
    "api_key": "your-sjtu-hpc-key",
    "base_url": "https://your-hpc-endpoint/v1"
  },
  "groq": {
    "api_key": "gsk_your-groq-key",
    "base_url": "https://api.groq.com/openai/v1",
    "model": "llama-3.3-70b-versatile"
  }
}
```

Or use the setup wizard:
```bash
python setup.py
```

### 3. Run

```bash
# Single question
python engine/brain.py "用Python写一个二分查找"

# 7-mechanism neural runner (research mode)
python engine/neuro_runner.py
```

---

## C++/Fortran Compilation (Optional, for speed)

The Python fallback works fine. For sub-μs routing, compile the native modules.

### Prerequisites

| OS | Requirements |
|----|--------------|
| **Windows** | [MSYS2](https://www.msys2.org/) → `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gcc-fortran mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-openmp` |
| **Linux** | `sudo apt install build-essential gfortran cmake python3-dev` |
| **macOS** | `brew install gcc cmake libomp` |

All platforms: `pip install pybind11 numpy`

### Build

```bash
cd collab-cloud
rm -rf build && mkdir build

cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Output:
```
build/lib/
├── synapse_router.cp311-win_amd64.pyd   # C++ router (Python importable)
├── libsynapse_encode.a                  # Fortran encoder (static lib)
├── libgcc_s_seh-1.dll                   # Runtime (auto-copied)
├── libstdc++-6.dll
└── libwinpthread-1.dll
```

### Verify

```bash
python -c "
import sys; sys.path.insert(0, 'build/lib')
import os
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory('build/lib')
import synapse_router
e = synapse_router.RouterEngine()
print('Native router:', e.stats())
"
# → {'lookups': 0, 'hit_rate': 0.0, 'avg_latency_ns': 0.0}
```

---

## Configuration Reference

### API Keys (`~/.claude/tools/llm-config.json`)

```json
{
  "deepseek_pro": {"api_key": "...", "base_url": "...", "model": "deepseek-chat"},
  "sjtu_zhiyuan": {"api_key": "...", "base_url": "...", "model": "deepseek-reasoner"},
  "groq":          {"api_key": "...", "base_url": "...", "model": "llama-3.3-70b"},
  "kimi":          {"api_key": "...", "base_url": "...", "model": "moonshot-v1"},
  "zhipu":         {"api_key": "...", "base_url": "...", "model": "glm-4"},
  "qwen3":         {"api_key": "...", "base_url": "...", "model": "qwen-plus"}
}
```

Models are auto-detected from available keys. At minimum you need one key.

### Router Configuration

Category→model rules are in `engine/brain.py`:
```python
CATEGORY_MODEL_RULES = {
    "math":         {"primary": "ds-think", "fallback": "ds-pro"},
    "code":         {"primary": "ds-pro",   "fallback": "ds-think"},
    "logic":        {"primary": "ds-pro",   "fallback": "ds-think"},
    "architecture": {"primary": "glm",      "fallback": "ds-pro"},
    "writing":      {"primary": "glm",      "fallback": "kimi"},
    "knowledge":    {"primary": "groq",     "fallback": "glm"},
}
```

---

## Architecture

```
Question → embed (MiniLM) → cosine to prototypes → category
         → difficulty estimate → model routing → API call
         
         ┌─ Easy (diff<0.35): cheap model (Groq/GLM)
K=1 ─────┼─ Medium: primary model for category
(always) └─ Hard (diff>0.75): ds-pro (strongest)
```

---

## Project Structure

```
collab-cloud/
├── engine/
│   ├── brain.py              ★ Main entry: classify → route → execute
│   ├── neural_mechanisms.py  ★ 7 paper-grounded neural mechanisms
│   ├── neuro_runner.py       ★ 7-mechanism harness + learning loop
│   ├── brainstem_wrapper.py  ★ Python brainstem (NumPy fallback)
│   ├── native_bridge.py      ★ C++/Fortran → Python interface
│   ├── *.f90                 ★ Fortran HD/SDM (OpenMP)
│   └── router_cpp/           ★ C++ routing engine (pybind11)
├── eval/                     ★ Benchmarks + evaluation data
├── config/                   ★ Default configs
├── BUILD.md                  ★ Compilation tutorial
└── README.md                 ★ This file
```

---

## Current Limitations

| Area | Status |
|------|--------|
| Classification | 10/10 on test set (semantic + fallback) |
| C++ Router | Compiled ✅, <4μs latency |
| Fortran Encoder | Compiled ✅ (static lib, needs C wrapper for Python) |
| Multi-model collab | **No proven benefit** — 3-5x cost for same accuracy |
| Benchmark data | 440 questions, 7 benchmarks, real API calls |
| Cold start | Router needs 20-50 episodes to learn per-model strengths |

---

## References

**36 papers** with DOIs, algorithm mappings, and active/archived status.
→ [`docs/REFERENCES.md`](docs/REFERENCES.md)

MIT License · [toromesht](https://github.com/toromesht)
