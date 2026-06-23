# SynapseFlow — Polyglot Neurosynaptic LLM Orchestration

> **4 languages. 1 principle. Zero Frankenstein.**
>
> Fortran parallel brainstem → C++ high-speed routing → Python orchestration → R statistical monitoring.
> Computer network algorithms (TCP/BGP/OSPF/BFD/CCN) for LLM model selection.
> Rao & Ballard predictive coding + Friston free energy for representation learning.

`v2.1 — Polyglot + Network Papers + Predictive Coding`

> ⚠️ **EARLY PROTOTYPE / EMBRYONIC STAGE**
> This is an experimental research project in active development. Not production-ready.
> APIs are unstable. Benchmarks are incomplete. C++/Fortran modules need compilation.
> Paper reproductions are aspirational — algorithm fidelity is being iteratively improved.
> Use for research exploration only. Feedback and contributions welcome.

---

## Architecture: Why 4 Languages?

```
PATH                 LANGUAGE    SPEED       ROLE
─────────────────────────────────────────────────────
Brainstem HD/SDM     Fortran     <100μs     OpenMP parallel 10k-bit encoding
Fast path selection  C++         <1μs       Lock-free routing cache
Congestion control   C++         <1μs       TCP AIMD per-model windows
Orchestration        Python      ~10ms      Predictive coding + FEP + API
Statistical monitor  R           ~1s        Changepoint detection + SPC
─────────────────────────────────────────────────────
```

**Fortran** parallelizes HD encoding across CPU cores with OpenMP.  
**C++** handles sub-microsecond path selection from hardened cache.  
**Python** runs predictive coding, FEP inference, and model API calls.  
**R** performs statistical changepoint detection and survival analysis offline.

---

## One Principle

```
F = -ln P(success | path) + KL[belief || prior]

All 11 synaptic mechanisms + 7 network algorithms
→ emerge from minimizing this single objective.
```

---

## Algorithm Stack

### Neuroscience → Representation Learning
| Module | Algorithm | Paper |
|--------|-----------|-------|
| `predictive_coding.py` | Hierarchical generative model | Rao & Ballard, *Nature Neuroscience* (1999) |
| `fep_unified.py` | Free Energy Principle | Friston, *Nature Reviews Neuroscience* (2010) |
| `brainstem.f90` | HD Computing + SDM | Kanerva, *MIT Press* (1988), *Cognitive Computation* (2009) |

### Computer Networks → Model Routing
| Module | Algorithm | Paper |
|--------|-----------|-------|
| `network_routing.py` | TCP Congestion Control (AIMD) | Jacobson, *SIGCOMM* (1988); Chiu & Jain (1989) |
| `network_routing.py` | BGP Path Vector Routing | Rekhter & Li, *RFC 1771* (1995) |
| `network_routing.py` | OSPF Link-State (Dijkstra) | Moy, *RFC 2328* (1998) |
| `network_routing.py` | BFD Failure Detection | Katz & Ward, *RFC 5880* (2010) |
| `network_routing.py` | CCN Content-Centric | Jacobson et al., *CoNEXT* (2009) |
| `network_routing.py` | SDN/OpenFlow | McKeown et al., *SIGCOMM* (2008) |

### Mathematics → Optimal Decisions
| Module | Algorithm | Paper |
|--------|-----------|-------|
| `math_router.py` | Johnson-Lindenstrauss Projection | Dasgupta & Gupta (2003) |
| `math_router.py` | LSH Prototype Matching | Gionis, Indyk, Motwani, *VLDB* (1999) |
| `math_router.py` | CUSUM Changepoint Detection | Lorden, *Annals Math Stat* (1971) |
| `math_router.py` | SPRT Evidence Accumulation | Wald, *Annals Math Stat* (1945) |
| `math_router.py` | Robbins-Monro SGD | *Annals Math Stat* (1951) |
| `math_router.py` | TD(λ) Eligibility Trace | Sutton & Barto, *MIT Press* (1998) |
| `math_router.py` | Thompson Sampling | Thompson, *Biometrika* (1933) |
| `path_learner.py` | Variable Forgetting Factor | Kulhavy & Zarrop, *Automatica* (1993) |

---

## Quick Start

```bash
git clone https://github.com/toromesht/llm-collab.git
cd llm-collab
pip install -r requirements.txt
python setup.py

# Full neuro pipeline
python engine/neuro_agent.py

# Or: classic mode
python engine/agent.py "Prove Lagrange's theorem"
```

### Fortran Brainstem (64-bit, OpenMP parallel)

```bash
gfortran -O3 -march=native -flto -fopenmp -m64 \
  engine/brainstem.f90 engine/brainstem_cli.f90 -o engine/brainstem_cli.exe
```

---

## Benchmark

| Category | Score | vs Industry | Note |
|----------|-------|-------------|------|
| Math (GSM8K 0-shot) | 77% | DS-Think: 97% | Router still learning; cold-start penalty |
| Code | 87% | DS-Think: 100% | — |
| DB Design | 100% | — | Small sample |
| Cost | $0.002/q | vs DS-Pro: $0.020 | 1/10 cost on routable queries |
| **Caveat** | **440 questions, real API calls, warm-start pending** | | |

---

## References

**20 papers** spanning neuroscience, computer networks, mathematical statistics, and control theory.

Full report: [`docs/ALGORITHM_REPORT.html`](docs/ALGORITHM_REPORT.html)

MIT License · [toromesht](https://github.com/toromesht)
