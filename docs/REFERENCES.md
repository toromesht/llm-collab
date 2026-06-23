# SynapseFlow — Reference Papers & Algorithm Mappings

> 共31篇论文，跨神经科学/计算机科学/数学/ML路由。

---

## 1. Neuroscience → Algorithm

| # | Paper | Journal | DOI | Algorithm in SynapseFlow |
|---|-------|---------|-----|--------------------------|
| 1 | Hafting, Fyhn, Molden, Moser & Moser (2005) "Microstructure of a spatial map in the entorhinal cortex" | *Nature* 436:801-806 | [10.1038/nature03721](https://doi.org/10.1038/nature03721) | `GridCellMap` — Random Fourier Feature encoder |
| 2 | Rao & Ballard (1999) "Predictive coding in the visual cortex" | *Nature Neuroscience* 2:79-87 | [10.1038/4580](https://doi.org/10.1038/4580) | `PredictiveCodingLayer` — Error-driven online predictor |
| 3 | Frey & Morris (1997) "Synaptic tagging and long-term potentiation" | *Nature* 385:533-536 | [10.1038/385533a0](https://doi.org/10.1038/385533a0) | `SynapticTaggingCapture` — Event-driven momentum |
| 4 | Song, Miller & Abbott (2000) "Competitive Hebbian learning through STDP" | *Nature Neuroscience* 3:919-926 | [10.1038/78829](https://doi.org/10.1038/78829) | `HebbianSTDP` — Exponential credit assigner |
| 5 | Hartline & Ratliff (1958) "Inhibitory interaction in the Limulus eye" | *J General Physiology* 42:1241-1255 | [10.1085/jgp.42.6.1241](https://doi.org/10.1085/jgp.42.6.1241) | `RecurrentLateralInhibition` — Softmax router |
| 6 | Kandel (2001) "The molecular biology of memory storage" (Nobel Lecture) | *Science* 294:1030-1038 | [10.1126/science.1067020](https://doi.org/10.1126/science.1067020) | `MemoryConsolidation` — Stable route cache |
| 7 | Bienenstock, Cooper & Munro (1982) "Theory for neuron selectivity" | *J Neuroscience* 2(1):32-48 | [10.1523/JNEUROSCI.02-01-00032.1982](https://doi.org/10.1523/JNEUROSCI.02-01-00032.1982) | BCM sliding threshold (archived) |
| 8 | Friston (2010) "The free-energy principle: a unified brain theory?" | *Nature Reviews Neuroscience* 11:127-138 | [10.1038/nrn2787](https://doi.org/10.1038/nrn2787) | FEP routing (archived) |
| 9 | Abraham & Bear (1996) "Metaplasticity" | *TINS* 19(4):126-130 | [10.1016/S0166-2236(96)80018-X](https://doi.org/10.1016/S0166-2236(96)80018-X) | Metaplasticity parameters |
| 10 | Turrigiano & Nelson (2004) "Homeostatic plasticity in the developing nervous system" | *Nature Reviews Neuroscience* 5:97-107 | [10.1038/nrn1327](https://doi.org/10.1038/nrn1327) | Synaptic scaling |
| 11 | Beggs & Plenz (2003) "Neuronal avalanches in neocortical circuits" | *J Neuroscience* 23(35):11167-11177 | [10.1523/JNEUROSCI.23-35-11167.2003](https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003) | Criticality tuning |

---

## 2. Computer Science → Implementation

| # | Paper | Venue | DOI | Implementation |
|---|-------|-------|-----|----------------|
| 12 | Kanerva (1988) "Sparse Distributed Memory" | MIT Press | [ISBN 0262111332](https://mitpress.mit.edu/9780262111331/) | `brainstem_wrapper.py` — SDM |
| 13 | Kanerva (2009) "Hyperdimensional Computing" | *Cognitive Computation* 1:139-159 | [10.1007/s12559-009-9009-8](https://doi.org/10.1007/s12559-009-9009-8) | `brainstem.f90` — HD encoding |
| 14 | Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines" | *NeurIPS* 2007 | [10.5555/2981562.2981710](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) | `GridCellMap` — RFF |
| 15 | Karger et al. (1997) "Consistent Hashing and Random Trees" | *STOC* 1997 | [10.1145/258533.258660](https://doi.org/10.1145/258533.258660) | `router_cpp/` — Hash ring |
| 16 | Fedus, Zoph, Shazeer (2022) "Switch Transformers" | *JMLR* 23(1) | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | `router_cpp/` — Top-K gating |
| 17 | Jacobson (1988) "Congestion Avoidance and Control" | *SIGCOMM* 1988 | [10.1145/52324.52356](https://doi.org/10.1145/52324.52356) | TCP AIMD for model windows |
| 18 | Spall (1992) "Multivariate Stochastic Approximation" | *IEEE Trans. Automatic Control* 37(3):332-341 | [10.1109/9.119632](https://doi.org/10.1109/9.119632) | SPSA tuning (archived — biased) |
| 19 | Stonebraker, Rowe, Hirohama (1990) "The Implementation of Postgres" | *IEEE TKDE* 2(1) | [10.1109/69.50912](https://doi.org/10.1109/69.50912) | `router_cpp/` — shared_mutex |

---

## 3. Mathematics & Statistics → Routing Logic

| # | Paper | Journal | DOI | Algorithm in SynapseFlow |
|---|-------|---------|-----|--------------------------|
| 20 | Ebbinghaus (1885) "Memory: A Contribution to Experimental Psychology" | (Classic text) | [YorkU](https://psychclassics.yorku.ca/Ebbinghaus/) | `SynapticDecay` — Forgetting curve |
| 21 | Wixted (2004) "The Psychology and Neuroscience of Forgetting" | *Psychological Review* 111:864 | [10.1037/0033-295X.111.4.864](https://doi.org/10.1037/0033-295X.111.4.864) | Dual-trace forgetting |
| 22 | Dasgupta & Gupta (2003) "Elementary Proof of JL Lemma" | *Random Structures & Algorithms* 22(1):60-65 | [10.1002/rsa.10073](https://doi.org/10.1002/rsa.10073) | `math_router.py` — JL projection |
| 23 | Gionis, Indyk, Motwani (1999) "Similarity Search in High Dimensions via Hashing" | *VLDB* 1999 | [10.5555/645925.671516](https://dl.acm.org/doi/10.5555/645925.671516) | `math_router.py` — LSH |
| 24 | Lorden (1971) "Procedures for Reacting to a Change in Distribution" | *Annals Math Stat* 42(6):1897-1908 | [10.1214/aoms/1177693055](https://doi.org/10.1214/aoms/1177693055) | `math_router.py` — CUSUM |
| 25 | Wald (1945) "Sequential Tests of Statistical Hypotheses" | *Annals Math Stat* 16(2):117-186 | [10.1214/aoms/1177731118](https://doi.org/10.1214/aoms/1177731118) | `math_router.py` — SPRT |
| 26 | Robbins & Monro (1951) "A Stochastic Approximation Method" | *Annals Math Stat* 22(3):400-407 | [10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586) | Online logistic difficulty |
| 27 | Thompson (1933) "On the Likelihood that One Unknown Probability Exceeds Another" | *Biometrika* 25(3-4):285-294 | [10.1093/biomet/25.3-4.285](https://doi.org/10.1093/biomet/25.3-4.285) | Thompson Sampling |
| 28 | Sutton & Barto (1998) "Reinforcement Learning: An Introduction" Ch.7 — TD(λ) | MIT Press | [ISBN 0262193981](http://incompleteideas.net/book/the-book.html) | `math_router.py` — Eligibility trace |
| 29 | Garivier & Moulines (2008) "On UCB Policies for Switching Bandit Problems" | *ALT* 2008 | [10.1007/978-3-540-87987-9_16](https://doi.org/10.1007/978-3-540-87987-9_16) | Discounted UCB |
| 30 | Kulhavy & Zarrop (1993) "On a General Concept of Forgetting" | *Automatica* 29(4):1015-1019 | [10.1016/0005-1098(93)90107-5](https://doi.org/10.1016/0005-1098(93)90107-5) | Variable forgetting factor |

---

## 4. LLM Routing / MoE (Recent)

| # | Paper | Venue | DOI |
|---|-------|-------|-----|
| 31 | Wong (2025) "Affinity Is Not Enough: FEP in MoE" | arXiv:2605.00604 | [arxiv.org/abs/2605.00604](https://arxiv.org/abs/2605.00604) |
| 32 | Ma et al. (2025) "ODAR: Active Inference Adaptive Routing" | arXiv:2602.23681 | [arxiv.org/abs/2602.23681](https://arxiv.org/abs/2602.23681) |
| 33 | Lai & Ye (2026) "When Routing Collapses" | arXiv:2602.03478 | [arxiv.org/abs/2602.03478](https://arxiv.org/abs/2602.03478) |
| 34 | Huang et al. (2024) "Harder Tasks Need More Experts" | ACL 2024 | [arXiv:2403.07652](https://arxiv.org/abs/2403.07652) |
| 35 | Wei et al. (2022) "Chain-of-Thought Prompting" | NeurIPS 2022 | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| 36 | Duecker et al. (2024) "Oscillations in ANN convert competing inputs" | *PLOS Comp Bio* 20(9) | [10.1371/journal.pcbi.1012429](https://doi.org/10.1371/journal.pcbi.1012429) |

---

## Algorithm → Code Map

```
Paper                          | Code File               | Active? | Reason
───────────────────────────────┼─────────────────────────┼─────────┼──────────────────────
Moser & Moser (2005)           | neural_mechanisms.py:13 | YES     | RFF encoder (GridCellMap)
Rao & Ballard (1999)           | neural_mechanisms.py:91 | YES     | Error-driven predictor
Frey & Morris (1997)           | neural_mechanisms.py:221| YES     | Event momentum
Song, Miller & Abbott (2000)   | neural_mechanisms.py:328| YES     | EWMA credit assigner
Hartline & Ratliff (1958)      | neural_mechanisms.py:404| YES     | Softmax router
Kandel (2001)                  | neural_mechanisms.py:484| YES     | Stable route cache
Ebbinghaus (1885)              | neural_mechanisms.py:557| YES     | Forgetting factor
Rahimi & Recht (2007)          | neural_mechanisms.py:13 | YES     | RFF (same as GridCell)
Dasgupta & Gupta (2003)        | math_router.py:41       | OPT     | JL projection
Gionis et al. (1999)           | math_router.py:97       | OPT     | LSH
Lorden (1971)                  | math_router.py:195      | OPT     | CUSUM
Wald (1945)                    | math_router.py:266      | OPT     | SPRT
Thompson (1933)                | math_router.py:421      | OPT     | Thompson sampling
Robbins & Monro (1951)         | math_router.py:163      | OPT     | Online logistic
Garivier & Moulines (2008)     | path_learner.py:14      | OPT     | Discounted UCB
Kulhavy & Zarrop (1993)        | path_learner.py:21      | OPT     | Adaptive forgetting
Kanerva (1988, 2009)           | brainstem.f90           | BUILD   | HD/SDM (Fortran)
Karger et al. (1997)           | router_cpp/src/         | BUILD   | Consistent hashing
Fedus et al. (2022)            | router_cpp/src/         | BUILD   | Top-K gating
Friston (2010)                 | brain.py (archived)     | NO      | Scale mismatch (n=36)
Bienenstock et al. (1982)      | brain.py (archived)     | NO      | Scale mismatch
Spall (1992)                   | brain.py (archived)     | NO      | Biased variant
Duecker et al. (2024)          | brain.py (archived)     | NO      | Wrong timescale

Legend: YES=active routing | OPT=optional backend | BUILD=compiled native | NO=archived
```

---

## Quick Access

所有 DOI 链接均可直接访问论文全文（部分需机构订阅）。
arXiv 论文全部免费开放获取。
MIT Press 书籍可通过 [mitpress.mit.edu](https://mitpress.mit.edu) 购买或图书馆访问。
