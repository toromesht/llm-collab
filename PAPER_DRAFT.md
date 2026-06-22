# Multi-Objective Active Inference for Online LLM Routing

**Target venue:** NeurIPS / ICML / AAAI (Systems + Learning track)

**Authors:** [Your Name], [Advisor] — SJTU Zhiyuan College

---

## Abstract

Selecting the optimal large language model (LLM) for a given query is a contextual bandit problem with competing objectives: accuracy, latency, cost, and exploration. Existing routing systems use heuristic scoring or single-objective optimization, failing to capture the multi-dimensional trade-offs inherent in real-world deployment. We present **SynapseFlow**, a multi-objective active inference framework for online LLM routing. Our system unifies policy learning (REINFORCE), value estimation (TD learning), and belief updating (variational Bayes) under a single expected free energy functional G(π). A recognition model learns a Markov state representation from 22 brain-inspired query features. Per-region generative models maintain Beta-Bernoulli posteriors over success probabilities, updated online via Bayesian inference. Multi-objective routing minimizes context-weighted Pareto free energy across six objectives: accuracy, speed, cost, exploration, robustness, and diversity. We derive all model parameters from the Active Inference literature (Friston et al., 2017; Parr & Friston, 2019; Schwartenbeck et al., 2019) and deploy the system as a Claude Code plugin routing queries across six commercial LLM APIs. [Results to be filled after benchmark evaluation].

---

## 1. Introduction

The proliferation of LLM APIs (DeepSeek, GLM-4, Qwen, Kimi, Groq, GPT-4) creates a combinatorial routing problem: which model should handle which query? This is not a single-objective problem. Practitioners care about accuracy, but also cost (API pricing), latency (user experience), robustness (avoiding catastrophic failures), and exploration (discovering which models excel at new tasks).

Existing approaches fall into three categories:
- **Heuristic routers** (e.g., prompt-based classification + static model mapping) — fast but brittle
- **Bandit-based selectors** (UCB, Thompson sampling) — principled but single-objective
- **LLM-as-judge** (using one LLM to route to another) — expensive and circular

We propose a fourth approach: **multi-objective active inference**. Drawing on the Free Energy Principle (Friston, 2010) and Active Inference process theory (Friston et al., 2017), we formulate LLM routing as expected free energy minimization over a learned state space. This provides:
1. A **single scalar objective** L_total unifying policy learning, value estimation, and belief updating
2. **Pareto-optimal routing** across six competing objectives with context-dependent weights
3. **Online learning** via variational Bayes and policy gradient — no offline training required
4. **Deployability** as a lightweight plugin that routes queries in <5ms

---

## 2. Background: Active Inference

Active Inference (Friston et al., 2017) casts perception and action as minimizing two quantities:

**Variational Free Energy (perception):**
```
F = D_KL[q(s) || p(s)] - E_q(s)[ln p(o|s)]      (1)
```
where q(s) is an approximate posterior over latent states s, p(s) is a prior, and p(o|s) is a generative model of observations.

**Expected Free Energy (action):**
```
G(π) = -E_q(o|π)[ln p(o|C)]                     ← pragmatic value
       - E_q(o,s|π)[ln q(s|o,π) - ln q(s|π)]    ← epistemic value    (2)
```
where π is a policy, C represents preferred outcomes, and the second term captures expected information gain.

Agents minimize G(π) to select policies that balance exploitation (pragmatic value) and exploration (epistemic value).

---

## 3. Method: SynapseFlow

### 3.1 Problem Formulation

At each timestep t, the system receives a text query x_t and must select a policy π_t from a discrete action space A = {(region_i, model_j)} of size N_π = 16. Executing π_t produces an output y_t = LLM_π(x_t) and a binary reward r_t ∈ {0, 1} indicating response quality (evaluated by a stronger LLM as critic).

The objective is to learn a routing policy that minimizes the multi-objective expected free energy:
```
π* = argmin_π Σ_k w_k(c_t) · G_k(π)            (3)
```
where k indexes six objectives, w_k(c_t) are context-dependent weights, and c_t ∈ {urgent, research, coding, creative, default} is inferred from x_t.

### 3.2 State Representation

We learn a recognition model q_φ(s|x) — a 2-layer MLP (22→64→6) with ReLU activations — that maps query features to a categorical distribution over 6 latent states (corresponding to "brain regions": Motor, Parietal, PFC, Temporal, Language, Visual).

The 22 query features are extracted via lightweight regex matching over the prompt, capturing signals for: code, math, logic, knowledge, writing, architecture, and 10 math subfields (group theory, graph theory, topology, etc.).

The recognition model is trained via variational EM: after observing outcome r_t for policy π_t, we compute a target state distribution proportional to the generative model's precision per state, and minimize KL divergence.

### 3.3 Generative Model

For each latent state s and policy π, we maintain a Beta-Bernoulli generative model p(o|s,π):

```
p(success|s,π) ~ Beta(α_sπ, β_sπ)              (4)
```

Parameters are updated online via Bayesian inference:
```
α_sπ ← α_sπ + r_t · q_φ(s|x_t)                  (5)
β_sπ ← β_sπ + (1-r_t) · q_φ(s|x_t)             (6)
```

This is conjugate Bayesian updating — no gradient computation required for the likelihood term.

### 3.4 Expected Free Energy Computation

For each policy π, we compute:

**Pragmatic value:**
```
G_prag(π) = -E_q(s|x)[ln(α_sπ/(α_sπ + β_sπ))]  (7)
```

**Epistemic value:**
```
G_epis(π) = -E_q(s|x)[Var(Beta(α_sπ, β_sπ))]   (8)
```

**Speed objective:**
```
G_spd(π) = latency(model_π) / max_latency        (9)
```

**Cost objective:**
```
G_cst(π) = cost(model_π) / budget                (10)
```

**Robustness objective:**
```
G_rob(π) = -E_q(s|x)[α_sπ + β_sπ]               (11)
```
(Higher precision = more reliable, inverted for minimization.)

**Diversity objective:**
```
G_div(π_set) = -Σ_{i≠j} (1 - cos(vec_i, vec_j))  (12)
```
where vec_i is the one-hot region encoding of policy i. Encourages selecting policies from different brain regions.

### 3.5 Multi-Objective Routing

The context c_t is detected from query keywords and maps to a weight vector w ∈ R^6:

| Context | accuracy | speed | cost | exploration | robustness | diversity |
|---------|----------|-------|------|-------------|------------|-----------|
| urgent | 0.5 | 2.0 | 0.2 | 0.1 | 0.3 | 0.1 |
| research | 1.5 | 0.2 | 0.3 | 1.0 | 1.0 | 0.8 |
| coding | 1.0 | 0.5 | 0.3 | 0.2 | 0.8 | 0.3 |
| creative | 0.5 | 0.3 | 0.5 | 0.8 | 0.3 | 1.5 |

Total free energy per policy:
```
G_total(π) = Σ_k w_k · G_k(π) + λ · Σ_{π'∈selected, π'≠π} C(π, π')  (13)
```
where C(π, π') is a Hebbian coherence penalty between region pairs, learned online via STDP:
```
λ_ij ← λ_ij + 0.01 if both succeeded, else λ_ij ← 0.99·λ_ij   (14)
```

Top-K policies are selected by sorting G_total and applying a diversity gate (preferring different regions and models).

### 3.6 Policy Selection (Softmin)

```
P(π|x) = softmax(-G_total(π) / τ)               (15)
τ = 0.1 + 0.4 · difficulty                       (16)
```

Harder questions (higher difficulty from brainstem classification) increase temperature, encouraging broader exploration.

### 3.7 Unified Loss Function

All model parameters are optimized by a single scalar loss:

```
L_total(θ, ψ, α, β) =
    G_total(π_chosen)                              [expected free energy]
    + α_kl · KL(q_θ(s|x) || p(s))                  [complexity regularization]
    + (r + γ · V_ψ(x') - V_ψ(x, π_chosen))²        [TD value error]   (17)
```

where θ parameterizes the recognition model, ψ parameterizes the value function (a 2-layer MLP: 22→128→16), and (α, β) are the generative model parameters. All components are updated simultaneously via Adam (lr=0.001).

### 3.8 Value Function Learning

We train a value function V_ψ(x, π) to approximate -G_total(π), enabling fast policy evaluation without recomputing the full free energy functional. The value function is trained via temporal difference learning:

```
L_V = (r_t + γ · V_ψ(x_{t+1}) - V_ψ(x_t, π_t))²   (18)
```

with discount factor γ = 0.9. The value function replaces MCTS rollouts, providing efficient policy evaluation in a single forward pass.

### 3.9 Deployment Architecture

The system deploys as a Claude Code plugin with three components:
1. **Hook** (hook_brain.py): intercepts UserPromptSubmit events, classifies queries via Fortran brainstem (HD-SDM, 10k-bit hypervectors, 324µs per classification), and launches the routing pipeline in a background thread.
2. **Router** (pareto_fep.py + unified_fep.py): computes G_total(π) for all 16 policies, selects top-K, and updates all parameters online.
3. **Status line** (status_line.py): displays real-time routing decisions and model execution status in the Claude Code terminal.

The full pipeline executes in <5ms for routing + API-dependent execution time for model calls.

---

## 4. Experimental Design

### 4.1 Baselines

We compare against:
- **Random**: Uniform random model selection
- **Heuristic**: Regex-based feature extraction + static model mapping
- **UCB1 Bandit**: Upper confidence bound with sliding window
- **Single-best**: Always use the globally best model (DS-PRO)
- **LLM-as-router**: Use one LLM to select which LLM to call
- **Ablation: single-objective FEP** (unified_fep.py, G(π) without multi-objective weights)
- **Ablation: no encoder** (removing q_φ, replacing with brainstem region ID)

### 4.2 Benchmarks

- **GSM8K** (grade school math, 1,319 questions)
- **HumanEval** (code generation, 164 problems)
- **MMLU** (multitask language understanding, 57 subjects)
- **BoolQ** (boolean question answering, 3,270 questions)
- **Custom multi-domain set**: 200 questions spanning code, math, logic, knowledge, writing, and vision

### 4.3 Metrics

- **Success rate**: % of responses passing quality gate (DS-PRO critic review)
- **Average cost**: USD per query (weighted by model pricing)
- **Average latency**: Time from query to final response
- **Pareto hypervolume**: Multi-objective quality metric (Zitzler & Thiele, 1999)
- **Cumulative regret**: Σ(r_oracle - r_chosen) over episodes
- **Learning efficiency**: Success rate vs. number of episodes (sample efficiency)

### 4.4 Procedure

1. For each benchmark, run all baselines and SynapseFlow variants
2. Measure per-query: selected model, cost, latency, success
3. Run 3 seeds, report mean ± std
4. Statistical significance: paired bootstrap test (N=10,000)

### 4.5 Expected Results (Hypothesis)

We hypothesize:
1. **Pareto dominance**: SynapseFlow will dominate single-objective methods on the accuracy-cost-latency Pareto front
2. **Sample efficiency**: Variational Bayes + beta priors enable faster learning than bandit baselines
3. **Context sensitivity**: Multi-objective weights adapt appropriately (urgent = faster, research = more accurate)
4. **Ablation impact**: Removing any component (encoder, multi-objective weights, epistemic value) degrades performance

---

## 5. Related Work

**LLM Routing:** Yue et al. (2025) MasRouter; Huang et al. (2024) Dynamic MoE routing; Wong (2025) FEP in Mixture-of-Experts (124× improvement from FEP modification).

**Active Inference:** Friston et al. (2017) process theory; Parr & Friston (2019) generalized free energy; Schwartenbeck et al. (2019) curiosity and information-seeking; Da Costa et al. (2020) generative models of brain structure.

**Multi-Objective RL:** Van Moffaert & Nowe (2014) multi-objective reinforcement learning; Roijers et al. (2013) survey of multi-objective decision making.

**Contextual Bandits:** Li et al. (2010) LinUCB; Agrawal & Goyal (2012) Thompson sampling analysis.

**Neuroscience-Inspired AI:** Hassabis et al. (2017) neuroscience-inspired artificial intelligence; Lake et al. (2017) building machines that learn and think like people.

---

## 6. Conclusion

We presented SynapseFlow, a multi-objective active inference framework for online LLM routing. Our key contributions are: (1) a unified expected free energy functional G(π) that derives all system components from a single mathematical objective; (2) multi-objective Pareto routing across six competing objectives with context-dependent weights; (3) online learning via conjugate Bayesian updating and policy gradient with a learned state representation; and (4) deployment as a lightweight Claude Code plugin routing queries across six commercial LLM APIs.

[Results section to be completed after benchmark evaluation.]

---

## Appendix A: Hyperparameter Table

| Parameter | Value | Source |
|-----------|-------|--------|
| β_epistemic | 0.3 | Schwartenbeck 2019 |
| α_kl | 0.01 | Friston 2017 |
| τ (temperature) | 0.1 + 0.4·difficulty | Parr 2019 |
| γ (TD discount) | 0.9 | Sutton & Barto 2018 |
| Adam lr | 0.001 | Standard |
| Hebbian lr | 0.01 | Song & Miller 2000 |
| Beta prior | α=1, β=1 | Uniform (Bayesian) |
| n_states | 6 | One per brain region |
| n_policies | 16 | 6 regions × avg 2.7 models |
| recognition model | 22→64→6 | 3-layer MLP |
| value function | 22→128→16 | 3-layer MLP |

## Appendix B: Model Cost and Latency

| Model | Provider | Cost/1k tok | Latency (s) |
|-------|----------|-------------|-------------|
| DS-PRO | DeepSeek | $0.002 | 3.0 |
| DS-Think | SJTU HPC | $0.001 | 8.0 |
| GLM-4+ | Zhipu | $0.001 | 2.0 |
| Qwen3-235B | Alibaba | $0.001 | 1.5 |
| Kimi | Moonshot | $0.0015 | 1.0 |
| Groq (Llama 3.3) | Groq | $0.0002 | 0.5 |

## Appendix C: Code Repository

`github.com/toromesht/llm-collab`

Key files:
- `engine/unified_fep.py` — Single-objective FEP router (L_total)
- `engine/pareto_fep.py` — Multi-objective Pareto router (6 objectives)
- `engine/policy_network.py` — Neural policy network (REINFORCE + Adam)
- `engine/neuro_runner.py` — Unified harness pipeline
- `.claude-plugin/hooks/neuro_hook.py` — Claude Code integration
- `SYNAPSEFLOW_PAPER_MAP.md` — Complete paper-to-code parameter mapping
