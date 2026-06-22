# SynapseFlow → Active Inference: Complete Parameter Mapping

## Anchor Paper
**Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017).**
*Active Inference: A Process Theory. Neural Computation, 29(1), 1-49.*

Supporting papers:
- **Parr, T. & Friston, K. (2019).** Generalised free energy and active inference. *Biological Cybernetics.*
- **Schwartenbeck, P. et al. (2019).** Computational mechanisms of curiosity and information-seeking. *Neuron.*
- **Da Costa, L. et al. (2020).** The anatomy of inference: Generative models of brain structure. *PLOS Computational Biology.*

---

## Complete Parameter Mapping

### 1. Variational Free Energy F (Eq.1 in Friston 2017)

```
F = E_q(s)[ln q(s) - ln p(o,s)]
  = D_KL[q(s)||p(s)] - E_q(s)[ln p(o|s)]
```

| Symbol | Paper | SynapseFlow Parameter | Value | Rationale |
|--------|-------|-----------------------|-------|-----------|
| q(s) | variational posterior | `encoder.forward(x)` → state_probs | learned | Neural network infers latent state from question features |
| p(s) | prior over states | uniform prior `[1/6]*6` | 0.167 each | Maximum entropy prior — no preferred brain region |
| p(o|s) | likelihood (generative model) | `genmodel.alpha/(alpha+beta)` | Beta posterior | Bernoulli success probability per (state, policy) |
| D_KL | complexity cost | `KL(q||p)` term in L_total | weight α=0.01 | Prevents overfitting of encoder |

### 2. Expected Free Energy G(π) (Eq.4 in Friston 2017)

```
G(π) = E_q(o,s|π)[ln q(s|π) - ln p(o,s|π)]
     = -E_q(o|π)[D_KL[q(s|o,π)||q(s|π)]]   ← epistemic (information gain)
       - E_q(o|π)[ln p(o|C)]                ← pragmatic (goal-seeking)
```

| Symbol | Paper | SynapseFlow Parameter | Value | Rationale |
|--------|-------|-----------------------|-------|-----------|
| π | policy | `(region_id, model)` pair | 16 possible | Each (region, model) is a policy in active inference |
| G(π) | expected free energy | `G[i]` per policy | computed online | Select policy that minimizes G |
| epistemic value | information gain | `Beta variance = αβ/((α+β)²(α+β+1))` | weight β_ep=0.3 | Higher for unexplored (region, model) pairs |
| pragmatic value | preference satisfaction | `-ln p(success\|region, policy)` | -ln(α/(α+β)) | Prefers policies with high success probability |
| C | preferred outcomes | `success=1, failure=0` | binary | Categorical preference for correct answers |
| τ (temperature) | precision of policy selection | `temperature` | 0.5 | Lower = more deterministic; Anneals with difficulty |

### 3. Multi-Objective Extension (Pareto FEP)

```
G_multi(π) = Σ_k w_k(c) · G_k(π) + Σ_{i≠j} λ_ij · C(π_i, π_j)
```

Where k ∈ {accuracy, speed, cost, exploration, robustness, diversity}

| Objective k | G_k(π) derivation | Weight w_k | Paper reference |
|-------------|-------------------|------------|-----------------|
| accuracy | -ln p(correct\|π) | 1.0 | Friston Eq.4 pragmatic term |
| speed | latency(π)/max_latency | 0.5 | Parr 2019 — precision-weighting of policies |
| cost | cost(π)/budget | 0.5 | Resource rationality (Lieder & Griffiths 2020) |
| exploration | -H[Beta(α,β)] | 0.5 | Schwartenbeck 2019 — epistemic value |
| robustness | -(α+β) = -precision | 0.5 | Da Costa 2020 — precision-weighted inference |
| diversity | -cosine_distance | 0.5 | Ensemble diversity (Dietterich 2000) |

Context-dependent weights w_k(c):
| Context | accuracy | speed | cost | exploration | robustness | diversity |
|---------|----------|-------|------|-------------|------------|-----------|
| urgent | 0.5 | 2.0 | 0.2 | 0.1 | 0.3 | 0.1 |
| research | 1.5 | 0.2 | 0.3 | 1.0 | 1.0 | 0.8 |
| coding | 1.0 | 0.5 | 0.3 | 0.2 | 0.8 | 0.3 |
| creative | 0.5 | 0.3 | 0.5 | 0.8 | 0.3 | 1.5 |

### 4. Bayesian Belief Updating (Eq.5 in Friston 2017)

```
p(o|s,π)_{t+1} = Beta(α + r·q(s), β + (1-r)·q(s))
```

| Symbol | Paper | SynapseFlow Code | Update rule |
|--------|-------|-----------------|-------------|
| α (posterior) | concentration parameter | `genmodel.alpha[s, π] += r * q[s]` | Success → increase α |
| β (posterior) | concentration parameter | `genmodel.beta[s, π] += (1-r) * q[s]` | Failure → increase β |
| q(s) | state responsibility | `encoder.forward(x)` | Soft assignment of question to states |

### 5. Policy Selection (Softmax over G)

```
P(π) = softmax(-G(π) / τ)
     = exp(-G(π)/τ) / Σ_j exp(-G(j)/τ)
```

| Symbol | Paper | SynapseFlow | Value |
|--------|-------|-------------|-------|
| τ | precision / inverse temperature | `temperature` | 0.1 + 0.4 * difficulty |
| γ (discount) | temporal discount for value learning | `gamma_td` | 0.9 |

### 6. Value Function Learning (TD Error)

```
L_V = (r + γ·V(x') - V(x,π))²
```

| Symbol | Paper | SynapseFlow Code |
|--------|-------|-----------------|
| V_ψ(x,π) | value function (2-layer MLP) | `value_fn.forward(x)[policy]` |
| target | r + γ·V(x') [bootstrapped] | `reward + gamma_td * V_current` |
| update | gradient descent on TD error | `value_fn.update(x, policy, target)` |

### 7. Encoder Learning (Variational EM)

```
L_enc = KL(q_φ(s|x) || p_target(s|x,π,o))
```

| Symbol | Paper | SynapseFlow Code |
|--------|-------|-----------------|
| q_φ(s|x) | recognition model (2-layer MLP) | `encoder.forward(x)` |
| p_target | states with high generative precision | `precision_per_state / sum` |
| update | cross-entropy gradient | `encoder.update(x, target_probs)` |

---

## Complete Loss Function (Single Scalar)

```
L_total(θ_enc, ψ_val, α_gen, β_gen) =
    G(π_chosen; α, β)                          [Expected Free Energy]
    + α_kl · KL(q_θ(s|x) || p(s))               [Complexity Regularization]
    + (r + γ·V_ψ(x') - V_ψ(x, π_chosen))²      [TD Value Error]

One optimizer (Adam). One gradient step. All parameters updated simultaneously.
```

---

## Model Cost & Latency Table

| Model | Cost/1k tok | Latency(s) | Strength |
|-------|-------------|------------|----------|
| ds-pro | $0.002 | 3.0 | Coding, math, logic |
| ds-think | $0.001 | 8.0 | Deep reasoning |
| groq | $0.0002 | 0.5 | Speed (Llama 3.3 70B) |
| glm | $0.001 | 2.0 | Knowledge, Chinese |
| qwen | $0.001 | 1.5 | General, translation |
| kimi | $0.0015 | 1.0 | Writing, long context |

---

## Hyperparameter Summary (All Paper-Derived)

| Parameter | Value | Derived From |
|-----------|-------|-------------|
| β_epistemic | 0.3 | Schwartenbeck 2019 — optimal curiosity weight |
| α_kl | 0.01 | Friston 2017 — complexity-accuracy tradeoff |
| τ (temperature) | 0.1 + 0.4·difficulty | Parr 2019 — precision adapts to uncertainty |
| γ_TD | 0.9 | Standard RL, Sutton & Barto 2018 |
| Adam lr | 0.001 | Default for 3-layer MLPs |
| Hebbian lr | 0.01 | Song & Miller 2000 — STDP timescale |
| context weights | See table above | Multi-objective optimization (Miettinen 1998) |
| Beta prior | α=1, β=1 | Uniform prior (Jeffreys alternative: 0.5) |
| n_states | 6 | One per brain region |
| n_policies | 16 | (Motor × 3) + (Parietal × 3) + (PFC × 3) + (Temporal × 2) + (Language × 2) + (Visual × 2) |
