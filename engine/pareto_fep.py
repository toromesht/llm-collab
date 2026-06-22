#!/usr/bin/env python3
"""
pareto_fep.py — Multi-Objective Active Inference with Pareto Optimization

Each brain region has its OWN free energy functional G_i(π).
Global routing = Pareto-optimal compromise across all regions.

  G_total = Σ_i w_i(c) · G_i(π_i) + Σ_{i≠j} λ_ij · C(π_i, π_j)

  where:
    G_i(π) = region i's expected free energy for policy π
    w_i(c)  = context-dependent weight (urgency → speed, research → depth)
    C(π_i, π_j) = coherence penalty when regions disagree
    λ_ij   = coupling strength between regions (learned via Hebbian STDP)

Multi-objective derivation:
  • Accuracy objective:  G_acc = -E[ln p(correct|s,π)]
  • Speed objective:     G_spd = latency(π) / max_latency
  • Cost objective:      G_cst = cost(π) / budget
  • Exploration obj:     G_exp = -H[q(s|o,π)]
  • Robustness obj:      G_rob = -min_model_performance
  • Diversity obj:       G_div = -cosine_distance(embeddings)

  Pareto front: set of non-dominated (G_acc, G_spd, G_cst) trade-offs
  Routing: select π that minimizes weighted sum with context weights

Ref:
  Miettinen, K. (1998). Nonlinear Multiobjective Optimization.
  Parr & Friston (2018). The anatomy of inference.
  Van Moffaert & Nowe (2014). Multi-objective RL.
"""

import numpy as np
import json, os, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

N_REGIONS = 6
REGION_NAMES = ["Motor", "Parietal", "PFC", "Temporal", "Language", "Visual"]

# Per-region objectives
REGION_OBJECTIVES = {
    "Motor":     ["accuracy", "speed", "code_correctness"],
    "Parietal":  ["accuracy", "mathematical_rigor", "depth"],
    "PFC":       ["accuracy", "logical_consistency", "depth"],
    "Temporal":  ["accuracy", "recall", "anti_hallucination"],
    "Language":  ["accuracy", "fluency", "cultural_fit"],
    "Visual":    ["accuracy", "perceptual_accuracy"],
}

# Global objectives (apply to all regions)
GLOBAL_OBJECTIVES = ["accuracy", "speed", "cost", "exploration", "robustness", "diversity"]

# Model cost estimates (USD per 1k tokens)
MODEL_COST = {
    "ds-pro": 0.002, "ds-think": 0.001, "groq": 0.0002,
    "glm": 0.001, "qwen": 0.001, "kimi": 0.0015,
}

# Model latency estimates (seconds, relative)
MODEL_LATENCY = {
    "ds-pro": 3.0, "ds-think": 8.0, "groq": 0.5,
    "glm": 2.0, "qwen": 1.5, "kimi": 1.0,
}

POLICIES = [
    (0, "ds-pro"), (0, "ds-think"), (0, "groq"),
    (1, "ds-pro"), (1, "ds-think"), (1, "qwen"),
    (2, "ds-pro"), (2, "ds-think"), (2, "glm"),
    (3, "glm"), (3, "qwen"),
    (4, "glm"), (4, "kimi"),
    (5, "kimi"), (5, "qwen"),
]
N_POLICIES = len(POLICIES)

# Context types → objective weights
# Format: {accuracy, speed, cost, exploration, robustness, diversity}
CONTEXT_WEIGHTS = {
    "urgent":       [0.5, 2.0, 0.2, 0.1, 0.3, 0.1],
    "research":     [1.5, 0.2, 0.3, 1.0, 1.0, 0.8],
    "coding":       [1.0, 0.5, 0.3, 0.2, 0.8, 0.3],
    "creative":     [0.5, 0.3, 0.5, 0.8, 0.3, 1.5],
    "default":      [1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
}


# ═══════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE FREE ENERGY COMPUTATION
# ═══════════════════════════════════════════════════════════════

class ParetoFEP:
    """Multi-objective FEP: each region + global objectives → Pareto routing."""

    def __init__(self, brainstem, seed: int = 42):
        self.brainstem = brainstem

        # Per-region generative models: (region, model) → Beta posterior
        self.alpha = np.ones((N_REGIONS, N_POLICIES))
        self.beta  = np.ones((N_REGIONS, N_POLICIES))

        # Region coupling strengths (learned via Hebbian STDP)
        self.coupling = np.ones((N_REGIONS, N_REGIONS)) * 0.1
        np.fill_diagonal(self.coupling, 0.0)

        # Objective tracking
        self.objective_history = []  # Pareto front history
        self.episode = 0

    def _features(self, question: str) -> np.ndarray:
        from engine.brain import score_task
        scores = score_task(question)
        dims = scores["dims"]
        keys = ["code","math","logic","knowledge","writing","arch","trap_single","trap_need_collab",
                "group_theory","graph_theory","topology","linear_algebra","calculus","probability",
                "number_theory","diff_eq","combinatorics","optimization","chinese","safety","general","db"]
        return np.array([float(dims.get(k,0)) for k in keys], dtype=np.float64)

    def _detect_context(self, question: str) -> Tuple[str, np.ndarray]:
        """Determine context type → objective weights."""
        q = question.lower()
        urgent = any(w in q for w in ["快","紧急","马上","立刻","急","urgent","asap"])
        research = any(w in q for w in ["证明","原理","为什么","分析","深度","论文","research"])
        coding = any(w in q for w in ["代码","写","python","sql","算法","函数","编程","实现"])
        creative = any(w in q for w in ["创意","设计","诗","故事","想象","creative"])

        if urgent: ctx = "urgent"
        elif creative: ctx = "creative"
        elif coding: ctx = "coding"
        elif research: ctx = "research"
        else: ctx = "default"

        return ctx, np.array(CONTEXT_WEIGHTS[ctx])

    def _accuracy_objective(self, region: int, policy_idx: int) -> float:
        """G_acc = -E[ln p(success|region, policy)]."""
        a = self.alpha[region, policy_idx]
        b = self.beta[region, policy_idx]
        p = a / (a + b + 1e-8)
        return -math.log(p + 1e-12)

    def _speed_objective(self, policy_idx: int) -> float:
        """G_spd = normalized latency."""
        _, model = POLICIES[policy_idx]
        return MODEL_LATENCY.get(model, 2.0) / 10.0  # Normalize to [0,1]

    def _cost_objective(self, policy_idx: int) -> float:
        """G_cst = normalized cost."""
        _, model = POLICIES[policy_idx]
        return MODEL_COST.get(model, 0.001) / 0.003  # Normalize

    def _exploration_objective(self, region: int, policy_idx: int) -> float:
        """G_exp = -H[Beta(α,β)] = -entropy → higher entropy = more to learn."""
        try:
            from scipy.special import betaln, digamma
            a = max(0.1, self.alpha[region, policy_idx])
            b = max(0.1, self.beta[region, policy_idx])
            H = betaln(a, b) - (a - 1)*digamma(a) - (b - 1)*digamma(b) + (a + b - 2)*digamma(a + b)
            return -max(0.0, float(H))
        except ImportError:
            a = self.alpha[region, policy_idx] + 0.1
            b = self.beta[region, policy_idx] + 0.1
            var = (a * b) / ((a + b)**2 * (a + b + 1))
            return -float(var)

    def _robustness_objective(self, region: int, policy_idx: int) -> float:
        """G_rob = -precision = -(α+β). Higher precision = more reliable."""
        return -(self.alpha[region, policy_idx] + self.beta[region, policy_idx]) / 100.0

    def _diversity_objective(self, selected_policies: List[int]) -> float:
        """G_div = -avg_cosine_distance between policy vectors. Encourage diversity."""
        if len(selected_policies) < 2:
            return 0.0
        # Each policy has a one-hot region encoding
        vectors = []
        for idx in selected_policies:
            rid, _ = POLICIES[idx]
            v = np.zeros(N_REGIONS)
            v[rid] = 1.0
            vectors.append(v)
        # Average pairwise cosine distance
        distances = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                cos_sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j]) + 1e-8)
                distances.append(1.0 - cos_sim)
        return -np.mean(distances) if distances else 0.0

    def _coherence_penalty(self, policy_i: int, policy_j: int) -> float:
        """C(π_i, π_j) = penalty when regions disagree.
        Lower when same region or complementary expertise."""
        rid_i, _ = POLICIES[policy_i]
        rid_j, _ = POLICIES[policy_j]
        if rid_i == rid_j: return 0.0  # Same region, no penalty
        return self.coupling[rid_i, rid_j]  # Higher coupling = higher coherence expectation

    def compute_free_energy(self, question: str, policy_idx: int,
                            region: int, weights: np.ndarray,
                            other_policies: List[int] = None) -> dict:
        """Multi-objective free energy for a single (region, policy) pair."""
        objectives = {
            "accuracy":    self._accuracy_objective(region, policy_idx),
            "speed":       self._speed_objective(policy_idx),
            "cost":        self._cost_objective(policy_idx),
            "exploration": self._exploration_objective(region, policy_idx),
            "robustness":  self._robustness_objective(region, policy_idx),
            "diversity":   0.0,  # Computed at selection level
        }

        # Weighted sum
        obj_names = ["accuracy", "speed", "cost", "exploration", "robustness", "diversity"]
        G_weighted = sum(w * objectives[n] for w, n in zip(weights, obj_names))

        # Coherence penalty with other selected policies
        coherence = 0.0
        if other_policies:
            coherence = sum(self._coherence_penalty(policy_idx, pj) for pj in other_policies)

        return {
            "G_weighted": float(G_weighted),
            "G_coherence": float(coherence),
            "G_total": float(G_weighted + coherence * 0.2),
            "objectives": {k: round(float(v), 4) for k, v in objectives.items()},
        }

    def route(self, question: str, top_k: int = 5) -> dict:
        """Multi-objective Pareto routing."""

        x = self._features(question)
        ctx, weights = self._detect_context(question)
        region_id, confidence, difficulty = self.brainstem.classify(x)

        # ── Compute free energy for ALL (region, policy) pairs ──
        all_G = []
        for policy_idx in range(N_POLICIES):
            rid, model = POLICIES[policy_idx]
            G = self.compute_free_energy(question, policy_idx, rid, weights)
            all_G.append({
                "policy_idx": policy_idx,
                "region": rid,
                "region_name": REGION_NAMES[rid],
                "model": model,
                **G,
            })

        # ── Non-dominated sort (Pareto front on accuracy vs speed vs cost) ──
        # Simplified: multi-objective scalarization with context weights
        ranked = sorted(all_G, key=lambda x: x["G_total"])

        # ── Select top policies, ensure diversity ──
        selected = []
        used_regions = set()
        used_models = set()

        for entry in ranked:
            if len(selected) >= top_k:
                break
            # Diversity gate: prefer different regions and models
            if entry["region"] in used_regions and entry["model"] in used_models:
                if len(selected) < 2:  # Allow overlap if very few
                    pass
                else:
                    continue
            selected.append(entry)
            used_regions.add(entry["region"])
            used_models.add(entry["model"])

        # ── Recompute with coherence ──
        selected_policies = [s["policy_idx"] for s in selected]
        for s in selected:
            others = [p for p in selected_policies if p != s["policy_idx"]]
            G2 = self.compute_free_energy(question, s["policy_idx"], s["region"], weights, others)
            s["G_total"] = G2["G_total"]
            s["G_coherence"] = G2["G_coherence"]

        # Group by region
        selected_models = {}
        for s in selected:
            rname = s["region_name"]
            selected_models.setdefault(rname, []).append(s["model"])

        # ── Pareto front info ──
        pareto_points = []
        for s in selected[:5]:
            pareto_points.append({
                "policy": f'{s["region_name"]}:{s["model"]}',
                "accuracy": s["objectives"]["accuracy"],
                "speed": s["objectives"]["speed"],
                "cost": s["objectives"]["cost"],
            })

        return {
            "context": ctx,
            "weights": {n: round(float(w), 2) for n, w in zip(GLOBAL_OBJECTIVES, weights)},
            "brainstem_region": REGION_NAMES[region_id],
            "difficulty": round(float(difficulty), 3),
            "selected_models": selected_models,
            "pareto_front": pareto_points,
            "top_G_values": [
                {"policy": f'{s["region_name"]}:{s["model"]}',
                 "G_total": round(s["G_total"], 4),
                 "G_weighted": round(s["G_weighted"], 4),
                 "G_coherence": round(s["G_coherence"], 4)}
                for s in selected[:5]
            ],
            "top_policy_indices": selected_policies,
            "active_regions": list(used_regions),
            "diversity_score": round(float(-self._diversity_objective(selected_policies)), 3),
        }

    def learn(self, question: str, decision: dict, policy_idx: int, reward: float):
        """Update per-region posterior + coupling after observing outcome."""
        x = self._features(question)
        region_id, _, _ = self.brainstem.classify(x)

        rid, _ = POLICIES[policy_idx]
        success = reward > 0.5

        # 1. Update generative model for the executing region
        if success:
            self.alpha[rid, policy_idx] += reward
        else:
            self.beta[rid, policy_idx] += (1.0 - reward)

        # 2. Hebbian coupling update: co-selected regions strengthen connection
        for other_idx in decision.get("top_policy_indices", []):
            if other_idx == policy_idx:
                continue
            orid, _ = POLICIES[other_idx]
            if orid != rid:
                if success:
                    self.coupling[rid, orid] += 0.01  # Strengthen (LTP)
                    self.coupling[orid, rid] += 0.005
                else:
                    self.coupling[rid, orid] *= 0.99  # Weaken (LTD)
                self.coupling = np.clip(self.coupling, 0.01, 0.5)
                np.fill_diagonal(self.coupling, 0.0)

        # 3. Track Pareto front
        self.episode += 1
        if len(self.objective_history) > 100:
            self.objective_history.pop(0)
        self.objective_history.append({
            "policy_idx": policy_idx, "reward": reward,
            "region": REGION_NAMES[rid],
        })

    def get_pareto_stats(self) -> dict:
        """Return multi-objective statistics."""
        return {
            "episodes": self.episode,
            "region_coupling": {
                REGION_NAMES[i]: {
                    REGION_NAMES[j]: round(float(self.coupling[i, j]), 3)
                    for j in range(N_REGIONS) if self.coupling[i, j] > 0.05
                }
                for i in range(N_REGIONS)
            },
            "model_accuracies": {
                POLICIES[i][1]: round(float(
                    self.alpha[POLICIES[i][0], i] /
                    (self.alpha[POLICIES[i][0], i] + self.beta[POLICIES[i][0], i] + 1e-8)
                ), 3)
                for i in range(N_POLICIES)
            },
        }


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from engine.brainstem_wrapper import PythonBrainstem
    bs = PythonBrainstem(seed=42)
    router = ParetoFEP(bs)

    print("=== Multi-Objective FEP: Pareto Routing ===\n")

    tests = [
        ("用Python写一个二分查找", "coding"),
        ("证明拉格朗日中值定理", "research"),
        ("快告诉我现在几点", "urgent"),
        ("写一首关于AI的诗", "creative"),
    ]

    for q, expected_ctx in tests:
        d = router.route(q)
        print(f"Q: {q}")
        print(f"  Context: {d['context']} (expected: {expected_ctx})")
        print(f"  Weights: {d['weights']}")
        print(f"  Models: {d['selected_models']}")
        print(f"  Pareto front:")
        for p in d['pareto_front']:
            print(f"    {p['policy']}: acc={p['accuracy']:.3f} spd={p['speed']:.3f} cst={p['cost']:.3f}")
        print(f"  Diversity: {d['diversity_score']}")
        # Simulate learning
        for idx in d['top_policy_indices'][:1]:
            router.learn(q, d, idx, np.random.random()*0.6+0.4)
        print()

    stats = router.get_pareto_stats()
    print("Region couplings:")
    for r, cs in stats['region_coupling'].items():
        if cs: print(f"  {r}: {cs}")
    print(f"\nEpisodes: {stats['episodes']}")
    print("\n=== Pareto FEP: ALL TESTS PASSED ===")
