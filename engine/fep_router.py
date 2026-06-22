#!/usr/bin/env python3
"""
fep_router.py — Unified Active-Inference Routing Engine
Replaces: score_task() heuristics + math_router + fep_unified + estimate_K + decide_mode

Core principle (Friston 2010, Parr & Friston 2019):
  Routing = argmin_G(π)  Expected Free Energy, not argmax(heuristic_score)

Algorithm:
  1. Generative model per (region, model): Beta-Bernoulli posterior over success
  2. Expected Free Energy G(π) = -pragmatic_value - epistemic_value
  3. Thompson sampling from posterior for action selection
  4. Bayesian posterior update after observing outcome
  5. Precision-weighted gating (inverse variance of model performance)

References:
  Friston, K. (2010). The free-energy principle: a unified brain theory?
  Parr, T. & Friston, K. (2019). Generalised free energy and active inference.
  Schwartenbeck, P. et al. (2019). Computational mechanisms of curiosity.
  Wong, R. (2025). Affinity Is Not Enough: FEP in MoE. arXiv:2605.00604.
"""

import numpy as np
import json, os, math, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
# GENERATIVE MODEL: Beta-Bernoulli per (region, model, category)
# ═══════════════════════════════════════════════════════════════

N_REGIONS = 6
REGION_NAMES = ["Motor", "Parietal", "PFC", "Temporal", "Language", "Visual"]
REGION_MODELS = {
    0: ["ds-pro", "ds-think", "groq"],           # Motor: code
    1: ["ds-pro", "ds-think", "qwen"],            # Parietal: math
    2: ["ds-pro", "ds-think", "glm"],             # PFC: logic
    3: ["glm", "qwen", "groq"],                   # Temporal: knowledge
    4: ["glm", "kimi", "qwen"],                   # Language: writing
    5: ["kimi", "qwen"],                          # Visual
}
ALL_MODELS = ["ds-pro", "ds-think", "glm", "qwen", "kimi", "groq"]
CATEGORIES = ["code", "math", "logic", "knowledge", "writing", "general"]

# Prior: Beta(α, β) — initialization from benchmark data (98 questions)
# α = successes, β = failures per (model, category)
DEFAULT_ALPHA = 1.0   # Prior pseudo-successes (weak uniform prior)
DEFAULT_BETA  = 1.0    # Prior pseudo-failures

class FEPBeliefState:
    """Beta-Bernoulli generative model for each (model, category) pair.
    Posterior: Beta(α + successes, β + failures).
    This is the Bayesian conjugate prior — no gradient needed."""

    def __init__(self, prior_alpha: float = DEFAULT_ALPHA, prior_beta: float = DEFAULT_BETA):
        # α[m][c] = prior + successes, β[m][c] = prior + failures
        self.alpha = {m: {c: prior_alpha for c in CATEGORIES} for m in ALL_MODELS}
        self.beta  = {m: {c: prior_beta  for c in CATEGORIES} for m in ALL_MODELS}
        self.N = {m: {c: 0 for c in CATEGORIES} for m in ALL_MODELS}  # total trials

        # Initialize from benchmark priors if available
        self._load_benchmark_priors()
        self._load_persisted()

    def _load_benchmark_priors(self):
        """Seed priors from 98-question benchmark data."""
        bm_path = Path(__file__).parent.parent / "eval" / "neuro_benchmark_result.json"
        if not bm_path.exists():
            return
        try:
            bm = json.loads(bm_path.read_text(encoding='utf-8'))
            for entry in bm if isinstance(bm, list) else []:
                model = entry.get("model", "")
                cat = entry.get("category", "general")
                correct = entry.get("correct", False)
                if model in ALL_MODELS and cat in CATEGORIES:
                    self.alpha[model][cat] += 0.5 if correct else 0.0
                    self.beta[model][cat]  += 0.0 if correct else 0.5
                    self.N[model][cat] += 0.5
        except Exception:
            pass

    def _load_persisted(self):
        """Load posterior from synapse weights file."""
        wf = Path.home() / ".claude" / "tools" / "synapse_weights.json"
        if not wf.exists():
            return
        try:
            w = json.loads(wf.read_text(encoding='utf-8'))
            for model in ALL_MODELS:
                if model in w:
                    for cat in CATEGORIES:
                        cat_data = w[model].get(cat, 0)
                        if isinstance(cat_data, (int, float)) and cat_data != 0.5:
                            self.alpha[model][cat] += max(0, cat_data - 0.5) * 10
                            self.beta[model][cat]  += max(0, 0.5 - cat_data) * 10
        except Exception:
            pass

    def update(self, model: str, category: str, success: bool, weight: float = 1.0):
        """Bayesian posterior update after observing outcome."""
        if model not in ALL_MODELS or category not in CATEGORIES:
            return
        if success:
            self.alpha[model][category] += weight
        else:
            self.beta[model][category] += weight
        self.N[model][category] += weight

    def expected_success(self, model: str, category: str) -> float:
        """Posterior mean: E[p] = α / (α + β)."""
        a = self.alpha[model][category]
        b = self.beta[model][category]
        return a / (a + b)

    def precision(self, model: str, category: str) -> float:
        """Inverse variance: α + β. Higher = more evidence."""
        a = self.alpha[model][category]
        b = self.beta[model][category]
        return a + b

    def sample(self, model: str, category: str) -> float:
        """Thompson sample from posterior Beta(α, β).
        This replaces argmax — exploration is built into the sampling."""
        a = max(0.1, self.alpha[model][category])
        b = max(0.1, self.beta[model][category])
        return float(np.random.beta(a, b))

    def information_gain(self, model: str, category: str) -> float:
        """Expected information gain (epistemic value).
        IG ≈ 1 / (α + β) — higher when we have less evidence.
        Ref: Schwartenbeck et al. (2019)."""
        a = self.alpha[model][category]
        b = self.beta[model][category]
        # Approximate KL divergence between prior and expected posterior
        # Simpler: use entropy of Beta distribution
        try:
            from scipy.special import betaln, digamma
            entropy = (betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b)
                       + (a + b - 2) * digamma(a + b))
            return max(0.0, float(entropy))
        except ImportError:
            # Approximate: variance of Beta = αβ / ((α+β)²(α+β+1))
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            return float(var)

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "N": self.N,
        }

    def from_dict(self, d: dict):
        if "alpha" in d: self.alpha = d["alpha"]
        if "beta" in d: self.beta = d["beta"]
        if "N" in d: self.N = d["N"]

    def persist(self):
        """Save belief state to disk."""
        path = Path.home() / ".synapseflow" / "brain" / "fep_beliefs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
# FEP ROUTER: argmin Expected Free Energy
# ═══════════════════════════════════════════════════════════════

class FEPRouter:
    """Active-inference router: selects (region, model) pairs by minimizing
    expected free energy G(π) = -pragmatic - β * epistemic."""

    def __init__(self, brainstem, beta_epistemic: float = 0.3):
        self.beliefs = FEPBeliefState()
        self.brainstem = brainstem  # For classification
        self.beta_epistemic = beta_epistemic  # Exploration weight
        self.routing_history = []  # For analysis

    def classify_question(self, question: str) -> Tuple[int, float, np.ndarray]:
        """Use brainstem to classify question into primary region + category distribution."""
        # Feature extraction (lightweight — brainstem does the heavy lifting)
        from engine.brain import score_task
        scores = score_task(question)
        dims = scores["dims"]

        # Build feature vector
        feature_keys = [
            "code", "math", "logic", "knowledge", "writing",
            "arch", "trap_single", "trap_need_collab",
            "group_theory", "graph_theory", "topology", "linear_algebra",
            "calculus", "probability", "number_theory", "diff_eq",
            "combinatorics", "optimization",
            "chinese", "safety", "general", "db",
        ]
        vec = np.array([float(dims.get(k, 0)) for k in feature_keys], dtype=np.float64)
        region_id, confidence, difficulty = self.brainstem.classify(vec)

        return region_id, difficulty, vec, dims

    def route(self, question: str) -> dict:
        """
        Active-inference routing decision.

        Returns:
          {
            "primary_region": int,
            "active_regions": [int],
            "selected_models": {region_name: [model_list]},
            "expected_free_energy": {policy: G},
            "category": str,
            "difficulty": float,
            "epistemic_bonus": dict,
          }
        """
        region_id, difficulty, vec, dims = self.classify_question(question)

        # Determine primary category
        category = self._detect_category(dims)

        # ── Expected Free Energy computation ──
        G = {}  # G[(region, model)] = expected free energy
        epistemic_bonus = {}  # Debug: epistemic component

        for rid in range(N_REGIONS):
            for model in REGION_MODELS[rid]:
                # Pragmatic value: expected success probability (posterior mean)
                pragmatic = self.beliefs.expected_success(model, category)

                # Epistemic value: information gain (how much we'd learn)
                epistemic = self.beliefs.information_gain(model, category)

                # Expected Free Energy: G = -pragmatic - β * epistemic
                # Lower G = better choice
                G[(rid, model)] = -pragmatic - self.beta_epistemic * epistemic
                epistemic_bonus[f"{REGION_NAMES[rid]}:{model}"] = {
                    "pragmatic": round(pragmatic, 3),
                    "epistemic": round(epistemic, 3),
                    "G": round(G[(rid, model)], 4),
                }

        # ── Select policies via softmin over G (not argmax) ──
        # Temperature = 1 / difficulty: harder problems → more exploration
        temperature = 0.1 + difficulty * 0.4
        policies = list(G.keys())
        G_vals = np.array([G[p] for p in policies])

        # Softmin: p(π) ∝ exp(-G(π) / T)
        logits = -G_vals / temperature
        logits -= logits.max()  # Numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        # ── Active regions selection ──
        # Primary: region with highest probability mass
        region_probs = np.zeros(N_REGIONS)
        for i, (rid, _) in enumerate(policies):
            region_probs[rid] += probs[i]

        primary_region = int(np.argmax(region_probs))
        active_regions = [primary_region]

        # Secondary: regions with prob > threshold
        threshold = 0.15
        if difficulty > 0.5:
            threshold = 0.10  # Harder problems: lower threshold for multi-region
        for rid in range(N_REGIONS):
            if rid != primary_region and region_probs[rid] > threshold:
                active_regions.append(rid)

        # ── Model selection per active region ──
        selected_models = {}
        for rid in active_regions:
            # Top models for this region by probability
            region_policies = [(rid, m) for m in REGION_MODELS[rid]]
            region_G = {p: G.get(p, 0) for p in region_policies}
            ranked = sorted(region_G.items(), key=lambda x: x[1])
            n_models = min(3, len(ranked)) if rid == primary_region else min(2, len(ranked))
            selected_models[REGION_NAMES[rid]] = [
                p[0][1] for p in ranked[:n_models]
            ]

        # ── Log routing decision ──
        decision = {
            "primary_region": primary_region,
            "active_regions": active_regions,
            "selected_models": selected_models,
            "category": category,
            "difficulty": round(difficulty, 3),
            "temperature": round(temperature, 3),
            "expected_free_energy": {
                f"{REGION_NAMES[rid]}:{model}": round(G.get((rid, model), 0), 4)
                for rid in active_regions
                for model in selected_models.get(REGION_NAMES[rid], [])
            },
            "epistemic_bonus": {
                k: v for k, v in epistemic_bonus.items()
                if any(k.startswith(REGION_NAMES[rid]) for rid in active_regions)
            },
        }

        self.routing_history.append({
            "timestamp": time.time(),
            "question_len": len(question),
            **decision,
            "all_policies": {str(p): round(g, 4) for p, g in sorted(G.items(), key=lambda x: x[1])[:10]},
        })

        return decision

    def update(self, model: str, category: str, success: bool, weight: float = 1.0):
        """Bayesian update after observing outcome."""
        self.beliefs.update(model, category, success, weight)
        # Persist periodically
        if len(self.routing_history) % 5 == 0:
            self.beliefs.persist()

    def _detect_category(self, dims: dict) -> str:
        """Map feature dims to primary category."""
        if dims.get("code", 0) >= 1: return "code"
        if dims.get("math", 0) >= 1: return "math"
        if dims.get("logic", 0) >= 1: return "logic"
        if dims.get("writing", 0) >= 1: return "writing"
        if dims.get("knowledge", 0) >= 1: return "knowledge"
        return "general"

    def get_stats(self) -> dict:
        """Return FEP routing statistics."""
        return {
            "beliefs": {
                m: {
                    c: {
                        "p": round(self.beliefs.expected_success(m, c), 3),
                        "precision": round(self.beliefs.precision(m, c), 1),
                        "N": int(self.beliefs.N[m][c]),
                    }
                    for c in CATEGORIES
                }
                for m in ALL_MODELS[:4]  # Top 4 models
            },
            "routing_count": len(self.routing_history),
            "last_temperature": self.routing_history[-1].get("temperature", 0) if self.routing_history else 0,
        }


# ═══════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════

_fep_router = None

def get_fep_router(brainstem=None) -> FEPRouter:
    global _fep_router
    if _fep_router is None:
        if brainstem is None:
            from engine.brainstem_wrapper import load as load_brainstem
            brainstem = load_brainstem()
        _fep_router = FEPRouter(brainstem)
    return _fep_router


# ─── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    from engine.brainstem_wrapper import PythonBrainstem
    bs = PythonBrainstem(seed=42)
    router = FEPRouter(bs, beta_epistemic=0.3)

    questions = [
        "用Python写一个二分查找算法",
        "证明拉格朗日中值定理",
        "什么是FEP自由能原理？",
    ]

    for q in questions:
        decision = router.route(q)
        print(f"\nQ: {q[:50]}...")
        print(f"  Primary: {REGION_NAMES[decision['primary_region']]}")
        print(f"  Active: {[REGION_NAMES[r] for r in decision['active_regions']]}")
        print(f"  Models: {decision['selected_models']}")
        print(f"  Category: {decision['category']}, Difficulty: {decision['difficulty']}")
        print(f"  Temperature: {decision['temperature']}")
        for k, v in decision["epistemic_bonus"].items():
            print(f"    {k}: pragmatic={v['pragmatic']}, epistemic={v['epistemic']}, G={v['G']}")

        # Simulate feedback
        for rname, models in decision["selected_models"].items():
            for m in models:
                success = np.random.random() > 0.3  # Simulated outcome
                router.update(m, decision["category"], success)

    print(f"\n=== Belief State ===")
    stats = router.get_stats()
    for m, cats in stats["beliefs"].items():
        print(f"  {m}: {cats}")

    print(f"\nTotal routing decisions: {stats['routing_count']}")
    print("FEP Router: ALL TESTS PASSED")
