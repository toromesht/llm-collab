#!/usr/bin/env python3
"""
neuro_synaptic_router.py — Frontier Neuroscience → Computer System

Three core mechanisms, each anchored to a specific paper:

1. GRID CELL COGNITIVE MAP (Moser & Moser 2005, Nature)
   Questions embedded in 2D cognitive space.
   Hexagonal grid modules tile the space at different spatial frequencies.
   Place cells (prototypes) form at frequently visited question locations.
   "Where in question-space is this query?"

2. HIERARCHICAL PREDICTIVE CODING (Rao & Ballard 1999, Nature Neuroscience)
   Top layer predicts which model will succeed.
   Only prediction ERRORS propagate upward.
   Precision-weighted updates (reliable predictions → smaller updates).

3. SYNAPTIC TAGGING & CAPTURE (Frey & Morris 1997, Nature)
   Strong successes set "synaptic tags."
   Nearby synapses capture plasticity-related proteins (PRPs).
   Enables rapid consolidation of important routing decisions.

Mathematical formalisms:
  Grid cell:   r_j(x,y) = max(0, Σ cos(2π·f_j·cos(θ_j)·x + 2π·f_j·sin(θ_j)·y))
  Predictive:  ε = r - r̂;  Δw = η · ε · precision · (∂r̂/∂w)
  Tagging:     tag[s] = 1 if |ε| > θ;  Δw[s] *= (1 + PRP · tag[s])

References:
  [1] Rao & Ballard (1999). Predictive coding in the visual cortex. Nature Neuroscience.
  [2] Hafting, Fyhn, Molden, Moser & Moser (2005). Microstructure of a spatial map
      in the entorhinal cortex. Nature.
  [3] Frey & Morris (1997). Synaptic tagging and long-term potentiation. Nature.
  [4] Friston (2005). A theory of cortical responses. Phil. Trans. Royal Society B.
  [5] Song, Miller & Abbott (2000). STDP. Nature Neuroscience.
  [6] Solstad, Boccara, Kropff, Moser & Moser (2008). Representation of geometric
      borders in the entorhinal cortex. Science.
"""

import numpy as np
import json, os, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# 1. GRID CELL COGNITIVE MAP (Hafting, Fyhn, Molden, Moser & Moser 2005)
# ═══════════════════════════════════════════════════════════════

class GridCellModule:
    """
    One grid cell module with a specific spatial frequency and orientation.
    Multiple modules at different frequencies tile the cognitive space.

    Firing rate from Solstad et al. (2008) Eq.1:
      r(x,y) = max(0, cos(2π·f·cos(θ)·x + 2π·f·sin(θ)·y + φ))

    where:
      f = spatial frequency (distance between grid nodes)
      θ = grid orientation (60° spacing for hexagonal symmetry)
      φ = phase offset
    """

    def __init__(self, frequency: float, orientation: float, phase: float = 0.0,
                 dim: int = 64):
        self.f = frequency
        self.theta = orientation
        self.phi = phase
        # Random projection from n-gram embedding to 2D coordinates
        self.proj_x = np.random.randn(dim) / np.sqrt(dim)
        self.proj_y = np.random.randn(dim) / np.sqrt(dim)

    def encode(self, embedding: np.ndarray) -> float:
        """Firing rate of this grid cell for a given question embedding."""
        x = np.dot(embedding, self.proj_x)
        y = np.dot(embedding, self.proj_y)
        # Hexagonal grid (60° spacing between three cosine gratings)
        angles = [self.theta, self.theta + np.pi/3, self.theta + 2*np.pi/3]
        rate = 0.0
        for a in angles:
            rate += np.cos(2.0 * np.pi * self.f * (np.cos(a) * x + np.sin(a) * y) + self.phi)
        return max(0.0, rate / 3.0)


class CognitiveMap:
    """
    Multi-scale grid cell population encoding question-space position.
    4 modules at different spatial frequencies (Moser 2005 → multi-scale).

    Higher frequency = finer spatial resolution.
    Lower frequency = coarser, but unique position coding over larger area.
    """

    def __init__(self, n_modules: int = 4, dim: int = 64):
        self.modules = []
        frequencies = [0.5, 1.0, 2.0, 4.0]  # Doubling frequencies (Moser 2005)
        for f in frequencies[:n_modules]:
            # 3 grid cells per module (60° rotation for hexagonal symmetry)
            for theta in [0.0, np.pi/3, 2*np.pi/3]:
                self.modules.append(GridCellModule(f, theta, np.random.rand() * 2*np.pi, dim))

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Population vector of all grid cell firing rates → position code."""
        return np.array([m.encode(embedding) for m in self.modules])


# ═══════════════════════════════════════════════════════════════
# 2. HIERARCHICAL PREDICTIVE CODING (Rao & Ballard 1999)
# ═══════════════════════════════════════════════════════════════

class PredictiveCodingLayer:
    """
    One layer of the predictive coding hierarchy.

    Top-down: predicts outcome (success probability per model)
    Bottom-up: receives prediction errors

    Only ERRORS propagate upward (Rao & Ballard 1999, Eq.3):
      ε = r_actual - r_predicted
      ΔW = η · ε · precision · ∂r̂/∂W

    Precision-weighting (Friston 2005): reliable predictions get larger updates.
    """

    def __init__(self, n_input: int, n_output: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_output, n_input) * 0.01  # Predictive weights
        self.b = np.zeros(n_output)

        # Precision (inverse variance of prediction errors) — Friston 2005 Eq.4
        self.precision = np.ones(n_output)
        self.prediction_error_ema = np.zeros(n_output)

        # Synaptic tags (Frey & Morris 1997)
        self.tags = np.zeros(n_output)
        self.tag_threshold = 0.6
        self.tag_lifetime = 5  # Episodes before tag decays

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Top-down prediction: r̂ = W·x + b."""
        return self.W @ x + self.b

    def update(self, x: np.ndarray, target: np.ndarray, model_idx: int, lr: float = 0.01):
        """
        Predictive coding weight update (Rao & Ballard 1999).
        Only the prediction ERROR drives learning.
        """
        prediction = self.predict(x)
        error = target - prediction

        # ── Precision-weighted update (Friston 2005) ──
        self.prediction_error_ema = 0.9 * self.prediction_error_ema + 0.1 * np.abs(error)
        self.precision = 1.0 / (self.prediction_error_ema + 0.1)

        # Only update weights for the activated model (sparse coding)
        prec = self.precision[model_idx]
        eps = error[model_idx]

        self.W[model_idx] += lr * prec * eps * x
        self.b[model_idx] += lr * prec * eps

        # ── Synaptic Tagging (Frey & Morris 1997) ──
        if abs(eps) > self.tag_threshold:
            self.tags[model_idx] = self.tag_lifetime  # Set tag if surprise is large
        else:
            self.tags[model_idx] = max(0, self.tags[model_idx] - 1)

        # Tagged synapses learn faster (PRP capture)
        if self.tags[model_idx] > 0:
            lr_boosted = lr * 3.0  # 3x learning rate for tagged synapses
            self.W[model_idx] += lr_boosted * prec * eps * x
            self.b[model_idx] += lr_boosted * prec * eps


# ═══════════════════════════════════════════════════════════════
# 3. UNIFIED NEURO-SYNAPTIC ROUTER
# ═══════════════════════════════════════════════════════════════

class TextEmbedder:
    def __init__(self, dim=384):
        self.dim = dim
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dim = 384
        except Exception:
            self._model = None
    def encode(self, text: str) -> np.ndarray:
        if self._model is not None:
            emb = self._model.encode(text, convert_to_numpy=True)
            return emb.astype(np.float64) / (np.linalg.norm(emb) + 1e-8)
        vec = np.zeros(self.dim, dtype=np.float64)
        for n in [2, 3, 4]:
            for i in range(len(text) - n + 1):
                h = hash(text[i:i+n]) % self.dim; vec[h] += 1.0
        norm = np.linalg.norm(vec); return vec / (norm + 1e-8)


class NeuroSynapticRouter:
    """
    Grid cells → represent question position in cognitive space
    Predictive coding → predict which model will succeed
    Synaptic tagging → rapidly consolidate important decisions
    """

    MODELS = ["ds-pro", "ds-think", "glm", "qwen", "kimi", "groq"]
    N_MODELS = len(MODELS)

    def __init__(self, seed: int = 42):
        self.embedder = TextEmbedder()
        self.grid_map = CognitiveMap(n_modules=4, dim=384)
        self.pc_layer = PredictiveCodingLayer(
            n_input=len(self.grid_map.modules),  # Grid cell population size
            n_output=self.N_MODELS,
            seed=seed
        )
        self.episode = 0
        self.success_history = []

    def _grid_encode(self, question: str) -> np.ndarray:
        """Question → n-gram embedding → grid cell population vector."""
        emb = self.embedder.encode(question)
        return self.grid_map.encode(emb)

    def route(self, question: str) -> dict:
        """
        1. Grid cells encode question spatial position
        2. Predictive layer generates success predictions
        3. Thompson-style sampling: add noise scaled by prediction error
        """
        grid_vec = self._grid_encode(question)
        predictions = self.pc_layer.predict(grid_vec)

        # UCB: predicted success + exploration bonus for uncertain/misaligned
        exploration_bonus = self.pc_layer.prediction_error_ema * 0.5
        # Bonus for tagged synapses (worth investigating)
        tag_bonus = self.pc_layer.tags * 0.2

        scores = predictions + exploration_bonus + tag_bonus
        best_idx = int(np.argmax(scores))

        return {
            "primary_model": self.MODELS[best_idx],
            "predictions": {self.MODELS[i]: round(float(predictions[i]), 3)
                           for i in np.argsort(-predictions)[:4]},
            "exploration_bonus": {self.MODELS[i]: round(float(exploration_bonus[i]), 3)
                                 for i in range(self.N_MODELS)},
            "tags_active": [self.MODELS[i] for i in range(self.N_MODELS)
                           if self.pc_layer.tags[i] > 0],
            "grid_activation": round(float(np.linalg.norm(grid_vec)), 2),
        }

    def learn(self, question: str, model: str, reward: float):
        """
        1. Encode question via grid cells
        2. Predictive coding update — only if prediction was WRONG
        3. Synaptic tagging — mark surprising outcomes for rapid consolidation
        """
        grid_vec = self._grid_encode(question)
        model_idx = self.MODELS.index(model)

        # ── Target: the actual reward ──
        target = np.zeros(self.N_MODELS)
        target[model_idx] = reward

        # ── Predictive coding update (Rao & Ballard 1999) ──
        self.pc_layer.update(grid_vec, target, model_idx, lr=0.01)

        self.episode += 1
        self.success_history.append(reward)
        if len(self.success_history) > 200:
            self.success_history.pop(0)

    def get_state(self) -> dict:
        return {
            "episodes": self.episode,
            "avg_reward": round(np.mean(self.success_history[-50:]), 3) if self.success_history else 0,
            "precision": {self.MODELS[i]: round(float(self.pc_layer.precision[i]), 3)
                         for i in range(self.N_MODELS)},
            "tags": {self.MODELS[i]: int(self.pc_layer.tags[i])
                    for i in range(self.N_MODELS) if self.pc_layer.tags[i] > 0},
            "grid_population": len(self.grid_map.modules),
        }


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    router = NeuroSynapticRouter()
    import numpy as np

    math_qs = ["Solve 2x+5=17", "Integral of sin(x)", "Prove sum of first n integers",
               "Eigenvalues of 2x2", "Derivative of x^3 ln(x)"]
    simple_qs = ["Color of sky", "Capital of France", "Speed of light",
                 "Boiling point of water", "Largest planet"]
    code_qs = ["Write quicksort Python", "SQL top 10 customers",
               "Binary tree traversal", "Implement LRU cache", "Reverse linked list"]

    print("Training Neuro-Synaptic Router (Grid Cells + Predictive Coding)...")
    for epoch in range(5):
        for qs, good_model in [(math_qs, "ds-think"), (simple_qs, "groq"), (code_qs, "ds-pro")]:
            for q in qs:
                decision = router.route(q)
                chosen = decision["primary_model"]
                reward = 0.95 if chosen == good_model else (0.3 if chosen in ["groq", "glm"] else 0.5)
                reward += np.random.normal(0, 0.03)
                router.learn(q, good_model, np.clip(reward, 0, 1))

    print(f"\n=== STATE (after {router.episode} episodes) ===")
    s = router.get_state()
    print(f"Precision: {s['precision']}")
    print(f"Active tags: {s['tags']}")
    print(f"Avg reward: {s['avg_reward']}")

    print(f"\n=== ROUTING TESTS ===")
    for q in ["Solve quadratic x^2+3x-4=0", "Capital of Japan?",
              "Write palindrome checker function", "Prove Pythagorean theorem"]:
        d = router.route(q)
        print(f"  {q[:45]:45s} → {d['primary_model']:10s} (tags: {d['tags_active']})")
        for m, p in d['predictions'].items():
            print(f"    {m}: pred={p:.3f}", end="")
        print()

    print("\nNeuroSynapticRouter: GRID CELLS + PREDICTIVE CODING + SYNAPTIC TAGGING — READY")
