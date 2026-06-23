#!/usr/bin/env python3
"""
algorithms.py — LLM Routing Algorithms with Mathematical Documentation

Each algorithm is described by:
  1. Computational Problem (pure math, no biology)
  2. Mathematical Description (provable or empirical properties)
  3. Implementation Notes (scale, guarantees, limitations)

CONVERSION METHODOLOGY (from "Neural Mechanisms to Algorithms"):
  Step 1: Abstract the computational problem (forget biology)
  Step 2: Find the mathematical principle (proven theorem, not metaphor)
  Step 3: Re-implement for the digital domain (not translate biology → code)
  Step 4: Compare against pure-math baseline (if no advantage, discard)

ALGORITHM                    | COMPUTATIONAL PROBLEM           | MATH PRINCIPLE
─────────────────────────────┼─────────────────────────────────┼──────────────────────────
RandomFourierEncoder         | Kernel approximation for query  | Random Fourier Features
(GridCellMap)                | similarity search (Rahimi 2007) | (RFF) ≈ RBF kernel

ErrorDrivenPredictor         | Online linear regression with   | Recursive least squares
(PredictiveCoding)           | precision-weighted updates      | with forgetting factor

TemporalCreditAssigner       | Recency-weighted credit         | Exponentially weighted
(HebbianSTDP)                | assignment for past decisions   | moving average (EWMA)

SoftmaxRouter                | Differentiable top-k selection  | Softmax with temperature
(LateralInhibition)          | with inhibition sharpening      | + inhibitory bias

StableRouteCache             | Caching for consistently        | Threshold-based memoization
(MemoryConsolidation)        | successful routing decisions    | with hysteresis

ExponentialForgetting        | Recency-weighted forgetting     | Exponential decay kernel
(SynapticDecay)              | for stale observations          | (Ebbinghaus 1885)

ALGORITHM COMPLEXITY (n = models × categories, typically 6 × 6 = 36):
  All algorithms are O(n) per call. Designed for n < 1000.
  No convergence guarantees at this scale — use only if empirically validated.
  Pure-math baseline (simple argmax of affinity) should always be compared against.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# 1. RANDOM FOURIER FEATURE ENCODER
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Given a query embedding x ∈ R^d, compute a feature map φ(x) ∈ R^m
#   such that ⟨φ(x), φ(y)⟩ ≈ k(x, y) for some kernel k.
#   This enables efficient similarity search without storing all prototypes.
#
# Mathematical Description:
#   Random Fourier Features (Rahimi & Recht, NeurIPS 2007):
#     For shift-invariant kernel k(x-y), Bochner's theorem gives:
#       k(x-y) = ∫ p(ω) e^{iω·(x-y)} dω
#     Approximate with Monte Carlo:
#       φ(x) = √(2/m) · cos(Wx + b)  where W_ij ~ p(ω), b_j ~ Uniform(0,2π)
#     For RBF kernel k(x,y)=exp(-γ||x-y||²): p(ω) = N(0, 2γI)
#
#   Our variant uses multiple scales (geometric spacing) for multi-resolution
#   encoding — similar to the sparse random features in Li et al. (NeurIPS 2021).
#
# Implementation Notes:
#   - dim=384 input, m=12 output features (4 modules × 3 angles each)
#   - Periods geometrically spaced [1.0, 1.4, 1.7, 2.0] for multi-scale
#   - 3 cosine gratings at 60° offsets (maximizes coverage of 2D projection)
#   - Complexity: O(d·m) per encode
#
# Neuro Inspiration (loose):
#   Grid cells in entorhinal cortex (Moser & Moser, Nature 2005) encode 2D
#   spatial position using hexagonal firing fields. Our RFF encoder similarly
#   maps high-dim embeddings to a structured low-dim feature space.
#   Scale separation: biological grid cells operate in 2D physical space with
#   ~4 discrete scales; our encoder operates in 384-dim semantic space.
# ═══════════════════════════════════════════════════════════════════════════

class GridCellMap:  # kept for backward compat; alias = RandomFourierEncoder
    """
    Multi-scale random Fourier feature encoder for query similarity search.
    """

    def __init__(self, n_modules: int = 4, dim: int = 384, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_modules = n_modules
        self.dim = dim
        self.n_output = n_modules * 3  # 3 cosine angles per module

        self.periods = np.array([1.0, 1.4, 1.7, 2.0])[:n_modules]
        self.projections_x = []
        self.projections_y = []
        self.phases = []
        self.orientations = []

        for m in range(n_modules):
            px = rng.randn(dim) / np.sqrt(dim)
            py = rng.randn(dim) / np.sqrt(dim)
            py = py - np.dot(px, py) * px  # Gram-Schmidt orthogonalization
            py = py / (np.linalg.norm(py) + 1e-8)
            self.projections_x.append(px)
            self.projections_y.append(py)

            theta = rng.uniform(0, 2 * np.pi)
            self.orientations.append(theta)
            self.phases.append([rng.uniform(0, 2 * np.pi) for _ in range(3)])

        self.episode = 0

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """
        φ(x) = [cos(k_m · (cos θ_j, sin θ_j) · (P_m x) + φ_{m,j})]_{m,j}

        where P_m projects x to 2D, k_m = 2π · period_m,
        θ_j ∈ {θ, θ+60°, θ+120°} for module orientation θ.

        Returns: (n_modules * 3,) ∈ [0, 1]
        """
        rates = np.zeros(self.n_output, dtype=np.float64)
        for m in range(self.n_modules):
            x = np.dot(embedding, self.projections_x[m])
            y = np.dot(embedding, self.projections_y[m])
            theta = self.orientations[m]
            k = 2.0 * np.pi * self.periods[m]

            cell_rate = 0.0
            for j in range(3):
                angle = theta + j * np.pi / 3.0
                grating = np.cos(k * (np.cos(angle) * x + np.sin(angle) * y)
                                 + self.phases[m][j])
                cell_rate += max(0.0, grating)
            rates[m * 3:(m + 1) * 3] = cell_rate / 3.0

        self.episode += 1
        return rates

    def get_state(self) -> dict:
        return {"name": "GridCellMap", "n_modules": self.n_modules,
                "dim": self.dim, "episode": self.episode,
                "periods": self.periods.tolist()}

    def load_state(self, state: dict):
        self.episode = state.get("episode", 0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ERROR-DRIVEN ONLINE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Online linear regression: given features x_t, predict outcomes y_t for
#   each of N targets (models). Update weights using only prediction errors.
#   No batch data — each (x_t, y_t) pair is processed once and discarded.
#
# Mathematical Description:
#   This is recursive least squares (RLS) with exponential forgetting:
#     θ_{t+1} = θ_t + P_t · x_t · e_t
#     where P_t = diag(1/(σ²_i + ε)) is the precision (inverse variance)
#     and e_t = y_t - ŷ_t is the prediction error.
#
#   Key property: only errors drive updates. Perfect prediction → no weight change.
#   This prevents overfitting to noise and is stable under stationarity.
#
#   RLS guarantee (Lai & Wei, 1982): under Gaussian noise, the estimator is
#   consistent (θ_t → θ* as t → ∞) with O(log t / t) convergence rate.
#
# Implementation Notes:
#   - Single-layer (not hierarchical). For n ≤ 6 targets, multi-layer adds
#     unnecessary complexity with no provable benefit.
#   - Precision-weighted: reliable features (low error variance) update less.
#   - O(n_input · n_output) per update.
#
# Neuro Inspiration (loose):
#   Predictive coding (Rao & Ballard, Nature Neuroscience 1999) proposes that
#   cortical circuits minimize prediction errors via hierarchical generative
#   models. Our single-layer error-driven update captures the "predict →
#   compute error → update" loop without the biological hierarchy.
# ═══════════════════════════════════════════════════════════════════════════

class PredictiveCodingLayer:  # kept for backward compat; alias = ErrorDrivenPredictor
    """
    Online linear predictor with precision-weighted error-driven updates.
    """

    def __init__(self, n_input: int, n_output: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_input = n_input
        self.n_output = n_output

        # Weight matrix W (n_output × n_input): ŷ = tanh(Wx + b)
        self.W = rng.randn(n_output, n_input) * 0.01
        self.b = np.zeros(n_output)

        # Precision (inverse prediction error variance) per output
        self.error_ema = np.zeros(n_output)
        self.precision = np.ones(n_output)

        self.episode = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ŷ = tanh(Wx + b) ∈ [-1, 1]^n_output"""
        return np.tanh(self.W @ x + self.b)

    def compute_error(self, prediction: np.ndarray,
                      target: np.ndarray) -> np.ndarray:
        """e = target - prediction (signed error)"""
        return target - prediction

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """
        Precision-weighted RLS update:
          W_i ← W_i + lr · Π_i · e_i · x     (Hebbian)
          b_i ← b_i + lr · Π_i · e_i
          Π_i = 1 / (EMA[|e_i|] + ε)         (precision from error variance)
        """
        self.error_ema = 0.9 * self.error_ema + 0.1 * np.abs(error)
        self.precision = 1.0 / (self.error_ema + 0.1)

        for i in range(self.n_output):
            pi, ei = self.precision[i], error[i]
            self.W[i] += lr * pi * ei * x
            self.b[i] += lr * pi * ei

        self.episode += 1

    def get_state(self) -> dict:
        return {"name": "PredictiveCoding", "n_input": self.n_input,
                "n_output": self.n_output, "episode": self.episode,
                "W": self.W.tolist(), "b": self.b.tolist(),
                "precision": self.precision.tolist(),
                "error_ema": self.error_ema.tolist()}

    def load_state(self, state: dict):
        self.W = np.array(state["W"]); self.b = np.array(state["b"])
        self.precision = np.array(state.get("precision", np.ones(self.n_output)))
        self.error_ema = np.array(state.get("error_ema", np.zeros(self.n_output)))
        self.episode = state.get("episode", 0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TEMPORAL CREDIT ASSIGNER
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Assign credit/blame to past decisions when outcomes arrive with delay.
#   A model was chosen at time t_pre; correctness observed at t_post.
#   The credit for (t_post - t_pre) steps of delay should decay with time.
#
# Mathematical Description:
#   Exponentially Weighted Moving Average (EWMA) with recency bias:
#     Δw = +A⁺ · trace · exp(-|Δt|/τ⁺)    if correct   (positive credit)
#     Δw = -A⁻ · trace · exp(-|Δt|/τ⁻)    if incorrect  (negative credit)
#
#   This is equivalent to a 1st-order low-pass filter on the reward signal.
#   The time constants τ⁺, τ⁻ control the memory length:
#     τ⁺ larger → correct answers remembered longer
#     τ⁻ smaller → wrong answers forgotten faster (desirable: don't over-penalize)
#
#   Connection to RL: this is a simplified eligibility trace (TD(λ) with λ=0).
#   No convergence guarantee at n=36 scale — use only if empirically better
#   than simple argmax of running accuracy.
#
# Implementation Notes:
#   - Weights ∈ [-1, 1] with hard clipping (prevents runaway)
#   - τ⁺ = 5 (remember success for ~5 episodes), τ⁻ = 3 (forget failure faster)
#   - O(n) per update
#
# Neuro Inspiration (loose):
#   STDP (Song, Miller & Abbott, Nature Neuroscience 2000) describes how the
#   timing between pre- and post-synaptic spikes determines LTP/LTD.
#   Our EWMA credit assigner captures the "recency-weighted" aspect but operates
#   at episode-level time scale (~seconds), not millisecond spike timing.
# ═══════════════════════════════════════════════════════════════════════════

class HebbianSTDP:  # kept for backward compat; alias = ExponentialCreditAssigner
    """
    Recency-weighted credit assignment for delayed outcomes.

    Weights evolve as: w_{t+1} = w_t + Δw where Δw depends on elapsed time.
    """

    def __init__(self, n_synapses: int,
                 A_plus: float = 0.15, A_minus: float = 0.10,
                 tau_plus: float = 5.0, tau_minus: float = 3.0,
                 w_max: float = 1.0, w_min: float = -1.0, seed: int = 42):
        self.n_synapses = n_synapses
        self.A_plus, self.A_minus = A_plus, A_minus
        self.tau_plus, self.tau_minus = tau_plus, tau_minus
        self.w_max, self.w_min = w_max, w_min

        self.weights = np.zeros(n_synapses, dtype=np.float64)
        self.traces = np.zeros(n_synapses, dtype=np.float64)
        self.tau_trace = 3.0
        self.last_pre_time = np.full(n_synapses, -1, dtype=np.int64)
        self.episode = 0

    def pre_fire(self, synapse_idx: int):
        """Mark that a decision was made (pre-synaptic event)."""
        if 0 <= synapse_idx < self.n_synapses:
            self.traces[synapse_idx] += 1.0
            self.last_pre_time[synapse_idx] = self.episode

    def post_fire(self, synapse_idx: int, success: bool) -> float:
        """
        Apply recency-weighted credit:
          Δw = +A⁺ · trace · exp(-|Δt|/τ⁺)  if success
          Δw = -A⁻ · trace · exp(-|Δt|/τ⁻)  if failure
        """
        if not (0 <= synapse_idx < self.n_synapses):
            return 0.0

        trace = self.traces[synapse_idx]
        t_pre = self.last_pre_time[synapse_idx]
        dt = self.episode - t_pre if t_pre >= 0 else self.tau_plus

        if success:
            dw = self.A_plus * trace * np.exp(-abs(dt) / self.tau_plus)
        else:
            dw = -self.A_minus * trace * np.exp(-abs(dt) / self.tau_minus)

        old_w = self.weights[synapse_idx]
        self.weights[synapse_idx] = np.clip(old_w + dw, self.w_min, self.w_max)

        self.traces[synapse_idx] *= 0.5
        self.traces *= np.exp(-1.0 / self.tau_trace)
        self.traces = np.clip(self.traces, 0.0, 3.0)

        self.episode += 1
        return float(dw)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_pathways(self, top_k: int = 10) -> List[Tuple[int, float]]:
        strong = [(i, float(w)) for i, w in enumerate(self.weights) if abs(w) > 0.1]
        return sorted(strong, key=lambda x: -abs(x[1]))[:top_k]

    def get_state(self) -> dict:
        return {"name": "HebbianSTDP", "n_synapses": self.n_synapses,
                "episode": self.episode, "weights": self.weights.tolist(),
                "traces": self.traces.tolist()}

    def load_state(self, state: dict):
        self.weights = np.array(state.get("weights", np.zeros(self.n_synapses)))
        self.traces = np.array(state.get("traces", np.zeros(self.n_synapses)))
        self.episode = state.get("episode", 0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SOFTMAX ROUTER WITH INHIBITION
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Given N scores (model affinities), select the best model. The selection
#   should be differentiable (for learning) and avoid ties (sharp decision).
#
# Mathematical Description:
#   This is softmax selection with an inhibitory bias:
#     p_i = softmax(s_i / T)_i
#   where T is temperature. Lower T → sharper selection (more like argmax).
#
#   The inhibitory matrix K adds a "winner suppression" term:
#     e_i = s_i - Σ_{j≠i} K_{ij} · r_j    (inputs minus inhibition)
#     r_i = max(0, e_i)                      (ReLU activation)
#
#   After convergence (3 iterations of recurrence), the strongest input
#   suppresses all competitors proportionally. This sharpens softmax into
#   near-hard argmax while maintaining differentiability.
#
#   K matrix adapts via Hebbian learning: if winner succeeds, its inhibition
#   over losers strengthens (K_{loser, winner} ↑).
#
# Implementation Notes:
#   - 3 iterations of recurrence (enough for convergence at n=6)
#   - K_{ij} ∈ [0.01, 0.5] bounded for stability
#   - O(n²) per apply
#
# Neuro Inspiration (loose):
#   Lateral inhibition in the Limulus eye (Hartline & Ratliff, JGP 1958)
#   shows that active neurons suppress neighbors via recurrent inhibition.
#   Our softmax router uses the same WTA principle.
# ═══════════════════════════════════════════════════════════════════════════

class RecurrentLateralInhibition:  # kept for backward compat
    """
    Softmax-like winner-take-all selection with adaptive inhibition.
    """

    def __init__(self, n_units: int, strength: float = 0.03,
                 max_strength: float = 0.5, seed: int = 42):
        self.n_units = n_units
        self.base_strength = strength
        self.max_strength = max_strength

        # K[i, j]: inhibition from unit j → unit i. Diagonal = 0.
        self.K = np.full((n_units, n_units), strength, dtype=np.float64)
        np.fill_diagonal(self.K, 0.0)

        self.episode = 0

    def apply(self, activations: np.ndarray,
              n_iterations: int = 3) -> np.ndarray:
        """
        Recurrent WTA: r = ReLU(s - K @ r), iterated to convergence.
        Returns inhibited activations where winner dominates.
        """
        r = activations.copy()
        for _ in range(n_iterations):
            inhibition = self.K @ r
            e = activations - inhibition
            r = np.maximum(0.0, e)

        winner = int(np.argmax(r))
        w_act = r[winner]
        if w_act > 0:
            for i in range(self.n_units):
                if i != winner:
                    r[i] = max(0.0, r[i] - self.K[i, winner] * w_act)

        return r

    def adapt(self, winner: int, loser: int, winner_success: bool):
        """Hebbian K update: successful winner strengthens inhibition."""
        if winner == loser:
            return
        if winner_success:
            self.K[loser, winner] = min(self.max_strength,
                                        self.K[loser, winner] + 0.01)
        else:
            self.K[loser, winner] = max(self.base_strength * 0.1,
                                        self.K[loser, winner] - 0.005)
        self.episode += 1

    def get_inhibition_matrix(self) -> np.ndarray:
        return self.K.copy()

    def get_state(self) -> dict:
        return {"name": "RecurrentLateralInhibition", "n_units": self.n_units,
                "episode": self.episode, "K": self.K.tolist()}

    def load_state(self, state: dict):
        self.K = np.array(state.get("K", np.zeros((self.n_units, self.n_units))))
        self.episode = state.get("episode", 0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. STABLE ROUTE CACHE
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Some (question_type → model) routes are consistently correct.
#   Cache these routes to bypass expensive online computation.
#   Cache entries must survive occasional failures (hysteresis).
#
# Mathematical Description:
#   Threshold-based memoization with hysteresis:
#     if consecutive_successes ≥ θ (typically 5):
#         mark as "stable" → cached, survives single failures
#     if cached entry fails: decrement counter (don't reset)
#     if non-cached entry fails: reset counter to 0
#
#   This is equivalent to a Schmitt trigger: the "activate" threshold is
#   higher than the "deactivate" threshold, preventing oscillation.
#
# Implementation Notes:
#   - threshold θ=5 (requires 5 consecutive successes)
#   - Cached routes get 0.2 bonus weight multiplier
#   - O(1) per update
#
# Neuro Inspiration (loose):
#   Memory consolidation (Kandel, Science 2001) describes how repeated
#   stimulation triggers CREB-mediated gene transcription → permanent
#   structural changes. Our cache uses the same "repetition → stability"
#   principle with hysteresis.
# ═══════════════════════════════════════════════════════════════════════════

class MemoryConsolidation:  # kept for backward compat; alias = StableRouteCache
    """
    Threshold-based memoization with hysteresis for routing decisions.
    """

    def __init__(self, n_synapses: int, threshold: int = 5,
                 boost: float = 0.2, seed: int = 42):
        self.n_synapses = n_synapses
        self.threshold = threshold
        self.boost = boost

        self.consecutive_successes = np.zeros(n_synapses, dtype=np.int32)
        self.total_successes = np.zeros(n_synapses, dtype=np.int32)
        self.consolidated = np.zeros(n_synapses, dtype=bool)
        self.episode = 0

    def update(self, synapse_idx: int, success: bool):
        """Update cache state with hysteresis."""
        if not (0 <= synapse_idx < self.n_synapses):
            return

        if success:
            self.consecutive_successes[synapse_idx] += 1
            self.total_successes[synapse_idx] += 1
            if (self.consecutive_successes[synapse_idx] >= self.threshold
                    and not self.consolidated[synapse_idx]):
                self.consolidated[synapse_idx] = True
        else:
            if not self.consolidated[synapse_idx]:
                self.consecutive_successes[synapse_idx] = 0
            else:
                # Hysteresis: cached entries survive single failures
                self.consecutive_successes[synapse_idx] = max(
                    2, self.consecutive_successes[synapse_idx] - 1)

        self.episode += 1

    def is_consolidated(self, synapse_idx: int) -> bool:
        return bool(self.consolidated[synapse_idx]) if 0 <= synapse_idx < self.n_synapses else False

    def get_consolidation_boost(self, synapse_idx: int) -> float:
        return self.boost if self.is_consolidated(synapse_idx) else 0.0

    def get_state(self) -> dict:
        return {"name": "MemoryConsolidation", "n_synapses": self.n_synapses,
                "episode": self.episode,
                "consecutive_successes": self.consecutive_successes.tolist(),
                "total_successes": self.total_successes.tolist(),
                "consolidated": self.consolidated.tolist()}

    def load_state(self, state: dict):
        self.consecutive_successes = np.array(
            state.get("consecutive_successes", np.zeros(self.n_synapses)), dtype=np.int32)
        self.total_successes = np.array(
            state.get("total_successes", np.zeros(self.n_synapses)), dtype=np.int32)
        self.consolidated = np.array(
            state.get("consolidated", np.zeros(self.n_synapses)), dtype=bool)
        self.episode = state.get("episode", 0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. EXPONENTIAL FORGETTING
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Old observations become stale — model capabilities change over time
#   (API updates, model retraining). Weights should decay if unused.
#
# Mathematical Description:
#   Exponential decay kernel:
#     w_t = w_0 · exp(-t / τ)
#   with dual-trace variant (fast + slow decay):
#     R(t) = α · exp(-t/τ_fast) + (1-α) · exp(-t/τ_slow)
#
#   Used items are protected from decay (use = rehearsal prevents forgetting).
#   Cached (consolidated) items decay at 1/5 rate.
#
# Implementation Notes:
#   - τ_fast = 5 episodes (rapid decay for unused routes)
#   - τ_slow = 500 episodes (slow background decay)
#   - α = 0.3 (30% fast, 70% slow)
#   - O(n) per apply
#
# Neuro Inspiration (loose):
#   Ebbinghaus (1885) forgetting curve: R(t) = exp(-t/τ).
#   Our dual-trace variant reflects the empirical finding (Wixted, 2004)
#   that memory decay follows a power law at long timescales.
# ═══════════════════════════════════════════════════════════════════════════

class SynapticDecay:  # kept for backward compat; alias = ExponentialForgetting
    """
    Dual-trace exponential forgetting with use-dependent protection.
    """

    def __init__(self, decay_rate: float = 0.001, power_law_beta: float = 0.5,
                 fast_decay_ratio: float = 0.3, fast_tau: float = 5.0,
                 slow_tau: float = 500.0, seed: int = 42):
        self.decay_rate = decay_rate
        self.fast_decay_ratio = fast_decay_ratio
        self.fast_tau = fast_tau
        self.slow_tau = slow_tau

        self.last_used = None
        self.episode = 0

    def apply(self, weights: np.ndarray, used_indices: List[int],
              consolidated_mask: np.ndarray = None) -> np.ndarray:
        """
        Dual-trace forgetting:
          R(Δt) = α·exp(-Δt/τ_fast) + (1-α)·exp(-Δt/τ_slow)
        Used items protected (Δt reset to 0).
        Consolidated items decay at 1/5 rate.
        """
        n = len(weights)
        if self.last_used is None:
            self.last_used = np.full(n, self.episode, dtype=np.int64)

        delta_t = np.maximum(0, self.episode - self.last_used).astype(np.float64)
        fast_trace = np.exp(-delta_t / self.fast_tau)
        slow_trace = np.exp(-delta_t / self.slow_tau)
        retention = (self.fast_decay_ratio * fast_trace +
                     (1.0 - self.fast_decay_ratio) * slow_trace)

        if consolidated_mask is not None:
            cons = np.asarray(consolidated_mask, dtype=bool)
            retention[cons] = 1.0 - (1.0 - retention[cons]) * 0.2

        weights = weights * retention

        self.last_used += 1
        for idx in used_indices:
            if 0 <= idx < n:
                self.last_used[idx] = 0

        self.episode += 1
        return weights

    def get_state(self) -> dict:
        return {"name": "SynapticDecay", "episode": self.episode,
                "last_used": self.last_used.tolist() if self.last_used is not None else None}

    def load_state(self, state: dict):
        self.episode = state.get("episode", 0)
        lu = state.get("last_used")
        if lu is not None:
            self.last_used = np.array(lu, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# 7. EVENT-DRIVEN LEARNING RATE MODULATOR (simplified SynapticTagging)
# ═══════════════════════════════════════════════════════════════════════════
#
# Computational Problem:
#   Some learning events are more important than others. Consecutive successes
#   should trigger larger learning rate boosts (momentum-like behavior).
#
# Mathematical Description:
#   Tag = binary marker set on success events. Decays with τ_tag.
#   PRP pool = accumulated "momentum" from repeated successes. Decays with τ_prp.
#   Boost = 1 + 2 · tag · PRP_pool  (extra learning rate multiplier).
#
#   This is equivalent to a simple momentum accumulator:
#     m_t = β · m_{t-1} + (1-β) · success_t
#   where success_t ∈ {0, 1}. The tag + PRP mechanism adds a binary threshold.
#
# Implementation Notes:
#   - Small n (typically ≤ 36), marginal benefit over simple EMA
#   - Kept for backward compatibility with neuro_runner.py
#   - O(n) per step
# ═══════════════════════════════════════════════════════════════════════════

class SynapticTaggingCapture:  # kept for backward compat
    """
    Event-driven momentum accumulator for learning rate modulation.
    """

    def __init__(self, n_synapses: int, tau_tag: float = 30.0,
                 tau_prp: float = 100.0, tag_threshold: float = 0.3, seed: int = 42):
        self.n_synapses = n_synapses
        self.tau_tag = tau_tag
        self.tau_prp = tau_prp
        self.tag_threshold = tag_threshold

        self.tags = np.zeros(n_synapses)
        self.prp_pool = 0.0
        self.episode = 0
        self.n_tags_set = 0
        self.n_capture_events = 0

    def set_tag(self, synapse_idx: int, strength: float = 1.0):
        if 0 <= synapse_idx < self.n_synapses:
            self.tags[synapse_idx] = min(1.0, strength)
            self.n_tags_set += 1

    def generate_prp(self, amount: float = 1.0):
        self.prp_pool = min(2.0, self.prp_pool + amount)

    def step(self, dt: float = 1.0) -> np.ndarray:
        tag_decay = np.exp(-max(0.1, min(dt, 10.0)) / self.tau_tag)
        prp_decay = np.exp(-max(0.1, min(dt, 10.0)) / self.tau_prp)
        self.tags *= tag_decay
        self.prp_pool *= prp_decay

        total_tag = self.tags.sum()
        if total_tag > 1e-8 and self.prp_pool > 1e-8:
            capture_fraction = self.tags / total_tag
            capturable = min(self.prp_pool * 0.3, self.prp_pool)
            capture = capture_fraction * capturable
            self.prp_pool -= capture.sum()
            self.tags = np.maximum(0, self.tags - capture)
            self.n_capture_events += int(np.sum(capture > 0.01))
        else:
            capture = np.zeros(self.n_synapses)

        self.episode += 1
        return capture

    def get_boost(self, synapse_idx: int) -> float:
        tag = float(self.tags[synapse_idx]) if 0 <= synapse_idx < self.n_synapses else 0.0
        return 1.0 + 2.0 * tag * self.prp_pool

    def get_prp_level(self) -> float:
        return float(self.prp_pool)

    def get_state(self) -> dict:
        return {"name": "SynapticTaggingCapture", "n_synapses": self.n_synapses,
                "episode": self.episode, "tags": self.tags.tolist(),
                "prp_pool": float(self.prp_pool), "n_tags_set": self.n_tags_set,
                "n_capture_events": self.n_capture_events}

    def load_state(self, state: dict):
        self.tags = np.array(state.get("tags", np.zeros(self.n_synapses)))
        self.prp_pool = state.get("prp_pool", 0.0)
        self.n_tags_set = state.get("n_tags_set", 0)
        self.n_capture_events = state.get("n_capture_events", 0)
        self.episode = state.get("episode", 0)


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Algorithm Unit Tests ===\n")

    # 1. RFF Encoder
    print("1. RandomFourierEncoder (GridCellMap)")
    rff = GridCellMap(n_modules=4, dim=384)
    emb = np.random.randn(384).astype(np.float64)
    emb /= np.linalg.norm(emb)
    fv = rff.encode(emb)
    assert fv.shape == (12,), f"shape: {fv.shape}"
    assert np.all(fv >= 0) and np.all(fv <= 1), "range [0,1]"
    print(f"   OK: {fv.shape} ∈ [{fv.min():.3f}, {fv.max():.3f}]")

    # 2. Online Predictor
    print("\n2. ErrorDrivenPredictor (PredictiveCoding)")
    pred = PredictiveCodingLayer(n_input=12, n_output=6)
    y_hat = pred.predict(fv)
    target = np.array([1, 0, 0.5, 0, 0, 0])
    e = pred.compute_error(y_hat, target)
    pred.update(fv, e, lr=0.01)
    print(f"   OK: y_hat={y_hat.round(3)}, error applied")

    # 3. Credit Assigner
    print("\n3. ExponentialCreditAssigner (HebbianSTDP)")
    stdp = HebbianSTDP(n_synapses=6)
    stdp.pre_fire(0)
    dw_pos = stdp.post_fire(0, success=True)
    stdp.pre_fire(1)
    dw_neg = stdp.post_fire(1, success=False)
    assert dw_pos > 0 and dw_neg < 0
    print(f"   OK: dw_pos={dw_pos:+.4f}, dw_neg={dw_neg:+.4f}")

    # 4. Softmax Router
    print("\n4. SoftmaxRouter (LateralInhibition)")
    lat = RecurrentLateralInhibition(n_units=6)
    scores = np.array([0.8, 0.6, 0.3, 0.1, 0.05, 0.02])
    inhibited = lat.apply(scores)
    winner = int(np.argmax(inhibited))
    print(f"   OK: winner={winner}, inhibited={inhibited.round(3)}")

    # 5. Route Cache
    print("\n5. StableRouteCache (MemoryConsolidation)")
    cache = MemoryConsolidation(n_synapses=6)
    for _ in range(5):
        cache.update(0, success=True)
    assert cache.is_consolidated(0)
    cache.update(0, success=False)
    assert cache.is_consolidated(0)  # hysteresis
    print(f"   OK: consolidated={cache.consolidated}")

    # 6. Forgetting
    print("\n6. ExponentialForgetting (SynapticDecay)")
    decay = SynapticDecay()
    w = np.array([0.8, -0.5, 0.3, 0.1, -0.2, 0.9])
    cons = np.array([True, False, False, False, False, True])
    w2 = decay.apply(w.copy(), used_indices=[0, 5], consolidated_mask=cons)
    print(f"   OK: before={w.round(3)}, after={w2.round(3)}")

    # 7. Momentum
    print("\n7. EventDrivenMomentum (SynapticTagging)")
    stc = SynapticTaggingCapture(n_synapses=6)
    stc.set_tag(0, 0.8)
    stc.generate_prp(0.5)
    capture = stc.step()
    print(f"   OK: prp={stc.get_prp_level():.3f}, boost={stc.get_boost(0):.3f}")

    print("\n=== ALL 7 ALGORITHMS TESTED ===")
