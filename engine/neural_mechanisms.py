#!/usr/bin/env python3
"""
neural_mechanisms.py — 7 Independent Neural Mechanisms, Each Anchored to a Paper

Each mechanism is a standalone module that can be composed into routing systems.
All mechanisms read from and write to a shared NeuroState (synaptic state vector).

MECHANISM           | PAPER                                    | FUNCTION
────────────────────┼──────────────────────────────────────────┼─────────────────────────
GridCellMap         | Hafting, Moser & Moser (2005) Nature     | Question → spatial position
PredictiveCoding    | Rao & Ballard (1999) Nature Neuroscience | Predict success, error-driven
SynapticTagging     | Frey & Morris (1997) Nature              | Tag important events
HebbianSTDP         | Song, Miller & Abbott (2000) Nat Neurosci| Wire together, fire together
LateralInhibition   | Hartline & Ratliff (1958) J Gen Physiol  | Winner suppresses others
MemoryConsolidation | Kandel (2001) Science (Nobel)            | L-LTP for repeated success
SynapticDecay       | Ebbinghaus (1885) Memory                 | Forgetting curve
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# 1. GRID CELL COGNITIVE MAP (Hafting, Fyhn, Molden, Moser & Moser 2005)
# ═══════════════════════════════════════════════════════════════

class GridCellMap:
    """
    Hexagonal grid cell population encoding question position in cognitive space.
    Ref: Hafting et al. (2005) "Microstructure of a spatial map in the entorhinal cortex."
    """

    def __init__(self, n_modules: int = 4, dim: int = 384, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.modules = []
        frequencies = [0.5, 1.0, 2.0, 4.0]  # Doubling (Moser 2005)
        for f in frequencies[:n_modules]:
            for theta in [0.0, np.pi/3, 2*np.pi/3]:
                proj_x = rng.randn(dim) / np.sqrt(dim)
                proj_y = rng.randn(dim) / np.sqrt(dim)
                phi = rng.rand() * 2 * np.pi
                self.modules.append((f, theta, phi, proj_x, proj_y))

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        rates = []
        for f, theta, phi, px, py in self.modules:
            x, y = np.dot(embedding, px), np.dot(embedding, py)
            rate = 0.0
            for a in [theta, theta + np.pi/3, theta + 2*np.pi/3]:
                rate += np.cos(2.0 * np.pi * f * (np.cos(a)*x + np.sin(a)*y) + phi)
            rates.append(max(0.0, rate / 3.0))
        return np.array(rates)


# ═══════════════════════════════════════════════════════════════
# 2. PREDICTIVE CODING (Rao & Ballard 1999)
# ═══════════════════════════════════════════════════════════════

class PredictiveCodingLayer:
    """
    Top-down prediction + bottom-up error propagation.
    Ref: Rao & Ballard (1999) "Predictive coding in the visual cortex."
    """

    def __init__(self, n_input: int, n_output: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_output, n_input) * 0.01
        self.b = np.zeros(n_output)
        self.precision = np.ones(n_output)
        self.error_ema = np.zeros(n_output)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.W @ x + self.b

    def update(self, x: np.ndarray, target: np.ndarray, idx: int, lr: float = 0.01):
        pred = self.predict(x)
        error = target - pred
        self.error_ema = 0.9 * self.error_ema + 0.1 * np.abs(error)
        self.precision = 1.0 / (self.error_ema + 0.1)
        p = self.precision[idx]; e = error[idx]
        self.W[idx] += lr * p * e * x
        self.b[idx] += lr * p * e


# ═══════════════════════════════════════════════════════════════
# 3. SYNAPTIC TAGGING & CAPTURE (Frey & Morris 1997)
# ═══════════════════════════════════════════════════════════════

class SynapticTagging:
    """
    Strong stimuli set tags; nearby synapses capture PRPs for rapid consolidation.
    Ref: Frey & Morris (1997) "Synaptic tagging and long-term potentiation."
    """

    def __init__(self, n_synapses: int, tag_threshold: float = 0.6, lifetime: int = 5):
        self.tags = np.zeros(n_synapses)
        self.threshold = tag_threshold
        self.lifetime = lifetime
        self.prp_pool = 1.0

    def update(self, error: np.ndarray):
        """Large prediction errors set tags; tags enable PRP capture."""
        for i in range(len(self.tags)):
            if abs(error[i]) > self.threshold:
                self.tags[i] = self.lifetime  # Set tag
            else:
                self.tags[i] = max(0, self.tags[i] - 1)  # Decay
        # PRP pool replenished by strong LTP events
        self.prp_pool = min(2.0, self.prp_pool + 0.1 * np.sum(error > self.threshold))

    def get_boost(self, idx: int) -> float:
        """Tagged synapse gets learning rate boost from PRP pool."""
        return 1.0 + 2.0 * (self.tags[idx] > 0) * self.prp_pool


# ═══════════════════════════════════════════════════════════════
# 4. HEBBIAN STDP (Song, Miller & Abbott 2000)
# ═══════════════════════════════════════════════════════════════

class HebbianSTDP:
    """
    Spike-Timing-Dependent Plasticity.
    Ref: Song, Miller & Abbott (2000) "Competitive Hebbian learning through STDP."
    """

    def __init__(self, n_synapses: int, A_plus: float = 0.15, A_minus: float = 0.10,
                 tau_plus: float = 5.0, tau_minus: float = 3.0):
        self.weights = np.zeros(n_synapses)
        self.traces = np.zeros(n_synapses)  # Pre-synaptic traces
        self.A_plus, self.A_minus = A_plus, A_minus
        self.tau_plus, self.tau_minus = tau_plus, tau_minus
        self.last_spike = -1

    def pre_fire(self, idx: int):
        """Pre-synaptic neuron fires → set eligibility trace."""
        self.traces[idx] += 1.0
        self.last_spike = idx

    def post_fire(self, idx: int, success: bool):
        """Post-synaptic neuron fires → STDP weight update."""
        trace = self.traces[idx]
        dt = 1.0
        if success and self.last_spike >= 0:
            dw = self.A_plus * trace * np.exp(-dt / self.tau_plus)
            self.weights[idx] += dw
        elif not success:
            dw = -self.A_minus * trace * np.exp(-dt / self.tau_minus)
            self.weights[idx] += dw
        self.traces *= np.exp(-1.0 / 3.0)  # Trace decay
        self.weights = np.clip(self.weights, -1.0, 1.0)

    def get_pathway_strength(self) -> List[Tuple[int, float]]:
        strong = [(i, w) for i, w in enumerate(self.weights) if abs(w) > 0.2]
        return sorted(strong, key=lambda x: -abs(x[1]))


# ═══════════════════════════════════════════════════════════════
# 5. LATERAL INHIBITION (Hartline & Ratliff 1958)
# ═══════════════════════════════════════════════════════════════

class LateralInhibition:
    """
    Winner-take-all: active unit suppresses neighbors.
    Ref: Hartline & Ratliff (1958) "Inhibitory interaction in the Limulus eye."
    """

    def __init__(self, n_units: int, strength: float = 0.03):
        self.strength = strength
        self.inhibition = np.ones((n_units, n_units)) * strength
        np.fill_diagonal(self.inhibition, 0.0)

    def apply(self, activations: np.ndarray, winner: int) -> np.ndarray:
        """Winner suppresses all others proportionally to their activation."""
        inhibited = activations.copy()
        for i in range(len(activations)):
            if i != winner:
                inhibited[i] -= self.strength * activations[winner]
        return np.maximum(0, inhibited)

    def adapt(self, winner: int, loser: int, winner_success: bool):
        """Strengthen inhibition from winner to loser if winner succeeded."""
        if winner_success:
            self.inhibition[winner, loser] = min(0.5, self.inhibition[winner, loser] + 0.01)
        else:
            self.inhibition[winner, loser] = max(0.01, self.inhibition[winner, loser] - 0.005)


# ═══════════════════════════════════════════════════════════════
# 6. MEMORY CONSOLIDATION — L-LTP (Kandel 2001)
# ═══════════════════════════════════════════════════════════════

class MemoryConsolidation:
    """
    Late-phase LTP: repeated success → permanent synaptic consolidation.
    Ref: Kandel (2001) "The molecular biology of memory storage."
    """

    def __init__(self, n_synapses: int, threshold: int = 5, boost: float = 0.2):
        self.consecutive_successes = np.zeros(n_synapses)
        self.consolidated = np.zeros(n_synapses, dtype=bool)
        self.threshold = threshold
        self.boost = boost

    def update(self, idx: int, success: bool):
        if success:
            self.consecutive_successes[idx] += 1
            if self.consecutive_successes[idx] >= self.threshold:
                self.consolidated[idx] = True  # Permanent!
        else:
            self.consecutive_successes[idx] = 0

    def is_consolidated(self, idx: int) -> bool:
        return bool(self.consolidated[idx])


# ═══════════════════════════════════════════════════════════════
# 7. SYNAPTIC DECAY (Ebbinghaus 1885)
# ═══════════════════════════════════════════════════════════════

class SynapticDecay:
    """
    Exponential forgetting curve.
    Ref: Ebbinghaus (1885) "Memory: A Contribution to Experimental Psychology."
    """

    def __init__(self, decay_rate: float = 0.001):
        self.decay_rate = decay_rate
        self.time_since_use = None

    def apply(self, weights: np.ndarray, used_indices: List[int]) -> np.ndarray:
        """All weights decay toward zero. Used synapses are protected."""
        weights = weights * (1.0 - self.decay_rate)
        for idx in used_indices:
            weights[idx] = weights[idx] / (1.0 - self.decay_rate)  # Undo decay for used
        return weights
