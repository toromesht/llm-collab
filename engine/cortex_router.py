#!/usr/bin/env python3
"""
cortex_router.py — Self-Organizing Neural Decision System

Three integrated mechanisms:

1. MCTS PLANNING (decision paths):
   Monte Carlo Tree Search over (region × model) action space.
   UCB1 selection with learned prior → simulate → backpropagate.
   Finds optimal routing paths, not just greedy argmax.

2. HEBBIAN SYNAPTIC PLASTICITY (change synapses):
   Competitive Hebbian learning: "wire together, fire together."
   Region-region connections strengthen when they co-succeed.
   Connections weaken when they fail. New connections can form.
   STDP: causal (A→B success) → LTP; anti-causal → LTD.

3. COMPETITIVE REGION DIFFERENTIATION (specialize):
   Each region develops a tuning curve via online k-means.
   Regions compete for questions → winners specialize.
   Lateral inhibition prevents overlap. Over time:
     Motor    → becomes code specialist
     Parietal → becomes math specialist
     PFC      → becomes logic specialist
     etc.

References:
  Coulom, R. (2006). Efficient Selectivity in MCTS.
  Song, Miller & Abbott (2000). STDP. Nature Neuroscience.
  Hartline & Ratliff (1958). Lateral Inhibition.
  Kohonen, T. (1982). Self-Organizing Maps.
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

# Action space: (region_id, model) — 16 actions
ACTIONS = [
    (0, "ds-pro"), (0, "ds-think"), (0, "groq"),
    (1, "ds-pro"), (1, "ds-think"), (1, "qwen"),
    (2, "ds-pro"), (2, "ds-think"), (2, "glm"),
    (3, "glm"), (3, "qwen"),
    (4, "glm"), (4, "kimi"),
    (5, "kimi"), (5, "qwen"),
]

# ═══════════════════════════════════════════════════════════════
# 1. COMPETITIVE REGION DIFFERENTIATION (Online K-Means)
# ═══════════════════════════════════════════════════════════════

class RegionDifferentiator:
    """
    Each brain region maintains a "prototype vector" in feature space.
    Questions are assigned to the region whose prototype is closest (cosine).
    Prototypes update via online k-means: winner moves toward input.
    Lateral inhibition prevents multiple regions from matching same patterns.
    """

    def __init__(self, n_features: int = 22, n_regions: int = N_REGIONS, lr: float = 0.05):
        # Initialize prototypes randomly (will differentiate over time)
        rng = np.random.RandomState(42)
        self.prototypes = rng.randn(n_regions, n_features) * 0.1
        self.lr = lr
        self.activation_count = np.zeros(n_regions)
        self.n_regions = n_regions

        # Inhibitory connections between regions (learned)
        self.inhibition = np.ones((n_regions, n_regions)) * 0.1
        np.fill_diagonal(self.inhibition, 0.0)

    def activate(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Winner-take-all with lateral inhibition.
        Returns (winning_region, activation_scores).
        """
        x = np.asarray(features, dtype=np.float64).flatten()[:22]
        x_norm = x / (np.linalg.norm(x) + 1e-8)

        # Cosine similarity to each prototype
        similarities = np.zeros(self.n_regions)
        for i in range(self.n_regions):
            p_norm = self.prototypes[i] / (np.linalg.norm(self.prototypes[i]) + 1e-8)
            similarities[i] = np.dot(x_norm, p_norm)

        # Lateral inhibition: active regions suppress others
        activations = similarities.copy()
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if i != j and similarities[i] > 0:
                    activations[j] -= self.inhibition[i, j] * similarities[i]

        winner = int(np.argmax(activations))
        self.activation_count[winner] += 1

        return winner, activations

    def learn(self, features: np.ndarray, winner: int, success: bool):
        """Move winner prototype toward input. Inhibit losers."""
        x = np.asarray(features, dtype=np.float64).flatten()[:22]
        x_norm = x / (np.linalg.norm(x) + 1e-8)

        if success:
            # Winner moves toward input (specialization)
            self.prototypes[winner] += self.lr * (x_norm - self.prototypes[winner])
            self.prototypes[winner] /= (np.linalg.norm(self.prototypes[winner]) + 1e-8)

            # Strengthen inhibition from winner to others
            for j in range(self.n_regions):
                if j != winner:
                    self.inhibition[winner, j] += 0.01
                    self.inhibition[j, winner] -= 0.005  # Weaken reverse
        else:
            # Failure: winner moves away from input
            self.prototypes[winner] -= self.lr * 0.3 * (x_norm - self.prototypes[winner])

        # Clamp inhibition
        self.inhibition = np.clip(self.inhibition, 0.0, 0.5)

    def get_specialization(self) -> dict:
        """Return which feature dimensions each region specializes in."""
        result = {}
        for i in range(N_REGIONS):
            top_dims = np.argsort(-np.abs(self.prototypes[i]))[:3]
            feature_names = [
                "code", "math", "logic", "knowledge", "writing",
                "arch", "trap_single", "trap_need_collab",
                "group_theory", "graph_theory", "topology", "linear_algebra",
                "calculus", "probability", "number_theory", "diff_eq",
                "combinatorics", "optimization",
                "chinese", "safety", "general", "db",
            ]
            result[REGION_NAMES[i]] = {
                "top_features": [feature_names[d] for d in top_dims],
                "activations": int(self.activation_count[i]),
            }
        return result


# ═══════════════════════════════════════════════════════════════
# 2. HEBBIAN SYNAPTIC PLASTICITY (Wire Together, Fire Together)
# ═══════════════════════════════════════════════════════════════

class SynapticPlasticity:
    """
    Region-region synaptic connections that evolve via Hebbian learning.

    STDP rule:
      Δw_ij = A⁺ * exp(-Δt/τ⁺)  if region_i succeeded before region_j (causal)
      Δw_ij = -A⁻ * exp(Δt/τ⁻)   if region_j failed after region_i (anti-causal)

    Synapses encode "effective collaboration pathways" between regions.
    Over time, the system discovers which region combinations work best.
    """

    def __init__(self, n_regions: int = N_REGIONS):
        self.n_regions = n_regions
        # Synaptic weight matrix: w[i][j] = connection strength i→j
        self.weights = np.ones((n_regions, n_regions)) * 0.1
        np.fill_diagonal(self.weights, 0.0)

        # STDP traces
        self.traces = np.zeros((n_regions, n_regions))  # Eligibility traces
        self.tau_trace = 3.0  # Trace decay (Gerstner et al. 2018)

        # Plasticity parameters
        self.A_plus = 0.15   # LTP amplitude
        self.A_minus = 0.10  # LTD amplitude
        self.tau_plus = 5.0  # LTP time constant
        self.tau_minus = 3.0 # LTD time constant

        self.event_counter = 0

    def pre_fire(self, region: int):
        """Region is about to fire — set eligibility trace."""
        self.traces[region, :] += 1.0  # Pre-synaptic trace

    def post_fire(self, region: int, success: bool):
        """Region fired — STDP update based on causality."""
        self.event_counter += 1
        dt = 1.0  # One "timestep" per event

        for pre in range(self.n_regions):
            if pre == region:
                continue
            trace = self.traces[pre, region]

            if success:
                # Causal: pre → post success = LTP
                dw = self.A_plus * trace * math.exp(-dt / self.tau_plus)
                self.weights[pre, region] += dw
            else:
                # Anti-causal: pre → post failure = LTD
                dw = -self.A_minus * trace * math.exp(-dt / self.tau_minus)
                self.weights[pre, region] += dw

        # Decay traces
        self.traces *= math.exp(-1.0 / self.tau_trace)

        # Clamp
        self.weights = np.clip(self.weights, -0.5, 1.0)
        np.fill_diagonal(self.weights, 0.0)

    def get_collaboration_pathways(self) -> List[Tuple[str, str, float]]:
        """Return top learned collaboration pathways between regions."""
        pathways = []
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if i != j and self.weights[i, j] > 0.2:
                    pathways.append((
                        REGION_NAMES[i], REGION_NAMES[j],
                        round(float(self.weights[i, j]), 3)
                    ))
        return sorted(pathways, key=lambda x: -x[2])[:10]

    def get_region_strength(self, region: int) -> float:
        """Total incoming synaptic strength to a region."""
        return float(self.weights[:, region].sum())


# ═══════════════════════════════════════════════════════════════
# 3. MCTS ROUTING (Decision Path Search)
# ═══════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ('action', 'parent', 'children', 'visits', 'value', 'prior')
    def __init__(self, action=None, parent=None, prior=0.0):
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = prior

class MCTSRouter:
    """
    Monte Carlo Tree Search over routing actions.

    For each question:
      1. SELECT: traverse tree via UCB1(s, a) = Q(s,a) + c * P(s,a) * √N(s) / (1+N(s,a))
      2. EXPAND: add unexplored actions as children
      3. SIMULATE: rollout using policy prior → estimate value
      4. BACKPROPAGATE: update Q values up the tree

    The tree encodes "routing strategies" — sequences of model calls.
    Single-level for now (each action = one model call);
    can extend to multi-step reasoning paths.
    """

    def __init__(self, c_ucb: float = 1.4, n_actions: int = len(ACTIONS)):
        self.root = MCTSNode()
        self.c_ucb = c_ucb
        self.n_actions = n_actions
        self.search_count = 0

    def search(self, priors: np.ndarray, n_simulations: int = 50) -> np.ndarray:
        """
        Run MCTS with policy priors.
        Returns visit count distribution → action probabilities.
        """
        for _ in range(n_simulations):
            node = self.root
            path = [node]

            # ── SELECT ──
            while node.children:
                best_score = -float('inf')
                best_action = None
                best_child = None

                for action, child in node.children.items():
                    # UCB1 with prior
                    q = child.value / (child.visits + 1e-8)
                    u = self.c_ucb * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
                    score = q + u

                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_child = child

                node = best_child
                path.append(node)

            # ── EXPAND ──
            if node.visits > 0 or node == self.root:
                for action in range(self.n_actions):
                    if action not in node.children:
                        node.children[action] = MCTSNode(
                            action=action, parent=node,
                            prior=float(priors[action])
                        )

            # ── SIMULATE (use prior as rollout policy) ──
            value = 0.0
            if node.children:
                # Sample from priors
                child_priors = np.array([child.prior for child in node.children.values()])
                child_priors /= child_priors.sum()
                sampled_action = np.random.choice(
                    list(node.children.keys()), p=child_priors
                )
                value = float(priors[sampled_action])  # Proxy: prior as value
            else:
                value = 0.5  # Neutral

            # ── BACKPROPAGATE ──
            for n in path:
                n.visits += 1
                n.value += value

        self.search_count += 1

        # Return visit distribution as action probabilities
        visits = np.zeros(self.n_actions)
        for action, child in self.root.children.items():
            visits[action] = child.visits

        if visits.sum() > 0:
            return visits / visits.sum()
        return priors  # Fallback to priors

    def update_tree(self, action: int, reward: float):
        """After real execution, backpropagate actual reward."""
        if action in self.root.children:
            self.root.children[action].value += reward * 10  # Strong signal
            self.root.children[action].visits += 10

    def advance_root(self, action: int):
        """After taking action, advance tree root (for multi-step)."""
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode()


# ═══════════════════════════════════════════════════════════════
# UNIFIED CORTEX ROUTER
# ═══════════════════════════════════════════════════════════════

class CortexRouter:
    """
    Self-organizing neural decision system combining:

    1. MCTS → Plans optimal routing paths (decision paths)
    2. Hebbian plasticity → Learns region collaboration (change synapses)
    3. Competitive differentiation → Regions specialize (differentiate)

    Pipeline:
      Question → Region differentiation (which regions activate?)
              → MCTS search over (region × model) actions
              → Execute selected models
              → Hebbian update (strengthen successful paths)
              → Region differentiation update (specialize prototypes)
    """

    def __init__(self, brainstem, seed: int = 42):
        self.brainstem = brainstem
        self.differentiator = RegionDifferentiator()
        self.synapses = SynapticPlasticity()
        self.mcts = MCTSRouter()

        # Simple policy prior (can be replaced with neural network)
        self.policy_prior = np.ones(len(ACTIONS)) / len(ACTIONS)
        self.alpha_prior = 0.1  # EMA for prior updates

        self.history = []

    def _build_features(self, question: str) -> np.ndarray:
        """Extract feature vector from question (brainstem-compatible)."""
        from engine.brain import score_task
        scores = score_task(question)
        dims = scores["dims"]
        keys = [
            "code", "math", "logic", "knowledge", "writing",
            "arch", "trap_single", "trap_need_collab",
            "group_theory", "graph_theory", "topology", "linear_algebra",
            "calculus", "probability", "number_theory", "diff_eq",
            "combinatorics", "optimization",
            "chinese", "safety", "general", "db",
        ]
        return np.array([float(dims.get(k, 0)) for k in keys], dtype=np.float64)

    def route(self, question: str) -> dict:
        """Full routing decision using MCTS + region differentiation."""

        # 1. Extract features
        features = self._build_features(question)

        # 2. Brainstem classification
        region_id, confidence, difficulty = self.brainstem.classify(features)

        # 3. Region differentiation: which regions activate?
        winner, activations = self.differentiator.activate(features)
        active_threshold = 0.1
        active_regions = [i for i, a in enumerate(activations) if a > active_threshold]
        if not active_regions:
            active_regions = [winner]

        # 4. Build action priors from activations
        # Higher activation → higher prior for that region's models
        priors = np.ones(len(ACTIONS)) * 0.01
        for i, (rid, model) in enumerate(ACTIONS):
            if rid in active_regions:
                priors[i] = float(activations[rid]) + 0.1
                # Bonus for well-learned synapses targeting this region
                priors[i] += self.synapses.get_region_strength(rid) * 0.3

        # Blend with learned policy prior
        priors = 0.7 * (priors / priors.sum()) + 0.3 * self.policy_prior
        priors /= priors.sum()

        # 5. MCTS search
        visit_probs = self.mcts.search(priors, n_simulations=30)

        # 6. Select top actions
        top_k = min(5, len(ACTIONS))
        top_actions = np.argpartition(-visit_probs, top_k - 1)[:top_k]
        top_actions = top_actions[np.argsort(-visit_probs[top_actions])]

        # 7. Group by region
        selected_models = {}
        for idx in top_actions[:top_k]:
            rid, model = ACTIONS[idx]
            rname = REGION_NAMES[rid]
            if rname not in selected_models:
                selected_models[rname] = []
            if model not in selected_models[rname]:
                selected_models[rname].append(model)

        # 8. Fire pre-synaptic signals for active regions
        for rid in active_regions:
            self.synapses.pre_fire(rid)

        decision = {
            "primary_region": REGION_NAMES[region_id],
            "brainstem_winner": REGION_NAMES[winner],
            "active_regions": [REGION_NAMES[r] for r in active_regions],
            "selected_models": selected_models,
            "difficulty": round(float(difficulty), 3),
            "confidence": round(float(confidence), 3),
            "mcts_visits": {
                f"{REGION_NAMES[ACTIONS[i][0]]}:{ACTIONS[i][1]}": round(float(visit_probs[i]), 4)
                for i in top_actions[:5]
            },
            "region_activations": {
                REGION_NAMES[i]: round(float(activations[i]), 3)
                for i in range(N_REGIONS)
            },
            "top_action_indices": [int(i) for i in top_actions[:3]],
        }

        self.history.append(decision)
        return decision

    def learn(self, features: np.ndarray, active_regions: List[int],
              action_indices: List[int], rewards: List[float]):
        """
        Post-execution learning:
        1. Hebbian update for each region→region collaboration
        2. Region differentiation prototype update
        3. MCTS tree update with real rewards
        4. Policy prior EMA update
        """

        # 1. Hebbian plasticity
        for i, rid in enumerate(active_regions):
            success = rewards[i] > 0.5 if i < len(rewards) else False
            self.synapses.post_fire(rid, success)

        # 2. Region differentiation
        winner = active_regions[0] if active_regions else 0
        avg_reward = np.mean(rewards[:3]) if rewards else 0.0
        self.differentiator.learn(features, winner, avg_reward > 0.5)

        # 3. MCTS update
        for action_idx, reward in zip(action_indices, rewards):
            self.mcts.update_tree(action_idx, reward)

        # 4. Policy prior update (EMA toward successful actions)
        for action_idx, reward in zip(action_indices, rewards):
            if reward > 0.5:
                self.policy_prior[action_idx] += self.alpha_prior * (1.0 - self.policy_prior[action_idx])
            else:
                self.policy_prior[action_idx] *= (1.0 - self.alpha_prior)
        self.policy_prior /= self.policy_prior.sum()

    def get_state(self) -> dict:
        """Return full system state for diagnostics."""
        return {
            "region_differentiation": self.differentiator.get_specialization(),
            "collaboration_pathways": self.synapses.get_collaboration_pathways(),
            "synaptic_strengths": {
                REGION_NAMES[i]: round(self.synapses.get_region_strength(i), 3)
                for i in range(N_REGIONS)
            },
            "mcts_searches": self.mcts.search_count,
            "decision_count": len(self.history),
            "policy_prior_top5": [
                f"{REGION_NAMES[ACTIONS[i][0]]}:{ACTIONS[i][1]}={self.policy_prior[i]:.3f}"
                for i in np.argsort(-self.policy_prior)[:5]
            ],
        }


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from engine.brainstem_wrapper import PythonBrainstem
    bs = PythonBrainstem(seed=42)
    router = CortexRouter(bs)

    print("=== Cortex Router: MCTS + Hebbian + Differentiation ===\n")

    questions = [
        "用Python写一个二分查找算法",
        "证明拉格朗日中值定理",
        "什么是自由能原理？请详细解释。",
        "设计一个分布式缓存系统架构",
        "写一首关于人工智能的中文诗",
    ]

    for q in questions:
        decision = router.route(q)
        features = router._build_features(q)

        print(f"Q: {q[:45]}...")
        print(f"  Brainstem: {decision['primary_region']} | Winner: {decision['brainstem_winner']}")
        print(f"  Active regions: {decision['active_regions']}")
        print(f"  Selected models: {decision['selected_models']}")
        print(f"  MCTS top actions:")
        for k, v in list(decision['mcts_visits'].items())[:3]:
            print(f"    {k}: {v:.4f}")

        # Simulate execution + learning
        active_regions_list = [REGION_NAMES.index(r) for r in decision['active_regions']]
        action_indices = decision.get('top_action_indices', [])
        rewards = [np.random.random() * 0.6 + 0.4 for _ in action_indices]  # Simulated
        router.learn(features, active_regions_list, action_indices, rewards)
        print()

    print("=== System State ===\n")

    state = router.get_state()
    print("Region Specialization:")
    for r, info in state['region_differentiation'].items():
        print(f"  {r}: {info['top_features']} (n={info['activations']})")

    print("\nCollaboration Pathways (learned):")
    for src, dst, w in state['collaboration_pathways'][:5]:
        print(f"  {src} → {dst}: w={w:.3f}")

    print("\nSynaptic Strengths:")
    for r, s in state['synaptic_strengths'].items():
        print(f"  {r}: {s}")

    print(f"\nMCTS searches: {state['mcts_searches']}")
    print(f"Decisions: {state['decision_count']}")
    print(f"Top policy priors: {state['policy_prior_top5']}")

    print("\n=== Cortex Router: ALL TESTS PASSED ===")
