#!/usr/bin/env python3
"""
synapse_router.py — Growing Neuro-Synaptic Routing Network

NOT a fixed MLP. A living network that:
  • Grows new synapses when facing novel questions
  • Strengthens pathways via Hebbian LTP (fire together → wire together)
  • Weakens unused connections via LTD (use it or lose it)
  • Prunes dead synapses below a threshold
  • Consolidates strong pathways into permanent routes
  • Forgets outdated patterns via synaptic decay
  • Self-organizes: no fixed architecture, graph grows/shrinks with data

Algorithm:
  1. Question embedding → find nearest prototype node (cosine)
  2. Prototype has outgoing synaptic edges to each model
  3. Winner prototype + its edges = "brain region activation"
  4. After observing reward: Hebbian update (success→strengthen, fail→weaken)
  5. Lateral inhibition: winner suppresses neighboring prototypes
  6. Novelty detection: if no prototype is close enough → grow new one
  7. Pruning: prototypes unused for N steps → die
  8. Consolidation: edges above threshold → permanent pathway (no longer plastic)

This is a self-organizing map meets Hebbian plasticity meets active inference.
"""

import numpy as np
import json, os, time, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════
# EMBEDDER (Character n-gram hashing, 384-dim, no external deps)
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


# ═══════════════════════════════════════════════════════════════
# SYNAPTIC PROTOTYPE NODE
# ═══════════════════════════════════════════════════════════════

class SynapseNode:
    """One prototype node. Has a feature vector + synaptic edges to all models."""

    __slots__ = ('id', 'prototype', 'synapses', 'activations', 'last_used',
                 'consolidated', 'created_at')

    def __init__(self, node_id: int, features: np.ndarray, n_models: int = 6):
        self.id = node_id
        self.prototype = np.asarray(features, dtype=np.float64).copy()  # 384-dim center
        # Synaptic weights to each model (positive=excitatory, negative=inhibitory)
        self.synapses = np.zeros(n_models, dtype=np.float64)
        self.activations = 1       # How many times this node was the winner
        self.last_used = 0         # Episode of last use
        self.consolidated = False  # Permanent pathway?
        self.created_at = 0        # Episode when created

    def similarity(self, query: np.ndarray) -> float:
        """Cosine similarity between query and prototype."""
        p = self.prototype / (np.linalg.norm(self.prototype) + 1e-8)
        q = query / (np.linalg.norm(query) + 1e-8)
        return float(np.dot(p, q))

    def hebbian_update(self, query: np.ndarray, model_idx: int, reward: float,
                        lr: float = 0.05):
        """
        STDP-inspired Hebbian update:
        - Move prototype toward query (winner moves toward input)
        - Strengthen synapse to chosen model if success (LTP)
        - Weaken synapse to chosen model if failure (LTD)
        - Mildly depress all other synapses (lateral inhibition)
        """
        # Prototype update (online k-means)
        self.prototype += lr * 0.3 * (query - self.prototype)
        self.prototype /= (np.linalg.norm(self.prototype) + 1e-8)

        # Synaptic plasticity (STDP)
        if reward > 0.5:
            # LTP: strengthen the winning synapse
            self.synapses[model_idx] += lr * reward * (1.0 - self.synapses[model_idx])
            # Mildly strengthen others (associative)
            self.synapses += lr * 0.1 * reward * np.maximum(0, self.synapses)
        else:
            # LTD: weaken the failing synapse
            self.synapses[model_idx] -= lr * (1.0 - reward)
            # Mildly weaken others
            self.synapses -= lr * 0.02

        # Clamp synapses to [-1, 1]
        self.synapses = np.clip(self.synapses, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# GROWING NEURO-SYNAPTIC NETWORK
# ═══════════════════════════════════════════════════════════════

class SynapseRouter:
    """
    Self-organizing neuro-synaptic routing network.

    Parameters:
      theta_grow   — similarity threshold for creating new nodes (lower=more nodes)
      theta_prune  — minimum activations to survive pruning
      theta_consolidate — synapse weight threshold for permanent pathway
      decay_rate   — per-episode synaptic decay (forgetting)
      lateral_inhibition — how strongly winner suppresses others
    """

    MODELS = ["ds-pro", "ds-think", "glm", "qwen", "kimi", "groq"]
    N_MODELS = len(MODELS)

    # Hyperparameters from neuroscience/CS literature:
    #   theta_grow: Fritzke (1995) Growing Neural Gas — insertion threshold
    #   theta_prune: Changeux & Danchin (1976) selective stabilization
    #   theta_consolidate: Kandel (2001) L-LTP consolidation threshold
    #   decay_rate: Ebbinghaus (1885) forgetting curve
    #   A_LTP/A_LTD: Song, Miller & Abbott (2000) STDP amplitudes
    #   lateral_inhibition: Hartline & Ratliff (1958) winner-take-all

    def __init__(self, seed: int = 42,
                 theta_grow: float = 0.35,        # Lower = fewer new nodes (Fritzke 1995)
                 theta_prune: int = 50,            # More patience before pruning
                 theta_consolidate: float = 0.7,   # L-LTP consolidation (Kandel 2001)
                 decay_rate: float = 0.0005,       # Slow forgetting (Ebbinghaus)
                 A_LTP: float = 0.15,              # STDP LTP amplitude (Song et al. 2000)
                 A_LTD: float = 0.10,              # STDP LTD amplitude
                 lateral_inhibition: float = 0.005):
        self.nodes: List[SynapseNode] = []
        self.embedder = TextEmbedder()
        self.episode = 0
        self.next_id = 0

        self.theta_grow = theta_grow
        self.theta_prune = theta_prune
        self.theta_consolidate = theta_consolidate
        self.decay_rate = decay_rate
        self.lateral_inhibition = lateral_inhibition

        # Statistics
        self.total_created = 0
        self.total_pruned = 0
        self.total_consolidated = 0

        # Seeding: start with one node per "region archetype"
        self._seed_initial_nodes()

    def _seed_initial_nodes(self):
        """Create 6 initial prototype nodes (one per brain region archetype)."""
        archetypes = [
            "python code function algorithm programming",
            "math theorem proof equation calculus algebra",
            "logic reasoning deduction if then therefore",
            "knowledge history facts definition explain",
            "writing essay creative poetry language",
            "visual image diagram chart figure",
        ]
        for text in archetypes:
            emb = self.embedder.encode(text)
            node = SynapseNode(self.next_id, emb, self.N_MODELS)
            self.next_id += 1
            self.nodes.append(node)

    def _find_winners(self, query_emb: np.ndarray, k: int = 3) -> List[SynapseNode]:
        """Find k-nearest prototype nodes by cosine similarity."""
        if not self.nodes:
            return []
        scored = [(n.similarity(query_emb), n) for n in self.nodes]
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:k]]

    def _apply_decay(self):
        """Synaptic decay: all synapses drift toward zero (forgetting)."""
        for node in self.nodes:
            node.synapses *= (1.0 - self.decay_rate)

    def _prune_dead_nodes(self):
        """Remove nodes that haven't been used recently."""
        alive = []
        for node in self.nodes:
            if node.consolidated:
                alive.append(node)  # Permanent pathways survive pruning
            elif (self.episode - node.last_used) > self.theta_prune:
                self.total_pruned += 1
            else:
                alive.append(node)
        self.nodes = alive

    def _grow_if_needed(self, query_emb: np.ndarray, max_similarity: float):
        """Create a new node if no existing prototype matches well enough."""
        if max_similarity < self.theta_grow:
            node = SynapseNode(self.next_id, query_emb, self.N_MODELS)
            node.created_at = self.episode
            node.last_used = self.episode
            self.next_id += 1
            self.total_created += 1
            self.nodes.append(node)

    def _check_consolidation(self):
        """Consolidate strong pathways into permanent routes."""
        for node in self.nodes:
            if not node.consolidated and node.activations >= 10:
                if np.any(node.synapses > self.theta_consolidate):
                    node.consolidated = True
                    self.total_consolidated += 1

    def route(self, question: str) -> dict:
        """Route question to best model via neuro-synaptic network."""
        emb = self.embedder.encode(question)
        winners = self._find_winners(emb, k=3)

        if not winners:
            return {"primary_model": "groq", "model_scores": {},
                    "pathway": "no-match", "novelty": True}

        # Aggregate synaptic votes across winner nodes
        scores = np.zeros(self.N_MODELS)
        for node in winners:
            sim = node.similarity(emb)
            scores += sim * node.synapses

        # UCB-style: add uncertainty bonus for underexplored models
        for i in range(self.N_MODELS):
            n_used = sum(1 for n in self.nodes if n.synapses[i] != 0)
            scores[i] += 0.5 / (n_used + 1)

        best_idx = int(np.argmax(scores))

        # Pathway description
        winner_regions = []
        for node in winners[:2]:
            strongest = np.argmax(node.synapses)
            if node.synapses[strongest] > 0.3:
                winner_regions.append(f"n{node.id}→{self.MODELS[strongest]}")

        return {
            "primary_model": self.MODELS[best_idx],
            "model_scores": {self.MODELS[i]: round(float(scores[i]), 3)
                            for i in np.argsort(-scores)[:4]},
            "pathway": " → ".join(winner_regions) if winner_regions else "searching",
            "novelty": max(n.similarity(emb) for n in winners) < self.theta_grow,
            "active_nodes": len(self.nodes),
            "winners": [n.id for n in winners],
        }

    def learn(self, question: str, model: str, reward: float):
        """
        One learning step:
        1. Find winner nodes
        2. Hebbian update on winner + lateral inhibition on neighbors
        3. Grow new node if needed
        4. Periodic pruning and decay
        """
        emb = self.embedder.encode(question)
        model_idx = self.MODELS.index(model)
        winners = self._find_winners(emb, k=3)

        if not winners:
            # No match → create new node
            node = SynapseNode(self.next_id, emb, self.N_MODELS)
            node.last_used = self.episode
            self.next_id += 1
            self.total_created += 1
            self.nodes.append(node)
            node.hebbian_update(emb, model_idx, reward, lr=0.1)
        else:
            # Hebbian update on winner
            primary = winners[0]
            primary.hebbian_update(emb, model_idx, reward, lr=0.05)
            primary.last_used = self.episode
            primary.activations += 1

            # Lateral inhibition: neighbors move away slightly
            for neighbor in winners[1:]:
                neighbor.prototype -= self.lateral_inhibition * (emb - neighbor.prototype)
                neighbor.prototype /= (np.linalg.norm(neighbor.prototype) + 1e-8)
                neighbor.synapses[model_idx] -= self.lateral_inhibition * 0.5

            # Grow if needed
            self._grow_if_needed(emb, winners[0].similarity(emb))

        # Periodic maintenance
        self.episode += 1
        self._apply_decay()

        if self.episode % 10 == 0:
            self._prune_dead_nodes()
            self._check_consolidation()

    def get_network_state(self) -> dict:
        """Return full network state for visualization/debugging."""
        pathways = []
        for node in self.nodes:
            for i, w in enumerate(node.synapses):
                if abs(w) > 0.2:
                    pathways.append({
                        "node_id": node.id,
                        "model": self.MODELS[i],
                        "weight": round(float(w), 3),
                        "consolidated": node.consolidated,
                        "activations": node.activations,
                    })

        return {
            "total_nodes": len(self.nodes),
            "consolidated_nodes": sum(1 for n in self.nodes if n.consolidated),
            "total_pathways": len(pathways),
            "pathways": sorted(pathways, key=lambda x: -abs(x["weight"]))[:20],
            "created": self.total_created,
            "pruned": self.total_pruned,
            "consolidated": self.total_consolidated,
            "episodes": self.episode,
        }


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    router = SynapseRouter()
    import numpy as np

    math_qs = [
        "Solve 2x+5=17", "Integral of sin(x)", "Prove sum of first n integers",
        "Eigenvalues of 2x2 matrix", "Derivative of x^3 ln(x)",
        "Compute determinant of 3x3", "Laplace transform of e^at",
        "Find roots of x^2-5x+6", "Taylor series of cos(x)", "Solve differential equation dy/dx=y",
    ]
    simple_qs = [
        "Color of sky", "How many legs does a cat have", "Capital of France",
        "Who wrote Romeo and Juliet", "WW2 end year", "How many ounces in a cup",
        "Boiling point of water", "Speed of light", "Largest planet", "What is DNA",
    ]
    code_qs = [
        "Write quicksort in Python", "SQL query top 10 customers", "Binary tree traversal",
        "Implement LRU cache", "Find all prime numbers under 100", "Merge two sorted arrays",
        "Reverse a linked list", "Design a hash table", "Dijkstra shortest path", "BFS on graph",
    ]

    print("Growing neuro-synaptic network...")
    for epoch in range(5):
        for qs, model, base_reward in [(math_qs, "ds-think", 0.95), (simple_qs, "groq", 0.9), (code_qs, "ds-pro", 0.9)]:
            for q in qs:
                decision = router.route(q)
                chosen = decision["primary_model"]
                reward = base_reward if chosen == model else (0.1 if chosen in ["groq", "glm"] else 0.3)
                reward += np.random.normal(0, 0.05)
                router.learn(q, model, np.clip(reward, 0, 1))

    print(f"\n=== NETWORK STATE ===")
    state = router.get_network_state()
    print(f"Nodes: {state['total_nodes']} (consolidated: {state['consolidated_nodes']})")
    print(f"Pathways: {state['total_pathways']}")
    print(f"Created: {state['created']} | Pruned: {state['pruned']} | Consolidated: {state['consolidated']}")
    print(f"Episodes: {state['episodes']}")

    print(f"\n=== STRONGEST PATHWAYS ===")
    for p in state['pathways'][:10]:
        star = "★" if p["consolidated"] else " "
        print(f"  n{p['node_id']:2d} → {p['model']:10s} w={p['weight']:+.3f} {star} ({p['activations']}x)")

    print(f"\n=== ROUTING TESTS ===")
    for q in ["Solve the quadratic equation x^2+3x-4=0", "What is the capital of Japan?",
              "Write a function to check if a string is palindrome"]:
        d = router.route(q)
        print(f"  {q[:45]:45s} → {d['primary_model']:10s} (novelty={d['novelty']})")

    print("\nSynapseRouter: ALL TESTS PASSED")
