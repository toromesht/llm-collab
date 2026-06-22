#!/usr/bin/env python3
"""
dual_network_runner.py — Two-Network Architecture

Network 1: ROUTING NETWORK (NeuroSynapticRouter)
  - Grid Cells encode question position in cognitive space
  - Predictive Coding predicts success per model
  - Output: which model to call

Network 2: PATHWAY NETWORK (SynapseRouter)
  - Growing/pruning Hebbian network
  - Learns routing pathways from outcomes
  - STDP: strengthen successful paths, weaken failed paths
  - Consolidation: strong pathways become permanent
  - Forgetting: unused pathways decay

Flow:
  Question → [Grid Cell Encode] → [Predictive Coding: which model?]
           → [API Call] → [Observe reward]
           → [Pathway Network: Hebbian update + grow/prune]
           → [Routing Network: Predictive coding weight update]
           → Both networks learn simultaneously from the same reward
"""

import sys,os,json,time,threading,numpy as np
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.neuro_synaptic_router import NeuroSynapticRouter
from engine.synapse_router import SynapseRouter
from engine.brain import call

MODELS = ["ds-pro", "ds-think", "glm", "qwen", "kimi", "groq"]
COST = {"ds-pro": 0.002, "ds-think": 0.001, "glm": 0.001, "qwen": 0.001, "kimi": 0.0015, "groq": 0.0002}
STATUS = os.path.expanduser("~/.claude/tools/neuro_status.json")

# ═══════════════════════════════════════════════════════════════
# DUAL NETWORK RUNNER
# ═══════════════════════════════════════════════════════════════

class DualNetworkRunner:
    def __init__(self):
        self.routing_net = NeuroSynapticRouter()   # Network 1: decides what to do
        self.pathway_net = SynapseRouter()          # Network 2: learns from outcomes
        self.episodes = 0

    def _write_status(self, stage, models=None, region=None, done=False):
        d = {}
        if os.path.exists(STATUS):
            try: d = json.loads(open(STATUS, "r", encoding="utf-8").read())
            except: pass
        d["stage"] = stage; d["done"] = done; d["timestamp"] = time.time()
        if models: d["models"] = models
        if region: d["region"] = region
        os.makedirs(os.path.dirname(STATUS), exist_ok=True)
        with open(STATUS, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def route(self, question: str) -> dict:
        """Network 1: Routing decision via Grid Cells + Predictive Coding."""
        return self.routing_net.route(question)

    def execute_and_learn(self, question: str):
        """Full pipeline: route → execute → learn in both networks."""
        # ── Phase 1: Route (Network 1) ──
        self._write_status("routing")
        decision = self.routing_net.route(question)
        chosen = decision["primary_model"]
        predictions = decision.get("predictions", {})

        # Mark running
        self._write_status("executing", models={chosen: "running"})

        # ── Phase 2: Execute API call ──
        try:
            t0 = time.time()
            prompt = f"Step by step. Answer precisely.\n\n{question}"
            response = call(chosen, prompt, max_tok=2000)
            latency = time.time() - t0
            # Simple quality check: response has content
            reward = 1.0 if len(str(response)) > 20 else 0.0
        except Exception as e:
            response = f"[ERR: {e}]"
            reward = 0.0

        # Mark done
        self._write_status("done", models={chosen: "done"})

        # ── Phase 3: Learn in BOTH networks ──
        # Network 1 (Routing): Predictive coding weight update
        self.routing_net.learn(question, chosen, reward)

        # Network 2 (Pathway): Hebbian STDP + grow/prune/consolidate
        self.pathway_net.learn(question, chosen, reward)

        # Periodically, sync pathway knowledge to routing network
        if self.pathway_net.episode % 10 == 0:
            self._sync_networks()

        self.episodes += 1

        # Log to repo
        repo_log = os.path.expanduser("~/Desktop/collab-cloud/data/brain_activity.jsonl")
        os.makedirs(os.path.dirname(repo_log), exist_ok=True)
        with open(repo_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "episode": self.episodes, "chosen": chosen,
                "reward": reward, "predictions": predictions,
                "routing_state": self.routing_net.get_state(),
                "pathway_state": self.pathway_net.get_network_state(),
                "timestamp": time.time(),
            }, ensure_ascii=False) + "\n")

        return {
            "chosen_model": chosen,
            "reward": reward,
            "response": str(response)[:500],
            "predictions": predictions,
            "episode": self.episodes,
        }

    def _sync_networks(self):
        """Sync: strong pathway edges → boost routing network predictions."""
        state = self.pathway_net.get_network_state()
        for p in state.get("pathways", [])[:10]:
            if abs(p["weight"]) > 0.3:
                model = p["model"]
                if model in self.routing_net.MODELS:
                    # Boost the predictive coding bias for this model
                    idx = self.routing_net.MODELS.index(model)
                    self.routing_net.pc_layer.b[idx] += p["weight"] * 0.01


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = DualNetworkRunner()

    # Test questions
    questions = [
        ("Solve: 2x+5=17. Show steps.", "ds-think"),
        ("What is the capital of France?", "groq"),
        ("Write a Python function to reverse a linked list.", "ds-pro"),
        ("Prove that sqrt(2) is irrational.", "ds-think"),
        ("How many legs does a spider have?", "groq"),
    ]

    print("=" * 60)
    print("DUAL NETWORK RUNNER — Grid Cells + Predictive Coding + Hebbian STDP")
    print("=" * 60)

    for epoch in range(3):
        print(f"\n--- Epoch {epoch+1} ---")
        for q, expected in questions:
            result = runner.execute_and_learn(q)
            status = "OK" if result["chosen_model"] == expected else "LEARN"
            print(f"  [{status:4s}] {q[:40]:40s} → {result['chosen_model']:10s} r={result['reward']:.1f}")
            if result["predictions"]:
                top = list(result["predictions"].items())[:2]
                print(f"         pred: {top}")

    print(f"\n=== ROUTING NETWORK (Grid Cells + Predictive Coding) ===")
    rs = runner.routing_net.get_state()
    print(f"  Episodes: {rs['episodes']} | Avg reward: {rs['avg_reward']}")
    print(f"  Precision: {rs['precision']}")
    print(f"  Tags: {rs['tags']}")

    print(f"\n=== PATHWAY NETWORK (Growing Hebbian STDP) ===")
    ps = runner.pathway_net.get_network_state()
    print(f"  Nodes: {ps['total_nodes']} (consolidated: {ps['consolidated_nodes']})")
    print(f"  Pathways: {ps['total_pathways']} | Created: {ps['created']} | Pruned: {ps['pruned']}")

    print(f"\n=== FINAL ROUTING TEST ===")
    for q in [
        "Solve the quadratic equation x^2+bx+c=0",
        "Tell me about Paris",
        "Implement Dijkstra's algorithm",
    ]:
        d = runner.route(q)
        print(f"  {q[:45]:45s} → {d['primary_model']:10s}")
        for m, p in list(d.get('predictions', {}).items())[:2]:
            print(f"    {m}: {p:.3f}", end="")
        print()

    print("\nDUAL NETWORK RUNNER: READY")
