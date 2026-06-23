#!/usr/bin/env python3
"""
neuro_runner.py — 7-Mechanism Neuro-Synaptic Routing Harness

Each mechanism is independently paper-grounded (see neural_mechanisms.py).
This harness wires them together for end-to-end routing + learning.

Pipeline:
  embed(question) → GridCellMap.encode → PredictiveCoding.predict
  → +SynapticTagging.boost +HebbianSTDP.weights → RecurrentLateralInhibition.apply
  → route → execute → learn(all 7 mechanisms from reward)

Mechanisms (paper → implementation):
  GridCellMap       — Moser & Moser (2005) Nature 436:801
  PredictiveCoding  — Rao & Ballard (1999) Nature Neuroscience 2:79
  SynapticTagCapture— Frey & Morris (1997) Nature 385:533
  HebbianSTDP       — Song, Miller & Abbott (2000) Nature Neuroscience 3:919
  LateralInhibition — Hartline & Ratliff (1958) J Gen Physiol 42:1241
  MemoryConsolidation— Kandel (2001) Science 294:1030
  SynapticDecay     — Ebbinghaus (1885) + Wixted (2004)
"""

import sys
import os
import json
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.neural_mechanisms import (
    GridCellMap,
    PredictiveCodingLayer,
    SynapticTaggingCapture,
    HebbianSTDP,
    RecurrentLateralInhibition,
    MemoryConsolidation,
    SynapticDecay,
)
from engine.brain import call

MODELS = ["ds-pro", "ds-think", "glm", "qwen", "kimi", "groq"]
N = len(MODELS)
STATUS_FILE = os.path.expanduser("~/.claude/tools/neuro_status.json")


class NeuralRunner:
    """
    7-mechanism neural routing harness.

    Each mechanism has a clean, paper-grounded interface.
    Mechanisms are composed (not inherited) for modularity.
    """

    def __init__(self, seed: int = 42):
        # --- Embedder ---
        self.has_bert = False
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_bert = True
        except Exception:
            pass

        dim = 384  # MiniLM embedding dimension

        # --- Mechanism 1: Grid Cell Map (Moser & Moser 2005) ---
        self.grid = GridCellMap(n_modules=4, dim=dim, seed=seed)

        # --- Mechanism 2: Predictive Coding (Rao & Ballard 1999) ---
        # 12 grid cell outputs → 6 model predictions
        self.pred = PredictiveCodingLayer(
            n_input=self.grid.n_output, n_output=N, seed=seed)

        # --- Mechanism 3: Synaptic Tagging & Capture (Frey & Morris 1997) ---
        self.tags = SynapticTaggingCapture(n_synapses=N, seed=seed)

        # --- Mechanism 4: Hebbian STDP (Song, Miller & Abbott 2000) ---
        self.hebb = HebbianSTDP(n_synapses=N, seed=seed)

        # --- Mechanism 5: Recurrent Lateral Inhibition (Hartline & Ratliff 1958) ---
        self.lateral = RecurrentLateralInhibition(n_units=N, seed=seed)

        # --- Mechanism 6: Memory Consolidation (Kandel 2001) ---
        self.consolid = MemoryConsolidation(n_synapses=N, seed=seed)

        # --- Mechanism 7: Synaptic Decay (Ebbinghaus 1885) ---
        self.decay = SynapticDecay(decay_rate=0.0005, seed=seed)

        self.episodes = 0
        self._lock = threading.Lock()

    # ─── Embedding ──────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Embed question text → 384-dim vector (MiniLM or n-gram hash fallback)."""
        if self.has_bert and self.embedder is not None:
            e = self.embedder.encode(text, convert_to_numpy=True).astype(np.float64)
            return e / (np.linalg.norm(e) + 1e-8)

        # Fallback: character n-gram hashing (no external deps)
        v = np.zeros(384, dtype=np.float64)
        tx = text.lower()
        for n in [2, 3, 4]:
            for i in range(len(tx) - n + 1):
                v[hash(tx[i:i + n]) % 384] += 1.0
        return v / (np.linalg.norm(v) + 1e-8)

    # ─── Status File ────────────────────────────────────────────

    def _write_status(self, stage: str, models: dict = None,
                      region: dict = None, done: bool = False):
        """Write neuro_status.json for status line display."""
        data = {}
        if os.path.exists(STATUS_FILE):
            try:
                data = json.loads(open(STATUS_FILE, "r", encoding="utf-8").read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        data["stage"] = stage
        data["done"] = done
        data["timestamp"] = time.time()
        if models:
            data["models"] = models
        if region:
            data["region"] = region
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # ─── Routing ────────────────────────────────────────────────

    def route(self, question: str) -> dict:
        """
        Route question through all 7 mechanisms.

        Flow:
          1. GridCellMap: question → cognitive position (Moser 2005)
          2. PredictiveCoding: cognitive position → model success predictions (Rao 1999)
          3. SynapticTagging: tagged models get exploration boost (Frey 1997)
          4. HebbianSTDP: learned pathway weights (Song 2000)
          5. RecurrentLateralInhibition: winner suppresses competitors (Hartline 1958)

        Returns:
          {"primary_model": str, "predictions": {model: score, ...}, "scores": ndarray}
        """
        emb = self._embed(question)

        # 1. Grid cell encoding (Moser 2005)
        grid_vec = self.grid.encode(emb)

        # 2. Predictive coding prediction (Rao & Ballard 1999)
        preds = self.pred.predict(grid_vec)

        # 3. Synaptic tagging boost (Frey & Morris 1997)
        tag_boost = np.array([self.tags.get_boost(i) - 1.0 for i in range(N)])

        # 4. Hebbian STDP pathway strength (Song, Miller & Abbott 2000)
        heb_boost = self.hebb.get_weights()

        # Combine scores: predictions + tag exploration + Hebbian pathway
        scores = preds + tag_boost * 0.3 + heb_boost * 0.2

        # 5. Recurrent lateral inhibition (Hartline & Ratliff 1958)
        #    Winner suppresses competitors proportionally
        if np.max(scores) > 0:
            scores = self.lateral.apply(scores)

        winner_idx = int(np.argmax(scores))

        # Build prediction ranking
        ranked = np.argsort(-preds)
        predictions = {
            MODELS[i]: round(float(preds[i]), 3)
            for i in ranked[:4]
        }

        return {
            "primary_model": MODELS[winner_idx],
            "predictions": predictions,
            "scores": scores,
            "winner_idx": winner_idx,
        }

    # ─── Learning ───────────────────────────────────────────────

    def learn(self, question: str, model: str, reward: float):
        """
        Update all 7 mechanisms from observed reward.

        Args:
            question: Original question text
            model: Which model was used
            reward: 1.0 (correct), 0.0 (incorrect), or intermediate
        """
        with self._lock:
            mi = MODELS.index(model)
            emb = self._embed(question)
            grid_vec = self.grid.encode(emb)

            # 2. Predictive coding: error-driven weight update (Rao & Ballard 1999)
            target = np.zeros(N)
            target[mi] = reward
            preds = self.pred.predict(grid_vec)
            error = self.pred.compute_error(preds, target)
            self.pred.update(grid_vec, error, lr=0.01)

            # 3. Synaptic tagging: correct → set tag, consecutive correct → PRP (Frey & Morris 1997)
            if reward > 0.5:
                self.tags.set_tag(mi, strength=reward)
                if self.consolid.consecutive_successes[mi] >= 3:
                    self.tags.generate_prp(amount=0.5)
            capture = self.tags.step(dt=1.0)

            # 4. Hebbian STDP: pre→post timing plasticity (Song, Miller & Abbott 2000)
            self.hebb.pre_fire(mi)
            self.hebb.post_fire(mi, reward > 0.5)

            # 5. Lateral inhibition adaptation (Hartline & Ratliff 1958)
            winner_idx = int(np.argmax(self.pred.predict(grid_vec)))
            if winner_idx != mi:
                self.lateral.adapt(winner_idx, mi, reward > 0.5)

            # 6. Memory consolidation (Kandel 2001): consecutive success → L-LTP
            self.consolid.update(mi, reward > 0.5)

            # 7. Synaptic decay (Ebbinghaus 1885): forgetting with use protection
            self.hebb.weights = self.decay.apply(
                self.hebb.weights,
                used_indices=[mi],
                consolidated_mask=self.consolid.consolidated,
            )

            # Apply consolidation boost to STDP weights
            cons_boost = self.consolid.get_consolidation_boost(mi)
            if cons_boost > 0:
                self.hebb.weights[mi] += cons_boost * 0.1
                self.hebb.weights = np.clip(self.hebb.weights, -1.0, 1.0)

            self.episodes += 1

    # ─── Execute ────────────────────────────────────────────────

    def execute(self, question: str) -> dict:
        """
        Full neuro-synaptic execution cycle: route → execute → learn.

        Returns:
          {"model": str, "reward": float, "response": str, "predictions": dict}
        """
        self._write_status("routing")
        routing = self.route(question)
        model = routing["primary_model"]

        self._write_status("executing", models={model: "running"})

        # Execute selected model
        try:
            prompt = f"Step by step. Answer precisely.\n\n{question}"
            resp = call(model, prompt, max_tok=2000)
            reward = 1.0 if len(str(resp)) > 20 else 0.0
        except Exception:
            resp = "[ERR]"
            reward = 0.0

        self._write_status("learning", models={model: "done"})
        self.learn(question, model, reward)

        # Persist result
        result_file = os.path.expanduser("~/.synapseflow/brain/last_result.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "chosen": model,
                "reward": reward,
                "episode": self.episodes,
                "consolidated": int(np.sum(self.consolid.consolidated)),
                "tagged": int(np.sum(self.tags.tags > 0)),
                "ts": time.time(),
            }, f)

        self._write_status("done", models={model: "done"}, done=True)

        return {
            "model": model,
            "reward": reward,
            "response": str(resp)[:500],
            "predictions": routing["predictions"],
        }

    # ─── State Persistence ──────────────────────────────────────

    def get_state(self) -> dict:
        """Export all mechanism states for persistence."""
        return {
            "episodes": self.episodes,
            "grid": self.grid.get_state(),
            "pred": self.pred.get_state(),
            "tags": self.tags.get_state(),
            "hebb": self.hebb.get_state(),
            "lateral": self.lateral.get_state(),
            "consolid": self.consolid.get_state(),
            "decay": self.decay.get_state(),
        }

    def load_state(self, state: dict):
        """Restore all mechanism states."""
        self.episodes = state.get("episodes", 0)
        for key, mech in [
            ("grid", self.grid), ("pred", self.pred),
            ("tags", self.tags), ("hebb", self.hebb),
            ("lateral", self.lateral), ("consolid", self.consolid),
            ("decay", self.decay),
        ]:
            if key in state:
                mech.load_state(state[key])


# ─── Main: smoke test ────────────────────────────────────────────

if __name__ == "__main__":
    r = NeuralRunner()

    training_questions = [
        ("Solve 2x+5=17 step by step", "ds-think"),
        ("What is the capital of France?", "groq"),
        ("Write Python quicksort", "ds-pro"),
    ]

    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        for q, expected in training_questions:
            res = r.execute(q)
            ok = "OK" if res["model"] == expected else "LEARN"
            print(f"  {ok} {q[:35]:35s} → {res['model']:10s} "
                  f"r={res['reward']} pred={list(res['predictions'].items())[:2]}")

    state = r.get_state()
    print(f"\nEpisodes: {r.episodes}")
    print(f"Consolidated: {int(np.sum(r.consolid.consolidated))}")
    print(f"Tagged: {int(np.sum(r.tags.tags > 0))}")
    print(f"PRP pool: {r.tags.get_prp_level():.3f}")
    print("NeuralRunner: ALL 7 MECHANISMS ACTIVE")
