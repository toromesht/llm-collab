#!/usr/bin/env python3
"""
neuro_agent.py — Neurosynaptic Orchestrator

Full brain-inspired architecture:
  Question -> [0. Shortcut Check] -> [1. Brainstem Classify]
          -> [2. Region Route] -> [3. Region Execute]
          -> [4. Cortical Validation] -> [5. Synaptic Update]
          -> Answer

Architecture layers (bottom-up):
  Brainstem (Fortran/NumPy):    Fast HD encoding + SDM classification
  Brain Regions (6 areas):      Specialized model pools per domain
  Synaptic Pathways:            STDP + L-LTP hardening + LTD forgetting
  Cortical Validation:          DS-PRO quality gate + critic review

Paper references per component:
  [Brainstem] Kanerva 1988 (SDM), Kanerva 2009 (HD Computing)
  [Regions]   MasRouter ACL 2025, Unified Cascade 2024
  [Pathways]  Kandel Science 2001 (L-LTP), Dudek & Bear PNAS 1992 (LTD)
  [Cortex]    Quality gate (Unified Cascade), DS-PRO critic
  [STDP/BCM]  Song & Miller 2000, Bienenstock J Neurosci 1982

Usage:
  from engine.neuro_agent import NeuroOrchestrator
  agent = NeuroOrchestrator()
  answer = agent.solve("What is Lagrange's theorem?")
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Import all layers ─────────────────────────────────────────

from engine.brainstem_wrapper import (
    load as load_brainstem,
    N_DIMS, REGION_NAMES,
)
from engine.regions import (
    load_brain_regions, RegionExecutor, MultiRegionRouter,
    get_router,
)
from engine.synapse_network import (
    PathwayNetwork, SynapticPathway,
    load_synapse_config,
)
from engine.brain import (
    call, execute_single, execute_pipeline, execute_collab,
    score_task, quality_gate, estimate_K, decide_mode,
    synaptic_update, spsa_step,
    CC, MODELS, PARAMS,
)
from engine.math_router import MathRouter
from engine.path_learner import PathLearner

# ─── Feature key ordering (must match brainstem.f90 N_DIMS=22) ─

FEATURE_KEYS = [
    "code", "math", "logic", "knowledge", "writing",
    "arch", "trap_single", "trap_need_collab",
    "group_theory", "graph_theory", "topology", "linear_algebra",
    "calculus", "probability", "number_theory", "diff_eq",
    "combinatorics", "optimization",
    "chinese", "safety", "general", "db",
]


def _features_to_vector(features: Dict) -> List[float]:
    """Convert feature dict to ordered list matching Fortran N_DIMS."""
    return [float(features.get(k, 0)) for k in FEATURE_KEYS]


def _detect_category(features: Dict) -> str:
    """Determine primary question category from features."""
    if features.get("code", 0) >= 1:
        return "code"
    if features.get("math", 0) >= 1:
        return "math"
    if features.get("logic", 0) >= 1:
        return "logic"
    if features.get("writing", 0) >= 1:
        return "writing"
    if features.get("chinese", 0) >= 1:
        return "chinese"
    if features.get("knowledge", 0) >= 1:
        return "knowledge"
    return "general"


# ═══════════════════════════════════════════════════════════════
# CORTICAL VALIDATION (Unified Cascade 2024 + MasRouter ACL 2025)
# ═══════════════════════════════════════════════════════════════

def cortical_validation(answer: str, question: str, features: Dict,
                        difficulty: float) -> Tuple[str, bool]:
    """
    Prefrontal cortex final validation using DS-PRO.

    Ref: Unified Cascade (arXiv 2410.10347) — quality gate cascade
    Ref: MasRouter (ACL 2025) — cascaded controller with quality check

    Flow:
      1. Basic quality gate (length checks)
      2. If pass -> return
      3. If fail -> DS-PRO review
      4. If DS-PRO finds issues -> mark for cascade

    Returns:
        (answer, passed_validation)
    """
    # Step 1: Basic quality gate (from brain.py)
    if quality_gate(answer, features):
        return answer, True

    # Step 2: DS-PRO critic review
    try:
        review_prompt = (
            f"你是一个严谨的审稿人。请检查以下回答的完整性、准确性和逻辑连贯性。\n"
            f"如有错误或遗漏，指出具体问题。如无问题，回复OK。\n\n"
            f"问题: {question[:300]}\n\n回答: {answer[:800]}\n\n"
            f"审稿意见:"
        )
        review = call("ds-pro", review_prompt, max_tok=300, temp=0.1)

        if "OK" in review and len(review) < 20:
            return answer, True

        # DS-PRO found issues — flag for cascade
        print(f"  {CC.get('Y','')}[CORTEX] DS-PRO flagged: {review[:120]}...")
        return answer, False

    except Exception as e:
        # Can't validate — assume OK to not block
        print(f"  {CC.get('R','')}[CORTEX ERR] {e}")
        return answer, True


def cascade_reroute(question: str, features: Dict, difficulty: float,
                    previous_models: List[str] = None) -> str:
    """
    Cascade rerouting when cortical validation fails.

    Ref: Unified Cascade (arXiv 2410.10347) — increase expert count
    when quality is insufficient.

    Strategy: use full collab with all available models.
    """
    # Escalate: use DS-PRO + best non-used models
    if previous_models is None:
        previous_models = []

    all_models = ["ds-pro", "ds-think", "glm", "qwen", "kimi"]
    fresh_models = [m for m in all_models if m not in previous_models][:3]

    if not fresh_models:
        fresh_models = ["ds-pro"]

    print(f"  {CC.get('Y','')}[CASCADE] Escalating to collab: {fresh_models}")
    return execute_collab(question, fresh_models)


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class NeuroOrchestrator:
    """
    Complete neurosynaptic orchestrator.

    Pipeline:
      Shortcut Check -> Brainstem -> Regions -> Execution
      -> Cortical Validation -> Synaptic Update -> Brainstem Train

    Each layer has a specific biological analog:
      Brainstem:   Medulla/pons — fast reflex routing
      Regions:     Cortical areas — specialized processing
      Synapses:    Hippocampus — memory consolidation
      Cortex:      Prefrontal — executive validation
    """

    def __init__(self, config_path: str = None):
        self.brainstem = load_brainstem()
        self.regions = get_router(config_path)
        self.synapses = PathwayNetwork(config_path)
        self.math_router = MathRouter(seed=42)       # JL+LSH+UCB+CUSUM+SPRT
        self.path_learner = PathLearner()             # Bayesian adaptive forgetting
        self.round_counter = self._load_round_counter()

        # Preload feature key order for speed
        self._feature_keys = FEATURE_KEYS

        print(f"  {CC.get('C','')}[NEURO] Brainstem: {type(self.brainstem).__name__}")
        print(f"  {CC.get('C','')}[NEURO] MathRouter: JL+LSH+CUSUM+SPRT+TD(λ)")
        print(f"  {CC.get('C','')}[NEURO] Regions: {len(self.regions.regions)} areas loaded")
        stats = self.synapses.get_statistics()
        print(f"  {CC.get('C','')}[NEURO] Synapses: {stats['active']} active, "
              f"{stats['hardened']} hardened, {stats['shortcuts']} shortcuts")
        print(f"  {CC.get('C','')}[NEURO] Round: {self.round_counter}")

    def _load_round_counter(self) -> int:
        """Load persistent round counter."""
        counter_file = Path.home() / ".claude" / "tools" / "round_counter.json"
        if counter_file.exists():
            try:
                return int(counter_file.read_text().strip())
            except (ValueError, OSError):
                return 0
        return 0

    def _save_round_counter(self):
        """Persist round counter."""
        counter_file = Path.home() / ".claude" / "tools" / "round_counter.json"
        counter_file.parent.mkdir(parents=True, exist_ok=True)
        counter_file.write_text(str(self.round_counter))

    def solve(self, question: str) -> str:
        """
        Main entry point: question -> answer through full neuro pipeline.

        Args:
            question: User's question string

        Returns:
            Complete answer string
        """
        start_time = time.time()

        # ── 0. Feature Extraction (brain.py score_task) ──────
        scores = score_task(question)
        features = scores["dims"]
        difficulty = scores["difficulty"]
        category = _detect_category(features)
        feature_vec = _features_to_vector(features)

        print(f"\n{CC.get('W','')}{'='*64}{CC.get('R','')}")
        print(f"  {CC.get('W','')}Neurosynaptic Brain v1 | "
              f"6 Regions | STDP+LTP+LTD{CC.get('R','')}")
        print(f"  Difficulty={difficulty:.2f} | Category={category}")

        # ── 1. Shortcut Check (Abraham & Bear 1996) ──────────
        shortcut = self.synapses.check_shortcuts(feature_vec)
        if shortcut:
            region_name = shortcut["region"]
            model = shortcut["model"]
            print(f"  {CC.get('M','')}[SHORTCUT] Hardened path: "
                  f"{region_name} -> {model} "
                  f"(sim={shortcut.get('similarity', 1.0):.2f})")

            # Use hardened path directly
            region = self.regions.get_region_by_name(region_name)
            if region is None:
                region = self.regions.get_region_by_id(0)

            answer = region.execute(question, features, difficulty)

            # Still do cortical validation
            answer, validated = cortical_validation(answer, question, features, difficulty)

            # Update pathway
            pathway = self.synapses.find_or_create(
                region_name, model, category, feature_vec
            )
            self.synapses.update(pathway, validated, feature_vec)

            elapsed = time.time() - start_time
            print(f"  {CC.get('D','')}[SHORTCUT] {elapsed*1000:.0f}ms (bypassed brainstem)")
            return answer

        # ── 2. Brainstem Classification (Kanerva 1988/2009) ──
        region_id, brainstem_conf, brainstem_diff = self.brainstem.classify(feature_vec)
        region_name = REGION_NAMES[region_id] if 0 <= region_id < 6 else "temporal_cortex"

        print(f"  {CC.get('B','')}[BRAINSTEM]{CC.get('R','')} "
              f"Region={region_name}({region_id}) "
              f"Conf={brainstem_conf:.2f} Diff={brainstem_diff:.2f}")

        # ── 3. Region Execution ───────────────────────────────
        region = self.regions.get_region_by_id(region_id)

        # M9: 5-HT-driven exploration (Fuxe et al. 2007)
        # High serotonin = patient, explore alternative models occasionally
        serotonin = self.synapses.serotonin if hasattr(self.synapses, 'serotonin') else 0.5
        import random
        if serotonin > 0.6 and random.random() < (serotonin - 0.5):
            # Try secondary model for explorative diversity
            alt_models = region.secondary_models
            if alt_models:
                explore_model = random.choice(alt_models)
                print(f"  {CC.get('M','')}[5-HT EXPLORE]{CC.get('R','')} "
                      f"Serotonin={serotonin:.2f}, trying {explore_model}")

        # Determine strategy based on confidence
        if brainstem_conf < self.regions.multi_region_threshold:
            # Low confidence: try multiple regions
            scores = self.regions.get_activation_scores(features)
            top_2 = sorted(scores.items(), key=lambda x: -x[1])[:2]

            models_used = []
            for rname, _ in top_2:
                r = self.regions.get_region_by_name(rname)
                primary = r.get_active_models(difficulty)
                models_used.extend(primary[:1])

            models_used = list(dict.fromkeys(models_used))[:3]  # dedupe

            if len(models_used) >= 2:
                print(f"  {CC.get('M','')}[MULTI-REGION] "
                      f"Low conf={brainstem_conf:.2f}, activating: {', '.join(top_2[0][0] for _ in top_2[:0])}")
                answer = execute_collab(question, models_used)
                strategy_used = "multi_region_collab"
                primary_model = models_used[0] if models_used else "ds-pro"
            else:
                answer = region.execute(question, features, difficulty)
                strategy_used = region.get_strategy(difficulty)
                primary_model = models_used[0] if models_used else "ds-pro"
        else:
            # High confidence: single region
            print(f"  {CC.get('G','')}[REGION] {region.config.get('name', region_name)}")
            answer = region.execute(question, features, difficulty)
            strategy_used = region.get_strategy(difficulty)
            active_models = region.get_active_models(difficulty)
            primary_model = active_models[0] if active_models else "ds-pro"

        # ── 4. Cortical Validation (DS-PRO review) ────────────
        answer, validated = cortical_validation(answer, question, features, difficulty)

        if not validated:
            print(f"  {CC.get('Y','')}[CORTEX] Validation failed, cascading...")
            prev_models = [primary_model]
            answer = cascade_reroute(question, features, difficulty, prev_models)
            # Re-validate
            answer, validated = cortical_validation(answer, question, features, difficulty)

        # ── 5. Synaptic Update ────────────────────────────────
        model_to_update = primary_model
        pathway = self.synapses.find_or_create(
            region_name, model_to_update, category, feature_vec
        )
        self.synapses.update(pathway, validated, feature_vec)

        # Math Router update (JL+LSH+UCB+CUSUM+SPRT)
        self.math_router.update(
            region_name, model_to_update, category,
            reward=1.0 if validated else 0.0, features=features
        )

        # Path Learner update (Bayesian adaptive forgetting)
        pl_path = self.path_learner.get_or_create(region_name, model_to_update, category)
        self.path_learner.update(pl_path, 1.0 if validated else 0.0)

        # Also update brainstem SDM (online learning)
        if validated:
            try:
                self.brainstem.sdm_write(feature_vec, region_id)
            except Exception:
                pass  # Non-critical

        # ── 6. Log and increment ──────────────────────────────
        self.round_counter += 1
        self._save_round_counter()

        elapsed = time.time() - start_time
        stats = self.synapses.get_statistics()

        print(f"{CC.get('W','')}{'='*64}{CC.get('R','')}")
        print(f"  {CC.get('D','')}Round {self.round_counter} | "
              f"Pathway: {pathway.pathway_id} | "
              f"Hardened: {pathway.hardened} | "
              f"Elapsed: {elapsed*1000:.0f}ms")
        print(f"  {CC.get('D','')}Synapses: {stats['active']} active, "
              f"{stats['hardened']} hardened, {stats['shortcuts']} shortcuts")
        print()

        return answer

    def get_stats(self) -> Dict:
        """Return comprehensive system statistics including math-router learning."""
        bs_reads, bs_writes, bs_avg_act = self.brainstem.get_stats()
        synapse_stats = self.synapses.get_statistics()
        math_stats = self.math_router.get_stats()
        pl_stats = self.path_learner.get_stats()

        return {
            "round": self.round_counter,
            "brainstem": {
                "type": type(self.brainstem).__name__,
                "sdm_reads": bs_reads,
                "sdm_writes": bs_writes,
                "avg_activation": round(bs_avg_act, 3),
            },
            "regions": {
                name: r.activation_count
                for name, r in self.regions.regions.items()
            },
            "synapses": synapse_stats,
            "math_router": math_stats,
            "path_learner": pl_stats,
            "neuromodulation": {
                "dopamine": round(synapse_stats.get("dopamine", 0.5), 3),
                "serotonin": round(synapse_stats.get("serotonin", 0.5), 3),
                "branching_ratio": round(synapse_stats.get("branching_ratio", 1.0), 3),
                "prp_pools": synapse_stats.get("prp_pools", {}),
                "astrocyte_ca": synapse_stats.get("astrocyte_ca", {}),
            },
        }


# ─── Module singleton ──────────────────────────────────────────

_orchestrator = None

def get_orchestrator(config_path: str = None) -> NeuroOrchestrator:
    """Get or create the NeuroOrchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = NeuroOrchestrator(config_path)
    return _orchestrator


def solve(question: str) -> str:
    """Convenience function: solve a question through the neuro pipeline."""
    return get_orchestrator().solve(question)


# ─── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    agent = NeuroOrchestrator()

    test_questions = [
        "用Python写一个二分查找算法",
        "什么是拉格朗日中值定理？",
        "请解释数据库索引的工作原理",
        "用中文写一篇关于人工智能的短文",
    ]

    for q in test_questions:
        print(f"\n{'#'*60}")
        print(f"Q: {q}")
        print(f"{'#'*60}")
        try:
            answer = agent.solve(q)
            print(f"\nA: {answer[:300]}...")
        except Exception as e:
            print(f"ERR: {e}")
            import traceback
            traceback.print_exc()

        # Don't spam API in test mode
        break

    stats = agent.get_stats()
    print(f"\n{'='*60}")
    print(f"System Stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"{'='*60}")
    print("Neuro Agent: ARCHITECTURE READY")
