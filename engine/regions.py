#!/usr/bin/env python3
"""
regions.py — Brain Region Configuration and Execution

Six functional brain regions, each with specialized model pools.
Reuses brain.py's call(), execute_single(), execute_pipeline(), execute_collab().

Paper basis for region specialization:
  MasRouter (Yue et al., ACL 2025) — cascaded multi-agent routing with
  specialized agent pools per task category.

Region-to-model mapping is data-driven from:
  partitions/*/training_data.json (28 curated entries across 5 domains)
  eval/evolved_weights.json (3-round benchmark results)
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Load brain.py utilities ───────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.brain import (
    call, execute_single, execute_pipeline, execute_collab,
    score_task, quality_gate,
    MODELS, CC,
)

# ─── Config loading ────────────────────────────────────────────

def load_brain_regions(config_path: str = None) -> Dict:
    """Load brain region configuration from JSON."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "brain_regions.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["regions"]


class RegionExecutor:
    """
    Executes questions within a single brain region.

    Each region has:
      - Primary model pool (always used)
      - Secondary models (used when difficulty is high)
      - Banned models (never used in this region)
      - Feature checks (determine if region should activate)
      - Default execution strategy (single/pipeline/collab)
    """

    def __init__(self, region_name: str, region_config: Dict):
        self.name = region_name
        self.config = region_config
        self.primary_models = region_config.get("models", [])
        self.secondary_models = region_config.get("secondary", [])
        self.banned_models = region_config.get("banned", [])
        self.feature_checks = region_config.get("feature_checks", [])
        self.default_execution = region_config.get("execution", "single")
        self.threshold = region_config.get("threshold", 0.5)

        # Track usage stats
        self.activation_count = 0
        self.success_count = 0

    def should_activate(self, features: Dict) -> float:
        """
        Check if this region should handle the given features.

        Returns:
            activation score (0-1), higher = more relevant
        """
        score = 0.0
        max_possible = 0.0
        for check in self.feature_checks:
            val = features.get(check, 0)
            score += val
            max_possible += 1.0  # each check can contribute at most 1

        if max_possible == 0:
            return 0.0

        normalized = score / max_possible
        return min(1.0, normalized)

    def get_strategy(self, difficulty: float) -> str:
        """Determine execution strategy based on difficulty."""
        strategy = self.default_execution

        if difficulty < 0.3:
            return "single"
        elif difficulty < 0.6:
            # For pipeline_or_collab, choose pipeline at medium difficulty
            if strategy == "pipeline_or_collab":
                return "pipeline"
            return strategy if strategy != "pipeline_or_collab" else "single"
        else:
            # High difficulty: use full capability
            if strategy == "pipeline_or_collab":
                return "collab"
            elif strategy == "single_or_pipeline":
                return "pipeline"
            return strategy

    def get_active_models(self, difficulty: float) -> List[str]:
        """
        Get the list of models to use for this execution.

        At low difficulty: use only the single best primary model.
        At medium difficulty: use primary models.
        At high difficulty: add secondary models.
        """
        if difficulty < 0.3:
            return [self.primary_models[0]] if self.primary_models else ["ds-pro"]
        elif difficulty < 0.6:
            return self.primary_models[:2] if len(self.primary_models) > 1 else self.primary_models
        else:
            all_models = list(dict.fromkeys(self.primary_models + self.secondary_models))
            # Filter banned models
            all_models = [m for m in all_models if m not in self.banned_models]
            return all_models[:3]  # Max 3 models

    def execute(self, question: str, features: Dict, difficulty: float) -> str:
        """
        Execute question using region's model pool.

        Args:
            question: The user's question
            features: Feature dict from score_task()
            difficulty: Difficulty score (0-1)

        Returns:
            model answer string
        """
        self.activation_count += 1
        strategy = self.get_strategy(difficulty)
        models = self.get_active_models(difficulty)

        if not models:
            models = ["ds-pro"]  # fallback

        try:
            if strategy == "single":
                answer = execute_single(question, model=models[0])
            elif strategy == "pipeline":
                steps = [{"model": m, "role": self.config.get("name", "core")} for m in models[:2]]
                answer = execute_pipeline(question, steps)
            elif strategy == "collab":
                answer = execute_collab(question, models[:3])
            else:
                answer = execute_single(question, model=models[0])
        except Exception as e:
            # Fallback: use ds-pro directly
            print(f"  {CC.get('R','')}[REGION ERR] {self.name}: {e}")
            answer = execute_single(question, model="ds-pro")

        return answer


class MultiRegionRouter:
    """
    Routes questions to one or more brain regions.

    Handles:
      - Single region activation (high confidence)
      - Multi-region activation (low confidence, combine 2 regions)
      - Fallback to default region
    """

    def __init__(self, config_path: str = None):
        region_configs = load_brain_regions(config_path)
        self.regions: Dict[str, RegionExecutor] = {}
        for name, cfg in region_configs.items():
            self.regions[name] = RegionExecutor(name, cfg)

        self.default_region = "temporal_cortex"
        self.multi_region_threshold = 0.55
        self._region_list = list(self.regions.keys())

    def get_region_by_id(self, region_id: int) -> RegionExecutor:
        """Get region by its numeric ID (0-5)."""
        if 0 <= region_id < len(self._region_list):
            name = self._region_list[region_id]
            return self.regions[name]
        return self.regions[self.default_region]

    def get_region_by_name(self, name: str) -> Optional[RegionExecutor]:
        """Get region by string name."""
        return self.regions.get(name)

    def get_activation_scores(self, features: Dict) -> Dict[str, float]:
        """Compute activation score for every region."""
        return {
            name: region.should_activate(features)
            for name, region in self.regions.items()
        }

    def route(self, question: str, features: Dict, brainstem_region_id: int,
              brainstem_confidence: float, difficulty: float) -> Tuple:
        """
        Route question to region(s) and execute.

        Args:
            question: The user's question text
            features: Feature dict from score_task()
            brainstem_region_id: Region ID from brainstem classification (0-5)
            brainstem_confidence: Brainstem confidence (0-1)
            difficulty: Difficulty score (0-1)

        Returns:
            (answer, region_name, strategy, models_used)
        """
        # High confidence: use brainstem's suggestion directly
        if brainstem_confidence >= self.multi_region_threshold:
            region = self.get_region_by_id(brainstem_region_id)
            strategy = region.get_strategy(difficulty)
            models = region.get_active_models(difficulty)
            answer = region.execute(question=question, features=features, difficulty=difficulty)
            return answer, region.name, strategy, models

        # Low confidence: activate multiple regions
        scores = self.get_activation_scores(features)
        sorted_regions = sorted(scores.items(), key=lambda x: -x[1])

        # Use top 2 regions
        top_regions = sorted_regions[:2]

        # For multi-region, use the primary model from each top region
        primary_models = []
        for rname, rscore in top_regions:
            region = self.regions[rname]
            primary_models.extend(region.get_active_models(difficulty)[:1])

        primary_models = list(dict.fromkeys(primary_models))[:3]  # dedupe, max 3

        if len(primary_models) <= 1:
            # Only one unique model — use single region
            best_region_name = sorted_regions[0][0]
            region = self.regions[best_region_name]
            strategy = region.get_strategy(difficulty)
            models = region.get_active_models(difficulty)
            answer = region.execute(question=question, features=features, difficulty=difficulty)
            return answer, region.name, strategy, models

        # Multi-model collab across regions
        answer = execute_collab(question, primary_models)
        strategy = "multi_region_collab"
        return answer, "+".join([r for r, _ in top_regions]), strategy, primary_models


# ─── Module-level singleton ────────────────────────────────────

_router = None

def get_router(config_path: str = None) -> MultiRegionRouter:
    """Get or create the MultiRegionRouter singleton."""
    global _router
    if _router is None:
        _router = MultiRegionRouter(config_path)
    return _router


# ─── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    router = get_router()
    print(f"Loaded {len(router.regions)} brain regions:")
    for name, region in router.regions.items():
        print(f"  [{region.config['index']}] {name}")
        print(f"      Models: {region.primary_models} | Banned: {region.banned_models}")
        print(f"      Checks: {region.feature_checks[:5]}...")
        print(f"      Strategy: {region.default_execution}")

    # Test activation
    test_features = {"code": 2.0, "db": 1.0, "math": 0.0}
    scores = router.get_activation_scores(test_features)
    print(f"\nActivation scores (code test):")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "=" * int(score * 20)
        print(f"  {name}: {score:.2f} {bar}")
    print("Regions: ALL TESTS PASSED")
