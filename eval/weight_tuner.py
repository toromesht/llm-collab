#!/usr/bin/env python3
"""
Reward-Penalty Weight Tuner — from 98 questions of eval data
  Reward: correct answer → +1 weight for that model×category
  Penalty: wrong answer → -0.5 weight (wrong is worse than not trying)
  Output: evolved model-category affinity matrix for brain.py routing
"""
import json, os, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Load all 3 eval rounds
eval_dir = Path(__file__).parent
results = {}
for f in sorted(eval_dir.glob("eval_*.json")):
    data = json.loads(f.read_text(encoding='utf-8'))
    if "results" in data and "version" in data:
        results[f"v{data['version']}"] = data

# ─── Initialize weights from V1-V3 data ──────────────
# Category-level model performance from all rounds
CATEGORIES = ["math", "code", "logic", "knowledge", "safety", "chinese", "science", "creative"]
MODELS = ["DS-PRO", "DS-Think", "GLM", "QWEN", "Kimi"]

# Accumulate (correct, total) per model×category
stats = {m: {c: [0, 0] for c in CATEGORIES} for m in MODELS}

# Map eval categories to standard categories
CAT_MAP = {
    "math": "math", "math_hard": "math", "advanced_math": "math",
    "code": "code", "algorithm": "code",
    "reasoning": "logic", "logic": "logic", "deep_logic": "logic",
    "knowledge": "knowledge", "knowledge_cn": "knowledge",
    "safety": "safety",
    "chinese_depth": "chinese",
    "science_tech": "science",
    "creative": "creative",
}

for ver, data in results.items():
    for model, r in data.get("results", {}).items():
        if model not in MODELS: continue
        cats = r.get("categories", {})
        for cat_name, cat_data in cats.items():
            std_cat = CAT_MAP.get(cat_name, cat_name)
            if std_cat not in CATEGORIES: continue
            # Try to extract score/total
            score = cat_data.get("score", cat_data.get("pct", 0) / 100 * cat_data.get("total", 10))
            total = cat_data.get("total", 10)
            if isinstance(score, float) and score <= 1.0:
                score = score * total
            stats[model][std_cat][0] += score
            stats[model][std_cat][1] += total

# ─── Calculate evolved weights with reward/penalty ────
print("Model-Category Performance Matrix (from 98 questions):")
print(f"{'Model':<12}", end="")
for c in CATEGORIES:
    print(f"{c:<10}", end="")
print(f"{'Overall':<10}")
print("-" * (12 + 10 * (len(CATEGORIES) + 1)))

evolved_weights = {}
for m in MODELS:
    total_correct = 0
    total_q = 0
    print(f"{m:<12}", end="")
    weights = {}
    for c in CATEGORIES:
        correct, total = stats[m][c]
        if total > 0:
            pct = correct / total * 100
            # Reward: strong performance → higher weight
            # Penalty: below 50% → negative contribution
            if pct >= 90:
                weight = 1.0   # Excellent
            elif pct >= 75:
                weight = 0.7   # Good
            elif pct >= 60:
                weight = 0.4   # Average
            elif pct >= 50:
                weight = 0.1   # Weak
            else:
                weight = -0.2  # Penalty: harmful to use
        else:
            pct = 0
            weight = 0
        weights[c] = {"pct": round(pct, 1), "weight": weight, "samples": total}
        print(f"{pct:.0f}%({weight:+.1f})".ljust(11), end="")
        total_correct += correct
        total_q += total
    overall = total_correct / total_q * 100 if total_q > 0 else 0
    print(f"{overall:.0f}%")
    evolved_weights[m] = {"overall": round(overall, 1), "weights": weights}

# ─── Best Model per Category ──────────────────────────
print(f"\nBest Model per Category (with confidence):")
for c in CATEGORIES:
    best_m = max(MODELS, key=lambda m: evolved_weights[m]["weights"][c]["weight"])
    best_w = evolved_weights[best_m]["weights"][c]
    worst_m = min(MODELS, key=lambda m: evolved_weights[m]["weights"][c]["weight"])
    print(f"  {c:<12}: {best_m} ({best_w['pct']}%, weight={best_w['weight']:+.1f}) | worst: {worst_m}")

# ─── Generate Routing Rules ──────────────────────────
print(f"\nDerived Routing Rules:")
rules = []
for m in MODELS:
    strong = [(c, w["pct"]) for c, w in evolved_weights[m]["weights"].items() if w["weight"] >= 0.7]
    avoid = [(c, w["pct"]) for c, w in evolved_weights[m]["weights"].items() if w["weight"] < 0]
    if strong:
        cats = ", ".join(f"{c}({p}%)" for c, p in strong)
        print(f"  USE {m} for: {cats}")
        rules.append({"model": m, "action": "use", "categories": [c for c,_ in strong]})
    if avoid:
        cats = ", ".join(f"{c}({p}%)" for c, p in avoid)
        print(f"  AVOID {m} for: {cats}")
        rules.append({"model": m, "action": "avoid", "categories": [c for c,_ in avoid]})

# ─── Multi-Model Combination Strategy ────────────────
print(f"\nMulti-Model Strategy for Hard Problems (logic cap at 50%):")
print(f"  Strategy: Top-2 models vote → 3rd model as tiebreaker")
for c in ["logic", "math", "code"]:
    ranked = sorted(MODELS, key=lambda m: -evolved_weights[m]["weights"][c]["weight"])
    top2 = ranked[:2]
    tiebreaker = ranked[2] if len(ranked) > 2 else None
    print(f"  {c}: {top2[0]} + {top2[1]} → tiebreaker: {tiebreaker}")

# ─── Save ────────────────────────────────────────────
outfile = eval_dir / "evolved_weights.json"
with open(outfile, "w", encoding="utf-8") as f:
    json.dump({
        "version": "from_98_questions",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "models": MODELS,
        "categories": CATEGORIES,
        "weights": evolved_weights,
        "routing_rules": rules,
    }, f, ensure_ascii=False, indent=2)

# Also save as brain.py compatible format
brain_weights = {}
for m in MODELS:
    brain_weights[m] = {
        cat: evolved_weights[m]["weights"][cat]["weight"]
        for cat in CATEGORIES
    }
tools = Path(os.path.expanduser('~/.claude/tools'))
with open(tools / "brain_weights.json", "w", encoding="utf-8") as f:
    json.dump({"weights": brain_weights, "best_per_category": {c: max(MODELS, key=lambda m: brain_weights[m].get(c,0)) for c in CATEGORIES}}, f, ensure_ascii=False, indent=2)

print(f"\nSaved: {outfile}")
print(f"Brain weights saved: ~/.claude/tools/brain_weights.json")
