#!/usr/bin/env python3
"""
brain.py — LLM Model Router (semantic classification → single-model routing)

Core flow (all code in this file is actually called by main()):
  1. classify_question() — MiniLM embedding → cosine similarity to prototypes
  2. score_task()        — category + difficulty + model affinity
  3. decide_route()      — rule-based single-model selection
  4. call()              — OpenAI-compatible API call
  5. execute_single()    — call model + return answer

The 7 neural mechanisms (neural_mechanisms.py → neuro_runner.py) are a
separate research experiment. They are NOT used in the production routing path.
See ONBOARDING.md for architecture details.
"""

import sys
import json
import os
import re
import math
import time
import numpy as np
from pathlib import Path
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════
# CONFIG — auto-discover from multiple paths
# ═══════════════════════════════════════════════════════════════

def _load_config():
    for path in [
        Path.home() / ".claude" / "tools" / "llm-config.json",
        Path.home() / ".synapseflow" / "config.json",
    ]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return {}

config = _load_config()

# ═══════════════════════════════════════════════════════════════
# MODELS — register here to add a new model
# ═══════════════════════════════════════════════════════════════

MODELS = {
    "ds-pro":   ("[DS-PRO]", lambda: (
        OpenAI(api_key=config["deepseek_pro"]["api_key"],
               base_url=config["deepseek_pro"]["base_url"]),
        config["deepseek_pro"]["model"])),
    "kimi":     ("[KIMI]", lambda: (
        OpenAI(api_key=config["kimi"]["api_key"],
               base_url=config["kimi"]["base_url"]),
        config["kimi"]["model"])),
    "glm":      ("[GLM]", lambda: (
        OpenAI(api_key=config["sjtu_zhiyuan"]["api_key"],
               base_url=config["sjtu_zhiyuan"]["base_url"]),
        "glm")),
    "qwen":     ("[QWEN]", lambda: (
        OpenAI(api_key=config["sjtu_zhiyuan"]["api_key"],
               base_url=config["sjtu_zhiyuan"]["base_url"]),
        "qwen")),
    "ds-think": ("[DS-Think]", lambda: (
        OpenAI(api_key=config["sjtu_zhiyuan"]["api_key"],
               base_url=config["sjtu_zhiyuan"]["base_url"]),
        "deepseek-reasoner")),
    "groq":     ("[GROQ]", lambda: (
        OpenAI(api_key=config["groq"]["api_key"],
               base_url=config["groq"]["base_url"]),
        config["groq"]["model"])),
}

CC = {"R": "\033[0m", "C": "\033[36m", "G": "\033[32m", "Y": "\033[33m",
      "D": "\033[2m", "E": "\033[31m", "M": "\033[35m", "B": "\033[1;34m",
      "W": "\033[1;37m"}

# ═══════════════════════════════════════════════════════════════
# EMBEDDER — lazy-loaded MiniLM (384-dim)
# ═══════════════════════════════════════════════════════════════

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _embedder = False
    return _embedder if _embedder is not False else None


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION — few-shot semantic prototypes
# ═══════════════════════════════════════════════════════════════

CATEGORY_PROTOTYPES = {
    "math": [
        "求解微分方程 dy/dx = y",
        "证明拉格朗日中值定理",
        "计算矩阵的特征值和特征向量",
        "求极限 lim(x→0) sin(x)/x",
        "解方程 2x+5=17",
    ],
    "code": [
        "用Python写一个二分查找算法",
        "实现快速排序",
        "写一个SQL查询找出前10名客户",
        "用递归实现二叉树遍历",
        "写一个LRU缓存",
    ],
    "logic": [
        "三个嫌疑犯只有一个说真话，找出罪犯",
        "12个球中有一个重量不同，用天平称三次找出",
        "蒙提霍尔问题：换门还是不换门",
        "如果A则B，非B所以非A，这个推理对吗",
        "所有猫都有四条腿，这个动物有四条腿，所以它是猫，对吗",
    ],
    "architecture": [
        "设计一个微服务架构的系统",
        "如何设计一个高可用的数据库集群",
        "设计一个电商系统的技术方案",
        "后端技术选型：Go vs Java vs Python",
        "分布式系统的CAP理论权衡",
    ],
    "writing": [
        "写一篇关于人工智能的论文摘要",
        "翻译这段英文到中文",
        "润色这段文字使其更流畅",
        "写一首关于秋天的诗",
    ],
    "knowledge": [
        "什么是量子纠缠",
        "法国首都是哪里",
        "解释相对论的基本原理",
        "DNA的结构是什么",
    ],
}

CATEGORY_MODEL_RULES = {
    "math":         {"primary": "ds-think", "fallback": "ds-pro",
                     "reason": "DS-Think 97% GSM8K"},
    "code":         {"primary": "ds-pro",   "fallback": "ds-think",
                     "reason": "DS-Pro strongest coder"},
    "logic":        {"primary": "ds-pro",   "fallback": "ds-think",
                     "reason": "All models weak at logic (~40%)"},
    "architecture": {"primary": "glm",      "fallback": "ds-pro",
                     "reason": "GLM best on knowledge/architecture tasks"},
    "writing":      {"primary": "glm",      "fallback": "kimi",
                     "reason": "GLM strongest Chinese writing"},
    "knowledge":    {"primary": "groq",     "fallback": "glm",
                     "reason": "Easy knowledge → cheap Groq"},
}

MODEL_COST = {  # USD per 1k tokens
    "ds-pro": 0.002, "ds-think": 0.001, "groq": 0.0002,
    "glm": 0.001, "qwen": 0.001, "kimi": 0.0015,
}
MODEL_LATENCY = {  # Seconds (relative)
    "ds-pro": 3.0, "ds-think": 8.0, "groq": 0.5,
    "glm": 2.0, "qwen": 1.5, "kimi": 1.0,
}


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def classify_question(question: str) -> dict:
    """Semantic classification via MiniLM embedding + prototype cosine similarity.

    Falls back to lightweight keyword heuristics if embedder unavailable.
    """
    embedder = _get_embedder()

    if embedder is not None:
        q_emb = embedder.encode(question, convert_to_numpy=True).astype(np.float64)
        cat_scores = {}
        for cat, prototypes in CATEGORY_PROTOTYPES.items():
            proto_embs = [embedder.encode(p, convert_to_numpy=True).astype(np.float64)
                          for p in prototypes]
            cat_scores[cat] = float(max(_cosine_sim(q_emb, pe) for pe in proto_embs))

        best_cat = max(cat_scores, key=cat_scores.get)
        confidence = cat_scores[best_cat]
        if confidence < 0.3:
            best_cat = "knowledge"
            confidence = 0.3

        return {
            "category": best_cat,
            "confidence": round(confidence, 3),
            "embedding": q_emb,
            "category_scores": {k: round(v, 3) for k, v in
                               sorted(cat_scores.items(), key=lambda x: -x[1])},
        }
    else:
        return _fallback_classify(question)


def _fallback_classify(question: str) -> dict:
    """Keyword fallback — no external deps needed."""
    q = question.lower()
    scores = {
        "math": len(re.findall(r'(方程|积分|导数|极限|证明.*定理|矩阵|向量|几何|代数|'
                               r'微积分|微分|拉格朗日|求解|解方程|计算.*积分)', q)),
        "code": len(re.findall(r'(写.*代码|写.*函数|python|sql|算法|编程|实现.*程序|'
                               r'class |def |import )', q)),
        "logic": len(re.findall(r'(说谎|谁说|真话|假话|悖论|称重|称.*次|天平|蒙提霍尔|'
                                r'三个.*一个.*真|如果.*那么.*对|推理|嫌疑犯)', q)),
        "architecture": len(re.findall(r'(架构设计|系统设计|技术方案|选型|微服务|'
                                       r'分布式|高可用|设计一个.*系统)', q)),
        "writing": len(re.findall(r'(写作|论文|翻译|润色|写.*诗|写.*文章|'
                                  r'写一篇|帮我写)', q)),
        "knowledge": len(re.findall(r'(什么是|是谁|在哪里|为什么|定义|概念|'
                                    r'原理|历史|如何|区别)', q)),
    }
    best_cat = max(scores, key=scores.get)
    if max(scores.values()) == 0:
        best_cat = "knowledge"
    return {"category": best_cat, "confidence": 0.5,
            "embedding": None, "category_scores": scores}


def estimate_difficulty(question: str, category: str) -> float:
    base = {"math": 0.55, "logic": 0.50, "code": 0.40,
            "architecture": 0.45, "writing": 0.30, "knowledge": 0.20}.get(category, 0.35)
    length_bonus = min(0.2, len(question) / 2000 * 0.2)
    reasoning = len(re.findall(r'(证明|推导|prove|设计|design|优化|optimize|分析|analyze)',
                               question.lower()))
    reasoning_bonus = min(0.2, reasoning * 0.1)
    return min(1.0, base + length_bonus + reasoning_bonus)


def score_task(question: str) -> dict:
    """Entry point: classify question → produce routing features."""
    cls = classify_question(question)
    cat = cls["category"]
    difficulty = estimate_difficulty(question, cat)

    rules = CATEGORY_MODEL_RULES[cat]
    model_affinity = {
        m: 0.8 if rules["primary"] == m else 0.3
        for m in ["ds-pro", "ds-think", "glm", "groq", "qwen", "kimi"]
    }
    if difficulty < 0.35 and rules["primary"] in ("ds-pro", "ds-think"):
        model_affinity[rules.get("fallback", "groq")] = 0.7

    return {
        "category": cat, "difficulty": round(difficulty, 3),
        "confidence": cls["confidence"], "model_affinity": model_affinity,
        "category_scores": cls["category_scores"],
        "embedding": cls.get("embedding"), "length": len(question),
        "dims": {},
    }


# ═══════════════════════════════════════════════════════════════
# ROUTING — single-model, cost-aware
# ═══════════════════════════════════════════════════════════════

def decide_route(scores: dict) -> dict:
    """Single-model routing based on category + difficulty.

    Empirical finding (440q benchmark): multi-model collab does NOT
    improve accuracy. Same accuracy, 3.8x latency, 3-5x API cost.
    Therefore K=1 always.
    """
    cat = scores["category"]
    difficulty = scores["difficulty"]
    rules = CATEGORY_MODEL_RULES.get(cat, CATEGORY_MODEL_RULES["knowledge"])

    if difficulty < 0.2:
        model = "groq"  # trivial → cheapest
    elif difficulty < 0.35:
        model = rules.get("fallback", rules["primary"])  # easy → cheaper
    elif difficulty > 0.75:
        model = "ds-pro"  # very hard → strongest
    else:
        model = rules["primary"]  # normal → best for category

    estimated_cost = MODEL_COST.get(model, 0.001)
    estimated_latency = MODEL_LATENCY.get(model, 2.0)

    return {
        "action": "single", "model": model, "K": 1,
        "reason": (f"cat={cat} diff={difficulty:.2f} → {model} "
                   f"(est_cost=${estimated_cost:.4f}/1k, est_lat={estimated_latency}s)"),
        "estimated_cost": estimated_cost, "estimated_latency": estimated_latency,
        "category": cat, "difficulty": difficulty,
    }


# ═══════════════════════════════════════════════════════════════
# API CALL
# ═══════════════════════════════════════════════════════════════

def call(model_id, prompt, system=None, max_tok=2000, temp=0.3):
    """Call a model via OpenAI-compatible API."""
    name, getter = MODELS[model_id]
    client, model_name = getter()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = client.chat.completions.create(
        model=model_name, messages=msgs, temperature=temp, max_tokens=max_tok)
    return r.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════
# UNIFIED ROUTER — pluggable backends: semantic|math|deep|path
# ═══════════════════════════════════════════════════════════════

# Lazy-loaded router instances
_router_instance = None
_router_type = None

ROUTER_TYPES = {
    "semantic": "Rule-based category→model (default, zero training)",
    "math":     "JL projection + LSH + Discounted UCB + CUSUM (online learning)",
    "deep":     "3-layer MLP + MC Dropout + Thompson sampling (online learning)",
    "path":     "Bayesian Beta posterior + Adaptive forgetting + UCB (online learning)",
}


def get_router(router_type: str = "semantic"):
    """Get or initialize a router instance. Lazy-loaded, cached globally.

    Args:
        router_type: 'semantic' | 'math' | 'deep' | 'path'

    Returns:
        Router object with .route(question) → dict interface
    """
    global _router_instance, _router_type
    if _router_type == router_type and _router_instance is not None:
        return _router_instance

    if router_type == "semantic" or router_type not in ("math", "deep", "path"):
        _router_instance = SemanticRouter()
        _router_type = "semantic"
    elif router_type == "math":
        try:
            from engine.math_router import MathRouter
            _router_instance = MathRouterAdapter()
            _router_type = "math"
        except ImportError:
            _router_instance = SemanticRouter()
            _router_type = "semantic"
    elif router_type == "deep":
        try:
            from engine.deep_router import DeepRouter
            _router_instance = DeepRouterAdapter()
            _router_type = "deep"
        except ImportError:
            _router_instance = SemanticRouter()
            _router_type = "semantic"
    elif router_type == "path":
        try:
            from engine.path_learner import PathLearner
            _router_instance = PathLearnerAdapter()
            _router_type = "path"
        except ImportError:
            _router_instance = SemanticRouter()
            _router_type = "semantic"

    return _router_instance


class SemanticRouter:
    """Rule-based category→model routing. Zero training, instant setup."""
    router_type = "semantic"

    def route(self, question: str) -> dict:
        s = score_task(question)
        d = decide_route(s)
        return {**d, "category": s["category"], "confidence": s["confidence"]}

    def learn(self, question: str, model: str, reward: float):
        pass  # No learning — rules are static

    def stats(self) -> dict:
        return {"type": "semantic", "episodes": 0}


class MathRouterAdapter:
    """Adapter: MathRouter (JL+LSH+UCB+CUSUM) → unified route interface."""
    router_type = "math"

    def __init__(self):
        from engine.math_router import MathRouter
        self._r = MathRouter()

    def route(self, question: str) -> dict:
        s = score_task(question)
        features = {k: float(v) for k, v in s.get("dims", {}).items() if v}
        features["category"] = s["category"]
        features["difficulty"] = s["difficulty"]
        try:
            region_name, region_id, conf, diff = self._r.classify(features)
            return {
                "action": "single", "model": "ds-pro", "K": 1,
                "reason": f"MathRouter: region={region_name} conf={conf:.2f} diff={diff:.2f}",
                "category": s["category"], "confidence": conf,
            }
        except Exception:
            return SemanticRouter().route(question)

    def learn(self, question: str, model: str, reward: float):
        try:
            s = score_task(question)
            self._r.update(
                region=s["category"], model=model, category=s["category"],
                success=reward > 0.5, features=s.get("dims", {}))
        except Exception:
            pass

    def stats(self) -> dict:
        return {"type": "math", "paths": len(self._r.paths)}


class DeepRouterAdapter:
    """Adapter: DeepRouter (3-layer MLP + Thompson) → unified route interface."""
    router_type = "deep"

    def __init__(self):
        from engine.deep_router import DeepRouter
        self._r = DeepRouter()

    def route(self, question: str) -> dict:
        try:
            d = self._r.route(question)
            return {
                "action": "single", "model": d.get("primary_model", "ds-pro"),
                "K": 1, "reason": f"DeepRouter: MLP+Thompson",
                "predictions": d.get("model_scores", {}),
            }
        except Exception:
            return SemanticRouter().route(question)

    def learn(self, question: str, model: str, reward: float):
        try:
            self._r.learn(question, model, reward)
        except Exception:
            pass

    def stats(self) -> dict:
        return {"type": "deep", "episodes": self._r.episodes,
                "loss": getattr(self._r, "total_loss", 0)}


class PathLearnerAdapter:
    """Adapter: PathLearner (Bayesian + Adaptive Forgetting + UCB) → route interface."""
    router_type = "path"

    def __init__(self):
        from engine.path_learner import PathLearner
        self._r = PathLearner()

    def route(self, question: str) -> dict:
        s = score_task(question)
        try:
            model, conf = self._r.route(
                question_features=s.get("dims", {}),
                brainstem_region=s["category"],
                difficulty=s["difficulty"])
            return {
                "action": "single", "model": model, "K": 1,
                "reason": f"PathLearner: Bayesian UCB conf={conf:.2f}",
                "category": s["category"], "confidence": conf,
            }
        except Exception:
            return SemanticRouter().route(question)

    def learn(self, question: str, model: str, reward: float):
        try:
            s = score_task(question)
            pid = f"{s['category']}::{model}::general"
            from engine.path_learner import PathState
            ps = PathState(path_id=pid, region=s["category"],
                          model=model, category="general")
            self._r.update(ps, reward)
        except Exception:
            pass

    def stats(self) -> dict:
        return {"type": "path", "paths": len(getattr(self._r, "paths", {}))}


# ═══════════════════════════════════════════════════════════════

def execute_single(question, model="ds-pro"):
    """Call a single model and return the answer."""
    print(f"  {CC['B']}[{MODELS[model][0]}]{CC['R']} working...")
    return call(model, question)


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: execute_pipeline & execute_collab — removed per empirical finding:
# "Multi-model collaboration does NOT improve accuracy over single-model
#  baselines. Same accuracy, 3.8x latency, 3-5x API cost."
#  (440-question benchmark, 7 benchmarks, real API calls)
#
# These functions are preserved as reference for future research.
# They are NOT called by the production routing path (K=1 always).
# ═══════════════════════════════════════════════════════════════
#
# def execute_pipeline(question, steps):
#     """[ARCHIVED] Sequential multi-model pipeline."""
#     results = []
#     for i, s in enumerate(steps):
#         mid, role = s["model"], s.get("role", "")
#         r = call(mid, f"任务: {role}\n\n完整上下文: {question}", max_tok=2500)
#         results.append({"step": i+1, "model": mid, "role": role, "result": r})
#     if len(results) == 1:
#         return results[0]["result"]
#     parts = "\n\n".join(f"--- {r['step']} ({r['role']}) ---\n{r['result']}" for r in results)
#     return call("glm", f"整合:\n\n{question}\n\n{parts}",
#                 system="保留所有代码、SQL、数字。中文作答。", max_tok=3500)
#
# def execute_collab(question, models):
#     """[ARCHIVED] Parallel multi-model collaboration."""
#     import threading
#     results, lock = {}, threading.Lock()
#     def worker(mid):
#         try:
#             r = call(mid, question, max_tok=2500)
#             with lock: results[mid] = r
#         except Exception as e:
#             with lock: results[mid] = f"[ERR:{e}]"
#     threads = [threading.Thread(target=worker, args=(m,)) for m in models]
#     for t in threads: t.start()
#     for t in threads: t.join()
#     valid = {m:r for m,r in results.items() if not str(r).startswith("[ERR")}
#     if len(valid) == 0: return "[ERROR: all models failed]"
#     if len(valid) == 1: return list(valid.values())[0]
#     sections = "\n\n".join(f"=== {MODELS[m][0]} ===\n{valid[m]}" for m in models if m in valid)
#     return call("glm", f"综合{len(valid)}个模型:\n\n{question}\n\n{sections}",
#                 system="保留所有代码/SQL/数字。取各模型之精华。中文作答。", max_tok=4000)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="SynapseFlow LLM Router")
    ap.add_argument("question", nargs="*", help="Question text")
    ap.add_argument("--router", choices=["semantic","math","deep","path"],
                    default="semantic",
                    help="semantic=rule(default) | math=JL+LSH+UCB | deep=MLP+Thompson | path=Bayesian+Forgetting")
    ap.add_argument("--learn", type=float, default=None,
                    help="Simulated reward for learning (0.0-1.0)")
    ap.add_argument("--list-routers", action="store_true",
                    help="Show available routers")
    args = ap.parse_args()

    if args.list_routers:
        for k, v in ROUTER_TYPES.items():
            print(f"  {k:12s} — {v}")
        return

    question = " ".join(args.question) if args.question else (
        sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not question:
        print("Usage: python brain.py [--router semantic|math|deep|path] <question>",
              file=sys.stderr)
        sys.exit(1)

    # 1. Route via selected backend
    router = get_router(args.router)
    decision = router.route(question)
    model = decision.get("model", "ds-pro")

    # Build display info from whatever the router returns
    category = decision.get("category", "?")
    confidence = decision.get("confidence", 0.5)
    difficulty = decision.get("difficulty",
        decision.get("estimated_cost", 0.3))

    print(f"\n{CC['W']}{'='*60}{CC['R']}")
    print(f"  {CC['W']}SynapseFlow — {router.router_type.upper()} Router{CC['R']}")
    print(f"  Category: {category} (conf={confidence:.2f})")
    print(f"  {decision.get('reason', '')}")
    print(f"{CC['W']}{'='*60}{CC['R']}")

    # 2. Execute
    answer = execute_single(question, model)

    # 3. Learn (if --learn flag provided)
    if args.learn is not None:
        router.learn(question, model, args.learn)
        print(f"  {CC['Y']}[LEARN] {model} reward={args.learn}{CC['R']}")

    # 4. Track cost
    est_tokens = max(10, len(question + answer) // 4)
    est_cost = MODEL_COST.get(model, 0.001) * est_tokens / 1000

    print(f"\n{CC['W']}{'='*60}{CC['R']}")
    print(answer[:3000] if len(answer) > 3000 else answer)
    print(f"\n{CC['D']}[{model}] router={router.router_type} "
          f"cat={category} "
          f"est_tok={est_tokens} "
          f"est_cost=${est_cost:.6f}{CC['R']}\n")

    # Log
    from datetime import datetime
    log = {
        "timestamp": datetime.now().isoformat(),
        "question_len": len(question),
        "router": router.router_type,
        "category": category,
        "model": model,
        "estimated_cost": round(est_cost, 6),
        "estimated_tokens": est_tokens,
    }
    log_dir = Path.home() / ".synapseflow" / "brain"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "decision-log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# BELOW: ARCHIVED — Neuro-synaptic mechanisms implemented but NOT wired into
# the production routing path. Preserved as reference implementations.
#
# Why not used:
#   1. No proven benefit over simple category→model rules (440q benchmark)
#   2. SPSA gradient estimate is NOT unbiased (one-eval variant violates Spall 1992)
#   3. 36-parameter space (6 models × 6 categories) too small for STDP+BCM+FEP convergence
#   4. Alpha oscillation / Kuramoto operate at wrong timescale (ms vs episodes)
#   5. Multi-model collaboration: same accuracy, 3.8x latency, 3-5x cost
#
# These are kept for:
#   - Paper reproduction reference (neuro_runner.py wires them independently)
#   - Future research if scale increases beyond current n=36
#   - Honest documentation of what was tried
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: SPSA — Simultaneous Perturbation Stochastic Approximation
# Spall, IEEE Trans. Automatic Control 37(3):332-341 (1992)
#
# PROBLEM: The "one-eval SPSA" variant used J_plus=J (current) and
# J_minus=last_J (EMA of past). This violates SPSA's requirement:
#   gradient = (J(θ+cΔ) - J(θ-cΔ)) / (2cΔ)
# Both evaluations must be AT THE SAME θ with independent perturbations.
# Using historical EMA as J_minus produces biased, non-convergent updates.
#
# CORRECT SPSA requires TWO API calls per step (evaluate θ+cΔ AND θ-cΔ),
# which doubles the already-high cost of multi-model routing.
# ═══════════════════════════════════════════════════════════════
#
# SPSA_STATE = {"k": 0, "last_J": 0.5, "a": 0.01, "c": 0.05, "A": 50}
#
# def spsa_step(was_correct, dims):
#     """[ARCHIVED] SPSA online hyperparameter optimization — NOT unbiased."""
#     s = SPSA_STATE
#     s["k"] += 1; k = s["k"]
#     a_k = s["a"] / (s["A"] + k) ** 0.602
#     c_k = s["c"] / k ** 0.101
#     J = 1.0 if was_correct else 0.0
#     s["last_J"] = 0.9 * s["last_J"] + 0.1 * J
#     param_keys = ["stdp_A_plus","stdp_A_minus","stdp_tau_plus","stdp_tau_minus",
#                   "bcm_alpha","lateral_alpha","prune_threshold","prune_rounds"]
#     delta = {k: random.choice([-1, 1]) for k in param_keys}
#     orig = {k: PARAMS[k] for k in param_keys}
#     for kk in param_keys:
#         PARAMS[kk] = max(0.001, orig[kk] + c_k * delta[kk])
#     _refresh_params()
#     J_plus = J  # ← BUG: should be independent evaluation
#     J_minus = s["last_J"]  # ← BUG: historical EMA, not J(θ-cΔ)
#     for kk in param_keys:
#         gradient = (J_plus - J_minus) / (2 * c_k * delta[kk] + 1e-8)  # biased!
#         PARAMS[kk] = max(0.001, orig[kk] - a_k * gradient)
#     _refresh_params()


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: STDP — Spike-Timing-Dependent Plasticity
# Song, Miller & Abbott, Nature Neuroscience 3:919-926 (2000)
#
# STDP weight update per (model, category) pair:
#   Δw = +A⁺·exp(-Δt/τ⁺) if correct   (LTP: causal pre→post)
#   Δw = -A⁻·exp(-Δt/τ⁻) if incorrect (LTD: anti-causal)
#
# NOTES: Reasonable for online learning but operates at episode timescale
# (seconds/minutes) vs biological millisecond scale. Mathematical guarantees
# of STDP (stability, convergence) depend on continuous spike timing and
# 10⁴+ synapses — not applicable at n=36 discrete events.
# ═══════════════════════════════════════════════════════════════
#
# STDP_A_PLUS, STDP_A_MINUS = 0.15, 0.10
# STDP_TAU_PLUS, STDP_TAU_MINUS = 5.0, 3.0
#
# def stdp_update(weight, model_category, correct, round_num):
#     """[ARCHIVED] Pair-based STDP weight update."""
#     cat = model_category
#     last_c = cat.get("last_correct", 0)
#     last_w = cat.get("last_wrong", 0)
#     if correct:
#         dt = round_num - last_c if last_c > 0 else 1
#         dw = STDP_A_PLUS * math.exp(-dt / STDP_TAU_PLUS)
#         cat["last_correct"] = round_num
#     else:
#         dt = round_num - last_w if last_w > 0 else 1
#         dw = -STDP_A_MINUS * math.exp(-dt / STDP_TAU_MINUS)
#         cat["last_wrong"] = round_num
#     return max(-1.0, min(1.0, weight + dw))


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: BCM — Bienenstock-Cooper-Munro Sliding Threshold
# Bienenstock, Cooper & Munro, J Neuroscience 2(1):32-48 (1982)
#
# Key idea: LTP/LTD threshold slides with neuron activity history.
#   If y > θ_M → LTP (above-average → strengthen)
#   If y < θ_M → LTD (below-average → weaken)
#   θ_M = E[y²] (moving average of squared activity)
#
# NOTES: Designed for rate-based neurons with continuous activity.
# Applied to binary (correct/wrong) outcomes at n=36 — the sliding
# threshold degenerates to a simple accuracy comparison.
# ═══════════════════════════════════════════════════════════════
#
# BCM_ALPHA = 0.01
#
# def bcm_threshold_update(cat, round_num):
#     """[ARCHIVED] BCM sliding threshold for plasticity modulation."""
#     rate = cat.get("firing_rate", 0.5)
#     threshold = rate ** 2  # θ_M = E[y²]
#     cat["bcm_threshold"] = threshold
#     recent_acc = (round_num - cat.get("last_wrong", 0)) / max(1,
#         round_num - cat.get("last_correct", 0) + round_num - cat.get("last_wrong", 0))
#     cat["firing_rate"] = rate + BCM_ALPHA * (recent_acc - rate)
#     return threshold


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: FEP — Free Energy Principle Temporal Memory
# Friston, Nature Reviews Neuroscience 11:127-138 (2010)
# Wong, R. "Affinity Is Not Enough" arXiv:2605.00604 (2025)
#
# Three mechanisms from Friston 2010 Eqs.1,3,4:
#   β(t): LIF membrane potential accumulating routing context
#   Π(t): per-expert inverse variance of prediction error
#   Anticipatory bonus = β(t) × Π(t) (super-additive)
#
# NOTES: The FEP implementation uses Beta-Bernoulli conjugate updates
# which is just Bayesian bandit. The "124x improvement" claim from
# Wong (2025) was measured on domain-boundary routing with large
# expert pools — not applicable to n=6 models.
# ═══════════════════════════════════════════════════════════════
#
# FEP_STATE = {}
#
# def fep_temporal_memory(model_name, correct, round_num):
#     """[ARCHIVED] β: LIF membrane potential for routing context."""
#     if model_name not in FEP_STATE:
#         FEP_STATE[model_name] = {"beta": 0.0, "precision": 1.0,
#                                   "pred_error_ema": 0.5, "last_round": round_num}
#     s = FEP_STATE[model_name]
#     dt = max(1, round_num - s["last_round"])
#     tau = 10.0
#     signal = 1.0 if correct else -0.2
#     s["beta"] = s["beta"] + dt * (-s["beta"]/tau + 0.3 * signal)
#     s["last_round"] = round_num
#     return s["beta"]
#
# def fep_precision_gate(model_name, correct):
#     """[ARCHIVED] Π: per-expert inverse variance of prediction error."""
#     if model_name not in FEP_STATE:
#         FEP_STATE[model_name] = {"beta": 0.0, "precision": 1.0, "pred_error_ema": 0.5}
#     s = FEP_STATE[model_name]
#     error = 0.0 if correct else 1.0
#     s["pred_error_ema"] = 0.9 * s["pred_error_ema"] + 0.1 * error
#     s["precision"] = 1.0 / (s["pred_error_ema"] + 0.01)
#     return s["precision"]


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: Alpha Oscillation Inhibition (Duecker et al., PLOS CB 2024)
# + Kuramoto Exploration (Front Comput Neurosci 2025)
# + Temporally Layered Architecture (Patel & Sejnowski, Neural Comp 2024)
#
# All three operate on biologically-relevant timescales (10Hz alpha,
# 5Hz theta, millisecond membrane dynamics) that don't map to
# episode-level LLM API calls (seconds to minutes).
#
# Alpha filter: h_j = σ(z_j - c·r_j - m·sin(2π·10·t))
# Kuramoto: dθ/dt = ω + (K/N)Σ sin(θ_j - θ_i)
# TLA: fast layer (heuristic, low energy) vs slow layer (deliberative)
#
# These add computational overhead (O(n²) per filter) with no empirical
# benefit at the current routing scale.
# ═══════════════════════════════════════════════════════════════
#
# ALPHA_STATE, THETA_STATE, TLA_STATE = {}, {"phase": 0.0, "K": 1.0}, {...}
# (implementations omitted — see git history for full code)


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: Synaptic Pruning + Homeostatic Normalization
#
# Pruning: P(prune) = 1/(1+exp(-k·(t-t₀))) where t=consecutive failures
# Homeostatic: Σ|w| normalized to constant per model
#
# NOTES: At n=36, pruning is equivalent to "if weight < threshold: ban".
# The sigmoidal decay curve adds unjustified complexity. Homeostatic
# normalization is just softmax at this scale.
# ═══════════════════════════════════════════════════════════════
#
# PRUNE_THRESHOLD, PRUNE_ROUNDS = -0.3, 5
# LATERAL_ALPHA = 0.03
# def lateral_inhibition(weights_by_model, winner, category): ...
# def homeostatic_normalize(weights_by_model): ...
# def prune_check(weights_by_model, category): ...


# ═══════════════════════════════════════════════════════════════
# ARCHIVED: 7-Timescale Memory System
#
# MEMORY_TYPES with distinct τ: STF(0.2s), STD(0.5s), E-LTP(30min),
# L-LTP(500h), E-LTD(20min), L-LTD(400h), Meta(100h)
#
# NOTES: 7 timescales for 36 parameters is over-parameterized.
# The mathematical guarantees of multi-timescale plasticity
# (Benna & Fusi, Nature Neuroscience 2016) require continuous
# synapses and 10⁴+ parameters for stable cascade dynamics.
# At n=36, a single EWMA (τ≈5) achieves equivalent behavior.
# ═══════════════════════════════════════════════════════════════
#
# MEMORY_TYPES = {
#     "STF": {"tau": 0.2, "amp": 0.05, ...},
#     "STD": {"tau": 0.5, "amp": -0.03, ...},
#     "E-LTP": {"tau": 30.0, "amp": 0.10, ...},
#     "L-LTP": {"tau": 500.0, "amp": 0.20, ...},
#     ...
# }
