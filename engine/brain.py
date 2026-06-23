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

def execute_single(question, model="ds-pro"):
    """Call a single model and return the answer."""
    print(f"  {CC['B']}[{MODELS[model][0]}]{CC['R']} working...")
    return call(model, question)


def execute_pipeline(question, steps):
    """
    [EXPERIMENTAL] Sequential multi-model pipeline.
    NOT used in production routing — no proven accuracy benefit.
    """
    results = []
    for i, s in enumerate(steps):
        mid, role = s["model"], s.get("role", "")
        print(f"  {CC['B']}[{MODELS[mid][0]}]{CC['R']} Step {i+1}: {role}...")
        r = call(mid, f"任务: {role}\n\n完整上下文: {question}", max_tok=2500)
        results.append({"step": i+1, "model": mid, "role": role, "result": r})
    if len(results) == 1:
        return results[0]["result"]
    parts = "\n\n".join(f"--- Step {r['step']} ({r['role']}) ---\n{r['result']}"
                        for r in results)
    print(f"  {CC['B']}[GLM]{CC['R']} Combining...")
    return call("glm", f"整合以下分步结果为一篇连贯回答:\n\n{question}\n\n{parts}",
                system="保留所有代码、SQL、数字。中文作答。", max_tok=3500)


def execute_collab(question, models):
    """
    [EXPERIMENTAL] Parallel multi-model collaboration.
    NOT used in production routing — same accuracy, 3-5x cost.
    """
    import threading
    results = {}
    lock = threading.Lock()

    def worker(mid):
        try:
            r = call(mid, question, max_tok=2500)
            with lock:
                results[mid] = r
        except Exception as e:
            with lock:
                results[mid] = f"[ERR:{e}]"

    threads = [threading.Thread(target=worker, args=(m,)) for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    valid = {m: r for m, r in results.items() if not str(r).startswith("[ERR")}
    if len(valid) == 0:
        return "[ERROR: all models failed]"
    if len(valid) == 1:
        return list(valid.values())[0]

    sections = "\n\n".join(f"=== {MODELS[m][0]} ===\n{valid[m]}"
                           for m in models if m in valid)
    print(f"  {CC['B']}[DS-PRO]{CC['R']} Synthesizing {len(valid)} answers...")
    return call("glm", f"综合以下{len(valid)}个模型的独立回答，取长补短:\n\n"
                f"{question}\n\n{sections}",
                system="保留所有代码/SQL/数字。取各模型之精华。中文作答。", max_tok=4000)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not question:
        print("Usage: python brain.py <question>", file=sys.stderr)
        sys.exit(1)

    # 1. Classify
    scores = score_task(question)
    decision = decide_route(scores)

    print(f"\n{CC['W']}{'='*60}{CC['R']}")
    print(f"  {CC['W']}SynapseFlow Router{CC['R']}")
    print(f"  Category: {scores['category']} (conf={scores['confidence']:.2f})")
    print(f"  Difficulty: {scores['difficulty']:.2f}")
    print(f"  {decision['reason']}")
    print(f"{CC['W']}{'='*60}{CC['R']}")

    # 2. Execute single model
    model = decision["model"]
    answer = execute_single(question, model)

    # 3. Track cost
    est_tokens = max(10, len(question + answer) // 4)
    est_cost = MODEL_COST.get(model, 0.001) * est_tokens / 1000

    # Output
    print(f"\n{CC['W']}{'='*60}{CC['R']}")
    print(answer[:3000] if len(answer) > 3000 else answer)
    print(f"\n{CC['D']}[{model}] cat={scores['category']} "
          f"diff={scores['difficulty']:.2f} "
          f"est_tok={est_tokens} "
          f"est_cost=${est_cost:.6f}{CC['R']}\n")

    # Log
    from datetime import datetime
    log = {
        "timestamp": datetime.now().isoformat(),
        "question_len": len(question),
        "category": scores["category"],
        "difficulty": scores["difficulty"],
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
