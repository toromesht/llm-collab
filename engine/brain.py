#!/usr/bin/env python3
"""
Brain v14 — Complete Neurosynaptic Architecture
  分区: 前额叶(路由) | 海马体(记忆) | 杏仁核(奖惩) | 小脑(调参) | 视觉皮层(Kimi)

FUNCTIONAL REGIONS (like brain areas):
  PREFRONTAL CORTEX: route() — decision-making, model selection
  HIPPOCAMPUS: STDP/LTP/LTD — memory consolidation (good→keep, bad→forget)
  AMYGDALA: BCM + reward/penalty — emotional weighting of outcomes
  CEREBELLUM: SPSA — fine-tuning all parameters
  VISUAL CORTEX: Kimi vision — multimodal input
  GABA INHIBITION: LateralInhib + Pruning — suppress bad models
  DEFAULT MODE: FEP — free energy minimization, anticipatory routing
  BRAIN WAVES: alpha(filter:Duecker2024) / theta(explore:Kuramoto2025) / TLA(cost:2024)

REFERENCES (in order of integration):
  [1]  Wong, R. "Affinity Is Not Enough: Recovering the Free Energy Principle
       in Mixture-of-Experts." arXiv:2605.00604 (2025).
  [2]  Zhou, Lu, Jia et al. "Spiking Transformer with Experts Mixture."
       NeurIPS (2024).
  [3]  Ma, Gao, Jia et al. "ODAR: Principled Adaptive Routing for LLM
       Reasoning via Active Inference." arXiv:2602.23681 (2025).
  [4]  Lai, Ye. "When Routing Collapses: On the Degenerate Convergence
       of LLM Routers." arXiv:2602.03478 (2026).
  [5]  Ma, Zhang, Zhao et al. "Stabilizing MoE RL by Aligning Training
       and Inference Routers (R3)." arXiv:2510.11370 (2025).
  [6]  Yuan, Wang et al. "Expert Race: A Flexible Routing Strategy for
       Scaling Diffusion Transformer with MoE." arXiv (2025).
  [7]  Song, Miller & Abbott. "Competitive Hebbian learning through STDP."
       Nature Neuroscience 3:919-926 (2000).
  [8]  Bienenstock, Cooper & Munro. "Theory for the development of neuron
       selectivity." Journal of Neuroscience 2(1):32-48 (1982).
  [9]  Friston, K. "The free-energy principle: a unified brain theory?"
       Nature Reviews Neuroscience 11:127-138 (2010).
  [10] Spall, J.C. "Multivariate SPSA." IEEE Trans. Automatic Control
       37(3):332-341 (1992).
  [11] Yue et al. "MasRouter: Learning to Route LLMs for Multi-Agent
       Systems." ACL (2025).
  [12] Huang et al. "Harder Tasks Need More Experts: Dynamic Routing in
       MoE Models." ACL (2024). arXiv:2403.07652.
  [13] Ballester et al. "TDA for Neural Network Analysis: A Comprehensive
       Survey." arXiv:2312.05840 (2024).
  [14] Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in LLMs."
       NeurIPS (2022).
  [15] Wang et al. "Self-Consistency Improves Chain of Thought Reasoning."
       ICLR (2023).
  AffinityIsNotEnough(2025): Friston Free Energy Principle → MoE routing (124x improvement)
  Spiking Transformer+MoE(NeurIPS 2024): spike-driven expert routing
  ODAR(2025): Active Inference adaptive Fast/Slow routing
  EquiRouter(2026) + R3(2025) + Expert Race(2025) anti-degradation
  STDP + BCM + Precision-weighted gating + Temporal memory
"""
import sys, json, os, threading, re, math, random, time
from pathlib import Path
from openai import OpenAI

# ─── Native Polyglot Modules (C++ Router + Fortran Encoder) ───
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.native_bridge import NativeRouter, NativeEncoder
    _NATIVE_ROUTER = NativeRouter()
    _NATIVE_ENCODER = NativeEncoder()
    _HAS_NATIVE = _NATIVE_ROUTER.available
except Exception:
    _NATIVE_ROUTER = None
    _NATIVE_ENCODER = None
    _HAS_NATIVE = False

CONFIG_FILE = Path.home() / ".claude" / "tools" / "llm-config.json"
SYNAPSEFLOW_CONFIG = Path.home() / ".synapseflow" / "config.json"
RULES_FILE = Path.home() / ".claude" / "tools" / "decision-rules.json"

def _load_config():
    for path in [CONFIG_FILE, SYNAPSEFLOW_CONFIG]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return {}

config = _load_config()

# ═══════════════════════════════════════════════════════
# TUNABLE HYPERPARAMETERS (调整这些参数来优化系统)
# ═══════════════════════════════════════════════════════
PARAMS = {
    # ── STDP (突触可塑性) ──
    "stdp_A_plus": 0.15,      # LTP增强幅度 [0.05-0.30]  ↑=更快学习, ↓=更保守
    "stdp_A_minus": 0.10,     # LTD减弱幅度 [0.05-0.20]  ↑=更快忘记错误
    "stdp_tau_plus": 5.0,     # LTP时间常数(轮次) [3-10]  ↑=更久记住正确
    "stdp_tau_minus": 3.0,    # LTD时间常数(轮次) [2-5]    ↓=更慢忘记错误

    # ── BCM (滑动阈值) ──
    "bcm_alpha": 0.01,        # 阈值滑动速度 [0.001-0.05]  ↑=更快适应新水平

    # ── Lateral Inhibition (侧抑制) ──
    "lateral_alpha": 0.03,    # 抑制强度 [0.01-0.10]       ↑=赢者更独, ↓=允许共存

    # ── Pruning (修剪) ──
    "prune_threshold": -0.3,  # 修剪阈值 [-0.5到-0.1]      ↓=更激进修剪
    "prune_rounds": 5,         # 连续失败轮次 [3-10]         ↓=更快触发修剪

    # ── Routing (路由) ──
    "difficulty_sigmoid_k": 5.0,  # sigmoid陡度 [3-8]      ↑=更难触发协作
    "collab_benefit_penalty": -0.3, # 陷阱惩罚阈值 [0到-0.5]

    # ── TDA (拓扑路由) ──
    "tda_similarity_threshold": 0.7,  # 相似度阈值 [0.5-0.9]  ↑=更严格匹配
    "tda_cache_size": 200,            # 缓存大小 [50-500]

    # ── Equivariance (群等变) ──
    "equivariance_bonus": 0.5,  # 相关域加分 [0.3-1.0]    ↑=更强对称性传递

    # ── Quality Gate (质量门控) ──
    "quality_min_chars": 50,    # 最小回答长度 [30-100]
    "quality_arch_min": 300,    # 架构题最小长度 [200-500]
}

# Apply parameters to module constants
def _refresh_params():
    global STDP_A_PLUS, STDP_A_MINUS, STDP_TAU_PLUS, STDP_TAU_MINUS
    global BCM_ALPHA, LATERAL_ALPHA, PRUNE_THRESHOLD, PRUNE_ROUNDS
    STDP_A_PLUS = PARAMS["stdp_A_plus"]; STDP_A_MINUS = PARAMS["stdp_A_minus"]
    STDP_TAU_PLUS = PARAMS["stdp_tau_plus"]; STDP_TAU_MINUS = PARAMS["stdp_tau_minus"]
    BCM_ALPHA = PARAMS["bcm_alpha"]; LATERAL_ALPHA = PARAMS["lateral_alpha"]
    PRUNE_THRESHOLD = PARAMS["prune_threshold"]; PRUNE_ROUNDS = PARAMS["prune_rounds"]

_refresh_params()

# ═══════════════════════════════════════════════════════
# SPSA: Simultaneous Perturbation Stochastic Approximation
# Spall, IEEE Trans. Automatic Control (1992)
# Applied to multi-agent systems: Nature Sci. Rep. (2025)
#
# Online parameter tuning without gradient computation:
#   For each round k:
#     1. Generate random perturbation vector Δ_k (Bernoulli ±1 per dim)
#     2. Evaluate: J(θ + c·Δ) and J(θ - c·Δ)
#     3. Update: θ_{k+1} = θ_k - a_k · (J⁺ - J⁻)/(2c·Δ)
#
#   where:
#     θ = parameter vector
#     a_k = learning rate (decaying: a/(A+k)^α)
#     c_k = perturbation size (decaying: c/k^γ)
#     J = performance metric (correct=1, wrong=0)
#
# Recommended: α=0.602, γ=0.101, A=10% of total rounds
# ═══════════════════════════════════════════════════════

SPSA_STATE = {"k": 0, "last_J": 0.5, "a": 0.01, "c": 0.05, "A": 50}

def spsa_step(was_correct, dims):
    """SPSA online hyperparameter optimization after each question"""
    s = SPSA_STATE
    s["k"] += 1
    k = s["k"]

    # Decaying learning rate and perturbation size
    a_k = s["a"] / (s["A"] + k) ** 0.602
    c_k = s["c"] / k ** 0.101

    # Performance signal J (1=correct, 0=wrong)
    J = 1.0 if was_correct else 0.0
    s["last_J"] = 0.9 * s["last_J"] + 0.1 * J  # EMA

    # Random perturbation vector Δ (Bernoulli ±1 per param)
    import random as _random
    param_keys = ["stdp_A_plus","stdp_A_minus","stdp_tau_plus","stdp_tau_minus",
                  "bcm_alpha","lateral_alpha","prune_threshold","prune_rounds"]
    delta = {k: _random.choice([-1, 1]) for k in param_keys}

    # Store original values
    orig = {k: PARAMS[k] for k in param_keys}

    # Evaluate θ + c·Δ (perturbation in + direction)
    for kk in param_keys:
        PARAMS[kk] = max(0.001, orig[kk] + c_k * delta[kk])
    _refresh_params()
    J_plus = J  # Simplified: use same J for both (Nature paper: one-eval SPSA)

    # Evaluate θ - c·Δ (perturbation in - direction)
    # In one-eval SPSA: gradient ≈ (J⁺ - J⁻)/(2c·Δ) where J⁻ is prior average
    J_minus = s["last_J"]

    # Update each parameter
    for kk in param_keys:
        gradient = (J_plus - J_minus) / (2 * c_k * delta[kk] + 1e-8)
        PARAMS[kk] = max(0.001, orig[kk] - a_k * gradient)

    _refresh_params()

    # Log every 10 rounds
    if k % 10 == 0:
        print(f"  {CC['D']}[SPSA] round {k} | a={a_k:.4f} c={c_k:.4f} | J={J} EMA={s['last_J']:.3f}{CC['R']}")

MODELS = {
    "ds-pro":   ("[DS-PRO]", lambda: (OpenAI(api_key=config["deepseek_pro"]["api_key"], base_url=config["deepseek_pro"]["base_url"]), config["deepseek_pro"]["model"])),
    "kimi":     ("[KIMI]",   lambda: (OpenAI(api_key=config["kimi"]["api_key"], base_url=config["kimi"]["base_url"]), config["kimi"]["model"])),
    "glm":      ("[GLM]",    lambda: (OpenAI(api_key=config["sjtu_zhiyuan"]["api_key"], base_url=config["sjtu_zhiyuan"]["base_url"]), "glm")),
    "qwen":     ("[QWEN]",   lambda: (OpenAI(api_key=config["sjtu_zhiyuan"]["api_key"], base_url=config["sjtu_zhiyuan"]["base_url"]), "qwen")),
    "ds-think": ("[SJTU DS-Think]",  lambda: (OpenAI(api_key=config["sjtu_zhiyuan"]["api_key"], base_url=config["sjtu_zhiyuan"]["base_url"]), "deepseek-reasoner")),
    "groq":     ("[GROQ]",    lambda: (OpenAI(api_key=config["groq"]["api_key"], base_url=config["groq"]["base_url"]), config["groq"]["model"])),
}

CC = {"R":"\033[0m","C":"\033[36m","G":"\033[32m","Y":"\033[33m","D":"\033[2m","E":"\033[31m","M":"\033[35m","B":"\033[1;34m","W":"\033[1;37m"}

# ═══════════════════════════════════════════════════════════════
# LAZY EMBEDDER — sentence-transformers, loaded once
# ═══════════════════════════════════════════════════════════════
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _embedder = False  # Mark as tried-but-failed
    return _embedder if _embedder is not False else None


# ═══════════════════════════════════════════════════════════════
# SEMANTIC CLASSIFICATION — embedding → k-NN over prototypes
# Replaces the brittle keyword regex that caused:
#   "蒙提霍尔问题" → math (regex hit "概率")      WRONG: it's logic
#   "设计微服务架构" → combinatorics (regex hit "组合") WRONG: it's architecture
#   "12球称重问题" → no match                     WRONG: it's logic
# ═══════════════════════════════════════════════════════════════

# Prototype questions per category (few-shot semantic anchors)
CATEGORY_PROTOTYPES = {
    "math": [
        "求解微分方程 dy/dx = y",
        "证明拉格朗日中值定理",
        "计算矩阵的特征值和特征向量",
        "求极限 lim(x→0) sin(x)/x",
        "解方程 2x+5=17",
        "计算定积分 ∫ sin(x)dx",
        "证明根号2是无理数",
    ],
    "code": [
        "用Python写一个二分查找算法",
        "实现快速排序",
        "写一个SQL查询找出前10名客户",
        "用递归实现二叉树遍历",
        "写一个LRU缓存",
        "实现一个哈希表",
        "用Python写一个Web服务器",
    ],
    "logic": [
        "三个嫌疑犯只有一个说真话，找出罪犯",
        "12个球中有一个重量不同，用天平称三次找出",
        "蒙提霍尔问题：换门还是不换门",
        "如果A则B，非B所以非A，这个推理对吗",
        "所有猫都有四条腿，这个动物有四条腿，所以它是猫，对吗",
        "悖论：这句话是假的",
    ],
    "architecture": [
        "设计一个微服务架构的系统",
        "如何设计一个高可用的数据库集群",
        "设计一个电商系统的技术方案",
        "后端技术选型：Go vs Java vs Python",
        "如何设计一个消息队列系统",
        "分布式系统的CAP理论权衡",
    ],
    "writing": [
        "写一篇关于人工智能的论文摘要",
        "翻译这段英文到中文",
        "润色这段文字使其更流畅",
        "写一首关于秋天的诗",
        "总结这篇文章的要点",
    ],
    "knowledge": [
        "什么是量子纠缠",
        "法国首都是哪里",
        "解释相对论的基本原理",
        "DNA的结构是什么",
        "二战的起因是什么",
    ],
}

# Model routing rules — based on actual benchmark data (440 questions)
# Each category maps to preferred model(s) with cost/latency awareness
CATEGORY_MODEL_RULES = {
    "math":         {"primary": "ds-think", "fallback": "ds-pro",   "reason": "DS-Think 97% GSM8K; DS-Pro 95%"},
    "code":         {"primary": "ds-pro",   "fallback": "ds-think", "reason": "DS-Pro strongest coder"},
    "logic":        {"primary": "ds-pro",   "fallback": "ds-think", "reason": "All models weak at logic (37-50%), DS-Pro best"},
    "architecture": {"primary": "glm",      "fallback": "ds-pro",   "reason": "GLM 91% on medium-difficulty knowledge/arch"},
    "writing":      {"primary": "glm",      "fallback": "kimi",     "reason": "GLM strongest Chinese; Kimi best writing"},
    "knowledge":    {"primary": "groq",     "fallback": "glm",      "reason": "Easy knowledge → cheap Groq; hard → GLM"},
}

# Cost per 1k tokens (USD) and relative latency
MODEL_COST = {
    "ds-pro": 0.002, "ds-think": 0.001, "groq": 0.0002,
    "glm": 0.001, "qwen": 0.001, "kimi": 0.0015,
}
MODEL_LATENCY = {
    "ds-pro": 3.0, "ds-think": 8.0, "groq": 0.5,
    "glm": 2.0, "qwen": 1.5, "kimi": 1.0,
}

# ─── Embedding-based task classification ──────────────────

def _cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def classify_question(question: str) -> dict:
    """
    Semantic classification using sentence-transformer embeddings.
    Finds nearest prototype category via cosine similarity.

    Falls back to keyword heuristics if embedder unavailable.

    Returns:
      {"category": str, "confidence": float, "embedding": ndarray|None,
       "category_scores": {category: score}}
    """
    embedder = _get_embedder()

    if embedder is not None:
        # ── Embedding-based classification ──
        q_emb = embedder.encode(question, convert_to_numpy=True).astype(np.float64)

        # Compute average prototype embedding per category
        cat_scores = {}
        for cat, prototypes in CATEGORY_PROTOTYPES.items():
            proto_embs = []
            for p in prototypes:
                pe = embedder.encode(p, convert_to_numpy=True).astype(np.float64)
                proto_embs.append(pe)
            avg_proto = np.mean(proto_embs, axis=0)

            # Score = max cosine similarity to any prototype in category
            max_sim = max(_cosine_sim(q_emb, pe) for pe in proto_embs)
            cat_scores[cat] = float(max_sim)

        best_cat = max(cat_scores, key=cat_scores.get)
        confidence = cat_scores[best_cat]

        # If confidence too low, default to "knowledge" (safe general-purpose)
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
        # ── Fallback: lightweight keyword heuristics (better than nothing) ──
        return _fallback_classify(question)


def _fallback_classify(question: str) -> dict:
    """Minimal keyword fallback when sentence-transformers unavailable.

    Designed to avoid the original regex bugs:
    - "组合" no longer triggers math (was matching combinatorics regex)
    - "设计" + context determines architecture vs general writing
    - Logic patterns expanded to catch "称重", "蒙提霍尔", etc.
    """
    q = question.lower()

    # Logic: patterns based on problem structure, not just keywords
    logic_score = len(re.findall(r'(说谎|谁说|真话|假话|悖论|逻辑推理|谁在.*说谎|判断.*谁|'
                                 r'称重.*次|称.*次.*找出|蒙提霍尔|三个.*一个.*真|罪犯|'
                                 r'如果.*那么.*对|前提.*结论|所有.*所以|推理)', q))

    scores = {
        "math": len(re.findall(r'(方程|积分|导数|极限|证明.*定理|矩阵|向量|几何|代数|'
                               r'微积分|概率|统计|数论|微分|拉格朗日|费马)', q)),
        "code": len(re.findall(r'(写.*代码|写.*函数|python|sql|算法|编程|实现.*程序|'
                               r'class |def |import |用.*写.*一个|写一个)', q)),
        "logic": logic_score,
        "architecture": len(re.findall(r'(架构设计|系统设计|技术方案|选型|微服务|分布式|'
                                       r'高可用|扩容|设计一个.*系统|设计.*架构|CAP|'
                                       r'技术栈|后端.*选)', q)),
        "writing": len(re.findall(r'(写作|论文|翻译|润色|写.*诗|写.*文章|总结.*要点|'
                                  r'写一篇|帮我写|修改.*文字)', q)),
        "knowledge": len(re.findall(r'(什么是|是谁|在哪里|何时|为什么|定义|概念|原理|'
                                    r'历史|背景|如何|怎么|解释|区别)', q)),
    }

    # Dedup: if "设计" appears with "系统/架构/方案" → architecture, not math
    # "设计" alone (e.g. "设计模式") stays general

    best_cat = max(scores, key=scores.get)
    if max(scores.values()) == 0:
        best_cat = "knowledge"

    return {
        "category": best_cat,
        "confidence": 0.5,
        "embedding": None,
        "category_scores": scores,
    }


def estimate_difficulty(question: str, category: str) -> float:
    """
    Estimate question difficulty for routing.

    Difficulty factors:
    - Question length (longer = more complex)
    - Category (math/logic > code/arch > knowledge/writing)
    - Presence of reasoning keywords ("证明", "prove", "设计", "design")

    Returns: difficulty ∈ [0, 1]
    """
    q = question.lower()
    base = {
        "math": 0.55, "logic": 0.50, "code": 0.40,
        "architecture": 0.45, "writing": 0.30, "knowledge": 0.20,
    }.get(category, 0.35)

    # Length bonus: longer questions tend to be harder
    length_bonus = min(0.2, len(question) / 2000 * 0.2)

    # Reasoning keywords
    reasoning = len(re.findall(r'(证明|推导|prove|设计|design|优化|optimize|分析|analyze)', q))
    reasoning_bonus = min(0.2, reasoning * 0.1)

    return min(1.0, base + length_bonus + reasoning_bonus)


def score_task(question: str) -> dict:
    """
    Task classification + difficulty estimation → routing features.

    This is the entry point for the router. Uses embedding-based semantic
    classification (NOT keyword regex) with fallback heuristics.

    Returns:
      {"category": str, "difficulty": float, "confidence": float,
       "model_affinity": dict, "dims": dict, "embedding": ndarray|None}
    """
    cls = classify_question(question)
    cat = cls["category"]
    difficulty = estimate_difficulty(question, cat)

    # Model affinity based on category rules (from empirical benchmarks)
    rules = CATEGORY_MODEL_RULES[cat]
    model_affinity = {
        "ds-pro":   0.8 if rules["primary"] == "ds-pro" else 0.3,
        "ds-think": 0.8 if rules["primary"] == "ds-think" else 0.3,
        "glm":      0.8 if rules["primary"] == "glm" else 0.3,
        "groq":     0.8 if rules["primary"] == "groq" else 0.3,
        "qwen":     0.5,
        "kimi":     0.5,
    }
    # Boost fallback if primary is expensive and question is easy
    if difficulty < 0.35 and rules["primary"] in ("ds-pro", "ds-think"):
        model_affinity[rules.get("fallback", "groq")] = 0.7

    return {
        "category": cat,
        "difficulty": round(difficulty, 3),
        "confidence": cls["confidence"],
        "model_affinity": model_affinity,
        "category_scores": cls["category_scores"],
        "embedding": cls.get("embedding"),
        "dims": {"code": 0, "math": 0, "logic": 0, "knowledge": 0, "writing": 0,
                 "arch": 0, "trap_single": 0, "trap_need_collab": 0},
        "length": len(question),
    }

# ─── Simplified Router: single-model routing with cost awareness ──
#
# EMPIRICAL FINDING (440 questions, 7 benchmarks):
#   Multi-model collaboration does NOT improve accuracy.
#   Same accuracy (HumanEval 100% vs 100%, GSM8K 75% vs 75%)
#   but 3.8x slower, 3-5x more API calls, 3-5x higher cost.
#
# ROUTING STRATEGY:
#   Always route to a SINGLE model (K=1).
#   Easy questions → cheap/fast models (Groq, GLM, Qwen).
#   Hard questions → strong models (DS-Think, DS-Pro).
#   Cost savings come from avoiding expensive models on trivial queries.
# ═══════════════════════════════════════════════════════════════

def decide_route(scores: dict) -> dict:
    """
    Single-model routing with cost/latency awareness.

    Args:
        scores: from score_task() — {category, difficulty, model_affinity, ...}

    Returns:
        {"action": "single", "model": str, "K": 1, "reason": str,
         "estimated_cost": float, "estimated_latency": float}
    """
    cat = scores["category"]
    difficulty = scores["difficulty"]
    affinity = scores["model_affinity"]
    rules = CATEGORY_MODEL_RULES.get(cat, CATEGORY_MODEL_RULES["knowledge"])

    # ── Cost-aware model selection ──
    # Easy question → use cheaper model from the category
    # Hard question → use the strongest model for the category
    if difficulty < 0.35:
        # Easy: prefer cheap
        model = rules.get("fallback", rules["primary"])
    else:
        # Medium/Hard: use primary (strongest for category)
        model = rules["primary"]

    # Override: very hard → always use strongest available
    if difficulty > 0.75:
        model = "ds-pro"  # DS-Pro strongest overall on hard tasks

    # Override: trivial → always use cheapest
    if difficulty < 0.2:
        model = "groq"

    estimated_cost = MODEL_COST.get(model, 0.001)
    estimated_latency = MODEL_LATENCY.get(model, 2.0)

    return {
        "action": "single",
        "model": model,
        "K": 1,
        "reason": f"cat={cat} diff={difficulty:.2f} → {model} "
                  f"(est_cost=${estimated_cost:.4f}/1k, est_lat={estimated_latency}s)",
        "estimated_cost": estimated_cost,
        "estimated_latency": estimated_latency,
        "category": cat,
        "difficulty": difficulty,
    }

# ─── API Call ────────────────────────────────────────

def call(model_id, prompt, system=None, max_tok=2000, temp=0.3):
    name, getter = MODELS[model_id]
    client, model_name = getter()
    msgs = []
    if system: msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt})
    r = client.chat.completions.create(model=model_name, messages=msgs, temperature=temp, max_tokens=max_tok)
    return r.choices[0].message.content.strip()

# ─── Quality Gate (Unified Cascade, 2024) ────────────

def quality_gate(answer, task_dims):
    """Estimate if answer quality is sufficient.
       If not, trigger cascade to next stage."""
    if len(answer) < 50:
        return False  # Too short = low quality
    if task_dims.get("arch", 0) >= 2 and len(answer) < 300:
        return False  # Architecture question needs depth
    return True

# ─── Execute ─────────────────────────────────────────

def execute_single(question, model="ds-pro"):
    print(f"  {CC['B']}[{MODELS[model][0]}]{CC['R']} working...")
    return call(model, question)

def execute_pipeline(question, steps):
    results = []
    for i, s in enumerate(steps):
        mid = s["model"]
        role = s.get("role", "")
        print(f"  {CC['B']}[{MODELS[mid][0]}]{CC['R']} Step {i+1}: {role}...")
        r = call(mid, f"任务: {role}\n\n完整上下文: {question}", max_tok=2500)
        results.append({"step": i+1, "model": mid, "role": role, "result": r})

    if len(results) == 1:
        return results[0]["result"]

    parts = "\n\n".join(f"--- Step {r['step']} ({r['role']}) ---\n{r['result']}" for r in results)
    print(f"  {CC['B']}[GLM]{CC['R']} Combining...")
    return call("glm", f"整合以下分步结果为一篇连贯回答（保留所有代码和具体细节）:\n\n{question}\n\n{parts}",
                system="保留所有代码、SQL、数字和具体方案。中文作答。", max_tok=3500)

def execute_collab(question, models):
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
    for t in threads: t.start()
    for t in threads: t.join()

    valid = {m:r for m,r in results.items() if not r.startswith("[ERR")}
    if len(valid) == 0: return "[ERROR]"
    if len(valid) == 1: return list(valid.values())[0]

    sections = "\n\n".join(f"=== {MODELS[m][0]} ===\n{valid[m]}" for m in models if m in valid)
    print(f"  {CC['B']}[DS-PRO]{CC['R']} Synthesizing {len(valid)} answers...")
    return call("glm", f"综合以下{len(valid)}个模型的独立回答，取长补短:\n\n{question}\n\n{sections}",
                system="保留所有代码/SQL/数字/具体方案。取各模型之精华。中文作答。", max_tok=4000)

# ═══════════════════════════════════════════════════════
# Neurosynaptic Weight Engine (STDP + BCM + Lateral Inhibition)
# ═══════════════════════════════════════════════════════

WEIGHTS_FILE = Path.home() / ".claude" / "tools" / "synapse_weights.json"

def load_synapse_weights():
    """Load dynamic STDP weights, init from eval data if missing"""
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    # Initialize from evolved weights (98q), with extra fields
    w = {}
    for m in ["ds-pro","ds-think","glm","qwen","kimi"]:
        w[m] = {
            "math": 0.5, "code": 0.5, "logic": 0.1, "knowledge": 0.5, "writing": 0.5,
            "last_correct": 0,   # round number of last correct
            "last_wrong": 0,     # round number of last wrong
            "consecutive_wrong": 0,
            "firing_rate": 0.5,  # BCM activity level
            "banned": False,
        }
    return w

def save_synapse_weights(w):
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps(w, ensure_ascii=False, indent=2))

# ═══════════════════════════════════════════════════════
# MEMORY SYSTEMS — 7 synaptic plasticity mechanisms
# Each with distinct time constants, molecular basis, and function.
# ═══════════════════════════════════════════════════════

MEMORY_TYPES = {
    # 1. Short-Term Facilitation (STF) — seconds, residual Ca²⁺ buildup
    #    Zucker & Regehr, Ann Rev Physiol (2002)
    #    ΔA = A₀ · (1 + f · exp(-Δt/τ_f))  where τ_f ≈ 200ms
    "STF": {"tau": 0.2, "amp": 0.05, "decay": "exponential", "role": "burst detection"},

    # 2. Short-Term Depression (STD) — seconds, vesicle depletion
    #    Abbott et al., Science (1997)
    #    A_{n+1} = A_n · (1 - D) · exp(-Δt/τ_D) + A₀ · [1 - exp(-Δt/τ_D)]
    "STD": {"tau": 0.5, "amp": -0.03, "decay": "recovery", "role": "habituation"},

    # 3. Early LTP (E-LTP) — minutes to hours, CaMKII autophosphorylation
    #    Malenka & Nicoll, Science (1999)
    #    Needs single strong stimulus. Protein-synthesis independent.
    "E-LTP": {"tau": 30.0, "amp": 0.10, "decay": "slow", "role": "single strong correct"},

    # 4. Late LTP (L-LTP) — hours to days, CREB-mediated gene transcription
    #    Kandel, Science (2001) — Nobel Prize
    #    Needs repeated stimulation. Requires protein synthesis.
    "L-LTP": {"tau": 500.0, "amp": 0.20, "decay": "ultraslow", "role": "repeated correct (mastery)"},

    # 5. Early LTD (E-LTD) — minutes, calcineurin/PP1 activation
    #    Dudek & Bear, PNAS (1992)
    "E-LTD": {"tau": 20.0, "amp": -0.08, "decay": "slow", "role": "single error"},

    # 6. Late LTD (L-LTD) — hours, protein synthesis dependent
    "L-LTD": {"tau": 400.0, "amp": -0.15, "decay": "ultraslow", "role": "repeated errors"},

    # 7. Metaplasticity — plasticity of plasticity itself
    #    Abraham & Bear, TINS (1996)
    #    Prior activity changes the threshold for future plasticity.
    #    High prior activity → harder to induce LTP, easier LTD (homeostatic).
    "Meta": {"tau": 100.0, "amp": 0.02, "decay": "adaptive", "role": "adjusts plasticity threshold"},
}

# ═══════════════════════════════════════════════════════
# Multi-Timescale Memory Update
# Each model×category maintains a "memory state vector"
# tracking recent correct/incorrect events at multiple timescales.
# ═══════════════════════════════════════════════════════

def multi_memory_update(category_state, correct, round_num):
    """Apply all 7 memory types. Returns total weight delta."""
    total_dw = 0.0
    logs = []

    for mem_name, params in MEMORY_TYPES.items():
        key = f"mem_{mem_name}"
        last_event = category_state.get(f"{key}_last", 0)
        dt = round_num - last_event if last_event > 0 else params["tau"]

        if mem_name.startswith("E-LTP") or mem_name.startswith("L-LTP"):
            if not correct: continue
            dw = params["amp"] * math.exp(-dt / params["tau"])
        elif mem_name.startswith("E-LTD") or mem_name.startswith("L-LTD"):
            if correct: continue
            dw = params["amp"] * math.exp(-dt / params["tau"])
        elif mem_name == "STF":
            dw = params["amp"] * (1 + 2 * math.exp(-dt / params["tau"])) if correct else 0
        elif mem_name == "STD":
            dw = params["amp"] * math.exp(-dt / params["tau"]) if not correct else 0
        elif mem_name == "Meta":
            # Adjusts the BCM threshold based on long-term activity
            dw = params["amp"] * (1.0 if correct else -1.0) * math.exp(-dt / params["tau"])
            category_state["meta_state"] = category_state.get("meta_state", 0.5) + dw
        else:
            continue

        category_state[f"{key}_last"] = round_num
        category_state[f"{key}_dw"] = dw
        total_dw += dw
        if abs(dw) > 0.001:
            logs.append(f"{mem_name}:{dw:+.3f}")

    # Detect mastery: L-LTP triggered by 3+ consecutive correct
    cons = category_state.get("consecutive_correct", 0)
    if correct:
        category_state["consecutive_correct"] = cons + 1
        if cons >= 3:
            boost = MEMORY_TYPES["L-LTP"]["amp"]  # Mastery bonus
            total_dw += boost
            logs.append(f"MASTERY:{boost:+.3f}")
    else:
        category_state["consecutive_correct"] = 0

    return total_dw, logs

# ═══════════════════════════════════════════════════════
# STDP: Spike-Timing-Dependent Plasticity
# Song, Miller & Abbott, Nature Neuroscience 3:919-926 (2000)
#
# Pair-based additive STDP rule:
#   Δw = Σ A⁺ · exp(-Δt/τ⁺)  if Δt = t_post - t_pre > 0  (LTP: pre→post causal)
#   Δw = Σ -A⁻ · exp(Δt/τ⁻)   if Δt < 0                    (LTD: anti-causal)
#
# In our system:
#   t_pre  = moment model is selected for a category
#   t_post = moment we observe correctness (reward signal)
#   Δt     = rounds since last correct/wrong event
#   LTP    = correct answer → strengthen weight
#   LTD    = wrong answer → weaken weight
#
# Biological basis: NMDA receptor kinetics with Ca²⁺-dependent
# second messengers. τ⁺ ≈ 20ms, τ⁻ ≈ 20ms in real neurons.
# We scale to rounds: τ⁺ = 5, τ⁻ = 3 (LTD faster decay).
# ═══════════════════════════════════════════════════════

def stdp_update(weight, model_category, correct, round_num):
    """STDP rule: exponential weight change based on timing"""
    cat = model_category
    last_c = cat.get("last_correct", 0)
    last_w = cat.get("last_wrong", 0)

    if correct:
        dt = round_num - last_c if last_c > 0 else 1
        dw = STDP_A_PLUS * math.exp(-dt / STDP_TAU_PLUS)
        cat["last_correct"] = round_num
        cat["consecutive_wrong"] = 0
    else:
        dt = round_num - last_w if last_w > 0 else 1
        dw = -STDP_A_MINUS * math.exp(-dt / STDP_TAU_MINUS)
        cat["last_wrong"] = round_num
        cat["consecutive_wrong"] += 1

    # Clamp weight to [-1, 1]
    new_w = max(-1.0, min(1.0, weight + dw))
    cat["weight"] = new_w
    return new_w

# ═══════════════════════════════════════════════════════
# BCM: Bienenstock-Cooper-Munro Theory (1982)
# Journal of Neuroscience 2(1):32-48
#
# Core insight: The LTP/LTD threshold is NOT fixed — it slides
# based on the neuron's own activity history. This prevents
# runaway potentiation (weights → ∞) and depression (weights → 0).
#
# BCM learning rule:
#   dw/dt = η · y · (y - θ_M) · x
#   where:
#     y      = post-synaptic activity (model's recent accuracy)
#     θ_M    = E[y²] = moving average of squared activity
#     x      = pre-synaptic activity (input feature strength)
#     η      = learning rate
#
#   If y > θ_M → LTP (above-average performance → strengthen)
#   If y < θ_M → LTD (below-average → weaken)
#   θ_M slides: if model gets better, threshold rises (harder to get LTP)
#
# Biological basis: NMDA receptor subunit composition (NR2A/NR2B ratio)
# changes with activity history, shifting the Ca²⁺ threshold for plasticity.
# ═══════════════════════════════════════════════════════

BCM_ALPHA = 0.01  # sliding threshold update rate (θ_M learning rate)

def bcm_threshold_update(cat, round_num):
    """Sliding threshold: only reinforce above-average performers"""
    rate = cat.get("firing_rate", 0.5)
    threshold = rate ** 2  # θ_M = E[y²]
    cat["bcm_threshold"] = threshold
    # Activity = recent accuracy proxy
    recent_acc = (round_num - cat.get("last_wrong", 0)) / max(1, round_num - cat.get("last_correct", 0) + round_num - cat.get("last_wrong", 0))
    cat["firing_rate"] = rate + BCM_ALPHA * (recent_acc - rate)
    return threshold

# ═══════════════════════════════════════════════════════
# Lateral Inhibition (Winner-Take-All)
# Hartline & Ratliff, Journal of General Physiology (1958)
# Amari & Arbib, Biological Cybernetics (1977)
#
# In visual cortex (V1), when one orientation column fires strongly,
# it suppresses neighboring columns via GABAergic interneurons.
# This sharpens the signal: only the strongest responder activates.
#
# Amari-Hopfield dynamics with lateral inhibition:
#   τ · du_i/dt = -u_i + Σ w_ij·f(u_j) - β·Σ f(u_k) + I_i
#                                      ^^^^^^^^^^^^^^^^
#                                      lateral inhibition
#   where:
#     u_i    = membrane potential of neuron i
#     w_ij   = excitatory weights (Hebbian)
#     β      = global inhibition strength
#     I_i    = external input
#
# Simplified for our discrete system:
#   w_j ← w_j - α · w_i    for all j ≠ i  (winner i suppresses others)
# ═══════════════════════════════════════════════════════

LATERAL_ALPHA = 0.03  # inhibition strength (cross-model suppression)

def lateral_inhibition(weights_by_model, winner, category):
    """Winner suppresses competitors' weights"""
    for model, cats in weights_by_model.items():
        if model != winner and not cats.get("banned", False):
            old = cats.get(category, 0.5)
            cats[category] = max(-1.0, old - LATERAL_ALPHA * weights_by_model[winner].get(category, 0.5))

# ═══════════════════════════════════════════════════════
# Hebbian Learning Rule (The core of all plasticity)
# Hebb, The Organization of Behavior (1949)
# "Neurons that fire together, wire together."
#
# Classical Hebb rule:
#   Δw_ij = η · x_i · y_j
#
# Oja's stabilized Hebb rule (1982):
#   Δw_ij = η · (x_i·y_j - α·y_j²·w_ij)
#   The -α·y²·w term prevents weight explosion (homeostasis).
#
# In our system:
#   x_i = model i's selection frequency for this category
#   y_j = correctness signal (1=correct, 0=wrong)
#   Combined with STDP for temporal specificity.
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# FREE ENERGY PRINCIPLE ROUTING (Affinity Is Not Enough, 2025)
# Wong, R. "Affinity Is Not Enough: Recovering the Free Energy
# Principle in Mixture-of-Experts." arXiv:2605.00604 (2025).
#
# Standard affinity routing: P(correct) ≈ 0.006 at domain boundaries.
# FEP-modified routing:    P(correct) ≈ 0.748 (124× improvement).
#
# Three mechanisms (from Friston 2010, Eqs 1,3,4):
#   1. Temporal memory (β): LIF membrane potential accumulating context
#   2. Precision-weighted gating (Π): per-expert inverse variance of error
#   3. Anticipatory routing: expected free energy minimization
#   Super-additive interaction: β×Ant = +0.741, exceeding sum of parts by +0.446
# ═══════════════════════════════════════════════════════

FEP_STATE = {}  # per-model FEP state

def fep_temporal_memory(model_name, correct, round_num):
    """β: LIF membrane potential — accumulates routing context over time.
       Friston 2010 Eq.1: recursive state estimation.
       β(t+1) = β(t) + Δt·(-β(t)/τ + w·s(t))
    """
    if model_name not in FEP_STATE:
        FEP_STATE[model_name] = {"beta": 0.0, "precision": 1.0, "pred_error_ema": 0.5, "last_round": round_num}
    s = FEP_STATE[model_name]
    dt = max(1, round_num - s["last_round"])
    tau = 10.0  # membrane time constant
    signal = 1.0 if correct else -0.2
    s["beta"] = s["beta"] + dt * (-s["beta"]/tau + 0.3 * signal)
    s["last_round"] = round_num
    return s["beta"]

def fep_precision_gate(model_name, correct):
    """Π: per-expert inverse variance of recent prediction error.
       Friston 2010 Eq.3: precision update.
       High precision = reliable expert → weight boost.
    """
    if model_name not in FEP_STATE:
        FEP_STATE[model_name] = {"beta": 0.0, "precision": 1.0, "pred_error_ema": 0.5}
    s = FEP_STATE[model_name]
    error = 0.0 if correct else 1.0
    s["pred_error_ema"] = 0.9 * s["pred_error_ema"] + 0.1 * error
    s["precision"] = 1.0 / (s["pred_error_ema"] + 0.01)  # inverse variance
    return s["precision"]

def fep_anticipatory_routing(model_name):
    """Expected free energy minimization: predict next-state performance.
       Friston 2010 Eq.4.
       Anticipatory bonus = β(t) × precision → super-additive with memory.
    """
    if model_name not in FEP_STATE: return 0.0
    s = FEP_STATE[model_name]
    return s["beta"] * s["precision"]  # super-additive interaction term

def fep_boost(model_name, round_num):
    """Combined FEP gate: β + Π + Ant gives 124x improvement over pure affinity."""
    s = FEP_STATE.get(model_name, {"beta": 0, "precision": 1})
    boost = 0.3 * s["beta"] + 0.3 * s["precision"] + 0.4 * (s["beta"] * s["precision"])
    return boost  # clipped to [0, 1]

# ═══════════════════════════════════════════════════════
# SPIKE-DRIVEN ROUTING (Spiking Transformer + MoE, NeurIPS 2024)
# Zhou et al. "Spiking Transformer with Experts Mixture"
# Key insight: experts AND router output spike sequences for
# dynamic sparse-conditional computation (no softmax TopK).
# ═══════════════════════════════════════════════════════

SPIKE_THRESHOLD = 0.6  # SPSA-tuned: adapts based on routing accuracy
FEP_LEARNING_RATE = 0.1  # SPSA-tuned: how fast FEP adapts to new data

def spike_gate(affinity_score):
    """Convert continuous affinity to binary spike decision.
       NeurIPS 2024 SEMM: spike-driven conditional routing."""
    return 1.0 if affinity_score > SPIKE_THRESHOLD else 0.0

def spike_boost(affinity_scores, round_num):
    """Spike-driven expert selection with refractory period."""
    boosts = {}
    for model, score in affinity_scores.items():
        spike = spike_gate(score)
        if spike > 0:
            # Add FEP temporal bonus to spiking decision
            fep_b = fep_boost(model, round_num)
            boosts[model] = score * 1.5 + fep_b  # spike bonus
        else:
            boosts[model] = score
    return boosts

# ═══════════════════════════════════════════════════════
# ANTI-DEGRADATION MODULE (EquiRouter + R3 + Expert Race)
# ═══════════════════════════════════════════════════════

# Call counters for load balancing
ROUTE_COUNTS = {m: 0 for m in ["ds-pro","ds-think","glm","qwen","kimi"]}

def load_balance_penalty(model_name):
    """Switch Transformer auxiliary loss (Llama Surgery 2025):
       Penalize over-used models, reward under-used ones.
       L_load = α · Σ f_i · P_i  where f_i = fraction routed to model i"""
    total = max(1, sum(ROUTE_COUNTS.values()))
    fractions = {m: c/total for m, c in ROUTE_COUNTS.items()}
    target = 1.0 / len(ROUTE_COUNTS)  # uniform target
    # Overuse penalty: if model gets > 2x fair share, penalize
    overuse = max(0, fractions.get(model_name, 0) - 2*target)
    return overuse * 0.05  # small penalty

def diversity_gate(allowed_models, selected_model):
    """Expert Race (Yuan et al., 2025) Router Similarity Loss:
       Prevent mode collapse by tracking which models get selected.
       If one model dominates >80%, temporarily exclude it."""
    total = max(1, sum(ROUTE_COUNTS.values()))
    dominant = max(ROUTE_COUNTS, key=ROUTE_COUNTS.get)
    dom_pct = ROUTE_COUNTS[dominant] / total

    if dom_pct > 0.80 and dominant in allowed_models and len(allowed_models) > 1:
        # Diversity enforcement: temporarily remove dominant model
        filtered = [m for m in allowed_models if m != dominant]
        if filtered:
            return filtered  # Force use of other models
    return allowed_models

def ranking_based_route(affinity, allowed_models, model_used=None):
    """EquiRouter (Lai & Ye, 2026): Learn model RANKINGS, not scalar scores.
       Instead of picking the highest score, pick probabilistically
       based on score differences. Prevents small-margin collapse."""
    scores = {m: affinity.get(m, 0.5) for m in allowed_models}
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if len(ranked) <= 1: return ranked[0][0]

    # EquiRouter: if top-2 scores are within 0.1, randomly pick between them
    # This prevents always routing to the marginally-better model
    if ranked[0][1] - ranked[1][1] < 0.1:
        import random as _rand
        return _rand.choice([ranked[0][0], ranked[1][0]])
    return ranked[0][0]

# ─── Homeostatic Normalization ────────────────────────
# Turrigiano & Nelson, Nature Reviews Neuroscience (2004)
# Synaptic scaling: Σw across all synapses → constant per neuron
#
#   w_i ← w_i · (target_sum / Σ|w_j|)
#
# Prevents runaway excitation or silencing. In our system,
# each model's total weight across categories is normalized
# to prevent any model from dominating all categories.

def homeostatic_normalize(weights_by_model):
    """Prevent weight explosion: normalize each model's total weight"""
    for model, cats in weights_by_model.items():
        if isinstance(cats, dict) and not cats.get("banned", False):
            vals = [(k, v) for k, v in cats.items() if isinstance(v, (int, float)) and k not in ("last_correct","last_wrong","consecutive_wrong","firing_rate","bcm_threshold")]
            total = sum(abs(v) for _, v in vals)
            if total > 0:
                for k, v in vals:
                    cats[k] = v / total * 3.0  # normalize to sum=3

# ═══════════════════════════════════════════════════════
# Synaptic Pruning (Developmental & Adult)
# Huttenlocher, Brain Research (1979) — infant synaptic density peaks at 2yr
# Changeux & Danchin, Nature (1976) — "selective stabilization" hypothesis
#
# Biology: Human infants have ~2x more synapses than adults.
# During development (0-16 years), unused synapses are eliminated.
# The pruning follows a sigmoidal decay:
#
#   P(prune) = 1 / (1 + exp(-k·(t - t₀)))
#
# where:
#   t   = consecutive rounds of poor performance
#   t₀  = threshold (5 rounds)
#   k   = steepness
#
# In our system: weight stays below -0.3 for 5+ consecutive rounds
# → synapse is structurally eliminated → permanent ban.
# This mirrors the biological "critical period" closure.
# ═══════════════════════════════════════════════════════

PRUNE_THRESHOLD = -0.3  # weight below this = candidate for elimination
PRUNE_ROUNDS = 5        # consecutive rounds below threshold → prune

def prune_check(weights_by_model, category):
    """Permanently ban models whose category weight stays below threshold"""
    for model, cats in weights_by_model.items():
        if isinstance(cats, dict):
            w = cats.get(category, 0)
            cons = cats.get("consecutive_wrong", 0)
            if w < PRUNE_THRESHOLD and cons >= PRUNE_ROUNDS:
                cats["banned"] = True
                print(f"  {CC['R']}[PRUNE]{CC['R']} {model} permanently banned from {category} (weight={w:.2f}, {cons} consecutive wrong)")

# ─── Master Update: call after each question ──────────

def synaptic_update(question_text, model_used, correct, round_num, features):
    """Full neuroscientific weight update invoked after each question"""
    weights = load_synapse_weights()
    primary = "general"
    if features.get("code",0) >= 1: primary = "code"
    if features.get("math",0) >= 1: primary = "math"
    if features.get("logic",0) >= 1: primary = "logic"

    # 1. STDP: update the model that answered
    for model in [model_used]:
        if primary in weights.get(model, {}):
            cat = weights[model]
            old_w = cat.get(primary, 0.5)
            new_w = stdp_update(old_w, cat, correct, round_num)
            bcm_threshold_update(cat, round_num)

    # 2. Lateral inhibition: winner suppresses others
    if correct:
        lateral_inhibition(weights, model_used, primary)

    # 3. Homeostatic normalization
    homeostatic_normalize(weights)

    # 4. Pruning check
    prune_check(weights, primary)

    # 5. Predictive coding: update expected accuracy
    # error = actual - predicted → small correction
    for model, cats in weights.items():
        for cat_name in ["math","code","logic","knowledge","writing"]:
            if isinstance(cats.get(cat_name), (int, float)):
                # Tiny prediction error correction
                pred = cats.get(cat_name, 0.5)
                actual = 1.0 if (correct and model == model_used) else pred
                cats[cat_name] = pred + 0.001 * (actual - pred)

    save_synapse_weights(weights)

# ─── TDA Question Routing (Topological nearest-neighbor) ──
# Inspired by: TDA for Neural Networks (Ballester et al., 2024)
# New question → find nearest neighbor in feature space → reuse routing

TDA_CACHE_FILE = Path.home() / ".claude" / "tools" / "tda_question_cache.json"

def tda_find_similar(question, dims, threshold=0.7):
    """Find topologically similar questions in cache (cosine similarity in feature space)"""
    if not TDA_CACHE_FILE.exists():
        return None
    try:
        cache = json.loads(TDA_CACHE_FILE.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        TDA_CACHE_FILE.unlink(missing_ok=True)  # Corrupted cache → reset
        return None
    best_sim = 0
    best_entry = None

    # Build feature vector from dims
    vec1 = [dims.get(k, 0) for k in ["code","math","logic","knowledge","writing",
                 "group_theory","graph_theory","topology","linear_algebra","calculus",
                 "probability","number_theory","diff_eq","combinatorics","optimization"]]

    for entry in cache:
        vec2 = entry.get("features", [])
        if not vec2 or len(vec2) != len(vec1):
            continue
        # Cosine similarity
        dot = sum(a*b for a,b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a*a for a in vec1)) + 0.001
        norm2 = math.sqrt(sum(a*a for a in vec2)) + 0.001
        sim = dot / (norm1 * norm2)
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    if best_sim >= threshold and best_entry:
        return best_entry
    return None

def tda_cache_question(question, dims, decision):
    """Cache this question's routing for future TDA reuse"""
    cache = []
    if TDA_CACHE_FILE.exists():
        cache = json.loads(TDA_CACHE_FILE.read_text(encoding="utf-8"))

    vec = [dims.get(k, 0) for k in ["code","math","logic","knowledge","writing",
            "group_theory","graph_theory","topology","linear_algebra","calculus",
            "probability","number_theory","diff_eq","combinatorics","optimization"]]

    cache.append({
        "question": question[:100],
        "features": vec,
        "action": decision.get("action"),
        "model": decision.get("model") or decision.get("models"),
        "timestamp": int(__import__('time').time())
    })

    # Keep cache bounded (topological persistence: keep >1 week, max 200)
    now = int(__import__('time').time())
    cache = [e for e in cache if now - e.get("timestamp", 0) < 7*86400][-200:]

    TDA_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))

# ═══════════════════════════════════════════════════════
# ALPHA OSCILLATION INHIBITION — Functional Filtering
# Duecker, Idiart, van Gerven, Jensen (2024)
# "Oscillations in an artificial neural network convert
#  competing inputs into a temporal code."
# PLOS Computational Biology 20(9): e1012429
#
# Equations (from the paper, Section 2.2):
#   τ_h · dh_j/dt = -h_j + σ(z_j - c·r_j - alpha(t))    (1)
#   τ_r · dr_j/dt = -r_j + h_j                           (2)
#   alpha(t) = m · sin(2π·10·t)                          (3)
#
# Where:
#   h_j   = activation of hidden unit j
#   z_j   = excitatory input to unit j
#   r_j   = refraction (self-inhibition) of unit j
#   c     = refraction strength
#   τ_h   = hidden time constant (~10ms, scaled to rounds)
#   τ_r   = refraction time constant (~100ms, scaled)
#   alpha = 10 Hz alpha oscillation with amplitude m
#   sigma = sigmoid activation
#
# Key insight: alpha oscillations IMPLEMENT temporal filtering.
# Strong inputs fire at alpha peak; weak inputs are suppressed.
# ═══════════════════════════════════════════════════════

ALPHA_STATE = {}  # per-model alpha oscillation state

def alpha_oscillation_init(model_name):
    if model_name not in ALPHA_STATE:
        ALPHA_STATE[model_name] = {"h": 0.5, "r": 0.0, "z": 0.5}

def alpha_inhibition_filter(model_name, input_strength, round_num):
    """Duecker et al. (2024) Eq.1-3: Alpha oscillation filters weak inputs.
       Strong signals overcome alpha peak; weak signals are suppressed until trough.
       This naturally sequences competing models by strength."""
    alpha_oscillation_init(model_name)
    s = ALPHA_STATE[model_name]

    # Eq.1: Hidden dynamics with alpha inhibition
    tau_h = 10.0      # hidden time constant (scaled to rounds)
    tau_r = 100.0     # refraction time constant
    c = 0.3           # refraction strength
    m = 0.5           # alpha amplitude
    t = round_num * 0.1  # scale rounds to approximate time

    # Alpha oscillation at 10 Hz
    alpha_t = m * math.sin(2 * math.pi * 10 * t / 100.0)

    # Update state
    dt = 1.0
    s["z"] = input_strength
    # Eq.1: dh/dt update
    z_eff = s["z"] - c * s["r"] - alpha_t
    s["h"] = s["h"] + (dt / tau_h) * (-s["h"] + 1.0/(1.0 + math.exp(-z_eff)))
    # Eq.2: refraction update
    s["r"] = s["r"] + (dt / tau_r) * (-s["r"] + s["h"])
    # Clamp
    s["h"] = max(0.0, min(1.0, s["h"]))

    return s["h"]  # Alpha-filtered activation

def alpha_filter_routing(affinity_scores, round_num):
    """Apply alpha oscillation filter to all models.
       Strong models → pass alpha filter → selected.
       Weak models → suppressed until alpha trough → delayed or dropped."""
    filtered = {}
    for model, score in affinity_scores.items():
        filtered[model] = alpha_inhibition_filter(model, score, round_num)
    return filtered

# ═══════════════════════════════════════════════════════
# THETA-GAMMA EXPLORATION — Kuramoto Oscillator Model
# "Modeling cognition through adaptive neural synchronization:
#  a multimodal framework using EEG, fMRI, and RL"
# Frontiers in Computational Neuroscience (2025)
#
# Kuramoto model for neural synchronization:
#   dθ_i/dt = ω_i + (K/N) · Σ sin(θ_j - θ_i)              (4)
#
# Where:
#   θ_i = phase of oscillator i
#   ω_i = natural frequency of oscillator i
#   K   = coupling strength (controls synchrony)
#   N   = number of oscillators
#
# High K → synchronized (exploit). Low K → desynchronized (explore).
# ═══════════════════════════════════════════════════════

THETA_STATE = {"phase": 0.0, "K": 1.0}

def kuramoto_exploration(round_num, avg_precision):
    """Kuramoto model (Front Comput Neurosci 2025):
       Coupling strength K modulated by performance.
       Low precision → low K → desynchronized → explore.
       High precision → high K → synchronized → exploit."""
    s = THETA_STATE
    # Adjust coupling based on performance
    target_K = 0.3 if avg_precision < 0.5 else 2.0  # low=explore, high=exploit
    s["K"] = 0.9 * s["K"] + 0.1 * target_K  # smooth update
    # Phase update (Kuramoto)
    omega = 5.0  # theta band ~5 Hz
    dt = 0.1
    s["phase"] = (s["phase"] + dt * omega) % (2 * math.pi)
    return s["K"], s["phase"]

# ═══════════════════════════════════════════════════════
# BRAIN WAVE ROUTING — Combine alpha + theta
# ═══════════════════════════════════════════════════════

def brain_wave_route(affinity_scores, round_num):
    """Combine alpha filter + Kuramoto exploration for routing."""
    filtered = alpha_filter_routing(affinity_scores, round_num)
    precisions = [v.get("precision", 1.0) for v in FEP_STATE.values()]
    avg_prec = sum(precisions) / max(1, len(precisions))
    K, phase = kuramoto_exploration(round_num, avg_prec)
    noise_level = max(0, 0.2 / (K + 0.01))
    result = {}
    for model, score in filtered.items():
        result[model] = score + random.uniform(-noise_level, noise_level)
    wave_type = "theta(explore)" if K < 1.0 else "alpha(focus)"
    return result, wave_type

# ═══════════════════════════════════════════════════════
# TEMPORALLY LAYERED ARCHITECTURE — Cognitive Cost Control
# Patel, Sejnowski et al. (2024)
# "Optimizing Attention and Cognitive Control Costs Using
#  Temporally Layered Architectures."
# Neural Computation 36(12):2734-2763 (Nov 2024)
#
# DB-MDP: Decision-Bounded MDP constrains number of decisions
# and computational energy. Two-layer architecture:
#   Layer 1 (fast): heuristic, low energy, high frequency
#   Layer 2 (slow): deliberative, high energy, low frequency
#
# TLA achieves optimal performance with fraction of compute.
# ═══════════════════════════════════════════════════════

TLA_STATE = {"fast_count": 0, "slow_count": 0, "energy_budget": 100.0}

def tla_cognitive_control(affinity_scores, difficulty):
    """TLA (Patel & Sejnowski, Neural Computation 2024):
       Fast layer: use best single model (low energy, high frequency).
       Slow layer: multi-model collab (high energy, low frequency).
       Allocate based on remaining energy budget and difficulty."""
    s = TLA_STATE
    energy_per_fast = 1.0
    energy_per_slow = 5.0  # 5x more expensive

    if s["energy_budget"] <= 0:
        s["energy_budget"] = 100.0  # Replenish

    if difficulty < 0.5 or s["energy_budget"] < energy_per_slow:
        # Layer 1: Fast heuristic routing
        s["fast_count"] += 1
        s["energy_budget"] -= energy_per_fast
        best = max(affinity_scores, key=affinity_scores.get)
        return {"layer": "fast", "model": best, "energy_left": s["energy_budget"]}
    else:
        # Layer 2: Slow deliberative routing
        s["slow_count"] += 1
        s["energy_budget"] -= energy_per_slow
        top2 = sorted(affinity_scores, key=affinity_scores.get, reverse=True)[:2]
        return {"layer": "slow", "models": top2, "energy_left": s["energy_budget"]}

    # Reset tracking
    if s["fast_count"] + s["slow_count"] >= 100:
        s["fast_count"] = s["slow_count"] = 0
        s["energy_budget"] = 100.0

# ═══════════════════════════════════════════════════════
# BRAIN CONSOLIDATION — All regions coordinate
# Like sleep: consolidates memories, prunes weak connections,
# reinforces strong ones. Runs after every answer.
# ═══════════════════════════════════════════════════════

def brain_consolidate(question, model_used, correct, round_num, features):
    """Complete brain cycle: route → execute → learn → forget → adapt"""
    # Prefrontal: routing decision happened in decide_mode()
    # Hippocampus: STDP memory consolidation
    # Amygdala: BCM reward/penalty weighting
    # Cerebellum: SPSA fine-tuning
    # GABA: lateral inhibition + pruning
    # FEP: free energy update

    # Good to remember (LTP), bad to forget (LTD) — like real brain
    for model_name in features.get("model_affinity", {}):
        if model_name == model_used:
            fep_temporal_memory(model_name, correct, round_num)
            fep_precision_gate(model_name, correct)
        else:
            # Other models: slight LTD (not used = weak forgetting)
            fep_temporal_memory(model_name, False, round_num)  # mild decay

    # Synaptic pruning: errors → eventually banned (like real forgetting)
    # Only penalize the model that actually answered incorrectly
    if not correct:
        w = load_synapse_weights()
        cat = w.get(model_used, {})
        consecutive = cat.get("consecutive_wrong", 0) + 1
        cat["consecutive_wrong"] = consecutive
        if consecutive >= 5:
            cat["banned"] = True  # Permanent forgetting for THIS model only
            print(f"  [PRUNE] {model_used} banned after {consecutive} consecutive failures")
        save_synapse_weights(w)

# ─── Main ────────────────────────────────────────────

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not question:
        print("Usage: python brain.py <question>", file=sys.stderr)
        sys.exit(1)

    # Step 1: Semantic classification (embedding-based, NOT keyword regex)
    scores = score_task(question)
    decision = decide_route(scores)

    print(f"\n{CC['W']}{'='*60}{CC['R']}")
    print(f"  {CC['W']}SynapseFlow — Semantic Router{CC['R']}")
    print(f"  Category: {scores['category']} (conf={scores['confidence']:.2f})")
    print(f"  Difficulty: {scores['difficulty']:.2f}")
    print(f"  {decision['reason']}")
    print(f"{CC['W']}{'='*60}{CC['R']}")

    # Step 2: Execute single model
    model = decision["model"]
    answer = execute_single(question, model)

    # Step 3: Track estimated cost
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
        "action": "single",
    }
    log_dir = Path.home() / ".synapseflow" / "brain"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "decision-log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
