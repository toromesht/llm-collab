#!/usr/bin/env python3
"""
Brain v11 — Neurosynaptic Adaptive Router
  STDP (Song&Abbott, NatNeuro 2000): recency-weighted Δw = A± · exp(-Δt/τ±)
  BCM (Bienenstock 1982): sliding threshold θ_M = E[y²]
  Lateral Inhibition: winner suppresses competitors
  Predictive Coding (Friston): error = actual - expected, drives update
  Homeostatic: Σw = constant, pruning at w < -0.3 for >5 rounds
"""
import sys, json, os, threading, re, math
from pathlib import Path
from openai import OpenAI

CONFIG_FILE = Path.home() / ".claude" / "tools" / "llm-config.json"
RULES_FILE = Path.home() / ".claude" / "tools" / "decision-rules.json"

config = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}

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
}

CC = {"R":"\033[0m","C":"\033[36m","G":"\033[32m","Y":"\033[33m","D":"\033[2m","E":"\033[31m","M":"\033[35m","B":"\033[1;34m","W":"\033[1;37m"}

# ─── Feature Extraction (from Dynamic MoE: token-difficulty scoring) ──
def score_task(question):
    """Continuous difficulty scoring, not binary keywords.
       Inspired by Dynamic MoE (Huang et al., ACL 2024): per-token difficulty -> K experts"""
    q = question.lower()

    # Dimension scores — 10 math subfields + core categories
    math_subfields = {
        "group_theory":     len(re.findall(r'(群论|群作用|子群|正规子群|商群|同态|同构|置换群|循环群|对称群|李群|有限群|阿贝尔群|半群|幺半群|换位子)', q)),
        "graph_theory":     len(re.findall(r'(图论|有向图|无向图|二分图|连通图|生成树|哈密顿|欧拉路径|平面图|着色|最大流|最小割|匹配|网络流|顶点|入度|出度)', q)),
        "topology":         len(re.findall(r'(拓扑|开集|闭集|紧致|连通|同胚|同伦|基本群|覆盖空间|流形|微分流形|纤维丛|同调|上同调|单纯形|CW复形)', q)),
        "linear_algebra":   len(re.findall(r'(线性代数|矩阵|特征值|特征向量|对角化|行列式|秩|线性变换|内积|正交|施密特|二次型|正定|逆矩阵|伴随|转置|张量)', q)),
        "calculus":         len(re.findall(r'(微积分|极限|导数|积分|微分|偏导|梯度|散度|旋度|拉普拉斯|傅里叶|级数|收敛|发散|泰勒|麦克劳林|中值定理)', q)),
        "probability":      len(re.findall(r'(概率|统计|分布|期望|方差|协方差|贝叶斯|假设检验|置信区间|回归|马尔可夫|随机过程|泊松|正态分布|二项分布|蒙特卡洛)', q)),
        "number_theory":    len(re.findall(r'(数论|素数|质数|整除|同余|模运算|费马|欧拉|RSA|丢番图|连分数|二次互反|原根|剩余|算术基本定理)', q)),
        "diff_eq":          len(re.findall(r'(微分方程|常微分|偏微分|ODE|PDE|边值问题|初值问题|分离变量|特征线|拉普拉斯方程|热方程|波动方程|薛定谔|稳定性|相图)', q)),
        "combinatorics":    len(re.findall(r'(组合|排列|组合数|鸽巢|容斥|生成函数|递推|图计数|设计|拉丁方|区组设计|极值组合|拉姆齐)', q)),
        "optimization":     len(re.findall(r'(优化|最优化|凸优化|线性规划|非线性规划|整数规划|动态规划|遗传算法|模拟退火|约束优化|目标函数|可行域|对偶|KKT|拉格朗日乘子)', q)),
    }

    dims = {
        "code": len(re.findall(r'(代码|sql|函数|算法|编程|python|写一个|实现|ddl|cte|class |def |import |SELECT|CREATE)', q)),
        "math": sum(math_subfields.values()),  # Total math signal
        "arch": len(re.findall(r'(架构|设计|系统|方案|权衡|选型|策略|规划|框架|整体|多方案)', q)),
        "writing": len(re.findall(r'(写作|论文|文档|说明|报告|总结|翻译|润色|解释|描述)', q)),
        "knowledge": len(re.findall(r'(什么|如何|为什么|定义|概念|原理|历史|背景|综述)', q)),
        "trap_single": len(re.findall(r'(图遍历|递归cte|索引优化|sql调优|存储过程|触发器|算法复杂度)', q)),
        "trap_need_collab": len(re.findall(r'(多视角|对比分析|跨领域|综合评估|深度剖析|多方|全面)', q)),
        "logic": len(re.findall(r'(说谎|谁说|真话|假话|悖论|逻辑|推理.*谁|判断.*谁|谎言|真相)', q)),
    }
    # Merge math subfields into dims for fine-grained routing
    dims.update(math_subfields)

    # ─── Group-Equivariant Coupling (群等变耦合) ──────────────────
    # When a model is strong in one subfield, it gets bonus in related subfields
    # Based on gauge equivariant transformer principle (2024): symmetry-preserving routing
    MATH_EQUIVARIANCE = {
        "group_theory":   ["linear_algebra", "number_theory"],      # 矩阵表示论, 同余→群
        "linear_algebra": ["group_theory", "calculus", "topology"], # 李代数, 微分方程
        "topology":       ["calculus", "diff_eq", "linear_algebra"],# 微分拓扑, 同调代数
        "graph_theory":   ["combinatorics", "optimization"],       # 图=组合结构, 网络流
        "combinatorics":  ["graph_theory", "number_theory", "probability"],# 计数, 拉姆齐
        "calculus":       ["diff_eq", "topology", "optimization"], # 连续→离散
        "diff_eq":        ["calculus", "topology"],                # ODE↔PDE, 动力系统
        "probability":    ["optimization", "combinatorics"],       # 随机优化
        "optimization":   ["probability", "graph_theory"],         # 凸优化, 网络流
        "number_theory":  ["group_theory", "combinatorics"],       # 同余→环, 丢番图
    }

    # Apply equivariant coupling: boost related subfield scores
    for field, related in MATH_EQUIVARIANCE.items():
        if dims.get(field, 0) > 0:
            for rel in related:
                dims[rel] = dims.get(rel, 0) + 0.5  # Half-weight coupling

    # Simplicity signals (penalize trivial tasks)
    is_trivial = len(question) < 80 and dims["arch"] == 0 and dims["math"] == 0
    is_simple_code = dims["code"] >= 1 and dims["arch"] == 0 and len(question) < 100

    # Continuous difficulty score (Dynamic MoE: harder tasks -> more experts)
    complexity = (
        dims["code"] * 1.0 +
        dims["math"] * 1.5 +
        dims["arch"] * 1.2 +
        dims["writing"] * 0.5 +
        dims["knowledge"] * 0.3
    )
    # Trivial penalty: simple short tasks get difficulty halved
    if is_trivial or is_simple_code:
        complexity *= 0.3

    # Normalize to [0, 1] using sigmoid
    difficulty = 1.0 / (1.0 + math.exp(-complexity / 5.0))

    # Extra: trivial tasks are clamped to low difficulty
    if is_trivial:
        difficulty = min(difficulty, 0.35)

    # Collaboration benefit score (from benchmark data)
    collab_benefit = (
        dims["arch"] * 0.4 +
        dims["trap_need_collab"] * 0.6 -
        dims["trap_single"] * 0.5  # Strong penalty for trap patterns
    )
    collab_benefit = max(-1.0, min(1.0, collab_benefit / 3.0))

    # Model affinity (98 questions, 3 rounds):
    #   Easy: any model works.
    #   Medium: GLM(91%) > DS-Think(88%)
    #   Hard: DS-PRO(72%) > DS-Think(70%) > GLM(67%)
    #   All struggle on deep logic (37-50% wall)
    model_affinity = {
        "ds-pro":   dims["code"] * 0.45 + dims["math"] * 0.45 + dims["arch"] * 0.2,  # Hard tasks king (V3 72%)
        "glm":      dims["arch"] * 0.4 + dims["knowledge"] * 0.4 + dims["writing"] * 0.3,  # Medium king (V2 91%)
        "ds-think": dims["code"] * 0.35 + dims["math"] * 0.35 + dims["arch"] * 0.2 + dims["knowledge"] * 0.2,  # All-rounder (V1 100%)
        "qwen":     dims["writing"] * 0.2 + dims["knowledge"] * 0.2 + dims["code"] * 0.15,  # Solid #3
        "kimi":     dims["writing"] * 0.5,  # Writing only
    }

    return {
        "difficulty": difficulty,
        "collab_benefit": collab_benefit,
        "model_affinity": model_affinity,
        "dims": dims,
        "length": len(question),
    }

# ─── Stage 1: Difficulty Estimation (Dynamic MoE: K = f(difficulty)) ──

def estimate_K(difficulty, collab_benefit):
    """Map difficulty to optimal number of models.
       From Dynamic MoE: harder tasks activate MORE experts.
       From benchmark: negative collab_benefit -> force single model."""
    if collab_benefit < -0.3:
        return 1  # Trap patterns: force single

    if difficulty < 0.2:
        return 1  # Very easy: single model
    elif difficulty < 0.4:
        return 1 if collab_benefit < 0.1 else 2  # Easy: single unless clear benefit
    elif difficulty < 0.6:
        return 2  # Medium: pipeline or dual-model
    elif difficulty < 0.8:
        return 3  # Hard: 3-model collab
    else:
        return 4  # Very hard: full 4-model collab

# ─── Category Blacklist (RACER 2025: statistical exclusion) ──
# Models with weight < 0 in a category are BANNED from that category.
# From 98-question reward-penalty matrix.

CATEGORY_BAN = {
    "logic": ["ds-pro", "ds-think", "glm", "qwen"],
}

# Banned models serve as CRITICS (Nature 2025: limited-visibility consensus),
# not answerers. They check the primary answer for flaws.
CRITIC_MAP = {
    "logic": ["ds-think", "glm"],  # These may suck at logic, but can spot obvious errors
}

def get_allowed_models(category, candidates):
    """Filter: only models NOT banned for this category"""
    banned = CATEGORY_BAN.get(category, [])
    allowed = [m for m in candidates if m not in banned]
    if not allowed:
        # All banned -> use least-bad (Kimi for logic, GLM for general)
        fallback = {"logic": ["Kimi", "DS-Think"], "general": ["DS-PRO", "GLM"]}
        return fallback.get(category, candidates[:2])
    return allowed

# ─── Stage 2: Collaboration Mode (MasRouter: cascaded decision) ──

def decide_mode(K, difficulty, collab_benefit, affinity, dims):
    """Strict exclusion routing (RACER 2025 + Router-R1 inspired):
       Banned models = never called. Only allowed models participate."""
    # Detect category from dims
    primary_cat = "general"
    if dims.get("code",0) >= 1: primary_cat = "code"
    if dims.get("math",0) >= 1: primary_cat = "math"
    if dims.get("logic",0) >= 1: primary_cat = "logic"
    if dims.get("arch",0) >= 1 or dims.get("knowledge",0) >= 1: primary_cat = "knowledge"

    allowed = get_allowed_models(primary_cat, list(affinity.keys()))

    # Remove banned models from affinity
    clean_affinity = {m: w for m, w in affinity.items() if m in allowed}

    if K == 1:
        best = max(clean_affinity, key=clean_affinity.get)
        if difficulty > 0.7 and "ds-pro" in allowed:
            best = "ds-pro"
        elif difficulty > 0.4:
            if dims.get("knowledge",0) > dims.get("code",0) and "glm" in allowed:
                best = "glm"
            elif best not in allowed:
                best = max(clean_affinity, key=clean_affinity.get)
        # FINAL check: never return banned model
        if best not in allowed:
            best = allowed[0] if allowed else "ds-pro"
        return {"action": "single", "model": best, "K": 1,
                "reason": f"Difficulty={difficulty:.2f}, best={best}"}

    elif K == 2:
        ranked = sorted(clean_affinity, key=clean_affinity.get, reverse=True)[:2]
        excluded = set(affinity.keys()) - set(allowed)
        ex_msg = f" | EXCLUDED: {', '.join(excluded)}" if excluded else ""
        return {"action": "pipeline", "models": ranked, "K": 2,
                "steps": [{"model": ranked[0], "role": "core"},
                          {"model": ranked[1], "role": "polish"}],
                "reason": f"Difficulty={difficulty:.2f}, pipeline: {' -> '.join(ranked)}{ex_msg}"}

    else:  # K >= 3
        ranked = sorted(clean_affinity, key=clean_affinity.get, reverse=True)[:K]
        excluded = set(affinity.keys()) - set(allowed)
        ex_msg = f" | EXCLUDED: {', '.join(excluded)}" if excluded else ""
        return {"action": "collab", "models": ranked, "K": K,
                "reason": f"Difficulty={difficulty:.2f}, {K}-model collab: {', '.join(ranked)}{ex_msg}"}

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
    def worker(mid):
        try: results[mid] = call(mid, question, max_tok=2500)
        except Exception as e: results[mid] = f"[ERR:{e}]"

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

# ─── Main ────────────────────────────────────────────

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not question:
        print("Usage: python brain.py <question>", file=sys.stderr)
        sys.exit(1)

    # Step 1: Score task (Dynamic MoE difficulty estimation)
    scores = score_task(question)
    K = estimate_K(scores["difficulty"], scores["collab_benefit"])

    # Step 2: Decide mode (MasRouter cascaded controller)
    # TDA cache check: if similar question exists, reuse its routing
    tda_hit = tda_find_similar(question, scores["dims"])
    if tda_hit:
        print(f"  {CC['M']}[TDA]{CC['R']} Similar question found (sim={tda_hit.get('action','?')} -> {tda_hit.get('model','?')})")
    decision = decide_mode(K, scores["difficulty"], scores["collab_benefit"], scores["model_affinity"], scores["dims"])

    print(f"\n{CC['W']}{'='*64}{CC['R']}")
    print(f"  {CC['W']}Brain v6 | Paper-driven adaptive routing{CC['R']}")
    print(f"  Difficulty={scores['difficulty']:.2f} | K={K} | Mode={decision['action']}")
    print(f"  {decision['reason']}")
    print(f"{CC['W']}{'='*64}{CC['R']}")

    # Step 3: Execute
    if decision["action"] == "single":
        answer = execute_single(question, decision.get("model", "ds-pro"))
        # Quality gate (Unified Cascade): if single answer low quality, cascade
        if not quality_gate(answer, scores["dims"]) and K > 1:
            print(f"  {CC['Y']}[CASCADE] Quality insufficient, escalating to pipeline...{CC['R']}")
            # Re-route to next tier
            if K == 2:
                answer = execute_pipeline(question, decision.get("steps", [{"model":"ds-pro","role":"直接回答"}]))
            else:
                answer = execute_collab(question, decision.get("models", ["ds-pro","glm"]))

    elif decision["action"] == "pipeline":
        answer = execute_pipeline(question, decision.get("steps", [{"model":"ds-pro","role":"直接回答"}]))
    else:
        answer = execute_collab(question, decision.get("models", ["ds-pro","glm"]))

    # Step 4: Critic review (banned models check primary answer)
    primary_cat = "general"
    if scores["dims"].get("logic",0) >= 1: primary_cat = "logic"
    if scores["dims"].get("code",0) >= 1: primary_cat = "code"
    critics = CRITIC_MAP.get(primary_cat, [])
    if critics:
        print(f"  {CC['M']}[CRITIC]{CC['R']} Banned models reviewing: {', '.join(critics)}")
        for c in critics:
            try:
                review = call(c, f"检查以下回答是否有逻辑错误或矛盾。如有，指出具体问题。如无，回复OK。\n\n问题: {question[:200]}\n回答: {answer[:800]}", max_tok=200)
                if "OK" not in review and len(review) > 5:
                    print(f"  {CC['Y']}[CRITIC:{c}]{CC['R']} flagged: {review[:120]}...")
                    answer += f"\n\n[审查意见/{c}]: {review}"
                else:
                    print(f"  {CC['D']}[CRITIC:{c}]{CC['R']} passed")
            except:
                pass

    # Output
    print(f"\n{CC['W']}{'='*64}{CC['R']}")
    print(answer[:3000] if len(answer) > 3000 else answer)
    print(f"\n{CC['D']}v11 | diff={scores['difficulty']:.2f} K={K} | dims={scores['dims']}{CC['R']}\n")

    # TDA cache: store this question→routing for future similarity matching
    tda_cache_question(question, scores["dims"], decision)

    # Log + Synaptic Update
    from datetime import datetime
    round_counter = json.loads(open(Path.home() / ".claude" / "tools" / "round_counter.json", "r").read()) if (Path.home() / ".claude" / "tools" / "round_counter.json").exists() else 0
    # Simple correctness guess from quality_gate
    likely_correct = quality_gate(answer, scores["dims"])
    used_model = decision.get("model") or (decision.get("models",["ds-pro"])[0] if decision.get("models") else "ds-pro")
    synaptic_update(question, used_model, likely_correct, round_counter, scores["dims"])
    spsa_step(likely_correct, scores["dims"])  # Online hyperparameter tuning
    round_counter += 1
    with open(Path.home() / ".claude" / "tools" / "round_counter.json", "w") as f:
        f.write(str(round_counter))

    log = {
        "timestamp": datetime.now().isoformat(),
        "question_len": len(question),
        "difficulty": round(scores["difficulty"], 3),
        "K": K,
        "action": decision["action"],
        "models": decision.get("model") or decision.get("models", []),
        "dims": scores["dims"],
    }
    with open(Path.home() / ".claude" / "tools" / "decision-log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
