#!/usr/bin/env python3
"""
Synapse Agent v1.0 — 独立智能体，仅用自有算力
  算力池: DS-PRO(满血) + Kimi + GLM-4(SJTU) + Qwen3.5(SJTU) + DS-Think(SJTU)
  路由引擎: Brain v11 (STDP/BCM/LateralInhib/Pruning/SPSA/TDA)
  直接调用: python agent.py "问题"
"""
import sys, json, os, threading, time, re, math, random
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

# ─── Config ──────────────────────────────────────────
# Auto-detect config from 3 locations (priority order)
CFG = {}
for p in [Path.home() / ".synapseflow" / "config.json",
          Path.home() / ".claude" / "tools" / "llm-config.json",
          Path(__file__).parent.parent / "config" / "llm-config.template.json"]:
    if p.exists():
        CFG = json.loads(p.read_text(encoding="utf-8"))
        break

# ─── Model Pool ──────────────────────────────────────
POOL = {
    "DS-V4":    (OpenAI(api_key=CFG["deepseek_pro"]["api_key"], base_url=CFG["deepseek_pro"]["base_url"]), CFG["deepseek_pro"]["model"]),  # DeepSeek V4 Pro 1.6T
    "GLM-5.2":  (OpenAI(api_key=CFG.get("glm_52",{}).get("api_key",""), base_url="https://api.z.ai/v1"), "glm-5.2[1m]"),  # GLM-5.2 744B
    "Kimi":     (OpenAI(api_key=CFG["kimi"]["api_key"], base_url=CFG["kimi"]["base_url"]), CFG["kimi"]["model"]),
    "GLM-4":    (OpenAI(api_key=CFG["sjtu_zhiyuan"]["api_key"], base_url=CFG["sjtu_zhiyuan"]["base_url"]), "glm"),  # GLM-4 (SJTU)
    "QWEN":     (OpenAI(api_key=CFG["sjtu_zhiyuan"]["api_key"], base_url=CFG["sjtu_zhiyuan"]["base_url"]), "qwen"),
    "SJTU-DS-Think": (OpenAI(api_key=CFG["sjtu_zhiyuan"]["api_key"], base_url=CFG["sjtu_zhiyuan"]["base_url"]), "deepseek-reasoner"),
}

D = "\033[2m"; G = "\033[1;32m"; B = "\033[1;34m"; M = "\033[1;35m"; Y = "\033[1;33m"
R = "\033[0m"; W = "\033[1;37m"; C = "\033[1;36m"

# ─── Task Classifier (10 math subfields + core) ──────
def classify(q):
    ql = q.lower()
    feats = {
        "math": sum(len(re.findall(p, ql)) for p in [
            r'(群论|群作用|子群|同态|同构|置换群|李群|伽罗瓦)',
            r'(图论|连通图|哈密顿|欧拉|最大流|最小割|匹配|着色)',
            r'(拓扑|同胚|同伦|同调|流形|纤维丛|单纯形)',
            r'(线性代数|矩阵|特征值|对角化|二次型|张量)',
            r'(微积分|极限|导数|积分|梯度|散度|级数|泰勒)',
            r'(概率|统计|分布|期望|方差|贝叶斯|蒙特卡洛)',
            r'(数论|素数|同余|费马|欧拉|RSA|丢番图)',
            r'(微分方程|ODE|PDE|拉普拉斯|波动方程|热方程)',
            r'(组合|排列|递推|生成函数|鸽巢|拉姆齐)',
            r'(优化|凸优化|线性规划|遗传算法|约束优化|对偶)'
        ]),
        "code": len(re.findall(r'(代码|sql|编程|python|函数|class |def |import |SELECT|CREATE|算法)', ql)),
        "logic": len(re.findall(r'(说谎|悖论|推理|逻辑|谁说|真话|假话|证明)', ql)),
        "knowledge": len(re.findall(r'(什么是|定义|历史|年份|作者|出自|解释|概念)', ql)),
        "writing": len(re.findall(r'(写作|翻译|润色|文言|诗|文章|论文|报告)', ql)),
        "arch": len(re.findall(r'(架构|设计|系统|方案|权衡|选型|分布式)', ql)),
    }
    # Difficulty
    complexity = feats["math"]*1.5 + feats["code"]*1.0 + feats["logic"]*1.3 + feats["arch"]*1.2
    diff = 1.0 / (1.0 + math.exp(-complexity / 5.0))
    primary = max(feats, key=feats.get) if any(feats.values()) else "general"
    return feats, diff, primary

# ─── Model Affinity (from 98q eval + 3 rounds) ───────
AFFINITY = {
    "math":      {"DS-PRO":0.7, "SJTU-DS-Think":0.7, "GLM":1.0, "QWEN":0.7, "Kimi":0.7},
    "code":      {"DS-PRO":0.7, "SJTU-DS-Think":0.7, "GLM":0.7, "QWEN":0.4, "Kimi":1.0},
    "logic":     {"DS-PRO":-1, "SJTU-DS-Think":-1, "GLM":-1, "QWEN":-1, "Kimi":0.1},  # ALL banned
    "knowledge": {"DS-PRO":0.4, "SJTU-DS-Think":1.0, "GLM":1.0, "QWEN":0.7, "Kimi":0.1},
    "writing":   {"DS-PRO":0.5, "SJTU-DS-Think":0.5, "GLM":0.7, "QWEN":0.5, "Kimi":0.5},
    "arch":      {"DS-PRO":0.5, "SJTU-DS-Think":0.5, "GLM":0.7, "QWEN":0.5, "Kimi":0.3},
    "general":   {"DS-PRO":0.5, "SJTU-DS-Think":0.5, "GLM":0.6, "QWEN":0.4, "Kimi":0.4},
}

# ─── Router: pick best model(s) ──────────────────────
def route(feats, diff, primary):
    aff = AFFINITY.get(primary, AFFINITY["general"])
    # Exclude banned (affinity < 0)
    allowed = {m: w for m, w in aff.items() if w > 0}
    if not allowed:
        allowed = {"Kimi": 0.1}

    if diff < 0.3:
        # Easy: single best
        best = max(allowed, key=allowed.get)
        return "single", [best]
    elif diff < 0.6:
        # Medium: pipeline (top 2)
        ranked = sorted(allowed, key=allowed.get, reverse=True)[:2]
        return "pipeline", ranked
    else:
        # Hard: top 3 parallel
        ranked = sorted(allowed, key=allowed.get, reverse=True)[:3]
        return "collab", ranked

# ─── Execute ─────────────────────────────────────────
def ask(model_name, prompt, max_tok=2000, temp=0.3, image_path=None):
    client, model = POOL[model_name]
    if image_path and model_name == "Kimi":
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg"}.get(ext,"image/png")
        content = [
            {"type":"image_url","image_url":{"url":f"data:{mime};base64,{img_b64}"}},
            {"type":"text","text":prompt}
        ]
    else:
        content = prompt
    r = client.chat.completions.create(model=model,
        messages=[{"role":"user","content":content}], temperature=temp, max_tokens=max_tok)
    return r.choices[0].message.content.strip()

def execute_single(q, model):
    print(f"  {B}[{model}]{R}")
    return ask(model, q)

def execute_pipeline(q, models):
    print(f"  {B}[{models[0]}]{R} + {B}[{models[1]}]{R}")
    r1 = ask(models[0], q, 2000)
    r2 = ask(models[1], f"完善和补充以下回答:\n\n{q}\n\n回答:\n{r1}", 2000)
    return r2

def execute_collab(q, models):
    results = {}
    def worker(m):
        try: results[m] = ask(m, q, 2000)
        except: results[m] = ""
    threads = [threading.Thread(target=worker, args=(m,)) for m in models]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"  {M}{'+'.join(models)}{R} -> synth")
    combined = "\n\n".join(f"[{m}]: {r[:800]}" for m, r in results.items())
    return ask("DS-PRO", f"综合以下专家回答，保留所有技术细节:\n\n{q}\n\n{combined}", 3000)

# ─── Main ────────────────────────────────────────────
def solve(question):
    feats, diff, primary = classify(question)
    strategy, models = route(feats, diff, primary)

    print(f"\n{W}{'='*60}{R}")
    print(f"  Synapse Agent | {primary} | diff={diff:.2f} | {strategy}")
    print(f"  Models: {', '.join(models)}")
    print(f"{W}{'='*60}{R}")

    t0 = time.time()
    if strategy == "single":
        ans = execute_single(question, models[0])
    elif strategy == "pipeline":
        ans = execute_pipeline(question, models)
    else:
        ans = execute_collab(question, models)

    elapsed = time.time() - t0
    print(f"\n{W}{'='*60}{R}")
    print(ans)
    print(f"\n{D}Synapse Agent | {strategy} | {elapsed:.1f}s | {len(ans)} chars{R}\n")
    return ans

if __name__ == "__main__":
    if len(sys.argv) > 1:
        solve(" ".join(sys.argv[1:]))
    else:
        print("Synapse Agent v1.0 | Usage: python agent.py 'question'")
