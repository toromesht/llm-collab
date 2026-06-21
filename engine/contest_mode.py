#!/usr/bin/env python3
"""
Contest Mode — Autonomous Multi-Model Contestant
  1. Analyze question → difficulty + category
  2. Decide strategy: single / CoT / self-consistency / multi-model vote
  3. Execute with best model(s) per category (from 98-question evolved weights)
  4. Output answer + confidence + trace
"""
import sys, json, os, threading, time, re
from pathlib import Path
from openai import OpenAI

CFG = json.loads(Path(os.path.expanduser('~/.claude/tools/llm-config.json')).read_text(encoding='utf-8'))

CLIENTS = {
    "DS-PRO": (OpenAI(api_key=CFG['deepseek_pro']['api_key'], base_url=CFG['deepseek_pro']['base_url']), CFG['deepseek_pro']['model']),
    "Kimi":   (OpenAI(api_key=CFG['kimi']['api_key'], base_url=CFG['kimi']['base_url']), CFG['kimi']['model']),
    "GLM":    (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "glm"),
    "QWEN":   (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "qwen"),
    "DS-Think": (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "deepseek-reasoner"),
}

# Evolved category weights (from 98 questions)
CAT_WEIGHTS = {
    "math":      {"DS-PRO": 0.7, "DS-Think": 0.7, "GLM": 1.0, "QWEN": 0.7, "Kimi": 0.7},
    "code":      {"DS-PRO": 0.7, "DS-Think": 0.7, "GLM": 0.7, "QWEN": 0.4, "Kimi": 1.0},
    "logic":     {"DS-PRO": -0.2, "DS-Think": -0.2, "GLM": -0.2, "QWEN": -0.2, "Kimi": 0.1},
    "knowledge": {"DS-PRO": 0.4, "DS-Think": 1.0, "GLM": 1.0, "QWEN": 0.7, "Kimi": 0.1},
    "writing":   {"DS-PRO": 0.5, "DS-Think": 0.5, "GLM": 0.7, "QWEN": 0.5, "Kimi": 0.5},
    "chinese":   {"DS-PRO": 0.7, "DS-Think": 0.7, "GLM": 0.7, "QWEN": 0.7, "Kimi": 0.0},
    "science":   {"DS-PRO": 0.7, "DS-Think": 0.7, "GLM": 0.7, "QWEN": 0.7, "Kimi": 0.0},
    "safety":    {"DS-PRO": 1.0, "DS-Think": 1.0, "GLM": 1.0, "QWEN": 1.0, "Kimi": 0.7},
}

D = "\033[2m"; G = "\033[1;32m"; B = "\033[1;34m"; M = "\033[1;35m"; Y = "\033[1;33m"; R = "\033[0m"; W = "\033[1;37m"

def ask(client, model, prompt, max_tok=2000, temp=0.3):
    r = client.chat.completions.create(model=model,
        messages=[{"role":"user","content":prompt}], temperature=temp, max_tokens=max_tok)
    return r.choices[0].message.content.strip()

# ─── Step 1: Question Analysis ────────────────────────

def analyze(question):
    """Classify question category + difficulty"""
    q = question.lower()
    cats = {}
    # Math: arithmetic words + numbers + unit patterns
    if re.search(r'(数学|计算|方程|概率|极限|矩阵|积分|导数|几何|等于|总共|一共|多少|剩余|百分比|比例|几倍|几分之几)', q): cats["math"] = 1
    if re.search(r'(sell|buy|give|left|total|each|how many|how much|percent)', q, re.I): cats["math"] = 1
    if re.search(r'\d+.*\d+.*\d+', q) and len([c for c in q if c.isdigit()]) > 3: cats["math"] = 1  # Number-heavy question
    if re.search(r'(代码|编程|函数|算法|SQL|python|写一个|实现|class |def |import |select|create|设计.*架构|设计.*数据库|架构|分布式)', q, re.I): cats["code"] = 1
    if re.search(r'(推理|逻辑|说谎|悖论|证明|谎言|真相|假话|谁说|判断|真话|假话)', q): cats["logic"] = 1
    if re.search(r'(什么是|定义|历史|年份|作者|出自|知识)', q): cats["knowledge"] = 1
    if re.search(r'(写作|文章|翻译|润色|文言|诗|文)', q): cats["chinese"] = 1

    primary = max(cats, key=cats.get) if cats else "general"
    difficulty = "easy" if len(q) < 30 else ("medium" if len(q) < 150 else "hard")

    return {"primary": primary, "cats": cats, "difficulty": difficulty, "length": len(q)}

# ─── Step 2: Strategy Decision ────────────────────────

def decide_strategy(analysis):
    """Intelligent strategy selection based on question analysis"""
    primary = analysis["primary"]
    difficulty = analysis["difficulty"]

    # Logic problems: ALL models suck → must use self-consistency
    if primary == "logic":
        return {"strategy": "self_consistency", "models": ["DS-Think", "GLM", "Kimi"],
                "reason": "Logic weakness (all <50%) -> 3-model self-consistency voting"}

    # Hard problems: multi-model collaboration
    if difficulty == "hard":
        return {"strategy": "multi_model", "models": ["DS-PRO", "GLM", "DS-Think"],
                "reason": "Hard problem → 3-model parallel + synthesis"}

    # Code problems: DS-PRO single (proven best)
    if primary == "code":
        return {"strategy": "single", "model": "DS-PRO",
                "reason": "Code problem → DS-PRO single (100% eval, highest SQL density)"}

    # Math problems: GLM single (proven 90% math)
    if primary == "math":
        return {"strategy": "single", "model": "GLM",
                "reason": "Math problem → GLM single (90% math accuracy)"}

    # Knowledge/Chinese: GLM single
    if primary in ("knowledge", "chinese"):
        return {"strategy": "single", "model": "GLM",
                "reason": f"{primary} → GLM single (100% knowledge accuracy)"}

    # Default: DS-PRO with CoT
    return {"strategy": "cot", "model": "DS-PRO",
            "reason": "General → DS-PRO with Chain-of-Thought"}

# ─── Step 3: Execute Strategy ─────────────────────────

def execute_single(question, model):
    print(f"  {B}[{model}]{R} executing...")
    return ask(CLIENTS[model][0], CLIENTS[model][1], question)

def execute_cot(question, model):
    print(f"  {B}[{model} + CoT]{R} step-by-step reasoning...")
    cot_prompt = f"请一步步推理，然后给出最终答案。\n\n问题：{question}"
    return ask(CLIENTS[model][0], CLIENTS[model][1], cot_prompt, max_tok=2500)

def execute_self_consistency(question, models, rounds=3):
    print(f"  {M}[Self-Consistency]{R} {len(models)} models x {rounds} reasoning paths...")
    all_answers = []
    for m in models:
        for r in range(rounds):
            cot_prompt = f"请一步步推理，然后给出最终答案。\n\n问题：{question}"
            ans = ask(CLIENTS[m][0], CLIENTS[m][1], cot_prompt, max_tok=1500, temp=0.3 + r*0.3)
            # Extract final answer
            final = re.split(r'最终答案|答案为|答案：', ans)[-1].strip() if re.search(r'最终答案|答案为|答案：', ans) else ans.strip()
            all_answers.append((m, final))
            print(f"    [{m}] path {r+1}: {final[:50]}...")

    # Weighted vote
    from collections import Counter
    weights = {m: max(0.1, CAT_WEIGHTS["logic"].get(m, 0.3)) for m in models}
    weighted = Counter()
    for m, ans in all_answers:
        weighted[ans] += weights[m]

    best_ans, best_count = weighted.most_common(1)[0]
    confidence = best_count / (len(models) * rounds * 0.5)  # normalize
    return best_ans, min(confidence, 0.95)

def execute_multi_model(question, models):
    print(f"  {M}[Multi-Model]{R} {len(models)} models parallel...")
    results = {}
    def worker(m):
        try:
            results[m] = ask(CLIENTS[m][0], CLIENTS[m][1], question, max_tok=2000)
        except Exception as e:
            results[m] = f"[ERR:{e}]"

    threads = [threading.Thread(target=worker, args=(m,)) for m in models]
    for t in threads: t.start()
    for t in threads: t.join()

    valid = {m: r for m, r in results.items() if not r.startswith("[ERR")}
    if len(valid) <= 1:
        return list(valid.values())[0] if valid else "[ALL FAILED]", 0.3

    sections = "\n\n".join(f"=== {m} ===\n{r[:1500]}" for m, r in valid.items())
    synth = ask(CLIENTS["DS-PRO"][0], CLIENTS["DS-PRO"][1],
        f"综合以下{len(valid)}个专家的独立回答，保留所有技术细节，输出最优方案：\n\n{question}\n\n{sections}",
        max_tok=3000)
    return synth, 0.75

# ─── Main Entry ───────────────────────────────────────

def solve(question):
    print(f"\n{W}{'='*64}{R}")
    print(f"  {W}CONTEST MODE — Autonomous Intelligent Contestant{R}")
    print(f"  Question: {question[:80]}...")
    print(f"{W}{'='*64}{R}")

    # Analyze
    print(f"\n  {D}[Analyze] Classifying question...{R}")
    analysis = analyze(question)
    print(f"  Category: {analysis['primary']} | Difficulty: {analysis['difficulty']} | Length: {analysis['length']}c")

    # Decide
    print(f"\n  {D}[Decide] Choosing strategy...{R}")
    strategy = decide_strategy(analysis)
    print(f"  {G}Strategy: {strategy['strategy']}{R} | {strategy['reason']}")

    # Execute
    print(f"\n  {D}[Execute] Running...{R}")
    t0 = time.time()

    if strategy["strategy"] == "single":
        answer = execute_single(question, strategy["model"])
        confidence = 0.85
        trace = [f"Single {strategy['model']}"]
    elif strategy["strategy"] == "cot":
        answer = execute_cot(question, strategy["model"])
        confidence = 0.78
        trace = [f"CoT via {strategy['model']}"]
    elif strategy["strategy"] == "self_consistency":
        answer, confidence = execute_self_consistency(question, strategy["models"])
        trace = [f"Self-consistency: {strategy['models']} x {3} rounds"]
    elif strategy["strategy"] == "multi_model":
        answer, confidence = execute_multi_model(question, strategy["models"])
        trace = [f"Multi-model: {strategy['models']}"]
    else:
        answer = execute_single(question, "DS-PRO")
        confidence = 0.7
        trace = ["Fallback DS-PRO"]

    elapsed = time.time() - t0

    # Output
    print(f"\n{W}{'='*64}{R}")
    print(f"  {W}ANSWER{R}")
    print(f"{W}{'='*64}{R}")
    try:
        print(f"\n{answer}\n")
    except UnicodeEncodeError:
        print(f"\n[Answer generated ({len(answer)} chars) - encoding ok]\n")
    print(f"{D}{'='*64}{R}")
    print(f"  Strategy: {strategy['strategy']} | Time: {elapsed:.1f}s | Confidence: {confidence:.0%}")
    print(f"  Category: {analysis['primary']} | Difficulty: {analysis['difficulty']}")
    print(f"  Trace: {' → '.join(trace)}")
    print(f"{D}{'='*64}{R}\n")

    return {"answer": answer, "confidence": confidence, "strategy": strategy["strategy"],
            "time": elapsed, "analysis": analysis, "trace": trace}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        solve(" ".join(sys.argv[1:]))
    else:
        # Demo test
        tests = [
            "用Python写一个快速排序算法",
            "甲说乙说谎，乙说丙说谎，丙说甲乙都说谎。谁说真话？",
            "设计一个全球分布式金融数据库架构，需考虑多活容灾、一致性权衡、分库分表策略",
            "请解释量子纠缠及其在量子计算中的应用",
        ]
        for q in tests:
            solve(q)
