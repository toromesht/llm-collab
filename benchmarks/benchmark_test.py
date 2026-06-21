#!/usr/bin/env python3
"""
Benchmark: Single DS-PRO vs Multi-Model Collaboration (DS-PRO + Kimi + GLM + Qwen)
Test set: 8 questions from GSM8K / HumanEval / MMLU / reasoning
Metrics: Accuracy, Response Time, Answer Quality
"""
import sys, json, os, threading, time
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

B="\033[1;34m"; M="\033[1;35m"; C="\033[1;36m"; Y="\033[1;33m"
G="\033[1;32m"; R="\033[1;31m"; D="\033[2m"; X="\033[0m"

CFG = json.loads(Path(os.path.expanduser('~/.claude/tools/llm-config.json')).read_text(encoding='utf-8'))
DS=OpenAI(api_key=CFG['deepseek_pro']['api_key'],base_url=CFG['deepseek_pro']['base_url']); DSM=CFG['deepseek_pro']['model']
KI=OpenAI(api_key=CFG['kimi']['api_key'],base_url=CFG['kimi']['base_url']); KIM=CFG['kimi']['model']
SJ=OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'],base_url=CFG['sjtu_zhiyuan']['base_url'])

# ═══════════════════════════════════════════════════════
# Test Set (questions with known correct answers)
# ═══════════════════════════════════════════════════════
TESTS = [
    {
        "id": "MATH-1",
        "type": "math",
        "question": "一个商店以20%的折扣出售一件商品，折扣后的价格是240元。这件商品的原价是多少？请只输出最终答案数字。",
        "answer": "300",
        "tolerance": 5
    },
    {
        "id": "MATH-2",
        "type": "math",
        "question": "小明有48颗糖，他给了小红1/3，又给了小华剩下的1/4，最后小明还剩多少颗糖？请逐步计算并给出最终数字。",
        "answer": "24",
        "tolerance": 2
    },
    {
        "id": "CODE-1",
        "type": "code",
        "question": "用Python写一个函数 is_prime(n)，判断n是否为质数。只需输出函数代码。",
        "answer": "def is_prime(n):",  # keyword check
        "tolerance": 0
    },
    {
        "id": "CODE-2",
        "type": "code",
        "question": "用Python写一个二分查找函数 binary_search(arr, target)，返回索引或-1。只需输出函数代码。",
        "answer": "def binary_search",  # keyword check
        "tolerance": 0
    },
    {
        "id": "KNOW-1",
        "type": "knowledge",
        "question": "2025年诺贝尔物理学奖授予了哪两位科学家？因为什么贡献？请简要回答（50字内）。",
        "answer": "Hopfield",  # keyword: John Hopfield and Geoffrey Hinton for machine learning
        "tolerance": 0
    },
    {
        "id": "KNOW-2",
        "type": "knowledge",
        "question": "中国「十五五」规划的起止年份是什么？请直接回答年份。",
        "answer": "2026",
        "tolerance": 0
    },
    {
        "id": "REASON-1",
        "type": "reasoning",
        "question": "如果所有的猫都怕水，Tom是一只猫。那么Tom怕水吗？请用逻辑三段论回答。",
        "answer": "怕水",
        "tolerance": 0
    },
    {
        "id": "REASON-2",
        "type": "reasoning",
        "question": "甲说：乙在说谎。乙说：丙在说谎。丙说：甲和乙都在说谎。请问谁在说真话？请逐步推理。",
        "answer": "乙",
        "tolerance": 0
    },
]

def ask(client, model, prompt, max_tok=500):
    r = client.chat.completions.create(model=model,
        messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=max_tok)
    return r.choices[0].message.content.strip()

def check_answer(test, model_answer):
    """Simple accuracy check"""
    ans = model_answer.lower()
    expected = test["answer"].lower()
    if expected in ans:
        return 1.0
    # For math questions, extract numbers
    if test["type"] == "math":
        import re
        nums = re.findall(r'\d+', ans)
        if nums:
            last_num = int(nums[-1])
            target = int(expected)
            if abs(last_num - target) <= test.get("tolerance", 5):
                return 1.0
    return 0.0

# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  BENCHMARK: Single DS-PRO vs Multi-Model Collaboration{X}")
print(f"{B}  Test Set: {len(TESTS)} questions (Math/Code/Knowledge/Reasoning){X}")
print(f"{B}{'='*70}{X}")

results = {"single": [], "multi": []}

for test in TESTS:
    tid = test["id"]
    print(f"\n{C}[{tid}]{X} {test['question'][:60]}...")

    # ─── SINGLE MODE: DS-PRO only ───
    t0 = time.time()
    try:
        single_ans = ask(DS, DSM, test["question"], 500)
    except Exception as e:
        single_ans = f"[ERR: {e}]"
    t_single = time.time() - t0
    single_acc = check_answer(test, single_ans)
    results["single"].append({"id": tid, "time": t_single, "acc": single_acc, "len": len(single_ans)})
    print(f"  {D}[SINGLE] DS-PRO{X}: {t_single:.1f}s | acc={single_acc} | {single_ans[:80]}...")

    # ─── MULTI MODE: 4 models parallel -> DS-PRO synth ───
    multi_results = {}
    def worker(name, client, model):
        try:
            multi_results[name] = ask(client, model, test["question"], 500)
        except Exception as e:
            multi_results[name] = f"[ERR: {e}]"

    t0_m = time.time()
    threads = [
        threading.Thread(target=worker, args=("ds-pro", DS, DSM)),
        threading.Thread(target=worker, args=("kimi", KI, KIM)),
        threading.Thread(target=worker, args=("glm", SJ, "glm")),
        threading.Thread(target=worker, args=("qwen", SJ, "qwen")),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    # Synthesis
    raw_answers = "\n".join([f"--- {n} ---\n{a[:200]}" for n, a in multi_results.items()])
    synth_prompt = f"综合以下4个AI对同一问题的回答，给出最优最终答案（简洁，直接输出结果）：\n问题：{test['question']}\n{raw_answers}"
    try:
        multi_final = ask(DS, DSM, synth_prompt, 800)
    except Exception as e:
        multi_final = f"[ERR: {e}]"

    t_multi = time.time() - t0_m
    multi_acc = check_answer(test, multi_final)
    results["multi"].append({"id": tid, "time": t_multi, "acc": multi_acc, "len": len(multi_final)})
    print(f"  {M}[MULTI] 4-model synth{X}: {t_multi:.1f}s | acc={multi_acc} | {multi_final[:80]}...")

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  RESULTS{X}")
print(f"{B}{'='*70}{X}")

print(f"\n{'ID':<10} {'Single Time':<14} {'Multi Time':<14} {'Single Acc':<12} {'Multi Acc':<12}")
print(f"{'-'*62}")
total_single_time = 0
total_multi_time = 0
total_single_acc = 0
total_multi_acc = 0

for i, test in enumerate(TESTS):
    s = results["single"][i]
    m = results["multi"][i]
    total_single_time += s["time"]
    total_multi_time += m["time"]
    total_single_acc += s["acc"]
    total_multi_acc += m["acc"]
    acc_diff = "✓" if m["acc"] >= s["acc"] else f"{R}✗{X}"
    print(f"{test['id']:<10} {s['time']:.1f}s{'':>8} {m['time']:.1f}s{'':>8} {s['acc']:.0f}{'':>10} {m['acc']:.0f} {acc_diff}{'':>7}")

n = len(TESTS)
print(f"\n{G}{'='*70}{X}")
print(f"{G}  AGGREGATE{X}")
print(f"{G}  Total time  - Single: {total_single_time:.1f}s | Multi: {total_multi_time:.1f}s{X}")
print(f"{G}  Avg time    - Single: {total_single_time/n:.1f}s | Multi: {total_multi_time/n:.1f}s{X}")
print(f"{G}  Accuracy    - Single: {total_single_acc}/{n} ({total_single_acc/n*100:.0f}%) | Multi: {total_multi_acc}/{n} ({total_multi_acc/n*100:.0f}%){X}")

if total_single_acc > 0:
    acc_improve = (total_multi_acc - total_single_acc) / total_single_acc * 100
    print(f"{G}  Acc improvement: {acc_improve:+.0f}%{X}")
time_ratio = total_multi_time / total_single_time if total_single_time > 0 else 0
print(f"{G}  Time multiplier: {time_ratio:.1f}x (multi uses {time_ratio:.1f}x wall time){X}")
# But note: multi runs 4 models in parallel, so per-model time is actually lower
print(f"{G}  Effective parallel speedup: 4 models in ~{total_multi_time/n:.1f}s (vs 1 model in {total_single_time/n:.1f}s){X}")
print(f"{G}{'='*70}{X}\n")
