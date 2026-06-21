#!/usr/bin/env python3
"""
Paper-Driven Reasoning Enhancer
  Chain-of-Thought (Wei et al. 2022): Force stepwise reasoning
  Self-Consistency (Wang et al. 2023): 3 reasoning paths → majority vote
  DeBERTa Confidence Gate (Nature 2025): Confidence threshold → escalate

Purpose: Patch the 35-50% logic wall by combining paper methods.
"""
import sys, json, os, threading, re, math
from pathlib import Path
from openai import OpenAI
from collections import Counter

CFG = json.loads(Path(os.path.expanduser('~/.claude/tools/llm-config.json')).read_text(encoding='utf-8'))
CLIENTS = {
    "DS-PRO": (OpenAI(api_key=CFG['deepseek_pro']['api_key'], base_url=CFG['deepseek_pro']['base_url']), CFG['deepseek_pro']['model']),
    "DS-Think": (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "deepseek-reasoner"),
    "GLM": (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "glm"),
    "QWEN": (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "qwen"),
    "Kimi": (OpenAI(api_key=CFG['kimi']['api_key'], base_url=CFG['kimi']['base_url']), CFG['kimi']['model']),
}

COT_PROMPT = """请一步步推理。先写下你的分析，然后给出最终答案。格式：
【推理过程】
(你的逐步推理)
【最终答案】
(答案)"""

SELF_CONSISTENCY_ROUNDS = 3

def ask(client, model, prompt, max_tok=800, temp=0.7):
    try:
        r = client.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}], temperature=temp, max_tokens=max_tok)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERR:{e}]"

def extract_final_answer(response):
    """Extract answer after 【最终答案】 marker"""
    m = re.search(r'【最终答案】\s*(.+)', response, re.S)
    if m: return m.group(1).strip()
    # Fallback: take last line
    lines = response.strip().split('\n')
    return lines[-1] if lines else response.strip()

def self_consistency(question, model_name, rounds=3):
    """Self-Consistency (Wang et al. 2023): Multiple reasoning paths → majority vote"""
    client, model = CLIENTS[model_name]
    answers = []
    reasonings = []
    for i in range(rounds):
        resp = ask(client, model, f"{COT_PROMPT}\n\n问题：{question}", temp=0.5 + i*0.2)
        ans = extract_final_answer(resp)
        answers.append(ans)
        reasonings.append(resp)
        print(f"    Path {i+1}: {ans[:60]}...")

    # Majority vote
    vote = Counter(answers).most_common(1)[0]
    confidence = vote[1] / rounds
    best_answer = vote[0]
    best_reasoning = reasonings[answers.index(best_answer)]
    return best_answer, confidence, best_reasoning

def confidence_gate(question, primary_model, secondary_model, threshold=0.66):
    """DeBERTa-inspired confidence gate (Nature 2025):
       Primary model answers → if confidence < threshold → escalate to secondary"""
    print(f"  [Gate] Primary: {primary_model}")
    answer, confidence, reasoning = self_consistency(question, primary_model, SELF_CONSISTENCY_ROUNDS)
    print(f"  [Gate] Confidence: {confidence:.0%} (threshold: {threshold:.0%})")

    if confidence >= threshold:
        return answer, confidence, [primary_model]
    else:
        print(f"  [Gate] Low confidence → escalating to {secondary_model}")
        answer2, conf2, reasoning2 = self_consistency(question, secondary_model, SELF_CONSISTENCY_ROUNDS)
        if conf2 > confidence:
            return answer2, conf2, [primary_model, secondary_model]
        else:
            return answer, confidence, [primary_model]

def multi_model_vote(question, models, rounds_per_model=2):
    """Multi-model voting for hardest problems. All models reason, majority wins."""
    all_answers = []
    for m in models:
        client, model = CLIENTS[m]
        for _ in range(rounds_per_model):
            resp = ask(client, model, f"{COT_PROMPT}\n\n问题：{question}")
            ans = extract_final_answer(resp)
            all_answers.append((m, ans))
            print(f"    [{m}] {ans[:50]}...")

    # Weighted by model strength (from eval: logic: Kimi>DS-Think>GLM>DS-PRO>QWEN)
    logic_weights = {"DS-PRO": 0.35, "DS-Think": 0.44, "GLM": 0.44, "QWEN": 0.42, "Kimi": 0.50}
    weighted = Counter()
    for m, ans in all_answers:
        weighted[ans] += logic_weights.get(m, 0.4)

    best = weighted.most_common(1)[0]
    return best[0], best[1] / (len(models) * rounds_per_model * 0.5)

def solve(question, difficulty_level="medium"):
    """Main entry: choose strategy based on difficulty"""
    print(f"\n{'='*60}")
    print(f"Reasoning Enhancer: {question[:60]}...")
    print(f"Difficulty: {difficulty_level}")
    print(f"{'='*60}")

    if difficulty_level == "easy":
        # Simple CoT with best logic model (Kimi)
        print("[Strategy] Single CoT (Kimi)")
        ans, conf, _ = confidence_gate(question, "Kimi", "DS-Think", 0.5)
    elif difficulty_level == "medium":
        # Self-consistency with escalation
        print("[Strategy] Self-Consistency + Confidence Gate")
        ans, conf, models_used = confidence_gate(question, "Kimi", "DS-Think", 0.66)
    else:
        # Full multi-model voting
        print("[Strategy] Multi-Model Voting (3 models)")
        ans, conf = multi_model_vote(question, ["Kimi", "DS-Think", "GLM"])
        models_used = ["Kimi", "DS-Think", "GLM"]

    print(f"\n{'='*60}")
    print(f"FINAL ANSWER: {ans}")
    print(f"Confidence: {conf:.0%} | Models used: {models_used}")
    print(f"{'='*60}\n")
    return ans, conf, models_used

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        solve(q, "hard" if len(q) > 100 else "medium")
    else:
        # Test with benchmark logic questions
        tests = [
            ("easy", "如果所有猫怕水，Tom是猫。Tom怕水吗？只回答怕或不怕。"),
            ("medium", "甲说乙说谎，乙说丙说谎，丙说甲乙都说谎。谁说真话？一步步推理。"),
            ("hard", "蒙提霍尔问题：三扇门一扇有奖。你选了一扇，主持人打开另一扇空门。你应该换门吗？为什么？给出概率计算。"),
        ]
        for diff, q in tests:
            solve(q, diff)
