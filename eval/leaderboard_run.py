#!/usr/bin/env python3
"""Run contest mode against GSM8K sample, compare to public leaderboard"""
import sys, json, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.contest_mode import solve

sys.stdout.reconfigure(encoding='utf-8')

# Load GSM8K sample
with open(os.path.join(os.path.dirname(__file__), 'gsm8k_sample.jsonl'), 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f if line.strip()]

print(f"{'='*70}")
print(f"LEADERBOARD COMPARISON: GSM8K Benchmark (20 questions)")
print(f"{'='*70}")

correct = 0; results = []
for i, q in enumerate(questions):
    question = q['question']
    expected = q['answer'].split('####')[-1].strip()
    print(f"\n--- Q{i+1}: {question[:80]}...")

    try:
        result = solve(question)
        answer = result['answer']
        # Extract number from answer
        nums = re.findall(r'\d+\.?\d*', answer)
        if nums and expected.replace(',','') in str(nums[-1]):
            correct += 1
            print(f"  CORRECT (expected {expected})")
            results.append(1)
        else:
            print(f"  WRONG (expected {expected}, got {' '.join(nums[-3:]) if nums else 'no number'})")
            results.append(0)
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append(0)

    time.sleep(0.5)  # Rate limit

pct = correct / len(questions) * 100
print(f"\n{'='*70}")
print(f"RESULTS: {correct}/{len(questions)} ({pct:.1f}%)")
print(f"{'='*70}")

# Leaderboard comparison
print(f"""
PUBLIC LEADERBOARD (GSM8K 5-shot):
  GPT-4o:       92-95%
  Claude 3.5:   88-92%
  Gemini 1.5:   87-91%
  DeepSeek-V3:  85-89%
  Qwen 2.5:     83-87%
  Llama 3 70B:  80-85%

CONTEST MODE (20 questions, 0-shot):
  SynapseFlow:  {pct:.0f}%

NOTE: Public scores are 5-shot (5 examples given before test).
      This test is 0-shot (no examples). 0-shot is ~5-15% lower.
""")

# Save
with open(os.path.join(os.path.dirname(__file__), 'leaderboard_result.json'), 'w', encoding='utf-8') as f:
    json.dump({"benchmark": "GSM8K", "questions": len(questions), "correct": correct,
               "pct": round(pct, 1), "mode": "0-shot", "details": results,
               "public_baselines": {"GPT-4o": 93, "Claude 3.5": 90, "Gemini 1.5": 89, "DeepSeek-V3": 87}},
              f, ensure_ascii=False, indent=2)
