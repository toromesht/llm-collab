#!/usr/bin/env python3
"""
Vision Benchmark — test Kimi's vision capability on standard tasks.
Downloads public test images and evaluates accuracy.
"""
import json, os, base64, time, sys
from pathlib import Path
from openai import OpenAI

CFG_FILE = Path.home() / ".synapseflow" / "config.json"
if not CFG_FILE.exists():
    CFG_FILE = Path.home() / ".claude" / "tools" / "llm-config.json"
CFG = json.loads(CFG_FILE.read_text(encoding="utf-8"))

KI = OpenAI(api_key=CFG["kimi"]["api_key"], base_url=CFG["kimi"]["base_url"])
KIM = CFG["kimi"]["model"]

# Standard vision test tasks
TESTS = [
    # Object recognition (simple)
    {"id": "V1", "type": "object", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/320px-Google_2015_logo.svg.png",
     "question": "What company logo is this? Answer in one word.", "answer": "Google"},
    {"id": "V2", "type": "object", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/320px-PNG_transparency_demonstration_1.png",
     "question": "How many dice are visible in this image? Answer with a number.", "answer": "2"},

    # Color recognition
    {"id": "V3", "type": "color", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Solid_red.svg/320px-Solid_red.svg.png",
     "question": "What color is this image? Answer in one word.", "answer": "red"},
    {"id": "V4", "type": "color", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Solid_blue.svg/320px-Solid_blue.svg.png",
     "question": "What color is this image? Answer in one word.", "answer": "blue"},

    # Text recognition
    {"id": "V5", "type": "ocr", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/OpenAI_Logo.svg/320px-OpenAI_Logo.svg.png",
     "question": "What company logo is this? Answer in one word.", "answer": "OpenAI"},

    # Count
    {"id": "V6", "type": "count", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Three_geometric_shapes.svg/320px-Three_geometric_shapes.svg.png",
     "question": "How many distinct shapes are in this image? Answer with a number.", "answer": "3"},
]

def test_vision():
    print(f"\n{'='*60}")
    print(f"Vision Benchmark — Kimi ({KIM})")
    print(f"{'='*60}\n")

    results = []
    for test in TESTS:
        print(f"[{test['id']}] {test['type']}: {test['question']}")
        try:
            # Download image and encode as base64
            import urllib.request
            img_data = base64.b64encode(urllib.request.urlopen(test['url'], timeout=15).read()).decode()

            resp = KI.chat.completions.create(
                model=KIM,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}},
                    {"type": "text", "text": test['question']}
                ]}],
                max_tokens=50, temperature=0
            )
            ans = resp.choices[0].message.content.strip()
            correct = test['answer'].lower() in ans.lower()
            results.append({**test, "model_answer": ans, "correct": correct})
            print(f"  -> {ans[:60]} | {'OK' if correct else 'WRONG (expected '+test['answer']+')'}")
        except Exception as e:
            print(f"  -> ERR: {e}")
            results.append({**test, "model_answer": f"ERR:{e}", "correct": False})
        time.sleep(0.5)

    acc = sum(1 for r in results if r['correct']) / len(results) * 100
    print(f"\n{'='*60}")
    print(f"Vision Accuracy: {acc:.0f}% ({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"{'='*60}")

    # Save
    out = Path(__file__).parent / "benchmark_vision_latest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": KIM, "accuracy_pct": acc, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")

if __name__ == "__main__":
    test_vision()
