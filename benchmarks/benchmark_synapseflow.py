#!/usr/bin/env python3
"""
SynapseFlow Public Benchmark Suite
MMLU | HumanEval | BoolQ | GSM8K | + Synaptic Training

Uses existing dataset files. Each run UPDATES synaptic pathways.

Usage: python benchmarks/benchmark_synapseflow.py [--bench all|mmlu|boolq|humaneval|math] [--max N]
"""

import sys, json, time, re, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.neuro_agent import NeuroOrchestrator
from engine.brain import call as brain_call

DATASET_DIR = Path(__file__).parent.parent / "dataset"
EVAL_DIR = Path(__file__).parent.parent / "eval"

def load_boolq(path=None):
    if path is None: path = DATASET_DIR / "boolq_100.json"
    with open(path, encoding="utf-8") as f: data = json.load(f)
    return [(item["q"], item["a"]) for item in data]

def load_mmlu(path=None):
    if path is None: path = DATASET_DIR / "mmlu_100.json"
    with open(path, encoding="utf-8") as f: data = json.load(f)
    return [(item["q"], item["a"]) for item in data]

def load_humaneval(path=None):
    if path is None: path = DATASET_DIR / "humaneval.json"
    with open(path, encoding="utf-8") as f: data = json.load(f)
    return [(item["q"], item["a"]) for item in data]

def load_gsm8k(path=None):
    if path is None: path = EVAL_DIR / "gsm8k_sample.jsonl"
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return [(item["question"], item["answer"]) for item in items]

def score_boolq(answer, expected):
    return str(expected).strip().lower() in answer.lower()[:50]

def score_mmlu(answer, expected):
    m = {"0":"A","1":"B","2":"C","3":"D"}
    exp = m.get(str(expected).strip().upper(), str(expected).strip().upper())
    return exp in answer.strip()[:10].upper()

def score_humaneval(answer, expected):
    al = answer.lower()
    return ("def " in al or "class " in al) and ("return " in al or "print(" in al)

def score_gsm8k(answer, expected):
    exp_match = re.search(r'####\s*([\d.,]+)', str(expected))
    if exp_match: exp_val = float(exp_match.group(1).replace(',',''))
    else:
        nums = re.findall(r'\d[\d,.]*\d|\d+', str(expected))
        if not nums: return False
        try: exp_val = float(nums[-1].replace(',',''))
        except: return False
    ans_nums = re.findall(r'\d[\d,.]*\d|\d+', str(answer))
    if not ans_nums: return False
    try: ans_val = float(ans_nums[-1].replace(',',''))
    except: return False
    return abs(ans_val - exp_val) < 0.5

BENCHMARKS = {
    "boolq": (load_boolq, score_boolq, "BoolQ", 100),
    "mmlu": (load_mmlu, score_mmlu, "MMLU", 100),
    "humaneval": (load_humaneval, score_humaneval, "HumanEval", 164),
    "math": (load_gsm8k, score_gsm8k, "GSM8K", 20),
}

def run_benchmark(bench_name, neuro, max_q=10):
    loader, scorer, display_name, total = BENCHMARKS[bench_name]
    questions = loader()[:max_q]

    neuro_ok = 0; single_ok = 0
    neuro_times = []; single_times = []

    print(f"\n{'='*60}")
    print(f"  {display_name} ({len(questions)} questions)")
    print(f"{'='*60}")

    for i, (q, a) in enumerate(questions):
        q = q[:800]
        print(f"  [{i+1}/{len(questions)}] ", end="", flush=True)

        t0 = time.time()
        try: na = neuro.solve(q)
        except: na = "[ERR]"
        nt = time.time() - t0
        nok = scorer(na, a)

        t0 = time.time()
        try: sa = brain_call("ds-pro", q, max_tok=800)
        except: sa = "[ERR]"
        st = time.time() - t0
        sok = scorer(sa, a)

        if nok: neuro_ok += 1
        if sok: single_ok += 1
        neuro_times.append(nt); single_times.append(st)

        m = "OK" if nok else ("~" if sok else "XX")
        print(f"{m} N={nok} S={sok} | n={nt:.1f}s s={st:.1f}s", flush=True)
        time.sleep(0.2)

    n = len(questions)
    npct = neuro_ok/n*100 if n else 0; spct = single_ok/n*100 if n else 0
    print(f"\n  {display_name}: Neuro={neuro_ok}/{n}={npct:.1f}% | Single={single_ok}/{n}={spct:.1f}%")
    return {"benchmark":display_name,"questions":n,"neuro_pct":round(npct,1),"single_pct":round(spct,1),
            "neuro_time":round(sum(neuro_times)/max(1,len(neuro_times)),1),
            "single_time":round(sum(single_times)/max(1,len(single_times)),1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="all", choices=["all","mmlu","boolq","humaneval","math"])
    p.add_argument("--max", type=int, default=8)
    args = p.parse_args()

    print("="*70)
    print("  SYNAPSEFLOW PUBLIC BENCHMARK + ONLINE TRAINING")
    print("="*70)

    neuro = NeuroOrchestrator()
    benches = list(BENCHMARKS.keys()) if args.bench=="all" else [args.bench]
    results = []

    for b in benches:
        r = run_benchmark(b, neuro, max_q=args.max)
        results.append(r)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for r in results:
        w = "NEURO" if r["neuro_pct"]>r["single_pct"] else ("SINGLE" if r["single_pct"]>r["neuro_pct"] else "TIE")
        print(f"  {r['benchmark']:<12} N={r['neuro_pct']:>5.1f}% S={r['single_pct']:>5.1f}% -> {w}")

    stats = neuro.get_stats()
    print(f"\n  Synaptic State after {sum(r['questions'] for r in results)} questions:")
    print(f"    Pathways: {stats['synapses']['active']} active, {stats['synapses']['hardened']} hardened, {stats['synapses']['pruned']} pruned")
    print(f"    Shortcuts: {stats['synapses']['shortcuts']}")
    print(f"    DA={stats['neuromodulation']['dopamine']:.3f} 5-HT={stats['neuromodulation']['serotonin']:.3f}")
    print(f"    sigma={stats['neuromodulation']['branching_ratio']:.3f}")

    out_path = EVAL_DIR / "synapseflow_benchmark.json"
    out_path.write_text(json.dumps({"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),"results":results,"synaptic_state":stats}, ensure_ascii=False, indent=2))
    print(f"\n  Saved: {out_path}")

if __name__ == "__main__":
    main()
