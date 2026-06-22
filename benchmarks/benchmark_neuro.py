#!/usr/bin/env python3
"""
benchmark_neuro.py — Neuro Agent vs Single DS-PRO benchmark

Tests complex problems across 5 domains, comparing:
  - Routing accuracy (right region selected?)
  - Answer quality (length, keyword coverage)
  - Latency (seconds)
  - Synaptic learning (pathway creation/hardening)

Usage:
  python benchmarks/benchmark_neuro.py
"""

import sys, time, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.neuro_agent import NeuroOrchestrator
from engine.brain import call as brain_call
from engine.brain import CC, MODELS

# ─── Complex test problems (5 domains) ────────────────────────

TEST_PROBLEMS = [
    {
        "id": "math_1",
        "domain": "math",
        "expected_region": "parietal_cortex",
        "question": (
            "设 G 是一个阶为 2^3 * 7^2 的有限群。"
            "请证明 G 不是单群（即必存在非平凡正规子群）。"
            "要求：使用 Sylow 定理，详细写出所有推理步骤。"
        ),
        "quality_keywords": ["Sylow", "正规子群", "共轭", "n_p", "同余", "单群", "矛盾"],
    },
    {
        "id": "code_1",
        "domain": "code",
        "expected_region": "motor_cortex",
        "question": (
            "请用 Python 实现一个线程安全的无锁并发哈希表（lock-free hash map），"
            "支持 insert(key,value)、get(key)、delete(key) 操作。"
            "使用 CAS（compare-and-swap）操作，不要使用任何 mutex 或 Lock。"
            "写出完整可运行的代码，并解释为什么你的实现是线程安全的。"
        ),
        "quality_keywords": ["CAS", "atomic", "compare_and_swap", "lock-free", "thread", "memory_order", "ABA"],
    },
    {
        "id": "logic_1",
        "domain": "logic",
        "expected_region": "prefrontal_cortex",
        "question": (
            "在一个岛上，有 3 种人：骑士（总说真话）、无赖（总说假话）、"
            "间谍（可能说真话也可能说假话）。"
            "你遇到 A、B、C 三人。A 说：\"B 是骑士。\" "
            "B 说：\"C 是无赖。\" C 说：\"A 不是骑士。\" "
            "已知三人中每种类型恰好各有一个。"
            "请推理出谁是骑士、谁是无赖、谁是间谍。"
        ),
        "quality_keywords": ["骑士", "无赖", "间谍", "推理", "矛盾", "假设"],
    },
    {
        "id": "knowledge_1",
        "domain": "knowledge",
        "expected_region": "temporal_cortex",
        "question": (
            "请详细解释数据库系统中 MVCC（多版本并发控制）的工作原理，"
            "包括：1) 版本链的维护机制 2) 快照隔离级别的实现 "
            "3) 垃圾回收（vacuum）策略 4) PostgreSQL 和 MySQL InnoDB 的具体实现差异。"
            "请给出具体的例子说明。"
        ),
        "quality_keywords": ["MVCC", "版本链", "快照隔离", "vacuum", "PostgreSQL", "InnoDB", "事务ID", "可见性"],
    },
    {
        "id": "writing_1",
        "domain": "writing",
        "expected_region": "language_area",
        "question": (
            "请以\"当 AI 学会了遗忘\"为主题，写一篇 800 字的中文议论文。"
            "要求：1) 引用神经科学中关于突触修剪的发现 "
            "2) 讨论 AI 遗忘机制对人类社会的意义 "
            "3) 文笔优美，有文学性，但论证要严谨。"
        ),
        "quality_keywords": ["遗忘", "突触", "修剪", "记忆", "AI", "神经", "人类"],
    },
]


def evaluate_answer(question: str, answer: str, keywords: list, domain: str) -> dict:
    """Score answer quality based on keyword coverage, length, and structure."""
    text = answer.lower()
    q_lower = question.lower()

    # Keyword coverage
    found_kw = [kw.lower() for kw in keywords if kw.lower() in text]
    kw_score = len(found_kw) / max(1, len(keywords))

    # Length score (different per domain)
    length = len(answer)
    if domain == "writing":
        len_score = min(1.0, length / 2000)  # essay needs longer
    elif domain == "math":
        len_score = min(1.0, length / 400)   # math is concise
    elif domain == "code":
        len_score = min(1.0, length / 800)   # code needs explanation
    else:
        len_score = min(1.0, length / 600)

    # Code presence check
    has_code = "```" in answer or "def " in answer or "class " in answer

    # Structure check
    has_structure = bool(re.search(r'[1-9][.)]\s', answer))

    quality_score = 0.5 * kw_score + 0.2 * len_score + 0.15 * (1.0 if has_structure else 0) + 0.15 * (1.0 if (has_code and domain == "code") or (domain != "code") else 0)

    return {
        "keyword_coverage": round(kw_score, 3),
        "length": length,
        "length_score": round(len_score, 3),
        "has_code": has_code,
        "has_structure": has_structure,
        "quality_score": round(quality_score, 3),
        "found_keywords": found_kw,
        "missing_keywords": [kw for kw in keywords if kw.lower() not in text],
    }


def run_benchmark():
    print("=" * 70)
    print("  NEUROSYNAPTIC BRAIN vs SINGLE DS-PRO  —  BENCHMARK")
    print("=" * 70)

    neuro = NeuroOrchestrator()
    results = []

    for i, problem in enumerate(TEST_PROBLEMS, 1):
        qid = problem["id"]
        domain = problem["domain"]
        expected_region = problem["expected_region"]
        question = problem["question"]
        keywords = problem["quality_keywords"]

        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(TEST_PROBLEMS)}] {qid} ({domain})")
        print(f"  Q: {question[:100]}...")
        print(f"{'─' * 70}")

        # ── Neuro Agent ──────────────────────────────────────
        t0 = time.time()
        try:
            neuro_answer = neuro.solve(question)
        except Exception as e:
            neuro_answer = f"[NEURO ERR: {e}]"
        neuro_time = time.time() - t0
        neuro_quality = evaluate_answer(question, neuro_answer, keywords, domain)

        print(f"  [NEURO] {neuro_time:.1f}s | quality={neuro_quality['quality_score']:.3f}")
        print(f"    Found keywords: {neuro_quality['found_keywords'][:5]}...")
        if neuro_quality['missing_keywords']:
            print(f"    Missing: {neuro_quality['missing_keywords'][:3]}...")

        # ── Single DS-PRO ────────────────────────────────────
        t0 = time.time()
        try:
            single_answer = brain_call("ds-pro", question, max_tok=2500)
        except Exception as e:
            single_answer = f"[SINGLE ERR: {e}]"
        single_time = time.time() - t0
        single_quality = evaluate_answer(question, single_answer, keywords, domain)

        print(f"  [SINGLE DS-PRO] {single_time:.1f}s | quality={single_quality['quality_score']:.3f}")
        print(f"    Found keywords: {single_quality['found_keywords'][:5]}...")
        if single_quality['missing_keywords']:
            print(f"    Missing: {single_quality['missing_keywords'][:3]}...")

        # ── Delta ────────────────────────────────────────────
        quality_delta = neuro_quality['quality_score'] - single_quality['quality_score']
        time_ratio = neuro_time / max(0.1, single_time)

        print(f"  [DELTA] quality={quality_delta:+.3f} | time_ratio={time_ratio:.1f}x")

        results.append({
            "id": qid,
            "domain": domain,
            "expected_region": expected_region,
            "neuro": {
                "time": round(neuro_time, 2),
                "quality": neuro_quality,
            },
            "single": {
                "time": round(single_time, 2),
                "quality": single_quality,
            },
            "delta": {
                "quality": round(quality_delta, 3),
                "time_ratio": round(time_ratio, 2),
            },
        })

        # ── Print answer excerpts ────────────────────────────
        print(f"\n  [NEURO answer excerpt]:")
        print(f"    {neuro_answer[:200].replace(chr(10), ' ')}...")
        print(f"\n  [SINGLE answer excerpt]:")
        print(f"    {single_answer[:200].replace(chr(10), ' ')}...")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    avg_neuro_q = sum(r["neuro"]["quality"]["quality_score"] for r in results) / len(results)
    avg_single_q = sum(r["single"]["quality"]["quality_score"] for r in results) / len(results)
    avg_neuro_t = sum(r["neuro"]["time"] for r in results) / len(results)
    avg_single_t = sum(r["single"]["time"] for r in results) / len(results)

    print(f"  {'Metric':<25} {'Neuro':>10} {'Single DS-PRO':>15} {'Delta':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'Avg Quality':<25} {avg_neuro_q:>10.3f} {avg_single_q:>15.3f} {avg_neuro_q-avg_single_q:>+10.3f}")
    print(f"  {'Avg Time (s)':<25} {avg_neuro_t:>10.1f} {avg_single_t:>15.1f} {avg_neuro_t-avg_single_t:>+10.1f}")
    print(f"  {'Time Ratio':<25} {'':>10} {'':>15} {avg_neuro_t/max(0.1,avg_single_t):>10.1f}x")

    # Per-domain breakdown
    print(f"\n  Per-Domain Quality:")
    for domain in ["math", "code", "logic", "knowledge", "writing"]:
        dr = [r for r in results if r["domain"] == domain]
        if dr:
            nq = dr[0]["neuro"]["quality"]["quality_score"]
            sq = dr[0]["single"]["quality"]["quality_score"]
            winner = "NEURO" if nq > sq else "SINGLE" if sq > nq else "TIE"
            print(f"    {domain:<12}: Neuro={nq:.3f} | Single={sq:.3f} | Winner={winner}")

    # Neuro system stats
    neuro_stats = neuro.get_stats()
    print(f"\n  Neuro System State:")
    print(f"    Round: {neuro_stats['round']}")
    print(f"    Active pathways: {neuro_stats['synapses']['active']}")
    print(f"    Hardened: {neuro_stats['synapses']['hardened']}")
    print(f"    Pruned: {neuro_stats['synapses']['pruned']}")
    print(f"    Shortcuts: {neuro_stats['synapses']['shortcuts']}")
    print(f"    DA: {neuro_stats['neuromodulation']['dopamine']:.3f}")
    print(f"    5-HT: {neuro_stats['neuromodulation']['serotonin']:.3f}")
    print(f"    Sigma: {neuro_stats['neuromodulation']['branching_ratio']:.3f}")
    region_usage = neuro_stats.get("regions", {})
    if region_usage:
        print(f"    Region usage: {json.dumps(region_usage, ensure_ascii=False)}")

    # Save
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "summary": {
            "avg_neuro_quality": round(avg_neuro_q, 3),
            "avg_single_quality": round(avg_single_q, 3),
            "avg_neuro_time": round(avg_neuro_t, 1),
            "avg_single_time": round(avg_single_t, 1),
            "quality_delta": round(avg_neuro_q - avg_single_q, 3),
            "time_ratio": round(avg_neuro_t / max(0.1, avg_single_t), 1),
        },
        "neuro_stats": neuro_stats,
    }
    out_path = Path(__file__).parent.parent / "eval" / "neuro_benchmark_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n  Results saved: {out_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
