#!/usr/bin/env python3
"""
parallel_ui.py — Real-time Parallel AI Execution Monitor
Shows brain regions + models running in parallel, live status.
"""

import sys, time, threading
from typing import Dict, List

REGION_MODEL_MAP = {
    "motor_cortex":     ["DS-PRO", "DS-Think"],
    "parietal_cortex":  ["DS-PRO", "QWEN3"],
    "prefrontal_cortex":["DS-PRO", "DS-Think"],
    "temporal_cortex":  ["GLM-4+", "QWEN3"],
    "language_area":    ["GLM-4+", "KIMI"],
    "visual_cortex":    ["KIMI", "QWEN3"],
}

REGION_NAMES_CN = {
    "motor_cortex":      "Motor Cortex (code)",
    "parietal_cortex":   "Parietal (math)",
    "prefrontal_cortex": "Prefrontal (logic)",
    "temporal_cortex":   "Temporal (knowledge)",
    "language_area":     "Language (writing)",
    "visual_cortex":     "Visual (multimodal)",
}


class ParallelMonitor:
    """Terminal UI for monitoring parallel LLM execution with brain regions."""

    def __init__(self):
        self.entries: List[dict] = []  # {region, model, status, start, end, len}
        self.lock = threading.Lock()
        self.width = 68

    def register_parallel(self, region: str, models: List[str]):
        """Register a region's models being called in parallel."""
        with self.lock:
            for m in models:
                self.entries.append({
                    "region": region,
                    "model": m,
                    "status": "pending",
                    "start": time.time(),
                    "end": 0,
                    "len": 0,
                })

    def start(self, model: str):
        with self.lock:
            for e in self.entries:
                if e["model"] == model and e["status"] == "pending":
                    e["status"] = "running"
                    e["start"] = time.time()
                    self._render()
                    return

    def done(self, model: str, response_len: int = 0):
        with self.lock:
            for e in self.entries:
                if e["model"] == model:
                    e["status"] = "done"
                    e["end"] = time.time()
                    e["len"] = response_len
                    self._render()
                    return

    def fail(self, model: str, error: str = ""):
        with self.lock:
            for e in self.entries:
                if e["model"] == model:
                    e["status"] = "failed"
                    e["end"] = time.time()
                    self._render()
                    return

    def _icon(self, status: str) -> str:
        return {"pending":"-","running":">","done":"*","failed":"X"}.get(status,"?")

    def _bar(self, status: str) -> str:
        if status == "running":
            bars = ["|","/","-","\\"]
            return bars[int(time.time()*4)%4]
        elif status == "done": return "#"
        elif status == "failed": return "!"
        return "."

    def _region_name(self, region: str) -> str:
        return REGION_NAMES_CN.get(region, region)

    def _render(self):
        lines = []
        lines.append(f"{'-'*self.width}")
        lines.append(f"  PARALLEL AI EXECUTION  |  BRAIN REGIONS ACTIVE")
        lines.append(f"  {'region':<24} {'model':<12} {'st':<4} {'time':>7}  {'output':>7}")

        for e in sorted(self.entries, key=lambda x: {"running":0,"pending":1,"done":2,"failed":3}.get(x["status"],4)):
            icon = self._icon(e["status"])
            bar  = self._bar(e["status"])
            rname = self._region_name(e["region"])
            elapsed = time.time() - e["start"] if e["start"] > 0 else 0
            if e["status"] == "done":
                tstr = f"{e['end']-e['start']:.1f}s"
            elif e["status"] == "running":
                tstr = f"{elapsed:.1f}s"
            else:
                tstr = "..."
            ostr = f"{e['len']}c" if e["status"]=="done" else "..."

            lines.append(f"  {icon} {rname:<22} {bar} {e['model']:<10} {tstr:>7}  {ostr:>7}")

        done = sum(1 for e in self.entries if e["status"]=="done")
        running = sum(1 for e in self.entries if e["status"]=="running")
        failed = sum(1 for e in self.entries if e["status"]=="failed")
        total = len(self.entries)
        lines.append(f"  {running} running | {done} done | {failed} failed | {total} total")
        lines.append(f"{'-'*self.width}")
        print("\n".join(lines), flush=True)

    def finalize(self):
        self._render()


def parallel_call(region_models: Dict[str, List[str]], question: str,
                  max_tok: int = 2000, temp: float = 0.3) -> Dict[str, str]:
    """
    Call multiple models across brain regions in parallel with live UI.

    Args:
        region_models: {region_name: [model1, model2, ...]}
        question: question text
        max_tok, temp: model parameters

    Returns:
        {model_name: response_text}
    """
    from engine.brain import call

    monitor = ParallelMonitor()
    results = {}
    all_models = []

    for region, models in region_models.items():
        monitor.register_parallel(region, models)
        all_models.extend(models)

    monitor._render()

    def worker(model):
        try:
            monitor.start(model)
            result = call(model, question, max_tok=max_tok, temp=temp)
            results[model] = result
            monitor.done(model, len(result))
        except Exception as e:
            results[model] = f"[ERR: {str(e)[:80]}]"
            monitor.fail(model, str(e)[:40])

    threads = [threading.Thread(target=worker, args=(m,)) for m in all_models]
    for t in threads: t.start()
    for t in threads: t.join()

    monitor.finalize()
    return results


if __name__ == "__main__":
    import random

    print("\n" + "=" * 68)
    print("  PARALLEL UI + BRAIN REGIONS TEST")
    print("=" * 68)

    monitor = ParallelMonitor()

    # Simulate: code question activates motor_cortex + prefrontal_cortex
    active_regions = {
        "motor_cortex":      ["DS-PRO", "DS-Think"],
        "prefrontal_cortex": ["DS-PRO", "DS-Think"],
        "temporal_cortex":   ["GLM-4+"],
    }

    for region, models in active_regions.items():
        monitor.register_parallel(region, models)

    all_models = []
    for ms in active_regions.values():
        all_models.extend(ms)

    def sim_worker(model):
        monitor.start(model)
        delay = 0.3 + random.random() * 1.5
        time.sleep(delay)
        if random.random() < 0.15:
            monitor.fail(model, "timeout")
        else:
            monitor.done(model, random.randint(400, 2500))

    threads = [threading.Thread(target=sim_worker, args=(m,)) for m in all_models]
    for t in threads: t.start()
    for t in threads: t.join()

    monitor.finalize()
    print("\n  [DONE] All models completed -- ready to synthesize answer.\n")
