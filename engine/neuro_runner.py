#!/usr/bin/env python3
"""
neuro_runner.py — Multi-Objective Active Inference Harness

L_total = G(π) + α·KL(q||p) + (r - V)²
G(π) = Σ_k w_k(c) · G_k(π)   [6 objectives, context-weighted]

Uses ParetoFEP for routing. Called by hook_brain.py.
"""
import sys, json, os, time, threading, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STATUS = os.path.expanduser('~/.claude/tools/neuro_status.json')
REGION_NAMES = ["Motor", "Parietal", "PFC", "Temporal", "Language", "Visual"]

def ws(stage, models=None, done=False, region=None, diff=0, obj_info=None):
    d = {}
    if os.path.exists(STATUS):
        try: d = json.loads(open(STATUS,'r',encoding='utf-8').read())
        except: pass
    d['stage'] = stage; d['done'] = done; d['timestamp'] = time.time()
    if models: d['models'] = models
    if region: d['region'] = region
    if diff: d['difficulty'] = diff
    if obj_info: d['objectives'] = obj_info
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS,'w',encoding='utf-8') as f: json.dump(d, f, ensure_ascii=False)


def run(prompt: str):
    from engine.brainstem_wrapper import load as load_brainstem
    from engine.pareto_fep import ParetoFEP
    from engine.brain import call

    # ── 1. Multi-objective routing ──
    ws('routing')
    bs = load_brainstem()
    router = ParetoFEP(bs)
    decision = router.route(prompt)

    model_list = []; region_map = {}
    for rname, models in decision['selected_models'].items():
        for m in models: model_list.append(m); region_map[m] = rname

    initial = {m: "pending" for m in model_list}
    ws('routing_done', models=initial, region=decision['selected_models'],
       diff=decision['difficulty'],
       obj_info={'context': decision['context'], 'weights': decision['weights']})

    # ── 2. Parallel execution ──
    ws('executing', models=initial, region=decision['selected_models'])
    results = {}; lock = threading.Lock()

    def _worker(model):
        with lock: initial[model] = "running"; ws('executing', models=initial, region=decision['selected_models'])
        try:
            result = call(model, prompt, max_tok=2000)
            with lock: results[model] = result; initial[model] = "done"
        except Exception as e:
            with lock: results[model] = f"[ERR:{str(e)[:80]}]"; initial[model] = "failed"
        with lock: ws('executing', models=initial, region=decision['selected_models'])

    threads = [threading.Thread(target=_worker, args=(m,)) for m in model_list]
    for t in threads: t.start()
    for t in threads: t.join()

    # ── 3. Cortex synthesis ──
    ws('cortex', models=initial, region=decision['selected_models'])
    valid = {m: r for m, r in results.items() if not r.startswith("[ERR")}
    if len(valid) > 1:
        combined = "\n\n".join(f"=== {m} ({region_map.get(m,'?')}) ===\n{r}" for m, r in valid.items())
        try: final = call("ds-pro", f"综合以下{len(valid)}个模型的独立回答给出最佳答案:\n\n{prompt}\n\n{combined}", max_tok=3000)
        except: final = list(valid.values())[0][:3000]
    elif len(valid) == 1: final = list(valid.values())[0]
    else: final = "[ALL FAILED]"

    # ── 4. Learn: Pareto posterior + Hebbian coupling ──
    ws('learning', models=initial, region=decision['selected_models'])
    for idx in decision['top_policy_indices']:
        _, model = [(0,"ds-pro"),(0,"ds-think"),(0,"groq"),(1,"ds-pro"),(1,"ds-think"),(1,"qwen"),
                     (2,"ds-pro"),(2,"ds-think"),(2,"glm"),(3,"glm"),(3,"qwen"),(4,"glm"),(4,"kimi"),
                     (5,"kimi"),(5,"qwen")][idx]
        result = results.get(model, "")
        reward = 1.0 if result and not result.startswith("[ERR") else 0.0
        router.learn(prompt, decision, idx, reward)

    # ── 5. Done ──
    final_models = {m: ("done" if m in valid else "failed") for m in model_list}
    ws('done', models=final_models, region=decision['selected_models'], done=True)

    rf = os.path.expanduser('~/.synapseflow/brain/last_result.json')
    os.makedirs(os.path.dirname(rf), exist_ok=True)
    with open(rf,'w',encoding='utf-8') as f:
        json.dump({'output': final[:5000], 'routing': decision['selected_models'],
                   'context': decision['context'], 'pareto': decision.get('pareto_front',[]),
                   'timestamp': time.time()}, f, ensure_ascii=False)

    repo_log = os.path.expanduser('~/Desktop/collab-cloud/data/brain_activity.jsonl')
    os.makedirs(os.path.dirname(repo_log), exist_ok=True)
    with open(repo_log,'a',encoding='utf-8') as f:
        f.write(json.dumps({'routing': decision['selected_models'], 'models': final_models,
            'difficulty': decision['difficulty'], 'context': decision['context'],
            'pareto': decision.get('pareto_front',[]), 'timestamp': time.time()}, ensure_ascii=False)+'\n')

    return final


if __name__ == '__main__':
    prompt = sys.stdin.read().strip()
    if not prompt: sys.exit(1)
    print(run(prompt))
