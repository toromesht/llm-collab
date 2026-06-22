#!/usr/bin/env python3
"""
neuro_runner.py — Unified harness runner.
Called by hook_brain.py in background thread.

Pipeline:
  1. CortexRouter: MCTS + Hebbian + region differentiation → routing decision
  2. Execute: parallel model calls via brain.py call()
  3. Learn: Hebbian update + region differentiation + MCTS backprop
  4. Cortex validation: DS-PRO review
  5. Persistent: save beliefs, pathways, state
"""
import sys, json, os, time, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STATUS_FILE = os.path.expanduser('~/.claude/tools/neuro_status.json')

_REGION_NAMES = ["Motor", "Parietal", "PFC", "Temporal", "Language", "Visual"]

def write_status(stage, models=None, done=False, region=None, difficulty=0):
    d = {}
    if os.path.exists(STATUS_FILE):
        try: d = json.loads(open(STATUS_FILE,'r',encoding='utf-8').read())
        except: pass
    d['stage'] = stage
    d['done'] = done
    d['timestamp'] = time.time()
    if models: d['models'] = models
    if region: d['region'] = region
    if difficulty: d['difficulty'] = difficulty
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)


def run(prompt: str):
    """Full harness pipeline."""
    from engine.brainstem_wrapper import load as load_brainstem
    from engine.cortex_router import CortexRouter, REGION_NAMES
    from engine.brain import call

    # ── Stage 1: Cortex Routing (MCTS + Hebbian + differentiation) ──
    write_status('cortex_routing')
    bs = load_brainstem()
    router = CortexRouter(bs)
    decision = router.route(prompt)

    # Build model list with regions
    model_list = []
    region_map = {}
    for rname, models in decision['selected_models'].items():
        for m in models:
            model_list.append(m)
            region_map[m] = rname

    # Write status: show models + regions
    initial_models = {m: "pending" for m in model_list}
    write_status('routing_done', models=initial_models,
                 region=decision['selected_models'],
                 difficulty=decision['difficulty'])

    # ── Stage 2: Parallel Model Execution ──
    write_status('executing', models=initial_models,
                 region=decision['selected_models'])
    results = {}
    lock = threading.Lock()

    def _worker(model):
        nonlocal results
        with lock:
            initial_models[model] = "running"
            write_status('executing', models=initial_models,
                        region=decision['selected_models'])
        try:
            result = call(model, prompt, max_tok=2000)
            with lock:
                results[model] = result
                initial_models[model] = "done"
        except Exception as e:
            with lock:
                results[model] = f"[ERR: {str(e)[:100]}]"
                initial_models[model] = "failed"
        with lock:
            write_status('executing', models=initial_models,
                        region=decision['selected_models'])

    threads = [threading.Thread(target=_worker, args=(m,)) for m in model_list]
    for t in threads: t.start()
    for t in threads: t.join()

    # ── Stage 3: Cortex Synthesis (DS-PRO) ──
    write_status('cortex', models=initial_models,
                 region=decision['selected_models'])
    valid_results = {m: r for m, r in results.items()
                     if not r.startswith("[ERR")}
    if len(valid_results) > 1:
        combined = "\n\n".join(f"=== {m} ({region_map.get(m,'?')}) ===\n{r}"
                               for m, r in valid_results.items())
        try:
            final = call("ds-pro",
                f"综合以下{len(valid_results)}个模型的独立回答，取长补短，给出一致的最佳答案:\n\n{prompt}\n\n{combined}",
                max_tok=3000)
            results['_synthesis'] = final
        except Exception:
            results['_synthesis'] = list(valid_results.values())[0][:3000]
    elif len(valid_results) == 1:
        results['_synthesis'] = list(valid_results.values())[0]
    else:
        results['_synthesis'] = "[ALL FAILED]"

    # ── Stage 4: Learn (Hebbian + differentiation + MCTS) ──
    write_status('learning', models=initial_models,
                 region=decision['selected_models'])
    features = router._build_features(prompt)
    active_regions_list = [_REGION_NAMES.index(r)
                           for r in decision['active_regions']]
    action_indices = decision.get('top_action_indices', [])
    rewards = [1.0 if r and not r.startswith("[ERR") else 0.0
               for m, r in results.items() if m != '_synthesis']
    rewards = rewards[:len(action_indices)]  # Align lengths
    if len(rewards) < len(action_indices):
        rewards += [0.5] * (len(action_indices) - len(rewards))

    router.learn(features, active_regions_list, action_indices, rewards)

    # ── Stage 5: Done ──
    final_models = {m: ("done" if m in valid_results else "failed")
                    for m in model_list}
    write_status('done', models=final_models,
                 region=decision['selected_models'], done=True)

    # Save result and state
    rf = os.path.expanduser('~/.synapseflow/brain/last_result.json')
    os.makedirs(os.path.dirname(rf), exist_ok=True)
    state = router.get_state()
    with open(rf, 'w', encoding='utf-8') as f:
        json.dump({
            'output': results.get('_synthesis', '')[:5000],
            'routing': decision.get('selected_models', {}),
            'state': state,
            'timestamp': time.time(),
        }, f, ensure_ascii=False)

    # Also append to repo log
    repo_log = os.path.expanduser('~/Desktop/collab-cloud/data/brain_activity.jsonl')
    os.makedirs(os.path.dirname(repo_log), exist_ok=True)
    with open(repo_log, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'routing': decision.get('selected_models', {}),
            'models': final_models,
            'difficulty': decision['difficulty'],
            'state': state,
            'timestamp': time.time(),
        }, ensure_ascii=False) + '\n')

    return results.get('_synthesis', ''), state


if __name__ == '__main__':
    prompt = sys.stdin.read().strip()
    if not prompt: sys.exit(1)
    answer, state = run(prompt)
    print(answer)
