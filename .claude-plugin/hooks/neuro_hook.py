#!/usr/bin/env python3
"""
hook_brain.py — Claude Code UserPromptSubmit Hook (v3 FAST)

FAST PATH (<5ms): inline brainstem classify → write neuro_status.json immediately.
                  Status line updates to show active brain regions INSTANTLY.

SLOW PATH (bg):  brain.py launched as background subprocess for LLM API calls.
                  Results written to neuro_status.json when complete.

Status line reads neuro_status.json every ~1s → real-time brain region display.
"""
import sys, json, subprocess, os, traceback, io, time, re, threading
from pathlib import Path

# Windows GBK compat
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BRAIN_PATH = os.path.expanduser('~/.claude/tools/brain.py')
STATUS_FILE = os.path.expanduser('~/.claude/tools/neuro_status.json')
TIMEOUT = 120

# ═══════════════════════════════════════════════════════════
# FAST INLINE BRAINSTEM (<5ms, no subprocess)
# ═══════════════════════════════════════════════════════════

_brainstem = None

def _get_brainstem():
    global _brainstem
    if _brainstem is None:
        try:
            sys.path.insert(0, str(Path.home() / 'Desktop/collab-cloud'))
            from engine.brainstem_wrapper import load as load_brainstem
            _brainstem = load_brainstem()
        except Exception:
            _brainstem = None
    return _brainstem

REGION_NAMES = [
    "Motor", "Parietal", "PFC",
    "Temporal", "Language", "Visual",
]

def fast_classify(prompt: str) -> dict:
    """Inline brainstem classification — <5ms."""
    import numpy as np
    q = prompt.lower()

    features = {
        "code": len(re.findall(r'(代码|sql|函数|算法|编程|python|写一个|实现|ddl|cte|class |def |import |SELECT|CREATE)', q)),
        "math": len(re.findall(r'(数学|概率|统计|方程|几何|代数|微积分|矩阵|向量|定理|证明|求导|积分|极限)', q)),
        "logic": len(re.findall(r'(说谎|谁说|真话|假话|悖论|逻辑|推理|判断|证明|充要|必要|充分)', q)),
        "knowledge": len(re.findall(r'(什么|如何|为什么|定义|概念|原理|历史|背景|综述)', q)),
        "writing": len(re.findall(r'(写作|论文|文档|说明|报告|总结|翻译|润色|解释|描述|写一篇)', q)),
        "arch": len(re.findall(r'(架构|设计|系统|方案|权衡|选型|策略|规划|框架)', q)),
        "trap_single": 0, "trap_need_collab": 0,
        "group_theory": 0, "graph_theory": 0, "topology": 0,
        "linear_algebra": 0, "calculus": 0, "probability": 0,
        "number_theory": 0, "diff_eq": 0, "combinatorics": 0, "optimization": 0,
        "chinese": len(re.findall(r'[一-鿿]', q)) / max(1, len(q)) * 2,
        "safety": 0, "general": 0.3, "db": 0,
    }
    keys = ["code","math","logic","knowledge","writing","arch","trap_single","trap_need_collab",
            "group_theory","graph_theory","topology","linear_algebra","calculus","probability",
            "number_theory","diff_eq","combinatorics","optimization",
            "chinese","safety","general","db"]
    vec = np.array([float(features.get(k, 0)) for k in keys], dtype=np.float64)

    bs = _get_brainstem()
    if bs is not None:
        region_id, confidence, difficulty = bs.classify(vec)
    else:
        region_id, confidence, difficulty = 0, 0.5, 0.3

    region_name = REGION_NAMES[region_id] if 0 <= region_id < 6 else "Temporal"

    # Determine which regions would be activated
    active_regions = [region_id]
    if difficulty > 0.5:
        if 2 not in active_regions:
            active_regions.append(2)  # PFC for hard problems
        if 3 not in active_regions:
            active_regions.append(3)  # Temporal for knowledge

    # Determine models per region (2-3 models each)
    model_region_map = {}
    if region_id == 0:  # Motor: code/execution
        model_region_map = {"ds-pro": "Motor", "ds-think": "Motor", "groq": "Motor"}
    elif region_id == 1:  # Parietal: math/numerical
        model_region_map = {"ds-pro": "Parietal", "ds-think": "Parietal", "qwen": "Parietal"}
    elif region_id == 2:  # PFC: logic/reasoning
        model_region_map = {"ds-pro": "PFC", "ds-think": "PFC", "glm": "PFC"}
    elif region_id == 3:  # Temporal: knowledge/memory
        model_region_map = {"glm": "Temporal", "qwen": "Temporal", "groq": "Temporal"}
    elif region_id == 4:  # Language: writing/chinese
        model_region_map = {"glm": "Language", "kimi": "Language", "qwen": "Language"}
    else:  # Visual: multimodal
        model_region_map = {"kimi": "Visual", "qwen": "Visual"}

    # Low difficulty = use Groq only (fast/cheap)
    if difficulty < 0.3:
        model_region_map = {"groq": _REGION_NAMES[region_id]}
    # High difficulty = add ds-think for deep reasoning
    if difficulty > 0.6 and "ds-think" not in model_region_map:
        model_region_map["ds-think"] = _REGION_NAMES[region_id]

    return {
        "region_id": region_id,
        "region_name": region_name,
        "confidence": round(float(confidence), 2),
        "difficulty": round(float(difficulty), 2),
        "active_regions": active_regions,
        "models": {},  # empty -> status_line shows region view
        "region_map": model_region_map,  # used by bg runner
        "pending_models": list(model_region_map.keys()),
    }


def write_status(routing: dict, done: bool = False, pathway: str = ""):
    """Write neuro_status.json for status_line.py to read."""
    data = {
        "active": len(routing.get("active_regions", [])),
        "active_regions": routing.get("active_regions", []),
        "models": routing.get("models", {}),
        "region": routing.get("region_map", {}),
        "difficulty": routing.get("difficulty", 0),
        "classification": {
            "region_id": routing.get("region_id", 0),
            "region_name": routing.get("region_name", ""),
            "confidence": routing.get("confidence", 0),
        },
        "pathway": pathway,
        "done": done,
        "timestamp": time.time(),
    }
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# BRAIN.PY BACKGROUND EXECUTION
# ═══════════════════════════════════════════════════════════

def run_brain_background(prompt: str, routing: dict):
    """Run brain.py in background with status file updates."""
    try:
        env = os.environ.copy()
        env['NEURO_MODE'] = '1'

        proc = subprocess.Popen(
            [sys.executable, BRAIN_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace',
            env=env
        )

        # Update status: models running (use pending_models, not models which is empty)
        pending = routing.get("pending_models", [])
        region_map = routing.get("region_map", {})
        models_running = {m: "running" for m in pending}
        routing["models"] = models_running
        routing["region"] = region_map
        write_status(routing, done=False)

        # Wait for brain.py to finish
        stdout, stderr = proc.communicate(input=prompt, timeout=TIMEOUT)

        # Update status: models done
        models_done = {m: "done" for m in routing.get("models", {})}
        routing["models"] = models_done

        pathway = extract_pathway(stdout)
        write_status(routing, done=True, pathway=pathway)

        # Write result file for next hook invocation
        result_file = os.path.expanduser('~/.synapseflow/brain/last_result.json')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "output": stdout[:5000] if stdout else "",
                "timestamp": time.time(),
            }, f, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        proc.kill()
        models_failed = {m: "failed" for m in routing.get("models", {})}
        routing["models"] = models_failed
        write_status(routing, done=True)
    except Exception:
        write_status(routing, done=True)


def extract_pathway(output: str) -> str:
    """Extract pathway from brain.py output."""
    for line in (output or "").split('\n'):
        if 'Brainstem' in line and '→' in line:
            return line.strip()
        if 'Region=' in line:
            return line.strip()
    return ''


# ═══════════════════════════════════════════════════════════
# HOOK MAIN
# ═══════════════════════════════════════════════════════════

def extract_prompt(hook_input: dict) -> str:
    for key in ('prompt', 'text', 'user_prompt', 'message', 'input', 'content'):
        val = hook_input.get(key, '')
        if val and isinstance(val, str) and len(val.strip()) > 2:
            return val.strip()
    for path in [['user_input', 'text'], ['message', 'content'], ['data', 'prompt']]:
        d = hook_input
        try:
            for k in path:
                d = d.get(k, {})
            if isinstance(d, str) and len(d.strip()) > 2:
                return d.strip()
        except Exception:
            continue
    return ''


def main():
    try:
        raw_stdin = sys.stdin.read()
        if not raw_stdin.strip():
            print(json.dumps({}))
            return
        hook_input = json.loads(raw_stdin)
    except json.JSONDecodeError:
        print(json.dumps({}))
        return

    prompt = extract_prompt(hook_input)
    if not prompt:
        print(json.dumps({}))
        return

    # Neuro toggle
    neuro_flag = os.path.expanduser('~/.claude/tools/neuro_mode_off')

    if prompt.startswith('neuro: off') or prompt.startswith('neuro:off'):
        Path(neuro_flag).touch()
        # Clear status
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)
        print(json.dumps({'systemMessage': '[Neuro] OFF', 'suppressOutput': False}))
        return

    if prompt.startswith('neuro: on') or prompt.startswith('neuro:on'):
        if os.path.exists(neuro_flag): os.remove(neuro_flag)
        print(json.dumps({'systemMessage': '[Neuro] ON', 'suppressOutput': False}))
        return

    if prompt.startswith('neuro:'):
        prompt = prompt.replace('neuro:', '', 1).strip()

    # Skip trivial
    skip = ['hello', 'hi', 'thanks', '好的', '谢谢', 'ok', '嗯']
    if prompt.lower().strip() in skip or len(prompt) < 2:
        print(json.dumps({}))
        return

    neuro_mode = not os.path.exists(neuro_flag)

    # ═══ FAST PATH: classify + write status immediately ═══
    if neuro_mode:
        routing = fast_classify(prompt)
        write_status(routing, done=False)

        # Launch brain.py in background for full LLM execution
        t = threading.Thread(target=run_brain_background, args=(prompt, dict(routing)), daemon=True)
        t.start()

        # Return immediately — status line already updated!
        region_icons = {0: "<>", 1: "π", 2: "∟", 3: "📚", 4: "💬", 5: "👁"}
        active_str = ", ".join(
            f'{region_icons.get(rid, "?")} {REGION_NAMES[rid]}'
            for rid in routing["active_regions"]
        )

        resp = {
            'systemMessage': f'[Brainstem] {active_str} (d={routing["difficulty"]:.2f})',
            'suppressOutput': False,
            'hookSpecificOutput': {
                'hookEventName': 'UserPromptSubmit',
                'additionalContext': (
                    f'\n{"─"*60}\n'
                    f'[Brainstem] Active: {active_str}\n'
                    f'   Confidence: {routing["confidence"]:.0%} | Difficulty: {routing["difficulty"]:.2f}\n'
                    f'   Models: {", ".join(routing["models"].keys())}\n'
                    f'{"─"*60}\n'
                )
            }
        }
        print(json.dumps(resp, ensure_ascii=False))
        return

    # Non-neuro mode: pass through
    print(json.dumps({}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({}))
