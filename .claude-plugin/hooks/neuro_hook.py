#!/usr/bin/env python3
"""
neuro_hook.py — UserPromptSubmit Hook (v2.2 FAST)

FAST PATH: brainstem classification + routing done INLINE (<5ms),
           returns routing status immediately to Claude Code.
           "brainstem standby" → "Motor Cortex active" in <50ms.

SLOW PATH: LLM API calls launched as daemon background process,
           results written to ~/.synapseflow/brain/last_result.json,
           injected as additionalContext on next interaction.
"""
import sys, json, subprocess, os, traceback, io, time
from pathlib import Path

# Windows GBK compat
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ENGINE_DIR = Path(__file__).parent.parent.parent / "engine"
sys.path.insert(0, str(ENGINE_DIR.parent))

BRAIN_PATH = os.path.expanduser('~/.claude/tools/brain.py')
RESULT_FILE = os.path.expanduser('~/.synapseflow/brain/last_result.json')
TIMEOUT = 120

# ─── Fast inline imports (lazy, only if engine dir exists) ───

_brainstem = None
def _get_brainstem():
    global _brainstem
    if _brainstem is None:
        try:
            from engine.brainstem_wrapper import load as load_brainstem
            _brainstem = load_brainstem()
        except Exception:
            _brainstem = None
    return _brainstem

def _fast_classify(prompt: str) -> dict:
    """Inline brainstem classification — <5ms, no subprocess."""
    import re
    q = prompt.lower()

    # Lightweight feature extraction (matching brain.py score_task dims)
    features = {
        "code": len(re.findall(r'(代码|sql|函数|算法|编程|python|写一个|实现|ddl|cte|class |def |import |SELECT|CREATE)', q)),
        "math": len(re.findall(r'(数学|概率|统计|方程|几何|代数|微积分|矩阵|向量|定理|证明|求导|积分|极限)', q)),
        "logic": len(re.findall(r'(说谎|谁说|真话|假话|悖论|逻辑|推理|判断|证明|充要|必要|充分)', q)),
        "knowledge": len(re.findall(r'(什么|如何|为什么|定义|概念|原理|历史|背景|综述)', q)),
        "writing": len(re.findall(r'(写作|论文|文档|说明|报告|总结|翻译|润色|解释|描述|写一篇)', q)),
        "arch": len(re.findall(r'(架构|设计|系统|方案|权衡|选型|策略|规划|框架)', q)),
        "trap_single": len(re.findall(r'(图遍历|递归cte|索引优化|sql调优)', q)),
        "trap_need_collab": len(re.findall(r'(多视角|对比分析|跨领域|综合评估|深度剖析)', q)),
        "group_theory": 0, "graph_theory": 0, "topology": 0,
        "linear_algebra": 0, "calculus": 0, "probability": 0,
        "number_theory": 0, "diff_eq": 0, "combinatorics": 0, "optimization": 0,
        "chinese": len(re.findall(r'[一-鿿]', q)) / max(1, len(q)) * 2,
        "safety": 0, "general": 0.3, "db": len(re.findall(r'(数据库|sql|mysql|postgresql|mongodb|redis)', q)),
    }

    feature_keys = [
        "code", "math", "logic", "knowledge", "writing",
        "arch", "trap_single", "trap_need_collab",
        "group_theory", "graph_theory", "topology", "linear_algebra",
        "calculus", "probability", "number_theory", "diff_eq",
        "combinatorics", "optimization",
        "chinese", "safety", "general", "db",
    ]

    import numpy as np
    vec = np.array([float(features.get(k, 0)) for k in feature_keys], dtype=np.float64)

    bs = _get_brainstem()
    if bs is not None:
        region_id, confidence, difficulty = bs.classify(vec)
    else:
        region_id, confidence, difficulty = 0, 0.5, 0.3

    REGION_NAMES = [
        "motor_cortex", "parietal_cortex", "prefrontal_cortex",
        "temporal_cortex", "language_area", "visual_cortex",
    ]

    REGION_DISPLAY = {
        "motor_cortex":      "<> Motor Cortex (Code/SQL/Algorithm)",
        "parietal_cortex":   "π Parietal Cortex (Math/Numerical)",
        "prefrontal_cortex": "∟ Prefrontal Cortex (Logic/Reasoning)",
        "temporal_cortex":   "📚 Temporal Cortex (Knowledge/Memory)",
        "language_area":     "💬 Language Area (Writing/Chinese)",
        "visual_cortex":     "👁 Visual Cortex (Vision/Multimodal)",
    }

    region_name = REGION_NAMES[region_id] if 0 <= region_id < 6 else "temporal_cortex"
    display_name = REGION_DISPLAY.get(region_name, region_name)

    # Determine secondary regions
    secondary = []
    if difficulty > 0.5:
        if region_id != 2:
            secondary.append("∟ Prefrontal Cortex")
        if region_id != 3:
            secondary.append("📚 Temporal Cortex")

    return {
        "region": display_name,
        "region_id": region_id,
        "region_name": region_name,
        "confidence": round(float(confidence), 2),
        "difficulty": round(float(difficulty), 2),
        "secondary": secondary,
    }


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


def run_brain_background(prompt: str, routing: dict):
    """Launch brain.py in background for full LLM execution.
    Results written to RESULT_FILE, picked up next interaction."""
    try:
        env = os.environ.copy()
        env['NEURO_MODE'] = '1'
        env['BRAIN_BG'] = '1'  # Signal brain.py: write results to file, don't print
        proc = subprocess.Popen(
            [sys.executable, BRAIN_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace',
            env=env
        )
        # Don't wait — fire and forget
        # Results stored for next hook invocation
    except Exception:
        pass


def check_background_result() -> str:
    """Check if previous background brain.py finished. Return results or empty."""
    if not os.path.exists(RESULT_FILE):
        return ""
    try:
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        if age > 300:  # Stale (>5min)
            return ""
        return data.get("output", "")
    except Exception:
        return ""


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

    # Neuro toggle handlers
    neuro_flag = os.path.expanduser('~/.claude/tools/neuro_mode_off')

    if prompt.startswith('neuro: off') or prompt.startswith('neuro:off'):
        Path(neuro_flag).touch()
        print(json.dumps({'systemMessage': '[Neuro] OFF — standard mode',
            'suppressOutput': True,
            'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit',
                'additionalContext': '\n[Neuro Monitor] OFF.\n'}}, ensure_ascii=False))
        return
    if prompt.startswith('neuro: on') or prompt.startswith('neuro:on'):
        if os.path.exists(neuro_flag): os.remove(neuro_flag)
        print(json.dumps({'systemMessage': '[Neuro] ON — brain+parallel UI active',
            'suppressOutput': True,
            'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit',
                'additionalContext': '\n[Neuro Monitor] ON.\n'}}, ensure_ascii=False))
        return

    if prompt.startswith('neuro:'):
        prompt = prompt.replace('neuro:', '', 1).strip()

    # Skip trivial
    skip_patterns = ['hello', 'hi', 'thanks', '好的', '谢谢', 'ok', '嗯']
    if prompt.lower().strip() in skip_patterns or len(prompt) < 2:
        print(json.dumps({}))
        return

    # ─── FAST PATH: Classify + return routing immediately (<5ms) ───
    neuro_mode = not os.path.exists(neuro_flag)

    t0 = time.perf_counter()
    routing = _fast_classify(prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ─── Check for previous background results ───
    prev_result = check_background_result()

    # ─── Launch LLM calls in background ───
    if neuro_mode and prev_result == "":
        # Only launch if no pending results
        threading = __import__('threading')
        t = threading.Thread(target=run_brain_background, args=(prompt, routing), daemon=True)
        t.start()

    # ─── Build response ───
    region_status = (
        f"\n{'─'*60}\n"
        f"[Brainstem] {routing['region']}\n"
        f"   Confidence: {routing['confidence']:.0%} | Difficulty: {routing['difficulty']:.2f}\n"
        f"   Classify time: {elapsed_ms:.1f}ms\n"
    )
    if routing['secondary']:
        region_status += f"   Secondary: {', '.join(routing['secondary'])}\n"
    region_status += f"{'─'*60}\n"

    # Append previous LLM results if available
    full_ctx = region_status
    if prev_result:
        full_ctx += (
            f"\n[Previous Multi-Model Results]\n"
            f"{prev_result[:3000]}\n"
            f"{'─'*60}\n"
        )

    resp = {
        'systemMessage': f'[Brainstem] {routing["region"]} (conf={routing["confidence"]:.0%})',
        'suppressOutput': True,
        'hookSpecificOutput': {
            'hookEventName': 'UserPromptSubmit',
            'additionalContext': full_ctx
        }
    }

    print(json.dumps(resp, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Fallback: return empty, let Claude handle normally
        print(json.dumps({
            'systemMessage': f'[Brain] Hook error: {e}',
            'suppressOutput': True,
            'hookSpecificOutput': {
                'hookEventName': 'UserPromptSubmit',
                'additionalContext': f'[Brain hook failed: {e}]'
            }
        }, ensure_ascii=False))
