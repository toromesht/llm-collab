#!/usr/bin/env python3
"""
UserPromptSubmit Hook — 多模型智能协作引擎
- 自动判断复杂度，简单走串行，复杂 4 模型并行
- 单模型故障不影响整体，降级运行
- 输出注入 Claude 上下文，过程透明
"""
import sys, json, subprocess, os, traceback, io
from pathlib import Path

# Windows GBK 兼容：强制 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BRAIN_PATH = os.path.expanduser('~/.claude/tools/brain.py')
TIMEOUT = 120  # 单次 brain.py 最长等待秒数

def log_error(msg):
    """将错误记录到诊断文件"""
    log_file = os.path.expanduser('~/.claude/tools/hook_errors.log')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[hook_brain] {msg}\n")

def extract_prompt(hook_input: dict) -> str:
    """从 hook stdin 中提取用户问题（兼容多种字段名）"""
    for key in ('prompt', 'text', 'user_prompt', 'message', 'input', 'content'):
        val = hook_input.get(key, '')
        if val and isinstance(val, str) and len(val.strip()) > 2:
            return val.strip()
    # 兜底：尝试嵌套路径
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

def run_brain(prompt: str, neuro_mode: bool = False) -> dict:
    """运行 brain.py，通过 stdin 传递问题（避免命令行 GBK 编码损坏），
       流式读取输出以实现实时反馈。"""
    try:
        env = os.environ.copy()
        if neuro_mode:
            env['NEURO_MODE'] = '1'
        proc = subprocess.Popen(
            [sys.executable, BRAIN_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace',
            env=env
        )
        # Pass prompt via stdin, not command line (avoids GBK corruption)
        stdout, stderr = proc.communicate(input=prompt, timeout=TIMEOUT)
        output = stdout.strip()
        if stderr:
            stderr_clean = '\n'.join(
                line for line in stderr.strip().split('\n')
                if 'gbk' not in line.lower() and 'UnicodeEncodeError' not in line
            )
            if stderr_clean:
                output += '\n\n[brain stderr]\n' + stderr_clean
        return {'ok': True, 'output': output, 'error': None}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {'ok': False, 'output': '', 'error': f'brain.py timeout ({TIMEOUT}s)'}
    except FileNotFoundError:
        return {'ok': False, 'output': '', 'error': f'brain.py not found: {BRAIN_PATH}'}
    except Exception as e:
        return {'ok': False, 'output': '', 'error': f'brain.py error: {e}'}


def main():
    try:
        raw_stdin = sys.stdin.read()
        if not raw_stdin.strip():
            # 空输入，放行
            print(json.dumps({}))
            return

        hook_input = json.loads(raw_stdin)
    except json.JSONDecodeError as e:
        log_error(f"stdin JSON 解析失败: {e}")
        print(json.dumps({}))
        return

    prompt = extract_prompt(hook_input)

    if not prompt:
        print(json.dumps({}))
        return

    # Neuro toggle: neuro: off to disable, otherwise always ON
    neuro_flag = os.path.expanduser('~/.claude/tools/neuro_mode_off')
    neuro_mode = not os.path.exists(neuro_flag)  # default ON

    if prompt.startswith('neuro: off') or prompt.startswith('neuro:off'):
        Path(neuro_flag).touch()
        resp = {'systemMessage': '[Neuro Monitor] OFF — standard mode',
                'suppressOutput': True,
                'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit',
                    'additionalContext': '\n[Neuro Monitor] OFF. Type `neuro: on` to re-enable.\n'}}
        print(json.dumps(resp, ensure_ascii=False))
        return
    if prompt.startswith('neuro: on') or prompt.startswith('neuro:on'):
        if os.path.exists(neuro_flag): os.remove(neuro_flag)
        resp = {'systemMessage': '[Neuro Monitor] ON — brain+parallel UI active',
                'suppressOutput': True,
                'hookSpecificOutput': {'hookEventName': 'UserPromptSubmit',
                    'additionalContext': '\n[Neuro Monitor] ON.\n'}}
        print(json.dumps(resp, ensure_ascii=False))
        return

    # neuro: prefix on a question = one-time force
    if prompt.startswith('neuro:'):
        prompt = prompt.replace('neuro:', '', 1).strip()
        neuro_mode = True

    # Skip trivial messages
    skip_patterns = ['hello', 'hi', 'thanks', '好的', '谢谢', 'ok', '嗯']
    if prompt.lower().strip() in skip_patterns or len(prompt) < 2:
        print(json.dumps({}))
        return

    # Execute brain.py (with NEURO_MODE env var if neuro: prefix used)
    result = run_brain(prompt, neuro_mode=neuro_mode)

    if result['ok']:
        ctx = (
            "\n============================================================\n"
            "[Brain v2] Multi-Model Collaboration Results\n"
            "   Route: Smart complexity analysis -> Simple(serial) / Complex(4-way parallel)\n"
            "   Compute: [DS-PRO] [KIMI] [GLM] [QWEN]\n"
            "============================================================\n\n"
            + result['output'] +
            "\n\n============================================================\n"
            "[Instruction] Synthesize the above multi-model results.\n"
            "   If models disagree, analyze and pick the best answer.\n"
            "   Note each model's core contribution.\n"
            "============================================================\n"
        )
        resp = {
            'systemMessage': '[Brain] Multi-model collaboration complete',
            'suppressOutput': True,
            'hookSpecificOutput': {
                'hookEventName': 'UserPromptSubmit',
                'additionalContext': ctx
            }
        }
    else:
        # 模型故障 → 反馈给 Claude，让我直接用单模型回答
        ctx = (
            "\n[WARNING] Multi-model engine unavailable\n"
            f"   Reason: {result['error']}\n"
            "   Fallback: single-model mode (DS-PRO only)\n"
        )
        resp = {
            'systemMessage': f'[Brain] Degraded: {result["error"]}',
            'suppressOutput': True,
            'hookSpecificOutput': {
                'hookEventName': 'UserPromptSubmit',
                'additionalContext': ctx
            }
        }

    print(json.dumps(resp, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log_error(f"未捕获异常: {e}\n{traceback.format_exc()}")
        # 降级：返回空 JSON，让 Claude 正常处理
        print(json.dumps({
            'systemMessage': f'[Brain] Hook error: {e}',
            'suppressOutput': True,
            'hookSpecificOutput': {
                'hookEventName': 'UserPromptSubmit',
                'additionalContext': f'[Brain hook failed: {e}]'
            }
        }, ensure_ascii=False))
