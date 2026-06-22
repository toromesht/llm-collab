#!/usr/bin/env python3
"""Diagnostic: capture UserPromptSubmit stdin to a file"""
import sys, json, os
data = sys.stdin.read()
diag_file = os.path.expanduser('~/.claude/tools/hook_stdin_diag.json')
with open(diag_file, 'w', encoding='utf-8') as f:
    json.dump({"raw": data, "parsed": json.loads(data) if data else None}, f, ensure_ascii=False, indent=2)
print(json.dumps({"systemMessage": "Diag captured"}))
