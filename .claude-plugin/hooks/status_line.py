#!/usr/bin/env python3
"""SynapseFlow status line — Claude Code polls this every ~1s"""
import os, json, sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')

SF = os.path.expanduser('~/.claude/tools/neuro_status.json')
if os.path.exists(os.path.expanduser('~/.claude/tools/neuro_mode_off')):
    print('NEURO OFF')
    sys.exit(0)

d = {}
if os.path.exists(SF):
    try: d = json.loads(open(SF,'r',encoding='utf-8').read())
    except: d = {}

models = d.get('models', {})
active = d.get('active_regions', [])
cls = d.get('classification', {})
diff = d.get('difficulty', 0)
region_map = d.get('region', {})
done = d.get('done', True)

running = any(st == 'running' for st in models.values())
has_done = any(st == 'done' for st in models.values())

icons = {0: '<>', 1: 'π', 2: '∟', 3: '📚', 4: '💬', 5: '👁'}
names = {0: 'Motor', 1: 'Parietal', 2: 'PFC', 3: 'Temporal', 4: 'Language', 5: 'Visual'}
COLORS = {0: '\033[36m', 1: '\033[35m', 2: '\033[32m', 3: '\033[33m', 4: '\033[31m', 5: '\033[34m'}
R = '\033[0m'
SHORT = {'ds-pro':'ds-p', 'ds-think':'ds-tk', 'glm':'glm', 'qwen':'qw', 'kimi':'ki', 'groq':'grq'}

def _color_region(rid, text):
    c = COLORS.get(rid, '')
    return f'{c}{text}{R}' if c else text

# ── State 1: Models actively running or done ──
# Color dots per model
MDOT = {'done': '\033[32m●\033[0m', 'running': '\033[36m◐\033[0m', 'failed': '\033[31m✕\033[0m', 'pending': '\033[2m○\033[0m'}
if running or has_done:
    parts = []
    for m, st in models.items():
        dot = MDOT.get(st, '○')
        ms = SHORT.get(m, m[:4])
        parts.append(f'{dot}{ms}')
    print(' '.join(parts))
    sys.exit(0)

# ── State 2: Brainstem classified → show regions + planned models ──
if active:
    region_parts = []
    for rid in active:
        icon = icons.get(rid, '?')
        name = names.get(rid, '?')
        # New format: region -> [models]
        rmodels = [SHORT.get(m, m[:4]) for m in region_map.get(name, [])]
        if rmodels:
            region_parts.append(f'{icon} {name}({",".join(rmodels)})')
        else:
            region_parts.append(f'{icon} {name}')
    line = ' | '.join(region_parts)
    if diff: line += f'  d={diff:.2f}'
    print(line)
    sys.exit(0)

# ── State 3: Classified, single region ──
if cls:
    rid = cls.get('region_id', 0)
    icon = icons.get(rid, '?')
    name = names.get(rid, '?')
    conf = cls.get('confidence', 0)
    nshort = name[:4].lower()
    rmodels = [SHORT.get(m, m[:4]) for m in region_map.get(name, [])]
    if rmodels:
        print(f'{icon} {name}({",".join(rmodels)})  conf={conf:.0%}')
    else:
        print(f'{icon} {name}  conf={conf:.0%} d={diff:.2f}')
    sys.exit(0)

# ── State 4: True standby ──
print('○ brainstem standby')
