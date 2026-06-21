#!/usr/bin/env python3
"""
Benchmark v2: Single DS-PRO vs Multi-Model on 5 HARD database/modeling problems
Metrics: Time, Depth (chars), Architecture Coverage, Concrete SQL/Code output
"""
import sys, json, os, threading, time, re
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

B="\033[1;34m"; M="\033[1;35m"; C="\033[1;36m"; Y="\033[1;33m"
G="\033[1;32m"; R="\033[1;31m"; D="\033[2m"; X="\033[0m"

CFG = json.loads(Path(os.path.expanduser('~/.claude/tools/llm-config.json')).read_text(encoding='utf-8'))
DS=OpenAI(api_key=CFG['deepseek_pro']['api_key'],base_url=CFG['deepseek_pro']['base_url']); DSM=CFG['deepseek_pro']['model']
KI=OpenAI(api_key=CFG['kimi']['api_key'],base_url=CFG['kimi']['base_url']); KIM=CFG['kimi']['model']
SJ=OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'],base_url=CFG['sjtu_zhiyuan']['base_url'])

def ask(client, model, prompt, max_tok=2000):
    r = client.chat.completions.create(model=model,
        messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=max_tok)
    return r.choices[0].message.content.strip()

def count_metrics(text):
    """Count quality indicators"""
    sql_count = len(re.findall(r'(SELECT|CREATE|INSERT|UPDATE|DELETE|WITH\s+RECURSIVE|JOIN|PARTITION|INDEX)', text, re.I))
    arch_count = len(re.findall(r'(架构|设计|权衡|矛盾|一致性|性能|CAP|分布式|分区|复制|缓存|Redis|图数据库|Neo4j)', text))
    code_blocks = len(re.findall(r'```', text)) // 2
    return {"chars": len(text), "sql": sql_count, "arch": arch_count, "code": code_blocks}

# ═══════════════════════════════════════════════════════
HARD_TESTS = [
    {
        "id": "DB-1",
        "title": "金融反洗钱-交易环检测",
        "question": """设计一个实时反洗钱系统：日处理10亿笔交易。需要检测1小时内、金额>1000美元、涉及3个以上账户的交易环路。

请回答：
1. 数据模型设计（表结构+索引+分区策略）
2. 环检测的SQL（使用递归CTE）
3. 架构决策：何时引入图数据库？混合架构如何设计？
4. 性能优化策略

要求具体、可落地，输出SQL代码。""",
    },
    {
        "id": "DB-2",
        "title": "全球库存质押-时空估值",
        "question": """设计全球库存质押融资系统：5个仓库分布在3个国家，需实时计算质押物美元总值。

规则：可用库存 = 当前库存 - 已售未发货 - 运输在途 + 已付款未到货。所有按最新汇率折算美元。

请回答：
1. 多币种多仓库的ER模型
2. 获取"当前时刻"库存快照的高效SQL（窗口函数）
3. 汇率实时同步方案
4. 如何保证金融级数据一致性

要求具体SQL代码。""",
    },
    {
        "id": "DB-3",
        "title": "动态合规规则引擎",
        "question": """每秒百万笔交易×数千条动态JSON规则。需要在毫秒级判断交易是否命中规则。

规则示例：{"and":[{"gt":["amount",10000]},{"in":["country",["IR","KP"]]}]}

请回答：
1. 规则存储模型和原子条件拆分
2. 查询优化策略（位图索引/布隆过滤器）
3. 分级缓存架构
4. 如何避免SQL注入和保证规则原子性

要求具体SQL和架构图思路。""",
    },
    {
        "id": "DB-4",
        "title": "身份图谱-多跳查询",
        "question": """10亿用户+100亿设备关联。风控需从可疑设备出发，6跳内找到所有关联用户，返回最短路径和关系类型。

请回答：
1. 图关系存储模型（关系型 vs 图数据库）
2. 6跳递归CTE的实现和性能瓶颈
3. 物化路径优化方案
4. 何时必须切到图数据库？迁移策略？

要求具体SQL（递归CTE）代码。""",
    },
    {
        "id": "DB-5",
        "title": "SaaS多租户-零中断迁移",
        "question": """多租户SaaS：一个大客户要求物理隔离。需在线迁移其数据到独立实例，业务零中断。

请回答：
1. 支持逻辑隔离+物理隔离的数据模型（tenant_id作为分区键）
2. 双写+CDC的迁移方案设计
3. 原子切换和回滚机制
4. 外键和索引在迁移后如何保持有效

要求具体DDL和迁移步骤。""",
    },
]

# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  BENCHMARK v2: 5 HARD Database/Modeling Problems{X}")
print(f"{B}  Single DS-PRO vs Multi-Model (DS-PRO+Kimi+GLM+Qwen){X}")
print(f"{B}{'='*70}{X}")

results = {"single": [], "multi": []}

for test in HARD_TESTS:
    tid = test["id"]
    title = test["title"]
    q = test["question"]

    print(f"\n{Y}{'─'*70}{X}")
    print(f"{Y}  [{tid}] {title}{X}")
    print(f"{Y}{'─'*70}{X}")

    # ─── SINGLE MODE ───
    print(f"  {D}[SINGLE] DS-PRO working...{X}", end="", flush=True)
    t0 = time.time()
    try:
        single_ans = ask(DS, DSM, q, 2500)
    except Exception as e:
        single_ans = f"[ERR: {e}]"
    t_single = time.time() - t0
    sm = count_metrics(single_ans)
    results["single"].append({"id": tid, "time": t_single, "metrics": sm})
    print(f"\r  {B}[SINGLE] DS-PRO{X}: {t_single:.1f}s | {sm['chars']}c | SQL:{sm['sql']} | Arch:{sm['arch']} | Code:{sm['code']}")

    # ─── MULTI MODE ───
    multi_ans = {}
    def worker(name, client, model):
        try:
            multi_ans[name] = ask(client, model, q, 1500)
        except Exception as e:
            multi_ans[name] = f"[ERR: {e}]"

    t0_m = time.time()
    threads = [
        threading.Thread(target=worker, args=("ds-pro", DS, DSM)),
        threading.Thread(target=worker, args=("kimi", KI, KIM)),
        threading.Thread(target=worker, args=("glm", SJ, "glm")),
        threading.Thread(target=worker, args=("qwen", SJ, "qwen")),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    # Show per-model stats
    for name in ["ds-pro","kimi","glm","qwen"]:
        a = multi_ans.get(name, "")
        mm = count_metrics(a)
        print(f"    {D}[{name}]{X} {len(a)}c | SQL:{mm['sql']} Arch:{mm['arch']}")

    # Synthesis
    sections = "\n".join([f"=== {n} ===\n{a[:1000]}" for n, a in multi_ans.items()])
    synth_prompt = f"""综合以下4个AI对同一问题的独立回答，取各模型之长，输出最优最终答案。
要求：结构化、具体可落地、包含SQL代码、架构决策清晰。

原问题：{q}

各模型回答：
{sections}

综合答案："""
    try:
        synth_ans = ask(DS, DSM, synth_prompt, 3000)
    except Exception as e:
        synth_ans = f"[ERR: {e}]"

    t_multi = time.time() - t0_m
    mm = count_metrics(synth_ans)
    results["multi"].append({"id": tid, "time": t_multi, "metrics": mm})
    print(f"  {M}[MULTI] Synth{X}: {t_multi:.1f}s | {mm['chars']}c | SQL:{mm['sql']} | Arch:{mm['arch']} | Code:{mm['code']}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  BENCHMARK RESULTS: 5 HARD Problems{X}")
print(f"{B}{'='*70}{X}")

print(f"\n{'Problem':<8} {'Single Time':<13} {'Multi Time':<13} {'S Chars':<10} {'M Chars':<10} {'S SQL':<7} {'M SQL':<7} {'S Arch':<7} {'M Arch':<7}")
print(f"{'-'*85}")

total_s_time = 0; total_m_time = 0
total_s_chars = 0; total_m_chars = 0
total_s_sql = 0; total_m_sql = 0
total_s_arch = 0; total_m_arch = 0

for i, test in enumerate(HARD_TESTS):
    s = results["single"][i]
    m = results["multi"][i]
    total_s_time += s["time"]; total_m_time += m["time"]
    total_s_chars += s["metrics"]["chars"]; total_m_chars += m["metrics"]["chars"]
    total_s_sql += s["metrics"]["sql"]; total_m_sql += m["metrics"]["sql"]
    total_s_arch += s["metrics"]["arch"]; total_m_arch += m["metrics"]["arch"]

    sql_diff = f"{G}+{m['metrics']['sql']-s['metrics']['sql']}{X}" if m['metrics']['sql']>s['metrics']['sql'] else (f"{R}{m['metrics']['sql']-s['metrics']['sql']}{X}" if m['metrics']['sql']<s['metrics']['sql'] else "=")
    arch_diff = f"{G}+{m['metrics']['arch']-s['metrics']['arch']}{X}" if m['metrics']['arch']>s['metrics']['arch'] else (f"{R}{m['metrics']['arch']-s['metrics']['arch']}{X}" if m['metrics']['arch']<s['metrics']['arch'] else "=")

    print(f"{test['id']:<8} {s['time']:.1f}s{'':>7} {m['time']:.1f}s{'':>7} {s['metrics']['chars']:<10} {m['metrics']['chars']:<10} {s['metrics']['sql']:<7} {m['metrics']['sql']} {sql_diff:<6} {s['metrics']['arch']:<7} {m['metrics']['arch']} {arch_diff:<5}")

n = len(HARD_TESTS)
print(f"\n{G}{'='*70}{X}")
print(f"{G}  AGGREGATE (N={n}){X}")
print(f"{G}  Time:       Single {total_s_time:.1f}s | Multi {total_m_time:.1f}s ({total_m_time/total_s_time:.1f}x){X}")
print(f"{G}  Chars:      Single {total_s_chars} | Multi {total_m_chars} (+{(total_m_chars-total_s_chars)/total_s_chars*100:.0f}%){X}")
print(f"{G}  SQL count:  Single {total_s_sql} | Multi {total_m_sql} (+{total_m_sql-total_s_sql}){X}")
print(f"{G}  Arch terms: Single {total_s_arch} | Multi {total_m_arch} (+{total_m_arch-total_s_arch}){X}")
print(f"{G}  Quality Gain: +{(total_m_chars-total_s_chars)/total_s_chars*100:.0f}% depth, +{total_m_sql-total_s_sql} SQL refs, +{total_m_arch-total_s_arch} arch insights{X}")
print(f"{G}{'='*70}{X}\n")
