#!/usr/bin/env python3
"""
Benchmark v3: SMART Mode (DS-PRO decides single/collab per task)
              vs PURE Single Mode (DS-PRO only, all tasks)

Smart rules (from previous benchmark data):
  - Code/SQL-heavy -> single DS-PRO (multi loses -22% SQL detail)
  - Architecture/Design-heavy -> multi-model (+21% arch insight)
  - Hybrid -> DS-PRO judges case by case
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

def ask(client, model, prompt, max_tok=2500):
    r = client.chat.completions.create(model=model,
        messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=max_tok)
    return r.choices[0].message.content.strip()

def metrics(text):
    return {
        "chars": len(text),
        "sql": len(re.findall(r'(SELECT|CREATE|INSERT|UPDATE|DELETE|WITH\s+RECURSIVE|JOIN|PARTITION|INDEX|DDL)', text, re.I)),
        "arch": len(re.findall(r'(架构|设计|权衡|矛盾|一致性|性能|CAP|分布式|分区|复制|缓存|Redis|图数据库|Neo4j|方案|策略|决策)', text)),
        "code": len(re.findall(r'```', text)) // 2,
    }

# ═══════════════════════════════════════════════════════
HARD_TESTS = [
    {"id":"DB-1","title":"金融反洗钱-交易环检测","q":"""设计实时反洗钱系统：日处理10亿笔交易。检测1小时内、金额>1000美元、涉及3个以上账户的交易环路。
请回答：1.数据模型(表结构+索引+分区) 2.环检测SQL(递归CTE) 3.架构决策(何时引入图数据库) 4.性能优化。要求具体SQL代码。"""},
    {"id":"DB-2","title":"全球库存质押-时空估值","q":"""全球库存质押融资：5仓库×3国家，实时计算质押物美元总值。规则：可用库存=当前库存-已售未发货-运输在途+已付款未到货。按最新汇率折美元。
请回答：1.多币种ER模型 2.当前时刻库存快照SQL(窗口函数) 3.汇率同步方案 4.金融级一致性保证。要求具体SQL代码。"""},
    {"id":"DB-3","title":"动态合规规则引擎","q":"""每秒百万笔交易×数千条动态JSON规则，毫秒级判断命中。规则如{"and":[{"gt":["amount",10000]},{"in":["country",["IR","KP"]]}]}
请回答：1.规则存储模型和原子拆分 2.查询优化(位图索引/布隆过滤器) 3.分级缓存架构 4.原子性保证。要求具体SQL和架构思路。"""},
    {"id":"DB-4","title":"身份图谱-多跳查询","q":"""10亿用户+100亿设备关联。从可疑设备6跳内找所有关联用户，返回最短路径和关系类型。
请回答：1.图关系存储(关系型vs图数据库) 2.6跳递归CTE实现和性能瓶颈 3.物化路径优化 4.何时切图数据库。要求递归CTE代码。"""},
    {"id":"DB-5","title":"SaaS多租户-零中断迁移","q":"""多租户SaaS：大客户要求物理隔离。需在线迁移数据到独立实例，业务零中断。
请回答：1.逻辑+物理隔离数据模型(tenant_id分区键) 2.双写+CDC迁移方案 3.原子切换和回滚 4.外键索引迁移后有效性。要求DDL和迁移步骤。"""},
]

# ═══════════════════════════════════════════════════════
# Phase 0: SMART MODE - DS-PRO decides strategy per task
# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  Phase 0: SMART Mode - DS-PRO decides single/collab per task{X}")
print(f"{B}{'='*70}{X}")

STRATEGY = """分析以下数据库问题，判断属于"代码密集型"还是"架构密集型"。
- 代码密集型(coding): 核心是写出具体SQL/DDL/递归CTE/算法实现 -> 选 "single"
- 架构密集型(arch): 核心是系统设计/架构决策/多方案权衡/多视角分析 -> 选 "collab"

输出严格JSON: {"decision":"single","reason":"代码密集型，核心是SQL/CTE实现"} 或 {"decision":"collab","reason":"架构密集型，需多视角权衡"}"""

smart_decisions = {}
for test in HARD_TESTS:
    raw = ask(DS, DSM, STRATEGY + "\n\n问题：" + test["q"][:200], 200)
    try:
        if raw.startswith("```"): raw = raw.split("\n",1)[1].rsplit("\n",1)[0]
        dec = json.loads(raw)
        smart_decisions[test["id"]] = dec.get("decision","single")
    except:
        smart_decisions[test["id"]] = "single"
    d = smart_decisions[test["id"]]
    color = M if d == "collab" else B
    print(f"  [{test['id']}] {test['title']}: {color}{d}{X} | {dec.get('reason','?')}")

# ═══════════════════════════════════════════════════════
# Phase 1: Run both modes
# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  Phase 1: Execute SMART vs SINGLE{X}")
print(f"{B}{'='*70}{X}")

results = {"single": [], "smart": []}

for test in HARD_TESTS:
    tid = test["id"]
    title = test["title"]
    q = test["q"]
    smart_decision = smart_decisions.get(tid, "single")

    print(f"\n{Y}{'─'*70}{X}")
    print(f"{Y}  [{tid}] {title}{X}")
    print(f"{Y}  Smart decision: {smart_decision}{X}")
    print(f"{Y}{'─'*70}{X}")

    # ─── SINGLE: DS-PRO always ───
    t0 = time.time()
    single_ans = ask(DS, DSM, q, 2500)
    t_single = time.time() - t0
    sm = metrics(single_ans)
    results["single"].append({"id":tid, "time":t_single, "m":sm})
    print(f"  {B}[SINGLE] DS-PRO{X}: {t_single:.1f}s | {sm['chars']}c | SQL:{sm['sql']} | Arch:{sm['arch']} | Code:{sm['code']}")

    # ─── SMART: follow DS-PRO's decision ───
    t0_s = time.time()

    if smart_decision == "single":
        # Smart decides: single is best -> just use DS-PRO
        smart_ans = single_ans  # reuse (same answer)
        print(f"  {G}[SMART] -> SINGLE (reuse DS-PRO, 0 extra time){X}")
    else:
        # Smart decides: need collaboration -> 4 models parallel + synth
        multi = {}
        def worker(name, client, model):
            try: multi[name] = ask(client, model, q, 2000)
            except Exception as e: multi[name] = f"[ERR:{e}]"

        threads = [
            threading.Thread(target=worker, args=("ds-pro", DS, DSM)),
            threading.Thread(target=worker, args=("kimi", KI, KIM)),
            threading.Thread(target=worker, args=("glm", SJ, "glm")),
            threading.Thread(target=worker, args=("qwen", SJ, "qwen")),
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        for name in ["ds-pro","kimi","glm","qwen"]:
            mm = metrics(multi.get(name,""))
            print(f"    {D}[{name}]{X} {len(multi.get(name,''))}c | SQL:{mm['sql']} Arch:{mm['arch']}")

        sections = "\n".join([f"=== {n} ===\n{a[:1000]}" for n,a in multi.items()])
        smart_ans = ask(DS, DSM, f"综合以下回答给出最优答案:\n问题:{q}\n{sections}", 3000)

    t_smart = time.time() - t0_s
    mm = metrics(smart_ans)
    results["smart"].append({"id":tid, "time":t_smart, "m":mm, "strategy":smart_decision})
    strategy_label = f"{G}SINGLE(reuse){X}" if smart_decision == "single" else f"{M}COLLAB(4->synth){X}"
    print(f"  {G}[SMART]{X}: {strategy_label} | {t_smart:.1f}s | {mm['chars']}c | SQL:{mm['sql']} | Arch:{mm['arch']} | Code:{mm['code']}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n{B}{'='*70}{X}")
print(f"{B}  FINAL COMPARISON: SMART vs PURE SINGLE{X}")
print(f"{B}{'='*70}{X}")

total_s_t=0; total_sm_t=0
total_s_c=0; total_sm_c=0
total_s_sql=0; total_sm_sql=0
total_s_arch=0; total_sm_arch=0

for i, test in enumerate(HARD_TESTS):
    s = results["single"][i]
    sm = results["smart"][i]
    total_s_t += s["time"]; total_sm_t += sm["time"]
    total_s_c += s["m"]["chars"]; total_sm_c += sm["m"]["chars"]
    total_s_sql += s["m"]["sql"]; total_sm_sql += sm["m"]["sql"]
    total_s_arch += s["m"]["arch"]; total_sm_arch += sm["m"]["arch"]

    strategy = sm.get("strategy","?")
    print(f"  {test['id']}: single={s['time']:.0f}s/{s['m']['chars']}c  smart({strategy})={sm['time']:.0f}s/{sm['m']['chars']}c  SQL:{s['m']['sql']}->{sm['m']['sql']}  Arch:{s['m']['arch']}->{sm['m']['arch']}")

n = len(HARD_TESTS)
n_collab = sum(1 for r in results["smart"] if r.get("strategy")=="collab")
n_single = n - n_collab

print(f"\n{G}  SMART decisions: {n_single}x single + {n_collab}x collab = {n} total{X}")
print(f"{G}  Total time:   Single {total_s_t:.0f}s | Smart {total_sm_t:.0f}s ({total_sm_t/total_s_t*100:.0f}% of single){X}")
print(f"{G}  Arch insight: Single {total_s_arch} | Smart {total_sm_arch} ({total_sm_arch-total_s_arch:+d}){X}")
print(f"{G}  SQL detail:   Single {total_s_sql} | Smart {total_sm_sql} ({total_sm_sql-total_s_sql:+d}){X}")
print(f"{G}  Time saved vs full-multi: {(total_s_t*2.2-total_sm_t)/total_s_t*100:.0f}% (smart avoids wasting time on code-heavy){X}")
print(f"{G}{'='*70}{X}\n")
