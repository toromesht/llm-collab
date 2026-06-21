#!/usr/bin/env python3
"""
Evaluator v3 — HARD Dataset: 50 advanced questions
  Math: calculus/linear algebra/probability
  Code: algorithms/data structures/optimization
  Logic: multi-step deduction/paradoxes
  Knowledge: academic-level specialized
  中文深度: classical Chinese/specialized terminology
"""
import sys,json,os,threading,time,re,random
from pathlib import Path
from openai import OpenAI
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

CFG = json.loads(Path(os.path.expanduser('~/.claude/tools/llm-config.json')).read_text(encoding='utf-8'))
CLIENTS = {
    "GLM":  (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "glm"),
    "DS-Think": (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "deepseek-reasoner"),
    "DS-PRO": (OpenAI(api_key=CFG['deepseek_pro']['api_key'], base_url=CFG['deepseek_pro']['base_url']), CFG['deepseek_pro']['model']),
    "QWEN": (OpenAI(api_key=CFG['sjtu_zhiyuan']['api_key'], base_url=CFG['sjtu_zhiyuan']['base_url']), "qwen"),
}

HARD = {
    "advanced_math": [
        ("AM1", "求极限 lim(x→0) (sin(x)-x)/x³。只输出数值（分数或小数）。", "-1/6"),
        ("AM2", "矩阵 A=[[1,2],[3,4]]，求 A 的特征值之和。只输出数字。", "5"),
        ("AM3", "∫₀¹ x·eˣ dx = ? 只输出数值(保留2位小数)。", "1.00"),
        ("AM4", "抛骰子直到出现6，期望掷多少次？只输出数字。", "6"),
        ("AM5", "函数 f(x)=x³-3x 的拐点是？只输出 x= 的值。", "0"),
        ("AM6", "从52张牌中抽5张，恰好3张A的概率？用分数表示。", "94/54145"),
        ("AM7", "解微分方程 dy/dx = y, 且 y(0)=1。求 y(1)。只输出数字(2位小数)。", "2.72"),
        ("AM8", "复数 (1+i)⁸ 的模等于？只输出数字。", "16"),
        ("AM9", "二次型 x²+4xy+y² 的矩阵是？只输出矩阵 [[a,b],[c,d]]。", "[[1,2],[2,1]]"),
        ("AM10","n→∞ 时 (1+2/n)^n 的极限？只输出表达式。", "e²"),
    ],
    "algorithm": [
        ("AL1","Dijkstra算法的时间复杂度用二叉堆实现是？只输出 O(...)。","O((V+E)logV"),
        ("AL2","用Python实现归并排序 merge_sort(arr)。只输出函数代码。","def merge_sort"),
        ("AL3","动态规划解0-1背包问题的状态转移方程是什么？只输出递推式。","dp[i][w] = max"),
        ("AL4","红黑树插入一个节点最多需要多少次旋转？只输出数字。","3"),
        ("AL5","用Python写LRU缓存类 LRUCache(capacity)。只输出代码。","class LRUCache"),
        ("AL6","N个节点的完全二叉树有多少个叶子节点？用N表示。","N+1)/2"),
        ("AL7","快速排序最坏时间复杂度是多少？只输出 O(...)。","O(n²)"),
        ("AL8","用Python实现并查集(Union-Find)，含路径压缩。只输出代码。","class UnionFind"),
        ("AL9","AVL树和红黑树的主要区别是什么？一句话（20字内）。","AVL更严格平衡"),
        ("AL10","堆排序是稳定的吗？只回答是或否。","否"),
    ],
    "deep_logic": [
        ("DL1","蒙提霍尔问题：三扇门，你选了一扇，主持人打开另一扇空门，你应该换吗？换后中奖概率是多少？只输出分数。","2/3"),
        ("DL2","岛上有100个蓝眼人和1个棕眼人，每个人看到别人眼睛颜色。外来者说'至少一个蓝眼'。第100天会发生什么？只回答核心事件。","所有蓝眼人离开"),
        ("DL3","两根绳子，每根燃烧完需1小时，但燃烧速度不均匀。如何测量45分钟？只描述方法（30字内）。","一根两头点一根一头点，第一根烧完时点第二根另一头"),
        ("DL4","12个外观相同的球，1个重量不同（不知轻重），天平最多称几次能找到？只输出数字。","3"),
        ("DL5","飞机座位悖论：100乘客，每人有指定座位。第1人随机坐。最后1人坐到自己座位的概率？只输出分数。","1/2"),
        ("DL6","命题 P→Q 的逆否命题是？只输出逻辑表达式。","¬Q→¬P"),
        ("DL7","说谎者悖论：'这句话是假的'。这个命题的真值是什么？只回答核心矛盾。","无法判定"),
        ("DL8","囚徒困境中，纳什均衡策略是什么？只输出策略名。","双方都背叛"),
    ],
    "chinese_depth": [
        ("CN1","'道可道，非常道'出自哪部经典？只输出书名。","道德经"),
        ("CN2","请用文言文翻译'天道酬勤'(20字内)。","天行健君子以自强不息"),
        ("CN3","'形而上者谓之道，形而下者谓之器'出自何处？只输出书名。","周易"),
        ("CN4","中国现存最早的天文台叫什么？只输出名称。","观星台"),
        ("CN5","二十四史中第一部纪传体断代史是？只输出书名。","汉书"),
        ("CN6","《孙子兵法》共有多少篇？只输出数字。","13"),
        ("CN7","'为天地立心，为生民立命'是谁说的？只输出姓名。","张载"),
        ("CN8","郡县制是谁最早在全国推行的？只输出姓名。","秦始皇"),
    ],
    "science_tech": [
        ("ST1","麦克斯韦方程组有几个方程？只输出数字。","4"),
        ("ST2","信息熵 H(X) = -Σp(x)log₂p(x)。均匀分布4个事件的熵？只输出数字。","2"),
        ("ST3","光速在真空中的数值（km/s）？只输出数字。","300000"),
        ("ST4","DNA双螺旋结构中，A与什么配对？只输出字母。","T"),
        ("ST5","摩尔质量的单位是？只输出单位符号。","g/mol"),
        ("ST6","二进制数 1010 转十进制是多少？只输出数字。","10"),
        ("ST7","Python中 GIL 的全称是什么？只输出英文。","Global Interpreter Lock"),
        ("ST8","TCP四次挥手中，主动关闭方最后处于什么状态？只输出状态名。","TIME_WAIT"),
    ],
    "creative": [
        ("CR1","以'雨夜'为题写一首五言绝句。只输出诗。","雨"),
        ("CR2","用一句话证明你理解量子纠缠（不用术语，比喻）。","骰子"),
    ],
}

def ask(client, model, prompt, max_tok=500):
    try:
        r = client.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=max_tok)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERR:{e}]"

def fuzzy_match(expected, response, category):
    """Flexible answer checking for hard questions"""
    resp = response.lower()
    # Strip common prefixes
    for prefix in ['答案是','答：','答案：','结果：','结果是']:
        resp = resp.replace(prefix, '').strip()

    if category == "advanced_math":
        nums = re.findall(r'[\d./\-e²]+', resp.replace(' ',''))
        if nums:
            try:
                val = float(nums[-1].replace('²','**2'))
                # Try eval for fractions
                if '/' in expected:
                    target = eval(expected)
                elif expected == 'e²':
                    target = 2.71828**2
                elif expected == '-1/6':
                    target = -1/6
                elif expected == '1.00':
                    target = 1.0
                elif expected == '94/54145':
                    return 1.0 if expected in resp else 0.0
                else:
                    target = float(expected)
                return 1.0 if abs(val - target) / max(1, abs(target)) < 0.1 else 0.0
            except: pass
        return 1.0 if expected.lower() in resp else 0.0

    elif category == "algorithm":
        return 1.0 if expected.lower() in resp else 0.0

    elif category == "deep_logic":
        if expected.lower() in resp: return 1.0
        # Special checks
        if "DL3" in str(expected) and "45分钟" in resp: return 1.0
        if "DL6" in str(expected):
            if any(s in resp for s in ['¬Q→¬P','~Q→~P','not Q→not P','非Q→非P','!Q→!P']): return 1.0
        return 0.0

    elif category in ("chinese_depth","science_tech"):
        return 1.0 if expected.lower() in resp else 0.0

    elif category == "creative":
        if "五言" in expected: return 1.0 if len(resp) >= 20 else 0.0
        return 1.0 if len(resp) >= 10 else 0.0

    return 1.0 if expected.lower() in resp else 0.0

print(f"{'='*70}")
print(f"Evaluator v3 — HARD Dataset: {sum(len(v) for v in HARD.values())} questions × {len(CLIENTS)} models")
print(f"{'='*70}")

all_scores = {}
for name, (client, model) in CLIENTS.items():
    print(f"\n[{name}]")
    cat_s = {}; total=0; correct=0
    for cat, tests in HARD.items():
        sc = 0
        for tid, q, ans in tests:
            resp = ask(client, model, q, 600)
            score = fuzzy_match(ans, resp, cat)
            sc += score
        n = len(tests); pct = round(sc/n*100,1)
        cat_s[cat] = {"score":sc,"total":n,"pct":pct}
        total+=n; correct+=sc
        bar = '█'*int(pct/10) + '░'*(10-int(pct/10))
        print(f"  {cat:<18} {sc}/{n} ({pct}%) {bar}")
    all_scores[name] = {"cats":cat_s,"overall":round(correct/total*100,1),"correct":correct,"total":total}
    print(f"  {'OVERALL':<18} {correct}/{total} ({all_scores[name]['overall']}%)")

# Final comparison across all 3 eval rounds
print(f"\n{'='*70}")
print(f"ALL 3 ROUNDS COMPARISON")
print(f"{'='*70}")
print(f"{'Model':<12} {'V1 Easy':<10} {'V2 Med':<10} {'V3 Hard':<10} {'Trend':<10}")
print("-"*52)
# Load previous results
v1 = {"DS-Think":100,"GLM":94.7,"QWEN":89.5,"DS-PRO":84.2,"Kimi":73.7}
v2 = {"GLM":90.9,"DS-Think":87.9,"QWEN":81.8,"DS-PRO":75.8,"Kimi":72.7}
for m in ["GLM","DS-Think","DS-PRO","QWEN"]:
    v3 = all_scores[m]["overall"]
    t = f"{v2[m]-v3:+.1f}" if m in v2 else "new"
    print(f"{m:<12} {v1.get(m,'?'):.0f}%{'':>4} {v2.get(m,'?'):.1f}%{'':>4} {v3}%{'':>4} V2->V3 {t}")

print(f"\nV3 HARD Dataset Rankings:")
ranks = sorted(all_scores.items(), key=lambda x:-x[1]["overall"])
for i,(m,r) in enumerate(ranks):
    print(f"  {i+1}. {m}: {r['overall']}% ({r['correct']}/{r['total']})")

# Save
outdir = Path(__file__).parent
report = {
    "timestamp": datetime.now().isoformat(),
    "version": 3,
    "difficulty": "hard",
    "questions": sum(len(v) for v in HARD.values()),
    "results": {m: {"overall":r["overall"],"categories":{k:{"pct":v["pct"]} for k,v in r["cats"].items()}} for m,r in all_scores.items()},
}
latest = outdir / "eval_latest.json"
with open(latest,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
outfile = outdir / f"eval_hard_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
with open(outfile,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
print(f"\nSaved: {outfile}")
