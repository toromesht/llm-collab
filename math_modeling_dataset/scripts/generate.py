#!/usr/bin/env python3
"""
Synthetic math modeling data generator.
Uses scipy/pulp to solve problems, then wraps in training format.
"""
import json, random, sys
from pathlib import Path

BASE = Path(__file__).parent.parent / "data"

def generate_transportation_problem(n_warehouses=3, n_customers=4):
    """Generate a classic transportation LP problem."""
    warehouses = [f"W{i+1}" for i in range(n_warehouses)]
    customers = [f"C{i+1}" for i in range(n_customers)]
    supply = [random.randint(80, 200) for _ in range(n_warehouses)]
    demand = [random.randint(60, 150) for _ in range(n_customers)]
    # Balance
    total_s = sum(supply)
    total_d = sum(demand)
    demand = [int(d * total_s / total_d) for d in demand]
    cost = [[random.randint(5, 50) for _ in range(n_customers)] for _ in range(n_warehouses)]

    cost_str = "\\n".join([f"  W{i+1}: " + ", ".join(f"C{j+1}={cost[i][j]}" for j in range(n_customers)) for i in range(n_warehouses)])

    problem = f"""某物流公司有{n_warehouses}个仓库和{n_customers}个客户点。
各仓库日供应量: {dict(zip(warehouses, supply))}
各客户日需求量: {dict(zip(customers, demand))}
单位运输成本矩阵:
{cost_str}

请建立数学模型，在满足所有客户需求的前提下最小化总运输成本。"""

    code = f'''from pulp import *
import numpy as np

warehouses = {warehouses}
customers = {customers}
supply = {dict(zip(warehouses, supply))}
demand = {dict(zip(customers, demand))}
cost = {cost}

prob = LpProblem("Transportation", LpMinimize)
x = LpVariable.dicts("x", (warehouses, customers), lowBound=0, cat="Continuous")

prob += lpSum([cost[i][j] * x[warehouses[i]][customers[j]] for i in range(len(warehouses)) for j in range(len(customers))])

for i, w in enumerate(warehouses):
    prob += lpSum([x[w][c] for c in customers]) <= supply[w]
for j, c in enumerate(customers):
    prob += lpSum([x[w][c] for w in warehouses]) >= demand[c]

prob.solve()
print(f"Status: {{LpStatus[prob.status]}}")
print(f"Optimal Cost: {{value(prob.objective)}}")
for w in warehouses:
    for c in customers:
        if value(x[w][c]) > 0:
            print(f"  {{w}} -> {{c}}: {{value(x[w][c])}} units")'''

    return {
        "domain": "物流与供应链",
        "difficulty": "简单",
        "model_type": "线性规划",
        "problem_statement": problem,
        "modeling_process": {
            "variables": "x_ij: 从仓库i运往客户j的货物数量",
            "objective_function": "min Z = sum_i sum_j c_ij * x_ij",
            "constraints": [
                "供应约束: sum_j x_ij <= S_i (每个仓库)",
                "需求约束: sum_i x_ij >= D_j (每个客户)",
                "非负: x_ij >= 0"
            ],
            "type": "线性规划 (Linear Programming)"
        },
        "solution_code": {"language": "Python", "library": "PuLP", "code": code},
        "solution_analysis": f"总供应量{total_s}，总需求量{total_d}。利用PuLP求解器得到最优运输方案和最小总成本。",
        "keywords": ["运输问题", "线性规划", "PuLP", "成本最小化"],
        "source": "synthetic_generation"
    }

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = BASE / "optimization" / "transportation.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            record = generate_transportation_problem(random.randint(2,5), random.randint(3,6))
            record["id"] = f"opt_trans_{i+1:04d}"
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {n} transportation problems -> {out}")
