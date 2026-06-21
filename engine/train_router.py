#!/usr/bin/env python3
"""
Frontier Router Training — 轻量分类器学习最优API调度
  输入: 22维问题特征向量
  输出: 8模型概率分布
  训练数据: dataset/training_array.json (20条)
  方法: STDP-inspired weight updates (biologically plausible)
  算力: CPU即可, 无需GPU
"""
import json, math, random, sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# Load training data
data = json.loads((Path(__file__).parent.parent / "dataset" / "training_array.json").read_text(encoding="utf-8"))

# Model list
MODELS = ["DS-V4", "Qwen3-235B", "GLM-4+", "Groq-Llama", "Kimi", "GLM-4", "QWEN", "DS-Think"]
N_FEATURES = 12  # simplified feature vector

def encode_features(entry):
    """Convert entry features to 12-dim vector"""
    feat = entry.get("features", {})
    keys = ["math","code","logic","knowledge","writing","arch",
            "chinese","safety","science","creative","general","db"]
    return [feat.get(k, 0) for k in keys]

def softmax(x):
    e = [math.exp(xi - max(x)) for xi in x]
    s = sum(e)
    return [ei/s for ei in e]

# ─── Router Model (simple linear + softmax) ──────────
class RouterModel:
    def __init__(self):
        # Weight matrix: N_MODELS x N_FEATURES
        self.W = [[random.uniform(-0.1, 0.1) for _ in range(N_FEATURES)]
                   for _ in range(len(MODELS))]
        self.bias = [0.0] * len(MODELS)
        self.lr = 0.01

    def forward(self, x):
        scores = [sum(w*f for w,f in zip(w_row, x)) + b
                  for w_row, b in zip(self.W, self.bias)]
        return softmax(scores)

    def train_step(self, x, target_model, lr_scale=1.0):
        """STDP-inspired update: reinforce target, suppress others"""
        probs = self.forward(x)
        target_idx = MODELS.index(target_model) if target_model in MODELS else 0

        for i in range(len(MODELS)):
            # Target: increase probability, Others: decrease
            delta = (1.0 - probs[i]) if i == target_idx else -probs[i]
            for j in range(N_FEATURES):
                self.W[i][j] += self.lr * lr_scale * delta * x[j]
            self.bias[i] += self.lr * lr_scale * delta

    def predict(self, entry):
        x = encode_features(entry)
        probs = self.forward(x)
        ranked = sorted(zip(MODELS, probs), key=lambda p: -p[1])
        return ranked

# ─── Training ────────────────────────────────────────
print("="*60)
print("Frontier Router Training — STDP-inspired classifier")
print(f"Models: {len(MODELS)} | Features: {N_FEATURES}")
print(f"Training data: {len(data['entries'])} entries")
print("="*60)

router = RouterModel()
epochs = 100

for epoch in range(epochs):
    random.shuffle(data['entries'])
    loss = 0
    for entry in data['entries']:
        best = entry.get('best_model', '')
        if not best or best == 'ALL': continue
        x = encode_features(entry)
        for model in best.replace('->','+').split('+'):
            model = model.strip()
            if model in MODELS:
                router.train_step(x, model)

    if epoch % 20 == 0:
        correct = 0; total = 0
        for entry in data['entries']:
            best = entry.get('best_model', '')
            if not best or best == 'ALL': continue
            ranked = router.predict(entry)
            top1 = ranked[0][0]
            if top1 in best: correct += 1
            total += 1
        print(f"Epoch {epoch}: accuracy={correct/total*100:.0f}% ({correct}/{total})")

# ─── Save model ──────────────────────────────────────
print(f"\nFinal accuracy: {correct}/{total} ({correct/total*100:.0f}%)")

# Test: show routing for key scenarios
tests = [
    ({"math":3,"code":0,"logic":0}, "Math hard problem"),
    ({"code":3,"math":0,"logic":0}, "Code generation"),
    ({"logic":3,"math":0,"code":0},"Logic puzzle"),
    ({"knowledge":3,"chinese":2}, "Chinese knowledge"),
]
for feat, desc in tests:
    entry = {"features": feat}
    ranked = router.predict(entry)
    print(f"\n{desc}:")
    for m, p in ranked[:3]:
        bar = "█"*int(p*20)
        print(f"  {m:<15} {p:.2%} {bar}")

# Save weights
out = Path(__file__).parent.parent / "dataset" / "router_weights.json"
weights_data = {"W": router.W, "bias": router.bias, "models": MODELS, "features": N_FEATURES}
with open(out, "w", encoding="utf-8") as f:
    json.dump(weights_data, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out}")
