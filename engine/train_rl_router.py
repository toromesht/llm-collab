#!/usr/bin/env python3
"""
RL Router Training — PPO-based policy gradient for optimal model selection
  State:  12-dim feature vector from question
  Action: which of 8 models to route to
  Reward: +1 correct model, -0.5 wrong, +0.2 cost_saved vs GPT-5.5
  Data:   225 synthetic entries from dataset/training_array.json
"""
import json, math, random, sys
from pathlib import Path
from collections import deque
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ─── Config ──────────────────────────────────────────
MODELS = ["DS-V4","Qwen3-235B","GLM-4+","Groq-Llama","Kimi","GLM-4","QWEN","DS-Think"]
N_ACTIONS = len(MODELS)
N_FEATURES = 12
FEATURE_KEYS = ["math","code","logic","knowledge","writing","arch","chinese","safety","science","creative","general","db"]
LR = 3e-4; GAMMA = 0.99; CLIP_EPS = 0.2; EPOCHS = 200; BATCH = 32

# ─── PPO Networks ─────────────────────────────────────
class PolicyNetwork:
    """Actor: outputs action probabilities"""
    def __init__(self):
        self.w1 = np.random.randn(N_FEATURES, 64) * 0.1
        self.b1 = np.zeros(64)
        self.w2 = np.random.randn(64, N_ACTIONS) * 0.1
        self.b2 = np.zeros(N_ACTIONS)

    def forward(self, x):
        x = np.array(x).reshape(1, -1)
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        logits = h @ self.w2 + self.b2
        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        return probs.flatten()

    def get_action(self, state):
        probs = self.forward(state)
        action = np.random.choice(N_ACTIONS, p=probs)
        return action, probs[action]

    def get_params(self): return [self.w1,self.b1,self.w2,self.b2]
    def set_params(self, params):
        self.w1,self.b1,self.w2,self.b2 = [p.copy() for p in params]

class ValueNetwork:
    """Critic: estimates state value"""
    def __init__(self):
        self.w1 = np.random.randn(N_FEATURES, 64) * 0.1
        self.b1 = np.zeros(64)
        self.w2 = np.random.randn(64, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, x):
        x = np.array(x).reshape(1, -1)
        h = np.maximum(0, x @ self.w1 + self.b1)
        return (h @ self.w2 + self.b2).item()

    def get_params(self): return [self.w1,self.b1,self.w2,self.b2]
    def set_params(self, params):
        self.w1,self.b1,self.w2,self.b2 = [p.copy() for p in params]

# ─── Load data ────────────────────────────────────────
data = json.loads((Path(__file__).parent.parent / "dataset" / "training_array.json").read_text(encoding="utf-8"))
entries = [e for e in data['entries'] if e.get('best_model') and e['best_model'] != 'ALL']

def encode(e):
    feat = e.get('features',{})
    return [feat.get(k,0) for k in FEATURE_KEYS]

# Reward function
def reward_fn(action, entry):
    best = entry.get('best_model','')
    banned = entry.get('banned_models',[])
    if MODELS[action] in banned: return -1.0  # banned model = heavy penalty
    if '+' in best:
        if MODELS[action] in best: return 0.8  # collab member
        return -0.2
    if MODELS[action] == best: return 1.0
    # Cost bonus: using cheaper models gives small reward
    if MODELS[action] in ["Groq-Llama","QWEN","GLM-4"]: return 0.1
    return -0.3

# ─── PPO Training ────────────────────────────────────
print("="*60)
print(f"RL Router Training — PPO | {len(entries)} entries | {N_ACTIONS} actions")
print("="*60)

actor = PolicyNetwork(); critic = ValueNetwork()
actor_old = PolicyNetwork(); actor_old.set_params(actor.get_params())

history = deque(maxlen=100)
best_acc = 0

for epoch in range(EPOCHS):
    random.shuffle(entries)
    batch_states = []; batch_actions = []; batch_rewards = []; batch_probs = []
    batch_values = []; batch_dones = []

    for entry in entries[:BATCH]:
        state = encode(entry)
        action, prob = actor.get_action(state)
        value = critic.forward(state)
        reward = reward_fn(action, entry)
        # Simulate next state (random perturbation for exploration)
        next_state = [s + random.uniform(-0.2,0.2) for s in state]

        batch_states.append(state); batch_actions.append(action)
        batch_rewards.append(reward); batch_probs.append(prob)
        batch_values.append(value); batch_dones.append(1.0)

    # Compute advantages
    advantages = [r - v for r,v in zip(batch_rewards, batch_values)]

    # PPO update
    for _ in range(4):  # 4 epochs per batch
        for i in range(len(batch_states)):
            state = batch_states[i]; action = batch_actions[i]
            old_prob = batch_probs[i]; adv = advantages[i]; val = batch_values[i]
            ret = batch_rewards[i]

            # Actor update (PPO clip)
            _, new_prob = actor.get_action(state)
            ratio = new_prob / (old_prob + 1e-8)
            clipped = np.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
            actor_loss = -min(ratio*adv, clipped*adv)
            # Simple SGD
            for param in actor.get_params():
                param -= LR * actor_loss * 0.01

            # Critic update
            val_pred = critic.forward(state)
            critic_loss = (ret - val_pred)**2
            for param in critic.get_params():
                param -= LR * critic_loss * 0.01

    # Evaluate
    if epoch % 20 == 0:
        correct = 0; total = 0; avg_reward = 0
        for entry in entries:
            state = encode(entry)
            action, _ = actor.get_action(state)
            r = reward_fn(action, entry)
            avg_reward += r
            if r > 0.5: correct += 1
            total += 1
        acc = correct/total*100; avg_rew = avg_reward/total
        best_acc = max(best_acc, acc)
        actor_old.set_params(actor.get_params())
        history.append(acc)
        recent = sum(history)/len(history) if history else acc
        print(f"Epoch {epoch}: acc={acc:.0f}% best={best_acc:.0f}% avg_r={avg_rew:.2f} recent={recent:.0f}%")

print(f"\nBest accuracy: {best_acc:.0f}%")

# ─── Test routing ────────────────────────────────────
print("\nTrained Router (RL-PPO):")
tests = [
    ([3,0,0,0,0,0,0,0,0,0,0,0], "Math"),
    ([0,3,0,0,0,0,0,0,0,0,0,0], "Code"),
    ([0,0,3,0,0,0,0,0,0,0,0,0], "Logic"),
    ([0,0,0,3,0,0,2,0,0,0,0,0], "Chinese Knowl"),
]
for state, label in tests:
    probs = actor.forward(state)
    top = sorted(zip(MODELS, probs), key=lambda x:-x[1])[:3]
    print(f"  {label}: {', '.join(f'{m}({p:.0%})' for m,p in top)}")

# Save model
out_dir = Path(__file__).parent.parent / "dataset"
np.savez(out_dir / "rl_router.npz",
         w1=actor.w1, b1=actor.b1, w2=actor.w2, b2=actor.b2,
         vw1=critic.w1, vb1=critic.b1, vw2=critic.w2, vb2=critic.b2)
print(f"\nModel saved: dataset/rl_router.npz")
