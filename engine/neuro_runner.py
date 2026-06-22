#!/usr/bin/env python3
"""
neuro_runner.py — 7-Mechanism Neural Routing Harness

Mechanisms (each independent, paper-anchored):
  GridCellMap        → encode question in cognitive space
  PredictiveCoding   → predict model success, error-driven learning
  SynapticTagging    → tag important events for rapid consolidation
  HebbianSTDP        → wire together, fire together
  LateralInhibition  → winner suppresses competitors
  MemoryConsolidation → L-LTP for repeated success
  SynapticDecay      → forgetting curve
"""
import sys,os,json,time,threading,numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.neural_mechanisms import *
from engine.brain import call

MODELS=["ds-pro","ds-think","glm","qwen","kimi","groq"]
N=len(MODELS)
STATUS=os.path.expanduser("~/.claude/tools/neuro_status.json")

class NeuralRunner:
    def __init__(self):
        # Embedder
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder=SentenceTransformer('all-MiniLM-L6-v2')
            self.has_bert=True
        except: self.has_bert=False

        # All 7 mechanisms
        self.grid=GridCellMap(n_modules=4,dim=384 if self.has_bert else 384)
        self.pred=PredictiveCodingLayer(n_input=len(self.grid.modules),n_output=N)
        self.tags=SynapticTagging(n_synapses=N)
        self.hebb=HebbianSTDP(n_synapses=N)
        self.lateral=LateralInhibition(n_units=N)
        self.consolid=MemoryConsolidation(n_synapses=N)
        self.decay=SynapticDecay(decay_rate=0.0005)
        self.episodes=0

    def _embed(self,text):
        if self.has_bert:
            e=self.embedder.encode(text,convert_to_numpy=True).astype(np.float64)
            return e/(np.linalg.norm(e)+1e-8)
        v=np.zeros(384); tx=text.lower()
        for n in[2,3,4]:
            for i in range(len(tx)-n+1): v[hash(tx[i:i+n])%384]+=1.0
        return v/(np.linalg.norm(v)+1e-8)

    def _ws(self,stage,models=None,region=None,done=False):
        d={}
        if os.path.exists(STATUS):
            try:d=json.loads(open(STATUS,"r",encoding="utf-8").read())
            except:pass
        d["stage"]=stage;d["done"]=done;d["timestamp"]=time.time()
        if models:d["models"]=models
        os.makedirs(os.path.dirname(STATUS),exist_ok=True)
        with open(STATUS,"w",encoding="utf-8") as f:json.dump(d,f,ensure_ascii=False)

    def route(self,question):
        emb=self._embed(question)
        grid_vec=self.grid.encode(emb)           # 1. Grid: where in cognitive space?
        preds=self.pred.predict(grid_vec)         # 2. Predict: which model will succeed?
        # 3. Tag info: tagged models get exploration boost
        tag_boost=self.tags.tags*0.3
        # 4. Hebbian weights: learned pathway strength
        heb_boost=self.hebb.weights*0.2
        scores=preds+tag_boost+heb_boost
        # 5. Lateral inhibition: winner suppresses others
        if np.max(scores)>0:
            winner=int(np.argmax(scores))
            scores=self.lateral.apply(scores,winner)
        best=int(np.argmax(scores))
        return {"primary_model":MODELS[best],"predictions":{MODELS[i]:round(float(preds[i]),3) for i in np.argsort(-preds)[:4]}}

    def learn(self,question,model,reward):
        emb=self._embed(question); mi=MODELS.index(model)
        grid_vec=self.grid.encode(emb)
        # 2. Predictive coding: only prediction error drives update
        target=np.zeros(N); target[mi]=reward
        self.pred.update(grid_vec,target,mi,lr=0.01)
        # 3. Synaptic tagging: large errors set tags
        error=target-self.pred.predict(grid_vec)
        self.tags.update(error)
        # 4. Hebbian: pre (question) → post (model success/failure)
        self.hebb.pre_fire(mi); self.hebb.post_fire(mi,reward>0.5)
        # 5. Lateral: adapt inhibition
        best=int(np.argmax(self.pred.predict(grid_vec)))
        if best!=mi: self.lateral.adapt(best,mi,reward>0.5)
        # 6. Consolidation
        self.consolid.update(mi,reward>0.5)
        # 7. Decay
        self.hebb.weights=self.decay.apply(self.hebb.weights,[mi])
        self.episodes+=1

    def execute(self,question):
        self._ws("routing")
        d=self.route(question); m=d["primary_model"]
        self._ws("executing",models={m:"running"})
        try:
            prompt=f"Step by step. Answer precisely.\n\n{question}"
            resp=call(m,prompt,max_tok=2000); reward=1.0 if len(str(resp))>20 else 0.0
        except: resp="[ERR]"; reward=0.0
        self._ws("done",models={m:"done"})
        self.learn(question,m,reward)
        # Log
        rf=os.path.expanduser("~/.synapseflow/brain/last_result.json")
        os.makedirs(os.path.dirname(rf),exist_ok=True)
        with open(rf,"w",encoding="utf-8") as f: json.dump({"chosen":m,"reward":reward,"episode":self.episodes,"ts":time.time()},f)
        return {"model":m,"reward":reward,"response":str(resp)[:500],"predictions":d["predictions"]}

if __name__=="__main__":
    r=NeuralRunner()
    qs=[("Solve 2x+5=17 step by step","ds-think"),("What is the capital of France?","groq"),("Write Python quicksort","ds-pro")]
    for epoch in range(3):
        print(f"\n--- Epoch {epoch+1} ---")
        for q,exp in qs:
            res=r.execute(q); ok="OK" if res["model"]==exp else "LEARN"
            print(f"{ok} {q[:35]:35s} -> {res['model']:10s} r={res['reward']} pred={list(res['predictions'].items())[:2]}")
    print(f"\nEpisodes: {r.episodes} | Consolidated: {sum(r.consolid.consolidated)} | Tagged: {sum(r.tags.tags>0)}")
    print("NeuralRunner: ALL 7 MECHANISMS ACTIVE")
