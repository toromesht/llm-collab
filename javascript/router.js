// SynapseFlow Router — JavaScript (browser/Node.js)
class SynapseRouter {
  constructor() {
    this.models = ['DS-V4','Qwen3-235B','GLM-4+','Groq-Llama','Kimi','GLM-4','QWEN','SJTU-DS-Think'];
    this.weights = {};
    this.round = 0;
    for (const m of this.models) {
      this.weights[m] = { math:0.5, code:0.5, logic:0.1, knowledge:0.5, writing:0.5,
                          lastCorrect:0, lastWrong:0, consWrong:0, banned:false };
    }
  }

  extract(q) {
    const cnt = (...kws) => kws.filter(k => q.includes(k)).length;
    return {
      math: cnt('定理','证明','方程','优化','概率','统计'),
      code: cnt('代码','Python','函数','编程','SQL'),
      logic: cnt('推理','悖论','说谎','逻辑','真话'),
      knowledge: cnt('什么是','定义','历史','解释'),
      group: cnt('群论','同态','置换群'),
      graph: cnt('图论','最短路径','连通'),
      calc: cnt('积分','导数','极限'),
      prob: cnt('概率','期望','分布'),
    };
  }

  route(question) {
    const f = this.extract(question);
    let best = this.models[0], bestScore = -Infinity;
    for (const [m, w] of Object.entries(this.weights)) {
      if (w.banned) continue;
      const score = f.code*w.code + f.math*w.math + f.logic*w.logic + f.knowledge*w.knowledge;
      if (score > bestScore) { bestScore = score; best = m; }
    }
    return best;
  }

  update(model, category, correct) {
    const w = this.weights[model], key = category in w ? category : 'knowledge';
    const dt = Math.max(1, this.round - (correct ? w.lastCorrect : w.lastWrong));
    const dw = correct ? 0.15 * Math.exp(-dt/5) : -0.10 * Math.exp(-dt/3);
    w[key] = Math.max(-1, Math.min(1, w[key] + dw));
    if (w[key] < -0.3 && ++w.consWrong >= 5) w.banned = true;
    this.round++;
  }
}

// Node.js export
if (typeof module !== 'undefined') module.exports = { SynapseRouter };
