// SynapseFlow Router — C++ optimized engine
// 10-100x faster than Python for high-throughput routing

#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace synapse {

struct RouterConfig {
    double stdp_A_plus = 0.15;     // LTP amplitude
    double stdp_A_minus = 0.10;    // LTD amplitude
    double stdp_tau_plus = 5.0;    // LTP time constant
    double stdp_tau_minus = 3.0;   // LTD time constant
    double bcm_alpha = 0.01;       // BCM threshold learning rate
    double lateral_alpha = 0.03;   // lateral inhibition
    double prune_threshold = -0.3; // pruning threshold
    int prune_rounds = 5;          // rounds before pruning
};

struct ModelWeight {
    double math = 0.5, code = 0.5, logic = 0.1, knowledge = 0.5, writing = 0.5;
    int last_correct = 0, last_wrong = 0, consecutive_wrong = 0;
    double firing_rate = 0.5;
    bool banned = false;
};

class Router {
    RouterConfig cfg;
    std::unordered_map<std::string, ModelWeight> weights;
    int round_counter = 0;

public:
    Router() {
        for (auto& m : {"DS-V4","Qwen3-235B","GLM-4+","Groq-Llama","Kimi","GLM-4","QWEN","DS-Think"})
            weights[m] = ModelWeight{};
    }

    // Feature extraction (10 math subfields + core categories)
    struct Features {
        double code=0, math=0, logic=0, knowledge=0, writing=0, arch=0;
        double group_theory=0, graph_theory=0, topology=0;
        double linear_algebra=0, calculus=0, probability=0;
        double number_theory=0, diff_eq=0, combinatorics=0, optimization=0;
    };

    Features extract(const std::string& q) {
        Features f;
        auto cnt = [&](const std::string& kw) { return q.find(kw) != std::string::npos; };
        f.math = cnt("定理")+cnt("证明")+cnt("方程")+cnt("优化");
        f.code = cnt("代码")+cnt("SQL")+cnt("Python")+cnt("函数");
        f.logic = cnt("推理")+cnt("悖论")+cnt("逻辑");
        f.knowledge = cnt("什么是")+cnt("定义")+cnt("历史");
        f.group_theory = cnt("群论")+cnt("同态")+cnt("置换群");
        f.graph_theory = cnt("图论")+cnt("连通")+cnt("最短路径");
        f.topology = cnt("拓扑")+cnt("同胚")+cnt("流形");
        f.calculus = cnt("积分")+cnt("导数")+cnt("极限");
        f.probability = cnt("概率")+cnt("期望")+cnt("分布");
        f.optimization = cnt("最优化")+cnt("凸优化")+cnt("线性规划");
        return f;
    }

    double difficulty(const Features& f) {
        double c = f.code + f.math*1.5 + f.logic*1.3 + f.arch*1.2;
        return 1.0 / (1.0 + std::exp(-c / 5.0));
    }

    // STDP weight update
    void update(const std::string& model, const std::string& category, bool correct) {
        auto& w = weights[model];
        double* wp = &w.math;
        if (category == "code") wp = &w.code;
        else if (category == "logic") wp = &w.logic;
        else if (category == "knowledge") wp = &w.knowledge;
        else if (category == "writing") wp = &w.writing;

        int dt = round_counter - (correct ? w.last_correct : w.last_wrong);
        if (dt <= 0) dt = 1;
        double dw = correct
            ? cfg.stdp_A_plus * std::exp(-dt / cfg.stdp_tau_plus)
            : -cfg.stdp_A_minus * std::exp(-dt / cfg.stdp_tau_minus);
        *wp = std::max(-1.0, std::min(1.0, *wp + dw));

        if (correct) { w.last_correct = round_counter; w.consecutive_wrong = 0; }
        else { w.last_wrong = round_counter; w.consecutive_wrong++; }

        if (*wp < cfg.prune_threshold && w.consecutive_wrong >= cfg.prune_rounds)
            w.banned = true;
        round_counter++;
    }

    std::string best_model(const std::string& question) {
        auto f = extract(question);
        double best_score = -999; std::string best;
        for (auto& [name, w] : weights) {
            if (w.banned) continue;
            double score = f.code * w.code + f.math * w.math
                         + f.logic * w.logic + f.knowledge * w.knowledge
                         + f.writing * w.writing;
            if (score > best_score) { best_score = score; best = name; }
        }
        return best;
    }
};

} // namespace synapse
