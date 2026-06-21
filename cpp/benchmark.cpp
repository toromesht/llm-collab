// C++ Benchmark: compare routing speed vs Python
#include "router.h"
#include <iostream>
#include <chrono>

int main() {
    synapse::Router router;
    std::vector<std::string> questions = {
        "证明群论中的拉格朗日定理",
        "用Python写一个快速排序算法",
        "甲说乙说谎，乙说丙说谎，谁说真话？",
        "什么是量子纠缠？",
        "设计一个分布式数据库架构",
    };

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; i++) {
        for (auto& q : questions) {
            router.best_model(q);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "50,000 routing decisions in " << ms << "ms (" << ms/50.0 << " us/decision)\n";
    return 0;
}
