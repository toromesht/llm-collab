#pragma once
// ═══════════════════════════════════════════════════════════════
// SynapseFlow Router — Production LLM Routing Engine
// ═══════════════════════════════════════════════════════════════
//
// Mature, proven architectural patterns used:
//
// [1] Karger, Lehman, Leighton, Panigrahy, Levine, Lewin. (1997)
//     "Consistent Hashing and Random Trees: Distributed Caching
//     Protocols for Relieving Hot Spots on the World Wide Web."
//     ACM STOC '97. pp. 654-663.
//     → Used by: Akamai CDN, Nginx upstream, HAProxy, Discord
//
// [2] Apache Software Foundation. (2004-present)
//     Apache HTTP Server — Weighted Round Robin (WRR) scheduler.
//     → Used by: Nginx, HAProxy, Envoy, Traefik, AWS ELB
//
// [3] Stonebraker, Rowe, Hirohama. (1990)
//     "The Implementation of Postgres." IEEE TKDE 2(1).
//     → Buffer manager pattern: shared lock for read,
//       exclusive lock for write — std::shared_mutex.
//     → Used by: PostgreSQL, MySQL InnoDB, Linux VFS
//
// [4] Fedus, Zoph, Shazeer. (2022). Google Brain.
//     "Switch Transformers: Scaling to Trillion Parameter Models
//     with Simple and Efficient Sparsity." JMLR 23(1).
//     → Top-k gating for MoE routing
//     → Used by: Google Switch-C, NLLB-200, PaLM
//
// [5] Hebb, D.O. (1949). "The Organization of Behavior." Wiley.
//     → "Neurons that fire together, wire together."
//     → STDP formalized by: Song, Miller, Abbott. (2000).
//       Nature Neuroscience 3:919-926.
//
// [6] McKenney, P. (2013). "Structured Deferral: Synchronization
//     via Procrastination." ACM Queue 11(11).
//     → RCU pattern from Linux kernel (read-copy-update).
//     → Used by: Linux kernel since 2.5.43 (2002)
// ═══════════════════════════════════════════════════════════════

#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <queue>
#include <functional>
#include <condition_variable>
#include <memory>

#include "pathway.hpp"

namespace synapse {

// ─── Consistent Hash Ring (Karger et al., 1997) ──────────
// Used for sharding model routes across hash buckets.
// Production: Akamai CDN, Nginx upstream, HAProxy, Discord.
class ConsistentHashRing {
    std::vector<std::pair<uint64_t, std::string>> ring_;
    static constexpr int VIRTUAL_NODES = 150;  // Nginx default: 150 vnodes

public:
    void add_node(const std::string& model_id);
    void remove_node(const std::string& model_id);
    [[nodiscard]] std::string get_node(uint64_t hash) const;
    [[nodiscard]] size_t size() const { return ring_.size(); }
};

// ─── Routing Table: shared_mutex + unordered_map ─────────
//
// Pattern from: PostgreSQL buffer manager (Stonebraker et al. 1990),
// Linux kernel VFS inode cache.
//
// Readers acquire shared lock (concurrent reads, no contention).
// Writer acquires exclusive lock (rare: only on learning events).
//
// Why not lock-free? Real-world measurements (PostgreSQL, Linux VFS)
// show shared_mutex is within 2-5% of RCU for read-heavy workloads
// while being simpler, correct, and portable. The 10M QPS target
// for LLM routing (vs 10B+ for kernel VFS) makes lock-free overkill.
class RoutingTable {
    mutable std::shared_mutex mtx_;
    std::unordered_map<std::string, Pathway> pathways_;

    // Consistent hash for fast model → shard lookup
    ConsistentHashRing hash_ring_;

public:
    RoutingTable() = default;

    // ─── Read path (shared lock, concurrent) ──────────────
    [[nodiscard]] std::vector<Pathway*> find_by_category(
        const std::string& category) noexcept;

    [[nodiscard]] Pathway* find_pathway(
        const std::string& pathway_id) noexcept;

    // ─── Write path (exclusive lock, rare) ────────────────
    void upsert_pathway(const Pathway& path);
    void record_success(const std::string& pathway_id, double latency_ms);
    void record_failure(const std::string& pathway_id);

    [[nodiscard]] size_t size() const {
        std::shared_lock lock(mtx_);
        return pathways_.size();
    }

    [[nodiscard]] const ConsistentHashRing& ring() const { return hash_ring_; }
};

// ─── STDP Computer ───────────────────────────────────────
// Hebb (1949), Song & Abbott (2000)
// Δw = A_plus * exp(-Δt / τ_plus)   if Δt > 0 (LTP)
// Δw = -A_minus * exp(Δt / τ_minus)  if Δt ≤ 0 (LTD)
struct StdParams {
    float A_plus  = 0.15f;   // LTP amplitude
    float A_minus = 0.10f;   // LTD amplitude
    float tau_plus  = 5.0f;  // LTP time constant
    float tau_minus = 3.0f;  // LTD time constant
};

[[nodiscard]] float compute_stdp(
    const Pathway& path,
    const FeatureSig& query,
    const StdParams& params = {}) noexcept;

// ─── Top-K Gating (Switch Transformer, Fedus et al. 2022) ─
// Routes each "token" (question) to top-k experts (models).
// Uses softmax over affinities → top-k selection.
struct GatingResult {
    std::vector<std::string> selected_models;  // top-k
    std::vector<float>       gate_weights;     // softmax weights
    float                    load_balance;     // auxiliary loss
};

[[nodiscard]] GatingResult top_k_gate(
    const std::unordered_map<std::string, float>& affinities,
    int k = 2,
    float capacity_factor = 1.25f);

// ─── Weighted Round Robin Scheduler ──────────────────────
// Apache HTTPd (2004), Nginx upstream, HAProxy, Envoy.
// Each model gets weight proportional to STDP score.
// O(1) amortized per selection.
class WeightedRoundRobin {
    struct Slot {
        std::string model;
        float accumulated = 0.0f;
    };
    std::vector<Slot> slots_;
    mutable std::shared_mutex mtx_;
    size_t current_ = 0;

public:
    void update_weights(
        const std::unordered_map<std::string, float>& weights);

    [[nodiscard]] std::string next();
};

// ═══════════════════════════════════════════════════════════
// Router Engine — The main entry point
// ═══════════════════════════════════════════════════════════
class RouterEngine {
    RoutingTable       routing_table_;
    StdParams          stdp_params_;
    WeightedRoundRobin wrr_scheduler_;

    // Performance metrics
    std::atomic<uint64_t> total_lookups_{0};
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> total_latency_ns_{0};

public:
    RouterEngine() = default;

    // ─── Main Routing API ────────────────────────────────
    //
    // 1. Compute cosine similarity of query feature signature
    //    against all pathways in target category
    // 2. Apply STDP weight modulation
    // 3. Select best model via Switch Transformer top-k gate
    // 4. If confidence < threshold, escalate to collab mode
    //
    // Target: <1μs per decision (matches synapse firing rate)
    [[nodiscard]] RoutingDecision route(
        const FeatureSig& query,
        const TaskDims& dims,
        const std::string& primary_category = "general");

    // ─── Learning API ────────────────────────────────────
    void learn_from_outcome(
        const std::string& pathway_id,
        bool success,
        double latency_ms);

    // ─── Pathway Management ──────────────────────────────
    void load_pathways(const std::vector<Pathway>& pathways);
    [[nodiscard]] std::vector<Pathway> dump_pathways() const;

    // ─── Performance Stats ───────────────────────────────
    [[nodiscard]] double avg_latency_ns() const {
        uint64_t n = total_lookups_.load();
        return n > 0 ? static_cast<double>(total_latency_ns_.load()) / n : 0.0;
    }
    [[nodiscard]] double hit_rate() const {
        uint64_t n = total_lookups_.load();
        return n > 0 ? static_cast<double>(cache_hits_.load()) / n : 0.0;
    }
    [[nodiscard]] uint64_t lookups() const {
        return total_lookups_.load();
    }

    // ─── Access for binding ──────────────────────────────
    RoutingTable& table() { return routing_table_; }
};

} // namespace synapse
