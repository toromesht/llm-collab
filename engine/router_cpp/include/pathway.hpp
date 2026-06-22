#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <array>

namespace synapse {

// ─── Feature Signature (22-dim) ──────────────────────────
// Maps to brain region activation pattern
using FeatureSig = std::array<float, 22>;

// ─── Pathway: a learned route from brain region to model ─
struct Pathway {
    std::string pathway_id;       // e.g. "motor_cortex::ds-pro::code"
    std::string source_region;    // brain region
    std::string target_model;     // LLM model ID
    std::string category;         // code | math | logic | writing | general

    FeatureSig feature_signature{};

    // ─── STDP Weights (Spike-Timing-Dependent Plasticity) ──
    float weight_stdp = 0.0f;     // Hebbian: LTP/LTD learning
    float weight_bcm  = 0.0f;     // Bienenstock-Cooper-Munro sliding threshold
    float weight_ltp  = 0.0f;     // Long-term potentiation

    // ─── State ────────────────────────────────────────────
    // Protected by RoutingTable::shared_mutex (PostgreSQL buffer pattern).
    // All read/write to these fields must hold at least shared_lock.
    uint64_t use_count   = 0;
    uint64_t fail_count  = 0;
    float    success_rate = 0.5f;

    bool hardened = false;        // Route consolidated (frozen)
    bool pruned   = false;        // Route eliminated

    double created_at = 0.0;
    double last_used  = 0.0;
    double last_failed = 0.0;

    // ─── Cosine Similarity ────────────────────────────────
    [[nodiscard]] float match_score(const FeatureSig& query) const noexcept;
};

// ─── Routing Decision ────────────────────────────────────
enum class Action { Single, Pipeline, Collab };

struct RoutingDecision {
    Action action = Action::Single;
    std::string model;               // best single model
    std::string region;              // brain region
    std::vector<std::string> models; // for collab
    float confidence = 0.0f;
    float latency_ns = 0.0f;        // decision latency

    [[nodiscard]] bool is_single()   const { return action == Action::Single; }
    [[nodiscard]] bool is_collab()   const { return action == Action::Collab; }
    [[nodiscard]] bool is_pipeline() const { return action == Action::Pipeline; }
};

// ─── Task Dimensions ─────────────────────────────────────
struct TaskDims {
    float code     = 0.0f;
    float math     = 0.0f;
    float logic    = 0.0f;
    float arch     = 0.0f;
    float writing  = 0.0f;
    float general  = 0.0f;

    [[nodiscard]] float difficulty() const noexcept {
        return 0.3f * std::max({code, math, logic})
             + 0.2f * std::max({arch, writing})
             + 0.2f * general;
    }
};

// ─── Routing Request ─────────────────────────────────────
struct RouteRequest {
    FeatureSig features;
    TaskDims    dims;
    std::string question_hash;  // for TDA cache
    bool        force_neuro = false;
};

} // namespace synapse
