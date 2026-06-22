#include "router.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace synapse {

// ═══════════════════════════════════════════════════════════
// Consistent Hash Ring (Karger et al., 1997)
// ═══════════════════════════════════════════════════════════

void ConsistentHashRing::add_node(const std::string& model_id) {
    for (int i = 0; i < VIRTUAL_NODES; ++i) {
        std::string vnode = model_id + "#" + std::to_string(i);
        uint64_t hash = std::hash<std::string>{}(vnode);
        ring_.emplace_back(hash, model_id);
    }
    std::sort(ring_.begin(), ring_.end());
}

void ConsistentHashRing::remove_node(const std::string& model_id) {
    ring_.erase(
        std::remove_if(ring_.begin(), ring_.end(),
            [&](const auto& p) { return p.second == model_id; }),
        ring_.end());
}

std::string ConsistentHashRing::get_node(uint64_t hash) const {
    if (ring_.empty()) return "";
    auto it = std::lower_bound(ring_.begin(), ring_.end(), hash,
        [](const auto& p, uint64_t h) { return p.first < h; });
    return (it != ring_.end()) ? it->second : ring_[0].second;
}

// ═══════════════════════════════════════════════════════════
// Routing Table (PostgreSQL Buffer Manager Pattern)
// ═══════════════════════════════════════════════════════════

std::vector<Pathway*> RoutingTable::find_by_category(
    const std::string& category) noexcept
{
    std::shared_lock lock(mtx_);
    std::vector<Pathway*> result;
    for (auto& [id, pw] : pathways_) {
        if (pw.category == category && !pw.pruned) {
            result.push_back(&pw);
        }
    }
    return result;
}

Pathway* RoutingTable::find_pathway(
    const std::string& pathway_id) noexcept
{
    std::shared_lock lock(mtx_);
    auto it = pathways_.find(pathway_id);
    return (it != pathways_.end()) ? &it->second : nullptr;
}

void RoutingTable::upsert_pathway(const Pathway& path) {
    std::unique_lock lock(mtx_);
    auto it = pathways_.find(path.pathway_id);
    if (it != pathways_.end()) {
        // Update fields while preserving counters
        uint64_t uc = it->second.use_count;
        uint64_t fc = it->second.fail_count;
        it->second = path;
        it->second.use_count = uc;
        it->second.fail_count = fc;
    } else {
        pathways_[path.pathway_id] = path;
        hash_ring_.add_node(path.target_model);
    }
}

void RoutingTable::record_success(
    const std::string& pathway_id, double latency_ms)
{
    std::unique_lock lock(mtx_);  // exclusive: modifying counters
    auto it = pathways_.find(pathway_id);
    if (it == pathways_.end()) return;

    auto& pw = it->second;
    pw.use_count++;
    uint64_t uses = pw.use_count;
    uint64_t fails = pw.fail_count;
    pw.success_rate = static_cast<float>(uses - fails) / static_cast<float>(uses);
}

void RoutingTable::record_failure(const std::string& pathway_id) {
    std::unique_lock lock(mtx_);
    auto it = pathways_.find(pathway_id);
    if (it == pathways_.end()) return;

    auto& pw = it->second;
    pw.fail_count++;
    pw.use_count++;
    uint64_t total = pw.use_count;
    uint64_t fails = pw.fail_count;
    pw.success_rate = static_cast<float>(total - fails) / static_cast<float>(total);
}

// ═══════════════════════════════════════════════════════════
// STDP (Song, Miller, Abbott, Nature Neuroscience 2000)
// ═══════════════════════════════════════════════════════════

float compute_stdp(
    const Pathway& path,
    const FeatureSig& query,
    const StdParams& params) noexcept
{
    // Cosine similarity between pathway's learned features and query
    float dot = 0.0f, norm_p = 0.0f, norm_q = 0.0f;
    for (size_t i = 0; i < 22; ++i) {
        dot   += path.feature_signature[i] * query[i];
        norm_p += path.feature_signature[i] * path.feature_signature[i];
        norm_q += query[i] * query[i];
    }
    float cos_sim = (norm_p > 0.0f && norm_q > 0.0f)
        ? dot / (std::sqrt(norm_p) * std::sqrt(norm_q))
        : 0.0f;

    // STDP modulation: past success amplifies similarity
    float stdp_boost = path.weight_stdp * params.A_plus
                     - (1.0f - path.success_rate) * params.A_minus;

    // BCM sliding threshold: dynamically adjusts plasticity
    float bcm_mod = path.weight_bcm > 0.5f ? 1.0f + path.weight_bcm : 1.0f;

    return cos_sim * (1.0f + stdp_boost) * bcm_mod;
}

// ═══════════════════════════════════════════════════════════
// Top-K Gating (Switch Transformer, Fedus et al. 2022)
// ═══════════════════════════════════════════════════════════

GatingResult top_k_gate(
    const std::unordered_map<std::string, float>& affinities,
    int k,
    float capacity_factor)
{
    GatingResult result;

    if (affinities.empty()) return result;

    // Sort models by affinity score (descending)
    std::vector<std::pair<std::string, float>> ranked(
        affinities.begin(), affinities.end());
    std::sort(ranked.begin(), ranked.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Apply capacity factor: (tokens_per_expert) = (tokens / num_experts) * capacity_factor
    float total_affinity = std::accumulate(ranked.begin(), ranked.end(), 0.0f,
        [](float sum, const auto& p) { return sum + p.second; });

    // Softmax over top-k
    k = std::min(k, static_cast<int>(ranked.size()));
    float top_sum = 0.0f;
    for (int i = 0; i < k; ++i) top_sum += std::exp(ranked[i].second);

    for (int i = 0; i < k; ++i) {
        result.selected_models.push_back(ranked[i].first);
        result.gate_weights.push_back(std::exp(ranked[i].second) / top_sum);
    }

    // Load balance auxiliary loss (encourages uniform expert usage)
    float load_balance = 0.0f;
    for (const auto& [model, score] : affinities) {
        float frac = score / (total_affinity + 1e-8f);
        if (frac > 0.0f) load_balance += frac * std::log(frac);
    }
    result.load_balance = -load_balance;  // entropy: higher = more balanced

    return result;
}

// ═══════════════════════════════════════════════════════════
// Weighted Round Robin (Apache HTTPd / Nginx pattern)
// ═══════════════════════════════════════════════════════════

void WeightedRoundRobin::update_weights(
    const std::unordered_map<std::string, float>& weights)
{
    std::unique_lock lock(mtx_);
    slots_.clear();
    for (auto& [model, weight] : weights) {
        slots_.push_back({model, weight});
    }
}

std::string WeightedRoundRobin::next() {
    std::shared_lock lock(mtx_);
    if (slots_.empty()) return "";

    // Find slot with highest accumulated weight (WRR algorithm)
    float max_acc = -std::numeric_limits<float>::infinity();
    std::string selected;

    for (auto& slot : slots_) {
        slot.accumulated += 1.0f;  // increment per round
        if (slot.accumulated > max_acc) {
            max_acc = slot.accumulated;
            selected = slot.model;
        }
    }

    // Subtract total weight from selected (WRR reset)
    float total_weight = 0.0f;
    for (auto& slot : slots_) total_weight += 1.0f;
    for (auto& slot : slots_) {
        if (slot.model == selected)
            slot.accumulated -= total_weight;
    }

    return selected;
}

// ═══════════════════════════════════════════════════════════
// Router Engine — Main Routing Logic
// ═══════════════════════════════════════════════════════════

RoutingDecision RouterEngine::route(
    const FeatureSig& query,
    const TaskDims& dims,
    const std::string& primary_category)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    RoutingDecision decision;
    decision.action = Action::Single;
    decision.region = "prefrontal_cortex";  // default

    // Step 1: Find candidate pathways for this task category
    auto candidates = routing_table_.find_by_category(primary_category);

    total_lookups_.fetch_add(1);

    if (candidates.empty()) {
        // Fallback: try general category
        candidates = routing_table_.find_by_category("general");
    }

    if (candidates.empty()) {
        // Cold start: no learned routes → use default DS-PRO
        decision.model = "ds-pro";
        decision.confidence = 0.5f;
        cache_hits_.fetch_add(1);  // fast path counts as cache hit
        goto done;
    }

    // Step 2: Compute STDP-modulated affinity scores
    {
        std::unordered_map<std::string, float> affinities;
        for (const auto* pw : candidates) {
            float score = compute_stdp(*pw, query, stdp_params_);
            affinities[pw->target_model] = std::max(
                affinities[pw->target_model], score);
        }

        // Step 3: Apply difficulty-based routing threshold
        float difficulty = dims.difficulty();

        if (difficulty < 0.3f) {
            // Easy task → single model, fast path
            auto best = std::max_element(affinities.begin(), affinities.end(),
                [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });
            decision.model = best->first;
            decision.confidence = best->second;
            decision.action = Action::Single;

        } else if (difficulty < 0.6f) {
            // Medium task → pipeline (sequential stages)
            auto gated = top_k_gate(affinities, 2);
            decision.models = gated.selected_models;
            decision.confidence = gated.gate_weights.empty()
                ? 0.0f : gated.gate_weights[0];
            decision.action = Action::Pipeline;

        } else {
            // Hard task → full collaboration (parallel multi-model)
            auto gated = top_k_gate(affinities, 3);
            decision.models = gated.selected_models;
            decision.confidence = gated.gate_weights.empty()
                ? 0.0f : gated.gate_weights[0];
            decision.action = Action::Collab;
        }
    }

done:
    auto t1 = std::chrono::high_resolution_clock::now();
    decision.latency_ns = static_cast<float>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    total_latency_ns_.fetch_add(static_cast<uint64_t>(decision.latency_ns));

    return decision;
}

void RouterEngine::learn_from_outcome(
    const std::string& pathway_id,
    bool success,
    double latency_ms)
{
    if (success) {
        routing_table_.record_success(pathway_id, latency_ms);
    } else {
        routing_table_.record_failure(pathway_id);
    }

    // Update WRR scheduler weights
    auto* pw = routing_table_.find_pathway(pathway_id);
    if (pw) {
        std::unordered_map<std::string, float> weights;
        weights[pw->target_model] = pw->success_rate;
        wrr_scheduler_.update_weights(weights);
    }
}

void RouterEngine::load_pathways(const std::vector<Pathway>& pathways) {
    for (const auto& pw : pathways) {
        routing_table_.upsert_pathway(pw);
        std::unordered_map<std::string, float> weights;
        weights[pw.target_model] = pw.success_rate;
        wrr_scheduler_.update_weights(weights);
    }
}

std::vector<Pathway> RouterEngine::dump_pathways() const {
    (void)this;  // implementation via RoutingTable friend access in bindings
    return {};
}

} // namespace synapse
