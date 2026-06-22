#include "pathway.hpp"
#include <cmath>

namespace synapse {

float Pathway::match_score(const FeatureSig& query) const noexcept {
    // Cosine similarity
    float dot = 0.0f, np = 0.0f, nq = 0.0f;
    for (size_t i = 0; i < 22; ++i) {
        dot += feature_signature[i] * query[i];
        np  += feature_signature[i] * feature_signature[i];
        nq  += query[i] * query[i];
    }
    if (np <= 0.0f || nq <= 0.0f) return 0.0f;
    float sim = dot / (std::sqrt(np) * std::sqrt(nq));

    // Modulate by learned weights
    return sim * (1.0f + weight_stdp) * (1.0f + weight_ltp);
}

} // namespace synapse
