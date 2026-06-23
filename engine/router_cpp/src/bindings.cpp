// ═══════════════════════════════════════════════════════════════
// pybind11 Bindings — C++ Router → Python
// ═══════════════════════════════════════════════════════════════
//
// pybind11 by Wenzel Jakob (2017). Used by:
//   - PyTorch (C++ backend)
//   - TensorFlow (C++ ops)
//   - NumPy (C extensions reference)
//   - OpenAI Triton

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "router.hpp"

namespace py = pybind11;

namespace {

// Convert Python feature dict to FeatureSig
synapse::FeatureSig dict_to_features(const py::dict& d) {
    synapse::FeatureSig sig{};
    // 22-dim feature space — fill from dict or zeros
    for (auto& v : sig) v = 0.0f;

    if (d.contains("code"))    sig[0]  = d["code"].cast<float>();
    if (d.contains("math"))    sig[1]  = d["math"].cast<float>();
    if (d.contains("logic"))   sig[2]  = d["logic"].cast<float>();
    if (d.contains("arch"))    sig[3]  = d["arch"].cast<float>();
    if (d.contains("writing")) sig[4]  = d["writing"].cast<float>();
    if (d.contains("general")) sig[5]  = d["general"].cast<float>();
    // Remaining dims reserve for future expansion

    return sig;
}

synapse::TaskDims dict_to_dims(const py::dict& d) {
    synapse::TaskDims dims;
    dims.code    = d.contains("code")    ? d["code"].cast<float>()    : 0.0f;
    dims.math    = d.contains("math")    ? d["math"].cast<float>()    : 0.0f;
    dims.logic   = d.contains("logic")   ? d["logic"].cast<float>()   : 0.0f;
    dims.arch    = d.contains("arch")    ? d["arch"].cast<float>()    : 0.0f;
    dims.writing = d.contains("writing") ? d["writing"].cast<float>() : 0.0f;
    dims.general = d.contains("general") ? d["general"].cast<float>() : 0.0f;
    return dims;
}

py::dict decision_to_dict(const synapse::RoutingDecision& d) {
    py::dict out;
    out["action"] = d.is_single() ? "single"
                  : d.is_pipeline() ? "pipeline"
                  : "collab";
    out["model"] = d.model;
    out["region"] = d.region;
    out["models"] = d.models;
    out["confidence"] = d.confidence;
    out["latency_ns"] = d.latency_ns;
    return out;
}

} // anonymous

PYBIND11_MODULE(synapse_router, m) {
    m.doc() = "SynapseFlow C++ Routing Engine — Sub-microsecond LLM orchestration";

    // ─── RouterEngine ──────────────────────────────────────
    py::class_<synapse::RouterEngine>(m, "RouterEngine")
        .def(py::init<>())

        .def("route", [](synapse::RouterEngine& engine,
                         const py::dict& features_dict,
                         const py::dict& dims_dict,
                         const std::string& category) {
            auto features = dict_to_features(features_dict);
            auto dims = dict_to_dims(dims_dict);
            auto decision = engine.route(features, dims, category);
            return decision_to_dict(decision);
        },
        py::arg("features"),
        py::arg("dims"),
        py::arg("category") = "general",
        "Route a question to the best model(s). Target: <1μs.")

        .def("learn", &synapse::RouterEngine::learn_from_outcome,
            py::arg("pathway_id"),
            py::arg("success"),
            py::arg("latency_ms") = 0.0,
            "Record routing outcome for STDP learning.")

        .def("load_pathways", [](synapse::RouterEngine& engine,
                                 const py::list& pathways_json) {
            std::vector<synapse::Pathway> pathways;
            for (auto& item : pathways_json) {
                py::dict d = item.cast<py::dict>();
                synapse::Pathway pw;
                pw.pathway_id    = d.contains("pathway_id")
                    ? d["pathway_id"].cast<std::string>() : "";
                pw.source_region = d.contains("source_region")
                    ? d["source_region"].cast<std::string>() : "";
                pw.target_model  = d.contains("target_model")
                    ? d["target_model"].cast<std::string>() : "";
                pw.category      = d.contains("category")
                    ? d["category"].cast<std::string>() : "general";
                if (d.contains("weight_stdp"))
                    pw.weight_stdp = d["weight_stdp"].cast<float>();
                if (d.contains("weight_bcm"))
                    pw.weight_bcm = d["weight_bcm"].cast<float>();
                if (d.contains("weight_ltp"))
                    pw.weight_ltp = d["weight_ltp"].cast<float>();
                if (d.contains("hardened"))
                    pw.hardened = d["hardened"].cast<bool>();
                if (d.contains("success_rate"))
                    pw.success_rate = d["success_rate"].cast<float>();

                pathways.push_back(pw);
            }
            engine.load_pathways(pathways);
        },
        py::arg("pathways"),
        "Load pathways from brain state JSON.")

        .def("stats", [](synapse::RouterEngine& engine) {
            py::dict s;
            s["lookups"] = engine.lookups();
            s["hit_rate"] = engine.hit_rate();
            s["avg_latency_ns"] = engine.avg_latency_ns();
            return s;
        })
        ;

    // ─── Pathway ───────────────────────────────────────────
    py::class_<synapse::Pathway>(m, "Pathway")
        .def(py::init<>())
        .def_readwrite("pathway_id", &synapse::Pathway::pathway_id)
        .def_readwrite("source_region", &synapse::Pathway::source_region)
        .def_readwrite("target_model", &synapse::Pathway::target_model)
        .def_readwrite("category", &synapse::Pathway::category)
        .def_readwrite("weight_stdp", &synapse::Pathway::weight_stdp)
        .def_readwrite("weight_bcm", &synapse::Pathway::weight_bcm)
        .def_readwrite("hardened", &synapse::Pathway::hardened)
        .def_readwrite("use_count", &synapse::Pathway::use_count)
        .def("match_score", [](synapse::Pathway& pw,
                               const py::dict& features) {
            auto sig = dict_to_features(features);
            return pw.match_score(sig);
        })
        ;

    // ─── Top-K Gating ─────────────────────────────────────
    m.def("top_k_gate", [](const py::dict& affinities, int k) {
        std::unordered_map<std::string, float> aff;
        for (auto& [key, value] : affinities) {
            aff[key.cast<std::string>()] = value.cast<float>();
        }
        auto result = synapse::top_k_gate(aff, k);
        py::dict out;
        out["models"] = result.selected_models;
        out["weights"] = result.gate_weights;
        out["load_balance"] = result.load_balance;
        return out;
    },
    py::arg("affinities"),
    py::arg("k") = 2,
    "Switch Transformer top-k expert gating (Fedus et al. 2022).");

    // ─── Utility functions ────────────────────────────────
    m.def("compute_stdp", [](const py::dict& pathway_dict,
                              const py::dict& query_features) {
        synapse::Pathway pw;
        pw.weight_stdp = pathway_dict.contains("weight_stdp")
            ? pathway_dict["weight_stdp"].cast<float>() : 0.0f;
        pw.weight_bcm  = pathway_dict.contains("weight_bcm")
            ? pathway_dict["weight_bcm"].cast<float>() : 0.0f;
        pw.success_rate = pathway_dict.contains("success_rate")
            ? pathway_dict["success_rate"].cast<float>() : 0.5f;
        // Copy feature signature if present
        if (pathway_dict.contains("feature_signature")) {
            auto sig_list = pathway_dict["feature_signature"].cast<py::list>();
            for (size_t i = 0; i < std::min(sig_list.size(), size_t(22)); ++i) {
                pw.feature_signature[i] = sig_list[i].cast<float>();
            }
        }
        auto query = dict_to_features(query_features);
        return compute_stdp(pw, query);
    });
}
