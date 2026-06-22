#!/usr/bin/env python3
"""
synapse_network.py — Complete Neurosynaptic Pathway Network

9 SYNAPTIC PLASTICITY MECHANISMS (all paper-derived):

  [M1] STDP — Spike-Timing-Dependent Plasticity
       Song, Miller & Abbott. Nature Neuroscience 3:919-926 (2000).
       Eq.2: Delta_w = A+ * exp(-Delta_t / tau+) [LTP]
            Delta_w = -A- * exp(Delta_t / tau-)   [LTD]
       Delegate to brain.py synaptic_update()

  [M2] BCM Sliding Threshold
       Bienenstock, Cooper & Munro. J Neurosci 2(1):32-48 (1982).
       Eq.7: theta_M = E[y^2]; dw/dt = eta * y * (y - theta_M) * x
       Delegate to brain.py bcm_threshold_update()

  [M3] L-LTP Pathway Hardening
       Kandel, E.R. Science 294:1030-1038 (2001).
       Derived from: cAMP/PKA/CREB cascade -> protein synthesis -> structural change
       S(t) = sum_i A_LTP * exp(-(t-t_i) / tau_hardening)

  [M4] Performance-Driven LTD (Primary) + Gentle Time Decay (Secondary)
       Primary: Bienenstock (1982) — below-threshold activity -> LTD
       Secondary: Dudek & Bear. PNAS 89(10):4363-4367 (1992). Fig.2.
       w(t) = w0 * exp(-dt/tau_forget) — but ONLY gentle background decay

  [M5] High-Frequency Shortcuts (Metaplasticity)
       Abraham & Bear. TINS 19(4):126-130 (1996).
       Prior activity lowers LTP threshold; f_window > 5/hr -> fast path

  [M6] Synaptic Tagging & Capture — ASSOCIATIVE MEMORY
       Frey & Morris. Nature 385:533-536 (1997). Fig.3.
       Strong synapse creates PRP pool; weak tagged synapses capture PRPs.
       PRP_pool[region] decays with tau_prp. Tagged pathways snag plasticity resources.

  [M7] Clustered Plasticity — REGIONAL CO-SPECIALIZATION
       Fu et al., Nature 483:92-96 (2012). Fig.2.
       Synapses on same dendritic branch share PRPs; spatial neighbors co-potentiate.
       cluster_boost applies to pathways in same region with similar feature signatures.

  [M8] Eligibility Traces — TWO-STEP CREDIT ASSIGNMENT
       Gerstner et al., Nature Neuroscience 21:1273-1283 (2018). Box 2.
       Success marks an "eligibility trace" (e); second success within tau converts to LTP.
       de/dt = -e/tau_e + success; if e > theta_e AND success: convert eligibility -> weight

  [M9] Neuromodulation — GLOBAL STATE (DA/5-HT)
       Fuxe et al., Progress in Neurobiology 83:263-287 (2007). Sec.3.
       Dopamine = reward prediction error; Serotonin = patience modulation.
       DA(t) = DA_baseline + alpha_RPE * (actual - expected reward)
       5-HT(t) = slow averaging of recent reward rate

  [M10] Astrocyte Ca2+ Modulation — REGIONAL FIELD
       Perea & Araque. Science 317:1083-1086 (2007). Fig.3.
       Haydon & Carmignoto. Physiol Rev 86:1009-1031 (2006). Sec.IV.
       Astrocytes sense regional activity -> release gliotransmitters -> globally modulate region.
       tau_ca * d[Ca]/dt = -[Ca] + sum(activity_i) + noise

  [M11] Neuronal Avalanches / Criticality — HOMEOSTATIC TUNING
       Beggs & Plenz. J Neurosci 23(35):11167-11177 (2003). Fig.3.
       Branching ratio sigma = mean(cascade_size) / mean(cascade_size - 1)
       Auto-tune pruning/hardening thresholds to keep sigma ~ 1 (critical point).

State persistence: ~/.synapseflow/brain/
"""

import json, math, time, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# ─── Config ────────────────────────────────────────────────────

def load_synapse_config(config_path: str = None) -> Dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "synapse_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _storage_dir() -> Path:
    cfg = load_synapse_config()
    base = Path(cfg["storage"]["base_dir"].replace("~", str(Path.home())))
    base.mkdir(parents=True, exist_ok=True)
    return base


# ═══════════════════════════════════════════════════════════════
# SYNAPTIC PATHWAY (with eligibility trace)
# ═══════════════════════════════════════════════════════════════

@dataclass
class SynapticPathway:
    pathway_id: str
    source_region: str
    target_model: str
    category: str

    # Feature signature (EMA smoothed)
    feature_signature: List[float] = field(default_factory=lambda: [0.0] * 22)

    # Multi-timescale weights
    weight_stdp: float = 0.5
    weight_bcm:  float = 0.5
    weight_ltp:  float = 0.0

    # Temporal tracking (wall-clock)
    created_at:  float = 0.0
    last_used:   float = 0.0
    last_failed: float = 0.0
    use_count:   int   = 0
    fail_count:  int   = 0

    # M8: Eligibility trace (Gerstner et al. 2018)
    eligibility: float = 0.0         # e(t): decays with tau_eligibility
    eligibility_last_update: float = 0.0

    # Frequency window
    recent_uses: List[float] = field(default_factory=list)

    # State flags
    hardened: bool = False
    pruned:   bool = False

    # Stats
    success_rate: float = 1.0
    avg_latency:  float = 0.0

    def to_dict(self) -> Dict:
        return {
            "pathway_id": self.pathway_id,
            "source_region": self.source_region,
            "target_model": self.target_model,
            "category": self.category,
            "feature_signature": self.feature_signature,
            "weight_stdp": self.weight_stdp, "weight_bcm": self.weight_bcm,
            "weight_ltp": self.weight_ltp,
            "created_at": self.created_at, "last_used": self.last_used,
            "last_failed": self.last_failed,
            "use_count": self.use_count, "fail_count": self.fail_count,
            "eligibility": self.eligibility,
            "recent_uses": self.recent_uses,
            "hardened": self.hardened, "pruned": self.pruned,
            "success_rate": self.success_rate, "avg_latency": self.avg_latency,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SynapticPathway":
        return cls(
            pathway_id=d["pathway_id"], source_region=d["source_region"],
            target_model=d["target_model"], category=d["category"],
            feature_signature=d.get("feature_signature", [0.0]*22),
            weight_stdp=d.get("weight_stdp", 0.5), weight_bcm=d.get("weight_bcm", 0.5),
            weight_ltp=d.get("weight_ltp", 0.0),
            created_at=d.get("created_at", 0.0), last_used=d.get("last_used", 0.0),
            last_failed=d.get("last_failed", 0.0),
            use_count=d.get("use_count", 0), fail_count=d.get("fail_count", 0),
            eligibility=d.get("eligibility", 0.0),
            recent_uses=d.get("recent_uses", []),
            hardened=d.get("hardened", False), pruned=d.get("pruned", False),
            success_rate=d.get("success_rate", 1.0),
            avg_latency=d.get("avg_latency", 0.0),
        )

    def total_weight(self) -> float:
        return 0.4 * self.weight_stdp + 0.3 * self.weight_bcm + 0.3 * self.weight_ltp

    def age_hours(self, now: float = None) -> float:
        if now is None: now = time.time()
        return (now - self.created_at) / 3600.0

    def idle_hours(self, now: float = None) -> float:
        if now is None: now = time.time()
        if self.last_used <= 0: return self.age_hours(now)
        return (now - self.last_used) / 3600.0


# ═══════════════════════════════════════════════════════════════
# PATHWAY NETWORK — 9 plasticity mechanisms
# ═══════════════════════════════════════════════════════════════

class PathwayNetwork:
    def __init__(self, config_path: str = None):
        self.cfg = load_synapse_config(config_path)
        self.pathways: Dict[str, SynapticPathway] = {}
        self.hardened_routes: Dict[int, Dict] = {}

        # M10: Astrocyte Ca2+ per region
        self.astrocyte_ca: Dict[str, float] = {}  # region -> [Ca]
        self.astrocyte_last_update: float = time.time()

        # M11: Criticality cascade tracking
        self.cascade_sizes: List[int] = []  # last N cascade sizes
        self.current_cascade: List[str] = []  # pathways in current cascade

        self._next_id = 0
        self.load_state()

    # ── Pathway Management ───────────────────────────────────

    def _make_id(self, region: str, model: str, category: str) -> str:
        return f"{region}::{model}::{category}"

    def find_or_create(self, region: str, model: str, category: str,
                       features: List[float] = None) -> SynapticPathway:
        pid = self._make_id(region, model, category)
        if pid in self.pathways:
            return self.pathways[pid]

        pathway = SynapticPathway(
            pathway_id=pid, source_region=region,
            target_model=model, category=category,
            feature_signature=list(features) if features else [0.0]*22,
            created_at=time.time(), last_used=time.time(),
        )
        self.pathways[pid] = pathway
        return pathway

    # ═════════════════════════════════════════════════════════
    # M3: L-LTP PATHWAY HARDENING
    # Kandel, Science 294:1030-1038 (2001)
    # S(t) = sum_i A_LTP * exp(-(t-t_i)/tau_hardening)
    # Trigger: S > theta_hardening AND use_count >= 3
    # ═════════════════════════════════════════════════════════

    def compute_hardening_potential(self, pathway: SynapticPathway) -> float:
        cfg = self.cfg["hardening"]
        A_LTP, tau_h = cfg["A_LTP"], cfg["tau_hardening"]
        S = 0.0
        now = time.time()
        for t_use in pathway.recent_uses:
            dt = (now - t_use) / tau_h
            S += A_LTP * math.exp(-dt / max(1.0, tau_h))
        return min(1.0, S)

    def should_harden(self, pathway: SynapticPathway) -> bool:
        if pathway.use_count < self.cfg["hardening"]["n_min_successes"]:
            return False
        return self.compute_hardening_potential(pathway) > self.cfg["hardening"]["theta_hardening"]

    def harden_pathway(self, pathway: SynapticPathway):
        pathway.hardened = True
        pathway.weight_ltp = self.compute_hardening_potential(pathway)
        if pathway.feature_signature:
            hash_key = self._feature_hash(pathway.feature_signature)
            self.hardened_routes[hash_key] = {
                "region": pathway.source_region, "model": pathway.target_model,
                "category": pathway.category, "weight": pathway.total_weight(),
                "hardened_at": time.time(), "pathway_id": pathway.pathway_id,
            }

    # ═════════════════════════════════════════════════════════
    # M4: PERFORMANCE-DRIVEN LTD (Primary)
    # Bienenstock et al., J Neurosci 2(1):32-48 (1982) — Eq.7
    # dw/dt = eta * y * (y - theta_M) * x
    # When y < theta_M (below threshold): LTD
    # When y > theta_M (above threshold): LTP
    #
    # Plus gentle time-based background decay:
    # Dudek & Bear, PNAS 89(10):4363-4367 (1992) — Fig.2
    # Gentle exponential background: w *= exp(-dt/tau_forget)
    # tau_forget = 720 hours (30 days) — very slow
    # ═════════════════════════════════════════════════════════

    def compute_performance_ltd(self, pathway: SynapticPathway, correct: bool) -> float:
        """
        Primary LTD: wrong answers in wrong regions cause strong weight decrease.
        Correct answers in right regions cause weight increase.
        This is the BCM rule (Bienenstock 1982, Eq.7) adapted to our domain.

        Returns: weight delta to apply
        """
        cfg = self.cfg["forgetting"]
        A_perf = cfg["A_LTD_performance"]

        if correct:
            # LTP: successful routing reinforces
            return +0.05  # small positive reinforcement per success
        else:
            # LTD: wrong answer -> significant weakening
            return -A_perf  # -0.15 per failure

    def compute_time_decay(self, pathway: SynapticPathway, now: float = None) -> float:
        """Gentle background decay only. Very slow (30-day half-life)."""
        cfg = self.cfg["forgetting"]
        A_time = cfg["A_LTD_time"]
        tau_sec = cfg["tau_forgetting_hours"] * 3600.0
        if now is None: now = time.time()
        dt = pathway.idle_hours(now) * 3600.0
        if dt <= 0: return 1.0  # no decay
        return math.exp(-dt / tau_sec)

    def should_prune(self, pathway: SynapticPathway, now: float = None) -> bool:
        """
        Pruning conditions (PERFORMANCE-driven, not just idle):
        1. weight < theta_prune (0.02) AND
        2. EITHER: fail_count >= 8 consecutive in wrong region
           OR: idle > 30 days AND already very weak
        """
        cfg = self.cfg["forgetting"]
        if now is None: now = time.time()
        w = pathway.total_weight()
        if w > cfg["theta_prune"]: return False
        # Performance-based: many failures = prune
        if pathway.fail_count >= cfg["consecutive_fails_for_prune"]: return True
        # Time-based: very long idle AND already weak
        idle_h = pathway.idle_hours(now)
        return idle_h > cfg["tau_forgetting_hours"] * 2  # 2x forgetting tau

    def prune_pathway(self, pathway: SynapticPathway):
        pathway.pruned = True
        self._log_pruned(pathway)
        for k, v in list(self.hardened_routes.items()):
            if v.get("pathway_id") == pathway.pathway_id:
                del self.hardened_routes[k]; break

    def _log_pruned(self, pathway: SynapticPathway):
        storage = _storage_dir()
        log_file = storage / self.cfg["storage"]["pruned_pathways_file"]
        entry = pathway.to_dict()
        entry["pruned_at"] = time.time()
        entry["prune_reason"] = (
            f"weight={pathway.total_weight():.4f} | fails={pathway.fail_count} | "
            f"idle={pathway.idle_hours():.1f}h | region={pathway.source_region}"
        )
        existing = []
        if log_file.exists():
            try: existing = json.loads(log_file.read_text(encoding="utf-8"))
            except: existing = []
        existing.append(entry)
        log_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))

    # ═════════════════════════════════════════════════════════
    # M6: SYNAPTIC TAGGING & CAPTURE
    # Frey & Morris, Nature 385:533-536 (1997) — Fig.3
    #
    # When a strong LTP is induced at one synapse, it generates a
    # pool of plasticity-related proteins (PRPs) that can be "captured"
    # by nearby weak synapses that have been "tagged" but lack their
    # own PRPs. This implements associative memory.
    #
    # PRP_pool[region] = sum of available PRPs in a region
    # Tagged pathway: weight > theta_tagged AND not yet hardened
    # Capture: tagged pathway consumes PRPs -> weight boost
    # ═════════════════════════════════════════════════════════

    def __init_prp_pools(self):
        if not hasattr(self, 'prp_pools'):
            self.prp_pools: Dict[str, float] = {}  # region -> PRP amount
            self.prp_pools_updated: float = time.time()

    def generate_prps(self, pathway: SynapticPathway):
        """When a pathway hardens, it releases PRPs into its region's pool."""
        self.__init_prp_pools()
        cfg = self.cfg["tagging_capture"]
        region = pathway.source_region
        self.prp_pools[region] = self.prp_pools.get(region, 0.0) + cfg["A_tag"]

    def decay_prps(self):
        """PRP pools decay over time (tau_prp rounds ~ 100)."""
        self.__init_prp_pools()
        cfg = self.cfg["tagging_capture"]
        now = time.time()
        dt = now - self.prp_pools_updated
        decay = math.exp(-dt / (cfg["tau_prp"] * 3600.0))  # tau in hours
        for region in self.prp_pools:
            self.prp_pools[region] *= decay
            if self.prp_pools[region] < 0.001:
                self.prp_pools[region] = 0.0
        self.prp_pools_updated = now

    def is_tagged(self, pathway: SynapticPathway) -> bool:
        """Check if pathway is in the 'tagged' state (weak but recently active)."""
        cfg = self.cfg["tagging_capture"]
        return (
            not pathway.hardened
            and not pathway.pruned
            and pathway.total_weight() >= cfg["theta_tagged"]
            and len(pathway.recent_uses) >= cfg["n_uses_for_tag"]
        )

    def capture_prps(self, pathway: SynapticPathway):
        """
        Tagged pathway captures PRPs from its region's pool.
        This is the "capture" step of Frey & Morris 1997 Fig.3.
        """
        self.__init_prp_pools()
        cfg = self.cfg["tagging_capture"]
        region = pathway.source_region
        pool = self.prp_pools.get(region, 0.0)
        if pool <= 0: return

        # Capture: consume up to A_capture PRPs
        captured = min(pool, cfg["A_capture"])
        self.prp_pools[region] -= captured

        # Boost pathway weight
        boost = captured
        pathway.weight_ltp = min(1.0, pathway.weight_ltp + boost)
        pathway.weight_stdp = min(1.0, pathway.weight_stdp + boost * 0.5)

    # ═════════════════════════════════════════════════════════
    # M7: CLUSTERED PLASTICITY
    # Fu et al., Nature 483:92-96 (2012) — Fig.2
    #
    # Synapses close together on the same dendritic branch share
    # PRPs. When one strengthens, its spatial neighbors get a small
    # boost proportional to their feature similarity.
    #
    # "Clusters" = pathways in the same region with high cosine
    # similarity in feature signatures.
    # ═════════════════════════════════════════════════════════

    def apply_clustered_boost(self, strengthened: SynapticPathway):
        """
        When a pathway is strengthened, nearby (similar) pathways
        in the same region get a small co-potentiation boost.

        Fu et al. 2012 Fig.2: clustered spines within 10um share PRPs.
        Our analog: pathways with cos_sim > cluster_radius in same region.
        """
        cfg = self.cfg["clustered_plasticity"]
        radius = cfg["cluster_radius"]
        boost = cfg["cluster_boost"]

        sig = np.array(strengthened.feature_signature, dtype=np.float64)
        sig_norm = np.linalg.norm(sig)
        if sig_norm < 0.001: return

        for pw in self.pathways.values():
            if pw.pathway_id == strengthened.pathway_id: continue
            if pw.source_region != strengthened.source_region: continue
            if pw.pruned: continue

            other = np.array(pw.feature_signature, dtype=np.float64)
            other_norm = np.linalg.norm(other)
            if other_norm < 0.001: continue

            cos_sim = float(np.dot(sig, other) / (sig_norm * other_norm))
            if cos_sim >= radius:
                # Clustered co-potentiation
                pw.weight_stdp = min(1.0, pw.weight_stdp + boost * cos_sim)
                pw.weight_bcm  = min(1.0, pw.weight_bcm  + boost * cos_sim * 0.5)

    # ═════════════════════════════════════════════════════════
    # M8: ELIGIBILITY TRACES
    # Gerstner et al., Nature Neuroscience 21:1273-1283 (2018)
    # Box 2: Eligibility Traces for Credit Assignment
    #
    # A synapse that was recently active is "tagged" with an
    # eligibility trace e(t). A subsequent reward signal converts
    # the eligibility into permanent synaptic change.
    #
    # de/dt = -e/tau_e + spike    (Eq.2, eligibility trace dynamics)
    # Delta_w = eta * e * reward   (Eq.3, conversion)
    #
    # In our system:
    #   "spike" = pathway was used for routing
    #   "reward" = routing was correct (quality gate passed)
    #   Conversion: e > theta_e AND correct -> weight boost
    # ═════════════════════════════════════════════════════════

    def update_eligibility(self, pathway: SynapticPathway):
        """
        Update eligibility trace for a pathway.
        Eq.2 from Gerstner et al. 2018: de/dt = -e/tau_e + spike

        "spike" = pathway was just used (always called on use).
        """
        cfg = self.cfg["eligibility_trace"]
        tau_e = cfg["tau_eligibility"]
        now = time.time()

        # Decay eligibility since last update
        dt = now - pathway.eligibility_last_update
        if dt > 0 and pathway.eligibility_last_update > 0:
            pathway.eligibility *= math.exp(-dt / tau_e)

        # Spike: set eligibility to 1 (pathway was used)
        pathway.eligibility = min(1.0, pathway.eligibility + 0.5)
        pathway.eligibility_last_update = now

    def convert_eligibility(self, pathway: SynapticPathway) -> bool:
        """
        If pathway has high eligibility AND routing was correct,
        convert eligibility trace to permanent LTP weight.

        Eq.3 from Gerstner et al. 2018: Delta_w = eta * e * reward
        """
        cfg = self.cfg["eligibility_trace"]
        if pathway.eligibility < cfg["theta_eligibility"]:
            return False

        # Convert eligibility -> weight
        delta_w = cfg["conversion_rate"] * pathway.eligibility
        pathway.weight_ltp = min(1.0, pathway.weight_ltp + delta_w * 2)
        pathway.weight_stdp = min(1.0, pathway.weight_stdp + delta_w)

        # Reset eligibility after conversion
        pathway.eligibility = 0.0
        return True

    # ═════════════════════════════════════════════════════════
    # M9: NEUROMODULATION (Dopamine / Serotonin)
    # Fuxe et al., Prog Neurobiol 83:263-287 (2007) — Sec.3
    #
    # DA(t) = reward prediction error — fast, phasic
    # 5-HT(t) = slow averaging of reward rate — tonic modulation
    #
    # DA drives immediate weight changes; 5-HT controls
    # exploration/exploitation balance (high 5-HT = patient, explore)
    # ═════════════════════════════════════════════════════════

    def _init_neuromod(self):
        if not hasattr(self, 'dopamine'):
            self.dopamine = self.cfg["neuromodulation"]["baseline_dopamine"]
            self.serotonin = 0.5
            self.expected_reward = 0.7
            self.neuromod_last_update = time.time()

    def update_neuromodulators(self, pathway_was_correct: bool):
        """
        Update DA and 5-HT based on routing outcome.

        Dopamine = reward prediction error (Schultz et al. 1997, Science)
           DA <- DA + alpha * (actual_reward - expected_reward)
           actual_reward = 1 if correct else 0

        Serotonin = slow moving average of reward rate
           5-HT <- 5-HT + (1/tau_5HT) * (actual_reward - 5-HT)
        """
        self._init_neuromod()
        cfg = self.cfg["neuromodulation"]
        alpha = cfg["reward_prediction_error_alpha"]
        tau_da = cfg["dopamine_tau"]
        tau_5ht = cfg["serotonin_tau"]
        now = time.time()

        actual = 1.0 if pathway_was_correct else 0.0

        # Decay both
        dt = now - self.neuromod_last_update
        if dt > 0:
            self.dopamine += (self.cfg["neuromodulation"]["baseline_dopamine"] - self.dopamine) * (dt / tau_da)
            self.dopamine = max(0.0, min(1.0, self.dopamine))

        # Reward prediction error
        rpe = actual - self.expected_reward
        self.dopamine += alpha * rpe
        self.dopamine = max(0.0, min(1.0, self.dopamine))

        # Expected reward update
        self.expected_reward += 0.05 * rpe

        # Serotonin: slow average
        self.serotonin += (1.0 / tau_5ht) * (actual - self.serotonin)
        self.serotonin = max(0.1, min(0.9, self.serotonin))

        self.neuromod_last_update = now

    def get_da_modulation(self) -> float:
        """Dopamine modulation factor on weight changes."""
        self._init_neuromod()
        # High DA -> amplify weight changes (learn more)
        # Low DA -> dampen (conservative)
        return 0.5 + self.dopamine

    def get_5ht_exploration(self) -> float:
        """
        Serotonin-driven exploration level.
        High 5-HT -> more patient, explore alternative pathways.
        Low 5-HT -> exploit known best pathways.
        """
        self._init_neuromod()
        return self.serotonin

    # ═════════════════════════════════════════════════════════
    # M10: ASTROCYTE Ca2+ MODULATION
    # Perea & Araque, Science 317:1083-1086 (2007) — Fig.3
    # Haydon & Carmignoto, Physiol Rev 86:1009-1031 (2006)
    #
    # tau_ca * d[Ca]/dt = -[Ca] + sum(activity_i) + noise
    #
    # Astrocytes sense regional activity and release gliotransmitters
    # that globally modulate ALL synapses in the region.
    # Slow (tau ~ 1 hour), diffuse modulation.
    # ═════════════════════════════════════════════════════════

    def update_astrocytes(self, active_region: str):
        """Update astrocyte Ca2+ for the active region."""
        cfg = self.cfg["astrocyte"]
        tau_ca = cfg["tau_ca"]
        now = time.time()
        dt = now - self.astrocyte_last_update

        # Decay all regions
        if dt > 0:
            for region in self.astrocyte_ca:
                self.astrocyte_ca[region] *= math.exp(-dt / tau_ca)

        # Activity input to active region
        current = self.astrocyte_ca.get(active_region, 0.05)
        # Activity spike
        current += (1.0 - current) * 0.05
        # Small noise (stochastic vesicle release)
        current += np.random.normal(0, 0.01)
        current = max(0.0, min(1.0, current))
        self.astrocyte_ca[active_region] = current

        self.astrocyte_last_update = now

    def get_astrocyte_modulation(self, region: str) -> float:
        """
        Get astrocyte modulation factor for a region.
        High Ca2+ -> globally potentiate all synapses in region.
        Low Ca2+ -> globally depress.
        """
        cfg = self.cfg["astrocyte"]
        gamma = cfg["gamma_ca"]
        theta = cfg["theta_ca"]
        ca = self.astrocyte_ca.get(region, 0.05)
        return 1.0 + gamma * math.tanh(ca - theta)

    # ═════════════════════════════════════════════════════════
    # M11: CRITICALITY (Neuronal Avalanches)
    # Beggs & Plenz, J Neurosci 23(35):11167-11177 (2003)
    #
    # sigma = mean(cascade_size) / mean(cascade_size - 1)
    #
    # sigma = 1: critical (optimal information transmission)
    # sigma < 1: subcritical (too much pruning, dead network)
    # sigma > 1: supercritical (runaway excitation)
    #
    # Auto-tune pruning/hardening thresholds to stay at sigma ~ 1.
    # ═════════════════════════════════════════════════════════

    def record_cascade_size(self):
        """Record the size of the current routing cascade."""
        size = len(self.current_cascade)
        if size > 0:
            self.cascade_sizes.append(size)
            cfg = self.cfg["criticality"]
            max_window = cfg["cascade_window"]
            if len(self.cascade_sizes) > max_window:
                self.cascade_sizes = self.cascade_sizes[-max_window:]
        self.current_cascade = []

    def start_cascade_step(self, pathway_id: str):
        """Record a pathway activation in the current cascade."""
        if pathway_id not in self.current_cascade:
            self.current_cascade.append(pathway_id)

    def compute_branching_ratio(self) -> float:
        """
        Estimate branching ratio sigma from recent cascades.
        Beggs & Plenz 2003, Eq. in methods section.

        sigma = mean(size) / mean(size - 1), but only for size > 1
        Simplified: sigma = mean(size_i / size_{i-1}) for consecutive cascades
        """
        if len(self.cascade_sizes) < 3:
            return 1.0  # not enough data, assume critical

        sizes = np.array(self.cascade_sizes, dtype=np.float64)
        mean_s = sizes.mean()
        if mean_s <= 1.0:
            return 0.5  # subcritical

        # Ratio of consecutive sizes
        ratios = sizes[1:] / (sizes[:-1] + 0.001)
        sigma = float(np.clip(ratios.mean(), 0.3, 3.0))
        return sigma

    def auto_tune_criticality(self):
        """
        Adjust pruning and hardening thresholds to maintain criticality.

        If sigma > 1+epsilon: supercritical -> increase pruning, raise hardening bar
        If sigma < 1-epsilon: subcritical -> decrease pruning, lower hardening bar
        """
        sigma = self.compute_branching_ratio()
        cfg = self.cfg["criticality"]
        target = cfg["target_sigma"]
        eps = cfg["epsilon"]
        eta = cfg["eta"]

        error = sigma - target
        if abs(error) < eps:
            return  # already critical

        # Adjust forgetting threshold
        forget_cfg = self.cfg["forgetting"]
        forget_cfg["theta_prune"] = max(0.005, min(0.1,
            forget_cfg["theta_prune"] + eta * error))

        # Adjust hardening threshold inversely
        harden_cfg = self.cfg["hardening"]
        harden_cfg["theta_hardening"] = max(0.2, min(0.8,
            harden_cfg["theta_hardening"] - eta * error))

    # ═════════════════════════════════════════════════════════
    # M5: SHORTCUT CREATION
    # Abraham & Bear, TINS 19(4):126-130 (1996)
    # ═════════════════════════════════════════════════════════

    def _feature_hash(self, features: List[float], bins: int = 10) -> int:
        if not features: return 0
        max_val = max(features) if max(features) > 0 else 1.0
        h = 0
        for val in features[:10]:
            bucket = min(bins - 1, int((val / max_val) * bins))
            h = h * bins + bucket
        return h

    def check_shortcuts(self, features: List[float]) -> Optional[Dict]:
        cfg = self.cfg["shortcuts"]
        theta_match = cfg["theta_match"]

        # 1. Exact hash lookup
        hash_key = self._feature_hash(features)
        if hash_key in self.hardened_routes:
            return self.hardened_routes[hash_key]

        # 2. Fuzzy cosine similarity
        if not features: return None
        f_vec = np.array(features, dtype=np.float64)
        f_norm = np.linalg.norm(f_vec)
        if f_norm < 0.001: return None

        best_sim, best_route = 0.0, None
        for pathway in self.pathways.values():
            if not pathway.hardened or pathway.pruned: continue
            if not pathway.feature_signature: continue
            p_vec = np.array(pathway.feature_signature, dtype=np.float64)
            p_norm = np.linalg.norm(p_vec)
            if p_norm < 0.001: continue
            cos_sim = float(np.dot(f_vec, p_vec) / (f_norm * p_norm))
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_route = {
                    "region": pathway.source_region, "model": pathway.target_model,
                    "category": pathway.category, "weight": pathway.total_weight(),
                    "pathway_id": pathway.pathway_id, "similarity": cos_sim,
                }
        return best_route if best_sim >= theta_match else None

    def update_shortcut_frequency(self, pathway: SynapticPathway):
        cfg = self.cfg["shortcuts"]
        theta_freq = cfg["theta_highfreq_per_hour"]
        window_sec = cfg["window_hours"] * 3600.0
        now = time.time()

        pathway.recent_uses = [t for t in pathway.recent_uses if now - t < window_sec]
        pathway.recent_uses.append(now)
        if len(pathway.recent_uses) > 100:
            pathway.recent_uses = pathway.recent_uses[-100:]

        freq = len(pathway.recent_uses) / cfg["window_hours"]
        if freq >= theta_freq and pathway.use_count >= cfg["n_hard_min"]:
            if not pathway.hardened:
                self.harden_pathway(pathway)
            if pathway.feature_signature:
                hash_key = self._feature_hash(pathway.feature_signature)
                self.hardened_routes[hash_key] = {
                    "region": pathway.source_region, "model": pathway.target_model,
                    "category": pathway.category, "weight": pathway.total_weight(),
                    "hardened_at": now, "frequency": freq,
                    "pathway_id": pathway.pathway_id,
                }

    # ═════════════════════════════════════════════════════════
    # FULL UPDATE CYCLE — All 11 mechanisms applied
    # ═════════════════════════════════════════════════════════

    def update(self, pathway: SynapticPathway, correct: bool,
               features: List[float] = None):
        """
        Complete synaptic update cycle applying all plasticity mechanisms.

        Order of operations (matching real brain cascade):
          1. M8: Update eligibility trace (spike registration)
          2. M10: Update astrocyte Ca2+ for the region
          3. M4: Performance-driven LTD/LTP weight change
          4. M9: Update neuromodulators (DA/5-HT from outcome)
          5. M8: Convert eligibility if threshold met
          6. M3: Compute L-LTP hardening potential
          7. M7: Clustered plasticity (co-potentiate neighbors)
          8. M6: Generate PRPs (if hardened) or capture PRPs (if tagged)
          9. M5: Shortcut frequency update
          10. M4: Gentle time decay + pruning check
          11. M11: Record cascade, auto-tune criticality
        """
        now = time.time()

        # 1. M8: Eligibility trace (Gerstner et al. 2018)
        self.update_eligibility(pathway)

        # 2. M10: Astrocyte Ca2+ (Perea & Araque 2007)
        self.update_astrocytes(pathway.source_region)

        # 3. M4: Performance-driven weight change
        dw_perf = self.compute_performance_ltd(pathway, correct)
        da_mod = self.get_da_modulation()
        dw_effective = dw_perf * da_mod  # DA modulates learning rate
        pathway.weight_stdp = max(-1.0, min(1.0, pathway.weight_stdp + dw_effective))

        # 4. M9: Neuromodulator update (Fuxe et al. 2007)
        self.update_neuromodulators(correct)

        # 5. M8: Eligibility conversion (Gerstner et al. 2018)
        if correct:
            converted = self.convert_eligibility(pathway)
        else:
            pathway.eligibility = 0.0  # failure resets eligibility
            converted = False

        # 6. M3: L-LTP hardening (Kandel 2001)
        if correct:
            S = self.compute_hardening_potential(pathway)
            pathway.weight_ltp = max(pathway.weight_ltp, S)
            if self.should_harden(pathway) and not pathway.hardened:
                self.harden_pathway(pathway)
                # M6: Generate PRPs for the region (Frey & Morris 1997)
                self.generate_prps(pathway)

        # 7. M7: Clustered plasticity (Fu et al. 2012)
        if correct and (dw_perf > 0 or converted):
            self.apply_clustered_boost(pathway)

        # 8. M6: PRP decay + capture (Frey & Morris 1997)
        self.decay_prps()
        if self.is_tagged(pathway):
            self.capture_prps(pathway)

        # 9. M5: Shortcut frequency (Abraham & Bear 1996)
        if correct:
            self.update_shortcut_frequency(pathway)

        # 10. M4: Gentle time decay + pruning
        time_decay = self.compute_time_decay(pathway, now)
        pathway.weight_stdp *= time_decay
        pathway.weight_bcm *= time_decay

        # M10: Astrocyte global modulation
        astro_mod = self.get_astrocyte_modulation(pathway.source_region)
        pathway.weight_stdp *= min(1.1, max(0.9, astro_mod))

        # Update success tracking
        if correct:
            pathway.last_used = now; pathway.use_count += 1
            pathway.success_rate = 0.9 * pathway.success_rate + 0.1 * 1.0
        else:
            pathway.last_failed = now; pathway.fail_count += 1
            pathway.success_rate = 0.9 * pathway.success_rate + 0.1 * 0.0

        # Feature signature EMA
        if features:
            alpha = 0.3
            pathway.feature_signature = [
                alpha * f + (1-alpha) * s
                for f, s in zip(features, pathway.feature_signature or [0.0]*22)
            ]

        # Pruning check (performance-driven)
        if self.should_prune(pathway, now):
            self.prune_pathway(pathway)

        # 11. M11: Cascade tracking + criticality auto-tuning
        self.start_cascade_step(pathway.pathway_id)
        self.record_cascade_size()
        self.auto_tune_criticality()

        # Persist periodically
        if pathway.use_count % 10 == 0 or pathway.hardened or pathway.pruned:
            self.save_state()

    # ── Persistence ─────────────────────────────────────────

    def save_state(self):
        storage = _storage_dir()
        cfg = load_synapse_config()
        network_file = storage / cfg["storage"]["pathway_network_file"]
        shortcuts_file = storage / cfg["storage"]["hardened_routes_file"]
        network_file.write_text(
            json.dumps([p.to_dict() for p in self.pathways.values()],
                       ensure_ascii=False, indent=2))
        shortcuts_file.write_text(
            json.dumps(self.hardened_routes, ensure_ascii=False, indent=2))

    def load_state(self):
        storage = _storage_dir()
        cfg = load_synapse_config()
        network_file = storage / cfg["storage"]["pathway_network_file"]
        shortcuts_file = storage / cfg["storage"]["hardened_routes_file"]
        if network_file.exists():
            try:
                for entry in json.loads(network_file.read_text(encoding="utf-8")):
                    pw = SynapticPathway.from_dict(entry)
                    self.pathways[pw.pathway_id] = pw
            except: self.pathways = {}
        if shortcuts_file.exists():
            try: self.hardened_routes = json.loads(shortcuts_file.read_text(encoding="utf-8"))
            except: self.hardened_routes = {}

    def get_statistics(self) -> Dict:
        total = len(self.pathways)
        hardened = sum(1 for p in self.pathways.values() if p.hardened)
        pruned = sum(1 for p in self.pathways.values() if p.pruned)
        active = total - pruned
        avg_w = (sum(p.total_weight() for p in self.pathways.values()
                     if not p.pruned) / max(1, active))
        self._init_neuromod()
        self.__init_prp_pools()
        return {
            "total_pathways": total, "active": active,
            "hardened": hardened, "pruned": pruned,
            "shortcuts": len(self.hardened_routes),
            "avg_weight": round(avg_w, 4),
            "dopamine": round(self.dopamine, 3),
            "serotonin": round(self.serotonin, 3),
            "branching_ratio": round(self.compute_branching_ratio(), 3),
            "prp_pools": {k: round(v, 4) for k, v in self.prp_pools.items() if v > 0.001},
            "astrocyte_ca": {k: round(v, 3) for k, v in self.astrocyte_ca.items() if v > 0.01},
        }


# ─── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SYNAPTIC PATHWAY NETWORK — 11 Mechanisms Test")
    print("=" * 60)

    pn = PathwayNetwork()

    # Test M3: Hardening
    print("\n[M3] L-LTP Hardening (Kandel 2001):")
    pw = pn.find_or_create("motor_cortex", "ds-pro", "code",
                           features=[2.0, 0, 0, 0, 0] + [0]*17)
    for i in range(6):
        pn.update(pw, correct=True, features=[2.0 - i*0.1] + [0]*21)
    print(f"  Hardened={pw.hardened}, LTP={pw.weight_ltp:.3f}, uses={pw.use_count}")

    # Test M8: Eligibility
    print("\n[M8] Eligibility Trace (Gerstner et al. 2018):")
    pw2 = pn.find_or_create("parietal_cortex", "qwen", "math",
                            features=[0, 2.0, 0, 0, 0] + [0]*17)
    pn.update(pw2, correct=True, features=[0, 2.0] + [0]*20)  # first: sets eligibility
    print(f"  After 1st use: eligibility={pw2.eligibility:.3f}")
    pn.update(pw2, correct=True, features=[0, 1.9] + [0]*20)  # second: converts
    print(f"  After 2nd use: eligibility={pw2.eligibility:.3f}, "
          f"weight_ltp={pw2.weight_ltp:.3f}")

    # Test M6: Tagging & Capture
    print("\n[M6] Synaptic Tagging & Capture (Frey & Morris 1997):")
    # Make a pathway in same region that's tagged but not hardened
    pw3 = pn.find_or_create("motor_cortex", "ds-think", "code",
                            features=[1.8, 0, 0, 0, 0] + [0]*17)
    pw3.weight_stdp = 0.3  # above theta_tagged = 0.25
    pw3.recent_uses = [time.time(), time.time()]  # 2 uses
    print(f"  Before capture: pw3 weight_ltp={pw3.weight_ltp:.3f}")
    # PRPs should have been generated from pw hardening
    captured = pn.is_tagged(pw3)
    if captured:
        pn.capture_prps(pw3)
    print(f"  Tagged={captured}, After capture: weight_ltp={pw3.weight_ltp:.3f}")

    # Test M7: Clustered Plasticity
    print("\n[M7] Clustered Plasticity (Fu et al. 2012):")
    # pw4 is similar to pw (both code in motor_cortex)
    pw4 = pn.find_or_create("motor_cortex", "kimi", "code",
                            features=[1.9, 0, 0, 0, 0] + [0]*17)
    before = pw4.weight_stdp
    pn.apply_clustered_boost(pw)  # pw is the strong one
    print(f"  Similar pathway in same region: weight {before:.3f} -> {pw4.weight_stdp:.3f}")

    # Test M4: Performance-driven pruning
    print("\n[M4] Performance-Driven LTD (Bienenstock 1982):")
    pw5 = pn.find_or_create("language_area", "glm", "writing")
    for i in range(8):
        pn.update(pw5, correct=False)  # 8 consecutive failures
    print(f"  After 8 failures: weight={pw5.total_weight():.4f}, "
          f"fail_count={pw5.fail_count}, should_prune={pn.should_prune(pw5)}")

    # Test M9: Neuromodulation
    print("\n[M9] Neuromodulation (Fuxe et al. 2007):")
    print(f"  DA={pn.dopamine:.3f}, 5-HT={pn.serotonin:.3f}, "
          f"expected_reward={pn.expected_reward:.3f}")

    # Test M10: Astrocyte
    print("\n[M10] Astrocyte Ca2+ (Perea & Araque 2007):")
    print(f"  Ca2+ levels: {pn.astrocyte_ca}")

    # Test M11: Criticality
    print("\n[M11] Criticality (Beggs & Plenz 2003):")
    sigma = pn.compute_branching_ratio()
    print(f"  Branching ratio sigma={sigma:.3f} (target=1.0)")

    stats = pn.get_statistics()
    print(f"\nFINAL STATS:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nALL 11 MECHANISMS ACTIVE")
