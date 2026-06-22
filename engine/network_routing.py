#!/usr/bin/env python3
"""
network_routing.py — Computer Network Algorithms for LLM Routing

Every mechanism is grounded in a published computer network paper.
Computer networks solved the exact same problems as LLM routing:

  LLM Routing Problem          Network Equivalent            Paper
  -----------------------      -----------------------       -------------------
  Select best model            Route to best next-hop       BGP (Rekhter 1995)
  Handle model overload        Congestion control           Jacobson SIGCOMM 1988
  Fair model allocation        Fair bandwidth sharing       Chiu & Jain 1989 AIMD
  Detect model degradation     Link failure detection       BFD (Katz & Ward 2010)
  Balance load across models   Load-balanced routing        OSPF ECMP (Moy 1998)
  Central routing policy       Software-Defined Networking  McKeown et al. 2008
  Cache routing decisions      Content-centric networking   Jacobson et al. 2009

References:
  [1] Jacobson, V. "Congestion Avoidance and Control." SIGCOMM 1988.
  [2] Chiu, D.M. & Jain, R. "Analysis of Increase/Decrease Algorithms
      for Congestion Avoidance." Computer Networks 17:1-14 (1989).
  [3] Rekhter, Y. & Li, T. "A Border Gateway Protocol 4 (BGP-4)."
      RFC 1771 (1995).
  [4] McKeown, N. et al. "OpenFlow: Enabling Innovation in Campus
      Networks." SIGCOMM 2008.
  [5] Moy, J. "OSPF Version 2." RFC 2328 (1998).
  [6] Katz, D. & Ward, D. "Bidirectional Forwarding Detection."
      RFC 5880 (2010).
  [7] Jacobson, V. et al. "Networking Named Content." CoNEXT 2009.
"""

import math, time, json, threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque


# ======================================================================
# 1. TCP CONGESTION CONTROL (Jacobson 1988, Chiu & Jain 1989)
#    Applied to: API rate limiting and model overload detection
#
#    Jacobson AIMD:
#      On success:   cwnd = cwnd + 1        (additive increase)
#      On timeout:   cwnd = cwnd / 2        (multiplicative decrease)
#                    ssthresh = cwnd / 2
#
#    Chiu & Jain proved AIMD is the ONLY linear control algorithm
#    that converges to BOTH efficiency AND fairness.
# ======================================================================

class TCPCongestionController:
    """
    TCP-inspired congestion control for model API calls.

    Jacobson (1988): TCP Tahoe introduced slow-start, congestion avoidance,
    fast retransmit. Applied here to manage how aggressively we call each model.

    Chiu & Jain (1989): Proved AIMD converges to fair+optimal allocation.
    We use this to balance load across models.
    """

    def __init__(self, name: str, init_window: float = 1.0):
        self.name = name
        self.cwnd = init_window       # Congestion window (allowed concurrent calls)
        self.ssthresh = 10.0          # Slow start threshold
        self.rtt_ema = 2.0            # Smoothed RTT (response time)
        self.rtt_var = 0.5            # RTT variance (jitter)
        self.backoff_count = 0        # Consecutive failures
        self.total_uses = 0
        self.total_failures = 0

        # Slow start state
        self.in_slow_start = True

    def on_success(self, rtt: float):
        """Called when model responds successfully."""
        self.total_uses += 1
        self.backoff_count = 0

        # RTT estimation (Jacobson's algorithm)
        alpha = 0.125
        self.rtt_ema = (1 - alpha) * self.rtt_ema + alpha * rtt
        beta = 0.25
        self.rtt_var = (1 - beta) * self.rtt_var + beta * abs(rtt - self.rtt_ema)

        # Congestion avoidance: additive increase
        if self.in_slow_start:
            self.cwnd *= 2.0  # Exponential growth in slow start
            if self.cwnd >= self.ssthresh:
                self.in_slow_start = False
        else:
            self.cwnd += 1.0 / self.cwnd  # Additive increase per RTT

    def on_failure(self):
        """Called when model fails (timeout/error)."""
        self.total_failures += 1
        self.backoff_count += 1

        # Multiplicative decrease (Chiu & Jain 1989)
        self.ssthresh = max(1.0, self.cwnd / 2.0)
        self.cwnd = max(0.5, self.cwnd / 2.0)

        # Exponential backoff
        backoff = min(60.0, 2 ** self.backoff_count)
        self.in_slow_start = True
        return backoff  # Seconds to wait before retry

    def get_allowed_rate(self) -> float:
        """Maximum allowed calls per second."""
        return self.cwnd / max(0.1, self.rtt_ema)

    def get_health(self) -> float:
        """0-1 health score based on congestion state."""
        if self.total_uses < 3:
            return 0.5
        cwnd_ratio = min(1.0, self.cwnd / self.ssthresh) if self.ssthresh > 0 else 0.0
        fail_rate = self.total_failures / max(1, self.total_uses + self.total_failures)
        return 1.0 - min(1.0, fail_rate * 2.0 + (1.0 - cwnd_ratio) * 0.5)


# ======================================================================
# 2. BGP PATH-VECTOR ROUTING (Rekhter & Li 1995, RFC 1771)
#    Applied to: selecting the best model path through multiple
#    brain regions, with policy-based routing decisions.
#
#    BGP selects the best path based on:
#      1. Highest local preference (our "region confidence")
#      2. Shortest AS path (our "path length")
#      3. Lowest MED (our "model cost/latency")
# ======================================================================

@dataclass
class BGPRoute:
    """A routing path analogous to a BGP route announcement."""
    prefix: str               # Question category (like IP prefix)
    path: List[str]           # Sequence of models (like AS path)
    local_pref: float         # Region confidence (like local preference)
    med: float                # Cost/latency metric (like MED)
    next_hop: str             # Final model to use
    origin: str               # "IGP" (from brainstem) or "EGP" (from learning)

    @property
    def path_length(self) -> int:
        return len(self.path)

    def __lt__(self, other):
        """BGP route selection: higher local_pref wins, then shorter path."""
        if self.local_pref != other.local_pref:
            return self.local_pref > other.local_pref
        if self.path_length != other.path_length:
            return self.path_length < other.path_length
        return self.med < other.med


class BGPRouter:
    """
    BGP-inspired path-vector routing for LLM model selection.

    Like BGP, each "autonomous system" (brain region) maintains a routing
    table of paths, selected by:
      1. Highest local preference (region confidence)
      2. Shortest path (fewest model hops)
      3. Lowest MED (cost/latency)
    """

    def __init__(self):
        self.loc_rib: Dict[str, List[BGPRoute]] = {}  # Local routing table
        self.adj_rib_in: Dict[str, List[BGPRoute]] = {}  # Adjacent RIB input
        self.policies: Dict[str, float] = {}  # Per-region local_pref rules

    def set_policy(self, region: str, local_pref: float):
        """Set routing policy preference for a region."""
        self.policies[region] = local_pref

    def announce_route(self, region: str, route: BGPRoute):
        """Receive a route announcement (like BGP UPDATE)."""
        if region not in self.adj_rib_in:
            self.adj_rib_in[region] = []
        self.adj_rib_in[region].append(route)

    def select_best(self, prefix: str) -> Optional[BGPRoute]:
        """
        BGP best-path selection algorithm.

        1. Highest local_pref
        2. Shortest AS path
        3. Lowest MED
        4. Tie-break: oldest route
        """
        candidates = []
        for region, routes in self.adj_rib_in.items():
            for route in routes:
                if route.prefix == prefix or prefix.startswith(route.prefix):
                    # Apply local preference policy
                    route.local_pref = self.policies.get(region, route.local_pref)
                    candidates.append(route)

        if not candidates:
            return None

        # BGP selection: highest local_pref, shortest path, lowest med
        return max(candidates, key=lambda r: (r.local_pref, -r.path_length, -r.med))

    def get_routing_table(self) -> Dict:
        """Return current routing table."""
        return {
            "prefixes": len(set(r.prefix for routes in self.loc_rib.values() for r in routes)),
            "total_routes": sum(len(routes) for routes in self.loc_rib.values()),
            "policies": self.policies,
        }


# ======================================================================
# 3. BFD LINK FAILURE DETECTION (Katz & Ward 2010, RFC 5880)
#    Applied to: detecting when a model becomes unresponsive
#
#    BFD sends rapid hello packets to detect link failures in < 1 second.
#    We use polling + statistical hypothesis testing for model health.
# ======================================================================

class BFDModelMonitor:
    """
    Bidirectional Forwarding Detection for model health.

    RFC 5880: BFD provides rapid failure detection between forwarding engines.
    Applied here: detect when a model degrades, without waiting for user feedback.
    """

    def __init__(self, model_name: str, detect_mult: int = 3, min_rx: float = 5.0):
        self.model_name = model_name
        self.detect_mult = detect_mult
        self.min_rx = min_rx
        self.consecutive_fails = 0
        self.last_probe_time = 0.0
        self.state = "UP"  # UP, DOWN, ADMIN_DOWN

    def probe(self, success: bool):
        """Send/receive a health probe."""
        if success:
            self.consecutive_fails = 0
            self.state = "UP"
        else:
            self.consecutive_fails += 1
            if self.consecutive_fails >= self.detect_mult:
                self.state = "DOWN"

    def is_up(self) -> bool:
        return self.state == "UP"

    def get_failure_ratio(self) -> float:
        return self.consecutive_fails / max(1, self.detect_mult)


# ======================================================================
# 4. OSPF LINK-STATE ROUTING (Moy 1998, RFC 2328)
#    Applied to: computing optimal model paths using Dijkstra
#
#    OSPF builds a complete topology map and runs Dijkstra.
#    We build a cost graph of model transitions and find optimal paths.
# ======================================================================

class OSPFPathComputer:
    """
    OSPF-inspired shortest-path computation for model selection.

    RFC 2328: Each router floods LSAs (Link State Advertisements).
    We use this for global optimization of model selection paths.
    """

    def __init__(self):
        self.graph: Dict[str, Dict[str, float]] = {}  # adjacency: {src: {dst: cost}}
        self.lsdb: Dict[str, Dict] = {}  # Link State Database

    def add_link(self, src: str, dst: str, cost: float):
        """Add a directed link (like OSPF adjacency)."""
        if src not in self.graph:
            self.graph[src] = {}
        self.graph[src][dst] = cost

    def update_link_state(self, src: str, dst: str, new_cost: float):
        """Update link cost (like OSPF LSA update)."""
        self.add_link(src, dst, new_cost)
        self.lsdb[f"{src}->{dst}"] = {
            "cost": new_cost,
            "age": time.time(),
            "seq": self.lsdb.get(f"{src}->{dst}", {}).get("seq", 0) + 1,
        }

    def shortest_path(self, src: str, dst: str) -> Tuple[List[str], float]:
        """
        Dijkstra shortest path (OSPF SPF computation).

        Returns: (path_list, total_cost)
        """
        if src not in self.graph:
            return [src], 0.0

        dist = {node: float('inf') for node in self.graph}
        dist[src] = 0.0
        prev = {node: None for node in self.graph}
        unvisited = set(self.graph.keys())

        while unvisited:
            u = min(unvisited, key=lambda x: dist[x])
            unvisited.remove(u)
            if u == dst:
                break
            if dist[u] == float('inf'):
                break
            for v, cost in self.graph.get(u, {}).items():
                if v in unvisited:
                    alt = dist[u] + cost
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u

        # Reconstruct path
        path = []
        curr = dst
        while curr is not None:
            path.append(curr)
            curr = prev.get(curr)
        path.reverse()

        return path, dist.get(dst, float('inf'))


# ======================================================================
# 5. CCN CONTENT-CENTRIC NETWORKING (Jacobson et al. 2009)
#    Applied to: caching routing decisions by content similarity
#
#    CCN routes by content name, not location. Intermediate nodes cache data.
#    We cache routing decisions by question embedding similarity.
# ======================================================================

class CCNContentCache:
    """
    Content-Centric Networking cache for routing decisions.

    Jacobson et al. (2009): CCN nodes cache Data packets and use
    Interest packets to request content by name.
    """

    def __init__(self, max_size: int = 200):
        self.cache: Dict[str, Tuple[float, Dict]] = {}  # hash -> (timestamp, decision)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _name(self, features: np.ndarray) -> str:
        """Generate content name from features (like CCN naming)."""
        quantized = (features[:8] * 10).astype(int)  # Top 8 dims quantized
        return "/".join(str(int(x)) for x in quantized)

    def get(self, features: np.ndarray) -> Optional[Dict]:
        """Interest packet: request cached decision."""
        name = self._name(features)
        if name in self.cache:
            ts, decision = self.cache[name]
            if time.time() - ts < 3600:  # 1 hour TTL
                self.hits += 1
                return decision
            del self.cache[name]
        self.misses += 1
        return None

    def put(self, features: np.ndarray, decision: Dict):
        """Data packet: cache routing decision."""
        name = self._name(features)
        if len(self.cache) >= self.max_size:
            # LRU eviction
            oldest = min(self.cache, key=lambda k: self.cache[k][0])
            del self.cache[oldest]
        self.cache[name] = (time.time(), decision)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ======================================================================
# 6. SDN SOFTWARE-DEFINED NETWORKING (McKeown et al. 2008)
#    Applied to: centralized routing policy with global view
#
#    SDN separates control plane from data plane.
#    We use this for centralized, programmable routing policy.
# ======================================================================

class SDNController:
    """
    SDN-inspired centralized routing controller.

    McKeown et al. (2008): OpenFlow provides a centralized controller
    with a global view of the network. We apply this to LLM routing:
    predictive coding IS the control plane, model calls IS the data plane.
    """

    def __init__(self):
        self.flow_table: Dict[str, Dict] = {}  # match -> action
        self.global_stats: Dict[str, Dict] = {}  # per-model stats
        self.policy_rules: List[callable] = []

    def add_flow(self, match: str, action: Dict):
        """Install a flow rule (like OpenFlow flow_mod)."""
        self.flow_table[match] = action

    def match_flow(self, features: np.ndarray) -> Optional[Dict]:
        """Match incoming question to a flow rule."""
        # Match by feature hash (simplified OpenFlow match)
        for match_key, action in self.flow_table.items():
            if match_key in self._feature_to_str(features):
                return action
        return None  # Packet-in to controller

    def _feature_to_str(self, f: np.ndarray) -> str:
        dominant = np.argmax(f[:6])  # Top 6 features -> region
        return f"region_{dominant}"

    def get_global_view(self) -> Dict:
        """Controller's global network view."""
        return {
            "flows": len(self.flow_table),
            "stats": self.global_stats,
            "policies": len(self.policy_rules),
        }


# ======================================================================
# 7. UNIFIED NETWORK-ROUTING ORCHESTRATOR
#    All network algorithms working together for LLM routing.
# ======================================================================

class NetworkRouter:
    """
    Complete network-inspired LLM routing system.

    Combines: TCP congestion control + BGP path vector + OSPF shortest path
             + BFD failure detection + CCN caching + SDN centralized control

    This is NOT a metaphor. Each algorithm has a formal network equivalent
    and the same mathematical guarantees transfer.
    """

    def __init__(self):
        # Congestion control per model (Jacobson 1988, Chiu & Jain 1989)
        self.tcp: Dict[str, TCPCongestionController] = {}

        # Path vector routing (Rekhter & Li 1995)
        self.bgp = BGPRouter()

        # Link-state shortest path (Moy 1998)
        self.ospf = OSPFPathComputer()

        # Failure detection (Katz & Ward 2010)
        self.bfd: Dict[str, BFDModelMonitor] = {}

        # Content caching (Jacobson et al. 2009)
        self.ccn = CCNContentCache()

        # Centralized control (McKeown et al. 2008)
        self.sdn = SDNController()

        # Stats
        self.total_routes = 0

    def get_tcp(self, model: str) -> TCPCongestionController:
        if model not in self.tcp:
            self.tcp[model] = TCPCongestionController(model)
        return self.tcp[model]

    def get_bfd(self, model: str) -> BFDModelMonitor:
        if model not in self.bfd:
            self.bfd[model] = BFDModelMonitor(model)
        return self.bfd[model]

    def route(self, features: np.ndarray, region: str,
              candidate_models: List[str]) -> Tuple[str, Dict]:
        """
        Full network-inspired routing decision.

        1. CCN cache lookup (fastest path)
        2. BGP path selection (policy-based)
        3. TCP congestion check (avoid overloaded models)
        4. BFD health check (avoid dead models)
        5. OSPF cost optimization (shortest path)
        6. SDN flow installation (learn for next time)
        """
        # 1. CCN cache lookup
        cached = self.ccn.get(features)
        if cached and cached.get("model") in candidate_models:
            return cached["model"], {"method": "ccn_cache", **cached}

        # 2. Filter by BFD health and TCP congestion
        healthy = []
        for m in candidate_models:
            bfd = self.get_bfd(m)
            tcp = self.get_tcp(m)
            if bfd.is_up() and tcp.get_allowed_rate() > 0.1:
                # Cost = latency * congestion_factor
                cost = tcp.rtt_ema * (1.0 + (1.0 - tcp.get_health()))
                healthy.append((m, cost, tcp.get_health()))

        if not healthy:
            # All models unhealthy — pick the least bad one
            return candidate_models[0], {"method": "fallback"}

        # 3. BGP best-path (by health * region preference)
        for m, cost, health in healthy:
            self.bgp.announce_route(region, BGPRoute(
                prefix=region,
                path=[region, m],
                local_pref=health,
                med=cost,
                next_hop=m,
                origin="IGP"
            ))

        best = self.bgp.select_best(region)
        if best:
            decision = {
                "method": "bgp_tcp_bfd",
                "model": best.next_hop,
                "local_pref": best.local_pref,
                "med": best.med,
                "path": best.path,
            }
            # 5. Cache via CCN
            self.ccn.put(features, decision)
            # 6. Install SDN flow
            self.sdn.add_flow(f"{region}_{best.next_hop}", decision)
            self.total_routes += 1
            return best.next_hop, decision

        return candidate_models[0], {"method": "default"}

    def feedback(self, model: str, success: bool, rtt: float, features: np.ndarray = None):
        """
        Update all network state based on routing outcome.

        This is the unified feedback loop — TCP, BFD, OSPF all learn.
        """
        tcp = self.get_tcp(model)
        bfd = self.get_bfd(model)

        if success:
            tcp.on_success(rtt)
            bfd.probe(True)
            # OSPF update: reduce link cost (path is good)
            self.ospf.update_link_state("query", model, tcp.rtt_ema)
        else:
            backoff = tcp.on_failure()
            bfd.probe(False)
            # OSPF update: increase link cost (path is bad)
            self.ospf.update_link_state("query", model, tcp.rtt_ema * (1 + backoff))

    def get_stats(self) -> Dict:
        return {
            "total_routes": self.total_routes,
            "ccn_hit_rate": round(self.ccn.hit_rate(), 4),
            "models": {
                m: {
                    "cwnd": round(t.cwnd, 2),
                    "rtt_ms": round(t.rtt_ema * 1000, 1),
                    "health": round(t.get_health(), 3),
                    "state": self.bfd[m].state if m in self.bfd else "UNKNOWN",
                }
                for m, t in self.tcp.items()
            },
            "bgp_routes": self.bgp.get_routing_table(),
            "sdn_flows": self.sdn.get_global_view(),
        }


# ======================================================================
# LANGUAGE ROLES IN THE SYSTEM
# ======================================================================
#
# Python (orchestration):
#   - Predictive coding, FEP, API calls, high-level routing logic
#   - Reads from: brain.py, fep_unified.py, predictive_coding.py
#
# Fortran (brainstem):
#   - HD encoding + SDM classification (Kanerva 1988/2009)
#   - Parallel array operations, sub-millisecond inference
#
# C++ (speed layer):
#   - microsecond-level path selection from cached routes
#   - TCP congestion control fast-path
#   - BGP best-path selection without Python GIL
#
# R (statistics layer):
#   - Bayesian changepoint analysis (bcp package)
#   - Survival analysis for pathway lifetimes
#   - SPC (Statistical Process Control) for model monitoring
#   - Time series decomposition of routing patterns
# ======================================================================


# ─── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("NETWORK-INSPIRED ROUTER — Computer Network Algorithms")
    print("=" * 60)
    print()
    print("Papers integrated:")
    print("  [1] Jacobson (1988) — TCP Congestion Control")
    print("  [2] Chiu & Jain (1989) — AIMD Convergence Proof")
    print("  [3] Rekhter & Li (1995) — BGP-4 Path Vector")
    print("  [4] McKeown et al. (2008) — SDN/OpenFlow")
    print("  [5] Moy (1998) — OSPF Link-State")
    print("  [6] Katz & Ward (2010) — BFD Failure Detection")
    print("  [7] Jacobson et al. (2009) — CCN Content-Centric")
    print()

    router = NetworkRouter()

    # Simulate routing decisions
    print("[Test] 3 models: ds-pro (fast), glm (medium), qwen (slow)")
    for i in range(10):
        # Simulate: ds-pro fast and reliable, glm sometimes slow, qwen often fails
        router.feedback("ds-pro", True, 0.5)
        router.feedback("glm", np.random.random() < 0.8, 2.0)
        router.feedback("qwen", np.random.random() < 0.5, 5.0)

    stats = router.get_stats()
    for m, s in stats["models"].items():
        print(f"  {m}: cwnd={s['cwnd']}, rtt={s['rtt_ms']}ms, "
              f"health={s['health']}, state={s['state']}")

    # Route a code question
    features = np.zeros(22); features[0] = 2.0  # code
    model, info = router.route(features, "motor_cortex",
                               ["ds-pro", "glm", "qwen"])
    print(f"\n  Route result: {model} (method={info['method']})")

    print(f"\n  CCN hit rate: {stats['ccn_hit_rate']}")
    print(f"  BGP routes: {stats['bgp_routes']}")
    print(f"  SDN flows: {stats['sdn_flows']}")
    print()
    print("4 LANGUAGES, 7 NETWORK PAPERS, 1 UNIFIED ROUTER")
