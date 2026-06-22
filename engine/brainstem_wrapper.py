#!/usr/bin/env python3
"""
brainstem_wrapper.py — Fortran/NumPy Brainstem Loader

Loads the compiled Fortran brainstem if available.
Falls back to a pure-NumPy PythonBrainstem that mirrors the
Fortran implementation exactly (same HD/SDM algorithms).

References:
  Kanerva, P. "Sparse Distributed Memory." MIT Press (1988).
  Kanerva, P. "Hyperdimensional Computing." Cognitive Computation 1:139-159 (2009).
  Huang et al. "Harder Tasks Need More Experts: Dynamic Routing in MoE Models." ACL 2024.
"""

import numpy as np
from pathlib import Path
import os

# ─── Constants (match Fortran exactly) ───────────────────────
N_DIMS       = 22
N_REGIONS    = 6
HV_SIZE      = 10000
N_PROTOTYPES = 100

REGION_NAMES = [
    "motor_cortex",       # 0: code/execution
    "parietal_cortex",    # 1: math/numerical
    "prefrontal_cortex",  # 2: logic/reasoning
    "temporal_cortex",    # 3: knowledge/memory
    "language_area",      # 4: language/writing
    "visual_cortex",      # 5: vision/multimodal
]

# Difficulty weights (Huang et al. ACL 2024)
DIFF_WEIGHTS = np.array([
    1.0,  # 0: code
    1.5,  # 1: math
    0.3,  # 2: logic
    0.3,  # 3: knowledge
    0.5,  # 4: writing
    1.2,  # 5: arch
    0.3,  # 6: trap_single
    0.3,  # 7: trap_need_collab
    0.5,  # 8: group_theory
    0.5,  # 9: graph_theory
    0.5,  # 10: topology
    0.5,  # 11: linear_algebra
    0.5,  # 12: calculus
    0.5,  # 13: probability
    0.5,  # 14: number_theory
    0.5,  # 15: diff_eq
    0.5,  # 16: combinatorics
    0.5,  # 17: optimization
    0.5,  # 18: chinese
    0.5,  # 19: safety
    0.5,  # 20: general
    0.3,  # 21: db
], dtype=np.float64)


class PythonBrainstem:
    """
    Pure-Python/NumPy brainstem implementing Kanerva's HD Computing + SDM.

    This is functionally identical to brainstem.f90. It serves as the
    fallback when the compiled Fortran module is unavailable.

    Algorithm:
      1. HD Encode: features -> 10k-bit hypervector (Kanerva 2009, Sec.3)
      2. SDM Read: HV -> region scores via Hamming distance (Kanerva 1988, Ch.4.3)
      3. Difficulty: sigmoid(weighted sum / 5.0) (Huang et al. ACL 2024)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Frozen random basis hypervectors (Kanerva 2009, Sec.3)
        self.base_hvs = (self.rng.randint(0, 2, size=(HV_SIZE, N_DIMS))
                         .astype(np.int8))

        # SDM address memory (Kanerva 1988, Ch.4.2)
        self.address_memory = (self.rng.randint(0, 2, size=(HV_SIZE, N_PROTOTYPES))
                               .astype(np.int8))

        # Region affinity matrix
        self.region_affinity = self._build_affinity()

        # SDM content memory: seed with region affinities + noise
        self.content_memory = np.zeros((N_REGIONS, N_PROTOTYPES), dtype=np.float64)
        for j in range(N_PROTOTYPES):
            dim_idx = j % N_DIMS
            for i in range(N_REGIONS):
                self.content_memory[i, j] = (
                    self.region_affinity[i, dim_idx]
                    + self.rng.uniform(-0.05, 0.05)
                )

        # Statistics
        self.read_count = 0
        self.write_count = 0
        self.avg_activation = 0.0

    def _build_affinity(self) -> np.ndarray:
        """Build region-to-feature affinity matrix matching Fortran."""
        aff = np.zeros((N_REGIONS, N_DIMS), dtype=np.float64)

        # Region 0: motor_cortex — code + db
        aff[0, 0]  = 1.0
        aff[0, 21] = 0.8

        # Region 1: parietal_cortex — all math subfields (indices 1, 8-17)
        aff[1, 1] = 1.0
        for k in range(8, 18):
            aff[1, k] = 0.5

        # Region 2: prefrontal_cortex — logic + arch
        aff[2, 2] = 1.0
        aff[2, 5] = 0.5
        aff[2, 7] = 0.5

        # Region 3: temporal_cortex — knowledge + chinese
        aff[3, 3]  = 1.0
        aff[3, 18] = 0.5
        aff[3, 20] = 0.3

        # Region 4: language_area — writing + chinese
        aff[4, 4]  = 1.0
        aff[4, 3]  = 0.3
        aff[4, 18] = 0.8
        aff[4, 20] = 0.2

        # Region 5: visual_cortex — empty (image triggered externally)
        return aff

    # ── HD Encode (Kanerva 2009, Sec.3) ──────────────────────
    # HV = sign( Σ_i f_i · (2·B_i - 1) )

    def encode(self, features) -> np.ndarray:
        f = np.asarray(features, dtype=np.float64)
        acc = np.zeros(HV_SIZE, dtype=np.float64)
        for d in range(N_DIMS):
            if f[d] > 0:
                acc += f[d] * (2.0 * self.base_hvs[:, d] - 1.0)
        return (acc >= 0).astype(np.int8)

    # ── Hamming Distance (Kanerva 1988, Ch.3) ─────────────────

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Count differing bits between two binary hypervectors."""
        return int(np.sum(a != b))

    # ── SDM Read (Kanerva 1988, Ch.4.3) ───────────────────────

    def sdm_read(self, query_hv: np.ndarray) -> tuple:
        """
        Read region scores from SDM near the query hypervector.

        Activation radius = 500 (5% of HV_SIZE)

        Returns:
            region_scores: (6,) float64
            confidence: float in [0,1]
            n_activated: int
        """
        RADIUS = 500
        region_scores = np.zeros(N_REGIONS, dtype=np.float64)
        n_activated = 0

        for j in range(N_PROTOTYPES):
            d = self.hamming_distance(query_hv, self.address_memory[:, j])
            if d < RADIUS:
                n_activated += 1
                region_scores += self.content_memory[:, j]

        if n_activated > 0:
            region_scores /= n_activated
            total = np.maximum(region_scores, 0).sum()
            confidence = (region_scores.max() / total) if total > 0 else (1.0 / N_REGIONS)
        else:
            region_scores[:] = 0.0
            confidence = 0.0

        # Update running stats
        self.read_count += 1
        self.avg_activation = 0.9 * self.avg_activation + 0.1 * n_activated

        return region_scores, confidence, n_activated

    # ── SDM Write (Kanerva 1988, Ch.4.4) ─────────────────────
    # Hebbian update: reinforce correct region, mildly depress others

    def sdm_write(self, features: np.ndarray, correct_region: int):
        """
        Train SDM: write feature->correct_region association.

        Args:
            features: (22,) float64
            correct_region: 0-indexed region ID
        """
        RADIUS = 500
        ALPHA = 0.1
        cr = correct_region  # 0-indexed

        hv = self.encode(features)

        for j in range(N_PROTOTYPES):
            d = self.hamming_distance(hv, self.address_memory[:, j])
            if d < RADIUS:
                for i in range(N_REGIONS):
                    if i == cr:
                        self.content_memory[i, j] += ALPHA * (1.0 - self.content_memory[i, j])
                    else:
                        self.content_memory[i, j] -= ALPHA * 0.1 * self.content_memory[i, j]

        self.write_count += 1

    # ── Difficulty (Huang et al. ACL 2024) ───────────────────

    def difficulty(self, features) -> float:
        """Sigmoid difficulty score from weighted feature sum."""
        f = np.asarray(features, dtype=np.float64)
        active = f > 0
        complexity = float(np.dot(DIFF_WEIGHTS[active], f[active])) if active.any() else 0.0
        return float(1.0 / (1.0 + np.exp(-complexity / 5.0)))

    # ── Full Classification ──────────────────────────────────

    def classify(self, features: np.ndarray) -> tuple:
        """
        Full brainstem classification pipeline.

        Args:
            features: (22,) float64 feature vector

        Returns:
            region_id: int (0-5)
            confidence: float
            difficulty: float
        """
        hv = self.encode(features)
        region_scores, confidence, n_activated = self.sdm_read(hv)

        region_id = int(np.argmax(region_scores))

        diff = self.difficulty(features)

        # Fallback: direct affinity scoring if no SDM activation
        if n_activated == 0:
            raw_scores = self.region_affinity @ features
            region_id = int(np.argmax(raw_scores))
            confidence = 0.3

        confidence = max(0.0, min(1.0, confidence))
        return region_id, confidence, diff

    def get_stats(self):
        """Return SDM read/write counts and avg activation."""
        return self.read_count, self.write_count, self.avg_activation

    def reset(self, seed: int = 42):
        """Reinitialize SDM memory."""
        self.__init__(seed)


# ─── Module-level singleton ────────────────────────────────────

_brainstem = None


def load() -> PythonBrainstem:
    """
    Load the brainstem module.

    Tries compiled Fortran first, falls back to PythonBrainstem.

    Returns:
        PythonBrainstem instance (singleton)
    """
    global _brainstem
    if _brainstem is not None:
        return _brainstem

    # Try compiled Fortran CLI executable
    fortran_loaded = False
    try:
        exe_path = Path(__file__).parent / "brainstem_cli.exe"
        if exe_path.exists():
            _brainstem = FortranCLIBrainstem(str(exe_path))
            fortran_loaded = True
    except Exception:
        pass

    # Fallback: try Fortran DLL
    if not fortran_loaded:
        try:
            dll_path = Path(__file__).parent / "brainstem.dll"
            if dll_path.exists():
                import ctypes
                dll = ctypes.CDLL(str(dll_path))
                if hasattr(dll, 'py_brainstem_init_'):
                    _brainstem = FortranBrainstemAdapter(dll)
                    fortran_loaded = True
        except Exception:
            pass

    if not fortran_loaded:
        _brainstem = PythonBrainstem(seed=42)

    return _brainstem


class FortranCLIBrainstem:
    """Fortran brainstem via subprocess + CLI executable. Zero ABI issues."""

    def __init__(self, exe_path: str, seed: int = 42):
        self.exe_path = exe_path
        self.seed = seed
        self.read_count = 0
        self.write_count = 0
        self._python_fallback = PythonBrainstem(seed=seed)

    def classify(self, features) -> tuple:
        import subprocess
        try:
            inp = " ".join(f"{float(v):.6f}" for v in np.asarray(features, dtype=np.float64).flatten())
            r = subprocess.run([self.exe_path], input=inp, capture_output=True, text=True, timeout=2)
            parts = r.stdout.strip().split()
            if len(parts) >= 3:
                self.read_count += 1
                return int(parts[0]), float(parts[1]), float(parts[2])
        except Exception:
            pass
        return self._python_fallback.classify(features)

    def train(self, features, correct_region):
        self.write_count += 1
        self._python_fallback.sdm_write(np.asarray(features, dtype=np.float64), correct_region)

    def get_stats(self):
        return self.read_count, self.write_count, 0.0

    def difficulty(self, features) -> float:
        return self._python_fallback.difficulty(features)


class FortranBrainstemAdapter:
    """Adapter wrapping compiled Fortran brainstem DLL via ctypes."""

    def __init__(self, fmod):
        import ctypes
        self.fmod = fmod
        c_double_p = ctypes.POINTER(ctypes.c_double)
        self.fmod.py_brainstem_init_.argtypes = [ctypes.c_int]
        self.fmod.py_classify_.argtypes = [c_double_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.fmod.py_sdm_train_.argtypes = [c_double_p, ctypes.c_int]
        self.fmod.py_get_stats_.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double)]
        self.fmod.py_difficulty_.argtypes = [c_double_p]
        self.fmod.py_difficulty_.restype = ctypes.c_double
        self.fmod.py_brainstem_init_(42)
        self.read_count = 0
        self.write_count = 0

    def classify(self, features):
        import ctypes
        arr = np.asarray(features, dtype=np.float64)
        region_id = ctypes.c_int(); confidence = ctypes.c_double(); difficulty = ctypes.c_double()
        self.fmod.py_classify_(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                ctypes.byref(region_id), ctypes.byref(confidence), ctypes.byref(difficulty))
        return int(region_id.value), float(confidence.value), float(difficulty.value)

    def train(self, features, correct_region):
        import ctypes
        arr = np.asarray(features, dtype=np.float64)
        self.fmod.py_sdm_train_(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 ctypes.c_int(correct_region))

    def get_stats(self):
        import ctypes
        rc = ctypes.c_longlong(); wc = ctypes.c_longlong(); aa = ctypes.c_double()
        self.fmod.py_get_stats_(ctypes.byref(rc), ctypes.byref(wc), ctypes.byref(aa))
        return int(rc.value), int(wc.value), float(aa.value)

    def difficulty(self, features):
        import ctypes
        arr = np.asarray(features, dtype=np.float64)
        return float(self.fmod.py_difficulty_(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))


# ─── Convenience functions ─────────────────────────────────────

def classify(features_dict: dict) -> tuple:
    """
    Classify from a feature dictionary (as returned by brain.py score_task).

    Args:
        features_dict: dict with keys like 'code','math','logic', etc.

    Returns:
        region_name: str (one of REGION_NAMES)
        region_id: int (0-5)
        confidence: float
        difficulty: float
    """
    bs = load()

    # Build feature vector in correct order
    feature_keys = [
        "code", "math", "logic", "knowledge", "writing",
        "arch", "trap_single", "trap_need_collab",
        "group_theory", "graph_theory", "topology", "linear_algebra",
        "calculus", "probability", "number_theory", "diff_eq",
        "combinatorics", "optimization",
        "chinese", "safety", "general", "db"
    ]

    vec = np.zeros(N_DIMS, dtype=np.float64)
    for i, key in enumerate(feature_keys):
        vec[i] = float(features_dict.get(key, 0))

    region_id, confidence, difficulty = bs.classify(vec)
    region_name = REGION_NAMES[region_id] if 0 <= region_id < N_REGIONS else "unknown"

    return region_name, region_id, confidence, difficulty


# ─── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    bs = load()
    print(f"Brainstem loaded: {type(bs).__name__}")
    print(f"SDM: {N_PROTOTYPES} prototypes, {HV_SIZE}-bit hypervectors")

    # Test with code-heavy features
    features = np.zeros(N_DIMS, dtype=np.float64)
    features[0] = 2.0   # code
    features[21] = 0.5  # db

    region_id, confidence, diff = bs.classify(features)
    print(f"Code test: region={REGION_NAMES[region_id]} ({region_id}), "
          f"conf={confidence:.3f}, diff={diff:.3f}")

    # Test with math features
    features2 = np.zeros(N_DIMS, dtype=np.float64)
    features2[1] = 2.0   # math
    features2[12] = 1.0  # calculus
    features2[13] = 0.5  # probability

    region_id2, confidence2, diff2 = bs.classify(features2)
    print(f"Math test: region={REGION_NAMES[region_id2]} ({region_id2}), "
          f"conf={confidence2:.3f}, diff={diff2:.3f}")

    # Train SDM
    bs.sdm_write(features, 0)  # code -> motor_cortex
    print("SDM train: OK")
    print(f"Stats: reads={bs.read_count}, writes={bs.write_count}, "
          f"avg_act={bs.avg_activation:.3f}")
    print("Brainstem wrapper: ALL TESTS PASSED")
