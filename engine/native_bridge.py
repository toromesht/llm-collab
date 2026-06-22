"""
Native Bridge — C++ Router + Fortran Encoder → brain.py
========================================================

Polyglot integration layer connecting the high-performance
C++ routing engine and Fortran parallel encoder into the
existing brain.py neurosynaptic architecture.

Build:  cd llm-collab && cmake -B build && cmake --build build
Usage:  from engine.native_bridge import NativeRouter, NativeEncoder
"""

import sys
from pathlib import Path
from typing import Optional, List
import numpy as np

# ─── Lazy-loaded native modules ───────────────────────────
_router = None
_available = None


def _check_native() -> bool:
    global _available, _router
    if _available is not None:
        return _available
    try:
        build_lib = Path(__file__).parent.parent / "build" / "lib"
        if build_lib.exists():
            sys.path.insert(0, str(build_lib))
        import synapse_router
        _router = synapse_router
        _available = True
        return True
    except ImportError:
        _available = False
        return False


def native_available() -> bool:
    return _check_native()


class NativeRouter:
    """C++ STDP routing engine. <1us target."""

    def __init__(self):
        self._engine = None
        if _check_native():
            self._engine = _router.RouterEngine()

    @property
    def available(self) -> bool:
        return self._engine is not None

    def load_pathways(self, pathways: List[dict]):
        if self._engine:
            self._engine.load_pathways(pathways)

    def route(self, features: dict, dims: dict, category: str = "general") -> dict:
        if self._engine:
            return self._engine.route(features, dims, category)
        return {"action": "single", "model": "ds-pro", "models": ["ds-pro"], "confidence": 0.5}

    def learn(self, pathway_id: str, success: bool, latency_ms: float = 0):
        if self._engine:
            self._engine.learn(pathway_id, success, latency_ms)

    def stats(self) -> dict:
        if self._engine:
            return self._engine.stats()
        return {"lookups": 0, "hit_rate": 0, "avg_latency_ns": 0}


class NativeEncoder:
    """Fortran OpenMP HD encoder. Kanerva (2009)."""

    HD_DIM = 10000

    def __init__(self):
        self._fortran_loaded = False

    @property
    def available(self) -> bool:
        return False  # Fortran needs iso_c_binding build

    def encode(self, text: str) -> np.ndarray:
        if not text or len(text) < 3:
            return np.zeros(self.HD_DIM, dtype=np.float32)
        hashes = set()
        for i in range(min(len(text) - 2, 500)):
            h = ord(text[i]) * 65536 + ord(text[i+1]) * 256 + ord(text[i+2])
            hashes.add(h)
        if not hashes:
            return np.zeros(self.HD_DIM, dtype=np.float32)
        accum = np.zeros(self.HD_DIM, dtype=np.float32)
        for h in hashes:
            rng = np.random.RandomState(h)
            accum += np.where(rng.random(self.HD_DIM) < 0.5, -1.0, 1.0)
        return np.where(accum >= 0, 1.0, -1.0).astype(np.float32)
