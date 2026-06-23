#!/usr/bin/env python3
"""
native_bridge.py — Polyglot Performance Bridge (C++ Router + Fortran Encoder → Python)

Build:  cd collab-cloud && mkdir build && cd build && cmake .. -G "MinGW Makefiles" && cmake --build .
Usage:  from engine.native_bridge import NativeRouter, NativeEncoder, native_available

ARCHITECTURE:
  Python (orchestration, API calls) ↔ C++ (sub-μs routing) + Fortran (OpenMP HD encoding)

  C++ Router:     Consistent Hashing (Karger 1997) + shared_mutex (PostgreSQL pattern)
                  + STDP (Song, Miller, Abbott 2000) + Top-K Gating (Fedus 2022 Switch Transformer)
                  + Weighted Round Robin (Apache/Nginx pattern)
  Fortran Encoder: Kanerva (1988) Sparse Distributed Memory + (2009) HD Computing
                  + OpenMP parallel 10k-bit hypervector encoding
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any


# ═══════════════════════════════════════════════════════════════════════════
# LAZY NATIVE MODULE LOADER
# ═══════════════════════════════════════════════════════════════════════════

_router_module = None
_encoder_module = None
_load_attempted = False


def _try_load_native():
    """Lazy-load compiled C++ router and Fortran encoder. Called once."""
    global _router_module, _encoder_module, _load_attempted
    if _load_attempted:
        return _router_module is not None
    _load_attempted = True

    build_dir = Path(__file__).parent.parent / "build" / "lib"
    if not build_dir.exists():
        return False

    # Add build/lib to Python path
    sys.path.insert(0, str(build_dir))

    # 1. Try loading C++ router (pybind11)
    try:
        import synapse_router
        _router_module = synapse_router
    except ImportError:
        pass

    # 2. Try loading Fortran encoder (ctypes)
    # Fortran shared lib compiled by CMake
    try:
        import ctypes
        dll_path = build_dir / "synapse_encode.dll"
        if dll_path.exists():
            _encoder_module = ctypes.CDLL(str(dll_path))
    except (OSError, ImportError):
        pass

    return _router_module is not None


def native_available() -> bool:
    """Check if compiled native modules are ready."""
    return _try_load_native()


# ═══════════════════════════════════════════════════════════════════════════
# C++ NATIVE ROUTER — STDP routing engine, <1μs target
# ═══════════════════════════════════════════════════════════════════════════

class NativeRouter:
    """
    C++ high-performance routing engine.

    Architecture:
      - Consistent Hash Ring (Karger et al., 1997): O(log N) model lookup
      - Routing Table (PostgreSQL buffer manager pattern): shared_mutex for concurrent reads
      - STDP Computer (Song, Miller, Abbott, 2000): cosine similarity × weight modulation
      - Top-K Gating (Switch Transformer, Fedus et al., 2022): softmax selection
      - WRR Scheduler (Apache/Nginx pattern): weighted fair queuing
      - Atomic counters for zero-lock performance stats

    Target latency: <1μs per routing decision.
    """

    def __init__(self):
        self._engine = None
        if _try_load_native() and _router_module is not None:
            try:
                self._engine = _router_module.RouterEngine()
            except Exception:
                self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def route(self, features: Dict[str, float], dims: Dict[str, float],
              category: str = "general") -> dict:
        """
        Route a question to best model(s).

        Args:
            features: Dict of 22-dim feature values
            dims: Dict with code/math/logic/arch/writing/general scores
            category: Primary task category

        Returns:
            {"action": "single"|"pipeline"|"collab", "model": str, "models": [str],
             "confidence": float, "region": str, "latency_ns": float}
        """
        if self._engine is not None:
            try:
                return self._engine.route(features, dims, category)
            except Exception:
                pass
        # Python fallback: heuristic routing
        return self._fallback_route(features, dims, category)

    def learn(self, pathway_id: str, success: bool, latency_ms: float = 0.0):
        """Record outcome for STDP learning."""
        if self._engine is not None:
            try:
                self._engine.learn(pathway_id, success, latency_ms)
            except Exception:
                pass

    def load_pathways(self, pathways: List[dict]):
        """Load learned pathways from brain state."""
        if self._engine is not None:
            try:
                self._engine.load_pathways(pathways)
            except Exception:
                pass

    def stats(self) -> dict:
        """Performance statistics."""
        if self._engine is not None:
            try:
                return self._engine.stats()
            except Exception:
                pass
        return {"lookups": 0, "hit_rate": 0.0, "avg_latency_ns": 0.0}

    @staticmethod
    def _fallback_route(features: dict, dims: dict, category: str) -> dict:
        """Pure-Python routing fallback when C++ module not compiled."""
        aff = {m: features.get(m, 0.5) for m in ["ds-pro", "ds-think", "glm", "qwen", "kimi", "groq"]}
        best = max(aff, key=aff.get)
        return {
            "action": "single",
            "model": best,
            "models": [best],
            "confidence": aff[best],
            "region": category,
            "latency_ns": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FORTRAN NATIVE ENCODER — OpenMP parallel HD/SDM, <100μs target
# ═══════════════════════════════════════════════════════════════════════════

class NativeEncoder:
    """
    Fortran OpenMP-parallel HD Computing encoder.

    Architecture:
      - Kanerva (2009): 10,000-bit hypervectors with random projection encoding
      - Kanerva (1988): Sparse Distributed Memory — 100 prototype auto-associative memory
      - OpenMP: Parallel bitwise operations across CPU cores

    Target latency: <100μs per 10k-bit encode (Fortran) vs ~5ms (NumPy).

    Falls back to PythonBrainstem if Fortran module unavailable.
    """

    HD_DIM = 10000  # 10k-bit hypervectors (Kanerva 2009)

    def __init__(self):
        self._fortran_dll = None
        if _try_load_native() and _encoder_module is not None:
            self._fortran_dll = _encoder_module

    @property
    def available(self) -> bool:
        return self._fortran_dll is not None

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode 22-dim feature vector → 10k-bit hypervector.

        Args:
            features: (22,) float64 feature vector.

        Returns:
            hv: (10000,) int8 binary hypervector (0/1 bits).
        """
        if self._fortran_dll is not None:
            try:
                import ctypes
                c_double_p = ctypes.POINTER(ctypes.c_double)
                out = np.zeros(self.HD_DIM, dtype=np.int8)
                # Call Fortran hd_encode(features, output, n_dims, hv_size)
                self._fortran_dll.hd_encode_(
                    features.ctypes.data_as(c_double_p),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                    ctypes.c_int(len(features)),
                    ctypes.c_int(self.HD_DIM),
                )
                return out
            except Exception:
                pass

        # NumPy fallback
        return self._fallback_encode(features)

    @staticmethod
    def _fallback_encode(features: np.ndarray) -> np.ndarray:
        """NumPy fallback HD encoding (Kanerva 2009, Sec.3)."""
        f = np.asarray(features, dtype=np.float64)
        rng = np.random.RandomState(42)
        basis = rng.randint(0, 2, size=(10000, len(f))).astype(np.float64)
        acc = np.zeros(10000, dtype=np.float64)
        for d in range(len(f)):
            if f[d] > 0:
                acc += f[d] * (2.0 * basis[:, d] - 1.0)
        return (acc >= 0).astype(np.int8)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════

_router_instance = None
_encoder_instance = None


def get_router() -> NativeRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = NativeRouter()
    return _router_instance


def get_encoder() -> NativeEncoder:
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = NativeEncoder()
    return _encoder_instance
