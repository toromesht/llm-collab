"""
SynapseFlow Bridge — Python ↔ C++ Router ↔ Fortran Encoder

Integrates the polyglot backend with brain.py's orchestration.
Replaces sequential model-by-model calls with:
  1. C++ sub-microsecond routing (shared_mutex + STDP)
  2. Fortran+OpenMP parallel HD encoding (10k-bit vectors)
  3. Python async IO for parallel HTTP calls to LLM APIs
"""

from .engine import SynapseEngine, SynapseConfig

__all__ = ["SynapseEngine", "SynapseConfig"]
__version__ = "2.1.0"
