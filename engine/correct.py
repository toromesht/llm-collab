#!/usr/bin/env python3
"""
correct.py — Weak Model Error Correction for LLM Output Verification

Plan B core innovation: use a CHEAP model ($0.0002/1k) to verify a STRONG
model's output ($0.001-0.002/1k), at 1/5 to 1/10 the cost of the original.

Algorithm:
  1. Strong model generates answer A
  2. Weak model receives (question, A) and asked: "Is A correct? Reply OK or point out errors."
  3. If OK → output A (cost: strong + weak)
  4. If NOT OK → retry strong model once, or fallback to weak model's own answer

Cost model:
  strong_call: $0.001-0.002/1k tokens (DS-Think / DS-Pro)
  weak_call:   $0.0002/1k tokens (Groq Llama 3.3-70B)
  verification ratio: 1/5 to 1/10 of strong model cost

Empirical question (needs benchmark validation):
  Can weak model (Llama 70B) reliably detect errors from strong model (DS-Think)?
  Likely YES for: knowledge, writing, simple code (grammar/logic visible without deep reasoning)
  Likely NO  for: advanced math proofs, complex logic chains
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.brain import call, MODELS


# ═══════════════════════════════════════════════════════════════
# WEAK MODEL CORRECTOR
# ═══════════════════════════════════════════════════════════════

class WeakModelCorrector:
    """
    Verify strong model output using a cheap verifier model.

    Usage:
        c = WeakModelCorrector(weak_model="groq")
        is_ok, corrected, confidence = c.verify(question, strong_answer)
    """

    # Categories where weak model verification is likely reliable
    VERIFIABLE_CATEGORIES = {"knowledge", "writing", "code"}
    # Categories where weak model verification is unreliable
    UNVERIFIABLE_CATEGORIES = {"math", "logic"}

    VERIFY_PROMPT = """You are a quality checker. Your ONLY job is to check if the provided answer correctly addresses the question.

Question: {question}

Proposed Answer: {answer}

Checklist:
1. Does the answer directly address the question?
2. Are there any factual errors?
3. Are there any logical errors?
4. Is anything important missing?

Reply with EXACTLY ONE of:
- "OK" if the answer is correct and complete
- "ERROR: <brief description of the problem>" if anything is wrong

Your response:"""

    CORRECT_PROMPT = """The original answer was flagged as having an error: {error_flag}

Please provide a CORRECTED answer to the original question:

Question: {question}

Give only the corrected answer, no commentary."""

    def __init__(self, weak_model: str = "groq"):
        """
        Args:
            weak_model: Model ID for verification (default: groq = cheapest)
        """
        if weak_model not in MODELS:
            raise ValueError(f"Unknown model: {weak_model}. Available: {list(MODELS.keys())}")
        self.weak_model = weak_model
        self.total_verifications = 0
        self.errors_caught = 0

    def verify(self, question: str, strong_answer: str,
               category: str = "general") -> Tuple[bool, str, float]:
        """
        Verify a strong model's answer using the weak model.

        Args:
            question: Original user question
            strong_answer: Answer produced by the strong model
            category: Task category (affects verification reliability)

        Returns:
            (is_correct: bool, final_answer: str, confidence: float)
        """
        self.total_verifications += 1

        # Quick gate: if answer is too short, it's likely wrong
        if len(str(strong_answer).strip()) < 20:
            return False, strong_answer, 0.2

        # For unverifiable categories, skip verification (avoid false flags)
        if category in self.UNVERIFIABLE_CATEGORIES:
            return True, strong_answer, 0.6  # lower confidence, but trust strong model

        # Build verification prompt
        prompt = self.VERIFY_PROMPT.format(question=question, answer=strong_answer)

        try:
            verdict = call(self.weak_model, prompt, max_tok=100, temp=0.1)
            verdict = verdict.strip()

            if verdict.upper().startswith("OK"):
                return True, strong_answer, 0.9
            else:
                self.errors_caught += 1
                # Error detected — extract error description
                error_desc = verdict.replace("ERROR:", "").strip()[:200]
                return False, strong_answer, 0.3

        except Exception as e:
            # Weak model call failed → trust strong model with low confidence
            return True, strong_answer, 0.5

    def correct(self, question: str, original_answer: str,
                error_flag: str = "unknown error") -> str:
        """
        Attempt to produce a corrected answer when verification fails.
        Uses the weak model itself to regenerate the answer (cheap fallback).

        Args:
            question: Original question
            original_answer: The flagged answer
            error_flag: Description of what was wrong

        Returns:
            Corrected answer
        """
        prompt = self.CORRECT_PROMPT.format(error_flag=error_flag, question=question)
        try:
            return call(self.weak_model, prompt, max_tok=2000, temp=0.3)
        except Exception:
            return original_answer  # fallback to original

    def stats(self) -> dict:
        return {
            "total_verifications": self.total_verifications,
            "errors_caught": self.errors_caught,
            "error_rate": (self.errors_caught / max(1, self.total_verifications)),
            "weak_model": self.weak_model,
        }


# ═══════════════════════════════════════════════════════════════
# FULL PLAN B PIPELINE: Brainstem → Route → Strong → Verify
# ═══════════════════════════════════════════════════════════════

class PlanBPipeline:
    """
    End-to-end Plan B pipeline:

    1. Brainstem classify → (region, confidence, difficulty)
    2. Route to strong model → generate answer
    3. Weak model verify → OK or correct
    4. Output with cost tracking
    """

    def __init__(self, weak_model: str = "groq", router_type: str = "semantic"):
        self.corrector = WeakModelCorrector(weak_model=weak_model)
        self.router_type = router_type
        self.total_cost = 0.0
        self.queries = 0

    def run(self, question: str, verbose: bool = True) -> dict:
        """
        Execute full Plan B pipeline on a single question.

        Returns:
          {"model": str, "answer": str, "verified": bool,
           "cost": float, "region": str, "difficulty": float}
        """
        from engine.brain import score_task, decide_route, execute_single, MODEL_COST

        self.queries += 1

        # Step 1: Classify + Route
        scores = score_task(question)
        decision = decide_route(scores)
        strong_model = decision["model"]
        category = scores["category"]
        difficulty = scores["difficulty"]

        if verbose:
            print(f"  [PIPE] cat={category} diff={difficulty:.2f} → {strong_model}")

        # Step 2: Strong model generates answer
        strong_answer = execute_single(question, strong_model)
        strong_cost = MODEL_COST.get(strong_model, 0.001) * max(10, (len(question) + len(str(strong_answer))) // 4) / 1000

        # Step 3: Weak model verification
        if verbose:
            print(f"  [VERIFY] checking with {self.corrector.weak_model}...")
        is_ok, final_answer, confidence = self.corrector.verify(
            question, str(strong_answer), category)

        verify_cost = 0.0002 * 0.2  # ~200 tokens for verification prompt + output

        # Step 4: If verification failed, try correction
        if not is_ok and category in self.corrector.VERIFIABLE_CATEGORIES:
            if verbose:
                print(f"  [CORRECT] verification failed, regenerating...")
            final_answer = self.corrector.correct(question, str(strong_answer))
            verify_cost += 0.0002 * 1.0  # extra cost for correction call

        total_cost = strong_cost + verify_cost
        self.total_cost += total_cost

        if verbose:
            status = "OK" if is_ok else "FIXED"
            print(f"  [{status}] cost=${total_cost:.6f} (strong=${strong_cost:.6f} + verify=${verify_cost:.6f})")

        return {
            "model": strong_model,
            "answer": str(final_answer)[:3000],
            "verified": is_ok,
            "confidence": confidence,
            "category": category,
            "difficulty": difficulty,
            "cost": total_cost,
            "strong_cost": strong_cost,
            "verify_cost": verify_cost,
        }

    def stats(self) -> dict:
        return {
            **self.corrector.stats(),
            "total_queries": self.queries,
            "total_cost": round(self.total_cost, 6),
            "avg_cost_per_query": round(self.total_cost / max(1, self.queries), 6),
            "router": self.router_type,
        }


# ═══════════════════════════════════════════════════════════════
# MAIN — test with real API calls
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SynapseFlow Plan B — Fortran Brainstem + Weak Model Corrector")
    print("=" * 60)

    pipeline = PlanBPipeline(weak_model="groq", router_type="semantic")

    test_questions = [
        ("什么是量子纠缠？简单解释一下。", "knowledge"),
        ("用Python写一个二分查找函数。", "code"),
        ("求解方程 2x+5=17，给出步骤。", "math"),
        ("法国的首都是哪里？", "knowledge"),
    ]

    for q, expected_cat in test_questions:
        print(f"\n{'─' * 50}")
        print(f"Q: {q}")
        result = pipeline.run(q, verbose=True)
        print(f"  Category: {result['category']} (expected: {expected_cat})")
        print(f"  Model: {result['model']} | Verified: {result['verified']}")
        print(f"  Cost: ${result['cost']:.6f}")
        print(f"  Answer: {result['answer'][:200]}...")

    print(f"\n{'=' * 60}")
    stats = pipeline.stats()
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Total cost: ${stats['total_cost']:.6f}")
    print(f"  Avg cost/query: ${stats['avg_cost_per_query']:.6f}")
    print(f"  Verifications: {stats['total_verifications']}")
    print(f"  Errors caught: {stats['errors_caught']}")
    print(f"  Error rate: {stats['error_rate']:.2%}")
    print("=" * 60)
