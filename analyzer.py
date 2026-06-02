"""Problem feature analysis for adaptive strategy selection.

Analyzes a SearchContext to produce a ProblemProfile that guides
the solver's strategy exploration, weight tuning, and pruning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from solver import Candidate, SearchContext


@dataclass
class ProblemProfile:
    """Structural summary of a problem instance."""

    task_count: int = 0
    courier_count: int = 0
    candidate_count: int = 0

    # --- candidate composition ---
    bundle_ratio: float = 0.0            # fraction of candidates with task_count > 1
    singleton_ratio: float = 0.0         # fraction of candidates with task_count == 1
    max_bundle_size: int = 1             # largest bundle in the instance

    # --- task coverage ---
    avg_candidates_per_task: float = 0.0
    min_candidates_per_task: int = 0
    max_candidates_per_task: int = 0

    # --- willingness distribution ---
    willingness_mean: float = 0.0
    willingness_std: float = 0.0
    willingness_low_ratio: float = 0.0   # willingness < 0.2
    willingness_high_ratio: float = 0.0  # willingness > 0.6

    # --- courier pressure ---
    avg_candidates_per_courier: float = 0.0
    courier_pressure_cv: float = 0.0     # coefficient of variation

    # --- structural findings ---
    dominated_count: int = 0             # candidates that are strictly dominated
    must_pick_task_count: int = 0        # tasks with exactly 1 candidate
    tight_coupling_count: int = 0        # pairs of tasks sharing >50% candidates

    # --- density ---
    conflict_density: float = 0.0        # fraction of candidate pairs sharing a task

    # --- recommended weight hints ---
    weight_hints: Dict[str, float] = field(default_factory=dict)


def analyze_problem(ctx: "SearchContext") -> ProblemProfile:
    """Build a ProblemProfile from a SearchContext.  Must be fast (<5ms)."""
    profile = ProblemProfile()
    candidates = ctx.candidates
    if not candidates:
        return profile

    profile.candidate_count = len(candidates)
    profile.task_count = len(ctx.task_to_idx)
    profile.courier_count = len(ctx.courier_to_idx)

    # --- candidate composition ---
    bundle_count = sum(1 for c in candidates if c.task_count > 1)
    profile.bundle_ratio = bundle_count / profile.candidate_count
    profile.singleton_ratio = 1.0 - profile.bundle_ratio
    profile.max_bundle_size = max(c.task_count for c in candidates)

    # --- task coverage ---
    task_counts = list(ctx.task_candidate_counts.values())
    if task_counts:
        profile.avg_candidates_per_task = sum(task_counts) / len(task_counts)
        profile.min_candidates_per_task = min(task_counts)
        profile.max_candidates_per_task = max(task_counts)

    # --- willingness ---
    ws = [c.willingness for c in candidates]
    profile.willingness_mean = sum(ws) / len(ws)
    profile.willingness_std = math.sqrt(
        sum((w - profile.willingness_mean) ** 2 for w in ws) / len(ws)
    )
    profile.willingness_low_ratio = sum(1 for w in ws if w < 0.2) / len(ws)
    profile.willingness_high_ratio = sum(1 for w in ws if w > 0.6) / len(ws)

    # --- courier pressure ---
    courier_counts = list(ctx.courier_candidate_counts.values())
    if courier_counts:
        mean_c = sum(courier_counts) / len(courier_counts)
        profile.avg_candidates_per_courier = mean_c
        if mean_c > 0:
            variance = sum((c - mean_c) ** 2 for c in courier_counts) / len(courier_counts)
            profile.courier_pressure_cv = math.sqrt(variance) / mean_c

    # --- dominated candidates ---
    profile.dominated_count = len(find_dominated_candidates(ctx))

    # --- must-pick tasks ---
    must_pick = find_must_pick_tasks(ctx)
    profile.must_pick_task_count = len(must_pick)

    # --- tight coupling ---
    coupling_pairs = find_tight_coupling_pairs(ctx)
    profile.tight_coupling_count = len(coupling_pairs)

    # --- conflict density ---
    profile.conflict_density = _compute_conflict_density(ctx)

    # --- weight hints ---
    profile.weight_hints = recommend_weights(profile)

    return profile


def find_dominated_candidates(ctx: "SearchContext") -> Set[int]:
    """Return indices of candidates that are strictly dominated.

    Candidate A is dominated by B if:
      - B covers a superset of A's tasks (A.task_mask is subset of B.task_mask)
      - B.score <= A.score
      - B.willingness >= A.willingness
      - A and B are not the same candidate
    """
    dominated = set()
    candidates = ctx.candidates
    n = len(candidates)
    if n > 50000:
        # Too many candidates for O(n^2) domination check; skip
        return dominated

    # Group by exact task_mask for faster comparison
    mask_groups: Dict[int, List[int]] = {}
    for i, c in enumerate(candidates):
        mask_groups.setdefault(c.task_mask, []).append(i)

    masks_sorted = sorted(mask_groups.keys(), key=lambda m: bin(m).count("1"), reverse=True)

    for i, c_a in enumerate(candidates):
        if i in dominated:
            continue
        # Check candidates with superset task masks
        for mask_b in masks_sorted:
            if mask_b == c_a.task_mask:
                continue
            if (c_a.task_mask & mask_b) != c_a.task_mask:
                continue  # mask_b doesn't cover all of c_a's tasks
            for j in mask_groups[mask_b]:
                c_b = candidates[j]
                if c_b.score <= c_a.score + 1e-9 and c_b.willingness >= c_a.willingness - 1e-9:
                    # c_b dominates c_a (covers superset, cheaper/equal score, >= willingness)
                    if c_b.score < c_a.score - 1e-9 or c_b.willingness > c_a.willingness + 1e-9:
                        dominated.add(i)
                        break
            if i in dominated:
                break

    return dominated


def find_must_pick_tasks(ctx: "SearchContext") -> Set[str]:
    """Return task IDs that have exactly one candidate (must-pick)."""
    must_pick = set()
    for task_id, count in ctx.task_candidate_counts.items():
        if count == 1:
            must_pick.add(task_id)
    return must_pick


def find_tight_coupling_pairs(
    ctx: "SearchContext", threshold: float = 0.5
) -> List[Tuple[str, str, float]]:
    """Find pairs of tasks that share a high fraction of candidates.

    Returns list of (task_a, task_b, overlap_ratio) sorted by overlap desc.
    Only considers top pairs to keep runtime low.
    """
    if ctx.task_count > 500 or len(ctx.candidates) > 80000:
        # Too large for full pairwise analysis
        return []

    # Build task -> set of candidate indices
    task_to_cand_indices: Dict[str, Set[int]] = {}
    for idx, c in enumerate(ctx.candidates):
        for task_id in c.task_ids:
            task_to_cand_indices.setdefault(task_id, set()).add(idx)

    task_ids = list(ctx.task_to_idx.keys())
    pairs = []
    checked = 0
    max_checks = 10000  # cap to avoid quadratic blowup

    for i in range(len(task_ids)):
        for j in range(i + 1, len(task_ids)):
            checked += 1
            if checked > max_checks:
                break
            ta, tb = task_ids[i], task_ids[j]
            set_a = task_to_cand_indices.get(ta, set())
            set_b = task_to_cand_indices.get(tb, set())
            if not set_a or not set_b:
                continue
            overlap = len(set_a & set_b)
            smaller = min(len(set_a), len(set_b))
            ratio = overlap / smaller if smaller > 0 else 0.0
            if ratio >= threshold:
                pairs.append((ta, tb, ratio))
        if checked > max_checks:
            break

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:100]


def recommend_weights(profile: ProblemProfile) -> Dict[str, float]:
    """Suggest initial strategy weight multipliers based on problem features.

    These are multiplicative hints applied to the base strategy weights,
    not absolute values.
    """
    hints: Dict[str, float] = {
        "score_weight": 1.0,
        "score_per_task_weight": 1.0,
        "willingness_weight": 1.0,
        "bundle_bias": 1.0,
        "scarcity_weight": 1.0,
        "courier_pressure_weight": 1.0,
        "low_willingness_penalty": 1.0,
    }

    # High bundle ratio → prioritize bundle strategies
    if profile.bundle_ratio > 0.6:
        hints["bundle_bias"] = 1.5
    elif profile.bundle_ratio < 0.2:
        hints["bundle_bias"] = 0.3

    # Low willingness → penalize more aggressively
    if profile.willingness_low_ratio > 0.5:
        hints["willingness_weight"] = 1.8
        hints["low_willingness_penalty"] = 2.0
    elif profile.willingness_low_ratio < 0.15:
        hints["willingness_weight"] = 0.5
        hints["low_willingness_penalty"] = 0.3

    # Scarce tasks (few candidates per task) → boost scarcity weight
    if profile.avg_candidates_per_task < 3.0:
        hints["scarcity_weight"] = 2.0
    elif profile.avg_candidates_per_task > 10.0:
        hints["scarcity_weight"] = 0.4

    # High courier pressure → boost courier_pressure weight
    if profile.courier_pressure_cv > 0.8:
        hints["courier_pressure_weight"] = 1.8

    # Many dominated candidates → problem is dense, score_per_task matters more
    if profile.dominated_count > profile.candidate_count * 0.1:
        hints["score_per_task_weight"] = 1.3

    return hints


def _compute_conflict_density(ctx: "SearchContext") -> float:
    """Fraction of task-candidate pairs that conflict (share a task)."""
    n = len(ctx.candidates)
    if n < 2:
        return 0.0

    # Sample-based estimate for large instances
    max_samples = 10000
    if n * (n - 1) // 2 > max_samples:
        import random
        random.seed(42)
        conflicts = 0
        for _ in range(max_samples):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if ctx.candidates[i].task_mask & ctx.candidates[j].task_mask:
                conflicts += 1
        return conflicts / max_samples

    # Exact for small instances
    conflicts = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if ctx.candidates[i].task_mask & ctx.candidates[j].task_mask:
                conflicts += 1
    return conflicts / total if total > 0 else 0.0


def format_profile_summary(profile: ProblemProfile) -> str:
    """One-line summary for logging."""
    return (
        f"tasks={profile.task_count} couriers={profile.courier_count} "
        f"cands={profile.candidate_count} bundle={profile.bundle_ratio:.0%} "
        f"w_mean={profile.willingness_mean:.2f} w_low={profile.willingness_low_ratio:.0%} "
        f"dominated={profile.dominated_count} must_pick={profile.must_pick_task_count} "
        f"coupling={profile.tight_coupling_count} density={profile.conflict_density:.2f}"
    )
