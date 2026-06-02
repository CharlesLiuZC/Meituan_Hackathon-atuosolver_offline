"""Domain operators for local search and candidate pruning.

Provides simulated annealing, candidate pruning, and task conflict
graph analysis to enhance the solver's local search capabilities.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from solver import Candidate, SearchContext, EvalResult

# Local imports - avoid circular dependency at module level
def _count_bits(value):
    return bin(value).count("1")


def _now_ms():
    import time
    return time.perf_counter() * 1000.0


def _has_time(deadline_ms, min_ms=50.0):
    if deadline_ms is None:
        return True
    return max(0.0, deadline_ms - _now_ms()) >= min_ms


# ---------------------------------------------------------------------------
# Candidate pruning
# ---------------------------------------------------------------------------

def prune_dominated_candidates(
    ctx: "SearchContext",
    dominated_indices: Optional[Set[int]] = None,
) -> List["Candidate"]:
    """Return a filtered candidate list with dominated candidates removed.

    If dominated_indices is not provided, computes them on the fly.
    """
    if dominated_indices is None:
        from analyzer import find_dominated_candidates
        dominated_indices = find_dominated_candidates(ctx)

    if not dominated_indices:
        return list(ctx.candidates)

    return [c for i, c in enumerate(ctx.candidates) if i not in dominated_indices]


def prune_dominated_and_rebuild(
    ctx: "SearchContext",
    dominated_indices: Optional[Set[int]] = None,
) -> "SearchContext":
    """Prune dominated candidates and rebuild SearchContext.

    Returns a new SearchContext with cleaned candidates.
    """
    from solver import build_search_context
    pruned = prune_dominated_candidates(ctx, dominated_indices)
    if len(pruned) == len(ctx.candidates):
        return ctx  # Nothing was pruned
    return build_search_context(pruned)


# ---------------------------------------------------------------------------
# Simulated Annealing local search
# ---------------------------------------------------------------------------

def sa_search(
    ctx: "SearchContext",
    initial_selected: List["Candidate"],
    deadline_ms: Optional[float] = None,
    time_budget_ms: float = 1500.0,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.9995,
    min_temp: float = 0.01,
) -> Tuple[List["Candidate"], "EvalResult"]:
    """Simulated Annealing local search to escape local optima.

    Neighborhood operations:
      - swap: replace one selected candidate with an unselected one
      - exchange: swap assignments between two selected candidates

    Acceptance: exp(-delta / T) for worse solutions.
    """
    from solver import evaluate_solution, greedy_select

    import time
    start_time = time.perf_counter()

    def elapsed_ms():
        return (time.perf_counter() - start_time) * 1000.0

    def time_is_up():
        return elapsed_ms() >= time_budget_ms or not _has_time(deadline_ms)

    if not initial_selected or time_is_up():
        return initial_selected, evaluate_solution(initial_selected, "sa_empty", 0.0)

    # Current state
    current = list(initial_selected)
    current_eval = evaluate_solution(current, "sa_current", 0.0)
    best = list(current)
    best_eval = current_eval

    # Build lookup structures
    selected_ids = set(id(c) for c in current)
    unselected = [c for c in ctx.candidates if id(c) not in selected_ids]

    if not unselected:
        return current, current_eval

    # Precompute candidate buckets by task_mask for faster neighbor generation
    task_to_unselected: Dict[int, List] = {}
    for c in unselected:
        task_to_unselected.setdefault(c.task_mask, []).append(c)

    temp = initial_temp
    iteration = 0
    max_iterations = 50000  # Safety cap

    while temp > min_temp and iteration < max_iterations and not time_is_up():
        iteration += 1

        # Choose neighborhood operation
        op = random.random()

        if op < 0.7 or len(current) < 2:
            # SWAP: replace one selected with one unselected
            if not current or not unselected:
                continue

            remove_idx = random.randint(0, len(current) - 1)
            removed = current[remove_idx]

            # Find compatible replacements (same task_mask, different courier)
            candidates_for_swap = task_to_unselected.get(removed.task_mask, [])
            if not candidates_for_swap:
                # Try any unselected candidate whose tasks don't conflict
                used_task_mask = 0
                used_courier_mask = 0
                for i, c in enumerate(current):
                    if i != remove_idx:
                        used_task_mask |= c.task_mask
                        used_courier_mask |= c.courier_bit
                candidates_for_swap = [
                    c for c in unselected
                    if not (c.task_mask & used_task_mask)
                    and not (c.courier_bit & used_courier_mask)
                ]

            if not candidates_for_swap:
                temp *= cooling_rate
                continue

            replacement = random.choice(candidates_for_swap)

            # Check compatibility
            used_task_mask = 0
            used_courier_mask = 0
            for i, c in enumerate(current):
                if i != remove_idx:
                    used_task_mask |= c.task_mask
                    used_courier_mask |= c.courier_bit

            if (replacement.task_mask & used_task_mask) or (replacement.courier_bit & used_courier_mask):
                temp *= cooling_rate
                continue

            # Make swap
            new_current = list(current)
            new_current[remove_idx] = replacement

        else:
            # EXCHANGE: swap assignments between two selected candidates
            i, j = random.sample(range(len(current)), 2)

            # Try exchanging by removing both and reinserting with different candidates
            ci, cj = current[i], current[j]

            # Build mask without ci and cj
            used_task_mask = 0
            used_courier_mask = 0
            for k, c in enumerate(current):
                if k != i and k != j:
                    used_task_mask |= c.task_mask
                    used_courier_mask |= c.courier_bit

            freed_mask = ci.task_mask | cj.task_mask

            # Find replacements for the freed tasks
            replacements = []
            remaining_mask = freed_mask
            used_courier_mask2 = used_courier_mask
            used_task_mask2 = used_task_mask

            for c in unselected:
                if not (c.task_mask & used_task_mask2) and not (c.courier_bit & used_courier_mask2):
                    if c.task_mask & freed_mask and not (c.task_mask & ~freed_mask):
                        replacements.append(c)
                        used_courier_mask2 |= c.courier_bit
                        used_task_mask2 |= c.task_mask
                        remaining_mask &= ~c.task_mask
                        if remaining_mask == 0:
                            break

            if remaining_mask != 0 or len(replacements) < 1:
                temp *= cooling_rate
                continue

            # Build new solution
            new_current = [c for k, c in enumerate(current) if k != i and k != j]
            new_current.extend(replacements)

        # Evaluate new solution
        new_eval = evaluate_solution(new_current, "sa_iter{}".format(iteration), elapsed_ms())
        if not new_eval.is_valid:
            temp *= cooling_rate
            continue

        delta = _sa_objective(new_eval) - _sa_objective(current_eval)

        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            current = new_current
            current_eval = new_eval

            if _sa_objective(new_eval) < _sa_objective(best_eval):
                best = list(new_current)
                best_eval = new_eval

        temp *= cooling_rate

    return best, best_eval


def _sa_objective(result: "EvalResult") -> float:
    """Convert EvalResult to a single scalar for SA optimization.

    Lower is better. Prioritizes coverage then score.
    """
    if not result.is_valid:
        return 1e9 + result.conflict_count * 1000

    # Maximize coverage (negative because lower is better)
    # Then minimize score
    return -result.covered_task_count * 10000 + result.total_score


# ---------------------------------------------------------------------------
# Task conflict graph analysis
# ---------------------------------------------------------------------------

def build_task_conflict_graph(
    ctx: "SearchContext",
) -> Dict[Tuple[str, str], int]:
    """Build a sparse conflict graph: (task_a, task_b) -> shared_candidate_count.

    Only stores pairs with overlap > 0.
    """
    task_pairs: Dict[Tuple[str, str], int] = {}

    for c in ctx.candidates:
        tasks = list(c.task_ids)
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                pair = (tasks[i], tasks[j]) if tasks[i] < tasks[j] else (tasks[j], tasks[i])
                task_pairs[pair] = task_pairs.get(pair, 0) + 1

    return task_pairs


def find_high_conflict_tasks(
    ctx: "SearchContext",
    top_k: int = 20,
) -> List[Tuple[str, int]]:
    """Find tasks involved in the most conflicts.

    Returns list of (task_id, total_conflict_count) sorted desc.
    """
    conflict_counts: Dict[str, int] = {}

    for c in ctx.candidates:
        if c.task_count <= 1:
            continue
        for task_id in c.task_ids:
            conflict_counts[task_id] = conflict_counts.get(task_id, 0) + c.task_count - 1

    ranked = sorted(conflict_counts.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def identify_independent_subproblems(
    ctx: "SearchContext",
) -> List[List[str]]:
    """Identify groups of tasks that can be solved independently.

    Tasks are in the same group if they share at least one candidate.
    Uses union-find for efficiency.
    """
    # Union-find
    parent: Dict[str, str] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Initialize
    for task_id in ctx.task_to_idx:
        parent[task_id] = task_id

    # Union tasks that share candidates
    for c in ctx.candidates:
        tasks = list(c.task_ids)
        for i in range(1, len(tasks)):
            union(tasks[0], tasks[i])

    # Group by root
    groups: Dict[str, List[str]] = {}
    for task_id in ctx.task_to_idx:
        root = find(task_id)
        groups.setdefault(root, []).append(task_id)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Candidate quality scoring for smarter initial ordering
# ---------------------------------------------------------------------------

def compute_candidate_utility(
    candidate: "Candidate",
    profile_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a utility score for a candidate considering multiple factors.

    Higher is better. Used as tiebreaker in greedy selection.
    """
    if profile_weights is None:
        profile_weights = {}

    score_w = profile_weights.get("score_weight", 1.0)
    spt_w = profile_weights.get("score_per_task_weight", 1.0)
    w_w = profile_weights.get("willingness_weight", 0.5)
    bundle_w = profile_weights.get("bundle_bias", 0.3)

    utility = (
        score_w * candidate.score
        + spt_w * candidate.score_per_task * 10
        + w_w * candidate.willingness * 100
        + bundle_w * (candidate.task_count - 1) * 20
    )

    return utility
