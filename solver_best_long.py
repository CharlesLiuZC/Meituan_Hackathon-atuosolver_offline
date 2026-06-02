"""Standalone AutoSolver for the Meituan courier-task assignment challenge.

The solver is intentionally deterministic and dependency-free.  It treats the
input rows as candidate assignments and searches for a valid subset that covers
as many unique tasks as possible, then minimizes the official-like penalty that
combines willingness-weighted score and expected rejection cost.
"""

import itertools
import time


EPS = 1e-9

CONFIG = {'time_budget_ms': 9200.0,
 'safety_margin_ms': 300.0,
 'auto_strategy_budget_ms': 300.0,
 'local_search_budget_ms': 2800.0,
 'backup_time_budget_ms': 300.0,
 'ilp_time_limit_seconds': 1.4,
 'enable_multi_courier_output': False,
 'acceptance_penalty': 125.0,
 'max_extra_couriers_per_bundle': 3,
 'min_backup_utility': 35.0,
 'min_remaining_ms': 45.0,
 'max_generated_strategies': 24,
 'max_exact_replace_tasks': 8,
 'max_candidates_per_mask': 40,
 'pair_top_k': 32,
 'triple_top_k': 32,
 'try_triples': True,
 'prune_dominated': False,
 'dynamic_penalty': True,
 'strategies': [(0.0393, 1.0468, 0.0696, 0.0911, 0.0737, 0.0858, 0),
                (0.0814, 1.103, 0.0971, 0.1826, 0.0994, 0.0493, 0),
                (0.0433, 1.1646, 0.0598, 0.3258, 0.0738, 0.1007, 1),
                (0.0473, 1.0864, 0.0681, 0.1108, 0.1561, 0.0403, 0),
                (0.0522, 1.1332, 0.0637, 0.39, 0.1036, 0.0769, 1),
                (0.0547, 1.1315, 0.0, 0.1416, 0.1148, 0.0611, 0)],
 '_runtime_acceptance_penalty': 125.0,
 '_runtime_special_case': False,
 '_runtime_penalty_profiles': [125.0]}

class Candidate:
    __slots__ = (
        "task_str",
        "task_ids",
        "task_mask",
        "courier_id",
        "courier_idx",
        "courier_bit",
        "score",
        "willingness",
        "task_count",
        "score_per_task",
        "min_task_degree",
        "sum_task_degree",
        "courier_degree",
    )

    def __init__(
        self,
        task_str,
        task_ids,
        task_mask,
        courier_id,
        courier_idx,
        score,
        willingness,
    ):
        self.task_str = task_str
        self.task_ids = task_ids
        self.task_mask = task_mask
        self.courier_id = courier_id
        self.courier_idx = courier_idx
        self.courier_bit = 1 << courier_idx
        self.score = score
        self.willingness = willingness
        self.task_count = len(task_ids)
        self.score_per_task = score / max(1, self.task_count)
        self.min_task_degree = 0
        self.sum_task_degree = 0
        self.courier_degree = 0


class Context:
    __slots__ = (
        "candidates",
        "task_to_idx",
        "courier_to_idx",
        "all_task_mask",
        "task_degrees",
        "courier_degrees",
        "mask_to_candidates",
        "max_score",
        "max_score_per_task",
        "max_task_degree",
        "max_courier_degree",
    )

    def __init__(self, candidates, task_to_idx, courier_to_idx):
        self.candidates = candidates
        self.task_to_idx = task_to_idx
        self.courier_to_idx = courier_to_idx
        self.all_task_mask = (1 << len(task_to_idx)) - 1
        self.task_degrees = {}
        self.courier_degrees = {}
        self.mask_to_candidates = {}
        self.max_score = 1.0
        self.max_score_per_task = 1.0
        self.max_task_degree = 1
        self.max_courier_degree = 1


class Eval:
    __slots__ = ("covered", "score", "penalty_score", "conflicts", "items")

    def __init__(self, covered, score, penalty_score, conflicts, items):
        self.covered = covered
        self.score = score
        self.penalty_score = penalty_score
        self.conflicts = conflicts
        self.items = items


def _now_ms():
    return time.perf_counter() * 1000.0


def _count_bits(value):
    return bin(value).count("1")


def _remaining(deadline_ms):
    return deadline_ms - _now_ms()


def _has_time(deadline_ms, min_ms=None):
    if min_ms is None:
        min_ms = CONFIG["min_remaining_ms"]
    return _remaining(deadline_ms) > min_ms


def parse_input(input_text):
    candidates = []
    task_to_idx = {}
    courier_to_idx = {}
    if not input_text:
        return candidates, task_to_idx, courier_to_idx

    lines = input_text.strip().splitlines()
    if not lines:
        return candidates, task_to_idx, courier_to_idx
    start = 1 if lines[0].strip().startswith("task_id_list") else 0

    for line in lines[start:]:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        task_str, courier_id, score_str, willingness_str = parts[:4]
        task_str = task_str.strip()
        courier_id = courier_id.strip()
        task_ids = tuple(t.strip() for t in task_str.split(",") if t.strip())
        if not task_ids or not courier_id:
            continue
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue

        task_mask = 0
        for task_id in task_ids:
            if task_id not in task_to_idx:
                task_to_idx[task_id] = len(task_to_idx)
            task_mask |= 1 << task_to_idx[task_id]
        if courier_id not in courier_to_idx:
            courier_to_idx[courier_id] = len(courier_to_idx)

        candidates.append(
            Candidate(
                task_str,
                task_ids,
                task_mask,
                courier_id,
                courier_to_idx[courier_id],
                score,
                willingness,
            )
        )
    return candidates, task_to_idx, courier_to_idx


def build_context(candidates, task_to_idx, courier_to_idx):
    ctx = Context(candidates, task_to_idx, courier_to_idx)
    if not candidates:
        return ctx

    ctx.max_score = max(1.0, max(abs(c.score) for c in candidates))
    ctx.max_score_per_task = max(1.0, max(abs(c.score_per_task) for c in candidates))

    for c in candidates:
        ctx.courier_degrees[c.courier_id] = ctx.courier_degrees.get(c.courier_id, 0) + 1
        for task_id in set(c.task_ids):
            ctx.task_degrees[task_id] = ctx.task_degrees.get(task_id, 0) + 1
        ctx.mask_to_candidates.setdefault(c.task_mask, []).append(c)

    if ctx.task_degrees:
        ctx.max_task_degree = max(ctx.task_degrees.values())
    if ctx.courier_degrees:
        ctx.max_courier_degree = max(ctx.courier_degrees.values())

    for c in candidates:
        degrees = [ctx.task_degrees.get(task_id, 0) for task_id in set(c.task_ids)]
        c.min_task_degree = min(degrees) if degrees else 0
        c.sum_task_degree = sum(degrees)
        c.courier_degree = ctx.courier_degrees.get(c.courier_id, 0)

    for items in ctx.mask_to_candidates.values():
        items.sort(key=lambda c: (candidate_penalty_cost(c), c.score, -c.willingness))

    return ctx


def configure_runtime(candidates, task_to_idx, courier_to_idx):
    """Set per-case search knobs from coarse data statistics.

    The public cases include low-willingness and scarce-courier regimes.  A
    single static risk weight is brittle, so the solver adapts the internal
    penalty used by all strategy comparisons while keeping the final output
    format unchanged.
    """
    if not candidates or not CONFIG.get("dynamic_penalty", True):
        CONFIG["_runtime_acceptance_penalty"] = float(CONFIG.get("acceptance_penalty", 100.0))
        return

    avg_willingness = sum(c.willingness for c in candidates) / float(len(candidates))
    task_count = max(1, len(task_to_idx))
    courier_count = max(1, len(courier_to_idx))
    courier_ratio = courier_count / float(task_count)
    candidate_density = len(candidates) / float(task_count * courier_count)

    penalty = float(CONFIG.get("acceptance_penalty", 100.0))
    if avg_willingness < 0.18:
        penalty = 150.0
    elif avg_willingness < 0.26:
        penalty = 130.0
    elif avg_willingness > 0.48:
        penalty = 92.0

    if courier_ratio <= 0.9:
        penalty += 12.0
    if candidate_density < 8.0:
        penalty += 8.0

    CONFIG["_runtime_acceptance_penalty"] = penalty
    CONFIG["_runtime_special_case"] = (
        avg_willingness < 0.26 or courier_ratio <= 0.9 or candidate_density < 8.0
    )
    profiles = [penalty]
    if avg_willingness < 0.18:
        profiles.extend([150.0, 100.0])
    elif avg_willingness < 0.26:
        profiles.extend([130.0, 100.0])
    elif courier_ratio <= 0.9:
        profiles.extend([130.0, 100.0])
    elif candidate_density < 8.0:
        profiles.extend([120.0, 100.0])

    deduped = []
    for value in profiles:
        rounded = round(float(value), 3)
        if rounded not in deduped:
            deduped.append(rounded)
    CONFIG["_runtime_penalty_profiles"] = deduped


def find_dominated(ctx):
    """Find candidates strictly dominated by another candidate.

    A is dominated by B if B covers a superset of tasks, has lower or equal
    penalty cost, and higher or equal willingness.
    """
    dominated = set()
    candidates = ctx.candidates
    n = len(candidates)
    if n > 60000:
        return dominated

    # Group by task_mask
    mask_to_indices = {}
    for i, c in enumerate(candidates):
        mask_to_indices.setdefault(c.task_mask, []).append(i)

    masks_by_size = sorted(mask_to_indices.keys(), key=lambda m: _count_bits(m), reverse=True)

    for i, ca in enumerate(candidates):
        if i in dominated:
            continue
        pa = candidate_penalty_cost(ca)
        for mb in masks_by_size:
            if mb == ca.task_mask:
                continue
            if (ca.task_mask & mb) != ca.task_mask:
                continue
            for j in mask_to_indices[mb]:
                cb = candidates[j]
                pb = candidate_penalty_cost(cb)
                if pb <= pa + 1e-9 and cb.willingness >= ca.willingness - 1e-9:
                    if pb < pa - 1e-9 or cb.willingness > ca.willingness + 1e-9:
                        dominated.add(i)
                        break
            if i in dominated:
                break
    return dominated


def prune_candidates(candidates, dominated_indices):
    """Remove dominated candidates from the list."""
    if not dominated_indices:
        return candidates
    return [c for i, c in enumerate(candidates) if i not in dominated_indices]


def candidate_penalty_cost(c):
    penalty = float(CONFIG.get("_runtime_acceptance_penalty", CONFIG.get("acceptance_penalty", 100.0)))
    return c.score * c.willingness + penalty * c.task_count * (1.0 - c.willingness)


def official_penalty_cost(c):
    return c.score * c.willingness + 100.0 * c.task_count * (1.0 - c.willingness)


def evaluate(selected, total_task_count=None):
    task_mask = 0
    courier_mask = 0
    conflicts = 0
    score = 0.0
    penalty_score = 0.0
    for c in selected:
        conflicts += _count_bits(task_mask & c.task_mask)
        conflicts += c.task_count - _count_bits(c.task_mask)
        if courier_mask & c.courier_bit:
            conflicts += 1
        task_mask |= c.task_mask
        courier_mask |= c.courier_bit
        score += c.score
        penalty_score += candidate_penalty_cost(c)
    covered = _count_bits(task_mask)
    if total_task_count is not None and total_task_count > covered:
        penalty_score += float(CONFIG.get("acceptance_penalty", 100.0)) * (total_task_count - covered)
    return Eval(covered, score, penalty_score, conflicts, len(selected))


def evaluate_with_penalty(selected, total_task_count=None, penalty=100.0):
    task_mask = 0
    courier_mask = 0
    conflicts = 0
    score = 0.0
    penalty_score = 0.0
    for c in selected:
        conflicts += _count_bits(task_mask & c.task_mask)
        conflicts += c.task_count - _count_bits(c.task_mask)
        if courier_mask & c.courier_bit:
            conflicts += 1
        task_mask |= c.task_mask
        courier_mask |= c.courier_bit
        score += c.score
        penalty_score += c.score * c.willingness + penalty * c.task_count * (1.0 - c.willingness)
    covered = _count_bits(task_mask)
    if total_task_count is not None and total_task_count > covered:
        penalty_score += penalty * (total_task_count - covered)
    return Eval(covered, score, penalty_score, conflicts, len(selected))


def is_better(new_eval, old_eval):
    if old_eval is None:
        return True
    if new_eval.conflicts != old_eval.conflicts:
        return new_eval.conflicts < old_eval.conflicts
    if new_eval.covered != old_eval.covered:
        return new_eval.covered > old_eval.covered
    if abs(new_eval.penalty_score - old_eval.penalty_score) > EPS:
        return new_eval.penalty_score < old_eval.penalty_score
    if abs(new_eval.score - old_eval.score) > EPS:
        return new_eval.score < old_eval.score
    return new_eval.items < old_eval.items


def greedy_select(ordered):
    selected = []
    used_tasks = 0
    used_couriers = 0
    for c in ordered:
        if c.task_mask & used_tasks:
            continue
        if c.courier_bit & used_couriers:
            continue
        selected.append(c)
        used_tasks |= c.task_mask
        used_couriers |= c.courier_bit
    return selected


def strategy_key(ctx, spec):
    score_w, per_task_w, willing_w, bundle_w, scarcity_w, courier_w, bundle_first = spec
    max_score = ctx.max_score
    max_score_per_task = ctx.max_score_per_task
    max_task_degree = float(ctx.max_task_degree)
    max_courier_degree = float(ctx.max_courier_degree)

    def key(c):
        scarcity = c.min_task_degree / max_task_degree if max_task_degree else 0.0
        courier_pressure = c.courier_degree / max_courier_degree if max_courier_degree else 0.0
        rank = (
            score_w * (c.score / max_score)
            + per_task_w * (c.score_per_task / max_score_per_task)
            - willing_w * c.willingness
            - bundle_w * (c.task_count - 1)
            + scarcity_w * scarcity
            + courier_w * courier_pressure
        )
        if bundle_first:
            return (0 if c.task_count > 1 else 1, rank, c.score, -c.willingness)
        return (rank, c.score, -c.willingness)

    return key


def base_orders(ctx):
    orders = [
        ("official_penalty", sorted(ctx.candidates, key=lambda c: (official_penalty_cost(c), c.score, -c.willingness))),
        ("official_penalty_per_task", sorted(ctx.candidates, key=lambda c: (official_penalty_cost(c) / max(1, c.task_count), official_penalty_cost(c), -c.willingness))),
        ("penalty", sorted(ctx.candidates, key=lambda c: (candidate_penalty_cost(c), c.score, -c.willingness))),
        ("penalty_per_task", sorted(ctx.candidates, key=lambda c: (candidate_penalty_cost(c) / max(1, c.task_count), candidate_penalty_cost(c), -c.willingness))),
        ("willingness_first", sorted(ctx.candidates, key=lambda c: (-c.willingness, candidate_penalty_cost(c) / max(1, c.task_count), c.score))),
        ("risk_adjusted", sorted(ctx.candidates, key=lambda c: (candidate_penalty_cost(c) / max(0.05, c.willingness + 0.05), candidate_penalty_cost(c), c.score))),
        ("bundle_penalty_per_task", sorted(ctx.candidates, key=lambda c: (0 if c.task_count > 1 else 1, candidate_penalty_cost(c) / max(1, c.task_count), -c.willingness, c.score))),
        ("score_per_willingness", sorted(ctx.candidates, key=lambda c: (c.score / max(0.01, c.willingness), c.score, -c.willingness))),
        ("single_penalty", sorted(ctx.candidates, key=lambda c: (0 if c.task_count == 1 else 1, candidate_penalty_cost(c), c.score))),
        ("score", sorted(ctx.candidates, key=lambda c: (c.score, c.score_per_task, -c.willingness))),
        ("per_task", sorted(ctx.candidates, key=lambda c: (c.score_per_task, c.score, -c.willingness))),
        ("bundle_first", sorted(ctx.candidates, key=lambda c: (0 if c.task_count > 1 else 1, candidate_penalty_cost(c), c.score))),
        ("scarcity", sorted(ctx.candidates, key=lambda c: (c.min_task_degree, c.sum_task_degree, c.score_per_task, c.score))),
    ]
    if CONFIG.get("_runtime_special_case", False):
        orders.extend([
            ("scarce_bundle_reliable", sorted(ctx.candidates, key=lambda c: (0 if c.task_count > 1 else 1, -c.willingness, candidate_penalty_cost(c) / max(1, c.task_count), c.score))),
            ("hard_bundle_official", sorted(ctx.candidates, key=lambda c: (0 if c.task_count > 1 else 1, official_penalty_cost(c) / max(1, c.task_count), -c.willingness, c.score))),
            ("hard_bundle_willingness", sorted(ctx.candidates, key=lambda c: (0 if c.task_count > 1 else 1, -c.willingness, official_penalty_cost(c) / max(1, c.task_count), c.score))),
            ("low_willingness_guard", sorted(ctx.candidates, key=lambda c: (official_penalty_cost(c) / max(0.04, c.willingness + 0.04), -c.willingness, c.score))),
        ])
    return orders


def generated_specs():
    specs = []
    seen = set()
    for spec in CONFIG["strategies"]:
        sig = tuple(round(x, 4) if isinstance(x, float) else x for x in spec)
        if sig not in seen:
            seen.add(sig)
            specs.append(spec)

    seeds = list(specs)
    deltas = [
        (0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0),
        (-0.03, 0.05, 0.00, 0.00, 0.00, 0.00, 0),
        (0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0),
        (0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0),
        (0.00, 0.00, 0.00, -0.04, 0.08, 0.00, 0),
        (0.00, 0.02, 0.02, 0.05, 0.05, 0.08, 0),
    ]
    for spec in seeds:
        for delta in deltas:
            if len(specs) >= CONFIG["max_generated_strategies"]:
                return specs
            mutated = (
                max(0.0, spec[0] + delta[0]),
                max(0.0, spec[1] + delta[1]),
                max(0.0, spec[2] + delta[2]),
                max(0.0, spec[3] + delta[3]),
                max(0.0, spec[4] + delta[4]),
                max(0.0, spec[5] + delta[5]),
                spec[6],
            )
            sig = tuple(round(x, 4) if isinstance(x, float) else x for x in mutated)
            if sig not in seen:
                seen.add(sig)
                specs.append(mutated)
    return specs


def rank_removals(selected):
    return sorted(
        selected,
        key=lambda c: (candidate_penalty_cost(c) / max(1, c.task_count), candidate_penalty_cost(c), c.score),
        reverse=True,
    )


def mask_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def exact_cover_freed(ctx, freed_mask, locked_courier_mask, score_ceiling, deadline_ms=None):
    if not freed_mask:
        return []
    if _count_bits(freed_mask) > CONFIG["max_exact_replace_tasks"]:
        return None

    relevant_by_bit = {}
    seen = set()
    submask = freed_mask
    while submask:
        for c in ctx.mask_to_candidates.get(submask, ())[: CONFIG["max_candidates_per_mask"]]:
            ident = id(c)
            if ident in seen:
                continue
            seen.add(ident)
            if c.courier_bit & locked_courier_mask:
                continue
            if c.task_mask & ~freed_mask:
                continue
            for bit in mask_bits(c.task_mask):
                relevant_by_bit.setdefault(bit, []).append(c)
        submask = (submask - 1) & freed_mask

    for bit in relevant_by_bit:
        relevant_by_bit[bit].sort(key=lambda c: (candidate_penalty_cost(c), c.score, -c.willingness))

    best_score = [score_ceiling]
    best_selected = [None]

    def dfs(covered_mask, used_couriers, score, selected):
        if deadline_ms is not None and not _has_time(deadline_ms, 30.0):
            return
        if score >= best_score[0] - EPS:
            return
        if covered_mask == freed_mask:
            best_score[0] = score
            best_selected[0] = list(selected)
            return
        remaining = freed_mask & ~covered_mask
        next_bit = None
        best_count = 10**9
        for bit in mask_bits(remaining):
            count = 0
            for c in relevant_by_bit.get(bit, ()):
                if c.task_mask & covered_mask:
                    continue
                if c.courier_bit & used_couriers:
                    continue
                count += 1
            if count < best_count:
                best_count = count
                next_bit = bit
        if next_bit is None or best_count == 0:
            return
        for c in relevant_by_bit.get(next_bit, ()):
            if c.task_mask & covered_mask:
                continue
            if c.courier_bit & used_couriers:
                continue
            dfs(covered_mask | c.task_mask, used_couriers | c.courier_bit, score + candidate_penalty_cost(c), selected + [c])

    dfs(0, 0, 0.0, [])
    return best_selected[0]


def replace_candidates(selected, removed_tuple, replacement):
    removed = set(removed_tuple)
    kept = [c for c in selected if c not in removed]
    kept.extend(replacement)
    return kept


def local_search(ctx, selected, deadline_ms):
    current = list(selected)
    total_tasks = _count_bits(ctx.all_task_mask)
    current_eval = evaluate(current, total_tasks)
    start_ms = _now_ms()
    budget = CONFIG["local_search_budget_ms"]

    # Phase A: fast single-swap (replace one candidate with same-task-mask alternative)
    selected_set = set(id(c) for c in current)
    unselected = [c for c in ctx.candidates if id(c) not in selected_set]

    # Build index: task_mask -> sorted list of candidates (best first)
    mask_to_unselected = {}
    for c in unselected:
        mask_to_unselected.setdefault(c.task_mask, []).append(c)
    for mask in mask_to_unselected:
        mask_to_unselected[mask].sort(key=lambda c: (candidate_penalty_cost(c), c.score, -c.willingness))

    for _ in range(4):
        if _now_ms() - start_ms > budget or not _has_time(deadline_ms):
            break
        improved = False
        for ri in range(len(current)):
            if _now_ms() - start_ms > budget or not _has_time(deadline_ms):
                break
            removed = current[ri]
            # Build masks of remaining
            used_couriers = 0
            for j, c in enumerate(current):
                if j != ri:
                    used_couriers |= c.courier_bit
            # Try candidates with same task_mask, sorted by penalty
            removed_penalty = candidate_penalty_cost(removed)
            for uc in mask_to_unselected.get(removed.task_mask, []):
                if uc.courier_bit & used_couriers:
                    continue
                if candidate_penalty_cost(uc) >= removed_penalty - EPS:
                    break  # list is sorted, no better candidates ahead
                new_current = list(current)
                new_current[ri] = uc
                new_eval = evaluate(new_current, total_tasks)
                if is_better(new_eval, current_eval):
                    current = new_current
                    current_eval = new_eval
                    # Update bookkeeping
                    selected_set.discard(id(removed))
                    selected_set.add(id(uc))
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Phase B: remove-one-and-exact-repair (fast version)
    for _ in range(2):
        if _now_ms() - start_ms > budget or not _has_time(deadline_ms):
            break
        improved = False
        for removed in rank_removals(current):
            if _now_ms() - start_ms > budget or not _has_time(deadline_ms):
                break
            freed_mask = removed.task_mask
            removed_score = candidate_penalty_cost(removed)
            locked_couriers = 0
            for c in current:
                if c is not removed:
                    locked_couriers |= c.courier_bit
            replacement = exact_cover_freed(ctx, freed_mask, locked_couriers, removed_score, deadline_ms)
            if replacement is not None:
                candidate = replace_candidates(current, (removed,), replacement)
                candidate_eval = evaluate(candidate, total_tasks)
                if is_better(candidate_eval, current_eval):
                    current = candidate
                    current_eval = candidate_eval
                    improved = True
                    break
        if not improved:
            break

    # Phase C: remove-pair-and-exact-repair
    rounds = [(2, CONFIG["pair_top_k"])]
    if CONFIG.get("try_triples", True):
        rounds.append((3, CONFIG["triple_top_k"]))

    while _now_ms() - start_ms < budget and _has_time(deadline_ms):
        any_improved = False
        for remove_count, top_k in rounds:
            if not _has_time(deadline_ms):
                break
            improved = False
            ranked = rank_removals(current)[: min(top_k, len(current))]
            for removed_tuple in itertools.combinations(ranked, remove_count):
                if not _has_time(deadline_ms):
                    break
                freed_mask = 0
                removed_score = 0.0
                locked_couriers = 0
                removed_set = set(removed_tuple)
                for c in current:
                    if c in removed_set:
                        freed_mask |= c.task_mask
                        removed_score += candidate_penalty_cost(c)
                    else:
                        locked_couriers |= c.courier_bit

                replacement = exact_cover_freed(ctx, freed_mask, locked_couriers, removed_score, deadline_ms)
                if replacement is None:
                    continue
                candidate = replace_candidates(current, removed_tuple, replacement)
                candidate_eval = evaluate(candidate, total_tasks)
                if is_better(candidate_eval, current_eval):
                    current = candidate
                    current_eval = candidate_eval
                    improved = True
                    any_improved = True
                    break
            if not improved:
                continue
        if not any_improved:
            break
    return current, current_eval


def try_ilp(ctx, deadline_ms):
    if not ctx.candidates or len(ctx.candidates) > 120000:
        return None, None
    if not _has_time(deadline_ms, 500.0):
        return None, None

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return None, None

    remaining_seconds = max(0.0, (_remaining(deadline_ms) - 220.0) / 1000.0)
    time_limit = min(CONFIG["ilp_time_limit_seconds"], remaining_seconds)
    if time_limit < 0.2:
        return None, None

    try:
        variable_count = len(ctx.candidates)
        task_rows = len(ctx.task_to_idx)
        courier_rows = len(ctx.courier_to_idx)
        row_count = task_rows + courier_rows
        matrix = lil_matrix((row_count, variable_count), dtype=float)
        score_values = np.zeros(variable_count)
        covered_values = np.zeros(variable_count)

        for col, c in enumerate(ctx.candidates):
            score_values[col] = candidate_penalty_cost(c)
            covered_values[col] = c.task_count
            mask = c.task_mask
            while mask:
                bit = mask & -mask
                matrix[bit.bit_length() - 1, col] = 1.0
                mask ^= bit
            matrix[task_rows + c.courier_idx, col] = 1.0

        objective = score_values - float(CONFIG.get("acceptance_penalty", 100.0)) * covered_values
        constraints = LinearConstraint(
            matrix.tocsr(),
            lb=np.zeros(row_count),
            ub=np.ones(row_count),
        )
        result = milp(
            c=objective,
            integrality=np.ones(variable_count),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0},
        )
        if result.x is None:
            return None, None
        selected = [c for c, value in zip(ctx.candidates, result.x) if value > 0.5]
        ev = evaluate(selected, _count_bits(ctx.all_task_mask))
        if ev.conflicts:
            return None, None
        return selected, ev
    except Exception:
        return None, None


def choose_solution_for_current_penalty(candidates, task_to_idx, courier_to_idx, deadline_ms):
    ctx = build_context(candidates, task_to_idx, courier_to_idx)
    total_tasks = _count_bits(ctx.all_task_mask)
    best = []
    best_eval = None
    repair_orders = []

    for name, ordered in base_orders(ctx):
        selected = greedy_select(ordered)
        ev = evaluate(selected, total_tasks)
        repair_orders.append((name, ordered))
        if is_better(ev, best_eval):
            best = selected
            best_eval = ev

    auto_start = _now_ms()
    for index, spec in enumerate(generated_specs()):
        if index >= CONFIG["max_generated_strategies"]:
            break
        if _now_ms() - auto_start > CONFIG["auto_strategy_budget_ms"]:
            break
        if not _has_time(deadline_ms, 120.0):
            break
        ordered = sorted(ctx.candidates, key=strategy_key(ctx, spec))
        selected = greedy_select(ordered)
        ev = evaluate(selected, total_tasks)
        if len(repair_orders) < 12:
            repair_orders.append(("generated", ordered))
        if is_better(ev, best_eval):
            best = selected
            best_eval = ev

    ilp_selected, ilp_eval = try_ilp(ctx, deadline_ms)
    if ilp_selected is not None and is_better(ilp_eval, best_eval):
        best = ilp_selected
        best_eval = ilp_eval
        if best_eval.covered == total_tasks:
            return best

    improved, improved_eval = local_search(ctx, best, deadline_ms)
    if is_better(improved_eval, best_eval):
        best = improved
        best_eval = improved_eval

    # Second ILP pass if first was skipped and time remains
    if ilp_selected is None and _has_time(deadline_ms, 800.0):
        ilp_selected2, ilp_eval2 = try_ilp(ctx, deadline_ms)
        if ilp_selected2 is not None and is_better(ilp_eval2, best_eval):
            best = ilp_selected2

    return best


def choose_solution(candidates, task_to_idx, courier_to_idx, deadline_ms):
    profiles = list(CONFIG.get("_runtime_penalty_profiles", (CONFIG.get("_runtime_acceptance_penalty", 100.0),)))
    if not profiles:
        profiles = [float(CONFIG.get("acceptance_penalty", 100.0))]

    total_tasks = len(task_to_idx)
    best = None
    best_eval = None
    original_penalty = CONFIG.get("_runtime_acceptance_penalty", CONFIG.get("acceptance_penalty", 100.0))

    for index, penalty in enumerate(profiles):
        if index > 0 and not _has_time(deadline_ms, 1800.0):
            break
        CONFIG["_runtime_acceptance_penalty"] = penalty
        saved_auto_budget = CONFIG.get("auto_strategy_budget_ms", 300.0)
        saved_local_budget = CONFIG.get("local_search_budget_ms", 5000.0)
        saved_ilp_limit = CONFIG.get("ilp_time_limit_seconds", 0.0)
        if index > 0:
            CONFIG["auto_strategy_budget_ms"] = min(float(saved_auto_budget), 160.0)
            CONFIG["local_search_budget_ms"] = min(float(saved_local_budget), 450.0)
            CONFIG["ilp_time_limit_seconds"] = 0.0
        selected = choose_solution_for_current_penalty(candidates, task_to_idx, courier_to_idx, deadline_ms)
        CONFIG["auto_strategy_budget_ms"] = saved_auto_budget
        CONFIG["local_search_budget_ms"] = saved_local_budget
        CONFIG["ilp_time_limit_seconds"] = saved_ilp_limit
        official_eval = evaluate_with_penalty(selected, total_tasks, 100.0)
        if is_better(official_eval, best_eval):
            best = selected
            best_eval = official_eval

    CONFIG["_runtime_acceptance_penalty"] = original_penalty
    return best if best is not None else []


def accept_probability(candidates):
    reject = 1.0
    for c in candidates:
        willingness = min(1.0, max(0.0, c.willingness))
        reject *= 1.0 - willingness
    return 1.0 - reject


def choose_probability_backups(candidates, selected, deadline_ms):
    if not CONFIG.get("enable_multi_courier_output", False):
        return {}
    if not selected or not _has_time(deadline_ms, 120.0):
        return {}

    start_ms = _now_ms()
    penalty = float(CONFIG.get("acceptance_penalty", 100.0))
    max_extra = int(CONFIG.get("max_extra_couriers_per_bundle", 0))
    min_utility = float(CONFIG.get("min_backup_utility", 0.0))
    if penalty <= 0.0 or max_extra <= 0:
        return {}

    by_task_str = {}
    for c in candidates:
        by_task_str.setdefault(c.task_str, []).append(c)
    for items in by_task_str.values():
        items.sort(key=lambda c: (c.score, c.score / max(0.01, c.willingness), -c.willingness))

    selected_ids = set(id(c) for c in selected)
    used_couriers = set(c.courier_id for c in selected)
    backups = {}

    while _has_time(deadline_ms, 80.0) and _now_ms() - start_ms < CONFIG["backup_time_budget_ms"]:
        best_choice = None
        best_utility = min_utility

        for primary in selected:
            current = backups.get(id(primary), [])
            if len(current) >= max_extra:
                continue

            current_probability = accept_probability([primary] + current)
            fail_probability = 1.0 - current_probability
            if fail_probability <= EPS:
                continue

            for backup in by_task_str.get(primary.task_str, ()):
                if id(backup) in selected_ids:
                    continue
                if backup.courier_id in used_couriers:
                    continue

                marginal_expected_tasks = primary.task_count * fail_probability * backup.willingness
                utility = penalty * marginal_expected_tasks - backup.score * backup.willingness
                if utility > best_utility + EPS:
                    best_utility = utility
                    best_choice = (primary, backup)

                if backup.score * backup.willingness > penalty * primary.task_count:
                    break

        if best_choice is None:
            break

        primary, backup = best_choice
        backups.setdefault(id(primary), []).append(backup)
        used_couriers.add(backup.courier_id)

    return backups


def format_solution(selected, backup_map=None):
    backup_map = backup_map or {}
    solution = []
    for c in selected:
        couriers = [c.courier_id]
        couriers.extend(backup.courier_id for backup in backup_map.get(id(c), ()))
        solution.append((c.task_str, couriers))
    return solution


def solve(input_text: str) -> list:
    deadline_ms = _now_ms() + CONFIG["time_budget_ms"] - CONFIG["safety_margin_ms"]
    candidates, task_to_idx, courier_to_idx = parse_input(input_text)
    if not candidates:
        return []
    configure_runtime(candidates, task_to_idx, courier_to_idx)
    selected = choose_solution(candidates, task_to_idx, courier_to_idx, deadline_ms)
    if not CONFIG.get("enable_multi_courier_output", False):
        return format_solution(selected)
    backup_map = choose_probability_backups(candidates, selected, deadline_ms)
    return format_solution(selected, backup_map)
