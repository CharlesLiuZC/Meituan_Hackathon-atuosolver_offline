"""Evaluation helpers for task-courier assignment solutions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from common.parser import parse_input


DEFAULT_MISSING_TASK_PENALTY = 100.0


def _split_tasks(task_id_list_str: str) -> list[str]:
    return [task.strip() for task in task_id_list_str.split(",") if task.strip()]


def _build_candidate_lookup(
    input_text: str,
) -> tuple[dict[tuple[str, str], float], set[str]]:
    candidates = parse_input(input_text)
    candidate_scores: dict[tuple[str, str], float] = {}
    all_tasks: set[str] = set()

    for score, task_str, courier_id, _willingness in candidates:
        candidate_key = (task_str, courier_id)
        if candidate_key in candidate_scores:
            candidate_scores[candidate_key] = min(candidate_scores[candidate_key], score)
        else:
            candidate_scores[candidate_key] = score
        all_tasks.update(_split_tasks(task_str))

    return candidate_scores, all_tasks


def evaluate_solution(
    input_text: str,
    solution: list,
    missing_task_penalty: float = DEFAULT_MISSING_TASK_PENALTY,
) -> dict[str, Any]:
    """Evaluate a solver output.

    The returned dict separates hard validity checks from the soft objective:

    objective_score = total_score + missing_tasks * missing_task_penalty

    Lower objective_score is better, but only valid solutions should be compared.
    """
    candidate_scores, all_tasks = _build_candidate_lookup(input_text)

    task_occurrences: Counter[str] = Counter()
    courier_occurrences: Counter[str] = Counter()
    invalid_candidates: list[dict[str, Any]] = []
    malformed_items: list[dict[str, Any]] = []

    total_score = 0.0
    valid_assignments = 0

    if not isinstance(solution, list):
        malformed_items.append({
            "index": None,
            "item": repr(solution),
            "reason": "solution must be a list",
        })
        solution_items = []
    else:
        solution_items = solution

    for index, item in enumerate(solution_items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            malformed_items.append({
                "index": index,
                "item": repr(item),
                "reason": "each solution item must be (task_id_list_str, courier_ids)",
            })
            continue

        task_str, courier_ids = item
        if not isinstance(task_str, str):
            malformed_items.append({
                "index": index,
                "item": repr(item),
                "reason": "task_id_list_str must be a string",
            })
            continue
        if not isinstance(courier_ids, list) or not courier_ids:
            malformed_items.append({
                "index": index,
                "item": repr(item),
                "reason": "courier_ids must be a non-empty list",
            })
            continue

        task_ids = _split_tasks(task_str)
        if not task_ids:
            malformed_items.append({
                "index": index,
                "item": repr(item),
                "reason": "task_id_list_str must contain at least one task",
            })
            continue

        for task_id in task_ids:
            task_occurrences[task_id] += 1

        for courier_id in courier_ids:
            if not isinstance(courier_id, str):
                malformed_items.append({
                    "index": index,
                    "item": repr(item),
                    "reason": "courier_id must be a string",
                })
                continue

            courier_occurrences[courier_id] += 1
            candidate_key = (task_str, courier_id)
            score = candidate_scores.get(candidate_key)
            if score is None:
                invalid_candidates.append({
                    "index": index,
                    "task_id_list": task_str,
                    "courier_id": courier_id,
                })
                continue

            total_score += score
            valid_assignments += 1

    covered_known_tasks = set(task_occurrences) & all_tasks
    unknown_tasks = sorted(set(task_occurrences) - all_tasks)
    missing_task_ids = sorted(all_tasks - covered_known_tasks)
    duplicate_tasks = sorted(
        task_id for task_id, count in task_occurrences.items() if count > 1
    )
    duplicate_couriers = sorted(
        courier_id for courier_id, count in courier_occurrences.items() if count > 1
    )

    total_tasks = len(all_tasks)
    covered_tasks = len(covered_known_tasks)
    missing_tasks = len(missing_task_ids)
    objective_score = total_score + missing_tasks * missing_task_penalty

    task_conflict = bool(duplicate_tasks)
    courier_conflict = bool(duplicate_couriers)
    has_invalid_candidates = bool(invalid_candidates)
    has_malformed_items = bool(malformed_items)
    has_unknown_tasks = bool(unknown_tasks)

    valid = not (
        task_conflict
        or courier_conflict
        or has_invalid_candidates
        or has_malformed_items
        or has_unknown_tasks
    )

    return {
        "valid": valid,
        "total_tasks": total_tasks,
        "covered_tasks": covered_tasks,
        "missing_tasks": missing_tasks,
        "missing_task_ids": missing_task_ids,
        "total_score": round(total_score, 4),
        "missing_task_penalty": float(missing_task_penalty),
        "objective_score": round(objective_score, 4),
        "task_conflict": task_conflict,
        "duplicate_tasks": duplicate_tasks,
        "courier_conflict": courier_conflict,
        "duplicate_couriers": duplicate_couriers,
        "invalid_candidate_count": len(invalid_candidates),
        "invalid_candidates": invalid_candidates,
        "malformed_item_count": len(malformed_items),
        "malformed_items": malformed_items,
        "unknown_tasks": unknown_tasks,
        "solution_items": len(solution_items),
        "valid_assignments": valid_assignments,
    }
