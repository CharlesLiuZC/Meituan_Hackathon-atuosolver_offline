r"""Benchmark helper for AutoSolver candidates.

Usage:
    python benchmark.py C:\path\to\case.txt
    python benchmark.py C:\path\to\data_dir
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any


def split_tasks(task_str: str) -> list[str]:
    return [task.strip() for task in task_str.split(",") if task.strip()]


def load_candidate_lookup(input_text: str) -> tuple[dict[tuple[str, str], tuple[float, float]], set[str]]:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    all_tasks: set[str] = set()
    for line in lines[start:]:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        task_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        key = (task_str.strip(), courier_id.strip())
        if key not in lookup or score < lookup[key][0]:
            lookup[key] = (score, willingness)
        all_tasks.update(split_tasks(task_str))
    return lookup, all_tasks


def evaluate_output(input_text: str, solution: list[Any]) -> dict[str, Any]:
    lookup, all_tasks = load_candidate_lookup(input_text)
    task_count: dict[str, int] = {}
    courier_count: dict[str, int] = {}
    total_score = 0.0
    parallel_penalty_score = 0.0
    penalty_score = 0.0
    expected_accepted_tasks = 0.0
    invalid = []
    malformed = []

    if not isinstance(solution, list):
        return {
            "valid": False,
            "error": "solution must be a list",
            "covered_tasks": 0,
            "total_tasks": len(all_tasks),
            "total_score": float("inf"),
            "penalty_score": float("inf"),
        }

    for index, item in enumerate(solution):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            malformed.append(index)
            continue
        task_str, courier_ids = item
        if not isinstance(task_str, str) or not isinstance(courier_ids, list):
            malformed.append(index)
            continue
        for task_id in split_tasks(task_str):
            task_count[task_id] = task_count.get(task_id, 0) + 1
        task_ids = split_tasks(task_str)
        reject_probability = 1.0
        weighted_score = 0.0
        first_accept_score = 0.0
        for courier_id in courier_ids:
            if not isinstance(courier_id, str):
                malformed.append(index)
                continue
            courier_count[courier_id] = courier_count.get(courier_id, 0) + 1
            candidate = lookup.get((task_str, courier_id))
            if candidate is None:
                invalid.append({"index": index, "task_str": task_str, "courier_id": courier_id})
            else:
                score, willingness = candidate
                total_score += score
                willingness = max(0.0, min(1.0, willingness))
                weighted_score += score * willingness
                first_accept_score += reject_probability * score * willingness
                reject_probability *= 1.0 - willingness
        accepted_probability = 1.0 - reject_probability
        expected_accepted_tasks += len(task_ids) * accepted_probability
        parallel_penalty_score += weighted_score + 100.0 * len(task_ids) * reject_probability
        penalty_score += first_accept_score + 100.0 * len(task_ids) * reject_probability

    covered = set(task_count) & all_tasks
    duplicate_tasks = sorted(task for task, count in task_count.items() if count > 1)
    duplicate_couriers = sorted(c for c, count in courier_count.items() if count > 1)
    unknown_tasks = sorted(set(task_count) - all_tasks)
    missing = sorted(all_tasks - covered)
    parallel_penalty_score += 100.0 * len(missing)
    penalty_score += 100.0 * len(missing)
    valid = not (duplicate_tasks or duplicate_couriers or unknown_tasks or invalid or malformed)

    return {
        "valid": valid,
        "covered_tasks": len(covered),
        "total_tasks": len(all_tasks),
        "missing_tasks": len(missing),
        "total_score": round(total_score, 6),
        "penalty_score": round(penalty_score, 6),
        "parallel_penalty_score": round(parallel_penalty_score, 6),
        "expected_accepted_tasks": round(expected_accepted_tasks, 6),
        "solution_items": len(solution),
        "duplicate_tasks": duplicate_tasks,
        "duplicate_couriers": duplicate_couriers,
        "unknown_tasks": unknown_tasks,
        "invalid_candidate_count": len(invalid),
        "malformed_item_count": len(malformed),
    }


def score_key(result: dict[str, Any]) -> tuple[int, float, int, float]:
    return (
        0 if result.get("valid") else 1,
        float(result.get("penalty_score", float("inf"))),
        int(result.get("total_tasks", 0)) - int(result.get("covered_tasks", 0)),
        float(result.get("total_score", float("inf"))),
    )


def collect_cases(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.txt"))


def benchmark_case(case_path: Path, module_name: str = "solver") -> dict[str, Any]:
    solver = importlib.import_module(module_name)
    importlib.reload(solver)
    input_text = case_path.read_text(encoding="utf-8")
    start = time.perf_counter()
    solution = solver.solve(input_text)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = evaluate_output(input_text, solution)
    result["case"] = str(case_path)
    result["elapsed_ms"] = round(elapsed_ms, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="A .txt case file or a directory containing cases.")
    parser.add_argument("--module", default="solver", help="Solver module name. Default: solver.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args()

    cases = collect_cases(Path(args.path))
    results = [benchmark_case(case, args.module) for case in cases]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for result in results:
        print(
            f"{Path(result['case']).name}: "
            f"valid={result['valid']} "
            f"covered={result['covered_tasks']}/{result['total_tasks']} "
            f"penalty={result['penalty_score']} "
            f"parallel_penalty={result.get('parallel_penalty_score', result['penalty_score'])} "
            f"raw_score={result['total_score']} "
            f"items={result['solution_items']} "
            f"time_ms={result['elapsed_ms']}"
        )
    if results:
        valid_count = sum(1 for result in results if result["valid"])
        avg_penalty = sum(float(result["penalty_score"]) for result in results) / len(results)
        avg_time = sum(float(result["elapsed_ms"]) for result in results) / len(results)
        avg_score = sum(float(result["total_score"]) for result in results) / len(results)
        print(
            f"SUMMARY: valid={valid_count}/{len(results)} "
            f"avg_penalty={avg_penalty:.6f} "
            f"avg_raw_score={avg_score:.6f} "
            f"avg_time_ms={avg_time:.3f}"
        )


if __name__ == "__main__":
    main()
