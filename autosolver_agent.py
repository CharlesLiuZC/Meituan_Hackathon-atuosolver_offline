"""Offline training agent that iterates solver configurations and writes solver.py.

The agent is deliberately local and deterministic: it mutates the strategy
weights in ``solver.CONFIG``, evaluates each candidate on the supplied cases,
keeps the best lexicographic result, and writes a standalone ``solver.py``.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import pprint
import random
import re
import time
from pathlib import Path
from typing import Any

from benchmark import evaluate_output, score_key


DEFAULT_CASE = Path(r"C:\Users\Charles\Desktop\meituan_autosolver-master\example\large_seed301.txt")


def collect_cases(paths: list[str]) -> list[Path]:
    if not paths:
        paths = [str(DEFAULT_CASE)]
    cases: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            cases.extend(sorted(path.glob("*.txt")))
        elif path.is_file():
            cases.append(path)
    if not cases:
        raise FileNotFoundError("No .txt cases found.")
    return cases


def case_weight(case_name: str) -> float:
    case_name = case_name.lower()
    if "low_willingness" in case_name:
        return 3.0
    if "scarce_couriers" in case_name or "scarce" in case_name:
        return 2.2
    if "high_noise" in case_name:
        return 1.4
    return 1.0


def average_rank(results: list[dict[str, Any]]) -> tuple[int, float, float, float]:
    invalid = sum(0 if item.get("valid") else 1 for item in results)
    penalty_score = sum(
        case_weight(str(item.get("case", ""))) * float(item["penalty_score"])
        for item in results
    )
    covered_gap = sum(
        case_weight(str(item.get("case", ""))) * (item["total_tasks"] - item["covered_tasks"])
        for item in results
    )
    elapsed_ms = sum(float(item["elapsed_ms"]) for item in results)
    return invalid, penalty_score, covered_gap, elapsed_ms


def evaluate_config(config: dict[str, Any], cases: list[Path]) -> dict[str, Any]:
    solver = importlib.import_module("solver")
    config = copy.deepcopy(config)
    config["enable_multi_courier_output"] = False
    config["acceptance_penalty"] = 100.0
    config["dynamic_penalty"] = False
    config["objective_mode"] = "expected_penalty"
    solver.CONFIG = copy.deepcopy(config)
    results = []
    for case in cases:
        input_text = case.read_text(encoding="utf-8")
        start = time.perf_counter()
        solution = solver.solve(input_text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = evaluate_output(input_text, solution)
        result["case"] = str(case)
        result["elapsed_ms"] = round(elapsed_ms, 3)
        results.append(result)
    return {
        "rank": average_rank(results),
        "results": results,
        "config": copy.deepcopy(config),
    }


def mutate_strategy(strategy: tuple[float, ...], rng: random.Random) -> tuple[Any, ...]:
    values = list(strategy)
    for index in range(6):
        scale = 0.05 if index != 1 else 0.08
        if rng.random() < 0.65:
            values[index] = max(0.0, float(values[index]) + rng.uniform(-scale, scale))
    if rng.random() < 0.12:
        values[6] = 1 - int(values[6])
    return tuple(round(float(v), 4) if i < 6 else int(v) for i, v in enumerate(values))


def mutate_config(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    config = copy.deepcopy(base)
    strategies = list(config["strategies"])
    action = rng.random()
    if action < 0.55 and strategies:
        index = rng.randrange(len(strategies))
        strategies[index] = mutate_strategy(tuple(strategies[index]), rng)
    elif action < 0.80 and strategies:
        parent = tuple(rng.choice(strategies))
        if len(strategies) < 18:
            strategies.append(mutate_strategy(parent, rng))
    elif len(strategies) > 4:
        del strategies[rng.randrange(len(strategies))]

    config["strategies"] = strategies
    config["max_generated_strategies"] = int(
        max(16, min(96, config["max_generated_strategies"] + rng.choice([-8, -4, 0, 4, 8])))
    )
    config["max_candidates_per_mask"] = int(
        max(20, min(80, config["max_candidates_per_mask"] + rng.choice([-8, -4, 0, 4, 8])))
    )
    config["pair_top_k"] = int(max(16, min(48, config["pair_top_k"] + rng.choice([-4, 0, 4]))))
    config["triple_top_k"] = int(max(10, min(32, config["triple_top_k"] + rng.choice([-4, 0, 4]))))
    # This is part of the official scoring function, not a trainable weight.
    config["acceptance_penalty"] = 100.0
    config["dynamic_penalty"] = False
    if rng.random() < 0.25:
        config["max_extra_couriers_per_bundle"] = int(
            max(0, min(5, int(config.get("max_extra_couriers_per_bundle", 0)) + rng.choice([-1, 1])))
        )
    config["enable_multi_courier_output"] = False
    if rng.random() < 0.30:
        config["backup_time_budget_ms"] = float(
            max(80.0, min(900.0, float(config.get("backup_time_budget_ms", 300.0)) + rng.choice([-100.0, -50.0, 50.0, 100.0])))
        )
    if rng.random() < 0.35:
        config["min_backup_utility"] = round(
            max(-30.0, min(60.0, float(config.get("min_backup_utility", 0.0)) + rng.choice([-10.0, -5.0, 5.0, 10.0]))),
            3,
        )
    return config


def config_signature(config: dict[str, Any]) -> str:
    stable = {
        "strategies": config["strategies"],
        "max_generated_strategies": config["max_generated_strategies"],
        "max_candidates_per_mask": config["max_candidates_per_mask"],
        "pair_top_k": config["pair_top_k"],
        "triple_top_k": config["triple_top_k"],
        "acceptance_penalty": config.get("acceptance_penalty"),
        "max_extra_couriers_per_bundle": config.get("max_extra_couriers_per_bundle"),
        "backup_time_budget_ms": config.get("backup_time_budget_ms"),
        "min_backup_utility": config.get("min_backup_utility"),
        "enable_multi_courier_output": config.get("enable_multi_courier_output"),
    }
    return json.dumps(stable, sort_keys=True)


def replace_config_in_source(source: str, config: dict[str, Any]) -> str:
    marker = "\n\nclass Candidate:"
    prefix, sep, suffix = source.partition(marker)
    if not sep:
        raise ValueError("Could not locate CONFIG block in solver.py.")
    prefix = re.sub(
        r"CONFIG\s*=\s*\{.*\}\s*$",
        "CONFIG = " + pprint.pformat(config, width=100, sort_dicts=False),
        prefix,
        flags=re.S,
    )
    return prefix + marker + suffix


def write_solver(config: dict[str, Any], solver_path: Path) -> None:
    source = solver_path.read_text(encoding="utf-8")
    solver_path.write_text(replace_config_in_source(source, config), encoding="utf-8")


def train(cases: list[Path], iterations: int, seed: int, solver_path: Path) -> dict[str, Any]:
    solver = importlib.import_module("solver")
    base_config = copy.deepcopy(solver.CONFIG)
    rng = random.Random(seed)

    best = evaluate_config(base_config, cases)
    history = [best]
    seen = {config_signature(base_config)}
    print(f"[0] rank={best['rank']} config=current")

    for iteration in range(1, iterations + 1):
        parent = best["config"] if rng.random() < 0.75 else rng.choice(history)["config"]
        candidate_config = mutate_config(parent, rng)
        signature = config_signature(candidate_config)
        if signature in seen:
            continue
        seen.add(signature)
        candidate = evaluate_config(candidate_config, cases)
        history.append(candidate)
        improved = candidate["rank"] < best["rank"]
        status = "KEEP" if improved else "drop"
        print(f"[{iteration}] rank={candidate['rank']} {status}")
        if improved:
            best = candidate
            write_solver(best["config"], solver_path)

    write_solver(best["config"], solver_path)
    return {
        "best_rank": best["rank"],
        "best_results": best["results"],
        "iterations": iterations,
        "cases": [str(case) for case in cases],
        "history_size": len(history),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case files or directories. Defaults to the provided sample.")
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--solver", default="solver.py")
    parser.add_argument("--summary", default="training_summary.json")
    args = parser.parse_args()

    cases = collect_cases(args.cases)
    summary = train(cases, args.iterations, args.seed, Path(args.solver))
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
