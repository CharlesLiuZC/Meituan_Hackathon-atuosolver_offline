"""Long-running offline trainer for the platform-validated v3 race solver.

The platform champion in solver.py is never overwritten. The trainer mutates
search policies in memory, evaluates them under the v3 unordered-race model,
and writes the strongest offline challenger to solver_trained_best.py.

Version 2 expands training to the synthetic 1500-scene case bank and uses a
target-weighted promotion rule.  The online weak spots are low_willingness and
scarce_couriers, so a candidate can become the offline challenger when those
hard scenes improve materially while the full validation set stays sane.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pprint
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from benchmark import evaluate_output


ROOT = Path(__file__).resolve().parent
BASELINE_SOLVER = ROOT / "solver.py"
PLATFORM_BACKUP = ROOT / "solver_platform_best_v3.py"
OUTPUT_SOLVER = ROOT / "solver_trained_best.py"
STATE_PATH = ROOT / "training_24h_state.json"
LOG_PATH = ROOT / "training_24h_history.jsonl"
TARGET_SOLVER = ROOT / "solver_low_scarce_challenger.py"
CASE_BANK_ROOT = ROOT / "training_cases" / "train"
AUTO_CASE_ROOT = ROOT / "training_cases_auto"
HARD_CASE_ROOT = ROOT / "training_cases_hard"
EPS = 1e-9

TARGET_KINDS = {"low_willingness", "scarce_couriers"}
KIND_WEIGHTS = {
    "low_willingness": 3.4,
    "scarce_couriers": 3.2,
    "normal": 1.0,
}
SCENE_WEIGHTS = {
    "low_willingness_seed501": 4.0,
    "scarce_couriers_seed401": 3.8,
    "large_seed301": 1.3,
    "large_seed302": 1.3,
}

GLOBAL_KEYS = (
    "auto_strategy_budget_ms",
    "local_search_budget_ms",
    "max_generated_strategies",
    "max_candidates_per_mask",
    "special_max_candidates_per_mask",
    "pair_top_k",
    "triple_top_k",
    "try_triples",
    "strategies",
)

BASE_RUNTIME = {
    "low_willingness": {
        "auto_strategy_budget_ms": 180.0,
        "local_search_budget_ms": 0.0,
        "multi_primary_time_budget_ms": 1100.0,
        "backup_time_budget_ms": 4600.0,
        "backup_reallocation_budget_ms": 2300.0,
        "max_extra_couriers_per_bundle": 8,
        "_runtime_multi_cost_mode": "race",
    },
    "scarce_couriers": {
        "auto_strategy_budget_ms": 300.0,
        "local_search_budget_ms": 2800.0,
        "multi_primary_time_budget_ms": 0.0,
        "backup_time_budget_ms": 1400.0,
        "backup_reallocation_budget_ms": 500.0,
        "max_extra_couriers_per_bundle": 5,
        "_runtime_multi_cost_mode": "race",
    },
    "normal": {
        "multi_primary_time_budget_ms": 3200.0,
        "backup_time_budget_ms": 900.0,
        "backup_reallocation_budget_ms": 320.0,
        "max_extra_couriers_per_bundle": 5,
        "_runtime_multi_cost_mode": "race",
    },
}

RUNTIME_CHOICES = {
    "low_willingness": {
        "auto_strategy_budget_ms": (100.0, 140.0, 180.0, 240.0, 320.0),
        "local_search_budget_ms": (0.0, 120.0, 240.0, 360.0, 520.0),
        "multi_primary_time_budget_ms": (650.0, 850.0, 1050.0, 1100.0, 1300.0, 1550.0),
        "backup_time_budget_ms": (3000.0, 3600.0, 4200.0, 4600.0, 5000.0, 5400.0),
        "backup_reallocation_budget_ms": (700.0, 1100.0, 1600.0, 2100.0, 2600.0, 3100.0),
        "max_extra_couriers_per_bundle": (5, 6, 7, 8, 9, 10, 12),
        "min_backup_utility": (-3.0, -1.0, -0.25, 0.0, 0.05),
        "_runtime_multi_cost_mode": ("race", "sequential"),
    },
    "scarce_couriers": {
        "auto_strategy_budget_ms": (90.0, 150.0, 220.0, 300.0, 420.0, 560.0),
        "local_search_budget_ms": (900.0, 1400.0, 1900.0, 2400.0, 2800.0, 3400.0, 4200.0),
        "backup_time_budget_ms": (350.0, 600.0, 850.0, 1100.0, 1400.0, 1750.0, 2200.0),
        "backup_reallocation_budget_ms": (0.0, 180.0, 360.0, 500.0, 800.0, 1200.0),
        "max_extra_couriers_per_bundle": (2, 3, 4, 5, 6, 7, 9),
        "min_backup_utility": (-1.0, -0.25, 0.0, 0.05, 0.25),
        "_runtime_multi_cost_mode": ("race", "sequential"),
    },
    "normal": {
        "multi_primary_time_budget_ms": (2400.0, 2800.0, 3200.0, 3600.0),
        "backup_time_budget_ms": (550.0, 700.0, 900.0, 1100.0),
        "backup_reallocation_budget_ms": (0.0, 160.0, 320.0, 520.0),
        "max_extra_couriers_per_bundle": (3, 4, 5, 6),
        "min_backup_utility": (0.0, 0.05),
    },
}


def load_module(path: Path, suffix: str):
    name = "trained_solver_" + suffix.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_profile() -> dict[str, Any]:
    module = load_module(BASELINE_SOLVER, "baseline_init")
    return {
        "global": {key: copy.deepcopy(module.CONFIG[key]) for key in GLOBAL_KEYS},
        "runtime": copy.deepcopy(BASE_RUNTIME),
    }


def profiled_module(profile: dict[str, Any], suffix: str):
    module = load_module(BASELINE_SOLVER, suffix)
    module.CONFIG.update(copy.deepcopy(profile["global"]))
    original_apply = module.apply_runtime_overrides

    def apply_runtime_overrides():
        saved = original_apply()
        case_type = module.CONFIG.get("_runtime_case_type", "normal")
        overrides = profile["runtime"].get(case_type, {})
        for key, value in overrides.items():
            if key not in saved:
                saved[key] = module.CONFIG.get(key)
            module.CONFIG[key] = value
        return saved

    module.apply_runtime_overrides = apply_runtime_overrides
    return module


def case_kind(path: Path) -> str:
    name = str(path).lower()
    if "low_willingness" in name:
        return "low_willingness"
    if "scarce_couriers" in name:
        return "scarce_couriers"
    return "normal"


def case_scene(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    for scene in SCENE_WEIGHTS:
        if scene in text:
            return scene
    parent = path.parent.name.lower()
    if parent and parent not in {"training_cases", "training_cases_auto", "hackthon data"}:
        return parent
    stem = path.stem.lower()
    if "low_willingness" in stem:
        return "low_willingness"
    if "scarce_couriers" in stem:
        return "scarce_couriers"
    if "high_noise" in stem:
        return "high_noise"
    if "large_seed301" in stem:
        return "large_seed301"
    if "large_seed302" in stem:
        return "large_seed302"
    if "medium" in stem:
        return "medium"
    if "small" in stem:
        return "small"
    if "tiny" in stem:
        return "tiny"
    return case_kind(path)


def case_weight(path: Path) -> float:
    scene = case_scene(path)
    if scene in SCENE_WEIGHTS:
        return SCENE_WEIGHTS[scene]
    return KIND_WEIGHTS.get(case_kind(path), 1.0)


def score_solution_race(module, input_text: str, solution: list) -> dict[str, Any]:
    report = evaluate_output(input_text, solution)
    if not report["valid"]:
        return {"valid": False, "covered": 0, "total": report["total_tasks"], "race": float("inf")}

    candidates, task_to_idx, courier_to_idx = module.parse_input(input_text)
    lookup = {}
    for candidate in candidates:
        key = (candidate.task_str, candidate.courier_id)
        previous = lookup.get(key)
        if previous is None or candidate.score < previous.score:
            lookup[key] = candidate

    module.configure_runtime(candidates, task_to_idx, courier_to_idx)
    saved = module.apply_runtime_overrides()
    try:
        race_cost = 0.0
        for task_str, courier_ids in solution:
            group = [lookup[(task_str, courier_id)] for courier_id in courier_ids]
            race_cost += module.multi_group_penalty(group)
        race_cost += 100.0 * report["missing_tasks"]
    finally:
        module.restore_runtime_overrides(saved)
    return {
        "valid": True,
        "covered": report["covered_tasks"],
        "total": report["total_tasks"],
        "race": race_cost,
        "raw": report["total_score"],
    }


def evaluate_profile(profile: dict[str, Any], cases: list[Path], label: str) -> dict[str, Any]:
    module = profiled_module(profile, label + "_" + str(time.time_ns()))
    results = []
    for path in cases:
        input_text = path.read_text(encoding="utf-8")
        started = time.perf_counter()
        solution = module.solve(input_text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = score_solution_race(module, input_text, solution)
        metrics.update({
            "case": str(path),
            "kind": case_kind(path),
            "scene": case_scene(path),
            "weight": case_weight(path),
            "elapsed_ms": round(elapsed_ms, 3),
        })
        results.append(metrics)
    invalid = sum(0 if result["valid"] else 1 for result in results)
    missing = sum(result["total"] - result["covered"] for result in results)
    weight_sum = sum(float(result.get("weight", 1.0)) for result in results) or 1.0
    mean_race = sum(result["race"] * float(result.get("weight", 1.0)) for result in results) / weight_sum
    unweighted_mean_race = sum(result["race"] for result in results) / max(1, len(results))
    target_results = [result for result in results if result["kind"] in TARGET_KINDS]
    target_mean_race = (
        sum(result["race"] for result in target_results) / len(target_results)
        if target_results else float("inf")
    )
    max_time = max((result["elapsed_ms"] for result in results), default=0.0)
    by_kind: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = by_kind.setdefault(result["kind"], {"count": 0, "race_sum": 0.0, "time_max": 0.0})
        bucket["count"] += 1
        bucket["race_sum"] += result["race"]
        bucket["time_max"] = max(bucket["time_max"], result["elapsed_ms"])
    for bucket in by_kind.values():
        bucket["mean_race"] = bucket["race_sum"] / max(1, bucket["count"])
        bucket.pop("race_sum", None)
    return {
        "invalid": invalid,
        "missing": missing,
        "mean_race": mean_race,
        "unweighted_mean_race": unweighted_mean_race,
        "target_mean_race": target_mean_race,
        "max_time_ms": max_time,
        "by_kind": by_kind,
        "results": results,
    }


def metric_key(summary: dict[str, Any]) -> tuple[int, int, float]:
    return summary["invalid"], summary["missing"], summary["mean_race"]


def target_metric_key(summary: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        summary["invalid"],
        summary["missing"],
        summary.get("target_mean_race", float("inf")),
        summary["mean_race"],
    )


def mutate_strategy(strategy: tuple, rng: random.Random) -> tuple:
    values = list(strategy)
    for index in range(6):
        if rng.random() < 0.6:
            scale = 0.09 if index == 1 else 0.055
            values[index] = max(0.0, float(values[index]) + rng.uniform(-scale, scale))
    if rng.random() < 0.08:
        values[6] = 1 - int(values[6])
    return tuple(round(value, 4) if index < 6 else int(value) for index, value in enumerate(values))


def mutate_profile(parent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    profile = copy.deepcopy(parent)
    operation = rng.random()
    if operation < 0.48:
        case_type = rng.choice((
            "low_willingness",
            "low_willingness",
            "scarce_couriers",
            "scarce_couriers",
            "scarce_couriers",
            "normal",
        ))
        key = rng.choice(list(RUNTIME_CHOICES[case_type]))
        profile["runtime"][case_type][key] = rng.choice(RUNTIME_CHOICES[case_type][key])
    elif operation < 0.73:
        key = rng.choice((
            "max_generated_strategies",
            "max_candidates_per_mask",
            "special_max_candidates_per_mask",
            "pair_top_k",
            "triple_top_k",
            "try_triples",
        ))
        choices = {
            "max_generated_strategies": (8, 12, 16, 20, 24, 32, 40),
            "max_candidates_per_mask": (12, 16, 20, 24, 32, 40, 52),
            "special_max_candidates_per_mask": (2, 4, 6, 8, 10, 12),
            "pair_top_k": (16, 22, 28, 34, 40, 50),
            "triple_top_k": (8, 10, 14, 20, 26, 32),
            "try_triples": (True, False),
        }
        profile["global"][key] = rng.choice(choices[key])
    else:
        strategies = list(profile["global"]["strategies"])
        if rng.random() < 0.2 and len(strategies) < 10:
            strategies.append(mutate_strategy(tuple(rng.choice(strategies)), rng))
        else:
            index = rng.randrange(len(strategies))
            strategies[index] = mutate_strategy(tuple(strategies[index]), rng)
        profile["global"]["strategies"] = strategies
    return profile


def profile_signature(profile: dict[str, Any]) -> str:
    return json.dumps(profile, sort_keys=True, separators=(",", ":"))


def materialize_solver(base_source: str, profile: dict[str, Any], output_path: Path) -> None:
    injected = (
        "\n_TRAINED_CONFIG_OVERRIDES = "
        + pprint.pformat(profile["global"], width=100, sort_dicts=False)
        + "\nCONFIG.update(_TRAINED_CONFIG_OVERRIDES)\n"
        + "_TRAINED_RUNTIME_OVERRIDES = "
        + pprint.pformat(profile["runtime"], width=100, sort_dicts=False)
        + "\n"
    )
    source = base_source.replace("\nclass Candidate:", injected + "\nclass Candidate:", 1)
    old = "    saved = {key: CONFIG.get(key) for key in overrides}\n    CONFIG.update(overrides)"
    new = (
        "    overrides.update(_TRAINED_RUNTIME_OVERRIDES.get(case_type, {}))\n"
        "    saved = {key: CONFIG.get(key) for key in overrides}\n"
        "    CONFIG.update(overrides)"
    )
    if old not in source:
        raise ValueError("Could not patch apply_runtime_overrides in the baseline solver.")
    output_path.write_text(source.replace(old, new, 1), encoding="utf-8")


def unique_paths(paths: list[Path]) -> list[Path]:
    out = []
    seen = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def discover_cases() -> tuple[list[Path], dict[str, list[Path]], dict[str, list[Path]]]:
    fixed = [
        ROOT / "Hackthon Data" / "large_seed301.txt",
        ROOT / "training_cases" / "synthetic_high_noise_30_seed601.txt",
        ROOT / "training_cases" / "synthetic_medium_30_seed201.txt",
        ROOT / "training_cases" / "synthetic_low_willingness_30_seed501.txt",
        ROOT / "training_cases" / "synthetic_low_willingness_30_seed502.txt",
        ROOT / "training_cases" / "synthetic_scarce_couriers_40_seed401.txt",
        ROOT / "training_cases" / "synthetic_scarce_couriers_40_seed402.txt",
    ]
    fixed = [path for path in fixed if path.exists()]
    pools = {"normal": [], "low_willingness": [], "scarce_couriers": []}
    scene_pools: dict[str, list[Path]] = {}

    def add_case(path: Path) -> None:
        if not path.exists() or path.suffix.lower() != ".txt":
            return
        kind = case_kind(path)
        pools.setdefault(kind, []).append(path)
        scene_pools.setdefault(case_scene(path), []).append(path)

    for path in fixed:
        add_case(path)
    for path in sorted(AUTO_CASE_ROOT.glob("*.txt")) if AUTO_CASE_ROOT.exists() else []:
        add_case(path)
    for path in sorted(CASE_BANK_ROOT.rglob("*.txt")) if CASE_BANK_ROOT.exists() else []:
        add_case(path)
    for path in sorted(HARD_CASE_ROOT.rglob("*.txt")) if HARD_CASE_ROOT.exists() else []:
        add_case(path)

    for kind, items in list(pools.items()):
        pools[kind] = unique_paths(sorted(items))
    for scene, items in list(scene_pools.items()):
        scene_pools[scene] = unique_paths(sorted(items))
    return unique_paths(fixed), pools, scene_pools


def spaced_samples(items: list[Path], count: int) -> list[Path]:
    if not items:
        return []
    return [items[(index + 1) * len(items) // (count + 1)] for index in range(min(count, len(items)))]


def random_samples(items: list[Path], count: int, rng: random.Random) -> list[Path]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    return rng.sample(items, count)


def build_validation(fixed: list[Path], pools: dict[str, list[Path]], scene_pools: dict[str, list[Path]]) -> list[Path]:
    validation = list(fixed)
    # The scene bank mirrors the ten platform rows.  Keep every scene present,
    # but heavily sample the two scenes that still dominate online regret.
    for scene in (
        "high_noise_seed601",
        "large_seed301",
        "large_seed302",
        "medium_seed201",
        "medium_seed202",
        "medium_seed203",
        "small_seed100",
        "tiny_seed42",
    ):
        validation.extend(spaced_samples(scene_pools.get(scene, []), 2))
    validation.extend(spaced_samples(scene_pools.get("low_willingness_seed501", []), 8))
    validation.extend(spaced_samples(scene_pools.get("scarce_couriers_seed401", []), 8))
    validation.extend(spaced_samples(pools.get("low_willingness", []), 4))
    validation.extend(spaced_samples(pools.get("scarce_couriers", []), 4))
    validation.extend(spaced_samples(pools.get("normal", []), 4))
    return unique_paths([path for path in validation if path.exists()])


def build_quick_cases(
    fixed: list[Path],
    pools: dict[str, list[Path]],
    scene_pools: dict[str, list[Path]],
    rng: random.Random,
    iteration: int,
) -> list[Path]:
    anchors = [
        ROOT / "Hackthon Data" / "large_seed301.txt",
        ROOT / "training_cases" / "synthetic_low_willingness_30_seed501.txt",
        ROOT / "training_cases" / "synthetic_scarce_couriers_40_seed401.txt",
    ]
    cases = [path for path in anchors if path.exists()]
    cases.extend(random_samples(scene_pools.get("low_willingness_seed501", []), 1, rng))
    cases.extend(random_samples(scene_pools.get("scarce_couriers_seed401", []), 1, rng))
    if iteration % 3 == 0:
        cases.extend(random_samples(pools.get("normal", []), 1, rng))
    elif iteration % 3 == 1:
        cases.extend(random_samples(pools.get("low_willingness", []), 1, rng))
    else:
        cases.extend(random_samples(pools.get("scarce_couriers", []), 1, rng))
    return unique_paths([path for path in cases if path.exists()])


def write_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-iterations", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deadline = time.time() + args.hours * 3600.0
    source = BASELINE_SOLVER.read_text(encoding="utf-8")
    if not PLATFORM_BACKUP.exists():
        shutil.copy2(BASELINE_SOLVER, PLATFORM_BACKUP)
    fixed, pools, scene_pools = discover_cases()
    validation = build_validation(fixed, pools, scene_pools)
    champion = initial_profile()
    best_validation = evaluate_profile(champion, validation, "validation_base")
    best_target_validation = best_validation
    materialize_solver(source, champion, OUTPUT_SOLVER)
    materialize_solver(source, champion, TARGET_SOLVER)
    iteration = 0
    accepted = 0
    seen = {profile_signature(champion)}

    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "deadline_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)),
        "iterations": iteration,
        "accepted": accepted,
        "case_counts": {
            "fixed": len(fixed),
            "normal": len(pools.get("normal", [])),
            "low_willingness": len(pools.get("low_willingness", [])),
            "scarce_couriers": len(pools.get("scarce_couriers", [])),
            "scenes": {scene: len(items) for scene, items in sorted(scene_pools.items())},
            "validation": len(validation),
        },
        "best_validation": best_validation,
        "best_target_validation": best_target_validation,
        "output_solver": str(OUTPUT_SOLVER),
        "target_solver": str(TARGET_SOLVER),
        "platform_backup": str(PLATFORM_BACKUP),
    }
    write_state(state)
    print(json.dumps({
        "event": "start",
        "pid": os.getpid(),
        "case_counts": state["case_counts"],
        "validation": metric_key(best_validation),
        "target": target_metric_key(best_target_validation),
    }, ensure_ascii=False), flush=True)

    try:
        while time.time() < deadline and (not args.max_iterations or iteration < args.max_iterations):
            iteration += 1
            candidate = mutate_profile(champion, rng)
            signature = profile_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)

            quick = build_quick_cases(fixed, pools, scene_pools, rng, iteration)
            incumbent_quick = evaluate_profile(champion, quick, "incumbent_" + str(iteration))
            candidate_quick = evaluate_profile(candidate, quick, "trial_" + str(iteration))
            promoted = False
            target_promoted = False
            if metric_key(candidate_quick) < metric_key(incumbent_quick):
                candidate_validation = evaluate_profile(candidate, validation, "validation_" + str(iteration))
                full_better = metric_key(candidate_validation) < metric_key(best_validation)
                target_better = target_metric_key(candidate_validation) < target_metric_key(best_target_validation)
                full_regression_ratio = (
                    candidate_validation["mean_race"] / max(EPS, best_validation["mean_race"])
                    if best_validation["mean_race"] < float("inf") else float("inf")
                )
                target_material = (
                    candidate_validation["target_mean_race"]
                    < best_target_validation["target_mean_race"] - 0.25
                )
                if full_better:
                    champion = candidate
                    best_validation = candidate_validation
                    if target_metric_key(candidate_validation) < target_metric_key(best_target_validation):
                        best_target_validation = candidate_validation
                    accepted += 1
                    promoted = True
                    materialize_solver(source, champion, OUTPUT_SOLVER)
                    materialize_solver(source, champion, TARGET_SOLVER)
                elif (
                    target_better
                    and target_material
                    and candidate_validation["invalid"] == 0
                    and candidate_validation["missing"] == 0
                    and full_regression_ratio <= 1.012
                ):
                    champion = candidate
                    best_target_validation = candidate_validation
                    accepted += 1
                    promoted = True
                    target_promoted = True
                    materialize_solver(source, champion, TARGET_SOLVER)
                    # Keep solver_trained_best as the deployable challenger too:
                    # the online average is dominated by the two target scenes.
                    materialize_solver(source, champion, OUTPUT_SOLVER)

            event = {
                "iteration": iteration,
                "accepted": accepted,
                "promoted": promoted,
                "target_promoted": target_promoted,
                "quick_incumbent": metric_key(incumbent_quick),
                "quick_candidate": metric_key(candidate_quick),
                "best_validation": metric_key(best_validation),
                "best_target_validation": target_metric_key(best_target_validation),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            state.update({
                "iterations": iteration,
                "accepted": accepted,
                "last_event": event,
                "best_validation": best_validation,
                "best_target_validation": best_target_validation,
            })
            write_state(state)
            print(json.dumps(event, ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        write_state(state)
        raise
    else:
        state.update({"status": "finished", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        write_state(state)
        print(json.dumps({
            "event": "finished",
            "iterations": iteration,
            "accepted": accepted,
            "best_validation": metric_key(best_validation),
            "best_target_validation": target_metric_key(best_target_validation),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
