"""Offline AutoResearch agent for the competition solver.

This module intentionally avoids online LLM calls.  It runs the same
deterministic strategy-generation and evaluation loop used by ``solver.solve``
so local experiments match the submitted solver as closely as possible.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import solver


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIRS = (
    ROOT_DIR / "Hackthon Data",
    ROOT_DIR / "example",
)


class CompetitionAgentGraph:
    """Small compatibility wrapper with the old LangGraph ``invoke`` shape."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages") or []
        if messages:
            query = getattr(messages[-1], "content", str(messages[-1]))
        else:
            query = ""
        return run_agent(query)


def _available_data_files() -> list[Path]:
    files: list[Path] = []
    for directory in DATA_DIRS:
        if directory.exists():
            files.extend(sorted(directory.glob("*.txt")))
    return files


def _resolve_data_file(query: str) -> Path:
    files = _available_data_files()
    if not files:
        raise FileNotFoundError("No .txt data files found in Hackthon Data/ or example/.")

    tokens = re.findall(r"[\w.\- ]+\.txt", query or "")
    normalized: dict[str, Path] = {}
    for path in files:
        normalized.setdefault(path.name.lower(), path)
    for token in tokens:
        path = normalized.get(token.strip().lower())
        if path is not None:
            return path

    for preferred in ("large_seed301.txt",):
        path = normalized.get(preferred)
        if path is not None:
            return path

    return files[0]


def _summarize_log(experiment_log: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    risk_results = [
        result for result in experiment_log if hasattr(result, "expected_accepted_tasks")
    ]
    deterministic_results = [
        result for result in experiment_log if not hasattr(result, "expected_accepted_tasks")
    ]
    ranked = sorted(risk_results, key=solver._risk_sort_key)
    ranked.extend(sorted(deterministic_results, key=solver._eval_sort_key))
    summary = []
    for result in ranked[:limit]:
        if hasattr(result, "expected_accepted_tasks"):
            summary.append(
                {
                    "strategy": result.profile_name,
                    "covered_tasks": result.covered_task_count,
                    "expected_accepted_tasks": round(result.expected_accepted_tasks, 4),
                    "total_score": round(result.total_score, 4),
                    "extra_couriers": result.extra_courier_count,
                    "min_accept_probability": round(result.min_accept_probability, 4),
                    "avg_accept_probability": round(result.avg_accept_probability, 4),
                    "valid": result.is_valid,
                }
            )
        else:
            summary.append(
                {
                    "strategy": result.strategy_name,
                    "covered_tasks": result.covered_task_count,
                    "total_score": round(result.total_score, 4),
                    "bundle_count": result.bundle_count,
                    "valid": result.is_valid,
                    "elapsed_ms": round(result.elapsed_ms, 3),
                }
            )
    return summary


def run_agent(user_input: str) -> dict[str, Any]:
    """Run the offline competition agent on a data file.

    The returned dict keeps the keys used by the original CLI while adding
    experiment summaries that explain which generated strategy won.
    """
    data_file = _resolve_data_file(user_input)
    input_text = data_file.read_text(encoding="utf-8")
    candidates, _, _ = solver.parse_input(input_text)

    experiment_log: list[Any] = []
    deadline_ms = solver._now_ms() + solver.SOLVE_TIME_BUDGET_MS - solver.SAFETY_MARGIN_MS

    start_time = time.perf_counter()
    selected = solver.choose_best_solution(
        candidates,
        deadline_ms=deadline_ms,
        experiment_log=experiment_log,
    )
    selected, backup_map, risk_eval = solver.choose_risk_aware_final_solution(
        candidates,
        selected,
        deadline_ms=deadline_ms,
        experiment_log=experiment_log,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    final_eval = solver.evaluate_solution(selected, "competition_agent_final", elapsed_ms)
    solution = solver.format_solution(selected, backup_map)
    best_strategy = None
    if experiment_log:
        risk_results = [
            result
            for result in experiment_log
            if hasattr(result, "expected_accepted_tasks")
        ]
        if risk_results:
            best_strategy = min(risk_results, key=solver._risk_sort_key).profile_name
        else:
            best_strategy = min(experiment_log, key=solver._eval_sort_key).strategy_name

    return {
        "data_file": str(data_file),
        "solution": solution,
        "base_total_score": round(final_eval.total_score, 4),
        "total_score": round(risk_eval.total_score, 4),
        "covered_tasks": final_eval.covered_task_count,
        "bundle_count": final_eval.bundle_count,
        "expected_accepted_tasks": round(risk_eval.expected_accepted_tasks, 4),
        "risk_total_score": round(risk_eval.total_score, 4),
        "extra_couriers": risk_eval.extra_courier_count,
        "min_accept_probability": round(risk_eval.min_accept_probability, 4),
        "avg_accept_probability": round(risk_eval.avg_accept_probability, 4),
        "valid": final_eval.is_valid and risk_eval.is_valid,
        "elapsed_ms": round(elapsed_ms, 3),
        "best_strategy": best_strategy,
        "experiment_count": len(experiment_log),
        "experiments": _summarize_log(experiment_log),
    }


agent_graph = CompetitionAgentGraph()


__all__ = ["run_agent", "agent_graph", "CompetitionAgentGraph"]
