"""Command-line evaluator for solver.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import solver
from common.evaluator import DEFAULT_MISSING_TASK_PENALTY, evaluate_solution


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate solver.py on a task-courier assignment data file."
    )
    parser.add_argument(
        "data_file",
        help="Path to a tab-separated input file, for example example/large_seed301.txt.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=DEFAULT_MISSING_TASK_PENALTY,
        help=(
            "Penalty added for each missing task. "
            f"Default: {DEFAULT_MISSING_TASK_PENALTY}."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full evaluation result as JSON.",
    )
    return parser


def _print_human_readable(result: dict) -> None:
    print("Evaluation result")
    print("=================")
    print(f"valid: {result['valid']}")
    print(f"total_tasks: {result['total_tasks']}")
    print(f"covered_tasks: {result['covered_tasks']}")
    print(f"missing_tasks: {result['missing_tasks']}")
    print(f"total_score: {result['total_score']}")
    print(f"missing_task_penalty: {result['missing_task_penalty']}")
    print(f"objective_score: {result['objective_score']}")
    print(f"solution_items: {result['solution_items']}")
    print(f"valid_assignments: {result['valid_assignments']}")
    print("")
    print("Validity checks")
    print("---------------")
    print(f"task_conflict: {result['task_conflict']}")
    print(f"courier_conflict: {result['courier_conflict']}")
    print(f"invalid_candidate_count: {result['invalid_candidate_count']}")
    print(f"malformed_item_count: {result['malformed_item_count']}")

    if result["missing_task_ids"]:
        print(f"missing_task_ids: {', '.join(result['missing_task_ids'])}")
    if result["duplicate_tasks"]:
        print(f"duplicate_tasks: {', '.join(result['duplicate_tasks'])}")
    if result["duplicate_couriers"]:
        print(f"duplicate_couriers: {', '.join(result['duplicate_couriers'])}")
    if result["unknown_tasks"]:
        print(f"unknown_tasks: {', '.join(result['unknown_tasks'])}")
    if result["invalid_candidates"]:
        print("invalid_candidates:")
        for item in result["invalid_candidates"][:10]:
            print(
                "  "
                f"index={item['index']} "
                f"task_id_list={item['task_id_list']} "
                f"courier_id={item['courier_id']}"
            )
        if len(result["invalid_candidates"]) > 10:
            print(f"  ... {len(result['invalid_candidates']) - 10} more")
    if result["malformed_items"]:
        print("malformed_items:")
        for item in result["malformed_items"][:10]:
            print(f"  index={item['index']} reason={item['reason']}")
        if len(result["malformed_items"]) > 10:
            print(f"  ... {len(result['malformed_items']) - 10} more")


def main() -> None:
    args = _build_parser().parse_args()
    input_text = Path(args.data_file).read_text(encoding="utf-8")
    solution = solver.solve(input_text)
    result = evaluate_solution(
        input_text,
        solution,
        missing_task_penalty=args.penalty,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human_readable(result)


if __name__ == "__main__":
    main()
