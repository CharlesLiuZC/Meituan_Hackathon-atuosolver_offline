"""CLI entry point for the offline competition agent."""

from agent import run_agent


def main():
    query = input("Query: ")
    result = run_agent(query)

    solution = result.get("solution")
    total_score = result.get("total_score")
    covered_tasks = result.get("covered_tasks", 0)

    if solution is not None:
        print(f"data_file: {result.get('data_file')}")
        print(f"valid: {result.get('valid')}")
        print(f"elapsed_ms: {result.get('elapsed_ms')}")
        print(f"best_strategy: {result.get('best_strategy')}")
        print(f"experiment_count: {result.get('experiment_count')}")
        print(f"base_total_score: {result.get('base_total_score')}")
        print(f"expected_accepted_tasks: {result.get('expected_accepted_tasks')}")
        print(f"risk_total_score: {result.get('risk_total_score')}")
        print(f"extra_couriers: {result.get('extra_couriers')}")
        print(f"min_accept_probability: {result.get('min_accept_probability')}")
        print(f"avg_accept_probability: {result.get('avg_accept_probability')}")
        print(f"\ntotal_score: {total_score}")
        print(f"covered_tasks: {covered_tasks}")
        print(f"solution: {solution}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
