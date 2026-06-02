# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task

Meituan delivery courier-task assignment optimization challenge. Given a set of delivery orders (tasks) and available couriers, with pre-computed scores for each task-courier combination, find an optimal assignment that maximizes covered tasks while minimizing total score. The system allows merging multiple orders into bundles assigned to one courier, and supports multi-courier assignment where the first courier to accept gets the order.

## Key Commands

```bash
# Run the standalone solver on a test case
python -c "import solver; txt=open('example/large_seed301.txt').read(); print(len(solver.solve(txt)))"

# Evaluate solver output
python evaluate.py example/large_seed301.txt
python evaluate.py example/large_seed301.txt --penalty 300
python evaluate.py example/large_seed301.txt --json

# Run the LLM-based auto-solver agent (requires DEEPSEEK_API_KEY in .env)
python main.py

# Run offline training agent (mutates solver configs, writes best to solver_trained_best.py)
python train_24h_agent.py --cases training_cases/

# Run benchmark evaluation
python benchmark.py example/large_seed301.txt

# Generate random test data
python generartor.py 40 80 33780 301
```

## Architecture

The project has two main components:

### 1. Static Solver (`solver.py`, ~1800 lines)

The production solver — deterministic, dependency-free (stdlib only). Core flow:

- `_parse_input()` parses TSV input into `Candidate` objects
- `Candidate` uses `__slots__` for performance: `task_mask` (bitmask), `courier_idx`, `score`, `willingness`, `task_count`, `score_per_task`, degree metrics
- `SearchContext` holds all candidates, task/courier index maps, and a deadline for time-budget control
- Strategy generation: multiple weighted sort-key functions stored in `CONFIG['strategies']`, each a 7-tuple of weights applied to candidate features
- Greedy selection with bitmask conflict detection (`task_mask` bitwise AND)
- Local search: simulated annealing via `operators.py` (candidate pruning, swap optimization)
- Multi-courier support: `CONFIG['enable_multi_courier_output']` controls whether bundles can have backup couriers
- Time-budgeted: `CONFIG['time_budget_ms'] = 7000`, partitioned across strategy search, local search, and backup allocation

`CONFIG` dict at the top of solver.py controls all solver behavior — strategies, time budgets, penalties, pruning thresholds.

### 2. Agent System (`agent/`, `main.py`, `autosolver_agent.py`)

Two agent variants:

- **`main.py` + `agent/graph.py`**: LangGraph-based agent using DeepSeek LLM. The agent writes `sort_key` functions (not full solvers) which are executed via `exec()` in `agent/tools.py`. Limited to 5 strategy attempts (`MAX_STRATEGIES`). Writes best strategy back to `solver.py`.

- **`autosolver_agent.py`**: Offline training agent. No LLM needed — mutates `CONFIG['strategies']` weights deterministically, evaluates on local cases, keeps best. Uses `benchmark.evaluate_output()` for scoring.

### Supporting modules

| File | Role |
|------|------|
| `common/parser.py` | Input TSV parsing → `(score, task_str, courier_id, willingness)` tuples |
| `common/evaluator.py` | Solution validation + objective scoring: `total_score + missing_tasks * penalty` |
| `analyzer.py` | Problem profiling: computes `ProblemProfile` with density, willingness distribution, dominated candidates, tight couplings |
| `operators.py` | Local search operators: simulated annealing, candidate pruning, conflict graph analysis |
| `benchmark.py` | Standalone evaluator: loads candidates, validates solution, computes parallel/priority penalties |
| `train_24h_agent.py` | Long-running trainer: mutates strategies, evaluates across case bank, writes challenger solver |
| `__validate_solver_*.py` | Frozen snapshots of solver.py for regression testing against platform results |

### Data flow

```
Input TSV → Candidate objects → Strategy weights → Greedy + bitmask selection
    → Local search (SA, pruning) → Solution [(task_str, [courier_id])]
    → evaluate.py / benchmark.py → objective_score
```

## Input/Output Format

Input: TSV with header `task_id_list	courier_id	total_score	willingness`

```
T0037,T0039	C028	52.016	0.582
T0012	C073	49.233	0.1485
```

Output from `solve(input_text)`:
```python
[("T0005,T0018", ["C045"]), ("T0030", ["C031"])]
```

Constraints: no task duplication across entries, no courier duplication across entries.

## Key Design Decisions

- **Bitmask task representation**: tasks are mapped to bit indices, `task_mask` enables O(1) conflict detection via bitwise AND
- **Strategy weights are 7-tuples**: `(w_score, w_willingness, w_task_count, w_score_per_task, w_bundle_ratio, w_courier_degree, flag)` — each tunes a different trade-off
- **Time budget partitioning**: solver splits its 7s budget across auto-strategy search, local search, and backup allocation phases
- **`__validate_solver_*.py`**: frozen copies of solver.py corresponding to platform-validated results — do not modify these; use them to verify regressions
- **`generartor.py`** (note: intentional misspelling) generates random test cases

## Python Version

Requires Python >= 3.12. Dependencies managed via `uv` (`pyproject.toml`). For solver.py only, no external deps needed.

## Environment

For LLM agent: copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`. Model config is in `agent/model.py` (deepseek-v4-pro). For LangSmith tracing, also set `LANGSMITH_API_KEY`.
