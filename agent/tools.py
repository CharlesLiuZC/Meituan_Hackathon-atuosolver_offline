import os
import math

from common.parser import parse_input


# ── file helpers ──────────────────────────────────────────────────────────────

def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "example")


def _read_file(filename: str) -> str:
    base_dir = _data_dir()
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        files = os.listdir(base_dir)
        return f"File not found: {filepath}\nAvailable files: {', '.join(files)}"
    with open(filepath, encoding="utf-8") as f:
        return f.read()


# ── tools exposed to the LLM ──────────────────────────────────────────────────

def list_example_files() -> str:
    """List all .txt files in the example/ directory."""
    base_dir = _data_dir()
    if not os.path.exists(base_dir):
        return "example/ directory not found."
    files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]
    if not files:
        return "No .txt files found in example/."
    return "\n".join(files)


def peek_data(filename: str, n: int = 20) -> str:
    """Show the first n rows of parsed data plus overall statistics.

    Use this to understand the data distribution before designing a strategy.
    """
    content = _read_file(filename)
    if content.startswith("File not found"):
        return content

    candidates = parse_input(content)
    if not candidates:
        return "No valid candidates found."

    scores = [c[0] for c in candidates]
    willingness = [c[3] for c in candidates]
    task_counts = [len(c[1].split(",")) for c in candidates]
    all_tasks = set()
    all_couriers = set()
    for c in candidates:
        for t in c[1].split(","):
            all_tasks.add(t.strip())
        all_couriers.add(c[2])

    lines = []
    lines.append(f"=== {filename}  Statistics ===")
    lines.append(f"Total candidates : {len(candidates)}")
    lines.append(f"Unique tasks     : {len(all_tasks)}")
    lines.append(f"Unique couriers  : {len(all_couriers)}")
    lines.append(f"Score range      : {min(scores):.4f} ~ {max(scores):.4f}")
    lines.append(f"Score mean/median: {sum(scores)/len(scores):.4f} / {sorted(scores)[len(scores)//2]:.4f}")
    lines.append(f"Willingness range: {min(willingness):.4f} ~ {max(willingness):.4f}")
    lines.append(f"Tasks/bundle range: {min(task_counts)} ~ {max(task_counts)}")
    lines.append(f"Multi-task bundles: {sum(1 for tc in task_counts if tc > 1)}")
    lines.append(f"Single-task bundles: {sum(1 for tc in task_counts if tc == 1)}")
    lines.append("")

    header = ("# {:<5s}  {:<25s}  {:<6s}  {:<10s}  {:<10s}  {:>5s}".format(
        "row", "task_id_list_str", "courier", "score", "willingness", "n_tasks"))
    lines.append(header)
    lines.append("-" * len(header))

    for i, (score, task_str, courier, willing) in enumerate(candidates[:n]):
        n_tasks = len(task_str.split(","))
        lines.append("  {:<5d}  {:<25s}  {:<6s}  {:<10.4f}  {:<10.4f}  {:>5d}".format(
            i, task_str, courier, score, willing, n_tasks))

    return "\n".join(lines)


def execute_strategy(filename: str, strategy_code: str) -> dict:
    """Execute a custom greedy strategy defined by your Python code.

    strategy_code must define a function named ``sort_key`` with this signature::

        def sort_key(c: dict) -> tuple:
            ...

    Each candidate dict ``c`` has these keys:
      - score: float           (total score of the bundle)
      - task_count: int        (number of tasks in the bundle)
      - score_per_task: float  (score / task_count)
      - willingness: float     (courier willingness)
      - task_str: str          (original task-id string)
      - courier_id: str        (courier identifier)

    Candidates are sorted by sort_key ascending, then greedily selected:
    skip any candidate whose tasks or courier overlap with already-selected ones.

    The function may also import ``math`` if needed.

    Returns {solution, total_score, covered_tasks, bundle_count, candidates_used}.
    """
    content = _read_file(filename)
    if content.startswith("File not found"):
        return {"error": content}

    raw = parse_input(content)
    if not raw:
        return {"error": "No valid candidates found."}

    candidates = []
    for score, task_str, courier_id, willingness in raw:
        task_ids = [t.strip() for t in task_str.split(",") if t.strip()]
        task_count = len(task_ids)
        candidates.append({
            "task_str": task_str,
            "task_ids": set(task_ids),
            "courier_id": courier_id,
            "score": score,
            "willingness": willingness,
            "task_count": task_count,
            "score_per_task": score / task_count if task_count else score,
        })

    # Compile the user's sort_key function in a sandboxed namespace
    namespace = {"math": math, "__builtins__": {}}
    try:
        exec(strategy_code, namespace)
    except Exception as e:
        return {"error": f"Failed to compile strategy_code: {e}"}

    sort_key = namespace.get("sort_key")
    if sort_key is None:
        return {"error": "strategy_code must define a function named 'sort_key'."}

    # Sort candidates by the user-provided key
    try:
        candidates.sort(key=sort_key)
    except Exception as e:
        return {"error": f"sort_key raised an error: {e}"}

    # Collect all unique task IDs for early-termination check
    all_task_ids: set[str] = set()
    for c in candidates:
        all_task_ids |= c["task_ids"]

    # Greedy selection
    selected = []
    used_tasks: set[str] = set()
    used_couriers: set[str] = set()
    total_score = 0.0
    tried = 0

    for c in candidates:
        tried += 1
        if c["courier_id"] in used_couriers:
            continue
        if c["task_ids"] & used_tasks:
            continue
        selected.append((c["task_str"], [c["courier_id"]]))
        used_tasks |= c["task_ids"]
        used_couriers.add(c["courier_id"])
        total_score += c["score"]
        if used_tasks == all_task_ids:
            break

    covered_tasks = set()
    for task_str, _ in selected:
        for t in task_str.split(","):
            covered_tasks.add(t.strip())

    return {
        "solution": selected,
        "total_score": round(total_score, 4),
        "covered_tasks": len(covered_tasks),
        "bundle_count": len(selected),
        "tried_candidates": tried,
    }

