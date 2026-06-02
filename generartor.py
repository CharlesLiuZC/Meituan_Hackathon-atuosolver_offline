import os
import random

from common.parser import parse_input


def generate_data_file(
    num_tasks: int = 40,
    num_couriers: int = 80,
    num_candidates: int | None = None,
    output_dir: str = "example",
    seed: int | None = None,
) -> str:
    """Generate a random data file with the same format as large_seed301.txt.

    Args:
        num_tasks: number of unique tasks (T0000..T{num_tasks-1:04d}).
        num_couriers: number of unique couriers (C000..C{num_couriers-1:03d}).
        num_candidates: total lines of candidate data.  Defaults to roughly
            ``num_couriers * (num_tasks + num_tasks*(num_tasks-1)//2) // 20``.
        output_dir: directory to write the file into (created if missing).
        seed: random seed for reproducibility (default: random).

    Returns:
        The file path of the generated file.
    """
    if seed is not None:
        random.seed(seed)

    task_ids = [f"T{i:04d}" for i in range(num_tasks)]
    courier_ids = [f"C{i:03d}" for i in range(num_couriers)]

    singles = [(t,) for t in task_ids]
    pairs = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            pairs.append((task_ids[i], task_ids[j]))

    all_bundles = singles + pairs

    if num_candidates is None:
        total_possible = len(all_bundles) * num_couriers
        num_candidates = max(500, total_possible // 20)

    combo_pool = [
        (bundle, courier)
        for bundle in all_bundles
        for courier in courier_ids
    ]
    if num_candidates < len(combo_pool):
        combos = random.sample(combo_pool, num_candidates)
    else:
        combos = combo_pool[:]

    lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]
    for bundle, courier in combos:
        task_str = ",".join(bundle)
        n = len(bundle)

        base = n * random.uniform(20, 50)
        noise = random.uniform(-5, 10)
        score = round(max(10.0, min(100.0, base + noise)), 4)

        willingness = round(random.uniform(0.01, 0.95), 4)

        lines.append(f"{task_str}\t{courier}\t{score}\t{willingness}")

    os.makedirs(output_dir, exist_ok=True)

    file_seed = random.randint(1, 99999)
    filename = f"large_seed{file_seed}.txt"
    filepath = os.path.join(output_dir, filename)

    while os.path.exists(filepath):
        file_seed = random.randint(1, 99999)
        filename = f"large_seed{file_seed}.txt"
        filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return filepath


if __name__ == "__main__":
    import sys

    kwargs = {}
    if len(sys.argv) > 1:
        kwargs["num_tasks"] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kwargs["num_couriers"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kwargs["num_candidates"] = int(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs["seed"] = int(sys.argv[4])

    path = generate_data_file(**kwargs)
    print(f"Generated: {path}")

    with open(path, encoding="utf-8") as f:
        data = parse_input(f.read())
    scores = [d[0] for d in data]
    t_counts = [len(d[1].split(",")) for d in data]
    print(f"  candidates: {len(data)}")
    print(f"  score range: {min(scores):.4f} ~ {max(scores):.4f}")
    print(f"  single-task: {sum(1 for t in t_counts if t == 1)}")
    print(f"  multi-task : {sum(1 for t in t_counts if t > 1)}")
