"""Generate extra hard low/scarce training cases for the AutoSolver trainer.

The bundled 1500 scene bank is broad, but the online regret is concentrated in
low_willingness_seed501 and scarce_couriers_seed401.  This script creates
deterministic perturbations of the existing training files so the long trainer
sees more variants of exactly those two regimes.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "training_cases_hard"
CASE_BANK = ROOT / "training_cases" / "train"
AUTO_CASES = ROOT / "training_cases_auto"
FIXED_CASES = ROOT / "training_cases"
HEADER = ["task_id_list", "courier_id", "total_score", "willingness"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def candidate_sources(scene: str) -> list[Path]:
    paths: list[Path] = []
    bank_dir = CASE_BANK / scene
    if bank_dir.exists():
        paths.extend(sorted(bank_dir.glob("*.txt")))
    if scene.startswith("low_willingness"):
        paths.extend(sorted(AUTO_CASES.glob("auto_low_willingness_*.txt")))
        paths.extend(sorted(FIXED_CASES.glob("synthetic_low_willingness_*.txt")))
    elif scene.startswith("scarce_couriers"):
        paths.extend(sorted(AUTO_CASES.glob("auto_scarce_couriers_*.txt")))
        paths.extend(sorted(FIXED_CASES.glob("synthetic_scarce_couriers_*.txt")))
    return [path for path in paths if path.exists()]


def perturb_rows(rows: list[dict[str, str]], scene: str, rng: random.Random) -> list[dict[str, str]]:
    score_scale = rng.uniform(0.94, 1.08)
    out: list[dict[str, str]] = []
    for row in rows:
        task_str = row["task_id_list"]
        is_pair = "," in task_str
        score = float(row["total_score"])
        willingness = float(row["willingness"])

        if scene.startswith("low_willingness"):
            # Low-willingness hidden cases reward using more backup riders, so
            # vary the acceptance tail aggressively while keeping occasional
            # moderate-probability riders in the pool.
            if rng.random() < 0.085:
                new_w = rng.uniform(0.18, 0.43)
            else:
                new_w = rng.betavariate(1.25, 10.8) * 0.74 + rng.uniform(0.005, 0.018)
            new_w = 0.72 * new_w + 0.28 * clamp(willingness * rng.uniform(0.65, 1.2), 0.01, 0.5)
            pair_shift = rng.gauss(4.0, 3.5) if is_pair else rng.gauss(-1.0, 2.5)
            new_score = score * score_scale + pair_shift + rng.gauss(0.0, 4.5)
            if rng.random() < 0.015:
                new_score += rng.gauss(12.0, 7.0)
        else:
            # Scarce-courier cases are topology-sensitive: 20 two-order bundles
            # plus a small number of backups is usually the interesting region.
            new_w = willingness * rng.uniform(0.86, 1.22) + rng.gauss(0.018, 0.045)
            pair_shift = rng.gauss(0.5, 5.5) if is_pair else rng.gauss(-2.2, 3.8)
            new_score = score * score_scale + pair_shift + rng.gauss(0.0, 3.5)
            if rng.random() < 0.04:
                new_w += rng.uniform(0.04, 0.12)

        out.append({
            "task_id_list": task_str,
            "courier_id": row["courier_id"],
            "total_score": f"{clamp(new_score, 10.0, 100.0):.3f}",
            "willingness": f"{clamp(new_w, 0.01, 0.95):.4f}",
        })

    rng.shuffle(out)
    return out


def write_case(path: Path, rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tasks = set()
    couriers = set()
    scores = []
    wills = []
    for row in rows:
        tasks.update(part.strip() for part in row["task_id_list"].split(",") if part.strip())
        couriers.add(row["courier_id"])
        scores.append(float(row["total_score"]))
        wills.append(float(row["willingness"]))
    return {
        "path": str(path.relative_to(ROOT)),
        "num_tasks": len(tasks),
        "num_couriers": len(couriers),
        "num_candidates": len(rows),
        "avg_score": round(statistics.mean(scores), 6),
        "avg_willingness": round(statistics.mean(wills), 6),
    }


def main() -> None:
    rng = random.Random(20260528)
    manifest = []
    specs = {
        "low_willingness_seed501": 160,
        "scarce_couriers_seed401": 160,
    }
    for scene, count in specs.items():
        sources = candidate_sources(scene)
        if not sources:
            raise FileNotFoundError(f"No source cases found for {scene}")
        for index in range(count):
            source = sources[index % len(sources)]
            rows = perturb_rows(read_rows(source), scene, rng)
            out = OUT_ROOT / scene / f"{scene}_hard{index:03d}.txt"
            meta = write_case(out, rows)
            meta.update({"scene": scene, "source": str(source.relative_to(ROOT))})
            manifest.append(meta)
        print(f"generated {scene}: {count}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(manifest)} hard cases to {OUT_ROOT}")


if __name__ == "__main__":
    main()
