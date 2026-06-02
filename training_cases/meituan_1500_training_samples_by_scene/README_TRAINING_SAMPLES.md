# 1500 Meituan AutoSolver Training Samples

150 synthetic-augmentation instances are provided for each requested scene.

Anchor file: `case_bank/train/large_seed301/large_seed301_aug000.txt` is an exact copy of the uploaded official `large_seed301(2).txt`.

Schema: `task_id_list\tcourier_id\ttotal_score\twillingness`. Each row is a single-order or two-order candidate, with no duplicated `(task_id_list, courier_id)` within a file.

Teacher-score targets transcribed from the feedback screenshot:
```json
{
  "high_noise_seed601": 497.06,
  "large_seed301": 680.13,
  "large_seed302": 640.43,
  "low_willingness_seed501": 1806.96,
  "medium_seed201": 494.86,
  "medium_seed202": 527.59,
  "medium_seed203": 508.65,
  "scarce_couriers_seed401": 1554.38,
  "small_seed100": 306.91,
  "tiny_seed42": 158.65
}
```

Recommended training:
```bash
python training/collect_experiments.py case_bank/train --memory memory/experiments.sqlite --budget-ms 5000
python training/tune_params.py case_bank/train/scarce_couriers_seed401 --rounds 500 --budget-ms 7000 --memory memory/experiments.sqlite
python training/tune_params.py case_bank/train/low_willingness_seed501 --rounds 500 --budget-ms 7000 --memory memory/experiments.sqlite
python training/train_selector.py --memory memory/experiments.sqlite --out models/strategy_selector.json
```

These files are training augmentations, not hidden official benchmark data.
