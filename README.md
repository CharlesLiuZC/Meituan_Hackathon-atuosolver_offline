# Meituan Hackathon AutoSolver Offline

美团黑客松配送分配问题的离线求解与训练项目。

项目目标是在评测要求的 `solve(input_text: str) -> list` 接口下，为订单和骑手候选组合生成尽可能优的分配方案：优先覆盖更多订单，再降低期望惩罚分数。

## 核心文件

- `solver.py`：当前可提交到评测平台的主求解器。
- `benchmark.py`：本地校验 `solver.py` 输出是否合法，并计算本地惩罚分。
- `autosolver_agent.py`：自动策略探索 Agent。
- `train_24h_agent.py`：长时间离线训练与参数搜索脚本。
- `expand_hard_training_cases.py`：生成 low willingness / scarce couriers 困难样本。
- `solver_platform_best_v3.py`、`solver_trained_best.py`、`solver_low_scarce_challenger.py`：历史安全版和训练候选版本。
- `project2.0/AutoResearch_Agent_Report.md`：技术报告草稿。

## 快速运行

```powershell
python benchmark.py "Hackthon Data"
```

或测试单个文件：

```powershell
python benchmark.py "Hackthon Data\large_seed301.txt"
```

## 提交接口

评测平台要求 `solver.py` 中定义：

```python
def solve(input_text: str) -> list:
    ...
```

返回格式为：

```python
[
    ("task_id", ["courier_id"]),
    ("task_id_a,task_id_b", ["courier_id_1", "courier_id_2"]),
]
```

## 数据说明

比赛原始数据、自动生成的大规模训练样本、日志和本地缓存不纳入 Git 管理，相关目录已写入 `.gitignore`：

- `Hackthon Data/`
- `training_cases/train/`
- `training_cases_auto/`
- `training_cases_hard/`
- `training_24h_*`
- `*.log`
- `*.sqlite`

这样仓库只保留源码、配置、报告和可复现实验脚本。
