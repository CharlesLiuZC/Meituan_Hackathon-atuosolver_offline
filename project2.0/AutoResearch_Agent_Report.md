# AutoSolver 配送分配 Agent 技术报告

## 1. 项目目标

本项目面向美团 Hackathon 配送分配赛题，实现一个可提交的 `solver.py`。系统不是固定贪心算法，而是在每个测试用例 10 秒限制内，自动生成多类策略、评估策略、保留当前最优解，并继续做局部修复和概率兜底派单的离线 AutoResearch Agent。

输出接口保持官方要求：

```python
def solve(input_text: str) -> list:
    return [(task_id_list_str, [courier_id, ...]), ...]
```

其中同一个订单或合单可以同时指派给多个骑手；每个骑手在同一轮次最多只能出现在一个输出项中。

## 2. 问题建模

每条输入候选记为：

```text
候选 i = (订单集合 S_i, 骑手 c_i, 成本 score_i, 接单概率 willingness_i)
```

Agent 分两层建模：

1. 主派单层：先选择不冲突的订单/合单和主骑手，目标是最大覆盖订单数，并在覆盖相同情况下最小化总分。
2. 概率兜底层：对主方案中接单概率低的订单/合单，使用剩余骑手追加派单，提升 `1 - Π(1 - willingness)` 形式的整体接起概率。

风险评估目标为：

```text
risk_adjusted_score =
    total_score + (covered_tasks - expected_accepted_tasks) * missing_penalty
```

其中 `expected_accepted_tasks` 是按接单概率计算的期望完成订单数，`missing_penalty` 在代码中由 `ACCEPTANCE_VALUE_PER_TASK` 控制。该目标同时体现“尽量多接起订单”和“额外骑手资源不能无限增加”。

## 3. Agent 自主迭代机制

Agent 的核心循环在 `solver.py` 内完成，不依赖在线 LLM 或外部 API，因此可以直接在评测环境运行。

流程如下：

```text
解析输入
  -> 构建订单 bitmask / 骑手 bitmask 索引
  -> 运行多种基础策略
  -> 根据历史结果生成策略基因变体
  -> 自动评估并保留最优主方案
  -> greedy repair 局部修复
  -> exact local repair 小邻域精确替换
  -> optional ILP 增强
  -> 概率兜底多骑手派单
  -> 输出当前最优方案
```

自主探索体现在三个层面：

- 策略生成：系统内置 `score_per_task`、低意愿惩罚、合单优先、稀缺订单优先等种子策略，并基于历史优胜策略派生新权重。
- 自动评估：每个策略输出统一转为评估结果，比较合法性、覆盖数、总分、意愿度和冲突情况。
- 自主改进：对当前最优解执行移除重排、局部精确替换和概率兜底 profile 搜索。

## 4. 求解技术

### 4.1 位掩码冲突判断

每个订单和骑手都映射为整数 bit。判断候选是否冲突只需：

```python
candidate.task_mask & used_task_mask
candidate.courier_bit & used_courier_mask
```

这使大量贪心排序、局部替换和修复操作可以在 10 秒内反复执行。

### 4.2 参数化策略基因

策略基因由以下特征组成：

- `score`
- `score_per_task`
- `willingness`
- `bundle_bias`
- `scarcity`
- `courier_pressure`
- `low_willingness_penalty`

Agent 会根据历史表现自动改变这些权重，形成下一批策略，而不是只运行人工写死的一条规则。

### 4.3 Optional ILP

若环境提供 SciPy，Agent 会构造 0-1 整数规划：

```text
变量：是否选择候选 i
约束：每个订单最多一次，每个骑手最多一次
目标：优先最大化覆盖订单数，其次最小化 score
```

ILP 设置了严格时间上限；失败、超时或依赖缺失时，系统自动回退到纯 Python 策略搜索。

### 4.4 概率兜底多派

主方案完成后，Agent 会尝试多种兜底 profile：

- `none`
- `conservative`
- `balanced`
- `aggressive`
- `acceptance_first_2`
- `acceptance_first_3`
- `max_reliability`

每个 profile 定义目标接起概率、最多追加骑手数、额外资源预算和期望接单价值。Agent 自动计算每次追加骑手的边际收益：

```text
marginal_expected_tasks =
    bundle_task_count * current_failure_probability * willingness
```

只有当边际期望收益能够覆盖资源成本时，才会追加该骑手。

## 5. 本地实验

测试数据：`Hackthon Data/large_seed301.txt`

| 版本 | 覆盖订单 | 输出项 | 骑手指派数 | 本地 total_score | 期望接起订单 | 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方示例贪心 | 40 | 40 | 40 | 424.468 | 未建模 | < 0.2s |
| 单骑手 AutoResearch | 40 | 24 | 24 | 392.818 | 12.5485 | 约 2.1s |
| 风险感知 AutoSolver | 40 | 40 | 60 | 794.306 | 36.7945 | 约 3.5-4.1s |

风险感知版本的最优 profile 为 `primary_reliable_score:aggressive`。它先选择更可靠的主骑手组合，再只对仍有风险的订单追加骑手，因此相比“低成本主解 + 大量补派”，同时降低资源分并提升期望接起订单数。

概率兜底后，输出仍满足：

- 无订单冲突。
- 无骑手冲突。
- 所有 `(task_id_list, courier_id)` 均来自原始候选。
- 单测试用例耗时远低于 10 秒。

## 6. 时间预算

当前提交版主要预算如下：

| 阶段 | 预算 |
| --- | ---: |
| 总时间预算 | 9.2s |
| 安全余量 | 0.35s |
| 策略变异搜索 | 0.45s |
| greedy repair | 1.5s |
| exact local repair | 1.5s |
| optional ILP | 2.2s |
| 概率兜底搜索 | 0.45s |

所有阶段均有时间守卫，任意阶段时间不足都会直接返回当前最优解，避免超时无输出。

## 7. 提交说明

正式提交文件是根目录 `solver.py`。它是单文件自包含实现，核心路径只依赖 Python 标准库；若评测环境存在 SciPy，则自动启用 ILP 增强，否则走纯 Python fallback。

本地演示入口是：

```bash
python main.py
```

评测入口是：

```bash
python evaluate.py "Hackthon Data/large_seed301.txt"
```

## 8. 总结

最终版本满足赛题三项要求：

1. 自主策略探索：运行时自动生成和变异策略基因，并尝试多个概率兜底 profile。
2. 自动评估筛选：每个策略结果都由统一 evaluator 比较，保留更优方案。
3. 迭代改进循环：基于历史优胜策略继续变异，并通过局部搜索和概率兜底持续改进。

该 AutoSolver 既能在传统本地 evaluator 下给出合法满覆盖解，也能利用 `willingness` 建模真实接单风险，在低接单概率订单上主动追加骑手，降低无人接单概率。
