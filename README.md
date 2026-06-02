# meituan_autosolver-master 参考文档

这份文档用于帮助你读懂 `meituan_autosolver-master` 文件夹中的程序结构、脚本用途和运行流程。项目本质上是一个“任务包-骑手分配”求解器：输入一批候选分配方案，每条候选表示“某个骑手可以接一个或多个任务，并对应一个总分和意愿度”；程序要从候选集中选出一部分，使任务不重复、骑手不重复，同时尽量覆盖更多任务并降低总分。

## 1. 项目简介

项目包含两套相关但用途不同的逻辑：

1. `solver.py`：静态求解器。
   - 定义比赛/评测需要的 `solve(input_text: str) -> list` 函数。
   - 不需要 DeepSeek API Key。
   - 可以直接被评测程序调用。

2. `main.py` + `agent/`：自动寻优 Agent。
   - 使用 LangChain、LangGraph 和 DeepSeek 模型。
   - Agent 会读取样例数据，尝试最多 5 个贪心排序策略。
   - 每次策略会交给 `execute_strategy()` 在本地数据上执行。
   - 发现更好的策略后，会把最优策略写回 `solver.py`。

也就是说，`solver.py` 是最终可提交/可复用的求解函数，而 `agent/` 是用于自动生成或改进 `solver.py` 的实验框架。

## 2. 目录结构

```text
meituan_autosolver-master/
├── main.py                  # 交互式 Agent 入口
├── evaluate.py              # 本地评测 solver.py 输出
├── solver.py                # 当前已生成的静态求解器
├── generartor.py            # 随机样例数据生成脚本，文件名原本如此拼写
├── pyproject.toml           # 项目依赖与 Python 版本要求
├── uv.lock                  # uv 依赖锁文件
├── .python-version          # Python 版本：3.12
├── .env.example             # 环境变量示例
├── common/
│   ├── __init__.py
│   ├── evaluator.py         # 评测函数：合法性、覆盖数、总分、综合成本
│   └── parser.py            # 通用输入解析函数
├── agent/
│   ├── __init__.py          # 导出 run_agent 和 agent_graph
│   ├── model.py             # DeepSeek 模型配置
│   ├── state.py             # LangGraph 状态结构定义
│   ├── tools.py             # 暴露给 LLM 调用的工具函数
│   └── graph.py             # Agent 工作流图与 solver.py 生成逻辑
└── example/
    └── large_seed301.txt    # 样例候选数据
```

## 3. 环境配置

项目要求 Python `>=3.12`，依赖写在 `pyproject.toml` 中：

```toml
dependencies = [
    "dotenv>=0.9.9",
    "ipython>=9.13.0",
    "langchain>=1.3.0",
    "langchain-deepseek>=1.0.1",
    "langgraph>=1.2.0",
]
```

如果使用 `uv`，可以在项目根目录运行：

```bash
uv sync
```

如果只想运行 `solver.py`，通常不需要安装 LangChain 或配置模型 API，因为 `solver.py` 只使用 Python 标准库。

如果要运行 Agent，则需要配置 `.env`：

```bash
cp .env.example .env
```

然后在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API Key
```

`agent/model.py` 当前固定使用：

```python
model="deepseek-v4-pro"
model_provider="deepseek"
base_url="https://api.deepseek.com/v4"
```

如果换模型，需要修改 `agent/model.py`。

## 4. 输入输出格式

### 输入格式

输入是制表符分隔的文本，第一行通常是表头：

```text
task_id_list	courier_id	total_score	willingness
T0037,T0039	C028	52.016	0.582
T0012	C073	49.233	0.1485
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `task_id_list` | 任务 ID 列表。单任务如 `T0012`，多任务包如 `T0037,T0039` |
| `courier_id` | 骑手 ID，如 `C028` |
| `total_score` | 该候选分配的总分，代码中越低越优先 |
| `willingness` | 骑手意愿度，代码中越高通常越优先 |

### 输出格式

`solver.py` 必须定义：

```python
def solve(input_text: str) -> list:
    ...
```

返回值格式：

```python
[
    ("T0005,T0018", ["C045"]),
    ("T0030", ["C031"]),
]
```

每个元素表示一个被选中的候选分配：

```python
(task_id_list_str, [courier_id])
```

约束条件：

- 同一个任务不能在多个返回项中重复出现。
- 同一个骑手不能在多个返回项中重复出现。
- 当前代码默认每个任务包只对应一个骑手，所以列表中通常只有一个 `courier_id`。

## 5. 快速运行

### 5.1 直接使用当前静态求解器

在项目根目录运行：

```bash
python3 - <<'PY'
import solver

with open("example/large_seed301.txt", encoding="utf-8") as f:
    input_text = f.read()

solution = solver.solve(input_text)
print(len(solution))
print(solution[:5])
PY
```

这条路径不需要 DeepSeek API Key，适合检查当前 `solver.py` 的效果。

### 5.2 评测当前 solver.py

在项目根目录运行：

```bash
python evaluate.py example/large_seed301.txt
```

评测器会自动读取数据文件，调用当前 `solver.solve()`，并输出：

- `valid`：方案是否合法。
- `covered_tasks`：覆盖任务数。
- `missing_tasks`：未覆盖任务数。
- `total_score`：已选候选的总分。
- `objective_score`：带漏任务惩罚的综合成本，越低越好。

默认漏任务惩罚参数是：

```text
missing_task_penalty = 100.0
```

综合成本公式是：

```text
objective_score = total_score + missing_tasks * missing_task_penalty
```

也就是说，漏掉 1 个任务默认等价于额外增加 `100.0` 分成本。这个默认值是根据当前样例候选分数大致在 `10 ~ 100` 之间设置的，含义是：如果少覆盖一个任务，只换来很小的总分下降，那通常不值得；但如果少覆盖少量任务能换来非常大的总分下降，综合成本也会体现这种取舍。

可以用 `--penalty` 调整这个权衡：

```bash
python evaluate.py example/large_seed301.txt --penalty 300
```

如果想看完整字典结果：

```bash
python evaluate.py example/large_seed301.txt --json
```

### 5.3 运行自动寻优 Agent

在项目根目录运行：

```bash
python main.py
```

程序会提示：

```text
问：
```

可以输入类似：

```text
请在 large_seed301.txt 上寻找一个更好的贪心策略
```

`main.py` 会调用 `run_agent(query)`，Agent 会使用工具查看 `example/` 中的数据并测试策略。如果发现最优策略，会由 `agent/graph.py` 中的 `save_solver_node()` 写回 `solver.py`。

注意：这条路径需要 `.env` 中配置 `DEEPSEEK_API_KEY`，并且需要可访问 DeepSeek API。

### 5.4 生成随机样例数据

在项目根目录运行：

```bash
python generartor.py 40 80 33780 301
```

参数含义依次是：

```text
num_tasks num_couriers num_candidates seed
```

例如上面的命令会生成：

- 40 个任务。
- 80 个骑手。
- 33780 条候选。
- 使用随机种子 `301`。

脚本会把新文件写入 `example/` 目录，文件名形如：

```text
large_seed12345.txt
```

## 6. Agent 工作流详解

Agent 的核心逻辑在 `agent/graph.py`。

### 6.1 状态结构

`agent/state.py` 定义了 `AgentState`：

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    solution: list | None
    total_score: float | None
    covered_tasks: int
    best_strategy_code: str | None
```

含义：

| 字段 | 含义 |
| --- | --- |
| `messages` | LLM 对话消息和工具调用结果 |
| `solution` | 当前最优解 |
| `total_score` | 当前最优解总分 |
| `covered_tasks` | 当前最优解覆盖的任务数 |
| `best_strategy_code` | 当前最优排序策略代码 |

### 6.2 LLM 可用工具

`agent/tools.py` 暴露了三个工具：

1. `list_example_files()`
   - 列出 `example/` 目录下的 `.txt` 文件。

2. `peek_data(filename, n=20)`
   - 查看数据前 `n` 行。
   - 统计候选数量、任务数、骑手数、分数范围、意愿度范围、单任务/多任务数量。

3. `execute_strategy(filename, strategy_code)`
   - 执行 LLM 写出来的 `sort_key(c)` 策略。
   - 先按 `sort_key` 对候选排序。
   - 再贪心选择不冲突的任务包。
   - 返回 `solution`、`total_score`、`covered_tasks`、`bundle_count` 等结果。

### 6.3 策略形式

Agent 不是直接写完整求解器，而是写一个排序函数：

```python
def sort_key(c):
    return (c["score_per_task"], -c["willingness"], -c["task_count"])
```

`c` 是一个候选字典，包含：

| 字段 | 含义 |
| --- | --- |
| `task_str` | 原始任务字符串 |
| `task_ids` | 任务 ID 集合 |
| `courier_id` | 骑手 ID |
| `score` | 总分 |
| `willingness` | 意愿度 |
| `task_count` | 任务包中任务数量 |
| `score_per_task` | 平均每个任务的分数 |

排序后，程序从前往后选择候选：

- 如果骑手已经被用过，跳过。
- 如果任务和已选任务有重叠，跳过。
- 否则加入解集。
- 如果所有任务都已经覆盖，提前结束。

### 6.4 最多尝试 5 次策略

`agent/graph.py` 中定义：

```python
MAX_STRATEGIES = 5
```

Agent 最多调用 `execute_strategy()` 5 次。每次工具返回结果后，`tools_node()` 会比较新结果和当前最好结果。

比较逻辑在 `_is_better()`：

```python
def _is_better(new_score: float, new_tasks: int, best_score, best_tasks) -> bool:
    if best_score is None:
        return True
    if new_tasks != best_tasks:
        return new_tasks > best_tasks
    return new_score < best_score
```

也就是说：

1. 优先覆盖更多任务。
2. 覆盖任务数相同时，选择总分更低的方案。

### 6.5 自动写回 solver.py

当 Agent 结束后，会进入 `save_solver_node()`：

```python
def save_solver_node(state: AgentState) -> dict:
    ...
```

它会把最佳 `sort_key` 嵌入到固定模板中，生成新的 `solver.py`。模板包括：

- 输入解析函数 `_parse_input()`。
- 候选字典构造逻辑。
- Agent 找到的 `sort_key()`。
- 通用贪心选择逻辑。

所以 `solver.py` 可以看作 Agent 的最终产物。

## 7. 各文件说明

### `main.py`

命令行入口：

```python
query = input("问：")
result = run_agent(query)
```

职责：

- 接收用户输入。
- 调用 `agent.run_agent()`。
- 打印 `total_score`、`covered_tasks` 和 `solution`。

适合用于“让 Agent 自动试策略并生成 solver”的场景。

### `solver.py`

当前静态求解器。核心步骤：

1. `_parse_input()` 解析输入文本。
2. 为每条候选构造字典。
3. 使用内置 `sort_key()` 排序。
4. 贪心选择无冲突候选。
5. 返回 `[(task_id_list_str, [courier_id]), ...]`。

当前 `sort_key()` 是：

```python
def sort_key(c):
    return (c["score_per_task"], -c["willingness"], -c["task_count"])
```

含义：

1. 优先选择平均每个任务分数更低的候选。
2. 分数接近时，优先选择意愿度更高的骑手。
3. 再优先选择包含任务数更多的任务包。

### `generartor.py`

随机数据生成脚本。注意文件名是 `generartor.py`，不是常见拼写 `generator.py`。

它会生成：

- 任务 ID：`T0000`、`T0001` 等。
- 骑手 ID：`C000`、`C001` 等。
- 单任务包和两任务包。
- 每条候选的 `total_score` 和 `willingness`。

生成完成后，它还会调用 `common.parser.parse_input()` 读回文件，并打印候选数、分数范围、单任务/多任务数量。

### `evaluate.py`

命令行评测入口：

```bash
python evaluate.py example/large_seed301.txt
```

它会读取指定数据文件，调用当前 `solver.solve(input_text)`，再用 `common.evaluator.evaluate_solution()` 计算结果。它不需要 DeepSeek API Key，也不会修改 `solver.py`。

可以通过 `--penalty` 修改漏任务惩罚：

```bash
python evaluate.py example/large_seed301.txt --penalty 100
```

也可以用 `--json` 输出完整评测字典，方便后续脚本或 Agent 复用。

### `common/evaluator.py`

通用评测器：

```python
def evaluate_solution(
    input_text: str,
    solution: list,
    missing_task_penalty: float = 100.0,
) -> dict:
    ...
```

它做两类事情。

第一类是硬性合法性检查：

- 是否有任务重复。
- 是否有骑手重复。
- 返回的 `(task_id_list_str, courier_id)` 是否存在于原始输入候选中。
- 返回格式是否符合 `(task_id_list_str, [courier_id, ...])`。

第二类是指标统计和综合评分：

```text
objective_score = total_score + missing_tasks * missing_task_penalty
```

其中默认：

```text
missing_task_penalty = 100.0
```

这个参数表示“漏掉一个任务要付出多少额外成本”。它解决了“只看覆盖数太死板”和“只看总分会让空答案占便宜”的问题。综合评分越低越好，但只有 `valid = True` 的方案才应该拿来认真比较。

### `common/parser.py`

通用解析器：

```python
def parse_input(input_text: str) -> list:
    ...
```

返回列表中每个元素是：

```python
(score, task_id_list_str, courier_id, willingness)
```

注意顺序和原始输入列不同。原始输入是：

```text
task_id_list, courier_id, total_score, willingness
```

解析后变成：

```python
score, task_id_list_str, courier_id, willingness
```

### `agent/model.py`

模型初始化文件：

```python
load_dotenv()

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v4",
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
)
```

职责：

- 从 `.env` 读取 `DEEPSEEK_API_KEY`。
- 初始化 DeepSeek 聊天模型。
- 提供给 `agent/graph.py` 绑定工具使用。

### `agent/tools.py`

工具函数集合。主要服务对象是 LLM，不是普通用户直接调用。

核心工具：

- `list_example_files()`：列文件。
- `peek_data()`：看数据和统计。
- `execute_strategy()`：执行 LLM 给出的排序策略。

`execute_strategy()` 内部会使用：

```python
namespace = {"math": math, "__builtins__": {}}
exec(strategy_code, namespace)
```

它限制了内置函数，理论上减少了策略代码做无关操作的能力。但它仍然是在本地执行动态代码，因此只适合执行可信 Agent 生成的简单排序函数。

### `agent/graph.py`

项目最核心的 Agent 编排文件。

包含：

- 系统提示词 `SYSTEM_PROMPT`。
- 工具注册 `_tool_fns`。
- LLM 节点 `agent_node()`。
- 工具执行节点 `tools_node()`。
- 求解器保存节点 `save_solver_node()`。
- 路由函数 `should_continue()`。
- LangGraph 图构建逻辑。
- 对外入口 `run_agent()`。

工作流可以概括为：

```text
用户输入
  ↓
agent_node 调用 LLM
  ↓
如果 LLM 要调用工具 → tools_node 执行工具
  ↓
工具结果回到 LLM
  ↓
最多尝试 5 次 execute_strategy
  ↓
save_solver_node 写回 solver.py
  ↓
结束
```

### `agent/state.py`

定义 LangGraph 的状态结构，见第 6.1 节。

### `agent/__init__.py`

导出：

```python
from agent.graph import run_agent, agent_graph
```

所以 `main.py` 可以直接写：

```python
from agent import run_agent
```

## 8. 当前 solver.py 策略说明

当前 `solver.py` 使用的是简单贪心策略。

它先为每条候选计算：

```python
score_per_task = score / task_count
```

然后按下面的优先级排序：

```python
(score_per_task, -willingness, -task_count)
```

解释：

1. `score_per_task` 越小越靠前：优先低成本覆盖任务。
2. `-willingness` 越小代表 `willingness` 越大：优先高意愿骑手。
3. `-task_count` 越小代表 `task_count` 越大：优先多任务包。

排序完成后，从前往后扫描：

```text
如果骑手未使用，且任务都未覆盖，则选择该候选。
否则跳过。
```

该算法不是全局最优算法，而是贪心启发式算法。优点是实现简单、速度快、容易解释；缺点是可能被局部最优影响。

在当前 `example/large_seed301.txt` 上，实测当前 `solver.py`：

| 指标 | 数值 |
| --- | --- |
| 候选数量 | 33780 |
| 唯一任务数 | 40 |
| 唯一骑手数 | 80 |
| 选择任务包数 | 28 |
| 覆盖任务数 | 40 |
| 总分 | 约 `399.904` |
| 任务冲突 | 无 |
| 骑手冲突 | 无 |

## 9. 样例数据说明

当前样例文件：

```text
example/large_seed301.txt
```

统计结果：

| 指标 | 数值 |
| --- | --- |
| 候选行数 | 33780 |
| 唯一任务数 | 40 |
| 唯一骑手数 | 80 |
| 分数范围 | `10.002` ~ `100.0` |
| 平均分数 | 约 `56.3942` |
| 意愿度范围 | `0.01` ~ `0.9498` |
| 单任务候选 | 3200 |
| 多任务候选 | 30580 |

数据中多任务候选很多，所以当前策略会优先考虑“平均每任务分数低”的任务包，而不是只看任务包总分。

## 10. 注意事项与改进方向

### 注意事项

- `main.py` 运行 Agent 时需要 DeepSeek API Key；`solver.py` 不需要。
- Agent 会写回 `solver.py`，所以运行 Agent 前最好备份当前求解器。
- `execute_strategy()` 会执行 LLM 生成的 Python 代码，虽然限制了 `__builtins__`，但仍应只用于可信环境。
- `generartor.py` 的文件名拼写不是 `generator.py`，运行时要按实际文件名输入。
- 当前贪心策略无法保证全局最优。

### 可能的改进方向

- 增加更多策略尝试次数，或把 `MAX_STRATEGIES` 从 5 调大。
- 在 `execute_strategy()` 中支持更复杂的策略，例如二阶段贪心、局部搜索、替换优化。
- 为 `solver.py` 增加本地评测脚本，自动检查任务冲突、骑手冲突、覆盖数和总分。
- 让 Agent 同时比较多个样例文件，避免只对一个样例过拟合。
- 把 `solver.py` 的解析逻辑复用 `common/parser.py`，减少重复代码；如果评测环境只提交单文件，则仍需保留独立解析逻辑。

## 11. 推荐阅读顺序

如果你想快速读懂这个项目，建议按下面顺序看：

1. 先看 `solver.py`，理解最终提交函数长什么样。
2. 再看 `common/parser.py`，理解输入如何解析。
3. 然后看 `agent/tools.py`，理解 Agent 如何测试策略。
4. 最后看 `agent/graph.py`，理解 LangGraph 如何组织 LLM、工具和写回流程。
5. 如果要生成数据，再看 `generartor.py`。
