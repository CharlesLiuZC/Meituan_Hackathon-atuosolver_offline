import json
import os
import textwrap

from langchain.messages import SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END

from agent.model import model
from agent.state import AgentState
from agent.tools import list_example_files, peek_data, execute_strategy

#最多调用次数
MAX_STRATEGIES = 5

#系统提示词
SYSTEM_PROMPT = f"""You are a task-assignment solver agent. Your job is to design and
execute a greedy strategy for the courier-task assignment problem.

## Problem
- Each candidate is a bundle of tasks assigned to a courier, with a score and willingness.
- You must select a subset of candidates such that:
  1. No task appears in more than one selected candidate.
  2. No courier appears in more than one selected candidate.
- Goal: minimize total score AND maximize the number of tasks covered.

## Rules
- You may call peek_data and list_example_files freely at any time.
- You have exactly {MAX_STRATEGIES} calls to execute_strategy. Use them wisely.
- After your {MAX_STRATEGIES}th execute_strategy result, you MUST summarize the best
  strategy you found and its results. Do NOT call execute_strategy again.

## Workflow

### Step 1 — Explore the data
Call peek_data(filename) to see the first 20 rows and overall statistics.
Look at: score distribution, willingness range, how many single vs multi-task
bundles exist, etc.

### Step 2 — Design your strategy
Write a Python ``sort_key`` function that ranks candidates.  The greedy algorithm
will pick the highest-ranked (lowest return value) non-conflicting candidate, then
move to the next, skipping any that overlap on tasks or couriers.

Your sort_key receives a dict ``c`` with these fields:
  c["score"]           — float, total bundle score (lower is better)
  c["task_count"]      — int, number of tasks in this bundle
  c["score_per_task"]  — float, c["score"] / c["task_count"]
  c["willingness"]     — float, courier willingness (higher is usually better)

The function must be named ``sort_key``.  Example:

    def sort_key(c):
        return (c["score_per_task"], -c["willingness"])

You may use ``math`` (already imported).  Do NOT access external files or the network.

### Step 3 — Execute and iterate
Call execute_strategy(filename, strategy_code) to run your strategy.
The result shows {{total_score, covered_tasks, bundle_count}}.

Analyze the result.  If it's not good enough, write an improved sort_key.
- Too few tasks covered?  → prefer bundles with more tasks.
- Score too high?        → penalize high-score or low-willingness candidates.
- Many singles left out? → prefer single-task candidates.

### Step 4 — Pick the best and report
After you've tried {MAX_STRATEGIES} strategies (or earlier if you're satisfied),
clearly report:
  - The best sort_key code
  - total_score, covered_tasks, bundle_count
  - Why this strategy worked best

Available tools:
- list_example_files()              — list .txt files in example/
- peek_data(filename, n=20)         — show first n rows + statistics
- execute_strategy(filename, code)  — run your sort_key function (max {MAX_STRATEGIES} calls)
"""

# Map tool name → function
_tool_fns = {
    "list_example_files": list_example_files,
    "peek_data": peek_data,
    "execute_strategy": execute_strategy,
}

model_with_tools = model.bind_tools(list(_tool_fns.values()))


def _count_strategy_calls(messages) -> int:
    """Count how many times execute_strategy has already returned a result."""
    return sum(
        1 for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "execute_strategy"
    )


def _is_better(new_score: float, new_tasks: int, best_score, best_tasks) -> bool:
    """Check if new result is better: prefer more tasks, then lower score."""
    if best_score is None:
        return True
    if new_tasks != best_tasks:
        return new_tasks > best_tasks
    return new_score < best_score


#<====================node========================>

#v1.0整体逻辑：agent->
def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """Execute tool calls and track the best execute_strategy result in state."""
    last_msg = state["messages"][-1]
    updates: dict = {"messages": []}

    best_score = state.get("total_score")
    best_tasks = state.get("covered_tasks", 0) or 0
    best_solution = state.get("solution")

    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_fn = _tool_fns.get(tool_name)

        if tool_fn is None:
            result_str = f"Unknown tool: {tool_name}"
        else:
            try:
                result = tool_fn(**tool_args)
                if isinstance(result, dict) and "error" not in result:
                    # Track best result for execute_strategy
                    if tool_name == "execute_strategy":
                        score = result.get("total_score", float("inf"))
                        tasks = result.get("covered_tasks", 0)
                        if _is_better(score, tasks, best_score, best_tasks):
                            best_score = score
                            best_tasks = tasks
                            best_solution = result.get("solution")
                            updates["total_score"] = best_score
                            updates["covered_tasks"] = best_tasks
                            updates["solution"] = best_solution
                            updates["best_strategy_code"] = tool_args.get("strategy_code", "")
                    result_str = json.dumps(result, ensure_ascii=False)
                else:
                    result_str = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                result_str = f"Tool error: {e}"

        updates["messages"].append(
            ToolMessage(content=result_str, name=tool_name, tool_call_id=tc["id"])
        )

    return updates


#<============save_solver_node================>

_SOLVER_PREFIX = '''"""Auto-generated solve function — best strategy from agent."""

import math


def _parse_input(input_text: str) -> list:
    """Parse the input text and return a list of candidate tuples."""
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        candidates.append(
            (score, task_id_list_str.strip(), courier_id.strip(), willingness)
        )
    return candidates


def solve(input_text: str) -> list:
    """Return [(task_id_list_str, [courier_id, ...]), ...]"""
    candidates_raw = _parse_input(input_text)
    if not candidates_raw:
        return []

    candidates = []
    for score, task_str, courier_id, willingness in candidates_raw:
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

'''

_SOLVER_SUFFIX = '''
    candidates.sort(key=sort_key)

    selected = []
    used_tasks = set()
    used_couriers = set()
    all_task_ids = set()
    for c in candidates:
        all_task_ids |= c["task_ids"]

    for c in candidates:
        if c["courier_id"] in used_couriers:
            continue
        if c["task_ids"] & used_tasks:
            continue
        selected.append((c["task_str"], [c["courier_id"]]))
        used_tasks |= c["task_ids"]
        used_couriers.add(c["courier_id"])
        if used_tasks == all_task_ids:
            break

    return selected
'''


def save_solver_node(state: AgentState) -> dict:
    """Generate solver.py from the best strategy code the agent discovered."""
    best_code = state.get("best_strategy_code")
    if not best_code:
        return {}

    indented_code = textwrap.indent(best_code.strip(), "    ")
    solver_content = _SOLVER_PREFIX + indented_code + "\n" + _SOLVER_SUFFIX

    solver_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "solver.py"
    )
    with open(solver_path, "w", encoding="utf-8") as f:
        f.write(solver_content)

    return {}


#<============routing_function===============>


#判断是否要继续调用工具，如果调用超5次execute_strategy，则直接返回当前最优解
#后续可将5次改为10s有效时间，这里因为没设置持久化所以10s内难以找到最有解，而选择调用5次execute_strategy
def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]

    if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
        return "save_solver"

    completed = _count_strategy_calls(state["messages"])
    for tc in last_msg.tool_calls:
        if tc.get("name") == "execute_strategy" and completed >= MAX_STRATEGIES:
            return "save_solver"

    return "tools"


# 构图
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tools_node)
graph_builder.add_node("save_solver", save_solver_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges(
    "agent", should_continue,
    {"tools": "tools", "save_solver": "save_solver"}
)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("save_solver", END)

agent_graph = graph_builder.compile()


def run_agent(user_input: str) -> AgentState:
    """Run the agent with a user query and return the final state."""
    from langchain.messages import HumanMessage

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "solution": None,
        "total_score": None,
        "covered_tasks": 0,
        "best_strategy_code": None,
    }
    return agent_graph.invoke(initial_state)
