from typing import Annotated

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    solution: list | None
    total_score: float | None
    covered_tasks: int
    best_strategy_code: str | None

