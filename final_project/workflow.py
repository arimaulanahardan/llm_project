# Mandatory Library
import os
from dataclasses import dataclass
from typing import Iterator, Dict, Any
from langchain_core.messages import (
    HumanMessage,
)

from langgraph.graph.graph import CompiledGraph

from src.agent import AnalystAgentState, AnalystAgent


def initialize_chatbot():
    financial_analyst_agent = AnalystAgent()
    data_scientist_agent = financial_analyst_agent.agent_chain(AnalystAgentState)
    return data_scientist_agent

def stream_events(agent: CompiledGraph, message: str, stock_symbol: str) -> Iterator[Dict[str, Any]]:
    messages = [HumanMessage(content=message)]
    # Generate response if last message is not from assistant
    inputs = {"messages": messages,'stock':stock_symbol}
    config = {"configurable": {"thread_id": "1"}}
    event = agent.stream(inputs, config, debug=True, stream_mode="values")
    return event