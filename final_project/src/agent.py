from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage

from langchain_ollama import ChatOllama

from src.tools import get_financial_metrics, get_stock_prices
from langgraph.prebuilt import ToolNode, tools_condition

llm = ChatOllama(model="llama3.2:1b")

class AnalystAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    stock: str

class AnalystAgent():
    def __init__(self):
        self.tools = [get_stock_prices, get_financial_metrics]
        self.llm_with_tool = llm.bind_tools(self.tools)
        self.system_prompt = """
You are a fundamental analyst specializing in evaluating company (whose symbol is {company}) performance based on stock prices, technical indicators, and financial metrics. Your task is to provide a comprehensive summary of the fundamental analysis for a given stock.

You have access to the following tools:
1. **get_stock_prices**: Retrieves the latest stock price, historical price data and technical Indicators like RSI, MACD, Drawdown and VWAP.
2. **get_financial_metrics**: Retrieves key financial metrics, such as revenue, earnings per share (EPS), price-to-earnings ratio (P/E), and debt-to-equity ratio.

### Your Task:
1. **Input Stock Symbol**: Use the provided stock symbol to query the tools and gather the relevant information.
2. **Analyze Data**: Evaluate the results from the tools and identify potential resistance, key trends, strengths, or concerns.
3. **Provide Summary**: Write a concise, well-structured summary that highlights:
    - Recent stock price movements, trends and potential resistance.
    - Key insights from technical indicators (e.g., whether the stock is overbought or oversold).
    - Financial health and performance based on financial metrics.

### Constraints:
- Use only the data provided by user
- Avoid speculative language; focus on observable data and trends.

### Output Format:
Respond in the following format:
"stock": "<{company}>",
"price_analysis": "<Detailed analysis of stock price trends>",
"technical_analysis": "<Detailed time series Analysis from ALL technical indicators>",
"financial_analysis": "<Detailed analysis from financial metrics>",
"final Summary": "<Full Conclusion based on the above analyses>"
"Asked Question Answer": "<Answer based on the details and analysis above>
        """
        pass

    def fundamental_analyst(self,state):
        messages = [
            SystemMessage(content=self.system_prompt.format(company=state['stock'])),
        ]  + state['messages']
        return {
            'messages': self.llm_with_tool.invoke(messages)
        }

    def agent_chain(self,state:AnalystAgentState):
        graph_builder = StateGraph(state)

        graph_builder.add_node('fundamental_analyst', self.fundamental_analyst)
        graph_builder.add_edge(START, 'fundamental_analyst')
        graph_builder.add_node(ToolNode(self.tools))
        graph_builder.add_conditional_edges('fundamental_analyst', tools_condition)
        graph_builder.add_edge('tools', 'fundamental_analyst')

        graph = graph_builder.compile()

        return graph