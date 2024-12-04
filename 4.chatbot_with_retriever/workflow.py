import os
import functools

from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

from langchain import hub
from langchain.agents import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from typing import Annotated, TypedDict, Union

from langchain import hub
from langchain.agents import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool

#langsmith
import configparser 
from langgraph.prebuilt import ToolNode

import configparser 
#config = configparser.ConfigParser()
#config.read('config.ini')
TAVILY_API_KEY = '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'
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

from typing import Annotated, List, Tuple, Union, Literal

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)

tavily_tool = TavilySearchResults(max_results=5)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class ChatEngine():
    def __init__(self,ChatState):
        self.LLM = ChatOllama(
            name="chat_llama3", 
            model="krith/meta-llama-3.1-8b-instruct:IQ2_M", 
            temperature=0
            )
        self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
            )
        self.db = Chroma(
            persist_directory="data/chroma_db", 
            embedding_function=self.embeddings
            )
        self.retriever = self.db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k":5}
            )
        self.retriever_tools = create_retriever_tool(
            self.retriever,
            "retrieve_resume",
            "Search and return information about candidate resume",
            )
        self.system_message = "you are AI assistant that can help recruiter to check the resume"
        self.tools = [self.retriever_tools]
        self.agent = self.create_react_agent(
            self.LLM, 
            self.tools, 
            self.system_message
            )
        self.workflow, self.chain = self.agent_chain(ChatState)


    def create_react_agent(self,llm, tools, system_message: str):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    " You are a helpful AI assistant"
                    " You will get a question from user."
                    " You can answer the question by using tools or not"
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_tools(tools)
    
    def agent_node(self,state, agent, name):
        result = agent.invoke(state)

        if isinstance(result, FunctionMessage):
            pass
        else:
            
            result = AIMessage(**result.dict(exclude={"type", "name"}), role=name, name=name)
        return {
            "messages": [result]
        }

    def router(self,state) -> Literal["call_tool", "__end__"]:
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        print('router')
        if last_message.tool_calls:
            # The previous agent is invoking a tool
            return "call_tool"
        return "__end__"
    
    def agent_chain(self, ChatState):
        agent = functools.partial(self.agent_node, agent=self.agent, name="smart_agent")
        tool_node = ToolNode(self.tools)

        # Define workflow and Direction
        workflow = StateGraph(ChatState)
        workflow.add_node("agent", agent)
        workflow.add_node("tool", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self.router, {"call_tool": "tool", "__end__": END}
        )
        workflow.add_edge("tool","agent")

        agent_chain = workflow.compile()

        return workflow, agent_chain
