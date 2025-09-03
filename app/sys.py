import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : list[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def process(state:AgentState) -> AgentState:
    """Processing Node"""

    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content = response.content))

    print(f"\nAI : {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node('PROCESSOR', process)
graph.add_edge(START, "PROCESSOR")
graph.add_edge("PROCESSOR", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter Message: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages":conversation_history})

    conversation_history = result["messages"]

    user_input = input("Enter Message: ")