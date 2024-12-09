import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

class AgentState(TypedDict):
    agent: str
    initialMessage: str
    responseToUser: str
    lnode: str
    category: str
    sessionState: Dict

class Category(BaseModel):
    category: str

class FirstAgent:
    def __init__(self, api_key: str):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

        workflow = StateGraph(AgentState)

        workflow.add_node("classifier",self.classifier)
        workflow.add_node("complaint",self.complaintAgent)
        workflow.add_node("sales",self.salesAgent)
        workflow.add_node("testimonial",self.testimonialAgent)
        workflow.add_node("catchall",self.catchallAgent)

        workflow.add_edge(START, "classifier")
        workflow.add_conditional_edges("classifier", self.main_router)
        workflow.add_edge("complaint", END)
        workflow.add_edge("sales", END)
        workflow.add_edge("testimonial", END)

        self.graph = workflow.compile()

    def classifier(self, state: AgentState):
        category = "DummyCategory"
        return {"category": category}

    def main_router(self, state: AgentState):
        category = state.get("category")
        if category == "complaint":
            return "complaint"
        elif category == "sales":
            return "sales"
        elif category == "testimonial":
            return "testimonial"
        else:
            return "catchall"

    def complaintAgent(self, state: AgentState):
        return {"responseToUser": "THis is a complaint"}

    def salesAgent(self, state: AgentState):
        return {"responseToUser": "This is a sales agent"}

    def testimonialAgent(self, state: AgentState):
        return {"responseToUser": "This is a testimonial agent"}

    def catchallAgent(self, state: AgentState):
        return {"responseToUser": "This is a catchall agent"}
        

