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

def create_llm_message(system_prompt):
    # Initialize empty list to store messages
    resp = []
    
    # Add system prompt as the first message. This will provide the overall instructions to LLM.
    resp.append(SystemMessage(content=system_prompt))
    
    # Get chat history from Streamlit's session state
    msgs = st.session_state.messages
    
    # Iterate through chat history, and based on the role (user or assistant) tag it as HumanMessage or AIMessage
    for m in msgs:
        if m["role"] == "user":
            # Add user messages as HumanMessage
            resp.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # Add assistant messages as AIMessage
            resp.append(AIMessage(content=m["content"]))
    
    # Return the formatted message list
    return resp

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
        CLASSIFIER_PROMPT=f"""
        You are an expert with deep knowledge of Customer Support. 
        Your job is to comprehend the message from the user even if it lacks specific keywords, 
        always maintain a friendly, professional, and helpful tone. 
        If a user greets you, greet them back by mirroring user's tone and verbosity, and offer assitance. 

        Based on user query, accurately classify customer requests into one of the following categories based on context 
        and content, even if specific keywords are not used.
        1. Complaint
        2. Sales
        3. Testimonial
        4. Other
        """
        llm_messages = create_llm_message(CLASSIFIER_PROMPT)
        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)

        category = llm_response.category
        print(f"{category=}, {llm_response=}")
        return {"category": category}

    def main_router(self, state: AgentState):
        print(f"{state=}")
        category = state.get("category")
        print(f"{category=}")
        if category == "Complaint":
            return "complaint"
        elif category == "Sales":
            return "sales"
        elif category == "Testimonial":
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


