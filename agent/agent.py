# agent.py
import os
import json
from typing import TypedDict, List, Annotated, Any
import pandas as pd
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- Initialize the LLM ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_API_BASE,
    temperature=0,
)

# --- Import tool logic ---
from tools import available_tools

# --- Define Agent State ---
class AgentState(TypedDict):
    input: str
    ticker: str
    fetched_df: Any
    processed_df: Any
    prediction: str
    image_base64: str
    messages: Annotated[List[AnyMessage], add_messages]
    error: str

# --- DEFINE THE PROMPT ---
system_prompt = """You are a smart, conversational financial assistant.
Your goal is to help users analyze stocks by using the tools available to you.

- When the user asks you to analyze a stock for the first time, you should start by using the `fetch_data` tool.
- If you have already fetched data, you can use the `preprocess_data` tool on it.
- If you have preprocessed data, you can use the `predict_performance` tool.
- If you have a prediction and the original data, you can use the `visualize_data` tool.
- You can use one or more tools in a single turn if it makes sense.
- If you don't have the necessary data for a tool, inform the user what you need to do first.
- If the user's request is unclear, ask for clarification.
- Always be helpful and clear in your responses."""

# --- DEFINE THE GRAPH NODES ---

def agent_node(state: AgentState):
    """Invokes the LLM to decide the next action or respond to the user."""
    print("\n--- AGENT: Deciding next step ---")
    
    messages = state['messages']
    
    if not messages or messages[0].type != "system":
        final_messages = [("system", system_prompt)] + list(messages)
    else:
        final_messages = messages

    response = llm.bind_tools(available_tools).invoke(final_messages)
    return {"messages": [response]}

# --- DEFINE THE ROUTER ---
def router(state: AgentState) -> str:
    """Decides whether to continue calling tools or finish the turn."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- BUILD THE STATEFUL GRAPH ---
memory = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(available_tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")
app = workflow.compile(checkpointer=memory)

# --- MAIN TEST BLOCK ---
if __name__ == '__main__':
    def run_conversation(session_id: str, user_input: str):
        print(f"\n--- User: {user_input} ---")
        
        # === THIS IS THE FIX ===
        # The key must be "thread_id" for this version of LangGraph
        config = {"configurable": {"thread_id": session_id}}
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        final_message = None
        for event in app.stream(inputs, config=config, stream_mode="values"):
            final_message = event["messages"][-1]
        
        if final_message:
            if final_message.tool_calls:
                 print(f"--- Assistant (ended on tool call): {final_message.tool_calls} ---")
            else:
                 print(f"--- Assistant (response): {final_message.content} ---")
        else:
            print("--- Agent finished without a new message. ---")


    conversation_id = f"test_session_{uuid.uuid4()}"
    run_conversation(conversation_id, "Analyze GOOGL for me please.")
    run_conversation(conversation_id, "Okay, now please visualize the results.")