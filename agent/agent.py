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
from langgraph.checkpoint.memory import MemorySaver

# --- Import tool logic AND the tools themselves ---
from tools import (
    available_tools,
    _fetch_data_logic, 
    _preprocess_data_logic, 
    _predict_performance_logic, 
    _visualize_data_logic
)

# --- Initialize the LLM ---
# (Your LLM setup code is correct and does not need changes)
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

# --- DEFINE THE AUTONOMOUS PROMPT ---
system_prompt = """You are an expert financial assistant. Your primary goal is to help users by analyzing stocks.

When a user asks you to 'analyze' a stock, you MUST perform the full sequence of actions without asking for permission:
1. First, call the `fetch_data` tool with the correct stock ticker.
2. Second, call the `preprocess_data` tool.
3. Third, call the `predict_performance` tool.
4. Finally, call the `visualize_data` tool to complete the analysis.

Only after the `visualize_data` tool has been called is your task complete. If you encounter an error, inform the user.
"""

# --- DEFINE THE GRAPH NODES ---

def agent_node(state: AgentState):
    """The "brain" of the agent. Decides the next tool to call."""
    print("\n--- AGENT: Deciding next step ---")
    messages = state['messages']
    if not messages or messages[0].type != "system":
        final_messages = [("system", system_prompt)] + list(messages)
    else:
        final_messages = messages
    response = llm.bind_tools(available_tools).invoke(final_messages)
    return {"messages": [response]}

def execute_tool_node(state: AgentState):
    """The "bridge" that executes real logic and updates the state."""
    print("\n--- TOOLS: Executing tools ---")
    action_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls), None)
    if not action_message:
        raise ValueError("No tool calls found in the last AIMessage.")

    tool_outputs = []
    current_state_updates = {}
    for tool_call in action_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        print(f"LLM decided to call tool: {tool_name} with args: {tool_args}")
        try:
            if tool_name == "fetch_data":
                output = _fetch_data_logic(ticker=tool_args.get("ticker"))
                current_state_updates["fetched_df"] = output
                current_state_updates["ticker"] = tool_args.get("ticker")
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Data fetched successfully.]"))
            elif tool_name == "preprocess_data":
                output = _preprocess_data_logic(df=state["fetched_df"])
                current_state_updates["processed_df"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Data preprocessed successfully.]"))
            elif tool_name == "predict_performance":
                output = _predict_performance_logic(processed_data=state["processed_df"])
                current_state_updates["prediction"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=output))
            elif tool_name == "visualize_data":
                output = _visualize_data_logic(data=state["fetched_df"], prediction=state["prediction"], ticker=state["ticker"])
                current_state_updates["image_base64"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Image generated successfully.]"))
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {repr(e)}"
            tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=json.dumps({"error": error_msg})))
            current_state_updates["error"] = error_msg
            print(error_msg)
    current_state_updates["messages"] = tool_outputs
    return current_state_updates

# THIS NODE IS NOW SIMPLER
def synthesize_final_answer_node(state: AgentState):
    """Creates the final user-facing response with text and image."""
    print("\n--- AGENT: Synthesizing final answer ---")
    ticker = state.get("ticker", "the stock")
    prediction = state.get("prediction", "unavailable")
    image_base64 = state.get("image_base64")

    response_content = (
        f"Based on my analysis, the model predicts that **{ticker.upper()}** "
        f"will **{prediction.upper()}** the market.\n\n"
        "Here is a visualization of the key financial metrics:"
    )
    final_message = AIMessage(content=response_content)
    if image_base64:
        setattr(final_message, 'image_base64', image_base64)
    
    # We only return the message here. The cleanup happens in the next node.
    return {"messages": [final_message]}

# --- NEW, DEDICATED CLEANUP NODE ---
def cleanup_state_node(state: AgentState):
    """Clears non-serializable objects from the state before the final save."""
    print("\n--- SYSTEM: Cleaning up state for saving ---")
    return {"fetched_df": None, "processed_df": None}


# --- DEFINE THE ROUTER ---
def router(state: AgentState) -> str:
    """The primary router for the graph."""
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tool"
    if isinstance(last_message, ToolMessage):
        if state.get("error"):
            return END # Or a dedicated error handling node
        ai_message_before = state['messages'][-2]
        if ai_message_before.tool_calls[0]['name'] == 'visualize_data':
            return "synthesize_final_answer"
        return "agent"
    return END

# --- BUILD THE STATEFUL GRAPH WITH THE NEW WIRING ---
memory = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("synthesize_final_answer", synthesize_final_answer_node)
# Add the new cleanup node
workflow.add_node("cleanup_state", cleanup_state_node) 

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {"execute_tool": "execute_tool", "__end__": END}
)
workflow.add_conditional_edges(
    "execute_tool",
    router,
    {"agent": "agent", "synthesize_final_answer": "synthesize_final_answer", "__end__": END}
)

# --- THIS IS THE KEY WIRING CHANGE ---
# After synthesizing the answer, go to cleanup, THEN end.
workflow.add_edge("synthesize_final_answer", "cleanup_state")
workflow.add_edge("cleanup_state", END)


app = workflow.compile(checkpointer=memory)

# --- MAIN TEST BLOCK ---
if __name__ == '__main__':
    def run_conversation(session_id: str, user_input: str):
        print(f"\n--- User: {user_input} ---")
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=user_input)]}
        final_message = None
        for event in app.stream(inputs, config=config, stream_mode="values"):
            final_message = event["messages"][-1]
        if final_message:
            print(f"\n--- Final Assistant Response ---\n{final_message.content}")
            if hasattr(final_message, 'image_base64'):
                print("\n[Image was generated and attached to the final message]")

    conversation_id = f"test_session_{uuid.uuid4()}"
    run_conversation(conversation_id, "Please do a full analysis of the GOOGL stock.")