# app.py
import streamlit as st
import os
import re
import base64

# --- Import your LangGraph agent ---
from agent import app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

# --- DEBUGGING LINES (KEEP TEMPORARILY) ---
debug_api_key = os.getenv("OPENROUTER_API_KEY")
debug_model = os.getenv("OPENROUTER_MODEL")
print(f"DEBUG app.py (Top of script): OPENROUTER_API_KEY loaded: {debug_api_key[:5]}...{debug_api_key[-5:] if debug_api_key else 'None'}")
print(f"DEBUG app.py (Top of script): OPENROUTER_MODEL loaded: {debug_model if debug_model else 'None'}")
# --- END DEBUGGING LINES ---

st.set_page_config(page_title="AI Investing Assistant", page_icon="📈", layout="wide")
st.title("📈 AI Investing Assistant")

# --- Initialize session state for messages ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Hello! I can help you analyze stock potential. What ticker are you interested in (e.g., 'Analyze AAPL')?")]

# --- Display existing messages ---
for msg in st.session_state.messages:
    if not isinstance(msg, BaseMessage):
        print(f"DEBUG app.py: Found non-BaseMessage in history: {msg}. Skipping.")
        continue

    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
            if hasattr(msg, 'image_base64') and msg.image_base64:
                try:
                    st.image(base64.b64decode(msg.image_base64), caption=f"Analysis")
                except Exception as e:
                    st.error(f"Could not display image: {e}")
    elif isinstance(msg, ToolMessage):
        with st.expander(f"Tool Call: {msg.name}"):
            st.json({"tool_id": msg.tool_call_id, "content": msg.content})

# --- Get user input ---
if prompt := st.chat_input("Enter a stock ticker to analyze..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    initial_agent_state = {
        "input": prompt,
        "ticker": "",
        "fetched_data_json": "",
        "processed_data_json": "",
        "prediction": "",
        "image_base64": "",
        "messages": st.session_state.messages.copy(),
        "error": ""
    }

    with st.chat_message("assistant"):
        st_thinking_placeholder = st.empty()
        final_agent_response = None
        final_state = None

        # In app.py, replace the try/except block with this:

        try:
            print("\n--- Streamlit: Starting agent stream ---")
            st_thinking_placeholder.write("Thinking...")

            final_state = None
            # Robustly collect all messages streamed from the agent
            all_stream_messages = []

            for chunk in app.stream(initial_agent_state, config={"configurable": {"thread_id": "any"}}):
                print(f"DEBUG app.py (Stream chunk): {chunk}")

                # Find and collect any messages from any node in the stream
                for key in chunk:
                    if isinstance(chunk[key], dict) and "messages" in chunk[key]:
                        # Ensure we only append BaseMessage objects
                        for msg in chunk[key]["messages"]:
                            if isinstance(msg, BaseMessage):
                                all_stream_messages.append(msg)

                if "__end__" not in chunk:
                    # Update UI with current step (optional but good for UX)
                    node_name = list(chunk.keys())[0]
                    if node_name == "llm_decide_tool":
                        st_thinking_placeholder.write("🧠 LLM is thinking...")
                    elif node_name == "execute_tool":
                        st_thinking_placeholder.write("🛠️ Executing a tool...")
                    elif node_name == "format_final_response":
                        st_thinking_placeholder.write("📝 Formatting response...")
                else:
                    # The stream is over, capture the final state
                    final_state = chunk["__end__"]
                    print(f"DEBUG app.py (Stream End): Final state captured.")
                    break

            st_thinking_placeholder.empty()

            final_agent_response = None
            # Search our collected messages for the final answer
            if all_stream_messages:
                print(f"DEBUG app.py: Searching through {len(all_stream_messages)} collected messages.")
                for msg in reversed(all_stream_messages):
                    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                        final_agent_response = msg
                        print(f"DEBUG app.py: Found final AIMessage response in collected stream.")
                        break

            if final_agent_response:
                st.write(final_agent_response.content)

                # Handle image display if present in the final state
                if final_state and final_state.get('image_base64'):
                    try:
                        st.image(base64.b64decode(final_state['image_base64']), caption="Financial Analysis")
                        # Attach image data to the message for session history
                        setattr(final_agent_response, 'image_base64', final_state['image_base64'])
                    except Exception as e:
                        print(f"DEBUG app.py: Error displaying image from final state: {e}")
                
                st.session_state.messages.append(final_agent_response)

            else:
                error_msg = "I could not find a final answer in the agent's response. Please try again."
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                print("DEBUG app.py: No valid final response found after checking all stream messages.")

            # Use st.rerun() to clear the input box and show the latest messages
            st.rerun()

        except Exception as e:
            st_thinking_placeholder.empty()
            error_msg = f"An unexpected error occurred during agent execution: {e}"
            st.error(error_msg)
            print(f"DEBUG app.py (Exception caught): {e}")
            # It's good practice to log the full traceback for debugging
            import traceback
            traceback.print_exc()
            st.session_state.messages.append(AIMessage(content=error_msg))
            st.rerun()
