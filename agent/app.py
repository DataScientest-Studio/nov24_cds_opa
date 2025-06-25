# app.py
import streamlit as st
import os
import uuid
import base64

# --- Import your LangGraph agent ---
from agent import app
from langchain_core.messages import HumanMessage, AIMessage

# --- DEBUGGING LINES (Optional) ---
# print(f"DEBUG app.py (Top of script): OPENROUTER_API_KEY loaded: {os.getenv('OPENROUTER_API_KEY', 'None')[:5]}")
# print(f"DEBUG app.py (Top of script): OPENROUTER_MODEL loaded: {os.getenv('OPENROUTER_MODEL', 'None')}")

st.set_page_config(page_title="AI Investing Assistant", page_icon="📈", layout="wide")
st.title("📈 AI Investing Assistant")

# --- Initialize session state for messages and a unique session ID ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I can help you analyze stock potential. What would you like to do?")]
# We need a unique ID for each chat session to maintain memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Display existing messages from history ---
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
            # Check for and display image attached to the message
            if hasattr(msg, 'image_base64') and msg.image_base64:
                try:
                    st.image(base64.b64decode(msg.image_base64), caption="Financial Analysis")
                except Exception as e:
                    st.error(f"Could not display image: {e}")
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

# --- Handle user input ---
if prompt := st.chat_input("What would you like to analyze?"):
    # Add user message to history and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # Display a "Thinking..." message while the agent is running
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("🧠 Thinking...")

        # Prepare the inputs for the agent
        inputs = {"messages": [HumanMessage(content=prompt)]}
        # The config MUST use "thread_id" for your version of LangGraph
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        try:
            final_message = None
            # Stream the events from the graph to get the final result
            for event in app.stream(inputs, config=config, stream_mode="values"):
                # The last message in the list is the latest output
                final_message = event["messages"][-1]

            # After the stream is done, we have the final message
            thinking_placeholder.empty() # Remove the "Thinking..." message

            if final_message:
                # Add the final response to our session history before rerunning
                st.session_state.messages.append(final_message)
            else:
                st.session_state.messages.append(AIMessage(content="Sorry, I encountered an issue and could not get a response."))

        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"An unexpected error occurred: {e}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
            import traceback
            traceback.print_exc()

        # Rerun the script to display the latest messages from session_state
        st.rerun()