# app.py
import streamlit as st
import os
import uuid
import base64
import pandas as pd
# --- Import de l'agent LangGraph ---
from agent import app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import base64
import os

# Fonction pour encoder une image locale en Base64
def get_image_as_base64(path):
    # Vérifie si le fichier existe
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

STELLA_AVATAR = "assets/avatar_stella.png" # Chemin vers l'avatar de Stella

st.set_page_config(page_title="Assistant financier IA", page_icon="📈", layout="wide")
st.title("📈 Assistant financier IA")

# --- Initialisation du session_state pour les messages et d'un ID de session unique ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello!  Je suis Stella. Je peux t'aider à analyser le potentiel d'une action. Que souhaites-tu faire ?")]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Affichage des messages existant depuis l'historique---
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=STELLA_AVATAR):
            st.write(msg.content)

            # Logique pour l'image
            if hasattr(msg, 'image_base64') and msg.image_base64:
                try:
                    st.image(base64.b64decode(msg.image_base64), caption="Analyse Financière")
                except Exception as e:
                    st.error(f"Impossible d'afficher l'image : {e}")

            # Logique pour le DataFrame
            if hasattr(msg, 'dataframe_json') and msg.dataframe_json:
                try:
                    df = pd.read_json(msg.dataframe_json, orient='split')
                    st.dataframe(df) 
                except Exception as e:
                    st.error(f"Impossible d'afficher le DataFrame : {e}")

    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

# --- Gestion de l'input utilisateur ---
if prompt := st.chat_input("Qu'est ce que je peux faire pour toi aujourd'hui ? 😊​"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant", avatar=STELLA_AVATAR):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("🧠 Hmm, laisse moi réfléchir...")

        inputs = {"messages": [HumanMessage(content=prompt)]}
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        try:
            final_message = None
            # Streame les events depuis le Graph pour arriver au résultat final
            for event in app.stream(inputs, config=config, stream_mode="values"):
                # On met à jour la variable `final_message` à chaque tour
                final_message = event["messages"][-1]

                # Si le dernier message est une décision de l'IA d'appeler un outil...
                if isinstance(final_message, AIMessage) and final_message.tool_calls:
                    tool_name = final_message.tool_calls[0]['name']
                    
                    # On met à jour le placeholder avec un message pertinent
                    if tool_name == 'fetch_data':
                        ticker = final_message.tool_calls[0]['args'].get('ticker', '')
                        thinking_placeholder.write(f"🔍 Recherche des données pour **{ticker}**...")
                    elif tool_name == 'preprocess_data':
                        thinking_placeholder.write("⚙️ Préparation des données pour l'analyse...")
                    elif tool_name == 'predict_performance':
                        thinking_placeholder.write("📈 Lancement du modèle de prédiction...")
                    elif tool_name == 'visualize_data':
                        thinking_placeholder.write("📊 Création de la visualisation finale...")

            thinking_placeholder.empty()

            if final_message:
                if isinstance(final_message, AIMessage) and not final_message.tool_calls:
                    st.session_state.messages.append(final_message)
                elif isinstance(final_message, ToolMessage):
                    error_content = f"Une erreur est survenue lors de l'exécution : {final_message.content}"
                    st.session_state.messages.append(AIMessage(content=error_content))
            else:
                st.session_state.messages.append(AIMessage(content="Désolé, je n'ai pas pu formuler de réponse. 😔"))

        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"Oups ! Une erreur inattendue s'est produite : {e}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
            import traceback
            traceback.print_exc()

        st.rerun()