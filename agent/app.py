# app.py
import streamlit as st
import os
import uuid
import base64
import pandas as pd
# --- Import de l'agent LangGraph ---
from agent import app
from langchain_core.messages import HumanMessage, AIMessage

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
img_base64 = get_image_as_base64(STELLA_AVATAR)

# --- Lignes de débugging (Optionel) ---
# print(f"DEBUG app.py (Top of script): OPENROUTER_API_KEY loaded: {os.getenv('OPENROUTER_API_KEY', 'None')[:5]}")
# print(f"DEBUG app.py (Top of script): OPENROUTER_MODEL loaded: {os.getenv('OPENROUTER_MODEL', 'None')}")

st.set_page_config(page_title="Assistant financier IA", page_icon="📈", layout="wide")
st.title("📈 Assistant financier IA")

# --- Initialisation du session_state pour les messages et d'un ID de session unique ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello!  Je suis Stella. Je peux t'aider à analyser le potentiel d'une action. Que souhaites-tu faire ?")]
# On a besoin d'un ID unique pour chaque session de chat afin de maintenir la mémoire
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
                    st.image(base64.b64decode(msg.image_base64), caption="Financial Analysis")
                except Exception as e:
                    st.error(f"Could not display image: {e}")

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
    # Ajout du message de l'utilisateur à l'historique pour l'affichage
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # Afficher le conteneur de la réponse de l'assistant
    with st.chat_message("assistant", avatar=STELLA_AVATAR):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("🧠 Hmm, laisse moi réfléchir...")

        # Prépare les entrées pour l'agent
        inputs = {"messages": [HumanMessage(content=prompt)]}
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        try:
            final_message = None
            # Streame les events depuis le Graph pour arriver au résultat final
            for event in app.stream(inputs, config=config, stream_mode="values"):
                # `event` est l'état complet de l'agent à chaque étape.
                # On regarde le dernier message pour savoir ce qui se passe.
                last_message = event["messages"][-1]

                # Si le dernier message est une décision de l'IA d'appeler un outil...
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    tool_name = last_message.tool_calls[0]['name']
                    
                    # On met à jour le placeholder avec un message pertinent
                    if tool_name == 'fetch_data':
                        ticker = last_message.tool_calls[0]['args'].get('ticker', '')
                        thinking_placeholder.write(f"🔍 Recherche des données financières pour **{ticker}**...")
                    elif tool_name == 'preprocess_data':
                        thinking_placeholder.write("⚙️ Préparation des données pour l'analyse...")
                    elif tool_name == 'predict_performance':
                        thinking_placeholder.write("📈 Génération de la prédiction de performance...")
                    elif tool_name == 'visualize_data':
                        thinking_placeholder.write("📊 Création de la visualisation finale...")
                
                # On met à jour `final_message` à chaque étape.
                # À la fin de la boucle, il contiendra la toute dernière réponse.
                final_message = last_message

            thinking_placeholder.empty() # On retire le message de réflexion

            if final_message and not final_message.tool_calls:
                # Ajouter la réponse finale à l'historique de session avant de relancer
                st.session_state.messages.append(final_message)
            else:
                st.session_state.messages.append(AIMessage(content="Désolé, j'ai rencontré un problème, et je n'ai pas pu formuler de réponse 😔."))

        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"Oups, je ne m'attendais pas à ça ! Une erreur inattendue s'est produite: {e}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
            import traceback
            traceback.print_exc()

        # Relancer le script pour aficher les derniers messages depuis session_state
        st.rerun()