# app.py
import streamlit as st
import os
import uuid
import base64
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from io import StringIO


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
st.title("📈 Analyste financier IA")

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

            # Logique pour le DataFrame (reste inchangée)
            if hasattr(msg, 'dataframe_json') and msg.dataframe_json:
                try:
                    df = pd.read_json(StringIO(msg.dataframe_json), orient='split')
                    st.dataframe(df) 
                except Exception as e:
                    st.error(f"Impossible d'afficher le DataFrame : {e}")

            # --- LOGIQUE DE VISUALISATION UNIQUE ET MISE À JOUR ---
            if hasattr(msg, 'plotly_json') and msg.plotly_json:
                try:
                    fig = go.Figure(pio.from_json(msg.plotly_json))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Impossible d'afficher le graphique : {e}")

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

        inputs = {"messages": st.session_state.messages} # On envoie tout l'historique
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        final_response = None
        
        try:
            # On streame les events pour afficher les étapes en temps réel
            for event in app.stream(inputs, config=config, stream_mode="values"):
                # `event` est l'état complet du graphe à chaque étape
                # On cherche la dernière AIMessage qui contient un appel d'outil
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    tool_name = last_message.tool_calls[0]['name']
                    tool_args = last_message.tool_calls[0]['args']
                    ticker = last_message.tool_calls[0]['args'].get('ticker', '')
                    company_name = tool_args.get('company_name', 'l\'entreprise demandée')
                    if tool_name == 'search_ticker':
                        thinking_placeholder.write(f"🔍 Recherche du ticker pour **{company_name}**...")
                    elif tool_name == 'fetch_data':
                        thinking_placeholder.write(f"🔍 Recherche des données pour **{ticker}**...")
                    elif tool_name == 'preprocess_data':
                        thinking_placeholder.write("⚙️ Préparation des données pour l'analyse...")
                    elif tool_name == 'predict_performance':
                        thinking_placeholder.write("📈 Lancement du modèle de prédiction...")
                    elif tool_name == 'create_dynamic_chart':
                        thinking_placeholder.write("📊 Création de la visualisation demandée...")

                # La réponse finale est la dernière AIMessage SANS appel d'outil
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    final_response = last_message

            thinking_placeholder.empty()

            # Une fois le stream terminé, on traite la réponse finale
            if final_response:
                st.session_state.messages.append(final_response)
            else:
                # Si aucune réponse claire n'est trouvée, on affiche un message par défaut
                fallback_response = AIMessage(content="Désolé, je n'ai pas pu terminer ma pensée. Peux-tu reformuler ?")
                st.session_state.messages.append(fallback_response)

        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"Oups ! Une erreur inattendue s'est produite : {e}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
            import traceback
            traceback.print_exc()

        # Rafraîchit la page pour afficher le nouveau message ajouté à l'historique
        st.rerun()