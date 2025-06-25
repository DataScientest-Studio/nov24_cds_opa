# agent.py
import os
import json
from typing import TypedDict, List, Annotated, Any
import pandas as pd
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

# --- Import des tools ---
from tools import (
    available_tools,
    _fetch_data_logic, 
    _preprocess_data_logic, 
    _predict_performance_logic, 
    _visualize_data_logic
)

# --- Initalisation du LLM ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY n'a  pas été enregistrée comme variable d'environnement.")

llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_API_BASE,
    temperature=0,
)

# Objet AgentState pour stocker et modifier l'état de l'agent entre les nœuds
class AgentState(TypedDict):
    input: str
    ticker: str
    fetched_df_json: str
    processed_df_json: str
    prediction: str
    image_base64: str
    messages: Annotated[List[AnyMessage], add_messages]
    error: str

# --- Prompt système (définition du rôle de l'agent) ---
system_prompt = """Ton nom est Stella. Tu es une assistante experte financière. Ton but principal est d'aider les utilisateurs en analysant des stocks.

Quand un utilisateur te demande d'analyser un stock, tu DOIS suivre une séquence d'actions prédéfinie. Une étape clé de cette analyse est la prédiction, gérée par l'outil `predict_performance`.

**Logique de Prédiction :**
Mon modèle n'est pas conçu pour te dire si une action est "bonne", mais pour t'avertir si elle présente des signaux de "gros danger".
- Si l'outil `predict_performance` renvoie **"Risque Élevé Détecté"**, cela signifie que mon modèle est très confiant que l'action va sous-performer. 
Tu dois présenter cela comme un avertissement clair.
- Si l'outil renvoie **"Aucun Risque Extrême Détecté"**, cela ne veut PAS dire que l'action est un bon investissement. 
Explique simplement à l'utilisateur que mon analyse n'a pas trouvé de signaux de danger évidents, mais que cela ne constitue pas une recommandation d'achat.

**Séquence d'actions en cas de demande d'analyse complète:**
1. Premièrement, tu dois appeler le tool `fetch_data` avec le ticker correspondant à la demande de l'utilisateur.
2. Deuxièmement, tu dois appeler le tool `preprocess_data` pour préparer les données récupérées.
3. Troisièmement, tu dois appeler le tool `predict_performance` pour prédire la performance du stock.
4. Enfin, tu dois appeler le tool `visualize_data` pour compléter l'analyse.

Tu disposes également de deux outils pour afficher des données :
- `display_raw_data`: Utilise cet outil UNIQUEMENT si l'utilisateur demande explicitement les données 'brutes' ou 'originales'.
- `display_processed_data`: Utilise cet outil par défaut quand l'utilisateur demande de 'voir les données', 'montrer le tableau', etc., car ce sont les données les plus utiles pour l'analyse.

Ta tâche sera complète uniquement après que le tool `visualize_data` ait été appelé. Si tu rencontres une erreur, informe l'utilisateur.
Tu dois toujours répondre en français et tutoyer ton interlocuteur. Tu ne dois pas faire de suppositions sur les données financières, tu dois te baser uniquement sur les données récupérées par le tool `fetch_data`.
Lorsque l'utilisateur te demande qui tu es, ou comment tu fonctionnes, tu te présenteras et lui expliqueras ton fonctionnement.
Tes réponses doivent toujours être polies, claires et concises.
"""

# --- Définition des noeuds du Graph ---

# Noeud 1 : agent_node, point d'entrée
def agent_node(state: AgentState):
    """Le 'cerveau' de l'agent. Décide le prochain outil à appeler."""
    print("\n--- AGENT: Décision de la prochaine étape... ---")
    messages = state['messages']
    if not messages or not isinstance(messages[0], SystemMessage):
        final_messages = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        final_messages = messages
    response = llm.bind_tools(available_tools).invoke(final_messages)
    return {"messages": [response]}

# Noeud 2 : execute_tool_node, exécute les outils en se basant sur la décision de l'agent_node (Noeud 1).
def execute_tool_node(state: AgentState):
    """Le "pont" qui exécute la logique réelle et met à jour l'état."""
    print("\n--- OUTILS: Exécution d'un outil ---")
    action_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls), None)
    if not action_message:
        raise ValueError("Aucun appel d'outil trouvé dans le dernier AIMessage.")

    tool_outputs = []
    current_state_updates = {}
    for tool_call in action_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        print(f"Le LLM a décidé d'appeler le tool : {tool_name} - avec les arguments : {tool_args}")
        try:
            if tool_name == "fetch_data":
                output = _fetch_data_logic(ticker=tool_args.get("ticker"))
                current_state_updates["fetched_df_json"] = output.to_json(orient='split')
                current_state_updates["ticker"] = tool_args.get("ticker")
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données récupérées avec succès.]"))
            
            elif tool_name == "preprocess_data":
                fetched_df = pd.read_json(state["fetched_df_json"], orient='split')
                output = _preprocess_data_logic(df=fetched_df)
                current_state_updates["processed_df_json"] = output.to_json(orient='split')
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données prétraitées avec succès.]"))
            
            elif tool_name == "predict_performance":
                processed_df = pd.read_json(state["processed_df_json"], orient='split')
                output = _predict_performance_logic(processed_data=processed_df)
                current_state_updates["prediction"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=output))
            
            elif tool_name == "visualize_data":
                fetched_df = pd.read_json(state["fetched_df_json"], orient='split')
                output = _visualize_data_logic(data=fetched_df, prediction=state["prediction"], ticker=state["ticker"])
                current_state_updates["image_base64"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Image générée avec succès.]"))
            
            elif tool_name in ["display_raw_data", "display_processed_data"]:
                if not state.get("fetched_df_json"):
                     tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="Erreur: Aucune donnée disponible à afficher."))
                else:
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Préparation de l'affichage des données.]"))
            
        except Exception as e:
            error_msg = f"Erreur lors de l'exécution de l'outil '{tool_name}': {repr(e)}"
            tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=json.dumps({"error": error_msg})))
            current_state_updates["error"] = error_msg
            print(error_msg)
            
    current_state_updates["messages"] = tool_outputs
    return current_state_updates

# Noeud 3 : synthesize_final_answer_node, synthétise la réponse finale à partir de l'état.
# agent.py

# ... (imports, y compris pandas as pd et AIMessage)

# Noeud 3 : synthesize_final_answer_node, synthétise la réponse finale à partir de l'état.
def synthesize_final_answer_node(state: AgentState):
    """Crée la réponse finale à présenter à l'utilisateur en interprétant le résultat de la prédiction."""
    print("\n--- AGENT: Synthétisation de la réponse finale ---")
    ticker = state.get("ticker", "l'action")
    prediction_result = state.get("prediction", "inconnu")
    image_base64 = state.get("image_base64")

    latest_year_str = "récentes" # Valeur par défaut si l'extraction échoue
    if state.get("processed_df_json"):
        try:
            # On charge les données qui ont servi à la prédiction
            processed_df = pd.read_json(state["processed_df_json"], orient='split')
            if not processed_df.empty:
                # On récupère la dernière ligne de l'index (ex: 'AAPL_2024')
                last_index = processed_df.index[-1]
                # On extrait l'année
                latest_year_str = str(last_index).split('_')[-1]
                next_year_str = str(int(latest_year_str) + 1)

        except Exception as e:
            print(f"Avertissement : Impossible d'extraire l'année des données : {e}")

    # On choisit la bonne formulation en fonction du résultat de la prédiction
    if prediction_result == "Risque Élevé Détecté":
        response_content = (
            f"⚠️ **Attention !** En se basant sur les données de **{latest_year_str}** pour l'action **{ticker.upper()}**, mon analyse a détecté des signaux indiquant un **risque élevé de sous-performance pour l'année à venir ({next_year_str})**.\n\n"
            f"Mon modèle est particulièrement confiant dans cette évaluation. Je te conseille la plus grande prudence."
        )
    elif prediction_result == "Aucun Risque Extrême Détecté":
        response_content = (
            f"En se basant sur les données de **{latest_year_str}** pour l'action **{ticker.upper()}**, mon analyse n'a **pas détecté de signaux de danger extrême pour l'année à venir ({next_year_str})**.\n\n"
            f"**Important :** Cela ne signifie pas que c'est un bon investissement. Cela veut simplement dire que mon modèle, qui est spécialisé dans la détection de "
            f"signaux très négatifs, n'en a pas trouvé ici. Mon rôle est de t'aider à éviter une erreur évidente, pas de te garantir un succès."
        )
    else:
        # Cas par défaut si la prédiction a échoué
        response_content = f"L'analyse des données de **{latest_year_str}** pour **{ticker.upper()}** a été effectuée, mais le résultat de la prédiction n'a pas pu être interprété."

    response_content += f"\n\nVoici une visualisation des indicateurs financiers clés de **{latest_year_str}** qui ont servi à cette analyse :"
    
    final_message = AIMessage(content=response_content)
    if image_base64:
        setattr(final_message, 'image_base64', image_base64)
    
    return {"messages": [final_message]}

# Noeud 4 : cleanup_state_node, nettoie l'état pour éviter de stocker des données lourdes.
def cleanup_state_node(state: AgentState):
    """Nettoie l'état pour éviter de stocker des données lourdes."""
    print("\n--- SYSTEM: Nettoyage du state avant la sauvegarde ---")
    return {"fetched_df_json": "", "processed_df_json": "", "image_base64": "", "prediction": "", "error": ""}

# Noeud 5 : prepare_data_display_node, prépare les données pour l'affichage en DataFrame
def prepare_data_display_node(state: AgentState):
    """Prépare un AIMessage avec un DataFrame spécifique attaché."""
    print("\n--- AGENT: Préparation du DataFrame pour l'affichage ---")
    
    tool_name_called = next(msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls).tool_calls[-1]['name']

    if tool_name_called == "display_processed_data" and state.get("processed_df_json"):
        df_json = state["processed_df_json"]
        message_content = "Voici les données **pré-traitées** que tu as demandées :"
    elif tool_name_called == "display_raw_data" and state.get("fetched_df_json"):
        df_json = state["fetched_df_json"]
        message_content = "Voici les données **brutes** que tu as demandées :"
    else:
        final_message = AIMessage(content="Désolé, les données demandées ne sont pas disponibles.")
        return {"messages": [final_message]}

    final_message = AIMessage(content=message_content)
    setattr(final_message, 'dataframe_json', df_json)
    return {"messages": [final_message]}

# --- Router principal pour diriger le flux du graph ---
def router(state: AgentState) -> str:
    """Le routeur principal du graphe, maintenant plus robuste."""
    print("\n--- ROUTEUR: Évaluation de l'état pour choisir la prochaine étape ---")
    
    last_message = state['messages'][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("Routeur -> Décision: Exécuter un outil.")
        return "execute_tool"
        
    if isinstance(last_message, ToolMessage):
        if state.get("error"):
            print("Routeur -> Décision: Erreur détectée, fin du processus.")
            return END
        
        ai_message_for_tool = next(
            (msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls),
            None
        )
        if not ai_message_for_tool:
            return END

        tool_name = ai_message_for_tool.tool_calls[-1]['name']
        
        if tool_name == 'visualize_data':
            print(f"Routeur -> Décision: Outil final '{tool_name}' exécuté, passage à la synthèse.")
            return "synthesize_final_answer"
            
        elif tool_name in ['display_raw_data', 'display_processed_data']:
            print(f"Routeur -> Décision: Outil d'affichage '{tool_name}' exécuté, préparation de l'affichage.")
            return "prepare_data_display"
            
        else: # Pour fetch_data, preprocess_data, predict_performance
            print(f"Routeur -> Décision: Outil intermédiaire '{tool_name}' exécuté, retour à l'agent.")
            return "agent"
            
    print("Routeur -> Décision: Fin du processus.")
    return END

# --- CONSTRUCTION DU GRAPH ---
memory = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("synthesize_final_answer", synthesize_final_answer_node)
workflow.add_node("cleanup_state", cleanup_state_node)
workflow.add_node("prepare_data_display", prepare_data_display_node) 

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {"execute_tool": "execute_tool", "__end__": END}
)
workflow.add_conditional_edges(
    "execute_tool",
    router,
    {
        "agent": "agent", 
        "synthesize_final_answer": "synthesize_final_answer", 
        "prepare_data_display": "prepare_data_display", 
        "__end__": END
    }
)

workflow.add_edge("synthesize_final_answer", "cleanup_state")
workflow.add_edge("prepare_data_display", END) 
workflow.add_edge("cleanup_state", END)


app = workflow.compile(checkpointer=memory)

# --- Crée une visualisation du Graph ---
try:
    graph = app.get_graph()
    image_bytes = graph.draw_mermaid_png()
    with open("agent_workflow.png", "wb") as f:
        f.write(image_bytes)
    
    print("\nVisualisation du graph sauvegardée dans le répertoire en tant que agent_workflow.png \n")

except Exception as e:
    print(f"\nJe n'ai pas pu généré la visualisation. Lancez 'pip install playwright' et 'playwright install'. Erreur: {e}\n")

# --- Bloc test main ---
if __name__ == '__main__':
    def run_conversation(session_id: str, user_input: str):
        print(f"\n--- User: {user_input} ---")
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=user_input)]}
        final_message = None
        for event in app.stream(inputs, config=config, stream_mode="values"):
            final_message = event["messages"][-1]
        if final_message:
            print(f"\n--- Réponse finale de l'assistant ---\n{final_message.content}")
            if hasattr(final_message, 'image_base64'):
                print("\n[L'image a été générée et ajoutée au message final]")

    conversation_id = f"test_session_{uuid.uuid4()}"
    run_conversation(conversation_id, "S'il te plaît, fais une analyse complète de GOOGL")