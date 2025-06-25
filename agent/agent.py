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

Quand un utilisateur te demande d'analyser un stock, tu DOIS suivre une séquence d'actions prédéfinie pour fournir une analyse complète :
1. Prmièrement, tu dois appeler le tool `fetch_data` avec le ticker correspondant à la demande de l'utilisateur.
2. Deuxièmement, tu dois appeler le tool `preprocess_data` pour préparer les données récupérées.
3. Troisièmement, tu dois appeler le tool `predict_performance` pour prédire la performance du stock.
4. Enfin, tu dois appeler le tool `visualize_data` pour compléter l'analyse.

Ta tâche sera complète uniquement après que le tool `visualize_data` ait été appelé. Si tu rencontres une erreur, informe l'utilisateur.
Tu dois toujours répondre en français. Tu ne dois pas faire de suppositions sur les données financières, tu dois te baser uniquement sur les données récupérées par le tool `fetch_data`.
Lorsque l'utilisateur te demande qui tu es, ou comment tu fonctionnes, tu te présenteras et lui expliqueras ton fonctionnement.
Tes réponses doivent toujours être polies, claires et concises.
"""

# --- Définition des noeuds du Graph ---

# Noeud 1 : agent_node, point d'entrée
def agent_node(state: AgentState):
    """Le 'cerveau' de l'agent. Décide le prochain outil à appeler."""
    print("\n--- AGENT: Décision de la prochaine étape... ---")
    messages = state['messages']
    if not messages or messages[0].type != "system":
        final_messages = [("system", system_prompt)] + list(messages)
    else:
        final_messages = messages
    response = llm.bind_tools(available_tools).invoke(final_messages)
    return {"messages": [response]}

# Noeud 2 : execute_tool_node, exécute les outils en se basant sur la décision de l'agent_node (Noeud 1).
def execute_tool_node(state: AgentState):
    """ Le 'pont' qui exécute la logique réelle et met à jour l'état."""
    print("\n--- TOOLS: Exécution d'un tool ---")
    action_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls), None)
    if not action_message:
        raise ValueError("Pas d'appel de tool trouvé dans le dernier AImessage.")

    tool_outputs = []
    current_state_updates = {}
    for tool_call in action_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        print(f"Le LLM a décidé d'appeler le tool : {tool_name} - avec les arguments : {tool_args}")
        try:
            if tool_name == "fetch_data":
                df_output = _fetch_data_logic(ticker=tool_args.get("ticker"))
                # Transforme le DataFrame en string JSON avant de le stocker dans state
                current_state_updates["fetched_df_json"] = df_output.to_json(orient='split')
                current_state_updates["ticker"] = tool_args.get("ticker")
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Data récupérée avec succès.]"))

            elif tool_name == "preprocess_data":
                # Repasse le JSON en DataFrame
                fetched_df = pd.read_json(state["fetched_df_json"], orient='split')
                processed_df_output = _preprocess_data_logic(df=fetched_df)
                # Retransforme le DataFrame en string JSON avant de le stocker dans state
                current_state_updates["processed_df_json"] = processed_df_output.to_json(orient='split')
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Preprocessing de la data effectué avec succès.]"))

            elif tool_name == "predict_performance":
                # Repasse le JSON en DataFrame
                processed_df = pd.read_json(state["processed_df_json"], orient='split')
                prediction_output = _predict_performance_logic(processed_data=processed_df)
                current_state_updates["prediction"] = prediction_output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=prediction_output))

            elif tool_name == "visualize_data":
                # Transfomr le DataFrame ORIGINEL pour visualisation
                fetched_df = pd.read_json(state["fetched_df_json"], orient='split')
                image_output = _visualize_data_logic(
                    data=fetched_df, 
                    prediction=state["prediction"], 
                    ticker=state["ticker"]
                )
                current_state_updates["image_base64"] = image_output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[L'image a été générée avec succès.]"))

        except Exception as e:
            error_msg = f"Erreur lors de l'exécution du tool: '{tool_name}': {repr(e)}"
            tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=json.dumps({"erreur": error_msg})))
            current_state_updates["error"] = error_msg
            print(error_msg)
    
    current_state_updates["messages"] = tool_outputs
    return current_state_updates

# Noeud 3 : synthesize_final_answer_node, synthétise la réponse finale à partir de l'état.
def synthesize_final_answer_node(state: AgentState):
    """Crée la réponse finale à présenter à l'utilisateur avec texte et image."""
    print("\n--- AGENT: Synthétisation de la réponse finale ---")
    ticker = state.get("ticker", "the stock")
    prediction = state.get("prediction", "unavailable")
    image_base64 = state.get("image_base64")

    response_content = (
        f"En se basant sur mon analyse, le modèle prédit que  **{ticker.upper()}** "
        f"va **{prediction.upper()}** le marché.\n\n"
        "Voici une visualisation des métriques financières clés:"
    )
    final_message = AIMessage(content=response_content)
    if image_base64:
        setattr(final_message, 'image_base64', image_base64)
    
    # On retourne seulement le message. Le nettoyage se fait dans le noeud suivant.
    return {"messages": [final_message]}

# Noeud 4 : cleanup_state_node, nettoie l'état pour éviter de stocker des données lourdes.
def cleanup_state_node(state: AgentState):
    """Nettoie l'état pour éviter de stocker des données lourdes."""
    print("\n--- SYSTEM: Nettoyage du state avant la sauvegarde ---")
    # Set the JSON strings to empty or None to clear memory
    return {"fetched_df_json": "", "processed_df_json": ""}

# --- Router principal pour diriger le flux du graph ---
def router(state: AgentState) -> str:
    """Le router du graph."""
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
    print("\n--- AGENT: Affichage du message final à l'utilisateur ---")
    return END

# --- Workflow de l'agent (structure des "edges" et "nodes" du Graph) ---
memory = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("synthesize_final_answer", synthesize_final_answer_node)
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

# Après la synthèse de la réponse finale, on nettoie puis on appelle END.
workflow.add_edge("synthesize_final_answer", "cleanup_state")
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