# agent.py
import os
import json
from typing import TypedDict, List, Annotated, Any
import pandas as pd
import plotly.express as px
import plotly.io as pio
import uuid
from io import StringIO
from src.fetch_data import APILimitError 

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
    _create_dynamic_chart_logic
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
    plotly_json: str  
    messages: Annotated[List[AnyMessage], add_messages]
    error: str

# --- Prompt système (définition du rôle de l'agent) ---
# agent.py

system_prompt = """Ton nom est Stella. Tu es une assistante experte financière. Ton but principal est d'aider les utilisateurs en analysant des actions.

**Séquence d'analyse complète :**
Quand un utilisateur te demande une analyse complète, tu DOIS suivre cette séquence d'outils :
1. `fetch_data` avec le ticker demandé.
2. `preprocess_data` pour nettoyer les données.
3. `predict_performance` pour obtenir un verdict.
Ta tâche est considérée comme terminée après l'appel à `predict_performance`. La réponse finale avec le graphique sera générée automatiquement.

**Analyse et Visualisation Dynamique :**
Quand un utilisateur te demande de "montrer", "visualiser", ou "comparer" des données spécifiques (par exemple, "montre-moi l'évolution du ROE"), tu DOIS utiliser l'outil `create_dynamic_chart`.

Pour cela, tu dois impérativement connaître la structure des données que tu manipules. Après l'étape `preprocess_data`, les données contiennent les colonnes suivantes :
`calendarYear`, `marketCap`, `marginProfit`, `roe`, `roic`, `revenuePerShare`, `debtToEquity`, `revenuePerShare_YoY_Growth`, `earningsYield`.

**Instructions pour `create_dynamic_chart` :**
1.  **Pour l'axe du temps, tu dois IMPÉRATIVEMENT utiliser la colonne `calendarYear` pour l'argument `x_column`. Ne suppose jamais l'existence d'une colonne 'year' ou 'date'.**
2.  Pour l'argument `y_column`, utilise le nom exact de la métrique demandée par l'utilisateur (par exemple, `roe`, `marginProfit`).
3.  Choisis le `chart_type` le plus pertinent : `line` pour une évolution dans le temps, `bar` pour une comparaison.
4.  Si les données ne sont pas encore disponibles, appelle d'abord `fetch_data`.

**Logique de Prédiction :**
- Si `predict_performance` renvoie "Risque Élevé Détecté", présente cela comme un avertissement clair.
- Si `predict_performance` renvoie "Aucun Risque Extrême Détecté", explique que cela n'est PAS une recommandation d'achat, mais simplement l'absence de signaux de danger majeurs.

Tu dois toujours répondre en français et tutoyer ton interlocuteur.
"""
# --- Définition des noeuds du Graph ---

# Noeud 1 : agent_node, point d'entrée
def agent_node(state: AgentState):
    """Le 'cerveau' de l'agent. Décide le prochain outil à appeler."""
    print("\n--- AGENT: Décision de la prochaine étape... ---")
    
    # Prépare le prompt système de base
    messages = [SystemMessage(content=system_prompt)]
    
    # --- INJECTION DE CONTEXTE DYNAMIQUE ---
    # Vérifie si des données prétraitées existent dans l'état
    if state.get("processed_df_json"):
        try:
            # Charge les colonnes disponibles à partir des données
            df = pd.read_json(StringIO(state["processed_df_json"]), orient='split')
            available_columns = df.columns.tolist()
            
            # Crée un message de contexte à injecter
            context_message = HumanMessage(
                content=f"""
                CONTEXTE IMPORTANT POUR TA PROCHAINE ACTION :
                Des données sont disponibles pour l'action '{state.get('ticker', '')}'.
                Les noms de colonnes EXACTS que tu peux utiliser pour les graphiques sont : {available_columns}.
                Quand tu utilises l'outil `create_dynamic_chart`, tu DOIS choisir un nom de colonne dans cette liste pour l'argument `y_column`.
                """
            )
            # Ajoute ce contexte juste avant les messages de l'utilisateur
            messages.append(context_message)
        except Exception as e:
            print(f"Avertissement : Impossible d'injecter le contexte des colonnes. Erreur : {e}")

    # Ajoute le reste de l'historique des messages
    messages.extend(state['messages'])

    # Invoque le LLM avec le contexte enrichi
    response = llm.bind_tools(available_tools).invoke(messages)
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
    
    # On gère le cas où plusieurs outils sont appelés, bien que ce soit rare ici.
    for tool_call in action_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        print(f"Le LLM a décidé d'appeler le tool : {tool_name} - avec les arguments : {tool_args}")

        try:

            if tool_name == "fetch_data":
                try:
                    output_df = _fetch_data_logic(ticker=tool_args.get("ticker"))
                    current_state_updates["fetched_df_json"] = output_df.to_json(orient='split')
                    current_state_updates["ticker"] = tool_args.get("ticker")
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données récupérées avec succès.]"))
                except APILimitError as e:
                    user_friendly_error = "Désolé, il semble que j'aie un problème d'accès à mon fournisseur de données. Peux-tu réessayer plus tard ?"
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=json.dumps({"error": user_friendly_error})))
                    current_state_updates["error"] = user_friendly_error

            elif tool_name == "preprocess_data":
                if not state.get("fetched_df_json"):
                    raise ValueError("Impossible de prétraiter les données car elles n'ont pas encore été récupérées.")
                fetched_df = pd.read_json(StringIO(state["fetched_df_json"]), orient='split')
                output = _preprocess_data_logic(df=fetched_df)
                current_state_updates["processed_df_json"] = output.to_json(orient='split')
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données prétraitées avec succès.]"))

            elif tool_name == "predict_performance":
                if not state.get("processed_df_json"):
                    raise ValueError("Impossible de faire une prédiction car les données n'ont pas encore été prétraitées.")
                processed_df = pd.read_json(StringIO(state["processed_df_json"]), orient='split')
                output = _predict_performance_logic(processed_data=processed_df)
                current_state_updates["prediction"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=output))

            elif tool_name == "visualize_data":
                if not state.get("fetched_df_json") or not state.get("prediction"):
                     raise ValueError("Données ou prédiction manquantes pour la visualisation.")
                fetched_df = pd.read_json(StringIO(state["fetched_df_json"]), orient='split')
                output = _visualize_data_logic(data=fetched_df, prediction=state["prediction"], ticker=state["ticker"])
                current_state_updates["image_base64"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Image générée avec succès.]"))
            
            elif tool_name == "create_dynamic_chart":
                data_json_for_chart = state.get("processed_df_json") or state.get("fetched_df_json")
                if not data_json_for_chart:
                    raise ValueError("Aucune donnée disponible pour créer un graphique. Appelle 'fetch_data' d'abord.")
                
                tool_args['data_json'] = data_json_for_chart
                chart_json = _create_dynamic_chart_logic(**tool_args)

                # On ajoute une vérification pour être sûr que la sortie est une chaîne de caractères
                if not isinstance(chart_json, str):
                    raise TypeError(f"L'outil a retourné un type inattendu : {type(chart_json)}")
                
                if "Erreur" in chart_json:
                    raise ValueError(chart_json) # Transforme l'erreur de l'outil en exception
                
                current_state_updates["plotly_json"] = chart_json
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Graphique interactif créé avec succès.]"))

            elif tool_name in ["display_raw_data", "display_processed_data"]:
                if not state.get("fetched_df_json"):
                     raise ValueError("Aucune donnée disponible à afficher.")
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Préparation de l'affichage des données.]"))
            
        except Exception as e:
            # Bloc de capture générique pour toutes les autres erreurs
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
def generate_final_response_node(state: AgentState):
    """
    Génère la réponse textuelle finale ET le graphique Plotly par défaut après une analyse complète.
    """
    print("\n--- AGENT: Génération de la réponse finale et du graphique ---")
    ticker = state.get("ticker", "l'action")
    prediction_result = state.get("prediction", "inconnu")

    # Logique pour extraire l'année
    latest_year_str = "récentes"
    next_year_str = "prochaine"
    processed_df = None # Initialisation
    if state.get("processed_df_json"):
        try:
            processed_df = pd.read_json(StringIO(state["processed_df_json"]), orient='split')
            if not processed_df.empty:
                last_index = processed_df.index[-1]
                latest_year_str = str(last_index).split('_')[-1]
                next_year_str = str(int(latest_year_str) + 1)
        except Exception as e:
            print(f"Avertissement : Impossible d'extraire l'année des données : {e}")

    # Logique de la réponse textuelle (identique à l'ancienne)
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
        # Cas par défaut si la prédiction a échoué ou est inattendue
        response_content = f"L'analyse des données de **{latest_year_str}** pour **{ticker.upper()}** a été effectuée, mais le résultat de la prédiction ('{prediction_result}') n'a pas pu être interprété."

    chart_json = None
    if processed_df is not None and not processed_df.empty:
        try:
            metrics_to_plot = ['roe', 'debtToEquity', 'earningsYield', 'marginProfit']
            chart_title = f"Indicateurs Clés pour {ticker.upper()} ({latest_year_str})"
            
            df_for_plot = processed_df[metrics_to_plot].iloc[-1].reset_index()
            df_for_plot.columns = ['indicateur', 'valeur']
            
            # APPEL DIRECT À PLOTLY EXPRESS (la bonne méthode)
            fig = px.bar(df_for_plot, x='indicateur', y='valeur', title=chart_title, color='indicateur')
            fig.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))
            chart_json = pio.to_json(fig)

            response_content += f"\n\nVoici une visualisation des indicateurs financiers clés de **{latest_year_str}** qui ont servi à cette analyse :"
        except Exception as e:
            print(f"Erreur lors de la création du graphique par défaut : {e}")
            response_content += "\n\n(Je n'ai pas pu générer le graphique associé en raison d'une erreur.)"
    
    final_message = AIMessage(content=response_content)
    if chart_json and "Error" not in chart_json:
        setattr(final_message, 'plotly_json', chart_json)
    
    return {"messages": [final_message]}

# Noeud 4 : cleanup_state_node, nettoie l'état pour éviter de stocker des données lourdes.
def cleanup_state_node(state: AgentState):
    """
    Nettoie l'état pour la prochaine interaction, en ne supprimant que les données
    spécifiques à la dernière réponse (le graphique) mais en gardant le contexte (les données).
    """
    print("\n--- SYSTEM: Nettoyage partiel du state avant la sauvegarde ---")
    return {"plotly_json": ""}

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

def prepare_chart_display_node(state: AgentState):
    """Prépare un AIMessage avec le graphique Plotly demandé par l'utilisateur."""
    print("\n--- AGENT: Préparation du graphique pour l'affichage ---")
    
    # Laisse le LLM générer une courte phrase d'introduction
    response = ("Voici le graphgique demandé : ")
    
    final_message = AIMessage(content=response)
    setattr(final_message, 'plotly_json', state["plotly_json"])
    
    return {"messages": [final_message]}

# --- Router principal pour diriger le flux du graph ---
# agent.py

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
        
        # Retrouve quel outil a été appelé
        ai_message_for_tool = next((msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls), None)
        if not ai_message_for_tool: return END
        tool_name = ai_message_for_tool.tool_calls[-1]['name']
        
        # --- LOGIQUE DE ROUTAGE CORRIGÉE ET EXPLICITE ---
        if tool_name == 'predict_performance':
            print(f"Routeur -> Décision: Outil final '{tool_name}' exécuté, passage à la réponse finale.")
            return "generate_final_response"
        
        elif tool_name in ['display_raw_data', 'display_processed_data']:
            print(f"Routeur -> Décision: Outil d'affichage '{tool_name}' exécuté, préparation de l'affichage des données.")
            return "prepare_data_display"

        # On a besoin d'un nouveau noeud pour présenter le graphique demandé spécifiquement
        elif tool_name == 'create_dynamic_chart':
             print(f"Routeur -> Décision: Outil '{tool_name}' exécuté, préparation de l'affichage du graphique.")
             return "prepare_chart_display" # Note: Assurez-vous d'avoir ce noeud et cette route

        else: # Pour fetch_data, preprocess_data
            print(f"Routeur -> Décision: Outil intermédiaire '{tool_name}' exécuté, retour à l'agent.")
            return "agent"
            
    print("Routeur -> Décision: Fin du processus.")
    return END

# --- CONSTRUCTION DU GRAPH ---
memory = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("generate_final_response", generate_final_response_node)
workflow.add_node("cleanup_state", cleanup_state_node)
workflow.add_node("prepare_data_display", prepare_data_display_node) 
workflow.add_node("prepare_chart_display", prepare_chart_display_node) 

workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", router, {"execute_tool": "execute_tool", "__end__": END})
workflow.add_conditional_edges(
    "execute_tool",
    router,
    {
        "agent": "agent", 
        "generate_final_response": "generate_final_response",
        "prepare_data_display": "prepare_data_display", 
        "prepare_chart_display": "prepare_chart_display",
        "__end__": END
    }
)

workflow.add_edge("generate_final_response", "cleanup_state")
workflow.add_edge("prepare_data_display", END) 
workflow.add_edge("prepare_chart_display", END)
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