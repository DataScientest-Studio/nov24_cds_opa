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
    _fetch_recent_news_logic,
    _search_ticker_logic,
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
    company_name: str
    fetched_df_json: str
    processed_df_json: str
    prediction: str
    plotly_json: str  
    messages: Annotated[List[AnyMessage], add_messages]
    error: str

# --- Prompt système (définition du rôle de l'agent) ---
# agent.py

system_prompt = """Ton nom est Stella. Tu es une assistante experte financière. Ton but principal est d'aider les utilisateurs en analysant des actions.

**Liste des outils disponibles**
1. `search_ticker`: Recherche le ticker boursier d'une entreprise à partir de son nom.
2. `fetch_data`: Récupère les données financières fondamentales pour un ticker boursier donné.
3. `preprocess_data`: Prépare les données financières récupérées pour la prédiction.
4. `predict_performance`: Prédit la performance d'une action en se basant sur les données prétraitées.
5. `display_raw_data`: Affiche le tableau de données financières brutes qui ont été initialement récupérées.
6. `display_processed_data`: Affiche le tableau de données financières traitées et nettoyées, prêtes pour l'analyse.
7. `create_dynamic_chart`: Crée un graphique interactif basé sur les données financières prétraitées.
8. `get_stock_news`: Récupère les dernières actualités pour un ticker donné.

Si l'utilisateur te demande comment tu fonctionnes, à quoi tu sers, ou toute autre demande similaire tu n'utiliseras pas d'outils. 
Tu expliqueras simplement ton rôle et tes fonctionnalités en donnant des exemples de demandes qu'on peut te faire.

**Séquence d'analyse complète**
Quand un utilisateur te demande une analyse complète, tu DOIS suivre cette séquence d'outils :
1. `search_ticker` si le nom de l'entreprise est donné plutôt que le ticker.
2. `fetch_data` avec le ticker demandé.
2. `preprocess_data` pour nettoyer les données.
3. `predict_performance` pour obtenir un verdict.
Ta tâche est considérée comme terminée après l'appel à `predict_performance`. La réponse finale avec le graphique sera générée automatiquement.

**IDENTIFICATION DU TICKER** 
Si l'utilisateur donne un nom de société (comme 'Apple' ou 'Microsoft') au lieu d'un ticker (comme 'AAPL' ou 'MSFT'), 
ta toute première action DOIT être d'utiliser l'outil `search_ticker` pour trouver le ticker correct.

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

**Actualités :**
Si l'utilisateur demande "les nouvelles", "les actualités" ou "ce qui se passe" pour une entreprise, utilise l'outil `get_stock_news`. 
Tu peux aussi proposer de le faire après une analyse complète.

Tu dois toujours répondre en français et tutoyer ton interlocuteur.
"""
# --- Définition des noeuds du Graph ---

# Noeud 1 : agent_node, point d'entrée
# agent.py

def agent_node(state: AgentState):
    """Le 'cerveau' de l'agent. Décide le prochain outil à appeler."""
    print("\n--- AGENT: Décision de la prochaine étape... ---")
    
    response = llm.bind_tools(available_tools).invoke(state['messages'])
    
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
            if tool_name == "search_ticker":
                company_name = tool_args.get("company_name")
                ticker = _search_ticker_logic(company_name=company_name)
                # On stocke le ticker ET le nom de l'entreprise
                current_state_updates["ticker"] = ticker
                current_state_updates["company_name"] = company_name 
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=f"[Ticker '{ticker}' trouvé.]"))

            elif tool_name == "fetch_data":
                try:
                    output_df = _fetch_data_logic(ticker=tool_args.get("ticker"))
                    current_state_updates["fetched_df_json"] = output_df.to_json(orient='split')
                    current_state_updates["ticker"] = tool_args.get("ticker")
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données récupérées avec succès.]"))
                except APILimitError as e:
                    user_friendly_error = "Désolé, il semble que j'aie un problème d'accès à mon fournisseur de données. Peux-tu réessayer plus tard ?"
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=json.dumps({"error": user_friendly_error})))
                    current_state_updates["error"] = user_friendly_error
            
            elif tool_name == "get_stock_news":
                ticker = state.get("ticker")
                company_name = state.get("company_name") or ticker
                
                if not ticker:
                    raise ValueError("Aucun ticker trouvé dans l'état pour chercher des nouvelles.")
                
                news_summary = _fetch_recent_news_logic(
                    ticker=ticker, 
                    company_name=company_name
                )

                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=news_summary))
                
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
    Ce noeud est le point de sortie pour une analyse de prédiction.
    """
    print("\n--- AGENT: Génération de la réponse finale et du graphique ---")
    
    # --- 1. Récupération des informations de l'état ---
    ticker = state.get("ticker", "l'action")
    prediction_result = state.get("prediction", "inconnu")
    processed_df_json = state.get("processed_df_json")

    # --- 2. Construction de la réponse textuelle ---
    response_content = ""
    latest_year_str = "récentes"
    next_year_str = "prochaine"
    
    if processed_df_json:
        try:
            df = pd.read_json(StringIO(processed_df_json), orient='split')
            if not df.empty and 'calendarYear' in df.columns:
                latest_year_str = df['calendarYear'].iloc[-1]
                next_year_str = str(int(latest_year_str) + 1)
        except Exception as e:
            print(f"Avertissement : Impossible d'extraire l'année des données : {e}")

    # Logique de la réponse textuelle basée sur la prédiction
    if prediction_result == "Risque Élevé Détecté":
        response_content = (
            f"⚠️ **Attention !** Pour l'action **{ticker.upper()}**, en se basant sur les données de **{latest_year_str}**, mon analyse a détecté des signaux indiquant un **risque élevé de sous-performance pour l'année à venir ({next_year_str})**.\n\n"
            "Mon modèle est particulièrement confiant dans cette évaluation. Je te conseille la plus grande prudence."
        )
    elif prediction_result == "Aucun Risque Extrême Détecté":
        response_content = (
            f"Pour l'action **{ticker.upper()}**, en se basant sur les données de **{latest_year_str}**, mon analyse n'a **pas détecté de signaux de danger extrême pour l'année à venir ({next_year_str})**.\n\n"
            "**Important :** Cela ne signifie pas que c'est un bon investissement. Cela veut simplement dire que mon modèle, spécialisé dans la détection de signaux très négatifs, n'en a pas trouvé ici. Mon rôle est de t'aider à éviter une erreur évidente, pas de te garantir un succès."
        )
    else:
        response_content = f"L'analyse des données pour **{ticker.upper()}** a été effectuée, mais le résultat de la prédiction n'a pas pu être interprété."

    # --- 3. Création du graphique de synthèse ---
    chart_json = None
    if processed_df_json:
        try:
            df = pd.read_json(StringIO(processed_df_json), orient='split')
            metrics_to_plot = ['roe', 'debtToEquity', 'earningsYield', 'marginProfit']
            
            # On s'assure que les colonnes existent avant de les utiliser
            plot_cols = [col for col in metrics_to_plot if col in df.columns]
            
            if not df.empty and plot_cols:
                chart_title = f"Indicateurs Clés pour {ticker.upper()} ({latest_year_str})"
                
                # Préparation des données pour le bar chart
                df_for_plot = df[plot_cols].iloc[-1].reset_index()
                df_for_plot.columns = ['Indicateur', 'Valeur']
                
                fig = px.bar(df_for_plot, x='Indicateur', y='Valeur', title=chart_title, color='Indicateur',
                             text_auto='.2f') # Affiche les valeurs sur les barres
                fig.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"), showlegend=False)
                chart_json = pio.to_json(fig)
                response_content += f"\n\nVoici une visualisation des indicateurs qui ont servi à cette analyse :"
            else:
                response_content += "\n\n(Impossible de générer le graphique : données ou colonnes manquantes)."

        except Exception as e:
            print(f"Erreur lors de la création du graphique par défaut : {e}")
            response_content += "\n\n(Je n'ai pas pu générer le graphique associé en raison d'une erreur.)"
    
    # --- 4. Création du message final ---
    final_message = AIMessage(content=response_content)
    if chart_json:
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

def prepare_news_display_node(state: AgentState):
    """Prépare un AIMessage avec les actualités formatées pour l'affichage."""
    print("\n--- AGENT: Préparation de l'affichage des actualités ---")
    
    # 1. Retrouver le ToolMessage qui contient le résultat des actualités
    # On cherche le dernier message de type ToolMessage dans l'historique
    tool_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, ToolMessage)), None)
    
    if not tool_message or not tool_message.content:
        final_message = AIMessage(content="Désolé, je n'ai pas pu récupérer les actualités.")
        return {"messages": [final_message]}

    # 2. Préparer le contenu textuel de la réponse
    ticker = state.get("ticker", "l'entreprise")
    company_name = state.get("company_name", ticker)
    
    response_content = f"Voici les dernières actualités que j'ai trouvées pour **{company_name.title()} ({ticker.upper()})** :"
    
    final_message = AIMessage(content=response_content)
    
    # 3. Attacher le JSON des actualités au message final
    # Le front-end (Streamlit) utilisera cet attribut pour afficher les articles
    setattr(final_message, 'news_json', tool_message.content)
    
    return {"messages": [final_message]}

# --- Router principal pour diriger le flux du graph ---

# agent.py

def router(state: AgentState) -> str:
    """Le routeur principal du graphe, version finale robuste."""
    print("\n--- ROUTEUR: Évaluation de l'état pour choisir la prochaine étape ---")

    # On récupère les messages de l'état
    messages = state['messages']
    
    # Y a-t-il une erreur ? C'est la priorité absolue.
    if state.get("error"):
        print("Routeur -> Décision: Erreur détectée, fin du processus.")
        return END

    # Le dernier message est-il une décision de l'IA d'appeler un outil ?
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        print("Routeur -> Décision: L'IA a fourni une réponse textuelle. Fin du cycle.")
        return END
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # C'est la première fois qu'on voit cette décision, on doit exécuter l'outil.
        print("Routeur -> Décision: Appel d'outil demandé, passage à execute_tool.")
        return "execute_tool"

    # Si le dernier message n'est PAS un appel à un outil, cela signifie probablement
    # qu'un outil vient de s'exécuter. Nous devons décider où aller ensuite.
    
    # On retrouve le dernier appel à un outil fait par l'IA
    ai_message_with_tool_call = next(
        (msg for msg in reversed(messages) if isinstance(msg, AIMessage) and msg.tool_calls),
        None
    )
    # S'il n'y en a pas, on ne peut rien faire de plus.
    if not ai_message_with_tool_call:
        print("Routeur -> Décision: Aucune action claire à prendre (pas d'appel d'outil trouvé), fin du processus.")
        return END
        
    tool_name = ai_message_with_tool_call.tool_calls[-1]['name']
    print(f"--- DEBUG ROUTEUR: Le dernier outil appelé était '{tool_name}'. ---")

    # Maintenant, on décide de la suite en fonction de cet outil.
    if tool_name == 'predict_performance':
        return "generate_final_response"
    elif tool_name in ['display_raw_data', 'display_processed_data']:
        return "prepare_data_display"
    elif tool_name == 'create_dynamic_chart':
        return "prepare_chart_display"
    elif tool_name == 'get_stock_news':
        return "prepare_news_display"
    else: # Pour search_ticker, fetch_data, preprocess_data
        return "agent"
    
# --- CONSTRUCTION DU GRAPH ---
memory = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("generate_final_response", generate_final_response_node)
workflow.add_node("cleanup_state", cleanup_state_node)
workflow.add_node("prepare_data_display", prepare_data_display_node) 
workflow.add_node("prepare_chart_display", prepare_chart_display_node)
workflow.add_node("prepare_news_display", prepare_news_display_node)

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
        "prepare_news_display": "prepare_news_display", 
        "__end__": END
    }
)

workflow.add_edge("generate_final_response", "cleanup_state")
workflow.add_edge("prepare_data_display", END) 
workflow.add_edge("prepare_chart_display", END)
workflow.add_edge("prepare_news_display", END)
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