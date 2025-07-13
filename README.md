Projet OPA - Agent Analyste financier
==============================

Ce repo est un projet de fin d'étude réalisé dans le cadre du cursus Data Scientist chez DataScientest et l'Ecole des Mines Paris par Jerry PETILAIRE, Gilles LENY, Samuel LEE KWET SUN et Mathis GENTHON

Organisation du projet
------------

    ├── LICENSE
    ├── README.md                    <- Le README décrivant le projet
    ├── environment.yml              <- Le fichier yml permettant de copier l'environnement conda nécessaire à l'exécution du projet.
    │
    ├── notebooks                    <- Tous les notebooks et données utilsés pendant la phase de recherche
    │   ├── csv                      <- Les données utilisées pendant le projet.
    │   ├── Gilles                   <- Le travail de recherche effectué par Gilles LENY
    │   ├── Jerry                    <- Le travail de recherche effectué par Jerry PETILAIRE
    │   ├── Samuel                   <- Le travail de recherche effectué par Samuel LEE KWET SUN
    │   └── Mathis                   <- Le travail de recherche effectué par Mathis GENTHON, convention 'OPA#X' pour l'ordre de recherche
    │
    ├── models                       <- Les modèles finaux entraînés
    │
    ├── reports                      <- Les rapports produits pour la présentation du projet
    │   └── figures                  <- Les graphiques générés lors de la phase d'exploration des données
    │
    ├── agent                        <- Le code source de l'agent analyste 
    │   ├── app.py                   <- Le fichier streamlit utilisé pour lancer l'agent 
    │   ├── agent.py                 <- La logique agentique dévéloppée dans le framework LangGraph
    │   ├── tools.py                 <- La fichier référençant les outils disponibles pour l'agent 
    │   ├── agent_workflow.png       <- Visualisation des noeuds et relations du LangGraph de l'agent
    │   │
    │   ├── src                      <- Le dossier contenant les scripts des outils
    │   ├── pages                    <- Le dossier contenant les différentes pages de l'application 
    │   ├── assets                   <- Le dossier contenant les assets nécessaires à l'UI de l'application 
    │   └── .streamlit               <- Le dossier contenant les fichiers de configuration pour Streamlit

    
--------

Liste des outils et capacités de l'agent
------------
1. `search_ticker`: Recherche le ticker boursier d'une entreprise à partir de son nom.
2. `fetch_data`: Récupère les données financières fondamentales pour un ticker boursier donné.
3. `preprocess_data`: Prépare les données financières récupérées pour la prédiction.
4. `analyze_risks`: Vérifie des signaux négatifs extrêmes se trouvent dans les données prétraitées.
5. `display_price_chart`: Affiche un graphique de l'évolution du prix (cours) d'une action. 
6. `display_raw_data`: Affiche le tableau de données financières brutes qui ont été initialement récupérées.
7. `display_processed_data`: Affiche le tableau de données financières traitées et nettoyées, prêtes pour l'analyse.
8. `create_dynamic_chart`: Crée un graphique interactif basé sur les données financières prétraitées.
9. `get_stock_news`: Récupère les dernières actualités pour un ticker donné.
10. `get_company_profile`: Récupère le profil d'une entreprise, incluant des informations clés comme le nom, le secteur, l'industrie, le CEO, etc.
11. `compare_stocks`: Compare plusieurs entreprises sur une métrique financière ou sur leur prix.
  
Graph de l'agent
------------
![Graph de l'agent](agent_workflow.png)
  

Mise en place de l'environnement 
------------
  
```bash
python3 -m venv stella
source stella/bin/activate
pip install -r requirements.txt
playwright install```
  

Obtention des clefs API
------------
  
Il est nécessaire d'obtenir plusieurs clef API pour faire fonctionner l'agent :
  
**Obtenir une clef API OpenRouter**
  
OpenRouter est le fournisseur de LLM utilisé :  
https://openrouter.ai/settings/keys
  
**Obtenir une clef API Financial Modeling Prep**
  
Financial Modeling Prep est le fournisseur de données financières :  
https://site.financialmodelingprep.com/developer/docs/dashboard

**Obtenir une clef API sur NewsAPI.org**
  
NewsAPI fournit les news liés aux entreprises :  
https://newsapi.org
  
**Obtenir une clef API LangSmith**
LangSmith permet de tracer l'agent et visualiser ce qu'il se passe dans l'application : 
https://smith.langchain.com/
  
Ajout des variables d'envrionnement 
------------
### Sur Windows
**Lancer ces lignes de commande en remplaçant ma_clef_api_x à chaque fois**
  
```setx LANGCHAIN_ENDPOINT "https://api.smith.langchain.com"```
  
```setx LANGCHAIN_PROJECT "stella"```
  
```setx LANGCHAIN_TRACING_V2 "true"```
  
```setx LANGSMITH_API_KEY "ma_clef_api_langsmith"```
  
```setx NEWS_API_KEY "ma_clef_newsapi"```
  
```setx OPENROUTER_API_KEY "ma_clef_api_openrouter"```
  
```setx FMP_API_KEY "ma_clef_api_fmp"```
  

### Sur Linux / anciens macOS

```nano ~/.bashrc```
  
**Ajouter ces lignes dans le fichier en remplaçant ma_clef_api à chaque fois**
   
```export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"```
  
```export LANGCHAIN_PROJECT="stella"```
  
```export LANGCHAIN_TRACING_V2="true"```
  
```export LANGSMITH_API_KEY="ma_clef_api_langsmith"```
  
```export OPENROUTER_API_KEY="ma_clef_api_openrouter"```
  
```export FMP_API_KEY="ma_clef_api_fmp"```
  
```export NEWS_API_KEY="ma_clef_newsapi"```

### Sur macOS (récent)
  
```nano ~/.zshrc```
  
**Ajouter ces lignes dans le fichier en remplaçant ma_clef_api à chaque fois**
  
```export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"```
  
```export LANGCHAIN_PROJECT="stella"```
  
```export LANGCHAIN_TRACING_V2="true"```
  
```export LANGSMITH_API_KEY="ma_clef_api_langsmith"```
  
```export OPENROUTER_API_KEY="ma_clef_api_openrouter"```
  
```export FMP_API_KEY="ma_clef_api_fmp"```
  
```export NEWS_API_KEY="ma_clef_newsapi"```

Lancement de l'agent
------------
  
**Activer l'envrionnement pour la session en cours**
  
```conda activate stella```

**Lancer l'application Streamlit**
  
```streamlit run agent/app.y```


