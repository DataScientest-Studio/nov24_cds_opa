Projet OPA - Agent Analyste financier
==============================

Ce repo est un projet de fin d'étude réalisé dans le cadre du cursus Data Scientist chez DataScientest et l'Ecole des Mines Paris par Jerry PETILAIRE, Gilles LENY, Samuel LEE KWET SUN et Mathis GENTHON

Organisation du projet
------------

    ├── LICENSE
    ├── README.md          <- Le README décrivant le projet
    ├── environment.yml    <- Le fichier yml permettant de copier l'environnement conda nécessaire à l'exécution du projet.
    │
    ├── notebooks          <- Tous les notebooks et données utilsés pendant la phase de recherche
    │   ├── csv            <- Les données utilisées pendant le projet.
    │   ├── Gilles         <- Le travail de recherche effectué par Gilles LENY
    │   ├── Jerry          <- Le travail de recherche effectué par Jerry PETILAIRE
    │   ├── Samuel         <- Le travail de recherche effectué par Samuel LEE KWET SUN
    │   └── Mathis GENTHON <- Le travail de recherche effectué par Mathis GENTHON, convention 'OPA#X' pour l'ordre de recherche
    │
    ├── models             <- Les modèles finaux entraînés
    │
    ├── reports            <- Les rapports produits pour la présentation du projet
    │   └── figures        <- Les graphiques générés lors de la phase d'exploration des données
    │
    ├── agent              <- Le code source de l'agent analyste 
    │   ├── app.py         <- Le fichier streamlit utilisé pour lancer l'agent via 'streamlit run agent/app.py'
    │   ├── agent.py       <- La logique agentique dévéloppée dans le framework LangGraph
    │   ├── tools.py       <- La fichier référençant les outils disponibles pour l'agent 
    │   │
    │   ├── src            <- Le dossier contenant les scripts des outils
    │   └── assets         <- Le dossier contenant les assets nécessaires à l'UI de l'application Streamlit
    
--------

Mise en place de l'envrionnement 
------------
Installer Miniconda : 
https://www.anaconda.com/docs/getting-started/miniconda/install

Vérfier l'installation : 
    conda --version

Pour créer l'environnement de l'agent contenant les dépendances : 
    conda env create -f environment.yml -n agent

Lancement de l'agent
------------

Activer l'envrionnement pour la session en cours : 
    conda activate agent

Lancer l'application Streamlit : 
    streamlit run agent/app.y

![Workflow de l'agent](agent_worlflow.png)
