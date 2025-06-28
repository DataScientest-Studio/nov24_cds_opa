# agent/pages/1_📊_Visualize_Run.py

import streamlit as st
import time
import os
import sys

# Astuce pour importer des modules depuis le répertoire parent (agent/)
# car Streamlit exécute ce script depuis le sous-dossier /pages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import generate_trace_animation_frames

# Configuration de la page
st.set_page_config(layout="wide", page_title="Agent Run Animator")
st.title("🎬 Visualisation de l'agent")
st.markdown("Visualisez pas à pas le chemin de décision de la dernière conversation avec l'agent.")

# Vérifier si une conversation a déjà eu lieu
if 'last_run_id' not in st.session_state:
    st.info("👋 Pour commencer, veuillez avoir une conversation avec l'agent sur la page principale 'app'.")
    st.stop()

# --- Interface de contrôle de l'animation ---
st.subheader("Contrôles")
col1, col2 = st.columns([3, 1])

with col1:
    speed = st.slider(
        "Vitesse de l'animation (secondes par étape)", 
        min_value=0.25, 
        max_value=2.0, 
        value=1.0,  # Valeur par défaut
        step=0.25
    )

with col2:
    # Ce bouton va lancer la récupération des données et l'animation
    st.write("") # Un peu d'espace pour aligner le bouton
    st.write("")
    animate_button = st.button("Lancer l'animation", use_container_width=True, type="primary")


# --- Logique de l'animation ---
if animate_button:
    # Récupérer l'ID de la dernière conversation
    last_run_id = st.session_state.last_run_id
    
    with st.spinner("Récupération de la trace depuis LangSmith et génération des frames..."):
        frames = generate_trace_animation_frames(last_run_id)

    if not frames:
        st.error("Impossible de récupérer la trace ou de générer l'animation. Vérifiez les logs du terminal.")
    else:
        st.success(f"Trace trouvée ! Lancement de l'animation pour {len(frames)} étapes.")
        
        # Créer des conteneurs vides qui seront mis à jour
        description_placeholder = st.empty()
        image_placeholder = st.empty()
        
        # Boucle d'animation
        for description, image_bytes in frames:
            with description_placeholder.container():
                st.markdown(f"### {description}")
            
            image_placeholder.image(image_bytes, use_container_width=True)
            
            # Attendre en fonction de la vitesse choisie
            time.sleep(speed)
            
        st.success("🎉 Animation terminée !")