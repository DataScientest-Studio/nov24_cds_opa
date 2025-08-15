# agent/pages/2_📄 Rapport de recherche.py

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Rapport de recherche")

st.title("📄 Rapport de recherche")

st.markdown("""
    <p>Ce rapport est un document en finalisation. </p>
    <p>Vous pouvez interagir avec le document directement ci-dessous :</p>
""", unsafe_allow_html=True)



pdf_url = "https://drive.usercontent.google.com/download?id=1iuRySCgm_xMnWsFptM0Ip_g1hnSVOJdV&export=download&authuser=0&confirm=t"
pdf_embed_code = f"""
<iframe src="https://docs.google.com/viewer?url={pdf_url}&embedded=true" 
        style="border: 0; width: 100%; height: 1200px;" 
        width="100%" 
        height="1200px" 
        frameborder="0" 
        allowfullscreen="true" 
        mozallowfullscreen="true" 
        webkitallowfullscreen="true">
</iframe>
"""

# Affiche le code HTML de l'iframe
components.html(pdf_embed_code, height=1250, scrolling=True) 

st.markdown(f"""
<a href="{pdf_url}" target="_self">
    <button style="background-color:#34FFBC;color:white;padding:10px 24px;border:none;border-radius:4px;">
        📄 Télécharger le rapport
    </button>
</a>
""", unsafe_allow_html=True)
st.divider()

st.info("  Ce rapport est un instantané du document. Pour la dernière version, consultez le Google Docs original ou téléchargez un nouveau PDF.")

